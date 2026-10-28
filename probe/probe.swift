// What a macOS runner's GPU can do: device, families, bandwidth, compute,
// simdgroup matrices, bf16, float atomics, dispatch cost, pool-style row
// gathers, and a file mmap'd straight into a GPU buffer (cold and warm).
import Foundation
import Metal

func log(_ s: String) {
    print(s)
    fflush(stdout)
}

struct ProbeError: Error, CustomStringConvertible {
    let description: String
}

let all = MTLCopyAllDevices()
log("MTLCopyAllDevices: \(all.count): \(all.map { $0.name })")
guard let dev = MTLCreateSystemDefaultDevice() else {
    log("NO DEFAULT METAL DEVICE")
    exit(2)
}
log("device: \(dev.name)")
if #available(macOS 14.0, *) {
    log("architecture: \(dev.architecture.name)")
}
log("unified memory: \(dev.hasUnifiedMemory), low power: \(dev.isLowPower), headless: \(dev.isHeadless), removable: \(dev.isRemovable)")
log(String(format: "recommendedMaxWorkingSetSize %.2f GB, maxBufferLength %.2f GB", Double(dev.recommendedMaxWorkingSetSize) / 1e9, Double(dev.maxBufferLength) / 1e9))
log("maxThreadgroupMemoryLength \(dev.maxThreadgroupMemoryLength), maxThreadsPerThreadgroup \(dev.maxThreadsPerThreadgroup.width)")
let fams: [(String, MTLGPUFamily)] = [("apple6", .apple6), ("apple7", .apple7), ("apple8", .apple8), ("apple9", .apple9), ("mac2", .mac2), ("common3", .common3), ("metal3", .metal3)]
log("families: " + fams.map { "\($0.0)=\(dev.supportsFamily($0.1))" }.joined(separator: " "))

guard let q = dev.makeCommandQueue() else {
    log("NO COMMAND QUEUE")
    exit(3)
}

func lib(_ src: String, _ ver: MTLLanguageVersion? = nil) throws -> MTLLibrary {
    let o = MTLCompileOptions()
    if let v = ver { o.languageVersion = v }
    return try dev.makeLibrary(source: src, options: o)
}

func pipe(_ l: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
    guard let f = l.makeFunction(name: name) else { throw ProbeError(description: "no function \(name)") }
    let p = try dev.makeComputePipelineState(function: f)
    return p
}

func buf(_ bytes: Int) -> MTLBuffer {
    return dev.makeBuffer(length: max(bytes, 16), options: .storageModeShared)!
}

/// GPU seconds of `reps` dispatches of `threads` threads (or `groups` threadgroups) in one command buffer.
func run(_ p: MTLComputePipelineState, _ bufs: [MTLBuffer], threads: Int = 0, groups: Int = 0, tg: Int, reps: Int = 1) -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for _ in 0..<reps {
        if groups > 0 {
            e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        } else {
            e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        }
    }
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let err = cb.error { log("  command buffer error: \(err)") }
    return cb.gpuEndTime - cb.gpuStartTime
}

func section(_ name: String, _ body: () throws -> Void) {
    log("== \(name)")
    do { try body() } catch { log("  FAILED: \(error)") }
}

let main = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void nop(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0xffffffffu) o[0] = 1; }

kernel void copy4(device const float4* a [[buffer(0)]], device float4* b [[buffer(1)]], uint i [[thread_position_in_grid]]) { b[i] = a[i]; }

kernel void read4(device const float4* a [[buffer(0)]], device float* o [[buffer(1)]], uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
    float4 s = 0;
    for (uint k = 0; k < 16; k++) s += a[i + k * n];
    if (s.x == 1234.5f) o[0] = s.y;
}

kernel void fma_f32(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    float4 x = float4(i & 7) * 1e-3f, y = float4(0.9999f), z = float4(0.5f);
    for (int k = 0; k < 512; k++) { x = fma(x, y, z); z = fma(z, y, x); }
    o[i] = x.x + x.y + x.z + x.w + z.x;
}

kernel void fma_f16(device half* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    half4 x = half4(i & 7) * 1e-3h, y = half4(0.9999h), z = half4(0.5h);
    for (int k = 0; k < 512; k++) { x = fma(x, y, z); z = fma(z, y, x); }
    o[i] = x.x + x.y + x.z + x.w + z.x;
}

// 4 independent 8x8x8 accumulations per simdgroup, 256 times: 1 Mi flops a simdgroup
kernel void mma_f16(device float* o [[buffer(0)]], uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
    simdgroup_half8x8 a = simdgroup_half8x8(0.001h), b = simdgroup_half8x8(0.999h);
    simdgroup_float8x8 c0 = simdgroup_float8x8(0), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < 256; k++) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    device float* d = o + (tg * nsg + sg) * 256;
    simdgroup_store(c0, d, 8); simdgroup_store(c1, d + 64, 8); simdgroup_store(c2, d + 128, 8); simdgroup_store(c3, d + 192, 8);
}

kernel void mma_f32(device float* o [[buffer(0)]], uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
    simdgroup_float8x8 a = simdgroup_float8x8(0.001f), b = simdgroup_float8x8(0.999f);
    simdgroup_float8x8 c0 = simdgroup_float8x8(0), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < 256; k++) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    device float* d = o + (tg * nsg + sg) * 256;
    simdgroup_store(c0, d, 8); simdgroup_store(c1, d + 64, 8); simdgroup_store(c2, d + 128, 8); simdgroup_store(c3, d + 192, 8);
}

kernel void mm8(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]]) {
    simdgroup_float8x8 a, b, c = simdgroup_float8x8(0);
    simdgroup_load(a, A, 8);
    simdgroup_load(b, B, 8);
    simdgroup_multiply_accumulate(c, a, b, c);
    simdgroup_store(c, C, 8);
}

kernel void simd_ops(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float v = float(lane);
    o[i * 3 + 0] = simd_sum(v);
    o[i * 3 + 1] = simd_max(v);
    o[i * 3 + 2] = simd_shuffle_xor(v, 1u);
}

kernel void atomic_add(device atomic_float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(o, 1.0f, memory_order_relaxed);
}

// pool-style reads: a threadgroup of 256 threads sums 32 rows of 256 halves (512 B each)
kernel void gather(device const uint* rows [[buffer(0)]], device const half* pool [[buffer(1)]], device float* o [[buffer(2)]],
                   uint t [[threadgroup_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
    float s = 0;
    for (int j = 0; j < 32; j++) s += float(pool[(size_t)rows[t * 32 + j] * 256 + l]);
    o[t * 256 + l] = s;
}
"""

var L: MTLLibrary? = nil
section("compile main kernels") {
    let t = Date()
    L = try lib(main)
    log(String(format: "  ok in %.0f ms", Date().timeIntervalSince(t) * 1e3))
}
guard let L = L else { exit(4) }

section("memory bandwidth") {
    let n = 64 << 20  // floats per buffer: 256 MB
    let a = buf(n * 4), b = buf(n * 4)
    let pa = a.contents().bindMemory(to: Float.self, capacity: n)
    for i in 0..<n { pa[i] = Float(i & 1023) }
    let cp = try pipe(L, "copy4"), rd = try pipe(L, "read4")
    _ = run(cp, [a, b], threads: n / 4, tg: 256)
    let tc = run(cp, [a, b], threads: n / 4, tg: 256, reps: 5)
    log(String(format: "  copy: %.1f GB/s (read + write)", 5.0 * 2 * Double(n * 4) / tc / 1e9))
    let tr = run(rd, [a, b], threads: n / 64, tg: 256, reps: 5)
    log(String(format: "  read: %.1f GB/s", 5.0 * Double(n * 4) / tr / 1e9))
    let pb = b.contents().bindMemory(to: Float.self, capacity: n)
    var bad = 0
    for i in stride(from: 0, to: n, by: 4099) where pb[i] != pa[i] { bad += 1 }
    log("  copy check: \(bad == 0 ? "ok" : "\(bad) wrong")")
}

section("compute") {
    let threads = 1 << 20
    let o = buf(threads * 4)
    for name in ["fma_f32", "fma_f16"] {
        let p = try pipe(L, name)
        _ = run(p, [o], threads: threads, tg: 256)
        let t = run(p, [o], threads: threads, tg: 256, reps: 4)
        log(String(format: "  %@: %.2f TFLOPS", name, 4.0 * Double(threads) * 8192 / t / 1e12))
    }
    for name in ["mma_f16", "mma_f32"] {
        let p = try pipe(L, name)
        let groups = 1024, tg = 256  // 8 simdgroups a threadgroup
        let ob = buf(groups * 8 * 256 * 4)
        _ = run(p, [ob], groups: groups, tg: tg)
        let t = run(p, [ob], groups: groups, tg: tg, reps: 4)
        log(String(format: "  %@ (simdgroup 8x8): %.2f TFLOPS", name, 4.0 * Double(groups * 8) * 1048576 / t / 1e12))
    }
}

section("simdgroup matrix 8x8 correctness") {
    let a = buf(256), b = buf(256), c = buf(256)
    let pa = a.contents().bindMemory(to: Float.self, capacity: 64), pb = b.contents().bindMemory(to: Float.self, capacity: 64)
    for i in 0..<64 {
        pa[i] = Float((i * 7) % 11) - 5
        pb[i] = Float((i * 5) % 13) - 6
    }
    _ = run(try pipe(L, "mm8"), [a, b, c], threads: 32, tg: 32)
    let pc = c.contents().bindMemory(to: Float.self, capacity: 64)
    var err: Float = 0
    for r in 0..<8 {
        for col in 0..<8 {
            var s: Float = 0
            for k in 0..<8 { s += pa[r * 8 + k] * pb[k * 8 + col] }
            err = max(err, abs(s - pc[r * 8 + col]))
        }
    }
    log("  max error \(err) \(err == 0 ? "(exact)" : "")")
}

section("simd reductions") {
    let o = buf(64 * 3 * 4)
    _ = run(try pipe(L, "simd_ops"), [o], threads: 64, tg: 64)
    let p = o.contents().bindMemory(to: Float.self, capacity: 192)
    log("  simd_sum \(p[0]) (want 496), simd_max \(p[1]) (want 31), shuffle_xor lane0 \(p[2]) (want 1); simd width \(try pipe(L, "simd_ops").threadExecutionWidth)")
}

section("float atomics") {
    let o = buf(4)
    o.contents().bindMemory(to: Float.self, capacity: 1)[0] = 0
    let t = run(try pipe(L, "atomic_add"), [o], threads: 1 << 20, tg: 256)
    log(String(format: "  sum %.0f (want 1048576), %.2f ms", o.contents().bindMemory(to: Float.self, capacity: 1)[0], t * 1e3))
}

section("bfloat (MSL 3.1)") {
    guard #available(macOS 14.0, *) else { throw ProbeError(description: "needs macOS 14") }
    let l = try lib("""
    #include <metal_stdlib>
    using namespace metal;
    kernel void bf(device const bfloat* a [[buffer(0)]], device float* o [[buffer(1)]], uint i [[thread_position_in_grid]]) { o[i] = float(a[i]) * 2.0f; }
    kernel void bfmm(device float* o [[buffer(0)]]) {
        simdgroup_bfloat8x8 a = simdgroup_bfloat8x8(bfloat(1.5f)), b = simdgroup_bfloat8x8(bfloat(2.0f));
        simdgroup_float8x8 c = simdgroup_float8x8(0);
        simdgroup_multiply_accumulate(c, a, b, c);
        simdgroup_store(c, o, 8);
    }
    """, .version3_1)
    let a = buf(64 * 2), o = buf(64 * 4)
    let pa = a.contents().bindMemory(to: UInt16.self, capacity: 64)
    for i in 0..<64 { pa[i] = UInt16(truncatingIfNeeded: Float(i).bitPattern >> 16) }
    _ = run(try pipe(l, "bf"), [a, o], threads: 64, tg: 64)
    let po = o.contents().bindMemory(to: Float.self, capacity: 64)
    log("  bf16 load: o[5] = \(po[5]) (want 10), o[63] = \(po[63]) (want 126)")
    do {
        _ = run(try pipe(l, "bfmm"), [o], threads: 32, tg: 32)
        log("  simdgroup_bfloat8x8: c[0] = \(po[0]) (want 3), c[1] = \(po[1]) (want 0)")
    } catch { log("  simdgroup_bfloat8x8: \(error)") }
}

section("dispatch cost") {
    let p = try pipe(L, "nop"), o = buf(4)
    _ = run(p, [o], threads: 32, tg: 32)
    let t0 = Date()
    let g = run(p, [o], threads: 32, tg: 32, reps: 1000)
    let wall = Date().timeIntervalSince(t0)
    log(String(format: "  1000 dispatches in one command buffer: %.2f us each on the GPU, %.2f us wall", g * 1e3, wall * 1e3))
    let t1 = Date()
    for _ in 0..<200 { _ = run(p, [o], threads: 32, tg: 32) }
    log(String(format: "  one command buffer, commit + wait: %.1f us round trip", Date().timeIntervalSince(t1) / 200 * 1e6))
}

func xorshift(_ x: inout UInt64) -> UInt64 {
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x
}

func gatherTest(_ name: String, _ pool: MTLBuffer, rows: Int, check: (Int, Int) -> Float) throws {
    let groups = 16384
    let idx = buf(groups * 32 * 4), o = buf(groups * 256 * 4)
    let pi = idx.contents().bindMemory(to: UInt32.self, capacity: groups * 32)
    var x: UInt64 = 0x9E3779B97F4A7C15
    for i in 0..<groups * 32 { pi[i] = UInt32(xorshift(&x) % UInt64(rows)) }
    let p = try pipe(L, "gather")
    let t1 = run(p, [idx, pool, o], groups: groups, tg: 256)
    let t2 = run(p, [idx, pool, o], groups: groups, tg: 256)
    let mb = Double(groups * 32 * 512) / 1e6
    log(String(format: "  %@: %.0f MB of random 512 B rows: first %.2f ms (%.1f GB/s), again %.2f ms (%.1f GB/s)", name, mb, t1 * 1e3, mb / t1 / 1e3, t2 * 1e3, mb / t2 / 1e3))
    let po = o.contents().bindMemory(to: Float.self, capacity: groups * 256)
    var bad = 0
    for g in stride(from: 0, to: groups, by: 997) {
        for l in stride(from: 0, to: 256, by: 37) {
            var s: Float = 0
            for j in 0..<32 { s += check(Int(pi[g * 32 + j]), l) }
            if abs(s - po[g * 256 + l]) > 1e-3 { bad += 1 }
        }
    }
    log("  check: \(bad == 0 ? "ok" : "\(bad) wrong")")
}

func val(_ r: Int, _ c: Int) -> Float { Float((r * 31 + c) % 17) * 0.25 }

section("pool gather from a GPU buffer (1 GB)") {
    let rows = 2 << 20
    let pool = buf(rows * 512)
    let p = pool.contents().bindMemory(to: Float16.self, capacity: rows * 256)
    for r in 0..<rows { for c in 0..<256 { p[r * 256 + c] = Float16(val(r, c)) } }
    try gatherTest("shared buffer", pool, rows: rows, check: val)
}

section("pool gather from a file mmap'd into a GPU buffer (bytesNoCopy, 2 GB, cold page cache)") {
    let rows = 4 << 20, bytes = rows * 512
    let path = NSTemporaryDirectory() + "pool_probe.bin"
    let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
    guard fd >= 0 else { throw ProbeError(description: "open \(path)") }
    defer { close(fd); unlink(path) }
    guard ftruncate(fd, off_t(bytes)) == 0 else { throw ProbeError(description: "ftruncate") }
    let tw = Date()
    var w = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)!
    let pw = w.bindMemory(to: Float16.self, capacity: rows * 256)
    for r in 0..<rows { for c in 0..<256 { pw[r * 256 + c] = Float16(val(r, c)) } }
    msync(w, bytes, MS_SYNC)
    munmap(w, bytes)
    log(String(format: "  wrote %.1f GB in %.1f s", Double(bytes) / 1e9, Date().timeIntervalSince(tw)))
    let pr = Process()
    pr.executableURL = URL(fileURLWithPath: "/usr/bin/sudo")
    pr.arguments = ["-n", "/usr/sbin/purge"]
    try pr.run()
    pr.waitUntilExit()
    log("  purge (drop page cache): exit \(pr.terminationStatus)")
    for prot in [PROT_READ, PROT_READ | PROT_WRITE] {
        w = mmap(nil, bytes, prot, MAP_SHARED, fd, 0)!
        let tb = Date()
        guard let b = dev.makeBuffer(bytesNoCopy: w, length: bytes, options: .storageModeShared, deallocator: nil) else {
            log("  makeBuffer(bytesNoCopy:) failed with prot \(prot)")
            munmap(w, bytes)
            continue
        }
        log(String(format: "  makeBuffer(bytesNoCopy:) prot %d: %.1f ms", prot, Date().timeIntervalSince(tb) * 1e3))
        let tg = Date()
        try gatherTest("mmap'd file (cold)", b, rows: rows, check: val)
        log(String(format: "  wall time incl. faults: %.0f ms", Date().timeIntervalSince(tg) * 1e3))
        munmap(w, bytes)
        break
    }
}

log("== done")
