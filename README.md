# HA++ (HUTTA Universal)

HA++ is a small compiled programming language for fast 3D graphics (Gaussian
splats, AR) and AI. You write your code once. From Windows, macOS or Linux,
HA++ builds it for Android, iPhone, PC and web browsers. It does not need
Xcode or the Android NDK.

- **CPU code** becomes native ARM64 / x86-64 machine code through LLVM, the
  same backend as Swift, Rust and C. It runs as fast as C.
- **GPU code** (`kernel` functions) becomes SPIR-V for Vulkan (Android,
  Windows, Linux) and Metal Shading Language for iPhone and Mac. Both are
  generated from the same source.
- **Bridges are generated for you:** a C header, a Swift package
  (XCFramework), Kotlin/Java classes with the JNI glue already inside the
  `.so`, and Python bindings.

![Gaussian splats rendered by the HA++ pipeline](docs/images/galaxy.png)

*30,000 Gaussian splats, rendered by `gaussian/splat.ha`: projection,
GPU radix sort and tile rasterizer, all written in HA++. This image was made
on a Linux server with the software Vulkan driver (lavapipe). An Android
phone runs the same `.so`.*

```
                                  ┌─► ARM64 .so ─────────► Android (jniLibs/arm64-v8a + Kotlin/Java)
 your .ha file ─► happ (Python) ─►├─► ARM64 .a ──────────► iPhone  (Swift package / XCFramework)
                  type checker    ├─► x86-64 .dll/.so ───► Windows / Linux / Android emulator
                  LLVM IR         ├─► WebAssembly .wasm ─► browsers, Safari on iPhone (+ JS loader)
                                  ├─► SPIR-V (embedded) ─► Vulkan GPUs: Android, Windows, Linux
                                  └─► Metal source ──────► iPhone / Mac GPUs
```

## Quick start

You need **Python 3.9+** and **LLVM** (clang + lld), which are free on every OS:

| Your computer | Install |
|---|---|
| Windows | `winget install LLVM.LLVM` and `winget install Python.Python.3.12` |
| macOS | `brew install llvm lld python` |
| Linux | `sudo apt install clang lld llvm python3 python3-numpy` |

For GPU kernels, also install `glslangValidator` (Vulkan SDK, or `apt install glslang-tools`).

```sh
./happ.sh doctor                        # which tools were found   (Windows: happ doctor)
./happ.sh run examples/hello.ha         # compile + run on this PC
./happ.sh build gaussian/splat.ha -t phones    # Android + iPhone libraries
./happ.sh build my.ha -t all            # all 12 targets
./happ.sh emit my.ha asm -t android-arm64            # see the ARM64 assembly
python3 gaussian/render.py --out splats.png    # render splats on this PC's GPU
python3 gaussian/build_ios.py                  # iPhone installer (.ipa), no Mac or Xcode needed
```

## A taste of the language

```rust
struct Splat { pos: vec3, color: vec4 }

// CPU function, callable from Swift, Kotlin, C and Python
export fn fade(s: *Splat, n: i64, k: f32) {
    for i in 0..n { s[i].color.a *= k; }          // vectorized for NEON / AVX2
}

// GPU function: runs on Vulkan and Metal; also on the CPU as fade_gpu_cpu(...)
@workgroup(64)
kernel fade_gpu(s: *Splat, n: u32, k: f32) {
    let i = global_id.x;
    if i < n { s[i].color.a *= k; }
}
```

- **Types:** `i8..i64`, `u8..u64`, `f16`, `f32`, `f64`, `bool`, `vec2/3/4`,
  `ivec`, `uvec`, `hvec` (f16), `mat2/3/4`, structs, fixed arrays `[N]T` and
  pointers `*T`.
- **No hidden conversions:** you convert with `as`. Literals adapt to their
  context.
- **Memory layout = C layout everywhere:** CPU, Vulkan and Metal read the
  same bytes. For example, `vec3` is 12 bytes.
- **Errors:** the compiler shows the source line, a caret and a hint.
- **Reference:** the full language is in [docs/LANGUAGE.md](docs/LANGUAGE.md).

## What is in this repository

| Path | What |
|---|---|
| `happ/` | the compiler: lexer, parser, type checker, LLVM backend, GPU backend (GLSL/Metal), bridges, build driver |
| `happ/std/math.ha` | standard library written in HA++: exp/log/sin/cos/tanh/pow…, sigmoid/gelu, quaternions, color packing |
| `happ/lib/mem.ha` | memory allocators in HA++ (`import "mem.ha";`): Arena, Pool and TLSF, an O(1) general-purpose allocator from real-time systems ([docs](docs/LANGUAGE.md#memory-allocators-happlibmemha)) |
| `runtime/gpu/` | Vulkan GPU runtime in C. It compiles with plain clang (no SDK) and is linked into Android/Windows/Linux libraries |
| `gaussian/` | Gaussian splat renderer for AR: the engine in HA++, hosts in Python (PC), Java (Android) and Swift, and a complete iPhone app built into `Splats.ipa` without a Mac ([gaussian/README.md](gaussian/README.md)) |
| `examples/ai/nn.ha` | matmul, softmax, layer norm, GELU: a transformer MLP block on GPU and CPU |
| `tests/` | test suite (see below) |
| `bench/` | speed comparisons: `bench.py` (HA++ vs C and NumPy), `alloc_bench.py` (allocators vs glibc and a C TLSF) |
| `docs/` | [language](docs/LANGUAGE.md), [platforms](docs/PLATFORMS.md), [GPU](docs/GPU.md), [splats](docs/SPLATS.md), [memory](docs/MEMORY.md) |

## Speed

HA++ gives LLVM the same information C does: plain loops, no garbage
collector, no runtime checks. Float math uses fast-math by default, so
reductions vectorize and multiply-adds fuse (turn this off with `@strict`).
The standard math library is branch-free HA++ code that LLVM inlines and
vectorizes.

Results of `python3 bench/bench.py` on the build server (x86-64, best of 7 runs):

| task | HA++ | same loop in C (clang -O3 -ffast-math) | NumPy |
|---|---|---|---|
| sigmoid of 10M floats | **15.1 ms** | 21.8 ms | 21.9 ms |
| saxpy, 10M floats | 2.7 ms | 2.7 ms | 7.7 ms |
| dot product, 10M floats | 2.7 ms | 2.7 ms | 2.7 ms |
| matmul 384×384 (simple loop) | 4.0 ms | 4.0 ms | **0.2 ms** |

- **Where HA++ matches or beats C:** it matches C, and beats it when a loop
  calls math functions.
- **Where NumPy wins:** NumPy's matrix multiply calls OpenBLAS, which is
  hand-tuned assembly on several threads. A simple loop in any language loses
  to it. For big matrices, use the GPU `matmul` kernel or call a BLAS library.
- **Phone GPUs:** the GPU numbers depend on the phone. They have not been
  measured yet, because no phone was available.

**Allocators** (`python3 bench/alloc_bench.py`, ns per operation, same compiler
and flags for all; details and latency percentiles in [docs/MEMORY.md](docs/MEMORY.md)):

| workload | glibc malloc | C TLSF (Conte) | HA++ TLSF | HA++ Arena / Pool |
|---|---|---|---|---|
| mixed sizes, random order | 95.7 | 62.2 | **59.9** | |
| per-frame buffers (64 per frame) | 156.9 | 56.4 | 42.5 | **11.2** (Arena) |
| 32-byte objects | 17.8 | 16.9 | 19.3 | **13.4** (Pool) |

HA++'s TLSF matches the C one on average speed and has a shorter `malloc`
tail. The C one has a shorter tail on large `realloc`s that move, because
glibc's `memcpy` uses `rep movsb`.

## What was tested (and what was not)

All of this runs in `python3 -m unittest discover -s tests` (32 tests):

| Area | Status |
|---|---|
| Compiler errors, literals, name rules | ✅ tested |
| Allocators (Arena, Pool, TLSF): randomized runs against a model, invariants checked after every operation; 22 of 22 planted bugs caught (`tests/mutate_alloc.py`) | ✅ tested |
| **Differential fuzzing:** random programs run on x86-64, ARM64, Vulkan and Metal emulation vs an exact model (`tests/fuzz.py`; 120 programs, about 540,000 results, 0 mismatches) | ✅ tested |
| CPU code on x86-64: language features, math accuracy (exp/log ≤ 2 ulp) | ✅ tested |
| **ARM64 code**, the phones' CPU: 166 results under `qemu-aarch64` match x86-64 exactly | ✅ tested |
| All 12 targets link without NDK/Xcode; Android `.so` is 16 KB aligned; generated C header compiles | ✅ tested |
| **WebAssembly** (`web-wasm32`, what browsers and Safari on iPhone run): same results as x86-64 in Node, in the fuzzer too | ✅ tested |
| Vulkan kernels (SPIR-V validated) run on a real Vulkan driver (lavapipe) and match the CPU | ✅ tested |
| GPU edge cases: matrix writes in buffers, atomics run once, integer division by 0 and -1, f16 math, nested arrays | ✅ tested |
| Vulkan runtime: hundreds of dispatches per batch, bad calls rejected and recovered from | ✅ tested |
| Splat pipeline: GPU sort order identical to NumPy, image within 1–2/255 | ✅ tested |
| AI block (matmul, GELU, layer norm, softmax) on GPU and CPU vs NumPy | ✅ tested |
| Android path via JNI: Kotlin/Java class + JNI glue + Vulkan runtime, run on a desktop JVM | ✅ tested |
| Java `SplatRenderer` (Android host) renders the same bytes as the Python host | ✅ tested |
| Metal kernels: compiled and run through a C++ stand-in for `metal_stdlib`, match the CPU | ⚠️ emulated, not Apple's compiler |
| Swift wrapper and `SplatRenderer.swift` | ⚠️ generated/written, not compiled (no Swift toolchain here) |
| iPhone app `Splats.ipa`: builds without Apple files; Mach-O checked (arm64, iOS 15, entry point, every import bound to its iOS library) | ⚠️ built and checked, not run on an iPhone |
| Running on a real Android phone / iPhone | ❌ not yet (no device available) |

## Honest limits (version 0.1)

- **iPhone without a Mac:** a C + HA++ app (like `gaussian/ios`) is built into
  a finished `.ipa` here, with no Apple files. You install it with Sideloadly or
  AltStore and your Apple ID. A Swift app needs
  [xtool](https://github.com/xtool-org/xtool) on Linux/Windows, or Xcode. xtool
  needs Apple's iOS SDK from the Xcode download. See
  [docs/PLATFORMS.md](docs/PLATFORMS.md).
- **GPU kernels are compute-only:** there are no vertex/fragment shaders. The
  splat renderer draws with a compute rasterizer, as the original 3DGS does.
- **No generics and no strings** beyond `print` in `happ run`. There is no
  built-in heap: memory comes from the host app (buffers and arrays), and
  `happ/lib/mem.ha` divides it up (Arena, Pool, TLSF).
- **`f64` limits:** `exp`, `sin` and the other transcendental functions exist
  only for `f16`/`f32`. `pow` is fast but approximate (about 3e-6 relative).
- **Out-of-range array and pointer access is not checked,** the same as in
  C. Integer division, on the other hand, is defined for every input, including
  `x / 0`.
