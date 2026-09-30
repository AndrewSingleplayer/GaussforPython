# Using HA++ libraries on each platform

Everything below starts from the same command. It runs on any computer
(Windows, macOS or Linux):

```sh
happ build my.ha -t phones        # android-arm64, android-x64, ios-arm64, ios-sim-arm64, ios-sim-x64
happ build my.ha -t all           # + windows-x64/arm64, macos-arm64/x64, linux-x64/arm64, web-wasm32
happ build my.ha -t web           # WebAssembly for browsers, Safari on iPhone included
```

The output goes to `build/`, next to the `.ha` file:

```
build/
  android/jniLibs/arm64-v8a/libmy.so     phones (ARM64)
  android/jniLibs/x86_64/libmy.so        Android emulator on a PC
  android/kotlin/com/happ/my/My.kt       Kotlin class (use this or the Java one)
  android/java/com/happ/my/My.java
  apple/My/                              Swift package: MyCore.xcframework + Swift wrapper
  include/my.h, module.modulemap         C / C++ / Objective-C / Swift header
  windows-x64/my.dll, my.lib             Windows
  linux-x64/libmy.so                     Linux
  macos-arm64/libmy.a ...                Mac (static; also inside the xcframework)
  gpu/*.comp, *.spv, my.metal            generated GPU code (also embedded in the libraries)
  python/my.py                           Python bindings
  web/my.wasm, web/my.mjs                WebAssembly + a JavaScript loader (browsers, Node)
```

These options change names:
- `--name`: the library name.
- `--package com.me.app` and `--class Native`: the Kotlin/Java names.

## Tools

| Tool | Needed for | Get it |
|---|---|---|
| Python 3.9+ | the compiler itself | python.org / winget / brew / apt |
| LLVM: `clang`, `ld.lld`, `lld-link`, `llvm-ar` | every target | Windows `winget install LLVM.LLVM` · macOS `brew install llvm lld` · Linux `apt install clang lld llvm` |
| `glslangValidator` | GPU kernels → SPIR-V | Vulkan SDK (lunarg.com) · `apt install glslang-tools` · `brew install glslang` |
| `wasm-ld` | the `web-wasm32` target | part of LLVM's lld (installed with the LLVM line above) |

`happ build` needs nothing else. `happ run` links a normal program against
your system's C library. On Windows that means installing Visual Studio Build
Tools (or MSVC), so clang can find the C runtime. macOS and Linux already
have it.

Run `happ doctor` to see what was found. You can point `HAPP_LLVM` at an LLVM
folder. The Android NDK's `toolchains/llvm/prebuilt/*/bin` also works: set
`ANDROID_NDK_HOME`.

---

## Android (built on Windows, Mac or Linux)

1. `happ build my.ha -t android`
2. In Android Studio, copy `build/android/jniLibs/` to `app/src/main/jniLibs/`.
   Copy `build/android/kotlin/...` (or the `java/...` one) to
   `app/src/main/java/...`, keeping the package folders.
3. Call it from Kotlin:

   ```kotlin
   val data = FloatArray(1000) { it * 0.01f }
   My.scale(data, data.size.toLong(), 2f)        // export fn scale(a: *f32, n: i64, k: f32)
   ```

**Type mapping:**

| HA++ | Kotlin/Java |
|---|---|
| `i32`/`u32` | `Int` |
| `i64`/`u64` | `Long` |
| `f32` | `Float` |
| `bool` | `Boolean` |
| pointer to a number type | primitive array (`FloatArray`, `IntArray`, …) |
| pointer to a struct/vector | direct `ByteBuffer` (`ByteBuffer.allocateDirect(n).order(ByteOrder.nativeOrder())`) |

Struct sizes and field offsets are constants in the class, e.g.
`SIZEOF_SPLAT` and `OFFSET_SPLAT_POS`. Kernel workgroup sizes are
`WORKGROUP_<KERNEL>_X/Y/Z`.

**Build settings:**
- **minSdk:** 24 for CPU code. Use **29** for GPU kernels, because every
  64-bit Android 10+ device has Vulkan 1.1.
- **16 KB pages:** the libraries are linked with 16 KB page alignment, which
  Google Play requires for Android 15+ targets. There's nothing to set.
- **No NDK, no CMake:** the `.so` files are complete. The JNI glue is inside
  them, and so is the Vulkan runtime when the file has kernels.

**GPU from Kotlin/Java:**

```kotlin
val gpu = My.gpuCreate()                         // 0 = no Vulkan
val buf = My.bufferCreate(gpu, 4L * n)
My.bufferData(buf).order(ByteOrder.nativeOrder()).asFloatBuffer().put(values)
val k = My.scaleKernel(gpu)                      // one per kernel
My.scaleRun(gpu, k, buf, n, 2f, (n + 63) / 64, 1, 1)   // buffers, scalars, workgroup counts
// several kernels in one GPU submit:
My.batchBegin(gpu); My.aRecord(...); My.bRecord(...); My.batchSubmit(gpu)
```

A complete example is `gaussian/android/SplatRenderer.java`, which
renders Gaussian splats. It is tested on a desktop JVM.

**ARCore:**
1. Get the view matrix with `frame.camera.getViewMatrix(m, 0)`.
2. Negate rows 1 and 2. ARCore uses y-up/z-back; `splat.ha` expects
   y-down/z-forward.
3. Get fx/fy/cx/cy from `camera.imageIntrinsics`, scaled to your render size.

**Emulator:** the `x86_64` library runs in the Android emulator on a PC.
Emulator Vulkan support varies, so check that `gpuCreate() != 0`.

## Android from a Mac

Same steps as above. HA++ on macOS needs Homebrew LLVM (`brew install llvm lld`),
because Apple's clang has no `ld.lld`.

---

## iPhone, with a Mac (the normal way)

1. `happ build my.ha -t ios` (add `macos-arm64,macos-x64` for Mac apps).
2. In Xcode, go to **File ▸ Add Package Dependencies ▸ Add Local…** and pick
   `build/apple/My`.
3. Use it from Swift:

   ```swift
   import My
   var data = [Float](repeating: 1, count: 1000)
   scale(&data, Int64(data.count), 2)            // C functions are available directly
   let gpu = try MyGPU()                         // Metal: kernels compiled on the device
   ```

## iPhone, without a Mac and without any Apple files (C apps)

An app written in C (plus HA++) can be built into an `.ipa` on Windows or Linux
with only LLVM. You don't need Xcode, the iOS SDK or an `Xcode.xip`. The app
talks to UIKit and Metal through the Objective-C runtime. The linker learns
which iOS library exports each function from small text files (`.tbd`) that the
build script writes itself.

`gaussian/build_ios.py` does this for the Gaussian splat viewer:

```
python3 gaussian/build_ios.py          # -> gaussian/build/ios/Splats.ipa
```

Install the `.ipa` with **Sideloadly** (Windows/Mac) or **AltStore**. These
tools sign it with your Apple ID. [gaussian/README.md](../gaussian/README.md)
has the steps. Use this route for small, self-contained apps. For Swift or
ARKit apps, use the xtool route below.

## iPhone, without a Mac (Swift apps, from Windows or Linux)

HA++ does everything up to the finished Swift package without Apple tools:
- It compiles for ARM64 iOS.
- It creates the static library with `llvm-ar`.
- It builds `MyCore.xcframework` (Info.plist and folders).
- It embeds the Metal source.

For the app itself, use [xtool](https://github.com/xtool-org/xtool), an
open-source Xcode replacement for Linux and Windows (through WSL). It builds,
signs and installs SwiftPM apps over USB.

1. Install xtool by following its instructions. It asks for an `Xcode.xip`,
   which you download from Apple with an Apple ID, and it extracts the iOS SDK
   from it.
2. In your xtool app's `Package.swift`, add
   `.package(path: "path/to/build/apple/My")` and the product `"My"` to your
   target.
3. Run `xtool dev` to build, sign and install on the iPhone connected by USB.

What to know:
- **License:** Apple's Xcode/SDK license says to use it on Apple-branded
  computers, so extracting the SDK on Windows/Linux is a legal gray area. The
  decision is yours.
- **Apple ID:** a free Apple ID lets you install on your own iPhone, and the
  app expires after 7 days. A paid Apple Developer account (about $99/year)
  gives 1-year signing and the App Store.
- **Real device for AR:** ARKit doesn't run in the simulator. The simulator
  slices (arm64 and x86_64) are included for non-AR testing on a Mac.
- **Metal compiler:** none is needed at build time. The generated `MyGPU`
  class compiles the embedded Metal source on the device with
  `makeLibrary(source:)`. The source ships inside your app, so it isn't
  downloaded code.
- **Not tested:** the Swift side (xcframework plus wrapper) hasn't been
  compiled on a Mac or with xtool yet, because there's no Apple toolchain on
  the build machine. The C/ARM64 code inside it is the same code that is
  tested under ARM64 emulation.

**ARKit:**
1. Get the view matrix with `frame.camera.viewMatrix(for: orientation)`.
2. Negate rows 1 and 2 (y-up → y-down, z-back → z-forward).
3. Get the intrinsics from `frame.camera.intrinsics`.

See `gaussian/apple/SplatRenderer.swift`.

---

## Windows

- `happ build my.ha -t windows-x64` produces `my.dll`, `my.lib` and
  `include/my.h`.
- The DLL doesn't depend on the C runtime. With kernels, it imports only 5
  `kernel32` functions and loads `vulkan-1.dll` at run time. Vulkan comes with
  GPU drivers.
- **CPU level:** x86-64 builds default to `x86-64-v3` (AVX2 + FMA, 2013+
  CPUs). For older PCs, use `--cpu x86-64-v2`.
- **Java on Windows:** it can use the same Java class, because the JNI glue is
  inside the DLL too.

## Linux

`happ build my.ha -t linux-x64` (or `linux-arm64`) produces `libmy.so` and
Python bindings.

## Web browsers and Safari on iPhone (WebAssembly)

`happ build my.ha -t web` makes `build/web/my.wasm` and `build/web/my.mjs`.
The module uses WebAssembly SIMD (128-bit vectors), bulk memory and
saturating float-to-int conversion. Safari on iOS 16.4 or newer, Chrome,
Firefox and Node 18+ support all three.

```js
import { load, SIZEOF } from "./my.mjs";
const lib = await load(new URL("./my.wasm", import.meta.url));
const n = 1000;
const p = lib.alloc(4 * n);                 // bytes inside the module's memory
lib.f32(p, n).set(myFloat32Array);          // copy data in
lib.exports.process(p, BigInt(n));          // an `export fn process(data: *f32, n: i64)`
const result = lib.f32(p, n);               // read results (take a new view after memory grows)
```

- **Types:** pointers are byte offsets (JS numbers). `i64`/`u64`
  parameters and results are JS `BigInt`. All other numbers are JS numbers.
  Prefer `u32`/`i32` counts in functions made for the web.
- **Layout:** the same as on every other target. WebAssembly addresses are
  4 bytes, but HA++ stores pointers inside structs and arrays as 8 bytes
  there too. So `size_of`, the C header and `SIZEOF`/`OFFSETOF` in the JS
  module all agree.
- **Kernels:** a browser gets the CPU version of each kernel
  (`<kernel>_cpu`). The GPU versions need WebGPU, which HA++ doesn't
  generate yet.
- **Math:** the same HA++ math library as elsewhere. WebAssembly has no
  fused multiply-add, so `exp`, `tanh` etc. can differ from the ARM64/x86
  result by an ulp. They stay within the same stated accuracy. The
  differential fuzzer (`tests/fuzz.py`) runs every random program as
  WebAssembly too.
- **No threads:** shared memory in browsers needs cross-origin isolation
  headers, so run long work in a Web Worker instead.

## Python (tools and AI experiments on a PC)

```python
import sys; sys.path.insert(0, "build/python")
import my, numpy as np
a = np.ones(1000, np.float32)
my.scale(a, len(a), 2.0)          # numpy arrays are passed as pointers
```

The GPU runtime functions (`ha_gpu_create`, `ha_dispatch`, …) are also
exported by the library. See `gaussian/render.py` for a full GPU host
written in Python.

## CPU tuning

| Target | Default | Faster option (newer devices only) |
|---|---|---|
| android-arm64 | `armv8-a`: every 64-bit Android phone | `--cpu armv8.2-a+fp16+dotprod`: native f16 math, int8 dot products |
| iOS | Apple's baseline for iOS 15 | `--cpu apple-a14` |
| x86-64 | `x86-64-v3` | `--cpu x86-64-v4` (AVX-512) |
