"""Build targets and the LLVM toolchain.

Every target is built with one LLVM install (clang + lld), from any host:
Windows, macOS or Linux. No Android NDK, no Xcode.
"""

import glob
import os
import platform
import shutil

# cpu: flags for clang. f16c: whether the default CPU converts f16 in hardware.
TARGETS = {
    "android-arm64": dict(triple="aarch64-linux-android24", arch="arm64", os="android", kind="shared",
                          cpu=["-march=armv8-a"], abi="arm64-v8a"),
    "android-x64": dict(triple="x86_64-linux-android24", arch="x86_64", os="android", kind="shared",
                        cpu=["-march=x86-64-v2"], abi="x86_64"),
    "ios-arm64": dict(triple="arm64-apple-ios15.0", arch="arm64", os="ios", kind="static", cpu=[],
                      slice="ios-arm64"),
    "ios-sim-arm64": dict(triple="arm64-apple-ios15.0-simulator", arch="arm64", os="ios", kind="static",
                          cpu=[], slice="ios-arm64_x86_64-simulator", sim=True),
    "ios-sim-x64": dict(triple="x86_64-apple-ios15.0-simulator", arch="x86_64", os="ios", kind="static",
                        cpu=["-march=x86-64-v2"], slice="ios-arm64_x86_64-simulator", sim=True),
    "macos-arm64": dict(triple="arm64-apple-macos11.0", arch="arm64", os="macos", kind="static", cpu=[],
                        slice="macos-arm64_x86_64"),
    "macos-x64": dict(triple="x86_64-apple-macos10.15", arch="x86_64", os="macos", kind="static",
                      cpu=["-march=x86-64-v2"], slice="macos-arm64_x86_64"),
    "windows-x64": dict(triple="x86_64-pc-windows-msvc", arch="x86_64", os="windows", kind="dll",
                        cpu=["-march=x86-64-v3"]),
    "windows-arm64": dict(triple="aarch64-pc-windows-msvc", arch="arm64", os="windows", kind="dll",
                          cpu=["-march=armv8.2-a"]),
    "linux-x64": dict(triple="x86_64-linux-gnu", arch="x86_64", os="linux", kind="shared",
                      cpu=["-march=x86-64-v3"]),
    "linux-arm64": dict(triple="aarch64-linux-gnu", arch="arm64", os="linux", kind="shared",
                        cpu=["-march=armv8-a"]),
    # browsers (Safari on iPhone included): SIMD128, bulk memory, saturating float->int in hardware
    "web-wasm32": dict(triple="wasm32-unknown-unknown", arch="wasm32", os="web", kind="wasm",
                       cpu=["-msimd128", "-mbulk-memory", "-mnontrapping-fptoint", "-msign-ext"]),
}

GROUPS = {
    "android": ["android-arm64", "android-x64"],
    "ios": ["ios-arm64", "ios-sim-arm64", "ios-sim-x64"],
    "phones": ["android-arm64", "android-x64", "ios-arm64", "ios-sim-arm64", "ios-sim-x64"],
    "desktop": ["windows-x64", "macos-arm64", "macos-x64", "linux-x64"],
    "web": ["web-wasm32"],
    "all": list(TARGETS),
}


def expand_targets(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        names = GROUPS.get(part, [part])
        for n in names:
            if n not in TARGETS:
                raise ValueError(f"unknown target '{n}'. Targets: {', '.join(TARGETS)}; "
                                 f"groups: {', '.join(GROUPS)}")
            if n not in out:
                out.append(n)
    return out


def target_info(name, cpu=None):
    t = dict(TARGETS[name])
    t["name"] = name
    if cpu:
        flag = "-mcpu=" if t["os"] in ("ios", "macos") and t["arch"] == "arm64" else "-march="
        t["cpu"] = [flag + cpu]
    flags = " ".join(t["cpu"])
    t["f16_helpers"] = t["arch"] == "wasm32" or t["arch"] == "x86_64" and not any(x in flags for x in ("x86-64-v3", "x86-64-v4",
                                                                               "native"))
    return t


def host_target():
    sysname = platform.system().lower()
    mach = platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
    osname = {"darwin": "macos", "windows": "windows"}.get(sysname, "linux")
    return f"{osname}-{arch}"


class Toolchain:
    """Finds clang, lld, llvm-ar, llvm-lipo and glslangValidator on this computer."""

    def __init__(self):
        self.dirs = []
        env = os.environ.get("HAPP_LLVM")
        if env:
            self.dirs.append(env if os.path.basename(env) == "bin" else os.path.join(env, "bin"))
        clang = shutil.which("clang")
        if clang:
            self.dirs.append(os.path.dirname(os.path.realpath(clang)))
            self.dirs.append(os.path.dirname(clang))
        self.dirs += sorted(glob.glob("/usr/lib/llvm-*/bin"), reverse=True)
        self.dirs += ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin",
                      "/opt/homebrew/opt/lld/bin", "/usr/local/opt/lld/bin",
                      r"C:\Program Files\LLVM\bin"]
        ndk = os.environ.get("ANDROID_NDK_HOME") or os.environ.get("ANDROID_NDK_ROOT")
        if ndk:
            self.dirs += glob.glob(os.path.join(ndk, "toolchains", "llvm", "prebuilt", "*", "bin"))
        vk = os.environ.get("VULKAN_SDK")
        if vk:
            self.dirs.append(os.path.join(vk, "bin"))
        self.cache = {}

    def find(self, name):
        if name in self.cache:
            return self.cache[name]
        exe = ".exe" if os.name == "nt" else ""
        found = None
        for d in self.dirs:
            p = os.path.join(d, name + exe)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                found = p
                break
        if found is None:
            found = shutil.which(name)
        self.cache[name] = found
        return found

    def require(self, name, why):
        p = self.find(name)
        if p is None:
            raise FileNotFoundError(
                f"'{name}' was not found ({why}).\n"
                f"  Install LLVM (free): Windows 'winget install LLVM.LLVM', macOS 'brew install llvm lld',\n"
                f"  Linux 'apt install clang lld llvm'. Or point HAPP_LLVM at an LLVM folder.")
        return p

    def report(self):
        rows = []
        for tool, why in [("clang", "compiles for every target"),
                          ("ld.lld", "links Android/Linux .so"),
                          ("lld-link", "links Windows .dll"),
                          ("llvm-ar", "makes iOS/macOS .a libraries"),
                          ("wasm-ld", "links WebAssembly modules (web browsers)"),
                          ("llvm-lipo", "merges iOS simulator slices"),
                          ("llvm-nm", "checks libraries for missing symbols"),
                          ("llvm-dlltool", "Windows GPU runtime (kernel32 import library)"),
                          ("glslangValidator", "compiles GPU kernels to SPIR-V (Vulkan)"),
                          ("spirv-val", "validates SPIR-V (optional)")]:
            rows.append((tool, self.find(tool), why))
        return rows
