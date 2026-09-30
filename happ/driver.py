"""The `happ` command line: check, run, emit, build, doctor."""

import argparse
import os
import re
import subprocess
import sys
import tempfile

from . import __version__
from . import ast as A
from .checker import check as check_program
from .errors import HappError
from .llvm import LLVMGen
from .parser import parse
from .targets import GROUPS, TARGETS, Toolchain, expand_targets, host_target, target_info

HERE = os.path.dirname(os.path.abspath(__file__))
STD_DIR = os.path.join(HERE, "std")
LIB_DIR = os.path.join(HERE, "lib")          # libraries found by `import "name.ha"` from any file
RUNTIME_DIR = os.path.join(os.path.dirname(HERE), "runtime")


# ------------------------------------------------------------------ loading

def load_program(path):
    modules, seen = [], set()

    def load(p, is_std=False):
        rp = os.path.realpath(p)
        if rp in seen:
            return
        seen.add(rp)
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            raise HappError(f"can't read {p}: {e.strerror}")
        if is_std:
            shown = f"std/{os.path.basename(p)}"
        elif rp.startswith(os.path.realpath(LIB_DIR) + os.sep):
            shown = f"lib/{os.path.relpath(rp, LIB_DIR)}"
        else:
            shown = os.path.relpath(p)
        m = parse(text, shown)
        m.is_std = is_std
        modules.append(m)
        for it in m.items:
            if isinstance(it, A.Import):
                target = os.path.join(os.path.dirname(p), it.path)
                if not os.path.exists(target):
                    target = os.path.join(LIB_DIR, it.path)       # a library that comes with HA++
                if not os.path.exists(target):
                    libs = sorted(n for n in os.listdir(LIB_DIR) if n.endswith(".ha"))
                    raise HappError(f"import not found: {it.path}", it.loc,
                                    f"libraries that come with HA++: {', '.join(libs)}")
                load(target, is_std)

    load(path)
    for name in sorted(os.listdir(STD_DIR)):
        if name.endswith(".ha"):
            load(os.path.join(STD_DIR, name), is_std=True)
    return check_program(modules)


def lib_name_for(path, override=None):
    name = override or os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if name[0].isdigit():
        name = "_" + name
    return name


def camel(name):
    parts = re.split(r"[_\W]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p) or "Happ"


def lib_roots(prog):
    roots = [f.name for f in prog.exports] + [k.name for k in prog.kernels]
    return roots


# ------------------------------------------------------------------ compiling

def show(path):
    """Short path for messages: relative inside the current folder, absolute outside it."""
    rel = os.path.relpath(path)
    return path if rel.startswith("..") else rel


def run_cmd(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise HappError(f"{what} failed:\n  $ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r


def clang_flags(tinfo, opt):
    flags = [f"--target={tinfo['triple']}", opt, "-Wno-override-module"] + list(tinfo["cpu"])
    if tinfo["os"] in ("android", "linux"):
        flags += ["-fPIC", "-ffunction-sections", "-fdata-sections"]
    return flags


def compile_ir(tc, ir, tinfo, obj, opt="-O3"):
    ll = os.path.splitext(obj)[0] + ".ll"
    with open(ll, "w") as f:
        f.write(ir)
    clang = tc.require("clang", "compiling HA++ output")
    run_cmd([clang] + clang_flags(tinfo, opt) + ["-c", ll, "-o", obj], "clang")
    return obj


def undefined_symbols(tc, obj):
    nm = tc.find("llvm-nm")
    if nm is None:
        return []
    r = subprocess.run([nm, "-u", "-j", obj], capture_output=True, text=True)
    return [s.strip() for s in r.stdout.split() if s.strip()]


def defined_symbols(tc, obj):
    nm = tc.find("llvm-nm")
    if nm is None:
        return set()
    r = subprocess.run([nm, "--defined-only", "-j", obj], capture_output=True, text=True)
    return {s.strip() for s in r.stdout.split() if s.strip()}


# C library functions the GPU runtime (runtime/gpu/ha_gpu.c) uses; every OS has them at run time
RUNTIME_LIBC = {"malloc", "calloc", "free", "memcpy", "memset", "dlopen", "dlsym",
                "__stack_chk_fail", "__stack_chk_guard"}
WIN_IMPORTS = ["LoadLibraryA", "GetProcAddress", "GetProcessHeap", "HeapAlloc", "HeapFree"]


def check_undefined(tc, objs, prog, tinfo, runtime=False):
    """Symbols the objects need from outside. Anything unexpected is a compiler bug."""
    allowed = {f.name for f in prog.externs}
    if runtime:
        allowed |= RUNTIME_LIBC | {"__imp_" + n for n in WIN_IMPORTS}
    defined = set()
    und = []
    for o in objs:
        defined |= defined_symbols(tc, o)
        und += undefined_symbols(tc, o)
    darwin = tinfo["os"] in ("ios", "macos")
    missing, external = [], []
    for s in dict.fromkeys(und):
        if s in defined:
            continue
        name = s[1:] if darwin and s.startswith("_") else s
        external.append(s)
        if name not in allowed:
            missing.append(name)
    if missing:
        raise HappError("the compiled code needs symbols HA++ doesn't provide: " + ", ".join(missing) +
                        "\n  (this is a compiler bug unless you declared them with 'extern fn')")
    return [s for s in external if not s.startswith("__imp_")]


def compile_runtime(tc, tinfo, work):
    """Compile the Vulkan GPU runtime for a target (plain clang, no SDK needed)."""
    src = os.path.join(RUNTIME_DIR, "gpu", "ha_gpu.c")
    obj = os.path.join(work, f"ha_gpu-{tinfo['name']}.o")
    clang = tc.require("clang", "compiling the GPU runtime")
    flags = [f"--target={tinfo['triple']}", "-O2", "-ffreestanding", "-std=c11", "-Wno-override-module",
             "-I", os.path.join(RUNTIME_DIR, "gpu")] + list(tinfo["cpu"])
    if tinfo["os"] in ("android", "linux"):
        flags += ["-fPIC", "-ffunction-sections", "-fdata-sections", "-fvisibility=hidden"]
    if tinfo["os"] == "android":
        flags += ["-D__ANDROID__"]
    if tinfo["os"] == "windows":
        flags += ["-fno-stack-protector", "-fms-extensions"]
    run_cmd([clang] + flags + ["-c", src, "-o", obj], "clang (GPU runtime)")
    return obj


def windows_import_lib(tc, tinfo, work):
    dlltool = tc.require("llvm-dlltool", "making kernel32.lib for Windows builds")
    deff = os.path.join(work, "kernel32.def")
    with open(deff, "w") as f:
        f.write("LIBRARY kernel32.dll\nEXPORTS\n" + "\n".join(WIN_IMPORTS) + "\n")
    lib = os.path.join(work, f"kernel32-{tinfo['arch']}.lib")
    machine = "i386:x86-64" if tinfo["arch"] == "x86_64" else "arm64"
    run_cmd([dlltool, "-m", machine, "-d", deff, "-l", lib], "llvm-dlltool")
    return lib


def make_stubs(tc, tinfo, symbols, workdir):
    """Tiny stand-in .so files so the linker records DT_NEEDED entries without a sysroot/NDK."""
    stub_dir = os.path.join(workdir, f"stubs-{tinfo['name']}")
    os.makedirs(stub_dir, exist_ok=True)
    if tinfo["os"] == "android":
        libs = ["libc.so", "libm.so", "libdl.so", "liblog.so"]
        if any(s.startswith("vk") for s in symbols):
            libs.append("libvulkan.so")
    else:
        libs = ["libc.so.6", "libm.so.6", "libdl.so.2"]
    lines = []
    for s in symbols:
        lines += [f"define void @{s}() {{", "  ret void", "}"]
    clang = tc.require("clang", "making link stubs")
    lld = tc.require("ld.lld", "linking Android libraries")
    first = True
    for lib in libs:
        ir = "\n".join(lines if first else ["; empty stub"]) + "\n"
        first = False
        base = os.path.join(stub_dir, lib.split(".so")[0])
        with open(base + ".ll", "w") as f:
            f.write(ir)
        run_cmd([clang, f"--target={tinfo['triple']}", "-c", "-fPIC", "-Wno-override-module",
                 base + ".ll", "-o", base + ".o"], "clang (stub)")
        run_cmd([lld, "-shared", "-soname", lib, base + ".o", "-o", os.path.join(stub_dir, lib)], "ld.lld (stub)")
    return [os.path.join(stub_dir, lib) for lib in libs]


def link(tc, tinfo, objs, out_path, lib_name, undefined, workdir, runtime=False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kind = tinfo["kind"]
    if kind == "shared":
        lld = tc.require("ld.lld", "linking Android/Linux .so libraries")
        cmd = [lld, "-shared", "-soname", os.path.basename(out_path), "--gc-sections", "--build-id=sha1",
               "-z", "noexecstack", "-z", "relro", "-z", "now", "--hash-style=both"]
        if tinfo["os"] == "android":
            # Google Play requires 16 KB page alignment for Android 15+ devices
            cmd += ["-z", "max-page-size=16384", "-z", "common-page-size=16384"]
        cmd += list(objs) + ["-o", out_path]
        if undefined:
            cmd += ["--no-as-needed"] + make_stubs(tc, tinfo, undefined, workdir)
        run_cmd(cmd, "ld.lld")
    elif kind == "static":
        ar = tc.require("llvm-ar", "making .a libraries")
        if os.path.exists(out_path):
            os.remove(out_path)
        run_cmd([ar, "rcs", "--format=darwin", out_path] + list(objs), "llvm-ar")
    elif kind == "dll":
        user = [u for u in undefined if u not in RUNTIME_LIBC]
        if user:
            raise HappError("Windows builds can't use 'extern fn' yet (no import libraries): "
                            + ", ".join(user))
        lld = tc.require("lld-link", "linking Windows .dll libraries")
        machine = "x64" if tinfo["arch"] == "x86_64" else "arm64"
        implib = os.path.splitext(out_path)[0] + ".lib"
        extra = [windows_import_lib(tc, tinfo, workdir)] if runtime else []
        run_cmd([lld, "/dll", "/noentry", "/nodefaultlib", f"/machine:{machine}", f"/out:{out_path}",
                 f"/implib:{implib}"] + list(objs) + extra, "lld-link")
    return out_path


def output_path(out_dir, tinfo, name):
    t = tinfo["name"]
    if tinfo["os"] == "android":
        return os.path.join(out_dir, "android", "jniLibs", tinfo["abi"], f"lib{name}.so")
    if tinfo["kind"] == "static":
        return os.path.join(out_dir, t, f"lib{name}.a")
    if tinfo["kind"] == "dll":
        return os.path.join(out_dir, t, f"{name}.dll")
    return os.path.join(out_dir, t, f"lib{name}.so")


# ------------------------------------------------------------------ commands

def cmd_check(args):
    prog = load_program(args.file)
    user_fns = [f for f in prog.fns.values() if not f.is_std]
    print(f"ok: {len(user_fns)} function(s), {len(prog.kernels)} kernel(s), "
          f"{len([s for s in prog.structs])} struct(s)")


def cmd_emit(args):
    prog = load_program(args.file)
    tname = args.target or host_target()
    tinfo = target_info(tname, args.cpu)
    if args.what == "glsl" or args.what == "metal":
        from .gpu import generate_glsl, generate_metal
        if args.what == "glsl":
            for k, src in generate_glsl(prog).items():
                print(f"// ===== kernel {k} =====\n{src}")
        else:
            print(generate_metal(prog))
        return
    roots = lib_roots(prog) if prog.main is None else ["main"]
    mode = "exe" if prog.main is not None else "lib"
    gen = LLVMGen(prog, tinfo, mode=mode, print_enabled=(mode == "exe"),
                  f16_helpers=tinfo["f16_helpers"], strict_math=args.strict_math)
    ir = gen.generate(roots)
    if args.what == "ir":
        print(ir)
        return
    tc = Toolchain()
    clang = tc.require("clang", "compiling")
    with tempfile.TemporaryDirectory() as tmp:
        ll = os.path.join(tmp, "m.ll")
        with open(ll, "w") as f:
            f.write(ir)
        flags = clang_flags(tinfo, "-O3")
        r = run_cmd([clang] + flags + ["-S", ll, "-o", "-"], "clang")
        text = r.stdout
        if args.what == "asm-clean":
            text = "\n".join(l for l in text.splitlines()
                             if not l.lstrip().startswith((".cfi", ".loc", ".file", "//", "#", ";"))
                             and l.strip())
        print(text)


def cmd_run(args):
    prog = load_program(args.file)
    if prog.main is None:
        raise HappError("this file has no 'fn main()'; use 'happ build' to make a library")
    tc = Toolchain()
    clang = tc.require("clang", "compiling")
    tname = host_target()
    tinfo = target_info(tname)
    gen = LLVMGen(prog, tinfo, mode="exe", print_enabled=True, strict_math=args.strict_math)
    ir = gen.generate(["main"])
    with tempfile.TemporaryDirectory() as tmp:
        ll = os.path.join(tmp, "prog.ll")
        exe = os.path.join(tmp, "prog.exe" if os.name == "nt" else "prog")
        with open(ll, "w") as f:
            f.write(ir)
        march = "-mcpu=native" if tinfo["arch"] == "arm64" else "-march=native"
        run_cmd([clang, "-O3", march, "-Wno-override-module", ll, os.path.join(RUNTIME_DIR, "host.c"),
                 "-o", exe], "clang")
        r = subprocess.run([exe] + args.args)
        sys.exit(r.returncode)


def build(file, targets, out_dir, name=None, package=None, cls=None, cpu=None, strict_math=False,
          quiet=False, bridges=True, gpu_runtime=True):
    """Build `file` for every target. Returns a dict of produced files."""
    prog = load_program(file)
    name = lib_name_for(file, name)
    cls = cls or camel(name)
    package = package or f"com.happ.{name.lower()}"
    tc = Toolchain()
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, ".work")
    os.makedirs(work, exist_ok=True)
    produced = {}
    log = (lambda *a: None) if quiet else (lambda *a: print(*a))
    if not prog.exports and not prog.kernels:
        raise HappError("nothing to build: mark functions with 'export fn' or write a 'kernel'")

    gpu_blobs, metal_src = {}, None
    if prog.kernels:
        from .gpu import build_gpu
        gpu = build_gpu(prog, tc, os.path.join(out_dir, "gpu"), name, log)
        produced.update(gpu["files"])
        gpu_blobs = {f"spirv_{k}": data for k, data in gpu["spirv"].items()}
        metal_src = gpu["metal"]

    warnings = set()
    use_runtime = bool(gpu_blobs) and gpu_runtime
    for tname in targets:
        tinfo = target_info(tname, cpu)
        apple = tinfo["os"] in ("ios", "macos")
        blobs = dict(gpu_blobs)
        if apple:
            blobs = {"metal_source": metal_src.encode()} if metal_src is not None else {}
        rt = use_runtime and not apple
        jni = None if apple else (package, cls)
        gen = LLVMGen(prog, tinfo, mode="lib", jni=jni, lib_name=name, gpu_blobs=blobs,
                      strict_math=strict_math, f16_helpers=tinfo["f16_helpers"], gpu_runtime=rt)
        ir = gen.generate(lib_roots(prog))
        warnings.update(gen.warnings)
        obj = os.path.join(work, f"{name}-{tname}.o")
        compile_ir(tc, ir, tinfo, obj)
        objs = [obj] + ([compile_runtime(tc, tinfo, work)] if rt else [])
        undefined = check_undefined(tc, objs, prog, tinfo, runtime=rt)
        out = output_path(out_dir, tinfo, name)
        link(tc, tinfo, objs, out, name, undefined, work, runtime=rt)
        produced[tname] = out
        log(f"  built {tname:14s} -> {show(out)}" + ("  (+ Vulkan runtime)" if rt else ""))
    for w in sorted(warnings):
        log(f"  note: {w}")
    if bridges:
        from .bridges import write_bridges
        spirv_kernels = [k[len("spirv_"):] for k in gpu_blobs]
        produced.update(write_bridges(prog, out_dir, name, package, cls, targets, produced, tc,
                                      metal_src, log, spirv_kernels=spirv_kernels, gpu_runtime=use_runtime,
                                      src=os.path.basename(file)))
    return produced


def cmd_build(args):
    targets = expand_targets(args.target)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.file)), "build")
    print(f"HA++ {__version__}: building {args.file} for {', '.join(targets)}")
    build(args.file, targets, out, name=args.name, package=args.package, cls=args.cls, cpu=args.cpu,
          strict_math=args.strict_math, gpu_runtime=not args.no_gpu_runtime)
    print(f"done. output in {show(out)}/")


def cmd_doctor(args):
    tc = Toolchain()
    print(f"HA++ {__version__} on {host_target()}\n")
    for tool, path, why in tc.report():
        mark = "ok " if path else "-- "
        print(f"  {mark} {tool:18s} {path or 'not found':45s} {why}")
    print("\nTargets:", ", ".join(TARGETS))
    print("Groups: ", ", ".join(f"{g} ({len(v)})" for g, v in GROUPS.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="happ", description="HA++ (HUTTA Universal) compiler")
    ap.add_argument("--version", action="version", version=f"HA++ {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="type-check a file")
    p.add_argument("file")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="compile and run a program with fn main() on this computer")
    p.add_argument("file")
    p.add_argument("args", nargs="*")
    p.add_argument("--strict-math", action="store_true", help="IEEE-exact float math (no fast-math)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("build", help="build libraries for phones / PCs")
    p.add_argument("file")
    p.add_argument("-t", "--target", default="phones",
                   help="targets or groups, comma separated (default: phones). "
                        f"Groups: {', '.join(GROUPS)}")
    p.add_argument("-o", "--out", help="output folder (default: build/ next to the file)")
    p.add_argument("--name", help="library name (default: file name)")
    p.add_argument("--package", help="Kotlin/Java package for Android (default: com.happ.<name>)")
    p.add_argument("--class", dest="cls", help="Kotlin/Java class name (default: CamelCase name)")
    p.add_argument("--cpu", help="CPU to optimize for, e.g. armv8.2-a+fp16+dotprod or x86-64-v2")
    p.add_argument("--strict-math", action="store_true", help="IEEE-exact float math (no fast-math)")
    p.add_argument("--no-gpu-runtime", action="store_true",
                   help="don't link the Vulkan runtime into Android/Windows/Linux libraries")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("emit", help="show generated code (LLVM IR, assembly, GLSL, Metal)")
    p.add_argument("file")
    p.add_argument("what", choices=["ir", "asm", "asm-clean", "glsl", "metal"])
    p.add_argument("-t", "--target", help="target (default: this computer)")
    p.add_argument("--cpu")
    p.add_argument("--strict-math", action="store_true")
    p.set_defaults(func=cmd_emit)

    p = sub.add_parser("doctor", help="show which tools were found")
    p.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    try:
        args.func(args)
    except HappError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except RecursionError:
        print("error: the program is nested too deeply (simplify the expression or split it into steps)",
              file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
