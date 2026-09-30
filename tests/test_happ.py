"""HA++ test suite.   python3 -m unittest discover -s tests -v

Tests skip (with a reason) when an optional tool is missing: qemu-aarch64 (ARM64
runs), glslangValidator + a Vulkan driver (GPU runs), javac/java (JNI), clang++
(Metal emulation).
"""
import contextlib
import ctypes
import io
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
sys.path.insert(0, ROOT)
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(ROOT, "gaussian"))

from happ.checker import check  # noqa: E402
from happ.driver import build, load_program, main as happ_main  # noqa: E402
from happ.errors import HappError  # noqa: E402
from happ.parser import parse  # noqa: E402
from happ.targets import Toolchain, expand_targets  # noqa: E402

TC = Toolchain()
P, U32, I64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64


def have(*tools):
    return all(TC.find(t) or shutil.which(t) for t in tools)


def vulkan_ok():
    if not have("glslangValidator"):
        return False
    try:
        r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=60)
        return r.returncode == 0 and "deviceName" in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


VULKAN = vulkan_ok()


def check_src(src):
    with open(os.path.join(ROOT, "happ", "std", "math.ha")) as fh:
        std_text = fh.read()
    m = parse(src, "test.ha")
    s = parse(std_text, "std/math.ha")
    s.is_std = True
    return check([m, s])


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="happ-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write(text)
        return path


# ====================================================================== front end

class FrontEnd(unittest.TestCase):
    ERRORS = [
        ("fn f() -> i32 { return 1.5; }", "expected an integer"),
        ("fn f(x: f32) -> f32 { return x + 1 as f64; }", "can't combine f32 and f64"),
        ("fn f() { let x = 1; x = 2; }", "declared with 'let'"),
        ("fn f() -> f32 { return y; }", "unknown name 'y'"),
        ("fn f() { let v = vec3(1.0, 2.0); }", "needs 3 components"),
        ("struct S { a: f32 } fn f(s: S) -> f32 { return s.b; }", "has no field 'b'"),
        ("fn f(x: i32) -> i32 { if x > 0 { return 1; } }", "without returning"),
        ("fn f() { break; }", "outside a loop"),
        ("fn f() { let x: u8 = 300; }", "doesn't fit in u8"),
        ("export fn f(v: vec3) {}", "only pass numbers, bools and pointers"),
        ("kernel k(x: *f64) {}", "GPUs can't store f64"),
        ("fn h(p: *f32) -> f32 { return p[0]; }\nkernel k(b: *f32) { b[0] = h(b); }", "takes a pointer"),
        ("fn f() { print(1); }\nkernel k(b: *f32) { f(); }", "print isn't available"),
        ("fn f() { barrier(); }", "only be used directly inside a kernel"),
        ("fn f() -> f32 { return sqr(2.0); }", "unknown function 'sqr'"),
        ("fn f() -> i32 { return 1 < 2 < 3; }", "cannot be chained"),
        ("struct A { b: B } struct B { a: A }", "contains itself"),
        ("fn f() { let x = 1.0u; }", "after number"),
        ("fn f(a: f32) -> f64 { return exp(a as f64); }", "not available for f64"),
        ("kernel k(a: *u32, b: *u32) { if a == b { a[0] = 1; } }", "can't be compared on the GPU"),
        ("fn f(a: *u32, b: *u8) -> bool { return a == b; }", "can't compare *u32 with *u8"),
        ("struct A { x: [size_of(A)]u8 }", "depends on itself"),
        ("struct P { q: Q } struct Q { p: P } struct R { x: [size_of(P)]u8 }", "contains itself"),
        ("fn size_of(x: i32) -> i32 { return x; }", "built-in name"),
        ("fn f() -> i64 { return size_of(Nope); }", "unknown type 'Nope'"),
        ("fn f() -> *u8 { return -1 as *u8; }", "doesn't fit in u64"),
    ]

    def test_error_messages(self):
        for src, expect in self.ERRORS:
            with self.subTest(src=src):
                with self.assertRaises(HappError) as cm:
                    check_src(src)
                self.assertIn(expect, str(cm.exception))

    def test_error_points_at_source(self):
        with self.assertRaises(HappError) as cm:
            check_src("fn f() -> f32 {\n    return unknown_thing;\n}")
        text = str(cm.exception)
        self.assertIn("test.ha:2:12", text)
        self.assertIn("return unknown_thing;", text)
        self.assertIn("^", text)

    def test_literals_adapt_to_context(self):
        prog = check_src("fn f(h: f16, u: u8, v: vec3) -> f16 { let a = u + 1; let b = v * 2.0; return h * 2.0; }")
        self.assertIn("f", prog.fns)

    def test_user_names_override_std(self):
        prog = check_src("const PI: f32 = 3.0; fn sigmoid(x: f32) -> f32 { return x; }")
        self.assertEqual(prog.consts["PI"].value, 3.0)
        self.assertFalse(prog.fns["sigmoid"].is_std)
        # the std library has locals such as 'a' and 'r'; user structs with those names don't break it
        check_src("struct a { x: f32 } struct r { y: f32 } fn f(p: a) -> f32 { return atan2(p.x, 1.0) + exp(p.x); }")
        with self.assertRaises(HappError):     # but in user code a local can't take a type's name
            check_src("struct a { x: f32 } fn f() -> f32 { let a = 1.0; return a; }")


# ====================================================================== CPU

class CPU(TempDirCase):
    def test_run_hello(self):
        path = self.write("hello.ha", """
struct P { pos: vec3, w: f32 }
fn main() -> i32 {
    var p = P(pos: vec3(1.0, 2.0, 3.0), w: 0.5);
    p.pos.xz = vec2(9.0, 8.0);
    print("hi", 42, p.pos, sigmoid(0.0), true);
    return 3;
}
""")
        r = subprocess.run([sys.executable, "-m", "happ", "run", path], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertEqual(r.stdout.strip(), "hi 42 (9, 2, 8) 0.5 true")

    def test_pointers_and_sizes(self):
        path = self.write("ptr.ha", """
struct Node { next: *Node, size: u64, tag: u32 }
struct Later { a: [size_of(Node)]u8, b: Tail }
struct Tail { v: vec3, h: f16 }
struct Outer { x: [size_of(Mid)]u8 }
struct Mid { c: Inner, d: [2]Inner }
struct Inner { v: vec4 }
const NODE = size_of(Node);
fn main() {
    let null = 0 as *Node;
    var a: Node;
    var b: Node;
    let pa = &a;
    let pb = &b;
    let lo = select(pa < pb, pa, pb);
    let hi = select(pa < pb, pb, pa);
    print(pa == null, pa != null, pa == pa, pa == pb, lo < hi, hi >= lo, lo > hi);
    let n: u64 = size_of(Node) * 3;
    print(NODE, n, align_of(Node), size_of(Later), size_of(Tail), size_of([3]vec3), size_of(*u8));
    print(size_of(Outer), size_of(Mid));
}
""")
        r = subprocess.run([sys.executable, "-m", "happ", "run", path], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split("\n")[:3], ["false true true false true true false",
                                                    "24 72 8 40 16 36 8", "48 48"])

    def test_features_native(self):
        import arm64_run
        values = arm64_run.run_native(self.tmp)
        ints = [v for t, v in values if t == "i"]
        self.assertEqual(ints[:17], [-56, 4, -2147483648, -3, -1, 1333333333, 3, 2, -4, 1073741820, 242, 24, 31,
                                     7, 250, 23416728348467685, 9223372030926249001])
        self.assertEqual(ints[17:22], [2147483647, -2147483648, 0, 255, 0])

    def test_math_accuracy(self):
        path = self.write("m.ha", "".join(
            f"export fn t_{f}(x: *f32, o: *f32, n: i64) {{ for i in 0..n {{ o[i] = {f}(x[i]); }} }}\n"
            for f in ("exp", "log", "sin", "cos", "tanh", "atan", "asin", "exp2", "log2")))
        targets = ["linux-x64"] + (["web-wasm32"] if have("node", "wasm-ld") else [])
        produced = build(path, targets, self.tmp, quiet=True, bridges=False)
        lib = ctypes.CDLL(produced["linux-x64"])
        rng = np.random.default_rng(1)
        cases = {"exp": (np.exp, rng.uniform(-80, 80, 200000), 2),
                 "log": (np.log, np.exp(rng.uniform(-80, 80, 200000)), 2),
                 "sin": (np.sin, rng.uniform(-50, 50, 200000), 1e-7),
                 "cos": (np.cos, rng.uniform(-50, 50, 200000), 1e-7),
                 "tanh": (np.tanh, rng.uniform(-10, 10, 200000), 2),
                 "atan": (np.arctan, rng.uniform(-50, 50, 200000), 3),
                 "asin": (np.arcsin, rng.uniform(-1, 1, 200000), 3),
                 "exp2": (np.exp2, rng.uniform(-120, 120, 200000), 2),
                 "log2": (np.log2, np.exp(rng.uniform(-80, 80, 200000)), 2)}
        wasm_out = {}
        if "web-wasm32" in produced:        # the same functions as WebAssembly (browsers, Safari on iPhone)
            for name, (_, x, _) in cases.items():
                x.astype(np.float32).tofile(os.path.join(self.tmp, f"in_{name}.bin"))
            js = self.write("math.mjs", f"""
import fs from "node:fs";
const {{ instance }} = await WebAssembly.instantiate(fs.readFileSync("{produced['web-wasm32']}"), {{ env: {{}} }});
const e = instance.exports, mem = e.memory;
for (const name of {list(cases)}) {{
    const x = new Float32Array(fs.readFileSync("{self.tmp}/in_" + name + ".bin").buffer.slice(0));
    const base = Math.ceil(Number(e.__heap_base.value) / 16) * 16, out = base + x.length * 4;
    const need = out + x.length * 4;
    if (need > mem.buffer.byteLength) mem.grow(Math.ceil((need - mem.buffer.byteLength) / 65536));
    new Float32Array(mem.buffer, base, x.length).set(x);
    e["t_" + name](base, out, BigInt(x.length));
    fs.writeFileSync("{self.tmp}/out_" + name + ".bin", new Uint8Array(mem.buffer, out, x.length * 4));
}}
""")
            subprocess.run(["node", js], check=True)
            for name in cases:
                wasm_out[name] = np.fromfile(os.path.join(self.tmp, f"out_{name}.bin"), np.float32)
        for name, (ref, x, tol) in cases.items():
            x = x.astype(np.float32)
            o = np.empty_like(x)
            f = getattr(lib, f"t_{name}")
            f(P(x.ctypes.data), P(o.ctypes.data), I64(len(x)))
            r = ref(x.astype(np.float64))
            for where, got in (("x86-64", o), ("wasm", wasm_out.get(name))):
                if got is None:
                    continue
                with self.subTest(fn=name, target=where):
                    if tol < 1:   # absolute error (sin/cos near zeros)
                        self.assertLess(np.abs(got - r).max(), tol)
                    else:         # ulp
                        ulp = np.spacing(np.abs(r.astype(np.float32))).astype(np.float64)
                        self.assertLessEqual((np.abs(got - r) / ulp).max(), tol)

    def test_fast_loops_vectorize(self):
        path = self.write("v.ha", "export fn f(x: *f32, n: i64) { for i in 0..n { x[i] = sigmoid(x[i]) * 2.0; } }")
        out = io.StringIO()
        for target, pattern in (("android-arm64", "fmul\tv"), ("linux-x64", "vmulps")):
            with self.subTest(target=target), contextlib.redirect_stdout(out):
                happ_main(["emit", path, "asm", "-t", target])
            self.assertIn(pattern, out.getvalue())


# ====================================================================== targets

class Targets(TempDirCase):
    SRC = """
struct V { p: vec3, h: f16 }
export fn scale(a: *f32, n: i64, k: f32) { for i in 0..n { a[i] = a[i] * k; } }
export fn norm(v: *V) -> f32 { return length(v.p) + (v.h as f32); }
"""

    def test_every_target_links_without_sdks(self):
        path = self.write("lib.ha", self.SRC)
        out = os.path.join(self.tmp, "build")
        produced = build(path, expand_targets("all"), out, quiet=True)
        readelf, readobj = TC.find("llvm-readelf"), TC.find("llvm-readobj")
        for t in expand_targets("all"):
            with self.subTest(target=t):
                with open(produced[t], "rb") as fh:
                    data = fh.read(8)
                if t.startswith(("android", "linux")):
                    self.assertEqual(data[:4], b"\x7fELF")
                elif t.startswith("windows"):
                    self.assertEqual(data[:2], b"MZ")
                elif t.startswith("web"):
                    self.assertEqual(data[:4], b"\x00asm")
                else:
                    self.assertEqual(data[:8], b"!<arch>\n")
        if readelf:
            for abi in ("arm64-v8a", "x86_64"):
                so = os.path.join(out, "android", "jniLibs", abi, "liblib.so")
                seg = subprocess.run([readelf, "-lW", so], capture_output=True, text=True).stdout
                self.assertIn("0x4000", seg, "Android 15+ needs 16 KB aligned segments")
                dyn = subprocess.run([readelf, "-d", so], capture_output=True, text=True).stdout
                self.assertNotIn("NEEDED", dyn, "a plain HA++ library needs no other library")
        if readobj:
            exp = subprocess.run([readobj, "--coff-exports", produced["windows-x64"]], capture_output=True,
                                 text=True).stdout
            for sym in ("scale", "norm", "Java_com_happ_lib_Lib_scale"):
                self.assertIn(sym, exp)
        with open(os.path.join(out, "apple", "Lib", "LibCore.xcframework", "Info.plist"), "rb") as fh:
            plist = plistlib.load(fh)
        ids = sorted(lib["LibraryIdentifier"] for lib in plist["AvailableLibraries"])
        self.assertEqual(ids, ["ios-arm64", "ios-arm64_x86_64-simulator", "macos-arm64_x86_64"])
        with open(os.path.join(out, "include", "lib.h")) as fh:
            header = fh.read()
        self.assertIn("float norm(V *v);", header)
        self.assertIn("HA_ASSERT_SIZE(V, 16);", header)
        # the generated C header compiles and agrees with HA++ layout
        cfile = self.write("use.c", '#include "lib.h"\nint main(void) { V v = {{1,2,3}, 0}; return (int)norm(&v); }\n')
        subprocess.run([TC.find("clang"), "-fsyntax-only", "-I", os.path.join(out, "include"), cfile], check=True)

    def test_header_with_linked_structs(self):
        # a struct pointing to itself, and to a struct defined after it, must give a valid C header
        path = self.write("links.ha", """
struct Item { next: *Item, owner: *List, v: f32 }
struct List { head: *Item, count: u32 }
export fn total(l: *List) -> f32 {
    var s = 0.0;
    var it = l.head;
    while it as u64 != 0 { s += it.v; it = it.next; }
    return s;
}
""")
        out = os.path.join(self.tmp, "build")
        build(path, ["linux-x64"], out, quiet=True)
        cfile = self.write("use.c", '#include "links.h"\nint main(void) { Item a = {0, 0, 2.0f}; List l = {&a, 1}; '
                                    'a.owner = &l; return (int)total(&l); }\n')
        for compiler, std in (("clang", "-std=c11"), ("clang++", "-std=c++17")):
            if have(compiler):
                subprocess.run([TC.find(compiler) or compiler, std, "-x", "c" if compiler == "clang" else "c++",
                                "-fsyntax-only", "-I", os.path.join(out, "include"), cfile], check=True)

    @unittest.skipUnless(have("node", "wasm-ld"), "needs node and wasm-ld")
    def test_wasm_runs_like_native(self):
        # the same program as WebAssembly (what Safari on an iPhone runs) and natively: identical results,
        # including structs holding pointers (8 bytes in HA++ on every target) and the TLSF allocator
        path = self.write("w.ha", """
import "mem.ha";
struct Pair { next: *Pair, v: u32 }
export fn soft(x: f32) -> f32 { return sigmoid(x) + tanh(x) + max(x, 0.5) + (x as f16) as f32; }
export fn pair_size() -> u32 { return size_of(Pair); }
export fn link_pairs(p: *Pair, n: u32) -> u32 {
    for i in 0..n - 1 { p[i].next = p + (i + 1); p[i].v = i * 3; }
    p[n - 1].next = 0 as *Pair;
    p[n - 1].v = 99;
    var s: u32 = 0;
    var q = p;
    while q as u64 != 0 { s += q.v; q = q.next; }
    return s;
}
export fn heap(mem: *u8, bytes: u32) -> i32 {
    let t = tlsf_create(mem, bytes as u64);
    let a = tlsf_malloc(t, 1000);
    let b = tlsf_malloc(t, 50);
    tlsf_free(t, a);
    let c = tlsf_realloc(t, b, 3000);
    return tlsf_check(t) * 1000 + ((c as u64) % 16) as i32 + select(c as u64 == 0, 500, 0);
}
""")
        out = os.path.join(self.tmp, "build")
        produced = build(path, ["web-wasm32", "linux-x64"], out, quiet=True)
        xs = [-30.0, -2.5, -1e-3, 0.0, 0.3, 1.0, 7.5, 80.0, 1e-5, 65504.0]
        js = self.write("run.mjs", f"""
import {{ load, SIZEOF }} from "{os.path.join(out, 'web', 'w.mjs')}";
const lib = await load(new URL("file://{produced['web-wasm32']}"));
const e = lib.exports;
const r = {xs}.map(x => e.soft(x));
r.push(e.pair_size(), SIZEOF.Pair);
r.push(e.link_pairs(lib.alloc(16 * 10), 10));
r.push(e.heap(lib.alloc(1 << 20), 1 << 20));
console.log(JSON.stringify(r));
""")
        got = subprocess.run(["node", js], capture_output=True, text=True, check=True).stdout
        import json
        wasm = json.loads(got)
        lib = ctypes.CDLL(produced["linux-x64"])
        lib.soft.restype, lib.soft.argtypes = ctypes.c_float, [ctypes.c_float]
        lib.link_pairs.restype, lib.link_pairs.argtypes = U32, [P, U32]
        lib.heap.restype, lib.heap.argtypes = ctypes.c_int32, [P, U32]
        pairs, mem = np.zeros(160, np.uint8), np.zeros(1 << 20, np.uint8)
        native = [lib.soft(x) for x in xs] + [16, 16, lib.link_pairs(pairs.ctypes.data, 10),
                                              lib.heap(mem.ctypes.data, 1 << 20)]
        # the std math functions may round an ulp differently: x86-64-v3 and ARM64 fuse multiply-adds
        # (fast-math contraction), WebAssembly has no fma instruction. Everything else is exact.
        np.testing.assert_allclose(wasm[:len(xs)], native[:len(xs)], rtol=2.5e-7, atol=0)
        self.assertEqual(wasm[len(xs):], native[len(xs):])
        self.assertEqual(wasm[-2:], [207, 0])

    @unittest.skipUnless(have("qemu-aarch64", "ld.lld"), "needs qemu-aarch64 and ld.lld")
    def test_arm64_matches_x86_64(self):
        import arm64_run
        native = arm64_run.run_native(self.tmp)
        arm = arm64_run.run_arm64(self.tmp)
        self.assertEqual(len(native), len(arm))
        self.assertEqual(arm64_run.compare(native, arm), [])


# ====================================================================== GPU

@unittest.skipUnless(have("clang") and os.path.exists("/usr/include/vulkan/vulkan.h"), "needs vulkan.h")
class VulkanHeader(unittest.TestCase):
    def test_minimal_header_matches_official(self):
        import vk_layout
        n, diffs = vk_layout.run()
        self.assertGreater(n, 200)
        self.assertEqual(diffs, [])


EDGE_N = 100            # tests/gpu_edge.ha: two workgroups, the second one partly idle
EDGE_M_BYTES = 76       # struct M { m: mat3, v: vec3, w: [3][2]f32, h: hvec2 }


def edge_inputs():
    rng = np.random.default_rng(5)
    n = EDGE_N
    ms = np.zeros((n, EDGE_M_BYTES // 4), np.float32)
    ms[:, :18] = rng.uniform(-10, 10, (n, 18)).astype(np.float32)
    ms.view(np.uint32)[:, 18] = rng.uniform(-8, 8, (n, 2)).astype(np.float16).view(np.uint32).ravel()
    iv = rng.integers(-2 ** 31, 2 ** 31, (n, 3), dtype=np.int64).astype(np.int32)
    iv[:4] = -2 ** 31                            # INT_MIN / -1 and INT_MIN % -1
    uv = rng.integers(0, 2 ** 32, (n, 3), dtype=np.uint64).astype(np.uint32)
    return [ms, np.zeros(3, np.uint32), rng.normal(size=2 * n).astype(np.float32), iv, uv,
            np.zeros(n, np.float32)]


def edge_cpu(lib):
    arrays = [a.copy() for a in edge_inputs()]
    lib.edge_cpu.argtypes = [P] * 6 + [ctypes.c_float, U32, U32, U32, U32]
    lib.edge_cpu(*[a.ctypes.data for a in arrays], 1.5, EDGE_N, 2, 1, 1)
    return arrays


def edge_compare(case, got, want):
    ms, cnt, hits, iv, uv, out = got
    ms_w, cnt_w, hits_w, iv_w, uv_w, out_w = want
    np.testing.assert_array_equal(cnt, [EDGE_N] * 3)          # every atomic ran exactly once
    np.testing.assert_array_equal(cnt_w, [EDGE_N] * 3)
    np.testing.assert_array_equal(iv, iv_w)
    np.testing.assert_array_equal(uv, uv_w)
    np.testing.assert_array_equal(hits, hits_w)
    np.testing.assert_allclose(ms[:, :18], ms_w[:, :18], rtol=1e-5, atol=1e-5)
    half = lambda a: np.ascontiguousarray(a[:, 18]).view(np.float16)
    np.testing.assert_allclose(half(ms), half(ms_w), rtol=2e-3)
    np.testing.assert_allclose(out, out_w, rtol=1e-5, atol=1e-4)
    i = np.arange(EDGE_N)                                     # m[1].y = 5 landed in memory (then x3 where
    np.testing.assert_array_equal(ms_w[:, 4], np.where((i % 3 == 1) & (i % 2 == 1), 15.0, 5.0))  # m[i%3][i%2])


@unittest.skipUnless(VULKAN, "needs glslangValidator and a Vulkan driver (e.g. lavapipe)")
class GPU(TempDirCase):
    def gpu_lib(self, src, name):
        out = build(src, ["linux-x64"], os.path.join(self.tmp, name), quiet=True)
        lib = ctypes.CDLL(out["linux-x64"])
        for f, r, a in [("ha_gpu_create", P, [ctypes.c_char_p, ctypes.c_size_t]),
                        ("ha_buffer_create", P, [P, ctypes.c_size_t]), ("ha_buffer_data", P, [P]),
                        ("ha_kernel_create", P, [P, P, ctypes.c_size_t, U32, U32]),
                        ("ha_dispatch", ctypes.c_int, [P, P, P, P, U32, U32, U32])]:
            fn = getattr(lib, f)
            fn.restype, fn.argtypes = r, a
        gpu = lib.ha_gpu_create(None, 0)
        self.assertTrue(gpu)
        return lib, gpu

    def run_kernel(self, lib, gpu, libname, kname, arrays, push, groups):
        getter = getattr(lib, f"{libname}_spirv_{kname}")
        getter.restype, getter.argtypes = P, [P]
        size = ctypes.c_uint64()
        blob = getter(ctypes.byref(size))
        k = lib.ha_kernel_create(gpu, blob, size.value, len(arrays), len(push))
        bufs = []
        for a in arrays:
            b = lib.ha_buffer_create(gpu, a.nbytes)
            ctypes.memmove(lib.ha_buffer_data(b), a.ctypes.data, a.nbytes)
            bufs.append(b)
        arr = (P * len(bufs))(*bufs)
        self.assertEqual(lib.ha_dispatch(gpu, k, arr, push or None, *groups), 0)
        return [np.frombuffer(ctypes.string_at(lib.ha_buffer_data(b), a.nbytes), a.dtype).reshape(a.shape).copy()
                for b, a in zip(bufs, arrays)]

    def test_kernels_match_cpu(self):
        src = os.path.join(TESTS, "kernels.ha")
        lib, gpu = self.gpu_lib(src, "k")
        rng = np.random.default_rng(0)
        n = 1000
        splats = rng.normal(size=(n, 14)).astype(np.float32)
        splats[:, 2] = rng.uniform(1, 5, n)
        push = struct.pack("<If", n, 500.0)
        _, gpu_out = self.run_kernel(lib, gpu, "kernels", "project", [splats, np.zeros((n, 7), np.float32)], push,
                                     ((n + 63) // 64, 1, 1))
        cpu_out = np.zeros((n, 7), np.float32)
        lib.project_cpu.argtypes = [P, P, U32, ctypes.c_float, U32, U32, U32]
        lib.project_cpu(splats.ctypes.data, cpu_out.ctypes.data, n, 500.0, (n + 63) // 64, 1, 1)
        self.assertLess((np.abs(gpu_out - cpu_out) / (np.abs(cpu_out) + 1e-3)).max(), 1e-3)
        keys = rng.integers(0, 2 ** 32, 100000, dtype=np.uint32)
        _, counts = self.run_kernel(lib, gpu, "kernels", "hist", [keys, np.zeros(256, np.uint32)],
                                    struct.pack("<I", len(keys)), ((len(keys) + 255) // 256, 1, 1))
        np.testing.assert_array_equal(counts, np.bincount(keys >> 24, minlength=256))

    def test_edge_cases_match_cpu(self):
        src = os.path.join(TESTS, "gpu_edge.ha")
        lib, gpu = self.gpu_lib(src, "edge")
        got = self.run_kernel(lib, gpu, "gpu_edge", "edge", edge_inputs(), struct.pack("<fI", 1.5, EDGE_N),
                              (2, 1, 1))
        edge_compare(self, got, edge_cpu(lib))

    def test_runtime_batches_and_errors(self):
        """Many dispatches in one batch (descriptor pools chained), bad calls, and recovery after them."""
        src = self.write("rt.ha", "kernel bump(a: *u32, b: *u32, n: u32) {\n"
                                  "    let i = global_id.x;\n    if i < n { a[i] += 1; b[i] += a[i]; }\n}\n")
        out = build(src, ["linux-x64"], os.path.join(self.tmp, "rt"), quiet=True)
        with open(os.path.join(os.path.dirname(out["linux-x64"]), "..", "gpu", "rt_bump.spv"), "rb") as f:
            spv = f.read()
        so = os.path.join(self.tmp, "libhagpu_small_pools.so")
        subprocess.run([TC.find("clang"), "-shared", "-fPIC", "-O2", "-DHA_MAX_SETS=3",
                        os.path.join(ROOT, "runtime", "gpu", "ha_gpu.c"), "-o", so], check=True)
        lib = ctypes.CDLL(so)
        sig = [("ha_gpu_create", P, [ctypes.c_char_p, ctypes.c_size_t]), ("ha_buffer_create", P, [P, ctypes.c_size_t]),
               ("ha_buffer_data", P, [P]), ("ha_kernel_create", P, [P, ctypes.c_char_p, ctypes.c_size_t, U32, U32]),
               ("ha_batch_begin", ctypes.c_int, [P]), ("ha_batch_submit", ctypes.c_int, [P]),
               ("ha_batch_dispatch", ctypes.c_int, [P, P, P, P, U32, U32, U32]),
               ("ha_kernel_destroy", None, [P]), ("ha_gpu_destroy", None, [P])]
        for f, r, a in sig:
            fn = getattr(lib, f)
            fn.restype, fn.argtypes = r, a
        gpu = lib.ha_gpu_create(None, 0)
        self.assertTrue(gpu)
        n = 100
        k = lib.ha_kernel_create(gpu, spv, len(spv), 2, 4)
        a, b = lib.ha_buffer_create(gpu, 4 * n), lib.ha_buffer_create(gpu, 4 * n)
        ctypes.memset(lib.ha_buffer_data(a), 0, 4 * n)
        ctypes.memset(lib.ha_buffer_data(b), 0, 4 * n)
        bufs = (P * 2)(a, b)
        push = struct.pack("<I", n)

        def batch(count, bad_at=()):
            self.assertEqual(lib.ha_batch_begin(gpu), 0)
            for j in range(count):
                if j in bad_at:     # a missing buffer: rejected before anything is recorded
                    self.assertEqual(lib.ha_batch_dispatch(gpu, k, (P * 2)(a, None), push, 2, 1, 1), -4)
                self.assertEqual(lib.ha_batch_dispatch(gpu, k, bufs, push, 2, 1, 1), 0)
            return lib.ha_batch_submit(gpu)

        self.assertEqual(batch(40, bad_at={5, 17}), 0)      # 40 dispatches with 3 sets per pool: 14 pools
        self.assertEqual(batch(7), 0)                       # pools were reset and are reused
        va = np.frombuffer(ctypes.string_at(lib.ha_buffer_data(a), 4 * n), np.uint32)
        vb = np.frombuffer(ctypes.string_at(lib.ha_buffer_data(b), 4 * n), np.uint32)
        self.assertTrue((va == 47).all())                   # every dispatch ran exactly once, in order
        self.assertTrue((vb == 47 * 48 // 2).all())
        self.assertEqual(lib.ha_batch_submit(gpu), -1)      # no batch open
        self.assertEqual(lib.ha_batch_dispatch(gpu, k, bufs, push, 1, 1, 1), -1)
        lib.ha_kernel_destroy(k)
        lib.ha_gpu_destroy(gpu)

    def test_splat_pipeline_matches_reference(self):
        import render as R
        import splat_reference as ref
        pipe = R.Pipeline(R.build(out_dir=os.path.join(self.tmp, "splat")))
        raw = R.synthetic_scene(1500, seed=3)
        pipe.upload(raw)
        packed = pipe.view("splats", np.uint32, len(raw) * 8).reshape(-1, 8).copy()
        np.testing.assert_array_equal(packed, ref.pack(raw))
        w, h = 200, 136
        cam = R.look_at([0, 2.0, -3.0], [0, 0, 0], w, h)
        img, total = pipe.render(cam, w, h, background=(8, 10, 22))
        rimg, rkeys, rvals = ref.render(packed, cam, w, h, background=(8, 10, 22))
        self.assertEqual(total, len(rkeys))
        np.testing.assert_array_equal(pipe.view("keys0", np.uint32, total), rkeys)
        np.testing.assert_array_equal(pipe.view("vals0", np.uint32, total), rvals)
        diff = np.abs(img.view(np.uint8).astype(int) - rimg.view(np.uint8).astype(int))
        self.assertLessEqual(diff.max(), 2)

    def test_mlp_block_matches_numpy(self):
        lib, gpu = self.gpu_lib(os.path.join(ROOT, "examples", "ai", "nn.ha"), "nn")
        rng = np.random.default_rng(0)
        m, k, n = 48, 96, 80
        a = rng.normal(size=(m, k)).astype(np.float32)
        b = rng.normal(size=(k, n)).astype(np.float32)
        c = np.zeros((m, n), np.float32)
        _, _, c_gpu = self.run_kernel(lib, gpu, "nn", "matmul", [a, b, c], struct.pack("<III", m, n, k),
                                      ((n + 15) // 16, (m + 15) // 16, 1))
        np.testing.assert_allclose(c_gpu, a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4)
        (s_gpu,) = self.run_kernel(lib, gpu, "nn", "softmax", [a.copy()], struct.pack("<II", m, k), (m, 1, 1))
        e = np.exp(a - a.max(1, keepdims=True))
        np.testing.assert_allclose(s_gpu, e / e.sum(1, keepdims=True), rtol=1e-5, atol=1e-6)
        lib.matmul_cpu.argtypes = [P, P, P, I64, I64, I64]
        lib.matmul_cpu(a.ctypes.data, b.ctypes.data, c.ctypes.data, m, n, k)
        np.testing.assert_allclose(c, a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(have("javac", "java"), "needs a JDK")
    def test_jni_from_java(self):
        out = os.path.join(self.tmp, "jk")
        build(os.path.join(TESTS, "kernels.ha"), ["linux-x64"], out, quiet=True)
        classes = os.path.join(self.tmp, "classes")
        subprocess.run(["javac", "-d", classes, os.path.join(out, "android", "java", "com", "happ", "kernels",
                                                             "Kernels.java"),
                        os.path.join(TESTS, "jvm", "KernelsTest.java")], check=True, capture_output=True)
        r = subprocess.run(["java", f"-Djava.library.path={os.path.join(out, 'linux-x64')}", "-cp", classes,
                            "KernelsTest"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count("correct"), 3, r.stdout)


    @unittest.skipUnless(have("javac", "java"), "needs a JDK")
    def test_android_splat_renderer_from_java(self):
        """gaussian/android/SplatRenderer.java must render exactly what the Python host renders."""
        import render as R
        out = os.path.join(self.tmp, "sb")
        lib = R.build(out_dir=out)
        raw = R.synthetic_scene(2000, seed=5)
        raw.tofile(os.path.join(self.tmp, "raw.bin"))
        w, h = 160, 112
        cam = R.look_at([0, 2.4, -3.5], [0, 0, 0], w, h)
        with open(os.path.join(self.tmp, "cam.bin"), "wb") as f:
            f.write(R.camera_bytes(cam, w, h, (w + 15) // 16, (h + 15) // 16))
        pipe = R.Pipeline(lib)
        pipe.upload(raw)
        img, _ = pipe.render(cam, w, h, background=(8, 10, 22))
        classes = os.path.join(self.tmp, "classes")
        subprocess.run(["javac", "-d", classes, os.path.join(out, "android", "java", "com", "happ", "splat", "Splat.java"),
                        os.path.join(ROOT, "gaussian", "android", "SplatRenderer.java"),
                        os.path.join(TESTS, "jvm", "SplatDemo.java")], check=True, capture_output=True)
        r = subprocess.run(["java", f"-Djava.library.path={os.path.join(out, 'linux-x64')}", "-cp", classes,
                            "SplatDemo", self.tmp, str(w), str(h)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        java_img = np.fromfile(os.path.join(self.tmp, "out.bin"), np.uint8)
        np.testing.assert_array_equal(java_img, img.view(np.uint8).ravel())


class Fuzz(TempDirCase):
    def test_short_differential_fuzz(self):
        """Random expressions on every available back end vs the Python model (tests/fuzz.py runs longer)."""
        import fuzz
        runner = fuzz.Runner(self.tmp)
        for seed in (11, 12):
            with self.subTest(seed=seed):
                bad, src, _ = fuzz.run_one(seed, runner, n_expr=12, depth=4)
                self.assertEqual(bad[:5], [], f"fuzz seed {seed} disagrees with the model")


@unittest.skipUnless(have("clang", "ld64.lld"), "needs clang and ld64.lld")
class IPhoneApp(TempDirCase):
    def test_ipa_builds_without_apple_files(self):
        """gaussian/build_ios.py: a complete iPhone app from LLVM alone (running it needs a real iPhone)."""
        import zipfile
        out = os.path.join(self.tmp, "ios")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "gaussian", "build_ios.py"), "--no-icon",
                            "--out", out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with zipfile.ZipFile(os.path.join(out, "Splats.ipa")) as z:
            names = z.namelist()
            self.assertIn("Payload/Splats.app/Splats", names)
            zi = z.getinfo("Payload/Splats.app/Splats")
            self.assertTrue((zi.external_attr >> 16) & 0o111, "the app binary must be executable")
            self.assertEqual(zi.create_system, 3, "Unix attributes, or unzippers drop the executable bit")
            info = plistlib.loads(z.read("Payload/Splats.app/Info.plist"))
            exe = z.read("Payload/Splats.app/Splats")
        self.assertEqual(info["CFBundleExecutable"], "Splats")
        self.assertEqual(info["MinimumOSVersion"], "15.0")
        self.assertEqual(struct.unpack("<II", exe[:8]), (0xFEEDFACF, 0x0100000C))   # 64-bit Mach-O, arm64
        otool = TC.find("llvm-otool")
        nm = TC.find("llvm-nm")
        if otool and nm:
            app = os.path.join(out, "Splats.app", "Splats")
            lc = subprocess.run([otool, "-l", app], capture_output=True, text=True, check=True).stdout
            for cmd in ("LC_MAIN", "LC_BUILD_VERSION", "LC_CODE_SIGNATURE", "LC_DYLD_CHAINED_FIXUPS"):
                self.assertIn(cmd, lc)
            self.assertRegex(lc, r"platform 2\b")                    # iOS
            # every symbol the app imports is one the .tbd files promise an iOS library exports
            import build_ios
            exported = {x for syms in build_ios.SYSTEM_SYMBOLS.values() for x in syms}
            undef = subprocess.run([nm, "-u", app], capture_output=True, text=True, check=True).stdout.split()
            self.assertTrue(undef)
            self.assertEqual(sorted(set(undef) - exported), [])
            # the display link must be added in the real common-modes object (compared by address)
            self.assertIn("_kCFRunLoopCommonModes", undef)


@unittest.skipUnless(have("clang++"), "needs clang++")
class MetalEmulation(TempDirCase):
    def test_metal_kernels_compile_and_match_cpu(self):
        from happ.gpu import generate_metal
        prog = load_program(os.path.join(TESTS, "kernels.ha"))
        with open(os.path.join(self.tmp, "k.metal"), "w") as f:
            f.write(generate_metal(prog))
        n = 500
        rng = np.random.default_rng(0)
        splats = rng.normal(size=(n, 14)).astype(np.float32)
        splats[:, 2] = rng.uniform(1, 5, n)
        splats.tofile(os.path.join(self.tmp, "in.bin"))
        drv = self.write("drv.cpp", f"""#include "k.metal"
#include <cstdio>
#include <vector>
int main() {{
    const unsigned n = {n};
    std::vector<float> s(n * 14), out(n * 7, 0.0f);
    FILE *f = fopen("in.bin", "rb"); fread(s.data(), 4, n * 14, f); fclose(f);
    ha_params_project p{{n, 500.0f}};
    static_assert(sizeof(hs_Splat) == 56 && sizeof(hs_Out2D) == 28, "layout");
    for (unsigned x = 0; x < ((n + 63) / 64) * 64; x++)
        project((hs_Splat *)s.data(), (hs_Out2D *)out.data(), p, metal::uint3{{x, 0, 0}}, metal::uint3{{x % 64, 0, 0}},
                metal::uint3{{x / 64, 0, 0}}, metal::uint3{{(n + 63) / 64, 1, 1}}, metal::uint3{{64, 1, 1}});
    f = fopen("out.bin", "wb"); fwrite(out.data(), 4, n * 7, f); fclose(f);
}}
""")
        exe = os.path.join(self.tmp, "drv")
        subprocess.run(["clang++", "-std=c++17", "-O1", "-I", os.path.join(TESTS, "metal_shim"),
                        "-Wno-unknown-attributes", "-Wno-ignored-attributes", drv, "-o", exe], check=True)
        subprocess.run([exe], check=True, cwd=self.tmp)
        metal_out = np.fromfile(os.path.join(self.tmp, "out.bin"), np.float32).reshape(n, 7)
        lib = ctypes.CDLL(build(os.path.join(TESTS, "kernels.ha"), ["linux-x64"], os.path.join(self.tmp, "b"),
                                quiet=True, bridges=False, gpu_runtime=False)["linux-x64"])
        lib.project_cpu.argtypes = [P, P, U32, ctypes.c_float, U32, U32, U32]
        cpu = np.zeros((n, 7), np.float32)
        lib.project_cpu(splats.ctypes.data, cpu.ctypes.data, n, 500.0, (n + 63) // 64, 1, 1)
        self.assertLess((np.abs(metal_out - cpu) / (np.abs(cpu) + 1e-3)).max(), 1e-3)
        # every example's Metal output at least compiles
        for ex in ("gaussian/splat.ha", "examples/ai/nn.ha"):
            with self.subTest(example=ex):
                src = os.path.join(self.tmp, "ex.metal")
                with open(src, "w") as f:
                    f.write(generate_metal(load_program(os.path.join(ROOT, ex))))
                subprocess.run(["clang++", "-std=c++17", "-fsyntax-only", "-x", "c++", "-I",
                                os.path.join(TESTS, "metal_shim"), "-Wno-unknown-attributes",
                                "-Wno-ignored-attributes", src], check=True)

    def test_metal_edge_cases_match_cpu(self):
        from happ.gpu import generate_metal
        src = os.path.join(TESTS, "gpu_edge.ha")
        with open(os.path.join(self.tmp, "k.metal"), "w") as f:
            f.write(generate_metal(load_program(src)))
        arrays = edge_inputs()
        for i, a in enumerate(arrays):
            a.tofile(os.path.join(self.tmp, f"in{i}.bin"))
        sizes = [a.nbytes for a in arrays]
        drv = self.write("drv.cpp", f"""#include "k.metal"
#include <cstdio>
#include <vector>
int main() {{
    const unsigned sizes[6] = {{{", ".join(map(str, sizes))}}};
    std::vector<std::vector<unsigned char>> b(6);
    for (int i = 0; i < 6; i++) {{
        char name[16]; snprintf(name, sizeof name, "in%d.bin", i);
        b[i].resize(sizes[i]);
        FILE *f = fopen(name, "rb"); fread(b[i].data(), 1, sizes[i], f); fclose(f);
    }}
    static_assert(sizeof(hs_M) == {EDGE_M_BYTES}, "layout");
    ha_params_edge p{{1.5f, {EDGE_N}u}};
    for (unsigned x = 0; x < 128; x++)
        edge((hs_M *)b[0].data(), (uint *)b[1].data(), (float *)b[2].data(), (packed_int3 *)b[3].data(),
             (packed_uint3 *)b[4].data(), (float *)b[5].data(), p, metal::uint3{{x, 0, 0}},
             metal::uint3{{x % 64, 0, 0}}, metal::uint3{{x / 64, 0, 0}}, metal::uint3{{2, 1, 1}},
             metal::uint3{{64, 1, 1}});
    for (int i = 0; i < 6; i++) {{
        char name[16]; snprintf(name, sizeof name, "out%d.bin", i);
        FILE *f = fopen(name, "wb"); fwrite(b[i].data(), 1, sizes[i], f); fclose(f);
    }}
}}
""")
        exe = os.path.join(self.tmp, "drv")
        subprocess.run(["clang++", "-std=c++17", "-O1", "-I", os.path.join(TESTS, "metal_shim"),
                        "-Wno-unknown-attributes", "-Wno-ignored-attributes", drv, "-o", exe], check=True)
        subprocess.run([exe], check=True, cwd=self.tmp)
        got = [np.fromfile(os.path.join(self.tmp, f"out{i}.bin"), a.dtype).reshape(a.shape)
               for i, a in enumerate(arrays)]
        lib = ctypes.CDLL(build(src, ["linux-x64"], os.path.join(self.tmp, "b"), quiet=True, bridges=False,
                                gpu_runtime=False)["linux-x64"])
        edge_compare(self, got, edge_cpu(lib))


if __name__ == "__main__":
    unittest.main()
