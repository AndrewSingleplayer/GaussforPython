"""Differential fuzzer: random HA++ expressions, run on every back end, compared with Python.

Each generated program computes many random expressions over i32 / u32 / f32 inputs.
The same code runs:
  * natively on x86-64 (export fn, via ctypes)
  * on ARM64 (the phones' CPU) under qemu-aarch64
  * as WebAssembly (the web-wasm32 target, as browsers run it) in Node
  * on a Vulkan GPU (the kernel version, via the HA++ runtime)
  * as Metal source, compiled with clang++ and tests/metal_shim (emulation)
and every result is compared with an exact Python model of HA++ semantics.

    python3 tests/fuzz.py --programs 50 --seed 1       # standalone, longer runs
"""
import argparse
import ctypes
import math
from fractions import Fraction
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from happ.driver import build, compile_ir, load_program  # noqa: E402
from happ.gpu import generate_metal  # noqa: E402
from happ.llvm import LLVMGen  # noqa: E402
from happ.targets import Toolchain, target_info  # noqa: E402

I32, U32, F32 = "i32", "u32", "f32"
NVARS = 4
ROWS = 64
F = np.float32


def wrap(v, t):
    if t == I32:
        return ((int(v) + (1 << 31)) % (1 << 32)) - (1 << 31)
    return int(v) % (1 << 32)


def f32(x):
    return float(F(x))


class Expr:
    __slots__ = ("op", "t", "kids", "val")

    def __init__(self, op, t, kids=(), val=None):
        self.op, self.t, self.kids, self.val = op, t, list(kids), val


class Gen:
    INT_LITS = {I32: [0, 1, -1, 2, 3, -3, 7, 15, 31, 100, -100, 12345, 2147483647, -2147483648],
                U32: [0, 1, 2, 3, 7, 15, 31, 100, 255, 65535, 2147483648, 4294967295, 3000000000]}
    F_LITS = [0.0, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5, 3.0, 0.25, 10.0, -7.75, 100.0, 0.125]

    def __init__(self, rng, gpu_safe=True):
        self.r = rng
        self.gpu_safe = gpu_safe

    def leaf(self, t):
        r = self.r
        if r.random() < 0.6:
            return Expr("var", t, val=r.randrange(NVARS))
        if t == F32:
            return Expr("lit", t, val=r.choice(self.F_LITS))
        return Expr("lit", t, val=r.choice(self.INT_LITS[t]))

    def gen(self, t, depth):
        r = self.r
        if depth <= 0 or r.random() < 0.25:
            return self.leaf(t)
        d = depth - 1
        if t == F32:
            op = r.choice(["+", "-", "*", "/", "%", "neg", "min", "max", "abs", "floor", "ceil", "trunc", "round",
                           "fract", "clamp", "mix", "step", "sqrt", "select", "cast_i", "cast_u", "mod"])
        else:
            op = r.choice(["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "neg", "not", "min", "max", "abs",
                           "clamp", "select", "popcount", "clz", "ctz", "cast_f", "cast_other"]
                          + (["sign"] if t == I32 else []))
        if op in ("+", "-", "*", "&", "|", "^", "min", "max"):
            return Expr(op, t, [self.gen(t, d), self.gen(t, d)])
        if op in ("/", "%", "mod"):
            # integers: sometimes divide by 0 or -1 on purpose (defined in HA++, guarded on GPUs)
            raw = t != F32 and r.random() < 0.3
            return Expr(op, t, [self.gen(t, d), self.gen(t, d)], val="raw" if raw else None)
        if op in ("<<", ">>"):
            return Expr(op, t, [self.gen(t, d), self.gen(t, d)])
        if op in ("neg", "not", "abs", "floor", "ceil", "trunc", "round", "fract", "sqrt", "popcount", "clz",
                  "ctz", "sign"):
            return Expr(op, t, [self.gen(t, d)])
        if op == "clamp":
            lo, hi = sorted(r.sample(self.F_LITS if t == F32 else self.INT_LITS[t], 2))
            return Expr(op, t, [self.gen(t, d)], val=(lo, hi))
        if op == "mix":
            return Expr(op, t, [self.gen(t, d), self.gen(t, d)], val=r.choice([0.25, 0.5, 0.75]))
        if op == "step":
            return Expr(op, t, [self.gen(t, d), self.gen(t, d)])
        if op == "select":
            return Expr(op, t, [self.gen_bool(d), self.gen(t, d), self.gen(t, d)])
        if op.startswith("cast"):
            src = {"cast_i": I32, "cast_u": U32, "cast_f": F32, "cast_other": U32 if t == I32 else I32}[op]
            kid = self.gen(src, d)
            if not has_var(kid):      # a literal-only operand would take a default type instead of src
                kid = Expr("var", src, val=r.randrange(NVARS))
            return Expr("cast", t, [kid])
        raise AssertionError(op)

    def gen_bool(self, depth):
        r = self.r
        if depth > 0 and r.random() < 0.3:
            op = r.choice(["&&", "||", "!"])
            if op == "!":
                return Expr("!", "bool", [self.gen_bool(depth - 1)])
            return Expr(op, "bool", [self.gen_bool(depth - 1), self.gen_bool(depth - 1)])
        t = r.choice([I32, U32, F32])
        left, right = self.gen(t, depth - 1), self.gen(t, depth - 1)
        if not has_var(left) and not has_var(right):
            left = Expr("var", t, val=r.randrange(NVARS))    # something must give the comparison its type
        return Expr(r.choice(["<", "<=", ">", ">=", "==", "!="]), "bool", [left, right])


def has_var(e):
    """True if a variable decides the type of e (select conditions don't count)."""
    if e.op == "var":
        return True
    kids = e.kids[1:] if e.op == "select" else e.kids
    return any(has_var(k) for k in kids)


def text(e):
    """HA++ source for an expression. Divisors / float->int casts are made safe here."""
    t, k = e.t, e.kids
    if e.op == "var":
        return {I32: "a", U32: "b", F32: "c"}[t] + str(e.val)
    if e.op == "lit":
        if t == F32:
            return repr(float(e.val)) if e.val >= 0 else f"({e.val!r})"
        return str(e.val) if e.val >= 0 else f"({e.val})"
    a = [text(x) for x in k]
    if e.op in ("+", "-", "*", "&", "|", "^", "<<", ">>", "<", "<=", ">", ">=", "==", "!=", "&&", "||"):
        return f"({a[0]} {e.op} {a[1]})"
    if e.op in ("/", "%"):
        if t == F32:
            return f"({a[0]} {e.op} (abs({a[1]}) + 1.0))"
        if e.val == "raw":
            return f"({a[0]} {e.op} (({a[1]} & 15) - 1))"
        return f"({a[0]} {e.op} (({a[1]} & 15) + 1))"
    if e.op == "mod":
        return f"mod({a[0]}, abs({a[1]}) + 1.0)"
    if e.op == "neg":
        if k[0].op == "lit":      # '-5' is folded into a literal, which must then fit the type
            return f"(0 - {a[0]})"
        return f"(-{a[0]})"
    if e.op == "not":
        return f"(~{a[0]})"
    if e.op == "!":
        return f"(!{a[0]})"
    if e.op in ("min", "max"):
        return f"{e.op}({a[0]}, {a[1]})"
    if e.op == "sqrt":
        return f"sqrt(abs({a[0]}))"
    if e.op in ("abs", "floor", "ceil", "trunc", "round", "fract", "popcount", "clz", "ctz", "sign"):
        return f"{e.op}({a[0]})"
    if e.op == "clamp":
        lo, hi = e.val
        fmt = (lambda v: repr(float(v)) if v >= 0 else f"({float(v)!r})") if t == F32 else \
            (lambda v: str(v) if v >= 0 else f"({v})")
        return f"clamp({a[0]}, {fmt(lo)}, {fmt(hi)})"
    if e.op == "mix":
        return f"mix({a[0]}, {a[1]}, {e.val!r})"
    if e.op == "step":
        return f"step({a[0]}, {a[1]})"
    if e.op == "select":
        return f"select({a[0]}, {a[1]}, {a[2]})"
    if e.op == "cast":
        src = k[0].t
        if src == F32:   # keep float->int in range: out-of-range is undefined on GPUs
            lim = "1.0e6" if t == I32 else "4.0e6"
            lo = "(-1.0e6)" if t == I32 else "0.0"
            return f"(clamp({a[0]}, {lo}, {lim}) as {t})"
        return f"({a[0]} as {t})"
    raise AssertionError(e.op)


class Unstable(Exception):
    """The exact result depends on rounding details HA++ leaves open (fast-math, FMA contraction,
    GPU division precision): e.g. floor() of a value that is within its rounding error of an integer."""


def ulp(x):
    return float(np.spacing(F(abs(x))))


def quantum(v):
    """Value of the lowest set bit of v (v = n * quantum, n odd); infinite for 0."""
    if v == 0:
        return math.inf
    fr = Fraction(v)
    n = abs(fr.numerator)
    return (n & -n) / fr.denominator


def reassoc_err(*vals):
    """Rounding a fast-math compiler may add by regrouping a sum or difference (reassoc): with
    (c - floor(c)) - m computed as c - (floor(c) + m), the intermediate floor(c) + m can round even
    when every step as written is exact. Sums of values on a common binary grid stay exact while
    their magnitude is below 2^24 grid steps; otherwise one rounding of the regrouped sum is possible."""
    total = sum(abs(v) for v in vals)
    q = min(quantum(v) for v in vals)
    return 0.0 if total < (1 << 24) * q else ulp(total)


class Eval:
    """Model of HA++ semantics. Every float value carries a bound on how far a correct fast-math back
    end may be from the model (fused multiply-add, x*(1/d) for x/d, GPU division ~2.5 ulp). A
    discontinuity (floor, cast, comparison...) closer than that bound raises Unstable, and the result
    is skipped instead of reported. Also tracks the largest float magnitude seen (for tolerances)."""

    def __init__(self, env):
        self.env = env
        self.scale = 1.0
        self.err = 0.0

    def f(self, x):
        x = f32(x)
        if not math.isfinite(x):
            raise Unstable("overflow")          # fast-math: infinities are not supported
        self.scale = max(self.scale, abs(x))
        return x

    def exact(self, fr):
        """Round an exact Fraction to f32: (value, rounding error or 0 if exact)."""
        r = self.f(float(fr))
        return r, (0.0 if Fraction(r) == fr else ulp(r))

    @staticmethod
    def cut(fn, x, ex):
        """fn(x) for a discontinuous fn, unless x is within its error bound of a jump."""
        if ex and fn(x - ex) != fn(x + ex):
            raise Unstable(fn)
        return fn(x)

    def ev(self, e):
        v, err = self.evf(e)
        self.err = err
        return v

    def evf(self, e):
        """(value, error bound). Integers and booleans are always exact (error 0)."""
        t, k, op = e.t, e.kids, e.op
        if op == "var":
            return self.env[t][e.val], 0.0
        if op == "lit":
            return (f32(e.val) if t == F32 else wrap(e.val, t)), 0.0
        if op in ("<", "<=", ">", ">=", "==", "!="):
            (x, ex), (y, ey) = self.evf(k[0]), self.evf(k[1])
            if ex + ey and abs(x - y) <= ex + ey:
                raise Unstable(op)
            return {"<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y, "==": x == y, "!=": x != y}[op], 0.0
        if op == "&&":
            return self.evf(k[0])[0] and self.evf(k[1])[0], 0.0
        if op == "||":
            return self.evf(k[0])[0] or self.evf(k[1])[0], 0.0
        if op == "!":
            return not self.evf(k[0])[0], 0.0
        if op == "select":
            return self.evf(k[1] if self.evf(k[0])[0] else k[2])
        if op == "cast":
            v, ev = self.evf(k[0])
            src = k[0].t
            if src == F32:
                lo, hi = (-1.0e6, 1.0e6) if t == I32 else (0.0, 4.0e6)
                return self.cut(lambda x: int(math.trunc(min(max(x, f32(lo)), f32(hi)))), v, ev), 0.0
            if t == F32:
                return self.f(v), 0.0              # int -> float rounds the same everywhere
            return wrap(v, t), 0.0
        if t == F32:
            return self.ev_float(e)
        return self.ev_int(e), 0.0

    def divisor(self, e):
        """abs(y) + 1.0, the (always >= 1) divisor the generated code uses."""
        y, ey = self.evf(e)
        d, rd = self.exact(Fraction(abs(y)) + 1)
        return d, ey + rd

    def ev_float(self, e):
        op, k = e.op, e.kids
        if op in ("/", "%", "mod"):
            x, ex = self.evf(k[0])
            d, ed = self.divisor(k[1])
            q = self.f(F(x) / F(d))
            eq = (ex + abs(q) * ed) / d + 3 * ulp(q)      # x*(1/d) or GPU division: a few ulp, always
            if op == "/":
                return q, eq
            base = self.cut(math.trunc if op == "%" else math.floor, q, eq)
            p, rp = self.exact(Fraction(d) * base)
            r, rr = self.exact(Fraction(x) - Fraction(p))
            # a fused multiply-add skips the rounding of d*base
            return r, ex + abs(base) * ed + rp + rr
        if op in ("+", "-", "*", "min", "max", "mix", "step"):
            (x, ex), (y, ey) = self.evf(k[0]), self.evf(k[1])
            fx, fy = Fraction(x), Fraction(y)
            if op == "+":
                r, rr = self.exact(fx + fy)
                return r, ex + ey + max(rr, reassoc_err(x, y))
            if op == "-":
                r, rr = self.exact(fx - fy)
                return r, ex + ey + max(rr, reassoc_err(x, y))
            if op == "*":
                r, rr = self.exact(fx * fy)
                return r, abs(x) * ey + abs(y) * ex + ex * ey + rr
            if op == "min":
                return min(x, y), max(ex, ey)
            if op == "max":
                return max(x, y), max(ex, ey)
            if op == "mix":
                a = Fraction(e.val)
                d1, r1 = self.exact(fy - fx)
                p1, r2 = self.exact(Fraction(d1) * a)
                r, r3 = self.exact(fx + Fraction(p1))
                # GLSL/Metal may use x*(1-a) + y*a instead; exact inputs give the same result either way
                u, r4 = self.exact(fx * (1 - a))
                w, r5 = self.exact(fy * a)
                _, r6 = self.exact(Fraction(u) + Fraction(w))
                rounding = 4 * ulp(max(abs(x), abs(y), abs(r))) if (r1 or r2 or r3 or r4 or r5 or r6) else 0.0
                return r, ex + ey + rounding
            # step(edge, x): 0 if x < edge else 1
            if ex + ey and abs(y - x) <= ex + ey:
                raise Unstable(op)
            return (0.0 if y < x else 1.0), 0.0
        x, ex = self.evf(k[0])
        if op == "neg":
            return -x, ex
        if op == "abs":
            return abs(x), ex
        if op == "floor":
            return float(self.cut(math.floor, x, ex)), 0.0
        if op == "ceil":
            return float(self.cut(math.ceil, x, ex)), 0.0
        if op == "trunc":
            return float(self.cut(math.trunc, x, ex)), 0.0
        if op == "round":
            def rnd(v):
                t = math.trunc(v)
                return t + (math.copysign(1, v) if abs(v - t) >= 0.5 else 0)
            return float(self.cut(rnd, x, ex)), 0.0
        if op == "fract":
            fl = self.cut(math.floor, x, ex)
            r, rr = self.exact(Fraction(x) - fl)
            return r, ex + max(rr, reassoc_err(x, float(fl)))
        if op == "clamp":
            lo, hi = e.val
            return min(max(x, f32(lo)), f32(hi)), ex
        if op == "sqrt":
            a = abs(x)
            r = self.f(math.sqrt(a))
            prop = math.sqrt(a + ex) - math.sqrt(max(a - ex, 0.0)) if ex else 0.0
            return r, prop + (3 * ulp(r) if Fraction(r) ** 2 != Fraction(a) else 0.0)
        raise AssertionError(op)

    def ev_int(self, e):
        op, k, t = e.op, e.kids, e.t
        x = self.evf(k[0])[0] if k else None
        y = self.evf(k[1])[0] if len(k) > 1 else None
        bits = 32
        if op == "+":
            return wrap(x + y, t)
        if op == "-":
            return wrap(x - y, t)
        if op == "*":
            return wrap(x * y, t)
        if op in ("/", "%") and e.val == "raw":
            d = wrap((y & 15) - 1, t)             # x/0 = 0, x%0 = x, MIN/-1 = MIN, x%-1 = 0
            if d == 0:
                return 0 if op == "/" else x
            if d == -1:
                return wrap(-x, t) if op == "/" else 0
            q = abs(x) // abs(d) * (1 if (x >= 0) == (d > 0) else -1)
            return wrap(q if op == "/" else x - d * q, t)
        if op in ("/", "%"):
            d = (y & 15) + 1 if t == U32 else (wrap(y & 15, t) + 1)
            q = abs(x) // d * (1 if x >= 0 else -1)
            return wrap(q if op == "/" else x - d * q, t)
        if op == "&":
            return wrap(x & y, t)
        if op == "|":
            return wrap(x | y, t)
        if op == "^":
            return wrap(x ^ y, t)
        if op == "<<":
            return wrap(x << (y & (bits - 1)), t)
        if op == ">>":
            return wrap(x >> (y & (bits - 1)), t)      # python >> is arithmetic for negatives
        if op == "neg":
            return wrap(-x, t)
        if op == "not":
            return wrap(~x, t)
        if op == "min":
            return min(x, y)
        if op == "max":
            return max(x, y)
        if op == "abs":
            return wrap(abs(x), t)
        if op == "sign":
            return (x > 0) - (x < 0)
        if op == "clamp":
            lo, hi = e.val
            return min(max(x, lo), hi)
        if op == "popcount":
            return bin(x & 0xFFFFFFFF).count("1")
        if op == "clz":
            u = x & 0xFFFFFFFF
            return 32 - u.bit_length()
        if op == "ctz":
            u = x & 0xFFFFFFFF
            return 32 if u == 0 else (u & -u).bit_length() - 1
        raise AssertionError(op)


def make_program(rng, n_expr=24, depth=4):
    g = Gen(rng)
    exprs = [(t, g.gen(t, depth)) for t in (I32, U32, F32) for _ in range(n_expr)]
    lines = []
    ni = sum(1 for t, _ in exprs if t == I32)
    nu = sum(1 for t, _ in exprs if t == U32)
    nf = sum(1 for t, _ in exprs if t == F32)
    body = []
    for v in range(NVARS):
        body.append(f"    let a{v} = ia[r * {NVARS} + {v}];")
        body.append(f"    let b{v} = ua[r * {NVARS} + {v}];")
        body.append(f"    let c{v} = fa[r * {NVARS} + {v}];")
    counters = {I32: 0, U32: 0, F32: 0}
    outs = {I32: ("oi", ni), U32: ("ou", nu), F32: ("of", nf)}
    for t, e in exprs:
        name, count = outs[t]
        body.append(f"    {name}[r * {count} + {counters[t]}] = {text(e)};")
        counters[t] += 1
    sig = "ia: *i32, ua: *u32, fa: *f32, oi: *i32, ou: *u32, of: *f32"
    lines.append(f"fn row(r: u32, {sig}) {{")
    lines += body
    lines.append("}")
    lines.append("")
    lines.append("@workgroup(64)")
    lines.append(f"kernel fz({sig}, n: u32) {{")
    lines.append("    let r = global_id.x;")
    lines.append("    if r >= n { return; }")
    lines += [x for x in body]
    lines.append("}")
    lines.append("")
    # byte-buffer entry point for the native and ARM64 runs: [n][ia][ua][fa] -> [oi][ou][of]
    lines.append("export fn test_main(inp: *u8, out: *u8) -> i64 {")
    lines.append("    let n = (inp as *u32)[0];")
    lines.append(f"    let ia = (inp + 16) as *i32;")
    lines.append(f"    let ua = (inp + 16 + (n * {4 * NVARS}) as i64) as *u32;")
    lines.append(f"    let fa = (inp + 16 + (n * {8 * NVARS}) as i64) as *f32;")
    lines.append("    let oi = out as *i32;")
    lines.append(f"    let ou = (out + (n * {4 * ni}) as i64) as *u32;")
    lines.append(f"    let of = (out + (n * {4 * (ni + nu)}) as i64) as *f32;")
    lines.append("    for r in 0..n { row(r, ia, ua, fa, oi, ou, of); }")
    lines.append(f"    return (n * {4 * (ni + nu + nf)}) as i64;")
    lines.append("}")
    return "\n".join(lines) + "\n", exprs, (ni, nu, nf)


def make_inputs(rng):
    ia = np.array([rng.choice(Gen.INT_LITS[I32]) if rng.random() < 0.4 else rng.randrange(-2 ** 31, 2 ** 31)
                   for _ in range(ROWS * NVARS)], np.int64).astype(np.int32)
    ua = np.array([rng.choice(Gen.INT_LITS[U32]) if rng.random() < 0.4 else rng.randrange(0, 2 ** 32)
                   for _ in range(ROWS * NVARS)], np.uint64).astype(np.uint32)
    fa = np.array([rng.randrange(-512, 512) / 8.0 if rng.random() < 0.7 else rng.uniform(-100, 100)
                   for _ in range(ROWS * NVARS)], np.float32)
    return ia, ua, fa


def reference(exprs, ia, ua, fa):
    """Model results per type, the float tolerance scale, and a mask of results to skip (Unstable)."""
    res = {I32: [], U32: [], F32: []}
    skip = {I32: [], U32: [], F32: []}
    scales, errs = [], []
    for r in range(ROWS):
        env = {I32: [int(v) for v in ia[r * NVARS:(r + 1) * NVARS]],
               U32: [int(v) for v in ua[r * NVARS:(r + 1) * NVARS]],
               F32: [float(v) for v in fa[r * NVARS:(r + 1) * NVARS]]}
        for t, e in exprs:
            ev = Eval(env)
            try:
                v, bad = ev.ev(e), False
            except Unstable:
                v, bad = 0, True
            res[t].append(v)
            skip[t].append(bad)
            if t == F32:
                scales.append(ev.scale)
                errs.append(ev.err)
    counts = {t: sum(1 for tt, _ in exprs if tt == t) for t in (I32, U32, F32)}
    shape = {t: (ROWS, counts[t]) for t in counts}
    return (np.array(res[I32], np.int64).reshape(shape[I32]),
            np.array(res[U32], np.int64).reshape(shape[U32]),
            np.array(res[F32], np.float64).reshape(shape[F32]),
            np.maximum(2e-5 * (1.0 + np.array(scales)), 2 * np.array(errs)).reshape(shape[F32]),
            {t: np.array(skip[t], bool).reshape(shape[t]) for t in skip})


def input_blob(ia, ua, fa):
    return struct.pack("<I12x", ROWS) + ia.tobytes() + ua.tobytes() + fa.tobytes()


# Runs a web-wasm32 build of the fuzz program: input blob on stdin, result bytes on stdout.
WASM_RUNNER = """import fs from "node:fs";
const [wasmPath, outBytes] = process.argv.slice(2);
const input = fs.readFileSync(0);
const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath), { env: {} });
const e = instance.exports, mem = e.memory;
const base = Math.ceil(Number(e.__heap_base.value) / 16) * 16;
const outPtr = base + Math.ceil(input.length / 16) * 16;
const need = outPtr + Number(outBytes) + 64;
if (need > mem.buffer.byteLength) mem.grow(Math.ceil((need - mem.buffer.byteLength) / 65536));
new Uint8Array(mem.buffer, base, input.length).set(input);
const n = Number(e.test_main(base, outPtr));
process.stdout.write(new Uint8Array(mem.buffer, outPtr, n));
"""


def split_output(buf, counts):
    ni, nu, nf = counts
    o = np.frombuffer(buf, np.uint8)
    oi = o[:ROWS * ni * 4].view(np.int32).reshape(ROWS, ni)
    ou = o[ROWS * ni * 4:ROWS * (ni + nu) * 4].view(np.uint32).reshape(ROWS, nu)
    of = o[ROWS * (ni + nu) * 4:ROWS * (ni + nu + nf) * 4].view(np.float32).reshape(ROWS, nf)
    return oi, ou, of


class Runner:
    def __init__(self, work, use_arm=True, use_gpu=True, use_metal=True, use_wasm=True):
        self.work = work
        self.tc = Toolchain()
        self.use_arm = use_arm and shutil.which("qemu-aarch64") and self.tc.find("ld.lld")
        self.use_wasm = use_wasm and shutil.which("node") and self.tc.find("wasm-ld")
        self.use_gpu = use_gpu and self.tc.find("glslangValidator")
        self.use_metal = use_metal and shutil.which("clang++")
        self.gpu = None

    def run_native(self, src, blob, out_bytes):
        self.count = getattr(self, "count", 0) + 1       # new folder: dlopen caches libraries by path
        lib = build(src, ["linux-x64"], os.path.join(self.work, f"nat{self.count}"), quiet=True,
                    bridges=False, gpu_runtime=bool(self.use_gpu))["linux-x64"]
        dll = ctypes.CDLL(lib)
        dll.test_main.restype = ctypes.c_int64
        dll.test_main.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        out = ctypes.create_string_buffer(out_bytes)
        n = dll.test_main(blob, out)
        return out.raw[:n], dll

    def run_arm(self, src, blob):
        tinfo = target_info("linux-arm64")
        prog = load_program(src)
        ir = LLVMGen(prog, tinfo, mode="lib").generate(["test_main"])
        obj = compile_ir(self.tc, ir, tinfo, os.path.join(self.work, "fz-arm64.o"))
        hobj = os.path.join(self.work, "harness-arm64.o")
        if not os.path.exists(hobj):
            subprocess.run([self.tc.find("clang"), "--target=aarch64-linux-gnu", "-O2", "-ffreestanding",
                            "-fno-stack-protector", "-c", os.path.join(ROOT, "tests", "arm64_harness.c"),
                            "-o", hobj], check=True)
        exe = os.path.join(self.work, "fz-arm64")
        subprocess.run([self.tc.find("ld.lld"), "-static", "-e", "_start", hobj, obj, "-o", exe], check=True)
        return subprocess.run(["qemu-aarch64", exe], input=blob, capture_output=True, check=True).stdout

    def run_wasm(self, src, blob, out_bytes):
        self.wcount = getattr(self, "wcount", 0) + 1
        wasm = build(src, ["web-wasm32"], os.path.join(self.work, f"wasm{self.wcount}"), quiet=True,
                     bridges=False)["web-wasm32"]
        js = os.path.join(self.work, "run_wasm.mjs")
        if not os.path.exists(js):
            with open(js, "w") as f:
                f.write(WASM_RUNNER)
        return subprocess.run(["node", js, wasm, str(out_bytes)], input=blob, capture_output=True,
                              check=True).stdout

    def run_gpu(self, dll, arrays, counts):
        P, U32_ = ctypes.c_void_p, ctypes.c_uint32
        for f, r, a in [("ha_gpu_create", P, [ctypes.c_char_p, ctypes.c_size_t]),
                        ("ha_buffer_create", P, [P, ctypes.c_size_t]), ("ha_buffer_data", P, [P]),
                        ("ha_buffer_destroy", None, [P]), ("ha_kernel_create", P, [P, P, ctypes.c_size_t, U32_, U32_]),
                        ("ha_kernel_destroy", None, [P]), ("ha_gpu_destroy", None, [P]),
                        ("ha_dispatch", ctypes.c_int, [P, P, P, P, U32_, U32_, U32_])]:
            fn = getattr(dll, f)
            fn.restype, fn.argtypes = r, a
        gpu = dll.ha_gpu_create(None, 0)
        if not gpu:
            return None
        getter = dll.fz_spirv_fz
        getter.restype, getter.argtypes = P, [P]
        size = ctypes.c_uint64()
        blob = getter(ctypes.byref(size))
        k = dll.ha_kernel_create(gpu, blob, size.value, 6, 4)
        ni, nu, nf = counts
        outs = [np.zeros(ROWS * ni, np.int32), np.zeros(ROWS * nu, np.uint32), np.zeros(ROWS * nf, np.float32)]
        bufs = []
        for a in list(arrays) + outs:
            b = dll.ha_buffer_create(gpu, max(a.nbytes, 4))
            ctypes.memmove(dll.ha_buffer_data(b), a.ctypes.data, a.nbytes)
            bufs.append(b)
        rc = dll.ha_dispatch(gpu, k, (P * 6)(*bufs), struct.pack("<I", ROWS), 1, 1, 1)
        assert rc == 0, rc
        res = []
        for b, a in zip(bufs[3:], outs):
            res.append(np.frombuffer(ctypes.string_at(dll.ha_buffer_data(b), a.nbytes), a.dtype).copy())
        for b in bufs:
            dll.ha_buffer_destroy(b)
        dll.ha_kernel_destroy(k)
        dll.ha_gpu_destroy(gpu)
        return res[0].reshape(ROWS, ni), res[1].reshape(ROWS, nu), res[2].reshape(ROWS, nf)

    def run_metal(self, src, arrays, counts):
        prog = load_program(src)
        mpath = os.path.join(self.work, "fz.metal")
        with open(mpath, "w") as f:
            f.write(generate_metal(prog))
        ni, nu, nf = counts
        drv = os.path.join(self.work, "mdrv.cpp")
        with open(drv, "w") as f:
            f.write(f"""#include "fz.metal"
#include <cstdio>
#include <vector>
int main() {{
    std::vector<int> ia({ROWS * NVARS}); std::vector<unsigned> ua({ROWS * NVARS}); std::vector<float> fa({ROWS * NVARS});
    std::vector<int> oi({max(1, ROWS * ni)}); std::vector<unsigned> ou({max(1, ROWS * nu)});
    std::vector<float> of({max(1, ROWS * nf)});
    FILE *f = fopen("fz_in.bin", "rb");
    fread(ia.data(), 4, ia.size(), f); fread(ua.data(), 4, ua.size(), f); fread(fa.data(), 4, fa.size(), f);
    fclose(f);
    ha_params_fz p{{{ROWS}u}};
    for (unsigned x = 0; x < 64; x++)
        fz(ia.data(), ua.data(), fa.data(), oi.data(), ou.data(), of.data(), p, metal::uint3{{x, 0, 0}},
           metal::uint3{{x, 0, 0}}, metal::uint3{{0, 0, 0}}, metal::uint3{{1, 1, 1}}, metal::uint3{{64, 1, 1}});
    f = fopen("fz_out.bin", "wb");
    fwrite(oi.data(), 4, {ROWS * ni}, f); fwrite(ou.data(), 4, {ROWS * nu}, f); fwrite(of.data(), 4, {ROWS * nf}, f);
    fclose(f);
}}
""")
        with open(os.path.join(self.work, "fz_in.bin"), "wb") as f:
            for a in arrays:
                f.write(a.tobytes())
        exe = os.path.join(self.work, "mdrv")
        r = subprocess.run(["clang++", "-std=c++17", "-O1", "-w", "-I", os.path.join(ROOT, "tests", "metal_shim"),
                            drv, "-o", exe], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Metal code did not compile:\n" + r.stderr[:3000])
        subprocess.run([exe], check=True, cwd=self.work)
        with open(os.path.join(self.work, "fz_out.bin"), "rb") as f:
            return split_output(f.read(), counts)


def compare(name, got, ref, exprs, counts):
    """Return a list of mismatch descriptions."""
    oi, ou, of = got
    ri, ru, rf, tol, skip = ref
    bad = []
    by_type = {I32: [e for t, e in exprs if t == I32], U32: [e for t, e in exprs if t == U32],
               F32: [e for t, e in exprs if t == F32]}
    for (g, rr, t) in ((oi, ri, I32), (ou, ru, U32)):
        if rr.size == 0:
            continue
        g64 = g.astype(np.int64)
        rr = np.array([[wrap(v, t) for v in row] for row in rr], np.int64)
        for row, col in zip(*np.nonzero((g64 != rr) & ~skip[t])):
            bad.append(f"{name} {t} expr#{col} row {row}: got {g64[row, col]}, want {rr[row, col]}: "
                       f"{text(by_type[t][col])}")
    if rf.size:
        diff = np.abs(of.astype(np.float64) - rf)
        for row, col in zip(*np.nonzero(~(diff <= tol) & ~skip[F32])):
            bad.append(f"{name} f32 expr#{col} row {row}: got {of[row, col]!r}, want {rf[row, col]!r}: "
                       f"{text(by_type[F32][col])}")
    return bad


def run_one(seed, runner, n_expr=24, depth=4):
    rng = random.Random(seed)
    src_text, exprs, counts = make_program(rng, n_expr, depth)
    src = os.path.join(runner.work, "fz.ha")
    with open(src, "w") as f:
        f.write(src_text)
    ia, ua, fa = make_inputs(rng)
    ref = reference(exprs, ia, ua, fa)
    blob = input_blob(ia, ua, fa)
    out_bytes = ROWS * 4 * sum(counts)
    bad = []
    raw, dll = runner.run_native(src, blob, out_bytes)
    bad += compare("x86-64", split_output(raw, counts), ref, exprs, counts)
    if runner.use_arm:
        bad += compare("arm64", split_output(runner.run_arm(src, blob), counts), ref, exprs, counts)
    if runner.use_wasm:
        bad += compare("wasm", split_output(runner.run_wasm(src, blob, out_bytes), counts), ref, exprs, counts)
    if runner.use_gpu:
        g = runner.run_gpu(dll, (ia, ua, fa), counts)
        if g is not None:
            bad += compare("vulkan", g, ref, exprs, counts)
    if runner.use_metal:
        bad += compare("metal-emu", runner.run_metal(src, (ia, ua, fa), counts), ref, exprs, counts)
    skipped = sum(int(m.sum()) for m in ref[4].values())
    return bad, src_text, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exprs", type=int, default=24)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep the work folder of the last program")
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="happ-fuzz-")
    runner = Runner(work)
    print(f"back ends: x86-64{' arm64' if runner.use_arm else ''}{' wasm' if runner.use_wasm else ''}"
          f"{' vulkan' if runner.use_gpu else ''}"
          f"{' metal-emu' if runner.use_metal else ''}")
    total_bad = total_skipped = 0
    for i in range(args.programs):
        seed = args.seed + i
        bad, src, skipped = run_one(seed, runner, args.exprs, args.depth)
        total_bad += len(bad)
        total_skipped += skipped
        status = "ok" if not bad else f"{len(bad)} MISMATCHES"
        note = f" ({skipped} rounding-sensitive results skipped)" if skipped else ""
        print(f"program {i + 1}/{args.programs} (seed {seed}): {3 * args.exprs} expressions x {ROWS} rows: "
              f"{status}{note}")
        for b in bad[:8]:
            print("   ", b)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    checked = args.programs * 3 * args.exprs * ROWS
    print(f"total mismatches: {total_bad} (checked {checked - total_skipped} results; skipped "
          f"{total_skipped} whose exact value depends on allowed fast-math rounding)")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
