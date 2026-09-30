"""CPU back end: checked HA++ program -> LLVM IR text.

Design rules that keep the output fast and dependency-free:
  * every local lives in an entry-block alloca; LLVM's mem2reg/SROA turn them
    into registers, then the -O3 pipeline vectorizes loops for NEON/AVX;
  * float math uses fast-math flags by default (reassoc/contract/...) so
    reductions vectorize and multiply-adds fuse (turn off with @strict);
  * no libc: struct copies use llvm.memcpy.inline, math functions come from
    the HA++ standard library, and functions carry "no-builtins" so LLVM never
    invents memset/memcpy calls. A library has no dependencies at all.
"""

import struct

from . import ast as A
from . import types as T
from .checker import STD_IMPL
from .errors import HappError

FAST_FLAGS = "reassoc nsz arcp contract afn"

def jni_mangle(s):
    out = []
    for ch in s:
        if ch.isascii() and ch.isalnum():
            out.append(ch)
        elif ch == "_":
            out.append("_1")
        elif ch == ";":
            out.append("_2")
        elif ch == "[":
            out.append("_3")
        elif ch in "./":
            out.append("_")
        else:
            out.append("_0%04x" % ord(ch))
    return "".join(out)


class V:
    """A value: LLVM operand text plus its HA++ type.

    Structs and arrays are 'in memory': `ref` is then a pointer to them."""
    __slots__ = ("ty", "ref", "fresh")

    def __init__(self, ty, ref, fresh=False):
        self.ty, self.ref = ty, ref
        self.fresh = fresh      # a new temporary that nothing else points to


class FnState:
    def __init__(self):
        self.lines = []
        self.allocas = []
        self.tmp = 0
        self.labels = 0
        self.terminated = False
        self.vars = {}
        self.loops = []
        self.fn = None
        self.sret = None
        self.fmf = FAST_FLAGS
        self.kernel_vars = {}
        self.kernel_return = None


class LLVMGen:
    def __init__(self, prog, target, mode="lib", print_enabled=False, jni=None,
                 lib_name="happ", gpu_blobs=None, strict_math=False, f16_helpers=False,
                 gpu_runtime=False):
        self.prog = prog
        self.target = target            # dict from targets.TARGETS
        self.mode = mode                # 'lib' (shared/static library) or 'exe' (happ run)
        self.print_enabled = print_enabled
        self.jni = jni                  # (package, class) or None
        self.lib_name = lib_name
        self.gpu_blobs = gpu_blobs or {}
        self.strict = strict_math
        self.f16_helpers = f16_helpers
        self.gpu_runtime = gpu_runtime
        self.need_f64_to_f16 = False
        self.decls = {}
        self.globals = []
        self.strings = {}
        self.attr_groups = {}
        self.out = []
        self.s = None
        self.warnings = []
        self.is_windows = target.get("os") == "windows"
        self.is_arm64 = target.get("arch") == "arm64"
        # wasm32 addresses are 4 bytes, but HA++ lays every pointer out as 8 bytes on every target:
        # there, pointers in memory are stored as i64 and converted on load/store
        self.ptr64mem = target.get("arch") == "wasm32"

    # ================================================================ types
    def ll(self, t):
        if isinstance(t, T.Scalar):
            if t.cls == "bool":
                return "i1"
            if t.cls == "float":
                return {16: "half", 32: "float", 64: "double"}[t.bits]
            return f"i{t.bits}"
        if isinstance(t, T.Vec):
            return f"<{t.n} x {self.ll(t.elem)}>"
        if isinstance(t, T.Mat):
            return f"[{t.n} x <{t.n} x float>]"
        if isinstance(t, (T.Ptr, T.Struct, T.Array)):
            return "ptr"
        if isinstance(t, T.Void):
            return "void"
        raise HappError(f"internal: no LLVM type for {t}")

    def llm(self, t):
        """Memory (storage) type: C layout, packed vectors."""
        if isinstance(t, T.Ptr) and self.ptr64mem:
            return "i64"
        if isinstance(t, T.Scalar) and t.cls == "bool":
            return "i8"
        if isinstance(t, T.Vec):
            return f"[{t.n} x {self.ll(t.elem)}]"
        if isinstance(t, T.Mat):
            return f"[{t.n * t.n} x float]"
        if isinstance(t, T.Struct):
            return f'%"S.{t.name}"'
        if isinstance(t, T.Array):
            return f"[{t.n} x {self.llm(t.elem)}]"
        return self.ll(t)

    @staticmethod
    def align(t):
        return T.size_align(t)[1]

    @staticmethod
    def ovl(t):
        """Intrinsic overload suffix, e.g. f32 / v4f32 / i64."""
        def sc(s):
            return ("f" if s.cls == "float" else "i") + str(s.bits)
        if isinstance(t, T.Vec):
            return f"v{t.n}{sc(t.elem)}"
        return sc(t)

    # ================================================================ constants
    def fconst(self, value, t):
        try:
            v = float(value)
        except OverflowError:
            v = float("inf") if value > 0 else float("-inf")
        if t.bits == 16:
            try:
                bits = struct.unpack("<H", struct.pack("<e", v))[0]
            except OverflowError:
                bits = 0x7C00 if v > 0 else 0xFC00
            return "0xH%04X" % bits
        if t.bits == 32:
            try:
                v = struct.unpack("<f", struct.pack("<f", v))[0]
            except (OverflowError, struct.error):
                v = float("inf") if v > 0 else float("-inf")
        return "0x%016X" % struct.unpack("<Q", struct.pack("<d", v))[0]

    def iconst(self, value, t):
        if t.cls == "bool":
            return "true" if value else "false"
        value = int(value) & ((1 << t.bits) - 1)
        if value >= 1 << (t.bits - 1):
            value -= 1 << t.bits
        return str(value)

    def const(self, value, t):
        if t.is_float:
            return V(t, self.fconst(value, t))
        return V(t, self.iconst(value, t))

    def scalar_const_text(self, value, s):
        return self.fconst(value, s) if s.is_float else self.iconst(value, s)

    def splat_const(self, value, t):
        """Constant of scalar type or broadcast over a vector type."""
        if t.is_vec:
            c = self.scalar_const_text(value, t.elem)
            lt = self.ll(t.elem)
            return V(t, "<" + ", ".join(f"{lt} {c}" for _ in range(t.n)) + ">")
        return self.const(value, t)

    # ================================================================ emission helpers
    def decl(self, name, text):
        self.decls[name] = text

    def intrinsic(self, name, ret, args):
        self.decl(name, f"declare {ret} @{name}({', '.join(args)})")
        return name

    def tmp(self):
        self.s.tmp += 1
        return f"%t{self.s.tmp}"

    def label(self, hint):
        self.s.labels += 1
        return f"{hint}{self.s.labels}"

    def emit(self, text):
        if self.s.terminated:
            self.start(self.label("dead"))
        self.s.lines.append("  " + text)

    def inst(self, text):
        t = self.tmp()
        self.emit(f"{t} = {text}")
        return t

    def start(self, label):
        self.s.lines.append(f"{label}:")
        self.s.terminated = False

    def br(self, label):
        self.emit(f"br label %{label}")
        self.s.terminated = True

    def cbr(self, cond, a, b):
        self.emit(f"br i1 {cond}, label %{a}, label %{b}")
        self.s.terminated = True

    def alloca(self, t, name="v"):
        self.s.tmp += 1
        ref = f"%{name}.{self.s.tmp}"
        self.s.allocas.append(f"  {ref} = alloca {self.llm(t)}, align {max(self.align(t), 1)}")
        return ref

    def fm(self):
        return self.s.fmf + " " if self.s.fmf else ""

    def fm_exact_div(self):
        """Fast-math flags without 'arcp': x/y must be exact where floor/trunc follows (mod, %)."""
        flags = [f for f in self.s.fmf.split() if f not in ("arcp", "afn")]
        return " ".join(flags) + " " if flags else ""

    INLINE_COPY_LIMIT = 512

    def memcpy(self, dst, src, t):
        size = T.size_align(t)[0]
        self.byte_loop(dst, src, size, self.align(t))

    def memzero(self, dst, t):
        size = T.size_align(t)[0]
        self.byte_loop(dst, None, size, self.align(t))

    def byte_loop(self, dst, src, size, align):
        """Copy (src) or zero (src=None) `size` bytes. Small: inline intrinsic. Large: a loop over 8-byte
        words that LLVM vectorizes (never a call to the C library, never megabytes of straight-line code)."""
        words = size // 8 if size > self.INLINE_COPY_LIMIT else 0
        if words:
            slot = self.alloca(T.I64, "i")
            self.emit(f"store i64 0, ptr {slot}, align 8")
            head, body, done = self.label("cpy"), self.label("cpyb"), self.label("cpye")
            self.br(head)
            self.start(head)
            i = self.inst(f"load i64, ptr {slot}, align 8")
            c = self.inst(f"icmp ult i64 {i}, {words}")
            self.cbr(c, body, done)
            self.start(body)
            dp = self.inst(f"getelementptr inbounds i64, ptr {dst}, i64 {i}")
            if src is None:
                self.emit(f"store i64 0, ptr {dp}, align {min(align, 8)}")
            else:
                sp = self.inst(f"getelementptr inbounds i64, ptr {src}, i64 {i}")
                w = self.inst(f"load i64, ptr {sp}, align {min(align, 8)}")
                self.emit(f"store i64 {w}, ptr {dp}, align {min(align, 8)}")
            n = self.inst(f"add i64 {i}, 1")
            self.emit(f"store i64 {n}, ptr {slot}, align 8")
            self.br(head)
            self.start(done)
        rest = size - words * 8
        if rest == 0:
            return
        off = words * 8
        d = self.inst(f"getelementptr inbounds i8, ptr {dst}, i64 {off}") if off else dst
        a = align if off == 0 else min(align, 8)
        if src is None:
            self.intrinsic("llvm.memset.inline.p0.i64", "void", ["ptr", "i8", "i64", "i1"])
            self.emit(f"call void @llvm.memset.inline.p0.i64(ptr align {a} {d}, i8 0, i64 {rest}, i1 false)")
        else:
            sp = self.inst(f"getelementptr inbounds i8, ptr {src}, i64 {off}") if off else src
            self.intrinsic("llvm.memcpy.inline.p0.p0.i64", "void", ["ptr", "ptr", "i64", "i1"])
            self.emit(f"call void @llvm.memcpy.inline.p0.p0.i64(ptr align {a} {d}, ptr align {a} {sp}, "
                      f"i64 {rest}, i1 false)")

    def string_global(self, text):
        if text in self.strings:
            return self.strings[text]
        data = text.encode("utf-8")
        name = f"@.str.{len(self.strings)}"
        enc = "".join(chr(b) if 32 <= b < 127 and b not in (34, 92) else "\\%02X" % b for b in data)
        self.globals.append(f'{name} = private unnamed_addr constant [{len(data)} x i8] c"{enc}", align 1')
        self.strings[text] = (name, len(data))
        return self.strings[text]

    # ================================================================ load/store
    def load(self, t, ptr):
        if t.in_memory:
            return V(t, ptr)
        if t.is_bool:
            b = self.inst(f"load i8, ptr {ptr}, align 1")
            return V(t, self.inst(f"trunc i8 {b} to i1"))
        if t.is_mat:
            agg = "poison"
            ct = self.ll(t.col)
            for i in range(t.n):
                p = self.inst(f"getelementptr inbounds float, ptr {ptr}, i64 {i * t.n}")
                c = self.inst(f"load {ct}, ptr {p}, align 4")
                agg = self.inst(f"insertvalue {self.ll(t)} {agg}, {ct} {c}, {i}")
            return V(t, agg)
        if t.is_ptr and self.ptr64mem:
            i = self.inst(f"load i64, ptr {ptr}, align 8")
            return V(t, self.inst(f"inttoptr i64 {i} to ptr"))
        return V(t, self.inst(f"load {self.ll(t)}, ptr {ptr}, align {self.align(t)}"))

    def store(self, v, ptr):
        t = v.ty
        if t.in_memory:
            if v.ref != ptr:
                self.memcpy(ptr, v.ref, t)
            return
        if t.is_bool:
            z = self.inst(f"zext i1 {v.ref} to i8")
            self.emit(f"store i8 {z}, ptr {ptr}, align 1")
            return
        if t.is_mat:
            ct = self.ll(t.col)
            for i in range(t.n):
                c = self.inst(f"extractvalue {self.ll(t)} {v.ref}, {i}")
                p = self.inst(f"getelementptr inbounds float, ptr {ptr}, i64 {i * t.n}")
                self.emit(f"store {ct} {c}, ptr {p}, align 4")
            return
        if t.is_ptr and self.ptr64mem:
            i = self.inst(f"ptrtoint ptr {v.ref} to i64")
            self.emit(f"store i64 {i}, ptr {ptr}, align 8")
            return
        self.emit(f"store {self.ll(t)} {v.ref}, ptr {ptr}, align {self.align(t)}")

    def zero_value(self, t):
        if t.is_bool:
            return V(t, "false")
        if t.is_scalar:
            return self.const(0, t)
        if t.is_vec or t.is_mat:
            return V(t, "zeroinitializer")
        if t.is_ptr:
            return V(t, "null")
        raise AssertionError(t)

    def spill(self, v):
        """Put a value in memory and return the pointer."""
        if v.ty.in_memory:
            return v.ref
        p = self.alloca(v.ty, "spill")
        self.store(v, p)
        return p

    # ================================================================ vectors
    def extract(self, v, i):
        return V(v.ty.elem, self.inst(f"extractelement {self.ll(v.ty)} {v.ref}, i64 {i}"))

    def build_vec(self, t, comps):
        cur = "poison"
        for i, c in enumerate(comps):
            cur = self.inst(f"insertelement {self.ll(t)} {cur}, {self.ll(t.elem)} {c.ref}, i64 {i}")
        return V(t, cur)

    def splat(self, s, t):
        if s.ref.startswith("0x") or s.ref.lstrip("-").isdigit() or s.ref in ("true", "false"):
            lt = self.ll(t.elem)
            return V(t, "<" + ", ".join(f"{lt} {s.ref}" for _ in range(t.n)) + ">")
        one = self.inst(f"insertelement <1 x {self.ll(t.elem)}> poison, {self.ll(t.elem)} {s.ref}, i64 0")
        mask = ", ".join("i32 0" for _ in range(t.n))
        return V(t, self.inst(f"shufflevector <1 x {self.ll(t.elem)}> {one}, <1 x {self.ll(t.elem)}> "
                              f"poison, <{t.n} x i32> <{mask}>"))

    def shuffle(self, v, lanes):
        t = T.Vec(v.ty.elem, len(lanes))
        mask = ", ".join(f"i32 {l}" for l in lanes)
        return V(t, self.inst(f"shufflevector {self.ll(v.ty)} {v.ref}, {self.ll(v.ty)} poison, "
                              f"<{len(lanes)} x i32> <{mask}>"))

    def broadcast_pair(self, a, b):
        if a.ty.is_vec and b.ty.is_scalar:
            b = self.splat(b, a.ty)
        elif b.ty.is_vec and a.ty.is_scalar:
            a = self.splat(a, b.ty)
        return a, b

    def lanes_of(self, v):
        if v.ty.is_vec:
            return [self.extract(v, i) for i in range(v.ty.n)]
        return [v]

    def map_lanes(self, t, fn, *vals):
        """Apply a scalar function lane by lane (for calls that have no vector form)."""
        if not t.is_vec:
            return fn(*vals)
        parts = [self.lanes_of(v) if v.ty.is_vec else [v] * t.n for v in vals]
        res = [fn(*[p[i] for p in parts]) for i in range(t.n)]
        return self.build_vec(T.Vec(res[0].ty, t.n), res)

    # ================================================================ module
    def attr_group(self, extra=()):
        attrs = ["nounwind", "uwtable", '"no-builtins"']
        if self.is_arm64:
            attrs.append('"frame-pointer"="non-leaf"')
        attrs.extend(extra)
        key = " ".join(attrs)
        if key not in self.attr_groups:
            self.attr_groups[key] = len(self.attr_groups)
        return f"#{self.attr_groups[key]}"

    def fn_symbol(self, fn):
        if fn.is_export or fn.is_extern:
            return f"@{fn.name}"
        return f'@"ha.{fn.name}"'

    def generate(self, roots):
        prog = self.prog
        for st in prog.structs:
            fields = ", ".join(self.llm(ft) for _, ft in st.fields)
            self.out.append(f'%"S.{st.name}" = type {{ {fields} }}')
        fns = prog.reachable(roots)
        body = []
        for fn in fns:
            if fn.is_extern:
                params = ", ".join(self.param_decl(p.ty) for p in fn.params)
                self.decl(fn.name, f"declare {self.ret_decl(fn)} @{fn.name}({params})")
                continue
            if fn.is_kernel:
                body.extend(self.gen_kernel_cpu(fn))
                continue
            body.extend(self.gen_fn(fn))
        if self.mode == "exe" and prog.main is not None:
            body.extend(self.gen_main_wrapper(prog.main))
        if self.jni:
            body.extend(self.gen_jni())
        body.extend(self.gen_gpu_blobs())
        body.extend(self.gen_runtime())
        lines = ["; generated by the HA++ compiler (happ)", ""]
        lines += self.out
        lines += [""] + self.globals + [""]
        lines += body
        lines += [""] + sorted(self.decls.values()) + [""]
        for key, idx in sorted(self.attr_groups.items(), key=lambda kv: kv[1]):
            lines.append(f"attributes #{idx} = {{ {key} }}")
        return "\n".join(lines) + "\n"

    def param_decl(self, t):
        """C ABI argument type: the caller extends small integers (required on x86-64 and Apple ARM64)."""
        lt = self.ll(t)
        if t.is_bool:
            return "i1 zeroext"
        if t.is_int and t.bits < 32:
            return f"{lt} {'signext' if t.is_signed else 'zeroext'}"
        return lt

    def ret_decl(self, fn):
        t = fn.ret_ty
        rt = self.ll(t)
        if fn.is_export or fn.is_extern:
            if t.is_bool:
                return "zeroext i1"
            if t.is_int and t.bits < 32:
                return ("signext " if t.is_signed else "zeroext ") + rt
        return rt

    # ================================================================ functions
    def gen_fn(self, fn, name_override=None, extra_params=(), kernel=None):
        self.s = s = FnState()
        s.fn = fn
        if fn.is_std:
            s.fmf = "contract"      # std math relies on exact operation order
        if self.strict or "strict" in fn.attrs:
            s.fmf = ""
        params = []
        if fn.ret_ty.in_memory:
            s.sret = "%ret"
            params.append("ptr noalias %ret")
        for p in fn.params:
            params.append(f"{self.ll(p.ty)} %p.{p.name}")
        params.extend(extra_params)
        ret_ll = "void" if fn.ret_ty.in_memory else self.ret_decl(fn)
        sym = name_override or self.fn_symbol(fn)
        extra = []
        if fn.is_std or "inline" in fn.attrs:
            extra.append("alwaysinline")
        if "noinline" in fn.attrs:
            extra.append("noinline")
        attrs = self.attr_group(extra)
        linkage = "" if (fn.is_export and not kernel) else "internal "
        dll = "dllexport " if (self.is_windows and fn.is_export and not kernel) else ""
        header = f"define {linkage}{dll}{ret_ll} {sym}({', '.join(params)}) {attrs} {{"
        self.start("entry")
        if kernel is not None:
            s.kernel_vars = kernel
        for p in fn.params:
            sym_obj = self.param_symbol(fn, p)
            if p.ty.in_memory:
                s.vars[sym_obj] = f"%p.{p.name}"
            else:
                slot = self.alloca(p.ty, p.name)
                self.store(V(p.ty, f"%p.{p.name}"), slot)
                s.vars[sym_obj] = slot
        self.block(fn.body)
        if not s.terminated:
            if fn.ret_ty.is_void or fn.ret_ty.in_memory:
                self.emit("ret void")
            else:
                self.emit("unreachable")
            s.terminated = True
        lines = [header, "entry:"] + s.allocas + s.lines[1:] + ["}", ""]
        self.s = None
        return lines

    @staticmethod
    def param_symbol(fn, p):
        return ("param", fn.name, p.name)

    # ================================================================ statements
    def block(self, b):
        for st in b.stmts:
            self.stmt(st)

    def stmt(self, st):
        s = self.s
        if isinstance(st, A.Let):
            slot = self.alloca(st.ty, st.name)
            s.vars[self.let_key(st)] = slot
            if st.value is None:
                if st.ty.in_memory:
                    self.memzero(slot, st.ty)
                else:
                    self.store(self.zero_value(st.ty), slot)
            else:
                self.store(self.expr(st.value), slot)
        elif isinstance(st, A.Assign):
            self.assign(st)
        elif isinstance(st, A.If):
            c = self.expr(st.cond)
            then_l, else_l, end_l = self.label("then"), self.label("else"), self.label("endif")
            self.cbr(c.ref, then_l, else_l if st.els is not None else end_l)
            self.start(then_l)
            self.block(st.then)
            self.br(end_l)
            if st.els is not None:
                self.start(else_l)
                if isinstance(st.els, A.If):
                    self.stmt(st.els)
                else:
                    self.block(st.els)
                self.br(end_l)
            self.start(end_l)
        elif isinstance(st, A.While):
            cond_l, body_l, end_l = self.label("while"), self.label("body"), self.label("wend")
            self.br(cond_l)
            self.start(cond_l)
            c = self.expr(st.cond)
            self.cbr(c.ref, body_l, end_l)
            self.start(body_l)
            s.loops.append((cond_l, end_l))
            self.block(st.body)
            s.loops.pop()
            self.br(cond_l)
            self.start(end_l)
        elif isinstance(st, A.For):
            t = st.ty
            slot = self.alloca(t, st.var)
            s.vars[self.let_key(st)] = slot
            start = self.expr(st.start)
            end = self.expr(st.end)
            end_slot = self.alloca(t, "end")
            self.store(end, end_slot)
            self.store(start, slot)
            cond_l, body_l, step_l, end_l = (self.label("for"), self.label("fbody"),
                                             self.label("fstep"), self.label("fend"))
            self.br(cond_l)
            self.start(cond_l)
            i = self.load(t, slot)
            e = self.load(t, end_slot)
            c = self.inst(f"icmp {'slt' if t.is_signed else 'ult'} {self.ll(t)} {i.ref}, {e.ref}")
            self.cbr(c, body_l, end_l)
            self.start(body_l)
            s.loops.append((step_l, end_l))
            self.block(st.body)
            s.loops.pop()
            self.br(step_l)
            self.start(step_l)
            i2 = self.load(t, slot)
            n = self.inst(f"add {self.ll(t)} {i2.ref}, 1")
            self.emit(f"store {self.ll(t)} {n}, ptr {slot}, align {self.align(t)}")
            self.br(cond_l)
            self.start(end_l)
        elif isinstance(st, A.Return):
            if s.kernel_return is not None:
                self.br(s.kernel_return)
                return
            if st.value is None:
                self.emit("ret void")
            elif s.sret is not None:
                v = self.expr(st.value)
                self.memcpy(s.sret, v.ref, v.ty)
                self.emit("ret void")
            else:
                v = self.expr(st.value)
                self.emit(f"ret {self.ll(v.ty)} {v.ref}")
            s.terminated = True
        elif isinstance(st, A.Break):
            self.br(s.loops[-1][1])
        elif isinstance(st, A.Continue):
            self.br(s.loops[-1][0])
        elif isinstance(st, A.ExprStmt):
            self.expr(st.expr)
        elif isinstance(st, A.Block):
            self.block(st)
        elif isinstance(st, A.Shared):
            raise HappError("internal: shared memory in CPU code", st.loc)
        else:
            raise AssertionError(st)

    @staticmethod
    def let_key(node):
        return ("let", id(node))

    def assign(self, st):
        tgt = st.target
        # multi-lane swizzle writes: v.xy = w
        if isinstance(tgt, A.Field) and tgt.kind == "swizzle" and len(tgt.info) > 1:
            base_ptr = self.addr(tgt.base)
            et = tgt.base.ty.elem
            if st.op == "=":
                val = self.expr(st.value)
            else:
                cur = self.shuffle(self.load(tgt.base.ty, base_ptr), tgt.info)     # base evaluated once
                val = self.arith(st.op[:-1], cur, self.expr(st.value), st.loc)
            for k, lane in enumerate(tgt.info):
                p = self.inst(f"getelementptr inbounds {self.ll(et)}, ptr {base_ptr}, i64 {lane}")
                self.store(self.extract(val, k), p)
            return
        ptr = self.addr(tgt)
        if st.op == "=":
            val = self.expr(st.value)
            if val.ty.in_memory and not val.fresh and self.through_pointer(tgt):
                tmp = self.alloca(val.ty, "ovl")          # source and target may overlap in memory
                self.memcpy(tmp, val.ref, val.ty)
                val = V(val.ty, tmp, fresh=True)
            self.store(val, ptr)
        else:
            cur = self.load(tgt.ty, ptr)
            rhs = self.expr(st.value)
            self.store(self.arith(st.op[:-1], cur, rhs, st.loc), ptr)

    @staticmethod
    def through_pointer(e):
        """Does this lvalue reach memory through a pointer (so it may alias other memory)?"""
        while True:
            if isinstance(e, A.Unary) and e.op == "*":
                return True
            if isinstance(e, A.Index):
                if e.base.ty.is_ptr:
                    return True
                e = e.base
            elif isinstance(e, A.Field):
                if e.kind == "ptrfield":
                    return True
                e = e.base
            else:
                return False

    # ================================================================ addresses
    def lookup_var(self, sym):
        s = self.s
        if sym.kind == "param":
            return s.vars[("param", s.fn.name, sym.name)]
        return s.vars[("let", id(sym.node))]

    def addr(self, e):
        """Pointer to an lvalue (or a temporary holding an rvalue)."""
        if isinstance(e, A.Name):
            sym = e.ref
            if sym.kind in ("local", "param", "shared"):
                return self.lookup_var(sym)
            return self.spill(self.expr(e))
        if isinstance(e, A.Index):
            bt = e.base.ty
            idx = self.index64(e.index)
            if bt.is_ptr:
                base = self.expr(e.base).ref
                return self.inst(f"getelementptr inbounds {self.llm(bt.pointee)}, ptr {base}, i64 {idx}")
            if bt.is_array:
                base = self.expr(e.base).ref
                return self.inst(f"getelementptr inbounds {self.llm(bt)}, ptr {base}, i64 0, i64 {idx}")
            if bt.is_vec:
                base = self.addr(e.base)
                return self.inst(f"getelementptr inbounds {self.ll(bt.elem)}, ptr {base}, i64 {idx}")
            if bt.is_mat:
                base = self.addr(e.base)
                off = self.inst(f"mul i64 {idx}, {bt.n}")
                return self.inst(f"getelementptr inbounds float, ptr {base}, i64 {off}")
        if isinstance(e, A.Field):
            if e.kind in ("field", "ptrfield"):
                st = e.base.ty.pointee if e.kind == "ptrfield" else e.base.ty
                base = self.expr(e.base).ref
                return self.inst(f'getelementptr inbounds {self.llm(st)}, ptr {base}, i32 0, i32 {e.info}')
            if e.kind == "swizzle" and len(e.info) == 1:
                base = self.addr(e.base)
                return self.inst(f"getelementptr inbounds {self.ll(e.base.ty.elem)}, ptr {base}, "
                                 f"i64 {e.info[0]}")
        if isinstance(e, A.Unary) and e.op == "*":
            return self.expr(e.operand).ref
        return self.spill(self.expr(e))

    def index64(self, e):
        v = self.expr(e)
        t = v.ty
        if t.bits == 64:
            return v.ref
        op = "sext" if t.is_signed else "zext"
        if v.ref.lstrip("-").isdigit():
            return v.ref if t.is_signed or not v.ref.startswith("-") else str(int(v.ref) & ((1 << t.bits) - 1))
        return self.inst(f"{op} {self.ll(t)} {v.ref} to i64")

    # ================================================================ expressions
    def expr(self, e):
        t = e.ty
        if isinstance(e, A.IntLit):
            return self.const(e.value, t)
        if isinstance(e, A.FloatLit):
            return self.const(e.value, t)
        if isinstance(e, A.BoolLit):
            return V(T.BOOL, "true" if e.value else "false")
        if isinstance(e, A.StrLit):
            return V(T.STR, e.value)
        if isinstance(e, A.Name):
            sym = e.ref
            if sym.kind == "const":
                return self.const(sym.value, t)
            if sym.kind == "kernelvar":
                return self.s.kernel_vars[sym.name]
            return self.load(sym.ty, self.lookup_var(sym))
        if isinstance(e, A.Unary):
            return self.unary(e)
        if isinstance(e, A.Binary):
            return self.binary(e)
        if isinstance(e, A.Cast):
            return self.convert(self.expr(e.expr), e.target)
        if isinstance(e, A.Call):
            return self.call(e)
        if isinstance(e, A.Index):
            bt = e.base.ty
            if bt.is_vec:
                v = self.expr(e.base)
                idx = self.index64(e.index)
                return V(t, self.inst(f"extractelement {self.ll(bt)} {v.ref}, i64 {idx}"))
            if bt.is_mat and isinstance(e.index, A.IntLit):
                v = self.expr(e.base)
                return V(t, self.inst(f"extractvalue {self.ll(bt)} {v.ref}, {e.index.value}"))
            if bt.is_mat:
                p = self.spill(self.expr(e.base))
                idx = self.index64(e.index)
                off = self.inst(f"mul i64 {idx}, {bt.n}")
                q = self.inst(f"getelementptr inbounds float, ptr {p}, i64 {off}")
                return self.load(t, q)
            return self.load(t, self.addr(e))
        if isinstance(e, A.Field):
            if e.kind == "swizzle":
                v = self.expr(e.base)
                if len(e.info) == 1:
                    return self.extract(v, e.info[0])
                return self.shuffle(v, e.info)
            return self.load(t, self.addr(e))
        if isinstance(e, A.ArrayLit):
            p = self.alloca(t, "arr")
            for i, x in enumerate(e.elems):
                q = self.inst(f"getelementptr inbounds {self.llm(t)}, ptr {p}, i64 0, i64 {i}")
                self.store(self.expr(x), q)
            return V(t, p, fresh=True)
        raise AssertionError(e)

    def unary(self, e):
        op = e.op
        if op == "&":
            return V(e.ty, self.addr(e.operand))
        if op == "*":
            return self.load(e.ty, self.expr(e.operand).ref)
        v = self.expr(e.operand)
        t = v.ty
        if op == "-":
            if t.is_mat:
                return self.mat_map(t, lambda c: V(t.col, self.inst(f"fneg {self.fm()}{self.ll(t.col)} {c.ref}")), v)
            if t.elem_scalar.is_float:
                return V(t, self.inst(f"fneg {self.fm()}{self.ll(t)} {v.ref}"))
            return V(t, self.inst(f"sub {self.ll(t)} zeroinitializer, {v.ref}"))
        if op == "!":
            return V(t, self.inst(f"xor i1 {v.ref}, true"))
        if op == "~":
            ones = self.splat_const(-1, t)
            return V(t, self.inst(f"xor {self.ll(t)} {v.ref}, {ones.ref}"))
        raise AssertionError(op)

    def binary(self, e):
        op = e.op
        if op in ("&&", "||"):
            slot = self.alloca(T.BOOL, "sc")
            a = self.expr(e.left)
            self.store(a, slot)
            rhs_l, end_l = self.label("rhs"), self.label("scend")
            if op == "&&":
                self.cbr(a.ref, rhs_l, end_l)
            else:
                self.cbr(a.ref, end_l, rhs_l)
            self.start(rhs_l)
            b = self.expr(e.right)
            self.store(b, slot)
            self.br(end_l)
            self.start(end_l)
            return self.load(T.BOOL, slot)
        a = self.expr(e.left)
        b = self.expr(e.right)
        if op in ("==", "!=", "<", ">", "<=", ">="):
            return self.compare(op, a, b)
        return self.arith(op, a, b, e.loc)

    def compare(self, op, a, b):
        t = a.ty
        if t.is_float:
            pred = {"==": "oeq", "!=": "une", "<": "olt", ">": "ogt", "<=": "ole", ">=": "oge"}[op]
            return V(T.BOOL, self.inst(f"fcmp {pred} {self.ll(t)} {a.ref}, {b.ref}"))
        if t.is_signed:
            pred = {"==": "eq", "!=": "ne", "<": "slt", ">": "sgt", "<=": "sle", ">=": "sge"}[op]
        else:
            pred = {"==": "eq", "!=": "ne", "<": "ult", ">": "ugt", "<=": "ule", ">=": "uge"}[op]
        return V(T.BOOL, self.inst(f"icmp {pred} {self.ll(t)} {a.ref}, {b.ref}"))

    def mat_map(self, t, fn, *mats):
        agg = "poison"
        for i in range(t.n):
            cols = [V(t.col, self.inst(f"extractvalue {self.ll(t)} {m.ref}, {i}")) if m.ty.is_mat else m
                    for m in mats]
            c = fn(*cols)
            agg = self.inst(f"insertvalue {self.ll(t)} {agg}, {self.ll(t.col)} {c.ref}, {i}")
        return V(t, agg)

    def mat_vec(self, m, v):
        t = m.ty
        acc = None
        for i in range(t.n):
            col = V(t.col, self.inst(f"extractvalue {self.ll(t)} {m.ref}, {i}"))
            lane = self.splat(self.extract(v, i), t.col)
            prod = self.inst(f"fmul {self.fm()}{self.ll(t.col)} {col.ref}, {lane.ref}")
            acc = prod if acc is None else self.inst(f"fadd {self.fm()}{self.ll(t.col)} {acc}, {prod}")
        return V(t.col, acc)

    def arith(self, op, a, b, loc):
        if a.ty.is_ptr:
            idx = b.ref
            if b.ty.bits != 64:
                idx = self.inst(f"{'sext' if b.ty.is_signed else 'zext'} {self.ll(b.ty)} {b.ref} to i64")
            if op == "-":
                idx = self.inst(f"sub i64 0, {idx}")
            return V(a.ty, self.inst(f"getelementptr inbounds {self.llm(a.ty.pointee)}, ptr {a.ref}, i64 {idx}"))
        if a.ty.is_mat or b.ty.is_mat:
            if a.ty.is_mat and b.ty.is_mat and op == "*":
                t = a.ty
                return self.mat_map(t, lambda c: self.mat_vec(a, c), b)
            if a.ty.is_mat and b.ty.is_vec:
                return self.mat_vec(a, b)
            if a.ty.is_mat and b.ty.is_mat:
                inst = "fadd" if op == "+" else "fsub"
                t = a.ty
                return self.mat_map(t, lambda x, y: V(t.col, self.inst(
                    f"{inst} {self.fm()}{self.ll(t.col)} {x.ref}, {y.ref}")), a, b)
            m, s = (a, b) if a.ty.is_mat else (b, a)
            t = m.ty
            sv = self.splat(s, t.col)
            inst = "fmul" if op == "*" else "fdiv"
            return self.mat_map(t, lambda c: V(t.col, self.inst(
                f"{inst} {self.fm()}{self.ll(t.col)} {c.ref}, {sv.ref}")), m)
        a, b = self.broadcast_pair(a, b)
        t = a.ty
        s = t.elem_scalar
        lt = self.ll(t)
        if s.is_float:
            fm = self.fm()
            if op == "%":
                q = self.inst(f"fdiv {self.fm_exact_div()}{lt} {a.ref}, {b.ref}")
                tr = self.call_intrinsic("llvm.trunc", t, [V(t, q)])
                p = self.inst(f"fmul {fm}{lt} {tr.ref}, {b.ref}")
                return V(t, self.inst(f"fsub {fm}{lt} {a.ref}, {p}"))
            inst = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}[op]
            return V(t, self.inst(f"{inst} {fm}{lt} {a.ref}, {b.ref}"))
        if op in ("<<", ">>"):
            mask = self.splat_const(s.bits - 1, t)
            amt = self.inst(f"and {lt} {b.ref}, {mask.ref}")
            inst = "shl" if op == "<<" else ("ashr" if s.is_signed else "lshr")
            return V(t, self.inst(f"{inst} {lt} {a.ref}, {amt}"))
        if s.is_bool:
            inst = {"&": "and", "|": "or", "^": "xor"}[op]
            return V(t, self.inst(f"{inst} i1 {a.ref}, {b.ref}"))
        if op in ("/", "%"):
            return self.int_divide(op, a, b, t, s)
        inst = {"+": "add", "-": "sub", "*": "mul", "&": "and", "|": "or", "^": "xor"}[op]
        return V(t, self.inst(f"{inst} {lt} {a.ref}, {b.ref}"))

    def int_divide(self, op, a, b, t, s):
        """HA++ integer division never traps: x/0 = 0 and x%0 = x (what ARM64 does in hardware),
        MIN/-1 = MIN and MIN%-1 = 0 (wrap-around). Constant divisors fold all checks away."""
        lt = self.ll(t)
        zero, one = self.splat_const(0, t), self.splat_const(1, t)
        ct = f"<{t.n} x i1>" if t.is_vec else "i1"
        bad = self.inst(f"icmp eq {lt} {b.ref}, {zero.ref}")
        if s.is_signed:
            m1 = self.inst(f"icmp eq {lt} {b.ref}, {self.splat_const(-1, t).ref}")
            bad = self.inst(f"or {ct} {bad}, {m1}")
        safe = self.inst(f"select {ct} {bad}, {lt} {one.ref}, {lt} {b.ref}")
        inst = ("sdiv" if s.is_signed else "udiv") if op == "/" else ("srem" if s.is_signed else "urem")
        r = self.inst(f"{inst} {lt} {a.ref}, {safe}")
        # with divisor 1: q = a, r = 0. Fix up the two special cases.
        isz = self.inst(f"icmp eq {lt} {b.ref}, {zero.ref}")
        if op == "/":
            if s.is_signed:
                neg = self.inst(f"sub {lt} {zero.ref}, {a.ref}")          # a / -1 = -a (wraps for MIN)
                r = self.inst(f"select {ct} {m1}, {lt} {neg}, {lt} {r}")
            r = self.inst(f"select {ct} {isz}, {lt} {zero.ref}, {lt} {r}")
        else:
            r = self.inst(f"select {ct} {isz}, {lt} {a.ref}, {lt} {r}")      # a % 0 = a; a % -1 = 0 already
        return V(t, r)

    # ================================================================ conversions
    def convert(self, v, dst):
        src = v.ty
        if src == dst:
            return v
        if src.is_ptr and dst.is_ptr:
            return V(dst, v.ref)
        if src.is_ptr:
            return V(dst, self.inst(f"ptrtoint ptr {v.ref} to {self.ll(dst)}"))
        if dst.is_ptr:
            return V(dst, self.inst(f"inttoptr {self.ll(src)} {v.ref} to ptr"))
        s, d = src.elem_scalar, dst.elem_scalar
        ls, ld = self.ll(src), self.ll(dst)
        if s.is_bool:
            return V(dst, self.inst(f"zext {ls} {v.ref} to {ld}"))
        if s.is_int and d.is_int:
            if s.bits == d.bits:
                return V(dst, v.ref)
            if s.bits > d.bits:
                return V(dst, self.inst(f"trunc {ls} {v.ref} to {ld}"))
            return V(dst, self.inst(f"{'sext' if s.is_signed else 'zext'} {ls} {v.ref} to {ld}"))
        if s.is_int and d.is_float:
            return V(dst, self.inst(f"{'sitofp' if s.is_signed else 'uitofp'} {ls} {v.ref} to {ld}"))
        if s.is_float and d.is_int:
            name = f"llvm.{'fptosi' if d.is_signed else 'fptoui'}.sat.{self.ovl(dst)}.{self.ovl(src)}"
            self.intrinsic(name, ld, [ls])
            return V(dst, self.inst(f"call {ld} @{name}({ls} {v.ref})"))
        if s.is_float and d.is_float:
            if s.bits < d.bits:
                return V(dst, self.inst(f"fpext {ls} {v.ref} to {ld}"))
            if s.bits == 64 and d.bits == 16 and not self.is_arm64 and not src.is_vec:
                self.need_f64_to_f16 = True
                return V(dst, self.inst(f'call half @"ha.f64_to_f16"(double {v.ref})'))
            return V(dst, self.inst(f"fptrunc {ls} {v.ref} to {ld}"))
        raise HappError(f"internal: can't convert {src} to {dst}")

    # ================================================================ calls
    def call_intrinsic(self, base, t, args, ret=None, extra_args=(), flags=True):
        rt = ret or t
        name = f"{base}.{self.ovl(t)}"
        self.intrinsic(name, self.ll(rt), [self.ll(a.ty) for a in args] + [x.split()[0] for x in extra_args])
        fm = self.fm() if flags and t.elem_scalar is not None and t.elem_scalar.is_float else ""
        arglist = ", ".join([f"{self.ll(a.ty)} {a.ref}" for a in args] + list(extra_args))
        return V(rt, self.inst(f"call {fm}{self.ll(rt)} @{name}({arglist})"))

    def call(self, e):
        if e.kind == "fn":
            vals = []
            for a, p in zip(e.args, e.target.params):
                v = self.expr(a.value)
                if p.ty.in_memory and not v.fresh:
                    tmp = self.alloca(p.ty, "arg")        # copy now: later arguments may change the original
                    self.memcpy(tmp, v.ref, p.ty)
                    v = V(p.ty, tmp, fresh=True)
                vals.append(v)
            return self.call_fn(e.target, vals)
        if e.kind == "ctor_vec":
            t = e.target
            comps = []
            for a in e.args:
                comps.extend(self.lanes_of(self.expr(a.value)))
            if len(comps) == 1 and t.n > 1:
                return self.splat(comps[0], t)
            return self.build_vec(t, comps)
        if e.kind == "ctor_mat":
            t = e.target
            vals = [self.expr(a.value) for a in e.args]
            if e.info == "diag":
                zero = self.const(0.0, T.F32)
                cols = [self.build_vec(t.col, [vals[0] if r == c else zero for r in range(t.n)])
                        for c in range(t.n)]
            elif e.info == "cols":
                cols = vals
            else:
                cols = [self.build_vec(t.col, vals[c * t.n:(c + 1) * t.n]) for c in range(t.n)]
            agg = "poison"
            for i, c in enumerate(cols):
                agg = self.inst(f"insertvalue {self.ll(t)} {agg}, {self.ll(t.col)} {c.ref}, {i}")
            return V(t, agg)
        if e.kind == "ctor_struct":
            st = e.target
            p = self.alloca(st, "obj")
            if len(e.info) < len(st.fields):
                self.memzero(p, st)
            for idx, val in e.info:
                q = self.inst(f"getelementptr inbounds {self.llm(st)}, ptr {p}, i32 0, i32 {idx}")
                self.store(self.expr(val), q)
            return V(st, p, fresh=True)
        if e.kind == "builtin":
            return self.builtin(e)
        raise AssertionError(e.kind)

    def call_fn(self, fn, vals):
        args = []
        ret_slot = None
        if fn.ret_ty.in_memory:
            ret_slot = self.alloca(fn.ret_ty, "ret")
            args.append(f"ptr {ret_slot}")
        for v, p in zip(vals, fn.params):
            if p.ty.in_memory:
                if v.fresh:
                    args.append(f"ptr {v.ref}")
                else:
                    tmp = self.alloca(p.ty, "arg")
                    self.memcpy(tmp, v.ref, p.ty)
                    args.append(f"ptr {tmp}")
            elif fn.is_extern:
                args.append(f"{self.param_decl(p.ty)} {v.ref}")
            else:
                args.append(f"{self.ll(p.ty)} {v.ref}")
        sym = self.fn_symbol(fn)
        if ret_slot is not None:
            self.emit(f"call void {sym}({', '.join(args)})")
            return V(fn.ret_ty, ret_slot, fresh=True)
        rt = self.ret_decl(fn)
        if fn.ret_ty.is_void:
            self.emit(f"call void {sym}({', '.join(args)})")
            return V(T.VOID, "")
        return V(fn.ret_ty, self.inst(f"call {rt} {sym}({', '.join(args)})"))

    def std_call(self, name, vals):
        fn = self.prog.fns[name]
        return self.call_fn(fn, vals)

    def float_std(self, fname, t, vals):
        """Call an f32 std-library function lane by lane (f16 goes through f32)."""
        def one(*lanes):
            s = lanes[0].ty
            if s == T.F16:
                ups = [self.convert(x, T.F32) for x in lanes]
                return self.convert(self.std_call(fname, ups), T.F16)
            return self.std_call(fname, list(lanes))
        return self.map_lanes(t, one, *vals)

    def builtin(self, e):
        name = e.name
        t = e.ty
        args = [self.expr(a.value) for a in e.args]
        fm = self.fm()
        if name in STD_IMPL:
            vals = args
            if len(vals) == 2:
                vals = list(self.broadcast_pair(*vals))
            return self.float_std(STD_IMPL[name], t, vals)
        if name == "sqrt":
            return self.call_intrinsic("llvm.sqrt", t, args)
        if name == "rsqrt":
            r = self.call_intrinsic("llvm.sqrt", t, args)
            one = self.splat_const(1.0, t)
            return V(t, self.inst(f"fdiv {fm}{self.ll(t)} {one.ref}, {r.ref}"))
        if name in ("floor", "ceil", "trunc"):
            return self.call_intrinsic(f"llvm.{name}", t, args)
        if name == "round":
            # half away from zero, exact: trunc(x) + (|x - trunc(x)| >= 0.5 ? copysign(1, x) : 0)
            x = args[0]
            lt = self.ll(t)
            tr = self.call_intrinsic("llvm.trunc", t, [x], flags=False)
            frac = V(t, self.inst(f"fsub {lt} {x.ref}, {tr.ref}"))
            af = self.call_intrinsic("llvm.fabs", t, [frac], flags=False)
            ge = self.inst(f"fcmp oge {lt} {af.ref}, {self.splat_const(0.5, t).ref}")
            one = self.call_intrinsic("llvm.copysign", t, [self.splat_const(1.0, t), x], flags=False)
            ct = f"<{t.n} x i1>" if t.is_vec else "i1"
            step = self.inst(f"select {ct} {ge}, {lt} {one.ref}, {lt} {self.splat_const(0.0, t).ref}")
            return V(t, self.inst(f"fadd {lt} {tr.ref}, {step}"))
        if name == "fract":
            f = self.call_intrinsic("llvm.floor", t, args)
            return V(t, self.inst(f"fsub {fm}{self.ll(t)} {args[0].ref}, {f.ref}"))
        if name == "abs":
            if t.elem_scalar.is_float:
                return self.call_intrinsic("llvm.fabs", t, args)
            if not t.elem_scalar.is_signed:
                return args[0]
            return self.call_intrinsic("llvm.abs", t, args, extra_args=["i1 false"])
        if name == "sign":
            x = args[0]
            s = t.elem_scalar
            zero, one, neg = (self.splat_const(v, t) for v in (0, 1, -1 if s.is_signed or s.is_float else 0))
            lt = self.ll(t)
            if s.is_float:
                gt = self.inst(f"fcmp ogt {lt} {x.ref}, {zero.ref}")
                ls = self.inst(f"fcmp olt {lt} {x.ref}, {zero.ref}")
            else:
                gt = self.inst(f"icmp {'sgt' if s.is_signed else 'ugt'} {lt} {x.ref}, {zero.ref}")
                ls = self.inst(f"icmp slt {lt} {x.ref}, {zero.ref}") if s.is_signed else None
            ct = self.ll(T.Vec(T.BOOL, t.n)) if t.is_vec else "i1"
            r = self.inst(f"select {ct} {gt}, {lt} {one.ref}, {lt} {zero.ref}")
            if ls is not None:
                r = self.inst(f"select {ct} {ls}, {lt} {neg.ref}, {lt} {r}")
            return V(t, r)
        if name in ("min", "max"):
            a, b = self.broadcast_pair(*args)
            return self.minmax(name, a, b)
        if name == "clamp":
            x, lo, hi = args
            lo = self.splat(lo, t) if t.is_vec and lo.ty.is_scalar else lo
            hi = self.splat(hi, t) if t.is_vec and hi.ty.is_scalar else hi
            return self.minmax("min", self.minmax("max", x, lo), hi)
        if name == "mix":
            a, b, w = args
            if t.is_vec and w.ty.is_scalar:
                w = self.splat(w, t)
            lt = self.ll(t)
            d = self.inst(f"fsub {fm}{lt} {b.ref}, {a.ref}")
            p = self.inst(f"fmul {fm}{lt} {d}, {w.ref}")
            return V(t, self.inst(f"fadd {fm}{lt} {a.ref}, {p}"))
        if name == "step":
            edge, x = args
            lt = self.ll(t)
            c = self.inst(f"fcmp olt {lt} {x.ref}, {edge.ref}")
            zero, one = self.splat_const(0.0, t), self.splat_const(1.0, t)
            ct = f"<{t.n} x i1>" if t.is_vec else "i1"
            return V(t, self.inst(f"select {ct} {c}, {lt} {zero.ref}, {lt} {one.ref}"))
        if name == "smoothstep":
            e0, e1, x = args
            if t.is_vec:
                e0 = self.splat(e0, t) if e0.ty.is_scalar else e0
                e1 = self.splat(e1, t) if e1.ty.is_scalar else e1
            lt = self.ll(t)
            num = self.inst(f"fsub {fm}{lt} {x.ref}, {e0.ref}")
            den = self.inst(f"fsub {fm}{lt} {e1.ref}, {e0.ref}")
            q = V(t, self.inst(f"fdiv {fm}{lt} {num}, {den}"))
            c = self.minmax("min", self.minmax("max", q, self.splat_const(0.0, t)), self.splat_const(1.0, t))
            two_c = self.inst(f"fmul {fm}{lt} {c.ref}, {self.splat_const(2.0, t).ref}")
            k = self.inst(f"fsub {fm}{lt} {self.splat_const(3.0, t).ref}, {two_c}")
            cc = self.inst(f"fmul {fm}{lt} {c.ref}, {c.ref}")
            return V(t, self.inst(f"fmul {fm}{lt} {cc}, {k}"))
        if name == "fma":
            return self.call_intrinsic("llvm.fmuladd", t, args)
        if name == "dot":
            return self.dot(args[0], args[1])
        if name == "length":
            d = self.dot(args[0], args[0])
            return self.call_intrinsic("llvm.sqrt", d.ty, [d])
        if name == "distance":
            diff = V(args[0].ty, self.inst(f"fsub {fm}{self.ll(args[0].ty)} {args[0].ref}, {args[1].ref}"))
            d = self.dot(diff, diff)
            return self.call_intrinsic("llvm.sqrt", d.ty, [d])
        if name == "normalize":
            v = args[0]
            d = self.dot(v, v)
            ln = self.call_intrinsic("llvm.sqrt", d.ty, [d])
            inv = V(d.ty, self.inst(f"fdiv {fm}{self.ll(d.ty)} {self.fconst(1.0, d.ty)}, {ln.ref}"))
            sv = self.splat(inv, t)
            return V(t, self.inst(f"fmul {fm}{self.ll(t)} {v.ref}, {sv.ref}"))
        if name == "cross":
            a, b = args
            a1, b1 = self.shuffle(a, [1, 2, 0]), self.shuffle(b, [2, 0, 1])
            a2, b2 = self.shuffle(a, [2, 0, 1]), self.shuffle(b, [1, 2, 0])
            lt = self.ll(t)
            p = self.inst(f"fmul {fm}{lt} {a1.ref}, {b1.ref}")
            q = self.inst(f"fmul {fm}{lt} {a2.ref}, {b2.ref}")
            return V(t, self.inst(f"fsub {fm}{lt} {p}, {q}"))
        if name == "transpose":
            m = args[0]
            cols = [V(t.col, self.inst(f"extractvalue {self.ll(t)} {m.ref}, {i}")) for i in range(t.n)]
            new = [self.build_vec(t.col, [self.extract(cols[r], c) for r in range(t.n)]) for c in range(t.n)]
            agg = "poison"
            for i, c in enumerate(new):
                agg = self.inst(f"insertvalue {self.ll(t)} {agg}, {self.ll(t.col)} {c.ref}, {i}")
            return V(t, agg)
        if name == "select":
            c, a, b = args
            return V(t, self.inst(f"select i1 {c.ref}, {self.ll(t)} {a.ref}, {self.ll(t)} {b.ref}"))
        if name == "mod":
            x, y = args
            if t.is_vec and y.ty.is_scalar:
                y = self.splat(y, t)
            lt = self.ll(t)
            q = V(t, self.inst(f"fdiv {self.fm_exact_div()}{lt} {x.ref}, {y.ref}"))
            f = self.call_intrinsic("llvm.floor", t, [q])
            p = self.inst(f"fmul {fm}{lt} {y.ref}, {f.ref}")
            return V(t, self.inst(f"fsub {fm}{lt} {x.ref}, {p}"))
        if name in ("f32_bits", "f32_from_bits", "f16_bits", "f16_from_bits", "f64_bits", "f64_from_bits"):
            return V(t, self.inst(f"bitcast {self.ll(args[0].ty)} {args[0].ref} to {self.ll(t)}"))
        if name == "pack_half2":
            h = self.inst(f"fptrunc <2 x float> {args[0].ref} to <2 x half>")
            return V(t, self.inst(f"bitcast <2 x half> {h} to i32"))
        if name == "unpack_half2":
            h = self.inst(f"bitcast i32 {args[0].ref} to <2 x half>")
            return V(t, self.inst(f"fpext <2 x half> {h} to <2 x float>"))
        if name == "popcount":
            return self.call_intrinsic("llvm.ctpop", t, args)
        if name in ("clz", "ctz"):
            return self.call_intrinsic("llvm.ctlz" if name == "clz" else "llvm.cttz", t, args,
                                       extra_args=["i1 false"])
        if name == "print":
            self.gen_print(args)
            return V(T.VOID, "")
        if name in ("atomic_add", "atomic_min", "atomic_max", "atomic_exchange"):
            buf, idx, val = args
            i64 = idx.ref if idx.ty.bits == 64 else self.inst(
                f"{'sext' if idx.ty.is_signed else 'zext'} {self.ll(idx.ty)} {idx.ref} to i64")
            p = self.inst(f"getelementptr inbounds {self.ll(t)}, ptr {buf.ref}, i64 {i64}")
            op = {"atomic_add": "add", "atomic_exchange": "xchg",
                  "atomic_min": "min" if t.is_signed else "umin",
                  "atomic_max": "max" if t.is_signed else "umax"}[name]
            return V(t, self.inst(f"atomicrmw {op} ptr {p}, {self.ll(t)} {val.ref} monotonic, align 4"))
        raise HappError(f"'{name}' isn't available in CPU code", e.loc)

    def minmax(self, name, a, b):
        t = a.ty
        s = t.elem_scalar
        if s == T.F16 and not self.is_arm64:
            wide = T.Vec(T.F32, t.n) if t.is_vec else T.F32
            r = self.minmax(name, self.convert(a, wide), self.convert(b, wide))
            return self.convert(r, t)
        if s.is_float:
            return self.call_intrinsic("llvm.minnum" if name == "min" else "llvm.maxnum", t, [a, b])
        base = ("s" if s.is_signed else "u") + name
        return self.call_intrinsic(f"llvm.{base}", t, [a, b])

    def dot(self, a, b):
        t = a.ty
        prod = V(t, self.inst(f"fmul {self.fm()}{self.ll(t)} {a.ref}, {b.ref}"))
        lanes = self.lanes_of(prod)
        acc = lanes[0]
        for x in lanes[1:]:
            acc = V(t.elem, self.inst(f"fadd {self.fm()}{self.ll(t.elem)} {acc.ref}, {x.ref}"))
        return acc

    # ================================================================ print
    def gen_print(self, args):
        if not self.print_enabled:
            if not self.warnings or "print" not in self.warnings[-1]:
                self.warnings.append("print() is ignored in library builds (it works in 'happ run')")
            return

        def rt(name, argtypes):
            self.decl(name, f"declare void @{name}({', '.join(argtypes)})")
            return name

        def one(v):
            t = v.ty
            if t == T.STR:
                name, n = self.string_global(v.ref)
                self.emit(f"call void @{rt('ha_rt_print_str', ['ptr', 'i64'])}(ptr {name}, i64 {n})")
            elif t.is_vec:
                self.emit(f"call void @{rt('ha_rt_print_open', [])}()")
                for i, lane in enumerate(self.lanes_of(v)):
                    if i:
                        self.emit(f"call void @{rt('ha_rt_print_comma', [])}()")
                    one(lane)
                self.emit(f"call void @{rt('ha_rt_print_close', [])}()")
            elif t.is_bool:
                z = self.inst(f"zext i1 {v.ref} to i32")
                self.emit(f"call void @{rt('ha_rt_print_bool', ['i32'])}(i32 {z})")
            elif t.is_float:
                if t.bits == 64:
                    self.emit(f"call void @{rt('ha_rt_print_f64', ['double'])}(double {v.ref})")
                else:
                    f = self.convert(v, T.F32) if t.bits == 16 else v
                    self.emit(f"call void @{rt('ha_rt_print_f32', ['float'])}(float {f.ref})")
            elif t.is_int:
                w = self.convert(v, T.I64 if t.is_signed else T.U64)
                fn = "ha_rt_print_i64" if t.is_signed else "ha_rt_print_u64"
                self.emit(f"call void @{rt(fn, ['i64'])}(i64 {w.ref})")
            elif t.is_ptr:
                self.emit(f"call void @{rt('ha_rt_print_ptr', ['ptr'])}(ptr {v.ref})")

        for i, a in enumerate(args):
            if i:
                self.emit(f"call void @{rt('ha_rt_print_space', [])}()")
            one(a)
        self.emit(f"call void @{rt('ha_rt_print_end', [])}()")

    # ================================================================ kernels on the CPU
    def gen_kernel_cpu(self, fn):
        """A kernel without barriers/shared/subgroups also runs on CPUs as <name>_cpu(...)."""
        if fn not in self.prog.cpu_kernels:
            return []
        wg = fn.sig["workgroup"]
        u3 = "<3 x i32>"
        extra = [f"{u3} %k.gid", f"{u3} %k.lid", f"{u3} %k.grp", f"{u3} %k.ngroups"]
        uvec3 = T.Vec(T.U32, 3)
        kvars = {"global_id": V(uvec3, "%k.gid"), "local_id": V(uvec3, "%k.lid"),
                 "group_id": V(uvec3, "%k.grp"), "num_groups": V(uvec3, "%k.ngroups"),
                 "group_size": V(uvec3, f"<i32 {wg[0]}, i32 {wg[1]}, i32 {wg[2]}>")}
        body_sym = f'@"ha.k.{fn.name}"'
        # kernel body: 'return' ends this invocation
        self.s = None
        lines = self.gen_kernel_body(fn, body_sym, extra, kvars)
        # driver: loops over every invocation of every workgroup
        params = [f"{self.ll(p.ty)} %p.{p.name}" for p in fn.params] + ["i32 %gx", "i32 %gy", "i32 %gz"]
        dll = "dllexport " if self.is_windows else ""
        attrs = self.attr_group()
        call_args = ", ".join(f"{self.ll(p.ty)} %p.{p.name}" for p in fn.params)
        d = [f"define {dll}void @{fn.name}_cpu({', '.join(params)}) {attrs} {{", "entry:"]
        d += [f"  %nx = mul i32 %gx, {wg[0]}", f"  %ny = mul i32 %gy, {wg[1]}", f"  %nz = mul i32 %gz, {wg[2]}",
              "  %ng0 = insertelement <3 x i32> poison, i32 %gx, i64 0",
              "  %ng1 = insertelement <3 x i32> %ng0, i32 %gy, i64 1",
              "  %ng = insertelement <3 x i32> %ng1, i32 %gz, i64 2",
              "  br label %zloop",
              "zloop:", "  %z = phi i32 [0, %entry], [%z1, %zstep]",
              "  %zc = icmp ult i32 %z, %nz", "  br i1 %zc, label %ybody, label %done",
              "ybody:", "  br label %yloop",
              "yloop:", "  %y = phi i32 [0, %ybody], [%y1, %ystep]",
              "  %yc = icmp ult i32 %y, %ny", "  br i1 %yc, label %xbody, label %zstep",
              "xbody:", "  br label %xloop",
              "xloop:", "  %x = phi i32 [0, %xbody], [%x1, %xloop.body]",
              "  %xc = icmp ult i32 %x, %nx", "  br i1 %xc, label %xloop.body, label %ystep",
              "xloop.body:",
              "  %g0 = insertelement <3 x i32> poison, i32 %x, i64 0",
              "  %g1 = insertelement <3 x i32> %g0, i32 %y, i64 1",
              "  %g = insertelement <3 x i32> %g1, i32 %z, i64 2",
              f"  %lid = urem <3 x i32> %g, <i32 {wg[0]}, i32 {wg[1]}, i32 {wg[2]}>",
              f"  %grp = udiv <3 x i32> %g, <i32 {wg[0]}, i32 {wg[1]}, i32 {wg[2]}>",
              f"  call void {body_sym}({call_args}{', ' if call_args else ''}<3 x i32> %g, <3 x i32> %lid, "
              f"<3 x i32> %grp, <3 x i32> %ng)",
              "  %x1 = add i32 %x, 1", "  br label %xloop",
              "ystep:", "  %y1 = add i32 %y, 1", "  br label %yloop",
              "zstep:", "  %z1 = add i32 %z, 1", "  br label %zloop",
              "done:", "  ret void", "}", ""]
        return lines + d

    def gen_kernel_body(self, fn, sym, extra, kvars):
        self.s = s = FnState()
        s.fn = fn
        if self.strict or "strict" in fn.attrs:
            s.fmf = ""
        s.kernel_vars = kvars
        s.kernel_return = "kret"
        params = [f"{self.ll(p.ty)} %p.{p.name}" for p in fn.params] + list(extra)
        attrs = self.attr_group(["alwaysinline"])
        header = f"define internal void {sym}({', '.join(params)}) {attrs} {{"
        self.start("entry")
        for p in fn.params:
            slot = self.alloca(p.ty, p.name)
            self.store(V(p.ty, f"%p.{p.name}"), slot)
            s.vars[("param", fn.name, p.name)] = slot
        self.block(fn.body)
        self.br("kret")
        self.start("kret")
        self.emit("ret void")
        lines = [header, "entry:"] + s.allocas + s.lines[1:] + ["}", ""]
        self.s = None
        return lines

    # ================================================================ entry points
    def gen_main_wrapper(self, fn):
        attrs = self.attr_group()
        if fn.ret_ty.is_void:
            return [f"define i32 @main() {attrs} {{", "entry:", '  call void @"ha.main"()', "  ret i32 0", "}", ""]
        return [f"define i32 @main() {attrs} {{", "entry:", '  %r = call i32 @"ha.main"()', "  ret i32 %r",
                "}", ""]

    JNI_SCALAR = {"bool": "i8", "i8": "i8", "u8": "i8", "i16": "i16", "u16": "i16", "i32": "i32",
                  "u32": "i32", "i64": "i64", "u64": "i64", "f32": "float", "f64": "double"}

    def gen_jni(self):
        """JNI entry points for Android (Kotlin/Java), emitted straight into the library.

        Primitive-type pointers map to Java arrays (pinned with GetPrimitiveArrayCritical),
        pointers to structs/vectors map to direct ByteBuffers (GetDirectBufferAddress)."""
        package, cls = self.jni
        attrs = self.attr_group()
        out = [
            f'define internal ptr @"ha.jni.pin"(ptr %env, ptr %arr) {attrs} {{',
            "entry:",
            "  %isnull = icmp eq ptr %arr, null",
            "  br i1 %isnull, label %null, label %get",
            "null:", "  ret ptr null",
            "get:",
            "  %tab = load ptr, ptr %env, align 8",
            "  %slot = getelementptr inbounds ptr, ptr %tab, i64 222",
            "  %fn = load ptr, ptr %slot, align 8",
            "  %p = call ptr %fn(ptr %env, ptr %arr, ptr null)",
            "  ret ptr %p", "}", "",
            f'define internal void @"ha.jni.unpin"(ptr %env, ptr %arr, ptr %p) {attrs} {{',
            "entry:",
            "  %isnull = icmp eq ptr %p, null",
            "  br i1 %isnull, label %done, label %rel",
            "rel:",
            "  %tab = load ptr, ptr %env, align 8",
            "  %slot = getelementptr inbounds ptr, ptr %tab, i64 223",
            "  %fn = load ptr, ptr %slot, align 8",
            "  call void %fn(ptr %env, ptr %arr, ptr %p, i32 0)",
            "  br label %done",
            "done:", "  ret void", "}", "",
            f'define internal ptr @"ha.jni.addr"(ptr %env, ptr %buf) {attrs} {{',
            "entry:",
            "  %isnull = icmp eq ptr %buf, null",
            "  br i1 %isnull, label %null, label %get",
            "null:", "  ret ptr null",
            "get:",
            "  %tab = load ptr, ptr %env, align 8",
            "  %slot = getelementptr inbounds ptr, ptr %tab, i64 230",
            "  %fn = load ptr, ptr %slot, align 8",
            "  %p = call ptr %fn(ptr %env, ptr %buf)",
            "  ret ptr %p", "}", "",
        ]
        dll = "dllexport " if self.is_windows else ""
        fns = list(self.prog.exports) + list(self.prog.cpu_kernels)
        for fn in fns:
            is_kernel = fn.is_kernel
            cname = f"{fn.name}_cpu" if is_kernel else fn.name
            params = list(fn.params)
            ptypes = [p.ty for p in params] + ([T.U32] * 3 if is_kernel else [])
            pnames = [p.name for p in params] + (["groups_x", "groups_y", "groups_z"] if is_kernel else [])
            sym = "Java_" + jni_mangle(package) + "_" + jni_mangle(cls) + "_" + jni_mangle(cname)
            jparams = ["ptr %env", "ptr %cls"]
            body, releases, call_args = [], [], []
            for i, (pt, pn) in enumerate(zip(ptypes, pnames)):
                if pt.is_ptr:
                    jparams.append(f"ptr %j{i}")
                    pte = pt.pointee
                    if pte.is_scalar:
                        body.append(f'  %a{i} = call ptr @"ha.jni.pin"(ptr %env, ptr %j{i})')
                        releases.append(f'  call void @"ha.jni.unpin"(ptr %env, ptr %j{i}, ptr %a{i})')
                    else:
                        body.append(f'  %a{i} = call ptr @"ha.jni.addr"(ptr %env, ptr %j{i})')
                    call_args.append(f"ptr %a{i}")
                elif pt.is_bool:
                    jparams.append(f"i8 %j{i}")
                    body.append(f"  %a{i} = icmp ne i8 %j{i}, 0")
                    call_args.append(f"i1 %a{i}")
                else:
                    jparams.append(f"{self.JNI_SCALAR[pt.name]} %j{i}")
                    call_args.append(f"{self.ll(pt)} %j{i}")
            rt = T.VOID if is_kernel else fn.ret_ty
            if rt.is_void:
                jret = "void"
                body.append(f"  call void @{cname}({', '.join(call_args)})")
                ret = "  ret void"
            elif rt.is_ptr:
                jret = "i64"
                body.append(f"  %r = call ptr @{cname}({', '.join(call_args)})")
                body.append("  %rj = ptrtoint ptr %r to i64")
                ret = "  ret i64 %rj"
            elif rt.is_bool:
                jret = "i8"
                body.append(f"  %r = call zeroext i1 @{cname}({', '.join(call_args)})")
                body.append("  %rj = zext i1 %r to i8")
                ret = "  ret i8 %rj"
            else:
                jret = self.JNI_SCALAR[rt.name]
                body.append(f"  %r = call {self.ret_decl(fn)} @{cname}({', '.join(call_args)})")
                ret = f"  ret {jret} %r"
            out.append(f"define {dll}{jret} @{sym}({', '.join(jparams)}) {attrs} {{")
            out.append("entry:")
            out.extend(body)
            out.extend(releases)
            out.append(ret)
            out.extend(["}", ""])
        if self.gpu_runtime:
            out += self.gen_jni_gpu(package, cls, attrs, dll)
        return out

    RUNTIME_DECLS = {
        "ha_gpu_create": "declare ptr @ha_gpu_create(ptr, i64)",
        "ha_gpu_destroy": "declare void @ha_gpu_destroy(ptr)",
        "ha_gpu_name": "declare ptr @ha_gpu_name(ptr)",
        "ha_gpu_has_f16": "declare i32 @ha_gpu_has_f16(ptr)",
        "ha_buffer_create": "declare ptr @ha_buffer_create(ptr, i64)",
        "ha_buffer_data": "declare ptr @ha_buffer_data(ptr)",
        "ha_buffer_size": "declare i64 @ha_buffer_size(ptr)",
        "ha_buffer_destroy": "declare void @ha_buffer_destroy(ptr)",
        "ha_kernel_create": "declare ptr @ha_kernel_create(ptr, ptr, i64, i32, i32)",
        "ha_kernel_destroy": "declare void @ha_kernel_destroy(ptr)",
        "ha_dispatch": "declare i32 @ha_dispatch(ptr, ptr, ptr, ptr, i32, i32, i32)",
        "ha_batch_begin": "declare i32 @ha_batch_begin(ptr)",
        "ha_batch_dispatch": "declare i32 @ha_batch_dispatch(ptr, ptr, ptr, ptr, i32, i32, i32)",
        "ha_batch_submit": "declare i32 @ha_batch_submit(ptr)",
    }

    def gen_jni_gpu(self, package, cls, attrs, dll):
        """JNI methods for the Vulkan runtime and typed per-kernel launchers (Android / desktop JVM)."""
        from .gpu import kernel_layout
        for name, text in self.RUNTIME_DECLS.items():
            self.decl(name, text)

        def jn(m):
            return "Java_" + jni_mangle(package) + "_" + jni_mangle(cls) + "_" + jni_mangle(m)

        def fn(ret, method, params, body):
            return [f"define {dll}{ret} @{jn(method)}(ptr %env, ptr %cls{''.join(', ' + p for p in params)}) "
                    f"{attrs} {{", "entry:"] + ["  " + b for b in body] + ["}", ""]

        def jcall(idx, ret, args):
            return ["%tab = load ptr, ptr %env, align 8",
                    f"%slot = getelementptr inbounds ptr, ptr %tab, i64 {idx}",
                    "%jfn = load ptr, ptr %slot, align 8",
                    f"%jr = call {ret} %jfn(ptr %env{''.join(', ' + a for a in args)})"]

        out = []
        out += fn("i64", "gpuCreate", [], ["%g = call ptr @ha_gpu_create(ptr null, i64 0)",
                                          "%r = ptrtoint ptr %g to i64", "ret i64 %r"])
        out += fn("void", "gpuDestroy", ["i64 %g"], ["%p = inttoptr i64 %g to ptr",
                                                     "call void @ha_gpu_destroy(ptr %p)", "ret void"])
        out += fn("ptr", "gpuName", ["i64 %g"], ["%p = inttoptr i64 %g to ptr",
                                                 "%s = call ptr @ha_gpu_name(ptr %p)"]
                  + jcall(167, "ptr", ["ptr %s"]) + ["ret ptr %jr"])
        out += fn("i8", "gpuHasF16", ["i64 %g"], ["%p = inttoptr i64 %g to ptr",
                                                  "%v = call i32 @ha_gpu_has_f16(ptr %p)",
                                                  "%b = icmp ne i32 %v, 0", "%r = zext i1 %b to i8", "ret i8 %r"])
        out += fn("i64", "bufferCreate", ["i64 %g", "i64 %n"], ["%p = inttoptr i64 %g to ptr",
                                                               "%b = call ptr @ha_buffer_create(ptr %p, i64 %n)",
                                                               "%r = ptrtoint ptr %b to i64", "ret i64 %r"])
        out += fn("ptr", "bufferData", ["i64 %b"], ["%p = inttoptr i64 %b to ptr",
                                                    "%d = call ptr @ha_buffer_data(ptr %p)",
                                                    "%n = call i64 @ha_buffer_size(ptr %p)"]
                  + jcall(229, "ptr", ["ptr %d", "i64 %n"]) + ["ret ptr %jr"])
        out += fn("void", "bufferDestroy", ["i64 %b"], ["%p = inttoptr i64 %b to ptr",
                                                        "call void @ha_buffer_destroy(ptr %p)", "ret void"])
        out += fn("void", "kernelDestroy", ["i64 %k"], ["%p = inttoptr i64 %k to ptr",
                                                        "call void @ha_kernel_destroy(ptr %p)", "ret void"])
        out += fn("i32", "batchBegin", ["i64 %g"], ["%p = inttoptr i64 %g to ptr",
                                                    "%r = call i32 @ha_batch_begin(ptr %p)", "ret i32 %r"])
        out += fn("i32", "batchSubmit", ["i64 %g"], ["%p = inttoptr i64 %g to ptr",
                                                     "%r = call i32 @ha_batch_submit(ptr %p)", "ret i32 %r"])
        for k in self.prog.kernels:
            blob = f"spirv_{k.name}"
            if blob not in self.gpu_blobs:
                continue
            buffers, scalars = kernel_layout(k)
            push_bytes = 4 * len(scalars)
            out += fn("i64", f"{k.name}Kernel", ["i64 %g"], [
                "%sz = alloca i64, align 8",
                f"%blob = call ptr @{self.lib_name}_{blob}(ptr %sz)",
                "%n = load i64, ptr %sz, align 8",
                "%p = inttoptr i64 %g to ptr",
                f"%k = call ptr @ha_kernel_create(ptr %p, ptr %blob, i64 %n, i32 {len(buffers)}, i32 {push_bytes})",
                "%r = ptrtoint ptr %k to i64", "ret i64 %r"])
            params = ["i64 %g", "i64 %k"] + [f"i64 %b{i}" for i in range(len(buffers))]
            params += [f"{'float' if p.ty.is_float else 'i32'} %s{i}" for i, p in enumerate(scalars)]
            params += ["i32 %gx", "i32 %gy", "i32 %gz"]
            for method, rt_fn in ((f"{k.name}Run", "ha_dispatch"), (f"{k.name}Record", "ha_batch_dispatch")):
                body = ["%gp = inttoptr i64 %g to ptr", "%kp = inttoptr i64 %k to ptr"]
                if buffers:
                    body.append(f"%bufs = alloca [{len(buffers)} x ptr], align 8")
                    for i in range(len(buffers)):
                        body += [f"%bp{i} = inttoptr i64 %b{i} to ptr",
                                 f"%bs{i} = getelementptr inbounds [{len(buffers)} x ptr], ptr %bufs, i64 0, i64 {i}",
                                 f"store ptr %bp{i}, ptr %bs{i}, align 8"]
                    bufs = "%bufs"
                else:
                    bufs = "null"
                if scalars:
                    body.append(f"%push = alloca [{len(scalars)} x i32], align 4")
                    for i, p in enumerate(scalars):
                        ty = "float" if p.ty.is_float else "i32"
                        body += [f"%ps{i} = getelementptr inbounds [{len(scalars)} x i32], ptr %push, i64 0, i64 {i}",
                                 f"store {ty} %s{i}, ptr %ps{i}, align 4"]
                    push = "%push"
                else:
                    push = "null"
                body += [f"%rc = call i32 @{rt_fn}(ptr %gp, ptr %kp, ptr {bufs}, ptr {push}, i32 %gx, i32 %gy, "
                         f"i32 %gz)", "ret i32 %rc"]
                out += fn("i32", method, params, body)
        return out

    def gen_gpu_blobs(self):
        """Embed compiled GPU code in the library: <lib>_spirv_<kernel>() and <lib>_metal_source()."""
        out = []
        dll = "dllexport " if self.is_windows else ""
        attrs = self.attr_group()
        for name, data in sorted(self.gpu_blobs.items()):
            g = f'@"ha.blob.{name}"'
            enc = "".join("\\%02X" % b for b in data)
            self.globals.append(f'{g} = private unnamed_addr constant [{len(data) + 1} x i8] c"{enc}\\00", align 4')
            sym = f"{self.lib_name}_{name}"
            out += [f"define {dll}ptr @{sym}(ptr %size) {attrs} {{", "entry:",
                    "  %isnull = icmp eq ptr %size, null",
                    "  br i1 %isnull, label %done, label %set",
                    "set:", f"  store i64 {len(data)}, ptr %size, align 8", "  br label %done",
                    "done:", f"  ret ptr {g}", "}", ""]
        return out

    # ================================================================ runtime support
    def gen_runtime(self):
        out = []
        if self.is_windows and self.mode == "lib":
            # the DLL has no C runtime: provide the two symbols MSVC-style code expects
            self.globals.append("@_fltused = dso_local global i32 0, align 4")
            if self.is_arm64:
                asm = [".text", ".globl __chkstk", ".p2align 2", "__chkstk:", "lsl x16, x15, #4",
                       "mov x17, sp", "1:", "sub x17, x17, #4096", "subs x16, x16, #4096",
                       "ldr xzr, [x17]", "b.gt 1b", "ret"]
            else:
                asm = [".text", ".globl __chkstk", ".p2align 4", "__chkstk:", "push %rcx", "push %rax",
                       "cmp $0x1000, %rax", "lea 24(%rsp), %rcx", "jb 1f", "2:", "sub $0x1000, %rcx",
                       "test %rcx, (%rcx)", "sub $0x1000, %rax", "cmp $0x1000, %rax", "ja 2b", "1:",
                       "sub %rax, %rcx", "test %rcx, (%rcx)", "pop %rax", "pop %rcx", "ret"]
            self.out.extend(f'module asm "{line}"' for line in asm)
        if self.need_f64_to_f16 or self.f16_helpers:
            out += self.gen_f64_to_f16()
        if self.f16_helpers:
            out += self.gen_f16_helpers()
        if self.ptr64mem:
            out += self.gen_wasm_helpers()
        return out

    def gen_f64_to_f16(self):
        """double -> half with one rounding: round to float with 'round to odd', then to half."""
        attrs = self.attr_group()
        self.intrinsic("llvm.fabs.f64", "double", ["double"])
        return [
            f'define internal half @"ha.f64_to_f16"(double %d) {attrs} {{',
            "entry:",
            "  %f = fptrunc double %d to float",
            "  %back = fpext float %f to double",
            "  %ad = call double @llvm.fabs.f64(double %d)",
            "  %ab = call double @llvm.fabs.f64(double %back)",
            "  %away = fcmp ogt double %ab, %ad",
            "  %bits = bitcast float %f to i32",
            "  %dec = sub i32 %bits, 1",
            "  %t = select i1 %away, i32 %dec, i32 %bits",
            "  %inexact = fcmp une double %back, %d",
            "  %odd = or i32 %t, 1",
            "  %r = select i1 %inexact, i32 %odd, i32 %t",
            "  %fo = bitcast i32 %r to float",
            "  %h = fptrunc float %fo to half",
            "  ret half %h", "}", ""]

    def gen_wasm_helpers(self):
        """WebAssembly's f32.min/max return NaN if either input is NaN; HA++'s min/max (like ARM64's
        fminnm/fmaxnm and x86 with LLVM's lowering) return the other operand. LLVM calls these C
        library functions for that on wasm, and HA++ modules have no C library, so they are defined
        here. Unused ones are removed by the linker."""
        attrs = self.attr_group()
        out = []
        for name, ty, pred in (("fmaxf", "float", "ogt"), ("fminf", "float", "olt"),
                               ("fmax", "double", "ogt"), ("fmin", "double", "olt")):
            out += [f"define hidden {ty} @{name}({ty} %a, {ty} %b) {attrs} {{",
                    "entry:",
                    f"  %pick = fcmp {pred} {ty} %a, %b",
                    f"  %m = select i1 %pick, {ty} %a, {ty} %b",
                    f"  %bnan = fcmp uno {ty} %b, %b",
                    f"  %r = select i1 %bnan, {ty} %a, {ty} %m",
                    f"  ret {ty} %r", "}", ""]
        return out

    def gen_f16_helpers(self):
        """Half<->float conversions for x86 CPUs without F16C (e.g. the Android emulator ABI) and for
        WebAssembly. On wasm, LLVM passes half values to these functions as their 16 bits (i16)."""
        attrs = self.attr_group()
        bits = self.ptr64mem           # wasm: i16 in, i16 out
        return [
            (f"define hidden float @__extendhfsf2(i16 %b) {attrs} {{" if bits else
             f"define hidden float @__extendhfsf2(half %h) {attrs} {{"),
            "entry:",
            *([] if bits else ["  %b = bitcast half %h to i16"]),
            "  %x = zext i16 %b to i32",
            "  %sign = and i32 %x, 32768",
            "  %s = shl i32 %sign, 16",
            "  %em = and i32 %x, 32767",
            "  %shifted = shl i32 %em, 13",
            "  %exp = and i32 %x, 31744",
            "  %isinf = icmp eq i32 %exp, 31744",
            "  %iszero = icmp eq i32 %exp, 0",
            "  %norm = add i32 %shifted, 939524096",       # rebias exponent: (127-15) << 23
            "  %infnan = or i32 %shifted, 2139095040",     # exponent all ones
            "  %magic = or i32 %shifted, 947912704",       # subnormal: 2^-14 trick (113 << 23)
            "  %mf = bitcast i32 %magic to float",
            "  %sub = fsub float %mf, 0x3F10000000000000",  # minus 2^-14
            "  %subb = bitcast float %sub to i32",
            "  %r1 = select i1 %isinf, i32 %infnan, i32 %norm",
            "  %r2 = select i1 %iszero, i32 %subb, i32 %r1",
            "  %r = or i32 %r2, %s",
            "  %f = bitcast i32 %r to float",
            "  ret float %f", "}", "",
            f"define hidden {'i16' if bits else 'half'} @__truncsfhf2(float %f) {attrs} {{",
            "entry:",
            "  %x = bitcast float %f to i32",
            "  %sign = lshr i32 %x, 16",
            "  %sgn = and i32 %sign, 32768",
            "  %abs = and i32 %x, 2147483647",
            "  %isnan = icmp ugt i32 %abs, 2139095040",
            "  %big = icmp uge i32 %abs, 1199570944",       # >= 65520.0 -> inf
            "  %small = icmp ult i32 %abs, 947912704",      # < 2^-14 -> subnormal path
            # normal: round to nearest even
            "  %t0 = lshr i32 %abs, 13",
            "  %odd = and i32 %t0, 1",
            "  %bias = add i32 %odd, 4095",
            "  %t1 = add i32 %abs, %bias",
            "  %t2 = lshr i32 %t1, 13",
            "  %norm = sub i32 %t2, 114688",               # (127-15) << 10
            # subnormal: add 0.5 (magic) and take low bits
            "  %af = bitcast i32 %abs to float",
            "  %sf = fadd float %af, 0x3FE0000000000000",
            "  %sb = bitcast float %sf to i32",
            "  %subn = sub i32 %sb, 1056964608",
            "  %r1 = select i1 %small, i32 %subn, i32 %norm",
            "  %r2 = select i1 %big, i32 31744, i32 %r1",
            "  %r3 = select i1 %isnan, i32 32256, i32 %r2",
            "  %r = or i32 %r3, %sgn",
            "  %h16 = trunc i32 %r to i16",
            *(["  ret i16 %h16", "}", ""] if bits else ["  %h = bitcast i16 %h16 to half", "  ret half %h", "}", ""]),
            f"define hidden {'i16' if bits else 'half'} @__truncdfhf2(double %d) {attrs} {{",
            "entry:",
            '  %h = call half @"ha.f64_to_f16"(double %d)',
            *(["  %hb = bitcast half %h to i16", "  ret i16 %hb", "}", ""] if bits else ["  ret half %h", "}", ""]),
        ]
