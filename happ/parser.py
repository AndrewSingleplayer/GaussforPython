"""Recursive-descent parser: tokens -> syntax tree."""

from . import ast as A
from .errors import HappError
from .lexer import tokenize

BINARY_LEVELS = [
    ("||",),
    ("&&",),
    ("==", "!=", "<", ">", "<=", ">="),
    ("|",),
    ("^",),
    ("&",),
    ("<<", ">>"),
    ("+", "-"),
    ("*", "/", "%"),
]
COMPARISONS = {"==", "!=", "<", ">", "<=", ">="}
ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}


class Parser:
    def __init__(self, text, path):
        self.toks = tokenize(text, path)
        self.pos = 0
        self.path = path

    # ------------------------------------------------------------ helpers
    @property
    def tok(self):
        return self.toks[self.pos]

    def peek(self, k=1):
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def advance(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def at(self, kind, value=None):
        t = self.tok
        return t.kind == kind and (value is None or t.value == value)

    def at_op(self, value):
        return self.at("op", value)

    def accept(self, kind, value=None):
        if self.at(kind, value):
            return self.advance()
        return None

    def expect(self, kind, value=None, what=None):
        if self.at(kind, value):
            return self.advance()
        want = what or (repr(value) if value is not None else kind)
        got = self.tok.value if self.tok.kind != "eof" else "end of file"
        raise HappError(f"expected {want}, found {got!r}", self.tok.loc)

    def expect_op(self, value):
        return self.expect("op", value)

    def ident(self, what="a name"):
        return self.expect("ident", what=what)

    # ------------------------------------------------------------ items
    def parse_module(self):
        items = []
        while not self.at("eof"):
            items.append(self.parse_item())
        return A.Module(self.path, items)

    def parse_attrs(self):
        attrs = {}
        while self.at_op("@"):
            at = self.advance()
            name = self.ident("an attribute name").value
            args = []
            if self.accept("op", "("):
                while not self.at_op(")"):
                    t = self.expect("int", what="an integer")
                    args.append(t.value)
                    if not self.accept("op", ","):
                        break
                self.expect_op(")")
            if name in attrs:
                raise HappError(f"duplicate attribute @{name}", at.loc)
            attrs[name] = (args, at.loc)
        return attrs

    def parse_item(self):
        attrs = self.parse_attrs()
        t = self.tok
        if attrs and not (t.kind == "kw" and t.value in ("fn", "export", "kernel", "extern")):
            first = next(iter(attrs.values()))[1]
            raise HappError("attributes (@...) can only be used on functions and kernels", first)
        if t.kind == "kw":
            if t.value == "import":
                self.advance()
                path = self.expect("str", what="a file path string").value
                self.expect_op(";")
                return A.Import(path, t.loc)
            if t.value == "const":
                self.advance()
                name = self.ident().value
                ty = None
                if self.accept("op", ":"):
                    ty = self.parse_type()
                self.expect_op("=")
                value = self.parse_expr()
                self.expect_op(";")
                return A.ConstDef(name, ty, value, t.loc)
            if t.value == "struct":
                self.advance()
                name = self.ident("a struct name").value
                self.expect_op("{")
                fields = []
                while not self.at_op("}"):
                    ft = self.ident("a field name")
                    self.expect_op(":")
                    fields.append((ft.value, self.parse_type(), ft.loc))
                    if not self.accept("op", ","):
                        break
                self.expect_op("}")
                return A.StructDef(name, fields, t.loc)
            if t.value == "export":
                self.advance()
                if not self.at("kw", "fn"):
                    raise HappError("'export' must be followed by 'fn'", self.tok.loc)
                return self.parse_fn(attrs, is_export=True)
            if t.value == "fn":
                return self.parse_fn(attrs)
            if t.value == "extern":
                self.advance()
                if not self.at("kw", "fn"):
                    raise HappError("'extern' must be followed by 'fn'", self.tok.loc)
                return self.parse_fn(attrs, is_extern=True)
            if t.value == "kernel":
                return self.parse_fn(attrs, is_kernel=True)
        raise HappError("expected 'fn', 'export fn', 'kernel', 'struct', 'const', 'extern fn' or 'import'",
                        t.loc)

    def parse_fn(self, attrs, is_export=False, is_extern=False, is_kernel=False):
        start = self.advance()   # 'fn' or 'kernel'
        name = self.ident("a function name").value
        self.expect_op("(")
        params = []
        while not self.at_op(")"):
            pt = self.ident("a parameter name")
            self.expect_op(":")
            params.append(A.Param(pt.value, self.parse_type(), pt.loc))
            if not self.accept("op", ","):
                break
        self.expect_op(")")
        ret = None
        if self.accept("op", "->"):
            ret = self.parse_type()
        body = None
        if is_extern:
            self.expect_op(";")
        else:
            body = self.parse_block()
        return A.FnDef(name, params, ret, body, start.loc, is_export=is_export,
                       is_extern=is_extern, is_kernel=is_kernel, attrs=attrs)

    def parse_type(self):
        t = self.tok
        if self.accept("op", "*"):
            return A.TPtr(self.parse_type(), t.loc)
        if self.accept("op", "["):
            size = self.parse_expr()
            self.expect_op("]")
            return A.TArray(self.parse_type(), size, t.loc)
        name = self.expect("ident", what="a type")
        return A.TName(name.value, name.loc)

    # ------------------------------------------------------------ statements
    def parse_block(self):
        start = self.expect_op("{")
        stmts = []
        while not self.at_op("}"):
            if self.at("eof"):
                raise HappError("missing '}'", start.loc)
            stmts.append(self.parse_stmt())
        self.advance()
        return A.Block(stmts, start.loc)

    def parse_stmt(self):
        t = self.tok
        if t.kind == "kw":
            v = t.value
            if v in ("let", "var"):
                self.advance()
                name = self.ident().value
                ty = None
                if self.accept("op", ":"):
                    ty = self.parse_type()
                value = None
                if self.accept("op", "="):
                    value = self.parse_expr()
                elif v == "let":
                    raise HappError("'let' needs a value; use 'var' for a zero-initialized variable",
                                    self.tok.loc)
                elif ty is None:
                    raise HappError("'var' without a value needs a type, e.g. 'var x: f32;'",
                                    self.tok.loc)
                self.expect_op(";")
                return A.Let(name, ty, value, v == "var", t.loc)
            if v == "shared":
                self.advance()
                name = self.ident().value
                self.expect_op(":")
                ty = self.parse_type()
                self.expect_op(";")
                return A.Shared(name, ty, t.loc)
            if v == "if":
                return self.parse_if()
            if v == "while":
                self.advance()
                cond = self.parse_expr()
                return A.While(cond, self.parse_block(), t.loc)
            if v == "for":
                self.advance()
                var = self.ident("a loop variable").value
                vty = None
                if self.accept("op", ":"):
                    vty = self.parse_type()
                self.expect("kw", "in")
                start = self.parse_expr()
                self.expect_op("..")
                end = self.parse_expr()
                node = A.For(var, start, end, self.parse_block(), t.loc)
                node.vtype = vty
                return node
            if v == "return":
                self.advance()
                value = None if self.at_op(";") else self.parse_expr()
                self.expect_op(";")
                return A.Return(value, t.loc)
            if v == "break":
                self.advance()
                self.expect_op(";")
                return A.Break(t.loc)
            if v == "continue":
                self.advance()
                self.expect_op(";")
                return A.Continue(t.loc)
        if self.at_op("{"):
            return self.parse_block()
        expr = self.parse_expr()
        if self.tok.kind == "op" and self.tok.value in ASSIGN_OPS:
            op = self.advance().value
            value = self.parse_expr()
            self.expect_op(";")
            return A.Assign(expr, op, value, t.loc)
        self.expect_op(";")
        return A.ExprStmt(expr, t.loc)

    def parse_if(self):
        t = self.advance()
        cond = self.parse_expr()
        then = self.parse_block()
        els = None
        if self.accept("kw", "else"):
            els = self.parse_if() if self.at("kw", "if") else self.parse_block()
        return A.If(cond, then, els, t.loc)

    # ------------------------------------------------------------ expressions
    def parse_expr(self, level=0):
        if level == len(BINARY_LEVELS):
            return self.parse_cast()
        ops = BINARY_LEVELS[level]
        left = self.parse_expr(level + 1)
        while self.tok.kind == "op" and self.tok.value in ops:
            opt = self.advance()
            right = self.parse_expr(level + 1)
            if opt.value in COMPARISONS and isinstance(left, A.Binary) and left.op in COMPARISONS \
                    and not left.paren:
                raise HappError("comparisons cannot be chained; use '&&'", opt.loc)
            left = A.Binary(opt.value, left, right, opt.loc)
        return left

    def parse_cast(self):
        e = self.parse_unary()
        while self.at("kw", "as"):
            t = self.advance()
            e = A.Cast(e, self.parse_type(), t.loc)
        return e

    def parse_unary(self):
        t = self.tok
        if t.kind == "op" and t.value in ("-", "!", "~", "&", "*"):
            self.advance()
            operand = self.parse_unary()
            if t.value == "-" and isinstance(operand, A.IntLit):
                return A.IntLit(-operand.value, t.loc)
            if t.value == "-" and isinstance(operand, A.FloatLit):
                return A.FloatLit(-operand.value, t.loc)
            return A.Unary(t.value, operand, t.loc)
        return self.parse_postfix()

    def parse_postfix(self):
        e = self.parse_primary()
        while True:
            t = self.tok
            if self.accept("op", "["):
                idx = self.parse_expr()
                self.expect_op("]")
                e = A.Index(e, idx, t.loc)
            elif self.accept("op", "."):
                name = self.ident("a field name")
                e = A.Field(e, name.value, name.loc)
            elif self.at_op("("):
                raise HappError("only named functions can be called", t.loc)
            else:
                return e

    def parse_primary(self):
        t = self.tok
        if t.kind == "int":
            self.advance()
            return A.IntLit(t.value, t.loc)
        if t.kind == "float":
            self.advance()
            return A.FloatLit(t.value, t.loc)
        if t.kind == "str":
            self.advance()
            return A.StrLit(t.value, t.loc)
        if t.kind == "kw" and t.value in ("true", "false"):
            self.advance()
            return A.BoolLit(t.value == "true", t.loc)
        if t.kind == "ident" and t.value in ("size_of", "align_of") and self.peek().kind == "op" \
                and self.peek().value == "(":
            self.advance()
            self.advance()
            ty = self.parse_type()
            self.expect_op(")")
            return A.SizeOf(t.value, ty, t.loc)
        if t.kind == "ident":
            self.advance()
            if self.accept("op", "("):
                args = []
                while not self.at_op(")"):
                    at = self.tok
                    if self.tok.kind == "ident" and self.peek().kind == "op" and self.peek().value == ":":
                        name = self.advance().value
                        self.advance()
                        args.append(A.Arg(name, self.parse_expr(), at.loc))
                    else:
                        args.append(A.Arg(None, self.parse_expr(), at.loc))
                    if not self.accept("op", ","):
                        break
                self.expect_op(")")
                return A.Call(t.value, args, t.loc)
            return A.Name(t.value, t.loc)
        if self.accept("op", "("):
            e = self.parse_expr()
            self.expect_op(")")
            if isinstance(e, A.Binary):
                e.paren = True
            return e
        if self.accept("op", "["):
            elems = []
            while not self.at_op("]"):
                elems.append(self.parse_expr())
                if not self.accept("op", ","):
                    break
            self.expect_op("]")
            return A.ArrayLit(elems, t.loc)
        got = t.value if t.kind != "eof" else "end of file"
        raise HappError(f"expected an expression, found {got!r}", t.loc)


def parse(text, path):
    return Parser(text, path).parse_module()
