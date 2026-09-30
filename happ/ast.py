"""Syntax tree for HA++.

Every node has a `loc`. Expression nodes also get a `ty` filled in by the
type checker, which the CPU and GPU back ends both read.
"""


class Node:
    __slots__ = ("loc",)


# ---------------------------------------------------------------- type syntax

class TName(Node):
    __slots__ = ("name",)

    def __init__(self, name, loc):
        self.name, self.loc = name, loc


class TPtr(Node):
    __slots__ = ("inner",)

    def __init__(self, inner, loc):
        self.inner, self.loc = inner, loc


class TArray(Node):
    __slots__ = ("inner", "size")

    def __init__(self, inner, size, loc):
        self.inner, self.size, self.loc = inner, size, loc


# ---------------------------------------------------------------- items

class Module(Node):
    __slots__ = ("path", "items", "is_std")

    def __init__(self, path, items, is_std=False):
        self.path, self.items, self.is_std, self.loc = path, items, is_std, None


class Import(Node):
    __slots__ = ("path",)

    def __init__(self, path, loc):
        self.path, self.loc = path, loc


class ConstDef(Node):
    __slots__ = ("name", "type", "value", "ty", "val")

    def __init__(self, name, type_, value, loc):
        self.name, self.type, self.value, self.loc = name, type_, value, loc
        self.ty = None
        self.val = None


class StructDef(Node):
    __slots__ = ("name", "fields", "ty")

    def __init__(self, name, fields, loc):
        self.name, self.fields, self.loc = name, fields, loc   # fields: [(name, TypeExpr, loc)]
        self.ty = None


class Param(Node):
    __slots__ = ("name", "type", "ty")

    def __init__(self, name, type_, loc):
        self.name, self.type, self.loc = name, type_, loc
        self.ty = None


class FnDef(Node):
    __slots__ = ("name", "params", "ret", "body", "is_export", "is_extern",
                 "is_kernel", "attrs", "ret_ty", "is_std", "sig", "uses")

    def __init__(self, name, params, ret, body, loc, is_export=False,
                 is_extern=False, is_kernel=False, attrs=None):
        self.name, self.params, self.ret, self.body, self.loc = name, params, ret, body, loc
        self.is_export, self.is_extern, self.is_kernel = is_export, is_extern, is_kernel
        self.attrs = attrs or {}
        self.ret_ty = None
        self.is_std = False
        self.sig = None
        self.uses = set()   # features used in the body, filled by the checker


# ---------------------------------------------------------------- statements

class Block(Node):
    __slots__ = ("stmts",)

    def __init__(self, stmts, loc):
        self.stmts, self.loc = stmts, loc


class Let(Node):
    __slots__ = ("name", "type", "value", "mutable", "ty")

    def __init__(self, name, type_, value, mutable, loc):
        self.name, self.type, self.value, self.mutable, self.loc = name, type_, value, mutable, loc
        self.ty = None


class Shared(Node):
    __slots__ = ("name", "type", "ty")

    def __init__(self, name, type_, loc):
        self.name, self.type, self.loc = name, type_, loc
        self.ty = None


class Assign(Node):
    __slots__ = ("target", "op", "value")

    def __init__(self, target, op, value, loc):
        self.target, self.op, self.value, self.loc = target, op, value, loc


class If(Node):
    __slots__ = ("cond", "then", "els")

    def __init__(self, cond, then, els, loc):
        self.cond, self.then, self.els, self.loc = cond, then, els, loc


class While(Node):
    __slots__ = ("cond", "body")

    def __init__(self, cond, body, loc):
        self.cond, self.body, self.loc = cond, body, loc


class For(Node):
    __slots__ = ("var", "start", "end", "body", "ty", "vtype")

    def __init__(self, var, start, end, body, loc):
        self.var, self.start, self.end, self.body, self.loc = var, start, end, body, loc
        self.ty = None
        self.vtype = None


class Return(Node):
    __slots__ = ("value",)

    def __init__(self, value, loc):
        self.value, self.loc = value, loc


class Break(Node):
    __slots__ = ()

    def __init__(self, loc):
        self.loc = loc


class Continue(Node):
    __slots__ = ()

    def __init__(self, loc):
        self.loc = loc


class ExprStmt(Node):
    __slots__ = ("expr",)

    def __init__(self, expr, loc):
        self.expr, self.loc = expr, loc


# ---------------------------------------------------------------- expressions

class Expr(Node):
    __slots__ = ("ty",)


class IntLit(Expr):
    __slots__ = ("value",)

    def __init__(self, value, loc):
        self.value, self.loc, self.ty = value, loc, None


class SizeOf(IntLit):
    """size_of(T) / align_of(T): an integer literal whose value the checker fills in from T's layout."""
    __slots__ = ("which", "type")

    def __init__(self, which, type_, loc):
        super().__init__(None, loc)
        self.which, self.type = which, type_


class FloatLit(Expr):
    __slots__ = ("value",)

    def __init__(self, value, loc):
        self.value, self.loc, self.ty = value, loc, None


class BoolLit(Expr):
    __slots__ = ("value",)

    def __init__(self, value, loc):
        self.value, self.loc, self.ty = value, loc, None


class StrLit(Expr):
    __slots__ = ("value",)

    def __init__(self, value, loc):
        self.value, self.loc, self.ty = value, loc, None


class Name(Expr):
    __slots__ = ("id", "ref")

    def __init__(self, id_, loc):
        self.id, self.loc, self.ty = id_, loc, None
        self.ref = None     # Symbol resolved by the checker


class Unary(Expr):
    __slots__ = ("op", "operand")

    def __init__(self, op, operand, loc):
        self.op, self.operand, self.loc, self.ty = op, operand, loc, None


class Binary(Expr):
    __slots__ = ("op", "left", "right", "paren")

    def __init__(self, op, left, right, loc):
        self.op, self.left, self.right, self.loc, self.ty = op, left, right, loc, None
        self.paren = False      # written as (a op b): a comparison in parentheses may be compared again


class Cast(Expr):
    __slots__ = ("expr", "type", "target")

    def __init__(self, expr, type_, loc):
        self.expr, self.type, self.loc, self.ty = expr, type_, loc, None
        self.target = None


class Arg(Node):
    __slots__ = ("name", "value")

    def __init__(self, name, value, loc):
        self.name, self.value, self.loc = name, value, loc


class Call(Expr):
    __slots__ = ("name", "args", "kind", "target", "info")

    def __init__(self, name, args, loc):
        self.name, self.args, self.loc, self.ty = name, args, loc, None
        self.kind = None     # 'fn' | 'builtin' | 'ctor_vec' | 'ctor_mat' | 'ctor_struct'
        self.target = None   # FnDef or Type
        self.info = None     # builtin detail


class Index(Expr):
    __slots__ = ("base", "index")

    def __init__(self, base, index, loc):
        self.base, self.index, self.loc, self.ty = base, index, loc, None


class Field(Expr):
    __slots__ = ("base", "name", "kind", "info")

    def __init__(self, base, name, loc):
        self.base, self.name, self.loc, self.ty = base, name, loc, None
        self.kind = None     # 'field' | 'swizzle' | 'ptrfield'
        self.info = None     # field index or swizzle lane list


class ArrayLit(Expr):
    __slots__ = ("elems",)

    def __init__(self, elems, loc):
        self.elems, self.loc, self.ty = elems, loc, None
