# HA++ language reference (v0.1)

HA++ is statically typed, compiled ahead of time, and has no garbage collector
or hidden runtime. A file (`.ha`) contains structs, constants, functions and
kernels. Order does not matter.

## Contents

1. [Types](#types)
2. [Literals and constants](#literals-and-constants)
3. [Variables](#variables)
4. [Functions](#functions)
5. [Structs, arrays, pointers](#structs-arrays-pointers)
6. [Vectors and matrices](#vectors-and-matrices)
7. [Operators](#operators)
8. [Conversions](#conversions)
9. [Control flow](#control-flow)
10. [GPU kernels](#gpu-kernels)
11. [Built-in functions](#built-in-functions)
12. [Standard library](#standard-library-happstdmathha)
13. [Memory allocators](#memory-allocators-happlibmemha)
14. [Attributes](#attributes)
15. [Memory layout](#memory-layout)
16. [CPU vs GPU differences](#cpu-vs-gpu-differences)
17. [Imports and names](#imports-and-names)

## Types

| Type | Meaning |
|---|---|
| `bool` | true / false (1 byte in memory) |
| `i8 i16 i32 i64` | signed integers; overflow wraps |
| `u8 u16 u32 u64` | unsigned integers |
| `f16 f32 f64` | half, single, double precision floats |
| `vec2 vec3 vec4` | f32 vectors |
| `ivec2..4` / `uvec2..4` / `hvec2..4` | i32 / u32 / f16 vectors |
| `mat2 mat3 mat4` | f32 square matrices, column-major |
| `[N]T` | fixed array, e.g. `[16]f32`, `[4][3]i32` |
| `*T` | pointer to `T` (memory owned by the host app) |
| `struct Name { … }` | C-compatible struct |

## Literals and constants

```rust
42        0xFF_FF      0b1010      1.5      2.0e-3      true      "text"
```

**Literals take their type from context.** In `x * 2.0`, the `2.0` becomes
`f16` when `x` is `f16`. `let n: u8 = 200` works; `let n: u8 = 300` is an
error. Without context, an integer is `i32` (`i64` or `u64` if it doesn't fit)
and a decimal is `f32`. A float literal too big for its type is an error.

HA++ has no literal suffixes. Write `1.0 as f16` instead of `1.0h`.

```rust
const TILE: u32 = 16;          // typed constant
const SCALE = 0.5;             // untyped: adapts like a literal
const N = TILE * 4;            // constant expressions are evaluated at compile time
```

Constant expressions follow the same rules as run-time code for their type:
integers wrap, shift amounts are masked, integer division truncates, and casts
saturate. Dividing by zero in a constant is a compile error. A constant can
only use other constants, not variables.

## Variables

```rust
let a = 1.0;              // immutable, type inferred (f32)
let b: u32 = 7;
var c = vec3(0.0);        // mutable
var d: [8]f32;            // zero-initialized
c.x += 1.0;               // compound assignment: += -= *= /= %= &= |= ^= <<= >>=
```

Every variable is initialized. `var x: T;` is all zeros.

## Functions

```rust
fn lerp3(a: vec3, b: vec3, t: f32) -> vec3 { return a + (b - a) * t; }   // private
export fn process(data: *f32, n: i64) { ... }    // C ABI, visible to Swift/Kotlin/C/Python
extern fn my_c_function(x: f32) -> f32;          // implemented by the host app (C ABI)
fn main() { ... }                                // only for `happ run`
fn main() -> i32 { return 0; }
```

- **Parameters:** they are read-only values. Structs and arrays are passed
  by value. The compiler avoids the copies when it can.
- **`export` functions:** they take and return only numbers, `bool` and
  pointers, the C ABI subset every language can call. Pass structs as `*T`.
- **Names:** there is no overloading.
- **Recursion:** it is allowed on the CPU but not in code reached from a
  kernel.
- **Missing returns:** a function that can reach its end without returning a
  value is a compile error.

## Structs, arrays, pointers

```rust
struct Gaussian { pos: vec3, scale: vec3, rot: vec4, opacity: f32, sh: [3]f32 }

let g = Gaussian(pos: vec3(0.0), opacity: 1.0);      // named: missing fields are zero
let h = Gaussian(p, s, r, 0.5, [0.1, 0.2, 0.3]);     // positional: all fields
var arr = [1.0, 2.0, 3.0];                           // [3]f32
arr[1] = 5.0;

export fn f(gs: *Gaussian, n: i64) {
    gs[0].opacity = 0.0;          // index a pointer
    gs.opacity = 0.0;             // pointer to struct: same as gs[0].opacity
    let second = gs + 1;          // pointer arithmetic (in elements)
    let p = &arr[2];              // address of a 'var' (or of an element / field)
    let bytes = gs as *u8;        // pointer casts
    let addr = gs as u64;
    let none = 0 as *Gaussian;    // the null pointer
    if gs != none && gs < second { ... }       // pointers of the same type compare (CPU only)
    let n = size_of(Gaussian);    // 56: bytes, known at compile time (align_of gives the alignment)
}
```

`size_of(T)` and `align_of(T)` work for every type, including arrays and
structs defined later in the file. They are integer constants that adapt to
their context like a literal: `let n: u64 = size_of(T) * count` needs no
cast, and `const BYTES = size_of(T);` and `var raw: [size_of(T)]u8;` work.

## Vectors and matrices

```rust
let v = vec3(1.0, 2.0, 3.0);
let w = vec4(v, 1.0);            // components from vectors and scalars
let s = vec3(0.5);               // splat
v.x   v.xy   v.zyx   v.rgb       // swizzles: xyzw or rgba, up to 4
var u = v;  u.xz = vec2(9.0, 8.0);  u[1] = 7.0;      // write lanes
v * 2.0   v + w.xyz   max(v, 0.0)                    // lane-wise, scalars broadcast

let m = mat3(1.0);                                   // identity (diagonal)
let r = mat3(col0, col1, col2);                      // from columns
let q = mat2(1.0, 2.0, 3.0, 4.0);                    // 4 scalars, column-major
m * v    m * m    transpose(m)    m[2] (column)    m[2].y
```

## Operators

From lowest to highest precedence:

| Operators | Notes |
|---|---|
| `\|\|` | short-circuit |
| `&&` | short-circuit |
| `== != < > <= >=` | can't be chained (`a < b < c` is an error); also pointers of the same type, on the CPU |
| `\|` | bitwise (bools too) |
| `^` | bitwise xor |
| `&` | bitwise and |
| `<< >>` | shift amount is masked to the type width; `>>` is arithmetic for signed types |
| `+ -` | also pointer + integer |
| `* / %` | integer `/` and `%` never trap, see below; `%` on floats is `x - y*trunc(x/y)` |
| `as` | conversion |
| unary `- ! ~ & *` | negate, not, bit-not, address-of, dereference |
| `f(x)  a[i]  a.b` | call, index, field / swizzle |

Both operands must have the same type. The exceptions are vector × scalar and
matrix × vector/scalar.

**Integer division is defined for every input**, on the CPU and on GPUs:
`x / 0 = 0`, `x % 0 = x`, `MIN / -1 = MIN` and `MIN % -1 = 0` (`MIN` is the
most negative value, e.g. `-2147483648` for `i32`). A constant divisor costs
nothing extra.

**Float `%` and `mod`** use an exact division, so the result is exact as long as
`|x / y| < 2^23`. For larger ratios the compiler may fuse `y * trunc(x/y)` and
the subtraction into one instruction. The result is then still close relative
to `x`, but it can differ between CPUs, like any fast-math result.

## Conversions

Conversions are only ever explicit, with `x as T`:

- **number ↔ number:**
  - Float → integer rounds toward zero and **saturates**: `1e10 as i32` =
    `2147483647`, `NaN as i32` = `0`. This matches Rust, and ARM64 does it in
    hardware.
  - Wider → narrower integers wrap.
- **bool → integer:** allowed. For the other direction, write `x != 0`.
- **vector ↔ vector:** same length, converted lane by lane: `vec3 as ivec3`.
- **pointers:** `*T ↔ *U`, and `*T ↔ u64/i64`. An integer literal becomes
  an address: `0 as *T` is the null pointer.
- **raw bits:** `f32_bits(x)`, `f32_from_bits(u)`, plus the same for f16/f64.

## Control flow

```rust
if a > b { ... } else if a == b { ... } else { ... }
while cond { ... }
for i in 0..n { ... }            // i goes from 0 to n-1; the end is evaluated once
for j: u32 in 0..16 { ... }      // choose the loop variable type
break;  continue;  return x;
let y = select(c, a, b);         // branch-free choice (both sides are evaluated)
```

Conditions must be `bool`. Integers are not truthy.

## GPU kernels

```rust
@workgroup(64)                  // threads per workgroup: 1-3 sizes, default 64
kernel scale(data: *vec4, n: u32, k: f32) {
    let i = global_id.x;        // uvec3: global_id, local_id, group_id, num_groups, group_size
    if i >= n { return; }       // 'return' ends this thread
    data[i] = data[i] * k;
}
```

**Parameters:**
- Pointer parameters are GPU buffers. On Vulkan they are bindings 0, 1, 2…
  in order; on Metal they are `[[buffer(i)]]`.
- Scalar parameters (`i32`, `u32`, `f32`) are push constants on Vulkan. On
  Metal they are one constant struct at the next buffer index. The generated
  header describes it as `ha_params_<kernel>`.

**What kernels can use:**
- Types: `i32`, `u32`, `f32`, `f16`, `bool`, vectors, matrices, and structs
  and arrays of those.
- Other functions: helper functions called from a kernel are compiled for the
  GPU too. They can't take pointers.
- Not allowed: `print`, `extern`, pointer arithmetic, `&`, 64-bit types and
  recursion.

**GPU-only features:**

```rust
shared tile: [256]f32;               // workgroup memory (top of the kernel body)
barrier();                           // wait for the whole workgroup (+ shared memory fence)
atomic_add(counts, idx, 1);          // also atomic_min / atomic_max / atomic_exchange;
                                     // on *i32/*u32 buffers or shared i32/u32 arrays; returns old value
subgroup_add(x)  subgroup_exclusive_add(x)  subgroup_lane()  subgroup_size()
```

**Every kernel also runs on the CPU.** A kernel that uses none of `shared`,
`barrier` or `subgroup_*` also gets a CPU version:
`<kernel>_cpu(params…, groups_x, groups_y, groups_z)`. It runs every
workgroup on the calling thread. If you define your own `<kernel>_cpu`
function, HA++ uses yours instead.

## Built-in functions

| Group | Functions |
|---|---|
| Float math (scalars & vectors) | `sqrt rsqrt exp exp2 log log2 pow sin cos tan asin acos atan atan2 tanh` |
| Rounding | `floor ceil round trunc fract mod` (`round`: halves away from zero; `mod`: GLSL-style `x - y*floor(x/y)`) |
| Common | `abs sign min max clamp mix step smoothstep fma select` |
| Geometry | `dot cross length distance normalize transpose` |
| Bits | `popcount clz ctz f32_bits f32_from_bits f16_bits f16_from_bits f64_bits f64_from_bits` |
| Packing | `pack_half2(vec2) -> u32`, `unpack_half2(u32) -> vec2` |
| Layout | `size_of(T)`, `align_of(T)`: compile-time integers (see [pointers](#structs-arrays-pointers)) |
| GPU | `barrier atomic_add atomic_min atomic_max atomic_exchange subgroup_*` |
| Debug | `print(a, b, …)`: prints in `happ run`; ignored in libraries (with a build note) |

On CPUs the transcendental functions are the HA++ standard library: accurate
to a few ulp, with no libm. On GPUs they are the GPU's own instructions, which
are faster and slightly less accurate.

## Standard library (`happ/std/math.ha`)

These are always available:
- **Constants:** `PI`, `TAU`, `E`.
- **Activation functions:** `sigmoid`, `relu`, `silu`, `softplus`, `gelu`
  (tanh form).
- **Small helpers:** `saturate`, `radians`, `degrees`.
- **Rotation:** `quat_to_mat3(vec4(w, x, y, z))`.
- **Colour packing:** `pack_unorm4(vec4) -> u32` and `unpack_unorm4(u32) -> vec4`
  (RGBA8).

If you define a function or constant with the same name, yours is used.

## Memory allocators (`happ/lib/mem.ha`)

HA++ has no heap of its own. The host app gives an allocator one block of
memory, and `import "mem.ha";` provides three ways to carve it up. None of
them calls the operating system. Every function has a small, fixed
worst-case cost, so it is safe inside a frame loop.

| Allocator | Use it for | Cost |
|---|---|---|
| `Arena` | per-frame scratch memory, freed all at once | a few instructions; no header per allocation |
| `Pool` | many objects of one size | O(1) alloc/free; starts in constant time |
| `Tlsf` | mixed sizes freed in any order (a general `malloc`) | O(1) malloc/free/realloc; 16-byte aligned; 16 bytes per block |

```rust
import "mem.ha";

export fn frame(scratch: *u8, bytes: u64) {
    var a: Arena;
    arena_init(&a, scratch, bytes);
    let keys = arena_alloc(&a, 4 * 1000, 16) as *u32;    // null (0 as *u32) if it doesn't fit
    ...
    arena_reset(&a);                                      // everything freed at once
}

export fn make_heap(mem: *u8, bytes: u64) -> *Tlsf {
    return tlsf_create(mem, bytes);         // the control block lives at the start of `mem`
}
export fn heap_alloc(t: *Tlsf, n: u64) -> *u8 { return tlsf_malloc(t, n); }
export fn heap_free(t: *Tlsf, p: *u8) { tlsf_free(t, p); }
```

| Functions | |
|---|---|
| Arena | `arena_init(a, mem, size)`, `arena_alloc(a, size, align)`, `arena_mark(a)`, `arena_reset_to(a, mark)`, `arena_reset(a)`; `a.peak` is the most memory it ever held |
| Pool | `pool_init(p, mem, bytes, block, align) -> count`, `pool_alloc(p)`, `pool_free(p, ptr)`, `pool_owns(p, ptr)`, `pool_reset(p)` |
| TLSF | `tlsf_init(t, mem, bytes)` or `tlsf_create(mem, bytes)`, `tlsf_add_pool`, `tlsf_malloc`, `tlsf_free`, `tlsf_realloc`, `tlsf_block_size(ptr)`, `tlsf_largest_free(t)`, `tlsf_check(t)` (0 = all invariants hold) |

TLSF is the Two-Level Segregated Fit allocator from real-time systems
(Masmano et al., 2004), laid out like Matthew Conte's C version. Free blocks
sit in 24 × 32 size-class lists with two bitmaps over them, so finding a
block that fits takes two count-trailing-zeros instructions instead of a
search. A freed block merges with its free neighbours right away. The
allocators are not thread-safe: give each thread its own.

`tests/test_alloc.py` checks them with randomized runs against a model
(overlap, alignment, contents kept, realloc copies, invariants after every
operation). `tests/mutate_alloc.py` plants 20 typical allocator bugs and
checks that the tests catch every one.

## Attributes

| Attribute | On | Effect |
|---|---|---|
| `@workgroup(x, y, z)` | kernel | threads per workgroup |
| `@inline` / `@noinline` | fn | inlining hint |
| `@strict` | fn | exact IEEE float math (no fast-math) in this function; `happ build --strict-math` for everything |

## Memory layout

Every type is laid out like the matching C type. Vectors and matrices are
**tightly packed floats**:

| Type | Layout |
|---|---|
| `vec3` | 12 bytes, alignment 4 |
| `vec4` | 16 bytes, alignment 4 |
| `mat4` | 64 bytes, column-major |
| `f16` | 2 bytes |

So one buffer of structs has the same bytes on:
- the CPU: C, Swift, Kotlin `ByteBuffer`, Python `ctypes`/NumPy;
- Vulkan: the generated GLSL uses std430 with scalar-only members;
- Metal: the generated code uses `packed_float3` and similar types.

The generated header checks every struct size with `static_assert`. The
Kotlin/Java class has `SIZEOF_NAME` and `OFFSET_NAME_FIELD` constants, plus
`WORKGROUP_KERNEL_X/Y/Z`. Two names that would give the same constant, such as
struct `a_b` field `c` and struct `a` field `b_c`, are a build error.

## CPU vs GPU differences

| | CPU | GPU |
|---|---|---|
| float → int out of range | saturates | undefined |
| integer division by 0 or `MIN / -1` | defined (see Operators) | the same, guarded in the generated code |
| `exp`, `sin`, … | HA++ std library (≤ a few ulp) | GPU instructions (driver precision) |
| f16 math functions (`sqrt`, `dot`, `round`, …) | f16 | Vulkan: computed in f32, then rounded to f16 (can be more precise than the CPU); Metal: native `half` |
| `atomic_*` inside `a[...] += x` or `select` | runs once | runs once (the index is computed once, into a temporary) |
| fast-math | on (`@strict` to disable) | the GPU compiler's own rules |

## Imports and names

```rust
import "common.ha";     // path relative to this file; everything shares one namespace
import "mem.ha";        // not next to this file: a library that comes with HA++ (happ/lib/)
```

A name can't be used twice. User definitions replace std-library definitions
with the same name. Built-in names like `dot` or `exp` can't be redefined.

Names are ASCII letters, digits and `_`. Some names end up in generated C,
Swift, Kotlin, Java, Python and Metal code: exported functions, kernels, their
parameters, structs and fields. These can't be a keyword of any of those
languages (`class`, `default`, `val`, `self`, `device`, …), because the
generated code would not compile. The compiler says which language reserves the
name. Names starting with `ha_` are also reserved, because the generated code
uses them.
