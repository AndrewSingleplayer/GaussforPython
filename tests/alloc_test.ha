// C entry points to happ/lib/mem.ha, for the randomized allocator tests (tests/test_happ.py)
// and the allocator benchmark (bench/alloc_bench.py).
import "mem.ha";

export fn t_sizeof_tlsf() -> u64 { return size_of(Tlsf); }
export fn t_tlsf_init(t: *Tlsf, mem: *u8, bytes: u64) -> bool { return tlsf_init(t, mem, bytes); }
export fn t_tlsf_add_pool(t: *Tlsf, mem: *u8, bytes: u64) -> bool { return tlsf_add_pool(t, mem, bytes); }
export fn t_tlsf_create(mem: *u8, bytes: u64) -> *Tlsf { return tlsf_create(mem, bytes); }
export fn t_tlsf_malloc(t: *Tlsf, size: u64) -> *u8 { return tlsf_malloc(t, size); }
export fn t_tlsf_free(t: *Tlsf, ptr: *u8) { tlsf_free(t, ptr); }
export fn t_tlsf_realloc(t: *Tlsf, ptr: *u8, size: u64) -> *u8 { return tlsf_realloc(t, ptr, size); }
export fn t_tlsf_block_size(ptr: *u8) -> u64 { return tlsf_block_size(ptr); }
export fn t_tlsf_largest_free(t: *Tlsf) -> u64 { return tlsf_largest_free(t); }
export fn t_tlsf_used(t: *Tlsf) -> u64 { return t.used; }
export fn t_tlsf_check(t: *Tlsf) -> i32 { return tlsf_check(t); }

export fn t_sizeof_arena() -> u64 { return size_of(Arena); }
export fn t_arena_init(a: *Arena, mem: *u8, size: u64) { arena_init(a, mem, size); }
export fn t_arena_alloc(a: *Arena, size: u64, align: u64) -> *u8 { return arena_alloc(a, size, align); }
export fn t_arena_mark(a: *Arena) -> u64 { return arena_mark(a); }
export fn t_arena_reset_to(a: *Arena, mark: u64) { arena_reset_to(a, mark); }
export fn t_arena_peak(a: *Arena) -> u64 { return a.peak; }

export fn t_sizeof_pool() -> u64 { return size_of(Pool); }
export fn t_pool_init(p: *Pool, mem: *u8, bytes: u64, block: u64, align: u64) -> u64 {
    return pool_init(p, mem, bytes, block, align);
}
export fn t_pool_alloc(p: *Pool) -> *u8 { return pool_alloc(p); }
export fn t_pool_free(p: *Pool, ptr: *u8) { pool_free(p, ptr); }
export fn t_pool_owns(p: *Pool, ptr: *u8) -> bool { return pool_owns(p, ptr); }
export fn t_pool_used(p: *Pool) -> u64 { return p.used; }
