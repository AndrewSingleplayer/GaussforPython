// Memory allocators for HA++: Arena, Pool and TLSF.
//
//     import "mem.ha";
//
// HA++ has no heap of its own. The host app hands each allocator one block of
// memory (a buffer, an mmap, a static array) and the allocator carves it up.
// Nothing here calls the operating system, so every function has a small,
// fixed worst-case cost. That is what a renderer needs inside its frame loop,
// where one slow malloc is a dropped frame.
//
// Which one to use:
//   Arena  per-frame scratch memory. An allocation is a few instructions and
//          there is no per-allocation overhead; everything is freed at once
//          with arena_reset (or back to an arena_mark).
//   Pool   many objects of one size (nodes, tiles, requests). O(1) alloc and
//          free, no fragmentation, no header per object.
//   TLSF   everything else: mixed sizes, freed in any order. O(1) malloc, free
//          and realloc with a bounded worst case, 16-byte aligned results,
//          16 bytes of overhead per block.
//
// All allocation functions return the null pointer (0 as *u8) when memory runs
// out. None of them are thread-safe: give each thread its own allocator.

// ================================================================ Arena

struct Arena {
    base: *u8,
    size: u64,
    used: u64,
    peak: u64,          // highest `used` since arena_init: how big the arena really needs to be
}

fn arena_init(a: *Arena, mem: *u8, size: u64) {
    a.base = mem;
    a.size = size;
    a.used = 0;
    a.peak = 0;
}

// `align` must be a power of two (1, 2, 4, 8, 16, ...); otherwise the result is null.
fn arena_alloc(a: *Arena, size: u64, align: u64) -> *u8 {
    if align == 0 || (align & (align - 1)) != 0 { return 0 as *u8; }
    let cur = (a.base as u64) + a.used;
    let pad = (0 - cur) & (align - 1);          // bytes up to the next multiple of align
    let room = a.size - a.used;
    if pad > room || size > room - pad { return 0 as *u8; }
    a.used += pad + size;
    a.peak = max(a.peak, a.used);
    return (cur + pad) as *u8;
}

// A mark is the arena's fill level; arena_reset_to(mark) frees everything allocated after it.
fn arena_mark(a: *Arena) -> u64 { return a.used; }

fn arena_reset_to(a: *Arena, mark: u64) {
    if mark <= a.used { a.used = mark; }
}

fn arena_reset(a: *Arena) { a.used = 0; }

// ================================================================ Pool

struct Pool {
    base: *u8,          // first block
    block: u64,         // bytes per block: a multiple of the alignment, at least 8
    count: u64,         // blocks in the pool
    fresh: u64,         // blocks fresh..count-1 were never handed out
    free: *u8,          // returned blocks; each keeps the next one in its first 8 bytes
    used: u64,          // blocks handed out right now
}

// Returns the number of blocks, 0 if the memory can't hold one (or `align` isn't a power of two).
// Nothing is written into `mem` here, so a pool of any size starts in constant time.
fn pool_init(p: *Pool, mem: *u8, bytes: u64, block: u64, align: u64) -> u64 {
    let al = max(align, 8);
    var n: u64 = 0;
    if (al & (al - 1)) == 0 && block <= bytes {
        let bs = (max(block, 8) + al - 1) & (0 - al);
        let start = mem as u64;
        let pad = (0 - start) & (al - 1);
        if pad < bytes { n = (bytes - pad) / bs; }
        p.base = (start + pad) as *u8;
        p.block = bs;
    } else {
        p.base = mem;
        p.block = 8;
    }
    p.count = n;
    p.fresh = 0;
    p.free = 0 as *u8;
    p.used = 0;
    return n;
}

fn pool_alloc(p: *Pool) -> *u8 {
    let head = p.free;
    if head as u64 != 0 {
        p.free = (head as **u8)[0];
        p.used += 1;
        return head;
    }
    if p.fresh < p.count {
        let b = p.base + p.fresh * p.block;
        p.fresh += 1;
        p.used += 1;
        return b;
    }
    return 0 as *u8;
}

// `ptr` must come from pool_alloc on this pool (or be null).
fn pool_free(p: *Pool, ptr: *u8) {
    if ptr as u64 == 0 { return; }
    (ptr as **u8)[0] = p.free;
    p.free = ptr;
    p.used -= 1;
}

// true if `ptr` is the start of a block this pool has handed out at some point.
fn pool_owns(p: *Pool, ptr: *u8) -> bool {
    let off = (ptr as u64) - (p.base as u64);
    return ptr >= p.base && off < p.fresh * p.block && off % p.block == 0;
}

fn pool_reset(p: *Pool) {
    p.fresh = 0;
    p.free = 0 as *u8;
    p.used = 0;
}

// ================================================================ TLSF
//
// Two-Level Segregated Fit (Masmano, Ripoll, Crespo and Real, ECRTS 2004),
// organised like Matthew Conte's widely used C implementation.
//
// Free blocks sit in 24 x 32 lists. The first level is the power of two of
// the block size (floor(log2)); the second level splits each power of two
// into 32 equal ranges. Two bitmaps say which lists are non-empty, so finding
// a block that fits is two count-trailing-zeros instructions, never a search.
// A freed block merges with its free neighbours in memory right away, so free
// space doesn't stay chopped into small pieces.
//
// Memory layout (every size is a multiple of 16, every result 16-byte aligned):
//
//   [prev_phys 8B][size|flags 8B][payload: size bytes ...][next block's header]
//                                 ^ pointer given to the caller
//
// A free block keeps its two list links in the first 16 payload bytes. Each
// pool ends with a used, zero-size block, so merging never runs past the end.

const TLSF_HEADER: u64 = 16;         // prev_phys + size
const TLSF_MIN_PAYLOAD: u64 = 16;    // room for the two free-list links
const TLSF_SL_LOG2: u64 = 5;         // 32 second-level lists per power of two
const TLSF_SL_COUNT: u32 = 32;
const TLSF_FL_COUNT: u32 = 24;       // first levels: sizes below 512 bytes, then 2^9 .. 2^31
const TLSF_SMALL: u64 = 512;         // below this, the lists are 16 bytes apart
const TLSF_MAX_BLOCK: u64 = 4294967280;   // 2^32 - 16: the largest block the lists can hold
const TLSF_MAX_POOLS: u32 = 16;
const TLSF_FREE: u64 = 1;            // flag bit: this block is free
const TLSF_PREV_FREE: u64 = 2;       // flag bit: the block just before it in memory is free

struct TlsfBlock {
    prev_phys: *TlsfBlock,          // the block just before this one in memory (null for a pool's first)
    size: u64,                      // payload bytes | TLSF_FREE | TLSF_PREV_FREE
    next_free: *TlsfBlock,          // free blocks only: the list this block is in
    prev_free: *TlsfBlock,
}

struct Tlsf {
    fl_bitmap: u32,                             // bit f: some list in first level f is non-empty
    sl_bitmap: [TLSF_FL_COUNT]u32,              // bit s of word f: list (f, s) is non-empty
    heads: [TLSF_FL_COUNT * TLSF_SL_COUNT]*TlsfBlock,
    pools: [TLSF_MAX_POOLS]*TlsfBlock,          // first block of each pool
    pool_count: u32,
    used: u64,                                  // payload bytes handed out now
    peak: u64,                                  // most payload bytes ever handed out at once
}

fn tlsf_size(b: *TlsfBlock) -> u64 { return b.size & (0 - 16); }

fn tlsf_next(b: *TlsfBlock) -> *TlsfBlock {
    return ((b as *u8) + TLSF_HEADER + tlsf_size(b)) as *TlsfBlock;
}

// (first level, second level) of the list that holds blocks of this size.
fn tlsf_mapping(size: u64) -> uvec2 {
    if size < TLSF_SMALL {
        return uvec2(0, (size >> 4) as u32);
    }
    let top = 63 - clz(size);                                   // floor(log2(size)), 9..31
    let sl = ((size >> (top - TLSF_SL_LOG2)) as u32) ^ TLSF_SL_COUNT;
    return uvec2((top - 8) as u32, sl);
}

// The first list whose blocks are all at least `size` bytes: round up to the next list start.
fn tlsf_mapping_search(size: u64) -> uvec2 {
    var s = size;
    if s >= TLSF_SMALL {
        let top = 63 - clz(s);
        s += (1 << (top - TLSF_SL_LOG2)) - 1;
    }
    return tlsf_mapping(s);
}

// The first non-empty list at (fl, sl) or after it; x = TLSF_FL_COUNT when there is none.
fn tlsf_find(t: *Tlsf, fl: u32, sl: u32) -> uvec2 {
    var f = fl;
    var sl_map = t.sl_bitmap[f] & (0xFFFFFFFF << sl);
    if sl_map == 0 {
        let fl_map = t.fl_bitmap & (0xFFFFFFFF << (f + 1));
        if fl_map == 0 { return uvec2(TLSF_FL_COUNT, 0); }
        f = ctz(fl_map);
        sl_map = t.sl_bitmap[f];
    }
    return uvec2(f, ctz(sl_map));
}

fn tlsf_insert(t: *Tlsf, b: *TlsfBlock) {
    let m = tlsf_mapping(tlsf_size(b));
    let i = m.x * TLSF_SL_COUNT + m.y;
    let head = t.heads[i];
    b.next_free = head;
    b.prev_free = 0 as *TlsfBlock;
    if head as u64 != 0 { head.prev_free = b; }
    t.heads[i] = b;
    t.fl_bitmap |= 1 << m.x;
    t.sl_bitmap[m.x] |= 1 << m.y;
}

fn tlsf_remove(t: *Tlsf, b: *TlsfBlock) {
    let next = b.next_free;
    let prev = b.prev_free;
    if next as u64 != 0 { next.prev_free = prev; }
    if prev as u64 != 0 {
        prev.next_free = next;
        return;
    }
    let m = tlsf_mapping(tlsf_size(b));
    t.heads[m.x * TLSF_SL_COUNT + m.y] = next;
    if next as u64 == 0 {
        t.sl_bitmap[m.x] &= ~(1 << m.y);
        if t.sl_bitmap[m.x] == 0 { t.fl_bitmap &= ~(1 << m.x); }
    }
}

// Payload size for a request of `size` bytes; 0 if no block can be that big.
fn tlsf_adjust(size: u64) -> u64 {
    if size > TLSF_MAX_BLOCK { return 0; }
    return max((size + 15) & (0 - 16), TLSF_MIN_PAYLOAD);
}

// Give a free block (b) or a used block's tail back to the lists: merge with free
// neighbours, mark free, insert. Doesn't touch t.used.
fn tlsf_release(t: *Tlsf, block: *TlsfBlock) {
    var b = block;
    if (b.size & TLSF_PREV_FREE) != 0 {
        let prev = b.prev_phys;
        tlsf_remove(t, prev);
        prev.size += TLSF_HEADER + tlsf_size(b);        // sizes are multiples of 16: flags stay as they are
        b = prev;
    }
    let next = tlsf_next(b);
    if (next.size & TLSF_FREE) != 0 {
        tlsf_remove(t, next);
        b.size += TLSF_HEADER + tlsf_size(next);
    }
    b.size |= TLSF_FREE;
    let after = tlsf_next(b);
    after.prev_phys = b;
    after.size |= TLSF_PREV_FREE;
    tlsf_insert(t, b);
}

// Cut used block b down to `need` payload bytes when the tail can hold a block of its own.
fn tlsf_trim(t: *Tlsf, b: *TlsfBlock, need: u64) {
    let size = tlsf_size(b);
    if size < need + TLSF_HEADER + TLSF_MIN_PAYLOAD { return; }
    let rest = ((b as *u8) + TLSF_HEADER + need) as *TlsfBlock;
    rest.prev_phys = b;
    rest.size = size - need - TLSF_HEADER;              // used for now; the block before it (b) is used
    b.size = need | (b.size & TLSF_PREV_FREE);
    t.used -= size - need;
    tlsf_release(t, rest);                              // also points the next block back at `rest`
}

// Adds another block of memory to the allocator (up to 16 pools). false if it is too small.
fn tlsf_add_pool(t: *Tlsf, mem: *u8, bytes: u64) -> bool {
    if t.pool_count >= TLSF_MAX_POOLS { return false; }
    let start = mem as u64;
    let pad = (0 - start) & 15;
    if bytes < pad + 2 * TLSF_HEADER + TLSF_MIN_PAYLOAD { return false; }
    let size = min((bytes - pad - 2 * TLSF_HEADER) & (0 - 16), TLSF_MAX_BLOCK);
    let b = (start + pad) as *TlsfBlock;
    b.prev_phys = 0 as *TlsfBlock;
    b.size = size | TLSF_FREE;
    let end = tlsf_next(b);
    end.prev_phys = b;
    end.size = TLSF_PREV_FREE;                          // size 0 and used: nothing merges past it
    t.pools[t.pool_count] = b;
    t.pool_count += 1;
    tlsf_insert(t, b);
    return true;
}

// Sets up `t` (kept outside the managed memory) with one pool. false if the memory is too small.
fn tlsf_init(t: *Tlsf, mem: *u8, bytes: u64) -> bool {
    t.fl_bitmap = 0;
    for f in 0..TLSF_FL_COUNT { t.sl_bitmap[f] = 0; }
    for i in 0..TLSF_FL_COUNT * TLSF_SL_COUNT { t.heads[i] = 0 as *TlsfBlock; }
    for p in 0..TLSF_MAX_POOLS { t.pools[p] = 0 as *TlsfBlock; }
    t.pool_count = 0;
    t.used = 0;
    t.peak = 0;
    return tlsf_add_pool(t, mem, bytes);
}

// One buffer for everything: the Tlsf control block goes at the start of `mem`,
// the rest becomes the pool. Null if the memory is too small.
fn tlsf_create(mem: *u8, bytes: u64) -> *Tlsf {
    let pad = (0 - (mem as u64)) & 15;
    let control = pad + ((size_of(Tlsf) + 15) & (0 - 16));
    if bytes <= control { return 0 as *Tlsf; }
    let t = (mem + pad) as *Tlsf;
    if !tlsf_init(t, mem + control, bytes - control) { return 0 as *Tlsf; }
    return t;
}

fn tlsf_malloc(t: *Tlsf, size: u64) -> *u8 {
    let need = tlsf_adjust(size);
    if need == 0 { return 0 as *u8; }
    let want = tlsf_mapping_search(need);
    if want.x >= TLSF_FL_COUNT { return 0 as *u8; }
    let at = tlsf_find(t, want.x, want.y);
    if at.x >= TLSF_FL_COUNT { return 0 as *u8; }
    let b = t.heads[at.x * TLSF_SL_COUNT + at.y];
    // b is the head of its list: unlink it
    let next = b.next_free;
    t.heads[at.x * TLSF_SL_COUNT + at.y] = next;
    if next as u64 != 0 {
        next.prev_free = 0 as *TlsfBlock;
    } else {
        t.sl_bitmap[at.x] &= ~(1 << at.y);
        if t.sl_bitmap[at.x] == 0 { t.fl_bitmap &= ~(1 << at.x); }
    }
    // mark it used (the block after it no longer has a free neighbour), then give back the tail
    b.size &= ~TLSF_FREE;
    let after = tlsf_next(b);
    after.size &= ~TLSF_PREV_FREE;
    t.used += tlsf_size(b);
    tlsf_trim(t, b, need);
    t.peak = max(t.peak, t.used);
    return (b as *u8) + TLSF_HEADER;
}

// `p` must come from this allocator (or be null).
fn tlsf_free(t: *Tlsf, p: *u8) {
    if p as u64 == 0 { return; }
    let b = (p - TLSF_HEADER) as *TlsfBlock;
    t.used -= tlsf_size(b);
    tlsf_release(t, b);
}

// Resize an allocation, in place when the block itself or the free block after it has room.
// Returns the (possibly moved) pointer, or null with `p` untouched when there is no memory.
// realloc(null, n) is malloc(n); realloc(p, 0) frees p and returns null.
fn tlsf_realloc(t: *Tlsf, p: *u8, size: u64) -> *u8 {
    if p as u64 == 0 { return tlsf_malloc(t, size); }
    if size == 0 {
        tlsf_free(t, p);
        return 0 as *u8;
    }
    let need = tlsf_adjust(size);
    if need == 0 { return 0 as *u8; }
    let b = (p - TLSF_HEADER) as *TlsfBlock;
    let cur = tlsf_size(b);
    if need <= cur {
        tlsf_trim(t, b, need);
        return p;
    }
    let next = tlsf_next(b);
    if (next.size & TLSF_FREE) != 0 && cur + TLSF_HEADER + tlsf_size(next) >= need {
        tlsf_remove(t, next);
        b.size += TLSF_HEADER + tlsf_size(next);
        let after = tlsf_next(b);
        after.prev_phys = b;
        after.size &= ~TLSF_PREV_FREE;
        t.used += tlsf_size(b) - cur;
        tlsf_trim(t, b, need);
        t.peak = max(t.peak, t.used);
        return p;
    }
    let q = tlsf_malloc(t, size);
    if q as u64 == 0 { return q; }
    let src = p as *u64;
    let dst = q as *u64;
    for i in 0..cur / 8 { dst[i] = src[i]; }
    tlsf_free(t, p);
    return q;
}

// Usable bytes behind a pointer from tlsf_malloc (at least what was asked for).
fn tlsf_block_size(p: *u8) -> u64 {
    return tlsf_size((p - TLSF_HEADER) as *TlsfBlock);
}

// Size of the largest block tlsf_malloc can return right now (0 when memory is full).
fn tlsf_largest_free(t: *Tlsf) -> u64 {
    if t.fl_bitmap == 0 { return 0; }
    let f = 31 - clz(t.fl_bitmap);
    let s = 31 - clz(t.sl_bitmap[f]);
    var best: u64 = 0;
    var b = t.heads[f * TLSF_SL_COUNT + s];
    while b as u64 != 0 {
        best = max(best, tlsf_size(b));
        b = b.next_free;
    }
    return best;
}

// Walks every pool and every list and checks the allocator's invariants.
// 0 if everything is consistent, otherwise the number of the first check that failed.
fn tlsf_check(t: *Tlsf) -> i32 {
    var free_blocks: u64 = 0;
    var used_bytes: u64 = 0;
    for pi in 0..t.pool_count {
        var b = t.pools[pi];
        var prev = 0 as *TlsfBlock;
        var prev_free = false;
        while true {
            if b.prev_phys != prev { return 1; }                          // physical back-link
            let is_free = (b.size & TLSF_FREE) != 0;
            if ((b.size & TLSF_PREV_FREE) != 0) != prev_free { return 2; }    // flag matches the block before
            let size = tlsf_size(b);
            if size == 0 {
                if is_free { return 3; }                                  // the end marker stays used
                break;
            }
            if size < TLSF_MIN_PAYLOAD { return 4; }
            if (((b as u64) + TLSF_HEADER) & 15) != 0 { return 5; }       // 16-byte aligned payload
            if is_free {
                if prev_free { return 6; }                                // free neighbours are always merged
                free_blocks += 1;
                let m = tlsf_mapping(size);
                var q = t.heads[m.x * TLSF_SL_COUNT + m.y];
                while q as u64 != 0 && q != b { q = q.next_free; }
                if q as u64 == 0 { return 7; }                            // it is in the list for its size
            } else {
                used_bytes += size;
            }
            prev = b;
            prev_free = is_free;
            b = tlsf_next(b);
        }
    }
    var listed: u64 = 0;
    for f in 0..TLSF_FL_COUNT {
        for s in 0..TLSF_SL_COUNT {
            var q = t.heads[f * TLSF_SL_COUNT + s];
            if (q as u64 != 0) != (((t.sl_bitmap[f] >> s) & 1) == 1) { return 8; }
            var back = 0 as *TlsfBlock;
            while q as u64 != 0 {
                if q.prev_free != back { return 9; }
                if (q.size & TLSF_FREE) == 0 { return 10; }
                let m = tlsf_mapping(tlsf_size(q));
                if m.x != f || m.y != s { return 11; }
                listed += 1;
                back = q;
                q = q.next_free;
            }
        }
        if (((t.fl_bitmap >> f) & 1) == 1) != (t.sl_bitmap[f] != 0) { return 12; }
    }
    if listed != free_blocks { return 13; }
    if used_bytes != t.used { return 14; }
    return 0;
}
