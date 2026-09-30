"""Mutation test for the allocator tests: plant a bug in happ/lib/mem.ha, check that
tests/test_alloc.py fails. A test suite that passes on buggy code proves nothing.

    python3 tests/mutate_alloc.py          # prints one line per planted bug; exit 1 if any survives
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from happ.driver import build  # noqa: E402
from happ.targets import host_target  # noqa: E402

# (what the bug is, text in mem.ha, replacement)
MUTATIONS = [
    ("TLSF free doesn't merge with the next block",
     "    let next = tlsf_next(b);\n    if (next.size & TLSF_FREE) != 0 {",
     "    let next = tlsf_next(b);\n    if false {"),
    ("TLSF free doesn't merge with the previous block",
     "    if (b.size & TLSF_PREV_FREE) != 0 {\n        let prev = b.prev_phys;",
     "    if false {\n        let prev = b.prev_phys;"),
    ("TLSF splits off tails too small to hold a block",
     "    if size < need + TLSF_HEADER + TLSF_MIN_PAYLOAD { return; }",
     "    if size < need + TLSF_MIN_PAYLOAD { return; }"),
    ("TLSF search doesn't round up to the next list",
     "        s += (1 << (top - TLSF_SL_LOG2)) - 1;",
     "        s += 0;"),
    ("TLSF remove leaves the list's bitmap bit set",
     "    if next as u64 == 0 {\n        t.sl_bitmap[m.x] &= ~(1 << m.y);",
     "    if next as u64 == 0 {\n        t.sl_bitmap[m.x] &= 0xFFFFFFFF;"),
    ("TLSF malloc leaves the neighbour's 'previous is free' flag",
     "    let after = tlsf_next(b);\n    after.size &= ~TLSF_PREV_FREE;\n    t.used += tlsf_size(b);",
     "    let after = tlsf_next(b);\n    t.used += tlsf_size(b);"),
    ("TLSF realloc copies half the data",
     "    for i in 0..cur / 8 { dst[i] = src[i]; }",
     "    for i in 0..cur / 16 { dst[i] = src[i]; }"),
    ("TLSF rounds sizes down instead of up",
     "    return max((size + 15) & (0 - 16), TLSF_MIN_PAYLOAD);",
     "    return max(size & (0 - 16), TLSF_MIN_PAYLOAD);"),
    ("TLSF realloc grows in place without checking the room",
     "    if (next.size & TLSF_FREE) != 0 && cur + TLSF_HEADER + tlsf_size(next) >= need {",
     "    if (next.size & TLSF_FREE) != 0 {"),
    ("TLSF search skips the exact list",
     "    var sl_map = t.sl_bitmap[f] & (0xFFFFFFFF << sl);",
     "    var sl_map = t.sl_bitmap[f] & (0xFFFFFFFF << (sl + 1));"),
    ("TLSF free forgets the next block's back-link",
     "    after.prev_phys = b;\n    after.size |= TLSF_PREV_FREE;",
     "    after.size |= TLSF_PREV_FREE;"),
    ("TLSF pool end marker is marked free",
     "    end.size = TLSF_PREV_FREE;",
     "    end.size = TLSF_PREV_FREE | TLSF_FREE;"),
    ("TLSF used-byte count ignores frees",
     "    t.used -= tlsf_size(b);\n    tlsf_release(t, b);",
     "    tlsf_release(t, b);"),
    ("Arena pads in the wrong direction",
     "    let pad = (0 - cur) & (align - 1);          // bytes up to",
     "    let pad = cur & (align - 1);          // bytes up to"),
    ("Arena bounds check ignores the padding",
     "    if pad > room || size > room - pad { return 0 as *u8; }",
     "    if pad > room || size > room { return 0 as *u8; }"),
    ("Arena accepts alignments that aren't powers of two",
     "    if align == 0 || (align & (align - 1)) != 0 { return 0 as *u8; }\n    let cur",
     "    if align == 0 { return 0 as *u8; }\n    let cur"),
    ("Pool hands out one block past the end",
     "    if p.fresh < p.count {",
     "    if p.fresh <= p.count {"),
    ("Pool free doesn't update the count",
     "    p.free = ptr;\n    p.used -= 1;",
     "    p.free = ptr;"),
    ("Pool loses the rest of the free list on alloc",
     "        p.free = (head as **u8)[0];",
     "        p.free = 0 as *u8;"),
    ("Pool rounds the block size down",
     "        let bs = (max(block, 8) + al - 1) & (0 - al);",
     "        let bs = max(block, 8) & (0 - al);"),
]


def main():
    src = open(os.path.join(ROOT, "happ", "lib", "mem.ha")).read()
    test_src = open(os.path.join(ROOT, "tests", "alloc_test.ha")).read()
    target = host_target()
    survived = []
    for what, old, new in MUTATIONS:
        if src.count(old) != 1:
            print(f"  ??  {what}: the text to change was found {src.count(old)} times; fix the mutation")
            survived.append(what)
            continue
        tmp = tempfile.mkdtemp(prefix="happ-mut-")
        try:
            with open(os.path.join(tmp, "mem.ha"), "w") as f:       # found before happ/lib/mem.ha
                f.write(src.replace(old, new))
            with open(os.path.join(tmp, "alloc_test.ha"), "w") as f:
                f.write(test_src)
            lib = build(os.path.join(tmp, "alloc_test.ha"), [target], os.path.join(tmp, "out"), quiet=True,
                        bridges=False)[target]
            env = dict(os.environ, HAPP_ALLOC_LIB=lib)
            try:
                r = subprocess.run([sys.executable, "-m", "unittest", "tests/test_alloc.py"], cwd=ROOT, env=env,
                                   capture_output=True, text=True, timeout=300)
                caught, how = r.returncode != 0, ("tests fail" if r.returncode > 0 else f"crash ({r.returncode})")
            except subprocess.TimeoutExpired:
                caught, how = True, "hangs (timeout)"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"  {'caught' if caught else 'MISSED'}  {what}{'  -> ' + how if caught else ''}")
        if not caught:
            survived.append(what)
    print(f"{len(MUTATIONS) - len(survived)} of {len(MUTATIONS)} planted bugs caught")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
