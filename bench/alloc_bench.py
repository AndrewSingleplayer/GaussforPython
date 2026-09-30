"""Allocator benchmark: HA++ (happ/lib/mem.ha) vs glibc malloc vs Matthew Conte's C TLSF.

    python3 bench/alloc_bench.py            # full run (about a minute)
    python3 bench/alloc_bench.py --quick    # a tenth of the work

Conte's TLSF (BSD license) is downloaded from GitHub into bench/.cache/ the first
time; it is not part of this repository. Both TLSFs and the benchmark are compiled
with the same clang, -O3 and the same CPU flags as HA++'s own build of the target.
"""
import argparse
import csv
import io
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from happ.driver import build  # noqa: E402
from happ.targets import Toolchain, host_target, target_info  # noqa: E402

CACHE = os.path.join(ROOT, "bench", ".cache")
TLSF_REPO = "https://github.com/mattconte/tlsf"
TLSF_COMMIT = "deff9ab509341f264addbd3c8ada533678591905"     # v3.1, the version this was measured with


def conte_tlsf(path=None):
    if path:
        return path
    d = os.path.join(CACHE, "tlsf")
    if not os.path.exists(os.path.join(d, "tlsf.c")):
        os.makedirs(CACHE, exist_ok=True)
        print(f"downloading {TLSF_REPO}")
        subprocess.run(["git", "clone", "--quiet", TLSF_REPO, d], check=True)
        subprocess.run(["git", "-C", d, "checkout", "--quiet", TLSF_COMMIT], check=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tlsf", help="folder with Conte's tlsf.c/tlsf.h (default: download)")
    ap.add_argument("--csv", help="also write the results here")
    args = ap.parse_args()

    tc = Toolchain()
    clang = tc.require("clang", "the benchmark")
    target = host_target()
    cpu = target_info(target)["cpu"]
    tlsf_dir = conte_tlsf(args.tlsf)
    with tempfile.TemporaryDirectory() as tmp:
        build(os.path.join(ROOT, "tests", "alloc_test.ha"), [target], tmp, quiet=True, bridges=False)
        happ_obj = os.path.join(tmp, ".work", f"alloc_test-{target}.o")
        exe = os.path.join(tmp, "alloc_bench")
        # -DNDEBUG: Conte's TLSF checks 31 asserts otherwise; a release build doesn't
        subprocess.run([clang, "-O3", "-DNDEBUG", *cpu, "-I", tlsf_dir,
                        os.path.join(ROOT, "bench", "alloc_bench.c"), os.path.join(tlsf_dir, "tlsf.c"), happ_obj,
                        "-lm", "-o", exe], check=True)
        print(f"compiled with: clang -O3 -DNDEBUG {' '.join(cpu)} (HA++ library built for {target})")
        r = subprocess.run([exe, "0.1" if args.quick else "1"], capture_output=True, text=True, check=True)
    lines = [l for l in r.stdout.splitlines() if not l.startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(r.stdout)
    what = {"mixed": "mixed sizes 16 B..64 KB, random malloc/realloc/free, up to 8192 alive",
            "frame": "per frame: 64 buffers of 64 B..256 KB, then all released",
            "nodes": "32-byte objects, random alloc/free, up to 65536 alive"}
    for test in ("mixed", "frame", "nodes"):
        print(f"\n{test}: {what[test]}\n")
        print("| allocator | ns/op | p50 | p99 | p99.9 | max |")
        print("|---|---|---|---|---|---|")
        for row in rows:
            if row["test"] != test:
                continue
            if row["ns_per_op"] == "failed":
                print(f"| {row['allocator']} | failed | | | | |")
                continue
            print(f"| {row['allocator']} | {float(row['ns_per_op']):.1f} | {float(row['p50_ns']):.0f} ns | "
                  f"{float(row['p99_ns']):.0f} ns | {float(row['p999_ns']):.0f} ns | {float(row['max_ns']):.0f} ns |")


if __name__ == "__main__":
    main()
