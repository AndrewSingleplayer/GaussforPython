"""Build the web splat viewer into web/dist/.

    python3 web/build.py                          # all scenes
    python3 web/build.py --scenes unicorn,halo    # some of them

Output:
  dist/index.html      the page as a complete document (GitHub Pages or any web server)
  dist/app.html        the same page without <html>/<head>/<body> (for a claude.ai artifact)
  dist/viewer.js, worker.js, engine.mjs          the viewer
  dist/splatweb.wasm   the HA++ module (web/splatweb.ha) compiled to WebAssembly
  dist/track.wasm, tracker.mjs, track-worker.js  floor tracking for AR (web/track.ha)
  dist/scenes.json, dist/scenes/*.wasm           the scenes (see web/pack.py), each wrapped as a
                                                 WebAssembly module with one data segment

Test locally: python3 -m http.server -d web/dist 8000, then open http://localhost:8000
"""
import argparse
import hashlib
import json
import re
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from happ.driver import build as happ_build  # noqa: E402
import pack  # noqa: E402

DIST = os.path.join(HERE, "dist")


def leb128(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def wasm_container(payload):
    """A valid WebAssembly module whose memory holds `payload` (one active data segment at 0).
    Web hosts that serve only web file types (like claude.ai artifacts) serve .wasm, and the viewer
    reads the bytes directly at `offset`, so it doesn't need to run WebAssembly to load a scene."""
    def section(sid, content):
        return bytes([sid]) + leb128(len(content)) + content
    pages = max(1, (len(payload) + 65535) // 65536)
    memory = section(5, leb128(1) + b"\x00" + leb128(pages))
    export = section(7, leb128(1) + leb128(6) + b"memory" + b"\x02" + leb128(0))
    segment = leb128(1) + b"\x00" + b"\x41\x00\x0b" + leb128(len(payload))
    head = b"\x00asm\x01\x00\x00\x00" + memory + export + bytes([11]) + leb128(len(segment) + len(payload)) + segment
    return head + payload, len(head)
TITLES = {"unicorn": "Unicorn plush", "skull": "Brick skull", "halo": "Halo diorama", "firepit": "Fire pit",
          "lizard": "Horned lizard", "racoons": "Raccoon family"}
ORDER = ["unicorn", "skull", "halo", "firepit", "lizard", "racoons"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=",".join(ORDER))
    ap.add_argument("--repack", action="store_true", help="pack scenes again even if the .hspl exists")
    args = ap.parse_args()
    names = [n for n in args.scenes.split(",") if n]
    os.makedirs(DIST, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wasm = happ_build(os.path.join(HERE, "splatweb.ha"), ["web-wasm32"], tmp, quiet=True,
                          bridges=False)["web-wasm32"]
        shutil.copyfile(wasm, os.path.join(DIST, "splatweb.wasm"))
        wasm = happ_build(os.path.join(HERE, "track.ha"), ["web-wasm32"], tmp, quiet=True,
                          bridges=False)["web-wasm32"]
        shutil.copyfile(wasm, os.path.join(DIST, "track.wasm"))
    # Every reference between the page's files carries this build's version (?v=...), so a browser
    # that still has some files of an older build cached can't mix them with new ones.
    sources = ("engine.mjs", "worker.js", "viewer.js", "ar.js", "tracker.mjs", "track-worker.js")
    digest = hashlib.sha256()
    for f in sources + ("app.html",):
        with open(os.path.join(HERE, "src", f), "rb") as fh:
            digest.update(fh.read())
    for f in ("splatweb.wasm", "track.wasm"):
        with open(os.path.join(DIST, f), "rb") as fh:
            digest.update(fh.read())
    version = digest.hexdigest()[:10]
    stamp = lambda text: re.sub(r"""(["'])\./([\w-]+\.(?:js|mjs|wasm|json))\1""",
                                lambda m: f"{m[1]}./{m[2]}?v={version}{m[1]}", text)
    for f in sources:
        with open(os.path.join(HERE, "src", f)) as fh:
            text = fh.read()
        with open(os.path.join(DIST, f), "w") as fh:
            fh.write(stamp(text))
    entries = []
    for name in names:
        path = os.path.join(DIST, "scenes", f"{name}.hspl")
        if args.repack or not os.path.exists(path):
            pack.pack(name, os.path.join(DIST, "scenes"))
        with open(path, "rb") as fh:
            data = fh.read()
        count = int.from_bytes(data[8:12], "little")
        wrapped, offset = wasm_container(data)
        with open(os.path.join(DIST, "scenes", f"{name}.wasm"), "wb") as fh:
            fh.write(wrapped)
        entries.append({"name": name, "title": TITLES.get(name, name), "file": f"./scenes/{name}.wasm",
                        "offset": offset, "length": len(data), "splats": count, "bytes": len(wrapped)})
    with open(os.path.join(DIST, "scenes.json"), "w") as fh:
        json.dump(entries, fh, indent=1)
    with open(os.path.join(HERE, "src", "app.html")) as fh:
        app = stamp(fh.read())
    with open(os.path.join(DIST, "app.html"), "w") as fh:
        fh.write(app)
    with open(os.path.join(DIST, "index.html"), "w") as fh:
        fh.write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                 '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                 '</head>\n<body>\n' + app + '</body>\n</html>\n')
    total = sum(e["bytes"] for e in entries)
    print(f"web viewer in {os.path.relpath(DIST)}: {len(entries)} scenes, {total / 1e6:.1f} MB of scene data, "
          f"version {version}")


if __name__ == "__main__":
    main()
