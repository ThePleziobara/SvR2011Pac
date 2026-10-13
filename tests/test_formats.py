#!/usr/bin/env python3
"""
Tests of the add-on's format modules, outside Blender (Python 3 + numpy only).

    python3 tests/test_formats.py ../ch100.pac ../PACS/Base/ch50.pac [...]

1) EPK8/PACH: repacking without changes gives the same file byte for byte.
2) JBOY: rebuilding without changes gives the same JBOY byte for byte (every JBOY child).
3) BPE: compressing and decompressing returns the data (a compressed child of at least ~8 KB).
4) DDS: every texture in the packs decodes (and, if Pillow is installed, matches it).

--no-textures skips test 4 (the slowest one).
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from svr_pac import bpe, dds  # noqa: E402
from svr_pac.jboy import Jboy  # noqa: E402
from svr_pac.pac_container import Pac, build_pac  # noqa: E402

try:
    from PIL import Image
except ImportError:
    Image = None


def same_content(a, b):
    """Same content (except pointers): meshes, nodes, textures, materials."""
    if (a.nodes, a.tex_names, a.materials) != (b.nodes, b.tex_names, b.materials):
        return False
    if len(a.meshes) != len(b.meshes):
        return False
    for x, y in zip(a.meshes, b.meshes):
        if (x.verts, x.weights, x.uv, x.n_eptr, x.d_types) != (y.verts, y.weights, y.uv, y.n_eptr, y.d_types):
            return False
        if [t.tobytes() for t in x.strips] != [t.tobytes() for t in y.strips]:
            return False
        if x.eparams() != y.eparams():
            return False
        skip = set(range(104, 116)) | set(range(148, 156))
        if any(x.record[i] != y.record[i] for i in range(len(x.record)) if i not in skip):
            return False
    return True


def check(path, textures=True):
    ok = True
    pac = Pac.open(path)
    name = os.path.basename(path)

    same = build_pac(pac.data, pac.models) == pac.data
    ok &= same
    print("%s  %d PACH %s  1) identical repack: %s" % (
        name, len(pac.models), pac.model_names(), "OK" if same else "FAIL"))

    for mi in range(len(pac.models)):
        for cid in pac.ids(mi):
            data = pac.child(cid, mi)
            if data[:4] != b"JBOY":
                continue
            j = Jboy.parse(data)
            new = j.build()
            if new == data:
                res = "IDENTICAL"
            elif same_content(j, Jboy.parse(new)):
                # some originals leave garbage in the alignment padding; it is written as zeros
                res = "SAME except padding (%d B)" % sum(1 for a, b in zip(data, new) if a != b)
            else:
                res = "DIFFERENT"; ok = False
            nv = sum(m.vc for m in j.meshes)
            nt = sum(len(m.triangles()) for m in j.meshes)
            print("    2) PACH %d JBOY 0x%-5x %3d nodes %3d meshes %6d vert %6d tri: %s" % (
                mi, cid, len(j.nodes) // 80, len(j.meshes), nv, nt, res))

    stored = pac.models[0][1]
    comp = [(len(r), cid) for cid, r in stored if bpe.is_bpe(r) and len(r) >= 8000]
    if comp:
        _, cid = min(comp)                      # the smallest one of at least ~8 KB (fast)
        data = pac.child(cid)
        same = bpe.decompress(bpe.compress(data)) == data
        ok &= same
        print("    3) BPE round trip (child 0x%x, %d B): %s" % (cid, len(data), "OK" if same else "FAIL"))

    for mi in range(len(pac.models) if textures else 0):
        for cid in pac.ids(mi):
            data = pac.child(cid, mi)
            if not dds.is_texpack(data):
                continue
            for tname, d in dds.texpack_read(data)[1]:
                img = dds.decode(d)
                w, h, _, fcc = dds.dds_info(d)
                msg = "%dx%d %s" % (w, h, fcc.decode("latin1").strip("\0") or "RGB")
                if Image is not None and fcc in (b"DXT1", b"DXT5"):
                    ref = np.asarray(Image.open(io.BytesIO(d)).convert("RGBA"), dtype=int)
                    diff = int(np.abs(ref - img.astype(int)).max())
                    good = diff <= 3              # different interpolation rounding
                    ok &= good
                    msg += "  vs Pillow: max diff %d %s" % (diff, "OK" if good else "FAIL")
                print("    4) PACH %d 0x%x %-16s %s" % (mi, cid, tname, msg))
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    tex = "--no-textures" not in args
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print(__doc__); sys.exit(2)
    ok = all([check(p, tex) for p in paths])
    print("\nALL OK" if ok else "\nTHERE ARE FAILURES")
    sys.exit(0 if ok else 1)
