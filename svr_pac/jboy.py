"""
jboy.py - reads and rebuilds SvR 2011 (PC) JBOY models, including the POF0 table. No bpy dependency.

Format (BIG-endian; every pointer and POF0 entry is measured from byte 8 of the file):
  header 0x48 B     'JBOY', +4/+0xC POF0 position, +0x18 mesh count, +0x1C records ptr,
                    +0x20 node count (102), +0x24 texture count, +0x28 nodes ptr, +0x2C texture
                    names ptr, +0x30 materials ptr, +0x34 material count
  record (180 B)    +0/+156 vertex count, +4 strip count, +8 palette bone count, +12 palette u32[20]
                    (node + 1), +92 influences per vertex, +96 material group, +104/+108/+112 ptr A/B/C,
                    +116 material id, +120 shader[16], +144 E pointer count, +148 ptr E,
                    +152 ptr D, +164 bounding sphere f32[4]
  per mesh          E (shader parameters) | A: u32 ptr -> vertices | vertices 28 B
                    (pos f32[3], normal f32[3], color u32) | weights nb*8 B (palette byte, 3 zeros,
                    f32 weight) | UV f32[2] | D: n_strips x (u32 6, u32 n, u32 ptr -> indices)
                    | u16 indices of each strip (triangle strip), each padded to 4
  then              nodes (aligned to 16, 80 B each) | texture names (16 B) | materials (32 B)
                    | 'POF0', u32 size, positions of every pointer (delta in words)
  node (80 B)       name[16], translation f32[3] @16, euler rotation f32[3] @32 (rad),
                    parent i32 @48 (-1 = root; almost always before the child,
                    but not always); local matrix T * Rz * Ry * Rx
Normals and triangle winding point INWARDS in the game (see importer.py).

Ported from jboy_rebuild.py (without WWE 13 support).
"""
import math
import struct

import numpy as np

REC_SIZE = 180
NO_PAL = 0xFF           # unused palette byte in the weight block


def _u32(b, o):
    return struct.unpack_from(">I", b, o)[0]


def _align(v, a):
    return (v + a - 1) // a * a


def _cstr(b):
    return b.split(b"\0")[0].decode("latin1")


# ===================================================================== POF0
def pof0_decode(b):
    p = b.find(b"POF0")
    size = _u32(b, p + 4)
    q, end, pos, out = p + 8, p + 8 + size, 0, []
    while q < end:
        c = b[q]
        t = c >> 6
        if t == 0:
            break
        if t == 1:
            d = c & 0x3F; q += 1
        elif t == 2:
            d = ((c & 0x3F) << 8) | b[q + 1]; q += 2
        else:
            d = ((c & 0x3F) << 24) | (b[q + 1] << 16) | (b[q + 2] << 8) | b[q + 3]; q += 4
        pos += d * 4
        out.append(pos)
    return out


def pof0_encode(positions):
    out, last = bytearray(), 0
    for p in sorted(positions):
        d = (p - last) // 4
        if d < 0x40:
            out.append(0x40 | d)
        elif d < 0x4000:
            out += bytes((0x80 | (d >> 8), d & 0xFF))
        else:
            out += bytes((0xC0 | (d >> 24), (d >> 16) & 0xFF, (d >> 8) & 0xFF, d & 0xFF))
        last = p
    out += b"\0" * (_align(len(out), 4) - len(out))
    return b"POF0" + struct.pack(">I", len(out)) + bytes(out)


# ===================================================================== matrices
def rotm(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ===================================================================== model
class Mesh:
    """One mesh. 'record' holds the record bytes (pointers are rewritten when building)."""

    def __init__(self, record, E, n_eptr, verts, weights, nb, uv, strips):
        self.record = bytearray(record)
        self.E = bytes(E)              # parameter block as is (pointers relative to the original E)
        self.E_base = 0                # original position of E (to relocate its pointer table)
        self.n_eptr = n_eptr           # entries of the pointer table at the start of E
        self.verts = bytes(verts)      # vc * 28
        self.weights = bytes(weights)  # vc * nb * 8
        self.nb = nb
        self.uv = bytes(uv)            # vc * 8
        self.strips = [np.asarray(s, dtype=">u2") for s in strips]   # one or more triangle strips
        self.d_types = [6] * len(self.strips)                         # 1st field of each D header

    @property
    def vc(self):
        return len(self.verts) // 28

    # ------------------------------------------------------------- convenience readers
    @property
    def shader(self):
        return _cstr(bytes(self.record[120:136]))

    @property
    def material_group(self):
        return _u32(self.record, 96)

    @property
    def material_id(self):
        return _u32(self.record, 116)

    def palette(self):
        """Node indices (in this mesh's JBOY) of each palette entry."""
        n = _u32(self.record, 8)
        return [_u32(self.record, 12 + 4 * i) - 1 for i in range(n)]

    def positions(self):
        return np.frombuffer(self.verts, ">f4").reshape(-1, 7)[:, 0:3].astype(np.float64)

    def normals(self):
        return np.frombuffer(self.verts, ">f4").reshape(-1, 7)[:, 3:6].astype(np.float64)

    def uvs(self):
        return np.frombuffer(self.uv, ">f4").reshape(-1, 2).astype(np.float64)

    def influences(self):
        """-> (palette indices int[vc, nb], weights float[vc, nb]); index 0xFF = unused."""
        W = np.frombuffer(self.weights, dtype=np.uint8).reshape(self.vc, self.nb, 8)
        wi = W[:, :, 0].astype(np.int64)
        ww = np.frombuffer(W[:, :, 4:8].tobytes(), ">f4").reshape(self.vc, self.nb).astype(np.float64)
        return wi, ww

    def triangles(self):
        tris = [strip_to_tris(s) for s in self.strips]
        return np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)

    def eparams(self):
        """E block parameters: {name: (offset_in_E, u32_value)}."""
        out = {}
        for i in range(self.n_eptr):
            p = _u32(self.E, 4 * i) - self.E_base
            out[_cstr(self.E[p:p + 16])] = (p, _u32(self.E, p + 20))
        return out

    def set_eparam(self, name, value):
        E = bytearray(self.E)
        p, _ = self.eparams()[name]
        struct.pack_into(">I", E, p + 20, value)
        self.E = bytes(E)


class Jboy:
    def __init__(self):
        self.header = bytearray(0x48)
        self.meshes = []
        self.nodes = b""        # n * 80
        self.tex_names = b""    # n * 16
        self.materials = b""    # n * 32

    # ------------------------------------------------------------- reading
    @classmethod
    def parse(cls, b):
        if b[:4] != b"JBOY":
            raise ValueError("not a JBOY model")
        if _u32(b, 0x08) == 0x10:
            raise ValueError("WWE 13 JBOY: not supported (SvR 2011 PC only)")
        j = cls()
        j.header = bytearray(b[:0x48])
        first = _u32(b, 0x1C) + 8
        for k in range(_u32(b, 0x18)):
            r = first + k * REC_SIZE
            rec = b[r:r + REC_SIZE]
            vc, nb = _u32(rec, 0), _u32(rec, 92)
            A, B, C = _u32(rec, 104), _u32(rec, 108), _u32(rec, 112)
            E, D = _u32(rec, 148), _u32(rec, 152)
            if B - A != vc * 28 + 4 or C - B != vc * nb * 8:
                raise ValueError("mesh %d: unexpected block sizes" % k)
            ns = _u32(rec, 4)
            heads = [struct.unpack_from(">3I", b, D + 8 + 12 * i) for i in range(ns)]
            m = Mesh(rec, b[E + 8:A + 8], _u32(rec, 144),
                     b[A + 12:A + 12 + vc * 28],
                     b[B + 8:B + 8 + vc * nb * 8], nb,
                     b[C + 8:C + 8 + vc * 8],
                     [np.frombuffer(b[p + 8:p + 8 + n * 2], ">u2") for _t, n, p in heads])
            m.E_base = E
            m.d_types = [t for t, _n, _p in heads]
            if any(int(s.max(initial=0)) >= vc for s in m.strips):
                raise ValueError("mesh %d: strip index out of range" % k)
            j.meshes.append(m)
        nn, nt, nm = _u32(b, 0x20), _u32(b, 0x24), _u32(b, 0x34)
        pn, pt, pm = _u32(b, 0x28) + 8, _u32(b, 0x2C) + 8, _u32(b, 0x30) + 8
        j.nodes = b[pn:pn + nn * 80]
        j.tex_names = b[pt:pt + nt * 16]
        j.materials = b[pm:pm + nm * 32]
        return j

    # ------------------------------------------------------------- convenience readers
    def texture_names(self):
        return [_cstr(self.tex_names[i:i + 16]) for i in range(0, len(self.tex_names), 16)]

    def material_names(self):
        return [_cstr(self.materials[i:i + 16]) for i in range(0, len(self.materials), 32)]

    def node_list(self):
        """Nodes with name, t, r, parent and world matrix 'W' (4x4, game space)."""
        out = []
        for i in range(len(self.nodes) // 80):
            o = i * 80
            out.append(dict(
                name=_cstr(self.nodes[o:o + 16]),
                t=np.array(struct.unpack_from(">3f", self.nodes, o + 16), dtype=np.float64),
                r=np.array(struct.unpack_from(">3f", self.nodes, o + 32), dtype=np.float64),
                parent=struct.unpack_from(">i", self.nodes, o + 48)[0]))
        # parents almost always come before their children, but not always (ch288, eyelashes:
        # 'root' hangs from an unnamed node at the end), so it is resolved recursively
        def world(i, seen=()):
            n = out[i]
            if "W" not in n:
                if i in seen:
                    raise ValueError("node %d (%s): parent cycle" % (i, n["name"]))
                L = np.eye(4); L[:3, :3] = rotm(*n["r"]); L[:3, 3] = n["t"]
                p = n["parent"]
                n["W"] = L if not 0 <= p < len(out) else world(p, seen + (i,)) @ L
            return n["W"]
        for i in range(len(out)):
            world(i)
        return out

    def mesh_texture(self, m, param="texDiffuse"):
        """Name of the texture used by a shader parameter (None if it has none)."""
        e = m.eparams().get(param)
        names = self.texture_names()
        if e is None or e[1] >= len(names):
            return None
        return names[e[1]]

    # ------------------------------------------------------------- writing
    def build(self):
        n = len(self.meshes)
        out = bytearray(8)                     # the first 8 bytes do not count for pointers
        out += self.header[8:0x48]
        ptrs = [0x14, 0x20, 0x24, 0x28]        # header fields that are pointers

        def pos():                             # current position in data space
            return len(out) - 8

        rec_at = pos()
        out += b"\0" * (n * REC_SIZE)
        for k, m in enumerate(self.meshes):
            rec = bytearray(m.record)
            # E: copy it and relocate its pointer table
            E = pos()
            eb = bytearray(m.E)
            for i in range(m.n_eptr):
                v = struct.unpack_from(">I", eb, 4 * i)[0]
                struct.pack_into(">I", eb, 4 * i, v - m.E_base + E)
                ptrs.append(E + 4 * i)
            out += eb
            A = pos()
            out += struct.pack(">I", A + 4); ptrs.append(A)
            out += m.verts
            B = pos(); out += m.weights
            C = pos(); out += m.uv
            D = pos()
            out += b"\0" * (12 * len(m.strips))
            for i, (t, s) in enumerate(zip(m.d_types, m.strips)):
                struct.pack_into(">III", out, 8 + D + 12 * i, t, len(s), pos())
                ptrs.append(D + 12 * i + 8)
                out += s.astype(">u2").tobytes()
                out += b"\0" * (_align(len(out), 4) - len(out))
            # record
            struct.pack_into(">I", rec, 0, m.vc)
            struct.pack_into(">I", rec, 4, len(m.strips))
            struct.pack_into(">I", rec, 92, m.nb)
            struct.pack_into(">III", rec, 104, A, B, C)
            struct.pack_into(">I", rec, 144, m.n_eptr)
            struct.pack_into(">I", rec, 148, E)
            struct.pack_into(">I", rec, 152, D)
            struct.pack_into(">I", rec, 156, m.vc)
            r = rec_at + k * REC_SIZE
            ptrs += [r + f for f in (104, 108, 112, 148, 152)]
            out[8 + r:8 + r + REC_SIZE] = rec
        out += b"\0" * (_align(pos(), 16) - pos())
        p_nodes = pos(); out += self.nodes
        p_tex = pos(); out += self.tex_names
        p_mat = pos(); out += self.materials
        p_pof = pos()
        out += pof0_encode(ptrs)
        out[0:4] = b"JBOY"
        struct.pack_into(">I", out, 0x04, p_pof)
        struct.pack_into(">I", out, 0x0C, p_pof)
        struct.pack_into(">I", out, 0x18, n)
        struct.pack_into(">I", out, 0x1C, rec_at)
        struct.pack_into(">I", out, 0x20, len(self.nodes) // 80)
        struct.pack_into(">I", out, 0x24, len(self.tex_names) // 16)
        struct.pack_into(">I", out, 0x28, p_nodes)
        struct.pack_into(">I", out, 0x2C, p_tex)
        struct.pack_into(">I", out, 0x30, p_mat)
        struct.pack_into(">I", out, 0x34, len(self.materials) // 32)
        return bytes(out)


# ===================================================================== strips
def strip_to_tris(idx):
    """Triangle strip -> triangles (n, 3). Odd ones are (s[i], s[i+2], s[i+1]);
    degenerate ones are skipped."""
    s = np.asarray(idx, dtype=np.int64)
    if len(s) < 3:
        return np.zeros((0, 3), dtype=np.int64)
    a, b, c = s[:-2], s[1:-1], s[2:]
    odd = (np.arange(len(a)) & 1).astype(bool)
    T = np.stack([a, np.where(odd, c, b), np.where(odd, b, c)], axis=1)
    keep = (a != b) & (b != c) & (a != c)
    return T[keep]


def tris_to_strip(tris):
    """Triangles (a, b, c) -> one u16 triangle strip (runs joined with degenerate triangles).
    Keeps the winding of every triangle according to the strip_to_tris rule. Ported from pac_replace_model.py."""
    tris = [tuple(int(x) for x in t) for t in tris]
    by_edge = {}
    for k, (a, b, c) in enumerate(tris):
        for e in ((a, b), (b, c), (c, a)):
            by_edge.setdefault(e, []).append(k)
    used = [False] * len(tris)

    def take(edge):
        for k in by_edge.get(edge, ()):
            if not used[k]:
                return k
        return None

    def third(k, u, v):
        a, b, c = tris[k]
        return ({a, b, c} - {u, v}).pop() if len({a, b, c}) == 3 else a

    s = []
    for start in range(len(tris)):
        if used[start]:
            continue
        a, b, c = tris[start]
        if s:
            s += [s[-1], a]
            if len(s) % 2:
                s.append(a)
        s += [a, b, c]
        used[start] = True
        while True:
            p = len(s) - 2                     # position of the next triangle
            u, v = s[-2], s[-1]
            # even: (u, v, x) must be a rotation of the triangle -> directed edge (u, v)
            # odd: (u, x, v) -> directed edge (v, u)
            k = take((u, v)) if p % 2 == 0 else take((v, u))
            if k is None:
                break
            used[k] = True
            s.append(third(k, u, v))
    return np.array(s, dtype=np.int64)


def check_strip(strip, tris):
    """True if the strip produces exactly the same triangles (with the same winding)."""
    def canon(T):
        return {tuple(np.roll(t, -int(np.argmin(t)))) for t in np.asarray(T)}
    return canon(strip_to_tris(strip)) == canon(tris)
