"""
pac_container.py - EPK8 / PACH container of SvR 2011 (PC) ch*.pac files. No bpy dependency.

Container (little-endian):
  0x0000  'EPK8', u32 12 + 16n, u32 data_size, u32 7
  0x0800  EMD group: 'EMD ', u16 4n, u16 1, u32 0, and n entries of 16 B (one per PACH / outfit):
            name[8] "%06d%02d" (e.g. 00010202, 00010212, 00010222: id 102, outfits 0, 1, 2),
            u32 offset (units of 0x800, from 0x4000), u32 size (units of 0x100)
  0x4000  PACH of each entry, each one starting at a multiple of 0x800:
            'PACH', u32 m, m x (u32 id, u32 offset, u32 size); data from 8 + 12m,
            offsets relative to that point, children aligned to 4
  end     0x800-byte block (zeros or 'EOP5/plugin version 0.03')

  entry_size = ceil(PACH_size / 0x100)
  data_size  = last_offset * 0x800 + ceil(last_size * 0x100 / 0x800) * 0x800
  file_size  = 0x4000 + data_size + 0x800

Children are kept as stored (BPE or not); 'unstore' decompresses them.
Ported from pacrepack.py and extended to several PACHs.
"""
import struct

from . import bpe

DATA0 = 0x4000
EMD = 0x800


def align(v, a):
    return (v + a - 1) // a * a


def unstore(raw):
    return bpe.decompress(raw) if bpe.is_bpe(raw) else raw


def _read_entries(d):
    if d[:4] != b"EPK8":
        raise ValueError("Not an EPK8 file (SvR 2011 PC character .pac).")
    if d[EMD:EMD + 4] != b"EMD ":
        raise ValueError("No EMD group at 0x800.")
    n = struct.unpack_from("<H", d, EMD + 4)[0] // 4
    out = []
    for i in range(n):
        o = EMD + 12 + 16 * i
        name = d[o:o + 8].decode("latin1")
        off, size = struct.unpack_from("<II", d, o + 8)
        out.append((name, off, size))
    return out


def read_pach(d, base):
    """-> list [(id, stored_bytes)] of the PACH starting at 'base'."""
    if d[base:base + 4] != b"PACH":
        raise ValueError("No PACH at 0x%x." % base)
    n = struct.unpack_from("<I", d, base + 4)[0]
    data0 = base + 8 + n * 12
    kids = []
    for i in range(n):
        cid, off, size = struct.unpack_from("<3I", d, base + 8 + i * 12)
        kids.append((cid, bytes(d[data0 + off:data0 + off + size])))
    return kids


def write_pach(children):
    table, blob, off = bytearray(), bytearray(), 0
    for cid, raw in children:
        off = align(off, 4)
        blob += b"\0" * (off - len(blob))
        table += struct.pack("<3I", cid, off, len(raw))
        blob += raw
        off += len(raw)
    return b"PACH" + struct.pack("<I", len(children)) + bytes(table) + bytes(blob)


def read_pac(d):
    """-> list [(name, [(id, stored_bytes), ...])], one per PACH (outfit)."""
    return [(name, read_pach(d, DATA0 + off * 0x800)) for name, off, _ in _read_entries(d)]


def build_pac(template, models):
    """models: list [(name, [(id, stored_bytes)])] -> new pac with the header of 'template'
    (everything before 0x4000 except the size fields, plus the 0x800 tail)."""
    head = bytearray(template[:DATA0])
    struct.pack_into("<I", head, 4, 12 + 16 * len(models))
    struct.pack_into("<H", head, EMD + 4, 4 * len(models))
    body, cur = bytearray(), 0
    for i, (name, children) in enumerate(models):
        pach = write_pach(children)
        field = align(len(pach), 0x100) // 0x100
        o = EMD + 12 + 16 * i
        head[o:o + 8] = name.encode("latin1")[:8].ljust(8, b"\0")
        struct.pack_into("<II", head, o + 8, cur, field)
        body += pach
        cur += align(field * 0x100, 0x800) // 0x800
        body += b"\0" * (cur * 0x800 - len(body))
    struct.pack_into("<I", head, 8, len(body))     # data size in the EPK8 header
    return bytes(head) + bytes(body) + template[-0x800:]   # tail (zeros or 'EOP5/plugin...')


class Pac:
    """An open ch*.pac. Children are decompressed on demand and cached.
    'model' selects the PACH (outfit); almost every PAC only has 0."""

    def __init__(self, data):
        self.data = bytes(data)
        self.models = read_pac(self.data)
        self._cache = {}

    @classmethod
    def open(cls, path):
        with open(path, "rb") as f:
            return cls(f.read())

    def model_names(self):
        return [name for name, _ in self.models]

    def ids(self, model=0):
        return [cid for cid, _ in self.models[model][1]]

    def raw(self, cid, model=0):
        return next(r for c, r in self.models[model][1] if c == cid)

    def child(self, cid, model=0):
        key = (model, cid)
        if key not in self._cache:
            self._cache[key] = unstore(self.raw(cid, model))
        return self._cache[key]

    def has(self, cid, model=0):
        return any(c == cid for c, _ in self.models[model][1])
