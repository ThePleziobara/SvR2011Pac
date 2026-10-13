"""
bpe.py - BPE compression used by SvR 2011 (PC) ch*.pac files. No bpy dependency.

    'BPE ', u32 256, u32 compressed_size, u32 uncompressed_size, blocks...

Each block (<= 4000 input bytes): pair table (Gage's format), LITTLE-endian u16 with the data
length, and the data. Ported from pacrepack.py.
"""
import struct

import numpy as np

BLOCK = 4000          # input bytes per block (like the original)
MAXCHARS = 200        # the original ends a block when it reaches 200 distinct values
MAXDEPTH = 15         # maximum pair nesting depth (the original reaches 15-38)
THRESHOLD = 3


def is_bpe(raw):
    return raw[:4] == b"BPE "


def parse_block(buf, p):
    """Reads one block: returns (left, right, compressed_data, new_p, table_bytes)."""
    left, right, c, p0 = list(range(256)), [0] * 256, 0, p
    while c < 256:
        cnt = buf[p]; p += 1
        if cnt > 127:
            c += cnt - 127; cnt = 0
        if c >= 256:
            break
        for _ in range(cnt + 1):
            left[c] = buf[p]; p += 1
            if c != left[c]:
                right[c] = buf[p]; p += 1
            c += 1
            if c == 256:
                break
    tab = bytes(buf[p0:p])
    size = buf[p] | (buf[p + 1] << 8); p += 2
    return left, right, bytes(buf[p:p + size]), p + size, tab


def write_table(left, right):
    """Serializes the pair table exactly like the original compressor (Gage)."""
    t, c = bytearray(), 0
    while c < 256:
        if c == left[c]:
            ln = 1; c += 1
            while ln < 127 and c < 256 and c == left[c]:
                ln += 1; c += 1
            t.append(ln + 127)
            if c == 256:
                break
            n_ent = 1
        else:
            ln = 1; c += 1
            while ln < 127 and c < 256 and (c != left[c] or (c + 1 < 256 and c + 1 != left[c + 1])):
                ln += 1; c += 1
            t.append(ln - 1)
            c -= ln
            n_ent = ln
        for _ in range(n_ent):
            t.append(left[c])
            if c != left[c]:
                t.append(right[c])
            c += 1
    return bytes(t)


def decompress(buf):
    if not is_bpe(buf):
        raise ValueError("not BPE data")
    dsz = struct.unpack_from("<I", buf, 12)[0]
    p, out = 16, bytearray()
    while len(out) < dsz:
        left, right, data, p, _ = parse_block(buf, p)
        cache = {}

        def ex(x, d=0):
            r = cache.get(x)
            if r is None:
                r = bytes((x,)) if (x == left[x] or d > 300) else ex(left[x], d + 1) + ex(right[x], d + 1)
                cache[x] = r
            return r
        out += b"".join([ex(x) for x in data])
    if len(out) != dsz:
        raise ValueError("BPE: %d bytes, expected %d" % (len(out), dsz))
    return bytes(out)


def _compress_block(chunk):
    """One block -> bytes (table + u16 + data)."""
    a = np.frombuffer(chunk, dtype=np.uint8).astype(np.int32)
    left, right, depth = list(range(256)), [0] * 256, [0] * 256
    used = set(np.unique(a).tolist())
    free = [c for c in range(256) if c not in used]
    while free and len(a) >= 2:
        key = a[:-1] * 256 + a[1:]
        vals, cn = np.unique(key, return_counts=True)
        order = np.argsort(cn)[::-1][:12]
        done = False
        for oi in order:
            cand = int(vals[oi])
            n = int(cn[oi])
            if n < THRESHOLD:
                break
            x, y = cand >> 8, cand & 255
            if 1 + max(depth[x], depth[y]) > MAXDEPTH:
                continue
            m = np.nonzero(key == cand)[0]
            if x == y and len(m) > 1:                      # runs aa..a: no overlapping
                start = np.concatenate(([True], np.diff(m) != 1))
                grp = np.cumsum(start) - 1
                first = m[start][grp]
                m = m[(m - first) % 2 == 0]
            if len(m) < THRESHOLD:
                continue
            new = free.pop(0)
            left[new], right[new], depth[new] = x, y, 1 + max(depth[x], depth[y])
            keep = np.ones(len(a), dtype=bool)
            keep[m + 1] = False
            a[m] = new
            a = a[keep]
            done = True
            break
        if not done:
            break
    return write_table(left, right) + struct.pack("<H", len(a)) + a.astype(np.uint8).tobytes()


def _split(data, cap):
    """Splits into blocks of <= BLOCK bytes; with cap, a block ends at MAXCHARS distinct values."""
    arr = np.frombuffer(data, dtype=np.uint8)
    pos = 0
    while pos < len(arr):
        win = arr[pos:pos + BLOCK]
        n = len(win)
        if cap and n > 1:
            vals, first = np.unique(win, return_index=True)
            if len(vals) >= MAXCHARS:
                n = int(np.sort(first)[MAXCHARS - 1]) + 1
        yield bytes(win[:n])
        pos += n


def compress(data):
    best = None
    for cap in (True, False):
        body = b"".join(_compress_block(ch) for ch in _split(data, cap))
        if best is None or len(body) < len(best):
            best = body
    return b"BPE " + struct.pack("<III", 256, len(best), len(data)) + best
