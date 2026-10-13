"""
dds.py - texture packs of ch*.pac files and DDS encoding/decoding. No bpy dependency.

Texture pack (children 0xA..0xF, little-endian; textures are looked up BY NAME):
  u32 n, u32 0x100, u32 0, u32 0x10, n x (name[16], 'dds\\0', u32 size, u32 offset, u32 0)
  DDS files aligned to 16 B.

Formats seen in ch100: DXT1, DXT5 and uncompressed A4R4G4B4 (Mask1/Mask2).
Reading: only level 0 is decoded (no mips). Writing: DXT5 with every mipmap,
and solid-color or zeroed textures (to neutralize normal, Mk and Mask*).
"""
import struct

import numpy as np


# ===================================================================== packs
def texpack_read(b):
    """-> (16-byte header, [[name, dds_bytes], ...])"""
    n = struct.unpack_from("<I", b, 0)[0]
    out = []
    for i in range(n):
        o = 16 + i * 32
        name = b[o:o + 16].split(b"\0")[0].decode("latin1")
        size, off = struct.unpack_from("<II", b, o + 20)
        out.append([name, b[off:off + size]])
    return b[:16], out


def texpack_write(head, items):
    n = len(items)
    out = bytearray(head)
    struct.pack_into("<I", out, 0, n)
    table, data = bytearray(), bytearray()
    off = 16 + 32 * n
    for name, dds in items:
        table += name.encode("latin1")[:16].ljust(16, b"\0") + b"dds\0" + struct.pack("<III", len(dds), off + len(data), 0)
        data += dds
        data += b"\0" * (-len(data) % 16)               # every DDS aligned to 16
    return bytes(out + table + data)


def is_texpack(b):
    """Checks that b has the shape of a texture pack."""
    if len(b) < 48:
        return False
    n, a, _z, c = struct.unpack_from("<4I", b, 0)
    if not (0 < n < 1000 and a == 0x100 and c == 0x10 and len(b) >= 16 + 32 * n):
        return False
    return b[16 + 16:16 + 20] == b"dds\0"


# ===================================================================== DDS
def dds_info(d):
    """-> (width, height, n_mips, fourcc)"""
    h, w = struct.unpack_from("<II", d, 12)
    mips = max(1, struct.unpack_from("<I", d, 28)[0])
    return w, h, mips, d[84:88]


def _565(c):
    r = ((c >> 11) & 31) * 255 // 31
    g = ((c >> 5) & 63) * 255 // 63
    b = (c & 31) * 255 // 31
    return np.stack([r, g, b], axis=-1).astype(np.int32)


def _color_blocks(cb, four_color):
    """cb: uint8 (n, 8) -> RGBA uint8 (n, 16, 4)."""
    c0 = cb[:, 0].astype(np.int32) | (cb[:, 1].astype(np.int32) << 8)
    c1 = cb[:, 2].astype(np.int32) | (cb[:, 3].astype(np.int32) << 8)
    bits = np.frombuffer(cb[:, 4:8].tobytes(), "<u4").astype(np.int64)
    p0, p1 = _565(c0), _565(c1)
    n = len(cb)
    pal = np.zeros((n, 4, 4), dtype=np.int32)
    pal[:, 0, :3], pal[:, 1, :3] = p0, p1
    pal[:, :, 3] = 255
    pal[:, 2, :3] = (2 * p0 + p1) // 3
    pal[:, 3, :3] = (p0 + 2 * p1) // 3
    if not four_color:                                    # DXT1 with c0 <= c1: 3 colors + transparent
        three = c0 <= c1
        pal[three, 2, :3] = (p0[three] + p1[three]) // 2
        pal[three, 3] = 0
    sel = (bits[:, None] >> (2 * np.arange(16))) & 3
    return np.take_along_axis(pal, sel[:, :, None].repeat(4, axis=2), axis=1).astype(np.uint8)


def _alpha_dxt5(ab):
    """ab: uint8 (n, 8) -> alpha uint8 (n, 16)."""
    a0, a1 = ab[:, 0].astype(np.int32), ab[:, 1].astype(np.int32)
    raw = np.zeros((len(ab), 8), dtype=np.uint8)
    raw[:, :6] = ab[:, 2:8]
    bits = np.frombuffer(raw.tobytes(), "<u8").astype(np.uint64)
    sel = ((bits[:, None] >> (3 * np.arange(16, dtype=np.uint64))) & np.uint64(7)).astype(np.int64)
    pal = np.zeros((len(ab), 8), dtype=np.int32)
    pal[:, 0], pal[:, 1] = a0, a1
    eight = a0 > a1
    for i in range(1, 7):                                 # 8 alphas: interpolation in 7 steps
        pal[eight, i + 1] = ((7 - i) * a0[eight] + i * a1[eight]) // 7
    for i in range(1, 5):                                 # 6 alphas + 0 + 255
        pal[~eight, i + 1] = ((5 - i) * a0[~eight] + i * a1[~eight]) // 5
    pal[~eight, 6], pal[~eight, 7] = 0, 255
    return np.take_along_axis(pal, sel, axis=1).astype(np.uint8)


def _alpha_dxt3(ab):
    bits = np.frombuffer(ab.tobytes(), "<u8").astype(np.uint64)
    sel = ((bits[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & np.uint64(15)).astype(np.uint8)
    return sel * 17


def _unblock(px, w, h):
    """(bh*bw, 16, 4) in 4x4 blocks -> image (h, w, 4)."""
    bw, bh = (w + 3) // 4, (h + 3) // 4
    img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w]


def _mask_channel(v, mask):
    if mask == 0:
        return None
    shift = (mask & -mask).bit_length() - 1
    top = mask >> shift
    return ((v >> shift) & top) * 255 // top


def decode(d):
    """DDS -> RGBA uint8 (height, width, 4), row 0 = top. Level 0 only."""
    if d[:4] != b"DDS ":
        raise ValueError("not a DDS file")
    w, h, _mips, fcc = dds_info(d)
    body = np.frombuffer(d, dtype=np.uint8, offset=128)
    nblk = ((w + 3) // 4) * ((h + 3) // 4)
    if fcc == b"DXT1":
        px = _color_blocks(body[:nblk * 8].reshape(nblk, 8), four_color=False)
    elif fcc in (b"DXT3", b"DXT5"):
        blk = body[:nblk * 16].reshape(nblk, 16)
        px = _color_blocks(np.ascontiguousarray(blk[:, 8:16]), four_color=True)
        ab = np.ascontiguousarray(blk[:, 0:8])
        px[:, :, 3] = _alpha_dxt5(ab) if fcc == b"DXT5" else _alpha_dxt3(ab)
    elif fcc == b"\0\0\0\0":
        bpp = struct.unpack_from("<I", d, 88)[0]
        masks = struct.unpack_from("<4I", d, 92)
        flags = struct.unpack_from("<I", d, 80)[0]
        if bpp not in (16, 24, 32):
            raise ValueError("uncompressed %d-bit DDS: not supported" % bpp)
        nb = bpp // 8
        raw = body[:w * h * nb].reshape(h * w, nb).astype(np.uint32)
        v = sum(raw[:, i] << (8 * i) for i in range(nb))
        img = np.full((h * w, 4), 255, dtype=np.uint8)
        for ch, mask in enumerate(masks):
            if ch == 3 and not flags & 0x1:               # no DDPF_ALPHAPIXELS
                continue
            c = _mask_channel(v, mask)
            if c is not None:
                img[:, ch] = c
        return img.reshape(h, w, 4)
    else:
        raise ValueError("DDS format %r not supported" % fcc)
    return _unblock(px, w, h)


# ===================================================================== writing
def _to565(c):
    c = c.astype(np.int32)
    return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)


def _from565(v):
    r = (v >> 11) & 31; g = (v >> 5) & 63; b = v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.int32)


def encode_dxt5(img):
    """img: RGBA uint8 (height, width, 4), multiples of 4 -> DXT5 bytes (no header).
    Ported from pac_replace_model.py (tested in game with CJ)."""
    h, w = img.shape[:2]
    blk = img.reshape(h // 4, 4, w // 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4).astype(np.int32)
    n = len(blk)
    rgb = blk[:, :, :3]
    # endpoints: bounding box of the block, slightly inset
    mn, mx = rgb.min(1), rgb.max(1)
    inset = (mx - mn) // 16
    c0, c1 = _to565(mx - inset), _to565(mn + inset)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    p0, p1 = _from565(c0), _from565(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3], 1)
    d = ((rgb[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)
    idx = d.argmin(-1).astype(np.uint32)
    idx[c0 == c1] = 0
    cbits = np.zeros(n, dtype=np.uint32)
    for t in range(16):
        cbits |= idx[:, t] << np.uint32(2 * t)
    # alpha: a0 = max, a1 = min, 8 levels
    a = blk[:, :, 3]
    a0, a1 = a.max(1), a.min(1)
    lv = np.stack([a0, a1] + [((7 - k) * a0 + k * a1) // 7 for k in range(1, 7)], 1)
    aidx = np.abs(a[:, :, None] - lv[:, None, :]).argmin(-1).astype(np.uint64)
    aidx[a0 == a1] = 0
    abits = np.zeros(n, dtype=np.uint64)
    for t in range(16):
        abits |= aidx[:, t] << np.uint64(3 * t)
    out = np.zeros((n, 16), dtype=np.uint8)
    out[:, 0], out[:, 1] = a0, a1
    for k in range(6):
        out[:, 2 + k] = (abits >> np.uint64(8 * k)) & np.uint64(0xFF)
    out[:, 8], out[:, 9] = c0 & 0xFF, c0 >> 8
    out[:, 10], out[:, 11] = c1 & 0xFF, c1 >> 8
    for k in range(4):
        out[:, 12 + k] = (cbits >> np.uint32(8 * k)) & np.uint32(0xFF)
    return out.tobytes()


def _half(img):
    """Halves the image with a 2x2 average (for the mipmaps)."""
    h, w = img.shape[:2]
    h2, w2 = max(1, h // 2), max(1, w // 2)
    a = img[:h2 * 2 if h > 1 else 1, :w2 * 2 if w > 1 else 1].astype(np.float32)
    if h > 1:
        a = (a[0::2] + a[1::2]) / 2
    if w > 1:
        a = (a[:, 0::2] + a[:, 1::2]) / 2
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def make_dds_dxt5(img):
    """RGBA uint8 (height, width, 4), row 0 = top -> DXT5 DDS with every mipmap."""
    h, w = img.shape[:2]
    mips, cur = [], img
    while True:
        ch, cw = cur.shape[:2]
        ph, pw = (ch + 3) // 4 * 4, (cw + 3) // 4 * 4
        a = np.zeros((ph, pw, 4), dtype=np.uint8)
        a[:ch, :cw] = cur
        if pw > cw:
            a[:, cw:] = a[:, cw - 1:cw]
        if ph > ch:
            a[ch:] = a[ch - 1:ch]
        mips.append(encode_dxt5(a))
        if cw == 1 and ch == 1:
            break
        cur = _half(cur)
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<7I", hdr, 4, 124, 0x000A1007, h, w, max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16, 0, len(mips))
    struct.pack_into("<2I", hdr, 76, 32, 4)                  # pixel format: size, DDPF_FOURCC
    hdr[84:88] = b"DXT5"
    struct.pack_into("<I", hdr, 108, 0x401008)              # caps: complex | texture | mipmap
    return bytes(hdr) + b"".join(mips)


def dds_solid(like, rgb, alpha=255):
    """DDS with the same size, format and mipmap count as 'like' (DXT1/DXT5), in a solid color."""
    w, h, mips, fcc = dds_info(like)
    c = int(_to565(np.array(rgb)))
    cblock = struct.pack("<HHI", c, c, 0)
    blk = cblock if fcc == b"DXT1" else bytes((alpha, alpha)) + b"\0" * 6 + cblock
    body = bytearray()
    for m in range(mips):
        bw, bh = max(1, (max(1, w >> m) + 3) // 4), max(1, (max(1, h >> m) + 3) // 4)
        body += blk * (bw * bh)
    return like[:128] + bytes(body)


def dds_zero(like):
    """Same DDS with all data set to zero (Mask1/Mask2: no effect)."""
    return like[:128] + bytes(len(like) - 128)
