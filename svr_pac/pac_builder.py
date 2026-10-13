"""
pac_builder.py - builds a new ch*.pac by putting a custom model on top of a base PAC.
No bpy dependency: it receives arrays already in game coordinates (prepared by exporter.py).

"Replace model" mode (the one tested in game with CJ, ported from pac_replace_model.py):
  1. Textures: every image of the model is packed into an atlas (power-of-2 sides) that becomes the
     'color' texture (DXT5 with mipmaps; alpha = the game's gloss). 'normal' becomes flat with a
     low alpha (the normal map's alpha is the specular mask: ~18 on skin and ~7 on clothes in the
     game's characters; 255 made CJ look like plastic), 'Mk' cyan and Mask* zero, so the base
     character's details do not show through.
  2. Meshes: each part is split so it never exceeds 20 bones or 65 535 vertices, converted to a
     triangle strip, and copies the record and E block (shader) of the closest "neutral" 'body'
     mesh of the base ('color' texture, not hands with BloodHand, not face). g_bDrawWrinkle
     (wrinkle maps) is turned off.
  3. Whatever is not body in the base (teeth, eyes) and every extra JBOY (hair, eyelashes,
     accessories) is left without triangles: vertices collapsed to a point, minimal bounding sphere.
     The blood layers (0x7530-0x7532) are rebuilt on the new skin instead (see blood_layers).
  4. The changed children are recompressed (BPE) and the container is rebuilt; the rest of the PAC
     (animations, parameters, portraits, other outfits) stays the same.

The base skeleton is NOT changed: the game's animations drive its bones.
"""
import struct

import numpy as np

from . import bpe, dds
from .jboy import Jboy, Mesh, check_strip, tris_to_strip
from .pac_container import Pac, build_pac

MAX_BONES = 20          # record palette: u32[20]
MAX_VERTS = 65535       # u16 indices
MAX_INFL = 4            # influences per vertex used by the game's PACs
FACE_BONES = {"atama", "d_kuchi", "d_kuchi_00", "d_kuchi_01", "u_kuchi", "u_kuchi_00", "l_kuchi", "r_kuchi",
              "l_u_kuchi", "r_u_kuchi", "hana", "miken", "l_eye", "r_eye", "l_mabuta", "r_mabuta"}
WARNING = "warning: "   # prefix of log lines shown as warnings by the operator
BLOOD_IDS = (0x7530, 0x7531, 0x7532)   # blood layers: face, chest, hair
BLOOD_OFFSET = 0.002    # the game's blood layers sit this far outside the skin
BLOOD_REACH = 0.4       # max distance (game units) from a base blood layer for a triangle to get blood
BLOOD_PLANAR_ERR = 0.035


class Part:
    """One piece of the model, in game coordinates (normals pointing INWARDS, as in the game).
    P, N: (n, 3) float; UV: (n, 2) in Blender convention (v up, 0..1);
    T: (m, 3) int; WI: (n, k) int, index into 'bones' (-1 = unused); WW: (n, k) float;
    bones: bone names; image: index into the image list (or None)."""

    def __init__(self, name, P, N, UV, T, WI, WW, bones, image=None):
        self.name, self.bones, self.image = name, list(bones), image
        self.P = np.asarray(P, dtype=np.float64)
        self.N = np.asarray(N, dtype=np.float64)
        self.UV = np.asarray(UV, dtype=np.float64)
        self.T = np.asarray(T, dtype=np.int64)
        self.WI = np.asarray(WI, dtype=np.int64)
        self.WW = np.asarray(WW, dtype=np.float64)


# ===================================================================== atlas
def build_atlas(images):
    """images: list of RGBA uint8 (height, width, 4). -> (atlas, [(x, y, width, height)]).
    Shelves by height; power-of-2 sides (height <= 2 * width)."""
    order = sorted(range(len(images)), key=lambda i: -images[i].shape[0])
    W = 1
    while W < max(im.shape[1] for im in images) or W * W < sum(im.shape[0] * im.shape[1] for im in images):
        W *= 2
    while True:
        rects, x, y, shelf = [None] * len(images), 0, 0, 0
        for i in order:
            h, w = images[i].shape[:2]
            if x + w > W:
                x, y, shelf = 0, y + shelf, 0
            rects[i] = (x, y, w, h)
            x += w
            shelf = max(shelf, h)
        H = 1
        while H < y + shelf:
            H *= 2
        if H <= W * 2:
            break
        W *= 2
    atlas = np.zeros((H, W, 4), dtype=np.uint8)
    atlas[...] = (128, 128, 128, 255)
    for im, (x, y, w, h) in zip(images, rects):
        atlas[y:y + h, x:x + w] = im
    return atlas, rects


# ===================================================================== meshes
def neutral_body_templates(j, nodes, g_body):
    """Meshes of the base 'body' group usable as a template for any mesh: their color texture is
    'color' (SvR hands use 'BloodHand*' and came out orange) and they are not part of the face
    (the game treats the face in a special way)."""
    out = []
    for m in j.meshes:
        if m.material_group != g_body or not len(m.triangles()):
            continue
        if (j.mesh_texture(m) or "").lower() != "color":
            continue
        pal = m.palette()
        wi = np.frombuffer(m.weights, dtype=np.uint8).reshape(m.vc, m.nb, 8)[:, 0, 0]
        dom = [nodes[pal[i]]["name"] for i in wi if i < len(pal)]
        if sum(1 for d in dom if d in FACE_BONES) > len(dom) / 2:
            continue
        out.append(m)
    if not out:
        raise ValueError("The base has no neutral 'body' mesh that can be used as a template.")
    return out


def blank_mesh(m, point=(0.0, 0.0, 0.0)):
    """Leaves a mesh with nothing visible: all indices equal (degenerate triangles), every vertex
    at one point and a minimal bounding sphere. (Clearing only the indices was tried with CJ
    and is not enough on its own.)"""
    m.strips = [np.full(len(s), s[0] if len(s) else 0, dtype=">u2") for s in m.strips]
    V = np.frombuffer(m.verts, ">f4").reshape(-1, 7).copy()
    V[:, 0:3] = point
    m.verts = V.astype(">f4").tobytes()
    struct.pack_into(">4f", m.record, 164, point[0], point[1], point[2], 0.001)


def split_parts(T, WI, WW, max_bones=MAX_BONES, max_verts=MAX_VERTS):
    """Splits the triangles into groups of <= max_bones bones and <= max_verts vertices."""
    def vbones(v):
        o = np.argsort(-WW[v])[:MAX_INFL]
        return {int(WI[v, j]) for j in o if WI[v, j] >= 0 and WW[v, j] > 1e-5}
    vb = [vbones(v) for v in range(len(WI))]
    chunks, cur, bones, verts = [], [], set(), set()
    for t in T:
        tb = vb[t[0]] | vb[t[1]] | vb[t[2]]
        if cur and (len(bones | tb) > max_bones or len(verts | set(t.tolist())) > max_verts):
            chunks.append(cur); cur, bones, verts = [], set(), set()
        cur.append(t); bones |= tb; verts |= set(t.tolist())
    if cur:
        chunks.append(cur)
    return [np.array(c, dtype=np.int64) for c in chunks]


def make_mesh(P, N, UV, T, WI, WW, bone_names, tname2i, tpl, group):
    """One SvR 2011 mesh (record + blocks) from arrays in game coordinates.
    UV already in game convention (v down). Returns (Mesh, number_of_unweighted_vertices)."""
    vc = len(P)
    if vc > MAX_VERTS:
        raise ValueError("too many vertices in one mesh")
    per_vertex, unweighted = [], 0
    for v in range(vc):
        acc = {}
        for j in range(WI.shape[1]):
            if WI[v, j] >= 0 and WW[v, j] > 1e-5:
                n = tname2i[bone_names[WI[v, j]]]
                acc[n] = acc.get(n, 0.0) + float(WW[v, j])
        if not acc:
            acc = {0: 1.0}                       # no weights: to the root node
            unweighted += 1
        per_vertex.append(sorted(acc.items(), key=lambda x: -x[1])[:MAX_INFL])
    nb = max(len(a) for a in per_vertex)
    pal = sorted({n for a in per_vertex for n, _ in a})
    if len(pal) > MAX_BONES:
        raise ValueError("more than %d bones in one mesh" % MAX_BONES)
    slot = {n: i for i, n in enumerate(pal)}
    wout = bytearray()
    for a in per_vertex:
        s = sum(w for _, w in a)
        for k in range(nb):
            wout += struct.pack(">B3xf", slot[a[k][0]], a[k][1] / s) if k < len(a) else struct.pack(">B3xf", 0xFF, 0.0)
    V = np.zeros((vc, 7), dtype=">f4")
    V[:, 0:3] = P
    V[:, 3:6] = N
    vb = bytearray(V.tobytes())
    for v in range(vc):
        vb[v * 28 + 24:v * 28 + 28] = b"\xff\xff\xff\xff"        # vertex color
    strip = tris_to_strip(T)
    if not check_strip(strip, T):
        raise RuntimeError("the strip does not reproduce the triangles")
    rec = bytearray(tpl.record[:180])
    struct.pack_into(">I", rec, 8, len(pal))
    for i in range(MAX_BONES):
        struct.pack_into(">I", rec, 12 + 4 * i, pal[i] + 1 if i < len(pal) else 0xFFFFFFFF)
    struct.pack_into(">I", rec, 96, group)
    c = (P.min(0) + P.max(0)) / 2
    r = float(np.sqrt(((P - c) ** 2).sum(1)).max()) * 1.5
    struct.pack_into(">4f", rec, 164, c[0], c[1], c[2], r)
    m = Mesh(rec, tpl.E, tpl.n_eptr, bytes(vb), bytes(wout), nb, np.asarray(UV, dtype=">f4").tobytes(), [strip])
    m.E_base = tpl.E_base
    if "g_bDrawWrinkle" in m.eparams():
        m.set_eparam("g_bDrawWrinkle", 0)
    return m, unweighted


def set_material_counts(j, counts):
    mat = bytearray(j.materials)
    for i, c in enumerate(counts):
        struct.pack_into(">I", mat, i * 32 + 24, c)
    j.materials = bytes(mat)


# ===================================================================== blood layers
def _planar_uv(m):
    """The game's face and chest blood layers use a front (x, y) or side (z, y) projection.
    -> 3x2 matrix c with uv = [a, b, 1] @ c, and its axes; None if neither fits (hair layers)."""
    P, uv = m.positions(), m.uvs()
    for ax in ([0, 1], [2, 1]):
        X = np.c_[P[:, ax], np.ones(len(P))]
        c = np.linalg.lstsq(X, uv, rcond=None)[0]
        if np.sqrt(((X @ c - uv) ** 2).mean()) < BLOOD_PLANAR_ERR:
            return c, ax
    return None


def _surface_uv(m, Q):
    """UV of the closest point of mesh m to each point of Q (projection on the nearest triangles,
    clamped barycentrics). Used for layers whose UVs are not a plane projection."""
    P, uv, T = m.positions(), m.uvs(), m.triangles()
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    cen = (A + B + C) / 3
    out = np.empty((len(Q), 2))
    for i, q in enumerate(Q):
        best = None
        for t in np.argsort(((cen - q) ** 2).sum(1))[:8]:
            e0, e1, d = B[t] - A[t], C[t] - A[t], q - A[t]
            g = np.array([[e0 @ e0, e0 @ e1], [e0 @ e1, e1 @ e1]])
            try:
                s, r = np.linalg.solve(g, [d @ e0, d @ e1])
            except np.linalg.LinAlgError:
                continue
            s, r = max(s, 0.0), max(r, 0.0)
            if s + r > 1:
                s, r = s / (s + r), r / (s + r)
            dist = np.linalg.norm(A[t] + s * e0 + r * e1 - q)
            if best is None or dist < best[0]:
                best = (dist, (1 - s - r) * uv[T[t, 0]] + s * uv[T[t, 1]] + r * uv[T[t, 2]])
        out[i] = best[1]
    return out


def blood_layers(base_layers, body, log=print, reach=None):
    """Rebuilds the blood layers (JBOY 0x7530 face, 0x7531 chest, 0x7532 hair) on the new skin.

    In the game each layer is a copy of the skin pushed BLOOD_OFFSET outwards, with its own UVs
    and an 8x8 white 'BloodFace'/'BloodBody' placeholder: the game draws the blood texture at
    run time through those UVs. Face and chest UVs are a front projection of the area.
    Here every triangle of the new 'body' close to a base layer mesh (and facing the same way)
    is copied to that layer, pushed outwards, with that mesh's projection (or, for hair, the UV of
    the closest point of the base layer) and the skin's weights.

    base_layers: {cid: Jboy of the base layer}; body: new body Jboy (same skeleton as the base).
    -> {cid: new Jboy}"""
    reach = BLOOD_REACH if reach is None else reach
    bnodes = body.node_list()
    g_body = [g.lower() for g in body.material_names()].index("body")
    # skin of the new body, as one vertex array
    P, N, WI, WW, T = [], [], [], [], []
    base_v = 0
    for m in body.meshes:
        tri = m.triangles()
        if m.material_group != g_body or not len(tri):
            continue
        pal = m.palette()
        wi, ww = m.influences()
        pad = MAX_INFL - wi.shape[1]                    # meshes have 1 to 4 influences per vertex
        wi = np.pad(wi, ((0, 0), (0, pad)), constant_values=0xFF)
        ww = np.pad(ww, ((0, 0), (0, pad)))
        P.append(m.positions()); N.append(m.normals())
        WI.append(np.where(wi < len(pal), np.array(pal + [-1])[np.minimum(wi, len(pal))], -1))
        WW.append(ww)
        T.append(tri + base_v)
        base_v += m.vc
    P, N, WI, WW, T = (np.concatenate(a) for a in (P, N, WI, WW, T))
    cen = P[T].mean(1)
    tn = N[T].sum(1)
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-9)

    # nearest base layer mesh of every skin triangle
    targets = [(cid, k, m) for cid in sorted(base_layers) for k, m in enumerate(base_layers[cid].meshes)
               if len(m.triangles())]
    best_d = np.full(len(T), np.inf)
    best_t = np.full(len(T), -1)
    for ti, (_cid, _k, m) in enumerate(targets):
        mp, mn = m.positions(), m.normals()
        d2 = ((cen[:, None, :] - mp[None, :, :]) ** 2).sum(2)
        j = d2.argmin(1)
        d = np.sqrt(d2[np.arange(len(T)), j])
        ok = (d < reach) & ((tn * mn[j]).sum(1) > 0.3) & (d < best_d)
        best_d[ok], best_t[ok] = d[ok], ti

    out = {}
    for cid, lay in base_layers.items():
        lnodes = lay.node_list()
        lname2i = {n["name"]: i for i, n in enumerate(lnodes)}
        lnames = [n["name"] for n in lnodes]

        def to_layer(n):                     # body node -> layer node (or its closest ancestor in it)
            while n >= 0 and bnodes[n]["name"] not in lname2i:
                n = bnodes[n]["parent"]
            return lname2i[bnodes[n]["name"]] if n >= 0 else 0
        node_map = np.array([to_layer(n) for n in range(len(bnodes))] + [-1])

        groups = lay.material_names()
        new_meshes = {g: [] for g in range(len(groups))}
        for k, m in enumerate(lay.meshes):
            ti = next((i for i, t in enumerate(targets) if t[0] == cid and t[1] == k), None)
            sel = T[best_t == ti] if ti is not None else T[:0]
            if not len(sel):
                blank_mesh(m)
                new_meshes[m.material_group].append(m)
                log("  blood 0x%x #%d: no skin near it, left empty" % (cid, k))
                continue
            used = np.unique(sel)
            remap = -np.ones(len(P), dtype=np.int64)
            remap[used] = np.arange(len(used))
            Ps = P[used] - N[used] * BLOOD_OFFSET          # game normals point inwards
            proj = _planar_uv(m)
            if proj is not None:
                c, ax = proj
                uv = np.c_[Ps[:, ax], np.ones(len(Ps))] @ c
            else:
                uv = _surface_uv(m, Ps)
            uv = np.clip(uv, 0.0, 1.0)
            wi = node_map[WI[used]]
            Tl = remap[sel]
            for chunk in split_parts(Tl, wi, WW[used]):
                u = np.unique(chunk)
                r = -np.ones(len(used), dtype=np.int64)
                r[u] = np.arange(len(u))
                nm, _ = make_mesh(Ps[u], N[used][u], uv[u], r[chunk], wi[u], WW[used][u],
                                  lnames, lname2i, m, m.material_group)
                new_meshes[m.material_group].append(nm)
            log("  blood 0x%x #%d: %d skin triangles (%s UVs)" % (
                cid, k, len(sel), "projected" if proj is not None else "closest point"))
        lay.meshes = [m for g in range(len(groups)) for m in new_meshes[g]]
        set_material_counts(lay, [len(new_meshes[g]) for g in range(len(groups))])
        out[cid] = lay
    return out


def add_blood(pac_data, base_data, model=0, log=print, reach=None):
    """Adds blood layers to an already exported PAC (one whose body was replaced and whose blood
    JBOYs were emptied), taking the layers of the base PAC it was built on as reference.
    Everything else in the PAC stays byte for byte. -> bytes of the new PAC."""
    pac, base = Pac(pac_data), Pac(base_data)
    layers = {cid: Jboy.parse(base.child(cid, model)) for cid in BLOOD_IDS
              if base.has(cid, model) and pac.has(cid, model)}
    if not layers:
        raise ValueError("The base PAC has no blood layers (JBOY 0x7530-0x7532).")
    body = Jboy.parse(pac.child(0, model))
    new = {cid: j.build() for cid, j in blood_layers(layers, body, log, reach).items()}
    models = []
    for mi, (name, kids) in enumerate(pac.models):
        if mi == model:
            kids = [(cid, (bpe.compress(new[cid]) if bpe.is_bpe(raw) else new[cid]) if cid in new else raw)
                    for cid, raw in kids]
        models.append((name, kids))
    return build_pac(pac_data, models)


# ===================================================================== entry point
def replace_model(base_data, parts, images, model=0, alpha=47, emd_id=None, log=print, specular=12, blood=True):
    """base_data: bytes of the base PAC; parts: list of Part; images: list of RGBA uint8 (row 0 = top).
    alpha: alpha of 'color' (gloss); specular: alpha of the flat 'normal' (specular mask);
    blood: rebuild the base's blood layers on the new skin (otherwise they are emptied).
    Returns the bytes of the new PAC."""
    pac = Pac(base_data)
    if model >= len(pac.models):
        raise ValueError("The base only has %d outfit(s)." % len(pac.models))
    tb = Jboy.parse(pac.child(0, model))
    tnodes = tb.node_list()
    tname2i = {n["name"]: i for i, n in enumerate(tnodes)}
    missing = sorted({b for p in parts for b in p.bones if b not in tname2i})
    if missing:
        raise ValueError("Bones not present in the base skeleton: " + ", ".join(missing))

    # ---- 1. atlas
    imgs, gray = list(images), None
    if not imgs or any(p.image is None for p in parts):        # parts without image: gray
        gray = len(imgs)
        imgs.append(np.full((16, 16, 4), (160, 160, 160, 255), dtype=np.uint8))
    atlas, rects = build_atlas(imgs)
    AH, AW = atlas.shape[:2]
    log("atlas %dx%d with %d images" % (AW, AH, len(imgs)))

    # ---- 2. meshes
    groups = [g.lower() for g in tb.material_names()]
    if "body" not in groups:
        raise ValueError("The base has no 'body' material group.")
    g_body = groups.index("body")
    tpls = neutral_body_templates(tb, tnodes, g_body)
    tcent = [t.positions().mean(0) for t in tpls]
    new_body, unweighted = [], 0
    for p in parts:
        UV = p.UV.copy()
        if UV.min() < -1e-3 or UV.max() > 1 + 1e-3:
            log("  " + WARNING + "%s has UVs outside 0..1 (tiled texture): they are clamped" % p.name)
            UV = np.clip(UV, 0.0, 1.0)
        x, y, w, h = rects[p.image if p.image is not None else gray]
        uv = np.empty_like(UV)
        uv[:, 0] = (x + UV[:, 0] * w) / AW
        uv[:, 1] = (y + (1.0 - UV[:, 1]) * h) / AH          # Blender: v up; game: v down
        T = p.T
        fn = np.cross(p.P[T[:, 1]] - p.P[T[:, 0]], p.P[T[:, 2]] - p.P[T[:, 0]])
        vn = p.N[T[:, 0]] + p.N[T[:, 1]] + p.N[T[:, 2]]
        if ((fn * vn).sum(1) < 0).mean() > 0.5:             # the winding must agree with the normal
            T = T[:, ::-1]                                   # (in the game, both point inwards)
        for ci, chunk in enumerate(split_parts(T, p.WI, p.WW)):
            used = np.unique(chunk)
            remap = -np.ones(len(p.P), dtype=np.int64)
            remap[used] = np.arange(len(used))
            c = p.P[used].mean(0)
            tpl = tpls[int(np.argmin([np.linalg.norm(c - tc) for tc in tcent]))]
            m, nw = make_mesh(p.P[used], p.N[used], uv[used], remap[chunk], p.WI[used], p.WW[used],
                              p.bones, tname2i, tpl, g_body)
            unweighted += nw
            new_body.append(m)
            log("  %s.%d: %5d vert %5d tri, strip %5d, %d infl., %2d bones, template body #%d" % (
                p.name, ci, m.vc, len(chunk), len(m.strips[0]), m.nb, len(m.palette()), tb.meshes.index(tpl)))
    if unweighted:
        log("  " + WARNING + "%d vertices without weights: assigned to the root bone" % unweighted)

    others = [m for m in tb.meshes if m.material_group != g_body]
    for m in others:
        blank_mesh(m)
    ordered = []
    for g in range(len(groups)):
        ordered += new_body if g == g_body else [m for m in others if m.material_group == g]
    tb.meshes = ordered
    set_material_counts(tb, [len(new_body) if g == g_body else sum(1 for m in others if m.material_group == g)
                             for g in range(len(groups))])
    new = {0: tb.build()}
    if blood:
        layers = {cid: Jboy.parse(pac.child(cid, model)) for cid in BLOOD_IDS if pac.has(cid, model)}
        if layers:
            for cid, j in blood_layers(layers, tb, log).items():
                new[cid] = j.build()
        else:
            log("  " + WARNING + "the base has no blood layers")
    blanked = []
    for cid in pac.ids(model):
        if cid != 0 and cid not in new:
            data = pac.child(cid, model)
            if data[:4] == b"JBOY":
                j = Jboy.parse(data)
                for m in j.meshes:
                    blank_mesh(m)
                new[cid] = j.build()
                blanked.append("0x%x" % cid)
    log("body: %d new meshes; emptied: base teeth/eyes and JBOY %s" % (len(new_body), ", ".join(blanked)))

    # ---- 3. textures
    arr = atlas.copy()
    arr[..., 3] = np.where(arr[..., 3] < 128, 0, alpha)     # alpha = gloss (0 where it was transparent)
    color = dds.make_dds_dxt5(arr)
    found = set()
    for cid in pac.ids(model):
        data = new.get(cid) or pac.child(cid, model)
        if not dds.is_texpack(data):
            continue
        head, items = dds.texpack_read(data)
        changed = False
        for it in items:
            n = it[0].lower()
            if n == "color":
                it[1] = color
            elif n == "normal" and dds.dds_info(it[1])[3] in (b"DXT1", b"DXT5"):
                it[1] = dds.dds_solid(it[1], (128, 128, 255), alpha=specular)
            elif n == "mk" and dds.dds_info(it[1])[3] in (b"DXT1", b"DXT5"):
                it[1] = dds.dds_solid(it[1], (0, 255, 255))
            elif n.startswith("mask"):
                it[1] = dds.dds_zero(it[1])
            else:
                continue
            found.add(n if not n.startswith("mask") else "mask")
            changed = True
        if changed:
            new[cid] = dds.texpack_write(head, items)
    if "color" not in found:
        raise ValueError("The base has no 'color' texture.")
    log("textures: color %dx%d DXT5 (alpha %d); normal flat (specular alpha %d); %s" % (
        AW, AH, alpha, specular, ", ".join(sorted(found - {"color", "normal"}))))

    # ---- 4. pack
    models = []
    for mi, (name, kids) in enumerate(pac.models):
        if mi != model:
            models.append((name, kids)); continue
        out = []
        for cid, raw in kids:
            if cid in new:
                out.append((cid, bpe.compress(new[cid]) if bpe.is_bpe(raw) else new[cid]))
            else:
                out.append((cid, raw))
        if emd_id is not None:
            old = name
            name = "%06d%s" % (emd_id, name[6:8])
            log("EMD entry: %s -> %s" % (old, name))
        models.append((name, out))
    return build_pac(base_data, models)
