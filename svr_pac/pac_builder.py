"""
pac_builder.py - builds a new ch*.pac by putting a custom model on top of a base PAC.
No bpy dependency: it receives arrays already in game coordinates (prepared by exporter.py).

"Replace model" mode (the one tested in game with CJ, ported from pac_replace_model.py):
  1. Textures: every image of the model is packed into an atlas (power-of-2 sides) that becomes the
     'color' texture (DXT5 with mipmaps; alpha = the game's gloss, or the image's own alpha on
     alpha-blended meshes such as hair). 'normal' gets the model's normal maps laid out like the
     atlas (flat where there is none; see dds.flip_normal_xy) with a low alpha (the normal map's
     alpha is the specular mask: ~18 on skin and ~7 on clothes in the game's characters; 255 made
     CJ look like plastic), 'Mk' cyan and Mask* zero, so the base character's details do not show
     through.
  2. Meshes: each part is split so it never exceeds 20 bones or 65 535 vertices, converted to a
     triangle strip, and copies the record and E block (shader) of the closest "neutral" 'body'
     mesh of the base ('color' texture, not hands with BloodHand, not face). g_bDrawWrinkle
     (wrinkle maps) is turned off.
  3. Parts whose target is an extra JBOY (long hair: 0x2710 with its kami_b* chains) go into that
     JBOY, copying the record and E block of its own meshes (alpha-blended, double-sided), with its
     skeleton updated from Blender (see update_skeleton) and the hair physics of child 0x3c.
     Whatever is not body in the base (teeth, eyes) and every other extra JBOY (hair, eyelashes,
     accessories) is left without triangles: vertices collapsed to a point, minimal bounding sphere.
     The blood layers (0x7530-0x7532) are rebuilt on the new skin instead (see blood_layers).
  4. The changed children are recompressed (BPE) and the container is rebuilt; the rest of the PAC
     (animations, parameters, portraits, other outfits) stays the same.

The body skeleton is NOT changed: the game's animations drive its bones. Only the bones that exist
just in an extra JBOY (hair chains, which the game moves with physics) can be moved or added.
"""
import math
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
HAIR_ID = 0x2710        # JBOY of the hair (long hair carries its own bone chains kami_bNN)
HAIR_PARAMS_ID = 0x3c   # text child: physics of each hair chain
# columns of 0x3c (from its Japanese header comments): radius, viscous resistance, gravity, spring
# constant, damping constant, attach (integer), length correction, bending stiffness, movement correction
HAIR_PARAMS = ("radius", "viscosity", "gravity", "spring", "damping", "attach", "length_fix", "stiffness",
               "move_fix")
ALPHA_FLAG = 0x100      # record +136: alpha-blended mesh (hair, eyelashes); its 'color' alpha is transparency


class Part:
    """One piece of the model, in game coordinates (normals pointing INWARDS, as in the game).
    P, N: (n, 3) float; UV: (n, 2) in Blender convention (v up, 0..1);
    T: (m, 3) int; WI: (n, k) int, index into 'bones' (-1 = unused); WW: (n, k) float;
    bones: bone names; image: index into the image list (or None);
    target: child id of the JBOY it goes into (0 = body, HAIR_ID = hair);
    alpha: the material is transparent (True), opaque (False) or unknown (None); in an extra JBOY it
    picks alpha-blended or opaque templates (the hair JBOY can mix strands and an opaque scalp)."""

    def __init__(self, name, P, N, UV, T, WI, WW, bones, image=None, target=0, alpha=None):
        self.name, self.bones, self.image, self.target = name, list(bones), image, target
        self.alpha = alpha
        self.P = np.asarray(P, dtype=np.float64)
        self.N = np.asarray(N, dtype=np.float64)
        self.UV = np.asarray(UV, dtype=np.float64)
        self.T = np.asarray(T, dtype=np.int64)
        self.WI = np.asarray(WI, dtype=np.int64)
        self.WW = np.asarray(WW, dtype=np.float64)


# ===================================================================== atlas
def build_atlas(images):
    """images: list of RGBA uint8 (height, width, 4). -> (atlas, [(x, y, width, height)]).
    Skyline (bottom-left): tallest first, each image where it ends highest, so a short image can sit
    under a wider one (1024x1024 + 1024x512 + 1024x512 -> 2048x1024); power-of-2 sides
    (height <= 2 * width)."""
    order = sorted(range(len(images)), key=lambda i: (-images[i].shape[0], -images[i].shape[1]))
    W = 1
    while W < max(im.shape[1] for im in images) or W * W < sum(im.shape[0] * im.shape[1] for im in images):
        W *= 2
    while True:
        sky = np.zeros(W, dtype=np.int64)              # height used in each column
        rects = [None] * len(images)
        for i in order:
            h, w = images[i].shape[:2]
            best = None
            for x in range(0, W - w + 1, 4):
                y = int(sky[x:x + w].max())
                if best is None or y < best[1]:
                    best = (x, y)
            x, y = best
            rects[i] = (x, y, w, h)
            sky[x:x + w] = y + h
        H = 1
        while H < sky.max():
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
def body_group(j):
    """Material group of the skin: 'body' or, in the bases that name it after the character
    (Melina, Eve, Alicia, Triple_H_FB... 33 of the 93 base PACs), the group with most vertices on 'color'."""
    groups = [g.lower() for g in j.material_names()]
    if "body" in groups:
        return groups.index("body")
    vc = {}
    for m in j.meshes:
        if (j.mesh_texture(m) or "").lower() == "color" and len(m.triangles()):
            vc[m.material_group] = vc.get(m.material_group, 0) + m.vc
    if not vc:
        raise ValueError("The base has no material group for the body ('body' or one using 'color').")
    return max(vc, key=vc.get)


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
    # only the main materials (record +116) of the base: a small accessory (Melina's id 7, 250
    # vertices by the neck) made Zoey's head and hair, its nearest template, look like chrome
    vc = {}
    for m in out:
        vc[m.material_id] = vc.get(m.material_id, 0) + m.vc
    main = [m for m in out if vc[m.material_id] >= 0.1 * sum(vc.values())]
    return main or out


def blank_mesh(m, point=(0.0, 0.0, 0.0)):
    """Leaves a mesh with nothing visible: all indices equal (degenerate triangles), every vertex
    at one point and a minimal bounding sphere. (Clearing only the indices was tried with CJ
    and is not enough on its own.) Only 3 vertices and one strip are kept: the emptied meshes of the
    base (long hair, teeth, eyes) would otherwise still count against the game's memory limit for
    the models of an attire with moving hair."""
    keep = min(3, m.vc)
    m.verts = m.verts[:28 * keep]
    m.weights = m.weights[:8 * m.nb * keep]
    m.uv = m.uv[:8 * keep]
    m.strips = [np.zeros(3, dtype=">u2")]
    m.d_types = m.d_types[:1]
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


def node_spheres(j):
    """Node +64 (the '4 floats' of each node): bounding sphere of the vertices weighted to that node,
    in bind space: centre of their box and half its diagonal; zeros if none (exact on ch100 and on
    ch293's hair). Recomputed for the JBOYs that get new meshes; emptied meshes do not count."""
    lo, hi = {}, {}
    for m in j.meshes:
        if not len(m.triangles()):
            continue
        pal, P = m.palette(), m.positions()
        wi, ww = m.influences()
        for k in range(m.nb):
            for p in np.unique(wi[(ww[:, k] > 0) & (wi[:, k] < len(pal)), k]):
                q = P[(wi[:, k] == p) & (ww[:, k] > 0)]
                n = pal[p]
                lo[n] = np.minimum(lo[n], q.min(0)) if n in lo else q.min(0)
                hi[n] = np.maximum(hi[n], q.max(0)) if n in hi else q.max(0)
    nodes = bytearray(j.nodes)
    for i in range(len(nodes) // 80):
        if i in lo:
            c = (lo[i] + hi[i]) / 2
            struct.pack_into(">4f", nodes, i * 80 + 64, c[0], c[1], c[2], np.linalg.norm(hi[i] - lo[i]) / 2)
        else:
            struct.pack_into(">4f", nodes, i * 80 + 64, 0.0, 0.0, 0.0, 0.0)
    j.nodes = bytes(nodes)


# ===================================================================== hair
def hair_params_read(data):
    """Child 0x3c: text with ';' comment lines (Shift-JIS) and one line per hair chain:
    root bone name + the HAIR_PARAMS values. -> {root: [9 floats]}"""
    out = {}
    for line in data.decode("latin1").splitlines():
        tk = line.split()
        if len(tk) == 1 + len(HAIR_PARAMS) and not tk[0].startswith(";"):
            try:
                out[tk[0]] = [float(v) for v in tk[1:]]
            except ValueError:
                pass
    return out


def _hair_line(name, vals):
    v = " ".join("%d" % round(x) if k == "attach" else "%.4f" % x for k, x in zip(HAIR_PARAMS, vals))
    return (" %s %s \r\n" % (name, v)).encode("latin1")


def hair_params_write(data, params):
    """Rewrites the lines of child 0x3c whose values changed and appends the chains it lacks;
    comments and untouched lines stay byte for byte. params: {root: [9 values]}"""
    old = hair_params_read(data)
    out = []
    for line in data.splitlines(keepends=True):
        tk = line.decode("latin1").split()
        if tk and tk[0] in old and tk[0] in params and not np.allclose(old[tk[0]], params[tk[0]], atol=5e-5):
            line = _hair_line(tk[0], params[tk[0]])
        out.append(line)
    new = [n for n in params if n not in old]
    if new:
        k = len(out)
        while k and not out[k - 1].strip():                  # before the trailing blank lines
            k -= 1
        if k and not out[k - 1].endswith(b"\n"):
            out[k - 1] += b"\r\n"
        out[k:k] = [_hair_line(n, params[n]) for n in new]
    return b"".join(out)


def euler_zyx(R):
    """Inverse of jboy.rotm: R = Rz @ Ry @ Rx -> (rx, ry, rz)."""
    ry = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 0.999999:
        return math.atan2(R[2, 1], R[2, 2]), ry, math.atan2(R[1, 0], R[0, 0])
    return math.atan2(-R[1, 2], R[1, 1]), ry, 0.0


def update_skeleton(j, body_nodes, bones, log=print, label=""):
    """Writes into the extra JBOY j its own bones (those not in the body, e.g. the kami_b* chains of the
    hair) as they are in Blender, so the hair can be refitted or get more chains.

    bones: {name: (parent_name, W)}, W 4x4 in the BODY's bind space (game coordinates).
    Nodes of j are never removed or reordered (indices may be referenced elsewhere); the ones in
    'bones' that j lacks are appended (copying the parent's node, sphere cleared). Body bones keep
    their node. A node whose local transform did not change keeps its bytes.
    Returns (M, names, added): M[i] = W_own(i) @ inverse(W_body(i)) takes a vertex from the body's
    bind pose to this JBOY's (identity unless the JBOY's pose differs, as in ch125); 'added' = new nodes."""
    nl = j.node_list()
    names = [n["name"] for n in nl]
    parents = [n["parent"] for n in nl]
    old_W = [n["W"] for n in nl]
    body_W = {n["name"]: n["W"] for n in body_nodes}
    nodes = bytearray(j.nodes)
    added, pending = [], [n for n in bones if n not in names]
    while pending:
        ready = [n for n in pending if bones[n][0] in names]
        if not ready:
            raise ValueError("%s: bones whose parent is not in the skeleton: %s" % (label, ", ".join(pending)))
        for n in ready:
            if len(n.encode("latin1", "replace")) > 15:
                raise ValueError("bone name longer than 15 characters: %s" % n)
            p = names.index(bones[n][0])
            node = bytearray(nodes[p * 80:p * 80 + 80])
            node[0:16] = n.encode("latin1", "replace").ljust(16, b"\0")
            struct.pack_into(">i", node, 48, p)
            node[64:80] = bytes(16)
            nodes += node
            names.append(n); parents.append(p); old_W.append(None)
            added.append(n)
            pending.remove(n)

    target, own = [None] * len(names), [None] * len(names)

    def tgt(i, seen=()):
        if target[i] is None:
            n, p = names[i], parents[i]
            if n in bones:
                target[i] = np.asarray(bones[n][1], dtype=np.float64)
            elif n in body_W:
                target[i] = body_W[n]
            elif 0 <= p < len(names) and p not in seen:      # unchanged local transform
                target[i] = tgt(p, seen + (i,)) @ np.linalg.inv(old_W[p]) @ old_W[i]
            else:
                target[i] = old_W[i]
        return target[i]

    def own_W(i, seen=()):
        if own[i] is None:
            n, p = names[i], parents[i]
            if n in body_W and n not in bones or not (0 <= p < len(names)) or p in seen:
                own[i] = old_W[i] if old_W[i] is not None else tgt(i)
            else:                                # same offset body -> own as its parent
                own[i] = own_W(p, seen + (i,)) @ np.linalg.inv(tgt(p)) @ tgt(i)
        return own[i]

    moved = 0
    for i in range(len(names)):
        p = parents[i]
        if names[i] in body_W and names[i] not in bones or not (0 <= p < len(names)):
            continue
        L = np.linalg.inv(own_W(p)) @ own_W(i)
        u, _s, vt = np.linalg.svd(L[:3, :3])                 # rotation only (no scale)
        R = u @ vt
        if old_W[i] is not None:
            L0 = np.linalg.inv(old_W[p]) @ old_W[i]
            if np.abs(L0[:3, 3] - L[:3, 3]).max() < 1e-4 and np.abs(L0[:3, :3] - R).max() < 1e-4:
                continue
            moved += 1
        struct.pack_into(">3f", nodes, i * 80 + 16, *L[:3, 3])
        struct.pack_into(">3f", nodes, i * 80 + 32, *euler_zyx(R))
    j.nodes = bytes(nodes)
    if moved or added:
        log("  %s skeleton: %d bone(s) moved, %d added%s" % (label, moved, len(added),
                                                              (": " + ", ".join(added)) if added else ""))
    M = {i: own_W(i) @ np.linalg.inv(tgt(i)) for i in range(len(names))}
    return M, names, added


def extra_templates(j):
    """Meshes of an extra JBOY usable as templates: with triangles and, if possible, using 'color'."""
    ms = [m for m in j.meshes if len(m.triangles())]
    good = [m for m in ms if (j.mesh_texture(m) or "").lower() == "color"]
    return good or ms


def to_bind(P, N, WI, WW, mats):
    """Moves vertices with the blend of one 4x4 matrix per bone (weights WI/WW, -1 = unused)."""
    ww = np.where(WI >= 0, WW, 0.0)
    s = ww.sum(1, keepdims=True)
    ww = np.where(s > 0, ww / np.maximum(s, 1e-12), 0.0)
    ww[s[:, 0] <= 0, 0] = 1.0
    stack = np.concatenate([mats, np.eye(4)[None]])
    A = np.einsum("vk,vkij->vij", ww, stack[np.where(WI >= 0, WI, len(mats))])
    return (np.einsum("vij,vj->vi", A[:, :3, :3], P) + A[:, :3, 3],
            np.einsum("vij,vj->vi", A[:, :3, :3], N))


def fit_base(img, like):
    """Shrinks an atlas (RGBA uint8) so it is not larger than the base PAC's texture 'like' (DDS) in
    either side: the game does not load a texture bigger than the one the character had (the model
    comes out white). Power-of-2 sides: box average. The UVs are relative, so they still match."""
    bw, bh = dds.dds_info(like)[:2]
    h, w = img.shape[:2]
    th, tw = min(h, bh), min(w, bw)
    if (th, tw) == (h, w):
        return img
    if h % th == 0 and w % tw == 0:
        a = img.reshape(th, h // th, tw, w // tw, 4).astype(np.float32).mean(axis=(1, 3))
        return (a + 0.5).astype(np.uint8)
    return resize_nearest(img, th, tw)


def resize_nearest(img, h, w):
    if img.shape[:2] == (h, w):
        return img
    ys = (np.arange(h) * img.shape[0] // h).clip(0, img.shape[0] - 1)
    xs = (np.arange(w) * img.shape[1] // w).clip(0, img.shape[1] - 1)
    return img[ys][:, xs]


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
    g_body = body_group(body)
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
def replace_model(base_data, parts, images, model=0, alpha=47, emd_id=None, log=print, specular=12, blood=True,
                  normals=None, extra_bones=None, hair_params=None):
    """base_data: bytes of the base PAC; parts: list of Part; images: list of RGBA uint8 (row 0 = top).
    alpha: alpha of 'color' (gloss); specular: alpha of 'normal' (specular mask);
    blood: rebuild the base's blood layers on the new skin (otherwise they are emptied);
    normals: list parallel to 'images' with each one's normal map (RGBA uint8, Blender/OpenGL
    convention, any size) or None; extra_bones: {name: (parent_name, W 4x4 game space, child id)}
    of the bones that only exist in an extra JBOY (hair chains), as they are in Blender;
    hair_params: {root bone: [values of HAIR_PARAMS]} for child 0x3c.
    Returns the bytes of the new PAC."""
    pac = Pac(base_data)
    if model >= len(pac.models):
        raise ValueError("The base only has %d outfit(s)." % len(pac.models))
    tb = Jboy.parse(pac.child(0, model))
    tnodes = tb.node_list()
    tname2i = {n["name"]: i for i, n in enumerate(tnodes)}
    extra_bones = extra_bones or {}

    # ---- 0. JBOYs that receive meshes: the body and, for long hair, the extra JBOYs with their skeleton
    groups = tb.material_names()
    g_body = body_group(tb)
    if groups[g_body].lower() != "body":
        log("body material group: %s" % groups[g_body])
    dest = {0: dict(j=tb, names=[n["name"] for n in tnodes], M=None, label="body",
                    tpls=neutral_body_templates(tb, tnodes, g_body), added=[])}
    for cid in sorted({p.target for p in parts} | {c for _p, _w, c in extra_bones.values()}):
        if cid == 0 or cid in dest:
            continue
        if cid in BLOOD_IDS or not pac.has(cid, model) or pac.child(cid, model)[:4] != b"JBOY":
            if any(p.target == cid for p in parts):
                raise ValueError("The base has no JBOY 0x%x for %s (choose a base with long hair)." % (
                    cid, ", ".join(p.name for p in parts if p.target == cid)))
            continue
        j = Jboy.parse(pac.child(cid, model))
        nodes0 = j.nodes
        bones = {n: (par, W) for n, (par, W, c) in extra_bones.items() if c == cid}
        M, names, added = update_skeleton(j, tnodes, bones, log, "JBOY 0x%x" % cid)
        tpls = extra_templates(j)
        if not tpls and any(p.target == cid for p in parts):
            raise ValueError("JBOY 0x%x of the base has no mesh to use as a template." % cid)
        dest[cid] = dict(j=j, names=names, M=M, label="JBOY 0x%x" % cid, tpls=tpls, added=added,
                         same=j.nodes == nodes0)
    for p in parts:
        d = dest[p.target]
        p.used = sorted({p.bones[i] for i in np.unique(p.WI[(p.WI >= 0) & (p.WW > 1e-5)])})
        if p.target == 0:
            missing = [b for b in p.used if b not in tname2i]
        else:
            missing = []
            for b in p.used:             # body bones missing from the extra JBOY: closest ancestor in it
                n = tname2i.get(b)
                while n is not None and n >= 0 and tnodes[n]["name"] not in d["names"]:
                    n = tnodes[n]["parent"]
                if b not in d["names"] and (n is None or n < 0):
                    missing.append(b)
        if missing:
            raise ValueError("Bones not present in the base skeleton%s: %s" % (
                "" if p.target == 0 else " (%s)" % d["label"], ", ".join(missing)))

    # ---- 1. atlas
    imgs, gray = list(images), None
    if not imgs or any(p.image is None for p in parts):        # parts without image: gray
        gray = len(imgs)
        imgs.append(np.full((16, 16, 4), (160, 160, 160, 255), dtype=np.uint8))
    atlas, rects = build_atlas(imgs)
    AH, AW = atlas.shape[:2]
    log("atlas %dx%d with %d images" % (AW, AH, len(imgs)))

    # ---- 2. meshes
    keep_alpha, unweighted = set(), 0
    for d in dest.values():
        d["tcent"] = [t.positions().mean(0) for t in d["tpls"]]
        d["new"] = []
    for p in parts:
        d = dest[p.target]
        j, names = d["j"], d["names"]
        if p.target == 0:
            bone_names, P, N = p.bones, p.P, p.N
        else:
            def present(b):                  # the bone, or its closest ancestor in this JBOY
                while b not in names:
                    b = tnodes[tnodes[tname2i[b]]["parent"]]["name"]
                return b
            bone_names = [present(b) if b in p.used else b for b in p.bones]
            P, N = p.P, p.N
            mats = np.array([d["M"][names.index(b)] if b in names else np.eye(4) for b in bone_names])
            if np.abs(mats - np.eye(4)).max() > 1e-6:          # the JBOY's bind pose differs from the body's
                P, N = to_bind(P, N, p.WI, p.WW, mats)
        name2i = {n: i for i, n in enumerate(names)}
        UV = p.UV.copy()
        if UV.min() < -1e-3 or UV.max() > 1 + 1e-3:
            log("  " + WARNING + "%s has UVs outside 0..1 (tiled texture): they are clamped" % p.name)
            UV = np.clip(UV, 0.0, 1.0)
        img = p.image if p.image is not None else gray
        x, y, w, h = rects[img]
        uv = np.empty_like(UV)
        uv[:, 0] = (x + UV[:, 0] * w) / AW
        uv[:, 1] = (y + (1.0 - UV[:, 1]) * h) / AH          # Blender: v up; game: v down
        T = p.T
        fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        vn = N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]]
        if ((fn * vn).sum(1) < 0).mean() > 0.5:             # the winding must agree with the normal
            T = T[:, ::-1]                                   # (in the game, both point inwards)
        tex_names = [t.lower() for t in j.texture_names()]
        for ci, chunk in enumerate(split_parts(T, p.WI, p.WW)):
            used = np.unique(chunk)
            remap = -np.ones(len(P), dtype=np.int64)
            remap[used] = np.arange(len(used))
            c = P[used].mean(0)
            cand = list(range(len(d["tpls"])))
            if p.target != 0 and p.alpha is not None:          # same kind: transparent or opaque
                same = [i for i in cand if bool(struct.unpack_from(">I", d["tpls"][i].record, 136)[0]
                                                & ALPHA_FLAG) == p.alpha]
                cand = same or cand
            tpl = d["tpls"][min(cand, key=lambda i: np.linalg.norm(c - d["tcent"][i]))]
            m, nw = make_mesh(P[used], N[used], uv[used], remap[chunk], p.WI[used], p.WW[used],
                              bone_names, name2i, tpl, g_body if p.target == 0 else tpl.material_group)
            if p.target != 0:
                for param, tex in (("texDiffuse", "color"), ("texNormal", "normal")):
                    if param in m.eparams() and tex in tex_names and (j.mesh_texture(m, param) or "").lower() != tex:
                        m.set_eparam(param, tex_names.index(tex))
                if struct.unpack_from(">I", tpl.record, 136)[0] & ALPHA_FLAG:
                    keep_alpha.add(img)
            unweighted += nw
            d["new"].append(m)
            log("  %s.%d: %5d vert %5d tri, strip %5d, %d infl., %2d bones, template %s #%d" % (
                p.name, ci, m.vc, len(chunk), len(m.strips[0]), m.nb, len(m.palette()), d["label"],
                j.meshes.index(tpl)))
    if unweighted:
        log("  " + WARNING + "%d vertices without weights: assigned to the root bone" % unweighted)

    new = {}
    for cid, d in dest.items():
        j = d["j"]
        if cid == 0:
            others = [m for m in tb.meshes if m.material_group != g_body]
            for m in others:
                blank_mesh(m)
            ordered = []
            for g in range(len(groups)):
                ordered += d["new"] if g == g_body else [m for m in others if m.material_group == g]
            tb.meshes = ordered
            set_material_counts(tb, [len(d["new"]) if g == g_body else
                                     sum(1 for m in others if m.material_group == g) for g in range(len(groups))])
            log("body: %d new meshes" % len(d["new"]))
        elif d["new"]:
            # each group gets its new meshes; groups without new ones keep theirs, emptied
            by_group = {}
            for m in d["new"]:
                by_group.setdefault(m.material_group, []).append(m)
            ordered, counts = [], []
            for g in range(len(j.material_names())):
                ms = by_group.get(g)
                if ms is None:
                    ms = [m for m in j.meshes if m.material_group == g]
                    for m in ms:
                        blank_mesh(m)
                ordered += ms
                counts.append(len(ms))
            j.meshes = ordered
            set_material_counts(j, counts)
            log("%s: %d new meshes" % (d["label"], len(d["new"])))
        elif d["same"]:
            continue                               # nothing for it: emptied below like the other extras
        else:                                      # only its skeleton changed: keep it, without triangles
            for m in j.meshes:
                blank_mesh(m)
        node_spheres(j)
        new[cid] = j.build()
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
    log("emptied: base teeth/eyes and JBOY %s" % (", ".join(blanked) or "-"))

    # ---- hair physics (0x3c): edited chains and the ones added in Blender
    params = dict(hair_params or {})
    if pac.has(HAIR_PARAMS_ID, model):
        old = pac.child(HAIR_PARAMS_ID, model)
        base_params = hair_params_read(old)
        default = next(iter(base_params.values()), None)
        added = [n for d in dest.values() for n in d["added"]]
        for n in added:
            if n not in params and extra_bones[n][0] in tname2i and default is not None:   # root of a new chain
                params[n] = list(default)
                log("  hair physics of the new chain %s: copied from the first one" % n)
        params = {n: v for n, v in params.items() if n in base_params or n in added}
        data = hair_params_write(old, params)
        if data != old:
            new[HAIR_PARAMS_ID] = data
            log("hair physics (0x3c): %d chain(s) written" % sum(
                1 for n in params if n not in base_params or not np.allclose(base_params[n], params[n], atol=5e-5)))
    elif any(d["added"] for d in dest.values()):
        log("  " + WARNING + "the base has no hair physics (child 0x3c): the new chains will not move")

    # ---- 3. textures
    arr = atlas.copy()
    gloss = np.where(arr[..., 3] < 128, 0, alpha).astype(np.uint8)  # alpha = gloss (0 where it was transparent)
    keep = np.zeros(arr.shape[:2], dtype=bool)
    for i in keep_alpha:                                     # alpha-blended meshes: alpha = transparency
        x, y, w, h = rects[i]
        keep[y:y + h, x:x + w] = True
    arr[..., 3] = np.where(keep, arr[..., 3], gloss)
    nmap = None
    if normals and any(n is not None for n in normals):
        nmap = np.empty_like(atlas)
        nmap[...] = (128, 128, 255, specular)
        for i, (x, y, w, h) in enumerate(rects):
            if i < len(normals) and normals[i] is not None:
                n = dds.flip_normal_xy(resize_nearest(normals[i], h, w))
                n[..., 3] = specular
                nmap[y:y + h, x:x + w] = n
    found, sizes = set(), {}
    for cid in pac.ids(model):
        data = new.get(cid) or pac.child(cid, model)
        if not dds.is_texpack(data):
            continue
        head, items = dds.texpack_read(data)
        changed = False
        for it in items:
            n = it[0].lower()
            if n == "color":
                px = fit_base(arr, it[1])
                it[1] = dds.make_dds_dxt5(px)
                sizes[n] = px.shape[1::-1]
            elif n == "normal" and nmap is not None:
                px = fit_base(nmap, it[1])
                it[1] = dds.make_dds_dxt5(px)
                sizes[n] = px.shape[1::-1]
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
    if sizes.get("color", (AW, AH)) != (AW, AH):
        log("  atlas %dx%d reduced to the base's texture size (the game does not load bigger ones)" % (AW, AH))
    log("textures: color %dx%d DXT5 (alpha %d%s); normal %s (specular alpha %d); %s" % (
        *sizes.get("color", (AW, AH)), alpha, ", transparency kept on %d image(s)" % len(keep_alpha) if keep_alpha else "",
        "%dx%d from %d normal map(s)" % (*sizes.get("normal", (AW, AH)), sum(n is not None for n in normals))
        if nmap is not None else "flat",
        specular, ", ".join(sorted(found - {"color", "normal"}))))

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
