"""
importer.py - builds the model of a ch*.pac in Blender: armature, meshes, weights, UVs and texture.

Coordinates: the game uses -Y up with the face towards -Z; Blender uses Z up with the face towards -Y.
  game (x, y, z) -> Blender (x, z, -y) * scale ;  UV: v' = 1 - v
Orientation: in the game's PACs the normals and the triangle winding point INWARDS (negative signed
volume; verified on every base PAC). Both are flipped on import so they point outwards in Blender
(blue faces in Face Orientation); exporter.py does the opposite.
The vertices are already in bind pose, so the armature is created in that pose.
"""
import os

import bpy
import numpy as np
from mathutils import Vector

from . import dds
from .jboy import NO_PAL, Jboy
from .pac_container import Pac

# game -> Blender (proper rotation, det = +1: it does not change the triangle winding)
G2B = np.array([[1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0]])

TEXPACKS = (0xA, 0xB, 0xC, 0xD, 0xE, 0xF)
LABELS = {0: "body", 0x2710: "hair", 0x4E20: "lash",
          0x7530: "blood_face", 0x7531: "blood_body", 0x7532: "blood_hair"}
BLOOD = (0x7530, 0x7531, 0x7532)       # blood layers (textures BloodFace/BloodBody/BloodHair)
WARNING = "warning: "                  # prefix of report lines shown as warnings by the operator


def child_label(cid):
    return LABELS.get(cid, "extra_%x" % cid)


# ===================================================================== textures
class TextureLibrary:
    """Looks textures up by name (case-insensitive) in the PAC texture packs and creates
    each Blender image only once."""

    def __init__(self, pac, model, prefix):
        self.prefix = prefix
        self.dds = {}
        for cid in TEXPACKS:
            if not pac.has(cid, model):
                continue
            data = pac.child(cid, model)
            if dds.is_texpack(data):
                for name, d in dds.texpack_read(data)[1]:
                    self.dds.setdefault(name.lower(), (name, d))
        self.images = {}
        self.missing = set()

    def image(self, name):
        if name is None:
            return None
        key = name.lower()
        if key in self.images:
            return self.images[key]
        img = None
        if key in self.dds:
            real, d = self.dds[key]
            try:
                px = dds.decode(d)
            except ValueError:
                px = None
            if px is not None:
                h, w = px.shape[:2]
                img = bpy.data.images.new("%s_%s" % (self.prefix, real), w, h, alpha=True)
                # the DDS goes top to bottom; Blender stores rows bottom to top
                img.pixels.foreach_set((np.flipud(px).astype(np.float32) / 255.0).ravel())
                img.alpha_mode = "CHANNEL_PACKED"            # alpha = gloss, not transparency
                img.pack()
        else:
            self.missing.add(name)
        self.images[key] = img
        return img


def make_material(name, image, shader):
    mat = bpy.data.materials.new(name)
    mat["svr_shader"] = shader
    if hasattr(mat, "use_nodes") and not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.location = (bsdf.location.x - 320, bsdf.location.y)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        # the alpha of 'color' is the game's gloss: it is not connected (still available on the node)
    return mat


# ===================================================================== armature
def build_armature(context, nodes, scale, name, collection):
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    collection.objects.link(obj)
    view_layer = context.view_layer
    prev = view_layer.objects.active
    view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    heads = [scale * (G2B @ n["W"][:3, 3]) for n in nodes]
    rots = [G2B @ n["W"][:3, :3] for n in nodes]
    children = {i: [] for i in range(len(nodes))}
    for i, n in enumerate(nodes):
        if n["parent"] >= 0:
            children[n["parent"]].append(i)

    min_len = 0.02 * scale / 0.1
    lengths = {}
    ebones = []
    for i, n in enumerate(nodes):
        eb = arm.edit_bones.new(n["name"])
        head = heads[i]
        kids = [heads[c] for c in children[i] if np.linalg.norm(heads[c] - head) > 1e-4]
        if kids:
            tail = np.mean(kids, axis=0)
            if np.linalg.norm(tail - head) < min_len:
                tail = kids[0]
        elif n["parent"] >= 0:
            d = head - heads[n["parent"]]
            nd = np.linalg.norm(d)
            d = d / nd if nd > 1e-6 else rots[i][:, 1]
            tail = head + d * max(lengths.get(n["parent"], min_len) * 0.5, min_len)
        else:
            tail = head + np.array([0.0, 0.0, min_len * 5])
        if np.linalg.norm(tail - head) < min_len:
            tail = head + (tail - head if np.linalg.norm(tail - head) > 1e-6 else np.array([0, 0, 1.0])) \
                / max(np.linalg.norm(tail - head), 1e-6) * min_len
        lengths[i] = float(np.linalg.norm(tail - head))
        eb.head = Vector(head)
        eb.tail = Vector(tail)
        eb.align_roll(Vector(rots[i][:, 2]))           # bone Z axis ~ Z axis of the game node
        # original node data, for the exporter
        eb["svr_index"] = n.get("index", i)
        eb["svr_child"] = n.get("child", 0)
        eb["svr_t"] = [float(v) for v in n["t"]]
        eb["svr_r"] = [float(v) for v in n["r"]]
        ebones.append(eb)
    for eb, n in zip(ebones, nodes):                   # a parent may come after its child
        if n["parent"] >= 0:
            eb.parent = ebones[n["parent"]]

    bpy.ops.object.mode_set(mode="OBJECT")
    if prev is not None:
        view_layer.objects.active = prev
    return obj


# ===================================================================== meshes
def build_mesh(name, P, N, UV, T, groups, collection, arm_obj, mat):
    """P, N: (vc, 3) in Blender space; UV (vc, 2) already flipped; T (nt, 3);
    groups: {bone_name: {vertex: weight}}."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(P.tolist(), [], T.tolist())
    me.validate(clean_customdata=False)
    uvl = me.uv_layers.new(name="UVMap")
    loop_v = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    uvl.data.foreach_set("uv", UV[loop_v].astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    nn = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    me.normals_split_custom_set_from_vertices(nn.tolist())
    if mat is not None:
        me.materials.append(mat)
    me.update()

    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)
    obj.parent = arm_obj
    for bone, vw in groups.items():
        vg = obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
        for v, w in vw.items():
            vg.add([v], w, "REPLACE")
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj
    return obj


def mesh_groups(m, node_names):
    """Weights of a mesh grouped by bone name."""
    pal = m.palette()
    wi, ww = m.influences()
    groups = {}
    covered = np.zeros(m.vc, dtype=bool)
    for v, j in zip(*np.nonzero((ww > 1e-6) & (wi != NO_PAL))):
        p = wi[v, j]
        if p >= len(pal) or not (0 <= pal[p] < len(node_names)):
            continue
        g = groups.setdefault(node_names[pal[p]], {})
        g[int(v)] = g.get(int(v), 0.0) + float(ww[v, j])
        covered[v] = True
    if not covered.all() and pal and 0 <= pal[0] < len(node_names):
        # vertices without influences go to the first palette bone so they do not stay behind
        g = groups.setdefault(node_names[pal[0]], {})
        for v in np.nonzero(~covered)[0]:
            g[int(v)] = 1.0
    return groups


def rebind(m, M, P, N):
    """Moves the vertices from the bind pose of their own JBOY to the armature's.
    M: {node index: 4x4 matrix (W_target * inverse(W_own))}."""
    pal = m.palette()
    wi, ww = m.influences()
    ok = (wi != NO_PAL) & (wi < len(pal)) & (ww > 1e-6)
    ww = np.where(ok, ww, 0.0)
    first = next(p for p in pal if p in M)
    none = ww.sum(1) <= 0
    ww[none, 0], wi[none, 0] = 1.0, pal.index(first)    # no weights: first palette bone
    ww = ww / ww.sum(1, keepdims=True)
    mats = np.array([M.get(p, M[first]) for p in pal] + [np.eye(4)])
    sel = np.where(wi < len(pal), wi, len(pal))
    A = np.einsum("vk,vkij->vij", ww, mats[sel])            # per-vertex blend of matrices
    P2 = np.einsum("vij,vj->vi", A[:, :3, :3], P) + A[:, :3, 3]
    N2 = np.einsum("vij,vj->vi", A[:, :3, :3], N)
    return P2, N2


# ===================================================================== entry point
def import_pac(context, filepath, scale=0.1, model=0, extras=True, textures=True, blood=False):
    pac = Pac.open(filepath)
    if model >= len(pac.models):
        raise ValueError("The PAC only has %d outfit(s) (PACH)." % len(pac.models))
    if not pac.has(0, model):
        raise ValueError("The PAC has no child 0 (body).")
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if len(pac.models) > 1:
        stem += "_outfit%d" % model

    coll = bpy.data.collections.new(stem)
    context.scene.collection.children.link(coll)

    body = Jboy.parse(pac.child(0, model))
    bones = body.node_list()
    for i, n in enumerate(bones):
        n["name"] = n["name"] or "node_%d" % i
        n["index"], n["child"] = i, 0
    n_body = len(bones)

    jboys = [(0, body)]
    if extras:
        for cid in pac.ids(model):
            if cid != 0 and (blood or cid not in BLOOD):
                data = pac.child(cid, model)
                if data[:4] == b"JBOY":
                    jboys.append((cid, Jboy.parse(data)))

    # Every JBOY carries its own skeleton in bind pose. Bones are matched to the body's by name;
    # those that only exist in an extra JBOY (e.g. kami_b*, the hair) are added to the armature,
    # hanging from their first common ancestor. If a JBOY's bind pose differs from the body's
    # (ch125: eyelashes ~9 units lower), its vertices are moved to the body's pose with
    # W_body * inverse(W_own) of each bone, weighted by the vertex weights.
    index = {n["name"]: i for i, n in enumerate(bones)}
    work = []
    agree = total = 0
    rebound = set()
    for cid, j in jboys:
        nl = bones[:n_body] if cid == 0 else j.node_list()
        names = [n["name"] or "node_%x_%d" % (cid, i) for i, n in enumerate(nl)]
        target = [None] * len(nl)               # W of each node in armature space

        def resolve(i, seen=()):
            if target[i] is None:
                if names[i] in index:
                    target[i] = bones[index[names[i]]]["W"]
                else:
                    p = nl[i]["parent"]
                    if 0 <= p < len(nl) and p not in seen:
                        target[i] = resolve(p, seen + (i,)) @ np.linalg.inv(nl[p]["W"]) @ nl[i]["W"]
                    else:
                        target[i] = nl[i]["W"]
            return target[i]

        def add_bone(i, seen=()):
            if names[i] in index:
                return index[names[i]]
            p = nl[i]["parent"]
            parent = add_bone(p, seen + (i,)) if 0 <= p < len(nl) and p not in seen else -1
            bones.append(dict(name=names[i], W=resolve(i), parent=parent, t=nl[i]["t"], r=nl[i]["r"],
                              index=i, child=cid))
            index[names[i]] = len(bones) - 1
            return index[names[i]]

        for k, m in enumerate(j.meshes):
            T = m.triangles()
            if len(T) == 0:                      # emptied mesh (only degenerate triangles)
                continue
            pal = [p for p in m.palette() if 0 <= p < len(nl)]
            for p in pal:
                add_bone(p)
            P, N = m.positions(), m.normals()
            if cid != 0 and pal:
                M = {p: resolve(p) @ np.linalg.inv(nl[p]["W"]) for p in pal}
                dev = max(np.abs(x - np.eye(4)).max() for x in M.values())
                if dev > 1e-6:
                    P, N = rebind(m, M, P, N)
                if dev > 0.05:                   # only reported when noticeable (game units)
                    rebound.add(child_label(cid))
            fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
            vn = N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]]
            agree += int((np.einsum("ij,ij->i", fn, vn) > 0).sum())        # winding consistent with normal
            total += len(T)
            work.append((cid, j, k, m, T, names, P, N))
    # the winding has to follow the normal; since the normal is flipped (game: inwards), the winding
    # is flipped too, except in the unusual case where they did not agree in the PAC
    flip = not (total and agree < total / 2)

    arm_obj = build_armature(context, bones, scale, stem + "_armature", coll)
    arm_obj["svr_pac"] = filepath
    arm_obj["svr_model"] = model
    arm_obj["svr_scale"] = scale

    texlib = TextureLibrary(pac, model, stem)
    materials = {}
    n_vert = 0
    for cid, j, k, m, T, names, P, N in work:
        if flip:
            T = T[:, ::-1]
        P = scale * (P @ G2B.T)
        N = -(N @ G2B.T)                         # game: inwards -> Blender: outwards
        UV = m.uvs().copy()
        UV[:, 1] = 1.0 - UV[:, 1]
        groups = mesh_groups(m, names)

        tex = j.mesh_texture(m) if textures else None
        key = (tex.lower() if tex else None, m.shader)
        if key not in materials:
            img = texlib.image(tex) if textures else None
            materials[key] = make_material("%s_%s_%s" % (stem, tex or "no_texture", m.shader), img, m.shader)
        name = "%s_%s_%02d" % (stem, child_label(cid), k)
        obj = build_mesh(name, P, N, UV, T, groups, coll, arm_obj, materials[key])
        obj["svr_child"] = cid
        obj["svr_mesh"] = k
        obj["svr_shader"] = m.shader
        obj["svr_material_group"] = m.material_group
        n_vert += m.vc

    report = ["%d bones (%d from the body), %d meshes, %d vertices" % (len(bones), n_body, len(work), n_vert)]
    if not flip:
        report.append(WARNING + "unusual triangle winding in this PAC (it did not match the normals)")
    if len(pac.models) > 1:
        report.append("outfit %d of %d" % (model, len(pac.models)))
    if rebound:
        report.append("bind pose different from the body's, corrected in: " + ", ".join(sorted(rebound)))
    if texlib.missing:
        report.append(WARNING + "textures not found: " + ", ".join(sorted(texlib.missing)))
    return arm_obj, report
