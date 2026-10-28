"""
exporter.py - reads from Blender a model bound to an armature imported with the add-on and writes a
new ch*.pac on top of the base PAC (pac_builder.replace_model does the format work).

Model requirements:
  - mesh(es) with an Armature modifier pointing at the imported armature (property 'svr_pac');
  - vertex groups named after the armature's bones (other groups are ignored);
  - in every material, an Image Texture node with the color image (they are packed into an atlas);
    optionally, an Image Texture plugged into a Normal Map node (tangent space, OpenGL as Blender).
Long hair: a mesh weighted to bones that only exist in an extra JBOY (the kami_b* chains imported from a
base with long hair) goes into that JBOY (hair), with the chains as they are in the armature (they can be
moved, and chains can be added by duplicating one) and the physics of the root bones' svr_hair_* properties.

Coordinates (inverse of importer.py): Blender (x, y, z) / scale -> game (x, -z, y);
UV: the atlas already applies v' = 1 - v. The armature's REST pose is exported.
Normals: in the game they point INWARDS, so they are flipped (pac_builder makes the triangle
winding agree with them). With outward normals CJ showed a black face: the light from above
darkened the forehead and lit the chin.
"""
import os

import bpy
import numpy as np

from . import pac_builder
from .importer import G2B, HAIR_PROP, WARNING


def find_armature(context):
    """Armature imported with the add-on: the active one, the one of the active/selected mesh, or the only one."""
    def ok(o):
        return o is not None and o.type == "ARMATURE" and "svr_pac" in o
    cands = [context.active_object] + list(context.selected_objects)
    for o in cands:
        if ok(o):
            return o
        if o is not None and o.type == "MESH":
            for m in o.modifiers:
                if m.type == "ARMATURE" and ok(m.object):
                    return m.object
    arms = [o for o in context.scene.objects if ok(o)]
    if len(arms) == 1:
        return arms[0]
    raise ValueError("Armature not found: select the imported armature (or a mesh bound to it).")


def bound_meshes(context, arm, selected_only):
    out = []
    for o in context.scene.objects:
        if o.type != "MESH":
            continue
        if not any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers):
            continue
        if selected_only and not o.select_get():
            continue
        if not o.visible_get():
            continue
        out.append(o)
    return out


def material_image(mat):
    """Color image of a material: the one linked to Base Color/Color, or the first Image Texture."""
    if mat is None or not mat.node_tree:
        return None
    texs = [n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
    for n in texs:
        for link in n.outputs["Color"].links:
            if link.to_socket.name in ("Base Color", "Color"):
                return n.image
    return texs[0].image if texs else None


def material_alpha(mat):
    """True if the material uses transparency (something plugged into, or a value below 1 in, the
    Principled BSDF's Alpha), False if not, None without a Principled BSDF."""
    if mat is None or not mat.node_tree:
        return None
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            a = n.inputs["Alpha"]
            return bool(a.links) or a.default_value < 0.999
    return None


def material_normal(mat):
    """(image, strength) of the Image Texture plugged into a Normal Map node, or (None, 1)."""
    if mat is None or not mat.node_tree:
        return None, 1.0
    for n in mat.node_tree.nodes:
        if n.type != "NORMAL_MAP" or n.space != "TANGENT" or not n.outputs["Normal"].links:
            continue
        for link in n.inputs["Color"].links:
            if link.from_node.type == "TEX_IMAGE" and link.from_node.image:
                return link.from_node.image, float(n.inputs["Strength"].default_value)
    return None, 1.0


def scale_normal(rgba, strength, directx=False):
    """Applies the Normal Map node's Strength (scales x and y and normalizes again).
    directx: the map uses the DirectX convention (Unreal, 3ds Max...): green is inverted to OpenGL's."""
    if directx:
        rgba = rgba.copy()
        rgba[..., 1] = 255 - rgba[..., 1]
    if abs(strength - 1.0) < 1e-6:
        return rgba
    v = rgba[..., :3].astype(np.float64) / 127.5 - 1.0
    v[..., :2] *= strength
    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    out = rgba.copy()
    out[..., :3] = np.clip((v + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)
    return out


def image_rgba(img, scale=1.0):
    """bpy Image -> RGBA uint8 (height, width, 4), row 0 = top. None if it has no pixels."""
    if img is None or img.size[0] == 0 or img.size[1] == 0:
        return None
    src = img
    if scale != 1.0:
        src = img.copy()
        src.scale(max(4, int(img.size[0] * scale)), max(4, int(img.size[1] * scale)))
    w, h = src.size
    px = np.empty(w * h * 4, dtype=np.float32)
    src.pixels.foreach_get(px)
    if src is not img:
        bpy.data.images.remove(src)
    rgba = (np.clip(px, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(h, w, 4)
    return np.flipud(rgba).copy()                      # Blender stores rows bottom to top


def bone_child(bone):
    """JBOY a bone belongs to: 0 = body, otherwise the extra JBOY it was imported from; bones created
    in Blender (no 'svr_child') belong to the JBOY of their closest imported ancestor, or to the hair."""
    b = bone
    while b is not None and "svr_child" not in b:
        b = b.parent
    if b is None or (b is not bone and int(b["svr_child"]) == 0):
        return pac_builder.HAIR_ID
    return int(b["svr_child"])


def mesh_parts(obj, arm, depsgraph, scale, bone_names, images, normals, image_index, atlas_scale, log,
               normal_dx=False):
    """One part per material of the evaluated mesh (in rest pose). Splits vertices by UV and normal.
    The parts go to the body, or to an extra JBOY if the mesh is weighted to bones of one (hair)."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nl = len(me.loops)
        M = arm.matrix_world.inverted() @ ev.matrix_world
        R = np.array(M.to_3x3())
        Mt = np.array(M)
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        co = co @ Mt[:3, :3].T + Mt[:3, 3]
        if hasattr(me, "corner_normals"):                                 # Blender 4.1+
            cn = np.empty(nl * 3); me.corner_normals.foreach_get("vector", cn); cn = cn.reshape(-1, 3)
        else:
            me.calc_normals_split()
            cn = np.empty(nl * 3); me.loops.foreach_get("normal", cn); cn = cn.reshape(-1, 3)
        cn = cn @ R.T
        cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-12)
        if me.uv_layers.active is not None:
            uv = np.empty(nl * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
        else:
            log("  " + WARNING + "%s has no UVs" % obj.name)
            uv = np.zeros((nl, 2))
        loop_v = np.empty(nl, dtype=np.int64); me.loops.foreach_get("vertex_index", loop_v)
        nt = len(me.loop_triangles)
        tl = np.empty(nt * 3, dtype=np.int64); me.loop_triangles.foreach_get("loops", tl); tl = tl.reshape(-1, 3)
        tm = np.empty(nt, dtype=np.int64); me.loop_triangles.foreach_get("material_index", tm)

        # per-vertex weights (only groups that are bones)
        gname = {g.index: g.name for g in ev.vertex_groups}
        bidx = {n: i for i, n in enumerate(bone_names)}
        ignored = set()
        vw = []
        for v in me.vertices:
            acc = {}
            for g in v.groups:
                n = gname.get(g.group)
                if g.weight <= 1e-6 or n is None:
                    continue
                if n not in bidx:
                    ignored.add(n); continue
                acc[bidx[n]] = acc.get(bidx[n], 0.0) + g.weight
            vw.append(sorted(acc.items(), key=lambda kv: -kv[1])[:pac_builder.MAX_INFL])
        if ignored:
            log("  " + WARNING + "%s: groups that are not bones, ignored: %s" % (obj.name, ", ".join(sorted(ignored))))
        used = {bone_names[b] for a in vw for b, _w in a}
        cids = {bone_child(arm.data.bones[n]) for n in used} - {0}
        if len(cids) > 1:
            raise ValueError("%s is weighted to bones of several JBOYs (%s): split it" % (
                obj.name, ", ".join("0x%x" % c for c in sorted(cids))))
        target = cids.pop() if cids else int(obj.get("svr_child", 0))
        if target in pac_builder.BLOOD_IDS:
            target = 0
        if target:
            log("  %s -> JBOY 0x%x" % (obj.name, target))

        parts = []
        for mi in np.unique(tm):
            mat = obj.material_slots[mi].material if mi < len(obj.material_slots) else None
            img = material_image(mat)
            nimg, strength = material_normal(mat)
            if nimg is not None and nimg == img:
                nimg = None
            key = (img.name if img else None, nimg.name if nimg else None, round(strength, 4))
            if key[0] is None and key[1] is None:
                key = None
            elif key[0] is not None:
                # the same color image with and without a normal map shares one place in the atlas
                same = [k for k in image_index if k[0] == key[0] and image_index[k] is not None]
                if key[1] is None and same:
                    key = same[0]
                elif key[1] is not None:
                    bare = next((k for k in same if k[1] is None), None)
                    if bare is not None and normals[image_index[bare]] is None:
                        nrgba = image_rgba(nimg, atlas_scale)
                        if nrgba is not None:
                            normals[image_index[bare]] = scale_normal(nrgba, strength, normal_dx)
                            image_index[key] = image_index[bare]
            if key is not None and key not in image_index:
                rgba = image_rgba(img, atlas_scale) if img else None
                nrgba = image_rgba(nimg, atlas_scale) if nimg else None
                if img is not None and rgba is None:
                    log("  " + WARNING + "image %s has no pixels (missing file?)" % img.name)
                if nimg is not None and nrgba is None:
                    log("  " + WARNING + "normal map %s has no pixels (missing file?)" % nimg.name)
                if rgba is None and nrgba is not None:                # normal map only: gray color
                    rgba = np.full(nrgba.shape, (160, 160, 160, 255), dtype=np.uint8)
                if rgba is None:
                    image_index[key] = None
                else:
                    image_index[key] = len(images); images.append(rgba)
                    normals.append(scale_normal(nrgba, strength, normal_dx) if nrgba is not None else None)
            loops = tl[tm == mi].ravel()
            # one game vertex per distinct (vertex, UV, normal)
            keys = np.concatenate([loop_v[loops, None].astype(np.float64),
                                   np.round(uv[loops], 6), np.round(cn[loops], 4)], axis=1)
            _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            src = loops[first]
            vids = loop_v[src]
            k = max(1, max((len(vw[v]) for v in vids), default=1))
            WI = -np.ones((len(src), k), dtype=np.int64); WW = np.zeros((len(src), k))
            for i, v in enumerate(vids):
                for j, (b, w) in enumerate(vw[v]):
                    WI[i, j], WW[i, j] = b, w
            P = (co[vids] / scale) @ G2B                  # Blender -> game
            N = -(cn[src] @ G2B)                         # the game stores normals pointing inwards
            T = inv.reshape(-1, 3)
            name = "%s/%s" % (obj.name, mat.name if mat else "no_material")
            parts.append(pac_builder.Part(name, P, N, uv[src], T, WI, WW, bone_names,
                                          image_index.get(key) if key else None, target, material_alpha(mat)))
        return parts
    finally:
        ev.to_mesh_clear()


def extra_skeleton(arm, scale):
    """Bones of the extra JBOYs (hair chains) as they are in the armature's rest pose, and the hair
    physics stored on their root bones. -> ({name: (parent, W 4x4 game space, child id)}, {root: [9 values]})
    The game frame of each bone comes from its rest matrix and 'svr_rot' (stored by the importer);
    bones without it (created in Blender) take bone Y = node X and bone Z = node Z, as in the chains."""
    bones, params = {}, {}
    for b in arm.data.bones:
        cid = bone_child(b)
        if cid == 0:
            continue
        if b.parent is None:
            raise ValueError("Bone %s has no parent: hair bones must hang from the head (atama)." % b.name)
        B = np.array(b.matrix_local)
        C = np.array(b["svr_rot"], dtype=np.float64).reshape(3, 3) if "svr_rot" in b else \
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        W = np.eye(4)
        W[:3, :3] = G2B.T @ B[:3, :3] @ C
        W[:3, 3] = G2B.T @ B[:3, 3] / scale
        bones[b.name] = (b.parent.name, W, cid)
        if bone_child(b.parent) == 0 and HAIR_PROP % pac_builder.HAIR_PARAMS[0] in b:
            params[b.name] = [float(b.get(HAIR_PROP % k, 0.0)) for k in pac_builder.HAIR_PARAMS]
    return bones, params


def export_pac(context, filepath, base_path="", alpha=47, atlas_scale=1.0, emd_id=-1, selected_only=False,
               specular=12, blood=True, normal_dx=False):
    log_lines = []

    def log(s):
        print(s); log_lines.append(s)

    arm = find_armature(context)
    base = base_path or arm["svr_pac"]
    if not os.path.exists(base):
        raise ValueError("Base PAC not found: %s" % base)
    #if os.path.abspath(base) == os.path.abspath(filepath):
    #    raise ValueError("The output file cannot be the base PAC.")
    meshes = bound_meshes(context, arm, selected_only)
    if not meshes:
        raise ValueError("No visible meshes bound to %s (Armature modifier)." % arm.name)
    scale = float(arm.get("svr_scale", 0.1))
    model = int(arm.get("svr_model", 0))
    bone_names = [b.name for b in arm.data.bones]
    log("base %s (outfit %d), armature %s, %d mesh(es): %s" % (
        os.path.basename(base), model, arm.name, len(meshes), ", ".join(o.name for o in meshes)))

    old_pose = arm.data.pose_position
    arm.data.pose_position = "REST"                    # the rest pose is exported (= the game's bind pose)
    try:
        context.view_layer.update()
        dg = context.evaluated_depsgraph_get()
        images, normals, image_index, parts = [], [], {}, []
        for o in meshes:
            parts += mesh_parts(o, arm, dg, scale, bone_names, images, normals, image_index, atlas_scale, log,
                                normal_dx)
        extra_bones, hair_params = extra_skeleton(arm, scale)
    finally:
        arm.data.pose_position = old_pose
        context.view_layer.update()

    with open(base, "rb") as f:
        base_data = f.read()
    data = pac_builder.replace_model(base_data, parts, images, model=model, alpha=alpha,
                                     emd_id=None if emd_id < 0 else emd_id, log=log, specular=specular,
                                     blood=blood, normals=normals, extra_bones=extra_bones,
                                     hair_params=hair_params)
    with open(filepath, "wb") as f:
        f.write(data)
    log("Written %s (%d bytes)" % (filepath, len(data)))
    return log_lines
