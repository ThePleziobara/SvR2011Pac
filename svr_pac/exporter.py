"""
exporter.py - reads from Blender a model bound to an armature imported with the add-on and writes a
new ch*.pac on top of the base PAC (pac_builder.replace_model does the format work).

Model requirements:
  - mesh(es) with an Armature modifier pointing at the imported armature (property 'svr_pac');
  - vertex groups named after the armature's bones (other groups are ignored);
  - in every material, an Image Texture node with the color image (they are packed into an atlas).

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
from .importer import G2B, WARNING


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
        if o.type != "MESH" in o:          
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


def mesh_parts(obj, arm, depsgraph, scale, bone_names, images, image_index, atlas_scale, log):
    """One part per material of the evaluated mesh (in rest pose). Splits vertices by UV and normal."""
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

        parts = []
        for mi in np.unique(tm):
            mat = obj.material_slots[mi].material if mi < len(obj.material_slots) else None
            img = material_image(mat)
            key = img.name if img else None
            if key is not None and key not in image_index:
                rgba = image_rgba(img, atlas_scale)
                if rgba is None:
                    log("  " + WARNING + "image %s has no pixels (missing file?)" % img.name)
                    image_index[key] = None
                else:
                    image_index[key] = len(images); images.append(rgba)
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
                                          image_index.get(key) if key else None))
        return parts
    finally:
        ev.to_mesh_clear()


def export_pac(context, filepath, base_path="", alpha=47, atlas_scale=1.0, emd_id=-1, selected_only=False,
               specular=12, blood=True):
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
        images, image_index, parts = [], {}, []
        for o in meshes:
            parts += mesh_parts(o, arm, dg, scale, bone_names, images, image_index, atlas_scale, log)
    finally:
        arm.data.pose_position = old_pose
        context.view_layer.update()

    with open(base, "rb") as f:
        base_data = f.read()
    data = pac_builder.replace_model(base_data, parts, images, model=model, alpha=alpha,
                                     emd_id=None if emd_id < 0 else emd_id, log=log, specular=specular,
                                     blood=blood)
    with open(filepath, "wb") as f:
        f.write(data)
    log("Written %s (%d bytes)" % (filepath, len(data)))
    return log_lines
