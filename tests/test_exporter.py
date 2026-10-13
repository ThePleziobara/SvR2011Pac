"""
Exporter round-trip test: exports the meshes bound to the armature of a .blend, imports the
resulting PAC again and compares it with the original model.

    blender -b --factory-startup ../CJ/escena/CJ_svr.blend --python tests/test_exporter.py -- /tmp/cj.pac [render.png]

Checks:
  1) the exported PAC repacks identically and its JBOYs rebuild identically;
  2) same triangle count as the Blender model;
  3) every re-imported vertex lies on a vertex of the original (float32 precision);
  4) the per-bone weights match the original's (after limiting to 4 and normalizing);
  5) with the same pose on both armatures, the deformed vertices match;
  6) orientation as in the game: negative signed volume and winding agreeing with the normals
     (both inwards), and the re-imported normals match Blender's.
The .blend is not saved.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import svr_pac  # noqa: E402
from svr_pac import exporter  # noqa: E402
from svr_pac.jboy import Jboy  # noqa: E402
from svr_pac.pac_container import Pac, build_pac  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
render = argv[1] if len(argv) > 1 else None
svr_pac.register()
ctx = bpy.context
ok = True

arm = exporter.find_armature(ctx)
src = exporter.bound_meshes(ctx, arm, False)
log = exporter.export_pac(ctx, out)

# 1) format
pac = Pac.open(out)
same = build_pac(pac.data, pac.models) == pac.data
jb = all(Jboy.parse(pac.child(c)).build() == pac.child(c) for c in pac.ids() if pac.child(c)[:4] == b"JBOY")
ok &= same and jb
print("1) PAC repacked identically: %s; JBOYs rebuilt identically: %s" % (same, jb))


def game_orientation(data):
    """(signed volume, fraction of triangles whose winding agrees with the normal) of the body."""
    j = Jboy.parse(data); vol = 0.0; agree = tot = 0
    for m in j.meshes:
        t = m.triangles()
        if not len(t):
            continue
        p, n = m.positions(), m.normals()
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        vol += np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
        agree += int(((np.cross(b - a, c - a) * (n[t[:, 0]] + n[t[:, 1]] + n[t[:, 2]])).sum(1) > 0).sum())
        tot += len(t)
    return vol, agree / max(tot, 1)


vb, ab = game_orientation(Pac.open(arm["svr_pac"]).child(0))
vo, ao = game_orientation(pac.child(0))
orient_ok = vo < 0 and vb < 0 and ao > 0.99
ok &= orient_ok
print("6) orientation in the game: base volume %+.1f (winding=normal %.1f%%), exported %+.1f (%.1f%%)  %s" % (
    vb, 100 * ab, vo, 100 * ao, "OK" if orient_ok else "FAIL"))


def evaluated(objs, arm_obj):
    """World vertices, triangle count and weights {bone: w} of the evaluated meshes."""
    dg = ctx.evaluated_depsgraph_get()
    P, W, nt = [], [], 0
    for o in objs:
        e = o.evaluated_get(dg); me = e.to_mesh(); me.calc_loop_triangles()
        nt += len(me.loop_triangles)
        names = {g.index: g.name for g in o.vertex_groups}
        for v in me.vertices:
            P.append(tuple(e.matrix_world @ v.co))
            acc = {names[g.group]: g.weight for g in v.groups if g.weight > 1e-6 and names.get(g.group) in arm_obj.data.bones}
            top = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
            s = sum(w for _, w in top) or 1.0
            W.append({k: w / s for k, w in top})
        e.to_mesh_clear()
    return np.array(P), W, nt


P0, W0, nt0 = evaluated(src, arm)
res = bpy.ops.import_scene.svr_pac(filepath=out, extras=False)
assert res == {"FINISHED"}, res
arm2 = ctx.view_layer.objects.active
new = [o for o in bpy.data.objects if o.type == "MESH" and o.get("svr_child") == 0
       and any(m.object == arm2 for m in o.modifiers if m.type == "ARMATURE")]
P1, W1, nt1 = evaluated(new, arm2)


# 6b) normals are compared per corner (each re-imported corner must have, at the same spot, an
# original corner with the same normal; summing per vertex fails on hard edges and double-sided parts)
def corners(objs):
    P, N = [], []
    for o in objs:
        me = o.data
        co = np.array([v.co for v in me.vertices])
        cn = np.empty(len(me.loops) * 3); me.corner_normals.foreach_get("vector", cn); cn = cn.reshape(-1, 3)
        lv = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get("vertex_index", lv)
        Mw = np.array(o.matrix_world)
        P.append(co[lv] @ Mw[:3, :3].T + Mw[:3, 3]); N.append(cn @ Mw[:3, :3].T)
    N = np.concatenate(N)
    return np.concatenate(P), N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


# 2) triangles
ok &= nt0 == nt1
print("2) triangles: original %d, re-imported %d  %s" % (nt0, nt1, "OK" if nt0 == nt1 else "FAIL"))

# 3) positions (each re-imported vertex -> closest original vertex)
size = np.ptp(P0, axis=0).max()
d = np.sqrt(((P1[:, None, :] - P0[None, :, :]) ** 2).sum(-1))
near = d.argmin(1)
err = d[np.arange(len(P1)), near].max() / size
ok &= err < 1e-5
print("3) vertices: original %d, re-imported %d (split by UV), max error %.1e (rel.)  %s" % (
    len(P0), len(P1), err, "OK" if err < 1e-5 else "FAIL"))

# 4) weights
werr = 0.0
for i, j in enumerate(near):
    a, b = W1[i], W0[j]
    for k in set(a) | set(b):
        werr = max(werr, abs(a.get(k, 0.0) - b.get(k, 0.0)))
ok &= werr < 1e-4
print("4) weights: max difference %.1e  %s" % (werr, "OK" if werr < 1e-4 else "FAIL"))

C0, n0 = corners(src)
C1, n1 = corners(new)
best = np.array([(n0[np.linalg.norm(C0 - C1[i], axis=1) < 1e-5 * size] @ n1[i]).max(initial=-2) for i in range(len(C1))])
nok = (best > 0.99).mean()
ok &= nok == 1.0
print("   normals: %d corners, equal to Blender's %.1f%% (min cosine %.3f)  %s" % (
    len(C1), 100 * nok, best.min(), "OK" if nok == 1.0 else "FAIL"))

# 5) same (local) pose on both armatures
pose = (("l_ninoude", "Y", -50), ("r_kote", "Z", 70), ("l_momo", "X", -35), ("r_sune", "X", 50),
        ("atama", "Z", 30), ("d_kuchi", "X", 20), ("mune", "X", 15))
for a in (arm, arm2):                     # same rest pose and object matrix: same rotation in armature space
    for bone, ax, ang in pose:
        pb = a.pose.bones[bone]
        hd = pb.matrix.to_translation()
        pb.matrix = Matrix.Translation(hd) @ Matrix.Rotation(math.radians(ang), 4, ax) @ Matrix.Translation(-hd) @ pb.matrix
        ctx.view_layer.update()
Q0, _, _ = evaluated(src, arm)
Q1, _, _ = evaluated(new, arm2)
perr = np.abs(Q1 - Q0[near]).max() / size
moved = int((np.linalg.norm(Q0 - P0, axis=1) > 1e-3).sum())
ok &= perr < 1e-4 and moved > 0
print("5) posed: %d vertices moved, max error %.1e (rel.)  %s" % (moved, perr, "OK" if perr < 1e-4 else "FAIL"))

if render:
    for o in src + [arm]:
        o.hide_render = True
    sc = ctx.scene
    cam = bpy.data.objects.new("cam_test", bpy.data.cameras.new("cam_test")); sc.collection.objects.link(cam)
    cam.location = (0.9, -3.6, 0.15); cam.rotation_euler = (math.radians(90), 0, math.radians(14)); sc.camera = cam
    sc.render.engine = "BLENDER_WORKBENCH"; sc.display.shading.color_type = "TEXTURE"
    sc.render.resolution_x, sc.render.resolution_y = 500, 700
    sc.render.filepath = os.path.abspath(render)
    bpy.ops.render.render(write_still=True)
    print("render:", sc.render.filepath)

print("\nALL OK" if ok else "\nTHERE ARE FAILURES")
sys.exit(0 if ok else 1)
