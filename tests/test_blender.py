"""
Importer test inside Blender, compared against the FBX written by pac2fbx.py.

    python3 ../pac2fbx.py ../ch100.pac -o /tmp/ch100.fbx
    blender -b --factory-startup --python tests/test_blender.py -- ../ch100.pac /tmp/ch100.fbx [render.png] [--blood]

Checks (with both scales normalized):
  1) the heads of the 102 bones match;
     the imported model points outwards in Blender (positive volume, winding = normals);
  2) the vertices of every mesh match at rest;
  3) after rotating l_ninoude, l_kote, atama and d_kuchi (same world-space rotation on both
     armatures), the deformed vertices still match.
(Meshes of extra JBOYs may differ slightly: the add-on moves them to the body's bind pose and
pac2fbx.py does not; on ch100 the hair differs by ~3.5e-4.)
Optionally renders the result at rest (front view, textured) to a PNG.
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

argv = sys.argv[sys.argv.index("--") + 1:]
pac_path, fbx_path = argv[0], argv[1]
render = next((a for a in argv[2:] if not a.startswith("--")), None)

bpy.ops.wm.read_factory_settings(use_empty=True)
svr_pac.register()

res = bpy.ops.import_scene.svr_pac(filepath=os.path.abspath(pac_path), blood="--blood" in argv)
assert res == {"FINISHED"}, res
ours = next(o for o in bpy.data.objects if o.type == "ARMATURE")
stem = ours.name[:-len("_armature")]
our_meshes = [o for o in bpy.data.objects if o.type == "MESH"]

bpy.ops.import_scene.fbx(filepath=os.path.abspath(fbx_path))
theirs = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o is not ours)
their_meshes = [o for o in bpy.data.objects if o.type == "MESH" and o not in our_meshes]
bpy.context.view_layer.update()


def heads(arm):
    return {b.name: np.array(arm.matrix_world @ b.head_local) for b in arm.data.bones}


H1, H2 = heads(ours), heads(theirs)
names = sorted(set(H1) & set(H2))
A = np.array([H1[n] for n in names]); B = np.array([H2[n] for n in names])
s = np.linalg.norm(A - A.mean(0)) / np.linalg.norm(B - B.mean(0))      # relative scale
size = np.ptp(A, axis=0).max()
err = np.abs(A - B * s).max() / size
ok = len(names) == 102 and err < 1e-4
print("1) common bones %d, FBX->add-on scale %.5f, max error %.2e (rel.)  %s" % (len(names), s, err, "OK" if ok else "FAIL"))
print("   character height: %.3f m" % np.ptp(A[:, 2]))


def fbx_name(o):
    # 'ch100_body_03' -> 'body_03_<shader without y>' ; extras: 'extra_7530' -> 'extra30000'
    label = o.name[len(stem) + 1:]
    part, k = label.rsplit("_", 1)
    blood = {"blood_face": 0x7530, "blood_body": 0x7531, "blood_hair": 0x7532}
    if part in blood:
        part = "extra%d" % blood[part]
    elif part.startswith("extra_"):
        part = "extra%d" % int(part[6:], 16)
    return "%s_%s_%s" % (part, k, o["svr_shader"][1:])


def verts(o):
    dg = bpy.context.evaluated_depsgraph_get()
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    v = np.array([e.matrix_world @ x.co for x in me.vertices])
    e.to_mesh_clear()
    return v


# in Blender the imported model must point outwards (blue faces in Face Orientation)
vol = agree = tot = 0
for o in our_meshes:
    me = o.data
    me.calc_loop_triangles()
    co = np.array([v.co for v in me.vertices])
    cn = np.empty(len(me.loops) * 3); me.corner_normals.foreach_get("vector", cn); cn = cn.reshape(-1, 3)
    for t in me.loop_triangles:
        a, b, c = co[list(t.vertices)]
        vol += np.dot(a, np.cross(b, c)) / 6
        agree += np.dot(np.cross(b - a, c - a), cn[list(t.loops)].sum(0)) > 0
        tot += 1
orient = vol > 0 and agree / tot > 0.99
ok &= orient
print("   orientation in Blender: volume %+.3f, winding=normal %.1f%%  %s" % (vol, 100 * agree / tot, "OK" if orient else "FAIL"))

pairs = []
by_name = {o.name: o for o in their_meshes}
for o in our_meshes:
    t = by_name.get(fbx_name(o))
    if t is None:
        print("   no match in the FBX:", o.name, fbx_name(o)); ok = False
    else:
        pairs.append((o, t))
print("   meshes: add-on %d, FBX %d, matched %d" % (len(our_meshes), len(their_meshes), len(pairs)))


def compare(tag):
    worst = 0.0
    good = True
    for o, t in pairs:
        a, b = verts(o), verts(t)
        if a.shape != b.shape:
            print("   %s: %d vs %d vertices" % (o.name, len(a), len(b))); good = False; continue
        worst = max(worst, np.abs(a - b * s).max() / size)
    good &= worst < 1e-3
    print("%s max vertex error %.2e (rel.)  %s" % (tag, worst, "OK" if good else "FAIL"))
    return good


ok &= compare("2) rest:")

if render:
    for o in [theirs] + their_meshes:
        o.hide_render = True
    sc = bpy.context.scene
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    zc = np.ptp(A[:, 2]) * 0.5 + A[:, 2].min()
    cam.location = (0.0, -np.ptp(A[:, 2]) * 2.2, zc)
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.camera = cam
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.color_type = "TEXTURE"
    sc.display.shading.light = "STUDIO"
    sc.render.resolution_x, sc.render.resolution_y = 600, 900
    sc.render.filepath = os.path.abspath(render)
    bpy.ops.render.render(write_still=True)
    print("render:", sc.render.filepath)


# same rotation (in world space, around the bone head) on both armatures
for arm in (ours, theirs):
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for bone, axis, ang in (("l_ninoude", "X", 50), ("l_kote", "Z", 60), ("atama", "Y", 30), ("d_kuchi", "X", 25)):
        pb = arm.pose.bones[bone]
        bpy.context.view_layer.update()
        Mw = arm.matrix_world
        head = (Mw @ pb.matrix).to_translation()
        R = Matrix.Translation(head) @ Matrix.Rotation(math.radians(ang), 4, axis) @ Matrix.Translation(-head)
        pb.matrix = Mw.inverted() @ R @ Mw @ pb.matrix
        bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()
ok &= compare("3) posed:")

moved = 0
for o, t in pairs:
    o_rest = np.array([v.co for v in o.data.vertices])
    moved += int((np.linalg.norm(verts(o) - o_rest, axis=1) > 1e-4).sum())
print("   vertices moved by the pose: %d" % moved)
ok &= moved > 0

print("\nALL OK" if ok else "\nTHERE ARE FAILURES")
sys.exit(0 if ok else 1)
