"""
operators.py - operators and menu entries:
  File > Import > SvR 2011 character (.pac)
  File > Export > SvR 2011 character (.pac)
"""
import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import exporter, importer

WARNING = "warning:"        # prefix of report/log lines shown as warnings


def _report_lines(op, lines):
    for line in lines:
        line = line.strip()
        if line.lower().startswith(WARNING):
            op.report({"WARNING"}, line[len(WARNING):].strip())


class IMPORT_SCENE_OT_svr_pac(bpy.types.Operator, ImportHelper):
    """Import the model of a SvR 2011 (PC) ch*.pac: armature, meshes, weights, UVs and texture"""
    bl_idname = "import_scene.svr_pac"
    bl_label = "Import ch*.pac"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filename_ext = ".pac"
    filter_glob: StringProperty(default="*.pac", options={"HIDDEN"})

    scale: FloatProperty(
        name="Scale", default=0.1, min=0.0001, soft_max=10.0,
        description="Scale of the game units (0.1: the character is ~1.8 m tall)")
    model: IntProperty(
        name="Outfit", default=0, min=0,
        description="PACH to import: some PACs contain several outfits (0 = the first one)")
    extras: BoolProperty(
        name="Hair, eyelashes and extras", default=True,
        description="Also import the other JBOY models of the PAC (hair, eyelashes, accessories)")
    textures: BoolProperty(
        name="Textures", default=True,
        description="Create the materials with the PAC 'color' texture (its alpha is the game's gloss)")
    blood: BoolProperty(
        name="Blood layers", default=False,
        description="Also import the blood meshes (JBOY 0x7530-0x7532), which cover face, chest and hair")

    def execute(self, context):
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            arm, report = importer.import_pac(context, self.filepath, self.scale, self.model,
                                              self.extras, self.textures, self.blood)
        except (ValueError, OSError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        _report_lines(self, report[1:])
        for line in report[1:]:
            if not line.lower().startswith(WARNING):
                self.report({"INFO"}, line)
        self.report({"INFO"}, "%s: %s" % (arm.name, report[0]))
        for o in context.view_layer.objects:
            o.select_set(False)
        arm.select_set(True)
        context.view_layer.objects.active = arm
        return {"FINISHED"}


class EXPORT_SCENE_OT_svr_pac(bpy.types.Operator, ExportHelper):
    """Write a new ch*.pac with the meshes bound to the imported armature, built on its base PAC"""
    bl_idname = "export_scene.svr_pac"
    bl_label = "Export ch*.pac"
    bl_options = {"REGISTER", "PRESET"}

    filename_ext = ".pac"
    filter_glob: StringProperty(default="*.pac", options={"HIDDEN"})

    base_path: StringProperty(
        name="Base PAC", subtype="FILE_PATH", default="",
        description="ch*.pac the new file is built on. Empty: the one the armature was imported from")
    alpha: IntProperty(
        name="Gloss (alpha)", default=47, min=0, max=255,
        description="Alpha of the 'color' texture: the game uses it as gloss (47 on ch100's skin)")
    specular: IntProperty(
        name="Specular (normal alpha)", default=12, min=0, max=255,
        description="Alpha of the flat 'normal' texture, which the game uses as the specular mask "
                    "(game characters: ~18 on skin, ~7 on clothes). Higher = shinier")
    atlas_scale: FloatProperty(
        name="Texture scale", default=1.0, min=0.125, max=8.0,
        description="Scale of each image before packing them into the atlas (2 = double resolution)")
    emd_id: IntProperty(
        name="Character id", default=-1, min=-1, max=999999,
        description="Change the id of the EMD entry (its first 6 digits); -1 = keep the base one")
    blood: BoolProperty(
        name="Blood layers", default=True,
        description="Rebuild the base's blood layers (face, chest, hair) on the new skin so the character "
                    "can bleed in game. Off: they are emptied")
    selected_only: BoolProperty(
        name="Selected only", default=False,
        description="Export only the selected meshes (otherwise, every visible mesh bound to the armature)")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "base_path")
        try:
            arm = exporter.find_armature(context)
            if not self.base_path:
                layout.label(text="Base: %s" % bpy.path.basename(arm["svr_pac"]), icon="INFO")
        except ValueError as e:
            layout.label(text=str(e), icon="ERROR")
        layout.prop(self, "alpha")
        layout.prop(self, "specular")
        layout.prop(self, "atlas_scale")
        layout.prop(self, "emd_id")
        layout.prop(self, "blood")
        layout.prop(self, "selected_only")

    def execute(self, context):
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            log = exporter.export_pac(context, self.filepath, bpy.path.abspath(self.base_path), self.alpha,
                                      self.atlas_scale, self.emd_id, self.selected_only, self.specular,
                                      self.blood)
        except (ValueError, OSError, RuntimeError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        _report_lines(self, log)
        self.report({"INFO"}, log[-1])
        return {"FINISHED"}


def menu_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_svr_pac.bl_idname, text="SvR 2011 character (.pac)")


def menu_export(self, context):
    self.layout.operator(EXPORT_SCENE_OT_svr_pac.bl_idname, text="SvR 2011 character (.pac)")


def register():
    bpy.utils.register_class(IMPORT_SCENE_OT_svr_pac)
    bpy.utils.register_class(EXPORT_SCENE_OT_svr_pac)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.utils.unregister_class(EXPORT_SCENE_OT_svr_pac)
    bpy.utils.unregister_class(IMPORT_SCENE_OT_svr_pac)
