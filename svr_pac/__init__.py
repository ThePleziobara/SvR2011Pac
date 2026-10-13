"""
SvR 2011 PAC - import and export WWE SmackDown vs. Raw 2011 (PC) ch*.pac files in Blender.

File > Import / Export > SvR 2011 character (.pac)

The format modules (pac_container, bpe, jboy, dds, pac_builder) do not depend on bpy and can be
used outside Blender; that is why bpy is not imported here until the add-on is registered.
"""
import importlib
import sys

# in dependency order: the ones below use the ones above
_MODULES = ("bpe", "pac_container", "jboy", "dds", "pac_builder", "importer", "exporter", "operators")


def _reload_modules():
    """When the extension is updated without restarting Blender, Blender reloads this __init__ but
    reuses the submodules it already imported (e.g. an old operators.py without the exporter).
    They are reloaded here so the installed code is always the one in use."""
    for name in _MODULES:
        mod = sys.modules.get("%s.%s" % (__name__, name))
        if mod is not None:
            importlib.reload(mod)


def register():
    _reload_modules()
    from . import operators
    operators.register()


def unregister():
    from . import operators
    operators.unregister()
