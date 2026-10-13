"""
Imports every given PAC (and all its outfits) in Blender and reports errors.

    blender -b --factory-startup --python tests/test_all_pacs.py -- ../PACS/Base/*.pac
"""
import os
import sys
import traceback

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import svr_pac  # noqa: E402
from svr_pac import importer  # noqa: E402
from svr_pac.pac_container import Pac  # noqa: E402

paths = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
svr_pac.register()
fails = 0
for path in paths:
    for model in range(len(Pac.open(path).models)):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            _, report = importer.import_pac(bpy.context, os.path.abspath(path), model=model, blood=True)
            print("%-10s outfit %d: %s" % (os.path.basename(path), model, " | ".join(report)))
        except Exception:  # noqa: BLE001
            fails += 1
            print("%-10s outfit %d: ERROR" % (os.path.basename(path), model))
            traceback.print_exc()
print("\n%d errors" % fails)
sys.exit(1 if fails else 0)
