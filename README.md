# SvR 2011 PAC — Blender add-on

Imports **WWE SmackDown vs. Raw 2011 (PC)** `ch*.pac` character files directly into Blender
(armature, meshes, weights, UVs and the `color` texture) and **exports** a custom model weighted to that
skeleton into a new `ch*.pac`. No intermediate FBX.

Author: **pleziobara** · Version: **0.3.1** · Blender **4.2 or later** (tested on 5.2.2)

Status: importer and "replace model" exporter. A model exported with it (CJ from GTA SA) has been
tested in game: character select, entrance and ring, with correct lighting.

Full documentation (architecture, flows, diagrams, formats and maintenance), in Spanish:
`DOCS/Addon SvR PAC/`.

## Installation

1. In Blender: *Edit > Preferences > Get Extensions*. In the drop-down menu at the top right (⌄),
   *Install from Disk…*.
2. Pick `dist/svr_pac-0.3.1.zip`.
3. It is enabled right away. If not, look for **SvR 2011 PAC** under *Add-ons* and tick it.

Nothing else is needed: it only uses `numpy`, which ships with Blender (Pillow is not required).
Since 0.2.1 updating does not require restarting Blender (coming from 0.1.0/0.2.0, restart once).

## Import

*File > Import > SvR 2011 character (.pac)*

| Option | Default | What it does |
|---|---|---|
| Scale | 0.1 | Game units → meters (the character is ~1.8 m tall) |
| Outfit | 0 | Some PACs contain several outfits (2–4 PACH); 0 is the first one |
| Hair, eyelashes and extras | on | Also imports the other JBOYs (hair `0x2710`, eyelashes `0x4E20`, accessories) |
| Textures | on | Material with the `color` texture (its alpha is the game's gloss: not connected) |
| Blood layers | off | Blood meshes `0x7530`–`0x7532`; they cover face, chest and hair in white |

Result: a `chNNN` collection with `chNNN_armature` and one mesh per JBOY record
(`chNNN_body_03`, `chNNN_hair_00`…). Meshes store `svr_child`, `svr_mesh`, `svr_shader` and
`svr_material_group` as custom properties; bones store `svr_index`, `svr_t` and `svr_r`.

## Export

*File > Export > SvR 2011 character (.pac)*

The model must be bound (*Armature* modifier) to the armature imported with this add-on, with vertex
groups named after its bones, and fitted to its rest pose. Every material needs its image in an
*Image Texture* node. In *Face Orientation* the model must be **blue** (normals pointing outwards).
The exporter:

- replaces the body of the base PAC with your meshes (split into pieces of ≤ 20 bones);
- packs every image into an atlas that becomes the `color` texture (DXT5, alpha = gloss);
- leaves the base's teeth, eyes, hair, eyelashes and accessories with nothing visible;
- rebuilds the blood layers (`0x7530` face, `0x7531` chest, `0x7532` hair) on the new skin: each skin triangle near a base layer is copied 0.002 units outwards with that layer's UVs (front projection for face/chest, closest point for hair), so the game can draw blood on it;
- writes normals and triangle winding pointing inwards, as the game expects;
- does not touch the skeleton, the animations or the rest of the PAC.

| Option | Default | What it does |
|---|---|---|
| Base PAC | (the imported one) | Empty = the PAC the armature came from |
| Gloss (alpha) | 47 | Alpha of `color` (gloss; game characters: ~48 skin, ~12 clothes) |
| Specular (normal alpha) | 12 | Alpha of the flat `normal` texture = specular mask (game: ~18 skin, ~7 clothes); the main cause of a plastic look |
| Texture scale | 1.0 | Scale of each image before the atlas |
| Character id | −1 | Changes the id of the EMD entry; −1 = keep the base one |
| Blood layers | on | Rebuild the blood layers on the new skin (off: they are emptied, no blood in game) |
| Selected only | off | Export only the selected meshes |

## Folder contents

```
addon_svr_pac/
├── svr_pac/                 # the add-on (this is what goes inside the zip)
│   ├── blender_manifest.toml
│   ├── __init__.py          # registration (bpy is not imported until registering; reloads submodules)
│   ├── operators.py         # operators and menu entries (import and export)
│   ├── importer.py          # Blender scene (armature, meshes, materials)
│   ├── exporter.py          # reads the scene for export (rest pose, coordinates, weights, images)
│   ├── pac_builder.py       # builds the new PAC (atlas, chunks, templates, emptying)
│   ├── pac_container.py     # EPK8 / EMD / PACH (read and write, several PACHs)
│   ├── bpe.py               # BPE compression (read and write)
│   ├── jboy.py              # JBOY (read and rebuild, POF0, several strips, triangles <-> strips)
│   └── dds.py               # texture packs; read DXT1/3/5 and uncompressed; write DXT5
├── dist/svr_pac-0.3.1.zip   # the file to install
├── build.sh                 # rebuilds the zip (Blender validates the manifest and packs it)
└── tests/
    ├── test_formats.py      # outside Blender: container, JBOY, BPE and DDS round trips
    ├── test_blender.py      # in Blender: compares with pac2fbx.py's FBX (rest and posed)
    ├── test_all_pacs.py     # in Blender: imports many PACs in a row and reports errors
    └── test_exporter.py     # in Blender: exports, re-imports and compares (vertices, weights, pose)
```

The format modules (`pac_container`, `bpe`, `jboy`, `dds`, `pac_builder`) do not depend on `bpy`.

## Tests

```sh
# outside Blender (Python 3 + numpy; Pillow optional to compare textures)
python3 tests/test_formats.py ../ch100.pac ../PACS/Base/*.pac        # --no-textures: faster

# in Blender
python3 ../pac2fbx.py ../ch100.pac -o /tmp/ch100.fbx
blender -b --factory-startup --python tests/test_blender.py -- ../ch100.pac /tmp/ch100.fbx [render.png] [--blood]
blender -b --factory-startup --python tests/test_all_pacs.py -- ../PACS/Base/*.pac
blender -b --factory-startup ../CJ/escena/CJ_svr.blend --python tests/test_exporter.py -- /tmp/cj.pac [render.png]
```

Results (94 PACs: `ch100.pac` and the 93 in `PACS/Base`):

- Container: all 94 repack byte for byte (including those with 2–4 outfits).
- JBOY: 531 of 567 byte for byte; the other 36 have the same content and only differ in padding.
- DDS: the 3044 DXT1/DXT5 textures match Pillow (max difference 1, rounding).
- Import: ch100 matches pac2fbx.py's FBX (bones and vertices, at rest and posed); all 132 outfits
  of the 94 PACs import without errors.
- Export (CJ): round trip with identical triangles, vertices, weights, normals and pose; the PAC has the
  game's orientation.

## Format facts found while building it

Missing from the previous documentation (and `jboy_rebuild.py` / `pac2fbx.py` / `pac_replace_model.py`
get them wrong):

- **The alpha of the `normal` texture is the specular mask** (~18 on skin, ~7 on clothes). Writing a flat
  normal map with alpha 255 made CJ look like plastic.
- **Normals and winding point inwards** in every game PAC (negative signed volume). Exporting them
  outwards caused CJ's black face; the add-on flips them on import and on export.
- **Several PACHs per file** (outfits). The EMD group has one 16-byte entry per PACH:
  `name[8]` (`00010202`, `00010212`…), `u32` offset in units of 0x800 from 0x4000 and
  `u32` size in units of 0x100. The EPK8 header `+4` is `12 + 16n`.
- **Several strips per mesh.** Record `+4` is the strip count. Block D is
  `n × (u32 6, u32 index_count, u32 ptr)` followed by each strip's indices, each padded to 4.
- **The body skeleton does not always have 102 bones** (94 to 139).
- **Parents do not always come before children** (ch288 eyelashes).
- **Extra JBOYs carry their own bones**: long hair brings `kami_b*`, hanging from `atama`.
- **An extra JBOY's bind pose may differ from the body's** (ch125: eyelashes and face blood ~9 units
  lower). The add-on moves those vertices to the body's pose with `W_body · W_own⁻¹` per bone.
- `0x7530`–`0x7532` are **blood layers** (textures `BloodFace`, `BloodBody`, `BloodHair`).
