# SvR 2011 PAC — Blender add-on

This addon was made using claude code.

A tool allows import files from **WWE SmackDown vs. Raw 2011 (PC)** `ch*.pac` character files directly into Blender
(armature, meshes, weights, UVs and the `color` and `normal` textures) and **exports** a custom model weighted to that
skeleton into a new `ch*.pac`. 

## Installation

1. In Blender: *Edit > Preferences > Get Extensions*. In the drop-down menu at the top right (⌄),
   *Install from Disk…*.
2. Pick `svr_pac-x.x.x.zip`.
3. It is enabled right away. If not, look for **SvR 2011 PAC** under *Add-ons* and tick it.

## Import

*File > Import > SvR 2011 character (.pac)*

| Option | Default | What it does |
|---|---|---|
| Scale | 0.1 | Game units → meters (the character is ~1.8 m tall) |
| Outfit | 0 | Some PACs contain several outfits (2–4 PACH); 0 is the first one |
| Hair, eyelashes and extras | on | Also imports the other JBOYs (hair `0x2710`, eyelashes `0x4E20`, accessories) |
| Textures | on | Material with the `color` texture (its alpha is the game's gloss: only connected on hair and other alpha-blended meshes) and the `normal` map in a *Normal Map* node |
| Blood layers | off | Blood meshes `0x7530`–`0x7532`; they cover face, chest and hair in white |

Result: a `chNNN` collection with `chNNN_armature` and one mesh per JBOY record
(`chNNN_body_03`, `chNNN_hair_00`…). Meshes store `svr_child`, `svr_mesh`, `svr_shader` and
`svr_material_group` as custom properties; bones store `svr_index`, `svr_child`, `svr_t`, `svr_r` and
`svr_rot`. With a long-hair base, the whole `kami_b*` chains are imported and each chain's root bone
stores its physics as `svr_hair_*` (see *Long hair*).

## Export

*File > Export > SvR 2011 character (.pac)*

The model must be bound (*Armature* modifier) to the armature imported with this add-on, with vertex
groups named after its bones, and fitted to its rest pose. Every material needs its image in an
*Image Texture* node; a normal map goes in another *Image Texture* plugged into a *Normal Map* node
(tangent space). In *Face Orientation* the model must be **blue** (normals pointing outwards).
The exporter:

- replaces the body of the base PAC with your meshes (split into pieces of ≤ 20 bones);
- packs every image into an atlas that becomes the `color` texture (DXT5, alpha = gloss; on hair, the image's own alpha = transparency);
- packs the normal maps the same way into the `normal` texture (flat where a material has none), converted to the game's convention;
- sends meshes weighted to hair chains (`kami_b*`) to the hair JBOY (see *Long hair*);
- leaves the base's teeth, eyes, hair, eyelashes and accessories with nothing visible, unless meshes go to them;
- rebuilds the blood layers (`0x7530` face, `0x7531` chest, `0x7532` hair) on the new skin: each skin triangle near a base layer is copied 0.002 units outwards with that layer's UVs (front projection for face/chest, closest point for hair), so the game can draw blood on it;
- writes normals and triangle winding pointing inwards, as the game expects;
- does not touch the body skeleton, the animations or the rest of the PAC.

| Option | Default | What it does |
|---|---|---|
| Base PAC | (the imported one) | Empty = the PAC the armature came from |
| Gloss (alpha) | 47 | Alpha of `color` (gloss; game characters: ~48 skin, ~12 clothes) |
| Specular (normal alpha) | 12 | Alpha of the `normal` texture = specular mask (game: ~18 skin, ~7 clothes); the main cause of a plastic look |
| DirectX normal maps | off | The normal maps use the DirectX convention (Unreal, 3ds Max…): their green is inverted |
| Texture scale | 1.0 | Scale of each image before the atlas |
| Character id | −1 | Changes the id of the EMD entry; −1 = keep the base one |
| Blood layers | on | Rebuild the blood layers on the new skin (off: they are emptied, no blood in game) |
| Selected only | off | Export only the selected meshes |


## Normal maps

The game's `normal` is a tangent-space normal map whose alpha is the specular mask. Its red **and** green
are inverted with respect to Blender's (OpenGL) convention; the add-on converts both ways, so in Blender
the maps work as usual. The vertices carry no tangents (the game's shader derives them), so nothing else
has to be exported. Maps made for DirectX (Unreal assets such as `T_..._N.png`) need *DirectX normal maps*.

## Long hair

Long hair moves with physics in the game: it lives in the hair JBOY (`0x2710`) with its own bone chains
`kami_bNN` → `kami_bNN_00` → … hanging from `atama`, and child `0x3c` gives each chain its physics.

1. Import a base with long hair (`ch293`, `ch109`, `ch175`, `ch112`…; 42 of the 93 base PACs).
2. Fit your hair to the `kami_b*` chains and weight it to them (and to `atama` near the scalp). Any mesh
   weighted to a `kami_b*` bone goes to the hair JBOY; its material decides the template: *Alpha*
   connected = alpha-blended strands, otherwise opaque.
3. The chains can be moved in Edit Mode, and new ones added by duplicating a chain (Shift+D) and
   renaming it `kami_bNN`, `kami_bNN_00`… (15 characters at most). Bones are never removed from the PAC.
4. Physics: the root bone of each chain has the `svr_hair_*` custom properties (Bone Properties >
   Custom Properties): radius, viscosity, gravity, spring, damping, attach, length_fix, stiffness and
   move_fix. A duplicated chain copies them.

Tested in Blender on ch293, ch109, ch175 and ch112 (`tests/test_hair.py`); in-game tests pending.