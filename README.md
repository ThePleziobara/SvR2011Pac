# SvR 2011 PAC — Blender add-on

This addon was made using claude code.

A tool allows import files from **WWE SmackDown vs. Raw 2011 (PC)** `ch*.pac` character files directly into Blender
(armature, meshes, weights, UVs and the `color` texture) and **exports** a custom model weighted to that
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


## Important

Models exported using svr pac 2011 addon do not rig its hair so do not expect physhics on them for now.