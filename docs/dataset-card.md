---
license: cc-by-4.0
task_categories:
  - image-to-3d
  - text-to-3d
tags:
  - 3d
  - multi-view
  - objaverse
  - gobjaverse
  - rendering
size_categories:
  - 10K<n<100K
viewer: false
---

<div align="center">

# <img src="https://raw.githubusercontent.com/zeyuanyin/Dome-Objaverse/main/web/assets/logo.svg" width="34" height="34" valign="middle" alt="Dome-Objaverse Logo" /> Dome-Objaverse

### Multi-view renders of 83,296 Objaverse objects — 48 views each, on a camera dome of 4 elevations &times; 12 azimuths, with the rendered pixels published.

<p align="center">
  <a href="https://huggingface.co/datasets/zeyuanyin/Dome-Objaverse"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E?style=for-the-badge&logoColor=black" alt="Hugging Face Dataset"></a>
  &nbsp;&nbsp;
  <a href="https://zeyuanyin.github.io/Dome-Objaverse/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Viewer%20Website-Online%20Demo-2f6df6?style=for-the-badge" alt="Viewer Website"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/zeyuanyin/Dome-Objaverse"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github" alt="GitHub Code"></a>
</p>

[![Stepping through all 48 views beside the camera that took each one](https://raw.githubusercontent.com/zeyuanyin/Dome-Objaverse/main/web/assets/spin.gif)](https://zeyuanyin.github.io/Dome-Objaverse/#object=0/10228&split=dome_objaverse)

*The 48 views in order: three azimuth rings at elevations 0°, 30° and 60°, plus a top-down ring at 90°. The highlighted camera on the dome is the one that took the image on the left.*

</div>

---

Rendering is the computationally demanding bottleneck of multi-view 3D datasets — typically over **100,000 CPU core hours** for 83k objects. Our rendered pixels already exist: 512×512 RGBA plus per-view normal and depth maps, ~1.5 TB for the primary split and freely downloadable, each object paired with its Objaverse UID and a Cap3D caption. *Dome* is literal for the primary `dome_objaverse` split: all cameras sit on the upper hemisphere (elevations 0° to 90°). The accompanying `gobjaverse_parquet` split preserves GObjaverse's original trajectory, which dips to elevation −90° (see "Splits & Metadata" below).

- **License**: CC-BY-4.0, ~1.5 TB for the primary split, parquet-packaged.
- **Viewer**: [zeyuanyin.github.io/Dome-Objaverse](https://zeyuanyin.github.io/Dome-Objaverse/) — search all 83,296 captions, browse every view, and see interactive 3D fused point clouds in-browser before downloading anything.
- **Code**: loading/decoding scripts, the rendering pipeline, and full schema docs live at [github.com/zeyuanyin/Dome-Objaverse](https://github.com/zeyuanyin/Dome-Objaverse).

## Comparison with other Objaverse render sets

| Dataset | Objects | Views / object | Elevation coverage | Rendered pixels published |
|---|---|---|---|---|
| [Zero123++ v1.2](https://github.com/SUDO-AI-3D/zero123plus) | — | 6 | two values (`20°` / `-10°`) | n/a *(model output config)* |
| [MVDream](https://arxiv.org/pdf/2308.16512) | — | 4 | single fixed elevation | n/a *(model output config)* |
| [SV3D](https://arxiv.org/pdf/2403.12008) | 150K | 21 | static `[-5°, 30°]` or dynamic | — |
| [TRELLIS-500K](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md) | 500K | 150 | dense | **No** *(metadata + render script only)* |
| [G-buffer Objaverse](https://aigc3d.github.io/gobjaverse/) | 280K | 40 | side rings `≤30°` (varies/object) + top/bottom | **Yes** *(~1.1 TB, original source renders)* |
| **[Dome-Objaverse](https://huggingface.co/datasets/zeyuanyin/Dome-Objaverse)** | **83,296** | **48** | **four rings: `0°`, `30°`, `60°`, `90°`** | **Yes** (~1.5 TB primary split, free download) |

Two things distinguish Dome-Objaverse: a **regular, fully-specified camera grid** (four rings of identical 12-step azimuth sweeps, every view addressable by `view_id`), and the fact that the **pixels are actually published** rather than left to the user to render.

## Splits & Metadata

Both splits cover the exact same **83,296 objects** with zero missing objects.

| Split | Description & Source | Views | Elevation | Azimuth |
|---|---|---|---|---|
| **`dome_objaverse`** *(Primary)* | Our custom dome render pass (1.4 TB) | **48** | `0°, 30°, 60°, 90°` (4 rings) | `0°, 30°, ..., 330°` (12 steps/ring) |
| **`gobjaverse_parquet`** | Re-packed from original [G-buffer Objaverse](https://aigc3d.github.io/gobjaverse/) renders for baseline supervision (1.1 TB) | **40** | Side rings `≤30°` (varies/object) + `±90°` poles | Mixed `15°` / `30°` steps |

- **`dome_objaverse`**: `view_id` 0–11 / 12–23 / 24–35 / 36–47 are elevation `0°` / `30°` / `60°` / `90°` respectively, each a 12-step azimuth sweep `0°, 30°, ..., 330°`.
- **`gobjaverse_parquet`**: camera parameters are read dynamically per-view from the embedded `00000.json` ... `00039.json` metadata files.

`metadata/objects.parquet` (5 MB, ships in this repo) is the join table, 83,296 rows with 100% coverage. The columns you'll actually use: `object_id` (e.g. `"0/10228"`), `objaverse_uid`, `glb_path`, `caption`, `camera_distance`. It also carries `group_id`/`index_id` (the two halves of `object_id`) and `sheet_id`/`sheet_pos` (internal thumbnail-sheet coordinates used only by the web viewer) — safe to ignore for training. Legacy per-field JSON files (`camera_distances.json`, `gobjaverse_index_to_objaverse.json`, `text_captions_cap3d.json`, `cobj_done_list.json`) are also included; their content is already consolidated into `objects.parquet`.

## Rendering

Full config in `pipeline/2-render/blender_script.py`; a decode example is in `pipeline/4-usage/load_views.py`. Two things that will silently produce wrong results if assumed otherwise:

- ⚠️ **Camera distance is per object**, `1.50`–`2.00` (see `camera_distance` in `objects.parquet`), not a fixed constant.
- ⚠️ **`nd_png` (16-bit normal + depth) channel order is reversed**: R = normal *z*, G = normal *y*, B = normal *x*, alpha = depth (`alpha / 65535 × 5.0`, planar along the camera axis). Use `cv2.imdecode(..., cv2.IMREAD_UNCHANGED)`, not PIL — it silently downcasts to 8-bit.

## License

Code in this repository is Apache-2.0 (see `LICENSE`), matching [AI2](https://allenai.org/)'s [objaverse-rendering](https://github.com/allenai/objaverse-rendering),
which `render/` derives from. The rendered image data on Hugging Face is CC-BY-4.0. Individual
Objaverse source meshes carry their own licenses — use `objaverse_uid` in
`metadata/objects.parquet` to look them up.

## Attribution

- Source meshes: [Objaverse](https://huggingface.co/datasets/allenai/objaverse) ([Allen Institute for AI / AI2](https://allenai.org/)).
- Original camera convention and the 280k render set: [GObjaverse](https://aigc3d.github.io/gobjaverse/) / [RichDreamer](https://github.com/xianggang/richdreamer).
- The curated 83k-object subset comes from [ashawkey/objaverse_filter](https://github.com/ashawkey/objaverse_filter).
- Captions: [Cap3D](https://huggingface.co/datasets/tiange/Cap3D).
- Rendering scripts derive from AI2's [objaverse-rendering](https://github.com/allenai/objaverse-rendering) (Apache-2.0; see `pipeline/2-render/LICENSE`).

Please also cite the original [Objaverse](https://huggingface.co/datasets/allenai/objaverse) / [GObjaverse](https://aigc3d.github.io/gobjaverse/) sources and state which split(s) you used.

## Citation

This dataset is an extension of our NeurIPS 2025 publication,
[TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](https://arxiv.org/abs/2511.16642).
If you find it helpful, please consider citing:

```bibtex
@inproceedings{
    Yin2025TRIM,
    title={{TRIM}: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming},
    author={Yin, Zeyuan and Liu, Xiaoming},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
    year={2025}
}
```
