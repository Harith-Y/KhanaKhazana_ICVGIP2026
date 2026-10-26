# KhanaKhazana: benchmark code and derived artifacts

Code and derived artifacts for **"KhanaKhazana: A Large-Scale Indian Food Dataset and Fine-Grained Classification Benchmark"**
(Harith Yerragolam, C. V. Jawahar; ICVGIP 2026, CVIT, IIIT Hyderabad).

**Project page:** https://harith-y.github.io/KhanaKhazana_ICVGIP2026/

## What is and is not released

| | Status |
|---|---|
| Benchmark code (training, evaluation, probes, controlled experiments, zero-shot VLM scripts) | Released here |
| Class list of every evaluated tier, head / mid / tail frequency groups, merge candidates | Released here (`release/`) |
| **Images** | **Not redistributed**: most source images come from restaurant sites and web image search, and their licensing is not settled |
| Image corpus for research use | **Available to researchers on request** (see below) |

### Requesting the corpus
Write to harith.yerragolam@research.iiit.ac.in (cc jawahar@iiit.ac.in) with your name, affiliation and intended research use.

## Repository layout

| Path | Contents |
|---|---|
| `Supplementary.pdf` | Supplementary material of the paper (protocol, probes, controlled runs, training-recipe note, tier statistics) |
| `train.py`, `evaluate.py` | Training (with `--seed`, W&B logging) and test-set evaluation (`--per_class`) |
| `models/classifier.py` | Backbones: ResNet, EfficientNet, ConvNeXt, MobileNet, DenseNet, ViT, and frozen DINOv3 / Perception Encoder linear probes |
| `configs/` | One YAML per reported experiment (relative paths) |
| `make_current_tier.py`, `build_popular_manifests.py`, `prepare_data_by_class_list.py` | Build tier manifests from a class list and your own copy of the corpus |
| `build_controlled_manifests.py` | Class-count versus images-per-class subsets |
| `check_manifest_coverage.py` | Reports how many manifest rows resolve to a file; the dataset loader silently skips missing files, so run this before training |
| `head_mid_tail.py` | Accuracy by training-image frequency group |
| `evaluate_*.py` | Zero-shot VLM evaluation (CLIP, GPT-4o, Gemini, Qwen2-VL, LLaVA, InternVL, Llama-3.2-Vision); API keys are read from environment variables |
| `docs/` | Project page, served by GitHub Pages from `master` / `docs` |
| `release/` | `merge_candidates.csv` (24 class pairs with at least 35% confusion), `class_frequency_groups.csv` (per-class image count and head / mid / tail group; the folder of unidentified images is excluded), `per_class_accuracy_popular1700.csv` (per-class accuracy and training-image count of the 1,700-class model), `tiers/` (class list of each tier in Table 7, with provenance in `tiers/README.md`) |
| `data/dish_popularity.json` | Popularity scores used to rank the first ~600 classes of each tier (the rest are ranked by image count) |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.11
pip install -r requirements.txt
```

## Reproducing the experiments (requires the image corpus)
Place the corpus as `data/proto_crawled_all/<class>/<image>.jpg`.

```bash
# 1. manifest for a tier, from its released class list
python prepare_data_by_class_list.py --classes_json release/tiers/popular_300cls.json --manifest_dir manifests/popular_300cls
python check_manifest_coverage.py manifests/popular_300cls          # expect 100%
# note: 7 of the 300 classes of this tier were built from an earlier corpus snapshot and no longer have images;
# the script prints a warning and builds the tier without them (see release/tiers/README.md)

# 2. fine-tune a backbone, then score the test split
python train.py --config configs/config_densenet121_popular_300cls.yaml --seed 0
python evaluate.py --config configs/config_densenet121_popular_300cls.yaml --checkpoint runs/<run>/best.pth --per_class

# 3. frozen DINOv3 / Perception Encoder linear probe
python train.py --config configs/config_dinov3_b16_probe_popular_300cls.yaml

# 4. controlled class-count vs images-per-class runs
python build_controlled_manifests.py
python train.py --config configs/config_ctrl_k100_ipc5.yaml --seed 0
```
Training logs to Weights & Biases by default; pass `--no_wandb` to disable.

## Citation
The citation will be added when the proceedings are published.

## License
To be added before the repository is made public.
