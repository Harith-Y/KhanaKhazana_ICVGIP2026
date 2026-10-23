"""
Controlled class-count vs images-per-class experiment (camera-ready C1).

For each (K, ipc): take the Popular-K tier manifest, keep exactly `ipc` random training images per class
(val/test are the tier's full splits, unchanged), and write
    manifests/ctrl_k{K}_ipc{ipc}/{train,val,test}.csv + classes.json
    config_ctrl_k{K}_ipc{ipc}.yaml            (DenseNet-121, same recipe as Popular-300)
Manifest rows are used as listed (NOT filtered by files present on this machine: the Ada data copy is an older
snapshot that still has classes since removed locally), and paths are rewritten to the Ada layout.
Copy manifests/ctrl_* and the configs to Ada, run check_manifest_coverage.py there, and check the `Train:`
line in each job log against the expected size printed here (FoodDataset silently drops missing files).

    python build_controlled_manifests.py
"""
import csv
import json
import os
import random
import re
from collections import defaultdict

LOCAL = "data/proto_crawled_all"
ADA = LOCAL   # paths are kept as they are in the source manifests
ADA_ROOT = "."
BASE_CFG = "configs/config_densenet121_popular_300cls.yaml"
# class-count sweep at 5 img/class (the 300/600 tiers have classes with only 6 train images),
# images-per-class sweep at K=100, and a second class-count point at 30 img/class
SPECS = [(100, 5), (200, 5), (300, 5), (600, 5), (100, 10), (100, 20), (100, 30), (200, 30),
         (200, 10), (200, 20)]   # last two added to fill the K=200 row


def read(path):
    return [(r["path"], r["label"]) for r in csv.DictReader(open(path))]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows((p.replace(LOCAL, ADA), l) for p, l in rows)


for K, ipc in SPECS:
    src = f"manifests/popular_{K}cls"
    name = f"ctrl_k{K}_ipc{ipc}"
    out = f"manifests/{name}"
    os.makedirs(out, exist_ok=True)

    by_cls = defaultdict(list)
    for p, l in read(f"{src}/train.csv"):
        by_cls[l].append((p, l))
    n_classes = len(json.load(open(f"{src}/classes.json")))
    short = [c for c in range(n_classes) if len(by_cls[str(c)]) < ipc]
    if short:
        raise SystemExit(f"{name}: {len(short)} classes have < {ipc} train images in the manifest")
    rng = random.Random(0)
    train = [r for c in range(n_classes) for r in rng.sample(by_cls[str(c)], ipc)]
    val, test = read(f"{src}/val.csv"), read(f"{src}/test.csv")

    write(f"{out}/train.csv", train)
    write(f"{out}/val.csv", val)
    write(f"{out}/test.csv", test)
    open(f"{out}/classes.json", "w").write(open(f"{src}/classes.json").read())

    cfg = open(BASE_CFG).read()
    cfg = re.sub(r"manifest_dir:\s*\S+", f"manifest_dir:  {ADA_ROOT}/manifests/{name}", cfg)
    cfg = re.sub(r"run_dir:\s*\S+", f"run_dir: {ADA_ROOT}/runs", cfg)
    cfg = cfg.replace("num_workers:      0", "num_workers:      4")
    open(f"configs/config_{name}.yaml", "w").write(cfg)
    print(f"{name:20s} classes {n_classes:4d}  expected  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
