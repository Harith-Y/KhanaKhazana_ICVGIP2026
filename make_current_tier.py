"""
Build a Popular-N manifest from the CURRENT corpus on this machine (camera-ready Table 7 repair).

Same ranking as build_popular_manifests.py (popularity score first, then image count) and the same split code
as prepare_data_by_class_list.py, but computed from the folders that exist here, and written to a NEW
folder, so no existing manifest is overwritten.

    python make_current_tier.py --proto /path/to/proto_crawled_all --size 1700
    -> manifests/popular_1700cls_current/{train,val,test}.csv + classes.json + stats.txt
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from build_popular_manifests import GOOD_EXT, POP_JSON, build_ranked_list

HERE = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("--proto", required=True, help="dataset root (class folders)")
ap.add_argument("--size", type=int, required=True)
ap.add_argument("--manifests_dir", default=str(HERE / "manifests"))
a = ap.parse_args()

proto = Path(a.proto)
counts = {d.name: sum(1 for f in d.iterdir() if f.suffix.lower() in GOOD_EXT)
          for d in proto.iterdir() if d.is_dir() and d.name != "unknown" and not d.name.startswith("_")}
ranked = build_ranked_list(json.loads(POP_JSON.read_text()), counts)
print(f"{len(counts)} class folders ({sum(1 for v in counts.values() if v == 0)} empty); ranked {len(ranked)}")
sel = [c for c in ranked if counts[c] > 0][: a.size]
if len(sel) < a.size:
    sys.exit(f"only {len(sel)} non-empty classes, need {a.size}")

name = f"popular_{a.size}cls_current"
mdir = Path(a.manifests_dir) / name
lst = Path(a.manifests_dir) / "_current_lists" / f"{name}.json"
lst.parent.mkdir(parents=True, exist_ok=True)
lst.write_text(json.dumps(sel, indent=2))
print(f"class list -> {lst}  (last class: {sel[-1]}, {counts[sel[-1]]} images)")

subprocess.run([sys.executable, str(HERE / "prepare_data_by_class_list.py"),
                "--classes_json", str(lst), "--manifest_dir", str(mdir), "--proto", str(proto)], check=True)

# sanity: classes with no training image cannot be learned
import csv
tr = {}
for r in csv.DictReader(open(mdir / "train.csv")):
    tr[r["label"]] = tr.get(r["label"], 0) + 1
n = len(json.loads((mdir / "classes.json").read_text()))
print(f"classes with 0 train images: {n - len(tr)} of {n}")
