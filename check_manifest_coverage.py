"""
Report how many manifest rows actually resolve to a file on THIS machine.
FoodDataset silently drops missing rows, so a stale/partial manifest trains on a smaller set with no warning.

Usage (run on the machine that trains):
    python check_manifest_coverage.py manifests/popular_600cls manifests/popular_1700cls ...
"""
import csv
import os
import sys

for m in sys.argv[1:]:
    parts = []
    for sp in ("train", "val", "test"):
        rows = list(csv.DictReader(open(os.path.join(m, f"{sp}.csv"))))
        ok = sum(os.path.exists(r["path"]) for r in rows)
        parts.append(f"{sp} {ok}/{len(rows)} ({100 * ok / len(rows):.1f}%)")
    print(f"{os.path.basename(m.rstrip('/')):<20} " + "   ".join(parts))
