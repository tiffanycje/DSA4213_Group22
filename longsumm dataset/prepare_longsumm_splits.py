#!/usr/bin/env python3
"""
prepare_longsumm_splits.py

Stratified (by summary-length folders) 72/9/9/10 splitting for LongSumm, plus
PDF download for the 10% held-out end-to-end pipeline set.

- Walks LongSumm/abstractive_summaries/by_clusters/*/*.json
- Uses the official README schema (id, summary: [sentences], pdf_url, ...).
- Produces JSONL files: train/val/test/heldout_pipeline
- Downloads PDFs for held-out (arXiv/ACL/any direct pdf_url) into out_dir/heldout_pdfs
- Deterministic (seeded) and uses Hamilton/Largest-Remainder rounding to ensure
  exact global totals while preserving per-bin proportions.

Usage:
    python prepare_longsumm_splits.py \
        --longsumm_root ./LongSumm \
        --out_dir ./longsumm_prepared \
        --seed 33
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longsumm_root", type=str, default="./LongSumm",
                    help="Path to your cloned LongSumm repo root")
    ap.add_argument("--out_dir", type=str, default="./longsumm_prepared",
                    help="Where to write splits and PDFs")
    ap.add_argument("--seed", type=int, default=33, help="Random seed")
    ap.add_argument("--timeout", type=int, default=60, help="Download timeout (s)")
    ap.add_argument("--user_agent", type=str, default="Mozilla/5.0",
                    help="HTTP User-Agent for downloads")
    return ap.parse_args()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read {p}: {e}")
        return None

def join_sentences(x):
    if isinstance(x, list):
        return " ".join([s.strip() for s in x if isinstance(s, str)]).strip()
    if isinstance(x, str):
        return x.strip()
    return ""

@dataclass
class Rec:
    id: str
    bin_name: str
    path: str
    summary_sentences: List[str]
    summary_text: str
    pdf_url: Optional[str]

def normalize(js: dict, bin_name: str, path: Path) -> Optional[Rec]:
    # Per README for abstractive summaries:
    #  id, blog_id, summary (array of sentences), author_id, pdf_url, ...
    rid = str(js.get("id") or js.get("paper_id") or js.get("uid") or "")
    summary_list = js.get("summary") or js.get("abstractive_summary") or js.get("long_summary")
    if isinstance(summary_list, str):
        # if provided as one long string, split sentences conservatively
        summary_list = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_list) if s.strip()]
    if not summary_list:
        return None
    pdf_url = js.get("pdf_url") or js.get("url") or js.get("paper_url") or js.get("source")
    return Rec(
        id=rid if rid else path.stem,
        bin_name=bin_name,
        path=str(path),
        summary_sentences=[s for s in summary_list if isinstance(s, str) and s.strip()],
        summary_text=join_sentences(summary_list),
        pdf_url=pdf_url if isinstance(pdf_url, str) else None,
    )

def list_bins(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])

def arxiv_abs_to_pdf(url: str) -> Optional[str]:
    try:
        m = re.search(r"arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+)(v[0-9]+)?", url)
        if m:
            ident = m.group(2)
            return f"https://arxiv.org/pdf/{ident}.pdf"
    except:
        pass
    return None

def acl_to_pdf(url: str) -> Optional[str]:
    try:
        m = re.search(r"aclanthology\.org/([A-Za-z0-9\.\-]+)/?", url)
        if m:
            code = m.group(1)
            if code.endswith(".pdf"):
                return f"https://aclanthology.org/{code}"
            return f"https://aclanthology.org/{code}.pdf"
    except:
        pass
    return None

def guess_pdf_url(source_url: Optional[str]) -> Optional[str]:
    if not source_url:
        return None
    if "arxiv.org" in source_url:
        return arxiv_abs_to_pdf(source_url) or source_url
    if "aclanthology.org" in source_url:
        return acl_to_pdf(source_url) or source_url
    if source_url.lower().endswith(".pdf"):
        return source_url
    return None

def http_download(url: str, dest: Path, timeout: int, user_agent: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        ensure_dir(dest.parent)
        with dest.open("wb") as f:
            f.write(data)
        return True
    except Exception as e:
        logging.info(f"Failed to download: {url} -> {dest} ({e})")
        return False

def largest_remainder_alloc(total_target: int, quotas: List[float]) -> List[int]:
    floors = [math.floor(q) for q in quotas]
    remainders = [q - f for q, f in zip(quotas, floors)]
    short = total_target - sum(floors)
    order = sorted(range(len(quotas)), key=lambda i: (-remainders[i], -quotas[i], i))
    alloc = floors[:]
    for i in order[:max(0, short)]:
        alloc[i] += 1
    return alloc

def main():
    args = parse_args()
    random.seed(args.seed)

    longsumm_root = Path(args.longsumm_root).resolve()
    abstr_dir = longsumm_root / "abstractive_summaries" / "by_clusters"
    out_dir = Path(args.out_dir).resolve()
    splits_dir = out_dir / "splits"
    pdf_dir = out_dir / "heldout_pdfs"
    ensure_dir(out_dir); ensure_dir(splits_dir); ensure_dir(pdf_dir)

    logging.basicConfig(
        filename=str(out_dir / "prepare.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not abstr_dir.exists():
        print(f"[ERROR] Not found: {abstr_dir}")
        sys.exit(1)

    # 1) Load all records, grouped per bin
    bin_to_recs: Dict[str, List[Rec]] = {}
    for b in list_bins(abstr_dir):
        recs = []
        for jf in sorted(b.glob("*.json")):
            js = read_json(jf)
            if not js:
                continue
            r = normalize(js, b.name, jf)
            if r:
                recs.append(r)
        if recs:
            bin_to_recs[b.name] = recs

    # Shuffle within each bin (deterministic)
    for k in bin_to_recs:
        random.shuffle(bin_to_recs[k])

    all_bins = sorted(bin_to_recs.keys())
    n_per_bin = {k: len(bin_to_recs[k]) for k in all_bins}
    N = sum(n_per_bin.values())
    if N == 0:
        print("[ERROR] No JSON records read. Check your repo contents.")
        sys.exit(1)
    print(f"[INFO] Total abstractive entries found: {N} across {len(all_bins)} bins.")

    # Targets: 10% held out; rest split 80/10/10
    target_held = round(N * 0.10)
    remaining_after_held = N - target_held
    target_train = round(remaining_after_held * 0.80)
    target_val   = round(remaining_after_held * 0.10)
    target_test  = remaining_after_held - target_train - target_val

    # 2) Allocate HELD-OUT per bin with Hamilton rounding
    held_quotas = [n_per_bin[k] * 0.10 for k in all_bins]
    held_alloc = largest_remainder_alloc(target_held, held_quotas)

    per_bin_held = dict(zip(all_bins, held_alloc))
    # 3) For each bin, allocate the remainder to train/val/test so that
    #    held + train + val + test == bin_size (per-bin exactness).
    per_bin_train, per_bin_val, per_bin_test = {}, {}, {}

    for k in all_bins:
        n = n_per_bin[k]          # size of this bin
        h = per_bin_held[k]       # held-out assigned to this bin
        m = n - h                 # items left for train/val/test in this bin
        if m < 0:
            m = 0

        # 80/10/10 within the remainder
        q_train = m * 0.80
        q_val   = m * 0.10
        q_test  = m * 0.10

        # floors
        t0  = int(math.floor(q_train))
        v0  = int(math.floor(q_val))
        te0 = int(math.floor(q_test))

        assigned = t0 + v0 + te0
        need = m - assigned  # how many to distribute due to rounding

        # largest remainders inside this bin
        remainders = [
            ("train", q_train - t0),
            ("val",   q_val   - v0),
            ("test",  q_test  - te0),
        ]
        remainders.sort(key=lambda x: x[1], reverse=True)

        t, v, te = t0, v0, te0
        for name, _ in remainders:
            if need <= 0:
                break
            if name == "train":
                t += 1
            elif name == "val":
                v += 1
            else:
                te += 1
            need -= 1

        # final safety: trim if any accidental overflow
        while (t + v + te) > m:
            # remove from smallest remainder bucket
            smallest = sorted(
                [("train", q_train - math.floor(q_train), t),
                ("val",   q_val   - math.floor(q_val),   v),
                ("test",  q_test  - math.floor(q_test),  te)],
                key=lambda x: x[1]
            )[0][0]
            if smallest == "train" and t > 0: t -= 1
            elif smallest == "val" and v > 0: v -= 1
            elif smallest == "test" and te > 0: te -= 1
            else: break

        per_bin_train[k] = t
        per_bin_val[k]   = v
        per_bin_test[k]  = te


    # 4) Slice records per bin into the four splits
    splits = {"heldout_pipeline": [], "train": [], "val": [], "test": []}

    print("[DBG] totals:",
      "train", sum(per_bin_train.values()),
      "val",   sum(per_bin_val.values()),
      "test",  sum(per_bin_test.values()),
      "held",  sum(per_bin_held.values()),
      "N",     N)


    for k in all_bins:
        recs = bin_to_recs[k]
        h = per_bin_held[k]; t = per_bin_train[k]; v = per_bin_val[k]; te = per_bin_test[k]
        start = 0
        splits["heldout_pipeline"].extend(recs[start:start+h]); start += h
        splits["train"].extend(recs[start:start+t]); start += t
        splits["val"].extend(recs[start:start+v]); start += v
        splits["test"].extend(recs[start:start+te]); start += te

    # 5) Sanity checks
    assert sum(len(v) for v in splits.values()) == N, "Split sizes do not add up to total"
    print(f"[OK] Split sizes → train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}  heldout={len(splits['heldout_pipeline'])}")

    # 6) Write JSONL files
    def rec_to_jsonl(r: Rec) -> dict:
        return {
            "id": r.id,
            "bin": r.bin_name,
            "summary_sentences": r.summary_sentences,
            "summary_text": r.summary_text,
            "pdf_url": r.pdf_url,
            "source_path": r.path,
        }

    for name, rows in splits.items():
        outp = splits_dir / f"{name}.jsonl"
        with outp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(rec_to_jsonl(r), ensure_ascii=False) + "\n")
        print(f"[WRITE] {name}: {len(rows)} → {outp}")

    # 7) Download PDFs for held-out
    import csv
    manifest_path = out_dir / "heldout_pdf_manifest.csv"
    ok, miss = 0, 0
    with manifest_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["id","bin","pdf_url","pdf_path","source_path"])
        for r in splits["heldout_pipeline"]:
            pdf_url = guess_pdf_url(r.pdf_url)
            if not pdf_url:
                w.writerow([r.id, r.bin_name, r.pdf_url or "", "", r.path])
                miss += 1
                continue
            dest = pdf_dir / f"{r.id if r.id else Path(r.path).stem}.pdf"
            if http_download(pdf_url, dest, timeout=args.timeout, user_agent=args.user_agent):
                w.writerow([r.id, r.bin_name, pdf_url, str(dest), r.path]); ok += 1
            else:
                w.writerow([r.id, r.bin_name, pdf_url, "", r.path]); miss += 1

    print(f"[PDF] Downloaded={ok}  Missing/Failed={miss}")
    print(f"[NOTE] Held-out PDF manifest: {manifest_path}")
    print(f"[DONE] Logs at: {out_dir/'prepare.log'}")

if __name__ == "__main__":
    main()
