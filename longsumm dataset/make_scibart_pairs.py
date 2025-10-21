#!/usr/bin/env python3
"""
make_scibart_pairs.py — Pair TalkSumm extractive summaries with LongSumm abstractive summaries.

Outputs:
  {out_dir}/splits/train.paired.jsonl
  {out_dir}/splits/val.paired.jsonl
  {out_dir}/splits/test.paired.jsonl

Each row:
  {"id": "...", "bin":"...", "source":"<extractive text>", "target":"<abstractive text>", ...}
"""

import argparse, json, re, zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ARXIV_RE = re.compile(r'arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+)(v[0-9]+)?', re.I)
ACL_RE   = re.compile(r'aclanthology\.org/([A-Za-z0-9.\-]+)', re.I)
HTTP_URL = re.compile(r'(https?://\S+)', re.I)

# ---------- helpers ----------

def key_from_url(url: str) -> Optional[str]:
    if not url: return None
    m = ARXIV_RE.search(url)
    if m: return f"arxiv:{m.group(2)}"
    m = ACL_RE.search(url)
    if m: return f"acl:{m.group(1)}"
    # some publishers link directly to PDFs with no obvious id; skip
    return None

def key_from_rec(rec: Dict) -> str:
    url = rec.get("pdf_url") or ""
    k = key_from_url(url)
    if k: return k
    sp = rec.get("source_path") or ""
    m = re.search(r'([0-9]+\.[0-9]+)', sp)      # arXiv-ish in filename
    if m: return f"arxiv:{m.group(1)}"
    return str(rec.get("id"))

def read_lines(p: Path) -> List[str]:
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()

def unzip_if_needed(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir

# ------- mapping: TalkSumm filename stem -> arxiv:/acl: key --------

def parse_mapping(mapping_file: Path, known_stems: Optional[set]=None) -> Tuple[Dict[str,str], Dict[str,str]]:
    """
    Return (stem_to_key, key_to_url).

    Handles typical formats:
      ID<TAB>TITLE<TAB>URL
      ID, TITLE, URL
      ID | TITLE | URL
      free text lines where we can still grab a URL

    Heuristic:
      - URL is the last http(s) token on the line
      - stem is the FIRST token that looks like a TalkSumm filename (digits/letters, no slashes)
        or, if known_stems is provided, the first token that matches a known stem.
    """
    stem_to_key: Dict[str, str] = {}
    key_to_url: Dict[str, str] = {}

    for ln in read_lines(mapping_file):
        if not ln.strip(): 
            continue

        # 1) get URL if present
        url = None
        url_matches = list(HTTP_URL.finditer(ln))
        if url_matches:
            url = url_matches[-1].group(1)  # prefer last URL on line

        # 2) pick a candidate stem
        # split on common delimiters
        tokens = re.split(r'[\t,|]', ln)
        cand_stem = None
        for tok in tokens:
            tok = tok.strip().replace(".txt","").replace(".tsv","")
            if not tok or tok.startswith("http"): 
                continue
            # prefer a token that is in known stems if provided
            if known_stems and tok in known_stems:
                cand_stem = tok; break
            # otherwise accept simple alnum tokens (common TalkSumm stems are numeric)
            if re.fullmatch(r'[A-Za-z0-9._\-]+', tok):
                cand_stem = tok; break

        key = key_from_url(url) if url else None
        if cand_stem and key:
            stem_to_key[cand_stem] = key
            key_to_url[key] = url or ""

    return stem_to_key, key_to_url

# ---------- TalkSumm text parsing ----------

def text_from_talksumm(lines: List[str]) -> str:
    """Each line: <sent_idx>\t<score>\t<sentence>"""
    sents=[]
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 3:
            sents.append(parts[2].strip())
    return " ".join(sents).strip()

# ---------- index building ----------

def build_extractive_index(extractive_root: Path,
                           stem_to_key: Optional[Dict[str,str]] = None) -> Tuple[Dict[str,str], set]:
    """
    Walk the extractive files and create:
      key -> extractive_text
    Also return the set of filename stems we saw (for mapping diagnostics).
    """
    idx: Dict[str, str] = {}
    stems_seen: set = set()

    for p in extractive_root.glob("**/*"):
        if p.is_dir(): continue
        if not p.name.lower().endswith((".txt",".tsv")): continue

        stem = p.stem
        stems_seen.add(stem)
        lines = read_lines(p)
        ext_text = text_from_talksumm(lines)

        # 1) use mapping if available
        k = stem_to_key.get(stem) if stem_to_key else None

        # 2) else try URL embedded in file
        if not k:
            content = "\n".join(lines)
            urlm = HTTP_URL.search(content)
            if urlm:
                k = key_from_url(urlm.group(1))

        # 3) else arXiv id in filename
        if not k:
            m = re.search(r'([0-9]+\.[0-9]+)', p.name)
            if m:
                k = f"arxiv:{m.group(1)}"

        # 4) else fallback to stem (will only match if abstractive side uses same stem)
        if not k:
            k = stem

        # store
        if k not in idx or (idx[k] == "" and ext_text):
            idx[k] = ext_text

    return idx, stems_seen

# ---------- IO ----------

def load_jsonl(p: Path) -> List[Dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def save_jsonl(p: Path, rows: List[Dict]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longsumm_root", required=True, type=str)
    ap.add_argument("--out_dir", default="./longsumm_prepared", type=str)
    ap.add_argument("--extractive_dir", default=None, type=str,
                    help="Folder with TalkSumm .txt/.tsv (optional)")
    ap.add_argument("--extractive_zip", default=None, type=str,
                    help="ZIP with TalkSumm files (optional)")
    ap.add_argument("--mapping_file", default=None, type=str,
                    help="talksumm_papers_titles_url.txt (strongly recommended)")
    ap.add_argument("--max_src_len", default=None, type=int)
    args = ap.parse_args()

    root     = Path(args.longsumm_root).resolve()
    out_dir  = Path(args.out_dir).resolve()
    splits   = out_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    # choose extractive source
    if args.extractive_dir:
        extr_dir = Path(args.extractive_dir).expanduser().resolve()
        if not extr_dir.exists():
            raise FileNotFoundError(f"--extractive_dir not found: {extr_dir}")
    elif args.extractive_zip:
        extr_dir = unzip_if_needed(Path(args.extractive_zip).expanduser().resolve(),
                                   out_dir / "extractive_unzipped")
    else:
        guess_zip = root / "extractive_summaries" / "talksumm_summaries.zip"
        if guess_zip.exists():
            extr_dir = unzip_if_needed(guess_zip, out_dir / "extractive_unzipped")
        else:
            raise FileNotFoundError("Provide --extractive_dir or --extractive_zip (zip not found).")

    # pass 0: list stems first (helps mapping quality)
    stems_probe = {p.stem for p in extr_dir.glob('**/*') if p.is_file()}

    # mapping
    stem_to_key = {}
    if args.mapping_file:
        mf = Path(args.mapping_file).expanduser().resolve()
        if mf.exists():
            stem_to_key, _ = parse_mapping(mf, known_stems=stems_probe)
            print(f"[INFO] Mapping loaded: {len(stem_to_key)} stems → keys from {mf}")
        else:
            print(f"[WARN] mapping file not found: {mf}")

    # build index
    print(f"[INFO] Building extractive index from: {extr_dir}")
    ext_idx, stems_seen = build_extractive_index(extr_dir, stem_to_key=stem_to_key)
    print(f"[INFO] Extractive files seen: {len(stems_seen)} | Index size (keys): {len(ext_idx)}")

    # diagnostics: how many stems got a key via mapping
    mapped_stems = set(stem_to_key.keys())
    if mapped_stems:
        print(f"[INFO] Stems mapped via mapping file: {len(mapped_stems & stems_seen)}")

    # pair per split
    for name in ("train","val","test"):
        sp = splits / f"{name}.jsonl"
        items = load_jsonl(sp)
        paired, missing = [], []
        for rec in items:
            k = key_from_rec(rec)
            ext = ext_idx.get(k) or ext_idx.get(str(rec.get("id")))
            if not ext:
                missing.append(rec); continue
            src = ext if args.max_src_len is None else ext[:args.max_src_len]
            tgt = rec.get("summary_text") or " ".join(rec.get("summary_sentences", []))
            paired.append({
                "id": rec.get("id"), "key": k, "bin": rec.get("bin"),
                "source": src, "target": tgt, "pdf_url": rec.get("pdf_url"),
                "source_path": rec.get("source_path"),
            })
        save_jsonl(splits / f"{name}.paired.jsonl", paired)
        save_jsonl(splits / f"{name}.paired.missing.jsonl", missing)
        print(f"[OK] {name}: paired={len(paired)}  missing_extractive={len(missing)}  → {splits/(name+'.paired.jsonl')}")

if __name__ == "__main__":
    main()
