#!/usr/bin/env python3
"""
make_scibart_pairs_titles.py

Pair LongSumm abstractive splits (train/val/test.jsonl) with TalkSumm
extractive summaries stored as TITLE-named files.

- Uses talksumm_papers_titles_url.txt to map title -> URL
- Builds keys from URL: arxiv:ID, acl:CODE, and url:<canonical>
- Also scans each extractive file for embedded URLs
- Pairs only the examples that are present in your splits

Outputs:
  {out_dir}/splits/train.paired.jsonl
  {out_dir}/splits/val.paired.jsonl
  {out_dir}/splits/test.paired.jsonl
"""

import argparse, json, re, zipfile, urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Iterable

ARXIV_RE = re.compile(r'arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+)(v[0-9]+)?', re.I)
ACL_RE   = re.compile(r'aclanthology\.org/([A-Za-z0-9.\-]+)', re.I)
HTTP_URL = re.compile(r'(https?://\S+)', re.I)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longsumm_root", required=True, type=str,
                    help="Path to LongSumm repo root (e.g., ./LongSumm-master-github)")
    ap.add_argument("--out_dir", default="./longsumm_prepared", type=str,
                    help="Where splits/ live and outputs go")
    ap.add_argument("--extractive_dir", default=None, type=str,
                    help="Folder with TalkSumm title-named .txt/.tsv files "
                         "(e.g., longsumm_prepared/extractive_unzipped/talksumm_summaries)")
    ap.add_argument("--extractive_zip", default=None, type=str,
                    help="ZIP with TalkSumm files (optional)")
    ap.add_argument("--mapping_file", default=None, type=str,
                    help="talksumm_papers_titles_url.txt (strongly recommended)")
    ap.add_argument("--max_src_len", default=None, type=int,
                    help="Optional: truncate extractive text (chars)")
    return ap.parse_args()

# ---------- normalization & keys ----------
def norm_title(s: str) -> str:
    # normalize for title matching: lowercase, remove punctuation/extra spaces
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)  # keep letters/digits/underscore + spaces
    return s

def canon_url(u: str) -> Optional[str]:
    if not u: return None
    try:
        u = u.strip().strip("()[]<>,.;")
        p = urllib.parse.urlparse(u)
        if p.scheme not in ("http","https"): return None
        net = p.netloc.lower()
        if net.startswith("www."): net = net[4:]
        path = re.sub(r"/+", "/", p.path or "/")
        # canonicalize arXiv & ACL
        m = ARXIV_RE.search(u)
        if m: return f"url:arxiv.org/pdf/{m.group(2)}.pdf"
        m = ACL_RE.search(u)
        if m:
            code = m.group(1)
            if not code.endswith(".pdf"): code += ".pdf"
            return f"url:aclanthology.org/{code}"
        return f"url:{net}{path}"
    except Exception:
        return None

def keys_from_url(u: str) -> List[str]:
    keys = []
    if not u: return keys
    m = ARXIV_RE.search(u)
    if m: keys.append(f"arxiv:{m.group(2)}")
    m = ACL_RE.search(u)
    if m: keys.append(f"acl:{m.group(1)}")
    cu = canon_url(u)
    if cu: keys.append(cu)
    return keys

def keys_from_abstr_record(rec: Dict) -> List[str]:
    keys = []
    u = rec.get("pdf_url") or ""
    keys += keys_from_url(u)
    sp = rec.get("source_path") or ""
    m = re.search(r'([0-9]+\.[0-9]+)', sp)
    if m: keys.append(f"arxiv:{m.group(1)}")
    rid = str(rec.get("id") or "")
    if rid: keys.append(rid)
    # dedup preserve order
    seen=set(); out=[]
    for k in keys:
        if k and k not in seen:
            out.append(k); seen.add(k)
    return out

# ---------- IO ----------
def read_lines(p: Path) -> List[str]:
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()

def unzip_if_needed(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir

# ---------- mapping: title -> URL (→ keys) ----------
def parse_title_url_mapping(mapping_file: Path) -> Dict[str, List[str]]:
    """
    Build mapping: normalized_title -> list(keys).
    Accepts lines like: '<title>\t<url>' or CSV/pipe, or free text with a URL.
    """
    mp: Dict[str, List[str]] = {}
    for ln in read_lines(mapping_file):
        if not ln.strip(): continue
        urls = [m.group(1) for m in HTTP_URL.finditer(ln)]
        if not urls: continue
        u = urls[-1]  # use last URL on the line
        # everything before the URL is considered title-ish
        title_part = ln.split(u, 1)[0].strip().strip("|,;\t:- ")
        if not title_part:
            # fallback: split by common delims and take first non-URL token
            toks = re.split(r'[\t,|]', ln)
            for t in toks:
                t = t.strip()
                if t and not t.startswith("http"):
                    title_part = t; break
        if not title_part:
            continue
        nt = norm_title(title_part)
        ks = keys_from_url(u)
        cu = canon_url(u)
        if cu: ks.append(cu)
        if ks:
            cur = mp.setdefault(nt, [])
            for k in ks:
                if k not in cur:
                    cur.append(k)
    return mp

# ---------- index TalkSumm extractives ----------
def text_from_talksumm(lines: List[str]) -> str:
    sents=[]
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 3:
            sents.append(parts[2].strip())
    return " ".join(sents).strip()

def build_extractive_index(extractive_root: Path,
                           title_key_map: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    """
    Build index: key -> extractive text
    Keys come from:
      - mapping (normalized filename title → URL-derived keys)
      - URLs embedded in the file
      - arXiv ID in filename (rare)
      - fallback: normalized filename title as a key (only useful if abstractive side has same title somewhere)
    """
    idx: Dict[str, str] = {}
    cnt_files = 0; cnt_from_map = 0; cnt_from_fileurl = 0; cnt_from_name = 0
    for p in extractive_root.glob("**/*"):
        if p.is_dir(): continue
        if not p.name.lower().endswith((".txt",".tsv")): continue
        cnt_files += 1
        lines = read_lines(p)
        ext_text = text_from_talksumm(lines)
        if not ext_text: continue

        stem_title = p.stem
        nt = norm_title(stem_title)

        keys: List[str] = []

        # 1) mapping by title
        if title_key_map and nt in title_key_map:
            keys += title_key_map[nt]; cnt_from_map += 1

        # 2) any URL embedded inside file
        content = "\n".join(lines)
        found_url = False
        for m in HTTP_URL.finditer(content):
            ks = keys_from_url(m.group(1))
            if ks:
                keys += ks; found_url = True
        if found_url: cnt_from_fileurl += 1

        # 3) arXiv id in filename
        m = re.search(r'([0-9]+\.[0-9]+)', p.name)
        if m:
            keys.append(f"arxiv:{m.group(1)}"); cnt_from_name += 1

        # 4) fallback: normalized title as last-ditch
        keys.append(f"title:{nt}")

        # store text under all unique keys
        seen=set()
        for k in keys:
            if not k or k in seen: continue
            seen.add(k)
            if k not in idx or (idx[k]=="" and ext_text):
                idx[k] = ext_text

    print(f"[INFO] TalkSumm files: {cnt_files} | via mapping: {cnt_from_map} | via file URLs: {cnt_from_fileurl} | via name/arXiv: {cnt_from_name}")
    return idx

def load_jsonl(p: Path) -> List[Dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def save_jsonl(p: Path, rows: List[Dict]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- main ----------
def main():
    args = parse_args()
    root     = Path(args.longsumm_root).resolve()
    out_dir  = Path(args.out_dir).resolve()
    splits   = out_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    # Resolve extractives
    if args.extractive_dir:
        extr_dir = Path(args.extractive_dir).expanduser().resolve()
        if not extr_dir.exists():
            raise FileNotFoundError(f"--extractive_dir not found: {extr_dir}")
    elif args.extractive_zip:
        extr_dir = unzip_if_needed(Path(args.extractive_zip).expanduser().resolve(),
                                   out_dir / "extractive_unzipped")
    else:
        # default to your unzipped location
        guess = out_dir / "extractive_unzipped" / "talksumm_summaries"
        if guess.exists():
            extr_dir = guess
        else:
            z = root / "extractive_summaries" / "talksumm_summaries.zip"
            if z.exists():
                extr_dir = unzip_if_needed(z, out_dir / "extractive_unzipped")
                extr_dir = out_dir / "extractive_unzipped" / "talksumm_summaries"
            else:
                raise FileNotFoundError("Provide --extractive_dir or unzip the zip first.")

    # Mapping file (title → URL → keys)
    title_key_map = None
    if args.mapping_file:
        mf = Path(args.mapping_file).expanduser().resolve()
        if mf.exists():
            title_key_map = parse_title_url_mapping(mf)
            print(f"[INFO] Mapping entries: {len(title_key_map)} (normalized titles)")
        else:
            print(f"[WARN] mapping file not found: {mf}")

    print(f"[INFO] Indexing extractives from: {extr_dir}")
    ext_idx = build_extractive_index(extr_dir, title_key_map)
    print(f"[INFO] Index size (keys): {len(ext_idx)}")

    # Pair each split
    for name in ("train","val","test"):
        sp = splits / f"{name}.jsonl"
        items = load_jsonl(sp)
        paired, missing = [], []
        for rec in items:
            # build candidate keys from abstractive record
            cand_keys = keys_from_abstr_record(rec)
            # also try a title key if pdf_url contains a title-ish last segment
            # (rare; mainly for completeness)
            ext = None; matched_key = None
            for k in cand_keys:
                ext = ext_idx.get(k)
                if ext:
                    matched_key = k; break
            if not ext:
                missing.append({"id": rec.get("id"), "pdf_url": rec.get("pdf_url"), "bin": rec.get("bin")})
                continue

            src = ext if args.max_src_len is None else ext[: args.max_src_len]
            tgt = rec.get("summary_text") or " ".join(rec.get("summary_sentences", []))
            paired.append({
                "id": rec.get("id"),
                "bin": rec.get("bin"),
                "source": src, "target": tgt,
                "match_key": matched_key,
                "pdf_url": rec.get("pdf_url"),
                "source_path": rec.get("source_path"),
            })

        save_jsonl(splits / f"{name}.paired.jsonl", paired)
        save_jsonl(splits / f"{name}.paired.missing.jsonl", missing)
        print(f"[OK] {name}: paired={len(paired)}  missing_extractive={len(missing)}  → {splits/(name+'.paired.jsonl')}")

if __name__ == "__main__":
    main()
