"""########################################
        GETTING DATA FOR EXTRACTION
########################################"""

import urllib.request
from time import sleep
import numpy as np
import re, logging, os, requests
from datetime import datetime

ACL_BASE_URL = "https://aclanthology.org/"
ARXIV_BASE_URL = "https://arxiv.org/pdf/"

if not os.path.exists("./Extraction/talksumm_dataset_bertsum(extractive)/academic_papers_pdfs"):
    os.makedirs("./Extraction/talksumm_dataset_bertsum(extractive)/academic_papers_pdfs")
    downloaded_titles = []
else:
    downloaded_titles = [
        fpath.split("/")[-1].split(".")[0] 
        for fpath in os.listdir("./Extraction/talksumm_dataset_bertsum(extractive)/academic_papers_pdfs")
    ]

def download_file(download_url, filename):
  '''Downloads and saves the pdf file'''
  response = urllib.request.urlopen(download_url)    
  file = open(filename + ".pdf", 'wb')
  file.write(response.read())
  file.close()


def get_pdf_links(response):
  '''Retrieves PDF URLs from response.text'''
  regex = re.compile('(http)(?!.*(http))(.*?)(\.pdf)')
  matches = list(set(["".join(link) for link in regex.findall(response.text)]))  
  return matches  


def multiple_links_handler(pdf_urls):
  '''Gets PDF URLs for specific sites where the generic method finds more than 1 URL'''
  pdf_urls = [link for link in pdf_urls if "-supp" not in link]

  if len(pdf_urls) > 1:
    springer_urls = [link for link in pdf_urls if "link.springer.com" in link]
    pdf_urls = springer_urls if len(springer_urls) > 0 else pdf_urls

  return pdf_urls


def no_links_handler(response, url):
  '''Gets PDF URLs for specific sites where the generic method finds no URLs'''

  if "aclweb" in url or "aclweb" in response.url:
    # Retrieve ACL code from original URL (uppercase)
    idx = -2 if url.endswith("/") else -1
    acl_code = url.split("/")[idx].upper()
    # PDF URL format for ACL papers 
    return [ACL_BASE_URL + acl_code + ".pdf"]
  
  if "arxiv" in url or "arxiv" in response.url:
    idx = -2 if url.endswith("/") else -1
    arxiv_code = url.split("/")[idx]
    # PDF URL format for ARXIV papers 
    return [ARXIV_BASE_URL + arxiv_code + ".pdf"]
  
  if "openreview" in url or "openreview" in response.url:
    openrv_url =  url if "openreview" in url else response.url
    return [openrv_url.replace("forum", "pdf")]
  
  if "transacl" in url or "transacl" in response.url:
    tacl_url =  url if "transacl" in url else response.url
    tacl_regex = re.compile('(http)(?!.*(http))(.*?)(\/tacl\/article\/view\/[[0-9]+\/[[0-9]+)')
    view_urls = list(set(["".join(link) for link in tacl_regex.findall(response.text)]))
    view_urls = [link for link in view_urls if not link.endswith("/0")]
    
    if len(view_urls) > 1:
      # Has more than 1 full text link, take the one with the lowest suffix id number 
      # (this is consistenly the desired TACL site URL)
      suffixes = [int(link.split("/")[-1]) for link in view_urls]
      min_ind = suffixes.index(min(suffixes))
      view_urls = [view_urls[min_ind]]

    if (len(view_urls) == 1):
      return [view_urls[0].replace("view", "download")]
    else:
      return view_urls

  if ("iaaa" in url or "iaaa" in response.url) or (
    "aaai" in url or "aaai" in response.url
  ):
    iaaa_url =  url if ("iaaa" in url or "iaaa" in url) else response.url
    iaaa_regex = re.compile('(http)(?!.*(http))(.*?)(\/paper\/view\/[[0-9]+\/[[0-9]+)')
    view_urls = list(set(["".join(link) for link in tacl_regex.findall(response.text)]))
    return view_urls

  if "mdpi" in url or "mdpi" in response.url:
    mdpi_url =  url if "mdpi" in url else response.url
    mdpi_regex = re.compile('(.*?)(\/[[0-9]+\/[[0-9]+\/[[0-9]+\/pdf)')
    view_urls = list(set(["".join(link) for link in tacl_regex.findall(response.text)]))
    return view_urls

  if "ceur-ws" in url or "ceur-ws" in response.url:
    ceur_url =  url if "ceur-ws" in url else response.url
    idx = -2 if url.endswith("/") else -1
    return [ceur_url + ceur_url.split("/")[idx] + ".pdf"] 

  if "isca-speech" in url or "isca-speech" in response.url:
    isca_url =  url if "isca-speech" in url else response.url
    isca_url = isca_url.replace("abstracts", "pdfs")
    return [isca_url.replace(".html", ".PDF")]

  return []  

failed_titles = []
with open("./Extraction/talksumm_dataset_bertsum(extractive)/talksumm_papers_urls.txt", "r") as input_txt:
  for line in input_txt.readlines():
    try:
      title, url = line.rstrip().split("\t") 

      if title in downloaded_titles:
        continue

      # Sleep to prevent connection reset 
      sleep(np.random.randint(1, 10))

      # Make request to given URL
      response = requests.get(url, allow_redirects=True)

      #Retrieve URLs to PDFs from response
      pdf_links = get_pdf_links(response)

      # Handle too many/too few links
      if len(pdf_links) > 1:
        pdf_links = multiple_links_handler(pdf_links)

      if len(pdf_links) < 1:
        pdf_links = no_links_handler(response, url)

      if len(pdf_links) == 1:
        download_url = pdf_links[0]
      elif url.endswith(".pdf"): # three provided URLs are PDF links
        download_url = url
      else:
        failed_titles.append((title, url))
        raise Exception(f'Got {len(pdf_links)} PDF URLs ({pdf_links})')

      # Download PDF
      download_file(download_url, "./Extraction/talksumm_dataset_bertsum(extractive)/academic_papers_pdfs/" + title)
    
    except Exception as e:
      failed_titles.append((title, url))
      continue









# -*- coding: utf-8 -*-
"""Extraction.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# installing packages


# installing packages
import nltk
nltk.data.path.append("/usr/local/share/nltk_data")
nltk.download('punkt')
nltk.download('punkt_tab')

"""# pre processing the data
## converting pdfs to jsonl - script:


"""

# CONVERTING TO JSONL

"""
Script to convert TalkSumm PDFs to BertSum format
This handles PDF extraction, cleaning, and formats data for BertSum training
"""

import os
import json
import re
from pathlib import Path
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import string

# Download sentence tokenizer (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


def _norm(text: str) -> str:
    """Normalize text for simple matching"""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


class TalkSummToBertSum:
    def __init__(self, pdf_dir, summary_dir, output_dir):
        self.pdf_dir = Path(pdf_dir)
        self.summary_dir = Path(summary_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._summary_index = { _norm(p.stem): p for p in self.summary_dir.glob("*.txt") }

        print("=" * 60)
        print("TalkSumm to BertSum Preprocessor")
        print("=" * 60)
        print(f"PDF directory: {self.pdf_dir}")
        print(f"Summary directory: {self.summary_dir}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                return text
        except Exception as e:
            return None

    def clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\d+\n', '\n', text)
        return text.strip()

    def process_single_file(self, pdf_filename):
        """Process a single PDF and its summary"""
        pdf_path = self.pdf_dir / pdf_filename
        base_name = Path(pdf_filename).stem

        summary_path = self._summary_index.get(_norm(base_name))
        if summary_path is None or not summary_path.exists():
            print(f"  ✗ No summary found for {pdf_filename}")
            return None

        raw_text = self.extract_text_from_pdf(pdf_path)
        if raw_text is None or len(raw_text.strip()) < 100:
            print(f"  ✗ Failed to extract text or text too short for {pdf_filename}")
            return None

        cleaned_text = self.clean_text(raw_text)
        sentences = sent_tokenize(cleaned_text)

        # Read summary
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read().strip()
        except Exception as e:
            return None

        # Create extractive labels (1 if sentence appears in summary, 0 otherwise)
        labels = [1 if _norm(s) in _norm(summary) else 0 for s in sentences]

        # Store raw sentences
        data_entry = {
            'src': sentences,  # List of sentence strings
            'labels': labels,  # Binary labels for extractive summarization
            'tgt': summary,    # Reference summary (for evaluation)
            'filename': pdf_filename
        }

        return data_entry

    def process_all_files(self):
        """Process all PDF files in the directory"""

        # Filter out duplicates/hidden/system PDFs
        pdf_files = sorted([
            f for f in self.pdf_dir.glob("*.pdf")
            if not f.name.startswith('.') and '(' not in f.name
        ])

        print(f"\nFound {len(pdf_files)} PDF files (filtered)")
        print("Starting processing...\n")

        processed_data = []
        failed_files = []

        for idx, pdf_file in enumerate(pdf_files, 1):
            if idx % 50 == 0:
                print(f"Progress: {idx}/{len(pdf_files)} files processed...")
            print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file.name[:50]}...")
            data = self.process_single_file(pdf_file.name)
            if data:
                processed_data.append(data)
                print(f"  ✓ Success (sentences: {len(data['src'])}, summary labels: {sum(data['labels'])})")
            else:
                failed_files.append(pdf_file.name)

        print("\n" + "=" * 60)
        print(f"Successfully processed: {len(processed_data)}")
        print(f"Failed: {len(failed_files)}")
        print("=" * 60)
        if failed_files:
            print("First 10 failed files:")
            print(failed_files[:10])
        return processed_data, failed_files

    def save_processed_data(self, processed_data):
        """Save all processed data as JSONL"""
        output_file = self.output_dir / 'talksumm_processed.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in processed_data:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Saved all data to: {output_file}")
        return output_file

    def create_train_val_test_split(self, processed_data, train_ratio=0.8, val_ratio=0.1):
        """Split data into train/val/test sets"""
        import random
        data_copy = processed_data.copy()
        random.seed(42)
        random.shuffle(data_copy)

        n = len(data_copy)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data = data_copy[:train_end]
        val_data = data_copy[train_end:val_end]
        test_data = data_copy[val_end:]

        splits = [('train', train_data), ('val', val_data), ('test', test_data)]
        for split_name, split_data in splits:
            output_file = self.output_dir / f'talksumm_{split_name}.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in split_data:
                    f.write(json.dumps(entry) + '\n')
            print(f"✓ {split_name.upper()}: {len(split_data)} samples -> {output_file}")

        return train_data, val_data, test_data


# ============================================================
# MAIN SCRIPT - to be adjusted to local directory
# ============================================================
if __name__ == "__main__":
    PDF_DIR = "./Extraction/talksumm_dataset_bertsum(extractive)/academic_papers_pdfs"
    SUMMARY_DIR = "./Extraction/talksumm_dataset_bertsum(extractive)/talksumm_summaries"
    OUTPUT_DIR = "./Extraction/Processed_Data"

    processor = TalkSummToBertSum(
        pdf_dir=PDF_DIR,
        summary_dir=SUMMARY_DIR,
        output_dir=OUTPUT_DIR
    )

    processed_data, failed_files = processor.process_all_files()
    processor.save_processed_data(processed_data)
    processor.create_train_val_test_split(processed_data)

"""# preprocessing data
## converting jsonl into datasetdict
"""

# ============================================================
# Convert processed JSONL files into HF DatasetDict
# ============================================================


from datasets import Dataset, DatasetDict
import json
from pathlib import Path

# Paths to your processed JSONL files in Drive
PROCESSED_DIR = Path("./Extraction/Processed_Data")
train_file = PROCESSED_DIR / "talksumm_train.jsonl"
val_file   = PROCESSED_DIR / "talksumm_val.jsonl"
test_file  = PROCESSED_DIR / "talksumm_test.jsonl"

def load_jsonl(file_path):
    """Load JSONL file into list of dicts"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


# Load each split
train_data = load_jsonl(train_file)
val_data   = load_jsonl(val_file)
test_data  = load_jsonl(test_file)

# Check the keys in your first example to see what columns you have
print("Sample keys:", train_data[0].keys())
# Should include: 'src', 'labels', 'tgt', 'filename'

# Convert lists to HF Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset   = Dataset.from_list(val_data)
test_dataset  = Dataset.from_list(test_data)

# Create DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Save DatasetDict to disk
OUTPUT_PATH = "./Extraction/bertsum-hf/data_hf"
dataset_dict.save_to_disk(OUTPUT_PATH)
print(f"DatasetDict saved to: {OUTPUT_PATH}")
# Load each split
train_data = load_jsonl(train_file)
val_data   = load_jsonl(val_file)
test_data  = load_jsonl(test_file)

# Check the keys in your first example to see what columns you have
print("Sample keys:", train_data[0].keys())
# Should include: 'src', 'tgt', 'src_sentences', 'filename'

# Convert lists to HF Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset   = Dataset.from_list(val_data)
test_dataset  = Dataset.from_list(test_data)

# Create DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Save DatasetDict to disk (so run_extsum.py can read it)
OUTPUT_PATH = "./Extraction/bertsum-hf/data_hf"
dataset_dict.save_to_disk(OUTPUT_PATH)
print(f"DatasetDict saved to: {OUTPUT_PATH}")

"""# set up to run bertsum"""

# =========================
# SETUP BERTSUM ENVIRONMENT
# =========================



# Download NLTK punkt tokenizer
import nltk
nltk.download('punkt')



# Check versions to confirm
import transformers, torch, datasets, evaluate, nltk, wandb
print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("Datasets:", datasets.__version__)
print("Evaluate:", evaluate.__version__)
print("NLTK:", nltk.__version__)
print("WandB:", wandb.__version__)

print("\n Environment setup complete. Ready to run BertSum training")

import os
os.environ["WANDB_DISABLED"] = "true"

"""# running bertsum"""
# ================================
# BERTSUM EXTRACTIVE TRAINING
# (inlined version of run_extsum.py)
# ================================
import os
from pathlib import Path
import gc
from functools import partial
from typing import Optional
import sys

import torch
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_from_disk

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
BERTSUM_ROOT = PROJECT_ROOT / "Extraction" / "bertsum-hf"

# Make sure we can import from the cloned bertsum repo
sys.path.append(str(BERTSUM_ROOT))

from src.bertsum import BertSummarizer, BertSummarizerConfig
from utils import CFG, seed_everything
from src.data_preparation import (
    preprocess_train,
    preprocess_validation,
    SummarizerDataset,
    collate_batch,
)
from src.trainer import SummarizerTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def run_bertsum_extractive(
    input_data_path: str = "Extraction/bertsum-hf/data_hf",
    output_dir: str = "Extraction/models",
    config_path: str = "Extraction/bertsum-hf/configs/config.json",
    preprocessed_output_dir: Optional[str] = None,
):
    """
    Inlined version of run_extsum.py.
    - input_data_path: HF dataset directory (from dataset_dict.save_to_disk)
    - output_dir: where to save model checkpoints
    - config_path: path to BERTSum config JSON
    - preprocessed_output_dir: optional directory to save preprocessed HF dataset
      (only used if cfg.needs_preprocessing = True)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    cfg_path = PROJECT_ROOT / config_path
    print(f"[BERTSUM] Loading config from: {cfg_path}")
    cfg = CFG(cfg_path, device)

    seed_everything(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)

    # ---- Load / preprocess dataset ----
    data_dir = PROJECT_ROOT / input_data_path
    print(f"[BERTSUM] Loading dataset from: {data_dir}")
    data = load_from_disk(data_dir)

    preprocess_train_partial = partial(preprocess_train, tokenizer=tokenizer, cfg=cfg)
    preprocess_validation_partial = partial(preprocess_validation, tokenizer=tokenizer, cfg=cfg)

    if cfg.needs_preprocessing:
        print("[BERTSUM] Preprocessing train/validation/test splits...")
        data_train = data["train"].map(
            preprocess_train_partial,
            batched=True,
            batch_size=cfg.preprocessing_batch_size,
            load_from_cache_file=False,
            remove_columns=data["train"].column_names,
            desc="Preprocessing train set",
        )

        data_validation = data["validation"].map(
            preprocess_validation_partial,
            batched=True,
            batch_size=cfg.preprocessing_batch_size,
            load_from_cache_file=False,
            remove_columns=data["validation"].column_names,
            desc="Preprocessing validation set",
        )

        data_test = data["test"].map(
            preprocess_validation_partial,
            batched=True,
            batch_size=cfg.preprocessing_batch_size,
            load_from_cache_file=False,
            remove_columns=data["test"].column_names,
            desc="Preprocessing test set",
        )

        if cfg.store_preprocessed_data:
            if preprocessed_output_dir is None:
                preprocessed_output_dir = "Extraction/bertsum-hf/data_hf_preprocessed"

            out_path = PROJECT_ROOT / preprocessed_output_dir
            print(f"[BERTSUM] Saving preprocessed dataset to: {out_path}")
            data["train"] = data_train
            data["validation"] = data_validation
            data["test"] = data_test
            data.save_to_disk(out_path)
            del data
            gc.collect()
    else:
        print("[BERTSUM] Using already-preprocessed dataset.")
        data_train = data["train"]
        data_validation = data["validation"]

    # Optional subsets for debugging
    if cfg.train_subset is not None:
        data_train = data_train.select(range(cfg.train_subset))

    if cfg.eval_subset is not None:
        data_validation = data_validation.select(range(cfg.eval_subset))

    # ---- Load model ----
    print("[BERTSUM] Loading model...")
    model_config = BertSummarizerConfig(checkpoint=cfg.checkpoint)
    model = BertSummarizer(config=model_config)

    # ---- Build datasets ----
    print("[BERTSUM] Building datasets...")
    train_dataset = SummarizerDataset(data_train)
    validation_dataset = SummarizerDataset(data_validation)

    # ---- Training arguments ----
    model_output_dir = PROJECT_ROOT / output_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BERTSUM] Training output dir: {model_output_dir}")

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        group_by_length=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=cfg.checkpoint_steps,
        save_steps=cfg.checkpoint_steps,
        include_inputs_for_metrics=True,
        prediction_loss_only=False,
        report_to="none",
        run_name=cfg.run_name,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
    )

    # ---- Trainer ----
    num_training_steps = (
        cfg.num_epochs * len(train_dataset) / cfg.train_batch_size / cfg.gradient_accumulation_steps
    )
    warmup_steps = int(cfg.warmup_ratio * num_training_steps)

    partial_collate_batch = partial(collate_batch, pad_token_id=tokenizer.pad_token_id)

    trainer = SummarizerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=partial_collate_batch,
        tokenizer=tokenizer,
        warmup_steps=warmup_steps,
        device=cfg.device,
        cfg=cfg,
        use_pos_weight=cfg.use_pos_weight,
        pos_weight_alpha=cfg.pos_weight_alpha,
    )

    # ---- Train ----
    print("[BERTSUM] Fine-tuning model...")
    trainer.train()

    # ---- Save final model ----
    final_path = model_output_dir / "bertsum"
    print(f"[BERTSUM] Saving final model to: {final_path}")
    trainer.save_model(str(final_path))
    print("[BERTSUM] Training complete.")





"""# rouge scores for first training"""

# ROUGE SCORE FOR FIRST TRAINING
from datasets import load_from_disk
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial
import sys

# Import your custom BertSummarizer
sys.path.append('./Extraction/bertsum-hf')
from src.bertsum import BertSummarizer, BertSummarizerConfig
from src.data_preparation import preprocess_validation
from utils import CFG

# Paths
checkpoint_path = "./Extraction/models/checkpoint-214"
data_path = "./Extraction/bertsum-hf/data_hf"
cfg_path = "./Extraction/bertsum-hf/configs/config.json"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, tokenizer, and config
model_config = BertSummarizerConfig.from_pretrained(checkpoint_path)
model = BertSummarizer.from_pretrained(checkpoint_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
cfg = CFG(Path(cfg_path), device=device)

model.to(device)
model.eval()

# Load validation dataset
val_dataset = load_from_disk(data_path)['validation']

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Maximum tokens per chunk (BERT limit)
MAX_TOKENS = 512
TOP_K_SENTENCES = 12

# Evaluation loop
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for example in tqdm(val_dataset, desc="Evaluating"):
    preprocess_fn = partial(preprocess_validation, tokenizer=tokenizer, cfg=cfg)
    processed = preprocess_fn({
        'src': [example['src']],
        'labels': [example['labels']],
        'tgt': [example['tgt']]
    })

    input_ids_full = processed['input_ids'][0]
    cls_ids_full = processed['cls_ids'][0]
    cls_mask_full = processed['mask_cls'][0]

    # Split into chunks of MAX_TOKENS
    chunk_starts = list(range(0, len(input_ids_full), MAX_TOKENS))
    sentence_scores_list = []

    with torch.no_grad():
        for start in chunk_starts:
            end = min(start + MAX_TOKENS, len(input_ids_full))
            chunk_input_ids = torch.tensor(input_ids_full[start:end]).unsqueeze(0).to(device)
            chunk_cls_ids = torch.tensor(cls_ids_full[start:end]).unsqueeze(0).to(device)
            chunk_cls_mask = torch.tensor(cls_mask_full[start:end]).unsqueeze(0).to(device)

            # Only keep valid cls positions
            valid_cls = (chunk_cls_ids < MAX_TOKENS).squeeze()
            chunk_cls_ids = chunk_cls_ids[:, valid_cls]
            chunk_cls_mask = chunk_cls_mask[:, valid_cls]

            if chunk_cls_ids.size(1) == 0:
                continue  # skip chunks with no sentences

            outputs = model(input_ids=chunk_input_ids, cls_ids=chunk_cls_ids, mask_cls=chunk_cls_mask)
            chunk_scores = outputs['logits'].squeeze()
            valid_mask = chunk_cls_mask.squeeze().bool()
            sentence_scores_list.append(chunk_scores[valid_mask])

        if len(sentence_scores_list) == 0:
            continue  # skip empty examples

        all_scores = torch.cat(sentence_scores_list)
        k = min(TOP_K_SENTENCES, len(all_scores))
        top_k_indices = sorted(torch.topk(all_scores, k).indices.cpu().tolist()) if k > 0 else []

    # Reconstruct summary
    selected_sentences = [example['src'][i] for i in top_k_indices if i < len(example['src'])]
    generated_summary = ' '.join(selected_sentences)
    reference_summary = example['tgt']

    # Compute ROUGE
    if generated_summary and reference_summary:
        scores = scorer.score(reference_summary, generated_summary)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

# Print average ROUGE
print(f"\nROUGE Scores on Validation Set:")
print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
print(f"Number of examples evaluated: {len(rouge1_scores)}")

"""# zero shot baseline"""

#** -Zero shot baseline inference - so that we can compare fine-tuned model performance against baseline model- **



# --- 4. Add repo to Python path ---
import sys
sys.path.append('./Extraction/bertsum-hf')

# --- 5. Download NLTK tokenizer ---
import nltk
nltk.download('punkt')

print("\n Environment ready.")

# ================================================================
# ZERO-SHOT BASELINE EVALUATION (no fine-tuning)
# ================================================================

from datasets import load_from_disk
from transformers import AutoTokenizer, BertModel
from rouge_score import rouge_scorer
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial
import sys

# ------------------------------------------------
# Add repo to path and import local modules
# ------------------------------------------------
sys.path.append('./Extraction/bertsum-hf')
from src.bertsum import BertSummarizer, BertSummarizerConfig
from src.data_preparation import preprocess_validation
from utils import CFG

# ------------------------------------------------
# Paths and device
# ------------------------------------------------
data_path = "./Extraction/bertsum-hf/data_hf"
cfg_path = "./Extraction/bertsum-hf/configs/config.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base pretrained model (no fine-tuning)
model_name = "bert-base-uncased"

# Load config and manually set checkpoint field
model_config = BertSummarizerConfig.from_pretrained(cfg_path)
model_config.checkpoint = model_name
# Initialize BertSum
model = BertSummarizer(config=model_config)
model.bert = BertModel.from_pretrained(model_name)
# Tokenizer + config
tokenizer = AutoTokenizer.from_pretrained(model_name)
cfg = CFG(Path(cfg_path), device=device)

model.to(device)
model.eval()

# Load test dataset
test_dataset = load_from_disk(data_path)["test"]
print(f"Loaded test dataset with {len(test_dataset)} samples.")

# ROUGE scorer setup
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

MAX_TOKENS = 512
TOP_K_SENTENCES = 3
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

# Evaluation loop
for example in tqdm(test_dataset, desc="Zero-shot evaluating"):
    preprocess_fn = partial(preprocess_validation, tokenizer=tokenizer, cfg=cfg)
    processed = preprocess_fn({
        "src": [example["src"]],
        "labels": [example["labels"]],
        "tgt": [example["tgt"]],
    })

    input_ids = torch.tensor(processed["input_ids"][0]).unsqueeze(0).to(device)
    cls_ids = torch.tensor(processed["cls_ids"][0]).unsqueeze(0).to(device)
    cls_mask = torch.tensor(processed["mask_cls"][0]).unsqueeze(0).to(device)

    # Truncate to 512 tokens if too long
    if input_ids.size(1) > MAX_TOKENS:
        input_ids = input_ids[:, :MAX_TOKENS]
        cls_ids = cls_ids[:, cls_ids.squeeze() < MAX_TOKENS]
        cls_mask = cls_mask[:, :cls_ids.size(1)]

    with torch.no_grad():
        segment_ids = torch.zeros_like(input_ids)
        mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            segment_ids=segment_ids,
            cls_ids=cls_ids,
            mask=mask,
            mask_cls=cls_mask,
        )
        sentence_scores = outputs["logits"].squeeze()

    # Select top sentences
    k = min(TOP_K_SENTENCES, len(sentence_scores))
    top_k_indices = sorted(torch.topk(sentence_scores, k).indices.cpu().tolist())
    generated_summary = " ".join(example["src"][i] for i in top_k_indices)
    reference_summary = example["tgt"]

    # Compute ROUGE
    if generated_summary and reference_summary:
        scores = scorer.score(reference_summary, generated_summary)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

# ------------------------------------------------
# Print results
# ------------------------------------------------
print("\n=== ZERO-SHOT ROUGE BASELINE (TEST SET) ===")
print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
print(f"Number of evaluated examples: {len(rouge1_scores)}")

"""# fine tuning bertsum stage 1"""
run_bertsum_extractive(
    input_data_path="Extraction/bertsum-hf/data_hf",
    output_dir="Extraction/models_finetuned",
    config_path="Extraction/bertsum-hf/configs/config_finetune.json",
)
"""# rouge scores for fine-tuned bertsum stage 1"""

# FINETUNED ROUGE
from datasets import load_from_disk
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial
import sys

# Import your custom BertSummarizer
sys.path.append('./Extraction/bertsum-hf')
from src.bertsum import BertSummarizer, BertSummarizerConfig
from src.data_preparation import preprocess_validation
from utils import CFG

# Paths
checkpoint_path = "./Extraction/models_finetuned/bertsum"
data_path = "./Extraction/bertsum-hf/data_hf"
cfg_path = "./Extraction/bertsum-hf/configs/config_finetune.json"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, tokenizer, and config
model_config = BertSummarizerConfig.from_pretrained(checkpoint_path)
model = BertSummarizer.from_pretrained(checkpoint_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
cfg = CFG(Path(cfg_path), device=device)

model.to(device)
model.eval()

# Load validation dataset
val_dataset = load_from_disk(data_path)['validation']

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Maximum tokens per chunk (BERT limit)
MAX_TOKENS = 512
TOP_K_SENTENCES = 12

# Evaluation loop
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for example in tqdm(val_dataset, desc="Evaluating"):
    preprocess_fn = partial(preprocess_validation, tokenizer=tokenizer, cfg=cfg)
    processed = preprocess_fn({
        'src': [example['src']],
        'labels': [example['labels']],
        'tgt': [example['tgt']]
    })

    input_ids_full = processed['input_ids'][0]
    cls_ids_full = processed['cls_ids'][0]
    cls_mask_full = processed['mask_cls'][0]

    # Split into chunks of MAX_TOKENS
    chunk_starts = list(range(0, len(input_ids_full), MAX_TOKENS))
    sentence_scores_list = []

    with torch.no_grad():
        for start in chunk_starts:
            end = min(start + MAX_TOKENS, len(input_ids_full))
            chunk_input_ids = torch.tensor(input_ids_full[start:end]).unsqueeze(0).to(device)
            chunk_cls_ids = torch.tensor(cls_ids_full[start:end]).unsqueeze(0).to(device)
            chunk_cls_mask = torch.tensor(cls_mask_full[start:end]).unsqueeze(0).to(device)

            # Only keep valid cls positions
            valid_cls = (chunk_cls_ids < MAX_TOKENS).squeeze()
            chunk_cls_ids = chunk_cls_ids[:, valid_cls]
            chunk_cls_mask = chunk_cls_mask[:, valid_cls]

            if chunk_cls_ids.size(1) == 0:
                continue  # skip chunks with no sentences

            outputs = model(input_ids=chunk_input_ids, cls_ids=chunk_cls_ids, mask_cls=chunk_cls_mask)
            chunk_scores = outputs['logits'].squeeze()
            valid_mask = chunk_cls_mask.squeeze().bool()
            sentence_scores_list.append(chunk_scores[valid_mask])

        if len(sentence_scores_list) == 0:
            continue  # skip empty examples

        all_scores = torch.cat(sentence_scores_list)
        k = min(TOP_K_SENTENCES, len(all_scores))
        top_k_indices = sorted(torch.topk(all_scores, k).indices.cpu().tolist()) if k > 0 else []

    # Reconstruct summary
    selected_sentences = [example['src'][i] for i in top_k_indices if i < len(example['src'])]
    generated_summary = ' '.join(selected_sentences)
    reference_summary = example['tgt']

    # Compute ROUGE
    if generated_summary and reference_summary:
        scores = scorer.score(reference_summary, generated_summary)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

# Print average ROUGE
print(f"\nROUGE Scores on Validation Set:")
print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
print(f"Number of examples evaluated: {len(rouge1_scores)}")

import json
from pathlib import Path

processed_dir = Path("./Extraction/Processed_Data")
processed_file = processed_dir / "talksumm_processed.jsonl"

# Load all entries
with open(processed_file, "r", encoding="utf-8") as f:
    processed_data = [json.loads(line) for line in f]

summary_lengths = []

for entry in processed_data:
    # Count sentences in the target summary by splitting on periods
    num_sentences = len(entry['tgt'].split('.'))
    summary_lengths.append(num_sentences)

average_summary_length = sum(summary_lengths) / len(summary_lengths)
max_summary_length = max(summary_lengths)
min_summary_length = min(summary_lengths)

print(f"Average summary length: {average_summary_length:.1f} sentences")
print(f"Max summary length: {max_summary_length} sentences")
print(f"Min summary length: {min_summary_length} sentences")

"""# fine-tuning bertsum stage 2"""

run_bertsum_extractive(
    input_data_path="Extraction/bertsum-hf/data_hf",
    output_dir="Extraction/model_finetuned_v2",
    config_path="Extraction/bertsum-hf/configs/config_finetune_v2.json",
)

"""# rouge scores for fine-tuned bertsum stage 2"""

# FINETUNED ROUGE
from datasets import load_from_disk
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial
import sys

# Import your custom BertSummarizer
sys.path.append('./Extraction/bertsum-hf')
from src.bertsum import BertSummarizer, BertSummarizerConfig
from src.data_preparation import preprocess_validation
from utils import CFG

# Paths
checkpoint_path = "./Extraction/model_finetuned_v2/bertsum"
data_path = "./Extraction/bertsum-hf/data_hf"
cfg_path = "./Extraction/bertsum-hf/configs/config_finetune_v2.json"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, tokenizer, and config
model_config = BertSummarizerConfig.from_pretrained(checkpoint_path)
model = BertSummarizer.from_pretrained(checkpoint_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
cfg = CFG(Path(cfg_path), device=device)

model.to(device)
model.eval()

# Load validation dataset
val_dataset = load_from_disk(data_path)['validation']

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Maximum tokens per chunk (BERT limit)
MAX_TOKENS = 512
TOP_K_SENTENCES = 45

# Evaluation loop
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for example in tqdm(val_dataset, desc="Evaluating"):
    preprocess_fn = partial(preprocess_validation, tokenizer=tokenizer, cfg=cfg)
    processed = preprocess_fn({
        'src': [example['src']],
        'labels': [example['labels']],
        'tgt': [example['tgt']]
    })

    input_ids_full = processed['input_ids'][0]
    cls_ids_full = processed['cls_ids'][0]
    cls_mask_full = processed['mask_cls'][0]

    # Split into chunks of MAX_TOKENS
    chunk_starts = list(range(0, len(input_ids_full), MAX_TOKENS))
    sentence_scores_list = []

    with torch.no_grad():
        for start in chunk_starts:
            end = min(start + MAX_TOKENS, len(input_ids_full))
            chunk_input_ids = torch.tensor(input_ids_full[start:end]).unsqueeze(0).to(device)
            chunk_cls_ids = torch.tensor(cls_ids_full[start:end]).unsqueeze(0).to(device)
            chunk_cls_mask = torch.tensor(cls_mask_full[start:end]).unsqueeze(0).to(device)

            # Only keep valid cls positions
            valid_cls = (chunk_cls_ids < MAX_TOKENS).squeeze()
            chunk_cls_ids = chunk_cls_ids[:, valid_cls]
            chunk_cls_mask = chunk_cls_mask[:, valid_cls]

            if chunk_cls_ids.size(1) == 0:
                continue  # skip chunks with no sentences

            outputs = model(input_ids=chunk_input_ids, cls_ids=chunk_cls_ids, mask_cls=chunk_cls_mask)
            chunk_scores = outputs['logits'].squeeze()
            valid_mask = chunk_cls_mask.squeeze().bool()
            sentence_scores_list.append(chunk_scores[valid_mask])

        if len(sentence_scores_list) == 0:
            continue  # skip empty examples

        all_scores = torch.cat(sentence_scores_list)
        k = min(len(example['tgt'].split('.')), len(all_scores))
        top_k_indices = sorted(torch.topk(all_scores, k).indices.cpu().tolist())


    # Reconstruct summary
    selected_sentences = [example['src'][i] for i in top_k_indices if i < len(example['src'])]
    generated_summary = ' '.join(selected_sentences)
    reference_summary = example['tgt']

    # Compute ROUGE
    if generated_summary and reference_summary:
        scores = scorer.score(reference_summary, generated_summary)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

# Print average ROUGE
print(f"\nROUGE Scores on Validation Set:")
print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
print(f"Number of examples evaluated: {len(rouge1_scores)}")

"""# rouge scores fine-tuned bertsum stage 2 on test set"""

# FINETUNED ROUGE EVALUATION ON TEST SET
from datasets import load_from_disk
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial
import sys

# Import your custom BertSummarizer
sys.path.append('./Extraction/bertsum-hf')
from src.bertsum import BertSummarizer, BertSummarizerConfig
from src.data_preparation import preprocess_validation
from utils import CFG

# Paths
checkpoint_path = "./Extraction/model_finetuned_v2/bertsum"
data_path = "./Extraction/bertsum-hf/data_hf"
cfg_path = "./Extraction/bertsum-hf/configs/config_finetune_v2.json"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, tokenizer, and config
model_config = BertSummarizerConfig.from_pretrained(checkpoint_path)
model = BertSummarizer.from_pretrained(checkpoint_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
cfg = CFG(Path(cfg_path), device=device)

model.to(device)
model.eval()

# Load validation dataset
test_dataset = load_from_disk(data_path)['test']

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Maximum tokens per chunk (BERT limit)
MAX_TOKENS = 512

# Evaluation loop
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for example in tqdm(test_dataset, desc="Evaluating"):
    preprocess_fn = partial(preprocess_validation, tokenizer=tokenizer, cfg=cfg)
    processed = preprocess_fn({
        'src': [example['src']],
        'labels': [example['labels']],
        'tgt': [example['tgt']]
    })

    input_ids_full = processed['input_ids'][0]
    cls_ids_full = processed['cls_ids'][0]
    cls_mask_full = processed['mask_cls'][0]

    # Split into chunks of MAX_TOKENS
    chunk_starts = list(range(0, len(input_ids_full), MAX_TOKENS))
    sentence_scores_list = []

    with torch.no_grad():
        for start in chunk_starts:
            end = min(start + MAX_TOKENS, len(input_ids_full))
            chunk_input_ids = torch.tensor(input_ids_full[start:end]).unsqueeze(0).to(device)
            chunk_cls_ids = torch.tensor(cls_ids_full[start:end]).unsqueeze(0).to(device)
            chunk_cls_mask = torch.tensor(cls_mask_full[start:end]).unsqueeze(0).to(device)

            # Only keep valid cls positions
            valid_cls = (chunk_cls_ids < MAX_TOKENS).squeeze()
            chunk_cls_ids = chunk_cls_ids[:, valid_cls]
            chunk_cls_mask = chunk_cls_mask[:, valid_cls]

            if chunk_cls_ids.size(1) == 0:
                continue  # skip chunks with no sentences

            outputs = model(input_ids=chunk_input_ids, cls_ids=chunk_cls_ids, mask_cls=chunk_cls_mask)
            chunk_scores = outputs['logits'].squeeze()
            valid_mask = chunk_cls_mask.squeeze().bool()
            sentence_scores_list.append(chunk_scores[valid_mask])

        if len(sentence_scores_list) == 0:
            continue  # skip empty examples

        all_scores = torch.cat(sentence_scores_list)
        # Dynamically select top-k sentences based on reference summary
        k = min(len(example['tgt'].split('.')), len(all_scores))
        top_k_indices = sorted(torch.topk(all_scores, k).indices.cpu().tolist()) if k > 0 else []

    # Reconstruct summary
    selected_sentences = [example['src'][i] for i in top_k_indices if i < len(example['src'])]
    generated_summary = ' '.join(selected_sentences)
    reference_summary = example['tgt']

    # Compute ROUGE
    if generated_summary and reference_summary:
        scores = scorer.score(reference_summary, generated_summary)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

# Print average ROUGE
print(f"\nROUGE Scores on Validation Set:")
print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
print(f"Number of examples evaluated: {len(rouge1_scores)}")

"""# extracting summaries for manual checking"""

# =============================================
# Extract summaries from DatasetDict using BertSum
# (LIMITED TO A FEW SAMPLES FOR MANUAL CHECKING)
# =============================================

import sys, os, json
sys.path.append("./Extraction/bertsum-hf")

import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from src.bertsum import BertSummarizer, BertSummarizerConfig
from tqdm import tqdm
import numpy as np

# -------------------------------
# Config
# -------------------------------
checkpoint_dir = "./Extraction/model_finetuned_v2/bertsum"
dataset_path  = "./Extraction/bertsum-hf/data_hf"
output_dir    = "./Extraction/summaries_analysis"

MAX_TOKENS = 512
TOP_K = 10
N_SAMPLES = 5   # <-- only extract 5 samples per split

os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Load model + tokenizer
# -------------------------------
tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
config = BertSummarizerConfig.from_pretrained(checkpoint_dir)
model = BertSummarizer.from_pretrained(checkpoint_dir, config=config)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------
# Load DatasetDict
# -------------------------------
dataset_dict = load_from_disk(dataset_path)

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_for_bertsum(item):
    sentences = item['src']  # <-- use 'src' from TalkSumm dataset
    input_ids, cls_ids = [], []

    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        if len(tokens) == 0:
            continue
        if len(input_ids) + len(tokens) + 1 > MAX_TOKENS:
            break
        cls_ids.append(len(input_ids))
        input_ids.extend([tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens))

    mask = [1] * len(input_ids)
    mask_cls = [1] * len(cls_ids)

    return {
        'src': sentences,
        'filename': item.get('filename', None),
        'input_ids': torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device),
        'cls_ids': torch.tensor(cls_ids, dtype=torch.long).unsqueeze(0).to(device),
        'mask': torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(device),
        'mask_cls': torch.tensor(mask_cls, dtype=torch.long).unsqueeze(0).to(device)
    }


# =============================================
# Extraction loop (LIMITED)
# =============================================

for split, dataset in dataset_dict.items():
    print(f"\nProcessing split: {split} ({len(dataset)} documents)")

    all_summaries = []

    # Only take the first N_SAMPLES examples from each split
    subset = dataset.select(range(min(N_SAMPLES, len(dataset))))

    for i, item in enumerate(tqdm(subset, desc=f"{split} progress (first {N_SAMPLES})")):

        # Preprocess
        processed = preprocess_for_bertsum(item)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                processed['input_ids'],
                cls_ids=processed['cls_ids'],
                mask=processed['mask'],
                mask_cls=processed['mask_cls']
            )

        # Extract logits
        scores = outputs['logits'].squeeze().cpu().numpy()
        if np.ndim(scores) == 0:
            scores = np.array([scores])

        top_k = min(TOP_K, len(scores))
        top_indices = scores.argsort()[-top_k:][::-1]

        # Select summary sentences
        selected_sentences = [item['src'][j] for j in sorted(top_indices)]
        summary = " ".join(selected_sentences)

        # Build output record
        record = {
            "id": item.get("id"),
            "filename": item.get("filename"),
            "pred_summary": summary,
            "gold_summary": item.get("tgt"),  # TalkSumm gold summary
            "sentences": item["src"]
        }

        all_summaries.append(record)

        # Print to console for inspection
        print("\n===================================")
        print(f"Sample {i} | Split: {split}")
        print(f"Filename: {record['filename']}")
        print("\n--- Predicted Summary ---")
        print(summary)
        print("\n--- Gold Summary ---")
        print(record["gold_summary"])
        print("===================================\n")

    # Save the samples to file
    output_file = os.path.join(output_dir, f"sample_summaries_{split}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in all_summaries:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(all_summaries)} samples to: {output_file}")

"""# results of extracted summaries vs gold summaries"""

import json
import numpy as np
import glob

# -------------------------------
# Helper functions
# -------------------------------

def compute_coverage(pred, gold):
    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    if len(gold_tokens) == 0:
        return 0
    return len(pred_tokens & gold_tokens) / len(gold_tokens)

def lead_bias(rec, quantile=0.2):
    total_sents = len(rec['sentences'])
    if total_sents == 0:
        return 0
    top_n = max(1, int(total_sents * quantile))  # first 20% of sentences
    selected_count = 0
    for sent in rec['sentences'][:top_n]:
        if sent in rec['pred_summary']:
            selected_count += 1
    return selected_count / max(1, len(rec['pred_summary'].split('.')))

# -------------------------------
# Process all splits
# -------------------------------

jsonl_files = glob.glob("./Extraction/summaries_analysis/sample_summaries_*.jsonl")

for file in jsonl_files:
    split_name = file.split('_')[-1].replace('.jsonl','')

    with open(file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    pred_lengths = [len(rec['pred_summary'].split()) for rec in records]
    gold_lengths = [len(rec['gold_summary'].split()) for rec in records]
    coverages = [compute_coverage(rec['pred_summary'], rec['gold_summary']) for rec in records]
    lead_biases = [lead_bias(rec) for rec in records]

    print(f"\nSplit: {split_name}")
    print(f"Predicted Summary Lengths: mean={np.mean(pred_lengths):.1f}, min={np.min(pred_lengths)}, max={np.max(pred_lengths)}")
    print(f"Gold Summary Lengths: mean={np.mean(gold_lengths):.1f}, min={np.min(gold_lengths)}, max={np.max(gold_lengths)}")
    print(f"Average Coverage: {np.mean(coverages)*100:.1f}%")
    print(f"Average Lead Bias (first 20% of doc): {np.mean(lead_biases)*100:.1f}%")



#####################################
#    getting the longsumm splits
#####################################

#!/usr/bin/env python3
"""
prepare_longsumm_splits_with_titles.py
-------------------------------------
• Stratified (by summary-length folder) 72/9/9/10 split for LongSumm.
• Downloads PDFs for the 10 % held-out set.
• Extracts paper titles from held-out PDFs and logs success/failure.
"""

def main(args=None):
    if args is None:
        args = parse_args()
    random.seed(args.seed)

    longsumm_root = Path(args.longsumm_root).resolve()

    # ------------------------------------------
    # AUTO-CLONE LONGSUMM IF FOLDER DOES NOT EXIST
    # ------------------------------------------
    import subprocess
    if not longsumm_root.exists():
        print(f"[INFO] LongSumm directory not found. Cloning repository into {longsumm_root} ...")
        ensure_dir(longsumm_root.parent)
        subprocess.run([
            "git", "clone",
            "https://github.com/WING-NUS/LongSumm",
            str(longsumm_root)
        ], check=True)
    else:
        print(f"[INFO] LongSumm repo already exists at: {longsumm_root}")
    # ------------------------------------------

    abstr_dir = longsumm_root / "abstractive_summaries" / "by_clusters"
    out_dir = Path(args.out_dir).resolve()
    splits_dir = out_dir / "splits"
    pdf_dir = out_dir / "heldout_pdfs"

    ensure_dir(out_dir)
    ensure_dir(splits_dir)
    ensure_dir(pdf_dir)

    logging.basicConfig(filename=str(out_dir / "prepare.log"), level=logging.INFO)

    if not abstr_dir.exists():
        print(f"[ERROR] Missing {abstr_dir}")
        sys.exit(1)

    # ----- Load records -----
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

    for k in bin_to_recs:
        random.shuffle(bin_to_recs[k])

    all_bins = sorted(bin_to_recs.keys())
    n_per_bin = {k: len(bin_to_recs[k]) for k in all_bins}
    N = sum(n_per_bin.values())
    print(f"[INFO] {N} total abstractive entries across {len(all_bins)} bins")

    # ----- Split ratios -----
    target_held = round(N * 0.10)
    remaining = N - target_held
    target_train = round(remaining * 0.80)
    target_val = round(remaining * 0.10)
    target_test = remaining - target_train - target_val

    held_alloc = largest_remainder_alloc(target_held, [n_per_bin[k]*0.10 for k in all_bins])
    per_bin_held = dict(zip(all_bins, held_alloc))

    # ----- Make splits -----
    splits = {"train": [], "val": [], "test": [], "heldout_pipeline": []}
    for k in all_bins:
        recs = bin_to_recs[k]
        h = per_bin_held[k]
        m = len(recs) - h
        t = int(round(m * 0.8))
        v = int(round(m * 0.1))
        te = m - t - v

        start = 0
        splits["heldout_pipeline"].extend(recs[start:start+h]); start += h
        splits["train"].extend(recs[start:start+t]); start += t
        splits["val"].extend(recs[start:start+v]); start += v
        splits["test"].extend(recs[start:start+te])

    def rec_to_jsonl(r: Rec):
        return {
            "id": r.id,
            "bin": r.bin_name,
            "summary_sentences": r.summary_sentences,
            "summary_text": r.summary_text,
            "pdf_url": r.pdf_url,
            "source_path": r.path
        }

    for name, rows in splits.items():
        outp = splits_dir / f"{name}.jsonl"
        with outp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(rec_to_jsonl(r), ensure_ascii=False) + "\n")
        print(f"[WRITE] {name}: {len(rows)}")

    # ----- Held-out PDF download + title extraction -----
    manifest = out_dir / "heldout_pdf_titles_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["id","bin","pdf_url","pdf_path","title","normalized_title","status"])
        ok_dl = miss_dl = ok_title = miss_title = 0

        for r in splits["heldout_pipeline"]:
            pdf_url = guess_pdf_url(r.pdf_url)
            if not pdf_url:
                w.writerow([r.id, r.bin_name, "", "", "", "", "no_pdf_url"])
                miss_dl += 1
                continue

            dest = pdf_dir / f"{r.id}.pdf"
            if not dest.exists():
                if not http_download(pdf_url, dest, args.timeout, args.user_agent):
                    w.writerow([r.id, r.bin_name, pdf_url, "", "", "", "download_failed"])
                    miss_dl += 1
                    continue
                ok_dl += 1

            title = extract_title_from_pdf(dest)
            norm = normalize_title(title) if title else ""
            status = "title_ok" if title else "title_missing"
            if title:
                ok_title += 1
            else:
                miss_title += 1

            w.writerow([r.id, r.bin_name, pdf_url, str(dest), title, norm, status])

    print(f"[PDF] ok={ok_dl} fail={miss_dl} | [TITLE] ok={ok_title} missing={miss_title}")
    print(f"[OUT] CSV → {manifest}")
    print(f"[DONE] Splits + held-out titles ready in {out_dir}")



if __name__ == "__main__":
    import types
    args = types.SimpleNamespace(
        longsumm_root="./Abstraction/longsumm_dataset_scibart(abstractive)/LongSumm-master-github",
        out_dir="./Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared",
        seed=33,
        timeout=60,
        user_agent="Mozilla/5.0"
    )

    main(args)    





#############################################################################################
#. putting academic papers from longsumm dataset into bertsum to get inputs to train scibart
#############################################################################################

"""# extracting summaries for scibart input"""

# Commented out IPython magic to ensure Python compatibility.

import nltk
nltk.data.path.append("/usr/local/share/nltk_data")
nltk.download('punkt')
nltk.download('punkt_tab')

# ============================================================
# Convert processed JSONL files into HF DatasetDict
# ============================================================
from datasets import Dataset, DatasetDict
import json
from pathlib import Path

# Paths to your processed JSONL files in Drive
PROCESSED_DIR = Path("./Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared/splits")
train_file = PROCESSED_DIR / "train.jsonl"
val_file   = PROCESSED_DIR / "val.jsonl"
test_file  = PROCESSED_DIR / "test.jsonl"

def load_jsonl(file_path):
    """Load JSONL file into list of dicts"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

# Load each split
train_data = load_jsonl(train_file)
val_data   = load_jsonl(val_file)
test_data  = load_jsonl(test_file)

# Check the keys in your first example to see what columns you have
print("Sample keys:", train_data[0].keys())
# Should include: 'src', 'tgt', 'src_sentences', 'filename'

# Convert lists to HF Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset   = Dataset.from_list(val_data)
test_dataset  = Dataset.from_list(test_data)

# Create DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Save DatasetDict to disk (so run_extsum.py can read it)
OUTPUT_PATH = "./Abstraction/experiments/ext_from_bert/data_hf"
dataset_dict.save_to_disk(OUTPUT_PATH)
print(f"DatasetDict saved to: {OUTPUT_PATH}")

# =============================================
# Extract summaries from DatasetDict using BertSum
# =============================================
import sys, os, json
sys.path.append("./Extraction/bertsum-hf")

import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from src.bertsum import BertSummarizer, BertSummarizerConfig
from tqdm import tqdm
import numpy as np

# -------------------------------
# Config
# -------------------------------
checkpoint_dir = "./Extraction/model_finetuned_v2/bertsum"
dataset_path  = "./Abstraction/experiments/ext_from_bert/data_hf"
output_dir    = "./Abstraction/experiments/ext_from_bert/summaries"
MAX_TOKENS = 512
TOP_K = 10

os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Load model + tokenizer
# -------------------------------
tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
config = BertSummarizerConfig.from_pretrained(checkpoint_dir)
model = BertSummarizer.from_pretrained(checkpoint_dir, config=config)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------
# Load DatasetDict
# -------------------------------
dataset_dict = load_from_disk(dataset_path)

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_for_bertsum(item):
    sentences = item['summary_sentences']
    input_ids, cls_ids = [], []

    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        if len(tokens) == 0:
            continue
        if len(input_ids) + len(tokens) + 1 > MAX_TOKENS:
            break
        cls_ids.append(len(input_ids))
        input_ids.extend([tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens))

    mask = [1] * len(input_ids)
    mask_cls = [1] * len(cls_ids)

    return {
        'src': sentences,
        'filename': item.get('filename', None),
        'input_ids': torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device),
        'cls_ids': torch.tensor(cls_ids, dtype=torch.long).unsqueeze(0).to(device),
        'mask': torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(device),
        'mask_cls': torch.tensor(mask_cls, dtype=torch.long).unsqueeze(0).to(device)
    }

# -------------------------------
# Extract summaries for all splits
# -------------------------------
for split, dataset in dataset_dict.items():
    print(f"\nProcessing split: {split} ({len(dataset)} documents)")

    all_summaries = []

    for i, item in enumerate(tqdm(dataset, desc=f"{split} progress")):
        processed = preprocess_for_bertsum(item)

        with torch.no_grad():
            outputs = model(
                processed['input_ids'],
                cls_ids=processed['cls_ids'],
                mask=processed['mask'],
                mask_cls=processed['mask_cls']
            )

        # Ensure scores is always a 1D array
        scores = outputs['logits'].squeeze().cpu().numpy()
        if np.ndim(scores) == 0:
            scores = np.array([scores])

        top_k = min(TOP_K, len(scores))
        top_indices = scores.argsort()[-top_k:][::-1]

        summary_sentences = [item['summary_sentences'][j] for j in sorted(top_indices)]
        summary = " ".join(summary_sentences)
        all_summaries.append(summary)

    # Save split summaries
    output_file = os.path.join(output_dir, f"summaries_{split}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for i, summary in enumerate(all_summaries):
            record = {
                "id": dataset[i]["id"],
                "summary": summary
            }
            f.write(json.dumps(record) + "\n")

    print(f" Saved {split} summaries to: {output_file}")










######################################################
#                  ABSTRACTION STAGE
#                    OF PIPELINE
######################################################

# -*- coding: utf-8 -*-
"""Abstractionv2.ipynb

### **1 Environment & Configuration**
Set up everything needed to train in Google Colab
"""





# --- Imports ---
import os, json, gc, math, random, time, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, TrainerCallback
)
import evaluate, textstat
from bert_score import score as bertscore

# --- Global configuration ---
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PROJECT_DIR = "./" # Changed to the shortcut path
OUT_DIR     = f"{PROJECT_DIR}/Abstraction/experiments"

# v2 folder to separate from older runs
ABST_DIR = f"{OUT_DIR}/abstract_v2"
FT_DIR   = f"{ABST_DIR}/finetuned"
Path(ABST_DIR).mkdir(parents=True, exist_ok=True)
Path(FT_DIR).mkdir(parents=True, exist_ok=True)

# Paths to the already-paired JSONLs created from LongSumm×BERTSum v2
TRAIN_JSONL = f"{ABST_DIR}/abst_train.jsonl"
VAL_JSONL   = f"{ABST_DIR}/abst_val.jsonl"
TEST_JSONL  = f"{ABST_DIR}/abst_test.jsonl"

SCIBART_NAME = "uclanlp/scibart-base"
MAX_INPUT, MAX_TARGET = 1024, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


import torch
print(torch.cuda.is_available())

import os


"""### **2 Pair LongSumm Splits with BERTSum-Extractive Outputs**
Align extractive inputs (BERTSum summaries) with LongSumm gold abstracts
"""

# --- Paths for pairing ---
SPLIT_DIR  = f"{PROJECT_DIR}/Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared/splits"
EXTV2_DIR  = f"{OUT_DIR}/ext_from_bert/summaries"   # fine-tuned BERTSum summaries

# --- Helper: load a JSONL file into dict keyed by paper_id/id ---
def _load_jsonl_to_map(path, key_candidates=("paper_id", "id")):
    data_map, key_used = {}, None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if key_used is None:
                for k in key_candidates:
                    if k in row:
                        key_used = k; break
                if key_used is None:
                    raise ValueError(f"No id key found in {path}")
            data_map[str(row[key_used])] = row
    return data_map

# --- Extractive-text getter (handles different field names) ---
def _get_extractive_text(row):
    for cand in ("extractive", "summary", "pred", "prediction", "extract", "text"):
        if cand in row and isinstance(row[cand], str) and row[cand].strip():
            return row[cand].strip()
    for cand in ("sentences", "summary_sentences"):
        if cand in row and isinstance(row[cand], list) and row[cand]:
            return " ".join(s.strip() for s in row[cand] if isinstance(s, str))
    return None

# --- Reference summary getter from LongSumm ---
def _get_reference_summary(item):
    if "summary_text" in item and item["summary_text"].strip():
        return item["summary_text"].strip()
    if "summary_sentences" in item and item["summary_sentences"]:
        return " ".join(s.strip() for s in item["summary_sentences"] if isinstance(s, str))
    return None

# --- Main pairing function ---
def pair_longsumm_with_extv2(split_jsonl, ext_jsonl, out_jsonl):
    ls_map = _load_jsonl_to_map(split_jsonl)
    ext_map = _load_jsonl_to_map(ext_jsonl)

    paired, missing_ext, missing_ref = [], 0, 0
    for pid, item in ls_map.items():
        ref = _get_reference_summary(item)
        if not ref:
            missing_ref += 1; continue
        ext_row = ext_map.get(str(pid))
        if not ext_row:
            missing_ext += 1; continue
        inp = _get_extractive_text(ext_row)
        if not inp:
            missing_ext += 1; continue

        paired.append({
            "paper_id": pid,
            "input": " ".join(inp.split()),       # BERTSum output
            "summary": " ".join(ref.split())      # LongSumm gold summary
        })

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for p in paired:
            f.write(json.dumps(p) + "\n")

    print(f"Paired {len(paired)} → {Path(out_jsonl).name} | "
          f"missing_ext={missing_ext}, missing_ref={missing_ref}")

pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/train.jsonl",
    f"{EXTV2_DIR}/summaries_train.jsonl",
    TRAIN_JSONL
)
pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/val.jsonl",
    f"{EXTV2_DIR}/summaries_validation.jsonl",
    VAL_JSONL
)
pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/test.jsonl",
    f"{EXTV2_DIR}/summaries_test.jsonl",
    TEST_JSONL
)


"""### **3 Load Dataset & Tokenize (SciBART Tokenizer, Dynamic Padding)**"""

from transformers import BartTokenizer, AutoTokenizer # Ensure BartTokenizer is imported

try:
    # Try to load SciBART tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(SCIBART_NAME, use_fast=False, trust_remote_code=True)
    print("Loaded SciBART tokenizer from hub.")
except Exception as e:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Ensure special tokens exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad/eos/bos IDs:", tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id)

data_files = {"train": TRAIN_JSONL, "validation": VAL_JSONL, "test": TEST_JSONL}
ds = load_dataset("json", data_files=data_files)

def preprocess(batch):
    enc = tokenizer(batch["input"], truncation=True, max_length=MAX_INPUT)
    # Updated: Replaced deprecated `as_target_tokenizer()` with `text_target`
    dec = tokenizer(text_target=batch["summary"], truncation=True, max_length=MAX_TARGET)
    enc["labels"] = dec["input_ids"]
    return enc

tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)
print("Tokenizer vocab size:", len(tokenizer))
print("Model embedding size:", model.get_input_embeddings().num_embeddings)

# Resize if needed
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))
    print("Resized embeddings to match tokenizer size.")

# Initialize data collator *after* model resizing with the actual model instance
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

import json
with open(TRAIN_JSONL) as f:
    print("Sample from paired dataset:\n", json.loads(next(f)))

print(tokenized["train"].features)

"""### **4 Baseline Zero-Shot SciBART Evaluation**

This section evaluates the pretrained SciBART model without any fine-tuning.

By running SciBART directly on the test set, we establish a baseline performance that represents how well the model generalises to scientific summarisation. This baseline will later be compared against the fine-tuned version to measure improvement.
"""

def load_scibart():
    model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)

    # --- ensure vocab alignment ---
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        print(f"Resizing model embeddings from {model.get_input_embeddings().num_embeddings} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # --- critical: special token alignment ---
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id  #

    # --- optional: safer generation defaults ---
    model.generation_config.num_beams = 4
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3

    model.eval()
    return model

@torch.no_grad()
def generate_batches(model, texts, batch_size=4, max_length=512, num_beams=4):
    """
    Generates summaries for a list of input texts using the given model.
    Ensures that the number of outputs matches the number of inputs.
    """
    model.eval()
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        outputs = model.generate(
            **enc,
            num_beams=num_beams,
            max_length=max_length,
            min_length=30,
            no_repeat_ngram_size=3,
            repetition_penalty=2.5,
            early_stopping=True
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Append results batch by batch
        all_preds.extend(decoded)

    print(f"Generated {len(all_preds)} predictions for {len(texts)} inputs")
    return all_preds

# --- Run baseline on test split ---
model_zero = load_scibart()
test_inputs, test_refs = ds["test"]["input"], ds["test"]["summary"]
baseline_preds = generate_batches(model_zero, test_inputs)

print("Test inputs:", len(test_inputs))
print("Baseline preds:", len(baseline_preds))
print("Test refs:", len(test_refs))

rouge = evaluate.load("rouge")
rouge_res = rouge.compute(predictions=baseline_preds, references=test_refs)
P,R,F1 = bertscore(baseline_preds, test_refs, lang="en", rescale_with_baseline=True)
fk = [textstat.flesch_kincaid_grade(p) for p in baseline_preds]
baseline_metrics = {
    **rouge_res,
    "bertscore_p": float(P.mean()),
    "bertscore_r": float(R.mean()),
    "bertscore_f1": float(F1.mean()),
    "fk_grade_mean": float(np.mean(fk))
}
json.dump(baseline_metrics, open(f"{ABST_DIR}/baseline_metrics.json", "w"), indent=2)
print("Baseline metrics:", baseline_metrics)

"""### **5 Fine-Tuning Configuration, Hyperparameter Saving & Per-Epoch Logging**"""

# Save hyperparameters for reproducibility
hparams = {
    "learning_rate": 2e-4, "batch_size": 2, "accumulation": 8,
    "epochs": 15, "max_input": MAX_INPUT, "max_target": MAX_TARGET,
    "model": SCIBART_NAME, "seed": SEED
}
json.dump(hparams, open(f"{FT_DIR}/hyperparameters.json", "w"), indent=2)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME)
model.config.pad_token_id = tokenizer.pad_token_id

# Per-epoch logger callback
class EpochLogger(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n=== Epoch {int(state.epoch)} completed ===")
        preview_prediction()

# Metric computation
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge = evaluate.load("rouge")
    # ROUGE (includes rogue1, 2, roguerougeL, rogueLsum)
    rouge_res = rouge.compute(predictions=preds_text, references=labels_text, use_stemmer=True)

    # BERTScore (returns tensors)
    P, R, F1 = bertscore(preds_text, labels_text, lang="en", rescale_with_baseline=True)
    bert_p   = float(P.mean())
    bert_r   = float(R.mean())
    bert_f1  = float(F1.mean())

    # Readability
    fk = [textstat.flesch_kincaid_grade(p) for p in preds_text]
    fk_mean = float(np.mean(fk)) if fk else 0.0

    return {
        "rouge1": rouge_res.get("rouge1", 0.0),
        "rouge2": rouge_res.get("rouge2", 0.0),
        "rougeL": rouge_res.get("rougeL", 0.0),
        "rougeLsum": rouge_res.get("rougeLsum", 0.0),
        "bertscore_p": bert_p,
        "bertscore_r": bert_r,
        "bertscore_f1": bert_f1,
        "fk_grade_mean": fk_mean
    }

from transformers import AutoModelForSeq2SeqLM

# Fresh load on CPU first
model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)

# Realign tokenizer & model
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

# Move cleanly to GPU
model.to("cuda" if torch.cuda.is_available() else "cpu")


from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding="longest"
)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer # Import Seq2SeqTrainingArguments and Seq2SeqTrainer

# --- Generation config fix ---
model.generation_config.num_beams = 4
model.generation_config.early_stopping = True

args = Seq2SeqTrainingArguments(
    output_dir=FT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=hparams["learning_rate"],
    per_device_train_batch_size=hparams["batch_size"],
    per_device_eval_batch_size=hparams["batch_size"],
    gradient_accumulation_steps=hparams["accumulation"],
    num_train_epochs=hparams["epochs"],
    warmup_ratio=0.06,
    weight_decay=0.01,
    predict_with_generate=True, # This parameter is valid for Seq2SeqTrainingArguments
    generation_max_length=MAX_TARGET,
    seed=SEED,
    report_to=[]
)

# Place this ABOVE the Seq2SeqTrainer block, after model/tokenizer are loaded
def preview_prediction():
    sample = ds["validation"][0]["input"]
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    outputs = model.generate(**inputs, max_length=256, num_beams=4)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n[Preview summary sample]\n", pred[:300], "\n")

# Optional: call once before training starts
preview_prediction()

trainer = Seq2SeqTrainer( # Use Seq2SeqTrainer instead of Trainer
    model=model,
    args=args,
    processing_class=tokenizer,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EpochLogger()]
)

train_output = trainer.train()
trainer.save_model(f"{FT_DIR}/checkpoint_final")

"""### **6 Qualitative Validation**"""

import random

def show_samples(split="validation", n=3):
    exs = random.sample(list(zip(ds[split]["input"], ds[split]["summary"])), n)
    for i, (inp, ref) in enumerate(exs, 1):
        gen = trainer.model.generate(
            **tokenizer(inp, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(DEVICE),
            max_length=MAX_TARGET, num_beams=4
        )
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"\n=== Example {i} ({split}) ===")
        print(f"[Input excerpt] {inp[:300]}…\n")
        print(f"[Reference] {ref}\n")
        print(f"[Generated] {pred}\n")

show_samples("validation", n=3)

"""### **7 Chunked Generation with Intermediate Saves**"""

def generate_in_chunks(texts, model, tokenizer, chunk_size=50, save_path=None):
    preds_all = []
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        padding=True, max_length=MAX_INPUT).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_length=MAX_TARGET, num_beams=4,
                                 early_stopping=True, no_repeat_ngram_size=3)
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds_all.extend(preds)
        # Incremental save
        with open(save_path, "a") as f:
            for t, p in zip(chunk, preds):
                f.write(json.dumps({"input_excerpt": t[:200], "prediction": p}) + "\n")
        print(f"Chunk {i//chunk_size+1} ({len(preds_all)}/{len(texts)}) done")
        gc.collect()
    return preds_all

val_preds = generate_in_chunks(
    ds["validation"]["input"],
    trainer.model,
    tokenizer,
    chunk_size=20,
    save_path=f"{ABST_DIR}/val_preds_chunks.jsonl"
)

"""### **8 Final Test Evaluation, Metrics, and Examples**"""

test_inputs, test_refs = ds["test"]["input"], ds["test"]["summary"]
test_preds = generate_in_chunks(
    test_inputs,
    trainer.model,
    tokenizer,
    chunk_size=20,
    save_path=f"{ABST_DIR}/test_preds_chunks.jsonl"
)

rouge = evaluate.load("rouge")
rouge_res = rouge.compute(predictions=test_preds, references=test_refs, use_stemmer=True)

P,R,F1  = bertscore(test_preds, test_refs, lang="en", rescale_with_baseline=True)
bert_p   = float(P.mean())
bert_r   = float(R.mean())
bert_f1  = float(F1.mean())

fk      = [textstat.flesch_kincaid_grade(t) for t in test_preds]
fk_mean = float(np.mean(fk)) if fk else 0.0

final_metrics = {
    "rouge1": rouge_res.get("rouge1", 0.0),
    "rouge2": rouge_res.get("rouge2", 0.0),
    "rougeL": rouge_res.get("rougeL", 0.0),
    "rougeLsum": rouge_res.get("rougeLsum", 0.0),
    "bertscore_p": bert_p,
    "bertscore_r": bert_r,
    "bertscore_f1": bert_f1,
    "fk_grade_mean": fk_mean
}
json.dump(final_metrics, open(f"{ABST_DIR}/final_test_metrics.json", "w"), indent=2)
print(" Final test metrics:", final_metrics)
for k, v in final_metrics.items():
    print(f"  {k:<15}: {v:.4f}")

# Qualitative test samples
for i in range(3):
    print(f"\n=== TEST EXAMPLE {i+1} ===")
    print("[Reference]:", test_refs[i][:400], "\n")
    print("[Generated ]:", test_preds[i][:400], "\n")

"""##

---
# **NEXT RUN: SciBART with LoRA and slightly adjusted hyperparameters(20 epoch)**
"""





# --- Imports ---
import os, json, gc, math, random, time, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, TrainerCallback
)
import evaluate, textstat
from bert_score import score as bertscore
rouge = evaluate.load("rouge")

# --- Global configuration ---
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PROJECT_DIR = "./" # Changed to the shortcut path

ABST_DIR = "./Abstraction/experiments/abstract_v2"
OUT_DIR = f"{ABST_DIR}/runs_withLoRA(usingbetterextsummaries)_v2"
FT_DIR = f"{OUT_DIR}/finetuned_scibart_withLoRA(usingbetterextsummaries)_v2"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FT_DIR, exist_ok=True)

# Paths to the already-paired JSONLs created from LongSumm×BERTSum v2
TRAIN_JSONL = f"{ABST_DIR}/abst_train.jsonl"
VAL_JSONL   = f"{ABST_DIR}/abst_val.jsonl"
TEST_JSONL  = f"{ABST_DIR}/abst_test.jsonl"

SCIBART_NAME = "uclanlp/scibart-base"
MAX_INPUT, MAX_TARGET = 1024, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""**pair longsumm splits with better ext summaries**


"""

# --- Paths for pairing ---
SPLIT_DIR  = f"{PROJECT_DIR}/Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared/splits"
EXTV2_DIR = f"{PROJECT_DIR}/Abstraction/experiments/ext_from_bert/summaries"  # fine-tuned BERTSum summaries

# --- Helper: load a JSONL file into dict keyed by paper_id/id ---
def _load_jsonl_to_map(path, key_candidates=("paper_id", "id")):
    data_map, key_used = {}, None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if key_used is None:
                for k in key_candidates:
                    if k in row:
                        key_used = k; break
                if key_used is None:
                    raise ValueError(f"No id key found in {path}")
            data_map[str(row[key_used])] = row
    return data_map

# --- Extractive-text getter (handles different field names) ---
def _get_extractive_text(row):
    for cand in ("extractive", "summary", "pred", "prediction", "extract", "text"):
        if cand in row and isinstance(row[cand], str) and row[cand].strip():
            return row[cand].strip()
    for cand in ("sentences", "summary_sentences"):
        if cand in row and isinstance(row[cand], list) and row[cand]:
            return " ".join(s.strip() for s in row[cand] if isinstance(s, str))
    return None

# --- Reference summary getter from LongSumm ---
def _get_reference_summary(item):
    if "summary_text" in item and item["summary_text"].strip():
        return item["summary_text"].strip()
    if "summary_sentences" in item and item["summary_sentences"]:
        return " ".join(s.strip() for s in item["summary_sentences"] if isinstance(s, str))
    return None

# --- Main pairing function ---
def pair_longsumm_with_extv2(split_jsonl, ext_jsonl, out_jsonl):
    ls_map = _load_jsonl_to_map(split_jsonl)
    ext_map = _load_jsonl_to_map(ext_jsonl)

    paired, missing_ext, missing_ref = [], 0, 0
    for pid, item in ls_map.items():
        ref = _get_reference_summary(item)
        if not ref:
            missing_ref += 1; continue
        ext_row = ext_map.get(str(pid))
        if not ext_row:
            missing_ext += 1; continue
        inp = _get_extractive_text(ext_row)
        if not inp:
            missing_ext += 1; continue

        paired.append({
            "paper_id": pid,
            "input": " ".join(inp.split()),       # BERTSum output
            "summary": " ".join(ref.split())      # LongSumm gold summary
        })

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for p in paired:
            f.write(json.dumps(p) + "\n")

    print(f" Paired {len(paired)} → {Path(out_jsonl).name} | "
          f"missing_ext={missing_ext}, missing_ref={missing_ref}")

pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/train.jsonl",
    f"{EXTV2_DIR}/summaries_train.jsonl",
    TRAIN_JSONL
)
pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/val.jsonl",
    f"{EXTV2_DIR}/summaries_validation.jsonl",
    VAL_JSONL
)
pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/test.jsonl",
    f"{EXTV2_DIR}/summaries_test.jsonl",
    TEST_JSONL
)

"""### **loading tokenizer**"""

#3 Load Dataset & Tokenize (SciBART Tokenizer, Dynamic Padding)
from transformers import BartTokenizer, AutoTokenizer # Ensure BartTokenizer is imported

try:
    # Try to load SciBART tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(SCIBART_NAME, use_fast=False, trust_remote_code=True)
    print("Loaded SciBART tokenizer from hub.")
except Exception as e:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Ensure special tokens exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad/eos/bos IDs:", tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id)

"""**loading dataset**"""

data_files = {"train": TRAIN_JSONL, "validation": VAL_JSONL, "test": TEST_JSONL}
ds = load_dataset("json", data_files=data_files)

def preprocess(batch):
    enc = tokenizer(batch["input"], truncation=True, max_length=MAX_INPUT)
    dec = tokenizer(text_target=batch["summary"], truncation=True, max_length=MAX_TARGET)

    enc["labels"] = dec["input_ids"]
    return enc

tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

tokenized

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)
print("Tokenizer vocab size:", len(tokenizer))
print("Model embedding size:", model.get_input_embeddings().num_embeddings)

# Resize if needed
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))
    print(" Resized embeddings to match tokenizer size.")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

import json
with open(f"{ABST_DIR}/abst_train.jsonl") as f:
    print("Sample from paired dataset:\n", json.loads(next(f)))

print(tokenized["train"].features)

"""**5 Fine-Tuning Configuration, Hyperparameter Saving & Per-Epoch Logging**

"""

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Save hyperparameters for reproducibility
hparams = {
    "learning_rate": 2e-4, "batch_size": 2, "accumulation": 8,
    "epochs": 20, "max_input": MAX_INPUT, "max_target": MAX_TARGET,
    "model": SCIBART_NAME, "seed": SEED
}
json.dump(hparams, open(f"{FT_DIR}/hyperparameters.json", "w"), indent=2)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)
from peft import LoraConfig, get_peft_model

# === LoRA configuration ===
lora_cfg = LoraConfig(
    r=16,                   # rank dimension
    lora_alpha=64,          # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # attention layers to tune
    lora_dropout=0.1,       # dropout for LoRA layers
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_cfg)

# Re-verify and fix embedding alignment
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))
    print(" Resized embeddings post-LoRA to match tokenizer size.")

# Re-set special token IDs
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id

print(len(tokenizer), model.get_input_embeddings().num_embeddings)


model.print_trainable_parameters()  # should show ≈1–2% trainable params
print("\n Sanity check — verifying tokenizer & embedding alignment...")
print("Tokenizer vocab size:", len(tokenizer))
print("Model embedding size:", model.get_input_embeddings().num_embeddings)
print(" Ready for LoRA fine-tuning!")
model.config.pad_token_id = tokenizer.pad_token_id

# Per-epoch logger callback
class EpochLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Epoch {state.epoch:.2f} | Step {state.global_step} | Loss {logs['loss']:.4f}")

# Metric computation
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Replace -100 with pad for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    preds_text  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(labels,  skip_special_tokens=True)

    # ROUGE (includes rogue1, 2, roguerougeL, rogueLsum)
    rouge_res = rouge.compute(predictions=preds_text, references=labels_text, use_stemmer=True)

    # BERTScore (returns tensors)
    P, R, F1 = bertscore(preds_text, labels_text, lang="en", rescale_with_baseline=True)
    bert_p   = float(P.mean())
    bert_r   = float(R.mean())
    bert_f1  = float(F1.mean())

    # Readability
    fk = [textstat.flesch_kincaid_grade(p) for p in preds_text]
    fk_mean = float(np.mean(fk)) if fk else 0.0

    return {
        "rouge1": rouge_res.get("rouge1", 0.0),
        "rouge2": rouge_res.get("rouge2", 0.0),
        "rougeL": rouge_res.get("rougeL", 0.0),
        "rougeLsum": rouge_res.get("rougeLsum", 0.0),
        "bertscore_p": bert_p,
        "bertscore_r": bert_r,
        "bertscore_f1": bert_f1,
        "fk_grade_mean": fk_mean
    }

args = Seq2SeqTrainingArguments(
    output_dir=FT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=hparams["learning_rate"],
    per_device_train_batch_size=hparams["batch_size"],
    per_device_eval_batch_size=hparams["batch_size"],
    gradient_accumulation_steps=hparams["accumulation"],
    num_train_epochs=hparams["epochs"],
    warmup_ratio=0.06,
    weight_decay=0.01,
    predict_with_generate=True,           # ✅ now valid
    generation_max_length=MAX_TARGET,
    seed=SEED,
    report_to=[],
    fp16=True
)


trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EpochLogger()]
)
train_output = trainer.train()
trainer.save_model(f"{FT_DIR}/checkpoint_final")

"""**6 Qualitative Validation**

"""

import random

def show_samples(split="validation", n=3):
    exs = random.sample(list(zip(ds[split]["input"], ds[split]["summary"])), n)
    for i, (inp, ref) in enumerate(exs, 1):
        gen = model.generate(
        **tokenizer(inp, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(DEVICE),
        max_length=MAX_TARGET,
        num_beams=6,                  #  more diverse exploration
        early_stopping=True,          #  stops when stable
        no_repeat_ngram_size=3        #  avoids repeated phrases
        )
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"\n=== Example {i} ({split}) ===")
        print(f"[Input excerpt] {inp[:300]}…\n")
        print(f"[Reference] {ref}\n")
        print(f"[Generated] {pred}\n")

show_samples("validation", n=3)

"""**7 Chunked Generation with Intermediate Saves**

"""

def generate_in_chunks(texts, model, tokenizer, chunk_size=50, save_path=None):
    preds_all = []
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        padding=True, max_length=MAX_INPUT).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_length=MAX_TARGET, num_beams=4,
                                 early_stopping=True, no_repeat_ngram_size=3)
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds_all.extend(preds)
        # Incremental save
        with open(save_path, "a") as f:
            for t, p in zip(chunk, preds):
                f.write(json.dumps({"input_excerpt": t[:200], "prediction": p}) + "\n")
        print(f" Chunk {i//chunk_size+1} ({len(preds_all)}/{len(texts)}) done")
        gc.collect()
    return preds_all

val_preds = generate_in_chunks(
    ds["validation"]["input"],
    model,
    tokenizer,
    chunk_size=20,
    save_path=f"{ABST_DIR}/val_preds_chunks.jsonl"
)

"""**8 Final Test Evaluation, Metrics, and Examples**"""

Stest_inputs, test_refs = ds["test"]["input"], ds["test"]["summary"]

# Generate predictions in chunks to avoid GPU OOM
test_preds = generate_in_chunks(
    test_inputs,
    model,
    tokenizer,
    chunk_size=20,
    save_path=f"{ABST_DIR}/test_preds_chunks.jsonl"
)

# ---- Compute metrics ----
rouge_res = rouge.compute(predictions=test_preds, references=test_refs, use_stemmer=True)

# BERTScore (returns tensors)
P, R, F1 = bertscore(test_preds, test_refs, lang="en", rescale_with_baseline=True)
bert_p   = float(P.mean())
bert_r   = float(R.mean())
bert_f1  = float(F1.mean())

# Readability (Flesch–Kincaid)
fk = [textstat.flesch_kincaid_grade(t) for t in test_preds]
fk_mean = float(np.mean(fk)) if fk else 0.0

# ---- Combine all metrics ----
final_metrics = {
    "rouge1": rouge_res.get("rouge1", 0.0),
    "rouge2": rouge_res.get("rouge2", 0.0),
    "rougeL": rouge_res.get("rougeL", 0.0),
    "rougeLsum": rouge_res.get("rougeLsum", 0.0),
    "bertscore_p": bert_p,
    "bertscore_r": bert_r,
    "bertscore_f1": bert_f1,
    "fk_grade_mean": fk_mean
}

# Save metrics to file
json.dump(final_metrics, open(f"{OUT_DIR}/final_test_metrics.json", "w"), indent=2)

print(" Final test metrics:")
for k, v in final_metrics.items():
    print(f"  {k:<15}: {v:.4f}")

# ---- Show qualitative samples ----
for i in range(3):
    print(f"\n=== TEST EXAMPLE {i+1} ===")
    print("[Reference]:", test_refs[i][:400], "\n")
    print("[Generated ]:", test_preds[i][:400], "\n")

"""## **LoRa fine-tuned Scibart run with 5 epochs**

> Did 20 epochs run in the previous run but results were optimal at 5 epochs


"""

# --- Imports ---
import os, json, gc, math, random, time, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, TrainerCallback
)
import evaluate, textstat
from bert_score import score as bertscore
rouge = evaluate.load("rouge")

# --- Global configuration ---
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PROJECT_DIR = "./" # Changed to the shortcut path

ABST_DIR = "./Abstraction/experiments/abstract_v2"
OUT_DIR = f"{ABST_DIR}/runs_withLoRA(5epoch)(usingbetterextsummaries)_v2"
FT_DIR = f"{OUT_DIR}/finetuned_scibart_withLoRA(5epoch)(usingbetterextsummaries)_v2"
 #SORRY FOR THE TERRIBLE NAMING CONV BUT THE ABOVE 2 LINES ARE THE ONLY CHANGES THAT NEED TO BE MADE OTHER THAN NO OF EPOCH FROM PREV RUN
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FT_DIR, exist_ok=True)

# Paths to the already-paired JSONLs created from LongSumm×BERTSum v2
TRAIN_JSONL = f"{ABST_DIR}/abst_train.jsonl"
VAL_JSONL   = f"{ABST_DIR}/abst_val.jsonl"
TEST_JSONL  = f"{ABST_DIR}/abst_test.jsonl"

SCIBART_NAME = "uclanlp/scibart-base"
MAX_INPUT, MAX_TARGET = 1024, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""**pair longsumm splits with better ext summaries**


"""

# --- Paths for pairing ---
SPLIT_DIR  = f"{PROJECT_DIR}/Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared/splits"
EXTV2_DIR = f"{PROJECT_DIR}/Abstraction/experiments/ext_from_bert/summaries"  # fine-tuned BERTSum summaries

# --- Helper: load a JSONL file into dict keyed by paper_id/id ---
def _load_jsonl_to_map(path, key_candidates=("paper_id", "id")):
    data_map, key_used = {}, None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if key_used is None:
                for k in key_candidates:
                    if k in row:
                        key_used = k; break
                if key_used is None:
                    raise ValueError(f"No id key found in {path}")
            data_map[str(row[key_used])] = row
    return data_map

# --- Extractive-text getter (handles different field names) ---
def _get_extractive_text(row):
    for cand in ("extractive", "summary", "pred", "prediction", "extract", "text"):
        if cand in row and isinstance(row[cand], str) and row[cand].strip():
            return row[cand].strip()
    for cand in ("sentences", "summary_sentences"):
        if cand in row and isinstance(row[cand], list) and row[cand]:
            return " ".join(s.strip() for s in row[cand] if isinstance(s, str))
    return None

# --- Reference summary getter from LongSumm ---
def _get_reference_summary(item):
    if "summary_text" in item and item["summary_text"].strip():
        return item["summary_text"].strip()
    if "summary_sentences" in item and item["summary_sentences"]:
        return " ".join(s.strip() for s in item["summary_sentences"] if isinstance(s, str))
    return None

# --- Main pairing function ---
def pair_longsumm_with_extv2(split_jsonl, ext_jsonl, out_jsonl):
    ls_map = _load_jsonl_to_map(split_jsonl)
    ext_map = _load_jsonl_to_map(ext_jsonl)

    paired, missing_ext, missing_ref = [], 0, 0
    for pid, item in ls_map.items():
        ref = _get_reference_summary(item)
        if not ref:
            missing_ref += 1; continue
        ext_row = ext_map.get(str(pid))
        if not ext_row:
            missing_ext += 1; continue
        inp = _get_extractive_text(ext_row)
        if not inp:
            missing_ext += 1; continue

        paired.append({
            "paper_id": pid,
            "input": " ".join(inp.split()),       # BERTSum output
            "summary": " ".join(ref.split())      # LongSumm gold summary
        })

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for p in paired:
            f.write(json.dumps(p) + "\n")

    print(f" Paired {len(paired)} → {Path(out_jsonl).name} | "
          f"missing_ext={missing_ext}, missing_ref={missing_ref}")

pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/train.jsonl",
    f"{EXTV2_DIR}/summaries_train.jsonl",
    TRAIN_JSONL
)
pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/val.jsonl",
    f"{EXTV2_DIR}/summaries_validation.jsonl",
    VAL_JSONL
)
pair_longsumm_with_extv2(
    f"{SPLIT_DIR}/test.jsonl",
    f"{EXTV2_DIR}/summaries_test.jsonl",
    TEST_JSONL
)

"""### **loading tokenizer**"""

#3 Load Dataset & Tokenize (SciBART Tokenizer, Dynamic Padding)
from transformers import BartTokenizer, AutoTokenizer # Ensure BartTokenizer is imported

try:
    # Try to load SciBART tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(SCIBART_NAME, use_fast=False, trust_remote_code=True)
    print(" Loaded SciBART tokenizer from hub.")
except Exception as e:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Ensure special tokens exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad/eos/bos IDs:", tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id)

"""**loading dataset**"""

data_files = {"train": TRAIN_JSONL, "validation": VAL_JSONL, "test": TEST_JSONL}
ds = load_dataset("json", data_files=data_files)

def preprocess(batch):
    enc = tokenizer(batch["input"], truncation=True, max_length=MAX_INPUT)
    dec = tokenizer(text_target=batch["summary"], truncation=True, max_length=MAX_TARGET)

    enc["labels"] = dec["input_ids"]
    return enc

tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

tokenized

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)
print("Tokenizer vocab size:", len(tokenizer))
print("Model embedding size:", model.get_input_embeddings().num_embeddings)

# Resize if needed
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))
    print(" Resized embeddings to match tokenizer size.")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

import json
with open(f"{ABST_DIR}/abst_train.jsonl") as f:
    print("Sample from paired dataset:\n", json.loads(next(f)))

print(tokenized["train"].features)

"""**5 Fine-Tuning Configuration, Hyperparameter Saving & Per-Epoch Logging**

"""

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Save hyperparameters for reproducibility
hparams = {
    "learning_rate": 2e-4, "batch_size": 2, "accumulation": 8,
    "epochs": 5, "max_input": MAX_INPUT, "max_target": MAX_TARGET,
    "model": SCIBART_NAME, "seed": SEED
}
json.dump(hparams, open(f"{FT_DIR}/hyperparameters.json", "w"), indent=2)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)
from peft import LoraConfig, get_peft_model

# === LoRA configuration ===
lora_cfg = LoraConfig(
    r=16,                   # rank dimension
    lora_alpha=64,          # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # attention layers to tune
    lora_dropout=0.1,       # dropout for LoRA layers
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_cfg)

# Re-verify and fix embedding alignment
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))
    print(" Resized embeddings post-LoRA to match tokenizer size.")

# Re-set special token IDs
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id

print(len(tokenizer), model.get_input_embeddings().num_embeddings)


model.print_trainable_parameters()  # should show ≈1–2% trainable params
print("\n Sanity check — verifying tokenizer & embedding alignment...")
print("Tokenizer vocab size:", len(tokenizer))
print("Model embedding size:", model.get_input_embeddings().num_embeddings)
print(" Ready for LoRA fine-tuning!")
model.config.pad_token_id = tokenizer.pad_token_id

# Per-epoch logger callback
class EpochLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Epoch {state.epoch:.2f} | Step {state.global_step} | Loss {logs['loss']:.4f}")

# Metric computation
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Replace -100 with pad for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    preds_text  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(labels,  skip_special_tokens=True)

    # ROUGE (includes rogue1, 2, roguerougeL, rogueLsum)
    rouge_res = rouge.compute(predictions=preds_text, references=labels_text, use_stemmer=True)

    # BERTScore (returns tensors)
    P, R, F1 = bertscore(preds_text, labels_text, lang="en", rescale_with_baseline=True)
    bert_p   = float(P.mean())
    bert_r   = float(R.mean())
    bert_f1  = float(F1.mean())

    # Readability
    fk = [textstat.flesch_kincaid_grade(p) for p in preds_text]
    fk_mean = float(np.mean(fk)) if fk else 0.0

    return {
        "rouge1": rouge_res.get("rouge1", 0.0),
        "rouge2": rouge_res.get("rouge2", 0.0),
        "rougeL": rouge_res.get("rougeL", 0.0),
        "rougeLsum": rouge_res.get("rougeLsum", 0.0),
        "bertscore_p": bert_p,
        "bertscore_r": bert_r,
        "bertscore_f1": bert_f1,
        "fk_grade_mean": fk_mean
    }

args = Seq2SeqTrainingArguments(
    output_dir=FT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=hparams["learning_rate"],
    per_device_train_batch_size=hparams["batch_size"],
    per_device_eval_batch_size=hparams["batch_size"],
    gradient_accumulation_steps=hparams["accumulation"],
    num_train_epochs=hparams["epochs"],
    warmup_ratio=0.06,
    weight_decay=0.01,
    predict_with_generate=True,           # ✅ now valid
    generation_max_length=MAX_TARGET,
    seed=SEED,
    report_to=[],
    fp16=True
)


trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EpochLogger()]
)
train_output = trainer.train()
trainer.save_model(f"{FT_DIR}/checkpoint_final")

"""**6 Qualitative Validation**

"""

import random

def show_samples(split="validation", n=3):
    exs = random.sample(list(zip(ds[split]["input"], ds[split]["summary"])), n)
    for i, (inp, ref) in enumerate(exs, 1):
        gen = model.generate(
        **tokenizer(inp, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(DEVICE),
        max_length=MAX_TARGET,
        num_beams=6,                  #  more diverse exploration
        early_stopping=True,          #  stops when stable
        no_repeat_ngram_size=3        #  avoids repeated phrases
        )
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"\n=== Example {i} ({split}) ===")
        print(f"[Input excerpt] {inp[:300]}…\n")
        print(f"[Reference] {ref}\n")
        print(f"[Generated] {pred}\n")

show_samples("validation", n=3)

"""**7 Chunked Generation with Intermediate Saves**

"""

def generate_in_chunks(texts, model, tokenizer, chunk_size=50, save_path=None):
    preds_all = []
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        padding=True, max_length=MAX_INPUT).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_length=MAX_TARGET, num_beams=4,
                                 early_stopping=True, no_repeat_ngram_size=3)
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds_all.extend(preds)
        # Incremental save
        with open(save_path, "a") as f:
            for t, p in zip(chunk, preds):
                f.write(json.dumps({"input_excerpt": t[:200], "prediction": p}) + "\n")
        print(f" Chunk {i//chunk_size+1} ({len(preds_all)}/{len(texts)}) done")
        gc.collect()
    return preds_all

val_preds = generate_in_chunks(
    ds["validation"]["input"],
    model,
    tokenizer,
    chunk_size=20,
    save_path=f"{ABST_DIR}/val_preds_chunks.jsonl"
)

"""**8 Final Test Evaluation, Metrics, and Examples**"""

test_inputs, test_refs = ds["test"]["input"], ds["test"]["summary"]

# Generate predictions in chunks to avoid GPU OOM
test_preds = generate_in_chunks(
    test_inputs,
    model,
    tokenizer,
    chunk_size=20,
    save_path=f"{ABST_DIR}/test_preds_chunks.jsonl"
)

# ---- Compute metrics ----
rouge_res = rouge.compute(predictions=test_preds, references=test_refs, use_stemmer=True)

# BERTScore (returns tensors)
P, R, F1 = bertscore(test_preds, test_refs, lang="en", rescale_with_baseline=True)
bert_p   = float(P.mean())
bert_r   = float(R.mean())
bert_f1  = float(F1.mean())

# Readability (Flesch–Kincaid)
fk = [textstat.flesch_kincaid_grade(t) for t in test_preds]
fk_mean = float(np.mean(fk)) if fk else 0.0

# ---- Combine all metrics ----
final_metrics = {
    "rouge1": rouge_res.get("rouge1", 0.0),
    "rouge2": rouge_res.get("rouge2", 0.0),
    "rougeL": rouge_res.get("rougeL", 0.0),
    "rougeLsum": rouge_res.get("rougeLsum", 0.0),
    "bertscore_p": bert_p,
    "bertscore_r": bert_r,
    "bertscore_f1": bert_f1,
    "fk_grade_mean": fk_mean
}

# Save metrics to file
json.dump(final_metrics, open(f"{OUT_DIR}/final_test_metrics.json", "w"), indent=2)

print(" Final test metrics:")
for k, v in final_metrics.items():
    print(f"  {k:<15}: {v:.4f}")

# ---- Show qualitative samples ----
for i in range(3):
    print(f"\n=== TEST EXAMPLE {i+1} ===")
    print("[Input extractive]:", test_inputs[i][:400], "\n")
    print("[Reference]:", test_refs[i][:400], "\n")
    print("[Generated ]:", test_preds[i][:400], "\n")

"""**TESTING lora ft scibart WITH HELDOUT SET'S EXTRACTIVE SUMMARIES - run again**"""

# ================================================================
# 9. HELD-OUT PIPELINE EVALUATION (Final Abstractive Test)
# Your LoRA + SciBART model and tokenizer must ALREADY be loaded.
# ================================================================

import json
import textstat
from bert_score import score as bertscore
import evaluate
rouge = evaluate.load("rouge")

# DIRECTORIES (confirm these are correct)
PIPELINE_EXTRACTIVE = f"{PROJECT_DIR}/Abstraction/experiments/pipeline_summaries_bertsum.jsonl"
PIPELINE_REFERENCE  = f"{PROJECT_DIR}/Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared/splits/heldout_pipeline.jsonl"

OUT_PIPELINE_PREDS    = f"{OUT_DIR}/pipeline_preds.jsonl"
OUT_PIPELINE_METRICS  = f"{OUT_DIR}/pipeline_metrics.json"


# ----------------------------------------------------------
# Load extractive summaries
# ----------------------------------------------------------
def load_pipeline_extractives(path):
    data = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            pid = row["filename"].replace(".pdf", "")  # "123.pdf" → "123"
            data[pid] = row
    return data


# ----------------------------------------------------------
# Load reference abstractive summaries
# ----------------------------------------------------------
def load_pipeline_references(path):
    data = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            pid = str(row["id"])   # LongSumm uses "id"
            data[pid] = row
    return data


ext_map = load_pipeline_extractives(PIPELINE_EXTRACTIVE)
ref_map = load_pipeline_references(PIPELINE_REFERENCE)


# ----------------------------------------------------------
# Align inputs and outputs
# ----------------------------------------------------------
paired = []

for pid, ext_row in ext_map.items():

    if pid not in ref_map:
        continue

    inp = _get_extractive_text(ext_row)
    if not inp:
        continue

    ref = _get_reference_summary(ref_map[pid])
    if not ref:
        continue

    paired.append((pid, inp, ref))

print(f" Loaded {len(paired)} held-out extractive → abstractive test samples.")


pipeline_ids     = [p[0] for p in paired]
pipeline_inputs  = [p[1] for p in paired]
pipeline_refs    = [p[2] for p in paired]


# ----------------------------------------------------------
# Generate abstractive predictions using your LoRA model
# ----------------------------------------------------------
pipeline_preds = generate_in_chunks(
    pipeline_inputs,
    model,
    tokenizer,
    chunk_size=20,
    save_path=OUT_PIPELINE_PREDS
)


# ----------------------------------------------------------
# Compute metrics
# ----------------------------------------------------------
rouge_res = rouge.compute(predictions=pipeline_preds, references=pipeline_refs, use_stemmer=True)

P, R, F1 = bertscore(pipeline_preds, pipeline_refs, lang="en", rescale_with_baseline=True)
bert_p   = float(P.mean())
bert_r   = float(R.mean())
bert_f1  = float(F1.mean())

fk = [textstat.flesch_kincaid_grade(t) for t in pipeline_preds]
fk_mean = float(sum(fk) / len(fk)) if fk else 0.0


# ----------------------------------------------------------
# Save final pipeline metrics
# ----------------------------------------------------------
pipeline_metrics = {
    "rouge1": rouge_res["rouge1"],
    "rouge2": rouge_res["rouge2"],
    "rougeL": rouge_res["rougeL"],
    "rougeLsum": rouge_res["rougeLsum"],
    "bertscore_p": bert_p,
    "bertscore_r": bert_r,
    "bertscore_f1": bert_f1,
    "fk_grade_mean": fk_mean,
    "num_items": len(paired)
}

json.dump(pipeline_metrics, open(OUT_PIPELINE_METRICS, "w"), indent=2)

print("\n=== FINAL HELD-OUT PIPELINE METRICS ===")
for k, v in pipeline_metrics.items():
    print(f"{k:<15}: {v:.4f}")


# ----------------------------------------------------------
# Show 3 qualitative examples
# ----------------------------------------------------------
print("\n=== Sample Held-Out Examples ===")

for i in range(3):
    print(f"\n--- Example {i+1} (Paper {pipeline_ids[i]}) ---")
    print("Extractive:", pipeline_inputs[i][:300], "...\n")
    print("Reference :", pipeline_refs[i][:350], "\n")
    print("Generated :", pipeline_preds[i][:350], "\n")

"""can try another run with different LoRA. configs
change to r=8
lora_alpha=32

# **ABLATION study: Using heldout pdfs directly onto abstractive (lora ft scibart), without running through extraction first**
"""

# ==============================================
# FIXED: FULL-PAPER CHUNKED EVALUATION PIPELINE
# ==============================================

import fitz
import os, json
from bert_score import score as bertscore
import evaluate
rouge = evaluate.load("rouge")

PDF_DIR = "./Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared/heldout_pdfs"
HELDOUT_JSONL = "./Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared/splits/heldout_pipeline.jsonl"

# ----- FIXED LOADER FOR YOUR JSON FORMAT -----
def load_gold_heldout(jsonl_path):
    gold = {}
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            pid = str(row["id"])              # <--- FIXED
            gold[pid] = row["summary_text"]   # <--- FIXED
    return gold

gold_heldout = load_gold_heldout(HELDOUT_JSONL)
print("Loaded gold summaries for", len(gold_heldout), "papers.")


# ======== PDF extraction ========
def load_pdf_text(paper_id):
    pdf_path = f"{PDF_DIR}/{paper_id}.pdf"
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ======== Chunking ========
def chunk_text(text, tokenizer, max_tokens=1024):
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), max_tokens):
        chunk_ids = ids[i:i+max_tokens]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk)
    return chunks


# ======== Summarise a single chunk ========
def summarize_chunk(chunk, model, tokenizer, max_length=256):
    enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                    max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=max_length,
            num_beams=6,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ======== Full PDF summarisation ========
def summarize_full_pdf(paper_id, model, tokenizer):
    text = load_pdf_text(paper_id)
    chunks = chunk_text(text, tokenizer)
    print(f" PDF {paper_id} → {len(chunks)} chunks")

    chunk_summaries = []
    for i, ch in enumerate(chunks, 1):
        print(f" Chunk {i}/{len(chunks)}")
        s = summarize_chunk(ch, model, tokenizer)
        chunk_summaries.append(s)

    combined = "\n".join(chunk_summaries)
    return combined


# ======== Evaluation ========
def evaluate_full_pdf(paper_id, model, tokenizer, gold_map):
    combined = summarize_full_pdf(paper_id, model, tokenizer)
    gold = gold_map[paper_id]

    # ROUGE
    rouge_res = rouge.compute(predictions=[combined], references=[gold], use_stemmer=True)

    # BERTScore
    P, R, F1 = bertscore([combined], [gold], lang="en", rescale_with_baseline=True)

    return {
        "rouge1": rouge_res["rouge1"],
        "rouge2": rouge_res["rouge2"],
        "rougeL": rouge_res["rougeL"],
        "rougeLsum": rouge_res["rougeLsum"],
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean())
    }


# ======== Batch Evaluation ========
results = {}
pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

for pdf in pdf_files:
    pid = pdf.replace(".pdf", "")

    if pid not in gold_heldout:
        print(f" Skipping {pid} — no gold summary")
        continue

    print(f"\n==========================")
    print(f" Evaluating {pid}")
    print(f"==========================")

    metrics = evaluate_full_pdf(pid, model, tokenizer, gold_heldout)
    results[pid] = metrics


# Save metrics
OUT_JSON = "./Abstraction/full_pdf_eval_metrics.json"
json.dump(results, open(OUT_JSON, "w"), indent=2)

print("\n Done! Saved results to:", OUT_JSON)

# ==========================================
# Compute OVERALL AVERAGE METRICS
# ==========================================

import numpy as np

metric_names = [
    "rouge1", "rouge2", "rougeL", "rougeLsum",
    "bertscore_p", "bertscore_r", "bertscore_f1"
]

overall_avg = {}

for m in metric_names:
    vals = [results[pid][m] for pid in results]
    overall_avg[m] = float(np.mean(vals))

print("\n==============================")
print("OVERALL AVERAGE METRICS")
print("==============================")
for k, v in overall_avg.items():
    print(f"{k:15}: {v:.4f}")

# Save averages to file
AVG_JSON = "./Abstraction/full_pdf_eval_average_metrics.json"
json.dump(overall_avg, open(AVG_JSON, "w"), indent=2)

print("\n Saved average metrics to:", AVG_JSON)

"""## **Full Fine-Tuned SciBART with 14 Epochs**

### **1 Environment & Configuration**
Set up everything needed to train in Google Colab
"""

# --- Imports ---
import os, json, gc, math, random, time, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, TrainerCallback
)
import evaluate, textstat
from bert_score import score as bertscore

# --- Global configuration ---
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PROJECT_DIR = "./" # Changed to the shortcut path
OUT_DIR     = f"{PROJECT_DIR}/Abstraction/experiments"

# v2 folder to separate from older runs
ABST_DIR = f"{OUT_DIR}/abstract_v2"
FT_DIR   = f"{ABST_DIR}/finetuned_with14epochs"
Path(ABST_DIR).mkdir(parents=True, exist_ok=True)
Path(FT_DIR).mkdir(parents=True, exist_ok=True)

# Paths to the already-paired JSONLs created from LongSumm×BERTSum v2
TRAIN_JSONL = f"{ABST_DIR}/abst_train.jsonl"
VAL_JSONL   = f"{ABST_DIR}/abst_val.jsonl"
TEST_JSONL  = f"{ABST_DIR}/abst_test.jsonl"

SCIBART_NAME = "uclanlp/scibart-base"
MAX_INPUT, MAX_TARGET = 1024, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""### **3 Load Dataset & Tokenize (SciBART Tokenizer, Dynamic Padding)**"""

from transformers import BartTokenizer, AutoTokenizer # Ensure BartTokenizer is imported

try:
    # Try to load SciBART tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(SCIBART_NAME, use_fast=False, trust_remote_code=True)
    print("Loaded SciBART tokenizer from hub.")
except Exception as e:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Ensure special tokens exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad/eos/bos IDs:", tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id)

data_files = {"train": TRAIN_JSONL, "validation": VAL_JSONL, "test": TEST_JSONL}
ds = load_dataset("json", data_files=data_files)

def preprocess(batch):
    enc = tokenizer(batch["input"], truncation=True, max_length=MAX_INPUT)
    # Updated: Replaced deprecated `as_target_tokenizer()` with `text_target`
    dec = tokenizer(text_target=batch["summary"], truncation=True, max_length=MAX_TARGET)
    enc["labels"] = dec["input_ids"]
    return enc

tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)
print("Tokenizer vocab size:", len(tokenizer))
print("Model embedding size:", model.get_input_embeddings().num_embeddings)

# Resize if needed
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))
    print("Resized embeddings to match tokenizer size.")

# Initialize data collator *after* model resizing with the actual model instance
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

import json
with open(TRAIN_JSONL) as f:
    print("Sample from paired dataset:\n", json.loads(next(f)))

print(tokenized["train"].features)

"""### **5 Fine-Tuning Configuration, Hyperparameter Saving & Per-Epoch Logging**"""

# Save hyperparameters for reproducibility
hparams = {
    "learning_rate": 2e-4, "batch_size": 2, "accumulation": 8,
    "epochs": 14, "max_input": MAX_INPUT, "max_target": MAX_TARGET,
    "model": SCIBART_NAME, "seed": SEED
}
json.dump(hparams, open(f"{FT_DIR}/hyperparameters.json", "w"), indent=2)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME)
model.config.pad_token_id = tokenizer.pad_token_id

# Per-epoch logger callback
class EpochLogger(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n=== Epoch {int(state.epoch)} completed ===")
        preview_prediction()

# Metric computation
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge = evaluate.load("rouge")
    # ROUGE (includes rogue1, 2, roguerougeL, rogueLsum)
    rouge_res = rouge.compute(predictions=preds_text, references=labels_text, use_stemmer=True)

    # BERTScore (returns tensors)
    P, R, F1 = bertscore(preds_text, labels_text, lang="en", rescale_with_baseline=True)
    bert_p   = float(P.mean())
    bert_r   = float(R.mean())
    bert_f1  = float(F1.mean())

    # Readability
    fk = [textstat.flesch_kincaid_grade(p) for p in preds_text]
    fk_mean = float(np.mean(fk)) if fk else 0.0

    return {
        "rouge1": rouge_res.get("rouge1", 0.0),
        "rouge2": rouge_res.get("rouge2", 0.0),
        "rougeL": rouge_res.get("rougeL", 0.0),
        "rougeLsum": rouge_res.get("rougeLsum", 0.0),
        "bertscore_p": bert_p,
        "bertscore_r": bert_r,
        "bertscore_f1": bert_f1,
        "fk_grade_mean": fk_mean
    }

from transformers import AutoModelForSeq2SeqLM

# Fresh load on CPU first
model = AutoModelForSeq2SeqLM.from_pretrained(SCIBART_NAME, trust_remote_code=True)

# Realign tokenizer & model
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

# Move cleanly to GPU
model.to("cuda" if torch.cuda.is_available() else "cpu")


from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding="longest"
)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer # Import Seq2SeqTrainingArguments and Seq2SeqTrainer

# --- Generation config fix ---
model.generation_config.num_beams = 4
model.generation_config.early_stopping = True

args = Seq2SeqTrainingArguments(
    output_dir=FT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=hparams["learning_rate"],
    per_device_train_batch_size=hparams["batch_size"],
    per_device_eval_batch_size=hparams["batch_size"],
    gradient_accumulation_steps=hparams["accumulation"],
    num_train_epochs=hparams["epochs"],
    warmup_ratio=0.06,
    weight_decay=0.01,
    predict_with_generate=True, # This parameter is valid for Seq2SeqTrainingArguments
    generation_max_length=MAX_TARGET,
    seed=SEED,
    report_to=[]
)

# Place this ABOVE the Seq2SeqTrainer block, after model/tokenizer are loaded
def preview_prediction():
    sample = ds["validation"][0]["input"]
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    outputs = model.generate(**inputs, max_length=256, num_beams=4)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n[Preview summary sample]\n", pred[:300], "\n")

# Optional: call once before training starts
preview_prediction()

trainer = Seq2SeqTrainer( # Use Seq2SeqTrainer instead of Trainer
    model=model,
    args=args,
    processing_class=tokenizer,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EpochLogger()]
)

train_output = trainer.train()
trainer.save_model(f"{FT_DIR}/checkpoint_fullft_14epochs")

"""### **6 Qualitative Validation**"""

import random

def show_samples(split="validation", n=3):
    exs = random.sample(list(zip(ds[split]["input"], ds[split]["summary"])), n)
    for i, (inp, ref) in enumerate(exs, 1):
        gen = trainer.model.generate(
            **tokenizer(inp, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(DEVICE),
            max_length=MAX_TARGET, num_beams=4
        )
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"\n=== Example {i} ({split}) ===")
        print(f"[Input excerpt] {inp[:300]}…\n")
        print(f"[Reference] {ref}\n")
        print(f"[Generated] {pred}\n")

show_samples("validation", n=3)

"""### **7 Chunked Generation with Intermediate Saves**"""

def generate_in_chunks(texts, model, tokenizer, chunk_size=50, save_path=None):
    preds_all = []
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        padding=True, max_length=MAX_INPUT).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_length=MAX_TARGET, num_beams=4,
                                 early_stopping=True, no_repeat_ngram_size=3)
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds_all.extend(preds)
        # Incremental save
        with open(save_path, "a") as f:
            for t, p in zip(chunk, preds):
                f.write(json.dumps({"input_excerpt": t[:200], "prediction": p}) + "\n")
        print(f"Chunk {i//chunk_size+1} ({len(preds_all)}/{len(texts)}) done")
        gc.collect()
    return preds_all

val_preds = generate_in_chunks(
    ds["validation"]["input"],
    trainer.model,
    tokenizer,
    chunk_size=20,
    save_path=f"{FT_DIR}/val_preds_chunks.jsonl"
)

"""### **8 Final Test Evaluation, Metrics, and Examples**"""

test_inputs, test_refs = ds["test"]["input"], ds["test"]["summary"]
test_preds = generate_in_chunks(
    test_inputs,
    trainer.model,
    tokenizer,
    chunk_size=20,
    save_path=f"{FT_DIR}/test_preds_chunks.jsonl"
)

rouge = evaluate.load("rouge")
rouge_res = rouge.compute(predictions=test_preds, references=test_refs, use_stemmer=True)

P,R,F1  = bertscore(test_preds, test_refs, lang="en", rescale_with_baseline=True)
bert_p   = float(P.mean())
bert_r   = float(R.mean())
bert_f1  = float(F1.mean())

fk      = [textstat.flesch_kincaid_grade(t) for t in test_preds]
fk_mean = float(np.mean(fk)) if fk else 0.0

final_metrics = {
    "rouge1": rouge_res.get("rouge1", 0.0),
    "rouge2": rouge_res.get("rouge2", 0.0),
    "rougeL": rouge_res.get("rougeL", 0.0),
    "rougeLsum": rouge_res.get("rougeLsum", 0.0),
    "bertscore_p": bert_p,
    "bertscore_r": bert_r,
    "bertscore_f1": bert_f1,
    "fk_grade_mean": fk_mean
}
json.dump(final_metrics, open(f"{FT_DIR}/final_test_metrics.json", "w"), indent=2)
print("Final test metrics:", final_metrics)
for k, v in final_metrics.items():
    print(f"  {k:<15}: {v:.4f}")

# Qualitative test samples
for i in range(3):
    print(f"\n=== TEST EXAMPLE {i+1} ===")
    print("[Reference]:", test_refs[i][:400], "\n")
    print("[Generated ]:", test_preds[i][:400], "\n")





#################################################
#        ABLATION WITH TEXTRANK
#################################################
# run_textrank_test.py

import json
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import os

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ==========================================
# Load JSONL test set
# ==========================================
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

TEST_PATH = ".Extraction/Processed_Data/talksumm_test.jsonl"
data = load_jsonl(TEST_PATH)
print(f"Loaded {len(data)} test documents")

# ==========================================
# TextRank summarization (10 sentences)
# ==========================================
def textrank_sumy(sentences, num_sentences=10):
    text = " ".join(sentences)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])

# ==========================================
# Evaluate using ROUGE
# ==========================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
results = []

for item in tqdm(data, desc="Running TextRank on test set"):
    src_sentences = item["src"]
    reference_summary = item["tgt"]

    generated_summary = textrank_sumy(src_sentences, num_sentences=10)

    scores = scorer.score(reference_summary, generated_summary)
    results.append({
        "filename": item["filename"],
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
        "predicted": generated_summary,
        "reference": reference_summary
    })

# Average ROUGE scores
avg_rouge1 = np.mean([r['rouge1'] for r in results])
avg_rouge2 = np.mean([r['rouge2'] for r in results])
avg_rougeL = np.mean([r['rougeL'] for r in results])

print("\n====== TEXT RANK TEST PERFORMANCE ======")
print(f"Avg ROUGE-1: {avg_rouge1:.4f}")
print(f"Avg ROUGE-2: {avg_rouge2:.4f}")
print(f"Avg ROUGE-L: {avg_rougeL:.4f}")
print("========================================")

# ==========================================
# Save results for abstractive model input
# ==========================================
OUTPUT_DIR = "./Extraction/textrank_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "textrank_test.jsonl")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"✓ Saved TextRank test results -> {OUTPUT_PATH}")




import nltk
nltk.data.path.append("/usr/local/share/nltk_data")
nltk.download('punkt')
nltk.download('punkt_tab')

import os
from PyPDF2 import PdfReader
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# -------- SETTINGS --------
PDF_FOLDER ="./Abstraction/longsumm_dataset_scibart(abstractive)/longsumm_prepared/heldout_pdfs"
OUTPUT_FOLDER = "./Abstraction/experiments/textrank_summaries.jsonl"
NUM_SENTENCES = 10                     # number of sentences in summary
# ---------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def summarize_textrank(text, num_sentences=5):
    """Generate a TextRank summary using Sumy."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)

    return " ".join(str(sentence) for sentence in summary)

# -------- PROCESS ALL PDFs --------
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs.")

for pdf_name in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_name)

    print(f"Processing: {pdf_name}")

    # 1. Extract text
    text = extract_text_from_pdf(pdf_path)
    if len(text.strip()) == 0:
        print(f"WARNING: No text extracted from {pdf_name}")
        continue

    # 2. Summarize using TextRank
    summary = summarize_textrank(text, NUM_SENTENCES)

    # 3. Save summary
    out_path = os.path.join(OUTPUT_FOLDER, pdf_name.replace(".pdf", "_summary.txt"))
    with open(out_path, "w") as f:
        f.write(summary)

print("Done.")




