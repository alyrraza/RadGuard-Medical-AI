"""
CheXbert Runner — AI report text se 14 labels nikalta hai.
Real inference fixed.py se liya gaya, backend ke liye adapt kiya.
"""

import os
import subprocess
import numpy as np
import pandas as pd
import torch

# ── Paths — server pe yeh change karna ──────────────────────
CHEXBERT_DIR  = os.environ.get("CHEXBERT_DIR",  "./CheXbert")
CHEXBERT_CKPT = os.environ.get("CHEXBERT_CKPT", "./CheXbert/src/chexbert.pth")

CHEXBERT_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema',
    'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices', 'No Finding'
]

# ── Keyword fallback jab CheXbert na chale ───────────────────
KEYWORDS = {
    'Enlarged Cardiomediastinum': ['mediastinum','mediastinal','cardiomediastinal'],
    'Cardiomegaly':               ['heart','cardiac','cardiomegaly'],
    'Lung Opacity':               ['opacity','haziness','infiltrate'],
    'Lung Lesion':                ['lesion','nodule','mass','tumor'],
    'Edema':                      ['edema','oedema','congestion'],
    'Consolidation':              ['consolidation'],
    'Pneumonia':                  ['pneumonia'],
    'Atelectasis':                ['atelectasis','collapse'],
    'Pneumothorax':               ['pneumothorax'],
    'Pleural Effusion':           ['effusion','pleural effusion'],
    'Pleural Other':              ['pleural','thickening'],
    'Fracture':                   ['fracture','rib'],
    'Support Devices':            ['tube','catheter','device','line','pacemaker'],
    'No Finding':                 ['normal','no acute','clear','unremarkable'],
}


def patch_chexbert():
    """CheXbert files patch karo — broken imports fix karta hai."""
    if not os.path.exists(CHEXBERT_DIR):
        return

    src = os.path.join(CHEXBERT_DIR, 'src')

    bert_tok = r'''import pandas as pd
from transformers import BertTokenizer
import json
from tqdm import tqdm
import argparse

def get_impressions_from_csv(path):
    df = pd.read_csv(path)
    imp = df['Report Impression']
    imp = imp.str.strip().replace('\n', ' ', regex=True).replace(r'\s+', ' ', regex=True).str.strip()
    return imp

def tokenize(impressions, tokenizer):
    new_impressions = []
    for i in range(impressions.shape[0]):
        tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        if tokenized_imp:
            token_ids = tokenizer.convert_tokens_to_ids(tokenized_imp)
            res = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
            if len(res) > 512:
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions

def load_list(path):
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    impressions = get_impressions_from_csv(args.data)
    new_impressions = tokenize(impressions, tokenizer)
    with open(args.output_path, 'w') as f:
        json.dump(new_impressions, f)
'''

    unlabeled = r'''import torch
from transformers import BertTokenizer
import bert_tokenizer
from torch.utils.data import Dataset

class UnlabeledDataset(Dataset):
    def __init__(self, csv_path):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        impressions = bert_tokenizer.get_impressions_from_csv(csv_path)
        self.encoded_imp = bert_tokenizer.tokenize(impressions, tokenizer)

    def __len__(self):
        return len(self.encoded_imp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        imp = torch.LongTensor(self.encoded_imp[idx])
        return {"imp": imp, "len": imp.shape[0]}
'''

    utils = '''import torch

def collate_fn(batch):
    max_len = max(b['len'] for b in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = []
    for i, b in enumerate(batch):
        imp = b['imp']
        padded[i, :len(imp)] = imp
        lengths.append(b['len'])
    return {'imp': padded, 'len': lengths}

def generate_attention_masks(batch, source_lengths, device):
    attention_mask = torch.zeros(batch.shape, dtype=torch.float)
    for i in range(batch.shape[0]):
        attention_mask[i, :source_lengths[i]] = 1
    return attention_mask.to(device)
'''

    os.makedirs(os.path.join(src, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(src, 'models'), exist_ok=True)

    with open(os.path.join(src, 'bert_tokenizer.py'), 'w') as f:
        f.write(bert_tok)
    with open(os.path.join(src, 'datasets', 'unlabeled_dataset.py'), 'w') as f:
        f.write(unlabeled)
    with open(os.path.join(src, 'utils.py'), 'w') as f:
        f.write(utils)

    for d in [src, os.path.join(src,'datasets'), os.path.join(src,'models')]:
        init = os.path.join(d, '__init__.py')
        if not os.path.exists(init):
            open(init, 'w').close()


def keyword_fallback(sentence: str) -> dict:
    sl = sentence.lower()
    result = {}
    for col, kws in KEYWORDS.items():
        result[col] = 1.0 if any(kw in sl for kw in kws) else float('nan')
    return result


def run_chexbert(sentences: list[str]) -> list[dict]:
    patch_chexbert()

    if not os.path.exists(CHEXBERT_DIR) or not os.path.exists(CHEXBERT_CKPT):
        print("⚠️  CheXbert nahi mila — keyword fallback use ho raha hai")
        return [keyword_fallback(s) for s in sentences]

    tmp_csv = '/tmp/chexbert_input.csv'
    tmp_out = '/tmp/chexbert_output'
    os.makedirs(tmp_out, exist_ok=True)

    stale = os.path.join(tmp_out, 'labeled_reports.csv')
    if os.path.exists(stale):
        os.remove(stale)

    pd.DataFrame({'Report Impression': sentences}).to_csv(tmp_csv, index=False)

    env = os.environ.copy()
    env['PYTHONPATH'] = CHEXBERT_DIR

    result = subprocess.run(
        ['python', os.path.join(CHEXBERT_DIR, 'src', 'label.py'),
         '-d', tmp_csv,
         '-o', tmp_out,
         '-c', CHEXBERT_CKPT],
        capture_output=True, text=True, env=env,
        cwd=CHEXBERT_DIR, timeout=120
    )

    out_csv = os.path.join(tmp_out, 'labeled_reports.csv')
    if not os.path.exists(out_csv):
        print(f"❌ CheXbert fail — keyword fallback\n{result.stderr[-500:]}")
        return [keyword_fallback(s) for s in sentences]

    df = pd.read_csv(out_csv)
    results = []
    for i in range(len(sentences)):
        row = {}
        for col in CHEXBERT_COLS:
            v = df.iloc[i][col] if col in df.columns else float('nan')
            row[col] = float(v) if not pd.isna(v) else float('nan')
        results.append(row)

    return results


def chexbert_to_tensor(label_dict: dict, device) -> torch.Tensor:
    CHEXBERT_COLS_ORDERED = [
        'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema',
        'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
        'Fracture', 'Support Devices', 'No Finding'
    ]
    vals = []
    for col in CHEXBERT_COLS_ORDERED:
        v = label_dict.get(col, float('nan'))
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv): vals.append(0.0)
            elif fv == 1.0:  vals.append(1.0)
            elif fv == 0.0:  vals.append(-1.0)
            elif fv == -1.0: vals.append(0.0)
            else:            vals.append(float(np.clip(fv, -1, 1)))
        except:
            vals.append(0.0)
    return torch.tensor(vals, dtype=torch.float32).unsqueeze(0).to(device)
