"""
AI Model Loader + Runner
Tumhara trained ChestXrayErrorDetector yahan load hota hai.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ── Paths ────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "./best_model_v8.pth")

# ── Device ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# ── Constants ────────────────────────────────────────────────
CONDITIONS = [
    'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity',
    'Lung_Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural_Effusion',
    'Pleural_Other', 'Fracture', 'Support_Devices', 'No_Finding'
]
IDX2ERROR = {0: 'SUPPORTED', 1: 'HALLUCINATED', 2: 'MISSING', 3: 'INACCURATE'}
CONDITION_TYPES = {
    'Enlarged_Cardiomediastinum': 1, 'Cardiomegaly': 1,
    'Lung_Opacity': 0, 'Lung_Lesion': 0, 'Edema': 0,
    'Consolidation': 0, 'Pneumonia': 0, 'Atelectasis': 0,
    'Pneumothorax': 2, 'Pleural_Effusion': 2, 'Pleural_Other': 2,
    'Fracture': 2, 'Support_Devices': 3, 'No_Finding': 4,
}
CONDITION_TYPE_IDS = torch.tensor(
    [CONDITION_TYPES[c] for c in CONDITIONS], dtype=torch.long
)

# ── Image transform ──────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── BioViL ───────────────────────────────────────────────────
try:
    from health_multimodal.image.model.pretrained import (
        get_biovil_t_image_encoder, BIOMED_VLP_BIOVIL_T, BIOVIL_T_COMMIT_TAG)
    from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer
    BIOVIL_AVAILABLE = True
    print("✅ BioViL-T available")
except ImportError:
    BIOVIL_AVAILABLE = False
    print("⚠️  BioViL-T nahi mila — DenseNet+ClinicalBERT fallback")


# ── Model Architecture ────────────────────────────────────────
class MLPBlock(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*expansion, dim), nn.Dropout(dropout))
    def forward(self, x): return x + self.net(x)


class MLPMixer(nn.Module):
    def __init__(self, num_tokens=2, token_dim=512, num_blocks=4, dropout=0.1):
        super().__init__()
        self.token_mixers   = nn.ModuleList([MLPBlock(num_tokens,2,dropout) for _ in range(num_blocks)])
        self.channel_mixers = nn.ModuleList([MLPBlock(token_dim,4,dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(token_dim)
    def forward(self, x):
        for tm, cm in zip(self.token_mixers, self.channel_mixers):
            x = x + tm(x.transpose(1,2)).transpose(1,2)
            x = x + cm(x)
        return self.norm(x).flatten(1)


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, num_conditions=14, text_dim=768, img_dim=512,
                 out_dim=512, num_heads=8, dropout=0.1, type_dim=64):
        super().__init__()
        self.num_conditions = num_conditions
        self.num_heads = num_heads; self.head_dim = out_dim//num_heads
        self.out_dim = out_dim; self.scale = self.head_dim**-0.5
        self.q_projs    = nn.ModuleList([nn.Linear(text_dim+type_dim, out_dim) for _ in range(num_conditions)])
        self.k_proj     = nn.Linear(img_dim, out_dim)
        self.v_proj     = nn.Linear(img_dim, out_dim)
        self.out_proj   = nn.Linear(out_dim, out_dim)
        self.norm1      = nn.LayerNorm(out_dim)
        rev_dim = out_dim//2
        self.rev_head_dim = rev_dim//num_heads; self.rev_dim = rev_dim
        self.rev_q_projs  = nn.ModuleList([nn.Linear(img_dim+type_dim, rev_dim) for _ in range(num_conditions)])
        self.rev_k_proj   = nn.Linear(text_dim, rev_dim)
        self.rev_v_proj   = nn.Linear(text_dim, rev_dim)
        self.rev_out_proj = nn.Linear(rev_dim, rev_dim)
        self.norm2        = nn.LayerNorm(rev_dim)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, text_cls, text_tokens, spatial_feat, type_emb, img_gap):
        B=text_cls.shape[0]; nr=spatial_feat.shape[2]*spatial_feat.shape[3]; T=text_tokens.shape[1]
        sp=spatial_feat.flatten(2).transpose(1,2)
        K1=self.k_proj(sp).view(B,nr,self.num_heads,self.head_dim).transpose(1,2)
        V1=self.v_proj(sp).view(B,nr,self.num_heads,self.head_dim).transpose(1,2)
        K2=self.rev_k_proj(text_tokens).view(B,T,self.num_heads,self.rev_head_dim).transpose(1,2)
        V2=self.rev_v_proj(text_tokens).view(B,T,self.num_heads,self.rev_head_dim).transpose(1,2)
        ia,ta,am=[],[],[]
        for ci in range(self.num_conditions):
            tc=type_emb[:,ci,:]
            q1=self.q_projs[ci](torch.cat([text_cls,tc],1)).view(B,1,self.num_heads,self.head_dim).transpose(1,2)
            a1=self.dropout(F.softmax(torch.matmul(q1,K1.transpose(-2,-1))*self.scale,dim=-1))
            o1=self.norm1(self.out_proj(torch.matmul(a1,V1).transpose(1,2).contiguous().view(B,self.out_dim)))
            q2=self.rev_q_projs[ci](torch.cat([img_gap,tc],1)).view(B,1,self.num_heads,self.rev_head_dim).transpose(1,2)
            a2=self.dropout(F.softmax(torch.matmul(q2,K2.transpose(-2,-1))*self.scale,dim=-1))
            o2=self.norm2(self.rev_out_proj(torch.matmul(a2,V2).transpose(1,2).contiguous().view(B,self.rev_dim)))
            ia.append(o1); ta.append(o2); am.append(a1.mean(dim=1).squeeze(1))
        return torch.stack(ia,1), torch.stack(ta,1), torch.stack(am,1)


class ChestXrayErrorDetector(nn.Module):
    def __init__(self, num_conditions=14, dropout=0.4):
        super().__init__()
        self.num_conditions = num_conditions
        if BIOVIL_AVAILABLE:
            self.vision_encoder = get_biovil_t_image_encoder()
            self.img_feat_dim   = 512
            self.text_encoder   = CXRBertModel.from_pretrained(
                BIOMED_VLP_BIOVIL_T, revision=BIOVIL_T_COMMIT_TAG)
            self.txt_feat_dim   = 768
        else:
            from torchvision import models
            from transformers import AutoModel
            dn = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.vision_encoder = nn.Sequential(*list(dn.children())[:-1])
            self.img_feat_dim   = 1024
            self.text_encoder   = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
            self.txt_feat_dim   = 768
        self.type_embedding   = nn.Embedding(5, 64)
        self.register_buffer('condition_type_ids', CONDITION_TYPE_IDS)
        self.cross_attention  = BidirectionalCrossAttention(
            num_conditions=num_conditions, text_dim=self.txt_feat_dim,
            img_dim=self.img_feat_dim, out_dim=512, num_heads=8,
            dropout=dropout, type_dim=64)
        self.text_proj        = nn.Sequential(
            nn.Linear(self.txt_feat_dim,512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(dropout))
        self.label_encoder    = nn.Sequential(
            nn.Linear(14,64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(dropout))
        self.mixer            = MLPMixer(num_tokens=2, token_dim=512, num_blocks=4, dropout=dropout)
        self.shared_mlp       = nn.Sequential(
            nn.Linear(1344,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512,256),  nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout))
        self.task1_heads      = nn.ModuleList([nn.Linear(256,4) for _ in range(num_conditions)])
        self.inaccurate_heads = nn.ModuleList([nn.Linear(256,1) for _ in range(num_conditions)])
        self.vision_gap       = nn.AdaptiveAvgPool2d((1,1))
        self.task2_mlp        = nn.Sequential(
            nn.Linear(self.img_feat_dim,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout))
        self.task2_heads      = nn.ModuleList([nn.Linear(256,1) for _ in range(num_conditions)])

    def _spatial(self, img):
        if BIOVIL_AVAILABLE:
            out = self.vision_encoder(img)
            B = out.patch_embeddings.shape[0]
            return out.patch_embeddings.transpose(1,2).reshape(B,512,14,14)
        return self.vision_encoder(img)

    def _text(self, ids, mask):
        out = self.text_encoder(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state[:,0,:], out.last_hidden_state

    def forward(self, image, input_ids, attention_mask, ai_chexbert):
        spatial=self._spatial(image); gap=self.vision_gap(spatial).flatten(1)
        cls,tk=self._text(input_ids,attention_mask); tp=self.text_proj(cls)
        B=image.shape[0]
        te=self.type_embedding(self.condition_type_ids.unsqueeze(0).expand(B,-1))
        ia,ta,attn=self.cross_attention(cls,tk,spatial,te,gap)
        ai_chexbert=torch.nan_to_num(ai_chexbert,nan=0.0,posinf=1.0,neginf=-1.0)
        lf=torch.nan_to_num(self.label_encoder(ai_chexbert),nan=0.0)
        t1l,inl=[],[]
        for ci in range(self.num_conditions):
            mx=self.mixer(torch.stack([ia[:,ci,:],tp],dim=1))
            sh=self.shared_mlp(torch.cat([mx,ta[:,ci,:],lf],dim=1))
            t1l.append(self.task1_heads[ci](sh))
            inl.append(self.inaccurate_heads[ci](sh))
        t1=torch.stack(t1l,dim=1); inc=torch.cat(inl,dim=1)
        t2f=self.task2_mlp(gap)
        t2=torch.cat([h(t2f) for h in self.task2_heads],dim=1)
        return t1,t2,attn,inc


# ── Global model instance (ek baar load, baar baar use) ─────
_model = None
_tokenizer = None


def get_model():
    global _model
    if _model is None:
        model_path = MODEL_PATH
        if not model_path or not os.path.exists(model_path):
            print("📥 HuggingFace Hub se model download ho raha hai...")
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="alyrraza/radguard-v11",
                filename="best_model_v11.pth"
            )
        print(f"📥 Model load ho raha hai: {model_path}")
        _model = ChestXrayErrorDetector(14, dropout=0.4)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        try:
            _model.load_state_dict(sd, strict=True)
            print("✅ Model weights loaded (strict)")
        except:
            _model.load_state_dict(sd, strict=False)
            print("✅ Model weights loaded (non-strict)")
        _model.to(device).eval()
        print(f"✅ Model ready on {device}")
    return _model


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        if BIOVIL_AVAILABLE:
            _tokenizer = CXRBertTokenizer.from_pretrained(
                BIOMED_VLP_BIOVIL_T, revision=BIOVIL_T_COMMIT_TAG)
        else:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        print("✅ Tokenizer ready")
    return _tokenizer


def tokenize_text(text: str):
    tok = get_tokenizer()
    if not text or str(text).strip() in ('', 'nan', 'None'):
        text = '[PAD]'
    e = tok(str(text), max_length=128, padding='max_length',
            truncation=True, return_tensors='pt')
    return e['input_ids'].to(device), e['attention_mask'].to(device)


def run_inference_on_sentence(image: Image.Image, sentence: str,
                               chex_tensor: torch.Tensor) -> tuple:
    model = get_model()
    img_t = val_transform(image).unsqueeze(0).to(device)
    ids, mask = tokenize_text(sentence)
    chex_tensor = torch.nan_to_num(chex_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    with torch.no_grad():
        t1l, t2l, attn, _ = model(img_t, ids, mask, chex_tensor)

    t1p  = np.nan_to_num(F.softmax(t1l[0], dim=-1).cpu().numpy(), nan=0.25)
    t2p  = np.nan_to_num(torch.sigmoid(t2l[0]).cpu().numpy(), nan=0.0)
    attn_np = np.nan_to_num(attn[0].cpu().numpy(), nan=0.0)

    preds, t2_preds = {}, {}
    for ci, cond in enumerate(CONDITIONS):
        pi = int(t1p[ci].argmax())
        preds[cond] = {
            'prediction': IDX2ERROR[pi],
            'confidence': float(t1p[ci][pi]),
            'probs': {k: float(t1p[ci][i]) for i, k in IDX2ERROR.items()}
        }
        t2_preds[cond] = {
            'present': bool(t2p[ci] > 0.5),
            'confidence': float(t2p[ci])
        }
    return preds, t2_preds, attn_np
