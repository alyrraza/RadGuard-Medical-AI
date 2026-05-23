import os
import io
import uuid
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Results folder — heatmaps yahan save honge
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/tmp/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Public-facing URL for heatmap links
# HF auto-sets SPACE_HOST as hostname only (no protocol)
_hf_host = os.environ.get("SPACE_HOST", "")
API_BASE_URL = f"https://{_hf_host}" if _hf_host else "http://localhost:7860"

app = FastAPI(title="RadGuard AI Engine", version="1.0")

# Results folder publicly accessible
app.mount("/results", StaticFiles(directory=RESULTS_DIR),
          name="results")

# CORS — Partner ka frontend kisi bhi URL se call kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_model():
    import os
    from huggingface_hub import hf_hub_download

    # Download chexbert.pth if not already present
    chexbert_ckpt = os.environ.get("CHEXBERT_CKPT", "/app/CheXbert/src/chexbert.pth")
    if not os.path.exists(chexbert_ckpt):
        print("📥 Downloading chexbert.pth from HuggingFace Hub...")
        os.makedirs(os.path.dirname(chexbert_ckpt), exist_ok=True)
        hf_hub_download(
            repo_id="alyrraza/radguard-v11",
            filename="chexbert.pth",
            local_dir=os.path.dirname(chexbert_ckpt),
        )
        print("✅ chexbert.pth downloaded")
    else:
        print("✅ chexbert.pth already present")

    from inference.model import get_model, get_tokenizer
    get_model()
    get_tokenizer()
    print("✅ Model ready!")


def generate_heatmap(image: Image.Image, attn_map: np.ndarray,
                     condition_name: str, request_id: str,
                     server_url: str) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        from scipy.ndimage import zoom as nd_zoom, gaussian_filter
        am  = attn_map.reshape(14, 14)
        aup = nd_zoom(am, 448/14, order=3)
        aup = gaussian_filter(aup, sigma=8)
        aup = np.clip(aup, 0, None)
    except:
        aup = np.array(
            Image.fromarray(
                attn_map.reshape(14,14).astype(np.float32))
            .resize((448,448), resample=Image.BICUBIC),
            dtype=np.float32)

    if aup.max() > aup.min():
        aup = (aup - aup.min()) / (aup.max() - aup.min())

    img448 = np.array(image.resize((448, 448)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.imshow(img448)
    ax.imshow(aup, cmap='jet', alpha=0.5)
    ax.set_title(
        condition_name.replace('_', ' '),
        color='white', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    filename = f"{request_id}_{condition_name}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    fig.savefig(filepath, dpi=100,
                bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)

    return f"{server_url}/results/{filename}"


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    ai_report: str = Form(""),
):
    """
    Main endpoint.
    Input:  X-ray image + AI report text
    Output: ELRRs + conditions + heatmap URLs
    """
    report_text = ai_report.strip()

    if not report_text:
        return JSONResponse(
            status_code=400,
            content={"error": "AI report text required"})

    image_bytes = await file.read()
    image = Image.open(
        io.BytesIO(image_bytes)).convert('RGB')

    try:
        from inference.pipeline import (
            run_full_pipeline, CONDITIONS)
        from inference.chexbert_runner import chexbert_to_tensor
        from inference.model import (
            run_inference_on_sentence, device)

        result = run_full_pipeline(image, report_text)

        server_url = API_BASE_URL
        request_id = str(uuid.uuid4())[:8]

        all_attn     = result.pop('all_attn')
        sentences    = result.pop('sentences')
        all_chexbert = result.pop('all_chexbert')

        heatmaps = {}
        active_names = [
            c['name'] for c in result['conditions']
        ][:4]

        if all_attn and len(all_attn) > 0:
            for cond in active_names:
                ci       = CONDITIONS.index(cond)
                attn_map = all_attn[0][ci]
                url      = generate_heatmap(
                    image, attn_map, cond,
                    request_id, server_url)
                heatmaps[cond] = url

        task2 = {}
        for cond in result['conditions']:
            task2[cond['name']] = {
                'xray_present': cond['xray_present'],
                'confidence':   cond['confidence'],
            }

        return JSONResponse(content={
            "task1_elrrs":         result['elrrs'],
            "task1_conditions":    result['conditions'],
            "task2_xray_findings": task2,
            "task3_heatmaps":      heatmaps,
            "not_mentioned":       result['not_mentioned'],
            "sentences_analyzed":  len(sentences),
            "request_id":          request_id,
        })

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error":     str(e),
                "traceback": traceback.format_exc()
            })


@app.get("/health")
def health():
    return {"status": "ok", "model": "RadGuard V11"}


@app.get("/")
def root():
    return {
        "message": "RadGuard AI Engine — use /analyze endpoint"
    }
