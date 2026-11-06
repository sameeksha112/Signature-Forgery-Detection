# backend/server.py

"""
FastAPI server for signature verification.

- Loads backbone_embedding.keras once at startup and reuses it.
- Optionally loads pair_siamese.keras and cohort_embs.npy if present in models dir.
- Endpoints:
    POST /register          (form: user_id, images...)      -> save images
    GET  /check/{user_id}                                   -> check if user exists ✅ NEW
    GET  /users                                              -> list users
    GET  /user/{user_id}/genuine                            -> list genuine image URLs
    POST /verify            (form: user_id, test_image, ...) -> verify and return result
    GET  /health                                             -> readiness
"""

import os
import sys
import time
import asyncio
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import tensorflow as tf

# add scripts folder so we can import verify.verify_with_loaded_model
BASE_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from verify import verify_with_loaded_model  # noqa: E402

# Model / dataset paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BACKBONE_MODEL_PATH = os.path.join(MODELS_DIR, "backbone_embedding.keras")
PAIR_MODEL_PATH = os.path.join(MODELS_DIR, "pair_siamese.keras")     # optional
COHORT_EMB_PATH = os.path.join(MODELS_DIR, "cohort_embs.npy")       # optional

# ensure dataset & models dirs exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="Signature Verifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved dataset images at /uploads/<user>/...
app.mount("/uploads", StaticFiles(directory=DATASET_DIR), name="uploads")

# Globals to hold loaded models/embeddings
EMB_MODEL: Optional[tf.keras.Model] = None
PAIR_MODEL: Optional[tf.keras.Model] = None
COHORT_EMBS: Optional[np.ndarray] = None

def safe_filename(filename: str) -> str:
    """Return a filesystem-safe basename for uploaded files."""
    return os.path.basename(filename)

@app.on_event("startup")
def load_models_at_startup():
    """
    Load embedding backbone (required) and optional pair model/cohort if available.
    """
    global EMB_MODEL, PAIR_MODEL, COHORT_EMBS

    if not os.path.exists(BACKBONE_MODEL_PATH):
        raise RuntimeError(f"Backbone model not found at {BACKBONE_MODEL_PATH}. Place backbone_embedding.keras there.")
    print("[startup] Loading backbone embedding model...")
    EMB_MODEL = tf.keras.models.load_model(BACKBONE_MODEL_PATH, compile=False)
    print("[startup] Backbone model loaded.")

    if os.path.exists(PAIR_MODEL_PATH):
        try:
            print("[startup] Loading pair model (optional)...")
            PAIR_MODEL = tf.keras.models.load_model(PAIR_MODEL_PATH, compile=False)
            print("[startup] Pair model loaded.")
        except Exception as e:
            print(f"[startup] Warning: failed to load pair model: {e}")
            PAIR_MODEL = None

    if os.path.exists(COHORT_EMB_PATH):
        try:
            COHORT_EMBS = np.load(COHORT_EMB_PATH)
            if COHORT_EMBS.dtype != np.float32:
                COHORT_EMBS = COHORT_EMBS.astype(np.float32)
            print(f"[startup] Loaded cohort embeddings, shape={COHORT_EMBS.shape}")
        except Exception as e:
            print(f"[startup] Warning: failed to load cohort embeddings: {e}")
            COHORT_EMBS = None

@app.get("/health")
def health():
    return {"ready": EMB_MODEL is not None}

# ========================================================
# ✅ NEW ENDPOINT: CHECK IF USER EXISTS
# ========================================================
@app.get("/check/{user_id}")
def check_user(user_id: str):
    user_dir = os.path.join(DATASET_DIR, user_id)
    return {"exists": os.path.exists(user_dir)}

# ========================================================
# ✅ UPDATED REGISTER ENDPOINT (duplicate + min 4 images)
# ========================================================
@app.post("/register")
async def register_user(
    user_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    # ✅ Require at least 4 images
    if not images or len(images) < 4:
        raise HTTPException(status_code=400, detail="At least 4 images are required")

    user_dir = os.path.join(DATASET_DIR, user_id)

    # ✅ Block duplicate users
    if os.path.exists(user_dir):
        raise HTTPException(status_code=409, detail="User already exists")

    genuine_dir = os.path.join(user_dir, "genuine")
    os.makedirs(genuine_dir, exist_ok=True)

    saved_files = []
    for idx, upload in enumerate(images, start=1):
        fname = f"g{int(time.time())}{idx}" + safe_filename(upload.filename)
        out_path = os.path.join(genuine_dir, fname)
        with open(out_path, "wb") as f:
            f.write(await upload.read())
        saved_files.append(out_path)

    return JSONResponse({"status": "ok", "saved": saved_files, "count": len(saved_files)})

# ---------------------------
# List registered users
# ---------------------------
@app.get("/users")
def list_users():
    users = []
    for name in sorted(os.listdir(DATASET_DIR)):
        p = os.path.join(DATASET_DIR, name)
        if os.path.isdir(p):
            users.append(name)
    return {"users": users}

# ---------------------------
# List genuine images (URLs) for a user
# ---------------------------
@app.get("/user/{user_id}/genuine")
def list_user_genuine(user_id: str, request: Request):
    genuine_dir = os.path.join(DATASET_DIR, user_id, "genuine")
    if not os.path.exists(genuine_dir):
        raise HTTPException(status_code=404, detail="User not found or no genuine images")
    files = [f for f in sorted(os.listdir(genuine_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    base = str(request.base_url).rstrip("/")
    urls = [f"{base}/uploads/{user_id}/genuine/{f}" for f in files]
    return {"count": len(urls), "images": urls}

# ---------------------------
# Verify endpoint
# ---------------------------
@app.post("/verify")
async def verify_endpoint(
    user_id: str = Form(...),
    test_image: UploadFile = File(...),
    tta: bool = Form(True),
    use_pair: bool = Form(False),
    use_cohort: bool = Form(False),
):
    global EMB_MODEL, PAIR_MODEL, COHORT_EMBS
    if EMB_MODEL is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")

    genuine_dir = os.path.join(DATASET_DIR, user_id, "genuine")
    if not os.path.exists(genuine_dir):
        raise HTTPException(status_code=404, detail="No such user or no registered images")

    genuines = sorted([
        os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(genuines) < 2:
        return JSONResponse(
            {"status": "error", "detail": "Need at least 2 genuine images for verification"},
            status_code=400
        )

    chosen = genuines[:4]

    tmp_dir = os.path.join(BASE_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_name = f"test_{int(time.time())}_" + safe_filename(test_image.filename)
    tmp_path = os.path.join(tmp_dir, tmp_name)

    with open(tmp_path, "wb") as f:
        f.write(await test_image.read())

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: verify_with_loaded_model(
                emb_model=EMB_MODEL,
                genuine_paths=chosen,
                test_path=tmp_path,
                scale=0.88,
                global_threshold=None,
                verbose=False,
                tta=tta,
                cohort_embs=COHORT_EMBS if use_cohort else None,
                pair_model=PAIR_MODEL if use_pair else None,
                weight_cosine=0.6,
                pad=8,
            )
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse({"status": "ok", "result": result})

# ---------------------------
# Run helper
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
