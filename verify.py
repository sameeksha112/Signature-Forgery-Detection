# backend/scripts/verify.py
#003_0_002 old

"""
Signature verification with TTA, ink-crop preprocessing, cohort z-norm, optional pair-model ensemble,
per-genuine weighting and quality checks.

Usage (CLI):
  python verify.py --backbone ../models/backbone_embedding.keras \
                   --genuine ...g1.png ...g2.png \
                   --test ...test.png --no-tta  # etc

Server should call verify_with_loaded_model(emb_model, genuine_paths, test_path, ...).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings

# import your existing preprocessing (must exist)
from preprocess import preprocess_for_model, IMG_SIZE

EPS = 1e-10

# -------------------------
# Utilities: cropping to ink bbox
# -------------------------
def crop_to_ink_bbox_tf(img, pad=8):
    """
    img: tf.Tensor float32 HxWx3 or HxWx1 in range either [0..1] or [0..255].
    Returns cropped/resized image (H,W,3) same height/width as IMG_SIZE.
    If no ink found, returns original resized image.
    This uses TF ops so it can be used inside inference.
    """
    # Ensure float32
    img = tf.cast(img, tf.float32)
    # convert to grayscale in 0..255 range
    # if in 0..1 convert to 0..255 for thresholding
    maxv = tf.reduce_max(img)
    img_scaled = tf.cond(maxv <= 1.1, lambda: img * 255.0, lambda: img)
    gray = tf.image.rgb_to_grayscale(img_scaled)[:, :, 0]  # HxW

    # threshold for ink: pixels darker than 250 considered ink
    thresh = 250.0
    ink_mask = tf.less(gray, thresh)
    coords = tf.where(ink_mask)  # Nx2 or empty

    H, W = IMG_SIZE  # target H, W

    def no_ink():
        # no ink found -> just resize original to target and return
        r = tf.image.resize(img, (H, W), method=tf.image.ResizeMethod.BILINEAR)
        return tf.cast(r, tf.float32)

    def has_ink():
        # convert coords to int32
        coords_int = tf.cast(coords, tf.int32)
        y_min = tf.reduce_min(coords_int[:, 0])
        x_min = tf.reduce_min(coords_int[:, 1])
        y_max = tf.reduce_max(coords_int[:, 0])
        x_max = tf.reduce_max(coords_int[:, 1])
        # apply pad but ensure in-bounds
        y0 = tf.maximum(0, y_min - pad)
        x0 = tf.maximum(0, x_min - pad)
        y1 = tf.minimum(tf.shape(gray)[0] - 1, y_max + pad)
        x1 = tf.minimum(tf.shape(gray)[1] - 1, x_max + pad)
        h = y1 - y0 + 1
        w = x1 - x0 + 1
        cropped = img[y0:y0+h, x0:x0+w, :]
        resized = tf.image.resize(cropped, (H, W), method=tf.image.ResizeMethod.BILINEAR)
        return tf.cast(resized, tf.float32)

    out = tf.cond(tf.equal(tf.shape(coords)[0], 0), no_ink, has_ink)
    return out

def preprocess_and_crop(img_path, pad=8):
    """
    Load image via preprocess_for_model, apply crop_to_ink_bbox_tf and return tensor HxWx3 (0..255 float32).
    """
    # preprocess_for_model should return HxWx3 float32 (0..255 or 0..1). We'll accept both.
    img = preprocess_for_model(tf.constant(img_path))
    # ensure shape known
    img.set_shape([None, None, 3])
    cropped = crop_to_ink_bbox_tf(img, pad=pad)
    return cropped  # float32

# -------------------------
# TTA embedding (uses preprocess_and_crop)
# -------------------------
def get_tta_embedding(emb_model, img_path, tta=True, pad=8):
    """
    Compute L2-normalized embedding using TTA:
      - original (cropped)
      - horizontal flip
      - small translations ±2 px
    Returns: 1D numpy array (embedding)
    """
    # Ensure file exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_tensor = preprocess_and_crop(img_path, pad=pad)  # HxWx3 (float32)
    variants = [img_tensor]

    if tta:
        variants.append(tf.image.flip_left_right(img_tensor))
        shifts = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        H, W = IMG_SIZE
        for dx, dy in shifts:
            pad_x, pad_y = abs(dx), abs(dy)
            paddings = [[pad_y, pad_y], [pad_x, pad_x], [0, 0]]
            padded = tf.pad(img_tensor, paddings, mode="REFLECT")
            start_y = pad_y + max(0, -dy)
            start_x = pad_x + max(0, -dx)
            cropped = tf.image.crop_to_bounding_box(padded, start_y, start_x, H, W)
            variants.append(cropped)

    batch = tf.stack(variants, axis=0)  # (N,H,W,3)
    # model's preprocess_input (EfficientNet) expects -1..1
    try:
        batch_proc = tf.keras.applications.efficientnet.preprocess_input(batch)
    except Exception:
        batch_proc = (batch / 127.5) - 1.0

    emb_batch = emb_model.predict(batch_proc, verbose=0)
    emb_batch = np.asarray(emb_batch, dtype=np.float32)
    # normalize each emb
    norms = np.linalg.norm(emb_batch, axis=1, keepdims=True) + EPS
    emb_batch = emb_batch / norms
    # mean then renormalize
    emb_mean = np.mean(emb_batch, axis=0)
    emb_mean = emb_mean / (np.linalg.norm(emb_mean) + EPS)
    return emb_mean

# -------------------------
# Enrollment helpers
# -------------------------
def enrollment_embeddings(emb_model, genuine_paths, tta=True, pad=8):
    embs = []
    for p in genuine_paths:
        embs.append(get_tta_embedding(emb_model, p, tta=tta, pad=pad))
    return np.stack(embs, axis=0)  # (N,D)

# -------------------------
# Score normalization (cohort z-norm)
# -------------------------
def compute_cohort_znorm_score(enrol_embs, test_emb, cohort_embs):
    """
    Compute cosine similarities then z-normalize using cohort embeddings.
    cohort_embs: numpy array (M,D) of cohort embeddings (L2-normalized).
    Returns znormed median score (float) and raw median.
    """
    # raw sims
    sims = np.dot(enrol_embs, test_emb)
    raw_med = float(np.median(sims))
    # cohort baseline: compute similarities of test_emb to cohort
    cohort_sims = np.dot(cohort_embs, test_emb)  # M
    mu = float(np.mean(cohort_sims))
    sigma = float(np.std(cohort_sims) + 1e-6)
    z = (raw_med - mu) / sigma
    return z, raw_med

# -------------------------
# Optional: ensemble with pairwise Siamese/sigmoid model (if you have one)
# -------------------------
def predict_pair_sigmoid(pair_model, a_path, b_path, tta=True, pad=8):
    """
    If you have a pair model that takes [imgA, imgB] -> sigmoid probability,
    run it on TTA variants (flip/shift) and return the mean prob.
    This tries to construct matching preprocessed batches.
    """
    # build variants for both images (use preprocess_and_crop)
    def build_variants(path):
        img = preprocess_and_crop(path, pad=pad)
        vars = [img]
        if tta:
            vars.append(tf.image.flip_left_right(img))
            shifts = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            H, W = IMG_SIZE
            for dx, dy in shifts:
                pad_x, pad_y = abs(dx), abs(dy)
                padded = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")
                start_y = pad_y + max(0, -dy)
                start_x = pad_x + max(0, -dx)
                cropped = tf.image.crop_to_bounding_box(padded, start_y, start_x, H, W)
                vars.append(cropped)
        return vars

    A_vars = build_variants(a_path)
    B_vars = build_variants(b_path)
    # pair each variant by index (assume same length)
    N = min(len(A_vars), len(B_vars))
    probs = []
    for i in range(N):
        a = A_vars[i]
        b = B_vars[i]
        try:
            a_proc = tf.keras.applications.efficientnet.preprocess_input(a)
            b_proc = tf.keras.applications.efficientnet.preprocess_input(b)
        except Exception:
            a_proc = (a / 127.5) - 1.0
            b_proc = (b / 127.5) - 1.0
        A_batch = tf.expand_dims(a_proc, 0)
        B_batch = tf.expand_dims(b_proc, 0)
        # pair models typically accept [A,B]
        pred = pair_model.predict([A_batch, B_batch], verbose=0).reshape(-1)
        probs.append(float(pred[0]))
    return float(np.mean(probs))

def ensemble_cosine_and_pair(pair_model, emb_model, enrol_paths, test_path, tta=True, weight_cosine=0.5, pad=8):
    """
    Compute final ensemble score by averaging cosine-based score and pair-model sigmoid.
    weight_cosine: weight for cosine part (0..1); 1-weight for sigmoid.
    Returns (ensemble_score, cosine_median, pair_mean)
    """
    enrol_embs = enrollment_embeddings(emb_model, enrol_paths, tta=tta, pad=pad)
    test_emb = get_tta_embedding(emb_model, test_path, tta=tta, pad=pad)
    cosine_sims = np.dot(enrol_embs, test_emb)
    cosine_med = float(np.median(cosine_sims))

    pair_probs = []
    for p in enrol_paths:
        prob = predict_pair_sigmoid(pair_model, p, test_path, tta=tta, pad=pad)
        pair_probs.append(prob)
    pair_mean = float(np.mean(pair_probs)) if len(pair_probs) > 0 else 0.0

    # scale pair_mean (0..1) to cosine-like range if you want, or combine directly
    # we'll map cosine [-1,1] -> [0,1] and average
    cos_scaled = (cosine_med + 1.0) / 2.0
    ensemble = weight_cosine * cos_scaled + (1 - weight_cosine) * pair_mean
    # map back to [0,1] notion; final decision can use threshold on ensemble
    return ensemble, cosine_med, pair_mean

# -------------------------
# Simple quality check
# -------------------------
def quality_check(img_path, min_ink_frac=0.002, pad=8):
    """
    Return (ink_frac, ok_bool) where ink_frac is fraction of non-white pixels after cropping.
    If below min_ink_frac -> low quality.
    """
    img = preprocess_and_crop(img_path, pad=pad)  # tf.Tensor
    img_np = img.numpy() if isinstance(img, tf.Tensor) else np.asarray(img)
    gray = np.mean(img_np, axis=2)
    ink_pixels = np.sum(gray < 250)
    frac = float(ink_pixels) / float(gray.size)
    ok = frac >= min_ink_frac
    return frac, ok

# -------------------------
# Main verify function (extended)
# -------------------------
def verify_with_loaded_model(emb_model,
                             genuine_paths,
                             test_path,
                             scale=0.88,
                             global_threshold=None,
                             verbose=True,
                             tta=True,
                             cohort_embs=None,
                             pair_model=None,
                             weight_cosine=0.6,
                             pad=8):
    """
    Extended verification with:
     - preprocessing crop, TTA
     - optional cohort z-norm (pass cohort_embs: ndarray MxD)
     - optional pair_model (Keras model) to ensemble with cosine
     - per-genuine weighting and quality check

    Returns dict similar to previous implementation, with extra keys if used.
    """
    if len(genuine_paths) == 0:
        raise ValueError("No genuine images provided")

    # quick quality check for test image
    try:
        ink_frac, ok = quality_check(test_path, pad=pad)
    except Exception:
        ink_frac, ok = 0.0, True  # don't fail on QC error
    if not ok:
        if verbose:
            print(f"[WARN] Test image low ink fraction ({ink_frac:.5f}) - low confidence")

    # compute enrol embeddings & test embedding
    enrol_embs = enrollment_embeddings(emb_model, genuine_paths, tta=tta, pad=pad)  # (N,D)
    test_emb = get_tta_embedding(emb_model, test_path, tta=tta, pad=pad)  # (D,)

    # L2-normalize safety
    enrol_embs = enrol_embs / (np.linalg.norm(enrol_embs, axis=1, keepdims=True) + EPS)
    test_emb = test_emb / (np.linalg.norm(test_emb) + EPS)

    # per-enrol scores and median
    sims = np.dot(enrol_embs, test_emb)
    median_sim = float(np.median(sims))
    mean_sim = float(np.mean(sims))

    # per-genuine weighting: weight by closeness of that enrol to enrol-mean (reduce effect of outliers)
    enrol_mean = np.mean(enrol_embs, axis=0)
    dist_to_mean = np.linalg.norm(enrol_embs - enrol_mean, axis=1)
    # invert distances to get weights (closer -> higher weight)
    inv = 1.0 / (dist_to_mean + 1e-6)
    weights = inv / np.sum(inv)
    weighted_score = float(np.sum(weights * sims))

    # dynamic threshold from enrol
    dynamic_th = None
    if enrol_embs.shape[0] >= 2:
        dynamic_th = None
        try:
            # reuse previous function dynamic_threshold_from_enrol logic
            sims_pairs = []
            for i in range(enrol_embs.shape[0]):
                for j in range(i+1, enrol_embs.shape[0]):
                    sims_pairs.append(float(np.dot(enrol_embs[i], enrol_embs[j])))
            median_pair = float(np.median(sims_pairs))
            dynamic_th = max(median_pair * scale, 0.45)
        except Exception:
            dynamic_th = None

    # optionally apply cohort z-norm (if cohort_embs provided)
    zscore = None
    raw_med = median_sim
    if cohort_embs is not None:
        try:
            zscore, raw_med = compute_cohort_znorm_score(enrol_embs, test_emb, cohort_embs)
        except Exception:
            zscore = None

    # ensemble with pair model if provided
    pair_mean = None
    ensemble_score = None
    if pair_model is not None:
        try:
            ensemble_score, cosine_med, pair_mean = ensemble_cosine_and_pair(pair_model, emb_model, genuine_paths, test_path, tta=tta, weight_cosine=weight_cosine, pad=pad)
        except Exception as e:
            warnings.warn(f"Pair model ensemble failed: {e}")
            ensemble_score = None

    # Decision: use ensemble_score (if present) mapped to [0,1],
    # otherwise use cosine median in [ -1,1 ] with thresholds.
    decision = "Forged"
    used_threshold = None
    confidence = None

    if ensemble_score is not None:
        # ensemble in [0,1], threshold ~0.5 (tunable)
        used_threshold = 0.5 if global_threshold is None else max(0.5, global_threshold)
        confidence = ensemble_score
        decision = "Genuine" if ensemble_score >= used_threshold else "Forged"
    else:
        # use dynamic threshold if available
        used_threshold = dynamic_th if dynamic_th is not None else (global_threshold if global_threshold is not None else 0.60)
        confidence = (median_sim + 1.0) / 2.0  # map -1..1 -> 0..1
        decision = "Genuine" if median_sim >= used_threshold else "Forged"
        # if zscore available, we can also factor that into confidence:
        if zscore is not None:
            # map zscore to 0..1 via sigmoid and combine
            z_conf = 1.0 / (1.0 + np.exp(-zscore))
            confidence = 0.6 * confidence + 0.4 * float(z_conf)

    # Compose result
    result = {
        "median_sim": median_sim,
        "mean_sim": mean_sim,
        "weighted_score": weighted_score,
        "dynamic_threshold": dynamic_th,
        "global_threshold": global_threshold,
        "used_threshold": used_threshold,
        "decision": decision,
        "confidence": float(confidence) if confidence is not None else None,
        "per_enrollment_scores": sims.tolist(),
        "cohort_zscore": float(zscore) if zscore is not None else None,
        "ensemble_score": float(ensemble_score) if ensemble_score is not None else None,
        "pair_mean": float(pair_mean) if pair_mean is not None else None,
        "test_ink_frac": float(ink_frac)
    }

    if verbose:
        print("\n=== verify_with_loaded_model (extended) ===")
        print("Enrollment images:", genuine_paths)
        print("Test image:", test_path)
        print("Per-enrollment sims:", [round(float(x), 4) for x in sims.tolist()])
        print(f"median_sim={median_sim:.4f}, mean_sim={mean_sim:.4f}, weighted_score={weighted_score:.4f}")
        print(f"dynamic_threshold={dynamic_th}, global_threshold={global_threshold}, used_threshold={used_threshold}")
        if zscore is not None:
            print(f"cohort_zscore={zscore:.4f} (raw_med={raw_med:.4f})")
        if ensemble_score is not None:
            print(f"ensemble_score={ensemble_score:.4f}, pair_mean={pair_mean:.4f}")
        print(f"decision={decision}, confidence={result['confidence']:.4f}")
        print("========================================\n")

    return result

# -------------------------
# Convenience CLI: loads model then calls above
# -------------------------
def verify(emb_model_path, genuine_paths, test_path, **kwargs):
    if not os.path.exists(emb_model_path):
        raise FileNotFoundError(f"Embedding model not found: {emb_model_path}")
    emb_model = load_model(emb_model_path, compile=False)
    return verify_with_loaded_model(emb_model, genuine_paths, test_path, **kwargs)