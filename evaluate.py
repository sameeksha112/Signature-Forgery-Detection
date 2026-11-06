# evaluate.py
"""
Compute ROC / EER using saved backbone. Uses pairs created from dataset_utils.collect_eval_pairs.
Outputs AUC and best threshold at EER-like point (closest to equal error).
"""

import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from dataset_utils import collect_eval_pairs
from preprocess import preprocess_for_model
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

EPS = 1e-10

def get_embedding(emb_model, path):
    img = preprocess_for_model(path)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    batch = tf.expand_dims(img, 0)
    emb = emb_model.predict(batch, verbose=0)[0]
    emb = emb / (np.linalg.norm(emb) + EPS)
    return emb

def evaluate(emb_path, pairs, max_samples=None):
    emb_model = load_model(emb_path, compile=False)
    y_true = []
    y_score = []
    for i, (a,b,lab) in enumerate(pairs):
        if max_samples and i >= max_samples: break
        ea = get_embedding(emb_model, a)
        eb = get_embedding(emb_model, b)
        sim = float(np.dot(ea, eb))
        y_true.append(lab)
        y_score.append(sim)
        if (i+1) % 100 == 0:
            print("Processed pairs:", i+1)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # Find EER (approx): point where FPR ~ (1-TPR)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_thresh = thresholds[eer_idx]
    return {"auc": roc_auc, "fpr": fpr, "tpr": tpr, "thresholds": thresholds, "eer": eer, "eer_threshold": eer_thresh}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, help="Path to backbone_embedding.keras")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    pairs = collect_eval_pairs("../dataset")
    print("Total pairs collected:", len(pairs))
    res = evaluate(args.backbone, pairs, max_samples=args.max_samples)
    print("AUC:", res["auc"])
    print("EER (approx):", res["eer"])
    print("EER threshold:", res["eer_threshold"])
