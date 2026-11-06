# dataset_utils.py
import os
import random
from collections import defaultdict
import tensorflow as tf

# uses preprocess.preprocess_for_model
from preprocess import preprocess_for_model, IMG_SIZE

DATASET_DIR = "../dataset"   # path relative when running scripts from scripts/ folder
MIN_GENUINE_PER_USER = 4    # users with fewer genuines will be skipped for ArcFace training

def collect_users(dataset_dir=DATASET_DIR):
    users = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    users.sort()
    return users

def build_label_map(dataset_dir=DATASET_DIR, min_genuine=MIN_GENUINE_PER_USER):
    """
    Map user_name -> label index for users containing >= min_genuine images.
    Returns dict user->label and list of valid users.
    """
    users = collect_users(dataset_dir)
    label_map = {}
    valid_users = []
    idx = 0
    for u in users:
        gdir = os.path.join(dataset_dir, u, "genuine")
        if not os.path.exists(gdir): continue
        imgs = [f for f in os.listdir(gdir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if len(imgs) >= min_genuine:
            label_map[u] = idx
            valid_users.append(u)
            idx += 1
    return label_map, valid_users

def list_train_images(label_map, dataset_dir=DATASET_DIR):
    """
    Return lists: image_paths, labels (int)
    Only genuine images for training ArcFace.
    """
    X = []
    y = []
    for user, label in label_map.items():
        gdir = os.path.join(dataset_dir, user, "genuine")
        files = [f for f in os.listdir(gdir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        files.sort()
        for f in files:
            X.append(os.path.join(gdir, f))
            y.append(label)
    return X, y

def train_val_split_users(valid_users, val_frac=0.2, seed=42):
    random.seed(seed)
    users = valid_users[:]
    random.shuffle(users)
    split = int(len(users)*(1.0 - val_frac))
    return users[:split], users[split:]

def make_tf_dataset(image_paths, labels, batch_size=32, augment_fn=None, shuffle=True):
    """
    image_paths: list of file paths
    labels: list of ints
    This yields (image_tensor, label)
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels))
    def _map_fn(path, label):
        img = preprocess_for_model(path)  # returns 0..255 RGB image tensor
        # apply model-specific preprocess later in train script (EfficientNet preprocess_input)
        return img, label
    ds = ds.map(lambda p, l: tf.numpy_function(func=lambda _p, _l: (_map_fn(_p.decode('utf-8'), int(_l))), inp=[p, l], Tout=[tf.float32, tf.int32]) if False else _map_fn(p, l),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Utility to produce pairs for evaluation (genuine vs genuine, genuine vs forged)
def collect_eval_pairs(dataset_dir="../dataset"):
    """
    Returns list of tuples (imgA_path, imgB_path, label) label=1 for same-user genuine pair, 0 otherwise.
    We'll collect:
      - genuine-genuine (same user) pairs (positive)
      - genuine-forged (same user) pairs (negative)
      - genuine-genuine cross-user (negative)
    """
    users = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    positives = []
    negatives = []
    for u in users:
        gdir = os.path.join(dataset_dir, u, "genuine")
        fdir = os.path.join(dataset_dir, u, "forged")
        if not os.path.exists(gdir): continue
        gimgs = [os.path.join(gdir, f) for f in os.listdir(gdir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        gimgs.sort()
        # same-user genuine pairs
        for i in range(len(gimgs)):
            for j in range(i+1, len(gimgs)):
                positives.append((gimgs[i], gimgs[j]))
        # genuine vs forged (same-user) - negatives
        if os.path.exists(fdir):
            fimgs = [os.path.join(fdir, f) for f in os.listdir(fdir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            for g in gimgs:
                for f in fimgs:
                    negatives.append((g, f))
    # cross-user negatives: sample random pairs
    all_g = []
    for u in users:
        gdir = os.path.join(dataset_dir, u, "genuine")
        if not os.path.exists(gdir): continue
        gimgs = [os.path.join(gdir, f) for f in os.listdir(gdir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        all_g.extend(gimgs)
    # create some random cross-user negatives
    import random
    for _ in range(min(2000, len(all_g)//2)):
        a = random.choice(all_g); b = random.choice(all_g)
        # ensure not same file
        if a != b:
            negatives.append((a,b))
    pairs = []
    for p in positives:
        pairs.append((p[0], p[1], 1))
    for n in negatives:
        pairs.append((n[0], n[1], 0))
    return pairs
