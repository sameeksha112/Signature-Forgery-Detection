# preprocess.py
import tensorflow as tf
import numpy as np

IMG_SIZE = (155, 220)  # (H, W)

def load_image_tensor(path):
    """Return float32 image 0..1 with 3 channels decoded (RGB)."""
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
    return img  # (H, W, 3)

def rgb_to_grayscale(img):
    """Return (H,W,1) grayscale float32"""
    gray = tf.image.rgb_to_grayscale(img)
    return gray

def enhance_contrast(img_gray):
    """Slight contrast adjustment and smoothing. Input 0..1"""
    img = tf.image.adjust_contrast(img_gray, 1.2)
    # small smoothing via average filter to remove high-frequency noise
    img = tf.expand_dims(img, 0)  # add batch
    # 3x3 average filter
    kernel = tf.constant(1.0, shape=(3,3,1,1)) / 9.0
    img = tf.nn.conv2d(img, kernel, strides=1, padding='SAME')
    img = tf.squeeze(img, 0)
    return img

def crop_to_ink_bbox(img_gray, pad=8, min_size=20):
    """
    Crop to bounding box of dark pixels (ink). If no ink found, returns input image (will be center-cropped/resized later).
    img_gray: (H,W,1)
    """
    # Flatten single channel
    flat = tf.squeeze(img_gray, axis=-1)  # (H,W)
    # threshold relative to mean: ink tends to be darker
    mean_val = tf.reduce_mean(flat)
    thresh = mean_val * 0.98 + 0.02  # heuristic
    coords = tf.where(flat < thresh)  # indices (N,2) - dtype may be int64 on some TF builds

    # Cast coords to int32 for consistent index math across TF ops (avoids int64/int32 mismatch)
    coords = tf.cast(coords, tf.int32)

    # If no ink coordinates found, fallback (return un-cropped image)
    if tf.shape(coords)[0] == 0:
        return img_gray

    y_min = tf.reduce_min(coords[:, 0])
    x_min = tf.reduce_min(coords[:, 1])
    y_max = tf.reduce_max(coords[:, 0])
    x_max = tf.reduce_max(coords[:, 1])

    # Ensure pad and shape types are int32
    pad_i = tf.cast(pad, tf.int32)
    H = tf.shape(img_gray)[0]
    W = tf.shape(img_gray)[1]

    y0 = tf.maximum(tf.cast(0, tf.int32), y_min - pad_i)
    x0 = tf.maximum(tf.cast(0, tf.int32), x_min - pad_i)
    y1 = tf.minimum(H - 1, y_max + pad_i)  # inclusive index
    x1 = tf.minimum(W - 1, x_max + pad_i)

    # Guards if bbox became degenerate
    # compute sizes
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    if h < min_size or w < min_size:
        return img_gray

    # slice (works with int32 indices)
    cropped = img_gray[y0: y1 + 1, x0: x1 + 1, :]
    return cropped

def resize_and_pad(img, target_size=IMG_SIZE):
    """Resize preserving aspect ratio and pad to target_size, output (H,W,1)."""
    img = tf.image.convert_image_dtype(img, tf.float32)
    ih = tf.shape(img)[0]; iw = tf.shape(img)[1]
    th, tw = target_size
    scale = tf.minimum(tf.cast(th, tf.float32)/tf.cast(ih, tf.float32),
                       tf.cast(tw, tf.float32)/tf.cast(iw, tf.float32))
    nh = tf.cast(tf.round(tf.cast(ih, tf.float32) * scale), tf.int32)
    nw = tf.cast(tf.round(tf.cast(iw, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, (nh, nw), method='bilinear')
    pad_top = (th - nh) // 2
    pad_left = (tw - nw) // 2
    pad_bottom = th - nh - pad_top
    pad_right = tw - nw - pad_left
    # pad with white background (value 1.0)
    img = tf.pad(img, [[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1.0)
    # Set static shape for TF graph tracing
    img.set_shape((th, tw, 1))
    return img

def to_rgb(img_gray):
    """Duplicate grayscale channel to 3-channels for pretrained backbone"""
    return tf.image.grayscale_to_rgb(img_gray)

def preprocess_for_model(path, target_size=IMG_SIZE):
    """
    Full preprocessing pipeline: load -> grayscale -> enhance -> crop -> resize-pad -> to_rgb -> final preproc
    Returns float32 0..255 image for feeding into tf.keras.applications preprocess_input
    Accepts `path` as a tf.string scalar tensor (works inside tf.data)
    """
    # Read and decode (path may be tf.Tensor string)
    img = load_image_tensor(path)
    gray = rgb_to_grayscale(img)
    gray = enhance_contrast(gray)
    cropped = crop_to_ink_bbox(gray, pad=8)
    resized = resize_and_pad(cropped, target_size=target_size)
    rgb = to_rgb(resized)  # (H,W,3) 0..1
    rgb_255 = rgb * 255.0
    return rgb_255
