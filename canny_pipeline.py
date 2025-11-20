import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os

# -----------------------
# Utility: 2D convolution
# -----------------------
def convolve2d(image, kernel, padding='reflect'):
    """
    Simple 2D convolution (same output size).
    image: 2D float32 array
    kernel: 2D float32 array (will be flipped to implement convolution)
    padding: 'zero', 'reflect', 'edge'
    """
    image = image.astype(np.float32)
    H, W = image.shape
    kH, kW = kernel.shape
    kernel = np.flipud(np.fliplr(kernel)).astype(np.float32)
    pad_h = kH // 2
    pad_w = kW // 2

    if padding == 'zero':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif padding == 'reflect':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    elif padding == 'edge':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        raise ValueError("Unknown padding mode")

    out = np.zeros_like(image, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            patch = padded[y:y+kH, x:x+kW]
            out[y, x] = np.sum(patch * kernel)
    return out

# -----------------------
# Gaussian kernel (separable possibility)
# -----------------------
def gaussian_kernel(size=5, sigma=1.0):
    """Return a square Gaussian kernel normalized to sum=1."""
    assert size % 2 == 1, "size must be odd"
    ax = np.arange(-(size//2), size//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

# -----------------------
# Gradients via Sobel
# -----------------------
def sobel_filters(img):
    """Return Ix, Iy, magnitude, angle (angle in radians)"""
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = Kx.T
    Ix = convolve2d(img, Kx, padding='reflect')
    Iy = convolve2d(img, Ky, padding='reflect')
    magnitude = np.hypot(Ix, Iy)  # sqrt(Ix^2 + Iy^2)
    angle = np.arctan2(Iy, Ix)    # radians in [-pi, pi]
    return Ix, Iy, magnitude, angle

# -----------------------
# Non-maximum suppression
# -----------------------
def non_max_suppression(magnitude, angle):
    """
    magnitude: 2D array
    angle: gradient angle in radians
    Returns thinned edge strength (same shape).
    """
    H, W = magnitude.shape
    out = np.zeros((H, W), dtype=np.float32)

    # Convert angles to degrees [0,180)
    ang = np.degrees(angle) % 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255
            a = ang[i, j]

            # direction 0
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # direction 45
            elif (22.5 <= a < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # direction 90
            elif (67.5 <= a < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # direction 135
            elif (112.5 <= a < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                out[i,j] = magnitude[i,j]
            else:
                out[i,j] = 0.0
    return out

# -----------------------
# Double threshold and hysteresis
# -----------------------
def double_threshold_and_hysteresis(img, low_ratio=0.05, high_ratio=0.15):
    """
    img: non-max suppressed magnitude (float)
    low_ratio, high_ratio: relative to max magnitude
    Returns binary edges (0/255 uint8)
    """
    high = img.max() * high_ratio
    low = high * low_ratio / high_ratio if high_ratio != 0 else img.max() * low_ratio
    # Simpler common heuristic: low = high * 0.5
    # But we keep parameters flexible.

    H, W = img.shape
    res = np.zeros((H, W), dtype=np.uint8)

    strong = 255
    weak = 50

    strong_i, strong_j = np.where(img >= high)
    zeros_i, zeros_j = np.where(img < low)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # Hysteresis: any weak pixel connected to strong via 8-neighborhood becomes strong
    # Use stack-based flood fill from strong pixels
    from collections import deque
    dq = deque(zip(strong_i.tolist(), strong_j.tolist()))
    while dq:
        i, j = dq.popleft()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni = i + di
                nj = j + dj
                if ni < 0 or nj < 0 or ni >= H or nj >= W:
                    continue
                if res[ni, nj] == weak:
                    res[ni, nj] = strong
                    dq.append((ni, nj))

    # suppress remaining weak to 0, set strong to 255
    res[res != strong] = 0
    return res

# -----------------------
# Full Canny pipeline
# -----------------------
def canny_edge_detector(img, gaussian_size=5, gaussian_sigma=1.0,
                        low_threshold_ratio=0.5, high_threshold_ratio=0.9):
    """
    img: 2D grayscale float array (0..255) or (0..1) — it's ok
    gaussian_size: odd integer
    gaussian_sigma: float
    low/high_threshold_ratio: relative fractions to define double threshold from median heuristic
        Note: here we accept ratios relative to the median-based heuristic computed below.
    """
    # 1. ensure float and normalized
    img = img.astype(np.float32)
    if img.max() > 2.0:
        # assume 0..255 -> normalize to 0..1 for internal stability (optional)
        img_norm = img / 255.0
    else:
        img_norm = img.copy()

    # 2. Gaussian blur
    gk = gaussian_kernel(gaussian_size, gaussian_sigma)
    smoothed = convolve2d(img_norm, gk, padding='reflect')

    # 3. Gradients
    Ix, Iy, magnitude, angle = sobel_filters(smoothed)

    # 4. Non-maximum suppression
    nms = non_max_suppression(magnitude, angle)

    # 5. Double threshold & hysteresis
    # better thresholds: compute median of non-zero gradients
    nz = nms[nms > 0]
    if nz.size == 0:
        # nothing to do
        return np.zeros_like(img, dtype=np.uint8)

    med = np.median(nz)
    # heuristics: high = med * factor_high; low = med * factor_low
    # Allow caller-specified ratios to control these factors
    factor_high = high_threshold_ratio  # e.g. 0.9
    factor_low = low_threshold_ratio    # e.g. 0.5
    high = med * factor_high
    low = med * factor_low

    # Use fallback logic: if med==0, base on max
    if med == 0:
        high = nms.max() * 0.15
        low = high * 0.5

    # Now call threshold/hysteresis routine
    # But our function expects ratio relative to max; convert:
    # Simpler: just produce an edges map by passing absolute thresholds via a variant function
    def dt_hyst_with_abs(img_abs, low_abs, high_abs):
        H, W = img_abs.shape
        res = np.zeros((H, W), dtype=np.uint8)
        strong = 255
        weak = 50
        strong_map = img_abs >= high_abs
        weak_map = (img_abs >= low_abs) & (img_abs < high_abs)
        res[strong_map] = strong
        res[weak_map] = weak

        from collections import deque
        dq = deque(zip(*np.nonzero(strong_map)))
        while dq:
            i, j = dq.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni = i + di
                    nj = j + dj
                    if ni < 0 or nj < 0 or ni >= H or nj >= W:
                        continue
                    if res[ni, nj] == weak:
                        res[ni, nj] = strong
                        dq.append((ni, nj))
        res[res != strong] = 0
        return res

    edges = dt_hyst_with_abs(nms, low_abs=low, high_abs=high)
    return edges

# -----------------------
# Example usage (commented)
# -----------------------
if __name__ == "__main__":
    # Example: load image and run Canny.
    img_path = "assets/comp_vision_test.jpg"
    if os.path.exists(img_path):
        img_pil = Image.open(img_path).convert("L")
        arr = np.array(img_pil, dtype=np.float32)
    else:
        # fallback: create synthetic test image if not present
        arr = np.zeros((256, 256), dtype=np.float32)
        rr, cc = np.indices(arr.shape)
        mask = ((rr - 128)**2 + (cc - 128)**2) < 60**2
        arr[mask] = 255.0
        arr[60:80, 40:200] = 255.0  # a horizontal bar

    edges = canny_edge_detector(arr,
                                gaussian_size=5, gaussian_sigma=1.2,
                                low_threshold_ratio=0.5, high_threshold_ratio=1.0)

    # visualize
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Input (grayscale)")
    plt.imshow(arr, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Canny edges")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.show()
