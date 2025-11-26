# farneback_simplified.py
# Упрощённая обучающая реализация Farnebäck-like dense optical flow (без OpenCV, без SciPy).
# Зависимости: numpy, pillow, matplotlib
# Запуск: python farneback_simplified.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# Utilities: gaussian kernel, 2D convolution (naive), bilinear sampling, warping
# ---------------------------
def gaussian_kernel(size=5, sigma=1.0):
    assert size % 2 == 1 and size >= 1
    ax = np.arange(-(size//2), size//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx*xx + yy*yy) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k

def convolve2d(image, kernel, padding='reflect'):
    """Naive 2D convolution (output same shape). image and kernel are 2D float arrays."""
    image = image.astype(np.float32)
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2
    if padding == 'reflect':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    elif padding == 'edge':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            region = padded[y:y+kH, x:x+kW]
            out[y, x] = np.sum(region * kernel)
    return out

def bilinear_sample(img, grid_x, grid_y):
    """Bilinear sample of 2D image at coordinates grid_x, grid_y (floats).
       grid_x, grid_y same shape -> returns sampled image of that shape.
    """
    H, W = img.shape
    gx = np.clip(grid_x, 0, W - 1)
    gy = np.clip(grid_y, 0, H - 1)
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wa = (x1 - gx) * (y1 - gy)
    wb = (gx - x0) * (y1 - gy)
    wc = (x1 - gx) * (gy - y0)
    wd = (gx - x0) * (gy - y0)

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def warp_image(img, flow):
    """Warp image img (H,W) using flow (H,W,2) where flow[y,x] = (u,v) (x and y disp)."""
    H, W = img.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    map_x = xs + flow[..., 0]
    map_y = ys + flow[..., 1]
    return bilinear_sample(img, map_x, map_y)

# ---------------------------
# Pyramid helpers
# ---------------------------
def pyr_down(img):
    """Gaussian blur then subsample by factor 2."""
    k = gaussian_kernel(size=5, sigma=1.0)
    blurred = convolve2d(img, k, padding='reflect')
    return blurred[::2, ::2]

def pyr_up(img, out_shape):
    """Upsample 2D image or 3D flow field to out_shape (H_out,W_out).
       For flow (H,W,2) upsample each channel and return shape (H_out,W_out,2).
    """
    H_out, W_out = out_shape
    if img.ndim == 2:
        H_in, W_in = img.shape
        # compute target grid in source coordinates
        ys = (np.arange(H_out) + 0.5) * (H_in / H_out) - 0.5
        xs = (np.arange(W_out) + 0.5) * (W_in / W_out) - 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        up = bilinear_sample(img, grid_x, grid_y)
        # smooth to fill holes (small Gaussian)
        kernel = gaussian_kernel(size=5, sigma=1.0)
        up = convolve2d(up, kernel, padding='reflect')
        return up
    elif img.ndim == 3 and img.shape[2] == 2:
        H_in, W_in, _ = img.shape
        ys = (np.arange(H_out) + 0.5) * (H_in / H_out) - 0.5
        xs = (np.arange(W_out) + 0.5) * (W_in / W_out) - 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        u_up = bilinear_sample(img[..., 0], grid_x, grid_y)
        v_up = bilinear_sample(img[..., 1], grid_x, grid_y)
        kernel = gaussian_kernel(size=5, sigma=1.0)
        u_up = convolve2d(u_up, kernel, padding='reflect')
        v_up = convolve2d(v_up, kernel, padding='reflect')
        up = np.stack([u_up, v_up], axis=2)
        return up
    else:
        raise ValueError("pyr_up: unsupported input shape {}".format(img.shape))

# ---------------------------
# Fit quadratic polynomial on each local window
# p(x,y) = p20*x^2 + p11*x*y + p02*y^2 + p10*x + p01*y + p00
# Return coeffs[y,x,:] = [p20,p11,p02,p10,p01,p00]
# ---------------------------
def fit_quadratic_field(img, win=5):
    H, W = img.shape
    r = win // 2
    # build design matrix A (N x 6)
    coords = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            coords.append([dx*dx, dx*dy, dy*dy, dx, dy, 1.0])
    A = np.array(coords, dtype=np.float32)  # N x 6
    ATA = A.T @ A
    # regularize tiny eigenvalues
    ATA += np.eye(6, dtype=np.float32) * 1e-6
    pinv = np.linalg.solve(ATA, A.T)  # 6 x N
    padded = np.pad(img, ((r, r), (r, r)), mode='reflect')
    coeffs = np.zeros((H, W, 6), dtype=np.float32)
    # sliding window fit
    N = win * win
    for y in range(H):
        for x in range(W):
            patch = padded[y:y+win, x:x+win].reshape(N)
            p = pinv @ patch
            coeffs[y, x, :] = p
    return coeffs

def poly_to_Ab(p):
    """Convert polynomial coefficients to A (2x2) and b (2x1)."""
    p20, p11, p02, p10, p01, p00 = p
    A = np.array([[p20, p11/2.0], [p11/2.0, p02]], dtype=np.float32)
    b = np.array([p10, p01], dtype=np.float32)
    return A, b

# ---------------------------
# Estimate per-pixel local displacement from polynomial coeffs
# using linearized relation: b2 ≈ b1 - 2*A1*d  =>  2*A1*d ≈ b1 - b2  => d = 0.5 * A1^{-1} (b1 - b2)
# Regularize when A1 is ill-conditioned.
# ---------------------------
def estimate_local_displacement(coeffs1, coeffs2, eps=1e-3):
    H, W, _ = coeffs1.shape
    flow = np.zeros((H, W, 2), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            p1 = coeffs1[y, x, :]
            p2 = coeffs2[y, x, :]
            A1, b1 = poly_to_Ab(p1)
            _, b2 = poly_to_Ab(p2)
            M = 2.0 * A1
            rhs = b1 - b2
            # regularize M by adding eps * trace(M) to diagonal
            M_reg = M + np.eye(2, dtype=np.float32) * (eps * (np.trace(M) + eps))
            # solve
            try:
                d = np.linalg.solve(M_reg, rhs) * 0.5
            except np.linalg.LinAlgError:
                d = np.zeros(2, dtype=np.float32)
            flow[y, x, 0] = d[0]
            flow[y, x, 1] = d[1]
    return flow

# ---------------------------
# Simplified multiscale Farneback-like pipeline
# ---------------------------
def farneback_simplified(img1, img2, levels=3, win=5):
    """
    img1, img2: float32 arrays normalized to 0..1
    levels: number of pyramid levels (1 = single scale)
    win: window size for polynomial fit (odd)
    """
    # build pyramids
    pyr1 = [img1.astype(np.float32)]
    pyr2 = [img2.astype(np.float32)]
    for i in range(1, levels):
        pyr1.append(pyr_down(pyr1[-1]))
        pyr2.append(pyr_down(pyr2[-1]))

    flow = None
    # coarse-to-fine
    for lvl in range(levels-1, -1, -1):
        I1 = pyr1[lvl]
        I2 = pyr2[lvl]
        H, W = I1.shape
        if flow is not None:
            flow = pyr_up(flow, (H, W)) * 2.0
            # warp I2 toward I1 using current estimate
            I2w = warp_image(I2, flow)
        else:
            flow = np.zeros((H, W, 2), dtype=np.float32)
            I2w = I2.copy()

        # fit quadratic coeffs on I1 and warped I2
        coeffs1 = fit_quadratic_field(I1, win=win)
        coeffs2 = fit_quadratic_field(I2w, win=win)
        # estimate local delta
        delta = estimate_local_displacement(coeffs1, coeffs2, eps=1e-3)
        # update flow
        flow = flow + delta
        # smooth flow a bit to reduce noise
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        flow[..., 0] = convolve2d(flow[..., 0], kernel, padding='reflect')
        flow[..., 1] = convolve2d(flow[..., 1], kernel, padding='reflect')
    return flow

# ---------------------------
# Visualization: convert flow to RGB using HSV mapping
# ---------------------------
def flow_to_rgb(flow, max_mag=None):
    H, W = flow.shape[:2]
    u = flow[..., 0]
    v = flow[..., 1]
    ang = np.arctan2(v, u)  # -pi..pi
    mag = np.sqrt(u * u + v * v)
    if max_mag is None:
        max_mag = np.percentile(mag, 96)
        if max_mag < 1e-6:
            max_mag = 1.0
    hue = (ang + np.pi) / (2 * np.pi)  # 0..1
    sat = np.clip(mag / max_mag, 0, 1)
    val = sat.copy()
    hsv = np.stack([hue, sat, val], axis=2)  # floats 0..1
    import matplotlib.colors as mcolors
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb_img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb_img

# ---------------------------
# Demo: try to load frame1.png / frame2.png or create synthetic pair
# ---------------------------
def demo(save_result=False):
    try:
        im1 = Image.open("frame1.png").convert("L")
        im2 = Image.open("frame2.png").convert("L")
        I1 = np.array(im1, dtype=np.float32) / 255.0
        I2 = np.array(im2, dtype=np.float32) / 255.0
        print("Loaded frame1.png and frame2.png from disk.")
    except Exception:
        print("No frame1.png/frame2.png — creating synthetic moving shapes.")
        H, W = 240, 320
        I1 = np.zeros((H, W), dtype=np.float32)
        # draw rectangles / blobs
        I1[60:140, 60:140] = 1.0
        I1[30:55, 190:260] = 0.8
        # shift by (dy,dx) = (4,6)
        I2 = np.roll(I1, shift=(4, 6), axis=(0, 1)).copy()
        # add slight texture/noise
        rng = np.random.RandomState(1)
        tex = (rng.normal(scale=0.03, size=(H, W))).astype(np.float32)
        I1 = np.clip(I1 + tex, 0.0, 1.0)
        I2 = np.clip(I2 + tex, 0.0, 1.0)

    print("Running simplified Farneback (this may be slow for large images)...")
    flow = farneback_simplified(I1, I2, levels=3, win=7)
    rgb = flow_to_rgb(flow)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1); plt.title("Frame 1"); plt.imshow(I1, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Frame 2"); plt.imshow(I2, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Estimated flow (HSV)"); plt.imshow(rgb); plt.axis('off')
    plt.tight_layout()
    plt.show()

    if save_result:
        Image.fromarray(rgb).save("farneback_flow_rgb.png")
        print("Saved farneback_flow_rgb.png")

if __name__ == "__main__":
    demo(save_result=True)
