
import numpy as np
import os

from PIL import Image

def convolve2d(image, kernel, padding='zero'):
    """
    image  : 2D numpy array (H x W)
    kernel : 2D numpy array (kH x kW)
    padding: 'zero', 'reflect', 'edge'
    """
    H, W = image.shape
    kH, kW = kernel.shape
    
    # 1) Перевернём ядро (настоящее свёртывание)
    kernel_flipped = np.flipud(np.fliplr(kernel))
    
    # 2) Паддинг
    pad_h = kH // 2
    pad_w = kW // 2

    if padding == 'zero':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    elif padding == 'reflect':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    elif padding == 'edge':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        raise ValueError("Unknown padding mode")

    # 3) Выход
    output = np.zeros_like(image, dtype=np.float32)

    # 4) Наивная свёртка двойным циклом
    for y in range(H):
        for x in range(W):
            region = padded[y:y+kH, x:x+kW]
            output[y, x] = np.sum(region * kernel_flipped)

    return output

def save_image(array, path):
    # Нормализуем в диапазон 0–255
    arr_norm = array - array.min()
    arr_norm = arr_norm / arr_norm.max()
    arr_uint8 = (arr_norm * 255).astype(np.uint8)

    # Создаём директорию, если её нет
    os.makedirs(os.path.dirname(path), exist_ok=True)

    Image.fromarray(arr_uint8).save(path)

def get_sobel_kernel(): 
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    return sobel_x

def get_gaussian_kernel():
    gaussian = np.array([
        [1, 4, 6, 4, 1],
        [4,16,24,16, 4],
        [6,24,36,24, 6],
        [4,16,24,16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float32) / 256.0

    return gaussian


def launch():
    img = Image.open("./assets/comp_vision_test.jpg").convert("L")
    image = np.array(img, dtype=np.float32)

    kernel = get_sobel_kernel()
    
    Ix = convolve2d(image, kernel)

    save_image(Ix, "./assets/comp_vision_test_modified.jpg")
    print("Сохранено в ./assets/comp_vision_test_modified.jpg")



if __name__ == "__main__":
    launch()