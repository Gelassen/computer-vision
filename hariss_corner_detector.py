import numpy as np
from PIL import Image
from convolution import convolve2d

def harris_corner_detector(image, k=0.04, window_size=3, threshold=1e-5):
    """
    image        : grayscale image as 2D numpy array (H x W)
    k            : Harris detector free parameter, typical 0.04-0.06
    window_size  : size of window for summing gradients
    threshold    : threshold for corner response
    
    returns:
    corners      : binary mask of corners (1 where corner, 0 elsewhere)
    R            : Harris response matrix
    """
    # --- 1. Вычисляем градиенты Ix и Iy ---
    # простые ядра Собеля
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    Ix = convolve2d(image, sobel_x, padding='reflect')
    Iy = convolve2d(image, sobel_y, padding='reflect')

    # --- 2. Вычисляем продукты градиентов ---
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # --- 3. Скользящее суммирование по окну ---
    # можно использовать свёртку с ядром единиц
    w = np.ones((window_size, window_size))
    Sx2 = convolve2d(Ix2, w, padding='reflect')
    Sy2 = convolve2d(Iy2, w, padding='reflect')
    Sxy = convolve2d(Ixy, w, padding='reflect')

    # --- 4. Вычисляем матрицу отклика Харриса ---
    H, W = image.shape
    R = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            M = np.array([[Sx2[y, x], Sxy[y, x]],
                          [Sxy[y, x], Sy2[y, x]]])
            R[y, x] = np.linalg.det(M) - k * (np.trace(M) ** 2)

    # --- 5. Порог для детекции углов ---
    corners = (R > threshold).astype(np.uint8)

    return corners, R

# --- Пример использования ---
def launch():
    # Открываем изображение
    img = Image.open("assets/comp_vision_test.jpg").convert("L")
    image = np.array(img, dtype=np.float32) / 255.0

    corners, R = harris_corner_detector(image, k=0.05, window_size=3, threshold=1e-4)
    
    # Сохраняем карту отклика (для визуализации)
    R_img = (R - R.min()) / (R.max() - R.min()) * 255
    Image.fromarray(R_img.astype(np.uint8)).save("assets/comp_vision_test_harris_R.jpg")

    # Сохраняем бинарную маску углов
    Image.fromarray((corners*255).astype(np.uint8)).save("assets/comp_vision_test_harris_corners.jpg")

if __name__ == "__main__":
    launch()
