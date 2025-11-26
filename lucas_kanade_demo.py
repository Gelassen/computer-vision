import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# --------------------------------------------------------
# 1. Генерация тестовых данных: квадрат сдвигается вправо
# --------------------------------------------------------
H, W = 200, 200
I1 = np.zeros((H, W), dtype=np.float32)
I2 = np.zeros((H, W), dtype=np.float32)

I1[60:140, 60:140] = 1.0
I2[60:140, 65:145] = 1.0   # сдвиг на 5 пикселей вправо

# --------------------------------------------------------
# 2. Градиенты яркости Ix, Iy, It
# --------------------------------------------------------
Ix = ndimage.sobel(I1, axis=1)
Iy = ndimage.sobel(I1, axis=0)
It = I2 - I1

# --------------------------------------------------------
# 3. Lucas–Kanade: решаем систему в каждом окне
# --------------------------------------------------------
win = 15  # окно 15×15 — можно увеличить

u = np.zeros_like(I1)
v = np.zeros_like(I1)

half = win // 2

for y in range(half, H - half):
    for x in range(half, W - half):
        Ix_win = Ix[y-half:y+half+1, x-half:x+half+1].flatten()
        Iy_win = Iy[y-half:y+half+1, x-half:x+half+1].flatten()
        It_win = It[y-half:y+half+1, x-half:x+half+1].flatten()

        A = np.vstack((Ix_win, Iy_win)).T
        b = -It_win

        # нормальное уравнение A^T A v = A^T b
        ATA = A.T @ A
        ATb = A.T @ b

        if np.linalg.det(ATA) > 1e-6:   # проверка: есть ли информация?
            flow = np.linalg.inv(ATA) @ ATb
            u[y, x] = flow[0]
            v[y, x] = flow[1]

# --------------------------------------------------------
# 4. Визуализация потоков
# --------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.imshow(I1, cmap='gray')
plt.quiver(np.arange(0, W, 10), 
           np.arange(0, H, 10), 
           u[::10, ::10], 
           v[::10, ::10], 
           color='red')
plt.title("Lucas–Kanade Optical Flow (from scratch)")
plt.show()
