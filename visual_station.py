import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "assets/test-comp-vision-tracking.mp4"
CONF_TH = 0.15  # Еще ниже порог для мяча
BALL_CLASSES = {"sports ball"}

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Simple motion detection
# -----------------------------
def simple_motion_check(prev, curr, bbox, threshold=20):
    """Простая проверка движения в области bbox"""
    x1, y1, x2, y2 = bbox
    
    # Вырезаем ROI
    prev_roi = prev[y1:y2, x1:x2]
    curr_roi = curr[y1:y2, x1:x2]
    
    if prev_roi.size == 0 or curr_roi.size == 0:
        return False
    
    # Сравниваем гистограммы
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(prev_gray, curr_gray)
    motion_score = np.mean(diff)
    
    return motion_score > threshold

# -----------------------------
# Main loop - SIMPLE VERSION
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, prev_frame = cap.read()
if not ret:
    print("Video error"); exit()

# Пропускаем первые кадры для стабилизации
for _ in range(10):
    ret, prev_frame = cap.read()

track_history = {}
track_id_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция ВСЕХ объектов
    results = model(frame, conf=0.2)[0]  # conf ниже для лучшей детекции
    
    current_detections = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]

        if name in BALL_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            current_detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'conf': conf
            })

    # Простой трекинг по расстоянию
    used_tracks = set()
    
    for det in current_detections:
        best_track_id = None
        min_distance = 50  # Максимальное расстояние для ассоциации
        
        for track_id, track_info in track_history.items():
            if track_id in used_tracks:
                continue
                
            last_center = track_info['center']
            current_center = det['center']
            
            distance = np.sqrt((last_center[0] - current_center[0])**2 + 
                             (last_center[1] - current_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                best_track_id = track_id
        
        if best_track_id is not None:
            # Обновляем существующий трек
            track_history[best_track_id] = {
                'center': det['center'],
                'bbox': det['bbox'],
                'conf': det['conf']
            }
            used_tracks.add(best_track_id)
            track_id = best_track_id
        else:
            # Новый трек
            track_id = track_id_counter
            track_id_counter += 1
            track_history[track_id] = {
                'center': det['center'],
                'bbox': det['bbox'],
                'conf': det['conf']
            }
    
    # Удаляем старые треки
    track_history = {tid: info for tid, info in track_history.items() 
                    if tid in used_tracks or np.random.random() > 0.1}  # иногда оставляем для перепроверки

    # Отрисовка
    for track_id, track_info in track_history.items():
        x1, y1, x2, y2 = track_info['bbox']
        conf = track_info['conf']
        
        # Проверка движения (только для отрисовки)
        is_moving = simple_motion_check(prev_frame, frame, (x1, y1, x2, y2))
        
        color = (0, 255, 0) if is_moving else (0, 128, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ball #{track_id} ({conf:.2f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    prev_frame = frame.copy()
    
    cv2.imshow("Tennis Ball Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()