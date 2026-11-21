import cv2
import os

video_path = "video.mp4"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Guarda en PNG para evitar compresión con pérdida (JPEG), 
    # o usa .jpg con calidad 100 si necesitas ahorrar espacio en disco
    filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(filename, frame)
    frame_idx += 1

cap.release()
print(f"Extracción completa: {frame_idx} frames guardados en '{output_dir}'.")