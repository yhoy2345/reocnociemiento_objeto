import cv2
from ultralytics import YOLO

# Cargar modelo entrenado (reemplaza con tu modelo .pt)
model = YOLO("yolov8n.pt")  # O "best.pt" si ya tienes uno entrenado

# Iniciar cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección
    results = model(frame, stream=True)

    # Dibujar resultados
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "celar", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar frame
    cv2.imshow("Detección en tiempo real", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()