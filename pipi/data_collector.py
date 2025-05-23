import cv2
import os
import time
import json

# Configuración
OBJECT_NAME = "estheban"  # Nombre del objeto a detectar
DATASET_PATH = "dataset"
IMAGES_PER_OBJECT = 30  # Número máximo de imágenes a capturar
ROI_SIZE = 300  # Tamaño de la región de interés
ANNOTATIONS_FILE = f"{DATASET_PATH}/{OBJECT_NAME}/annotations.json"  # Archivo para guardar anotaciones

# Crear directorios necesarios
os.makedirs(f"{DATASET_PATH}/{OBJECT_NAME}", exist_ok=True)

# Variables globales
annotations = {}
drawing = False
ix, iy = -1, -1
current_boxes = []
current_image = None
image_path = ""

def draw_boxes(image, boxes):
    """Dibuja los bounding boxes en la imagen"""
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def mouse_callback(event, x, y, flags, param):
    """Callback para eventos del mouse"""
    global ix, iy, drawing, current_boxes, current_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = current_image.copy()
            cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 1)
            draw_boxes(temp_image, current_boxes)
            cv2.imshow("Anotación - Dibuja bounding boxes", temp_image)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - ix) > 10 and abs(y - iy) > 10:  # Ignorar clicks pequeños
            current_boxes.append([min(ix, x), min(iy, y), max(ix, x), max(iy, y)])
            current_image = draw_boxes(current_image.copy(), current_boxes)
            cv2.imshow("Anotación - Dibuja bounding boxes", current_image)

def save_annotation():
    """Guarda las anotaciones actuales"""
    global current_boxes, image_path, annotations
    
    if current_boxes and image_path:
        annotations[os.path.basename(image_path)] = current_boxes
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(annotations, f)
        print(f"Anotaciones guardadas para {os.path.basename(image_path)}")

def reset_annotation():
    """Reinicia las anotaciones para la imagen actual"""
    global current_boxes, current_image
    current_boxes = []
    current_image = original_image.copy()
    cv2.imshow("Anotación - Dibuja bounding boxes", current_image)

# Iniciar cámara
cap = cv2.VideoCapture(0)
count = 0

print(f"Capturando {IMAGES_PER_OBJECT} imágenes para: {OBJECT_NAME}")
print("Instrucciones:")
print("1. Presiona 's' para empezar a capturar")
print("2. Presiona 'c' para capturar una imagen")
print("3. Dibuja bounding boxes con el mouse")
print("4. Presiona 'g' para guardar anotaciones")
print("5. Presiona 'r' para reiniciar anotaciones")
print("6. Presiona 'q' para terminar")

# Modo de vista previa
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Dibujar área de interés
    cv2.rectangle(frame, 
                 (center_x - ROI_SIZE//2, center_y - ROI_SIZE//2),
                 (center_x + ROI_SIZE//2, center_y + ROI_SIZE//2),
                 (0, 255, 0), 2)
    
    cv2.imshow("Posiciona el objeto - Presiona 's' para empezar", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.destroyAllWindows()
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Modo de captura y anotación
cv2.namedWindow("Anotación - Dibuja bounding boxes")
cv2.setMouseCallback("Anotación - Dibuja bounding boxes", mouse_callback)

while count < IMAGES_PER_OBJECT:
    ret, frame = cap.read()
    if not ret:
        continue
    
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Extraer ROI
    roi = frame[center_y - ROI_SIZE//2:center_y + ROI_SIZE//2, 
                center_x - ROI_SIZE//2:center_x + ROI_SIZE//2]
    
    # Mostrar vista previa
    display_frame = frame.copy()
    cv2.rectangle(display_frame, 
                 (center_x - ROI_SIZE//2, center_y - ROI_SIZE//2),
                 (center_x + ROI_SIZE//2, center_y + ROI_SIZE//2),
                 (0, 255, 0), 2)
    
    cv2.putText(display_frame, f"Capturadas: {count}/{IMAGES_PER_OBJECT}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Vista previa de captura", display_frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('c'):  # Capturar imagen
        timestamp = int(time.time())
        image_path = f"{DATASET_PATH}/{OBJECT_NAME}/{OBJECT_NAME}_{timestamp}_{count}.jpg"
        cv2.imwrite(image_path, roi)
        print(f"Imagen guardada: {image_path}")
        
        # Preparar para anotación
        original_image = roi.copy()
        current_image = roi.copy()
        current_boxes = []
        cv2.imshow("Anotación - Dibuja bounding boxes", current_image)
        
    elif key == ord('g') and current_image is not None:  # Guardar anotaciones
        save_annotation()
        count += 1
        current_image = None
        
    elif key == ord('r') and current_image is not None:  # Reiniciar anotaciones
        reset_annotation()
        
    elif key == ord('q'):  # Salir
        break

cap.release()
cv2.destroyAllWindows()
print("Captura completada!")
print(f"Anotaciones guardadas en: {ANNOTATIONS_FILE}")