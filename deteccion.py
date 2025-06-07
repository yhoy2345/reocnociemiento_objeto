import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import time

class ObjectDetector:
    def __init__(self):
        # Configuración CEL CAMRA
        self.ip_address = "192.168.1.4"
        self.port = "8080"
        self.rtsp_url = f"http://{self.ip_address}:{self.port}/video"
        self.model = YOLO("yolov8n.pt")
        self.video_cap = None
        self.dark_mode = False
        self.colors = {
            'light': {
                'background': (240, 240, 240),
                'text': (20, 20, 20),
                'box': (0, 100, 255),
                'highlight': (0, 200, 255)
            },
            'dark': {
                'background': (40, 40, 40),
                'text': (240, 240, 240),
                'box': (0, 255, 200),
                'highlight': (0, 255, 150)
            }
        }
        self.current_colors = self.colors['dark']
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.box_thickness = 2
        self.fps_smoothing = 0.9
        self.smoothed_fps = 30
        
    def initialize_camera(self):
        """Inicializa la conexión con la cámara IP"""
        self.video_cap = cv2.VideoCapture(self.rtsp_url)
        if not self.video_cap.isOpened():
            raise ConnectionError("No se pudo conectar a la cámara IP")
            
    def toggle_dark_mode(self):
        """Alterna entre modo claro y oscuro"""
        self.dark_mode = not self.dark_mode
        self.current_colors = self.colors['dark' if self.dark_mode else 'light']
        
    def draw_detection(self, frame, x, y, w, h, class_name, confidence):
        """Dibuja un cuadro de detección en el frame"""
        # Coordenadas del rectángulo
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        
        # Dibujar rectángulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.current_colors['box'], self.box_thickness)
        
        # Texto de la etiqueta
        label = f"{class_name} {confidence:.0%}"
        
        # Tamaño del texto
        (text_width, text_height), _ = cv2.getTextSize(
            label, self.font, self.font_scale, self.font_thickness)
        
        # Dibujar fondo para el texto
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), self.current_colors['box'], -1)
        
        # Dibujar texto
        cv2.putText(frame, label, (x1, y1 - 5), 
                   self.font, self.font_scale, 
                   self.current_colors['background'], self.font_thickness)
        
        # Punto central
        cv2.circle(frame, (x, y), 3, self.current_colors['highlight'], -1)
        
    def draw_stats(self, frame, fps, detections_count):
        """Dibuja las estadísticas en el frame"""
        stats_bg_height = 50
        cv2.rectangle(frame, (0, 0), (frame.shape[1], stats_bg_height), 
                      self.current_colors['background'], -1)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                    self.font, 0.8, self.current_colors['highlight'], 2)
        
        # Conteo de detecciones
        detections_text = f"Objetos: {detections_count}"
        cv2.putText(frame, detections_text, (frame.shape[1] - 150, 30), 
                    self.font, 0.8, self.current_colors['highlight'], 2)
        
        # Modo
        mode_text = "Modo: Oscuro" if self.dark_mode else "Modo: Claro"
        cv2.putText(frame, mode_text, (frame.shape[1] // 2 - 70, 30), 
                    self.font, 0.8, self.current_colors['highlight'], 2)
        
    def process_frame(self, frame):
        """Procesa un frame y devuelve las detecciones"""
        results = self.model(frame)[0]
        detections = []
        
        for detection in results.boxes:
            x, y, w, h = detection.xywh[0].numpy().astype(int)
            confidence = detection.conf.item()
            class_id = int(detection.cls.item())
            class_name = self.model.names[class_id]
            
            if confidence > 0.5:
                detections.append((x, y, w, h, class_name, confidence))
                
        return detections
        
    def run(self):
        """Ejecuta el bucle principal de detección"""
        self.initialize_camera()
        
        prev_time = time.time()
        
        while True:
            # Leer frame
            ret, frame = self.video_cap.read()
            if not ret:
                print("Error: No se pudo leer el frame")
                break
                
            # Voltear el frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Procesar detecciones
            detections = self.process_frame(frame)
            
            # Dibujar detecciones
            for detection in detections:
                self.draw_detection(frame, *detection)
                
            # Calcular FPS suavizado
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            self.smoothed_fps = (self.fps_smoothing * self.smoothed_fps + 
                                 (1 - self.fps_smoothing) * fps)
            prev_time = curr_time
            
            # Dibujar estadísticas
            self.draw_stats(frame, self.smoothed_fps, len(detections))
            
            # Mostrar frame
            cv2.imshow("Detección de Objetos - YOLOv8", frame)
            
            # Manejo de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.toggle_dark_mode()
                
        # Liberar recursos
        self.video_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run()