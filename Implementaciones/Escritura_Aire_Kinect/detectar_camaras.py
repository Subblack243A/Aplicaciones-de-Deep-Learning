"""
Script para detectar todas las cámaras disponibles en el sistema.
Esto nos ayuda a saber si el Kinect está siendo reconocido y en qué índice.
"""
import cv2

print("=" * 50)
print("  Detectando cámaras disponibles...")
print("=" * 50)

camaras_encontradas = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            backend = cap.getBackendName()
            camaras_encontradas.append(i)
            print(f"\nCámara #{i}:")
            print(f"   Resolución: {w} x {h}")
            print(f"   Backend: {backend}")
            
            # Mostrar la imagen de cada cámara por 3 segundos
            cv2.putText(frame, f"Camara #{i} - {w}x{h}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(f"Camara #{i}", frame)
        cap.release()

if not camaras_encontradas:
    print("\nNo se encontró ninguna cámara.")
else:
    print(f"\n{'=' * 50}")
    print(f"  Se encontraron {len(camaras_encontradas)} cámara(s): {camaras_encontradas}")
    print(f"  Si el Kinect tiene resolución 640x480, ese es el índice correcto.")
    print(f"{'=' * 50}")

print("\nPresiona cualquier tecla en las ventanas de las cámaras para cerrar...")
cv2.waitKey(0)
cv2.destroyAllWindows()
