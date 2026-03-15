import threading
from servidor import iniciar_servidor
from detector import DetectorOcular

# Arrancar servidor en hilo separado
hilo_servidor = threading.Thread(target=iniciar_servidor, daemon=True)
hilo_servidor.start()

print("🌐 Servidor iniciado en http://localhost:8000")
print("👁️  Iniciando eye tracking...")

# Arrancar detector
detector = DetectorOcular()
detector.ejecutar()