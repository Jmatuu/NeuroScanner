import threading
import queue
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import requests

class DetectorOcular:

    def __init__(self):
        self.PARPADO_IZQ = [33, 160, 158, 133, 153, 144]
        self.PARPADO_DER = [362, 385, 387, 263, 373, 380]
        self.IRIS_IZQ    = [468, 469, 470, 471, 472]
        self.IRIS_DER    = [473, 474, 475, 476, 477]
        self.fijaciones          = []
        self.inicio_fijacion     = None
        self.zona_actual         = None
        self.TIEMPO_MIN_FIJACION = 0.1
        self._t_zonas    = 0
        self._t_stats    = 0
        self._zona_cache = "OJOS"
        # Sistema de emociones nuevo
        self._calibrado          = False
        self._buffer_calibracion = []
        self._buffer_max         = 60   # Zona segura: luz y distancia controladas → menos frames necesarios
        self._linea_base         = {}
        self._emocion_cache      = "CALIBRANDO..."
        self._confianza_cache    = 0.0
        self._t_emocion          = 0
        self._persistencia       = {
    'EAR': 0, 'CORRUGADOR': 0, 'TENSION_LABIAL': 0, 'MAR': 0
}


        self.cap = cv2.VideoCapture(0) # Configuracion de la camara 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  360)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        mp_face_mesh   = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Dashboard
        self.session_id_web = None
        self._cola_envio = queue.Queue(maxsize=10)
        self._ultimo_envio = 0
        self._ultimo_left  = True
        self._ultimo_right = True
        self._hilo_envio = threading.Thread(target=self._worker_envio, daemon=True)
        self._hilo_envio.start()

        # Hilo que sincroniza session_id_web con el servidor
        self._hilo_sesion = threading.Thread(target=self._worker_sesion, daemon=True)
        self._hilo_sesion.start()

        # Plano cartesiano  ← al final del __init__
        self.gaze_plot = np.zeros((400, 400, 3), dtype=np.uint8)
        self.dibujar_ejes()
        
        # Contador de parpadeos
        self.parpadeos          = 0
        self.ojo_izq_cerrado    = False
        self.ojo_der_cerrado    = False

        # Deteccion de stimming
        self._stimming_activo        = False
        self._stimming_cache         = "NINGUNO"
        # Parpadeos en rafaga
        self._historial_parpadeos    = []   # timestamps de parpadeos recientes
        self._RAFAGA_VENTANA         = 2.0  # segundos
        self._RAFAGA_MIN             = 3    # parpadeos minimos en la ventana
        # Balanceo de cabeza
        self._historial_yaw          = []   # (timestamp, yaw)
        self._historial_roll         = []   # (timestamp, roll)
        self._BALANCEO_VENTANA       = 3.0  # segundos
        self._BALANCEO_CAMBIOS_MIN   = 4    # cambios de direccion minimos
        self._BALANCEO_UMBRAL        = 8.0  # grados minimos por cambio
        # Gestos faciales repetitivos
        self._historial_labial       = []   # (timestamp, tension_labial)
        self._historial_corrugador   = []   # (timestamp, corrugador)
        self._GESTO_VENTANA          = 3.0  # segundos
        self._GESTO_CICLOS_MIN       = 3    # ciclos minimos detectados
        self._ema = {
    'EAR': None, 'CORRUGADOR': None, 
    'TENSION_LABIAL': None, 'MAR': None
}
        self._alpha_ema = 0.3

    def ojo_abierto(self, landmarks, indices_parpado, h, w):
        sup = landmarks[indices_parpado[1]]
        inf = landmarks[indices_parpado[5]]
        dy  = abs((sup.y - inf.y) * h)
        return dy > 3, round(dy, 1)
    
    def rostro_valido(self, landmarks, h, w):
        try:
            # Usar coordenadas normalizadas directamente
            lp = landmarks[468]  # centro iris izquierdo
            rp = landmarks[473]  # centro iris derecho
            
            # Verificar que los landmarks están dentro del frame
            if not (0 < lp.x < 1 and 0 < lp.y < 1):
                return False
            if not (0 < rp.x < 1 and 0 < rp.y < 1):
                return False
                
            # Verificar distancia entre puntos del iris izquierdo
            p0 = landmarks[self.IRIS_IZQ[0]]
            p1 = landmarks[self.IRIS_IZQ[2]]
            radio_izq = abs(p0.x - p1.x) + abs(p0.y - p1.y)
            
            # Verificar distancia entre puntos del iris derecho
            p2 = landmarks[self.IRIS_DER[0]]
            p3 = landmarks[self.IRIS_DER[2]]
            radio_der = abs(p2.x - p3.x) + abs(p2.y - p3.y)

            return radio_izq > 0.005 and radio_der > 0.005
        except:
            return False
        
    def dibujar_advertencia(self, frame, h, w):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, "⚠ ROSTRO PARCIALMENTE OCLUIDO",
                    (w//2 - 280, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, "Pausando analisis...",
                    (w//2 - 150, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
    def definir_zonas(self, landmarks, h, w):
        ojos_puntos = [33, 133, 362, 263, 70, 300]
        xs = [int(landmarks[i].x * w) for i in ojos_puntos]
        ys = [int(landmarks[i].y * h) for i in ojos_puntos]
        zona_ojos = (min(xs) - 10, min(ys) - 15, max(xs) + 10, max(ys) + 15)

        nariz_puntos = [168, 4, 294, 64]
        xs = [int(landmarks[i].x * w) for i in nariz_puntos]
        ys = [int(landmarks[i].y * h) for i in nariz_puntos]
        zona_nariz = (min(xs) - 10, min(ys), max(xs) + 10, max(ys) + 10)

        boca_puntos = [61, 291, 0, 17]
        xs = [int(landmarks[i].x * w) for i in boca_puntos]
        ys = [int(landmarks[i].y * h) for i in boca_puntos]
        zona_boca = (min(xs) - 10, min(ys) - 5, max(xs) + 10, max(ys) + 10)

        return zona_ojos, zona_nariz, zona_boca

    def zona_mirada(self, cx, cy, zonas):
        zona_ojos, zona_nariz, zona_boca = zonas
        if zona_ojos[0]  < cx < zona_ojos[2]  and zona_ojos[1]  < cy < zona_ojos[3]:
            return "OJOS"
        elif zona_nariz[0] < cx < zona_nariz[2] and zona_nariz[1] < cy < zona_nariz[3]:
            return "NARIZ"
        elif zona_boca[0]  < cx < zona_boca[2]  and zona_boca[1]  < cy < zona_boca[3]:
            return "BOCA"
        else:
            return "OTRO"

    def estimar_gaze(self, landmarks, indices_iris, indices_parpado, h, w):
        puntos_iris = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices_iris]
        cx_iris    = np.mean([p[0] for p in puntos_iris])
        cy_iris    = np.mean([p[1] for p in puntos_iris])
        radio_iris = np.linalg.norm(np.array(puntos_iris[0]) - np.array([cx_iris, cy_iris]))

        sup = landmarks[indices_parpado[1]]
        inf = landmarks[indices_parpado[5]]
        izq = landmarks[indices_parpado[0]]
        der = landmarks[indices_parpado[3]]

        cx_ojo = ((izq.x + der.x) / 2) * w
        cy_ojo = ((sup.y + inf.y) / 2) * h

        dx = (cx_iris - cx_ojo) / (radio_iris + 1e-5)
        dy = (cy_iris - cy_ojo) / (radio_iris + 1e-5)

        dir_v = "ARRIBA" if dy < -0.2 else "ABAJO" if dy > 0.2 else "CENTRO"
        dir_h = "DERECHA" if dx < -0.2 else "IZQUIERDA" if dx > 0.2 else "CENTRO"

        return dir_v, dir_h, round(dx, 2), round(dy, 2)
    
    def calcular_direccion_9zonas(self, landmarks, h, w):
        try:
            # Ojo izquierdo
            lp          = landmarks[468]
            l_outer     = landmarks[33]
            l_inner     = landmarks[133]
            l_top       = landmarks[159]
            l_bottom    = landmarks[145]
            l_width     = max(0.001, l_inner.x - l_outer.x)
            l_height    = max(0.001, l_bottom.y - l_top.y)
            l_rel_x     = (lp.x - l_outer.x) / l_width
            l_rel_y     = (lp.y - l_top.y)   / l_height

            # Ojo derecho
            rp          = landmarks[473]
            r_inner     = landmarks[362]
            r_outer     = landmarks[263]
            r_top       = landmarks[386]
            r_bottom    = landmarks[374]
            r_width     = max(0.001, r_outer.x - r_inner.x)
            r_height    = max(0.001, r_bottom.y - r_top.y)
            r_rel_x     = (rp.x - r_inner.x) / r_width
            r_rel_y     = (rp.y - r_top.y)   / r_height

            # Promedio ambos ojos
            avg_x = (l_rel_x + r_rel_x) / 2
            avg_y = (l_rel_y + r_rel_y) / 2

            # Coordenadas normalizadas -1 a +1
            gaze_x = (avg_x - 0.5) * 2
            gaze_y = (0.5 - avg_y) * 2

            # Clasificar en 9 zonas
            dx = avg_x - 0.5
            dy = avg_y - 0.5
            umbral_h = 0.12
            umbral_v = 0.08

            if abs(dx) < umbral_h and abs(dy) < umbral_v:
                direccion = "CENTRO"
            elif dx < -umbral_h and abs(dy) < umbral_v:
                direccion = "IZQUIERDA"
            elif dx > umbral_h and abs(dy) < umbral_v:
                direccion = "DERECHA"
            elif abs(dx) < umbral_h and dy < -umbral_v:
                direccion = "ARRIBA"
            elif abs(dx) < umbral_h and dy > umbral_v:
                direccion = "ABAJO"
            elif dx < -umbral_h and dy < -umbral_v:
                direccion = "ARRIBA_IZQ"
            elif dx < -umbral_h and dy > umbral_v:
                direccion = "ABAJO_IZQ"
            elif dx > umbral_h and dy < -umbral_v:
                direccion = "ARRIBA_DER"
            elif dx > umbral_h and dy > umbral_v:
                direccion = "ABAJO_DER"
            else:
                direccion = "CENTRO"

            return direccion, round(gaze_x, 4), round(gaze_y, 4)

        except:
            return "CENTRO", 0.0, 0.0  # ← si algo falla, devuelve CENTRO
        
    def dibujar_ojo(self, frame, landmarks, indices_parpado, indices_iris, h, w, nombre):

        puntos_parpado = []
        for idx in indices_parpado:
            lm = landmarks[idx]
            puntos_parpado.append([int(lm.x * w), int(lm.y * h)])
        puntos_parpado = np.array(puntos_parpado, np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [puntos_parpado], (200, 200, 200))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [puntos_parpado], True, (0, 255, 255), 1)

        puntos_iris = []

        for idx in indices_iris:
            lm = landmarks[idx]
            puntos_iris.append((int(lm.x * w), int(lm.y * h)))

        cx = int(np.mean([p[0] for p in puntos_iris]))
        cy = int(np.mean([p[1] for p in puntos_iris]))
        radio_iris = int(np.linalg.norm(np.array(puntos_iris[0]) - np.array([cx, cy])))

        overlay2 = frame.copy()
        cv2.circle(overlay2, (cx, cy), radio_iris, (255, 130, 0), -1)
        cv2.addWeighted(overlay2, 0.2, frame, 0.8, 0, frame)
        cv2.circle(frame, (cx, cy), radio_iris, (255, 130, 0), 1)

        radio_pupila = max(2, radio_iris // 3)
        overlay3 = frame.copy()
        cv2.circle(overlay3, (cx, cy), radio_pupila, (0, 0, 0), -1)
        cv2.addWeighted(overlay3, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, (cx, cy), radio_pupila, (50, 50, 50), 1)

        abierto, apertura = self.ojo_abierto(landmarks, indices_parpado, h, w)
        estado       = "Abierto" if abierto else "Cerrado"
        color_estado = (0, 255, 0) if abierto else (0, 0, 255)

        return cx, cy

    def dibujar_panel(self, frame, h, w):
        if not self.fijaciones:
            return

        total  = len(self.fijaciones)
        conteo = {'OJOS': 0, 'NARIZ': 0, 'BOCA': 0, 'OTRO': 0}
        for f in self.fijaciones:
            if f['zona'] in conteo:
                conteo[f['zona']] += 1

        px, py = w - 200, 10
        cv2.rectangle(frame, (px, py), (w - 5, py + 80), (20, 20, 20), -1)
        cv2.putText(frame, f"Fij: {total}",
                    (px + 5, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"OJOS:{conteo['OJOS']} NAR:{conteo['NARIZ']} BOC:{conteo['BOCA']}",
                    (px + 5, py + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Parp: {self.parpadeos}",
                    (px + 5, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    def guardar_resultados(self):
        if not self.fijaciones:
            print("No hay fijaciones registradas para guardar.")
            return

        fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        total      = len(self.fijaciones)
        conteo     = {'OJOS': 0, 'NARIZ': 0, 'BOCA': 0, 'OTRO': 0}
        duraciones = []

        for f in self.fijaciones:
            if f['zona'] in conteo:
                conteo[f['zona']] += 1
            duraciones.append(f['duracion_ms'])

        dpf       = round(np.mean(duraciones), 1)
        porc_ojos = round((conteo['OJOS'] / total) * 100)

        if porc_ojos < 40:
            patron = "PATRON TEA: poco contacto visual (OJOS < 40%)"
        elif porc_ojos > 60:
            patron = "PATRON TIPICO: buen contacto visual (OJOS > 60%)"
        else:
            patron = "PATRON INTERMEDIO: requiere mas analisis"

        nombre_csv = f"sesion_{fecha_hora}.csv"
        with open(nombre_csv, 'w') as archivo:
            archivo.write("zona,duracion_ms,timestamp\n")
            for fij in self.fijaciones:
                archivo.write(f"{fij['zona']},{fij['duracion_ms']},{fij['timestamp']}\n")

        nombre_txt = f"resumen_{fecha_hora}.txt"
        with open(nombre_txt, 'w', encoding='utf-8') as archivo:
            archivo.write("===== RESUMEN SESION TEA =====\n")
            archivo.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            archivo.write(f"Total fijaciones:  {total}\n")
            archivo.write(f"Duracion promedio: {dpf} ms\n\n")
            archivo.write(f"OJOS:  {conteo['OJOS']}  fijaciones ({porc_ojos}%)\n")
            archivo.write(f"NARIZ: {conteo['NARIZ']} fijaciones ({round((conteo['NARIZ'] / total) * 100)}%)\n")
            archivo.write(f"BOCA:  {conteo['BOCA']}  fijaciones ({round((conteo['BOCA']  / total) * 100)}%)\n\n")
            archivo.write(f"{patron}\n")

        print(f"\n Archivos guardados:")
        print(f"   📄 {nombre_csv}")
        print(f"   📋 {nombre_txt}")

    def _worker_sesion(self):
        """Consulta /session/current cada 5 segundos y actualiza session_id_web."""
        while True:
            try:
                res = requests.get("http://localhost:8000/session/current", timeout=1)
                data = res.json()
                nuevo_id = data.get('session_id')
                if nuevo_id != self.session_id_web:
                    self.session_id_web = nuevo_id
                    if nuevo_id:
                        print(f"✅ Sesion detectada: {nuevo_id[:8]}...")
                    else:
                        print("⏸ Sin sesion activa")
            except:
                pass
            time.sleep(5)

    def _worker_envio(self):
        ultimo_gaze = 0
        while True:
            try:
                dato = self._cola_envio.get(timeout=1)
                es_evento = dato.get('blink_detected') or dato.get('_forzar')
                ahora = time.time()
                # Cada 500ms — detección de emociones
                
                if es_evento or (ahora - ultimo_gaze >= 0.5):
                    requests.post("http://localhost:8000/data", json=dato, timeout=0.5)
                    if not es_evento:
                        ultimo_gaze = ahora
            except queue.Empty:
                pass
            except:
                pass

    def send_to_dashboard(self, gaze_x, gaze_y, left_open, right_open, blink, direccion, zona, session_id=None):
        dato = {
            "gaze_x":             float(gaze_x),
            "gaze_y":             float(gaze_y),
            "left_eye_open":      bool(left_open),
            "right_eye_open":     bool(right_open),
            "blink_detected":     bool(blink),
            "total_parpadeos":    self.parpadeos,
            "emotion":            self._emocion_cache,
            "emotion_confidence": float(self._confianza_cache),
            "zona_cara":          zona,
            "stimming":           self._stimming_cache,
            "stimming_activo":    bool(self._stimming_activo),
            "session_id":         session_id,
            "_forzar":            bool(left_open != self._ultimo_left or right_open != self._ultimo_right)
        }

        self._ultimo_left  = left_open
        self._ultimo_right = right_open

        try:
            # Si es evento importante, limpiar cola y meter el nuevo
            if dato['blink_detected'] or dato['_forzar']:
                while not self._cola_envio.empty():
                    self._cola_envio.get_nowait()
            self._cola_envio.put_nowait(dato)
        except queue.Full:
            pass
    
    def dibujar_ejes(self):
        cv2.line(self.gaze_plot, (20, 200), (380, 200), (100, 100, 100), 1)
        cv2.line(self.gaze_plot, (200, 20), (200, 380), (100, 100, 100), 1)
        cv2.putText(self.gaze_plot, "IZQ",    (10, 195),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(self.gaze_plot, "DER",    (350, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(self.gaze_plot, "ARRIBA", (210, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(self.gaze_plot, "ABAJO",  (210, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(self.gaze_plot, "(0,0)",  (205, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
       

    def actualizar_plano(self, gaze_x, gaze_y, direccion):
        plot_x = int(200 + gaze_x * 180)
        plot_y = int(200 - gaze_y * 180)
        plot_x = max(10, min(390, plot_x))
        plot_y = max(10, min(390, plot_y))

        # Color según dirección
        colores = {
            'CENTRO':     (255, 0, 255),
            'ARRIBA':     (0, 0, 255),
            'ABAJO':      (0, 255, 0),
            'IZQUIERDA':  (0, 255, 255),
            'DERECHA':    (255, 255, 0),
            'ARRIBA_IZQ': (0, 128, 255),
            'ARRIBA_DER': (128, 0, 255),
            'ABAJO_IZQ':  (0, 255, 128),
            'ABAJO_DER':  (128, 255, 0),
        }
        color = colores.get(direccion, (255, 255, 255))
        cv2.circle(self.gaze_plot, (plot_x, plot_y), 4, color, -1)   
    
    def detectar_parpadeo(self, abierto_izq, abierto_der):
        parpadeo_detectado = False

        # Ojo izquierdo
        if not abierto_izq and not self.ojo_izq_cerrado:
            self.ojo_izq_cerrado = True
        elif abierto_izq and self.ojo_izq_cerrado:
            self.ojo_izq_cerrado = False
            parpadeo_detectado   = True

        # Ojo derecho
        if not abierto_der and not self.ojo_der_cerrado:
            self.ojo_der_cerrado = True
        elif abierto_der and self.ojo_der_cerrado:
            self.ojo_der_cerrado = False
            parpadeo_detectado   = True

        if parpadeo_detectado:
            self.parpadeos += 1

        return parpadeo_detectado
    
    def _calcular_metricas(self, landmarks):
        try:
            # Distancia interpupilar para normalizar
            p33  = landmarks[33]
            p263 = landmarks[263]
            dip  = np.sqrt((p33.x - p263.x)**2 + (p33.y - p263.y)**2)
            if dip < 0.001:
                return None

            # EAR ojo izquierdo
            ear_v = np.sqrt((landmarks[159].x - landmarks[145].x)**2 +
                            (landmarks[159].y - landmarks[145].y)**2)
            ear_h = np.sqrt((landmarks[33].x  - landmarks[133].x)**2 +
                            (landmarks[33].y  - landmarks[133].y)**2)
            ear   = (ear_v / ear_h) / dip if ear_h > 0 else 0

            # Corrugador — promedio de distancia entre pares de landmarks especificos
            # Landmarks 107,105 (ceja izq) y 336,334 (ceja der) recomendados por especialista
            corr_izq = np.sqrt((landmarks[107].x - landmarks[105].x)**2 +
                               (landmarks[107].y - landmarks[105].y)**2)
            corr_der = np.sqrt((landmarks[336].x - landmarks[334].x)**2 +
                               (landmarks[336].y - landmarks[334].y)**2)
            corrugador = ((corr_izq + corr_der) / 2) / dip

            # Tension labial — ancho de comisuras
            tension_labial = np.sqrt((landmarks[61].x - landmarks[291].x)**2 +
                                     (landmarks[61].y - landmarks[291].y)**2) / dip

            # MAR — apertura vertical de boca
            mar = np.sqrt((landmarks[13].x - landmarks[14].x)**2 +
                          (landmarks[13].y - landmarks[14].y)**2) / dip
            
            # Definir metricas como diccionario con valores iniciales
            metricas = {
                'EAR': ear,
                'CORRUGADOR': corrugador,
                'TENSION_LABIAL': tension_labial,
                'MAR': mar
            }
            
            # Aplicar filtro EMA
            for key in ['EAR', 'CORRUGADOR', 'TENSION_LABIAL', 'MAR']:
                if self._ema[key] is None:
                    self._ema[key] = metricas[key]
                else:
                    self._ema[key] = round(
                        self._alpha_ema * metricas[key] + 
                        (1 - self._alpha_ema) * self._ema[key], 4
                    )
                metricas[key] = self._ema[key]
            
            return metricas

        except:
            return None
    def _calcular_angulos_cabeza(self, landmarks):
        try:
            # Landmarks rigidos
            nariz     = landmarks[1]
            menton    = landmarks[152]
            ojo_izq   = landmarks[33]
            ojo_der   = landmarks[263]
            temp_izq  = landmarks[234]
            temp_der  = landmarks[454]

            # Yaw — giro izquierda/derecha
            dist_izq = abs(nariz.x - temp_izq.x)
            dist_der = abs(nariz.x - temp_der.x)
            yaw = (dist_izq - dist_der) / (dist_izq + dist_der + 1e-5) * 90

            # Roll — inclinacion lateral
            roll = np.degrees(np.arctan2(
                ojo_der.y - ojo_izq.y,
                ojo_der.x - ojo_izq.x
            ))

            # Pitch — inclinacion arriba/abajo
            frente = landmarks[10]
            pitch = np.degrees(np.arctan2(
                menton.y - frente.y,
                menton.z - frente.z + 1e-5
            ))

            return round(yaw, 1), round(pitch, 1), round(roll, 1)

        except:
            return 0.0, 0.0, 0.0

    def _rostro_frontal(self, landmarks):
        try:
            # Zona segura: camara fija a 40-50cm, altura de ojos → angulos mas estrictos
            MAX_YAW   = 20   # giro izquierda/derecha
            MAX_PITCH = 15   # inclinacion arriba/abajo
            MAX_ROLL  = 10   # inclinacion lateral
            yaw, pitch, roll = self._calcular_angulos_cabeza(landmarks)
            return abs(yaw) < MAX_YAW and abs(pitch) < MAX_PITCH and abs(roll) < MAX_ROLL
        except:
            return True
    
    def _actualizar_calibracion(self, landmarks):
        if self._calibrado:
            return

        if not self._rostro_frontal(landmarks):
            
            return

        metricas = self._calcular_metricas(landmarks)
        if metricas is None:
            
            return

        self._buffer_calibracion.append(metricas)


        if len(self._buffer_calibracion) >= self._buffer_max:
            for key in ['EAR', 'CORRUGADOR', 'TENSION_LABIAL', 'MAR']:
                valores = [f[key] for f in self._buffer_calibracion]
                self._linea_base[key] = round(float(np.median(valores)), 4)

            self._calibrado = True
            self._emocion_cache = "NEUTRAL"
            print(f"Calibracion completa: {self._linea_base}")

    def _detectar_stimming(self, landmarks, parpadeo_detectado):
        """Detecta tres tipos de stimming: rafaga de parpadeos, balanceo de cabeza
        y gestos faciales repetitivos. Retorna (tipo_stimming, activo)."""
        ahora = time.time()

        # ── 1. Parpadeos en rafaga ─────────────────────────────────────────
        if parpadeo_detectado:
            self._historial_parpadeos.append(ahora)

        # Limpiar parpadeos fuera de la ventana
        self._historial_parpadeos = [
            t for t in self._historial_parpadeos
            if ahora - t <= self._RAFAGA_VENTANA
        ]
        rafaga = len(self._historial_parpadeos) >= self._RAFAGA_MIN

        # ── 2. Balanceo de cabeza ──────────────────────────────────────────
        try:
            yaw, pitch, roll = self._calcular_angulos_cabeza(landmarks)

            self._historial_yaw.append((ahora, yaw))
            self._historial_roll.append((ahora, roll))

            # Limpiar fuera de ventana
            self._historial_yaw  = [(t, v) for t, v in self._historial_yaw
                                    if ahora - t <= self._BALANCEO_VENTANA]
            self._historial_roll = [(t, v) for t, v in self._historial_roll
                                    if ahora - t <= self._BALANCEO_VENTANA]

            def contar_cambios(historial, umbral):
                """Cuenta cuantas veces la señal cambia de direccion."""
                if len(historial) < 3:
                    return 0
                valores = [v for _, v in historial]
                cambios = 0
                for i in range(1, len(valores) - 1):
                    delta_ant = valores[i]     - valores[i - 1]
                    delta_sig = valores[i + 1] - valores[i]
                    if abs(delta_ant) >= umbral and delta_ant * delta_sig < 0:
                        cambios += 1
                return cambios

            cambios_yaw  = contar_cambios(self._historial_yaw,  self._BALANCEO_UMBRAL)
            cambios_roll = contar_cambios(self._historial_roll, self._BALANCEO_UMBRAL)
            balanceo = (cambios_yaw >= self._BALANCEO_CAMBIOS_MIN or
                        cambios_roll >= self._BALANCEO_CAMBIOS_MIN)
        except:
            balanceo = False

        # ── 3. Gestos faciales repetitivos ────────────────────────────────
        try:
            metricas = self._calcular_metricas(landmarks)
            if metricas:
                self._historial_labial.append((ahora, metricas['TENSION_LABIAL']))
                self._historial_corrugador.append((ahora, metricas['CORRUGADOR']))

            # Limpiar fuera de ventana
            self._historial_labial     = [(t, v) for t, v in self._historial_labial
                                          if ahora - t <= self._GESTO_VENTANA]
            self._historial_corrugador = [(t, v) for t, v in self._historial_corrugador
                                          if ahora - t <= self._GESTO_VENTANA]

            def contar_ciclos(historial):
                """Cuenta picos alternantes (subida + bajada = 1 ciclo)."""
                if len(historial) < 4:
                    return 0
                valores  = [v for _, v in historial]
                media    = np.mean(valores)
                std      = np.std(valores)
                if std < 0.002:   # señal demasiado plana → no hay gesto
                    return 0
                ciclos   = 0
                encima   = valores[0] > media
                for v in valores[1:]:
                    if encima and v < media:
                        ciclos += 1
                        encima  = False
                    elif not encima and v > media:
                        encima  = True
                return ciclos

            ciclos_labial     = contar_ciclos(self._historial_labial)
            ciclos_corrugador = contar_ciclos(self._historial_corrugador)
            gesto_repetitivo  = (ciclos_labial     >= self._GESTO_CICLOS_MIN or
                                 ciclos_corrugador >= self._GESTO_CICLOS_MIN)
        except:
            gesto_repetitivo = False

        # ── Resultado final — prioridad: rafaga > balanceo > gesto ────────
        if rafaga:
            return "RAFAGA_PARPADEOS", True
        elif balanceo:
            return "BALANCEO_CABEZA", True
        elif gesto_repetitivo:
            return "GESTO_REPETITIVO", True
        else:
            return "NINGUNO", False

    def _detectar_emocion(self, landmarks):
        if not self._calibrado:
            return "CALIBRANDO...", 0.0

        metricas = self._calcular_metricas(landmarks)
        if metricas is None:
            return self._emocion_cache, self._confianza_cache

        lb = self._linea_base

        # Calcular variaciones porcentuales respecto a linea base
        var_ear    = (metricas['EAR']            - lb['EAR'])            / (lb['EAR']            + 1e-5) * 100
        var_corr   = (metricas['CORRUGADOR']     - lb['CORRUGADOR'])     / (lb['CORRUGADOR']     + 1e-5) * 100
        var_labial = (metricas['TENSION_LABIAL'] - lb['TENSION_LABIAL']) / (lb['TENSION_LABIAL'] + 1e-5) * 100

        # MAR — umbral absoluto (evita valores extremos por baseline cercano a cero)
        # Zona segura: 40-50cm, luz controlada → umbrales fijos confiables
        MAR_UMBRAL_ABIERTO = 0.015   # boca entreabierta
        MAR_UMBRAL_AMPLIO  = 0.030   # boca claramente abierta / vocalizacion

        # Activaciones con umbrales del especialista
        ear_bajo   = var_ear    < -30
        ear_alto   = var_ear    >  20
        corrugador = var_corr   >   8
        labial     = var_labial >  30
        mar_alto   = metricas['MAR'] > MAR_UMBRAL_AMPLIO

        # Filtro de persistencia — contar frames consecutivos
        self._persistencia['EAR']           = self._persistencia['EAR']           + 1 if ear_bajo or ear_alto else 0
        self._persistencia['CORRUGADOR'] = self._persistencia['CORRUGADOR']       + 1 if corrugador else 0    
        self._persistencia['TENSION_LABIAL']= self._persistencia['TENSION_LABIAL']+ 1 if labial                else 0
        self._persistencia['MAR']           = self._persistencia['MAR']           + 1 if mar_alto              else 0

        FRAMES_MIN = 3

        ear_confirmado    = self._persistencia['EAR']            >= FRAMES_MIN
        corr_confirmado   = self._persistencia['CORRUGADOR']     >= FRAMES_MIN
        labial_confirmado = self._persistencia['TENSION_LABIAL'] >= FRAMES_MIN
        mar_confirmado    = self._persistencia['MAR']            >= FRAMES_MIN

        # Firmas emocionales combinadas
        if ear_bajo and corr_confirmado:
            return "SOBRECARGA", round(abs(var_corr) / 100, 2)
        elif ear_alto and not mar_alto:
            return "SORPRESA", round(abs(var_ear) / 100, 2)
        elif corr_confirmado and not ear_bajo:
            return "FRUSTRACION", round(abs(var_corr) / 100, 2)
        elif labial_confirmado:
            return "ANSIEDAD", round(abs(var_labial) / 100, 2)
        elif mar_alto and not ear_bajo:
            return "VOCALIZACION", round(min(metricas['MAR'] / MAR_UMBRAL_AMPLIO, 1.0), 2)
        elif ear_confirmado and ear_alto:
            return "HIPER-FOCO", round(abs(var_ear) / 100, 2)
        else:
            return "NEUTRAL", 1.0
        
    def ejecutar(self):
        with self.face_mesh:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                gaze_x_norm = 0.0
                gaze_y_norm = 0.0
                direccion   = "CENTRO"
                ahora       = time.time()

                if results.multi_face_landmarks:
                    for face in results.multi_face_landmarks:
                        lm = face.landmark

                        if not self.rostro_valido(lm, h, w):
                            self.dibujar_advertencia(frame, h, w)
                            continue

                        # Cada 150ms — calcular zonas, direccion y calibracion
                        if ahora - self._t_zonas >= 0.1:
                            zonas = self.definir_zonas(lm, h, w)
                            zona_ojos, zona_nariz, zona_boca = zonas
                            direccion, gaze_x_norm, gaze_y_norm = self.calcular_direccion_9zonas(lm, h, w)
                            self._actualizar_calibracion(lm)
                         # Deteccion de emocion
                            if self._calibrado:
                                self._emocion_cache, self._confianza_cache = self._detectar_emocion(lm)
                            if "ARRIBA" in direccion:
                                self._zona_cache = "OTRO"
                            elif direccion == "CENTRO":
                                self._zona_cache = "OJOS"
                            elif "ABAJO" in direccion and abs(gaze_y_norm) < 0.5:
                                self._zona_cache = "NARIZ"
                            else:
                                self._zona_cache = "BOCA"

                            self._t_zonas = ahora

                        zona_detectada = self._zona_cache

                        # Dibujar zonas en cada frame
                        cv2.rectangle(frame, (zona_ojos[0],  zona_ojos[1]),  (zona_ojos[2],  zona_ojos[3]),  (255, 255, 0), 1)
                        cv2.rectangle(frame, (zona_nariz[0], zona_nariz[1]), (zona_nariz[2], zona_nariz[3]), (0, 255, 255), 1)
                        cv2.rectangle(frame, (zona_boca[0],  zona_boca[1]),  (zona_boca[2],  zona_boca[3]),  (0, 100, 255), 1)

                        cv2.putText(frame, "Ojos",  (zona_ojos[0],  zona_ojos[1]  - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        cv2.putText(frame, "Nariz", (zona_nariz[0], zona_nariz[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        cv2.putText(frame, "Boca",  (zona_boca[0],  zona_boca[1]  - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

                        cv2.putText(frame, f"Emocion: {self._emocion_cache}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                        self.dibujar_ojo(frame, lm, self.PARPADO_IZQ, self.IRIS_IZQ, h, w, "Ojo Izq")
                        self.dibujar_ojo(frame, lm, self.PARPADO_DER, self.IRIS_DER, h, w, "Ojo Der")

                        abierto_izq, _ = self.ojo_abierto(lm, self.PARPADO_IZQ, h, w)
                        abierto_der, _ = self.ojo_abierto(lm, self.PARPADO_DER, h, w)
                        parpadeo = self.detectar_parpadeo(abierto_izq, abierto_der)

                        # Deteccion de stimming (cada frame, usa parpadeo ya calculado)
                        self._stimming_cache, self._stimming_activo = self._detectar_stimming(lm, parpadeo)

                        self.send_to_dashboard(
                            gaze_x_norm, gaze_y_norm,
                            abierto_izq, abierto_der,
                            parpadeo, direccion,
                            zona_detectada,
                            self.session_id_web
                        )

                        cv2.putText(frame, f"Dir: {direccion}",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Zona: {zona_detectada}",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                        # Mostrar stimming en pantalla si está activo
                        if self._stimming_activo:
                            cv2.putText(frame, f"STIMMING: {self._stimming_cache}",
                                        (10, 230), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 255), 2)

                        tiempo_ahora = time.time()

                        if zona_detectada == self.zona_actual:
                            if self.inicio_fijacion is not None:
                                duracion = tiempo_ahora - self.inicio_fijacion
                                if duracion >= self.TIEMPO_MIN_FIJACION:
                                    cv2.putText(frame, f"Fijando: {self.zona_actual}",
                                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, (255, 255, 0), 2)
                        else:
                            if self.zona_actual is not None and self.inicio_fijacion is not None:
                                duracion = tiempo_ahora - self.inicio_fijacion
                                if duracion >= self.TIEMPO_MIN_FIJACION:
                                    self.fijaciones.append({
                                        'zona': self.zona_actual,
                                        'duracion_ms': round(duracion * 1000, 1),
                                        'timestamp': datetime.now().strftime("%H:%M:%S")
                                    })
                                    # Limitar a 500 fijaciones para no crecer sin limite en RAM
                                    if len(self.fijaciones) > 500:
                                        self.fijaciones.pop(0)
                            self.zona_actual     = zona_detectada
                            self.inicio_fijacion = tiempo_ahora

                if ahora - self._t_stats >= 0.5:
                    self._t_stats = ahora

                self.dibujar_panel(frame, h, w)

                cv2.putText(frame, "Q: salir | R: reset plano", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if results.multi_face_landmarks:
                    self.actualizar_plano(gaze_x_norm, gaze_y_norm, direccion)

                cv2.imshow("Plano de Mirada", self.gaze_plot)
                cv2.imshow("Detector de Ojos", frame)

                tecla = cv2.waitKey(1) & 0xFF

                if cv2.getWindowProperty('Detector de Ojos', cv2.WND_PROP_VISIBLE) < 1:
                    break
                if tecla == ord('q') or tecla == ord('Q'):
                    break
                elif tecla == ord('r') or tecla == ord('R'):
                    self.gaze_plot = np.zeros((400, 400, 3), dtype=np.uint8)
                    self.dibujar_ejes()

        self.guardar_resultados()
        self.cap.release()
        cv2.destroyAllWindows()