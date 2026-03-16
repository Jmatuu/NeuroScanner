import cv2
import mediapipe as mp
import numpy as np
import time
import requests 
from datetime import datetime

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

        self.cap = cv2.VideoCapture(0)

        mp_face_mesh   = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Mapa de calor y dashboard
        self.mapa_calor     = None
        self.mostrar_calor  = False
        self.session_id_web = None

        # Plano cartesiano  ← al final del __init__
        self.gaze_plot = np.zeros((400, 400, 3), dtype=np.uint8)
        self.dibujar_ejes()
        
        # Contador de parpadeos
        self.parpadeos          = 0
        self.ojo_izq_cerrado    = False
        self.ojo_der_cerrado    = False

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

        offset_x = cx + radio_iris + 8
        cv2.putText(frame, f"{nombre}",                 (offset_x, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        cv2.putText(frame, f"Iris: ({cx},{cy})",         (offset_x, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 130, 0), 1)
        cv2.putText(frame, f"Pupila r={radio_pupila}px", (offset_x, cy),      cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, f"{estado} ({apertura}px)",   (offset_x, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_estado, 1)

        return cx, cy

    def dibujar_panel(self, frame, h, w):
        if not self.fijaciones:
            return

        total      = len(self.fijaciones)
        conteo     = {'OJOS': 0, 'NARIZ': 0, 'BOCA': 0, 'OTRO': 0}
        duraciones = []

        for f in self.fijaciones:
            if f['zona'] in conteo:
                conteo[f['zona']] += 1
            duraciones.append(f['duracion_ms'])

        dpf = round(np.mean(duraciones), 1)

        px, py = 10, h - 160
        cv2.rectangle(frame, (px, py), (px + 280, h - 10), (20, 20, 20), -1)
        cv2.rectangle(frame, (px, py), (px + 280, h - 10), (100, 100, 100), 1)

        cv2.putText(frame, "ESTADISTICAS TEA",           (px + 10, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 255, 255), 1)
        cv2.putText(frame, f"Total fijaciones: {total}", (px + 10, py + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Dur. promedio: {dpf} ms",   (px + 10, py + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        zonas_info = [
            ('OJOS',  conteo['OJOS'],  (255, 255, 0)),
            ('NARIZ', conteo['NARIZ'], (0, 255, 255)),
            ('BOCA',  conteo['BOCA'],  (0, 100, 255)),
        ]

        y_offset = py + 85
        for nombre_zona, cantidad, color in zonas_info:
            porcentaje  = round((cantidad / total) * 100) if total > 0 else 0
            largo_barra = int((cantidad / total) * 150)   if total > 0 else 0
            cv2.putText(frame, f"{nombre_zona}: {cantidad} ({porcentaje}%)",
                        (px + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.rectangle(frame, (px + 10, y_offset + 4),
                          (px + 10 + largo_barra, y_offset + 12), color, -1)
            y_offset += 28

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

        print(f"\n✅ Archivos guardados:")
        print(f"   📄 {nombre_csv}")
        print(f"   📋 {nombre_txt}")

    def send_to_dashboard(self, gaze_x, gaze_y, left_open, right_open, blink, direccion, zona, session_id=None):
        try:
            requests.post("http://localhost:8000/data", json={
                "gaze_x":             float(gaze_x),
                "gaze_y":             float(gaze_y),
                "left_eye_open":      bool(left_open),
                "right_eye_open":     bool(right_open),
                "blink_detected":     bool(blink),
                "emotion":            direccion,
                "zona_cara":          zona,
                "emotion_confidence": 0.9,
                "session_id":         session_id
            }, timeout=0.1)
        except:
            pass

    def actualizar_mapa_calor(self, gaze_x, gaze_y, h, w):
        # Convertir coordenadas normalizadas a píxeles
        px = int(gaze_x * w)
        py = int(gaze_y * h)

        # Agregar calor en esa zona (radio de influencia = 30px)
        radio = 30
        for y in range(max(0, py - radio), min(h, py + radio)):
            for x in range(max(0, px - radio), min(w, px + radio)):
                distancia = np.sqrt((x - px)**2 + (y - py)**2)
                if distancia < radio:
                    intensidad = 1 - (distancia / radio)
                    self.mapa_calor[y, x] += intensidad * 0.1

    def dibujar_mapa_calor(self, frame):
        if self.mapa_calor is None:
            return frame

        # Normalizar el mapa
        mapa_norm = self.mapa_calor.copy()
        if mapa_norm.max() > 0:
            mapa_norm = (mapa_norm / mapa_norm.max() * 255).astype(np.uint8)

        # Aplicar color
        mapa_color = cv2.applyColorMap(mapa_norm, cv2.COLORMAP_JET)

        # Mezclar con el frame original
        resultado = cv2.addWeighted(frame, 0.6, mapa_color, 0.4, 0)
        return resultado
    
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
    
    def ejecutar(self):
        with self.face_mesh:     # ← 8 espacios ✅
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: 
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Inicializar mapa de calor con el tamaño del frame
                if self.mapa_calor is None:
                    self.mapa_calor = np.zeros((h, w), dtype=np.float32)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                # Valores por defecto si no hay rostro válido
                gaze_x_norm = 0.0
                gaze_y_norm = 0.0
                direccion   = "CENTRO"

                if results.multi_face_landmarks:
                    for face in results.multi_face_landmarks:
                        lm = face.landmark

                        # Verificar si el rostro es válido
                        if not self.rostro_valido(lm, h, w):
                            self.dibujar_advertencia(frame, h, w)
                            continue

                        zonas = self.definir_zonas(lm, h, w)

                        zona_ojos, zona_nariz, zona_boca = zonas

                        cv2.rectangle(frame, (zona_ojos[0],  zona_ojos[1]),  (zona_ojos[2],  zona_ojos[3]),  (255, 255, 0), 1)
                        cv2.rectangle(frame, (zona_nariz[0], zona_nariz[1]), (zona_nariz[2], zona_nariz[3]), (0, 255, 255), 1)
                        cv2.rectangle(frame, (zona_boca[0],  zona_boca[1]),  (zona_boca[2],  zona_boca[3]),  (0, 100, 255), 1)

                        cv2.putText(frame, "Ojos",  (zona_ojos[0],  zona_ojos[1]  - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        cv2.putText(frame, "Nariz", (zona_nariz[0], zona_nariz[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        cv2.putText(frame, "Boca",  (zona_boca[0],  zona_boca[1]  - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

                        self.dibujar_ojo(frame, lm, self.PARPADO_IZQ, self.IRIS_IZQ, h, w, "Ojo Izq")
                        self.dibujar_ojo(frame, lm, self.PARPADO_DER, self.IRIS_DER, h, w, "Ojo Der")

                        dir_v, dir_h, dx, dy = self.estimar_gaze(lm, self.IRIS_IZQ, self.PARPADO_IZQ, h, w)
                        direccion, gaze_x_norm, gaze_y_norm = self.calcular_direccion_9zonas(lm, h, w)

                        # Zona cara según dirección vertical
                        if "ARRIBA" in direccion:
                            zona_detectada = "OTRO"
                        elif direccion == "CENTRO":
                            zona_detectada = "OJOS"
                        elif "ABAJO" in direccion and abs(gaze_y_norm) < 0.5:
                            zona_detectada = "NARIZ"
                        else:
                            zona_detectada = "BOCA"

                        # Actualizar mapa de calor
                        self.actualizar_mapa_calor(gaze_x_norm, gaze_y_norm, h, w)

                        # Enviar al dashboard
                        abierto_izq, _ = self.ojo_abierto(lm, self.PARPADO_IZQ, h, w)
                        abierto_der, _ = self.ojo_abierto(lm, self.PARPADO_DER, h, w)
                        parpadeo = self.detectar_parpadeo(abierto_izq, abierto_der)

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

                        cv2.putText(frame, f"X:{gaze_x_norm:.2f} Y:{gaze_y_norm:.2f}",
                                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        cv2.putText(frame, f"Parpadeos: {self.parpadeos}",
                                    (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

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

                            self.zona_actual = zona_detectada
                            self.inicio_fijacion = tiempo_ahora

                self.dibujar_panel(frame, h, w)

                if self.mostrar_calor and self.mapa_calor is not None:
                    frame = self.dibujar_mapa_calor(frame)

                cv2.putText(frame, "H: mapa calor | Q: salir", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if results.multi_face_landmarks:
                    self.actualizar_plano(gaze_x_norm, gaze_y_norm, direccion)

                cv2.imshow("Plano de Mirada", self.gaze_plot)
                cv2.imshow("Detector de Ojos", frame)

                tecla = cv2.waitKey(1) & 0xFF

                # Detectar si cerraron la ventana con la X
                if cv2.getWindowProperty('Detector de Ojos', cv2.WND_PROP_VISIBLE) < 1:
                    break
                if tecla == ord('q') or tecla == ord('Q'):
                    break
                elif tecla == ord('h') or tecla == ord('H'):
                    self.mostrar_calor = not self.mostrar_calor
                elif tecla == ord('r') or tecla == ord('R'):
                    self.gaze_plot = np.zeros((400, 400, 3), dtype=np.uint8)
                    self.dibujar_ejes()

        self.guardar_resultados()
        self.cap.release()
        cv2.destroyAllWindows() 
        