from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import uuid
import threading
import logging

app = Flask(__name__)
CORS(app)

# ── Estado en memoria ──────────────────────────────────────────────────────
ultimo_dato   = None
sesiones      = {}
sesion_activa = None

# ── Endpoints de datos ─────────────────────────────────────────────────────

@app.route('/data', methods=['POST'])
def recibir_dato():
    global ultimo_dato
    dato = request.get_json()
    dato['timestamp'] = datetime.now().isoformat()

    # Guardar en sesión activa si existe
    if sesion_activa and sesion_activa in sesiones:
        sesiones[sesion_activa]['registros'].append(dato)

    ultimo_dato = dato
    return jsonify({'ok': True})


@app.route('/data/current', methods=['GET'])
def dato_actual():
    if ultimo_dato is None:
        return jsonify({'status': 'no_data'})
    return jsonify(ultimo_dato)


# ── Endpoints de sesión ────────────────────────────────────────────────────

@app.route('/session/start', methods=['POST'])
def iniciar_sesion():
    global sesion_activa
    body       = request.get_json()
    session_id = str(uuid.uuid4())

    sesiones[session_id] = {
        'session_id': session_id,
        'patient_id': body.get('patient_id', 'desconocido'),
        'notes':      body.get('notes', ''),
        'started_at': datetime.now().isoformat(),
        'ended_at':   None,
        'registros':  []
    }

    sesion_activa = session_id
    print(f"✅ Sesión iniciada: {session_id[:8]}... | Paciente: {body.get('patient_id')}")

    return jsonify({
        'session_id': session_id,
        'started_at': sesiones[session_id]['started_at']
    })


@app.route('/session/current', methods=['GET'])
def sesion_actual():
    if sesion_activa and sesion_activa in sesiones:
        return jsonify({
            'session_id': sesion_activa,
            'patient_id': sesiones[sesion_activa]['patient_id'],
            'started_at': sesiones[sesion_activa]['started_at']
        })
    return jsonify({'session_id': None})


@app.route('/session/<session_id>/end', methods=['POST'])
def terminar_sesion(session_id):
    global sesion_activa
    if session_id not in sesiones:
        return jsonify({'error': 'sesión no encontrada'}), 404

    sesiones[session_id]['ended_at'] = datetime.now().isoformat()
    total = len(sesiones[session_id]['registros'])
    sesion_activa = None

    print(f"🔴 Sesión terminada: {session_id[:8]}... | {total} registros")
    return jsonify({'ok': True, 'total_registros': total})


# ── Página principal ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── Arranque ───────────────────────────────────────────────────────────────

def iniciar_servidor():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
@app.route('/session/<session_id>/export', methods=['GET'])
def exportar_sesion(session_id):
    if session_id not in sesiones:
        return jsonify({'error': 'sesión no encontrada'}), 404
    
    sesion = sesiones[session_id]
    return jsonify(sesion)

if __name__ == '__main__':
    iniciar_servidor()