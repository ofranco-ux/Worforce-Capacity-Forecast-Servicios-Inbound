import os
import sys
import math
import gc
import re
import json
import threading
import hashlib
import time
import sqlite3
import io
import csv
from functools import lru_cache
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, make_response, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np

# Dependencias locales del forecast. Se distribuyen junto al servidor para que
# el calendario de México y el modelo de machine learning estén siempre activos.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(BASE_DIR, 'vendor')
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import holidays
from sklearn.ensemble import RandomForestRegressor

CACHE_FILE_LLAMADAS = os.path.join(BASE_DIR, 'forecast_cache_llamadas.json')
CACHE_FILE_CHAT = os.path.join(BASE_DIR, 'forecast_cache_chat.json')
CONFIG_FILE = os.path.join(BASE_DIR, 'wfm_config.json') 
EXCEL_DEFAULT = os.path.join(BASE_DIR, 'historico.xlsx')
WFM_ACTION_LOG_FILE = os.path.join(BASE_DIR, 'wfm_action_log.json')
WFM_ROSTER_DB = os.path.join(BASE_DIR, 'wfm_roster.db')
_WFM_ROSTER_LOCK = threading.RLock()
_WFM_ACTION_LOG_LOCK = threading.RLock()

# Caché en memoria para evitar volver a leer/parsear JSON grandes cada vez que
# el usuario cambia entre Llamadas, Chat y Consolidado. El disco queda como
# respaldo entre reinicios, pero la respuesta normal sale de RAM.
_FORECAST_MEMORY_CACHE = {'llamadas': None, 'chat': None}
_FORECAST_CACHE_GENERATION = {'llamadas': 0, 'chat': 0}
_FORECAST_CACHE_LOCK = threading.RLock()
_EXCEL_INFO_CACHE = {}
_ASSISTANT_PLAN_CACHE = {}
_ASSISTANT_PLAN_CACHE_LOCK = threading.RLock()
_ASSISTANT_PLAN_CACHE_MAX = 64
_ASSISTANT_PLAN_CACHE_TTL = 600

def _cache_path(mode):
    return CACHE_FILE_CHAT if str(mode).lower() == 'chat' else CACHE_FILE_LLAMADAS

def _guardar_cache_forecast(mode, data):
    mode = 'chat' if str(mode).lower() == 'chat' else 'llamadas'
    with _FORECAST_CACHE_LOCK:
        _FORECAST_MEMORY_CACHE[mode] = data
        _FORECAST_CACHE_GENERATION[mode] += 1
        generation = _FORECAST_CACHE_GENERATION[mode]

    # Persistir en segundo plano evita que json.dump bloquee la respuesta HTTP.
    # Si se lanza un cálculo nuevo antes de terminar, el resultado anterior no
    # sobreescribe al más reciente.
    def _persistir():
        try:
            payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            with _FORECAST_CACHE_LOCK:
                if generation != _FORECAST_CACHE_GENERATION[mode]:
                    return
                target = _cache_path(mode)
                tmp = f"{target}.{generation}.tmp"
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(payload)
                os.replace(tmp, target)
        except Exception as e:
            print(f"No se pudo persistir caché {mode}: {e}")

    threading.Thread(target=_persistir, daemon=True, name=f'wfm-cache-{mode}').start()

def _leer_cache_forecast(mode):
    mode = 'chat' if str(mode).lower() == 'chat' else 'llamadas'
    with _FORECAST_CACHE_LOCK:
        mem = _FORECAST_MEMORY_CACHE.get(mode)
    if isinstance(mem, list) and mem:
        return mem

    target = _cache_path(mode)
    if not os.path.exists(target):
        return None
    try:
        with open(target, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            with _FORECAST_CACHE_LOCK:
                _FORECAST_MEMORY_CACHE[mode] = data
            return data
    except Exception:
        pass
    return None

def _excel_signature(excel_path):
    try:
        st = os.stat(excel_path)
        return (os.path.abspath(excel_path), st.st_mtime_ns, st.st_size)
    except Exception:
        return (os.path.abspath(excel_path), None, None)

def _cache_excel_info_get(excel_path, key):
    return _EXCEL_INFO_CACHE.get((_excel_signature(excel_path), key))

def _cache_excel_info_set(excel_path, key, value):
    sig = _excel_signature(excel_path)
    # Mantener sólo entradas de la versión actual del Excel.
    for old_key in list(_EXCEL_INFO_CACHE):
        if old_key[0][0] == sig[0] and old_key[0] != sig:
            _EXCEL_INFO_CACHE.pop(old_key, None)
    _EXCEL_INFO_CACHE[(sig, key)] = value
    return value

# =====================================================================
# 🧨 EXTERMINADOR DE CACHÉ
# =====================================================================
for cache_file in [CACHE_FILE_LLAMADAS, CACHE_FILE_CHAT]:
    try:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Borrando caché viejo: {cache_file}")
    except Exception as e:
        print(f"No se pudo borrar {cache_file}: {str(e)}")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

VENTANAS_SERVICIO = {
    'ambulancia servicios': {'inicio': 0 * 60, 'fin': 24 * 60},
    'asignación hogar': {'inicio': 0 * 60, 'fin': 24 * 60},
    'asignacion hogar': {'inicio': 0 * 60, 'fin': 24 * 60},
    'asignación vial': {'inicio': 0 * 60, 'fin': 24 * 60},
    'asignacion vial': {'inicio': 0 * 60, 'fin': 24 * 60},
    'coppel servicios': {'inicio': 0 * 60, 'fin': 24 * 60},
    'liverpool servicios': {'inicio': 0 * 60, 'fin': 24 * 60},
    'multicampañas': {'inicio': 0 * 60, 'fin': 24 * 60},
    'multicampanas': {'inicio': 0 * 60, 'fin': 24 * 60},
    'seguimiento hogar': {'inicio': 0 * 60, 'fin': 24 * 60},
    'seguimiento vial': {'inicio': 0 * 60, 'fin': 24 * 60},
    'suburbia servicios': {'inicio': 0 * 60, 'fin': 24 * 60},
    'experiencias liverpool': {'inicio': 9 * 60, 'fin': 21 * 60},
    'experiencias suburbia': {'inicio': 9 * 60, 'fin': 21 * 60},
    'retenciones suburbia': {'inicio': 9 * 60, 'fin': 20 * 60},
    'retenciones liverpool': {'inicio': 9 * 60, 'fin': 20 * 60}
}

# En Ambulancia Servicios, el 30 % de las atenciones requiere a dos personas
# simultáneamente: una con el cliente y otra gestionando el servicio de ambulancia.
REGLA_DOBLE_COBERTURA_AMBULANCIA = {
    'campana': 'ambulancia servicios',
    'porcentaje_volumen': 0.30,
    'personas_por_atencion': 2,
}

def es_ambulancia_servicios(campana):
    camp_key = re.sub(r'\s+', ' ', str(campana).strip().lower())
    campana_objetivo = REGLA_DOBLE_COBERTURA_AMBULANCIA['campana']
    return bool(camp_key) and (campana_objetivo in camp_key or camp_key in campana_objetivo)

def factor_cobertura_ambulancia(campana):
    if not es_ambulancia_servicios(campana):
        return 1.0
    regla = REGLA_DOBLE_COBERTURA_AMBULANCIA
    return 1.0 + regla['porcentaje_volumen'] * (regla['personas_por_atencion'] - 1)

def normalizar_nombre_campana(campana):
    return re.sub(r'\s+', ' ', str(campana or '').strip().lower())

def normalizar_config_campanas(config):
    if not isinstance(config, dict):
        return {}
    normalizada = {}
    for key, value in config.items():
        if not isinstance(value, dict):
            continue
        nombre = value.get('campaign') or value.get('campana') or key
        norm = normalizar_nombre_campana(nombre)
        if norm:
            normalizada[norm] = value
    return normalizada

def objetivos_campana(campana, target_sl, target_time, campaign_settings=None):
    settings = normalizar_config_campanas(campaign_settings)
    cfg = settings.get(normalizar_nombre_campana(campana), {})
    sl = clean_num(cfg.get('targetSl', cfg.get('target_sl', target_sl)), target_sl)
    asa = clean_num(cfg.get('targetTime', cfg.get('target_time', target_time)), target_time)
    sl = min(100.0, max(1.0, float(sl)))
    asa = max(1.0, float(asa))
    return sl, asa

def forzar_cuadre_dashboard(df_final):
    if df_final.empty:
        return df_final

    # Mismo criterio de cuadre, pero evitando filtrar y reagrupar todo el
    # DataFrame una vez por mes y una vez por día. En horizontes largos esta
    # parte era una de las más costosas.
    first_idx = (
        df_final.reset_index()
        .groupby(['Fecha', 'Intervalo'], sort=True)['index']
        .first()
    )

    picos_camp_mes = (
        df_final.groupby(['Mes', 'Campaña'], sort=True)['Agentes_Requeridos']
        .max()
        .groupby(level=0)
        .sum()
    )
    totales_intervalo_mes = df_final.groupby(
        ['Mes', 'Fecha', 'Intervalo'], sort=True
    )['Agentes_Requeridos'].sum()

    if not totales_intervalo_mes.empty:
        claves_pico_mes = totales_intervalo_mes.groupby(level=0).idxmax()
        for mes, clave in claves_pico_mes.items():
            pico_actual = float(totales_intervalo_mes.loc[clave])
            diferencia = float(picos_camp_mes.get(mes, 0)) - pico_actual
            if diferencia > 0:
                _, fecha_pico, hora_pico = clave
                idx = first_idx.get((fecha_pico, hora_pico))
                if idx is not None:
                    df_final.loc[idx, 'Agentes_Requeridos'] += diferencia

    # El cuadre diario se calcula después del mensual, igual que antes, para
    # conservar el mismo orden de efectos sobre Agentes_Requeridos.
    picos_camp_dia = (
        df_final.groupby(['Fecha', 'Campaña'], sort=True)['Agentes_Requeridos']
        .max()
        .groupby(level=0)
        .sum()
    )
    totales_intervalo_dia = df_final.groupby(
        ['Fecha', 'Intervalo'], sort=True
    )['Agentes_Requeridos'].sum()

    if not totales_intervalo_dia.empty:
        claves_pico_dia = totales_intervalo_dia.groupby(level=0).idxmax()
        for fecha, clave in claves_pico_dia.items():
            pico_actual = float(totales_intervalo_dia.loc[clave])
            diferencia = float(picos_camp_dia.get(fecha, 0)) - pico_actual
            if diferencia > 0:
                _, hora_pico = clave
                idx = first_idx.get((fecha, hora_pico))
                if idx is not None:
                    df_final.loc[idx, 'Agentes_Requeridos'] += diferencia

    return df_final

def pronosticar_macro_campana(df_diario_campana, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls):
    sub = df_diario_campana.sort_values(col_fecha).copy()
    sub['dia_semana'] = sub[col_fecha].dt.weekday
    dow_median = {}
    for i in range(7):
        vols = sub[(sub['dia_semana'] == i) & (sub[col_calls] > 0)][col_calls].tail(5)
        dow_median[i] = vols.median() if len(vols) > 0 else sub[col_calls].mean()
            
    recent_mean = sub[col_calls].tail(14).mean()
    preds_finales = []
    fecha_actual = fecha_inicio_forecast
    for d in range(dias_futuros):
        wd = fecha_actual.weekday()
        pred = dow_median.get(wd, recent_mean) * 0.70 + recent_mean * 0.30
        preds_finales.append(max(0.0, float(pred)))
        fecha_actual += timedelta(days=1)
    return preds_finales

@lru_cache(maxsize=16)
def _festivos_mexico(years_tuple):
    return holidays.country_holidays('MX', years=list(years_tuple))

def pronosticar_con_machine_learning(df_diario_campana, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls):
    df_ml = df_diario_campana.sort_values(col_fecha).copy()
    anos_presentes = list(df_ml[col_fecha].dt.year.unique())
    anos_presentes.append(fecha_inicio_forecast.year)
    anos_presentes.append((fecha_inicio_forecast + timedelta(days=dias_futuros)).year)
    years_key = tuple(sorted(set(int(x) for x in anos_presentes)))
    festivos_pais = _festivos_mexico(years_key)

    q1, q3 = df_ml[col_calls].quantile(0.25), df_ml[col_calls].quantile(0.75)
    iqr = q3 - q1
    df_ml['calls_clean'] = np.clip(df_ml[col_calls], max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr)
    df_ml['baseline'] = df_ml['calls_clean'].shift(1).rolling(window=10, min_periods=1).mean()
    df_ml['ratio'] = np.where(df_ml['baseline'] > 0, df_ml['calls_clean'] / df_ml['baseline'], 1.0)

    r_q1, r_q3 = df_ml['ratio'].quantile(0.25), df_ml['ratio'].quantile(0.75)
    r_iqr = r_q3 - r_q1
    df_ml['ratio_smooth'] = np.clip(df_ml['ratio'], max(0.2, r_q1 - 1.5 * r_iqr), r_q3 + 1.5 * r_iqr)

    df_ml['dia_semana'] = df_ml[col_fecha].dt.weekday
    df_ml['dia_mes'] = df_ml[col_fecha].dt.day
    df_ml['es_inicio_mes'] = (df_ml['dia_mes'] <= 5).astype(int)
    df_ml['es_quincena'] = df_ml['dia_mes'].isin([14, 15, 16, 29, 30, 31, 1]).astype(int)
    df_ml['es_festivo'] = df_ml[col_fecha].apply(lambda x: 1 if x in festivos_pais else 0)

    df_train = df_ml.dropna().copy()
    if len(df_train) < 14:
        return [max(0.0, float(df_diario_campana.tail(7)[col_calls].mean()))] * dias_futuros

    features = ['dia_semana', 'es_inicio_mes', 'es_quincena', 'es_festivo']
    modelo = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=2)
    modelo.fit(df_train[features], df_train['ratio_smooth'])

    # El ratio que predice el bosque depende sólo del calendario. Generarlo para
    # todo el horizonte en una sola llamada evita crear un DataFrame y ejecutar
    # model.predict() una vez por día. El baseline sigue siendo recursivo, por
    # lo que la lógica y el resultado del forecast se conservan.
    fechas_futuras = [fecha_inicio_forecast + timedelta(days=d) for d in range(dias_futuros)]
    X_pred = pd.DataFrame({
        'dia_semana': [f.weekday() for f in fechas_futuras],
        'es_inicio_mes': [1 if f.day <= 5 else 0 for f in fechas_futuras],
        'es_quincena': [1 if f.day in [14, 15, 16, 29, 30, 31, 1] else 0 for f in fechas_futuras],
        'es_festivo': [1 if f in festivos_pais else 0 for f in fechas_futuras],
    })
    ratios_futuros = modelo.predict(X_pred[features])

    historial_calls = [float(x) for x in df_ml['calls_clean'].tolist()]
    preds_finales = []
    for pred_ratio in ratios_futuros:
        ultimos = historial_calls[-10:] if len(historial_calls) >= 10 else historial_calls
        current_baseline = float(np.mean(ultimos)) if ultimos else 0.0
        pred_vol = max(0.0, float(current_baseline * float(pred_ratio)))
        preds_finales.append(pred_vol)
        historial_calls.append(pred_vol)

    return preds_finales

def buscar_archivo_excel():
    try:
        archivos = [f for f in os.listdir(BASE_DIR) if f.lower().endswith('.xlsx') and not f.startswith('~')]
        if not archivos: return None
        for f in archivos:
            if 'data' in f.lower() or 'servicios' in f.lower() or 'historico' in f.lower(): 
                return os.path.join(BASE_DIR, f)
        return os.path.join(BASE_DIR, archivos[0])
    except: return None

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path.startswith('api/'):
        return jsonify({"error": "Endpoint API no encontrado"}), 404
    rutas_a_buscar = [BASE_DIR, os.getcwd(), os.path.dirname(BASE_DIR)]
    for ruta in rutas_a_buscar:
        if path != "" and os.path.exists(os.path.join(ruta, path)):
            return send_from_directory(ruta, path)
    for ruta in rutas_a_buscar:
        target_path = os.path.join(ruta, 'index.html')
        if os.path.exists(target_path):
            response = make_response(send_from_directory(ruta, 'index.html'))
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return response
    return jsonify({"error": "ALERTA CRITICA: No se encontro el archivo index.html."}), 404

@app.route('/favicon.ico')
def favicon(): return '', 204

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    if request.method == 'POST':
        try:
            new_config = request.get_json(force=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(new_config, f)
            return jsonify({'status': 'Guardado'}), 200
        except Exception as e: return jsonify({'error': str(e)}), 500
    else:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f: return jsonify(json.load(f)), 200
            except: pass
        return jsonify({'targetSl': 80, 'targetTime': 20, 'merma': 30}), 200

def clean_num(val, default=0.0):
    if pd.isna(val) or val is None: return default
    try:
        val_str = str(val).strip().replace(',', '.')
        val_str = re.sub(r'[^0-9.]', '', val_str)
        return float(val_str) if val_str else default
    except: return default

def parse_aht_to_seconds(val):
    if pd.isna(val) or val is None: return 180.0
    if isinstance(val, (int, float)): return float(val) if float(val) > 15 else float(val) * 60.0
    val_str = str(val).strip()
    if ':' in val_str:
        p = val_str.split(':')
        try:
            if len(p) == 3: return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
            elif len(p) == 2: return int(p[0]) * 60 + float(p[1])
        except: pass
    return clean_num(val_str, 180.0)

def format_aht_str(seconds):
    if pd.isna(seconds) or seconds is None or seconds <= 0: return "00:00:00"
    secs = int(round(seconds))
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"

def clean_interval_str(val):
    try:
        if pd.isna(val): return "00:00"
        val_str = str(val).strip()
        if hasattr(val, 'hour') and hasattr(val, 'minute'): hh, mm = val.hour, val.minute
        else:
            m = re.search(r'(\d{1,2}):(\d{2})', val_str)
            if m: hh, mm = int(m.group(1)), int(m.group(2))
            else: return "00:00"
        if mm < 15: mm_round = 0
        elif mm < 45: mm_round = 30
        else:
            mm_round = 0
            hh = (hh + 1) % 24
        return f"{hh:02d}:{mm_round:02d}"
    except: return "00:00"

ERLANG_CACHE = {}
def erlang_c_sl_optimizado(A, N, AHT, target_time):
    if N <= A or A <= 0 or N <= 0: return 0.0
    key = (round(A, 2), N, round(AHT, 1), target_time)
    if key in ERLANG_CACHE: return ERLANG_CACHE[key]
    try:
        sum_terms, current_term = 1.0, 1.0
        int_N = min(int(N), 1000)
        for k in range(1, int_N):
            current_term *= (A / k)
            sum_terms += current_term
        last_term = current_term * (A / N) / (1.0 - (A / N))
        pw = last_term / (sum_terms + last_term)
        sl = 1.0 - (pw * math.exp(-(N - A) * (target_time / AHT)))
        resultado = round(max(0.0, min(100.0, sl * 100.0)), 1)
        ERLANG_CACHE[key] = resultado
        return resultado
    except: return 0.0

def calcular_agentes_requeridos_erlang_c(A, aht, target_time, target_sl):
    if A <= 0 or aht <= 0: return 0
    base_n = int(math.floor(A + math.sqrt(A))) if A > 50 else int(math.floor(A)) + 1
    for n in range(base_n, base_n + 150):
        if erlang_c_sl_optimizado(A, n, aht, target_time) >= target_sl:
            return n
    return base_n

def parse_time_str(t_str):
    if not t_str: return None
    t = re.sub(r'[^\d:]', '', str(t_str).lower())
    if not t: return None
    if ':' not in t: t += ':00'
    try:
        p = t.split(':')
        return int(p[0]) * 60 + int(p[1])
    except: return None

def esta_en_ventana_servicio(campana, intervalo_str):
    camp_key = str(campana).strip().lower()
    min_in = parse_time_str(intervalo_str)
    if min_in is None: return True
    for key, window in VENTANAS_SERVICIO.items():
        if key in camp_key or camp_key in key:
            return window['inicio'] <= min_in < window['fin']
    return True

def encontrar_columna(df, posibles):
    for p in posibles:
        for c in df.columns:
            if p.strip().lower() in str(c).strip().lower(): return c
    return None

def generar_intervalos_cobertura(start_min, end_min):
    intervals = []
    curr = start_min
    if start_min < end_min:
        while curr < end_min:
            intervals.append(f"{int(curr // 60):02d}:{int(curr % 60):02d}")
            curr += 30
    else: 
        while curr < 24 * 60:
            intervals.append(f"{int(curr // 60):02d}:{int(curr % 60):02d}")
            curr += 30
        curr = 0
        while curr < end_min:
            intervals.append(f"{int(curr // 60):02d}:{int(curr % 60):02d}")
            curr += 30
    return intervals

def procesar_hoja_roster(df_roster):
    dias_map = {'lunes': 'Lunes', 'martes': 'Martes', 'miércoles': 'Miércoles', 'miercoles': 'Miércoles', 
                'jueves': 'Jueves', 'viernes': 'Viernes', 'sábado': 'Sábado', 'sabado': 'Sábado', 'domingo': 'Domingo'}
    roster_cov, roster_total_camp, roster_total_dia_camp = {}, {}, {}
    col_camp = encontrar_columna(df_roster, ['campaña', 'campana', 'skill', 'servicio'])
    col_agente = encontrar_columna(df_roster, ['agente', 'nombre', 'asesor', 'ejecutivo', 'id'])
    if not col_camp: return roster_cov, roster_total_camp, roster_total_dia_camp
        
    for idx, row in df_roster.iterrows():
        if col_agente:
            agente_val = str(row[col_agente]).strip()
            if agente_val.lower() == 'nan' or agente_val == '': continue
        camp = str(row[col_camp]).strip().title()
        if camp == 'Nan' or camp == '': continue
        
        roster_total_camp[camp] = roster_total_camp.get(camp, 0) + 1
        for col in df_roster.columns:
            c_lower = str(col).lower().strip()
            if c_lower in dias_map:
                dia_real = dias_map[c_lower]
                horario = str(row[col]).strip().upper()
                if horario != 'DD-DD' and 'NAN' not in horario and '-' in horario:
                    key_dia = (camp, dia_real)
                    roster_total_dia_camp[key_dia] = roster_total_dia_camp.get(key_dia, 0) + 1
                    parts = horario.split('-')
                    if len(parts) == 2:
                        s_min = parse_time_str(parts[0].strip())
                        e_min = parse_time_str(parts[1].strip())
                        if s_min is not None and e_min is not None:
                            for inv in generar_intervalos_cobertura(s_min, e_min):
                                roster_cov[(camp, dia_real, inv)] = roster_cov.get((camp, dia_real, inv), 0) + 1
    return roster_cov, roster_total_camp, roster_total_dia_camp


# =====================================================================
# V6 · ROSTER OPERATIVO EDITABLE
# =====================================================================
_ROSTER_DAYS = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']

def _roster_conn():
    conn = sqlite3.connect(WFM_ROSTER_DB, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    return conn

def _roster_db_init():
    with _WFM_ROSTER_LOCK:
        conn = _roster_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS roster_agents (
                    agent_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL DEFAULT '',
                    supervisor TEXT NOT NULL DEFAULT '',
                    coordinator TEXT NOT NULL DEFAULT '',
                    campaign TEXT NOT NULL,
                    channel TEXT NOT NULL DEFAULT 'Llamadas',
                    status TEXT NOT NULL DEFAULT 'Activo',
                    lunes TEXT NOT NULL DEFAULT 'DD-DD',
                    martes TEXT NOT NULL DEFAULT 'DD-DD',
                    miercoles TEXT NOT NULL DEFAULT 'DD-DD',
                    jueves TEXT NOT NULL DEFAULT 'DD-DD',
                    viernes TEXT NOT NULL DEFAULT 'DD-DD',
                    sabado TEXT NOT NULL DEFAULT 'DD-DD',
                    domingo TEXT NOT NULL DEFAULT 'DD-DD',
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL DEFAULT 'wfm'
                );
                CREATE TABLE IF NOT EXISTS roster_overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    work_date TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    source_action_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL DEFAULT 'wfm',
                    UNIQUE(agent_id, work_date),
                    FOREIGN KEY(agent_id) REFERENCES roster_agents(agent_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS roster_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    change_type TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    reason TEXT,
                    changed_at TEXT NOT NULL,
                    changed_by TEXT NOT NULL DEFAULT 'wfm'
                );
                CREATE INDEX IF NOT EXISTS idx_roster_campaign ON roster_agents(campaign);
                CREATE INDEX IF NOT EXISTS idx_roster_supervisor ON roster_agents(supervisor);
                CREATE INDEX IF NOT EXISTS idx_roster_channel ON roster_agents(channel);
                CREATE INDEX IF NOT EXISTS idx_roster_override_date ON roster_overrides(work_date);
            """)
            # Migración compatible con bases V6 existentes.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(roster_agents)").fetchall()}
            if 'coordinator' not in columns:
                conn.execute("ALTER TABLE roster_agents ADD COLUMN coordinator TEXT NOT NULL DEFAULT ''")
            conn.commit()
        finally:
            conn.close()
    _roster_seed_from_excel_if_empty()

def _roster_normalize_schedule(value):
    raw = str(value or '').strip().upper()
    if not raw or raw in ('NAN','NONE','DESCANSO','DESC','OFF','DD-DD'):
        return 'DD-DD'
    if '-' not in raw:
        raise ValueError(f'Horario inválido: {raw}. Usa HH:MM-HH:MM o DD-DD.')
    start_raw, end_raw = [x.strip() for x in raw.split('-',1)]
    start = parse_time_str(start_raw)
    end = parse_time_str(end_raw)
    if start is None or end is None:
        raise ValueError(f'Horario inválido: {raw}. Usa HH:MM-HH:MM o DD-DD.')
    return f"{int(start//60):02d}:{int(start%60):02d}-{int(end//60):02d}:{int(end%60):02d}"

def _roster_row_dict(row):
    if not row:
        return None
    return {
        'agentId': row['agent_id'],
        'fullName': row['full_name'] or '',
        'supervisor': row['supervisor'] or '',
        'coordinator': row['coordinator'] or '',
        'campaign': row['campaign'] or '',
        'channel': row['channel'] or 'Llamadas',
        'status': row['status'] or 'Activo',
        'schedules': {
            'Lunes': row['lunes'] or 'DD-DD',
            'Martes': row['martes'] or 'DD-DD',
            'Miércoles': row['miercoles'] or 'DD-DD',
            'Jueves': row['jueves'] or 'DD-DD',
            'Viernes': row['viernes'] or 'DD-DD',
            'Sábado': row['sabado'] or 'DD-DD',
            'Domingo': row['domingo'] or 'DD-DD'
        },
        'source': row['source'] or '',
        'createdAt': row['created_at'],
        'updatedAt': row['updated_at'],
        'updatedBy': row['updated_by'] or ''
    }

def _roster_history_add(conn, agent_id, change_type, before_obj, after_obj, reason='', changed_by='wfm'):
    conn.execute(
        """INSERT INTO roster_history
        (agent_id, change_type, before_json, after_json, reason, changed_at, changed_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            agent_id, change_type,
            json.dumps(before_obj, ensure_ascii=False) if before_obj is not None else None,
            json.dumps(after_obj, ensure_ascii=False) if after_obj is not None else None,
            str(reason or '')[:500],
            datetime.now().isoformat(timespec='seconds'),
            str(changed_by or 'wfm')[:80]
        )
    )

def _roster_seed_from_excel_if_empty():
    with _WFM_ROSTER_LOCK:
        conn = _roster_conn()
        try:
            if conn.execute("SELECT COUNT(*) FROM roster_agents").fetchone()[0]:
                return
        finally:
            conn.close()

    excel_path = buscar_archivo_excel()
    if not excel_path or not os.path.exists(excel_path):
        return
    try:
        xls = pd.ExcelFile(excel_path, engine='openpyxl')
    except Exception:
        return

    now = datetime.now().isoformat(timespec='seconds')
    migrated = []
    for sh in xls.sheet_names:
        low = sh.lower()
        if not any(x in low for x in ('roster','plantilla','platilla','horario')):
            continue
        try:
            df = pd.read_excel(xls, sheet_name=sh, engine='openpyxl')
        except Exception:
            continue

        col_campaign = encontrar_columna(df, ['campaña','campana','skill','servicio'])
        col_id = encontrar_columna(df, ['id agente','agent id','agent_id','id','agente'])
        col_name = encontrar_columna(df, ['nombre completo','full name','nombre','asesor','ejecutivo'])
        col_supervisor = encontrar_columna(df, ['supervisor','team leader','tl','jefe'])
        col_coordinator = encontrar_columna(df, ['coordinador','coordinator','coord','coordinadora'])
        col_channel = encontrar_columna(df, ['canal','channel'])
        if not col_campaign:
            continue

        aliases = {
            'lunes':'Lunes','martes':'Martes','miércoles':'Miércoles','miercoles':'Miércoles',
            'jueves':'Jueves','viernes':'Viernes','sábado':'Sábado','sabado':'Sábado','domingo':'Domingo'
        }
        day_cols = {}
        for c in df.columns:
            key = str(c).strip().lower()
            if key in aliases:
                day_cols[aliases[key]] = c
        default_channel = 'Chat' if ('chat' in low or 'mensaje' in low) else 'Llamadas'

        for idx, row in df.iterrows():
            campaign = str(row.get(col_campaign, '')).strip()
            if not campaign or campaign.lower() == 'nan':
                continue
            agent_id = str(row.get(col_id, '')).strip() if col_id else ''
            if not agent_id or agent_id.lower() == 'nan':
                agent_id = f'AG-{idx+1:05d}'
            full_name = str(row.get(col_name, '')).strip() if col_name else ''
            if full_name.lower() == 'nan' or full_name == agent_id:
                full_name = ''
            supervisor = str(row.get(col_supervisor, '')).strip() if col_supervisor else ''
            if supervisor.lower() == 'nan':
                supervisor = ''
            coordinator = str(row.get(col_coordinator, '')).strip() if col_coordinator else ''
            if coordinator.lower() == 'nan':
                coordinator = ''
            channel = str(row.get(col_channel, '')).strip() if col_channel else default_channel
            if not channel or channel.lower() == 'nan':
                channel = default_channel
            channel = 'Chat' if 'chat' in channel.lower() else 'Llamadas'
            schedules = {}
            for day in _ROSTER_DAYS:
                raw = row.get(day_cols.get(day), 'DD-DD') if day in day_cols else 'DD-DD'
                try:
                    schedules[day] = _roster_normalize_schedule(raw)
                except Exception:
                    schedules[day] = 'DD-DD'
            migrated.append((agent_id, full_name, supervisor, coordinator, campaign.title(), channel, schedules))

    if not migrated:
        return

    with _WFM_ROSTER_LOCK:
        conn = _roster_conn()
        try:
            for agent_id, full_name, supervisor, coordinator, campaign, channel, s in migrated:
                conn.execute(
                    """INSERT OR IGNORE INTO roster_agents
                    (agent_id,full_name,supervisor,coordinator,campaign,channel,status,lunes,martes,miercoles,jueves,viernes,sabado,domingo,source,created_at,updated_at,updated_by)
                    VALUES (?,?,?,?,?,?,'Activo',?,?,?,?,?,?,?,'excel_migration',?,?,?)""",
                    (agent_id,full_name,supervisor,coordinator,campaign,channel,
                     s['Lunes'],s['Martes'],s['Miércoles'],s['Jueves'],s['Viernes'],s['Sábado'],s['Domingo'],
                     now,now,'system')
                )
            conn.commit()
        finally:
            conn.close()

def _invalidate_roster_dependent_caches():
    global _EXCEL_INFO_CACHE
    with _FORECAST_CACHE_LOCK:
        for mode in ('llamadas','chat'):
            _FORECAST_MEMORY_CACHE[mode] = None
            _FORECAST_CACHE_GENERATION[mode] += 1
            try:
                target = _cache_path(mode)
                if os.path.exists(target):
                    os.remove(target)
            except Exception:
                pass
    with _ASSISTANT_PLAN_CACHE_LOCK:
        _ASSISTANT_PLAN_CACHE.clear()
    _EXCEL_INFO_CACHE = {}

def _roster_fetch_agents(channel=None, include_inactive=True):
    _roster_db_init()
    conn = _roster_conn()
    try:
        sql = "SELECT * FROM roster_agents WHERE 1=1"
        args = []
        if channel and str(channel).lower() != 'all':
            sql += " AND lower(channel)=?"
            args.append(str(channel).lower())
        if not include_inactive:
            sql += " AND lower(status)='activo'"
        sql += " ORDER BY campaign, supervisor, full_name, agent_id"
        return [_roster_row_dict(r) for r in conn.execute(sql,args).fetchall()]
    finally:
        conn.close()

def _roster_dataframe(channel):
    agents = _roster_fetch_agents(channel=channel, include_inactive=False)
    return pd.DataFrame([{
        'Agente': a['agentId'], 'Nombre Completo': a['fullName'], 'Supervisor': a['supervisor'], 'Coordinador': a['coordinator'],
        'Campaña': a['campaign'], 'Canal': a['channel'], 'Estado': a['status'], **a['schedules']
    } for a in agents]) if agents else pd.DataFrame()

def _roster_override_deltas(channel, df_roster):
    coverage_delta, day_delta = {}, {}
    if df_roster is None or df_roster.empty:
        return coverage_delta, day_delta
    by_agent = {str(r.get('Agente','')).strip(): r for _, r in df_roster.iterrows()}
    _roster_db_init()
    conn = _roster_conn()
    try:
        rows = conn.execute(
            """SELECT o.agent_id,o.work_date,o.schedule,a.campaign
               FROM roster_overrides o JOIN roster_agents a ON a.agent_id=o.agent_id
               WHERE lower(a.status)='activo' AND lower(a.channel)=lower(?)""",
            (channel,)
        ).fetchall()
    finally:
        conn.close()

    def intervals(schedule):
        schedule = str(schedule or 'DD-DD').strip().upper()
        if schedule == 'DD-DD' or '-' not in schedule:
            return set()
        p = schedule.split('-',1)
        s, e = parse_time_str(p[0].strip()), parse_time_str(p[1].strip())
        return set(generar_intervalos_cobertura(s,e)) if s is not None and e is not None else set()

    for r in rows:
        base_row = by_agent.get(str(r['agent_id']))
        if base_row is None:
            continue
        try:
            dt = datetime.strptime(str(r['work_date'])[:10], '%Y-%m-%d')
        except Exception:
            continue
        day = _ROSTER_DAYS[dt.weekday()]
        camp = str(r['campaign']).strip().title()
        date = str(r['work_date'])[:10]
        base_set = intervals(base_row.get(day,'DD-DD'))
        override_set = intervals(r['schedule'])
        for inv in base_set - override_set:
            coverage_delta[(camp,date,inv)] = coverage_delta.get((camp,date,inv),0) - 1
        for inv in override_set - base_set:
            coverage_delta[(camp,date,inv)] = coverage_delta.get((camp,date,inv),0) + 1
        if bool(base_set) != bool(override_set):
            day_delta[(camp,date)] = day_delta.get((camp,date),0) + (1 if override_set else -1)
    return coverage_delta, day_delta

def _roster_forecast_metrics(xls_file, channel):
    try:
        df_db = _roster_dataframe(channel)
        if not df_db.empty:
            cov,total_camp,total_day = procesar_hoja_roster(df_db)
            override_cov,override_day = _roster_override_deltas(channel,df_db)
            return cov,total_camp,total_day,override_cov,override_day
    except Exception as e:
        print(f'Roster DB fallback ({channel}): {e}')

    sheet_roster = None
    for sh in xls_file.sheet_names:
        low = sh.lower()
        if channel.lower() == 'chat':
            if any(x in low for x in ('plantilla','platilla','roster')) and ('chat' in low or 'mensaje' in low):
                sheet_roster = sh; break
        else:
            if any(x in low for x in ('plantilla','platilla','roster','horario')) and 'chat' not in low:
                sheet_roster = sh; break
    if sheet_roster:
        try:
            df = pd.read_excel(xls_file,sheet_name=sheet_roster,engine='openpyxl')
            cov,total_camp,total_day = procesar_hoja_roster(df)
            return cov,total_camp,total_day,{},{}
        except Exception:
            pass
    return {},{},{},{},{}

def _roster_validate_payload(payload, existing=None):
    payload, existing = payload or {}, existing or {}
    agent_id = str(payload.get('agentId') or existing.get('agentId') or '').strip()
    if not agent_id:
        raise ValueError('El ID del agente es obligatorio.')
    full_name = str(payload.get('fullName',existing.get('fullName','')) or '').strip()
    supervisor = str(payload.get('supervisor',existing.get('supervisor','')) or '').strip()
    coordinator = str(payload.get('coordinator',existing.get('coordinator','')) or '').strip()
    campaign = str(payload.get('campaign',existing.get('campaign','')) or '').strip()
    if not campaign:
        raise ValueError('La campaña es obligatoria.')
    channel = 'Chat' if 'chat' in str(payload.get('channel',existing.get('channel','Llamadas'))).lower() else 'Llamadas'
    status = str(payload.get('status',existing.get('status','Activo')) or 'Activo').strip().title()
    if status not in ('Activo','Inactivo'):
        status = 'Activo'
    incoming, old_sched = payload.get('schedules') or {}, existing.get('schedules') or {}
    schedules = {d:_roster_normalize_schedule(incoming.get(d,old_sched.get(d,'DD-DD'))) for d in _ROSTER_DAYS}
    return {'agentId':agent_id,'fullName':full_name,'supervisor':supervisor,'coordinator':coordinator,'campaign':campaign.title(),'channel':channel,'status':status,'schedules':schedules}

def _roster_get_agent(agent_id):
    _roster_db_init()
    conn = _roster_conn()
    try:
        return _roster_row_dict(conn.execute("SELECT * FROM roster_agents WHERE agent_id=?",(agent_id,)).fetchone())
    finally:
        conn.close()

def _roster_save_agent(payload, actor='wfm', reason='Edición manual', allow_create=True):
    _roster_db_init()
    agent_id = str((payload or {}).get('agentId') or '').strip()
    before = _roster_get_agent(agent_id) if agent_id else None
    if before is None and not allow_create:
        raise ValueError('No se encontró el agente.')
    clean = _roster_validate_payload(payload,before)
    s, now = clean['schedules'], datetime.now().isoformat(timespec='seconds')
    with _WFM_ROSTER_LOCK:
        conn = _roster_conn()
        try:
            if before:
                conn.execute(
                    """UPDATE roster_agents SET full_name=?,supervisor=?,coordinator=?,campaign=?,channel=?,status=?,
                    lunes=?,martes=?,miercoles=?,jueves=?,viernes=?,sabado=?,domingo=?,updated_at=?,updated_by=? WHERE agent_id=?""",
                    (clean['fullName'],clean['supervisor'],clean['coordinator'],clean['campaign'],clean['channel'],clean['status'],
                     s['Lunes'],s['Martes'],s['Miércoles'],s['Jueves'],s['Viernes'],s['Sábado'],s['Domingo'],now,actor,clean['agentId'])
                )
                change_type='agent_updated'
            else:
                conn.execute(
                    """INSERT INTO roster_agents
                    (agent_id,full_name,supervisor,coordinator,campaign,channel,status,lunes,martes,miercoles,jueves,viernes,sabado,domingo,source,created_at,updated_at,updated_by)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'manual',?,?,?)""",
                    (clean['agentId'],clean['fullName'],clean['supervisor'],clean['coordinator'],clean['campaign'],clean['channel'],clean['status'],
                     s['Lunes'],s['Martes'],s['Miércoles'],s['Jueves'],s['Viernes'],s['Sábado'],s['Domingo'],now,now,actor)
                )
                change_type='agent_created'
            after = _roster_row_dict(conn.execute("SELECT * FROM roster_agents WHERE agent_id=?",(clean['agentId'],)).fetchone())
            _roster_history_add(conn,clean['agentId'],change_type,before,after,reason,actor)
            conn.commit()
        finally:
            conn.close()
    _invalidate_roster_dependent_caches()
    return after

@app.route('/api/roster', methods=['GET'])
def roster_list():
    try:
        agents = _roster_fetch_agents(request.args.get('channel'), str(request.args.get('includeInactive','true')).lower()!='false')
        campaigns = sorted({a['campaign'] for a in agents if a['campaign']})
        supervisors = sorted({a['supervisor'] for a in agents if a['supervisor']})
        coordinators = sorted({a['coordinator'] for a in agents if a['coordinator']})
        active = sum(1 for a in agents if a['status']=='Activo')
        last_update = max([a['updatedAt'] for a in agents if a.get('updatedAt')] or [''])
        return jsonify({'agents':agents,'campaigns':campaigns,'supervisors':supervisors,'coordinators':coordinators,'days':_ROSTER_DAYS,'stats':{
            'total':len(agents),'active':active,'inactive':len(agents)-active,'campaigns':len(campaigns),'supervisors':len(supervisors),'coordinators':len(coordinators),'lastUpdate':last_update
        }}),200
    except Exception as e:
        return jsonify({'error':f'No se pudo cargar el roster: {str(e)}'}),500

@app.route('/api/roster/agent', methods=['POST'])
def roster_save_agent():
    try:
        payload = request.get_json(force=True,silent=False) or {}
        actor = str(payload.pop('actorMode','wfm') or 'wfm')[:80]
        reason = str(payload.pop('reason','Edición manual') or 'Edición manual')[:500]
        return jsonify({'ok':True,'agent':_roster_save_agent(payload,actor,reason,True)}),200
    except sqlite3.IntegrityError:
        return jsonify({'error':'Ya existe un agente con ese ID.'}),409
    except Exception as e:
        return jsonify({'error':str(e)}),400

@app.route('/api/roster/bulk', methods=['POST'])
def roster_bulk_update():
    try:
        payload=request.get_json(force=True,silent=False) or {}
        ids=[str(x).strip() for x in payload.get('agentIds',[]) if str(x).strip()]
        changes=payload.get('changes') or {}
        actor=str(payload.get('actorMode') or 'wfm')[:80]
        if not ids:
            return jsonify({'error':'Selecciona al menos un agente.'}),400
        updated=0
        for agent_id in ids:
            before=_roster_get_agent(agent_id)
            if not before: continue
            merged=dict(before)
            if str(changes.get('campaign','')).strip(): merged['campaign']=str(changes['campaign']).strip()
            if 'supervisor' in changes: merged['supervisor']=str(changes['supervisor']).strip()
            if 'coordinator' in changes: merged['coordinator']=str(changes['coordinator']).strip()
            if changes.get('status') in ('Activo','Inactivo'): merged['status']=changes['status']
            if changes.get('channel') in ('Llamadas','Chat'): merged['channel']=changes['channel']
            _roster_save_agent(merged,actor,'Edición masiva',False); updated+=1
        return jsonify({'ok':True,'updated':updated}),200
    except Exception as e:
        return jsonify({'error':str(e)}),400

def _roster_import_columns(df):
    return {
        'id':encontrar_columna(df,['id agente','agent id','agent_id','id','agente']),
        'name':encontrar_columna(df,['nombre completo','full name','nombre','asesor','ejecutivo']),
        'supervisor':encontrar_columna(df,['supervisor','team leader','tl','jefe']),
        'coordinator':encontrar_columna(df,['coordinador','coordinator','coord','coordinadora']),
        'campaign':encontrar_columna(df,['campaña','campana','skill','servicio']),
        'channel':encontrar_columna(df,['canal','channel']),
        'status':encontrar_columna(df,['estado','status'])
    }

@app.route('/api/roster/import', methods=['POST'])
def roster_import():
    try:
        if 'file' not in request.files:
            return jsonify({'error':'Selecciona un archivo CSV o XLSX.'}),400
        f=request.files['file']; name=(f.filename or '').lower()
        actor=str(request.form.get('actorMode') or 'wfm')[:80]
        if name.endswith('.csv'): df=pd.read_csv(f)
        elif name.endswith('.xlsx'): df=pd.read_excel(f,engine='openpyxl')
        else: return jsonify({'error':'Formato no soportado. Usa CSV o XLSX.'}),400
        cols=_roster_import_columns(df)
        if not cols['id'] or not cols['campaign']:
            return jsonify({'error':'La carga requiere ID Agente y Campaña.'}),400
        aliases={'lunes':'Lunes','martes':'Martes','miércoles':'Miércoles','miercoles':'Miércoles','jueves':'Jueves','viernes':'Viernes','sábado':'Sábado','sabado':'Sábado','domingo':'Domingo'}
        day_cols={}
        for c in df.columns:
            key=str(c).strip().lower()
            if key in aliases: day_cols[aliases[key]]=c
        imported,errors=0,[]
        for idx,row in df.iterrows():
            try:
                agent_id=str(row.get(cols['id'],'')).strip(); campaign=str(row.get(cols['campaign'],'')).strip()
                if not agent_id or agent_id.lower()=='nan' or not campaign or campaign.lower()=='nan': continue
                existing=_roster_get_agent(agent_id) or {}
                schedules={d:(row.get(day_cols[d],'DD-DD') if d in day_cols else (existing.get('schedules') or {}).get(d,'DD-DD')) for d in _ROSTER_DAYS}
                def safe(col, fallback=''):
                    if not col: return fallback
                    val=str(row.get(col,'')).strip()
                    return '' if val.lower()=='nan' else val
                payload={
                    'agentId':agent_id,'fullName':safe(cols['name'],existing.get('fullName','')),
                    'supervisor':safe(cols['supervisor'],existing.get('supervisor','')),
                    'coordinator':safe(cols['coordinator'],existing.get('coordinator','')),
                    'campaign':campaign,'channel':safe(cols['channel'],existing.get('channel','Llamadas')) or 'Llamadas',
                    'status':(safe(cols['status'],existing.get('status','Activo')) or 'Activo').title(),
                    'schedules':schedules
                }
                _roster_save_agent(payload,actor,f'Carga masiva: {f.filename}',True); imported+=1
            except Exception as row_error:
                errors.append({'row':int(idx)+2,'error':str(row_error)})
                if len(errors)>=20: break
        return jsonify({'ok':True,'imported':imported,'errors':errors}),200
    except Exception as e:
        return jsonify({'error':f'No se pudo importar el roster: {str(e)}'}),400

@app.route('/api/roster/export.csv', methods=['GET'])
def roster_export_csv():
    agents=_roster_fetch_agents(include_inactive=True)
    output=io.StringIO(); writer=csv.writer(output)
    writer.writerow(['ID Agente','Nombre Completo','Supervisor','Coordinador','Campaña','Canal','Estado',*_ROSTER_DAYS])
    for a in agents:
        writer.writerow([a['agentId'],a['fullName'],a['supervisor'],a['coordinator'],a['campaign'],a['channel'],a['status'],*[a['schedules'].get(d,'DD-DD') for d in _ROSTER_DAYS]])
    return send_file(io.BytesIO(output.getvalue().encode('utf-8-sig')),mimetype='text/csv; charset=utf-8',as_attachment=True,download_name='roster_operativo.csv')

@app.route('/api/roster/template.csv', methods=['GET'])
def roster_template_csv():
    output=io.StringIO(); writer=csv.writer(output)
    writer.writerow(['ID Agente','Nombre Completo','Supervisor','Coordinador','Campaña','Canal','Estado',*_ROSTER_DAYS])
    writer.writerow(['A-00001','Nombre Apellido','Supervisor 1','Coordinador 1','Coppel Servicios','Llamadas','Activo','08:00-17:00','08:00-17:00','08:00-17:00','08:00-17:00','08:00-17:00','DD-DD','DD-DD'])
    return send_file(io.BytesIO(output.getvalue().encode('utf-8-sig')),mimetype='text/csv; charset=utf-8',as_attachment=True,download_name='plantilla_carga_roster.csv')


@app.route('/api/roster/template.xlsx', methods=['GET'])
def roster_template_xlsx():
    columns = ['ID Agente','Nombre Completo','Supervisor','Coordinador','Campaña','Canal','Estado',*_ROSTER_DAYS]
    example = [[
        'A-00001','Nombre Apellido','Supervisor 1','Coordinador 1','Coppel Servicios',
        'Llamadas','Activo','08:00-17:00','08:00-17:00','08:00-17:00',
        '08:00-17:00','08:00-17:00','DD-DD','DD-DD'
    ]]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(example, columns=columns).to_excel(writer, sheet_name='Carga roster', index=False)
        pd.DataFrame([
            ['ID Agente','Obligatorio. Debe ser único.'],
            ['Nombre Completo','Nombre y apellidos del agente.'],
            ['Supervisor','Supervisor actual. Puede modificarse después.'],
            ['Coordinador','Coordinador actual. Puede modificarse después.'],
            ['Campaña','Obligatorio. Campaña/skill del agente.'],
            ['Canal','Llamadas o Chat.'],
            ['Estado','Activo o Inactivo.'],
            ['Lunes-Domingo','Usa HH:MM-HH:MM. Para descanso usa DD-DD.']
        ], columns=['Campo','Regla']).to_excel(writer, sheet_name='Instrucciones', index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='plantilla_carga_masiva_roster.xlsx'
    )

@app.route('/api/roster/history', methods=['GET'])
def roster_history():
    _roster_db_init()
    try: limit=max(1,min(int(request.args.get('limit',200)),1000))
    except Exception: limit=200
    conn=_roster_conn()
    try:
        result=[]
        for r in conn.execute("SELECT * FROM roster_history ORDER BY id DESC LIMIT ?",(limit,)).fetchall():
            result.append({'id':r['id'],'agentId':r['agent_id'],'changeType':r['change_type'],
                'before':json.loads(r['before_json']) if r['before_json'] else None,
                'after':json.loads(r['after_json']) if r['after_json'] else None,
                'reason':r['reason'] or '','changedAt':r['changed_at'],'changedBy':r['changed_by']})
        return jsonify(result),200
    finally: conn.close()

@app.route('/api/roster/apply-recommendation', methods=['POST'])
def roster_apply_recommendation():
    try:
        payload=request.get_json(force=True,silent=False) or {}
        action_id=str(payload.get('actionId') or '').strip()
        move=payload.get('move') or {}; actor=str(payload.get('actorMode') or 'ops')[:80]
        records=_wfm_action_log_read()
        action=next((r for r in records if r.get('actionId')==action_id),None)
        if not action or action.get('decision')!='accepted':
            return jsonify({'error':'Primero debes aprobar el movimiento desde el Plan de Frank.'}),400
        agent_id=str(move.get('agentId') or move.get('agent') or '').strip()
        work_date=str(move.get('date') or '')[:10]
        proposed=_roster_normalize_schedule(move.get('proposedSchedule'))
        agent=_roster_get_agent(agent_id)
        if not agent: return jsonify({'error':f'No se encontró el agente {agent_id} en el roster operativo.'}),404
        try: dt=datetime.strptime(work_date,'%Y-%m-%d')
        except Exception: return jsonify({'error':'La recomendación no contiene una fecha válida.'}),400
        day=_ROSTER_DAYS[dt.weekday()]; base_schedule=agent['schedules'].get(day,'DD-DD')
        _roster_db_init()
        with _WFM_ROSTER_LOCK:
            conn=_roster_conn()
            try:
                old=conn.execute("SELECT schedule FROM roster_overrides WHERE agent_id=? AND work_date=?",(agent_id,work_date)).fetchone()
                effective_before=old['schedule'] if old else base_schedule
                now=datetime.now().isoformat(timespec='seconds')
                conn.execute(
                    """INSERT INTO roster_overrides(agent_id,work_date,schedule,source_action_id,created_at,updated_at,updated_by)
                    VALUES(?,?,?,?,?,?,?) ON CONFLICT(agent_id,work_date) DO UPDATE SET
                    schedule=excluded.schedule,source_action_id=excluded.source_action_id,updated_at=excluded.updated_at,updated_by=excluded.updated_by""",
                    (agent_id,work_date,proposed,action_id,now,now,actor))
                _roster_history_add(conn,agent_id,'frank_recommendation_applied',
                    {'date':work_date,'day':day,'schedule':effective_before},
                    {'date':work_date,'day':day,'schedule':proposed},f'Recomendación Frank {action_id}',actor)
                conn.commit()
            finally: conn.close()
        now=datetime.now().isoformat(timespec='seconds')
        for r in records:
            if r.get('actionId')==action_id:
                r.setdefault('history',[]).append({'decision':r.get('decision'),'implementationStatus':r.get('implementationStatus','pending'),'updatedAt':r.get('updatedAt')})
                r['decision']='accepted'; r['implementationStatus']='applied'; r['appliedAt']=now; r['updatedAt']=now; r['actorMode']=actor; r['move']=move
                break
        _wfm_action_log_write(records); _invalidate_roster_dependent_caches()
        return jsonify({'ok':True,'agentId':agent_id,'fullName':agent.get('fullName',''),'date':work_date,'day':day,'before':effective_before,'after':proposed}),200
    except Exception as e:
        return jsonify({'error':f'No se pudo aplicar el movimiento al roster: {str(e)}'}),500


def procesar_archivo_llamadas(file_source, target_sl=80.0, target_time=20.0, merma=0.20, dias_futuros=45, campaign_settings=None):
    xls_file = pd.ExcelFile(file_source, engine='openpyxl')
    sheet_calls = xls_file.sheet_names[0]
    for s in xls_file.sheet_names:
        if 'llam' in s.lower() or 'hist' in s.lower() or 'datos' in s.lower(): sheet_calls = s; break
            
    roster_coverage, roster_total_camp, roster_total_dia_camp, roster_override_cov, roster_override_day = _roster_forecast_metrics(xls_file, 'Llamadas')

    df_raw = pd.read_excel(xls_file, sheet_name=sheet_calls, engine='openpyxl')
    col_calls = encontrar_columna(df_raw, ['recibidas', 'llamadas', 'calls', 'volumen', 'ofrecidas', 'entrada'])
    col_aht = encontrar_columna(df_raw, ['aht', 'tmo', 'handle', 'duracion'])
    col_camp = encontrar_columna(df_raw, ['campaña', 'campana', 'skill', 'servicio', 'ring group'])
    col_inter = encontrar_columna(df_raw, ['intervalo', 'hora', 'time'])
    col_fecha = encontrar_columna(df_raw, ['fecha', 'date'])

    if not col_camp: col_camp = df_raw.columns[0]
    if not col_fecha: col_fecha = df_raw.columns[1]
    if not col_inter: col_inter = df_raw.columns[2]
    if not col_calls: col_calls = df_raw.columns[3]

    df_raw[col_camp] = df_raw[col_camp].astype(str).str.strip().str.title()
    df_raw[col_fecha] = pd.to_datetime(df_raw[col_fecha], dayfirst=True, errors='coerce').dt.normalize()
    df_raw = df_raw.dropna(subset=[col_fecha])
    df_raw[col_calls] = [clean_num(x, 0.0) for x in df_raw[col_calls]]

    df_valido = df_raw[df_raw[col_calls] > 0]
    if df_valido.empty: raise ValueError("El archivo de Llamadas no tiene volumen mayor a cero.")
    
    max_fecha_real = df_valido[col_fecha].max()
    df_raw = df_raw[df_raw[col_fecha] <= max_fecha_real]

    if col_aht: df_raw[col_aht] = [parse_aht_to_seconds(x) for x in df_raw[col_aht]]
    else: df_raw['AHT_Calc'] = 180.0; col_aht = 'AHT_Calc'

    df_raw['Inter_Clean'] = df_raw[col_inter].apply(clean_interval_str)
    df_raw['Total_Segundos_Handle'] = df_raw[col_calls] * df_raw[col_aht]

    df = df_raw.groupby([col_fecha, col_camp, 'Inter_Clean']).agg({col_calls: 'sum', 'Total_Segundos_Handle': 'sum'}).reset_index()
    df[col_aht] = np.where(df[col_calls] > 0, df['Total_Segundos_Handle'] / df[col_calls], 180.0)
    df = df.drop(columns=['Total_Segundos_Handle'])

    dias_espanol = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
    meses_espanol = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    df['Dia_Semana_Clean'] = df[col_fecha].dt.weekday.apply(lambda w: dias_espanol[w])

    fecha_inicio_forecast = max_fecha_real + timedelta(days=1)
    aht_global_campana = df.groupby(col_camp)[col_aht].apply(lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 180.0).to_dict()
    df_diario = df.groupby([col_fecha, col_camp])[col_calls].sum().reset_index()
    campanas_unicas = list(set(df[col_camp].unique()).union(set(roster_total_camp.keys())))
    predicciones_futuras = {}

    for camp in campanas_unicas:
        sub = df_diario[df_diario[col_camp] == camp].sort_values(col_fecha).reset_index(drop=True)
        if sub.empty: continue
        ultimos_14_dias = sub.tail(14)[col_calls]
        cv = ultimos_14_dias.std() / ultimos_14_dias.mean() if ultimos_14_dias.mean() > 0 else 0
        if cv < 0.20 and ultimos_14_dias.mean() >= 250:
            preds_finales = pronosticar_macro_campana(sub, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls)
        else:
            preds_finales = pronosticar_con_machine_learning(sub, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls)
        predicciones_futuras[camp] = preds_finales

    vol_historico_por_campana = {c: float(df_diario[df_diario[col_camp] == c][col_calls].mean()) for c in campanas_unicas}
    df['En_Ventana'] = [esta_en_ventana_servicio(c, i) for c, i in zip(df[col_camp], df['Inter_Clean'])]
    df_filtrado = df[df['En_Ventana']].copy()

    df_reciente = df_filtrado[df_filtrado[col_fecha] >= (max_fecha_real - timedelta(days=28))]
    if df_reciente.empty: df_reciente = df_filtrado.copy()
    
    perfil_dia = df_reciente.groupby([col_camp, 'Dia_Semana_Clean', 'Inter_Clean']).agg(
        total_calls=(col_calls, 'mean'), avg_aht=(col_aht, lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    ).reset_index()
    totales_dia = perfil_dia.groupby([col_camp, 'Dia_Semana_Clean'])['total_calls'].transform('sum')
    perfil_dia['weight'] = np.where(totales_dia > 0, perfil_dia['total_calls'] / totales_dia, 0)
    mapa_dia = {(r[col_camp], r['Dia_Semana_Clean'], r['Inter_Clean']): {'weight': r['weight'], 'aht': r['avg_aht']} for _, r in perfil_dia.iterrows()}

    perfil_global = df_reciente.groupby([col_camp, 'Inter_Clean']).agg(total_calls=(col_calls, 'mean')).reset_index()
    totales_global = perfil_global.groupby([col_camp])['total_calls'].transform('sum')
    perfil_global['weight'] = np.where(totales_global > 0, perfil_global['total_calls'] / totales_global, 0)
    mapa_perfil_global = {(r[col_camp], r['Inter_Clean']): r['weight'] for _, r in perfil_global.iterrows()}
    
    todos_los_intervalos_crudos = [f"{int(h):02d}:{int(m):02d}" for h in range(24) for m in (0, 30)]
    intervalos_operativos_por_camp = {camp: [i for i in todos_los_intervalos_crudos if esta_en_ventana_servicio(camp, i)] for camp in campanas_unicas}

    del df_raw, df, df_diario, df_filtrado, df_reciente
    gc.collect()

    factor_asistencia = max(0.01, 1.0 - merma)
    data_processed = []

    for camp in campanas_unicas:
        vol_historico_camp = vol_historico_por_campana.get(camp, 0.0)
        blend_factor = min(1.0, max(0.0, (vol_historico_camp - 50) / 200.0))
        camp_target_sl, camp_target_time = objetivos_campana(camp, target_sl, target_time, campaign_settings)

        for d in range(dias_futuros):
            fecha_actual = fecha_inicio_forecast + timedelta(days=d)
            str_fecha = fecha_actual.strftime('%Y-%m-%d')
            str_mes = f"{meses_espanol[fecha_actual.month]} {fecha_actual.year}"
            nombre_dia = dias_espanol[fecha_actual.weekday()]

            vol_diario = predicciones_futuras.get(camp, [0]*dias_futuros)[d]
            intervalos_validos = intervalos_operativos_por_camp.get(camp, [])

            pesos_crudos = []
            for inter in intervalos_validos:
                w_dia = mapa_dia.get((camp, nombre_dia, inter), {}).get('weight', 0.0)
                w_glob = mapa_perfil_global.get((camp, inter), 0.0)
                if w_dia == 0.0: w_dia = w_glob
                w_final = (w_dia * blend_factor) + (w_glob * (1.0 - blend_factor))
                pesos_crudos.append(w_final)

            suma_pesos = sum(pesos_crudos)
            if suma_pesos > 0: pesos_norm = [p / suma_pesos for p in pesos_crudos]
            elif len(intervalos_validos) > 0: pesos_norm = [1.0 / len(intervalos_validos)] * len(intervalos_validos)
            else: pesos_norm = []

            exact_calls = [vol_diario * p for p in pesos_norm]
            floor_calls = [int(math.floor(c)) for c in exact_calls]
            remainders = [(exact_calls[i] - floor_calls[i], i) for i in range(len(exact_calls))]
            remainders.sort(reverse=True, key=lambda x: x[0])
            
            diff = int(round(vol_diario)) - sum(floor_calls)
            for i in range(diff):
                if i < len(remainders): floor_calls[remainders[i][1]] += 1

            aht_global = aht_global_campana.get(camp, 180.0)
            for idx_inter, inter in enumerate(intervalos_validos):
                calls_int = floor_calls[idx_inter]
                calls_float = exact_calls[idx_inter] 

                info_p = mapa_dia.get((camp, nombre_dia, inter), {})
                aht_real = info_p.get('aht', 0.0)
                if aht_real > 0 and not pd.isna(aht_real): aht = aht_real
                else: aht = aht_global
                if calls_int <= 0: aht = 0.0

                factor_cobertura = factor_cobertura_ambulancia(camp)
                a_erlang_raw = (calls_float * aht * factor_cobertura) / 1800.0 if (aht > 0 and calls_float > 0) else 0.0
                req_ftes = calcular_agentes_requeridos_erlang_c(a_erlang_raw, aht, camp_target_time, camp_target_sl) if calls_float > 0 else 0
                req_hc = math.ceil(req_ftes / factor_asistencia) if req_ftes > 0 else 0
                
                hc_roster = max(0, roster_coverage.get((str(camp), nombre_dia.capitalize(), inter), 0) + roster_override_cov.get((str(camp), str_fecha, inter), 0))
                tot_camp = roster_total_camp.get(str(camp), 0)
                tot_camp_dia = max(0, roster_total_dia_camp.get((str(camp), nombre_dia.capitalize()), 0) + roster_override_day.get((str(camp), str_fecha), 0))

                data_processed.append({
                    'Campaña': str(camp), 'Fecha': str_fecha, 'Mes': str_mes,
                    'Día_Semana': nombre_dia.capitalize(), 'Intervalo': inter,
                    'Llamadas': calls_int, 'AHT': format_aht_str(aht), 'AHT_Segundos': int(round(aht)),
                    'Agentes_Requeridos': req_hc, 'HC_Actual_Roster': hc_roster,
                    'Total_Roster_Campana': tot_camp, 'Total_Roster_Dia': tot_camp_dia,
                    'Factor_Cobertura_Ambulancia': factor_cobertura,
                    'Volumen_Doble_Cobertura': round(calls_float * REGLA_DOBLE_COBERTURA_AMBULANCIA['porcentaje_volumen'], 2) if es_ambulancia_servicios(camp) else 0.0,
                    'Target_SL': camp_target_sl, 'Target_ASA': camp_target_time,
                    'FTE_Erlang': int(req_ftes), 'Concurrencia_Aplicada': 1.0,
                    'Factor_Correccion': 1.0
                })

    df_final = pd.DataFrame(data_processed)
    if not df_final.empty:
        df_final = forzar_cuadre_dashboard(df_final)
        data_processed = df_final.to_dict('records')

    _guardar_cache_forecast('llamadas', data_processed)
    return data_processed

def procesar_archivo_chat(file_source, target_sl=80.0, target_time=20.0, merma=0.20, concurrencia=3.0, dias_futuros=45, campaign_settings=None):
    xls_file = pd.ExcelFile(file_source, engine='openpyxl')
    sheet_chat = None
    for s in xls_file.sheet_names:
        if ('chat' in s.lower() or 'mensaje' in s.lower()) and ('plantilla' not in s.lower() and 'roster' not in s.lower() and 'platilla' not in s.lower()): 
            sheet_chat = s; break
    if not sheet_chat: raise ValueError("No se encontró pestaña Chat.")

    roster_coverage, roster_total_camp, roster_total_dia_camp, roster_override_cov, roster_override_day = _roster_forecast_metrics(xls_file, 'Chat')

    df_raw = pd.read_excel(xls_file, sheet_name=sheet_chat, engine='openpyxl')
    col_calls = encontrar_columna(df_raw, ['recibidos', 'recibidas', 'llamadas', 'chats', 'mensajes'])
    col_aht = encontrar_columna(df_raw, ['aht', 'tmo', 'handle', 'duracion'])
    col_camp = encontrar_columna(df_raw, ['campaña', 'campana', 'skill'])
    col_inter = encontrar_columna(df_raw, ['intervalo', 'hora', 'time'])
    col_fecha = encontrar_columna(df_raw, ['fecha', 'date'])

    if not col_camp: col_camp = df_raw.columns[0]
    if not col_fecha: col_fecha = df_raw.columns[1]
    if not col_inter: col_inter = df_raw.columns[2]
    if not col_calls: col_calls = df_raw.columns[3]

    df_raw[col_camp] = df_raw[col_camp].astype(str).str.strip().str.title()
    df_raw[col_fecha] = pd.to_datetime(df_raw[col_fecha], dayfirst=True, errors='coerce').dt.normalize()
    df_raw = df_raw.dropna(subset=[col_fecha])
    df_raw[col_calls] = [clean_num(x, 0.0) for x in df_raw[col_calls]]

    df_valido = df_raw[df_raw[col_calls] > 0]
    if df_valido.empty: raise ValueError("El archivo Chat no tiene volumen mayor a cero.")
    
    max_fecha_real = df_valido[col_fecha].max()
    df_raw = df_raw[df_raw[col_fecha] <= max_fecha_real]

    if col_aht: df_raw[col_aht] = [parse_aht_to_seconds(x) for x in df_raw[col_aht]]
    else: df_raw['AHT_Calc'] = 600.0; col_aht = 'AHT_Calc'

    df_raw['Inter_Clean'] = df_raw[col_inter].apply(clean_interval_str)
    df_raw['Total_Segundos_Handle'] = df_raw[col_calls] * df_raw[col_aht]

    df = df_raw.groupby([col_fecha, col_camp, 'Inter_Clean']).agg({col_calls: 'sum', 'Total_Segundos_Handle': 'sum'}).reset_index()
    df[col_aht] = np.where(df[col_calls] > 0, df['Total_Segundos_Handle'] / df[col_calls], 600.0)
    df = df.drop(columns=['Total_Segundos_Handle'])

    dias_espanol = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
    meses_espanol = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    df['Dia_Semana_Clean'] = df[col_fecha].dt.weekday.apply(lambda w: dias_espanol[w])

    fecha_inicio_forecast = max_fecha_real + timedelta(days=1)
    aht_global_campana = df.groupby(col_camp)[col_aht].apply(lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 600.0).to_dict()
    df_diario = df.groupby([col_fecha, col_camp])[col_calls].sum().reset_index()
    campanas_unicas = list(set(df[col_camp].unique()).union(set(roster_total_camp.keys())))
    predicciones_futuras = {}

    for camp in campanas_unicas:
        sub = df_diario[df_diario[col_camp] == camp].sort_values(col_fecha).reset_index(drop=True)
        if sub.empty: continue
        ultimos_14_dias = sub.tail(14)[col_calls]
        cv = ultimos_14_dias.std() / ultimos_14_dias.mean() if ultimos_14_dias.mean() > 0 else 0
        if cv < 0.20 and ultimos_14_dias.mean() >= 250:
            preds_finales = pronosticar_macro_campana(sub, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls)
        else:
            preds_finales = pronosticar_con_machine_learning(sub, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls)
        predicciones_futuras[camp] = preds_finales

    vol_historico_por_campana = {c: float(df_diario[df_diario[col_camp] == c][col_calls].mean()) for c in campanas_unicas}
    df['En_Ventana'] = [esta_en_ventana_servicio(c, i) for c, i in zip(df[col_camp], df['Inter_Clean'])]
    df_filtrado = df[df['En_Ventana']].copy()
    df_reciente = df_filtrado[df_filtrado[col_fecha] >= (max_fecha_real - timedelta(days=28))]
    if df_reciente.empty: df_reciente = df_filtrado.copy()
    
    perfil_dia = df_reciente.groupby([col_camp, 'Dia_Semana_Clean', 'Inter_Clean']).agg(
        total_calls=(col_calls, 'mean'), avg_aht=(col_aht, lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    ).reset_index()
    totales_dia = perfil_dia.groupby([col_camp, 'Dia_Semana_Clean'])['total_calls'].transform('sum')
    perfil_dia['weight'] = np.where(totales_dia > 0, perfil_dia['total_calls'] / totales_dia, 0)
    mapa_dia = {(r[col_camp], r['Dia_Semana_Clean'], r['Inter_Clean']): {'weight': r['weight'], 'aht': r['avg_aht']} for _, r in perfil_dia.iterrows()}

    perfil_global = df_reciente.groupby([col_camp, 'Inter_Clean']).agg(total_calls=(col_calls, 'mean')).reset_index()
    totales_global = perfil_global.groupby([col_camp])['total_calls'].transform('sum')
    perfil_global['weight'] = np.where(totales_global > 0, perfil_global['total_calls'] / totales_global, 0)
    mapa_perfil_global = {(r[col_camp], r['Inter_Clean']): r['weight'] for _, r in perfil_global.iterrows()}
    
    todos_los_intervalos_crudos = [f"{int(h):02d}:{int(m):02d}" for h in range(24) for m in (0, 30)]
    intervalos_operativos_por_camp = {camp: [i for i in todos_los_intervalos_crudos if esta_en_ventana_servicio(camp, i)] for camp in campanas_unicas}

    del df_raw, df, df_diario, df_filtrado, df_reciente
    gc.collect()

    factor_asistencia = max(0.01, 1.0 - merma)
    data_processed = []

    for camp in campanas_unicas:
        vol_historico_camp = vol_historico_por_campana.get(camp, 0.0)
        blend_factor = min(1.0, max(0.0, (vol_historico_camp - 50) / 200.0))
        camp_target_sl, camp_target_time = objetivos_campana(camp, target_sl, target_time, campaign_settings)

        for d in range(dias_futuros):
            fecha_actual = fecha_inicio_forecast + timedelta(days=d)
            str_fecha = fecha_actual.strftime('%Y-%m-%d')
            str_mes = f"{meses_espanol[fecha_actual.month]} {fecha_actual.year}"
            nombre_dia = dias_espanol[fecha_actual.weekday()]
            vol_diario = predicciones_futuras.get(camp, [0]*dias_futuros)[d]
            intervalos_validos = intervalos_operativos_por_camp.get(camp, [])

            pesos_crudos = []
            for inter in intervalos_validos:
                w_dia = mapa_dia.get((camp, nombre_dia, inter), {}).get('weight', 0.0)
                w_glob = mapa_perfil_global.get((camp, inter), 0.0)
                if w_dia == 0.0: w_dia = w_glob
                w_final = (w_dia * blend_factor) + (w_glob * (1.0 - blend_factor))
                pesos_crudos.append(w_final)

            suma_pesos = sum(pesos_crudos)
            if suma_pesos > 0: pesos_norm = [p / suma_pesos for p in pesos_crudos]
            elif len(intervalos_validos) > 0: pesos_norm = [1.0 / len(intervalos_validos)] * len(intervalos_validos)
            else: pesos_norm = []

            exact_calls = [vol_diario * p for p in pesos_norm]
            floor_calls = [int(math.floor(c)) for c in exact_calls]
            remainders = [(exact_calls[i] - floor_calls[i], i) for i in range(len(exact_calls))]
            remainders.sort(reverse=True, key=lambda x: x[0])
            
            diff = int(round(vol_diario)) - sum(floor_calls)
            for i in range(diff):
                if i < len(remainders): floor_calls[remainders[i][1]] += 1

            aht_global = aht_global_campana.get(camp, 600.0)
            for idx_inter, inter in enumerate(intervalos_validos):
                calls_int = floor_calls[idx_inter]
                calls_float = exact_calls[idx_inter] 
                info_p = mapa_dia.get((camp, nombre_dia, inter), {})
                aht_real = info_p.get('aht', 0.0)
                if aht_real > 0 and not pd.isna(aht_real): aht = aht_real
                else: aht = aht_global
                if calls_int <= 0: aht = 0.0

                aht_efectivo = aht / max(1.0, concurrencia)
                a_erlang_raw = (calls_float * aht_efectivo) / 1800.0 if (aht_efectivo > 0 and calls_float > 0) else 0.0
                req_ftes = calcular_agentes_requeridos_erlang_c(a_erlang_raw, aht_efectivo, camp_target_time, camp_target_sl) if calls_float > 0 else 0
                req_hc = math.ceil(req_ftes / factor_asistencia) if req_ftes > 0 else 0.0

                hc_roster = max(0, roster_coverage.get((str(camp), nombre_dia.capitalize(), inter), 0) + roster_override_cov.get((str(camp), str_fecha, inter), 0))
                tot_camp = roster_total_camp.get(str(camp), 0)
                tot_camp_dia = max(0, roster_total_dia_camp.get((str(camp), nombre_dia.capitalize()), 0) + roster_override_day.get((str(camp), str_fecha), 0))

                data_processed.append({
                    'Campaña': str(camp), 'Fecha': str_fecha, 'Mes': str_mes, 'Día_Semana': nombre_dia.capitalize(),
                    'Intervalo': inter, 'Llamadas': calls_int, 'AHT': format_aht_str(aht),
                    'AHT_Segundos': int(round(aht)), 'Agentes_Requeridos': req_hc, 'HC_Actual_Roster': hc_roster,
                    'Total_Roster_Campana': tot_camp, 'Total_Roster_Dia': tot_camp_dia,
                    'Target_SL': camp_target_sl, 'Target_ASA': camp_target_time,
                    'FTE_Erlang': int(req_ftes), 'Concurrencia_Aplicada': float(max(1.0, concurrencia)),
                    'Factor_Correccion': 1.0
                })

    df_final = pd.DataFrame(data_processed)
    if not df_final.empty:
        df_final = forzar_cuadre_dashboard(df_final)
        data_processed = df_final.to_dict('records')

    _guardar_cache_forecast('chat', data_processed)
    return data_processed

@app.route('/api/campaigns', methods=['GET'])
def get_campaigns():
    mode = request.args.get('mode', 'llamadas').lower()
    excel_path = buscar_archivo_excel()
    if not excel_path:
        return jsonify([]), 200
    cached = _cache_excel_info_get(excel_path, f'campaigns:{mode}')
    if cached is not None:
        return jsonify(cached), 200
    try:
        xls = pd.ExcelFile(excel_path, engine='openpyxl')
        sheet = None
        if mode == 'chat':
            for sh in xls.sheet_names:
                low = sh.lower()
                if ('chat' in low or 'mensaje' in low) and all(x not in low for x in ['plantilla', 'roster', 'platilla']):
                    sheet = sh
                    break
        else:
            sheet = xls.sheet_names[0]
            for sh in xls.sheet_names:
                low = sh.lower()
                if 'llam' in low or 'hist' in low or 'datos' in low:
                    sheet = sh
                    break
        if not sheet:
            return jsonify([]), 200
        preview = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
        col_camp = encontrar_columna(preview, ['campaña', 'campana', 'skill', 'servicio', 'ring group'])
        if not col_camp:
            return jsonify([]), 200
        campanas = sorted({str(x).strip().title() for x in preview[col_camp].dropna().tolist() if str(x).strip() and str(x).strip().lower() != 'nan'})
        _cache_excel_info_set(excel_path, f'campaigns:{mode}', campanas)
        return jsonify(campanas), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_forecast_status():
    excel_path = buscar_archivo_excel()
    if not excel_path:
        return jsonify({'archivo': None, 'ultimo_dato': None, 'actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 200
    cached = _cache_excel_info_get(excel_path, 'status')
    if cached is not None:
        return jsonify(cached), 200
    ultimo_dato = None
    try:
        xls = pd.ExcelFile(excel_path, engine='openpyxl')
        sheet = xls.sheet_names[0]
        for sh in xls.sheet_names:
            if 'llam' in sh.lower() or 'hist' in sh.lower() or 'datos' in sh.lower():
                sheet = sh; break
        preview = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
        col_fecha = encontrar_columna(preview, ['fecha', 'date'])
        if col_fecha:
            fechas = pd.to_datetime(preview[col_fecha], dayfirst=True, errors='coerce').dropna()
            if not fechas.empty: ultimo_dato = fechas.max().strftime('%Y-%m-%d')
    except Exception:
        pass
    payload = {
        'archivo': os.path.basename(excel_path),
        'ultimo_dato': ultimo_dato,
        'actualizacion': datetime.fromtimestamp(os.path.getmtime(excel_path)).strftime('%Y-%m-%d %H:%M:%S')
    }
    _cache_excel_info_set(excel_path, 'status', payload)
    return jsonify(payload), 200

@app.route('/api/latest', methods=['GET'])
def get_latest_forecast():
    mode = request.args.get('mode', 'llamadas')
    cache_data = _leer_cache_forecast(mode)
    if isinstance(cache_data, list) and cache_data:
        return jsonify(cache_data), 200
            
    excel_path = buscar_archivo_excel()
    if excel_path:
        try:
            sl, tt, merma, dias, concurrencia = 80.0, 20.0, 30.0, 130, 3.0
            campaign_settings = {}
            if os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                        sl, tt = float(cfg.get('targetSl', 80.0)), float(cfg.get('targetTime', 20.0))
                        merma_data = cfg.get('merma', 30.0)
                        merma = float(merma_data.get(mode, 30.0)) if isinstance(merma_data, dict) else float(merma_data)
                        dias = int(clean_num(cfg.get('dias'), dias))
                        concurrencia = float(clean_num(cfg.get('concurrencia'), concurrencia))
                        all_settings = cfg.get('campaignSettings', {})
                        campaign_settings = all_settings.get(mode, {}) if isinstance(all_settings, dict) else {}
                except: pass
            
            merma_pct = merma / 100.0
            if mode == 'chat': data = procesar_archivo_chat(excel_path, target_sl=sl, target_time=tt, merma=merma_pct, concurrencia=concurrencia, dias_futuros=dias, campaign_settings=campaign_settings)
            else: data = procesar_archivo_llamadas(excel_path, target_sl=sl, target_time=tt, merma=merma_pct, dias_futuros=dias, campaign_settings=campaign_settings)
            gc.collect()
            return jsonify(data), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify([]), 200

@app.route('/api/process', methods=['POST', 'GET'])
def process_data():
    if request.method == 'GET': return jsonify({'status': 'API activa'}), 200
    excel_path = buscar_archivo_excel()
    if not excel_path: return jsonify({'error': 'No se encontro Excel (.xlsx).'}), 400
    try:
        try:
            campaign_settings = json.loads(request.form.get('campaign_settings', '{}') or '{}')
        except Exception:
            campaign_settings = {}
        data = procesar_archivo_llamadas(
            excel_path, 
            float(clean_num(request.form.get('target_sl'), 80.0)), 
            float(clean_num(request.form.get('target_time'), 20.0)), 
            float(clean_num(request.form.get('merma'), 30.0)) / 100.0, 
            int(clean_num(request.form.get('dias'), 45)),
            campaign_settings=campaign_settings
        )
        gc.collect()
        return jsonify(data)
    except Exception as e:
        gc.collect()
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_chat', methods=['POST'])
def process_chat_data():
    excel_path = buscar_archivo_excel()
    if not excel_path: return jsonify({'error': 'No se encontro Excel (.xlsx).'}), 400
    try:
        try:
            campaign_settings = json.loads(request.form.get('campaign_settings', '{}') or '{}')
        except Exception:
            campaign_settings = {}
        data = procesar_archivo_chat(
            excel_path, 
            float(clean_num(request.form.get('target_sl'), 80.0)),
            float(clean_num(request.form.get('target_time'), 20.0)),
            float(clean_num(request.form.get('merma'), 30.0)) / 100.0, 
            float(clean_num(request.form.get('concurrencia'), 3.0)), 
            int(clean_num(request.form.get('dias'), 45)),
            campaign_settings=campaign_settings
        )
        gc.collect()
        return jsonify(data)
    except Exception as e:
        gc.collect()
        return jsonify({'error': str(e)}), 500


# =====================================================================
# ASISTENTE WFM · MOTOR DETERMINÍSTICO SOBRE EL CONTEXTO DEL DASHBOARD
# =====================================================================
def _assistant_norm(text):
    return re.sub(r'\s+', ' ', str(text or '').strip().lower())

def _assistant_minutes(hhmm):
    try:
        h, m = str(hhmm).split(':')[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return 0

def _assistant_hhmm_minutes(minutes):
    minutes = int(minutes) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"

def _assistant_movement_actions(deficits, surpluses, limit=4):
    actions = []
    used = set()

    for d in deficits:
        missing = abs(int(round(float(d.get('gap', 0)))))
        if missing <= 0:
            continue

        d_date = str(d.get('date', ''))
        d_channel = str(d.get('channel', ''))
        d_min = _assistant_minutes(d.get('interval', '00:00'))

        candidates = []
        for idx, s in enumerate(surpluses):
            if idx in used:
                continue
            if str(s.get('date', '')) != d_date or str(s.get('channel', '')) != d_channel:
                continue
            extra = int(round(float(s.get('gap', 0))))
            if extra <= 0:
                continue
            s_min = _assistant_minutes(s.get('interval', '00:00'))
            distance = abs(s_min - d_min)
            if distance <= 240:
                candidates.append((distance, idx, s, extra))

        candidates.sort(key=lambda x: (x[0], -x[3]))
        if not candidates:
            continue

        _, idx, source, extra = candidates[0]
        move = min(missing, extra)
        if move <= 0:
            continue
        used.add(idx)

        source_time = source.get('interval', '—')
        target_time = d.get('interval', '—')
        channel = d_channel or 'canal'
        actions.append(
            f"Revisar hasta {move} HC de {channel} alrededor de {source_time} para reforzar {target_time} "
            f"({d_date}). Priorizar ajustes de entrada y salida antes de mover plantilla entre campañas."
        )
        if len(actions) >= limit:
            break

    return actions

def _assistant_cards(context):
    period = context.get('periodCapacity') or {}
    worst = context.get('worst') or {}
    cards = []

    required = int(round(float(period.get('required', 0) or 0)))
    actual = int(round(float(period.get('actual', 0) or 0)))
    gap = int(round(float(period.get('gap', actual - required) or 0)))
    coverage = int(round(float(period.get('coverage', 100) or 100)))

    cards.append({
        'title': 'HC requerido',
        'value': f"{required} HC",
        'detail': str(context.get('scopeLabel') or 'Periodo')
    })
    cards.append({
        'title': 'HC actual',
        'value': f"{actual} HC",
        'detail': f"Cobertura {coverage}%"
    })
    cards.append({
        'title': 'Brecha de plantilla',
        'value': f"{gap} HC",
        'detail': 'Actual - requerido'
    })

    if worst:
        cards.append({
            'title': 'Intervalo más crítico',
            'value': f"{abs(int(round(float(worst.get('gap', 0)))))} HC",
            'detail': f"{worst.get('date','')} · {worst.get('interval','')}"
        })
    return cards[:4]

def _assistant_explain_metric(question):
    q = _assistant_norm(question)
    glossary = [
        (['shrinkage', 'merma'], 'Shrinkage o merma representa el porcentaje de tiempo pagado que no está disponible para atender demanda, por ejemplo descansos, capacitación, incidencias o actividades no productivas. A mayor merma, mayor HC requerido.'),
        (['aht', 'tmo', 'tiempo medio'], 'AHT/TMO es el tiempo promedio de manejo por interacción. Si aumenta el AHT con el mismo volumen, aumenta la carga y normalmente también el HC requerido.'),
        (['asa'], 'ASA objetivo es el tiempo medio de espera que se busca mantener. Un ASA más exigente normalmente requiere más capacidad.'),
        (['nivel de servicio', 'sl ', 'service level'], 'El Nivel de Servicio define el porcentaje objetivo de interacciones que deben atenderse dentro del tiempo objetivo. Una meta más alta suele incrementar el HC requerido.'),
        (['erlang'], 'Erlang C convierte volumen, AHT y objetivo de servicio en agentes simultáneos requeridos para llamadas. El dashboard después ajusta ese requerimiento por merma para llegar al HC necesario.'),
        (['concurrencia'], 'En Chat, la concurrencia indica cuántas conversaciones puede manejar un agente simultáneamente. Una mayor concurrencia reduce la carga efectiva por conversación, aunque debe mantenerse dentro de límites operativos realistas.')
    ]
    for keys, answer in glossary:
        if any(k in q for k in keys):
            return answer
    return None


def _assistant_roster_detail(excel_path):
    """Extract person-level schedules; V6 uses the operational roster DB first."""
    try:
        db_agents = _roster_fetch_agents(include_inactive=False)
        if db_agents:
            agents = []
            for a in db_agents:
                schedules = {}
                for day, raw in (a.get('schedules') or {}).items():
                    if raw == 'DD-DD' or '-' not in str(raw):
                        continue
                    p = str(raw).split('-', 1)
                    start, end = parse_time_str(p[0].strip()), parse_time_str(p[1].strip())
                    if start is not None and end is not None:
                        schedules[day] = {'raw': raw, 'start': int(start), 'end': int(end)}
                if schedules:
                    agents.append({
                        'agent': a.get('fullName') or a.get('agentId'),
                        'agentId': a.get('agentId'),
                        'fullName': a.get('fullName') or '',
                        'supervisor': a.get('supervisor') or '',
                        'coordinator': a.get('coordinator') or '',
                        'campaign': a.get('campaign') or '',
                        'channel': a.get('channel') or '',
                        'schedules': schedules
                    })
            if agents:
                return {'available':True,'sheet':'wfm_roster.db','agents':agents,'reason':''}
    except Exception as e:
        print(f'Frank roster DB fallback: {e}')

    cached = _cache_excel_info_get(excel_path, 'assistant_roster_detail:v2')
    if cached is not None:
        return cached

    result = {'available': False, 'sheet': None, 'agents': [], 'reason': ''}
    try:
        xls = pd.ExcelFile(excel_path, engine='openpyxl')
        roster_sheet = None
        for sh in xls.sheet_names:
            low = sh.lower()
            if 'roster' in low or 'plantilla' in low or 'platilla' in low or 'horario' in low:
                roster_sheet = sh
                break
        if not roster_sheet:
            result['reason'] = 'No se encontró una hoja de roster/plantilla.'
            return _cache_excel_info_set(excel_path, 'assistant_roster_detail:v2', result)

        df_roster = pd.read_excel(xls, sheet_name=roster_sheet, engine='openpyxl')
        col_camp = encontrar_columna(df_roster, ['campaña', 'campana', 'skill', 'servicio'])
        col_agent = encontrar_columna(df_roster, ['agente', 'nombre', 'asesor', 'ejecutivo', 'id'])
        col_channel = encontrar_columna(df_roster, ['canal', 'channel'])

        if not col_camp:
            result['reason'] = 'La hoja de roster no contiene una columna de campaña/skill.'
            return _cache_excel_info_set(excel_path, 'assistant_roster_detail:v2', result)

        dias_map = {
            'lunes': 'Lunes', 'martes': 'Martes', 'miércoles': 'Miércoles', 'miercoles': 'Miércoles',
            'jueves': 'Jueves', 'viernes': 'Viernes', 'sábado': 'Sábado', 'sabado': 'Sábado',
            'domingo': 'Domingo'
        }
        day_columns = {}
        for col in df_roster.columns:
            key = str(col).strip().lower()
            if key in dias_map:
                day_columns[dias_map[key]] = col

        agents = []
        for row_idx, row in df_roster.iterrows():
            campaign = str(row.get(col_camp, '')).strip()
            if not campaign or campaign.lower() == 'nan':
                continue

            agent = str(row.get(col_agent, '')).strip() if col_agent else ''
            if not agent or agent.lower() == 'nan':
                agent = f'Agente {row_idx + 1}'

            channel = ''
            if col_channel:
                channel = str(row.get(col_channel, '')).strip()
                if channel.lower() == 'nan':
                    channel = ''

            schedules = {}
            for day_name, col in day_columns.items():
                raw = str(row.get(col, '')).strip().upper()
                if not raw or raw == 'NAN' or raw == 'DD-DD' or '-' not in raw:
                    continue
                parts = raw.split('-', 1)
                if len(parts) != 2:
                    continue
                start = parse_time_str(parts[0].strip())
                end = parse_time_str(parts[1].strip())
                if start is None or end is None:
                    continue
                schedules[day_name] = {
                    'raw': raw,
                    'start': int(start),
                    'end': int(end)
                }

            if schedules:
                agents.append({
                    'agent': agent,
                    'campaign': str(campaign).title(),
                    'channel': channel.title() if channel else '',
                    'schedules': schedules
                })

        result = {
            'available': bool(agents),
            'sheet': roster_sheet,
            'agents': agents,
            'reason': '' if agents else 'No se encontraron horarios válidos por persona en el roster.'
        }
    except Exception as e:
        result['reason'] = f'No se pudo leer el roster: {str(e)}'

    return _cache_excel_info_set(excel_path, 'assistant_roster_detail:v2', result)

def _assistant_span_intervals(start_min, end_min):
    start_min = int(start_min) % (24 * 60)
    end_min = int(end_min) % (24 * 60)
    return generar_intervalos_cobertura(start_min, end_min)

def _assistant_schedule_text(start_min, end_min):
    return f"{_assistant_hhmm_minutes(start_min)}-{_assistant_hhmm_minutes(end_min)}"

def _assistant_day_name(date_text):
    try:
        dt = datetime.strptime(str(date_text)[:10], '%Y-%m-%d')
        return ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'][dt.weekday()]
    except Exception:
        return ''

def _assistant_norm_channel(value):
    v = _assistant_norm(value)
    if 'chat' in v or 'mensaje' in v:
        return 'chat'
    if 'llam' in v or 'call' in v:
        return 'llamadas'
    return v

def _assistant_optimizer_state(slots):
    state = {}
    for row in slots:
        try:
            date = str(row.get('date', ''))[:10]
            interval = str(row.get('interval', '00:00'))
            campaign = normalizar_nombre_campana(row.get('campaign', ''))
            channel = _assistant_norm_channel(row.get('channel', ''))
            required = float(row.get('required', 0) or 0)
            actual = float(row.get('actual', 0) or 0)
        except Exception:
            continue
        if not date or not campaign or not interval:
            continue
        key = (date, campaign, channel, interval)
        if key not in state:
            state[key] = {'required': 0.0, 'actual': actual}
        state[key]['required'] += required
        # HC actual is a roster snapshot for the campaign/interval, so use max
        # rather than summing duplicates from repeated source rows.
        state[key]['actual'] = max(state[key]['actual'], actual)
    return state

def _assistant_state_metrics(state):
    total_deficit = 0.0
    worst_gap = 0.0
    deficit_slots = 0
    for row in state.values():
        gap = float(row['actual']) - float(row['required'])
        if gap < 0:
            total_deficit += -gap
            deficit_slots += 1
            worst_gap = min(worst_gap, gap)
    return {
        'total_deficit': round(total_deficit, 2),
        'worst_gap': round(worst_gap, 2),
        'deficit_slots': deficit_slots
    }

def _assistant_apply_shift_to_state(state, date, campaign_norm, channel_norm, old_intervals, new_intervals):
    updated = {k: {'required': v['required'], 'actual': v['actual']} for k, v in state.items()}
    old_set, new_set = set(old_intervals), set(new_intervals)
    for interval in old_set - new_set:
        key = (date, campaign_norm, channel_norm, interval)
        if key in updated:
            updated[key]['actual'] = max(0.0, updated[key]['actual'] - 1.0)
    for interval in new_set - old_set:
        key = (date, campaign_norm, channel_norm, interval)
        if key in updated:
            updated[key]['actual'] += 1.0
    return updated

def _assistant_new_deficits_created(before, after):
    created = 0
    for key, b in before.items():
        a = after.get(key, b)
        before_gap = float(b['actual']) - float(b['required'])
        after_gap = float(a['actual']) - float(a['required'])
        if before_gap >= 0 and after_gap < 0:
            created += 1
    return created

def _assistant_improved_intervals(before, after, limit=10):
    improvements = []
    for key, b in before.items():
        a = after.get(key, b)
        b_def = max(0.0, float(b['required']) - float(b['actual']))
        a_def = max(0.0, float(a['required']) - float(a['actual']))
        if a_def + 1e-9 < b_def:
            date, campaign, channel, interval = key
            improvements.append({
                'date': date,
                'campaign': campaign.title(),
                'channel': channel.title(),
                'interval': interval,
                'beforeGap': round(float(b['actual']) - float(b['required']), 1),
                'afterGap': round(float(a['actual']) - float(a['required']), 1),
                'improvement': round(b_def - a_def, 1)
            })
    improvements.sort(key=lambda x: (-x['improvement'], x['date'], x['interval']))
    return improvements[:limit]


def _assistant_context_signature(context, excel_path, max_moves):
    client_sig = str(context.get('contextSignature') or '').strip()
    excel_sig = _excel_signature(excel_path)
    if client_sig:
        base = f"{client_sig}|{excel_sig}|{int(max_moves)}"
        return hashlib.sha1(base.encode('utf-8')).hexdigest()

    slots = context.get('optimizationSlots') or context.get('campaignSlots') or []
    compact = []
    for row in slots:
        compact.append((
            str(row.get('date', ''))[:10],
            str(row.get('interval', '')),
            normalizar_nombre_campana(row.get('campaign', '')),
            _assistant_norm_channel(row.get('channel', '')),
            round(float(row.get('required', 0) or 0), 2),
            round(float(row.get('actual', 0) or 0), 2)
        ))
    compact.sort()
    payload = {
        'mode': context.get('mode'),
        'worst': context.get('worst'),
        'slots': compact,
        'excel': excel_sig,
        'moves': int(max_moves)
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()

def _assistant_plan_cache_get(key):
    now = time.time()
    with _ASSISTANT_PLAN_CACHE_LOCK:
        item = _ASSISTANT_PLAN_CACHE.get(key)
        if not item:
            return None
        created, value = item
        if now - created > _ASSISTANT_PLAN_CACHE_TTL:
            _ASSISTANT_PLAN_CACHE.pop(key, None)
            return None
        return value

def _assistant_plan_cache_set(key, value):
    now = time.time()
    with _ASSISTANT_PLAN_CACHE_LOCK:
        _ASSISTANT_PLAN_CACHE[key] = (now, value)
        if len(_ASSISTANT_PLAN_CACHE) > _ASSISTANT_PLAN_CACHE_MAX:
            oldest = sorted(_ASSISTANT_PLAN_CACHE.items(), key=lambda kv: kv[1][0])
            for stale_key, _ in oldest[:max(1, len(oldest) - _ASSISTANT_PLAN_CACHE_MAX)]:
                _ASSISTANT_PLAN_CACHE.pop(stale_key, None)
    return value

def _assistant_candidate_effect(state, cand, current_metrics):
    old_set = set(cand['oldIntervals'])
    new_set = set(cand['newIntervals'])
    lost = old_set - new_set
    gained = new_set - old_set

    deficit_before = 0.0
    deficit_after = 0.0
    worst_relief = 0.0
    touched = False

    for interval, delta in [(x, -1.0) for x in lost] + [(x, 1.0) for x in gained]:
        key = (cand['date'], cand['campaignNorm'], cand['channelNorm'], interval)
        row = state.get(key)
        if not row:
            continue
        touched = True
        req = float(row['required'])
        actual = float(row['actual'])
        before_gap = actual - req
        after_actual = max(0.0, actual + delta)
        after_gap = after_actual - req

        # Never create a brand-new deficit in a previously covered interval.
        if before_gap >= 0 and after_gap < 0:
            return None

        deficit_before += max(0.0, -before_gap)
        deficit_after += max(0.0, -after_gap)

        if before_gap <= float(current_metrics.get('worst_gap', 0)) + 1e-9 and after_gap > before_gap:
            worst_relief = max(worst_relief, after_gap - before_gap)

    if not touched:
        return None

    reduction = deficit_before - deficit_after
    if reduction <= 0:
        return None

    score = reduction * 100.0 + worst_relief * 12.0 - (abs(cand['deltaMinutes']) / 30.0)
    return {
        'score': score,
        'reduction': reduction,
        'worst_relief': worst_relief
    }

def _assistant_apply_shift_inplace(state, cand):
    old_set = set(cand['oldIntervals'])
    new_set = set(cand['newIntervals'])
    for interval in old_set - new_set:
        key = (cand['date'], cand['campaignNorm'], cand['channelNorm'], interval)
        if key in state:
            state[key]['actual'] = max(0.0, float(state[key]['actual']) - 1.0)
    for interval in new_set - old_set:
        key = (cand['date'], cand['campaignNorm'], cand['channelNorm'], interval)
        if key in state:
            state[key]['actual'] = float(state[key]['actual']) + 1.0

def _assistant_optimize_roster(context, max_moves=6):
    slots = context.get('optimizationSlots') or context.get('campaignSlots') or []
    if not isinstance(slots, list) or not slots:
        return {
            'available': False,
            'reason': 'No se recibió el detalle por campaña e intervalo necesario para optimizar horarios.',
            'moves': []
        }

    excel_path = buscar_archivo_excel()
    if not excel_path:
        return {'available': False, 'reason': 'No se encontró el Excel fuente del roster.', 'moves': []}

    cache_key = _assistant_context_signature(context, excel_path, max_moves)
    cached_plan = _assistant_plan_cache_get(cache_key)
    if cached_plan is not None:
        return cached_plan

    roster = _assistant_roster_detail(excel_path)
    if not roster.get('available'):
        return _assistant_plan_cache_set(cache_key, {
            'available': False,
            'reason': roster.get('reason') or 'Roster no disponible.',
            'moves': [],
            'cached': False
        })

    baseline = _assistant_optimizer_state(slots)
    if not baseline:
        return _assistant_plan_cache_set(cache_key, {
            'available': False,
            'reason': 'No fue posible construir la cobertura base.',
            'moves': [],
            'cached': False
        })

    baseline_metrics = _assistant_state_metrics(baseline)
    if baseline_metrics['total_deficit'] <= 0:
        return _assistant_plan_cache_set(cache_key, {
            'available': True,
            'reason': 'La selección no tiene déficit que requiera movimientos.',
            'baseline': baseline_metrics,
            'after': baseline_metrics,
            'moves': [],
            'impactIntervals': [],
            'candidateCount': 0,
            'cached': False
        })

    active_campaigns = {key[1] for key in baseline.keys()}
    active_dates = sorted({key[0] for key in baseline.keys()})
    channels = sorted({key[2] for key in baseline.keys() if key[2]})
    worst = context.get('worst') or {}
    target_channel = _assistant_norm_channel(worst.get('channel', ''))
    if not target_channel:
        mode = _assistant_norm(context.get('mode', ''))
        target_channel = 'chat' if mode == 'chat' else ('llamadas' if mode == 'llamadas' else (channels[0] if len(channels) == 1 else ''))

    # Index channels once instead of scanning the whole state for every agent/date.
    state_channels_by_campaign_date = {}
    baseline_deficit_keys = set()
    for key, row in baseline.items():
        date, campaign_norm, channel_norm, interval = key
        state_channels_by_campaign_date.setdefault((date, campaign_norm), set()).add(channel_norm)
        if float(row['actual']) - float(row['required']) < 0:
            baseline_deficit_keys.add(key)

    candidate_instances = []
    for roster_agent in roster.get('agents', []):
        campaign_norm = normalizar_nombre_campana(roster_agent.get('campaign', ''))
        if campaign_norm not in active_campaigns:
            continue

        roster_channel = _assistant_norm_channel(roster_agent.get('channel', ''))
        if roster_channel and target_channel and roster_channel != target_channel:
            continue

        for date in active_dates:
            day_name = _assistant_day_name(date)
            sched = (roster_agent.get('schedules') or {}).get(day_name)
            if not sched:
                continue

            state_channels = sorted(state_channels_by_campaign_date.get((date, campaign_norm), set()))
            if target_channel and target_channel in state_channels:
                channel_norm = target_channel
            elif roster_channel and roster_channel in state_channels:
                channel_norm = roster_channel
            elif len(state_channels) == 1:
                channel_norm = state_channels[0]
            else:
                channel_norm = target_channel or (state_channels[0] if state_channels else '')
            if not channel_norm:
                continue

            start, end = int(sched['start']), int(sched['end'])
            old_intervals = _assistant_span_intervals(start, end)
            if not old_intervals:
                continue

            for delta in (-90, -60, -30, 30, 60, 90):
                new_start_abs = start + delta
                end_abs = end if end > start else end + (24 * 60)
                new_end_abs = end_abs + delta
                if new_start_abs < 0 or new_start_abs >= 24 * 60:
                    continue
                if new_end_abs <= new_start_abs or new_end_abs > 2 * 24 * 60:
                    continue

                new_start = new_start_abs % (24 * 60)
                new_end = new_end_abs % (24 * 60)
                new_intervals = _assistant_span_intervals(new_start, new_end)
                if not new_intervals:
                    continue
                if any(not esta_en_ventana_servicio(roster_agent.get('campaign', ''), interval) for interval in new_intervals):
                    continue

                old_set, new_set = set(old_intervals), set(new_intervals)
                gained = new_set - old_set

                # Major pruning: only evaluate moves that add coverage to a known deficit.
                if not any((date, campaign_norm, channel_norm, inv) in baseline_deficit_keys for inv in gained):
                    continue

                candidate_instances.append({
                    'agent': roster_agent.get('agent', 'Agente'),
                    'agentId': roster_agent.get('agentId') or roster_agent.get('agent', 'Agente'),
                    'fullName': roster_agent.get('fullName', ''),
                    'supervisor': roster_agent.get('supervisor', ''),
                    'coordinator': roster_agent.get('coordinator', ''),
                    'campaign': roster_agent.get('campaign', ''),
                    'campaignNorm': campaign_norm,
                    'channelNorm': channel_norm,
                    'channel': channel_norm.title(),
                    'date': date,
                    'day': day_name,
                    'currentSchedule': _assistant_schedule_text(start, end),
                    'proposedSchedule': _assistant_schedule_text(new_start, new_end),
                    'deltaMinutes': delta,
                    'oldIntervals': old_intervals,
                    'newIntervals': new_intervals
                })

    if not candidate_instances:
        return _assistant_plan_cache_set(cache_key, {
            'available': False,
            'reason': 'El roster tiene horarios, pero no encontré movimientos compatibles que agreguen cobertura a las franjas con déficit.',
            'baseline': baseline_metrics,
            'moves': [],
            'candidateCount': 0,
            'cached': False
        })

    # Mutable copy only once. Candidate evaluation no longer copies the full state.
    current = {k: {'required': float(v['required']), 'actual': float(v['actual'])} for k, v in baseline.items()}
    selected = []
    used_agent_dates = set()

    for _ in range(max(1, min(int(max_moves), 10))):
        current_metrics = _assistant_state_metrics(current)
        best = None

        for cand in candidate_instances:
            identity = (cand['agent'], cand['date'])
            if identity in used_agent_dates:
                continue

            effect = _assistant_candidate_effect(current, cand, current_metrics)
            if not effect:
                continue

            if best is None or effect['score'] > best['effect']['score']:
                best = {'candidate': cand, 'effect': effect}

        if best is None:
            break

        cand = best['candidate']
        before_metrics = current_metrics

        touched_intervals = set(cand['oldIntervals']) | set(cand['newIntervals'])
        touched_before = {}
        for interval in touched_intervals:
            key = (cand['date'], cand['campaignNorm'], cand['channelNorm'], interval)
            if key in current:
                touched_before[key] = {
                    'required': float(current[key]['required']),
                    'actual': float(current[key]['actual'])
                }

        _assistant_apply_shift_inplace(current, cand)
        after_metrics = _assistant_state_metrics(current)
        used_agent_dates.add((cand['agent'], cand['date']))

        touched_after = {}
        for key in touched_before:
            if key in current:
                touched_after[key] = {
                    'required': float(current[key]['required']),
                    'actual': float(current[key]['actual'])
                }

        move_impact = _assistant_improved_intervals(touched_before, touched_after, limit=8)
        action_raw = '|'.join([
            cache_key,
            str(cand['agent']),
            str(cand['campaign']),
            str(cand['date']),
            str(cand['currentSchedule']),
            str(cand['proposedSchedule'])
        ])
        action_id = hashlib.sha1(action_raw.encode('utf-8')).hexdigest()[:18]

        selected.append({
            'actionId': action_id,
            'agent': cand['agent'],
            'agentId': cand.get('agentId', cand['agent']),
            'fullName': cand.get('fullName', ''),
            'supervisor': cand.get('supervisor', ''),
            'coordinator': cand.get('coordinator', ''),
            'campaign': cand['campaign'],
            'channel': cand['channel'],
            'date': cand['date'],
            'day': cand['day'],
            'currentSchedule': cand['currentSchedule'],
            'proposedSchedule': cand['proposedSchedule'],
            'deltaMinutes': cand['deltaMinutes'],
            'deficitReduction': round(before_metrics['total_deficit'] - after_metrics['total_deficit'], 1),
            'worstGapBefore': before_metrics['worst_gap'],
            'worstGapAfter': after_metrics['worst_gap'],
            'impactIntervals': move_impact,
            'guardrail': 'Sin déficit nuevo dentro del alcance analizado'
        })

        if after_metrics['total_deficit'] <= 0:
            break

    after_metrics = _assistant_state_metrics(current)
    impact = _assistant_improved_intervals(baseline, current, limit=12)

    result = {
        'available': True,
        'sheet': roster.get('sheet'),
        'baseline': baseline_metrics,
        'after': after_metrics,
        'moves': selected,
        'impactIntervals': impact,
        'candidateCount': len(candidate_instances),
        'scopeDates': active_dates[:8],
        'reason': '' if selected else 'No encontré movimientos de ±30/60/90 min que reduzcan déficit sin crear otro faltante dentro del alcance analizado.',
        'cached': False
    }
    return _assistant_plan_cache_set(cache_key, result)


def _assistant_data_quality(context):
    excel_path = buscar_archivo_excel()
    info = {
        'forecastRows': int(context.get('rowCount', 0) or 0),
        'optimizationSlots': len(context.get('optimizationSlots') or []),
        'rosterAvailable': False,
        'rosterAgents': 0,
        'sourceFile': '',
        'sourceModified': ''
    }
    if not excel_path:
        return info
    try:
        roster = _assistant_roster_detail(excel_path)
        info['rosterAvailable'] = bool(roster.get('available'))
        info['rosterAgents'] = len(roster.get('agents') or [])
        info['sourceFile'] = os.path.basename(excel_path)
        st = os.stat(excel_path)
        info['sourceModified'] = datetime.fromtimestamp(st.st_mtime).isoformat(timespec='minutes')
    except Exception:
        pass
    return info

def _wfm_action_log_read():
    with _WFM_ACTION_LOG_LOCK:
        if not os.path.exists(WFM_ACTION_LOG_FILE):
            return []
        try:
            with open(WFM_ACTION_LOG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

def _wfm_action_log_write(records):
    with _WFM_ACTION_LOG_LOCK:
        tmp = WFM_ACTION_LOG_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp, WFM_ACTION_LOG_FILE)

def _wfm_action_stats(records):
    total = len(records)
    accepted = sum(1 for r in records if r.get('decision') == 'accepted')
    rejected = sum(1 for r in records if r.get('decision') == 'rejected')
    applied = sum(1 for r in records if r.get('implementationStatus') == 'applied')
    pending = sum(
        1 for r in records
        if r.get('decision') == 'accepted' and r.get('implementationStatus') != 'applied'
    )
    ops_resolved = sum(1 for r in records if _assistant_norm(r.get('actorMode')) in ('ops', 'operaciones'))
    accepted_reduction = sum(
        float((r.get('move') or {}).get('deficitReduction', 0) or 0)
        for r in records if r.get('decision') == 'accepted'
    )
    applied_reduction = sum(
        float((r.get('move') or {}).get('deficitReduction', 0) or 0)
        for r in records if r.get('implementationStatus') == 'applied'
    )
    return {
        'resolved': total,
        'accepted': accepted,
        'rejected': rejected,
        'applied': applied,
        'pendingApplication': pending,
        'opsResolved': ops_resolved,
        'opsAutonomy': round((ops_resolved / total) * 100) if total else 0,
        'acceptedEstimatedDeficitReduction': round(accepted_reduction, 1),
        'appliedEstimatedDeficitReduction': round(applied_reduction, 1)
    }

@app.route('/api/wfm_action_decisions', methods=['GET'])
def wfm_action_decisions():
    records = _wfm_action_log_read()
    records = sorted(records, key=lambda r: str(r.get('updatedAt', '')), reverse=True)
    try:
        limit = max(1, min(int(request.args.get('limit', 100)), 500))
    except Exception:
        limit = 100
    return jsonify({
        'records': records[:limit],
        'stats': _wfm_action_stats(records)
    }), 200

@app.route('/api/wfm_action_decision', methods=['POST'])
def wfm_action_decision():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        action_id = str(payload.get('actionId') or '').strip()[:80]
        decision = _assistant_norm(payload.get('decision'))
        if not action_id or decision not in ('accepted', 'rejected', 'applied'):
            return jsonify({'error': 'Decisión inválida.'}), 400

        move = payload.get('move') or {}
        actor_mode = str(payload.get('actorMode') or 'wfm')[:40]
        scope_label = str(payload.get('scopeLabel') or '')[:80]
        scope_detail = str(payload.get('scopeDetail') or '')[:120]
        now = datetime.now().isoformat(timespec='seconds')

        records = _wfm_action_log_read()
        existing = next((r for r in records if r.get('actionId') == action_id), None)

        if decision == 'applied':
            if not existing or existing.get('decision') != 'accepted':
                return jsonify({'error': 'Primero debes aprobar el movimiento antes de marcarlo como aplicado.'}), 400

            history = existing.setdefault('history', [])
            history.append({
                'decision': existing.get('decision'),
                'implementationStatus': existing.get('implementationStatus', 'pending'),
                'updatedAt': existing.get('updatedAt')
            })
            existing.update({
                'decision': 'accepted',
                'implementationStatus': 'applied',
                'appliedAt': now,
                'actorMode': actor_mode,
                'scopeLabel': scope_label,
                'scopeDetail': scope_detail,
                'move': move or existing.get('move') or {},
                'updatedAt': now
            })
            record = existing
        elif existing:
            history = existing.setdefault('history', [])
            history.append({
                'decision': existing.get('decision'),
                'implementationStatus': existing.get('implementationStatus'),
                'updatedAt': existing.get('updatedAt')
            })
            existing.update({
                'decision': decision,
                'implementationStatus': 'pending' if decision == 'accepted' else 'not_applicable',
                'actorMode': actor_mode,
                'scopeLabel': scope_label,
                'scopeDetail': scope_detail,
                'move': move,
                'updatedAt': now
            })
            if decision == 'accepted':
                existing['approvedAt'] = now
                existing.pop('appliedAt', None)
            else:
                existing['rejectedAt'] = now
                existing.pop('approvedAt', None)
                existing.pop('appliedAt', None)
            record = existing
        else:
            record = {
                'actionId': action_id,
                'decision': decision,
                'implementationStatus': 'pending' if decision == 'accepted' else 'not_applicable',
                'actorMode': actor_mode,
                'scopeLabel': scope_label,
                'scopeDetail': scope_detail,
                'move': move,
                'createdAt': now,
                'updatedAt': now,
                'history': []
            }
            if decision == 'accepted':
                record['approvedAt'] = now
            else:
                record['rejectedAt'] = now
            records.append(record)

        _wfm_action_log_write(records)
        return jsonify({
            'ok': True,
            'record': record,
            'stats': _wfm_action_stats(records)
        }), 200
    except Exception as e:
        return jsonify({'error': f'No se pudo registrar la decisión: {str(e)}'}), 500

@app.route('/api/wfm_assistant', methods=['POST'])
def wfm_assistant():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        intent = _assistant_norm(payload.get('intent', 'chat'))
        question = str(payload.get('question', '')).strip()[:500]
        context = payload.get('context') or {}
        if not isinstance(context, dict) or int(context.get('rowCount', 0) or 0) <= 0:
            return jsonify({'error': 'No hay contexto de forecast disponible para analizar.'}), 400
        if intent != 'auto' and not question:
            return jsonify({'error': 'Escribe una pregunta para el asistente.'}), 400

        q = _assistant_norm(question)
        analysis_scope = str(context.get('analysisScope') or 'day').lower()
        scope_label = str(context.get('scopeLabel') or {'month':'Mensual','week':'Semanal','day':'Diario'}.get(analysis_scope, 'Diario'))
        scope_detail = str(context.get('scopeDetail') or 'periodo seleccionado')
        deficits = context.get('deficits') or []
        surpluses = context.get('surpluses') or []
        worst = context.get('worst') or None
        campaigns = context.get('campaigns') or []
        cards = _assistant_cards(context)
        period = context.get('periodCapacity') or {}
        period_required = int(round(float(period.get('required', 0) or 0)))
        period_actual = int(round(float(period.get('actual', 0) or 0)))
        period_gap = int(round(float(period.get('gap', period_actual - period_required) or 0)))
        period_coverage = int(round(float(period.get('coverage', 100) or 100)))
        actions = []
        note = 'Las sugerencias son de planeación. Antes de aplicar cambios valida jornada, descansos, skills, ausentismo y restricciones laborales.'

        if intent == 'auto':
            severity = 'ok'
            schedule_plan = None
            channel_label = 'Consolidado' if _assistant_norm(context.get('mode')) == 'general' else ('Chat' if _assistant_norm(context.get('mode')) == 'chat' else 'Llamadas')

            if period_gap < 0:
                severity = 'critical' if period_gap <= -5 or period_coverage < 85 else 'warning'
                headline = f"{channel_label} · {period_gap} HC · {period_coverage}%"
                if period_gap <= -2 or period_coverage < 95:
                    schedule_plan = _assistant_optimize_roster(context, max_moves=4)

                reply = (
                    f"En el periodo {scope_detail} se requieren {period_required} HC y hay {period_actual} HC; "
                    f"la brecha de plantilla es de {abs(period_gap)} HC y la cobertura es {period_coverage}%."
                )
                if worst:
                    reply += (
                        f" El intervalo más presionado es {worst.get('date','')} a las {worst.get('interval','')} "
                        f"con una brecha operativa de {int(round(float(worst.get('gap',0))))} HC."
                    )
                if schedule_plan and schedule_plan.get('moves'):
                    before = schedule_plan.get('baseline', {})
                    after = schedule_plan.get('after', {})
                    reply += (
                        f" Encontré {len(schedule_plan['moves'])} movimientos candidatos; en la simulación por intervalos "
                        f"la exposición baja de {before.get('total_deficit',0):g} a {after.get('total_deficit',0):g} HC-intervalo."
                    )
            else:
                headline = f"{channel_label} · cobertura estable"
                reply = (
                    f"El periodo {scope_detail} está cubierto: {period_actual} HC actuales vs "
                    f"{period_required} requeridos ({period_coverage}% de cobertura)."
                )

            return jsonify({
                'autonomous': True,
                'analysisScope': analysis_scope,
                'scopeLabel': scope_label,
                'scopeDetail': scope_detail,
                'severity': severity,
                'headline': headline,
                'reply': reply,
                'cards': cards,
                'actions': [],
                'schedule_plan': schedule_plan,
                'data_quality': _assistant_data_quality(context),
                'note': note
            }), 200

        metric_answer = _assistant_explain_metric(question)
        if metric_answer:
            reply = (
                f"{metric_answer} Para el periodo {scope_detail}: {period_required} HC requeridos, "
                f"{period_actual} HC actuales y una brecha de {period_gap} HC."
            )
            if worst:
                reply += f" El intervalo más presionado está en {worst.get('date','')} a las {worst.get('interval','')}."
            return jsonify({
                'reply': reply,
                'scopeLabel': scope_label,
                'scopeDetail': scope_detail,
                'cards': cards,
                'actions': [],
                'note': note
            }), 200

        movement_terms = ['mover', 'movimiento', 'movimientos', 'horario', 'horarios', 'turno', 'reacomodar', 'redistribuir', 'ajuste', 'optimiza', 'optimizar']
        deficit_terms = ['déficit', 'deficit', 'faltante', 'falta', 'riesgo', 'cobertura', 'crítico', 'critico']
        why_terms = ['por qué', 'porque', 'requerido', 'necesito', 'necesidad', 'hc requerido']
        summary_terms = ['resumen', 'prioridad', 'prioridades', 'acciones', 'qué debo hacer', 'que debo hacer']

        if any(t in q for t in movement_terms):
            schedule_plan = _assistant_optimize_roster(context, max_moves=6)
            if period_gap < 0:
                reply = (
                    f"Para el periodo {scope_detail} la brecha de plantilla es de {abs(period_gap)} HC: "
                    f"{period_actual} actuales vs {period_required} requeridos ({period_coverage}% de cobertura)."
                )
                if worst:
                    reply += f" La franja más presionada es {worst.get('date','')} a las {worst.get('interval','')}."
            else:
                reply = (
                    f"El periodo {scope_detail} está cubierto con {period_actual} HC actuales vs "
                    f"{period_required} requeridos."
                )

            if schedule_plan.get('moves'):
                actions = [
                    f"{m['agent']} · {m['campaign']} · {m['date']}: {m['currentSchedule']} → {m['proposedSchedule']} "
                    f"({m['deltaMinutes']:+d} min)."
                    for m in schedule_plan['moves']
                ]
                before = schedule_plan.get('baseline', {})
                after = schedule_plan.get('after', {})
                reply += (
                    f" Encontré {len(schedule_plan['moves'])} movimientos concretos de roster que reducen el déficit "
                    f"HC-intervalo de {before.get('total_deficit',0):g} a {after.get('total_deficit',0):g} "
                    f"sin crear un faltante nuevo dentro del alcance analizado."
                )
            else:
                actions = _assistant_movement_actions(deficits, surpluses)
                reason = schedule_plan.get('reason') or ''
                if reason:
                    reply += f" {reason}"
                elif actions:
                    reply += ' Encontré capacidad cercana para revisar, aunque no pude asociarla a personas específicas del roster.'
                else:
                    reply += ' No encontré un movimiento seguro con los datos actuales.'

            note = (
                'Simulación de planeación: los cambios no se aplican automáticamente. '
                'Valida jornada, skills, transporte, restricciones laborales y cualquier excepción individual.'
            )
            return jsonify({
                'reply': reply,
                'scopeLabel': scope_label,
                'scopeDetail': scope_detail,
                'cards': cards,
                'actions': actions,
                'schedule_plan': schedule_plan,
                'data_quality': _assistant_data_quality(context),
                'note': note
            }), 200
        elif any(t in q for t in deficit_terms):
            if period_gap < 0:
                reply = (
                    f"Para el periodo {scope_detail} se requieren {period_required} HC y hay {period_actual} HC. "
                    f"Faltan {abs(period_gap)} HC y la cobertura de plantilla es {period_coverage}%."
                )
                if worst:
                    reply += (
                        f" El intervalo más presionado es {worst.get('date','')} a las {worst.get('interval','')}, "
                        f"donde la brecha operativa es {int(round(float(worst.get('gap',0))))} HC."
                    )
                actions = ['Abrir Plan para revisar candidatos de entrada/salida y simular su impacto.']
            else:
                reply = (
                    f"El periodo {scope_detail} no presenta déficit de plantilla: {period_actual} HC actuales vs "
                    f"{period_required} requeridos ({period_coverage}% de cobertura)."
                )
        elif any(t in q for t in why_terms):
            reply = (
                f"Para el periodo {scope_detail}, la necesidad de plantilla es {period_required} HC y el roster actual es "
                f"{period_actual} HC, por lo que la brecha es {period_gap} HC. El HC requerido se obtiene de la carga "
                f"proyectada y después se convierte a plantilla considerando jornada, descanso y merma."
            )
            if worst:
                reply += f" La franja más exigente está en {worst.get('date','')} a las {worst.get('interval','')}."
        elif any(t in q for t in summary_terms):
            if period_gap < 0:
                reply = (
                    f"Resumen del periodo {scope_detail}: {period_required} HC requeridos, {period_actual} HC actuales, "
                    f"brecha de {abs(period_gap)} HC y cobertura de {period_coverage}%."
                )
                if worst:
                    reply += f" Prioridad intraperiodo: {worst.get('date','')} {worst.get('interval','')}."
                actions = ['Revisar el Plan de acción y validar los movimientos de roster con mayor impacto.']
            else:
                reply = (
                    f"El periodo {scope_detail} está cubierto: {period_actual} HC actuales vs "
                    f"{period_required} requeridos ({period_coverage}% de cobertura)."
                )
        else:
            if period_gap < 0:
                reply = (
                    f"El periodo {scope_detail} tiene una brecha de plantilla de {abs(period_gap)} HC: "
                    f"{period_actual} actuales vs {period_required} requeridos ({period_coverage}% de cobertura)."
                )
                if worst:
                    reply += f" El punto más presionado está en {worst.get('date','')} a las {worst.get('interval','')}."
                reply += ' Puedes preguntarme por déficit, movimientos de horarios, HC requerido, AHT, merma, SL, ASA o concurrencia.'
            else:
                reply = (
                    f"El periodo {scope_detail} está cubierto con {period_actual} HC actuales vs "
                    f"{period_required} requeridos. Puedes preguntarme por proyección, roster o prioridades operativas."
                )

        return jsonify({
            'reply': reply,
            'scopeLabel': scope_label,
            'scopeDetail': scope_detail,
            'cards': cards,
            'actions': actions,
            'note': note
        }), 200
    except Exception as e:
        return jsonify({'error': f'No se pudo ejecutar el asistente WFM: {str(e)}'}), 500

_roster_db_init()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


