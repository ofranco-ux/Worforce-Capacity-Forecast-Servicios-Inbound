import os
import math
import gc
import re
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import pandas as pd
import numpy as np

# --- LIBRERÍAS DE MACHINE LEARNING ---
import holidays
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE_IN = os.path.join(BASE_DIR, 'forecast_cache_in.json')
CACHE_FILE_OUT = os.path.join(BASE_DIR, 'forecast_cache_out.json')
CACHE_FILE_CHAT = os.path.join(BASE_DIR, 'forecast_cache_chat.json')
CONFIG_FILE = os.path.join(BASE_DIR, 'wfm_config.json') 
EXCEL_DEFAULT = os.path.join(BASE_DIR, 'historico.xlsx')

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

# =====================================================================
# 🧠 MOTOR WFM ULTRA-PERRÓN (Medianas Inmunes y Anti-Serrucho)
# =====================================================================
def pronosticar_con_machine_learning(df_diario_campana, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls):
    df_ml = df_diario_campana.sort_values(col_fecha).copy()
    
    anos_presentes = list(df_ml[col_fecha].dt.year.unique())
    anos_presentes.append(fecha_inicio_forecast.year)
    anos_presentes.append((fecha_inicio_forecast + timedelta(days=dias_futuros)).year)
    anos_unicos = list(set(anos_presentes))
    
    festivos_pais = holidays.CountryHoliday('MX', years=anos_unicos)
    df_ml['dia_semana'] = df_ml[col_fecha].dt.weekday
    
    # BASE ESTABLE: Promedio Olímpico (A prueba de balas)
    dow_olimpico = {}
    for i in range(7):
        vols_dow = df_ml[(df_ml['dia_semana'] == i) & (df_ml[col_calls] > 0)][col_calls]
        vols_list = vols_dow.tail(5).tolist()
        if len(vols_list) >= 4:
            vols_list.sort()
            dow_olimpico[i] = float(np.mean(vols_list[1:-1]))
        elif len(vols_list) > 0:
            dow_olimpico[i] = float(np.mean(vols_list))
        else:
            dow_olimpico[i] = float(df_ml[col_calls].mean())

    def cap_outliers(group):
        if len(group) < 4: return group
        q1, q3 = group.quantile(0.25), group.quantile(0.75)
        iqr = q3 - q1
        return np.clip(group, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        
    df_ml['calls_smooth'] = df_ml.groupby('dia_semana')[col_calls].transform(cap_outliers) if len(df_ml) >= 14 else df_ml[col_calls]

    df_ml['lag_1'] = df_ml['calls_smooth'].shift(1)
    df_ml['lag_2'] = df_ml['calls_smooth'].shift(2)
    df_ml['lag_7'] = df_ml['calls_smooth'].shift(7)
    df_ml['lag_14'] = df_ml['calls_smooth'].shift(14)
    
    df_ml['rolling_mean_3'] = df_ml['calls_smooth'].shift(1).rolling(window=3, min_periods=1).mean()
    df_ml['rolling_mean_7'] = df_ml['calls_smooth'].shift(1).rolling(window=7, min_periods=1).mean()
    
    df_ml['dia_mes'] = df_ml[col_fecha].dt.day
    df_ml['es_quincena'] = df_ml['dia_mes'].apply(lambda x: 1 if x in [14, 15, 16, 29, 30, 31, 1] else 0)
    df_ml['es_festivo'] = df_ml[col_fecha].apply(lambda x: 1 if x in festivos_pais else 0)
    
    df_train = df_ml.dropna().copy()
    vol_promedio_historico = df_ml[col_calls].mean()
    
    if len(df_train) < 14:
        return [max(0.0, float(vol_promedio_historico))] * dias_futuros

    features = ['lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_3', 'rolling_mean_7', 'dia_semana', 'dia_mes', 'es_quincena', 'es_festivo']
    modelo = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=7)
    modelo.fit(df_train[features], df_train['calls_smooth'])
    
    # DETECTOR DE DESFASE ESTRUCTURAL
    ultimos_7_media = df_diario_campana.tail(7)[col_calls].mean()
    ultimos_14 = df_diario_campana.tail(14)[col_calls]
    media_14 = ultimos_14.mean()
    std_14 = ultimos_14.std()
    cv = (std_14 / media_14) if media_14 > 0 else 0
    
    historico_previo = df_diario_campana.iloc[:-7][col_calls].mean() if len(df_diario_campana) > 7 else ultimos_7_media
    desfase_agresivo = True if (ultimos_7_media < historico_previo * 0.85) or (ultimos_7_media > historico_previo * 1.15) else False
    
    # LA MAGIA: Pesos por Mediana (Inmunes a outliers de fin de semana)
    df_recent_4w = df_diario_campana.tail(28).copy()
    df_recent_4w['wd'] = df_recent_4w[col_fecha].dt.weekday
    dow_medians = df_recent_4w.groupby('wd')[col_calls].median()
    if dow_medians.sum() == 0: 
        dow_medians = df_recent_4w.groupby('wd')[col_calls].mean()
    overall_median = dow_medians.mean()
    dow_weights = {k: (v / overall_median if overall_median > 0 else 1.0) for k, v in dow_medians.items()}

    historial_simulado = df_ml.to_dict('records')
    preds_finales_ajustadas = []
    fecha_actual = fecha_inicio_forecast
    
    for d in range(dias_futuros):
        vols_smooth = [r.get('calls_smooth', r.get(col_calls, 0)) for r in historial_simulado]
        
        lag_1_val = vols_smooth[-1]
        lag_2_val = vols_smooth[-2] if len(vols_smooth) >= 2 else lag_1_val
        lag_7_val = vols_smooth[-7] if len(vols_smooth) >= 7 else lag_1_val
        lag_14_val = vols_smooth[-14] if len(vols_smooth) >= 14 else lag_7_val
        rm_3_val = np.mean(vols_smooth[-3:]) if len(vols_smooth) >= 3 else np.mean(vols_smooth)
        rm_7_val = np.mean(vols_smooth[-7:]) if len(vols_smooth) >= 7 else np.mean(vols_smooth)
        
        X_pred = pd.DataFrame([{
            'lag_1': lag_1_val,
            'lag_2': lag_2_val,
            'lag_7': lag_7_val,
            'lag_14': lag_14_val,
            'rolling_mean_3': rm_3_val,
            'rolling_mean_7': rm_7_val,
            'dia_semana': fecha_actual.weekday(),
            'dia_mes': fecha_actual.day,
            'es_quincena': 1 if fecha_actual.day in [14, 15, 16, 29, 30, 31, 1] else 0,
            'es_festivo': 1 if fecha_actual in festivos_pais else 0
        }])
        
        pred_ml = float(modelo.predict(X_pred[features])[0])
        pred_ml = max(0.0, pred_ml)
        wd = fecha_actual.weekday()
        base_oli = dow_olimpico.get(wd, media_14)
        
        base_reciente = ultimos_7_media * dow_weights.get(wd, 1.0)
        
        # ÁRBITRO ENSAMBLADO
        if desfase_agresivo or media_14 < 200:
            peso_ml, peso_oli, peso_rec = 0.10, 0.05, 0.85
        elif cv < 0.15 and vol_promedio_historico >= 200:
            peso_ml, peso_oli, peso_rec = 0.20, 0.70, 0.10
        else:
            peso_ml, peso_oli, peso_rec = 0.40, 0.30, 0.30

        pred_ensamblada = (pred_ml * peso_ml) + (base_oli * peso_oli) + (base_reciente * peso_rec)
        pred_final = max(0.0, float(pred_ensamblada))
        
        # Inercia Corta
        if media_14 < 200:
            tendencia = min(1.10, max(0.90, rm_3_val / rm_7_val if rm_7_val > 0 else 1.0))
            pred_final = pred_final * (1.0 + ((tendencia - 1.0) * max(0.0, 1.0 - (d * 0.15))))
        
        preds_finales_ajustadas.append(pred_final)
        
        historial_simulado.append({
            col_fecha: fecha_actual,
            col_calls: pred_final,
            'calls_smooth': pred_final
        })
        fecha_actual += timedelta(days=1)
        
    return preds_finales_ajustadas

# =====================================================================
# ⚙️ MÓDULOS AUXILIARES Y DISTRIBUCIÓN INTRADÍA
# =====================================================================
def buscar_archivo_excel():
    if os.path.exists(EXCEL_DEFAULT): return EXCEL_DEFAULT
    try:
        archivos = [f for f in os.listdir(BASE_DIR) if f.lower().endswith('.xlsx') and not f.startswith('~')]
        if not archivos: return None
        for f in archivos:
            if 'historico' in f.lower(): return os.path.join(BASE_DIR, f)
        return os.path.join(BASE_DIR, archivos[0])
    except: return None

@app.route('/')
@app.route('/index.html')
def serve_index():
    rutas_a_buscar = [BASE_DIR, os.getcwd(), os.path.dirname(BASE_DIR)]
    for ruta in rutas_a_buscar:
        target_path = os.path.join(ruta, 'index.html')
        if os.path.exists(target_path):
            response = make_response(send_from_directory(ruta, 'index.html'))
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
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
    validos = [n for n in range(base_n, base_n + 150) if erlang_c_sl_optimizado(A, n, aht, target_time) >= target_sl]
    return validos[0] if validos else base_n

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

def procesar_archivo_excel(file_source, target_sl=80.0, target_time=20.0, merma=0.20, dias_futuros=45):
    xls_file = pd.ExcelFile(file_source, engine='openpyxl')
    sheet_calls = xls_file.sheet_names[0]
    for s in xls_file.sheet_names:
        if 'llam' in s.lower() or 'hist' in s.lower() or 'datos' in s.lower(): sheet_calls = s; break
            
    sheet_roster = None
    for s in xls_file.sheet_names:
        s_lower = s.lower()
        if ('roster' in s_lower or 'plantilla' in s_lower or 'platilla' in s_lower or 'horario' in s_lower) and 'out' not in s_lower and 'salida' not in s_lower and 'chat' not in s_lower and 'mensaje' not in s_lower: 
            sheet_roster = s
            break
            
    if not sheet_roster:
        for s in xls_file.sheet_names:
            if 'roster' in s.lower() or 'plantilla' in s.lower() or 'platilla' in s_lower: 
                sheet_roster = s; break

    roster_coverage, roster_total_camp, roster_total_dia_camp = {}, {}, {}
    if sheet_roster:
        try:
            df_roster = pd.read_excel(xls_file, sheet_name=sheet_roster, engine='openpyxl')
            roster_coverage, roster_total_camp, roster_total_dia_camp = procesar_hoja_roster(df_roster)
        except: pass

    df_raw = pd.read_excel(xls_file, sheet_name=sheet_calls, engine='openpyxl')
    col_calls = encontrar_columna(df_raw, ['recibidas', 'llamadas', 'calls', 'volumen', 'ofrecidas', 'entrada'])
    col_aht = encontrar_columna(df_raw, ['aht', 'tmo', 'handle', 'duracion'])
    col_camp = encontrar_columna(df_raw, ['campaña', 'campana', 'skill', 'servicio', 'ring group'])
    col_inter = encontrar_columna(df_raw, ['intervalo', 'hora', 'time'])
    col_fecha = encontrar_columna(df_raw, ['fecha', 'date'])

    df_raw[col_camp] = df_raw[col_camp].astype(str).str.strip().str.title()
    df_raw[col_fecha] = pd.to_datetime(df_raw[col_fecha], dayfirst=True, errors='coerce').dt.normalize()
    df_raw = df_raw.dropna(subset=[col_fecha])
    df_raw[col_calls] = [clean_num(x, 0.0) for x in df_raw[col_calls]]

    df_valido = df_raw[df_raw[col_calls] > 0]
    if df_valido.empty: raise ValueError("El archivo no tiene volumen mayor a cero.")
    
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
    campanas_unicas = df[col_camp].unique()

    predicciones_futuras = {}

    for camp in campanas_unicas:
        sub = df_diario[df_diario[col_camp] == camp].sort_values(col_fecha).reset_index(drop=True)
        if sub.empty: continue
        preds_finales = pronosticar_con_machine_learning(sub, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls)
        predicciones_futuras[camp] = preds_finales

    df['En_Ventana'] = [esta_en_ventana_servicio(c, i) for c, i in zip(df[col_camp], df['Inter_Clean'])]
    df_filtrado = df[df['En_Ventana']].copy()

    df_reciente = df_filtrado[df_filtrado[col_fecha] >= (max_fecha_real - timedelta(days=28))]
    if df_reciente.empty: df_reciente = df_filtrado.copy()
    
    # 1. PERFIL POR DÍA DE LA SEMANA (Para Campañas Macro)
    perfil_dia = df_reciente.groupby([col_camp, 'Dia_Semana_Clean', 'Inter_Clean']).agg(
        total_calls=(col_calls, 'sum'),
        avg_aht=(col_aht, lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    ).reset_index()
    totales_dia = perfil_dia.groupby([col_camp, 'Dia_Semana_Clean'])['total_calls'].transform('sum')
    perfil_dia['weight'] = np.where(totales_dia > 0, perfil_dia['total_calls'] / totales_dia, 0)
    mapa_dia = {(r[col_camp], r['Dia_Semana_Clean'], r['Inter_Clean']): {'weight': r['weight'], 'aht': r['avg_aht']} for _, r in perfil_dia.iterrows()}

    # 2. PERFIL GLOBAL SUAVIZADO (Filtro Anti-Serrucho para Campañas Micro)
    perfil_global = df_reciente.groupby([col_camp, 'Inter_Clean']).agg(total_calls=(col_calls, 'sum')).reset_index()
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
        vol_historico_camp = float(df_diario[df_diario[col_camp] == camp][col_calls].mean())
        es_micro_campana = vol_historico_camp < 250 # Umbral de activación del Anti-Serrucho

        for d in range(dias_futuros):
            fecha_actual = fecha_inicio_forecast + timedelta(days=d)
            str_fecha = fecha_actual.strftime('%Y-%m-%d')
            str_mes = f"{meses_espanol[fecha_actual.month]} {fecha_actual.year}"
            nombre_dia = dias_espanol[fecha_actual.weekday()]

            vol_diario = predicciones_futuras.get(camp, [0]*dias_futuros)[d]
            intervalos_validos = intervalos_operativos_por_camp.get(camp, [])

            pesos_crudos = []
            for inter in intervalos_validos:
                if es_micro_campana:
                    # Aplica la Curva Global para aplanar los picos aleatorios de 1 sola llamada
                    w = mapa_perfil_global.get((camp, inter), 0.0)
                else:
                    w = mapa_dia.get((camp, nombre_dia, inter), {}).get('weight', 0.0)
                    if w == 0.0: w = mapa_perfil_global.get((camp, inter), 0.0)
                pesos_crudos.append(w)

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

                req_ftes = (calls_float * aht) / 1800.0 if (aht > 0 and calls_float > 0) else 0.0
                req_hc = math.ceil(req_ftes / factor_asistencia) if req_ftes > 0 else 0
                
                hc_roster = roster_coverage.get((str(camp), nombre_dia.capitalize(), inter), 0)
                tot_camp = roster_total_camp.get(str(camp), 0)
                tot_camp_dia = roster_total_dia_camp.get((str(camp), nombre_dia.capitalize()), 0)

                data_processed.append({
                    'Campaña': str(camp),
                    'Fecha': str_fecha,
                    'Mes': str_mes,
                    'Día_Semana': nombre_dia.capitalize(),
                    'Intervalo': inter,
                    'Llamadas': calls_int,
                    'AHT': format_aht_str(aht),
                    'AHT_Segundos': int(round(aht)),
                    'Agentes_Requeridos': req_hc,
                    'HC_Actual_Roster': hc_roster,
                    'Total_Roster_Campana': tot_camp,
                    'Total_Roster_Dia': tot_camp_dia,
                    'Factor_Correccion': 1.0
                })

    try:
        with open(CACHE_FILE_IN, 'w', encoding='utf-8') as f: json.dump(data_processed, f)
    except: pass
    return data_processed

def procesar_archivo_outbound(file_source, merma=0.20, dias_futuros=45):
    xls_file = pd.ExcelFile(file_source, engine='openpyxl')
    
    sheet_out = None
    for s in xls_file.sheet_names:
        if 'out' in s.lower() or 'salida' in s.lower(): 
            sheet_out = s
            break
            
    if not sheet_out:
        raise ValueError("No se encontró una pestaña llamada 'Out' o 'Salida' en el archivo Excel para procesar Outbound.")

    sheet_roster = None
    for s in xls_file.sheet_names:
        s_lower = s.lower()
        if ('plantilla' in s_lower or 'platilla' in s_lower or 'roster' in s_lower) and ('out' in s_lower or 'salida' in s_lower):
            sheet_roster = s
            break

    roster_coverage, roster_total_camp, roster_total_dia_camp = {}, {}, {}
    if sheet_roster:
        try:
            df_roster = pd.read_excel(xls_file, sheet_name=sheet_roster, engine='openpyxl')
            roster_coverage, roster_total_camp, roster_total_dia_camp = procesar_hoja_roster(df_roster)
        except: pass

    df_raw = pd.read_excel(xls_file, sheet_name=sheet_out, engine='openpyxl')
    col_calls = encontrar_columna(df_raw, ['realizadas', 'llamadas', 'out'])
    col_aht = encontrar_columna(df_raw, ['aht', 'tmo', 'handle', 'duracion'])
    col_camp = encontrar_columna(df_raw, ['campaña', 'campana', 'skill'])
    col_inter = encontrar_columna(df_raw, ['intervalo', 'hora', 'time'])
    col_fecha = encontrar_columna(df_raw, ['fecha', 'date'])

    df_raw[col_camp] = df_raw[col_camp].astype(str).str.strip().str.title()
    df_raw[col_fecha] = pd.to_datetime(df_raw[col_fecha], dayfirst=True, errors='coerce').dt.normalize()
    df_raw = df_raw.dropna(subset=[col_fecha])
    df_raw[col_calls] = [clean_num(x, 0.0) for x in df_raw[col_calls]]

    df_valido = df_raw[df_raw[col_calls] > 0]
    if df_valido.empty: raise ValueError("El archivo Outbound no tiene volumen mayor a cero.")
    
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
    campanas_unicas = df[col_camp].unique()

    predicciones_futuras = {}

    for camp in campanas_unicas:
        sub = df_diario[df_diario[col_camp] == camp].sort_values(col_fecha).reset_index(drop=True)
        if sub.empty: continue
        preds_finales = pronosticar_con_machine_learning(sub, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls)
        predicciones_futuras[camp] = preds_finales

    df['En_Ventana'] = [esta_en_ventana_servicio(c, i) for c, i in zip(df[col_camp], df['Inter_Clean'])]
    df_filtrado = df[df['En_Ventana']].copy()

    df_reciente = df_filtrado[df_filtrado[col_fecha] >= (max_fecha_real - timedelta(days=28))]
    if df_reciente.empty: df_reciente = df_filtrado.copy()
    
    perfil_dia = df_reciente.groupby([col_camp, 'Dia_Semana_Clean', 'Inter_Clean']).agg(
        total_calls=(col_calls, 'sum'),
        avg_aht=(col_aht, lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    ).reset_index()
    totales_dia = perfil_dia.groupby([col_camp, 'Dia_Semana_Clean'])['total_calls'].transform('sum')
    perfil_dia['weight'] = np.where(totales_dia > 0, perfil_dia['total_calls'] / totales_dia, 0)
    mapa_dia = {(r[col_camp], r['Dia_Semana_Clean'], r['Inter_Clean']): {'weight': r['weight'], 'aht': r['avg_aht']} for _, r in perfil_dia.iterrows()}

    perfil_global = df_reciente.groupby([col_camp, 'Inter_Clean']).agg(total_calls=(col_calls, 'sum')).reset_index()
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
        vol_historico_camp = float(df_diario[df_diario[col_camp] == camp][col_calls].mean())
        es_micro_campana = vol_historico_camp < 250

        for d in range(dias_futuros):
            fecha_actual = fecha_inicio_forecast + timedelta(days=d)
            str_fecha = fecha_actual.strftime('%Y-%m-%d')
            str_mes = f"{meses_espanol[fecha_actual.month]} {fecha_actual.year}"
            nombre_dia = dias_espanol[fecha_actual.weekday()]

            vol_diario = predicciones_futuras.get(camp, [0]*dias_futuros)[d]
            intervalos_validos = intervalos_operativos_por_camp.get(camp, [])

            pesos_crudos = []
            for inter in intervalos_validos:
                if es_micro_campana:
                    w = mapa_perfil_global.get((camp, inter), 0.0)
                else:
                    w = mapa_dia.get((camp, nombre_dia, inter), {}).get('weight', 0.0)
                    if w == 0.0: w = mapa_perfil_global.get((camp, inter), 0.0)
                pesos_crudos.append(w)

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

                req_ftes = (calls_float * aht) / 1800.0 if (aht > 0 and calls_float > 0) else 0.0
                req_hc = math.ceil(req_ftes / factor_asistencia) if req_ftes > 0 else 0
                
                hc_roster = roster_coverage.get((str(camp), nombre_dia.capitalize(), inter), 0)
                tot_camp = roster_total_camp.get(str(camp), 0)
                tot_camp_dia = roster_total_dia_camp.get((str(camp), nombre_dia.capitalize()), 0)

                data_processed.append({
                    'Campaña': str(camp),
                    'Fecha': str_fecha,
                    'Mes': str_mes,
                    'Día_Semana': nombre_dia.capitalize(),
                    'Intervalo': inter,
                    'Llamadas': calls_int,
                    'AHT': format_aht_str(aht),
                    'AHT_Segundos': int(round(aht)),
                    'Agentes_Requeridos': req_hc,
                    'HC_Actual_Roster': hc_roster,
                    'Total_Roster_Campana': tot_camp,
                    'Total_Roster_Dia': tot_camp_dia,
                    'Factor_Correccion': 1.0
                })

    try:
        with open(CACHE_FILE_OUT, 'w', encoding='utf-8') as f: json.dump(data_processed, f)
    except: pass
    return data_processed

def procesar_archivo_chat(file_source, target_sl=80.0, target_time=20.0, merma=0.20, concurrencia=3.0, dias_futuros=45):
    xls_file = pd.ExcelFile(file_source, engine='openpyxl')
    
    sheet_chat = None
    for s in xls_file.sheet_names:
        if ('chat' in s.lower() or 'mensaje' in s.lower()) and ('plantilla' not in s.lower() and 'roster' not in s.lower() and 'platilla' not in s.lower()): 
            sheet_chat = s
            break
            
    if not sheet_chat:
        raise ValueError("No se encontró una pestaña llamada 'Chat' o 'Mensajes' en el archivo Excel.")

    sheet_roster = None
    for s in xls_file.sheet_names:
        s_lower = s.lower()
        if ('plantilla' in s_lower or 'platilla' in s_lower or 'roster' in s_lower) and ('chat' in s_lower or 'mensaje' in s_lower):
            sheet_roster = s
            break

    roster_coverage, roster_total_camp, roster_total_dia_camp = {}, {}, {}
    if sheet_roster:
        try:
            df_roster = pd.read_excel(xls_file, sheet_name=sheet_roster, engine='openpyxl')
            roster_coverage, roster_total_camp, roster_total_dia_camp = procesar_hoja_roster(df_roster)
        except: pass

    df_raw = pd.read_excel(xls_file, sheet_name=sheet_chat, engine='openpyxl')
    col_calls = encontrar_columna(df_raw, ['recibidos', 'recibidas', 'llamadas', 'chats', 'mensajes'])
    col_aht = encontrar_columna(df_raw, ['aht', 'tmo', 'handle', 'duracion'])
    col_camp = encontrar_columna(df_raw, ['campaña', 'campana', 'skill'])
    col_inter = encontrar_columna(df_raw, ['intervalo', 'hora', 'time'])
    col_fecha = encontrar_columna(df_raw, ['fecha', 'date'])

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
    campanas_unicas = df[col_camp].unique()

    predicciones_futuras = {}

    for camp in campanas_unicas:
        sub = df_diario[df_diario[col_camp] == camp].sort_values(col_fecha).reset_index(drop=True)
        if sub.empty: continue
        preds_finales = pronosticar_con_machine_learning(sub, dias_futuros, fecha_inicio_forecast, col_fecha, col_calls)
        predicciones_futuras[camp] = preds_finales

    df['En_Ventana'] = [esta_en_ventana_servicio(c, i) for c, i in zip(df[col_camp], df['Inter_Clean'])]
    df_filtrado = df[df['En_Ventana']].copy()

    df_reciente = df_filtrado[df_filtrado[col_fecha] >= (max_fecha_real - timedelta(days=28))]
    if df_reciente.empty: df_reciente = df_filtrado.copy()
    
    perfil_dia = df_reciente.groupby([col_camp, 'Dia_Semana_Clean', 'Inter_Clean']).agg(
        total_calls=(col_calls, 'sum'),
        avg_aht=(col_aht, lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    ).reset_index()
    totales_dia = perfil_dia.groupby([col_camp, 'Dia_Semana_Clean'])['total_calls'].transform('sum')
    perfil_dia['weight'] = np.where(totales_dia > 0, perfil_dia['total_calls'] / totales_dia, 0)
    mapa_dia = {(r[col_camp], r['Dia_Semana_Clean'], r['Inter_Clean']): {'weight': r['weight'], 'aht': r['avg_aht']} for _, r in perfil_dia.iterrows()}

    perfil_global = df_reciente.groupby([col_camp, 'Inter_Clean']).agg(total_calls=(col_calls, 'sum')).reset_index()
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
        vol_historico_camp = float(df_diario[df_diario[col_camp] == camp][col_calls].mean())
        es_micro_campana = vol_historico_camp < 250

        for d in range(dias_futuros):
            fecha_actual = fecha_inicio_forecast + timedelta(days=d)
            str_fecha = fecha_actual.strftime('%Y-%m-%d')
            str_mes = f"{meses_espanol[fecha_actual.month]} {fecha_actual.year}"
            nombre_dia = dias_espanol[fecha_actual.weekday()]

            vol_diario = predicciones_futuras.get(camp, [0]*dias_futuros)[d]
            intervalos_validos = intervalos_operativos_por_camp.get(camp, [])

            pesos_crudos = []
            for inter in intervalos_validos:
                if es_micro_campana:
                    w = mapa_perfil_global.get((camp, inter), 0.0)
                else:
                    w = mapa_dia.get((camp, nombre_dia, inter), {}).get('weight', 0.0)
                    if w == 0.0: w = mapa_perfil_global.get((camp, inter), 0.0)
                pesos_crudos.append(w)

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
                
                req_ftes = calcular_agentes_requeridos_erlang_c(a_erlang_raw, aht_efectivo, target_time, target_sl) if calls_float > 0 else 0
                req_hc = math.ceil(req_ftes / factor_asistencia) if req_ftes > 0 else 0.0

                hc_roster = roster_coverage.get((str(camp), nombre_dia.capitalize(), inter), 0)
                tot_camp = roster_total_camp.get(str(camp), 0)
                tot_camp_dia = roster_total_dia_camp.get((str(camp), nombre_dia.capitalize()), 0)

                data_processed.append({
                    'Campaña': str(camp),
                    'Fecha': str_fecha,
                    'Mes': str_mes,
                    'Día_Semana': nombre_dia.capitalize(),
                    'Intervalo': inter,
                    'Llamadas': calls_int,
                    'AHT': format_aht_str(aht),
                    'AHT_Segundos': int(round(aht)),
                    'Agentes_Requeridos': req_hc,
                    'HC_Actual_Roster': hc_roster,
                    'Total_Roster_Campana': tot_camp,
                    'Total_Roster_Dia': tot_camp_dia,
                    'Factor_Correccion': 1.0
                })

    try:
        with open(CACHE_FILE_CHAT, 'w', encoding='utf-8') as f: json.dump(data_processed, f)
    except: pass
    return data_processed

@app.route('/api/latest', methods=['GET'])
def get_latest_forecast():
    mode = request.args.get('mode', 'inbound')
    if mode == 'outbound': target_cache = CACHE_FILE_OUT
    elif mode == 'chat': target_cache = CACHE_FILE_CHAT
    else: target_cache = CACHE_FILE_IN
    
    if os.path.exists(target_cache):
        try:
            with open(target_cache, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                if isinstance(cache_data, list) and len(cache_data) > 0: 
                    return jsonify(cache_data), 200
        except Exception as e:
            pass
            
    excel_path = buscar_archivo_excel()
    if excel_path:
        try:
            sl, tt, merma, dias, concurrencia = 80.0, 20.0, 30.0, 130, 3.0
            if os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                        sl = float(cfg.get('targetSl', 80.0))
                        tt = float(cfg.get('targetTime', 20.0))
                        
                        merma_data = cfg.get('merma', 30.0)
                        if isinstance(merma_data, dict):
                            merma = float(merma_data.get(mode, 30.0))
                        else:
                            merma = float(merma_data)
                except: pass
            
            merma_pct = merma / 100.0
            
            if mode == 'outbound':
                data = procesar_archivo_outbound(excel_path, merma=merma_pct, dias_futuros=dias)
            elif mode == 'chat':
                data = procesar_archivo_chat(excel_path, target_sl=sl, target_time=tt, merma=merma_pct, concurrencia=concurrencia, dias_futuros=dias)
            else:
                data = procesar_archivo_excel(excel_path, target_sl=sl, target_time=tt, merma=merma_pct, dias_futuros=dias)
                
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
        data = procesar_archivo_excel(
            excel_path, 
            float(clean_num(request.form.get('target_sl'), 80.0)), 
            float(clean_num(request.form.get('target_time'), 20.0)), 
            float(clean_num(request.form.get('merma'), 30.0)) / 100.0, 
            int(clean_num(request.form.get('dias'), 45))
        )
        gc.collect()
        return jsonify(data)
    except Exception as e:
        gc.collect()
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_outbound', methods=['POST'])
def process_outbound_data():
    excel_path = buscar_archivo_excel()
    if not excel_path: return jsonify({'error': 'No se encontro Excel (.xlsx).'}), 400
    try:
        data = procesar_archivo_outbound(
            excel_path, 
            float(clean_num(request.form.get('merma'), 30.0)) / 100.0, 
            int(clean_num(request.form.get('dias'), 45))
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
        data = procesar_archivo_chat(
            excel_path, 
            float(clean_num(request.form.get('target_sl'), 80.0)),
            float(clean_num(request.form.get('target_time'), 20.0)),
            float(clean_num(request.form.get('merma'), 30.0)) / 100.0, 
            float(clean_num(request.form.get('concurrencia'), 3.0)), 
            int(clean_num(request.form.get('dias'), 45))
        )
        gc.collect()
        return jsonify(data)
    except Exception as e:
        gc.collect()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
