import os
import io
import json
import base64
import tempfile
import warnings
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Use 'Agg' backend for non-interactive plotting
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import gradio as gr
import requests
from scipy.spatial import Voronoi, cKDTree
from scipy.cluster.hierarchy import linkage, dendrogram
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CAMINHOS E CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────────────────────
# Define o diretório base como o diretório onde o script 'app.py' está localizado.
BASE = Path(__file__).resolve().parent

# Caminhos para os arquivos de entrada (dados)
# Assumimos que os arquivos .parquet estão na mesma pasta que o app.py
INPUT_PATH = BASE / "station_geo.parquet"
CLIMATE_PATH = BASE / "station_climate.parquet"
MONTHLY_PATH = BASE / "station_climate_monthly.parquet"

# O diretório para os outputs GeoJSON será uma subpasta 'geojson_outputs'
# dentro do diretório base.
GEOJSON_DIR = BASE / "geojson_outputs"

# O logo também é esperado na mesma pasta que o app.py
LOGO_PATH = str(BASE / "logoGT.png")

# Linhas de transmissão (GeoParquet com LineString/MultiLineString em EPSG:4326)
# Assumimos que este arquivo também está na mesma pasta que o app.py
LINHAS_TRANSMISSAO_PATH = BASE / "linhas_transmissao.parquet"

# Cria o diretório para os outputs GeoJSON se não existir
GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

# Estilo visual das LTs nos mapas Matplotlib
LT_COLOR = "#FF6F00"
LT_LINEWIDTH = 0.7
LT_ALPHA = 0.75
LT_ZORDER = 8

# Paleta de 32 cores distintas para clusters
PALETTE_32 = [
    "#1976D2","#E53935","#43A047","#FB8C00","#8E24AA","#00ACC1",
    "#F4511E","#3949AB","#00897B","#FFB300","#6D4C41","#546E7A",
    "#D81B60","#00BCD4","#7CB342","#FDD835","#5E35B1","#039BE5",
    "#E64A19","#00838F","#558B2F","#F9A825","#4527A0","#01579B",
    "#BF360C","#006064","#33691E","#F57F17","#4A148C","#01579B",
    "#880E4F","#1B5E20",
]
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
CLIMATE_COLS = [
    "temp_mean","temp_max","temp_min",
    "precip_mean","precip_sum",
    "wind_mean","wind_max","gust_max",
    "rh_mean",
]
MONTHLY_VARS = CLIMATE_COLS
CLIMATE_LABELS = {
    "temp_mean" : "Mean Temp. (°C)",
    "temp_max" : "Max Temp. (°C)",
    "temp_min" : "Min Temp. (°C)",
    "precip_mean" : "Mean Precip. (mm)",
    "precip_sum" : "Total Precip. (mm)",
    "wind_mean" : "Mean Wind Speed (m/s)",
    "wind_max" : "Max Wind Speed (m/s)",
    "gust_max" : "Max Wind Gust (m/s)",
    "rh_mean" : "Mean Relative Humidity (%)",
}
CLIMATE_CMAPS = {
    "temp_mean":"RdYlBu_r", "temp_max":"hot_r", "temp_min":"Blues",
    "precip_mean":"YlGnBu", "precip_sum":"YlGnBu", "wind_mean":"PuBu",
    "wind_max":"PuRd", "gust_max":"Reds", "rh_mean":"BuGn",
}
CLIMATE_UNITS = {
    "temp_mean":"°C", "temp_max":"°C", "temp_min":"°C",
    "precip_mean":"mm","precip_sum":"mm","wind_mean":"m/s",
    "wind_max":"m/s", "gust_max":"m/s", "rh_mean":"%",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOGO
# ─────────────────────────────────────────────────────────────────────────────
try:
    logo_arr = mpimg.imread(LOGO_PATH)
    print("✔ Logo carregado:", LOGO_PATH)
except Exception as e:
    logo_arr = None
    print(f"⚠ Logo não encontrado: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAPEFILES — Brasil e estados
# ─────────────────────────────────────────────────────────────────────────────
print("\nCarregando shapefiles...")
TMP = tempfile.mkdtemp()
def download_shapefile(url: str, tmp_dir: str, shp_name: str) -> gpd.GeoDataFrame:
    """Baixa e abre um shapefile zipado de uma URL pública."""
    zip_path = os.path.join(tmp_dir, "shape.zip")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)
    return gpd.read_file(f"/vsizip/{zip_path}/{shp_name}")

gdf_world = download_shapefile(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
    TMP, "ne_110m_admin_0_countries.shp",
)
gdf_br = None
for _col in ["SOVEREIGNT","NAME","NAME_LONG","ADMIN"]:
    if _col in gdf_world.columns:
        _m = gdf_world[gdf_world[_col].str.contains("Brazil|Brasil", na=False)]
        if len(_m) > 0:
            gdf_br = _m.to_crs(epsg=4326)
            break
if gdf_br is None:
    raise RuntimeError("Polígono do Brasil não encontrado.")

try:
    gdf_states_world = download_shapefile(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_1_states_provinces.zip",
        TMP, "ne_110m_admin_1_states_provinces.shp",
    )
    gdf_states = None
    for _col in ["admin","ADMIN","sovereignt","SOVEREIGNT"]:
        if _col in gdf_states_world.columns:
            _m = gdf_states_world[
                gdf_states_world[_col].str.contains("Brazil|Brasil", na=False)
            ]
            if len(_m) > 0:
                gdf_states = _m.to_crs(epsg=4326)
                break
    has_states = gdf_states is not None and len(gdf_states) > 0
except Exception:
    has_states = False
print(f"✔ Brasil OK | Estados: {has_states}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. LINHAS DE TRANSMISSÃO
# carregamento → validação → índice espacial cKDTree
# ─────────────────────────────────────────────────────────────────────────────
gdf_lt = None # GeoDataFrame das LTs
has_lt = False # disponibilidade
_lt_tree = None # cKDTree(lon, lat) dos vértices
_lt_coords = None # np.ndarray (N, 2): [lon, lat]
print("\n── Linhas de Transmissão ─────────────────────────")
if LINHAS_TRANSMISSAO_PATH.exists():
    try:
        _raw = gpd.read_parquet(str(LINHAS_TRANSMISSAO_PATH))
        print(f" Registros brutos : {len(_raw)}")
        print(f" CRS original : {_raw.crs}")
        print(f" Tipos de geometria: {_raw.geometry.geom_type.unique().tolist()}")
        # Reprojeção para WGS84 se necessário
        if _raw.crs is None:
            _raw = _raw.set_crs(epsg=4326)
            print(" AVISO: CRS não definido → assumindo EPSG:4326")
        elif _raw.crs.to_epsg() != 4326:
            _raw = _raw.to_crs(epsg=4326)
            print(" Reprojetado → EPSG:4326")
        # Filtra pelo bounding-box do Brasil e remove inválidos
        _br_bbox = box(-75.5, -35.5, -27.5, 6.0)
        _raw = _raw[_raw.geometry.intersects(_br_bbox)].copy()
        _raw = _raw[~_raw.geometry.isna() & _raw.geometry.is_valid]
        if _raw.empty:
            print(" AVISO: nenhuma geometria válida no Brasil.")
        else:
            gdf_lt = _raw
            has_lt = True
            # ── Índice espacial cKDTree sobre todos os vértices das LTs ──────
            # Extrai coordenadas (lon, lat) de cada segmento de linha.
            # Isso permite busca O(log N) da LT mais próxima de qualquer ponto
            # — equivalente ao que geolib/Turf.js fazem internamente.
            _pts = []
            for _geom in gdf_lt.geometry:
                if _geom is None: continue
                if _geom.geom_type == "LineString":
                    _pts.extend(_geom.coords)
                elif _geom.geom_type == "MultiLineString":
                    for _p in _geom.geoms:
                        _pts.extend(_p.coords)
            if _pts:
                _lt_coords = np.array(_pts, dtype=np.float64) # (N,2) lon,lat
                _lt_tree = cKDTree(_lt_coords)
                print(f" Vértices indexados: {len(_lt_coords)}")
            print(f" Linhas válidas : {len(gdf_lt)}")
            print(" ✔ LTs prontas.")
    except Exception as _e:
        print(f" ERRO: {_e}")
else:
    print(f" AVISO: arquivo não encontrado em {LINHAS_TRANSMISSAO_PATH}")
print("─" * 50)

# ─────────────────────────────────────────────────────────────────────────────
# 6. FUNÇÕES DE DISTÂNCIA — Haversine puro (sem dependências externas)
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """
    Distância em km entre dois pontos geográficos — fórmula de Haversine.
    Parâmetros
    ----------
    lat1, lon1 : ponto de origem (graus decimais WGS84)
    lat2, lon2 : ponto de destino (graus decimais WGS84)
    Retorno
    -------
    float : distância em quilômetros
    """
    R = 6_371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1.0 - a))

def distancia_lt_km(lat: float, lon: float) -> float:
    """
    Distância em km do ponto (lat, lon) ao vértice de LT mais próximo.
    Usa o índice cKDTree para busca O(log N) inicial em espaço euclidiano
    (graus) e refina com Haversine para precisão métrica real.
    Retorna np.nan se as LTs não estiverem disponíveis.
    """
    if _lt_tree is None or _lt_coords is None:
        return np.nan
    _, idx = _lt_tree.query([lon, lat], k=1)
    lt_lon, lt_lat = _lt_coords[idx]
    return haversine_km(lat, lon, lt_lat, lt_lon)

# Variáveis globais para compartilhar dados entre callbacks do Gradio
last_df_temp = None
last_cells = None
last_centroids = None
last_actual_k = None
last_cluster_id = None
STATION_COL = None # Definido após o carregamento dos dados

def calcular_distancias_cluster(cluster_id_str: str) -> pd.DataFrame:
    """
    Calcula a distância (km) da LT mais próxima para:
    1. O centróide do cluster
    2. Cada estação individualmente
    Retorna DataFrame com colunas [Tipo, Nome, Latitude, Longitude, Dist LT (km)]
    ordenado: centróide primeiro, depois estações por distância crescente.
    """
    global last_df_temp, last_centroids
    if not cluster_id_str or last_df_temp is None:
        return pd.DataFrame()

    c_idx = int(cluster_id_str) - 1
    rows = []

    # ── Centróide ─────────────────────────────────────────────────────────────
    if last_centroids is not None and c_idx < len(last_centroids):
        c_lat = float(last_centroids[c_idx][0])
        c_lon = float(last_centroids[c_idx][1])
        d = distancia_lt_km(c_lat, c_lon)
        rows.append({
            "Tipo" : "Centroid",
            "Nome" : f"Cluster {cluster_id_str} centroid",
            "Latitude" : round(c_lat, 5),
            "Longitude" : round(c_lon, 5),
            "Dist LT (km)": round(d, 3) if not np.isnan(d) else "N/A",
            "_ord" : 0,
        })

    # ── Estações ──────────────────────────────────────────────────────────────
    subset = last_df_temp[last_df_temp["cluster_geo"] == c_idx]
    for _, row in subset.iterrows():
        s_lat = float(row["latitude"])
        s_lon = float(row["longitude"])
        d = distancia_lt_km(s_lat, s_lon)
        nome = (str(row[STATION_COL])
                if STATION_COL and STATION_COL in row.index
                else f"({s_lat:.3f}, {s_lon:.3f})")
        rows.append({
            "Tipo" : "Station",
            "Nome" : nome,
            "Latitude" : round(s_lat, 5),
            "Longitude" : round(s_lon, 5),
            "Dist LT (km)": round(d, 3) if not np.isnan(d) else "N/A",
            "_ord" : d if not np.isnan(d) else 1e9,
        })

    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows).sort_values("_ord").drop(columns="_ord")
    return df_out.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. DATA LOADING AND JOIN
# ─────────────────────────────────────────────────────────────────────────────
df_geo = pd.read_parquet(INPUT_PATH)
df_climate = pd.read_parquet(CLIMATE_PATH)

print("Colunas station_geo :", df_geo.columns.tolist())
print("Colunas station_climate:", df_climate.columns.tolist())

KEY_CANDIDATES = ["station_id","wmo_code","id","codigo","code"]
geo_key, climate_key = None, None
for _c in KEY_CANDIDATES:
    if _c in df_geo.columns and geo_key is None: geo_key = _c
    if _c in df_climate.columns and climate_key is None: climate_key = _c

if geo_key is None: raise RuntimeError("Chave de join não encontrada em station_geo.")
if climate_key is None: raise RuntimeError("Chave de join não encontrada em station_climate.")

df_geo[geo_key] = df_geo[geo_key].astype(str).str.strip()
df_climate[climate_key] = df_climate[climate_key].astype(str).str.strip()

if climate_key != geo_key:
    df_climate = df_climate.rename(columns={climate_key: geo_key})

print("Aplicando KNN Imputer...")
imputer = KNNImputer(n_neighbors=5, weights="distance")
df_climate[CLIMATE_COLS] = imputer.fit_transform(df_climate[CLIMATE_COLS])
print(f"Nulls restantes: {df_climate[CLIMATE_COLS].isna().sum().sum()}")

df_merged = df_geo.merge(df_climate, on=geo_key, how="left")
df_merged = df_merged.dropna(subset=["latitude","longitude"])
df_merged = df_merged[
    (df_merged["latitude"] >= -35.0) & (df_merged["latitude"] <= 5.5) &
    (df_merged["longitude"] >= -75.0) & (df_merged["longitude"] <= -28.0)
].copy()
df = df_merged.copy()
n_stations_total = len(df)
n_with_climate = df[CLIMATE_COLS].notna().all(axis=1).sum()
n_without_climate = n_stations_total - n_with_climate
print(f"Estações: {n_stations_total} | Com clima: {n_with_climate}")

for _c in ["station","station_name","nome","estacao","name"]:
    if _c in df.columns:
        STATION_COL = _c
        break
print("Coluna de nome:", STATION_COL)

# ─────────────────────────────────────────────────────────────────────────────
# 8. SÉRIE TEMPORAL MENSAL
# ─────────────────────────────────────────────────────────────────────────────
if MONTHLY_PATH.exists():
    df_monthly = pd.read_parquet(MONTHLY_PATH)
    df_monthly["station_id"] = df_monthly["station_id"].astype(str).str.strip()
    df_monthly["year"] = df_monthly["year"].astype(int)
    df_monthly["month"] = df_monthly["month"].astype(int)
    AVAILABLE_YEARS = sorted(df_monthly["year"].unique().tolist())
    print("Dados mensais:", df_monthly.shape, "| Anos:", AVAILABLE_YEARS)
else:
    df_monthly = None
    AVAILABLE_YEARS = []
    print("AVISO: station_climate_monthly.parquet não encontrado.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. CLUSTERING — configuração inicial
# ─────────────────────────────────────────────────────────────────────────────
GEO_FEATURES = ["latitude","longitude"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[GEO_FEATURES].values)
brasil_poly = unary_union(gdf_br.geometry)

# ─────────────────────────────────────────────────────────────────────────────
# 10. BASE PLOT HELPERS — Matplotlib
# ─────────────────────────────────────────────────────────────────────────────
def plot_brasil_base(ax) -> None:
    """Renderiza o mapa base do Brasil (polígono + divisões estaduais)."""
    gdf_br.plot(ax=ax, color="#C5E1F4", edgecolor="#335C67", linewidth=0.8, zorder=1)
    if has_states:
        gdf_states.plot(ax=ax, color="none", edgecolor="#90A4AE",
                         linewidth=0.4, zorder=2)
    ax.set_xlim(-75.5, -27.5)
    ax.set_ylim(-35.5, 6.0)
    ax.set_facecolor("#DDEEFF")
    ax.grid(True, linestyle="--", alpha=0.25)

def plot_linhas_transmissao(ax, label_legenda: bool = True):
    """Desenha LTs no eixo Matplotlib e retorna handle para legenda (ou None)."""
    if not has_lt or gdf_lt is None:
        return None
    gdf_lt.plot(ax=ax, color=LT_COLOR, linewidth=LT_LINEWIDTH,
                alpha=LT_ALPHA, zorder=LT_ZORDER)
    if label_legenda:
        return mlines.Line2D([], [], color=LT_COLOR, linewidth=1.4,
                             alpha=LT_ALPHA, label="Transmission Lines")
    return None

def voronoi_clipped(centroids_latlon: np.ndarray,
                    clip_poly, margin: float = 3.0) -> list:
    """
    Gera células de Voronoi e as recorta pelo polígono do Brasil.
    Parâmetros
    ----------
    centroids_latlon : (k, 2) — [lat, lon] dos centróides
    clip_poly : Shapely geometry do Brasil
    margin : graus de margem além do bounding-box
    """
    pts_xy = centroids_latlon[:, [1, 0]] # lon, lat (x, y)
    minx, miny, maxx, maxy = clip_poly.bounds
    bbox = box(minx-margin, miny-margin, maxx+margin, maxy+margin)
    vor = Voronoi(pts_xy)
    cells = []
    for i in range(len(pts_xy)):
        region = vor.regions[vor.point_region[i]]
        if not region:
            cells.append(None)
            continue
        poly = Polygon(vor.vertices[region]) if -1 not in region else bbox
        clipped = poly.intersection(clip_poly)
        cells.append(clipped if not clipped.is_empty else None)
    return cells

# ─────────────────────────────────────────────────────────────────────────────
# 11. MAPA INTERATIVO FOLIUM / LEAFLET ← NOVO v3
# ─────────────────────────────────────────────────────────────────────────────
def _tooltip_cluster_html(c_idx: int, df_temp) -> str:
    """HTML do tooltip de hover de uma célula Voronoi."""
    mask = df_temp["cluster_geo"] == c_idx
    subset = df_temp[mask]
    if subset.empty:
        return f"<b>Cluster {c_idx+1}</b>"
    c_lat = float(subset["latitude"].mean())
    c_lon = float(subset["longitude"].mean())
    n_st = len(subset)
    d_lt = distancia_lt_km(c_lat, c_lon)
    d_str = f"{d_lt:.1f} km" if not np.isnan(d_lt) else "N/A"
    clim_rows = "".join(
        f"<tr>"
        f"<td style='padding:1px 6px;color:#546E7A;white-space:nowrap'>{CLIMATE_LABELS[col]}</td>"
        f"<td style='padding:1px 6px;font-weight:600'>"
        f"{subset[col].dropna().mean():.2f}&nbsp;{CLIMATE_UNITS[col]}</td></tr>"
        for col in CLIMATE_COLS
        if col in subset.columns and not subset[col].dropna().empty
    )
    return f"""
    <div style='font-family:Inter,Arial,sans-serif;font-size:12px;min-width:230px'>
    <div style='background:#1565C0;color:#fff;padding:6px 10px;
    border-radius:6px 6px 0 0;font-weight:700;font-size:13px'>
    Cluster&nbsp;{c_idx+1}
    </div>
    <div style='padding:6px 10px;border:1px solid #e0e0e0;border-top:none;
    border-radius:0 0 6px 6px;background:#fff'>
    <table style='border-collapse:collapse;width:100%'>
    <tr>
    <td style='padding:1px 6px;color:#546E7A'>Estações</td>
    <td style='padding:1px 6px;font-weight:600'>{n_st}</td>
    </tr>
    <tr>
    <td style='padding:1px 6px;color:#546E7A'>Centróide</td>
    <td style='padding:1px 6px;font-weight:600'>{c_lat:.3f}°,&nbsp;{c_lon:.3f}°</td>
    </tr>
    <tr>
    <td style='padding:1px 6px;color:#FF6F00;font-weight:600'>LT mais próxima</td>
    <td style='padding:1px 6px;font-weight:700;color:#FF6F00'>{d_str}</td>
    </tr>
    <tr>
    <td colspan='2' style='padding:4px 6px 2px;color:#1565C0;font-weight:600;
    border-top:1px solid #e0e0e0;font-size:11px'>
    Médias climáticas
    </td>
    </tr>
    {clim_rows}
    </table>
    </div>
    </div>"""

def _popup_station_html(row) -> str:
    """HTML do popup de clique numa estação."""
    s_lat = float(row["latitude"])
    s_lon = float(row["longitude"])
    c_idx = int(row["cluster_geo"])
    s_name = (str(row[STATION_COL])
              if STATION_COL and STATION_COL in row.index else "—")
    d_lt = distancia_lt_km(s_lat, s_lon)
    d_str = f"{d_lt:.1f} km" if not np.isnan(d_lt) else "N/A"
    clim_rows = "".join(
        f"<tr>"
        f"<td style='color:#546E7A;padding:1px 5px;white-space:nowrap'>{CLIMATE_LABELS[col]}</td>"
        f"<td style='font-weight:600;padding:1px 5px'>"
        f"{float(row[col]):.2f}&nbsp;{CLIMATE_UNITS[col]}</td></tr>"
        for col in CLIMATE_COLS
        if col in row.index and not np.isnan(float(row.get(col, np.nan)))
    )
    return f"""
    <div style='font-family:Inter,Arial,sans-serif;font-size:12px;min-width:210px'>
    <b style='color:#1565C0;font-size:13px'>{s_name}</b><br>
    <span style='color:#546E7A'>Cluster {c_idx+1}
    &nbsp;·&nbsp;{s_lat:.4f}°,&nbsp;{s_lon:.4f}°</span>
    <hr style='margin:4px 0;border-color:#e0e0e0'>
    <span style='color:#FF6F00;font-weight:600'>LT mais próxima: {d_str}</span>
    <table style='border-collapse:collapse;width:100%;margin-top:4px'>
    {clim_rows}
    </table>
    </div>"""

def gerar_mapa_folium(show_lt: bool = True) -> str:
    """
    Gera mapa Folium (Leaflet.js) interativo e retorna HTML puro.
    Camadas togláveis via LayerControl:
    • Voronoi Cells — hover tooltip com dados do cluster e dist LT
    • Weather Stations — popup com dados climáticos por estação
    • Cluster Centroids — marcadores numerados com DivIcon
    • Transmission Lines — LTs em laranja (se disponíveis)
    """
    global last_df_temp, last_cells, last_actual_k, last_centroids
    if last_df_temp is None or last_cells is None:
        return ("<div style='padding:24px;color:#546E7A;text-align:center'>"
                "Execute o clustering para gerar o mapa interativo.</div>")
    # Mapa base CartoDB — fundo neutro para as LTs laranjas se destacarem
    m = folium.Map(location=[-15.0, -52.0], zoom_start=4,
                   tiles="CartoDB positron", attr="© CartoDB",
                   prefer_canvas=True)
    # ── Camada 1: Células de Voronoi ──────────────────────────────────────────
    # Cada célula é um Feature GeoJSON independente para highlight individual.
    vgroup = folium.FeatureGroup(name="Voronoi Cells", show=True)
    for i, cell in enumerate(last_cells):
        if cell is None or cell.is_empty: continue
        color = PALETTE_32[i % len(PALETTE_32)]
        try:
            folium.GeoJson(
                {"type":"Feature",
                 "geometry":cell.__geo_interface__,
                 "properties":{"cluster_id": i+1}},
                # Estilo padrão: semi-transparente
                style_function=lambda feat, c=color: {
                    "fillColor":c,"fillOpacity":0.28,
                    "color":"#78909C","weight":0.7,
                },
                # Estilo ao hover (mouseover): mais opaco, borda destacada
                highlight_function=lambda feat, c=color: {
                    "fillColor":c,"fillOpacity":0.60,
                    "color":"#0D47A1","weight":2.2,
                },
                tooltip=folium.Tooltip(
                    _tooltip_cluster_html(i, last_df_temp),
                    sticky=True, parse_html=True,
                ),
            ).add_to(vgroup)
        except Exception:
            continue
    vgroup.add_to(m)
    # ── Camada 2: Estações ────────────────────────────────────────────────────
    sgroup = folium.FeatureGroup(name="Weather Stations", show=True)
    for _, row in last_df_temp.iterrows():
        c_idx = int(row["cluster_geo"])
        color = PALETTE_32[c_idx % len(PALETTE_32)]
        s_name = (str(row[STATION_COL])
                  if STATION_COL and STATION_COL in row.index else "—")
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=4, color=color,
            fill=True, fill_color=color, fill_opacity=0.88, weight=0.8,
            tooltip=folium.Tooltip(s_name, sticky=False),
            popup=folium.Popup(_popup_station_html(row), max_width=290),
        ).add_to(sgroup)
    sgroup.add_to(m)
    # ── Camada 3: Centróides numerados ────────────────────────────────────────
    cgroup = folium.FeatureGroup(name="Cluster Centroids", show=True)
    if last_centroids is not None:
        for i, cent in enumerate(last_centroids):
            c_lat, c_lon = float(cent[0]), float(cent[1])
            d_lt = distancia_lt_km(c_lat, c_lon)
            d_str = f"{d_lt:.1f} km" if not np.isnan(d_lt) else "N/A"
            # DivIcon: badge circular com número do cluster
            icon_html = (
                f"<div style='background:#fff;border:2.5px solid #263238;"
                f"border-radius:50%;width:22px;height:22px;"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:9px;font-weight:700;color:#1565C0;"
                f"box-shadow:0 2px 6px rgba(0,0,0,.28)'>{i+1}</div>"
            )
            folium.Marker(
                location=[c_lat, c_lon],
                icon=folium.DivIcon(html=icon_html,
                                    icon_size=(22,22), icon_anchor=(11,11)),
                tooltip=folium.Tooltip(
                    f"<b>Centróide — Cluster {i+1}</b><br>"
                    f"LT mais próxima: <b style='color:#FF6F00'>{d_str}</b>",
                    sticky=False,
                ),
            ).add_to(cgroup)
    cgroup.add_to(m)
    # ── Camada 4: Linhas de transmissão ───────────────────────────────────────
    if show_lt and has_lt and gdf_lt is not None:
        ltgroup = folium.FeatureGroup(name="Transmission Lines", show=True)
        folium.GeoJson(
            gdf_lt.__geo_interface__,
            style_function=lambda feat: {
                "color":LT_COLOR,"weight":1.3,"opacity":LT_ALPHA
            },
            tooltip=folium.Tooltip("Linha de Transmissão", sticky=False),
        ).add_to(ltgroup)
        ltgroup.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    raw_html = m._repr_html_()
    return (
        "<div style='width:100%;height:580px;border-radius:10px;"
        "overflow:hidden;border:1px solid #CFD8DC;"
        "box-shadow:0 2px 10px rgba(0,0,0,.12)'>"
        f"{raw_html}</div>"
    )

def gerar_display_cluster_html(cluster_id_str: str) -> str:
    """
    Cluster Summary Card HTML — atualiza automaticamente ao mudar o
    cluster_selector (disparado por cluster_selector.change).
    Exibe: ID, N estações, centróide, distância LT e cards de médias climáticas.
    """
    global last_df_temp, last_centroids
    if last_df_temp is None or not cluster_id_str:
        return "<p style='color:#546E7A;padding:12px'>Selecione um cluster.</p>"
    c_idx = int(cluster_id_str) - 1
    subset = last_df_temp[last_df_temp["cluster_geo"] == c_idx]
    if subset.empty:
        return (f"<p style='color:#B71C1C;padding:12px'>"
                f"Cluster {cluster_id_str} sem dados.</p>")
    c_lat = float(subset["latitude"].mean())
    c_lon = float(subset["longitude"].mean())
    n_st = len(subset)
    d_lt = distancia_lt_km(c_lat, c_lon)
    d_str = f"{d_lt:.1f} km" if not np.isnan(d_lt) else "N/A"
    # Cards das médias climáticas
    cards = ""
    for col in CLIMATE_COLS:
        if col not in subset.columns: continue
        vals = subset[col].dropna()
        if vals.empty: continue
        mean_val = vals.mean()
        color_hex = mcolors.to_hex(cm.get_cmap(CLIMATE_CMAPS.get(col,"viridis"))(0.6))
        cards += (
            f"<div style='background:#F7F9FB;border:1px solid #E3EAF0;"
            f"border-radius:8px;padding:8px 12px;min-width:140px;flex:1 1 140px'>"
            f"<div style='font-size:10px;color:#78909C;margin-bottom:2px'>"
            f"{CLIMATE_LABELS[col]}</div>"
            f"<div style='font-size:1.15rem;font-weight:700;color:{color_hex}'>"
            f"{mean_val:.2f}&nbsp;"
            f"<span style='font-size:.8rem;font-weight:400'>{CLIMATE_UNITS[col]}</span>"
            f"</div></div>"
        )
    return (
        f"<div style='font-family:Inter,Arial,sans-serif;padding:4px 0'>"
        # Cabeçalho azul
        f"<div style='display:flex;align-items:center;gap:16px;"
        f"background:linear-gradient(135deg,#0D47A1,#1976D2 60%,#42A5F5);"
        f"border-radius:10px;padding:12px 20px;color:#fff;margin-bottom:12px'>"
        f"<div style='width:48px;height:48px;border-radius:50%;"
        f"background:rgba(255,255,255,.18);display:flex;align-items:center;"
        f"justify-content:center;font-size:1.4rem;font-weight:700;"
        f"flex-shrink:0'>{cluster_id_str}</div>"
        f"<div>"
        f"<div style='font-size:1.1rem;font-weight:700'>Cluster {cluster_id_str}</div>"
        f"<div style='font-size:.88rem;opacity:.85'>"
        f"{n_st} estações &nbsp;·&nbsp; Centróide: {c_lat:.3f}°, {c_lon:.3f}°</div>"
        f"<div style='font-size:.88rem;color:#FFD54F;font-weight:600;margin-top:2px'>"
        f"⚡ LT mais próxima: {d_str}</div>"
        f"</div></div>"
        # Cards de médias climáticas
        f"<div style='display:flex;flex-wrap:wrap;gap:8px'>{cards}</div>"
        f"</div>"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 12. CLUSTERING + VORONOI DIAGRAM (Matplotlib estático)
# ─────────────────────────────────────────────────────────────────────────────
def gerar_voronoi(n_raw, show_lt: bool = True):
    """
    Executa K-Means geográfico e gera:
    1. Diagrama de Voronoi estático (Matplotlib)
    2. Mapa interativo Leaflet (HTML Folium)
    3. Tabelas de resumo espacial e climático
    Retorno: (fig_mpl, summary_str, geo_df, clim_df, folium_html)
    """
    global last_df_temp, last_cells, last_centroids, last_actual_k
    n_per_cluster = max(1, int(n_raw))
    k_clusters = max(2, int(n_stations_total / n_per_cluster))
    km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df_temp = df.copy()
    df_temp["cluster_geo"] = labels
    last_df_temp = df_temp.copy()
    centroids_list = []
    for c in range(k_clusters):
        mask = df_temp["cluster_geo"] == c
        if mask.sum() > 0:
            centroids_list.append(df_temp.loc[mask, GEO_FEATURES].mean().values)
    centroids_geo = np.array(centroids_list)
    actual_k = len(centroids_geo)
    last_centroids = centroids_geo
    last_actual_k = actual_k
    if actual_k < 2:
        last_cells = None
        return None, "Insufficient clusters.", pd.DataFrame(), pd.DataFrame(), ""
    cells = voronoi_clipped(centroids_geo, brasil_poly)
    last_cells = cells
    n_valid = sum(c is not None for c in cells)
    # GeoJSON export
    geojson_features = []
    for i, cell in enumerate(cells):
        if cell is None or cell.is_empty: continue
        cid = i % len(PALETTE_32)
        geoms = [cell] if isinstance(cell, Polygon) else list(cell.geoms)
        for part in geoms:
            geojson_features.append({
                "geometry" : part.__geo_interface__,
                "properties": {"cluster_id": cid+1, "fill_color": PALETTE_32[cid]},
            })
    geojson_path = None
    if geojson_features:
        gdf_gj = gpd.GeoDataFrame.from_features(geojson_features, crs="EPSG:4326")
        geojson_path = GEOJSON_DIR / f"voronoi_n{n_per_cluster:02d}.geojson"
        gdf_gj.to_file(str(geojson_path), driver="GeoJSON")
    # ── Mapa Matplotlib ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 11))
    fig.patch.set_facecolor("#F7F9FB")
    plot_brasil_base(ax)
    for i, cell in enumerate(cells):
        if cell is None or cell.is_empty: continue
        color = PALETTE_32[i % len(PALETTE_32)]
        try:
            polys = [cell] if isinstance(cell, Polygon) else list(cell.geoms)
            for p in polys:
                x, y = p.exterior.xy
                ax.fill(x, y, alpha=0.30, color=color, zorder=2)
                ax.plot(x, y, color="#78909C", lw=0.4, zorder=3)
        except Exception:
            continue
    for c in range(actual_k):
        mask = df_temp["cluster_geo"] == c
        if mask.sum() > 0:
            ax.scatter(df_temp.loc[mask,"longitude"],
                       df_temp.loc[mask,"latitude"],
                       s=12, alpha=0.9,
                       color=PALETTE_32[c % len(PALETTE_32)],
                       edgecolor="white", linewidth=0.3, zorder=4)
    ax.scatter(centroids_geo[:,1], centroids_geo[:,0],
               s=70, marker="*", color="#FFFFFF",
               edgecolors="#263238", linewidths=0.7, zorder=5,
               label="Cluster centroids")
    lt_handle = plot_linhas_transmissao(ax) if show_lt else None
    handles, labels_leg = ax.get_legend_handles_labels()
    if lt_handle:
        handles.append(lt_handle)
        labels_leg.append("Transmission Lines")
    ax.legend(handles, labels_leg, loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_title(
        f"Voronoi Diagram — Geographic Clustering (Brazil)\n"
        f"K-Means | k={actual_k} | ~{n_per_cluster} stations/cluster "
        f"| n={n_stations_total}"
        + (" | + LT" if show_lt and has_lt else ""),
        fontsize=13, fontweight="bold", color="#1565C0", pad=14,
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    plt.tight_layout()
    # ── Resumo textual ────────────────────────────────────────────────────────
    sizes = df_temp["cluster_geo"].value_counts()
    lt_msg = (f" Transmission lines : {len(gdf_lt)} features, {len(_lt_coords)} vertices indexed."
              if has_lt else " Transmission lines : Not available.")
    summary_str = (
        f"Clustering Parameters:\n"
        f"  N stations/cluster (target): {n_per_cluster}\n"
        f"  K-Means clusters (actual): {actual_k}\n"
        f"  Total stations: {n_stations_total}\n"
        f"  {lt_msg}\n\n"
        f"Cluster Sizes (stations):\n"
        f"{sizes.to_string()}\n\n"
        f"GeoJSON output: {geojson_path.name if geojson_path else 'N/A'}"
    )
    # ── Tabelas de resumo ─────────────────────────────────────────────────────
    geo_summary = df_temp.groupby("cluster_geo").agg(
        n_stations=("station_id", "count"),
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
    ).reset_index()
    geo_summary["Cluster ID"] = geo_summary["cluster_geo"] + 1
    geo_summary["Centroid"] = geo_summary.apply(
        lambda r: f"{r['centroid_lat']:.3f}, {r['centroid_lon']:.3f}", axis=1
    )
    geo_summary["Stations"] = geo_summary["n_stations"]
    geo_df = geo_summary[["Cluster ID", "Stations", "Centroid"]]

    clim_summary = df_temp.groupby("cluster_geo")[CLIMATE_COLS].mean().reset_index()
    clim_summary["Cluster ID"] = clim_summary["cluster_geo"] + 1
    clim_df = clim_summary[["Cluster ID"] + CLIMATE_COLS]

    folium_html = gerar_mapa_folium(show_lt=show_lt)

    return fig, summary_str, geo_df, clim_df, folium_html

def plot_cluster_voronoi_only(cluster_id_str: str, show_lt: bool = True):
    """
    Plota o diagrama de Voronoi para um cluster específico, com estações e centróide.
    """
    global last_df_temp, last_cells, last_centroids
    if last_df_temp is None or last_cells is None or not cluster_id_str:
        return None

    c_idx = int(cluster_id_str) - 1
    subset = last_df_temp[last_df_temp["cluster_geo"] == c_idx]
    if subset.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 9))
    fig.patch.set_facecolor("#F7F9FB")
    plot_brasil_base(ax)

    # Plotar apenas a célula Voronoi do cluster selecionado
    cell = last_cells[c_idx]
    if cell is not None and not cell.is_empty:
        color = PALETTE_32[c_idx % len(PALETTE_32)]
        try:
            polys = [cell] if isinstance(cell, Polygon) else list(cell.geoms)
            for p in polys:
                x, y = p.exterior.xy
                ax.fill(x, y, alpha=0.35, color=color, zorder=2)
                ax.plot(x, y, color="#78909C", lw=0.6, zorder=3)
        except Exception:
            pass

    # Plotar estações do cluster
    ax.scatter(subset["longitude"], subset["latitude"],
               s=25, alpha=0.9,
               color=PALETTE_32[c_idx % len(PALETTE_32)],
               edgecolor="white", linewidth=0.5, zorder=4,
               label=f"Stations (Cluster {cluster_id_str})")

    # Plotar centróide do cluster
    if last_centroids is not None and c_idx < len(last_centroids):
        c_lat, c_lon = last_centroids[c_idx]
        ax.scatter(c_lon, c_lat,
                   s=100, marker="*", color="#FFFFFF",
                   edgecolors="#263238", linewidths=1.0, zorder=5,
                   label=f"Centroid (Cluster {cluster_id_str})")

    lt_handle = plot_linhas_transmissao(ax) if show_lt else None
    handles, labels_leg = ax.get_legend_handles_labels()
    if lt_handle:
        handles.append(lt_handle)
        labels_leg.append("Transmission Lines")
    ax.legend(handles, labels_leg, loc="lower right", fontsize=9, framealpha=0.85)

    ax.set_title(
        f"Cluster {cluster_id_str} Detail — Voronoi Region\n"
        f"({len(subset)} stations)",
        fontsize=13, fontweight="bold", color="#1565C0", pad=14,
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    plt.tight_layout()
    return fig

def cluster_station_details(cluster_id_str: str):
    """
    Retorna detalhes das estações de um cluster, tensor e informações.
    """
    global last_df_temp, STATION_COL
    if last_df_temp is None or not cluster_id_str:
        return pd.DataFrame(), "", "", []

    c_idx = int(cluster_id_str) - 1
    subset = last_df_temp[last_df_temp["cluster_geo"] == c_idx].copy()

    if subset.empty:
        return pd.DataFrame(), "", "", []

    # Tabela de estações
    cols_to_show = ["latitude", "longitude"] + CLIMATE_COLS
    if STATION_COL:
        cols_to_show.insert(0, STATION_COL)
    df_out = subset[cols_to_show].round(2)

    # Tensor (apenas lat/lon)
    tensor_data = subset[GEO_FEATURES].values
    tensor_str = str(tensor_data)
    tensor_info = (f"Shape: {tensor_data.shape}\n"
                   f"Dtype: {tensor_data.dtype}")

    # Choices para multiselect de estações
    choices = []
    if STATION_COL:
        choices = subset[STATION_COL].astype(str).tolist()

    return df_out, tensor_str, tensor_info, choices

def climate_summary_for_cluster(cluster_id_str: str):
    """
    Calcula o resumo climático (média, min, max, N) para um cluster.
    """
    global last_df_temp
    if last_df_temp is None or not cluster_id_str:
        return pd.DataFrame()

    c_idx = int(cluster_id_str) - 1
    subset = last_df_temp[last_df_temp["cluster_geo"] == c_idx]

    if subset.empty:
        return pd.DataFrame()

    summary_data = []
    for col in CLIMATE_COLS:
        vals = subset[col].dropna()
        if not vals.empty:
            summary_data.append({
                "Variable": CLIMATE_LABELS[col],
                "Mean": vals.mean(),
                "Min": vals.min(),
                "Max": vals.max(),
                "N": len(vals),
            })
    df_summary = pd.DataFrame(summary_data).round(2)
    return df_summary

# ─────────────────────────────────────────────────────────────────────────────
# 13. HEATMAP CLIMÁTICO
# ─────────────────────────────────────────────────────────────────────────────
def gerar_heatmap_clima(climate_var: str, show_lt: bool = True):
    """
    Gera um heatmap das médias climáticas por célula Voronoi.
    """
    global last_df_temp, last_cells, last_actual_k
    if last_df_temp is None or last_cells is None or last_actual_k is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 11))
    fig.patch.set_facecolor("#F7F9FB")
    plot_brasil_base(ax)

    # Calcular média da variável climática por cluster
    cluster_means = last_df_temp.groupby("cluster_geo")[climate_var].mean()

    # Normalizar valores para o colormap
    if not cluster_means.empty:
        norm = mcolors.Normalize(vmin=cluster_means.min(), vmax=cluster_means.max())
        cmap = cm.get_cmap(CLIMATE_CMAPS.get(climate_var, "viridis"))
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap("viridis")

    for i, cell in enumerate(last_cells):
        if cell is None or cell.is_empty: continue
        if i in cluster_means.index:
            mean_val = cluster_means.loc[i]
            color = cmap(norm(mean_val))
        else:
            color = "lightgray" # Cor para clusters sem dados climáticos

        try:
            polys = [cell] if isinstance(cell, Polygon) else list(cell.geoms)
            for p in polys:
                x, y = p.exterior.xy
                ax.fill(x, y, alpha=0.7, color=color, zorder=2)
                ax.plot(x, y, color="#78909C", lw=0.4, zorder=3)
        except Exception:
            pass

    # Adicionar estações como pontos brancos
    ax.scatter(last_df_temp["longitude"], last_df_temp["latitude"],
               s=8, alpha=0.8, color="white", edgecolor="gray", linewidth=0.2, zorder=4)

    # Adicionar linhas de transmissão
    plot_linhas_transmissao(ax, label_legenda=False) # Não adicionar à legenda do heatmap

    # Adicionar barra de cores
    if not cluster_means.empty:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
        cbar.set_label(f"{CLIMATE_LABELS[climate_var]}", fontsize=10)

    ax.set_title(
        f"Climate Heatmap — {CLIMATE_LABELS[climate_var]}\n"
        f"Mean per Voronoi Cluster (k={last_actual_k})",
        fontsize=13, fontweight="bold", color="#1565C0", pad=14,
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 14. BOXPLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _style_bp(bp, n_boxes: int, cmap_obj, unit: str):
    """Aplica estilo aos boxplots."""
    for i, patch in enumerate(bp["boxes"]):
        color = cmap_obj(i / (n_boxes - 1) if n_boxes > 1 else 0.5)
        patch.set_facecolor(color)
        patch.set_edgecolor("#1A237E")
        patch.set_linewidth(1.2)
    for median in bp["medians"]:
        median.set_color("#1A237E")
        median.set_linewidth(2.2)

def _annotate_medians(ax, data_list, x_positions, unit: str, suppress: bool = False):
    """Anota medianas nos boxplots."""
    if suppress: return
    for i, data in enumerate(data_list):
        if len(data) > 0:
            median = np.median(data)
            ax.text(x_positions[i], median, f"{median:.1f}{unit}",
                    verticalalignment="bottom", horizontalalignment="center",
                    fontsize=7, color="#1A237E", fontweight="bold")

def _reference_line(ax, data_list, unit: str, loc: str = "best"):
    """Adiciona linha de referência (média geral) ao boxplot."""
    all_vals = [v for v in data_list if len(v) > 0]
    if all_vals:
        gm = float(np.mean(np.concatenate(all_vals)))
        ax.axhline(gm, color="#E53935", linestyle=":", linewidth=1.3,
                   label=f"Overall mean: {gm:.1f} {unit}")
        ax.legend(fontsize=8.5, framealpha=0.8, loc=loc)

def _build_stats_table(df_data, climate_var: str, group_col: str,
                       group_labels: dict, unit: str) -> pd.DataFrame:
    """
    Constrói uma tabela de estatísticas descritivas para boxplots.
    """
    stats = df_data.groupby(group_col)[climate_var].agg(
        Mean="mean", Median="median", Std="std", Min="min", Max="max", N="count"
    ).reset_index()
    stats[group_col] = stats[group_col].map(group_labels)
    stats = stats.rename(columns={group_col: "Group"})
    stats["Unit"] = unit
    return stats.round(2)

def _get_cluster_station_ids(cluster_id_str: str) -> list:
    """Retorna IDs das estações em um cluster."""
    global last_df_temp
    if last_df_temp is None or not cluster_id_str:
        return []
    c_idx = int(cluster_id_str) - 1
    subset = last_df_temp[last_df_temp["cluster_geo"] == c_idx]
    return subset["station_id"].astype(str).tolist()

def _build_boxplot_base(data_list: list, tick_labels: list,
                        climate_var: str, title: str, compact: bool = False):
    """
    Função base para construir boxplots Matplotlib.
    """
    label = CLIMATE_LABELS.get(climate_var, climate_var)
    unit = CLIMATE_UNITS.get(climate_var, "")
    cmap_obj = cm.get_cmap(CLIMATE_CMAPS.get(climate_var,"viridis"),
                            max(len(data_list), 2))
    n_boxes = len(data_list)
    fig_w = min(max(8, n_boxes * (0.9 if compact else 1.6)), 28)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    fig.patch.set_facecolor("#F7F9FB")
    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        widths=0.38 if compact else 0.55,
        showfliers=True,
        medianprops=dict(color="#1A237E", linewidth=2.2),
        whiskerprops=dict(linestyle="--", color="#546E7A", linewidth=0.9),
        capprops=dict(color="#546E7A", linewidth=1.0),
        flierprops=dict(marker="o", markersize=3.5, alpha=0.45,
                        markerfacecolor="#455A64", linestyle="none"),
    )
    _style_bp(bp, n_boxes, cmap_obj, unit)
    # Anotações de mediana: suprime se compact e muitas caixas
    _annotate_medians(ax, data_list, range(1, n_boxes+1), unit,
                      suppress=(compact and n_boxes > 20))
    # Separadores verticais leves no modo compacto — ajudam a identificar cada caixa
    if compact and n_boxes > 1:
        for pos in range(1, n_boxes+1):
            ax.axvline(pos+0.5, color="#E0E0E0", linewidth=0.5, zorder=0)
    all_vals = [v for v in data_list if len(v) > 0]
    if all_vals:
        gm = float(np.mean(np.concatenate(all_vals)))
        ax.axhline(gm, color="#E53935", linestyle=":", linewidth=1.3,
                   label=f"Overall mean: {gm:.1f} {unit}")
        ax.legend(fontsize=8.5, framealpha=0.8, loc="upper right")
    ax.set_xticks(range(1, n_boxes+1))
    fs = max(5, min(9, int(110 / max(n_boxes, 1))))
    ax.set_xticklabels(tick_labels, fontsize=fs, rotation=35, ha="right")
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax.set_facecolor("#F7F9FB")
    ax.grid(axis="y", linestyle="--", alpha=0.38)
    plt.tight_layout()
    return fig

def _resolve_vals(row_df, climate_var: str, station_id: str = None) -> np.ndarray:
    """
    Resolve valores de uma estação para o boxplot.
    Prioriza dados mensais (distribuição mais rica) sobre o dado agregado.
    """
    vals = []
    if df_monthly is not None and station_id:
        sub_m = df_monthly[df_monthly["station_id"] == station_id]
        if climate_var in sub_m.columns:
            vals = sub_m[climate_var].dropna().values.tolist()
    if not vals and climate_var in row_df.columns: # Fallback para dados agregados se não houver mensais
        vals = row_df[climate_var].dropna().values.tolist()
    return np.array(vals)

def gerar_boxplot_agregado(cluster_id_str: str, climate_var: str,
                           mode: str, selected_stations: list):
    """
    Boxplot agregado para o cluster selecionado.
    Modos
    -----
    cluster — 1 caixa: todos os dados do cluster
    all_side_by_side — todas as estações lado a lado, ordenadas por mediana ← NOVO v3
    compare — caixas por estação selecionada (≥2)
    single — 1 caixa para a estação selecionada
    """
    global last_df_temp, STATION_COL, df_monthly
    if last_df_temp is None or not cluster_id_str: return None
    c_idx = int(cluster_id_str) - 1
    subset = last_df_temp[last_df_temp["cluster_geo"] == c_idx]
    if subset.empty or climate_var not in subset.columns: return None
    label = CLIMATE_LABELS.get(climate_var, climate_var)

    # ── CLUSTER ───────────────────────────────────────────────────────────────
    if mode == "cluster":
        vals = subset[climate_var].dropna().values
        if not len(vals): return None
        return _build_boxplot_base(
            [vals], [f"Cluster {cluster_id_str}\n({len(vals)} obs)"],
            climate_var, f"Cluster Distribution — {label}",
        )
    # ── ALL STATIONS SIDE-BY-SIDE ← NOVO v3 ────────────────────────────────
    # Todas as estações do cluster num único painel, justapostas e ordenadas
    # por mediana crescente para facilitar a comparação visual direta.
    if mode == "all_side_by_side":
        data_list, labels_list, medians = [], [], []
        for _, row in subset.iterrows():
            sid = str(row.get("station_id","")) if "station_id" in row.index else None
            name = (str(row[STATION_COL]) if STATION_COL and STATION_COL in row.index
                    else f"({row['latitude']:.2f},{row['longitude']:.2f})")
            vals = _resolve_vals(pd.DataFrame([row]), climate_var, sid)
            if len(vals) > 0:
                data_list.append(vals)
                labels_list.append(name)
                medians.append(float(np.median(vals)))
        if not data_list: return None
        order = np.argsort(medians)
        data_list = [data_list[i] for i in order]
        labels_list = [labels_list[i] for i in order]
        return _build_boxplot_base(
            data_list, labels_list, climate_var,
            f"All Stations Side-by-Side — {label}\n"
            f"Cluster {cluster_id_str} · {len(data_list)} stations (sorted by median)",
            compact=True,
        )
    # ── SINGLE ────────────────────────────────────────────────────────────────
    if mode == "single":
        if not selected_stations: return None
        sname = selected_stations[0]
        row_match = (subset[subset[STATION_COL].astype(str)==sname]
                     if STATION_COL and STATION_COL in subset.columns
                     else pd.DataFrame())
        if row_match.empty: return None
        row = row_match.iloc[0]
        sid = str(row["station_id"]) if "station_id" in row.index else None
        vals = _resolve_vals(pd.DataFrame([row]), climate_var, sid)
        if not len(vals): return None
        return _build_boxplot_base(
            [vals], [sname], climate_var,
            f"Single Station — {sname}\n{label}",
        )
    # ── COMPARE ───────────────────────────────────────────────────────────────
    if mode == "compare":
        if not selected_stations or len(selected_stations) < 2:
            vals = subset[climate_var].dropna().values
            return _build_boxplot_base(
                [vals], [f"Cluster {cluster_id_str}"], climate_var,
                f"Cluster Distribution — {label} (select ≥2 stations for comparison)",
            )
        data_list, labels_list = [], []
        for sname in selected_stations:
            row_match = (subset[subset[STATION_COL].astype(str)==sname]
                         if STATION_COL and STATION_COL in subset.columns
                         else pd.DataFrame())
            if row_match.empty: continue
            row = row_match.iloc[0]
            sid = str(row["station_id"]) if "station_id" in row.index else None
            vals = _resolve_vals(pd.DataFrame([row]), climate_var, sid)
            if len(vals) > 0:
                data_list.append(vals)
                labels_list.append(sname)
        if not data_list: return None
        return _build_boxplot_base(
            data_list, labels_list, climate_var,
            f"Station Comparison — {label}",
        )
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 17. BOXPLOT MENSAL (sub-modo "Monthly Time-Series")
# ─────────────────────────────────────────────────────────────────────────────
def gerar_boxplot_mensal(cluster_id_str: str, climate_var: str,
                         selected_years: list, selected_months: list,
                         selected_stations_ts: list, ts_mode: str):
    """
    Boxplots mensais de série temporal.
    Modos (ts_mode)
    ---------------
    all_months — painel único, X=mês, todas as estações do cluster
    by_year — um painel por ano (grid 3 colunas)
    by_station — um painel por estação (grid 3 colunas)
    Retorno: (fig, stats_df)
    """
    global df_monthly, last_df_temp, STATION_COL
    if df_monthly is None or not cluster_id_str: return None, pd.DataFrame()
    station_ids = _get_cluster_station_ids(cluster_id_str)
    if not station_ids: return None, pd.DataFrame()
    sub = df_monthly[df_monthly["station_id"].isin(station_ids)].copy()
    if sub.empty: return None, pd.DataFrame()
    if selected_years:
        sub = sub[sub["year"].isin([int(y) for y in selected_years])]
    if selected_months:
        sub = sub[sub["month"].isin([int(m) for m in selected_months])]
    if sub.empty or climate_var not in sub.columns: return None, pd.DataFrame()
    label = CLIMATE_LABELS.get(climate_var, climate_var)
    unit = CLIMATE_UNITS.get(climate_var, "")
    years_used = sorted(sub["year"].unique().tolist())
    months_used = sorted(sub["month"].unique().tolist())
    n_years = len(years_used)
    plt.close("all")

    def _make_bp(ax_, sub_data, months_f, data_f):
        """Helper interno: boxplot mensal num eixo, com estilo padronizado."""
        n_m = len(months_f)
        cmap_ = cm.get_cmap(CLIMATE_CMAPS.get(climate_var,"viridis"), max(n_m, 2))
        bp_ = ax_.boxplot(
            list(data_f), patch_artist=True, widths=0.58, showfliers=True,
            positions=list(range(1, n_m+1)),
            medianprops=dict(color="#1A237E", linewidth=2.2),
            whiskerprops=dict(linestyle="--", color="#546E7A", linewidth=0.9),
            capprops=dict(color="#546E7A", linewidth=1.0),
            flierprops=dict(marker="o", markersize=3, alpha=0.4,
                            markerfacecolor="#455A64", linestyle="none"),
        )
        _style_bp(bp_, n_m, cmap_, unit)
        _annotate_medians(ax_, list(data_f), range(1, n_m+1), unit)
        _reference_line(ax_, list(data_f), unit, loc="best")
        ax_.set_xticks(range(1, n_m+1))
        ax_.set_xticklabels([MONTH_LABELS[m-1] for m in months_f], fontsize=9)
        ax_.set_facecolor("#F7F9FB")
        ax_.grid(axis="y", linestyle="--", alpha=0.35)

    # ── ALL_MONTHS ────────────────────────────────────────────────────────────
    if ts_mode == "all_months":
        valid = [(m, sub.loc[sub["month"]==m, climate_var].dropna().values)
                 for m in months_used
                 if len(sub.loc[sub["month"]==m, climate_var].dropna()) > 0]
        if not valid: return None, pd.DataFrame()
        months_f, data_f = zip(*valid)
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#F7F9FB")
        _make_bp(ax, sub, months_f, data_f)
        ax.set_ylabel(label, fontsize=11)
        n_obs = sum(len(d) for d in data_f)
        ax.set_title(
            f"Monthly Climatological Boxplot — {label}\n"
            f"Cluster {cluster_id_str} · {len(station_ids)} stations · "
            f"{n_years} year(s) · {n_obs} observations",
            fontsize=13, fontweight="bold", color="#0D47A1", pad=12,
        )
        plt.tight_layout()
        stats = _build_stats_table(
            sub, climate_var, "month",
            {m: MONTH_LABELS[m-1] for m in range(1,13)}, unit,
        )
        return fig, stats
    # ── BY_YEAR ───────────────────────────────────────────────────────────────
    if ts_mode == "by_year":
        ncols = min(n_years, 3)
        nrows = int(np.ceil(n_years / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6.5*ncols, 5.5*nrows),
                                 sharey=True, squeeze=False)
        fig.patch.set_facecolor("#F7F9FB")
        for yi, year in enumerate(years_used):
            ax_ = axes[yi//ncols][yi%ncols]
            sub_y = sub[sub["year"]==year]
            valid = [(m, sub_y.loc[sub_y["month"]==m, climate_var].dropna().values)
                     for m in months_used
                     if len(sub_y.loc[sub_y["month"]==m, climate_var].dropna()) > 0]
            if not valid:
                ax_.set_visible(False); continue
            months_f, data_f = zip(*valid)
            _make_bp(ax_, sub_y, months_f, data_f)
            ax_.set_title(f"{year} · {sub_y['station_id'].nunique()} stations",
                          fontsize=11, fontweight="bold", color="#1565C0")
            ax_.set_ylabel(label if yi%ncols==0 else "", fontsize=9)
        for idx in range(n_years, nrows*ncols):
            axes[idx//ncols][idx%ncols].set_visible(False)
        fig.suptitle(
            f"Monthly Boxplot by Year — {label}\n"
            f"Cluster {cluster_id_str} · {len(station_ids)} stations",
            fontsize=14, fontweight="bold", color="#0D47A1", y=1.01,
        )
        plt.tight_layout()
        stats = _build_stats_table(sub, climate_var, "year",
                                   {y: str(y) for y in years_used}, unit)
        return fig, stats
    # ── BY_STATION ────────────────────────────────────────────────────────────
    if ts_mode == "by_station":
        if not selected_stations_ts:
            return gerar_boxplot_mensal(
                cluster_id_str, climate_var,
                selected_years, selected_months, [], "all_months",
            )
        name_to_id = {}
        if last_df_temp is not None and STATION_COL in last_df_temp.columns:
            c_idx_ = int(cluster_id_str) - 1
            cl_sub = last_df_temp[last_df_temp["cluster_geo"]==c_idx_]
            name_to_id = {
                str(r.get(STATION_COL,"")): str(r.get("station_id",""))
                for _, r in cl_sub.iterrows()
            }
        pairs = [(s, name_to_id.get(s,s))
                 for s in selected_stations_ts
                 if name_to_id.get(s,s) in station_ids]
        if not pairs: return None, pd.DataFrame()
        n_st = len(pairs)
        ncols = min(n_st, 3)
        nrows = int(np.ceil(n_st/ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6.5*ncols, 5.5*nrows),
                                 sharey=True, squeeze=False)
        fig.patch.set_facecolor("#F7F9FB")
        all_stats = []
        for si, (sname, sid) in enumerate(pairs):
            ax_ = axes[si//ncols][si%ncols]
            sub_s = sub[sub["station_id"]==sid]
            valid = [(m, sub_s.loc[sub_s["month"]==m, climate_var].dropna().values)
                     for m in months_used
                     if len(sub_s.loc[sub_s["month"]==m, climate_var].dropna()) > 0]
            if not valid:
                ax_.set_visible(False); continue
            months_f, data_f = zip(*valid)
            _make_bp(ax_, sub_s, months_f, data_f)
            ax_.set_title(sname, fontsize=10, fontweight="bold", color="#1565C0")
            ax_.set_ylabel(label if si%ncols==0 else "", fontsize=9)
            sf = _build_stats_table(
                sub_s, climate_var, "month",
                {m: MONTH_LABELS[m-1] for m in range(1,13)}, unit,
            )
            sf.insert(0, "Station", sname)
            all_stats.append(sf)
        for idx in range(n_st, nrows*ncols):
            axes[idx//ncols][idx%ncols].set_visible(False)
        fig.suptitle(
            f"Station{'s' if n_st>1 else ''} Monthly Boxplot — {label}\n"
            f"Cluster {cluster_id_str} · {n_st} station(s)",
            fontsize=13, fontweight="bold", color="#0D47A1", y=1.01,
        )
        plt.tight_layout()
        stats = (pd.concat(all_stats, ignore_index=True)
                 if all_stats else pd.DataFrame())
        return fig, stats
    return None, pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 18. DISPATCHER UNIFICADO ← NOVO v3
# ─────────────────────────────────────────────────────────────────────────────
def gerar_analise_unificada(
    cluster_id_str: str,
    climate_var: str,
    analysis_type: str, # "aggregate" | "monthly"
    aggregate_mode: str,
    selected_stations: list,
    monthly_mode: str,
    selected_years: list,
    selected_months: list,
    selected_stations_ts: list,
):
    """
    Dispatcher da seção unificada de análise climática (v3).
    Roteia para gerar_boxplot_agregado ou gerar_boxplot_mensal e retorna
    (fig, stats_df).
    """
    if analysis_type == "aggregate":
        fig = gerar_boxplot_agregado(
            cluster_id_str, climate_var, aggregate_mode, selected_stations
        )
        return fig, pd.DataFrame()
    if analysis_type == "monthly":
        months_int = [int(m) for m in selected_months] if selected_months else []
        return gerar_boxplot_mensal(
            cluster_id_str, climate_var,
            selected_years, months_int,
            selected_stations_ts, monthly_mode,
        )
    return None, pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 19. DENDROGRAM
# ─────────────────────────────────────────────────────────────────────────────
def gerar_dendrograma(max_samples: int = 200, method: str = "ward"):
    """Dendrograma de clustering hierárquico sobre amostra aleatória."""
    plt.close("all")
    X = X_scaled.copy()
    if X.shape[0] > max_samples:
        X = X[np.random.choice(X.shape[0], size=int(max_samples), replace=False)]
    Z = linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#F7F9FB")
    dendrogram(Z, ax=ax, leaf_rotation=90.0, leaf_font_size=8.0)
    ax.set_title(
        f"Hierarchical Clustering Dendrogram ({method})\n"
        f"Sample: {len(X)} stations",
        fontsize=12, fontweight="bold", color="#1565C0",
    )
    ax.set_xlabel("Stations (sample)", fontsize=9)
    ax.set_ylabel("Distance (scaled)", fontsize=9)
    ax.set_facecolor("#F7F9FB")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 20. CSS + CABEÇALHO
# ─────────────────────────────────────────────────────────────────────────────
CLIMATE_CHOICES = [(CLIMATE_LABELS[c], c) for c in CLIMATE_COLS]
YEAR_CHOICES = [str(y) for y in AVAILABLE_YEARS]
MONTH_CHOICES = [(MONTH_LABELS[i], str(i+1)) for i in range(12)]

CSS = """
* { font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important; }
.repotex-header {
display:flex;align-items:center;justify-content:center;gap:24px;
padding:18px 24px;
background:linear-gradient(135deg,#0D47A1 0%,#1976D2 60%,#42A5F5 100%);
border-radius:12px;margin-bottom:8px;
box-shadow:0 4px 16px rgba(13,71,161,.18);flex-wrap:wrap;
}
.repotex-logo { width:220px;height:auto;border-radius:8px;flex-shrink:0;object-fit:contain; }
.repotex-title-block { text-align:center;flex:1 1 200px; }
.repotex-title { margin:0;font-size:clamp(1.3rem,2.5vw,2rem);font-weight:700;
color:#FFF;letter-spacing:.5px;line-height:1.2; }
.repotex-subtitle { margin:6px 0 0;font-size:clamp(.78rem,1.4vw,1rem);
color:#BBDEFB;font-weight:400; }
.section-heading { font-size:1.05rem;font-weight:600;color:#1565C0;
border-left:4px solid #1976D2;padding-left:10px;
margin:20px 0 6px; }
.sub-heading { font-size:.95rem;font-weight:600;color:#37474F;
border-left:3px solid #90A4AE;padding-left:8px;
margin:12px 0 4px; }
.gradio-container .gr-button-primary {
background:#1976D2 !important;border-radius:8px !important;
font-weight:600 !important;transition:background .2s !important; }
.gradio-container .gr-button-primary:hover { background:#1565C0 !important; }
.gradio-container .gr-button-secondary { border-radius:8px !important; }
@media (max-width:768px) {
.repotex-header { flex-direction:column;text-align:center; }
.repotex-logo { width:100px; }
}
.gradio-container button[aria-label="Download"],
.gradio-container button[aria-label="Fullscreen"],
.gradio-container button[aria-label="Share"],
.media-controls,.toolbar { display:none !important; }
"""
_HEADER_TPL = """
<div class="repotex-header">
<img src="data:image/png;base64,{logo_b64}" class="repotex-logo" alt="Gridtech Logo"/>
<div class="repotex-title-block">
<h1 class="repotex-title">RePoTEx</h1>
<p class="repotex-subtitle">
Voronoi Geographic Clustering & Climate Analysis — INMET Weather Stations
</p>
</div>
</div>
"""
if logo_arr is not None:
    _buf = io.BytesIO()
    plt.imsave(_buf, logo_arr, format="png")
    _logo_b64 = base64.b64encode(_buf.getvalue()).decode()
else:
    _logo_b64 = ""
HEADER_HTML = _HEADER_TPL.format(logo_b64=_logo_b64)

# ─────────────────────────────────────────────────────────────────────────────
# 21. GRADIO INTERFACE — v3
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Gridtech — RePoTEx Dashboard v3",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css=CSS,
) as app:
    gr.HTML(HEADER_HTML)
    # Status das LTs
    _n_v = len(_lt_coords) if _lt_coords is not None else 0
    gr.HTML(
        f'<p style="color:#2E7D32;font-weight:600;margin:4px 0 8px">'
        f"✔ Transmission lines loaded — {len(gdf_lt) if has_lt else 0} features "
        f"· {_n_v} vertices indexed</p>"
        if has_lt else
        '<p style="color:#B71C1C;font-weight:600;margin:4px 0 8px">'
        "⚠ Transmission lines not available — file not found or invalid</p>"
    )
    # ════════════════════════════════════════════════════════════════════════
    # ① Clustering Parameters
    # ════════════════════════════════════════════════════════════════════════
    gr.HTML('<p class="section-heading">① Clustering Parameters</p>')
    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            slider_n = gr.Slider(
                minimum=1, maximum=20, step=1, value=5,
                label="N — Stations per Cluster",
                info="Número aproximado de estações por cluster (1–20)",
            )
            chk_lt_voronoi = gr.Checkbox(
                value=True, label="Show Transmission Lines",
                interactive=has_lt,
                info="" if has_lt else "Requer linhas_transmissao.parquet",
            )
            btn_run = gr.Button("Generate Voronoi Diagram",
                                variant="primary", size="lg")
        gr.HTML('<p class="section-heading">Clustering Summary</p>')
        summary_box = gr.Textbox(label="Summary", lines=18, interactive=False)
        with gr.Column(scale=2):
            voronoi_plot = gr.Plot(
                label="Voronoi Diagram — Clusters, Stations and Centroids"
            )
    # ════════════════════════════════════════════════════════════════════════
    # ② Interactive Map — Leaflet/Folium ← NOVO v3
    # ════════════════════════════════════════════════════════════════════════
    gr.HTML(
        '<p class="section-heading">'
        '② Interactive Map — Hover Tooltips & Transmission Line Distances'
        '</p>'
        '<p style="font-size:.88rem;color:#546E7A;margin:0 0 8px">'
        'How to use the interactive map '
        '• Hover over a Voronoi cell to see the cluster ID, number of stations, climate averages, and distance (km) to the nearest transmission line.'
        '• Click a station to view its individual climate data and distance to the nearest transmission line.'
        '• Use the layer control in the upper-right corner to toggle Voronoi cells, stations, centroids, and transmission lines.'
        '</p>'
    )
    folium_map_html = gr.HTML(
        value=(
            "<div style='padding:24px;color:#546E7A;text-align:center'>"
            "Execute o clustering para gerar o mapa interativo.</div>"
        ),
    )
    # ════════════════════════════════════════════════════════════════════════
    # ③ Cluster Summaries
    # ════════════════════════════════════════════════════════════════════════
    gr.HTML('<p class="section-heading">③ Cluster Summaries</p>')
    geo_table = gr.Dataframe(
        label="Cluster Spatial Summary — ID | N Stations | Centroid | Stations",
        interactive=False, wrap=True,
    )
    clim_table = gr.Dataframe(
        label="Climate Averages per Cluster",
        interactive=False, wrap=False,
    )
    # ════════════════════════════════════════════════════════════════════════
    # ④ Climate Heatmap
    # ════════════════════════════════════════════════════════════════════════
    gr.HTML('<p class="section-heading">④ Climate Heatmap — Brazil Map</p>')
    with gr.Row():
        with gr.Column(scale=2):
            heatmap_var = gr.Radio(
                choices=CLIMATE_CHOICES, value="temp_mean",
                label="Climate Variable for Heatmap",
            )
        with gr.Column(scale=1):
            chk_lt_heatmap = gr.Checkbox(
                value=True, label="Show Transmission Lines",
                interactive=has_lt,
            )
            btn_heatmap = gr.Button("Generate Heatmap", variant="secondary")
    heatmap_plot = gr.Plot(
        label="Climate Heatmap — Voronoi Regions Colored by Cluster Mean"
    )
    # ════════════════════════════════════════════════════════════════════════
    # ⑤ Cluster Station Detail
    # ════════════════════════════════════════════════════════════════════════
    gr.HTML('<p class="section-heading">⑤ Cluster Station Detail</p>')
    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            cluster_selector = gr.Dropdown(
                label="Select Cluster ID", choices=[], value=None,
                info="Execute o clustering primeiro.",
            )
            chk_lt_detail = gr.Checkbox(
                value=True, label="Show Transmission Lines (Detail Map)",
                interactive=has_lt,
            )
            btn_detail = gr.Button("Show Cluster Detail", variant="secondary")
        with gr.Column(scale=2):
            cluster_map = gr.Plot(
                label="Selected Cluster — Voronoi Region and Stations"
            )
    # Cluster Summary Card — atualiza ao mudar cluster_selector ← NOVO v3
    gr.HTML('<p class="sub-heading">Cluster Summary Card</p>')
    cluster_display_html = gr.HTML(
        value="<p style='color:#546E7A;padding:12px'>Selecione um cluster.</p>",
    )
    with gr.Row():
        with gr.Column(scale=2):
            stations_table = gr.Dataframe(
                label="Stations in Selected Cluster", interactive=False
            )
        with gr.Column(scale=1):
            tensor_box = gr.Textbox(label="Lat/Lon Tensor (NumPy)",
                                    lines=6, interactive=False)
            tensor_info = gr.Textbox(label="Tensor Info", interactive=False)
    climate_detail_table = gr.Dataframe(
        label="Climate Summary — Selected Cluster (Mean | Min | Max | N)",
        interactive=False, wrap=False,
    )
    # Tabela de distâncias LT ← NOVO v3
    gr.HTML('<p class="sub-heading">Distance to Nearest Transmission Line (km)</p>')
    lt_distance_table = gr.Dataframe(
        label="LT Distance — Centroid & All Stations",
        interactive=False, wrap=False,
    )
    # ════════════════════════════════════════════════════════════════════════
    # ⑥ Climate Analysis — Boxplot & Time-Series ← NOVO v3 (seção unificada)
    # ════════════════════════════════════════════════════════════════════════
    gr.HTML(
        '<p class="section-heading">'
        '⑥ Climate Analysis — Boxplot & Time-Series'
        '</p>'
        '<p style="font-size:.88rem;color:#546E7A;margin:0 0 6px">'
        'Selecione <b>Aggregate</b> para boxplots estáticos por estação, '
        'ou <b>Monthly Time-Series</b> para análise temporal. '
        'Todas as variáveis climáticas estão disponíveis em ambos os modos.'
        '</p>'
    )
    analysis_type = gr.Radio(
        choices=[
            ("Aggregate (station boxplots)", "aggregate"),
            ("Monthly Time-Series", "monthly"),
        ],
        value="aggregate",
        label="Analysis Type",
    )
    # Variável climática — compartilhada entre os dois sub-modos
    unified_climate_var = gr.Radio(
        choices=CLIMATE_CHOICES,
        value="temp_mean",
        label="Climate Variable",
    )
    # ── Sub-painel Aggregate ──────────────────────────────────────────────────
    gr.HTML('<p class="sub-heading">Aggregate Options</p>')
    aggregate_mode = gr.Radio(
        choices=[
            ("Cluster Distribution", "cluster"),
            ("All Stations Side-by-Side", "all_side_by_side"), # ← NOVO v3
            ("Station Comparison (≥2)", "compare"),
            ("Single Station", "single"),
        ],
        value="cluster",
        label="Aggregate Mode",
        info=(
            "All Stations Side-by-Side: todos os boxplots do cluster no mesmo "
            "painel, justapostos e ordenados por mediana crescente."
        ),
    )
    station_multiselect = gr.CheckboxGroup(
        choices=[],
        label="Select Stations (Comparison / Single modes)",
        info="Preenchido ao carregar o detalhe do cluster.",
    )
    # ── Sub-painel Monthly ────────────────────────────────────────────────────
    gr.HTML('<p class="sub-heading">Monthly Time-Series Options</p>')
    with gr.Row():
        with gr.Column(scale=1):
            monthly_mode = gr.Radio(
                choices=[
                    ("All Months — single panel", "all_months"),
                    ("By Year — one panel per year", "by_year"),
                    ("By Station — panels per station", "by_station"),
                ],
                value="all_months",
                label="Time-Series Mode",
            )
        with gr.Column(scale=1):
            ts_years = gr.CheckboxGroup(
                choices=YEAR_CHOICES, label="Years (vazio = todos)",
            )
            ts_months = gr.CheckboxGroup(
                choices=MONTH_CHOICES, label="Months (vazio = todos)",
            )
            ts_station_multiselect = gr.CheckboxGroup(
                choices=[],
                label="Stations for Time-Series (By Station mode)",
            )
    btn_analise = gr.Button(
        "Generate Climate Analysis", variant="primary", size="lg"
    )
    unified_plot = gr.Plot(label="Climate Analysis — Boxplot / Time-Series")
    unified_stats_table = gr.Dataframe(
        label="Descriptive Statistics",
        interactive=False, wrap=False,
    )
    # ════════════════════════════════════════════════════════════════════════
    # ⑦ Dendrogram
    # ════════════════════════════════════════════════════════════════════════
    gr.HTML('<p class="section-heading">⑦ Hierarchical Clustering — Dendrogram</p>')
    with gr.Row():
        dendro_samples = gr.Slider(
            minimum=50, maximum=400, step=50, value=200,
            label="Max Stations in Sample",
        )
        dendro_method = gr.Dropdown(
            choices=["ward","average","complete","single"],
            value="ward", label="Linkage Method",
        )
    dendro_plot = gr.Plot(label="Dendrogram — Hierarchical Clustering")
    btn_dendro = gr.Button("Generate Dendrogram", variant="secondary")

    # ════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ════════════════════════════════════════════════════════════════════════
    def cb_run(n_value, show_lt):
        """Executa clustering e atualiza diagrama, mapa Folium e tabelas."""
        fig, summary, geo_df, clim_df, folium_html = gerar_voronoi(
            n_value, show_lt=show_lt
        )
        cluster_ids = (
            [str(int(c)) for c in geo_df["Cluster ID"].tolist()]
            if geo_df is not None and len(geo_df) > 0 else []
        )
        default_id = cluster_ids[0] if cluster_ids else None
        detail_fig = (
            plot_cluster_voronoi_only(default_id, show_lt=show_lt)
            if default_id else None
        )
        return (
            fig, summary, geo_df, clim_df,
            folium_html,
            gr.update(choices=cluster_ids, value=default_id),
            detail_fig,
        )

    btn_run.click(
        fn=cb_run,
        inputs=[slider_n, chk_lt_voronoi],
        outputs=[
            voronoi_plot, summary_box,
            geo_table, clim_table,
            folium_map_html,
            cluster_selector, cluster_map,
        ],
    )

    # Executa clustering automaticamente ao abrir o dashboard
    app.load(
        fn=cb_run,
        inputs=[slider_n, chk_lt_voronoi],
        outputs=[
            voronoi_plot, summary_box,
            geo_table, clim_table,
            folium_map_html,
            cluster_selector, cluster_map,
        ],
    )

    def cb_detail(cluster_id_str, show_lt):
        """
        Atualiza o painel de detalhe do cluster:
        • Mapa Matplotlib do cluster
        • Cluster Summary Card HTML ← NOVO v3
        • Tabela de estações e tensor
        • Tabela climática
        • Tabela de distâncias LT ← NOVO v3
        • Seletores de estação (aggregate e monthly)
        """
        df_out, tensor_str, info, choices = cluster_station_details(cluster_id_str)
        fig_detail = plot_cluster_voronoi_only(cluster_id_str, show_lt=show_lt)
        df_clim = climate_summary_for_cluster(cluster_id_str)
        df_dist = calcular_distancias_cluster(cluster_id_str) # ← NOVO v3
        disp_html = gerar_display_cluster_html(cluster_id_str) # ← NOVO v3
        return (
            fig_detail, disp_html,
            df_out, tensor_str, info,
            df_clim, df_dist,
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, value=[]),
        )

    btn_detail.click(
        fn=cb_detail,
        inputs=[cluster_selector, chk_lt_detail],
        outputs=[
            cluster_map, cluster_display_html,
            stations_table, tensor_box, tensor_info,
            climate_detail_table, lt_distance_table,
            station_multiselect,
            ts_station_multiselect,
        ],
    )

    # Atualiza o Summary Card ao mudar cluster_selector (sem clicar no botão)
    cluster_selector.change(
        fn=gerar_display_cluster_html,
        inputs=[cluster_selector],
        outputs=[cluster_display_html],
    )

    def cb_analise(
        cluster_id_str, climate_var,
        a_type,
        agg_mode, sel_stations,
        m_mode, years, months, stations_ts,
    ):
        """Dispatcher da análise unificada de boxplot."""
        return gerar_analise_unificada(
            cluster_id_str, climate_var,
            a_type,
            agg_mode, sel_stations,
            m_mode, years, months, stations_ts,
        )

    btn_analise.click(
        fn=cb_analise,
        inputs=[
            cluster_selector, unified_climate_var,
            analysis_type,
            aggregate_mode, station_multiselect,
            monthly_mode, ts_years, ts_months, ts_station_multiselect,
        ],
        outputs=[unified_plot, unified_stats_table],
    )

    btn_heatmap.click(
        fn=gerar_heatmap_clima,
        inputs=[heatmap_var, chk_lt_heatmap],
        outputs=heatmap_plot,
    )

    btn_dendro.click(
        fn=gerar_dendrograma,
        inputs=[dendro_samples, dendro_method],
        outputs=dendro_plot,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 22. LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
import os
# O Render define uma variável de ambiente 'PORT'. Usamos ela, ou 7860 como fallback.
port = int(os.environ.get("PORT", 7860))
# 'server_name="0.0.0.0"' é crucial para que o aplicativo seja acessível externamente no container.
app.launch(share=False, debug=False, server_name="0.0.0.0", server_port=port)