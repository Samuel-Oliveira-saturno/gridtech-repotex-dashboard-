import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend não interativo (necessário no server)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import geopandas as gpd
import gradio as gr
import requests
import tempfile
import os
import warnings

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ─────────────────────────────
# CONFIGURAÇÕES GERAIS E CAMINHOS
# ─────────────────────────────

# Em produção (Render), usamos caminhos relativos ao arquivo app.py
BASE        = Path(__file__).parent
INPUT_PATH  = BASE / "data" / "station_geo.parquet"
GEOJSON_DIR = BASE / "geojson_outputs"
GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

LOGO_PATH   = BASE / "logoGT.png"

# Paleta de cores para clusters (até 32)
PALETTE_32 = [
    "#1976D2","#E53935","#43A047","#FB8C00","#8E24AA","#00ACC1",
    "#F4511E","#3949AB","#00897B","#FFB300","#6D4C41","#546E7A",
    "#D81B60","#00BCD4","#7CB342","#FDD835","#5E35B1","#039BE5",
    "#E64A19","#00838F","#558B2F","#F9A825","#4527A0","#01579B",
    "#BF360C","#006064","#33691E","#F57F17","#4A148C","#01579B",
    "#880E4F","#1B5E20",
]

# ─────────────────────────────
# CARREGAMENTO DA LOGO (CABEÇALHO)
# ─────────────────────────────

try:
    logo_arr = mpimg.imread(LOGO_PATH)
    print("Logo loaded:", LOGO_PATH)
except Exception as e:
    logo_arr = None
    print("Could not load logo:", e)

# ─────────────────────────────
# CARREGAMENTO DE DADOS BASE
# ─────────────────────────────

print("Loading base data...")

TMP = tempfile.mkdtemp()

def download_shapefile(url, tmp_dir, shp_name):
    """
    Faz download de um shapefile zipado do Natural Earth (ou similar)
    e carrega via caminho virtual do GDAL.
    """
    zip_path = os.path.join(tmp_dir, "shape.zip")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)
    return gpd.read_file(f"/vsizip/{zip_path}/{shp_name}")

# Países
gdf_world = download_shapefile(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
    TMP,
    "ne_110m_admin_0_countries.shp"
)

# Brasil
gdf_br = None
for col in ["SOVEREIGNT", "NAME", "NAME_LONG", "ADMIN"]:
    if col in gdf_world.columns:
        match = gdf_world[gdf_world[col].str.contains("Brazil|Brasil", na=False)]
        if len(match) > 0:
            gdf_br = match.to_crs(epsg=4326)
            break
if gdf_br is None:
    raise RuntimeError("Brazil polygon not found in world shapefile.")

# Estados do Brasil (se disponíveis)
try:
    gdf_states_world = download_shapefile(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_1_states_provinces.zip",
        TMP,
        "ne_110m_admin_1_states_provinces.shp"
    )
    gdf_states = None
    for col in ["admin", "ADMIN", "sovereignt", "SOVEREIGNT"]:
        if col in gdf_states_world.columns:
            match = gdf_states_world[
                gdf_states_world[col].str.contains("Brazil|Brasil", na=False)
            ]
            if len(match) > 0:
                gdf_states = match.to_crs(epsg=4326)
                break
    has_states = gdf_states is not None and len(gdf_states) > 0
except Exception:
    has_states = False

# Estações meteorológicas
df_raw = pd.read_parquet(INPUT_PATH)
df = df_raw.dropna(subset=["latitude", "longitude"]).copy()

# Filtro de bounding box aproximado do Brasil (sanidade)
df = df[
    (df["latitude"] >= -35.0) & (df["latitude"] <= 5.5) &
    (df["longitude"] >= -75.0) & (df["longitude"] <= -28.0)
].copy()

n_stations_total = len(df)
print("Stations loaded:", n_stations_total)

# Detecta possível coluna com nome da estação
STATION_COL = None
for candidate in ["station", "station_name", "nome", "estacao", "name"]:
    if candidate in df.columns:
        STATION_COL = candidate
        break
print("Station name column:", STATION_COL)

# Features para clusterização
GEO_FEATURES = ["latitude", "longitude"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[GEO_FEATURES].values)

# Polígono unificado do Brasil para recorte do Voronoi
brasil_poly = unary_union(gdf_br.geometry)

# Variáveis globais para visualizações de detalhe
last_df_temp     = None  # DataFrame com cluster_geo
last_cells       = None  # Lista de polígonos de Voronoi (shapely)
last_centroids   = None  # Centróides lat/lon
last_actual_k    = None  # Número de clusters efetivos

# ─────────────────────────────
# FUNÇÕES AUXILIARES DE PLOT
# ─────────────────────────────

def plot_brasil_base(ax):
    """
    Desenha o fundo do mapa do Brasil (polígono e estados).
    """
    gdf_br.plot(
        ax=ax, color="#C5E1F4",
        edgecolor="#335C67", linewidth=0.8, zorder=1
    )
    if has_states:
        gdf_states.plot(
            ax=ax, color="none",
            edgecolor="#90A4AE", linewidth=0.4, zorder=2
        )
    ax.set_xlim(-75.5, -27.5)
    ax.set_ylim(-35.5, 6.0)
    ax.set_facecolor("#DDEEFF")
    ax.grid(True, linestyle="--", alpha=0.25)


def voronoi_clipped(centroids_latlon, clip_poly, margin=3.0):
    """
    Calcula o diagrama de Voronoi a partir dos centróides (lat, lon),
    transformando para (lon, lat) no plano, e recorta ao polígono do Brasil.

    Retorna uma lista de polígonos shapely (ou None em casos degenerados).
    """
    pts_xy = centroids_latlon[:, [1, 0]]   # (lon, lat)
    minx, miny, maxx, maxy = clip_poly.bounds
    bbox   = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
    vor    = Voronoi(pts_xy)
    cells  = []

    for i in range(len(pts_xy)):
        region = vor.regions[vor.point_region[i]]
        if not region:
            cells.append(None)
            continue
        # Regiões com -1 são infinitas; usamos bounding box
        poly    = Polygon(vor.vertices[region]) if -1 not in region else bbox
        clipped = poly.intersection(clip_poly)
        cells.append(clipped if not clipped.is_empty else None)

    return cells

# ─────────────────────────────
# FUNÇÃO PRINCIPAL DE CLUSTERIZAÇÃO + VORONOI
# ─────────────────────────────

def gerar_voronoi(n_raw):
    """
    Executa K-Means e gera:
      - mapa principal de Voronoi (polígonos + estações + centróides),
      - resumo estatístico,
      - tabela de clusters com lista de estações.

    n_raw: inteiro N = estações por cluster (1..20).
    """
    n_stations_per_cluster = int(n_raw)
    if n_stations_per_cluster < 1:
        n_stations_per_cluster = 1

    # Define número de clusters (K) ~ n_total / N, com mínimo 2
    k_clusters = max(2, int(n_stations_total / n_stations_per_cluster))

    # K-Means
    km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    df_temp = df.copy()
    df_temp["cluster_geo"] = labels

    # Atualiza globais
    global last_df_temp, last_cells, last_centroids, last_actual_k
    last_df_temp = df_temp.copy()

    # Centróides reais (em lat/lon)
    centroids_geo_list = []
    for c in range(k_clusters):
        mask = df_temp["cluster_geo"] == c
        if mask.sum() > 0:
            centroids_geo_list.append(
                df_temp.loc[mask, GEO_FEATURES].mean(axis=0).values
            )

    centroids_geo = np.array(centroids_geo_list)
    actual_k = len(centroids_geo)
    last_centroids = centroids_geo
    last_actual_k  = actual_k

    if actual_k < 2:
        last_cells = None
        empty_df = pd.DataFrame(columns=[
            "cluster_id", "n_stations", "centroid_lat",
            "centroid_lon", "stations"
        ])
        return None, "Insufficient clusters to generate Voronoi.", empty_df

    # Células de Voronoi recortadas ao Brasil
    cells = voronoi_clipped(centroids_geo, brasil_poly)
    last_cells = cells
    n_valid = sum(c is not None for c in cells)

    # Exporta GeoJSON (opcional)
    geojson_features = []
    for i, cell in enumerate(cells):
        if cell is None or cell.is_empty:
            continue
        cid = i % len(PALETTE_32)
        color_hex = PALETTE_32[cid]
        props = {
            "name": f"Cluster {cid+1} (N={n_stations_per_cluster})",
            "cluster_id": cid + 1,
            "fill_color": color_hex,
            "fill_opacity": 0.22,
            "stroke_color": "#78909C",
            "stroke_width": 0.5
        }
        geoms = [cell] if isinstance(cell, Polygon) else list(cell.geoms)
        for part in geoms:
            geojson_features.append({
                "geometry": part.__geo_interface__,
                "properties": props.copy()
            })

    geojson_path = None
    if geojson_features:
        gdf_geojson = gpd.GeoDataFrame.from_features(geojson_features, crs="EPSG:4326")
        geojson_path = GEOJSON_DIR / f"voronoi_n{n_stations_per_cluster:02d}.geojson"
        gdf_geojson.to_file(str(geojson_path), driver="GeoJSON")

    # ── MAPA PRINCIPAL: VORONOI + ESTAÇÕES + CENTRÓIDES ──────

    fig, ax = plt.subplots(figsize=(10, 11))
    fig.patch.set_facecolor("#F7F9FB")
    ax.set_facecolor("#DDEEFF")

    plot_brasil_base(ax)

    # 1) Regiões de Voronoi coloridas por cluster
    for i, cell in enumerate(cells):
        if cell is None or cell.is_empty:
            continue
        color = PALETTE_32[i % len(PALETTE_32)]
        try:
            polys = [cell] if isinstance(cell, Polygon) else list(cell.geoms)
            for p in polys:
                x, y = p.exterior.xy
                ax.fill(x, y, alpha=0.30, color=color, zorder=2)
                ax.plot(x, y, color="#78909C", lw=0.4, zorder=3)
        except Exception:
            continue

    # 2) Pontos das estações, coloridos por cluster
    for c in range(actual_k):
        mask = df_temp["cluster_geo"] == c
        if mask.sum() > 0:
            ax.scatter(
                df_temp.loc[mask, "longitude"],
                df_temp.loc[mask, "latitude"],
                s=12,
                alpha=0.9,
                color=PALETTE_32[c % len(PALETTE_32)],
                edgecolor="white",
                linewidth=0.3,
                zorder=4,
                label=None
            )

    # 3) Centróides de clusters
    ax.scatter(
        centroids_geo[:, 1], centroids_geo[:, 0],
        s=70,
        marker="*",
        color="#FFFFFF",
        edgecolors="#263238",
        linewidths=0.7,
        zorder=5,
        label="Cluster centroids"
    )

    ax.set_title(
        "Voronoi Diagram — Geographic Clustering (Brazil)\n"
        "K-Means  |  k={}  |  ~{} stations/cluster  |  n={} stations".format(
            actual_k, n_stations_per_cluster, n_stations_total
        ),
        fontsize=13,
        fontweight="bold",
        color="#1565C0",
        pad=14
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    plt.tight_layout()

    # ── RESUMO ESTATÍSTICO ────────────────────────────────────
    sizes = df_temp["cluster_geo"].value_counts()
    geojson_info = str(geojson_path) if geojson_path else "N/A"
    summary = (
        "Clustering Summary\n"
        + "-" * 40 + "\n"
        + "  Total stations    : {}\n".format(n_stations_total)
        + "  N per cluster     : {}\n".format(n_stations_per_cluster)
        + "  Clusters generated: {}\n".format(actual_k)
        + "  Voronoi cells     : {}\n".format(n_valid)
        + "-" * 40 + "\n"
        + "  Min cluster size  : {} stations\n".format(sizes.min())
        + "  Median cluster    : {:.0f} stations\n".format(sizes.median())
        + "  Max cluster size  : {} stations\n".format(sizes.max())
        + "-" * 40 + "\n"
        + "  GeoJSON saved at  :\n  {}".format(geojson_info)
    )

    # ── TABELA DE CLUSTERS ────────────────────────────────────
    rows = []
    for c in range(actual_k):
        mask = df_temp["cluster_geo"] == c
        subset = df_temp[mask]
        if subset.empty:
            continue

        if STATION_COL:
            station_list = " | ".join(
                subset[STATION_COL].dropna().astype(str).tolist()
            )
        else:
            station_list = " | ".join(
                "({:.2f},{:.2f})".format(r["latitude"], r["longitude"])
                for _, r in subset.iterrows()
            )

        lat_c = subset["latitude"].mean()
        lon_c = subset["longitude"].mean()

        rows.append({
            "cluster_id"   : int(c + 1),
            "n_stations"   : int(len(subset)),
            "centroid_lat" : round(lat_c, 4),
            "centroid_lon" : round(lon_c, 4),
            "stations"     : station_list,
        })

    cluster_df = pd.DataFrame(rows, columns=[
        "cluster_id", "n_stations", "centroid_lat",
        "centroid_lon", "stations"
    ])

    return fig, summary, cluster_df

# ─────────────────────────────
# FUNÇÕES DE DETALHE POR CLUSTER
# ─────────────────────────────

def cluster_station_details(cluster_id_str):
    """
    Retorna:
      - DataFrame com estações do cluster selecionado (nome/lat/lon),
      - representação textual do tensor (NumPy array),
      - texto com shape do tensor.
    """
    global last_df_temp
    if last_df_temp is None:
        return pd.DataFrame(), "", "Run clustering first."

    if not cluster_id_str:
        return pd.DataFrame(), "", "Select a cluster."

    cluster_id = int(cluster_id_str)
    c_index = cluster_id - 1

    if "cluster_geo" not in last_df_temp.columns:
        return pd.DataFrame(), "", "Internal error: 'cluster_geo' not found."

    subset = last_df_temp[last_df_temp["cluster_geo"] == c_index]
    if subset.empty:
        return pd.DataFrame(), "", f"No stations found for cluster {cluster_id}"

    cols = ["latitude", "longitude"]
    if STATION_COL:
        cols = [STATION_COL, "latitude", "longitude"]

    df_out = subset[cols].reset_index(drop=True)

    coords = subset[["latitude", "longitude"]].values
    tensor = np.array(coords, dtype=float)
    tensor_str = str(tensor)
    info = f"Tensor shape: {tensor.shape}"

    return df_out, tensor_str, info


def plot_cluster_voronoi_only(cluster_id_str):
    """
    Mapa de detalhe para o cluster selecionado:
      - mostra TODAS as regiões de Voronoi em cinza claro,
      - destaca a região do cluster selecionado em azul,
      - plota APENAS os pontos das estações daquele cluster,
      - NÃO mostra centróides.
    """
    global last_cells, last_centroids, last_actual_k, last_df_temp

    if last_cells is None or last_centroids is None or last_actual_k is None:
        return None

    if not cluster_id_str:
        return None

    cluster_id = int(cluster_id_str)
    c_index = cluster_id - 1
    if c_index < 0 or c_index >= last_actual_k:
        return None

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor("#F7F9FB")
    ax.set_facecolor("#DDEEFF")

    plot_brasil_base(ax)

    # 1) Todas as regiões de Voronoi em cinza claro
    for i, cell in enumerate(last_cells):
        if cell is None or cell.is_empty:
            continue
        base_color = "#CFD8DC"
        try:
            polys = [cell] if isinstance(cell, Polygon) else list(cell.geoms)
            for p in polys:
                x, y = p.exterior.xy
                ax.fill(x, y, alpha=0.20, color=base_color, zorder=2)
                ax.plot(x, y, color="#78909C", lw=0.4, zorder=3)
        except Exception:
            continue

    # 2) Destaca a região do cluster selecionado em azul
    cell_sel = last_cells[c_index] if c_index < len(last_cells) else None
    if cell_sel is not None and not cell_sel.is_empty:
        highlight_color = "#1976D2"
        try:
            polys = [cell_sel] if isinstance(cell_sel, Polygon) else list(cell_sel.geoms)
            for p in polys:
                x, y = p.exterior.xy
                ax.fill(x, y, alpha=0.35, color=highlight_color, zorder=4)
                ax.plot(x, y, color="#0D47A1", lw=0.8, zorder=5)
        except Exception:
            pass

    # 3) Pontos das estações APENAS do cluster selecionado
    if last_df_temp is not None and "cluster_geo" in last_df_temp.columns:
        subset = last_df_temp[last_df_temp["cluster_geo"] == c_index]
        if not subset.empty:
            ax.scatter(
                subset["longitude"],
                subset["latitude"],
                s=32,
                color="#FFFFFF",
                edgecolor="#0D47A1",
                linewidth=0.6,
                alpha=0.95,
                zorder=6,
                label="Stations in selected cluster"
            )

    ax.set_title(
        f"Voronoi Region and Stations — Cluster {cluster_id}",
        fontsize=12,
        fontweight="bold",
        color="#1565C0",
        pad=10
    )
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    return fig

# ─────────────────────────────
# INTERFACE GRADIO
# ─────────────────────────────

with gr.Blocks(
    title="Gridtech — Voronoi Clustering Dashboard",
    theme=gr.themes.Soft()
) as app:

    # Cabeçalho com logo + textos
    with gr.Row():
        with gr.Column(scale=0, min_width=120):
            gr.Image(
                value=logo_arr,
                label=None,
                interactive=False,
                show_label=False,
                show_download_button=False,
                show_fullscreen_button=False,
                show_share_button=False  # se existir na sua versão
            )
        with gr.Column(scale=1):
            gr.Markdown(
                "## Project — RePoTEx\n"
                "Voronoi Geographic Clustering of INMET Weather Stations"
            )

    # Linha principal: parâmetros + mapa principal
    with gr.Row():
        # Painel de controles
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### Clustering Parameters")

            slider_n = gr.Slider(
                minimum=1, maximum=20, step=1, value=5,
                label="N — Stations per cluster",
                info="Any integer between 1 and 20"
            )

            btn_run = gr.Button("Generate Voronoi Diagram", variant="primary", size="lg")

            gr.Markdown("### Summary")
            summary_box = gr.Textbox(
                label="Clustering Summary",
                lines=14,
                interactive=False
            )

        # Mapa principal
        with gr.Column(scale=2):
            voronoi_plot = gr.Plot(label="Voronoi Diagram (Clusters, Stations, Centroids)")

    # Tabela de clusters
    gr.Markdown("### Cluster Summary Table")
    cluster_table = gr.Dataframe(
        label="Cluster Summary (with station list)",
        interactive=False,
        wrap=True
    )

    # Seção de detalhes por cluster
    gr.Markdown("### Cluster Station Detail")
    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            cluster_selector = gr.Dropdown(
                label="Select Cluster ID",
                choices=[],
                value=None
            )
            btn_detail = gr.Button("Show Cluster Stations")
        with gr.Column(scale=2):
            cluster_map = gr.Plot(label="Selected Cluster Voronoi Region and Stations")

    with gr.Row():
        with gr.Column(scale=2):
            stations_table = gr.Dataframe(
                label="Stations in Selected Cluster",
                interactive=False
            )
        with gr.Column(scale=1):
            tensor_box = gr.Textbox(
                label="Latitude/Longitude Tensor (NumPy)",
                lines=6,
                interactive=False
            )
            tensor_info = gr.Textbox(
                label="Tensor Info",
                interactive=False
            )

    gr.Markdown(
        "---\n"
        "**How it works:** K-Means groups stations geographically using latitude/longitude. "
        "K is computed as `total_stations / N`. Centroids define Voronoi cells, which are "
        "clipped to Brazil's boundary."
    )

    # ── Callback para rodar clustering e atualizar tudo ───────
    def run_and_update(n_value):
        fig, summary, c_df = gerar_voronoi(n_value)
        cluster_ids = [str(int(c)) for c in c_df["cluster_id"].tolist()]
        default_id = cluster_ids[0] if cluster_ids else None
        detail_fig = plot_cluster_voronoi_only(default_id) if default_id else None
        return (
            fig,
            summary,
            c_df,
            gr.update(choices=cluster_ids, value=default_id),
            detail_fig
        )

    btn_run.click(
        fn=run_and_update,
        inputs=slider_n,
        outputs=[voronoi_plot, summary_box, cluster_table, cluster_selector, cluster_map]
    )

    app.load(
        fn=run_and_update,
        inputs=slider_n,
        outputs=[voronoi_plot, summary_box, cluster_table, cluster_selector, cluster_map]
    )

    # ── Callback de detalhe do cluster ────────────────────────
    def detail_and_map(cluster_id_str):
        df_out, tensor_str, info = cluster_station_details(cluster_id_str)
        fig = plot_cluster_voronoi_only(cluster_id_str)
        return df_out, tensor_str, info, fig

    btn_detail.click(
        fn=detail_and_map,
        inputs=cluster_selector,
        outputs=[stations_table, tensor_box, tensor_info, cluster_map]
    )

# ─────────────────────────────
# LANÇAMENTO PARA PRODUÇÃO (RENDER)
# ─────────────────────────────

if __name__ == "__main__":
    # Render fornece a porta em PORT
    port = int(os.getenv("PORT", "7860"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        debug=False
    )