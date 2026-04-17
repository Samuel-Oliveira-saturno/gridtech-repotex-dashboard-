# Project — RePoTEx 🌎📡  
**Voronoi Geographic Clustering of INMET Weather Stations**

---

## 1. Visão Geral

O **Project — RePoTEx** é um dashboard interativo para **clusterização geográfica** de estações meteorológicas do **INMET** (Instituto Nacional de Meteorologia) utilizando:

- **K-Means** para agrupamento geográfico (latitude/longitude);
- **Diagramas de Voronoi** para delimitação de áreas de influência;
- **GeoPandas / Shapely** para operações geoespaciais;
- **Gradio** para interface web;
- Deploy em ambiente de nuvem via **Render**.

O objetivo é fornecer uma ferramenta visual para apoiar análises de cobertura espacial, planejamento de redes de monitoramento e exploração de agrupamentos de estações.

---

## 2. Funcionalidades Principais

1. 🎯 **Clusterização Geográfica com K-Means**
   - Agrupamento das estações com base em latitude e longitude.
   - Definição dinâmica do número de estações por cluster (N) via slider.

2. 🗺️ **Mapa de Voronoi Clippado ao Brasil**
   - Geração de células de Voronoi a partir dos centróides dos clusters.
   - Recorte das células ao polígono do Brasil (fronteira nacional e, quando disponível, estados).

3. 📊 **Resumo Estatístico dos Clusters**
   - Número total de estações.
   - Número de clusters gerados.
   - Tamanho mínimo, mediano e máximo dos clusters.
   - Contagem de células de Voronoi válidas.

4. 📋 **Tabela de Clusters**
   - Listagem de clusters com:
     - `cluster_id`;
     - número de estações;
     - coordenadas do centróide (lat/lon);
     - lista das estações (por nome, se disponível).

5. 🔍 **Detalhamento por Cluster**
   - Seleção de um cluster específico em dropdown.
   - Mapa focado na região de Voronoi do cluster selecionado.
   - Tabela com estações daquele cluster (nome/latitude/longitude).
   - Tensor NumPy (lat/lon) exibido como texto, com informação de shape.

6. 🗂️ **Exportação GeoJSON**
   - Geração opcional de arquivos `.geojson` com as células de Voronoi recortadas, salvas em `geojson_outputs/`.

---

## 3. Estrutura do Projeto

```text
gridtech-repotex-dashboard-/
├─ app.py                  # Aplicação principal (Gradio)
├─ requirements.txt        # Dependências Python
├─ logoGT.png              # Logo exibida no cabeçalho do dashboard
├─ data/
│  └─ station_geo.parquet  # Base de estações do INMET (latitude/longitude e outros campos)
└─ geojson_outputs/        # Saída de arquivos GeoJSON (gerados em runtime)
```

---

## 4. Tecnologias Utilizadas

- **Linguagem**: Python 3.x  
- **Interface Web**: Gradio  
- **Geoprocessamento**:
  - GeoPandas
  - Shapely
  - SciPy (Voronoi)
- **Machine Learning**:
  - scikit-learn (KMeans)
  - StandardScaler
- **Visualização**:
  - Matplotlib
- **Dados**:
  - Pandas
  - PyArrow (leitura Parquet)
- **Infraestrutura**:
  - Deploy em Render (Web Service)
  - Código hospedado no GitHub

---

## 5. Instalação e Execução Local

### 5.1. Pré-requisitos

- Python 3.9+ instalado.
- `git` instalado (opcional, se for clonar o repositório).
- Dependências do `requirements.txt`.

### 5.2. Clonando o repositório

```bash
git clone https://github.com/Samuel-Oliveira-saturno/gridtech-repotex-dashboard-.git
cd gridtech-repotex-dashboard-
```

### 5.3. Criando ambiente virtual (opcional, mas recomendado)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou
.\.venv\Scripts\activate   # Windows
```

### 5.4. Instalando dependências

```bash
pip install -r requirements.txt
```

### 5.5. Executando a aplicação localmente

```bash
python app.py
```

Em seguida, acesse no navegador:

```text
http://localhost:7860
```

---

## 6. Deploy em Produção (Render)

O projeto foi preparado para deploy como **Web Service** no Render.

### 6.1. Configuração de ambiente

No painel do Render, ao criar o Web Service:

- **Source**: GitHub → `Samuel-Oliveira-saturno/gridtech-repotex-dashboard-`
- **Environment**: Python
- **Build Command**:

  ```bash
  pip install -r requirements.txt
  ```

- **Start Command**:

  ```bash
  python app.py
  ```

O arquivo `app.py` está configurado para usar a porta definida pelo Render:

```python
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "7860"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        debug=False
    )
```

### 6.2. Auto-deploy

Recomendado habilitar **Auto Deploy** a partir da branch principal (`main`/`master`), para que todo `git push` gere um novo deploy automático.

---

## 7. Dados de Entrada

O aplicativo espera um arquivo Parquet com as estações meteorológicas em:

```text
data/station_geo.parquet
```

### 7.1. Colunas mínimas esperadas

- `latitude` (float)  
- `longitude` (float)

Opcionalmente, para exibir nomes de estações:

- Uma das colunas: `station`, `station_name`, `nome`, `estacao`, `name`  
  (o código detecta automaticamente a primeira que encontrar).

### 7.2. Pré-processamento aplicado

- Remoção de linhas com `latitude` ou `longitude` nulas.
- Filtro de sanidade geográfica para manter apenas pontos dentro do retângulo aproximado do Brasil:

  - Latitude: de -35.0 a 5.5  
  - Longitude: de -75.0 a -28.0  

---

## 8. Como o Dashboard Funciona

### 8.1. Fluxo geral

1. Carregamento dos dados das estações (Parquet).
2. Aplicação de K-Means em `(latitude, longitude)`:
   - Número de clusters `k` é definido aproximadamente por:
     |$
     k \approx \frac{\text{número total de estações}}{N}
     $|
     onde `N` é o número de estações por cluster, ajustado via slider.

3. Cálculo dos centróides dos clusters em coordenadas geográficas.
4. Geração do diagrama de Voronoi (SciPy), com recorte ao polígono do Brasil (Shapely).
5. Renderização do mapa principal:
   - Polígonos de Voronoi coloridos;
   - Estações por cluster;
   - Centróides marcados com estrela.

6. Geração de:
   - Resumo textual;
   - Tabela de clusters;
   - GeoJSON opcional.

7. Detalhamento de cluster:
   - Mapa focado no cluster selecionado;
   - Tabela com estações;
   - Tensor NumPy (latitude/longitude) exibido em texto.

---

## 9. Interface do Usuário (Gradio)

### 9.1. Elementos principais

- **Cabeçalho**:
  - Logo (`logoGT.png`) à esquerda.
  - Título: `Project — RePoTEx` e subtítulo explicativo.

- **Painel de parâmetros**:
  - Slider: `N — Stations per cluster` (1 a 20).
  - Botão: `Generate Voronoi Diagram`.
  - Caixa de texto: resumo do clustering.

- **Visualizações**:
  - `Plot` principal: mapa do Brasil com clusters + Voronoi + centróides.
  - Tabela de clusters com lista de estações.

- **Seção de detalhes**:
  - Dropdown: seleção de `cluster_id`.
  - Mapa detalhado da região de Voronoi do cluster selecionado.
  - Tabela de estações do cluster.
  - Caixa de texto com o tensor NumPy e shape.

### 9.2. Ajustes de UI

- Botões padrão de **download/fullscreen/share** sobre imagens/plots são ocultados via CSS para manter uma interface mais limpa, sem afetar a funcionalidade principal.

---

## 10. Limitações e Possíveis Melhorias

### 10.1. Limitações atuais

- O agrupamento considera apenas latitude/longitude (não incorpora atributos meteorológicos).
- A resolução dos shapefiles de países/estados é do Natural Earth (110m), adequada para visão geral, mas não para análises muito detalhadas.
- A geração de Voronoi pode ser custosa se o número de clusters for muito alto.

### 10.2. Melhorias futuras

- Incluir **filtros por região** (região, estado) e por tipo de estação.
- Permitir parametrização de `k` diretamente, além de `N`.
- Incorporar métricas adicionais (ex.: distância média intra-cluster).
- Oferecer opções de exportação de resultados via botões específicos do dashboard (CSV/GeoJSON), em vez de depender de botões do frontend padrão.

---

## 11. Contato e Contribuição

Contribuições são bem-vindas! Sugestões de melhoria, correções de bugs ou novas funcionalidades podem ser feitas via:

- **Issues** no repositório GitHub.  
- **Pull Requests**, seguindo boas práticas:
  - Descrever claramente o objetivo da alteração;
  - Manter o código organizado e comentado;
  - Garantir que o app continua rodando localmente (`python app.py`).

---

**Project — RePoTEx**  
Dashboard de clusterização geográfica e Voronoi para estações do INMET, com foco em análise espacial, visualização e explorabilidade interativa.
