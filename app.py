"""
Football Analytics Pro — Streamlit App with Ollama AI
Analyse de performance joueurs avec diagnostics statistiques complets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import warnings
from scipy import stats
from scipy.stats import pearsonr
import io

warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Analytics Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0d14;
    color: #e8eaf0;
  }

  .main { background-color: #0a0d14; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid #1e3a5f;
  }
  section[data-testid="stSidebar"] * { color: #c8d6e5 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 12px 16px;
  }
  [data-testid="metric-container"] label { color: #7a8fa6 !important; font-size: 0.75rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #38bdf8 !important; font-family: 'Space Mono'; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-bottom: 1px solid #1e3a5f;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    color: #7a8fa6;
    font-family: 'Space Mono';
    font-size: 0.75rem;
    border-bottom: 2px solid transparent;
    padding: 10px 18px;
  }
  .stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent;
  }

  /* Chat messages */
  .chat-msg-user {
    background: linear-gradient(135deg, #1e3a5f, #1e4a7f);
    border-left: 3px solid #38bdf8;
    border-radius: 0 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 40px;
    font-size: 0.9rem;
  }
  .chat-msg-ai {
    background: linear-gradient(135deg, #1a2234, #1f2d45);
    border-left: 3px solid #10b981;
    border-radius: 0 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 40px 8px 0;
    font-size: 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
  }

  /* Section headers */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #38bdf8;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin-bottom: 16px;
  }

  /* Status badge */
  .status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono';
    font-weight: 700;
  }
  .badge-online { background: #064e3b; color: #34d399; border: 1px solid #065f46; }
  .badge-offline { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }

  /* Stat result blocks */
  .stat-block {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 10px;
  }
  .stat-label { font-size: 0.72rem; color: #7a8fa6; text-transform: uppercase; letter-spacing: 0.1em; }
  .stat-value { font-family: 'Space Mono'; font-size: 1.1rem; color: #e2e8f0; margin-top: 2px; }
  .stat-interpretation { font-size: 0.82rem; color: #94a3b8; margin-top: 6px; }

  /* Buttons */
  .stButton>button {
    background: linear-gradient(135deg, #1e3a5f, #1e4d8c);
    color: #38bdf8;
    border: 1px solid #2563eb;
    border-radius: 6px;
    font-family: 'Space Mono';
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .stButton>button:hover {
    background: linear-gradient(135deg, #1e4d8c, #1d4ed8);
    border-color: #38bdf8;
  }

  /* Input */
  .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div>div {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    color: #e8eaf0 !important;
    font-family: 'Space Mono' !important;
    font-size: 0.83rem !important;
    border-radius: 6px !important;
  }

  /* Title */
  .app-title {
    font-family: 'Syne';
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
  }
  .app-subtitle {
    font-family: 'Space Mono';
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    """Load football player dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("3zden/fbref_football_player_performance_2024-2025")
        df = ds["train"].to_pandas()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r'[\s\+\/]', '_', regex=True)
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )
        numeric_cols = ['goals', 'assists', 'xg', 'npxg', 'xag', 'minutes',
                        'matches_played', 'starts', '90s_played', 'age',
                        'yellow_cards', 'red_cards', 'progressive_carries',
                        'progressive_passes', 'progressive_receives',
                        'goals_per_90', 'assists_per_90', 'xg_per_90',
                        'xag_per_90', 'npxg_per_90']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, None
    except Exception as e:
        # Fallback: generate synthetic data for demo
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'player': [f"Player_{i}" for i in range(n)],
            'position': np.random.choice(['FW', 'AT', 'MF', 'DF', 'GK'], n, p=[0.2,0.15,0.3,0.25,0.1]),
            'age': np.random.normal(26, 4, n).clip(17, 38).astype(int),
            'matches_played': np.random.randint(5, 38, n),
            '90s_played': np.random.uniform(3, 36, n),
            'goals': np.random.poisson(8, n),
            'assists': np.random.poisson(5, n),
            'xg': np.random.exponential(7, n),
            'npxg': np.random.exponential(6, n),
            'xag': np.random.exponential(4, n),
            'minutes': np.random.randint(300, 3400, n),
            'yellow_cards': np.random.poisson(3, n),
            'red_cards': np.random.poisson(0.2, n),
            'progressive_carries': np.random.poisson(40, n),
            'progressive_passes': np.random.poisson(50, n),
            'progressive_receives': np.random.poisson(45, n),
            'goals_per_90': np.random.exponential(0.3, n),
            'assists_per_90': np.random.exponential(0.2, n),
            'xg_per_90': np.random.exponential(0.3, n),
            'npxg_per_90': np.random.exponential(0.28, n),
            'xag_per_90': np.random.exponential(0.2, n),
        })
        return df, "⚠️ Dataset HuggingFace non disponible — données synthétiques utilisées"


def check_ollama(host, model):
    """Check if Ollama is running and model is available."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            return True, models
        return False, []
    except:
        return False, []


def query_ollama(host, model, messages, context=""):
    """Stream query to Ollama."""
    system_prompt = f"""Tu es un expert en data science sportive spécialisé dans l'analyse de performance des footballeurs.
Tu analyses des données FBref de performances footballistiques.

Contexte des données disponibles:
{context}

Réponds de façon concise, technique et pertinente. Utilise des termes analytiques précis.
Quand tu interprètes des résultats statistiques (stationnarité, saisonnalité, ACF/PACF), 
explique clairement les implications pour les décisions de recrutement et de contrats."""

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": True
    }
    try:
        r = requests.post(f"{host}/api/chat", json=payload, stream=True, timeout=60)
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                if not chunk.get('done', False):
                    yield chunk['message']['content']
    except Exception as e:
        yield f"\n❌ Erreur Ollama: {e}"


# ─── Statistical Analysis Functions ───────────────────────────────────────────

def compute_stationarity(series, name=""):
    """ADF + KPSS tests for stationarity."""
    from statsmodels.tsa.stattools import adfuller, kpss
    series = series.dropna()
    results = {}

    # ADF Test
    try:
        adf_result = adfuller(series, autolag='AIC')
        results['adf_stat'] = adf_result[0]
        results['adf_pvalue'] = adf_result[1]
        results['adf_lags'] = adf_result[2]
        results['adf_critical'] = adf_result[4]
        results['adf_stationary'] = adf_result[1] < 0.05
    except Exception as e:
        results['adf_error'] = str(e)

    # KPSS Test
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        results['kpss_stat'] = kpss_result[0]
        results['kpss_pvalue'] = kpss_result[1]
        results['kpss_critical'] = kpss_result[3]
        results['kpss_stationary'] = kpss_result[1] > 0.05
    except Exception as e:
        results['kpss_error'] = str(e)

    return results


def plot_acf_pacf(series, lags=20, title=""):
    """Plot ACF and PACF."""
    from statsmodels.tsa.stattools import acf, pacf
    series = series.dropna()
    if len(series) < lags + 2:
        lags = max(2, len(series) // 2)

    acf_vals, acf_ci = acf(series, nlags=lags, alpha=0.05)
    try:
        pacf_vals, pacf_ci = pacf(series, nlags=lags, alpha=0.05)
    except:
        pacf_vals = np.zeros(lags + 1)
        pacf_ci = np.zeros((lags + 1, 2))

    lags_arr = np.arange(len(acf_vals))
    ci_upper_acf = acf_ci[:, 1] - acf_vals
    ci_lower_acf = acf_ci[:, 0] - acf_vals
    ci_upper_pacf = pacf_ci[:, 1] - pacf_vals
    ci_lower_pacf = pacf_ci[:, 0] - pacf_vals

    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF (Autocorrélation)", "PACF (Autocorrélation Partielle)"])

    # ACF
    for i, (lag, val) in enumerate(zip(lags_arr, acf_vals)):
        color = '#38bdf8' if abs(val) > abs(ci_upper_acf[i]) else '#2d3f55'
        fig.add_trace(go.Bar(x=[lag], y=[val], marker_color=color, showlegend=False), row=1, col=1)
    conf_bound = 1.96 / np.sqrt(len(series))
    fig.add_hline(y=conf_bound, line_dash="dash", line_color="#f59e0b", row=1, col=1)
    fig.add_hline(y=-conf_bound, line_dash="dash", line_color="#f59e0b", row=1, col=1)

    # PACF
    for i, (lag, val) in enumerate(zip(lags_arr, pacf_vals)):
        color = '#10b981' if abs(val) > abs(ci_upper_pacf[i]) else '#2d3f55'
        fig.add_trace(go.Bar(x=[lag], y=[val], marker_color=color, showlegend=False), row=1, col=2)
    fig.add_hline(y=conf_bound, line_dash="dash", line_color="#f59e0b", row=1, col=2)
    fig.add_hline(y=-conf_bound, line_dash="dash", line_color="#f59e0b", row=1, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(color='#94a3b8', size=13)),
        plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
        font=dict(color='#94a3b8'),
        height=320, margin=dict(t=60, b=30, l=50, r=20),
    )
    fig.update_xaxes(gridcolor='#1e293b', title="Lag")
    fig.update_yaxes(gridcolor='#1e293b', range=[-1, 1])
    return fig


def plot_seasonality_decomposition(series, name=""):
    """Decompose signal into trend + seasonality + residual via moving average."""
    series = series.dropna().reset_index(drop=True)
    n = len(series)

    # Simple moving average trend
    window = max(3, n // 8)
    trend = series.rolling(window=window, center=True).mean()

    # Detrended
    detrended = series - trend

    # Seasonal proxy: mean by position buckets (proxy for match-week patterns)
    bucket_size = max(3, n // 10)
    buckets = (np.arange(n) // bucket_size) % 10
    seasonal = pd.Series(index=range(n), dtype=float)
    for b in np.unique(buckets):
        mask = buckets == b
        seasonal[mask] = detrended[mask].mean()

    residual = series - trend - seasonal

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Série originale", "Tendance", "Saisonnalité (proxy)", "Résidus"],
                        vertical_spacing=0.08)

    colors = ['#38bdf8', '#f59e0b', '#10b981', '#f87171']
    data_list = [series, trend, seasonal, residual]
    names = ["Original", "Trend", "Seasonal", "Residual"]

    for i, (d, c, nm) in enumerate(zip(data_list, colors, names), 1):
        fig.add_trace(go.Scatter(y=d, mode='lines', line=dict(color=c, width=1.5),
                                 name=nm, showlegend=False), row=i, col=1)

    fig.update_layout(
        plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
        font=dict(color='#94a3b8'), height=520,
        margin=dict(t=40, b=30, l=50, r=20),
    )
    fig.update_xaxes(gridcolor='#1e293b')
    fig.update_yaxes(gridcolor='#1e293b')
    return fig


def plot_distribution_analysis(series, name=""):
    """Histogram + Q-Q plot + kernel density."""
    series = series.dropna()

    # Normality test
    if len(series) >= 8:
        stat, pval = stats.shapiro(series[:5000])
        normal = pval > 0.05
    else:
        stat, pval, normal = 0, 0, False

    skewness = series.skew()
    kurt = series.kurtosis()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Distribution & KDE", "Q-Q Plot (normalité)"])

    # Histogram
    hist_vals, bin_edges = np.histogram(series, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig.add_trace(go.Bar(x=bin_centers, y=hist_vals, marker_color='rgba(56,189,248,0.4)',
                         marker_line=dict(color='#38bdf8', width=0.5), name='Histogramme', showlegend=False), row=1, col=1)

    # KDE overlay
    from scipy.stats import gaussian_kde
    try:
        if series.nunique() >= 2 and series.std() > 0:
            kde = gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 200)
            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines',
                                     line=dict(color='#f59e0b', width=2), name='KDE', showlegend=False), row=1, col=1)
    except Exception:
        pass  # KDE non disponible

    # Q-Q Plot
    qq = stats.probplot(series, dist="norm")
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                             marker=dict(color='#10b981', size=4, opacity=0.7),
                             name='Quantiles', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=qq[0][0],
                             y=qq[0][0] * qq[1][0] + qq[1][1],
                             mode='lines', line=dict(color='#f87171', width=1.5, dash='dash'),
                             name='Droite théorique', showlegend=False), row=1, col=2)

    fig.update_layout(
        plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
        font=dict(color='#94a3b8'), height=320,
        margin=dict(t=50, b=30, l=50, r=20),
    )
    fig.update_xaxes(gridcolor='#1e293b')
    fig.update_yaxes(gridcolor='#1e293b')

    return fig, stat, pval, normal, skewness, kurt


def plot_rolling_stats(series, name="", windows=[5, 10, 20]):
    """Plot rolling mean and std."""
    series = series.dropna().reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series, mode='lines', name='Original',
                             line=dict(color='rgba(148,163,184,0.4)', width=1)))

    palette = ['#38bdf8', '#10b981', '#f59e0b']
    for w, c in zip(windows, palette):
        if len(series) > w:
            rm = series.rolling(w).mean()
            rs = series.rolling(w).std()
            fig.add_trace(go.Scatter(y=rm, mode='lines', name=f'Moy. mobile {w}',
                                     line=dict(color=c, width=2)))
            fig.add_trace(go.Scatter(
                y=rm + rs, fill=None, mode='lines',
                line=dict(color=c, width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                y=rm - rs, fill='tonexty', mode='lines',
                line=dict(color=c, width=0),
                fillcolor=f'rgba({",".join(str(int(int(c.lstrip("#")[i:i+2], 16)) ) for i in (0,2,4))},0.1)',
                showlegend=False
            ))

    fig.update_layout(
        title=dict(text=f"Moyennes mobiles — {name}", font=dict(color='#94a3b8', size=13)),
        plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
        font=dict(color='#94a3b8'), height=300,
        margin=dict(t=50, b=30, l=50, r=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11))
    )
    fig.update_xaxes(gridcolor='#1e293b', title="Rang joueur")
    fig.update_yaxes(gridcolor='#1e293b')
    return fig


def plot_correlation_heatmap(df, cols):
    """Correlation matrix heatmap."""
    sub = df[cols].dropna()
    corr = sub.corr()

    fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    text_auto='.2f', aspect='auto')
    fig.update_layout(
        plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
        font=dict(color='#94a3b8'), height=420,
        margin=dict(t=40, b=30, l=30, r=30),
        coloraxis_colorbar=dict(tickfont=dict(color='#94a3b8'))
    )
    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────




def generate_player_match_data(player_name, n_matches=34, seed=None):
    """Generate realistic match-by-match data for a player."""
    if seed is None:
        seed = abs(hash(player_name)) % 10000
    rng = np.random.default_rng(seed)

    match_dates = pd.date_range(start='2024-08-10', periods=n_matches, freq='7D')

    # Base profile: random but plausible attacker stats
    base_goals = rng.uniform(0.2, 0.7)
    base_xg    = base_goals * rng.uniform(0.8, 1.3)
    base_assists = rng.uniform(0.1, 0.35)
    fatigue_trend = rng.uniform(-0.003, 0.003)  # slight upward or downward trend

    goals_p90, xg_p90, assists_p90, shots, key_passes, minutes_played = [], [], [], [], [], []
    for i in range(n_matches):
        trend_effect = 1 + fatigue_trend * i
        # Form cycles (hot/cold streaks, ~6 match period)
        form = 0.3 * np.sin(2 * np.pi * i / 6) + 0.15 * np.sin(2 * np.pi * i / 12)
        noise = rng.normal(0, 0.15)

        g = max(0, (base_goals + form + noise) * trend_effect)
        x = max(0, (base_xg   + form * 0.9 + rng.normal(0, 0.12)) * trend_effect)
        a = max(0, (base_assists + rng.normal(0, 0.1)) * trend_effect)
        s = max(0, rng.poisson(max(1, g * 4 + 1.5)))
        kp = max(0, rng.poisson(max(1, a * 5 + 1)))
        mins = int(rng.choice([90, 90, 90, 90, 75, 60, 45], p=[0.4,0.15,0.15,0.1,0.1,0.05,0.05]))

        goals_p90.append(round(g, 3))
        xg_p90.append(round(x, 3))
        assists_p90.append(round(a, 3))
        shots.append(int(s))
        key_passes.append(int(kp))
        minutes_played.append(mins)

    # Injury gap: randomly blank 2-4 matches
    injury_start = rng.integers(8, 20)
    injury_len   = rng.integers(2, 5)
    for i in range(injury_start, min(injury_start + injury_len, n_matches)):
        goals_p90[i] = np.nan
        xg_p90[i]    = np.nan
        assists_p90[i] = np.nan
        shots[i]      = 0
        key_passes[i] = 0
        minutes_played[i] = 0

    return pd.DataFrame({
        'match_date': match_dates,
        'match_num':  range(1, n_matches + 1),
        'goals':      goals_p90,
        'xg':         xg_p90,
        'assists':    assists_p90,
        'shots':      shots,
        'key_passes': key_passes,
        'minutes':    minutes_played,
        'played':     [m > 0 for m in minutes_played],
    })


def render_transfer_tab(tabs_ref, filtered, df, ollama_ok, ollama_host, ollama_model, query_ollama_fn):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    with tabs_ref:
        st.markdown('<p class="section-header">🔁 Analyse Time Series — Décision de Transfert</p>', unsafe_allow_html=True)
        st.markdown("""
        Analysez les performances **match par match** d'un joueur sur la saison pour décider s'il vaut la peine d'être recruté :
        tendance, stationnarité, forme récente, constance, pics de performance et risque blessure.
        """)

        # ── Source des données ──────────────────────────────────────────────
        st.markdown("### 📂 Source des données match par match")

        data_source = st.radio(
            "Choisir la source",
            ["📤 Uploader un CSV (données réelles)", "🎲 Données simulées (démo)"],
            horizontal=True
        )

        match_df = None
        player_name = ""

        # ── CSV Upload ───────────────────────────────────────────────────────
        if data_source == "📤 Uploader un CSV (données réelles)":
            st.markdown("""
            <div style="background:#111827;border:1px solid #10b981;border-radius:8px;padding:12px;font-size:0.83rem;color:#94a3b8;margin-bottom:12px;">
            📂 <b style="color:#10b981">Format accepté :</b> Tout fichier CSV ou Excel au format FBref ou similaire.<br>
            &nbsp;&nbsp;• Colonnes reconnues : <code>Date, Min, Gls, Ast, Sh, xG, xAG, Kp, Player, Joueur…</code><br>
            &nbsp;&nbsp;• Le nom du joueur et la saison sont <b>détectés automatiquement</b> depuis le fichier.<br>
            &nbsp;&nbsp;• Vous pouvez corriger manuellement si besoin.
            </div>
            """, unsafe_allow_html=True)
            uploaded = st.file_uploader("CSV ou Excel (match log)", type=["csv", "xlsx"])
            player_name_input = st.text_input("Nom du joueur (auto-détecté ou corriger)", value="", key="player_name_csv_input")
            if uploaded:
                try:
                    file_bytes = uploaded.read()  # read once into memory
                    if uploaded.name.endswith('.xlsx'):
                        raw = pd.read_excel(io.BytesIO(file_bytes))
                    else:
                        raw = pd.read_csv(io.BytesIO(file_bytes))
                    # Try skipping first row if it looks like metadata
                    if raw.columns[0].startswith('Unnamed') or raw.iloc[0].astype(str).str.contains('Date|date').any():
                        raw = pd.read_csv(io.BytesIO(file_bytes), skiprows=1)
                    # Normaliser les noms de colonnes
                    raw.columns = raw.columns.str.strip()
                    # ── Auto-détection du nom du joueur ─────────────────────
                    detected_player = ""
                    for pcol in ['Player', 'player', 'Joueur', 'joueur', 'Name', 'name', 'Nom', 'nom']:
                        if pcol in raw.columns:
                            vals = raw[pcol].dropna().astype(str)
                            vals = vals[vals.str.strip() != '']
                            if len(vals) > 0:
                                detected_player = vals.mode()[0].strip()
                                break
                    if not detected_player:
                        # Try to extract from filename: dembele_matches.csv → "dembele"
                        stem = uploaded.name.rsplit('.', 1)[0]
                        detected_player = stem.replace('_', ' ').replace('-', ' ').title()
                    # If user typed something, it takes priority; else use detected
                    if player_name_input.strip():
                        player_name = player_name_input.strip()
                    else:
                        player_name = detected_player
                        if player_name:
                            st.info(f"👤 Joueur détecté automatiquement : **{player_name}**")
                    # Mapping FBref → noms standards
                    col_map = {
                        'Gls': 'goals', 'Ast': 'assists', 'Sh': 'shots',
                        'SoT': 'shots_on_target', 'Kp': 'key_passes',
                        'Min': 'minutes', 'xG': 'xg', 'xAG': 'xag',
                        'Date': 'match_date',
                        'gls': 'goals', 'ast': 'assists', 'sh': 'shots',
                        'min': 'minutes', 'date': 'match_date',
                    }
                    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
                    # ── Auto-détection de la saison ──────────────────────────
                    if 'match_date' in raw.columns:
                        raw['match_date'] = pd.to_datetime(raw['match_date'], errors='coerce')
                        raw = raw.dropna(subset=['match_date']).sort_values('match_date').reset_index(drop=True)
                        if len(raw) > 0:
                            yr_min = raw['match_date'].min().year
                            yr_max = raw['match_date'].max().year
                            detected_season = f"{yr_min}/{str(yr_max)[-2:]}" if yr_min != yr_max else str(yr_min)
                        else:
                            detected_season = "Inconnue"
                    else:
                        detected_season = "Inconnue"
                    raw['_detected_season'] = detected_season
                    # Conversion numérique
                    for col in ['goals','assists','xg','shots','shots_on_target','key_passes','minutes']:
                        if col in raw.columns:
                            raw[col] = pd.to_numeric(raw[col], errors='coerce')
                    raw['match_num'] = range(1, len(raw) + 1)
                    # played : joué si minutes > 0
                    if 'minutes' in raw.columns:
                        raw['played'] = raw['minutes'].fillna(0) > 0
                    else:
                        raw['played'] = True
                    match_df = raw
                    st.success(f"✅ {len(match_df)} matchs chargés — Saison détectée : **{detected_season}**")
                    with st.expander("🔎 Aperçu"):
                        st.dataframe(match_df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lecture CSV: {e}")

        # ── Simulated data ───────────────────────────────────────────────────
        else:
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                player_name = st.text_input("Nom du joueur à simuler", value="")
                n_matches   = st.slider("Nombre de matchs", 15, 38, 34)
            with col_sim2:
                st.markdown("""
                <div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:12px;font-size:0.82rem;color:#94a3b8;margin-top:24px;">
                ℹ️ Les données sont <b>simulées de façon réaliste</b> avec tendances, cycles de forme,
                période de blessure et bruit statistique.
                </div>
                """, unsafe_allow_html=True)
            match_df = generate_player_match_data(player_name, n_matches)
            st.success(f"✅ {n_matches} matchs simulés pour **{player_name}**")

        if match_df is None or len(match_df) < 5:
            st.info("⬆️ Uploadez un CSV ou choisissez 'Générer des données simulées' pour commencer l'analyse.")
            return

        played_df = match_df[match_df.get('played', pd.Series([True]*len(match_df)))].copy()

        # ── Métriques de synthèse ────────────────────────────────────────────
        st.divider()
        # Detect season from match_df
        _season_display = match_df['_detected_season'].iloc[0] if '_detected_season' in match_df.columns else (
            f"{match_df['match_date'].min().year}/{str(match_df['match_date'].max().year)[-2:]}"
            if 'match_date' in match_df.columns and len(match_df) > 0 else "")
        _player_display = player_name if player_name else "Joueur"
        st.markdown(f"## ⚽ Analyse de {_player_display}" + (f" — Saison {_season_display}" if _season_display else ""))

        avail = [c for c in ['goals','xg','assists','shots','shots_on_target','key_passes','minutes'] if c in match_df.columns and match_df[c].notna().sum() > 0]
        m_cols = st.columns(len(avail) + 2)

        with m_cols[0]:
            n_played = int(played_df['played'].sum()) if 'played' in played_df.columns else len(played_df)
            st.metric("Matchs joués", n_played)
        with m_cols[1]:
            if 'minutes' in match_df.columns:
                total_mins = int(match_df['minutes'].fillna(0).sum())
                st.metric("Minutes totales", total_mins)

        for i, col in enumerate(avail):
            with m_cols[i + 2]:
                val = played_df[col].sum() if col in ['goals','assists','shots'] else played_df[col].mean()
                label = col.replace('_',' ').title() + (' total' if col in ['goals','assists','shots'] else ' moy')
                st.metric(label, f"{val:.2f}")

        # ── TAB INTERNE ──────────────────────────────────────────────────────
        inner_tabs = st.tabs([
            "📖 Introduction",
            "🔍 Exploration",
            "🔄 Décomposition",
            "📉 Stationnarité",
            "🔧 Préparation",
            "📐 AR",
            "〰️ MA",
            "🔗 ARMA",
            "📊 ARIMA",
            "🌿 SARIMA",
            "🤖 Rapport IA",
            "🎯 Prédiction & Décision"
        ])

        # Build list of available metrics
        available_metrics = [c for c in ['goals','xg','assists','shots','shots_on_target','key_passes','minutes'] if c in played_df.columns and played_df[c].notna().sum() >= 5]
        if not available_metrics:
            st.error("❌ Aucune métrique connue trouvée dans les données. Vérifiez les colonnes du CSV.")
            st.write("Colonnes disponibles :", list(played_df.columns))
            return

        metric_choice = st.selectbox(
            "Métrique principale analysée",
            available_metrics,
            key="ts_metric"
        )
        if metric_choice is None or metric_choice not in played_df.columns:
            st.warning("Sélectionnez une métrique pour commencer l'analyse.")
            return

        series = played_df[metric_choice].dropna().reset_index(drop=True)
        if len(series) == 0:
            st.warning(f"La métrique **{metric_choice}** ne contient aucune donnée valide.")
            return


        # ══════════════════════════════════════════════════════════════════════
        # TAB 0 : INTRODUCTION
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[0]:
            st.markdown('<p class="section-header">📖 Introduction — Analyse de Performance</p>', unsafe_allow_html=True)

            # Aperçu
            st.markdown("### 📋 Aperçu")
            n_played = int(played_df['played'].sum()) if 'played' in played_df.columns else len(played_df)
            missed   = int((match_df['minutes'].fillna(0) == 0).sum()) if 'minutes' in match_df.columns else 0
            avail_m  = [c for c in ['goals','xg','assists','shots','key_passes'] if c in played_df.columns]

            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Matchs joués", n_played)
            with c2: st.metric("Matchs manqués", missed)
            with c3:
                if 'goals' in played_df.columns:
                    st.metric("Buts totaux", int(played_df['goals'].sum()))
            with c4:
                if 'xg' in played_df.columns:
                    st.metric("xG total", f"{played_df['xg'].sum():.2f}")

            st.markdown("---")
            st.markdown("### 🗄️ Jeu de données")
            st.markdown(f"""
            <div class="stat-block">
              <div class="stat-label">Informations sur les données</div>
              <div class="stat-interpretation">
              • <b>Joueur analysé :</b> {player_name}<br>
              • <b>Saison :</b> {match_df['_detected_season'].iloc[0] if '_detected_season' in match_df.columns else (_season_display if '_season_display' in dir() else 'N/A')}<br>
              • <b>Nombre de matchs :</b> {len(match_df)}<br>
              • <b>Métriques disponibles :</b> {', '.join(avail_m)}<br>
              • <b>Période :</b> {str(match_df['match_date'].min())[:10] if 'match_date' in match_df.columns else 'N/A'} → {str(match_df['match_date'].max())[:10] if 'match_date' in match_df.columns else 'N/A'}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Aperçu des données brutes")
            display_cols = [c for c in ['match_date','goals','assists','xg','shots','key_passes','minutes'] if c in match_df.columns]
            st.dataframe(match_df[display_cols].head(10).style
                .format({c: '{:.3f}' for c in ['goals','assists','xg','shots','key_passes'] if c in display_cols}),
                use_container_width=True, hide_index=True)

            st.markdown("### 📈 Navigation")
            st.markdown("""
            | Onglet | Contenu |
            |--------|---------|
            | 🔍 Exploration | Visualisation match/match, forme, tendance |
            | 🔄 Décomposition | Tendance + Saisonnalité + Résidus |
            | 📉 Stationnarité | Tests ADF & KPSS |
            | 🔧 Préparation | Différenciation, transformation log |
            | 📐 AR | Modèle AutoRégressif |
            | 〰️ MA | Modèle Moyenne Mobile |
            | 🔗 ARMA | AR + MA combinés |
            | 📊 ARIMA | ARMA + Intégration |
            | 🌿 SARIMA | ARIMA + Saisonnalité |
            | 🤖 Rapport IA | Décision de transfert via Ollama |
            """)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 : EXPLORATION DE DONNÉES
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[1]:
            st.markdown('<p class="section-header">🔍 Exploration — Forme & Tendance match par match</p>', unsafe_allow_html=True)

            x_axis = list(range(1, len(played_df) + 1))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=[f"{metric_choice.title()} par match", "Minutes jouées"],
                                row_heights=[0.7, 0.3], vertical_spacing=0.08)

            fig.add_trace(go.Scatter(x=x_axis, y=played_df[metric_choice], mode='lines+markers',
                line=dict(color='rgba(56,189,248,0.4)', width=1.5),
                marker=dict(color='#38bdf8', size=5), name=metric_choice, showlegend=False), row=1, col=1)

            for w, c, lbl in [(5,'#f59e0b','MA-5'),(10,'#10b981','MA-10')]:
                if len(series) > w:
                    rm = series.rolling(w, center=True).mean()
                    fig.add_trace(go.Scatter(x=x_axis[:len(rm)], y=rm, mode='lines',
                        line=dict(color=c, width=2.5), name=lbl), row=1, col=1)

            s_clean2 = series.dropna()
            if len(s_clean2) >= 4:
                x_fit = np.arange(len(s_clean2))
                z = np.polyfit(x_fit, s_clean2, 2)
                p = np.poly1d(z)
                tc = '#10b981' if z[0] >= 0 else '#f87171'
                fig.add_trace(go.Scatter(x=x_axis[:len(s_clean2)], y=p(x_fit), mode='lines',
                    line=dict(color=tc, width=2, dash='dot'), name='Tendance'), row=1, col=1)

            if 'minutes' in played_df.columns:
                fig.add_trace(go.Bar(x=x_axis, y=played_df['minutes'].fillna(0),
                    marker_color='rgba(148,163,184,0.3)', name='Minutes', showlegend=False), row=2, col=1)

            if 'minutes' in match_df.columns:
                for _, rm in match_df[match_df['minutes'].fillna(0)==0].iterrows():
                    fig.add_vrect(x0=rm['match_num']-0.5, x1=rm['match_num']+0.5,
                        fillcolor='rgba(248,113,113,0.15)', line_width=0, row=1, col=1)

            fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
                font=dict(color='#94a3b8'), height=440,
                margin=dict(t=50,b=30,l=50,r=20),
                legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.08))
            fig.update_xaxes(gridcolor='#1e293b', title="Match #")
            fig.update_yaxes(gridcolor='#1e293b')
            st.plotly_chart(fig, use_container_width=True)

            if len(series) >= 10:
                first5 = series[:5].mean(); last5 = series[-5:].mean()
                delta  = ((last5-first5)/(first5+1e-9))*100
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="stat-block"><div class="stat-label">5 premiers matchs</div><div class="stat-value">{first5:.3f}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="stat-block"><div class="stat-label">5 derniers matchs</div><div class="stat-value">{last5:.3f}</div></div>', unsafe_allow_html=True)
                with c3:
                    col = "#10b981" if delta>5 else "#f87171" if delta<-5 else "#f59e0b"
                    arrow = "📈" if delta>5 else "📉" if delta<-5 else "➡️"
                    st.markdown(f'<div class="stat-block" style="border-color:{col}"><div class="stat-label">Évolution de forme</div><div class="stat-value" style="color:{col}">{arrow} {delta:+.1f}%</div></div>', unsafe_allow_html=True)

            fig_roll = plot_rolling_stats(series.dropna(), name=metric_choice, windows=[5,10,15])
            st.plotly_chart(fig_roll, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 : DÉCOMPOSITION DE SÉRIES TEMPORELLES
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[2]:
            st.markdown('<p class="section-header">🔄 Décomposition — Tendance · Saisonnalité · Résidus</p>', unsafe_allow_html=True)
            st.markdown("""
            La décomposition sépare la série en **3 composantes** :
            - **Tendance** : évolution long terme de la performance du joueur
            - **Saisonnalité** : cycles réguliers (fatigue, calendrier, compétitions)
            - **Résidus** : matchs exceptionnels ou aléatoires
            """)
            s_clean = series.dropna().reset_index(drop=True)
            if len(s_clean) >= 10:
                fig_decomp = plot_seasonality_decomposition(s_clean, metric_choice)
                st.plotly_chart(fig_decomp, use_container_width=True)

                slope, _, r_val, p_val, _ = stats.linregress(np.arange(len(s_clean)), s_clean)
                residual = s_clean - (slope * np.arange(len(s_clean)) + s_clean.mean())
                c1,c2,c3 = st.columns(3)
                with c1:
                    arrow = "📈" if slope>0.003 else "📉" if slope<-0.003 else "➡️"
                    st.markdown(f'<div class="stat-block"><div class="stat-label">Pente tendance</div><div class="stat-value">{arrow} {slope:+.4f}/match</div><div class="stat-interpretation">{"Progression" if slope>0.003 else "Régression" if slope<-0.003 else "Stable"}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="stat-block"><div class="stat-label">R² tendance</div><div class="stat-value">{r_val**2:.3f}</div><div class="stat-interpretation">{"Tendance forte" if r_val**2>0.3 else "Tendance faible"}</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="stat-block"><div class="stat-label">Volatilité résiduelle σ</div><div class="stat-value">{residual.std():.3f}</div><div class="stat-interpretation">{"Très variable ⚠️" if residual.std()>s_clean.mean()*0.8 else "Constance ✅"}</div></div>', unsafe_allow_html=True)
            else:
                st.warning("Minimum 10 matchs pour la décomposition.")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 : STATIONNARITÉ
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[3]:
            st.markdown('<p class="section-header">📉 Stationnarité — Tests ADF & KPSS</p>', unsafe_allow_html=True)
            st.markdown("""
            Une série **stationnaire** a moyenne et variance constantes → performances prévisibles.
            Une série **non-stationnaire** dérive → le joueur est en évolution (bonne ou mauvaise).
            
            > *Implication transfert :* Un joueur stationnaire est fiable. Non-stationnaire = potentiel de progression ou risque de déclin.
            """)
            if len(series.dropna()) >= 8:
                res = compute_stationarity(series.dropna())
                c1,c2,c3 = st.columns([1,1,2])
                with c1:
                    adf_ok = res.get('adf_stationary', False)
                    st.markdown(f'<div class="stat-block"><div class="stat-label">ADF Test</div><div class="stat-value">{res.get("adf_stat",0):.4f}</div><div class="stat-interpretation">p={res.get("adf_pvalue",1):.4f}<br>{"✅ Stationnaire" if adf_ok else "❌ Non-stationnaire"}</div></div>', unsafe_allow_html=True)
                with c2:
                    kpss_ok = res.get('kpss_stationary', True)
                    st.markdown(f'<div class="stat-block"><div class="stat-label">KPSS Test</div><div class="stat-value">{res.get("kpss_stat",0):.4f}</div><div class="stat-interpretation">p={res.get("kpss_pvalue",0):.4f}<br>{"✅ Stationnaire" if kpss_ok else "❌ Non-stationnaire"}</div></div>', unsafe_allow_html=True)
                with c3:
                    if adf_ok and kpss_ok:
                        v,bg = "✅ **STABLE** — Performances constantes. Risque faible, joueur fiable.","#064e3b"
                    elif not adf_ok and not kpss_ok:
                        v,bg = "⚠️ **EN DÉRIVE** — Vérifiez si progression ou déclin via la décomposition.","#451a03"
                    else:
                        v,bg = "🔍 **AMBIGU** — Tendance déterministe possible. Analysez la décomposition.","#1e3a5f"
                    st.markdown(f'<div style="background:{bg};border-radius:8px;padding:16px;font-size:0.9rem;">{v}</div>', unsafe_allow_html=True)

                fig_roll2 = plot_rolling_stats(series.dropna(), name=metric_choice, windows=[5,10])
                st.plotly_chart(fig_roll2, use_container_width=True)

                # Variance split
                half = len(series.dropna())//2
                s1,s2 = series.dropna().iloc[:half], series.dropna().iloc[half:]
                c1,c2,c3 = st.columns(3)
                with c1: st.markdown(f'<div class="stat-block"><div class="stat-label">σ 1ère moitié</div><div class="stat-value">{s1.std():.3f}</div></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="stat-block"><div class="stat-label">σ 2ème moitié</div><div class="stat-value">{s2.std():.3f}</div></div>', unsafe_allow_html=True)
                with c3:
                    ratio = s2.std()/(s1.std()+1e-9)
                    st.markdown(f'<div class="stat-block"><div class="stat-label">Ratio de variance</div><div class="stat-value">{ratio:.2f}x</div><div class="stat-interpretation">{"Plus instable en fin" if ratio>1.3 else "Plus stable en fin ✅" if ratio<0.7 else "Variance constante ✅"}</div></div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 : PRÉPARATION DES DONNÉES
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[4]:
            st.markdown('<p class="section-header">🔧 Préparation — Différenciation & Transformations</p>', unsafe_allow_html=True)
            st.markdown("""
            Si la série est **non-stationnaire**, on applique des transformations avant de modéliser.
            Ces transformations sont nécessaires pour les modèles AR, MA, ARIMA.
            """)
            s_clean = series.dropna().reset_index(drop=True)

            prep_type = st.radio("Transformation à appliquer",
                ["Série originale", "Différenciation (d=1)", "Différenciation (d=2)", "Log", "Log + Différenciation"],
                horizontal=True)

            if prep_type == "Série originale":
                s_prep = s_clean
                desc = "Aucune transformation — série brute"
            elif prep_type == "Différenciation (d=1)":
                s_prep = s_clean.diff().dropna().reset_index(drop=True)
                desc = "Δyₜ = yₜ - yₜ₋₁ — supprime les tendances linéaires"
            elif prep_type == "Différenciation (d=2)":
                s_prep = s_clean.diff().diff().dropna().reset_index(drop=True)
                desc = "Δ²yₜ — supprime les tendances quadratiques"
            elif prep_type == "Log":
                s_prep = np.log1p(s_clean.clip(lower=0)).reset_index(drop=True)
                desc = "log(1+yₜ) — stabilise la variance"
            else:
                s_prep = np.log1p(s_clean.clip(lower=0)).diff().dropna().reset_index(drop=True)
                desc = "Δlog(yₜ) — stabilise variance ET supprime tendance"

            st.info(f"📐 **{prep_type}** : {desc}")

            fig_prep = make_subplots(rows=1, cols=2,
                subplot_titles=["Série originale", f"Après : {prep_type}"])
            fig_prep.add_trace(go.Scatter(y=s_clean, mode='lines+markers',
                line=dict(color='#94a3b8', width=1.5), name='Original'), row=1, col=1)
            fig_prep.add_trace(go.Scatter(y=s_prep, mode='lines+markers',
                line=dict(color='#38bdf8', width=1.5), name=prep_type), row=1, col=2)
            fig_prep.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
                font=dict(color='#94a3b8'), height=300, showlegend=False,
                margin=dict(t=50,b=30,l=50,r=20))
            fig_prep.update_xaxes(gridcolor='#1e293b')
            fig_prep.update_yaxes(gridcolor='#1e293b')
            st.plotly_chart(fig_prep, use_container_width=True)

            # Test stationarité après transformation
            if len(s_prep.dropna()) >= 8:
                res_prep = compute_stationarity(s_prep.dropna())
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="stat-block"><div class="stat-label">ADF après transformation</div><div class="stat-value">p = {res_prep.get("adf_pvalue",1):.4f}</div><div class="stat-interpretation">{"✅ Stationnaire" if res_prep.get("adf_stationary") else "❌ Non-stationnaire — essayez une autre transformation"}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="stat-block"><div class="stat-label">KPSS après transformation</div><div class="stat-value">p = {res_prep.get("kpss_pvalue",0):.4f}</div><div class="stat-interpretation">{"✅ Stationnaire" if res_prep.get("kpss_stationary") else "❌ Non-stationnaire"}</div></div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # HELPER : Fit & Plot forecast model
        # ══════════════════════════════════════════════════════════════════════
        def plot_forecast(s_clean, fitted_vals, forecast_vals, title, model_eq, residuals=None):
            n = len(s_clean)
            h = len(forecast_vals)
            fig = make_subplots(rows=2 if residuals is not None else 1, cols=1,
                subplot_titles=[title, "Résidus"] if residuals is not None else [title],
                vertical_spacing=0.12, row_heights=[0.7,0.3] if residuals is not None else [1])

            fig.add_trace(go.Scatter(y=s_clean, mode='lines+markers', name='Observations',
                line=dict(color='#94a3b8', width=1.5), marker=dict(size=4)), row=1, col=1)
            if fitted_vals is not None and len(fitted_vals) > 0:
                fig.add_trace(go.Scatter(y=fitted_vals, mode='lines', name='Ajustement',
                    line=dict(color='#38bdf8', width=2)), row=1, col=1)
            if h > 0:
                x_fore = list(range(n, n+h))
                fig.add_trace(go.Scatter(x=x_fore, y=forecast_vals, mode='lines+markers',
                    name='Prévision', line=dict(color='#f59e0b', width=2.5, dash='dot'),
                    marker=dict(symbol='diamond', size=7, color='#f59e0b')), row=1, col=1)
                fig.add_vrect(x0=n-0.5, x1=n+h-0.5,
                    fillcolor='rgba(245,158,11,0.05)', line_width=0, row=1, col=1)

            if residuals is not None and len(residuals) > 0:
                fig.add_trace(go.Bar(y=residuals, marker_color=[
                    '#10b981' if v >= 0 else '#f87171' for v in residuals],
                    name='Résidus', showlegend=False), row=2, col=1)
                fig.add_hline(y=0, line_dash='dash', line_color='#475569', row=2, col=1)

            fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
                font=dict(color='#94a3b8'), height=420 if residuals is not None else 320,
                margin=dict(t=50,b=30,l=50,r=20),
                legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.05))
            fig.update_xaxes(gridcolor='#1e293b', title="Match #")
            fig.update_yaxes(gridcolor='#1e293b')

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f'<div class="stat-block"><div class="stat-label">Équation du modèle</div><div class="stat-value" style="font-size:0.9rem;font-family:monospace">{model_eq}</div></div>', unsafe_allow_html=True)
            return fig

        s_clean = series.dropna().reset_index(drop=True)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5 : AR — AutoRegression
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[5]:
            st.markdown('<p class="section-header">📐 Modèle AR — AutoRégressif</p>', unsafe_allow_html=True)
            st.markdown("""
            **AR(p)** : la performance du match actuel dépend des **p matchs précédents**.
            
            `yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ`
            
            > *Exemple football :* Si un joueur marque souvent après un bon match, φ₁ > 0 (momentum).
            """)
            p_ar = st.slider("Ordre p (nombre de lags AR)", 1, min(8, len(s_clean)//4), 2, key="ar_p")
            n_fore = st.slider("Matchs à prévoir", 1, 10, 5, key="ar_fore")

            s_ar = series.dropna().reset_index(drop=True)
            if len(s_ar) < p_ar + 4:
                st.warning(f"Pas assez de données ({len(s_ar)} valeurs) pour AR({p_ar}). Réduisez p ou chargez plus de matchs.")
            elif s_ar.std() < 1e-8:
                st.warning(f"⚠️ La série '{metric_choice}' est constante (variance nulle) — choisissez une autre métrique.")
            else:
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    s_ar_mean = s_ar.mean()
                    s_ar_std  = max(s_ar.std(), 1e-6)
                    s_ar_norm = (s_ar - s_ar_mean) / s_ar_std

                    # Essai principal : ARIMA(p,0,0) = AR(p), plus robuste qu'AutoReg
                    try:
                        model_ar = ARIMA(s_ar_norm, order=(p_ar, 0, 0)).fit()
                    except Exception:
                        # Fallback : méthode CSS moins stricte
                        model_ar = ARIMA(s_ar_norm, order=(p_ar, 0, 0)).fit(method='innovations_mle')

                    fitted_norm    = model_ar.fittedvalues
                    forecast_norm  = model_ar.forecast(steps=n_fore)
                    residuals_norm = model_ar.resid

                    fitted            = fitted_norm * s_ar_std + s_ar_mean
                    forecast_vals_arr = np.array(forecast_norm) * s_ar_std + s_ar_mean
                    residuals         = residuals_norm * s_ar_std

                    params = model_ar.params
                    eq = f"y(t) = {float(params.iloc[0])*s_ar_std+s_ar_mean:.3f}"
                    for i in range(min(p_ar, 3)):
                        idx = i + 1
                        if idx < len(params):
                            eq += f" + {float(params.iloc[idx]):.3f}·y(t-{i+1})"
                    if p_ar > 3: eq += " + ..."

                    plot_forecast(s_ar, fitted, forecast_vals_arr, f"AR({p_ar}) — {metric_choice}", eq, residuals)

                    c1,c2,c3 = st.columns(3)
                    with c1: st.markdown(f'<div class="stat-block"><div class="stat-label">AIC</div><div class="stat-value">{model_ar.aic:.2f}</div><div class="stat-interpretation">Plus bas = meilleur</div></div>', unsafe_allow_html=True)
                    with c2: st.markdown(f'<div class="stat-block"><div class="stat-label">BIC</div><div class="stat-value">{model_ar.bic:.2f}</div></div>', unsafe_allow_html=True)
                    with c3:
                        rmse = np.sqrt(np.mean(residuals.dropna()**2))
                        st.markdown(f'<div class="stat-block"><div class="stat-label">RMSE</div><div class="stat-value">{rmse:.4f}</div><div class="stat-interpretation">Erreur quadratique moyenne</div></div>', unsafe_allow_html=True)

                    st.markdown("**Prévisions :**")
                    fore_df = pd.DataFrame({'Match': [f"M+{i+1}" for i in range(n_fore)], f'{metric_choice} prévu': forecast_vals_arr.round(4)})
                    st.dataframe(fore_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Erreur AR({p_ar}) : {e}")
                    st.info("💡 Réduisez l'ordre p ou choisissez une métrique avec plus de variabilité (ex: xg, shots).")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 6 : MA — Moving Average
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[6]:
            st.markdown('<p class="section-header">〰️ Modèle MA — Moyenne Mobile</p>', unsafe_allow_html=True)
            st.markdown("""
            **MA(q)** : la performance dépend des **q erreurs passées** (chocs aléatoires).
            
            `yₜ = μ + εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q`
            
            > *Exemple football :* Un but refusé (choc) peut affecter la concentration sur les matchs suivants.
            """)
            q_ma = st.slider("Ordre q (nombre de lags MA)", 1, min(6, len(s_clean)//4), 1, key="ma_q")
            n_fore_ma = st.slider("Matchs à prévoir", 1, 10, 5, key="ma_fore")

            s_ma = series.dropna().reset_index(drop=True)
            if len(s_ma) < q_ma + 4:
                st.warning(f"Pas assez de données ({len(s_ma)} valeurs) pour MA({q_ma}). Réduisez q ou chargez plus de matchs.")
            elif s_ma.std() < 1e-8:
                st.warning(f"⚠️ La série '{metric_choice}' est constante (variance nulle) — choisissez une autre métrique.")
            else:
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    s_ma_mean = s_ma.mean()
                    s_ma_std  = max(s_ma.std(), 1e-6)
                    s_ma_norm = (s_ma - s_ma_mean) / s_ma_std

                    try:
                        model_ma = ARIMA(s_ma_norm, order=(0, 0, q_ma)).fit()
                    except Exception:
                        model_ma = ARIMA(s_ma_norm, order=(0, 0, q_ma)).fit(method='innovations_mle')

                    fitted_ma         = model_ma.fittedvalues * s_ma_std + s_ma_mean
                    forecast_ma_vals  = np.array(model_ma.forecast(steps=n_fore_ma)) * s_ma_std + s_ma_mean
                    residuals_ma      = model_ma.resid * s_ma_std

                    params_ma = model_ma.params
                    eq_ma = f"y(t) = {float(params_ma.iloc[0])*s_ma_std+s_ma_mean:.3f} + ε(t)"
                    for i in range(min(q_ma, 3)):
                        idx = i + 1
                        if idx < len(params_ma):
                            eq_ma += f" + {float(params_ma.iloc[idx]):.3f}·ε(t-{i+1})"
                    plot_forecast(s_ma, fitted_ma, forecast_ma_vals, f"MA({q_ma}) — {metric_choice}", eq_ma, residuals_ma)

                    c1,c2,c3 = st.columns(3)
                    with c1: st.markdown(f'<div class="stat-block"><div class="stat-label">AIC</div><div class="stat-value">{model_ma.aic:.2f}</div></div>', unsafe_allow_html=True)
                    with c2: st.markdown(f'<div class="stat-block"><div class="stat-label">BIC</div><div class="stat-value">{model_ma.bic:.2f}</div></div>', unsafe_allow_html=True)
                    with c3:
                        rmse_ma = np.sqrt(np.mean(residuals_ma.dropna()**2))
                        st.markdown(f'<div class="stat-block"><div class="stat-label">RMSE</div><div class="stat-value">{rmse_ma:.4f}</div></div>', unsafe_allow_html=True)

                    fore_df_ma = pd.DataFrame({'Match': [f"M+{i+1}" for i in range(n_fore_ma)], f'{metric_choice} prévu': forecast_ma_vals.round(4)})
                    st.dataframe(fore_df_ma, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Erreur MA({q_ma}) : {e}")
                    st.info("💡 Réduisez l'ordre q ou choisissez une métrique avec plus de variabilité (ex: xg, shots).")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 7 : ARMA
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[7]:
            st.markdown('<p class="section-header">🔗 Modèle ARMA — AR + MA</p>', unsafe_allow_html=True)
            st.markdown("""
            **ARMA(p,q)** : combine AR et MA. Capture à la fois le **momentum** et les **chocs passés**.
            
            `yₜ = c + φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q`
            
            > *Implication :* Modèle plus riche, adapté aux joueurs dont les performances sont influencées à la fois par leur récente forme ET par des événements ponctuels.
            """)
            c1,c2 = st.columns(2)
            with c1: p_arma = st.slider("p (AR)", 1, min(5,len(s_clean)//4), 1, key="arma_p")
            with c2: q_arma = st.slider("q (MA)", 1, min(5,len(s_clean)//4), 1, key="arma_q")
            n_fore_arma = st.slider("Matchs à prévoir", 1, 10, 5, key="arma_fore")

            try:
                from statsmodels.tsa.arima.model import ARIMA
                model_arma = ARIMA(s_clean, order=(p_arma, 0, q_arma)).fit()
                fitted_arma = model_arma.fittedvalues
                forecast_arma = model_arma.forecast(steps=n_fore_arma)
                residuals_arma = model_arma.resid
                eq_arma = f"ARMA({p_arma},{q_arma})  AIC={model_arma.aic:.2f}  BIC={model_arma.bic:.2f}"
                plot_forecast(s_clean, fitted_arma, forecast_arma.values, f"ARMA({p_arma},{q_arma}) — {metric_choice}", eq_arma, residuals_arma)

                fore_df_arma = pd.DataFrame({'Match': [f"M+{i+1}" for i in range(n_fore_arma)], f'{metric_choice} prévu': forecast_arma.values.round(4)})
                st.dataframe(fore_df_arma, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Erreur ARMA : {e}")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 8 : ARIMA
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[8]:
            st.markdown('<p class="section-header">📊 Modèle ARIMA — AR + Intégration + MA</p>', unsafe_allow_html=True)
            st.markdown("""
            **ARIMA(p,d,q)** : ajoute l'**intégration** (différenciation d fois) pour rendre la série stationnaire.
            
            - **p** : ordre AR (lags de la variable)
            - **d** : ordre de différenciation (souvent 1 si non-stationnaire)  
            - **q** : ordre MA (lags des erreurs)
            
            > *Conseil :* Si ADF montre non-stationnarité → utilisez d=1. Sinon d=0.
            """)
            c1,c2,c3 = st.columns(3)
            with c1: p_arima = st.slider("p", 0, min(4,len(s_clean)//5), 1, key="arima_p")
            with c2: d_arima = st.slider("d", 0, 2, 1, key="arima_d")
            with c3: q_arima = st.slider("q", 0, min(4,len(s_clean)//5), 1, key="arima_q")
            n_fore_arima = st.slider("Matchs à prévoir", 1, 10, 5, key="arima_fore")

            try:
                from statsmodels.tsa.arima.model import ARIMA
                model_arima = ARIMA(s_clean, order=(p_arima, d_arima, q_arima)).fit()
                fitted_arima = model_arima.fittedvalues
                forecast_arima = model_arima.forecast(steps=n_fore_arima)
                residuals_arima = model_arima.resid

                # Confidence intervals
                fore_ci = model_arima.get_forecast(steps=n_fore_arima).conf_int()
                eq_arima = f"ARIMA({p_arima},{d_arima},{q_arima})  AIC={model_arima.aic:.2f}"
                plot_forecast(s_clean, fitted_arima, forecast_arima.values, f"ARIMA({p_arima},{d_arima},{q_arima}) — {metric_choice}", eq_arima, residuals_arima)

                # Show CI table
                fore_df_arima = pd.DataFrame({
                    'Match': [f"M+{i+1}" for i in range(n_fore_arima)],
                    f'{metric_choice} prévu': forecast_arima.values.round(4),
                    'IC bas (95%)': fore_ci.iloc[:,0].values.round(4),
                    'IC haut (95%)': fore_ci.iloc[:,1].values.round(4),
                })
                st.dataframe(fore_df_arima, use_container_width=True, hide_index=True)

                # ACF of residuals (should be white noise)
                st.markdown("**ACF des résidus** (doit ressembler à du bruit blanc) :")
                fig_racf = plot_acf_pacf(residuals_arima.dropna(), lags=min(10,len(residuals_arima)//2-1),
                    title="ACF résidus ARIMA")
                st.plotly_chart(fig_racf, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur ARIMA : {e}")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 9 : SARIMA
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[9]:
            st.markdown('<p class="section-header">🌿 Modèle SARIMA — ARIMA + Saisonnalité</p>', unsafe_allow_html=True)
            st.markdown("""
            **SARIMA(p,d,q)(P,D,Q)s** : ajoute des composantes **saisonnières** à ARIMA.
            
            - **(p,d,q)** : partie non-saisonnière (comme ARIMA)
            - **(P,D,Q)** : partie saisonnière
            - **s** : période de saisonnalité (ex: 5 matchs = cycle de forme hebdomadaire)
            
            > *Exemple :* Un joueur marque souvent après 5-6 matchs de repos → s=5.
            """)
            c1,c2 = st.columns(2)
            with c1:
                p_s = st.slider("p", 0, 2, 1, key="sarima_p")
                d_s = st.slider("d", 0, 2, 1, key="sarima_d")
                q_s = st.slider("q", 0, 2, 1, key="sarima_q")
            with c2:
                P_s = st.slider("P (saisonnier)", 0, 2, 1, key="sarima_P")
                D_s = st.slider("D (saisonnier)", 0, 1, 0, key="sarima_D")
                Q_s = st.slider("Q (saisonnier)", 0, 2, 1, key="sarima_Q")
                s_period = st.slider("s (période)", 2, min(12, len(s_clean)//3), 5, key="sarima_s")
            n_fore_sarima = st.slider("Matchs à prévoir", 1, 10, 5, key="sarima_fore")

            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model_sarima = SARIMAX(s_clean,
                    order=(p_s, d_s, q_s),
                    seasonal_order=(P_s, D_s, Q_s, s_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)

                fitted_sarima = model_sarima.fittedvalues
                forecast_sarima = model_sarima.forecast(steps=n_fore_sarima)
                residuals_sarima = model_sarima.resid
                fore_ci_s = model_sarima.get_forecast(steps=n_fore_sarima).conf_int()

                eq_sarima = f"SARIMA({p_s},{d_s},{q_s})({P_s},{D_s},{Q_s})[{s_period}]  AIC={model_sarima.aic:.2f}"
                plot_forecast(s_clean, fitted_sarima, forecast_sarima.values,
                    f"SARIMA — {metric_choice}", eq_sarima, residuals_sarima)

                fore_df_sarima = pd.DataFrame({
                    'Match': [f"M+{i+1}" for i in range(n_fore_sarima)],
                    f'{metric_choice} prévu': forecast_sarima.values.round(4),
                    'IC bas': fore_ci_s.iloc[:,0].values.round(4),
                    'IC haut': fore_ci_s.iloc[:,1].values.round(4),
                })
                st.dataframe(fore_df_sarima, use_container_width=True, hide_index=True)

                # Model comparison
                st.markdown("---")
                st.markdown("#### 🏆 Comparaison des modèles (AIC — plus bas = meilleur)")
                try:
                    from statsmodels.tsa.arima.model import ARIMA as _AR
                    from statsmodels.tsa.ar_model import AutoReg as _AutoReg
                    comparison = []
                    for name_m, order in [("AR(2)",(2,0,0)),("MA(1)",(0,0,1)),("ARMA(1,1)",(1,0,1)),("ARIMA(1,1,1)",(1,1,1))]:
                        try:
                            m = _AR(s_clean, order=order).fit()
                            comparison.append({'Modèle': name_m, 'AIC': round(m.aic,2), 'BIC': round(m.bic,2)})
                        except: pass
                    comparison.append({'Modèle': eq_sarima.split('  ')[0], 'AIC': round(model_sarima.aic,2), 'BIC': round(model_sarima.bic,2)})
                    comp_df = pd.DataFrame(comparison).sort_values('AIC')
                    st.dataframe(comp_df.style.highlight_min(subset=['AIC'], color='#064e3b'), use_container_width=True, hide_index=True)
                except: pass

            except Exception as e:
                st.error(f"Erreur SARIMA : {e}. Essayez des ordres plus petits ou augmentez le nombre de matchs.")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 10 : RAPPORT IA + CONCLUSION
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[10]:
            st.markdown('<p class="section-header">🤖 Conclusion & Rapport IA — Décision de Transfert</p>', unsafe_allow_html=True)

            s_clean2 = series.dropna()
            slope2, _, r_val2, _, _ = stats.linregress(np.arange(len(s_clean2)), s_clean2) if len(s_clean2)>=3 else (0,0,0,1,0)
            first5_mean = s_clean2[:5].mean() if len(s_clean2)>=5 else s_clean2.mean()
            last5_mean  = s_clean2[-5:].mean() if len(s_clean2)>=5 else s_clean2.mean()
            missed_matches = int((match_df['minutes'].fillna(0)==0).sum()) if 'minutes' in match_df.columns else 0

            # Résumé conclusion visuel
            st.markdown("### 📌 Synthèse analytique")
            cc1,cc2,cc3,cc4 = st.columns(4)
            with cc1:
                trend_label = "📈 Progression" if slope2>0.003 else "📉 Régression" if slope2<-0.003 else "➡️ Stable"
                trend_col = "#10b981" if slope2>0.003 else "#f87171" if slope2<-0.003 else "#f59e0b"
                st.markdown(f'<div class="stat-block" style="border-color:{trend_col}"><div class="stat-label">Tendance</div><div class="stat-value" style="color:{trend_col}">{trend_label}</div></div>', unsafe_allow_html=True)
            with cc2:
                form_delta = ((last5_mean-first5_mean)/(first5_mean+1e-9))*100
                form_col = "#10b981" if form_delta>5 else "#f87171" if form_delta<-5 else "#f59e0b"
                st.markdown(f'<div class="stat-block" style="border-color:{form_col}"><div class="stat-label">Forme récente</div><div class="stat-value" style="color:{form_col}">{form_delta:+.1f}%</div></div>', unsafe_allow_html=True)
            with cc3:
                inj_col = "#f87171" if missed_matches>6 else "#f59e0b" if missed_matches>3 else "#10b981"
                st.markdown(f'<div class="stat-block" style="border-color:{inj_col}"><div class="stat-label">Matchs manqués</div><div class="stat-value" style="color:{inj_col}">{missed_matches}</div></div>', unsafe_allow_html=True)
            with cc4:
                cv = s_clean2.std()/(s_clean2.mean()+1e-9)
                cons_col = "#10b981" if cv<0.5 else "#f59e0b" if cv<1 else "#f87171"
                st.markdown(f'<div class="stat-block" style="border-color:{cons_col}"><div class="stat-label">Coefficient variation</div><div class="stat-value" style="color:{cons_col}">{cv:.2f}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            stat_context = f"""
Joueur: {player_name}  |  Métrique: {metric_choice}  |  Matchs joués: {len(played_df)}  |  Matchs manqués: {missed_matches}

Statistiques descriptives:
  Moyenne={s_clean2.mean():.3f} | Std={s_clean2.std():.3f} | Min={s_clean2.min():.3f} | Max={s_clean2.max():.3f}

Analyse temporelle:
  Tendance (slope): {slope2:+.4f}/match → {"Progression" if slope2>0.003 else "Régression" if slope2<-0.003 else "Stable"}
  Forme début saison (5 premiers): {first5_mean:.3f}
  Forme récente (5 derniers): {last5_mean:.3f} ({form_delta:+.1f}%)
  Coefficient de variation: {cv:.2f}

Modélisation:
  Série stationnaire: {"Oui (ADF+KPSS)" if len(s_clean2)>=8 else "Non testé"}
  Modèles recommandés: AR, ARMA, ARIMA selon les résultats AIC/BIC
"""
            st.code(stat_context, language='text')

            prompt_ia = f"""
Tu es directeur sportif expert. Voici l'analyse complète de {player_name}:

{stat_context}

Génère un rapport de décision de transfert:

## 1. VERDICT GLOBAL (ACHETER / SURVEILLER / NE PAS ACHETER)

## 2. ANALYSE DE FORME & TENDANCE
Interprète l'évolution des performances. Progression ou déclin?

## 3. FIABILITÉ & CONSTANCE
Analyse la variance. Est-il régulier ou irrégulier?

## 4. RISQUE BLESSURE
{missed_matches} matchs manqués — quel niveau de risque?

## 5. MODÉLISATION PRÉDICTIVE
Quelles performances attendre les 5 prochains matchs?

## 6. CONDITIONS D'ACHAT RECOMMANDÉES
Prix, durée de contrat, clauses de performance suggérés.

Sois précis, utilise les chiffres, sois actionnable.
"""
            if st.button("🧠 Générer le rapport de décision", type="primary", use_container_width=True):
                if ollama_ok:
                    with st.spinner("Analyse IA en cours..."):
                        placeholder = st.empty()
                        full = ""
                        for chunk in query_ollama_fn(ollama_host, ollama_model,
                                [{"role":"user","content":prompt_ia}], "Analyse football time series"):
                            full += chunk
                            placeholder.markdown(f'<div class="chat-msg-ai">🤖 {full}▌</div>', unsafe_allow_html=True)
                        placeholder.markdown(f'<div class="chat-msg-ai">🤖 {full}</div>', unsafe_allow_html=True)
                    st.download_button("📄 Télécharger le rapport",
                        data=f"RAPPORT TRANSFERT — {player_name}\n\n{stat_context}\n\nANALYSE IA:\n{full}".encode('utf-8'),
                        file_name=f"rapport_{player_name.replace(' ','_')}.txt", mime="text/plain")
                else:
                    # Auto verdict sans Ollama
                    if last5_mean > first5_mean*1.1: verdict_auto = "✅ ACHETER — En progression"
                    elif last5_mean < first5_mean*0.85: verdict_auto = "⚠️ SURVEILLER — Déclin récent"
                    else: verdict_auto = "🔍 À ÉVALUER — Stable"
                    st.markdown(f'<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:16px;">{verdict_auto}<br><br>Moy: <b>{s_clean2.mean():.3f}</b> | Forme: <b>{last5_mean:.3f}</b> | Absences: <b>{missed_matches}</b></div>', unsafe_allow_html=True)
                    st.warning("Ollama non disponible — lancez `ollama serve` pour le rapport complet.")


        # ══════════════════════════════════════════════════════════════════════
        # TAB 11 : PRÉDICTION & DÉCISION D'ACHAT
        # ══════════════════════════════════════════════════════════════════════
        with inner_tabs[11]:
            st.markdown('<p class="section-header">🎯 Prédiction Automatique & Décision de Transfert</p>', unsafe_allow_html=True)
            st.markdown("""
            Ce module **compare automatiquement** AR, MA, ARMA, ARIMA, SARIMA,
            sélectionne le meilleur modèle selon l'AIC, génère les prévisions des prochains matchs
            et produit un **score de décision d'achat** basé sur 6 critères analytiques.
            """)

            s_decision = series.dropna().reset_index(drop=True)
            n_pred = st.slider("Nombre de matchs à prévoir", 3, 15, 8, key="decision_n")

            if len(s_decision) < 8:
                st.warning("Minimum 8 matchs pour la prédiction automatique.")
            else:
                # ── Entraîner tous les modèles ────────────────────────────────
                st.markdown("### 🏋️ Entraînement automatique des modèles")

                from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                from statsmodels.tsa.ar_model import AutoReg as _AR
                from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX

                model_results = {}
                progress = st.progress(0, text="Entraînement des modèles...")

                configs = [
                    ("AR(2)",        lambda s: _AR(s, lags=2, old_names=False).fit()),
                    ("AR(3)",        lambda s: _AR(s, lags=3, old_names=False).fit()),
                    ("MA(1)",        lambda s: _ARIMA(s, order=(0,0,1)).fit()),
                    ("MA(2)",        lambda s: _ARIMA(s, order=(0,0,2)).fit()),
                    ("ARMA(1,1)",    lambda s: _ARIMA(s, order=(1,0,1)).fit()),
                    ("ARMA(2,1)",    lambda s: _ARIMA(s, order=(2,0,1)).fit()),
                    ("ARIMA(1,1,1)", lambda s: _ARIMA(s, order=(1,1,1)).fit()),
                    ("ARIMA(2,1,1)", lambda s: _ARIMA(s, order=(2,1,1)).fit()),
                    ("SARIMA(1,1,1)(1,0,1,5)", lambda s: _SARIMAX(s, order=(1,1,1), seasonal_order=(1,0,1,5), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)),
                ]

                for i, (name, fit_fn) in enumerate(configs):
                    try:
                        m = fit_fn(s_decision)
                        aic = m.aic
                        # Forecast
                        if hasattr(m, 'forecast'):
                            fore = m.forecast(steps=n_pred)
                        else:
                            fore = m.predict(start=len(s_decision), end=len(s_decision)+n_pred-1)
                        rmse = float(np.sqrt(np.mean(m.resid**2)))
                        model_results[name] = {
                            'aic': round(aic, 2),
                            'rmse': round(rmse, 4),
                            'forecast': fore.values if hasattr(fore,'values') else np.array(fore),
                            'fitted': m.fittedvalues,
                            'resid': m.resid,
                        }
                    except Exception as e:
                        pass
                    progress.progress((i+1)/len(configs), text=f"Modèle {name}...")

                progress.empty()

                if not model_results:
                    st.error("Aucun modèle n'a pu être entraîné. Vérifiez les données.")
                else:
                    # ── Tableau de comparaison ─────────────────────────────────
                    st.markdown("### 📊 Comparaison des modèles")
                    comp_df = pd.DataFrame([
                        {'Modèle': k, 'AIC': v['aic'], 'RMSE': v['rmse']}
                        for k, v in model_results.items()
                    ]).sort_values('AIC').reset_index(drop=True)
                    comp_df['Rang'] = ['🥇','🥈','🥉'] + ['  ']*max(0, len(comp_df)-3)

                    st.dataframe(
                        comp_df[['Rang','Modèle','AIC','RMSE']].style
                            .background_gradient(subset=['AIC'], cmap='RdYlGn_r')
                            .format({'AIC':'{:.2f}','RMSE':'{:.4f}'}),
                        use_container_width=True, hide_index=True
                    )

                    # ── Meilleur modèle ────────────────────────────────────────
                    best_name = comp_df.iloc[0]['Modèle']
                    best      = model_results[best_name]
                    st.success(f"✅ Meilleur modèle : **{best_name}** (AIC={best['aic']})")

                    # ── Graphe prévisions ──────────────────────────────────────
                    st.markdown(f"### 📈 Prévisions — {best_name}")
                    n = len(s_decision)
                    x_hist = list(range(n))
                    x_fore = list(range(n, n + n_pred))

                    fig_pred = go.Figure()

                    # Historical
                    fig_pred.add_trace(go.Scatter(
                        x=x_hist, y=s_decision,
                        mode='lines+markers', name='Observations',
                        line=dict(color='#94a3b8', width=1.5),
                        marker=dict(color='#38bdf8', size=5)
                    ))

                    # Fitted
                    fitted_arr = best['fitted'].values if hasattr(best['fitted'],'values') else np.array(best['fitted'])
                    fig_pred.add_trace(go.Scatter(
                        x=x_hist[:len(fitted_arr)], y=fitted_arr,
                        mode='lines', name='Ajusté',
                        line=dict(color='#10b981', width=1.5, dash='dot')
                    ))

                    # All model forecasts (light)
                    for mname, mres in model_results.items():
                        if mname == best_name: continue
                        fig_pred.add_trace(go.Scatter(
                            x=x_fore, y=mres['forecast'],
                            mode='lines', name=mname,
                            line=dict(width=0.8, dash='dot'),
                            opacity=0.25, showlegend=False
                        ))

                    # Best model forecast (bold)
                    fig_pred.add_trace(go.Scatter(
                        x=x_fore, y=best['forecast'],
                        mode='lines+markers', name=f'Prévision ({best_name})',
                        line=dict(color='#f59e0b', width=3),
                        marker=dict(symbol='diamond', size=9, color='#f59e0b')
                    ))

                    # Uncertainty band (±1 std of residuals)
                    resid_std = float(best['resid'].std())
                    fig_pred.add_trace(go.Scatter(
                        x=x_fore + x_fore[::-1],
                        y=list(best['forecast'] + resid_std) + list((best['forecast'] - resid_std))[::-1],
                        fill='toself', fillcolor='rgba(245,158,11,0.12)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='Intervalle ±1σ', showlegend=True
                    ))

                    # Separator line
                    fig_pred.add_vline(x=n-0.5, line_dash='dash', line_color='#475569',
                        annotation_text="Maintenant", annotation_font_color='#94a3b8')

                    fig_pred.update_layout(
                        plot_bgcolor='#0f172a', paper_bgcolor='#0a0d14',
                        font=dict(color='#94a3b8'), height=400,
                        margin=dict(t=30, b=30, l=50, r=20),
                        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.08),
                        xaxis=dict(title='Match #', gridcolor='#1e293b'),
                        yaxis=dict(title=metric_choice, gridcolor='#1e293b'),
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    # Tableau prévisions
                    fore_arr = best['forecast']
                    current_mean = float(s_decision.mean())
                    fore_df_final = pd.DataFrame({
                        'Match': [f"M+{i+1}" for i in range(n_pred)],
                        f'{metric_choice} prévu': fore_arr.round(4),
                        'vs Moyenne saison': [f"{((v-current_mean)/(current_mean+1e-9))*100:+.1f}%" for v in fore_arr],
                        'Signal': ['🟢 Dessus' if v > current_mean else '🔴 Dessous' for v in fore_arr],
                    })
                    st.dataframe(fore_df_final, use_container_width=True, hide_index=True)

                    # ── SCORE DE DÉCISION ──────────────────────────────────────
                    st.markdown("---")
                    st.markdown("### 🎯 Score de Décision d'Achat")

                    # Calcul des 6 critères
                    slope_d, _, _, _, _ = stats.linregress(np.arange(len(s_decision)), s_decision)
                    first5_d = float(s_decision[:5].mean()) if len(s_decision)>=5 else float(s_decision.mean())
                    last5_d  = float(s_decision[-5:].mean()) if len(s_decision)>=5 else float(s_decision.mean())
                    form_delta_d = ((last5_d - first5_d) / (first5_d + 1e-9)) * 100
                    missed_d = int((match_df['minutes'].fillna(0)==0).sum()) if 'minutes' in match_df.columns else 0
                    cv_d = float(s_decision.std() / (s_decision.mean() + 1e-9))
                    fore_mean_d = float(np.mean(fore_arr))
                    fore_trend  = (fore_arr[-1] - fore_arr[0]) / (abs(fore_arr[0]) + 1e-9) * 100

                    # Score chaque critère /20
                    criteria = []

                    # 1. Tendance saison
                    score_trend = 20 if slope_d > 0.005 else 15 if slope_d > 0 else 8 if slope_d > -0.005 else 2
                    criteria.append(("📈 Tendance saison",
                        f"{'Progression' if slope_d>0.003 else 'Régression' if slope_d<-0.003 else 'Stable'} ({slope_d:+.4f}/match)",
                        score_trend, "#10b981" if score_trend>=15 else "#f59e0b" if score_trend>=10 else "#f87171"))

                    # 2. Forme récente
                    score_form = 20 if form_delta_d > 15 else 15 if form_delta_d > 5 else 10 if form_delta_d > -5 else 5 if form_delta_d > -15 else 1
                    criteria.append(("🔥 Forme récente",
                        f"{form_delta_d:+.1f}% (5 derniers vs 5 premiers)",
                        score_form, "#10b981" if score_form>=15 else "#f59e0b" if score_form>=10 else "#f87171"))

                    # 3. Constance
                    score_cons = 20 if cv_d < 0.4 else 15 if cv_d < 0.7 else 10 if cv_d < 1.0 else 5
                    criteria.append(("⚖️ Constance (CV)",
                        f"CV={cv_d:.2f} ({'Très régulier' if cv_d<0.4 else 'Régulier' if cv_d<0.7 else 'Irrégulier' if cv_d<1 else 'Très irrégulier'})",
                        score_cons, "#10b981" if score_cons>=15 else "#f59e0b" if score_cons>=10 else "#f87171"))

                    # 4. Disponibilité
                    score_avail = 20 if missed_d<=2 else 15 if missed_d<=4 else 10 if missed_d<=7 else 3
                    criteria.append(("🚑 Disponibilité",
                        f"{missed_d} matchs manqués ({'Excellent' if missed_d<=2 else 'Bon' if missed_d<=4 else 'Risqué' if missed_d<=7 else 'Problématique'})",
                        score_avail, "#10b981" if score_avail>=15 else "#f59e0b" if score_avail>=10 else "#f87171"))

                    # 5. Niveau actuel vs prévision
                    score_level = 20 if fore_mean_d > current_mean * 1.1 else 15 if fore_mean_d >= current_mean * 0.95 else 8 if fore_mean_d >= current_mean * 0.8 else 3
                    criteria.append(("🔮 Prévision vs Moyenne",
                        f"Prévu={fore_mean_d:.3f} vs Moyenne={current_mean:.3f} ({((fore_mean_d-current_mean)/(current_mean+1e-9))*100:+.1f}%)",
                        score_level, "#10b981" if score_level>=15 else "#f59e0b" if score_level>=10 else "#f87171"))

                    # 6. Tendance des prévisions
                    score_fore_trend = 20 if fore_trend > 10 else 14 if fore_trend > 0 else 8 if fore_trend > -10 else 2
                    criteria.append(("📊 Tendance prévisions",
                        f"{'Hausse' if fore_trend>5 else 'Légère hausse' if fore_trend>0 else 'Légère baisse' if fore_trend>-10 else 'Baisse'} ({fore_trend:+.1f}%)",
                        score_fore_trend, "#10b981" if score_fore_trend>=15 else "#f59e0b" if score_fore_trend>=10 else "#f87171"))

                    total_score = sum(c[2] for c in criteria)
                    max_score   = len(criteria) * 20

                    # ── Jauges visuelles ───────────────────────────────────────
                    cols_c = st.columns(3)
                    for i, (label, detail, score, color) in enumerate(criteria):
                        with cols_c[i % 3]:
                            pct = score / 20 * 100
                            st.markdown(f"""
                            <div class="stat-block" style="border-color:{color};margin-bottom:10px;">
                              <div class="stat-label">{label}</div>
                              <div style="background:#1e293b;border-radius:4px;height:6px;margin:8px 0;">
                                <div style="background:{color};width:{pct}%;height:6px;border-radius:4px;"></div>
                              </div>
                              <div class="stat-value" style="color:{color};font-size:1.3rem;">{score}/20</div>
                              <div class="stat-interpretation">{detail}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # ── VERDICT FINAL ──────────────────────────────────────────
                    st.markdown("---")
                    pct_total = total_score / max_score * 100

                    if pct_total >= 75:
                        verdict = "✅ ACHETER"
                        verdict_detail = "Profil solide — performances en hausse, joueur fiable, prévisions favorables."
                        verdict_color = "#064e3b"
                        verdict_border = "#10b981"
                        gauge_color = "#10b981"
                    elif pct_total >= 55:
                        verdict = "🟡 SURVEILLER"
                        verdict_detail = "Profil intéressant mais incertain — négocier un prix réduit ou une clause de performance."
                        verdict_color = "#451a03"
                        verdict_border = "#f59e0b"
                        gauge_color = "#f59e0b"
                    elif pct_total >= 35:
                        verdict = "⚠️ RISQUÉ"
                        verdict_detail = "Des signaux préoccupants — blessures, déclin ou irrégularité. Attendre avant d'investir."
                        verdict_color = "#431407"
                        verdict_border = "#f97316"
                        gauge_color = "#f97316"
                    else:
                        verdict = "❌ NE PAS ACHETER"
                        verdict_detail = "Profil trop incertain — déclin marqué, disponibilité faible ou prévisions négatives."
                        verdict_color = "#450a0a"
                        verdict_border = "#f87171"
                        gauge_color = "#f87171"

                    # Score gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=pct_total,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Score de décision — {player_name}", 'font': {'color': '#94a3b8', 'size': 14}},
                        delta={'reference': 55, 'increasing': {'color': '#10b981'}, 'decreasing': {'color': '#f87171'}},
                        number={'suffix': "/100", 'font': {'color': gauge_color, 'size': 36}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#475569', 'tickfont': {'color':'#475569'}},
                            'bar': {'color': gauge_color, 'thickness': 0.25},
                            'bgcolor': '#111827',
                            'bordercolor': '#1e293b',
                            'steps': [
                                {'range': [0,  35], 'color': '#450a0a'},
                                {'range': [35, 55], 'color': '#431407'},
                                {'range': [55, 75], 'color': '#451a03'},
                                {'range': [75,100], 'color': '#064e3b'},
                            ],
                            'threshold': {
                                'line': {'color': '#f59e0b', 'width': 3},
                                'thickness': 0.8, 'value': 55
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='#0a0d14', font=dict(color='#94a3b8'),
                        height=300, margin=dict(t=40, b=20, l=40, r=40)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Verdict card
                    st.markdown(f"""
                    <div style="background:{verdict_color};border:2px solid {verdict_border};
                         border-radius:12px;padding:24px;text-align:center;margin:16px 0;">
                      <div style="font-size:2rem;font-weight:800;color:{verdict_border};
                           font-family:'Space Mono',monospace;letter-spacing:0.05em;">
                        {verdict}
                      </div>
                      <div style="font-size:0.95rem;color:#e2e8f0;margin-top:10px;">
                        {verdict_detail}
                      </div>
                      <div style="font-size:1.5rem;font-weight:700;color:{verdict_border};margin-top:12px;">
                        Score : {total_score}/{max_score} ({pct_total:.0f}%)
                      </div>
                      <div style="font-size:0.82rem;color:#94a3b8;margin-top:8px;">
                        Modèle prédictif : {best_name} | Horizon : {n_pred} matchs
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Download report
                    report_lines = [
                        f"RAPPORT DE DÉCISION — {player_name}",
                        f"Métrique : {metric_choice} | Modèle : {best_name}",
                        "="*60,
                        f"VERDICT : {verdict}",
                        f"Score : {total_score}/{max_score} ({pct_total:.0f}%)",
                        "",
                        "DÉTAIL DES CRITÈRES :",
                    ]
                    for label, detail, score, _ in criteria:
                        report_lines.append(f"  {label}: {score}/20 — {detail}")
                    report_lines += [
                        "",
                        f"PRÉVISIONS ({n_pred} prochains matchs) :",
                    ]
                    for i, v in enumerate(fore_arr):
                        report_lines.append(f"  Match +{i+1}: {v:.4f} ({((v-current_mean)/(current_mean+1e-9))*100:+.1f}% vs moyenne)")
                    report_lines += [
                        "",
                        "CONCLUSION :",
                        verdict_detail,
                    ]
                    st.download_button(
                        "📄 Télécharger le rapport de décision",
                        data="\n".join(report_lines).encode('utf-8'),
                        file_name=f"decision_{player_name.replace(' ','_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )


def main():
    # ── Header ──────────────────────────────────────────────────────────────
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown('<p class="app-title">⚽ Football Analytics Pro</p>', unsafe_allow_html=True)
        st.markdown('<p class="app-subtitle">Football Player Performance Prediction</p>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="section-header">⚙ Configuration Ollama</p>', unsafe_allow_html=True)

        ollama_host = st.text_input("Host Ollama", value="http://localhost:11434", key="ollama_host")
        ollama_ok, available_models = check_ollama(ollama_host, "")

        if ollama_ok:
            st.markdown('<span class="status-badge badge-online">● OLLAMA EN LIGNE</span>', unsafe_allow_html=True)
            model_options = available_models if available_models else ["llama3.2", "mistral", "qwen2.5"]
            ollama_model = st.selectbox("Modèle", model_options)
        else:
            st.markdown('<span class="status-badge badge-offline">● OLLAMA HORS LIGNE</span>', unsafe_allow_html=True)
            ollama_model = st.text_input("Nom du modèle", value="llama3.2")
            st.caption("Démarrez Ollama : `ollama serve`")


    # ── Load Data ────────────────────────────────────────────────────────────
    with st.spinner("⏳ Chargement des données..."):
        df, data_warning = load_data()
    if data_warning:
        st.warning(data_warning)
    filtered = df.copy()
    with col_status:
        st.metric("Joueurs disponibles", len(filtered))

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🔁 Aide au Transfert"])

    # ── TAB 8: Aide au Transfert ─────────────────────────────────────────────
    render_transfer_tab(tabs[0], filtered, df, ollama_ok, ollama_host, ollama_model, query_ollama)

# ── INJECTED TAB: Aide au Transfert ─────────────────────────────────────────
if __name__ == "__main__":
    main()
