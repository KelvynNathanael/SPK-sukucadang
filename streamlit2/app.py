import warnings
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

LOCAL_DATA_PATH = Path(__file__).parent / 'DataPenjualanGaikindo.xlsx'

from fuzzy_ahp import (
    RI_TABLE,
    buildCriteriaTable,
    buildPairwiseMatrix,
    checkConsistencyRatio,
    fuzzyExtentAnalysis,
    loadAndClean,
    scoreAndRank,
    findInconsistencies,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SPK Suku Cadang Mobil",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; color: #1f2937; margin-bottom: 0.2rem; }
    .subtitle    { color: #6b7280; margin-bottom: 1.2rem; font-size: 0.95rem; }
    .icon-label  { display: flex; align-items: center; gap: 8px; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .cr-ok   { color: #16a34a; font-weight: 600; }
    .cr-warn { color: #dc2626; font-weight: 600; }
    .metric-good { color: #16a34a; }
    .metric-warn { color: #f59e0b; }
    .metric-bad  { color: #dc2626; }
    .discontinued-badge {
        display: inline-block;
        background: #f3f4f6;
        color: #6b7280;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

CRITERIA_NAMES = ["C1 (Tren)", "C2 (CC)", "C3 (GVW)", "C4 (HP)"]

PAIRWISE_PAIRS: list[tuple[tuple[int, int], str, str]] = [
    ((0, 1), "C1 (Tren Populasi)", "C2 (Kapasitas CC)"),
    ((0, 2), "C1 (Tren Populasi)", "C3 (GVW)"),
    ((0, 3), "C1 (Tren Populasi)", "C4 (Horse Power)"),
    ((1, 2), "C2 (Kapasitas CC)",  "C3 (GVW)"),
    ((1, 3), "C2 (Kapasitas CC)",  "C4 (Horse Power)"),
    ((2, 3), "C3 (GVW)",           "C4 (Horse Power)"),
]

SCALE_STEPS = [-9, -7, -5, -3, 1, 3, 5, 7, 9]

ACTIVE_YEAR = 2025

# ── Komponen suku cadang & frekuensi penggantian (per kendaraan/tahun) ────────
# Frekuensi = berapa kali rata-rata komponen ini diganti dalam setahun per kendaraan
KOMPONEN: dict[str, float] = {
    "Filter Oli":   2.0,   # ganti tiap ~6 bulan
    "Oli Mesin":    2.0,
    "Filter Udara": 1.0,   # ganti tiap ~12 bulan
    "Kampas Rem":   0.5,   # ganti tiap ~2 tahun
    "Busi":         0.5,
}


def prefOptions(cxLabel: str, cyLabel: str) -> list[tuple[int, str]]:
    return [
        (-9, f"{cyLabel} Mutlak Lebih Penting"),
        (-7, f"{cyLabel} Sangat Lebih Penting"),
        (-5, f"{cyLabel} Cukup Lebih Penting"),
        (-3, f"{cyLabel} Sedikit Lebih Penting"),
        (1,  "Sama Penting"),
        (3,  f"{cxLabel} Sedikit Lebih Penting"),
        (5,  f"{cxLabel} Cukup Lebih Penting"),
        (7,  f"{cxLabel} Sangat Lebih Penting"),
        (9,  f"{cxLabel} Mutlak Lebih Penting"),
    ]

def maxConsecutiveNonZero(series: pd.Series) -> int:
    max_run = cur = 0
    for v in series:
        if v > 0:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run

def isDiscontinued(row: pd.Series, monthCols: list[str], activeYear: int = ACTIVE_YEAR) -> bool:
    yearCols = [c for c in monthCols if c.endswith(f"_{activeYear}")]
    if not yearCols:
        return False
    return float(row[yearCols].sum()) == 0


def hasEnoughConsecutive(row: pd.Series, monthCols: list[str], minConsec: int = 24) -> bool:
    series = pd.Series(row[monthCols].values, dtype=float).fillna(0).clip(lower=0)
    first_sale = findFirstSale(series)
    series = series.iloc[first_sale:].reset_index(drop=True)
    return maxConsecutiveNonZero(series) >= minConsec


def buildYearLabels(monthCols: list[str]) -> list[str]:
    MONTH_MAP = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    labels = []
    for col in monthCols:
        parts = col.split("_")
        if len(parts) == 2:
            m = MONTH_MAP.get(parts[0], 1)
            y = parts[1]
            labels.append(f"{y}-{m:02d}")
        else:
            labels.append(col)
    return labels


def buildForecastYearLabels(lastHistLabel: str, periods: int) -> list[str]:
    parts = lastHistLabel.split("-")
    year, month = int(parts[0]), int(parts[1])
    labels = []
    for _ in range(periods):
        month += 1
        if month > 12:
            month = 1
            year += 1
        labels.append(f"{year}-{month:02d}")
    return labels

def findFirstSale(series: pd.Series) -> int:
    nonzero = np.where(series.values > 0)[0]
    if len(nonzero) == 0:
        return len(series)
    return int(nonzero[0])

@st.cache_data(show_spinner=False)
def loadDataFromBytes(fileBytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(fileBytes)
        tmpPath = tmp.name
    return loadAndClean(tmpPath)


@st.cache_data(show_spinner=False)
def forecastOne(
    monthlySales: tuple,
    periods: int = 12,
    season: int = 12,
    holdoutMonths: int = 12,
) -> tuple[list[float], list[float], float | None, float | None]:
    series = pd.Series(monthlySales, dtype=float).fillna(0).clip(lower=0)

    first_sale = findFirstSale(series)
    trimOffset = first_sale
    series     = series.iloc[first_sale:].reset_index(drop=True)

    if maxConsecutiveNonZero(series) < season * 2:
        meanVal      = series.tail(min(12, len(series))).mean()
        fittedPadded = [0.0] * trimOffset + series.tolist()
        return fittedPadded, [max(0.0, float(meanVal))] * periods, None, None

    rmse: float | None = None
    mape: float | None = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ── Full model fit dulu ────────────────────────────────────────────
            fullModel = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=season,
                initialization_method="estimated",
            ).fit(optimized=True)

            fittedTrimmed = fullModel.fittedvalues.clip(lower=0).tolist()
            fittedPadded  = [0.0] * trimOffset + fittedTrimmed
            fc            = fullModel.forecast(periods).clip(lower=0)

            # ── Hitung RMSE & MAPE dari residual in-sample ─────────────────────
            # Pakai N bulan terakhir fitted vs actual sebagai "holdout" evaluasi
            # Ini valid karena: kita tidak pakai fitted untuk forecast,
            # forecast tetap dari fullModel. Metrik ini menunjukkan
            # seberapa baik model menangkap pola historis.
            evalN = min(holdoutMonths, len(series) - season)  # minimal sisakan 1 siklus
            if evalN > 0:
                actualEval = series.iloc[-evalN:].values
                fittedEval = np.array(fittedTrimmed[-evalN:])

                rmse = float(np.sqrt(np.mean((actualEval - fittedEval) ** 2)))

                nonzeroMask = actualEval > 0
                if nonzeroMask.any():
                    rawMape = np.abs(
                        (actualEval[nonzeroMask] - fittedEval[nonzeroMask])
                        / actualEval[nonzeroMask]
                    ) * 100
                    mape = float(np.mean(np.clip(rawMape, 0, 200)))

            return fittedPadded, fc.tolist(), rmse, mape

    except Exception:
        meanVal      = series.tail(min(12, len(series))).mean()
        fittedPadded = [0.0] * trimOffset + series.tolist()
        return fittedPadded, [max(0.0, float(meanVal))] * periods, None, None

@st.cache_data(show_spinner=False)
def computeAllForecasts(
    df: pd.DataFrame,
    monthCols: list[str],
    periods: int = 12,
) -> tuple[dict, dict, dict, dict]:
    forecastTotals: dict      = {}
    forecastSeriesDict: dict  = {}
    fittedSeriesDict: dict    = {}
    metricsDict: dict         = {}
    discontinuedCount         = 0
    shortDataCount            = 0

    progress = st.progress(0.0, text="Memproses Holt-Winters...")
    n = len(df)

    for i, idx in enumerate(df.index):
        row = df.loc[idx]

        if isDiscontinued(row, monthCols):
            discontinuedCount         += 1
            forecastTotals[idx]        = 0.0
            forecastSeriesDict[idx]    = []
            fittedSeriesDict[idx]      = []
            metricsDict[idx]           = {
                "rmse": None, "mape": None,
                "discontinued": True, "skip_reason": "no_sales_2025",
            }

        elif not hasEnoughConsecutive(row, monthCols, minConsec=24):
            shortDataCount            += 1
            forecastTotals[idx]        = 0.0
            forecastSeriesDict[idx]    = []
            fittedSeriesDict[idx]      = []
            metricsDict[idx]           = {
                "rmse": None, "mape": None,
                "discontinued": True, "skip_reason": "consec_lt_24",
            }

        else:
            sales = tuple(row[monthCols].values.tolist())
            fitted, fc, rmse, mape     = forecastOne(sales, periods=periods, holdoutMonths=12)
            forecastTotals[idx]        = float(sum(fc))
            forecastSeriesDict[idx]    = fc
            fittedSeriesDict[idx]      = fitted
            metricsDict[idx]           = {
                "rmse": rmse, "mape": mape,
                "discontinued": False, "skip_reason": None,
            }

        if i % max(1, n // 30) == 0:
            progress.progress((i + 1) / n, text=f"Memproses... {i+1}/{n}")

    progress.empty()

    return forecastTotals, forecastSeriesDict, fittedSeriesDict, metricsDict


def calcEoq(demand: float, orderingCost: float, holdingCost: float) -> int:
    if holdingCost <= 0 or demand <= 0:
        return 0
    return int(np.ceil(np.sqrt((2 * demand * orderingCost) / holdingCost)))


def mapeColorClass(mape: float | None) -> str:
    if mape is None:
        return ""
    if mape < 10:
        return "metric-good"
    if mape < 20:
        return "metric-warn"
    return "metric-bad"

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-size:1.1rem;font-weight:600;margin-bottom:0">'
        '&nbsp;Preferensi Bengkel</p>',
        unsafe_allow_html=True,
    )

    uploadedFile = st.file_uploader(
        "Upload Data Gaikindo",
        type=["xlsx"],
        help="File Excel berisi data penjualan kendaraan dari Gaikindo.",
    )

    if Path(LOCAL_DATA_PATH).exists():
        with open(LOCAL_DATA_PATH, "rb") as f:
            st.download_button(
                label="Download Data Gaikindo",
                data=f,
                file_name="DataPenjualanGaikindo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Download dulu, lalu upload di bawah.",
            )
    else:
        st.warning(f"File lokal tidak ditemukan:\n`{LOCAL_DATA_PATH}`")

    st.divider()
    st.markdown(
        '<p style="font-weight:600">'
        '&nbsp;Bobot Kriteria (Fuzzy AHP)</p>',
        unsafe_allow_html=True,
    )
    st.caption("4 kriteria - 6 perbandingan berpasangan")

    pairwiseValues: dict[tuple[int, int], int] = {}
    for (pair, cx, cy) in PAIRWISE_PAIRS:
        opts = prefOptions(cx, cy)
        val = st.select_slider(
            f"{cx}  vs  {cy}",
            options=[o[0] for o in opts],
            value=1,
            format_func=lambda v, _opts=opts: dict(_opts)[v],
            key=f"pw_{pair[0]}_{pair[1]}",
        )
        st.markdown('<div class="margin-bottom: 1.5rem;"> </div>', unsafe_allow_html=True)
        pairwiseValues[pair] = val

    st.divider()
    st.markdown(
        '<p style="font-weight:600">'
        '&nbsp;Parameter EOQ</p>',
        unsafe_allow_html=True,
    )

    targetServis = st.number_input(
        "Kapasitas Servis Bengkel per Bulan (kendaraan)",
        min_value=1, max_value=10000, value=100, step=10,
        help=(
            "Total kendaraan yang mampu diservis bengkel Anda per bulan. "
            "Demand suku cadang tiap model dihitung secara proporsional "
            "dari tren penjualan C1 (historis + forecast) terhadap total "
            "tren semua model dalam Top-N yang dipilih."
        ),
    )
    biayaPesan = st.number_input(
        "Biaya Sekali Pesan (Rp)",
        min_value=0, value=5_000, step=1_000,
        help="Biaya tetap setiap kali melakukan pemesanan suku cadang.",
    )
    biayaSimpan = st.number_input(
        "Biaya Simpan per Unit per Tahun (Rp)",
        min_value=0, value=5_000, step=1_000,
        help="Biaya penyimpanan per unit suku cadang per tahun.",
    )

    st.divider()
    enableForecast = st.toggle(
        "Aktifkan Forecast Holt-Winters",
        value=False,
        help=(
            "Forecast hanya dilakukan untuk model yang masih aktif di 2025. "
            "Model discontinue (tidak ada penjualan di 2025) di-skip otomatis."
        ),
    )

    forecastPeriods = (
        st.slider("Periode Forecast (bulan)", 6, 24, 12, 6)
        if enableForecast
        else 12
    )

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-header">Sistem Rekomendasi Suku Cadang Mobil</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">'
    "SPK Penentuan Prioritas Stok Fast-Moving &nbsp;&middot;&nbsp; "
    "Fuzzy AHP (4 Kriteria) &amp; Holt-Winters"
    "</p>",
    unsafe_allow_html=True,
)

if uploadedFile is None and not st.session_state.get("use_local"):
    st.info("Mulai dengan upload file data Gaikindo (.xlsx) di sidebar.")
    with st.expander("Tentang Sistem", expanded=True):
        st.markdown("""
        Sistem ini membantu UMKM bengkel otomotif menentukan **prioritas stok suku cadang fast-moving**
        berdasarkan tren populasi kendaraan di pasar.

        **Metode yang dipakai:**
        - **Holt-Winters** — peramalan tren penjualan
        - **Fuzzy AHP** — pembobotan kriteria (4 kriteria, 6 perbandingan)
        - **Economic Order Quantity (EOQ)** — rekomendasi kuantitas pemesanan

        **Deteksi Discontinue:**
        > Model yang tidak memiliki penjualan di seluruh bulan tahun **2025** dianggap
        > discontinue dan **tidak di-forecast**. C1 tetap dihitung dari akumulasi historis.

        **Cara hitung demand EOQ:**
        > Demand tiap model dihitung dari **proporsi tren penjualan C1** (historis + forecast)
        > model tersebut terhadap total C1 semua model dalam Top-N, dikalikan kapasitas
        > servis bengkel dan frekuensi penggantian komponen.

        **Kriteria Evaluasi:**
        | Kode | Kriteria | Keterangan |
        |------|----------|------------|
        | C1 | Tren Populasi | Akumulasi historis + forecast (opsional, hanya model aktif) |
        | C2 | Kapasitas Mesin (CC) | Kapasitas Mesin (CC) kendaraan |
        | C3 | Gross Vehicle Weight (GVW) | Berat Kendaraan |
        | C4 | Horse Power (HP) | Besar Tenaga Kendaraan |
        """)
    st.stop()

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
try:
    with st.spinner("Memuat data..."):
        dfFull, monthCols = loadDataFromBytes(uploadedFile.getvalue())
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

if len(dfFull) == 0:
    st.warning("Data kosong setelah cleaning. Periksa format file Excel.")
    st.stop()

yearLabels = buildYearLabels(monthCols)

# ─── SIDEBAR FILTER TAHUN & CC ────────────────────────────────────────────────
allYears = sorted({int(c.split("_")[1]) for c in monthCols})

with st.sidebar:
    st.divider()
    st.markdown(
        '<p style="font-weight:600">&nbsp;Filter Tahun Model</p>',
        unsafe_allow_html=True,
    )
    yearRange = st.slider(
        "Tampilkan model aktif di tahun",
        min_value=allYears[0],
        max_value=allYears[-1],
        value=(allYears[0], allYears[-1]),
        help=(
            "Hanya tampilkan model yang memiliki penjualan > 0 "
            "minimal 1 bulan dalam rentang tahun ini."
        ),
    )

    st.divider()
    st.markdown(
        '<p style="font-weight:600">&nbsp;Filter Kapasitas Mesin (CC)</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Filter ini mempengaruhi ranking, forecast, dan EOQ secara global. "
        "Gunakan filter CC di tab Ranking untuk filter tampilan saja."
    )

    ccColFull = dfFull["CC"].fillna(0)
    ccMinGlobal = int(ccColFull[ccColFull > 0].min()) if (ccColFull > 0).any() else 0
    ccMaxGlobal = int(ccColFull.max())

    if ccMaxGlobal > ccMinGlobal:
        globalCcRange = st.slider(
            "Rentang CC (cc)",
            min_value=ccMinGlobal,
            max_value=ccMaxGlobal,
            value=(ccMinGlobal, ccMaxGlobal),
            step=100,
            help=(
                "Model dengan CC di luar rentang ini akan dikeluarkan dari seluruh "
                "perhitungan (AHP, forecast, EOQ)."
            ),
        )
    else:
        globalCcRange = (ccMinGlobal, ccMaxGlobal)
        st.info(f"Semua model memiliki CC yang sama ({ccMinGlobal} cc).")

# ─── FILTER ROWS ──────────────────────────────────────────────────────────────
yearFilterCols = [
    c for c in monthCols
    if yearRange[0] <= int(c.split("_")[1]) <= yearRange[1]
]

activeInRange = dfFull[yearFilterCols].sum(axis=1) > 0
df = dfFull[activeInRange].copy()

ccMask = (
    (df["CC"].fillna(0) >= globalCcRange[0]) &
    (df["CC"].fillna(0) <= globalCcRange[1])
)
df = df[ccMask].reset_index(drop=True)

if len(df) == 0:
    st.warning(
        f"Tidak ada model yang cocok dengan filter tahun **{yearRange[0]}–{yearRange[1]}** "
        f"dan CC **{globalCcRange[0]}–{globalCcRange[1]} cc**. "
        "Coba perlebar rentang di sidebar."
    )
    st.stop()

filterInfoParts = []
if yearRange != (allYears[0], allYears[-1]):
    filterInfoParts.append(f"Tahun **{yearRange[0]}–{yearRange[1]}**")
if globalCcRange != (ccMinGlobal, ccMaxGlobal):
    filterInfoParts.append(f"CC **{globalCcRange[0]}–{globalCcRange[1]} cc**")

if filterInfoParts:
    filteredOut = len(dfFull) - len(df)
    st.info(
        f"Filter aktif: {' · '.join(filterInfoParts)} — "
        f"menampilkan **{len(df)} model** "
        f"({filteredOut} model disembunyikan)."
    )

modelToDfIdx = {
    (df.loc[i, "BRAND"], df.loc[i, "MODEL"]): i for i in df.index
}

# ─── FORECAST ─────────────────────────────────────────────────────────────────
forecastTotals     = None
forecastSeriesDict: dict = {}
fittedSeriesDict: dict   = {}
metricsDict: dict        = {}

if enableForecast:
    forecastTotals, forecastSeriesDict, fittedSeriesDict, metricsDict = computeAllForecasts(
        df, monthCols, periods=forecastPeriods
    )

# ─── PIPELINE FUZZY AHP ───────────────────────────────────────────────────────
criteriadf         = buildCriteriaTable(df, monthCols, forecastTotals)
matrix             = buildPairwiseMatrix(pairwiseValues, n=4, verbose=False)
weights, fahpDebug = fuzzyExtentAnalysis(matrix, verbose=False)
lambdaMax, cr      = checkConsistencyRatio(matrix, verbose=False)
rankedDf           = scoreAndRank(criteriadf, weights)
inconsistencyInfo  = findInconsistencies(pairwiseValues, weights, matrix, CRITERIA_NAMES)

c3Available = criteriadf.attrs.get("c3_available", False)
c4Available = criteriadf.attrs.get("c4_available", False)

crOk     = cr <= 0.10
crClass  = "cr-ok" if crOk else "cr-warn"
crSymbol = "✓" if crOk else "✗"
crText   = "Konsisten" if crOk else "Tidak Konsisten"

st.markdown(
    f'<div style="padding:8px 14px;border-radius:8px;'
    f'background:{"#f0fdf4" if crOk else "#fef2f2"};'
    f'border:1px solid {"#86efac" if crOk else "#fca5a5"};margin-bottom:12px">'
    f'<span class="{crClass}">{crSymbol} Consistency Ratio (CR) = {cr:.4f} — {crText}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

if not crOk:
    violations   = inconsistencyInfo["transitivity_violations"]
    topDevs      = inconsistencyInfo["top_deviations"]
    skippedPairs = inconsistencyInfo["skipped_pairs"]

    with st.expander("Lihat detail inkonsistensi & saran perbaikan", expanded=True):
        st.markdown("#### Kontradiksi logika (pelanggaran transitivitas)")
        st.caption(
            "Jika A lebih penting dari B, dan B lebih penting dari C, "
            "maka A **harus** lebih penting dari C."
        )
        if violations:
            for v in violations:
                st.warning(v["explanation"], icon="⚠️")
        else:
            st.success("Tidak ada kontradiksi logika. Inkonsistensi bersifat numerik.")

        st.divider()
        st.markdown("#### Pasangan yang paling perlu diperbaiki")

        if skippedPairs:
            st.info(f"Pasangan di-skip (kolom tidak tersedia): {', '.join(skippedPairs)}")
        if violations:
            st.warning("Perbaiki siklus logika dulu sebelum melihat saran numerik.")
        elif not topDevs:
            st.success("Semua pasangan sudah konsisten secara numerik.")
        else:
            st.caption("Nilai 'ideal' adalah rasio bobot Fuzzy AHP. Semakin jauh, semakin besar CR.")
            saatyOptions = [1, 3, 5, 7, 9]
            for rank, d in enumerate(topDevs, 1):
                implied = d["implied"]
                if implied >= 1:
                    suggestedRaw   = min(saatyOptions, key=lambda x: abs(x - implied))
                    suggestedLabel = f"{d['label_i']} **{suggestedRaw}x lebih penting** dari {d['label_j']}"
                else:
                    inv            = 1 / implied if implied > 0 else 9
                    suggestedRaw   = min(saatyOptions, key=lambda x: abs(x - inv))
                    suggestedLabel = f"{d['label_j']} **{suggestedRaw}x lebih penting** dari {d['label_i']}"

                st.markdown(
                    f"**#{rank} — {d['label_i']} vs {d['label_j']}**  \n"
                    f"Kamu pilih: `{d['actual']:.2f}` &nbsp;|&nbsp; "
                    f"Idealnya: `{d['implied']:.2f}` &nbsp;|&nbsp; Saran: {suggestedLabel}"
                )

wCols   = st.columns(4)
wLabels = ["C1 Tren", "C2 CC", "C3 GVW", "C4 HP"]
unavail = [False, False, not c3Available, not c4Available]
for col, lbl, w, na in zip(wCols, wLabels, weights, unavail):
    note = " *(tidak ada kolom)*" if na else ""
    col.metric(f"{lbl}{note}", f"{w*100:.1f}%")

st.divider()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tabRank, tabForecast, tabEoq = st.tabs([
    "Ranking Prioritas",
    "Visualisasi Tren & Akurasi",
    "Rekomendasi EOQ",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RANKING
# ═══════════════════════════════════════════════════════════════════════════════
with tabRank:
    st.markdown("<h3>Tabel Ranking Model Kendaraan</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        brandFilter = st.multiselect(
            "Filter Merek",
            options=sorted(rankedDf["Brand"].unique()),
            default=[],
        )

    with col2:
        ccColRanked = rankedDf["CC"].fillna(0)
        ccMinTab = int(ccColRanked[ccColRanked > 0].min()) if (ccColRanked > 0).any() else 0
        ccMaxTab = int(ccColRanked.max())

        if ccMaxTab > ccMinTab:
            ccRangeTab = st.slider(
                "Filter CC (tampilan)",
                min_value=ccMinTab,
                max_value=ccMaxTab,
                value=(ccMinTab, ccMaxTab),
                step=100,
                help=(
                    "Filter CC ini hanya mempengaruhi tampilan tabel di tab ini. "
                    "Tidak mempengaruhi perhitungan AHP, forecast, maupun EOQ."
                ),
            )
        else:
            ccRangeTab = (ccMinTab, ccMaxTab)

    with col3:
        topN = st.slider("Tampilkan Top-N", 5, min(100, len(rankedDf)), 20)

    displayDf = rankedDf.copy()

    if brandFilter:
        displayDf = displayDf[displayDf["Brand"].isin(brandFilter)]

    displayDf = displayDf[
        (displayDf["CC"].fillna(0) >= ccRangeTab[0]) &
        (displayDf["CC"].fillna(0) <= ccRangeTab[1])
    ]

    totalRanked = len(rankedDf)
    totalShown  = len(displayDf)
    if totalShown < totalRanked:
        st.caption(
            f"Menampilkan **{min(topN, totalShown)}** dari **{totalShown}** model "
            f"(filter aktif dari total {totalRanked} model)."
        )

    showCols  = ["Brand", "Model", "CC", "C1_total", "C3_gvw", "C4_hp"]
    renameMap = {
        "Brand":    "Merek",
        "Model":    "Model",
        "CC":       "CC (cc)",
        "C1_total": "Tren Populasi",
        "C3_gvw":   "GVW (C3)",
        "C4_hp":    "HP (C4)",
    }

    st.dataframe(
        displayDf[showCols].head(topN).rename(columns=renameMap),
        use_container_width=True,
        column_config={
            "Tren Populasi": st.column_config.NumberColumn(format="%d"),
            "GVW (C3)":      st.column_config.NumberColumn(format="%g"),
            "HP (C4)":       st.column_config.NumberColumn(format="%g"),
        },
    )

    csv = rankedDf.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download Hasil Ranking (CSV)",
        data=csv,
        file_name="hasil_ranking_fuzzy_ahp.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST / TREN
# ═══════════════════════════════════════════════════════════════════════════════
with tabForecast:
    st.markdown("<h3>Tren Peramalan per Model</h3>", unsafe_allow_html=True)

    if not enableForecast:
        st.info("Aktifkan **Forecast Holt-Winters** di sidebar untuk melihat garis prediksi dan metrik akurasi.")

    if enableForecast and metricsDict:
        activeMetrics = [
            {
                "idx": idx,
                "Brand": df.loc[idx, "BRAND"],
                "Model": df.loc[idx, "MODEL"],
                **m,
            }
            for idx, m in metricsDict.items()
            if not m["discontinued"] and m["rmse"] is not None
        ]

        if activeMetrics:
            metricsDf = pd.DataFrame(activeMetrics)
            avgRmse   = metricsDf["rmse"].mean()
            avgMape   = metricsDf["mape"].dropna().mean()
            totalDisc = sum(1 for m in metricsDict.values() if m["discontinued"])
            totalAct  = sum(1 for m in metricsDict.values() if not m["discontinued"])

            st.markdown("#### Ringkasan Akurasi Holt-Winters")
            st.caption(
                f"Model aktif (diforecast): **{totalAct}** &nbsp;|&nbsp; "
                f"Discontinue (di-skip): **{totalDisc}**"
            )

            colM1, colM2, colM3, colM4 = st.columns(4)
            colM1.metric("Model Aktif", totalAct)
            colM2.metric("Model Discontinue", totalDisc)
            colM3.metric(
                "Rata-rata RMSE",
                f"{avgRmse:,.0f}" if not np.isnan(avgRmse) else "—",
                help="Root Mean Squared Error rata-rata (satuan: unit penjualan)",
            )
            colM4.metric(
                "Rata-rata MAPE",
                f"{avgMape:.1f}%" if not np.isnan(avgMape) else "—",
                help="Mean Absolute Percentage Error rata-rata. < 10% sangat baik.",
            )

            st.divider()

    topModels = rankedDf.head(50).apply(
        lambda r: f"#{r.name} {r['Brand']} - {r['Model']}", axis=1
    ).tolist()

    if enableForecast and metricsDict:
        labeledModels = []
        for label in topModels:
            try:
                _, brandPart = label.split(" ", 1)
                brand, model = brandPart.split(" - ", 1)
                dfIdx = modelToDfIdx.get((brand, model))
                if dfIdx is not None and metricsDict.get(dfIdx, {}).get("discontinued"):
                    labeledModels.append(label + " [discontinue]")
                else:
                    labeledModels.append(label)
            except ValueError:
                labeledModels.append(label)
    else:
        labeledModels = topModels

    selected = st.multiselect(
        "Pilih model untuk divisualisasikan (max 5)",
        options=labeledModels,
        default=labeledModels[:3] if len(labeledModels) >= 3 else labeledModels[:1],
        max_selections=5,
    )

    showFitted = False
    if enableForecast:
        showFitted = st.checkbox(
            "Tampilkan garis prediksi dalam periode historis (in-sample fit)",
            value=True,
            help=(
                "Menampilkan seberapa baik model Holt-Winters mencocokkan data historis. "
                "Semakin dekat garis prediksi dengan garis aktual, semakin baik model."
            ),
        )

    if selected:
        fig = go.Figure()
        selectedMetrics = []

        COLOR_PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]

        for selIdx, sel in enumerate(selected):
            cleanSel = sel.replace(" [discontinue]", "")
            try:
                _, brandPart = cleanSel.split(" ", 1)
                brand, model = brandPart.split(" - ", 1)
            except ValueError:
                continue

            dfIdx = modelToDfIdx.get((brand, model))
            if dfIdx is None:
                continue

            historical   = df.loc[dfIdx, monthCols].values.astype(float)
            xHist        = yearLabels
            isDisc       = metricsDict.get(dfIdx, {}).get("discontinued", False)
            modelLabel   = f"{brand} {model}"
            baseColor    = COLOR_PALETTE[selIdx % len(COLOR_PALETTE)]
            lineColor    = "#9ca3af" if isDisc else baseColor

            fig.add_trace(go.Scatter(
                x=xHist,
                y=historical,
                mode="lines",
                name=f"{modelLabel} — Aktual" + (" (discontinue)" if isDisc else ""),
                line=dict(width=2, color=lineColor),
                legendgroup=modelLabel,
            ))

            if enableForecast and not isDisc:
                fitted = fittedSeriesDict.get(dfIdx, [])
                fc     = forecastSeriesDict.get(dfIdx, [])

                if showFitted and fitted:
                    fig.add_trace(go.Scatter(
                        x=xHist,
                        y=fitted,
                        mode="lines",
                        name=f"{modelLabel} — Fitted (HW)",
                        line=dict(width=1.5, color=lineColor, dash="dot"),
                        legendgroup=modelLabel,
                        opacity=0.75,
                    ))

                if fc:
                    fcLabels = buildForecastYearLabels(xHist[-1], len(fc))
                    fig.add_trace(go.Scatter(
                        x=[xHist[-1]] + fcLabels,
                        y=[float(historical[-1])] + list(fc),
                        mode="lines",
                        name=f"{modelLabel} — Forecast",
                        line=dict(width=2, color=lineColor, dash="dash"),
                        legendgroup=modelLabel,
                    ))

            if enableForecast and dfIdx in metricsDict:
                m = metricsDict[dfIdx]
                selectedMetrics.append({
                    "Model":    modelLabel,
                    "Status":   "Discontinue" if m["discontinued"] else "Aktif",
                    "RMSE":     f"{m['rmse']:,.1f}" if m["rmse"] is not None else "—",
                    "MAPE (%)": f"{m['mape']:.2f}%" if m["mape"] is not None else "—",
                })

        title_suffix = ""
        if enableForecast and showFitted:
            title_suffix = " | garis titik-titik = fitted HW | garis putus-putus = forecast"
        elif enableForecast:
            title_suffix = " | garis putus-putus = forecast ke depan"

        fig.update_layout(
            title=f"Penjualan Tahunan per Model{title_suffix}",
            xaxis_title="Tahun-Bulan",
            yaxis_title="Volume Penjualan (unit/bulan)",
            hovermode="x unified",
            height=520,
            xaxis=dict(
                tickangle=-45,
                tickmode="array",
                tickvals=[lbl for lbl in yearLabels if lbl.endswith("-01")],
                ticktext=[lbl[:4] for lbl in yearLabels if lbl.endswith("-01")],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Sumbu X menampilkan label tahun (tick di bulan Januari). "
            "Hover untuk melihat nilai tiap bulan secara tepat."
        )

        if enableForecast and selectedMetrics:
            st.markdown("**Metrik Akurasi Model yang Dipilih**")
            st.caption(
                "RMSE & MAPE dihitung dari 12 bulan terakhir fitted vs aktual (in-sample). "
                "MAPE < 10% = sangat baik | 10–20% = baik | > 20% = perlu perhatian."
            )
            st.dataframe(
                pd.DataFrame(selectedMetrics).set_index("Model"),
                use_container_width=True,
            )

            # ── Tabel detail aktual vs prediksi per model ──────────────────────────
            st.markdown("**Detail Aktual vs Prediksi (12 bulan terakhir)**")
            for sel in selected:
                cleanSel = sel.replace(" [discontinue]", "")
                try:
                    _, brandPart = cleanSel.split(" ", 1)
                    brand, model = brandPart.split(" - ", 1)
                except ValueError:
                    continue

                dfIdx = modelToDfIdx.get((brand, model))
                if dfIdx is None:
                    continue

                m = metricsDict.get(dfIdx, {})
                if m.get("discontinued") or not fittedSeriesDict.get(dfIdx):
                    continue

                historical = df.loc[dfIdx, monthCols].values.astype(float)
                fitted     = fittedSeriesDict.get(dfIdx, [])

                if not fitted or len(fitted) < 12:
                    continue

                # Ambil 12 bulan terakhir
                evalN       = min(12, len(historical) - 12)
                actualEval  = historical[-evalN:]
                fittedEval  = np.array(fitted[-evalN:])
                labelEval   = yearLabels[-evalN:]

                detailRows = []
                for lbl, act, pred in zip(labelEval, actualEval, fittedEval):
                    act   = float(act)
                    pred  = float(pred)
                    err   = act - pred
                    pct   = abs(err / act) * 100 if act > 0 else None
                    detailRows.append({
                        "Bulan":        lbl,
                        "Aktual":       int(round(act)),
                        "Prediksi":     int(round(pred)),
                        "Error":        int(round(err)),
                        "APE (%)":      round(pct, 2) if pct is not None else None,
                    })

                detailDf = pd.DataFrame(detailRows)

                # Hitung ulang RMSE & MAPE dari tabel ini (verifikasi)
                rmseVerif = float(np.sqrt(np.mean((detailDf["Error"]) ** 2)))
                mapeVerif = detailDf["APE (%)"].dropna().mean()

                with st.expander(f"📊 {brand} {model} — RMSE: {rmseVerif:,.1f} | MAPE: {mapeVerif:.2f}%"):
                    st.dataframe(
                        detailDf.set_index("Bulan"),
                        use_container_width=True,
                        column_config={
                            "Aktual":   st.column_config.NumberColumn(format="%d"),
                            "Prediksi": st.column_config.NumberColumn(format="%d"),
                            "Error":    st.column_config.NumberColumn(format="%d"),
                            "APE (%)":  st.column_config.NumberColumn(format="%.2f%%"),
                        },
                    )
                    st.caption( 
                        f"RMSE (verifikasi manual): **{rmseVerif:,.1f}** &nbsp;|&nbsp; "
                        f"MAPE (verifikasi manual): **{mapeVerif:.2f}%**"
                    )
            else:
                st.info("Pilih minimal satu model di atas untuk melihat grafik tren.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EOQ
# ═══════════════════════════════════════════════════════════════════════════════
with tabEoq:
    st.markdown("<h3>Rekomendasi Pembelian Suku Cadang Fast-Moving</h3>",
                unsafe_allow_html=True)

    topNEoq  = st.slider("Hitung EOQ untuk Top-N model", 1, 10, 5)
    topEoqDf = rankedDf.head(topNEoq).copy()

    # ── Demand berbasis proporsi tren penjualan C1 ────────────────────────────
    # C1_total = akumulasi historis + forecast (kalau aktif).
    # Proporsi tiap model = C1_model / sum(C1 semua Top-N).
    # Demand tahunan model = proporsi × kapasitas_bengkel_per_tahun.
    # Logika: bengkel melayani sebanyak targetServis kendaraan/bulan,
    # dan model yang lebih banyak terjual di pasar = lebih besar kemungkinan masuk bengkel.

    totalC1 = float(topEoqDf["C1_total"].sum())

    kapasitasTahunan = targetServis * 12  # kendaraan/tahun yang dilayani bengkel

    st.info(
        f"**Kapasitas bengkel:** `{targetServis} kendaraan/bulan` "
        f"= `{kapasitasTahunan:,} kendaraan/tahun`  \n"
        f"**Demand per model** = proporsi tren penjualan C1 × kapasitas bengkel × frekuensi komponen.  \n"
        f"Total C1 Top-{topNEoq}: `{totalC1:,.0f}` unit (dari data Gaikindo{' + forecast' if enableForecast else ''})."
    )

    with st.expander("ℹ️ Penjelasan frekuensi penggantian komponen", expanded=False):
        st.markdown("""
        | Komponen | Frekuensi/tahun | Keterangan |
        |----------|----------------|------------|
        | Filter Oli | 2× | Ganti tiap ~6 bulan atau ~10.000 km |
        | Oli Mesin | 2× | Bersamaan dengan filter oli |
        | Filter Udara | 1× | Ganti tiap ~12 bulan atau ~20.000 km |
        | Kampas Rem | 0,5× | Ganti tiap ~2 tahun atau ~40.000 km |
        | Busi | 0,5× | Ganti tiap ~2 tahun (busi konvensional) |
        """)

    eoqRows = []
    for rank, row in topEoqDf.iterrows():
        c1Val    = float(row["C1_total"])
        # proporsi model ini terhadap seluruh Top-N
        proporsi = c1Val / totalC1 if totalC1 > 0 else 1.0 / max(len(topEoqDf), 1)

        # kendaraan model ini yang dilayani bengkel per tahun
        kendaraanPerTahun = kapasitasTahunan * proporsi

        for compName, freqPerTahun in KOMPONEN.items():
            # demand = kendaraan yg dilayani × frekuensi ganti komponen per tahun
            demand = max(1, int(round(kendaraanPerTahun * freqPerTahun)))

            eoq      = max(1, calcEoq(demand, biayaPesan, biayaSimpan))
            freqPesan = round(demand / eoq, 1)
            tic       = (demand / eoq) * biayaPesan + (eoq / 2) * biayaSimpan

            eoqRows.append({
                "Rank":                   rank,
                "Model":                  f"{row['Brand']} {row['Model']}",
                "Komponen":               compName,
                "Proporsi C1 (%)":        round(proporsi * 100, 2),
                "Est. Kendaraan/Thn":     demand // int(freqPerTahun) if freqPerTahun >= 1 else demand * 2,
                "Demand/Thn (pcs)":       demand,
                "EOQ (pcs/pesan)":        eoq,
                "Frekuensi Pesan/Thn":    freqPesan,
                "Est. Total Biaya/Thn":   f"Rp {tic:,.0f}",
            })

    eoqDf = pd.DataFrame(eoqRows)

    st.dataframe(
        eoqDf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Proporsi C1 (%)": st.column_config.NumberColumn(
                "Proporsi C1 (%)",
                help="Porsi tren penjualan model ini dari total Top-N.",
                format="%.2f%%",
            ),
            "Demand/Thn (pcs)": st.column_config.NumberColumn(
                "Demand/Thn (pcs)",
                help="Estimasi jumlah komponen yang dibutuhkan per tahun.",
                format="%d",
            ),
            "EOQ (pcs/pesan)": st.column_config.NumberColumn(
                "EOQ (pcs/pesan)",
                help=(
                    "Jumlah unit optimal dibeli SEKALI PESAN "
                    "untuk meminimalkan total biaya pesan + simpan."
                ),
                format="%d",
            ),
            "Frekuensi Pesan/Thn": st.column_config.NumberColumn(
                "Frekuensi Pesan/Thn",
                help="Berapa kali pesan dalam setahun = Demand ÷ EOQ.",
                format="%.1f×",
            ),
        },
    )

    csv = eoqDf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Hasil EOQ (CSV)",
        data=csv,
        file_name="hasil_rekomendasi_eoq.csv",
        mime="text/csv",
    )

st.divider()
st.caption("2026 — Sistem Rekomendasi Suku Cadang Fast Moving — Kelvyn — Skripsi S1 Teknik Informatika")