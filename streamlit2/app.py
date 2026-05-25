import warnings
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

LOCAL_DATA_PATH = '../DataPenjualanGaikindo.xlsx'

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
    return maxConsecutiveNonZero(series) >= minConsec


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
) -> tuple[list[float], float | None, float | None]:
    series = pd.Series(monthlySales, dtype=float).fillna(0).clip(lower=0)

    if maxConsecutiveNonZero(series) < season * 2:
        meanVal = series.tail(min(12, len(series))).mean()
        return [max(0.0, float(meanVal))] * periods, None, None

    rmse: float | None = None
    mape: float | None = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            minTrainLen = season * 2
            if len(series) >= minTrainLen + holdoutMonths:
                trainSeries = series.iloc[:-holdoutMonths]
                actualHoldout = series.iloc[-holdoutMonths:].values

                valModel = ExponentialSmoothing(
                    trainSeries,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=season,
                    initialization_method="estimated",
                ).fit(optimized=True)

                predHoldout = valModel.forecast(holdoutMonths).clip(lower=0).values

                rmse = float(np.sqrt(np.mean((actualHoldout - predHoldout) ** 2)))

                nonzeroMask = actualHoldout > 0
                if nonzeroMask.any():
                    rawMape = np.abs(
                        (actualHoldout[nonzeroMask] - predHoldout[nonzeroMask])
                        / actualHoldout[nonzeroMask]
                    ) * 100
                    mape = float(np.mean(np.clip(rawMape, 0, 200)))

            fullModel = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=season,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = fullModel.forecast(periods).clip(lower=0)
            return fc.tolist(), rmse, mape

    except Exception:
        meanVal = series.tail(min(12, len(series))).mean()
        return [max(0.0, float(meanVal))] * periods, None, None


@st.cache_data(show_spinner=False)
def computeAllForecasts(
    df: pd.DataFrame,
    monthCols: list[str],
    periods: int = 12,
) -> tuple[dict, dict, dict]:
    forecastTotals: dict     = {}
    forecastSeriesDict: dict = {}
    metricsDict: dict        = {}
    discontinuedCount        = 0
    shortDataCount           = 0

    progress = st.progress(0.0, text="Memproses Holt-Winters...")
    n = len(df)

    for i, idx in enumerate(df.index):
        row = df.loc[idx]

        if isDiscontinued(row, monthCols):
            discontinuedCount      += 1
            forecastTotals[idx]     = 0.0
            forecastSeriesDict[idx] = []
            metricsDict[idx]        = {
                "rmse": None, "mape": None,
                "discontinued": True, "skip_reason": "no_sales_2025",
            }

        elif not hasEnoughConsecutive(row, monthCols, minConsec=24):
            shortDataCount         += 1
            forecastTotals[idx]     = 0.0
            forecastSeriesDict[idx] = []
            metricsDict[idx]        = {
                "rmse": None, "mape": None,
                "discontinued": True, "skip_reason": "consec_lt_24",
            }

        else:
            sales = tuple(row[monthCols].values.tolist())
            fc, rmse, mape          = forecastOne(sales, periods=periods, holdoutMonths=12)
            forecastTotals[idx]     = float(sum(fc))
            forecastSeriesDict[idx] = fc
            metricsDict[idx]        = {
                "rmse": rmse, "mape": mape,
                "discontinued": False, "skip_reason": None,
            }

        if i % max(1, n // 30) == 0:
            progress.progress((i + 1) / n, text=f"Memproses... {i+1}/{n}")

    progress.empty()

    msgs = []
    if discontinuedCount > 0:
        msgs.append(
            f"**{discontinuedCount} model** tidak ada penjualan di {ACTIVE_YEAR} "
            f"(discontinue)."
        )
    if shortDataCount > 0:
        msgs.append(
            f"**{shortDataCount} model** punya data berturut-turut < 24 bulan "
            f"→ di-skip dari forecast & metrik akurasi."
        )

    return forecastTotals, forecastSeriesDict, metricsDict

def calcEoq(demand, orderingCost, holdingCost):
    if holdingCost <= 0 or demand <= 0:
        return 0
    return int(np.ceil(np.sqrt((2 * demand * orderingCost) / holdingCost)))


def calcDemand(targetPerBulan, freqPerTahun):
    return int(targetPerBulan * 12 * freqPerTahun)


def mapeColorClass(mape: float | None) -> str:
    if mape is None:
        return ""
    if mape < 10:
        return "metric-good"
    if mape < 20:
        return "metric-warn"
    return "metric-bad"

# ─── SIDEBAR BAGIAN ATAS (tidak bergantung data) ──────────────────────────────
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
        "Kapasitas Total Bengkel per Bulan (kendaraan)",
        min_value=1, max_value=10000, value=200, step=10,
        help=(
            "Total kendaraan yang mampu diservis bengkel per bulan. "
            "Permintaan suku cadang tiap model dialokasikan secara proporsional "
            "terhadap skor prioritas Fuzzy AHP."
        ),
    )
    biayaPesan = st.number_input(
        "Biaya Sekali Pesan (Rp)",
        min_value=0, value=5_000, step=1_000,
        help=(
            "Biaya tetap setiap kali melakukan pemesanan suku cadang, "
            "terlepas dari jumlah unit yang dipesan."
        ),
    )
    biayaSimpan = st.number_input(
        "Biaya Simpan per Unit per Tahun (Rp)",
        min_value=0, value=5_000, step=1_000,
        help=(
            "Biaya penyimpanan per unit suku cadang per tahun."
        )
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

# ─── SIDEBAR FILTER TAHUN (bergantung data) ───────────────────────────────────
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

# ─── FILTER ROWS: hanya model yang punya penjualan di rentang tahun terpilih ──
yearFilterCols = [
    c for c in monthCols
    if yearRange[0] <= int(c.split("_")[1]) <= yearRange[1]
]

activeInRange = dfFull[yearFilterCols].sum(axis=1) > 0
df = dfFull[activeInRange].reset_index(drop=True)

if len(df) == 0:
    st.warning(
        f"Tidak ada model dengan penjualan di tahun {yearRange[0]}–{yearRange[1]}. "
        "Coba perlebar rentang tahun di sidebar."
    )
    st.stop()

# Tampilkan info filter jika tidak semua tahun dipilih
if yearRange != (allYears[0], allYears[-1]):
    filteredOut = len(dfFull) - len(df)
    st.info(
        f"Filter aktif: **{yearRange[0]}–{yearRange[1]}** — "
        f"menampilkan **{len(df)} model** "
        f"({filteredOut} model disembunyikan karena tidak aktif di rentang ini)."
    )

# ─── REBUILD modelToDfIdx setelah filter ──────────────────────────────────────
modelToDfIdx = {
    (df.loc[i, "BRAND"], df.loc[i, "MODEL"]): i for i in df.index
}

# ─── FORECAST (pakai df yang sudah difilter) ──────────────────────────────────
forecastTotals    = None
forecastSeriesDict: dict = {}
metricsDict: dict        = {}

if enableForecast:
    forecastTotals, forecastSeriesDict, metricsDict = computeAllForecasts(
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
    f'&nbsp;&nbsp;<span style="color:#6b7280;font-size:0.85rem">'
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

tabRank, tabFahp, tabForecast, tabEoq = st.tabs([
    "Ranking Prioritas",
    "Detail Fuzzy AHP",
    "Visualisasi Tren & Akurasi",
    "Rekomendasi EOQ",
])


with tabRank:
    st.markdown("<h3>Tabel Ranking Model Kendaraan</h3>", unsafe_allow_html=True)

    colA, colB = st.columns([3, 1])
    with colA:
        topN = st.slider("Tampilkan Top-N", 5, min(100, len(rankedDf)), 20)
    with colB:
        brandFilter = st.multiselect(
            "Filter Merek",
            options=sorted(rankedDf["Brand"].unique()),
            default=[],
        )

    displayDf = rankedDf.copy()
    if brandFilter:
        displayDf = displayDf[displayDf["Brand"].isin(brandFilter)]

    showCols  = ["Brand", "Model", "CC", "C1_total", "C2_cc", "C3_gvw", "C4_hp"]
    renameMap = {
        "Brand":    "Merek",
        "Model":    "Model",
        "CC":       "CC",
        "C1_total": "Tren Populasi",
        "C2_cc":    "CC (C2)",
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



with tabFahp:
    st.markdown("<h3>Detail Perhitungan Fuzzy AHP (4 Kriteria)</h3>", unsafe_allow_html=True)
    st.caption("Chang's Extent Analysis — 4×4 matrix")

    st.markdown("**Matriks Perbandingan Berpasangan (nilai tengah TFN)**")
    nCrit = 4
    tableData = {}
    for i, ci in enumerate(CRITERIA_NAMES):
        row = {}
        for j, cj in enumerate(CRITERIA_NAMES):
            m = float(matrix[i][j][1])
            row[cj] = f"{m:.3f}" if m != 1.0 else "1.000"
        tableData[ci] = row
    st.dataframe(pd.DataFrame(tableData).T, use_container_width=False)

    st.divider()

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Nilai Sintesis Fuzzy (Si)**")
        Si = fahpDebug["Si"]
        siRows = [
            {"Kriteria": CRITERIA_NAMES[i], "l": round(l, 6), "m": round(m, 6), "u": round(u, 6)}
            for i, (l, m, u) in enumerate(Si)
        ]
        st.dataframe(pd.DataFrame(siRows).set_index("Kriteria"), use_container_width=True)

    with colR:
        st.markdown("**Bobot Akhir (W)**")
        wRows = [
            {"Kriteria": CRITERIA_NAMES[i], "Bobot": round(float(w), 6), "Persen": f"{float(w)*100:.2f}%"}
            for i, w in enumerate(weights)
        ]
        st.dataframe(pd.DataFrame(wRows).set_index("Kriteria"), use_container_width=True)

    st.divider()
    st.markdown("**Matriks Degree of Possibility V(Si ≥ Sj)**")
    V = fahpDebug["V_matrix"]
    vDf = pd.DataFrame(V.round(6), index=CRITERIA_NAMES, columns=CRITERIA_NAMES)
    st.dataframe(vDf, use_container_width=False)

    st.divider()
    st.markdown("**Uji Konsistensi**")
    ciVal = (lambdaMax - nCrit) / (nCrit - 1)
    riVal = RI_TABLE[nCrit]
    statusStr = "Konsisten ✓" if crOk else "Tidak Konsisten ✗"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("lambda_max", f"{lambdaMax:.4f}")
    col2.metric("CI",         f"{ciVal:.4f}")
    col3.metric("RI (n=4)",   f"{riVal}")
    col4.metric("CR", f"{cr:.4f}", delta=statusStr, delta_color="normal" if crOk else "inverse")

    figW = go.Figure(go.Bar(
        x=CRITERIA_NAMES,
        y=[float(w) * 100 for w in weights],
        text=[f"{float(w)*100:.1f}%" for w in weights],
        textposition="auto",
        marker_color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"],
    ))
    figW.update_layout(
        title="Visualisasi Bobot Kriteria",
        yaxis_title="Bobot (%)",
        height=350,
        showlegend=False,
    )
    st.plotly_chart(figW, use_container_width=True)

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

    if selected:
        fig = go.Figure()
        selectedMetrics = []

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

            historical = df.loc[dfIdx, monthCols].values
            xHist      = list(range(len(historical)))
            isDisc     = metricsDict.get(dfIdx, {}).get("discontinued", False)
            modelLabel = f"{brand} {model}"

            lineColor  = "#9ca3af" if isDisc else None
            fig.add_trace(go.Scatter(
                x=xHist,
                y=historical,
                mode="lines",
                name=f"{modelLabel} (Historis)" + (" — discontinue" if isDisc else ""),
                line=dict(width=2, color=lineColor),
            ))

            if enableForecast and not isDisc and dfIdx in forecastSeriesDict and forecastSeriesDict[dfIdx]:
                fc   = forecastSeriesDict[dfIdx]
                xFc  = list(range(len(historical), len(historical) + len(fc)))
                fig.add_trace(go.Scatter(
                    x=[xHist[-1]] + xFc,
                    y=[float(historical[-1])] + list(fc),
                    mode="lines",
                    name=f"{modelLabel} (Forecast)",
                    line=dict(width=2, dash="dash"),
                ))

            if enableForecast and dfIdx in metricsDict:
                m = metricsDict[dfIdx]
                selectedMetrics.append({
                    "Model":        modelLabel,
                    "Status":       "Discontinue" if m["discontinued"] else "Aktif",
                    "RMSE":         f"{m['rmse']:,.1f}" if m["rmse"] is not None else "—",
                    "MAPE (%)":     f"{m['mape']:.2f}%" if m["mape"] is not None else "—",
                })

        fig.update_layout(
            title="Penjualan Bulanan (garis abu = discontinue, garis putus = forecast)",
            xaxis_title="Periode (bulan ke-)",
            yaxis_title="Volume Penjualan (unit)",
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        if enableForecast and selectedMetrics:
            st.markdown("**Metrik Akurasi Model yang Dipilih**")
            st.caption(
                "Model discontinue tidak di-forecast sehingga tidak memiliki metrik."
            )
            st.dataframe(
                pd.DataFrame(selectedMetrics).set_index("Model"),
                use_container_width=True,
            )

    else:
        st.info("Pilih minimal satu model di atas untuk melihat grafik tren.")

with tabEoq:
    st.markdown("<h3>Rekomendasi Pembelian Suku Cadang Fast-Moving</h3>", unsafe_allow_html=True)

    komponen = {
        "Filter Oli":   2.0,
        "Oli Mesin":    2.0,
        "Filter Udara": 1.0,
        "Kampas Rem":   0.5,
        "Busi":         0.5,
    }

    topNEoq = st.slider("Hitung EOQ untuk Top-N model", 1, 10, 5)
    topEoqDf = rankedDf.head(topNEoq).copy()

    scoreMin = float(topEoqDf["score"].min())
    scoreMax = float(topEoqDf["score"].max())
    scoreDiff = scoreMax - scoreMin

    def normalizeScore(s):
        if scoreDiff == 0:
            return 0.5
        return (float(s) - scoreMin) / scoreDiff

    eoqRows = []
    for rank, row in topEoqDf.iterrows():
        norm = normalizeScore(row["score"])
        for compName, freq in komponen.items():
            demand = int(round(targetServis * freq * (1 + norm) * 12))
            eoq    = calcEoq(demand, biayaPesan, biayaSimpan)
            tic    = (demand / max(eoq, 1)) * biayaPesan + (eoq / 2) * biayaSimpan
            eoqRows.append({
                "Rank":              rank,
                "Model":             f"{row['Brand']} {row['Model']}",
                "Komponen":          compName,
                "Demand/Tahun":      demand,
                "EOQ (pcs)":         eoq,
                "Total Biaya/Tahun": f"Rp {tic:,.0f}",
            })

    eoqDf = pd.DataFrame(eoqRows)
    st.dataframe(eoqDf, use_container_width=True, hide_index=True)

st.divider()
st.caption("2026 — Sistem Rekomendasi Suku Cadang Fast Moving — Kelvyn — Skripsi S1 Teknik Informatika")