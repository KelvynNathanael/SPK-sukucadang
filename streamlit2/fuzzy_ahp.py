import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional

def isMonthCol(col: str) -> bool:
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    parts = col.split("_")
    return len(parts) == 2 and parts[0] in months and parts[1].isdigit()

def loadAndClean(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

    monthCols = [c for c in df.columns if isMonthCol(c)]
    if not monthCols:
        raise ValueError(
            "Tidak ada kolom bulan ditemukan (format: JAN_2012, FEB_2012, ...)"
        )

    for col in monthCols:
        df[col] = pd.to_numeric(
            df[col].astype(str)
                   .str.replace(",", "")
                   .str.replace("-", "0")
                   .str.strip(),
            errors="coerce",
        ).fillna(0).clip(lower=0)

    heavyKeywords = ["TRUCK", "BUS", "HEAVY", "TRUK", "PICK UP", "PICKUP"]
    if "CATEGORYTYPE" in df.columns:
        maskHeavy = df["CATEGORYTYPE"].str.upper().str.contains(
            "|".join(heavyKeywords), na=False
        )
        df = df[~maskHeavy].reset_index(drop=True)

    df["CC"] = pd.to_numeric(df["CC"], errors="coerce")
    df = df.dropna(subset=["CC"])
    df = df[df["CC"] > 0].reset_index(drop=True)

    df["totalSales"] = df[monthCols].sum(axis=1)
    df["bulanAktif"] = (df[monthCols] > 0).sum(axis=1)
    df = df[df["bulanAktif"] >= 24].reset_index(drop=True)

    return df, monthCols

def findCol(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    for col in df.columns:
        upper = col.upper()
        if any(kw in upper for kw in keywords):
            return col
    return None


def buildCriteriaTable(
    df: pd.DataFrame,
    monthCols: List[str],
    forecastTotals: Optional[Dict] = None,
) -> pd.DataFrame:
    result = df[["BRAND", "MODEL", "CC", "bulanAktif"]].copy()
    result = result.rename(columns={"BRAND": "Brand", "MODEL": "Model"})

    # C1 - Tren populasi (historis +/- forecast)
    filteredTotal = df[monthCols].sum(axis=1)
    if forecastTotals:
        result["C1_total"] = result.index.map(
            lambda i: filteredTotal[i] + forecastTotals.get(i, 0)
        )
    else:
        result["C1_total"] = filteredTotal

    # C2 - Kapasitas Mesin (CC)
    result["C2_cc"] = result["CC"]

    # C3 - Gross Vehicle Weight (GVW)
    gvwCol = findCol(df, ["GVW", "GROSS_VEHICLE", "GROSS VEHICLE", "BERAT"])
    if gvwCol:
        result["C3_gvw"] = (
            pd.to_numeric(df[gvwCol], errors="coerce").fillna(0).clip(lower=0)
        )
        c3Available = True
    else:
        result["C3_gvw"] = 0.0
        c3Available = False

    # C4 - Horse Power (HP / PS / KW)
    hpCol = findCol(df, ["HP", "HORSE", "POWER", " PS", "_PS", "KW"])
    if hpCol:
        result["C4_hp"] = (
            pd.to_numeric(df[hpCol], errors="coerce").fillna(0).clip(lower=0)
        )
        c4Available = True
    else:
        result["C4_hp"] = 0.0
        c4Available = False

    result.attrs["c3_available"] = c3Available
    result.attrs["c4_available"] = c4Available

    return result[
        ["Brand", "Model", "CC", "C1_total", "C2_cc", "C3_gvw", "C4_hp", "bulanAktif"]
    ]

TFN_SCALE: Dict[int, Tuple[float, float, float]] = {
    1: (1.0, 1.0, 1.0),
    3: (1.0, 3.0, 5.0),
    5: (3.0, 5.0, 7.0),
    7: (5.0, 7.0, 9.0),
    9: (7.0, 9.0, 9.0),
}

LINGUISTIC_LABEL: Dict[int, str] = {
    1: "Sama Penting",
    3: "Sedikit Lebih Penting",
    5: "Cukup Lebih Penting",
    7: "Sangat Lebih Penting",
    9: "Mutlak Lebih Penting",
}

RI_TABLE: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
    5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.49,
}


def getTfn(value: int) -> Tuple[float, float, float]:
    if value == 0:
        raise ValueError("Nilai perbandingan tidak boleh 0")
    absVal = abs(value)
    validScales = [1, 3, 5, 7, 9]
    closest = min(validScales, key=lambda x: abs(x - absVal))
    l, m, u = TFN_SCALE[closest]
    if value < 0:
        return (round(1.0 / u, 6), round(1.0 / m, 6), round(1.0 / l, 6))
    return (l, m, u)

def buildPairwiseMatrix(
    pairwiseValues: Dict[Tuple[int, int], int],
    n: int = 4,
    verbose: bool = True,
) -> np.ndarray:
    matrix = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = (1.0, 1.0, 1.0)
            elif i < j:
                v = pairwiseValues.get((i, j), 1)
                matrix[i][j] = getTfn(v)
            else:
                v = pairwiseValues.get((j, i), 1)
                matrix[i][j] = getTfn(-v)
    return matrix

def checkConsistencyRatio(
    matrix: np.ndarray,
    verbose: bool = True,
) -> Tuple[float, float]:
    n = matrix.shape[0]
    crisp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            crisp[i][j] = float(matrix[i][j][1])

    colSums = crisp.sum(axis=0)
    normMatrix = crisp / colSums
    weights = normMatrix.mean(axis=1)

    weightedSum = crisp @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        lambdaVec = np.where(weights > 1e-12, weightedSum / weights, 0.0)
    lambdaMax = lambdaVec[weights > 1e-12].mean()

    ci = (lambdaMax - n) / max(n - 1, 1)
    ri = RI_TABLE.get(n, 1.49)
    cr = 0.0 if ri == 0 else ci / ri

    if verbose:
        print(f"lambdaMax={lambdaMax:.4f}  CI={ci:.4f}  RI={ri}  CR={cr:.4f}")

    return lambdaMax, cr

def fuzzyExtentAnalysis(
    matrix: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    n = matrix.shape[0]

    # Step 1: Fuzzy synthetic extent Si
    rowSums = []
    for i in range(n):
        l = sum(float(matrix[i][j][0]) for j in range(n))
        m = sum(float(matrix[i][j][1]) for j in range(n))
        u = sum(float(matrix[i][j][2]) for j in range(n))
        rowSums.append((l, m, u))

    totalL = sum(rs[0] for rs in rowSums)
    totalM = sum(rs[1] for rs in rowSums)
    totalU = sum(rs[2] for rs in rowSums)

    Si = [
        (
            round(rl / totalU, 6),
            round(rm / totalM, 6),
            round(ru / totalL, 6),
        )
        for (rl, rm, ru) in rowSums
    ]

    # Step 2: Degree of possibility V(Si >= Sj)
    def degreeOfPossibility(M1: tuple, M2: tuple) -> float:
        l1, m1, u1 = M1
        l2, m2, u2 = M2
        if m2 >= m1:
            return 1.0
        if l1 >= u2:
            return 0.0
        denom = (m2 - u2) - (m1 - l1)
        if abs(denom) < 1e-12:
            return 0.0
        return (l1 - u2) / denom

    # vMatrix[i][j] = V(Sj >= Si) -> how much Sj dominates Si
    vMatrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                vMatrix[i][j] = 1.0
            else:
                vMatrix[i][j] = round(degreeOfPossibility(Si[j], Si[i]), 6)

    # Step 3: Raw weight = min V(Si >= Sk) for all k != i
    rawWeights = np.zeros(n)
    for i in range(n):
        others = [vMatrix[i][j] for j in range(n) if j != i]
        rawWeights[i] = min(others) if others else 1.0

    # Step 4: Normalise
    totalW = rawWeights.sum()
    weights = rawWeights / totalW if totalW > 0 else np.full(n, 1.0 / n)

    debug = {
        "Si": Si,
        "V_matrix": vMatrix,
        "raw_weights": rawWeights,
        "weights": weights,
    }

    if verbose:
        print("Weights:", weights)

    return weights, debug


def minmaxNormalize(df: pd.DataFrame, col: str) -> pd.Series:
    mn = df[col].min()
    mx = df[col].max()
    if mx == mn:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return (df[col] - mn) / (mx - mn)


def scoreAndRank(
    criteriadf: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    result = criteriadf.copy()

    result["C1_norm"] = minmaxNormalize(result, "C1_total").round(6)
    result["C2_norm"] = minmaxNormalize(result, "C2_cc").round(6)
    result["C3_norm"] = minmaxNormalize(result, "C3_gvw").round(6)
    result["C4_norm"] = minmaxNormalize(result, "C4_hp").round(6)

    result["score"] = (
        weights[0] * result["C1_norm"]
        + weights[1] * result["C2_norm"]
        + weights[2] * result["C3_norm"]
        + weights[3] * result["C4_norm"]
    ).round(6)

    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    result.index += 1
    result.index.name = "Rank"
    return result

def findInconsistencies(
    pairwiseValues: Dict[Tuple[int, int], int],
    weights: np.ndarray,
    matrix: np.ndarray,
    criteriaNames: list,
) -> Dict:
    n = len(criteriaNames)
    direction: Dict[Tuple[int, int], int] = {}
    for (i, j), v in pairwiseValues.items():
        if v > 1:
            direction[(i, j)] = 1
            direction[(j, i)] = -1
        elif v < -1:
            direction[(i, j)] = -1
            direction[(j, i)] = 1
        else:
            direction[(i, j)] = 0
            direction[(j, i)] = 0
            
    transitivityViolations = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                dAb = direction.get((a, b), 0)
                dBc = direction.get((b, c), 0)
                dAc = direction.get((a, c), 0)
                if dAb == 1 and dBc == 1 and dAc == -1:
                    transitivityViolations.append({
                        "a": a, "b": b, "c": c,
                        "label_a": criteriaNames[a],
                        "label_b": criteriaNames[b],
                        "label_c": criteriaNames[c],
                        "explanation": (
                            f"Kamu bilang **{criteriaNames[a]} > {criteriaNames[b]}** "
                            f"dan **{criteriaNames[b]} > {criteriaNames[c]}**, "
                            f"tapi juga **{criteriaNames[c]} > {criteriaNames[a]}** "
                            f"- ini kontradiksi logika!"
                        ),
                    })

    seen: set = set()
    uniqueViolations = []
    for v in transitivityViolations:
        key = tuple(sorted([v["a"], v["b"], v["c"]]))
        if key not in seen:
            seen.add(key)
            uniqueViolations.append(v)

    crisp = np.array([[float(matrix[i][j][1]) for j in range(n)] for i in range(n)])

    colSums = crisp.sum(axis=0)
    colSumsSafe = np.where(colSums > 1e-12, colSums, 1.0)
    crispNorm = crisp / colSumsSafe
    crispWeights = crispNorm.mean(axis=1)

    deviations = []
    skippedPairs: List[str] = []

    for (i, j) in pairwiseValues.keys():
        wI, wJ = float(crispWeights[i]), float(crispWeights[j])

        if wI < 1e-6 and wJ < 1e-6:
            skippedPairs.append(f"{criteriaNames[i]} vs {criteriaNames[j]}")
            continue

        wJSafe = max(wJ, 1e-6)
        actual  = crisp[i][j]
        implied = wI / wJSafe
        dev     = abs(actual - implied)
        relativeDev = dev / max(implied, 0.01)

        deviations.append({
            "i": i, "j": j,
            "label_i": criteriaNames[i],
            "label_j": criteriaNames[j],
            "actual":             round(actual, 3),
            "implied":            round(implied, 3),
            "deviation":          round(dev, 4),
            "relative_deviation": round(relativeDev, 4),
        })

    deviations.sort(key=lambda x: -x["relative_deviation"])

    return {
        "transitivity_violations": uniqueViolations,
        "top_deviations":          deviations[:3],
        "skipped_pairs":           skippedPairs,
    }