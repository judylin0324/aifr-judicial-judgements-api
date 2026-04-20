"""
裁判書量化實證研究 — FastAPI 後端
載入 CSV 資料，提供篩選、統計、圖表資料的 API。
"""
import os, math, re
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ════════════════════════════════════════════════════════════
#  App Setup
# ════════════════════════════════════════════════════════════
app = FastAPI(title="裁判書量化實證研究 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════
#  Data Loading
# ════════════════════════════════════════════════════════════
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "Data")))

# Registry: case_type_key -> { label, file_pattern, category }
CASE_TYPES = {
    "criminal_litigation": {
        "label": "刑事訴訟", "category": "criminal",
        "file_pattern": "刑事訴訟",
    },
    "civil_litigation": {
        "label": "民事訴訟", "category": "civil",
        "file_pattern": "民事訴訟",
    },
    "civil_nonlitig": {
        "label": "民事非訟", "category": "civil",
        "file_pattern": "民事非訟",
    },
    "family_litigation": {
        "label": "家事訴訟", "category": "family",
        "file_pattern": "家事訴訟",
    },
}

# In-memory store: key -> DataFrame
DATA: dict[str, pd.DataFrame] = {}


def _find_csv(pattern: str) -> Optional[Path]:
    """Find first CSV matching pattern in DATA_DIR."""
    for f in sorted(DATA_DIR.glob("*.csv")):
        if pattern in f.stem:
            return f
    return None


def _load_all():
    """Load all available CSV files into memory at startup."""
    for key, info in CASE_TYPES.items():
        csv_path = _find_csv(info["file_pattern"])
        if csv_path and csv_path.exists():
            print(f"Loading {key}: {csv_path.name} ...")
            df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, low_memory=False)
            df = df.fillna("")
            DATA[key] = df
            print(f"  → {len(df)} rows, {len(df.columns)} cols")
        else:
            print(f"Warning: no CSV found for {key} (pattern: {info['file_pattern']})")


@app.on_event("startup")
async def startup():
    _load_all()


# ════════════════════════════════════════════════════════════
#  Utility Functions
# ════════════════════════════════════════════════════════════
def clean(v):
    return str(v).strip() if pd.notna(v) else ""


def split_pipe(v):
    return [x.strip() for x in str(v).split("|") if x.strip()]


def parse_months(s):
    """Parse YYMM string to total months."""
    v = re.sub(r"\D", "", str(s))
    if not v:
        return None
    if len(v) >= 4:
        y, m = int(v[:-2]) or 0, int(v[-2:]) or 0
    elif len(v) == 3:
        y, m = int(v[0]) or 0, int(v[1:]) or 0
    else:
        m = int(v) or 0
        y = 0
    return y * 12 + m


def parse_int_loose(v):
    pure = re.sub(r"[^\d-]", "", str(v))
    if not pure:
        return None
    try:
        return int(pure)
    except ValueError:
        return None


def top_n(values, n=10):
    c = Counter(values)
    return [k for k, _ in c.most_common(n)]


def quantile(sorted_arr, q):
    if not sorted_arr:
        return 0
    p = (len(sorted_arr) - 1) * q
    b = int(p)
    if b + 1 < len(sorted_arr):
        return sorted_arr[b] + (p - b) * (sorted_arr[b + 1] - sorted_arr[b])
    return sorted_arr[b]


def box_stats(values):
    s = sorted(values)
    if not s:
        return None
    q1 = quantile(s, 0.25)
    median = quantile(s, 0.5)
    q3 = quantile(s, 0.75)
    iqr = q3 - q1
    lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = [v for v in s if lf <= v <= uf]
    return {
        "q1": q1, "median": median, "q3": q3, "iqr": iqr,
        "whiskerLow": inliers[0] if inliers else s[0],
        "whiskerHigh": inliers[-1] if inliers else s[-1],
        "outliers": [v for v in s if v < lf or v > uf],
        "min": s[0], "max": s[-1],
    }


# ════════════════════════════════════════════════════════════
#  Stacked Bar Helper
# ════════════════════════════════════════════════════════════
def _build_stacked_bar(df, group_col, segment_col, top_n=8):
    """Build stacked bar data: top N groups, each broken into segment proportions."""
    filtered = df[(df[group_col].str.strip() != "") & (df[segment_col].str.strip() != "")]
    if filtered.empty:
        return {"data": [], "segments": []}

    segments = [s for s, _ in Counter(filtered[segment_col]).most_common()]
    top_groups = [g for g, _ in Counter(filtered[group_col]).most_common(top_n)]

    data = []
    for group in top_groups:
        sub = filtered[filtered[group_col] == group]
        total = len(sub)
        cats = sub[segment_col].value_counts().to_dict()
        row = {"name": group, "__total": total, "__counts": {s: cats.get(s, 0) for s in segments}}
        acc = 0
        for i, s in enumerate(segments):
            if i < len(segments) - 1:
                t = round((cats.get(s, 0) / total) * 1000) if total else 0
                row[s] = t
                acc += t
            else:
                row[s] = 1000 - acc
        data.append(row)

    return {"data": data, "segments": segments}


# ════════════════════════════════════════════════════════════
#  Common Filter Application
# ════════════════════════════════════════════════════════════
def get_ym_key(row):
    y = clean(row.get("終結年", row.get("c0_全案終結日期-年", "")))
    m = clean(row.get("終結月", row.get("c0_全案終結日期-月", "")))
    if y and m:
        return f"{y.zfill(3)}/{m.zfill(2)}"
    return None


def apply_common_filters(df, params: dict) -> pd.DataFrame:
    """Apply common filters that exist across all case types."""
    result = df

    # Court filter
    court = params.get("court")
    if court:
        courts = court.split(",")
        col = "c0_法院別" if "c0_法院別" in result.columns else "法院別"
        result = result[result[col].isin(courts)]

    # Year-month range
    ym_min = params.get("ym_min", "")
    ym_max = params.get("ym_max", "")
    if ym_min or ym_max:
        def ym_filter(row):
            ym = get_ym_key(row)
            if not ym:
                return False
            if ym_min and ym < ym_min:
                return False
            if ym_max and ym > ym_max:
                return False
            return True
        mask = result.apply(ym_filter, axis=1)
        result = result[mask]

    return result


# ════════════════════════════════════════════════════════════
#  Criminal Litigation Specific
# ════════════════════════════════════════════════════════════
CRIME_FLAG_KEYS = {
    "褫奪公權": "c11_褫奪公權", "少年犯": "c11_少年犯",
    "幫助犯": "c11_幫助犯", "未遂犯": "c11_未遂犯", "家庭暴力": "c11_家庭暴力",
}
AG_MIT_CATS = ["無加重無減輕", "僅有加重法條", "僅有減輕法條", "有加重有減輕"]

# Column name aliases (classified CSV may use different names than Vue v2 expected)
def _col(df, *candidates):
    """Return the first column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return candidates[0]  # fallback


def get_ag_mit_cat(row):
    # Try multiple column names for aggravation/mitigation
    ha = bool(split_pipe(row.get("c111_量刑加重", row.get("量刑加重", ""))))
    hm = bool(split_pipe(row.get("c112_量刑減輕", row.get("量刑減輕", ""))))
    if ha and hm:
        return "有加重有減輕"
    if ha:
        return "僅有加重法條"
    if hm:
        return "僅有減輕法條"
    return "無加重無減輕"


def get_ag_mit_cat_vec(df):
    """Vectorized version of get_ag_mit_cat for performance."""
    agg_col = _col(df, "c111_量刑加重", "量刑加重")
    mit_col = _col(df, "c112_量刑減輕", "量刑減輕")
    ha = df[agg_col].str.strip().ne("") if agg_col in df.columns else pd.Series(False, index=df.index)
    hm = df[mit_col].str.strip().ne("") if mit_col in df.columns else pd.Series(False, index=df.index)
    result = pd.Series("無加重無減輕", index=df.index)
    result[ha & ~hm] = "僅有加重法條"
    result[~ha & hm] = "僅有減輕法條"
    result[ha & hm] = "有加重有減輕"
    return result


def apply_criminal_filters(df, params):
    """Apply criminal-litigation-specific filters."""
    result = apply_common_filters(df, params)

    # Case classification
    cls_filter = params.get("cls")
    if cls_filter:
        result = result[result["案件分類"].isin(cls_filter.split(","))]

    # Ending
    end_filter = params.get("ending")
    if end_filter:
        result = result[result["c0_全案終結情形"].isin(end_filter.split(","))]

    # Defense
    def_filter = params.get("defense")
    if def_filter:
        result = result[result["c1_辯護及代理"].isin(def_filter.split(","))]

    # Procedure
    proc_filter = params.get("procedure")
    if proc_filter:
        result = result[result["c1_裁判程序"].isin(proc_filter.split(","))]

    # Probation
    prb_filter = params.get("probation")
    if prb_filter:
        prb_col = "c1_是否宣告緩刑" if "c1_是否宣告緩刑" in result.columns else "c1_宣告緩刑"
        if prb_col in result.columns:
            vals = prb_filter.split(",")
            mask = pd.Series(False, index=result.index)
            if "有緩刑" in vals:
                mask |= result[prb_col] == "1"
            if "無緩刑" in vals:
                mask |= result[prb_col] == "0"
            result = result[mask]

    # Recidivist
    rec_filter = params.get("recidivist")
    if rec_filter:
        rec_col = "c1_是否為累犯" if "c1_是否為累犯" in result.columns else "c1_累犯"
        if rec_col in result.columns:
            vals = rec_filter.split(",")
            mask = pd.Series(False, index=result.index)
            if "累犯" in vals:
                mask |= result[rec_col] == "1"
            if "非累犯" in vals:
                mask |= result[rec_col] == "0"
            result = result[mask]

    # Article (定罪法條)
    art_filter = params.get("article")
    if art_filter:
        result = result[result["定罪法條"].isin(art_filter.split(","))]

    # Result (罪名裁判結果)
    res_filter = params.get("result")
    if res_filter:
        result = result[result["罪名裁判結果"].isin(res_filter.split(","))]

    return result


def criminal_filter_options(df):
    """Get all unique filter option values for criminal litigation."""
    def count_col(col):
        if col not in df.columns:
            return []
        return [{"val": k, "count": int(v)} for k, v in
                df[col].value_counts().items() if clean(k)]

    ym_set = set()
    y_col = "終結年" if "終結年" in df.columns else "c0_全案終結日期-年"
    m_col = "終結月" if "終結月" in df.columns else "c0_全案終結日期-月"
    for _, row in df[[y_col, m_col]].drop_duplicates().iterrows():
        y, m = clean(row[y_col]), clean(row[m_col])
        if y and m:
            ym_set.add(f"{y.zfill(3)}/{m.zfill(2)}")

    # Probation (column may be c1_宣告緩刑 or c1_是否宣告緩刑)
    prb = {}
    prb_col = "c1_是否宣告緩刑" if "c1_是否宣告緩刑" in df.columns else "c1_宣告緩刑"
    if prb_col in df.columns:
        prb_counts = df[prb_col].value_counts()
        if "1" in prb_counts:
            prb["有緩刑"] = int(prb_counts["1"])
        if "0" in prb_counts:
            prb["無緩刑"] = int(prb_counts["0"])

    # Recidivist (column may be c1_累犯 or c1_是否為累犯)
    rec = {}
    rec_col = "c1_是否為累犯" if "c1_是否為累犯" in df.columns else "c1_累犯"
    if rec_col in df.columns:
        rec_counts = df[rec_col].value_counts()
        if "1" in rec_counts:
            rec["累犯"] = int(rec_counts["1"])
        if "0" in rec_counts:
            rec["非累犯"] = int(rec_counts["0"])

    # Aggravation / Mitigation
    agg_counter, mit_counter = Counter(), Counter()
    agg_col = _col(df, "c111_量刑加重", "量刑加重")
    mit_col = _col(df, "c112_量刑減輕", "量刑減輕")
    if agg_col in df.columns:
        for v in df[agg_col]:
            for item in split_pipe(v):
                agg_counter[item] += 1
    if mit_col in df.columns:
        for v in df[mit_col]:
            for item in split_pipe(v):
                mit_counter[item] += 1

    return {
        "classes": count_col("案件分類"),
        "courts": count_col("c0_法院別"),
        "endings": count_col("c0_全案終結情形"),
        "results": count_col("罪名裁判結果"),
        "articles": count_col("定罪法條"),
        "defs": count_col("c1_辯護及代理"),
        "procs": count_col("c1_裁判程序"),
        "ym": sorted(ym_set),
        "probs": [{"val": k, "count": v} for k, v in prb.items()],
        "recid": [{"val": k, "count": v} for k, v in rec.items()],
        "aggr": [{"val": k, "count": int(v)} for k, v in agg_counter.most_common()],
        "miti": [{"val": k, "count": int(v)} for k, v in mit_counter.most_common()],
    }


def criminal_stats(df):
    """Compute summary statistics for criminal litigation."""
    jid_col = "裁判書ID" if "裁判書ID" in df.columns else "c0_裁判書ID"
    judgments = df[jid_col].nunique()

    # Use classified columns if available
    if "判決被告人數" in df.columns:
        # Each row is one defendant-crime pair; get unique judgments' defendant counts
        jdf = df.drop_duplicates(subset=[jid_col])
        defendants = int(jdf["判決被告人數"].apply(lambda x: parse_int_loose(x) or 0).sum())
    else:
        defendants = len(df)

    crime_count = len(df)
    law_count = df["定罪法條"].replace("", pd.NA).dropna().nunique() if "定罪法條" in df.columns else 0
    return {
        "judgments": int(judgments),
        "defendants": int(defendants),
        "crimes": int(crime_count),
        "uniqueLaws": int(law_count),
    }


def criminal_charts(df):
    """Build all chart data for criminal litigation."""
    # All rows are crime rows in the classified output
    crimes = df

    # Chart 1: Case structure heatmap
    judgments = df.drop_duplicates(subset=["裁判書ID"])
    jf = judgments[judgments["案件分類"].str.strip() != ""]
    heatmap = _build_heatmap(jf, "c0_全案終結情形", "案件分類")

    # Chart 2: Law stacked bar (ag/mit distribution) — vectorized
    law_arts = crimes[crimes["定罪法條"].str.strip() != ""]
    top_laws = [k for k, _ in Counter(law_arts["定罪法條"]).most_common(8)]
    # Compute ag/mit category for all crimes at once
    crimes_with_cat = law_arts.copy()
    crimes_with_cat["_agmit"] = get_ag_mit_cat_vec(crimes_with_cat)
    stack_data = []
    for law in top_laws:
        sub = crimes_with_cat[crimes_with_cat["定罪法條"] == law]
        cats = sub["_agmit"].value_counts().to_dict()
        total = len(sub)
        row = {"law": law, "__total": total, "__counts": {c: cats.get(c, 0) for c in AG_MIT_CATS}}
        acc = 0
        for i, c in enumerate(AG_MIT_CATS):
            if i < len(AG_MIT_CATS) - 1:
                t = round((cats.get(c, 0) / total) * 1000) if total else 0
                row[c] = t
                acc += t
            else:
                row[c] = 1000 - acc
        stack_data.append(row)

    # Chart 3 & 4: Violin + Box — vectorized
    # Pre-compute ag/mit category for all crimes once
    all_agmit = get_ag_mit_cat_vec(crimes)
    metrics = {}
    for metric_key, col, parse_fn, unit in [
        ("imprisonment", "c11_宣告有期徒刑", parse_months, "月"),
        ("detention", "c11_拘役日數", parse_int_loose, "日"),
        ("fine", "c11_罰金金額", parse_int_loose, "元"),
    ]:
        if col not in crimes.columns:
            metrics[metric_key] = {"violin": [], "box": [], "unit": unit}
            continue
        parsed = crimes[col].apply(parse_fn)
        mask = parsed.notna() & (crimes["定罪法條"].str.strip() != "")
        valid_idx = mask[mask].index
        valid_vals = parsed[valid_idx]
        valid_laws = crimes.loc[valid_idx, "定罪法條"]
        valid_agmit = all_agmit[valid_idx]

        violin_data = []
        box_data = []
        laws_for_metric = [k for k, _ in Counter(valid_laws).most_common(8)]

        for law in laws_for_metric:
            law_mask = valid_laws == law
            vals = sorted(valid_vals[law_mask].tolist())
            if not vals:
                continue
            mean_v = sum(vals) / len(vals)
            median_v = quantile(vals, 0.5)

            violin_data.append({
                "name": law, "values": vals,
                "mean": round(mean_v, 2), "median": round(median_v, 2), "n": len(vals),
            })

            bs = box_stats(vals)
            if bs:
                # Vectorized dominant ag/mit category
                law_agmit = valid_agmit[law_mask].value_counts()
                dominant = law_agmit.index[0] if not law_agmit.empty else AG_MIT_CATS[0]
                box_data.append({
                    "law": law, "n": len(vals), "dominant": dominant,
                    **{k: round(v, 2) if isinstance(v, float) else v for k, v in bs.items()},
                })

        metrics[metric_key] = {
            "violin": violin_data,
            "box": box_data,
            "unit": unit,
        }

    return {
        "heatmap": heatmap,
        "lawStack": {"data": stack_data, "segments": AG_MIT_CATS},
        "metrics": metrics,
        "topLaws": top_laws,
    }


# ════════════════════════════════════════════════════════════
#  Civil Litigation Specific
# ════════════════════════════════════════════════════════════
def apply_civil_filters(df, params):
    result = apply_common_filters(df, params)
    for col, param_key in [
        ("終結情形大分類", "ending"),
        ("案由大分類", "cause"),
        ("律師代理情形", "lawyer"),
        ("訴訟標的金額級距", "amount"),
    ]:
        val = params.get(param_key)
        if val and col in result.columns:
            result = result[result[col].isin(val.split(","))]

    # 國賠
    guo = params.get("national_comp")
    if guo and "是否國賠事件" in result.columns:
        result = result[result["是否國賠事件"] == guo]

    # Countersuit
    counter = params.get("countersuit")
    if counter and "是否反訴" in result.columns:
        result = result[result["是否反訴"] == counter]

    # Appeal
    appeal = params.get("appeal")
    if appeal and appeal in ["可上訴", "不可上訴"]:
        if appeal == "可上訴" and "c0_得上訴" in result.columns:
            result = result[result["c0_得上訴"].str.strip() != ""]
        elif appeal == "不可上訴" and "c0_不得上訴" in result.columns:
            result = result[result["c0_不得上訴"].str.strip() != ""]

    return result


def civil_filter_options(df):
    def count_col(col):
        if col not in df.columns:
            return []
        return [{"val": k, "count": int(v)} for k, v in
                df[col].value_counts().items() if clean(k)]

    ym_set = set()
    y_col = "c0_全案終結日期-年" if "c0_全案終結日期-年" in df.columns else "終結年"
    m_col = "c0_全案終結日期-月" if "c0_全案終結日期-月" in df.columns else "終結月"
    for _, row in df[[y_col, m_col]].drop_duplicates().iterrows():
        y, m = clean(row[y_col]), clean(row[m_col])
        if y and m:
            ym_set.add(f"{y.zfill(3)}/{m.zfill(2)}")

    # Build appeals options
    appeals = {}
    if "c0_得上訴" in df.columns:
        has_appeal = df[df["c0_得上訴"].str.strip() != ""].shape[0]
        if has_appeal > 0:
            appeals["可上訴"] = has_appeal
    if "c0_不得上訴" in df.columns:
        no_appeal = df[df["c0_不得上訴"].str.strip() != ""].shape[0]
        if no_appeal > 0:
            appeals["不可上訴"] = no_appeal

    return {
        "courts": count_col("法院別") or count_col("c0_法院別"),
        "endings": count_col("終結情形大分類"),
        "causes": count_col("案由大分類"),
        "lawyers": count_col("律師代理情形"),
        "amounts": count_col("訴訟標的金額級距"),
        "ym": sorted(ym_set),
        "nationalComp": count_col("是否國賠事件"),
        "countersuits": count_col("是否反訴"),
        "appeals": [{"val": k, "count": v} for k, v in appeals.items()],
    }


def civil_stats(df):
    jid_col = "裁判書ID" if "裁判書ID" in df.columns else "c0_裁判書ID"

    # Calculate average amount
    avg_amount = 0
    if "c0_訴訟標的金額" in df.columns:
        amounts = df["c0_訴訟標的金額"].apply(parse_int_loose)
        valid_amounts = amounts.dropna()
        if len(valid_amounts) > 0:
            avg_amount = round(valid_amounts.mean(), 0)

    # Calculate lawyer rate
    lawyer_rate = 0
    if "律師代理情形" in df.columns:
        with_lawyer = df[df["律師代理情形"].str.strip() != ""].shape[0]
        total = len(df)
        if total > 0:
            lawyer_rate = round((with_lawyer / total) * 100, 2)

    return {
        "judgments": int(df[jid_col].nunique()),
        "totalRows": len(df),
        "avgAmount": int(avg_amount),
        "lawyerRate": lawyer_rate,
    }


def civil_charts(df):
    """Build chart data for civil litigation."""
    # 1. Ending distribution (pie/bar)
    ending_counts = df["終結情形大分類"].value_counts().to_dict()
    ending_data = [{"name": k, "count": int(v)} for k, v in ending_counts.items() if clean(k)]

    # 2. Cause distribution
    cause_counts = df["案由大分類"].value_counts().to_dict()
    cause_data = [{"name": k, "count": int(v)} for k, v in cause_counts.items() if clean(k)]

    # 3. Lawyer representation × ending heatmap
    lawyer_heatmap = {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    if "律師代理情形" in df.columns:
        lawyer_heatmap = _build_heatmap(df, "終結情形大分類", "律師代理情形")

    # 4. Amount distribution × ending
    amount_heatmap = {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    if "訴訟標的金額級距" in df.columns:
        amount_heatmap = _build_heatmap(df, "終結情形大分類", "訴訟標的金額級距")

    # 5. Cause × Ending stacked bar
    cause_ending_stack = {"data": [], "segments": []}
    if "案由大分類" in df.columns and "終結情形大分類" in df.columns:
        cause_ending_stack = _build_stacked_bar(df, "案由大分類", "終結情形大分類", top_n=8)

    # 6. Amount box-whisker plot
    amount_box = []
    if "c0_訴訟標的金額" in df.columns and "案由大分類" in df.columns:
        df_valid = df[(df["案由大分類"].str.strip() != "") & (df["c0_訴訟標的金額"].str.strip() != "")]
        if not df_valid.empty:
            top_causes = [c for c, _ in Counter(df_valid["案由大分類"]).most_common(8)]
            for cause in top_causes:
                sub = df_valid[df_valid["案由大分類"] == cause]
                amounts = sub["c0_訴訟標的金額"].apply(parse_int_loose)
                valid_amounts = sorted(amounts.dropna().tolist())
                if valid_amounts:
                    bs = box_stats(valid_amounts)
                    if bs:
                        amount_box.append({
                            "cause": cause, "n": len(valid_amounts),
                            **{k: round(v, 0) if isinstance(v, float) else v for k, v in bs.items()},
                        })

    # 7. Court × Ending heatmap
    court_heatmap = {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    court_col = "法院別" if "法院別" in df.columns else "c0_法院別"
    if court_col in df.columns and "終結情形大分類" in df.columns:
        court_heatmap = _build_heatmap(df, "終結情形大分類", court_col)

    # 8. Lawyer × Ending stacked bar
    lawyer_ending_stack = {"data": [], "segments": []}
    if "律師代理情形" in df.columns and "終結情形大分類" in df.columns:
        lawyer_ending_stack = _build_stacked_bar(df, "律師代理情形", "終結情形大分類")

    return {
        "endingDist": ending_data,
        "causeDist": cause_data,
        "lawyerHeatmap": lawyer_heatmap,
        "amountHeatmap": amount_heatmap,
        "causeEndingStack": cause_ending_stack,
        "amountBox": amount_box,
        "courtHeatmap": court_heatmap,
        "lawyerEndingStack": lawyer_ending_stack,
    }


# ════════════════════════════════════════════════════════════
#  Civil Non-litigation Specific
# ════════════════════════════════════════════════════════════
def apply_nonlitig_filters(df, params):
    result = apply_common_filters(df, params)
    for col, param_key in [
        ("終結情形大分類", "ending"),
        ("案由大分類", "cause"),
    ]:
        val = params.get(param_key)
        if val and col in result.columns:
            result = result[result[col].isin(val.split(","))]

    debt = params.get("is_debt")
    if debt and "是否消債事件" in result.columns:
        result = result[result["是否消債事件"] == debt]

    # Applicant filter
    applicant = params.get("applicant")
    if applicant and "c0_聲請人別" in result.columns:
        result = result[result["c0_聲請人別"].isin(applicant.split(","))]

    return result


def nonlitig_filter_options(df):
    def count_col(col):
        if col not in df.columns:
            return []
        return [{"val": k, "count": int(v)} for k, v in
                df[col].value_counts().items() if clean(k)]

    ym_set = set()
    y_col = "c0_全案終結日期-年" if "c0_全案終結日期-年" in df.columns else "終結年"
    m_col = "c0_全案終結日期-月" if "c0_全案終結日期-月" in df.columns else "終結月"
    for _, row in df[[y_col, m_col]].drop_duplicates().iterrows():
        y, m = clean(row[y_col]), clean(row[m_col])
        if y and m:
            ym_set.add(f"{y.zfill(3)}/{m.zfill(2)}")

    return {
        "courts": count_col("法院別") or count_col("c0_法院別"),
        "endings": count_col("終結情形大分類"),
        "causes": count_col("案由大分類"),
        "ym": sorted(ym_set),
        "isDebt": count_col("是否消債事件"),
        "applicants": count_col("c0_聲請人別"),
    }


def nonlitig_stats(df):
    jid_col = "裁判書ID" if "裁判書ID" in df.columns else "c0_裁判書ID"

    # Calculate debt rate
    debt_rate = 0
    if "是否消債事件" in df.columns:
        debt_cases = df[df["是否消債事件"] == "是"].shape[0]
        total = len(df)
        if total > 0:
            debt_rate = round((debt_cases / total) * 100, 2)

    return {
        "judgments": int(df[jid_col].nunique()),
        "totalRows": len(df),
        "debtRate": debt_rate,
    }


def nonlitig_charts(df):
    """Build chart data for civil non-litigation."""
    ending_counts = df["終結情形大分類"].value_counts().to_dict()
    ending_data = [{"name": k, "count": int(v)} for k, v in ending_counts.items() if clean(k)]

    cause_counts = df["案由大分類"].value_counts().to_dict()
    cause_data = [{"name": k, "count": int(v)} for k, v in cause_counts.items() if clean(k)]

    # Cause × Ending heatmap
    heatmap = _build_heatmap(df, "終結情形大分類", "案由大分類")

    # Debt vs non-debt comparison
    debt_data = []
    if "是否消債事件" in df.columns:
        for label in ["是", "否"]:
            sub = df[df["是否消債事件"] == label]
            if not sub.empty:
                endings = sub["終結情形大分類"].value_counts().to_dict()
                debt_data.append({"label": label, "count": len(sub), "endings": {k: int(v) for k, v in endings.items()}})

    # Court × Ending heatmap
    court_heatmap = {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    court_col = "法院別" if "法院別" in df.columns else "c0_法院別"
    if court_col in df.columns and "終結情形大分類" in df.columns:
        court_heatmap = _build_heatmap(df, "終結情形大分類", court_col)

    # Cause × Ending stacked bar
    cause_ending_stack = {"data": [], "segments": []}
    if "案由大分類" in df.columns and "終結情形大分類" in df.columns:
        cause_ending_stack = _build_stacked_bar(df, "案由大分類", "終結情形大分類", top_n=8)

    # Debt × Ending heatmap
    debt_ending_heatmap = {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    if "是否消債事件" in df.columns and "終結情形大分類" in df.columns:
        debt_ending_heatmap = _build_heatmap(df, "終結情形大分類", "是否消債事件")

    # Applicant distribution
    applicant_dist = []
    if "c0_聲請人別" in df.columns:
        applicant_counts = df[df["c0_聲請人別"].str.strip() != ""]["c0_聲請人別"].value_counts()
        applicant_dist = [{"name": k, "count": int(v)} for k, v in applicant_counts.items()]

    return {
        "endingDist": ending_data,
        "causeDist": cause_data,
        "causeEndingHeatmap": heatmap,
        "debtComparison": debt_data,
        "courtHeatmap": court_heatmap,
        "causeEndingStack": cause_ending_stack,
        "debtEndingHeatmap": debt_ending_heatmap,
        "applicantDist": applicant_dist,
    }


# ════════════════════════════════════════════════════════════
#  Family Litigation Specific
# ════════════════════════════════════════════════════════════
def apply_family_filters(df, params):
    result = apply_common_filters(df, params)
    for col, param_key in [
        ("終結情形大分類", "ending"),
        ("案由大分類", "cause"),
        ("律師代理情形", "lawyer"),
    ]:
        val = params.get(param_key)
        if val and col in result.columns:
            result = result[result[col].isin(val.split(","))]

    # Divorce initiator
    initiator = params.get("initiator")
    if initiator and "主動離婚者" in result.columns:
        result = result[result["主動離婚者"].isin(initiator.split(","))]

    # Divorce reason
    reason = params.get("divorce_reason")
    if reason and "離婚原因" in result.columns:
        result = result[result["離婚原因"].isin(reason.split(","))]

    return result


def family_filter_options(df):
    def count_col(col):
        if col not in df.columns:
            return []
        return [{"val": k, "count": int(v)} for k, v in
                df[col].value_counts().items() if clean(k)]

    ym_set = set()
    y_col = "c0_全案終結日期-年" if "c0_全案終結日期-年" in df.columns else "終結年"
    m_col = "c0_全案終結日期-月" if "c0_全案終結日期-月" in df.columns else "終結月"
    for _, row in df[[y_col, m_col]].drop_duplicates().iterrows():
        y, m = clean(row[y_col]), clean(row[m_col])
        if y and m:
            ym_set.add(f"{y.zfill(3)}/{m.zfill(2)}")

    return {
        "courts": count_col("法院別") or count_col("c0_法院別"),
        "endings": count_col("終結情形大分類"),
        "causes": count_col("案由大分類"),
        "lawyers": count_col("律師代理情形"),
        "ym": sorted(ym_set),
        "divorceReasons": count_col("離婚原因"),
        "initiators": count_col("主動離婚者"),
    }


def family_stats(df):
    jid_col = "裁判書ID" if "裁判書ID" in df.columns else "c0_裁判書ID"

    # Calculate divorce rate
    divorce_rate = 0
    if "案由大分類" in df.columns:
        divorce_cases = df[df["案由大分類"] == "離婚"].shape[0]
        total = len(df)
        if total > 0:
            divorce_rate = round((divorce_cases / total) * 100, 2)

    # Calculate lawyer rate
    lawyer_rate = 0
    if "律師代理情形" in df.columns:
        with_lawyer = df[df["律師代理情形"].str.strip() != ""].shape[0]
        total = len(df)
        if total > 0:
            lawyer_rate = round((with_lawyer / total) * 100, 2)

    return {
        "judgments": int(df[jid_col].nunique()),
        "totalRows": len(df),
        "divorceRate": divorce_rate,
        "lawyerRate": lawyer_rate,
    }


def family_charts(df):
    """Build chart data for family litigation."""
    ending_data = [{"name": k, "count": int(v)}
                   for k, v in df["終結情形大分類"].value_counts().items() if clean(k)]

    cause_data = [{"name": k, "count": int(v)}
                  for k, v in df["案由大分類"].value_counts().items() if clean(k)]

    # Lawyer × Ending heatmap
    heatmap = {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    if "律師代理情形" in df.columns:
        heatmap = _build_heatmap(df, "終結情形大分類", "律師代理情形")

    # Divorce reason distribution
    divorce_data = []
    if "離婚原因" in df.columns:
        divorce_counts = df[df["離婚原因"].str.strip() != ""]["離婚原因"].value_counts()
        divorce_data = [{"name": k, "count": int(v)} for k, v in divorce_counts.items()]

    # Initiator distribution
    initiator_data = []
    if "主動離婚者" in df.columns:
        init_counts = df[df["主動離婚者"].str.strip() != ""]["主動離婚者"].value_counts()
        initiator_data = [{"name": k, "count": int(v)} for k, v in init_counts.items()]

    # Court × Ending heatmap
    court_heatmap = {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    court_col = "法院別" if "法院別" in df.columns else "c0_法院別"
    if court_col in df.columns and "終結情形大分類" in df.columns:
        court_heatmap = _build_heatmap(df, "終結情形大分類", court_col)

    # Cause × Ending stacked bar
    cause_ending_stack = {"data": [], "segments": []}
    if "案由大分類" in df.columns and "終結情形大分類" in df.columns:
        cause_ending_stack = _build_stacked_bar(df, "案由大分類", "終結情形大分類")

    # Initiator × Divorce Reason heatmap
    initiator_reason_heatmap = {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    if "主動離婚者" in df.columns and "離婚原因" in df.columns:
        initiator_reason_heatmap = _build_heatmap(df, "離婚原因", "主動離婚者")

    # Lawyer × Ending stacked bar
    lawyer_ending_stack = {"data": [], "segments": []}
    if "律師代理情形" in df.columns and "終結情形大分類" in df.columns:
        lawyer_ending_stack = _build_stacked_bar(df, "律師代理情形", "終結情形大分類")

    return {
        "endingDist": ending_data,
        "causeDist": cause_data,
        "lawyerHeatmap": heatmap,
        "divorceReasons": divorce_data,
        "initiators": initiator_data,
        "courtHeatmap": court_heatmap,
        "causeEndingStack": cause_ending_stack,
        "divorceInitiatorHeatmap": initiator_reason_heatmap,
        "lawyerEndingStack": lawyer_ending_stack,
    }


# ════════════════════════════════════════════════════════════
#  Shared Heatmap Builder
# ════════════════════════════════════════════════════════════
def _build_heatmap(df, x_col, y_col, x_limit=8, y_limit=8):
    filtered = df[(df[x_col].str.strip() != "") & (df[y_col].str.strip() != "")]
    if filtered.empty:
        return {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}

    x_top_set = set(k for k, _ in Counter(filtered[x_col]).most_common(x_limit))
    y_top_set = set(k for k, _ in Counter(filtered[y_col]).most_common(y_limit))

    # Map non-top to "其他" — vectorized
    xs = filtered[x_col].where(filtered[x_col].isin(x_top_set), "其他")
    ys = filtered[y_col].where(filtered[y_col].isin(y_top_set), "其他")

    # Cross-tabulate
    ct = pd.crosstab(ys, xs)
    x_totals = ct.sum(axis=0).sort_values(ascending=False)
    y_totals = ct.sum(axis=1).sort_values(ascending=False)

    x_labels = list(x_totals.index)
    y_labels = list(y_totals.index)

    matrix = []
    max_val = 0
    for y in y_labels:
        row = []
        for x in x_labels:
            v = int(ct.at[y, x]) if y in ct.index and x in ct.columns else 0
            if v > max_val:
                max_val = v
            row.append(v)
        matrix.append(row)

    return {"xLabels": x_labels, "yLabels": y_labels, "matrix": matrix, "max": max_val}


# ════════════════════════════════════════════════════════════
#  Judgment List (paginated)
# ════════════════════════════════════════════════════════════
def get_judgment_list(df, case_type, page=0, page_size=10):
    """Get paginated judgment list."""
    jid_col = "裁判書ID" if "裁判書ID" in df.columns else "c0_裁判書ID"
    court_col = "c0_法院別" if "c0_法院別" in df.columns else "法院別"
    cause_col = "c0_案由" if "c0_案由" in df.columns else ""

    grouped = df.groupby(jid_col)
    jids = list(grouped.groups.keys())
    total = len(jids)
    total_pages = math.ceil(total / page_size) if total else 1

    page_jids = jids[page * page_size: (page + 1) * page_size]

    items = []
    for jid in page_jids:
        rows = grouped.get_group(jid)
        first = rows.iloc[0]
        item = {
            "jid": jid,
            "court": clean(first.get(court_col, "")),
            "cause": clean(first.get(cause_col, "")) if cause_col else "",
            "rowCount": len(rows),
        }

        if case_type == "criminal_litigation":
            item["ending"] = clean(first.get("c0_全案終結情形", ""))
            item["cls"] = clean(first.get("案件分類", ""))
            item["defendants"] = parse_int_loose(first.get("判決被告人數", "1")) or 1
            item["crimeCount"] = parse_int_loose(first.get("判決總罪數", "")) or len(rows)
            item["lawCount"] = rows["定罪法條"].replace("", pd.NA).dropna().nunique() if "定罪法條" in rows.columns else 0
        elif case_type in ("civil_litigation", "family_litigation"):
            item["ending"] = clean(first.get("終結情形大分類", ""))
            item["causeCat"] = clean(first.get("案由大分類", ""))
            if "律師代理情形" in first.index:
                item["lawyer"] = clean(first.get("律師代理情形", ""))
        elif case_type == "civil_nonlitig":
            item["ending"] = clean(first.get("終結情形大分類", ""))
            item["causeCat"] = clean(first.get("案由大分類", ""))

        items.append(item)

    return {
        "items": items,
        "page": page,
        "totalPages": total_pages,
        "totalJudgments": total,
    }


# ════════════════════════════════════════════════════════════
#  API Routes
# ════════════════════════════════════════════════════════════

@app.get("/api/types")
async def list_types():
    """List available case types."""
    result = []
    for key, info in CASE_TYPES.items():
        available = key in DATA
        row_count = len(DATA[key]) if available else 0
        result.append({
            "key": key,
            "label": info["label"],
            "category": info["category"],
            "available": available,
            "rowCount": row_count,
        })
    return result


@app.get("/api/{case_type}/options")
async def get_options(case_type: str):
    """Get filter options for a case type."""
    if case_type not in DATA:
        return JSONResponse(status_code=404, content={"error": f"Unknown type: {case_type}"})
    df = DATA[case_type]
    if case_type == "criminal_litigation":
        return criminal_filter_options(df)
    elif case_type == "civil_litigation":
        return civil_filter_options(df)
    elif case_type == "civil_nonlitig":
        return nonlitig_filter_options(df)
    elif case_type == "family_litigation":
        return family_filter_options(df)
    return {}


@app.get("/api/{case_type}/data")
async def get_data(
    case_type: str,
    court: Optional[str] = None,
    ym_min: Optional[str] = None,
    ym_max: Optional[str] = None,
    ending: Optional[str] = None,
    cause: Optional[str] = None,
    lawyer: Optional[str] = None,
    amount: Optional[str] = None,
    cls: Optional[str] = None,
    defense: Optional[str] = None,
    procedure: Optional[str] = None,
    probation: Optional[str] = None,
    recidivist: Optional[str] = None,
    article: Optional[str] = None,
    result: Optional[str] = None,
    national_comp: Optional[str] = None,
    countersuit: Optional[str] = None,
    appeal: Optional[str] = None,
    is_debt: Optional[str] = None,
    applicant: Optional[str] = None,
    initiator: Optional[str] = None,
    divorce_reason: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
):
    """Get filtered data: stats, charts, and paginated judgment list."""
    if case_type not in DATA:
        return JSONResponse(status_code=404, content={"error": f"Unknown type: {case_type}"})

    params = {
        "court": court, "ym_min": ym_min, "ym_max": ym_max,
        "ending": ending, "cause": cause, "lawyer": lawyer, "amount": amount,
        "cls": cls, "defense": defense, "procedure": procedure,
        "probation": probation, "recidivist": recidivist, "article": article,
        "result": result, "national_comp": national_comp, "countersuit": countersuit,
        "appeal": appeal, "is_debt": is_debt, "applicant": applicant,
        "initiator": initiator, "divorce_reason": divorce_reason,
    }
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    df = DATA[case_type]

    # Apply filters
    if case_type == "criminal_litigation":
        filtered = apply_criminal_filters(df, params)
        stats = criminal_stats(filtered)
        charts = criminal_charts(filtered)
    elif case_type == "civil_litigation":
        filtered = apply_civil_filters(df, params)
        stats = civil_stats(filtered)
        charts = civil_charts(filtered)
    elif case_type == "civil_nonlitig":
        filtered = apply_nonlitig_filters(df, params)
        stats = nonlitig_stats(filtered)
        charts = nonlitig_charts(filtered)
    elif case_type == "family_litigation":
        filtered = apply_family_filters(df, params)
        stats = family_stats(filtered)
        charts = family_charts(filtered)
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown type"})

    judgments = get_judgment_list(filtered, case_type, page, page_size)

    return {
        "stats": stats,
        "charts": charts,
        "judgments": judgments,
        "filteredRows": len(filtered),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "types_loaded": list(DATA.keys())}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
