"""
裁判書量化實證研究 — FastAPI 後端 V3
完整篩選（含交集/聯集）、統計、圖表、地圖數據、判決詳情
"""
import os, math, re, json
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI(title="裁判書量化實證研究 API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Middleware to fix double-slash paths (e.g. //api/types -> /api/types)
class NormalizePathMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if "//" in request.scope["path"]:
            request.scope["path"] = re.sub(r"/+", "/", request.scope["path"])
        return await call_next(request)

app.add_middleware(NormalizePathMiddleware)

# ════════════════════════════════════════════════════════════
#  Flag / Keyword Definitions
# ════════════════════════════════════════════════════════════
SECURITY_FLAGS = [
    ("c1_保安處分-感化教育", "感化教育"), ("c1_保安處分-監護", "監護"),
    ("c1_保安處分-禁戒", "禁戒"), ("c1_保安處分-強制工作", "強制工作"),
    ("c1_保安處分-保護管束", "保護管束"), ("c1_保安處分-強制治療", "強制治療"),
    ("c1_保安處分-驅逐出境", "驅逐出境"), ("c1_保安處分-安置輔導", "安置輔導"),
    ("c1_保安處分-其他", "其他"),
]
COMPENSATION_FLAGS = [
    ("c1_協商賠償支付對象-被害人", "被害人"), ("c1_協商賠償支付對象-公庫", "公庫"),
    ("c1_協商賠償支付對象-公益團體", "公益團體"), ("c1_協商賠償支付對象-地方自治團體", "地方自治團體"),
]
CONFISCATION_FLAGS = [
    ("c1_宣告沒收適用法律-刑法第38條第1項(違禁物)", "§38-1違禁物"),
    ("c1_宣告沒收適用法律-刑法第38條第2項(犯罪工具產物)", "§38-2犯罪工具"),
    ("c1_宣告沒收適用法律-刑法第38條第3項(犯罪工具產物)", "§38-3犯罪工具"),
    ("c1_宣告沒收適用法律-刑法第38-1條第1項(犯罪所得)", "§38-1-1犯罪所得"),
    ("c1_宣告沒收適用法律-刑法第38-1條第2項(犯罪所得)", "§38-1-2犯罪所得"),
    ("c1_宣告沒收適用法律-刑法第38-1條第2項第1款(犯罪所得)", "§38-1-2-1"),
    ("c1_宣告沒收適用法律-刑法第38-1條第2項第2款(犯罪所得)", "§38-1-2-2"),
    ("c1_宣告沒收適用法律-刑法第38-1條第2項第3款(犯罪所得)", "§38-1-2-3"),
]
PROBCOND_FLAGS = [
    ("c1_宣告緩刑附記應遵收事項-向被害人道歉", "向被害人道歉"),
    ("c1_宣告緩刑附記應遵收事項-立悔過書", "立悔過書"),
    ("c1_宣告緩刑附記應遵收事項-向被害人支付相當數額之財產或非財產上之損害賠償", "損害賠償"),
    ("c1_宣告緩刑附記應遵收事項-向公庫支付一定之金額", "向公庫支付"),
    ("c1_宣告緩刑附記應遵收事項-向指定之政府機關（構）、行政法人、社區、公益機構或團體，提供40小時以上240小時以下之義務勞務", "義務勞務"),
    ("c1_宣告緩刑附記應遵收事項-完成戒癮治療、精神治療、心理輔導或其他適當之處遇措施", "治療輔導"),
    ("c1_宣告緩刑附記應遵收事項-保護被害人安全之必要命令", "保護命令"),
    ("c1_宣告緩刑附記應遵收事項-預防再犯所為之必要命令", "預防再犯"),
]
DV_FLAGS = [
    ("c1_違反家暴法緩刑期間應遵守事項-第38條第2項第1款禁止實施家庭暴力", "禁止暴力"),
    ("c1_違反家暴法緩刑期間應遵守事項-第38條第2項第3款強制遷出", "強制遷出"),
    ("c1_違反家暴法緩刑期間應遵守事項第38條第2項第2款禁止騷擾等行為", "禁止騷擾"),
    ("c1_違反家暴法緩刑期間應遵守事項第38條第2項第5款強制戒癮治療", "戒癮"),
    ("c1_違反家暴法緩刑期間應遵守事項第38條第2項第5款強制精神治療", "精神治療"),
    ("c1_違反家暴法緩刑期間應遵守事項第38條第2項第5款強制心理輔導", "心理輔導"),
    ("c1_違反家暴法緩刑期間應遵守事項第38條第2項第5款強制其他治療、輔導", "其他治療輔導"),
    ("c1_違反家暴法緩刑期間應遵守事項第38條第2項第6款其他必要保護被害人等措施", "其他保護措施"),
    ("c1_違反家暴法緩刑期間應遵守事項第38條第2項第4款強制遠離", "強制遠離"),
    ("c1_違反家暴法緩刑期間應遵守事項第38條第2項第5款強制認知教育輔導", "認知教育"),
]
CRIME_FLAGS = [
    ("c11_是否褫奪公權", "褫奪公權"), ("c11_罪犯類型-少年犯", "少年犯"),
    ("c11_罪犯類型-幫助犯", "幫助犯"), ("c11_罪犯類型-未遂犯", "未遂犯"),
    ("c11_罪犯類型-家庭暴力", "家庭暴力"),
]
AG_MIT_CATS = ["無加重無減輕", "僅有加重法條", "僅有減輕法條", "有加重有減輕"]

# Expanded civil keyword lists
CIVIL_ACTIONS = ["侵權行為", "損害賠償", "國家賠償", "代位請求", "債務人異議", "分配表異議",
    "拆屋還地", "塗銷", "清償", "給付", "返還", "確認", "分割", "撤銷", "除權",
    "遷讓", "履行", "拍賣", "再審"]
CIVIL_SUBJECTS = ["簽帳卡", "信用卡", "現金卡", "借款", "消費借貸", "債務", "消費款",
    "票款", "電信費", "不當得利", "房屋", "管理費", "貨款", "遺產", "工程款",
    "土地", "租金", "保險", "本票", "股票", "買賣價金", "薪資", "資遣費",
    "抵押", "共有物", "契約"]

# ════════════════════════════════════════════════════════════
#  Data Loading
# ════════════════════════════════════════════════════════════
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "Data")))
CASE_TYPES = {
    "criminal_litigation": {"label": "刑事訴訟", "category": "criminal", "file_pattern": "刑事訴訟"},
    "civil_litigation": {"label": "民事訴訟", "category": "civil", "file_pattern": "民事訴訟"},
    "family_litigation": {"label": "家事訴訟", "category": "family", "file_pattern": "家事訴訟"},
}
DATA: dict[str, pd.DataFrame] = {}
CACHE: dict[str, dict] = {}  # precomputed default results

def _find_csv(pattern):
    for f in sorted(DATA_DIR.glob("*.csv")):
        if pattern in f.stem:
            return f
    return None

def _extract_keyword(text, keywords):
    t = str(text or "")
    for kw in keywords:
        if kw in t:
            return kw
    return "其他"

def _preprocess(key, df):
    if key == "criminal_litigation":
        if "案件分類" in df.columns:
            df.loc[df["案件分類"] == "多人一罪", "案件分類"] = "多人多罪"
        agg_col = "c111_量刑加重" if "c111_量刑加重" in df.columns else None
        mit_col = "c112_量刑減輕" if "c112_量刑減輕" in df.columns else None
        if agg_col and mit_col:
            ha = df[agg_col].str.strip().ne("")
            hm = df[mit_col].str.strip().ne("")
            cat = pd.Series("無加重無減輕", index=df.index)
            cat[ha & ~hm] = "僅有加重法條"
            cat[~ha & hm] = "僅有減輕法條"
            cat[ha & hm] = "有加重有減輕"
            df["_ag_mit"] = cat
        prb_col = "c1_是否宣告緩刑" if "c1_是否宣告緩刑" in df.columns else None
        if prb_col:
            df["_probation"] = df[prb_col].map({"1": "有緩刑", "0": "無緩刑"}).fillna("")
    elif key == "civil_litigation":
        if "c0_案由" in df.columns:
            df["_action"] = df["c0_案由"].apply(lambda x: _extract_keyword(x, CIVIL_ACTIONS))
            df["_subject"] = df["c0_案由"].apply(lambda x: _extract_keyword(x, CIVIL_SUBJECTS))
    return df

def _load_all():
    for key, info in CASE_TYPES.items():
        csv_path = _find_csv(info["file_pattern"])
        if csv_path and csv_path.exists():
            print(f"Loading {key}: {csv_path.name} ...")
            df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, low_memory=False)
            df = df.fillna("")
            df = _preprocess(key, df)
            DATA[key] = df
            print(f"  → {len(df)} rows, {len(df.columns)} cols")
        else:
            print(f"Warning: no CSV found for {key}")
    # Precompute defaults
    for key in DATA:
        print(f"Precomputing defaults for {key}...")
        CACHE[key] = _compute_response(key, DATA[key])
    print("All precomputation done.")

@app.on_event("startup")
async def startup():
    _load_all()

# ════════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════════
def clean(v):
    return str(v).strip() if pd.notna(v) else ""

def split_pipe(v):
    return [x.strip() for x in str(v).split("|") if x.strip()]

def parse_months(s):
    v = re.sub(r"\D", "", str(s))
    if not v: return None
    if len(v) >= 4: y, m = int(v[:-2]) or 0, int(v[-2:]) or 0
    elif len(v) == 3: y, m = int(v[0]) or 0, int(v[1:]) or 0
    else: m = int(v) or 0; y = 0
    return y * 12 + m

def count_col(df, col):
    if col not in df.columns: return []
    return [{"val": k, "count": int(v)} for k, v in df[col].value_counts().items() if clean(k)]

def get_ym(df):
    ym_set = set()
    y_col = "終結年" if "終結年" in df.columns else "c0_全案終結日期-年"
    m_col = "終結月" if "終結月" in df.columns else "c0_全案終結日期-月"
    if y_col in df.columns and m_col in df.columns:
        for _, row in df[[y_col, m_col]].drop_duplicates().iterrows():
            y, m = clean(row[y_col]), clean(row[m_col])
            if y and m:
                ym_set.add(f"{y.zfill(3)}/{m.zfill(2)}")
    return sorted(ym_set)

def flag_opts(df, flags):
    result = []
    for col, label in flags:
        if col in df.columns:
            cnt = (df[col] == "1").sum()
            if cnt > 0:
                result.append({"val": label, "count": int(cnt)})
    return result

def quantile(sorted_arr, q):
    if not sorted_arr: return 0
    p = (len(sorted_arr) - 1) * q
    b = int(p)
    if b + 1 < len(sorted_arr):
        return sorted_arr[b] + (p - b) * (sorted_arr[b + 1] - sorted_arr[b])
    return sorted_arr[b]

def box_stats(values):
    s = sorted(values)
    if not s: return None
    q1, median, q3 = quantile(s, 0.25), quantile(s, 0.5), quantile(s, 0.75)
    iqr = q3 - q1
    lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = [v for v in s if lf <= v <= uf]
    return {"q1": q1, "median": median, "q3": q3, "iqr": iqr,
            "whiskerLow": inliers[0] if inliers else s[0],
            "whiskerHigh": inliers[-1] if inliers else s[-1],
            "outliers": [v for v in s if v < lf or v > uf], "min": s[0], "max": s[-1]}

# ════════════════════════════════════════════════════════════
#  Shared Chart Builders
# ════════════════════════════════════════════════════════════
def _build_heatmap(df, x_col, y_col, x_limit=8, y_limit=8):
    filtered = df[(df[x_col].str.strip() != "") & (df[y_col].str.strip() != "")]
    if filtered.empty:
        return {"xLabels": [], "yLabels": [], "matrix": [], "max": 0}
    x_top_set = set(k for k, _ in Counter(filtered[x_col]).most_common(x_limit))
    y_top_set = set(k for k, _ in Counter(filtered[y_col]).most_common(y_limit))
    xs = filtered[x_col].where(filtered[x_col].isin(x_top_set), "其他")
    ys = filtered[y_col].where(filtered[y_col].isin(y_top_set), "其他")
    ct = pd.crosstab(ys, xs)
    x_totals = ct.sum(axis=0).sort_values(ascending=False)
    y_totals = ct.sum(axis=1).sort_values(ascending=False)
    x_labels = [x for x in x_totals.index if x != "其他"]
    if "其他" in x_totals.index: x_labels.append("其他")
    y_labels = [y for y in y_totals.index if y != "其他"]
    if "其他" in y_totals.index: y_labels.append("其他")
    matrix, max_val = [], 0
    for y in y_labels:
        row = []
        for x in x_labels:
            v = int(ct.at[y, x]) if y in ct.index and x in ct.columns else 0
            if v > max_val: max_val = v
            row.append(v)
        matrix.append(row)
    return {"xLabels": x_labels, "yLabels": y_labels, "matrix": matrix, "max": max_val}

def _build_court_class_bar(df, court_col="c0_法院別", class_col="案件分類"):
    """Build stacked vertical bar data: X=courts (all), segments=case classes."""
    filtered = df[(df[court_col].str.strip() != "") & (df[class_col].str.strip() != "")]
    if filtered.empty:
        return {"courts": [], "segments": [], "data": []}
    segments = [s for s, _ in Counter(filtered[class_col]).most_common()]
    court_order = [c for c, _ in Counter(filtered[court_col]).most_common()]
    ABBR = {
        "臺北地方法院": "臺北", "新北地方法院": "新北", "士林地方法院": "士林",
        "桃園地方法院": "桃園", "新竹地方法院": "新竹", "苗栗地方法院": "苗栗",
        "臺中地方法院": "臺中", "南投地方法院": "南投", "彰化地方法院": "彰化",
        "雲林地方法院": "雲林", "嘉義地方法院": "嘉義", "臺南地方法院": "臺南",
        "高雄地方法院": "高雄", "橋頭地方法院": "橋頭", "屏東地方法院": "屏東",
        "宜蘭地方法院": "宜蘭", "花蓮地方法院": "花蓮", "臺東地方法院": "臺東",
        "基隆地方法院": "基隆", "澎湖地方法院": "澎湖", "金門地方法院": "金門",
        "連江地方法院": "連江", "福建金門地方法院": "金門", "福建連江地方法院": "連江",
    }
    data = []
    for court in court_order:
        sub = filtered[filtered[court_col] == court]
        row = {"court": court, "abbr": ABBR.get(court, court.replace("地方法院", "")), "total": int(len(sub))}
        counts = {}
        for seg in segments:
            c = int(len(sub[sub[class_col] == seg]))
            row[seg] = c
            counts[seg] = c
        row["__counts"] = counts
        data.append(row)
    return {"courts": [d["abbr"] for d in data], "segments": segments, "data": data}

def _build_stacked_bar(df, group_col, segment_col, top_n=8):
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
                row[s] = t; acc += t
            else:
                row[s] = 1000 - acc
        data.append(row)
    return {"data": data, "segments": segments}

def _build_dual_axis_bar(df, group_col, segment_col, top_n=8):
    """Bar chart with actual counts + proportion line on secondary axis."""
    filtered = df[(df[group_col].str.strip() != "") & (df[segment_col].str.strip() != "")]
    if filtered.empty:
        return {"groups": [], "segments": [], "data": []}
    segments = [s for s, _ in Counter(filtered[segment_col]).most_common()]
    top_groups = [g for g, _ in Counter(filtered[group_col]).most_common(top_n)]
    grand_total = len(filtered)
    data = []
    for group in top_groups:
        sub = filtered[filtered[group_col] == group]
        total = len(sub)
        cats = sub[segment_col].value_counts().to_dict()
        row = {"name": group, "total": total, "pct": round(float(total / grand_total) * 100, 1) if grand_total else 0}
        for s in segments:
            row[s] = cats.get(s, 0)
        data.append(row)
    return {"groups": top_groups, "segments": segments, "data": data, "grandTotal": grand_total}

# ════════════════════════════════════════════════════════════
#  Court Map Data Builder
# ════════════════════════════════════════════════════════════
def _build_court_map(df, court_col, category_col=None, lawyer_col=None, ending_col=None, top_cats=5, extra_fn=None):
    """Build per-court statistics for map visualization.
    extra_fn(sub, count) -> dict of extra fields to merge into info.
    """
    if court_col not in df.columns:
        return []
    courts = df[court_col].value_counts()
    total_all = len(df)
    result = []
    for court, count in courts.items():
        if not clean(court):
            continue
        info = {"court": court, "count": int(count), "pct": round(float(count / total_all) * 100, 1) if total_all else 0}
        sub = df[df[court_col] == court]
        # Top categories
        if category_col and category_col in sub.columns:
            cat_counts = sub[category_col].value_counts()
            # Take top N non-其他, then append 其他 at end
            top_items = []
            other_count = 0
            for k, v in cat_counts.items():
                if not clean(k):
                    continue
                if k == "其他":
                    other_count = int(v)
                elif len(top_items) < top_cats:
                    top_items.append({"name": k, "count": int(v), "pct": round(float(v / count) * 100, 1)})
            if other_count > 0:
                top_items.append({"name": "其他", "count": other_count, "pct": round(float(other_count / count) * 100, 1)})
            info["topCats"] = top_items
        # Lawyer rate
        if lawyer_col and lawyer_col in sub.columns:
            with_lawyer = int((sub[lawyer_col] != "雙方無律師").sum())
            info["lawyerRate"] = round(float(with_lawyer / count) * 100, 1) if count else 0
        # Extra function for custom per-court stats
        if extra_fn:
            info.update(extra_fn(sub, count))
        result.append(info)
    return result

# ════════════════════════════════════════════════════════════
#  Common Filter Application
# ════════════════════════════════════════════════════════════
def _apply_ym_filter(df, ym_min, ym_max):
    if not ym_min and not ym_max: return df
    y_col = "終結年" if "終結年" in df.columns else "c0_全案終結日期-年"
    m_col = "終結月" if "終結月" in df.columns else "c0_全案終結日期-月"
    if y_col not in df.columns or m_col not in df.columns: return df
    ym = df[y_col].str.zfill(3) + "/" + df[m_col].str.zfill(2)
    mask = pd.Series(True, index=df.index)
    if ym_min: mask &= ym >= ym_min
    if ym_max: mask &= ym <= ym_max
    return df[mask]

def _apply_csv_filter(df, col, values, mode="or"):
    if not values or col not in df.columns: return df
    vals = values.split(",") if isinstance(values, str) else values
    return df[df[col].isin(vals)]

def _apply_flag_filter(df, flags, selected_labels, mode="or"):
    if not selected_labels: return df
    labels = selected_labels.split(",") if isinstance(selected_labels, str) else selected_labels
    label_to_col = {label: col for col, label in flags}
    cols = [(label_to_col[l], l) for l in labels if l in label_to_col]
    if not cols: return df
    if mode == "and":
        mask = pd.Series(True, index=df.index)
        for col, _ in cols:
            if col in df.columns: mask &= df[col] == "1"
    else:
        mask = pd.Series(False, index=df.index)
        for col, _ in cols:
            if col in df.columns: mask |= df[col] == "1"
    return df[mask]

def _apply_pipe_filter(df, col, selected, mode="or"):
    if not selected or col not in df.columns: return df
    vals = selected.split(",") if isinstance(selected, str) else selected
    if mode == "and":
        mask = pd.Series(True, index=df.index)
        for v in vals: mask &= df[col].str.contains(re.escape(v.strip()), na=False)
    else:
        pattern = "|".join(re.escape(v.strip()) for v in vals)
        mask = df[col].str.contains(pattern, na=False)
    return df[mask]

def _get_logic(logic_dict, key):
    return logic_dict.get(key, "or")

# ════════════════════════════════════════════════════════════
#  Criminal Litigation
# ════════════════════════════════════════════════════════════
def criminal_filter_options(df):
    agg_col = "c111_量刑加重" if "c111_量刑加重" in df.columns else None
    mit_col = "c112_量刑減輕" if "c112_量刑減輕" in df.columns else None
    agg_vals, mit_vals = {}, {}
    if agg_col:
        for v in df[agg_col]:
            for t in split_pipe(v): agg_vals[t] = agg_vals.get(t, 0) + 1
    if mit_col:
        for v in df[mit_col]:
            for t in split_pipe(v): mit_vals[t] = mit_vals.get(t, 0) + 1
    result_col = "c11_被告罪名裁判結果" if "c11_被告罪名裁判結果" in df.columns else "罪名裁判結果"
    return {
        "classes": count_col(df, "案件分類"),
        "courts": count_col(df, "c0_法院別"),
        "endings": count_col(df, "c0_全案終結情形"),
        "defs": count_col(df, "c1_辯護及代理"),
        "procs": count_col(df, "c1_裁判程序"),
        "probs": count_col(df, "_probation") if "_probation" in df.columns else [],
        "articles": count_col(df, "定罪法條"),
        "results": count_col(df, result_col),
        "aggr": [{"val": k, "count": v} for k, v in sorted(agg_vals.items(), key=lambda x: -x[1])],
        "miti": [{"val": k, "count": v} for k, v in sorted(mit_vals.items(), key=lambda x: -x[1])],
        "ym": get_ym(df),
        "security": flag_opts(df, SECURITY_FLAGS),
        "compensation": flag_opts(df, COMPENSATION_FLAGS),
        "confiscation": flag_opts(df, CONFISCATION_FLAGS),
        "probcond": flag_opts(df, PROBCOND_FLAGS),
        "dv": flag_opts(df, DV_FLAGS),
        "crimeFlags": flag_opts(df, CRIME_FLAGS),
    }

def apply_criminal_filters(df, params, logic):
    result = df
    result = _apply_csv_filter(result, "案件分類", params.get("cls"))
    result = _apply_csv_filter(result, "c0_法院別", params.get("court"))
    result = _apply_csv_filter(result, "c0_全案終結情形", params.get("ending"))
    result = _apply_ym_filter(result, params.get("ym_min"), params.get("ym_max"))
    result = _apply_csv_filter(result, "c1_辯護及代理", params.get("defense"))
    result = _apply_csv_filter(result, "c1_裁判程序", params.get("procedure"))
    result = _apply_csv_filter(result, "_probation", params.get("probation"))
    result = _apply_csv_filter(result, "定罪法條", params.get("article"))
    result_col = "c11_被告罪名裁判結果" if "c11_被告罪名裁判結果" in result.columns else "罪名裁判結果"
    result = _apply_csv_filter(result, result_col, params.get("result"))
    result = _apply_flag_filter(result, SECURITY_FLAGS, params.get("security"), _get_logic(logic, "security"))
    result = _apply_flag_filter(result, COMPENSATION_FLAGS, params.get("compensation"), _get_logic(logic, "compensation"))
    result = _apply_flag_filter(result, CONFISCATION_FLAGS, params.get("confiscation"), _get_logic(logic, "confiscation"))
    result = _apply_flag_filter(result, PROBCOND_FLAGS, params.get("probcond"), _get_logic(logic, "probcond"))
    result = _apply_flag_filter(result, DV_FLAGS, params.get("dv"), _get_logic(logic, "dv"))
    result = _apply_flag_filter(result, CRIME_FLAGS, params.get("crime_flags"), _get_logic(logic, "crime_flags"))
    agg_col = "c111_量刑加重" if "c111_量刑加重" in result.columns else None
    mit_col = "c112_量刑減輕" if "c112_量刑減輕" in result.columns else None
    if agg_col: result = _apply_pipe_filter(result, agg_col, params.get("aggravation"), _get_logic(logic, "aggravation"))
    if mit_col: result = _apply_pipe_filter(result, mit_col, params.get("mitigation"), _get_logic(logic, "mitigation"))
    return result

def criminal_stats(df):
    jid_col = "裁判書ID"
    return {
        "judgmentCount": int(df[jid_col].nunique()) if jid_col in df.columns else 0,
        "defendantCount": int(df["判決被告人數"].apply(lambda x: int(x) if str(x).isdigit() else 1).sum()) if "判決被告人數" in df.columns else len(df),
        "crimeCount": len(df),
        "uniqueLawCount": int(df["定罪法條"].nunique()) if "定罪法條" in df.columns else 0,
    }

def criminal_charts(df, params=None):
    charts = {}
    # Court × Class stacked bar (all courts)
    if "c0_法院別" in df.columns and "案件分類" in df.columns:
        charts["courtClassBar"] = _build_court_class_bar(df, "c0_法院別", "案件分類")

    # Law stacked bar
    if "定罪法條" in df.columns and "_ag_mit" in df.columns:
        charts["lawStack"] = _build_stacked_bar(df, "定罪法條", "_ag_mit", top_n=8)
    # Violin
    violin = []
    if "定罪法條" in df.columns and "c11_宣告有期徒刑" in df.columns:
        top_laws = [l for l, _ in Counter(df[df["定罪法條"].str.strip() != ""]["定罪法條"]).most_common(8)]
        for law in top_laws:
            sub = df[df["定罪法條"] == law]
            vals = sub["c11_宣告有期徒刑"].apply(parse_months).dropna().tolist()
            if vals:
                s = sorted(vals)
                violin.append({"name": law, "values": s, "mean": sum(s)/len(s), "median": quantile(s, 0.5), "n": len(s)})
    charts["violin"] = violin
    # Box-whisker
    box_data = []
    if "定罪法條" in df.columns and "c11_宣告有期徒刑" in df.columns:
        top_laws = [l for l, _ in Counter(df[df["定罪法條"].str.strip() != ""]["定罪法條"]).most_common(8)]
        for law in top_laws:
            sub = df[df["定罪法條"] == law]
            vals = sub["c11_宣告有期徒刑"].apply(parse_months).dropna().tolist()
            if vals:
                bs = box_stats(vals)
                if bs:
                    dominant = "無加重無減輕"
                    if "_ag_mit" in sub.columns:
                        am_counts = sub["_ag_mit"].value_counts()
                        if len(am_counts) > 0: dominant = am_counts.index[0]
                    box_data.append({"law": law, "n": len(vals), "dominant": dominant, **bs})
    charts["boxWhisker"] = box_data
    return charts

# ════════════════════════════════════════════════════════════
#  Civil Litigation
# ════════════════════════════════════════════════════════════
def civil_filter_options(df):
    return {
        "courts": count_col(df, "法院別") or count_col(df, "c0_法院別"),
        "endings": count_col(df, "終結情形大分類"),
        "actions": count_col(df, "_action") if "_action" in df.columns else [],
        "subjects": count_col(df, "_subject") if "_subject" in df.columns else [],
        "lawsuitTypes": count_col(df, "c0_訴訟標的類別"),
        "amountTiers": count_col(df, "訴訟標的金額級距"),
        "lawyers": count_col(df, "律師代理情形"),
        "nationalComp": count_col(df, "是否國賠事件"),
        "agencyTypes": count_col(df, "c0_被請求機關之機關類別"),
        "compTypes": count_col(df, "c0_賠償類別"),
        "publicTypes": count_col(df, "c0_公職類別"),
        "electionTypes": count_col(df, "c0_選舉類別"),
        "ym": get_ym(df),
    }

def apply_civil_filters(df, params, logic):
    result = df
    court_col = "法院別" if "法院別" in df.columns else "c0_法院別"
    result = _apply_csv_filter(result, court_col, params.get("court"))
    result = _apply_csv_filter(result, "終結情形大分類", params.get("ending"))
    result = _apply_csv_filter(result, "_action", params.get("action"))
    result = _apply_csv_filter(result, "_subject", params.get("subject"))
    result = _apply_csv_filter(result, "c0_訴訟標的類別", params.get("lawsuit_type"))
    result = _apply_csv_filter(result, "訴訟標的金額級距", params.get("amount_tier"))
    result = _apply_csv_filter(result, "律師代理情形", params.get("lawyer"))
    result = _apply_csv_filter(result, "是否國賠事件", params.get("national_comp"))
    result = _apply_csv_filter(result, "c0_被請求機關之機關類別", params.get("agency_type"))
    result = _apply_csv_filter(result, "c0_賠償類別", params.get("comp_type"))
    result = _apply_csv_filter(result, "c0_公職類別", params.get("public_type"))
    result = _apply_csv_filter(result, "c0_選舉類別", params.get("election_type"))
    result = _apply_ym_filter(result, params.get("ym_min"), params.get("ym_max"))
    return result

def civil_stats(df):
    jid_col = "裁判書ID" if "裁判書ID" in df.columns else "c0_裁判書ID"
    total = len(df)
    lawyer_rate = 0
    if "律師代理情形" in df.columns and total > 0:
        with_lawyer = (df["律師代理情形"] != "雙方無律師").sum()
        lawyer_rate = round(float(with_lawyer / total) * 100, 1)
    return {"judgmentCount": int(df[jid_col].nunique()) if jid_col in df.columns else 0, "lawyerRate": lawyer_rate}

def _civil_top_causes(df, cause_col="c0_案由", top_n=5):
    """Extract top N most common cause keywords from c0_案由 (full text, not action/subject split)."""
    if cause_col not in df.columns:
        return df.assign(_cause_top=lambda x: "其他") if len(df) else df
    # Use the original c0_案由 to find the most common full-text patterns
    all_keywords = CIVIL_ACTIONS + CIVIL_SUBJECTS
    cause_counts = Counter()
    for text in df[cause_col]:
        t = str(text or "")
        matched = False
        for kw in all_keywords:
            if kw in t:
                cause_counts[kw] += 1
                matched = True
                break
        if not matched:
            cause_counts["其他"] += 1
    # Top N keywords (excluding 其他)
    top_kws = [k for k, _ in cause_counts.most_common() if k != "其他"][:top_n]
    def classify(text):
        t = str(text or "")
        for kw in top_kws:
            if kw in t:
                return kw
        return "其他"
    return df.assign(_cause_top=df[cause_col].apply(classify))

def civil_charts(df):
    charts = {}
    court_col = "法院別" if "法院別" in df.columns else "c0_法院別"
    # Lawyer × Ending dual-axis bar (Chart 1)
    if "律師代理情形" in df.columns and "終結情形大分類" in df.columns:
        charts["lawyerEndingBar"] = _build_dual_axis_bar(df, "律師代理情形", "終結情形大分類")
    # Court map data (Chart 2) — top 5 causes (not split by action/subject), no win rates
    cause_col = "c0_案由" if "c0_案由" in df.columns else ""
    if cause_col:
        df_with_cause = _civil_top_causes(df, cause_col, top_n=5)
        charts["courtMap"] = _build_court_map(df_with_cause, court_col, "_cause_top", "律師代理情形")
    else:
        charts["courtMap"] = _build_court_map(df, court_col, None, "律師代理情形")
    # Amount × Lawyer heatmap (Chart 3, sorted by amount)
    if "訴訟標的金額級距" in df.columns and "律師代理情形" in df.columns:
        amt_order = ["<10萬", "10-50萬", "50-100萬", "100-500萬", "500-1000萬", ">1000萬"]
        hm = _build_heatmap(df, "律師代理情形", "訴訟標的金額級距", x_limit=6, y_limit=8)
        if hm["yLabels"]:
            ordered_y = [a for a in amt_order if a in hm["yLabels"]]
            remaining = [y for y in hm["yLabels"] if y not in ordered_y and y != "其他"]
            if "其他" in hm["yLabels"]: remaining.append("其他")
            new_y = ordered_y + remaining
            old_y_idx = {y: i for i, y in enumerate(hm["yLabels"])}
            new_matrix = [hm["matrix"][old_y_idx[y]] for y in new_y if y in old_y_idx]
            hm["yLabels"] = [y for y in new_y if y in old_y_idx]
            hm["matrix"] = new_matrix
        charts["amountLawyerHeatmap"] = hm
    # Action heatmap and Subject heatmap (Chart 4) — 其他 always last (already handled by _build_heatmap)
    if "_action" in df.columns and "_subject" in df.columns:
        charts["actionSubjectHeatmap"] = _build_heatmap(df, "_subject", "_action", x_limit=10, y_limit=10)
    return charts

# ════════════════════════════════════════════════════════════
#  Family Litigation
# ════════════════════════════════════════════════════════════
def family_filter_options(df):
    return {
        "courts": count_col(df, "法院別") or count_col(df, "c0_法院別"),
        "endings": count_col(df, "終結情形大分類"),
        "causes": count_col(df, "案由大分類"),
        "lawyers": count_col(df, "律師代理情形"),
        "initiators": count_col(df, "主動離婚者") or count_col(df, "c0_主動離婚者"),
        "divorceReasons": count_col(df, "離婚原因") or count_col(df, "c0_離婚原因"),
        "ym": get_ym(df),
    }

def apply_family_filters(df, params, logic):
    result = df
    court_col = "法院別" if "法院別" in df.columns else "c0_法院別"
    result = _apply_csv_filter(result, court_col, params.get("court"))
    result = _apply_csv_filter(result, "終結情形大分類", params.get("ending"))
    result = _apply_csv_filter(result, "案由大分類", params.get("cause"))
    result = _apply_csv_filter(result, "律師代理情形", params.get("lawyer"))
    init_col = "主動離婚者" if "主動離婚者" in result.columns else "c0_主動離婚者"
    result = _apply_csv_filter(result, init_col, params.get("initiator"))
    reason_col = "離婚原因" if "離婚原因" in result.columns else "c0_離婚原因"
    result = _apply_csv_filter(result, reason_col, params.get("divorce_reason"))
    result = _apply_ym_filter(result, params.get("ym_min"), params.get("ym_max"))
    return result

def family_stats(df):
    jid_col = "裁判書ID" if "裁判書ID" in df.columns else "c0_裁判書ID"
    total = len(df)
    lawyer_rate = 0
    if "律師代理情形" in df.columns and total > 0:
        with_lawyer = (df["律師代理情形"] != "雙方無律師").sum()
        lawyer_rate = round(float(with_lawyer / total) * 100, 1)
    divorce_rate = 0
    if "案由大分類" in df.columns and total > 0:
        divorce_cases = (df["案由大分類"] == "離婚").sum()
        divorce_rate = round(float(divorce_cases / total) * 100, 1)
    return {"judgmentCount": int(df[jid_col].nunique()) if jid_col in df.columns else 0, "lawyerRate": lawyer_rate, "divorceRate": divorce_rate}

def family_charts(df):
    charts = {}
    court_col = "法院別" if "法院別" in df.columns else "c0_法院別"
    reason_col = "離婚原因" if "離婚原因" in df.columns else "c0_離婚原因"
    init_col = "主動離婚者" if "主動離婚者" in df.columns else "c0_主動離婚者"
    divorce_sub = df[df["案由大分類"] == "離婚"] if "案由大分類" in df.columns else pd.DataFrame()
    inherit_sub = df[df["案由大分類"] == "繼承"] if "案由大分類" in df.columns else pd.DataFrame()

    # Chart 1: Lawyer × Cause dual-axis bar
    if "律師代理情形" in df.columns and "案由大分類" in df.columns:
        charts["lawyerCauseBar"] = _build_dual_axis_bar(df, "律師代理情形", "案由大分類")

    # Chart 2: Cause distribution (案由 with 其他 split out)
    if "案由大分類" in df.columns:
        cause_counts = df[df["案由大分類"].str.strip() != ""]["案由大分類"].value_counts()
        charts["causeDist"] = [{"name": k, "count": int(v)} for k, v in cause_counts.items() if clean(k)]

    # Chart 3: Court map — divorce/inherit toggle, with initiator lawyer rates
    # Divorce map: per-court stats + per-initiator lawyer rates
    def _divorce_extra(sub, count):
        extra = {}
        if init_col in sub.columns and "律師代理情形" in sub.columns:
            rates = {}
            for initiator in ["男方", "女方", "雙方"]:
                init_sub = sub[sub[init_col] == initiator]
                if len(init_sub) > 0:
                    with_l = (init_sub["律師代理情形"] != "雙方無律師").sum()
                    rates[initiator] = round(float(with_l / len(init_sub)) * 100, 1)
            extra["initiatorLawyerRates"] = rates
        return extra

    def _inherit_extra(sub, count):
        extra = {}
        if "律師代理情形" in sub.columns:
            with_l = (sub["律師代理情形"] != "雙方無律師").sum()
            extra["lawyerRate"] = round(float(with_l / count) * 100, 1) if count else 0
        return extra

    charts["divorceCourtMap"] = _build_court_map(
        divorce_sub, court_col, None, "律師代理情形", None, extra_fn=_divorce_extra
    ) if not divorce_sub.empty else []
    charts["inheritCourtMap"] = _build_court_map(
        inherit_sub, court_col, None, "律師代理情形", None, extra_fn=_inherit_extra
    ) if not inherit_sub.empty else []
    charts["divorceTotal"] = len(divorce_sub)
    charts["inheritTotal"] = len(inherit_sub)

    # Chart 4: Divorce analysis — initiator pie + cross-analysis (initiator × reason, stacked by ending)
    divorce_analysis = {"initiatorDist": [], "crossAnalysis": []}
    if not divorce_sub.empty:
        # Initiator distribution (pie chart)
        if init_col in divorce_sub.columns:
            ic = divorce_sub[divorce_sub[init_col].str.strip() != ""][init_col].value_counts()
            divorce_analysis["initiatorDist"] = [{"name": k, "count": int(v)} for k, v in ic.items()]
        # Cross-analysis: for each initiator, show reason distribution stacked by ending
        if init_col in divorce_sub.columns and reason_col in divorce_sub.columns and "終結情形大分類" in divorce_sub.columns:
            cross = []
            for initiator in ["男方", "女方", "雙方"]:
                init_sub = divorce_sub[divorce_sub[init_col] == initiator]
                if init_sub.empty:
                    continue
                reason_ending = init_sub[init_sub[reason_col].str.strip() != ""]
                if reason_ending.empty:
                    continue
                # Build stacked bar: group by reason, segment by ending
                cross.append({
                    "initiator": initiator,
                    "stack": _build_stacked_bar(reason_ending, reason_col, "終結情形大分類", top_n=6)
                })
            divorce_analysis["crossAnalysis"] = cross
    charts["divorceAnalysis"] = divorce_analysis
    return charts

# ════════════════════════════════════════════════════════════
#  Judgment List (paginated, with detail)
# ════════════════════════════════════════════════════════════
def get_judgment_list(df, case_type, page=0, page_size=10):
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
        item = {"jid": jid, "court": clean(first.get(court_col, "")), "cause": clean(first.get(cause_col, ""))}
        if case_type == "criminal_litigation":
            item["ending"] = clean(first.get("c0_全案終結情形", ""))
            item["cls"] = clean(first.get("案件分類", ""))
            item["defendants"] = int(first.get("判決被告人數", 1)) if "判決被告人數" in rows.columns else 1
            item["law"] = clean(first.get("定罪法條", ""))
            result_col = "c11_被告罪名裁判結果" if "c11_被告罪名裁判結果" in rows.columns else "罪名裁判結果"
            item["result"] = clean(first.get(result_col, ""))
            item["sentence"] = clean(first.get("c11_宣告有期徒刑", ""))
            item["probation"] = first.get("c1_是否宣告緩刑", "") == "1"
            item["defense"] = clean(first.get("c1_辯護及代理", ""))
            # Flags
            tags = []
            for col, label in CRIME_FLAGS:
                if col in rows.columns and first.get(col, "") == "1": tags.append(label)
            item["tags"] = tags
        else:
            end_col = "終結情形大分類" if "終結情形大分類" in df.columns else ""
            item["ending"] = clean(first.get(end_col, ""))
            if "律師代理情形" in df.columns:
                item["lawyer"] = clean(first.get("律師代理情形", ""))
            if "訴訟標的金額級距" in df.columns:
                item["amountTier"] = clean(first.get("訴訟標的金額級距", ""))
            if "訴訟標的金額" in df.columns:
                item["amount"] = clean(first.get("訴訟標的金額", ""))
            if "案由大分類" in df.columns:
                item["causeCat"] = clean(first.get("案由大分類", ""))
            if case_type == "family_litigation":
                init_col = "主動離婚者" if "主動離婚者" in df.columns else "c0_主動離婚者"
                reason_col = "離婚原因" if "離婚原因" in df.columns else "c0_離婚原因"
                item["initiator"] = clean(first.get(init_col, ""))
                item["divorceReason"] = clean(first.get(reason_col, ""))
        items.append(item)
    return {"items": items, "total": total, "totalPages": total_pages}

# ════════════════════════════════════════════════════════════
#  Unified Compute
# ════════════════════════════════════════════════════════════
def _compute_response(case_type, df, params=None, page=0, page_size=10):
    if case_type == "criminal_litigation":
        stats = criminal_stats(df)
        charts = criminal_charts(df, params)
    elif case_type == "civil_litigation":
        stats = civil_stats(df)
        charts = civil_charts(df)
    elif case_type == "family_litigation":
        stats = family_stats(df)
        charts = family_charts(df)
    else:
        return {"stats": {}, "charts": {}, "judgments": {"items": [], "total": 0, "totalPages": 1}}
    judgments = get_judgment_list(df, case_type, page, page_size)
    return {"stats": stats, "charts": charts, "judgments": judgments, "filteredRows": len(df)}

# ════════════════════════════════════════════════════════════
#  API Routes
# ════════════════════════════════════════════════════════════
@app.get("/api/types")
async def get_types():
    return [
        {"key": k, "label": v["label"], "category": v["category"], "available": k in DATA, "rowCount": len(DATA.get(k, []))}
        for k, v in CASE_TYPES.items()
    ]

@app.get("/api/{case_type}/options")
async def get_options(case_type: str):
    if case_type not in DATA:
        return JSONResponse(status_code=404, content={"error": f"Unknown type: {case_type}"})
    df = DATA[case_type]
    if case_type == "criminal_litigation": return criminal_filter_options(df)
    elif case_type == "civil_litigation": return civil_filter_options(df)
    elif case_type == "family_litigation": return family_filter_options(df)
    return {}

@app.get("/api/{case_type}/data")
async def get_data(
    case_type: str,
    court: Optional[str] = None, ym_min: Optional[str] = None, ym_max: Optional[str] = None,
    ending: Optional[str] = None, logic: Optional[str] = None,
    page: int = Query(0, ge=0), page_size: int = Query(10, ge=1, le=100),
    cls: Optional[str] = None, defense: Optional[str] = None, procedure: Optional[str] = None,
    probation: Optional[str] = None, article: Optional[str] = None, result: Optional[str] = None,
    security: Optional[str] = None, compensation: Optional[str] = None, confiscation: Optional[str] = None,
    probcond: Optional[str] = None, dv: Optional[str] = None, crime_flags: Optional[str] = None,
    aggravation: Optional[str] = None, mitigation: Optional[str] = None,
    action: Optional[str] = None, subject: Optional[str] = None, lawsuit_type: Optional[str] = None,
    amount_tier: Optional[str] = None, lawyer: Optional[str] = None, national_comp: Optional[str] = None,
    agency_type: Optional[str] = None, comp_type: Optional[str] = None,
    public_type: Optional[str] = None, election_type: Optional[str] = None,
    cause: Optional[str] = None, initiator: Optional[str] = None, divorce_reason: Optional[str] = None,
):
    if case_type not in DATA:
        return JSONResponse(status_code=404, content={"error": f"Unknown type: {case_type}"})

    logic_dict = {}
    if logic:
        try: logic_dict = json.loads(logic)
        except: pass

    params = {k: v for k, v in {
        "court": court, "ym_min": ym_min, "ym_max": ym_max, "ending": ending,
        "cls": cls, "defense": defense, "procedure": procedure, "probation": probation,
        "article": article, "result": result,
        "security": security, "compensation": compensation, "confiscation": confiscation,
        "probcond": probcond, "dv": dv, "crime_flags": crime_flags,
        "aggravation": aggravation, "mitigation": mitigation,
        "action": action, "subject": subject, "lawsuit_type": lawsuit_type,
        "amount_tier": amount_tier, "lawyer": lawyer, "national_comp": national_comp,
        "agency_type": agency_type, "comp_type": comp_type, "public_type": public_type,
        "election_type": election_type,
        "cause": cause, "initiator": initiator, "divorce_reason": divorce_reason,
    }.items() if v is not None}

    # If no filters, return cached default (but still paginate)
    has_filters = any(v for k, v in params.items() if k not in ("ym_min", "ym_max") and v)
    has_ym = bool(params.get("ym_min") or params.get("ym_max"))

    if not has_filters and not has_ym and case_type in CACHE:
        cached = CACHE[case_type]
        # Re-paginate
        df = DATA[case_type]
        judgments = get_judgment_list(df, case_type, page, page_size)
        return {**cached, "judgments": judgments}

    df = DATA[case_type]
    if case_type == "criminal_litigation":
        filtered = apply_criminal_filters(df, params, logic_dict)
        stats = criminal_stats(filtered)
        charts = criminal_charts(filtered, params)
    elif case_type == "civil_litigation":
        filtered = apply_civil_filters(df, params, logic_dict)
        stats = civil_stats(filtered)
        charts = civil_charts(filtered)
    elif case_type == "family_litigation":
        filtered = apply_family_filters(df, params, logic_dict)
        stats = family_stats(filtered)
        charts = family_charts(filtered)
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown type"})

    judgments = get_judgment_list(filtered, case_type, page, page_size)
    return {"stats": stats, "charts": charts, "judgments": judgments, "filteredRows": len(filtered)}

@app.get("/health")
async def health():
    return {"status": "ok", "types_loaded": list(DATA.keys())}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
