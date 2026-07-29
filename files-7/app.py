"""
Sustainability & Demand Intelligence Platform — Market Validation App
SINGLE-FILE VERSION.

Everything lives in this one file. There are no imports between your own
files, so no module can fail to upload. Your repo needs exactly two things:

    app.py       <- this file
    data/        <- the six CSV files

Run locally:  streamlit run app.py
"""
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Demand Intelligence — Market Validation",
                   page_icon="🍽️", layout="wide",
                   initial_sidebar_state="expanded")


# ======================================================================
# SHARED HELPERS
# ======================================================================

DATA = Path(__file__).resolve().parent / "data"


def _require(filename: str) -> Path:
    """Resolve a data file, failing with a readable message instead of a traceback."""
    path = DATA / filename
    if not path.exists():
        st.error(
            f"**Data file missing: `data/{filename}`**\n\n"
            f"Looked in: `{DATA}`\n\n"
            "If you are deploying from GitHub, confirm the `data/` folder and all "
            "five CSV files were actually committed — folder uploads through the "
            "GitHub web interface sometimes drop files silently."
        )
        st.stop()
    return path


LIKERT = [
    "q16_willing_share_data",
    "q17_food_cost_major_expense",
    "q18_pressure_reduce_cost_per_meal",
    "q19_shortage_worse_than_waste",
    "q20_trust_data_over_experience",
    "q21_esg_reporting_required",
    "q22_staff_would_adopt_easily",
    "q23_need_proof_before_paying",
]

LIKERT_LABEL = {
    "q16_willing_share_data": "Willing to share data (NDA)",
    "q17_food_cost_major_expense": "Food cost is major expense",
    "q18_pressure_reduce_cost_per_meal": "Pressure to cut cost/meal",
    "q19_shortage_worse_than_waste": "Shortage worse than waste",
    "q20_trust_data_over_experience": "Trusts data over experience",
    "q21_esg_reporting_required": "ESG reporting required",
    "q22_staff_would_adopt_easily": "Staff would adopt easily",
    "q23_need_proof_before_paying": "Needs proof before paying",
}

NUMERIC = [
    "q4_meals_per_day",
    "q5_num_outlets",
    "q7_monthly_food_spend_lakh",
    "q9_est_waste_pct",
    "q24_pilot_likelihood",
    "q27_min_accuracy_required_pct",
]

RANGES = {
    "q4_meals_per_day": (30, 20000),
    "q9_est_waste_pct": (0.5, 60),
    "q7_monthly_food_spend_lakh": (0.1, 5000),
    "q27_min_accuracy_required_pct": (50, 99),
    "q5_num_outlets": (1, 100),
}

MULTISELECT = {
    "q10_surplus_handling": "Surplus handling",
    "q13_digital_records_kept": "Digital records kept",
    "q28_adoption_barriers": "Adoption barriers",
    "q29_features_wanted": "Features wanted",
    "q30_purchase_approvers": "Purchase approvers",
}

PALETTE = ["#1F3864", "#2E75B6", "#7DA7D9", "#C55A11", "#E8A33D", "#548235"]


# ──────────────────────────────────────────────────────────────────
# Loading & cleaning
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    return pd.read_csv(_require("survey_responses.csv"))


@st.cache_data(show_spinner=False)
def load_answer_key() -> pd.DataFrame:
    return pd.read_csv(_require("segment_answer_key.csv"))


@st.cache_data(show_spinner=False)
def load_onehot() -> pd.DataFrame:
    return pd.read_csv(_require("multiselect_onehot.csv"))


@st.cache_data(show_spinner=False)
def load_register() -> pd.DataFrame:
    return pd.read_csv(_require("chart_register.csv"))


def _parse_money(v):
    """'Rs 12,500' -> 12500.0"""
    if pd.isna(v):
        return np.nan
    s = re.sub(r"[^0-9.]", "", str(v))
    return float(s) if s else np.nan


@st.cache_data(show_spinner=False)
def clean() -> tuple[pd.DataFrame, dict]:
    """Return the cleaned frame plus an audit log of what was changed."""
    df = load_raw()
    audit = {"rows_raw": len(df)}

    # 1. duplicate submissions
    dupes = int(df["respondent_id"].duplicated().sum())
    df = df.drop_duplicates(subset="respondent_id").reset_index(drop=True)
    audit["duplicates_removed"] = dupes

    # 2. dirty category text
    df["q2_city_tier"] = (
        df["q2_city_tier"].astype(str).str.strip().str.lower()
        .str.replace("-", " ", regex=False).str.title()
        .replace("Nan", np.nan)
    )
    df["q1_org_type"] = (
        df["q1_org_type"].astype(str).str.strip().str.capitalize()
        .replace("Nan", np.nan)
    )

    # 3. money stored as text
    df["wtp"] = df["q26_max_wtp_per_site_month_inr"].map(_parse_money)

    # 4. impossible values -> NaN
    impossible = 0
    for col, (lo, hi) in RANGES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & ~df[col].between(lo, hi)
        impossible += int(bad.sum())
        df.loc[bad, col] = np.nan
    audit["impossible_values"] = impossible

    # 5. empty strings -> NaN
    for col in MULTISELECT:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    # 6. missingness record BEFORE imputation
    audit["missing_pct"] = (df.isna().mean() * 100).round(1).to_dict()

    # 7. derived features
    df["dig_count"] = (
        df["q13_digital_records_kept"].fillna("")
        .apply(lambda s: 0 if str(s).strip() in ("", "None") else len(str(s).split(";")))
    )
    df["hist_depth"] = df["q15_historical_data_depth"].map({
        "Can share 12+ months history": 3,
        "Can share 3-6 months history": 2,
        "Only current month available": 1,
        "No historical records": 0,
    })
    df["maturity"] = df["q8_leftover_tracking_method"].map({
        "Never": 0, "Visual estimate": 1, "Weigh occasionally": 2,
        "Weigh every service": 3, "Digital system": 4,
    })
    df["adopter"] = (df["q24_pilot_likelihood"] >= 4).astype(int)

    # 8. missing-indicator for the MNAR column, then median-impute
    df["waste_pct_missing"] = df["q9_est_waste_pct"].isna().astype(int)
    for col in LIKERT + NUMERIC + ["dig_count", "hist_depth", "maturity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    audit["rows_clean"] = len(df)
    return df, audit


def explode_multiselect(df: pd.DataFrame, col: str) -> pd.Series:
    """Count of each item in a semicolon-delimited multi-select column."""
    return (
        df[col].dropna().str.split(";").explode().str.strip()
        .loc[lambda s: s.ne("")].value_counts()
    )


# ──────────────────────────────────────────────────────────────────
# Association rules (hand-rolled apriori — no extra dependency)
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def association_rules(min_support=0.05, min_conf=0.40, min_lift=1.2) -> pd.DataFrame:
    oh = load_onehot().set_index("respondent_id")
    sup = oh.mean()
    rows = []
    for a, b in combinations(oh.columns, 2):
        both = float((oh[a] & oh[b]).mean())
        if both < min_support:
            continue
        for x, y in ((a, b), (b, a)):
            conf = both / sup[x] if sup[x] else 0
            lift = conf / sup[y] if sup[y] else 0
            if conf >= min_conf and lift >= min_lift:
                rows.append({
                    "Antecedent": x, "Consequent": y,
                    "Support": round(both, 3),
                    "Confidence": round(conf, 3),
                    "Lift": round(lift, 2),
                    "Antecedent support": round(float(sup[x]), 3),
                })
    out = pd.DataFrame(rows)
    return out.sort_values("Lift", ascending=False).reset_index(drop=True) if len(out) else out


# ──────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────
def inr(x) -> str:
    if pd.isna(x):
        return "—"
    return f"₹{x:,.0f}"


def apply_theme(fig, height=420, legend_bottom=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        font=dict(family="Inter, Segoe UI, Arial", size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        title_font=dict(size=16),
    )
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="left", x=0))
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def page_header(title: str, subtitle: str = "", icon: str = ""):
    st.markdown(
        f"<h1 style='margin-bottom:0.1rem'>{icon} {title}</h1>"
        f"<p style='color:#6b7280;margin-top:0;font-size:1.02rem'>{subtitle}</p>"
        "<hr style='margin:0.6rem 0 1.4rem 0;border:none;border-top:2px solid #1F3864'>",
        unsafe_allow_html=True,
    )


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Shared filter controls. Returns the filtered frame."""
    st.sidebar.markdown("### Filters")
    orgs = sorted(df["q1_org_type"].dropna().unique())
    tiers = sorted(df["q2_city_tier"].dropna().unique())

    sel_org = st.sidebar.multiselect("Organisation type", orgs, default=orgs)
    sel_tier = st.sidebar.multiselect("City tier", tiers, default=tiers)
    lo, hi = int(df["q4_meals_per_day"].min()), int(df["q4_meals_per_day"].max())
    sel_meals = st.sidebar.slider("Meals per day", lo, hi, (lo, hi), step=50)

    out = df[
        df["q1_org_type"].isin(sel_org)
        & df["q2_city_tier"].isin(sel_tier)
        & df["q4_meals_per_day"].between(*sel_meals)
    ]
    st.sidebar.caption(f"**{len(out)}** of {len(df)} respondents selected")
    st.sidebar.markdown("---")
    st.sidebar.warning(
        "**Synthetic data.** Generated from an assumed market model to "
        "demonstrate analytical method. Not evidence about the real market.",
        icon="⚠️",
    )
    return out


# ======================================================================
def page_overview():
    # Make the app folder importable no matter where Streamlit is launched from
    # (repo root, sub-folder, or Streamlit Cloud's /mount/src/... path).




    df_all, audit = clean()
    df = sidebar_filters(df_all)

    page_header(
        "Market Validation Dashboard",
        "AI demand forecasting for buffets, cafeterias and institutional kitchens — India",
        "🍽️",
    )

    if df.empty:
        st.warning("No respondents match the current filters.")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────
    adopters = df["adopter"].mean() * 100
    median_wtp = df["wtp"].median()
    mean_waste = df["q9_est_waste_pct"].mean()
    tam_month = (df["adopter"] * df["wtp"].fillna(0)).sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Respondents", f"{len(df):,}")
    c2.metric("Adoption intent", f"{adopters:.0f}%",
              help="Share rating pilot likelihood 4 or 5 out of 5")
    c3.metric("Median WTP", inr(median_wtp), help="₹ per site per month")
    c4.metric("Mean stated waste", f"{mean_waste:.1f}%")
    c5.metric("Revenue in sample", inr(tam_month),
              help="Sum of stated WTP across likely adopters, per month")

    st.caption(
        "Adoption intent is *stated*, not revealed. Survey intent typically "
        "overstates real conversion by a wide margin — treat these as a ranking "
        "of segments, not a forecast of sales."
    )

    st.markdown("")

    # ── Row 1 ─────────────────────────────────────────────────────────
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Adoption intent by organisation type")
        tab = (df.groupby("q1_org_type")
                 .agg(adoption=("adopter", "mean"),
                      n=("adopter", "size"),
                      wtp=("wtp", "median"))
                 .reset_index()
                 .sort_values("adoption"))
        tab["adoption"] *= 100
        fig = px.bar(tab, x="adoption", y="q1_org_type", orientation="h",
                     text=tab["adoption"].round(0).astype(int).astype(str) + "%",
                     labels={"adoption": "Adoption intent (%)", "q1_org_type": ""},
                     color="adoption", color_continuous_scale=["#C8D6E8", "#1F3864"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig, 380, legend_bottom=False),
                        width="stretch")
        st.caption(
            f"Sample sizes: " +
            " · ".join(f"{r.q1_org_type} n={r.n}" for r in tab.itertuples())
        )

    with right:
        st.subheader("Willingness to pay by city tier")
        fig = px.box(df.dropna(subset=["wtp"]), x="q2_city_tier", y="wtp",
                     color="q2_city_tier", points="outliers",
                     category_orders={"q2_city_tier": ["Tier 1", "Tier 2", "Tier 3"]},
                     labels={"wtp": "₹ per site per month", "q2_city_tier": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_theme(fig, 380, legend_bottom=False),
                        width="stretch")
        st.caption("Tier 1 commands a clear premium — price the tiers separately.")

    # ── Row 2 ─────────────────────────────────────────────────────────
    st.markdown("---")
    l2, r2 = st.columns(2)

    with l2:
        st.subheader("Waste rate vs measurement maturity")
        order = ["Never", "Visual estimate", "Weigh occasionally",
                 "Weigh every service", "Digital system"]
        sub = df[df["waste_pct_missing"] == 0]
        fig = px.box(sub, x="q8_leftover_tracking_method", y="q9_est_waste_pct",
                     category_orders={"q8_leftover_tracking_method": order},
                     color="q8_leftover_tracking_method",
                     labels={"q9_est_waste_pct": "Stated waste (%)",
                             "q8_leftover_tracking_method": ""})
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(apply_theme(fig, 400, legend_bottom=False),
                        width="stretch")
        n_missing = int(df["waste_pct_missing"].sum())
        st.warning(
            f"**{n_missing} respondents ({n_missing/len(df)*100:.0f}%) could not "
            "state a waste figure at all** — overwhelmingly those who never track it. "
            "They are excluded from this chart, which biases it toward "
            "measurement-mature sites and understates the true market problem. "
            "This missingness is a finding, not a nuisance.",
            icon="🔍",
        )

    with r2:
        st.subheader("Top adoption barriers")
        barriers = explode_multiselect(df, "q28_adoption_barriers").head(8)
        fig = px.bar(x=barriers.values / len(df) * 100, y=barriers.index,
                     orientation="h",
                     labels={"x": "% of respondents citing", "y": ""},
                     text=[f"{v/len(df)*100:.0f}%" for v in barriers.values])
        fig.update_traces(marker_color="#C55A11", textposition="outside",
                          cliponaxis=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(apply_theme(fig, 400, legend_bottom=False),
                        width="stretch")
        st.caption("Association rules on the next page show these do not occur "
                   "independently — they cluster into two distinct objection types.")

    # ── Row 3 ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Feature demand")
    feat = explode_multiselect(df, "q29_features_wanted")
    fig = px.bar(x=feat.index, y=feat.values / len(df) * 100,
                 labels={"x": "", "y": "% wanting this feature"})
    fig.update_traces(marker_color="#2E75B6")
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(apply_theme(fig, 380, legend_bottom=False),
                    width="stretch")

    st.markdown("---")
    with st.expander("What this app contains"):
        st.markdown(
            """
    | Page | Analysis | Business question |
    |---|---|---|
    | **Data Quality** | Cleaning audit, missingness patterns | Can we trust the inputs? |
    | **Segmentation** | k-means clustering | Which buyer personas exist? |
    | **Adoption Model** | Logistic regression / gradient boosting | Who do we sell to first? |
    | **Pricing Model** | OLS regression on willingness to pay | What do we charge whom? |
    | **Association Rules** | Apriori on multi-select responses | Which objections and features travel together? |
    | **Dashboard Roadmap** | 107-chart phased register | What do we build, and when? |

    **Data caveat.** The survey responses are synthetic — generated from an assumed
    model of the Indian food service market to demonstrate analytical method and
    decision logic. Every relationship recovered here was designed into the
    generator. This is valid as a demonstration of pipeline and reasoning; it is
    **not** evidence about the real market and must not be presented as such.
    """
        )



# ======================================================================
def page_quality():
    # pages/ lives one level below the app root — add the root to sys.path so
    # `common` resolves regardless of the launch directory or host platform.




    raw = load_raw()
    df, audit = clean()

    page_header("Data Quality & Cleaning Audit",
                "What was wrong with the raw responses, and what was done about it",
                "🧹")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", f"{audit['rows_raw']:,}")
    c2.metric("Duplicates removed", audit["duplicates_removed"])
    c3.metric("Impossible values voided", audit["impossible_values"])
    c4.metric("Clean rows", f"{audit['rows_clean']:,}")

    st.markdown("---")

    # ── Missingness ───────────────────────────────────────────────────
    st.subheader("Missingness by column")

    miss = (pd.Series(audit["missing_pct"])
            .sort_values(ascending=False)
            .loc[lambda s: s > 0])

    fig = px.bar(x=miss.values, y=miss.index, orientation="h",
                 labels={"x": "% missing", "y": ""},
                 text=[f"{v:.1f}%" for v in miss.values])
    fig.update_traces(marker_color="#C55A11", textposition="outside", cliponaxis=False)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(apply_theme(fig, 380, legend_bottom=False), width="stretch")

    st.markdown("### The missingness is not random")

    left, right = st.columns([1, 1])

    with left:
        tracking = raw.groupby("q8_leftover_tracking_method")["q9_est_waste_pct"] \
                      .apply(lambda s: s.isna().mean() * 100).sort_values()
        fig = px.bar(x=tracking.index, y=tracking.values,
                     labels={"x": "How they track leftovers",
                             "y": "% who could not state a waste figure"},
                     text=[f"{v:.0f}%" for v in tracking.values])
        fig.update_traces(marker_color="#1F3864", textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(apply_theme(fig, 400, legend_bottom=False), width="stretch")

    with right:
        st.info(
            """
    **This is MNAR — missing not at random.**

    The respondents who cannot state a waste percentage are overwhelmingly
    those who never measure it. They are also, almost certainly, the ones
    with the worst waste problem.

    **Consequences you must state in any write-up:**

    - Dropping these rows biases the sample toward measurement-mature sites
      and **understates** the market's waste problem
    - Mean-imputing pulls them toward the average, which is exactly wrong —
      their true values are likely above it
    - The approach used here: **retain a missing-indicator flag**
      (`waste_pct_missing`) alongside median imputation, so models can use
      the fact of missingness as a signal in its own right

    Commercially, this is the finding: **the sites that can't answer the
    question are your best prospects**, because they have the most to gain
    and no way of knowing it.
    """,
            icon="🔍",
        )

    st.markdown("---")

    # ── Dirty data catalogue ──────────────────────────────────────────
    st.subheader("Data problems found and handled")

    problems = pd.DataFrame([
        ["Duplicate submissions", f"{audit['duplicates_removed']} rows",
         "Same respondent_id submitted twice",
         "Dropped on respondent_id, keeping first"],
        ["MNAR missingness", "q9_est_waste_pct (~25%)",
         "Concentrated among non-trackers",
         "Missing-indicator flag + median imputation"],
        ["MNAR missingness", "q7_monthly_food_spend_lakh (~9%)",
         "Owners and banquet venues withhold spend",
         "Median imputation; flagged as a limitation"],
        ["MCAR missingness", "7 columns, 3–9%",
         "No pattern detected",
         "Median or mode imputation"],
        ["Inconsistent categories", "q2_city_tier",
         "'tier-1', ' Tier 2 ', 'TIER 3'",
         "Strip, lowercase, replace hyphen, title-case"],
        ["Inconsistent categories", "q1_org_type",
         "Case and trailing-whitespace variants",
         "Strip and capitalise"],
        ["Numeric stored as text", "q26_max_wtp (~16%)",
         "Formatted as 'Rs 12,500'",
         "Regex strip non-numerics, coerce to float"],
        ["Impossible values", f"{audit['impossible_values']} cells",
         "Meals/day of 99999 and 0; negative spend; waste of 110% and −3%",
         "Range filter, then treat as missing"],
        ["Empty strings vs NaN", "q30_purchase_approvers",
         "Blank string is not counted as missing by default",
         "Converted to NaN before any count"],
    ], columns=["Problem type", "Where", "What it looked like", "How it was handled"])

    st.dataframe(problems, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Distribution inspector ────────────────────────────────────────
    st.subheader("Distribution inspector")
    num_cols = ["q4_meals_per_day", "q7_monthly_food_spend_lakh", "q9_est_waste_pct",
                "wtp", "q27_min_accuracy_required_pct", "q5_num_outlets"]
    col = st.selectbox("Variable", num_cols, index=0)

    a, b = st.columns(2)
    with a:
        st.markdown("**Raw (before cleaning)**")
        rawnum = pd.to_numeric(
            raw[col] if col in raw.columns else raw["q26_max_wtp_per_site_month_inr"]
            .astype(str).str.replace(r"[^0-9.]", "", regex=True).replace("", np.nan),
            errors="coerce")
        fig = px.histogram(rawnum.dropna(), nbins=40, labels={"value": col})
        fig.update_traces(marker_color="#C55A11")
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_theme(fig, 300, legend_bottom=False), width="stretch")
        st.caption(f"n={rawnum.notna().sum()} · min={rawnum.min():,.1f} · max={rawnum.max():,.1f}")

    with b:
        st.markdown("**Cleaned**")
        fig = px.histogram(df[col].dropna(), nbins=40, labels={"value": col})
        fig.update_traces(marker_color="#1F3864")
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_theme(fig, 300, legend_bottom=False), width="stretch")
        st.caption(f"n={df[col].notna().sum()} · min={df[col].min():,.1f} · max={df[col].max():,.1f}")

    with st.expander("View the cleaned dataset"):
        st.dataframe(df.head(200), width="stretch")
        st.download_button("Download cleaned CSV",
                           df.to_csv(index=False).encode(),
                           "survey_cleaned.csv", "text/csv")



# ======================================================================
def page_segmentation():
    # pages/ lives one level below the app root — add the root to sys.path so
    # `common` resolves regardless of the launch directory or host platform.




    df_all, _ = clean()
    df = sidebar_filters(df_all)

    page_header("Buyer Segmentation",
                "k-means clustering on attitudes and data readiness — who are we actually selling to?",
                "🧭")

    FEATURES = LIKERT + ["dig_count", "hist_depth", "maturity"]

    k = st.sidebar.slider("Number of clusters (k)", 2, 7, 4)

    X = StandardScaler().fit_transform(df[FEATURES])
    km = KMeans(n_clusters=k, n_init=25, random_state=0)
    labels = km.fit_predict(X)
    df = df.assign(cluster=labels)

    sil = silhouette_score(X, labels)

    # ── diagnostics ───────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Clusters", k)
    c2.metric("Silhouette score", f"{sil:.3f}")

    key = load_answer_key()
    merged = df[["respondent_id", "cluster"]].merge(key, on="respondent_id", how="inner")
    ari = adjusted_rand_score(merged["true_segment"], merged["cluster"]) if len(merged) else np.nan
    c3.metric("ARI vs held-out labels", f"{ari:.3f}",
              help="Adjusted Rand Index against the generator's true segments. "
                   "1.0 = perfect recovery, 0 = random. Available here only because "
                   "the data is synthetic — you would never have this on real data.")

    if sil < 0.15:
        st.info(
            f"**Silhouette of {sil:.3f} is low, and that is normal for attitudinal "
            "survey data.** Human attitudes form overlapping gradients, not separated "
            "blobs. Judge the solution on whether the personas are *actionable and "
            "distinguishable*, not on the silhouette. Reporting a weak silhouette "
            "honestly is stronger than hiding it.",
            icon="ℹ️",
        )

    st.markdown("---")

    # ── elbow & silhouette curve ──────────────────────────────────────
    with st.expander("How many clusters? (elbow and silhouette diagnostics)"):
        rows = []
        for kk in range(2, 9):
            m = KMeans(n_clusters=kk, n_init=15, random_state=0).fit(X)
            rows.append({"k": kk, "Inertia": m.inertia_,
                         "Silhouette": silhouette_score(X, m.labels_)})
        diag = pd.DataFrame(rows)
        a, b = st.columns(2)
        with a:
            fig = px.line(diag, x="k", y="Inertia", markers=True, title="Elbow curve")
            st.plotly_chart(apply_theme(fig, 320, legend_bottom=False), width="stretch")
        with b:
            fig = px.line(diag, x="k", y="Silhouette", markers=True, title="Silhouette by k")
            st.plotly_chart(apply_theme(fig, 320, legend_bottom=False), width="stretch")

    # ── persona profile ───────────────────────────────────────────────
    st.subheader("Persona profiles")

    profile = df.groupby("cluster").agg(
        n=("respondent_id", "size"),
        adoption=("adopter", "mean"),
        wtp=("wtp", "median"),
        meals=("q4_meals_per_day", "median"),
        waste=("q9_est_waste_pct", "mean"),
        readiness=("dig_count", "mean"),
        trust=("q20_trust_data_over_experience", "mean"),
        fear=("q19_shortage_worse_than_waste", "mean"),
        esg=("q21_esg_reporting_required", "mean"),
    ).reset_index()
    profile["adoption"] = (profile["adoption"] * 100).round(0)
    profile["share"] = (profile["n"] / len(df) * 100).round(0)

    # priority score: adoption x wtp x readiness, normalised
    p = profile.copy()
    for c in ["adoption", "wtp", "readiness"]:
        rng = p[c].max() - p[c].min()
        p[c + "_n"] = (p[c] - p[c].min()) / rng if rng else 0.5
    profile["priority"] = (
        0.45 * p["adoption_n"] + 0.35 * p["wtp_n"] + 0.20 * p["readiness_n"]
    ).round(2)

    display = profile[["cluster", "n", "share", "adoption", "wtp", "meals",
                       "waste", "readiness", "priority"]].copy()
    display.columns = ["Cluster", "n", "Share %", "Adoption %", "Median WTP (₹)",
                       "Median meals/day", "Mean waste %", "Digital records",
                       "Priority score"]
    st.dataframe(
        display.sort_values("Priority score", ascending=False)
               .style.format({"Median WTP (₹)": "{:,.0f}", "Median meals/day": "{:,.0f}",
                              "Mean waste %": "{:.1f}", "Digital records": "{:.1f}"})
               .background_gradient(subset=["Priority score"], cmap="Blues"),
        width="stretch", hide_index=True)

    best = profile.sort_values("priority", ascending=False).iloc[0]
    st.success(
        f"**Target Cluster {int(best.cluster)} first** — {best.adoption:.0f}% adoption "
        f"intent, median WTP {inr(best.wtp)}, {best.readiness:.1f} digital record types "
        f"already kept. Highest combination of intent, budget and onboardability.",
        icon="🎯",
    )

    st.markdown("---")

    # ── radar ─────────────────────────────────────────────────────────
    l, r = st.columns([1.1, 1])

    with l:
        st.subheader("Attitude fingerprint")
        z = df.groupby("cluster")[LIKERT].mean()
        z = (z - df[LIKERT].mean()) / df[LIKERT].std()   # standardised deviation
        fig = go.Figure()
        for i, (cl, row) in enumerate(z.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=row.values, theta=[LIKERT_LABEL[c] for c in LIKERT],
                fill="toself", name=f"Cluster {cl}",
                line_color=PALETTE[i % len(PALETTE)], opacity=0.65))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1.6, 1.6])))
        st.plotly_chart(apply_theme(fig, 480), width="stretch")
        st.caption("Values are standard deviations from the sample mean. "
                   "Distance from centre = how distinctive that attitude is.")

    with r:
        st.subheader("Cluster separation (PCA projection)")
        pcs = PCA(n_components=2, random_state=0).fit_transform(X)
        plot = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        plot["Cluster"] = [f"Cluster {c}" for c in labels]
        plot["WTP"] = df["wtp"].fillna(0).values
        fig = px.scatter(plot, x="PC1", y="PC2", color="Cluster", size="WTP",
                         size_max=16, opacity=0.75)
        st.plotly_chart(apply_theme(fig, 480), width="stretch")
        st.caption("Two components only — overlap here is expected and does not "
                   "invalidate the clusters.")

    st.markdown("---")

    # ── composition ───────────────────────────────────────────────────
    st.subheader("What each cluster is made of")
    tabs = st.tabs(["Organisation type", "City tier", "Production decided by",
                    "Preferred pricing"])
    for tab, col, title in zip(
            tabs,
            ["q1_org_type", "q2_city_tier", "q14_production_decided_by",
             "q25_pricing_model_preferred"],
            ["Organisation type", "City tier", "Production decided by", "Preferred pricing"]):
        with tab:
            ct = pd.crosstab(df["cluster"], df[col], normalize="index") * 100
            fig = px.bar(ct.reset_index().melt(id_vars="cluster"),
                         x="cluster", y="value", color=col,
                         labels={"value": "% of cluster", "cluster": "Cluster"})
            fig.update_layout(barmode="stack")
            st.plotly_chart(apply_theme(fig, 400), width="stretch")

    st.download_button("Download cluster assignments",
                       df[["respondent_id", "cluster"]].to_csv(index=False).encode(),
                       "cluster_assignments.csv", "text/csv")



# ======================================================================
def page_adoption():
    # pages/ lives one level below the app root — add the root to sys.path so
    # `common` resolves regardless of the launch directory or host platform.




    df_all, _ = clean()
    df = sidebar_filters(df_all)

    page_header("Adoption Prediction",
                "Who will pilot this? Classification model turned into a sales targeting rule",
                "🎯")

    FEATURES = LIKERT + ["dig_count", "hist_depth", "maturity",
                         "q4_meals_per_day", "q9_est_waste_pct", "waste_pct_missing"]
    NICE = {**LIKERT_LABEL,
            "dig_count": "Digital records kept (count)",
            "hist_depth": "Historical data depth",
            "maturity": "Leftover tracking maturity",
            "q4_meals_per_day": "Meals per day",
            "q9_est_waste_pct": "Stated waste %",
            "waste_pct_missing": "Could not state waste %"}

    model_name = st.sidebar.radio("Model", ["Logistic regression", "Random forest"])

    X = df[FEATURES].values
    y = df["adopter"].values

    if y.sum() < 15 or (len(y) - y.sum()) < 15:
        st.warning("Not enough adopters or non-adopters in the current filter to fit a model.")
        st.stop()

    if model_name == "Logistic regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    else:
        model = RandomForestClassifier(500, min_samples_leaf=4, random_state=0)

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc_cv = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    acc_cv = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]

    threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                                  help="Lower = chase more leads, waste more sales time. "
                                       "Higher = fewer, better-qualified leads.")
    pred = (proba >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary",
                                                       zero_division=0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC (5-fold CV)", f"{auc_cv.mean():.3f}", f"± {auc_cv.std():.3f}")
    c2.metric("Accuracy (CV)", f"{acc_cv.mean():.3f}")
    c3.metric("Precision", f"{prec:.3f}", help="Of leads we chase, how many convert")
    c4.metric("Recall", f"{rec:.3f}", help="Of real adopters, how many we find")
    c5.metric("Base rate", f"{y.mean():.1%}", help="Adoption rate with no model at all")

    st.markdown("---")

    l, r = st.columns([1, 1])

    with l:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(yte, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC {roc_auc_score(yte, proba):.3f})",
                                 line=dict(color="#1F3864", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="#999", dash="dash")))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(apply_theme(fig, 400), width="stretch")

    with r:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(yte, pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Not adopter", "Adopter"], y=["Not adopter", "Adopter"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig, 400, legend_bottom=False), width="stretch")
        fp, fn = cm[0, 1], cm[1, 0]
        st.caption(f"**{fp} false positives** (sales time wasted) · "
                   f"**{fn} false negatives** (real prospects missed). "
                   "Move the threshold in the sidebar to trade one against the other.")

    st.markdown("---")

    # ── drivers ───────────────────────────────────────────────────────
    st.subheader("What drives adoption")

    if model_name == "Logistic regression":
        coefs = model.named_steps["logisticregression"].coef_[0]
        imp = pd.DataFrame({"Feature": [NICE.get(f, f) for f in FEATURES],
                            "Effect": coefs}).sort_values("Effect")
        fig = px.bar(imp, x="Effect", y="Feature", orientation="h",
                     color=imp["Effect"] > 0,
                     color_discrete_map={True: "#548235", False: "#C55A11"},
                     labels={"Effect": "Standardised coefficient (log-odds)", "Feature": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_theme(fig, 460, legend_bottom=False), width="stretch")
        st.caption("Green increases adoption probability, orange decreases it. "
                   "Coefficients are on standardised features, so magnitudes are comparable.")
    else:
        pi = permutation_importance(model, Xte, yte, n_repeats=15, random_state=0,
                                    scoring="roc_auc")
        imp = pd.DataFrame({"Feature": [NICE.get(f, f) for f in FEATURES],
                            "Importance": pi.importances_mean,
                            "SD": pi.importances_std}).sort_values("Importance")
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", error_x="SD",
                     labels={"Importance": "Drop in ROC-AUC when shuffled", "Feature": ""})
        fig.update_traces(marker_color="#1F3864")
        st.plotly_chart(apply_theme(fig, 460, legend_bottom=False), width="stretch")
        st.caption("Permutation importance — how much test AUC falls when each "
                   "feature is randomly shuffled.")

    st.markdown("---")

    # ── lead scoring tool ─────────────────────────────────────────────
    st.subheader("Lead scoring tool")
    st.caption("Score a prospect against the model. This is the operational output — "
               "the model is only worth building if it changes who the sales team calls.")

    cols = st.columns(4)
    inputs = {}
    with cols[0]:
        inputs["q17_food_cost_major_expense"] = st.slider("Food cost is a major expense", 1, 5, 4)
        inputs["q18_pressure_reduce_cost_per_meal"] = st.slider("Pressure to cut cost/meal", 1, 5, 4)
        inputs["q16_willing_share_data"] = st.slider("Willing to share data", 1, 5, 3)
    with cols[1]:
        inputs["q19_shortage_worse_than_waste"] = st.slider("Shortage worse than waste", 1, 5, 4)
        inputs["q20_trust_data_over_experience"] = st.slider("Trusts data over experience", 1, 5, 3)
        inputs["q21_esg_reporting_required"] = st.slider("ESG reporting required", 1, 5, 3)
    with cols[2]:
        inputs["q22_staff_would_adopt_easily"] = st.slider("Staff would adopt easily", 1, 5, 3)
        inputs["q23_need_proof_before_paying"] = st.slider("Needs proof before paying", 1, 5, 4)
        inputs["dig_count"] = st.slider("Digital record types kept", 0, 6, 3)
    with cols[3]:
        inputs["hist_depth"] = st.slider("Historical data depth (0–3)", 0, 3, 2)
        inputs["maturity"] = st.slider("Leftover tracking maturity (0–4)", 0, 4, 2)
        inputs["q4_meals_per_day"] = st.number_input("Meals per day", 50, 12000, 800, 50)
        inputs["q9_est_waste_pct"] = st.number_input("Estimated waste %", 1.0, 45.0, 18.0, 0.5)

    inputs["waste_pct_missing"] = 0
    vec = np.array([[inputs[f] for f in FEATURES]])
    p = float(model.predict_proba(vec)[0, 1])

    g1, g2 = st.columns([1, 1.4])
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%"},
            title={"text": "Adoption probability"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#1F3864"},
                   "steps": [{"range": [0, 35], "color": "#F2DCDB"},
                             {"range": [35, 60], "color": "#FFEB9C"},
                             {"range": [60, 100], "color": "#C6EFCE"}],
                   "threshold": {"line": {"color": "black", "width": 3},
                                 "value": threshold * 100}}))
        st.plotly_chart(apply_theme(fig, 320, legend_bottom=False), width="stretch")

    with g2:
        if p >= 0.60:
            st.success(f"**Qualified lead ({p:.0%}).** Prioritise for direct outreach "
                       "and offer a paid pilot.", icon="✅")
        elif p >= 0.35:
            st.warning(f"**Nurture ({p:.0%}).** Offer the free 4-week shadow pilot to "
                       "remove risk before asking for budget.", icon="⚠️")
        else:
            st.error(f"**Deprioritise ({p:.0%}).** Low readiness or high resistance — "
                     "onboarding cost likely exceeds contract value.", icon="⛔")

        st.markdown(
            """
    **Honest limits on this model**

    - Trained on **stated intent**, not observed purchase. Survey intent
      systematically overstates real conversion — use this to *rank* prospects,
      not to forecast revenue.
    - Data is synthetic; the relationships were designed in. On real data,
      expect a substantially lower AUC.
    - The strongest signals (trust in data, fear of shortage) are attitudes
      you cannot observe before speaking to a prospect — so this is a
      **discovery-call scoring tool**, not a list-buying filter.
    """
        )



# ======================================================================
def page_pricing():
    # pages/ lives one level below the app root — add the root to sys.path so
    # `common` resolves regardless of the launch directory or host platform.




    df_all, _ = clean()
    df = sidebar_filters(df_all)

    page_header("Pricing & Willingness to Pay",
                "OLS regression on stated WTP — what drives price tolerance, and what to charge whom",
                "💰")

    TARGETS = {
        "Willingness to pay (₹/site/month)": "wtp",
        "Estimated waste %": "q9_est_waste_pct",
    }
    target_label = st.sidebar.selectbox("Regression target", list(TARGETS))
    target = TARGETS[target_label]

    CANDIDATES = {
        "q4_meals_per_day": "Meals per day",
        "q7_monthly_food_spend_lakh": "Monthly food spend (₹ lakh)",
        "dig_count": "Digital records kept",
        "maturity": "Tracking maturity",
        "hist_depth": "Historical data depth",
        "q9_est_waste_pct": "Estimated waste %",
        **LIKERT_LABEL,
    }
    CANDIDATES = {k: v for k, v in CANDIDATES.items() if k != target}

    default = [c for c in ["q4_meals_per_day", "q7_monthly_food_spend_lakh",
                           "q17_food_cost_major_expense", "q20_trust_data_over_experience",
                           "q21_esg_reporting_required", "q23_need_proof_before_paying",
                           "dig_count", "q9_est_waste_pct"] if c in CANDIDATES]

    chosen = st.sidebar.multiselect("Predictors", list(CANDIDATES),
                                    default=default,
                                    format_func=lambda c: CANDIDATES[c])
    add_tier = st.sidebar.checkbox("Include city tier dummies", True)

    if not chosen:
        st.warning("Select at least one predictor.")
        st.stop()

    reg = df.dropna(subset=[target]).copy()
    X = reg[chosen].copy()
    names = [CANDIDATES[c] for c in chosen]

    if add_tier:
        dummies = pd.get_dummies(reg["q2_city_tier"], prefix="Tier", drop_first=True)
        X = pd.concat([X, dummies.astype(float)], axis=1)
        names += list(dummies.columns)

    X.columns = names
    X = sm.add_constant(X)
    model = sm.OLS(reg[target].astype(float), X.astype(float)).fit()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{model.rsquared:.3f}")
    c2.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")
    c3.metric("Observations", f"{int(model.nobs):,}")
    c4.metric("F-test p-value", f"{model.f_pvalue:.2e}")

    st.markdown("---")

    # ── coefficients ──────────────────────────────────────────────────
    st.subheader("Coefficients")

    coef = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std error": model.bse.values,
        "t": model.tvalues.values,
        "p-value": model.pvalues.values,
        "CI low": model.conf_int()[0].values,
        "CI high": model.conf_int()[1].values,
    }).query("Variable != 'const'")
    coef["Significant"] = coef["p-value"] < 0.05

    l, r = st.columns([1.2, 1])

    with l:
        plot = coef.sort_values("Coefficient")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot["Coefficient"], y=plot["Variable"], orientation="h",
            error_x=dict(type="data",
                         array=plot["CI high"] - plot["Coefficient"],
                         arrayminus=plot["Coefficient"] - plot["CI low"]),
            marker_color=["#548235" if s else "#BFBFBF" for s in plot["Significant"]],
        ))
        fig.update_layout(xaxis_title=f"Effect on {target_label}", yaxis_title="")
        st.plotly_chart(apply_theme(fig, 460, legend_bottom=False), width="stretch")
        st.caption("Green = significant at p<0.05. Bars show 95% confidence intervals. "
                   "Grey bars are not distinguishable from zero.")

    with r:
        st.dataframe(
            coef[["Variable", "Coefficient", "p-value", "Significant"]]
            .sort_values("Coefficient", ascending=False)
            .style.format({"Coefficient": "{:,.1f}", "p-value": "{:.4f}"}),
            width="stretch", hide_index=True, height=460)

    # ── diagnostics ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model diagnostics")
    d1, d2, d3 = st.columns(3)

    fitted = model.fittedvalues
    resid = model.resid

    with d1:
        fig = px.scatter(x=fitted, y=resid, opacity=0.55,
                         labels={"x": "Fitted values", "y": "Residuals"},
                         title="Residuals vs fitted")
        fig.add_hline(y=0, line_dash="dash", line_color="#C55A11")
        st.plotly_chart(apply_theme(fig, 320, legend_bottom=False), width="stretch")

    with d2:
        fig = px.histogram(resid, nbins=35, title="Residual distribution",
                           labels={"value": "Residual"})
        fig.update_traces(marker_color="#1F3864")
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_theme(fig, 320, legend_bottom=False), width="stretch")

    with d3:
        theo = np.sort(np.random.default_rng(0).normal(size=len(resid)))
        fig = px.scatter(x=theo, y=np.sort(resid.values),
                         labels={"x": "Theoretical quantiles", "y": "Sample quantiles"},
                         title="Q-Q plot", opacity=0.55)
        st.plotly_chart(apply_theme(fig, 320, legend_bottom=False), width="stretch")

    st.caption("Check for funnel shapes in residuals-vs-fitted (heteroscedasticity) "
               "and departures from the line in the Q-Q plot. Both are common with "
               "stated-WTP data, which is bounded below at zero and heavily rounded.")

    # ── pricing table ─────────────────────────────────────────────────
    if target == "wtp":
        st.markdown("---")
        st.subheader("Price list derived from the model")

        price = (df.dropna(subset=["wtp"])
                   .groupby(["q1_org_type", "q2_city_tier"])["wtp"]
                   .agg(n="size", p25=lambda s: s.quantile(.25),
                        median="median", p75=lambda s: s.quantile(.75))
                   .reset_index()
                   .query("n >= 8")
                   .sort_values("median", ascending=False))
        price.columns = ["Organisation type", "City tier", "n",
                         "25th pct (₹)", "Median (₹)", "75th pct (₹)"]

        st.dataframe(
            price.style.format({"25th pct (₹)": "{:,.0f}", "Median (₹)": "{:,.0f}",
                                "75th pct (₹)": "{:,.0f}"})
                 .background_gradient(subset=["Median (₹)"], cmap="Blues"),
            width="stretch", hide_index=True)

        st.info(
            """
    **Price at the 25th percentile, not the median.**

    Stated WTP overstates real WTP — respondents face no budget consequence
    when answering a survey. Pricing at the median means roughly half your
    target segment refuses at the first quote, and you lose deals you could
    have won.

    Segments with fewer than 8 responses are suppressed from this table;
    the estimates are too unstable to price against.
    """,
            icon="💡",
        )

    with st.expander("Full regression output"):
        st.code(model.summary().as_text(), language=None)



# ======================================================================
def page_rules():
    # pages/ lives one level below the app root — add the root to sys.path so
    # `common` resolves regardless of the launch directory or host platform.




    df, _ = clean()

    page_header("Association Rules",
                "Which barriers, features and behaviours travel together — apriori on multi-select responses",
                "🔗")

    st.sidebar.markdown("### Rule thresholds")
    min_sup = st.sidebar.slider("Minimum support", 0.02, 0.30, 0.05, 0.01,
                                help="Share of respondents in which both items appear")
    min_conf = st.sidebar.slider("Minimum confidence", 0.20, 0.90, 0.40, 0.05,
                                 help="P(consequent | antecedent)")
    min_lift = st.sidebar.slider("Minimum lift", 1.0, 3.0, 1.2, 0.05,
                                 help="How much more likely than chance. 1.0 = independent")

    rules = association_rules(min_sup, min_conf, min_lift)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rules found", len(rules))
    c2.metric("Max lift", f"{rules['Lift'].max():.2f}" if len(rules) else "—")
    c3.metric("Respondents", f"{len(load_onehot()):,}")

    if rules.empty:
        st.warning("No rules meet these thresholds. Lower the minimums in the sidebar.")
        st.stop()

    st.markdown("---")

    # ── headline rule ─────────────────────────────────────────────────
    top = rules.iloc[0]
    st.success(
        f"**Strongest rule: {top['Antecedent']} → {top['Consequent']}** "
        f"(support {top['Support']:.2f}, confidence {top['Confidence']:.2f}, "
        f"lift {top['Lift']:.2f}). "
        f"Respondents citing *{top['Antecedent']}* are {top['Lift']:.2f}× more likely "
        f"than chance to also cite *{top['Consequent']}*.",
        icon="🔗",
    )

    # ── scatter + table ───────────────────────────────────────────────
    l, r = st.columns([1, 1.1])

    with l:
        st.subheader("Rule landscape")
        fig = px.scatter(rules, x="Support", y="Confidence", size="Lift",
                         color="Lift", color_continuous_scale="Blues",
                         hover_data=["Antecedent", "Consequent"], size_max=26)
        st.plotly_chart(apply_theme(fig, 430, legend_bottom=False), width="stretch")
        st.caption("Top-right and large = frequent, reliable and non-obvious. "
                   "Those are the rules worth acting on.")

    with r:
        st.subheader("All rules")
        st.dataframe(
            rules[["Antecedent", "Consequent", "Support", "Confidence", "Lift"]]
            .style.background_gradient(subset=["Lift"], cmap="Blues")
            .format({"Support": "{:.3f}", "Confidence": "{:.2f}", "Lift": "{:.2f}"}),
            width="stretch", hide_index=True, height=430)

    # ── network ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Rule network")

    top_n = st.slider("Show top N rules by lift", 5, min(40, len(rules)),
                      min(18, len(rules)))
    sub = rules.head(top_n)

    nodes = sorted(set(sub["Antecedent"]) | set(sub["Consequent"]))
    rng = np.random.default_rng(3)
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, angles)}

    edge_traces = []
    for _, row in sub.iterrows():
        x0, y0 = pos[row["Antecedent"]]
        x1, y1 = pos[row["Consequent"]]
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=max(0.8, (row["Lift"] - 1) * 5), color="rgba(31,56,100,0.35)"),
            hoverinfo="text",
            text=f"{row['Antecedent']} → {row['Consequent']}<br>lift {row['Lift']:.2f}",
            showlegend=False))

    deg = pd.concat([sub["Antecedent"], sub["Consequent"]]).value_counts()
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text", text=nodes, textposition="top center",
        textfont=dict(size=11),
        marker=dict(size=[10 + 4 * deg.get(n, 1) for n in nodes],
                    color="#1F3864", line=dict(width=2, color="white")),
        hoverinfo="text", showlegend=False)

    fig = go.Figure(edge_traces + [node_trace])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    st.plotly_chart(apply_theme(fig, 560, legend_bottom=False), width="stretch")
    st.caption("Edge thickness = lift. Node size = how many rules the item appears in.")

    # ── item frequency ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Item frequency by question")

    tabs = st.tabs(list(MULTISELECT.values()))
    for tab, (col, label) in zip(tabs, MULTISELECT.items()):
        with tab:
            counts = explode_multiselect(df, col)
            fig = px.bar(x=counts.values / len(df) * 100, y=counts.index,
                         orientation="h",
                         labels={"x": "% of respondents", "y": ""},
                         text=[f"{v/len(df)*100:.0f}%" for v in counts.values])
            fig.update_traces(marker_color="#2E75B6", textposition="outside",
                              cliponaxis=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(apply_theme(fig, 380, legend_bottom=False),
                            width="stretch")

    # ── interpretation ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("How to read this commercially")

    st.markdown(
        """
    Association rules matter here because **objections are not independent**.
    If two barriers reliably co-occur, they are usually two symptoms of one
    underlying cause — and answering one without the other loses the deal.

    | Pattern | What it means | What to do |
    |---|---|---|
    | **Chef autonomy concerns → Don't trust accuracy** | The stated objection is accuracy, but the real objection is loss of control. Arguing about MAPE will not resolve it | Lead with advisory mode and chef override rights, not accuracy statistics |
    | **ESG reporting → Waste reports in ₹** | Sustainability buyers still need the finance number to get budget approval | Ship both in the same tier or neither sells |
    | **No digital records → Cost** | Sites without data see only the onboarding burden, not the return | Qualify these out early, or price a data-capture setup fee separately |
    | **Me alone (approver) → No digital records** | Small unorganised operators: fast to decide, expensive to onboard | Low priority despite short sales cycles |

    **A caution on interpretation.** Lift measures co-occurrence, not causation,
    and with this many item pairs some rules will clear the threshold by chance
    alone. Treat rules as hypotheses to test in discovery calls, not as findings.
    """
    )

    st.download_button("Download rules as CSV",
                       rules.to_csv(index=False).encode(),
                       "association_rules.csv", "text/csv")



# ======================================================================
def page_roadmap():
    # pages/ lives one level below the app root — add the root to sys.path so
    # `common` resolves regardless of the launch directory or host platform.




    reg = load_register()

    page_header("Dashboard Roadmap",
                "107 specified visualizations across 10 dashboards, phased by data availability",
                "🗺️")

    PHASE_NAME = {1: "Phase 1 — MVP", 2: "Phase 2 — Growth", 3: "Phase 3 — Enterprise"}
    reg["Phase name"] = reg["Phase"].map(PHASE_NAME)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total charts", len(reg))
    for col, ph in zip([c2, c3, c4], [1, 2, 3]):
        col.metric(PHASE_NAME[ph], int((reg["Phase"] == ph).sum()))

    st.markdown("---")

    l, r = st.columns([1.2, 1])

    with l:
        st.subheader("Charts by dashboard and phase")
        pivot = (reg.groupby(["Dashboard", "Phase name"]).size()
                    .reset_index(name="Charts"))
        fig = px.bar(pivot, x="Charts", y="Dashboard", color="Phase name",
                     orientation="h",
                     category_orders={"Phase name": list(PHASE_NAME.values())},
                     color_discrete_map={"Phase 1 — MVP": "#1F3864",
                                         "Phase 2 — Growth": "#2E75B6",
                                         "Phase 3 — Enterprise": "#9DC3E6"})
        fig.update_layout(barmode="stack")
        st.plotly_chart(apply_theme(fig, 460), width="stretch")

    with r:
        st.subheader("Build order")
        order = pd.DataFrame({
            "#": range(1, 11),
            "Dashboard": ["Executive", "Forecast", "Waste", "Sustainability",
                          "Finance", "AI Decision", "Supply Chain", "Customer",
                          "Geographic", "Validation"],
            "Data prerequisite": [
                "All downstream sources",
                "Production log + POS",
                "Two-bin leftover weighing",
                "Emission factors + disposal routes",
                "Cost ledger + billing",
                "Trusted forecast + capacity limits",
                "Inventory & procurement systems",
                "Feedback / ratings system",
                "Site master with coordinates",
                "Acceptance survey (available now)"],
        })
        st.dataframe(order, width="stretch", hide_index=True, height=460)

    st.info(
        "**You currently have data for one of these ten dashboards.** The survey "
        "supports Validation today. The other nine are specifications awaiting "
        "pilot operational data — which is fine, provided you present them as "
        "specifications rather than implying they exist.",
        icon="📌",
    )

    st.markdown("---")

    # ── register browser ──────────────────────────────────────────────
    st.subheader("Chart register")

    f1, f2, f3 = st.columns(3)
    with f1:
        dash = st.multiselect("Dashboard", sorted(reg["Dashboard"].unique()),
                              default=sorted(reg["Dashboard"].unique()))
    with f2:
        phases = st.multiselect("Phase", [1, 2, 3], default=[1],
                                format_func=lambda p: PHASE_NAME[p])
    with f3:
        audiences = sorted({a.strip() for row in reg["Audience"]
                            for a in str(row).split(",")})
        aud = st.selectbox("Audience", ["All"] + audiences)

    view = reg[reg["Dashboard"].isin(dash) & reg["Phase"].isin(phases)]
    if aud != "All":
        view = view[view["Audience"].str.contains(aud, na=False)]

    st.caption(f"**{len(view)}** charts match")

    st.dataframe(
        view[["ID", "Dashboard", "Phase", "Chart", "Chart_Type", "Refresh",
              "ML_Model", "Audience"]],
        width="stretch", hide_index=True, height=380)

    # ── detail view ───────────────────────────────────────────────────
    if len(view):
        st.markdown("---")
        st.subheader("Chart detail")
        pick = st.selectbox("Select a chart",
                            view["ID"] + " — " + view["Chart"])
        row = view[view["ID"] == pick.split(" — ")[0]].iloc[0]

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"### {row['Chart']}")
            st.markdown(f"**Dashboard:** {row['Dashboard']}  |  "
                        f"**{PHASE_NAME[row['Phase']]}**")
            st.markdown(f"**Chart type:** {row['Chart_Type']}")
            st.markdown(f"**X-axis:** {row['X_Axis']}")
            st.markdown(f"**Y-axis:** {row['Y_Axis']}")
            st.markdown(f"**KPI formula:** `{row['KPI_Formula']}`")
            st.markdown(f"**Filters:** {row['Filters']}")
            st.markdown(f"**Drill-down:** {row['Drilldown']}")
        with d2:
            st.markdown(f"**Data source:** {row['Data_Source']}")
            st.markdown(f"**Refresh:** {row['Refresh']}")
            st.markdown(f"**Model:** {row['ML_Model']}")
            st.markdown(f"**RAG thresholds:** {row['RAG_Thresholds']}")
            st.markdown(f"**Audience:** {row['Audience']}")
            st.success(f"**Business insight:** {row['Business_Insight']}")
            st.warning(f"**Decision enabled:** {row['Decision_Enabled']}")
            if isinstance(row["Data_Note"], str) and row["Data_Note"].strip():
                st.error(f"**Data note:** {row['Data_Note']}")

    st.markdown("---")
    st.download_button("Download full register (CSV)",
                       reg.to_csv(index=False).encode(),
                       "chart_register.csv", "text/csv")



# ======================================================================
def page_strategy():
    # pages/ lives one level below the app root — add the root to sys.path so
    # `common` resolves regardless of the launch directory or host platform.





    @st.cache_data(show_spinner=False)
    def load_m2():
        m2 = pd.read_csv(_require("survey_module2_strategy.csv"))
        m1, _ = clean()
        return m1.merge(m2, on="respondent_id", how="inner")


    df = load_m2()

    FEATURES = {
        "f1_inventory_mgmt": ("Inventory management", "u1_inventory_mgmt"),
        "f2_expiry_alerts": ("Expiry alerts", None),
        "f3_internal_stock_transfer": ("Internal multi-site stock transfer", "u3_internal_stock_transfer"),
        "f4_external_sharing_marketplace": ("External inventory-sharing marketplace", "u4_external_sharing_marketplace"),
        "f5_auto_procurement": ("Automated procurement", "u5_auto_procurement"),
        "f6_supplier_price_benchmark": ("Supplier price benchmarking", "u6_supplier_price_benchmark"),
        "f7_menu_recipe_costing": ("Menu & recipe costing", None),
        "f8_esg_reporting_pack": ("ESG reporting pack", "u8_esg_reporting_pack"),
        "f9_staff_scheduling": ("Staff scheduling", None),
    }

    page_header("Feature Roadmap, Pricing & Go-to-Market",
                "What to build, what to charge, who to sell to and how to reach them",
                "🚀")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📦 Feature prioritisation", "💵 Price sensitivity",
         "🎯 Sales strategy", "📣 Expansion & promotion"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — FEATURE PRIORITISATION
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        rows = []
        for col, (label, ucol) in FEATURES.items():
            s = df[col]
            rows.append({
                "Feature": label,
                "Mean interest": s.mean(),
                "Top-2 box %": (s >= 4).mean() * 100,
                "Detractors %": (s <= 2).mean() * 100,
                "Median uplift (₹)": df[ucol].median() if ucol else np.nan,
                "col": col, "ucol": ucol,
            })
        feat = pd.DataFrame(rows).sort_values("Top-2 box %", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Strongest feature", feat.iloc[0]["Feature"],
                  f"{feat.iloc[0]['Top-2 box %']:.0f}% top-2 box")
        c2.metric("Weakest feature", feat.iloc[-1]["Feature"],
                  f"{feat.iloc[-1]['Top-2 box %']:.0f}% top-2 box",
                  delta_color="inverse")
        c3.metric("Respondents", f"{len(df):,}")

        st.markdown("---")
        st.subheader("Interest ranking")

        plot = feat.sort_values("Top-2 box %")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=plot["Feature"], x=plot["Top-2 box %"], orientation="h",
                             name="Interested (4–5)", marker_color="#1F3864",
                             text=[f"{v:.0f}%" for v in plot["Top-2 box %"]],
                             textposition="outside"))
        fig.add_trace(go.Bar(y=plot["Feature"], x=-plot["Detractors %"], orientation="h",
                             name="Not interested (1–2)", marker_color="#C55A11",
                             text=[f"{v:.0f}%" for v in plot["Detractors %"]],
                             textposition="outside"))
        fig.update_layout(barmode="relative", xaxis_title="% of respondents")
        fig.add_vline(x=0, line_color="#888")
        st.plotly_chart(apply_theme(fig, 470), width="stretch")

        st.markdown("---")
        st.subheader("Build / gate / kill matrix")

        mat = feat.dropna(subset=["Median uplift (₹)"]).copy()
        fig = px.scatter(mat, x="Top-2 box %", y="Median uplift (₹)",
                         text="Feature", size=[28] * len(mat),
                         color="Top-2 box %", color_continuous_scale="Blues")
        fig.update_traces(textposition="top center")
        med_x, med_y = 50, mat["Median uplift (₹)"].median()
        fig.add_vline(x=med_x, line_dash="dash", line_color="#888")
        fig.add_hline(y=med_y, line_dash="dash", line_color="#888")
        fig.update_layout(coloraxis_showscale=False,
                          xaxis_title="Breadth of demand (% top-2 box)",
                          yaxis_title="Depth of demand (median ₹ uplift/site/month)")
        st.plotly_chart(apply_theme(fig, 500, legend_bottom=False), width="stretch")
        st.caption("Top-right = build now (wide demand, real money). "
                   "Bottom-left = kill. Top-left = niche, gate it to a segment. "
                   "Bottom-right = wanted but not paid for — bundle it, don't price it.")

        st.markdown("---")
        st.subheader("The two findings that should change your roadmap")

        a, b = st.columns(2)

        with a:
            multi = df["q5_num_outlets"] >= 3
            cmp_df = pd.DataFrame({
                "Operator type": ["Multi-site (3+ outlets)", "Single/dual site"],
                "Mean interest": [df.loc[multi, "f3_internal_stock_transfer"].mean(),
                                  df.loc[~multi, "f3_internal_stock_transfer"].mean()],
                "n": [int(multi.sum()), int((~multi).sum())],
            })
            fig = px.bar(cmp_df, x="Operator type", y="Mean interest",
                         text=cmp_df["Mean interest"].round(2),
                         color="Operator type",
                         color_discrete_sequence=["#1F3864", "#C0C6D0"])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_range=[0, 5])
            st.plotly_chart(apply_theme(fig, 340, legend_bottom=False), width="stretch")
            st.success(
                f"**Internal stock transfer is not one feature, it's two markets.** "
                f"Multi-site operators rate it {cmp_df.iloc[0]['Mean interest']:.2f}/5; "
                f"single-site operators {cmp_df.iloc[1]['Mean interest']:.2f}/5. "
                "Build it, but gate it to multi-site plans and price it as an upsell — "
                "putting it in the base product wastes engineering on 60% of your users.",
                icon="✅")

        with b:
            ext = df["f4_external_sharing_marketplace"]
            dist = ext.value_counts().sort_index()
            fig = px.bar(x=dist.index, y=dist.values / len(df) * 100,
                         labels={"x": "Interest (1–5)", "y": "% of respondents"})
            fig.update_traces(marker_color="#C55A11")
            st.plotly_chart(apply_theme(fig, 340, legend_bottom=False), width="stretch")
            st.error(
                f"**External inventory sharing tests badly — {(ext>=4).mean()*100:.0f}% "
                f"interested, {(ext<=2).mean()*100:.0f}% actively not.** "
                "This is the finding worth acting on. Sharing stock with other "
                "organisations means food-safety liability with no statutory protection, "
                "competitive exposure, and a marketplace that needs density before it "
                "works at all. Park it. If you build it anyway, you'll have spent your "
                "runway on the one feature nobody asked for.",
                icon="⛔")

        st.markdown("---")
        st.subheader("Feature interest by segment")
        seg_col = st.selectbox("Break down by",
                               ["q1_org_type", "q2_city_tier", "q14_production_decided_by"],
                               format_func=lambda c: {"q1_org_type": "Organisation type",
                                                      "q2_city_tier": "City tier",
                                                      "q14_production_decided_by": "Production decided by"}[c])
        heat = df.groupby(seg_col)[list(FEATURES)].mean()
        heat.columns = [FEATURES[c][0] for c in heat.columns]
        fig = px.imshow(heat.T, text_auto=".1f", aspect="auto",
                        color_continuous_scale="RdYlGn", zmin=1.8, zmax=4.6,
                        labels=dict(color="Mean interest"))
        st.plotly_chart(apply_theme(fig, 520, legend_bottom=False), width="stretch")

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — PRICE SENSITIVITY
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Van Westendorp Price Sensitivity Meter")
        st.caption("Four questions per respondent: at what price is this too cheap "
                   "(quality doubt), a bargain, expensive, and too expensive to consider.")

        vw = df.dropna(subset=["vw_too_cheap", "vw_bargain",
                               "vw_expensive", "vw_too_expensive"])
        grid = np.arange(1000, 45000, 250)
        too_cheap = np.array([(vw.vw_too_cheap >= p).mean() for p in grid])
        bargain = np.array([(vw.vw_bargain >= p).mean() for p in grid])
        expensive = np.array([(vw.vw_expensive <= p).mean() for p in grid])
        too_exp = np.array([(vw.vw_too_expensive <= p).mean() for p in grid])

        def cross(a, b):
            return int(grid[np.argmin(np.abs(a - b))])

        opp = cross(too_cheap, too_exp)
        ipp = cross(bargain, expensive)
        pmc = cross(too_cheap, expensive)
        pme = cross(bargain, too_exp)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Optimal price point", inr(opp),
                  help="Where 'too cheap' and 'too expensive' cross — resistance is minimised")
        k2.metric("Indifference price", inr(ipp),
                  help="Where 'bargain' and 'expensive' cross — the perceived normal price")
        k3.metric("Range floor", inr(pmc))
        k4.metric("Range ceiling", inr(pme))

        fig = go.Figure()
        for series, name, color, dash in [
            (too_cheap, "Too cheap", "#C55A11", "dot"),
            (bargain, "Bargain / not expensive", "#548235", "solid"),
            (expensive, "Expensive", "#2E75B6", "solid"),
            (too_exp, "Too expensive", "#1F3864", "dot"),
        ]:
            fig.add_trace(go.Scatter(x=grid, y=series * 100, name=name,
                                     line=dict(color=color, dash=dash, width=2.5)))
        fig.add_vrect(x0=pmc, x1=pme, fillcolor="#548235", opacity=0.10,
                      line_width=0, annotation_text="Acceptable range",
                      annotation_position="top left")
        fig.add_vline(x=opp, line_color="#000", line_dash="dash",
                      annotation_text=f"OPP {inr(opp)}", annotation_position="top right")
        fig.update_layout(xaxis_title="Price (₹ per site per month)",
                          yaxis_title="% of respondents")
        st.plotly_chart(apply_theme(fig, 480), width="stretch")

        st.info(
            f"""
    **Recommended list price: {inr(opp)} per site per month**, with the acceptable
    band running {inr(pmc)}–{inr(pme)}.

    **Discount to the floor for the first 10 logos.** Reference customers are worth
    more than margin in year one, and the range floor is where you stop leaving
    money on the table while still closing quickly.

    **Never price below {inr(pmc)}.** Below the floor the 'too cheap' curve rises —
    buyers start doubting the product works. In enterprise software a price that is
    too low is a credibility problem, not a competitive advantage.
    """,
            icon="💡")

        st.markdown("---")
        st.subheader("Investment appetite beyond the subscription")

        i1, i2, i3 = st.columns(3)
        i1.metric("Median setup fee accepted", inr(df.max_setup_fee_inr.median()),
                  help="One-time onboarding and data-capture setup")
        i2.metric("Median pilot budget", inr(df.pilot_budget_inr.median()),
                  help="Total budget available for a paid pilot")
        i3.metric("Would fund hardware",
                  f"{(df.would_fund_hardware == 'Yes').mean()*100:.0f}%",
                  help="Smart scales and weighing equipment")

        c1, c2 = st.columns(2)
        with c1:
            cl = df["contract_length_pref"].value_counts(normalize=True) * 100
            fig = px.pie(values=cl.values, names=cl.index, hole=0.5,
                         title="Preferred contract length")
            st.plotly_chart(apply_theme(fig, 380), width="stretch")
        with c2:
            fig = px.box(df.dropna(subset=["pilot_budget_inr"]),
                         x="q1_org_type", y="pilot_budget_inr",
                         color="q1_org_type",
                         labels={"pilot_budget_inr": "Pilot budget (₹)", "q1_org_type": ""})
            fig.update_layout(showlegend=False)
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(apply_theme(fig, 380, legend_bottom=False), width="stretch")

        st.warning(
            f"**Only {(df.contract_length_pref=='Annual prepay (discount)').mean()*100:.0f}% "
            "will prepay annually.** Plan cash flow for monthly billing, not for the "
            "annual-contract SaaS model your financial projections probably assume. "
            "This single fact changes your runway calculation materially.",
            icon="⚠️")

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — SALES STRATEGY
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Segment scorecard")

        seg = (df.groupby("q1_org_type")
                 .agg(n=("respondent_id", "size"),
                      adoption=("adopter", "mean"),
                      wtp=("wtp", "median"),
                      pilot_budget=("pilot_budget_inr", "median"),
                      meals=("q4_meals_per_day", "median"),
                      readiness=("dig_count", "mean"),
                      expand=("expansion_intent", lambda s: (s == "Roll out to all sites if pilot succeeds").mean()))
                 .reset_index())
        seg["adoption"] *= 100
        seg["expand"] *= 100

        # composite priority
        z = seg.copy()
        for c in ["adoption", "wtp", "readiness", "expand"]:
            rng_ = z[c].max() - z[c].min()
            z[c] = (z[c] - z[c].min()) / rng_ if rng_ else 0.5
        seg["Priority"] = (0.35*z.adoption + 0.25*z.wtp + 0.25*z.readiness + 0.15*z.expand).round(2)

        show = seg[["q1_org_type", "n", "adoption", "wtp", "pilot_budget",
                    "readiness", "expand", "Priority"]].copy()
        show.columns = ["Segment", "n", "Adoption %", "Median WTP (₹)",
                        "Pilot budget (₹)", "Digital readiness", "Multi-site expand %", "Priority"]
        st.dataframe(
            show.sort_values("Priority", ascending=False)
                .style.format({"Adoption %": "{:.0f}", "Median WTP (₹)": "{:,.0f}",
                               "Pilot budget (₹)": "{:,.0f}", "Digital readiness": "{:.1f}",
                               "Multi-site expand %": "{:.0f}"})
                .background_gradient(subset=["Priority"], cmap="Greens"),
            width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader("Sales play per segment")

        plays = pd.DataFrame([
            ["Corporate cafeteria", "Land & expand",
             "Cost per meal + ESG disclosure",
             "Free 4-week shadow pilot at one site → all sites",
             "Procurement + Finance (multi-stakeholder)", "8–14 weeks"],
            ["Institutional", "Volume play",
             "Cost per meal, budget certainty",
             "Paid pilot funded from existing waste-disposal budget",
             "Head office / administration", "12–20 weeks"],
            ["Contract caterer", "Channel partnership",
             "Win client tenders with sustainability metrics",
             "White-label the analytics into their client reporting",
             "Ops director + their end client", "10–16 weeks"],
            ["Hotel buffet", "Premium / lighthouse",
             "Guest experience protected, brand ESG story",
             "Advisory-mode only, chef keeps full override",
             "F&B Manager → Exec Chef (chef holds veto)", "6–12 weeks"],
            ["Banquet & wedding", "Deprioritise",
             "Weak fit — abundance is the product they sell",
             "Do not pursue at MVP",
             "Owner (fast decision, low readiness)", "—"],
        ], columns=["Segment", "Play", "Lead with", "Motion", "Buying centre", "Cycle"])
        st.dataframe(plays, width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader("What actually blocks the deal")

        blk = (df["purchase_blockers"].dropna().str.split(";").explode()
                 .str.strip().value_counts() / len(df) * 100)
        fig = px.bar(x=blk.values, y=blk.index, orientation="h",
                     labels={"x": "% citing this blocker", "y": ""},
                     text=[f"{v:.0f}%" for v in blk.values])
        fig.update_traces(marker_color="#C55A11", textposition="outside", cliponaxis=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(apply_theme(fig, 400, legend_bottom=False), width="stretch")

        st.info(
            """
    **Two of these blockers are solvable by product decisions, not sales effort.**

    *"Need proof from a comparable site"* — this is why your first three pilots must
    span three different segments, not three sites of the same client. One reference
    per segment unlocks that segment; three references in one segment unlock one.

    *"IT / data security review"* — get SOC 2 readiness and a standard DPA drafted
    before you need them. Discovering this mid-cycle costs 6–8 weeks per deal.

    *"Budget cycle timing"* — reframe the ask. Waste disposal is already a budgeted
    line item; funding the pilot from it avoids a new budget request entirely.
    """,
            icon="🔑")

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — EXPANSION & PROMOTION
    # ══════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("How they want to be reached")

        c1, c2 = st.columns(2)
        with c1:
            ch = (df["preferred_channels"].dropna().str.split(";").explode()
                    .str.strip().value_counts() / len(df) * 100)
            fig = px.bar(x=ch.values, y=ch.index, orientation="h",
                         labels={"x": "% preferring", "y": ""},
                         text=[f"{v:.0f}%" for v in ch.values])
            fig.update_traces(marker_color="#1F3864", textposition="outside", cliponaxis=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(apply_theme(fig, 400, legend_bottom=False), width="stretch")
            st.caption("Preferred sales channel")

        with c2:
            info = (df["information_sources"].dropna().str.split(";").explode()
                      .str.strip().value_counts() / len(df) * 100)
            fig = px.bar(x=info.values, y=info.index, orientation="h",
                         labels={"x": "% using", "y": ""},
                         text=[f"{v:.0f}%" for v in info.values])
            fig.update_traces(marker_color="#2E75B6", textposition="outside", cliponaxis=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(apply_theme(fig, 400, legend_bottom=False), width="stretch")
            st.caption("Where they learn about new solutions")

        top_ch = ch.index[0]
        st.success(
            f"**{top_ch} is the top channel at {ch.iloc[0]:.0f}%, and peer/WhatsApp "
            "networks are the top information source.** This is a referral-led market, "
            "not a performance-marketing one. Digital outreach ranks last. Budget for "
            "customer success and reference cultivation, not ad spend — the money you "
            "would put into LinkedIn ads buys more pipeline as pilot subsidies.",
            icon="📣")

        st.markdown("---")
        st.subheader("Referral and expansion readiness")

        e1, e2, e3 = st.columns(3)
        nps_val = (df.nps_0_10.ge(9).mean() - df.nps_0_10.le(6).mean()) * 100
        e1.metric("Concept NPS", f"{nps_val:.0f}",
                  help="Promoters (9–10) minus detractors (0–6), on the concept alone")
        e2.metric("Would introduce a peer",
                  f"{(df.would_refer == 'Yes, happy to introduce').mean()*100:.0f}%")
        e3.metric("Would roll out to all sites",
                  f"{(df.expansion_intent == 'Roll out to all sites if pilot succeeds').mean()*100:.0f}%")

        c1, c2 = st.columns(2)
        with c1:
            ref = df["would_refer"].value_counts(normalize=True) * 100
            fig = px.pie(values=ref.values, names=ref.index, hole=0.5,
                         title="Willingness to refer",
                         color_discrete_sequence=["#C0C6D0", "#548235", "#C55A11"])
            st.plotly_chart(apply_theme(fig, 380), width="stretch")
        with c2:
            exp_by = (df.groupby("q1_org_type")["expansion_intent"]
                        .apply(lambda s: (s == "Roll out to all sites if pilot succeeds").mean() * 100)
                        .sort_values())
            fig = px.bar(x=exp_by.values, y=exp_by.index, orientation="h",
                         labels={"x": "% who would roll out to all sites", "y": ""},
                         text=[f"{v:.0f}%" for v in exp_by.values])
            fig.update_traces(marker_color="#548235", textposition="outside", cliponaxis=False)
            st.plotly_chart(apply_theme(fig, 380, legend_bottom=False), width="stretch")

        if nps_val < 0:
            st.warning(
                f"""
    **Concept NPS is {nps_val:.0f} — negative, and that is the honest and expected
    result at this stage.**

    People do not become promoters of a concept they have never used. NPS measured
    before a product exists tells you almost nothing about the product and quite a
    lot about the category's scepticism.

    What matters is the {(df.would_refer=='Maybe, after seeing results').mean()*100:.0f}%
    who said they would refer **after seeing results**. That is your actual referral
    engine, and it is gated entirely on shipping a pilot that works. Re-measure NPS
    at week 12 of the first pilots; only that number is worth putting in a deck.
    """,
                icon="⚠️")

        st.markdown("---")
        st.subheader("Expansion sequence")

        st.markdown(
            """
    | Stage | Target | Motion | Success gate |
    |---|---|---|---|
    | **0–6 months** | 3 pilot sites, 3 different segments | Free shadow pilot, founder-led | ≥20% waste reduction, no stockout increase |
    | **6–12 months** | 15–20 sites, one city | Convert pilots, harvest referrals | 3 written references, one per segment |
    | **12–24 months** | 60–100 sites, 2–3 cities | Contract-caterer channel partnership | Payback under 3 months, proven at scale |
    | **24–36 months** | Multi-city, benchmark tier launches | Land-and-expand within multi-site accounts | 30+ clients per cohort for anonymised benchmarking |

    **Geographic sequencing:** win one city completely before opening a second.
    Referral networks and peer WhatsApp groups are local, and density is what makes
    both your support economics and your benchmark data product work. A thin
    national footprint gives you neither.
    """)



# ======================================================================
# NAVIGATION
# ======================================================================
PAGES = {
    "🍽️  Overview": page_overview,
    "🧹  Data Quality": page_quality,
    "🧭  Segmentation": page_segmentation,
    "🎯  Adoption Model": page_adoption,
    "💰  Pricing Model": page_pricing,
    "🔗  Association Rules": page_rules,
    "🗺️  Dashboard Roadmap": page_roadmap,
    "🚀  Feature & GTM Strategy": page_strategy,
}

st.sidebar.markdown("## Navigation")
_choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
st.sidebar.markdown("---")

PAGES[_choice]()
