"""
Smart Beverage Vending for Indian Campuses - analytics dashboard
Run locally from the project folder:   streamlit run code/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA, MOD, OUTS = ROOT / "data", ROOT / "outputs" / "models", ROOT / "outputs"
SEED = 42

st.set_page_config(page_title="Campus Vending Analytics", page_icon="🥤", layout="wide")

# ----------------------------------------------------------------- constants
GREEN, AMBER, RED, BLUE, GREY = "#2a7f62", "#e1a034", "#c0504d", "#3b6ea5", "#8c8c8c"
SEG_ORDER = ["Most Probable", "Medium Probable", "Least Probable"]
SEG_COLORS = dict(zip(SEG_ORDER, [GREEN, AMBER, RED]))
YN_COLORS = {"Yes": GREEN, "No": RED}
ORDERS = {
    "Wait_Band": ["0-5 min", "6-10 min", "11-20 min", "20+ min"],
    "Allowance_Band": ["<5k", "5k-10k", "10k-15k", "15k+"],
    "City_Tier": ["Tier 1", "Tier 2", "Tier 3"],
    "Year_of_Study": ["1st Year", "2nd Year", "3rd Year", "4th Year", "Postgraduate", "PhD"],
    "Preferred_Time_Slot": ["Morning (8-11)", "Afternoon (11-3)", "Evening (3-7)", "Late Night (7-12)"],
    "UPI_Usage_Frequency": ["Rarely", "Weekly", "Daily"],
    "Segment": SEG_ORDER,
}
NUM = ["Age", "Monthly_Allowance_INR", "Hours_On_Campus_Daily", "Beverages_Per_Week",
       "Cafeteria_Wait_Minutes", "Cafeteria_Satisfaction", "Tech_Comfort", "Health_Consciousness",
       "Price_Sensitivity", "Importance_Convenience", "Importance_Hygiene", "UPI_Freq_Num",
       "City_Tier_Num", "Num_Beverages_Liked", "Is_Hostel", "Is_Gym", "Is_Late_Night",
       "Used_Vending_Flag", "Digital_Payer"]
CAT = ["Gender", "Year_of_Study", "Course_Stream", "University_Type", "Region", "Current_Beverage_Source"]
CLU = ["Log_Allowance", "Beverages_Per_Week", "Cafeteria_Satisfaction", "Tech_Comfort",
       "Price_Sensitivity", "Importance_Convenience", "Num_Beverages_Liked"]
OUTCOMES = ["Likely_To_Use_Flag", "Interest_Level", "Expected_Monthly_Spend_INR", "Willing_Price_Per_Drink_INR"]
NICE = {  # readable labels for charts
    "Monthly_Allowance_INR": "Monthly allowance (₹)", "Beverages_Per_Week": "Drinks bought per week",
    "Cafeteria_Wait_Minutes": "Cafeteria wait (min)", "Cafeteria_Satisfaction": "Cafeteria satisfaction",
    "Tech_Comfort": "Tech comfort", "Health_Consciousness": "Health consciousness",
    "Price_Sensitivity": "Price sensitivity", "Importance_Convenience": "Importance of convenience",
    "Importance_Hygiene": "Importance of hygiene", "Num_Beverages_Liked": "No. of drinks liked",
    "Hours_On_Campus_Daily": "Hours on campus", "Is_Hostel": "Lives in hostel", "Is_Gym": "Gym / sports",
    "Is_Late_Night": "Late-night studier", "Used_Vending_Flag": "Used vending before",
    "Digital_Payer": "Pays digitally", "UPI_Freq_Num": "UPI usage frequency", "City_Tier_Num": "City tier (1-3)",
    "Expected_Monthly_Spend_INR": "Expected monthly spend (₹)", "Willing_Price_Per_Drink_INR": "Price per drink (₹)",
    "Likely_To_Use_Flag": "Likely to use", "Interest_Level": "Interest level", "Age": "Age",
    "Log_Allowance": "Allowance (log)",
}
nice = lambda c: NICE.get(c, c.replace("_", " "))
def rupee(v):
    """Indian digit grouping: 557790 -> ₹5,57,790"""
    neg, n = v < 0, str(int(round(abs(v))))
    if len(n) > 3:
        head, tail = n[:-3], n[-3:]
        head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        n = head + "," + tail
    return ("-₹" if neg else "₹") + n


def insight(text):
    st.markdown(f"<div style='border-left:4px solid {GREEN};padding:6px 12px;margin:-6px 0 18px;"
                f"background:rgba(42,127,98,0.07);border-radius:4px;font-size:0.93rem'>💡 {text}</div>",
                unsafe_allow_html=True)


def chart(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------- data
@st.cache_data
def load():
    df = pd.read_csv(DATA / "campus_beverage_survey_scored.csv")
    raw = pd.read_csv(DATA / "campus_beverage_survey_raw.csv")
    log = (OUTS / "cleaning_log.txt").read_text().splitlines()
    try:
        dic = pd.read_excel(ROOT / "Campus_Vending_Survey_Workbook.xlsx", sheet_name="Data_Dictionary")
    except Exception:
        dic = pd.DataFrame()
    df["Log_Allowance"] = np.log(df["Monthly_Allowance_INR"])
    return df, raw, log, dic


df_all, raw, clean_log, dictionary = load()
BEV = [c for c in df_all.columns if c.startswith("Likes_")]
bev_name = lambda c: c.replace("Likes_", "").replace("_", " ")
pre = lambda: ColumnTransformer([("num", StandardScaler(), NUM),
                                 ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT)])


@st.cache_resource
def train_classifiers():
    X, y = df_all[NUM + CAT], df_all["Likely_To_Use_Flag"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=25),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=SEED),
    }
    fitted, rows, probs = {}, [], {}
    for name, m in models.items():
        p = Pipeline([("prep", pre()), ("model", m)]).fit(X_tr, y_tr)
        pr = p.predict_proba(X_te)[:, 1]
        pred = (pr >= 0.5).astype(int)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
                     "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
                     "ROC_AUC": roc_auc_score(y_te, pr), "Train_Accuracy": accuracy_score(y_tr, p.predict(X_tr))})
        fitted[name], probs[name] = p, pr
    res = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    try:
        cv = pd.read_csv(MOD / "A_classification_comparison.csv")[["Model", "CV_F1 (5-fold)"]]
        res = res.merge(cv, on="Model", how="left")
    except Exception:
        pass
    return fitted, probs, y_te.values, res


@st.cache_resource
def clustering():
    sc = StandardScaler().fit(df_all[CLU])
    Z = sc.transform(df_all[CLU])
    ks, inertia, sil = list(range(2, 9)), [], []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z, km.labels_))
    km = KMeans(3, n_init=50, random_state=SEED).fit(Z)
    rate = pd.Series(df_all["Likely_To_Use_Flag"].values).groupby(km.labels_).mean().sort_values(ascending=False)
    label_map = dict(zip(rate.index, SEG_ORDER))
    return sc, km, label_map, pd.DataFrame({"k": ks, "Inertia": inertia, "Silhouette": sil})


@st.cache_data
def fit_regression(ridge_alpha, lasso_alpha):
    X, y = df_all[NUM + CAT], df_all["Expected_Monthly_Spend_INR"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    rows, coefs, preds = [], {}, {}
    names = None
    for name, m in {"Linear": LinearRegression(), "Ridge": Ridge(alpha=ridge_alpha),
                    "Lasso": Lasso(alpha=lasso_alpha, max_iter=20000)}.items():
        p = Pipeline([("prep", pre()), ("model", m)]).fit(X_tr, y_tr)
        pt = p.predict(X_te)
        names = [n.split("__", 1)[1] for n in p.named_steps["prep"].get_feature_names_out()]
        coefs[name] = p.named_steps["model"].coef_
        preds[name] = pt
        rows.append({"Model": name, "R² (test)": r2_score(y_te, pt), "R² (train)": r2_score(y_tr, p.predict(X_tr)),
                     "MAE (₹)": mean_absolute_error(y_te, pt), "RMSE (₹)": mean_squared_error(y_te, pt) ** 0.5,
                     "Features kept": int((np.abs(p.named_steps["model"].coef_) > 1e-6).sum())})
    return pd.DataFrame(rows), pd.DataFrame(coefs, index=names), pd.DataFrame(preds).assign(Actual=y_te.values)


@st.cache_resource
def spend_model():
    return Pipeline([("prep", pre()), ("model", Lasso(alpha=4.58, max_iter=20000))]).fit(
        df_all[NUM + CAT], df_all["Expected_Monthly_Spend_INR"])


@st.cache_data
def mine_rules(min_support):
    basket = df_all[BEV].astype(bool)
    basket.columns = [bev_name(c) for c in BEV]
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    r = association_rules(freq, metric="lift", min_threshold=1.0)
    r["IF student wants"] = r["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    r["THEN also wants"] = r["consequents"].apply(lambda s: " + ".join(sorted(s)))
    r["Drinks in rule"] = r["antecedents"].apply(len) + r["consequents"].apply(len)
    return r[["IF student wants", "THEN also wants", "support", "confidence", "lift", "Drinks in rule"]]


fitted, probs, y_test, clf_res = train_classifiers()
scaler, kmeans, seg_map, k_scan = clustering()
df_all["Segment"] = pd.Series(kmeans.labels_).map(seg_map).values
BEST_CLF = clf_res.iloc[0]["Model"]

# ----------------------------------------------------------------- sidebar
st.sidebar.title("🥤 Campus Vending")
st.sidebar.caption("Consumer-survey analytics for a smart beverage vending start-up in Indian universities.")
with st.sidebar.expander("🔎 Filter respondents", expanded=False):
    st.caption("Filters apply to the **Descriptive** and **Diagnostic** tabs. Models always use all 2,000 students.")
    filt = {}
    for col in ["City_Tier", "University_Type", "Residence_Type", "Gender", "Course_Stream", "Segment"]:
        opts = ORDERS.get(col, sorted(df_all[col].unique()))
        filt[col] = st.multiselect(nice(col), opts, default=opts)
df = df_all.copy()
for col, vals in filt.items():
    df = df[df[col].isin(vals)]
st.sidebar.metric("Students in view", f"{len(df):,} / {len(df_all):,}")
st.sidebar.markdown("---")
st.sidebar.caption("Data is **synthetic**, generated to mimic realistic student behaviour, for academic use.")


def rate_by(data, col, val="Likely_To_Use_Flag", how="mean"):
    g = data.groupby(col, observed=True).agg(v=(val, how), n=(val, "size")).reset_index()
    if col in ORDERS:
        g[col] = pd.Categorical(g[col], ORDERS[col], ordered=True)
        g = g.sort_values(col)
    return g


# ----------------------------------------------------------------- header
st.title("🥤 Smart Beverage Vending for Indian Campuses")
st.caption("Validating a start-up idea with a 2,000-student consumer survey · descriptive, diagnostic and "
           "predictive analytics (classification, clustering, association rules, regression)")

tabs = st.tabs(["🏠 Overview", "🧹 Data & Cleaning", "📊 Descriptive", "🔍 Diagnostic", "🎯 Classification",
                "👥 Clustering", "🛒 Association", "💰 Regression", "🧮 Predict & Business Case"])

# ================================================================= OVERVIEW
with tabs[0]:
    st.markdown("""
**The idea.** Install high-tech, cashless beverage vending machines across Indian university campuses. They would
serve fresh juice, cold coffee, flavoured milk, iced tea, energy drinks and more through a touchscreen with UPI payment
and automated dispensing, placed at hostels, libraries and academic blocks. The aim is to cut students' dependence on
crowded cafeterias and nearby shops.

**The question this dashboard answers:** *is there enough demand, who will buy, what will they buy together, and how much
will they spend?*
""")
    a = df_all
    c = st.columns(5)
    c[0].metric("Would use the machine", f"{a.Likely_To_Use_Flag.mean():.0%}")
    c[1].metric("Interest 4–5 out of 5", f"{(a.Interest_Level >= 4).mean():.0%}")
    c[2].metric("Prefer cashless", f"{a.Digital_Payer.mean():.0%}")
    c[3].metric("Median spend of likely users", rupee(a.loc[a.Likely_To_Use_Flag == 1, "Expected_Monthly_Spend_INR"].median()) + "/mo")
    c[4].metric("Median acceptable price", rupee(a.Willing_Price_Per_Drink_INR.median()) + "/drink")

    st.subheader("Feasibility verdict")
    st.success("**Feasible, with conditions.** Demand and payment readiness are strong. Margins are thin, so success "
               "depends on choosing the right campuses, locations and product mix, which is exactly what the models below identify.")
    l, r = st.columns(2)
    with l:
        st.markdown("""**✅ Why it can work**
- UPI is part of everyday campus life, so cashless-only machines are realistic
- Long cafeteria queues and no late-night options create a real pain point
- Thousands of students in a small area means high footfall per machine
- One agreement with a university opens up many machine locations""")
    with r:
        st.markdown("""**⚠️ What must be managed**
- Fresh juice and milk spoil quickly, so wastage and FSSAI food-safety compliance matter
- High machine cost means a long payback (see *Predict & Business Case*)
- Universities may have exclusive canteen contracts
- Students are price-sensitive, so prices must vary by city tier""")

    st.subheader("Analytics framework")
    rules_all = mine_rules(0.10)
    top3 = rules_all[(rules_all["Drinks in rule"] == 3) & (rules_all.confidence >= 0.5)].sort_values("lift", ascending=False).iloc[0]
    reg_tbl, _, _ = fit_regression(116.9, 4.58)
    seg_rates = df_all.groupby("Segment")["Likely_To_Use_Flag"].mean().reindex(SEG_ORDER)
    cards = [
        ("🎯 Classification", "Will a student use the machine?",
         f"Best model: **{BEST_CLF}**, ROC-AUC **{clf_res.iloc[0].ROC_AUC:.2f}**, accuracy {clf_res.iloc[0].Accuracy:.0%}"),
        ("👥 Clustering", "Who is most, medium and least likely to buy?",
         f"Adoption **{seg_rates.iloc[0]:.0%} / {seg_rates.iloc[1]:.0%} / {seg_rates.iloc[2]:.0%}** across the 3 K-Means segments"),
        ("🛒 Association", "If they want 2 drinks, do they want a 3rd?",
         f"{{{top3['IF student wants']}}} → **{top3['THEN also wants']}**: {top3.confidence:.0%} confidence, lift {top3.lift:.2f}"),
        ("💰 Regression", "How much will they spend?",
         f"Lasso explains **{reg_tbl.loc[2, 'R² (test)']:.0%}** of spend using only {reg_tbl.loc[2, 'Features kept']} of 45 features"),
    ]
    cols = st.columns(4)
    for col, (t, q, ans) in zip(cols, cards):
        with col.container(border=True):
            st.markdown(f"**{t}**")
            st.caption(q)
            st.markdown(ans)

    st.subheader("Recommended go-to-market")
    st.markdown("""
1. **Where:** start with Tier 1 and Tier 2 private or deemed universities that have large hostels.
2. **Placement:** hostel blocks (late night), then the library and academic blocks (daytime).
3. **Payment:** cashless only (UPI and campus ID).
4. **Price:** about ₹65 in Tier 1 and about ₹55 in Tier 2 campuses.
5. **Range:** a *Study Fuel* machine (cold coffee, iced tea, energy drink, flavoured milk) and a *Fit* machine
   (fresh juice, smoothie, coconut water, protein shake).
6. **Customers:** win the *Most Probable* segment with a monthly drink pass, and convert *Medium Probable* students with
   "skip the queue" messaging and combo offers.
""")

# ================================================================= DATA & CLEANING
with tabs[1]:
    st.subheader("From raw survey to analysis-ready data")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{len(raw):,}")
    c[1].metric("Clean rows", f"{len(df_all):,}", f"{len(df_all) - len(raw)} duplicates")
    c[2].metric("Missing cells (raw)", f"{int(raw.isna().sum().sum()):,}", "→ 0 after cleaning", delta_color="off")
    c[3].metric("Columns", f"{raw.shape[1]} → {df_all.shape[1] - 1}", "engineered features", delta_color="off")

    l, r = st.columns(2)
    with l:
        miss = raw.isna().sum()
        miss = (miss[miss > 0] / len(raw) * 100).sort_values()
        fig = px.bar(x=miss.values, y=[nice(i) for i in miss.index], orientation="h",
                     labels={"x": "% missing", "y": ""}, title="Missing answers in the raw survey",
                     color_discrete_sequence=[AMBER])
        chart(fig, 340)
        insight("Respondents skipped some questions, most often allowance (~4%). Numeric gaps were filled with the "
                "median <b>within the same residence type</b>, which is more realistic than one overall median. "
                "Category gaps were filled with the most common answer (mode).")
    with r:
        g = raw["Gender"].fillna("(blank)").value_counts().reset_index()
        g.columns = ["Raw answer", "Count"]
        fig = px.bar(g, x="Raw answer", y="Count", title="Inconsistent labels, e.g. Gender in the raw data",
                     color_discrete_sequence=[RED])
        chart(fig, 340)
        insight(f"Free-text answers produced {raw['Gender'].nunique()} spellings for gender (\"M\", \"male\", \" Male\"…). "
                "These were standardised to 3 clean categories. City tier, payment method and the target were cleaned the same way.")

    st.markdown("#### Cleaning steps (auto-logged by `02_clean_and_transform.py`)")
    st.code("\n".join(clean_log), language=None)

    with st.expander("📖 Data dictionary"):
        if len(dictionary):
            st.dataframe(dictionary, width="stretch", hide_index=True)
    with st.expander("👀 Preview clean data"):
        st.dataframe(df_all.drop(columns=["Log_Allowance"]).head(200), width="stretch", hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download raw survey (CSV)", raw.to_csv(index=False), "campus_beverage_survey_raw.csv")
    d2.download_button("⬇️ Download clean + scored data (CSV)", df_all.drop(columns=["Log_Allowance"]).to_csv(index=False),
                       "campus_beverage_survey_clean_scored.csv")

# ================================================================= DESCRIPTIVE
with tabs[2]:
    if df.empty:
        st.warning("No students match the current filters.")
    else:
        st.caption(f"Showing **{len(df):,}** students (change the sidebar filters to compare groups).")
        k = st.columns(4)
        k[0].metric("Likely to use", f"{df.Likely_To_Use_Flag.mean():.1%}")
        k[1].metric("Median spend / month", rupee(df.Expected_Monthly_Spend_INR.median()))
        k[2].metric("Median price / drink", rupee(df.Willing_Price_Per_Drink_INR.median()))
        k[3].metric("Drinks bought / week", f"{df.Beverages_Per_Week.mean():.1f}")

        l, r = st.columns(2)
        with l:
            s = df["Likely_To_Use"].value_counts()
            fig = px.pie(names=s.index, values=s.values, hole=0.55, title="Would students use the machine?",
                         color=s.index, color_discrete_map=YN_COLORS)
            chart(fig)
            insight(f"<b>{df.Likely_To_Use_Flag.mean():.0%}</b> say yes. That is a large addressable base, and the "
                    "near 50/50 split also makes a balanced target for the classification models.")
        with r:
            s = df["Interest_Level"].value_counts().sort_index()
            fig = px.bar(x=s.index.astype(str), y=s.values / len(df) * 100, title="Overall interest (1 = none, 5 = very high)",
                         labels={"x": "Interest level", "y": "% of students"},
                         color=s.index.astype(str), color_discrete_sequence=[RED, "#e07b67", GREY, "#6fb59a", GREEN])
            fig.update_layout(showlegend=False)
            chart(fig)
            insight(f"<b>{(df.Interest_Level >= 4).mean():.0%}</b> rate their interest 4 or 5 out of 5 (top-2-box). "
                    f"Only {(df.Interest_Level <= 2).mean():.0%} are clearly negative.")

        l, r = st.columns(2)
        with l:
            b = (df[BEV].mean() * 100).sort_values()
            fig = px.bar(x=b.values, y=[bev_name(i) for i in b.index], orientation="h", title="Which drinks do students want?",
                         labels={"x": "% interested", "y": ""}, color_discrete_sequence=[BLUE])
            chart(fig, 420)
            insight(f"<b>{bev_name(b.index[-1])}</b> ({b.iloc[-1]:.0f}%) and <b>{bev_name(b.index[-2])}</b> "
                    f"({b.iloc[-2]:.0f}%) are the anchor products. {bev_name(b.index[0])} ({b.iloc[0]:.0f}%) is a niche item, "
                    "best stocked only near gyms and sports complexes.")
        with r:
            fig = px.box(df, x="Likely_To_Use", y="Expected_Monthly_Spend_INR", color="Likely_To_Use",
                         color_discrete_map=YN_COLORS, points=False, category_orders={"Likely_To_Use": ["Yes", "No"]},
                         title="Expected monthly spend: likely users vs others",
                         labels={"Expected_Monthly_Spend_INR": "₹ per month", "Likely_To_Use": "Likely to use"})
            fig.update_layout(showlegend=False)
            chart(fig, 420)
            m = df.groupby("Likely_To_Use")["Expected_Monthly_Spend_INR"].median()
            insight(f"Likely users expect to spend a median <b>{rupee(m.get('Yes', 0))}</b> a month, against "
                    f"{rupee(m.get('No', 0))} for others. Spend is right-skewed: a small group of heavy users brings in a "
                    "large share of revenue.")

        l, r = st.columns(2)
        with l:
            fig = px.box(df, x="City_Tier", y="Willing_Price_Per_Drink_INR", category_orders={"City_Tier": ORDERS["City_Tier"]},
                         title="Acceptable price per drink by city tier", color_discrete_sequence=[BLUE],
                         labels={"Willing_Price_Per_Drink_INR": "₹ per drink", "City_Tier": ""})
            chart(fig)
            t = df.groupby("City_Tier")["Willing_Price_Per_Drink_INR"].median()
            insight("Median acceptable price: " + ", ".join(f"{k} <b>{rupee(v)}</b>" for k, v in t.items())
                    + ". One national price list would either lose Tier 3 buyers or leave money on the table in Tier 1.")
        with r:
            s = df["Preferred_Payment"].value_counts()
            fig = px.pie(names=s.index, values=s.values, hole=0.55, title="Preferred payment method",
                         color_discrete_sequence=[GREEN, BLUE, AMBER, RED])
            chart(fig)
            insight(f"<b>{df.Digital_Payer.mean():.0%}</b> prefer a cashless method (UPI alone "
                    f"{(df.Preferred_Payment == 'UPI').mean():.0%}). A cashless-only machine is cheaper to run "
                    "(no coin mechanism or cash collection) and loses few customers.")

        l, r = st.columns(2)
        with l:
            ct = pd.crosstab(df["Preferred_Location"], df["Preferred_Time_Slot"], normalize="all") * 100
            ct = ct.reindex(columns=[c for c in ORDERS["Preferred_Time_Slot"] if c in ct.columns])
            fig = px.imshow(ct.round(1), text_auto=True, color_continuous_scale="Greens", aspect="auto",
                            title="Where and when students want a machine (% of students)", labels={"x": "", "y": ""})
            fig.update_coloraxes(showscale=False)
            chart(fig, 400)
            loc = df["Preferred_Location"].value_counts(normalize=True)
            insight(f"<b>{loc.index[0]}</b> ({loc.iloc[0]:.0%}) and <b>{loc.index[1]}</b> ({loc.iloc[1]:.0%}) are the top "
                    "spots. The late-night slot is concentrated at hostel blocks, which is when cafeterias are closed.")
        with r:
            s = df["Most_Valued_Feature"].value_counts(normalize=True).sort_values() * 100
            fig = px.bar(x=s.values, y=s.index, orientation="h", title="Single most valued machine feature",
                         labels={"x": "% of students", "y": ""}, color_discrete_sequence=[AMBER])
            chart(fig, 400)
            insight(f"<b>{s.index[-1]}</b> and <b>{s.index[-2]}</b> lead, so the pitch should be "
                    "\"cheaper and faster than the canteen\", not just \"high-tech\".")

# ================================================================= DIAGNOSTIC
with tabs[3]:
    if df.empty:
        st.warning("No students match the current filters.")
    else:
        st.markdown("Diagnostic analytics asks **why**: which student characteristics go together with adoption and spend?")
        opts = {"Residence type": "Residence_Type", "Cafeteria wait": "Wait_Band", "Monthly allowance": "Allowance_Band",
                "City tier": "City_Tier", "University type": "University_Type", "Price sensitivity (1-5)": "Price_Sensitivity",
                "Tech comfort (1-5)": "Tech_Comfort", "Cafeteria satisfaction (1-5)": "Cafeteria_Satisfaction",
                "Importance of convenience (1-5)": "Importance_Convenience", "Late-night study": "Late_Night_Study",
                "Course stream": "Course_Stream", "Year of study": "Year_of_Study", "Gender": "Gender"}
        pick = st.selectbox("Break adoption and spend down by…", list(opts), index=1)
        col = opts[pick]
        g = rate_by(df, col)
        s = rate_by(df, col, "Expected_Monthly_Spend_INR", "median")
        l, r = st.columns(2)
        with l:
            fig = px.bar(g, x=col, y=g.v * 100, text=(g.v * 100).round(0).astype(int).astype(str) + "%",
                         title=f"Adoption rate by {pick.lower()}", labels={"y": "% likely to use", col: ""},
                         color_discrete_sequence=[GREEN], hover_data={"n": True})
            fig.update_xaxes(type="category")
            chart(fig)
        with r:
            fig = px.bar(s, x=col, y="v", text=s.v.map(rupee), title=f"Median expected spend by {pick.lower()}",
                         labels={"v": "₹ per month", col: ""}, color_discrete_sequence=[BLUE])
            fig.update_xaxes(type="category")
            chart(fig)
        hi, lo = g.loc[g.v.idxmax()], g.loc[g.v.idxmin()]
        insight(f"Adoption is highest for <b>{hi[col]}</b> ({hi.v:.0%}, n={hi.n}) and lowest for <b>{lo[col]}</b> "
                f"({lo.v:.0%}, n={lo.n}), a gap of <b>{(hi.v - lo.v) * 100:.0f} percentage points</b>. "
                + ("A gap under ~8 points is small, so this factor is not a strong driver." if hi.v - lo.v < 0.08
                   else "This is a meaningful driver to use when choosing campuses and targeting students."))

        st.markdown("#### What drives adoption?")
        drivers = [c for c in NUM if c not in ("Age",)]
        cr = df[drivers + ["Likely_To_Use_Flag"]].corr()["Likely_To_Use_Flag"].drop("Likely_To_Use_Flag").sort_values()
        fig = px.bar(x=cr.values, y=[nice(i) for i in cr.index], orientation="h",
                     color=np.where(cr.values > 0, "Pushes adoption up", "Pulls adoption down"),
                     color_discrete_map={"Pushes adoption up": GREEN, "Pulls adoption down": RED},
                     title="Correlation of each factor with 'likely to use'", labels={"x": "Correlation (r)", "y": ""})
        chart(fig, 520)
        insight(f"The strongest positive drivers are <b>{nice(cr.index[-1])}</b> (r={cr.iloc[-1]:.2f}) and "
                f"<b>{nice(cr.index[-2])}</b> (r={cr.iloc[-2]:.2f}). The strongest negatives are <b>{nice(cr.index[0])}</b> "
                f"(r={cr.iloc[0]:.2f}) and <b>{nice(cr.index[1])}</b> (r={cr.iloc[1]:.2f}): students who are happy with the "
                "cafeteria, or very price-sensitive, are hard to win. Outcome answers (interest, spend, price) are left out "
                "here because they are results, not causes.")

        l, r = st.columns(2)
        with l:
            pv = df.pivot_table(index="Price_Sensitivity", columns="Tech_Comfort", values="Likely_To_Use_Flag", aggfunc="mean") * 100
            fig = px.imshow(pv.round(0), text_auto=True, color_continuous_scale="RdYlGn", zmin=0, zmax=100, aspect="auto",
                            title="Adoption % by price sensitivity × tech comfort",
                            labels={"x": "Tech comfort (1-5)", "y": "Price sensitivity (1-5)", "color": "%"})
            chart(fig, 420)
            insight("The two attitudes combine: tech-savvy students who don't mind paying (top right) are the easiest "
                    "wins, and price-sensitive low-tech students (bottom left) almost never adopt. Target campuses where "
                    "the first group dominates.")
        with r:
            num = ["Monthly_Allowance_INR", "Beverages_Per_Week", "Cafeteria_Wait_Minutes", "Cafeteria_Satisfaction",
                   "Tech_Comfort", "Price_Sensitivity", "Importance_Convenience", "Num_Beverages_Liked",
                   "Willing_Price_Per_Drink_INR", "Expected_Monthly_Spend_INR", "Likely_To_Use_Flag"]
            cm = df[num].corr().round(2)
            cm.index = cm.columns = [nice(c) for c in num]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                            title="Correlation matrix (key variables)")
            fig.update_traces(textfont_size=9)
            chart(fig, 420)
            insight("Allowance and drinks per week move together with spend, and no two predictors are so highly "
                    "correlated (|r| > 0.8) that they would cause multicollinearity in the regression.")

# ================================================================= CLASSIFICATION
with tabs[4]:
    st.markdown(f"""**Question:** will a student use the machine (Yes/No)? This is a *supervised classification* problem.
75% of students (1,500) train the models and 25% (500) test them on data they have never seen. **Leakage rule:** only profile,
habit and attitude answers are used as inputs. Interest level, spend and price are left out because they are the answer in disguise.""")
    show = clf_res.copy()
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in show.columns if c != "Model"})
    best = clf_res.iloc[0]
    rf_row = clf_res.set_index("Model").loc["Random Forest"]
    insight(f"<b>{BEST_CLF}</b> separates buyers from non-buyers best (ROC-AUC <b>{best.ROC_AUC:.3f}</b>, "
            "\"good\" on the 0.8–0.9 scale). The complex models do not win, because the drivers are mostly additive. "
            f"Random Forest shows <b>overfitting</b>: {rf_row.Train_Accuracy:.0%} accuracy on training data against "
            f"{rf_row.Accuracy:.0%} on test data.")

    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        for name, pr in probs.items():
            fpr, tpr, _ = roc_curve(y_test, pr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({roc_auc_score(y_test, pr):.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="Coin flip"))
        fig.update_layout(title="ROC curves (AUC in brackets)", xaxis_title="False positive rate",
                          yaxis_title="True positive rate (recall)")
        chart(fig, 430)
        insight("A curve hugging the top-left corner is better. Every model beats the coin-flip diagonal, so the "
                "survey answers really do predict adoption.")
    with r:
        m = clf_res.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
        fig = px.bar(m, x="Model", y="value", color="variable", barmode="group", title="Metric comparison",
                     labels={"value": "", "Model": ""})
        fig.update_yaxes(range=[0.5, 0.9])
        chart(fig, 430)
        insight("Recall is lower than precision for every model: at the default 0.5 cut-off the models are cautious "
                "and miss some real buyers. The threshold lab below shows how to trade one against the other.")

    st.markdown("#### 🎚️ Threshold lab")
    a, b = st.columns([1, 2])
    with a:
        mname = st.selectbox("Model", list(probs), index=list(probs).index(BEST_CLF))
        thr = st.slider("Flag as 'will use' when probability ≥", 0.05, 0.95, 0.50, 0.05)
        pred = (probs[mname] >= thr).astype(int)
        mm = {"Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0),
              "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred, zero_division=0)}
        for k_, v_ in mm.items():
            st.metric(k_, f"{v_:.1%}")
    with b:
        cmx = confusion_matrix(y_test, pred)
        fig = px.imshow(cmx, text_auto=True, color_continuous_scale="Blues", x=["Predicted No", "Predicted Yes"],
                        y=["Actual No", "Actual Yes"], title=f"Confusion matrix: {mname} @ {thr:.2f}")
        fig.update_coloraxes(showscale=False)
        chart(fig, 380)
        insight(f"At this threshold the model catches <b>{cmx[1, 1]}</b> of {cmx[1].sum()} real buyers (recall {mm['Recall']:.0%}), "
                f"and <b>{cmx[0, 1]}</b> non-buyers are wrongly flagged. For a <b>launch campaign</b> (cheap flyers, free "
                "samples), lower the threshold to maximise recall. For <b>costly actions</b> (placing a machine for one "
                "hostel), raise it to maximise precision.")

    rf = fitted["Random Forest"]
    names = [n.split("__", 1)[1] for n in rf.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values().tail(12)
    fig = px.bar(x=imp.values, y=[nice(i) for i in imp.index], orientation="h", color_discrete_sequence=[BLUE],
                 title="Random Forest: top 12 drivers of adoption", labels={"x": "Importance", "y": ""})
    chart(fig, 420)
    insight(f"Allowance, the number of drinks liked, cafeteria wait and price sensitivity matter most. This matches the "
            "diagnostic tab, so the models and the EDA tell the same story.")

# ================================================================= CLUSTERING
with tabs[5]:
    st.markdown("""**Question:** which natural customer groups exist? K-Means is *unsupervised*: it never sees the "likely to use" answer.
We cluster on 7 habit and attitude variables (allowance, drinks per week, cafeteria satisfaction, tech comfort, price sensitivity,
convenience importance, number of drinks liked), all standardised. Then we rank the clusters by adoption to get
**Most / Medium / Least Probable**.""")
    l, r = st.columns(2)
    with l:
        fig = px.line(k_scan, x="k", y="Inertia", markers=True, title="Elbow method",
                      labels={"Inertia": "Within-cluster sum of squares"}, color_discrete_sequence=[BLUE])
        chart(fig, 320)
    with r:
        fig = px.line(k_scan, x="k", y="Silhouette", markers=True, title="Silhouette score (higher = better separated)",
                      color_discrete_sequence=[RED])
        chart(fig, 320)
    insight(f"The elbow bends at <b>k = 3</b>. k = 2 has a slightly higher silhouette ({k_scan.Silhouette.iloc[0]:.2f} vs "
            f"{k_scan.Silhouette.iloc[1]:.2f}) but only splits 'yes' from 'no'. k = 3 gives actionable tiers. A silhouette near "
            "0.13 is normal for survey data: segments are overlapping gradients, not separate islands. "
            "<b>Lesson learned:</b> adding Yes/No flags (hostel, cash payer) made K-Means split on a single flag, so they "
            "are used only to <i>describe</i> the segments.")

    prof = df_all.groupby("Segment").agg(
        Students=("Respondent_ID", "count"), Adoption=("Likely_To_Use_Flag", "mean"),
        Spend=("Expected_Monthly_Spend_INR", "median"), Allowance=("Monthly_Allowance_INR", "median"),
        Price=("Willing_Price_Per_Drink_INR", "median"), Wait=("Cafeteria_Wait_Minutes", "mean"),
        CafeSat=("Cafeteria_Satisfaction", "mean"), PriceSens=("Price_Sensitivity", "mean"),
        Conv=("Importance_Convenience", "mean"), Liked=("Num_Beverages_Liked", "mean"),
        Hostel=("Is_Hostel", "mean")).reindex(SEG_ORDER)
    personas = {
        "Most Probable": ("💎 Affluent convenience seekers",
                          "Higher allowance, low price sensitivity, like many drinks.",
                          "Launch target. Offer premium drinks, a monthly pass and loyalty points."),
        "Medium Probable": ("⏳ Frustrated queue-haters",
                            "Longest cafeteria waits and lowest satisfaction, but price-sensitive.",
                            "Use 'skip the queue' messaging, combo deals and value pricing."),
        "Least Probable": ("🏠 Satisfied budget traditionalists",
                           "Happy with the cafeteria, low need for convenience, like few drinks.",
                           "Don't spend on acquiring them; reach them with low-price staples only."),
    }
    cols = st.columns(3)
    for col, seg in zip(cols, SEG_ORDER):
        p = prof.loc[seg]
        with col.container(border=True):
            st.markdown(f"<span style='color:{SEG_COLORS[seg]};font-weight:700'>{seg.upper()}</span>", unsafe_allow_html=True)
            st.markdown(f"### {personas[seg][0]}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Share", f"{p.Students / len(df_all):.0%}")
            m2.metric("Adoption", f"{p.Adoption:.0%}")
            m3.metric("Spend/mo", rupee(p.Spend))
            st.caption(f"Allowance {rupee(p.Allowance)} · cafeteria wait {p.Wait:.0f} min · satisfaction {p.CafeSat:.1f}/5 · "
                       f"price sensitivity {p.PriceSens:.1f}/5 · drinks liked {p.Liked:.1f}")
            st.markdown(f"**Profile:** {personas[seg][1]}")
            st.markdown(f"**Strategy:** {personas[seg][2]}")

    l, r = st.columns(2)
    with l:
        zcols = ["Allowance", "Wait", "CafeSat", "PriceSens", "Conv", "Liked", "Hostel", "Spend", "Adoption"]
        labels = ["Allowance", "Cafe wait", "Cafe satisfaction", "Price sensitivity", "Convenience", "Drinks liked",
                  "Hostel share", "Spend", "Adoption"]
        z = (prof[zcols] - prof[zcols].mean()) / prof[zcols].std()
        txt = prof[zcols].apply(lambda c: c.map(lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"))
        fig = go.Figure(go.Heatmap(z=z.values, x=labels, y=SEG_ORDER, text=txt.values, texttemplate="%{text}",
                                   colorscale="RdYlGn", showscale=False, textfont=dict(size=13)))
        fig.update_layout(title="Segment profiles (colour = relatively high or low)")
        fig.update_yaxes(autorange="reversed")
        chart(fig, 330)
    with r:
        bs = df_all.groupby("Segment")[BEV].mean().reindex(SEG_ORDER) * 100
        bs.columns = [bev_name(c) for c in BEV]
        fig = px.imshow(bs.round(0), text_auto=True, color_continuous_scale="Purples", aspect="auto",
                        title="% of each segment interested in each drink", labels={"x": "", "y": ""})
        fig.update_coloraxes(showscale=False)
        chart(fig, 330)
    insight("The segments differ mainly in <b>allowance, price sensitivity and cafeteria pain</b>, not in demographics. "
            "The Most Probable segment wants more of every drink, so it is the best audience for a varied premium machine.")

    xopts = ["Monthly_Allowance_INR", "Price_Sensitivity", "Cafeteria_Wait_Minutes", "Cafeteria_Satisfaction", "Beverages_Per_Week"]
    yopts = ["Expected_Monthly_Spend_INR", "Num_Beverages_Liked", "Willing_Price_Per_Drink_INR", "Beverages_Per_Week"]
    a, b = st.columns(2)
    xs = a.selectbox("X axis", xopts, format_func=nice)
    ys = b.selectbox("Y axis", yopts, format_func=nice)
    jit = df_all.copy()
    for c_ in (xs, ys):
        if jit[c_].nunique() < 30:
            jit[c_] = jit[c_] + np.random.default_rng(0).uniform(-0.25, 0.25, len(jit))
    fig = px.scatter(jit, x=xs, y=ys, color="Segment", category_orders={"Segment": SEG_ORDER}, opacity=0.55,
                     color_discrete_map=SEG_COLORS, labels={xs: nice(xs), ys: nice(ys)}, title="Explore the segments")
    chart(fig, 450)

# ================================================================= ASSOCIATION
with tabs[6]:
    st.markdown("""**Question:** if a student wants drink A (and B), which drink C do they also want? Each student's ticked
drinks are treated as a *shopping basket*, and the **Apriori** algorithm finds combinations that occur more often than chance.
- **Support:** share of all students who want the whole combination (how common it is)
- **Confidence:** of students who want the IF drinks, the share who also want the THEN drink
- **Lift:** how many times more likely than chance. Above 1 is a real link, and 1.3 or more is strong.""")

    st.markdown("#### 🧃 Bundle finder: pick the drinks a student already wants")
    picks = st.multiselect("Student wants…", [bev_name(c) for c in BEV], default=["Protein Shake", "Smoothie"], max_selections=3)
    if picks:
        basket = df_all[BEV].astype(bool)
        basket.columns = [bev_name(c) for c in BEV]
        grp = basket[basket[picks].all(axis=1)]
        rows = []
        for d in basket.columns:
            if d in picks:
                continue
            conf, base = grp[d].mean(), basket[d].mean()
            rows.append({"Drink": d, "Confidence": conf, "Baseline": base, "Lift": conf / base})
        rec = pd.DataFrame(rows).sort_values("Lift", ascending=False)
        l, r = st.columns([3, 2])
        with l:
            fig = go.Figure()
            fig.add_bar(y=rec.Drink, x=rec.Baseline * 100, orientation="h", name="All students", marker_color="#c9c9c9")
            fig.add_bar(y=rec.Drink, x=rec.Confidence * 100, orientation="h", name=f"Students who want {' + '.join(picks)}",
                        marker_color=GREEN)
            fig.update_layout(barmode="group", title="% who also want each other drink", xaxis_title="%",
                              yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.15))
            chart(fig, 460)
        with r:
            st.metric("Students matching", f"{len(grp):,}", f"{len(grp) / len(df_all):.1%} of all (support)", delta_color="off")
            top = rec.iloc[0]
            st.metric(f"Best add-on: {top.Drink}", f"{top.Confidence:.0%} confidence", f"lift {top.Lift:.2f}")
            insight(f"Of the students who want <b>{' + '.join(picks)}</b>, <b>{top.Confidence:.0%}</b> also want "
                    f"<b>{top.Drink}</b>, against {top.Baseline:.0%} of all students ({top.Lift:.2f}× more likely). "
                    "Stock these together and offer them as a combo. Drinks with lift below 1 are <i>less</i> likely "
                    "than average, so they belong in a different machine.")

    st.markdown("#### All association rules")
    a, b, c_, d = st.columns(4)
    ms = a.slider("Min support", 0.05, 0.30, 0.10, 0.01)
    mc = b.slider("Min confidence", 0.30, 0.95, 0.50, 0.05)
    ml = c_.slider("Min lift", 1.0, 2.0, 1.2, 0.05)
    size = d.radio("Rule size", ["All", "2 drinks", "3 drinks", "4 drinks"], horizontal=True)
    rr = mine_rules(ms)
    rr = rr[(rr.confidence >= mc) & (rr.lift >= ml)]
    if size != "All":
        rr = rr[rr["Drinks in rule"] == int(size[0])]
    rr = rr.sort_values("lift", ascending=False)
    st.caption(f"{len(rr)} rules match")
    st.dataframe(rr, hide_index=True, width="stretch", height=320, column_config={
        "support": st.column_config.NumberColumn(format="%.3f"), "confidence": st.column_config.ProgressColumn(
            format="%.2f", min_value=0, max_value=1), "lift": st.column_config.NumberColumn(format="%.2f")})
    if len(rr):
        fig = px.scatter(rr, x="support", y="confidence", color="lift", size="Drinks in rule", hover_data=["IF student wants", "THEN also wants"],
                         color_continuous_scale="Viridis", title="Rules map (hover over a bubble)")
        chart(fig, 400)
        insight("Rules in the top-right are both common and reliable, so they make the best default bundles. "
                "Bright (high-lift) rules lower down are strong but niche, so use them for targeted offers.")

    basket = df_all[BEV].astype(bool)
    basket.columns = [bev_name(c) for c in BEV]
    lm = pd.DataFrame({a_: {b_: np.nan if a_ == b_ else (basket[a_] & basket[b_]).mean() / (basket[a_].mean() * basket[b_].mean())
                            for b_ in basket.columns} for a_ in basket.columns})
    fig = px.imshow(lm.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=0.6, zmax=1.4,
                    title="Pairwise lift between drinks (blue = wanted together)")
    chart(fig, 520)
    insight("Two clear groups appear. The <b>health group</b> (fresh juice, smoothie, coconut water, protein shake) suits "
            "machines at sports complexes and gyms. The <b>caffeine group</b> (cold coffee, iced tea, energy drink) suits "
            "libraries and late-night hostel machines. Traditional drinks (lassi, nimbu pani, flavoured milk) form a "
            "weaker third group.")

# ================================================================= REGRESSION
with tabs[7]:
    st.markdown("""**Question:** how much (₹ per month) will a student spend? Spend is a number, so this is *regression*.
- **Linear:** plain least squares
- **Ridge (L2):** shrinks all coefficients to reduce overfitting
- **Lasso (L1):** shrinks coefficients and can set weak ones to exactly zero, which acts as automatic feature selection

Move the sliders to see the penalty (alpha) at work. The defaults are the values chosen by 5-fold cross-validation.""")
    a, b = st.columns(2)
    ra = a.select_slider("Ridge alpha", options=[0.1, 1, 10, 50, 116.9, 300, 1000, 5000], value=116.9)
    la = b.select_slider("Lasso alpha", options=[0.1, 1, 4.58, 10, 25, 50, 100, 200], value=4.58)
    tbl, coefs, preds = fit_regression(ra, la)
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "R² (test)": st.column_config.NumberColumn(format="%.3f"), "R² (train)": st.column_config.NumberColumn(format="%.3f"),
        "MAE (₹)": st.column_config.NumberColumn(format="₹%.0f"), "RMSE (₹)": st.column_config.NumberColumn(format="₹%.0f")})
    lr = tbl.set_index("Model")
    insight(f"Linear explains <b>{lr.loc['Linear', 'R² (test)']:.0%}</b> of the variation in spend, and its average error is "
            f"{rupee(lr.loc['Linear', 'MAE (₹)'])}. Lasso reaches {lr.loc['Lasso', 'R² (test)']:.0%} with only "
            f"<b>{lr.loc['Lasso', 'Features kept']}</b> of 45 features. The rest of the variation comes from personal habits "
            "a survey can't capture, which is normal for stated-spend data. Push Lasso alpha to 100+ and watch features drop out "
            "and R² fall: that is the bias-variance trade-off.")

    l, r = st.columns(2)
    with l:
        top = coefs.reindex(coefs["Linear"].abs().sort_values(ascending=False).index).head(14)
        top.index = [nice(i) for i in top.index]
        m = top.reset_index(names="Feature").melt(id_vars="Feature", var_name="Model", value_name="Coefficient")
        fig = px.bar(m, y="Feature", x="Coefficient", color="Model", barmode="group", orientation="h",
                     title="Top coefficients (₹/month per +1 std-dev or vs base category)",
                     color_discrete_sequence=[GREY, BLUE, GREEN])
        fig.update_yaxes(autorange="reversed")
        chart(fig, 560)
    with r:
        fig = px.scatter(preds, x="Actual", y="Lasso", opacity=0.5, title="Lasso: actual vs predicted spend",
                         labels={"Actual": "Actual ₹/month", "Lasso": "Predicted ₹/month"}, color_discrete_sequence=[BLUE])
        mx = max(preds.Actual.max(), preds.Lasso.max())
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color=RED), name="Perfect"))
        chart(fig, 560)
    dropped = [nice(i) for i in coefs.index[coefs["Lasso"].abs() < 1e-6]]
    insight("Spend rises with <b>drinks bought per week, allowance and the number of drinks liked</b>, and falls with "
            "<b>price sensitivity and cafeteria satisfaction</b>. Hostel residents spend more. "
            f"Lasso dropped {len(dropped)} features, mostly demographics (gender, year, stream, region): <i>how</i> a "
            "student lives matters more than <i>who</i> they are. The model under-predicts the very heaviest spenders "
            "(points above the line on the right).")
    with st.expander("Features removed by Lasso"):
        st.write(", ".join(dropped) if dropped else "None at this alpha")

# ================================================================= PREDICT & BUSINESS CASE
with tabs[8]:
    st.markdown("#### 🧑‍🎓 Score a student")
    st.caption("Fill in a profile. The best classifier predicts adoption, Lasso predicts spend, and K-Means assigns a segment.")
    f1, f2, f3 = st.columns(3)
    with f1:
        allowance = st.slider("Monthly allowance (₹)", 2000, 30000, 9000, 500)
        residence = st.selectbox("Residence", ["Hostel", "PG / Rented", "Day Scholar"])
        tier = st.selectbox("City tier", ORDERS["City_Tier"])
        uni = st.selectbox("University type", ["Private", "Government", "Deemed", "Autonomous College"])
        stream = st.selectbox("Course stream", sorted(df_all.Course_Stream.unique()), index=2)
    with f2:
        bpw = st.slider("Drinks bought per week", 0, 20, 6)
        wait = st.slider("Usual cafeteria wait (min)", 1, 45, 12)
        csat = st.slider("Cafeteria satisfaction", 1, 5, 3)
        tech = st.slider("Tech comfort", 1, 5, 4)
        price = st.slider("Price sensitivity", 1, 5, 3)
    with f3:
        conv = st.slider("Importance of convenience", 1, 5, 4)
        health = st.slider("Health consciousness", 1, 5, 3)
        pay = st.selectbox("Preferred payment", ["UPI", "Debit/Credit Card", "Campus ID / Wallet", "Cash"])
        late = st.toggle("Studies late at night", True)
        gym = st.toggle("Gym / sports", False)
        liked = st.multiselect("Drinks they like", [bev_name(c) for c in BEV], default=["Cold Coffee", "Iced Tea"])

    prof_row = df_all.iloc[[0]].copy()
    med = df_all[NUM].median()
    for c_ in NUM:
        prof_row[c_] = med[c_]
    prof_row = prof_row.assign(
        Monthly_Allowance_INR=allowance, Beverages_Per_Week=bpw, Cafeteria_Wait_Minutes=wait, Cafeteria_Satisfaction=csat,
        Tech_Comfort=tech, Price_Sensitivity=price, Importance_Convenience=conv, Health_Consciousness=health,
        Num_Beverages_Liked=len(liked), Is_Hostel=int(residence == "Hostel"), Is_Gym=int(gym), Is_Late_Night=int(late),
        Digital_Payer=int(pay != "Cash"), City_Tier_Num=int(tier[-1]), UPI_Freq_Num=3 if pay == "UPI" else 2,
        University_Type=uni, Course_Stream=stream, Log_Allowance=np.log(allowance),
        Hours_On_Campus_Daily=9.0 if residence == "Hostel" else 7.5)
    p_use = fitted[BEST_CLF].predict_proba(prof_row[NUM + CAT])[0, 1]
    spend = max(0.0, float(spend_model().predict(prof_row[NUM + CAT])[0]))
    seg = seg_map[int(kmeans.predict(scaler.transform(prof_row[CLU]))[0])]
    o1, o2, o3 = st.columns(3)
    o1.metric("Probability of using the machine", f"{p_use:.0%}", "likely user" if p_use >= 0.5 else "unlikely user",
              delta_color="normal" if p_use >= 0.5 else "inverse")
    o2.metric("Predicted monthly spend", rupee(spend))
    with o3:
        st.markdown(f"**Segment**<br><span style='font-size:1.6rem;color:{SEG_COLORS[seg]};font-weight:700'>{seg}</span>",
                    unsafe_allow_html=True)
    if liked:
        basket = df_all[BEV].astype(bool)
        basket.columns = [bev_name(c) for c in BEV]
        grp = basket[basket[liked].all(axis=1)] if len(liked) <= 3 else basket[basket[liked].sum(axis=1) >= 3]
        if len(grp) >= 20:
            lifts = {d: grp[d].mean() / basket[d].mean() for d in basket.columns if d not in liked}
            best_d = max(lifts, key=lifts.get)
            insight(f"Cross-sell suggestion from association rules: offer <b>{best_d}</b> "
                    f"({grp[best_d].mean():.0%} of similar students want it, lift {lifts[best_d]:.2f}).")

    st.markdown("---")
    st.markdown("#### 🏦 Business case per campus")
    st.caption("Adoption rate and spend come from the survey. The cost inputs are rough estimates: change them to test scenarios.")
    b1, b2, b3 = st.columns(3)
    with b1:
        students = st.number_input("Students on campus", 1000, 50000, 5000, 500)
        machines = st.slider("Machines on campus", 1, 30, 8)
        realise = st.slider("Stated intent → real purchase", 10, 100, 30, 5, format="%d%%",
                            help="People overstate intent in surveys. 30% is a conservative rule of thumb.") / 100
    with b2:
        cogs = st.slider("Ingredients and packaging (% of revenue)", 20, 70, 45, 1, format="%d%%") / 100
        comm = st.slider("University commission (% of revenue)", 0, 30, 12, 1, format="%d%%") / 100
        opex = st.number_input("Upkeep, refilling, power, fees per machine per month (₹)", 0, 100000, 15000, 1000)
    with b3:
        capex = st.number_input("Cost of one machine incl. install (₹)", 50000, 2000000, 450000, 25000)
        adopt = df_all.Likely_To_Use_Flag.mean()
        spend_u = df_all.loc[df_all.Likely_To_Use_Flag == 1, "Expected_Monthly_Spend_INR"].mean()
        st.metric("Survey adoption rate", f"{adopt:.1%}")
        st.metric("Avg spend of likely users", rupee(spend_u) + "/mo")
    rev_campus = students * adopt * spend_u * realise
    rev_m = rev_campus / machines
    profit_m = rev_m * (1 - cogs - comm) - opex
    drinks_day = rev_m / df_all.Willing_Price_Per_Drink_INR.median() / 30
    payback = capex / profit_m if profit_m > 0 else np.inf
    k = st.columns(5)
    k[0].metric("Revenue / campus / month", rupee(rev_campus))
    k[1].metric("Revenue / machine / month", rupee(rev_m))
    k[2].metric("Drinks / machine / day", f"{drinks_day:.0f}")
    k[3].metric("Profit / machine / month", rupee(profit_m))
    k[4].metric("Payback (months)", "never" if np.isinf(payback) else f"{payback:.0f}")
    months = np.arange(0, 49)
    cash = -capex * machines + profit_m * machines * months
    fig = go.Figure(go.Scatter(x=months, y=cash / 1e5, mode="lines", fill="tozeroy", line=dict(color=GREEN if profit_m > 0 else RED),
                               hovertemplate="Month %{x}: ₹%{y:.1f} lakh<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color=GREY)
    fig.update_layout(title="Cumulative campus cash position (₹ lakh)", xaxis_title="Months after launch", yaxis_title="₹ lakh")
    chart(fig, 340)
    if np.isinf(payback):
        insight("With these assumptions each machine loses money every month. Reduce costs or raise usage before launching.")
    else:
        insight(f"Each machine needs about <b>{drinks_day:.0f} drinks a day</b> and pays back in about <b>{payback:.0f} months</b>. "
                "The biggest levers are the realisation rate (marketing to the Most Probable segment), placement at "
                "high-footfall hostels and libraries, and leasing machines or co-branding with beverage companies to cut "
                "the upfront cost.")
