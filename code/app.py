"""
Streamlit dashboard - Smart Campus Beverage Vending: consumer survey analytics
Run from the project folder:   streamlit run code/app.py
(Run 01 -> 02 -> 03 -> 04 scripts first so the data and model outputs exist.)
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, MOD = ROOT / "data", ROOT / "outputs" / "models"
st.set_page_config(page_title="Campus Vending Analytics", layout="wide")


@st.cache_data
def load():
    return pd.read_csv(DATA / "campus_beverage_survey_scored.csv")


df_all = load()
st.title("Smart Beverage Vending for Indian Campuses")
st.caption("Synthetic consumer survey (n = 2,000 students) - descriptive, diagnostic and predictive analytics")

# ---------------- sidebar filters ----------------
st.sidebar.header("Filters")
f = {}
for col in ["City_Tier", "University_Type", "Residence_Type", "Gender", "Course_Stream"]:
    opts = sorted(df_all[col].unique())
    f[col] = st.sidebar.multiselect(col.replace("_", " "), opts, default=opts)
df = df_all.copy()
for col, vals in f.items():
    df = df[df[col].isin(vals)]
st.sidebar.write(f"**{len(df):,}** respondents selected")
if df.empty:
    st.warning("No respondents match these filters.")
    st.stop()

# ---------------- KPIs ----------------
k = st.columns(5)
k[0].metric("Respondents", f"{len(df):,}")
k[1].metric("Likely to use", f"{df.Likely_To_Use_Flag.mean():.1%}")
k[2].metric("Median spend / month", f"Rs. {df.Expected_Monthly_Spend_INR.median():,.0f}")
k[3].metric("Median price / drink", f"Rs. {df.Willing_Price_Per_Drink_INR.median():,.0f}")
k[4].metric("Cashless preference", f"{df.Digital_Payer.mean():.1%}")

tabs = st.tabs(["Descriptive", "Diagnostic", "Classification", "Clustering", "Association", "Regression"])

with tabs[0]:
    c1, c2 = st.columns(2)
    bev = (df.filter(like="Likes_").mean() * 100).sort_values()
    bev.index = bev.index.str.replace("Likes_", "").str.replace("_", " ")
    c1.plotly_chart(px.bar(bev, orientation="h", title="Drink interest (%)", labels={"value": "%", "index": ""}),
                    width="stretch")
    c2.plotly_chart(px.histogram(df, x="Expected_Monthly_Spend_INR", color="Likely_To_Use", nbins=40,
                                 barmode="overlay", title="Expected monthly spend (Rs.)"), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.pie(df, names="Preferred_Payment", title="Preferred payment", hole=0.45),
                    width="stretch")
    c4.plotly_chart(px.density_heatmap(df, x="Preferred_Time_Slot", y="Preferred_Location", text_auto=True,
                                       title="Where & when students want a machine"), width="stretch")
    st.info("Cold coffee and fresh juice lead demand; most students prefer UPI, so a cashless machine is viable.")

with tabs[1]:
    driver = st.selectbox("Compare adoption rate by", ["Residence_Type", "Wait_Band", "Allowance_Band", "City_Tier",
                                                       "Price_Sensitivity", "Tech_Comfort", "Cafeteria_Satisfaction",
                                                       "Importance_Convenience", "Late_Night_Study", "Course_Stream"])
    g = df.groupby(driver, observed=True)["Likely_To_Use_Flag"].mean().mul(100).reset_index()
    st.plotly_chart(px.bar(g, x=driver, y="Likely_To_Use_Flag", labels={"Likely_To_Use_Flag": "% likely to use"},
                           title=f"Adoption rate by {driver.replace('_', ' ')}"), width="stretch")
    num = ["Monthly_Allowance_INR", "Beverages_Per_Week", "Cafeteria_Wait_Minutes", "Cafeteria_Satisfaction",
           "Tech_Comfort", "Health_Consciousness", "Price_Sensitivity", "Importance_Convenience", "Is_Hostel",
           "Is_Late_Night", "Num_Beverages_Liked", "Willing_Price_Per_Drink_INR", "Expected_Monthly_Spend_INR",
           "Likely_To_Use_Flag"]
    st.plotly_chart(px.imshow(df[num].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation matrix", height=650), width="stretch")

with tabs[2]:
    st.subheader("Will a student use the machine?")
    st.dataframe(pd.read_csv(MOD / "A_classification_comparison.csv"), width="stretch")
    c1, c2 = st.columns(2)
    c1.image(str(MOD / "A2_roc_curves.png"))
    c2.image(str(MOD / "A4_feature_importance.png"))
    st.image(str(MOD / "A3_confusion_matrices.png"))

with tabs[3]:
    st.subheader("Customer segments (K-Means, k = 3)")
    seg = df.groupby("Segment").agg(Students=("Respondent_ID", "count"),
                                    Adoption=("Likely_To_Use_Flag", "mean"),
                                    Median_Spend=("Expected_Monthly_Spend_INR", "median"),
                                    Median_Allowance=("Monthly_Allowance_INR", "median"),
                                    Price_Sensitivity=("Price_Sensitivity", "mean"),
                                    Cafe_Satisfaction=("Cafeteria_Satisfaction", "mean"),
                                    Drinks_Liked=("Num_Beverages_Liked", "mean")).round(2)
    st.dataframe(seg.reindex(["Most Probable", "Medium Probable", "Least Probable"]), width="stretch")
    x = st.selectbox("X axis", ["Monthly_Allowance_INR", "Price_Sensitivity", "Cafeteria_Wait_Minutes"], 0)
    y = st.selectbox("Y axis", ["Expected_Monthly_Spend_INR", "Num_Beverages_Liked", "Beverages_Per_Week"], 0)
    st.plotly_chart(px.scatter(df, x=x, y=y, color="Segment", opacity=0.6,
                               color_discrete_map={"Most Probable": "#2a7f62", "Medium Probable": "#e1a034",
                                                   "Least Probable": "#c0504d"}), width="stretch")
    st.image(str(MOD / "B1_elbow_silhouette.png"))

with tabs[4]:
    st.subheader("Which drinks go together? (Apriori)")
    rules = pd.read_csv(MOD / "C_association_rules.csv")
    c1, c2, c3 = st.columns(3)
    ms = c1.slider("Min support", 0.10, 0.40, 0.10, 0.01)
    mc = c2.slider("Min confidence", 0.50, 0.95, 0.55, 0.05)
    ml = c3.slider("Min lift", 1.1, 2.0, 1.2, 0.05)
    n = st.radio("Rule size", ["All", "2 drinks (A -> B)", "3 drinks ({A,B} -> C)"], horizontal=True)
    r = rules[(rules.support >= ms) & (rules.confidence >= mc) & (rules.lift >= ml)]
    if n.startswith("2"):
        r = r[r.n_items == 2]
    elif n.startswith("3"):
        r = r[r.n_items == 3]
    st.write(f"{len(r)} rules")
    st.dataframe(r, width="stretch")
    st.image(str(MOD / "C3_pairwise_lift.png"))

with tabs[5]:
    st.subheader("How much will a student spend per month?")
    st.dataframe(pd.read_csv(MOD / "D_regression_comparison.csv"), width="stretch")
    st.dataframe(pd.read_csv(MOD / "D_regression_coefficients.csv", index_col=0)
                 .sort_values("Lasso", key=abs, ascending=False), width="stretch")
    c1, c2 = st.columns(2)
    c1.image(str(MOD / "D2_actual_vs_predicted.png"))
    c2.image(str(MOD / "D1_coefficients.png"))
