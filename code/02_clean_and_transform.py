"""
02_clean_and_transform.py
Cleans the raw survey and engineers model-ready features.

Steps (each one is logged so it can be quoted in the report):
 1. Remove duplicate form submissions
 2. Standardise inconsistent category labels
 3. Fix impossible values (age, Likert out of 1-5, negative spend)
 4. Treat outliers (allowance / spend) with IQR capping
 5. Impute missing values (median for numbers, mode for categories)
 6. Feature engineering (encoded flags, ratios, interest probability segment)

Outputs: ../data/campus_beverage_survey_clean.csv
         ../outputs/cleaning_log.txt
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "campus_beverage_survey_raw.csv"
(ROOT / "outputs").mkdir(exist_ok=True)
log = []


def note(msg):
    print(msg)
    log.append(msg)


df = pd.read_csv(RAW)
note(f"Raw shape: {df.shape}")

# 1. Duplicates -----------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
note(f"[1] Dropped {before - len(df)} exact duplicate submissions -> {len(df)} rows")
dup_ids = df["Respondent_ID"].duplicated().sum()
df = df.drop_duplicates("Respondent_ID")
note(f"    Dropped {dup_ids} repeated Respondent_IDs")

# 2. Standardise labels -------------------------------------------------------
maps = {
    "Gender": {"male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other"},
    "City_Tier": {"tier1": "Tier 1", "tier-1": "Tier 1", "tier 1": "Tier 1",
                  "tier2": "Tier 2", "tier-2": "Tier 2", "tier 2": "Tier 2", "tier 3": "Tier 3"},
    "Preferred_Payment": {"upi": "UPI", "gpay/upi": "UPI", "phonepe": "UPI",
                          "debit/credit card": "Debit/Credit Card", "cash": "Cash",
                          "campus id / wallet": "Campus ID / Wallet"},
    "Likely_To_Use": {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"},
}
for col, m in maps.items():
    n_before = df[col].nunique()
    df[col] = df[col].astype(str).str.strip().str.lower().map(m).fillna(df[col].str.strip())
    note(f"[2] {col}: {n_before} spellings -> {df[col].nunique()} clean categories {sorted(df[col].unique())}")

# 3. Impossible values -------------------------------------------------------------
bad_age = ~df["Age"].between(16, 40)
note(f"[3] {bad_age.sum()} impossible ages (outside 16-40) set to missing")
df.loc[bad_age, "Age"] = np.nan
likert_cols = ["Cafeteria_Satisfaction", "Tech_Comfort", "Health_Consciousness",
               "Price_Sensitivity", "Importance_Convenience", "Importance_Hygiene", "Interest_Level"]
for c in likert_cols:
    bad = ~df[c].between(1, 5) & df[c].notna()
    if bad.any():
        note(f"    {c}: {bad.sum()} answers outside 1-5 set to missing")
        df.loc[bad, c] = np.nan
neg = df["Expected_Monthly_Spend_INR"] < 0
note(f"    {neg.sum()} negative spend values set to missing")
df.loc[neg, "Expected_Monthly_Spend_INR"] = np.nan

# 4. Outliers: IQR capping -----------------------------------------------------
for c in ["Monthly_Allowance_INR"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    lo, hi = max(q1 - 3 * (q3 - q1), 0), q3 + 3 * (q3 - q1)
    out = (df[c] < lo) | (df[c] > hi)
    note(f"[4] {c}: {out.sum()} extreme outliers outside [{lo:,.0f}, {hi:,.0f}] -> set to missing, then imputed")
    df.loc[out, c] = np.nan
# Spend: a domain rule is safer than IQR here (spend is naturally right-skewed,
# so IQR would wrongly cut genuine heavy users). Spend above Rs.10,000/month on
# campus drinks, or above the student's own allowance, is not credible.
bad_spend = (df["Expected_Monthly_Spend_INR"] > 10000) | \
            (df["Expected_Monthly_Spend_INR"] > df["Monthly_Allowance_INR"])
note(f"    Expected_Monthly_Spend_INR: {bad_spend.sum()} non-credible values (> Rs.10k or > allowance) -> missing")
df.loc[bad_spend, "Expected_Monthly_Spend_INR"] = np.nan
# Allowance below Rs.1,000/month is a typo (e.g. 10, 75)
tiny = df["Monthly_Allowance_INR"] < 1000
df.loc[tiny, "Monthly_Allowance_INR"] = np.nan
note(f"    {tiny.sum()} allowance entries below Rs.1,000 treated as typos")

# 5. Missing values -----------------------------------------------------------------
missing = df.isna().sum()
note("[5] Missing values before imputation:\n" + missing[missing > 0].to_string())
num_cols = df.select_dtypes("number").columns
for c in num_cols:
    if df[c].isna().any():
        # median within residence type is more realistic than a global median
        df[c] = df[c].fillna(df.groupby("Residence_Type")[c].transform("median")).fillna(df[c].median())
for c in df.select_dtypes(exclude="number").columns:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode()[0])
likert_cols_present = [c for c in likert_cols if c in df]
df[likert_cols_present] = df[likert_cols_present].round().astype(int)
df["Age"] = df["Age"].round().astype(int)
note(f"    Missing values after imputation: {int(df.isna().sum().sum())}")

# 6. Feature engineering --------------------------------------------------------------
bev_cols = [c for c in df.columns if c.startswith("Likes_")]
df["Num_Beverages_Liked"] = df[bev_cols].sum(axis=1)
df["Likely_To_Use_Flag"] = (df["Likely_To_Use"] == "Yes").astype(int)
df["Is_Hostel"] = (df["Residence_Type"] == "Hostel").astype(int)
df["Is_Gym"] = (df["Gym_or_Sports"] == "Yes").astype(int)
df["Is_Late_Night"] = (df["Late_Night_Study"] == "Yes").astype(int)
df["Used_Vending_Flag"] = (df["Used_Vending_Machine_Before"] == "Yes").astype(int)
df["Digital_Payer"] = (df["Preferred_Payment"] != "Cash").astype(int)
df["City_Tier_Num"] = df["City_Tier"].str.extract(r"(\d)").astype(int)
df["UPI_Freq_Num"] = df["UPI_Usage_Frequency"].map({"Rarely": 1, "Weekly": 2, "Daily": 3})
df["Spend_Share_of_Allowance_%"] = (100 * df["Expected_Monthly_Spend_INR"]
                                    / df["Monthly_Allowance_INR"]).round(2)
df["Allowance_Band"] = pd.cut(df["Monthly_Allowance_INR"], [0, 5000, 10000, 15000, 1e9],
                              labels=["<5k", "5k-10k", "10k-15k", "15k+"])
df["Wait_Band"] = pd.cut(df["Cafeteria_Wait_Minutes"], [0, 5, 10, 20, 1e9],
                         labels=["0-5 min", "6-10 min", "11-20 min", "20+ min"])
note(f"[6] Engineered 12 new features; final shape {df.shape}")

df.to_csv(ROOT / "data" / "campus_beverage_survey_clean.csv", index=False)
(ROOT / "outputs" / "cleaning_log.txt").write_text("\n".join(log))
print("Saved clean data + cleaning log")
