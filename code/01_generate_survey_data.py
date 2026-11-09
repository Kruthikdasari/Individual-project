"""
01_generate_survey_data.py
Synthetic consumer survey for a smart beverage vending machine start-up
targeting Indian universities and colleges.

Every respondent gets a hidden "adoption propensity" built from realistic
drivers (convenience pain, tech comfort, allowance, hostel life, cafeteria
dissatisfaction, price sensitivity). That hidden score then drives the
observable answers, so the algorithms have real, explainable patterns to find.

A small amount of realistic "survey mess" is injected on purpose
(missing values, duplicates, inconsistent labels, typos, outliers) so the
data-cleaning step has genuine work to do.

Output: ../data/campus_beverage_survey_raw.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 2000
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)


def pick(options, probs, n=N):
    return rng.choice(options, size=n, p=np.array(probs) / np.sum(probs))


def likert(score, noise=0.8):
    """Turn a continuous score (~0 centred) into a 1-5 Likert answer."""
    v = 3 + score + rng.normal(0, noise, len(score))
    return np.clip(np.round(v), 1, 5).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- demographics
gender = pick(["Male", "Female", "Other"], [0.53, 0.45, 0.02])
year = pick(["1st Year", "2nd Year", "3rd Year", "4th Year", "Postgraduate", "PhD"],
            [0.22, 0.21, 0.19, 0.15, 0.18, 0.05])
age_base = {"1st Year": 18, "2nd Year": 19, "3rd Year": 20, "4th Year": 21,
            "Postgraduate": 23, "PhD": 26}
age = np.array([age_base[y] for y in year]) + rng.integers(0, 3, N)

stream = pick(["Engineering", "Management", "Science", "Commerce",
               "Arts & Humanities", "Medical & Pharmacy", "Law", "Design"],
              [0.30, 0.15, 0.13, 0.12, 0.11, 0.09, 0.05, 0.05])
uni_type = pick(["Private", "Government", "Deemed", "Autonomous College"],
                [0.38, 0.27, 0.20, 0.15])
city_tier = pick(["Tier 1", "Tier 2", "Tier 3"], [0.45, 0.38, 0.17])
region = pick(["South", "North", "West", "East", "Central", "North-East"],
              [0.28, 0.25, 0.22, 0.12, 0.08, 0.05])
residence = pick(["Hostel", "PG / Rented", "Day Scholar"], [0.44, 0.24, 0.32])

# --------------------------------------------------------------- money & habits
tier_mult = np.select([city_tier == "Tier 1", city_tier == "Tier 2"], [1.35, 1.0], 0.75)
uni_mult = np.select([uni_type == "Private", uni_type == "Deemed"], [1.30, 1.15], 0.90)
pg_mult = np.where(np.isin(year, ["Postgraduate", "PhD"]), 1.25, 1.0)  # stipends / older
allowance = rng.lognormal(mean=np.log(7000), sigma=0.45, size=N) * tier_mult * uni_mult * pg_mult
allowance = np.clip(np.round(allowance, -2), 1500, 40000)

hours_on_campus = np.clip(np.round(rng.normal(7.5, 1.8, N)
                                   + np.where(residence == "Hostel", 1.5, 0), 1), 3, 14)
gym = pick(["Yes", "No"], [0.38, 0.62])
gym_flag = (gym == "Yes").astype(int)
hostel_flag = (residence == "Hostel").astype(int)

late_night_p = sigmoid(-0.6 + 1.3 * hostel_flag
                       + 0.5 * np.isin(stream, ["Engineering", "Medical & Pharmacy", "Law"]))
late_night = np.where(rng.random(N) < late_night_p, "Yes", "No")
late_flag = (late_night == "Yes").astype(int)

current_source = pick(["Campus Cafeteria", "Nearby Shop / Tapri", "Food Delivery App",
                       "Bring From Home", "Campus Kiosk"],
                      [0.42, 0.27, 0.10, 0.12, 0.09])
caf_wait = np.clip(np.round(rng.gamma(3.0, 4.0, N)
                            + np.where(uni_type == "Government", 4, 0)), 1, 45)
caf_sat = likert(-0.07 * (caf_wait - 12) + np.where(uni_type == "Private", 0.3, -0.1), 0.9)

tech_comfort = likert(0.45 + 0.4 * np.isin(stream, ["Engineering", "Design", "Management"])
                      + 0.3 * (city_tier == "Tier 1") - 0.3 * (city_tier == "Tier 3"), 0.8)
upi_freq = np.where(tech_comfort >= 4,
                    pick(["Daily", "Weekly", "Rarely"], [0.78, 0.18, 0.04]),
                    pick(["Daily", "Weekly", "Rarely"], [0.45, 0.35, 0.20]))
pay_pref = np.where(upi_freq == "Daily",
                    pick(["UPI", "Debit/Credit Card", "Campus ID / Wallet", "Cash"],
                         [0.80, 0.07, 0.08, 0.05]),
                    pick(["UPI", "Debit/Credit Card", "Campus ID / Wallet", "Cash"],
                         [0.45, 0.12, 0.08, 0.35]))

health = likert(0.2 + 0.8 * gym_flag + 0.25 * (gender == "Female")
                + 0.3 * (stream == "Medical & Pharmacy") - 0.4, 0.9)
price_sens = likert(-0.00012 * (allowance - 8000) + 0.3 * (uni_type == "Government")
                    + 0.3 * (city_tier == "Tier 3"), 0.8)
conv_importance = likert(0.35 + 0.035 * (caf_wait - 12) + 0.35 * hostel_flag
                         + 0.15 * (hours_on_campus - 7.5), 0.8)
hygiene_importance = likert(0.6 + 0.3 * health / 5, 0.8)

used_vending = np.where(rng.random(N) < sigmoid(-1.0 + 0.6 * (city_tier == "Tier 1")
                                                + 0.5 * (uni_type == "Private")), "Yes", "No")
bev_per_week = np.clip(np.round(rng.gamma(2.2, 2.2, N) + 1.2 * hostel_flag
                                + 1.0 * late_flag + 0.00008 * allowance), 0, 28).astype(int)

# ----------------------------------------------------- hidden adoption propensity
z = lambda x: (x - x.mean()) / x.std()
propensity = (0.55 * z(conv_importance) + 0.50 * z(tech_comfort)
              + 0.45 * z(np.log(allowance)) - 0.45 * z(price_sens)
              - 0.40 * z(caf_sat) + 0.30 * z(caf_wait)
              + 0.35 * hostel_flag + 0.30 * late_flag
              + 0.25 * (used_vending == "Yes") + 0.20 * z(bev_per_week)
              + 0.15 * (pay_pref == "Cash") * -1.5
              + rng.normal(0, 0.55, N))
propensity = z(propensity)

interest_level = np.clip(np.round(3.1 + 1.05 * propensity + rng.normal(0, 0.35, N)), 1, 5).astype(int)
p_use = sigmoid(-0.15 + 1.9 * propensity)
likely_to_use = np.where(rng.random(N) < p_use, "Yes", "No")
use_flag = (likely_to_use == "Yes").astype(int)

# -------------------------------- beverage interest basket (10 items, for ARM)
# Three latent "taste tribes" make co-purchase patterns realistic.
caffeine_t = rng.normal(0, 1, N) + 0.6 * late_flag + 0.3 * (stream == "Engineering")
health_t = rng.normal(0, 1, N) + 0.35 * (health - 3) + 0.4 * gym_flag
desi_t = rng.normal(0, 1, N) + 0.5 * np.isin(region, ["North", "West", "Central"]) \
         + 0.3 * (city_tier != "Tier 1")
base = -0.6 + 0.6 * propensity  # interested people tick more boxes

bev = {
    "Cold_Coffee":          base + 1.25 * caffeine_t + 0.6,
    "Iced_Tea":             base + 0.85 * caffeine_t + 0.35 * health_t - 0.1,
    "Energy_Drink":         base + 1.05 * caffeine_t + 0.5 * gym_flag - 0.7 * (health - 3) / 2 - 0.5,
    "Fresh_Juice":          base + 1.15 * health_t + 0.4,
    "Smoothie":             base + 1.00 * health_t + 0.2 * (gender == "Female") - 0.3,
    "Protein_Shake":        base + 0.70 * health_t + 1.5 * gym_flag - 1.2,
    "Coconut_Water":        base + 0.90 * health_t + 0.3 * (region == "South") - 0.4,
    "Flavoured_Milk":       base + 0.70 * desi_t + 0.2 * caffeine_t + 0.1,
    "Lassi_Buttermilk":     base + 1.10 * desi_t - 0.2,
    "Nimbu_Pani_Lemonade":  base + 0.80 * desi_t + 0.3 * health_t - 0.1,
}
bev_cols = list(bev)
bev_df = pd.DataFrame({k: (rng.random(N) < sigmoid(v)).astype(int) for k, v in bev.items()})
# Everyone who answers picks at least one drink
none_picked = bev_df.sum(axis=1) == 0
bev_df.loc[none_picked, "Cold_Coffee"] = 1

# ----------------------------------------------------------- preferences
pref_location = np.where(hostel_flag == 1,
                         pick(["Hostel Block", "Library", "Academic Block", "Sports Complex", "Canteen Area"],
                              [0.40, 0.20, 0.20, 0.12, 0.08]),
                         pick(["Hostel Block", "Library", "Academic Block", "Sports Complex", "Canteen Area"],
                              [0.05, 0.30, 0.38, 0.12, 0.15]))
pref_location = np.where((gym_flag == 1) & (rng.random(N) < 0.15), "Sports Complex", pref_location)
pref_time = np.where(late_flag == 1,
                     pick(["Morning (8-11)", "Afternoon (11-3)", "Evening (3-7)", "Late Night (7-12)"],
                          [0.15, 0.25, 0.25, 0.35]),
                     pick(["Morning (8-11)", "Afternoon (11-3)", "Evening (3-7)", "Late Night (7-12)"],
                          [0.28, 0.40, 0.27, 0.05]))
subscription = np.where(propensity > 0.7, pick(["Yes", "Maybe", "No"], [0.55, 0.35, 0.10]),
                        np.where(propensity > -0.4, pick(["Yes", "Maybe", "No"], [0.20, 0.50, 0.30]),
                                 pick(["Yes", "Maybe", "No"], [0.05, 0.30, 0.65])))
top_feature = pick(["Lower price than cafe", "No queue / speed", "24x7 availability",
                    "Hygiene / no human touch", "Healthy options", "Loyalty points & offers"],
                   [0.22, 0.22, 0.17, 0.12, 0.14, 0.13])

# -------------------------------------------------------------- spend (regression)
n_drinks_liked = bev_df.sum(axis=1).values
wtp_per_drink = (38 + 0.0016 * allowance - 4.5 * (price_sens - 3)
                 + 3.5 * (health - 3) * bev_df["Fresh_Juice"].values
                 + 9 * bev_df["Protein_Shake"].values
                 + 6 * (city_tier == "Tier 1") - 5 * (city_tier == "Tier 3")
                 + 4 * propensity + rng.normal(0, 6, N))
wtp_per_drink = np.clip(np.round(wtp_per_drink), 20, 150)

exp_uses_per_week = np.clip(0.25 * bev_per_week * sigmoid(1.6 * propensity + 0.3)
                            + 0.8 * hostel_flag * use_flag + 0.5 * late_flag
                            + 0.12 * n_drinks_liked + rng.normal(0, 0.6, N), 0, 14)
monthly_spend = np.round(exp_uses_per_week * 4.3 * wtp_per_drink * rng.normal(1, 0.12, N), -1)
monthly_spend = np.clip(monthly_spend, 0, None)
# A student who says "No" still might buy occasionally, but much less.
monthly_spend = np.where(use_flag == 0, np.round(monthly_spend * 0.35, -1), monthly_spend)

# ------------------------------------------------------------------ assemble
df = pd.DataFrame({
    "Respondent_ID": [f"R{1000 + i}" for i in range(N)],
    "Age": age, "Gender": gender, "Year_of_Study": year, "Course_Stream": stream,
    "University_Type": uni_type, "City_Tier": city_tier, "Region": region,
    "Residence_Type": residence, "Monthly_Allowance_INR": allowance.astype(int),
    "Hours_On_Campus_Daily": hours_on_campus, "Beverages_Per_Week": bev_per_week,
    "Current_Beverage_Source": current_source, "Cafeteria_Wait_Minutes": caf_wait.astype(int),
    "Cafeteria_Satisfaction": caf_sat, "Preferred_Payment": pay_pref,
    "UPI_Usage_Frequency": upi_freq, "Tech_Comfort": tech_comfort,
    "Health_Consciousness": health, "Price_Sensitivity": price_sens,
    "Importance_Convenience": conv_importance, "Importance_Hygiene": hygiene_importance,
    "Gym_or_Sports": gym, "Late_Night_Study": late_night,
    "Used_Vending_Machine_Before": used_vending,
    "Preferred_Location": pref_location, "Preferred_Time_Slot": pref_time,
    "Most_Valued_Feature": top_feature,
})
for c in bev_cols:
    df[f"Likes_{c}"] = bev_df[c].values
df["Subscription_Interest"] = subscription
df["Interest_Level"] = interest_level
df["Willing_Price_Per_Drink_INR"] = wtp_per_drink.astype(int)
df["Expected_Monthly_Spend_INR"] = monthly_spend.astype(int)
df["Likely_To_Use"] = likely_to_use

# --------------------------------------------- inject realistic survey mess
mess = np.random.default_rng(7)
# 1) inconsistent category spellings (free-text style entry)
for col, variants in {
    "Gender": {"Male": ["male", "M", " Male"], "Female": ["female", "F", "Female "]},
    "City_Tier": {"Tier 1": ["tier1", "Tier-1"], "Tier 2": ["tier 2", "Tier-2"]},
    "Preferred_Payment": {"UPI": ["upi", "GPay/UPI", "PhonePe"]},
    "Likely_To_Use": {"Yes": ["yes", "Y"], "No": ["no", "N"]},
}.items():
    for good, bads in variants.items():
        idx = df.index[df[col] == good]
        hit = mess.choice(idx, size=int(0.06 * len(idx)), replace=False)
        df.loc[hit, col] = mess.choice(bads, size=len(hit))
# 2) missing values (people skip questions)
for col, frac in {"Monthly_Allowance_INR": 0.04, "Cafeteria_Wait_Minutes": 0.03,
                  "Tech_Comfort": 0.02, "Health_Consciousness": 0.02,
                  "Preferred_Location": 0.025, "Willing_Price_Per_Drink_INR": 0.02,
                  "Region": 0.015, "Hours_On_Campus_Daily": 0.02}.items():
    df.loc[mess.choice(df.index, int(frac * N), replace=False), col] = np.nan
# 3) impossible / outlier entries (typos)
df.loc[mess.choice(df.index, 6, replace=False), "Age"] = [99, 2, 150, 0, 210, 88]
df.loc[mess.choice(df.index, 5, replace=False), "Monthly_Allowance_INR"] = [500000, 900000, 75, 250000, 10]
df.loc[mess.choice(df.index, 4, replace=False), "Expected_Monthly_Spend_INR"] = [-200, 95000, -50, 120000]
df.loc[mess.choice(df.index, 3, replace=False), "Cafeteria_Satisfaction"] = [7, 0, 9]
# 4) duplicate submissions (same form submitted twice)
dups = df.sample(30, random_state=3)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=11).reset_index(drop=True)

df.to_csv(OUT / "campus_beverage_survey_raw.csv", index=False)
print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {OUT / 'campus_beverage_survey_raw.csv'}")
