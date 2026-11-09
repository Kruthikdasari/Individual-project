"""
03_eda.py  -  Descriptive & diagnostic analytics
Every chart is saved to ../outputs/eda/ and a one-line insight for each is
written to ../outputs/eda/eda_insights.txt (numbers are computed, not typed).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data" / "campus_beverage_survey_clean.csv")
sns.set_theme(style="whitegrid", palette="deep")
YES, NO = "#2a7f62", "#c0504d"
insights = []


def save(name, insight):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=130)
    plt.close()
    insights.append(f"{name}: {insight}")
    print(name, "->", insight)


# 1. Target split
share = df["Likely_To_Use"].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(share.index, share.values, color=[NO if i == "No" else YES for i in share.index])
for i, v in enumerate(share.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
ax.set(title="Would students use the vending machine?", ylabel="% of respondents")
save("01_target_split", f"{share['Yes']:.1f}% say they would use the machine - a large addressable base "
     f"and a fairly balanced target for classification.")

# 2. Interest level distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Interest_Level", data=df, ax=ax, color="#4c72b0")
ax.set(title="Interest level (1 = not at all, 5 = very)", xlabel="Interest level", ylabel="Respondents")
top2 = (df["Interest_Level"] >= 4).mean() * 100
save("02_interest_level", f"{top2:.1f}% rate interest 4 or 5 (top-2-box).")

# 3. Adoption by residence
r = df.groupby("Residence_Type")["Likely_To_Use_Flag"].mean().sort_values() * 100
fig, ax = plt.subplots(figsize=(6, 4))
r.plot.barh(ax=ax, color="#4c72b0")
ax.set(title="Adoption rate by residence type", xlabel="% likely to use")
save("03_adoption_by_residence", f"Hostel students adopt most ({r['Hostel']:.1f}%) vs day scholars "
     f"({r['Day Scholar']:.1f}%) - hostels are the priority install site.")

# 4. Cafeteria wait vs adoption
w = df.groupby("Wait_Band", observed=True)["Likely_To_Use_Flag"].mean() * 100
w = w.reindex(["0-5 min", "6-10 min", "11-20 min", "20+ min"])
fig, ax = plt.subplots(figsize=(6, 4))
w.plot(kind="bar", ax=ax, color="#dd8452", rot=0)
ax.set(title="Longer cafeteria queues -> higher adoption", xlabel="Usual cafeteria wait",
       ylabel="% likely to use")
save("04_wait_vs_adoption", f"Adoption rises from {w.iloc[0]:.1f}% (0-5 min wait) to {w.iloc[-1]:.1f}% "
     f"(20+ min) - queue pain is a core purchase driver.")

# 5. Allowance band vs adoption & spend
a = df.groupby("Allowance_Band", observed=True).agg(adopt=("Likely_To_Use_Flag", "mean"),
                                                     spend=("Expected_Monthly_Spend_INR", "median"))
a = a.reindex(["<5k", "5k-10k", "10k-15k", "15k+"])
fig, ax1 = plt.subplots(figsize=(6.5, 4))
ax1.bar(a.index, a["adopt"] * 100, color="#4c72b0", label="% likely to use")
ax2 = ax1.twinx()
ax2.plot(a.index, a["spend"], color="#c44e52", marker="o", label="Median monthly spend (Rs.)")
ax2.grid(False)
ax1.set(title="Pocket money drives both adoption and spend", xlabel="Monthly allowance (Rs.)",
        ylabel="% likely to use")
ax2.set_ylabel("Median expected spend (Rs.)")
save("05_allowance_vs_adoption_spend", f"Adoption goes from {a['adopt'].iloc[0]*100:.1f}% to "
     f"{a['adopt'].iloc[-1]*100:.1f}% and median spend from Rs.{a['spend'].iloc[0]:.0f} to "
     f"Rs.{a['spend'].iloc[-1]:.0f} across allowance bands.")

# 6. Payment preference
p = df["Preferred_Payment"].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(6, 4))
p.plot.bar(ax=ax, color="#55a868", rot=15)
ax.set(title="Preferred payment method", ylabel="% of respondents")
save("06_payment_preference", f"{100 - p.get('Cash', 0):.1f}% prefer a cashless method (UPI alone "
     f"{p['UPI']:.1f}%) - a cashless-only machine is viable.")

# 7. Beverage popularity
bev = df.filter(like="Likes_").mean().sort_values() * 100
bev.index = bev.index.str.replace("Likes_", "").str.replace("_", " ")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
bev.plot.barh(ax=ax, color="#8172b2")
ax.set(title="Which drinks do students want?", xlabel="% interested")
save("07_beverage_popularity", f"Top drinks: {bev.index[-1]} ({bev.iloc[-1]:.0f}%), {bev.index[-2]} "
     f"({bev.iloc[-2]:.0f}%), {bev.index[-3]} ({bev.iloc[-3]:.0f}%). Lowest: {bev.index[0]} "
     f"({bev.iloc[0]:.0f}%) - a niche SKU.")

# 8. Spend distribution
fig, ax = plt.subplots(figsize=(6.5, 4))
sns.histplot(data=df, x="Expected_Monthly_Spend_INR", hue="Likely_To_Use", bins=40,
             palette={"Yes": YES, "No": NO}, ax=ax, element="step")
ax.set(title="Expected monthly spend", xlabel="Rs. per month")
m = df.groupby("Likely_To_Use")["Expected_Monthly_Spend_INR"].median()
save("08_spend_distribution", f"Median spend Rs.{m['Yes']:.0f}/month for likely users vs Rs.{m['No']:.0f} "
     f"for others; distribution is right-skewed (a few heavy users).")

# 9. Willingness to pay per drink
fig, ax = plt.subplots(figsize=(6.5, 4))
sns.boxplot(data=df, x="City_Tier", y="Willing_Price_Per_Drink_INR", order=["Tier 1", "Tier 2", "Tier 3"],
            ax=ax, color="#64b5cd")
ax.set(title="Willingness to pay per drink by city tier", ylabel="Rs. per drink", xlabel="")
t = df.groupby("City_Tier")["Willing_Price_Per_Drink_INR"].median()
save("09_wtp_by_tier", f"Median acceptable price: Tier 1 Rs.{t['Tier 1']:.0f}, Tier 2 Rs.{t['Tier 2']:.0f}, "
     f"Tier 3 Rs.{t['Tier 3']:.0f} - price lists should vary by tier.")

# 10. Preferred location x time heatmap
ct = pd.crosstab(df["Preferred_Location"], df["Preferred_Time_Slot"], normalize="all") * 100
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={"label": "% of respondents"})
ax.set(title="Where and when students want a machine", xlabel="", ylabel="")
loc = df["Preferred_Location"].value_counts(normalize=True) * 100
save("10_location_time_heatmap", f"Most-wanted spots: {loc.index[0]} ({loc.iloc[0]:.0f}%) and "
     f"{loc.index[1]} ({loc.iloc[1]:.0f}%); late-night demand concentrates at hostels.")

# 11. Correlation heatmap
num = ["Age", "Monthly_Allowance_INR", "Hours_On_Campus_Daily", "Beverages_Per_Week",
       "Cafeteria_Wait_Minutes", "Cafeteria_Satisfaction", "Tech_Comfort", "Health_Consciousness",
       "Price_Sensitivity", "Importance_Convenience", "Is_Hostel", "Is_Late_Night",
       "Num_Beverages_Liked", "Interest_Level", "Willing_Price_Per_Drink_INR",
       "Expected_Monthly_Spend_INR", "Likely_To_Use_Flag"]
corr = df[num].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation matrix")
# Drop the target and other *outcome* answers so we only rank genuine drivers
c = corr["Likely_To_Use_Flag"].drop(["Likely_To_Use_Flag", "Interest_Level",
                                     "Expected_Monthly_Spend_INR", "Willing_Price_Per_Drink_INR"]).sort_values()
save("11_correlation_heatmap", f"(Outcome columns excluded) Strongest positive driver to adoption: {c.index[-1]} (r={c.iloc[-1]:.2f}), "
     f"{c.index[-2]} (r={c.iloc[-2]:.2f}); strongest negative: {c.index[0]} (r={c.iloc[0]:.2f}), "
     f"{c.index[1]} (r={c.iloc[1]:.2f}).")

# 12. Driver chart: correlation with adoption
fig, ax = plt.subplots(figsize=(7, 5))
c.plot.barh(ax=ax, color=[YES if v > 0 else NO for v in c.values])
ax.set(title="What pushes adoption up (green) or down (red)?", xlabel="Correlation with Likely_To_Use")
save("12_adoption_drivers", "Convenience, tech comfort, allowance and queue length push adoption up; "
     "price sensitivity and cafeteria satisfaction pull it down.")

# 13. Price sensitivity vs tech comfort (diagnostic)
pv = df.pivot_table(index="Price_Sensitivity", columns="Tech_Comfort",
                    values="Likely_To_Use_Flag", aggfunc="mean") * 100
fig, ax = plt.subplots(figsize=(6.5, 4.5))
sns.heatmap(pv, annot=True, fmt=".0f", cmap="Greens", ax=ax, cbar_kws={"label": "% likely to use"})
ax.set(title="Adoption % by price sensitivity x tech comfort")
save("13_price_x_tech", f"Tech-savvy, low-price-sensitivity students adopt at ~{pv.iloc[0, -1]:.0f}% while "
     f"price-sensitive, low-tech students adopt at ~{pv.iloc[-1, 0]:.0f}%.")

# 14. Most valued feature
f = df["Most_Valued_Feature"].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(6.5, 4))
f.sort_values().plot.barh(ax=ax, color="#937860")
ax.set(title="Single most valued machine feature", xlabel="% of respondents")
save("14_valued_feature", f"Top value propositions: {f.index[0]} ({f.iloc[0]:.0f}%) and {f.index[1]} "
     f"({f.iloc[1]:.0f}%).")

(OUT / "eda_insights.txt").write_text("\n".join(insights))
print(f"\nSaved {len(insights)} charts to {OUT}")
