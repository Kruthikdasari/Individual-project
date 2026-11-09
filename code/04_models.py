"""
04_models.py - the four machine-learning questions

 A. CLASSIFICATION  Will this student use the machine? (Yes / No)
 B. CLUSTERING      Which natural customer segments exist? -> Most / Medium / Least probable
 C. ASSOCIATION     If a student wants drink A (and B), which drink C do they also want?
 D. REGRESSION      How much (Rs./month) will a student spend? Linear vs Ridge vs Lasso

Leakage rule: the predictors are only profile, habit and attitude questions.
Answers that ARE the outcome in disguise (Interest_Level, Subscription_Interest,
Willing_Price, Expected_Spend) are never used to predict adoption.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, r2_score, mean_absolute_error,
                             mean_squared_error, silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "models"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
df = pd.read_csv(ROOT / "data" / "campus_beverage_survey_clean.csv")
SEED = 42
summary = []


def say(msg=""):
    print(msg)
    summary.append(str(msg))


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=130)
    plt.close()


NUM = ["Age", "Monthly_Allowance_INR", "Hours_On_Campus_Daily", "Beverages_Per_Week",
       "Cafeteria_Wait_Minutes", "Cafeteria_Satisfaction", "Tech_Comfort", "Health_Consciousness",
       "Price_Sensitivity", "Importance_Convenience", "Importance_Hygiene", "UPI_Freq_Num",
       "City_Tier_Num", "Num_Beverages_Liked", "Is_Hostel", "Is_Gym", "Is_Late_Night",
       "Used_Vending_Flag", "Digital_Payer"]
CAT = ["Gender", "Year_of_Study", "Course_Stream", "University_Type", "Region",
       "Current_Beverage_Source"]
pre = ColumnTransformer([("num", StandardScaler(), NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT)])

# =====================================================================
# A. CLASSIFICATION
# =====================================================================
say("=" * 70 + "\nA. CLASSIFICATION - predicting Likely_To_Use (Yes/No)\n" + "=" * 70)
X, y = df[NUM + CAT], df["Likely_To_Use_Flag"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
say(f"Train {len(X_tr)} / Test {len(X_te)} rows; Yes-share train {y_tr.mean():.1%}, test {y_te.mean():.1%}")

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=25),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=SEED),
}
rows, probs, fitted = [], {}, {}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, model in classifiers.items():
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(X_tr, y_tr)
    pred, prob = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                 "Precision": precision_score(y_te, pred), "Recall": recall_score(y_te, pred),
                 "F1": f1_score(y_te, pred), "ROC_AUC": roc_auc_score(y_te, prob),
                 "CV_F1 (5-fold)": cv_f1,
                 "Train_Accuracy": accuracy_score(y_tr, pipe.predict(X_tr))})
    probs[name], fitted[name] = prob, pipe
res = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).round(3)
res.to_csv(OUT / "A_classification_comparison.csv", index=False)
say(res.to_string(index=False))
best = res.iloc[0]["Model"]
say(f"\nBest model by ROC-AUC: {best}")

# metric comparison chart
m = res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
m.plot.bar(figsize=(10, 5), rot=20, colormap="tab10")
plt.ylim(0.5, 1)
plt.title("Classification models - test-set performance")
plt.legend(loc="lower right", ncol=5, fontsize=8)
savefig("A1_model_comparison")

# ROC curves
plt.figure(figsize=(6.5, 5.5))
for name, prob in probs.items():
    fpr, tpr, _ = roc_curve(y_te, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curves")
plt.legend(fontsize=8)
savefig("A2_roc_curves")

# confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, (name, pipe) in zip(axes.ravel(), fitted.items()):
    cm = confusion_matrix(y_te, pipe.predict(X_te))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set(title=name, xlabel="Predicted", ylabel="Actual")
axes.ravel()[-1].axis("off")
savefig("A3_confusion_matrices")

# feature importance (Random Forest - easy to explain)
rf = fitted["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
imp.index = imp.index.str.replace("num__", "").str.replace("cat__", "")
imp = imp.sort_values().tail(15)
imp.plot.barh(figsize=(7, 6), color="#4c72b0")
plt.title("Random Forest - top 15 drivers of adoption")
savefig("A4_feature_importance")
say("Top 5 drivers (Random Forest): " + ", ".join(imp.index[::-1][:5]))

# Decision-tree rules (plain-English, great for the report)
from sklearn.tree import export_text
dt = fitted["Decision Tree"]
dt_names = [n.replace("num__", "").replace("cat__", "") for n in dt.named_steps["prep"].get_feature_names_out()]
(OUT / "A5_decision_tree_rules.txt").write_text(
    "Note: numeric features are standardised (0 = average, 1 = one std-dev above average)\n\n"
    + export_text(dt.named_steps["model"], feature_names=dt_names, max_depth=3))

# Score every respondent with the best model (used for the segment table)
df["Adoption_Probability"] = fitted[best].predict_proba(X)[:, 1].round(3)

# =====================================================================
# B. CLUSTERING
# =====================================================================
say("\n" + "=" * 70 + "\nB. CLUSTERING - K-Means customer segmentation\n" + "=" * 70)
# Only continuous habit / attitude variables. Yes/No flags (hostel, cash payer) were
# tested first but they dominate K-Means distances and split students by a single
# flag instead of by overall buying behaviour. Allowance is log-transformed because
# it is right-skewed. Flags are still used afterwards to DESCRIBE each segment.
df["Log_Allowance"] = np.log(df["Monthly_Allowance_INR"])
CLU = ["Log_Allowance", "Beverages_Per_Week", "Cafeteria_Satisfaction", "Tech_Comfort",
       "Price_Sensitivity", "Importance_Convenience", "Num_Beverages_Liked"]
Z = StandardScaler().fit_transform(df[CLU])
ks, inertia, sil = range(2, 9), [], []
for k in ks:
    km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Z, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertia, "o-")
ax[0].set(title="Elbow method", xlabel="k (number of clusters)", ylabel="Within-cluster sum of squares")
ax[1].plot(list(ks), sil, "o-", color="#c44e52")
ax[1].set(title="Silhouette score (higher = better separated)", xlabel="k")
savefig("B1_elbow_silhouette")
say("k : silhouette -> " + ", ".join(f"{k}:{s:.3f}" for k, s in zip(ks, sil)))

# k=2 has the highest silhouette but only splits "likely" vs "unlikely". k=3 is the
# elbow, scores close behind, and gives the actionable Most / Medium / Least tiers.
# Silhouette ~0.13 is typical of survey data: segments overlap (gradients, not islands).
K = 3
km = KMeans(K, n_init=50, random_state=SEED).fit(Z)
df["Cluster"] = km.labels_
rate = df.groupby("Cluster")["Likely_To_Use_Flag"].mean().sort_values(ascending=False)
label_map = dict(zip(rate.index, ["Most Probable", "Medium Probable", "Least Probable"]))
df["Segment"] = df["Cluster"].map(label_map)
order = ["Most Probable", "Medium Probable", "Least Probable"]

profile = df.groupby("Segment").agg(
    Size=("Respondent_ID", "count"), Adoption_Rate=("Likely_To_Use_Flag", "mean"),
    Avg_Adoption_Prob=("Adoption_Probability", "mean"),
    Median_Allowance=("Monthly_Allowance_INR", "median"), Drinks_Per_Week=("Beverages_Per_Week", "mean"),
    Cafe_Wait_Min=("Cafeteria_Wait_Minutes", "mean"), Cafe_Satisfaction=("Cafeteria_Satisfaction", "mean"),
    Tech_Comfort=("Tech_Comfort", "mean"), Price_Sensitivity=("Price_Sensitivity", "mean"),
    Convenience_Imp=("Importance_Convenience", "mean"), Health=("Health_Consciousness", "mean"),
    Hostel_Share=("Is_Hostel", "mean"), Late_Night_Share=("Is_Late_Night", "mean"),
    Digital_Payer_Share=("Digital_Payer", "mean"), Drinks_Liked=("Num_Beverages_Liked", "mean"),
    Median_Spend=("Expected_Monthly_Spend_INR", "median"),
    Median_WTP_Per_Drink=("Willing_Price_Per_Drink_INR", "median")).reindex(order)
profile["Share_%"] = 100 * profile["Size"] / len(df)
profile = profile.round(2)
profile.to_csv(OUT / "B_cluster_profiles.csv")
say(profile.T.to_string())
say(f"Silhouette at k=3: {silhouette_score(Z, km.labels_):.3f}")

# PCA scatter
pcs = PCA(2, random_state=SEED).fit_transform(Z)
plt.figure(figsize=(7, 5.5))
pal = {"Most Probable": "#2a7f62", "Medium Probable": "#e1a034", "Least Probable": "#c0504d"}
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Segment"], hue_order=order, palette=pal, s=14, alpha=0.7)
plt.title("Customer segments (K-Means, shown on 2 PCA axes)")
plt.xlabel("PC1")
plt.ylabel("PC2")
savefig("B2_cluster_scatter")

# segment profile heatmap (z-scores so different units are comparable)
prof_cols = ["Median_Allowance", "Drinks_Per_Week", "Cafe_Wait_Min", "Cafe_Satisfaction", "Tech_Comfort",
             "Price_Sensitivity", "Convenience_Imp", "Health", "Hostel_Share", "Late_Night_Share",
             "Digital_Payer_Share", "Drinks_Liked", "Median_Spend", "Adoption_Rate"]
pz = (profile[prof_cols] - profile[prof_cols].mean()) / profile[prof_cols].std()
plt.figure(figsize=(11, 3.6))
lbl = profile[prof_cols].apply(lambda col: col.map(lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"))
sns.heatmap(pz, annot=lbl, fmt="", cmap="RdYlGn", center=0, cbar=False)
plt.title("Segment profiles (colour = relative high/low, numbers = actual averages)")
plt.ylabel("")
savefig("B3_segment_profiles")

# beverage preference by segment
bevcols = [c for c in df.columns if c.startswith("Likes_")]
bs = df.groupby("Segment")[bevcols].mean().reindex(order) * 100
bs.columns = bs.columns.str.replace("Likes_", "").str.replace("_", " ")
plt.figure(figsize=(11, 3.3))
sns.heatmap(bs, annot=True, fmt=".0f", cmap="Purples", cbar=False)
plt.title("% of each segment interested in each drink")
plt.ylabel("")
savefig("B4_segment_beverages")

# =====================================================================
# C. ASSOCIATION RULE MINING
# =====================================================================
say("\n" + "=" * 70 + "\nC. ASSOCIATION RULES - beverage bundles (Apriori)\n" + "=" * 70)
basket = df[bevcols].astype(bool)
basket.columns = basket.columns.str.replace("Likes_", "")
freq = apriori(basket, min_support=0.10, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.1)
rules = rules[rules["confidence"] >= 0.50].copy()
fmt = lambda s: " + ".join(sorted(s))
rules["IF_student_wants"] = rules["antecedents"].apply(fmt)
rules["THEN_also_wants"] = rules["consequents"].apply(fmt)
rules["n_items"] = rules["antecedents"].apply(len) + rules["consequents"].apply(len)
cols = ["IF_student_wants", "THEN_also_wants", "support", "confidence", "lift", "n_items"]
rules = rules.sort_values("lift", ascending=False)[cols].round(3)
rules.to_csv(OUT / "C_association_rules.csv", index=False)
say(f"Frequent itemsets (support >= 10%): {len(freq)};  rules (lift>1.1, confidence>=50%): {len(rules)}")
say("\nTop 2-item rules (A -> B):")
say(rules[rules.n_items == 2].head(10).to_string(index=False))
say("\nTop 3-item rules ({A, B} -> C)  <- 'if they want 2, do they want a 3rd?'")
say(rules[rules.n_items == 3].head(10).to_string(index=False))

plt.figure(figsize=(7, 5))
sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], s=rules["n_items"] * 25,
                 cmap="viridis", alpha=0.75)
plt.colorbar(sc, label="Lift")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association rules (bubble size = items in rule)")
savefig("C1_rules_scatter")

top = rules.head(12).iloc[::-1]
plt.figure(figsize=(9, 5.5))
plt.barh(top["IF_student_wants"] + "  ->  " + top["THEN_also_wants"], top["lift"], color="#8172b2")
plt.xlabel("Lift (times more likely than chance)")
plt.title("Strongest beverage bundles")
savefig("C2_top_rules")

# lift matrix between single drinks
items = basket.columns
lift_m = pd.DataFrame(index=items, columns=items, dtype=float)
for a in items:
    for b in items:
        lift_m.loc[a, b] = np.nan if a == b else (basket[a] & basket[b]).mean() / (basket[a].mean() * basket[b].mean())
plt.figure(figsize=(9, 7))
sns.heatmap(lift_m, annot=True, fmt=".2f", cmap="RdBu_r", center=1)
plt.title("Pairwise lift between drinks (>1 = bought together more than chance)")
savefig("C3_pairwise_lift")

# =====================================================================
# D. REGRESSION
# =====================================================================
say("\n" + "=" * 70 + "\nD. REGRESSION - predicting Expected_Monthly_Spend_INR\n" + "=" * 70)
Xr, yr = df[NUM + CAT], df["Expected_Monthly_Spend_INR"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=SEED)
alphas = np.logspace(-2, 3, 60)
regs = {"Linear": LinearRegression(),
        "Ridge": RidgeCV(alphas=alphas),
        "Lasso": LassoCV(alphas=np.logspace(-1, 2.5, 60), cv=5, random_state=SEED, max_iter=20000)}
reg_rows, reg_fit = [], {}
for name, model in regs.items():
    pipe = Pipeline([("prep", pre), ("model", model)]).fit(Xr_tr, yr_tr)
    p = pipe.predict(Xr_te)
    cv_r2 = cross_val_score(pipe, Xr, yr, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2").mean()
    alpha = getattr(pipe.named_steps["model"], "alpha_", None)
    reg_rows.append({"Model": name, "R2_test": r2_score(yr_te, p), "R2_train": r2_score(yr_tr, pipe.predict(Xr_tr)),
                     "CV_R2 (5-fold)": cv_r2, "MAE_Rs": mean_absolute_error(yr_te, p),
                     "RMSE_Rs": mean_squared_error(yr_te, p) ** 0.5,
                     "Chosen_alpha": alpha, "Non_zero_coefs": int((np.abs(pipe.named_steps["model"].coef_) > 1e-6).sum())})
    reg_fit[name] = pipe
reg_res = pd.DataFrame(reg_rows).round(3)
reg_res.to_csv(OUT / "D_regression_comparison.csv", index=False)
say(reg_res.to_string(index=False))

coef_names = [n.replace("num__", "").replace("cat__", "") for n in reg_fit["Linear"].named_steps["prep"].get_feature_names_out()]
coefs = pd.DataFrame({n: reg_fit[n].named_steps["model"].coef_ for n in regs}, index=coef_names).round(2)
coefs.to_csv(OUT / "D_regression_coefficients.csv")
dropped = coefs.index[coefs["Lasso"].abs() < 1e-6].tolist()
say(f"\nLasso removed {len(dropped)} weak features: {', '.join(dropped)}")
topc = coefs.reindex(coefs["Lasso"].abs().sort_values(ascending=False).index).head(10)
say("\nTop 10 spend drivers (Lasso, Rs. change per +1 std-dev / vs base category):")
say(topc.to_string())

c15 = coefs.reindex(coefs["Linear"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]
c15.plot.barh(figsize=(8, 7), width=0.8)
plt.axvline(0, color="k", lw=0.8)
plt.title("Coefficients: Linear vs Ridge vs Lasso (top 15)")
plt.xlabel("Rs. per month")
savefig("D1_coefficients")

best_reg = reg_res.sort_values("R2_test", ascending=False).iloc[0]["Model"]
p = reg_fit[best_reg].predict(Xr_te)
plt.figure(figsize=(6, 5.5))
plt.scatter(yr_te, p, s=10, alpha=0.5)
lim = [0, max(yr_te.max(), p.max())]
plt.plot(lim, lim, "r--")
plt.xlabel("Actual spend (Rs.)")
plt.ylabel("Predicted spend (Rs.)")
plt.title(f"{best_reg} regression: actual vs predicted")
savefig("D2_actual_vs_predicted")

plt.figure(figsize=(6.5, 4))
sns.histplot(yr_te - p, bins=40, kde=True)
plt.title("Residuals (actual - predicted)")
plt.xlabel("Rs.")
savefig("D3_residuals")

# =====================================================================
df.drop(columns=["Log_Allowance"]).to_csv(ROOT / "data" / "campus_beverage_survey_scored.csv", index=False)
(OUT / "model_summary.txt").write_text("\n".join(summary))
print("\nAll model outputs saved to", OUT)
