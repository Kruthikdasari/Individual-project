# Smart Beverage Vending for Indian Campuses: feasibility and findings

> The dataset is **synthetic**, generated to reflect realistic student behaviour. Every number below comes from that simulated survey. It shows what the analysis *would* tell you; it is not proof about real students. Validate the key assumptions with a short real survey (even 100 to 150 students) before pitching to investors or universities.

## 1. Is the idea feasible?

**Verdict: feasible, with conditions.** Demand is real, but the margins are thin, so the economics only work if you place machines carefully and keep the product range tight.

**Why the idea works**
- **Cashless is ready.** UPI is used by most Indian students every day. In the survey, about 84% prefer a cashless method (67% UPI).
- **There is a pain point.** Cafeteria queues are long at peak hours, and many campuses have nothing open late at night for hostel students. Adoption rises from about 38% (waits of 0 to 5 minutes) to about 67% (waits over 20 minutes).
- **Captive, dense demand.** Thousands of students sit in a small area for 7 to 10 hours a day, so footfall per machine is high.
- **B2B channel.** One agreement with a university unlocks many machines at once.

**Risks you must address in the report**
| Risk | Why it matters | Mitigation |
|---|---|---|
| Fresh juice and milk spoil quickly | High wastage and food-safety (FSSAI) compliance | Start with shelf-stable or chilled packs (cold coffee, iced tea, flavoured milk, lassi); add fresh juice only at high-footfall machines |
| Machine cost (assumed about ₹3 to 6 lakh each) | Payback is long | Lease the machines, or co-brand with beverage companies (they may subsidise machines) |
| Universities have exclusive canteen contracts | You may not get permission | Pitch "late-night and hostel coverage", which complements the canteen instead of competing with it; offer revenue share |
| Students are price-sensitive | Willingness to pay is about ₹42 (Tier 3) to ₹69 (Tier 1) per drink | Price by city tier; offer a monthly drink pass |
| Downtime and refilling | An empty or broken machine loses trust | IoT stock alerts and a route-based refilling team |

**Rough unit economics** (`Excel_Calculations` sheet in the workbook. These inputs are my assumptions, so edit the yellow cells): for a campus of 5,000 students with 8 machines, and assuming only 30% of stated intent turns into real purchases, revenue is about ₹70k per machine per month. That is roughly **40 drinks a day**, giving about ₹15k monthly profit and **about 30 months to pay back each machine**. This is viable but not a quick win. The analytics below show how to shorten the payback: target the right students, the right spots and the right bundles.

## 2. Why each algorithm, and what it found

### A. Classification: "Will this student use the machine?"
- **Why:** The target is Yes/No, so this is supervised classification. The model tells you which kinds of students convert, which helps you choose campuses and target marketing.
- **Leakage rule:** Only profile, habit and attitude questions are used as predictors. Interest level, spend and willingness to pay are excluded because they are the answer in disguise.
- **Results (test set, 500 students):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.744 | 0.756 | 0.679 | 0.716 | **0.823** |
| SVM | 0.718 | 0.740 | 0.624 | 0.677 | 0.799 |
| Gradient Boosting | 0.742 | 0.748 | 0.688 | 0.716 | 0.797 |
| Random Forest | 0.714 | 0.726 | 0.637 | 0.679 | 0.792 |
| KNN | 0.712 | 0.734 | 0.616 | 0.670 | 0.768 |
| Naive Bayes | 0.688 | 0.700 | 0.599 | 0.645 | 0.766 |
| Decision Tree | 0.672 | 0.671 | 0.603 | 0.636 | 0.729 |

- **Reading it:** An AUC of about 0.82 counts as "good" on the guide's scale. Random Forest scores 0.91 on training data but only 0.71 on test data, which is **overfitting**, and it is worth discussing in the report. Logistic Regression wins because the drivers are mostly additive. The top drivers are monthly allowance, number of drinks liked, cafeteria wait, price sensitivity (negative) and importance of convenience.

### B. Clustering: Most, Medium and Least Probable segments
- **Why:** There is no answer key, so we let K-Means discover natural groups. We then rank the groups by adoption to get your three tiers.
- **Features:** 7 continuous habit and attitude variables. I tested Yes/No flags first, but they hijacked the clusters: it split purely on "pays cash" and the Medium and Least groups both had about 40% adoption, which was useless. That is a good point for the report.
- **Choice of k:** The elbow bends at 3. k=2 has a slightly higher silhouette score (0.15 vs 0.13) but only gives "yes vs no". A silhouette of about 0.13 means the segments overlap, which is normal for survey data.

| Segment | Share | Adoption | Median spend per month | Who they are | What to do |
|---|---|---|---|---|---|
| **Most Probable**: "Affluent convenience seekers" | 33% | 73% | ₹780 | Allowance about ₹13.4k, low price sensitivity, like about 6 drinks | Premium SKUs, subscription pass, launch here first |
| **Medium Probable**: "Frustrated queue-haters" | 32% | 46% | ₹190 | Longest cafeteria waits (16 min), lowest cafeteria satisfaction, value convenience but price-sensitive | "Skip the queue" messaging, combo discounts, value-priced drinks |
| **Least Probable**: "Satisfied budget traditionalists" | 35% | 25% | ₹100 | Happy with the cafeteria, low convenience need, like about 3 drinks | Don't spend on acquiring them; reach them with low-price staples only |

### C. Association rule mining: "If they want 2, will they want a 3rd?"
- **Why:** The 10 drinks are a "basket". Apriori finds which drinks are wanted together, which tells you what to stock in the same machine and what to bundle.
- **How to read a rule:** *support* is how common the combination is, *confidence* is P(then | if), and *lift* is how many times more likely than chance (above 1 is a real link).
- **Results:** 206 rules (support ≥ 10%, confidence ≥ 50%, lift > 1.1).
  - 2-drink rules: Energy Drink → Cold Coffee (76% confidence), Smoothie → Fresh Juice (69%), Protein Shake → Fresh Juice (68%).
  - 3-drink rules, which answer your exact question: **{Protein Shake + Smoothie} → Fresh Juice, 80% confidence, lift 1.57**; {Coconut Water + Protein Shake} → Fresh Juice (78%); {Cold Coffee + Iced Tea} → Energy Drink (56%, lift 1.48).
  - Two clear "tribes" appear: a **health tribe** (juice, smoothie, coconut water, protein shake), which suits machines near the sports complex and gym, and a **caffeine tribe** (cold coffee, iced tea, energy drink), which suits library and late-night hostel machines.

### D. Regression: "How much will they spend?"
- **Why:** Monthly spend is a number, so we use regression. We compare Linear, Ridge (L2, shrinks coefficients) and Lasso (L1, can drop features entirely).

| Model | Test R² | 5-fold CV R² | MAE (₹) | RMSE (₹) | Features kept |
|---|---|---|---|---|---|
| Linear | 0.552 | 0.592 | 252 | 319 | 45 |
| Ridge (α≈117) | 0.562 | 0.596 | 249 | 316 | 45 |
| Lasso (α≈4.6) | 0.562 | 0.600 | 249 | 316 | **21** |

- **Reading it:** All three models perform about the same, but **Lasso matches the accuracy with fewer than half the features**. It dropped gender, year, stream, region and university type entirely. In other words, *who* the student is matters less than *how they live*. The main spend drivers (₹ per +1 standard deviation) are drinks per week (+142), allowance (+134), number of drinks liked (+98), price sensitivity (−84), hostel resident (+73), tech comfort (+65) and cafeteria satisfaction (−62).

## 3. Strategy in one paragraph
Launch at **Tier 1 and Tier 2 private or deemed universities** with large hostels. Place the first machines at **hostel blocks (late night) and the library or academic blocks**. Run **cashless only** (UPI plus campus ID). Price by tier: about ₹60 to 70 in Tier 1 and about ₹50 in Tier 2. Stock two bundles: a "Study Fuel" machine (cold coffee, iced tea, energy drink, flavoured milk) and a "Fit" machine (juice, smoothie, coconut water, protein shake). Acquire customers in the *Most Probable* segment with a monthly pass, and convert the *Medium Probable* segment with "no queue" messaging and combo offers.
