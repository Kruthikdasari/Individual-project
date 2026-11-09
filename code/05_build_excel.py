"""
05_build_excel.py - packs data, data dictionary and model results into one workbook
Output: ../Campus_Vending_Survey_Workbook.xlsx
"""
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
D, M = ROOT / "data", ROOT / "outputs" / "models"
OUT = ROOT / "Campus_Vending_Survey_Workbook.xlsx"

dictionary = [
    ("Respondent_ID", "ID", "Unique survey respondent code", "Identifier"),
    ("Age", "Numeric", "Age in years", "Profile"),
    ("Gender", "Categorical", "Male / Female / Other", "Profile"),
    ("Year_of_Study", "Categorical", "1st-4th Year, Postgraduate, PhD", "Profile"),
    ("Course_Stream", "Categorical", "Engineering, Management, Science, Commerce, Arts, Medical, Law, Design", "Profile"),
    ("University_Type", "Categorical", "Private / Government / Deemed / Autonomous College", "Profile"),
    ("City_Tier", "Categorical", "Tier 1 / 2 / 3 city of the campus", "Profile"),
    ("Region", "Categorical", "Region of India", "Profile"),
    ("Residence_Type", "Categorical", "Hostel / PG-Rented / Day Scholar", "Profile"),
    ("Monthly_Allowance_INR", "Numeric", "Monthly pocket money / stipend (Rs.)", "Profile"),
    ("Hours_On_Campus_Daily", "Numeric", "Hours spent on campus per day", "Habit"),
    ("Beverages_Per_Week", "Numeric", "Drinks bought per week today", "Habit"),
    ("Current_Beverage_Source", "Categorical", "Where they buy drinks now", "Habit"),
    ("Cafeteria_Wait_Minutes", "Numeric", "Usual cafeteria queue time (min)", "Pain point"),
    ("Cafeteria_Satisfaction", "Likert 1-5", "Satisfaction with cafeteria", "Pain point"),
    ("Preferred_Payment", "Categorical", "UPI / Card / Campus wallet / Cash", "Habit"),
    ("UPI_Usage_Frequency", "Categorical", "Daily / Weekly / Rarely", "Habit"),
    ("Tech_Comfort", "Likert 1-5", "Comfort with touchscreens / apps", "Attitude"),
    ("Health_Consciousness", "Likert 1-5", "How health-conscious", "Attitude"),
    ("Price_Sensitivity", "Likert 1-5", "How much price matters", "Attitude"),
    ("Importance_Convenience", "Likert 1-5", "Importance of speed / convenience", "Attitude"),
    ("Importance_Hygiene", "Likert 1-5", "Importance of hygiene", "Attitude"),
    ("Gym_or_Sports", "Yes/No", "Regular gym or sports", "Lifestyle"),
    ("Late_Night_Study", "Yes/No", "Often studies late at night", "Lifestyle"),
    ("Used_Vending_Machine_Before", "Yes/No", "Has used a vending machine", "Lifestyle"),
    ("Preferred_Location", "Categorical", "Best spot for the machine", "Preference"),
    ("Preferred_Time_Slot", "Categorical", "When they would use it most", "Preference"),
    ("Most_Valued_Feature", "Categorical", "Single most valued feature", "Preference"),
    ("Likes_<Drink> (10 cols)", "Binary 0/1", "Interest in each of 10 drinks - used for ASSOCIATION RULES", "Basket"),
    ("Subscription_Interest", "Categorical", "Interest in a monthly drink pass", "Outcome"),
    ("Interest_Level", "Likert 1-5", "Overall interest in the machine", "Outcome"),
    ("Willing_Price_Per_Drink_INR", "Numeric", "Max acceptable price per drink (Rs.)", "Outcome"),
    ("Expected_Monthly_Spend_INR", "Numeric", "Expected monthly spend (Rs.) - REGRESSION target", "Outcome"),
    ("Likely_To_Use", "Yes/No", "Would use the machine - CLASSIFICATION target", "Outcome"),
    ("--- engineered in cleaning ---", "", "", ""),
    ("Num_Beverages_Liked", "Numeric", "Count of drinks ticked", "Engineered"),
    ("Likely_To_Use_Flag / Is_Hostel / Is_Gym / Is_Late_Night / Used_Vending_Flag / Digital_Payer",
     "Binary 0/1", "0/1 versions of Yes/No fields", "Engineered"),
    ("City_Tier_Num / UPI_Freq_Num", "Ordinal", "Ordered numeric codes", "Engineered"),
    ("Spend_Share_of_Allowance_%", "Numeric", "Spend as % of allowance", "Engineered"),
    ("Allowance_Band / Wait_Band", "Categorical", "Binned versions for charts", "Engineered"),
    ("--- added by models ---", "", "", ""),
    ("Adoption_Probability", "0-1", "Best classifier's predicted probability of use", "Model output"),
    ("Cluster / Segment", "Categorical", "K-Means segment: Most / Medium / Least Probable", "Model output"),
]

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    pd.DataFrame(dictionary, columns=["Column", "Type", "Description", "Role"]).to_excel(
        xw, sheet_name="Data_Dictionary", index=False)
    pd.read_csv(D / "campus_beverage_survey_raw.csv").to_excel(xw, sheet_name="Raw_Data", index=False)
    scored = pd.read_csv(D / "campus_beverage_survey_scored.csv")
    scored.to_excel(xw, sheet_name="Clean_Scored_Data", index=False)
    pd.Series((ROOT / "outputs" / "cleaning_log.txt").read_text().splitlines(), name="Cleaning step").to_excel(
        xw, sheet_name="Cleaning_Log", index=False)
    pd.read_csv(M / "A_classification_comparison.csv").to_excel(xw, sheet_name="A_Classification", index=False)
    pd.read_csv(M / "B_cluster_profiles.csv").to_excel(xw, sheet_name="B_Segments", index=False)
    pd.read_csv(M / "C_association_rules.csv").to_excel(xw, sheet_name="C_Association_Rules", index=False)
    pd.read_csv(M / "D_regression_comparison.csv").to_excel(xw, sheet_name="D_Regression", index=False)
    pd.read_csv(M / "D_regression_coefficients.csv").rename(columns={"Unnamed: 0": "Feature"}).to_excel(
        xw, sheet_name="D_Coefficients", index=False)

# ---- live-formula pivot so the Excel "calculations" are visible -------------
wb = load_workbook(OUT)
ws = wb.create_sheet("Excel_Calculations", 1)
data = wb["Clean_Scored_Data"]
hdr = {c.value: get_column_letter(c.column) for c in data[1]}
n = data.max_row
rng = lambda col: f"Clean_Scored_Data!${hdr[col]}$2:${hdr[col]}${n}"
ws["A1"] = "Live Excel formulas over Clean_Scored_Data (recalculate automatically)"
ws["A1"].font = Font(bold=True, size=12)
ws.append([])
ws.append(["Group", "Value", "Respondents", "Likely to use (count)", "Adoption rate", "Avg monthly spend (Rs.)"])
for c in ws[3]:
    c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="2A7F62")
groups = [("Residence_Type", ["Hostel", "PG / Rented", "Day Scholar"]),
          ("City_Tier", ["Tier 1", "Tier 2", "Tier 3"]),
          ("Segment", ["Most Probable", "Medium Probable", "Least Probable"]),
          ("Preferred_Payment", ["UPI", "Debit/Credit Card", "Campus ID / Wallet", "Cash"])]
r = 4
for col, vals in groups:
    for v in vals:
        ws.cell(r, 1, col)
        ws.cell(r, 2, v)
        ws.cell(r, 3, f'=COUNTIF({rng(col)},B{r})')
        ws.cell(r, 4, f'=COUNTIFS({rng(col)},B{r},{rng("Likely_To_Use")},"Yes")')
        ws.cell(r, 5, f"=IFERROR(D{r}/C{r},0)").number_format = "0.0%"
        ws.cell(r, 6, f'=IFERROR(AVERAGEIF({rng(col)},B{r},{rng("Expected_Monthly_Spend_INR")}),0)').number_format = "#,##0"
        r += 1
r += 1
ws.cell(r, 1, "Overall").font = Font(bold=True)
ws.cell(r, 3, f"=COUNTA({rng('Respondent_ID')})")
ws.cell(r, 4, f'=COUNTIF({rng("Likely_To_Use")},"Yes")')
ws.cell(r, 5, f"=D{r}/C{r}").number_format = "0.0%"
ws.cell(r, 6, f"=AVERAGE({rng('Expected_Monthly_Spend_INR')})").number_format = "#,##0"
r += 2
ws.cell(r, 1, "Simple revenue sizing (edit the yellow inputs)").font = Font(bold=True, size=12)
inputs = [("Students on one campus", 5000), ("Adoption rate (from survey)", f"=E{r-2}"),
          ("Avg monthly spend per adopter (Rs.)",
           f'=AVERAGEIF({rng("Likely_To_Use")},"Yes",{rng("Expected_Monthly_Spend_INR")})'),
          ("Realisation factor (stated intent -> real purchase)", 0.3),
          ("Machines per campus", 8),
          ("Ingredient + packaging cost (% of revenue)", 0.45),
          ("University commission / rent (% of revenue)", 0.12),
          ("Upkeep, refilling staff, power, UPI fees per machine / month (Rs.)", 15000),
          ("Cost of one smart machine incl. install (Rs.)", 450000)]
start = r + 1
for i, (k, v) in enumerate(inputs):
    ws.cell(start + i, 1, k)
    c = ws.cell(start + i, 2, v)
    if not str(v).startswith("="):
        c.fill = PatternFill("solid", fgColor="FFF2CC")
ws.cell(start + 1, 2).number_format = "0.0%"
ws.cell(start + 2, 2).number_format = "#,##0"
for i in (5, 6):
    ws.cell(start + i, 2).number_format = "0%"
for i in (7, 8):
    ws.cell(start + i, 2).number_format = "#,##0"
o = start + len(inputs)
outs = [("Est. monthly revenue per campus (Rs.)", f"=B{start}*B{start+1}*B{start+2}*B{start+3}", "#,##0"),
        ("Est. monthly revenue per machine (Rs.)", f"=B{o}/B{start+4}", "#,##0"),
        ("Drinks per machine per day (at median price)",
         f'=B{o+1}/MEDIAN({rng("Willing_Price_Per_Drink_INR")})/30', "0"),
        ("Monthly profit per machine (Rs.)", f"=B{o+1}*(1-B{start+5}-B{start+6})-B{start+7}", "#,##0"),
        ("Payback period per machine (months)", f'=IF(B{o+3}>0,B{start+8}/B{o+3},"never")', "0.0")]
for i, (k, f, nf) in enumerate(outs):
    ws.cell(o + i, 1, k).font = Font(bold=True)
    ws.cell(o + i, 2, f).number_format = nf

for sh in wb.worksheets:
    for col in sh.columns:
        width = max(len(str(c.value)) if c.value is not None and not str(c.value).startswith("=") else 12
                    for c in col[:200])
        sh.column_dimensions[get_column_letter(col[0].column)].width = min(max(width + 2, 10), 60)
    if sh.title not in ("Excel_Calculations", "Cleaning_Log"):
        for c in sh[1]:
            c.font = Font(bold=True)
        sh.freeze_panes = "A2"
wb["Data_Dictionary"].column_dimensions["A"].width = 45
wb["Excel_Calculations"].column_dimensions["A"].width = 52
wb.save(OUT)
print("Saved", OUT)
