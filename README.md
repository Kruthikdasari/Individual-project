# Campus Beverage Vending: survey analytics project

## Folder map
```
data/
  campus_beverage_survey_raw.csv      2,030 rows, raw survey with deliberate mess (for the cleaning marks)
  campus_beverage_survey_clean.csv    2,000 rows, cleaned and feature-engineered
  campus_beverage_survey_scored.csv   clean data plus adoption probability and segment
code/
  01_generate_survey_data.py          builds the synthetic survey (seed 42, reproducible)
  02_clean_and_transform.py           duplicates, labels, impossible values, outliers, imputation, new features
  03_eda.py                           14 descriptive and diagnostic charts plus auto-written insights
  04_models.py                        classification, clustering, association rules, regression
  05_build_excel.py                   builds the Excel workbook
  app.py                              Streamlit dashboard (9 tabs: overview, cleaning, EDA, 4 models, predictor and business case)
outputs/
  cleaning_log.txt                    every cleaning step with counts
  eda/                                charts plus eda_insights.txt
  models/                             charts, comparison tables (CSV), model_summary.txt
Campus_Vending_Survey_Workbook.xlsx   dictionary, raw and clean data, live Excel formulas, all results
FINDINGS.md                           feasibility verdict and interpretation of every algorithm
```

## Run it
```bash
pip install pandas numpy scikit-learn mlxtend matplotlib seaborn openpyxl streamlit plotly
```
```bash
cd code && python 01_generate_survey_data.py && python 02_clean_and_transform.py && python 03_eda.py && python 04_models.py && python 05_build_excel.py
```
```bash
streamlit run code/app.py
```

## Live dashboard (Streamlit Community Cloud)
1. Push this folder to a **public** GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub, then click **Create app** and choose **Deploy a public app from GitHub**.
3. Repository: `<your-username>/<repo-name>`, Branch: `main`, Main file path: `code/app.py`.
4. Click **Deploy**. The app installs `requirements.txt` and goes live at `https://<name>.streamlit.app`.
