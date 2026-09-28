# Data

The raw data is not stored in this repository. Please download it from the source.

**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository)
**Link:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
**File used:** `diabetic_data.csv` (101,766 inpatient encounters, 49 columns as loaded in the notebook)

## Download steps

1. Open the UCI link above and click **Download**.
2. Unzip the archive. It contains `diabetic_data.csv` and `IDs_mapping.csv`.
3. Put `diabetic_data.csv` in the repository root, next to `diabetes_readmission.ipynb`. The notebook reads it with `pd.read_csv("diabetic_data.csv")`.

`IDs_mapping.csv` explains the codes used in `admission_type_id`, `discharge_disposition_id` and `admission_source_id`. The notebook does not need it, but it helps when reading the feature importance chart.

## Target column

`readmitted` has three values: `NO`, `>30` (readmitted after more than 30 days) and `<30` (readmitted within 30 days). For modelling I combine `<30` and `>30` into one class, so the model predicts readmission at any time.
