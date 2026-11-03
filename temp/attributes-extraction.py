import os
import pickle
import numpy as np
import pandas as pd

RAW_INPUT_FILE = "ASRS_DBOnline.csv"
CLEAN_OUTPUT_CSV = "asrs_selected_features.csv"
CLEAN_OUTPUT_PKL = "asrs_selected_features.pkl"

selected_columns = [
    # Identifier
    "ACN",
    # Target label
    "Primary Problem",
    # Text features (NLP)
    "Narrative",
    "Narrative.1",  # Second crew member's narrative (if available)
    "Synopsis",     # Concise summary by NASA analyst
    # Causal factors (Domain context / Association mining)
    "Contributing Factors / Situations",
    "Aircraft Component",
    "Work Environment Factor",
    # Operational & environmental context
    "Flight Phase",
    "Flight Conditions",
    "Weather Elements / Visibility",
    "Operating Under FAR Part",
    "Altitude.MSL.Single Value",
    # Crew background
    "Experience",
    # Event detection
    "Detector",
    "When Detected",
    # Exploratory Data Analysis (EDA)
    "Make Model Name",
    "Anomaly",
    "Result",
]

# 1. Load raw file (skipping category header row)
print(f"Loading raw file: {RAW_INPUT_FILE}...")
try:
    df_raw = pd.read_csv(RAW_INPUT_FILE, header=1, low_memory=False, encoding="utf-8")
except UnicodeDecodeError:
    df_raw = pd.read_csv(RAW_INPUT_FILE, header=1, low_memory=False, encoding="latin1")

# 2. Filter target columns safely
missing_in_file = [col for col in selected_columns if col not in df_raw.columns]
if missing_in_file:
    print(f"Warning: The following columns were not found in the source file: {missing_in_file}")

available_cols = [col for col in selected_columns if col in df_raw.columns]
df = df_raw[available_cols].copy()

# 3. Concatenate crew narratives into a single unified text feature
narr_1 = df["Narrative"].fillna("").astype(str).str.strip()
narr_2 = (
    df["Narrative.1"].fillna("").astype(str).str.strip()
    if "Narrative.1" in df.columns
    else ""
)

df["Combined_Narrative"] = (narr_1 + " " + narr_2).str.strip()
df["Combined_Narrative"] = df["Combined_Narrative"].replace("", np.nan)

# Cast altitude to numeric, handling missing/invalid strings safely
if "Altitude.MSL.Single Value" in df.columns:
    df["Altitude.MSL.Single Value"] = pd.to_numeric(
        df["Altitude.MSL.Single Value"], errors="coerce"
    )

# 4. Export to clean CSV and binary Pickle
df.to_csv(CLEAN_OUTPUT_CSV, index=False, encoding="utf-8")
df.to_pickle(CLEAN_OUTPUT_PKL)

print(f"Clean CSV exported to: {CLEAN_OUTPUT_CSV}")
print(f"Binary DataFrame exported to: {CLEAN_OUTPUT_PKL}")

# --- SUMMARY OVERVIEW FOR LAB PRESENTATION ---
print("\n" + "=" * 70)
print("DATASET OVERVIEW FOR LAB SESSION")
print("=" * 70)
print(f"Total records: {len(df)}")
print(f"Selected columns count: {df.shape[1]}")
print(f"Clean CSV file size: {round(os.path.getsize(CLEAN_OUTPUT_CSV) / (1024 * 1024), 2)} MB")

print("\n--- TARGET CLASS DISTRIBUTION (Primary Problem) ---")
if "Primary Problem" in df.columns:
    target_summary = pd.DataFrame({
        "Count": df["Primary Problem"].value_counts(dropna=False),
        "Percentage (%)": (df["Primary Problem"].value_counts(dropna=False, normalize=True) * 100).round(2)
    })
    print(target_summary.head(10))

print("\n--- FEATURE COVERAGE & NULL SUMMARY ---")
coverage = pd.DataFrame({
    "Dtype": df.dtypes,
    "Non-Null Count": df.notna().sum(),
    "Fill Rate (%)": ((df.notna().sum() / len(df)) * 100).round(1)
})
print(coverage.to_string())
