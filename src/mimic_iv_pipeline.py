"""
mimic_iv_pipeline.py
"""

import os
import json
import re
from datetime import datetime
import pandas as pd
import polars as pl
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

output_file = open("../results/mimic_iv_output.txt", "w")
sys.stdout = output_file

# ---------------------------------------------------------------------
# 1. Load Tables (from CSVs instead of database)
# ---------------------------------------------------------------------
print("Loading tables from CSVs...")

def load_csv(
    file: str,
    usecols=None,
    lazy=False,
    schema_overrides=None,
):
    if lazy:
        df = pl.scan_csv(
            file,
            infer_schema_length=50_000,
            schema_overrides=schema_overrides,
            low_memory=True,
        )

        # SAFE for LazyFrame
        cols = df.collect_schema().names()
        df = df.rename({c: c.lower() for c in cols})

    else:
        df = pl.read_csv(
            file,
            infer_schema_length=50_000,
            schema_overrides=schema_overrides,
            low_memory=True,
        )
        df = df.rename({c: c.lower() for c in df.columns})

    if usecols is not None:
        usecols = [c.lower() for c in usecols]
        df = df.select(usecols)

    return df

try:
    # ---- smaller / medium tables (eager)
    admissions_df = load_csv("../mimic-iv-data/admissions.csv")
    d_icd_diagnoses_df = load_csv("../mimic-iv-data/d_icd_diagnoses.csv",
        schema_overrides={"icd_code": pl.Utf8, "icd_version": pl.Int8,})
    d_labitems_df = load_csv("../mimic-iv-data/d_labitems.csv")
    diagnoses_icd_df = load_csv("../mimic-iv-data/diagnoses_icd.csv",
        schema_overrides={"icd_code": pl.Utf8, "icd_version": pl.Int8,})
    patients_df = load_csv("../mimic-iv-data/patients.csv")
    transfers_df = load_csv("../mimic-iv-data/transfers.csv")

    # ---- HUGE tables (lazy)
    labevents_df = load_csv("../mimic-iv-data/labevents.csv", lazy=True)
    pharmacy_df = load_csv("../mimic-iv-data/pharmacy.csv", lazy=True)
    prescriptions_df = load_csv("../mimic-iv-data/prescriptions.csv", lazy=True)
    poe_df = load_csv("../mimic-iv-data/poe.csv", lazy=True)
    poe_detail_df = load_csv("../mimic-iv-data/poe_detail.csv", lazy=True)

except Exception as e:
    print("Error loading tables:", e)
    raise

# ---------------------------
# 2. Print table sizes
# ---------------------------
print(
    "Loaded:\n",
    f"admissions={admissions_df.height:,}\n",
    f"d_icd_diagnoses={d_icd_diagnoses_df.height:,}\n",
    f"d_labitems={d_labitems_df.height:,}\n",
    f"diagnoses_icd={diagnoses_icd_df.height:,}\n",
    f"patients={patients_df.height:,}\n",
    f"transfers={transfers_df.height:,}\n",
    f"labevents=LAZY\n",
    f"pharmacy=LAZY\n",
    f"poe=LAZY\n",
    f"poe_detail=LAZY\n",
    f"prescriptions=LAZY"
)

def show_head(name, df, n=5):
    print(f"\n=== {name} (first {n} rows) ===")
    if isinstance(df, pl.LazyFrame):
        print(df.head(n).collect())
    else:
        print(df.head(n))

# Show first 5 rows of all tables
#show_head("admissions", admissions_df)

# ---------------------------
# 3. Apply filters
# ---------------------------

patients_df = patients_df.filter(
    (pl.col("anchor_age") >= 18) &
    (pl.col("anchor_age") <= 75) &
    (pl.col("anchor_year_group") == "2017 - 2019") &
    (pl.col("dod").is_null())
)

diagnoses_icd_df = diagnoses_icd_df.filter(
    (pl.col("icd_version") == 10) &
    (pl.col("seq_num").is_in([1, 2, 3])) &
    (pl.col("icd_code").str.starts_with("E11"))
)

d_icd_diagnoses_df = d_icd_diagnoses_df.filter(
    pl.col("icd_version") == 10
)

admissions_df = (
    admissions_df
    # Drop rows with null admittime or dischtime
    .filter(
        pl.col("admittime").is_not_null() &
        pl.col("dischtime").is_not_null()
    )
    # Apply row-level inclusion criteria
    .filter(
        # Patient is alive (no in-hospital death)
        (
            pl.col("deathtime").is_null() |
            (pl.col("hospital_expire_flag") == 0)
        )
        &
        # Discharge after admission
        (pl.col("dischtime") > pl.col("admittime"))
    )
    # Include only certain admission types
    .filter(
        pl.col("admission_type").is_in(["URGENT", "EW EMER.", "DIRECT EMER.", "OBSERVATION ADMIT"])
    )
)

# Length of Stay filter
min_los_hours = 0.0   # set to 24.0 if you want to enable

admissions_df = (
    admissions_df
    .with_columns([
        (
            (pl.col("dischtime").str.to_datetime() -
             pl.col("admittime").str.to_datetime())
            .dt.total_hours()
        ).alias("los_hours")
    ])
    .filter(pl.col("los_hours") >= min_los_hours)
)

# Exclude AMA
#admissions_df = admissions_df.filter(
#    ~pl.col("discharge_location")
#      .str.to_uppercase()
#      .str.contains("AMA|AGAINST MEDICAL ADVICE")
#)

# Filter rows where 'careunit' contains exactly 'ICU' or 'CCU'
transfers_df_icu_or_ccu = (
    transfers_df
    .filter(
        pl.col("careunit")
        .fill_null("")
        .str.contains(r"ICU|CCU") # Only ICU and CCU
    )
)

# Keep only required columns (still lazy)
pharmacy_df = pharmacy_df.select([
    "hadm_id",
    "pharmacy_id",
    "poe_id",
    "medication"
])

print(
    "\nTable rows after filtering:\n",
    f"admissions={admissions_df.height:,}\n",
    f"d_icd_diagnoses={d_icd_diagnoses_df.height:,}\n",
    f"d_labitems={d_labitems_df.height:,}\n",
    f"diagnoses_icd={diagnoses_icd_df.height:,}\n",
    f"patients={patients_df.height:,}\n",
    f"transfers={transfers_df.height:,}\n",
    f"labevents=LAZY\n",
    f"pharmacy=LAZY\n",
    f"poe=LAZY\n",
    f"poe_detail=LAZY\n",
    f"prescriptions=LAZY"
)

# -----------------------------------------
# 4.1. Build diabetes admission cohort
# -----------------------------------------

# Unique hospital admission IDs that have any diabetes ICD code.
diabetes_hadm_ids = (
    diagnoses_icd_df
    .select("hadm_id")
    .unique()
)

# Only includes admissions with: a diabetes diagnosis, a valid patient record
cohort_df = (
    admissions_df
    .join(diabetes_hadm_ids, on="hadm_id", how="inner")
    #.join(diabetes_hadm_ids, on="hadm_id", how="inner")
    .join(
        patients_df.select(["subject_id", "gender", "anchor_age", "anchor_year_group"]),
        on="subject_id",
        how="inner"
    )
)

columns_to_drop = [
    "admit_provider_id",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    #"race",
    "edregtime",
    "edouttime"
]

cohort_df = cohort_df.drop(columns_to_drop)

# Create a dataframe of unique admissions that went to ICU/CCU
icu_admissions_df = (
    transfers_df_icu_or_ccu
    .select(["subject_id", "hadm_id"])
    .unique()
)

# Keep only admissions that had ICU/MICU exposure
cohort_with_icu_df = (
    cohort_df
    .join(icu_admissions_df, on=["subject_id", "hadm_id"], how="inner")
)

print("\nDiabetes admissions (cohort_df):", cohort_df.height)
print("\nDiabetes admissions with ICU/CCU (cohort_with_icu_df):", cohort_with_icu_df.height)

# -----------------------------------------
# 4.2. Create combined column in d_labitems
# -----------------------------------------

d_labitems_combined_df = (
    d_labitems_df
    .with_columns(
        pl.concat_str(
            ["category", "fluid", "label"],
            separator="_"
        ).alias("category_fluid_label")
    )
    .select(["itemid", "category_fluid_label"])
)

# -----------------------------------------
# 4.3. Merge with labevents
# -----------------------------------------

labevents_enriched_df = (
    labevents_df
    .join(
        d_labitems_combined_df.lazy(),
        on="itemid",
        how="left"
    )
    .select([
        "labevent_id",
        "subject_id",
        "hadm_id",
        "itemid",
        "category_fluid_label"
    ])
)

labevents_agg_df = (
    labevents_enriched_df
    .group_by(["subject_id", "hadm_id"])
    .agg(
        pl.col("category_fluid_label")
        .unique()
        .implode()
        .list.join("&")
        .alias("lab_list")
    )
)

# -----------------------------------------
# 4.4. Prepare prescriptions table
# -----------------------------------------

prescriptions_subset_df = (
    prescriptions_df
    .select([
        "subject_id",
        "hadm_id",
        "poe_id",
        "drug"
    ])
)

prescriptions_agg_df = (
    prescriptions_subset_df
    .group_by(["subject_id", "hadm_id", "poe_id"])
    .agg(
        pl.col("drug")
        .unique()
        .implode()
        .list.join("&")
        .alias("drug_list")
    )
)

poe_full_df = (
    poe_df
    .join(
        labevents_agg_df,
        on=["subject_id", "hadm_id"],
        how="left"
    )
    .join(
        prescriptions_agg_df,
        on=["subject_id", "hadm_id", "poe_id"],
        how="left"
    )
)

# -----------------------------------------
# 4.5. Merge POE with poe_detail, transfers and pharmacy
# -----------------------------------------

poe_detail_agg = (
    poe_detail_df
    .group_by("poe_id")
    .agg([
        pl.col("field_name").implode().list.join(",").alias("field_names"),
        pl.col("field_value").implode().list.join(",").alias("field_values")
    ])
)

transfers_agg = (
    transfers_df
    .group_by(["subject_id", "hadm_id"])
    .agg([
        pl.col("careunit")
        .unique()
        .implode()
        .list.join(",")
        .alias("careunit_list")
    ])
)

# Make transfers_agg lazy
transfers_agg_lazy = transfers_agg.lazy()

pharmacy_agg = (
    pharmacy_df
    .group_by(["poe_id", "hadm_id"])
    .agg([
        pl.col("medication").unique().implode().list.join("_").alias("medication"),
        pl.len().alias("medication_count")
    ])
)

poe_enriched_df = (
    poe_full_df
    .join(poe_detail_agg, on="poe_id", how="left")
    .join(transfers_agg_lazy, on=["subject_id", "hadm_id"], how="left")
    .join(pharmacy_agg, on=["poe_id", "hadm_id"], how="left")
    .select([
        "poe_id", "subject_id", "hadm_id", 
        "ordertime", "order_type", "order_subtype",
        "field_names", "field_values", 
        "careunit_list", "medication", "medication_count", 
        "drug_list", "lab_list"
    ])
)

# -----------------------------------------
# 5. Restrict POE to cohort admissions
# -----------------------------------------

poe_cohort_df = (
    poe_enriched_df.with_columns([pl.col("ordertime").str.to_datetime()])
    .join(
        cohort_with_icu_df.select(["subject_id", "hadm_id", "admittime", "dischtime", "gender", "race", "anchor_age", "anchor_year_group"])
                 .with_columns([
                    pl.col("admittime").str.to_datetime(),
                    pl.col("dischtime").str.to_datetime()])
                 .lazy(),
        on=["subject_id", "hadm_id"],
        how="inner"
    )
    # Ensures the order actually occurred during the hospital stay, not before or after
    .filter(
        (pl.col("ordertime") >= pl.col("admittime")) &
        (pl.col("ordertime") <= pl.col("dischtime")) #&
        # Keep only orders within first 24 hours after admission
        #(pl.col("ordertime") <= pl.col("admittime") + pl.duration(hours=24))
    )
)

poe_cohort_df = poe_cohort_df.collect()

print("\nPOE orders for diabetes cohort admissions:", poe_cohort_df.height)

poe_cohort_df.write_csv("poe_cohort_diabetes.csv")
print("Saved poe_cohort_df to poe_cohort_diabetes.csv")

print("\n===== Cohort Overview =====")
n_rows = poe_cohort_df.height
n_patients = poe_cohort_df.select("subject_id").n_unique()
n_admissions = poe_cohort_df.select("hadm_id").n_unique()
print(f"Total POE rows: {n_rows:,}")
print(f"Unique patients: {n_patients:,}")
print(f"Unique admissions: {n_admissions:,}")

missing_admissions_count = (
    cohort_with_icu_df.select("hadm_id").unique()
    .filter(
        ~pl.col("hadm_id").is_in(poe_cohort_df.select("hadm_id").unique().to_series())
    )
    .height
)

print("Admissions with ICU but no POE orders:", missing_admissions_count)

patients_unique_df = (
    poe_cohort_df
    .select(["subject_id", "gender", "race", "anchor_age"])
    .unique("subject_id")
)

print("\n===== Gender Distribution =====")
gender_dist = (
    patients_unique_df
    .group_by("gender")
    .len()
    .with_columns(
        (pl.col("len") / pl.sum("len") * 100).round(2).alias("percent")
    )
)
print(gender_dist)

print("\n===== Race Distribution =====")
race_dist = (
    patients_unique_df
    .group_by("race")
    .len()
    .sort("len", descending=True)
    .with_columns(
        (pl.col("len") / pl.sum("len") * 100).round(2).alias("percent")
    )
)
print(race_dist)

print("\n===== Age Statistics =====")
age_stats = patients_unique_df.select([
    pl.col("anchor_age").mean().alias("mean_age"),
    pl.col("anchor_age").median().alias("median_age"),
    pl.col("anchor_age").min().alias("min_age"),
    pl.col("anchor_age").max().alias("max_age"),
    pl.col("anchor_age").std().alias("std_age")
])
print(age_stats)

print("\n===== Orders per Admission =====")
orders_per_adm = (
    poe_cohort_df
    .group_by("hadm_id")
    .len()
)
order_stats = orders_per_adm.select([
    pl.col("len").mean().alias("mean_orders_per_admission"),
    pl.col("len").median().alias("median_orders_per_admission"),
    pl.col("len").min().alias("min_orders"),
    pl.col("len").max().alias("max_orders"),
    pl.col("len").std().alias("std_orders")
])
print(order_stats)

print("\n===== Order Type Distribution =====")
order_type_dist = (
    poe_cohort_df
    .group_by("order_type")
    .len()
    .sort("len", descending=True)
)
print(order_type_dist)

# -----------------------------------------
# 6. Create order tokens
# -----------------------------------------

# Convert array columns to single space-separated strings
poe_cohort_df = poe_cohort_df.with_columns([
    pl.col("medication")
      .str.replace_all(" ", "_")
      .alias("medication"),

    pl.col("field_values")
      .str.replace_all(" ", "_")
      .alias("field_values")
])

poe_tokens_df = (
    poe_cohort_df
    .filter(pl.col("order_type").str.to_lowercase() != "lab")   # skip labs
    .with_columns(
        # Conditionally create order_token
        pl.when(pl.col("order_type").str.to_lowercase() == "medications")
          .then(
              pl.concat_str(
                  [
                      pl.col("order_type").fill_null("UNK"),
                      pl.col("medication").fill_null("UNK")  # add medication here
                  ],
                  separator="_"
              )
          )
          .when(pl.col("order_type").is_in(["General Care", "ADT orders", "IV therapy"]))
            .then(
                pl.concat_str(
                    [
                        pl.col("order_type").fill_null("UNK"),
                        pl.col("field_values").fill_null("UNK")
                    ],
                    separator="_"
              )
          )
          .otherwise(
              pl.concat_str(
                  [
                      pl.col("order_type").fill_null("UNK"),
                      pl.col("order_subtype").fill_null("UNK")
                  ],
                  separator="_"
              )
          )
          .alias("order_token")
    )
    .select(["hadm_id", "order_token"])
)

lab_tokens_df = (
    poe_cohort_df
    .filter(pl.col("order_type").str.to_lowercase() == "lab")
    .select(["subject_id", "hadm_id", "lab_list"])
    .unique(["subject_id", "hadm_id"])   # ensures labs generated once per admission
    .with_columns(
        pl.col("lab_list")
        .fill_null("")
        .str.split("&")
        .alias("lab_split")
    )
    .explode("lab_split")
    .with_columns(
        pl.concat_str(
            [
                pl.lit("lab"),
                pl.col("lab_split")
            ],
            separator="_"
        ).alias("order_token")
    )
    .select(["hadm_id", "order_token"])
)

poe_tokens_df = pl.concat(
    [poe_tokens_df, lab_tokens_df],
    how="vertical"
)

# How many unique tokens?
print('\nUnique tokens: ', poe_tokens_df.select("order_token").n_unique())

# Most common tokens
print('\nMost common tokens: ', poe_tokens_df.group_by("order_token").len().sort("len", descending=True))

# -----------------------------------------
# 7. Aggregate tokens per admission
# -----------------------------------------

documents_df = (
    poe_tokens_df
    .group_by("hadm_id")
    .agg(
        pl.col("order_token").alias("tokens")
    )
)

print(f"\nNumber of admissions (rows) in documents_df: {documents_df.height}")

documents_df_pd = documents_df.to_pandas()
documents_df_pd["n_orders"] = documents_df_pd["tokens"].apply(len)

# Summary stats
print("\nOrders per admission summary:")
print(documents_df_pd["n_orders"].agg(["min", "max", "mean", "median", "std"]))

# -----------------------------------------
# 8. Prepare text for LDA
# -----------------------------------------

documents_df = documents_df.with_columns(
    pl.col("tokens")
    .list.eval(pl.element().str.replace_all(r"[^\w__]", "", literal=False))  # keep letters, digits, __
    .list.join(" ")
    .alias("doc_text")
)

doc_texts = documents_df["doc_text"].to_list()
hadm_ids  = documents_df["hadm_id"].to_list()

vectorizer = CountVectorizer(
    tokenizer=lambda x: x.split(),  # split only on spaces
    preprocessor=None,               
    min_df=5,                       # drop very rare orders
    max_df=0.95                     # drop tokens appearing in >=95% of admissions
)

X = vectorizer.fit_transform(doc_texts)

n_docs, n_terms = X.shape
print("\nDocument-Term Matrix (DTM) Summary:")
print(f"Number of admissions (documents): {n_docs:,}")
print(f"Number of unique order tokens (features): {n_terms:,}")
print(f"Total matrix size: {n_docs:,} x {n_terms:,}")

tokens = vectorizer.get_feature_names_out()

# -----------------------------------------
# 9. Find best k for LDA
# -----------------------------------------

topic_range = list(range(3, 21))  # try different range for topics

results = []

for k in topic_range:
    print(f"Fitting LDA with k={k}...")
    
    lda = LatentDirichletAllocation(
        n_components=k,
        max_iter=20,
        learning_method="batch",
        random_state=42,
        n_jobs=-1
    )
    
    lda.fit(X)
    
    perplexity = lda.perplexity(X)
    log_likelihood = lda.score(X)
    
    results.append({
        "k": k,
        "perplexity": perplexity,
        "log_likelihood": log_likelihood
    })

results_df = pd.DataFrame(results)
print(results_df)

# -----------------------------------------
# 10. Plot results to find best k value for lda
# -----------------------------------------

plt.figure()
plt.plot(results_df["k"], results_df["perplexity"])
plt.xlabel("Number of Topics (k)")
plt.ylabel("Perplexity (lower is better)")
plt.title("LDA Perplexity vs Number of Topics")
plt.savefig("../results/lda_perplexity_vs_k.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: results/lda_perplexity_vs_k.png")

plt.figure()
plt.plot(results_df["k"], results_df["log_likelihood"])
plt.xlabel("Number of Topics (k)")
plt.ylabel("Log Likelihood (higher is better)")
plt.title("LDA Log Likelihood vs Number of Topics")
plt.savefig("../results/lda_loglikelihood_vs_k.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: results/lda_loglikelihood_vs_k.png")

# -----------------------------------------
# 11. Fit LDA
# -----------------------------------------

# Select best k value based on perplexity & log_likelihood
n_topics = 5

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=10, # use 20
    learning_method="batch",
    random_state=42,
)

lda_topic_matrix = lda.fit_transform(X)

print("\nLDA output shape:", lda_topic_matrix.shape)

# -----------------------------------------
# 11. Attach topics to admissions
# -----------------------------------------

topic_cols = [f"topic_{k}" for k in range(n_topics)]

lda_df = pl.DataFrame(
    lda_topic_matrix,
    schema=topic_cols
).with_columns(
    pl.Series("hadm_id", hadm_ids)
)

admissions_with_topics_df = (
    cohort_df
    .join(lda_df, on="hadm_id", how="inner")
)

# -----------------------------------------
# 12. Assign dominant topic per admission
# -----------------------------------------

# Get dominant topic index for each admission
dominant_topics = np.argmax(lda_topic_matrix, axis=1)

# Create dataframe with hadm_id and assigned topic
topic_assignment_df = pd.DataFrame({
    "hadm_id": hadm_ids,
    "dominant_topic": dominant_topics
})

# Count admissions per topic
topic_counts = (
    topic_assignment_df["dominant_topic"]
    .value_counts()
    .sort_index()
)

print("\nAdmissions per Topic:")
for topic, count in topic_counts.items():
    print(f"Topic {topic}: {count} admissions")
    
    
# -----------------------------------------
# 13. Inspect topic meanings
# -----------------------------------------

feature_names = vectorizer.get_feature_names_out()

def print_topics(model, feature_names, n_top_words=5):
    for topic_idx, topic in enumerate(model.components_):
        top_features = [
            feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]
        ]
        print(f"\nTopic {topic_idx}:")
        print(", ".join(top_features))

print_topics(lda, feature_names)

# -----------------------------------------
# 14. Build Final Feature Matrix
# -----------------------------------------

feature_cols = (
    [f"topic_{i}" for i in range(n_topics)]
)

features_df = admissions_with_topics_df.select(["hadm_id"] + feature_cols)

X_features = features_df.select(feature_cols).to_numpy()
print("\nFeature matrix shape:", X_features.shape)

# -----------------------------------------
# 15. Standardize Features
# -----------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# -----------------------------------------
# 16. K-Means clustering on latent embeddings
# -----------------------------------------

# Choose number of clusters
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Attach cluster labels to admissions
cluster_df = pl.DataFrame({
    "hadm_id": features_df["hadm_id"].to_list(),
    "cluster": clusters
})

admissions_clustered_df = admissions_with_topics_df.join(cluster_df, on="hadm_id")
print(admissions_clustered_df.select(["hadm_id", "cluster"] + [f"topic_{i}" for i in range(n_topics)]).head())

cluster_topic_summary = (
    admissions_clustered_df
    .group_by("cluster")
    .agg([pl.col(f"topic_{i}").mean() for i in range(n_topics)])
    .sort("cluster")
)

print(cluster_topic_summary)

print(admissions_clustered_df.group_by("cluster").len())

# -----------------------------------------
# 17. Analyze relationship with length of stay (LOS)
# -----------------------------------------

admissions_clustered_df = admissions_clustered_df.collect() \
    if isinstance(admissions_clustered_df, pl.LazyFrame) \
    else admissions_clustered_df

# Compute LOS in days
admissions_clustered_df = admissions_clustered_df.with_columns(
    (
        (
            pl.col("dischtime").str.to_datetime() 
            - pl.col("admittime").str.to_datetime()
        )
        .dt.total_seconds() / 86400
    ).alias("los_days")
)

# Inspect LOS by cluster
cluster_summary = admissions_clustered_df.group_by("cluster").agg([
    pl.col("los_days").mean().alias("los_mean"),
    pl.col("los_days").std().alias("los_std")
])

print('Cluster Summary:', cluster_summary)

# Convert to pandas
df_glm = admissions_clustered_df.to_pandas()

# Use cluster as categorical predictor
df_glm["cluster"] = df_glm["cluster"].astype("category")

print(df_glm["los_days"].describe())
print((df_glm["los_days"] <= 0).sum())

# Gamma GLM with log link
gamma_model = smf.glm("los_days ~ cluster", data=df_glm,
                      family=sm.families.Gamma(sm.families.links.log())).fit()
print(gamma_model.summary())

output_file.close()

