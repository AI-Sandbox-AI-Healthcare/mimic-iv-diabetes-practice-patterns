# README

# Purpose of the Study

This project is a feasibility study using MIMIC-IV v3.1, a de-identified electronic health record (EHR) dataset (~10GB across multiple tables), to analyze inpatient Type 2 Diabetes (T2DM) practice patterns.

We use EHR orders as a proxy for clinical care and practice patterns, and derive orderset phenotypes using:

Latent Dirichlet Allocation (LDA)
Clustering approaches

We then explore relationships between these derived phenotypes and clinical outcomes, such as length of stay (LOS), using appropriate statistical models (e.g., Gamma GLM).

# Dependencies
Install all required dependencies before running the pipeline:

> pip install -r requirements.txt

# How to Run
Run the full pipeline with:

> python3 mimic_iv_pipeline.py

# This script will:

Process raw MIMIC-IV data

Apply cohort selection and filtering

Generate order-based representations

Derive phenotypes (LDA)

Perform exploratory outcome analysis

Produce output files
