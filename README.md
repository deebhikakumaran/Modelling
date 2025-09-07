# Modelling

This repository contains the code implementations for two key data science projects worked on during the internship at Zoho.

## 1. Persona Modelling

The **Persona Modelling** module includes code to develop comprehensive personas for the Accounts, Contacts, and Leads modules of Zoho CRM. These personas help in segmenting customers and prospects based on their engagement patterns and behaviors, thereby enabling targeted marketing and sales strategies.

### Key Features:

- Data preprocessing and feature engineering for engagement metrics (emails, calls, meetings, etc.).
- Clustering algorithms for grouping entities into distinct personas.
- Quantile binning and scoring for engagement and activity classification.
- Scoring mechanisms balancing multiple interaction dimensions.
- Modules for each CRM object — Accounts, Contacts, and Leads — with tailored persona definitions.

The persona modelling folder contains:

- Data extraction and cleaning scripts.
- Feature engineering utilities.
- Clustering and classification model implementations.
- Evaluation and visualization notebooks.

## 2. Deal Trend Modelling

The **Deal Trend Modelling** module focuses on analyzing and predicting deal outcomes and trends over time within Zoho CRM. It involves building time-series based machine learning models to score deals dynamically, enabling proactive pipeline management.

### Key Components:

- Processing of chronological activity data for deals (calls, emails, quotes, etc.).
- Sentiment, intent, and emotion scoring and trend extraction using regression techniques.
- LSTM-based sequence modelling of deal activities for trend prediction.
- Continuous trend score generation and classification into “Uptrend”, “Downtrend”, or “Stable”.
- Time-based train-test splitting ensuring robust forecasting.
- Visualization scripts for plotting deal trend evolution and highlighting new deals within pipeline context.

The deal trend modelling folder includes:

- Data ingestion and VOC extraction scripts.
- Feature extraction and trend analysis tools.
- LSTM model training and evaluation notebooks.
- Prediction serving utilities with dynamic trend scoring.
- Visualization and reporting components.

---

    These projects were developed as part of internship at Zoho, aimed at enhancing CRM analytics capabilities through advanced machine learning and time-series analysis.


---
