# Hybrid Recommendation Engine

#Authors:

1. Naga Venkata Kolli
2. Alexey Kresin

## Overview

This project implements a **hybrid recommendation system** that combines:

- Collaborative Filtering (user behavior)
- Content-Based Filtering (item attributes)

The system provides personalized recommendations while addressing the **cold-start problem** (new users or items with limited interaction data).

---

## Project Objectives

- Build a scalable recommendation pipeline
- Compare baseline recommendation approaches
- Develop a hybrid model combining multiple signals
- Analyze sparsity and its impact on recommendations
- Address cold-start scenarios

---

## Dataset

This project uses the **MovieLens 100K dataset** provided by GroupLens.

> Note: The dataset is **not included** in this repository.

To download the dataset:

bash
python src/data_processing/download_data.py

## Dataset location after download:

data/raw/movielens/

## Project Structure

hybrid-recommendation-engine/
│
├── data/
│   ├── raw/              # raw dataset (not tracked)
│   └── processed/        # processed data
│
├── notebooks/            # Jupyter notebooks (EDA)
│
├── src/
│   ├── data_processing/  # data loading & preprocessing
│   ├── models/           # CF, CB, Hybrid models
│   ├── evaluation/       # metrics and comparison
│   ├── utils/            # helper functions
│   ├── evaluate.py       # evaluation script
│   └── demo.py           # demo script
│
├── experiments/          # experiments and results
├── docs/                 # diagrams (draw.io), reports
│
├── requirements.txt
├── README.md
└── .gitignore

## Installation:

1. Clone repository:
git clone https://github.com/AlexeyKresin/hybrid-recommendation-engine.git
cd hybrid-recommendation-engine

2. Install Dependencies:
pip install -r requirements.txt

#How to run:

Run Evaluation:
python -m src.evaluate

Run Demo:
python -m src.demo

# What it shows:
This will:

Generate recommendations for a sample user
Show results from:
Collaborative Filtering
Content-Based Filtering
Hybrid Model

####################
#################### MODELS
####################


#Collaborative Filtering:

User-based approach
Uses cosine similarity
Mean-centered ratings
Supports:
Ranking
Rating prediction

#Content-Based Filtering

Uses item features (e.g., genres)
Builds user profiles from liked items
Computes similarity between user and items

# Hybrid Model
Combines both approaches:
hybrid_score = α * CF + (1 - α) * CB\
α controls contribution of each model
Helps mitigate cold-start issues

####################
#################### EVALUATION Metrics
####################

Evaluation Metrics
Precision@K → ranking quality
RMSE → rating prediction accuracy
Evaluation Strategy
Per-user train/test split
Simulates real-world recommendation scenario

#Key Concepts

Key Concepts
Sparsity
Most users rate only a small subset of items
Leads to sparse user-item matrix
Cold-Start Problem
New users → no interaction history
New items → no ratings

Hybrid model helps address both cases.

#Future Improvements

Item-based collaborative filtering
Matrix factorization (SVD)
Deep learning recommendation models
Feature engineering improvements
Hyperparameter tuning
Real-time recommendation API