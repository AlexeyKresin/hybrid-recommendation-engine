# Hybrid Recommendation Engine

## Overview

This project implements a **hybrid recommendation system** that combines:

* Collaborative Filtering (user behavior)
* Content-Based Filtering (item attributes)

The system is designed to provide personalized recommendations while addressing the **cold-start problem** (new users or new items with little or no interaction data).

---

## Project Objectives

* Build a scalable recommendation pipeline
* Compare baseline recommendation approaches
* Develop a hybrid model combining multiple signals
* Analyze sparsity and its impact on recommendations
* Address cold-start scenarios

---

## Dataset

This project uses the **MovieLens 100K dataset** provided by GroupLens.

> Note: The dataset is **not included** in this repository to keep it lightweight.

---

## Project Structure

```
hybrid-recommendation-engine/
│
├── data/
│   ├── raw/              # raw dataset (not tracked by Git)
│   └── processed/        # processed data
│
├── notebooks/            # Jupyter notebooks (EDA, experiments)
│
├── src/
│   ├── data_processing/  # data loading and preprocessing
│   ├── models/           # recommendation models
│   ├── evaluation/       # evaluation metrics
│   └── utils/            # helper functions
│
├── experiments/          # model experiments
├── docs/                 # documentation and reports
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AlexeyKresin/hybrid-recommendation-engine.git
cd hybrid-recommendation-engine
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Download the dataset

Run:

```bash
python src/data_processing/download_data.py
```

This will automatically download and extract the dataset into:

```
data/raw/movielens/
```

---

## Usage

### Run Exploratory Data Analysis (EDA)

Open and run:

```
notebooks/eda_movielens.ipynb
```

This notebook includes:

* Dataset overview
* Sparsity analysis
* Rating distribution visualization

---

## Key Concepts

### Sparsity

The user-item interaction matrix is highly sparse, meaning:

* Most users rate only a small fraction of items
* This motivates the use of recommendation systems

### Cold-Start Problem

Occurs when:

* A new user has no history
* A new item has no ratings

The hybrid model aims to address this limitation.

---

## Models (Planned)

* User-based Collaborative Filtering
* Item-based Collaborative Filtering
* Content-Based Filtering
* Hybrid Recommendation Model (main contribution)

---

## Evaluation Metrics

* RMSE (Root Mean Squared Error)
* Precision@K
* Recall@K

---

## Team

* Alexey Kresin
* Naga Venkata Kolli

---

## Future Improvements

* Neural recommendation models (deep learning)
* Embedding-based approaches
* Real-time recommendation API
* Deployment (optional)

---

## License

This project is licensed under the MIT License.
