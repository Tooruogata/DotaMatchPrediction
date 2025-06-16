# DotaMatchPrediction

This project predicts Dota 2 match outcomes using machine learning models, leveraging data from [OpenDota](https://www.opendota.com/). The workflow includes data extraction, cleaning, feature engineering, and model training (including XGBoost and linear/logistic regression).

## Project Structure

```
.
├── .devcontainer/           # VS Code dev container config
│   ├── devcontainer.json
│   └── dockerfile
├── data/
│   ├── bronze/              # Raw data (downloaded)
│   ├── silver/              # Cleaned/transformed data
│   └── gold/                # (Reserved for final datasets)
├── src/
│   ├── fetch_data.ipynb     # Data extraction from OpenDota API
│   ├── data_wrangling.ipynb # Data cleaning & feature engineering
│   └── model.ipynb          # Model training & evaluation
├── requirements.txt         # Python dependencies
├── README.md
└── .gitignore
```

## Setup & Installation

1. **Clone the repository:**


   ```sh
   git clone https://github.com/Tooruogata/DotaMatchPrediction.git
   cd DotaMatchPrediction
   ```

2. **Set up docker image:**

Set up the docker image:
   ```sh
   docker build -t DotaMatchPrediction:latest -f .devcontainer/Dockerfile .
   docker run -dit --name DotaMatchPrediction -v "$repopath:/workspace" -w /workspace DotaMatchPrediction:latest
   ```


## Data Pipeline

1. **Data Extraction:**  
   Use `src/fetch_data.ipynb` to download match data from OpenDota using SQL queries via their API. Data is saved in `data/bronze/all_dota_matches.csv`.

2. **Data Wrangling:**  
   Clean and transform the raw data using `src/data_wrangling.ipynb`. This notebook performs feature engineering and outputs a processed dataset to `data/silver/data_transformed.csv`.

## Modeling

- **Model Training & Evaluation (in progress):**  
  `src/model.ipynb` contains code for:
  - Linear Regression
  - Logistic Regression
  - XGBoost Classifier (with grid search)
  - Model evaluation (accuracy, confusion matrix, classification report, plots)

## Usage

1. **Fetch Data:**  
   Run all cells in `src/fetch_data.ipynb` to download and save raw match data.

2. **Process Data:**  
   Run all cells in `src/data_wrangling.ipynb` to clean and engineer features.

3. **Train Models:**  
   Run all cells in `src/model.ipynb` to train and evaluate models.