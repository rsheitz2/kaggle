# Kaggle Projects

A collection of Kaggle competition entries. Each competition lives in its own subdirectory
with its own README, feature engineering code, and model training pipeline.

## Projects

| Competition | Directory | Description |
|---|---|---|
| [Titanic](https://www.kaggle.com/c/titanic) | `titanic/` | Binary survival classification — getting started competition |

## Common conventions

- `data/` is gitignored in every project. Download competition data with the Kaggle CLI:
  ```bash
  kaggle competitions download -c <competition-name> -p <project>/data/
  unzip <project>/data/<competition-name>.zip -d <project>/data/
  ```
- Model experiments are tracked with MLflow. Run `mlflow ui` from within a project directory
  to browse results at http://127.0.0.1:5000.
- EDA notebooks are Emacs org-babel (`.org`) files with Plotly charts saved as interactive HTML.

## Setup

```bash
pip install pandas numpy scikit-learn mlflow plotly xgboost lightgbm kaggle
mkdir -p ~/.kaggle
# Place your kaggle.json API token at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
