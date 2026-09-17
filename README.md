# Crude Oil Classification

Classifies crude oil into API-gravity classes (light / medium / heavy) from laboratory measurements, using the open NOAA Oil Library. Includes a Streamlit app for single-sample prediction.

The interesting part of this project is not the accuracy number — it is what happened when the first version's accuracy was checked.

## Overview

Crude oil is graded by API gravity: above 31° is light, 22–31° medium, below 22° heavy. The grade drives refining route, transport handling and price, and it is normally read straight off a density measurement. The question here is whether the grade can be recovered from *other* laboratory properties — viscosity, pour point, flash point, SARA fractions, sulfur — for cases where density is not to hand or is disputed.

The project has two parts: a data pipeline that turns 1,458 raw assay records into a modelling table, and a set of experiments that establish what the data can and cannot support.

## What Went Wrong First, and How It Was Found

The first version of this work reported **98.4% accuracy** with a Random Forest. That number was an artefact of two defects.

### 1. Target leakage

The label is defined as `classify_oil(API)` — thresholds on API gravity. API gravity is not an independent measurement; it is computed from density by

```
API = 141.5 / SG − 131.5
```

Density was left in the feature matrix. So the model was handed a quantity from which the label follows by arithmetic.

This is measurable rather than a matter of opinion. Applying the closed-form conversion and the class thresholds to density alone — **no model, no training, no fitting of any kind** — reproduces the label on **99.35%** of the dataset. The correlation between density and API is −0.98. A Random Forest scoring 98.4% was not learning chemistry; it was recovering a formula, slightly worse than the formula itself does.

### 2. Mixed units

The NOAA records report the same property in several units, and the original extraction did not normalise them:

| Field | Units present in the raw data |
|---|---|
| density | `g/mL`, `kg/m^3`, `g/cm^3`, `g/cm³` |
| dynamic viscosity | `mPa.s`, `mPas`, `mPa s`, `cP`, `kg/(m s)` |
| pour point, flash point | `C`, `K` |
| SARA fractions | `%`, `fraction` |
| sulfur | `%` |

The resulting `density` column mixed 311 values near 0.87 with 307 values near 870 — the same physical quantity differing by a factor of 1000. Viscosity was worse: the original code converted only `Pa.s`, leaving `kg/(m s)` (the most common unit in the corpus, 508 records) unscaled by a factor of 1000, alongside three spellings of milliPascal-seconds it did not recognise. Pour and flash points mixed Celsius with Kelvin; SARA fractions mixed percentages with fractions.

Fixing the units alone moved the honest model from 79.0% to 85.5% accuracy.

### 3. Imputation before splitting

Missing values were filled with column means computed over the entire dataset, before the train/test split — so test-set statistics informed the training data. All preprocessing now happens inside an sklearn `Pipeline`, fitted on training folds only.

## Technical Approach

**Extraction.** `build_dataset.py` walks the raw JSON assays, selects the measurement nearest 15 °C for temperature-dependent properties, converts every value into one canonical unit, and drops values whose unit is unrecognised rather than passing them through unscaled. 618 of 1,458 assays carry the API, density and viscosity needed to be usable.

**Honest feature set.** Density and API are excluded. What remains: viscosity, pour point, flash point, the four SARA fractions, and sulfur.

**Missing data is the central constraint.** Coverage varies sharply across the corpus:

| Feature | Present in |
|---|---|
| viscosity | 100% |
| asphaltenes | 73% |
| pour_point | 72% |
| flash_point | 47% |
| aromatics | 40% |
| saturates | 39% |
| resins | 38% |
| sulfur | **11%** |

Sulfur is present in one assay in nine. In the original version it was mean-filled, which made it a near-constant column contributing nothing — while the app still asked users to enter it and the write-up still listed it as a predictor. It is retained here because the honest evaluation shows it carries some signal where present, but the coverage figure is published alongside it.

**Modelling.** Viscosity spans seven orders of magnitude and is log-transformed. Median imputation, scaling (for the distance- and gradient-based models) and the estimator live in one `Pipeline`. Evaluation is stratified 5-fold cross-validation plus a held-out 20% test split, with a majority-class baseline reported alongside — with a 54/23/23 class split, accuracy alone is not interpretable.

## Technologies

Python · scikit-learn · pandas · NumPy · Matplotlib · Streamlit

## Results

All figures are test-set values, reproducible from `results/`.

### With density included — the original, leaking setup

| Model | CV accuracy | Test accuracy | Macro F1 |
|---|---|---|---|
| Random Forest | 0.992 ± 0.007 | 0.976 | 0.976 |
| Logistic Regression | 0.955 ± 0.008 | 0.968 | 0.957 |
| KNN | 0.879 ± 0.031 | 0.879 | 0.855 |
| Closed-form rule, no learning | — | **0.994** | — |

The rule beats every model. That is the signature of leakage: if arithmetic on one column outperforms a trained ensemble on all columns, the ensemble is approximating the arithmetic.

### Density and API removed — the honest baseline

| Model | CV accuracy | Test accuracy | Macro F1 |
|---|---|---|---|
| **Random Forest** | **0.858 ± 0.027** | **0.855** | **0.838** |
| KNN | 0.783 ± 0.035 | 0.774 | 0.726 |
| Logistic Regression | 0.767 ± 0.025 | 0.782 | 0.707 |
| Majority-class baseline | 0.537 ± 0.003 | 0.532 | 0.232 |

**85.5% against a 53.2% baseline**, from measurements that do not define the label. Lower than the headline the leaking version produced, and the only one of the two that means anything.

### Per-class performance (Random Forest, held out)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| heavy | 0.90 | 0.90 | 0.90 | 29 |
| light | 0.87 | 0.91 | 0.89 | 66 |
| medium | 0.77 | 0.69 | 0.73 | 29 |

The extremes are separable; **medium is where the model struggles**, which is what one would expect — it is a middle band defined by two arbitrary thresholds, and samples near either boundary look like their neighbours. Of 29 medium samples, 7 are called light and 2 heavy.

### Feature ablation — each feature alone

| Feature | CV accuracy alone | Coverage |
|---|---|---|
| viscosity | 0.704 | 100% |
| saturates | 0.668 | 39% |
| resins | 0.657 | 38% |
| asphaltenes | 0.642 | 73% |
| aromatics | 0.607 | 40% |
| sulfur | 0.571 | 11% |
| flash_point | 0.563 | 47% |
| pour_point | 0.523 | 72% |

Viscosity carries most of the signal, which is physically sensible — viscosity and density both track molecular weight. Pour point alone is barely above the 0.537 baseline.

## How to Run

```bash
git clone https://github.com/VsProger/crude-oil-classification.git
cd crude-oil-classification

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Rebuild the dataset, reproduce the experiments, retrain the served model:

```bash
python build_dataset.py     # oil/*.json -> dataset.csv
python experiments.py       # leakage study, honest baseline, ablation -> results/
python train_model.py       # -> model.pkl and figures
```

Run the app:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. It accepts partial input — unavailable measurements are imputed inside the pipeline — and reports the predicted class with per-class probabilities and a count of how many measurements were actually supplied. It does **not** accept density or API gravity, for the reason above.

## Project Structure

```
├── build_dataset.py       raw assays -> dataset.csv, with unit normalisation
├── experiments.py         leakage demonstration, honest baseline, ablation
├── train_model.py         trains and persists the served model
├── features.py            feature list and transforms shared by train and serve
├── app.py                 Streamlit interface
├── model.pkl              fitted pipeline (preprocessing included)
├── EDA.ipynb              exploratory analysis
├── diplom_oil.ipynb       original thesis notebook, kept for provenance
├── oil/                   NOAA Oil Library JSON records
└── results/               metrics (CSV/JSON) and figures
```

`features.py` exists so the transformer inside `model.pkl` resolves to a stable import path — defining it in the training script would pickle a reference to `__main__` and fail to load anywhere else.

## Data

[NOAA Oil Library](https://response.restoration.noaa.gov/oil-and-chemical-spills/oil-spills/oil-library.html) — an open collection of crude oil assays published by the US National Oceanic and Atmospheric Administration. 1,458 records, of which 618 carry the fields required here.

## Future Improvements

- **Predict API gravity directly as a regression**, rather than the thresholded class. The thresholds are a convention imposed on a continuous quantity, and they are what makes the medium class hard; predicting the underlying number and banding afterwards sidesteps that and gives a calibrated error in degrees.
- **Model missingness instead of imputing it.** Whether an assay reports SARA at all is not random — it reflects which laboratory ran it and why. Missingness indicators, or gradient boosting with native NaN support, would use that signal rather than erase it.
- **Report prediction confidence in the app**, refusing to answer when too few measurements are supplied. A prediction from viscosity plus seven imputed medians should not be presented like a fully measured one.
- **Validate against a second source.** The conclusions rest on one corpus with a particular sampling bias toward spill-relevant oils; a second assay database would show whether the 85% holds.
