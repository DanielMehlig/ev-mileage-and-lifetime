---
language:
- en
license: bsd-3-clause
library_name: pytorch
tags:
- research
- pytorch
- transformer
- tabular-sequence-modeling
- forecasting
- electric-vehicles
- vehicle-survival
- scrappage-prediction
- lifecycle-emissions
- united-kingdom
pretty_name: MOT Transformer for Vehicle Mileage and Scrappage Forecasting
---

# MOT Transformer for Vehicle Mileage and Scrappage Forecasting

## Model Overview

This repository accompanies the study *Real-world mileage and survival of electric vehicles in Britain and implications for lifecycle emissions*. The model is a PyTorch transformer that learns from longitudinal UK MOT test records to forecast:

1. future annual vehicle mileage
2. the probability that a vehicle is scrapped before its next observed test

These forecasts are then used in downstream lifecycle greenhouse gas emissions analysis for battery electric vehicles (BEVs) and comparator powertrains.

This is a research model intended to reproduce the paper’s analytical workflow. It is not packaged here as a general-purpose production inference service, and it should not be used for individual-level regulatory, insurance, credit, or safety decisions.

## Model Details

### Model type

The core model is a transformer encoder for tabular sequential data with two prediction heads:

- a regression head for next-period mileage
- a binary classification head for next-period scrappage probability

### Architecture

The implementation is defined in `VehicleTransformer` in the project codebase. At a high level, it uses:

- a linear embedding layer for tabular input features
- positional encoding
- a stacked transformer encoder
- a mileage prediction head
- a scrappage prediction head with sigmoid output

The training notebook configures the model with:

- `d_model = 128`
- `nhead = 8`
- `num_layers = 6`
- `dim_feedforward = 256`
- `num_epochs = 5`

The training code uses Adam optimization, a ReduceLROnPlateau scheduler, dropout in the output heads, and gradient clipping.

### Inputs

The training pipeline builds sequences from repeated MOT observations for each vehicle. The modeling code uses these core features:

- `fuel_type`
- `last_test`
- `mileage_per_year`
- `test_mileage`
- `age_year`
- `time_between_tests`

Sequences are padded to a fixed maximum length. The code constructs training sequences from at least three prior observations, with the next observation used as the prediction target.

### Outputs

For each vehicle sequence, the model predicts:

- next observed annual mileage
- probability of scrappage before the next test cycle

In the project workflow, these predictions are used iteratively to simulate future vehicle trajectories and support emissions calculations.

## Intended Use

### Primary intended use

This model is intended for:

- research on real-world vehicle usage and survival
- analysis of fleet-level mileage and scrappage patterns in Britain
- scenario generation for lifecycle emissions analysis
- reproducing or extending the results from the associated paper

### Out-of-scope use

This model should not be used for:

- predicting outcomes for a specific owner or vehicle in a commercial setting
- regulatory enforcement
- insurance pricing
- warranty adjudication
- consumer credit or financing decisions
- any high-stakes decision about an individual person

## Training Data

The repository documents a multi-source data workflow based on public datasets. The main sources are:

- UK MOT test results
- Vehicle Certification Agency (VCA) data
- European Environment Agency (EEA) data
- EV Database specifications
- electricity generation mix data
- DfT vehicle licensing statistics (including survival-rate context)
- global and UK BEV sales data

The README indicates that MOT data from 2005 onward are used, with repository folders for annual MOT result files through 2024. The project workflow merges MOT records with vehicle specifications and emissions-related attributes.

### Data characteristics

The data are observational rather than experimental. As a result:

- vehicle usage is not randomly assigned
- later BEV cohorts may still be immature in lifetime terms
- missingness and source harmonization matter
- the observed MOT population reflects the British vehicle fleet and MOT regime, not a global vehicle population

## Preprocessing

The preprocessing workflow performs tasks such as:

- merging MOT records with VCA, EEA, and EV specification sources
- filtering the MOT data using consistent study criteria
- generating sequential examples for transformer training
- encoding categorical variables
- standardizing numerical variables

The data processing notebook also prepares downstream attributes used in lifecycle analysis, including battery capacity, fuel efficiency, vehicle mass, and CO2 intensity fields.

## Training Procedure

The training code uses:

- a vehicle-level train/test split
- label encoding for categorical variables
- standard scaling for numerical variables
- MSE loss for mileage prediction
- binary cross-entropy loss for scrappage prediction
- additional weighting to address class imbalance in scrappage
- a combined optimization objective

A validation-based threshold search is used for the scrappage output. The notebook output indicates a best threshold near `0.229`, and the simulation workflow uses a threshold of approximately `0.22` for iterative future-state generation.

## Evaluation

The codebase evaluates model performance using both regression and classification metrics.

### Mileage prediction metrics

- RMSE
- MAE
- R-squared
- adjusted R-squared
- median absolute error

### Scrappage prediction metrics

- accuracy
- AUC-ROC
- precision
- recall
- F1 score

### Suggested paper-aligned results section

Replace the placeholders below with the exact values reported in the manuscript or supplementary information.

- Mileage RMSE: `[insert manuscript/SI value]`
- Mileage MAE: `[insert manuscript/SI value]`
- Mileage R-squared: `[insert manuscript/SI value]`
- Scrappage AUC-ROC: `[insert manuscript/SI value]`
- Scrappage F1: `[insert manuscript/SI value]`
- Decision threshold used for simulation: `0.22`

If you want the Hugging Face model page to render structured evaluation results, these can be added later in `model-index` metadata once the final benchmark values are confirmed.

## Downstream Use in Lifecycle Emissions Analysis

The transformer forecasts are used as inputs to a lifecycle analysis workflow that estimates cumulative CO2e emissions over vehicle lifetimes.

The emissions notebook combines simulated vehicle trajectories with:

- electricity carbon intensity assumptions
- charging and transmission/distribution losses
- battery manufacturing emissions
- chassis manufacturing emissions
- tailpipe and well-to-tank emissions for non-BEV vehicles

This means the model’s outputs are not the final scientific claim on their own. They are one component in a larger analytical pipeline.

## Limitations

Several limitations are important for interpreting this model:

- The model is specific to Britain’s MOT system, vehicle fleet, and data-generating process.
- The scrappage label is an operational proxy derived from the observed testing history, not a perfect direct observation of all retirement mechanisms.
- Early BEV cohorts may not yet have fully observed end-of-life patterns.
- Observational fleet data may reflect policy, consumer behavior, and market structure specific to the study period.
- Forecast quality may degrade when simulating far beyond the empirical support of the historical data.
- The model is designed for fleet-level research patterns, not precise individual-vehicle forecasting.

## Bias, Risks, and Ethical Considerations

Because this model is trained on historical real-world data, it can reproduce historical patterns that are specific to:

- geography
- vehicle market composition
- technology adoption timing
- infrastructure constraints
- socioeconomic usage differences

Potential risks include:

- overgeneralizing UK results to other countries
- interpreting correlation-rich observational forecasts as causal claims
- using fleet-level research outputs in high-stakes individual settings
- overstating certainty for newer vehicle cohorts, especially BEVs with limited lifetime observations

This model should be used with domain expertise and alongside uncertainty analysis.

## Environmental and Societal Context

This model is part of a research workflow focused on understanding how real-world mileage and survival patterns affect lifecycle emissions comparisons across vehicle types. It is intended to support evidence-based discussion of transport decarbonization rather than prescribe decisions for any specific person or organization.

## Repository Contents

The project workflow is organized as follows:

- `MOT_data_processing.ipynb`: data preparation and feature construction
- `MOT_mileage_and_scrappage_analysis.ipynb`: mileage and survival analysis
- `MOT_transformer_model.ipynb`: model training, evaluation, and simulation
- `MOT_emission_calculations.ipynb`: lifecycle emissions calculations
- `MOT_transformer_model_module.py`: reusable model and training utilities

## Reproducibility

To reproduce the workflow, users will need to:

- obtain the public source datasets listed in the repository README
- place them in the expected folder structure
- create the Python environment from `environment.yml`
- run the notebooks in repository order

Because some source data must be downloaded separately, reproduction depends on consistent data access and local setup.


## License

The repository is released under the BSD 3-Clause License.

## Contact

For repository questions, contact the maintainer listed in the repository README.