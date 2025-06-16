
# Improving electric vehicle mileage and lifetime reinforce lifecycle emission benefits 
## Overview

This repository contains the code and data processing workflows for "Improving electric vehicle mileage and lifetime reinforce lifecycle emission benefits". The analysis uses the MOT testing data alongside vehicle specifications to analyze the real-world usage patterns, lifespans, and lifecycle greenhouse gas emissions of different vehicle types in Britain's fleet.

## Contents
This repository contains four jupyter notebooks and one python source file, that need to be run in the following order to replicate our results:

### 1. MOT_data_processing.ipynb
Preprocessing steps for the MOT dataset including:
- Adding vehicle attribute data from three databases (VCA, EEA, EV Database)
- Filtering the MOT data using consistent criteria
- Preprocessing data for transformer model training and prediction

### 2. MOT_mileage_and_scrappage_analysis.ipynb
Analysis producing Figures 1-3 from the paper:
- Figure 1: BEV global and UK sales data
-Figure 2: Annual and cumulative mileage by vehicle age
- Figure 3: Vehicle survival rates by age and mileage

### 3. MOT_transformer_model.ipynb
Implementation of the transformer neural network model that:
- Uses preprocessed data from MOT_data_processing.ipynb
- Trains on historical data to predict future vehicle usage patterns
- Uses functions from MOT_transformer_model_module.py

### 4. MOT_emission_calculations.ipynb
Lifecycle greenhouse gas emissions calculations that:
- Use historical data from MOT_data_processing.ipynb
- Incorporate projected future data from MOT_transformer_model.ipynb
- Produce Figure 4 showing lifecycle CO2e emissions

### MOT_transformer_model_module.py
Conatains the classes and fucntions used in MOT_transformer_model.ipynb. 

### Data needs to be populated into the respective folders and correctly formatted:
1. MOT data: Each MOT data year must be added to the correct "Year_Result" folder as a single .csv file. 
2. EEA data must be added to the eea_data folder
3. VCA datanmust be added to the vca_data folder
4. EV Databasae data must be added to the ev_data folder

## Python Environment/Dependencies

Please see environment.yml

## Data Sources
MOT test results (2005-2023) https://www.data.gov.uk/dataset/e3939ef8-30c7-4ca8-9c7c-ad9475cc9b2f/anonymised_mot_test
Vehicle Certification Agency (VCA) https://carfueldata.vehicle-certification-agency.gov.uk/
European Environment Agency (EEA) https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b
EV Database  (battery capacity, energy consumption) https://ev-database.org/
Electricity generation mix data https://doi.org/10.1016/j.enpol.2016.12.037
