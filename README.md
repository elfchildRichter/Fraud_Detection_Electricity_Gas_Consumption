<!-- P.S. My work doesn't fit into one Jupyter notebook, so I've uploaded the files to a GitHub repository, where you can find detailed information about the machine learning approach used. Here is the link:
[Fraud_Detection](https://github.com/elfchildRichter/Fraud_Detection) https://github.com/elfchildRichter/Fraud_Detection
<br> -->

UC Boulder MSDS course work

DTSA-5509 Intro to Machine Learning Final Project

# Project: Fraud Detection in Electricity and Gas Consumption

## Project Overview
This project develops a supervised machine learning model to detect fraudulent activities in electricity and gas consumption. It is a classification and the model will be trained on labeled data indicating fraudulent and non-fraudulent activities.

## Project Goals
The goal of this project is to reduce financial losses suffered by the Tunisian Company of Electricity and Gas (STEG) due to meter tampering by consumers. By accurately detecting fraud, which could enhance the company's revenue and ensure fair billing practices. This initiative not only serves an economic purpose but also promotes ethical consumer behavior.

## Data Source and Description
The dataset for this project is obtained from the official [STEG website](https://www.steg.com.tn/en/institutionnel/mission.html) and the [Zindi platform](https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge). It encompasses two main tabulated datasets: **client data** (client) and **billing history** (invoice), covering the period from 2005 to 2019.


1. **Client Data**
   - **Client_id**: Unique identifier for each client.
   - **District**: Geographic district of the client.
   - **Client_catg**: Category of the client.
   - **Region**: Area where the client is located.
   - **Creation_date**: Date when the client account was created.
   - **Target**: Indicates fraud (1) or not fraud (0).


2. **Invoice Data**
   - **Client_id**: Unique identifier for each client.
   - **Invoice_date**: Date when the invoice was issued.
   - **Tarif_type**: Type of tariff applied.
   - **Counter_number**, **Counter_statue**, **Counter_code**, **Reading_remarque**, **Counter_coefficient**: Various meter-related information.
   - **Consommation_level_1** to **Consommation_level_4**: Different levels of consumption.
   - **Old_index**, **New_index**: Meter reading indices.
   - **Months_number**: Number of months covered by the invoice.
   - **Counter_type**: Type of counter used.


- Categorical: 'district', 'client_catg', 'region', 'counter_type', 'counter_number', 'counter_statue', 'counter_code', 'reading_remqrque'

- Numerical: other features besides categorical features


<br>

The repository includes:

## EDA and Dat Preprocessing
Conduct exploratory data analysis on the training and testing data, including data visualization and statistical analysis, cleaning and preprocessing the raw data.

## Baseline Model
Create an initial prediction model using the preprocessed dataset and the LightGBM, serving as the basis for subsequent improvements.

## LightGBM_1 to LightGBM_5
Models 1 to 5 represent stages in the process following feature engineering, feature selectoin based on feature importance and correlation, and hyperparameter tuning. Compare and analyze the predictive results of these models to evaluate the impact of various methods and approaches.

## Final Model and Prediction
Summarize the procedures, such as data preprocessing, aggregation, and hyperparameter tuning, to establish the final model and predict the target for unknown test data.

## Brief Summary

- In the original dataset, features such as 'region', 'counter_number', 'counter_code', 'new_index', 'old_index', and 'tarif_type' have a significant impact on the prediction model.

- Introducing new features like 'uniq_counter' and 'uniq_counter_num' evidently enhances the model's performance.

- Feature selection based on calculating feature importance and correlation is an effective strategy to prevent overfitting without compromising model performance.

- Statistical values derived from categorical and numerical featues substantially improve the model's accuracy.

- Hyperparameter tuning using Bayesian optimization can identify an improved set of parameters, leading to a slight enhancement in model efficacy.

<!-- - More details are available in the repository. -->
