# Project: Fraud Detection in Electricity and Gas Consumption

## Project Overview
This project focuses on developing a supervised machine learning model to detect fraudulent activities in electricity and gas consumption. It is a classification to identify clients involved in fraudulent manipulations of meters. The type of learning employed is supervised learning, as the model will be trained on labeled data indicating fraudulent and non-fraudulent activities.

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
