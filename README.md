# Applying Machine Learning methods for rainfall prediction based on pressure, temperature, and humidity

The goal of this project is to analyze own dataset containing temperature, humidity, and pressure recorded by an IoT sensor. The project aims to investigate whether it is possible to predict rainfall using traditional machine learning techniques and recurrent neural networks.

## Repository Structure

- `datasets/` - directory containing multiple CSV files with data used for training and testing the models.
- `svr.py` - contains the Support Vector Regressor model, hyperparameter tuning, evaluation of the best model on test and validation sets.
- `dt.py` - contains the Decision Tree Regressor model and hyperparameter tuning, evaluation of the best model on test and validation sets.
- `rfr.py` - contains the Random Forest Regressor model and hyperparameter tuning, evaluation of the best model on test and validation sets.
- `lr.py` - contains the Linear Regression model, predictions on test and validation sets and evaluations.
- `lstm.py` - contains the LSTM model and predictions on test and validation set.
- `Log_T33.keras` - this file contains the LSTM model that achieved the best results.
- `load_model.py` - allows loading the saved model from `LSTM.py` and using it to make predictions. 
- `utils.py` - contains functions for data processing.
- `plots.py` - contains scripts for generating plots.
- `requirements.txt`

## Running the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/wiktoriasw/Data-Science-project.git
    ```

2. Navigate to the project directory:  
    ```bash
    cd Data-Science-project
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the selected model for example:
    ```bash
    python SVR.py
    ```

## Results

A full analysis of the project results and a detailed discussion are available in the attached document `Results_PL.pdf`. This document includes the following sections:

1. **Introduction** (page 2): A brief overview of the project context.
2. **Technology Stack** (page 2): An overview of the tools and technologies used in the project.
3. **Data Characteristics** (page 3): Description of the collected data, its structure, and main features.
4. **Project Goals** (page 6): Definition of the analysis goals and research questions.
5. **Analysis Process** (page 7):
   - **Data Preparation** (page 7): Description of the data preprocessing.
   - **Data Splitting** (page 10): Details on splitting the data into sets.
   - **Data Analysis Using Traditional ML Techniques** (page 10):
     - Support Vector Regressor (SVR) (page 10)
     - Decision Tree Regressor (DTR) (page 13)
     - Random Forest Regressor (RFR) (page 15)
     - Linear Regression (LR) (page 17)
   - **Data Analysis Using Recurrent Neural Networks (LSTM)** (page 19)
6. **Summary** (page 23): Key conclusions from the analysis.
7. **Discussion and Conclusions** (page 23): A broader discussion on the results and their implications.
8. **Bibliography**: A list of literature and sources used in the project.







