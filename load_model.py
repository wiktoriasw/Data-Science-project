# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
"""The file contains functions for data processing."""
# ## Importy

import keras
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from tensorflow.keras.utils import plot_model

from utils import generate_data, get_labels, prepare_dataframes, read_csv

dataframes = read_csv("./datasets", 'csv')

res = prepare_dataframes(dataframes)

df = get_labels(res, '2024-02-13')

df['rain_1h'] = df['rain_1h'].fillna(0)
df['snow_1h'] = df['snow_1h'].fillna(0)
df['total percipitation'] = df['rain_1h'] + df['snow_1h']
df['last_changed_count'] = df.groupby('last_changed')['last_changed'].transform('count')
df = df.drop_duplicates(subset=['last_changed'])
df = df.loc[:, ['last_changed', 'pressure', 'out_temp', 'out_hum', 'total percipitation']]
df = df.reset_index(drop=True)

# # Tabela 1 (time, P, T, H, R)

df

# # Tabela 2 (time, 0, -1, -2, -3...)

df2 = generate_data(df, 5)
df2

# # Parametry sieci

# +
config = {}
config["step"] = 5
config["epsilon"] = 1e-1
config["units 1st layer"] = 50
#config["dropout 1st layer"] = 0.2
config["units 2nd layer"] = 50
#config["dropout 2nd layer"] = 0.2
config["units 3rd layer"] = 50
config["dropout 3rd layer"] = 0.1
config["units 4th layer"] = 50
config["dropout 4th layer"] = 'None'
config["learning rate"] = 1e-5
config["optimizer"] = 'adam'
config["loss"] = 'mean_squared_error'

config["epochs"] = 100
config["batch size"] = 32

config['MSE'] = 0
config['R2'] = 0


# -

# # Transformacja danych + podział

def transform_data(df, step: int, epsilon: float):
    df = df.copy()
    df['total percipitation'] = np.log(df['total percipitation'] + epsilon)
    df2 = generate_data(df, step)

    return df2

df2 = transform_data(df, config["step"], config["epsilon"])

training_set = df2[df2['time'] < '2023-12-13 22:00:00+00:00']
validation_set = df2[(df2['time'] < '2024-01-13 22:00:00+00:00') & (df2['time'] >= '2023-12-13 22:00:00+00:00')]
testing_set = df2[(df2['time'] <= '2024-02-13 23:00:00+00:00') & (df2['time'] >= '2024-01-13 22:00:00+00:00')]
# +
X_train = training_set.iloc[:,2:]
y_train = training_set.iloc[:,1:2]

X_val = validation_set.iloc[:,2:]
y_val = validation_set.iloc[:,1:2]

X_test = testing_set.iloc[:,2:]
y_test = testing_set.iloc[:,1:2]
# -

# # Nazwa eksperymentu do wczytania

name = 'Log_T33.keras'
#Log_T11.weights.h5
#Log_T33.keras

# # Wczytywanie modelu

loaded_model = keras.saving.load_model(f'D:/Nextcloud/Workspace/Data Science/Projekt/Wyniki/{name}')

# +
#loaded_model.get_weights()
# -

loaded_model.summary()

plot_model(
    loaded_model,
    to_file='model.png',
    show_shapes=False,
    show_layer_names=True,
    rankdir='LR',
    expand_nested=True,
    dpi=100
    )

# # Predykcja

y_pred = loaded_model.predict(X_val)


def invert_transformed_data(y_pred, y_val, epsilon):

    df_pred = pd.DataFrame(y_pred, columns=["TP t+1"])
    df_val = pd.DataFrame(y_val, columns=["TP t+1"])

    df_pred["TP t+1"] = np.exp(df_pred["TP t+1"]) - epsilon
    df_val["TP t+1"] = np.exp(df_val["TP t+1"]) - epsilon
    return df_pred, df_val


y_pred_inv, y_val_inv = invert_transformed_data(y_pred, y_val, config["epsilon"])

y_val_inv = y_val_inv.reset_index(drop=True)
y_val_inv.compare(y_pred_inv)

y_pred_inv = y_pred_inv.rename(columns={'TP t+1':'Predicted TP'})
results = pd.concat([y_pred_inv, y_val_inv], axis=1)

# +
trace1 = go.Scatter(
    x=results.index,
    y=results['TP t+1'],
    mode='lines',
    name='Rzeczywiste TP',
    line=dict(color='red')
)
trace2 = go.Scatter(
    x=results.index,
    y=results['Predicted TP'],
    mode='lines',
    name='Przewidywane TP',
    line=dict(color='blue')
)

fig = go.Figure()

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(
    xaxis_title='Indeksy pomiarów w zbiorze walidacyjnym',
    yaxis_title='Suma opadów [mm]')

fig.show()

# +
MSE = mean_squared_error(y_val_inv, y_pred_inv)
r2 = r2_score(y_val_inv, y_pred_inv)
MAE = mean_absolute_error(y_val_inv, y_pred_inv)
print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'Mean Absolute Error: {MAE}',
 )
# -

# # Predykcja na zbiorze testowym
#

y_pred_test = loaded_model.predict(X_test)

y_pred_test_inv, y_test_inv = invert_transformed_data(y_pred_test, y_test, config["epsilon"])

y_test_inv = y_test_inv.reset_index(drop=True)
y_test_inv.compare(y_pred_test_inv)

y_pred_test_inv = y_pred_test_inv.rename(columns={'TP t+1':'Predicted TP'})
results_2 = pd.concat([y_pred_test_inv, y_test_inv], axis=1)

sorted_res_2 = results_2.sort_values(by='Predicted TP').reset_index(drop=True)

results_2 #0,02

# # Macierz pomyłek na zbiorze testowym

results_2['Label Predicted'] = results_2.apply(lambda row: 0 if row['Predicted TP'] < 0.05 else 1, axis=1)

results_2['Label Real'] = results_2.apply(lambda row: 0 if row['TP t+1'] < 0.05 else 1, axis=1)

results_2

cm = confusion_matrix(results_2['Label Real'], results_2['Label Predicted'])

disp = ConfusionMatrixDisplay(cm)

disp.plot()

# +
trace1 = go.Scatter(
    x=results.index,
    y=results_2['TP t+1'],
    mode='lines',
    name='Rzeczywiste TP',
    line=dict(color='red')
)
trace2 = go.Scatter(
    x=results.index,
    y=results_2['Predicted TP'],
    mode='lines',
    name='Przewidywane TP',
    line=dict(color='blue')
)

fig = go.Figure()

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(
    xaxis_title='Indeksy pomiarów w zbiorze testowym',
    yaxis_title='Suma opadów [mm]')

fig.show()

# +
MSE = mean_squared_error(y_test_inv, y_pred_test_inv)
r2 = r2_score(y_test_inv, y_pred_test_inv)
MAE = mean_absolute_error(y_test_inv, y_pred_test_inv)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'Mean Absolute Error: {MAE}',
 )
# -
