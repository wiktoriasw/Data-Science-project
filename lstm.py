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

# %load_ext autoreload
# %autoreload 2

"""
The file contains the LSTM model
and predictions on test and validation set.
"""

# # Imports

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import tensorflow.keras as keras
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential

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

# # Table 1 (time, P, T, H, R)

df

# +
#df.value_counts('total percipitation')
# -

# # Table 2 (time, 0, -1, -2, -3...)

df2 = generate_data(df, 5)
df2

# # Name of experiment

NAME = 'Log_T33'

# # Network parameters

# +
config = {}
config["step"] = 5
config["epsilon"] = 1e-1
config["units 1st layer"] = 50
config["dropout 1st layer"] = 0.1
config["units 2nd layer"] = 50
config["units 3rd layer"] = 50
config["dropout 3rd layer"] = 0.1
config["learning rate"] = 1e-5
config["optimizer"] = 'adam'
config["loss"] = 'mean_squared_error'

config["epochs"] = 100
config["batch size"] = 32

#config['MSE'] = 0
#config['R2'] = 0
# -

# # Data transformation and sets

def transform_data(df, step: int, epsilon: float):

    df = df.copy()
    df['total percipitation'] = np.log(df['total percipitation'] + epsilon)
    df2 = generate_data(df, step)

    return df2


df2 = transform_data(df, config["step"], config["epsilon"])

training_set = df2[df2['time'] < '2023-12-13 22:00:00+00:00']
validation_set = df2[
    (df2['time'] < '2024-01-13 22:00:00+00:00') &
    (df2['time'] >= '2023-12-13 22:00:00+00:00')
]
testing_set = df2[
    (df2['time'] <= '2024-02-13 22:00:00+00:00') &
    (df2['time'] >= '2024-01-13 22:00:00+00:00')
]

# +
X_train = training_set.iloc[:,2:]
y_train = training_set.iloc[:,1:2]

X_val = validation_set.iloc[:,2:]
y_val = validation_set.iloc[:,1:2]

X_test = testing_set.iloc[:,2:]
y_test = testing_set.iloc[:,1:2]
# -

len(X_train.index), len(X_val.index), len(X_test.index)

# - Training set to 2033 obserwacje, z czego 1591 nie pada (0.0 mm), a 442 pada (!= 0.0 mm)
# - Validation set to 744 obserwacje, z czego 513 nie pada (0.0. mm), a 231 pada (!= 0.0mm)
# - Testing set to 721 obserwacje, z czego 557 nie pada (0.0. mm), a 164 pada (!= 0.0. mm)

# # Network model

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# +
regressor = Sequential()

regressor.add(
    LSTM(
        units=config["units 1st layer"],
        return_sequences=True,
        input_shape=(X_train.shape[1], 1),))

regressor.add(LSTM(units=config["units 2nd layer"], return_sequences=True))

regressor.add(LSTM(units=config["units 3rd layer"]))

regressor.add(Dropout(config["dropout 3rd layer"]))

regressor.add(Dense(units=1))
regressor.add(LeakyReLU(negative_slope=0.3))

optimizer = Adam(learning_rate=config["learning rate"])

regressor.compile(optimizer=optimizer, loss=config["loss"])

# +
path_checkpoint = f'Wyniki/{NAME}.keras'

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=False,
    save_best_only=True,
)
# -

history = regressor.fit(
    X_train,
    y_train,
    epochs=config["epochs"],
    validation_data = (X_val, y_val),
    batch_size=config["batch size"],
    callbacks=[modelckpt_callback],
)

# # Visualizing the loss function

# +
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = np.arange(len(loss))

trace1 = go.Scatter(
    x=epochs, y=loss, mode='lines', name='Training loss', line=dict(color='red')
)
trace2 = go.Scatter(
    x=epochs, y=val_loss, mode='lines', name='Validation loss', line=dict(color='blue')
)

fig = go.Figure()

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(
    xaxis_title='Nr epoki',
    yaxis_title='Loss')

fig.update_yaxes(range=[0, max(loss)+1])

fig.show()
# -

# # Prediction

y_pred = regressor.predict(X_val)


# # Inverse data transformation

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
    name='Real TP',
    line=dict(color='red'),
)

trace2 = go.Scatter(
    x=results.index,
    y=results['Predicted TP'],
    mode='lines',
    name='Predicted TP',
    line=dict(color='blue'),
)

fig = go.Figure()

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(
    title='Predicted vs. Real Total Percipitation',
    xaxis_title='Time',
    yaxis_title='Sum of percipitation',
)

fig.show()

# +
MSE = mean_squared_error(y_val_inv, y_pred_inv)
r2 = r2_score(y_val_inv, y_pred_inv)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}',
)
# -

# # Conclusions
# - nie zna charakterystyki lutego
# - Log T32, Log T11

# # Prediction on the test set

y_pred_test = regressor.predict(X_test)

y_pred_test_inv, y_test_inv = invert_transformed_data(y_pred_test, y_test, config["epsilon"])

y_test_inv = y_test_inv.reset_index(drop=True)
y_test_inv.compare(y_pred_test_inv)

y_pred_test_inv = y_pred_test_inv.rename(columns={'TP t+1':'Predicted TP'})
results_2 = pd.concat([y_pred_test_inv, y_test_inv], axis=1)

# +
trace1 = go.Scatter(
    x=results.index,
    y=results_2['TP t+1'],
    mode='lines',
    name='Real TP',
    line=dict(color='red'),
)

trace2 = go.Scatter(
    x=results.index,
    y=results_2['Predicted TP'],
    mode='lines',
    name='Predicted TP',
    line=dict(color='blue'),
)

fig = go.Figure()

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(
    title='Predicted vs. Real Total Percipitation',
    xaxis_title='Time',
    yaxis_title='Sum of percipitation',
)

fig.show()

# +
MSE = mean_squared_error(y_test_inv, y_pred_test_inv)
r2 = r2_score(y_test_inv, y_pred_test_inv)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}',
)
