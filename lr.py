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
This file contains the Linear Regression model,
predictions on test and validation sets and evaluations.
"""

import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from tqdm.notebook import tqdm

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

# ## Table 1 (time, P, T, H, R)

df

df.value_counts('total percipitation')

# ## Table 2 (time, 0, -1, -2, -3...)

df2 = generate_data(df, 5)
df2

X = df2.iloc[:,2:]
y = df2.iloc[:,1:2]

X

y

# +
kfold = KFold(n_splits = 5)

classifiers = [
    LinearRegression(),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        #'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        #'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_lr_1 = pd.DataFrame(results_list)
# -

df_lr_1

# ## LR coef

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
)

# +
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
# -

print(
    f'Wzór funkcji: '
    f'y = {lr.coef_[0][0]} * x1 '
    f'+ {lr.coef_[0][1]} * x2 '
    f'+ {lr.coef_[0][2]} * x3 '
    f'+ {lr.coef_[0][3]} * x4 '
    f'+ {lr.coef_[0][4]} * x5 '
    f'+ {lr.coef_[0][5]} * x6 '
    f'+ {lr.coef_[0][6]} * x7 '
    f'+ {lr.coef_[0][7]} * x8 '
    f'+ {lr.coef_[0][8]} * x9 '
    f'+ {lr.coef_[0][9]} * x10 '
    f'+ {lr.coef_[0][10]} * x11 '
    f'+ {lr.coef_[0][11]} * x12 '
    f'+ {lr.coef_[0][12]} * x13 '
    f'+ {lr.coef_[0][13]} * x14 '
    f'+ {lr.coef_[0][14]} * x15 '
    f'+ {lr.coef_[0][15]} * x16 '
    f'+ {lr.coef_[0][16]} * x17 '
    f'+ {lr.coef_[0][17]} * x18 '
    f'+ {lr.coef_[0][18]} * x19 '
    f'+ {lr.coef_[0][19]} * x20 '
    f'+ {lr.intercept_}'
)

lr.coef_

# ### The obtained model

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

lr = LinearRegression().fit(X_train, y_train)

y_pred_test = lr.predict(X_test)

# +
MSE = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'MAE: {MAE}',
 )

# +
y_pred_test = pd.DataFrame(y_pred_test, columns  = ['Predicted TP'])
y_test = y_test.reset_index(drop=True)
results2 = pd.concat([y_pred_test, y_test], axis=1)

trace1 = go.Scatter(
    x=results2.index,
    y=results2['TP t+1'],
    mode='lines',
    name='Rzeczywiste TP',
    line=dict(color='red'),
)

trace2 = go.Scatter(
    x=results2.index,
    y=results2['Predicted TP'],
    mode='lines',
    name='Przewidywane TP',
    line=dict(color='blue'),
)

fig = go.Figure()

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(
    xaxis_title='Indeksy pomiarów w zbiorze testowym',
    yaxis_title='Suma opadów [mm]')

fig.show()
# -

results2.sort_values(by='Predicted TP')

results2[results2['Predicted TP']==0]

y_pred_val = lr.predict(X_val)

# +
MSE = mean_squared_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
MAE = mean_absolute_error(y_val, y_pred_val)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'MAE: {MAE}'
 )

# +
y_pred_val = pd.DataFrame(y_pred_val, columns  = ['Predicted TP'])
y_val = y_val.reset_index(drop=True)
results = pd.concat([y_pred_val, y_val], axis=1)

trace1 = go.Scatter(
    x=results.index,
    y=results['TP t+1'],
    mode='lines',
    name='Rzeczywiste TP',
    line=dict(color='red'),
)

trace2 = go.Scatter(
    x=results.index,
    y=results['Predicted TP'],
    mode='lines',
    name='Przewidywane TP',
    line=dict(color='blue'),
)

fig = go.Figure()

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(
    xaxis_title='Indeksy pomiarów w zbiorze walidacyjnym',
    yaxis_title='Suma opadów [mm]',
    )

fig.show()
