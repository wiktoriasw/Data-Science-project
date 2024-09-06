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
This file contains contains the Random Forest Regressor model
and hyperparameter tuning, evaluation of the best model on test and validation sets.
"""

# # Imports

#from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
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

# ## Test 1: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=500, min_samples_leaf=100, min_samples_split=540),
    RandomForestRegressor(n_estimators=1000,min_samples_leaf=100, min_samples_split=540),
]

results_list = []
for classifier in tqdm(classifiers):

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_1 = pd.DataFrame(results_list)
# -

df_rft_1

# ## Test 2: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=800, min_samples_leaf=100, min_samples_split=540),
    RandomForestRegressor(n_estimators=1500,min_samples_leaf=100, min_samples_split=540),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_2 = pd.DataFrame(results_list)
# -

df_rft_2

# ## Test 3: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=400, min_samples_leaf=100, min_samples_split=540),
    RandomForestRegressor(n_estimators=700,min_samples_leaf=100, min_samples_split=540),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_3 = pd.DataFrame(results_list)
# -

df_rft_3

# ## Test 4: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=100, min_samples_split=540),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_4 = pd.DataFrame(results_list)
# -

df_rft_4

# ## Test 5: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=550, min_samples_leaf=100, min_samples_split=540),
    RandomForestRegressor(n_estimators=650, min_samples_leaf=100, min_samples_split=540),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_5 = pd.DataFrame(results_list)
# -

df_rft_5

# ## Test 6: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=50, min_samples_split=440),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=150, min_samples_split=640),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'), 
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_6 = pd.DataFrame(results_list)
# -

df_rft_6

# ## Test 7: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=50, min_samples_split=540),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=150, min_samples_split=540),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)
        result[scoring] = score.mean()

    results_list.append(result)

df_rft_7 = pd.DataFrame(results_list)
# -

df_rft_7

# ## Test 8: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=100, min_samples_split=540),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),        
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_8 = pd.DataFrame(results_list)
# -

df_rft_8

# ## Test 9: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=100, min_samples_split=440),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=100, min_samples_split=640),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_9 = pd.DataFrame(results_list)
# -

df_rft_9

# ## Test 10: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=60, min_samples_split=540),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=70, min_samples_split=540),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=540),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=90, min_samples_split=540),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_10 = pd.DataFrame(results_list)
# -

df_rft_10

# ## Test 11: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=90, min_samples_split=440),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=90, min_samples_split=540),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=90, min_samples_split=640),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),    
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_11 = pd.DataFrame(results_list)
# -

df_rft_11

# ## Test 12: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=440),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=540),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=640),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_12 = pd.DataFrame(results_list)
# -

df_rft_12

# ## Test 13: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=500),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=540),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=600),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_13 = pd.DataFrame(results_list)
# -

df_rft_13

# ## Test 14: RFR

# +
kfold = KFold(n_splits = 5)

classifiers = [
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=510),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=520),
    RandomForestRegressor(n_estimators=600, min_samples_leaf=80, min_samples_split=530),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
        'n_estimators': classifier.get_params().get('n_estimators'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_rft_14 = pd.DataFrame(results_list)
# -

df_rft_14

# ## The obtained model

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

rfr = RandomForestRegressor(n_estimators=600,min_samples_leaf=510, min_samples_split=80)
rfr.fit(X_train, y_train.values.ravel())
y_pred_test = rfr.predict(X_test)

# ## Evaluation of the test set

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
    yaxis_title='Suma opadów [mm]',
)

fig.show()
# -

# ##  Evaluation of the validation set

y_pred_val = rfr.predict(X_val)

# +
MSE = mean_squared_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
MAE = mean_absolute_error(y_val, y_pred_val)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'MAE: {MAE}',
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
