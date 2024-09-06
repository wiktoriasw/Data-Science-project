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

# # Imports

"""
This file contains the implementation of Decision Tree Regressor model
and hyperparameter tuning, evaluation of the best model on test and validation sets."""

import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

from IPython.display import Image
from tqdm.notebook import tqdm

import pydotplus

import plotly.graph_objs as go

from utils import read_csv, prepare_dataframes, get_labels, generate_data


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

# # DT - set splits

X = df2.iloc[:,2:]
y = df2.iloc[:,1:2]

X

y

# ## Test 1: DT, min_samples_leaf

# +
kfold = KFold(n_splits = 5)

classifiers = [
    DecisionTreeRegressor(min_samples_leaf=10),
    DecisionTreeRegressor(min_samples_leaf=20),
    DecisionTreeRegressor(min_samples_leaf=50),
    DecisionTreeRegressor(min_samples_leaf=100),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        #'Degree': classifier.get_params().get('degree'),
        #'C': classifier.get_params().get('C'),
        #'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_dt_1 = pd.DataFrame(results_list)
# -

df_dt_1

# ## Test 2: DT, min_samples_leaf

# +
kfold = KFold(n_splits = 5)

classifiers = [
    DecisionTreeRegressor(min_samples_leaf=1000),
    DecisionTreeRegressor(min_samples_leaf=10000),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        #'Degree': classifier.get_params().get('degree'),
        #'C': classifier.get_params().get('C'),
        #'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_dt_2 = pd.DataFrame(results_list)
# -

df_dt_2

# ## Test 3: DT, min_samples_leaf

# +
kfold = KFold(n_splits = 5)

classifiers = [
    DecisionTreeRegressor(min_samples_leaf=100),
    DecisionTreeRegressor(min_samples_leaf=200),
    DecisionTreeRegressor(min_samples_leaf=300),
    DecisionTreeRegressor(min_samples_leaf=400),
    DecisionTreeRegressor(min_samples_leaf=500),
    DecisionTreeRegressor(min_samples_leaf=600),
    DecisionTreeRegressor(min_samples_leaf=700),
    DecisionTreeRegressor(min_samples_leaf=800),
    DecisionTreeRegressor(min_samples_leaf=900),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        #'Degree': classifier.get_params().get('degree'),
        #'C': classifier.get_params().get('C'),
        #'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_dt_3 = pd.DataFrame(results_list)
# -

df_dt_3 #600 bo R2 i 100

# ## Test 4: DT, min_samples_leaf, min_samples_split

# +
kfold = KFold(n_splits = 5)

classifiers = [
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=2),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=10),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=50),

    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=2),
    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=10),
    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=50),

]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_dt_4 = pd.DataFrame(results_list)
# -

df_dt_4

# ## Test 5: DT, min_samples_leaf, min_samples_split

# +
kfold = KFold(n_splits = 5)

classifiers = [
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=100),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=500),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=600),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=700),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=800),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=900),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=1000),

    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=100),
    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=500),
    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=600),
    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=700),
    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=800),
    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=900),
    DecisionTreeRegressor(min_samples_leaf=600, min_samples_split=1000),

]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_dt_5 = pd.DataFrame(results_list)
# -

df_dt_5 #mis_samples_leaf = 100, min_samples_split = 500

# ## Test 6: DT, min_samples_leaf, min_samples_split

# +
kfold = KFold(n_splits = 5)

classifiers = [
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=400),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=450),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=500),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=550),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_dt_6 = pd.DataFrame(results_list)
# -

df_dt_6

# ## Test 7: DT, min_samples_leaf, min_samples_split

# +
kfold = KFold(n_splits = 5)

classifiers = [
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=420),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=430),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=440),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=450),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=460),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=470),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=480),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=490),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=500),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=510),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=520),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=530),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=540),

]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_dt_7 = pd.DataFrame(results_list)
# -

df_dt_7 #min_samples_leaf= 100, min_samples_spli = 530

# ## Test 8: DT, min_samples_leaf, min_samples_split

# +
kfold = KFold(n_splits = 5)

classifiers = [
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=550),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=560),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=570),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=580),
    DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=590),

]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'min_samples_leaf': classifier.get_params().get('min_samples_leaf'),
        'min_samples_split': classifier.get_params().get('min_samples_split'),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_dt_8 = pd.DataFrame(results_list)
# -

df_dt_8

# ## DT  visualisation

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

len(X_train), len(y_train)

dtr = DecisionTreeRegressor(min_samples_leaf=100, min_samples_split=530)
dtr.fit(X_train, y_train.values.ravel())
y_pred = dtr.predict(X_test)

decision_path_matrix = dtr.decision_path(X_train, check_input=True)

# +
#print(decision_path_matrix)
# -

dtr.get_depth()

dtr.get_n_leaves()

dtr.get_metadata_routing()

dot_data = tree.export_graphviz(dtr, feature_names=X.columns.values, out_file=None, filled=True)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())

graph.write_png('DT_color.png')

# ## The resulting model

training_set = df2[df2['time'] < '2023-12-13 22:00:00+00:00']
validation_set = df2[(df2['time'] < '2024-01-13 22:00:00+00:00') & (df2['time'] >= '2023-12-13 22:00:00+00:00')]
testing_set = df2[(df2['time'] <= '2024-02-13 22:00:00+00:00') & (df2['time'] >= '2024-01-13 22:00:00+00:00') ]

# +
X_train = training_set.iloc[:,2:]
y_train = training_set.iloc[:,1:2]

X_val = validation_set.iloc[:,2:]
y_val = validation_set.iloc[:,1:2]

X_test = testing_set.iloc[:,2:]
y_test = testing_set.iloc[:,1:2]
# -

dtr = DecisionTreeRegressor(min_samples_leaf=540, min_samples_split=100)
dtr.fit(X_train, y_train.values.ravel())
y_pred_test = dtr.predict(X_test)

# ## Evaluation of the test set

# +
MSE = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)


print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'MAE: {MAE}'
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
    line=dict(color='red')
)
trace2 = go.Scatter(
    x=results2.index,
    y=results2['Predicted TP'],
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
# -

results2.sort_values(by='Predicted TP')

# ## Evaluation of the validation set
y_pred_val = dtr.predict(X_val)

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
# -
