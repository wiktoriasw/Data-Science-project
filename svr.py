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
This file contains the implementation of a support vector regressor model,
hyperparameter tuning, evaluation of the best model on test and validation sets.
"""

# # Imports

import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import (confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from tqdm.notebook import tqdm

from utils import generate_data, get_labels, prepare_dataframes, read_csv

# # Data processing

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

# # SVR

X = df2.iloc[:,2:]
y = df2.iloc[:,1:2]

X

y

# ## Test 1: Poly, RBF, C=[0.1,1,10]

# +
kfold = KFold(n_splits = 5)

classifiers = [
    SVR(kernel='poly', C=1, degree=1),
    SVR(kernel='poly', C=1, degree=2),
    SVR(kernel='poly', C=1, degree=3),
    SVR(kernel='poly', C=1, degree=4),

    SVR(kernel='poly', C=0.1, degree=1),
    SVR(kernel='poly', C=0.1, degree=2),
    SVR(kernel='poly', C=0.1, degree=3),
    SVR(kernel='poly', C=0.1, degree=4),

    SVR(kernel='poly', C=10, degree=1),
    SVR(kernel='poly', C=10, degree=2),
    SVR(kernel='poly', C=10, degree=3),
    SVR(kernel='poly', C=10, degree=4),

    SVR(kernel='rbf', C=0.1, degree=1),
    SVR(kernel='rbf', C=0.1, degree=2),
    SVR(kernel='rbf', C=0.1, degree=3),
    SVR(kernel='rbf', C=0.1, degree=4),

    SVR(kernel='rbf', C=1, degree=1),
    SVR(kernel='rbf', C=1, degree=2),
    SVR(kernel='rbf', C=1, degree=3),
    SVR(kernel='rbf', C=1, degree=4),

    SVR(kernel='rbf', C = 10, degree=1),
    SVR(kernel='rbf', C = 10, degree=2),
    SVR(kernel='rbf', C = 10, degree=3),
    SVR(kernel='rbf', C = 10, degree=4),

    SVR(kernel='rbf', C = 20, degree=1),
    SVR(kernel='rbf', C = 20, degree=2),
    SVR(kernel='rbf', C = 20, degree=3),
    SVR(kernel='rbf', C = 20, degree=4),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'Kernel': classifier.get_params().get('kernel'),
        'Degree': classifier.get_params().get('degree'),
        'C': classifier.get_params().get('C'),
        'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_svr_1 = pd.DataFrame(results_list)
# -

df_svr_1

# ## Test 2: RBF, C=[20,50,100]

# +
kfold = KFold(n_splits = 5)

classifiers = [
    SVR(kernel='rbf', C = 20, degree=1),
    SVR(kernel='rbf', C = 20, degree=2),
    SVR(kernel='rbf', C = 20, degree=3),
    SVR(kernel='rbf', C = 20, degree=4),

    SVR(kernel='rbf', C = 50, degree=1),
    SVR(kernel='rbf', C = 50, degree=2),
    SVR(kernel='rbf', C = 50, degree=3),
    SVR(kernel='rbf', C = 50, degree=4),

    SVR(kernel='rbf', C = 100, degree=1),
    SVR(kernel='rbf', C = 100, degree=2),
    SVR(kernel='rbf', C = 100, degree=3),
    SVR(kernel='rbf', C = 100, degree=4),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'Kernel': classifier.get_params().get('kernel'),
        'Degree': classifier.get_params().get('degree'),
        'C': classifier.get_params().get('C'),
        'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_svr_2 = pd.DataFrame(results_list)
# -

df_svr_2

# ## Test 3: RBF, C=[1e3,1e4,1e5]

# +
kfold = KFold(n_splits = 5)

classifiers = [
    SVR(kernel='rbf', C = 1000, degree=1),
    SVR(kernel='rbf', C = 1000, degree=2),
    SVR(kernel='rbf', C = 1000, degree=3),
    SVR(kernel='rbf', C = 1000, degree=4),

    SVR(kernel='rbf', C = 10000, degree=1),
    SVR(kernel='rbf', C = 10000, degree=2),
    SVR(kernel='rbf', C = 10000, degree=3),
    SVR(kernel='rbf', C = 10000, degree=4),

    SVR(kernel='rbf', C = 100000, degree=1),
    SVR(kernel='rbf', C = 100000, degree=2),
    SVR(kernel='rbf', C = 100000, degree=3),
    SVR(kernel='rbf', C = 100000, degree=4),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'Kernel': classifier.get_params().get('kernel'),
        'Degree': classifier.get_params().get('degree'),
        'C': classifier.get_params().get('C'),
        'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_svr_3 = pd.DataFrame(results_list) #df_svr_3
# -

df_svr_3

# ## Test 4: RBF, C=[1e6,1e7,1e8]

# +
kfold = KFold(n_splits = 5)

classifiers = [
    SVR(kernel='rbf', C = 1e6, degree=1),
    SVR(kernel='rbf', C = 1e6, degree=2),
    SVR(kernel='rbf', C = 1e6, degree=3),
    SVR(kernel='rbf', C = 1e6, degree=4),

    SVR(kernel='rbf', C = 1e7, degree=1),
    SVR(kernel='rbf', C = 1e7, degree=2),
    SVR(kernel='rbf', C = 1e7, degree=3),
    SVR(kernel='rbf', C = 1e7, degree=4),

    SVR(kernel='rbf', C = 1e8, degree=1),
    SVR(kernel='rbf', C = 1e8, degree=2),
    SVR(kernel='rbf', C = 1e8, degree=3),
    SVR(kernel='rbf', C = 1e8, degree=4),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'Kernel': classifier.get_params().get('kernel'),
        'Degree': classifier.get_params().get('degree'),
        'C': classifier.get_params().get('C'),
        'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_svr_4 = pd.DataFrame(results_list)
# -

df_svr_4

# ## Test 5: RBF C=[1e4, epsilon]

# +
kfold = KFold(n_splits = 5)

classifiers = [
    SVR(kernel='rbf', C=1e4, epsilon=1e-2),
    SVR(kernel='rbf', C=1e4, epsilon=0.05),
    SVR(kernel='rbf', C=1e4, epsilon=0.1),
    SVR(kernel='rbf', C=1e4, epsilon=0.5),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'Kernel': classifier.get_params().get('kernel'),
        'Degree': classifier.get_params().get('degree'),
        'C': classifier.get_params().get('C'),
        'epsilon': classifier.get_params().get('epsilon'),
        'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_svr_5 = pd.DataFrame(results_list)
# -

df_svr_5

# ## Test 6: Poly, C=1e4, degree = [2,3..,9]

# +
kfold = KFold(n_splits = 5)

classifiers = [
    SVR(kernel='poly', C=1e4, degree=2),
    SVR(kernel='poly', C=1e4, degree=3),
    SVR(kernel='poly', C=1e4, degree=4),
    SVR(kernel='poly', C=1e4, degree=5),
    SVR(kernel='poly', C=1e4, degree=6),
    SVR(kernel='poly', C=1e4, degree=7),
    SVR(kernel='poly', C=1e4, degree=8),
    SVR(kernel='poly', C=1e4, degree=9),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'Kernel': classifier.get_params().get('kernel'),
        'Degree': classifier.get_params().get('degree'),
        'C': classifier.get_params().get('C'),
        'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_svr_6 = pd.DataFrame(results_list)
# -

df_svr_6

# ## Test 7: Poly, C=[1e3,1e5], degree = [1, 2, 3]

# +
kfold = KFold(n_splits = 5)

classifiers = [
    SVR(kernel='poly', C=1e3, degree=2),
    SVR(kernel='poly', C=1e3, degree=3),
    SVR(kernel='poly', C=1e3, degree=4),

    SVR(kernel='poly', C=1e5, degree=2),
    SVR(kernel='poly', C=1e5, degree=3),
    SVR(kernel='poly', C=1e5, degree=4),
]

results_list = []
for classifier in tqdm(classifiers):
    #classifier.set_params(**{'C': 0.5})
    #print(type(classifier).__name__, classifier.get_params())

    result = {
        'Classifier': type(classifier).__name__, 
        'Kernel': classifier.get_params().get('kernel'),
        'Degree': classifier.get_params().get('degree'),
        'C': classifier.get_params().get('C'),
        'Params': classifier.get_params(),
        #**classifier.get_params(),
    }

    for scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        score = cross_val_score(classifier, X, y.values.ravel(), cv = kfold, scoring=scoring)

        result[scoring] = score.mean()

    results_list.append(result)

df_svr_7 = pd.DataFrame(results_list)
# -

df_svr_7

# ## Grid Search

# +
#from sklearn.model_selection import GridSearchCV

# +
# X = df2.iloc[:,2:]
# y = df2.iloc[:,1:2]

# +
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
# )
# -

parameters = {
     'kernel': ['poly', 'rbf'],
     'degree': [2, 3],
     'C': [1, 10], 
     #'gamma': [1, 10],
}

# +
# clf = GridSearchCV(
#       SVR(),
#       parameters,
#       scoring = 'neg_mean_squared_error',
#       cv = 3)

# +
#clf.fit(X_train, y_train.values.ravel())

# +
#print("Najlepsze parametry:", clf.best_params_)

# +
#clf.cv_results_

# +
#clf.best_score_

# +
#clf.scorer_ #True?
# -

# ## The obtained model

# ### Division of data into sets

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

# ### Model with the best hyperparameters

svr = SVR(kernel='rbf', C=1e4, epsilon=0.1).fit(X_train, y_train.values.ravel())

# ### Evaluation of the model on a test set

y_pred_test = svr.predict(X_test)

# +
MSE = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'MAE: {MAE}',
)
# -

# ### Prediction graph on test set

y_pred = svr.predict(X_train)

# +
MSE = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
MAE = mean_absolute_error(y_train, y_pred)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'MAE: {MAE}',
)
# -

y_pred_test = pd.DataFrame(y_pred_test, columns  = ['Predicted TP'])

y_test = y_test.reset_index(drop=True)

results2 = pd.concat([y_pred_test, y_test], axis=1)

# +
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

results2.sort_values(by='Predicted TP')

confusion_matrix(y_test, y_pred_test)

# ## Evaluation of the model on the validation set

y_pred_val = svr.predict(X_val)

# +
MSE = mean_squared_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
MAE = mean_absolute_error(y_val, y_pred_val)

print(
    f'Mean Squared Error: {MSE}\n',
    f'R2 coefficient: {r2}\n',
    f'MAE: {MAE}',
)
# -

# ### Prediction graph on validation set

y_pred_val = pd.DataFrame(y_pred_val, columns  = ['Predicted TP'])

y_val = y_val.reset_index(drop=True)

results = pd.concat([y_pred_val, y_val], axis=1)

results

# +
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
# -
