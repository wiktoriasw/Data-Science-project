"""
The file contains functions for data processing.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm


def prepare_data(df):
    df = df.copy()

    if 'attributes' in df.columns:
        del df['attributes']

    if 'last_updated' in df.columns:
        del df['last_updated']

    df = df[(df['state'] != 'unavailable') & (df['state'] != 'unknown')]

    df['last_changed'] = pd.to_datetime(df['last_changed'], format='mixed')
    df['state'] = df['state'].astype(float)

    return df


def fill_data(df, name: str):
    df = df.copy()
    start_date = df['last_changed'].min().floor('5min')
    end_date = df['last_changed'].max().ceil('5min')

    time_column = pd.date_range(start=start_date, end=end_date, freq='5min', tz='UTC')
    time_series = pd.DataFrame(time_column, columns=['last_changed'])

    df = pd.merge(df, time_series, on='last_changed', how='outer').sort_values(by='last_changed')

    df = df.infer_objects(copy=False)

    df = df.fillna(df.interpolate(method='linear'))

    del df['entity_id']

    df = df.rename(columns={'state': name})

    df = df[
        (df['last_changed'].dt.minute % 5 == 0) &
        (df['last_changed'].dt.second == 0) &
        (df['last_changed'].dt.microsecond == 0)
    ]

    return df


def read_csv(root_dir: str, ext: str):
    dataframes = {}
    map_name = {
        'sensor.lumi_lumi_weather_2309e207_humidity.csv': 'out_hum',
        'sensor.lumi_lumi_weather_2309e207_temperature.csv': 'out_temp',
        'sensor.lumi_lumi_weather_2309e207_pressure.csv': 'pressure',
    }
    for item in os.listdir(root_dir):
        if item.endswith(ext) and item in map_name:
            df = pd.read_csv(os.path.join(root_dir, item), index_col=0, low_memory=False)
            dataframes[map_name[item]] = df

    return dataframes


def prepare_dataframes(dataframes: dict):
    prev = pd.DataFrame(columns=['last_changed'])
    order = ['last_changed']

    for name in tqdm(dataframes.keys()):
        print(name)

        dataframes[name] = prepare_data(dataframes[name])
        dataframes[name] = fill_data(dataframes[name], name)

        prev = pd.merge(prev, dataframes[name], on='last_changed', how='outer')

        order.append(name)

    prev = prev.sort_values(by='last_changed')
    prev = prev.reset_index(drop=True)

    return prev[order]


def get_labels(df, end_date: str):
    out_res = df[['last_changed', 'pressure', 'out_temp', 'out_hum']]
    out_res = out_res.loc[out_res['last_changed'] < end_date]

    df_labels = pd.read_csv('e643944c093a13ae980c6ba2b91c6474.csv')

    df_labels['dt_iso'] = pd.to_datetime(
        df_labels['dt_iso'],
        format='%Y-%m-%d %H:%M:%S %z UTC',
        utc=True,
    )

    df_labels = df_labels[(
        (df_labels['dt_iso'] < out_res['last_changed'].max()) &
        (df_labels['dt_iso'] > out_res['last_changed'].min())
    )]

    df_labels = df_labels[['dt_iso', 'rain_1h', 'snow_1h']]
    df_labels = df_labels.rename(columns={'dt_iso': 'last_changed'})

    out_res = out_res[
        (out_res['last_changed'].dt.minute == 0) &
        (out_res['last_changed'].dt.second == 0) &
        (out_res['last_changed'].dt.microsecond == 0)
    ]

    out_res_labelled = pd.merge(
        out_res,
        df_labels,
        on='last_changed',
        how='outer').sort_values(by='last_changed')

    return out_res_labelled


def transform_data(df, step: int, epsilon: float):
    """
    Argument df to Tabela 1(time, P, T, H, R)
    """

    df = df.copy()
    df['total percipitation'] = np.log(df['total percipitation'] + epsilon)

    df2 = generate_data(df, step)
    df2['Label'] = df2['TP t+1'].apply(lambda x: 0 if x <= np.log(epsilon) else 1)

    X2 = df2.iloc[:, 2:-1]
    y2 = df2.iloc[:, 1:2]

    sc = MinMaxScaler(feature_range=(0, 1))
    X2_scaled = sc.fit_transform(X2)
    y2_scaled = sc.fit_transform(y2)

    return df2, X2_scaled, y2_scaled, sc


def invert_transformed_data(y_pred, y_test, sc, epsilon):
    y_pred = sc.inverse_transform(y_pred)
    y_test = sc.inverse_transform(y_test)

    df_pred = pd.DataFrame(y_pred, columns=["TP t+1"])
    df_test = pd.DataFrame(y_test, columns=["TP t+1"])

    df_pred["TP t+1"] = np.exp(df_pred["TP t+1"]) - epsilon
    df_test["TP t+1"] = np.exp(df_test["TP t+1"]) - epsilon

    return df_pred, df_test


def generate_data(df, step: int):
    df = df.copy()
    df = df.rename(
        columns={
            "last_changed": "time",
            "out_hum": "Hum t0",
            "out_temp": "Temp t0",
            "pressure": "Press t0",
            "total percipitation": "TP t0",
        }
    )

    df.insert(1, column='TP t+1', value=df['TP t0'].shift(-1))

    for i in range(1, step + 1):
        for name in ['Press', 'Temp', 'Hum', 'TP']:
            df[f'{name} t-{i}'] = df[f'{name} t0'].shift(i)

    df = df.dropna().reset_index(drop=True)
    return df
