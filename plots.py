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

# #

# %load_ext autoreload
# %autoreload 2

"""The file contains scripts for generating plots."""

# # Importy

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots

from utils import get_labels, prepare_dataframes, read_csv

# # Przetwarzanie danych

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

df.describe()

# # Występowanie opadów w czasie

# +
fig = px.bar(
    df,
    x="last_changed",
    y="total percipitation",
    color_discrete_sequence=['#0000ff'],
)

fig.update_layout(plot_bgcolor='#f2fbfa')

fig.update_xaxes(
    title_text="Czas",
    showline=True,
    linewidth=1,
    linecolor='black',
    ticks='outside',
)

fig.update_yaxes(
    title_text="Suma opadów [mm]",
    tickmode='linear',
    tick0=0,
    dtick=2.5,
    ticks='outside',
    linewidth=1,
    linecolor='black',
)

fig.show()
# -

# # Histogram sumy opadów

# +
fig = px.histogram(df, x=df['total percipitation'], nbins = 100)

fig.update_layout(yaxis_type='log')

fig.update_xaxes(
    title='Suma opadów [mm]',
    showline=True,
    linewidth=0.5,
    linecolor='black',
    tickmode='linear',
    ticks="outside",
)

fig.update_yaxes(
    title='Liczba wystąpień',
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.update_traces(marker=dict(line=dict(width=1, color='Black')))
#fig.update_xaxes(range=[0, df['total percipitation'].max()+0.1])


#new_num_bins = 31
#fig.update_traces(nbinsx=new_num_bins)

fig.show()
# -

df['total percipitation'].max()

df['total percipitation'].min()

# +
#print (list(df['total percipitation'].sort_values()))
# -

# # Pomiar temperatury w czasie

df['out_temp'].max()

df['out_temp'].min()

# +
fig = px.line(
    df,
    x="last_changed",
    y="out_temp",
    color_discrete_sequence=['red'],
)

fig.update_xaxes(
    title='Czas',
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.update_yaxes(
    title='Temperatura [°C]',
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.show()
# -

# # Pomiar wilgotności w czasie

df['out_hum'].max()

df['out_hum'].min()

# +
fig = px.line(
    df,
    x="last_changed",
    y="out_hum",
    color_discrete_sequence=['#1f0daa'],
)

fig.update_xaxes(
    title='Czas',
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.update_yaxes(
    title='Wilgotność [%]',
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.show()
# -

# # Pomiar ciśnienia w czasie

df['pressure'].max()

df['pressure'].min()

# +
fig = px.line(
    df,
    x="last_changed",
    y="pressure",
    color_discrete_sequence=['#7d8b94'],
)

fig.update_xaxes(
    title='Czas',
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.update_yaxes(
    title='Ciśnienie [hPa]',
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.show()

# +
fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

fig.add_trace(
    go.Scatter(
        x=df["last_changed"],
        y=df["out_temp"],
        mode="lines",
        name="Temperature",
        line=dict(color='red'),
    ),
    row=1,
    col=1,
)

fig.update_yaxes(
    title_text="Temperature [°C]",
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=df["last_changed"],
        y=df["out_hum"],
        mode="lines",
        name="Humidity",
        line=dict(color='#1f0daa'),
    ),
    row=2,
    col=1,
)

fig.update_yaxes(
    title_text="Humidity [%]",
    row=2,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df["last_changed"],
        y=df["pressure"],
        #color_discrete_sequence=['#7d8b94'],
        name='Pressure',
    ),
    row=3,
    col=1,
)

fig.update_yaxes(
    title_text="Pressure [hPa]",
    row=3,
    col=1,
)

fig.update_xaxes(row=1, col=1, showline=True, linewidth=0.5, linecolor='black', ticks="outside")
fig.update_xaxes(row=2, col=1, showline=True, linewidth=0.5, linecolor='black', ticks="outside")
fig.update_xaxes(row=3, col=1, showline=True, linewidth=0.5, linecolor='black', ticks="outside")


fig.update_layout(height=600)
fig.show()
# -

nowe_nazwy = {
    'last_changed': 'Czas',
    'pressure': 'Ciśnienie',
    'out_temp': 'Temperatura',
    'out_hum': 'Wilgotność',
    'total percipitation': 'Suma opadów',
}

df = df.rename(columns=nowe_nazwy)


corr = df.corr()
#plt.figure(figsize = (12,12))
_ = sns.heatmap(corr, annot = True)
plt.savefig("heatmapa.png", bbox_inches='tight')

# # Box plot temperatury

df['week_number'] = df['last_changed'].dt.isocalendar().week

df

# +
fig = px.box(df, x="week_number", y="out_temp", color_discrete_sequence=['red'],)
fig.update_xaxes(type='category')  # Ustawienie osi x jako kategorialnej
fig.update_yaxes(
    title_text="Temperatura [°C]",
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.update_xaxes(
    title_text="Numer tygodnia",
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.show()

#agregować dla
# -

# # Box plot wilgotności

# +
fig = px.box(df, x="week_number", y="out_hum", color_discrete_sequence=['#1f0daa'])
fig.update_xaxes(type='category')
fig.update_yaxes(
    title_text="Wilgotność [%]",
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.update_xaxes(
    title_text="Numer tygodnia",
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.show()
# -

# # Box plot ciśnienia

# +
fig = px.box(df, x="week_number", y="pressure", color_discrete_sequence=['#7d8b94'])
fig.update_xaxes(type='category')
fig.update_yaxes(
    title_text="Ciśnienie [hPa]",
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)

fig.update_xaxes(
    title_text="Numer tygodnia",
    showline=True,
    linewidth=0.5,
    linecolor='black',
    ticks="outside",
)
fig.show()
