# import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.graph_objects as go
from plotly.figure_factory import create_distplot
from plotly.offline import plot as po_plot
# , download_plotlyjs
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from collections import Counter

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ... import SETTINGS


def create_histogram_plot(column_data):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Histogram(x=column_data), row=1, col=1)
    fig.update_layout(**SETTINGS['histogram_plot']['layout'])
    return po_plot(fig, config=dict(SETTINGS['histogram_plot']['plot_config']), output_type='div')


def create_pie_plot(column_data):
    data = Counter(column_data)
    count_df = pd.DataFrame.from_dict(data, orient='index', columns=['Occurrences'])

    if len(count_df) > 5:
        sorted_df = count_df.sort_values(by=SETTINGS['pie_plot']['occurrences_text'], ascending=False)
        top_5 = sorted_df.iloc[:5]
        others_sum = sorted_df.iloc[5:].sum()
        others = pd.DataFrame({SETTINGS['pie_plot']['occurrences_text']: [others_sum]}, index=['Others'])
        pie_df = pd.concat([top_5, others], ignore_index=False)
    else:
        pie_df = count_df

    fig = px.pie(pie_df, values=SETTINGS['pie_plot']['occurrences_text'], names=list(pie_df.index), hover_data=[SETTINGS['pie_plot']['occurrences_text']],
                 labels={SETTINGS['pie_plot']['occurrences_text']: SETTINGS['pie_plot']['occurrences_text']})
    fig.update_layout(**SETTINGS['pie_plot']['layout'])
    fig.update_traces(textinfo='percent+label')
    return po_plot(fig, config=dict(SETTINGS['pie_plot']['plot_config']), output_type='div')


def feature_desc_hist_array(df):
    hists_array = []

    for column in df.columns:
        if not is_numeric_dtype(df[column]) and df[column].nunique() == df[column].size:
            plot = SETTINGS['feature_desc_hist_array']['unique_values_text'] % df[column].size
        elif is_numeric_dtype(df[column]):
            plot = create_histogram_plot(df[column])
        else:
            plot = create_pie_plot(df[column])

        hists_array.append(plot)

    return hists_array

# def feature_desc_hist_array(df):
#     hists_array = []
#     for column in df.columns:
#         if not is_numeric_dtype(df[column]) and df[column].nunique() == df[column].size:
#             plot = f'{df[column].size} Unique Values'
#         elif is_numeric_dtype(df[column]):
#             fig = create_distplot([df[column]], [column], show_rug=False)
#             fig.update_layout(autosize=False, height=150, width=150, margin={'t': 5, 'l': 5, 'r': 5, 'b': 5},
#                               showlegend=False, bargap=0.05)
#             plot = po(fig, config=dict({"displayModeBar": False}), output_type='div')
#         else:
#             data = Counter(df[column])
#             count_df = pd.DataFrame.from_dict(data, orient='index', columns=['Occurrences'])
#             if df[column].nunique() > 5:
#                 # Option 1
#                 sorted_df = count_df.sort_values(by='Occurrences', ascending=False)
#                 pie_df = sorted_df.iloc[:5]
#                 pie_df = pd.concat(
#                     [pie_df, pd.DataFrame({"Occurrences": sorted_df["Occurrences"].iloc[5:].sum()}, index=['Others'])],
#                     ignore_index=False)
#
#                 # Option 2 -> deprecated
#                 # pie_df = pie_df.append(pd.DataFrame({"vals": sorted_df["vals"].iloc[5:].sum()}, index=['Others']),
#                 #                        ignore_index=False)
#
#                 # Option 3 -> concat 'others' df rather than new pd.DataFrame (fewer options so need to fix index, columns)
#                 # top_n = partial_df.sort_values('vals', ascending=False)[:5]
#                 # others = partial_df.sort_values('vals', ascending=False)[5:].sum().to_frame()
#                 # others.index = ['Others']
#                 # others.columns = ['vals']
#                 # pie_df = pd.concat([top_n, others], axis=0)
#                 # print(pie_df)
#             else:
#                 pie_df = count_df
#
#             fig = px.pie(pie_df, values='Occurrences', names=list(pie_df.index),
#                          # title='Groups',
#                          hover_data=['Occurrences'], labels={'Occurrences': 'Occurrences'})
#             fig.update_layout(autosize=False, height=170, width=150, margin={'t': 5, 'l': 5, 'r': 5, 'b': 5},
#                               showlegend=False)  # showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0)
#             fig.update_traces(textinfo='percent+label')
#             plot = po(fig, config=dict({"displayModeBar": False}), output_type='div')
#
#         hists_array.append(plot)
#     return hists_array
