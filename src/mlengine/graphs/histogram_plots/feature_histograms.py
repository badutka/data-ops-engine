# import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.graph_objects as go
from plotly.figure_factory import create_distplot
from plotly.offline import plot as po, download_plotlyjs
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from collections import Counter


def feature_desc_hist_array(df):
    hists_array = []
    for column in df.columns:
        if not is_numeric_dtype(df[column]) and df[column].nunique() == df[column].size:
            plot = f'{df[column].size} Unique Values'
        elif is_numeric_dtype(df[column]):
            fig = create_distplot([df[column]], [column], show_rug=False)
            fig.update_layout(autosize=False, height=150, width=150, margin={'t': 5, 'l': 5, 'r': 5, 'b': 5},
                              showlegend=False, bargap=0.05)
            plot = po(fig, config=dict({"displayModeBar": False}), output_type='div')
        else:
            data = Counter(df[column])
            count_df = pd.DataFrame.from_dict(data, orient='index', columns=['Occurrences'])
            if df[column].nunique() > 5:
                # Option 1
                sorted_df = count_df.sort_values(by='Occurrences', ascending=False)
                pie_df = sorted_df.iloc[:5]
                pie_df = pd.concat(
                    [pie_df, pd.DataFrame({"Occurrences": sorted_df["Occurrences"].iloc[5:].sum()}, index=['Others'])],
                    ignore_index=False)

                # Option 2 -> deprecated
                # pie_df = pie_df.append(pd.DataFrame({"vals": sorted_df["vals"].iloc[5:].sum()}, index=['Others']),
                #                        ignore_index=False)

                # Option 3 -> concat 'others' df rather than new pd.DataFrame (fewer options so need to fix index, columns)
                # top_n = partial_df.sort_values('vals', ascending=False)[:5]
                # others = partial_df.sort_values('vals', ascending=False)[5:].sum().to_frame()
                # others.index = ['Others']
                # others.columns = ['vals']
                # pie_df = pd.concat([top_n, others], axis=0)
                # print(pie_df)
            else:
                pie_df = count_df

            fig = px.pie(pie_df, values='Occurrences', names=list(pie_df.index),
                         # title='Groups',
                         hover_data=['Occurrences'], labels={'Occurrences': 'Occurrences'})
            fig.update_layout(autosize=False, height=170, width=150, margin={'t': 5, 'l': 5, 'r': 5, 'b': 5},
                              showlegend=False)  # showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0)
            fig.update_traces(textinfo='percent+label')
            plot = po(fig, config=dict({"displayModeBar": False}), output_type='div')

        hists_array.append(plot)
    return hists_array
