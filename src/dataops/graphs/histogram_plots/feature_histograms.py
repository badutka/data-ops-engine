from collections import Counter
import pandas as pd

import plotly.express as px
from plotly.offline import plot as po_plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dataops import settings
from dataops.data_manager import dataframe


def create_histogram_plot(column_data, output='fig'):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Histogram(x=column_data), row=1, col=1)
    fig.update_traces(name=column_data.name)
    if output == 'fig':
        return fig
    return po_plot(fig, config=dict(settings.common.histogram_plot['plot_config']), output_type='div')


def create_pie_plot(column_data, others=5, output='fig'):
    text = settings.common.pie_plot['occurrences_text']
    count_df = pd.DataFrame.from_dict(Counter(column_data), orient='index', columns=[text])

    if len(count_df) > others:
        pie_df = dataframe.group_below_top_n(count_df, others, text)
    else:
        pie_df = count_df

    fig = px.pie(pie_df, values=text, names=list(pie_df.index), hover_data=[text],
                 labels={text: text})

    fig.update_traces(textinfo='percent+label')

    if output == 'fig':
        return fig

    return po_plot(fig, config=dict(settings.common.pie_plot['plot_config']), output_type='div')


def feature_desc_hist_array(df, opt_cat='pie'):
    plots_array = []
    titles_array = []

    for column in df.columns:
        if not dataframe.is_numeric(df[column]) and df[column].nunique() == df[column].size:
            plot = create_histogram_plot(df[column])
            titles_array.append(f'{df[column].size} unique values.')
        elif dataframe.is_numeric(df[column]):
            plot = create_histogram_plot(df[column])
            titles_array.append(f'Histogram of {column}.')
        else:
            if opt_cat == 'pie':
                plot = create_pie_plot(df[column])
            else:
                plot = create_histogram_plot(df[column])
            titles_array.append(f'Pie chart of {column}.')

        plots_array.append(plot)

    # Determine the number of rows and columns in the grid
    num_rows = 5
    num_cols = len(plots_array) // num_rows + (len(plots_array) % num_rows > 0)

    subplot_types = ['pie' if plot.data and isinstance(plot.data[0], go.Pie) else 'xy' for plot in plots_array]

    combined_fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        specs=[[{"type": subplot_types[i * num_cols + j]} if i * num_cols + j < len(subplot_types) else {"type": "xy"} for j in range(num_cols)] for i in range(num_rows)],
        subplot_titles=titles_array
    )

    # Add plots to subplots
    for i, plot in enumerate(plots_array):
        col_num = (i % num_cols) + 1
        row_num = (i // num_cols) + 1
        for trace in plot.data:
            combined_fig.add_trace(trace, row=row_num, col=col_num)

    # Update layout and show the combined figure
    combined_fig.update_layout(**settings.common.histogram_plot['layout'])
    combined_fig.show()
