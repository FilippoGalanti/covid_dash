import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from dash.dependencies import Input, Output, State
import pandas as pd
import matplotlib.pyplot as plt
from covid_functions import covid
import dash_styles
from datetime import datetime, timedelta
import seaborn as sns
import dash_table
import numpy as np

url_new = r'https://covid.ourworldindata.org/data/owid-covid-data.csv'
local_url = r'C:\Users\Filippo Galanti\Desktop\Python Course\02 Projects\Covid_v2\data.csv'

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create the color Palette
color_list = sns.color_palette('hls', 8)
colorscale = color_list.as_hex()

# get and adjust data
df = covid.create_df(local_url,url_new)
countries, continents = covid.country_list(df), covid.continent_list(df)

# set / create key parameters
type, display_as, geo_choice, graph_type = ['Cases', 'Deaths'], ['Absolute', 'For 100k pop.'], ['Countries', 'Continents'], ['Daily Trend', 'Total Cases', 'Mortality']
min_date, max_date = covid.min_date(df), covid.max_date(df)

# check if there are data for max_date
updated_continent_list = df[df['dateRep'].dt.strftime('%Y-%m-%d') == str(max_date)]
if len(updated_continent_list['countriesAndTerritories'].unique()) < 100:
    max_date = max_date - timedelta(days=1)

controls = dbc.FormGroup([
    html.P('Countries / Continents:', style={'textAlign': 'center'}),
    dcc.Dropdown(id='geo-choice',
                 options=[{'label': i, 'value': i} for i in geo_choice],
                value='Countries'),
    html.P('Countries:', style = {'textAlign': 'center'}),
    dcc.Dropdown(id='Countries',
                 options=[{'label': i, 'value': i} for i in countries],
                 value=['Italy'],
                 multi=True),
    html.Br(),
    html.P('Cases / Deaths:', style={'textAlign': 'center'}),
    dcc.Dropdown(id='type',
                 options=[{'label': i, 'value': i} for i in type],
                 value='Cases'),
    html.Br(),
    html.P('Display as:', style={'textAlign':'center'}),
    dcc.Dropdown(id='display-as',
                 options=[{'label': i, 'value': i} for i in display_as],
                 value='Absolute'),
    html.Br(),
    html.P('Graph Type:', style={'textAlign': 'center'}),
    dcc.Dropdown(id='graph-type',
                 options=[{'label': i, 'value': i} for i in graph_type],
                 value='Daily Trend'),
    html.Br(),
    html.P('Period:', style={'textAlign': 'center'}),
    dcc.DatePickerRange(id='date-picker',
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        start_date=min_date,
                        end_date=max_date,
                        calendar_orientation='horizontal',
                        number_of_months_shown=2,
                        show_outside_days=True,
                        day_size=25,
                        month_format="MMM, YY"
                        ),
    html.Br(),
    html.P('Show Period:', style={'textAlign': 'center'}),
    dbc.RadioItems(id='last-n-weeks',
                   options=[
                       {'label': '  Last Month', 'value': 4},
                       {'label': '  Last 2 Months', 'value': 8},
                       {'label': '  Last 3 Months', 'value': 12},
                       {'label': '  Last 4 Months', 'value': 16}],
                   labelStyle={'display': 'block'},
                   labelCheckedStyle={"color": "red"}
                ),                    
    html.P(),
    dbc.Button(
        id='submit-button',
        n_clicks=0,
        children='Update',
        color='primary',
        block=True)
])

content_first_row = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody(
        children=[
        html.H4('Last Date'),
        html.H5(id='today-cases')
        ]))),
    dbc.Col(dbc.Card(dbc.CardBody(
        children=[
        html.H4('Change vs LW'),
        html.H5(id='change-1-week')
        ]))),
    dbc.Col(dbc.Card(dbc.CardBody(
        children=[
        html.H4('Change vs 14 Days'),
        html.H5(id='change-14-days')
        ]))),
    dbc.Col(dbc.Card(dbc.CardBody(
        children=[
        html.H4('Change vs 4WK'),
        html.H5(id='change-4-weeks')
        ])))
])

content_second_row = dash_table.DataTable(
    id = 'table',
    style_cell={'textAlign': 'center', 'padding': '5px'},
    style_as_list_view=True,
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold'
    },
)

content_third_row = dbc.Row([
    dcc.Graph(
        id='main-graph',
        config={
            'displayModeBar':False
        }
    )
])

content_fourth_row = dbc.Row([])

content = html.Div([
        html.H2('Covid-19 Dashboard', style=dash_styles.TEXT_STYLE),
        html.Hr(),
        content_first_row,
        content_second_row,
        content_third_row,
        content_fourth_row
    ],style=dash_styles.CONTENT_STYLE)

sidebar = html.Div([
    html.H2('Filters', style=dash_styles.TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=dash_styles.SIDEBAR_STYLE)

app.layout = html.Div([sidebar, content])

@app.callback(Output('date-picker', 'start_date'),
              [Input('last-n-weeks','value')])
def update_start_date(past_weeks):
    today = max_date
    start_date = today - timedelta(days = int(past_weeks)*7)
    return start_date

@app.callback(
    Output('Countries', 'options'),
    [Input('geo-choice', 'value')])
def dropdwon_list (choice):
    if choice == 'Countries':
        return [{'label': i, 'value': i} for i in countries]
    elif choice == 'Continents':
        return [{'label': i, 'value': i} for i in continents]

@app.callback([Output('main-graph', 'figure'),
               Output('today-cases', 'children'),
               Output('change-1-week', 'children'),
               Output('change-14-days', 'children'),
               Output('change-4-weeks', 'children'),
               Output('table', 'data'),
               Output('table', 'columns')],
              [Input('submit-button', 'n_clicks')],
              [State('geo-choice', 'value'),
               State('Countries', 'value'),
               State('type', 'value'),
               State('display-as', 'value'),
               State('graph-type', 'value'),
               State('date-picker', 'start_date'),
               State('date-picker', 'end_date')
               ])
def update_graph(n_clicks, geo_choice, locations, type, display_as, graph_type, start_date, end_date):

    data_to_display, i, output_table = [], 0, pd.DataFrame(columns=['Location', f'{end_date}', '1 week ago', 'Delta vs 1 wk',
                                                            '14 days ago', 'Delta vs 14 days', '4 weeks ago', 'Delta vs 4 weeks'])

    if geo_choice == 'Continents':
        temporary_df = covid.continent_data(df, locations)
        main_column = 'continentExp'
    elif geo_choice == 'Countries':
        temporary_df = covid.country_data(df, locations)
        main_column = 'countriesAndTerritories'

    if type == 'Cases' and display_as == 'Absolute':
        target_value = 'new_cases_smoothed'
        target_value_display = 'cases'
    elif type == 'Cases' and display_as == 'For 100k pop.':
        target_value = 'Cases_per_100k'
        target_value_display = 'cases'
    elif type == 'Deaths' and display_as == 'Absolute':
        target_value = 'new_deaths_smoothed'
        target_value_display = 'deaths'
    else:
        target_value = 'Deaths_per_100k'
        target_value_display = 'deaths'

    # Daily Graph
    if graph_type == 'Daily Trend':

        for element in locations:
            df_loc = temporary_df[temporary_df[main_column] == element][['dateRep', target_value, target_value_display]]
            df_loc.set_index('dateRep', inplace = True)
            # update graph traces
            df_loc = df_loc[(df_loc.index >= start_date) & (df_loc.index <= end_date)]
            data_to_display.append({'x': df_loc.index, 'y': df_loc[target_value], 'name': element, 'line': dict(color=colorscale[i], width=2), 'mode':'lines'})
            # prepare data for table
            today_cases = int(df_loc[df_loc.index == end_date][target_value_display].sum())
            data_1_week = int(df_loc[df_loc.index == covid.past_dates(df_loc, end_date)[0]][target_value_display].sum())
            data_14_days = int(df_loc[df_loc.index == covid.past_dates(df_loc, end_date)[1]][target_value_display].sum())
            data_4_weeks = int(df_loc[df_loc.index == covid.past_dates(df_loc, end_date)[2]][target_value_display].sum())
            
            today_format_cases = '{:,}'.format(today_cases)
            change_1_week = '{:.2%}'.format((today_cases-data_1_week)/data_1_week)
            change_14_days = '{:.2%}'.format((today_cases-data_14_days)/data_14_days)
            change_4_weeks = '{:.2%}'.format((today_cases-data_4_weeks)/data_4_weeks)

            output_table.loc[i] = [element, today_format_cases, '{:,}'.format(data_1_week), change_1_week, 
                                '{:,}'.format(data_14_days), change_14_days, 
                                '{:,}'.format(data_4_weeks), change_4_weeks]

            i += 1

        output_figure = {'data': data_to_display, 'layout': {'legend': {'orientation': "h", 'xanchor' : 'center', 'x':0.5}, 'autosize': False, 'height': 600, 'width': 1150}}

        today_cases = temporary_df[temporary_df['dateRep'] == end_date][target_value_display].sum()
        data_1_week = temporary_df[temporary_df['dateRep'] == covid.past_dates(temporary_df, end_date)[0]][target_value_display].sum()
        data_14_days = temporary_df[temporary_df['dateRep'] == covid.past_dates(temporary_df, end_date)[1]][target_value_display].sum()
        data_4_weeks = temporary_df[temporary_df['dateRep'] == covid.past_dates(temporary_df, end_date)[2]][target_value_display].sum()

        today_total_cases = '{:,}'.format(int(temporary_df[temporary_df['dateRep'] == end_date][target_value_display].sum()))
        change_1_week = '{:.2%}'.format((today_cases-data_1_week)/data_1_week)
        change_14_days = '{:.2%}'.format((today_cases-data_14_days)/data_14_days)
        change_4_weeks = '{:.2%}'.format((today_cases-data_4_weeks)/data_4_weeks)

        data_table = output_table.to_dict('records')
        columns_output = [
            {"name": i, "id": i} for i in (output_table.columns)
        ]
    
    # Total Graph
    elif graph_type == 'Total Cases':
        if display_as == 'Absolute':
            if main_column == 'continentExp':
                countries_list = df[df['continentExp'].isin(locations)]['countriesAndTerritories'].unique()
                temporary_df = covid.country_data(df, countries_list)
                df_loc = temporary_df[temporary_df['countriesAndTerritories'].isin(countries_list)][['dateRep', 'countriesAndTerritories', 'popData2019', 'median_age', target_value]]
                df_loc.set_index('dateRep', inplace = True)
                df_loc = df_loc[(df_loc.index >= start_date) & (df_loc.index <= end_date)]
                df_loc_pivot = pd.pivot_table(df_loc, values=target_value, index = ['countriesAndTerritories', 'median_age'], aggfunc = np.sum)
                df_loc_pivot = df_loc_pivot.reset_index(level=['median_age'])
                df_loc_pivot.sort_values(by=target_value, ascending=True, inplace=True)
                trace_1 = go.Bar(x=df_loc_pivot.index, y=df_loc_pivot[target_value], 
                        marker={'color':df_loc_pivot['median_age'],
                        'colorscale':'redor',
                        'showscale':True,
                        'colorbar':{'title': 'Median Age'}})
                data_to_display.append(trace_1)
                
            else:
                df_loc = temporary_df[temporary_df[main_column].isin(locations)][['dateRep', main_column, target_value]]
                df_loc.set_index('dateRep', inplace = True)
                df_loc = df_loc[(df_loc.index >= start_date) & (df_loc.index <= end_date)]
                df_loc_pivot = pd.pivot_table(df_loc, values=target_value, index = main_column, aggfunc = np.sum)
                df_loc_pivot.sort_values(by=target_value, ascending=True, inplace=True)
                trace_1 = go.Bar(x=df_loc_pivot.index, y=df_loc_pivot[target_value], marker={'color': '#191970'})
                data_to_display.append(trace_1)

            output_figure = {'data': data_to_display, 'layout': go.Layout(xaxis_tickangle=-45)}
        
        else:
            if type == 'Cases':
                target_value = 'cases'
            else:
                target_value = 'deaths'

            if main_column == 'continentExp':

                countries_list = df[df['continentExp'].isin(locations)]['countriesAndTerritories'].unique()
                temporary_df = covid.country_data(df, countries_list)
                df_loc = temporary_df[temporary_df['countriesAndTerritories'].isin(countries_list)][['dateRep', 'countriesAndTerritories', 'popData2019', 'median_age', target_value]]
                df_loc.set_index('dateRep', inplace = True)
                df_loc = df_loc[(df_loc.index >= start_date) & (df_loc.index <= end_date)]
                df_loc_pivot = pd.pivot_table(df_loc, values=target_value, index=['countriesAndTerritories', 'popData2019', 'median_age'], aggfunc=np.sum)
                df_loc_pivot = df_loc_pivot.reset_index(level=['popData2019', 'median_age'])
                df_loc_pivot['per_100k_pop'] = (df_loc_pivot[target_value]/df_loc_pivot['popData2019'])*100000
                df_loc_pivot.sort_values(by='per_100k_pop', ascending=True, inplace=True)
                trace_1 = go.Bar(x=df_loc_pivot.index, y=df_loc_pivot['per_100k_pop'],
                                 marker={'color': df_loc_pivot['median_age'],
                                         'colorscale': 'redor',
                                         'showscale': True,
                                         'colorbar': {'title': 'Median Age'}})
                data_to_display.append(trace_1)
                
            else:
                df_loc = temporary_df[temporary_df[main_column].isin(locations)][['dateRep', 'popData2019', main_column, target_value]]
                df_loc.set_index('dateRep', inplace = True)
                df_loc = df_loc[(df_loc.index >= start_date) & (df_loc.index <= end_date)]
                df_loc_pivot = pd.pivot_table(df_loc, values=target_value, index = [main_column,'popData2019'] , aggfunc = np.sum)
                df_loc_pivot = df_loc_pivot.reset_index(level=['popData2019'])
                df_loc_pivot['per_100k_pop'] = (df_loc_pivot[target_value]/df_loc_pivot['popData2019'])*100000
                df_loc_pivot.sort_values(by=target_value, ascending=True, inplace=True)
                trace_1 = go.Bar(x=df_loc_pivot.index, y=df_loc_pivot['per_100k_pop'], marker={'color': '#191970'})
                data_to_display.append(trace_1)

            output_figure = {'data': data_to_display, 'layout': go.Layout(xaxis_tickangle=-45)}

        today_total_cases = 'N/A'
        change_1_week = 'N/A'
        change_14_days = 'N/A'
        change_4_weeks = 'N/A'
        data_table = []
        columns_output = []

    # Mortality Graph
    else:
        df_mortality = covid.mortality_df(df)
        i = 0
        symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'pentagon',
                   'hexagram', 'star', 'diamond', 'hourglass', 'bowtie', 'asterisk', 'hash', 'y', 'line']

        if geo_choice == 'Continents':
            target_list = df[df['continentExp'].isin(locations)]['countriesAndTerritories'].unique()
        else:
            target_list = locations

        for continent in continents:
            df_mortality_continent = df_mortality[df_mortality['continentExp'] == continent]
            trace = go.Scatter(
                x=df_mortality_continent['deaths_per_million'],
                y=df_mortality_continent['cases_per_million'],
                mode='markers+text',
                name = continent,
                #text=[i for i in df_mortality_continent.index if i in target_list],
                #textposition='top center',
                hovertext=[
                    f'{i} / Rate {(df_mortality_continent.loc[i, "mortality_rate"]):.2%}' for i in df_mortality_continent.index],
                hoverinfo = 'text',
                marker_symbol=symbols[i],
                marker=dict(
                    size=df_mortality_continent['median_age']/3,
                    color=df_mortality_continent['mortality_rate'],
                    colorscale='YlOrRd',
                    reversescale=False,
                    cmax=0.05,
                    cmin=0,
                    showscale=True,
                    line={'width':1}
                ))
            i += 1 
            data_to_display.append(trace)

        output_figure = {'data': data_to_display,
                         'layout': go.Layout(yaxis=dict(type='log', title='Cases per million (log)', dtick= 1),
                                             xaxis=dict(type='log', title='Deaths per million (log)', dtick= 1),
                                             hovermode='closest',
                                             hoverlabel=dict(
                                                bgcolor="white",
                                                font_size=16,
                                                font_family="Calibri Light"),
                                             legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1)
                                            )}

        today_total_cases = 'N/A'
        change_1_week = 'N/A'
        change_14_days = 'N/A'
        change_4_weeks = 'N/A'
        data_table = []
        columns_output = []

    return output_figure, today_total_cases, change_1_week, change_14_days, change_4_weeks, data_table, columns_output

if __name__ == '__main__':
    app.run_server()


