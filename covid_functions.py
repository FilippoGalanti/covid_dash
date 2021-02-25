import pandas as pd
from datetime import timedelta, date, datetime
import os.path
import numpy as np

class covid:

    def create_df(self, url):
        # to check if file exists
        # self is local path, url is online location
        if os.path.exists(self):
            df = pd.read_csv(self)
            max_date = df["dateRep"].max()
            max_date = datetime.strptime(max_date, '%Y-%m-%d').date()

            if max_date + timedelta(days=1) == date.today():
                print("Existing file already updated.")
                return covid.get_data(self)
            else:
                df = covid.get_data(url)
                df = covid.adjust_column_names(df)
                df.to_csv('data.csv', index=False)
                print('File exists but is not updated. Updating file.')
                return df
        else:
            df = covid.get_data(url)
            df = covid.adjust_column_names(df)
            df.to_csv('data.csv', index=False)
            print('File does not exists. Downloading and creating file.')
            return df

    def download_data(self):
        df = pd.read_csv(self, parse_dates=['date'], dayfirst=True)
        return df.to_csv('data.csv', index=False)

    def get_data(self):
        df = pd.read_csv(self, parse_dates=[3], dayfirst=False)
        df = covid.adjust_column_names(df)
        df['Cases_per_100k'] = (df['cases']/df['popData2019'])*100000
        df['Deaths_per_100k'] = (df['deaths']/df['popData2019'])*100000
        df['Year'] = df['dateRep'].dt.year
        df['Month'] = df['dateRep'].dt.month.map("{:02}".format)
        df['week_number'] = df['dateRep'].dt.isocalendar().week.map("{:02}".format)
        df['Year_Week'] = df['Year'].astype(str) + df['week_number'].astype(str)
        df['Year_Month'] = df['Year'].astype(str) + df['Month'].astype(str)
        return df
    
    def adjust_column_names(self):
        output_df = self.rename({'date': 'dateRep', 'continent': 'continentExp',
                                 'location': 'countriesAndTerritories', 'population': 'popData2019', 'new_cases': 'cases', 'new_deaths': 'deaths'}, axis=1)
        return output_df
    
    def country_data(self, country_list=[]):
        # Cases and deaths for given country
        df_output = pd.DataFrame()
        for country in country_list:
            df_country = self[self['countriesAndTerritories'] == country][['dateRep', 'countriesAndTerritories',
                                                                           'popData2019', 'continentExp', 'cases', 'new_cases_smoothed', 'Cases_per_100k', 
                                                                           'deaths', 'new_deaths_smoothed', 'Deaths_per_100k', 'median_age']]
            df_output = df_output.append(df_country)
        return df_output

    def continent_data(self,continent_list=[]):
        # Cases and deaths for continents
        df_output = pd.DataFrame()
        for continent in continent_list:
            output_df = pd.pivot_table(self[self['continentExp'] == continent], values=[
                                       'cases', 'deaths', 'popData2019', 'new_cases_smoothed', 'new_deaths_smoothed'], index=['dateRep', 'continentExp'], aggfunc=np.sum)
            output_df['Cases_per_100k'] = (output_df['cases']/output_df['popData2019'])*100000
            output_df['Deaths_per_100k'] = (output_df['deaths']/output_df['popData2019'])*100000
            output_df = output_df.reset_index(level=['dateRep','continentExp'])
            df_output = df_output.append(output_df)
        return df_output

    def max_date(self):
        return self["dateRep"].max().date()

    def min_date(self):
        return self["dateRep"].min().date()
    
    def continent_groupby(self):
        continent_total = self.groupby('continentExp').sum()[['cases', 'deaths']]
        continent_total['Mortality'] = (continent_total['deaths']/continent_total['cases']).map(lambda n: '{:,.2%}'.format(n))
        return continent_total
    
    def country_list (self):
        return self['countriesAndTerritories'].unique()

    def continent_list (self):
        return self['continentExp'].unique()
    
    def past_dates(self, max_date):
        today = datetime.strptime(str(max_date), '%Y-%m-%d')
        week_ago = today - timedelta(days = 7)
        past_14 = today - timedelta(days = 14)
        four_weeks = today - timedelta(days = 28)
        return week_ago, past_14, four_weeks
    
    def country_trend_cases(self, country_list=[]):
        output_df = pd.DataFrame(columns=['Country', 'Today Cases', '1 Week Ago', '14 days ago', '4 weeks Ago'])
        i = 0
        if country_list == []:
            country_list = covid.country_list(self)
        for country in country_list:
            if country not in ['Cases_on_an_international_conveyance_Japan', 'Vanuatu']:
                df_country = self[self['countriesAndTerritories'] == country][['dateRep', 'cases']]
                week_ago, past_14, four_weeks = covid.past_dates(df_country)
                today_cases = df_country[df_country['dateRep'] == df_country['dateRep'].max()]['cases'].values[0]
                week_ago_cases = df_country[df_country['dateRep'] == week_ago]['cases'].values[0]
                past_14_days_cases = df_country[df_country['dateRep'] == past_14]['cases'].values[0]
                month_ago_cases = df_country[df_country['dateRep'] == four_weeks]['cases'].values[0]
                output_df.loc[i] = [country, today_cases, week_ago_cases, past_14_days_cases, month_ago_cases]
                i += 1
        output_df['Change vs LW'] = ((output_df['Today Cases'] - output_df['1 Week Ago'])/output_df['1 Week Ago'].replace({0: np.nan})*1).apply('{:.2%}'.format)
        output_df['Change vs 14 days'] = ((output_df['Today Cases'] - output_df['14 days ago'])/output_df['14 days ago'].replace({0:np.nan})*1).apply('{:.2%}'.format)
        output_df['Change vs 4 weeks'] = ((output_df['Today Cases'] - output_df['4 weeks Ago'])/output_df['4 weeks Ago'].replace({0:np.nan})*1).apply('{:.2%}'.format)
        output_df = output_df.replace(np.nan, 0, regex=True)
        output_df = output_df.round({'Change vs LW': 2, 'Change vs 14 days': 2, 'Change vs 4 weeks':2})
        return output_df
    
    def cases_by_month (self,country=[]):
        cases_pivot = pd.pivot_table(self, index='countriesAndTerritories', columns='Year_Month', values='cases', aggfunc = np.sum)
        if country == []:
            return cases_pivot
        else:
            return cases_pivot[cases_pivot.index.isin(country)]
    
    def deaths_by_month(self,country=[]):
        deaths_pivot = pd.pivot_table(self, index='countriesAndTerritories', columns='Year_Month', values='deaths', aggfunc = np.sum)
        if country == []:
            return deaths_pivot
        else:
            return deaths_pivot[deaths_pivot.index.isin(country)]
        
    def cases_by_week(self, country=[]):
        cases_pivot = pd.pivot_table(self, index='countriesAndTerritories', columns='Year_Week', values='cases', aggfunc=np.sum)
        if country == []:
            return cases_pivot
        else:
            return cases_pivot[cases_pivot.index.isin(country)]
    
    def deaths_by_week(self,country=[]):
        deaths_pivot = pd.pivot_table(self, index='countriesAndTerritories', columns='Year_Week', values='deaths', aggfunc = np.sum)
        if country == []:
            return deaths_pivot
        else:
            return deaths_pivot[deaths_pivot.index.isin(country)]

    def cases_by_month_continents(self, continents=[]):
        cases_pivot = pd.pivot_table(self, index='continentExp', columns='Year_Month', values='cases', aggfunc=np.sum)
        if continents == []:
            return cases_pivot
        else:
            return cases_pivot[cases_pivot.index.isin(continents)]

    def deaths_by_month_continents(self,continents=[]):
        deaths_pivot = pd.pivot_table(self, index='continentExp', columns='Year_Month', values='deaths', aggfunc = np.sum)
        if continents == []:
            return deaths_pivot
        else:
            return deaths_pivot[deaths_pivot.index.isin(continents)]
        
    def cases_by_week_continents(self, continents=[]):
        cases_pivot = pd.pivot_table(self, index='continentExp', columns='Year_Week', values='cases', aggfunc=np.sum)
        if continents == []:
            return cases_pivot
        else:
            return cases_pivot[cases_pivot.index.isin(continents)]
    
    def deaths_by_week_continents(self,continents=[]):
        deaths_pivot = pd.pivot_table(self, index='continentExp', columns='Year_Week', values='deaths', aggfunc = np.sum)
        if continents == []:
            return deaths_pivot
        else:
            return deaths_pivot[deaths_pivot.index.isin(continents)]

    def cases_for_100k(self, country=[]):
        cases_pivot = pd.pivot_table(self, index='countriesAndTerritories', columns='dateRep', values='Cases_per_100k')
        if country == []:
            return cases_pivot
        else:
            return cases_pivot[cases_pivot.index.isin(country)]
    
    def mortality_df(self):
        df = pd.pivot_table(self, index=['countriesAndTerritories', 'continentExp', 'popData2019', 'median_age'], values=[
                            'cases', 'deaths'], aggfunc=np.sum)
        df = df.reset_index(level=['continentExp', 'popData2019', 'median_age'])
        df['deaths_per_million'] = (df['deaths']/df['popData2019'])*1000000
        df['cases_per_million'] = (df['cases']/df['popData2019'])*1000000
        df['mortality_rate'] = df['deaths']/df['cases']
        return df


