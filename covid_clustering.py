# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:00:23 2020

@author: jesus
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

covid_df = pd.read_csv("covid_19_clean_complete.csv")
covid_df = covid_df.loc[covid_df['Date'] == '5/25/20']
covid_df = covid_df.groupby('Country/Region').sum()

population_df = pd.read_csv("population.csv")
population_df = population_df[["Country/Region", "Population (2020)"]]
covid_df = covid_df.merge(right=population_df,  on="Country/Region")

covid_df['Confirmed'] = covid_df['Confirmed']/covid_df['Population (2020)']
covid_df['Deaths'] = covid_df['Deaths']/covid_df['Population (2020)']
covid_df['Recovered'] = covid_df['Recovered']/covid_df['Population (2020)']

covid_df = covid_df.drop(['Lat', 'Long', 'Population (2020)'], axis=1)


threedee = plt.figure().gca(projection='3d')
threedee.scatter(covid_df['Confirmed'], covid_df['Deaths'], covid_df['Recovered'])
threedee.set_xlabel('Confirmed')
threedee.set_ylabel('Deaths')
threedee.set_zlabel('Recovered')
plt.show()


plt.scatter(covid_df['Confirmed'], covid_df['Deaths'])
plt.plot()
plt.xlim(0, 0.016)
plt.ylim(0, 0.00045)
plt.xlabel('Confirmed')
plt.ylabel('Deaths')
plt.show()




