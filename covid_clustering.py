# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:00:23 2020

@author: jesus
"""

import pandas as pd

covid_df = pd.read_csv("covid_19_clean_complete.csv")
covid_df = covid_df.loc[covid_df['Date'] == '5/25/20']
covid_df = covid_df.groupby('Country/Region').sum()

population_df = pd.read_csv("population.csv")
population_df = population_df[["Country/Region", "Population (2020)"]]
covid_df = covid_df.merge(right=population_df,  on="Country/Region")