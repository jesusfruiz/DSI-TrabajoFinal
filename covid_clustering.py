# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:00:23 2020

@author: jesus
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np

# 1. Auxiliar Methods
def bic(K, cidx, X):
    k = 0
    P = len(X.iloc[1,:])
    N = len(X)
    sigma_j = dict()
    xi = []
    while k < K:
        suma = 0
        group_k = list(filter(lambda x: cidx[x] == k, range(len(cidx))))
        sigma = dict()
        sigma_j[k] = dict()
        j = 0
        while j < P:    
            sigma[j] = statistics.stdev(X.iloc[:,1])**2
            if len(group_k) < 2:
                sigma_j[k][j] = 0
            else:                
                sigma_j[k][j] = statistics.stdev(X.iloc[group_k,1])**2
            suma = suma + 0.5 * math.log(sigma[j] + sigma_j[k][j])    
            j+=1
        xi.append(-1 * len(group_k) * suma)
        k+=1
    return -2*sum(xi)+2*K*P*math.log(N) 

def plot_3d(x, y, z, labels=None, cmap=None):
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(x, y, z, c=labels, cmap=cmap)
    threedee.set_xlabel('Confirmed')
    threedee.set_ylabel('Deaths')
    threedee.set_zlabel('Recovered')
    plt.show()    

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height,3)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plotDataByGroupRate(labels, confirmed, deaths, recovered):
    x = np.arange(len(labels))  # the label locations
    width = 0.33  # the width of the bars
    
    fig, ax = plt.subplots()
    
    rects1 = ax.bar(x, confirmed, width, label='Confirmed')
    rects3 = ax.bar(x, recovered, width, label='Recovered')
    rects2 = ax.bar(x, deaths, width, label='Deaths')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('People')
    ax.set_title('Confirmed, deaths and recovered people by group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
        
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    
    fig.tight_layout()
    
    plt.show()


# 2. Data reading and preprocessing
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

# 3. Plotting Data
plot_3d(covid_df['Confirmed'], covid_df['Deaths'], covid_df['Recovered'])

plt.scatter(covid_df['Confirmed'], covid_df['Deaths'])
plt.plot()
plt.xlim(0, 0.016)
plt.ylim(0, 0.00045)
plt.xlabel('Confirmed')
plt.ylabel('Deaths')
plt.show()

# 4. Ideal cluster number
K=2
BIC = []

while K <= 10:
    kmeans = KMeans(n_clusters=K, init='random', n_init=20)
    kmeans.fit(covid_df.drop(['Country/Region'], axis=1))
    BIC.append(bic(K, kmeans.labels_, covid_df))
    K += 1

X = list(range(2, 11))

plt.scatter(X, BIC)
plt.plot()
plt.show()

K = 3 # Three is the ideal number

# 5 Outliers Detection: Jackknife
data = covid_df.drop(['Country/Region'], axis=1)
kmeans = KMeans(n_clusters=K, init='random', n_init=40)
SSE = dict()
for i in range(len(data)):
    data_aux = data.drop(i)
    kmeans.fit(data_aux)
    SSE[i] = kmeans.inertia_

sigma=statistics.stdev(SSE.values())
mu=statistics.mean(SSE.values())
umbral=2;

outliers_index = []
for i in range(len(covid_df)):
    if abs(SSE[i]-mu)>umbral*sigma:
        outliers_index.append(i);

out_label = []
for i in range(len(covid_df)):
    if i in outliers_index:
        col = 'r'
        out_label.append(1)
    else:
        col = 'w'
        out_label .append(2)
        
plot_3d(covid_df['Confirmed'], covid_df['Deaths'], covid_df['Recovered'], labels=out_label, cmap='brg')

outliers = []
print("Los outliers son:")
for i in outliers_index:
    print(i)
    print(covid_df.iloc[i,:])
    print("")
    outliers.append(covid_df.iloc[i,:])
    
covid_df = covid_df.drop(outliers_index) #Remove the outlier from data
    
# print data without outlier
plot_3d(covid_df['Confirmed'], covid_df['Deaths'], covid_df['Recovered'])

# 6. Perform Clustering
gm = GaussianMixture(n_components=K, max_iter=100, tol=0.000001, covariance_type='spherical')

gm.fit(covid_df.drop(['Country/Region'], axis=1))
labels = gm.predict(covid_df.drop(['Country/Region'], axis=1))


plot_3d(covid_df['Confirmed'], covid_df['Deaths'], covid_df['Recovered'], labels=labels, cmap='brg')

# 7. Analysis Data
covid_df['labels'] = labels

groups = {}
representatives = {}

for i in range(K):
    groups[i] = covid_df.loc[covid_df['labels'] == i]

for i in range(K):
    print(f"Group {i}:")
    print(groups[i].mean())
    representatives[i] = groups[i].mean()




# plot groups data graphs (deaths and revoeries rate by group)

labels = ['G0', 'G1', 'G2']
confirmed = [100, 100, 100]
deaths = [groups[0].mean()['Deaths']/groups[0].mean()['Confirmed']*100, 
          groups[1].mean()['Deaths']/groups[1].mean()['Confirmed']*100, 
          groups[2].mean()['Deaths']/groups[2].mean()['Confirmed']*100]
recovered = [groups[0].mean()['Recovered']/groups[0].mean()['Confirmed']*100, 
             groups[1].mean()['Recovered']/groups[1].mean()['Confirmed']*100, 
             groups[2].mean()['Recovered']/groups[2].mean()['Confirmed']*100]

plotDataByGroupRate(labels, confirmed, deaths, recovered)


