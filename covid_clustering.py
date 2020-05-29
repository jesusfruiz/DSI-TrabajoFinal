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

#2.1 Calculo número ideal de clusters
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

K = 3

#2.2 Detección y Eliminación de Outliers: Jackknife
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

outliers = []
for i in range(len(covid_df)):
    if abs(SSE[i]-mu)>umbral*sigma:
        outliers.append(i);

#estimator = PCA (n_components = 2)
#X_pca = estimator.fit_transform(data)
#print(estimator.explained_variance_ratio_) 
X_pca = covid_df.drop(["Deaths"], axis = 1)

for i in range(len(X_pca)):
    if i in outliers:
        col = 'r'
    else:
        col = 'w'
        
    plt.plot(X_pca.iloc[i, 0], X_pca.iloc[i, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
plt.scatter(X_pca.iloc[:,0], X_pca.iloc[:,1])
plt.show()

outlier = []
print("Los outliers son:")
for i in outliers:
    print(i)
    print(covid_df.iloc[i,:])
    print("")
    outlier.append(covid_df.iloc[i,:])
    covid_df = covid_df.drop(i) #Remove the outlier from data
    
threedee = plt.figure().gca(projection='3d')
threedee.scatter(covid_df['Confirmed'], covid_df['Deaths'], covid_df['Recovered'])
threedee.set_xlabel('Confirmed')
threedee.set_ylabel('Deaths')
threedee.set_zlabel('Recovered')
plt.show()

gm = GaussianMixture(n_components=K, max_iter=100, tol=0.00001)

gm.fit(covid_df.drop(['Country/Region'], axis=1))
labels = gm.predict(covid_df.drop(['Country/Region'], axis=1))

#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

threedee = plt.figure().gca(projection='3d')
threedee.scatter(covid_df['Confirmed'], covid_df['Deaths'], covid_df['Recovered'], c=labels, cmap='viridis')
threedee.set_xlabel('Confirmed')
threedee.set_ylabel('Deaths')
threedee.set_zlabel('Recovered')
plt.show()
    

    




