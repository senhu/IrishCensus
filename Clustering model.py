# -*- coding: utf-8 -*-
"""
Last modified on 31 July 2020

@author: Dr Sen Hu (UCD)

This file corresponds to SOME PARTS OF the results concerning ONLY Irish census data in the paper:
    Hu et al (2020), "A spatial machine learning model for analyzing customers' lapse behaviour in life insurance", Annals of Actuarial Science.

Note that only data set of Dublin is considered for the analysis. 
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
"""
Load the filtered census data set (for Dublin)
Omitted data directory here, fill in your own directory for the data set
And explore the data
"""

census = pd.read_csv('NewCensusData_final_Dublin.csv')
census = census.drop(census.columns[0], axis=1)
census.head()
census.shape
census.columns
census = census.dropna(axis=0, how='any')

census["Age0_4"].describe()
census["Age5_14"].describe()
census["Age25_44"].describe()
census["Age65over"].describe()
census["EU_National"].describe()
census["ROW_National"].describe()

#%%
"""
***********
* Model 2 *
***********
"""

#%%
"""
Phase 1 of Model 2: custering census data using k-medoids

No need to re-scale the census data since all variables are of percnetage unit.
Use the elbow method (elbow plot) to decide the optimum value of K (number of clusters). 
"""

from sklearn_extra.cluster import KMedoids

pc_data = census.iloc[:, 0:69]
pc_data.head()
pc_data.describe()

sse = {}
for k in range(1, 21):
    kmedoids = KMedoids(n_clusters=k, random_state=10)
    kmedoids.fit(pc_data)
    sse[k] = kmedoids.inertia_

plt.xlabel('Number of clustering components')
plt.ylabel('Sum of Distance')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

"""
K=8 is chosen based on the elbow plot. 
Note that detailed analysis reasoning is included in the paper.
"""

kmedoids = KMedoids(n_clusters=8, random_state=10)
kmedoids.fit(pc_data)
cluster_labels = kmedoids.labels_
census["cluster1"] = cluster_labels+1
census["cluster1"].value_counts()
census.head()

#%%
"""
Now 8 census clusters are found
We can profile these clusters i.e. finding key characteristics of each cluster. 

The first way to profile them is to compare their census summary information across.
"""

# set the color pallette for plotting, consistent with those in the paper. 
my_color = {1:"thistle", 2:"burlywood",3:"palegoldenrod",4:"lightpink",5:"paleturquoise",6:"lightgrey",7:"lightsteelblue",8:"darkseagreen"}

# Demographic information
sns.boxplot(x="cluster1", y="Age5_14", data=census, palette=my_color).set(xlabel='Cluster',ylabel='Age 5-14 population percentage')
sns.boxplot(x="cluster1", y="Age25_44", data=census, palette=my_color).set(xlabel='Cluster',ylabel='Age 25-44 population percentage')
sns.boxplot(x="cluster1", y="Age65over", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Age 65 and over population percentage")
sns.boxplot(x="cluster1", y="Born_outside_Ireland", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Born outside Ireland population percentage")

# Household decomposition
sns.boxplot(x="cluster1", y="HouseShare", data=census, palette=my_color).set(xlabel='Cluster',ylabel='House share percentage') 
sns.boxplot(x="cluster1", y="NonDependentKids", data=census, palette=my_color).set(xlabel="Cluster",ylabel='Family with non-dependent children percentage') #
sns.boxplot(x="cluster1", y="Dink", data=census, palette=my_color).set(xlabel="Cluster", ylabel='Couple with no children percentage')
sns.boxplot(x="cluster1", y="Married", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Married couple percentage")

# Housing
sns.boxplot(x="cluster1", y="Flats", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Flat dwelling percentage")
sns.boxplot(x="cluster1", y="RentPublic", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Public rent percentage")
sns.boxplot(x="cluster1", y="RentPrivate", data=census, palette=my_color).set(xlabel="Cluster",ylabel="Private rent percentage")
sns.boxplot(x="cluster1", y="Owned", data=census, palette=my_color).set(xlabel="Cluster",ylabel="Housing owned outright percentage")

# Socio-economic information
sns.boxplot(x="cluster1", y="HE", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Third level and above educated percentage")
sns.boxplot(x="cluster1", y="Employed", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Employment percentage")
sns.boxplot(x="cluster1", y="TwoCars", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Ownership of 2 or moe cars percentage")
sns.boxplot(x="cluster1", y="SC_professional", data=census, palette=my_color).set(xlabel="Cluster", ylabel="Professional social class percentage")

# Employment
sns.boxplot(x="cluster1", y="Unemployed", data=census, palette=my_color).set(xlabel="Cluster",ylabel="Unemployment percentage")

# Misc
sns.boxplot(x="cluster1", y="Internet", data=census, palette=my_color).set(xlabel="Cluster",ylabel="Broadband percentage")


#%%
"""
Another way to profile the census clusters is to plot them on a Dublin map

First download the census small area boundary shape files
Then load the shape files
Omitted data directory here, fill in your own directory
"""

import shapefile as shp

sf = shp.Reader("Census2016_Small_Areas_generalised20m")

# explore the data 
len(sf.shapes())
sf.shape()
sf.shapes()
dir(sf.shapes()[1])
sf.shapeRecord(1)
sf.record(1)[0:8]
sf.records()

"""
The shapefiles store all information about each small area, 
including descriptive information and boundary geometry information 

We need to extract such information properly before using it
by reading a shapefile into a pandas dataframe with a coords column holding the boundary geometry information.
"""

def read_shapefile(sf):
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

Ireland = read_shapefile(sf)
Ireland.shape
Ireland.head()
Ireland.iloc[1,]
Ireland["coords"][1]
Ireland["COUNTYNAME"].value_counts()

# extract only Dublin information
Dublin = Ireland[Ireland['COUNTYNAME'].isin(['Fingal','Dublin City','South Dublin','Dún Laoghaire-Rathdown'])]
Dublin.shape
Dublin.columns
Dublin["SMALL_AREA"]
Dublin["GEOGID"]

"""
First we can plot all the small area boundary data
"""
def plot_map(dat, x_lim = None, y_lim = None, figsize = (9,11)):
    plt.figure(figsize = figsize)
    for shape in dat["coords"]:
        x = [ i[0] for i in shape ]
        y = [ i[1] for i in shape ]
        plt.plot(x, y, 'k:', alpha=.9)        
    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)

plot_map(Dublin, figsize=(9,11))
plot_map(Dublin, figsize=(9,11), x_lim=(-6.34, -6.11), y_lim=(53.26,53.41))

"""
Combine the census clustering infomation with the boundary shape data
Then we can fill in colours for each small area depending on its cluster membership
"""

census.SAID
census.SMALL_AREA
census_sub = census[["SAID", "cluster1"]]
plotdat = pd.merge(left=Dublin, right=census_sub, how='left', left_on="SMALL_AREA", right_on="SAID")
plotdat.shape

plotdat["COUNTYNAME"].value_counts()
SouthDublin = plotdat[plotdat["COUNTYNAME"].isin(["South Dublin"])]
DublinCity = plotdat[plotdat["COUNTYNAME"].isin(["Dublin City"])]
Fingal = plotdat[plotdat["COUNTYNAME"].isin(["Fingal"])]
DunLR = plotdat[plotdat["COUNTYNAME"].isin(["Dún Laoghaire-Rathdown"])]

# set the color palette, consistent with the plots in the paper
my_color = ['thistle', 'burlywood', 'palegoldenrod', 'lightpink', 'paleturquoise', 'lightgrey', 'lightsteelblue', 'darkseagreen'] 

def plot_map_fill_cluster_colour(dat, x_lim = None, y_lim = None, 
                         figsize = (12,15), 
                         legend = True,
                         color = my_color):
    
    plt.figure(figsize = figsize)
    fig, ax = plt.subplots(figsize = figsize)
    for shape in dat["coords"]:
        x = [ i[0] for i in shape ]
        y = [ i[1] for i in shape ]
        plt.plot(x, y, 'k--',alpha=0.07)
    
    for c in range(1,9):
        #print(c)
        tempdat = dat.loc[ dat["cluster1"] == c , ]    
        for id in range(len(tempdat)):
            shape_ex = tempdat["coords"].iloc[id]
            x_lon = np.zeros((len(shape_ex),1))
            y_lat = np.zeros((len(shape_ex),1))
            for ip in range(len(shape_ex)):
                x_lon[ip] = shape_ex[ip][0]
                y_lat[ip] = shape_ex[ip][1]
            ax.fill(x_lon,y_lat, color[(c-1)])
    
    if (legend == True): 
        f = lambda m,c: ax.plot([],[],marker=m, color=c, ls="none")[0]
        handles = [f("s", color[i]) for i in range(8)]
        labels = ['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Cluster 7','Cluster 8']
        ax.legend(handles,labels, loc='lower right',frameon=False,
                  markerscale=4, fontsize='xx-large', bbox_to_anchor=(1.3,0))
               
    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)


plot_map_fill_cluster_colour(plotdat, figsize=(9, 11), legend=True)
plot_map_fill_cluster_colour(plotdat, x_lim=(-6.34, -6.11), y_lim=(53.26,53.41), figsize=(9,11),legend=False)
plot_map_fill_cluster_colour(SouthDublin, figsize=(9,9),legend=False)
plot_map_fill_cluster_colour(DublinCity, figsize=(9,9),legend=False)
plot_map_fill_cluster_colour(Fingal, figsize=(9,9),legend=False)
plot_map_fill_cluster_colour(DunLR, figsize=(9,9),legend=False)
