#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr 
from scipy.stats import spearmanr
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore')


# In[ ]:


#READING FILE
dec_city   =   pd.read_csv('../Dataset/Decadal_air_data_cities.csv')
dec_st     =   pd.read_csv('../Dataset/Decadal_Air_data_states.csv')
indus      =   pd.read_csv('../Dataset/Industries_2011_2020.csv')
motor_veh  =   pd.read_csv('../Dataset/MotorVehicles_2011_2020.csv')
population =   pd.read_csv('../Dataset/Population_2011_2020.csv') 


# In[ ]:


st = dec_st.State.unique()
ct_st = dec_city.State.unique()
ind_st = indus.State.unique()   
mt_st = motor_veh.State.unique() 
pop_st = population.State.unique() 


# In[ ]:


dec_st


# In[ ]:


dec_st.shape,indus.shape


# In[ ]:


population.head()


# In[ ]:


#CREATING A DATAFRAME CONSISTING OF ONLY REQUIRED COLUMNS FROM ALL CSV FILES
df = dec_st[['year','SO2', 'NO2', 'PM2.5']]        #FROM STATEDATA
df['noi'] = indus['Number']                       #FROM INSUSTRIES
df['nom'] = motor_veh['Number']                   #FROM MoTOR VEHC
df['pop'] = population['Population']              #FROM CENSUS POP
df


# In[ ]:


label = ['Year', 'SO2', 'NO2', 'PM2.5', 'Indust', 'vehicles', 'Population']


# In[ ]:


#scatter plots
fig = plt.figure(0,figsize=(16,6))
fig.suptitle('Scatter plots of features', fontsize=16)

for i in range(df.shape[1]):            # through columns
    for j in range(df.shape[1]):
        ax = plt.subplot2grid((7,7), (i,j))
        ax.scatter(df.iloc[:,j], df.iloc[:,i])
        
        if j==0:
            ax.set_ylabel(label[i])
        if i==6:
            ax.set_xlabel(label[j])
            if j==0:
                ax.set_xticks([2010,2020])
        if j!=0 and i!=6:
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
        if j==0 and i!=6:
            #x_axis = ax.axes.get_xaxis()
            #x_axis.set_visible(False)
            ax.axes.xaxis.set_ticklabels([])
        if i==6 and j!=0:
            #y_axis = ax.axes.get_yaxis()
            #y_axis.set_visible(False)
            ax.axes.yaxis.set_ticklabels([])
    
fig.savefig('../Images/Correlaton_Scatterplots.png')
plt.show()


# ### CORRELATION

# In[ ]:


corr_pearson = []                    
corr_spearman = []
for i in range(df.shape[1]):            # ITERATING THROUGH EACH COLUMN
    tp = []                 #STORING COEFFICICENTS in a LIST
    ts=[]
    for j in range(df.shape[1]):         # AND CALCULATING WITH EACH COLUMN
        cr1,p1 = pearsonr(df.iloc[:,i],df.iloc[:,j])                 # CALCULATING PEARSON COEFFICICENTS
        tp.append(round(cr1,4))
        
        cr2,p2 = spearmanr(df.iloc[:,i],df.iloc[:,j])                 # CALCULATING SPEARMAN COEFFICICENTS
        ts.append(round(cr2,4))
        
    corr_pearson.append(tp)
    corr_spearman.append(ts)


# CORRELATION MATRIX

# In[ ]:


L = corr_spearman
count = 0

label = ['year', 'so2', 'no2', 'pm2.5', 'industry', 'vehicle', 'population']

fig, ax = plt.subplots(figsize=(16,10))
fig.suptitle('Correlation Matrix', fontsize=16)

min_val, max_val = 0, 7

im = ax.matshow(L, cmap=plt.cm.Oranges)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.15)

for i in range(7):
    for j in range(7):
        c = L[j][i]
        ax.text(i,j, str(c), va='center', ha='center')
        ax.set_xticklabels(['']+label)
        ax.set_yticklabels(['']+label)

fig.colorbar(im, cax=cax, orientation='vertical')
fig.savefig('../Images/Correlation_Matrix.png')
plt.show()


# ### COAL PRODUCTION AND COAL CONSUMPTION IN INDIA

# In[ ]:


coal_prod  =   pd.read_csv(r'../Dataset/Coal_Production_2011_2020.csv')
coal_cons  =   pd.read_csv(r'../Dataset/Industrial_Coal_Consumption_2011_2020.csv')


# In[ ]:


plt.figure(figsize=(16,6))
sns.lineplot(x = coal_cons['year'],y=coal_cons['TOT(MT)'])
sns.lineplot(x = coal_cons['year'],y=coal_prod['TOT(MT)'])
plt.grid(True)
plt.legend(['COAL CONSUMPTION','COAL PRODUCTION'])
plt.savefig('../Images/Coal_prod_cons.png')

