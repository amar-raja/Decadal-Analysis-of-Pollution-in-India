#!/usr/bin/env python
# coding: utf-8

# **Loading Libraries And Dataset**

# In[1]:


import numpy as np
import pandas as pd
import csv
import json
import statistics 
from scipy.stats import chi2
import os


# In[2]:


data=pd.read_csv('../Dataset/Decadal_Air_data_states.csv')


# In[3]:


data


# **Finding Expected value of All States (i.e For India) per Year**

# In[4]:


expected_df=data.groupby(['year']).mean()


# In[5]:


data=data.merge(expected_df,on=['year'])


# **Chi Square Test For Finding Outlier**
# 
# $\chi^2=\sum_{i=1}^{i=N}\frac{(o_i-E_i)^2}{E_i}$
# 

# In[6]:


data['Chi Square Value']=((data.SO2_x-data.SO2_y)**2/data.SO2_y)+((data.NO2_x-data.NO2_y)**2/data.NO2_y)+((data['PM2.5_x']-data['PM2.5_y'])**2/data['PM2.5_y'])


# **Hypothesis Test**
# 
# 1) p-value obtain from Chi-Square Distribution
# 
# 2) Null Hypothesis test with Significance level 1%

# In[7]:


def pvalue(x):
    return 1-chi2.cdf(x,df=2)


# In[8]:


data['p-value']=data['Chi Square Value'].apply(lambda x:pvalue(x))


# If p-value > Significance value then it is not an outlier else it's an outlier

# In[9]:


def outlier(x,significant_value):
    if x<significant_value:
        return 1
    else:
        return 0


# In[10]:


data['Outlier']=data['p-value'].apply(lambda x:outlier(x,0.01))


# In[11]:


data['Mpc']=(data['SO2_x']+data['NO2_x']+data['PM2.5_x'])/3


# In[12]:


data=data.sort_values(['State','year'])[['State','year','Mpc','p-value','Outlier']]


# In[13]:


data.columns=['State','Year','Mpc','p-value','outlier']


# In[14]:


state_co=pd.read_csv('../Dataset/State-Coordinates.csv')


# In[15]:


data_map=data.merge(state_co,on=['State'])


# In[16]:


data.to_csv("../Dataset/Generated/chi-square/Air_Quality_chiscore.csv",index=False)


# In[17]:


data_map.to_csv("../Generate Map/Datasets/chi.csv",index=False)

