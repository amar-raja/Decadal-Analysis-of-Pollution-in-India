#!/usr/bin/env python
# coding: utf-8

# **Loading Libraries and Datasets**

# In[1]:


import numpy as np
import pandas as pd
import csv
import json
import statistics 
import os


# In[2]:


with open('../Dataset/neighbors.csv', newline='') as f:
    reader = csv.reader(f)
    neighbor_list = list(reader)


# In[3]:


neighbor={}
for i in range(1,35):
    neighbor[neighbor_list[i][0]]=neighbor_list[i][1:]


# In[4]:


not_present=['Manipur','Sikkim']
all_state_present=neighbor.keys()-not_present
temp={}
for i in all_state_present:
    temp[i]=[j for j in neighbor[i] if j not in not_present ]
neighbor=temp


# In[5]:


data=pd.read_csv('../Dataset/Decadal_Air_data_states.csv')


# **Finding Mean Pollutant Concentration**

# In[6]:


data['Mpc']=(data['SO2']+data['NO2']+data['PM2.5'])/3


# **Finding Z-score**
# 
# 1) Finding Mpc of all Neighbor of a particular state in a particular year
# 
# 2) Calculate mean and standard Deviation of Neighbor Mpc
# 
# 3) Calculating Z-score with this formula 
# 

# $Z-Score=\frac{(\mu-\sigma)^2}{\sigma}$ 

# **Finding HotSpot and ColdSpot**
# 
# 1) If Mpc of a state is less than Mpc mean of neighbor State + half the std deviation of Neighbor Mpc then it is a Coldspot and represent by -1
# 
# 2) If Mpc of a state is more than Mpc mean of neighbor State + half the std deviation of Neighbor Mpc then it is a Hotspot and represent by 1
# 
# 3) Else its a neural spot represent by 0

# In[7]:


data['zscore']=np.nan
data['spot']=np.nan


# In[8]:


for i in range(len(data)):
    State=data.iloc[i]['State']
    year=data.iloc[i]['year']
    mpc=data.iloc[i]['Mpc']
    nbr_mpc=[]    
    for ele in neighbor[State]:
        nbr_mpc.append(data[(data.State==ele)&(data.year==year)]['Mpc'].values[0])
    if(len(nbr_mpc)<2):
        nbr_mean=round(nbr_mpc[0],3)
        nbr_std=round(0,3)
        zscore=round(0,3)
    else:
        nbr_mean=round(statistics.mean(nbr_mpc),3)
        nbr_std=round(statistics.stdev(nbr_mpc,nbr_mean),3)
        zscore = round((mpc-nbr_mean)/nbr_std,3)
    if (mpc>(nbr_mean+(nbr_std)/2)):
        spot = 1
    elif(mpc<(nbr_mean-(nbr_std)/2)):
        spot = -1
    else:
        spot=0
    data.loc[i,'zscore']=zscore
    data.loc[i,'spot']=spot
    


# In[9]:


data.columns=['State', 'Year', 'SO2', 'NO2', 'PM2.5', 'Mpc', 'zscore', 'spot']


# **Finding Topmost Hotspot and Coldspot**

# In[10]:


sort_data=data.sort_values(['Year','zscore'],ascending=[True, False])
sort_data.reset_index(drop=True,inplace=True)


# In[11]:


topSpots_df = pd.DataFrame(columns=['Year','spot','State1','State2','State3','State4','State5'])
i=0
while(i<310):
    hotstate_list=[]
    coldstate_list=[]
    hotstate = sort_data[i:i+5]
    coldstate = sort_data[i+26:i+31]
    year_id = str(int(hotstate.iloc[0]['Year']))
    hotstate_list=list(hotstate['State'])
    coldstate_list=list(coldstate['State'])
    topSpots_df=topSpots_df.append({'Year':year_id,'spot': 'hot','State1':hotstate_list[0],'State2':hotstate_list[1],'State3':hotstate_list[2],'State4':hotstate_list[3],'State5':hotstate_list[4]},ignore_index=True)
    topSpots_df=topSpots_df.append({'Year':year_id,'spot': 'cold','State1':coldstate_list[0],'State2':coldstate_list[1],'State3':coldstate_list[2],'State4':coldstate_list[3],'State5':coldstate_list[4]},ignore_index=True)  
    i+=31


# In[12]:


topSpots_df.to_csv("../Dataset/Generated/zscore/TopSpots.csv",index=False)


# In[13]:


data.to_csv("../Dataset/Generated/zscore/Air_Quality_Mpc_zscore.csv",index=False)


# In[14]:


state_co=pd.read_csv('../Dataset/State-Coordinates.csv')


# In[15]:


data.merge(state_co,on=['State']).to_csv("../Generate Map/Datasets/zscore.csv",index=False)

