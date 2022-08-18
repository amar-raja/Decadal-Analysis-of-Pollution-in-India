#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
from bs4 import BeautifulSoup
import requests
import xlrd
import json
import math
from scipy import interpolate
import warnings
warnings.simplefilter(action='ignore')
import os 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Getting StationId,City,State mapping
file = '../Raw/stations.csv'
df_stations = pd.read_csv(file)
df_stations.drop(['StationName',"Status"],axis=1,inplace=True)
df_stations.drop_duplicates(inplace=True,ignore_index=True)


# # Data cleaning of 2015-2020 values

# In[3]:


#daily values of stations from 2015
file_s = '../Raw/station_day.csv'
df_station_d = pd.read_csv(file_s,parse_dates=["Date"])


# In[4]:


#merging station daily values with their city and state names
df_station_day=pd.merge(df_stations,df_station_d,how="inner",left_on="StationId",right_on="StationId")
df_station_day["year"]=df_station_day.Date.dt.year


# In[5]:


#null values in station day
print(len(df_station_day))
df_station_day.isnull().sum()


# In[6]:


#dropping Benzene,Xylene,Toluene,NH3 and PM10 as they have morethan 25% null values
df_station_day.drop(["Benzene","Xylene","Toluene","NH3","PM10","AQI_Bucket","O3","NOx"],axis=1,inplace=True)


# In[7]:


#function to extrapolate station wise values

def f1(ct,tot_yrs):
    ct.reset_index(drop=True,inplace=True)
    state_name  = ct['State'].unique()
    st_name  = ct['StationId'].unique()
    ct_name  = ct['City'].unique()

    tot_yr=tot_yrs
    
    yr = ct['year'].values
    if len(yr)==1:
        return ct
   
    absent_yr = [i for i in tot_yr if i not in yr]
    f_s02 = interpolate.interp1d(ct['year'],ct['SO2'],kind='nearest',fill_value = "extrapolate")
    f_n02 = interpolate.interp1d(ct['year'],ct['NO2'],kind='nearest',fill_value = "extrapolate")
    f_pm25 = interpolate.interp1d(ct['year'],ct['PM2.5'],kind='nearest',fill_value = "extrapolate")
    
   
    x_new_s02 = [f_s02(i) for i in absent_yr]
    x_new_n02 = [f_n02(i) for i in absent_yr]
    x_new_pm25 = [f_pm25(i) for i in absent_yr]
    
    #plt.plot(ct['year'],ct['SO2'],absent_yr,x_new_s02,'*')
    
    for i in range(len(x_new_s02)):
        ct.loc[len(ct.index)] = [state_name[0],ct_name[0],st_name[0],absent_yr[i],x_new_s02[i],x_new_n02[i],x_new_pm25[i]]
        
    return ct


# In[8]:


#groupby stationid and then interpoalte linearly
df_station_day=df_station_day.groupby(by=["StationId"]).apply(lambda group: group.interpolate(method='index',limit_direction='both'))

#heatmap for null values
sns.heatmap(df_station_day.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[9]:


#groupby city and interpolate linearly
df_station_day=df_station_day.groupby(by=["City"]).apply(lambda group: group.interpolate(method='index',limit_direction='both'))

#heatmap for null values
sns.heatmap(df_station_day.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


#annual mean of StationId values 
df_station_day = df_station_day.groupby(['StationId','City',"State","year"],as_index=False).mean()
df_test=df_station_day[['State','City','StationId','year','SO2','NO2','PM2.5']]

#heatmap for null values
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[11]:


#extrapolating the station data for unavailable years
df_2015_final=pd.DataFrame()
for i in df_test["StationId"].unique():
    tot_yrs=[2020,2019,2018,2017,2016,2015]
    fun=f1(df_test[df_test["StationId"]==i],tot_yrs)
    fun.sort_values(by=['year'],inplace=True)
    df_2015_final=pd.concat([df_2015_final,fun])
    
df_2015_final


# In[12]:


df_2015_final["SO2"]=df_2015_final["SO2"].astype(float)
df_2015_final["NO2"]=df_2015_final["NO2"].astype(float)
df_2015_final["PM2.5"]=df_2015_final["PM2.5"].astype(float)
df_2015_final.dtypes


# In[13]:


#mean of annual city wise values
df_2015_final = df_2015_final.groupby(["State",'City',"year"],as_index=False).mean()


# In[14]:


df_2015_final[df_2015_final.isnull().any(axis=1)]


# In[15]:


df_2015_final["City"].value_counts()
#df_2015_final["year"].unique()


# # Data Preprocessing 1990-2015 data

# In[16]:


#read 1990-2014 data file
df_2011=pd.read_csv('../Raw/data.csv',usecols=["stn_code","state","location","so2","no2","rspm","date"],parse_dates=["date"])


# In[17]:


df_2011.info()


# In[18]:


#take data values not null
df_2011=df_2011[~df_2011["date"].isnull()]

df_2011["year"]=df_2011.date.dt.year

#rename columns
df_2011.columns=['StationId','State','City','SO2','NO2',"PM2.5",'Date','year']
df_2011


# In[19]:


#heatmap of null values
sns.heatmap(df_2011.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[20]:


count = pd.DataFrame(df_2011["StationId"].value_counts())
count.reset_index(inplace=True)
count.columns=["StationId","Count"]
count=count[count["Count"]<=2]
count


# In[21]:


single_stations = count["StationId"].tolist()
#single_stations
df_2011=df_2011[~df_2011["StationId"].isin(single_stations)]


# In[22]:


df_2011=df_2011.groupby(by=["StationId"]).apply(lambda group: group.interpolate(method='index',limit_direction='both'))

sns.heatmap(df_2011.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[23]:


df_2011=df_2011.groupby(by=["City"]).apply(lambda group: group.interpolate(method='index',limit_direction='both'))
sns.heatmap(df_2011.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[24]:


#stationid wise mean
df_2011 = df_2011.groupby(['StationId','City',"State","year"],as_index=False).mean()
df_2011=df_2011[['State','City','StationId','year','SO2','NO2','PM2.5']]
df_2011=df_2011[df_2011["year"]>2010]
sns.heatmap(df_2011.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[25]:


df_2011[df_2011["PM2.5"].isnull()]


# In[26]:


len(df_2011["City"].unique())


# In[27]:


#extrapolating stationwise values
df_2011_final=pd.DataFrame()
for i in df_2011["StationId"].unique():
    tot_yrs=[2015,2014,2013,2012,2011]
    fun=f1(df_2011[df_2011["StationId"]==i],tot_yrs)
    fun.sort_values(by=['year'],inplace=True)
    df_2011_final=pd.concat([df_2011_final,fun])
    
df_2011_final


# In[28]:


df_2011_final["year"].value_counts()


# In[29]:


len(df_2011_final["City"].unique())


# In[30]:


df_2011_final["SO2"]=df_2011_final["SO2"].astype(float)
df_2011_final["NO2"]=df_2011_final["NO2"].astype(float)
df_2011_final["PM2.5"]=df_2011_final["PM2.5"].astype(float)
df_2011_final.dtypes


# In[31]:


df_2011_final = df_2011_final.groupby(["State",'City',"year"]).mean()
df_2011_final


# In[32]:


for year in range(2011,2016):
    df_2011_final.loc[('Delhi','Delhi',year)]=df_2011_final.loc[[('Delhi','Delhi',year),('Uttar Pradesh','Noida',year)]].mean()
df_2011_final.reset_index(inplace=True)


# In[33]:


ts_cities=df_2011_final[df_2011_final["State"]=="Telangana"]["City"].unique()
for city in ts_cities:
    df_2011_final.loc[df_2011_final.City==city, 'State'] = "Telangana"


# In[34]:


df_2011_final=df_2011_final.replace('Bangalore','Bengaluru')
df_2011_final


# In[35]:


len(df_2011_final["City"].unique())


# In[36]:


df_concat = pd.concat([df_2011_final,df_2015_final])
df_concat


# In[37]:


df_concat = df_concat.groupby(["State",'City',"year"],as_index=False).mean()
df_concat


# In[38]:


def f2(ct):
    ct.reset_index(drop=True,inplace=True)
    state_name  = ct['State'].unique()
    ct_name  = ct['City'].unique()
    
    tot_yr = [2020,2019,2018,2017,2016,2015,2014,2013,2012,2011]
    
    yr = ct['year'].values
    if len(yr)==1:
        return ct
    absent_yr = [i for i in tot_yr if i not in yr]
    f_s02 = interpolate.interp1d(ct['year'],ct['SO2'],kind='linear',fill_value = "extrapolate")
    f_n02 = interpolate.interp1d(ct['year'],ct['NO2'],kind='linear',fill_value = "extrapolate")
    f_pm25 = interpolate.interp1d(ct['year'],ct['PM2.5'],kind='linear',fill_value = "extrapolate")
    
    x_new_s02 = [f_s02(i) for i in absent_yr]
    x_new_n02 = [f_n02(i) for i in absent_yr]
    x_new_pm25 = [f_pm25(i) for i in absent_yr]
    #print(x_new_s02)
    #plt.plot(ct['year'],ct['SO2'],absent_yr,x_new_s02,'*')
#     print(slinear
#     print(x_new_s02)
    
    for i in range(len(x_new_s02)):
        ct.loc[len(ct.index)] = [state_name[0],ct_name[0],absent_yr[i],x_new_s02[i],x_new_n02[i],x_new_pm25[i]]
    return ct


# In[39]:


df_concat_final=pd.DataFrame()
for i in df_concat["City"].unique():
    fun=f2(df_concat[df_concat["City"]==i])
    fun.sort_values(by=['year'],inplace=True)
    df_concat_final=pd.concat([df_concat_final,fun])


# In[40]:


df_concat_final["SO2"]=df_concat_final["SO2"].astype(float)
df_concat_final["NO2"]=df_concat_final["NO2"].astype(float)
df_concat_final["PM2.5"]=df_concat_final["PM2.5"].astype(float)
df_concat_final.dtypes


# In[41]:


cities=df_2015_final["City"].unique()
cities_2011=df_2011_final["City"].unique()
city_common=[i for i in cities if i in cities_2011]
print(len(cities),len(cities_2011),len(city_common))
print(city_common)
df_cities=df_concat_final[df_concat_final["City"].isin(city_common)]
df_cities.reset_index(drop=True,inplace=True)
df_cities


# In[42]:


df_cities[df_cities["City"]=="Kochi"]


# In[43]:


df_cities["City"].value_counts()


# In[44]:


df_states = df_concat_final.groupby(["State","year"],as_index=False).mean()
df_states.reset_index(drop=True,inplace=True)
df_states


# In[45]:


df_states["State"].value_counts()


# In[46]:


def check_negatives(df):
    prec_val = -999
    cols=['SO2','NO2','PM2.5']
    no_neg=0
    # iterate over columns
    for i in cols:
        # resetting value over each column
        prec_val = -999
        # iterate over rows
        for j in range(df.shape[0]):
            # accessing the cell value
            cell = df.at[j, i]
            # check if cell value is negative
            if(cell < 0):
                # check if prec_val is not default
                # set value
                if(prec_val != -999):
                    print(i,j,cell,prec_val)
                    # replace the cell value
                    df.at[j, i] = prec_val
            else:
                # store the latest value in variable
                prec_val = df.at[j, i]
    return df


# In[47]:


df_cities = check_negatives(df_cities)
df_states=check_negatives(df_states)


# cols=['SO2','NO2','PM2.5']
# nonneg=0
#     # iterate over columns
# for i in cols:
#     # iterate over rows
#     for j in range(df_cities.shape[0]):
#         # accessing the cell value
#         cell = df_cities.at[j, i]
#         # check if cell value is negative
#         if(cell < 0):
#             print(i,j,cell)
#         else:
#             nonneg+=1
# print(nonneg)

# In[48]:


df_cities.to_csv("../Dataset/Decadal_air_data_cities.csv",index=False)
df_states.to_csv("../Dataset/Decadal_Air_data_states.csv",index=False)


# # Data Cleaning and Preprocessing for Industries data

# In[49]:


inputfile  = '../Raw/Industries 2009-2015.xlsx'
df_ind1 = pd.read_excel(inputfile, sheet_name='State-wise',skiprows=10,usecols=[0,1,2,3,4,5,6,7])
df_ind1.columns=["State","2009","2010","2011","2012","2013","2014","2015"]
df_ind1.dropna(inplace=True)
null_states=["Telangana","Lakshadweep"]
df_ind1=df_ind1[~df_ind1["State"].isin(null_states)]
df_ind1.reset_index(drop=True,inplace=True)


# In[50]:


ind2= '../Raw/2001-2008_industry_data.csv'
df_ind2=pd.read_csv(ind2)
df_ind2.dropna(inplace=True)
df_ind2.rename(columns={"state/ut":"State"},inplace=True)
df_ind2.reset_index(drop=True,inplace=True)


# In[51]:


states1=df_ind1["State"].unique()
states2=df_ind2['State'].unique()
not_Common=[state for state in states2 if state not in states1]
not_Common


# In[52]:


df_ind1.replace({"UttaraKhand":"Uttarakhand","A & N. Island":"Andaman & Nicobar Islands","Dadra & N Haveli":"Dadra & Nagar Haveli"},inplace=True)


# In[53]:


df_ind= pd.merge(df_ind2,df_ind1,how="inner",left_on="State",right_on="State")
df_ind


# In[54]:


df_ind = pd.melt(df_ind,id_vars=['State'],
        var_name='year', value_name='Number')


# In[55]:


df_ind.sort_values(by=['State','year'],inplace=True)


# In[56]:


df_ind["year"]=df_ind["year"].astype(int)


# In[57]:


def extrapolate_years(ct):
    ct.reset_index(drop=True,inplace=True)
    state_name  = ct['State'].unique()
    tot_yr =np.arange(2020, 2000, -1)
    yr = ct['year'].values
    
    if len(yr)==1:
        return ct
    absent_yr = [i for i in tot_yr if i not in yr]
    f_num = interpolate.interp1d(ct['year'],ct['Number'],kind='linear',fill_value = "extrapolate")
    x_new_num = [f_num(i) for i in absent_yr]
    
    #plt.plot(ct['year'],ct['Number'],absent_yr,x_new_num,'*')
    
    for i in range(len(x_new_num)):
        ct.loc[len(ct.index)] = [state_name[0],absent_yr[i],x_new_num[i]]
    return ct


# In[58]:


df_ind_final=pd.DataFrame()
for i in df_ind["State"].unique():
    #tot_yrs=[2020,2019,2018,2017,2016,2015]
    fun=extrapolate_years(df_ind[df_ind["State"]==i])
    fun.sort_values(by=['year'],inplace=True)
    df_ind_final=pd.concat([df_ind_final,fun])


# In[59]:


df_ind_final=df_ind_final[df_ind_final["year"]>2010]
df_ind_final


# In[60]:


df_ind_final.to_csv("../Dataset/Industries_2011_2020.csv",index=False)


# # Data Cleaning and Preprocessing Motor Vehicles

# In[61]:


mv_data  = "../Raw/MotorVehicles 2001-2016.xlsx"
df_mv = pd.read_excel(mv_data,skiprows=8)

years = list(map(str, np.arange(2001, 2017,1)))
cols=["State"]+years
df_mv.columns=np.array(cols)

df_mv.dropna(inplace=True)
df_mv["State"] = df_mv["State"].str.strip()
null_states=["Telangana","TOTAL UTs","GRAND TOTAL","TOTAL STATES"]
df_mv=df_mv[~df_mv["State"].isin(null_states)]

df_mv.replace({'Orissa': 'Odisha',
               'Chhatisgarh': 'Chhattisgarh',
               'A. & N. Islands': 'Andaman & Nicobar Islands',
               'D. & N. Haveli': 'Dadra & Nagar Haveli'},inplace=True)

df_mv.reset_index(drop=True,inplace=True)
df_mv


# In[62]:


df_mv=df_mv.replace(["\*",'\+','\**',"\$","\&","\##","\#","\,",'\(R\)'],"",regex=True)
df_mv


# In[63]:


df_mv = pd.melt(df_mv,id_vars=['State'],
        var_name='year', value_name='Number')
df_mv["Number"]=df_mv["Number"].astype(int)
df_mv["year"]=df_mv["year"].astype(int)
df_mv.sort_values(by=['State','year'],inplace=True)
df_mv


# In[64]:


df_mv_final=pd.DataFrame()
for i in df_mv["State"].unique():
    #tot_yrs=[2020,2019,2018,2017,2016,2015]
    fun=extrapolate_years(df_mv[df_mv["State"]==i])
    fun.sort_values(by=['year'],inplace=True)
    df_mv_final=pd.concat([df_mv_final,fun])
    
df_mv_final=df_mv_final[df_mv_final["year"]>2010]
df_mv_final


# In[65]:


df_mv_final.to_csv("../Dataset/MotorVehicles_2011_2020.csv",index=False)


# # Population data 

# 2001

# In[66]:


dict_population={}
temp_pop=[]
temp_state=[]
temp_state_name=[]
result=[]
BASE_URL = "https://censusindia.gov.in/Census_Data_2001/Census_data_finder/A_Series/Total_population.htm"
html = requests.get(BASE_URL, verify=False).text
soup = BeautifulSoup(html, "html.parser")
tds = soup.find_all(class_='xl296353')
tds_pop = soup.find_all(class_='xl306353')
for td in tds_pop:
    if td.text!='\xa0' :
        temp_pop.append(td.text)


# In[67]:


for i in range(0,len(temp_pop),3):
    temp_state.append(temp_pop[i])
for td in tds:
    if td.text!='\xa0' :
        temp_state_name.append(td.text)


# In[68]:


for i in range(0,len(temp_state_name)):
    if temp_state_name[i]=='Andaman &\r\n  Nicobar Islands' or temp_state_name[i]=='Lakshadweep' or temp_state_name[i]=='Tripura':
        continue
    else:
        if "\r\n " in temp_state_name[i]:
            res=temp_state_name[i].replace("\r\n ","")
            dict_population[res]=float(temp_state[i].replace(',',''))
        elif temp_state_name[i]=="Orissa":
            res= temp_state_name[i].replace("Orissa","Odisha")
            dict_population[res]=float(temp_state[i].replace(',',''))
        elif temp_state_name[i]=="Uttaranchal":
            res= temp_state_name[i].replace("Uttaranchal","Uttarakhand")
            dict_population[res]=float(temp_state[i].replace(',',''))
        elif temp_state_name[i]=="Pondicherry":
            res= temp_state_name[i].replace("Pondicherry","Puducherry")
            dict_population[res]=float(temp_state[i].replace(',',''))
        elif temp_state_name[i]=="Manipur*":
            res= temp_state_name[i].rstrip('*')
            dict_population[res]=float(temp_state[i].replace(',',''))
        else:
            dict_population[temp_state_name[i]]=float(temp_state[i].replace(',',''))


# In[69]:


df_2001=pd.DataFrame()
df_2001["State"]=dict_population.keys()
df_2001["Population"]=dict_population.values()
df_2001["Population"]=df_2001["Population"].astype(int)

df_2001.to_csv("../Dataset/Census2001.csv",index=False)


# In[70]:


df_2001["State"]=df_2001["State"].str.upper()
df_2001


# 2011

# In[71]:


url="http://censusindia.gov.in/pca/DDW_PCA0000_2011_Indiastatedist.xlsx"
df_2011=pd.read_excel(url,usecols=[0,6,7,8,10])


# In[72]:


df_2011=df_2011[(df_2011["Level"]=="STATE") & (df_2011["TRU"]=="Total")]
df_2011.drop(["State","Level","TRU"],axis=1,inplace=True)
df_2011.reset_index(drop=True,inplace=True)
df_2011


# In[73]:


df_2001.replace('DELHI','NCT OF DELHI',inplace=True)
df_2011.rename(columns={'Name':'State'},inplace=True)


# In[74]:


df_pop=pd.merge(df_2001,df_2011,how='inner',left_on='State',right_on='State')
df_pop.columns=['State','2001','2011']
df_pop


# In[75]:


df_pop["2001"]=df_pop["2001"].astype(int)
df_pop["2011"]=df_pop["2011"].astype(int)
df_pop=df_pop.set_index("State")


# In[76]:


df_pop_test=df_pop.copy()
df_pop_test["1+gr"]=df_pop_test["2011"]/df_pop_test["2001"]

sr = pd.Series([1,1,0.1], index =["2001","2011","1+gr"])
df_pop_test=df_pop_test.pow(sr,axis=1)
df_pop_test["2001"]=df_pop_test["2001"].astype(int)
df_pop_test["2011"]=df_pop_test["2011"].astype(int)
df_pop_test.reset_index(inplace=True)
df_pop_test


# In[77]:


for i in range(2002,2021):
    year=str(i)
    df_pop_test[year]=df_pop_test["2001"]*(df_pop_test["1+gr"])**(i-2001)
    df_pop_test[year]=df_pop_test[year].astype('int64') 
df_pop_test=df_pop_test[["State","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]]
df_pop_test


# In[78]:


#extrapolated population of India in 2020
df_pop_test["2020"].sum()


# In[79]:


df_pop_final = pd.melt(df_pop_test,id_vars=['State'],
        var_name='year', value_name='Population')
df_pop_final["Population"]=df_pop_final["Population"].astype(int)
df_pop_final["year"]=df_pop_final["year"].astype(int)
df_pop_final.sort_values(by=['State','year'],inplace=True)
df_pop_final.reset_index(drop=True,inplace=True)
df_pop_final


# In[80]:


df_pop_final.to_csv("../Dataset/Population_2011_2020.csv",index=False)


# # Energy Consumption 

# Production of Coal

# In[81]:


df_prod=pd.read_excel("../Raw/Production of coal.xlsx",skiprows=8,usecols=[0,3],names=['year','TOT(MT)'])
df_prod.dropna(inplace=True)
df_prod['year']=np.arange(2001,2017,1)
df_prod


# In[82]:


#function to extrapolate station wise values

def coal(ct,tot_yrs):
    ct.reset_index(drop=True,inplace=True)

    tot_yr=tot_yrs
    yr = ct['year'].values
    if len(yr)==1:
        return ct
    absent_yr = [i for i in tot_yr if i not in yr]
    f_prod = interpolate.interp1d(ct['year'],ct['TOT(MT)'],kind='linear',fill_value = "extrapolate")
   
    x_new_prod = [f_prod(i) for i in absent_yr]
    
    plt.plot(ct['year'],ct['TOT(MT)'],absent_yr,x_new_prod,'*')

    for i in range(len(x_new_prod)):
        ct.loc[len(ct.index)] = [absent_yr[i],x_new_prod[i]]
    ct["year"]=ct["year"].astype(int)  
    ct["TOT(MT)"]=ct["TOT(MT)"].astype(float).round(2)
    ct.sort_values(by=['year'],inplace=True)
    return ct


# In[83]:


tot_yrs=np.arange(2020, 2000, -1)
df_prod_final=coal(df_prod,tot_yrs)
df_prod_final


# In[84]:


df_prod_final=df_prod_final[df_prod_final["year"]>2010]
df_prod_final.to_csv("../Dataset/Coal_Production_2011_2020.csv",index=False)


# Industrial Coal Consumption

# In[85]:


df_cons=pd.read_excel("../Raw/Industrial coal consumption.xlsx",skiprows=6,usecols=[0,10],names=['year','TOT(MT)'])
df_cons.dropna(inplace=True)
df_cons['year']=np.arange(2001,2017,1)
df_cons


# In[86]:


tot_yrs=np.arange(2020, 2000, -1)
df_cons_final=coal(df_cons,tot_yrs)
df_cons_final


# In[87]:


df_cons_final=df_cons_final[df_cons_final["year"]>2010]
df_cons_final.to_csv("../Dataset/Industrial_Coal_Consumption_2011_2020.csv",index=False)

