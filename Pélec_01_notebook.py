#!/usr/bin/env python
# coding: utf-8

# # Projet : Anticipez les besoins en consommation électrique de bâtiments

# ## Objectif :
# 
# - Seattle, ville neutre en émissions de carbone en 2050
# - émissions des bâtiments non destinés à l’habitation
# - Comprendre du mieux possible nos données
# - Prédictions des émissions de CO2 et de la consommation totale d’énergie

# # 1. Analyse de la forme des données :

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


pd.set_option('display.max_row', 100)
pd.set_option('display.max_column', 99)


# 
# # 2015

# In[4]:


data_2015 = pd.read_csv(r"C:\Users\SAMSUNG\Desktop\OpenClassrooms\P4_OC\P4_Donnees\2015_Building_Energy_Benchmarking.csv", sep=",")


# In[5]:


df_2015 = data_2015.copy()


# In[6]:


df_2015.shape


# In[7]:


df_2015.dtypes.value_counts()


# In[8]:


df_2015.head()


# In[9]:


data_2016 = pd.read_csv(r"C:\Users\SAMSUNG\Desktop\OpenClassrooms\P4_OC\P4_Donnees\2016_Building_Energy_Benchmarking.csv", sep=",")


# In[10]:


df_2016 = data_2016.copy()


# In[11]:


df_2016.shape


# In[12]:


df_2016.dtypes.value_counts()


# In[13]:


df_2016.head()


# In[14]:


columns = df_2015.columns
columns


# In[15]:


df_2015.describe()


# In[16]:


df_2015["Location"].values[0]


# In[17]:


# Création des colonnes city_state_zip et lat_long :
df_2015["city_state_zip"] = df_2015["Location"].str.split("\n").map(lambda x: x[1])
df_2015["lat_long"] = df_2015["Location"].str.split("\n").map(lambda x: x[2])


# In[18]:


df_2015["city_state_zip"].values[0]


# In[19]:


df_2015["lat_long"].values[0]


# In[20]:


# Création de la colonne ZipeCode pour df_2015 :
df_2015["ZipCode"] = df_2015["city_state_zip"].str.split(" ").map(lambda x: x[2])


# In[21]:


df_2016["ZipCode"].dtypes


# In[22]:


df_2015["ZipCode"].dtypes


# In[23]:


df_2015["ZipCode"] = df_2015["ZipCode"].astype(float)


# In[24]:


df_2015["ZipCode"].values[0]


# In[25]:


# Création de la colonne Latitude pour df_2015 :
df_2015["Latitude"] = df_2015["lat_long"].str.split(",").map(lambda x: x[0]).str.replace('(','')


# In[26]:


df_2015["Latitude"].values[0]


# In[27]:


df_2016["Latitude"].dtypes


# In[28]:


df_2015["Latitude"].dtypes


# In[29]:


df_2015["Latitude"] = df_2015["Latitude"].astype(float)


# In[30]:


df_2015["Latitude"].values[0]


# In[31]:


# Création de la colonne Longitude pour df_2015 :
df_2015["Longitude"] = df_2015["lat_long"].str.split(", ").map(lambda x: x[1]).str.replace(')','')


# In[32]:


df_2015["Longitude"].values[0]


# In[33]:


df_2016["Longitude"].dtypes


# In[34]:


df_2015["Longitude"].dtypes


# In[35]:


df_2015["Longitude"] = df_2015["Longitude"].astype(float)


# In[36]:


df_2015["Longitude"].values[0]


# In[37]:


df_2015.shape


# In[38]:


df_2015.head()


# In[39]:


columns_2015 = df_2015.columns
columns_2015


# In[40]:


# Renommer GHGEmissions(MetricTonsCO2e) -> TotalGHGEmissions(MetricTonsCO2e)
df_2015 = df_2015.rename(columns={'GHGEmissions(MetricTonsCO2e)': 'TotalGHGEmissions(MetricTonsCO2e)'})


# In[41]:


print(df_2015.columns)


# In[42]:


print(df_2016.columns)


# In[43]:


# Renommer GHGEmissionsIntensity -> GHGEmissionsIntensity(kgCO2e/ft2)
df_2016 = df_2016.rename(columns={'GHGEmissionsIntensity': 'GHGEmissionsIntensity(kgCO2e/ft2)'})


# In[44]:


# Renommer TotalGHGEmissions -> TotalGHGEmissions(MetricTonsCO2e)
df_2016 = df_2016.rename(columns={'TotalGHGEmissions': 'TotalGHGEmissions(MetricTonsCO2e)'})


# In[45]:


print(df_2016.columns)


# In[46]:


# afficher les colonnes en commun
colomns_in_common = df_2015.columns & df_2016.columns
colomns_in_common


# In[47]:


# concatener les deux dataframes df_2015 et df_2016
df_2015_2016 = pd.concat([df_2015[colomns_in_common], df_2016[colomns_in_common]], axis=0, sort=False)
df_2015_2016.shape


# In[48]:


df_2015_2016.head()


# In[49]:


df_2015_2016.tail()


# In[50]:


df_2015_2016.dtypes.value_counts()


# In[51]:


plt.figure(figsize=(20,10))
sns.heatmap(df_2015_2016.isna(), cbar=False)


# In[52]:


(df_2015_2016.isna().sum()/df_2015_2016.shape[0]).sort_values(ascending=True)


# In[53]:


# on retire SecondLargest et ThirdLargest
# => on retire les colonnes avec +50% de valeurs manquantes
df_2015_2016 = df_2015_2016[df_2015_2016.columns[df_2015_2016.isna().sum()/df_2015_2016.shape[0] < 0.5]]
df_2015_2016.head()


# In[54]:


df_2015_2016.shape


# In[55]:


# on filtre sur NonResidential dans BuildingType
df_2015_2016 = df_2015_2016[df_2015_2016['BuildingType'].isin(['NonResidential'])]
df_2015_2016.shape


# In[56]:


df_2015_2016['ComplianceStatus']


# In[57]:


df_2015_2016['ComplianceStatus'].value_counts()


# In[58]:


df_2015_2016['DefaultData']


# In[59]:


df_2015_2016['DefaultData'].value_counts()


# In[60]:


# on filtre sur Compliant dans ComplianceStatus
df_2015_2016 = df_2015_2016[df_2015_2016['ComplianceStatus'].isin(['Compliant'])]
df_2015_2016.shape


# In[61]:


df_2015_2016['DefaultData'].value_counts()


# In[62]:


# retirer les Yes
df_2015_2016 = df_2015_2016[df_2015_2016['DefaultData']!='Yes']


# In[63]:


df_2015_2016.shape


# In[64]:


df_2015_2016.head()


# In[65]:


df_2015_2016.to_csv(r"C:\Users\SAMSUNG\Desktop\OpenClassrooms\P4_OC\P4_Donnees\2015_2016_notnorm_Building_Energy_Benchmarking.csv", index=False)


# In[66]:


columns = df_2015_2016.columns
columns


# In[67]:


df_2015_2016[["PropertyGFATotal", "PropertyGFAParking", "PropertyGFABuilding(s)", "LargestPropertyUseTypeGFA"]].head() 
# PropertyGFATotal = PropertyGFAParking + PropertyGFABuilding(s)


# In[68]:


df_2015_2016[["GHGEmissionsIntensity(kgCO2e/ft2)", "TotalGHGEmissions(MetricTonsCO2e)", "PropertyGFATotal"]].head() 
# GHGEmissionsIntensity(kgCO2e/ft2) ~ TotalGHGEmissions(MetricTonsCO2e) / PropertyGFATotal
# (ex : 1.99, 1.80, 0.72, 2.83)


# In[69]:


df_2015_2016 = df_2015_2016[['OSEBuildingID', 'DataYear', 'BuildingType', 'PrimaryPropertyType','PropertyName', 
                             'TaxParcelIdentificationNumber', 'CouncilDistrictCode','Neighborhood', 'YearBuilt', 
                             'NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking', 
                             'PropertyGFABuilding(s)', 'ListOfAllPropertyUseTypes', 'LargestPropertyUseType',
                             'LargestPropertyUseTypeGFA', 'ENERGYSTARScore', 'SiteEUI(kBtu/sf)', 'SourceEUI(kBtu/sf)', 
                             'SiteEnergyUse(kBtu)',  'SteamUse(kBtu)', 'Electricity(kBtu)', 'NaturalGas(kBtu)', 
                             'TotalGHGEmissions(MetricTonsCO2e)', 'GHGEmissionsIntensity(kgCO2e/ft2)', 'DefaultData', 
                             'ComplianceStatus', 'ZipCode', 'Latitude', 'Longitude']]
# SiteEUIWN(kBtu/sf), SourceEUIWN(kBtu/sf), SiteEnergyUseWN(kBtu) : projections sur 30 ans
# NaturalGas(therms), Electricity(kWh) : on considère la mesure kbtu 


# In[70]:


df_2015_2016.describe()


# In[71]:


df_2015_2016.describe().shape


# ## Avec energyscore :

# In[72]:


# liste de colonnes numériques pour la regression avec ENERGYSTARScore (list_cols_training_with_energyscore) 
list_cols_training_with_energyscore = df_2015_2016.describe().columns
list_cols_training_with_energyscore


# In[73]:


list_cols_training_with_energyscore = df_2015_2016[['OSEBuildingID', 'DataYear', 'CouncilDistrictCode', 'LargestPropertyUseType', 'YearBuilt',
                                                    'NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal',
                                                    'PropertyGFAParking', 'PropertyGFABuilding(s)',
                                                    'LargestPropertyUseTypeGFA', 'ENERGYSTARScore', 'SiteEUI(kBtu/sf)',
                                                    'SourceEUI(kBtu/sf)', 'SiteEnergyUse(kBtu)',  'SteamUse(kBtu)',
                                                    'Electricity(kBtu)', 'NaturalGas(kBtu)', 'TotalGHGEmissions(MetricTonsCO2e)',
                                                    'GHGEmissionsIntensity(kgCO2e/ft2)', 'ZipCode', 'Latitude', 'Longitude']]


# In[74]:


list_cols_training_with_energyscore.shape


# # 2. Analyse de fond :

# ## Boxplot

# In[75]:


for col in list_cols_training_with_energyscore[['NumberofFloors', 'TotalGHGEmissions(MetricTonsCO2e)', 'SiteEnergyUse(kBtu)']]:
    plt.figure()
    sns.boxplot(x=list_cols_training_with_energyscore[col])


# ## Analyse des outliers sur 2/3 variables

# In[76]:


df1_2015_2016 = list_cols_training_with_energyscore['OSEBuildingID']


# In[77]:


df2_2015_2016 = list_cols_training_with_energyscore['DataYear']


# In[78]:


df3_2015_2016 = list_cols_training_with_energyscore['CouncilDistrictCode']


# In[79]:


df4_2015_2016 = list_cols_training_with_energyscore['LargestPropertyUseType']


# In[80]:


df5_2015_2016 = list_cols_training_with_energyscore['YearBuilt']


# In[81]:


df6_2015_2016 = list_cols_training_with_energyscore['NumberofBuildings']


# In[82]:


# nbr floors
Q3 = list_cols_training_with_energyscore['NumberofFloors'].quantile(0.75)
Q1 = list_cols_training_with_energyscore['NumberofFloors'].quantile(0.25)
IQR = Q3 - Q1
filter=list_cols_training_with_energyscore['NumberofFloors']<= Q3 + 1.5 * IQR
df7_2015_2016 = list_cols_training_with_energyscore[filter]


# In[83]:


df8_2015_2016 = list_cols_training_with_energyscore['PropertyGFATotal']


# In[84]:


df9_2015_2016 = list_cols_training_with_energyscore['PropertyGFAParking']


# In[85]:


df10_2015_2016 = list_cols_training_with_energyscore['PropertyGFABuilding(s)']


# In[86]:


df11_2015_2016 = list_cols_training_with_energyscore['LargestPropertyUseTypeGFA']


# In[87]:


df12_2015_2016 = list_cols_training_with_energyscore['ENERGYSTARScore']


# In[88]:


df13_2015_2016 = list_cols_training_with_energyscore['SiteEUI(kBtu/sf)']


# In[89]:


# target 
Q3 = list_cols_training_with_energyscore['SiteEnergyUse(kBtu)'].quantile(0.75)
Q1 = list_cols_training_with_energyscore['SiteEnergyUse(kBtu)'].quantile(0.25)
IQR = Q3 - Q1
filter=list_cols_training_with_energyscore['SiteEnergyUse(kBtu)']<= Q3 + 1.5 * IQR
df14_2015_2016 = list_cols_training_with_energyscore['SiteEnergyUse(kBtu)']


# In[90]:


df15_2015_2016= list_cols_training_with_energyscore['SteamUse(kBtu)']


# In[91]:


df16_2015_2016 = list_cols_training_with_energyscore['Electricity(kBtu)']


# In[92]:


df17_2015_2016 = list_cols_training_with_energyscore['NaturalGas(kBtu)']


# In[93]:


# target
Q3 = list_cols_training_with_energyscore['TotalGHGEmissions(MetricTonsCO2e)'].quantile(0.75)
Q1 = list_cols_training_with_energyscore['TotalGHGEmissions(MetricTonsCO2e)'].quantile(0.25)
IQR = Q3 - Q1
filter=list_cols_training_with_energyscore['TotalGHGEmissions(MetricTonsCO2e)']<= Q3 + 1.5 * IQR
df18_2015_2016 = list_cols_training_with_energyscore[filter]


# In[94]:


df19_2015_2016 = list_cols_training_with_energyscore['GHGEmissionsIntensity(kgCO2e/ft2)']


# In[95]:


df20_2015_2016 = list_cols_training_with_energyscore['ZipCode']


# In[96]:


df21_2015_2016 = list_cols_training_with_energyscore['Latitude']


# In[97]:


df22_2015_2016 = list_cols_training_with_energyscore['Longitude']


# In[98]:


union_index = df1_2015_2016.index.union(df2_2015_2016.index).union(df3_2015_2016.index).union(df4_2015_2016.index).union(df5_2015_2016.index).union(df6_2015_2016.index).union(df7_2015_2016.index).union(df8_2015_2016.index).union(df9_2015_2016.index).union(df10_2015_2016.index).union(df11_2015_2016.index).union(df12_2015_2016.index).union(df13_2015_2016.index).union(df14_2015_2016.index).union(df15_2015_2016.index).union(df16_2015_2016.index).union(df17_2015_2016.index).union(df18_2015_2016.index).union(df19_2015_2016.index).union(df20_2015_2016.index).union(df21_2015_2016.index).union(df22_2015_2016.index)


# In[99]:


list_cols_training_with_energyscore = list_cols_training_with_energyscore.loc[union_index]
list_cols_training_with_energyscore


# In[100]:


list_cols_training_with_energyscore.shape # shape après analyse outliers > shape avant analyse outliers -> bizarre


# In[101]:


plt.figure(figsize=(20,10))
sns.heatmap(list_cols_training_with_energyscore.isna(), cbar=False)


# In[102]:


(list_cols_training_with_energyscore.isna().sum()/list_cols_training_with_energyscore.shape[0]).sort_values(ascending=True)


# In[103]:


list_cols_training_with_energyscore.shape


# In[104]:


list_cols_training_with_energyscore[list_cols_training_with_energyscore.duplicated(['OSEBuildingID'])].shape # Seattle OSE Building Identification Number (Building ID) 
                                                                               # A unique building identification number
                                                                               # assigned by OSE to each covered building to facilitate annual benchmarking submission and compliance tracking.


# In[105]:


# retrait des doublons
list_cols_training_with_energyscore = list_cols_training_with_energyscore.drop_duplicates(keep='first')


# In[106]:


list_cols_training_with_energyscore.shape


# In[107]:


# retrait des NaN
list_cols_training_with_energyscore = list_cols_training_with_energyscore.dropna()


# In[108]:


list_cols_training_with_energyscore.shape


# In[109]:


print(list_cols_training_with_energyscore.columns)


# ## Distribution

# In[110]:


for col in list_cols_training_with_energyscore[['OSEBuildingID', 'DataYear', 'CouncilDistrictCode', 'YearBuilt', 'NumberofBuildings',
                                                'NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking','PropertyGFABuilding(s)', 
                                                'LargestPropertyUseTypeGFA','ENERGYSTARScore', 'SiteEUI(kBtu/sf)', 'SourceEUI(kBtu/sf)',
                                                'SiteEnergyUse(kBtu)', 'SteamUse(kBtu)', 'Electricity(kBtu)', 'NaturalGas(kBtu)', 
                                                'TotalGHGEmissions(MetricTonsCO2e)', 'GHGEmissionsIntensity(kgCO2e/ft2)', 'ZipCode', 'Latitude',
                                                'Longitude']]:
    plt.figure()
    sns.distplot(list_cols_training_with_energyscore[col])


# ## Target

# In[111]:


# retrait valeurs à 0 dans la target # TotalGHGEmissions(MetricTonsCO2e)
# Sur ce filtre, il est important de regarder combien de valeurs on retire afin de vérifier que l'on ne supprime pas toutes les données
nb_rows_removed = list_cols_training_with_energyscore[~((list_cols_training_with_energyscore["TotalGHGEmissions(MetricTonsCO2e)"].notnull()) & (list_cols_training_with_energyscore["TotalGHGEmissions(MetricTonsCO2e)"] > 0))].shape[0]
print("Number of rows removed : {}".format(nb_rows_removed))
list_cols_training_with_energyscore = list_cols_training_with_energyscore[(list_cols_training_with_energyscore["TotalGHGEmissions(MetricTonsCO2e)"].notnull()) & (list_cols_training_with_energyscore["TotalGHGEmissions(MetricTonsCO2e)"] > 0)]


# In[112]:


# Dans ce cas, 2 données est acceptable. 


# In[113]:


# retrait valeurs à 0 dans la target # SiteEnergyUse(kBtu)
# Sur ce filtre, il est important de regarder combien de valeurs on retire afin de vérifier que l'on ne supprime pas toutes les données
nb_rows_removed = list_cols_training_with_energyscore[~((list_cols_training_with_energyscore["SiteEnergyUse(kBtu)"].notnull()) & (list_cols_training_with_energyscore["SiteEnergyUse(kBtu)"] > 0))].shape[0]
print("Number of rows removed : {}".format(nb_rows_removed))
list_cols_training_with_energyscore = list_cols_training_with_energyscore[(list_cols_training_with_energyscore["SiteEnergyUse(kBtu)"].notnull()) & (list_cols_training_with_energyscore["SiteEnergyUse(kBtu)"] > 0)]


# In[114]:


# Dans ce cas, 0 donnée. 


# In[115]:


# target # TotalGHGEmissions(MetricTonsCO2e)
# passage au log
np.log(list_cols_training_with_energyscore["TotalGHGEmissions(MetricTonsCO2e)"]).hist(bins=100)


# In[116]:


# target # SiteEnergyUse(kBtu)
# passage au log
np.log(list_cols_training_with_energyscore["SiteEnergyUse(kBtu)"]).hist(bins=100)


# In[117]:


# intégrer Log_TotalGHGEmissions(MetricTonsCO2e) dans le dataframe
list_cols_training_with_energyscore["Log_TotalGHGEmissions(MetricTonsCO2e)"] = np.log(list_cols_training_with_energyscore["TotalGHGEmissions(MetricTonsCO2e)"])


# In[118]:


# intégrer Log_SiteEnergyUse(kBtu) dans le dataframe
list_cols_training_with_energyscore["Log_SiteEnergyUse(kBtu)"] = np.log(list_cols_training_with_energyscore["SiteEnergyUse(kBtu)"])


# In[119]:


print(list_cols_training_with_energyscore.columns)


# In[120]:


# supprimer les colonnes # TotalGHGEmissions(MetricTonsCO2e) # SiteEnergyUse(kBtu)
list_cols_training_with_energyscore.drop(['TotalGHGEmissions(MetricTonsCO2e)', 'SiteEnergyUse(kBtu)'], axis='columns', inplace=True)


# In[121]:


print(list_cols_training_with_energyscore.columns)


# In[122]:


list_cols_training_with_energyscore.shape


# ## Variables

# In[123]:


# normaliser DataYear
((list_cols_training_with_energyscore['DataYear']-list_cols_training_with_energyscore['DataYear'].mean())/list_cols_training_with_energyscore['DataYear'].std()).hist(bins=100)


# In[124]:


# intégrer N_DataYear dans le dataframe
list_cols_training_with_energyscore["N_DataYear"] = (list_cols_training_with_energyscore['DataYear']-list_cols_training_with_energyscore['DataYear'].mean())/list_cols_training_with_energyscore['DataYear'].std()


# In[125]:


# normaliser YearBuilt
((list_cols_training_with_energyscore['YearBuilt']-list_cols_training_with_energyscore['YearBuilt'].mean())/list_cols_training_with_energyscore['YearBuilt'].std()).hist(bins=100)


# In[126]:


# intégrer N_YearBuilt dans le dataframe
list_cols_training_with_energyscore["N_YearBuilt"] = (list_cols_training_with_energyscore['YearBuilt']-list_cols_training_with_energyscore['YearBuilt'].mean())/list_cols_training_with_energyscore['YearBuilt'].std()


# In[127]:


# normaliser NumberofBuildings
((list_cols_training_with_energyscore['NumberofBuildings']-list_cols_training_with_energyscore['NumberofBuildings'].mean())/list_cols_training_with_energyscore['NumberofBuildings'].std()).hist(bins=100)


# In[128]:


# intégrer N_NumberofBuildings dans le dataframe
list_cols_training_with_energyscore["N_NumberofBuildings"] = (list_cols_training_with_energyscore['NumberofBuildings']-list_cols_training_with_energyscore['NumberofBuildings'].mean())/list_cols_training_with_energyscore['NumberofBuildings'].std()


# In[129]:


# retrait valeurs à 0 dans la target # NumberofFloors
# non, car on souhaite conserver les étages "0" dans l'étude


# In[130]:


# normaliser NumberofFloors
((list_cols_training_with_energyscore['NumberofFloors']-list_cols_training_with_energyscore['NumberofFloors'].mean())/list_cols_training_with_energyscore['NumberofFloors'].std()).hist(bins=100)


# In[131]:


# intégrer N_NumberofFloors dans le dataframe
list_cols_training_with_energyscore["N_NumberofFloors"] = (list_cols_training_with_energyscore['NumberofFloors']-list_cols_training_with_energyscore['NumberofFloors'].mean())/list_cols_training_with_energyscore['NumberofFloors'].std()


# In[132]:


# retrait valeurs à 0 dans la target # TotalGHGEmissions
# Sur ce filtre, il est important de regarder combien de valeurs on retire afin de vérifier que l'on ne supprime pas toutes les données
#nb_rows_removed = num_cols_training_with_energyscore[~((num_cols_training_with_energyscore["PropertyGFATotal"].notnull()) & (num_cols_training_with_energyscore["PropertyGFATotal"] > 0))].shape[0]
#print("Number of rows removed : {}".format(nb_rows_removed))
#num_cols_training_with_energyscore = num_cols_training_with_energyscore[(num_cols_training_with_energyscore["PropertyGFATotal"].notnull()) & (num_cols_training_with_energyscore["PropertyGFATotal"] > 0)]


# In[133]:


# Dans ce cas, 1482 données sur un total de 1956 données


# In[134]:


# normaliser PropertyGFATotal
((list_cols_training_with_energyscore['PropertyGFATotal']-list_cols_training_with_energyscore['PropertyGFATotal'].mean())/list_cols_training_with_energyscore['PropertyGFATotal'].std()).hist(bins=100)


# In[135]:


# intégrer N_PropertyGFATotal dans le dataframe
list_cols_training_with_energyscore["N_PropertyGFATotal"] = (list_cols_training_with_energyscore['PropertyGFATotal']-list_cols_training_with_energyscore['PropertyGFATotal'].mean())/list_cols_training_with_energyscore['PropertyGFATotal'].std()


# In[136]:


# normaliser PropertyGFAParking
((list_cols_training_with_energyscore['PropertyGFAParking']-list_cols_training_with_energyscore['PropertyGFAParking'].mean())/list_cols_training_with_energyscore['PropertyGFAParking'].std()).hist(bins=100)


# In[137]:


# intégrer N_PropertyGFAParking dans le dataframe
list_cols_training_with_energyscore["N_PropertyGFAParking"] = (list_cols_training_with_energyscore['PropertyGFAParking']-list_cols_training_with_energyscore['PropertyGFAParking'].mean())/list_cols_training_with_energyscore['PropertyGFAParking'].std()


# In[138]:


# normaliser PropertyGFABuilding(s)
((list_cols_training_with_energyscore['PropertyGFABuilding(s)']-list_cols_training_with_energyscore['PropertyGFABuilding(s)'].mean())/list_cols_training_with_energyscore['PropertyGFABuilding(s)'].std()).hist(bins=100)


# In[139]:


# intégrer N_PropertyGFABuilding(s) dans le dataframe
list_cols_training_with_energyscore["N_PropertyGFABuilding(s)"] = (list_cols_training_with_energyscore['PropertyGFABuilding(s)']-list_cols_training_with_energyscore['PropertyGFABuilding(s)'].mean())/list_cols_training_with_energyscore['PropertyGFABuilding(s)'].std()


# In[140]:


# normaliser LargestPropertyUseTypeGFA
((list_cols_training_with_energyscore['LargestPropertyUseTypeGFA']-list_cols_training_with_energyscore['LargestPropertyUseTypeGFA'].mean())/list_cols_training_with_energyscore['LargestPropertyUseTypeGFA'].std()).hist(bins=100)


# In[141]:


# intégrer N_LargestPropertyUseTypeGFA dans le dataframe
list_cols_training_with_energyscore["N_LargestPropertyUseTypeGFA"] = (list_cols_training_with_energyscore['LargestPropertyUseTypeGFA']-list_cols_training_with_energyscore['LargestPropertyUseTypeGFA'].mean())/list_cols_training_with_energyscore['LargestPropertyUseTypeGFA'].std()


# In[142]:


# normaliser ENERGYSTARScore
((list_cols_training_with_energyscore['ENERGYSTARScore']-list_cols_training_with_energyscore['ENERGYSTARScore'].mean())/list_cols_training_with_energyscore['ENERGYSTARScore'].std()).hist(bins=100)


# In[143]:


# intégrer N_ENERGYSTARScore dans le dataframe
list_cols_training_with_energyscore["N_ENERGYSTARScore"] = (list_cols_training_with_energyscore['ENERGYSTARScore']-list_cols_training_with_energyscore['ENERGYSTARScore'].mean())/list_cols_training_with_energyscore['ENERGYSTARScore'].std()


# In[144]:


# normaliser SiteEUI(kBtu/sf)
((list_cols_training_with_energyscore['SiteEUI(kBtu/sf)']-list_cols_training_with_energyscore['SiteEUI(kBtu/sf)'].mean())/list_cols_training_with_energyscore['SiteEUI(kBtu/sf)'].std()).hist(bins=100)


# In[145]:


# intégrer N_SiteEUI(kBtu/sf) dans le dataframe
list_cols_training_with_energyscore["N_SiteEUI(kBtu/sf)"] = (list_cols_training_with_energyscore['SiteEUI(kBtu/sf)']-list_cols_training_with_energyscore['SiteEUI(kBtu/sf)'].mean())/list_cols_training_with_energyscore['SiteEUI(kBtu/sf)'].std()


# In[146]:


# normaliser SourceEUI(kBtu/sf)
((list_cols_training_with_energyscore['SourceEUI(kBtu/sf)']-list_cols_training_with_energyscore['SourceEUI(kBtu/sf)'].mean())/list_cols_training_with_energyscore['SourceEUI(kBtu/sf)'].std()).hist(bins=100)


# In[147]:


# intégrer N_SourceEUI(kBtu/sf) dans le dataframe
list_cols_training_with_energyscore["N_SourceEUI(kBtu/sf)"] = (list_cols_training_with_energyscore['SourceEUI(kBtu/sf)']-list_cols_training_with_energyscore['SourceEUI(kBtu/sf)'].mean())/list_cols_training_with_energyscore['SourceEUI(kBtu/sf)'].std()


# In[148]:


# normaliser SteamUse(kBtu)
((list_cols_training_with_energyscore['SteamUse(kBtu)']-list_cols_training_with_energyscore['SteamUse(kBtu)'].mean())/list_cols_training_with_energyscore['SteamUse(kBtu)'].std()).hist(bins=100)


# In[149]:


# intégrer N_SteamUse(kBtu) dans le dataframe
list_cols_training_with_energyscore["N_SteamUse(kBtu)"] = (list_cols_training_with_energyscore['SteamUse(kBtu)']-list_cols_training_with_energyscore['SteamUse(kBtu)'].mean())/list_cols_training_with_energyscore['SteamUse(kBtu)'].std()


# In[150]:


# normaliser Electricity(kBtu)
((list_cols_training_with_energyscore['Electricity(kBtu)']-list_cols_training_with_energyscore['Electricity(kBtu)'].mean())/list_cols_training_with_energyscore['Electricity(kBtu)'].std()).hist(bins=100)


# In[151]:


# intégrer N_Electricity(kBtu) dans le dataframe
list_cols_training_with_energyscore["N_Electricity(kBtu)"] = (list_cols_training_with_energyscore['Electricity(kBtu)']-list_cols_training_with_energyscore['Electricity(kBtu)'].mean())/list_cols_training_with_energyscore['Electricity(kBtu)'].std()


# In[152]:


# normaliser NaturalGas(kBtu)
((list_cols_training_with_energyscore['NaturalGas(kBtu)']-list_cols_training_with_energyscore['NaturalGas(kBtu)'].mean())/list_cols_training_with_energyscore['NaturalGas(kBtu)'].std()).hist(bins=100)


# In[153]:


# intégrer N_NaturalGas(kBtu) dans le dataframe
list_cols_training_with_energyscore["N_NaturalGas(kBtu)"] = (list_cols_training_with_energyscore['NaturalGas(kBtu)']-list_cols_training_with_energyscore['NaturalGas(kBtu)'].mean())/list_cols_training_with_energyscore['NaturalGas(kBtu)'].std()


# In[154]:


# normaliser GHGEmissionsIntensity(kgCO2e/ft2)
((list_cols_training_with_energyscore['GHGEmissionsIntensity(kgCO2e/ft2)']-list_cols_training_with_energyscore['GHGEmissionsIntensity(kgCO2e/ft2)'].mean())/list_cols_training_with_energyscore['GHGEmissionsIntensity(kgCO2e/ft2)'].std()).hist(bins=100)


# In[155]:


# intégrer N_GHGEmissionsIntensity(kgCO2e/ft2) dans le dataframe
list_cols_training_with_energyscore["N_GHGEmissionsIntensity(kgCO2e/ft2)"] = (list_cols_training_with_energyscore['GHGEmissionsIntensity(kgCO2e/ft2)']-list_cols_training_with_energyscore['GHGEmissionsIntensity(kgCO2e/ft2)'].mean())/list_cols_training_with_energyscore['GHGEmissionsIntensity(kgCO2e/ft2)'].std()


# In[156]:


# normaliser Latitude
((list_cols_training_with_energyscore['Latitude']-list_cols_training_with_energyscore['Latitude'].mean())/list_cols_training_with_energyscore['Latitude'].std()).hist(bins=100)


# In[157]:


# intégrer N_Latitude dans le dataframe
list_cols_training_with_energyscore["N_Latitude"] = (list_cols_training_with_energyscore['Latitude']-list_cols_training_with_energyscore['Latitude'].mean())/list_cols_training_with_energyscore['Latitude'].std()


# In[158]:


# normaliser Latitude
((list_cols_training_with_energyscore['Longitude']-list_cols_training_with_energyscore['Longitude'].mean())/list_cols_training_with_energyscore['Longitude'].std()).hist(bins=100)


# In[159]:


# intégrer N_Longitude dans le dataframe
list_cols_training_with_energyscore["N_Longitude"] = (list_cols_training_with_energyscore['Longitude']-list_cols_training_with_energyscore['Longitude'].mean())/list_cols_training_with_energyscore['Longitude'].std()


# In[160]:


print(list_cols_training_with_energyscore.columns)


# In[161]:


# supprimer les colonnes ['OSEBuildingID', 'DataYear', 'CouncilDistrictCode', 'YearBuilt','NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal',
                    #     'PropertyGFAParking', 'PropertyGFABuilding(s)','LargestPropertyUseTypeGFA', 'ENERGYSTARScore', 'SiteEUI(kBtu/sf)',
                    #     'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)','SiteEnergyUseWN(kBtu)', 'SteamUse(kBtu)', 'Electricity(kWh)',
                    #     'Electricity(kBtu)', 'NaturalGas(therms)', 'NaturalGas(kBtu)',
                    #     'GHGEmissionsIntensity(kgCO2e/ft2)', 'ZipCode', 'Latitude', 'Longitude]
num_cols_training_with_energyscore = list_cols_training_with_energyscore[['Log_TotalGHGEmissions(MetricTonsCO2e)', 'Log_SiteEnergyUse(kBtu)',
                                                                          'N_DataYear', 'N_YearBuilt', 'N_NumberofBuildings', 'N_NumberofFloors',
                                                                          'N_PropertyGFATotal', 'N_PropertyGFAParking', 
                                                                          'N_PropertyGFABuilding(s)', 'N_LargestPropertyUseTypeGFA', 'N_ENERGYSTARScore', 
                                                                          'N_SiteEUI(kBtu/sf)', 'N_SourceEUI(kBtu/sf)', 'N_SteamUse(kBtu)',
                                                                          'N_Electricity(kBtu)', 'N_NaturalGas(kBtu)', 'N_GHGEmissionsIntensity(kgCO2e/ft2)',
                                                                          'N_Latitude', 'N_Longitude']]


# In[162]:


print(num_cols_training_with_energyscore.columns)


# In[163]:


num_cols_training_with_energyscore.shape 


# In[164]:


print(num_cols_training_with_energyscore.columns)


# # 3. Analyse plus détaillée

# In[165]:


# pairplot # property
sns.pairplot(num_cols_training_with_energyscore[['Log_TotalGHGEmissions(MetricTonsCO2e)', 'Log_SiteEnergyUse(kBtu)', 'N_ENERGYSTARScore','N_NumberofFloors', 
                 'N_PropertyGFAParking','N_LargestPropertyUseTypeGFA']])


# In[166]:


# pairplot # énergie (~ "forte corrélation")
sns.pairplot(num_cols_training_with_energyscore[['Log_TotalGHGEmissions(MetricTonsCO2e)', 'Log_SiteEnergyUse(kBtu)', 'N_ENERGYSTARScore','N_SiteEUI(kBtu/sf)', 
                 'N_SourceEUI(kBtu/sf)', 'N_Electricity(kBtu)', 'N_GHGEmissionsIntensity(kgCO2e/ft2)']])


# In[167]:


# pairplot # énergie ("faible corrélation")
sns.pairplot(num_cols_training_with_energyscore[['Log_TotalGHGEmissions(MetricTonsCO2e)', 'Log_SiteEnergyUse(kBtu)', 'N_ENERGYSTARScore','N_SteamUse(kBtu)', 'N_NaturalGas(kBtu)']])


# ## Matrice de corrélation

# In[172]:


num_cols_training_with_energyscore.corr()


# In[173]:


sns.clustermap(num_cols_training_with_energyscore.corr())


# In[174]:


corrMatrix = num_cols_training_with_energyscore[['Log_TotalGHGEmissions(MetricTonsCO2e)', 'Log_SiteEnergyUse(kBtu)',
                                                 'N_NumberofBuildings', 'N_NumberofFloors','N_PropertyGFATotal', 
                                                 'N_PropertyGFAParking','N_PropertyGFABuilding(s)', 
                                                 'N_LargestPropertyUseTypeGFA']].corr()
sns.heatmap(corrMatrix, annot=True)
plt.show() # le faire par petit groupe # heatmap plus facile à interpréter que le clustermap


# In[175]:


corrMatrix = num_cols_training_with_energyscore[['Log_TotalGHGEmissions(MetricTonsCO2e)', 'Log_SiteEnergyUse(kBtu)',
                                                 'N_SiteEUI(kBtu/sf)', 'N_SourceEUI(kBtu/sf)','N_SteamUse(kBtu)', 'N_Electricity(kBtu)', 
                                                 'N_NaturalGas(kBtu)','N_GHGEmissionsIntensity(kgCO2e/ft2)']].corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()                      


# In[176]:


corrMatrix = num_cols_training_with_energyscore[['Log_TotalGHGEmissions(MetricTonsCO2e)', 'Log_SiteEnergyUse(kBtu)',
                                                 'N_ENERGYSTARScore', 'N_DataYear', 'N_YearBuilt', 'N_Latitude', 'N_Longitude']].corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()    


# # One-hot-encoding

# In[177]:


# liste de colonnes catégorielles pour la regression avec ENERGYSTARScore (cat_cols_training_with_energyscore)
cat_cols_training_with_energyscore = list_cols_training_with_energyscore[['LargestPropertyUseType', 'CouncilDistrictCode', 'ZipCode']]
cat_cols_training_with_energyscore.shape


# In[178]:


print(cat_cols_training_with_energyscore['LargestPropertyUseType'])


# In[179]:


cat_cols_training_with_energyscore['LargestPropertyUseType'].value_counts()


# In[180]:


cat_cols_training_with_energyscore['LargestPropertyUseType'].value_counts().shape


# In[181]:


cat_cols_training_with_energyscore['CouncilDistrictCode'].value_counts()


# In[182]:


cat_cols_training_with_energyscore['CouncilDistrictCode'].value_counts().shape


# In[183]:


cat_cols_training_with_energyscore['ZipCode'].value_counts()


# In[184]:


cat_cols_training_with_energyscore['ZipCode'].value_counts().shape


# In[185]:


pd.get_dummies(cat_cols_training_with_energyscore["LargestPropertyUseType"],prefix='LargestPropertyUseType',drop_first=True)


# In[186]:


pd.get_dummies(cat_cols_training_with_energyscore["CouncilDistrictCode"],prefix='CouncilDistrictCode',drop_first=True)


# In[187]:


pd.get_dummies(cat_cols_training_with_energyscore["ZipCode"],prefix='ZipCode',drop_first=True)


# In[188]:


# ajouter des colonnes factices au dataframe
# LargestPropertyUseType
cat_cols_training_with_energyscore = pd.concat([cat_cols_training_with_energyscore,pd.get_dummies(cat_cols_training_with_energyscore['LargestPropertyUseType'], prefix='LargestPropertyUseType')],axis=1)
cat_cols_training_with_energyscore


# In[189]:


# CouncilDistrictCode
cat_cols_training_with_energyscore = pd.concat([cat_cols_training_with_energyscore,pd.get_dummies(cat_cols_training_with_energyscore['CouncilDistrictCode'], prefix='CouncilDistrictCode')],axis=1)
cat_cols_training_with_energyscore


# In[190]:


# ZipCode
cat_cols_training_with_energyscore = pd.concat([cat_cols_training_with_energyscore,pd.get_dummies(cat_cols_training_with_energyscore['ZipCode'], prefix='ZipCode')],axis=1)
cat_cols_training_with_energyscore


# In[191]:


# supprimer les colonnes originales LargestPropertyUseType, CouncilDistrictCode, ZipCode
cat_cols_training_with_energyscore = cat_cols_training_with_energyscore.drop(['LargestPropertyUseType', 'CouncilDistrictCode', 'ZipCode'],1)
cat_cols_training_with_energyscore


# In[192]:


df = pd.concat([cat_cols_training_with_energyscore, num_cols_training_with_energyscore], axis = 1)
df


# In[193]:


df.to_csv(r"C:\Users\SAMSUNG\Desktop\OpenClassrooms\P4_OC\P4_Donnees\2015_2016_Building_Energy_Benchmarking.csv", index=False)

