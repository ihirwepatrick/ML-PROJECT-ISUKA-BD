#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
}

df = pd.DataFrame(data)
print(df)


# In[6]:


data = pd.read_csv(r'D:\DataAnalyticsProjects\5_CricketT20Analytics\data_collection\t20_csv_files\dim_match_summary.csv')

data


# In[8]:


data['match_id'].max()


# In[11]:


data['winner'][(data['team1'] == 'Namibia') | (data['team2'] == 'Namibia')]


# In[15]:


data.fillna(0, inplace=True)


# In[ ]:





# In[ ]:





# In[16]:


data.dtypes


# In[18]:


data.info()


# In[23]:


data.to_excel("myExportedSheet.xlsx", sheet_name="passengers", index=False)
print("Export successful!")


# In[ ]:





# In[25]:


theExcelData = pd.read_excel('./myExportedSheet.xlsx')
theExcelData


# In[29]:


titanic = pd.read_csv(r'C:\Users\Patrick\Downloads\titanic.csv')
titanic


# In[30]:


titanic[["Age", "Sex"]].shape


# In[31]:


titanic.fillna(0)


# In[32]:


type(titanic[["Age", "Sex"]])


# In[40]:


above_35 = titanic['Name'][titanic["Age"] > 35]
above_35


# In[41]:


titanic["Age"] > 35


# In[81]:


class_23_1 = titanic[titanic['Pclass'].isin([2, 3])]
class_23_1



# In[82]:


class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]
class_23


# In[84]:


print(class_23.equals(class_23_1))


# In[73]:


titanic.fillna(0, inplace = True);
titanic


# In[87]:


print(class_23.equals(class_23_1))


# In[96]:


age_no_na = titanic[titanic["Age"].notna()]
age_no_na


# In[97]:


titanic.iloc[9:25, 3:5]


# In[98]:


female_sex = titanic[titanic['Sex'] == 'female']
female_sex


# In[102]:


temp = titanic.iloc[0:3, 3] = "anonymous"
temp


# In[ ]:




