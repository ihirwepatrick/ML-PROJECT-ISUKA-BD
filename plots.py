#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


air_quality = pd.read_csv("./air_quality_no2.csv", index_col=0, parse_dates=True)
air_quality.head()


# In[ ]:





# In[ ]:





# In[11]:


air_quality.plot()


# In[12]:


air_quality["station_paris"].plot()


# In[13]:


air_quality.plot.scatter(x="station_london", y="station_paris", alpha=0.5)


# In[14]:


[
    method_name
    for method_name in dir(air_quality.plot)
    if not method_name.startswith("_")
]


# In[37]:


air_quality.plot.box()
plt.show()


# In[38]:


fig, axs = plt.subplots(figsize=(12, 4))


# In[58]:


air_quality.plot.area(ax=axs)
axs.set_ylabel("NO$_2$ concentration")
fig.savefig("no2_concentrations.png")
plt.show()


# In[62]:


sns.set(style="whitegrid")  # Apply a whitegrid style to the plot


# In[64]:


fig, axs = plt.subplots(figsize=(12, 6))  # Slightly larger for better visibility
air_quality.plot.area(ax=axs, alpha=0.6, colormap='viridis')  # Add transparency and a colormap

axs.set_ylabel("NO$_2$ concentration", fontsize=14)
axs.set_xlabel("Date", fontsize=14)
axs.set_title("NO$_2$ Concentrations Over Time", fontsize=16)

axs.grid(True, linestyle='--', alpha=0.7)  # Enhance grid lines
axs.legend(title='Station', fontsize=12, title_fontsize=14)  # Customize legend
sns.despine()  # Remove top and right spines for a cleaner look

fig.savefig("styled_no2_concentrations.svg", dpi=300)  # Higher resolution for better quality
plt.show()


# In[1]:


import pandas as pd
import plotly.express as px
import plotly.io as pio

# Sample DataFrame (replace this with your actual data)
data = {
    'level_0': range(5),
    'index': range(5),
    'datetime': pd.date_range(start='1/1/2021', periods=5, freq='D'),
    'station_antwerp': [20, 21, 19, 18, 17],
    'station_paris': [15, 16, 15, 14, 13],
    'station_london': [10, 11, 10, 9, 8]
}

air_quality = pd.DataFrame(data)

# Print the column names to check
print("Column names:", air_quality.columns)

# Convert 'datetime' to datetime if necessary
date_column = 'datetime'
air_quality[date_column] = pd.to_datetime(air_quality[date_column])

# Drop the 'level_0' and 'index' columns if they are not needed for plotting
air_quality = air_quality.drop(columns=['level_0', 'index'])

# Melt the DataFrame to long format
air_quality_long = air_quality.melt(id_vars=[date_column], 
                                    value_vars=['station_antwerp', 'station_paris', 'station_london'],
                                    var_name='Station', 
                                    value_name='NO2')

# Create the Plotly area plot
fig = px.area(air_quality_long, x=date_column, y='NO2', color='Station',
              title='NO2 Concentrations Over Time',
              labels={'NO2': 'NO2 Concentration', date_column: 'Date'},
              template='plotly_white')

# Customize the layout for better aesthetics
fig.update_layout(
    title_font=dict(size=20, family='Arial'),
    xaxis_title_font=dict(size=16, family='Arial'),
    yaxis_title_font=dict(size=16, family='Arial'),
    legend_title_font=dict(size=14, family='Arial'),
    hovermode='x unified'
)

fig.update_traces(mode='lines', hoverinfo='all')  # Enhance hover information

# Save interactive plot as HTML
pio.write_html(fig, file='interactive_no2_concentrations.html', auto_open=True)

# Display the interactive plot in a Jupyter notebook (if applicable)
fig.show()


# In[2]:


import kagglehub

# Download latest version
path = kagglehub.model_download("rishitdagli/plant-disease/tensorFlow2/plant-disease")

print("Path to model files:", path)

