import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly as pl


file_path = "/Users/kushagrarajpurohit/Desktop/CmpSci_Project/Motor_Vehicle_Collisions_-_Crashes_20260428.csv"
df = pd.read_csv(file_path, low_memory=False)

df_clean = df.copy(deep=True)

# Drop high-missing columns and redundant collumns
df_clean = df_clean.drop(columns=[
    'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5',
    'CONTRIBUTING FACTOR VEHICLE 5', 'CONTRIBUTING FACTOR VEHICLE 4', 
    'CONTRIBUTING FACTOR VEHICLE 3', 'LOCATION',
    'OFF STREET NAME', 'CROSS STREET NAME','NUMBER_OF_PEDESTRIANS_INJURED','NUMBER_OF_PEDESTRIANS_KILLED'
    'NUMBER_OF_CYCLIST_INJURED','NUMBER_OF_CYCLIST_KILLED','NUMBER_OF_MOTORIST_INJURED','NUMBER_OF_MOTORIST_KILLED'
], errors='ignore')

'''
VEHICLE TYPE CODE 3,4,5  -- Had more that 80% of the data missing 
CONTRIBUTING FACTOR VEHICLE 3,4,5 -- Had more that 80% of the data missing
Location -- Redundant since already have Longitude and Latitude
OFF STREET NAME -- About 80% of this column was missing
CROSS STREET NAME -- NOT the street where the accident happened and ON STREET is more useful since its where the street where the accident ACTUALLY happened
'''



# Removing rows while they are still empty
# Drop rows if ALL 4 geo columns are missing
geo_cols = ['BOROUGH', 'ZIP CODE', 'LATITUDE', 'LONGITUDE']
df_clean = df_clean.dropna(subset=geo_cols, how='all')

#Drop rows if BOTH contributing factors are missing
df_clean = df_clean.dropna(subset=['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2'], how='all')

#Drop rows if BOTH vehicle type codes are missing
df_clean = df_clean.dropna(subset=['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2'], how='all')

# Drop row of empty vehicle type 1 since a few results(10) were put on the wrong vehicle collumn
df_clean = df_clean.dropna(subset='VEHICLE TYPE CODE 1', how='all')



#FILL STAGE: Label the remaining missing pieces
df_clean['CONTRIBUTING FACTOR VEHICLE 1'] = df_clean['CONTRIBUTING FACTOR VEHICLE 1'].fillna('Unspecified')
df_clean['CONTRIBUTING FACTOR VEHICLE 2'] = df_clean['CONTRIBUTING FACTOR VEHICLE 2'].fillna('Unspecified')
df_clean['BOROUGH'] = df_clean['BOROUGH'].fillna('NA')
df_clean['ZIP CODE'] = df_clean['ZIP CODE'].fillna('NA')
df_clean['NUMBER OF PERSONS INJURED'] = df_clean['NUMBER OF PERSONS INJURED'].fillna(0)
df_clean['NUMBER OF PERSONS KILLED'] = df_clean['NUMBER OF PERSONS KILLED'].fillna(0)
df_clean['VEHICLE TYPE CODE 2'] = df_clean['VEHICLE TYPE CODE 2'].fillna('NA')
df_clean['ON STREET NAME'] = df_clean['ON STREET NAME'].fillna('NA')

'''
CONTRIBUTING FACTOR VEHICLE 1,2 -- Unspecified is the NA for these 2 columns and would be better to stick to the formating of the data set.
BOROUGH and ZIP CODE -- 'Na' becuase its catgorical data
NUMBER OF PERSONS INJURED & NUMBER OF PERSONS KILLED -- 0 because its numeric data and if it was 'NA' it would change the collumn into a OBJ(string)
VEHICLE TYPE CODE 2 -- 'Na' becuase its catgorical data
ON STREET NAME -- 'Na' becuase its catgorical data
'''


#-----------------Additional filtering & Grouping-----------------------

bike_variants = ['BICYCLE', 'BICYC', 'BIKE', 'BK']

moto_variants = [
    'MOTORBIKE', '2 WHE', '2YDSWHEELL', '50 CC MOTO', '50CC MINI', '50CC SCOOT', 
    'DIRT BIKE', 'DIRTB', 'DIRTBIKE', 'MNI-MOTORC', 'MO PA', 'MO PE', 'MO PED', 
    'MO-PE', 'MO-PED', 'MOBILITY', 'MOBILITY S', 'MOBILTY SC', 'MOOPER', 'MOP', 
    'MOP PAD', 'MOPAD', 'MOPD', 'MOPED', 'MOPED 150C', 'MOPED BIKE', 'MOPED CLAS', 
    'MOPED GAS', 'MOPED GASO', 'MOPED SCOO', 'MOPEN', 'MOPER', 'MOPET', 'MOPOED', 
    'MOPPED', 'MOT', 'MOT S', 'MOTER', 'MOTO-SCOOT', 'MOTOR', 'MOTOR DIRT', 
    'MOTOR SCOO', 'MOTOR UNIC', 'MOTOR WHEE', 'MOTOR. SCO', 'MOTORBIKE', 
    'MOTORCYCLE', 'MOTORED SC', 'MOTORHOME', 'MOTORIST S', 'MOTORIZED', 
    'MOTORIZED HOME', 'MOTORIZEDS', 'MOTORSCOOT', 'MOTORSCOOTER', 'YAMAH', 'YAMAHA'
]

ebike_variants = [
    'E-SCOOTER', 'E - B', 'E AMB', 'E BIK', 'E BIKE', 'E BIKE NO', 'E BIKE UNI', 
    'E BIKE W P', 'E COM', 'E MOPED', 'E MOTORCYC', 'E REVEL SC', 'E SCO', 
    'E SCOOTER', 'E- BI', 'E- MOTOR B', 'EBIKE', 'ESCOO', 'ESCOOTER', 
    'ESCOOTER S', 'ESCOOTER W', 'ESCOOTERSI'
]

# catorgorizing ()
vehicle_group = {
    'SPORT UTILITY / STATION WAGON / SPORT UTILITY VEHICLE': 'SUV',
    'STATION WAGON/SPORT UTILITY VEHICLE': 'SUV',
    'PASSENGER VEHICLE': 'SEDAN',
    '4 DR SEDAN': 'SEDAN',
    '2 DR SEDAN': 'SEDAN',
    'PICK-UP TRUCK': 'PICKUP',
    'UNKNOWN': 'UNSPECIFIED'
}
'''
Since Data was entered manually, there was alot of discrepency between speliings for similiar items
'''



for x in bike_variants: vehicle_group[x] = 'BIKE'
for x in moto_variants: vehicle_group[x] = 'MOTORCYCLE'
for x in ebike_variants: vehicle_group[x] = 'E-BIKE'

#Clean strings and make sure they are following the same syntax
for col in ['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2']:
    df_clean[col] = df_clean[col].str.upper().str.strip()
    df_clean[col] = df_clean[col].replace(vehicle_group)

#Final aggregation check
#all_vehicles_involved = pd.concat([df_clean['VEHICLE TYPE CODE 1'], df_clean['VEHICLE TYPE CODE 2']])
#print("--- Top 20 Vehicles (Combined 1 & 2) ---")
#print(all_vehicles_involved.value_counts().head(20))
#----------------- VRU SUBSET & TIME EXTRACTION -----------------

#  Extract the hour (0-23) from CRASH TIME string
df_clean['CRASH TIME'] = pd.to_datetime(df_clean['CRASH TIME'], format='%H:%M', errors='coerce').dt.hour

# Define vulnerable road users (VRU)
vru_list = ['BIKE', 'MOTORCYCLE', 'E-BIKE']

#Create the VRU dataframe (Linear filter)
is_vru = df_clean['VEHICLE TYPE CODE 1'].isin(vru_list)
df_vru = df_clean[is_vru].copy(deep = True)

#Drop any rows where the time didn't convert properly
df_vru = df_vru.dropna(subset=['CRASH TIME'])

#--------------Density Plot (distribution-oriented visualization)-------------------------
#Make sure of numeric type
df_vru['CRASH TIME'] = pd.to_numeric(df_vru['CRASH TIME'], errors='coerce')

#Plotting
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

'''
Density Plot can reveal to use when most crashes for 2 wheeled 
'''


# We use common_norm=False so that smaller groups (like E-Bikes) 
# aren't flattened by the massive number of regular Bike crashes.
sns.kdeplot(data=df_vru, x='CRASH TIME', hue='VEHICLE TYPE CODE 1', 
            fill=True, common_norm=False, palette='inferno', alpha=0.1)

plt.title('Crash Probability by Hour: Vulnerable Road Users', fontsize=15)
plt.xlabel('Hour of Day (0-23)', fontsize=12)
plt.ylabel('Density of Crashes', fontsize=12)
plt.xticks(range(0, 24))
plt.xlim(0, 23)
plt.tight_layout()

plt.show()

#--------------Heat Map (multivariate visualization)-------------------------
# Group by Borough and Hour, then count the crashes
#unstack rotates the graph around and make it easier to undertand
heatmap_data = df_vru.groupby(['BOROUGH', 'CRASH TIME']).size().unstack(fill_value=0)

# Filter out 'NA' to keep the map focused on known locations
if 'NA' in heatmap_data.index:
    heatmap_data = heatmap_data.drop('NA')


plt.figure(figsize=(16, 8))
sns.set_theme(style="white")


sns.heatmap(heatmap_data, cmap='inferno', cbar_kws={'label': 'Number of Crashes'})

# 5. Final Polish for the Rubric
plt.title('Frequency of VRU Crashes: Borough vs. Hour of Day', fontsize=16)
plt.xlabel('Hour of Day (0-23)', fontsize=12)
plt.ylabel('Borough', fontsize=12)

plt.tight_layout()
plt.show()

si_total = df_vru[df_vru['BOROUGH'] == 'STATEN ISLAND'].shape[0]
print(f"Total VRU crashes in Staten Island: {si_total}")
bk_hourly_peak = df_vru[df_vru['BOROUGH'] == 'BROOKLYN']['CRASH TIME'].value_counts().max()
print(f"Brooklyn Peak Hourly Crashes: {bk_hourly_peak}")
q_hourly_peak = df_vru[df_vru['BOROUGH'] == 'QUEENS']['CRASH TIME'].value_counts().max()
print(f"Queens Peak Hourly Crashes: {q_hourly_peak}")
br_hourly_peak = df_vru[df_vru['BOROUGH'] == 'BRONX']['CRASH TIME'].value_counts().max()
print(f"Bronx Peak Hourly Crashes: {br_hourly_peak}")
m_hourly_peak = df_vru[df_vru['BOROUGH'] == 'MANHATTAN']['CRASH TIME'].value_counts().max()
print(f"Manhattan Peak Hourly Crashes: {m_hourly_peak}")
si_hourly_peak = df_vru[df_vru['BOROUGH'] == 'STATEN ISLAND']['CRASH TIME'].value_counts().max()
print(f"Statebn Island Peak Hourly Crashes: {si_hourly_peak}")

#--------------Bar plot(Aggregated Data)------------
# 1. Group by BOTH Borough and Vehicle Type
# We calculate the mean number of injuries
severity_grouped = df_vru.groupby(['BOROUGH', 'VEHICLE TYPE CODE 1'])['NUMBER OF PERSONS INJURED'].mean().reset_index()

# 2. Filter out 'NA' if needed
severity_grouped = severity_grouped[severity_grouped['BOROUGH'] != 'NA']

# 3. Plotting
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# 'hue' adds the second layer of grouping
sns.barplot(data=severity_grouped, 
            x='BOROUGH', 
            y='NUMBER OF PERSONS INJURED', 
            hue='VEHICLE TYPE CODE 1',
            palette='colorblind')

plt.title('Severity Analysis: Mean Injuries by Borough & Vehicle Type', fontsize=16)
plt.ylabel('Average Number of Injuries')
plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

#--------------Horizontal Bar Chart (Contributing Factors)-------------------------
# Combine both contributing factor columns into one series
all_factors = pd.concat([
    df_vru['CONTRIBUTING FACTOR VEHICLE 1'],
    df_vru['CONTRIBUTING FACTOR VEHICLE 2']
])

# Remove unspecified/NA noise
all_factors = all_factors[~all_factors.isin(['Unspecified', 'NA'])]

# Get top 10 and convert to a DataFrame for seaborn
top_factors = all_factors.value_counts().head(10).reset_index()
top_factors.columns = ['CONTRIBUTING FACTOR', 'COUNT']

# Sort so longest bar is on top
top_factors = top_factors.sort_values('COUNT', ascending=True)

plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

'''
Horizontal bar chart shows us the most common reasons VRU crashes happen.
Combining both vehicle factor columns gives a complete picture rather
than only looking at the primary cause.
'''

sns.barplot(data=top_factors, x='COUNT', y='CONTRIBUTING FACTOR', palette='inferno')

plt.title('Top 10 Contributing Factors in VRU Crashes', fontsize=16)
plt.xlabel('Number of Crashes', fontsize=12)
plt.ylabel('Contributing Factor', fontsize=12)

plt.tight_layout()
plt.show()

#-------Geo Spatial/Plotly-------
# Filter out rows that dont have coordinates
df_geo = df_vru.dropna(subset=['LATITUDE', 'LONGITUDE']).copy(deep=True)

# Keep only coordinates that are actually in NYC
df_geo = df_geo[df_geo['LATITUDE'] > 40]
df_geo = df_geo[df_geo['LATITUDE'] < 41]
df_geo = df_geo[df_geo['LONGITUDE'] > -74.5]
df_geo = df_geo[df_geo['LONGITUDE'] < -73.5]

# Convert the dataframe into a GeoDataFrame so we can plot it on a map
gdf = gpd.GeoDataFrame(df_geo, geometry=gpd.points_from_xy(df_geo['LONGITUDE'], df_geo['LATITUDE']), crs='EPSG:4326')

'''
gpd.points_from_xy() pairs up the longitude and latitude columns into geometry points
crs EPSG:4326 is just the standard GPS coordinate system
'''

# Give each vehicle type its own color
color_map = {'BIKE': 'orange', 'MOTORCYCLE': 'purple', 'E-BIKE': 'cyan'}

# Build a separate trace (layer) for each vehicle type
traces = []
for vtype, group in gdf.groupby('VEHICLE TYPE CODE 1'):
    trace = pl.graph_objs.Scattermapbox(
        lat=group.geometry.y,
        lon=group.geometry.x,
        mode='markers',
        marker=pl.graph_objs.scattermapbox.Marker(size=5, color=color_map.get(vtype, 'gray'), opacity=0.5),
        name=vtype,
        text='Borough: ' + group['BOROUGH'] + ' | Hour: ' + group['CRASH TIME'].astype(str) + ' | Injured: ' + group['NUMBER OF PERSONS INJURED'].astype(str),
        hoverinfo='text'
    )
    traces.append(trace)

'''
We loop through each vehicle type and make a seperate layer for it
That way each one gets its own color and can be toggled on/off in the legend
'''

# Set up the map layout
layout = pl.graph_objs.Layout(
    title='NYC VRU Crash Locations by Vehicle Type',
    mapbox=pl.graph_objs.layout.Mapbox(style='open-street-map', center=pl.graph_objs.layout.mapbox.Center(lat=40.730, lon=-73.935), zoom=10),
    height=700
)

fig = pl.graph_objs.Figure(data=traces, layout=layout)
pl.offline.plot(fig, filename='vru_crashes_map.html', auto_open=True)
