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
    'OFF STREET NAME', 'CROSS STREET NAME'
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

#----------NOTE-----------
'''
Since Data was entered manually, there was alot of discrepency between speliings for similiar items
'''
#-------------------------


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
df_vru = df_clean[is_vru].copy()

#Drop any rows where the time didn't convert properly
df_vru = df_vru.dropna(subset=['CRASH TIME'])

#----------------------------------------------------------------
