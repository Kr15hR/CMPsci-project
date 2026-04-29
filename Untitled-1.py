import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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


#Final Audit
print("--- Original DATA TYPES ---")
print(df.dtypes)

print("--- FINAL DATA TYPES ---")
print(df_clean.dtypes)

print("\n--- Original MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- REMAINING MISSING VALUES ---")
print(df_clean.isnull().sum())

print(f"\nOriginal rows: {len(df)}")
print(f"Final cleaned rows: {len(df_clean)}")

print("\n---Percentage of Original Rows kept---")
print((len(df_clean)/len(df))*100)