
import pandas as pd

try:
    df = pd.read_csv('unified_bc_blue_carbon.csv')
    unique_data_sources = df['data_source'].unique()
    for source in unique_data_sources:
        print(source)
except FileNotFoundError:
    print("Error: unified_bc_blue_carbon.csv not found.")
except KeyError:
    print("Error: 'data_source' column not found in unified_bc_blue_carbon.csv.")
