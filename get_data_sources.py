
import pandas as pd

try:
    df = pd.read_csv('unified_bc_blue_carbon.csv')
    # We do a quick extraction of all unique data sources to see where our inputs are coming from.
    # This helps verify if 'Janousek' or 'CRD' data successfully merged into the final dataset.
    unique_data_sources = df['data_source'].unique()
    for source in unique_data_sources:
        print(source)
except FileNotFoundError:
    print("Error: unified_bc_blue_carbon.csv not found.")
except KeyError:
    print("Error: 'data_source' column not found in unified_bc_blue_carbon.csv.")
