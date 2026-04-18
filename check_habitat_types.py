
import pandas as pd
from schema import schema

try:
    df = pd.read_csv('unified_bc_blue_carbon.csv')

    # Get allowed habitat_type values from schema.py
    allowed_habitat_types = schema['habitat_type']['values']
    print(f"Allowed habitat types: {allowed_habitat_types}")

    # Identify entries with invalid habitat_type
    invalid_habitat_types = df[~df['habitat_type'].isin(allowed_habitat_types)]['habitat_type'].unique()

    if len(invalid_habitat_types) > 0:
        print(f"Invalid habitat types found: {list(invalid_habitat_types)}")
        # Print some rows with invalid habitat_type to understand context
        print("""
Sample rows with invalid habitat types:""")
        print(df[~df['habitat_type'].isin(allowed_habitat_types)][['site_id', 'data_source', 'habitat_type', 'notes']].head())
    else:
        print("No invalid habitat types found based on schema.")

except FileNotFoundError:
    print("Error: unified_bc_blue_carbon.csv not found.")
except KeyError as e:
    print(f"Error: Column or schema key not found: {e}.")
