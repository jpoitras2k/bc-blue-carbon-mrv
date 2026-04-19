# The schema defines the strict data types, required fields, and acceptable values 
# for our blue carbon dataset. This ensures consistency when merging data from 
# different sources (like CRD, Janousek, Hakai) into a single analytical pipeline.
schema = {
    'site_id': {'type': str, 'required': True, 'format': r'[A-Z]{2}-[A-Z]{3}-\d{3}'},
    'site_name': {'type': str, 'required': True},
    'latitude': {'type': float, 'required': True, 'range': (48.0, 60.0)},
    'longitude': {'type': float, 'required': True, 'range': (-140.0, -114.0)},
    'region': {'type': str, 'required': True, 'values': ['Salish Sea', 'West Coast VI', 'Central Coast', 'North Coast', 'Haida Gwaii', 'NE Pacific']},
    'habitat_type': {'type': str, 'required': True, 'values': ['eelgrass', 'eelgrass_dwarf', 'kelp', 'saltmarsh', 'mixed_seagrass']},
    'sediment_type': {'type': str, 'required': False, 'values': ['mud', 'sand', 'mud_sand', 'gravel', 'rocky', 'organic', 'unknown']},
    'carbon_density_gCm2': {'type': float, 'required': False},
    'sequestration_rate_gCm2yr': {'type': float, 'required': False},
    'measurement_depth_cm': {'type': int, 'required': False},
    'survey_year': {'type': int, 'required': True},
    'data_source': {'type': str, 'required': True},
    'access_type': {'type': str, 'required': True, 'values': ['open', 'restricted', 'grey_literature', 'academic', 'estimated']},
    'notes': {'type': str, 'required': False}
}

# Data Gap Flags for notes column
# We use these flags in the 'notes' column to explicitly mark rows that need 
# special handling down the line, such as 'CARBON_GAP:' triggering our ML model 
# to impute missing carbon density values.
DATA_GAP_FLAGS = {
    'is_carbon_gap': 'CARBON_GAP:',
    'is_coord_approx': 'COORD_APPROX:',
    'is_year_est': 'YEAR_EST:',
    'is_dupe_check': 'DUPE_CHECK:'
}