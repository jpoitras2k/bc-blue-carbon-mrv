# BC Blue Carbon Unified CSV Schema

**File:** `unified_bc_blue_carbon.csv`  
**Purpose:** Merges disparate eelgrass meadow carbon sequestration datasets across British Columbia and Vancouver Island into a single normalized format for ML analysis and the Kaggle Uncharted Data Challenge.

---

## Column Definitions

| Column | Type | Units | Required | Description |
|--------|------|-------|----------|-------------|
| `site_id` | string | — | Yes | Unique identifier for the survey site (e.g., `BC-EEL-001`). Format: `{province}-{habitat_code}-{sequence}` |
| `site_name` | string | — | Yes | Human-readable site name (e.g., "Saanich Inlet North", "Tofino Mudflats") |
| `latitude` | float | decimal degrees (WGS84) | Yes | Latitude of site centroid. Range: 48.0–60.0 |
| `longitude` | float | decimal degrees (WGS84) | Yes | Longitude of site centroid. Range: -140.0–-114.0 |
| `region` | string | — | Yes | Regional classification. Values: `Salish Sea`, `West Coast VI`, `Central Coast`, `North Coast`, `Haida Gwaii`, `NE Pacific` |
| `habitat_type` | string | — | Yes | Primary habitat. Values: `eelgrass` (Zostera marina), `eelgrass_dwarf` (Zostera japonica), `kelp`, `saltmarsh`, `mixed_seagrass` |
| `sediment_type` | string | — | No | Substrate classification. Values: `mud`, `sand`, `mud_sand`, `gravel`, `rocky`, `organic`, `unknown` |
| `carbon_density_gCm2` | float | g C m⁻² | No | Organic carbon stock in surface sediment (typically 0–30 cm). NULL if not measured. |
| `sequestration_rate_gCm2yr` | float | g C m⁻² yr⁻¹ | No | Carbon sequestration rate. NULL if not measured or estimated. |
| `measurement_depth_cm` | integer | cm | No | Depth of sediment core used for carbon measurement. NULL if not applicable. |
| `survey_year` | integer | — | Yes | Year of primary data collection (e.g., 2019). Use earliest year for multi-year datasets. |
| `data_source` | string | — | Yes | Source organization or publication (e.g., `DFO`, `BC Data Catalogue`, `UVic`, `SeaChange MCS`) |
| `access_type` | string | — | Yes | Data availability. Values: `open`, `restricted`, `grey_literature`, `academic`, `estimated` |
| `notes` | string | — | No | Free-text field for caveats, data quality flags, coordinate precision notes, or gap indicators |

---

## Example Rows

```csv
site_id,site_name,latitude,longitude,region,habitat_type,sediment_type,carbon_density_gCm2,sequestration_rate_gCm2yr,measurement_depth_cm,survey_year,data_source,access_type,notes
BC-EEL-001,Saanich Inlet North,48.6200,-123.4800,Salish Sea,eelgrass,mud,142.5,83.2,25,2018,UVic Blue Carbon Lab,academic,Peer-reviewed; high confidence
BC-EEL-002,Tofino Mudflats,49.1530,-125.9070,West Coast VI,eelgrass,mud_sand,,56.0,,2020,BC Shore Spawners Alliance,grey_literature,Location data only; no sediment cores
BC-EEL-003,Cowichan Bay,48.7700,-123.6100,Salish Sea,eelgrass_dwarf,sand,89.3,,15,2019,DFO Coastal Survey,open,Dwarf eelgrass — lower carbon density expected
BC-EEL-004,Bamfield Inlet,48.8350,-125.1370,West Coast VI,eelgrass,mud,,,, 2021,NRCan Benthic Mapping,open,GIS polygon only; carbon data gap flagged
```

---

## Data Gap Flags

Use the `notes` field with standardized prefixes to track known gaps:
- `CARBON_GAP:` — site has location/habitat data but no carbon density measurement
- `COORD_APPROX:` — coordinates are approximate (>500m accuracy)
- `YEAR_EST:` — survey year is estimated from publication date
- `DUPE_CHECK:` — possible duplicate with another source; needs reconciliation

---

## Schema Version

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Created | 2026-04-12 |
| Author | Engineer (Paperclip KEE-24) |
| Project | BC Blue Carbon ML — Kaggle Uncharted Data Challenge |
| Parent task | [KEE-16](/KEE/issues/KEE-16) |
