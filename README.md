# U.S. Emerging Tech Hub Ranking Engine

A Python-based data pipeline that identifies emerging U.S. real estate markets by analyzing demographic growth, tech sector wage trends, and housing affordability.

## Overview

This pipeline ranks U.S. Metropolitan Statistical Areas (MSAs) based on "Leading Indicators" of wealth influx, focusing on:
- **High-skill demographic growth** (Indian population, advanced degree holders)
- **Accelerating tech sector wages** (NAICS 5415 & 5182)
- **Housing affordability** relative to San Jose, CA

## Data Sources

### Census Bureau (ACS 5-Year Estimates)
- **API**: `https://api.census.gov/data/{year}/acs/acs5`
- **Variables**:
  - `B02015_001E`: Total Asian Population
  - `B02015_021E`: Asian Indian Population (corrected from B02015_009E)
  - `B15003_023E`: Master's Degree holders
  - `B15003_025E`: Doctorate Degree holders
  - `B19013_001E`: Median Household Income
  - `B25077_001E`: Median Home Value

### Bureau of Labor Statistics (QCEW)
- **API**: `https://data.bls.gov/cew/data/api/{year}/{quarter}/industry/{naics}.csv`
- **Industries**:
  - `5415`: Computer Systems Design and Related Services
  - `5182`: Data Processing, Hosting, and Related Services
- **Metrics**: Quarterly wages, establishments, employment

## Scoring Algorithm

```
Boom Score = (0.4 × Tech Wage Score) + (0.3 × Indian Pop Score) + (0.3 × Affordability Score)
```

Where:
- **Tech Wage Score**: Normalized CAGR of tech sector quarterly wages (2022-2023)
- **Indian Pop Score**: Normalized CAGR of Indian population (2018-2023)
- **Affordability Score**: Inverse relationship to median home prices (lower prices = higher score)

## Setup

### Prerequisites
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies
- `requests` >= 2.31.0
- `pandas` >= 2.0.0
- `numpy` >= 1.24.0

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the pipeline
python3 tech_hub_pipeline.py
```

The pipeline will:
1. Fetch Census ACS data for 2018 and 2023 (~945 MSAs)
2. Fetch BLS QCEW data for 8 quarters (Q1 2022 - Q4 2023)
3. Calculate 5-year CAGRs for demographics
4. Calculate tech wage growth rates
5. Merge datasets using CBSA codes
6. Calculate composite Boom Probability Scores
7. Export ranked results to `emerging_tech_hubs.csv`

## Output

The pipeline generates `emerging_tech_hubs.csv` with the following columns:

| Column | Description |
|--------|-------------|
| Rank | Overall ranking (1 = highest boom score) |
| Metropolitan Area | MSA name |
| Boom Score | Composite score (0-100) |
| Median Home Value | 2023 median home price |
| Indian Population (2023) | Count of residents of Indian descent |
| Indian Pop Growth (CAGR %) | 5-year compound annual growth rate |
| Advanced Degrees (2023) | Count of Master's + Doctorate holders |
| Advanced Degree Growth (CAGR %) | 5-year CAGR |
| Tech Wage Growth (CAGR %) | Tech sector wage CAGR (2022-2023) |
| Latest Quarterly Tech Wages ($) | Q4 2023 total wages |
| Tech Employment | Tech sector employees (month 3) |
| Tech Establishments | Number of tech companies |
| Median Household Income | 2023 median income |
| Tech Wage Score (0-100) | Normalized component score |
| Indian Pop Score (0-100) | Normalized component score |
| Affordability Score (0-100) | Normalized component score |

## Sample Results

Top emerging markets combine:
- **Strong tech wage growth** (>20% CAGR)
- **Affordable housing** (median < $400k vs. San Jose's $1.34M)
- **Growing Indian population** (high-skill immigration proxy)

Example insights:
- **Indianapolis, IN**: 283% tech wage growth, 26k Indian residents (+84%), $244k homes
- **Charlotte, NC**: 153% tech wage growth, 49k Indian residents (+110%), $319k homes
- **Greensboro, NC**: 150% tech wage growth, 6.6k Indian residents (+84%), $208k homes
- **Albuquerque, NM**: 169% tech wage growth, 2.9k Indian residents (+48%), $264k homes

## Interpreting Results

### High Boom Scores indicate:
- Markets with explosive tech sector growth
- Affordable entry points compared to established hubs (SF, Seattle, Boston)
- Potential for real estate appreciation as tech talent relocates

### Limitations:
1. **Small Sample Bias**: MSAs with very small baseline populations may show extreme growth rates (e.g., 100%+ from doubling a small number)
2. **Negative Growth**: Some metros show -100% or negative growth due to industry reclassification, establishment closures, or data gaps
3. **Short Time Horizon**: Tech wage growth uses only 2 years of data (2022-2023) due to QCEW availability
4. **MSA Definition Changes**: Census MSA boundaries may change between 2018-2023, affecting growth calculations

## Technical Notes

### CBSA Code Mapping
The pipeline automatically maps between:
- **Census CBSA codes**: 5-digit codes (e.g., `10180` for Abilene, TX)
- **BLS area_fips**: Format `CXXXX` (e.g., `C1018` for Abilene, TX)

Conversion: Remove 'C' prefix, left-strip zeros, append '0'
```python
'C1018' → '1018' → '10180'
```

### API Limits
- Census API works without authentication but has rate limits
- To avoid rate limiting, sign up for a key at: https://api.census.gov/data/key_signup.html

## Customization

### Modify Scoring Weights
Edit `calculate_boom_score()` in `tech_hub_pipeline.py`:
```python
df['boom_score'] = (
    0.4 * df['tech_wage_score'] +      # Adjust tech weight
    0.3 * df['indian_pop_score'] +     # Adjust demographic weight
    0.3 * df['affordability_score']    # Adjust affordability weight
)
```

### Change Time Periods
Edit `fetch_all_qcew_data()` and `get_census_data()` to use different years:
```python
quarters_to_fetch = [
    (2021, '1'), (2021, '2'), ...  # Modify quarters
]

census_2018 = self.get_census_data(2017)  # Use different baseline year
```

### Add Additional Metrics
Extend `get_census_data()` with more ACS variables:
```python
variables = [
    "B02015_009E",  # Indian
    "B03002_006E",  # Asian
    "B25064_001E",  # Median Gross Rent
    # Add more variables from Census API
]
```

## Troubleshooting

### "Invalid Key" Error
- Census API key may not be activated
- Pipeline works without key but may hit rate limits
- Set `CENSUS_API_KEY = None` in `main()` to disable key usage

### "No MSAs matched" Error
- Check that CBSA codes are mapping correctly
- Verify Census and BLS data are being fetched successfully
- Enable debug logging to inspect FIPS code matching

### Missing Data
- Small MSAs may have suppressed demographic data
- Tech wage data only available for metros with substantial tech presence
- Use `how='inner'` merge to keep only MSAs with complete data

## License

This project is for educational and research purposes. Data sources are public APIs provided by U.S. Census Bureau and Bureau of Labor Statistics.

## References

- [Census ACS API Documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- [BLS QCEW Data Files](https://www.bls.gov/cew/downloadable-data-files.htm)
- [NAICS Code Definitions](https://www.census.gov/naics/)
- [CBSA Definitions](https://www.census.gov/programs-surveys/metro-micro/about.html)
