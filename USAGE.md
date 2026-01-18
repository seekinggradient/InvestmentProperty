# Tech Hub Investment Pipeline - Usage Guide

## Quick Start

### 1. Generate Rankings (First Time)
```bash
# Fetch all data fresh and cache it
python tech_hub_pipeline.py --no-cache
```

This will:
- Fetch data from Census, BLS, Zillow, and FRED APIs
- Cache raw data in `cache/` directory
- Generate `emerging_tech_hubs.csv` with 101 ranked markets

**Runtime**: ~2-3 minutes (with all API calls)

---

### 2. Analyze Results
```bash
# Run all analyses
python analyze_results.py

# Show top 20 markets
python analyze_results.py --top 20

# Filter by state
python analyze_results.py --state TX

# Show only cash flow analysis
python analyze_results.py --analysis cashflow
```

---

### 3. Re-run Pipeline (Using Cache)
```bash
# Uses cached data - MUCH faster!
python tech_hub_pipeline.py
```

**Runtime**: ~30 seconds (from cache)

---

## Command Reference

### tech_hub_pipeline.py
Main data pipeline that fetches, processes, and ranks markets.

```bash
# Use cached data (fast)
python tech_hub_pipeline.py

# Force fresh data fetch
python tech_hub_pipeline.py --no-cache

# Custom cache directory
python tech_hub_pipeline.py --cache-dir=my_cache

# Custom output file
python tech_hub_pipeline.py --output=results_2025.csv

# Help
python tech_hub_pipeline.py --help
```

**What gets cached:**
- Census ACS 2018 data → `cache/census_2018.pkl`
- Census ACS 2023 data → `cache/census_2023.pkl`
- Zillow ZHVI/ZORI data → `cache/zillow_data.pkl`
- FRED unemployment data → `cache/fred_unemployment.pkl`

**NOT cached** (fetched fresh each time):
- BLS QCEW data (16 API calls for quarterly wage data)

---

### analyze_results.py
Standalone analysis script - run anytime to generate insights.

```bash
# All analyses
python analyze_results.py

# Specific analysis type
python analyze_results.py --analysis top        # Top markets only
python analyze_results.py --analysis states     # By state
python analyze_results.py --analysis cashflow   # Best cash flow
python analyze_results.py --analysis growth     # Fastest growth
python analyze_results.py --analysis affordable # Most affordable

# Filter by state
python analyze_results.py --state CA           # California only
python analyze_results.py --state TX --top 15  # Top 15 in Texas

# Custom input file
python analyze_results.py --file=old_results.csv
```

**Available Analyses:**
1. **Top Markets**: Overall best investment opportunities
2. **By State**: Best markets per state
3. **Cash Flow**: Lowest price-to-rent ratios (best rental returns)
4. **Growth**: Highest tech wage growth (appreciation potential)
5. **Affordable**: Lowest median home values (entry points)

---

## Typical Workflow

### Weekly Update (Recommended)
```bash
# 1. Update with fresh data
python tech_hub_pipeline.py --no-cache

# 2. Analyze your target states
python analyze_results.py --state TX
python analyze_results.py --state PA
python analyze_results.py --state FL

# 3. Review cash flow opportunities
python analyze_results.py --analysis cashflow --top 20
```

### Daily Analysis (No Fresh Fetch)
```bash
# Just analyze existing data - instant results
python analyze_results.py
python analyze_results.py --state CA
python analyze_results.py --analysis growth
```

### Custom Python Analysis
```python
import pandas as pd

# Load data (skip 2 header rows)
df = pd.read_csv('emerging_tech_hubs.csv', skiprows=[1, 2])

# Your custom filtering
texas_affordable = df[(df['State'] == 'TX') &
                      (df['Median Home Value'] < 300000) &
                      (df['Price-to-Rent Ratio'] < 16)]

# Analyze
print(texas_affordable[['Rank', 'Metropolitan Area', 'Boom Score']])
```

---

## CSV Structure

The output CSV has 3 header rows:
- **Row 1**: Column names
- **Row 2**: Technical descriptions (data sources, formulas)
- **Row 3**: Interpretation guidance (how to use for investing)
- **Row 4+**: Market data (101 markets)

When reading in Python:
```python
df = pd.read_csv('emerging_tech_hubs.csv', skiprows=[1, 2])
```

When opening in Excel:
- Row 2: Hover for technical details
- Row 3: Read for investment guidance
- Filter by State column to focus on specific states

---

## Data Freshness

| Data Source | Frequency | Cache | Notes |
|-------------|-----------|-------|-------|
| Census ACS | Annual (lags 1 year) | ✅ Yes | 2023 is latest available |
| BLS QCEW | Quarterly (lags 6mo) | ❌ No | Updates Q3 2023 - Q2 2025 |
| Zillow | Monthly | ✅ Yes | Home values & rents |
| FRED | Monthly | ✅ Yes | State unemployment rates |

**Recommendation**:
- Re-run with `--no-cache` weekly to get latest Zillow/FRED data
- BLS QCEW updates quarterly, so monthly updates won't show new wage data

---

## Troubleshooting

### "File not found" error in analyze_results.py
```bash
# Make sure you ran the pipeline first
python tech_hub_pipeline.py
```

### Cache is stale (want fresh data)
```bash
# Option 1: Force fresh fetch
python tech_hub_pipeline.py --no-cache

# Option 2: Delete cache manually
rm -rf cache/
python tech_hub_pipeline.py
```

### API rate limiting
```bash
# Census API rate limited? Wait 5 minutes or get API key
# Sign up: https://api.census.gov/data/key_signup.html
# Then edit tech_hub_pipeline.py: CENSUS_API_KEY = "your_key"
```

### Want to analyze old results
```bash
# Save output with date
python tech_hub_pipeline.py --output=results_2025_01_15.csv

# Analyze old file
python analyze_results.py --file=results_2025_01_15.csv
```

---

## Performance

| Operation | Runtime | Notes |
|-----------|---------|-------|
| First run (--no-cache) | ~2-3 min | All API calls + caching |
| Cached run | ~30 sec | Uses cached data |
| analyze_results.py | <1 sec | Just reads CSV |

**Bottleneck**: BLS QCEW (16 API calls, ~1 sec each)

---

## Next Steps

1. **Run weekly**: `python tech_hub_pipeline.py --no-cache`
2. **Daily analysis**: `python analyze_results.py --state <YOUR_STATE>`
3. **Custom filters**: Edit `analyze_results.py` or write your own Python scripts
4. **Export for Excel**: The CSV opens directly in Excel with formatted headers

Happy investing! 🏡📈
