# Investment Property Analysis - Project Summary

## Goal

Find U.S. real estate markets likely to experience property value appreciation ("booms") for investment purposes. The user wants to identify areas that are:
1. **Likely to appreciate** in value (capital gains)
2. **Good for cash flow** (rental income via low price-to-rent ratios)

---

## Current Files

| File | Purpose |
|------|---------|
| `tech_hub_pipeline.py` | Main data pipeline - fetches data, calculates scores, outputs CSV |
| `analyze_results.py` | Standalone script to analyze/filter the output CSV |
| `backtest_algorithm.py` | Tests if our algorithm would have predicted past winners |
| `emerging_tech_hubs.csv` | Output: 101 ranked markets with all metrics |
| `backtest_results_2018_2023.csv` | Backtest results comparing predictions vs actual |
| `cache/` | Cached API responses (pickle files) |

---

## Data Sources Currently Used

### 1. Census ACS 5-Year Estimates (via Census API)
- **Years fetched**: 2018, 2023
- **Variables**:
  - `B02015_021E`: Asian Indian Population
  - `B15003_023E` + `B15003_025E`: Masters + Doctorate holders
  - `B19013_001E`: Median Household Income
  - `B25077_001E`: Median Home Value
- **Coverage**: ~930 Metropolitan Statistical Areas (MSAs)
- **Cached**: `cache/census_2018.pkl`, `cache/census_2023.pkl`

### 2. BLS QCEW (Quarterly Census of Employment and Wages)
- **Purpose**: Tech sector wage growth
- **NAICS codes**: 5415 (Computer Systems Design), 5182 (Data Processing)
- **Time period**: Q3 2023 → Q2 2025 (wage CAGR)
- **NOT cached**: Fetched fresh each run

### 3. Zillow Research Data (public CSVs)
- **ZHVI**: Zillow Home Value Index (current home values)
- **ZORI**: Zillow Observed Rent Index (monthly rents)
- **Derived**: Price-to-Rent Ratio = ZHVI / (ZORI × 12)
- **Coverage**: ~900 metros
- **Cached**: `cache/zillow_data.pkl`

### 4. FRED (Federal Reserve Economic Data)
- **Data**: State-level unemployment rates
- **Coverage**: All 50 states + DC
- **Cached**: `cache/fred_unemployment.pkl`

---

## Current Algorithm: "Boom Score"

```
Boom Score = 0.30 × Tech Wage Score
           + 0.25 × Indian Pop Score (hybrid)
           + 0.25 × Affordability Score
           + 0.20 × Rental Cash Flow Score
```

### Component Details:

| Component | Weight | What It Measures | Formula |
|-----------|--------|------------------|---------|
| Tech Wage Score | 30% | Tech sector growth | CAGR of tech wages (NAICS 5415+5182) |
| Indian Pop Score | 25% | High-skill migration | CAGR × log₁₀(Population) |
| Affordability Score | 25% | Entry point opportunity | Inverse of (price / San Jose price) |
| Rental Cash Flow Score | 20% | Immediate returns | Inverse of Price-to-Rent ratio |

### Filters Applied:
- **Minimum 5,000 Indian population** (to avoid small-sample bias)
- Results: ~101 markets pass the filter

---

## Key Assumptions (UNVALIDATED)

The algorithm assumes:

1. **Indian population growth predicts appreciation** - Areas with growing Indian populations will see rising home values (proxy for high-income tech worker migration)

2. **Tech wage growth predicts appreciation** - Areas with growing tech sectors will see rising home values

3. **These patterns from 2018-2023 will continue** - Historical trends are predictive of future trends

---

## Backtesting Results (2018-2023)

We tested whether our algorithm's features would have predicted actual appreciation:

### Correlation with Actual Appreciation:
| Feature | Correlation | Significance |
|---------|-------------|--------------|
| Home Value (2018) | 0.053 | Weak* |
| Indian Pop Absolute | -0.016 | None |
| Indian Pop Growth (CAGR) | -0.009 | None |
| Advanced Degree Growth | 0.007 | None |
| **Indian Pop Hybrid Score** | **0.005** | **None** |

### Key Finding:
**Our core features showed near-zero correlation with actual 2018-2023 appreciation.**

The actual top appreciating markets (2018-2023) were small resort/retirement towns (Thomaston, Clewiston, Mountain Home) - NOT tech hubs with Indian population growth.

### Possible Explanations:
- COVID-19 fundamentally changed migration patterns (remote work boom)
- 2018-2023 was an unusual period (may not repeat)
- Our proxy variables may not actually predict appreciation

---

## Output CSV Structure

The `emerging_tech_hubs.csv` has 3 header rows:
1. **Row 1**: Column names
2. **Row 2**: Technical descriptions
3. **Row 3**: Interpretation guidance (how to use each metric)

Key columns:
- Rank, Metropolitan Area, State, Boom Score
- Median Home Value, Indian Population (2018, 2023), Growth rates
- Price-to-Rent Ratio, Tech Wage Growth, Unemployment Rate
- Component scores (0-100 normalized)

---

## Current Top Markets (as ranked)

| Rank | Market | Boom Score | Home Value | P/R Ratio |
|------|--------|------------|------------|-----------|
| 1 | Pittsburgh, PA | 77.9 | $204k | 12.5 |
| 2 | Trenton-Princeton, NJ | 77.0 | $351k | 14.3 |
| 3 | Detroit, MI | 73.9 | $237k | 14.4 |
| 4 | Bloomington, IL | 72.2 | $198k | 12.8 |
| 5 | Albany-Schenectady, NY | 70.2 | $268k | 13.7 |

---

## What's Missing / Opportunities

### Data sources NOT yet integrated:
- **IRS Migration Data** - Net migration flows, income levels of movers
- **Job posting data** - Leading indicator of employment growth
- **Building permits** - New construction pipeline
- **Crime/school data** - Quality of life metrics
- **Remote work trends** - Post-COVID migration patterns

### Algorithm improvements needed:
- **Validate assumptions** with more backtesting periods
- **Find features that actually correlate** with appreciation
- **Multi-objective optimization** - Separate scores for cash flow vs appreciation
- **Machine learning** - Learn weights from historical data

---

## How to Run

```bash
# Activate environment
source venv/bin/activate

# Run main pipeline (uses cache)
python tech_hub_pipeline.py

# Force fresh data fetch
python tech_hub_pipeline.py --no-cache

# Analyze results
python analyze_results.py --state CA
python analyze_results.py --analysis cashflow --top 20

# Run backtest
python backtest_algorithm.py
```

---

## API Keys Required

| Service | Key Location | Purpose |
|---------|--------------|---------|
| Census API | Hardcoded in `tech_hub_pipeline.py` | Demographic data |
| FRED API | Hardcoded in `tech_hub_pipeline.py` | Unemployment data |
| Zillow | None needed | Public CSVs |
| BLS QCEW | None needed | Public API |

---

## Bottom Line

**Current state**: We have a working pipeline that ranks 101 U.S. metros using a composite score based on Indian population growth, tech wages, affordability, and rental yields.

**Problem**: Backtesting shows our core assumptions (Indian pop growth predicts appreciation) had near-zero predictive power for 2018-2023.

**Next step options**:
1. Find different features that actually correlate with appreciation
2. Accept the model limitations and use it as one input among many
3. Build a new model from scratch with validated predictors
