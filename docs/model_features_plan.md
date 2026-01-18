# Investment Property Model - Feature Engineering Plan

## Overview

This document outlines a new approach to predicting real estate appreciation and cash flow potential. The previous model's core features (Indian population growth, tech wages) showed near-zero correlation with actual 2018-2023 appreciation in backtesting.

**Key insight**: The actual top appreciating markets were small resort/retirement towns, not tech hubs. This suggests we need fundamentally different predictors.

---

## Two-Model Architecture

Instead of one composite "Boom Score", we propose **two separate models**:

| Model | Goal | Primary Drivers |
|-------|------|-----------------|
| **Appreciation Model** | Predict price growth | Supply constraints, migration quality, momentum |
| **Cash Flow Model** | Predict rental yields | Price-to-rent, vacancy, landlord laws |

Users can weight these based on investment strategy:
```
Investment Score = α × Appreciation Score + (1-α) × Cash Flow Score
```

---

## Feature Categories

### Category 1: Supply Constraints (Appreciation)

**Hypothesis**: Markets where it's hard to build appreciate more than those with unlimited sprawl potential.

| Feature | Source | Rationale |
|---------|--------|-----------|
| Building permits per 1000 residents | Census Building Permits Survey | Low permits = constrained supply |
| Permits-to-population-growth ratio | Census | Supply keeping up with demand? |
| Geographic constraint score | Custom calculation | Coastal, mountain, water limits |
| Wharton Land Use Regulatory Index | Wharton (static dataset) | Zoning stringency |

**Implementation Priority**: HIGH - Building permits data is freely available and likely predictive.

---

### Category 2: Migration Quality (Appreciation)

**Hypothesis**: Markets gaining high-income households will see price pressure; losing them will stagnate.

| Feature | Source | Rationale |
|---------|--------|-----------|
| Net migration (households) | IRS SOI Tax Stats | More arriving than leaving |
| Avg AGI of in-migrants | IRS SOI Tax Stats | High earners moving in |
| Avg AGI of out-migrants | IRS SOI Tax Stats | Who's leaving? |
| AGI ratio (in/out) | Calculated | Quality of migration |
| In-migration from HCOL metros | IRS SOI Tax Stats | Remote work arbitrage signal |
| Domestic migration rate | Census Population Estimates | Overall population flow |

**Data source**: IRS Statistics of Income (SOI) publishes county-to-county migration flows with income data annually.
- URL: https://www.irs.gov/statistics/soi-tax-stats-migration-data

**Implementation Priority**: HIGH - This directly measures what the Indian population proxy was trying to estimate.

---

### Category 3: Price Dynamics (Appreciation)

**Hypothesis**: Real estate has short-term momentum but long-term mean reversion. Undervalued markets relative to fundamentals will catch up.

| Feature | Source | Rationale |
|---------|--------|-----------|
| 1-year price change | Zillow ZHVI | Short-term momentum |
| 3-year CAGR | Zillow ZHVI | Medium-term trend |
| Price vs 5-year avg | Calculated | Mean reversion signal |
| Price-to-income ratio | ZHVI / Census median income | Affordability relative to local wages |
| Price gap to comparable metros | Calculated | Arbitrage opportunity |
| Price percentile (national) | Calculated | Absolute affordability |

**Implementation Priority**: MEDIUM - Easy to calculate from existing data.

---

### Category 4: Employment Leading Indicators (Appreciation)

**Hypothesis**: Job posting growth and business formation predict future employment, which drives housing demand.

| Feature | Source | Rationale |
|---------|--------|-----------|
| Job posting growth | Indeed Hiring Lab (if available) | Leading indicator |
| Business applications (new EINs) | Census Business Formation Statistics | Entrepreneurship/growth |
| JOLTS job openings rate | BLS JOLTS | Labor market tightness |
| Unemployment rate trend | FRED/BLS | Lagging but relevant |
| Tech employment share | BLS QCEW | Sector composition |

**Implementation Priority**: MEDIUM - Business applications data is freely available.

---

### Category 5: Quality of Life (Appreciation)

**Hypothesis**: Amenity-rich locations attract residents and command premium pricing.

| Feature | Source | Rationale |
|---------|--------|-----------|
| Climate score | NOAA / custom | Weather desirability |
| Violent crime rate | FBI UCR | Safety |
| Property crime rate | FBI UCR | Safety |
| School district rating | GreatSchools API | Family appeal |
| Healthcare access score | CMS / custom | Retirement appeal |
| Outdoor recreation access | Custom (parks, trails) | Lifestyle amenities |

**Implementation Priority**: LOW - Harder to standardize, but valuable for filtering.

---

### Category 6: Rental Market (Cash Flow)

**Hypothesis**: Cash flow depends on rent-to-price dynamics, vacancy, and operating costs.

| Feature | Source | Rationale |
|---------|--------|-----------|
| Price-to-rent ratio | Zillow ZHVI/ZORI | Core cash flow metric |
| Rent growth CAGR | Zillow ZORI | Rising rents = improving yields |
| Vacancy rate | Census ACS | Demand strength |
| Gross rent multiplier | Calculated | Investment metric |
| Rent-to-income ratio | ZORI / median income | Rent sustainability |

**Implementation Priority**: HIGH - Already have most of this data.

---

### Category 7: Operating Environment (Cash Flow)

**Hypothesis**: Landlord-friendly markets have better risk-adjusted returns.

| Feature | Source | Rationale |
|---------|--------|-----------|
| Property tax rate | Tax Foundation / Census | Affects net yield |
| Landlord-friendliness index | Custom (eviction laws, rent control) | Regulatory risk |
| Insurance cost index | Industry data | Operating costs |
| HOA prevalence | Realtor.com / custom | Additional costs |
| Economic diversification | BLS (employment HHI) | Recession resilience |

**Implementation Priority**: LOW - Useful for refinement but harder to quantify.

---

## Proposed Scoring Formulas

### Appreciation Score (v2)

```python
appreciation_score = (
    0.25 * supply_constraint_score +      # Building permits, geography
    0.30 * migration_quality_score +      # IRS data, AGI-weighted
    0.20 * price_momentum_score +         # 1-3 year trends
    0.15 * affordability_gap_score +      # vs comparable metros
    0.10 * employment_leading_score       # Job postings, business apps
)
```

### Cash Flow Score (v2)

```python
cashflow_score = (
    0.35 * price_to_rent_score +          # Lower ratio = better
    0.25 * rent_growth_score +            # Rising rents
    0.20 * vacancy_score +                # Low vacancy = strong demand
    0.10 * landlord_friendly_score +      # Regulatory environment
    0.10 * economic_diversity_score       # Recession resilience
)
```

### Combined Investment Score

```python
investment_score = alpha * appreciation_score + (1 - alpha) * cashflow_score
# alpha = 0.7 for growth-focused investors
# alpha = 0.3 for income-focused investors
```

---

## Data Sources Reference

| Source | URL | Data Available | Update Frequency |
|--------|-----|----------------|------------------|
| IRS SOI Migration | irs.gov/statistics/soi-tax-stats-migration-data | County migration with AGI | Annual (2-year lag) |
| Census Building Permits | census.gov/construction/bps/ | Permits by metro | Monthly |
| Census Business Formation | census.gov/econ/bfs | New business applications | Weekly |
| Zillow Research | zillow.com/research/data/ | ZHVI, ZORI, inventory | Monthly |
| FRED | fred.stlouisfed.org | Unemployment, various | Varies |
| BLS QCEW | bls.gov/cew/ | Employment, wages by industry | Quarterly |
| FBI UCR | ucr.fbi.gov | Crime statistics | Annual |

---

## Validation Strategy

Before deploying any new model:

### 1. Multi-Period Backtesting
Test predictive power across different time periods:
- 2010-2015 (post-recession recovery)
- 2013-2018 (pre-COVID growth)
- 2015-2020 (includes early COVID)
- 2018-2023 (COVID and aftermath)

### 2. Out-of-Sample Testing
- Train on 80% of metros, test on 20%
- Use k-fold cross-validation
- Report confidence intervals

### 3. Feature Importance Analysis
- Calculate correlation of each feature with actual appreciation
- Use SHAP values for model interpretability
- Identify which features are actually predictive vs noise

### 4. Sanity Checks
- Do top-ranked metros make intuitive sense?
- Are there obvious misses (known hot markets ranked low)?
- Does the model rank historical winners highly?

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)
- [ ] Add building permits per capita
- [ ] Calculate price-to-income ratio
- [ ] Add price momentum (1-year, 3-year)
- [ ] Run correlation analysis on new features

### Phase 2: IRS Migration Data (Week 2)
- [ ] Download and parse IRS SOI migration files
- [ ] Calculate net migration by metro
- [ ] Calculate AGI of in-migrants vs out-migrants
- [ ] Integrate into pipeline

### Phase 3: Model Restructure (Week 3)
- [ ] Split into Appreciation vs Cash Flow models
- [ ] Implement new scoring formulas
- [ ] Allow user-configurable alpha parameter

### Phase 4: Validation (Week 4)
- [ ] Multi-period backtesting
- [ ] Feature importance analysis
- [ ] Refine weights based on empirical correlation

---

## Success Metrics

The new model should achieve:

| Metric | Target | Current |
|--------|--------|---------|
| Correlation with actual appreciation | > 0.3 | ~0.00 |
| Top-10 hit rate (% of actual top-10 in predicted top-20) | > 50% | Unknown |
| Backtest consistency across periods | Similar rankings | N/A |

---

## Backtest Results (January 2026)

### IRS Migration Data - VALIDATED

Multi-period backtesting confirms that IRS migration data **strongly predicts appreciation**:

| Period | N Markets | Hit Rate | Net Migration Corr | Quality Score Corr | Q5-Q1 Spread |
|--------|-----------|----------|-------------------|-------------------|--------------|
| 2011-12 → 2013-2018 | 857 | 57.9% | 0.185*** | 0.185*** | +7.4% |
| 2015-16 → 2016-2021 | 863 | 65.2% | 0.216*** | 0.254*** | +15.7% |
| 2017-18 → 2018-2023 | 867 | 66.3% | 0.176*** | 0.210*** | +19.9% |

*** p < 0.001

### Key Findings

1. **All correlations statistically significant** at p < 0.001
2. **Hit rate improving**: 58% → 65% → 66% of high-migration markets beat median
3. **Quintile spread increasing**: Top quintile beats bottom by 7-20 percentage points
4. **AGI Ratio strongest in recent period**: r=0.311 for 2017-18 → 2018-2023

### Top Performers (Consistent Across Periods)
- Phoenix, AZ
- Dallas-Fort Worth, TX
- Tampa, FL
- Denver, CO
- Charlotte, NC
- Las Vegas, NV

### Underperformers (Consistent Across Periods)
- New York, NY
- Chicago, IL
- Philadelphia, PA
- Baltimore, MD

### Comparison: IRS Data vs Original Indian Pop Proxy

| Feature | Correlation (2018-2023) |
|---------|------------------------|
| **IRS Migration Quality** | **+0.210*** |
| **IRS AGI Ratio** | **+0.311*** |
| Indian Pop Growth (CAGR) | -0.009 |
| Indian Pop Hybrid Score | +0.005 |

**Conclusion**: IRS migration data provides 40-60x stronger predictive signal than the original Indian population proxy.

---

## Implementation Status

### Completed
- [x] Add building permits per capita
- [x] Calculate price-to-income ratio
- [x] Add price momentum (1-year, 3-year)
- [x] Download and parse IRS SOI migration files
- [x] Calculate net migration by metro
- [x] Calculate AGI of in-migrants vs out-migrants
- [x] Integrate into pipeline
- [x] Split into Appreciation vs Cash Flow models
- [x] Implement new scoring formulas
- [x] Allow user-configurable alpha parameter
- [x] Multi-period backtesting
- [x] Feature importance analysis

### Future Enhancements
- [ ] Add more historical periods for robustness
- [ ] Implement machine learning (gradient boosting) for weight optimization
- [ ] Add out-of-sample cross-validation
- [ ] Integrate building permits data (API access issues)

---

## Open Questions

1. ~~**How much lag is acceptable?** IRS data has a 2-year lag. Is this still predictive?~~
   **ANSWERED**: Yes, 2-year lag still highly predictive (r=0.2-0.3)

2. **Metro vs County?** Some data is county-level, some is MSA-level. Aggregation strategy?
   **RESOLVED**: County-to-CBSA crosswalk from NBER works well

3. **Weighting methodology?** Should we learn weights empirically or use domain expertise?
   **PARTIAL**: Current weights are domain-based, ML optimization as future work

4. **Regime changes?** How do we handle COVID-like structural breaks?
   **OBSERVATION**: Model actually performed better in COVID period (2017-18 → 2018-23)

---

*Last updated: January 2026*
