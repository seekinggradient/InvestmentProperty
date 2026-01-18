#!/usr/bin/env python3
"""
Historical Backtesting of Tech Hub Algorithm

Tests if our 2025 algorithm would have predicted the actual winners
of 2018-2023 (or other configurable time periods).

Validates assumptions:
- Does Indian population growth predict price appreciation?
- Does tech wage growth predict appreciation?
- Which features are the BEST predictors?

Usage:
    python backtest_algorithm.py                          # Default: 2018-2023
    python backtest_algorithm.py --start 2018 --end 2023  # Custom period
    python backtest_algorithm.py --no-cache               # Force fresh fetch

Note: Census ACS data with consistent variables available from 2013 onwards.
      For backtests starting before 2018, limited historical lookback may apply.
"""

import pandas as pd
import numpy as np
import requests
import argparse
from pathlib import Path
import pickle
from typing import Tuple
from scipy import stats


class BacktestAnalyzer:
    """Backtest investment algorithm using historical data"""

    def __init__(self, start_year: int = 2018, end_year: int = 2023,
                 cache_dir: str = 'cache', use_cache: bool = True):
        self.start_year = start_year
        self.end_year = end_year
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_key(self, key: str) -> Path:
        return self.cache_dir / f"backtest_{key}.pkl"

    def _load_cache(self, key: str) -> pd.DataFrame:
        cache_path = self._cache_key(key)
        if self.use_cache and cache_path.exists():
            print(f"  Loading from cache: {key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return pd.DataFrame()

    def _save_cache(self, key: str, data: pd.DataFrame):
        with open(self._cache_key(key), 'wb') as f:
            pickle.dump(data, f)
        print(f"  Cached: {key}")

    def fetch_historical_zillow_appreciation(self) -> pd.DataFrame:
        """
        Fetch historical Zillow ZHVI data to calculate actual appreciation
        Returns: DataFrame with metro, start_price, end_price, appreciation_pct
        """
        cache_key = f"zillow_hist_{self.start_year}_{self.end_year}"
        cached = self._load_cache(cache_key)
        if not cached.empty:
            return cached

        print(f"\nFetching historical Zillow data ({self.start_year}-{self.end_year})...")

        try:
            # Zillow ZHVI has monthly data back to 2000
            url = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
            df = pd.read_csv(url)

            # Find columns for start and end years (use December of each year)
            start_col = f"{self.start_year}-12-31"
            end_col = f"{self.end_year}-12-31"

            # If exact date not available, find closest
            date_cols = [col for col in df.columns if '-' in str(col) and col not in ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']]

            # Find closest dates
            start_col_actual = min([col for col in date_cols if col.startswith(str(self.start_year))],
                                   key=lambda x: abs(pd.to_datetime(x).month - 12))
            end_col_actual = min([col for col in date_cols if col.startswith(str(self.end_year))],
                                 key=lambda x: abs(pd.to_datetime(x).month - 12))

            print(f"  Using dates: {start_col_actual} → {end_col_actual}")

            # Extract data
            result = df[['RegionName', start_col_actual, end_col_actual]].copy()
            result.columns = ['msa_name_zillow', 'price_start', 'price_end']

            # Calculate appreciation
            result['price_start'] = pd.to_numeric(result['price_start'], errors='coerce')
            result['price_end'] = pd.to_numeric(result['price_end'], errors='coerce')
            result = result.dropna(subset=['price_start', 'price_end'])

            # Calculate % appreciation and CAGR
            years = self.end_year - self.start_year
            result['appreciation_pct'] = ((result['price_end'] / result['price_start']) - 1) * 100
            result['appreciation_cagr'] = (((result['price_end'] / result['price_start']) ** (1/years)) - 1) * 100

            # Clean metro names for matching
            result['msa_match_key'] = result['msa_name_zillow'].str.split(',').str[0].str.lower().str.strip()

            print(f"  Retrieved appreciation data for {len(result)} metros")
            self._save_cache(cache_key, result)

            return result

        except Exception as e:
            print(f"  Error fetching Zillow data: {e}")
            return pd.DataFrame()

    def fetch_historical_census(self, year: int) -> pd.DataFrame:
        """Fetch Census ACS 5-Year data for a historical year"""
        cache_key = f"census_hist_{year}"
        cached = self._load_cache(cache_key)
        if not cached.empty:
            return cached

        print(f"\nFetching Census ACS 5-Year {year} data...")

        try:
            variables = [
                "B02015_021E",  # Asian Indian
                "B15003_023E",  # Master's
                "B15003_025E",  # Doctorate
                "B19013_001E",  # Median HH Income
                "B25077_001E",  # Median Home Value
                "NAME"
            ]

            url = f"https://api.census.gov/data/{year}/acs/acs5"
            params = {
                'get': ','.join(variables),
                'for': 'metropolitan statistical area/micropolitan statistical area:*'
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data[1:], columns=data[0])

            # Rename columns
            df.columns = ['indian_pop', 'masters', 'doctorate', 'median_income',
                         'median_home_value', 'msa_name', 'msa_code']

            # Convert to numeric
            for col in ['indian_pop', 'masters', 'doctorate', 'median_income', 'median_home_value']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['advanced_degrees'] = df['masters'] + df['doctorate']
            df['year'] = year

            # Clean metro names
            df['msa_match_key'] = df['msa_name'].str.split('-').str[0].str.split(',').str[0].str.lower().str.strip()

            print(f"  Retrieved {len(df)} MSAs")
            self._save_cache(cache_key, df)

            return df

        except Exception as e:
            print(f"  Error fetching Census {year}: {e}")
            return pd.DataFrame()

    def calculate_historical_features(self) -> pd.DataFrame:
        """
        Calculate features for the start year (e.g., 2015)
        Same features we use in current algorithm
        """
        print(f"\n{'='*80}")
        print(f"CALCULATING {self.start_year} FEATURES (What we would have known then)")
        print(f"{'='*80}")

        # We need census data from 5 years before start_year and at start_year
        # to calculate growth rates
        # Note: Census ACS 5-Year data with consistent API available from 2015 onwards
        baseline_year = max(2015, self.start_year - 5)

        print(f"  Using baseline year: {baseline_year} (actual lookback: {self.start_year - baseline_year} years)")

        census_baseline = self.fetch_historical_census(baseline_year)
        census_start = self.fetch_historical_census(self.start_year)

        if census_baseline.empty or census_start.empty:
            print("Failed to fetch Census data")
            return pd.DataFrame()

        # Merge to calculate growth
        merged = census_baseline.merge(
            census_start,
            on='msa_match_key',
            suffixes=(f'_{baseline_year}', f'_{self.start_year}')
        )

        # Calculate CAGR based on actual years
        years = self.start_year - baseline_year
        def calc_cagr(start, end):
            if pd.isna(start) or pd.isna(end) or start <= 0:
                return 0
            return ((end / start) ** (1 / years) - 1) * 100

        merged['indian_pop_cagr'] = merged.apply(
            lambda x: calc_cagr(x[f'indian_pop_{baseline_year}'], x[f'indian_pop_{self.start_year}']),
            axis=1
        )

        merged['advanced_degree_cagr'] = merged.apply(
            lambda x: calc_cagr(x[f'advanced_degrees_{baseline_year}'], x[f'advanced_degrees_{self.start_year}']),
            axis=1
        )

        # Select final columns
        features = merged[[
            'msa_match_key',
            f'msa_name_{self.start_year}',
            f'indian_pop_{self.start_year}',
            'indian_pop_cagr',
            f'advanced_degrees_{self.start_year}',
            'advanced_degree_cagr',
            f'median_home_value_{self.start_year}',
            f'median_income_{self.start_year}'
        ]].copy()

        features.columns = [
            'msa_match_key', 'msa_name', 'indian_pop', 'indian_pop_cagr',
            'advanced_degrees', 'advanced_degree_cagr',
            'median_home_value', 'median_income'
        ]

        # Calculate hybrid Indian pop score (same as current algorithm)
        features['indian_pop_hybrid'] = features['indian_pop_cagr'] * np.log10(features['indian_pop'].clip(lower=1))

        print(f"  Calculated features for {len(features)} metros")

        return features

    def run_backtest(self) -> pd.DataFrame:
        """
        Main backtesting logic:
        1. Get historical features (what we knew in start_year)
        2. Get actual appreciation (what happened start_year → end_year)
        3. Analyze correlation
        """
        print(f"\n{'='*80}")
        print(f"BACKTESTING: {self.start_year} Algorithm → {self.end_year} Results")
        print(f"{'='*80}")

        # Get features from start year
        features = self.calculate_historical_features()

        # Get actual appreciation
        appreciation = self.fetch_historical_zillow_appreciation()

        if features.empty or appreciation.empty:
            print("Failed to get required data")
            return pd.DataFrame()

        # Merge
        backtest = features.merge(appreciation, on='msa_match_key', how='inner')

        print(f"\n  Successfully matched {len(backtest)} metros with both features and outcomes")

        return backtest

    def analyze_correlations(self, df: pd.DataFrame):
        """Analyze which features correlate with actual appreciation"""
        print(f"\n{'='*80}")
        print("CORRELATION ANALYSIS: Which Features Predicted Appreciation?")
        print(f"{'='*80}\n")

        # Features to test
        feature_cols = {
            'indian_pop_cagr': f'Indian Pop Growth ({self.start_year-5}-{self.start_year})',
            'indian_pop_hybrid': 'Indian Pop Hybrid Score (CAGR × log pop)',
            'indian_pop': f'Indian Pop Absolute ({self.start_year})',
            'advanced_degree_cagr': f'Advanced Degree Growth ({self.start_year-5}-{self.start_year})',
            'median_home_value': f'Home Value ({self.start_year})',
            'median_income': f'Median Income ({self.start_year})'
        }

        # Calculate correlations
        correlations = []
        for col, label in feature_cols.items():
            if col in df.columns:
                # Remove NaN and infinite values
                valid_data = df[[col, 'appreciation_cagr']].replace([np.inf, -np.inf], np.nan).dropna()

                if len(valid_data) > 10:  # Need enough data points
                    corr, p_value = stats.pearsonr(valid_data[col], valid_data['appreciation_cagr'])
                    correlations.append({
                        'Feature': label,
                        'Correlation': corr,
                        'P-Value': p_value,
                        'Significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                    })

        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False, key=abs)

        print("Correlation with Actual Appreciation (CAGR):")
        print(f"{'Feature':<50s} {'Correlation':>12s} {'P-Value':>10s} {'Sig':>5s}")
        print("-" * 80)

        for _, row in corr_df.iterrows():
            print(f"{row['Feature']:<50s} {row['Correlation']:>12.3f} {row['P-Value']:>10.4f} {row['Significance']:>5s}")

        print("\nInterpretation:")
        print("  Correlation >  0.5: Strong positive relationship")
        print("  Correlation >  0.3: Moderate positive relationship")
        print("  Correlation > -0.3: Weak relationship")
        print("  Correlation < -0.5: Strong negative relationship")
        print("  *, **, ***: Statistical significance (p < 0.05, 0.01, 0.001)")

        return corr_df

    def show_winners_losers(self, df: pd.DataFrame, n: int = 10):
        """Show which metros appreciated most vs our predictions"""
        print(f"\n{'='*80}")
        print(f"ACTUAL WINNERS ({self.start_year}-{self.end_year})")
        print(f"{'='*80}\n")

        # Calculate algorithm scores (simple version - just use hybrid score)
        df['algorithm_score'] = df['indian_pop_hybrid']
        df['algorithm_rank'] = df['algorithm_score'].rank(ascending=False)

        # Top appreciators
        winners = df.nlargest(n, 'appreciation_cagr')

        print(f"Top {n} Markets by Actual Appreciation:")
        print(f"{'Metro':<35s} {'Actual CAGR':>12s} {'Algo Rank':>10s} {'Indian Pop Growth':>18s}")
        print("-" * 80)

        for _, row in winners.iterrows():
            metro_name = row['msa_name_zillow'].split(',')[0][:34]
            print(f"{metro_name:<35s} {row['appreciation_cagr']:>11.1f}% "
                  f"{int(row['algorithm_rank']):>10d} {row['indian_pop_cagr']:>17.1f}%")

        print(f"\n{'='*80}")
        print(f"ALGORITHM'S {self.start_year} TOP PICKS (How Did They Do?)")
        print(f"{'='*80}\n")

        # Top algorithm picks
        algo_picks = df.nsmallest(n, 'algorithm_rank')

        print(f"Algorithm's Top {n} Picks:")
        print(f"{'Metro':<35s} {'Predicted Rank':>14s} {'Actual CAGR':>12s} {'Hit/Miss':>10s}")
        print("-" * 80)

        for _, row in algo_picks.iterrows():
            metro_name = row['msa_name_zillow'].split(',')[0][:34]
            hit_miss = "✅ HIT" if row['appreciation_cagr'] > df['appreciation_cagr'].median() else "❌ MISS"
            print(f"{metro_name:<35s} {int(row['algorithm_rank']):>14d} "
                  f"{row['appreciation_cagr']:>11.1f}% {hit_miss:>10s}")


def main():
    parser = argparse.ArgumentParser(
        description='Backtest investment algorithm using historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--start', type=int, default=2018,
                       help='Start year for backtest (default: 2018)')
    parser.add_argument('--end', type=int, default=2023,
                       help='End year for backtest (default: 2023)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Force fresh data fetch')
    parser.add_argument('--top', type=int, default=15,
                       help='Number of top markets to show (default: 15)')

    args = parser.parse_args()

    # Create analyzer
    analyzer = BacktestAnalyzer(
        start_year=args.start,
        end_year=args.end,
        use_cache=not args.no_cache
    )

    # Run backtest
    results = analyzer.run_backtest()

    if results.empty:
        print("\nBacktest failed - could not get required data")
        return

    # Analyze correlations
    correlations = analyzer.analyze_correlations(results)

    # Show winners and algorithm picks
    analyzer.show_winners_losers(results, n=args.top)

    # Save results
    output_file = f'backtest_results_{args.start}_{args.end}.csv'
    results.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Full results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
