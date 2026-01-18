#!/usr/bin/env python3
"""
Multi-Period Backtesting for Investment Property Model v2

Tests whether model features (migration, price momentum, affordability)
predicted actual appreciation across multiple historical periods.

Periods tested:
- 2011-2012 migration → 2015-2018 appreciation
- 2015-2016 migration → 2018-2021 appreciation
- 2017-2018 migration → 2020-2023 appreciation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from irs_migration import IRSMigrationProcessor


class BacktestEngine:
    """Multi-period backtesting for appreciation model"""

    def __init__(self, data_dir: str = 'data', cache_dir: str = 'cache'):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.irs_processor = IRSMigrationProcessor(data_dir, cache_dir)

    def get_census_home_values(self, year: int) -> pd.DataFrame:
        """Get Census median home values for a given year"""
        import requests
        import pickle

        cache_path = self.cache_dir / f'census_{year}.pkl'
        if cache_path.exists():
            print(f"  Loading Census {year} from cache")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"  Fetching Census {year} data...")
        url = f"https://api.census.gov/data/{year}/acs/acs5"
        params = {
            'get': 'NAME,B25077_001E',  # Median home value
            'for': 'metropolitan statistical area/micropolitan statistical area:*'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data[1:], columns=['msa_name', 'median_home_value', 'msa_code'])
            df['median_home_value'] = pd.to_numeric(df['median_home_value'], errors='coerce')
            df['year'] = year

            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)

            return df

        except Exception as e:
            print(f"  Error fetching Census {year}: {e}")
            return pd.DataFrame()

    def calculate_appreciation(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Calculate actual appreciation between two Census periods"""
        print(f"\nCalculating appreciation {start_year} → {end_year}...")

        start_df = self.get_census_home_values(start_year)
        end_df = self.get_census_home_values(end_year)

        if start_df.empty or end_df.empty:
            print("  Could not fetch Census data")
            return pd.DataFrame()

        # Merge start and end values
        merged = start_df.merge(
            end_df[['msa_code', 'median_home_value']],
            on='msa_code',
            suffixes=('_start', '_end')
        )

        # Calculate appreciation
        years = end_year - start_year
        merged['appreciation_pct'] = (
            (merged['median_home_value_end'] - merged['median_home_value_start']) /
            merged['median_home_value_start'] * 100
        )
        merged['appreciation_cagr'] = (
            (merged['median_home_value_end'] / merged['median_home_value_start']) ** (1/years) - 1
        ) * 100

        # Remove invalid values
        merged = merged[
            (merged['median_home_value_start'] > 0) &
            (merged['median_home_value_end'] > 0) &
            (merged['appreciation_pct'].notna())
        ]

        print(f"  Calculated appreciation for {len(merged)} MSAs")
        print(f"  Mean appreciation: {merged['appreciation_pct'].mean():.1f}%")
        print(f"  Median appreciation: {merged['appreciation_pct'].median():.1f}%")

        return merged

    def run_period_backtest(self, migration_year: str, appreciation_start: int,
                           appreciation_end: int) -> dict:
        """
        Run backtest for a single period.

        Args:
            migration_year: IRS migration year pair (e.g., '1516')
            appreciation_start: Start year for appreciation calculation
            appreciation_end: End year for appreciation calculation
        """
        print(f"\n{'='*70}")
        print(f"BACKTEST: Migration {migration_year} → Appreciation {appreciation_start}-{appreciation_end}")
        print(f"{'='*70}")

        # Get migration data
        try:
            migration_df = self.irs_processor.process_year_pair(migration_year)
        except FileNotFoundError:
            print(f"  Migration data not available for {migration_year}")
            return None

        # Get actual appreciation
        appreciation_df = self.calculate_appreciation(appreciation_start, appreciation_end)

        if appreciation_df.empty:
            return None

        # Merge migration and appreciation data
        merged = migration_df.merge(
            appreciation_df[['msa_code', 'appreciation_pct', 'appreciation_cagr',
                           'median_home_value_start', 'median_home_value_end']],
            on='msa_code',
            how='inner'
        )

        print(f"\n  Merged {len(merged)} MSAs with both migration and appreciation data")

        if len(merged) < 30:
            print("  Warning: Low sample size may affect statistical significance")

        # Calculate correlations
        results = {
            'period': f"{migration_year} → {appreciation_start}-{appreciation_end}",
            'n_markets': len(merged),
            'correlations': {}
        }

        features = [
            ('net_migration_returns', 'Net Migration (households)'),
            ('avg_agi_inflow', 'Avg AGI Inflow'),
            ('agi_ratio', 'AGI Ratio (in/out)'),
            ('migration_quality', 'Migration Quality Score'),
        ]

        print("\n  Feature Correlations with Actual Appreciation:")
        print("  " + "-"*60)

        for col, name in features:
            if col in merged.columns:
                # Filter valid values
                valid = merged[[col, 'appreciation_pct']].dropna()

                if len(valid) > 10:
                    corr, p_value = stats.pearsonr(valid[col], valid['appreciation_pct'])
                    spearman, sp_p = stats.spearmanr(valid[col], valid['appreciation_pct'])

                    results['correlations'][col] = {
                        'pearson': corr,
                        'pearson_p': p_value,
                        'spearman': spearman,
                        'spearman_p': sp_p
                    }

                    sig_marker = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                    print(f"  {name:30s}: r={corr:+.3f} (p={p_value:.3f}) {sig_marker}")

        # Top/Bottom analysis
        print("\n  Top 10 Markets by Migration Quality:")
        top10 = merged.nlargest(10, 'migration_quality')[
            ['msa_name', 'migration_quality', 'appreciation_pct']
        ]
        for _, row in top10.iterrows():
            print(f"    {row['msa_name'][:40]:40s}: {row['appreciation_pct']:+.1f}%")

        print("\n  Bottom 10 Markets by Migration Quality:")
        bottom10 = merged.nsmallest(10, 'migration_quality')[
            ['msa_name', 'migration_quality', 'appreciation_pct']
        ]
        for _, row in bottom10.iterrows():
            print(f"    {row['msa_name'][:40]:40s}: {row['appreciation_pct']:+.1f}%")

        # Calculate hit rate
        median_migration = merged['migration_quality'].median()
        median_appreciation = merged['appreciation_pct'].median()

        high_migration = merged[merged['migration_quality'] > median_migration]
        high_migration_high_appreciation = high_migration[high_migration['appreciation_pct'] > median_appreciation]

        hit_rate = len(high_migration_high_appreciation) / len(high_migration) * 100 if len(high_migration) > 0 else 0
        results['hit_rate'] = hit_rate

        print(f"\n  Hit Rate: {hit_rate:.1f}%")
        print(f"  (% of high-migration markets that beat median appreciation)")

        # Average appreciation by quintile
        merged['migration_quintile'] = pd.qcut(merged['migration_quality'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        quintile_appreciation = merged.groupby('migration_quintile')['appreciation_pct'].mean()

        print("\n  Appreciation by Migration Quintile:")
        for q, apr in quintile_appreciation.items():
            print(f"    {q}: {apr:+.1f}%")

        results['quintile_appreciation'] = quintile_appreciation.to_dict()

        return results

    def run_all_backtests(self) -> pd.DataFrame:
        """Run backtests for all available periods"""

        # Define test periods
        # (migration_year, appreciation_start, appreciation_end)
        periods = [
            ('1112', 2013, 2018),  # 2011-12 migration → 2013-2018 appreciation
            ('1516', 2016, 2021),  # 2015-16 migration → 2016-2021 appreciation
            ('1718', 2018, 2023),  # 2017-18 migration → 2018-2023 appreciation
        ]

        all_results = []

        for migration_year, apr_start, apr_end in periods:
            result = self.run_period_backtest(migration_year, apr_start, apr_end)
            if result:
                all_results.append(result)

        # Summary
        print("\n" + "="*70)
        print("BACKTEST SUMMARY")
        print("="*70)

        if not all_results:
            print("No valid backtest results")
            return pd.DataFrame()

        summary_data = []
        for r in all_results:
            row = {
                'Period': r['period'],
                'N Markets': r['n_markets'],
                'Hit Rate': f"{r['hit_rate']:.1f}%",
            }
            if 'net_migration_returns' in r['correlations']:
                row['Net Migration Corr'] = f"{r['correlations']['net_migration_returns']['pearson']:.3f}"
            if 'migration_quality' in r['correlations']:
                row['Quality Score Corr'] = f"{r['correlations']['migration_quality']['pearson']:.3f}"
            if 'Q5' in r['quintile_appreciation'] and 'Q1' in r['quintile_appreciation']:
                spread = r['quintile_appreciation']['Q5'] - r['quintile_appreciation']['Q1']
                row['Q5-Q1 Spread'] = f"{spread:+.1f}%"

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))

        # Save results
        output_path = self.data_dir / 'backtest_results_v2.csv'
        summary_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        return summary_df


def main():
    """Run multi-period backtests"""
    print("\n" + "="*70)
    print("INVESTMENT PROPERTY MODEL v2 - MULTI-PERIOD BACKTEST")
    print("="*70)

    engine = BacktestEngine()
    results = engine.run_all_backtests()

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("""
The backtest measures whether IRS migration data (net migration, AGI quality)
predicted actual home price appreciation in subsequent years.

Key metrics:
- Pearson correlation: Linear relationship between migration and appreciation
- Hit rate: % of high-migration markets that beat median appreciation
- Q5-Q1 spread: Appreciation difference between top and bottom migration quintiles

Significance levels: * p<0.1, ** p<0.05, *** p<0.01
""")


if __name__ == "__main__":
    main()
