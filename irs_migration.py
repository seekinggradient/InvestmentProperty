#!/usr/bin/env python3
"""
IRS SOI Migration Data Processor

Processes IRS county-to-county migration data and aggregates to MSA level.
Calculates net migration, average AGI of migrants, and migration quality metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import pickle


class IRSMigrationProcessor:
    """Process IRS migration data and aggregate to MSA level"""

    def __init__(self, data_dir: str = 'data', cache_dir: str = 'cache'):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.crosswalk = None

    def load_crosswalk(self) -> pd.DataFrame:
        """Load county-to-CBSA crosswalk"""
        if self.crosswalk is not None:
            return self.crosswalk

        crosswalk_path = self.data_dir / 'cbsa2fipsxw_2023.csv'
        if not crosswalk_path.exists():
            raise FileNotFoundError(f"Crosswalk file not found: {crosswalk_path}")

        df = pd.read_csv(crosswalk_path, dtype=str)

        # Create combined FIPS code (state + county)
        df['fips'] = df['fipsstatecode'] + df['fipscountycode']

        # Keep only relevant columns
        self.crosswalk = df[['fips', 'cbsacode', 'cbsatitle', 'metropolitanmicropolitanstatis']].copy()
        self.crosswalk = self.crosswalk.rename(columns={
            'cbsacode': 'msa_code',
            'cbsatitle': 'msa_name',
            'metropolitanmicropolitanstatis': 'msa_type'
        })

        print(f"  Loaded crosswalk: {len(self.crosswalk)} county-MSA mappings")
        return self.crosswalk

    def load_migration_data(self, year_pair: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load IRS migration inflow and outflow data for a given year pair.

        Args:
            year_pair: Two-digit year pair like '2122' for 2021-2022
        """
        inflow_path = self.data_dir / f'countyinflow{year_pair}.csv'
        outflow_path = self.data_dir / f'countyoutflow{year_pair}.csv'

        if not inflow_path.exists() or not outflow_path.exists():
            raise FileNotFoundError(f"Migration files not found for {year_pair}")

        # Load inflow data (try multiple encodings for older files)
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                inflow = pd.read_csv(inflow_path, dtype={
                    'y1_statefips': str, 'y1_countyfips': str,
                    'y2_statefips': str, 'y2_countyfips': str
                }, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        # Load outflow data
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                outflow = pd.read_csv(outflow_path, dtype={
                    'y1_statefips': str, 'y1_countyfips': str,
                    'y2_statefips': str, 'y2_countyfips': str
                }, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        print(f"  Loaded migration data {year_pair}: {len(inflow)} inflow, {len(outflow)} outflow records")

        return inflow, outflow

    def aggregate_to_msa(self, inflow: pd.DataFrame, outflow: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate county-level migration data to MSA level.

        For each MSA, calculates:
        - Total in-migrants (returns and individuals)
        - Total out-migrants (returns and individuals)
        - Net migration
        - Total AGI of in-migrants and out-migrants
        - Average AGI per return for in and out migrants
        """
        crosswalk = self.load_crosswalk()

        # --- Process Inflow Data ---
        # Strip leading zeros from FIPS codes for consistent comparison
        inflow['y1_statefips_clean'] = inflow['y1_statefips'].str.lstrip('0').replace('', '0')
        inflow['y1_countyfips_clean'] = inflow['y1_countyfips'].str.lstrip('0').replace('', '0')

        # Filter to only "Different State" migrations (code 97, 3) to focus on meaningful relocations
        # Also include total US (97, 0) for overall picture
        inflow_filtered = inflow[
            ((inflow['y1_statefips_clean'] == '97') & (inflow['y1_countyfips_clean'] == '3')) |  # Different state
            ((inflow['y1_statefips_clean'] == '97') & (inflow['y1_countyfips_clean'] == '0'))    # Total US
        ].copy()

        # Create destination FIPS code (properly padded)
        inflow_filtered['dest_fips'] = (
            inflow_filtered['y2_statefips'].str.zfill(2) +
            inflow_filtered['y2_countyfips'].str.zfill(3)
        )

        # Merge with crosswalk to get MSA
        inflow_msa = inflow_filtered.merge(
            crosswalk[['fips', 'msa_code', 'msa_name']],
            left_on='dest_fips',
            right_on='fips',
            how='inner'
        )

        # Aggregate inflow by MSA - only different state migrations
        inflow_diff_state = inflow_msa[inflow_msa['y1_countyfips_clean'] == '3']
        inflow_agg = inflow_diff_state.groupby(['msa_code', 'msa_name']).agg({
            'n1': 'sum',  # Returns (households)
            'n2': 'sum',  # Exemptions (individuals)
            'agi': 'sum'  # Total AGI (in thousands)
        }).reset_index()
        inflow_agg.columns = ['msa_code', 'msa_name', 'inflow_returns', 'inflow_individuals', 'inflow_agi']

        # --- Process Outflow Data ---
        # Strip leading zeros from FIPS codes for consistent comparison
        outflow['y2_statefips_clean'] = outflow['y2_statefips'].str.lstrip('0').replace('', '0')
        outflow['y2_countyfips_clean'] = outflow['y2_countyfips'].str.lstrip('0').replace('', '0')

        outflow_filtered = outflow[
            ((outflow['y2_statefips_clean'] == '97') & (outflow['y2_countyfips_clean'] == '3')) |  # Different state
            ((outflow['y2_statefips_clean'] == '97') & (outflow['y2_countyfips_clean'] == '0'))    # Total US
        ].copy()

        # Create origin FIPS code (properly padded)
        outflow_filtered['origin_fips'] = (
            outflow_filtered['y1_statefips'].str.zfill(2) +
            outflow_filtered['y1_countyfips'].str.zfill(3)
        )

        # Merge with crosswalk
        outflow_msa = outflow_filtered.merge(
            crosswalk[['fips', 'msa_code', 'msa_name']],
            left_on='origin_fips',
            right_on='fips',
            how='inner'
        )

        # Aggregate outflow by MSA - only different state migrations
        outflow_diff_state = outflow_msa[outflow_msa['y2_countyfips_clean'] == '3']
        outflow_agg = outflow_diff_state.groupby(['msa_code', 'msa_name']).agg({
            'n1': 'sum',
            'n2': 'sum',
            'agi': 'sum'
        }).reset_index()
        outflow_agg.columns = ['msa_code', 'msa_name', 'outflow_returns', 'outflow_individuals', 'outflow_agi']

        # --- Merge and Calculate Metrics ---
        migration = inflow_agg.merge(outflow_agg, on=['msa_code', 'msa_name'], how='outer')
        migration = migration.fillna(0)

        # Net migration
        migration['net_migration_returns'] = migration['inflow_returns'] - migration['outflow_returns']
        migration['net_migration_individuals'] = migration['inflow_individuals'] - migration['outflow_individuals']

        # Average AGI per return (in thousands)
        migration['avg_agi_inflow'] = migration['inflow_agi'] / migration['inflow_returns'].clip(lower=1)
        migration['avg_agi_outflow'] = migration['outflow_agi'] / migration['outflow_returns'].clip(lower=1)

        # AGI ratio: are arrivals richer than departures?
        migration['agi_ratio'] = migration['avg_agi_inflow'] / migration['avg_agi_outflow'].clip(lower=1)

        # Net AGI flow
        migration['net_agi_flow'] = migration['inflow_agi'] - migration['outflow_agi']

        # Migration quality score: combines net migration with AGI quality
        # Higher = more people moving in with higher incomes
        migration['migration_quality'] = (
            migration['net_migration_returns'] * migration['agi_ratio']
        )

        print(f"  Aggregated to {len(migration)} MSAs")

        return migration

    def process_year_pair(self, year_pair: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Process migration data for a given year pair with caching.

        Args:
            year_pair: Two-digit year pair like '2122' for 2021-2022
            use_cache: Whether to use cached data if available
        """
        cache_path = self.cache_dir / f'irs_migration_{year_pair}.pkl'

        if use_cache and cache_path.exists():
            print(f"  Loading IRS migration from cache: {year_pair}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"Processing IRS migration data for {year_pair}...")

        inflow, outflow = self.load_migration_data(year_pair)
        migration = self.aggregate_to_msa(inflow, outflow)

        # Cache the result
        with open(cache_path, 'wb') as f:
            pickle.dump(migration, f)
        print(f"  Cached: irs_migration_{year_pair}")

        return migration

    def get_migration_trends(self, year_pairs: list) -> pd.DataFrame:
        """
        Calculate migration trends across multiple year pairs.

        Returns CAGR of net migration and AGI metrics.
        """
        all_years = {}
        for yp in year_pairs:
            df = self.process_year_pair(yp)
            # Add year suffix to columns
            year_label = f"20{yp[:2]}_{yp[2:]}"
            df_renamed = df[['msa_code', 'msa_name', 'net_migration_returns',
                            'avg_agi_inflow', 'agi_ratio', 'migration_quality']].copy()
            df_renamed.columns = ['msa_code', 'msa_name'] + [
                f'{c}_{year_label}' for c in ['net_migration', 'avg_agi_inflow', 'agi_ratio', 'migration_quality']
            ]
            all_years[yp] = df_renamed

        # Merge all years
        if len(all_years) < 2:
            return list(all_years.values())[0]

        result = list(all_years.values())[0]
        for df in list(all_years.values())[1:]:
            result = result.merge(df, on=['msa_code', 'msa_name'], how='outer')

        return result


def main():
    """Test the IRS migration processor"""
    processor = IRSMigrationProcessor()

    # Process available year pairs
    available_years = ['1112', '1516', '1718', '2122']

    for yp in available_years:
        try:
            migration = processor.process_year_pair(yp, use_cache=False)
            print(f"\n{yp} Migration Data Summary:")
            print(f"  Total MSAs: {len(migration)}")
            print(f"  Top 5 by net migration:")
            top5 = migration.nlargest(5, 'net_migration_returns')[
                ['msa_name', 'net_migration_returns', 'avg_agi_inflow', 'agi_ratio']
            ]
            print(top5.to_string(index=False))
            print(f"\n  Bottom 5 by net migration:")
            bottom5 = migration.nsmallest(5, 'net_migration_returns')[
                ['msa_name', 'net_migration_returns', 'avg_agi_inflow', 'agi_ratio']
            ]
            print(bottom5.to_string(index=False))
        except FileNotFoundError as e:
            print(f"  Skipping {yp}: {e}")

    print("\n" + "="*70)
    print("Processing complete!")


if __name__ == "__main__":
    main()
