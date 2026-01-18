#!/usr/bin/env python3
"""
U.S. Investment Property Analysis Engine v2

Identifies emerging real estate markets using two models:
1. APPRECIATION MODEL - Predicts price growth via:
   - Supply constraints (building permits per capita)
   - Migration quality (IRS data - income-weighted)
   - Price momentum (1-year, 3-year trends)
   - Affordability gap (vs comparable metros)
   - Employment indicators

2. CASH FLOW MODEL - Predicts rental yields via:
   - Price-to-rent ratio
   - Rent growth trends
   - Vacancy rates
   - Economic diversification

Legacy features (v1) still included for comparison:
- Indian population growth, tech wages, advanced degrees
"""

import requests
import pandas as pd
import numpy as np
from io import StringIO
import time
import os
import pickle
import argparse
from typing import Dict, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import IRS migration processor
try:
    from irs_migration import IRSMigrationProcessor
    IRS_AVAILABLE = True
except ImportError:
    IRS_AVAILABLE = False


class TechHubAnalyzer:
    """Pipeline for identifying emerging tech markets"""

    def __init__(self, census_api_key: str, fred_api_key: str = None,
                 cache_dir: str = 'cache', use_cache: bool = True):
        self.census_api_key = census_api_key
        self.fred_api_key = fred_api_key
        self.base_census_url = "https://api.census.gov/data"
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key"""
        return self.cache_dir / f"{key}.pkl"

    def _load_from_cache(self, key: str) -> pd.DataFrame:
        """Load data from cache if available"""
        cache_path = self._get_cache_path(key)
        if self.use_cache and cache_path.exists():
            print(f"  Loading from cache: {key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return pd.DataFrame()

    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """Save data to cache"""
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Cached: {key}")

    def get_census_data(self, year: int) -> pd.DataFrame:
        """
        Pull ACS 5-year estimates for MSAs (all sizes)

        Variables:
        - B02015_001E: Total Asian Population
        - B02015_021E: Asian Indian Population
        - B15003_023E: Master's Degree
        - B15003_025E: Doctorate Degree
        - B19013_001E: Median Household Income
        - B25077_001E: Median Home Value
        """
        # Check cache first
        cache_key = f"census_{year}"
        cached_data = self._load_from_cache(cache_key)
        if not cached_data.empty:
            return cached_data

        print(f"Fetching Census ACS 5-Year {year} data...")

        variables = [
            "B02015_001E",  # Total Asian
            "B02015_021E",  # Asian Indian
            "B15003_023E",  # Master's
            "B15003_025E",  # Doctorate
            "B19013_001E",  # Median HH Income
            "B25077_001E",  # Median Home Value
            "NAME"
        ]

        url = f"{self.base_census_url}/{year}/acs/acs5"
        params = {
            'get': ','.join(variables),
            'for': 'metropolitan statistical area/micropolitan statistical area:*'
        }

        # Add key if provided and valid
        if self.census_api_key:
            params['key'] = self.census_api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data[1:], columns=data[0])

            # Rename columns
            df.columns = [
                'total_asian', 'indian_pop', 'masters', 'doctorate',
                'median_income', 'median_home_value', 'msa_name', 'msa_code'
            ]

            # Convert to numeric
            numeric_cols = ['total_asian', 'indian_pop', 'masters', 'doctorate',
                          'median_income', 'median_home_value']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Create advanced degree column
            df['advanced_degrees'] = df['masters'] + df['doctorate']

            # Add year identifier
            df['year'] = year

            # Clean MSA names (remove state codes for cleaner output)
            df['msa_clean'] = df['msa_name'].str.replace(r',.*', '', regex=True)

            print(f"  Retrieved {len(df)} MSAs")

            # Save to cache
            self._save_to_cache(cache_key, df)

            return df

        except Exception as e:
            print(f"Error fetching Census data for {year}: {e}")
            return pd.DataFrame()

    def get_qcew_data(self, year: int, quarter: str, naics: str) -> pd.DataFrame:
        """
        Fetch BLS QCEW data for specific NAICS code

        NAICS Codes:
        - 5415: Computer Systems Design and Related Services
        - 5182: Data Processing, Hosting, and Related Services
        """
        url = f"https://data.bls.gov/cew/data/api/{year}/{quarter}/industry/{naics}.csv"

        print(f"Fetching QCEW {year} Q{quarter} data for NAICS {naics}...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))

            # Filter for MSA-level data only (agglvl_code == 46 for CBSA/MSA)
            df = df[df['agglvl_code'] == 46].copy()

            # Clean area_fips: Remove 'C' prefix and pad to 5 digits
            df['cbsa_code'] = df['area_fips'].str.replace('C', '', regex=False).str.lstrip('0') + '0'

            # Keep relevant columns
            df = df[[
                'area_fips', 'cbsa_code', 'industry_code',
                'total_qtrly_wages', 'qtrly_estabs', 'month3_emplvl'
            ]].copy()

            df['year'] = year
            df['quarter'] = quarter
            df['naics'] = naics

            print(f"  Retrieved {len(df)} MSA records")
            return df

        except Exception as e:
            print(f"Error fetching QCEW data for {year} Q{quarter} NAICS {naics}: {e}")
            return pd.DataFrame()

    def fetch_all_qcew_data(self) -> pd.DataFrame:
        """Fetch last 8 quarters of QCEW data for tech sectors"""

        # Updated to use most recent available data (2024-2025)
        quarters_to_fetch = [
            (2023, '3'), (2023, '4'),
            (2024, '1'), (2024, '2'), (2024, '3'), (2024, '4'),
            (2025, '1'), (2025, '2')
        ]

        naics_codes = ['5415', '5182']

        all_data = []

        for year, quarter in quarters_to_fetch:
            for naics in naics_codes:
                df = self.get_qcew_data(year, quarter, naics)
                if not df.empty:
                    all_data.append(df)
                time.sleep(0.5)  # Rate limiting

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            return combined
        else:
            return pd.DataFrame()

    def calculate_wage_growth(self, qcew_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate wage growth rates by MSA"""

        print("Calculating tech wage growth...")

        # Aggregate by area and time period
        agg_df = qcew_df.groupby(['area_fips', 'cbsa_code', 'year', 'quarter']).agg({
            'total_qtrly_wages': 'sum',
            'qtrly_estabs': 'sum',
            'month3_emplvl': 'sum'
        }).reset_index()

        # Create time period identifier
        agg_df['period'] = agg_df['year'].astype(str) + '_' + agg_df['quarter']

        # Get earliest and latest periods for each MSA
        growth_data = []

        for area_fips in agg_df['area_fips'].unique():
            area_data = agg_df[agg_df['area_fips'] == area_fips].sort_values(['year', 'quarter'])

            if len(area_data) >= 2:
                earliest = area_data.iloc[0]
                latest = area_data.iloc[-1]

                # Calculate CAGR
                periods_elapsed = len(area_data) / 4  # Convert quarters to years

                if earliest['total_qtrly_wages'] > 0 and periods_elapsed > 0:
                    wage_cagr = (
                        (latest['total_qtrly_wages'] / earliest['total_qtrly_wages']) **
                        (1 / periods_elapsed) - 1
                    ) * 100
                else:
                    wage_cagr = 0

                growth_data.append({
                    'area_fips': area_fips,
                    'cbsa_code': earliest['cbsa_code'],
                    'earliest_wages': earliest['total_qtrly_wages'],
                    'latest_wages': latest['total_qtrly_wages'],
                    'wage_cagr': wage_cagr,
                    'tech_establishments': latest['qtrly_estabs'],
                    'tech_employment': latest['month3_emplvl']
                })

        growth_df = pd.DataFrame(growth_data)
        print(f"  Calculated growth for {len(growth_df)} MSAs")

        return growth_df

    def calculate_demographic_growth(self, census_2018: pd.DataFrame,
                                    census_2023: pd.DataFrame) -> pd.DataFrame:
        """Calculate 5-year CAGR for demographic indicators"""

        print("Calculating demographic growth rates...")

        # Merge 2018 and 2023 data
        merged = census_2018.merge(
            census_2023,
            on='msa_code',
            suffixes=('_2018', '_2023')
        )

        # Calculate CAGR for key metrics
        years = 5

        def calc_cagr(start, end):
            if pd.isna(start) or pd.isna(end) or start <= 0:
                return 0
            return ((end / start) ** (1 / years) - 1) * 100

        merged['indian_pop_cagr'] = merged.apply(
            lambda x: calc_cagr(x['indian_pop_2018'], x['indian_pop_2023']), axis=1
        )

        merged['advanced_degree_cagr'] = merged.apply(
            lambda x: calc_cagr(x['advanced_degrees_2018'], x['advanced_degrees_2023']), axis=1
        )

        merged['income_cagr'] = merged.apply(
            lambda x: calc_cagr(x['median_income_2018'], x['median_income_2023']), axis=1
        )

        # Keep both baseline (2018) and latest (2023) values
        result = merged[[
            'msa_code', 'msa_name_2023', 'msa_clean_2023',
            'indian_pop_2018', 'indian_pop_2023',
            'advanced_degrees_2018', 'advanced_degrees_2023',
            'median_income_2023', 'median_home_value_2023',
            'indian_pop_cagr', 'advanced_degree_cagr', 'income_cagr'
        ]].copy()

        result.columns = [
            'msa_code', 'msa_name', 'msa_clean',
            'indian_pop_2018', 'indian_pop',
            'advanced_degrees_2018', 'advanced_degrees',
            'median_income', 'median_home_value',
            'indian_pop_cagr', 'advanced_degree_cagr', 'income_cagr'
        ]

        print(f"  Calculated growth for {len(result)} MSAs")

        return result

    def merge_datasets(self, demographic_df: pd.DataFrame,
                      wage_growth_df: pd.DataFrame) -> pd.DataFrame:
        """Merge Census and BLS data using CBSA codes"""

        print("Merging Census and BLS datasets...")

        # Merge on CBSA codes
        # Census has 'msa_code', BLS data now has 'cbsa_code' (cleaned)
        merged = demographic_df.merge(
            wage_growth_df,
            left_on='msa_code',
            right_on='cbsa_code',
            how='inner'
        )

        print(f"  Successfully merged {len(merged)} MSAs")

        return merged

    def get_zillow_data(self) -> pd.DataFrame:
        """
        Fetch Zillow Home Value Index (ZHVI) and Observed Rent Index (ZORI)
        Calculate price-to-rent ratios for investment analysis
        """
        # Check cache first
        cache_key = "zillow_data"
        cached_data = self._load_from_cache(cache_key)
        if not cached_data.empty:
            return cached_data

        print("Fetching Zillow real estate data...")

        try:
            # Fetch ZHVI (Home Values)
            zhvi_url = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
            zhvi_df = pd.read_csv(zhvi_url)

            # Get most recent month column (last column)
            latest_month = zhvi_df.columns[-1]
            zhvi_df = zhvi_df[['RegionName', latest_month]].copy()
            zhvi_df.columns = ['msa_name_zillow', 'zhvi_latest']

            # Fetch ZORI (Rents)
            zori_url = "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv"
            zori_df = pd.read_csv(zori_url)

            # Get most recent month
            latest_month_zori = zori_df.columns[-1]
            zori_df = zori_df[['RegionName', latest_month_zori]].copy()
            zori_df.columns = ['msa_name_zillow', 'zori_latest']

            # Merge ZHVI and ZORI
            zillow_df = zhvi_df.merge(zori_df, on='msa_name_zillow', how='inner')

            # Calculate price-to-rent ratio (annual)
            zillow_df['price_to_rent_ratio'] = zillow_df['zhvi_latest'] / (zillow_df['zori_latest'] * 12)

            # Clean MSA names for matching - extract first city name only
            # Zillow: "New York, NY" -> Census: "New York-Newark-Jersey City, NY-NJ-PA Metro Area"
            # Strategy: Use first city name before comma
            zillow_df['msa_match_key'] = zillow_df['msa_name_zillow'].str.split(',').str[0].str.lower().str.strip()

            print(f"  Retrieved Zillow data for {len(zillow_df)} metros")

            # Save to cache
            self._save_to_cache(cache_key, zillow_df)

            return zillow_df

        except Exception as e:
            print(f"  Warning: Could not fetch Zillow data: {e}")
            return pd.DataFrame()

    def get_irs_migration_data(self) -> pd.DataFrame:
        """
        Download IRS county-to-county migration data
        Aggregate to MSA level to show net migration and income of migrants

        Note: IRS data lags by ~2 years, using most recent available
        """
        print("Downloading IRS migration data...")

        try:
            # Most recent year available is typically 2 years behind
            year = 2021  # Update this as newer data becomes available

            # IRS provides county-level inflow data
            url = f"https://www.irs.gov/pub/irs-soi/county{year-1}{year[-2:]}inflow.csv"

            # Note: Actual implementation would require downloading, unzipping,
            # and aggregating county-level data to MSA level using CBSA definitions
            # This is complex and would significantly increase processing time

            print(f"  IRS migration data integration requires county-to-MSA mapping")
            print(f"  Skipping for now - can be added in future enhancement")
            return pd.DataFrame()

        except Exception as e:
            print(f"  Warning: Could not fetch IRS data: {e}")
            return pd.DataFrame()

    def get_state_unemployment_fred(self) -> pd.DataFrame:
        """
        Fetch state-level unemployment rates from FRED API
        More practical than metro-level (only 50 series IDs vs 400+)
        """
        if not self.fred_api_key:
            print("  Skipping FRED unemployment data (no API key provided)")
            return pd.DataFrame()

        # Check cache first
        cache_key = "fred_unemployment"
        cached_data = self._load_from_cache(cache_key)
        if not cached_data.empty:
            return cached_data

        print("Fetching state unemployment rates from FRED...")

        try:
            # State abbreviations to FRED series ID mapping
            state_abbrevs = [
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
            ]

            unemployment_data = []

            for state in state_abbrevs:
                # FRED series format: STATEABBR + 'UR' (e.g., 'CAUR' for California)
                series_id = f"{state}UR"

                url = f"{self.fred_base_url}/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': 1  # Most recent observation only
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if 'observations' in data and len(data['observations']) > 0:
                        latest = data['observations'][0]
                        unemployment_data.append({
                            'state_abbrev': state,
                            'unemployment_rate': float(latest['value']),
                            'date': latest['date']
                        })

            if len(unemployment_data) > 0:
                unemp_df = pd.DataFrame(unemployment_data)
                print(f"  Retrieved unemployment rates for {len(unemp_df)} states")

                # Save to cache
                self._save_to_cache(cache_key, unemp_df)

                return unemp_df
            else:
                print(f"  No unemployment data retrieved")
                return pd.DataFrame()

        except Exception as e:
            print(f"  Warning: Could not fetch FRED unemployment data: {e}")
            return pd.DataFrame()

    def get_building_permits_data(self) -> pd.DataFrame:
        """
        Fetch building permits data from Census Bureau

        Uses the Building Permits Survey (BPS) annual data by MSA
        Returns permits per 1000 residents as a supply constraint metric
        """
        cache_key = "building_permits"
        cached_data = self._load_from_cache(cache_key)
        if not cached_data.empty:
            return cached_data

        print("Fetching Census building permits data...")

        try:
            # Census Building Permits API for Metropolitan areas
            # Get most recent annual data (typically 1-2 year lag)
            year = 2023

            url = f"https://api.census.gov/data/{year}/bps/metro"
            params = {
                'get': 'NAME,BLDG_PERMITS,POP',
                'for': 'metropolitan statistical area/micropolitan statistical area:*'
            }

            if self.census_api_key:
                params['key'] = self.census_api_key

            response = requests.get(url, params=params, timeout=30)

            if response.status_code != 200:
                # Fallback: try the annual survey endpoint
                print("  Primary endpoint unavailable, trying annual survey...")
                url = f"https://api.census.gov/data/{year}/cbp"
                params = {
                    'get': 'NAME,ESTAB',
                    'for': 'metropolitan statistical area/micropolitan statistical area:*'
                }
                if self.census_api_key:
                    params['key'] = self.census_api_key
                response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data[1:], columns=data[0])

                # Clean up column names based on what we got
                if 'BLDG_PERMITS' in df.columns:
                    df['building_permits'] = pd.to_numeric(df['BLDG_PERMITS'], errors='coerce')
                    if 'POP' in df.columns:
                        df['permits_pop'] = pd.to_numeric(df['POP'], errors='coerce')
                        df['permits_per_1000'] = (df['building_permits'] / df['permits_pop']) * 1000

                # Extract MSA code
                msa_col = [c for c in df.columns if 'metropolitan' in c.lower() or 'msa' in c.lower()]
                if msa_col:
                    df['msa_code'] = df[msa_col[0]]

                print(f"  Retrieved building permits for {len(df)} metros")
                self._save_to_cache(cache_key, df)
                return df
            else:
                print(f"  Building permits API returned status {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"  Warning: Could not fetch building permits: {e}")
            return pd.DataFrame()

    def get_zillow_historical_data(self) -> pd.DataFrame:
        """
        Fetch Zillow ZHVI historical data to calculate price momentum

        Returns:
        - Current home value
        - 1-year price change
        - 3-year price change (CAGR)
        """
        cache_key = "zillow_historical"
        cached_data = self._load_from_cache(cache_key)
        if not cached_data.empty:
            return cached_data

        print("Fetching Zillow historical data for momentum calculation...")

        try:
            # Fetch ZHVI (Home Values) - full historical
            zhvi_url = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
            zhvi_df = pd.read_csv(zhvi_url)

            # Get date columns (format: YYYY-MM-DD)
            date_cols = [c for c in zhvi_df.columns if c.startswith('20')]

            if len(date_cols) < 36:
                print(f"  Warning: Only {len(date_cols)} months of data available")

            # Get latest, 1-year ago, and 3-year ago columns
            latest_col = date_cols[-1]
            one_year_ago_col = date_cols[-12] if len(date_cols) >= 12 else date_cols[0]
            three_year_ago_col = date_cols[-36] if len(date_cols) >= 36 else date_cols[0]

            result_df = zhvi_df[['RegionName']].copy()
            result_df['zhvi_current'] = zhvi_df[latest_col]
            result_df['zhvi_1yr_ago'] = zhvi_df[one_year_ago_col]
            result_df['zhvi_3yr_ago'] = zhvi_df[three_year_ago_col]

            # Calculate momentum metrics
            result_df['price_change_1yr'] = (
                (result_df['zhvi_current'] - result_df['zhvi_1yr_ago']) /
                result_df['zhvi_1yr_ago'] * 100
            )

            # 3-year CAGR
            result_df['price_cagr_3yr'] = (
                (result_df['zhvi_current'] / result_df['zhvi_3yr_ago']) ** (1/3) - 1
            ) * 100

            # Match key for merging
            result_df['msa_match_key'] = result_df['RegionName'].str.split(',').str[0].str.lower().str.strip()

            print(f"  Calculated price momentum for {len(result_df)} metros")
            print(f"  Latest data: {latest_col}, 1yr ago: {one_year_ago_col}, 3yr ago: {three_year_ago_col}")

            self._save_to_cache(cache_key, result_df)
            return result_df

        except Exception as e:
            print(f"  Warning: Could not fetch Zillow historical: {e}")
            return pd.DataFrame()

    def get_zillow_rent_historical(self) -> pd.DataFrame:
        """
        Fetch Zillow ZORI historical data to calculate rent growth
        """
        cache_key = "zillow_rent_historical"
        cached_data = self._load_from_cache(cache_key)
        if not cached_data.empty:
            return cached_data

        print("Fetching Zillow rent historical data...")

        try:
            zori_url = "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv"
            zori_df = pd.read_csv(zori_url)

            date_cols = [c for c in zori_df.columns if c.startswith('20')]

            latest_col = date_cols[-1]
            one_year_ago_col = date_cols[-12] if len(date_cols) >= 12 else date_cols[0]
            three_year_ago_col = date_cols[-36] if len(date_cols) >= 36 else date_cols[0]

            result_df = zori_df[['RegionName']].copy()
            result_df['zori_current'] = zori_df[latest_col]
            result_df['zori_1yr_ago'] = zori_df[one_year_ago_col]
            result_df['zori_3yr_ago'] = zori_df[three_year_ago_col]

            # Rent growth metrics
            result_df['rent_change_1yr'] = (
                (result_df['zori_current'] - result_df['zori_1yr_ago']) /
                result_df['zori_1yr_ago'] * 100
            )

            result_df['rent_cagr_3yr'] = (
                (result_df['zori_current'] / result_df['zori_3yr_ago']) ** (1/3) - 1
            ) * 100

            result_df['msa_match_key'] = result_df['RegionName'].str.split(',').str[0].str.lower().str.strip()

            print(f"  Calculated rent growth for {len(result_df)} metros")

            self._save_to_cache(cache_key, result_df)
            return result_df

        except Exception as e:
            print(f"  Warning: Could not fetch rent historical: {e}")
            return pd.DataFrame()

    def get_irs_migration_data(self, year_pair: str = '2122') -> pd.DataFrame:
        """
        Fetch IRS migration data aggregated to MSA level.

        Args:
            year_pair: Two-digit year pair (e.g., '2122' for 2021-2022)

        Returns DataFrame with:
        - net_migration_returns: Net household migration
        - avg_agi_inflow: Average AGI of incoming migrants (thousands)
        - agi_ratio: Ratio of incoming vs outgoing AGI
        - migration_quality: Combined migration quality score
        """
        if not IRS_AVAILABLE:
            print("  Warning: IRS migration module not available")
            return pd.DataFrame()

        cache_key = f"irs_migration_{year_pair}"
        cached_data = self._load_from_cache(cache_key)
        if not cached_data.empty:
            return cached_data

        print(f"Fetching IRS migration data ({year_pair})...")

        try:
            processor = IRSMigrationProcessor(data_dir='data', cache_dir=str(self.cache_dir))
            migration_df = processor.process_year_pair(year_pair, use_cache=False)

            # Select key columns for pipeline integration
            result_df = migration_df[[
                'msa_code', 'msa_name',
                'net_migration_returns', 'net_migration_individuals',
                'avg_agi_inflow', 'avg_agi_outflow', 'agi_ratio',
                'net_agi_flow', 'migration_quality'
            ]].copy()

            print(f"  Retrieved IRS migration data for {len(result_df)} MSAs")

            self._save_to_cache(cache_key, result_df)
            return result_df

        except Exception as e:
            print(f"  Warning: Could not fetch IRS migration data: {e}")
            return pd.DataFrame()

    def calculate_appreciation_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Appreciation Score (v2) - predicts price growth

        WITH IRS migration data:
        Score = 0.30 × Migration Quality Score (IRS net migration + AGI ratio)
              + 0.25 × Price Momentum Score (1yr + 3yr trends)
              + 0.20 × Supply Constraint Score (inverse of permits per capita)
              + 0.15 × Affordability Gap Score (vs comparable metros)
              + 0.10 × Tech Wage Score (employment growth)

        WITHOUT IRS data (fallback):
        Score = 0.25 × Supply Constraint Score
              + 0.25 × Price Momentum Score
              + 0.25 × Affordability Gap Score
              + 0.15 × Tech Wage Score
              + 0.10 × Migration Proxy Score (Indian pop)
        """
        print("Calculating Appreciation Scores (v2)...")

        has_irs_data = 'migration_quality' in df.columns and df['migration_quality'].notna().sum() > 0

        if has_irs_data:
            print("  Using IRS migration data for appreciation scoring")
        else:
            print("  Using Indian pop proxy (IRS data not available)")

        def normalize(series):
            if series.std() == 0:
                return pd.Series([50] * len(series), index=series.index)
            return ((series - series.min()) / (series.max() - series.min())) * 100

        # Supply Constraint Score - LOWER permits = MORE constrained = HIGHER score
        if 'permits_per_1000' in df.columns and df['permits_per_1000'].notna().sum() > 0:
            df['supply_constraint_score'] = normalize(
                1 / df['permits_per_1000'].clip(lower=0.1)
            )
        else:
            df['supply_constraint_score'] = normalize(df['median_home_value'].fillna(df['median_home_value'].median()))

        # Price Momentum Score - combines 1yr and 3yr trends
        if 'price_change_1yr' in df.columns and df['price_change_1yr'].notna().sum() > 0:
            momentum_1yr = normalize(df['price_change_1yr'].fillna(0))
            momentum_3yr = normalize(df['price_cagr_3yr'].fillna(0)) if 'price_cagr_3yr' in df.columns else momentum_1yr
            df['price_momentum_score'] = 0.6 * momentum_1yr + 0.4 * momentum_3yr
        else:
            df['price_momentum_score'] = 50

        # Affordability Gap Score - price relative to income
        df['price_to_income'] = df['median_home_value'] / df['median_income'].clip(lower=1)
        df['affordability_gap_score'] = normalize(1 / df['price_to_income'].clip(lower=0.1))

        # Tech Wage Score
        df['tech_wage_score'] = normalize(df['wage_cagr'].fillna(0))

        # Migration Score - use IRS data if available, otherwise Indian pop proxy
        if has_irs_data:
            # IRS Migration Quality Score
            # Combines net migration with income quality of migrants
            df['migration_score'] = normalize(df['migration_quality'].fillna(0))

            # Combined Appreciation Score (with IRS data - higher migration weight)
            df['appreciation_score'] = (
                0.30 * df['migration_score'] +
                0.25 * df['price_momentum_score'] +
                0.20 * df['supply_constraint_score'] +
                0.15 * df['affordability_gap_score'] +
                0.10 * df['tech_wage_score']
            )
        else:
            # Fallback: Indian pop proxy
            df['indian_pop_hybrid'] = df['indian_pop_cagr'] * np.log10(df['indian_pop'].clip(lower=1))
            df['migration_score'] = normalize(df['indian_pop_hybrid'].fillna(0))

            # Combined Appreciation Score (without IRS data)
            df['appreciation_score'] = (
                0.25 * df['supply_constraint_score'] +
                0.25 * df['price_momentum_score'] +
                0.25 * df['affordability_gap_score'] +
                0.15 * df['tech_wage_score'] +
                0.10 * df['migration_score']
            )

        return df

    def calculate_cashflow_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Cash Flow Score (v2) - predicts rental yields

        Score = 0.40 × Price-to-Rent Score (lower ratio = better)
              + 0.30 × Rent Growth Score (higher growth = better)
              + 0.20 × Income Stability Score (higher median income = stable tenants)
              + 0.10 × Unemployment Score (lower = better)
        """
        print("Calculating Cash Flow Scores (v2)...")

        def normalize(series):
            if series.std() == 0:
                return pd.Series([50] * len(series), index=series.index)
            return ((series - series.min()) / (series.max() - series.min())) * 100

        # Price-to-Rent Score - LOWER ratio = BETTER cash flow
        if 'price_to_rent_ratio' in df.columns and df['price_to_rent_ratio'].notna().sum() > 0:
            df['price_to_rent_score'] = normalize(
                1 / df['price_to_rent_ratio'].clip(lower=1)
            )
        else:
            df['price_to_rent_score'] = 50

        # Rent Growth Score - HIGHER growth = BETTER
        if 'rent_change_1yr' in df.columns and df['rent_change_1yr'].notna().sum() > 0:
            df['rent_growth_score'] = normalize(df['rent_change_1yr'].fillna(0))
        else:
            df['rent_growth_score'] = 50

        # Income Stability Score - HIGHER income = more stable tenants
        df['income_stability_score'] = normalize(df['median_income'].fillna(df['median_income'].median()))

        # Unemployment Score - LOWER unemployment = BETTER
        if 'unemployment_rate' in df.columns and df['unemployment_rate'].notna().sum() > 0:
            df['unemployment_score'] = normalize(
                1 / df['unemployment_rate'].clip(lower=0.1)
            )
        else:
            df['unemployment_score'] = 50

        # Combined Cash Flow Score
        df['cashflow_score'] = (
            0.40 * df['price_to_rent_score'] +
            0.30 * df['rent_growth_score'] +
            0.20 * df['income_stability_score'] +
            0.10 * df['unemployment_score']
        )

        return df

    def calculate_combined_score(self, df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
        """
        Calculate combined Investment Score

        Investment Score = alpha × Appreciation Score + (1-alpha) × Cash Flow Score

        Args:
            alpha: Weight for appreciation (0-1). Higher = more growth-focused.
                   Default 0.5 = balanced
        """
        print(f"Calculating Combined Investment Score (alpha={alpha})...")

        df['investment_score'] = (
            alpha * df['appreciation_score'] +
            (1 - alpha) * df['cashflow_score']
        )

        return df

    def calculate_boom_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Boom Probability Score

        Score = (0.3 × Tech Wages) + (0.25 × Hybrid Indian Pop) +
                (0.25 × Home Affordability) + (0.2 × Rental Cash Flow)

        Hybrid Indian Pop = CAGR × log₁₀(Population 2023)
        Rental Cash Flow = Inverse of Price-to-Rent ratio (lower P/R = better)

        Minimum threshold: 5,000 Indian population to avoid small-sample bias
        """

        print("Calculating Boom Probability Scores...")

        # Filter: Only keep MSAs with Indian population >= 5,000
        initial_count = len(df)
        df = df[df['indian_pop'] >= 5000].copy()
        filtered_count = len(df)
        print(f"  Filtered to {filtered_count} MSAs with Indian pop >= 5,000 (removed {initial_count - filtered_count})")

        if len(df) == 0:
            raise Exception("No MSAs meet the minimum population threshold")

        # Get San Jose median home value as benchmark
        san_jose_price = df[
            df['msa_clean'].str.contains('San Jose', case=False, na=False)
        ]['median_home_value'].values

        if len(san_jose_price) > 0:
            san_jose_price = san_jose_price[0]
            print(f"  San Jose benchmark price: ${san_jose_price:,.0f}")
        else:
            san_jose_price = 1_200_000  # Fallback estimate
            print(f"  Using estimated San Jose price: ${san_jose_price:,.0f}")

        # Normalize components to 0-100 scale
        def normalize(series):
            if series.std() == 0:
                return pd.Series([50] * len(series))
            return ((series - series.min()) / (series.max() - series.min())) * 100

        # Calculate normalized scores
        df['tech_wage_score'] = normalize(df['wage_cagr'])

        # Hybrid Indian Pop Score = CAGR × log₁₀(Population 2023)
        # This balances growth momentum with absolute community size
        df['indian_pop_hybrid'] = df['indian_pop_cagr'] * np.log10(df['indian_pop'])
        df['indian_pop_score'] = normalize(df['indian_pop_hybrid'])

        # Affordability: inverse of price relative to San Jose
        df['price_ratio'] = df['median_home_value'] / san_jose_price
        df['affordability_score'] = normalize(1 / df['price_ratio'].clip(lower=0.01))

        # Rental Cash Flow Score: inverse of price-to-rent ratio
        # Lower P/R ratio = better rental returns = higher score
        if 'price_to_rent_ratio' in df.columns:
            # For metros with Zillow data
            df['rental_cashflow_score'] = normalize(1 / df['price_to_rent_ratio'].clip(lower=1))
            pr_count = df['price_to_rent_ratio'].notna().sum()
            print(f"  Calculated rental cash flow scores for {pr_count} metros with Zillow data")
        else:
            # No Zillow data - use neutral score
            df['rental_cashflow_score'] = 50
            print(f"  No Zillow data - using neutral rental cash flow score")

        # Combined Boom Score (NEW weights)
        df['boom_score'] = (
            0.30 * df['tech_wage_score'] +
            0.25 * df['indian_pop_score'] +
            0.25 * df['affordability_score'] +
            0.20 * df['rental_cashflow_score']
        )

        # Sort by boom score
        df = df.sort_values('boom_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)

        print(f"  Scored and ranked {len(df)} markets")

        return df

    def run_pipeline(self, alpha: float = 0.5) -> pd.DataFrame:
        """
        Execute full analysis pipeline

        Args:
            alpha: Weight for appreciation vs cash flow (0-1).
                   0.7 = growth-focused, 0.3 = income-focused, 0.5 = balanced
        """

        print("\n" + "="*70)
        print("U.S. INVESTMENT PROPERTY ANALYSIS ENGINE v2")
        print("="*70 + "\n")

        # Step 1: Get Census data (ACS 5-Year for all metros)
        census_2018 = self.get_census_data(2018)
        time.sleep(1)
        census_2023 = self.get_census_data(2023)

        if census_2018.empty or census_2023.empty:
            raise Exception("Failed to fetch Census data")

        # Step 2: Get QCEW wage data
        qcew_data = self.fetch_all_qcew_data()

        if qcew_data.empty:
            raise Exception("Failed to fetch QCEW data")

        # Step 3: Calculate growth rates
        demographic_growth = self.calculate_demographic_growth(census_2018, census_2023)
        wage_growth = self.calculate_wage_growth(qcew_data)

        # Step 4: Merge datasets
        merged_data = self.merge_datasets(demographic_growth, wage_growth)

        # Create matching key for Census data - extract first city
        merged_data['msa_match_key'] = merged_data['msa_name'].str.split('-').str[0].str.split(',').str[0].str.lower().str.strip()

        # Step 5: Get Zillow current data (for price-to-rent)
        zillow_data = self.get_zillow_data()

        if not zillow_data.empty:
            # Dedupe Zillow data to avoid many-to-many joins
            zillow_deduped = zillow_data.drop_duplicates(subset=['msa_match_key'], keep='first')
            merged_data = merged_data.merge(
                zillow_deduped,
                on='msa_match_key',
                how='left'
            )
            print(f"  Merged Zillow current data for {merged_data['price_to_rent_ratio'].notna().sum()} metros")

        # Step 6: Get Zillow historical data (for price momentum) - NEW
        zillow_historical = self.get_zillow_historical_data()

        if not zillow_historical.empty:
            # Merge historical data - dedupe to avoid many-to-many joins
            cols_to_merge = ['msa_match_key', 'price_change_1yr', 'price_cagr_3yr', 'zhvi_current']
            zillow_hist_deduped = zillow_historical[cols_to_merge].drop_duplicates(subset=['msa_match_key'], keep='first')
            merged_data = merged_data.merge(
                zillow_hist_deduped,
                on='msa_match_key',
                how='left'
            )
            print(f"  Merged price momentum data for {merged_data['price_change_1yr'].notna().sum()} metros")

        # Step 7: Get Zillow rent historical data (for rent growth) - NEW
        rent_historical = self.get_zillow_rent_historical()

        if not rent_historical.empty:
            # Merge rent data - dedupe to avoid many-to-many joins
            cols_to_merge = ['msa_match_key', 'rent_change_1yr', 'rent_cagr_3yr']
            rent_hist_deduped = rent_historical[cols_to_merge].drop_duplicates(subset=['msa_match_key'], keep='first')
            merged_data = merged_data.merge(
                rent_hist_deduped,
                on='msa_match_key',
                how='left'
            )
            print(f"  Merged rent growth data for {merged_data['rent_change_1yr'].notna().sum()} metros")

        # Step 8: Get building permits data - NEW
        permits_data = self.get_building_permits_data()

        if not permits_data.empty and 'permits_per_1000' in permits_data.columns:
            # Need to match on MSA code
            merged_data = merged_data.merge(
                permits_data[['msa_code', 'permits_per_1000', 'building_permits']],
                on='msa_code',
                how='left'
            )
            print(f"  Merged building permits for {merged_data['permits_per_1000'].notna().sum()} metros")

        # Step 9: Get state unemployment rates from FRED
        unemployment_data = self.get_state_unemployment_fred()

        if not unemployment_data.empty:
            merged_data['state_abbrev'] = merged_data['msa_name'].str.extract(r',\s*([A-Z]{2})', expand=False)
            merged_data = merged_data.merge(
                unemployment_data[['state_abbrev', 'unemployment_rate']],
                on='state_abbrev',
                how='left'
            )
            print(f"  Merged unemployment data for {merged_data['unemployment_rate'].notna().sum()} metros")

        # Step 10: Get IRS migration data - NEW
        irs_migration = self.get_irs_migration_data(year_pair='2122')

        if not irs_migration.empty:
            # Merge on MSA code
            irs_cols = ['msa_code', 'net_migration_returns', 'avg_agi_inflow',
                       'agi_ratio', 'migration_quality']
            merged_data = merged_data.merge(
                irs_migration[irs_cols],
                on='msa_code',
                how='left'
            )
            print(f"  Merged IRS migration data for {merged_data['migration_quality'].notna().sum()} metros")

        # Step 11: Calculate legacy boom score (v1)
        merged_data = self.calculate_boom_score(merged_data)

        # Step 11: Calculate NEW v2 scores
        merged_data = self.calculate_appreciation_score(merged_data)
        merged_data = self.calculate_cashflow_score(merged_data)
        merged_data = self.calculate_combined_score(merged_data, alpha=alpha)

        # Re-rank by investment_score (v2)
        merged_data = merged_data.sort_values('investment_score', ascending=False).reset_index(drop=True)
        merged_data['rank_v2'] = range(1, len(merged_data) + 1)

        print(f"\n  Final dataset: {len(merged_data)} markets scored")

        return merged_data

    def export_results(self, df: pd.DataFrame, filename: str = 'investment_properties.csv'):
        """Export results to CSV with v2 scores"""

        # Select and order columns for output - v2 scores first
        output_cols = [
            'rank_v2',
            'msa_name',
            'state_abbrev',
            'investment_score',
            'appreciation_score',
            'cashflow_score',
            'boom_score',  # Legacy v1 score for comparison
            'median_home_value',
            'price_to_income',
            'price_change_1yr',
            'price_cagr_3yr',
            'indian_pop_2018',
            'indian_pop',
            'indian_pop_cagr',
            'wage_cagr',
            'latest_wages',
            'tech_employment',
            'median_income',
            # Component scores
            'supply_constraint_score',
            'price_momentum_score',
            'affordability_gap_score',
            'price_to_rent_score',
            'rent_growth_score'
        ]

        # Add unemployment rate if available
        if 'unemployment_rate' in df.columns:
            output_cols.append('unemployment_rate')

        # Add Zillow columns if available
        if 'price_to_rent_ratio' in df.columns:
            output_cols.extend(['zhvi_latest', 'zori_latest', 'price_to_rent_ratio'])

        # Add rent growth if available
        if 'rent_change_1yr' in df.columns:
            output_cols.append('rent_change_1yr')

        # Add IRS migration columns if available
        if 'net_migration_returns' in df.columns:
            output_cols.extend(['net_migration_returns', 'avg_agi_inflow', 'agi_ratio', 'migration_score'])

        # Filter to only existing columns
        output_cols = [c for c in output_cols if c in df.columns]

        export_df = df[output_cols].copy()

        # Column name mapping for v2
        rename_map = {
            'rank_v2': 'Rank',
            'msa_name': 'Metropolitan Area',
            'state_abbrev': 'State',
            'investment_score': 'Investment Score (v2)',
            'appreciation_score': 'Appreciation Score',
            'cashflow_score': 'Cash Flow Score',
            'boom_score': 'Boom Score (Legacy v1)',
            'median_home_value': 'Median Home Value',
            'price_to_income': 'Price-to-Income Ratio',
            'price_change_1yr': '1-Year Price Change (%)',
            'price_cagr_3yr': '3-Year Price CAGR (%)',
            'indian_pop_2018': 'Indian Pop (2018)',
            'indian_pop': 'Indian Pop (2023)',
            'indian_pop_cagr': 'Indian Pop CAGR (%)',
            'wage_cagr': 'Tech Wage CAGR (%)',
            'latest_wages': 'Latest Tech Wages ($)',
            'tech_employment': 'Tech Employment',
            'median_income': 'Median Income',
            'supply_constraint_score': 'Supply Constraint (0-100)',
            'price_momentum_score': 'Price Momentum (0-100)',
            'affordability_gap_score': 'Affordability Gap (0-100)',
            'price_to_rent_score': 'Price-to-Rent (0-100)',
            'rent_growth_score': 'Rent Growth (0-100)',
            'unemployment_rate': 'Unemployment Rate (%)',
            'zhvi_latest': 'Zillow Home Value',
            'zori_latest': 'Zillow Monthly Rent',
            'price_to_rent_ratio': 'Price-to-Rent Ratio',
            'rent_change_1yr': '1-Year Rent Change (%)',
            'net_migration_returns': 'Net Migration (Households)',
            'avg_agi_inflow': 'Avg AGI Inflow ($K)',
            'agi_ratio': 'AGI Ratio (In/Out)',
            'migration_score': 'Migration Score (0-100)'
        }

        export_df = export_df.rename(columns=rename_map)

        # Export to CSV
        export_df.to_csv(filename, index=False)

        print(f"\n{'='*70}")
        print(f"Results exported to: {filename}")
        print(f"{'='*70}\n")

        # Print top 10 markets with v2 scores
        print("TOP 10 INVESTMENT MARKETS (v2 Model):\n")
        display_cols = ['Rank', 'Metropolitan Area', 'Investment Score (v2)',
                        'Appreciation Score', 'Cash Flow Score']
        display_cols = [c for c in display_cols if c in export_df.columns]
        print(export_df[display_cols].head(10).to_string(index=False))
        print()


def main():
    """Run the investment property analysis pipeline"""

    parser = argparse.ArgumentParser(
        description='U.S. Investment Property Analysis Engine v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python tech_hub_pipeline.py                    # Balanced (alpha=0.5)
  python tech_hub_pipeline.py --alpha 0.7        # Growth-focused
  python tech_hub_pipeline.py --alpha 0.3        # Cash flow-focused
  python tech_hub_pipeline.py --no-cache         # Force fresh data fetch

Alpha parameter:
  0.0 = 100% cash flow focus (rental yield)
  0.5 = Balanced (default)
  1.0 = 100% appreciation focus (price growth)
        '''
    )

    parser.add_argument('--no-cache', action='store_true',
                       help='Force fresh data fetch, ignore cache')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Directory for caching data (default: cache)')
    parser.add_argument('--output', type=str, default='investment_properties.csv',
                       help='Output CSV filename (default: investment_properties.csv)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Appreciation vs cash flow weight (0-1, default: 0.5)')

    args = parser.parse_args()

    # Note: Census API key not required for basic queries
    # If rate limited, sign up for a key at https://api.census.gov/data/key_signup.html
    CENSUS_API_KEY = None

    # FRED API key (free, instant signup at https://fred.stlouisfed.org/docs/api/api_key.html)
    FRED_API_KEY = "95f00ff15aa5e67977abc35c7094a0b1"

    # Create analyzer with cache settings
    analyzer = TechHubAnalyzer(
        CENSUS_API_KEY,
        FRED_API_KEY,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache
    )

    try:
        print(f"\nCache mode: {'DISABLED (fetching fresh data)' if args.no_cache else f'ENABLED (using {args.cache_dir}/)'}")
        print(f"Output file: {args.output}")
        print(f"Alpha (appreciation weight): {args.alpha}")
        if args.alpha >= 0.7:
            print("  -> Growth-focused strategy")
        elif args.alpha <= 0.3:
            print("  -> Cash flow-focused strategy")
        else:
            print("  -> Balanced strategy")
        print()

        results = analyzer.run_pipeline(alpha=args.alpha)
        analyzer.export_results(results, filename=args.output)

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
