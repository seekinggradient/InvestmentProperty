#!/usr/bin/env python3
"""
Analyze Investment Opportunities from Tech Hub Rankings

This script provides various analysis views of the ranked markets.
Run this anytime after generating emerging_tech_hubs.csv

Usage:
    python analyze_results.py                    # Show all analyses
    python analyze_results.py --state TX        # Filter by state
    python analyze_results.py --top 20          # Show top N markets
"""

import pandas as pd
import argparse
from pathlib import Path


def load_data(filename='emerging_tech_hubs.csv'):
    """Load results CSV, skipping header rows"""
    if not Path(filename).exists():
        print(f"Error: {filename} not found!")
        print("Run: python tech_hub_pipeline.py first")
        return None

    # Skip rows 1-2 (technical description + interpretation guidance)
    df = pd.read_csv(filename, skiprows=[1, 2])
    return df


def show_top_markets(df, n=10):
    """Display top N markets"""
    print("="*120)
    print(f"TOP {n} EMERGING TECH HUB INVESTMENT OPPORTUNITIES")
    print("="*120)
    print()

    cols = [
        'Rank', 'Metropolitan Area', 'State', 'Boom Score',
        'Median Home Value', 'Price-to-Rent Ratio',
        'State Unemployment Rate (%)', 'Tech Wage Growth (CAGR %)'
    ]

    display = df[cols].head(n).copy()

    # Format
    display['Metropolitan Area'] = display['Metropolitan Area'].str.split(',').str[0]
    display['Boom Score'] = display['Boom Score'].apply(lambda x: f"{x:.1f}")
    display['Median Home Value'] = display['Median Home Value'].apply(lambda x: f"${int(x):,}")
    display['Price-to-Rent Ratio'] = display['Price-to-Rent Ratio'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    display['State Unemployment Rate (%)'] = display['State Unemployment Rate (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    display['Tech Wage Growth (CAGR %)'] = display['Tech Wage Growth (CAGR %)'].apply(lambda x: f"{x:.1f}%")

    print(display.to_string(index=False))
    print()


def show_state_analysis(df, state=None):
    """Show best markets by state"""
    print("="*120)
    print(f"BEST MARKETS BY STATE{f' (Filtered: {state})' if state else ''}")
    print("="*120)
    print()

    if state:
        state_df = df[df['State'] == state].head(10)
        if len(state_df) == 0:
            print(f"No markets found for state: {state}")
            return

        cols = ['Rank', 'Metropolitan Area', 'Boom Score', 'Median Home Value', 'Price-to-Rent Ratio']
        display = state_df[cols].copy()

        display['Metropolitan Area'] = display['Metropolitan Area'].str.split(',').str[0]
        display['Boom Score'] = display['Boom Score'].apply(lambda x: f"{x:.1f}")
        display['Median Home Value'] = display['Median Home Value'].apply(lambda x: f"${int(x):,}")
        display['Price-to-Rent Ratio'] = display['Price-to-Rent Ratio'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

        print(display.to_string(index=False))
        print()
    else:
        # Show top market per state
        state_counts = df.groupby('State').size().sort_values(ascending=False)

        print(f"States with most opportunities (total markets in top 101):")
        for state, count in state_counts.head(15).items():
            top_market = df[df['State'] == state].iloc[0]
            print(f"  {state}: {count:2d} markets - Best: {top_market['Metropolitan Area'].split(',')[0]:30s} (Rank #{int(top_market['Rank'])})")
        print()


def show_cash_flow_analysis(df, n=10):
    """Show best cash flow markets (low P/R ratio)"""
    print("="*120)
    print(f"BEST CASH FLOW MARKETS (Lowest Price-to-Rent Ratio)")
    print("="*120)
    print()

    # Filter out markets without P/R data and sort by P/R
    cash_flow = df[df['Price-to-Rent Ratio'].notna()].copy()
    cash_flow = cash_flow.sort_values('Price-to-Rent Ratio').head(n)

    cols = ['Rank', 'Metropolitan Area', 'State', 'Price-to-Rent Ratio',
            'Median Home Value', 'Zillow Rent (Latest Monthly)', 'Boom Score']

    display = cash_flow[cols].copy()

    display['Metropolitan Area'] = display['Metropolitan Area'].str.split(',').str[0]
    display['Price-to-Rent Ratio'] = display['Price-to-Rent Ratio'].apply(lambda x: f"{x:.1f}")
    display['Median Home Value'] = display['Median Home Value'].apply(lambda x: f"${int(x):,}")
    display['Zillow Rent (Latest Monthly)'] = display['Zillow Rent (Latest Monthly)'].apply(lambda x: f"${int(x):,}" if pd.notna(x) else "N/A")
    display['Boom Score'] = display['Boom Score'].apply(lambda x: f"{x:.1f}")

    print(display.to_string(index=False))
    print()
    print("📊 P/R Ratio Guide: <13 = Excellent | 13-15 = Good | 15-20 = Fair | >20 = Rent don't buy")
    print()


def show_growth_markets(df, n=10):
    """Show fastest growing tech markets"""
    print("="*120)
    print(f"FASTEST GROWING TECH MARKETS (Tech Wage CAGR)")
    print("="*120)
    print()

    growth = df.sort_values('Tech Wage Growth (CAGR %)', ascending=False).head(n)

    cols = ['Rank', 'Metropolitan Area', 'State', 'Tech Wage Growth (CAGR %)',
            'Indian Pop Growth (CAGR %)', 'Median Home Value', 'Boom Score']

    display = growth[cols].copy()

    display['Metropolitan Area'] = display['Metropolitan Area'].str.split(',').str[0]
    display['Tech Wage Growth (CAGR %)'] = display['Tech Wage Growth (CAGR %)'].apply(lambda x: f"{x:.1f}%")
    display['Indian Pop Growth (CAGR %)'] = display['Indian Pop Growth (CAGR %)'].apply(lambda x: f"{x:.1f}%")
    display['Median Home Value'] = display['Median Home Value'].apply(lambda x: f"${int(x):,}")
    display['Boom Score'] = display['Boom Score'].apply(lambda x: f"{x:.1f}")

    print(display.to_string(index=False))
    print()


def show_affordable_markets(df, n=10):
    """Show most affordable markets"""
    print("="*120)
    print(f"MOST AFFORDABLE MARKETS (Lowest Median Home Value)")
    print("="*120)
    print()

    affordable = df.sort_values('Median Home Value').head(n)

    cols = ['Rank', 'Metropolitan Area', 'State', 'Median Home Value',
            'Price-to-Rent Ratio', 'Boom Score', 'Tech Wage Growth (CAGR %)']

    display = affordable[cols].copy()

    display['Metropolitan Area'] = display['Metropolitan Area'].str.split(',').str[0]
    display['Median Home Value'] = display['Median Home Value'].apply(lambda x: f"${int(x):,}")
    display['Price-to-Rent Ratio'] = display['Price-to-Rent Ratio'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    display['Boom Score'] = display['Boom Score'].apply(lambda x: f"{x:.1f}")
    display['Tech Wage Growth (CAGR %)'] = display['Tech Wage Growth (CAGR %)'].apply(lambda x: f"{x:.1f}%")

    print(display.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze emerging tech hub investment opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--file', type=str, default='emerging_tech_hubs.csv',
                       help='Input CSV file (default: emerging_tech_hubs.csv)')
    parser.add_argument('--state', type=str,
                       help='Filter by state (e.g., TX, CA, PA)')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of markets to show (default: 10)')
    parser.add_argument('--analysis', type=str, choices=['all', 'top', 'states', 'cashflow', 'growth', 'affordable'],
                       default='all', help='Type of analysis to run (default: all)')

    args = parser.parse_args()

    # Load data
    df = load_data(args.file)
    if df is None:
        return

    print(f"\n📊 Loaded {len(df)} markets from {args.file}\n")

    # Run requested analyses
    if args.analysis in ['all', 'top']:
        show_top_markets(df, args.top)

    if args.analysis in ['all', 'states']:
        show_state_analysis(df, args.state)

    if args.analysis in ['all', 'cashflow']:
        show_cash_flow_analysis(df, args.top)

    if args.analysis in ['all', 'growth']:
        show_growth_markets(df, args.top)

    if args.analysis in ['all', 'affordable']:
        show_affordable_markets(df, args.top)


if __name__ == "__main__":
    main()
