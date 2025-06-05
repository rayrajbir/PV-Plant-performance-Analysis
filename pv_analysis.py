import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path
import argparse

class SolarPVDataProcessor:
    def __init__(self, data_directory='data'):
        """
        Initialize the processor with the data directory path.
        
        Args:
            data_directory (str): Path to the directory containing PR and GHI folders
        """
        self.data_directory = data_directory
        self.processed_data = None
        
    def preprocess_data(self):
        print("Starting data preprocessing...")
        
        # Initialize dictionaries to store data
        pr_data = {}
        ghi_data = {}
        
        # Process PR data
        pr_path = os.path.join(self.data_directory, 'PR')
        print(f"Processing PR data from: {pr_path}")
        
        if os.path.exists(pr_path):
            # Get all CSV files in PR directory and subdirectories
            pr_files = glob.glob(os.path.join(pr_path, '**', '*.csv'), recursive=True)
            print(f"Found {len(pr_files)} PR files")
            
            for file_path in pr_files:
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Check if the CSV has Date and PR columns
                    if 'Date' in df.columns and 'PR' in df.columns:
                        # Process each row in the CSV
                        for _, row in df.iterrows():
                            date_str = str(row['Date'])
                            pr_value = pd.to_numeric(row['PR'], errors='coerce')
                            pr_data[date_str] = pr_value
                    else:
                        # Fallback: extract date from filename and use first numeric column
                        filename = os.path.basename(file_path)
                        date_str = filename.replace('_PR.csv', '').replace('.csv', '')
                        
                        # Find first numeric column
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            pr_value = df[numeric_cols[0]].iloc[0] if len(df) > 0 else np.nan
                            pr_data[date_str] = pr_value
                        
                except Exception as e:
                    print(f"Error processing PR file {file_path}: {e}")
                    continue
        
        # Process GHI data
        ghi_path = os.path.join(self.data_directory, 'GHI')
        print(f"Processing GHI data from: {ghi_path}")
        
        if os.path.exists(ghi_path):
            # Get all CSV files in GHI directory and subdirectories
            ghi_files = glob.glob(os.path.join(ghi_path, '**', '*.csv'), recursive=True)
            print(f"Found {len(ghi_files)} GHI files")
            
            for file_path in ghi_files:
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Check if the CSV has Date and GHI columns
                    if 'Date' in df.columns and 'GHI' in df.columns:
                        # Process each row in the CSV
                        for _, row in df.iterrows():
                            date_str = str(row['Date'])
                            ghi_value = pd.to_numeric(row['GHI'], errors='coerce')
                            ghi_data[date_str] = ghi_value
                    else:
                        # Fallback: extract date from filename and use first numeric column
                        filename = os.path.basename(file_path)
                        date_str = filename.replace('_GHI.csv', '').replace('.csv', '')
                        
                        # Find first numeric column
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            ghi_value = df[numeric_cols[0]].iloc[0] if len(df) > 0 else np.nan
                            ghi_data[date_str] = ghi_value
                        
                except Exception as e:
                    print(f"Error processing GHI file {file_path}: {e}")
                    continue
        
        # Combine PR and GHI data
        all_dates = set(list(pr_data.keys()) + list(ghi_data.keys()))
        all_data = []
        
        for date_str in all_dates:
            try:
                # Parse date
                date_obj = pd.to_datetime(date_str)
                
                # Get PR and GHI values
                pr_val = pr_data.get(date_str, np.nan)
                ghi_val = ghi_data.get(date_str, np.nan)
                
                all_data.append({
                    'Date': date_obj,
                    'GHI': ghi_val,
                    'PR': pr_val
                })
                
            except Exception as e:
                print(f"Error processing date {date_str}: {e}")
                continue
        
        # Create DataFrame
        self.processed_data = pd.DataFrame(all_data)
        self.processed_data = self.processed_data.sort_values('Date').reset_index(drop=True)
        
        # Ensure PR and GHI columns are numeric
        self.processed_data['PR'] = pd.to_numeric(self.processed_data['PR'], errors='coerce')
        self.processed_data['GHI'] = pd.to_numeric(self.processed_data['GHI'], errors='coerce')
        
        print(f"Data preprocessing complete. Total rows: {len(self.processed_data)}")
        print(f"PR data type: {self.processed_data['PR'].dtype}")
        print(f"GHI data type: {self.processed_data['GHI'].dtype}")
        print(f"Sample of processed data:")
        print(self.processed_data.head())
        
        return self.processed_data
    
    def save_processed_data(self, filename='processed_solar_data.csv'):
        if self.processed_data is not None:
            self.processed_data.to_csv(filename, index=False)
            print(f"Processed data saved to {filename}")
        else:
            print("No processed data to save. Run preprocess_data() first.")
    
    def calculate_budget_line(self, start_date, end_date):
        budget_values = {}
        
        # Define budget parameters
        initial_budget = 73.9
        annual_reduction = 0.008  # 0.8%
        
        # Calculate budget for each date
        current_date = start_date
        while current_date <= end_date:
            # Budget year runs from July to June
            if current_date.month >= 7:
                budget_year = current_date.year - 2019  # First year starts July 2019
            else:
                budget_year = current_date.year - 2020  # Second half of budget year
            
            # Calculate budget value for this year
            budget_value = initial_budget * (1 - annual_reduction) ** budget_year
            budget_values[current_date] = budget_value
            
            current_date += timedelta(days=1)
        
        return budget_values
    
    def get_color_for_ghi(self, ghi_value):
        if pd.isna(ghi_value):
            return 'gray'
        elif ghi_value < 2:
            return 'navy'
        elif 2 <= ghi_value < 4:
            return 'lightblue'
        elif 4 <= ghi_value < 6:
            return 'orange'
        else:
            return 'brown'
    
    def generate_visualization(self, start_date=None, end_date=None, figsize=(15, 10)):

        if self.processed_data is None:
            print("No processed data available. Run preprocess_data() first.")
            return
        
        # Filter data by date range if specified
        df = self.processed_data.copy()
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        if len(df) == 0:
            print("No data available for the specified date range.")
            return
        
        # Create the main plot
        fig, ax = plt.subplots(figsize=figsize)
        
        df['PR_30MA'] = df['PR'].rolling(window=30, center=True).mean()
        
        ax.plot(df['Date'], df['PR_30MA'], color='red', linewidth=2, 
                label='30-day Moving Average', alpha=0.8)
        
        # Calculate budget line
        budget_dict = self.calculate_budget_line(df['Date'].min(), df['Date'].max())
        budget_dates = list(budget_dict.keys())
        budget_values = list(budget_dict.values())
        
        ax.plot(budget_dates, budget_values, color='darkgreen', linewidth=2, 
                label='Target Budget PR', linestyle='--')
        
        # Create scatter plot with GHI-based coloring
        colors = [self.get_color_for_ghi(ghi) for ghi in df['GHI']]
        scatter = ax.scatter(df['Date'], df['PR'], c=colors, alpha=0.6, s=20)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Performance Ratio (%)', fontsize=12)
        ax.set_title('PV Plant Performance Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend for GHI colors
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', 
                      markersize=8, label='GHI < 2'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=8, label='GHI 2-4'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=8, label='GHI 4-6'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', 
                      markersize=8, label='GHI > 6'),
        ]
        
        # Create two legends
        legend1 = ax.legend(handles=legend_elements, title='Daily Irradiation (GHI)', 
                           loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax.add_artist(legend1)
        
        ax.legend(['30-day Moving Average', 'Target Budget PR'], 
                 loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # Add statistics text box
        self.add_statistics_box(ax, df)
        
        # Add points above budget calculation
        self.add_budget_analysis(ax, df, budget_dict)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('pr_performance_graph.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'pr_performance_graph.png'")
        
        plt.show()
    
    def add_statistics_box(self, ax, df):
        # Calculate statistics
        stats_text = []
        
        # Last 7, 30, 60, 90 days averages
        for days in [7, 30, 60, 90]:
            if len(df) >= days:
                avg_pr = df['PR'].tail(days).mean()
                stats_text.append(f"Last {days} days: {avg_pr:.1f}%")
        
        # Add overall statistics
        stats_text.append(f"Overall Avg: {df['PR'].mean():.1f}%")
        stats_text.append(f"Max PR: {df['PR'].max():.1f}%")
        stats_text.append(f"Min PR: {df['PR'].min():.1f}%")
        
        # Create text box
        textstr = '\n'.join(stats_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    def add_budget_analysis(self, ax, df, budget_dict):
        # Calculate points above budget
        df_with_budget = df.copy()
        df_with_budget['Budget'] = df_with_budget['Date'].map(budget_dict)
        df_with_budget['Above_Budget'] = df_with_budget['PR'] > df_with_budget['Budget']
        
        points_above_budget = df_with_budget['Above_Budget'].sum()
        total_points = len(df_with_budget)
        percentage_above = (points_above_budget / total_points) * 100
        
        # Add text annotation
        budget_text = f"Points above budget: {points_above_budget}/{total_points} ({percentage_above:.1f}%)"
        ax.text(0.02, 0.02, budget_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))


def main():
    parser = argparse.ArgumentParser(description='Solar PV Data Analysis')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing PR and GHI folders')
    
    args = parser.parse_args()
    
    # Initialize the processor
    processor = SolarPVDataProcessor(args.data_dir)
    
    # Process the data
    processed_df = processor.preprocess_data()
    
    # Save processed data to CSV
    processor.save_processed_data('processed_solar_data.csv')
    
    # Generate visualization
    processor.generate_visualization(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    print("\nAnalysis complete!")
    print(f"- Processed data saved to: processed_solar_data.csv")
    print(f"- Visualization saved to: pr_performance_graph.png")
    print(f"- Total data points: {len(processed_df)}")


if __name__ == "__main__":
    main()


# Example usage for testing without command line arguments:
import sys
if __name__ == "__main__" and len(sys.argv) == 1:
    # Create sample data for demonstration
    processor = SolarPVDataProcessor('data')
    
    # If you have the actual data, uncomment these lines:
    # processed_df = processor.preprocess_data()
    # processor.save_processed_data('processed_solar_data.csv')
    # processor.generate_visualization()
    
    print("To use with actual data:")
    print("Place your PR and GHI folders in a 'data' directory")
    print("1. Run: python script.py")
    print("2. For date range filtering: python script.py --start_date 2024-01-01 --end_date 2024-06-30")