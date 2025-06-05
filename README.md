# PV Plant Performance Analysis

## Overview
This project processes and visualizes solar photovoltaic (PV) plant performance data using Performance Ratio (PR) and Global Horizontal Irradiance (GHI) metrics to create comprehensive performance analysis and visualization.

## Dataset Description

### Parameters
- **PR (Performance Ratio)**: Tracks daily performance of the PV plant. Higher values indicate better performance with no issues.
- **GHI (Global Horizontal Irradiance)**: Measures total daily irradiation. Higher values indicate sunnier conditions.

### Data Structure
```
PR/
├── 2023-01/
│   ├── 2023-01-01_PR.csv
│   ├── 2023-01-06_PR.csv
│   └── ...
└── 2023-XX/

GHI/
├── 2023-01/
│   ├── 2023-01-01_GHI.csv
│   ├── 2023-01-06_GHI.csv
│   └── ...
└── 2023-XX/
```

## Objectives

### 1. Data Processing
- Consolidate all PR and GHI data into a single CSV file
- Output format: 3 columns (Date, GHI, PR)
- Expected output: 982 rows of data

### 2. Data Visualization
Create a comprehensive performance graph featuring:
- **Scatter Plot**: Daily PR values color-coded by GHI levels
- **30-Day Moving Average**: Red line showing PR trends
- **Budget Line**: Dynamic target performance line
- **Performance Statistics**: Summary metrics for various time periods

## Technical Requirements

### Data Processing Function
- Single function to preprocess all data
- Clean, readable, and well-organized code
- Handles multiple directories and file formats

### Visualization Function
- Single function to generate the complete graph
- Implements all required visual elements
- Dynamic calculations for budget lines and statistics

## Visualization Specifications

### Color Coding (GHI-based)
- **Navy Blue**: GHI < 2
- **Light Blue**: GHI 2-4
- **Orange**: GHI 4-6
- **Brown**: GHI > 6

### Budget Line Calculation
- **Initial Value**: 73.9 (July 2019 - June 2020)
- **Annual Degradation**: 0.8% per year
- **Year 2**: 73.1 (July 2020 - June 2021)
- **Year 3**: 72.3 (July 2021 - June 2022)
- Calculated dynamically, not hardcoded

### Graph Elements
1. **Red Line**: 30-day moving average of PR
2. **Scatter Points**: Daily PR values (position) with GHI color coding
3. **Green Budget Line**: Dynamic annual performance targets
4. **Statistics Panel**: Average PR for 7, 30, 60+ day periods
5. **Legend**: Color coding reference for GHI ranges

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib plotly seaborn datetime
```

### Basic Usage
```python
# Process data
processed_data = preprocess_pv_data(data_directory)

# Generate visualization
generate_pr_visualization(processed_data)
```

### Bonus Feature - Date Range Analysis
```python
# Generate visualization for specific date range
generate_pr_visualization(
    processed_data,
    start_date="2024-01-01",
    end_date="2024-06-30"
)
```

## Project Structure
```
pv-analysis/
├── data/
│   ├── PR/
│   └── GHI/
├── output/
│   ├── consolidated_data.csv
│   └── pr_performance_graph.png
├── src/
│   ├── data_processor.py
│   ├── visualization.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Key Features

### Data Processing
- Automated file discovery and processing
- Date parsing and validation
- Data quality checks and cleaning
- Efficient memory management for large datasets

### Visualization
- Interactive performance dashboard
- Multi-layered analysis (daily, monthly, yearly trends)
- Dynamic budget line calculations
- Color-coded performance indicators
- Statistical summaries

### Analysis Capabilities
- Performance trend identification
- Seasonal pattern recognition
- Budget vs. actual performance comparison
- Outlier detection and analysis

## Expected Outputs

1. **consolidated_data.csv**: Single file containing all processed data
2. **Performance Graph**: Comprehensive visualization with all specified elements
3. **Python Scripts**: Clean, documented code for processing and visualization

## Performance Metrics Tracked

- Daily PR values with trend analysis
- 30-day moving averages
- Short-term performance (7-day average)
- Medium-term performance (30-day average)
- Long-term performance (60+ day averages)
- Budget compliance tracking
- Weather impact analysis (via GHI correlation)

## Notes

- Data trends may vary from reference graphs due to data modifications
- All calculations are performed dynamically
- Code is organized for maintainability and readability
- Error handling implemented for robust data processing

## Future Enhancements

- Real-time data processing capability
- Advanced statistical analysis
- Machine learning-based performance prediction
- Interactive web dashboard
- Automated report generation
