# T2DM HbA1c Analysis with Plotting

This document describes the plotting functionality added to the T2DM HbA1c analysis script.

## Overview

The `t2dm_hba1c_analysis.py` script now includes optional plotting capabilities to visualize the distribution of HbA1c measurements among Type 2 Diabetes Mellitus (T2DM) patients. This helps answer the question: **"Of the T2D patients, what is the distribution of HbA1c measurements per patient? How many have at least two measurements, making them eligible for our trend-prediction task?"**

## Usage

### Basic Analysis (No Plotting)
```bash
python t2dm_hba1c_analysis.py diagnoses_icd.csv labevents.csv
```

### Analysis with Basic Plotting
```bash
python t2dm_hba1c_analysis.py diagnoses_icd.csv labevents.csv --plot
```

### Analysis with Detailed Plotting
```bash
python t2dm_hba1c_analysis.py diagnoses_icd.csv labevents.csv --plot --detailed
```

## Command Line Arguments

- `diagnoses_file`: Path to the diagnoses_icd.csv file
- `labevents_file`: Path to the labevents.csv file
- `--plot`: Generate plots of the HbA1c measurement distribution
- `--detailed`: Generate detailed plots with additional visualizations (requires --plot)

## Generated Plots

### Basic Distribution Plot (`hba1c_measurement_distribution.png`)

This plot contains two visualizations:

1. **Bar Chart**: Shows the distribution of HbA1c measurements per patient
   - Blue bars: Patients with 1 measurement
   - Red bars: Patients with 2+ measurements (eligible for trend prediction)
   - Each bar shows the exact number of patients

2. **Pie Chart**: Shows the breakdown of T2DM patients
   - Eligible patients (≥2 measurements)
   - Patients with 1 measurement
   - Patients with no HbA1c measurements

### Detailed Distribution Plot (`hba1c_detailed_distribution.png`)

This comprehensive plot contains four visualizations:

1. **Cumulative Distribution**: Shows how many patients have at least N measurements
   - Red dashed line indicates the eligibility threshold (≥2 measurements)

2. **Log-Scale Distribution**: Shows the measurement distribution on a logarithmic scale
   - Useful for seeing patterns in the tail of the distribution

3. **Statistics Summary**: Text box with key statistics
   - Total T2DM patients
   - Patients with HbA1c measurements
   - Patients with ≥2 measurements
   - Total HbA1c measurements
   - Average measurements per patient

4. **Top Measurement Counts**: Bar chart of the 10 most common measurement counts
   - Shows which measurement counts are most frequent

## Key Insights from Plots

The plots help answer several important questions:

1. **Eligibility for Trend Prediction**: How many T2DM patients have sufficient HbA1c measurements (≥2) for trend analysis?

2. **Measurement Distribution**: What is the typical number of HbA1c measurements per patient?

3. **Data Quality**: What percentage of T2DM patients have any HbA1c measurements at all?

4. **Sample Size**: How many patients are available for different types of analysis?

## Example Output

When running with `--plot`, you'll see output like:

```
Generating plots...
Distribution plot saved to: hba1c_measurement_distribution.png
Detailed distribution plot saved to: hba1c_detailed_distribution.png
```

## Dependencies

The plotting functionality requires:
- matplotlib (≥3.8.0)
- seaborn (≥0.13.0)
- numpy (≥2.0.0)

These are already included in the `requirements.txt` file.

## Testing

You can test the plotting functionality with sample data:

```bash
python test_plotting.py
```

This will generate test plots to verify the functionality works correctly.

## File Outputs

The script generates several output files:

1. `t2dm_hba1c_results.txt`: Detailed text results
2. `hba1c_measurement_distribution.png`: Basic distribution plots (with --plot)
3. `hba1c_detailed_distribution.png`: Detailed analysis plots (with --plot --detailed)

## Interpretation

- **Eligible Patients**: Patients with ≥2 HbA1c measurements are considered eligible for trend prediction tasks
- **Measurement Counts**: The distribution shows how many patients have 1, 2, 3, etc. measurements
- **Percentages**: Key metrics include what percentage of T2DM patients have HbA1c data and what percentage are eligible for trend analysis

This plotting functionality makes it easy to assess the quality and quantity of HbA1c data available for T2DM patients and determine the feasibility of trend prediction analysis. 