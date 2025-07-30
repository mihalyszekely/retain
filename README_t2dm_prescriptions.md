# Type 2 Diabetes Prescriptions Analysis

This script analyzes prescriptions for patients with Type 2 Diabetes Mellitus (T2DM) to identify the most common medications prescribed to this cohort.

## Overview

The script performs a two-step analysis:

1. **Identifies T2DM patients** from the diagnoses file using ICD-9 and ICD-10 codes:
   - ICD-9: Codes starting with "250" that are 5 characters long and end with "0" or "2"
   - ICD-10: Codes starting with "E11"

2. **Analyzes prescriptions** for the identified T2DM patients to find the most commonly prescribed medications.

## Usage

```bash
python t2dm_prescriptions_analysis.py <diagnoses_file> <prescriptions_file> [options]
```

### Arguments

- `diagnoses_file`: Path to the diagnoses_icd CSV file
- `prescriptions_file`: Path to the prescriptions CSV file

### Options

- `--chunk-size N`: Number of rows to process at once (default: 1000000)
- `--top-n N`: Number of top medications to display (default: 50)
- `--plot`: Create a bar plot of top 20 medications
- `--plot-file FILE`: Output file for the plot (default: t2dm_medications_plot.png)

### Example

```bash
# Basic analysis
python t2dm_prescriptions_analysis.py diagnoses_icd.csv prescriptions.csv --top-n 100

# With plotting
python t2dm_prescriptions_analysis.py diagnoses_icd.csv prescriptions.csv --plot

# With custom plot file
python t2dm_prescriptions_analysis.py diagnoses_icd.csv prescriptions.csv --plot --plot-file my_medications.png
```

## Output

The script provides:

1. **Console output** showing:
   - Progress during processing
   - Summary statistics
   - Top N most common medications with counts

2. **Detailed results file** (`t2dm_prescriptions_results.txt`) containing:
   - Summary statistics
   - Complete list of all medications prescribed to T2DM patients with counts

3. **Bar plot** (if `--plot` option is used):
   - Visual representation of top 20 most common medications
   - Saved as PNG file with high resolution
   - Includes count labels on bars

## File Requirements

### Diagnoses File
Must contain columns:
- `subject_id`: Patient identifier
- `icd_version`: ICD version (9 or 10)
- `icd_code`: ICD diagnosis code

### Prescriptions File
Must contain columns:
- `subject_id`: Patient identifier
- `drug`: Medication name (will be used for counting)

## Dependencies

- pandas: For data processing
- matplotlib: For plotting (required if using `--plot` option)
- seaborn: For enhanced plotting styles (required if using `--plot` option)

## Performance

- Processes large files in chunks to manage memory usage
- Uses efficient pandas operations for filtering and counting
- Provides progress updates during processing

## Example Output

```
================================================================================
Type 2 Diabetes Prescriptions Analysis
================================================================================
Diagnoses file: diagnoses_icd.csv
Prescriptions file: prescriptions.csv
Chunk size: 1,000,000
================================================================================
Identifying T2DM patients from diagnoses_icd.csv...
Identified 15,234 unique T2DM patients
Analyzing prescriptions for T2DM patients from prescriptions.csv...
Processing in chunks of 1,000,000 rows...
Processed 2,500,000 total prescriptions
Found 45,678 prescriptions for T2DM patients

================================================================================
ANALYSIS RESULTS
================================================================================
Total T2DM patients: 15,234
Total prescriptions processed: 2,500,000
Prescriptions for T2DM patients: 45,678
Unique medications prescribed: 1,234
================================================================================

Top 50 Most Common Medications for T2DM Patients:
--------------------------------------------------------------------------------
Rank  Count      Medication
--------------------------------------------------------------------------------
1     2,345      insulin regular
2     1,890      metformin
3     1,567      lantus
4     1,234      novolog
5     987         glipizide
...
```

## Notes

- Drug names are converted to lowercase and stripped of whitespace for consistent counting
- The script handles missing data gracefully
- Results are sorted by frequency (most common first)
- Memory usage is optimized for large datasets 