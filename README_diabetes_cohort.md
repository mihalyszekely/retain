# Type 2 Diabetes Cohort Analysis

This script identifies patients with Type 2 Diabetes Mellitus (T2DM) using ICD-9 and ICD-10 diagnostic codes and counts unique patients and hospitalizations.

## ICD Codes Used for Type 2 Diabetes

### ICD-9 Codes (250.x0 and 250.x2)
- **Pattern**: Codes starting with "250", exactly 5 characters long, ending with "0" or "2"
- **Examples**: 25000, 25002, 25010, 25012, etc.
- **Rationale**: In ICD-9, the 5th digit indicates:
  - 0 = Uncomplicated diabetes
  - 2 = Diabetes with complications

### ICD-10 Codes (E11.x)
- **Pattern**: All codes starting with "E11"
- **Examples**: E1100, E1101, E1102, E1110, E1111, etc.
- **Rationale**: E11 specifically codes for Type 2 Diabetes Mellitus

## Features

- **Memory Efficient**: Processes large CSV files in chunks
- **Dual Code Support**: Handles both ICD-9 and ICD-10 codes
- **Accurate Filtering**: Uses precise pattern matching for code identification
- **Progress Tracking**: Shows real-time progress and counts
- **Comprehensive Results**: Counts unique patients and hospitalizations

## Usage

```bash
python diabetes_cohort_analysis.py diagnoses_icd_MIMIC-IV.csv
```

## Requirements

- Python 3.6+
- pandas library

## Output

The script provides:
- Total unique patients with T2DM diagnoses
- Total unique hospitalizations with T2DM diagnoses
- Breakdown by ICD-9 vs ICD-10 codes
- Progress information during processing
- Results saved to `t2dm_cohort_results.txt`

## Example Output

```
============================================================
Type 2 Diabetes Cohort Analysis
============================================================
Target file: diagnoses_icd_MIMIC-IV.csv
============================================================
Processing diagnoses_icd_MIMIC-IV.csv in chunks of 100,000 rows...
Looking for Type 2 Diabetes codes:
  ICD-9 codes: 100 codes (250.x0 and 250.x2)
  ICD-10 codes: 100 codes (E11.x)
============================================================
Chunk 1: Found 45 ICD-9 and 23 ICD-10 T2DM diagnoses
  Total unique patients so far: 67
  Total unique hospitalizations so far: 89
...
============================================================
FINAL RESULTS
============================================================
Total unique patients with T2DM: 15,234
Total unique hospitalizations with T2DM: 18,567
Total T2DM diagnoses (ICD-9): 12,345
Total T2DM diagnoses (ICD-10): 8,901
Total T2DM diagnoses: 21,246
============================================================
```

## Code Selection Rationale

### ICD-9 Codes (250.x0 and 250.x2)
- **250.x0**: Uncomplicated Type 2 Diabetes
- **250.x2**: Type 2 Diabetes with complications
- These codes specifically identify Type 2 Diabetes (not Type 1)

### ICD-10 Codes (E11.x)
- **E11**: Type 2 Diabetes Mellitus
- More specific than ICD-9, with detailed subcategories
- All E11 codes represent Type 2 Diabetes

## File Format

The script expects the `diagnoses_icd.csv` file to have these columns:
- `subject_id`: Unique patient identifier
- `hadm_id`: Unique hospitalization identifier  
- `icd_code`: ICD diagnostic code
- `icd_version`: ICD version (9 or 10)

## Performance

- Default chunk size: 100,000 rows
- Progress indicators every 10 chunks
- Memory-efficient processing for large files 