# HbA1c Measurement Counter

This script counts the total number of HbA1c measurements in the MIMIC-III labevents table.

## Overview

The script processes the large `labevents.csv` file in chunks to efficiently count all HbA1c measurements. HbA1c has `itemid` 50852 in the `d_labitems` table.

## Features

- **Memory Efficient**: Processes large CSV files in chunks (default: 100,000 rows per chunk)
- **Progress Tracking**: Shows progress as it processes chunks
- **Error Handling**: Robust error handling for file not found and processing errors
- **Detailed Output**: Shows count per chunk and running total

## Usage

```bash
python count_hba1c.py labevents.csv
```

## Requirements

- Python 3.6+
- pandas library

Install pandas if not already installed:
```bash
pip install pandas
```

## Output

The script will display:
- Progress information for each chunk processed
- Running total of HbA1c measurements found
- Final count of all HbA1c measurements

## Example Output

```
============================================================
HbA1c Measurement Counter
============================================================
Target file: labevents.csv
Looking for itemid: 50852 (HbA1c)
============================================================
Processing labevents.csv in chunks of 100,000 rows...
Chunk 1: Found 45 HbA1c measurements (Total so far: 45)
Chunk 2: Found 52 HbA1c measurements (Total so far: 97)
...
============================================================
FINAL RESULT: 1,234 total HbA1c measurements found
============================================================
```

## File Format

The script expects the `labevents.csv` file to have the following columns:
- `itemid`: The laboratory item identifier (we filter for 50852 for HbA1c)
- Other columns are present but not used for counting

## Performance

- Default chunk size: 100,000 rows
- Can be modified in the script if needed for different memory constraints
- Progress indicators every 10 chunks for very large files 