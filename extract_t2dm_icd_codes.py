#!/usr/bin/env python3
"""
Script to extract Type 2 Diabetes ICD codes from d_icd_diagnoses.csv

This script filters the d_icd_diagnoses table to extract:
- ICD-9 codes: 250x0 or 250x2 (where x is any digit 0-9)
- ICD-10 codes: E11* (all codes starting with E11)

The results are saved to a file for use as features in the Type 2 Diabetes cohort.
"""

import pandas as pd
import re
import sys
from pathlib import Path

def extract_t2dm_icd_codes(input_file, output_file):
    """
    Extract Type 2 Diabetes ICD codes from the d_icd_diagnoses.csv file.
    
    Args:
        input_file (str): Path to the d_icd_diagnoses.csv file
        output_file (str): Path to save the extracted codes
    """
    
    print(f"Reading ICD codes from {input_file}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} total ICD codes")
        
        # Filter for Type 2 Diabetes codes
        t2dm_codes = []
        
        for _, row in df.iterrows():
            icd_code = str(row['icd_code']).strip()
            icd_version = row['icd_version']
            long_title = str(row['long_title']).strip()
            
            # ICD-9 codes: 250x0 or 250x2 (where x is any digit 0-9)
            if icd_version == 9:
                # Pattern: 250 followed by any digit, then 0 or 2
                if re.match(r'^250[0-9][02]$', icd_code):
                    t2dm_codes.append({
                        'icd_code': icd_code,
                        'icd_version': icd_version,
                        'long_title': long_title,
                        'pattern_matched': '250x0 or 250x2'
                    })
            
            # ICD-10 codes: E11* (all codes starting with E11)
            elif icd_version == 10:
                if icd_code.startswith('E11'):
                    t2dm_codes.append({
                        'icd_code': icd_code,
                        'icd_version': icd_version,
                        'long_title': long_title,
                        'pattern_matched': 'E11*'
                    })
        
        # Convert to DataFrame
        t2dm_df = pd.DataFrame(t2dm_codes)
        
        if len(t2dm_df) == 0:
            print("No Type 2 Diabetes ICD codes found!")
            return
        
        print(f"Found {len(t2dm_df)} Type 2 Diabetes ICD codes:")
        print(f"- ICD-9 codes: {len(t2dm_df[t2dm_df['icd_version'] == 9])}")
        print(f"- ICD-10 codes: {len(t2dm_df[t2dm_df['icd_version'] == 10])}")
        
        # Save to file
        t2dm_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Display summary
        print("\nSummary of extracted codes:")
        print("=" * 80)
        
        # Show ICD-9 codes
        icd9_codes = t2dm_df[t2dm_df['icd_version'] == 9]
        if len(icd9_codes) > 0:
            print(f"\nICD-9 Codes ({len(icd9_codes)} found):")
            for _, row in icd9_codes.iterrows():
                print(f"  {row['icd_code']}: {row['long_title']}")
        
        # Show ICD-10 codes (first 10 for brevity)
        icd10_codes = t2dm_df[t2dm_df['icd_version'] == 10]
        if len(icd10_codes) > 0:
            print(f"\nICD-10 Codes ({len(icd10_codes)} found):")
            for i, (_, row) in enumerate(icd10_codes.iterrows()):
                if i < 10:  # Show first 10
                    print(f"  {row['icd_code']}: {row['long_title']}")
                elif i == 10:
                    print(f"  ... and {len(icd10_codes) - 10} more ICD-10 codes")
                    break
        
        print("\n" + "=" * 80)
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    """Main function to run the extraction."""
    
    input_file = "d_icd_diagnoses.csv"
    output_file = "t2dm_icd_codes.csv"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        print("Please ensure the file is in the current directory.")
        sys.exit(1)
    
    print("Type 2 Diabetes ICD Code Extractor")
    print("=" * 40)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Extract the codes
    extract_t2dm_icd_codes(input_file, output_file)
    
    print("\nExtraction completed successfully!")

if __name__ == "__main__":
    main() 