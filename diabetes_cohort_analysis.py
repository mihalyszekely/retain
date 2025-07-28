#!/usr/bin/env python3
"""
Script to identify Type 2 Diabetes cohort using ICD-9 and ICD-10 codes.
Counts unique patients and hospitalizations with Type 2 Diabetes diagnoses.
"""

import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

def get_t2dm_patterns():
    """
    Define pattern matching rules for Type 2 Diabetes Mellitus ICD codes.
    
    Returns:
        dict: Dictionary with pattern matching rules for T2DM
    """
    return {
        'icd9_pattern': {
            'startswith': '250',
            'length': 5,
            'endswith': ['0', '2']
        },
        'icd10_pattern': {
            'startswith': 'E11'
        }
    }

def analyze_diabetes_cohort(diagnoses_file, chunk_size=1000000):
    """
    Analyze diabetes cohort from diagnoses file.
    
    Args:
        diagnoses_file (str): Path to the diagnoses_icd CSV file
        chunk_size (int): Number of rows to process at once
    
    Returns:
        dict: Analysis results with counts and details
    """
    patterns = get_t2dm_patterns()
    
    # Sets to track unique patients and hospitalizations
    unique_patients = set()
    unique_hospitalizations = set()
    
    # Counters for different code types
    icd9_count = 0
    icd10_count = 0
    
    print(f"Processing {diagnoses_file} in chunks of {chunk_size:,} rows...")
    print("Looking for Type 2 Diabetes codes:")
    print(f"  ICD-9 pattern: {patterns['icd9_pattern']['startswith']}.x0 and {patterns['icd9_pattern']['startswith']}.x2")
    print(f"  ICD-10 pattern: {patterns['icd10_pattern']['startswith']}*")
    print("=" * 60)
    
    chunk_num = 0
    
    try:
        for chunk in pd.read_csv(diagnoses_file, chunksize=chunk_size, low_memory=False):
            chunk_num += 1
            
            # Filter for T2DM codes using pattern matching
            # ICD-9: codes starting with 250, 5 characters long, ending with 0 or 2
            icd9_mask = (
                (chunk['icd_version'] == 9) & 
                (chunk['icd_code'].str.startswith(patterns['icd9_pattern']['startswith'], na=False)) &
                (chunk['icd_code'].str.len() == patterns['icd9_pattern']['length']) &
                (chunk['icd_code'].str.endswith(tuple(patterns['icd9_pattern']['endswith']), na=False))
            )
            
            # ICD-10: codes starting with E11
            icd10_mask = (
                (chunk['icd_version'] == 10) & 
                (chunk['icd_code'].str.startswith(patterns['icd10_pattern']['startswith'], na=False))
            )
            
            # Get T2DM diagnoses
            t2dm_icd9 = chunk[icd9_mask]
            t2dm_icd10 = chunk[icd10_mask]
            
            # Update counts
            icd9_count += len(t2dm_icd9)
            icd10_count += len(t2dm_icd10)
            
            # Update unique patients and hospitalizations
            if len(t2dm_icd9) > 0:
                unique_patients.update(t2dm_icd9['subject_id'].unique())
                unique_hospitalizations.update(t2dm_icd9['hadm_id'].unique())
            
            if len(t2dm_icd10) > 0:
                unique_patients.update(t2dm_icd10['subject_id'].unique())
                unique_hospitalizations.update(t2dm_icd10['hadm_id'].unique())
            
            print(f"Chunk {chunk_num}: Found {len(t2dm_icd9)} ICD-9 and {len(t2dm_icd10)} ICD-10 T2DM diagnoses")
            print(f"  Total unique patients so far: {len(unique_patients):,}")
            print(f"  Total unique hospitalizations so far: {len(unique_hospitalizations):,}")
            
            if chunk_num % 10 == 0:
                print(f"Processed {chunk_num * chunk_size:,} rows...")
    
    except FileNotFoundError:
        print(f"Error: File '{diagnoses_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
    
    return {
        'unique_patients': len(unique_patients),
        'unique_hospitalizations': len(unique_hospitalizations),
        'icd9_diagnoses': icd9_count,
        'icd10_diagnoses': icd10_count,
        'total_diagnoses': icd9_count + icd10_count,
        'patient_list': list(unique_patients),
        'hospitalization_list': list(unique_hospitalizations)
    }

def main():
    """Main function to run the diabetes cohort analysis."""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python diabetes_cohort_analysis.py <diagnoses_icd.csv>")
        print("Example: python diabetes_cohort_analysis.py diagnoses_icd_MIMIC-IV.csv")
        sys.exit(1)
    
    diagnoses_file = sys.argv[1]
    
    # Check if file exists
    if not Path(diagnoses_file).exists():
        print(f"Error: File '{diagnoses_file}' does not exist.")
        sys.exit(1)
    
    print("=" * 60)
    print("Type 2 Diabetes Cohort Analysis")
    print("=" * 60)
    print(f"Target file: {diagnoses_file}")
    print("=" * 60)
    
    # Analyze diabetes cohort
    results = analyze_diabetes_cohort(diagnoses_file)
    
    if results is not None:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Total unique patients with T2DM: {results['unique_patients']:,}")
        print(f"Total unique hospitalizations with T2DM: {results['unique_hospitalizations']:,}")
        print(f"Total T2DM diagnoses (ICD-9): {results['icd9_diagnoses']:,}")
        print(f"Total T2DM diagnoses (ICD-10): {results['icd10_diagnoses']:,}")
        print(f"Total T2DM diagnoses: {results['total_diagnoses']:,}")
        print("=" * 60)
        
        # Save results to file
        output_file = "t2dm_cohort_results.txt"
        with open(output_file, 'w') as f:
            f.write("Type 2 Diabetes Cohort Analysis Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total unique patients with T2DM: {results['unique_patients']:,}\n")
            f.write(f"Total unique hospitalizations with T2DM: {results['unique_hospitalizations']:,}\n")
            f.write(f"Total T2DM diagnoses (ICD-9): {results['icd9_diagnoses']:,}\n")
            f.write(f"Total T2DM diagnoses (ICD-10): {results['icd10_diagnoses']:,}\n")
            f.write(f"Total T2DM diagnoses: {results['total_diagnoses']:,}\n")
        
        print(f"Results saved to: {output_file}")
    else:
        print("Failed to analyze diabetes cohort.")
        sys.exit(1)

if __name__ == "__main__":
    main() 