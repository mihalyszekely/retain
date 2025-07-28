#!/usr/bin/env python3
"""
Script to analyze HbA1c measurement distribution among T2DM patients.
Identifies T2DM patients and counts their HbA1c measurements to determine
how many have sufficient measurements for trend prediction analysis.
"""

import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict, Counter

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

def identify_t2dm_patients(diagnoses_file, chunk_size=1000000):
    """
    Identify all patients with Type 2 Diabetes diagnoses.
    
    Args:
        diagnoses_file (str): Path to the diagnoses_icd CSV file
        chunk_size (int): Number of rows to process at once
    
    Returns:
        set: Set of subject_ids with T2DM diagnoses
    """
    patterns = get_t2dm_patterns()
    t2dm_patients = set()
    
    print(f"Identifying T2DM patients from {diagnoses_file}...")
    
    try:
        for chunk in pd.read_csv(diagnoses_file, chunksize=chunk_size, low_memory=False):
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
            
            # Add patients to set
            if len(t2dm_icd9) > 0:
                t2dm_patients.update(t2dm_icd9['subject_id'].unique())
            if len(t2dm_icd10) > 0:
                t2dm_patients.update(t2dm_icd10['subject_id'].unique())
    
    except FileNotFoundError:
        print(f"Error: File '{diagnoses_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing diagnoses file: {e}")
        return None
    
    print(f"Identified {len(t2dm_patients):,} unique T2DM patients")
    return t2dm_patients

def analyze_hba1c_by_t2dm_patients(labevents_file, t2dm_patients, chunk_size=1000000):
    """
    Analyze HbA1c measurements for T2DM patients.
    
    Args:
        labevents_file (str): Path to the labevents.csv file
        t2dm_patients (set): Set of subject_ids with T2DM
        chunk_size (int): Number of rows to process at once
    
    Returns:
        dict: Analysis results with measurement counts and distribution
    """
    HBA1C_ITEMID = 50852
    patient_hba1c_counts = defaultdict(int)
    total_hba1c_measurements = 0
    
    print(f"Analyzing HbA1c measurements for T2DM patients from {labevents_file}...")
    
    try:
        for chunk in pd.read_csv(labevents_file, chunksize=chunk_size):
            # Filter for HbA1c measurements and T2DM patients
            hba1c_mask = chunk['itemid'] == HBA1C_ITEMID
            t2dm_mask = chunk['subject_id'].isin(t2dm_patients)
            
            # Get HbA1c measurements for T2DM patients
            t2dm_hba1c = chunk[hba1c_mask & t2dm_mask]
            
            # Count measurements per patient
            for _, row in t2dm_hba1c.iterrows():
                patient_hba1c_counts[row['subject_id']] += 1
                total_hba1c_measurements += 1
    
    except FileNotFoundError:
        print(f"Error: File '{labevents_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing labevents file: {e}")
        return None
    
    # Analyze distribution
    measurement_counts = list(patient_hba1c_counts.values())
    patients_with_measurements = len(patient_hba1c_counts)
    patients_with_2plus = sum(1 for count in measurement_counts if count >= 2)
    
    # Create distribution summary
    count_distribution = Counter(measurement_counts)
    
    return {
        'total_t2dm_patients': len(t2dm_patients),
        'patients_with_hba1c': patients_with_measurements,
        'patients_with_2plus_measurements': patients_with_2plus,
        'total_hba1c_measurements': total_hba1c_measurements,
        'patient_measurement_counts': dict(patient_hba1c_counts),
        'measurement_distribution': dict(count_distribution),
        'eligible_patients': [pid for pid, count in patient_hba1c_counts.items() if count >= 2]
    }

def main():
    """Main function to run the T2DM HbA1c analysis."""
    
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python t2dm_hba1c_analysis.py <diagnoses_icd.csv> <labevents.csv>")
        print("Example: python t2dm_hba1c_analysis.py diagnoses_icd.csv labevents.csv")
        sys.exit(1)
    
    diagnoses_file = sys.argv[1]
    labevents_file = sys.argv[2]
    
    # Check if files exist
    if not Path(diagnoses_file).exists():
        print(f"Error: File '{diagnoses_file}' does not exist.")
        sys.exit(1)
    if not Path(labevents_file).exists():
        print(f"Error: File '{labevents_file}' does not exist.")
        sys.exit(1)
    
    print("=" * 70)
    print("T2DM Patient HbA1c Measurement Analysis")
    print("=" * 70)
    print(f"Diagnoses file: {diagnoses_file}")
    print(f"Lab events file: {labevents_file}")
    print("=" * 70)
    
    # Step 1: Identify T2DM patients
    t2dm_patients = identify_t2dm_patients(diagnoses_file)
    if t2dm_patients is None:
        sys.exit(1)
    
    # Step 2: Analyze HbA1c measurements for T2DM patients
    results = analyze_hba1c_by_t2dm_patients(labevents_file, t2dm_patients)
    if results is None:
        sys.exit(1)
    
    # Display results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Total T2DM patients: {results['total_t2dm_patients']:,}")
    print(f"T2DM patients with HbA1c measurements: {results['patients_with_hba1c']:,}")
    print(f"T2DM patients with ≥2 HbA1c measurements: {results['patients_with_2plus_measurements']:,}")
    print(f"Total HbA1c measurements among T2DM patients: {results['total_hba1c_measurements']:,}")
    
    # Calculate percentages
    if results['total_t2dm_patients'] > 0:
        pct_with_measurements = (results['patients_with_hba1c'] / results['total_t2dm_patients']) * 100
        pct_eligible = (results['patients_with_2plus_measurements'] / results['total_t2dm_patients']) * 100
        print(f"Percentage of T2DM patients with HbA1c: {pct_with_measurements:.1f}%")
        print(f"Percentage eligible for trend prediction: {pct_eligible:.1f}%")
    
    # Display measurement distribution
    print("\nHbA1c Measurement Distribution:")
    print("-" * 40)
    for count in sorted(results['measurement_distribution'].keys()):
        patients = results['measurement_distribution'][count]
        print(f"  {count} measurement(s): {patients:,} patients")
    
    print("=" * 70)
    
    # Save detailed results
    output_file = "t2dm_hba1c_results.txt"
    with open(output_file, 'w') as f:
        f.write("T2DM Patient HbA1c Measurement Analysis Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total T2DM patients: {results['total_t2dm_patients']:,}\n")
        f.write(f"T2DM patients with HbA1c measurements: {results['patients_with_hba1c']:,}\n")
        f.write(f"T2DM patients with ≥2 HbA1c measurements: {results['patients_with_2plus_measurements']:,}\n")
        f.write(f"Total HbA1c measurements among T2DM patients: {results['total_hba1c_measurements']:,}\n")
        f.write(f"Percentage of T2DM patients with HbA1c: {pct_with_measurements:.1f}%\n")
        f.write(f"Percentage eligible for trend prediction: {pct_eligible:.1f}%\n\n")
        
        f.write("Measurement Distribution:\n")
        f.write("-" * 30 + "\n")
        for count in sorted(results['measurement_distribution'].keys()):
            patients = results['measurement_distribution'][count]
            f.write(f"{count} measurement(s): {patients:,} patients\n")
        
        f.write(f"\nEligible patients (≥2 measurements): {len(results['eligible_patients']):,}\n")
        f.write("Eligible patient IDs (first 10): " + 
                str(results['eligible_patients'][:10]) + "\n")
    
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main() 