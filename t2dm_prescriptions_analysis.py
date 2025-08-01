#!/usr/bin/env python3
"""
Script to analyze prescriptions for Type 2 Diabetes cohort.
Identifies the most common medications prescribed to T2D patients.
"""

"""
usage:
python t2dm_prescriptions_analysis.py diagnoses_icd_MIMIC-IV.csv prescriptions.csv
"""

import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

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
    Identify patients with Type 2 Diabetes from diagnoses file.
    
    Args:
        diagnoses_file (str): Path to the diagnoses_icd CSV file
        chunk_size (int): Number of rows to process at once
    
    Returns:
        set: Set of subject_ids for patients with T2DM
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
            
            # Get T2DM patients from this chunk
            t2dm_icd9 = chunk[icd9_mask]
            t2dm_icd10 = chunk[icd10_mask]
            
            # Update set of T2DM patients
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

def analyze_prescriptions_for_t2dm(prescriptions_file, t2dm_patients, chunk_size=1000000):
    """
    Analyze prescriptions for T2DM patients to find most common medications.
    
    Args:
        prescriptions_file (str): Path to the prescriptions CSV file
        t2dm_patients (set): Set of subject_ids for T2DM patients
        chunk_size (int): Number of rows to process at once
    
    Returns:
        dict: Analysis results with medication counts and details
    """
    medication_counts = Counter()
    total_prescriptions = 0
    t2dm_prescriptions = 0
    
    print(f"Analyzing prescriptions for T2DM patients from {prescriptions_file}...")
    print(f"Processing in chunks of {chunk_size:,} rows...")
    
    try:
        for chunk in pd.read_csv(prescriptions_file, chunksize=chunk_size, low_memory=False):
            total_prescriptions += len(chunk)
            
            # Filter for T2DM patients
            t2dm_mask = chunk['subject_id'].isin(t2dm_patients)
            t2dm_chunk = chunk[t2dm_mask]
            
            if len(t2dm_chunk) > 0:
                t2dm_prescriptions += len(t2dm_chunk)
                
                # Count medications by drug name
                # Clean drug names: convert to lowercase and strip whitespace
                drug_names = t2dm_chunk['drug'].str.lower().str.strip()
                
                # Update medication counts
                medication_counts.update(drug_names)
        
        print(f"Processed {total_prescriptions:,} total prescriptions")
        print(f"Found {t2dm_prescriptions:,} prescriptions for T2DM patients")
        
    except FileNotFoundError:
        print(f"Error: File '{prescriptions_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing prescriptions file: {e}")
        return None
    
    return {
        'medication_counts': medication_counts,
        'total_prescriptions': total_prescriptions,
        't2dm_prescriptions': t2dm_prescriptions,
        'unique_medications': len(medication_counts)
    }

def create_medication_plot(medication_counts, top_n=20, output_file="t2dm_medications_plot.png"):
    """
    Create a bar plot of the top N most common medications.
    
    Args:
        medication_counts (Counter): Counter object with medication counts
        top_n (int): Number of top medications to plot
        output_file (str): Output file path for the plot
    """
    # Get top N medications
    top_medications = medication_counts.most_common(top_n)
    
    if not top_medications:
        print("No medications found to plot.")
        return
    
    # Prepare data for plotting
    medications = [item[0] for item in top_medications]
    counts = [item[1] for item in top_medications]
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    bars = plt.bar(range(len(medications)), counts, color='steelblue', alpha=0.7)
    
    # Customize the plot
    plt.title(f'Top {top_n} Most Common Medications for T2DM Patients', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Medication', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Prescriptions', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(range(len(medications)), medications, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    # Show the plot
    plt.show()

def main():
    """Main function to run the T2DM prescriptions analysis."""
    
    parser = argparse.ArgumentParser(description='Analyze prescriptions for T2DM cohort')
    parser.add_argument('diagnoses_file', help='Path to diagnoses_icd CSV file')
    parser.add_argument('prescriptions_file', help='Path to prescriptions CSV file')
    parser.add_argument('--chunk-size', type=int, default=1000000, 
                       help='Number of rows to process at once (default: 1000000)')
    parser.add_argument('--top-n', type=int, default=50,
                       help='Number of top medications to display (default: 50)')
    parser.add_argument('--plot', action='store_true',
                       help='Create a bar plot of top 20 medications')
    parser.add_argument('--plot-file', type=str, default='t2dm_medications_plot.png',
                       help='Output file for the plot (default: t2dm_medications_plot.png)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.diagnoses_file).exists():
        print(f"Error: File '{args.diagnoses_file}' does not exist.")
        sys.exit(1)
    
    if not Path(args.prescriptions_file).exists():
        print(f"Error: File '{args.prescriptions_file}' does not exist.")
        sys.exit(1)
    
    print("=" * 80)
    print("Type 2 Diabetes Prescriptions Analysis")
    print("=" * 80)
    print(f"Diagnoses file: {args.diagnoses_file}")
    print(f"Prescriptions file: {args.prescriptions_file}")
    print(f"Chunk size: {args.chunk_size:,}")
    print("=" * 80)
    
    # Step 1: Identify T2DM patients
    t2dm_patients = identify_t2dm_patients(args.diagnoses_file, args.chunk_size)
    
    if t2dm_patients is None:
        print("Failed to identify T2DM patients.")
        sys.exit(1)
    
    if len(t2dm_patients) == 0:
        print("No T2DM patients found.")
        sys.exit(1)
    
    # Step 2: Analyze prescriptions for T2DM patients
    results = analyze_prescriptions_for_t2dm(args.prescriptions_file, t2dm_patients, args.chunk_size)
    
    if results is None:
        print("Failed to analyze prescriptions.")
        sys.exit(1)
    
    # Step 3: Display results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Total T2DM patients: {len(t2dm_patients):,}")
    print(f"Total prescriptions processed: {results['total_prescriptions']:,}")
    print(f"Prescriptions for T2DM patients: {results['t2dm_prescriptions']:,}")
    print(f"Unique medications prescribed: {results['unique_medications']:,}")
    print("=" * 80)
    
    # Display top medications
    print(f"\nTop {args.top_n} Most Common Medications for T2DM Patients:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Count':<10} {'Medication'}")
    print("-" * 80)
    
    for i, (medication, count) in enumerate(results['medication_counts'].most_common(args.top_n), 1):
        print(f"{i:<5} {count:<10} {medication}")
    
    # Create plot if requested
    if args.plot:
        print(f"\nCreating plot of top 20 medications...")
        create_medication_plot(results['medication_counts'], top_n=20, output_file=args.plot_file)
    
    # Save detailed results to file
    output_file = "t2dm_prescriptions_results.txt"
    with open(output_file, 'w') as f:
        f.write("Type 2 Diabetes Prescriptions Analysis Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total T2DM patients: {len(t2dm_patients):,}\n")
        f.write(f"Total prescriptions processed: {results['total_prescriptions']:,}\n")
        f.write(f"Prescriptions for T2DM patients: {results['t2dm_prescriptions']:,}\n")
        f.write(f"Unique medications prescribed: {results['unique_medications']:,}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("All Medications Prescribed to T2DM Patients:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Count':<10} {'Medication'}\n")
        f.write("-" * 60 + "\n")
        
        for medication, count in results['medication_counts'].most_common():
            f.write(f"{count:<10} {medication}\n")
    
    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main() 