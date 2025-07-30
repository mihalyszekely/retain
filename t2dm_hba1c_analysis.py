#!/usr/bin/env python3
"""
Script to analyze HbA1c measurement distribution among T2DM patients.
Identifies T2DM patients and counts their HbA1c measurements to determine
how many have sufficient measurements for trend prediction analysis.
"""

"""
# Basic analysis without plotting
python t2dm_hba1c_analysis.py diagnoses_icd_MIMIC-IV.csv labevents.csv

# Analysis with basic plotting
python t2dm_hba1c_analysis.py diagnoses_icd_MIMIC-IV.csv labevents.csv --plot

# Analysis with detailed plotting
python t2dm_hba1c_analysis.py diagnoses_icd_MIMIC-IV.csv labevents.csv --plot --detailed
"""

import pandas as pd
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

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

def create_measurement_distribution_plot(results, output_file="hba1c_measurement_distribution.png"):
    """
    Create a bar plot showing the distribution of HbA1c measurements per patient.
    The bar plot is wider, and the pie chart is about half its width.
    
    Args:
        results (dict): Analysis results from analyze_hba1c_by_t2dm_patients
        output_file (str): Output file path for the plot
    """
    distribution = results['measurement_distribution']
    
    # Prepare data for plotting
    measurement_counts = sorted(distribution.keys())
    patient_counts = [distribution[count] for count in measurement_counts]
    
    # Create figure with subplots, bar plot is wider
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [4, 1]})
    
    # Plot 1: Bar chart of measurement distribution
    bars = ax1.bar(measurement_counts, patient_counts, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    ## Set y-axis to log scale
    #ax1.set_yscale('log')
    
    # Highlight bars for 2+ measurements (eligible for trend prediction)
    for i, count in enumerate(measurement_counts):
        if count >= 2:
            bars[i].set_color('lightcoral')
            bars[i].set_edgecolor('darkred')
    
    ax1.set_xlabel('Number of HbA1c Measurements per Patient')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Distribution of HbA1c Measurements Among T2DM Patients')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Pie chart showing eligible vs non-eligible patients
    eligible = results['patients_with_2plus_measurements']
    non_eligible = results['patients_with_hba1c'] - eligible
    no_measurements = results['total_t2dm_patients'] - results['patients_with_hba1c']
    
    sizes = [eligible, non_eligible, no_measurements]
    labels = [f'Eligible (≥2 measurements)\n{eligible:,} patients', 
              f'1 measurement\n{non_eligible:,} patients',
              f'No HbA1c measurements\n{no_measurements:,} patients']
    colors = ['lightcoral', 'lightblue', 'lightgray']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    
    # Enhance text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('T2DM Patients: Eligibility for Trend Prediction')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_file}")
    plt.show()

def create_detailed_distribution_plot(results, output_file="hba1c_detailed_distribution.png"):
    """
    Create a detailed visualization showing the measurement distribution with statistics.
    
    Args:
        results (dict): Analysis results from analyze_hba1c_by_t2dm_patients
        output_file (str): Output file path for the plot
    """
    distribution = results['measurement_distribution']
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Cumulative distribution
    measurement_counts = sorted(distribution.keys())
    patient_counts = [distribution[count] for count in measurement_counts]
    cumulative_patients = np.cumsum(patient_counts)
    
    ax1.plot(measurement_counts, cumulative_patients, 'o-', linewidth=2, markersize=6)
    ax1.axhline(y=results['patients_with_2plus_measurements'], color='red', 
                linestyle='--', alpha=0.7, label=f'Eligible threshold ({results["patients_with_2plus_measurements"]:,} patients)')
    ax1.set_xlabel('Number of HbA1c Measurements')
    ax1.set_ylabel('Cumulative Number of Patients')
    ax1.set_title('Cumulative Distribution of HbA1c Measurements')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Log-scale distribution
    ax2.semilogy(measurement_counts, patient_counts, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of HbA1c Measurements')
    ax2.set_ylabel('Number of Patients (log scale)')
    ax2.set_title('HbA1c Measurement Distribution (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistics summary
    stats_data = {
        'Metric': ['Total T2DM Patients', 'With HbA1c', 'With ≥2 Measurements', 'Eligible %'],
        'Count': [
            results['total_t2dm_patients'],
            results['patients_with_hba1c'],
            results['patients_with_2plus_measurements'],
            round((results['patients_with_2plus_measurements'] / results['total_t2dm_patients']) * 100, 1)
        ]
    }
    
    # Create a simple text-based summary
    ax3.axis('off')
    summary_text = f"""
    T2DM HbA1c Analysis Summary
    
    Total T2DM Patients: {results['total_t2dm_patients']:,}
    Patients with HbA1c: {results['patients_with_hba1c']:,} ({results['patients_with_hba1c']/results['total_t2dm_patients']*100:.1f}%)
    Patients with ≥2 measurements: {results['patients_with_2plus_measurements']:,} ({results['patients_with_2plus_measurements']/results['total_t2dm_patients']*100:.1f}%)
    Total HbA1c measurements: {results['total_hba1c_measurements']:,}
    Average measurements per patient: {results['total_hba1c_measurements']/results['patients_with_hba1c']:.1f}
    """
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    # Plot 4: Top measurement counts (most common)
    top_counts = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    top_measurements = [str(item[0]) for item in top_counts]
    top_patients = [item[1] for item in top_counts]
    
    bars = ax4.bar(range(len(top_counts)), top_patients, color='lightgreen', edgecolor='darkgreen')
    ax4.set_xlabel('Number of HbA1c Measurements')
    ax4.set_ylabel('Number of Patients')
    ax4.set_title('Top 10 Most Common Measurement Counts')
    ax4.set_xticks(range(len(top_counts)))
    ax4.set_xticklabels(top_measurements)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Detailed distribution plot saved to: {output_file}")
    plt.show()

def main():
    """Main function to run the T2DM HbA1c analysis."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze HbA1c measurement distribution among T2DM patients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python t2dm_hba1c_analysis.py diagnoses_icd.csv labevents.csv
  python t2dm_hba1c_analysis.py diagnoses_icd.csv labevents.csv --plot
  python t2dm_hba1c_analysis.py diagnoses_icd.csv labevents.csv --plot --detailed
        """
    )
    
    parser.add_argument('diagnoses_file', help='Path to the diagnoses_icd.csv file')
    parser.add_argument('labevents_file', help='Path to the labevents.csv file')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate plots of the HbA1c measurement distribution')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed plots with additional visualizations (requires --plot)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.diagnoses_file).exists():
        print(f"Error: File '{args.diagnoses_file}' does not exist.")
        sys.exit(1)
    if not Path(args.labevents_file).exists():
        print(f"Error: File '{args.labevents_file}' does not exist.")
        sys.exit(1)
    
    print("=" * 70)
    print("T2DM Patient HbA1c Measurement Analysis")
    print("=" * 70)
    print(f"Diagnoses file: {args.diagnoses_file}")
    print(f"Lab events file: {args.labevents_file}")
    if args.plot:
        print("Plotting enabled: Will generate distribution visualizations")
    print("=" * 70)
    
    # Step 1: Identify T2DM patients
    t2dm_patients = identify_t2dm_patients(args.diagnoses_file)
    if t2dm_patients is None:
        sys.exit(1)
    
    # Step 2: Analyze HbA1c measurements for T2DM patients
    results = analyze_hba1c_by_t2dm_patients(args.labevents_file, t2dm_patients)
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
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        create_measurement_distribution_plot(results)
        
        if args.detailed:
            create_detailed_distribution_plot(results)
    
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