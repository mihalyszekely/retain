#!/usr/bin/env python3
"""
Script to count total HbA1c measurements in the labevents table.
HbA1c has itemid 50852 in the d_labitems table.
Processes the large labevents.csv file in chunks for memory efficiency.
"""

import pandas as pd
import sys
from pathlib import Path

def count_hba1c_measurements(labevents_file, chunk_size=1000000):
    """
    Count total HbA1c measurements in labevents.csv file.
    
    Args:
        labevents_file (str): Path to the labevents.csv file
        chunk_size (int): Number of rows to process at once
    
    Returns:
        int: Total count of HbA1c measurements
    """
    # HbA1c itemid from d_labitems table
    HBA1C_ITEMID = 50852
    
    total_count = 0
    chunk_num = 0
    
    print(f"Processing {labevents_file} in chunks of {chunk_size:,} rows...")
    
    try:
        # Process file in chunks
        for chunk in pd.read_csv(labevents_file, chunksize=chunk_size):
            chunk_num += 1
            
            # Count HbA1c measurements in this chunk
            hba1c_count = len(chunk[chunk['itemid'] == HBA1C_ITEMID])
            total_count += hba1c_count
            
            print(f"Chunk {chunk_num}: Found {hba1c_count:,} HbA1c measurements (Total so far: {total_count:,})")
            
            # Optional: Add a progress indicator for very large files
            if chunk_num % 10 == 0:
                print(f"Processed {chunk_num * chunk_size:,} rows...")
    
    except FileNotFoundError:
        print(f"Error: File '{labevents_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
    
    return total_count

def main():
    """Main function to run the HbA1c counting script."""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python count_hba1c.py <labevents.csv>")
        print("Example: python count_hba1c.py labevents.csv")
        sys.exit(1)
    
    labevents_file = sys.argv[1]
    
    # Check if file exists
    if not Path(labevents_file).exists():
        print(f"Error: File '{labevents_file}' does not exist.")
        sys.exit(1)
    
    print("=" * 60)
    print("HbA1c Measurement Counter")
    print("=" * 60)
    print(f"Target file: {labevents_file}")
    print(f"Looking for itemid: 50852 (HbA1c)")
    print("=" * 60)
    
    # Count HbA1c measurements
    total_hba1c = count_hba1c_measurements(labevents_file)
    
    if total_hba1c is not None:
        print("=" * 60)
        print(f"FINAL RESULT: {total_hba1c:,} total HbA1c measurements found")
        print("=" * 60)
    else:
        print("Failed to count HbA1c measurements.")
        sys.exit(1)

if __name__ == "__main__":
    main() 