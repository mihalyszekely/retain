import pandas as pd

# File path to your labevents.csv
file_path = 'labevents.csv'

# Define chunk size (adjust if you have more or less RAM)
chunk_size = 1_000_000

# Initialize a counter for HbA1c rows
hba1c_count = 0
chunk_num = 0

# Read the file in chunks
for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
    chunk_num += 1
    count_in_chunk = (chunk['itemid'] == 50852).sum()
    hba1c_count += count_in_chunk
    print(f"Processed chunk {chunk_num}: Found {count_in_chunk} HbA1c measurements, running total: {hba1c_count}")

print(f"Total number of HbA1c measurements: {hba1c_count}")