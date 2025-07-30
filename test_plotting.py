#!/usr/bin/env python3
"""
Test script to verify the plotting functionality of the T2DM HbA1c analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from t2dm_hba1c_analysis import create_measurement_distribution_plot, create_detailed_distribution_plot

def create_sample_results():
    """Create sample results for testing the plotting functions."""
    return {
        'total_t2dm_patients': 5000,
        'patients_with_hba1c': 3200,
        'patients_with_2plus_measurements': 1800,
        'total_hba1c_measurements': 8500,
        'patient_measurement_counts': {1: 1400, 2: 800, 3: 600, 4: 300, 5: 100},
        'measurement_distribution': {1: 1400, 2: 800, 3: 600, 4: 300, 5: 100},
        'eligible_patients': list(range(1000, 2800))  # Sample patient IDs
    }

def test_plotting():
    """Test the plotting functions with sample data."""
    print("Testing plotting functionality...")
    
    # Create sample results
    sample_results = create_sample_results()
    
    # Test basic distribution plot
    print("Creating basic distribution plot...")
    create_measurement_distribution_plot(sample_results, "test_distribution.png")
    
    # Test detailed distribution plot
    print("Creating detailed distribution plot...")
    create_detailed_distribution_plot(sample_results, "test_detailed_distribution.png")
    
    print("Plotting tests completed successfully!")
    print("Check the generated PNG files to verify the plots look correct.")

if __name__ == "__main__":
    test_plotting() 