#!/usr/bin/env python3
"""
Test script to find .obj files with pattern matching
"""

import os
import glob

def test_obj_finder():
    """Test the obj file finding logic"""
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    pifuhd_dir = os.path.join(project_root, 'model', 'PiFu-singleview')
    results_dir = os.path.join(pifuhd_dir, 'results')
    final_output_dir = os.path.join(results_dir, 'pifuhd_final', 'recon')
    
    print(f"Project root: {project_root}")
    print(f"PiFuHD dir: {pifuhd_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Final output dir: {final_output_dir}")
    
    # Test pattern matching
    patterns = [
        os.path.join(final_output_dir, 'result_*_256.obj'),
        os.path.join(results_dir, 'pifuhd_final', 'recon', 'result_*_256.obj'),
        os.path.join(final_output_dir, '*.obj'),
        os.path.join(results_dir, 'pifuhd_final', 'recon', '*.obj'),
    ]
    
    obj_files = []
    
    for pattern in patterns:
        print(f"\nTesting pattern: {pattern}")
        matching_files = glob.glob(pattern)
        if matching_files:
            print(f"Found {len(matching_files)} files:")
            for file in matching_files:
                print(f"  {file}")
                obj_files.append(file)
        else:
            print("No files found")
    
    # Remove duplicates
    obj_files = list(set(obj_files))
    
    print(f"\nTotal unique .obj files found: {len(obj_files)}")
    for file in obj_files:
        print(f"  {file}")
    
    return obj_files

if __name__ == "__main__":
    obj_files = test_obj_finder() 