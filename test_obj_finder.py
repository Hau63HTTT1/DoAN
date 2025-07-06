#!/usr/bin/env python3
"""
Test script to find .obj files in PiFuHD results directory
"""

import os
import sys

def find_obj_files():
    """Find all .obj files in the PiFuHD directory"""
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    pifuhd_dir = os.path.join(project_root, 'model', 'PiFu-singleview')
    results_dir = os.path.join(pifuhd_dir, 'results')
    
    print(f"Project root: {project_root}")
    print(f"PiFuHD dir: {pifuhd_dir}")
    print(f"Results dir: {results_dir}")
    
    obj_files = []
    
    # Search in results directory
    if os.path.exists(results_dir):
        print(f"\nSearching in results directory: {results_dir}")
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.obj'):
                    obj_path = os.path.join(root, file)
                    obj_files.append(obj_path)
                    print(f"Found: {obj_path}")
    
    # Search in entire PiFuHD directory
    if os.path.exists(pifuhd_dir):
        print(f"\nSearching in entire PiFuHD directory: {pifuhd_dir}")
        for root, dirs, files in os.walk(pifuhd_dir):
            for file in files:
                if file.endswith('.obj'):
                    obj_path = os.path.join(root, file)
                    if obj_path not in obj_files:
                        obj_files.append(obj_path)
                        print(f"Found: {obj_path}")
    
    if not obj_files:
        print("\nNo .obj files found!")
        
        # List directory contents for debugging
        print(f"\nContents of results directory:")
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                print(f"  {root}:")
                for d in dirs:
                    print(f"    [DIR] {d}")
                for f in files:
                    print(f"    [FILE] {f}")
        else:
            print(f"  Results directory does not exist: {results_dir}")
    
    return obj_files

if __name__ == "__main__":
    obj_files = find_obj_files()
    print(f"\nTotal .obj files found: {len(obj_files)}") 