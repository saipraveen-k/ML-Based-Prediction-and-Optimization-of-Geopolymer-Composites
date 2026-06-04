"""
Clean Up Waste Outputs Script
==============================
This script scans the outputs directory and deletes redundant older files 
that have been generated with historical timestamps, keeping only the 
latest file for each plot and report category.
"""

import os
import re
import sys

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import print_section

def clear_waste(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
        
    print_section(f"CLEANING DIRECTORY: {directory}")
    files = os.listdir(directory)
    
    # Matches files ending with _YYYYMMDD_HHMMSS.ext
    pattern = re.compile(r'^(.*?)(?:_?\d{8}_\d{6})(\.[a-zA-Z0-9]+)$')
    
    groups = {}
    non_timestamped_count = 0
    
    for f in files:
        path = os.path.join(directory, f)
        if not os.path.isfile(path):
            continue
            
        match = pattern.match(f)
        if match:
            prefix, ext = match.groups()
            group_key = (prefix, ext)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(f)
        else:
            non_timestamped_count += 1
            
    print(f"Found {len(files)} total files.")
    print(f"Found {len(groups)} distinct timestamped file groups.")
    print(f"Found {non_timestamped_count} static (non-timestamped) files (will be preserved).")
    
    deleted_count = 0
    
    for (prefix, ext), group_files in groups.items():
        if len(group_files) <= 1:
            continue
            
        # Sort files alphabetically (which works chronologically for YYYYMMDD_HHMMSS format)
        group_files.sort()
        
        # Keep the latest file
        latest_file = group_files[-1]
        to_delete = group_files[:-1]
        
        print(f"\nGroup: {prefix}{ext}")
        print(f"  [KEEP] Latest: {latest_file}")
        
        for df in to_delete:
            dpath = os.path.join(directory, df)
            try:
                os.remove(dpath)
                print(f"  [DELETE] Removed older file: {df}")
                deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to delete {df}: {e}")
                
    print(f"\nCompleted cleaning for {directory}. Removed {deleted_count} redundant files.")

def main():
    print_section("STARTING OUTPUT CLEANUP PIPELINE")
    
    clear_waste('outputs/graphs')
    clear_waste('outputs/reports')
    clear_waste('outputs/metrics')
    
    print_section("CLEANUP PIPELINE COMPLETED")

if __name__ == '__main__':
    main()
