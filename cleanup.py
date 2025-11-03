#!/usr/bin/env python3
"""
Cleanup script to remove non-financial models and datasets from AI Shield
"""

import os
import glob

def cleanup_directory():
    """Remove all non-financial models and datasets"""
    
    print("🧹 AI Shield Directory Cleanup")
    print("=" * 40)
    
    # Files to keep (essential system files)
    keep_files = {
        'sample_model.pkl',
        'demo_model.joblib', 
        'demo_dataset.csv',
        'demo_analysis_results.json',
        'sample_dataset.csv'
    }
    
    removed_count = 0
    
    # Clean models directory
    print("\n📁 Cleaning models directory...")
    models_dir = 'models'
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file not in keep_files:
                file_path = os.path.join(models_dir, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"  ❌ Removed: {file}")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {file}: {e}")
                else:
                    print(f"  ⏭️  Skipped: {file} (not a file)")
            else:
                print(f"  ✅ Kept: {file}")
    
    # Clean data directory  
    print("\n📊 Cleaning data directory...")
    data_dir = 'data'
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            # Keep essential files and sample datasets
            if file not in keep_files and not file.startswith('sample_dataset'):
                file_path = os.path.join(data_dir, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"  ❌ Removed: {file}")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {file}: {e}")
                else:
                    print(f"  ⏭️  Skipped: {file} (not a file)")
            else:
                print(f"  ✅ Kept: {file}")
    
    # Clean results directory if it contains old analysis files
    print("\n📋 Cleaning results directory...")
    results_dir = 'results'
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith(('.json', '.html', '.txt')) and 'demo' not in file.lower():
                file_path = os.path.join(results_dir, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"  ❌ Removed: {file}")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {file}: {e}")
            else:
                print(f"  ✅ Kept: {file}")
    
    # Check uploads directory
    print("\n📤 Cleaning uploads directory...")
    uploads_dir = 'uploads'
    if os.path.exists(uploads_dir):
        upload_files = os.listdir(uploads_dir)
        if upload_files:
            print(f"  📋 Found {len(upload_files)} files in uploads - keeping for now")
            for file in upload_files[:5]:  # Show first 5
                print(f"    • {file}")
            if len(upload_files) > 5:
                print(f"    ... and {len(upload_files) - 5} more")
        else:
            print("  ✅ Uploads directory is empty")
    
    print("\n" + "=" * 40)
    print(f"🎉 Cleanup Complete!")
    print(f"📊 Removed {removed_count} files")
    
    # Show what's left
    print(f"\n📁 Remaining files:")
    if os.path.exists(models_dir):
        models_remaining = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
        print(f"  Models: {len(models_remaining)} files")
        for file in models_remaining:
            print(f"    • {file}")
    
    if os.path.exists(data_dir):
        data_remaining = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        print(f"  Data: {len(data_remaining)} files")
        for file in data_remaining:
            print(f"    • {file}")
    
    print(f"\n✅ Directory is now clean and ready for financial models!")

if __name__ == "__main__":
    # Change to the AI-Shield directory
    os.chdir('/Users/chetan/AI-Shield')
    cleanup_directory()