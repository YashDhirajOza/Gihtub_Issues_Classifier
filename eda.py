import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def explore_github_issues_data(file_path='github_issues_sample'):
    """
    Comprehensive data exploration for GitHub issues dataset
    """
    print("🔍 GITHUB ISSUES DATA EXPLORER")
    print("=" * 50)
    
    # Try to load the file with different formats
    df = None
    possible_extensions = ['', '.csv', '.json', '.xlsx', '.tsv', '.txt']
    
    for ext in possible_extensions:
        try:
            full_path = file_path + ext
            if ext == '.csv' or ext == '':
                df = pd.read_csv(full_path)
            elif ext == '.json':
                df = pd.read_json(full_path)
            elif ext == '.xlsx':
                df = pd.read_excel(full_path)
            elif ext == '.tsv':
                df = pd.read_csv(full_path, sep='\t')
            elif ext == '.txt':
                df = pd.read_csv(full_path, sep='\t')
            
            if df is not None:
                print(f"✅ Successfully loaded: {full_path}")
                break
                
        except Exception as e:
            continue
    
    if df is None:
        print("❌ Could not load the file. Tried these formats:")
        print("   - CSV (.csv)")
        print("   - JSON (.json)")
        print("   - Excel (.xlsx)")
        print("   - TSV (.tsv)")
        print("   - Text (.txt)")
        print("\nPlease ensure the file exists and is in one of these formats.")
        return None
    
    # Basic info
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Show first few rows
    print(f"\n📋 FIRST 3 ROWS:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 100)
    print(df.head(3))
    
    # Data types and missing values
    print(f"\n🔍 DATA TYPES & MISSING VALUES:")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info_df.to_string(index=False))
    
    # Analyze text columns (likely title, body, description)
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' and col.lower() in ['title', 'body', 'description', 'summary', 'issue_title', 'issue_body']:
            text_columns.append(col)
    
    if text_columns:
        print(f"\n📝 TEXT ANALYSIS:")
        for col in text_columns:
            if df[col].notna().sum() > 0:
                lengths = df[col].dropna().str.len()
                print(f"\n{col.upper()}:")
                print(f"  Non-empty entries: {df[col].notna().sum()}")
                print(f"  Average length: {lengths.mean():.0f} characters")
                print(f"  Length range: {lengths.min():.0f} - {lengths.max():.0f}")
                
                # Show sample
                sample_text = df[col].dropna().iloc[0]
                print(f"  Sample: '{sample_text[:100]}{'...' if len(sample_text) > 100 else ''}'")
    
    # Analyze potential priority/label columns
    potential_priority_cols = []
    for col in df.columns:
        if col.lower() in ['priority', 'label', 'severity', 'importance', 'type', 'category', 'status']:
            potential_priority_cols.append(col)
    
    if potential_priority_cols:
        print(f"\n🏷️  PRIORITY/LABEL ANALYSIS:")
        for col in potential_priority_cols:
            unique_values = df[col].value_counts()
            print(f"\n{col.upper()}:")
            print(f"  Unique values: {len(unique_values)}")
            print("  Distribution:")
            for value, count in unique_values.head(10).items():
                percentage = (count / len(df)) * 100
                print(f"    {value}: {count} ({percentage:.1f}%)")
    
    # Look for URL patterns
    url_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10)
            if any('http' in str(val) or 'github' in str(val).lower() for val in sample_values):
                url_columns.append(col)
    
    if url_columns:
        print(f"\n🔗 URL COLUMNS DETECTED:")
        for col in url_columns:
            print(f"  {col}: Contains URLs (likely issue links)")
    
    # Keyword analysis for automatic priority detection
    print(f"\n🔤 KEYWORD ANALYSIS FOR PRIORITY DETECTION:")
    
    # Combine all text columns
    all_text = ""
    for col in text_columns:
        all_text += " " + df[col].fillna("").str.lower().str.cat(sep=" ")
    
    if all_text.strip():
        # Priority-related keywords
        high_priority_keywords = ['crash', 'critical', 'urgent', 'security', 'vulnerability', 'bug', 'error', 'broken', 'fails', 'blocker']
        medium_priority_keywords = ['improvement', 'enhancement', 'feature', 'performance', 'slow', 'usability', 'missing']
        low_priority_keywords = ['documentation', 'typo', 'cosmetic', 'suggestion', 'idea', 'minor', 'cleanup']
        
        print("\nHigh Priority Keywords Found:")
        for keyword in high_priority_keywords:
            count = all_text.count(keyword)
            if count > 0:
                print(f"  '{keyword}': {count} times")
        
        print("\nMedium Priority Keywords Found:")
        for keyword in medium_priority_keywords:
            count = all_text.count(keyword)
            if count > 0:
                print(f"  '{keyword}': {count} times")
        
        print("\nLow Priority Keywords Found:")
        for keyword in low_priority_keywords:
            count = all_text.count(keyword)
            if count > 0:
                print(f"  '{keyword}': {count} times")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not potential_priority_cols:
        print("⚠️  No priority labels detected. You'll need to:")
        print("   1. Create priority labels manually, or")
        print("   2. Use the automatic labeling based on keywords")
    
    if not text_columns:
        print("⚠️  No clear text columns detected. Please check column names.")
    else:
        print("✅ Text columns found - ready for ML classification")
    
    # Suggest column mapping
    print(f"\n📋 SUGGESTED COLUMN MAPPING:")
    for col in df.columns:
        col_lower = col.lower()
        if 'title' in col_lower:
            print(f"  '{col}' → Issue Title")
        elif 'body' in col_lower or 'description' in col_lower:
            print(f"  '{col}' → Issue Body/Description")
        elif 'url' in col_lower or 'link' in col_lower:
            print(f"  '{col}' → Issue URL")
        elif 'priority' in col_lower or 'label' in col_lower:
            print(f"  '{col}' → Priority Label")
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Review the column mapping above")
    print("2. If you have priority labels, proceed with ML training")
    print("3. If no priority labels, use keyword-based initial labeling")
    print("4. Run the main classifier script to train your model")
    
    return df

# Usage example
if __name__ == "__main__":
    # Explore your data
    df = explore_github_issues_data('github_issues_sample')
    
    if df is not None:
        print(f"\n" + "="*50)
        print("Data exploration complete! 🎉")
        print("You can now use this information to configure the ML classifier.")