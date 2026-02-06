"""
Advanced Data Loader for Complex and Unstructured CSV/XLSX Data
Handles various data types, missing values, encoding issues, and data quality problems
"""

import pandas as pd
import numpy as np
import chardet
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AdvancedDataLoader:
    """
    Advanced data loader that can handle:
    - Multiple file formats (CSV, XLSX, XLS)
    - Various encodings
    - Missing values and inconsistent data types
    - Unstructured/messy data
    - Complex headers and multi-line headers
    - Mixed data types in columns
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.encoding = None
        self.data_quality_report = {}
        self.original_dtypes = {}
        self.cleaned_columns = []
        
    def detect_encoding(self) -> str:
        """Detect file encoding for CSV files"""
        try:
            with open(self.file_path, 'rb') as f:
                raw_data = f.read(100000)  # Read first 100KB
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                # Fallback encodings if confidence is low
                if confidence < 0.7:
                    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                        try:
                            with open(self.file_path, 'r', encoding=enc) as test_file:
                                test_file.read(1000)
                            encoding = enc
                            break
                        except:
                            continue
                
                self.encoding = encoding
                return encoding
        except Exception as e:
            print(f"Encoding detection failed: {e}, using utf-8")
            self.encoding = 'utf-8'
            return 'utf-8'
    
    def load_file(self) -> pd.DataFrame:
        """Load file with intelligent parsing"""
        file_ext = self.file_path.lower().split('.')[-1]
        
        try:
            if file_ext == 'csv':
                self.df = self._load_csv_smart()
            elif file_ext in ['xlsx', 'xls']:
                self.df = self._load_excel_smart()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            return self.df
        
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def _load_csv_smart(self) -> pd.DataFrame:
        """Smart CSV loading with multiple fallback strategies"""
        encoding = self.detect_encoding()
        
        # Try different loading strategies
        strategies = [
            # Strategy 1: Standard load
            {'sep': ',', 'encoding': encoding, 'on_bad_lines': 'skip'},
            # Strategy 2: Auto-detect separator
            {'sep': None, 'encoding': encoding, 'on_bad_lines': 'skip', 'engine': 'python'},
            # Strategy 3: Tab-separated
            {'sep': '\t', 'encoding': encoding, 'on_bad_lines': 'skip'},
            # Strategy 4: Semicolon-separated (European format)
            {'sep': ';', 'encoding': encoding, 'on_bad_lines': 'skip'},
            # Strategy 5: Pipe-separated
            {'sep': '|', 'encoding': encoding, 'on_bad_lines': 'skip'},
            # Strategy 6: UTF-8 fallback
            {'sep': ',', 'encoding': 'utf-8', 'on_bad_lines': 'skip', 'encoding_errors': 'ignore'},
            # Strategy 7: Latin-1 fallback
            {'sep': ',', 'encoding': 'latin-1', 'on_bad_lines': 'skip'},
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                df = pd.read_csv(self.file_path, **strategy)
                if len(df.columns) > 1 and len(df) > 0:  # Valid dataframe
                    print(f"✓ CSV loaded successfully using strategy {i+1}")
                    return df
            except Exception as e:
                if i == len(strategies) - 1:
                    raise Exception(f"All CSV loading strategies failed: {str(e)}")
                continue
        
        raise Exception("Failed to load CSV file")
    
    def _load_excel_smart(self) -> pd.DataFrame:
        """Smart Excel loading with error handling"""
        try:
            # Try reading all sheets and combine if multiple
            excel_file = pd.ExcelFile(self.file_path)
            
            if len(excel_file.sheet_names) == 1:
                # Single sheet
                df = pd.read_excel(self.file_path, sheet_name=0)
            else:
                # Multiple sheets - try to intelligently select or combine
                print(f"Found {len(excel_file.sheet_names)} sheets: {excel_file.sheet_names}")
                
                # Load the first sheet by default
                df = pd.read_excel(self.file_path, sheet_name=0)
                
                # Store info about other sheets
                self.data_quality_report['multiple_sheets'] = True
                self.data_quality_report['sheet_names'] = excel_file.sheet_names
                self.data_quality_report['using_sheet'] = excel_file.sheet_names[0]
            
            return df
        
        except Exception as e:
            # Fallback: try with openpyxl engine
            try:
                df = pd.read_excel(self.file_path, engine='openpyxl')
                return df
            except:
                raise Exception(f"Excel loading failed: {str(e)}")
    
    def clean_column_names(self):
        """Clean and standardize column names"""
        if self.df is None:
            return
        
        original_cols = self.df.columns.tolist()
        cleaned_cols = []
        
        for col in original_cols:
            # Convert to string
            col_str = str(col)
            
            # Remove special characters, keep alphanumeric and underscores
            col_clean = re.sub(r'[^a-zA-Z0-9_\s]', '', col_str)
            
            # Replace spaces with underscores
            col_clean = col_clean.strip().replace(' ', '_')
            
            # Convert to lowercase
            col_clean = col_clean.lower()
            
            # Remove multiple underscores
            col_clean = re.sub(r'_+', '_', col_clean)
            
            # Remove leading/trailing underscores
            col_clean = col_clean.strip('_')
            
            # Ensure non-empty
            if not col_clean:
                col_clean = f'column_{len(cleaned_cols)}'
            
            # Ensure uniqueness
            if col_clean in cleaned_cols:
                counter = 1
                while f"{col_clean}_{counter}" in cleaned_cols:
                    counter += 1
                col_clean = f"{col_clean}_{counter}"
            
            cleaned_cols.append(col_clean)
        
        self.df.columns = cleaned_cols
        self.cleaned_columns = list(zip(original_cols, cleaned_cols))
    
    def infer_and_convert_types(self):
        """Intelligently infer and convert column types"""
        if self.df is None:
            return
        
        self.original_dtypes = self.df.dtypes.to_dict()
        
        for col in self.df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            # Try to convert to numeric
            try:
                # Remove common non-numeric characters
                temp_series = self.df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '').str.strip()
                numeric_converted = pd.to_numeric(temp_series, errors='coerce')
                
                # If more than 70% are valid numbers, convert
                valid_ratio = numeric_converted.notna().sum() / len(numeric_converted)
                if valid_ratio > 0.7:
                    self.df[col] = numeric_converted
                    continue
            except:
                pass
            
            # Try to convert to datetime
            try:
                datetime_converted = pd.to_datetime(self.df[col], errors='coerce')
                valid_ratio = datetime_converted.notna().sum() / len(datetime_converted)
                
                if valid_ratio > 0.7:
                    self.df[col] = datetime_converted
                    continue
            except:
                pass
            
            # Try to convert to boolean
            try:
                unique_vals = self.df[col].dropna().unique()
                if len(unique_vals) <= 3:
                    bool_mapping = {
                        'yes': True, 'no': False, 'y': True, 'n': False,
                        'true': True, 'false': False, 't': True, 'f': False,
                        '1': True, '0': False, 1: True, 0: False
                    }
                    
                    if all(str(v).lower() in bool_mapping for v in unique_vals):
                        self.df[col] = self.df[col].map(lambda x: bool_mapping.get(str(x).lower(), x))
                        continue
            except:
                pass
            
            # Keep as object/string for categorical or text data
    
    def handle_missing_values(self) -> Dict[str, Any]:
        """Analyze and handle missing values"""
        if self.df is None:
            return {}
        
        missing_info = {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
                
                # Handle missing values based on data type
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # For numeric: fill with median (robust to outliers)
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    # For datetime: forward fill
                    self.df[col] = self.df[col].ffill()
                else:
                    # For categorical/text: fill with mode or 'Unknown'
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                    else:
                        self.df[col].fillna('Unknown', inplace=True)
        
        return missing_info
    
    def remove_duplicates(self) -> int:
        """Remove duplicate rows"""
        if self.df is None:
            return 0
        
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.df)
        
        return removed_rows
    
    def detect_outliers(self) -> Dict[str, List[int]]:
        """Detect outliers in numeric columns using IQR method"""
        if self.df is None:
            return {}
        
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ].index.tolist()
            
            if outlier_indices:
                outliers[col] = {
                    'count': len(outlier_indices),
                    'percentage': round((len(outlier_indices) / len(self.df)) * 100, 2),
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                }
        
        return outliers
    
    def generate_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        if self.df is None:
            return {}
        
        report = {
            'file_info': {
                'path': self.file_path,
                'encoding': self.encoding,
                'rows': len(self.df),
                'columns': len(self.df.columns)
            },
            'column_info': {
                'original_names': [c[0] for c in self.cleaned_columns] if self.cleaned_columns else self.df.columns.tolist(),
                'cleaned_names': self.df.columns.tolist(),
                'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            'data_quality': {
                'missing_values': self.data_quality_report.get('missing_values', {}),
                'duplicates_removed': self.data_quality_report.get('duplicates_removed', 0),
                'outliers': self.data_quality_report.get('outliers', {})
            },
            'statistics': {}
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['statistics']['numeric'] = self.df[numeric_cols].describe().to_dict()
        
        # Add value counts for categorical columns (top 5 values)
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        report['statistics']['categorical'] = {}
        for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
            value_counts = self.df[col].value_counts().head(5).to_dict()
            report['statistics']['categorical'][col] = {
                'unique_values': int(self.df[col].nunique()),
                'top_values': value_counts
            }
        
        return report
    
    def process(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete data processing pipeline
        Returns: (processed_dataframe, quality_report)
        """
        print("🔄 Starting advanced data processing...")
        
        # Step 1: Load file
        print("  📂 Loading file...")
        self.load_file()
        print(f"  ✓ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Step 2: Clean column names
        print("  🧹 Cleaning column names...")
        self.clean_column_names()
        
        # Step 3: Infer and convert types
        print("  🔍 Inferring data types...")
        self.infer_and_convert_types()
        
        # Step 4: Handle missing values
        print("  🔧 Handling missing values...")
        missing_info = self.handle_missing_values()
        self.data_quality_report['missing_values'] = missing_info
        
        # Step 5: Remove duplicates
        print("  🗑️  Removing duplicates...")
        dup_removed = self.remove_duplicates()
        self.data_quality_report['duplicates_removed'] = dup_removed
        if dup_removed > 0:
            print(f"  ✓ Removed {dup_removed} duplicate rows")
        
        # Step 6: Detect outliers
        print("  📊 Detecting outliers...")
        outliers = self.detect_outliers()
        self.data_quality_report['outliers'] = outliers
        
        # Step 7: Generate report
        print("  📋 Generating quality report...")
        quality_report = self.generate_data_quality_report()
        
        print("✅ Data processing complete!")
        
        return self.df, quality_report


def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to load and process data in one call
    
    Args:
        file_path: Path to CSV or XLSX file
    
    Returns:
        Tuple of (processed_dataframe, quality_report)
    """
    loader = AdvancedDataLoader(file_path)
    return loader.process()
