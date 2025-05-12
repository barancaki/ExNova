import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio
import os
from typing import List, Tuple, Dict
import tempfile

class ExcelComparator:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))

    def _preprocess_dataframe(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to a string representation for comparison."""
        # Convert all columns to string type
        df = df.astype(str)
        # Concatenate all values
        return ' '.join(df.values.flatten())

    def _calculate_structural_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Calculate structural similarity between two DataFrames."""
        # Compare shape
        shape_similarity = 1 - abs(df1.shape[0] - df2.shape[0]) / max(df1.shape[0], df2.shape[0])
        
        # Compare column names
        col_similarity = ratio(' '.join(df1.columns), ' '.join(df2.columns))
        
        return (shape_similarity + col_similarity) / 2

    def _find_matching_rows(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Find rows that are similar between two DataFrames."""
        # Convert all columns to string type for comparison
        df1_str = df1.astype(str)
        df2_str = df2.astype(str)
        
        # Get common columns
        common_cols = list(set(df1.columns) & set(df2.columns))
        
        # Initialize results
        matching_rows = []
        
        # Compare each row in df1 with each row in df2
        for idx1, row1 in df1_str.iterrows():
            for idx2, row2 in df2_str.iterrows():
                row_similarity = 0
                for col in common_cols:
                    # Calculate similarity for this column's values
                    col_similarity = ratio(str(row1[col]), str(row2[col]))
                    row_similarity += col_similarity
                
                # Average similarity across columns
                row_similarity /= len(common_cols) if common_cols else 1
                
                # If similarity is above threshold, add to matching rows
                if row_similarity >= self.similarity_threshold:
                    match_data = {
                        'File 1 Row': idx1 + 1,
                        'File 2 Row': idx2 + 1,
                        'Similarity Score': f"{row_similarity:.2%}"
                    }
                    # Add the actual values from both files
                    for col in common_cols:
                        match_data[f'File 1 - {col}'] = row1[col]
                        match_data[f'File 2 - {col}'] = row2[col]
                    matching_rows.append(match_data)
        
        return pd.DataFrame(matching_rows) if matching_rows else pd.DataFrame()

    def _calculate_content_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Calculate content similarity between two DataFrames using TF-IDF and cosine similarity."""
        text1 = self._preprocess_dataframe(df1)
        text2 = self._preprocess_dataframe(df2)
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity

    def _create_training_data_file(self, file_paths: List[str]) -> str:
        """Create an Excel file containing all input data with analysis."""
        # Create a temporary file for the training data
        training_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.xlsx',
            prefix='training_data_'
        )

        with pd.ExcelWriter(training_file.name, engine='openpyxl') as writer:
            # Sheet for file information
            file_info = []
            all_columns = set()
            file_data = {}

            # First pass: collect information about files
            for file_path in file_paths:
                df = pd.read_excel(file_path)
                file_name = os.path.basename(file_path)
                file_data[file_name] = df
                
                file_info.append({
                    'File Name': file_name,
                    'Number of Rows': len(df),
                    'Number of Columns': len(df.columns),
                    'Columns': ', '.join(df.columns),
                    'File Size (bytes)': os.path.getsize(file_path)
                })
                all_columns.update(df.columns)

            # Write file information
            pd.DataFrame(file_info).to_excel(
                writer, 
                sheet_name='File Information',
                index=False
            )

            # Write each file's data to its own sheet
            for file_name, df in file_data.items():
                sheet_name = f"Data - {file_name[:30]}"  # Limit sheet name length
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Create a column comparison sheet
            column_comparison = []
            all_columns = sorted(list(all_columns))
            for col in all_columns:
                col_info = {
                    'Column Name': col,
                    'Present in Files': ', '.join(
                        fname for fname, df in file_data.items() 
                        if col in df.columns
                    ),
                    'Unique Values Count': sum(
                        len(df[col].unique()) 
                        for df in file_data.values() 
                        if col in df.columns
                    )
                }
                column_comparison.append(col_info)

            pd.DataFrame(column_comparison).to_excel(
                writer,
                sheet_name='Column Analysis',
                index=False
            )

        return training_file.name

    def compare_files(self, file_paths: List[str]) -> Tuple[str, str]:
        """Compare multiple Excel files and return paths to results and training data files."""
        comparison_results = []
        detailed_matches = []
        
        # Compare each pair of files
        for i, file1 in enumerate(file_paths):
            for j, file2 in enumerate(file_paths[i+1:], i+1):
                try:
                    df1 = pd.read_excel(file1)
                    df2 = pd.read_excel(file2)
                    
                    structural_sim = self._calculate_structural_similarity(df1, df2)
                    content_sim = self._calculate_content_similarity(df1, df2)
                    
                    # Calculate overall similarity
                    overall_sim = (structural_sim + content_sim) / 2
                    
                    if overall_sim >= self.similarity_threshold:
                        file1_name = os.path.basename(file1)
                        file2_name = os.path.basename(file2)
                        
                        comparison_results.append({
                            'File 1': file1_name,
                            'File 2': file2_name,
                            'Structural Similarity': f"{structural_sim:.2%}",
                            'Content Similarity': f"{content_sim:.2%}",
                            'Overall Similarity': f"{overall_sim:.2%}"
                        })
                        
                        # Find matching rows between the files
                        matching_data = self._find_matching_rows(df1, df2)
                        if not matching_data.empty:
                            matching_data.insert(0, 'File 1', file1_name)
                            matching_data.insert(1, 'File 2', file2_name)
                            detailed_matches.append(matching_data)
                            
                except Exception as e:
                    print(f"Error comparing {file1} and {file2}: {str(e)}")
                    continue

        # Create results DataFrame
        results_df = pd.DataFrame(comparison_results)
        
        # Save results to temporary file
        results_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.xlsx',
            prefix='comparison_results_'
        ).name
        
        # Save training data to temporary file
        training_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.xlsx',
            prefix='training_data_'
        ).name
        
        # Add metadata and results to Excel file
        with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
            if not comparison_results:
                # Create empty DataFrame with message if no similarities found
                pd.DataFrame({
                    'Message': ['No significant similarities found between the uploaded files.']
                }).to_excel(writer, sheet_name='Comparison Results', index=False)
            else:
                # Write overall comparison results
                results_df.to_excel(writer, sheet_name='Overall Comparisons', index=False)
                
                # Write detailed matching data
                if detailed_matches:
                    pd.concat(detailed_matches, ignore_index=True).to_excel(
                        writer, 
                        sheet_name='Matching Content', 
                        index=False
                    )
            
            # Add summary sheet
            summary_data = {
                'Metric': [
                    'Number of Files Compared',
                    'Similarity Threshold',
                    'Number of File Pairs with Similarities',
                    'Total Number of Matching Rows'
                ],
                'Value': [
                    len(file_paths),
                    f"{self.similarity_threshold:.0%}",
                    len(comparison_results),
                    sum(len(df) for df in detailed_matches) if detailed_matches else 0
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        # Create training data file
        with pd.ExcelWriter(training_file, engine='openpyxl') as writer:
            # Write file information
            file_info = []
            all_columns = set()
            file_data = {}

            # First pass: collect information about files
            for file_path in file_paths:
                df = pd.read_excel(file_path)
                file_name = os.path.basename(file_path)
                file_data[file_name] = df
                
                file_info.append({
                    'File Name': file_name,
                    'Number of Rows': len(df),
                    'Number of Columns': len(df.columns),
                    'Columns': ', '.join(df.columns),
                    'File Size (bytes)': os.path.getsize(file_path)
                })
                all_columns.update(df.columns)

            pd.DataFrame(file_info).to_excel(
                writer, 
                sheet_name='File Information',
                index=False
            )

            # Write each file's data
            for file_name, df in file_data.items():
                sheet_name = f"Data - {file_name[:30]}"  # Limit sheet name length
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Write column comparison
            column_comparison = []
            all_columns = sorted(list(all_columns))
            for col in all_columns:
                col_info = {
                    'Column Name': col,
                    'Present in Files': ', '.join(
                        fname for fname, df in file_data.items() 
                        if col in df.columns
                    ),
                    'Unique Values Count': sum(
                        len(df[col].unique()) 
                        for df in file_data.values() 
                        if col in df.columns
                    )
                }
                column_comparison.append(col_info)

            pd.DataFrame(column_comparison).to_excel(
                writer,
                sheet_name='Column Analysis',
                index=False
            )
        
        return results_file, training_file 