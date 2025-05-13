import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio
import os
from typing import List, Tuple, Dict
import tempfile
from concurrent.futures import ThreadPoolExecutor
import gc

class ExcelComparator:
    def __init__(self, similarity_threshold: float = 0.7, chunk_size: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))

    def _read_excel_in_chunks(self, file_path: str) -> pd.DataFrame:
        """Read Excel file in chunks to handle large files."""
        print(f"Reading file: {os.path.basename(file_path)}")
        try:
            # First try reading with default engine
            df = pd.read_excel(file_path)
            print(f"Successfully read file using default engine: {len(df)} rows")
            return df
        except Exception as e:
            print(f"Failed to read with default engine, trying chunked reading: {str(e)}")
            try:
                # If file is too large, read in chunks
                chunks = []
                chunk_count = 0
                for chunk in pd.read_excel(file_path, chunksize=self.chunk_size):
                    chunks.append(chunk)
                    chunk_count += 1
                    if chunk_count % 10 == 0:  # Log progress every 10 chunks
                        print(f"Read {chunk_count} chunks...")
                    gc.collect()  # Force garbage collection after each chunk
                
                result = pd.concat(chunks, ignore_index=True)
                print(f"Successfully read file in chunks: {len(result)} rows")
                return result
            except Exception as e:
                print(f"Failed to read file in chunks: {str(e)}")
                raise

    def _preprocess_dataframe(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to a string representation for comparison."""
        # Process in chunks for large DataFrames
        if len(df) > self.chunk_size:
            chunks = [df[i:i + self.chunk_size] for i in range(0, len(df), self.chunk_size)]
            processed = []
            for chunk in chunks:
                chunk_str = ' '.join(chunk.astype(str).values.flatten())
                processed.append(chunk_str)
                gc.collect()  # Force garbage collection after each chunk
            return ' '.join(processed)
        
        # For smaller DataFrames, process normally
        df = df.astype(str)
        return ' '.join(df.values.flatten())

    def _calculate_structural_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Calculate structural similarity between two DataFrames."""
        # Compare shape
        shape_similarity = 1 - abs(df1.shape[0] - df2.shape[0]) / max(df1.shape[0], df2.shape[0])
        
        # Compare column names
        col_similarity = ratio(' '.join(df1.columns), ' '.join(df2.columns))
        
        return (shape_similarity + col_similarity) / 2

    def _find_matching_rows(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Find rows that are similar between two DataFrames using parallel processing."""
        df1_str = df1.astype(str)
        df2_str = df2.astype(str)
        common_cols = list(set(df1.columns) & set(df2.columns))
        matching_rows = []

        def process_chunk(chunk_data):
            chunk_matches = []
            chunk_df1, start_idx1, chunk_df2, start_idx2 = chunk_data
            
            for i, row1 in chunk_df1.iterrows():
                for j, row2 in chunk_df2.iterrows():
                    row_similarity = 0
                    for col in common_cols:
                        col_similarity = ratio(str(row1[col]), str(row2[col]))
                        row_similarity += col_similarity
                    
                    row_similarity /= len(common_cols) if common_cols else 1
                    
                    if row_similarity >= self.similarity_threshold:
                        match_data = {
                            'File 1 Row': start_idx1 + i + 1,
                            'File 2 Row': start_idx2 + j + 1,
                            'Similarity Score': f"{row_similarity:.2%}"
                        }
                        for col in common_cols:
                            match_data[f'File 1 - {col}'] = row1[col]
                            match_data[f'File 2 - {col}'] = row2[col]
                        chunk_matches.append(match_data)
            
            return chunk_matches

        # Split DataFrames into chunks for parallel processing
        chunks1 = [df1_str[i:i + self.chunk_size] for i in range(0, len(df1_str), self.chunk_size)]
        chunks2 = [df2_str[i:i + self.chunk_size] for i in range(0, len(df2_str), self.chunk_size)]

        # Prepare chunk combinations for parallel processing
        chunk_combinations = []
        for i, chunk1 in enumerate(chunks1):
            for j, chunk2 in enumerate(chunks2):
                chunk_combinations.append((
                    chunk1, 
                    i * self.chunk_size,
                    chunk2,
                    j * self.chunk_size
                ))

        # Process chunks in parallel
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(process_chunk, chunk_combinations))

        # Combine results
        for chunk_matches in chunk_results:
            matching_rows.extend(chunk_matches)
            gc.collect()  # Force garbage collection

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
        """Compare multiple Excel files with improved memory management."""
        comparison_results = []
        detailed_matches = []
        
        total_comparisons = len(file_paths) * (len(file_paths) - 1) // 2
        print(f"Total number of comparisons to perform: {total_comparisons}")
        comparison_count = 0
        
        # Compare each pair of files
        for i, file1 in enumerate(file_paths):
            for j, file2 in enumerate(file_paths[i+1:], i+1):
                comparison_count += 1
                print(f"Processing comparison {comparison_count}/{total_comparisons}: {os.path.basename(file1)} vs {os.path.basename(file2)}")
                
                try:
                    # Read files in chunks with error handling
                    try:
                        print(f"Reading file 1: {os.path.basename(file1)}")
                        df1 = self._read_excel_in_chunks(file1)
                        print(f"Successfully read file 1: {len(df1)} rows")
                    except Exception as e:
                        print(f"Error reading file 1 ({os.path.basename(file1)}): {str(e)}")
                        continue

                    try:
                        print(f"Reading file 2: {os.path.basename(file2)}")
                        df2 = self._read_excel_in_chunks(file2)
                        print(f"Successfully read file 2: {len(df2)} rows")
                    except Exception as e:
                        print(f"Error reading file 2 ({os.path.basename(file2)}): {str(e)}")
                        continue

                    print("Calculating structural similarity...")
                    structural_sim = self._calculate_structural_similarity(df1, df2)
                    print(f"Structural similarity: {structural_sim:.2%}")

                    print("Calculating content similarity...")
                    content_sim = self._calculate_content_similarity(df1, df2)
                    print(f"Content similarity: {content_sim:.2%}")
                    
                    # Calculate overall similarity
                    overall_sim = (structural_sim + content_sim) / 2
                    print(f"Overall similarity: {overall_sim:.2%}")
                    
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
                        
                        print("Finding matching rows...")
                        matching_data = self._find_matching_rows(df1, df2)
                        if not matching_data.empty:
                            print(f"Found {len(matching_data)} matching rows")
                            matching_data.insert(0, 'File 1', file1_name)
                            matching_data.insert(1, 'File 2', file2_name)
                            detailed_matches.append(matching_data)
                        else:
                            print("No matching rows found")
                    
                    # Clear memory after each comparison
                    del df1, df2
                    gc.collect()
                            
                except Exception as e:
                    print(f"Error comparing {os.path.basename(file1)} and {os.path.basename(file2)}: {str(e)}")
                    import traceback
                    print("Comparison error details:", traceback.format_exc())
                    continue

        print("Creating result files...")
        # Create results file with optimized memory usage
        results_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.xlsx',
            prefix='comparison_results_'
        ).name

        print("Creating training data file...")
        # Create training data file
        training_file = self._create_training_data_file(file_paths)

        print("Writing results to Excel...")
        # Write results in chunks
        with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
            if not comparison_results:
                pd.DataFrame({
                    'Message': ['No significant similarities found between the uploaded files.']
                }).to_excel(writer, sheet_name='Comparison Results', index=False)
            else:
                # Write overall comparison results
                pd.DataFrame(comparison_results).to_excel(
                    writer, 
                    sheet_name='Overall Comparisons',
                    index=False
                )
                
                # Write detailed matching data in chunks
                if detailed_matches:
                    print(f"Writing {len(detailed_matches)} detailed matches...")
                    for i, chunk in enumerate(detailed_matches):
                        chunk_size = 10000  # Adjust based on available memory
                        for j in range(0, len(chunk), chunk_size):
                            chunk_df = chunk[j:j + chunk_size]
                            if i == 0 and j == 0:
                                chunk_df.to_excel(
                                    writer,
                                    sheet_name='Matching Content',
                                    index=False
                                )
                            else:
                                chunk_df.to_excel(
                                    writer,
                                    sheet_name='Matching Content',
                                    startrow=j + 1,
                                    header=False,
                                    index=False
                                )
                            gc.collect()  # Force garbage collection after each chunk

        print("Comparison process completed successfully")
        return results_file, training_file 