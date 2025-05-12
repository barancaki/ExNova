# Excel AI Comparison Tool

This tool allows users to upload multiple Excel files and compare them using AI-powered similarity analysis. The system analyzes the content and structure of Excel files and identifies similarities between them, providing detailed analysis in two output Excel files.

## Features

- Web-based interface for easy file upload
- Support for multiple Excel file formats (.xlsx, .xls)
- AI-powered similarity analysis using:
  - TF-IDF vectorization for content analysis
  - Cosine similarity metrics
  - Levenshtein distance for structural comparison
- Comprehensive output in a ZIP file containing:
  - Comparison results file
  - Training data analysis file

## Output Files

The tool generates a ZIP file containing two Excel files:

### 1. comparison_results.xlsx
- **Overall Comparisons Sheet**: Shows similarity scores between file pairs
  - Structural similarity
  - Content similarity
  - Overall similarity percentage
- **Matching Content Sheet**: Detailed view of matching rows
  - Row numbers from both files
  - Actual matching content
  - Similarity scores for each match
- **Summary Sheet**: Analysis overview
  - Number of files compared
  - Similarity threshold used
  - Number of file pairs with similarities
  - Total number of matching rows

### 2. training_data.xlsx
- **File Information Sheet**: Metadata about each uploaded file
  - File names
  - Number of rows and columns
  - Column names
  - File sizes
- **Data Sheets**: One sheet per input file showing complete content
- **Column Analysis Sheet**: Cross-file column analysis
  - List of all unique columns
  - Files containing each column
  - Unique value counts per column

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Open the web interface in your browser
2. Upload two or more Excel files using the drag-and-drop interface
3. Click "Compare Files" to start the analysis
4. Download the ZIP file containing the results
5. Extract the ZIP to view both the comparison results and training data analysis

## Technical Details

The system uses:
- **Pandas**: For Excel file processing and data manipulation
- **Scikit-learn**: For TF-IDF vectorization and similarity analysis
- **Flask**: For web interface and file handling
- **Python-Levenshtein**: For string similarity calculations
- **Custom similarity metrics**: For Excel-specific comparisons

### Similarity Analysis
- **Structural Similarity**: Compares file shapes and column names
- **Content Similarity**: Uses TF-IDF and cosine similarity
- **Row-level Matching**: Identifies similar rows across files
- **Column Analysis**: Tracks column presence across files

## File Size Limits
- Maximum file size: 16MB per file
- Supported formats: .xlsx, .xls

## Error Handling
- Invalid file format detection
- Proper cleanup of temporary files
- Detailed error messages for troubleshooting

## Security Features
- Secure filename handling
- Temporary file cleanup
- Input validation 