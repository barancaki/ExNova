# Excel AI Tool

A powerful Flask-based web application that provides advanced Excel file processing capabilities with AI integration.

## Features

### 1. Excel Comparator
- Compare multiple Excel files simultaneously
- Generate detailed analysis reports
- Optimized for large files (up to 100MB)
- Chunked file reading and parallel processing
- Memory-efficient processing with garbage collection
- Automated report generation in ZIP format

### 2. Matching Data Finder
- Find matching rows across multiple Excel files
- Target specific columns for comparison
- Adjustable similarity threshold (1-100%)
- Support for both exact and fuzzy matching
- Column preview functionality
- Detailed matching reports with summary sheets
- Interactive column selection interface

### 3. Excel Prompt Writer
- AI-powered Excel file analysis
- Generate intelligent insights and summaries
- Custom prompt generation based on file content
- Integration with OpenAI's GPT-3.5 Turbo
- Context-aware responses

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd excel-ai-tool
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Excel Comparator
1. Navigate to the Excel Comparator section
2. Upload multiple Excel files (up to 100MB each)
3. Click "Compare Files"
4. Download the ZIP file containing comparison results

### Matching Data Finder
1. Go to the Matching Data Finder section
2. Upload two or more Excel files
3. Enter the column names you want to match
4. Use the column preview button to see available columns
5. Adjust the similarity threshold as needed
6. Click "Find Matching Data"
7. Download the ZIP file with matching results

### Excel Prompt Writer
1. Access the Prompt Writer section
2. Upload an Excel file
3. Enter your question or analysis request
4. Receive AI-generated insights and analysis

## Technical Details

- Built with Flask
- Uses pandas for Excel processing
- Implements Levenshtein distance for fuzzy matching
- OpenAI GPT-3.5 Turbo integration
- Optimized for performance with large files
- Memory-efficient processing
- Parallel processing capabilities
- Error handling and logging

## Requirements

- Python 3.7+
- Flask
- pandas
- openpyxl
- python-Levenshtein
- openai
- python-dotenv
- httpx

## Error Handling

The application includes comprehensive error handling for:
- File size limits
- Invalid file formats
- Processing errors
- API timeouts
- Memory constraints

## Security

- Secure file handling
- Environment variable protection
- No file storage on server
- Automatic cleanup of temporary files
- Protected API key handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
