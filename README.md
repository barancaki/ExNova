# ExNova Excel AI Tool

A powerful Flask-based web application that provides advanced Excel file processing capabilities with AI integration using Google's Gemini Pro model.

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

### 3. Excel Prompt Writer with AI
- AI-powered Excel file analysis using Google's Gemini Pro
- Generate intelligent insights and summaries
- Custom prompt generation based on file content
- Smart rate limiting and retry logic
- Context-aware responses
- Automatic handling of API quotas
- Exponential backoff for API requests

## Smart API Management
- Automatic rate limiting (1 request per second minimum)
- Exponential backoff retry logic
- Graceful handling of API quota limits
- Smart request queuing
- Automatic error recovery
- Detailed error logging

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
GOOGLE_API_KEY=your_api_key_here
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
   - The system automatically handles rate limits
   - Retries are automatic if API limits are reached
   - Responses are optimized for clarity and relevance

## Technical Details

### Core Technologies
- Built with Flask
- Uses pandas for Excel processing
- Implements Levenshtein distance for fuzzy matching
- Google Gemini Pro AI integration
- Tenacity for robust retry handling
- Rate limiting and request management

### Performance Features
- Optimized for large files
- Memory-efficient processing
- Parallel processing capabilities
- Chunked file reading
- Automatic garbage collection
- Smart request queuing

### API Management
- Automatic rate limiting
- Exponential backoff retry logic
- Smart error handling
- Request queuing
- Quota management
- Detailed error logging

## Requirements

### Python and Core Dependencies
- Python 3.7+
- Flask==3.0.0
- pandas==2.1.4
- numpy==1.26.2
- scikit-learn==1.3.2

### Excel Processing
- openpyxl==3.1.2
- python-Levenshtein==0.23.0

### AI and API Integration
- google-generativeai==0.3.2
- python-dotenv==1.0.0
- httpx>=0.24.1,<1.0.0
- tenacity==8.2.3

## Error Handling

The application includes comprehensive error handling for:
- API rate limits and quotas
- File size limits
- Invalid file formats
- Processing errors
- Network timeouts
- Memory constraints
- API authentication issues

## Security

- Secure file handling
- Environment variable protection
- No file storage on server
- Automatic cleanup of temporary files
- Protected API key handling
- Rate limit protection

## Troubleshooting

### API Rate Limits
The application automatically handles rate limits by:
1. Implementing minimum delays between requests
2. Using exponential backoff for retries
3. Providing clear error messages
4. Logging detailed error information

### Common Issues
1. API Key Issues:
   - Ensure your GOOGLE_API_KEY is correctly set in .env
   - Verify API key has necessary permissions
   - Check quota limits in Google Cloud Console

2. File Processing:
   - Ensure files are valid Excel format
   - Check file size limits
   - Verify file permissions

3. Performance:
   - Monitor memory usage for large files
   - Check system resources
   - Review log files for bottlenecks

## Created By Baran Çakı



1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Ensure all tests pass
5. Consider performance implications
