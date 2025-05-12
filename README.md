# Excel AI Comparison Tool

A powerful web-based tool that combines Excel file analysis with AI capabilities. This application provides advanced file comparison features and AI-powered prompt generation for Excel data analysis.

## Main Features

### Excel Comparison
- Upload and compare multiple Excel files simultaneously
- AI-powered similarity analysis using TF-IDF and cosine similarity
- Structural and content-based comparison
- Detailed matching analysis with row-level comparisons
- Support for multiple Excel file formats (.xlsx, .xls)

### AI Features
- AI-powered prompt generation for Excel analysis
- Intelligent data summarization
- GPT-3.5 Turbo integration for custom data analysis
- Context-aware responses based on Excel content

### User Interface
- Modern web-based interface
- Drag-and-drop file upload
- Real-time processing feedback
- Downloadable ZIP results
- Separate prompt generator interface

## Output Files

### ZIP Package Contents
1. **comparison_results.xlsx**
   - Overall Comparisons Sheet (similarity scores)
   - Matching Content Sheet (detailed matches)
   - Summary Sheet (analysis overview)
   - Similarity metrics (structural, content, overall percentage)

2. **training_data.xlsx**
   - File Information Sheet (metadata)
   - Individual Data Sheets (per file)
   - Column Analysis Sheet (cross-file analysis)

3. **prompt.txt** (if prompt was provided)
   - Contains the analysis prompt used

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps
1. Clone the repository
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

### Running the Application
```bash
python app.py
```
Access the application at `http://localhost:5000`

## Usage Guide

### Excel Comparison
1. Navigate to the home page
2. Upload two or more Excel files using drag-and-drop
3. (Optional) Add an analysis prompt
4. Click "Compare Files"
5. Download and extract the results ZIP file

### Prompt Generation
1. Go to the Prompt Generator page
2. Upload a single Excel file
3. Receive AI-generated analysis prompts
4. View file summary and suggested prompts

## Technical Details

### Core Technologies
- **Flask**: Web framework
- **Pandas**: Excel processing
- **Scikit-learn**: ML algorithms
- **OpenAI GPT-3.5**: AI analysis
- **Python-Levenshtein**: String comparisons

### Analysis Methods
- TF-IDF vectorization
- Cosine similarity metrics
- Levenshtein distance
- Custom Excel-specific algorithms

## Limitations and Constraints

### File Restrictions
- Maximum file size: 16MB per file
- Supported formats: .xlsx, .xls
- Minimum files for comparison: 2
- Maximum files: No hard limit (performance-dependent)

### API Limits
- OpenAI API rate limits apply
- Token limits for GPT-3.5 Turbo responses

## Security and Error Handling

### Security Features
- Secure filename handling
- Temporary file cleanup
- Input validation and sanitization
- Environment variable protection

### Error Management
- Detailed error messages
- Graceful failure handling
- Automatic cleanup of temporary files
- Invalid format detection

## Support and Troubleshooting

### Common Issues
- File size exceeded
- Unsupported file format
- API key configuration
- Memory limitations

### Best Practices
- Use clean, well-formatted Excel files
- Keep file sizes reasonable
- Ensure proper column headers
- Back up original files before processing

## License
[Your License Information Here]

## Contributing
[Your Contribution Guidelines Here] 