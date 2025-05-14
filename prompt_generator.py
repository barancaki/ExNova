import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

class PromptGenerator:
    def __init__(self):
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        # Use the most capable model
        self.model = genai.GenerativeModel('gemini-pro')
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits by waiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_with_retry(self, prompt):
        """Generate content with retry logic."""
        self._wait_for_rate_limit()
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "RATE_LIMIT_EXCEEDED" in str(e):
                print("Rate limit exceeded, retrying after backoff...")
                raise  # This will trigger the retry
            raise  # Re-raise other exceptions
        
    def analyze_excel(self, file_path):
        """Analyze an Excel file and extract key information."""
        try:
            df = pd.read_excel(file_path)
            
            # Get basic information about the dataset
            columns = df.columns.tolist()
            sample_data = df.head(5).to_dict()
            data_types = df.dtypes.to_dict()
            
            # Create a dataset summary
            summary = {
                'total_rows': len(df),
                'total_columns': len(columns),
                'columns': columns,
                'data_types': {str(k): str(v) for k, v in data_types.items()},
                'sample_data': sample_data
            }
            
            return summary
        except Exception as e:
            raise Exception(f"Error analyzing Excel file: {str(e)}")
    
    def generate_prompt(self, excel_summary):
        """Generate an AI prompt based on the Excel file analysis."""
        try:
            # Create a detailed prompt for the AI
            system_message = """You are an AI assistant specialized in analyzing Excel data. 
            Based on the provided Excel file structure and content, generate a comprehensive prompt 
            that will help users effectively work with this data."""
            
            # Convert the summary into a readable format
            context = f"""
            Excel File Analysis:
            - Total Rows: {excel_summary['total_rows']}
            - Total Columns: {excel_summary['total_columns']}
            - Columns: {', '.join(excel_summary['columns'])}
            - Data Types: {excel_summary['data_types']}
            """
            
            # Generate the prompt using Gemini with retry logic
            prompt = f"{system_message}\n\nBased on this Excel file structure, generate a detailed prompt:\n{context}"
            return self._generate_with_retry(prompt)
        
        except Exception as e:
            raise Exception(f"Error generating prompt: {str(e)}")
    
    def process_file(self, file_path):
        """Process an Excel file and generate a prompt."""
        summary = self.analyze_excel(file_path)
        prompt = self.generate_prompt(summary)
        return {
            'summary': summary,
            'generated_prompt': prompt
        } 