import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class PromptGenerator:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
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
            
            # Generate the prompt using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Based on this Excel file structure, generate a detailed prompt:\n{context}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
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