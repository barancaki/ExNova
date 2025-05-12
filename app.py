import os
from flask import Flask, request, render_template, send_file, jsonify, url_for
from werkzeug.utils import secure_filename
from excel_comparator import ExcelComparator
import zipfile
import tempfile
import shutil
from pathlib import Path
import io
from prompt_generator import PromptGenerator
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import httpx

load_dotenv()

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize OpenAI client with custom httpx client
http_client = httpx.Client()
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    http_client=http_client
)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prompt-writer')
def prompt_writer():
    return render_template('prompt_generator.html')

@app.route('/get-ai-response', methods=['POST'])
def get_ai_response():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    prompt = request.form.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    if len(files) < 1:
        return jsonify({'error': 'Please upload at least one file'}), 400

    try:
        # Process each Excel file and create context
        excel_contexts = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read Excel file
                df = pd.read_excel(filepath)
                
                # Create context for this file
                context = f"File: {filename}\n"
                context += f"Columns: {', '.join(df.columns.tolist())}\n"
                context += f"Total Rows: {len(df)}\n"
                context += f"Sample Data (first 5 rows):\n{df.head().to_string()}\n\n"
                
                excel_contexts.append(context)
                
                # Clean up
                os.remove(filepath)

        if not excel_contexts:
            return jsonify({'error': 'No valid Excel files processed'}), 400

        # Combine all contexts
        full_context = "Excel Files Analysis:\n\n" + "\n".join(excel_contexts)

        # Get response from OpenAI using GPT-3.5-turbo
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing Excel data. Provide clear, concise answers about the Excel files."},
                {"role": "user", "content": f"Context about the Excel files:\n\n{full_context}\n\nUser Question: {prompt}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Return the AI response
        return jsonify({
            'response': response.choices[0].message.content
        })

    except Exception as e:
        import traceback
        print("Error details:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return 'No files uploaded', 400
    
    files = request.files.getlist('files')
    prompt = request.form.get('prompt', '')
    
    if len(files) < 1:
        return 'Please upload at least one file for processing', 400

    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filenames.append(filepath)

    if not filenames:
        return 'No valid Excel files uploaded', 400

    try:
        # Initialize comparator and process files
        comparator = ExcelComparator()
        results_file, training_file = comparator.compare_files(filenames)
        
        # Clean up uploaded files
        for filepath in filenames:
            try:
                os.remove(filepath)
            except:
                pass

        try:
            # Create ZIP file in memory
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add the results file
                with open(results_file, 'rb') as f:
                    zf.writestr('comparison_results.xlsx', f.read())
                
                # Add the training file
                with open(training_file, 'rb') as f:
                    zf.writestr('training_data.xlsx', f.read())
                
                # Add the prompt to a text file
                if prompt:
                    zf.writestr('prompt.txt', prompt)

            # Clean up the individual Excel files
            os.remove(results_file)
            os.remove(training_file)

            # Seek to the beginning of the memory file
            memory_file.seek(0)

            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name='excel_analysis_results.zip'
            )

        except Exception as e:
            print(f"Error creating ZIP: {str(e)}")
            raise

    except Exception as e:
        # Print the full error for debugging
        import traceback
        print("Error details:", traceback.format_exc())
        return f"An error occurred: {str(e)}", 500

@app.route('/generate-prompt', methods=['POST'])
def generate_prompt():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.xlsx'):
        return jsonify({'error': 'Only Excel files (.xlsx) are allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate prompt
        generator = PromptGenerator()
        result = generator.process_file(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'summary': result['summary'],
            'generated_prompt': result['generated_prompt']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/prompt-generator')
def prompt_generator_page():
    return render_template('prompt_generator.html')

if __name__ == '__main__':
    app.run(debug=True) 