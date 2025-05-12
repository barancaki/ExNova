import os
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from excel_comparator import ExcelComparator
import zipfile
import tempfile
import shutil
from pathlib import Path
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return 'No files uploaded', 400
    
    files = request.files.getlist('files')
    if len(files) < 2:
        return 'Please upload at least 2 files for comparison', 400

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

if __name__ == '__main__':
    app.run(debug=True) 