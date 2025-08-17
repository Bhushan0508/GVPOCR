from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os  # For file path manipulation
import google.generativeai as genai
import json
import time
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = '/tmp'
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_file_stat(file_path):
    try:
        print(file_path)
        # Get all file details using os.stat()
        file_stats = os.stat(file_path)

        print(f"File Size: {file_stats.st_size} bytes")
        print(f"Permissions (octal): {oct(file_stats.st_mode)}")
        print(f"Last Modified Time: {time.ctime(file_stats.st_mtime)}")
        print(f"Last Accessed Time: {time.ctime(file_stats.st_atime)}")
        print(f"Creation Time: {time.ctime(file_stats.st_ctime)}") # On Unix, this is the last metadata change time
        print(f"Inode Number: {file_stats.st_ino}")
        file_stats_dict = {
            'size': file_stats.st_size,
            'mode': file_stats.st_mode,
            'inode': file_stats.st_ino,
            'device': file_stats.st_dev,
            'links': file_stats.st_nlink,
            'uid': file_stats.st_uid,
            'gid': file_stats.st_gid,
            'last_access_time': file_stats.st_atime,
            'last_modification_time': file_stats.st_mtime,
            'creation_time': file_stats.st_ctime
        }
        json_string = json.dumps(file_stats_dict, indent=4)
        print(json_string)
        return json_string
    except Exception as e:
        print(e)
        return ""
    
def read_prompt():
    try:
        with open('prompt.txt', 'r') as file:
            prompt = file.read()
        return prompt
    except FileNotFoundError:
        print("Prompt file not found.")
        return ""
    except Exception as e:
        print(f"An error occurred while reading the prompt: {e}")
        return ""

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'input' not in request.files:
            return render_template('index.html', output="") # Redirect to same page if no file
        # Save image
        file = request.files['input']
        print(file)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Use genai
        sample_file = genai.upload_file(path=filepath)
        #os.remove(filepath)
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        
        base_prompt = read_prompt().strip()
        if not base_prompt:
            base_prompt = "OCR this image. Generate Metadata. Give both results in JSON format for further analysis. Format the OCR text to presentable format. "
        # Add file path info to prompt
        prompt = f"{base_prompt}\nInclude the file name '{filepath}' in the metadata in addition to your metadata. Also include the file size, permissions, last modified time, and creation time in the metadata.\n{get_file_stat(filepath)}"
        os.remove(filepath)
        print("Prompt text:", prompt)
        response = model.generate_content([prompt, sample_file])
        return render_template('index.html', output=response.text)

    # Render the form on GET request
    return render_template('index.html')


@app.route('/para', methods=['POST'])
def call_another_service():
    """
    This route receives a JSON payload with a 'text' key,
    sends it to the Gemini API for processing, and returns the response.
    It expects a POST request with a JSON body like: {"text": "Your prompt here"}
    """
    try:
        # Get the JSON data from the incoming request.
        incoming_data = request.get_json()
        if not incoming_data or 'text' not in incoming_data:
            return jsonify({"error": "Request body must be JSON and contain a 'text' key."}), 400

        # Call the Gemini API
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        prompt = f"<|USER|>Please modify the text to generate paragraphs and also add suitable paragraph headers for the text \"{incoming_data['text']}\"<|ASSISTANT|>"

        response = model.generate_content([prompt])
        print("Response from Gemini API:", response.text)
        # The response object has a `text` attribute for the content.
        return jsonify({"response": response.text}), 200

    except json.JSONDecodeError:
        # Handle JSON decoding errors (e.g., if the request body is not valid JSON)
        app.logger.error("Invalid JSON in request body.")
        return jsonify({"error": "Invalid JSON in request body."}), 400  # Bad Request
    except KeyError:
        # This will be caught if 'text' is not in incoming_data
        app.logger.error("Missing 'text' key in JSON payload.")
        return jsonify({"error": "Missing 'text' key in JSON payload."}), 400
    except Exception as e:
        # Handle other errors, like from the Gemini API itself.
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"An error occurred with the generative AI service: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
