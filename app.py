from flask import Flask, request, render_template
import os  # For file path manipulation
import google.generativeai as genai
import json
import time
app = Flask(__name__)
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
        text = "OCR this image. Generate Metadata. Give both results in JSON format for further analysis.Format the OCR text to presentable format.  Include the file name "+filepath +" in metadata in addition to your metadata. Also include the "+ get_file_stat(filepath)  +" the metadata in addition"
        os.remove(filepath)
        response = model.generate_content([text, sample_file])
        return render_template('index.html', output=response.text)

    # Render the form on GET request
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
