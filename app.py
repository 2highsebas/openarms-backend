#!/usr/bin/env python
# Simple Python Flask server to run the audio processing APIs
# Place this file in your backend directory and run: python app.py

import os
import sys
import json

# Force CPU-only execution for PyTorch/Demucs
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS

from process_stems import split_audio
from analyze_tempo import analyze_tempo

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/')
def health():
    return jsonify({'status': 'ok', 'message': 'Backend is running'}), 200

@app.route('/api/stems', methods=['POST'])
def stems():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        import base64
        import tempfile
        
        # Save temp file in a safe location
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, file.filename)
            output_dir = os.path.join(temp_dir, 'stems')
            
            file.save(input_path)
            
            # Process
            split_audio(input_path, output_dir)
            
            # Read and encode stems as base64
            stems_data = {}
            for stem_name in ['vocals', 'drums', 'bass', 'other']:
                stem_file = os.path.join(output_dir, f'{stem_name}.wav')
                if os.path.exists(stem_file):
                    with open(stem_file, 'rb') as f:
                        stems_data[stem_name] = base64.b64encode(f.read()).decode('utf-8')
            
            return jsonify(stems_data)
    except Exception as e:
        import traceback
        print(f"ERROR in /api/stems: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/tempo', methods=['POST'])
def tempo():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        import tempfile
        
        # Save temp file in a safe location
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, file.filename)
            file.save(audio_path)
            
            # Process
            result = analyze_tempo(audio_path)
            
            return jsonify(result)
    except Exception as e:
        import traceback
        print(f"ERROR in /api/tempo: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
