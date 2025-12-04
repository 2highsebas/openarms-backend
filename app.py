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
        print("=== Stem separation request received ===")
        
        if 'file' not in request.files:
            print("ERROR: No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"Processing file: {file.filename}")
        
        import base64
        import tempfile
        
        # Save temp file in a safe location
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, file.filename)
            output_dir = os.path.join(temp_dir, 'stems')
            
            print(f"Saving to: {input_path}")
            file.save(input_path)
            
            # Verify file was saved
            if not os.path.exists(input_path):
                raise Exception("Failed to save uploaded file")
            
            file_size = os.path.getsize(input_path)
            print(f"File saved successfully: {file_size} bytes")
            
            # Process
            print("Starting stem separation...")
            result = split_audio(input_path, output_dir)
            print(f"Separation complete. Success: {result}")
            
            # Read and encode stems as base64
            stems_data = {}
            missing_stems = []
            
            for stem_name in ['vocals', 'drums', 'bass', 'other']:
                stem_file = os.path.join(output_dir, f'{stem_name}.wav')
                if os.path.exists(stem_file):
                    stem_size = os.path.getsize(stem_file)
                    print(f"Reading {stem_name}.wav ({stem_size} bytes)")
                    with open(stem_file, 'rb') as f:
                        stems_data[stem_name] = base64.b64encode(f.read()).decode('utf-8')
                    print(f"Encoded {stem_name}: {len(stems_data[stem_name])} chars")
                else:
                    missing_stems.append(stem_name)
                    print(f"WARNING: Missing {stem_name}.wav")
            
            if missing_stems:
                raise Exception(f"Missing stems: {', '.join(missing_stems)}")
            
            print("=== Stem separation successful ===")
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
