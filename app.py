#!/usr/bin/env python
# Simple Python Flask server to run the audio processing APIs
# Place this file in your backend directory and run: python app.py

from flask import Flask, request, jsonify
import os
import sys
import json
sys.path.insert(0, os.path.dirname(__file__))

from process_stems import split_audio
from analyze_tempo import analyze_tempo

app = Flask(__name__)

@app.route('/api/stems', methods=['POST'])
def stems():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temp file
        temp_path = f'/tmp/{file.filename}'
        file.save(temp_path)
        
        # Process
        result = split_audio(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tempo', methods=['POST'])
def tempo():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temp file
        temp_path = f'/tmp/{file.filename}'
        file.save(temp_path)
        
        # Process
        result = analyze_tempo(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
