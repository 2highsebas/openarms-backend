import sys
import os
import json
import librosa
import numpy as np

def analyze_tempo(audio_path):
    """
    Analyze audio file for tempo/BPM, key, and beat information
    Returns JSON with analysis results
    """
    try:
        print(f"Loading audio: {audio_path}")
        
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        print("Detecting tempo and beats...")
        
        # Tempo detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        
        # Get beat times in seconds
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Onset strength (tempo curve)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Tempogram for tempo variation over time
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        
        # Key detection using chromagram
        print("Detecting key...")
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Average chroma to find predominant pitch class
        chroma_vals = np.mean(chroma, axis=1)
        key_index = np.argmax(chroma_vals)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = keys[key_index]
        
        # Determine if major or minor (simplified)
        # Compare major vs minor chord profiles
        major_profile = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        minor_profile = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        
        # Rotate profiles to match detected key
        major_profile = np.roll(major_profile, key_index)
        minor_profile = np.roll(minor_profile, key_index)
        
        # Correlate with chroma
        major_corr = np.corrcoef(chroma_vals, major_profile)[0, 1]
        minor_corr = np.corrcoef(chroma_vals, minor_profile)[0, 1]
        
        scale = "Major" if major_corr > minor_corr else "Minor"
        
        # Duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Prepare result
        result = {
            "success": True,
            "bpm": round(tempo, 2),
            "key": detected_key,
            "scale": scale,
            "duration": round(duration, 2),
            "beat_count": len(beat_times),
            "beat_times": beat_times[:20].tolist(),  # First 20 beats for visualization
            "tempo_confidence": float(np.std(tempogram)),  # Lower is more consistent
        }
        
        print("Analysis complete!")
        print(json.dumps(result, indent=2))
        
        return result
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_tempo.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    result = analyze_tempo(audio_path)
    
    # Output as JSON for API consumption
    print("__RESULT_JSON__")
    print(json.dumps(result))
