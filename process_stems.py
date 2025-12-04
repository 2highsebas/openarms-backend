import sys
import os
import torch
import soundfile as sf
import math

def split_audio(input_path, output_path):
    """AI-powered audio stem separation using Demucs.
    Returns True if high quality separation succeeded, False if fallback used."""
    
    # Clean output directory to prevent cached/glitched stems
    import shutil
    if os.path.exists(output_path):
        print(f"Cleaning old stems from: {output_path}")
        shutil.rmtree(output_path)
    
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting stem separation for: {input_path}")
    print(f"Output directory: {output_path}")

    separation_done = False
    fallback_used = False

    # Primary Demucs path
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import numpy as np
        try:
            import librosa
        except ImportError:
            print("librosa not installed. Install with: pip install librosa")
            raise
        
        print("Loading Demucs model...")
        model = get_model('htdemucs')
        model.cpu()
        model.eval()
        
        print("Loading audio file...")
        # Load audio using soundfile (more reliable)
        audio_data, sr = sf.read(input_path, always_2d=True)
        
        # Convert to torch tensor and transpose to [channels, samples]
        wav = torch.from_numpy(audio_data.T).float()
        
        # Resample if needed using librosa (avoid torchaudio/torchcodec dependency)
        target_sr = model.samplerate
        if sr != target_sr:
            print(f"Resampling from {sr}Hz to {target_sr}Hz using librosa...")
            # librosa expects shape (samples, channels) or mono; convert accordingly
            audio_np = wav.numpy().T  # shape (samples, channels)
            resampled_channels = []
            for ch in range(audio_np.shape[1]):
                resampled = librosa.resample(audio_np[:, ch], orig_sr=sr, target_sr=target_sr)
                resampled_channels.append(resampled)
            # Pad channels to same length (due to rounding differences)
            max_len = max(len(c) for c in resampled_channels)
            for i,c in enumerate(resampled_channels):
                if len(c) < max_len:
                    resampled_channels[i] = np.pad(c, (0, max_len - len(c)))
            resampled_np = np.stack(resampled_channels, axis=0)  # (channels, samples)
            wav = torch.from_numpy(resampled_np).float()
            sr = target_sr
        
        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        
        print("Separating audio into stems (this may take a minute)...")
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(model, wav[None], device='cpu')[0]
        
        # sources shape: [stems, channels, samples]
        # htdemucs outputs: drums, bass, other, vocals (in that order)
        stem_names = ['drums', 'bass', 'other', 'vocals']
        
        print("Saving separated stems...")
        for i, name in enumerate(stem_names):
            output_file = os.path.join(output_path, f"{name}.wav")
            # Convert to numpy and save
            audio_data = sources[i].cpu().numpy()
            sf.write(output_file, audio_data.T, sr)
            print(f"Saved {name}.wav")
        separation_done = True
        
        # Verify stems are actually different
        print("Verifying stem separation quality...")
        vocals_data = sources[3].cpu().numpy()  # vocals index
        drums_data = sources[0].cpu().numpy()   # drums index
        mix_estimate = sources.sum(0).cpu().numpy()
        
        # Check if stems are different
        are_different = not np.allclose(vocals_data, drums_data, rtol=0.1)
        # Compute simple SNR between vocals and drums
        diff_power = np.mean((vocals_data - drums_data)**2)
        drums_power = np.mean(drums_data**2) + 1e-9
        snr = 10 * math.log10(drums_power / diff_power) if diff_power > 0 else float('inf')
        print(f"SNR (drums vs vocals): {snr:.2f} dB")
        # Correlation with mixture
        def corr(a,b):
            return float(np.corrcoef(a.flatten(), b.flatten())[0,1])
        print(f"Corr(mix, vocals): {corr(mix_estimate, vocals_data):.3f}")
        print(f"Corr(mix, drums): {corr(mix_estimate, drums_data):.3f}")
        if are_different:
            print("OK: Stems are successfully separated.")
        else:
            print("WARNING: Validation suggests stems may be similar. Keeping results anyway.")

    except Exception as e:
        print(f"ERROR during Demucs separation: {e}")
        import traceback; traceback.print_exc()

    # Only invoke fallback if Demucs path failed before saving stems
    if not separation_done:
        print("Fallback: simple frequency-based pseudo-separation.")
        try:
            from pydub import AudioSegment
            from pydub.effects import low_pass_filter, high_pass_filter
            
            audio = AudioSegment.from_file(input_path)
            
            # Vocals (mid-high frequencies with voice range)
            vocals = high_pass_filter(audio, 200)
            vocals = low_pass_filter(vocals, 3000)
            vocals.export(os.path.join(output_path, "vocals.wav"), format="wav")
            
            # Drums (transients and high frequencies)
            drums = high_pass_filter(audio, 60)
            drums.export(os.path.join(output_path, "drums.wav"), format="wav")
            
            # Bass (very low frequencies)
            bass = low_pass_filter(audio, 250)
            bass.export(os.path.join(output_path, "bass.wav"), format="wav")
            
            # Other (full spectrum)
            audio.export(os.path.join(output_path, "other.wav"), format="wav")
            
            print("Fallback separation complete (approximate, not true source separation).")
            fallback_used = True
        except Exception as fallback_error:
            print(f"Fallback failed: {fallback_error}")
            print("Creating duplicate files as last resort.")
            import shutil
            for stem in ['vocals', 'drums', 'bass', 'other']:
                try:
                    shutil.copy(input_path, os.path.join(output_path, f"{stem}.wav"))
                except Exception as copy_error:
                    print(f"Failed to copy {stem}: {copy_error}")
            fallback_used = True

    print("Separation pipeline finished.")
    return separation_done and not fallback_used

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_stems.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    split_audio(input_path, output_path)
