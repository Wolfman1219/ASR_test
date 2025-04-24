from flask import Flask, render_template, request, jsonify
import os
import time
import soundfile as sf
import numpy as np
from funasr import AutoModel
import subprocess
import tempfile
from silero_vad import VADIterator, load_silero_vad, read_audio, collect_chunks, get_speech_timestamps

vad_model = load_silero_vad()
SAMPLING_RATE = 16000
vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLING_RATE)
app = Flask(__name__)

# Initialize your model - update paths as needed
model = AutoModel(
    model="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_v1_old",
    model_path="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_v1_old/model.pt.best", 
    config_path="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_v1_old", 
    config_name="config.yaml"
)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = '/opt/ai_users/abdurakhim/paraformer_train/application/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Check if the post request has the audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Get optional hotword parameter
    hotword = request.form.get('hotword', '')
    
    # Create temporary files
    timestamp = int(time.time())
    temp_input_file = f"{UPLOAD_FOLDER}/recording_input_{timestamp}{os.path.splitext(file.filename)[1]}"
    temp_output_file = f"{UPLOAD_FOLDER}/recording_output_{timestamp}.wav"
    
    try:
    # Save the uploaded file temporarily
        file.save(temp_input_file)
        
        # Convert to WAV with 16kHz mono if it's not already
        convert_to_wav_mono_16k(temp_input_file, temp_output_file)
        
        wav = read_audio(temp_output_file, sampling_rate=SAMPLING_RATE)

        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            return_seconds=False,  # Return speech timestamps in seconds (default is samples)
            )
        if len(speech_timestamps) == 0:
            return jsonify({'error': 'No speech detected in the audio'}), 400
        chunks = collect_chunks(speech_timestamps, wav)
        # Process with your model
        result = model.generate(
            input=chunks,
            batch_size_s=300,
            hotword=hotword if hotword else None
        )
        
        # Get the transcription result
        transcription = result[0]['text']
        
        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        for temp_file in [temp_input_file, temp_output_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def convert_to_wav_mono_16k(input_file, output_file):
    """Convert audio to WAV format with mono channel and 16kHz sample rate"""
    try:
        # First try using soundfile if the format is supported
        data, samplerate = sf.read(input_file)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample to 16kHz if needed
        if samplerate != 16000:
            # Simple resampling - in production, use a proper resampling library
            # like librosa or scipy for better quality
            data = resample(data, samplerate, 16000)
        
        # Save as WAV
        sf.write(output_file, data, 16000, subtype='PCM_16')
    except Exception as e:
        # If soundfile fails, try using ffmpeg as fallback
        try:
            subprocess.run([
                'ffmpeg', '-i', input_file,
                '-ac', '1',  # mono
                '-ar', '16000',  # 16kHz
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-y',  # overwrite output file
                output_file
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as ffmpeg_error:
            raise Exception(f"Audio conversion failed: {str(ffmpeg_error)}")
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install FFmpeg or check if it's in your PATH.")

def resample(audio_data, orig_sr, target_sr):
    """
    Basic resampling function - for production use a proper library
    like librosa or scipy for better quality
    """
    # Simple linear resampling - not ideal but works for demo
    scale = target_sr / orig_sr
    n_samples = int(len(audio_data) * scale)
    resampled = np.interp(
        np.linspace(0, len(audio_data) - 1, n_samples),
        np.arange(len(audio_data)),
        audio_data
    )
    return resampled

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=51515)