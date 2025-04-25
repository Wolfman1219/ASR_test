from flask import Flask, render_template, request, jsonify
import os
import time
import soundfile as sf
import numpy as np
import subprocess
from silero_vad import VADIterator, load_silero_vad, read_audio, collect_chunks, get_speech_timestamps
import kenlm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import sys # For float('inf')

lm_model_PATH = 'lm_word_based_3.arpa'
VOCAB_PATH = 'vocab.txt'
MAX_WORD_LENGTH = 20  # Maximum length of a word candidate
OOV_LOG_PROB_PER_CHAR = -7.0 # Heuristic penalty (log10 probability per char) for OOV words.

try:
    lm_model = kenlm.Model(lm_model_PATH)
    print(f"KenLM lm_model loaded successfully from {lm_model_PATH}")
    print(f"lm_model Order: {lm_model.order}")
except Exception as e:
    print(f"Error loading KenLM lm_model from {lm_model_PATH}: {e}")
    sys.exit(1)

try:
    with open(VOCAB_PATH) as f:
        vocab = set(line.strip() for line in f if line.strip()) # Ensure no empty lines
    print(f"Vocabulary loaded successfully from {VOCAB_PATH}. Size: {len(vocab)}")
except Exception as e:
    print(f"Error loading vocabulary from {VOCAB_PATH}: {e}")
    sys.exit(1)

def segment_optimized(text, max_word_length=MAX_WORD_LENGTH, oov_log_prob_per_char=OOV_LOG_PROB_PER_CHAR):
    """
    Segments text using KenLM with stateful scoring and OOV handling.

    Args:
        text (str): The input text string (without spaces).
        max_word_length (int): Maximum length of candidate words.
        oov_log_prob_per_char (float): Penalty (log10 prob per char) for OOV words.

    Returns:
        str: The best segmentation found, joined by spaces.
    """
    n = len(text)
    if n == 0:
        return ""

    # DP table: Stores (min_neg_log_prob, previous_index) for reaching index i
    # Initialize with infinity cost and -1 index
    best_score = [(float('inf'), -1)] * (n + 1)
    # DP table for KenLM states: Stores the KenLM state *after* processing the
    best_state = [None] * (n + 1)
    # Base case: Start of sequence
    best_score[0] = (0.0, -1) # Cost 0 at index 0
    # Initialize with NullContext state (or BeginSentence if you want <s>)
    # NullContext is usually preferred for segmentation within a larger text.
    initial_state = kenlm.State()
    lm_model.NullContextWrite(initial_state)
    best_state[0] = initial_state
    # --- Dynamic Programming ---
    for i in range(1, n + 1):
        # Iterate through possible start positions k for a word ending at i
        for k in range(max(0, i - max_word_length), i):
            word = text[k:i].lower() # Convert to lowercase for case-insensitive matching
            prev_neg_log_prob, _ = best_score[k]

            # If the previous state is unreachable, skip
            if prev_neg_log_prob == float('inf'):
                continue

            prev_lm_state = best_state[k]
            current_neg_log_prob = float('inf')
            # word_id = -1 # Placeholder
            # Check if word is in vocabulary
            if word in vocab:
                try:
                    # Use BaseScore for stateful scoring
                    # It calculates the log10 prob of 'word' given 'prev_lm_state'
                    # and updates the state accordingly into 'current_lm_state'.
                    current_lm_state = kenlm.State()
                    log10_prob = lm_model.BaseScore(prev_lm_state, word, current_lm_state)
                    word_neg_log_prob = -log10_prob
                    current_neg_log_prob = prev_neg_log_prob + word_neg_log_prob
                except KeyError:
                    # Should not happen if word is in vocab set, but good practice
                    # Fall through to OOV handling below
                    print(f"Warning: Word '{word}' in vocab but not found by model. Treating as OOV.")
            # Handle OOV words (if not found in vocab or if BaseScore failed)
            if current_neg_log_prob == float('inf'): # Trigger OOV if not handled above
                     # Use the <unk> token score and add a length-based penalty
                     current_lm_state = kenlm.State()
                     log10_prob_unk = lm_model.BaseScore(prev_lm_state, "<unk>", current_lm_state) # Score using <unk> contextually
                     # Add heuristic penalty based on length
                     oov_penalty = - (len(word) * oov_log_prob_per_char) # Penalty is positive cost
                     word_neg_log_prob = -log10_prob_unk + oov_penalty
                     current_neg_log_prob = prev_neg_log_prob + word_neg_log_prob
                   # word_id = UNK_ID # Mark as OOV using the model's <unk> id for state tracking
                   # print(f"  -> OOV Word: '{word}', PrevCost: {prev_neg_log_prob:.2f}, WordCost: {word_neg_log_prob:.2f} (unk={-log10_prob_unk:.2f}, penalty={oov_penalty:.2f}), NewTotal: {current_neg_log_prob:.2f}")
            # Update DP table if this path is better
            if current_neg_log_prob < best_score[i][0]:
                best_score[i] = (current_neg_log_prob, k)
                best_state[i] = current_lm_state
                # print(f"  ** Update best_score[{i}]: Cost={current_neg_log_prob:.2f}, PrevIdx={k}")

    # --- Backtracking ---
    if best_score[n][0] == float('inf'):
        print("Warning: Could not find a valid segmentation for the input.")
        # Fallback: return the original text or handle error appropriately
        return text # Or perhaps raise an error or return ""
    segmented_words = []
    current_index = n
    while current_index > 0:
        neg_log_prob, prev_index = best_score[current_index]
        word = text[prev_index:current_index]
        segmented_words.append(word)
        current_index = prev_index
    # Reverse the words to get the correct order
    segmented_words.reverse()
    print(segmented_words)
    return ' '.join(segmented_words).lower()




vad_model = load_silero_vad()
SAMPLING_RATE = 16000
vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLING_RATE)
app = Flask(__name__)

# Initialize your model - update paths as needed
model_id = "/opt/ai_users/abdurakhim/whisper_fine_tuning/whisper-small-hi"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
)
processor = AutoProcessor.from_pretrained(model_id)
device = "cuda:0"
model.to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device=device,
)
# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = '/opt/ai_users/abdurakhim/paraformer_train/application/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')
import numpy as np
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
    
    # try:
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
    result = pipe(np.array(chunks))
    
    # Get the transcription result
    transcription = result['text']
    # segemented_text = segment_optimized(transcription)
    return jsonify({'transcription': transcription})
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500
    # finally:
    #     # Clean up temporary files
    #     for temp_file in [temp_input_file, temp_output_file]:
    #         if os.path.exists(temp_file):
    #             try:
    #                 os.remove(temp_file)
    #             except:
    #                 pass

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