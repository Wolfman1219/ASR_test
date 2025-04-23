from symspellpy import SymSpell
import kenlm
import sys # For float('inf')

# --- Configuration ---
lm_model_PATH = 'lm_word_based_3.arpa'
VOCAB_PATH = 'vocab.txt'
MAX_WORD_LENGTH = 20  # Maximum length of a word candidate
OOV_LOG_PROB_PER_CHAR = -7.0 # Heuristic penalty (log10 probability per char) for OOV words.
                               # Adjust this value based on your language/lm_model.
                               # More negative = higher penalty = less likely to choose OOV.

# --- Load Resources ---
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

# Get the ID for the unknown word token from the lm_model
# UNK_ID = lm_model.unk()
# print(f"lm_model's Unknown Word ID (<unk>): {UNK_ID}")



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
    print(text)
    n = len(text)
    if n == 0:
        return ""

    # DP table: Stores (min_neg_log_prob, previous_index) for reaching index i
    # Initialize with infinity cost and -1 index
    best_score = [(float('inf'), -1)] * (n + 1)

    # DP table for KenLM states: Stores the KenLM state *after* processing the
    # word ending at index i along the best path found so far.
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
                    # print(f"  -> Vocab Word: '{word}', PrevCost: {prev_neg_log_prob:.2f}, WordCost: {word_neg_log_prob:.2f}, NewTotal: {current_neg_log_prob:.2f}")

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

    # --- Final Score Calculation (Optional) ---
    # You can calculate the final sentence score if needed, handling </s>
    # final_state = best_state[n]
    # final_score_neg_log_prob = best_score[n][0]
    # Optionally add EOS score:
    # eos_state = kenlm.State() # Doesn't matter, only need score
    # log10_prob_eos = model.BaseScore(final_state, "</s>", eos_state)
    # final_score_neg_log_prob -= log10_prob_eos
    # print(f"Final Segmentation Score (Negative Log10 Prob): {final_score_neg_log_prob}")

    print(segmented_words)
    return ' '.join(segmented_words).lower()


from flask import Flask, render_template, request, jsonify
import os
import time
import soundfile as sf
import numpy as np
from funasr import AutoModel


app = Flask(__name__)

# Initialize your model - update paths as needed
model = AutoModel(
    model="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_uzbek_natural_v1",
    model_path="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_uzbek_natural_v1/model.pt.best", 
    config_path="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_uzbek_natural_v1", 
    config_name="config.yaml"
)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
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
    
    # Save the uploaded file temporarily
    timestamp = int(time.time())
    filename = f"{UPLOAD_FOLDER}/recording_{timestamp}.wav"
    file.save(filename)
    
    try:
        # Process with your model
        result = model.generate(
            input=filename,
            batch_size_s=300,
            hotword=hotword if hotword else None
        )
        
        # Get the transcription result
        transcription = result[0]['text']
        # input_line = "m a n a r a q i t l a r h a m m a n a p a u n z d o r s d e g a n j o y d a k õ r a b y o n y y p p i i n a ş u p p u u z o r f d a i n g l i z l a r b o y g g i z l l r r n i k o n g i l i v a d a g i r a k e t a l a r n i i ş l a t a d i b i r i n ç i m a r z a m o n a m a r z a m o n a v i r r a k e t a b o r k i b õ z a r a k e t t a g a õ x ş a g a n a a u y y r r a a b t u n n i r f r a n s s z z l a r n i b i r n i m a s s a a s r g a a õ b b r r a d i z k i i ç i n a g i n n i n g l i z z i n m a s i i g a q a r a n g t e x n o l o g i y a n d a h ş a t g a t u ş a t t a a u n i k õ r i i r a k e t t a n v a y r o t i l i ş i n i k õ r i b".replace(' ', '')

        segemented_text = segment_optimized(transcription)
        
        # Clean up the temporary file
        os.remove(filename)
        
        return jsonify({'transcription': segemented_text})
    except Exception as e:
    # Clean up the temporary file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=51515)
