import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = 'elon_musk_model.keras'
TOKENIZER_PATH = 'tokenizer.pickle'
MAX_SEQ_LEN = 57  

print("Loading model and tokenizer...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)


def sample_with_temp(preds, temperature=0.7):
    """
    Higher temp = more creative/random.
    Lower temp = more confident/repetitive.
    0.7 to 0.8 is the 'Sweet Spot' for Elon tweets.
    """
    preds = np.asarray(preds).astype('float64')
   
    preds = np.log(preds + 1e-7) / temperature 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_elon_text(seed_text, next_words=20, temp=0.7):
    result = seed_text
    
    for _ in range(next_words):
        
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=MAX_SEQ_LEN-1, padding='pre')
        
        
        predictions = model.predict(token_list, verbose=0)[0]
        
        
        idx = sample_with_temp(predictions, temp)
        
       
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == idx:
                output_word = word
                break
        
        if not output_word: break
        result += " " + output_word
        
    return result


print("\n" + "="*40)
print("ELON AI GENERATOR (v2: No-Loop Edition)")
print("="*40)

seeds = ["The future of", "Tesla is", "Mars will be"]

for s in seeds:
    
    print(f"\n[PROMPT]: {s}")
    print(f"[AI]: {generate_elon_text(s, next_words=15, temp=0.7)}")