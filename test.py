import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = 'elon_musk_model.keras'
TOKENIZER_PATH = 'tokenizer.pickle'
MAX_SEQ_LEN = 57

def load_env():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError("Files missing.")
        
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def get_next_word(preds, temp=0.7):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_text(model, tokenizer, seed, length=15, temp=0.8, penalty=1.5):
    sentence = seed
    history = seed.lower().split()

    for _ in range(length):
        tokens = tokenizer.texts_to_sequences([sentence])[0]
        tokens = pad_sequences([tokens], maxlen=MAX_SEQ_LEN-1, padding='pre')
        
        preds = model.predict(tokens, verbose=0)[0]

        for word, index in tokenizer.word_index.items():
            if word in history:
                preds[index] /= (penalty * history.count(word))
        
        preds = preds / np.sum(preds)
        idx = get_next_word(preds, temp)
        
        word_found = None
        for word, i in tokenizer.word_index.items():
            if i == idx:
                word_found = word
                break
        
        if not word_found:
            break
            
        if len(history) > 0 and word_found == history[-1]:
            continue

        sentence += " " + word_found
        history.append(word_found)
        
    return sentence

if __name__ == "__main__":
    m, t = load_env()
    
    seeds = ["The future of", "Tesla is", "Mars will be"]

    for s in seeds:
        res = generate_text(m, t, s)
        print(f"\nPrompt: {s}")
        print(f"Result: {res}")