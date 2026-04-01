import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(page_title="AI Tweet Generator")


@st.cache_resource
def load_assets():
    model = load_model('elon_musk_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()
MAX_SEQ_LEN = 57


def sample_with_temp(preds, temperature=0.7):
    """Adjusts randomness of the next word choice."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


st.sidebar.header("Generation Settings")
temp = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.5, 0.8, help="Higher is more creative, lower is more predictable.")
rep_penalty = st.sidebar.slider("Anti-Loop (Repetition Penalty)", 1.0, 2.0, 1.2, help="Higher values prevent the AI from repeating the same words.")


st.title(" AI Generator")
st.markdown("This AI mimics  writing style using a **Deep Learning LSTM model**.")


seed_text = st.text_input("Enter a starting phrase:", "The future")
next_words = st.slider("Number of words to generate:", 5, 50, 20)

if st.button("Generate Tweet"):
    with st.spinner("AI is thinking..."):
        output_text = seed_text
        generated_words = seed_text.lower().split()
        
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([output_text])[0]
            token_list = pad_sequences([token_list], maxlen=MAX_SEQ_LEN-1, padding='pre')
            
            
            predictions = model.predict(token_list, verbose=0)[0]
            
           
            for word, index in tokenizer.word_index.items():
                if word in generated_words:
                    count = generated_words.count(word)
                    predictions[index] /= (rep_penalty * count)
            
            
            predictions = predictions / np.sum(predictions)
            idx = sample_with_temp(predictions, temp)
            
          
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == idx:
                    output_word = word
                    break
            
            if not output_word:
                break
                
           
            if len(generated_words) > 0 and output_word == generated_words[-1]:
                continue 

            output_text += " " + output_word
            generated_words.append(output_word)
        
        st.success("### Generated Text:")
        st.write(f"*{output_text}*")


st.divider()
st.info("Built with TensorFlow, Keras, and Streamlit.")