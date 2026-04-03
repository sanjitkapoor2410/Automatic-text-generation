Automatic Text Generation — LSTM Tweet Style Model
Python 3.x
TensorFlow / Keras
Streamlit
LSTM
NLP
A deep learning project that trains a custom two-layer LSTM language model on social media posts and generates 
new text in the same writing style. Given a seed phrase, the model predicts subsequent words using temperature 
sampling and a repetition penalty — all served via an interactive Streamlit web app.

Project structure

1.preprocessing.py
Data cleaning, tokenization, n-gram sequence generation

2.model.py
LSTM architecture definition (Embedding → LSTM → Dropout → LSTM → Dense)

3.train.py
Model training with ModelCheckpoint and EarlyStopping

4.test.py
CLI script for quick generation testing

5.app.py
Interactive Streamlit web application

Generated files (after training): elon_musk_model.keras, tokenizer.pickle, X_data.npy, y_data.npy


Model architecture

Embedding
vocab → 100-dim vectors

LSTM (150)
return_sequences=True

Dropout
rate = 0.2

LSTM (100)
context summary

Dense
softmax over vocab

Parameter	Value

Vocabulary size:	27,324 words
Max sequence length:	57 tokens
Optimizer:	adam
Loss:	sparse_categorical_crossentropy
Batch size:	256
Max epochs:	5 (with early stopping, patience=3)


Getting started

1. Install dependencies
pip install tensorflow streamlit numpy pandas

2. Prepare the dataset
Place your CSV file (must have a fullText column) in the project root as all_musk_posts.csv, then run:

python preprocessing.py
This generates tokenizer.pickle, X_data.npy, and y_data.npy.

3. Train the model
python train.py
Saves the best model automatically as elon_musk_model.keras.

4. Test via CLI
python test.py

6. Launch the web app
streamlit run app.py

How generation works

1.Tokenize the seed-
The input phrase is converted to a token sequence and padded to MAX_SEQ_LEN - 1.

2.Predict next word-
The model outputs a probability distribution over all ~27k words in the vocabulary.

3.Apply repetition penalty-
Probabilities for recently-used words are divided by penalty × count to reduce looping.

4.Temperature sampling-
Probabilities are scaled by temperature and sampled via multinomial distribution — controlling creativity.

5.Append and repeat-
The chosen word is added to the output, and the process repeats until the target word count is reached.

#Tech stack:
TensorFlow
Keras
NumPy
Pandas
Streamlit
Pickle


#Dataset:
The model was trained on all_musk_posts.csv — a dataset of social media posts. 
The dataset is not included in this repository. 
You can substitute any CSV file with a fullText column to train the model on a different writing style.

