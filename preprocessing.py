# currently this python file involves 3 steps right now to do their function
#step1- it consist of cleaning the file in this where we extract the required data from csv file 
#and clean it by using regex library that remove special character,urls ,https etc so that we get array of words

#step2 - now after cleaning it we do the tokenization means converts the words into the numbers format so we can feed it to our model
# here we first initialize the tokenizer and after we convert it and save it to use this later in our model

#step3 - here we do sequencing that is we convert the words into number and store in in an input_sequence array
# here we run a loop for all the sequences and since they can be of different length we are using the padded sequence in which we are using the pre padding
# and at last we are defining x and y where x is all the words except the last word and y is the last word for our model that we define later in model.py file
 





import pandas as pd
import re
import pickle 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


df = pd.read_csv("all_musk_posts.csv")

a = df[['fullText']]
clean_text = []

for tweet in df["fullText"].astype(str):
    tweet = tweet.lower()

    tweet = re.sub(r'https?://S+|www\.\S+','',tweet)

    tweet = re.sub(r'[@#]\S+','',tweet)

    tweet = re.sub(r'[^a-z\s]','',tweet)

    tweet = " ".join(tweet.split())

    if len(tweet)>0:
        clean_text.append(tweet)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_text)
total_words = len(tokenizer.word_index)+1

with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)


input_sequences = []
for line in clean_text:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,maxlen = max_sequence_len,padding='pre'))

X = input_sequences[:,:-1]
y = input_sequences[:,-1]

print(f"Data Cleaning:{len(df)} -> {len(clean_text)} tweets")
print(f"Vocaoblary size:{total_words}")
print(f"Total sequences{len(X)}")
print(f"maximum sequence length:{max_sequence_len}")

np.save("X_data.npy",X)
np.save("y_data.npy",y)