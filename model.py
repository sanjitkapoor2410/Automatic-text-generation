#in this model.py file we create an sequential model where it take previous input like total_words and max_sequence_len to create the Neural Network
# 1. embidding layer stores the number in the 100 dimensional space
# 2. then we add an lstm layer from left to right and it remembers the beginnning while looks at the end
# in this retur_sequnence truw maens that it tells us that this layer pass its entire memory to the next layer
# 3. dropout layer helps to prevent overfitting as it value is 0.2 means during each training step it turns off the 20% os the neurons and that the remaining data will learn the patterns
# 4. the second lstm layer will the content from first lstm layer and use it to take the context to predict the next word
# 5.the dense layer is the decision layer where it has total_words that we get from preprocessing.py file and apply the softmax activation function
# 6. at last we compile the model and get our desire output

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding,Dense,Dropout
import tensorflow as tf

def create_model(total_words,max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words,100))

    model.add(LSTM(150,return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(100))

    model.add(Dense(total_words,activation='softmax'))

    model.build(input_shape=(None,max_sequence_len-1))

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

if __name__=='__main__':
    test_model = create_model(1000,20)
    test_model.summary()