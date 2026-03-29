#first we take the input values X and y from np.save and load it here.bassicaly this np.load and save functions work as storage system that store out input as it is as we define in preprocessing.py file earlier
# in order to use that we create the ".npy file to store the input data"
# we create this because this is better than csv as this file is the binary file that require the less space,its speed to save and upload is high and last its shape is preserved

# here in the tensorflow library we use callback functions that saves the input features and calls it whenever we need it.without using it we need to start all over again it take lot of time
# in this first we use ModelCheckpoint which basically autosaves our features by using save_best_only=True which tells us that it only overwrite the save file if the current epcoh 'loss' is lower then the previous best loss value that it it updates the loss alue and we get the smartest version we needed

# then second we use EarlyStopping which use to terminate the process if the model stops to imporve itself here we define it value 3 which means that if loss doesnt get any better for 3 epochs in the row then the process need to be stop
import numpy as np
import pickle as pe
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from model import create_model

X = np.load("X_data.npy")
y = np.load("y_data.npy")


model = create_model(total_words=27324,max_sequence_len=57)

checkpoint = ModelCheckpoint("elon_musk_model.keras",monitor='loss',save_best_only=True,verbose=1)

early_stop = EarlyStopping(monitor='loss',patience=3)

model.fit(X,y,epochs=5,batch_size=256,callbacks=[checkpoint,early_stop],verbose=1)

print("it is completed")