from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential,load_model
from keras.layers.convolutional import Conv1D
from tensorflow.keras.optimizers import Adam
from set_seeds import reset_random_seeds

def cnn_model(inp_dim, out_dim):
  reset_random_seeds()
  model = Sequential()
  model.add(Conv1D(filters=27, kernel_size=4, activation= "relu", input_shape=(inp_dim,1)))
  model.add(Flatten())
  model.add(Dense(units=384, activation='relu'))
  model.add(Dropout(rate=0.0))
  model.add(Dense(units=192, activation='relu'))
  model.add(Dropout(rate=0.4))
  model.add(Dense(out_dim, activation="softmax"))
  optimizer = Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  return model