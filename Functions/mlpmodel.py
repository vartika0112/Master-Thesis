from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.optimizers import Adam
from set_seeds import reset_random_seeds

def mlp_model(inp_dim, out_dim):
  reset_random_seeds()
  model = Sequential()
  model.add(Dense(units=448, input_dim=inp_dim, activation='relu'))
  model.add(Dropout(rate=0.3))
  model.add(Dense(units=384, activation='relu'))
  model.add(Dropout(rate=0.3))
  model.add(Dense(units=160, activation='relu'))
  model.add(Dense(out_dim, activation='softmax'))
  optimizer = Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  return model