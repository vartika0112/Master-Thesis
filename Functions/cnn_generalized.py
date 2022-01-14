import os
import time as tm
from set_seeds import reset_random_seeds
from plot import acc_loss_plot
from results import result
#from onehotencode import onehotencode
#import load_dataset
#from split_dataset import split_dataset
from cnnmodel import cnn_model
#import mlp_generalized


def run_cnn(x_tr, y_tr, x_val, y_val, x_train, y_train, x_test, y_test, ohe):
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    model = cnn_model(input_dim, output_dim)
    # train the model to check for overfitting
    print("Train the model to check for overfitting")
    history = model.fit(x_tr, y_tr, validation_data=(x_val,y_val), epochs=2, batch_size=20, verbose=0)
    acc_loss_plot(history)
    # train the final model
    print("train the final model")
    start = tm.time()
    model.fit(x_train, y_train, epochs=2)
    training_time = tm.time()-start
    print('Training time:', training_time)
    # predict the labels of test set
    start = tm.time()
    y_pred = model.predict(x_test, verbose=False)
    test_time = tm.time()-start
    print('Predict time:', test_time)
    print("The results are as follows:")
    # convert the dummy labels back into original labels
    inv_pred = ohe.inverse_transform(y_pred).ravel()
    inv_true = ohe.inverse_transform(y_test).ravel()
    # get the results
    result(inv_true, inv_pred)