from sklearn.model_selection import train_test_split

def split_dataset(X, y_dummy):
    x_train, x_test, y_train, y_test = train_test_split(X, y_dummy, test_size=0.2,
                                                    random_state=123, stratify=y_dummy)
    print("shape of train set: ",x_train.shape, " and labels: ",y_train.shape, "\n")
    print("shape of test set: ",x_test.shape, " and labels: ",y_test.shape, "\n")
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=123, stratify= y_train)
    print("Shape of train and validation set to check for overfitting")
    print("shape of train set: ",x_tr.shape, " and labels: ",y_tr.shape, "\n")
    print("shape of validation set: ",x_val.shape, " and labels: ",y_val.shape)
    return x_tr, y_tr, x_val, y_val, x_train, y_train, x_test, y_test