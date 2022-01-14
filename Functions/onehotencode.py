from sklearn.preprocessing import OneHotEncoder

def onehotencode(y):
    ohe = OneHotEncoder(sparse = False)
    y_dummy = ohe.fit_transform(y.to_numpy().reshape(-1, 1))
    return ohe, y_dummy