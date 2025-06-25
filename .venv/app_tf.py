import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import math
import copy
import pymysql
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from db_config import host, user, password, db_name
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler



# DataBase
try:
    connection = pymysql.connect(
        host=host,
        port=3306,
        user=user,
        password=password,
        database=db_name,
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * from apartments")
            rows = cursor.fetchall()
            arr_x = []
            arr_y = []
            X = []
            for row in rows:
                arr_x.append(row['city'])
                arr_x.append(row['area'])
                arr_x.append(row['rooms'])
                arr_y.append(row['price'])
                X.append(arr_x)
                arr_x = []
            X = np.array(X)
            Y = np.array(arr_y)
    finally:
        connection.close()
    print(f"Successfully connected to {db_name}")
except Exception as ex:
    print(f"Connection refused \n{ex}")


# Input features
X_train = np.concatenate([X[0:30], X[50:80], X[100:130], X[150:180]])
Y_train = np.concatenate([Y[0:30], Y[50:80], Y[100:130], Y[150:180]])
X_dev = np.concatenate([X[30:40], X[80:90], X[130:140], X[180:190]])
Y_dev = np.concatenate([Y[30:40], Y[80:90], Y[130:140], Y[180:190]])
X_test = np.concatenate([X[40:50], X[90:100], X[140:150], X[190:200]])
Y_test = np.concatenate([Y[40:50], Y[90:100], Y[140:150], Y[190:200]])

# Normalization
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_dev_scaled = scaler_Y.transform(Y_dev.reshape(-1, 1)).flatten()
Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()


def tensorflow_model(x_train, y_train, x_dev, y_dev, x_test, y_test, lambda_):
    #Normalizing data
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(x_train)
    x_train_n = norm_l(x_train)
    x_dev_n = norm_l(x_dev)
    x_test_n = norm_l(x_test)

    tf.random.set_seed(1234)
    model = Sequential(
        [
            Dense(20, activation='relu', name='layer1', kernel_regularizer=l2(lambda_)),
            Dense(5, activation='relu', name='layer2', kernel_regularizer=l2(lambda_)),
            Dense(1, name='layer3', kernel_regularizer=l2(lambda_))
        ]
    )

    model.compile(
        loss = MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
    )

    model.fit(
        x_train_n, y_train,
        validation_data=(x_dev_n, y_dev),
        epochs=100,
    )

    model.summary()
    predictions = model.predict(x_test_n)
    predictions = scaler_Y.inverse_transform(predictions).flatten()
    y_test_orig = scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    loss, mae, mape = model.evaluate(x_test_n, y_test, verbose=2)
    return model, predictions, y_test_orig, loss, mae, mape


def app():
    try:
        lam = float(input("Print lambda:"))
        model, predictions, y_test_orig, loss, mae, mape = tensorflow_model(X_train, Y_train_scaled, X_dev, Y_dev_scaled, X_test, Y_test_scaled, lam)
        print(f"Predictions: {predictions[:5]}")
        print(f"True Values: {y_test_orig[:5]}")
        mae_orig = np.mean(np.abs(predictions - y_test_orig))
        mape_orig = np.mean(np.abs((predictions - y_test_orig) / y_test_orig)) * 100
        print(f"MAE in original scale: {mae_orig}")
        print(f"MAPE in original scale: {mape_orig}%")
    except ValueError:
        return print("Lambda needs to be int or float")

app()