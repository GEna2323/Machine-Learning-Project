import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout


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



# model = Sequential([
#     Dense(3, activation="sigmoid"),
#     Dense(1, activation="sigmoid")
# ])
#
# predictions = model(X1[:1]).numpy()
# tf.nn.softmax(predictions).numpy()
#
# model.compile(
#     loss = BinaryCrossentropy,
#     optimizer = 'adam'
# )
# model.fit(
#     X1, Y1, epochs = 5
# )
#
# model.predict(X_test)
# model.evaluate(X_test, Y_test, verbose=2)


# Load data
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0


#Normalizing data
X_normalized = X.copy()
for j in range(X.shape[1]):
    X_normalized[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_normalized)


# Ініціалізація параметрів
w_init = np.zeros(X_poly.shape[1])
b_init = 20000

def tensorflow_model(x, y):
    #Normalizing data (tensorflow)
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(x)  # learns mean, variance
    xn = norm_l(x)


    tf.random.set_seed(1234)
    model = Sequential(
        [
            Dense(20, activation='relu', name='layer1'),
            Dense(4, activation='relu', name='layer2'),
            Dense(1, activation='relu', name='layer3')
        ]
    )


    model.compile(
            loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )


    model.fit(
        xn, y,
        epochs = 10,
    )


    model.summary()
    return


def app():
    A = 0.001
    iterations = 100000
    f_wb = 0.0
    try:
        choice = int(input("Choose 1(simple regression) or 2(tensorflow) to train model:"))
    except ValueError:
        print(f"You can write only a number")
        return

    if choice == 1:
        reg = input("Activate regularization l2 (y/n):")
        w_final, b_final, j_history = gradient_descent(X_poly, Y, w_init, b_init, compute_cost, compute_gradient, A, iterations, reg)
        #App code
        x_str = input("Введи масив значень: [Пробіг, Витрата палива, Час розгону, Рік, М/А]:")
        x_test = [float(x) for x in x_str.split(',')]
        #Normalizing my input data
        x_test_normalized = np.array(x_test)
        for j in range(len(x_test_normalized)):
            x_test_normalized[j] = (x_test[j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
        x_test_poly = poly.transform([x_test_normalized])
        f_wb = np.dot(x_test_poly[0], w_final) + b_final
        print(f"Передбачена ціна: ${f_wb:.2f}")
    elif choice == 2:
        tensorflow_model(X, Y)
        return
    else:
        print(f"There is no variant {choice}, you can only choose 1 or 2!")
        return


    with open("model tests.txt", "r+", encoding='utf-8') as file:
        file.write(str(f_wb))

    # My plot (cost / iterations)
    plt.plot(j_history)
    plt.xlabel('Ітерація')
    plt.ylabel('Cost')
    plt.title('Зміна функції втрат для цін автомобілів')
    plt.yscale('log')
    plt.show()
app()