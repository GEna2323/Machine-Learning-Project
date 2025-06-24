import math
import copy
import pymysql
import numpy as np
import matplotlib.pyplot as plt
from db_config import host, user, password, db_name
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
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_dev_scaled = scaler_X.transform(X_dev)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_dev_scaled = scaler_Y.transform(Y_dev.reshape(-1, 1)).flatten()

# Initialize parameters
w_init = np.zeros(X_train.shape[1])
b_init = 0

# Cost function linear
def compute_cost(x, y, w, b, lambda_):
    m = len(x)
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        cost += (f_wb_i - y[i])**2
    cost /= (2*m)
    # L2 regularization
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost


# Computer gradient linear
def compute_gradient(x, y, w, b, lambda_):
    m, n = x.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.0
    for i in range(m):
        err = ((np.dot(x[i], w) + b) - y[i])
        for j in range(n):
            dj_dw[j] += err * x[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    #L2 regularization
    dj_dw += (lambda_ / m) * w
    return dj_dw, dj_db


# Gradient descent linear
def gradient_descent(x, x_dev, y, y_, y_dev, w_in, b_in, cost_function, gradient_function, a, num_iters, regularization):
    j_train = []
    error_train = []
    error_dev = []
    w = np.array(copy.deepcopy(w_in), dtype=np.float64)
    b = b_in
    for i in range(num_iters):
        prediction = make_prediction(x, w, b)
        prediction_ = make_prediction(x_dev, w, b)
        mae_, mape_ = mae_error(prediction, y_)
        mae__, mape__ = mae_error(prediction_, y_dev)
        error_train.append(mape_)
        error_dev.append(mape__)
        dj_dw, dj_db = gradient_function(x, y, w, b, regularization)
        w -= a * dj_dw
        b -= a * dj_db
        cost = cost_function(x, y, w, b, regularization)
        j_train.append(cost)
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i}: cost = {cost:.4f}, w = {w}, b = {b:}")
    return w, b, j_train, error_train, error_dev


# Mean Absolute Error (MAE) for evaluation
def mae_error (predict, y):
    mae = np.mean(np.abs(predict - y))
    mape = np.mean(np.abs((predict - y) / y)) * 100
    return mae, mape


# Prediction on test set
def make_prediction(x, w, b):
    predictions_scaled = np.dot(x, w) + b
    predict = scaler_Y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    return predict


# Main app
def app():
    A = 0.1
    iterations = 30000
    w_final = []
    b_final = 0
    reg = input("Regularization l2 (y/n):")
    lam = 0
    if reg == "n":
        w_final, b_final, j_history, j_train, j_dev = gradient_descent(X_train_scaled, X_dev_scaled, Y_train_scaled, Y_train, Y_dev, w_init, b_init, compute_cost, compute_gradient, A, iterations, lam)
        prediction_dev = make_prediction(X_dev_scaled, w_final, b_final)
        prediction_train = make_prediction(X_train_scaled, w_final, b_final)
        for i in range(len(prediction_dev)):
            print(f"\tMy prediction/Actual on dev set: {prediction_dev[i]:.2f} : {Y_dev[i]}")
        mae_dev, mape_dev = mae_error(prediction_dev, Y_dev)
        mae_train, mape_train = mae_error(prediction_train, Y_train)
        print(f"\tMean Absolute Error J_cv(MAE): {mae_dev:.2f} = {mape_dev:.2f}%")
        print(f"\tMean Absolute Error J_train(MAE): {mae_train:.2f} = {mape_train:.2f}%")
    elif reg == "y":
        choice = input("Want to choose lambda(1) or to train(2):")
        if choice == "1":
            try:
                lam = float(input("Print lambda:"))
            except ValueError:
                return print(f"Value needs to be int or float")
            w_final, b_final, j_history, j_train, j_dev = gradient_descent(X_train_scaled, X_dev_scaled, Y_train_scaled, Y_train, Y_dev, w_init, b_init, compute_cost, compute_gradient, A, iterations, lam)
            prediction_dev = make_prediction(X_dev_scaled, w_final, b_final)
            prediction_train = make_prediction(X_train_scaled, w_final, b_final)
            for i in range(len(prediction_dev)):
                print(f"\tMy prediction/Actual on dev set: {prediction_dev[i]:.2f} : {Y_dev[i]}")
            mae_dev, mape_dev = mae_error(prediction_dev, Y_dev)
            mae_train, mape_train = mae_error(prediction_train, Y_train)
            print(f"\tMean Absolute Error J_cv(MAE): {mae_dev:.2f} = {mape_dev:.2f}%")
            print(f"\tMean Absolute Error J_train(MAE): {mae_train:.2f} = {mape_train:.2f}%")
        elif choice == "2":
            try:
                const = float(input("Print const that will add to lambda every iteration:"))
            except ValueError:
                return print(f"Value needs to be int or float")
            for i in range(10):
                print(f"\nFor lambda {lam}:")
                w_final, b_final, j_history, j_train, j_dev = gradient_descent(X_train_scaled, X_dev_scaled, Y_train_scaled, Y_train, Y_dev, w_init, b_init, compute_cost, compute_gradient, A, iterations, lam)
                prediction = make_prediction(X_dev_scaled, w_final, b_final)
                mae_, mape_ = mae_error(prediction, Y_dev)
                print(f"\tMean Absolute Error (MAE): {mae_:.2f} = {mape_:.2f}%")
                lam += const
        else:
            return print(f"No that variant: {choice}")
    else:
        return print(f"No that variant: {reg}")


    try:
        # Plot predicted vs actual
        plt.scatter(Y_dev, prediction_dev, color='blue', label='Predictions')
        plt.scatter(Y_train, prediction_train, color='red', label='Ideal')
        plt.xlabel('Prices for apartment')
        plt.ylabel('X')
        plt.title('Predicted vs Actual Prices')
        plt.legend()
        plt.show()

        # Plot j_train vs j_dev
        plt.plot(j_train, color='blue', label='J_train')
        plt.plot(j_dev, color='brown', label='J_dev')
        plt.xlabel('Iterations')
        plt.ylabel('Percent error')
        plt.title('j_train vs j_dev')
        plt.legend()
        plt.show()

        # Plot cost
        plt.plot(j_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Function Change')
        plt.yscale('log')
        plt.show()
    except UnboundLocalError:
        return print("Plot error")
app()