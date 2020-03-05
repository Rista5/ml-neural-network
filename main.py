import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def create_model(input_shape: int, num_of_classes=7, dense_neurons=64):
    input = keras.layers.Input(shape=(input_shape))
    layer1 = keras.layers.Dense(dense_neurons, activation=keras.activations.relu, name='dense1')(input)
    layer2 = keras.layers.Dense(dense_neurons, activation=keras.activations.relu, name='dense2')(layer1)
    layer3 = keras.layers.Dense(num_of_classes, activation=keras.activations.softmax, name='dense3')(layer2)

    model = keras.models.Model(inputs=input, outputs=layer3)

    opt = keras.optimizers.Adam(0.001)
    opt2 = keras.optimizers.RMSprop()
    # keras.metrics.Accuracy
    # keras.metrics.TopKCategoricalAccuracy()
    # keras.metrics.categorical_crossentropy
    # keras.metrics
    model.compile(optimizer=opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def convert_categorical_values_to_numerical(df: pd.DataFrame, columns: []):
    for col in columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    return df


def split_train_validation_set(df: pd.DataFrame, validation_percent=0.1):
    size = df.shape[0]
    mask = np.random.rand(size) > validation_percent
    train = df[mask]
    test = df[~mask]
    return train, test


def accuracy(predictions: np.ndarray, real_values: np.ndarray):
    equals = np.in1d(predictions, real_values)
    tmp = len(list(filter(lambda a: a == True, equals)))
    return tmp / len(predictions)


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("Accuracy: ", acc)
    print("F1 score: ", f1)
    print("Confusion matrix: ", conf_matrix)


def categorical_cross_entropy(actual, predicted):
    from math import log
    sum_score = 0.0
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            sum_score += actual[i][j] * log(1e-15 + predicted[i][j])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score


def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val


def mae(predictions, targets):
    differences = predictions - targets
    absolute_differences = np.absolute(differences)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences


def cross_entropy(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5)))/N
    return ce_loss


FILE_NAME = './nursery.data'
data = pd.read_csv(FILE_NAME)
cols = data.columns
data = convert_categorical_values_to_numerical(data, cols)
train, test = split_train_validation_set(data)

model = create_model(input_shape=data.shape[1] - 1)

X = train.values[:, 0:8]
Y = train.values[:, 8]
scaler = StandardScaler()
X = scaler.fit_transform(X)

model.fit(x=X, y=Y, epochs=1, verbose=2, validation_split=0.1)
model.save('model.h5')

X_test = test.values[:, 0:8]
X_test = scaler.transform(X_test)
Y_test = test.values[:, 8]
predictions = model.predict(X_test)
Y_pred = predictions.argmax(axis=1)

evaluate_model(Y_test, Y_pred)

# testiraj razlicite metrike, optimizatore, arhitekture ...
