import tensorflow as tf
import numpy as np
import sklearn
import csv

encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)

with open("sonar.csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = list(reader)

X = np.array(list(record[:60] for record in data))
X = X.astype(np.float32)
Y_raw = np.array(list([record[60]] for record in data))
Y = encoder.fit_transform(Y_raw)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ]
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train,
                    Y_train,
                    epochs=120,
                    batch_size=32,
                    validation_split=0.2)

predicted = model.predict(np.array([X_train[0]]))
expected = Y_train[0]
print(f"expected={expected}, predicted={predicted}")
