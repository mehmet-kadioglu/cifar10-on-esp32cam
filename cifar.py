import sys
from matplotlib import pyplot
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, Input, InputLayer
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import tensorflow
import tensorflow_model_optimization

from tensorflow_model_optimization.python.core.keras.compat import keras

print("\n\n\n")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Veriseti sekil ozeti")
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s\n' % (X_test.shape, y_test.shape))

print("Veriseti preprocessing yapmadan once:")
print("X_test: ", X_test[:1])
print("y_test: ", y_test[:1])
print("\n")

# one hot encode uygula labellar uzerinde
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Resim pixellerini 0-255 arasindan 0-1 arasina float olarak cek
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Veriseti preprocessing yaptiktan sonra:")
print("X_test: ", X_test[:1])
print("y_test: ", y_test[:1])
print("\n")



# UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. 
# When using Sequential models, prefer using an `Input(shape)` object as the 
# first layer in the model instead.
# Uyarisi alindigindan dolayi asagidaki sekilde yapildi
input_shape = (32, 32, 3)
input_layer = Input(shape=input_shape)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(32, 32, 3)),

    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.3),

    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1)

model.save('models/cifar_model_midterm.keras')




print("Model Dosyadan Yukleniyor...\n")
loaded_model = load_model('models/cifar_model_midterm.keras')
loaded_model.summary()

print("Model Degerlendiriliyor...")
_, acc = loaded_model.evaluate(X_test, y_test, verbose=1)
print('Dogruluk(Accuracy) Yuzdelik Oran: %.3f' % (acc * 100.0))



print("X_test uzerinden tahmin yapiliyor...")
y_pred = loaded_model.predict(X_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
print(y_test)
print(y_pred)

precision = precision_score(y_test, y_pred, average='micro')
print("Precision:", precision)

recall = recall_score(y_test, y_pred, average='micro')
print("Recall:", recall)

f1 = f1_score(y_test, y_pred, average='micro')
print("F1 Score:", f1)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)




# Pruning
prune_low_magnitude = tensorflow_model_optimization.sparsity.keras.prune_low_magnitude
pruning_params = {
    'pruning_schedule': tensorflow_model_optimization.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                             final_sparsity=0.90,
                                                             begin_step=2000,
                                                             end_step=10000)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_for_pruning.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, callbacks=[tensorflow_model_optimization.sparsity.keras.UpdatePruningStep()])

# Quantization
converter = tensorflow.lite.TFLiteConverter.from_keras_model(model_for_pruning)
converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Fine-tuning
model_for_pruning.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model_for_pruning.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)

# Evaluation
loss, accuracy = model_for_pruning.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')