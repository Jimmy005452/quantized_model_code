import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
import numpy as np
import os

#載入資料集
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#轉換圖片為白底黑字
train_images = 255 - train_images
test_images = 255 - test_images

#將圖片複製為三通道圖形
train_images = tf.expand_dims(train_images, 3)
test_images = tf.expand_dims(test_images, 3)

#調整圖片格式
train_images = tf.tile(train_images, multiples = [1, 1, 1, 3])
test_images = tf.tile(test_images, multiples = [1, 1, 1, 3])

#驗證集
val_x = train_images[:5000]
val_y = train_labels[:5000]

#圖形增強
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.2
)

datagen.fit(train_images)
train_generator = datagen.flow(train_images, train_labels, batch_size=32)
valid_generator = datagen.flow(test_images, test_labels, batch_size=32)

#模型架構
inputLayer = layers.Input(shape=(28,28, 3))
C1 = layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_images[0].shape, padding='same')(inputLayer)
Flatten_C1 = layers.Flatten()(C1)
earlyExit_C1 = layers.Dense(name='output1', units=10, activation='softmax')(Flatten_C1)

S2 = layers.AveragePooling2D()(C1)
Flatten_S2 = layers.Flatten()(S2)
earlyExit_S2 = layers.Dense(name='output2', units=10, activation='softmax')(Flatten_S2)

C3 = layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid')(S2)
Flatten_C3 = layers.Flatten()(C3)
earlyExit_C3 = layers.Dense(name='output3', units=10, activation='softmax')(Flatten_C3)

S4 = layers.AveragePooling2D()(C3)

Flatten = layers.Flatten()(S4)
earlyExit_S4 = layers.Dense(name='output4', units=10, activation='softmax')(Flatten)

F5 = layers.Dense(120, activation='tanh')(Flatten)
earlyExit_F5 = layers.Dense(name='output5', units=10, activation='softmax')(F5)

F6 = layers.Dense(84, activation='tanh')(F5)

outputLayer = layers.Dense(name='output6', units=10, activation='softmax')(F6)

mnist_model = keras.models.Model(inputs=inputLayer, outputs=[earlyExit_C1, earlyExit_S2, earlyExit_C3, earlyExit_S4, earlyExit_F5, outputLayer])

#儲存訓練資料
root_logdir = os.path.join(os.curdir, "logs\\fit\\")

def get_run_logdir(run_id):
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir("NoQ_1out_model")
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

#訓練模型與評估模型
mnist_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
mnist_model.fit(train_generator, steps_per_epoch=len(train_images)/32, epochs=15, validation_data=valid_generator, validation_steps=len(test_images)/32, callbacks=[tensorboard_cb])
mnist_model.evaluate(test_images, [test_labels, test_labels, test_labels, test_labels, test_labels, test_labels])

#儲存模型
mnist_model.save('./quantisationModel/mnist_6out_model.h5')

#創建並儲存每個出口之模型
modelC1 = keras.models.Model(inputs=inputLayer, outputs=earlyExit_C1)
modelC1.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
modelC1.save('./quantisationModel/modelC1.h5')
modelS2 = keras.models.Model(inputs=inputLayer, outputs=earlyExit_S2)
modelS2.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
modelS2.save('./quantisationModel/modelS2.h5')
modelC3 = keras.models.Model(inputs=inputLayer, outputs=earlyExit_C3)
modelC3.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
modelC3.save('./quantisationModel/modelC3.h5')
modelS4 = keras.models.Model(inputs=inputLayer, outputs=earlyExit_S4)
modelS4.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
modelS4.save('./quantisationModel/modelS4.h5')
modelF5 = keras.models.Model(inputs=inputLayer, outputs=earlyExit_F5)
modelF5.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
modelF5.save('./quantisationModel/modelF5.h5')
modelLast = keras.models.Model(inputs=inputLayer, outputs=outputLayer)
modelLast.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
modelLast.save('./quantisationModel/modelLast.h5')

#創建量化模型
quantize_model = tfmot.quantization.keras.quantize_model

#創建各個輸出之量化模型
q_aware_modelC1 = quantize_model(modelC1)
q_aware_modelC1.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
q_aware_modelS2 = quantize_model(modelS2)
q_aware_modelS2.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
q_aware_modelC3 = quantize_model(modelC3)
q_aware_modelC3.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
q_aware_modelS4 = quantize_model(modelS4)
q_aware_modelS4.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
q_aware_modelF5 = quantize_model(modelF5)
q_aware_modelF5.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
q_aware_modelLast = quantize_model(modelLast)
q_aware_modelLast.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

#創建訓練集
train_images_subset = train_images[0:1000] # out of 60000
train_labels_subset = train_labels[0:1000]

#訓練量化模型
q_aware_modelC1.fit(train_images_subset, train_labels_subset, batch_size=500, epochs=1, validation_split=0.1)
q_aware_modelC1.evaluate(test_images, test_labels)
q_aware_modelS2.fit(train_images_subset, train_labels_subset, batch_size=500, epochs=1, validation_split=0.1)
q_aware_modelS2.evaluate(test_images, test_labels)
q_aware_modelC3.fit(train_images_subset, train_labels_subset, batch_size=500, epochs=1, validation_split=0.1)
q_aware_modelC3.evaluate(test_images, test_labels)
q_aware_modelS4.fit(train_images_subset, train_labels_subset, batch_size=500, epochs=1, validation_split=0.1)
q_aware_modelS4.evaluate(test_images, test_labels)
q_aware_modelF5.fit(train_images_subset, train_labels_subset, batch_size=500, epochs=1, validation_split=0.1)
q_aware_modelF5.evaluate(test_images, test_labels)
q_aware_modelLast.fit(train_images_subset, train_labels_subset, batch_size=500, epochs=1, validation_split=0.1)
q_aware_modelLast.evaluate(test_images, test_labels)

#儲存無量化模型為TFLite model
def save_model(model, save_name="mymodel"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("./" + save_name + ".tflite", 'wb') as f:
        f.write(tflite_model)

#儲存量化模型為TFLite model
def save_model_with_Q(model, save_name="mymodel"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("./" + save_name + ".tflite", 'wb') as f:
        f.write(tflite_model)

#轉換模型為TFLite模型
save_model(modelC1, "modelC1")
save_model(modelS2, "modelS2")
save_model(modelC3, "modelC3")
save_model(modelS4, "modelS4")
save_model(modelF5, "modelF5")
save_model(modelLast, "modelLast")
save_model_with_Q(q_aware_modelC1, "q_aware_modelC1")
save_model_with_Q(q_aware_modelS2, "q_aware_modelS2")
save_model_with_Q(q_aware_modelC3, "q_aware_modelC3")
save_model_with_Q(q_aware_modelS4, "q_aware_modelS4")
save_model_with_Q(q_aware_modelF5, "q_aware_modelF5")
save_model_with_Q(q_aware_modelLast, "q_aware_modelLast")

#顯示模型架構
modelC1.summary()
modelS2.summary()
modelC3.summary()
modelS4.summary()
modelF5.summary()
modelLast.summary()

q_aware_modelC1.summary()
q_aware_modelS2.summary()
q_aware_modelC3.summary()
q_aware_modelS4.summary()
q_aware_modelF5.summary()
q_aware_modelLast.summary()