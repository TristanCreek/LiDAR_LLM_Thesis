import os
import time
import random
import pathlib
import shutil
import tensorflow as tf
from Layers import GetPatches, GetBatchCosts, ApplyWeights

import numpy as np
import cv2

def get_model(img_size, batch_size, ring_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 1), batch_size=batch_size)
    patches_4 = GetPatches(num_patches=4)(inputs)
    patches_16 = GetPatches(num_patches=16)(inputs)
    costs1 = GetBatchCosts(ring_size=ring_size)(inputs)
    costs4 = GetBatchCosts(ring_size=ring_size, patches=True)(patches_4)
    costs16 = GetBatchCosts(ring_size=ring_size, patches=True)(patches_16)
    costs = tf.keras.layers.concatenate([costs1, costs4, costs16], axis=1)
    weighted_costs = ApplyWeights()(costs)
    dense1 = tf.keras.layers.Dense(512, activation='relu', trainable=True)(weighted_costs)
    sigmoid = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)
    model = tf.keras.Model(inputs=inputs, outputs=sigmoid)
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model

def get_dataset(pics_path, img_size, batch_size):
    data_dir = pathlib.Path(pics_path)
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                           validation_split=0.2,
                                                           shuffle=True,
                                                           subset='training',
                                                           color_mode='grayscale',
                                                           seed=random.randint(0, 10000),
                                                           image_size=(img_size, img_size),
                                                           batch_size=batch_size,
                                                           label_mode='binary')
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                         validation_split=0.2,
                                                         subset='validation',
                                                         shuffle=True,
                                                         color_mode='grayscale',
                                                         seed=random.randint(0, 10000),
                                                         image_size=(img_size, img_size),
                                                         batch_size=batch_size,
                                                         label_mode='binary')
    return train_ds, val_ds

def train_and_save_model(path, img_size=512):
    batch_size = 1
    ring_size = 1
    model = get_model(img_size, batch_size, ring_size)
    train_ds, val_ds = get_dataset(path + '\\train', img_size, batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=5, callbacks=[callback])

    # Delete old model if it exists
    dirpath = pathlib.Path(path + '\\model')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    # Save new model
    model.save(path + '\\model')

def load_and_evaluate_model(path, img_size=512):
    batch_size = 1
    tests = path + '\\test\\masked_pics'
    test_paths = []
    test_names = []
    for test in os.listdir(tests):
        test_paths.append(os.path.join(tests, test))
        test_names.append(test)

    model = tf.keras.models.load_model(path + '\\model')
    results = []
    for test_path, test_name in zip(test_paths, test_names):
        dataset = tf.keras.utils.image_dataset_from_directory(test_path,
                                                               color_mode='grayscale',
                                                               shuffle=True,
                                                               image_size=(img_size, img_size),
                                                               batch_size=batch_size,
                                                               label_mode='binary')


        accuracy = model.evaluate(dataset, verbose=0)
        results.append(test_name + ' - Accuracy: ' + str(accuracy[1]))

    return results


if __name__ == '__main__':
    start_time = time.time()
    train_and_save_model('path')
    results = load_and_evaluate_model('path')
    for result in results:
        print(result)
    print("{:.9f} seconds".format(time.time() - start_time))
