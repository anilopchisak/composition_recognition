import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.models import model_from_json
from sklearn.model_selection import train_test_split
import time

# Функция для предобработки изображений
def img_proc(img_path):
    # Загрузка изображения в оттенках серого
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # уменьшение шума
    img = cv2.bilateralFilter(img, 30, 20, 20)
    # Изменение размера изображения до 28х28 пикселей
    img = cv2.resize(img, (300, 300))
    # Добавление размерности к изображению для работы с нейронной сетью
    img = img.reshape((300, 300, 1))
    # Нормализация значений пикселей в диапазон от 0 до 1
    img = img.astype('float32') / 255.0

    return img

# Функция для создания и обучения модели
def model_create_and_train(train_data, train_labels):
    # Создаем модель сверточной нейронной сети
    model = models.Sequential()

    # Добавляем слои свертки с функцией активации ReLU и слой пулинга
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Добавляем слои полносвязной нейронной сети
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Компиляция модели с выбором оптимизатора adam, функции потерь SparseCategoricalCrossentropy и метрик
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time()  # текущее время (старт обучения)

    # Обучение модели на тренировочных данных
    # Обучение: epochs - количество эпох (циклов),
    #           batch_size - размер набора обучающих данных для корреции весов
    history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

    # Вывод затраченного на обучение времени
    print("--- %s seconds ---" % (time.time() - start_time))

    # Вывод структуры модели
    print("Input size: ", train_data.shape[1])
    model.summary()

    return model


# Сохраняем структуру модели в файл model.json, а весовые коэффициенты в weights.h5
def model_save(model):
    json_file = 'model.json'
    model_json = model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)

    model.save_weights('weights.h5')


# Загружаем модель из файла json_file, а веса из weights.h5
def model_load(json_file, weights_file):
    with open(json_file, 'r') as f:
        loaded_model = model_from_json(f.read())

    loaded_model.load_weights(weights_file)

    return loaded_model


if __name__ == '__main__':
    dataset = pd.read_csv(encoding="utf-8")
    in_data = dataset.iloc[:,1:10]
    out_data = dataset.iloc[:, 10:11].values

    # подготовка изображений к обучению
    img_col = ['']
    for col_name in img_col:
        in_data[col_name] = img_proc(in_data[col_name])


    # Разделяем из общего набора данных тренировочную (90%) и тестовую выборку (10%)
    train_data, test_data, train_labels, test_labels = train_test_split(in_data, out_data, test_size=0.1)

    # Создаем новую модель НС и обучаем на тренировочной выборке
    model = model_create_and_train(train_data, train_labels)
    # Если необходимо загрузить сохраненную модель из файла, раскомментируйте строку ниже и закомментируйте строку выше
    # model = load_model('model.json', 'weights.h5')

    # Прогоняем через обученную модель тестовый набор данных для оценки точности
    out_pred = model.predict(test_data)

    # Оцениваем точность модели на тестовых данных
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print('Test accuracy:', test_acc)

    print("Save model to file ?:")
    q = input()
    if q.lower() == 'y':
        model_save(model)