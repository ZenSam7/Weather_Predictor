from keras import Sequential
import keras
from keras.layers import Flatten, Dense, SimpleRNN, LSTM, BatchNormalization, Conv1D
import tensorflow as tf
import numpy as np
import Weather_Data as WD
from time import time

# Убираем предупреждения
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
tf.get_logger().setLevel(logging.ERROR)


def what_device_use(device="cpu"):
    if device.lower() == "gpu":
        # Работаем с GPU
        tf.config.set_visible_devices(tf.config.list_physical_devices("GPU"), "GPU")

    if device.lower() == "cpu":
        # Работаем с CPU
        tf.config.set_visible_devices([], "GPU")


def load_data(name_db="moscow", len_test_data=1_000):
    """Загружаем данные"""
    global train_data, train_data_answer, test_data, test_data_answer
    """
    В WD.get_moscow_data     399_195 записей      (все данные идут с шагом в 1 часа)
    В WD.get_plank_history   420_551 записей      (все данные идут с шагом в 10 минут)
    В WD.get_weather_history  96_453 записей      (все данные идут с шагом в 1 час)
    В WD.get_fresh_data        1_440 записей      (данные за последние 60 дней, идут с шагом в 1 час)

    Причём get_plank_history и get_weather_history имеют одинаковй формат данных, т.е. их можно объединить\n

    ВНИМАНИЕ!: moscow_data и fresh_data отличается от plank_history и weather_history тем,
    что последнее значение это облачность (%)
    """
    if name_db == "moscow":
        DATA_out = WD.get_moscow_data()
    elif name_db == "plank":
        DATA_out = WD.get_plank_history()
    elif name_db == "weather":
        DATA_out = WD.get_weather_history()
    elif name_db == "fresh":
        DATA_out = WD.get_fresh_data()
    print(">>> Dataset loaded\n")


    """DATA_in == Данные погоды, начиная с 1ого дня (принимает)
       DATA_out == Данные погоды, начиная с 2ого дня (должен предсказать)"""

    # Создаём смещени назад во времени
    DATA_in = DATA_out[:-1]
    DATA_out = DATA_out[1:]

    DATA_out = np.array(DATA_out).reshape((len(DATA_out), 1, 7))
    DATA_in = np.array(DATA_in).reshape((len(DATA_out), 1, 7))

    DATA_out = WD.normalize(DATA_out - DATA_in) # Остаточное обучение (+ нормализуем от -1 до 1)
    DATA_out = DATA_out[:, :, 3:]               # ИИшке не надо предсказывать время

    # Разделяем часть для обучения и для тестирования
    # В качестве ответа записываем значение природного явления
    train_data = DATA_in[:-len_test_data]
    train_data_answer = np.reshape(np.array([DATA_out[:-len_test_data, 0, :]]), (len(train_data), 1, 4))

    test_data = DATA_in[-len_test_data:]
    test_data_answer = np.reshape(np.array([DATA_out[-len_test_data:, 0, :]]), (len_test_data, 1, 4))


def create_ai(num_layers_conv=3, num_ai_layers=5, num_neurons=32):
    """Создаём ИИшки"""
    global ai
    # Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
    # Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы

    # У всех нейронок одна архитектура и один вход
    input_layer = keras.Input((1, 7))


    class Architecture:
        def get_ai(self):
            num_conv_neurons = 8
            list_layers = []
            # Добавляем Conv1D
            for _ in range(num_layers_conv):
                list_layers.append(Conv1D(num_conv_neurons, 7, padding="same"))
                num_conv_neurons *= 2

            # Добавляем остальные слои
            for i in range(num_ai_layers):
                if i % 2 == 0:
                    list_layers.append(Dense(num_neurons, activation="relu"))
                else:
                    list_layers.append(LSTM(num_neurons, return_sequences=True, unroll=True))


            return Sequential(list_layers)(input_layer)

    # Создаём 4 полностью независимые нейронки
    temperature = Dense(1, activation="tanh", name="temp")(Architecture().get_ai())
    pressure = Dense(1, activation="tanh", name="press")(Architecture().get_ai())
    humidity = Dense(1, activation="tanh", name="humid")(Architecture().get_ai())
    cloud_or_wind = Dense(1, activation="tanh", name="cloud_wind")(Architecture().get_ai())


    ai = keras.Model(input_layer, [temperature, pressure, humidity, cloud_or_wind], name="Weather_Predictor")
    ai.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error",
               loss_weights={"temp": 100_000, "press": 10_000, "humid": 10_000, "cloud_wind": 10_000})
               # Отдаём приоритет температуре, и увеличиваем ошибки (иначе они будут <<1)


def ai_name(name):
    """Сохранения / Загрузки"""
    global save_path, SAVE_NAME

    save_path = lambda ai_name: f"Saves Weather Prophet/{ai_name}"
    SAVE_NAME = lambda num: f"{name}~{num}"


def load_ai(loading_with_learning_cycle=0, print_summary=False):
    """ЗАГРУЖАЕМСЯ"""
    global ai

    print(f">>> Loading the {SAVE_NAME(loading_with_learning_cycle)}", end="\t\t")
    ai = tf.keras.models.load_model(save_path(SAVE_NAME(loading_with_learning_cycle)))
    print("Done\n")

    if print_summary:
        ai.summary(); print()



def train_ai(start_on, finish_on,
             save_every_learning_cycle=True,
             epochs=3, batch_size=100,  verbose=2,
             print_ai_answers=True, len_prints_ai_answers=100,
             print_weather_predict=True, len_predict_days=3,
             use_callbacks=False, callbacks_min_delta=10, callbacks_patience=3):
    """Обучение"""
    if use_callbacks:
        callbacks = [keras.callbacks.EarlyStopping(monitor="loss",
                    min_delta=callbacks_min_delta, patience=callbacks_patience, verbose=False)]


    for learning_cycle in range(start_on, finish_on):
        print(f">>> Learning the {SAVE_NAME(learning_cycle)}")
        ai.fit(train_data, train_data_answer,
               epochs=epochs, batch_size=batch_size,
               verbose=verbose, shuffle=False,
               callbacks=callbacks if use_callbacks else None)
        print()


        # Сохраняем
        if save_every_learning_cycle:
            print(f">>> Saving the {SAVE_NAME(learning_cycle)}", end="\t\t")
            ai.save(save_path(SAVE_NAME(learning_cycle)))
            print("Done (Ignore the WARNING)\n")


        # Выводим данные и сравниваем
        if print_ai_answers:
            WD.print_ai_answers(ai, test_data, len_prints_ai_answers)

        if print_weather_predict:
            WD.print_weather_predict(ai, len_predict_days)


"""Скрипт"""
if __name__ == "__main__":
    what_device_use("cpu")
    ai_name("NN")

    # create_ai(5, 5, 32)
    load_ai(0, print_summary=True)

    load_data("moscow")
    train_ai(0, 5, epochs=3, batch_size=200, verbose=1)
