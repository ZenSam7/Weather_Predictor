from keras import Sequential
import keras
import os
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


def load_data(name_db="moscow", len_test_data=0):
    """Загружаем данные"""
    global train_data, train_data_answer, test_data, test_data_answer

    # В WD.get_moscow_data     477_603 записей      (все данные идут с шагом в 1 часа)
    # В WD.get_fresh_data        1_455 записей      (данные за последние 60 дней, идут с шагом в 1 час)

    if name_db == "moscow":
        DATA_out = WD.get_moscow_data()
    elif name_db == "fresh":
        DATA_out = WD.get_fresh_data()
    print(">>> Dataset loaded\n")


    """DATA_in == Данные погоды, начиная с 1ого дня (принимает)
       DATA_out == Данные погоды, начиная с 2ого дня (должен предсказать)"""

    # Создаём смещени назад во времени
    DATA_in = DATA_out[:-1]
    DATA_out = DATA_out[1:]

    # Преобразуем данные
    DATA_out = np.array(DATA_out).reshape((len(DATA_out), 1, 8))
    DATA_in = np.array(DATA_in).reshape((len(DATA_out), 1, 8))

    DATA_out = WD.normalize(DATA_out - DATA_in) # Остаточное обучение + нормализуем от -1 до 1
    DATA_out = DATA_out[:, :, 3:]               # ИИшке не надо предсказывать время

    # Разделяем часть для обучения и для тестирования
    train_data = DATA_in[:-len_test_data] if len_test_data > 0 else DATA_in
    train_data_answer = DATA_out[:-len_test_data] if len_test_data > 0 else DATA_out

    if len_test_data > 0:
        test_data = DATA_in[-len_test_data:]
        test_data_answer = DATA_out[-len_test_data:, 0, :]
    else:
        test_data, test_data_answer = [], []


def create_ai(num_layers_conv=3, num_ai_layers=5, num_neurons=32):
    """Создаём ИИшки"""
    global ai
    # Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
    # Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы

    # У всех нейронок одна архитектура и один вход
    input_layer = keras.Input((1, 8))


    class Architecture:
        def get_ai(self):
            num_conv_neurons = 8
            list_layers = []
            # Добавляем Conv1D
            for _ in range(num_layers_conv):
                list_layers.append(Conv1D(num_conv_neurons, 8, padding="same"))
                num_conv_neurons *= 2

            # Добавляем остальные слои
            for i in range(num_ai_layers):
                if i % 2 == 0:
                    list_layers.append(Dense(num_neurons, activation="relu"))
                else:
                    list_layers.append(LSTM(num_neurons, return_sequences=True, unroll=True))


            return Sequential(list_layers)(input_layer)

    # Создаём 5 полностью независимые нейронки
    temperature = Dense(1, activation="tanh", name="temp")(Architecture().get_ai())
    pressure = Dense(1, activation="tanh", name="press")(Architecture().get_ai())
    humidity = Dense(1, activation="tanh", name="humid")(Architecture().get_ai())
    cloud = Dense(1, activation="tanh", name="cloud")(Architecture().get_ai())
    rain = Dense(1, activation="tanh", name="rain")(Architecture().get_ai())


    ai = keras.Model(input_layer, [temperature, pressure, humidity, cloud, rain], name="Weather_Predictor")
    ai.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error",
               loss_weights={"temp": 100_000, "press": 10_000, "humid": 10_000, "cloud": 10_000, "rain": 100_000},)
               # Отдаём приоритет температуре и осадкам, и увеличиваем ошибки (иначе они будут ≈0)


def ai_name(name):
    """Сохранения / Загрузки"""
    global save_path, SAVE_NAME

    save_path = lambda ai_name: f"Saves Weather Prophet/{ai_name}"
    SAVE_NAME = lambda num: f"{name}~{num}"


def load_ai(loading_with_learning_cycle=-1, print_summary=False):
    """ЗАГРУЖАЕМСЯ"""
    global ai

    # Вычисляем номер последнего сохранения с текущем именем
    if loading_with_learning_cycle == -1:
        loading_with_learning_cycle = int(sorted([save_name if SAVE_NAME(0)[:-2] in save_name
                    else None for save_name in os.listdir("Saves Weather Prophet")])[-1].split("~")[-1])

    print(f">>> Loading the {SAVE_NAME(loading_with_learning_cycle)}", end="\t\t")
    ai = tf.keras.models.load_model(save_path(SAVE_NAME(loading_with_learning_cycle)))
    print("Done\n")

    if print_summary:
        ai.summary(); print()



def train_ai(start_on=-1, finish_on=99, # Начинаем с номера последнего сохранения до finish_on
             save_every_learning_cycle=True,    # Сохранять ли каждую ИИшку
             epochs=3, batch_size=100,  verbose=2, # Параметры fit()
             print_ai_answers=True, len_prints_ai_answers=100, # Выводить и сравнивать данные, или нет
             print_weather_predict=True, len_predict_days=3, # Выводить ли  прогноз погоды
             use_callbacks=False, callbacks_min_delta=10, callbacks_patience=3, # Параметры callbacks
             shift_dataset=True, start_with_dataset_offset=0, # Смещаем данные на 1 час каждый цикл
                     # (т.е. после первого смещения ИИшка должна предсказывать на 2 часа вперёд, потом на 3...)
             ):
    """Обучение"""
    global train_data, train_data_answer, test_data, test_data_answer
    num_dataset_offset = 1

    # Сдвигаемм наборы данных
    if start_with_dataset_offset > 0:
        num_dataset_offset += start_with_dataset_offset
        train_data = train_data[: -start_with_dataset_offset]
        train_data_answer = train_data_answer[start_with_dataset_offset:]
        if len(train_data) > 0:
            test_data = test_data[: -start_with_dataset_offset]
            test_data_answer = test_data_answer[start_with_dataset_offset:]


    callbacks = [keras.callbacks.EarlyStopping(monitor="loss",
                min_delta=callbacks_min_delta, patience=callbacks_patience, verbose=False)] \
        if use_callbacks else None

    # Продолжаем с последнего сохранения если start_on == -1 (или создаём новое)
    if start_on == -1:
        try:
            start_on = int(sorted([save_name if SAVE_NAME(0)[:-2] in save_name else None
                            for save_name in os.listdir("Saves Weather Prophet")])[-1].split("~")[-1])
        except:
            start_on = 0

    # Циклы обучения
    for learning_cycle in range(start_on, finish_on):
        print(f">>> Learning the {SAVE_NAME(learning_cycle)}\t\t\tСмещение данных: {num_dataset_offset} ч")
        ai.fit(train_data, train_data_answer,
               epochs=epochs, batch_size=batch_size,
               verbose=verbose, shuffle=False, callbacks=callbacks)
        print()


        # Сохраняем
        if save_every_learning_cycle:
            print(f">>> Saving the {SAVE_NAME(learning_cycle)}", end="\t\t")
            ai.save(save_path(SAVE_NAME(learning_cycle)))
            print("Done (Ignore the WARNING)\n")


        # Выводим данные и сравниваем
        if print_ai_answers:
            # Используем train_data если test_data нет
            WD.print_ai_answers(ai, test_data if len(test_data)>0 else train_data, len_prints_ai_answers)
        if print_weather_predict:
            WD.print_weather_predict(ai, len_predict_days)


        # Создаём смещение данных на 1 час
        if shift_dataset:
            num_dataset_offset += 1
            train_data = train_data[: -1]
            train_data_answer = train_data_answer[1:]
            if len(train_data) > 0:
                test_data = test_data[: -1]
                test_data_answer = test_data_answer[1:]


"""Скрипт"""
if __name__ == "__main__":
    what_device_use("gpu")
    ai_name("AI_v4")

    # create_ai(5, 7, 64)
    load_ai(-1, print_summary=True)

    load_data("moscow", len_test_data=0)  # "fresh"
    train_ai(epochs=1, batch_size=100, verbose=1)
