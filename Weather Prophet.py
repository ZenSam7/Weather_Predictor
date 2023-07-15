from keras import Sequential
import keras
from keras.layers import Flatten, Dense, SimpleRNN, LSTM, BatchNormalization, Conv1D
import tensorflow as tf
from time import time
import numpy as np
from Weather_Data import get_moscow_data, get_plank_history, get_weather_history

# Убираем предупреждения
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
tf.get_logger().setLevel(logging.ERROR)


# # Работаем с GPU
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
# Работаем с CPU
tf.config.set_visible_devices([], 'GPU')


def print_ai_ans(sequence_len=50, print_real_data=True, print_ai_pred=True,
                 print_ai_pred_ai_pred=True, print_errors=True, print_mean_error=True,
                 learning_cycle=0):
    """ Отображаем предсказания ИИшек, и правильные ответ"""

    real_data, ai_predict, ai_pred_on_ai_pred, error = [], [], [], []
    rand = np.random.randint(0, len(test_data) - sequence_len - 1)

    # Проверяем на тестовых данных
    for data in test_data[rand: rand + sequence_len]:
        real_data.append(data.tolist()[0])

        ai_pred = [i[0, 0, 0] for i in ai.predict(np.resize(data, (1, 1, 7)), verbose=False)]
        ai_pred = [round(ai_pred[i] + real_data[-1][i + 3], 1) for i in range(4)]
        ai_predict.append(ai_pred)

        # На сколько % ошиблась нейронка, по сравнению с реальными данными
        err = []
        for ans_data, ai_data in zip(real_data[-1][3:], ai_pred):
            ans_data, ai_data = abs(ans_data), abs(ai_data)
            err.append(abs(ans_data - ai_data) / (ans_data + 20) * 100)
        error.append([round(e, 1) for e in err])

        # Скармливаем ИИшке своё же предсказание
        ai_on_ai = [round(i[0, 0, 0], 1) for i in ai.predict(
            np.resize(real_data[-1][:3] + ai_pred, (1, 1, 7)), verbose=False)]
        ai_on_ai = [round(ai_on_ai[i] + real_data[-1][i + 3], 1) for i in range(4)]
        ai_pred_on_ai_pred.append(ai_on_ai)

    # Выводим данные
    if print_real_data:
        print("Time \t\t\t\t Real Data", end="")
    if print_ai_pred:
        print(" \t\t\t\t\t\t Ai Predict", end="")
    if print_ai_pred_ai_pred:
        print("\t\t\t\t AI data from AI", end="")
    if print_mean_error:
        print("\t\t\t Error AI")

    for real, pred, pred_on_pred, err in zip(real_data, ai_predict, ai_pred_on_ai_pred, error):
        real = [round(i, 1) for i in real]
        if print_real_data:
            print(real[:3], "\t", real[3:], end="")
        if print_ai_pred:
            print("\t", pred, end="")
        if print_ai_pred_ai_pred:
            print("\t", pred_on_pred, end="")
        if print_errors:
            print("\t", err, "%")

    if print_mean_error:
        print(f"{learning_cycle}:", "Mean Error:",
              round(sum([sum(e) for e in error]) / (4 * len(error)), 1))


"""Загружаем данные"""
# В get_moscow_data 133_066 записей
# В get_plank_history 420_551 записей
# В get_weather_history 96_453 записей

# Причём get_plank_history и get_weather_history имеют одинаковй формат данных, т.е. их можно объединить

# ВНИМАНИЕ!: get_moscow_data отличается от get_plank_history и get_weather_history тем,
# что последнее значение это облачность (%)
DATA_out = get_plank_history() + get_weather_history()
print(">>> Dataset loaded\n")


"""Создаём ИИшки"""
# Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
# Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы

# У всех нейронок одна архитектура и один вход
input_layer = keras.Input((1, 7))


def get_ai(name):
    model = Sequential([
        Conv1D(8, 7, padding="same"),
        BatchNormalization(),
        Conv1D(16, 7, padding="same"),
        BatchNormalization(),
        Conv1D(32, 7, padding="same"),
        BatchNormalization(),

        Dense(50, activation="relu"),
        BatchNormalization(),
        LSTM(50, return_sequences=True, unroll=True),
        BatchNormalization(),
        Dense(50, activation="relu"),
        BatchNormalization(),
        LSTM(50, return_sequences=True, unroll=True),
        BatchNormalization(),
        Dense(50, activation="relu"),

        Dense(1, activation="linear"),
    ])(input_layer)

    output = Dense(1, activation="linear", name=name)(model)

    return output


temperature = get_ai("temp")          # temperature
pressure = get_ai("press")            # pressure
humidity = get_ai("humid")            # humidity
cloud_or_wind = get_ai("cloud_wind")  # cloud or wind


ai = keras.Model(input_layer, [temperature, pressure, humidity, cloud_or_wind])
ai.compile(optimizer=keras.optimizers.Adagrad(0.001), loss="mean_squared_error",
           loss_weights={"temp": 10.0, "press": 1.0, "humid": 1.0, "cloud_wind": 1.0})
           # Отдаём приоритет температуре

# ai.summary()


"""Сохранения / Загрузки"""
def save_path(name): return f"Saves Weather Prophet/{name}"

def SAVE_NAME(num): return f"5_save~{num}"

# Как загружать: ai = tf.keras.models.load_model(save_path(AI_NAME))
# Как сохранять: ai.save(save_path(AI_NAME))

# ЗАГРУЖАЕМСЯ
print(f">>> Loading the {SAVE_NAME(3)}", end="\t\t")
ai = tf.keras.models.load_model(save_path(SAVE_NAME(3)))
print("Done\n")


"""DATA_in == Данные погоды, начиная с 1ого дня (принимает)
   DATA_out == Данные погоды, начиная с 2ого дня (должен предсказать)"""

# Создаём смещени назад во времени
DATA_in = DATA_out[:-1]
DATA_out = DATA_out[1:]

DATA_out = np.array(DATA_out).reshape((len(DATA_out), 1, 7))
DATA_in = np.array(DATA_in).reshape((len(DATA_out), 1, 7))

DATA_out = DATA_out - DATA_in   # Остаточное обучение
DATA_out = DATA_out[:, :, 3:]   # (ИИшке не надо предсказывать время)


"""Обучение"""

callbacks = [
    #keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=5, verbose=False),
]

# Берём % от входных данных
test_size = int(len(DATA_in) * 0.01)

# Разделяем часть для обучения и для тестирования
# В качестве ответа записываем значение природного явления
train_data = DATA_in[:-test_size]
train_data_answer = np.reshape(np.array([DATA_out[:-test_size, 0, :]]), (len(train_data), 1, 4))

test_data = DATA_in[-test_size:]
test_data_answer = np.reshape(np.array([DATA_out[-test_size:, 0, :]]), (test_size, 1, 4))


for learning_cycle in range(6, 99):
    print(f">>> Learning the {SAVE_NAME(learning_cycle)}")

    ai.fit(train_data, train_data_answer, epochs=4, batch_size=50, verbose=False,
           shuffle=False, callbacks=callbacks)

    # print(">>> Testing:")
    # ai.evaluate(test_data, test_data_answer, batch_size=10, verbose=True)

    print("\n")


    # Сохраняем
    print(f">>> Saving the {SAVE_NAME(learning_cycle)}", end="  ")
    ai.save(save_path(SAVE_NAME(learning_cycle)))
    print("Done (Ignore the WARNING)\n")


    print_ai_ans(50, True, True, False, True, True)