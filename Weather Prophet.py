from keras import Sequential
import keras
from keras.layers import Flatten, Dense, SimpleRNN, LSTM, BatchNormalization, Conv1D
import tensorflow as tf
from time import time
import numpy as np
from Weather_Data import get_moscow_data, get_plank_history, get_weather_history, print_ai_answers

# Убираем предупреждения
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
tf.get_logger().setLevel(logging.ERROR)



"""Загружаем данные"""
# В get_moscow_data 133_066 записей
# В get_plank_history 420_551 записей
# В get_weather_history 96_453 записей

# Причём get_plank_history и get_weather_history имеют одинаковй формат данных, т.е. их можно объединить

# ВНИМАНИЕ!: get_moscow_data отличается от get_plank_history и get_weather_history тем,
# что последнее значение это облачность (%)
DATA_out = get_weather_history()
print(">>> Dataset loaded\n")



"""Создаём ИИшки"""
BATCH_SIZE = 50  # Размер батча, который мы подаём и выводим из ИИшки
# (Отличается от размера батча в ai.fit())

# Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
# Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы

# У всех нейронок одна архитектура и один вход
input_layer = keras.Input((BATCH_SIZE, 7))


def get_ai(name):
    model = Sequential([
        Conv1D(8, 7, padding="same"),
        BatchNormalization(),
        Conv1D(16, 7, padding="same"),
        BatchNormalization(),
        Conv1D(32, 7, padding="same"),
        BatchNormalization(),

        Dense(64, activation="tanh"),
        BatchNormalization(),
        Dense(64, activation="tanh"),
        BatchNormalization(),
        Dense(64, activation="tanh"),
    ])(input_layer)

    output = Dense(1, activation="tanh", name=name)(model)

    return output


temperature = get_ai("temp")          # temperature
pressure = get_ai("press")            # pressure
humidity = get_ai("humid")            # humidity
cloud_or_wind = get_ai("cloud_wind")  # cloud or wind


ai = keras.Model(input_layer, [temperature, pressure, humidity, cloud_or_wind])
ai.compile(optimizer=keras.optimizers.Adagrad(0.001), loss="mean_absolute_error",
           loss_weights={"temp": 100.0, "press": 10.0, "humid": 10.0, "cloud_wind": 10.0})
           # Отдаём приоритет температуре

ai.summary(); print()



"""Сохранения / Загрузки"""
def save_path(name): return f"Saves Weather Prophet/{name}"

def SAVE_NAME(num): return f"AI_v2.0~{num}"

# Как загружать: ai = tf.keras.models.load_model(save_path(AI_NAME))
# Как сохранять: ai.save(save_path(AI_NAME))



"""DATA_in == Данные погоды, начиная с 1ого дня (принимает)
   DATA_out == Данные погоды, начиная с 2ого дня (должен предсказать)"""

# Создаём смещени назад во времени и изменяем размер так,
# чтобы можно было подать сразу BATCH_SIZE веркторов
# (это позволяет не использовать RNN, ведь подаём мы последние BATCH_SIZE записей)
DATA_in = DATA_out[:-1][: len(DATA_out) // BATCH_SIZE * BATCH_SIZE]
DATA_out = DATA_out[1:][: len(DATA_out) // BATCH_SIZE * BATCH_SIZE]

DATA_in = np.array([DATA_in[i: i +BATCH_SIZE] for i in range(len(DATA_out) - BATCH_SIZE)])
DATA_out = np.array([DATA_out[i: i +BATCH_SIZE] for i in range(len(DATA_out) - BATCH_SIZE)])[:, :, 3:]



"""Обучение"""
# % от всех данных
test_size = int(len(DATA_in) * 0.05)

# Разделяем часть для обучения и для тестирования
# В качестве ответа записываем значение природного явления
train_data = DATA_in[:-test_size]
train_data_answer = DATA_out[:-test_size]

test_data = DATA_in[-test_size:]
test_data_answer = [DATA_out[-test_size:]]


for learning_cycle in range(1, 99):
    ЗАГРУЖАЕМСЯ
    print(f">>> Loading the {SAVE_NAME(learning_cycle)}", end="\t\t")
    ai = tf.keras.models.load_model(save_path(SAVE_NAME(learning_cycle)))
    print("Done")
    learning_cycle += 1


    print(f">>> Learning the {SAVE_NAME(learning_cycle)}")

    ai.fit(train_data, train_data_answer, epochs=5, batch_size=150, verbose=True, shuffle=False)

    print("\n")


    # Сохраняем
    print(f">>> Saving the {SAVE_NAME(learning_cycle)}", end="  ")
    ai.save(save_path(SAVE_NAME(learning_cycle)))
    print("Done (Ignore the WARNING)\n")

    # Выфводим данные и сравниваем их "на глаз"
    print_ai_answers(ai, test_data, learning_cycle, BATCH_SIZE)
