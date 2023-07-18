from keras import Sequential
import keras
from keras.layers import Flatten, Dense, SimpleRNN, LSTM, BatchNormalization, Conv1D, Dropout
import tensorflow as tf
from time import time
import numpy as np
from Weather_Data import get_moscow_data, get_plank_history, get_weather_history, print_ai_answers

# Убираем предупреждения
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
tf.get_logger().setLevel(logging.ERROR)


# # Работаем с GPU
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
# Работаем с CPU
tf.config.set_visible_devices([], 'GPU')


"""Загружаем данные"""
# В get_moscow_data 133_066 записей
# В get_plank_history 420_551 записей
# В get_weather_history 96_453 записей

# Причём get_plank_history и get_weather_history имеют одинаковй формат данных, т.е. их можно объединить

# ВНИМАНИЕ!: get_moscow_data отличается от get_plank_history и get_weather_history тем,
# что последнее значение это облачность (%)
DATA_out = get_plank_history()
print(">>> Dataset loaded\n")



"""Создаём ИИшки"""
# Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
# Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы

# У всех нейронок одна архитектура и один вход
input_layer = keras.Input((1, 7))


class Arcitecture():
    def get_ai(self, name: str):
        model = Sequential([
            Conv1D(8, 3, padding="same"),
            BatchNormalization(),
            Conv1D(16, 3, padding="same"),
            BatchNormalization(),

            Dense(32, activation="tanh"),
            BatchNormalization(),
            LSTM(32, return_sequences=True, unroll=True),
            BatchNormalization(),
            Dense(32, activation="tanh"),
            BatchNormalization(),
            LSTM(32, return_sequences=True, unroll=True),
            BatchNormalization(),
            Dense(32, activation="tanh"),
        ])(input_layer)

        output = Dense(1, activation="tanh", name=name)(model)
        return output


# Создаём 4 полностью незаввисимые нейросети
temperature = Arcitecture()
temperature = temperature.get_ai("temp")

pressure = Arcitecture()
pressure = pressure.get_ai("press")

humidity = Arcitecture()
humidity = humidity.get_ai("humid")

cloud_or_wind = Arcitecture()
cloud_or_wind = cloud_or_wind.get_ai("cloud_wind")




ai = keras.Model(input_layer, [temperature, pressure, humidity, cloud_or_wind], name="Weather_Predictor")
ai.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error",
           loss_weights={"temp": 10_000, "press": 1000, "humid": 1000, "cloud_wind": 1000})
           # Отдаём приоритет температуре, и увеличиваем ошибки (иначе они будут <1)

ai.summary(); print()



"""Сохранения / Загрузки"""
def save_path(name): return f"Saves Weather Prophet/{name}"

def SAVE_NAME(num): return f"AI_v2.0~{num}"

# Как загружать: ai = tf.keras.models.load_model(save_path(AI_NAME))
# Как сохранять: ai.save(save_path(AI_NAME))



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
# Берём больше, чем выводим через print_ai_answers
test_size = 10_000

# Разделяем часть для обучения и для тестирования
# В качестве ответа записываем значение природного явления
train_data = DATA_in[:-test_size]
train_data_answer = np.reshape(np.array([DATA_out[:-test_size, 0, :]]), (len(train_data), 1, 4))

test_data = DATA_in[-test_size:]
test_data_answer = np.reshape(np.array([DATA_out[-test_size:, 0, :]]), (test_size, 1, 4))


for learning_cycle in range(0, 99):
    # # ЗАГРУЖАЕМСЯ
    # print(f">>> Loading the {SAVE_NAME(learning_cycle)}", end="\t\t")
    # ai = tf.keras.models.load_model(save_path(SAVE_NAME(learning_cycle)))
    # print("Done\n")
    # learning_cycle += 1


    print(f">>> Learning the {SAVE_NAME(learning_cycle)}")

    ai.fit(train_data, train_data_answer, epochs=3, batch_size=100, verbose=True, shuffle=False)

    print("\n")


    # Сохраняем
    print(f">>> Saving the {SAVE_NAME(learning_cycle)}", end="  ")
    ai.save(save_path(SAVE_NAME(learning_cycle)))
    print("Done (Ignore the WARNING)")


    # Выводим данные и сравниваем их "на глаз"
    print_ai_answers(ai, train_data, 100)
