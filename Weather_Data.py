from numpy import tanh, arctanh
import numpy as np
import time
from time import time as Time
import datetime as dt
from urllib.request import urlretrieve
import gzip
import os
import tqdm

np.set_printoptions(suppress=True) # Убраем экспонинцеальную запись
have_cloud = False

def norm_temperature(x, convert_back=False):
    """Нормализуем температуру от -1 до 1"""
    if convert_back:
        return round(arctanh(x)*20, 1)

    return tanh(x/20)

def norm_pressure(x, convert_back=False):
    """Нормализуем давление от -1 до 1"""
    if convert_back:
        return round(arctanh(x) *20 +750, 1)

    return tanh( (x -750)/20 )

def norm_humidity(x, convert_back=False):
    """Нормализуем влажность от -1 до 1"""
    if convert_back:
        return int(x *50 +50)

    return (x -50)/50

def norm_cloud(x, convert_back=False):
    """Нормализуем облачность от -1 до 1"""
    if convert_back:
        return int(x *50 +50)

    return (x -50)/50

def norm_wind(x, convert_back=False):
    """Нормализуем скорость ветра от -1 до 1"""
    if convert_back:
        return round(arctanh(x) *6 +15, 1)

    return tanh((x -15) /6)

def norm_hours(x, convert_back=False):
    """Нормализуем время суток от -1 до 1"""
    if convert_back:
        return int(x*12 +12)

    return (x -12)/12

def norm_day(x, convert_back=False):
    """Нормализуем номер дня от -1 до 1"""
    if convert_back:
        return int(x *15.5 +15.5)

    return (x -15.5) /15.5

def norm_month(x, convert_back=False):
    """Нормализуем номер месяца от -1 до 1"""
    if convert_back:
        return int(x *6 +6)

    return (x -6)/6



def clamp(num, Min, Max):
    return min(max(Min, num), Max)

def normalize(x, convert_back=False):
    """Нормализуем данные от -1 до 1"""

    if convert_back:
        with open("Datasets/Info_About_Last_Dataset.txt", "r") as save:
            save = save.read().split("\n")
            MIN_DATA = float(save[1][4:])
            MAX_DATA = float(save[2][4:])

        result = (x +1) / 2
        result = result * (MAX_DATA - MIN_DATA) + MIN_DATA
        return result


    # Сохраняем информацию о том, как потом нормализовать данные обратно
    with open("Datasets/Info_About_Last_Dataset.txt", "w+") as save:
        save.write(f"Data set data for last saved AI\n"
                   f"MIN={np.min(x)}\n"
                   f"MAX={np.max(x)}")

    # Сначала нормализуем от 0 до 1
    result = x - np.min(x)
    if np.max(x) != 0.0:
        result = result / np.max(result)

    # Потом от -1 до 1
    result = result *2 -1

    return result


def conv_ai_ans(List):
    return [
        norm_temperature(List[0], True),
        norm_pressure(List[1], True),
        norm_humidity(List[2], True),
        norm_cloud(List[3], True) if have_cloud else norm_wind(List[3], True),
    ]




def get_moscow_data():
    """
    -> out
    0) Время (часы)
    1) Время (день)
    2) Время (месяц)
    3) Температура
    4) Давление (мм.рт.ст.)
    5) Влажность
    6) Облачность
    """

    have_cloud = True
    DATA = []
    for NAME_DATASET in ["Москва (ВДНХ)", "Москва (Центр)", "Москва (Аэропорт)"]:
        with open(f"Datasets/{NAME_DATASET}.csv") as dataset:
            # Без первой строки с начальным символом;
            # Без первой (идём от староого к новому) записи, т.к. она используется для смещения ответа
            # (т.е. чтобы мы на основе предыдущей записи создавали следующую)
            records = dataset.readlines()[1:][::-1]

            for string in records:
                data = string.split(";")[:-2]

                # Если попался брак, то пропускаем шаг
                if '' in data or len(data) != 5:
                    continue

                processed_data = [0 for _ in range(7)]

                # Преобразуем строку
                # data[0] -> часы (в течении дня)
                # datap[1] -> день
                # data[2] -> месяц
                processed_data[0] = norm_hours(int(data[0][11:13]))
                processed_data[1] = norm_day(int(data[0][:2]))
                processed_data[2] = norm_month(int(data[0][3:5]))
                processed_data[3] = norm_temperature(clamp(float(data[1].replace(",", ".")), -40, 40))
                processed_data[4] = norm_pressure(clamp(float(data[2].replace(",", ".")), 700, 800))
                processed_data[5] = norm_humidity(int(data[3]))
                processed_data[6] = norm_cloud(int(data[4]))

                DATA.append(processed_data)

    # Заполняем промежуточными значеними (т.к. у нас данные идут с шагом в 3 часа)
    DATA = np.array(DATA)
    conv_DATA = []
    for i in range(len(DATA) - 1):
        for conved in np.linspace(DATA[i], DATA[i + 1], num=4).tolist()[1:]:
            conv_DATA.append(conved)
    DATA = conv_DATA

    return DATA


def get_plank_history():
    """
    <- inp
    0) Время
    1) Давление (паскаль)
    2) ℃
    3) Температура (Кельвин)
    4) Точка росы
    5) Относительная влажность
    6) Давление пара насыщения
    7) Давление газа
    8) Дефицит давления пара
    9) Удельная влажность
    10) Концентрация водяного пара
    11) Герметичный (?)
    12) Скорость ветра (м/с)
    13) Максимальная скорость ветра
    14) Направление ветра в градусах


    -> out
    0) Время (часы) (float)
    1) Время (день)
    2) Время (месяц)
    3) ℃
    4) Давление (мм.рт.ст.)
    5) Влажность (%)
    6) Скорость ветра (м/с)
    """

    DATA = []
    with open("Datasets/max_planck_weather_ts.txt") as file:
        for string in file.readlines():
            data = string.split(",")

            # Если попался брак, то пропускаем шаг
            if '' in data or len(data) != 15: continue

            processed_data = [0 for _ in range(7)]

            processed_data[0] = norm_hours(int(data[0][11:13]) + int(data[0][14:16]) /60)
            processed_data[1] = norm_day(int(data[0][:2]))
            processed_data[2] = norm_month(int(data[0][3:5]))
            processed_data[3] = norm_temperature(clamp(float(data[2]), -40, 40))
            processed_data[4] = norm_pressure(clamp(float(data[1]) *100 / 133.322, 700, 800))
            processed_data[5] = norm_humidity(float(data[5]))
            processed_data[6] = norm_wind(clamp(float(data[12]) *10, 0, 25))

            DATA.append(processed_data)
    return DATA


def get_weather_history():
    """
    <- inp
    0) Время
    1) Описание
    2) Тип осадков
    3) Температура ℃
    4) Кажущаяся температура ℃
    5) Влажность
    6) Скорость ветра (km/h)
    7) Направление ветра °
    8) Видимость (km)
    9) Loud Cover
    10) Давление (миллибар)
    11) Ежедневная сводка


    -> out
    0) Время (часы)
    1) Время (день)
    2) Время (месяц)
    3) Температура
    4) Давление (мм.рт.ст.)
    5) Влажность (%)
    6) Скорость ветра (м/c)
    """

    DATA = []
    with open("Datasets/weatherHistory.txt") as file:
        for string in file.readlines():
            data = string.split(",")

            # Если попался брак, то пропускаем шаг
            if '' in data or len(data) != 12: continue

            processed_data = [0 for _ in range(7)]

            processed_data[0] = norm_hours(int(data[0][11:13]))
            processed_data[1] = norm_day(int(data[0][8:10]))
            processed_data[2] = norm_month(int(data[0][5:7]))
            processed_data[3] = norm_temperature(clamp(float(data[3]), -40, 40))
            processed_data[4] = norm_pressure(clamp(float(data[10]) * 0.750063755419211, 700, 800))
            processed_data[5] = norm_humidity(float(data[5]) * 100)
            processed_data[6] = norm_wind(clamp(float(data[6]) * 1_000 / 3_600, 0, 25))

            DATA.append(processed_data)
    return DATA


def get_fresh_data(how_days=60):
    now_date = dt.datetime.today()
    last_date = now_date - dt.timedelta(days=how_days)     # Берём промежутк в how_days дней

    now_date = now_date.strftime("%d.%m.%Y")
    last_date = last_date.strftime("%d.%m.%Y")

    # Скачиваем архивчик
    for i in range(1, 10):
        try:
            urlretrieve(f"https://ru{i}.rp5.ru/download/files.synop/27/27612." \
                            f"{last_date}.{now_date}.1.0.0.ru.utf8.00000000.csv.gz",
                        f"Datasets/FRESH_ARCHIVE.csv.gz")
        except:
            pass
        else:
            break


    # Загружаем данные
    with open("Datasets/FRESH_ARCHIVE.csv.gz", "rb") as byte_file:
        data = str(gzip.decompress(byte_file.read()), "utf-8")

        data = [string.split(";") for string in data.split("\n")[7:]]

        # Убираем всякие "", ' ', '\r', '', '""'
        processed_data = []
        for record in data:
            to_append = []
            for i in record:
                val = i.replace('"', '').replace('\r', '')
                # if val == "" or val == " " or val == '""' or val == '\r':
                #     pass
                # else:
                to_append.append(val)
            processed_data.append(to_append)
        data = processed_data[:-1]

        # Оставляем только необходимые данные (ещё будет влажность)
        required_data = []
        for d in data:
            to_app_required_data = [d[0], d[1], d[2], d[5]]

            # Влажность...
            humidity = d[10].replace('.', '').replace('%', '').split()
            if '–' in humidity[0]:
                humidity[0] = humidity[0].split('–')
                humidity = humidity[0]

            #  качестве влажности выбираем максимальое значение
            for ind, h in enumerate(humidity):
                if h.isnumeric():
                    humidity[ind] = int(h)
                else:
                    humidity[ind] = 0

            to_app_required_data += [str(max(humidity))]

            required_data.append(to_app_required_data)

    # Удаляем архив
    os.remove("Datasets/FRESH_ARCHIVE.csv.gz")



    # Переводим строки в числа для ИИшки
    DATA = []
    for data in required_data:
        # Если попался брак, то пропускаем шаг
        if '' in data or len(data) != 5: continue

        processed_data = [0 for _ in range(7)]

        # Преобразуем строку
        # data[0] -> часы (в течении дня)
        # datap[1] -> день
        # data[2] -> месяц
        processed_data[0] = norm_hours(int(data[0][11:13]))
        processed_data[1] = norm_day(int(data[0][:2]))
        processed_data[2] = norm_month(int(data[0][3:5]))
        processed_data[3] = norm_temperature(clamp(float(data[1].replace(",", ".")), -40, 40))
        processed_data[4] = norm_pressure(clamp(float(data[2].replace(",", ".")), 700, 800))
        processed_data[5] = norm_humidity(int(data[3]))
        processed_data[6] = norm_cloud(int(data[4]))

        DATA.append(processed_data)

    # Заполняем промежуточными значеними (т.к. у нас данные идут с шагом в 3 часа)
    DATA = np.array(DATA)
    conv_DATA = []
    for i in range(len(DATA) - 1):
        for conved in np.linspace(DATA[i], DATA[i + 1], num=4).tolist()[1:]:
            conv_DATA.append(conved)
    DATA = conv_DATA

    return DATA





def print_ai_answers(ai, real_data, batch_size):
    print("\n")
    print("Time\t\t\tReal Data\t\t\t\t\tAI answer\t\t\t\t\tErrors ∆")

    total_errors = []

    rand = np.random.randint(len(real_data) -batch_size)
    real_matrix = np.reshape(np.array([real_data[rand: rand +batch_size]]), (batch_size, 7))

    for b in range(1, batch_size):
        # Случайный батч
        real_data_list = np.resize(real_matrix[b], (7)).tolist()

        ai_ans_list = np.reshape(np.array(ai.predict([[real_data_list]], verbose=False)), (4))
        # Не забываем про остаточное обучение
        ai_ans_list = normalize(ai_ans_list, True)
        ai_ans_list = (ai_ans_list + real_matrix[b][3:]).tolist()

        # Конвертируем данные из промежутка [-1; 1] в нормальную физическую величину
        real_data_list = [
                norm_hours(real_data_list[0], True),
                norm_day(real_data_list[1], True),
                norm_month(real_data_list[2], True),
        ] + conv_ai_ans(real_data_list[3:])

        ai_ans_list = conv_ai_ans(ai_ans_list)

        # В качестве ошибки просто добавляем разность между ответом ИИ и реальностью
        errors = np.array( np.abs(
            np.array(real_data_list[3:]) - np.array(ai_ans_list)
        ))
        total_errors.append(errors)


        # Выводим всё
        print(np.round(np.array(real_data_list[:3]),      1), "\t",
              np.round(np.array(real_data_list[3:]),      1), "\t",
              np.round(np.array(ai_ans_list),             1), "\t",
              np.round(np.array(errors),                       1))


    total_errors = np.array(total_errors)

    print("\nMean errors:",
          "\n\t Temperature:  ", np.round(np.mean(total_errors[:, 0]), 1),
          "\n\t Pressure:     ", np.round(np.mean(total_errors[:, 1]), 1),
          "\n\t Humidity:     ", np.round(np.mean(total_errors[:, 2]), 1),
          "\n\t Wind:         ", np.round(np.mean(total_errors[:, 3]), 1),
          "\n\n\t TOTAL:        ",np.round(np.mean(total_errors     ), 1))

    print("\n")



def print_weather_predict(ai, len_predict_days=3):
    print(f"Prediction for the next {len_predict_days} days:\t\t\t",
          f"(Temperature, Pressure, Humidity, {'Cloud' if have_cloud else 'Wind'}",
          f"in ℃, mmHg, %, {'%' if have_cloud else 'm/s'})",)


    predict_for_ai = np.array([[i] for i in get_fresh_data()][::-1])

    # Строим прогноз
    for i in range(len_predict_days *24):
        # Скармливаем ИИшке данные за все предыдущие дни
        ai_pred = ai.predict_on_batch(predict_for_ai)
        ai_pred = np.reshape(np.array(ai_pred)[:, -1], (4))
        ai_pred = ai_pred + np.array(predict_for_ai[-1, 0, 3:])  # Остаточное обучение

        # Обновляем время
        time = predict_for_ai[-1, 0][:3]
        time[0] += 1/12                          # Увеличиваем часы
        time[1] += 1/15.5 if time[0] >1 else 0   # Увеличиваем день
        time[2] += 1/6    if time[1] >1 else 0   # Увеличиваем месяц

        # Следим, чтобы зачения не выходили за границы
        time = [-1 if i>1 else i for i in time]


        # Добавляем время к ответу ИИ
        # predict_for_ai.append([time + ai_pred.tolist()])
        predict_for_ai = np.insert(predict_for_ai, [predict_for_ai.shape[0]],
                                   [time + ai_pred.tolist()], axis=0)

        # Выводим прогноз
        time_for_human = [norm_hours(time[0], True),
                          norm_day(time[1], True),
                          norm_month(time[2], True)]

        predictional_weather = conv_ai_ans(ai_pred)
        print(f"{'{:02}'.format(time_for_human[1])}.{'{:02}'.format(time_for_human[2])}",
              f"{'{:02}'.format(time_for_human[0])}:00:\t",
              predictional_weather[0],
              predictional_weather[1],
              predictional_weather[2],
              predictional_weather[3],
              )

    print("\n")