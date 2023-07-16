from numpy import tanh, arctanh
import numpy as np


def norm_temperature(x, convert_back=False):
    """Нормализуем температуру от -1 до 1"""
    if convert_back:
        return round(arctanh(x)*20, 1)

    return tanh(x/20)

def norm_pressure(x, convert_back=False):
    """Нормализуем давление от -1 до 1"""
    if convert_back:
        return round(arctanh(x) *15 +755, 1)

    return tanh( (x -755)/15 )

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
        return round(arctanh(x) *2 +2, 1)

    return tanh((x -2) /2)

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

    DATA = []
    for NAME_DATASET in ["Москва (ВДНХ)", "Москва (Центр)", "Москва (Аэропорт)"]:
        with open(f"Datasets/{NAME_DATASET}.csv") as dataset:
            # Без первой строки с начальным символом;
            # Без первой (идём от староого к новому) записи, т.к. она используется для смещения ответа
            # (т.е. чтобы мы на основе предыдущей записи создавали следующую)
            records = dataset.readlines()[1:][::-1]
            len_file = len(records)
            num_data = 0

            for string in records:
                data = string.split(";")[:-2]

                # Если попался брак, то пропускаем шаг
                if '' in data or len(data) != 7:
                    continue

                processed_data = [0 for _ in range(7)]

                # Преобразуем строку
                # data[0] -> часы (в течении дня)
                # datap[1] -> день
                # data[2] -> месяц
                processed_data[0] = norm_hours(int(data[0][11:13]))
                processed_data[1] = norm_day(int(data[0][:2]))
                processed_data[2] = norm_month(int(data[0][3:5]))
                processed_data[3] = norm_temperature(float(data[1].replace(",", ".")))
                processed_data[4] = norm_pressure(float(data[2].replace(",", ".")))
                processed_data[5] = norm_humidity(int(data[3]))
                processed_data[6] = norm_cloud(int(data[4]))

                DATA.append(processed_data)
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
            processed_data[3] = norm_temperature(float(data[2]))
            processed_data[4] = norm_pressure(float(data[1]) *100 / 133.322)
            processed_data[5] = norm_humidity(float(data[5]))
            processed_data[6] = norm_wind(float(data[12]) *10)

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
            processed_data[3] = norm_temperature(float(data[3]))
            processed_data[4] = norm_pressure(float(data[10]) * 0.750063755419211)
            processed_data[5] = norm_humidity(float(data[5]) * 100)
            processed_data[6] = norm_wind(float(data[6]) * 1_000 / 3_600)

            DATA.append(processed_data)
    return DATA




def print_ai_answers(ai, real_data, learning_cycle, batch_size):
    print("\n")
    print("Time\t\t\tReal Data\t\t\t\t\tAI answer\t\t\t\t\tAI answer on AI\t\t\t\tErrors ∆")

    # Случайный батч
    real_matrix = np.array([real_data[np.random.randint(len(real_data))]])

    real_data_list = real_matrix.tolist()[0]
    ai_ans_list = np.reshape(np.array(ai.predict(real_matrix, verbose=False)), (batch_size, 4)).tolist()

    ai_ans_with_time = np.array( [time + ans for time, ans in zip(
                                 [time[:3] for time in real_data_list], ai_ans_list)])
    ai_ans_on_ai_list = ai.predict( np.reshape(ai_ans_with_time, (1, batch_size, 7)), verbose=False)
    ai_ans_on_ai_list = np.reshape(np.array(ai_ans_on_ai_list), (batch_size, 4)).tolist()


    # Конвертируем данные из промежутка [-1; 1] в нормальную физическую величину
    converted_real_data = []
    for record in real_data_list:
        converted_real_data.append([
            norm_hours(record[0], True),
            norm_day(record[1], True),
            norm_month(record[2], True),
            norm_temperature(record[3], True),
            norm_pressure(record[4], True),
            norm_humidity(record[5], True),
            norm_wind(record[6], True),
        ])

    converted_ai_ans_list = []
    for ai_ans in ai_ans_list:
        converted_ai_ans_list.append([
            norm_temperature(ai_ans[0], True),
            norm_pressure(ai_ans[1], True),
            norm_humidity(ai_ans[2], True),
            norm_wind(ai_ans[3], True),
        ])

    converted_ai_ans_on_ai_list = []
    for ai_ans_on_ai in ai_ans_list:
        converted_ai_ans_on_ai_list.append([
            norm_temperature(ai_ans_on_ai[0], True),
            norm_pressure(ai_ans_on_ai[1], True),
            norm_humidity(ai_ans_on_ai[2], True),
            norm_wind(ai_ans_on_ai[3], True),
        ])


    # В качестве ошибки просто добавляем разность между ответом ИИ и реальностью
    errors = []
    for real, ai_ans in zip(converted_real_data, converted_ai_ans_list):
        real = np.array(real[3:])
        ai_ans = np.array(ai_ans)

        errors.append(np.abs(real - ai_ans))
    errors = np.array(errors)


    # Выводим всё
    for real, ai_ans, ai_on_ai, err in zip(converted_real_data,
              converted_ai_ans_list, converted_ai_ans_on_ai_list, errors):
        print(np.round(np.array(real[:3]),  1), "\t",
              np.round(np.array(real[3:]),  1), "\t",
              np.round(np.array(ai_ans),    1), "\t",
              np.round(np.array(ai_on_ai),  1), "\t",
              np.round(np.array(err),       1))


    print("\nMean errors:",
          "\n\t Temperature:  ", np.round(np.mean(errors[:, 0]), 1),
          "\n\t Pressure:     ", np.round(np.mean(errors[:, 1]), 1),
          "\n\t Humidity:     ", np.round(np.mean(errors[:, 2]), 1),
          "\n\t Wind:         ", np.round(np.mean(errors[:, 3]), 1),
          "\n\n\t TOTAL:        ", np.round(np.mean(errors     ),  1))

    print("\n")
