from numpy import tanh, arctanh
import numpy as np

np.set_printoptions(suppress=True) # Убраем экспонинцеальную запись



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
            processed_data[6] = norm_wind(clamp(float(data[12]) *10, 0, 30))

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
            processed_data[6] = norm_wind(clamp(float(data[6]) * 1_000 / 3_600, 0, 30))

            DATA.append(processed_data)
    return DATA




def print_ai_answers(ai, real_data, batch_size):
    print("\n")
    print("Time\t\t\tReal Data\t\t\t\t\tAI answer\t\t\t\t\tAI answer on AI\t\t\t\tErrors ∆")

    total_errors = []

    rand = np.random.randint(len(real_data) -batch_size)
    real_matrix = np.reshape(np.array([real_data[rand: rand +batch_size]]), (batch_size, 7))

    for b in range(batch_size):
        # Случайный батч
        real_data_list = np.resize(real_matrix[b], (7)).tolist()

        ai_ans_list = np.reshape(np.array(ai.predict([[real_data_list]], verbose=False)), (4)).tolist()
        ai_ans_list = (np.array(ai_ans_list) + np.array(real_data_list[3:])).tolist() # Остаточное обучение

        ai_ans_with_time = np.array(real_data_list[:3] + ai_ans_list)
        ai_on_ai_list = ai.predict(np.reshape(ai_ans_with_time, (1, 1, 7)), verbose=False)
        ai_on_ai_list = np.reshape(np.array(ai_on_ai_list), (4)).tolist()
        ai_on_ai_list = (np.array(ai_on_ai_list) + np.array(real_data_list[3:])).tolist() # Остаточное обучение


        # Конвертируем данные из промежутка [-1; 1] в нормальную физическую величину
        def conv_ai_ans(List):
            return [norm_temperature(List[0], True),
                    norm_pressure(List[1], True),
                    norm_humidity(List[2], True),
                    norm_wind(List[3], True),]

        real_data_list = [
                norm_hours(real_data_list[0], True),
                norm_day(real_data_list[1], True),
                norm_month(real_data_list[2], True),
                norm_temperature(real_data_list[3], True),
                norm_pressure(real_data_list[4], True),
                norm_humidity(real_data_list[5], True),
                norm_wind(real_data_list[6], True),
        ]

        ai_ans_list = conv_ai_ans(ai_ans_list)
        ai_on_ai_list = conv_ai_ans(ai_on_ai_list)


        # В качестве ошибки просто добавляем разность между ответом ИИ и реальностью
        errors = np.array( np.abs(
            np.array(real_data_list[3:]) - np.array(ai_ans_list)
        ))
        total_errors.append(errors)


        # Выводим всё
        print(np.round(np.array(real_data_list[:3]),      1), "\t",
              np.round(np.array(real_data_list[3:]),      1), "\t",
              np.round(np.array(ai_ans_list),             1), "\t",
              np.round(np.array(ai_on_ai_list), 1), "\t",
              np.round(np.array(errors),                       1))


    total_errors = np.array(total_errors)

    print("\nMean errors:",
          "\n\t Temperature:  ", np.round(np.mean(total_errors[:, 0]), 1),
          "\n\t Pressure:     ", np.round(np.mean(total_errors[:, 1]), 1),
          "\n\t Humidity:     ", np.round(np.mean(total_errors[:, 2]), 1),
          "\n\t Wind:         ", np.round(np.mean(total_errors[:, 3]), 1),
          "\n\n\t TOTAL:        ",np.round(np.mean(total_errors     ), 1))

    print("\n")