import random
import math
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def check_and_load_data():
    """Проверяем существование файла и загружаем данные"""
    
    # Проверяем текущую директорию
    current_dir = os.getcwd()
    print(f"Текущая директория: {current_dir}")
    
    # Проверяем файлы в текущей директории
    files = os.listdir(current_dir)
    print("Файлы в текущей директории:")
    for file in files:
        if file.endswith('.json'):
            print(f"  - {file}")
    
    # Пытаемся найти файл
    filename = 'sample_generation.json'
    
    if os.path.exists(filename):
        print(f"Файл {filename} найден!")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print(f"Файл {filename} не найден в текущей директории!")
        print("Создаем новый файл с выборками...")
        return generate_and_save_samples()

def generate_and_save_samples():
    """Генерация всех выборок и сохранение в JSON"""
    # Параметры распределений
    theta_uniform = 79
    m_erlang = 3
    theta_erlang = 1/6
    
    sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
    num_repeats = 5
    
    discrete_uniform_result = {}
    erlang_result = {}
    
    print("Генерация выборок...")
    for size in sample_sizes:
        print(f"Размер {size}...")
        discrete_uniform_result[str(size)] = []
        erlang_result[str(size)] = []
        
        for j in range(num_repeats):
            # Генерация для дискретного равномерного распределения
            sample_uniform = []
            for i in range(size):
                k = discrete_uniform_rand_variable(theta_uniform)
                sample_uniform.append(k)
            discrete_uniform_result[str(size)].append(sample_uniform)
            
            # Генерация для распределения Эрланга
            sample_erlang = []
            for i in range(size):
                k = erlang_rand_variable(m_erlang, theta_erlang)
                sample_erlang.append(k)
            erlang_result[str(size)].append(sample_erlang)
    
    # Сохранение в JSON
    result = {
        "discrete_uniform": discrete_uniform_result,
        "erlang": erlang_result
    }
    
    with open('sample_generation.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    
    print("Файл sample_generation.json успешно создан!")
    return result

def discrete_uniform_rand_variable(theta):
    """Моделирование случайной величины для дискретного равномерного распределения"""
    s = random.random()
    k = 1
    distribution_func = 1 / theta
    while s > distribution_func:
        k += 1
        distribution_func = k / theta
    return k

def erlang_rand_variable(m, theta):
    """Моделирование случайной величины для распределения Эрланга"""
    k = 0
    for _ in range(m):
        s = random.random()
        while s == 0:
            s = random.random()
        k += -math.log(s) / theta
    return k

"""2. Построение эмпирических функций распределения"""

# Параметры распределений
theta_uniform = 79
m_erlang = 3
theta_erlang = 1/6

def empirical_func(distribution, size, number, data): 
    """Поиск значений для создания графика"""
    # Используем переданные данные вместо загрузки из файла
    
    sample = data[distribution][str(size)][number-1]
    dist = []
    
    if distribution == 'discrete_uniform':
        name = 'Дискретное равномерное распределение I'
        # Теоретическая функция распределения для дискретного равномерного
        k_values = np.arange(0, theta_uniform + 2)  # От 0 до theta+1 для полного покрытия
        distribution_law = []
        for k in k_values:
            if k < 1:
                F = 0
            elif k >= theta_uniform:
                F = 1
            else:
                F = k / theta_uniform
            distribution_law.append(F)
        dist.append(k_values)
        dist.append(distribution_law)
        
    elif distribution == 'erlang':
        name = 'Распределение Эрланга'
        x_values = np.linspace(0, max(sample) + 5, 100)  # +5 для лучшего отображения хвоста
        probability = []
        for x in x_values:
            # Теоретическая функция распределения Эрланга
            sum_term = 0
            for k in range(m_erlang):
                sum_term += (theta_erlang * x) ** k / math.factorial(k)
            F = 1 - math.exp(-theta_erlang * x) * sum_term
            probability.append(F)
        dist.append(x_values)
        dist.append(probability)
    
    # Эмпирическая функция распределения
    sorted_sample = np.sort(sample)
    F_empirical = np.arange(1, size + 1) / size

    return [sorted_sample, F_empirical, name, dist]

def separate_graph(distribution, size, number, data):
    """Функция для построения отдельных графиков"""
    data_empirical = empirical_func(distribution, size, number, data)

    plt.figure(figsize=(10, 6))
    
    # Эмпирическая функция распределения (ступенчатая)
    plt.step(data_empirical[0], data_empirical[1], color='blue', where='post', label='ЭФР', linewidth=2)
    
    # Теоретическая функция распределения
    if distribution == 'discrete_uniform':
        # Для дискретного равномерного - ступенчатая функция
        plt.step(data_empirical[3][0], data_empirical[3][1], color='red', where='post', 
                label='Теоретическая ФР', linewidth=2, linestyle='--')
    else:
        # Для Эрланга - непрерывная функция
        plt.plot(data_empirical[3][0], data_empirical[3][1], color='red', label='Теоретическая ФР', linewidth=2)

    plt.title(f'Эмпирическая функция распределения\n{data_empirical[2]}, выборка {number}, размер {size}')
    plt.xlabel('t')
    plt.ylabel('F(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.show()

def all_graphs(distribution, size, data):
    """Функция для построения графиков для всех выборок распределения одного размера"""
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    plt.figure(figsize=(12, 8))
    
    # Сначала рисуем теоретическую ФР (общую для всех)
    data_first = empirical_func(distribution, size, 1, data)
    
    if distribution == 'discrete_uniform':
        plt.step(data_first[3][0], data_first[3][1], color='red', where='post', 
                label='Теоретическая ФР', linewidth=3, linestyle='--')
    else:
        plt.plot(data_first[3][0], data_first[3][1], color='red', 
                label='Теоретическая ФР', linewidth=3)
    
    # Затем рисуем все эмпирические ФР
    for i in range(1, 6):  # 5 выборок
        data_empirical = empirical_func(distribution, size, i, data)
        plt.step(data_empirical[0], data_empirical[1], color=colors[i-1], where='post', 
                label=f'ЭФР {i}', alpha=0.7, linewidth=1.5)

    plt.title(f'Эмпирические функции распределения\n{data_first[2]}, размер {size}')
    plt.xlabel('t')
    plt.ylabel('F(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.show()

"""Вычисление Dm,n"""

def calculation_D(distribution, size_m, size_n, number_m, number_n, data):
    """Вычисление статистики D между двумя выборками"""
    F_m = empirical_func(distribution, size_m, number_m, data)
    F_n = empirical_func(distribution, size_n, number_n, data)

    # Объединяем точки из двух функций
    all_values = []
    all_values.extend(F_n[0])
    all_values.extend(F_m[0])
    all_values = sorted(set(all_values))

    max_sup = 0
    for value in all_values:
        # Находим значение эмпирической ФР для выборки n в точке value
        value_n = 0
        for i in range(len(F_n[0])):
            x = F_n[0][i]  # точка скачка - ось х
            if x <= value:
                value_n = F_n[1][i]  # значение ЭФР - ось у
            else:
                break
        
        # Находим значение эмпирической ФР для выборки m в точке value
        value_m = 0
        for i in range(len(F_m[0])):
            x = F_m[0][i]
            if x <= value:
                value_m = F_m[1][i]
            else:
                break

        sup = abs(value_n - value_m)
        if sup > max_sup:
            max_sup = sup

    D = np.sqrt((size_n * size_m) / (size_n + size_m)) * max_sup
    D = round(D, 6)
    return D

def calculate_all_D_statistics(distribution, data):
    """Вычисление всех статистик D для всех пар объемов и всех пар выборок"""
    sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
    results = {}
    
    print(f"Вычисление статистик D для {distribution} распределения:")
    
    for i, size_m in enumerate(sizes):
        for j, size_n in enumerate(sizes):
            if i <= j:  # Чтобы избежать дублирования (D_{m,n} = D_{n,m})
                key = f"D_{size_m}_{size_n}"
                d_values = []
                
                # Вычисляем D для всех пар выборок с объемами size_m и size_n
                for num_m in range(1, 6):  # 5 выборок для размера m
                    for num_n in range(1, 6):  # 5 выборок для размера n
                        d = calculation_D(distribution, size_m, size_n, num_m, num_n, data)
                        d_values.append(d)
                
                mean_d = np.mean(d_values)
                std_d = np.std(d_values)
                results[key] = {
                    'mean': round(mean_d, 6),
                    'std': round(std_d, 6),
                    'min': round(min(d_values), 6),
                    'max': round(max(d_values), 6)
                }
                
                print(f"{key}: среднее = {mean_d:.4f}, std = {std_d:.4f}")
    
    return results

if __name__ == "__main__":
    data = check_and_load_data()
    
    print("\nПостроение отдельных графиков для всех размеров выборок...")
    
    sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
    
    for size in sizes:
        print(f"Дискретное равномерное распределение, размер {size}")
        separate_graph('discrete_uniform', size, 1, data)
    
    for size in sizes:
        print(f"Распределение Эрланга, размер {size}")
        separate_graph('erlang', size, 1, data)
    
    print("\nПостроение графиков для всех выборок каждого размера...")
    
    for size in sizes:
        print(f"Дискретное равномерное распределение, все выборки размера {size}")
        all_graphs('discrete_uniform', size, data)
    
    for size in sizes:
        print(f"Распределение Эрланга, все выборки размера {size}")
        all_graphs('erlang', size, data)
    
    print("\nВычисление статистик D...")
    
    uniform_d_stats = calculate_all_D_statistics('discrete_uniform', data)
    
    erlang_d_stats = calculate_all_D_statistics('erlang', data)
    
    results = {
        'discrete_uniform': uniform_d_stats,
        'erlang': erlang_d_stats
    }
    
    with open('d_statistics_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nРезультаты сохранены в файл 'd_statistics_results.json'")