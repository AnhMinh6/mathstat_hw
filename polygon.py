import matplotlib.pyplot as plt
import numpy as np
import math
import json

"""3. Построение полигона частот и сравнение с теоретическими распределениями"""

# Параметры распределений
theta_uniform = 79
m_erlang = 3
theta_erlang = 1/6

def frequency_polygon_discrete_uniform(size, number, data):
    """
    Специальная функция для построения полигона частот дискретного равномерного распределения
    """
    sample = data['discrete_uniform'][str(size)][number - 1]
    
    # Для дискретного распределения считаем частоты каждого значения
    unique_values, counts = np.unique(sample, return_counts=True)
    
    # Создаем полный диапазон значений от 1 до theta_uniform
    full_range = np.arange(1, theta_uniform + 1)
    full_counts = np.zeros_like(full_range, dtype=float)
    
    # Заполняем частоты для существующих значений
    for i, val in enumerate(unique_values):
        idx = np.where(full_range == val)[0]
        if len(idx) > 0:
            full_counts[idx[0]] = counts[i]
    
    # Построение полигона частот
    plt.plot(full_range, full_counts, 'bo-', markersize=4, linewidth=1, label='Полигон частот')
    
    # Теоретическая частота
    theoretical_freq = np.full_like(full_range, size/theta_uniform, dtype=float)
    plt.plot(full_range, theoretical_freq, 'r-', linewidth=2, label='Теоретическая частота')
    
    plt.title(f'Полигон частот\nДискретное равномерное распределение I, выборка {number}, размер {size}')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(1, theta_uniform + 1, 5))  # Чтобы не загромождать ось X
    plt.show()

def frequency_polygon_erlang(size, number, data):
    """
    Функция для построения полигона частот распределения Эрланга
    """
    sample = data['erlang'][str(size)][number - 1]
    
    # Определяем оптимальное количество интервалов по правилу Старджеса
    n_bins = min(20, int(1 + 3.322 * math.log(size)))
    
    # Построение гистограммы
    frequencies, intervals = np.histogram(sample, bins=n_bins, density=False)
    centers = (intervals[:-1] + intervals[1:]) / 2
    
    # Полигон частот
    plt.plot(centers, frequencies, 'bo-', markersize=4, linewidth=1, label='Полигон частот')
    
    # Теоретическая плотность распределения Эрланга
    x_values = np.linspace(0, max(sample), 100)
    probability_density = (theta_erlang ** m_erlang * x_values ** (m_erlang - 1) * 
                         np.exp(-theta_erlang * x_values)) / math.factorial(m_erlang - 1)
    
    # Масштабирование для сравнения с частотами
    bin_width = (max(sample) - min(sample)) / n_bins
    theoretical_frequencies = probability_density * size * bin_width
    
    plt.plot(x_values, theoretical_frequencies, 'r-', linewidth=2, label='Теоретическая плотность')
    
    plt.title(f'Полигон частот\nРаспределение Эрланга, выборка {number}, размер {size}')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def multiple_frequency_polygons_discrete_uniform(size, data):
    """
    Построение полигонов частот для всех 5 выборок дискретного равномерного распределения
    """
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    plt.figure(figsize=(15, 8))
    
    # Теоретическая частота
    full_range = np.arange(1, theta_uniform + 1)
    theoretical_freq = np.full_like(full_range, size/theta_uniform, dtype=float)
    plt.plot(full_range, theoretical_freq, 'r-', linewidth=3, label='Теоретическая частота')
    
    # Для каждой выборки строим полигон
    for i in range(5):
        sample = data['discrete_uniform'][str(size)][i]
        unique_values, counts = np.unique(sample, return_counts=True)
        
        # Создаем полный диапазон частот
        full_counts = np.zeros_like(full_range, dtype=float)
        for j, val in enumerate(unique_values):
            idx = np.where(full_range == val)[0]
            if len(idx) > 0:
                full_counts[idx[0]] = counts[j]
        
        plt.plot(full_range, full_counts, color=colors[i], marker='o', markersize=2, 
                label=f'Выборка {i+1}', linewidth=1, alpha=0.7)

    plt.title(f'Полигоны частот\nДискретное равномерное распределение I, размер {size}')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(1, theta_uniform + 1, 10))
    plt.show()

def multiple_frequency_polygons_erlang(size, data):
    """
    Построение полигонов частот для всех 5 выборок распределения Эрланга
    """
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    plt.figure(figsize=(12, 8))
    
    # Определяем общий диапазон по всем выборкам
    all_samples = []
    for i in range(5):
        sample = data['erlang'][str(size)][i]
        all_samples.extend(sample)
    
    n_bins = min(20, int(1 + 3.322 * math.log(size)))
    
    # Теоретическая плотность
    x_values = np.linspace(min(all_samples), max(all_samples), 100)
    probability_density = (theta_erlang ** m_erlang * x_values ** (m_erlang - 1) * 
                         np.exp(-theta_erlang * x_values)) / math.factorial(m_erlang - 1)
    
    bin_width = (max(all_samples) - min(all_samples)) / n_bins
    theoretical_frequencies = probability_density * size * bin_width
    
    plt.plot(x_values, theoretical_frequencies, 'r-', linewidth=3, label='Теоретическая плотность')
    
    # Для каждой выборки строим полигон
    for i in range(5):
        sample = data['erlang'][str(size)][i]
        frequencies, intervals = np.histogram(sample, bins=n_bins, density=False)
        centers = (intervals[:-1] + intervals[1:]) / 2
        plt.plot(centers, frequencies, color=colors[i], marker='o', markersize=3, 
                label=f'Выборка {i+1}', linewidth=1.5, alpha=0.7)

    plt.title(f'Полигоны частот\nРаспределение Эрланга, размер {size}')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Основная программа
if __name__ == "__main__":
    with open('sample_generation.json', 'r') as f:
        data = json.load(f)
    
    sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
    
    print("Построение полигонов частот для отдельных выборок...")
    
    # Для дискретного равномерного распределения
    for size in sizes:
        print(f"Дискретное равномерное распределение, размер {size}")
        frequency_polygon_discrete_uniform(size, 1, data)
    
    # Для распределения Эрланга
    for size in sizes:
        print(f"Распределение Эрланга, размер {size}")
        frequency_polygon_erlang(size, 1, data)
    
    print("\nПостроение полигонов частот для всех выборок каждого размера...")
    
    # Для дискретного равномерного распределения
    for size in sizes:
        print(f"Дискретное равномерное распределение, все выборки размера {size}")
        multiple_frequency_polygons_discrete_uniform(size, data)
    
    # Для распределения Эрланга
    for size in sizes:
        print(f"Распределение Эрланга, все выборки размера {size}")
        multiple_frequency_polygons_erlang(size, data)