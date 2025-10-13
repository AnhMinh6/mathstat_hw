import random
import math
import json

"""Дискретное равномерное распределение I"""

def discrete_uniform_rand_variable(theta): # Моделирование случайной величины
    s = random.random()
    k = 1
    distribution_func = 1 / theta  # Fs(X) = P(s <= X)
    while s > distribution_func:
        k += 1
        distribution_func = k / theta
    return k

theta_uniform = 79
def discrete_uniform_sample(n):
    sample = []
    for i in range(n):
        k = discrete_uniform_rand_variable(theta_uniform)
        sample.append(k)
    return sample

"""Распределение Эрланга"""

def erlang_rand_variable(m, theta): # Моделирование случайной величины
    k = 0
    for _ in range(m):
        s = random.random()
        while s == 0:
            s = random.random()
        k += -math.log(s) / theta
    return k

m_erlang = 3
theta_erlang = 1/6
def erlang_sample(n):
    sample = []
    for i in range(n):
        k = erlang_rand_variable(m_erlang, theta_erlang)
        sample.append(k)
    return sample

"""1. Генерация выборок выбранных случайных величин"""

size = [5, 10, 100, 200, 400, 600, 800, 1000]
discrete_uniform_result = {}
erlang_result = {}
for i in size:
    if i not in discrete_uniform_result:
        discrete_uniform_result[i] = []
        erlang_result[i] = []
    for j in range(5):
        discrete_uniform_result[i].append(discrete_uniform_sample(i))
        erlang_result[i].append(erlang_sample(i))

# Запись в файл
result = {
    "discrete_uniform": discrete_uniform_result,
    "erlang": erlang_result
}
with open('sample_generation.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)

print("Выборки успешно сгенерированы и сохранены в sample_generation.json")
print(f"Параметры дискретного равномерного распределения: θ = {theta_uniform}")
print(f"Параметры распределения Эрланга: m = {m_erlang}, θ = {theta_erlang}")

# Вывод информации о сгенерированных данных
print("\nИнформация о выборках:")
for dist_name, dist_data in result.items():
    print(f"\n{dist_name}:")
    for size, samples in dist_data.items():
        print(f"  Размер {size}: {len(samples)} выборок")
        if len(samples) > 0:
            first_sample = samples[0]
            print(f"    Первая выборка (первые 5 значений): {first_sample[:5]}")