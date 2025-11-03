import json
import numpy as np
import math

with open('sample_generation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

theta_uniform = 79
m_erlang = 3
theta_erlang = 1/6

def method_of_moments_uniform(sample):
    """
    Метод моментов для дискретного равномерного распределения
    Теоретическое среднее: E[X] = (θ + 1)/2
    Оценка: θ̂ = 2X̄ - 1
    """
    sample_mean = np.mean(sample)
    theta_hat = 2 * sample_mean - 1
    return round(theta_hat, 4)

def maximum_likelihood_uniform(sample):
    """
    Метод максимального правдоподобия для дискретного равномерного распределения
    Оценка: θ̂ = max(X_i)
    """
    theta_hat = max(sample)
    return theta_hat

def method_of_moments_erlang(sample):
    """
    Метод моментов для распределения Эрланга
    Теоретическое среднее: E[X] = m/θ
    Теоретическая дисперсия: Var[X] = m/θ²
    Оценка θ через среднее: θ̂ = m/X̄
    """
    sample_mean = np.mean(sample)
    theta_hat = m_erlang / sample_mean
    return round(theta_hat, 4)

def maximum_likelihood_erlang(sample):
    """
    Метод максимального правдоподобия для распределения Эрланга
    Оценка: θ̂ = m/X̄
    """
    sample_mean = np.mean(sample)
    theta_hat = m_erlang / sample_mean
    return round(theta_hat, 4)

print("ОЦЕНКИ ПАРАМЕТРОВ ДЛЯ ДИСКРЕТНОГО РАВНОМЕРНОГО РАСПРЕДЕЛЕНИЯ I")
print("=" * 80)
print(f"Истинное значение параметра: θ = {theta_uniform}")
print()

sizes = [5, 10, 100, 200, 400, 600, 800, 1000]

print("Объем | Выборка | Метод моментов | Метод макс. правдоподобия | Отклонение ММ | Отклонение ММП")
print("-" * 100)

for size in sizes:
    samples = data['discrete_uniform'][str(size)]
    for i, sample in enumerate(samples, 1):
        mm_estimate = method_of_moments_uniform(sample)
        mle_estimate = maximum_likelihood_uniform(sample)
        
        mm_deviation = abs(mm_estimate - theta_uniform)
        mle_deviation = abs(mle_estimate - theta_uniform)
        
        print(f"{size:5d} | {i:7d} | {mm_estimate:14.1f} | {mle_estimate:25d} | {mm_deviation:13.1f} | {mle_deviation:14.1f}")

print("\n" + "=" * 80)
print("ОЦЕНКИ ПАРАМЕТРОВ ДЛЯ РАСПРЕДЕЛЕНИЯ ЭРЛАНГА")
print("=" * 80)
print(f"Истинное значение параметра: θ = {theta_erlang} (m = {m_erlang})")
print()

print("Объем | Выборка | Метод моментов | Метод макс. правдоподобия | Отклонение ММ | Отклонение ММП")
print("-" * 100)

for size in sizes:
    samples = data['erlang'][str(size)]
    for i, sample in enumerate(samples, 1):
        mm_estimate = method_of_moments_erlang(sample)
        mle_estimate = maximum_likelihood_erlang(sample)
        
        mm_deviation = abs(mm_estimate - theta_erlang)
        mle_deviation = abs(mle_estimate - theta_erlang)
        
        print(f"{size:5d} | {i:7d} | {mm_estimate:14.4f} | {mle_estimate:25.4f} | {mm_deviation:13.4f} | {mle_deviation:14.4f}")

print("\n" + "=" * 80)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ КАЧЕСТВА ОЦЕНОК")
print("=" * 80)

print("\nДИСКРЕТНОЕ РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ:")
print("Объем | Среднее ММ | Среднее ММП | СКО ММ | СКО ММП")
print("-" * 60)

for size in sizes:
    samples = data['discrete_uniform'][str(size)]
    mm_estimates = []
    mle_estimates = []
    
    for sample in samples:
        mm_estimates.append(method_of_moments_uniform(sample))
        mle_estimates.append(maximum_likelihood_uniform(sample))
    
    mm_mean = np.mean(mm_estimates)
    mle_mean = np.mean(mle_estimates)
    mm_std = np.std(mm_estimates)
    mle_std = np.std(mle_estimates)
    
    print(f"{size:5d} | {mm_mean:10.2f} | {mle_mean:11.2f} | {mm_std:6.2f} | {mle_std:7.2f}")

print("\nРАСПРЕДЕЛЕНИЕ ЭРЛАНГА:")
print("Объем | Среднее ММ | Среднее ММП | СКО ММ | СКО ММП")
print("-" * 60)

for size in sizes:
    samples = data['erlang'][str(size)]
    mm_estimates = []
    mle_estimates = []
    
    for sample in samples:
        mm_estimates.append(method_of_moments_erlang(sample))
        mle_estimates.append(maximum_likelihood_erlang(sample))
    
    mm_mean = np.mean(mm_estimates)
    mle_mean = np.mean(mle_estimates)
    mm_std = np.std(mm_estimates)
    mle_std = np.std(mle_estimates)
    
    print(f"{size:5d} | {mm_mean:10.4f} | {mle_mean:11.4f} | {mm_std:6.4f} | {mle_std:7.4f}")