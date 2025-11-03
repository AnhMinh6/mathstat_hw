import json
import numpy as np
import math

with open('sample_generation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

theta_uniform = 79
m_erlang = 3
theta_erlang = 1/6

def optimal_estimator_uniform(sample, tau_type='variance'):
    """Оптимальные оценки для дискретного равномерного распределения"""
    n = len(sample)
    
    if tau_type == 'variance':
        # τ(θ) = Var[X] = (θ² - 1)/12
        sample_var = np.var(sample, ddof=0)
        tau_hat = sample_var
        theta_hat = np.sqrt(12 * sample_var + 1)
        return round(tau_hat, 4), round(theta_hat, 4)
    
    elif tau_type == 'probability':
        # τ(θ) = P(X ≤ k) для фиксированного k
        k = 40
        tau_hat = np.mean(np.array(sample) <= k)
        theta_hat = k / tau_hat if tau_hat > 0 else theta_uniform
        return round(tau_hat, 4), round(theta_hat, 4)

def optimal_estimator_erlang(sample, tau_type='coefficient_of_variation'):
    """Оптимальные оценки для распределения Эрланга"""
    n = len(sample)
    sample_mean = np.mean(sample)
    
    if tau_type == 'coefficient_of_variation':
        # τ(θ) = CV[X] = σ/μ = 1/√m
        sample_std = np.std(sample, ddof=0)
        tau_hat = sample_std / sample_mean if sample_mean > 0 else 0
        theta_hat = m_erlang / sample_mean
        return round(tau_hat, 4), round(theta_hat, 4)

print("ОПТИМАЛЬНЫЕ ОЦЕНКИ ДЛЯ ДИСКРЕТНОГО РАВНОМЕРНОГО РАСПРЕДЕЛЕНИЯ I")
print("=" * 90)
print(f"Истинное значение параметра: θ = {theta_uniform}")
print(f"Функция τ(θ) = Var[X] = (θ² - 1)/12 = {(theta_uniform**2 - 1)/12:.1f}")
print()

sizes = [5, 10, 100, 200, 400, 600, 800, 1000]

print("Объем | Выборка | Оценка τ(θ) | Оценка θ из τ(θ) | Отклонение τ | Отклонение θ")
print("-" * 100)

uniform_results = []
for size in sizes:
    samples = data['discrete_uniform'][str(size)]
    for i, sample in enumerate(samples, 1):
        tau_hat, theta_hat = optimal_estimator_uniform(sample, 'variance')
        true_tau = (theta_uniform**2 - 1) / 12
        tau_dev = abs(tau_hat - true_tau)
        theta_dev = abs(theta_hat - theta_uniform)
        
        uniform_results.append({'size': size, 'tau_hat': tau_hat, 'theta_hat': theta_hat})
        print(f"{size:5d} | {i:7d} | {tau_hat:11.1f} | {theta_hat:16.1f} | {tau_dev:12.1f} | {theta_dev:11.1f}")

print("\n" + "=" * 90)
print("ОПТИМАЛЬНЫЕ ОЦЕНКИ ДЛЯ РАСПРЕДЕЛЕНИЯ ЭРЛАНГА")
print("=" * 90)
print(f"Истинное значение параметра: θ = {theta_erlang:.4f} (m = {m_erlang})")
print(f"Функция τ(θ) = CV[X] = σ/μ = 1/√m = {1/math.sqrt(m_erlang):.4f}")
print()

print("Объем | Выборка | Оценка τ(θ) | Оценка θ из τ(θ) | Отклонение τ | Отклонение θ")
print("-" * 100)

erlang_results = []
for size in sizes:
    samples = data['erlang'][str(size)]
    for i, sample in enumerate(samples, 1):
        tau_hat, theta_hat = optimal_estimator_erlang(sample, 'coefficient_of_variation')
        true_tau = 1 / math.sqrt(m_erlang)
        tau_dev = abs(tau_hat - true_tau)
        theta_dev = abs(theta_hat - theta_erlang)
        
        erlang_results.append({'size': size, 'tau_hat': tau_hat, 'theta_hat': theta_hat})
        print(f"{size:5d} | {i:7d} | {tau_hat:11.4f} | {theta_hat:16.4f} | {tau_dev:12.4f} | {theta_dev:11.4f}")

print("\n" + "=" * 90)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ ОПТИМАЛЬНЫХ ОЦЕНОК")
print("=" * 90)

print("\nДИСКРЕТНОЕ РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ:")
print("Объем | Ср. оценка τ | Ср. оценка θ | СКО τ | СКО θ")
print("-" * 70)

for size in sizes:
    size_results = [r for r in uniform_results if r['size'] == size]
    tau_estimates = [r['tau_hat'] for r in size_results]
    theta_estimates = [r['theta_hat'] for r in size_results]
    
    tau_mean = np.mean(tau_estimates)
    theta_mean = np.mean(theta_estimates)
    tau_std = np.std(tau_estimates)
    theta_std = np.std(theta_estimates)
    
    print(f"{size:5d} | {tau_mean:11.2f} | {theta_mean:12.2f} | {tau_std:5.2f} | {theta_std:5.2f}")

print("\nРАСПРЕДЕЛЕНИЕ ЭРЛАНГА:")
print("Объем | Ср. оценка τ | Ср. оценка θ | СКО τ | СКО θ")
print("-" * 70)

for size in sizes:
    size_results = [r for r in erlang_results if r['size'] == size]
    tau_estimates = [r['tau_hat'] for r in size_results]
    theta_estimates = [r['theta_hat'] for r in size_results]
    
    tau_mean = np.mean(tau_estimates)
    theta_mean = np.mean(theta_estimates)
    tau_std = np.std(tau_estimates)
    theta_std = np.std(theta_estimates)
    
    print(f"{size:5d} | {tau_mean:11.4f} | {theta_mean:12.4f} | {tau_std:5.4f} | {theta_std:5.4f}")