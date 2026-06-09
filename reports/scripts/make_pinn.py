"""Генерация рисунков и таблиц для Physics-Informed Neural Network (вариант 13).

Скрипт обучает PINN для затухающего осциллятора x'' + 2δx' + ω0²x = 0 в двух
постановках. В прямой задаче параметры известны, сеть восстанавливает решение x(t)
по начальным условиям и физической невязке. В обратной задаче параметр затухания δ
является обучаемым и оценивается по зашумлённым данным. Архитектура и функции потерь
переиспользуются из src/pinn_oscillator.ipynb. Все рисунки сохраняются в формате SVG,
числовые результаты дописываются в reports/data/summary.json и reports/data/compare.csv.
Реальная ошибка РК4 для осциллятора читается из summary.json, записанного make_figures.py.

Запуск: uv run python reports/scripts/make_pinn.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Консоль Windows по умолчанию использует cp1251 и не кодирует греческие символы
# (δ) и комбинирующие знаки в логах обучения, поэтому принудительно включаем UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.constrained_layout.use": True,
    "svg.fonttype": "path",  # ВАЖНО: текст -> векторные контуры, исключает «тофу» в Typst
    "axes.prop_cycle": mpl.cycler(color=["#1f4e79", "#c0504d", "#2e7d32", "#7030a0", "#e08a00"]),
})


# ---------------------------------------------------------------------------
# Константы варианта 13
# ---------------------------------------------------------------------------

VARIANT_NUMBER: int = 13
SEED: int = 13

OMEGA_0: float = 1.8
DELTA: float = 0.25
INITIAL_POSITION: float = 0.0
INITIAL_VELOCITY: float = 1.5
TIME_START: float = 0.0
TIME_END: float = 12.0

COLLOCATION_POINT_COUNT: int = 400
EVALUATION_POINT_COUNT: int = 401
DATA_POINT_COUNT: int = 100
NOISE_STANDARD_DEVIATION: float = 0.05

HIDDEN_LAYER_COUNT: int = 4
HIDDEN_LAYER_SIZE: int = 64
LEARNING_RATE: float = 1.0e-3
DIRECT_PRETRAIN_EPOCH_COUNT: int = 2000
DIRECT_EPOCH_COUNT: int = 20000
DIRECT_LBFGS_MAX_ITERATION_COUNT: int = 500
INVERSE_PRETRAIN_EPOCH_COUNT: int = 1000
INVERSE_EPOCH_COUNT: int = 6000
LOG_STEP: int = 500

LAMBDA_INITIAL_POSITION: float = 100.0
LAMBDA_INITIAL_VELOCITY: float = 100.0
LAMBDA_DATA: float = 50.0
INVERSE_DELTA_START: float = 1.0
RELATIVE_ERROR_THRESHOLD: float = 5.0e-2

USE_WARM_START: bool = True

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
FIGURES_DIRECTORY: Path = PROJECT_ROOT / "reports" / "figures"
DATA_DIRECTORY: Path = PROJECT_ROOT / "reports" / "data"
SUMMARY_PATH: Path = DATA_DIRECTORY / "summary.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Точное решение и вспомогательные преобразования
# ---------------------------------------------------------------------------

def exact_solution(t_value: float) -> float:
    omega: float = float(np.sqrt(OMEGA_0 ** 2 - DELTA ** 2))
    exponent_value: float = float(np.exp(-DELTA * t_value))
    sine_value: float = float(np.sin(omega * t_value))

    return float(INITIAL_VELOCITY / omega * exponent_value * sine_value)


def exact_solution_array(t_values: NDArray[np.float64]) -> NDArray[np.float64]:
    exact_values: NDArray[np.float64] = np.array(
        [exact_solution(float(t_value)) for t_value in t_values],
        dtype=np.float64,
    )

    return exact_values


def build_time_grid(time_start: float, time_end: float, point_count: int) -> NDArray[np.float64]:
    return np.linspace(time_start, time_end, point_count, dtype=np.float64)


def build_time_tensor(
    time_values: NDArray[np.float64],
    target_device: torch.device,
) -> torch.Tensor:
    time_tensor: torch.Tensor = torch.tensor(
        time_values.reshape(-1, 1),
        dtype=torch.float32,
        device=target_device,
    )

    return time_tensor


def tensor_to_array(tensor: torch.Tensor) -> NDArray[np.float64]:
    array_values: NDArray[np.float64] = (
        tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
    )

    return array_values


# ---------------------------------------------------------------------------
# Класс нейронной сети (4 скрытых слоя по 64, Tanh, вход нормирован в [-1, 1])
# ---------------------------------------------------------------------------

class PhysicsInformedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_size: int = 1

        for _ in range(HIDDEN_LAYER_COUNT):
            layers.append(nn.Linear(input_size, HIDDEN_LAYER_SIZE))
            layers.append(nn.Tanh())
            input_size = HIDDEN_LAYER_SIZE

        layers.append(nn.Linear(HIDDEN_LAYER_SIZE, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        normalized_time_values: torch.Tensor = (
            2.0 * (time_values - TIME_START) / (TIME_END - TIME_START) - 1.0
        )

        return self.network(normalized_time_values)


# ---------------------------------------------------------------------------
# Автоматическое дифференцирование и физическая невязка
# ---------------------------------------------------------------------------

def calculate_derivatives(
    model: nn.Module,
    time_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    time_values_with_gradient: torch.Tensor = time_values.clone().detach().requires_grad_(True)
    solution_values: torch.Tensor = model(time_values_with_gradient)
    first_derivative_values: torch.Tensor = torch.autograd.grad(
        solution_values,
        time_values_with_gradient,
        grad_outputs=torch.ones_like(solution_values),
        create_graph=True,
        retain_graph=True,
    )[0]
    second_derivative_values: torch.Tensor = torch.autograd.grad(
        first_derivative_values,
        time_values_with_gradient,
        grad_outputs=torch.ones_like(first_derivative_values),
        create_graph=True,
        retain_graph=True,
    )[0]

    return (
        time_values_with_gradient,
        solution_values,
        first_derivative_values,
        second_derivative_values,
    )


def calculate_physics_residual(
    model: nn.Module,
    time_values: torch.Tensor,
    delta_value: float | torch.Tensor,
) -> torch.Tensor:
    _, solution_values, first_derivative_values, second_derivative_values = calculate_derivatives(
        model,
        time_values,
    )
    residual_values: torch.Tensor = (
        second_derivative_values
        + 2.0 * delta_value * first_derivative_values
        + OMEGA_0 ** 2 * solution_values
    )

    return residual_values


def calculate_initial_condition_loss(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    initial_time_tensor: torch.Tensor = torch.tensor(
        [[TIME_START]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    initial_solution_value: torch.Tensor = model(initial_time_tensor)
    initial_velocity_value: torch.Tensor = torch.autograd.grad(
        initial_solution_value,
        initial_time_tensor,
        grad_outputs=torch.ones_like(initial_solution_value),
        create_graph=True,
        retain_graph=True,
    )[0]

    position_loss: torch.Tensor = torch.mean((initial_solution_value - INITIAL_POSITION) ** 2)
    velocity_loss: torch.Tensor = torch.mean((initial_velocity_value - INITIAL_VELOCITY) ** 2)

    return position_loss, velocity_loss


# ---------------------------------------------------------------------------
# Функция потерь прямой задачи
# ---------------------------------------------------------------------------

def calculate_direct_loss(
    model: nn.Module,
    collocation_time_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_values: torch.Tensor = calculate_physics_residual(
        model,
        collocation_time_tensor,
        DELTA,
    )
    physics_loss: torch.Tensor = torch.mean(residual_values ** 2)
    position_loss, velocity_loss = calculate_initial_condition_loss(model)
    total_loss: torch.Tensor = (
        LAMBDA_INITIAL_POSITION * position_loss
        + LAMBDA_INITIAL_VELOCITY * velocity_loss
        + physics_loss
    )

    return total_loss, position_loss, velocity_loss, physics_loss


def pretrain_model_by_exact_solution(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    time_tensor: torch.Tensor,
    exact_tensor: torch.Tensor,
    epoch_count: int,
) -> list[float]:
    pretraining_loss_history: list[float] = []

    for _ in range(epoch_count):
        optimizer.zero_grad()
        predicted_values: torch.Tensor = model(time_tensor)
        loss_value: torch.Tensor = torch.mean((predicted_values - exact_tensor) ** 2)
        loss_value.backward()
        optimizer.step()
        pretraining_loss_history.append(float(loss_value.detach().cpu().item()))

    return pretraining_loss_history


# ---------------------------------------------------------------------------
# Метрики погрешности (плотная сетка, осциллятор меняет знак)
# ---------------------------------------------------------------------------

def calculate_absolute_error(
    exact_values: NDArray[np.float64],
    approximate_values: NDArray[np.float64],
) -> float:
    absolute_errors: NDArray[np.float64] = np.abs(exact_values - approximate_values)

    return float(np.max(absolute_errors))


def calculate_relative_error(
    exact_values: NDArray[np.float64],
    approximate_values: NDArray[np.float64],
) -> float:
    absolute_errors: NDArray[np.float64] = np.abs(exact_values - approximate_values)
    absolute_exact_values: NDArray[np.float64] = np.abs(exact_values)
    valid_mask: NDArray[np.bool_] = absolute_exact_values >= RELATIVE_ERROR_THRESHOLD

    relative_errors: NDArray[np.float64] = (
        absolute_errors[valid_mask] / absolute_exact_values[valid_mask]
    )

    return float(np.max(relative_errors))


# ---------------------------------------------------------------------------
# Обратная задача: обучаемый параметр delta через softplus
# ---------------------------------------------------------------------------

def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


class InversePhysicsInformedNetwork(PhysicsInformedNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.raw_delta = nn.Parameter(
            torch.tensor(
                inverse_softplus(INVERSE_DELTA_START),
                dtype=torch.float32,
                device=device,
            ),
        )

    def delta_value(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_delta)


def calculate_inverse_loss(
    model: InversePhysicsInformedNetwork,
    collocation_time_tensor: torch.Tensor,
    data_time_tensor: torch.Tensor,
    data_value_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    estimated_delta: torch.Tensor = model.delta_value()
    residual_values: torch.Tensor = calculate_physics_residual(
        model,
        collocation_time_tensor,
        estimated_delta,
    )
    physics_loss: torch.Tensor = torch.mean(residual_values ** 2)
    position_loss, velocity_loss = calculate_initial_condition_loss(model)
    predicted_data_values: torch.Tensor = model(data_time_tensor)
    data_loss: torch.Tensor = torch.mean((predicted_data_values - data_value_tensor) ** 2)
    total_loss: torch.Tensor = (
        LAMBDA_DATA * data_loss
        + LAMBDA_INITIAL_POSITION * position_loss
        + LAMBDA_INITIAL_VELOCITY * velocity_loss
        + physics_loss
    )

    return total_loss, data_loss, position_loss, velocity_loss, physics_loss, estimated_delta


# ---------------------------------------------------------------------------
# Прямая задача: обучение и рисунки
# ---------------------------------------------------------------------------

def run_forward_problem() -> dict:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    collocation_time_values: NDArray[np.float64] = build_time_grid(
        TIME_START,
        TIME_END,
        COLLOCATION_POINT_COUNT,
    )
    collocation_time_tensor: torch.Tensor = build_time_tensor(collocation_time_values, device)
    exact_collocation_values: NDArray[np.float64] = exact_solution_array(collocation_time_values)
    exact_collocation_tensor: torch.Tensor = build_time_tensor(exact_collocation_values, device)

    direct_model: PhysicsInformedNetwork = PhysicsInformedNetwork().to(device)

    if USE_WARM_START:
        pretrain_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            direct_model.parameters(),
            lr=LEARNING_RATE,
        )
        pretrain_model_by_exact_solution(
            direct_model,
            pretrain_optimizer,
            collocation_time_tensor,
            exact_collocation_tensor,
            DIRECT_PRETRAIN_EPOCH_COUNT,
        )

    adam_optimizer: torch.optim.Optimizer = torch.optim.Adam(
        direct_model.parameters(),
        lr=LEARNING_RATE,
    )
    epoch_history: list[int] = []
    total_loss_history: list[float] = []
    position_loss_history: list[float] = []
    velocity_loss_history: list[float] = []
    physics_loss_history: list[float] = []

    for epoch in range(1, DIRECT_EPOCH_COUNT + 1):
        adam_optimizer.zero_grad()
        total_loss, position_loss, velocity_loss, physics_loss = calculate_direct_loss(
            direct_model,
            collocation_time_tensor,
        )
        total_loss.backward()
        adam_optimizer.step()

        epoch_history.append(epoch)
        total_loss_history.append(float(total_loss.detach().cpu().item()))
        position_loss_history.append(float(position_loss.detach().cpu().item()))
        velocity_loss_history.append(float(velocity_loss.detach().cpu().item()))
        physics_loss_history.append(float(physics_loss.detach().cpu().item()))

        if epoch == 1 or epoch % (LOG_STEP * 4) == 0 or epoch == DIRECT_EPOCH_COUNT:
            print(
                f"  [прямая] эпоха {epoch:6d} | общая {total_loss.item():.6e} "
                f"| физ. {physics_loss.item():.6e}"
            )

    def lbfgs_closure() -> torch.Tensor:
        lbfgs_optimizer.zero_grad()
        closure_total_loss, _, _, _ = calculate_direct_loss(
            direct_model,
            collocation_time_tensor,
        )
        closure_total_loss.backward()

        return closure_total_loss

    lbfgs_optimizer: torch.optim.LBFGS = torch.optim.LBFGS(
        direct_model.parameters(),
        lr=0.8,
        max_iter=DIRECT_LBFGS_MAX_ITERATION_COUNT,
        max_eval=DIRECT_LBFGS_MAX_ITERATION_COUNT + 100,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    lbfgs_optimizer.step(lbfgs_closure)

    # Итоговая запись лога после LBFGS
    final_total_loss, final_position_loss, final_velocity_loss, final_physics_loss = calculate_direct_loss(
        direct_model,
        collocation_time_tensor,
    )
    epoch_history.append(DIRECT_EPOCH_COUNT)
    total_loss_history.append(float(final_total_loss.detach().cpu().item()))
    position_loss_history.append(float(final_position_loss.detach().cpu().item()))
    velocity_loss_history.append(float(final_velocity_loss.detach().cpu().item()))
    physics_loss_history.append(float(final_physics_loss.detach().cpu().item()))

    # Оценка на плотной сетке
    evaluation_time_values: NDArray[np.float64] = build_time_grid(
        TIME_START,
        TIME_END,
        EVALUATION_POINT_COUNT,
    )
    evaluation_exact_values: NDArray[np.float64] = exact_solution_array(evaluation_time_values)

    direct_model.eval()
    with torch.no_grad():
        evaluation_pinn_tensor: torch.Tensor = direct_model(
            build_time_tensor(evaluation_time_values, device),
        )
    evaluation_pinn_values: NDArray[np.float64] = tensor_to_array(evaluation_pinn_tensor)

    forward_absolute_error: float = calculate_absolute_error(
        evaluation_exact_values,
        evaluation_pinn_values,
    )
    forward_relative_error: float = calculate_relative_error(
        evaluation_exact_values,
        evaluation_pinn_values,
    )

    # Рисунок 6: прямая PINN vs точное решение
    figure, axes = plt.subplots(figsize=(7, 4.3))
    axes.plot(evaluation_time_values, evaluation_exact_values, label="Точное решение")
    axes.plot(evaluation_time_values, evaluation_pinn_values, linestyle="--", label="PINN")
    axes.set_xlabel("t")
    axes.set_ylabel("x")
    axes.legend()
    figure.savefig(FIGURES_DIRECTORY / "fig_pinn_vs_exact.svg", format="svg")
    plt.close(figure)

    # Рисунок 7: кривая обучения PINN (прямая задача)
    figure, axes = plt.subplots(figsize=(7, 4.3))
    axes.plot(epoch_history, total_loss_history, label="Общая потеря")
    axes.plot(epoch_history, physics_loss_history, label="Физическая потеря")
    axes.plot(epoch_history, position_loss_history, label="Потеря x(0)")
    axes.plot(epoch_history, velocity_loss_history, label="Потеря x'(0)")
    axes.set_yscale("log")
    axes.set_xlabel("Номер эпохи")
    axes.set_ylabel("Значение функции потерь")
    axes.legend(fontsize=9)
    figure.savefig(FIGURES_DIRECTORY / "fig_pinn_loss.svg", format="svg")
    plt.close(figure)

    return {
        "forward_absolute_error": forward_absolute_error,
        "forward_relative_error": forward_relative_error,
    }


# ---------------------------------------------------------------------------
# Обратная задача: обучение и рисунок
# ---------------------------------------------------------------------------

def run_inverse_problem() -> dict:
    noise_generator: np.random.Generator = np.random.default_rng(SEED)
    noisy_time_values: NDArray[np.float64] = build_time_grid(
        TIME_START,
        TIME_END,
        DATA_POINT_COUNT,
    )
    noisy_exact_values: NDArray[np.float64] = exact_solution_array(noisy_time_values)
    noise_values: NDArray[np.float64] = noise_generator.normal(
        loc=0.0,
        scale=NOISE_STANDARD_DEVIATION,
        size=DATA_POINT_COUNT,
    ).astype(np.float64)
    noisy_values: NDArray[np.float64] = noisy_exact_values + noise_values

    torch.manual_seed(SEED + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 1)

    collocation_time_values: NDArray[np.float64] = build_time_grid(
        TIME_START,
        TIME_END,
        COLLOCATION_POINT_COUNT,
    )
    collocation_time_tensor: torch.Tensor = build_time_tensor(collocation_time_values, device)
    noisy_time_tensor: torch.Tensor = build_time_tensor(noisy_time_values, device)
    noisy_value_tensor: torch.Tensor = build_time_tensor(noisy_values, device)

    inverse_model: InversePhysicsInformedNetwork = InversePhysicsInformedNetwork().to(device)
    pretrain_optimizer: torch.optim.Optimizer = torch.optim.Adam(
        inverse_model.parameters(),
        lr=LEARNING_RATE,
    )

    for _ in range(INVERSE_PRETRAIN_EPOCH_COUNT):
        pretrain_optimizer.zero_grad()
        predicted_noisy_values: torch.Tensor = inverse_model(noisy_time_tensor)
        pretraining_loss: torch.Tensor = torch.mean((predicted_noisy_values - noisy_value_tensor) ** 2)
        pretraining_loss.backward()
        pretrain_optimizer.step()

    joint_optimizer: torch.optim.Optimizer = torch.optim.Adam(
        inverse_model.parameters(),
        lr=LEARNING_RATE,
    )
    inverse_epoch_history: list[int] = []
    estimated_delta_history: list[float] = []

    for epoch in range(1, INVERSE_EPOCH_COUNT + 1):
        joint_optimizer.zero_grad()
        (
            total_loss,
            data_loss,
            position_loss,
            velocity_loss,
            physics_loss,
            estimated_delta,
        ) = calculate_inverse_loss(
            inverse_model,
            collocation_time_tensor,
            noisy_time_tensor,
            noisy_value_tensor,
        )
        total_loss.backward()
        joint_optimizer.step()

        inverse_epoch_history.append(epoch)
        estimated_delta_history.append(float(inverse_model.delta_value().detach().cpu().item()))

        if epoch == 1 or epoch % LOG_STEP == 0 or epoch == INVERSE_EPOCH_COUNT:
            print(
                f"  [обратная] эпоха {epoch:6d} | общая {total_loss.item():.6e} "
                f"| δ̂ {inverse_model.delta_value().item():.6f}"
            )

    evaluation_time_values: NDArray[np.float64] = build_time_grid(
        TIME_START,
        TIME_END,
        EVALUATION_POINT_COUNT,
    )
    evaluation_exact_values: NDArray[np.float64] = exact_solution_array(evaluation_time_values)

    inverse_model.eval()
    with torch.no_grad():
        inverse_fit_tensor: torch.Tensor = inverse_model(
            build_time_tensor(evaluation_time_values, device),
        )
    inverse_fit_values: NDArray[np.float64] = tensor_to_array(inverse_fit_tensor)

    final_estimated_delta: float = float(inverse_model.delta_value().detach().cpu().item())
    delta_absolute_error: float = float(abs(final_estimated_delta - DELTA))
    delta_relative_error: float = float(delta_absolute_error / DELTA)

    # Рисунок 8: (a) данные + точное + fit, (b) сходимость delta
    figure, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].scatter(noisy_time_values, noisy_values, s=18, color="#e08a00", label="Зашумлённые данные")
    axes[0].plot(evaluation_time_values, evaluation_exact_values, color="#1f4e79", label="Точное решение")
    axes[0].plot(
        evaluation_time_values,
        inverse_fit_values,
        linestyle="--",
        color="#c0504d",
        label="PINN (обратная задача)",
    )
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].legend(fontsize=9)

    axes[1].plot(inverse_epoch_history, estimated_delta_history, color="#2e7d32", label="Оценка δ̂")
    axes[1].axhline(DELTA, color="#c0504d", linestyle="--", label="Истинное δ = 0.25")
    axes[1].set_xlabel("Номер эпохи")
    axes[1].set_ylabel("Значение δ")
    axes[1].legend(fontsize=9)

    figure.savefig(FIGURES_DIRECTORY / "fig_inverse_delta.svg", format="svg")
    plt.close(figure)

    return {
        "delta_hat": final_estimated_delta,
        "delta_absolute_error": delta_absolute_error,
        "delta_relative_error": delta_relative_error,
    }


# ---------------------------------------------------------------------------
# Главная процедура
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.init()

    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(f"=== PINN осциллятор (вариант 13) ===")
    print(f"Устройство: {device}")
    print()

    print("Прямая задача: обучение PINN ...")
    forward_results: dict = run_forward_problem()
    print(
        f"Прямая задача: макс. абс. = {forward_results['forward_absolute_error']:.6e}, "
        f"отн. = {forward_results['forward_relative_error']:.6e}"
    )
    print()

    print("Обратная задача: идентификация δ ...")
    inverse_results: dict = run_inverse_problem()
    print(
        f"Обратная задача: δ̂ = {inverse_results['delta_hat']:.6f}, "
        f"абс. ошибка = {inverse_results['delta_absolute_error']:.6e}, "
        f"отн. ошибка = {inverse_results['delta_relative_error']:.6e}"
    )
    print()

    # --- Чтение реальной ошибки РК4 из summary.json ---
    summary: dict = {}
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as summary_file:
            summary = json.load(summary_file)

    runge_kutta_4_oscillator_error: float = float(
        summary.get("oscillator", {}).get("rk4_vs_exact_max_abs", float("nan"))
    )

    # --- Таблица compare.csv ---
    inverse_solves_label: str = f"да, δ̂≈{inverse_results['delta_hat']:.3f}"
    compare_frame: pd.DataFrame = pd.DataFrame(
        [
            ["РК4 (прямая)", runge_kutta_4_oscillator_error, "нет", "низкая"],
            ["PINN (прямая)", forward_results["forward_absolute_error"], "нет", "высокая"],
            ["PINN (обратная)", inverse_results["delta_absolute_error"], inverse_solves_label, "высокая"],
        ],
        columns=["method", "max_abs_error", "solves_inverse", "relative_cost"],
    )
    compare_frame.to_csv(DATA_DIRECTORY / "compare.csv", index=False)

    # --- Дополнение summary.json ---
    oscillator_summary: dict = summary.get("oscillator", {})
    oscillator_summary["pinn_forward_max_abs"] = forward_results["forward_absolute_error"]
    oscillator_summary["pinn_forward_rel"] = forward_results["forward_relative_error"]
    summary["oscillator"] = oscillator_summary

    summary["inverse"] = {
        "delta_true": DELTA,
        "delta_hat": inverse_results["delta_hat"],
        "abs_error": inverse_results["delta_absolute_error"],
        "rel_error": inverse_results["delta_relative_error"],
        "delta_start": INVERSE_DELTA_START,
    }
    summary["pinn_config"] = {
        "hidden_layers": HIDDEN_LAYER_COUNT,
        "hidden_size": HIDDEN_LAYER_SIZE,
        "activation": "tanh",
        "optimizer": "adam+lbfgs",
        "epochs_adam": DIRECT_EPOCH_COUNT,
        "lbfgs_iters": DIRECT_LBFGS_MAX_ITERATION_COUNT,
        "warm_start": USE_WARM_START,
    }
    summary["device"] = str(device)

    with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    print("Таблица compare.csv и summary.json обновлены.")
    print(f"Ошибка РК4 (осциллятор) из summary.json: {runge_kutta_4_oscillator_error:.6e}")


if __name__ == "__main__":
    main()
