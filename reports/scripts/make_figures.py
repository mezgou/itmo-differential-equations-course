"""Генерация рисунков и таблиц для классических численных методов (вариант 13).

Скрипт решает задачу Коши y' = -y/x - y^2, y(1) = 2 на отрезке [1, 2] методами
Эйлера, Хойна и Рунге-Кутты 4-го порядка, проводит анализ порядка сходимости,
строит области абсолютной устойчивости и решает затухающий осциллятор варианта 13
методом Рунге-Кутты для систем. Все рисунки сохраняются в формате SVG, числовые
результаты — в CSV и в общий файл reports/data/summary.json.

Запуск: uv run python reports/scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

# Консоль Windows по умолчанию использует cp1251 и не кодирует кириллицу в выводе,
# поэтому принудительно включаем UTF-8 для корректной печати результатов.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
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

X_START: float = 1.0
Y_START: float = 2.0
X_END: float = 2.0
STEP_COUNT: int = 20
STEP_SIZE: float = (X_END - X_START) / STEP_COUNT

CONVERGENCE_STEP_COUNTS: list[int] = [10, 20, 40, 80, 160, 320, 640]
RUNGE_KUTTA_FIT_STEP_COUNT_MAX: int = 160

OSCILLATOR_OMEGA_0: float = 1.8
OSCILLATOR_DELTA: float = 0.25
OSCILLATOR_INITIAL_POSITION: float = 0.0
OSCILLATOR_INITIAL_VELOCITY: float = 1.5
OSCILLATOR_TIME_START: float = 0.0
OSCILLATOR_TIME_END: float = 12.0
OSCILLATOR_STEP_COUNT: int = 600

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
FIGURES_DIRECTORY: Path = PROJECT_ROOT / "reports" / "figures"
DATA_DIRECTORY: Path = PROJECT_ROOT / "reports" / "data"
SUMMARY_PATH: Path = DATA_DIRECTORY / "summary.json"


# ---------------------------------------------------------------------------
# Точное решение, правая часть, сетка
# ---------------------------------------------------------------------------

def exact_solution(x_value: float) -> float:
    return float(1.0 / (x_value * (np.log(x_value) + 0.5)))


def derivative_function(x_value: float, y_value: float) -> float:
    return -y_value / x_value - y_value ** 2


def build_grid(x_start: float, x_end: float, step_count: int) -> NDArray[np.float64]:
    return np.linspace(x_start, x_end, step_count + 1)


# ---------------------------------------------------------------------------
# Численные методы (скалярная задача)
# ---------------------------------------------------------------------------

def solve_euler(
    derivative: Callable[[float, float], float],
    x_values: NDArray[np.float64],
    y_start: float,
) -> NDArray[np.float64]:
    y_values: NDArray[np.float64] = np.zeros(len(x_values), dtype=np.float64)
    y_values[0] = y_start

    for index in range(len(x_values) - 1):
        step_size: float = x_values[index + 1] - x_values[index]
        y_values[index + 1] = y_values[index] + step_size * derivative(x_values[index], y_values[index])

    return y_values


def solve_heun(
    derivative: Callable[[float, float], float],
    x_values: NDArray[np.float64],
    y_start: float,
) -> NDArray[np.float64]:
    y_values: NDArray[np.float64] = np.zeros(len(x_values), dtype=np.float64)
    y_values[0] = y_start

    for index in range(len(x_values) - 1):
        current_x: float = x_values[index]
        next_x: float = x_values[index + 1]
        current_y: float = y_values[index]
        step_size: float = next_x - current_x

        predicted_y: float = current_y + step_size * derivative(current_x, current_y)
        current_slope: float = derivative(current_x, current_y)
        predicted_slope: float = derivative(next_x, predicted_y)

        y_values[index + 1] = current_y + step_size * (current_slope + predicted_slope) / 2

    return y_values


def solve_runge_kutta_4(
    derivative: Callable[[float, float], float],
    x_values: NDArray[np.float64],
    y_start: float,
) -> NDArray[np.float64]:
    y_values: NDArray[np.float64] = np.zeros(len(x_values), dtype=np.float64)
    y_values[0] = y_start

    for index in range(len(x_values) - 1):
        current_x: float = x_values[index]
        current_y: float = y_values[index]
        step_size: float = x_values[index + 1] - current_x

        slope_1: float = derivative(current_x, current_y)
        slope_2: float = derivative(
            current_x + step_size / 2,
            current_y + step_size * slope_1 / 2,
        )
        slope_3: float = derivative(
            current_x + step_size / 2,
            current_y + step_size * slope_2 / 2,
        )
        slope_4: float = derivative(
            current_x + step_size,
            current_y + step_size * slope_3,
        )

        y_values[index + 1] = current_y + step_size * (
            slope_1 + 2 * slope_2 + 2 * slope_3 + slope_4
        ) / 6

    return y_values


# ---------------------------------------------------------------------------
# Численный метод для систем (векторное состояние)
# ---------------------------------------------------------------------------

def solve_runge_kutta_4_system(
    system_function: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    x_values: NDArray[np.float64],
    initial_state: NDArray[np.float64],
) -> NDArray[np.float64]:
    state_values: NDArray[np.float64] = np.zeros(
        (len(x_values), len(initial_state)),
        dtype=np.float64,
    )
    state_values[0] = initial_state

    for index in range(len(x_values) - 1):
        current_x: float = x_values[index]
        current_state: NDArray[np.float64] = state_values[index]
        step_size: float = x_values[index + 1] - x_values[index]

        slope_1: NDArray[np.float64] = system_function(current_x, current_state)
        slope_2: NDArray[np.float64] = system_function(
            current_x + step_size / 2,
            current_state + step_size * slope_1 / 2,
        )
        slope_3: NDArray[np.float64] = system_function(
            current_x + step_size / 2,
            current_state + step_size * slope_2 / 2,
        )
        slope_4: NDArray[np.float64] = system_function(
            current_x + step_size,
            current_state + step_size * slope_3,
        )

        state_values[index + 1] = current_state + step_size * (
            slope_1 + 2 * slope_2 + 2 * slope_3 + slope_4
        ) / 6

    return state_values


# ---------------------------------------------------------------------------
# Метрики погрешности (узел x0 исключён, знаменатель — приближённые значения)
# ---------------------------------------------------------------------------

def calculate_absolute_error(
    exact_values: NDArray[np.float64],
    approximate_values: NDArray[np.float64],
) -> float:
    exact_values_without_initial_point: NDArray[np.float64] = exact_values[1:]
    approximate_values_without_initial_point: NDArray[np.float64] = approximate_values[1:]

    absolute_errors: NDArray[np.float64] = np.abs(
        exact_values_without_initial_point - approximate_values_without_initial_point
    )

    return float(np.max(absolute_errors))


def calculate_relative_error(
    exact_values: NDArray[np.float64],
    approximate_values: NDArray[np.float64],
) -> float:
    exact_values_without_initial_point: NDArray[np.float64] = exact_values[1:]
    approximate_values_without_initial_point: NDArray[np.float64] = approximate_values[1:]

    absolute_errors: NDArray[np.float64] = np.abs(
        exact_values_without_initial_point - approximate_values_without_initial_point
    )
    absolute_approximate_values: NDArray[np.float64] = np.abs(
        approximate_values_without_initial_point
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_errors: NDArray[np.float64] = np.where(
            absolute_approximate_values != 0.0,
            absolute_errors / absolute_approximate_values,
            np.nan,
        )

    return float(np.nanmax(relative_errors))


# ---------------------------------------------------------------------------
# Затухающий осциллятор варианта 13
# ---------------------------------------------------------------------------

def oscillator_exact_solution(t_value: float) -> float:
    omega: float = float(np.sqrt(OSCILLATOR_OMEGA_0 ** 2 - OSCILLATOR_DELTA ** 2))
    exponent_value: float = float(np.exp(-OSCILLATOR_DELTA * t_value))
    sine_value: float = float(np.sin(omega * t_value))

    return float(OSCILLATOR_INITIAL_VELOCITY / omega * exponent_value * sine_value)


def oscillator_system_derivative(
    t_value: float,
    state_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    position_value: np.float64 = state_values[0]
    velocity_value: np.float64 = state_values[1]
    acceleration_value: np.float64 = (
        -2.0 * OSCILLATOR_DELTA * velocity_value - OSCILLATOR_OMEGA_0 ** 2 * position_value
    )

    return np.array([velocity_value, acceleration_value], dtype=float)


# ---------------------------------------------------------------------------
# Функции устойчивости R(z) для z = h * lambda
# ---------------------------------------------------------------------------

def euler_stability_function(z_values: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 1.0 + z_values


def runge_kutta_2_stability_function(z_values: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 1.0 + z_values + z_values ** 2 / 2.0


def runge_kutta_4_stability_function(z_values: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return (
        1.0
        + z_values
        + z_values ** 2 / 2.0
        + z_values ** 3 / 6.0
        + z_values ** 4 / 24.0
    )


# ---------------------------------------------------------------------------
# Рисунок 1: сравнение точного и приближённых решений
# ---------------------------------------------------------------------------

def make_solution_comparison_figure(
    x_values: NDArray[np.float64],
    exact_values: NDArray[np.float64],
    euler_values: NDArray[np.float64],
    heun_values: NDArray[np.float64],
    runge_kutta_4_values: NDArray[np.float64],
) -> None:
    figure, axes = plt.subplots(figsize=(7, 4.3))

    axes.plot(x_values, exact_values, label="Точное решение")
    axes.plot(x_values, euler_values, marker="o", markersize=4, label="Метод Эйлера")
    axes.plot(x_values, heun_values, marker="s", markersize=4, label="Метод Хойна")
    axes.plot(x_values, runge_kutta_4_values, marker="^", markersize=4, label="Метод Рунге-Кутты 4-го порядка")

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend()

    figure.savefig(FIGURES_DIRECTORY / "fig_solution_comparison.svg", format="svg")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Рисунок 2: поточечная абсолютная погрешность
# ---------------------------------------------------------------------------

def make_pointwise_error_figure(
    x_values: NDArray[np.float64],
    exact_values: NDArray[np.float64],
    euler_values: NDArray[np.float64],
    heun_values: NDArray[np.float64],
    runge_kutta_4_values: NDArray[np.float64],
) -> None:
    figure, axes = plt.subplots(figsize=(7, 4.3))

    plotted_x_values: NDArray[np.float64] = x_values[1:]
    axes.plot(
        plotted_x_values,
        np.abs(exact_values[1:] - euler_values[1:]),
        marker="o",
        markersize=4,
        label="Метод Эйлера",
    )
    axes.plot(
        plotted_x_values,
        np.abs(exact_values[1:] - heun_values[1:]),
        marker="s",
        markersize=4,
        label="Метод Хойна",
    )
    axes.plot(
        plotted_x_values,
        np.abs(exact_values[1:] - runge_kutta_4_values[1:]),
        marker="^",
        markersize=4,
        label="Метод Рунге-Кутты 4-го порядка",
    )

    axes.set_yscale("log")
    axes.set_xlabel("x")
    axes.set_ylabel("Абсолютная погрешность")
    axes.legend()

    figure.savefig(FIGURES_DIRECTORY / "fig_pointwise_error.svg", format="svg")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Анализ порядка сходимости
# ---------------------------------------------------------------------------

def compute_convergence_data() -> dict[str, NDArray[np.float64]]:
    step_size_values: list[float] = []
    euler_error_values: list[float] = []
    heun_error_values: list[float] = []
    runge_kutta_4_error_values: list[float] = []

    for step_count in CONVERGENCE_STEP_COUNTS:
        x_values: NDArray[np.float64] = build_grid(X_START, X_END, step_count)
        exact_values: NDArray[np.float64] = np.array(
            [exact_solution(x_value) for x_value in x_values],
            dtype=np.float64,
        )

        euler_values: NDArray[np.float64] = solve_euler(derivative_function, x_values, Y_START)
        heun_values: NDArray[np.float64] = solve_heun(derivative_function, x_values, Y_START)
        runge_kutta_4_values: NDArray[np.float64] = solve_runge_kutta_4(derivative_function, x_values, Y_START)

        step_size_values.append((X_END - X_START) / step_count)
        euler_error_values.append(calculate_absolute_error(exact_values, euler_values))
        heun_error_values.append(calculate_absolute_error(exact_values, heun_values))
        runge_kutta_4_error_values.append(calculate_absolute_error(exact_values, runge_kutta_4_values))

    return {
        "step_size_values": np.array(step_size_values, dtype=np.float64),
        "euler_error_values": np.array(euler_error_values, dtype=np.float64),
        "heun_error_values": np.array(heun_error_values, dtype=np.float64),
        "runge_kutta_4_error_values": np.array(runge_kutta_4_error_values, dtype=np.float64),
    }


def estimate_observed_order(
    step_size_values: NDArray[np.float64],
    error_values: NDArray[np.float64],
    fit_mask: NDArray[np.bool_] | None = None,
) -> float:
    if fit_mask is None:
        fit_mask = np.ones(len(step_size_values), dtype=bool)

    log_step_sizes: NDArray[np.float64] = np.log(step_size_values[fit_mask])
    log_errors: NDArray[np.float64] = np.log(error_values[fit_mask])
    slope_value, _ = np.polyfit(log_step_sizes, log_errors, 1)

    return float(slope_value)


def make_convergence_order_figure(
    convergence_data: dict[str, NDArray[np.float64]],
    observed_orders: dict[str, float],
) -> None:
    figure, axes = plt.subplots(figsize=(7, 4.3))

    step_size_values: NDArray[np.float64] = convergence_data["step_size_values"]

    method_specifications: list[tuple[str, str, str, float]] = [
        ("euler_error_values", "Эйлер", "o", observed_orders["p_euler"]),
        ("heun_error_values", "Хойн", "s", observed_orders["p_heun"]),
        ("runge_kutta_4_error_values", "РК4", "^", observed_orders["p_runge_kutta_4"]),
    ]
    palette: list[str] = ["#1f4e79", "#c0504d", "#2e7d32"]

    for (error_key, method_label, marker_symbol, observed_order), color_value in zip(
        method_specifications, palette
    ):
        error_values: NDArray[np.float64] = convergence_data[error_key]
        axes.loglog(
            step_size_values,
            error_values,
            marker=marker_symbol,
            markersize=5,
            color=color_value,
            label=f"{method_label}: p̂≈{observed_order:.2f}",
        )

    # Опорные прямые наклонов 1, 2, 4, привязанные к данным каждого метода
    reference_specifications: list[tuple[int, NDArray[np.float64]]] = [
        (1, convergence_data["euler_error_values"]),
        (2, convergence_data["heun_error_values"]),
        (4, convergence_data["runge_kutta_4_error_values"]),
    ]
    for slope_order, error_values in reference_specifications:
        anchor_step_size: float = float(step_size_values[0])
        anchor_error: float = float(error_values[0])
        reference_errors: NDArray[np.float64] = (
            anchor_error * (step_size_values / anchor_step_size) ** slope_order
        )
        axes.loglog(
            step_size_values,
            reference_errors,
            linestyle="--",
            linewidth=0.8,
            color="0.45",
            label=f"наклон {slope_order}",
        )

    axes.set_xlabel("Шаг h")
    axes.set_ylabel("Макс. абсолютная погрешность")
    axes.legend(fontsize=8, ncol=2)

    figure.savefig(FIGURES_DIRECTORY / "fig_convergence_order.svg", format="svg")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Рисунок 4: области абсолютной устойчивости
# ---------------------------------------------------------------------------

def make_stability_regions_figure() -> None:
    real_axis_values: NDArray[np.float64] = np.linspace(-5.0, 2.0, 700)
    imaginary_axis_values: NDArray[np.float64] = np.linspace(-4.0, 4.0, 700)
    real_grid, imaginary_grid = np.meshgrid(real_axis_values, imaginary_axis_values)
    z_grid: NDArray[np.complex128] = real_grid + 1j * imaginary_grid

    method_specifications: list[tuple[str, Callable[[NDArray[np.complex128]], NDArray[np.complex128]], str]] = [
        ("Эйлер", euler_stability_function, "#1f4e79"),
        ("Хойн (РК2)", runge_kutta_2_stability_function, "#c0504d"),
        ("РК4", runge_kutta_4_stability_function, "#2e7d32"),
    ]

    figure, axes = plt.subplots(figsize=(7, 4.3))

    for method_label, stability_function, color_value in method_specifications:
        amplification_magnitude: NDArray[np.float64] = np.abs(stability_function(z_grid))
        axes.contourf(
            real_grid,
            imaginary_grid,
            amplification_magnitude,
            levels=[0.0, 1.0],
            colors=[color_value],
            alpha=0.18,
        )
        axes.contour(
            real_grid,
            imaginary_grid,
            amplification_magnitude,
            levels=[1.0],
            colors=[color_value],
            linewidths=1.6,
        )
        # Прокси-линия для легенды (заполненные контуры в легенду не попадают)
        axes.plot([], [], color=color_value, linewidth=1.6, label=f"{method_label}: |R(z)|=1")

    axes.axhline(0.0, color="0.3", linewidth=0.8)
    axes.axvline(0.0, color="0.3", linewidth=0.8)
    axes.set_aspect("equal")
    axes.set_xlabel("Re(z), z = hλ")
    axes.set_ylabel("Im(z)")
    axes.legend(fontsize=9, loc="upper left")

    figure.savefig(FIGURES_DIRECTORY / "fig_stability_regions.svg", format="svg")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Рисунок 5: осциллятор РК4 (система) vs точное решение
# ---------------------------------------------------------------------------

def make_oscillator_figure() -> tuple[float, float]:
    time_values: NDArray[np.float64] = build_grid(
        OSCILLATOR_TIME_START,
        OSCILLATOR_TIME_END,
        OSCILLATOR_STEP_COUNT,
    )
    initial_state: NDArray[np.float64] = np.array(
        [OSCILLATOR_INITIAL_POSITION, OSCILLATOR_INITIAL_VELOCITY],
        dtype=float,
    )

    state_values: NDArray[np.float64] = solve_runge_kutta_4_system(
        oscillator_system_derivative,
        time_values,
        initial_state,
    )
    runge_kutta_4_position_values: NDArray[np.float64] = state_values[:, 0]
    exact_position_values: NDArray[np.float64] = np.array(
        [oscillator_exact_solution(t_value) for t_value in time_values],
        dtype=np.float64,
    )

    omega: float = float(np.sqrt(OSCILLATOR_OMEGA_0 ** 2 - OSCILLATOR_DELTA ** 2))
    maximum_absolute_error: float = float(
        np.max(np.abs(exact_position_values - runge_kutta_4_position_values))
    )

    figure, axes = plt.subplots(figsize=(7, 4.3))

    axes.plot(time_values, exact_position_values, label="Точное решение")
    axes.plot(
        time_values,
        runge_kutta_4_position_values,
        linestyle="--",
        label="РК4 (система)",
    )

    axes.set_xlabel("t")
    axes.set_ylabel("x")
    axes.legend()

    figure.savefig(FIGURES_DIRECTORY / "fig_oscillator_rk4_vs_exact.svg", format="svg")
    plt.close(figure)

    return omega, maximum_absolute_error


# ---------------------------------------------------------------------------
# Главная процедура
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)

    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # --- Базовая сетка n = 20 ---
    x_values: NDArray[np.float64] = build_grid(X_START, X_END, STEP_COUNT)
    exact_values: NDArray[np.float64] = np.array(
        [exact_solution(x_value) for x_value in x_values],
        dtype=np.float64,
    )
    euler_values: NDArray[np.float64] = solve_euler(derivative_function, x_values, Y_START)
    heun_values: NDArray[np.float64] = solve_heun(derivative_function, x_values, Y_START)
    runge_kutta_4_values: NDArray[np.float64] = solve_runge_kutta_4(derivative_function, x_values, Y_START)

    euler_absolute_error: float = calculate_absolute_error(exact_values, euler_values)
    euler_relative_error: float = calculate_relative_error(exact_values, euler_values)
    heun_absolute_error: float = calculate_absolute_error(exact_values, heun_values)
    heun_relative_error: float = calculate_relative_error(exact_values, heun_values)
    runge_kutta_4_absolute_error: float = calculate_absolute_error(exact_values, runge_kutta_4_values)
    runge_kutta_4_relative_error: float = calculate_relative_error(exact_values, runge_kutta_4_values)

    # --- Рисунки 1 и 2 ---
    make_solution_comparison_figure(x_values, exact_values, euler_values, heun_values, runge_kutta_4_values)
    make_pointwise_error_figure(x_values, exact_values, euler_values, heun_values, runge_kutta_4_values)

    # --- Анализ сходимости (Рисунок 3) ---
    convergence_data: dict[str, NDArray[np.float64]] = compute_convergence_data()
    step_count_array: NDArray[np.int64] = np.array(CONVERGENCE_STEP_COUNTS, dtype=np.int64)
    runge_kutta_4_fit_mask: NDArray[np.bool_] = step_count_array <= RUNGE_KUTTA_FIT_STEP_COUNT_MAX

    observed_orders: dict[str, float] = {
        "p_euler": estimate_observed_order(
            convergence_data["step_size_values"],
            convergence_data["euler_error_values"],
        ),
        "p_heun": estimate_observed_order(
            convergence_data["step_size_values"],
            convergence_data["heun_error_values"],
        ),
        "p_runge_kutta_4": estimate_observed_order(
            convergence_data["step_size_values"],
            convergence_data["runge_kutta_4_error_values"],
            fit_mask=runge_kutta_4_fit_mask,
        ),
    }

    make_convergence_order_figure(convergence_data, observed_orders)

    # --- Рисунок 4 (устойчивость) ---
    make_stability_regions_figure()

    # --- Рисунок 5 (осциллятор) ---
    oscillator_omega, oscillator_maximum_absolute_error = make_oscillator_figure()

    # --- Таблица solutions.csv (21 узел, n = 20) ---
    solutions_frame: pd.DataFrame = pd.DataFrame({
        "x": x_values,
        "exact": exact_values,
        "euler": euler_values,
        "heun": heun_values,
        "rk4": runge_kutta_4_values,
    })
    solutions_frame.to_csv(DATA_DIRECTORY / "solutions.csv", index=False)

    # --- Таблица errors.csv ---
    errors_frame: pd.DataFrame = pd.DataFrame(
        [
            ["Euler", euler_absolute_error, euler_relative_error],
            ["Heun", heun_absolute_error, heun_relative_error],
            ["RK4", runge_kutta_4_absolute_error, runge_kutta_4_relative_error],
        ],
        columns=["method", "abs_error", "rel_error"],
    )
    errors_frame.to_csv(DATA_DIRECTORY / "errors.csv", index=False)

    # --- Таблица convergence.csv ---
    convergence_frame: pd.DataFrame = pd.DataFrame({
        "n": CONVERGENCE_STEP_COUNTS,
        "h": convergence_data["step_size_values"],
        "err_euler": convergence_data["euler_error_values"],
        "err_heun": convergence_data["heun_error_values"],
        "err_rk4": convergence_data["runge_kutta_4_error_values"],
    })
    convergence_frame.to_csv(DATA_DIRECTORY / "convergence.csv", index=False)

    # --- Таблица orders.csv ---
    orders_frame: pd.DataFrame = pd.DataFrame(
        [
            ["Euler", 1, observed_orders["p_euler"]],
            ["Heun", 2, observed_orders["p_heun"]],
            ["RK4", 4, observed_orders["p_runge_kutta_4"]],
        ],
        columns=["method", "p_theoretical", "p_observed"],
    )
    orders_frame.to_csv(DATA_DIRECTORY / "orders.csv", index=False)

    # --- summary.json ---
    summary: dict = {}
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as summary_file:
            summary = json.load(summary_file)

    summary["ode_n20"] = {
        "euler": {"abs": euler_absolute_error, "rel": euler_relative_error},
        "heun": {"abs": heun_absolute_error, "rel": heun_relative_error},
        "rk4": {"abs": runge_kutta_4_absolute_error, "rel": runge_kutta_4_relative_error},
        "phi_at_2": exact_solution(2.0),
    }
    summary["convergence"] = {
        "n_list": list(CONVERGENCE_STEP_COUNTS),
        "h_list": convergence_data["step_size_values"].tolist(),
        "err_euler": convergence_data["euler_error_values"].tolist(),
        "err_heun": convergence_data["heun_error_values"].tolist(),
        "err_rk4": convergence_data["runge_kutta_4_error_values"].tolist(),
        "p_euler": observed_orders["p_euler"],
        "p_heun": observed_orders["p_heun"],
        "p_rk4": observed_orders["p_runge_kutta_4"],
        "rk4_fit_n_max": RUNGE_KUTTA_FIT_STEP_COUNT_MAX,
    }
    summary["oscillator"] = {
        "omega": oscillator_omega,
        "rk4_vs_exact_max_abs": oscillator_maximum_absolute_error,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    # --- Печать результатов ---
    print("=== Классические методы (вариант 13) ===")
    print(f"n = {STEP_COUNT}, h = {STEP_SIZE}")
    print(f"phi(2) = {exact_solution(2.0):.6f}")
    print()
    print("Погрешности при n = 20:")
    print(f"  Эйлер: abs = {euler_absolute_error:.6e}, rel = {euler_relative_error:.6e}")
    print(f"  Хойн:  abs = {heun_absolute_error:.6e}, rel = {heun_relative_error:.6e}")
    print(f"  РК4:   abs = {runge_kutta_4_absolute_error:.6e}, rel = {runge_kutta_4_relative_error:.6e}")
    print()
    print("Наблюдаемые порядки сходимости:")
    print(f"  Эйлер: p = {observed_orders['p_euler']:.4f}")
    print(f"  Хойн:  p = {observed_orders['p_heun']:.4f}")
    print(f"  РК4:   p = {observed_orders['p_runge_kutta_4']:.4f} (фит по n <= {RUNGE_KUTTA_FIT_STEP_COUNT_MAX})")
    print()
    print("Осциллятор:")
    print(f"  omega = {oscillator_omega:.7f}")
    print(f"  РК4 vs точное, макс. абс. = {oscillator_maximum_absolute_error:.6e}")
    print()
    print("Рисунки и таблицы сохранены.")


if __name__ == "__main__":
    main()
