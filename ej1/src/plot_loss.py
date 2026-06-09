"""
plot_loss.py

Este script permite visualizar la evolución de la loss durante el entrenamiento
del autoencoder.

Lee el archivo:
    outputs/training_log.csv

y genera una figura:
    outputs/training_loss.png
"""

import csv
import os
import argparse
import matplotlib.pyplot as plt


def moving_average(values, window_size=100):
    """
    Calcula una media móvil para suavizar la curva de pérdida.

    Parámetros:
    - values: lista con los valores de loss
    - window_size: tamaño de la ventana para suavizar

    Retorna:
    - lista suavizada de valores
    """
    if window_size <= 1:
        return values

    smoothed = []

    for i in range(len(values)):
        # Tomamos los últimos 'window_size' valores disponibles
        start = max(0, i - window_size + 1)
        window = values[start:i + 1]

        # Calculamos el promedio de la ventana
        smoothed.append(sum(window) / len(window))

    return smoothed


def load_training_log(csv_path):
    """
    Carga el archivo CSV generado durante el entrenamiento.

    El archivo debe tener dos columnas:
    - Epoch
    - Loss

    Retorna:
    - epochs: lista de épocas
    - losses: lista de valores de pérdida
    """
    epochs = []
    losses = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epochs.append(int(row["Epoch"]))
            losses.append(float(row["Loss"]))

    return epochs, losses


def plot_loss(csv_path, output_path, smooth_window=100, log_scale=False):
    """
    Grafica la curva de loss del entrenamiento.

    Parámetros:
    - csv_path: ruta del archivo training_log.csv
    - output_path: ruta donde se guarda la figura
    - smooth_window: tamaño de la ventana para suavizar la curva
    - log_scale: si True, usa escala logarítmica en el eje Y
    """
    # Cargar datos del entrenamiento
    epochs, losses = load_training_log(csv_path)

    # Calcular la curva suavizada
    smoothed_losses = moving_average(losses, window_size=smooth_window)

    # Crear la figura
    plt.figure(figsize=(10, 6))

    # Curva original de la loss
    plt.plot(
        epochs,
        losses,
        alpha=0.35,
        label="Loss original"
    )

    # Curva suavizada para ver mejor la tendencia
    plt.plot(
        epochs,
        smoothed_losses,
        linewidth=2,
        label=f"Loss suavizada, ventana={smooth_window}"
    )

    # Opcional: escala logarítmica si la loss baja mucho
    if log_scale:
        plt.yscale("log")

    # Títulos y etiquetas
    plt.title("Evolución de la loss durante el entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Loss")

    # Grilla y leyenda
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Guardar figura
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Mostrar figura en pantalla
    plt.show()

    print(f"Figura guardada en: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizar la loss del entrenamiento")

    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/training_log.csv",
        help="Ruta del archivo CSV con la loss"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/training_loss.png",
        help="Ruta donde se guardará la figura"
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=100,
        help="Tamaño de la ventana para suavizar la loss"
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Usar escala logarítmica en el eje Y"
    )

    args = parser.parse_args()

    plot_loss(
        csv_path=args.csv,
        output_path=args.output,
        smooth_window=args.smooth,
        log_scale=args.log
    )