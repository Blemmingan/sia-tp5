"""
grid_search_cv.py

Grid Search con validación cruzada para el autoencoder del ejercicio 1a.

Objetivo:
- Probar distintas arquitecturas, optimizadores, funciones de activación y learning rates.
- Comparar las configuraciones con K-Fold Cross Validation.
- Reentrenar la mejor configuración usando los 32 caracteres.
- Guardar resultados en CSV para poder usarlos en la presentación.

Dónde poner este archivo:
    ej1/src/grid_search_cv.py

Cómo ejecutarlo desde la carpeta ej1:
    $env:PYTHONPATH="."       # PowerShell, Windows
    python src/grid_search_cv.py

O en Linux/Mac/Git Bash:
    PYTHONPATH=. python3 src/grid_search_cv.py
"""

import os
import csv
import json
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.font_loader import load_font
from src.autoencoder import Autoencoder
from src.optimizers import get_optimizer


# ============================================================
# 1. Espacio de búsqueda
# ============================================================
# Acá definimos todas las configuraciones que queremos probar.
# Puedes agregar o sacar arquitecturas, optimizadores, activaciones o learning rates.

GRID = {
    "architecture": [
        {
            "name": "shallow_8",
            "encoder": [35, 8, 2],
            "decoder": [2, 8, 35],
        },
        {
            "name": "medium_16_8",
            "encoder": [35, 16, 8, 2],
            "decoder": [2, 8, 16, 35],
        },
        {
            "name": "wide_24_12",
            "encoder": [35, 24, 12, 2],
            "decoder": [2, 12, 24, 35],
        },
        {
            "name": "deep_16_10_5",
            "encoder": [35, 16, 10, 5, 2],
            "decoder": [2, 5, 10, 16, 35],
        },
    ],
    "optimizer": ["sgd", "momentum", "adam"],
    "hidden_act": ["tanh", "relu", "leaky_relu"],
    "learning_rate": [0.001, 0.01, 0.05],
}


# ============================================================
# 2. Funciones auxiliares para loss, evaluación y folds
# ============================================================

def binary_cross_entropy(y_true, y_pred):
    """
    Calcula la Binary Cross Entropy.

    Se usa porque los píxeles originales son binarios: 0 o 1.
    La salida del autoencoder es continua entre 0 y 1 gracias a sigmoid.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(
        y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
    )


def evaluate_model(model, X):
    """
    Evalúa un modelo sobre un conjunto X.

    Como es un autoencoder, la entrada y la salida esperada son iguales.

    Retorna:
    - loss: error BCE promedio
    - max_pixel_error: máximo número de píxeles incorrectos en un carácter
    - mean_pixel_error: promedio de errores de píxeles por carácter
    - n_ok_1px: cantidad de caracteres reconstruidos con <= 1 píxel incorrecto
    - pixel_errors: lista con el error de cada carácter
    """
    y_true = X
    y_pred = model.forward(X)

    loss = binary_cross_entropy(y_true, y_pred)

    # Binarizamos la salida con umbral 0.5 para comparar con los patrones originales.
    y_bin = (y_pred >= 0.5).astype(np.float32)

    # Error por carácter: número de píxeles diferentes entre original y reconstrucción.
    pixel_errors = np.sum(np.abs(y_bin - y_true), axis=1).astype(int)

    return {
        "loss": float(loss),
        "max_pixel_error": int(np.max(pixel_errors)),
        "mean_pixel_error": float(np.mean(pixel_errors)),
        "n_ok_1px": int(np.sum(pixel_errors <= 1)),
        "pixel_errors": pixel_errors.tolist(),
    }


def make_k_folds(n_samples, k=4, seed=42):
    """
    Crea índices para K-Fold Cross Validation sin usar sklearn.

    Para 32 caracteres y k=4:
    - cada fold valida con 8 caracteres
    - cada fold entrena con 24 caracteres
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)

    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))

    return splits


# ============================================================
# 3. Entrenamiento de una configuración
# ============================================================

def train_one_model(X_train, config, epochs=5000, seed=42, loss_type="bce"):
    """
    Entrena un autoencoder para una configuración específica.

    Parámetros:
    - X_train: datos de entrenamiento
    - config: diccionario con arquitectura, optimizador, activación y learning rate
    - epochs: cantidad de épocas
    - seed: semilla para que el experimento sea reproducible
    - loss_type: "bce" o "mse"

    Retorna:
    - model: modelo entrenado con los mejores pesos encontrados
    - best_train_loss: menor loss de entrenamiento
    - history: lista con la loss en cada época
    """
    np.random.seed(seed)

    model = Autoencoder(
        encoder_layers=config["encoder"],
        decoder_layers=config["decoder"],
        hidden_act=config["hidden_act"],
        out_act="sigmoid",
    )

    optimizer = get_optimizer(
        config["optimizer"],
        lr=config["learning_rate"],
        beta1=0.9,
        beta2=0.999,
        momentum=0.9,
    )

    best_train_loss = float("inf")
    best_params = None
    history = []

    # En este ejercicio usamos full-batch: todos los patrones de entrenamiento juntos.
    # Para autoencoders pequeños y 32 datos, esto es suficiente y estable.
    for epoch in range(epochs):
        model.forward(X_train)
        grads, train_loss = model.backward(
            X_train,
            X_train,
            loss_type=loss_type,
            grad_clip_val=1.0,
        )
        optimizer.update(model.params, grads)
        history.append(float(train_loss))

        # Guardamos los mejores pesos según la loss de entrenamiento.
        if train_loss < best_train_loss:
            best_train_loss = float(train_loss)
            best_params = [np.copy(p) for p in model.params]

    # Restaurar el mejor modelo encontrado durante el entrenamiento.
    model.params = best_params

    return model, best_train_loss, history


# ============================================================
# 4. Grid Search con Cross Validation
# ============================================================

def build_configurations(quick=False):
    """
    Convierte el GRID en una lista de configuraciones concretas.

    Ejemplo de configuración generada:
    - arquitectura medium_16_8
    - optimizador adam
    - activación tanh
    - learning rate 0.01
    """
    configs = []

    # Modo rápido: sirve para verificar que el script funciona sin probar todo el grid.
    # Después, para la presentación, conviene correr el grid completo.
    if quick:
        architectures = [GRID["architecture"][1], GRID["architecture"][2]]
        optimizers = ["adam"]
        hidden_acts = ["tanh", "relu"]
        learning_rates = [0.01, 0.05]
    else:
        architectures = GRID["architecture"]
        optimizers = GRID["optimizer"]
        hidden_acts = GRID["hidden_act"]
        learning_rates = GRID["learning_rate"]

    for arch, optimizer, hidden_act, lr in itertools.product(
        architectures,
        optimizers,
        hidden_acts,
        learning_rates,
    ):
        configs.append(
            {
                "arch_name": arch["name"],
                "encoder": arch["encoder"],
                "decoder": arch["decoder"],
                "optimizer": optimizer,
                "hidden_act": hidden_act,
                "learning_rate": lr,
            }
        )

    return configs


def run_grid_search(X, k=4, epochs=5000, seed=42, loss_type="bce", output_dir="outputs_grid", quick=False):
    """
    Ejecuta Grid Search con K-Fold Cross Validation.

    Para cada configuración:
    1. Divide los 32 caracteres en K folds.
    2. Entrena con K-1 folds.
    3. Valida con el fold restante.
    4. Promedia la loss y los errores de píxeles.
    5. Reentrena la misma configuración con los 32 caracteres para verificar el objetivo final.

    El objetivo final del TP es reconstruir los 32 caracteres con máximo 1 píxel incorrecto.
    La validación cruzada sirve para comparar configuraciones de manera más sistemática.
    """
    os.makedirs(output_dir, exist_ok=True)

    configs = build_configurations(quick=quick)
    splits = make_k_folds(n_samples=len(X), k=k, seed=seed)

    all_results = []

    print(f"Total de configuraciones a probar: {len(configs)}")
    if quick:
        print("Modo rápido activado: se prueba solo un subconjunto del grid.")
    print(f"K-Fold: k={k}")
    print(f"Épocas por entrenamiento: {epochs}")
    print("-" * 80)

    for config_id, config in enumerate(configs, start=1):
        fold_results = []

        print(
            f"[{config_id}/{len(configs)}] "
            f"arch={config['arch_name']} | "
            f"opt={config['optimizer']} | "
            f"act={config['hidden_act']} | "
            f"lr={config['learning_rate']}"
        )

        # ------------------------------
        # Cross Validation
        # ------------------------------
        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            X_train = X[train_idx]
            X_val = X[val_idx]

            # Cambiamos un poco la semilla por fold para evitar exactamente la misma inicialización.
            fold_seed = seed + fold_id

            model, best_train_loss, history = train_one_model(
                X_train=X_train,
                config=config,
                epochs=epochs,
                seed=fold_seed,
                loss_type=loss_type,
            )

            train_eval = evaluate_model(model, X_train)
            val_eval = evaluate_model(model, X_val)

            fold_results.append(
                {
                    "fold": fold_id,
                    "train_loss": train_eval["loss"],
                    "val_loss": val_eval["loss"],
                    "train_max_pixel_error": train_eval["max_pixel_error"],
                    "val_max_pixel_error": val_eval["max_pixel_error"],
                    "train_mean_pixel_error": train_eval["mean_pixel_error"],
                    "val_mean_pixel_error": val_eval["mean_pixel_error"],
                    "val_n_ok_1px": val_eval["n_ok_1px"],
                }
            )

        # ------------------------------
        # Promedios de Cross Validation
        # ------------------------------
        mean_train_loss = float(np.mean([r["train_loss"] for r in fold_results]))
        mean_val_loss = float(np.mean([r["val_loss"] for r in fold_results]))
        mean_val_max_pixel_error = float(np.mean([r["val_max_pixel_error"] for r in fold_results]))
        mean_val_mean_pixel_error = float(np.mean([r["val_mean_pixel_error"] for r in fold_results]))
        mean_val_n_ok_1px = float(np.mean([r["val_n_ok_1px"] for r in fold_results]))

        # ------------------------------
        # Reentrenamiento con los 32 caracteres
        # ------------------------------
        # Esto es importante porque el TP pide aprender todo el set de 32 patrones.
        final_model, final_best_train_loss, final_history = train_one_model(
            X_train=X,
            config=config,
            epochs=epochs,
            seed=seed,
            loss_type=loss_type,
        )
        full_eval = evaluate_model(final_model, X)

        success_full_dataset = full_eval["max_pixel_error"] <= 1

        row = {
            "config_id": config_id,
            "arch_name": config["arch_name"],
            "encoder": "-".join(map(str, config["encoder"])),
            "decoder": "-".join(map(str, config["decoder"])),
            "optimizer": config["optimizer"],
            "hidden_act": config["hidden_act"],
            "learning_rate": config["learning_rate"],
            "cv_train_loss_mean": mean_train_loss,
            "cv_val_loss_mean": mean_val_loss,
            "cv_val_max_pixel_error_mean": mean_val_max_pixel_error,
            "cv_val_mean_pixel_error_mean": mean_val_mean_pixel_error,
            "cv_val_n_ok_1px_mean": mean_val_n_ok_1px,
            "full_train_loss": full_eval["loss"],
            "full_max_pixel_error": full_eval["max_pixel_error"],
            "full_mean_pixel_error": full_eval["mean_pixel_error"],
            "full_n_ok_1px": full_eval["n_ok_1px"],
            "success_full_dataset": success_full_dataset,
        }

        all_results.append(row)

        print(
            f"    CV val loss={mean_val_loss:.6f} | "
            f"Full max pixel error={full_eval['max_pixel_error']} | "
            f"Full chars <=1px={full_eval['n_ok_1px']}/32 | "
            f"Success={success_full_dataset}"
        )

    # Ordenamos resultados:
    # 1. Primero las configs que logran aprender los 32 caracteres.
    # 2. Luego menor error máximo en full dataset.
    # 3. Luego menor loss de validación promedio.
    all_results = sorted(
        all_results,
        key=lambda r: (
            not r["success_full_dataset"],
            r["full_max_pixel_error"],
            r["cv_val_loss_mean"],
        ),
    )

    save_results_csv(all_results, os.path.join(output_dir, "grid_search_results.csv"))
    save_best_config(all_results[0], os.path.join(output_dir, "best_config.json"))
    plot_grid_results(all_results, os.path.join(output_dir, "grid_search_top_results.png"))

    return all_results


# ============================================================
# 5. Guardado de resultados
# ============================================================

def save_results_csv(results, csv_path):
    """
    Guarda todos los resultados en un CSV.
    Este archivo es ideal para hacer una tabla en la presentación.
    """
    if not results:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados guardados en: {csv_path}")


def save_best_config(best_result, json_path):
    """
    Guarda la mejor configuración en JSON.
    """
    with open(json_path, "w") as f:
        json.dump(best_result, f, indent=4)

    print(f"Mejor configuración guardada en: {json_path}")


def plot_grid_results(results, output_path, top_n=15):
    """
    Genera una figura simple con las mejores configuraciones.

    Se grafica el error máximo de píxeles en el dataset completo.
    Esto permite ver rápidamente qué configuraciones cumplen el objetivo del TP.
    """
    top_results = results[:top_n]

    labels = []
    values = []

    for r in top_results:
        label = (
            f"{r['arch_name']}\n"
            f"{r['optimizer']}, {r['hidden_act']}, lr={r['learning_rate']}"
        )
        labels.append(label)
        values.append(r["full_max_pixel_error"])

    plt.figure(figsize=(14, 7))
    plt.bar(range(len(values)), values)
    plt.axhline(1, linestyle="--", linewidth=2, label="Objetivo: máximo 1 píxel")
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("Máximo error de píxeles en los 32 caracteres")
    plt.title("Comparación de configuraciones del autoencoder")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figura comparativa guardada en: {output_path}")


# ============================================================
# 6. Programa principal
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search CV para Autoencoder")

    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="Cantidad de épocas para cada entrenamiento",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Número de folds para cross validation",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_grid",
        help="Carpeta donde guardar los resultados",
    )

    parser.add_argument(
        "--font",
        type=str,
        default="font.h",
        help="Ruta al archivo font.h",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Probar solo un subconjunto pequeño del grid para verificar que todo funciona",
    )

    args = parser.parse_args()

    # Cargar los 32 caracteres de 5x7.
    X = load_font(args.font)

    print(f"Dataset cargado: {X.shape[0]} caracteres, {X.shape[1]} píxeles por carácter")

    results = run_grid_search(
        X=X,
        k=args.k,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        quick=args.quick,
    )

    best = results[0]

    print("\n" + "=" * 80)
    print("MEJOR CONFIGURACIÓN")
    print("=" * 80)
    print(f"Arquitectura: {best['encoder']} -> {best['decoder']}")
    print(f"Optimizador: {best['optimizer']}")
    print(f"Activación: {best['hidden_act']}")
    print(f"Learning rate: {best['learning_rate']}")
    print(f"CV val loss promedio: {best['cv_val_loss_mean']:.6f}")
    print(f"Error máximo en los 32 caracteres: {best['full_max_pixel_error']}")
    print(f"Caracteres con <= 1 píxel incorrecto: {best['full_n_ok_1px']}/32")
    print(f"Cumple objetivo del TP: {best['success_full_dataset']}")
