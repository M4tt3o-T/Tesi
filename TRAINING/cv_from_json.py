import argparse
import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def resolve_path(
    data_root: Path, path_template: str, class_name: str, file_base: str
) -> Path:
    rel = path_template.format(class_name=class_name, file=file_base)
    p = Path(rel)
    if not p.is_absolute():
        p = data_root / p
    return p.with_suffix(".csv") if p.suffix.lower() != ".csv" else p


def sniff_sep(sample: str) -> Optional[str]:
    # Try to sniff common delimiters
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\\t", "|"])
        return dialect.delimiter
    except Exception:
        return None


def load_table(
    csv_path: Path, sep: Optional[str] = None, decimal: str = ".", index_col=None
) -> pd.DataFrame:
    # Read a small sample to sniff delimiter if sep is None
    if sep is None:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(65536)
        autod = sniff_sep(sample)
    else:
        autod = sep
    try:
        df = pd.read_csv(
            csv_path, sep=autod if autod else ",", decimal=decimal, index_col=index_col
        )
        # If only one column and we forced comma, retry alternative common delimiters
        if df.shape[1] == 1 and sep is None:
            for alt in [";", "\\t", "|"]:
                df_alt = pd.read_csv(
                    csv_path, sep=alt, decimal=decimal, index_col=index_col
                )
                if df_alt.shape[1] > 1:
                    df = df_alt
                    break
        # Drop fully empty columns created by bad separators
        df = df.dropna(axis=1, how="all")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing file: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Failed reading {csv_path}: {e}")


def split_xy(
    df: pd.DataFrame, label_col: Optional[str]
) -> Tuple[pd.DataFrame, np.ndarray]:
    if df.shape[1] < 2:
        raise ValueError(
            "CSV must contain at least 2 columns (features + label). "
            "Hint: check the delimiter. Try --sep ';' or ensure the file has a header."
        )
    if label_col is None:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype(str).to_numpy()
    else:
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. Columns: {list(df.columns)}"
            )
        y = df[label_col].astype(str).to_numpy()
        X = df.drop(columns=[label_col])

    # Keep numeric features only
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] != 16:
        print(
            f"Warning: expected 16 numeric features, got {X_num.shape[1]}. Using numeric columns only."
        )
    if X_num.empty:
        raise ValueError(
            "No usable numeric features after filtering to numeric columns."
        )
    X_num = X_num.fillna(0.0)
    return X_num.reset_index(drop=True), y


def prepare_xy(
    listing: Dict[str, List[str]],
    data_root: Path,
    path_template: str,
    sep: Optional[str],
    decimal: str,
    index_col,
    label_col: Optional[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    frames = []
    labels = []
    for class_name, files in listing.items():
        for file_base in files:
            csv_path = resolve_path(data_root, path_template, class_name, file_base)
            df = load_table(csv_path, sep=sep, decimal=decimal, index_col=index_col)
            X_part, y_part = split_xy(df, label_col)
            frames.append(X_part)
            labels.append(y_part)
    X = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    y = np.concatenate(labels) if labels else np.array([])
    return X, y


def build_model(model_type: str) -> Pipeline:
    if model_type == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(25, 50, 25), max_iter=400, random_state=42
                    ),
                ),
            ]
        )
    elif model_type == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("knn", KNeighborsClassifier(n_neighbors=100)),
            ]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def align_columns(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_cols = X_train.columns
    X_test_aligned = X_test.reindex(columns=train_cols, fill_value=0)
    return X_train, X_test_aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="Path to splits JSON file")
    ap.add_argument("--data-root", default=".", help="Root folder containing CSV data")
    ap.add_argument(
        "--path-template",
        default="{class_name}/{file}.csv",
        help="Relative path template from data-root to CSV",
    )
    ap.add_argument(
        "--sep",
        default=None,
        help="CSV separator. Default: auto-detect among , ; 	 |",
    )
    ap.add_argument("--decimal", default=".", help="CSV decimal point")
    ap.add_argument(
        "--index-col", default=None, help="CSV index column name or integer", type=str
    )
    ap.add_argument(
        "--label-col", default=None, help="Label column name (default: last column)"
    )
    ap.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "knn"],
        help="Model type: mlp or knn (default: mlp)",
    )
    ap.add_argument(
        "--save-report", default=None, help="Optional path to save per-fold report CSV"
    )
    args = ap.parse_args()

    index_col = None
    if args.index_col is not None and str(args.index_col).lower() != "none":
        try:
            index_col = int(args.index_col)
        except ValueError:
            index_col = args.index_col

    data_root = Path(args.data_root).resolve()

    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    metrics = []
    rows_for_report = []
    all_labels = set()

    # First pass: load data per fold and show class distributions
    folds_data = []  # list of dicts with fold_name, X_train, y_train, X_test, y_test
    for fold_name, fold in splits.items():
        train_listing = fold.get("train", {})
        test_listing = fold.get("valid", {})

        X_train, y_train = prepare_xy(
            train_listing,
            data_root,
            args.path_template,
            sep=args.sep,
            decimal=args.decimal,
            index_col=index_col,
            label_col=args.label_col,
        )
        X_test, y_test = prepare_xy(
            test_listing,
            data_root,
            args.path_template,
            sep=args.sep,
            decimal=args.decimal,
            index_col=index_col,
            label_col=args.label_col,
        )

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError(f"{fold_name}: empty train or test set.")

        all_labels.update(y_train)
        all_labels.update(y_test)

        folds_data.append(
            {
                "fold_name": fold_name,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }
        )

        # Summary counts
        total_train = len(y_train)
        total_test = len(y_test)
        ser_train = pd.Series(y_train).value_counts().sort_index()
        ser_test = pd.Series(y_test).value_counts().sort_index()
        classes = sorted(set(ser_train.index).union(set(ser_test.index)))

        print(f"=== {fold_name} class distribution ===")
        print(
            f"{'Class':30} {'Train #':>10} {'Train %':>9} {'Test #':>10} {'Test %':>9}"
        )
        for cls in classes:
            n_tr = int(ser_train.get(cls, 0))
            n_te = int(ser_test.get(cls, 0))
            total_cls = n_tr + n_te
            if total_cls > 0:
                p_tr = n_tr / total_cls * 100.0
                p_te = n_te / total_cls * 100.0
            else:
                p_tr = 0.0
                p_te = 0.0
            print(f"{str(cls)[:30]:30} {n_tr:10d} {p_tr:8.2f}% {n_te:10d} {p_te:8.2f}%")
        total_all = total_train + total_test
        if total_all > 0:
            p_tr_total = total_train / total_all * 100.0
            p_te_total = total_test / total_all * 100.0
        else:
            p_tr_total = 0.0
            p_te_total = 0.0
        print(
            f"{'TOTAL':30} {total_train:10d} {p_tr_total:8.2f}% {total_test:10d} {p_te_total:8.2f}%"
        )

    labels = sorted(all_labels)

    # Confirm before training
    try:
        resp = input("\nProceed with training? [y/N]: ").strip().lower()
    except EOFError:
        resp = "n"
    if resp not in ("y", "yes"):
        print("Aborted by user.")
        return

    # Second pass: train and evaluate
    cms = []
    for item in folds_data:
        fold_name = item["fold_name"]
        X_train = item["X_train"]
        y_train = item["y_train"]
        X_test = item["X_test"]
        y_test = item["y_test"]

        # Align columns
        X_train, X_test = align_columns(X_train, X_test)

        model = build_model(args.model)
        model.fit(X_train.values, y_train)
        y_pred = model.predict(X_test.values)

        acc = accuracy_score(y_test, y_pred)
        metrics.append(acc)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        cms.append(cm_normalized)

        rows_for_report.append(
            {"fold": fold_name, "accuracy": acc, "n_test_samples": len(y_test)}
        )
        print(f"{fold_name}: accuracy={acc:.4f}, n_test={len(y_test)}")

    mean_acc = float(np.mean(metrics)) if metrics else float("nan")
    std_acc = float(np.std(metrics, ddof=1)) if len(metrics) > 1 else 0.0

    print(f"Mean accuracy over {len(metrics)} folds: {mean_acc:.4f}")
    print(f"Std deviation: {std_acc:.4f}")

    if cms:
        mean_cm = np.mean(cms, axis=0)
        std_cm = np.std(cms, axis=0, ddof=1) if len(cms) > 1 else np.zeros_like(mean_cm)

        # Plot Mean Confusion Matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            mean_cm,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
        )
        plt.title("Mean Confusion Matrix (%)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # Plot Std Confusion Matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            std_cm,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Reds",
        )
        plt.title("Std Confusion Matrix (%)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    if args.save_report:
        out = pd.DataFrame(rows_for_report)
        out.loc[len(out.index)] = {
            "fold": "mean",
            "accuracy": mean_acc,
            "n_test_samples": int(
                np.sum([r["n_test_samples"] for r in rows_for_report])
            ),
        }
        out.loc[len(out.index)] = {
            "fold": "std",
            "accuracy": std_acc,
            "n_test_samples": None,
        }
        out.to_csv(args.save_report, index=False)
        print(f"Saved report to {args.save_report}")


if __name__ == "__main__":
    main()
