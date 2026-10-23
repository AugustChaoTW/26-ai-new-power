"""End-to-end Iris analysis pipeline."""

from pathlib import Path
from typing import Any

import numpy as np

from iris_analysis.data import CLASS_NAMES, load_iris_dataframe, save_dataset
from iris_analysis.eda import (
    class_balance,
    dataset_overview,
    describe_features,
    detect_outliers,
    perform_pca,
)
from iris_analysis.model import build_model, evaluate_model, split_data, train_model
from iris_analysis.report import generate_report
from iris_analysis.visualize import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_learning_curve,
    plot_pca_3d,
)


def run_pipeline(
    data_path: Path | str = "data/iris.csv",
    figure_dir: Path | str = "outputs/figures",
    report_path: Path | str = "reports/paper.md",
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Execute the full study and return the collected results."""
    data_path = Path(data_path)
    figure_dir = Path(figure_dir)
    report_path = Path(report_path)

    df = load_iris_dataframe()
    save_dataset(df, data_path)

    overview = dataset_overview(df)
    describe_table = describe_features(df)
    outliers = detect_outliers(df)
    balance = class_balance(df)
    pca = perform_pca(df)

    split = split_data(df, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = split
    model = build_model(random_state=random_state)
    model, scaler = train_model(model, X_train, y_train)
    evaluation = evaluate_model(model, scaler, X_test, y_test)

    figures = {
        "pca_3d": plot_pca_3d(
            pca["transformed"],
            df["species"],
            CLASS_NAMES,
            pca["explained_variance_ratio"],
            figure_dir / "pca_3d.png",
        ),
        "feature_distributions": plot_feature_distributions(
            df, output_path=figure_dir / "feature_distributions.png"
        ),
        "correlation_heatmap": plot_correlation_heatmap(
            df, output_path=figure_dir / "correlation_heatmap.png"
        ),
        "learning_curve": plot_learning_curve(
            model.loss_curve_, figure_dir / "learning_curve.png"
        ),
        "confusion_matrix": plot_confusion_matrix(
            evaluation["confusion_matrix"], CLASS_NAMES, figure_dir / "confusion_matrix.png"
        ),
        "class_metrics": plot_class_metrics(
            evaluation["report"], CLASS_NAMES, figure_dir / "class_metrics.png"
        ),
    }

    context: dict[str, Any] = {
        "data_path": data_path,
        "overview": overview,
        "describe_table": describe_table,
        "outliers": outliers,
        "class_balance": balance,
        "pca": pca,
        "split_info": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": test_size,
            "random_state": random_state,
            "train_counts": np.bincount(y_train, minlength=len(CLASS_NAMES)).tolist(),
            "test_counts": np.bincount(y_test, minlength=len(CLASS_NAMES)).tolist(),
        },
        "model_info": {
            "hidden_layers": model.hidden_layer_sizes,
            "activation": model.activation,
            "solver": model.solver,
            "learning_rate_init": model.learning_rate_init,
            "alpha": model.alpha,
            "max_iter": model.max_iter,
            "n_iter": int(model.n_iter_),
            "random_state": model.random_state,
            "initial_loss": float(model.loss_curve_[0]),
            "final_loss": float(model.loss_curve_[-1]),
        },
        "evaluation": evaluation,
        "figures": figures,
    }

    generate_report(context, report_path)
    context["report_path"] = report_path
    return context
