"""Tests for all figures (written before implementation)."""

from pathlib import Path

from iris_analysis.data import CLASS_NAMES, FEATURES, load_iris_dataframe
from iris_analysis.eda import perform_pca
from iris_analysis.model import build_model, evaluate_model, split_data, train_model
from iris_analysis.visualize import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_learning_curve,
    plot_pca_3d,
)


def _assert_image(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 1000


def test_pca_3d_plot_is_created(tmp_path: Path):
    df = load_iris_dataframe()
    pca = perform_pca(df)
    output = tmp_path / "pca_3d.png"

    plot_pca_3d(
        pca["transformed"],
        df["species"],
        CLASS_NAMES,
        pca["explained_variance_ratio"],
        output,
    )

    _assert_image(output)


def test_eda_plots_are_created(tmp_path: Path):
    df = load_iris_dataframe()
    distributions = tmp_path / "feature_distributions.png"
    correlation = tmp_path / "correlation_heatmap.png"

    plot_feature_distributions(df, FEATURES, output_path=distributions)
    plot_correlation_heatmap(df, FEATURES, output_path=correlation)

    _assert_image(distributions)
    _assert_image(correlation)


def test_model_plots_are_created(tmp_path: Path):
    df = load_iris_dataframe()
    X_train, X_test, y_train, y_test = split_data(df, random_state=42)
    model, scaler = train_model(build_model(random_state=42), X_train, y_train)
    evaluation = evaluate_model(model, scaler, X_test, y_test)
    learning = tmp_path / "learning_curve.png"
    confusion = tmp_path / "confusion_matrix.png"
    metrics = tmp_path / "class_metrics.png"

    plot_learning_curve(model.loss_curve_, learning)
    plot_confusion_matrix(evaluation["confusion_matrix"], CLASS_NAMES, confusion)
    plot_class_metrics(evaluation["report"], CLASS_NAMES, metrics)

    _assert_image(learning)
    _assert_image(confusion)
    _assert_image(metrics)
