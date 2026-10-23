"""End-to-end tests for pipeline and generated paper (written before implementation)."""

from pathlib import Path

from iris_analysis.pipeline import run_pipeline


def test_pipeline_produces_data_figures_and_paper(tmp_path: Path):
    data_path = tmp_path / "data" / "iris.csv"
    figure_dir = tmp_path / "outputs" / "figures"
    report_path = tmp_path / "reports" / "paper.md"

    result = run_pipeline(
        data_path=data_path,
        figure_dir=figure_dir,
        report_path=report_path,
        random_state=42,
    )

    assert data_path.exists()
    assert report_path.exists()
    assert result["evaluation"]["accuracy"] >= 0.85

    expected_figures = {
        "pca_3d",
        "feature_distributions",
        "correlation_heatmap",
        "learning_curve",
        "confusion_matrix",
        "class_metrics",
    }
    existing = {path.stem for path in figure_dir.glob("*.png")}
    assert expected_figures <= existing

    report = report_path.read_text(encoding="utf-8")
    for heading in (
        "摘要",
        "研究背景",
        "資料集",
        "資料探索",
        "方法",
        "實驗結果",
        "討論",
        "結論",
        "參考文獻",
    ):
        assert heading in report
    for figure in expected_figures:
        assert f"{figure}.png" in report
    assert "setosa" in report
    assert "accuracy" in report.lower()
