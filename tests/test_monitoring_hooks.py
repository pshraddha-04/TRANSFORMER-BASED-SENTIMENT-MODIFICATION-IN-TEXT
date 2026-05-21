from app.services.monitoring_service import PredictionMonitor


def test_prediction_monitor_snapshot_counts():
    monitor = PredictionMonitor(log_every_n=100, dominance_threshold=0.9, window_size=2)
    monitor.record_prediction("joy")
    monitor.record_prediction("sadness")
    monitor.record_prediction("joy")

    snapshot = monitor.snapshot()
    assert snapshot["total"] == 3
    assert snapshot["counts"]["joy"] == 2
    assert snapshot["counts"]["sadness"] == 1
    assert snapshot["distribution"]["joy"] == 0.6667
    assert snapshot["window_total"] == 2
    assert snapshot["window_counts"]["joy"] == 1
    assert snapshot["window_counts"]["sadness"] == 1
    assert snapshot["window_size"] == 2


def test_prediction_monitor_emits_drift_warning(caplog):
    monitor = PredictionMonitor(log_every_n=4, dominance_threshold=0.7)

    with caplog.at_level("WARNING"):
        monitor.record_prediction("anger")
        monitor.record_prediction("anger")
        monitor.record_prediction("anger")
        monitor.record_prediction("anger")

    assert "drift signal" in caplog.text


def test_prediction_monitor_uses_rolling_window_for_drift(caplog):
    monitor = PredictionMonitor(log_every_n=6, dominance_threshold=0.7, window_size=4)

    with caplog.at_level("WARNING"):
        for label in ["anger", "joy", "joy", "anger", "anger", "anger"]:
            monitor.record_prediction(label)

    assert "drift signal" in caplog.text
    snapshot = monitor.snapshot()
    assert snapshot["distribution"]["anger"] == 0.6667
    assert snapshot["window_distribution"]["anger"] == 0.75


