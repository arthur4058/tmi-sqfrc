import numpy as np
import torch

from scripts.fair_comparison_models import DabiriCNN, DeepInsightViT, MASOMSFAdapter
from scripts.fair_comparison_protocol import evaluate_probabilities, validate_protocol


def test_shared_metric_definition():
    labels = np.array([0, 1, 2, 3, 4])
    probabilities = np.eye(5, dtype=np.float32)
    metrics = evaluate_probabilities(probabilities, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_user_disjoint_five_rate_protocol():
    manifest = validate_protocol()
    assert len(manifest["user_partitions"]["train"]) == 44
    assert len(manifest["user_partitions"]["val"]) == 6
    assert len(manifest["user_partitions"]["test"]) == 12
    assert set(manifest["rates"]) == {"5", "10", "20", "30", "60"}


def test_adapted_model_output_shapes():
    sequence = torch.zeros(2, 9, 64)
    assert DabiriCNN()(sequence).shape == (2, 5)
    assert MASOMSFAdapter()(sequence).shape == (2, 5)
    assert DeepInsightViT(9)(torch.zeros(2, 1, 9, 9)).shape == (2, 5)
