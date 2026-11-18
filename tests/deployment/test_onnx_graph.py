import pytest

from src.deployment.onnx_graph import topologically_sort_graph


def test_topological_sort_repairs_late_producer_order():
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor = onnx.TensorProto.FLOAT
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["hidden"], ["output"], name="consumer"),
            helper.make_node("Identity", ["input"], ["hidden"], name="producer"),
        ],
        "unsorted",
        [helper.make_tensor_value_info("input", tensor, [1])],
        [helper.make_tensor_value_info("output", tensor, [1])],
    )

    with pytest.raises(Exception):
        onnx.checker.check_model(helper.make_model(graph))

    topologically_sort_graph(graph)
    repaired = helper.make_model(graph)
    onnx.checker.check_model(repaired)
    assert [node.name for node in graph.node] == ["producer", "consumer"]


def test_topological_sort_rejects_missing_producer():
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Relu", ["missing"], ["output"])],
        "invalid",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )

    with pytest.raises(ValueError, match="missing"):
        topologically_sort_graph(graph)
