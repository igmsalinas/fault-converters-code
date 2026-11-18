"""Small ONNX graph-normalization helpers shared by deployment exporters."""


def topologically_sort_graph(graph):
    """Sort an ONNX GraphProto in place using stable dependency order.

    Older ``onnxconverter-common`` releases can append FP32 I/O Cast nodes after
    nodes that consume them. ONNX Runtime may tolerate this, but the checker and
    TensorRT parser require producers to precede consumers.
    """
    available = {value.name for value in graph.input}
    available.update(value.name for value in graph.initializer)
    remaining = list(graph.node)
    ordered = []

    while remaining:
        deferred = []
        progressed = False
        for node in remaining:
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(name for name in node.output if name)
                progressed = True
            else:
                deferred.append(node)
        if not progressed:
            unresolved = sorted({
                name
                for node in deferred
                for name in node.input
                if name and name not in available
            })
            raise ValueError(
                "Cannot topologically sort ONNX graph; unresolved inputs: {}".format(
                    unresolved
                )
            )
        remaining = deferred

    del graph.node[:]
    graph.node.extend(ordered)
    return graph
