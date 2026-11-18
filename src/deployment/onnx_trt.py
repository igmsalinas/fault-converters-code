"""
Keras to ONNX and TensorRT Compilation Pipeline
================================================

Handles conversion of Keras models to ONNX format, and subsequent
compilation into standalone NVIDIA TensorRT engines.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from typing import Optional, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)

_register_keras_serializable = getattr(
    keras.utils,
    "register_keras_serializable",
    lambda **_kwargs: (lambda cls: cls),
)


@_register_keras_serializable(package="CARLADeployment")
class TensorRTCompatibleMultiHeadAttention(keras.layers.Layer):
    """Self-attention expressed with rank-3 matmuls for legacy TensorRT.

    TensorRT 8.2 (JetPack 4.6) can parse the graph emitted for Keras
    ``MultiHeadAttention`` but produces numerically incorrect results.  This
    layer deliberately keeps the same weight shapes/order as Keras MHA while
    lowering inference to simple reshape, transpose, matmul and softmax ops.
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: Optional[int] = None,
        use_bias: bool = True,
        output_shape: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim if value_dim is not None else key_dim)
        self.use_bias = bool(use_bias)
        # ``Layer.output_shape`` is a read-only property in the standalone
        # Keras 2.7 shipped for JetPack 4.6.  Keep the serialized constructor
        # key for compatibility, but never shadow the framework property.
        self._output_shape_override = output_shape

    def build(self, input_shape):
        query_shape = (
            input_shape[0]
            if isinstance(input_shape, (list, tuple))
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
            else input_shape
        )
        input_dim = int(query_shape[-1])
        output_dim = int(
            self._output_shape_override
            if self._output_shape_override is not None
            else input_dim
        )
        self.query_kernel = self.add_weight(
            name="query_kernel", shape=(input_dim, self.num_heads, self.key_dim)
        )
        self.query_bias = (
            self.add_weight(name="query_bias", shape=(self.num_heads, self.key_dim))
            if self.use_bias
            else None
        )
        self.key_kernel = self.add_weight(
            name="key_kernel", shape=(input_dim, self.num_heads, self.key_dim)
        )
        self.key_bias = (
            self.add_weight(name="key_bias", shape=(self.num_heads, self.key_dim))
            if self.use_bias
            else None
        )
        self.value_kernel = self.add_weight(
            name="value_kernel", shape=(input_dim, self.num_heads, self.value_dim)
        )
        self.value_bias = (
            self.add_weight(name="value_bias", shape=(self.num_heads, self.value_dim))
            if self.use_bias
            else None
        )
        self.output_kernel = self.add_weight(
            name="output_kernel", shape=(self.num_heads, self.value_dim, output_dim)
        )
        self.output_bias = (
            self.add_weight(name="output_bias", shape=(output_dim,))
            if self.use_bias
            else None
        )
        super().build(input_shape)

    def _project(self, x, kernel, bias, head_dim):
        projected = tf.matmul(
            x,
            tf.reshape(kernel, (int(x.shape[-1]), self.num_heads * head_dim)),
        )
        if bias is not None:
            projected = projected + tf.reshape(bias, (self.num_heads * head_dim,))
        return tf.reshape(projected, (-1, int(x.shape[1]), self.num_heads, head_dim))

    def call(self, query, value=None, key=None, **kwargs):
        del kwargs
        value = query if value is None else value
        key = value if key is None else key
        q = tf.transpose(
            self._project(query, self.query_kernel, self.query_bias, self.key_dim),
            (0, 2, 1, 3),
        )
        k = tf.transpose(
            self._project(key, self.key_kernel, self.key_bias, self.key_dim),
            (0, 2, 1, 3),
        )
        v = tf.transpose(
            self._project(value, self.value_kernel, self.value_bias, self.value_dim),
            (0, 2, 1, 3),
        )
        # TensorRT 8.2 occasionally mis-fuses rank-4 batched matmuls after a
        # large Dense+Reshape decoder projection. Fold batch and heads so both
        # attention matmuls are unambiguous rank-3 operations.
        sequence_length = int(query.shape[1])
        q = tf.reshape(q, (-1, sequence_length, self.key_dim))
        k = tf.reshape(k, (-1, sequence_length, self.key_dim))
        v = tf.reshape(v, (-1, sequence_length, self.value_dim))
        scores = tf.matmul(q, k, transpose_b=True) * tf.cast(
            self.key_dim**-0.5, q.dtype
        )
        # Avoid TensorRT 8.2's incorrect Softmax fusion for the decoder's
        # [heads, sequence, sequence] attention tensor.
        shifted_scores = scores - tf.reduce_max(scores, axis=-1, keepdims=True)
        exponentials = tf.exp(shifted_scores)
        attention_weights = exponentials / tf.reduce_sum(
            exponentials, axis=-1, keepdims=True
        )
        context = tf.matmul(attention_weights, v)
        context = tf.reshape(context, (-1, self.num_heads, sequence_length, self.value_dim))
        context = tf.transpose(context, (0, 2, 1, 3))
        context = tf.reshape(context, (-1, sequence_length, self.num_heads * self.value_dim))
        output = tf.matmul(
            context,
            tf.reshape(self.output_kernel, (self.num_heads * self.value_dim, -1)),
        )
        if self.output_bias is not None:
            output = output + self.output_bias
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "use_bias": self.use_bias,
                "output_shape": self._output_shape_override,
            }
        )
        return config


def make_tensorrt_compatible_model(model: keras.Model) -> keras.Model:
    """Clone a Keras model with TensorRT-8-compatible attention layers."""

    def clone_layer(layer):
        if isinstance(layer, keras.Model):
            return keras.models.clone_model(layer, clone_function=clone_layer)
        if isinstance(layer, keras.layers.MultiHeadAttention):
            config = layer.get_config()
            output_shape = config.get("output_shape")
            if isinstance(output_shape, (list, tuple)) and len(output_shape) == 1:
                output_shape = output_shape[0]
            return TensorRTCompatibleMultiHeadAttention(
                num_heads=config["num_heads"],
                key_dim=config["key_dim"],
                value_dim=config.get("value_dim"),
                use_bias=config.get("use_bias", True),
                output_shape=output_shape,
                name=layer.name,
            )
        return layer.__class__.from_config(layer.get_config())

    compatible = keras.models.clone_model(model, clone_function=clone_layer)
    source_weights = model.get_weights()
    target_weights = compatible.get_weights()
    if len(source_weights) != len(target_weights):
        raise ValueError(
            "TensorRT-compatible clone changed the number of model weights: "
            f"{len(source_weights)} != {len(target_weights)}"
        )
    for index, (source, target) in enumerate(zip(source_weights, target_weights)):
        if source.shape != target.shape:
            raise ValueError(
                f"TensorRT-compatible weight {index} changed shape: "
                f"{source.shape} != {target.shape}"
            )
    compatible.set_weights(source_weights)
    return compatible


def _convert_recurrent_keras_to_onnx(
    model: keras.Model,
    output_path: str,
    opset: int,
) -> None:
    """Export a Keras recurrent autoencoder with native ONNX RNN operators.

    ``tf2onnx`` lowers Keras 3 GRU/LSTM layers to TensorFlow ``Loop`` graphs.
    TensorRT 8.2 can parse those graphs but its legacy Myelin compiler fails while
    fusing them on Jetson Nano. TensorRT 8.2 natively supports ONNX ``GRU`` and
    ``LSTM`` nodes, so this exporter maps the trained Keras weights to those nodes
    without changing the model equations.

    The supported topology is the encoder/decoder structure used by this project:
    unidirectional GRU or LSTM stacks, optional inference-only Dropout and
    reshape/flatten layers, Dense/Activation layers, one RepeatVector, and a final
    TimeDistributed(Dense). Unsupported recurrent configurations fail explicitly
    instead of silently falling back to the incompatible loop graph.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    recurrent_types = (keras.layers.GRU, keras.layers.LSTM)

    def collect_recurrent_layers(container):
        found = []
        for layer in getattr(container, "layers", []):
            if isinstance(layer, recurrent_types):
                found.append(layer)
            elif isinstance(layer, keras.layers.Bidirectional):
                raise ValueError("Bidirectional recurrent export is not supported")
            elif isinstance(layer, keras.Model):
                found.extend(collect_recurrent_layers(layer))
        return found

    recurrent_layers = collect_recurrent_layers(model)
    if not recurrent_layers:
        raise ValueError("Native recurrent export requires GRU or LSTM layers")

    cell_types = {type(layer) for layer in recurrent_layers}
    if len(cell_types) != 1:
        raise ValueError("Mixed GRU/LSTM stacks are not supported")
    cell_type = next(iter(cell_types))
    op_type = "GRU" if cell_type is keras.layers.GRU else "LSTM"

    input_shape = model.input_shape
    if len(input_shape) != 3 or any(dim is None for dim in input_shape[1:]):
        raise ValueError(f"Recurrent export requires a static sequence shape: {input_shape}")
    sequence_length, feature_count = map(int, input_shape[1:])

    try:
        encoder = model.get_layer("encoder")
    except ValueError as exc:
        raise ValueError("Recurrent export requires a named encoder model") from exc
    try:
        projection_head = model.get_layer("projection_head")
    except ValueError:
        projection_head = None
    try:
        decoder = model.get_layer("decoder")
    except ValueError:
        decoder = None
    if projection_head is None and decoder is None:
        raise ValueError(
            "Recurrent export requires either projection_head (CARLA detector) "
            "or decoder (autoencoder)"
        )

    nodes = []
    initializers = []

    def add_initializer(name, value, dtype=np.float32):
        initializers.append(
            numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)
        )

    def add_squeeze(source, output, axis, name):
        axes_name = f"{name}_axes"
        add_initializer(axes_name, [axis], np.int64)
        nodes.append(
            helper.make_node("Squeeze", [source, axes_name], [output], name=name)
        )

    def activation_name(activation):
        serialized = keras.activations.serialize(activation)
        if isinstance(serialized, dict):
            serialized = serialized.get("config", serialized.get("class_name", ""))
        return str(serialized).lower()

    def add_activation(source, activation, prefix):
        name = activation_name(activation)
        if name in {"linear", "none"}:
            return source
        operator = {
            "relu": "Relu",
            "sigmoid": "Sigmoid",
            "tanh": "Tanh",
            "elu": "Elu",
        }.get(name)
        if operator is None:
            raise ValueError(f"Unsupported recurrent-export activation: {name}")
        output = f"{prefix}_{name}"
        nodes.append(helper.make_node(operator, [source], [output], name=output))
        return output

    def add_dense(source, layer, prefix):
        if not isinstance(layer, keras.layers.Dense):
            raise ValueError(f"Expected Dense layer, got {type(layer).__name__}")
        weights = layer.get_weights()
        if len(weights) not in {1, 2}:
            raise ValueError(f"Unexpected Dense weights for {layer.name}")
        kernel_name = f"{prefix}_kernel"
        add_initializer(kernel_name, weights[0])
        matmul_output = f"{prefix}_matmul"
        nodes.append(
            helper.make_node(
                "MatMul", [source, kernel_name], [matmul_output], name=matmul_output
            )
        )
        output = matmul_output
        if len(weights) == 2:
            bias_name = f"{prefix}_bias_value"
            add_initializer(bias_name, weights[1])
            output = f"{prefix}_bias"
            nodes.append(
                helper.make_node(
                    "Add", [matmul_output, bias_name], [output], name=output
                )
            )
        return add_activation(output, layer.activation, prefix)

    def add_batch_normalization(source, layer, prefix):
        gamma, beta, moving_mean, moving_variance = layer.get_weights()
        scale = gamma / np.sqrt(moving_variance + float(layer.epsilon))
        offset = beta - moving_mean * scale
        scale_name = f"{prefix}_scale"
        offset_name = f"{prefix}_offset"
        add_initializer(scale_name, scale)
        add_initializer(offset_name, offset)
        scaled = f"{prefix}_scaled"
        output = f"{prefix}_output"
        nodes.append(helper.make_node("Mul", [source, scale_name], [scaled], name=scaled))
        nodes.append(helper.make_node("Add", [scaled, offset_name], [output], name=output))
        return output

    def add_recurrent(source, layer, prefix):
        if layer.go_backwards or layer.stateful or layer.unroll:
            raise ValueError(
                f"Unsupported recurrent configuration in {layer.name}: "
                "go_backwards/stateful/unroll must be false"
            )
        if activation_name(layer.activation) != "tanh" or activation_name(
            layer.recurrent_activation
        ) != "sigmoid":
            raise ValueError(
                f"{layer.name} requires the standard tanh/sigmoid activations"
            )

        kernel, recurrent_kernel, bias = layer.get_weights()
        units = int(layer.units)
        attributes = {"hidden_size": units, "direction": "forward"}

        if isinstance(layer, keras.layers.LSTM):
            # Keras gate order is [input, forget, cell, output], while ONNX uses
            # [input, output, forget, cell]. Keras stores one combined bias;
            # map it to the ONNX input bias and use a zero recurrent bias.
            gate_order = [0, 3, 1, 2]
            kernel_parts = np.split(kernel, 4, axis=1)
            recurrent_parts = np.split(recurrent_kernel, 4, axis=1)
            bias_parts = np.split(bias, 4)
            onnx_kernel = np.concatenate(
                [kernel_parts[index] for index in gate_order], axis=1
            ).T[None, ...]
            onnx_recurrent = np.concatenate(
                [recurrent_parts[index] for index in gate_order], axis=1
            ).T[None, ...]
            input_bias = np.concatenate([bias_parts[index] for index in gate_order])
            onnx_bias = np.concatenate([input_bias, np.zeros_like(input_bias)])[None, ...]
            outputs = [f"{prefix}_sequence", f"{prefix}_hidden", f"{prefix}_cell"]
        else:
            if not layer.reset_after:
                raise ValueError(f"{layer.name} requires GRU reset_after=True")
            # Keras and ONNX both use [update, reset, hidden] gate order. With
            # reset_after=True, Keras' two bias rows map directly to ONNX Wb/Rb
            # and ONNX linear_before_reset=1.
            onnx_kernel = kernel.T[None, ...]
            onnx_recurrent = recurrent_kernel.T[None, ...]
            if bias.ndim != 2 or bias.shape[0] != 2:
                raise ValueError(f"Unexpected reset-after GRU bias shape: {bias.shape}")
            onnx_bias = np.concatenate([bias[0], bias[1]])[None, ...]
            outputs = [f"{prefix}_sequence", f"{prefix}_hidden"]
            attributes["linear_before_reset"] = 1

        for suffix, value in (
            ("kernel", onnx_kernel),
            ("recurrent", onnx_recurrent),
            ("bias", onnx_bias),
        ):
            add_initializer(f"{prefix}_{suffix}", value)
        nodes.append(
            helper.make_node(
                op_type,
                [
                    source,
                    f"{prefix}_kernel",
                    f"{prefix}_recurrent",
                    f"{prefix}_bias",
                ],
                outputs,
                name=prefix,
                **attributes,
            )
        )
        output = f"{prefix}_output"
        add_squeeze(
            outputs[0] if layer.return_sequences else outputs[1],
            output,
            1 if layer.return_sequences else 0,
            f"{prefix}_squeeze",
        )
        return output, "sequence" if layer.return_sequences else "batch"

    def flattened_layers(container):
        for layer in container.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            if isinstance(layer, keras.Sequential):
                yield from flattened_layers(layer)
            elif hasattr(layer, "layers_list"):
                yield from layer.layers_list
            else:
                yield layer

    nodes.append(
        helper.make_node(
            "Transpose", ["input_1"], ["encoder_sequence"], perm=[1, 0, 2]
        )
    )
    value = "encoder_sequence"
    layout = "sequence"
    for index, layer in enumerate(flattened_layers(encoder)):
        prefix = f"encoder_{index}_{layer.name}"
        if isinstance(layer, recurrent_types):
            if layout != "sequence":
                raise ValueError("A recurrent encoder layer received batch-major data")
            value, layout = add_recurrent(value, layer, prefix)
        elif isinstance(layer, keras.layers.Dense):
            value = add_dense(value, layer, prefix)
        elif isinstance(layer, keras.layers.Activation):
            value = add_activation(value, layer.activation, prefix)
        elif isinstance(layer, (keras.layers.Dropout, keras.layers.Reshape, keras.layers.Flatten)):
            continue
        else:
            raise ValueError(f"Unsupported encoder layer: {type(layer).__name__}")

    if layout != "batch":
        raise ValueError("The recurrent encoder must end with return_sequences=False")

    if projection_head is not None:
        for index, layer in enumerate(flattened_layers(projection_head)):
            prefix = f"projection_{index}_{layer.name}"
            if isinstance(layer, keras.layers.Dense):
                value = add_dense(value, layer, prefix)
            elif isinstance(layer, keras.layers.BatchNormalization):
                value = add_batch_normalization(value, layer, prefix)
            elif isinstance(layer, keras.layers.Activation):
                value = add_activation(value, layer.activation, prefix)
            elif isinstance(layer, keras.layers.Dropout):
                continue
            else:
                raise ValueError(
                    f"Unsupported projection layer: {type(layer).__name__}"
                )
        nodes.append(helper.make_node("Identity", [value], ["output_1"], name="output"))
        embedding_dim = int(projection_head.output_shape[-1])
        graph = helper.make_graph(
            nodes,
            f"native_{op_type.lower()}_carla_detector",
            [
                helper.make_tensor_value_info(
                    "input_1", TensorProto.FLOAT, [1, sequence_length, feature_count]
                )
            ],
            [helper.make_tensor_value_info("output_1", TensorProto.FLOAT, [1, embedding_dim])],
            initializers,
        )
        onnx_model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", opset)],
            producer_name="fault_converters_native_recurrent_carla_export",
        )
        onnx_model.ir_version = 7
        onnx.checker.check_model(onnx_model)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(onnx_model, str(output))
        return

    repeated = False
    for index, layer in enumerate(flattened_layers(decoder)):
        prefix = f"decoder_{index}_{layer.name}"
        if isinstance(layer, keras.layers.RepeatVector):
            if repeated or int(layer.n) != sequence_length:
                raise ValueError("Decoder requires one full-length RepeatVector")
            axes_name = f"{prefix}_axes"
            repeats_name = f"{prefix}_repeats"
            add_initializer(axes_name, [0], np.int64)
            add_initializer(repeats_name, [sequence_length, 1, 1], np.int64)
            once = f"{prefix}_once"
            output = f"{prefix}_output"
            nodes.append(helper.make_node("Unsqueeze", [value, axes_name], [once]))
            nodes.append(helper.make_node("Tile", [once, repeats_name], [output]))
            value, layout, repeated = output, "sequence", True
        elif isinstance(layer, recurrent_types):
            if layout != "sequence":
                raise ValueError("A recurrent decoder layer received batch-major data")
            value, layout = add_recurrent(value, layer, prefix)
        elif isinstance(layer, keras.layers.Dense):
            value = add_dense(value, layer, prefix)
        elif isinstance(layer, keras.layers.TimeDistributed):
            value = add_dense(value, layer.layer, prefix)
        elif isinstance(layer, keras.layers.Activation):
            value = add_activation(value, layer.activation, prefix)
        elif isinstance(layer, (keras.layers.Dropout, keras.layers.Reshape, keras.layers.Flatten)):
            continue
        else:
            raise ValueError(f"Unsupported decoder layer: {type(layer).__name__}")

    if layout != "sequence":
        raise ValueError("The recurrent decoder must produce a sequence")
    nodes.append(
        helper.make_node(
            "Transpose", [value], ["output_1"], perm=[1, 0, 2], name="output_batch_major"
        )
    )
    graph = helper.make_graph(
        nodes,
        f"native_{op_type.lower()}_autoencoder",
        [
            helper.make_tensor_value_info(
                "input_1", TensorProto.FLOAT, [1, sequence_length, feature_count]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output_1", TensorProto.FLOAT, [1, sequence_length, feature_count]
            )
        ],
        initializers,
    )
    onnx_model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="fault_converters_native_recurrent_export",
    )
    # IR v7 is supported by the ONNX parser shipped with TensorRT 8.2.
    onnx_model.ir_version = 7
    onnx.checker.check_model(onnx_model)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(output))


def convert_keras_to_onnx(
    model: keras.Model,
    output_path: str,
    opset: int = 13,
) -> bool:
    """
    Convert a Keras model to ONNX format using tf2onnx.

    Args:
        model: Trained Keras model.
        output_path: Path where the output .onnx model should be written.
        opset: ONNX opset version. Defaults to 13, the newest opset fully
            supported by TensorRT 8.2 in JetPack 4.6.

    Returns:
        bool: True if conversion succeeded, False otherwise.
    """
    logger.info("Initiating Keras to ONNX conversion...")
    try:
        import tf2onnx
        import onnx
    except ImportError:
        logger.warning(
            "Required packages for ONNX conversion (tf2onnx, onnx) are not installed. "
            "Please install them via: pip install tf2onnx onnx"
        )
        return False

    try:
        def contains_recurrent_layer(layer):
            if isinstance(layer, (keras.layers.GRU, keras.layers.LSTM)):
                return True
            return any(
                contains_recurrent_layer(child)
                for child in getattr(layer, "layers", [])
                if child is not layer
            )

        if contains_recurrent_layer(model):
            logger.info(
                "Exporting recurrent model with native ONNX GRU/LSTM operators "
                "for TensorRT 8.2."
            )
            _convert_recurrent_keras_to_onnx(model, output_path, opset)
            logger.info(f"Successfully saved native recurrent ONNX model to {output_path}")
            return True

        # Keras MHA exports are mis-executed by TensorRT 8.2 on JetPack 4.6.
        # Substitute an exactly equivalent rank-4 matmul representation before
        # tracing, without altering the trained weights or the source model.
        export_model = model
        def contains_multi_head_attention(layer):
            if isinstance(layer, keras.layers.MultiHeadAttention):
                return True
            return any(
                contains_multi_head_attention(child)
                for child in getattr(layer, "layers", [])
                if child is not layer
            )

        if contains_multi_head_attention(model):
            logger.info(
                "Replacing Keras MultiHeadAttention with TensorRT-compatible "
                "static attention for export."
            )
            export_model = make_tensorrt_compatible_model(model)

        # Fix the batch dimension to 1 so the exported model is optimised for
        # single-sample edge inference (static input shape [1, seq_len, features]).
        input_shape = export_model.input_shape
        static_shape = (1,) + tuple(input_shape[1:])
        input_signature = [tf.TensorSpec(static_shape, tf.float32, name="input_1")]
        
        logger.info(f"Converting Keras model with fixed batch-1 input shape: {static_shape} and opset: {opset}...")
        
        # Convert model using tf2onnx from_keras
        onnx_model, _ = tf2onnx.convert.from_keras(
            export_model,
            input_signature=input_signature,
            opset=opset,
        )
        
        # Save ONNX model
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(onnx_model, str(out_file))
        logger.info(f"Successfully saved ONNX model to {out_file}")
        return True
    except Exception as e:
        logger.error(f"Keras to ONNX conversion failed with error: {e}")
        return False


def compile_onnx_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    precision_mode: str = "FP16",
    calibration_data: Optional[np.ndarray] = None,
) -> bool:
    """
    Compile an ONNX model into a standalone TensorRT engine.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Path where the output .engine file should be saved.
        precision_mode: "FP32", "FP16", or "INT8".
        calibration_data: Optional calibration data for INT8 quantization.

    Returns:
        bool: True if compilation succeeded, False otherwise.
    """
    logger.info(f"Initiating ONNX to TensorRT compilation (precision: {precision_mode})...")
    
    # (We rely on trt.BuilderFlag.FP16 natively instead of manual ONNX cast to prevent NaNs)
    # 1. Defensive Check: Check if TensorRT is available in Python
    try:
        import tensorrt as trt
    except ImportError:
        logger.warning(
            "The 'tensorrt' Python package is not installed. "
            "You can compile this ONNX model using NVIDIA's command-line tool 'trtexec' instead:\n"
            f"  trtexec --onnx={onnx_path} --saveEngine={engine_path} --{precision_mode.lower()}\n"
            "To install Python bindings, please refer to TensorRT installation documentation."
        )
        return False

    try:
        # 2. Build the TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        
        # Create network with explicit batch dimension
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        else:
            # In TensorRT 10.0+, explicit batch is the only mode and the flag is deprecated/removed.
            network_flags = 0
        network = builder.create_network(network_flags)
        
        # Parse ONNX file
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX Parser Error: {parser.get_error(error)}")
                return False
                
        # Configure the builder
        config = builder.create_builder_config()
        
        # We allow TRT to use available memory instead of restricting to 1GB to prevent OOM
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        # Configure optimization profile for dynamic shapes if any are present
        has_dynamic = False
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            if any(dim == -1 or dim is None for dim in input_tensor.shape):
                has_dynamic = True
                break

        if has_dynamic:
            logger.info("Dynamic input shapes detected. Creating optimization profile...")
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                shape = list(input_tensor.shape)
                name = input_tensor.name
                
                # Replace any dynamic dimensions with a fixed batch of 1, since
                # the models are exported and optimised for single-sample inference.
                min_shape = [1 if dim == -1 or dim is None else dim for dim in shape]
                opt_shape = [1 if dim == -1 or dim is None else dim for dim in shape]
                max_shape = [1 if dim == -1 or dim is None else dim for dim in shape]
                
                logger.info(f"Setting profile shape for input '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
                profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
        
        # Precision flags
        if precision_mode == "FP16":
            if hasattr(trt.BuilderFlag, "FP16"):
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled.")
            else:
                logger.info("FP16 builder flag not available in this TensorRT version (strongly typed network default). Pre-cast ONNX was loaded.")
        elif precision_mode == "INT8":
            if calibration_data is None or len(calibration_data) == 0:
                logger.error(
                    "INT8 requires representative calibration data; refusing to "
                    "create an uncalibrated engine."
                )
                return False

            import pycuda.driver as cuda
            import pycuda.autoprimaryctx  # noqa: F401

            calibration_array = np.ascontiguousarray(calibration_data, dtype=np.float32)
            cache_path = str(Path(engine_path).with_suffix(".calibration.cache"))

            class NumpyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
                """TensorRT 8.x entropy calibrator for fixed batch-1 inputs."""

                def __init__(self, data, cache_file):
                    trt.IInt8EntropyCalibrator2.__init__(self)
                    self.data = data
                    self.cache_file = cache_file
                    self.index = 0
                    self.device_input = cuda.mem_alloc(self.data[0].nbytes)

                def get_batch_size(self):
                    return 1

                def get_batch(self, names):
                    if self.index >= len(self.data):
                        return None
                    sample = np.ascontiguousarray(self.data[self.index : self.index + 1])
                    cuda.memcpy_htod(self.device_input, sample)
                    self.index += 1
                    return [int(self.device_input)]

                def read_calibration_cache(self):
                    path = Path(self.cache_file)
                    return path.read_bytes() if path.exists() else None

                def write_calibration_cache(self, cache):
                    Path(self.cache_file).write_bytes(cache)

            calibrator = NumpyEntropyCalibrator(calibration_array, cache_path)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            logger.info(
                "INT8 entropy calibration enabled with %d representative samples.",
                len(calibration_array),
            )
        elif precision_mode == "FP32":
            logger.info("FP32 precision selected. No extra flags needed.")

        # 4. Serialize and save the engine
        logger.info("Building and serializing TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to build serialized network engine.")
            return False
            
        out_engine = Path(engine_path)
        out_engine.parent.mkdir(parents=True, exist_ok=True)
        with open(out_engine, "wb") as f:
            f.write(serialized_engine)
            
        logger.info(f"Successfully saved TensorRT engine to {out_engine}")
        return True
    except Exception as e:
        logger.error(f"TensorRT compilation failed with error: {e}")
        return False
