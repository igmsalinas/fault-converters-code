"""Deployment and quantization toolkit with lazy backend imports.

TensorRT-only processes must not pay TensorFlow/Keras memory and CUDA-context
costs on a small unified-memory edge device.  These wrappers preserve the
historical package-level API while importing each implementation only on use.
"""


def load_deployment_datasets(*args, **kwargs):
    from .utils import load_deployment_datasets as implementation
    return implementation(*args, **kwargs)


def get_calibration_dataset(*args, **kwargs):
    from .utils import get_calibration_dataset as implementation
    return implementation(*args, **kwargs)


def get_file_size_mb(*args, **kwargs):
    from .utils import get_file_size_mb as implementation
    return implementation(*args, **kwargs)


def quantize_keras_native(*args, **kwargs):
    from .keras_quantization import quantize_keras_native as implementation
    return implementation(*args, **kwargs)


def convert_to_tflite(*args, **kwargs):
    from .keras_quantization import convert_to_tflite as implementation
    return implementation(*args, **kwargs)


def convert_keras_to_onnx(*args, **kwargs):
    from .onnx_trt import convert_keras_to_onnx as implementation
    return implementation(*args, **kwargs)


def compile_onnx_to_tensorrt(*args, **kwargs):
    from .onnx_trt import compile_onnx_to_tensorrt as implementation
    return implementation(*args, **kwargs)


def quantize_for_vitis_ai(*args, **kwargs):
    from .vitis_ai import quantize_for_vitis_ai as implementation
    return implementation(*args, **kwargs)


def create_keras_runner(*args, **kwargs):
    from .runners import create_keras_runner as implementation
    return implementation(*args, **kwargs)


def create_tflite_runner(*args, **kwargs):
    from .runners import create_tflite_runner as implementation
    return implementation(*args, **kwargs)


def create_onnx_runner(*args, **kwargs):
    from .runners import create_onnx_runner as implementation
    return implementation(*args, **kwargs)


def create_tensorrt_runner(*args, **kwargs):
    from .runners import create_tensorrt_runner as implementation
    return implementation(*args, **kwargs)


def run_inference_loop(*args, **kwargs):
    from .runners import run_inference_loop as implementation
    return implementation(*args, **kwargs)


def run_full_performance_suite(*args, **kwargs):
    from .timing_memory_benchmark import run_full_performance_suite as implementation
    return implementation(*args, **kwargs)


def save_performance_report(*args, **kwargs):
    from .timing_memory_benchmark import save_performance_report as implementation
    return implementation(*args, **kwargs)


def run_deployment_evaluations(*args, **kwargs):
    from .timing_evaluation import run_deployment_evaluations as implementation
    return implementation(*args, **kwargs)


def save_evaluation_report(*args, **kwargs):
    from .timing_evaluation import save_evaluation_report as implementation
    return implementation(*args, **kwargs)


__all__ = [
    "load_deployment_datasets", "get_calibration_dataset", "get_file_size_mb",
    "quantize_keras_native", "convert_to_tflite", "convert_keras_to_onnx",
    "compile_onnx_to_tensorrt", "quantize_for_vitis_ai", "create_keras_runner",
    "create_tflite_runner", "create_onnx_runner", "create_tensorrt_runner",
    "run_inference_loop", "run_full_performance_suite", "save_performance_report",
    "run_deployment_evaluations", "save_evaluation_report",
]
