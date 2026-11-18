"""
Inference Runner Factories
==========================

Provides reusable callable runner factories for each model format
(Keras, TFLite, ONNX, TensorRT). These runners encapsulate the
format-specific setup (session loading, buffer allocation, quant/dequant)
behind a single ``Callable[[np.ndarray], np.ndarray]`` interface.

Used by the benchmark, evaluation, and performance modules to avoid
duplicating inference code per format.
"""

import time
from pathlib import Path
import numpy as np
from typing import Optional, Callable, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _prepare_input(data: np.ndarray) -> np.ndarray:
    """Return the canonical input representation used by every backend."""
    array = np.asarray(data, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("Inference input must contain at least one sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Inference input contains NaN or Inf values.")
    return np.ascontiguousarray(array)


def _validate_output(
    input_data: np.ndarray,
    output: np.ndarray,
    expected_sample_shape=None,
) -> np.ndarray:
    """Validate the common runner contract for reconstruction or embeddings."""
    array = np.asarray(output)
    if array.ndim < 1 or array.shape[0] != input_data.shape[0]:
        raise ValueError(
            f"Runner output batch shape {array.shape} is incompatible with input "
            f"batch shape {input_data.shape}."
        )
    if expected_sample_shape is not None and tuple(array.shape[1:]) != tuple(
        expected_sample_shape
    ):
        raise ValueError(
            f"Runner output sample shape {array.shape[1:]} does not match expected "
            f"shape {tuple(expected_sample_shape)}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Runner output contains NaN or Inf values.")
    return array.astype(np.float32, copy=False)


def run_inference_once(
    runner: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
) -> np.ndarray:
    """Run one backend through the canonical input/output contract."""
    input_data = _prepare_input(data)
    expected_shape = getattr(runner, "output_shape", input_data.shape[1:])
    return _validate_output(input_data, runner(input_data), expected_shape)


def measure_inference_latency(
    runner: Callable[[np.ndarray], np.ndarray],
    input_data: np.ndarray,
    num_warmup: int,
    num_runs: int,
) -> Dict[str, Any]:
    """Measure every backend with one identical, synchronous timing loop.

    Runners must return a host NumPy array.  GPU runners therefore synchronize
    before returning, which prevents asynchronous work from escaping the timed
    region. Input conversion and validation happen once before timing so they
    do not unfairly dominate fast backends.
    """
    if num_warmup < 0 or num_runs < 1:
        raise ValueError("num_warmup must be >= 0 and num_runs must be >= 1.")

    sample = _prepare_input(input_data)
    expected_shape = getattr(runner, "output_shape", sample.shape[1:])
    _validate_output(sample, runner(sample), expected_shape)

    for _ in range(num_warmup):
        runner(sample)

    end_to_end_latencies = np.empty(num_runs, dtype=np.float64)
    device_latencies = np.empty(num_runs, dtype=np.float64)
    cuda_timed_call = getattr(runner, "cuda_timed_call", None)
    for index in range(num_runs):
        started = time.perf_counter()
        if cuda_timed_call is None:
            output = runner(sample)
        else:
            output, device_latencies[index] = cuda_timed_call(sample)
        end_to_end_latencies[index] = (time.perf_counter() - started) * 1000.0

    _validate_output(sample, output, expected_shape)
    primary_latencies = (
        device_latencies if cuda_timed_call is not None else end_to_end_latencies
    )
    return {
        "latency_mean_ms": float(np.mean(primary_latencies)),
        "latency_std_ms": float(np.std(primary_latencies, ddof=1 if num_runs > 1 else 0)),
        "latency_min_ms": float(np.min(primary_latencies)),
        "latency_max_ms": float(np.max(primary_latencies)),
        "timing_source": "cuda_event" if cuda_timed_call is not None else "perf_counter",
        "timing_scope": (
            getattr(runner, "cuda_timing_scope", "device_execution")
            if cuda_timed_call is not None else "synchronous_model_call"
        ),
        "end_to_end_latency_mean_ms": float(np.mean(end_to_end_latencies)),
        "end_to_end_latency_std_ms": float(
            np.std(end_to_end_latencies, ddof=1 if num_runs > 1 else 0)
        ),
        "end_to_end_latency_min_ms": float(np.min(end_to_end_latencies)),
        "end_to_end_latency_max_ms": float(np.max(end_to_end_latencies)),
        "timing_batch_size": int(sample.shape[0]),
        "timing_warmup_runs": int(num_warmup),
        "timing_runs": int(num_runs),
    }


def create_keras_runner(
    model: Any,
    enable_cuda_timing: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a callable that runs inference on a Keras model.

    Args:
        model: Compiled Keras model.

    Returns:
        Callable accepting an input array and returning predictions.
    """
    def runner(x: np.ndarray) -> np.ndarray:
        return model(np.asarray(x, dtype=np.float32), training=False).numpy()

    runner.output_shape = tuple(model.output_shape[1:])

    if enable_cuda_timing:
        try:
            import tensorflow as tf

            if tf.config.list_physical_devices("GPU"):
                try:
                    from cuda import cudart
                except ImportError:
                    try:
                        from cuda.bindings import runtime as cudart
                    except ImportError:
                        import pycuda.driver as cuda
                        import pycuda.autoprimaryctx  # noqa: F401

                        start_event = cuda.Event()
                        end_event = cuda.Event()

                        def cuda_timed_call(x: np.ndarray):
                            tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                            cuda.Context.synchronize()
                            start_event.record()
                            start_event.synchronize()
                            output_tensor = model(tensor, training=False)
                            # TensorFlow owns non-default CUDA streams. A device-wide
                            # barrier is required before recording the stop event on
                            # the PyCUDA stream; otherwise the interval can undercount
                            # asynchronous TensorFlow work.
                            cuda.Context.synchronize()
                            end_event.record()
                            end_event.synchronize()
                            output = output_tensor.numpy()
                            return output, float(start_event.time_till(end_event))
                    else:
                        _, start_event = cudart.cudaEventCreate()
                        _, end_event = cudart.cudaEventCreate()

                        def cuda_timed_call(x: np.ndarray):
                            tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                            cudart.cudaDeviceSynchronize()
                            cudart.cudaEventRecord(start_event, 0)
                            cudart.cudaEventSynchronize(start_event)
                            output_tensor = model(tensor, training=False)
                            cudart.cudaDeviceSynchronize()
                            cudart.cudaEventRecord(end_event, 0)
                            cudart.cudaEventSynchronize(end_event)
                            _, elapsed_ms = cudart.cudaEventElapsedTime(
                                start_event, end_event
                            )
                            output = output_tensor.numpy()
                            return output, float(elapsed_ms)
                else:
                    _, start_event = cudart.cudaEventCreate()
                    _, end_event = cudart.cudaEventCreate()

                    def cuda_timed_call(x: np.ndarray):
                        tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                        cudart.cudaDeviceSynchronize()
                        cudart.cudaEventRecord(start_event, 0)
                        cudart.cudaEventSynchronize(start_event)
                        output_tensor = model(tensor, training=False)
                        cudart.cudaDeviceSynchronize()
                        cudart.cudaEventRecord(end_event, 0)
                        cudart.cudaEventSynchronize(end_event)
                        _, elapsed_ms = cudart.cudaEventElapsedTime(
                            start_event, end_event
                        )
                        output = output_tensor.numpy()
                        return output, float(elapsed_ms)

                runner.cuda_timed_call = cuda_timed_call
                runner.cuda_timing_scope = "device_synchronized_tensorflow_model_call"
        except Exception as error:
            logger.debug("CUDA-event timing unavailable for Keras runner: %s", error)
    return runner


def create_tflite_runner(tflite_path: str) -> Callable[[np.ndarray], np.ndarray]:
    """Create a callable that runs inference on a TFLite model.

    Handles INT8 quantization/dequantization transparently.

    Args:
        tflite_path: Path to a .tflite model file.

    Returns:
        Callable accepting a batch array and returning float32 predictions.
        Supports dynamic batch sizes by resizing the interpreter internally.
    """
    # Keep TensorFlow out of TensorRT-only processes.  This matters on Jetson,
    # where CPU and GPU share a small LPDDR pool and importing TensorFlow can
    # reserve hundreds of megabytes before the engine is even deserialized.
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]

    is_int8_input = input_details["dtype"] == np.int8
    input_quant = input_details.get("quantization", (1.0, 0))
    is_int8_output = output_details["dtype"] == np.int8
    output_quant = output_details.get("quantization", (1.0, 0))

    # Track the current allocated batch size to avoid unnecessary resizes
    _current_batch_size = [input_details["shape"][0]]

    def runner(x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]

        # Resize interpreter if batch size changed
        if batch_size != _current_batch_size[0]:
            new_shape = list(input_details["shape"])
            new_shape[0] = batch_size
            interpreter.resize_tensor_input(input_index, new_shape)
            interpreter.allocate_tensors()
            _current_batch_size[0] = batch_size

        # Prepare input (handle INT8 quantization with [-128, 127] saturation)
        if is_int8_input:
            scale, zero_point = input_quant
            quantized = np.round(x.astype(np.float32) / scale + zero_point)
            x_prep = np.clip(quantized, -128, 127).astype(np.int8)
        else:
            x_prep = np.asarray(x, dtype=np.float32)

        interpreter.set_tensor(input_index, x_prep)
        interpreter.invoke()
        out_raw = interpreter.get_tensor(output_index)

        # Dequantize output if needed
        if is_int8_output:
            out_scale, out_zero_point = output_quant
            return (out_raw.astype(np.float32) - out_zero_point) * out_scale
        return out_raw.astype(np.float32)

    runner.output_shape = tuple(int(value) for value in output_details["shape"][1:])
    return runner


def create_onnx_runner(onnx_path: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Create a callable that runs inference on an ONNX model via ORT.

    Args:
        onnx_path: Path to a .onnx model file.

    Returns:
        Callable accepting a batch array and returning predictions,
        or None if onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime is not installed. Cannot create ONNX runner.")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def runner(x: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: np.asarray(x, dtype=np.float32)})[0]

    output_shape = session.get_outputs()[0].shape[1:]
    runner.output_shape = (
        tuple(int(value) for value in output_shape)
        if all(isinstance(value, int) and value > 0 for value in output_shape)
        else None
    )
    return runner


def create_tensorrt_runner(
    engine_path: str,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Create a callable that runs inference on a TensorRT engine.

    Allocates CUDA device buffers matching the engine's I/O bindings
    and runs synchronous inference.

    Args:
        engine_path: Path to a serialized .engine file.

    Returns:
        Callable accepting a batch array (float32) and returning predictions,
        or None if tensorrt/pycuda is not available.
    """
    engine_file = Path(engine_path)
    if not engine_file.is_file() or engine_file.stat().st_size == 0:
        logger.error(f"TensorRT engine file is missing or empty: {engine_path}")
        return None

    try:
        import tensorrt as trt
    except ImportError:
        logger.warning(
            "The 'tensorrt' Python package is not installed. Cannot create TensorRT runner."
        )
        return None

    try:
        from cuda import cudart
    except ImportError:
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            try:
                import pycuda.driver as cuda
                # Share CUDA's primary context with TensorFlow/TensorRT. The
                # separate context created by pycuda.autoinit can crash during
                # teardown on JetPack 4.6 when several engines were loaded.
                import pycuda.autoprimaryctx  # noqa: F401
                _use_pycuda = True
            except ImportError:
                logger.warning(
                    "Neither 'cuda-python' nor 'pycuda' is installed. "
                    "Cannot create TensorRT runner (CUDA memory management required)."
                )
                return None
        else:
            _use_pycuda = False
    else:
        _use_pycuda = False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    try:
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    except FileNotFoundError:
        logger.error(f"TensorRT engine file not found: {engine_path}")
        return None

    if engine is None:
        logger.error(f"Failed to deserialize TensorRT engine from {engine_path}")
        return None

    context = engine.create_execution_context()
    if context is None:
        logger.error(f"Failed to create TensorRT execution context for {engine_path}")
        return None

    # TensorRT 10 uses named I/O tensors. JetPack 4.6 ships TensorRT 8.2,
    # which uses binding indices. Resolve both into one internal interface.
    modern_api = hasattr(engine, "num_io_tensors")
    input_name = output_name = None
    input_index = output_index = None

    if modern_api:
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and input_name is None:
                input_name = name
            elif mode == trt.TensorIOMode.OUTPUT and output_name is None:
                output_name = name
        if input_name is not None:
            input_index = engine.get_binding_index(input_name) if hasattr(engine, "get_binding_index") else None
        if output_name is not None:
            output_index = engine.get_binding_index(output_name) if hasattr(engine, "get_binding_index") else None
        output_dtype = trt.nptype(engine.get_tensor_dtype(output_name)) if output_name else np.float32
    else:
        for index in range(engine.num_bindings):
            if engine.binding_is_input(index) and input_index is None:
                input_index = index
                input_name = engine.get_binding_name(index)
            elif not engine.binding_is_input(index) and output_index is None:
                output_index = index
                output_name = engine.get_binding_name(index)
        output_dtype = trt.nptype(engine.get_binding_dtype(output_index)) if output_index is not None else np.float32

    if input_name is None or output_name is None:
        logger.error("Could not identify TensorRT input/output tensors.")
        return None

    buffer_cache = {"shape": None, "input": None, "output": None, "output_shape": None}

    if _use_pycuda:
        stream = cuda.Stream()
        start_event = cuda.Event()
        end_event = cuda.Event()
    else:
        stream = 0
        _, start_event = cudart.cudaEventCreate()
        _, end_event = cudart.cudaEventCreate()

    def _set_shapes(input_shape):
        if modern_api:
            if not context.set_input_shape(input_name, input_shape):
                raise RuntimeError(f"TensorRT rejected input shape {input_shape}")
            shape = tuple(context.get_tensor_shape(output_name))
        else:
            engine_shape = tuple(engine.get_binding_shape(input_index))
            if any(dimension < 0 for dimension in engine_shape):
                if not context.set_binding_shape(input_index, input_shape):
                    raise RuntimeError(f"TensorRT rejected input shape {input_shape}")
            shape = tuple(context.get_binding_shape(output_index))
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(f"TensorRT output shape is unresolved: {shape}")
        return shape

    def _ensure_buffers(input_array):
        output_shape = _set_shapes(tuple(input_array.shape))
        if buffer_cache["shape"] == tuple(input_array.shape):
            return np.empty(output_shape, dtype=output_dtype)

        if _use_pycuda:
            if buffer_cache["input"] is not None:
                buffer_cache["input"].free()
                buffer_cache["output"].free()
            buffer_cache["input"] = cuda.mem_alloc(input_array.nbytes)
            output_array = np.empty(output_shape, dtype=output_dtype)
            buffer_cache["output"] = cuda.mem_alloc(output_array.nbytes)
        else:
            if buffer_cache["input"] is not None:
                cudart.cudaFree(buffer_cache["input"])
                cudart.cudaFree(buffer_cache["output"])
            _, buffer_cache["input"] = cudart.cudaMalloc(input_array.nbytes)
            output_array = np.empty(output_shape, dtype=output_dtype)
            _, buffer_cache["output"] = cudart.cudaMalloc(output_array.nbytes)

        buffer_cache["shape"] = tuple(input_array.shape)
        buffer_cache["output_shape"] = output_shape
        return output_array

    def _launch():
        if modern_api:
            context.set_tensor_address(input_name, int(buffer_cache["input"]))
            context.set_tensor_address(output_name, int(buffer_cache["output"]))
            return context.execute_async_v3(stream_handle=int(stream.handle) if _use_pycuda else stream)
        bindings = [0] * engine.num_bindings
        bindings[input_index] = int(buffer_cache["input"])
        bindings[output_index] = int(buffer_cache["output"])
        return context.execute_async_v2(
            bindings=bindings,
            stream_handle=int(stream.handle) if _use_pycuda else stream,
        )

    def _execute(x: np.ndarray, cuda_timed: bool = False):
        input_array = np.ascontiguousarray(x, dtype=np.float32)
        output_array = _ensure_buffers(input_array)

        if _use_pycuda:
            cuda.memcpy_htod(buffer_cache["input"], input_array)
            if cuda_timed:
                start_event.record(stream)
            if not _launch():
                raise RuntimeError("TensorRT execution failed.")
            if cuda_timed:
                end_event.record(stream)
                end_event.synchronize()
                device_ms = float(start_event.time_till(end_event))
            else:
                stream.synchronize()
                device_ms = None
            cuda.memcpy_dtoh(output_array, buffer_cache["output"])
        else:
            cudart.cudaMemcpy(
                buffer_cache["input"], input_array.ctypes.data, input_array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            if cuda_timed:
                cudart.cudaEventRecord(start_event, stream)
            if not _launch():
                raise RuntimeError("TensorRT execution failed.")
            if cuda_timed:
                cudart.cudaEventRecord(end_event, stream)
                cudart.cudaEventSynchronize(end_event)
                _, device_ms = cudart.cudaEventElapsedTime(start_event, end_event)
                device_ms = float(device_ms)
            else:
                cudart.cudaDeviceSynchronize()
                device_ms = None
            cudart.cudaMemcpy(
                output_array.ctypes.data, buffer_cache["output"], output_array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )

        return output_array, device_ms

    def runner(x: np.ndarray) -> np.ndarray:
        return _execute(x, cuda_timed=False)[0]

    def cuda_timed_call(x: np.ndarray):
        return _execute(x, cuda_timed=True)

    # The shared protocol detects this hook and records CUDA-event device time
    # plus end-to-end perf_counter time from the same invocation.
    runner.cuda_timed_call = cuda_timed_call
    runner.cuda_timing_scope = "tensorrt_engine_execution"
    runner.output_shape = None
    return runner


def run_inference_loop(
    runner: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    batch_size: int = 1,
) -> np.ndarray:
    """Run inference across a full dataset using a runner callable.

    Splits data into batches to keep memory usage bounded.

    Args:
        runner: Callable produced by one of the ``create_*_runner`` factories.
        data: Full dataset array of shape ``(N, seq_len, features)``.
        batch_size: Maximum batch size per runner call.

    Returns:
        Concatenated backend outputs with the sample axis preserved.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    input_data = _prepare_input(data)
    results = []
    for i in range(0, len(input_data), batch_size):
        batch = input_data[i : i + batch_size]
        results.append(run_inference_once(runner, batch))
    return np.concatenate(results, axis=0)


def compare_runner_outputs(
    reference_runner: Callable[[np.ndarray], np.ndarray],
    candidate_runner: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    threshold: Optional[float] = None,
    batch_size: int = 1,
    max_samples: int = 32,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    score_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Quantify backend-output, anomaly-score and decision drift."""
    sample_count = min(len(data), max_samples)
    if sample_count < 1:
        raise ValueError("Consistency comparison requires at least one sample.")
    sample = _prepare_input(data[:sample_count])
    reference = run_inference_loop(reference_runner, sample, batch_size=batch_size)
    candidate = run_inference_loop(candidate_runner, sample, batch_size=batch_size)

    return compare_output_arrays(
        reference,
        candidate,
        sample,
        threshold=threshold,
        atol=atol,
        rtol=rtol,
        score_function=score_function,
    )


def compare_output_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    input_data: np.ndarray,
    threshold: Optional[float] = None,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    score_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Quantify drift for already-computed outputs under the common contract."""
    sample = _prepare_input(input_data)
    reference = _validate_output(sample, reference)
    candidate = _validate_output(sample, candidate, reference.shape[1:])

    difference = candidate - reference
    reference_norm = float(np.linalg.norm(reference.ravel()))
    result = {
        "consistency_samples": int(len(sample)),
        "output_mse_vs_baseline": float(np.mean(np.square(difference))),
        "output_mean_abs_error_vs_baseline": float(np.mean(np.abs(difference))),
        "output_max_abs_error_vs_baseline": float(np.max(np.abs(difference))),
        "output_relative_l2_vs_baseline": float(
            np.linalg.norm(difference.ravel()) / max(reference_norm, np.finfo(np.float32).eps)
        ),
        "output_within_tolerance": bool(
            np.allclose(candidate, reference, atol=atol, rtol=rtol)
        ),
    }
    if score_function is None:
        # Backward-compatible names for genuine reconstruction backends only.
        result.update(
            {
                "reconstruction_mse_vs_baseline": result["output_mse_vs_baseline"],
                "reconstruction_mean_abs_error_vs_baseline": result[
                    "output_mean_abs_error_vs_baseline"
                ],
                "reconstruction_max_abs_error_vs_baseline": result[
                    "output_max_abs_error_vs_baseline"
                ],
                "reconstruction_relative_l2_vs_baseline": result[
                    "output_relative_l2_vs_baseline"
                ],
                "reconstruction_within_tolerance": result[
                    "output_within_tolerance"
                ],
            }
        )

    if threshold is not None:
        if score_function is None:
            reference_scores = np.mean(np.square(sample - reference), axis=(1, 2))
            candidate_scores = np.mean(np.square(sample - candidate), axis=(1, 2))
        else:
            reference_scores = score_function(reference)
            candidate_scores = score_function(candidate)
        reference_predictions = reference_scores > threshold
        candidate_predictions = candidate_scores > threshold
        result["score_mean_abs_error_vs_baseline"] = float(
            np.mean(np.abs(candidate_scores - reference_scores))
        )
        result["score_max_abs_error_vs_baseline"] = float(
            np.max(np.abs(candidate_scores - reference_scores))
        )
        result["anomaly_disagreement_rate_vs_baseline"] = float(
            np.mean(reference_predictions != candidate_predictions)
        )

    return result
