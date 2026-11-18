"""
Power Converter Fault Anomaly Detection Visualization Suite
===========================================================

Modular visualization library supporting:
- Publication and slide deck styling with adaptive luminance contrast
- Circuit schematics (Buck converter, etc.)
- Frequency response and Bode deviation curves
- Deep learning neural architecture diagrams (CARLA, Conv1D, GRU, LSTM, Transformer, VAE)
- Hardware deployment and benchmark heatmaps (Jetson Nano, TensorRT, TFLite, ONNX, Keras GPU)
- Anomaly detection performance and ablation visualizations
"""

from .style import (
    apply_publication_style,
    apply_deck_style,
    adaptive_contrast,
    style_axis,
    save_figure,
    PUBLICATION_PALETTE,
    RUNTIME_COLORS,
    PAPER,
    INK,
    MUTED,
)

from .schematic import draw_buck_converter_schematic

from .frequency_response import (
    plot_bode_deviation,
    plot_paper_bode_degradation,
    plot_transfer_function_comparison,
)

from .architecture import (
    plot_carla_architecture,
    plot_hero_autoencoder,
    HERO_MODELS,
)

from .deployment import (
    plot_precision_matrix,
    plot_latency_comparison,
    plot_optimization_fidelity,
    plot_resource_heatmaps,
    plot_power_energy,
    plot_memory_footprint,
    plot_efficiency_throughput,
    plot_thermal_utilization,
    plot_native_gpu_precision,
    plot_speedup_fidelity,
    plot_backend_compatibility,
    RESOURCE_BACKENDS,
)

from .metrics import (
    plot_carla_metrics_comparison,
    plot_negative_sampler_ablation,
)

from .plotter import (
    load_transfer_function,
    plot_amplitude,
    plot_phase,
    plot_transfer_function,
    compare_transfer_functions,
    create_comparison_plot,
)

from .paper_tables import (
    TableSpec,
    PublicationTableExporter,
    build_variable_load_paper_tables,
    build_jsc_extension_tables,
)

from .variable_load_figures import (
    plot_fig01_system_and_carla_framework,
    plot_fig02_physics_ssa_fidelity_and_bode,
    plot_fig03_carla_transformer_architecture,
    plot_fig04_ds01_vs_ds02_and_fault_recall,
    plot_fig05_pr_curves_and_10seed_ablation,
    plot_fig06_encoder_backbones_and_scorer_matrix,
    plot_fig07_pca_cliff_and_7level_fra_noise,
    plot_fig08_learning_curves_and_lofo_generalization,
    plot_fig09_attention_and_jacobian_saliency,
)

from .jsc_extension_figures import (
    plot_jsc_fig01_int8_recalibration_rescue,
    plot_jsc_fig02_prototype_scorer_pareto,
)

__all__ = [
    # Styling
    "apply_publication_style",
    "apply_deck_style",
    "adaptive_contrast",
    "style_axis",
    "save_figure",
    "PUBLICATION_PALETTE",
    "RUNTIME_COLORS",
    "PAPER",
    "INK",
    "MUTED",
    # Schematic
    "draw_buck_converter_schematic",
    # Frequency Response
    "plot_bode_deviation",
    "plot_transfer_function_comparison",
    # Architecture
    "plot_carla_architecture",
    "plot_hero_autoencoder",
    "HERO_MODELS",
    # Deployment
    "plot_precision_matrix",
    "plot_latency_comparison",
    "plot_resource_heatmaps",
    "plot_thermal_utilization",
    "plot_native_gpu_precision",
    "plot_speedup_fidelity",
    "plot_backend_compatibility",
    "RESOURCE_BACKENDS",
    # Metrics
    "plot_carla_metrics_comparison",
    "plot_negative_sampler_ablation",
    # Plotter
    "load_transfer_function",
    "plot_amplitude",
    "plot_phase",
    "plot_transfer_function",
    "compare_transfer_functions",
    "create_comparison_plot",
    # Publication Tables & Variable-Load / JSC Extension Figures
    "TableSpec",
    "PublicationTableExporter",
    "build_variable_load_paper_tables",
    "build_jsc_extension_tables",
    "plot_fig01_system_and_carla_framework",
    "plot_fig02_physics_ssa_fidelity_and_bode",
    "plot_fig03_carla_transformer_architecture",
    "plot_fig04_ds01_vs_ds02_and_fault_recall",
    "plot_fig05_pr_curves_and_10seed_ablation",
    "plot_fig06_encoder_backbones_and_scorer_matrix",
    "plot_fig07_pca_cliff_and_7level_fra_noise",
    "plot_fig08_learning_curves_and_lofo_generalization",
    "plot_fig09_attention_and_jacobian_saliency",
    "plot_jsc_fig01_int8_recalibration_rescue",
    "plot_jsc_fig02_prototype_scorer_pareto",
]
