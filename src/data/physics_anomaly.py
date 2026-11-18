"""
Physics-Based Anomaly Injection
===============================

Generates CARLA negative (anomalous) samples with a component-perturbed analytic
small-signal model rather than heuristic signal-space distortions.

Rationale
---------
The CARLA encoder consumes the **complex frequency response** ``H(jw)`` of the
converter (the preprocessor emits normalized ``Real``/``Imag`` channels by
default, ``use_real_imag=True``). A component fault is therefore most faithfully
modelled in the ``s``-plane: a capacitor ESR increase drags the ESR zero towards
the origin, a capacitance drop pushes the LC resonance up, etc.

Instead of reconstructing an *absolute* Bode plot from the perturbed model (which
would discard every per-sample characteristic and collapse the contrastive
negatives), we apply the fault as a **multiplicative transfer ratio**::

    G(jw) = H_fault(jw) / H_nominal(jw)
    H_out(jw) = H_sample(jw) * G(jw)

``G`` is ``~1`` where the fault has no effect and warps magnitude *and* phase
coherently around the affected poles/zeros. Each perturbed sample keeps its own
identity (a *hard* negative) while carrying a physics-informed, magnitude/phase-coupled
degradation warp. This is an analytic augmentation around a fixed nominal model, not a
per-sample circuit re-simulation.

Design
------
Converters are modelled as a polymorphic class hierarchy:

- :class:`PowerConverter` (abstract) — holds the component values and declares the
  two responsibilities every converter must define: its **own transfer function**
  (:meth:`~PowerConverter.transfer_function`) and its **degradable components**
  (:attr:`~PowerConverter.fault_modes`). Concrete subclasses
  (:class:`BuckConverter`, :class:`BoostConverter`, :class:`BuckBoostConverter`)
  implement the topology-specific control-to-output model. New converters —
  including higher-order ones (Cuk, SEPIC, flyback, resonant) — are added simply
  by subclassing; each auto-registers by ``name`` (Open/Closed).
- :class:`FaultMode` — a degradable component + the multiplicative range its value
  can drift over. A fault is applied at the **component level**: scale the
  component, ask the converter to rebuild its analytic transfer function, done. The
  damping and zero shifts therefore follow the chosen model without root-identification
  heuristics, and the approach extends to any converter order represented by a subclass.

The second-order PWM converters share the canonical control-to-output form
(Erickson & Maksimovic, *Fundamentals of Power Electronics*, 2nd ed., Table
8.2)::

    Gvd(s) = Gd0 * (1 + s/wz_esr) * (1 - s/wz_rhp)
                 / (1 + s/(Q*w0) + (s/w0)^2)

with topology-specific ``Gd0``, ``w0``, ``Q`` and (for boost / buck-boost) an
extra right-half-plane zero ``wz_rhp``.

Reference:
    Darban et al., "CARLA: Self-supervised Contrastive Representation Learning
    for Time Series Anomaly Detection", arXiv:2308.09296
    Erickson & Maksimovic, "Fundamentals of Power Electronics", 2nd ed., ch. 8.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import scipy.signal as signal

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Object
# ---------------------------------------------------------------------------
@dataclass
class TransferFunctionModel:
    """Abstract ``s``-plane representation of the plant (zeros, poles, gain)."""

    zeros: np.ndarray
    poles: np.ndarray
    gain: float

    def frequency_response(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate ``H(jw)`` on an angular-frequency grid (rad/s)."""
        _, h = signal.freqs_zpk(self.zeros, self.poles, self.gain, worN=omega)
        return h


# ---------------------------------------------------------------------------
# Fault modes (degradable components)
# ---------------------------------------------------------------------------
class PhysicsAnomalyType(str, Enum):
    """Physically grounded converter degradation modes (topology-agnostic)."""

    CAPACITOR_ESR = "capacitor_esr"  # Output-cap ESR increase (ageing/dry-out)
    CAPACITANCE_DROP = "capacitance_drop"  # Output-cap capacitance loss
    INDUCTOR_SATURATION = "inductor_saturation"  # Effective inductance drop
    INDUCTOR_ESR = "inductor_esr"  # Winding/DCR increase
    SWITCH_DEGRADATION = "switch_degradation"  # Rds_on / conduction loss increase


@dataclass(frozen=True)
class FaultMode:
    """
    A degradable component and the multiplicative range its value drifts over.

    A fault is applied by scaling ``component`` by a sampled ``severity`` and
    rebuilding the converter's transfer function, e.g. ``capacitor_esr`` scales
    ``esr_cap`` by 2..8x, ``capacitance_drop`` scales ``capacitance`` by
    0.4..0.85x. Because the model is rebuilt from the perturbed component, the
    resulting pole/zero and damping shifts are exact — no root heuristics.
    """

    name: str  # fault identifier (matches PhysicsAnomalyType)
    component: str  # converter component field to scale
    severity_range: Tuple[float, float]  # (low, high) multiplier on the component
    log_uniform: bool = False  # sample the multiplier log-uniformly (symmetric)

    def sample_severity(self, rng: np.random.Generator) -> float:
        lo, hi = self.severity_range
        if self.log_uniform:
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(rng.uniform(lo, hi))


#: Standard degradation modes shared by the basic PWM converters. Severity bands
#: mirror the anomalous ranges declared per converter in
#: ``data/<converter>/component_ranges.json`` so injected negatives sit *outside*
#: the healthy tolerance envelope (no overlap / gray zone).
DEFAULT_FAULT_MODES: Tuple[FaultMode, ...] = (
    FaultMode(PhysicsAnomalyType.CAPACITOR_ESR.value, "esr_cap", (3.0, 8.0)),
    FaultMode(PhysicsAnomalyType.CAPACITANCE_DROP.value, "capacitance", (0.30, 0.70)),
    FaultMode(PhysicsAnomalyType.INDUCTOR_SATURATION.value, "inductance", (0.40, 0.70)),
    FaultMode(PhysicsAnomalyType.INDUCTOR_ESR.value, "esr_ind", (2.0, 6.0)),
    FaultMode(PhysicsAnomalyType.SWITCH_DEGRADATION.value, "rds", (3.0, 15.0)),
)


def available_fault_modes() -> List[str]:
    """Names of the standard fault modes."""
    return [fm.name for fm in DEFAULT_FAULT_MODES]


# ---------------------------------------------------------------------------
# Second-order canonical assembly
# ---------------------------------------------------------------------------
def _assemble_second_order(
    gd0: float,
    w0: float,
    q: float,
    wz_esr: Optional[float],
    wz_rhp: Optional[float],
) -> TransferFunctionModel:
    """
    Assemble a ``zpk`` model from canonical parameters.

    Builds the poles from ``s^2 + (w0/Q) s + w0^2`` (real or complex), adds the
    LHP ESR zero and optional RHP zero, and scales the gain so ``|H(0)| == Gd0``.
    """
    zeros: List[complex] = []
    if wz_esr is not None:
        zeros.append(complex(-wz_esr, 0.0))  # LHP ESR zero
    if wz_rhp is not None:
        zeros.append(complex(+wz_rhp, 0.0))  # RHP zero (boost / buck-boost)

    poles = np.roots([1.0, w0 / q, w0 * w0]).astype(complex)
    zeros_arr = np.asarray(zeros, dtype=complex)

    num_dc = np.prod(-zeros_arr) if zeros_arr.size else (1.0 + 0j)
    den_dc = np.prod(-poles)  # == w0^2
    gain = float(np.real(abs(gd0) * den_dc / num_dc))

    return TransferFunctionModel(zeros=zeros_arr, poles=poles, gain=gain)


# ---------------------------------------------------------------------------
# Power converters (abstract base + concrete topologies)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerConverter(ABC):
    """
    Abstract PWM converter: component values + its own control-to-output model.

    Each concrete converter is responsible for two things:

    - :meth:`transfer_function` — its small-signal control-to-output ``zpk`` model
      (any order), built from the component fields below.
    - :attr:`fault_modes` — the degradable components it exposes (defaults to the
      standard set; override to add / restrict faults for a given converter).

    Subclasses auto-register by their :attr:`name`, so :meth:`from_name` /
    :func:`make_converter` can build any registered converter, and adding a new
    topology is just a subclass definition (Open/Closed).
    """

    vin: float = 50.0  # Input voltage [V]
    inductance: float = 200e-6  # L [H]
    capacitance: float = 100e-6  # C [F]
    load: float = 4.0  # Load resistance R [ohm]
    esr_cap: float = 20e-3  # Output-capacitor ESR [ohm]
    esr_ind: float = 100e-3  # Inductor DCR [ohm]
    # High-side (or symmetric effective) switch on-resistance [ohm]. When
    # ``rds_low`` is ``None``, both switches share ``rds``; set ``rds_low``
    # explicitly to model asymmetric synchronous-rectifier conduction or PSIM's
    # ideal third-quadrant body-diode clamp (``rds_low=0.0``).
    rds: float = 10e-3  # High-side / effective switch on-resistance [ohm]
    rds_low: Optional[float] = None  # Optional low-side switch on-resistance [ohm]
    duty: float = 0.4  # Nominal duty cycle D
    include_parasitic_drop: bool = True  # Include series resistive DC gain drop

    @property
    def effective_rds_low(self) -> float:
        """Low-side switch resistance, defaulting to ``rds`` when symmetric."""
        return self.rds if self.rds_low is None else self.rds_low

    #: Registry key; each concrete subclass sets this.
    name: ClassVar[str] = ""
    _registry: ClassVar[Dict[str, Type["PowerConverter"]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", ""):
            PowerConverter._registry[cls.name] = cls

    # -- responsibilities every converter defines -----------------------------
    @abstractmethod
    def transfer_function(self) -> TransferFunctionModel:
        """Nominal small-signal control-to-output ``zpk`` model."""

    @property
    def fault_modes(self) -> Tuple[FaultMode, ...]:
        """Degradable-component fault modes exposed by this converter."""
        return DEFAULT_FAULT_MODES

    # -- component perturbation ------------------------------------------------
    def scaled(self, **component_factors: float) -> "PowerConverter":
        """Return a copy with the named component fields multiplied by factors."""
        updates = {}
        for fieldname, factor in component_factors.items():
            if not hasattr(self, fieldname):
                raise ValueError(
                    f"{type(self).__name__} has no component {fieldname!r}"
                )
            current = getattr(self, fieldname)
            if fieldname == "rds_low" and current is None:
                current = self.rds
            updates[fieldname] = current * factor
        return replace(self, **updates)

    # -- construction / discovery ---------------------------------------------
    @classmethod
    def from_name(cls, name: Union[str, Enum], **components) -> "PowerConverter":
        key = name.value if isinstance(name, Enum) else str(name).strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown converter {name!r}. Available: {cls.available()}. "
                f"Define a PowerConverter subclass with name = {key!r} to add it."
            )
        return cls._registry[key](**components)

    @classmethod
    def available(cls) -> List[str]:
        """Names of all registered converter topologies."""
        return sorted(cls._registry)

    # -- shared canonical assembly (second-order topologies) ------------------
    def _canonical(
        self, m: float, gd0: float, wz_rhp: Optional[float]
    ) -> TransferFunctionModel:
        """
        Build the canonical second-order model from this converter's components.

        ``m`` is the conversion factor (``1`` for buck, ``1 - D`` for boost /
        buck-boost). Parasitics fold into ``Q`` via
        ``1/Q = w0*Le/R + w0*C*Rc + (Rl+Ron)/(w0*Le)`` with ``Le = L / m^2``.
        """
        L = max(self.inductance, 1e-15)
        C = max(self.capacitance, 1e-15)
        R = max(self.load, 1e-9)
        rc = max(self.esr_cap, 0.0)
        series_r = max(self.esr_ind, 0.0) + max(self.rds, 0.0)

        w0 = m / np.sqrt(L * C)
        le = L / (m * m)  # output-referred inductance
        inv_q = (w0 * le / R) + (w0 * C * rc) + (series_r / (w0 * le))
        q = 1.0 / max(inv_q, 1e-12)
        wz_esr = 1.0 / (rc * C) if rc > 0.0 else None
        return _assemble_second_order(abs(gd0), w0, q, wz_esr, wz_rhp)


@dataclass(frozen=True)
class BuckConverter(PowerConverter):
    """Synchronous Buck (step-down) small-signal control-to-output model.

    When ``include_parasitic_drop=True`` (default), uses the exact state-space
    averaged second-order transfer function including series resistive divider
    attenuation and asymmetric switch-drop modulation:

        Gvd(s) = Veff * R * (1 + s*Rc*C)
                 / ((R + Rs) + s*(L + Rs*C*(R + Rc) + R*Rc*C) + s^2*L*C*(R + Rc))

    where ``Rs = Rl + D*Rds1 + (1-D)*Rds2``, ``IL = D*Vin/(R + Rs)``, and
    ``Veff = Vin - IL*(Rds1 - Rds2)``. Setting ``include_parasitic_drop=False``
    reproduces the lossless-DC textbook approximation ``Gd0 = Vin``.
    """

    name: ClassVar[str] = "buck"

    def transfer_function(self) -> TransferFunctionModel:
        if not self.include_parasitic_drop:
            return self._canonical(m=1.0, gd0=self.vin, wz_rhp=None)

        L = max(self.inductance, 1e-15)
        C = max(self.capacitance, 1e-15)
        R = max(self.load, 1e-9)
        rc = max(self.esr_cap, 0.0)
        rl = max(self.esr_ind, 0.0)
        d = float(np.clip(self.duty, 0.0, 1.0))
        rds1 = max(self.rds, 0.0)
        rds2 = max(self.effective_rds_low, 0.0)

        rs = rl + d * rds1 + (1.0 - d) * rds2
        il = d * self.vin / (R + rs)
        veff = self.vin - il * (rds1 - rds2)

        a2 = L * C * (R + rc)
        a1 = L + rs * C * (R + rc) + R * rc * C
        a0 = R + rs

        poles = np.roots([1.0, a1 / a2, a0 / a2]).astype(complex)
        if rc > 0.0:
            zeros = np.asarray([complex(-1.0 / (rc * C), 0.0)], dtype=complex)
            gain = float((veff * R * rc * C) / a2)
        else:
            zeros = np.asarray([], dtype=complex)
            gain = float((veff * R) / a2)
        return TransferFunctionModel(zeros=zeros, poles=poles, gain=gain)


@dataclass(frozen=True)
class BoostConverter(PowerConverter):
    """Boost (step-up): RHP zero ``wz = (1-D)^2 R / L``. ``Gd0 = Vg/(1-D)^2``."""

    name: ClassVar[str] = "boost"

    def transfer_function(self) -> TransferFunctionModel:
        m = 1.0 - float(np.clip(self.duty, 1e-3, 1.0 - 1e-3))
        wz_rhp = (m * m) * max(self.load, 1e-9) / max(self.inductance, 1e-15)
        return self._canonical(m=m, gd0=self.vin / (m * m), wz_rhp=wz_rhp)


@dataclass(frozen=True)
class BuckBoostConverter(PowerConverter):
    """Buck-boost: RHP zero ``wz = (1-D)^2 R / (D L)``. ``Gd0 = Vg/(1-D)^2``."""

    name: ClassVar[str] = "buck_boost"

    def transfer_function(self) -> TransferFunctionModel:
        d = float(np.clip(self.duty, 1e-3, 1.0 - 1e-3))
        m = 1.0 - d
        wz_rhp = (m * m) * max(self.load, 1e-9) / (d * max(self.inductance, 1e-15))
        return self._canonical(m=m, gd0=self.vin / (m * m), wz_rhp=wz_rhp)


def make_converter(name: Union[str, Enum], **components) -> PowerConverter:
    """Instantiate a registered converter by name (e.g. ``make_converter('boost')``)."""
    return PowerConverter.from_name(name, **components)


def available_converters() -> List[str]:
    """Names of all registered converter topologies."""
    return PowerConverter.available()


# ---------------------------------------------------------------------------
# Pipeline Orchestrator (drop-in negative-sample generator for CARLA)
# ---------------------------------------------------------------------------
class PhysicsAnomalyInjector:
    """
    Converts a healthy (normalized) complex-response sample into a physically
    consistent anomalous one by degrading a converter component.

    The injector is a drop-in replacement for :class:`AnomalyInjector`: its
    :meth:`inject` returns ``(anomalous_sample, mask, anomaly_type)``.

    Args:
        frequencies: Frequency grid of the samples, in **Hz** (shape ``(seq_len,)``).
        scaler: Fitted feature scaler exposing ``transform`` / ``inverse_transform``
            (operates on the ``Real``/``Imag`` — or amplitude — feature channels).
        use_real_imag: Whether the two channels are ``(Real, Imag)`` of ``H`` (default)
            or a single amplitude-in-dB channel.
        fault_modes: Names of the fault modes to sample from (default: all the
            converter exposes).
        converter: The :class:`PowerConverter` to model (default: nominal buck).
            Pass any :class:`PowerConverter` subclass instance to model any topology.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        scaler=None,
        use_real_imag: bool = True,
        fault_modes: Optional[List[str]] = None,
        converter: Optional[PowerConverter] = None,
        seed: Optional[int] = None,
        adapt_load: bool = False,
        strict_margin: bool = False,
    ):
        frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
        if frequencies.size == 0:
            raise ValueError("PhysicsAnomalyInjector requires a non-empty frequency grid")

        self.frequencies = frequencies
        self.omega = 2.0 * np.pi * frequencies  # rad/s
        self.scaler = scaler
        self.use_real_imag = use_real_imag
        self.converter = converter or BuckConverter()
        self.rng = np.random.default_rng(seed)
        self.adapt_load = bool(adapt_load)
        self.strict_margin = bool(strict_margin)

        available = {fm.name: fm for fm in self.converter.fault_modes}
        if fault_modes:
            selected = []
            for n in fault_modes:
                key = n.value if isinstance(n, Enum) else str(n)
                if key in available:
                    selected.append(available[key])
            raw_modes = selected or list(available.values())
        else:
            raw_modes = list(available.values())

        if self.strict_margin:
            adjusted_modes = []
            for fm in raw_modes:
                lo, hi = fm.severity_range
                if fm.component in DEFAULT_HEALTHY_RANGES:
                    h_lo, h_hi = DEFAULT_HEALTHY_RANGES[fm.component]
                    if hi < 1.0:
                        hi = max(lo, min(hi, h_lo / h_hi))
                    elif lo > 1.0 and (h_hi / h_lo) < hi:
                        lo = min(hi, max(lo, h_hi / h_lo))
                adjusted_modes.append(replace(fm, severity_range=(float(lo), float(hi))))
            self._fault_modes = adjusted_modes
        else:
            self._fault_modes = raw_modes
        self.fault_mode_names = [fm.name for fm in self._fault_modes]

        # Nominal response, evaluated once.
        self.nominal_model = self.converter.transfer_function()
        self._h_nominal = self.nominal_model.frequency_response(self.omega)

        # Optional 1-D lookup table mapping resonant peak |H| -> load multiplier
        self._load_grid: Optional[np.ndarray] = None
        self._load_peaks: Optional[np.ndarray] = None
        if self.adapt_load and hasattr(self.converter, "load"):
            grid = np.linspace(0.35, 2.50, 44)
            peaks = np.array(
                [
                    float(
                        np.max(
                            np.abs(
                                self.converter.scaled(load=float(m))
                                .transfer_function()
                                .frequency_response(self.omega)
                            )
                        )
                    )
                    for m in grid
                ],
                dtype=float,
            )
            order = np.argsort(peaks)
            self._load_peaks = peaks[order]
            self._load_grid = grid[order]

        logger.info(
            "Initialized PhysicsAnomalyInjector (%s) with %d fault modes over %d freqs",
            self.converter.name,
            len(self._fault_modes),
            self.frequencies.size,
        )

    # -- normalization helpers ------------------------------------------------
    def _to_features(self, sample: np.ndarray) -> np.ndarray:
        """Un-normalize a sample to physical feature space (Real/Imag or dB)."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(sample[None, ...])[0]
        return sample.copy()

    def _to_normalized(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(features[None, ...])[0]
        return features

    # -- core -----------------------------------------------------------------
    def _estimate_load_multiplier(self, h_sample: np.ndarray) -> float:
        if self._load_peaks is None or self._load_grid is None:
            return 1.0
        obs_peak = float(np.max(np.abs(h_sample)))
        return float(np.interp(obs_peak, self._load_peaks, self._load_grid))

    def _fault_ratio(
        self, fault: FaultMode, severity: float, load_mult: Optional[float] = None
    ) -> np.ndarray:
        """Multiplicative transfer ratio ``G(jw) = H_fault / H_nominal``."""
        eps = 1e-30
        if load_mult is not None and abs(load_mult - 1.0) > 1e-3 and hasattr(self.converter, "load"):
            base_conv = self.converter.scaled(load=float(load_mult))
            denom = base_conv.transfer_function().frequency_response(self.omega)
            faulted = base_conv.scaled(**{fault.component: severity})
            h_fault = faulted.transfer_function().frequency_response(self.omega)
        else:
            denom = self._h_nominal
            faulted = self.converter.scaled(**{fault.component: severity})
            h_fault = faulted.transfer_function().frequency_response(self.omega)
        return h_fault / np.where(np.abs(denom) < eps, eps, denom)

    def _resolve_fault(self, anomaly_type) -> FaultMode:
        if anomaly_type is None:
            return self._fault_modes[int(self.rng.integers(len(self._fault_modes)))]
        key = anomaly_type.value if isinstance(anomaly_type, Enum) else str(anomaly_type)
        for fm in self._fault_modes:
            if fm.name == key:
                return fm
        raise ValueError(
            f"Fault mode {anomaly_type!r} not available for {self.converter.name}. "
            f"Available: {self.fault_mode_names}"
        )

    def inject(
        self,
        data: np.ndarray,
        anomaly_type: Optional[str] = None,
        severity: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Inject a physics-informed transfer-ratio fault into a single sample.

        Args:
            data: Normalized sample, shape ``(seq_len, n_features)``.
            anomaly_type: Specific fault mode name (random if ``None``).
            severity: Component multiplier (sampled from the fault range if ``None``).

        Returns:
            ``(anomalous_sample, mask, anomaly_type)`` with an all-``True`` mask
            (the fault is a structural / collective anomaly).
        """
        fault = self._resolve_fault(anomaly_type)
        if severity is None:
            severity = fault.sample_severity(self.rng)

        features = self._to_features(np.asarray(data, dtype=float))
        load_mult = None
        if self.adapt_load and self.use_real_imag and features.shape[-1] >= 2:
            h_obs = features[:, 0] + 1j * features[:, 1]
            load_mult = self._estimate_load_multiplier(h_obs)

        ratio = self._fault_ratio(fault, severity, load_mult=load_mult)

        if self.use_real_imag and features.shape[-1] >= 2:
            h_sample = features[:, 0] + 1j * features[:, 1]
            h_out = h_sample * ratio
            out_features = features.copy()
            out_features[:, 0] = np.real(h_out)
            out_features[:, 1] = np.imag(h_out)
        else:
            # Amplitude-only (dB) channel: apply the magnitude ratio in dB.
            out_features = features.copy()
            out_features[:, 0] = features[:, 0] + 20.0 * np.log10(
                np.clip(np.abs(ratio), 1e-12, None)
            )

        anomalous = self._to_normalized(out_features).astype(data.dtype, copy=False)
        mask = np.ones(anomalous.shape[0], dtype=bool)
        return anomalous, mask, fault.name

    def inject_batch(
        self,
        data: np.ndarray,
        anomaly_prob: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject faults into a batch ``(batch, seq_len, n_features)``."""
        batch_size, seq_len = data.shape[0], data.shape[1]
        anomalous = data.copy()
        masks = np.zeros((batch_size, seq_len), dtype=bool)
        is_anomaly = self.rng.random(batch_size) < anomaly_prob

        for i in range(batch_size):
            if is_anomaly[i]:
                anomalous[i], masks[i], _ = self.inject(data[i])
        return anomalous, masks, is_anomaly


DEFAULT_HEALTHY_RANGES: Dict[str, Tuple[float, float]] = {
    "capacitance": (0.80, 1.20),
    "esr_cap": (0.50, 2.00),
    "inductance": (0.80, 1.20),
    "esr_ind": (0.85, 1.40),
    "rds": (1.00, 2.20),
}


def _count_bode_peaks(h: np.ndarray, min_prominence_db: float = 0.5) -> int:
    mag_db = 20.0 * np.log10(np.clip(np.abs(h), 1e-15, None))
    peaks, _ = signal.find_peaks(mag_db, prominence=min_prominence_db)
    return int(peaks.size)


def evaluate_transfer_ratio_error(
    frequencies: np.ndarray,
    converter: Optional[PowerConverter] = None,
    healthy_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    n_samples: int = 256,
    healthy_spread: bool = True,
    seed: int = 42,
    min_prominence_db: float = 0.5,
) -> Dict[str, float]:
    """Quantify first-order approximation error of the multiplicative transfer ratio.

    Compares the ratio-synthesized negative ``X(jw; theta_h) * H(jw; theta_f) / H0(jw)``
    against the exact small-signal response of the same healthy unit after the
    sampled component transitions into its degraded value.
    """
    conv0 = converter or BuckConverter()
    ranges = healthy_ranges or DEFAULT_HEALTHY_RANGES
    omega = 2.0 * np.pi * np.asarray(frequencies, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)

    h_nom = conv0.transfer_function().frequency_response(omega)
    eps = 1e-30
    denom = np.where(np.abs(h_nom) < eps, eps, h_nom)
    fault_modes = conv0.fault_modes

    rel_errors = np.zeros(n_samples, dtype=float)
    ratio_multi_peaks = 0
    true_multi_peaks = 0

    for i in range(n_samples):
        if healthy_spread:
            h_factors = {
                comp: float(rng.uniform(lo, hi))
                for comp, (lo, hi) in ranges.items()
                if hasattr(conv0, comp)
            }
        else:
            h_factors = {comp: 1.0 for comp in ranges if hasattr(conv0, comp)}

        anchor_conv = conv0.scaled(**h_factors)
        h_anchor = anchor_conv.transfer_function().frequency_response(omega)

        fm = fault_modes[int(rng.integers(len(fault_modes)))]
        sev = fm.sample_severity(rng)

        # Ratio negative around nominal baseline
        h_nom_fault = conv0.scaled(**{fm.component: sev}).transfer_function().frequency_response(omega)
        h_ratio = h_anchor * (h_nom_fault / denom)

        # True degraded response of the same unit (faulted component replaced by degraded multiplier)
        unit_fault_factors = dict(h_factors)
        unit_fault_factors[fm.component] = sev
        h_true = conv0.scaled(**unit_fault_factors).transfer_function().frequency_response(omega)

        norm_true = max(float(np.linalg.norm(h_true)), eps)
        rel_errors[i] = float(np.linalg.norm(h_ratio - h_true)) / norm_true

        if _count_bode_peaks(h_ratio, min_prominence_db=min_prominence_db) >= 2:
            ratio_multi_peaks += 1
        if _count_bode_peaks(h_true, min_prominence_db=min_prominence_db) >= 2:
            true_multi_peaks += 1

    return {
        "rel_l2_median": float(np.median(rel_errors)),
        "rel_l2_mean": float(np.mean(rel_errors)),
        "rel_l2_p90": float(np.percentile(rel_errors, 90)),
        "rel_l2_max": float(np.max(rel_errors)),
        "multi_peak_fraction": float(ratio_multi_peaks / max(n_samples, 1)),
        "true_multi_peak_fraction": float(true_multi_peaks / max(n_samples, 1)),
    }


def apply_polar_fra_jitter(
    data: np.ndarray,
    scaler=None,
    use_real_imag: bool = True,
    max_sigma_db: float = 0.0,
    max_sigma_deg: float = 0.0,
    clean_prob: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply physical swept-sine FRA magnitude/phase jitter in complex H(jw) space.

    Supports both 3-D ``(B, L, C)`` and 4-D ``(B, K, L, C)`` normalized batches by
    inverting ``scaler`` to physical feature space, applying per-frequency polar
    perturbations ``H(jw) * 10**(eps_dB/20) * exp(j * eps_rad)`` with random noise
    scale ``u ~ U[0, 1]`` per curve (and zero noise with probability ``clean_prob``),
    and transforming back to normalized space.
    """
    arr = np.asarray(data)
    if (max_sigma_db <= 0.0 and max_sigma_deg <= 0.0) or clean_prob >= 1.0:
        return arr.copy()

    orig_shape = arr.shape
    seq_len, n_feat = orig_shape[-2], orig_shape[-1]
    flat = arr.reshape(-1, seq_len, n_feat).astype(np.float64, copy=True)
    m = flat.shape[0]
    gen = rng if rng is not None else np.random.default_rng()

    features = scaler.inverse_transform(flat).copy() if scaler is not None else flat
    scale = gen.uniform(0.0, 1.0, size=(m, 1))
    if clean_prob > 0.0:
        is_clean = gen.random(size=(m, 1)) < float(clean_prob)
        scale = np.where(is_clean, 0.0, scale)

    eps_db = (
        gen.normal(0.0, 1.0, size=(m, seq_len)) * (scale * float(max_sigma_db))
        if max_sigma_db > 0.0
        else 0.0
    )
    eps_rad = (
        gen.normal(0.0, 1.0, size=(m, seq_len))
        * np.deg2rad(scale * float(max_sigma_deg))
        if max_sigma_deg > 0.0
        else 0.0
    )

    if use_real_imag and n_feat >= 2:
        h = features[:, :, 0] + 1j * features[:, :, 1]
        gain = (10.0 ** (eps_db / 20.0)) if max_sigma_db > 0.0 else 1.0
        rot = np.exp(1j * eps_rad) if max_sigma_deg > 0.0 else 1.0
        h_noisy = h * gain * rot
        features[:, :, 0] = np.real(h_noisy)
        features[:, :, 1] = np.imag(h_noisy)
    else:
        features[:, :, 0] = features[:, :, 0] + eps_db

    out = scaler.transform(features) if scaler is not None else features
    return out.reshape(orig_shape).astype(arr.dtype, copy=False)
