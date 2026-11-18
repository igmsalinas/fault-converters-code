import numpy as np
import pytest
from dataclasses import dataclass
from typing import ClassVar

from src.data.physics_anomaly import (
    BoostConverter,
    BuckBoostConverter,
    BuckConverter,
    DEFAULT_FAULT_MODES,
    FaultMode,
    PhysicsAnomalyInjector,
    PhysicsAnomalyType,
    PowerConverter,
    TransferFunctionModel,
    available_converters,
    available_fault_modes,
    evaluate_transfer_ratio_error,
    make_converter,
)


@pytest.fixture
def frequencies():
    # Log sweep resembling the buck AC-sweep grid (100 Hz .. 200 kHz).
    return np.logspace(2, np.log10(2e5), 256)


class _IdentityScaler:
    """Per-feature standard scaler stub for (batch, seq, feat) arrays."""

    def __init__(self, features):
        self.mean_ = features.mean(axis=(0, 1), keepdims=True)
        self.std_ = features.std(axis=(0, 1), keepdims=True) + 1e-8

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        return x * self.std_ + self.mean_


def _scaler_and_sample(converter, frequencies):
    H = converter.transfer_function().frequency_response(2 * np.pi * frequencies)
    feats = np.stack([H.real, H.imag], axis=-1)
    scaler = _IdentityScaler(feats[None])
    sample = scaler.transform(feats[None])[0].astype(np.float32)
    return scaler, sample, H


# ---------------------------------------------------------------------------
# Converter hierarchy + transfer functions
# ---------------------------------------------------------------------------
def test_converters_auto_register():
    assert set(available_converters()) >= {"buck", "boost", "buck_boost"}
    assert isinstance(make_converter("boost"), BoostConverter)
    assert isinstance(make_converter("buck"), BuckConverter)


def test_from_name_unknown_raises():
    with pytest.raises(ValueError, match="Unknown converter"):
        PowerConverter.from_name("sepic")


def test_power_converter_is_abstract():
    with pytest.raises(TypeError):
        PowerConverter()  # abstract transfer_function


def test_buck_transfer_function_physics():
    model = BuckConverter().transfer_function()
    assert model.zeros.shape == (1,)  # only the LHP ESR zero
    assert model.poles.shape == (2,)
    f0 = np.abs(model.poles[0]) / (2 * np.pi)
    assert 900 < f0 < 1400  # LC resonance ~ 1.1 kHz
    fz = np.abs(model.zeros[0]) / (2 * np.pi)
    assert 60e3 < fz < 100e3  # ESR zero ~ 80 kHz
    assert np.all(model.zeros.real < 0)  # minimum phase


def _real_zeros(model):
    z = model.zeros
    return z[np.abs(z.imag) < 1e-6]


@pytest.mark.parametrize("cls", [BoostConverter, BuckBoostConverter])
def test_boost_family_has_rhp_zero(cls):
    model = cls().transfer_function()
    real = _real_zeros(model)
    assert real[real.real > 0].size == 1  # RHP zero
    assert real[real.real < 0].size == 1  # ESR zero


def test_dc_gain_matches_topology_formula():
    vg, d = 50.0, 0.4
    dc_buck = np.abs(BuckConverter(vin=vg, duty=d).transfer_function()
                     .frequency_response(np.array([1.0]))[0])
    assert dc_buck == pytest.approx(vg, rel=0.05)
    for cls in (BoostConverter, BuckBoostConverter):
        dc = np.abs(cls(vin=vg, duty=d).transfer_function()
                    .frequency_response(np.array([1.0]))[0])
        assert dc == pytest.approx(vg / (1 - d) ** 2, rel=0.05)


def test_resonance_scales_with_conversion_ratio():
    d = 0.4
    buck = BuckConverter(duty=d).transfer_function()
    boost = BoostConverter(duty=d).transfer_function()
    ratio = np.abs(boost.poles[0]) / np.abs(buck.poles[0])
    assert ratio == pytest.approx(1 - d, rel=0.05)


def test_custom_parameters_set_resonance_and_gain():
    conv = BuckConverter(vin=12.0, inductance=47e-6, capacitance=220e-6, load=2.0, duty=0.42)
    model = conv.transfer_function()
    f0 = np.abs(model.poles[0]) / (2 * np.pi)
    rs = conv.esr_ind + conv.duty * conv.rds + (1.0 - conv.duty) * conv.effective_rds_low
    expected_f0 = (
        1.0
        / (2 * np.pi * np.sqrt(47e-6 * 220e-6))
        * np.sqrt((2.0 + rs) / (2.0 + conv.esr_cap))
    )
    assert f0 == pytest.approx(expected_f0, rel=1e-3)
    dc = np.abs(model.frequency_response(np.array([1e-3]))[0])
    il = conv.duty * 12.0 / (2.0 + rs)
    veff = 12.0 - il * (conv.rds - conv.effective_rds_low)
    expected_dc = veff * 2.0 / (2.0 + rs)
    assert dc == pytest.approx(expected_dc, rel=1e-3)


def test_scaled_returns_same_type_with_scaled_component():
    conv = BuckConverter()
    faulted = conv.scaled(capacitance=0.5, esr_cap=4.0)
    assert isinstance(faulted, BuckConverter)
    assert faulted.capacitance == pytest.approx(conv.capacitance * 0.5)
    assert faulted.esr_cap == pytest.approx(conv.esr_cap * 4.0)
    assert conv.capacitance == 100e-6  # original unchanged (frozen)


def test_scaled_unknown_component_raises():
    with pytest.raises(ValueError, match="no component"):
        BuckConverter().scaled(nonexistent=2.0)


# ---------------------------------------------------------------------------
# Fault modes
# ---------------------------------------------------------------------------
def test_default_fault_modes_cover_enum():
    assert {fm.name for fm in DEFAULT_FAULT_MODES} == {t.value for t in PhysicsAnomalyType}
    assert available_fault_modes() == [fm.name for fm in DEFAULT_FAULT_MODES]
    assert {fm.component for fm in DEFAULT_FAULT_MODES} == {
        "esr_cap", "capacitance", "inductance", "esr_ind", "rds"
    }
    # Load variation is a healthy operating-condition dimension in dataset_01,
    # so it must never be synthesized as a contrastive fault negative.
    assert all(fm.component != "load" for fm in DEFAULT_FAULT_MODES)


def test_fault_mode_severity_sampling_in_range():
    rng = np.random.default_rng(0)
    fm = FaultMode("x", "capacitance", (0.4, 0.85))
    vals = [fm.sample_severity(rng) for _ in range(200)]
    assert all(0.4 <= v <= 0.85 for v in vals)

    log_fm = FaultMode("y", "load", (0.3, 3.0), log_uniform=True)
    lv = [log_fm.sample_severity(rng) for _ in range(200)]
    assert all(0.3 <= v <= 3.0 for v in lv)


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------
def test_inject_produces_finite_distinct_negative(frequencies):
    scaler, sample, _ = _scaler_and_sample(BuckConverter(), frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler, seed=0)
    for name in available_fault_modes():
        anomalous, mask, atype = inj.inject(sample, anomaly_type=name)
        assert atype == name
        assert anomalous.shape == sample.shape
        assert anomalous.dtype == sample.dtype
        assert np.all(np.isfinite(anomalous))
        assert mask.all()
        assert not np.allclose(anomalous, sample)


def test_inject_defaults_to_buck(frequencies):
    inj = PhysicsAnomalyInjector(frequencies=frequencies, seed=0)
    assert isinstance(inj.converter, BuckConverter)


@pytest.mark.parametrize("cls", [BuckConverter, BoostConverter, BuckBoostConverter])
def test_injector_generalizes_across_topologies(frequencies, cls):
    scaler, sample, _ = _scaler_and_sample(cls(), frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler,
                                 converter=cls(), seed=0)
    for name in available_fault_modes():
        anomalous, _, _ = inj.inject(sample, anomaly_type=name)
        assert np.all(np.isfinite(anomalous))
        assert not np.allclose(anomalous, sample)


def test_reactive_faults_preserve_dc_gain_and_resistive_faults_attenuate_it(frequencies):
    low_freqs = np.logspace(-2, 5, 256)  # down to 0.01 Hz to check true DC limit
    scaler, sample, H = _scaler_and_sample(BuckConverter(), low_freqs)
    inj = PhysicsAnomalyInjector(frequencies=low_freqs, scaler=scaler, seed=1)
    dc_healthy = abs(H[0])

    reactive_modes = (
        PhysicsAnomalyType.CAPACITOR_ESR.value,
        PhysicsAnomalyType.CAPACITANCE_DROP.value,
        PhysicsAnomalyType.INDUCTOR_SATURATION.value,
    )
    for name in reactive_modes:
        anomalous, _, _ = inj.inject(sample, anomaly_type=name, severity=None)
        faulty = scaler.inverse_transform(anomalous[None])[0]
        dc_faulty = abs(faulty[0, 0] + 1j * faulty[0, 1])
        assert dc_faulty == pytest.approx(dc_healthy, rel=1e-3)

    resistive_modes = (
        (PhysicsAnomalyType.INDUCTOR_ESR.value, 5.0),
        (PhysicsAnomalyType.SWITCH_DEGRADATION.value, 10.0),
    )
    for name, sev in resistive_modes:
        anomalous, _, _ = inj.inject(sample, anomaly_type=name, severity=sev)
        faulty = scaler.inverse_transform(anomalous[None])[0]
        dc_faulty = abs(faulty[0, 0] + 1j * faulty[0, 1])
        assert dc_faulty < dc_healthy * 0.99


def test_buck_legacy_lossless_dc_mode():
    conv = BuckConverter(include_parasitic_drop=False)
    dc = np.abs(conv.transfer_function().frequency_response(np.array([1e-3]))[0])
    assert dc == pytest.approx(conv.vin, rel=1e-4)


def test_buck_asymmetric_switch_resistances_and_d05_blind_spot():
    # At D = 0.5, d G_vd(0) / d Rds_2 at first order has (2D - 1) = 0 numerator-denominator cancellation!
    base = BuckConverter(vin=50.0, load=4.0, esr_ind=0.1, rds=0.01, rds_low=0.01, duty=0.5)
    high_fault = base.scaled(rds=10.0)
    low_fault = base.scaled(rds_low=10.0)
    dc_base = np.abs(base.transfer_function().frequency_response(np.array([1e-4]))[0])
    dc_high = np.abs(high_fault.transfer_function().frequency_response(np.array([1e-4]))[0])
    dc_low = np.abs(low_fault.transfer_function().frequency_response(np.array([1e-4]))[0])
    # High-side fault drops DC gain substantially more than low-side fault
    assert (dc_base - dc_high) > 10.0 * abs(dc_base - dc_low)


def test_buck_matches_impedance_divider_identity(frequencies):
    conv = BuckConverter(
        vin=50.0,
        inductance=180e-6,
        capacitance=85e-6,
        load=4.0,
        esr_cap=50e-3,
        esr_ind=250e-3,
        rds=80e-3,
        rds_low=0.0,
        duty=0.4,
    )
    omega = 2.0 * np.pi * frequencies
    s = 1j * omega
    h_model = conv.transfer_function().frequency_response(omega)

    # Independent circuit derivation via complex impedance divider Z_out(s) / (Z_L(s) + Z_out(s))
    rs = conv.esr_ind + conv.duty * conv.rds
    il = conv.duty * conv.vin / (conv.load + rs)
    veff = conv.vin - il * conv.rds
    z_cap = conv.esr_cap + 1.0 / (s * conv.capacitance)
    z_out = (conv.load * z_cap) / (conv.load + z_cap)
    z_ind = rs + s * conv.inductance
    h_impedance = veff * (z_out / (z_ind + z_out))

    assert np.allclose(h_model, h_impedance, rtol=1e-10, atol=1e-10)


def test_evaluate_transfer_ratio_error_bounds(frequencies):
    stats = evaluate_transfer_ratio_error(
        frequencies=frequencies,
        n_samples=64,
        seed=42,
    )
    assert 0.0 < stats["rel_l2_median"] < 0.35
    assert stats["rel_l2_p90"] >= stats["rel_l2_median"]
    assert 0.0 <= stats["multi_peak_fraction"] <= 1.0
    # At exact nominal anchor (zero tolerance spread), the multiplicative ratio is exact!
    exact_stats = evaluate_transfer_ratio_error(
        frequencies=frequencies,
        n_samples=16,
        healthy_spread=False,
        seed=42,
    )
    assert exact_stats["rel_l2_max"] < 1e-10


def test_esr_increase_lifts_high_frequency_magnitude(frequencies):
    scaler, sample, H = _scaler_and_sample(BuckConverter(), frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler, seed=2)
    anomalous, _, _ = inj.inject(
        sample, anomaly_type=PhysicsAnomalyType.CAPACITOR_ESR, severity=5.0
    )
    faulty = scaler.inverse_transform(anomalous[None])[0]
    mag_healthy = np.abs(H)
    mag_faulty = np.abs(faulty[:, 0] + 1j * faulty[:, 1])
    assert mag_faulty[-1] > mag_healthy[-1]


def test_inject_batch_respects_probability(frequencies):
    scaler, sample, _ = _scaler_and_sample(BuckConverter(), frequencies)
    batch = np.repeat(sample[None], 8, axis=0)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler, seed=3)

    _, _, none = inj.inject_batch(batch, anomaly_prob=0.0)
    assert not none.any()

    anom, masks, flags = inj.inject_batch(batch, anomaly_prob=1.0)
    assert flags.all()
    assert masks.all()
    assert not np.allclose(anom, batch)


def test_unknown_fault_mode_raises(frequencies):
    inj = PhysicsAnomalyInjector(frequencies=frequencies, seed=0)
    with pytest.raises(ValueError, match="not available"):
        inj.inject(np.zeros((frequencies.size, 2), dtype=np.float32),
                   anomaly_type="not_a_fault")


def test_amplitude_only_channel(frequencies):
    H = BuckConverter().transfer_function().frequency_response(2 * np.pi * frequencies)
    db = (20 * np.log10(np.abs(H)))[:, None].astype(np.float32)
    inj = PhysicsAnomalyInjector(
        frequencies=frequencies, scaler=None, use_real_imag=False, seed=5
    )
    anomalous, _, _ = inj.inject(db, anomaly_type=PhysicsAnomalyType.CAPACITANCE_DROP)
    assert anomalous.shape == db.shape
    assert np.all(np.isfinite(anomalous))
    assert not np.allclose(anomalous, db)


# ---------------------------------------------------------------------------
# Extensibility: a custom higher-order converter (subclass auto-registers)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _FourthOrderConverter(PowerConverter):
    """Two-resonance converter the canonical second-order form cannot express."""

    name: ClassVar[str] = "fourth_order_test"

    def transfer_function(self) -> TransferFunctionModel:
        poles = np.array(
            [-500 + 6000j, -500 - 6000j, -800 + 30000j, -800 - 30000j], dtype=complex
        )
        # ESR zero depends on esr_cap * capacitance so capacitor faults perturb it.
        zeros = np.array([-1.0 / (self.esr_cap * self.capacitance)], dtype=complex)
        return TransferFunctionModel(zeros=zeros, poles=poles, gain=1e18)


def test_custom_subclass_registers_and_flows_through_injector(frequencies):
    assert "fourth_order_test" in available_converters()
    conv = make_converter("fourth_order_test")
    assert isinstance(conv, _FourthOrderConverter)

    model = conv.transfer_function()
    assert model.poles.shape == (4,)  # canonical template could never do this

    scaler, sample, _ = _scaler_and_sample(conv, frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler,
                                 converter=conv, seed=0)
    anomalous, _, _ = inj.inject(sample, anomaly_type=PhysicsAnomalyType.CAPACITOR_ESR)
    assert np.all(np.isfinite(anomalous))
    assert not np.allclose(anomalous, sample)


def test_apply_polar_fra_jitter_shapes_clean_prob_and_polar_scaling(frequencies):
    from src.data.physics_anomaly import apply_polar_fra_jitter

    conv = BuckConverter()
    scaler, sample, _ = _scaler_and_sample(conv, frequencies)
    batch_3d = np.repeat(sample[None, ...], 8, axis=0)
    batch_4d = np.repeat(batch_3d[:, None, ...], 2, axis=1)

    # Zero jitter or clean_prob=1.0 must preserve inputs identically
    out_zero = apply_polar_fra_jitter(
        batch_3d, scaler=scaler, use_real_imag=True, max_sigma_db=0.0, max_sigma_deg=0.0
    )
    assert np.allclose(out_zero, batch_3d)

    out_clean = apply_polar_fra_jitter(
        batch_3d,
        scaler=scaler,
        use_real_imag=True,
        max_sigma_db=0.3,
        max_sigma_deg=1.5,
        clean_prob=1.0,
        rng=np.random.default_rng(7),
    )
    assert np.allclose(out_clean, batch_3d)

    # Non-zero polar FRA jitter perturbs both 3-D and 4-D batches proportional to |H(jw)|
    out_noisy_3d = apply_polar_fra_jitter(
        batch_3d,
        scaler=scaler,
        use_real_imag=True,
        max_sigma_db=0.3,
        max_sigma_deg=1.5,
        clean_prob=0.0,
        rng=np.random.default_rng(7),
    )
    out_noisy_4d = apply_polar_fra_jitter(
        batch_4d,
        scaler=scaler,
        use_real_imag=True,
        max_sigma_db=0.3,
        max_sigma_deg=1.5,
        clean_prob=0.0,
        rng=np.random.default_rng(8),
    )
    assert out_noisy_3d.shape == batch_3d.shape
    assert out_noisy_4d.shape == batch_4d.shape
    assert np.all(np.isfinite(out_noisy_3d))
    assert np.all(np.isfinite(out_noisy_4d))
    assert not np.allclose(out_noisy_3d, batch_3d)

    # Polar FRA perturbation in physical space must be much larger at DC/resonance than at 50 kHz
    phys_clean = scaler.inverse_transform(batch_3d)
    phys_noisy = scaler.inverse_transform(out_noisy_3d)
    delta = np.linalg.norm(phys_noisy - phys_clean, axis=-1)  # (8, 101)
    assert float(delta[:, 0].mean()) > 5.0 * float(delta[:, -1].mean())


def test_adapt_load_and_strict_margin_under_variable_rout(frequencies):
    conv = BuckConverter()
    scaler, _, _ = _scaler_and_sample(conv, frequencies)

    inj_fixed = PhysicsAnomalyInjector(
        frequencies=frequencies,
        scaler=scaler,
        converter=conv,
        seed=42,
        adapt_load=False,
    )
    inj_adapt = PhysicsAnomalyInjector(
        frequencies=frequencies,
        scaler=scaler,
        converter=conv,
        seed=42,
        adapt_load=True,
        strict_margin=True,
    )

    # Heavy-load anchor (Rout = 0.5x = 2 Ohm) undergoing ESR_ind = 4.0x fault
    omega = 2.0 * np.pi * frequencies
    conv_heavy = conv.scaled(load=0.5)
    h_heavy = conv_heavy.transfer_function().frequency_response(omega)
    feat_heavy = np.column_stack([h_heavy.real, h_heavy.imag])
    norm_heavy = scaler.transform(feat_heavy[None, ...])[0]

    h_true_fault = conv_heavy.scaled(esr_ind=4.0).transfer_function().frequency_response(omega)

    anom_fixed, _, _ = inj_fixed.inject(norm_heavy, anomaly_type="inductor_esr", severity=4.0)
    anom_adapt, _, _ = inj_adapt.inject(norm_heavy, anomaly_type="inductor_esr", severity=4.0)

    feat_fixed = scaler.inverse_transform(anom_fixed[None, ...])[0]
    feat_adapt = scaler.inverse_transform(anom_adapt[None, ...])[0]
    h_fixed = feat_fixed[:, 0] + 1j * feat_fixed[:, 1]
    h_adapt = feat_adapt[:, 0] + 1j * feat_adapt[:, 1]

    err_fixed = np.linalg.norm(h_fixed - h_true_fault) / np.linalg.norm(h_true_fault)
    err_adapt = np.linalg.norm(h_adapt - h_true_fault) / np.linalg.norm(h_true_fault)

    # Load-conditioned ratio must be dramatically closer to true 2-Ohm fault response
    assert err_adapt < 0.03
    assert err_adapt < 0.25 * err_fixed

    # Strict margin: sampled capacitive drop severity must guarantee 1.20 * sev < 0.80
    cap_fault = [fm for fm in inj_adapt._fault_modes if fm.component == "capacitance"][0]
    sampled = [cap_fault.sample_severity(np.random.default_rng(s)) for s in range(50)]
    assert max(sampled) <= (0.80 / 1.20) + 1e-6
