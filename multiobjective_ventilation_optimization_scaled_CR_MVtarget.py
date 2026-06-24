"""
Multi-objective in-silico optimization of protective ventilation
================================================================

Scaled C/R + fixed-PEEP + primary MV-target analysis.

Design:
- PEEP is fixed at 5 cmH2O and is not optimized.
- Compartmental mechanics are scaled for realistic whole-system mechanics:
    * compartmental compliance values are divided by 6
    * compartmental resistance values are multiplied by 6
- Primary analysis is restricted to a comparable minute ventilation target:
    * MV target = 7.0 L/min
    * default tolerance = ±0.5 L/min, so 6.5–7.5 L/min
- Sensitivity analysis keeps the broader feasible MV range:
    * 4.0–12.0 L/min

Outputs:
Primary analysis:
- MVtarget_primary_all_simulation_results.csv
- MVtarget_primary_feasible_simulation_results.csv
- MVtarget_primary_strategy_summary.csv

Sensitivity analysis:
- MVtarget_sensitivity_feasible_simulation_results.csv
- MVtarget_sensitivity_strategy_summary.csv

Mechanics:
- MVtarget_mechanics_summary.csv

Run:
    python multiobjective_ventilation_optimization_scaled_CR_MVtarget.py
"""

import itertools
import numpy as np
import pandas as pd


CMH2O_L_TO_J = 0.0980665
COMPLIANCE_SCALE = 1.0 / 6.0
RESISTANCE_SCALE = 6.0


# ---------------------------------------------------------------------
# 1. Model mechanics
# ---------------------------------------------------------------------

def scale_mechanics(C, R):
    """
    Scales compartmental mechanics to preserve realistic whole-system mechanics
    in a six-compartment parallel model.
    """
    return np.asarray(C, dtype=float) * COMPLIANCE_SCALE, np.asarray(R, dtype=float) * RESISTANCE_SCALE


def get_phenotypes():
    """
    Compliance: L/cmH2O
    Resistance: cmH2O*s/L

    The raw phenotype gradients are scaled by scale_mechanics().
    """
    raw_phenotypes = {
        "compliance_dominant": {
            "C": np.array([0.020, 0.028, 0.036, 0.044, 0.052, 0.060]),
            "R": np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
        },
        "resistance_dominant": {
            "C": np.array([0.040, 0.040, 0.040, 0.040, 0.040, 0.040]),
            "R": np.array([4.0, 6.4, 8.8, 11.2, 13.6, 16.0]),
        },
        "mixed": {
            "C": np.array([0.020, 0.029, 0.038, 0.047, 0.056, 0.065]),
            "R": np.array([16.0, 13.6, 11.2, 8.8, 6.4, 4.0]),
        },
        "severe_mixed": {
            "C": np.array([0.015, 0.025, 0.038, 0.055, 0.075, 0.095]),
            "R": np.array([20.0, 16.0, 12.0, 8.0, 5.0, 3.0]),
        },
    }

    phenotypes = {}
    for name, props in raw_phenotypes.items():
        C_scaled, R_scaled = scale_mechanics(props["C"], props["R"])
        phenotypes[name] = {"C": C_scaled, "R": R_scaled}

    return phenotypes


def calculate_system_mechanics(C, R):
    """
    Returns approximate equivalent mechanics for the parallel model.
    """
    C_total = float(np.sum(C))
    R_equivalent = float(1.0 / np.sum(1.0 / R))
    return C_total, R_equivalent


# ---------------------------------------------------------------------
# 2. Waveforms and breath simulation
# ---------------------------------------------------------------------

def generate_flow_waveform(vt_l, ti_s, pause_fraction, waveform, dt=0.002):
    n_steps = int(np.round(ti_s / dt))
    active_ti = ti_s * (1.0 - pause_fraction)
    active_steps = max(1, int(np.round(active_ti / dt)))

    q = np.zeros(n_steps)

    if waveform == "square":
        q_active = np.ones(active_steps)
    elif waveform == "decelerating":
        q_active = np.linspace(1.5, 0.5, active_steps)
    elif waveform == "sinusoidal":
        x = np.linspace(0, np.pi, active_steps)
        q_active = np.sin(x)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")

    q_active = np.maximum(q_active, 0)
    q_active = q_active * (vt_l / (np.sum(q_active) * dt))
    q[:active_steps] = q_active

    return q


def simulate_breath(C, R, vt_l, rr, ti_s, waveform, pause_fraction, peep_cmH2O=5.0, dt=0.002):
    """
    Volume-controlled breath in a parallel-compartment model.

    q_i = (Paw_above - V_i/C_i) / R_i
    sum(q_i) = Q_total

    Therefore:
    Paw_above = (Q_total + sum(V_i/(C_i*R_i))) / sum(1/R_i)

    Energetic calculations are performed above PEEP.
    """
    C = np.asarray(C, dtype=float)
    R = np.asarray(R, dtype=float)
    n_comp = len(C)

    q_total = generate_flow_waveform(vt_l, ti_s, pause_fraction, waveform, dt=dt)
    n_steps = len(q_total)

    V = np.zeros(n_comp)
    comp_energy_positive = np.zeros(n_comp)
    paw_above_series = np.zeros(n_steps)

    global_energy_cmh2o_l = 0.0
    denom = np.sum(1.0 / R)
    inv_CR = 1.0 / (C * R)
    inv_R = 1.0 / R

    for k, qtot in enumerate(q_total):
        paw_above = (qtot + np.sum(V * inv_CR)) / denom
        q_comp = paw_above * inv_R - V * inv_CR

        global_energy_cmh2o_l += paw_above * qtot * dt
        comp_energy_positive += paw_above * np.maximum(q_comp, 0.0) * dt

        V += q_comp * dt
        paw_above_series[k] = paw_above

    global_energy_j = global_energy_cmh2o_l * CMH2O_L_TO_J
    mp_j_min = global_energy_j * rr

    comp_energy_j = comp_energy_positive * CMH2O_L_TO_J
    total_comp_energy_j = np.sum(comp_energy_j)

    ref = comp_energy_j / total_comp_energy_j
    eii = np.std(ref) / np.mean(ref)
    dces = np.max(ref)

    ppeak = peep_cmH2O + np.max(paw_above_series)
    pplat = peep_cmH2O + paw_above_series[-1]
    driving_pressure = pplat - peep_cmH2O
    minute_ventilation = vt_l * rr

    result = {
        "global_energy_J": global_energy_j,
        "MP_J_min": mp_j_min,
        "EII": eii,
        "DCES": dces,
        "Ppeak_cmH2O": ppeak,
        "Pplat_cmH2O": pplat,
        "driving_pressure_cmH2O": driving_pressure,
        "minute_ventilation_L_min": minute_ventilation,
        "end_inspiratory_volume_L": np.sum(V),
    }

    for i in range(n_comp):
        result[f"REF_{i+1}"] = ref[i]
        result[f"comp_energy_J_{i+1}"] = comp_energy_j[i]

    return result


# ---------------------------------------------------------------------
# 3. Optimization utilities
# ---------------------------------------------------------------------

def add_normalized_columns(df, columns, group_col="phenotype"):
    df = df.copy()

    for col in columns:
        norm_col = f"{col}_norm"
        df[norm_col] = np.nan

        for phenotype, sub in df.groupby(group_col):
            x = sub[col].to_numpy(dtype=float)
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)

            if np.isclose(x_max, x_min):
                values = np.zeros_like(x)
            else:
                values = (x - x_min) / (x_max - x_min)

            df.loc[sub.index, norm_col] = values

    return df


def pareto_minimize(points):
    """
    Boolean Pareto mask. All objectives are minimized.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        dominated = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        if np.any(dominated):
            is_pareto[i] = False

    return is_pareto


def add_pareto_flags(df, objective_cols=("MP_J_min", "EII", "DCES")):
    df = df.copy()
    df["pareto_optimal"] = False

    for phenotype, sub in df.groupby("phenotype"):
        mask = pareto_minimize(sub[list(objective_cols)].to_numpy())
        df.loc[sub.index, "pareto_optimal"] = mask

    return df


def prepare_analysis_subset(df, analysis_name, mv_min, mv_max):
    """
    Applies pressure and minute-ventilation feasibility filters, then calculates
    normalized objective values, balanced score, Pareto flags, and strategy summary.
    """
    subset = df[
        (df["Ppeak_cmH2O"] <= 35.0) &
        (df["minute_ventilation_L_min"] >= mv_min) &
        (df["minute_ventilation_L_min"] <= mv_max)
    ].copy()

    subset["analysis_name"] = analysis_name
    subset["MV_min_L_min"] = mv_min
    subset["MV_max_L_min"] = mv_max

    subset = add_normalized_columns(
        subset,
        columns=["MP_J_min", "EII", "DCES"],
        group_col="phenotype",
    )

    subset["balanced_score"] = (
        subset["MP_J_min_norm"] +
        subset["EII_norm"] +
        subset["DCES_norm"]
    )

    subset = add_pareto_flags(subset, objective_cols=("MP_J_min", "EII", "DCES"))
    strategy_summary = summarize_strategies(subset)

    return subset, strategy_summary


def summarize_strategies(df):
    rows = []

    for phenotype, sub in df.groupby("phenotype"):
        strategy_rows = {
            "minimum_MP": sub.loc[sub["MP_J_min"].idxmin()],
            "minimum_EII": sub.loc[sub["EII"].idxmin()],
            "minimum_DCES": sub.loc[sub["DCES"].idxmin()],
            "balanced": sub.loc[sub["balanced_score"].idxmin()],
        }

        for strategy, row in strategy_rows.items():
            out = row.to_dict()
            out["strategy"] = strategy
            rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 4. Main parameter sweep
# ---------------------------------------------------------------------

def run_parameter_sweep(
    fixed_peep_cmH2O=5.0,
    primary_mv_target=7.0,
    primary_mv_tolerance=0.5,
    sensitivity_mv_min=4.0,
    sensitivity_mv_max=12.0,
    include_severe_mixed=True,
    dt=0.002,
):
    phenotypes = get_phenotypes()
    if not include_severe_mixed:
        phenotypes.pop("severe_mixed", None)

    pbw_kg = 70

    vt_ml_per_kg_values = [4, 5, 6, 7, 8]
    rr_values = [10, 15, 20, 25, 30]
    ti_values = [0.6, 1.0, 1.5]
    waveforms = ["square", "decelerating", "sinusoidal"]
    pause_values = [0.0, 0.1, 0.2]

    rows = []

    for phenotype_name, props in phenotypes.items():
        C = props["C"]
        R = props["R"]
        C_total, R_equivalent = calculate_system_mechanics(C, R)

        grid = itertools.product(
            vt_ml_per_kg_values,
            rr_values,
            ti_values,
            waveforms,
            pause_values,
        )

        for vt_ml_kg, rr, ti_s, waveform, pause_fraction in grid:
            vt_l = vt_ml_kg * pbw_kg / 1000.0

            out = simulate_breath(
                C=C,
                R=R,
                vt_l=vt_l,
                rr=rr,
                ti_s=ti_s,
                waveform=waveform,
                pause_fraction=pause_fraction,
                peep_cmH2O=fixed_peep_cmH2O,
                dt=dt,
            )

            rows.append({
                "phenotype": phenotype_name,
                "PBW_kg": pbw_kg,
                "VT_ml_per_kg": vt_ml_kg,
                "VT_ml": vt_l * 1000,
                "RR_min": rr,
                "Ti_s": ti_s,
                "waveform": waveform,
                "pause_fraction": pause_fraction,
                "PEEP_cmH2O": fixed_peep_cmH2O,
                "C_total_L_per_cmH2O": C_total,
                "R_equivalent_cmH2O_s_per_L": R_equivalent,
                **out,
            })

    all_results = pd.DataFrame(rows)

    # Primary analysis: comparable minute ventilation target.
    primary_mv_min = primary_mv_target - primary_mv_tolerance
    primary_mv_max = primary_mv_target + primary_mv_tolerance

    primary_subset, primary_strategy_summary = prepare_analysis_subset(
        all_results,
        analysis_name="primary_MV_target_7.0_L_min",
        mv_min=primary_mv_min,
        mv_max=primary_mv_max,
    )

    # Sensitivity analysis: broad feasible MV range.
    sensitivity_subset, sensitivity_strategy_summary = prepare_analysis_subset(
        all_results,
        analysis_name="sensitivity_MV_4_to_12_L_min",
        mv_min=sensitivity_mv_min,
        mv_max=sensitivity_mv_max,
    )

    # Mechanics summary.
    mechanics_summary = []
    for phenotype_name, props in phenotypes.items():
        C_total, R_equivalent = calculate_system_mechanics(props["C"], props["R"])
        mechanics_summary.append({
            "phenotype": phenotype_name,
            "C_total_L_per_cmH2O": C_total,
            "R_equivalent_cmH2O_s_per_L": R_equivalent,
            "C_values_L_per_cmH2O": ";".join([f"{x:.5f}" for x in props["C"]]),
            "R_values_cmH2O_s_per_L": ";".join([f"{x:.2f}" for x in props["R"]]),
        })
    mechanics_summary = pd.DataFrame(mechanics_summary)

    # Add comparison summary: minimum MP vs balanced, per phenotype and analysis.
    comparison_summary = build_min_mp_vs_balanced_comparison(
        primary_strategy_summary,
        sensitivity_strategy_summary,
    )

    # Save outputs.
    all_results.to_csv("MVtarget_primary_all_simulation_results.csv", index=False)
    primary_subset.to_csv("MVtarget_primary_feasible_simulation_results.csv", index=False)
    primary_strategy_summary.to_csv("MVtarget_primary_strategy_summary.csv", index=False)

    sensitivity_subset.to_csv("MVtarget_sensitivity_feasible_simulation_results.csv", index=False)
    sensitivity_strategy_summary.to_csv("MVtarget_sensitivity_strategy_summary.csv", index=False)

    mechanics_summary.to_csv("MVtarget_mechanics_summary.csv", index=False)
    comparison_summary.to_csv("MVtarget_minMP_vs_balanced_comparison.csv", index=False)

    return {
        "all_results": all_results,
        "primary_subset": primary_subset,
        "primary_strategy_summary": primary_strategy_summary,
        "sensitivity_subset": sensitivity_subset,
        "sensitivity_strategy_summary": sensitivity_strategy_summary,
        "mechanics_summary": mechanics_summary,
        "comparison_summary": comparison_summary,
    }


def build_min_mp_vs_balanced_comparison(primary_strategy_summary, sensitivity_strategy_summary):
    rows = []

    for analysis_label, df in [
        ("primary_MV_target_7.0_L_min", primary_strategy_summary),
        ("sensitivity_MV_4_to_12_L_min", sensitivity_strategy_summary),
    ]:
        for phenotype, sub in df.groupby("phenotype"):
            min_mp = sub[sub["strategy"] == "minimum_MP"].iloc[0]
            balanced = sub[sub["strategy"] == "balanced"].iloc[0]

            rows.append({
                "analysis_name": analysis_label,
                "phenotype": phenotype,
                "MP_minimum_MP": min_mp["MP_J_min"],
                "MP_balanced": balanced["MP_J_min"],
                "MP_relative_change_percent": 100.0 * (balanced["MP_J_min"] - min_mp["MP_J_min"]) / min_mp["MP_J_min"],
                "EII_minimum_MP": min_mp["EII"],
                "EII_balanced": balanced["EII"],
                "EII_relative_change_percent": 100.0 * (balanced["EII"] - min_mp["EII"]) / min_mp["EII"],
                "DCES_minimum_MP": min_mp["DCES"],
                "DCES_balanced": balanced["DCES"],
                "DCES_relative_change_percent": 100.0 * (balanced["DCES"] - min_mp["DCES"]) / min_mp["DCES"],
                "MV_minimum_MP": min_mp["minute_ventilation_L_min"],
                "MV_balanced": balanced["minute_ventilation_L_min"],
                "minimum_MP_setting": f"VT {min_mp['VT_ml_per_kg']} ml/kg, RR {min_mp['RR_min']}/min, Ti {min_mp['Ti_s']} s, {min_mp['waveform']}, pause {min_mp['pause_fraction']}",
                "balanced_setting": f"VT {balanced['VT_ml_per_kg']} ml/kg, RR {balanced['RR_min']}/min, Ti {balanced['Ti_s']} s, {balanced['waveform']}, pause {balanced['pause_fraction']}",
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    outputs = run_parameter_sweep(
        fixed_peep_cmH2O=5.0,
        primary_mv_target=7.0,
        primary_mv_tolerance=0.5,
        sensitivity_mv_min=4.0,
        sensitivity_mv_max=12.0,
        include_severe_mixed=True,
        dt=0.002,
    )

    print("Scaled C/R fixed-PEEP simulation completed.")
    print("Primary MV target: 7.0 L/min ± 0.5 L/min")
    print("Sensitivity MV range: 4.0–12.0 L/min")
    print()
    print(f"All scenarios: {len(outputs['all_results'])}")
    print(f"Primary feasible scenarios: {len(outputs['primary_subset'])}")
    print(f"Sensitivity feasible scenarios: {len(outputs['sensitivity_subset'])}")
    print(f"Primary Pareto-optimal scenarios: {outputs['primary_subset']['pareto_optimal'].sum()}")
    print(f"Sensitivity Pareto-optimal scenarios: {outputs['sensitivity_subset']['pareto_optimal'].sum()}")
    print()
    print("Mechanics summary:")
    print(outputs["mechanics_summary"].to_string(index=False))
    print()
    print("Primary strategy summary:")
    print(outputs["primary_strategy_summary"][
        [
            "phenotype", "strategy", "VT_ml_per_kg", "RR_min", "Ti_s", "waveform",
            "pause_fraction", "PEEP_cmH2O", "MP_J_min", "EII", "DCES",
            "Ppeak_cmH2O", "Pplat_cmH2O", "driving_pressure_cmH2O",
            "minute_ventilation_L_min", "pareto_optimal"
        ]
    ].to_string(index=False))
    print()
    print("Minimum MP vs balanced comparison:")
    print(outputs["comparison_summary"].to_string(index=False))
