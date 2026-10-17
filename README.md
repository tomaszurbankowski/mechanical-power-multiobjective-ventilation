# Mechanical Power Multi-Objective Ventilation Optimization

This repository contains the Python code and generated numerical output files for the manuscript:

**Global Mechanical Power and Regional Energy Distribution in Mechanically Heterogeneous Lungs: An In Silico Multi-Objective Optimization Study**

## Overview

This project implements a deterministic in silico simulation framework for evaluating ventilation settings in mechanically heterogeneous lungs.

The model uses a six-compartment parallel resistance–compliance structure with four predefined mechanical configurations. Ventilator settings are systematically varied under volume-controlled ventilation with PEEP fixed at 5 cmH₂O. Each simulated scenario is evaluated using global and regional energy-based metrics.

The main optimization objectives are:

- Global inspiratory mechanical power (MP)
- Energy Inequality Index (EII)
- Dominant Compartment Energy Share (DCES)

The analysis compares individual-objective minimization with balanced multi-objective selection and Pareto-front analysis.

The primary analysis includes 324 feasible scenarios within a minute-ventilation range of 6.5–7.5 L/min and a peak airway pressure limit of 35 cmH₂O. A sensitivity analysis examines a broader minute-ventilation range of 4.0–12.0 L/min.

## Supplementary analysis: Comparison with simpler global indices

An additional within-model analysis compares the original optimization strategies with simpler global indices:

- Equilibrium driving pressure (ΔP_eq = VT/Ctotal)
- Composite index (4ΔP_eq + RR)
- Peak airway pressure (Ppeak)

The comparison uses the original primary feasible dataset of 324 scenarios without modifying the underlying simulation model, energy definitions, or feasibility criteria.

All numerically tied minima are retained to avoid arbitrary selection among equivalent solutions.

The analysis is implemented in `compare_global_indices.py`, and the corresponding numerical results are provided in:

- `global_indices_minimizer_summary.csv` — summary of minimum values, tied solutions, and associated energy metrics for each mechanical configuration.
- `global_indices_all_minimizers.csv` — detailed results for all minimizing scenarios.

The comparison evaluates differences between model-based selection criteria. It does not establish clinical superiority, predict lung injury, or validate the selected ventilator settings in patients.

## Supplementary sensitivity analysis: Passive expiration and incomplete emptying

An additional within-model sensitivity analysis assessed the impact of finite passive expiration and breath-to-breath volume history on the original 324 primary feasible scenarios. Unlike the reference inspiration-only analysis, this extension carries residual end-expiratory compartment volumes into subsequent breaths and iterates the respiratory cycle to a periodic steady state.

Expiration is modeled as passive exponential emptying into the prescribed external PEEP, with inspiratory and expiratory resistance assumed equal in each compartment. The original inspiratory flow waveforms, numerical time step, energy definitions, and feasibility criteria are retained. Expiratory resistive dissipation is evaluated separately and is not added to inspiratory mechanical power.

At matched settings, model-derived inspiratory MP increased by up to 14.5%. Minimum-MP settings were unchanged across all four mechanical configurations, whereas the balanced settings changed in three. These observations describe sensitivity to the assumed expiratory dynamics; they are not a validation of patient-specific auto-PEEP or lung-protective strategies.

The analysis is implemented in `assess_expiration_sensitivity.py`. Its numerical results are provided in:

- `expiration_primary_scenarios.csv` — original and periodic-state metrics, residual volumes, and diagnostic measures for all 324 scenarios.
- `expiration_selected_strategies.csv` — original and reselected minimum-MP and balanced strategies.
- `expiration_summary.csv` — configuration-level summaries of the effects of finite expiration.

## Repository contents

**Main simulation code**

- `multiobjective_ventilation_optimization_scaled_CR_MVtarget.py`

**Main simulation outputs**

- `MVtarget_primary_all_simulation_results.csv`
- `MVtarget_primary_feasible_simulation_results.csv`
- `MVtarget_primary_strategy_summary.csv`
- `MVtarget_sensitivity_feasible_simulation_results.csv`
- `MVtarget_sensitivity_strategy_summary.csv`
- `MVtarget_mechanics_summary.csv`
- `MVtarget_minMP_vs_balanced_comparison.csv`

**Supplementary comparison**

- `compare_global_indices.py`
- `global_indices_minimizer_summary.csv`
- `global_indices_all_minimizers.csv`

**Passive-expiration sensitivity analysis**

- `assess_expiration_sensitivity.py`
- `expiration_primary_scenarios.csv`
- `expiration_selected_strategies.csv`
- `expiration_summary.csv`

**Documentation and dependencies**

- `requirements.txt`
- `README.md`
- `LICENSE`

## Reproducibility

The simulations and supplementary analyses were implemented in Python using NumPy and pandas.

Install the required dependencies:

`pip install -r requirements.txt`

To reproduce the original simulations:

`python multiobjective_ventilation_optimization_scaled_CR_MVtarget.py`

To reproduce the supplementary comparison using the existing primary feasible dataset:

`python compare_global_indices.py --input-csv MVtarget_primary_feasible_simulation_results.csv --output-dir .`

The supplementary script generates the comparison outputs without modifying the original simulation data.

To reproduce the passive-expiration sensitivity analysis using the existing primary feasible dataset:

`python assess_expiration_sensitivity.py --output-dir .`

This extension reads the original model and primary feasible CSV without overwriting them. It also generates a verification JSON file and the editable Supplementary Table S3 (Word); these additional files need not be tracked in the repository.

## Scope and limitations

The model is intended as a mechanistic, hypothesis-generating framework rather than a patient-specific simulator.

Regional energy fractions describe the relative distribution of positive compartmental inspiratory inflow work. They do not directly measure tissue stress, strain, or injurious energy dissipation.

The predefined mechanical configurations are not derived from individual patient data. Consequently, the reported optimization results should not be interpreted as validated clinical recommendations or evidence that a particular strategy reduces ventilator-induced lung injury.

The additional passive-expiration extension remains linear and omits airway closure, expiratory flow limitation, direction-dependent resistance, and separate expiratory-circuit resistance. Its equivalent intrinsic-pressure descriptor is an analytical model quantity rather than a clinical measurement.
