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

The analysis is implemented in `compare_simple_indices_R1_5.py`, and the corresponding numerical results are provided in:

- `R1_5_minimizer_summary.csv` — summary of minimum values, tied solutions, and associated energy metrics for each mechanical configuration.
- `R1_5_all_minimizers.csv` — detailed results for all minimizing scenarios.

The comparison evaluates differences between model-based selection criteria. It does not establish clinical superiority, predict lung injury, or validate the selected ventilator settings in patients.

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

- `compare_simple_indices_R1_5.py`
- `R1_5_minimizer_summary.csv`
- `R1_5_all_minimizers.csv`

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

`python compare_simple_indices_R1_5.py --input-csv MVtarget_primary_feasible_simulation_results.csv --output-dir .`

The supplementary script generates the comparison outputs without modifying the original simulation data.

## Scope and limitations

The model is intended as a mechanistic, hypothesis-generating framework rather than a patient-specific simulator.

Regional energy fractions describe the relative distribution of positive compartmental inspiratory inflow work. They do not directly measure tissue stress, strain, or injurious energy dissipation.

The predefined mechanical configurations are not derived from individual patient data. Consequently, the reported optimization results should not be interpreted as validated clinical recommendations or evidence that a particular strategy reduces ventilator-induced lung injury.
