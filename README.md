# Mechanical Power Multi-Objective Ventilation Optimization

This repository contains the Python code and generated numerical output files for the manuscript:

**Mechanical Power as a Co-Objective in Protective Ventilation: An In Silico Multi-Objective Optimization Study in Mechanically Heterogeneous Lungs**

## Overview

This project implements a deterministic in-silico simulation framework for evaluating ventilation settings in mechanically heterogeneous lungs.

The model uses a six-compartment parallel lung structure with predefined compliance–resistance heterogeneity. Ventilator settings are systematically varied under volume-controlled ventilation, and each simulated scenario is evaluated using global and regional energy-based metrics.

The main objectives are:

- global inspiratory mechanical power,
- Energy Inequality Index,
- Dominant Compartment Energy Share.

The analysis compares minimum mechanical power selection with balanced multi-objective optimization and Pareto-front analysis.

## Repository contents

```text
multiobjective_ventilation_optimization_scaled_CR_MVtarget.py
MVtarget_primary_all_simulation_results.csv
MVtarget_primary_feasible_simulation_results.csv
MVtarget_primary_strategy_summary.csv
MVtarget_sensitivity_feasible_simulation_results.csv
MVtarget_sensitivity_strategy_summary.csv
MVtarget_mechanics_summary.csv
MVtarget_minMP_vs_balanced_comparison.csv
requirements.txt
README.md
