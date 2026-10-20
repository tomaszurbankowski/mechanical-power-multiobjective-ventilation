# Mechanical Power Multi-Objective Ventilation Optimization

This repository contains the Python code and generated numerical output files for the manuscript:

**Global Mechanical Power and Regional Energy Distribution in Mechanically Heterogeneous Lungs: An In Silico Multi-Objective Optimization Study**

## Overview

This project implements a deterministic in silico simulation framework for evaluating ventilation settings in mechanically heterogeneous lungs.

The model uses a six-compartment parallel resistance–compliance structure with four predefined mechanical configurations. Ventilator settings are systematically varied under volume-controlled ventilation. In the reference simulations, prescribed PEEP is fixed at 5 cmH₂O. Each simulated scenario is evaluated using global and regional energy-based metrics. Additional within-model checks assess passive expiration, numerical resolution, and prescribed-PEEP pressure offsets.

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

## Descriptive characterization of the reference primary feasible set

A separate descriptive analysis summarizes the original reference primary feasible dataset of 324 scenarios (81 per mechanical configuration). Unlike the passive-expiration extension, this analysis uses the original inspiration-only simulations with compartmental volumes reset before each breath.

The analysis reports configuration counts, frequencies of retained ventilator settings (including the joint tidal-volume–respiratory-rate combinations), and the minimum, 25th percentile, median, 75th percentile, and maximum of MP, EII, and DCES within each configuration.

The analysis is implemented in `summarize_primary_feasible_scenarios.py`. The numerical summaries are provided in:

- `primary_feasible_setting_counts.csv` — counts and frequencies of retained ventilation settings, with denominators and percentages.
- `primary_feasible_metric_summary.csv` — distributional summaries of MP, EII, and DCES for each mechanical configuration.

These summaries describe the predefined simulation grid rather than patient frequencies or sampling uncertainty. No new simulations were performed and no strategy selections were changed. The script also generates Supplementary Table S4 in Word format, which is not tracked in this repository.

## Supplementary numerical-resolution analysis: Time step and parameter grid

A separate numerical sensitivity analysis evaluates both inspiratory integration time-step resolution and the spacing of the prescribed ventilator settings in the **original linear inspiration-only, volume-reset model**. It is distinct from the passive-expiration sensitivity analysis.

Three successive midpoint refinements (G1–G3) were compared with the original parameter grid (G0). All grids retained the original parameter ranges, three categorical flow waveforms, prescribed PEEP, and primary feasibility constraints. The respective grids contained 324, 2700, 34992, and 454308 feasible combinations. Comparisons across grids used a common 0.0005-s time step, whereas additional tests examined temporal-resolution sensitivity.

The direction of the model-derived balanced-versus-minimum-MP trade-off persisted across the tested grids, but some selected respiratory rates and pause fractions changed. A separate lower-minute-ventilation-boundary check further demonstrated that the tested grids did not establish continuous global optima. These are numerical, within-model findings, not evidence of clinical superiority, convergence of Pareto-front membership, or robustness of the passive-expiration extension to grid refinement.

The analysis is implemented in `assess_grid_time_sensitivity.py`. The numerical outputs are:

- `numerical_grid_definition.csv` — grid levels, spacing, and full/feasible scenario counts.
- `numerical_grid_summary.csv` — objective ranges and balanced-versus-minimum-MP changes for each grid and configuration.
- `numerical_grid_selected_strategies.csv` — grid-specific and fixed-reference-normalization strategy selections.
- `numerical_time_step_summary.csv` — maximum absolute and relative differences at matched settings and different time steps.
- `numerical_time_step_selected_strategies.csv` — selected settings across the tested time steps.
- `numerical_boundary_check.csv` — directly evaluated settings on the 6.5 L/min feasibility boundary.

The script can also generate a verification JSON file and Supplementary Table S5 in Word format; these generated files are not tracked in the repository. The original model, primary data, and published numerical outputs are not overwritten.

## Supplementary check: Prescribed-PEEP pressure offsets and feasibility

A separate analysis examines prescribed PEEP offsets of 5, 10, 15, and 20 cmH₂O in the **original linear inspiration-only, volume-reset model**. The original 324 primary setting combinations are re-evaluated at each PEEP (1296 scenario evaluations), keeping compartmental resistance and compliance, flow waveforms, the 0.002-s time step, relative initial volumes, and above-PEEP energy definitions unchanged. This check is distinct from the passive-expiration and refined-parameter-grid extensions.

In this fixed-mechanics model, changing prescribed PEEP shifts absolute airway pressure but does not affect matched-setting airway pressure above PEEP, relative compartmental flows and volumes, or the modeled MP, REF, EII, and DCES. The peak-airway-pressure feasibility constraint of 35 cmH₂O is reapplied at each PEEP value, and minimum-MP and balanced strategies are reselected within each retained feasible subset with normalization recalculated. Feasible counts across all four mechanical configurations are 324, 324, 315, and 261 at PEEP 5, 10, 15, and 20 cmH₂O, respectively. The selected minimum-MP and balanced settings remain unchanged in this check.

This **pressure-offset/feasibility check is not physiological PEEP optimization**. It does not model PEEP-dependent recruitment, closure, overdistension, changes in compartmental mechanics, or patient-specific absolute lung volumes. The tested PEEP values are illustrative offsets rather than clinical recommendations.

The analysis is implemented in `assess_peep_offset_sensitivity.py`. Its numerical outputs are:

- `peep_offset_scenarios.csv` — all 1296 matched scenario/PEEP evaluations, feasibility flags, and invariant-metric checks.
- `peep_offset_summary.csv` — feasibility counts, pressure ranges, and objective normalization bounds for each configuration and PEEP value.
- `peep_offset_selected_strategies.csv` — reselected minimum-MP and balanced settings for each configuration and PEEP value.

The script can also generate a verification JSON file and Supplementary Table S6 in Word format; neither output is tracked in this repository. The original simulation model and CSV files remain unchanged.

## Supplementary analysis: Compartmental energy, power, and energy accounting

This analysis uses the **original linear inspiration-only, volume-reset model**, retaining the original 2,700 scenarios and the same primary (324) and broader-minute-ventilation (1,944) feasible sets. It does not change the model equations, reference energy metrics, balanced score, or previous analyses. It is separate from the passive-expiration, PEEP-offset, and refined-parameter-grid studies.

The original compartmental quantity E_i is **positive-inflow work at a branch inlet per inspiration**, not net signed work or tissue dissipation. The corresponding compartmental positive-inflow power is MP_i(+) = RR × E_i (J/min). The analysis also reports the largest absolute compartmental energy per breath and largest compartmental power across branches, as distinct from DCES, which is a *normalized fraction*.

The additional energy-accounting diagnostics separately report signed inlet work, outward-work magnitude, branch-resistive dissipation, and elastic-energy change relative to the prescribed-PEEP equilibrium. The finite-step Euler energy-balance term is identified separately as a numerical, not physiological, contribution. Lumped branch-resistive dissipation cannot be equated with experimentally validated tissue-level dissipation or injury risk.

Converting all six compartmental energies to powers using the same RR within a given scenario leaves REF, EII, and DCES algebraically unchanged. However, **the original balanced selections increased maximum absolute compartmental positive-inflow power** compared with minimum global MP by **32.12%, 2.54%, 4.45%, and 5.36%**, respectively, in the compliance-dominant, resistance-dominant, mixed, and severe-mixed configurations of the primary set. The corresponding increases in the broader-MV set were 72.26%, 8.56%, 4.45%, and 10.58%. Thus, a decrease in inequality or dominant *share* is not proof of reduced maximum *absolute* regional work rate.

The analysis is implemented in `assess_compartment_energy_power.py`. Its numerical outputs are:

- `compartment_energy_power_scenarios.csv` — original settings and metrics, six positive-inflow powers, aggregate energy components, and feasibility flags for all 2,700 scenarios.
- `compartment_energy_power_components.csv` — six branch-level energy/work components and rates for every scenario (16,200 rows).
- `compartment_energy_power_selected_strategies.csv` — original minimum-MP and balanced settings, plus all tied minimum-maximum-energy and minimum-maximum-power solutions.
- `compartment_energy_power_summary.csv` — configuration-level ranges and tie counts.
- `compartment_energy_power_comparison.csv` — balanced-versus-minimum-MP comparisons in both feasible sets.

The script can additionally generate a verification JSON and Supplementary Table S7 in Word format; these optional outputs are not tracked in this repository. It checks agreement with the original reference results and does not overwrite original model inputs.

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

**Descriptive characterization of the primary feasible set**

- `summarize_primary_feasible_scenarios.py`
- `primary_feasible_setting_counts.csv`
- `primary_feasible_metric_summary.csv`

**Temporal and parameter-grid sensitivity analysis**

- `assess_grid_time_sensitivity.py`
- `numerical_grid_definition.csv`
- `numerical_grid_summary.csv`
- `numerical_grid_selected_strategies.csv`
- `numerical_time_step_summary.csv`
- `numerical_time_step_selected_strategies.csv`
- `numerical_boundary_check.csv`

**Prescribed-PEEP offset and feasibility analysis**

- `assess_peep_offset_sensitivity.py`
- `peep_offset_scenarios.csv`
- `peep_offset_summary.csv`
- `peep_offset_selected_strategies.csv`

**Compartmental energy and power analysis**

- `assess_compartment_energy_power.py`
- `compartment_energy_power_scenarios.csv`
- `compartment_energy_power_components.csv`
- `compartment_energy_power_selected_strategies.csv`
- `compartment_energy_power_summary.csv`
- `compartment_energy_power_comparison.csv`

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

To reproduce the reference primary feasible-set counts and distributional summaries from the original CSV:

`python summarize_primary_feasible_scenarios.py --full-csv MVtarget_primary_all_simulation_results.csv --output-dir . --no-docx`

Omit `--no-docx` to also generate Supplementary Table S4 (Word). The input CSV files remain unchanged; this script only summarizes existing simulations. It additionally writes a locally generated verification JSON file that need not be tracked.

To reproduce the additional temporal and parameter-grid resolution analysis using the original model and primary feasible dataset:

`python assess_grid_time_sensitivity.py --output-dir . --no-docx`

Omit `--no-docx` to also generate Supplementary Table S5 (Word). The script additionally writes a local verification JSON file. The grid-resolution comparison includes up to 454308 feasible combinations and may require substantial computation time; the original model and input CSV are not modified. The tested grids do not certify continuous global optimality.

To reproduce the prescribed-PEEP pressure-offset and feasibility analysis using the original model and primary feasible CSV:

`python assess_peep_offset_sensitivity.py --output-dir . --no-docx`

Omit `--no-docx` to also generate Supplementary Table S6 (Word). The script re-evaluates the original primary settings at four illustrative PEEP values, writes a local verification JSON file, and does not modify the source model or reference CSV. Existing PEEP-offset output files in the specified output directory are replaced upon rerun. This is not physiological PEEP optimization.

To reproduce the supplementary compartmental work/power and energy-accounting analysis from the original 2,700-scenario CSV and model script:

`python assess_compartment_energy_power.py --output-dir . --no-docx`

Run this command in the repository root, where `MVtarget_primary_all_simulation_results.csv` and `multiobjective_ventilation_optimization_scaled_CR_MVtarget.py` are already present. The script reconstructs both feasible subsets and verifies the original model outputs without modifying those inputs. Omit `--no-docx` to also generate Supplementary Table S7 (Word). A verification JSON is also generated locally. Rerunning into the same output directory replaces this supplementary analysis's generated output files.

## Scope and limitations

The model is intended as a mechanistic, hypothesis-generating framework rather than a patient-specific simulator.

Regional energy fractions describe the relative distribution of positive compartmental inspiratory inflow work. They do not directly measure tissue stress, strain, or injurious energy dissipation.

The predefined mechanical configurations are not derived from individual patient data. Consequently, the reported optimization results should not be interpreted as validated clinical recommendations or evidence that a particular strategy reduces ventilator-induced lung injury.

The additional passive-expiration extension remains linear and omits airway closure, expiratory flow limitation, direction-dependent resistance, and separate expiratory-circuit resistance. Its equivalent intrinsic-pressure descriptor is an analytical model quantity rather than a clinical measurement.

The prescribed-PEEP check varies only a pressure offset within fixed compartmental mechanics. Its matched-setting invariance is a structural property of the equations, not evidence of physiological insensitivity to PEEP. It cannot identify a clinically optimal or protective PEEP setting.

The additional compartmental-power analysis distinguishes relative allocation from absolute work rate. Lower REF-based EII and DCES under balanced selection did **not** correspond to lower maximum absolute compartmental positive-inflow power. These model-derived rates and the separately reported branch-resistive dissipation are not validated measures of tissue injury.
