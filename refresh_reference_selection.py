#!/usr/bin/env python3
"""Refresh Pareto flags and reference strategy selections without rerunning breaths.

Run with the corrected main simulation script in the repository root:
    python refresh_reference_selection.py --input-dir . --output-dir selection_corrected
Use --output-dir . only to intentionally replace the derived CSVs in the repository.

All scenario measurements, normalization values, and feasibility memberships are
preserved as their original CSV strings. Only Pareto flags and the representative
rows in the compact strategy summaries can change. All tied minima are exported.
Inherited REFERENCE Pareto flags in S2/S3 exports are synchronized when present;
no periodic-state Pareto front is computed and no periodic metric is altered.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

MODEL = 'multiobjective_ventilation_optimization_scaled_CR_MVtarget.py'
KEYS = ('phenotype','VT_ml_per_kg','RR_min','Ti_s','waveform','pause_fraction','PEEP_cmH2O')
ORDER = ('compliance_dominant','resistance_dominant','mixed','severe_mixed')
NUM_META = ('objective_minimum','n_tied_minima','representative')


def read_csv(path: Path) -> tuple[list[str], list[dict[str,str]]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames or not rows:
            raise ValueError(f'Empty CSV: {path.name}')
        return reader.fieldnames, rows


def key(row: Any) -> tuple:
    return tuple(str(row[k]) if k in ('phenotype','waveform') else float(row[k]) for k in KEYS)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_model(path: Path):
    spec = importlib.util.spec_from_file_location('corrected_reference_model', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot import {path}')
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    if not hasattr(model, 'strategy_minimizers'):
        raise ValueError('Use the corrected main model with strategy_minimizers().')
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('selection_corrected'))
    parser.add_argument('--model-file', type=Path)
    args = parser.parse_args()
    try:
        src, out = args.input_dir.resolve(), args.output_dir.resolve()
        model_path = args.model_file or src/MODEL
        model = load_model(model_path)
        pending: dict[str,tuple[list[str],list[dict]]] = {}
        report: dict[str,Any] = {
            'rtol': model.SELECTION_RTOL, 'atol': model.SELECTION_ATOL,
            'scenario_values_recalculated': False, 'counts': [],
            'reference_flag_copies': {}, 'source_sha256': {},
            'note': 'Numerical selection correction only. No model trajectories or energy definitions are modified.'}
        summary_rows, changed_rows, primary_flags = [], [], {}
        for prefix, n_expected in (('primary',324), ('sensitivity',1944)):
            fname = f'MVtarget_{prefix}_feasible_simulation_results.csv'
            sname = f'MVtarget_{prefix}_strategy_summary.csv'
            path = src/fname
            fields, rows = read_csv(path)
            sfields, old_selected = read_csv(src/sname)
            report['source_sha256'][fname] = hashlib.sha256(path.read_bytes()).hexdigest()
            df = pd.read_csv(path, float_precision='round_trip')
            if len(df) != n_expected or df.phenotype.nunique() != 4:
                raise ValueError(f'Unexpected reference dataset: {fname}')
            if df.duplicated(list(KEYS)).any():
                raise ValueError(f'Duplicate scenario key: {fname}')
            refreshed = model.add_pareto_flags(df)
            minima = model.strategy_minimizers(refreshed)
            reps = model.summarize_strategies(refreshed)
            raw_by_key = {key(row):row for row in rows}
            flags = {key(row):bool(row.pareto_optimal) for _,row in refreshed.iterrows()}
            new_rows = [dict(row,pareto_optimal=str(flags[key(row)])) for row in rows]
            # Only the Boolean flag is changed: verify every remaining CSV cell.
            if any(a[k] != b[k] for a,b in zip(rows,new_rows) for k in fields if k!='pareto_optimal'):
                raise AssertionError('A scenario measurement changed during export')
            pending[fname] = fields,new_rows
            if prefix == 'primary':
                primary_flags = flags
            rep_keys = {}
            new_selected=[]
            old_lookup={(r['phenotype'],r['strategy']):r for r in old_selected}
            for _, row in reps.iterrows():
                raw = dict(raw_by_key[key(row)])
                raw['pareto_optimal'] = str(flags[key(row)])
                raw['strategy'] = row.strategy
                new_selected.append({c:raw[c] for c in sfields})
                rep_keys[(row.phenotype,row.strategy)] = key(row)
                old=old_lookup[(row.phenotype,row.strategy)]
                if row.strategy in ('minimum_MP','balanced') and any(old[c]!=raw[c] for c in sfields):
                    raise AssertionError(f'Minimum-MP/balanced row changed: {prefix}/{row.phenotype}')
                if not flags[key(row)]:
                    raise AssertionError('A compact-summary representative is dominated')
                if key(old)!=key(row):
                    changed_rows.append({'analysis_set':prefix,'phenotype':row.phenotype,'strategy':row.strategy,
                                         **{f'old_{k}':old[k] for k in KEYS[1:]},
                                         **{f'new_{k}':raw[k] for k in KEYS[1:]}})
            pending[sname]=sfields,new_selected
            tie_fields=fields+['strategy','minimized_objective','objective_minimum','n_tied_minima','representative']
            ties=[]
            for _,row in minima.iterrows():
                raw=dict(raw_by_key[key(row)])
                raw.update(pareto_optimal=str(flags[key(row)]),strategy=row.strategy,
                           minimized_objective=row.minimized_objective,
                           objective_minimum=format(float(row.objective_minimum),'.17g'),
                           n_tied_minima=str(int(row.n_tied_minima)),representative=str(bool(row.representative)))
                ties.append(raw)
            pending[f'MVtarget_{prefix}_all_minimizers.csv']=tie_fields,ties
            for phenotype in ORDER:
                sub=refreshed[refreshed.phenotype.eq(phenotype)]
                row={'analysis_set':prefix,'phenotype':phenotype,'n_feasible':len(sub),
                     'n_pareto':int(sub.pareto_optimal.sum())}
                for st in ('minimum_MP','minimum_EII','minimum_DCES','balanced'):
                    row[f'{st}_n_tied_minima']=len(minima[minima.phenotype.eq(phenotype)&minima.strategy.eq(st)])
                summary_rows.append(row)
                old_count=int(df.loc[df.phenotype.eq(phenotype),'pareto_optimal'].sum())
                report['counts'].append(dict(row,previous_n_pareto=old_count))
        pending['reference_selection_summary.csv']=list(summary_rows[0]),summary_rows
        # These exports carry the original reference flag, not a recomputed S2/S3 front.
        for fname in ('global_indices_all_minimizers.csv','expiration_primary_scenarios.csv','expiration_selected_strategies.csv'):
            path=src/fname
            if not path.is_file():
                report['reference_flag_copies'][fname]='Not present; skipped'
                continue
            fields,rows=read_csv(path)
            new=[]; changes=0
            for row in rows:
                if key(row) not in primary_flags:
                    raise ValueError(f'Unknown reference key in {fname}')
                fresh=str(primary_flags[key(row)])
                changes+=row['pareto_optimal'].lower()!=fresh.lower()
                new.append(dict(row,pareto_optimal=fresh))
            if any(a[k]!=b[k] for a,b in zip(rows,new) for k in fields if k!='pareto_optimal'):
                raise AssertionError(f'A non-flag value changed in {fname}')
            pending[fname]=fields,new
            report['reference_flag_copies'][fname]={'rows':len(rows),'reference_flags_changed':int(changes),
                                                   'all_other_cells_unchanged':True}
        out.mkdir(parents=True,exist_ok=True)
        for fname,(fields,rows) in pending.items():
            write_csv(out/fname,fields,rows)
        report['regional_representative_changes']=changed_rows
        report['minimum_MP_and_balanced_all_original_cells_unchanged']=True
        report['feasible_measurements_all_original_cells_unchanged']=True
        (out/'reference_selection_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print('No breath dynamics rerun; minimum-MP and balanced unchanged.')
        print(f'Wrote corrected post-processing outputs to {out}')
        return 0
    except (OSError,ValueError,AssertionError,KeyError,ImportError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
