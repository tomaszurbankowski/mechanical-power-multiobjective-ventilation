#!/usr/bin/env python3
"""Regression checks for numerical Pareto comparisons and reference tied minima.
Run from the repository root: python test_selection_precision.py
"""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('model',ROOT/'multiobjective_ventilation_optimization_scaled_CR_MVtarget.py')
MODEL=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
METRICS=['MP_J_min','EII','DCES']
SETTINGS=['VT_ml_per_kg','RR_min','Ti_s','waveform','pause_fraction']
ORDER=['compliance_dominant','resistance_dominant','mixed','severe_mixed']


class NumericalSelectionTests(unittest.TestCase):
    def test_roundoff_does_not_protect_high_power_point(self):
        x=np.array([[3.568312,.3639129134268181,.25622795394781395],
                    [4.460390,.36391291342681786,.2562279539478139]])
        np.testing.assert_array_equal(MODEL.pareto_minimize(x),[True,False])
        np.testing.assert_array_equal(MODEL.pareto_minimize(x,0,0),[True,True])

    def test_genuine_tradeoff_is_retained(self):
        x=np.array([[3.,.31,.24],[4.,.25,.22],[5.,.35,.29]])
        np.testing.assert_array_equal(MODEL.pareto_minimize(x),[True,True,False])

    def test_duplicates_and_empty_input(self):
        np.testing.assert_array_equal(MODEL.pareto_minimize([[1.,2.],[1.,2.]]),[True,True])
        self.assertEqual(len(MODEL.pareto_minimize(np.empty((0,3)))),0)

    def test_invalid_input_rejected(self):
        for x in ([1.,2.],[[1.,float('nan')]],[[1.,float('inf')]]):
            with self.assertRaises(ValueError): MODEL.pareto_minimize(x)
        with self.assertRaises(ValueError): MODEL.pareto_minimize([[1.,2.]],rtol=-1)

    def test_expected_fronts_and_tolerance_sensitivity(self):
        for prefix in ('primary','sensitivity'):
            df=pd.read_csv(ROOT/f'MVtarget_{prefix}_feasible_simulation_results.csv',float_precision='round_trip')
            reference=None
            for rt,at in ((1e-12,1e-14),(1e-10,1e-12),(1e-8,1e-10),(1e-6,1e-8)):
                fresh=MODEL.add_pareto_flags(df,rtol=rt,atol=at)
                counts=[int(fresh.loc[fresh.phenotype.eq(p),'pareto_optimal'].sum()) for p in ORDER]
                self.assertEqual(counts,[7,4,4,5])
                if reference is None: reference=fresh.pareto_optimal.to_numpy()
                else: np.testing.assert_array_equal(reference,fresh.pareto_optimal)
            np.testing.assert_array_equal(reference,df.pareto_optimal)

    def test_order_invariance(self):
        df=pd.read_csv(ROOT/'MVtarget_sensitivity_feasible_simulation_results.csv',float_precision='round_trip')
        rng=np.random.default_rng(20261023)
        for _,group in df.groupby('phenotype'):
            x=group[METRICS].to_numpy()
            expected=MODEL.pareto_minimize(x)
            for _ in range(4):
                perm=rng.permutation(len(x))
                inverse=np.argsort(perm)
                np.testing.assert_array_equal(expected,MODEL.pareto_minimize(x[perm])[inverse])

    def test_tie_counts_and_representatives(self):
        for prefix,expected_ties,expected_rr in (('primary',3,25),('sensitivity',18,15)):
            df=pd.read_csv(ROOT/f'MVtarget_{prefix}_feasible_simulation_results.csv',float_precision='round_trip')
            ties=MODEL.strategy_minimizers(df)
            reps=MODEL.summarize_strategies(df)
            self.assertEqual(len(reps),16)
            self.assertTrue(reps.pareto_optimal.all())
            for p in ORDER:
                for strategy in ('minimum_EII','minimum_DCES'):
                    group=ties[ties.phenotype.eq(p)&ties.strategy.eq(strategy)]
                    self.assertEqual(len(group),expected_ties)
                    self.assertEqual(int(group.representative.sum()),1)
                    r=group[group.representative].iloc[0]
                    self.assertEqual(r.VT_ml_per_kg,4)
                    self.assertEqual(r.RR_min,expected_rr)
            stored=pd.read_csv(ROOT/f'MVtarget_{prefix}_strategy_summary.csv',float_precision='round_trip')
            for p in ORDER:
                for st in ('minimum_MP','minimum_EII','minimum_DCES','balanced'):
                    a=reps[reps.phenotype.eq(p)&reps.strategy.eq(st)].iloc[0]
                    b=stored[stored.phenotype.eq(p)&stored.strategy.eq(st)].iloc[0]
                    self.assertEqual(tuple(a[SETTINGS]),tuple(b[SETTINGS]))


if __name__=='__main__':
    unittest.main(verbosity=2)
