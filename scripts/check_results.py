"""Check stored benchmark evidence and reproduce the 18 distribution outcomes.

Does not overwrite historical timings or reference data. Requires the scientific
requirements and the sibling shufflesnap source installation.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import shufflesnap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from bench_common import (DISTRIBUTIONS, RESULTS, assignment_cost,
                          build_target_grid, gap_pct, make_points, seed_default)


def main():
    cache = json.loads((RESULTS / 'exact_cache.json').read_text())
    distributions = pd.read_csv(RESULTS / 'distributions.csv')
    for row in distributions.itertuples():
        side = int(round(row.n ** 0.5))
        ref = cache[f'{row.distribution}:{side}x{side}:seed{row.instance_seed}']['cost']
        np.testing.assert_allclose(ref, row.optimum, rtol=1e-12)
        points = make_points(row.distribution, row.n, row.instance_seed)
        result = shufflesnap.window_cleanup(points, seed_default(points, row.n), rows=side, cols=side)
        assert result['converged'], (row.distribution, row.instance_seed)
        np.testing.assert_allclose(gap_pct(result['final_cost'], ref), row.final_gap_pct,
                                   rtol=1e-7, atol=1e-10)
    targets = build_target_grid(200, 200)
    for family in DISTRIBUTIONS:
        points = make_points(family, 40000, 0)
        ref = cache[f'{family}:200x200:seed0']['cost']
        for method in ['shufflesnap', 'exact', 'bspot', 'rasterfairy']:
            data = np.load(RESULTS / 'assignments' / f'{method}_{family}_200_seed0.npz')
            assignment = data['assignment']
            np.testing.assert_array_equal(np.sort(assignment), np.arange(40000))
            cost = assignment_cost(points, targets, assignment)
            assert cost >= ref - 1e-9, (family, method, cost, ref)
            if method == 'exact':
                np.testing.assert_allclose(cost, ref, rtol=1e-11)
    matched = pd.read_csv(RESULTS / 'bspot_time_matched.csv')
    assert (matched['shufflesnap_elapsed_s'] <= matched['bspot_time_s']).all()
    assert len(matched) == 6 * 3 * 5
    print('PASS: 18 reproduced final gaps, 24 bijective figure assignments, '
          'auction cache agreement, and 90 in-budget checkpoints.')


if __name__ == '__main__':
    main()
