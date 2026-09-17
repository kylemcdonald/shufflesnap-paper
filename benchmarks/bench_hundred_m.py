"""Refresh the 100-million-point comparison with raw logs and unit-square inputs.

Run after bench_billion.py --side 10000 --seed 0 --budget 500 --no-save-assignment.
Requires the rebuilt official and memory-lean BSP-OT drivers. The billion-point
measurements are retained; only their separate ref_1e8 comparison is replaced.
"""
from pathlib import Path
import gc
import json
import os
import subprocess
import tempfile
import time

from bench_billion import generate_meander
from bench_common import REPO, results_dir
from refresh_bspot_references import load_summary


def main():
    results = results_dir()
    baseline = load_summary(results / 'billion_10000_seed0.json', side=10000, budget_s=500)
    assert baseline['coordinate_system'] == 'unit-square'
    environment = json.loads((results / 'environment_unit_square.json').read_text())
    logdir = results / 'hundred_m_logs'
    logdir.mkdir(exist_ok=True)
    ref = {'side': 10000, 'coordinate_system': 'unit-square',
           'shufflesnap_final_cost_500s_run': baseline['final_cost'],
           'BSP-OT_revision': environment['BSP-OT_revision'],
           'compiler': environment['compiler'],
           'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'raw_logs': 'hundred_m_logs/'}
    with tempfile.TemporaryDirectory(prefix='shufflesnap-100m-') as tmp:
        path = Path(tmp) / 'points.bin'
        points = generate_meander(10000**2, seed=0)
        points.tofile(path)
        del points
        gc.collect()
        for tag, driver, plans, threads in [
            ('stock_16plan', 'bspot_grid', 16, 4),
            ('stock_1plan', 'bspot_grid', 1, 1),
            ('giga_1plan', 'bspot_grid_giga', None, 1),
        ]:
            cmd = [str(REPO/'solvers/cpp'/driver), '--points', str(path),
                   '--rows', '10000', '--cols', '10000', '--seed', '0']
            if plans is not None:
                cmd += ['--plans', str(plans)]
            metadata = {'command': [x if x != str(path) else '<unit-square meander input>' for x in cmd],
                        'OMP_NUM_THREADS': threads, 'coordinate_system': 'unit-square',
                        'instance_seed': 0}
            (logdir/f'{tag}.json').write_text(json.dumps(metadata, indent=2)+'\n')
            with (logdir/f'{tag}.stdout.txt').open('w') as out, (logdir/f'{tag}.stderr.txt').open('w') as err:
                subprocess.run(['/usr/bin/time', '-f', '%M', '-o', str(logdir/f'{tag}.rss_kib.txt'), *cmd],
                               env=dict(os.environ, OMP_NUM_THREADS=str(threads)),
                               stdout=out, stderr=err, check=True)
            values = dict(line.split('=',1) for line in (logdir/f'{tag}.stdout.txt').read_text().splitlines() if '=' in line)
            assert values['bijective'] == '1'
            ref[tag+'_cost'] = float(values['assignment_cost'])
            ref[tag+'_time_s'] = float(values['solve_time_s'])
            rss_key = tag+'_peak_rss_gib' + ('_omp4' if plans == 16 else '')
            ref[rss_key] = int((logdir/f'{tag}.rss_kib.txt').read_text()) / 2**20
            print(tag, ref[tag+'_cost'], ref[tag+'_time_s'], ref[rss_key], flush=True)
    (logdir/'ref_1e8.json').write_text(json.dumps(ref, indent=2)+'\n')
    summary_path = results/'billion_bspot_31623_seed0.json'
    summary = json.loads(summary_path.read_text())
    summary.setdefault('coordinate_system', 'legacy-inset-0.03')
    summary['ref_1e8'] = ref
    pending = summary_path.with_suffix('.pending.json')
    pending.write_text(json.dumps(summary, indent=1)+'\n')
    subprocess.run([os.sys.executable, str(REPO/'benchmarks/refresh_bspot_references.py'),
                    '--bspot', str(pending)], check=True)
    pending.replace(summary_path)


if __name__ == '__main__':
    main()
