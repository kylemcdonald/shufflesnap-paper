from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    large=pd.read_json(ROOT/'scale.jsonl',lines=True)
    print('LARGE RUNS (no optimal reference at these sizes)')
    print(large[['side','instance','method','elapsed','cost','grouping','solving','polishing','rounds','converged']].sort_values(['side','instance','method']).to_string(index=False))
    pairs=[]
    for (side,instance),group in large.groupby(['side','instance']):
        base=group[group.method=='baseline'].iloc[0]
        for row in group.itertuples():
            def first_time(trace,target):return next((t['time'] for t in trace if t['cost']<=target),None)
            target=base.cost*1.001
            pairs.append(dict(side=side,instance=instance,method=row.method,time=row.elapsed,time_ratio=row.elapsed/base.elapsed,cost_difference_pct=100*(row.cost/base.cost-1),grouping_share=row.grouping/row.elapsed,baseline_target_time=first_time(base.trace,target),target_time=first_time(row.trace,target),converged=row.converged))
    pairs=pd.DataFrame(pairs);pairs.to_csv(ROOT/'scaling_comparison.csv',index=False)
    print('\nLarge-grid medians (3 instances for baseline/shifted4; only seed 0 for linear1)')
    print(pairs.groupby(['side','method']).median(numeric_only=True).to_string())
    init=pd.read_json(ROOT/'init.jsonl',lines=True)
    a=init[init.method=='shifted4'].set_index(['family','instance','initseed']).sort_index()
    b=init[init.method=='baseline'].set_index(['family','instance','initseed']).sort_index()
    print('\nIndependent initial assignments:',len(a),'pairs')
    print('lower cost:',int((a.cost<b.cost).sum()),'faster:',int((a.elapsed<b.elapsed).sum()),'median time ratio:',float((a.elapsed/b.elapsed).median()),'median gap ratio:',float((a.gap/b.gap).median()))
    controls=pd.read_json(ROOT/'controls.jsonl',lines=True)
    print('\nControls (local only capped at 30s)')
    print(controls[['family','method','elapsed','gap','rounds','converged']].sort_values(['family','method']).to_string(index=False))
    summary={'init_pairs':len(a),'init_lower_cost':int((a.cost<b.cost).sum()),'init_faster':int((a.elapsed<b.elapsed).sum()),'init_time_ratio':float((a.elapsed/b.elapsed).median()),'init_gap_ratio':float((a.gap/b.gap).median())}
    (ROOT/'control_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
