from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullLocator
ROOT=Path(__file__).resolve().parent

def main():
    df=pd.read_json(ROOT/'full.jsonl',lines=True)
    df.groupby(['family','instance','method']).agg(
        gap_min=('gap','min'),gap_median=('gap','median'),gap_max=('gap','max'),
        time_min=('elapsed','min'),time_median=('elapsed','median'),time_max=('elapsed','max')
    ).to_csv(ROOT/'seed_spread.csv')
    columns=['elapsed','gap','grouping','solving','polishing','passes','rounds','time_to_0.1','time_to_0.01']
    paired=df.groupby(['family','instance','method'])[columns].median()
    summary=paired.groupby('method').median()
    summary.to_csv(ROOT/'summary.csv')
    print('Target hit counts (of 75 runs/method):\n',df.groupby('method')[["time_to_0.1","time_to_0.01"]].count())
    print(summary.to_string(float_format=lambda x:f'{x:.5f}'))
    family=paired.groupby(['family','method'])[['gap','elapsed','time_to_0.1','time_to_0.01']].median()
    family.to_csv(ROOT/'families.csv');print('\n',family.to_string(float_format=lambda x:f'{x:.5f}'))
    base=paired.xs('baseline',level='method')
    wins={}
    for method in df.method.unique():
        if method=='baseline':continue
        sub=paired.xs(method,level='method')
        wins[method]={'lower_gap_instances':int((sub.gap<base.gap).sum()),'faster_instances':int((sub.elapsed<base.elapsed).sum()),'median_time_ratio':float((sub.elapsed/base.elapsed).median()),'median_gap_ratio':float((sub.gap/base.gap).median())}
    (ROOT/'comparisons.json').write_text(json.dumps(wins,indent=2));print('\n',json.dumps(wins,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    order=['baseline','linear1','linear4','geometric1','geometric4','shifted4']
    colors=plt.get_cmap('tab10').colors
    for i,m in enumerate(order):
        sub=paired.xs(m,level='method')
        axes[0].scatter(sub.elapsed,sub.gap,s=20,alpha=.6,color=colors[i],label=m)
    axes[0].set(xscale='log',yscale='log',xlabel='Total runtime (s)',ylabel='Final auction-reference gap (%)')
    axes[0].set_xticks([.2,.5,1,2]);axes[0].xaxis.set_major_formatter(ScalarFormatter());axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend(fontsize=7)
    times=df.groupby('method')[['grouping','solving','polishing']].median().reindex(order)
    bottom=np.zeros(len(order))
    for col,c in zip(times.columns,['#e69f00','#56b4e9','#009e73']):
        axes[1].bar(order,times[col],bottom=bottom,label='native cleanup' if col=='polishing' else col,color=c);bottom+=times[col]
    axes[1].tick_params(axis='x',rotation=45);axes[1].set_ylabel('Median component time (s)');axes[1].legend(fontsize=7)
    for i,m in enumerate(['baseline','linear1','geometric4','shifted4']):
        runs=df[(df.family=='meander')&(df.instance==0)&(df.method==m)]
        for j,row in enumerate(runs.itertuples()):
            t=np.array([a['time'] for a in row.trace]);gap=np.array([100*(a['cost']/row.reference-1) for a in row.trace])
            axes[2].plot(t,gap,alpha=.55,color=colors[order.index(m)],label=m if j==0 else None)
    axes[2].set(xscale='log',yscale='log',xlabel='Elapsed time (s)',ylabel='Meander seed 0 gap (%)');axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180)
if __name__=='__main__':main()
