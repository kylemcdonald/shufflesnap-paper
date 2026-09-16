"""Isolated experiment: never writes manuscript data, figures, or production code."""
from pathlib import Path
import argparse,hashlib,json,sys,tempfile,subprocess,time,platform
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'benchmarks'))
from bench_common import make_points,build_target_grid,assignment_cost,seed_default,DISTRIBUTIONS
CONFIGS={'baseline':('baseline',1,0),'linear1':('linear',1,0),'linear4':('linear',4,0),'geometric1':('geometric',1,0),'geometric4':('geometric',4,0),'shifted4':('geometric',4,1)}
def run(family,side,instance,method,groupseed=0,threads=24,initseed=0):
    points=make_points(family,side*side,instance);initial=np.random.default_rng(initseed).permutation(side*side).astype(np.int64)
    mode,passes,shift=CONFIGS[method]
    with tempfile.TemporaryDirectory() as tmp:
        file=Path(tmp)/'input.bin'
        with file.open('wb') as f:points.astype('float64').tofile(f);initial.astype('int64').tofile(f)
        output=subprocess.check_output([str(ROOT/'build/driver'),str(file),str(side),mode,str(threads),str(groupseed),str(passes),str(shift)],text=True)
        assignment=np.fromfile(str(file)+'.assignment',dtype=np.int64)
    np.testing.assert_array_equal(np.sort(assignment),np.arange(side*side))
    trace=[]
    for line in output.splitlines():
        x=line.split(',')
        if x[0]=='trace':trace.append({'time':float(x[1]),'cost':float(x[2]),'phase':x[3],'scale':int(x[4]),'attempted':int(x[5]) if len(x)>5 else None,'accepted':int(x[6]) if len(x)>6 else None})
        else:summary=list(map(float,x[1:]))
    elapsed,cost,grouping,solving,polishing,passes,rounds,converged=summary
    checked=assignment_cost(points,build_target_grid(side,side),assignment)
    np.testing.assert_allclose(cost,checked,rtol=1e-11)
    assert np.all(np.diff([r['cost'] for r in trace])<=1e-8)
    cache=json.loads((ROOT.parents[1]/'benchmarks/results/exact_cache.json').read_text())
    ref=cache.get(f'{family}:{side}x{side}:seed{instance}',{}).get('cost')
    result=dict(assignment_sha256=hashlib.sha256(assignment.tobytes()).hexdigest(),family=family,side=side,instance=instance,method=method,groupseed=groupseed,threads=threads,initseed=initseed,elapsed=elapsed,cost=cost,grouping=grouping,solving=solving,polishing=polishing,passes=int(passes),rounds=int(rounds),converged=bool(converged),reference=ref,trace=trace)
    if ref:
        result['gap']=100*(cost/ref-1)
        for target in [1,.1,.01]:result[f'time_to_{target}']=next((r['time'] for r in trace if 100*(r['cost']/ref-1)<=target),None)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',default='pilot');a=p.parse_args()
    methods=list(CONFIGS)
    if a.stage=='pilot':jobs=[(f,200,0,m,0,24) for f in DISTRIBUTIONS for m in methods]
    elif a.stage=='full':jobs=[(f,200,i,m,g,24) for f in DISTRIBUTIONS for i in [0,1,2] for g in range(5) for m in methods]
    elif a.stage=='ablations':
        CONFIGS.update({'noshuffle':('noshuffle',4,1),'cycles':('cycles',1,0)})
        jobs=[(f,200,0,'noshuffle',0,24) for f in ['meander','ring','two_blobs']] + [(f,200,i,m,0,24) for f in DISTRIBUTIONS for i in [0,1,2] for m in ['baseline','cycles']]
    elif a.stage=='focused':
        CONFIGS.update({'shifted2':('geometric',2,1)})
        jobs=[(f,200,i,m,g,24) for f in DISTRIBUTIONS for i in [0,1,2] for g in [0,1,2] for m in ['baseline','shifted2','shifted4']]
    elif a.stage=='controls':
        CONFIGS.update({'local':('local',1,0),'alloffsets':('alloffsets',1,0)})
        jobs=[(f,200,0,m,0,24) for f in ['meander','ring','two_blobs'] for m in ['local','alloffsets','baseline']]
    elif a.stage=='init':
        jobs=[(f,200,i,m,7,24,j) for f in DISTRIBUTIONS for i in [0,1,2] for j in [1,2,3] for m in ['baseline','shifted4']]
    elif a.stage=='scale':jobs=[('meander',s,i,m,i,24) for s in [400,800,1131] for i in [0,1,2] for m in (['baseline','shifted4','linear1'] if i==0 else ['baseline','shifted4'])]
    else:raise ValueError(a.stage)
    np.random.default_rng(413).shuffle(jobs)
    output=ROOT/f'{a.stage}.jsonl'
    with output.open('w') as f:
        for args in jobs:
            r=run(*args);f.write(json.dumps(r)+'\n');f.flush()
            print(r['family'],r['side'],r['instance'],r['method'],r['groupseed'],'%.3fs'%r['elapsed'],'gap',r.get('gap'),'polish',r['rounds'],flush=True)
