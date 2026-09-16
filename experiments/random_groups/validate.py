"""Cross-check the extracted baseline and parallel grouping implementation."""
import hashlib,json
from pathlib import Path
import numpy as np
import shufflesnap
from run import run,make_points,seed_default
ROOT=Path(__file__).resolve().parent
checks=[]
for side in [7,40,200]:
 for method in ['baseline','shifted4']:
  a=run('meander',side,0,method,13,1)
  b=run('meander',side,0,method,13,24)
  assert a['assignment_sha256']==b['assignment_sha256'],(side,method)
  checks.append(f'{side}x{side} {method}: identical assignments with 1 and 24 threads')
  if method=='baseline':
   points=make_points('meander',side*side,0)
   result=shufflesnap.window_cleanup(points,seed_default(points,side*side),rows=side,cols=side)
   digest=hashlib.sha256(np.asarray(result['assignment'],dtype=np.int64).tobytes()).hexdigest()
   assert digest==a['assignment_sha256']
   checks.append(f'{side}x{side}: extracted baseline matches installed production assignment')
(ROOT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print('\n'.join(checks))
