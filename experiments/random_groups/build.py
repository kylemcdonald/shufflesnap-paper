from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
source=ROOT.parents[2]/'shufflesnap/src/core.cpp'
text=source.read_text().split('NB_MODULE(')[0]
text='\n'.join(line for line in text.splitlines() if not ('nanobind' in line or 'nb::literals' in line))
(ROOT/'build').mkdir(exist_ok=True)
(ROOT/'build/core.hpp').write_text(text)
subprocess.run(['c++','-O3','-std=c++20','-pthread',str(ROOT/'driver.cpp'),'-o',str(ROOT/'build/driver')],check=True)
