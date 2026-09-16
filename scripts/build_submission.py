"""Build the reviewer PDF and a minimal, self-contained arXiv source archive.

Requires tectonic on PATH. Run from any directory; outputs go to dist/.
The archive is compiled independently before it is packaged.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
DIST = ROOT / 'dist'


def compile_tex(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tectonic', '--keep-intermediates', '--keep-logs',
                    '--outdir', str(out), 'main.tex'], cwd=folder, check=True)
    log = (out / 'main.log').read_text(errors='replace')
    failures = [line for line in log.splitlines() if any(
        term in line for term in ['Undefined control sequence', 'Overfull \\hbox',
                                 'Overfull \\vbox', 'There were undefined references',
                                 'Citation `'])]
    if failures:
        raise RuntimeError('Resolve typesetting errors before packaging:\n' + '\n'.join(failures))


def main():
    DIST.mkdir(exist_ok=True)
    build = PAPER / 'build'
    compile_tex(PAPER, build)
    source = (PAPER / 'main.tex').read_text()
    figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source)
    inputs = re.findall(r'\\input\{([^}]+)\}', source)
    with tempfile.TemporaryDirectory(prefix='shufflesnap-submission-') as tmp:
        stage = Path(tmp) / 'source'
        stage.mkdir()
        for name in ['main.tex', 'references.bib', *inputs]:
            shutil.copy2(PAPER / name, stage / name)
        shutil.copy2(build / 'main.bbl', stage / 'main.bbl')
        (stage / 'figures').mkdir()
        for name in sorted(set(figures)):
            shutil.copy2(PAPER / 'figures' / name, stage / 'figures' / name)
        # A clean source-only compile catches missing files or accidental paths.
        compile_tex(stage, Path(tmp) / 'build')
        pdf = DIST / 'shufflesnap-review.pdf'
        shutil.copy2(Path(tmp) / 'build' / 'main.pdf', pdf)
        archive = DIST / 'shufflesnap-arxiv-source.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in sorted(stage.rglob('*')):
                if file.is_file():
                    z.write(file, file.relative_to(stage))
    files = [pdf, archive]
    manifest = {p.name: {'bytes': p.stat().st_size,
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    (DIST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in files:
        print(p)


if __name__ == '__main__':
    main()
