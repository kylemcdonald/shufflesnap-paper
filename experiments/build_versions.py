"""Build the paper's editions and check them.

    python experiments/build_versions.py [full|long|medium|short|all] [--draft]

``full`` is the complete paper in paper/.  The shorter editions are standalone files,
versions/{long,medium,short}/main.tex.  Their figures, tables, bibliography and listings
come from paper/; an edition may override any of them with a file at the same relative
path next to its main.tex.  Each edition is staged into a clean directory with only the
files it uses, compiled with tectonic, and rejected if the log shows an overfull box or
an undefined reference or citation.  The body ends at \\label{page:body-end}; references
start on a new page and do not count.  The body must have exactly the target length
(10, 6 or 3 pages) unless --draft is given.  Outputs: dist/shufflesnap-<edition>.pdf,
a source archive next to it (ready for arXiv, with a 00README.json that selects XeLaTeX), and
versions/<edition>/build-manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DIST = ROOT / "dist"
TARGETS = {"long": 10, "medium": 6, "short": 3}
# arXiv processing instructions, written into every source archive: the papers load fontspec,
# so they need XeLaTeX (supported by arXiv since November 2025; fonts are loaded by file name,
# as arXiv requires).  https://info.arxiv.org/help/00README.html
ARXIV_README = {"process": {"compiler": "xelatex"}, "texlive_version": 2025,
                "sources": [{"filename": "main.tex", "usage": "toplevel"}]}
PACKAGE_REFERENCE = ROOT.parent / "shufflesnap" / "src" / "shufflesnap" / "reference.py"


def _assets(tex: str):
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    figures += re.findall(r"\\panel(?:\[[^\]]*\])?\{([^}#]+)\}", tex)
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    listings = re.findall(r"\\lstinputlisting(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    keep = lambda xs: sorted({x for x in xs if "#" not in x})  # skip macro parameters
    return keep(figures), keep(inputs), keep(listings)


def _expand(edition_dir: Path, tex: str, seen=()) -> str:
    """The source with every \\input file's text appended (recursively), so that the
    figures and listings used inside an input file are staged too."""
    out = tex
    for rel in re.findall(r"\\input\{([^}]+)\}", tex):
        rel = rel if rel.endswith(".tex") else rel + ".tex"
        if "#" in rel or rel in seen:
            continue
        out += "\n" + _expand(edition_dir, _resolve(edition_dir, rel).read_text(), seen + (rel,))
    return out


def _resolve(edition_dir: Path, rel: str, exts=("",)) -> Path:
    for base in (edition_dir, PAPER):
        for ext in exts:
            p = base / (rel + ext)
            if p.is_file():
                return p
    raise FileNotFoundError(f"{rel} not found next to the edition or in paper/")


def _check_log(log: str, name: str):
    problems = []
    for pat in (r"Overfull \\hbox", r"Overfull \\vbox", r"Reference `[^']+' on page \d+ undefined",
                r"Citation `[^']+' on page \d+ undefined", r"There were undefined references",
                r"Label\(s\) may have changed"):
        for m in re.finditer(pat, log):
            line = log[m.start():log.find("\n", m.start())]
            problems.append(line.strip())
    if problems:
        raise RuntimeError(f"{name}: " + "; ".join(sorted(set(problems))[:12]))


def _compile(stage: Path):
    cmd = ["tectonic", "-X", "compile", "main.tex", "--keep-logs", "--keep-intermediates"]
    p = subprocess.run(cmd, cwd=stage, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("tectonic failed:\n" + p.stdout[-3000:] + p.stderr[-3000:])
    return (stage / "main.log").read_text(errors="replace")


def _body_pages(aux: str):
    m = re.search(r"\\newlabel\{page:body-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    return int(m.group(1)) if m else None


def _check_reference_copy():
    """paper/reference.py is printed in the paper; it must equal the package's file."""
    local = PAPER / "reference.py"
    if PACKAGE_REFERENCE.is_file() and local.read_bytes() != PACKAGE_REFERENCE.read_bytes():
        raise RuntimeError("paper/reference.py differs from the package's shufflesnap/reference.py; copy it again")


def build_full():
    _check_reference_copy()
    with tempfile.TemporaryDirectory(prefix="shufflesnap-full-") as tmp:
        stage = Path(tmp) / "source"
        shutil.copytree(PAPER, stage, ignore=shutil.ignore_patterns("*.aux", "*.log", "*.out", "*.bbl", "*.blg",
                                                                    "*.toc", "main.pdf", "__pycache__", "build"))
        log = _compile(stage)
        _check_log(log, "full")
        out = PAPER
        shutil.copy2(stage / "main.pdf", out / "main.pdf")
        pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(out / "main.pdf")],
                                                                  capture_output=True, text=True).stdout).group(1))
        return _publish("full", stage, pages, None)


def build(name: str, strict: bool = True):
    edition = ROOT / "versions" / name
    src = edition / "main.tex"
    tex = src.read_text()
    figures, inputs, listings = _assets(_expand(edition, tex))
    if listings:
        _check_reference_copy()
    with tempfile.TemporaryDirectory(prefix=f"shufflesnap-{name}-") as tmp:
        stage = Path(tmp) / "source"
        (stage / "figures").mkdir(parents=True)
        shutil.copy2(src, stage / "main.tex")
        shutil.copy2(_resolve(edition, "refs.bib"), stage / "refs.bib")
        for f in figures:
            rel = f if f.startswith("figures/") else "figures/" + f
            path = _resolve(edition, rel, exts=("", ".pdf", ".png"))
            shutil.copy2(path, stage / "figures" / path.name)
        for f in inputs:
            rel = f if f.endswith(".tex") else f + ".tex"
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_resolve(edition, rel), dest)
        for f in listings:
            dest = stage / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_resolve(edition, f), dest)
        log = _compile(stage)
        _check_log(log, name)
        count = _body_pages((stage / "main.aux").read_text())
        if count is None:
            raise RuntimeError(f"{name}: missing \\label{{page:body-end}}")
        if strict and count != TARGETS[name]:
            raise RuntimeError(f"{name}: expected {TARGETS[name]} body pages, got {count}")
        build_dir = edition / "build"
        build_dir.mkdir(exist_ok=True)
        for f in ("main.pdf", "main.log", "main.aux", "main.bbl"):
            if (stage / f).exists():
                shutil.copy2(stage / f, build_dir / f)
        return _publish(name, stage, count, TARGETS[name])


def _publish(name, stage: Path, count, target):
    DIST.mkdir(exist_ok=True)
    dest = DIST / f"shufflesnap-{name}.pdf"
    shutil.copy2(stage / "main.pdf", dest)
    archive = DIST / f"shufflesnap-{name}-source.zip"
    keep = {".tex", ".bib", ".bbl", ".pdf", ".png", ".py"}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p.suffix in keep and p.name != "main.pdf":
                z.write(p, p.relative_to(stage))
        z.writestr("00README.json", json.dumps(ARXIV_README, indent=2) + "\n")
    total = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(dest)], capture_output=True,
                                                             text=True).stdout).group(1))
    result = dict(edition=name, body_pages=count, target_body_pages=target, total_pages=total, pdf=dest.name,
                  pdf_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), source_archive=archive.name)
    manifest = (ROOT / "versions" / name / "build-manifest.json") if name != "full" else (
        ROOT / "versions" / "full-build-manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edition", nargs="?", default="all", choices=["full", *TARGETS, "all"])
    ap.add_argument("--draft", action="store_true", help="do not enforce the body length")
    a = ap.parse_args()
    names = ["full", *TARGETS] if a.edition == "all" else [a.edition]
    failed = False
    for n in names:
        try:
            build_full() if n == "full" else build(n, strict=not a.draft)
        except Exception as e:  # report every edition, then fail
            print(f"FAILED {n}: {e}", file=sys.stderr, flush=True)
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
