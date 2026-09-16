# Publication review and handoff

Prepared 2026-09-15 for author and human review. The paper has not been submitted to arXiv.

## Changes made

- Renamed the library, import package, paper and print repositories, active file paths, URLs, workflows, and generated figure labels to ShuffleSnap.
- Rewrote the abstract and central claims to distinguish linear work per round and logarithmically many initial scales from the unbounded polishing tail and empirical accuracy.
- Removed certified-optimum language: stored auction results are high-accuracy floating-point references without independent dual certificates.
- Reported the uniform-family exception, RMS interpretation, and 26/30 random-initialization wins explicitly.
- Clarified fixed occupied subsets for spare cells and `O(n+m)` storage for sparse grids.
- Corrected related-work scope for Euclidean versus squared-Euclidean/RMS matching; added the RMS approximation paper and Sca²Gri.
- Preserved BSP-OT's measured short-budget advantage; clarified checkpoint granularity, timing exclusions, parallelism, and limits of the custom billion-point port comparison.
- Corrected the convergence plot's disappearing-seed median artifact and the ablation plot's NaN-skipping success artifact.
- Reran convergence using the same deterministic permutation as the package and other benchmark families. Its environment is recorded separately.
- Fixed float placement so figures do not accumulate after the references.
- Added a clean-source build and arXiv archive script, pinned scientific requirements, an immutable solver source link, and an evidence-checking script.

## Fable collaboration

The separate Fable bridge reviewed the manuscript, implementation, and data. Its first review identified the plotting bugs, seed mismatch, incomplete source provenance, initialization overstatement, stale numerical comparisons, and undocumented comparator details. Those findings were incorporated. Optional new datasets, stronger certificates, and additional initializer experiments were treated as future work rather than prerequisites for human review.

Full review responses and local validation logs are retained in the workspace's sibling `publication-review/` directory. Fable's second review concluded: **ready for author/human review**, with no remaining factual blocker. Its final suggestions were incorporated by clarifying the partial-success ablation caption, quantifying the very small cycling improvement, reducing repeated caveats, and keeping exploratory render variants local. The immutable GitHub source was accepted as adequate before PyPI publication. A subsequently discovered missing SciPy test dependency was fixed in the package metadata and CI; the paper links to that updated source commit.

## Validation

The library source is pinned in the manuscript. The publication checks cover package tests, wheel/source-distribution metadata and installation, reproducing all 18 distribution final gaps, 24 cached bijections and reference costs, and all 90 time-matched checkpoints. `scripts/build_submission.py` separately compiles the exact contents of the source archive and rejects undefined references and overfull boxes. The rendered PDF is inspected for figure placement and legibility.

## PyPI release

[ShuffleSnap 0.3.0](https://pypi.org/project/shufflesnap/0.3.0/) was published on 2026-09-16. All platform builds passed; a clean PyPI installation passed all 25 tests. The paper retains its pinned source revision, whose solver code is unchanged in the release.

## Remaining author/reviewer items

1. Review the PDF and get human feedback on novelty, interpretation, and presentation. The key technical limits are explicit: no global approximation guarantee, small synthetic samples, one real embedding, and incomplete historical provenance for the 100-million-point BSP-OT summaries.
2. Before arXiv submission, choose the category, license, and author metadata, and decide on an acknowledgement of AI assistance in development and manuscript preparation. No new coauthor or institutional affiliation has been assumed.
3. Submit the source archive only after incorporating human feedback. It contains the manuscript, generated statistics/table, bibliography, and required figures; no private print-order records or development logs.

The historical large benchmark runs were not repeated. Their original timings are retained and their limitations documented; the seed-corrected convergence run is the exception.
