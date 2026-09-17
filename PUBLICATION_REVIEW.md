# Publication review and handoff

Updated 2026-09-17 for author and human review. The paper has not been submitted to arXiv.

## Coordinate-margin removal

The solver now places grid centers across `[0,1]` on each axis; a singleton
axis uses `0.5`. The `margin` parameter is removed throughout the Python API,
native kernel, examples, and benchmark drivers. Input coordinates remain the
caller's responsibility: the API does not normalize them. The paper's benchmark
harness normalizes every method's inputs identically.

Three further simplifications preserve behavior: the public API returns its
freshly owned assignment without another copy; the showcase renderer uses the
returned positions instead of rebuilding the grid; and the paper renderer no
longer supports an obsolete reference-assignment cache format. Reference costs
and gallery assignments now come from the same solve. Cache entries fingerprint
both input coordinates and target geometry, preventing reuse across conventions.

Fable reviewed these changes and the ownership of returned buffers. All 52
library tests pass, including endpoint, singleton-axis, cost-consistency, and
buffer-lifetime checks. The stopping tolerance, clipped trailing windows, masks,
and fixed occupied-cell controls remain: each affects meaningful behavior.
Physical print margins and figure spacing are unrelated and unchanged.

All 39 auction references were recomputed from unit-square inputs. Their costs
agree with the old costs divided by `0.94²` to within `1.3e-14` relative error.
All 15 headline distribution runs converged; their reference gaps changed by at
most `2.4e-12` percentage points. Family median gaps remain 0.006–0.030%, with
median runtimes of 0.21–0.24 seconds in this rerun. Timing differences between
runs should not be attributed to margin removal alone. In the refreshed
time-matched comparison, BSP-OT wins four of five families, with ShuffleSnap
ahead on the ring. The manuscript now states that exception.

The convergence, schedule/window ablations, initialization, scaling through
4.2 million points, MNIST, and BSP-OT experiments were regenerated, along with
their figures. All 27 scaling runs, 33 schedule/window runs, and 264
initialization runs converged. The machine environment, dependencies, source
hashes, and upstream comparator revision are recorded with the results.

Fable checked the refreshed tables and manuscript claims. Its review led to a
clearer explanation of the ring's sensitivity to completed-round timing, a more
careful abstract, and simpler coordinate/provenance text. Detailed collaboration
and comparison reports are retained in `publication-review/remove-margin/` in
the parent workspace.

The 100-million-point comparison was also rerun with raw driver logs: ShuffleSnap
completed 14 rounds in 536 seconds under a requested 500-second budget; official
16-plan BSP-OT took 248 seconds and used 29.3 GiB. Only the billion-point solver
measurements remain historical. Its saved assignment is rendered in unit-square
coordinates, with the distinction stated in Appendix D. Fable verified that the
comparison ratios never mix coordinate conventions.

The updated paper retains a three-page main section; appendices and references
bring the complete draft to 19 pages. The manuscript and a clean copy of the
source archive both compile without undefined references or overfull boxes.
Evidence checks reproduced all 15 final gaps, verified 20 bijective figure
assignments, and checked all 75 matched-time checkpoints.

The implementation is pinned to `afd6626` (full hash in the manuscript), version
`0.4.0.dev0`. This is an unreleased source revision; PyPI remains at `0.3.0`.

## Earlier rewrite from the approved outline

The approved-outline draft had three main pages, with five appendices and
references bringing that PDF to 18 pages. Fable led the prose rewrite; Codex integrated the text,
checked claims against code and stored results, built the new method diagram,
and checked the typeset pages. The main explanation follows four steps: a small
reassignment, longer reach through strides, coverage through shifted phases,
and coarse-to-fine refinement. One table preserves the quality/runtime tradeoff.

The appendices now have one home each for algorithm details, evaluation and
MNIST, convergence and ablations, scaling, and related work. Uniform benchmarks
and the separate random-cell-group experiment remain outside the manuscript.
The outline is marked approved and implemented.

The new vector diagram shows an actual cost-lowering reassignment before/after,
then adjacent versus strided cells and overlapping phases. The descent-strip
renderer now finishes its recorded schedule with local rounds, matching the
production default assignment exactly; it previously restarted a full schedule
for the final frame. No solver or benchmark results changed.

Validation reproduced 15 distribution outcomes, checked 20 cached bijections and
reference costs, and checked 75 time-matched checkpoints. Both the manuscript and
the self-contained source archive compile without undefined references or
overfull boxes. Collaboration drafts and review responses are retained under
`publication-review/outline-rewrite/` in the parent workspace.

Fable’s final review found no major factual error. Its requests to define
“embedding,” clarify the family introduction and schematic window sizes, and
state the schedule-ablation range more precisely were incorporated. Its layout
concern (an almost empty fourth page) was resolved; the main paper is three pages.

## Earlier compact-paper revision

The main paper is now three pages: motivation, the window/stride technique, two
explanatory figures, and one headline comparison table. Detailed algorithms,
convergence, scaling, ablations, masks, MNIST, reference validation, and related
work are in the appendices. The main comparison retains BSP-OT's short-budget
advantage instead of implying equal-runtime superiority.

The trailing partial windows remain necessary for arbitrary grid sizes. On a
seven-cell axis, the clipped shifted window is the only bridge across the last
unshifted boundary. An independent exact-solve experiment fails to repair an
edge swap when that window is removed and repairs it to zero cost when retained.
Twenty new regression cases cover horizontal/vertical axes, five edge-sensitive
sizes, and local/default schedules; all 45 package tests pass. The kernel's
behavior is unchanged, so no wholesale benchmark rerun or billion-point rerun
was needed. The 15 retained distribution outcomes were reproduced again.

Fable independently confirmed this boundary analysis and reviewed the compact
manuscript. Its diagram, stride-definition, stale timing-range, and typo fixes
were incorporated; duplicate figures were removed. The headline table uses
three-instance medians from the quality runs, while five repeats apply only to
the separate time-matched experiment. Reviews are retained under `publication-review/revision-compact/` in
the parent workspace.

## Changes made

- Renamed the library, import package, paper and print repositories, active file paths, URLs, workflows, and generated figure labels to ShuffleSnap.
- Rewrote the abstract and central claims to distinguish linear work per round and logarithmically many initial scales from the unbounded polishing tail and empirical accuracy.
- Removed certified-optimum language: stored auction results are high-accuracy floating-point references without independent dual certificates.
- Restricted the manuscript to five distribution families; removed the uniform benchmark from all figures, tables, and statistics. Historical raw data remain archived.
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

The library source is pinned in the manuscript. The publication checks cover package tests, wheel/source-distribution metadata and installation, reproducing all 15 retained distribution final gaps, 20 cached bijections and reference costs, and all 75 retained time-matched checkpoints. `scripts/build_submission.py` separately compiles the exact contents of the source archive and rejects undefined references and overfull boxes. The rendered PDF is inspected for figure placement and legibility.

## PyPI release

[ShuffleSnap 0.3.0](https://pypi.org/project/shufflesnap/0.3.0/) was published on 2026-09-16. All platform builds passed; a clean PyPI installation passed all 25 tests. The paper retains its pinned source revision, whose solver code is unchanged in the release.

## Remaining author/reviewer items

1. Review the PDF and get human feedback on novelty, interpretation, and presentation. The key technical limits are explicit: no global approximation guarantee, small synthetic samples, one real embedding, and incomplete historical provenance for the 100-million-point BSP-OT summaries.
2. Before arXiv submission, choose the category, license, and author metadata, and decide on an acknowledgement of AI assistance in development and manuscript preparation. No new coauthor or institutional affiliation has been assumed.
3. Submit the source archive only after incorporating human feedback. It contains the manuscript, generated statistics/table, bibliography, and required figures; no private print-order records or development logs.

The historical large benchmark runs were not repeated. Their original timings are retained and their limitations documented; the seed-corrected convergence run is the exception.
