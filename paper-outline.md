# Paper outline

**Status: approved by the author and implemented in the September 16 rewrite.**
Numerical examples were refreshed on September 17 after removing the coordinate margin.
This document sets the story, space budget, evidence, and appendix destinations.
Feedback can change a section's purpose, move a topic, or change a visual without
needing to edit manuscript sentences.

**Central message:** Small assignment problems are inexpensive to solve exactly.
ShuffleSnap uses them to exchange points between widely separated grid cells,
then reduces that reach to refine the layout.

## Structural choices to steer

These choices guide the current manuscript and future outline-level feedback.

| Choice | Proposed direction |
|---|---|
| Main-paper promise | Explain a simple way to obtain low-displacement, one-to-one grid assignments; accuracy is empirical. |
| Reading order | The task → one small reassignment → longer reach → full-grid coverage → the complete schedule → evidence and tradeoff. |
| Main-paper emphasis | Roughly half the space explains the method. |
| Task illustration | Merge into the local before/after panel; no separate task figure. |
| Visual budget | One coordinated method figure, the existing descent strip, and one compact results table. |
| Descent strip | Keep in the main paper as qualitative evidence; detailed convergence curves stay in the appendix. |
| Related work | Essential positioning in the introduction; extended discussion in an appendix, without a separate main-text literature section. |
| Appendix structure | Five question-led appendices, combining protocol with comparative results and algorithm details with implementation. |
| Main evidence | Five-family final quality/runtime comparison, its short-budget tradeoff, and a brief comparison showing why strides matter. |
| Scope boundary | Reorganize existing evidence. No uniform benchmarks, random-cell-group experiments, or new experimental requirements. |

**Length budget, including visuals:** title/abstract ≈0.4 page; problem ≈0.5;
method ≈1½ pages including the method figure; results/scope/availability ≈1 page
including the descent strip and table. Aim for about 3½ pages, with a hard
ceiling of four before the appendices. These are layout budgets, not quotas to
fill. Use three numbered main sections; the method's four steps can be short
run-in headings rather than four formal subsections.

---

# Main paper

## Abstract — the task, the mechanism, the payoff

**Job:** give the whole idea in 80–100 words.

- Assign every point to its own grid cell while limiting squared displacement.
- Solve small assignments using widely spaced cells first and adjacent cells last.
- Give one scoped accuracy result: median reference gaps of 0.006–0.030% across
  five tested families at 40,000 points, with additional computation compared
  with the tested BSP-OT construction. Keep its advantage at very short budgets
  visible; do not let the abstract imply equal-time superiority.
- Leave initialization findings, scale records, and implementation details out.

## 1. From a point cloud to a grid

**Reader should leave knowing:** what stays fixed, what can change, and why a
bijection is harder than rounding coordinates.

- Start with the practical task: an embedding supplies desired item positions;
  an atlas needs one item per cell. Rounding causes collisions and holes.
- Define the fully occupied grid setting and total squared displacement with
  one equation. The input embedding and grid centers stay fixed; their matching
  changes. Long displacements incur a larger penalty.
- Briefly locate the method among alternatives: dense assignment is expensive;
  spatial construction (RasterFairy, BSP-OT) is fast but can leave displacement
  to refine; local-assignment sorters (LAS/FLAS) optimize different feature-space
  layouts. Keep this to a compact positioning paragraph, not a literature survey.
- Identify the contribution as **small exact reassignments organized into a
  coarse-to-fine schedule of strided grid neighborhoods**. Do not imply that
  local assignment solves themselves are new or that the method coarsens the
  underlying point set.

**Visual entry point:** introduce the points, cells, and displacement through
Figure 1(a), which is reused to explain a local reassignment in Section 2. There
is one physical figure, not a separate task drawing followed by a similar solve
drawing. Do not use the schematic as evidence of comparative accuracy.

**Keep out:** solver taxonomy, dense-solver implementation details, mask behavior,
benchmark definitions, and records at large scale.

## 2. Small assignments at several scales

**Reader should leave able to explain the algorithm without opening an appendix.**
The order matters: explain the operation, then its reach, then how it covers the
grid, then how those pieces form a complete run.

### Step 1 — Improve a small group of cells

- Begin with any valid assignment; the default is a seeded random permutation.
- Introduce a window of grid cells (default 6×6, at most 36 cells). Collect the
  points **currently assigned to those
  cells**, not points lying nearby in the input plane.
- Rematch those same points to those same cells exactly. Accept only an
  improvement in squared displacement.
- Establish feasibility preservation and decreasing accepted costs immediately.

### Step 2 — Make the same solve reach farther

- Adjacent-cell windows can only make local exchanges.
- Define stride as the spacing between selected rows and columns. Increasing
  stride increases reach without increasing the number of cells in a solve.
- Emphasize that the grid and assignment remain at full resolution throughout.

### Step 3 — Cover the grid and connect windows

- Process every interleaved subgrid, not just the one highlighted in a diagram.
- Within a phase, windows are disjoint and can be solved in parallel.
- The unshifted, horizontal, vertical, and combined half-window shifts form a
  round. Their overlap permits exchanges across earlier window boundaries.
- Mention smaller edge windows only briefly; their asymmetric clipping rule
  belongs in Appendix A.

### Step 4 — Reduce the stride, then finish locally

- Give the full sequence: random assignment → one round at each decreasing
  stride → repeat adjacent-window rounds until no move is accepted.
- Explain broad placement followed by fine correction. Put the exact starting
  stride, rectangular schedule, and numerical stopping rule in Appendix A.
- State linear work per round and linear memory for fixed-size windows on a
  fully occupied grid. Immediately distinguish these from an unbounded number
  of finishing rounds and an empirical final error.

**Figure 1 — The task, operation, and reach.** One coordinated diagram:
(a) points and their cells, with an actual local reassignment before/after;
(b) adjacent versus strided selection of the same number of cells; (c) phases crossing a window boundary.
A smaller illustrative window is acceptable if clearly labeled; state that the
default is 6×6. The reader must see what changes during a solve, not just colored
neighborhood outlines. This requires a new before/after drawing and consolidation
of the existing window/stride schematics. Exact panel geometry is a later illustration decision.

**Keep out:** pseudocode, proofs, timing curves, ablations, API options, and detailed
convergence traces. The short descent image sequence belongs with the main
results below.

## 3. What the refinement buys

**Reader should leave knowing:** when the method is useful and what its extra
accuracy costs.

- Establish the comparison's scope: five synthetic families, 40,000 points,
  three instances per family, one desktop. Define “reference gap” in a phrase.
- Present the five-family table of BSP-OT and ShuffleSnap final gaps and times.
  ShuffleSnap has lower family-median final costs in these tests, at roughly
  0.21–0.24 s versus 0.04–0.08 s.
- State the other side immediately: BSP-OT has lower median gaps at its measured
  short runtimes on four of five families; the ring is the exception. The endpoint
  table is not an equal-budget comparison.
- Give the mechanism its own short paragraph: adjacent-window refinement stalls
  at a higher gap on the tested meander inputs. Keep the configurations and
  curves in Appendix C; do not imply a pure schedule ablation when starts differ.
  Pair this observation with the descent strip, while distinguishing numerical
  evidence from a qualitative picture.
- Close with scope and availability: this optimizes displacement to regular
  grid targets, not semantic neighborhoods; the evidence is limited and there
  is no global accuracy guarantee. Give package and repository links.

**Figure 2 — From a shuffle to a layout.** Retain `multiscale_action.png` as a
compact strip showing broad placement followed by refinement. Its job is to make
the progression visible, not establish an accuracy or convergence guarantee.
Do not repeat it in the appendix.

**Table 1 — Final quality and runtime.** Retain all five families, both runtimes,
and readable precision. Specify the different stopping conditions in the caption.
Do not collapse the tradeoff into a single speedup or improvement factor.

**No separate conclusion:** the final paragraph should state the use case and
limits, rather than repeat the abstract.

---

# Appendices — one home for each detailed question

Main-text highlights may summarize appendix findings, but do not reproduce whole
figures or repeat the method explanation. The destinations below are proposed
section groupings, not a requirement to create a subsection for every bullet.

## A. Algorithm, implementation, and scope

**Question:** exactly what does the solver do, and what can it guarantee?

1. **Specification:** coordinates, assignment direction, owner array, strided
   subgrids, phase offsets, independent rectangular strides, exact schedule,
   and compact pseudocode including acceptance and stopping. Include the locality
   bottleneck argument here, distinguishing limited reach from a convergence bound.
2. **Properties:** feasibility, strict descent, exact-arithmetic termination,
   numerical tolerance, and the meaning of a scheduled-window fixed point.
   Separate per-round work, initial schedule length, and unbounded polishing.
3. **Boundaries:** the seven-cell example and diagram explaining trailing
   clipping; distinguish connectivity from the existence of an improving path.
4. **Other grid shapes:** rectangular grids, fixed occupied subsets, spare cells,
   and boolean masks. Explain that occupied-cell selection is not optimized.
5. **Implementation:** native small solver, parallel phases, on-demand costs,
   storage accounting, time budgets, optional polishing, API, and versioned code.

**Owns:** boundary and mask diagrams. Benchmark-specific normalization and timers
belong in B, not repeated here. Long reproduction commands remain in the README.

## B. Evaluation protocol and comparative results

**Question:** how are the numbers measured, and what do the resulting assignments
look like compared with other methods?

**Protocol and references:**

1. Inputs, five distribution families, sizes, instance/initialization seeds,
   normalization, hardware, thread settings, and sample sizes.
2. Reference gaps, auction validation against dense solvers, and the absence of
   independent optimality certificates. Separate reference comparisons from
   progress relative to a run's own final cost.
3. Runtime inclusion/exclusion rules, stopping, timing repeats versus independent
   inputs, aggregation, completed-round checkpoints, and provenance limitations.
   Keep the common reproduction definitions here.

**Comparisons and application:**

4. Full five-family gallery: RasterFairy, BSP-OT, ShuffleSnap, and reference
   assignments; variation across instances and limits of visual claims.
5. Full BSP-OT table: settings, short-budget advantage, checkpoint sensitivity,
   and comparator scope. Describe the comparator once here; initializer and
   stress-test sections refer back and give only their specific differences.
6. MNIST: embedding, full atlas, detail crop, and its convergence trace. Define
   neighborhood metrics here and compare them separately from displacement.
   Treat a single example as limited evidence of visual quality.

**Owns:** common protocol, reference validation, distribution gallery, detailed
comparison table, and MNIST figures. The main endpoint table is a summary, not a
replacement for these details. Historical stress-test caveats are recorded here;
D points to them when discussing those measurements.

## C. Convergence and design choices

**Question:** why use strides, this schedule, random initialization, and 6×6 windows?

1. **How descent progresses:** convergence curves, including the local-only
   comparison. Refer back to main Figure 2 for the qualitative progression;
   distinguish early structural correction from late numerical polishing.
2. **Schedules and starts:** reversed, shortened, and repeated schedules;
   random, raster, RasterFairy, and BSP-OT starts within the experiments actually
   performed. Identify coupled changes and retain random-seed variation.
3. **Window size and threads:** measured quality/runtime tradeoffs, partial
   threshold successes, throughput limitations, and deterministic final output.

**Owns:** synthetic convergence, initialization, and ablation figures; the
application-specific MNIST trace stays in B.
Avoid treating the existing experiments as a full factorial comparison or a
proof that the defaults are best for all inputs.

## D. Scaling and memory

**Question:** what happens as the number of points grows?

1. Thousands to millions: per-round work, total convergence time, early-stop
   behavior, and memory. Clearly mark where reference solutions are available.
2. Billion-point stress test: consolidate runtime, memory, progress, and the
   visualization. Explain the custom BSP-OT port and unequal configurations;
   point to the historical-provenance limitations in B.

**Owns:** scaling plots and billion-point figure. The stress test demonstrates
feasibility, not certified accuracy or controlled equal-budget superiority.

## E. Relationship to prior work

**Question:** how does this objective and mechanism differ from nearby methods?

1. Exact and approximate geometric assignment; distinguish Euclidean-distance
   results from squared-distance guarantees.
2. Spatial construction and transport: RasterFairy, BSP-OT, soft versus bijective
   solutions, and coarsening a problem versus changing its neighborhoods.
3. Visual sorting and grid layouts: LAS/FLAS, feature-space and perceptual
   objectives, and assigning every item versus selecting representatives.

The introduction retains the essential positioning. This appendix supplies depth,
not a missing novelty argument. Use one bibliography for main text and appendices.

---

## Figure and table inventory

Each current figure has one destination or is explicitly superseded. This inventory records the routing used in the rewrite.

| Existing asset | Proposed destination or action |
|---|---|
| `method_overview.pdf` | Superseded by new main Figure 1. |
| `windows_and_phases.png`, `strided_windows.png` | Source material for main Figure 1; consolidate/redraw rather than reuse as extra figures. |
| New before/after reassignment panel | New illustrative drawing for Figure 1(a); no new benchmark. |
| `multiscale_action.png` | Main Figure 2 only. |
| `headline_table.tex` | Main Table 1; retain both quality and runtime. |
| `boundary_windows.pdf`, `masks.png` | A: boundary behavior and occupied subsets. |
| `distributions.png`, `bspot_table.tex` | B: gallery and detailed comparison. |
| `mnist_atlas_full_with_crop_box.png`, `mnist_atlas_detail.png`, `mnist_trace.png` | B: MNIST example and its metrics. |
| `convergence.png`, `init.png`, `ablations.png` | C: convergence and design choices. |
| `scaling.png`, `billion_interp75.png` | D: scaling and stress test. |

**Work implied by this outline:** most appendix content already exists and needs
moving, deduplication, and cross-reference updates. New editorial work is the
coordinated Figure 1, its caption, and the shorter main explanation/positioning.
This is not a request for new experimental evidence.

**Review this outline by changing the choices at the top, section purposes,
content destinations, or visual budget.** The manuscript follows these choices; future revisions can begin here. This outline does not authorize
new experiments or import the random-cell-group investigation into the paper.
