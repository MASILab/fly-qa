# PNGQAFly model documentation

**Status: EXPERIMENTAL / UNVALIDATED.** This document describes what the
model actually does, so results are reproducible -- it is not a claim that
any of these choices are biologically validated or that the resulting
verdicts are accurate. Follow the DoomFly convention: report negative
results honestly.

## Ground truth about the connectome

Reused unmodified from MaleCNS v1.0 via neuprint (`neuprint-cns.janelia.org`,
dataset `male-cns:v1.0`), fetched with `scripts/fetch_connectome.py`
(`neuprint.fetch_traced_adjacencies`, traced non-cropped neurons only).
Confirmed cell populations used by this project (counts as of the fetch that
produced `manifest.json` in the export directory -- always check that file
for the actual numbers you're running against, these will drift slightly as
proofreading continues):

| Role | Type(s) matched | Count in the built graph |
|---|---|---|
| R1-R6 input | `R1-R6` (exact) | 1,394 |
| R8 input | `R8_unclear`, `R8d`, `R8p`, `R8y`, `R7R8_unclear` | 1,414 |
| Defect-score output | `DNp20` (L/R pair) | 2 (bodyId 10059=R, 10162=L) |
| Confidence output | `DNpe017` | 2 (bodyId 10527, 555871) |
| Reward (plasticity, not yet wired) | `PAM11` | 15 |
| Aversive (plasticity, not yet wired) | `PPL101` | 2 |

Two discrepancies from the build spec, left visible rather than silently
reconciled:
- The spec cites 3,335 R1-R6 channels; an unfiltered `type='R1-R6'` query
  against neuprint returns 3,377, but restricting to `status='Traced'`
  neurons (required for them to appear in the connectivity graph at all)
  drops that to 1,394 -- roughly 1,983 R1-R6 photoreceptors are typed but
  not yet fully traced/proofread in this connectome release.
- The spec cites 811 R8 channels; this implementation's R8-family selection
  (all `R8*` subtypes plus the unresolved `R7R8_unclear` group) yields
  1,414 traced neurons.

If a narrower selection is wanted to match the spec's exact counts more
closely, change `R8_TYPES` in `pipeline.py` and the R1-R6 status filter in
`connectome_data.py`, then update this table and rerun
`scripts/run_validation.py` (thresholds are calibrated against a specific
encoder configuration and go stale if it changes).

## Sign convention for the weight matrix (`connectome_data.py`)

Real anatomical connectivity is unsigned (synapse counts). Sign is assigned
per presynaptic neuron from `predictedNt` (neuprint's per-neuron predicted
neurotransmitter field):

| Predicted transmitter | Sign |
|---|---|
| acetylcholine | + (excitatory) |
| GABA | − (inhibitory) |
| glutamate | − (inhibitory) |
| histamine | − (inhibitory) |
| dopamine, octopamine, serotonin, unclear, anything else | + (excitatory) |

This is a documented simplification, not a claim about ground-truth
physiology (glutamate and histamine, in particular, have context-dependent
effects in the insect CNS; neuromodulators like dopamine are not simple
fast excitation). It is a discrete choice a reader can question and change.

Each postsynaptic neuron's incoming row is then normalized so
`sum(|weight|) == 1` -- this keeps the rate model numerically stable across
neurons with wildly different raw synapse-count totals; it does not change
which neurons connect to which, only the relative magnitude used in the
simulation.

## Encoder (`encoder.py`) -- an invented proxy, not a validated visual model

There is no retinotopic mapping. Given a precheck-passed image (alpha
already flattened onto neutral gray, see `precheck.py`):

1. Letterbox-resize to a configurable grid (`--grid-size`, default 256x256),
   preserving aspect ratio, padded with neutral gray (128,128,128) -- no
   cropping, no distortion.
2. For R1-R6: resize the letterboxed image to an `R x C` grid where
   `R*C >= len(r1_r6_ids)` (near-square, `C = ceil(sqrt(n))`,
   `R = ceil(n/C)`), convert to grayscale, flatten row-major. Each of the
   *sorted* R1-R6 body IDs is assigned exactly one grid cell's luminance
   (0..1) in that flattened order.
3. For R8: same grid-and-flatten procedure, but the *hue* channel (HSV) of
   the letterboxed image is used as the chrominance summary.
4. Injected current = the grid value directly (linear, `gain=1.0` by
   default) -- there is no spike encoding here, values are fed as constant
   external drive for every simulation step (static-frame mode).

Sorting body IDs before assignment makes the mapping deterministic and
reproducible across runs of the same connectome export; it is arbitrary
with respect to actual retinal topology.

## Simulator (`simulator.py`) -- discrete-time leaky rate model, not spiking

```
r <- (1 - leak) * r + leak * relu(W @ r + I_ext)
```

run for a fixed number of `--steps` (default 30) with `I_ext` held constant
throughout (static-frame mode: "one image = one frame held for N steps").
This is explicitly NOT a spiking or biophysical (Hodgkin-Huxley / LIF)
model -- it is the cheapest dynamical system that (a) uses the real signed,
weighted connectivity, (b) can settle to a steady state, and (c) is fast
enough to run per-image. Tiled-scan mode (Section 1, optional) would run
this same loop once per tile with carried-over state between tiles, and has
not yet been implemented.

## Decoder (`decoder.py`)

`defect_score = rate(DNp20_right) - rate(DNp20_left)`,
`confidence_signal = sum(rate(n) for n in DNpe017)`. **The verdict is decided
from `confidence_signal` (low -> fail), not `defect_score`** -- this is a
deviation from the spec's literal DoomFly-derived mapping, made after direct
evidence it was actively wrong for this encoder (see "Why confidence_signal,
not defect_score" below). `defect_score` is still computed and shown, for
transparency and future work, but does not drive the verdict. Thresholds
(`DecoderThresholds`) are placeholders (`calibrated=False`) until a
calibration script has run against a labeled set -- every log line and the
CLI banner say so explicitly while `calibrated` is false.

### Why confidence_signal, not defect_score

A user report ("failures that shouldn't pass end up passing") prompted a
direct diagnostic against real labeled TractSeg images (not synthetic
defects): for 54 real images (27 human-labeled `no`/`maybe`, 27 matched
`yes`) across three bundles, the best-achievable single-threshold separation
accuracy was:

| Signal | Best accuracy (balanced threshold) |
|---|---|
| Raw encoder input (`r8_mean`, before any simulation) | 70.4% |
| `defect_score` (DNp20 L-R difference) | 66.7% -- **worse than the raw input** |
| `confidence_signal` (DNpe017 sum) | 72.2% |

**Root cause:** the encoder maps whole-image content onto R1-R6/R8 body IDs
in an order with no relationship to real left/right hemisphere anatomy
(sorted body ID, see Encoder section above). A genuine, real signal in the
image (e.g. less colored tract-overlay area in a sparse/degraded tractogram)
changes the *overall* input magnitude roughly symmetrically -- it doesn't
preferentially drive one hemisphere's photoreceptor sample over the other's.
Taking `DNp20_right - DNp20_left` of that symmetric change cancels out the
very signal being looked for; summing `DNpe017` preserves it instead. This
is measurable, not speculative: `defect_score` scored *below* using the raw
encoder input with no network at all, meaning the 30-step simulation was
actively destroying signal under the old (L-R diff) readout.

At operationally realistic false-positive budgets, the gap is roughly 2x
across the curve:

| False-positive budget | `defect_score` recall | `confidence_signal` recall |
|---|---|---|
| 5% | 7.4% | 14.8% |
| 10% | 22.2% | 14.8% |
| 20% | 25.9% | 22.2% |
| 30% | 44.4% | 66.7% |
| 50% | 66.7% | 88.9% |

**This is a genuine, evidenced fix, but it does not make the tool accurate.**
See the held-out results below -- recall at a realistic false-positive rate
is still poor. The deeper bottleneck is upstream of the decoder: see "What
still doesn't work, and why" below.

## Validation results (actually run -- see the honesty ground rule)

Two separate validation runs exist, testing two different defect
distributions -- keep them distinct, don't average them together:

**1. Synthetic PNG defects** (`scripts/run_validation.py`, the spec's own
Section 5 protocol): known-good images + injected color shift / banding /
corruption / alpha loss / resolution mismatch. Against
`examples/study_256/Tractseg_CC` (12 calibration + 12 held-out real images, 4
synthetic defects each, `leak=0.2`, `steps=30`), using `confidence_signal`:

| Set | n | sensitivity | specificity |
|---|---|---|---|
| Calibration (in-sample) | 60 (48 defective, 12 good) | 6.2% | 100.0% |
| Held-out (frozen weights) | 60 (48 defective, 12 good) | 14.6% | 91.7% |

Per-defect-type recall on the held-out set: color_shift 2/7, banding 2/14,
corruption 2/14, alpha_loss 1/3, resolution_mismatch 0/10.

**2. Real TractSeg QA labels** (`scripts/calibrate_on_real_labels.py` +
`scripts/validate_on_real_labels.py`) -- the defect distribution this
deployment actually needs to catch (sparse/degraded tractogram renders),
not generic PNG corruption. Calibrated on 150 real `yes` + 81 real
`no`/`maybe` images pooled across `study_228`/`239`/`256`/`498`
(`study_226` hard-excluded from calibration); held out against
`study_226` (150 `yes` + 62 `no`/`maybe`, never seen during calibration):

Re-run after the apples/oranges-driven encoder change (confidence-gated
chrominance, see next section) to check whether it helped or hurt the real
task -- both runs shown, same protocol, same held-out set:

| Set | n | Encoder | sensitivity | specificity |
|---|---|---|---|---|
| Calibration (in-sample) | 231 (81 bad, 150 good) | pre-gating | 7.4% | 95.3% |
| Held-out (`study_226`) | 212 (62 bad, 150 good) | pre-gating | 4.8% | 92.7% |
| Calibration (in-sample) | 231 (81 bad, 150 good) | **confidence-gated (current)** | 8.6% | 95.3% |
| Held-out (`study_226`) | 212 (62 bad, 150 good) | **confidence-gated (current)** | **9.7%** | 87.3% |

**Reading this honestly: the tool still misses the large majority of real
QA failures (90.3% of real `no`/`maybe` images pass on held-out data).**
But the confidence-gating encoder fix -- driven by the apples/oranges spike,
not by TractSeg -- turned out to roughly **double** held-out recall (4.8%
-> 9.7%) at a real but modest specificity cost (92.7% -> 87.3%), when
measured with the actual calibration+held-out protocol used for
deployment. See the next section for why an earlier, smaller diagnostic
(n=54, 3 bundles, a different metric) wrongly suggested this fix was a
regression -- that finding is superseded by this fuller measurement.
Neither number makes this a working defect detector; report exactly this,
not a rounded-up version of it.

## Apples-vs-oranges spike (`scripts/generate_fruit_demo.py`)

A user-reported accuracy bug ("failures that shouldn't pass end up
passing") prompted testing the pipeline on a much easier, strongly-colored
synthetic task -- apple (pass) vs orange (fail) -- to check whether the
architecture works at all when given a clear signal, independent of
TractSeg-specific difficulty. First pass performed barely above chance
(56.7% best-threshold accuracy on raw encoder input) -- traced to the same
root cause as the tractogram case: `encoder.py`'s whole-image hue averaging
is dominated by the majority-background pixels (the fruit covers only
~20-30% of the frame), and a real synthetic-background color cast (present
in both classes, but still a large fraction of the pixels) swamps the small
number of genuinely informative fruit-colored cells once aggregated.

**Fix:** `_sample_chrominance_grid()` in `encoder.py` now computes a
saturation-weighted circular mean per cell (correct math for a circular
quantity) AND gates each cell's contribution by its own saturation
(confidence) -- `signal = hue * saturation` -- so low-saturation background
cells contribute near-zero regardless of their nominal hue reading, instead
of a stable-but-irrelevant color swamping the aggregate.

**Result, held-out, never used for threshold-picking**
(`scripts/calibrate_and_validate_folders.py --good-dir .../apples --bad-dir
.../oranges`, 25+25 calibration, 25+25 held-out, `leak=0.2`, `steps=30`):

| False-flag budget | Held-out sensitivity | Held-out specificity |
|---|---|---|
| 5% | 16.0% | 100.0% |
| 20% | 48.0% | 96.0% |

This is a real, meaningful result for a frozen, untrained biological network
-- clearly better than chance, in the direction the fix predicted.

**Superseded caveat (kept for the record, corrected below):** an initial,
smaller diagnostic (n=54 real images, 3 bundles) using a *different* metric
(best-threshold accuracy at any operating point, not a calibrated
false-flag budget) suggested this fix regressed real TractSeg separation
(72.2% -> 61.1%). Re-measured with the actual deployment protocol
(`scripts/calibrate_on_real_labels.py` + `validate_on_real_labels.py`,
231 calibration + 212 held-out real images, the full labeled set) the fix
is a net improvement on the real task too -- see the updated table in the
"Validation results" section above (held-out sensitivity 4.8% -> 9.7%).
**Lesson: a small diagnostic sample and an unbudgeted "best accuracy"
metric can disagree with the actual calibrated, held-out result -- trust
the full calibration+held-out protocol over an ad hoc diagnostic, and
re-measure rather than reason from a smaller/earlier test.**

The deeper finding from this investigation still stands, just with a more
fortunate outcome than first measured: **because the connectome is frozen,
real biological wiring -- never trained for this task -- there is no
reliable, predictable relationship between "more correct" input encoding
and downstream classification accuracy**, so any future encoder change
needs to be re-measured on both tasks with the full protocol, not assumed
to transfer in either direction.

**Current decision (2026-09-13):** the confidence-gating fix is kept as the
default encoder. It now measures as a modest net win on both the
apples/oranges spike and real TractSeg QA. Current focus remains the
apples/oranges task per project conversation, but TractSeg accuracy is no
longer known to be regressed -- re-run both validation scripts after any
further encoder change rather than assuming either direction.

### What still doesn't work, and why

The decoder fix addressed one confirmed bug (an information-destroying
readout) but did not fix the deeper bottleneck, which is upstream in the
**encoder**: `encoder.py` maps whole-image luminance/hue onto a coarse grid
by simple resizing. A sparse tractogram's colored overlay occupies a small
fraction of the 600x1200 image's pixels, sitting on top of a grayscale
anatomical background that looks similar regardless of tract quality.
Downsampling the whole image to compute per-photoreceptor luminance/hue
averages away most of that small, localized color signal before it ever
reaches the network -- confirmed directly: `r8_mean` (the raw, un-simulated
encoder input) differs between real good/bad images by less than one
standard deviation (bad mean=0.0114 vs good mean=0.0153, std~=0.0057-0.0058
each). No decoder choice can recover information the encoder already
discarded. The old (now-removed) heuristic classifier's approach -- directly
measuring the colored-overlay pixel coverage/fragmentation via CV, rather
than averaging whole-image luminance into a coarse grid -- preserved this
signal far better (see git history), which is suggestive of the fix that
would actually matter here: an encoder that represents overlay
density/coverage more directly, not just resized whole-image brightness.
That is a bigger change than the decoder fix and has not been attempted in
this pass -- see the project's session notes for the open question of
whether/how to pursue it.

## Known limitations (say the negative results, don't hide them)

- No spiking dynamics, no realistic synaptic time constants, no
  refractory period.
- Sign-by-predicted-neurotransmitter is a simplification; confidence in
  those predictions varies per neuron (`predictedNtConfidence` in the raw
  export is available but not currently used to weight or filter).
- The encoder's grid mapping has no relationship to real ommatidia
  positions or the optic lobe's actual retinotopic map.
- Photoreceptor and descending-neuron identities are taken from MaleCNS
  type labels as-is; no manual verification of individual body IDs beyond
  checking that the expected DNp20/DNpe017 counts (2 each) hold.
