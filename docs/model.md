# The Fly Who Does QA - Model

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
evidence it was actively wrong for this encoder.
`defect_score` is still computed and shown, for
transparency and future work, but does not drive the verdict. Thresholds
(`DecoderThresholds`) are placeholders (`calibrated=False`) until a
calibration script has run against a labeled set -- every log line and the
CLI banner say so explicitly while `calibrated` is false.

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
