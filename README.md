# PNGQAFly (fly_qa)

**EXPERIMENTAL / UNVALIDATED.** This is a research/art exercise, not a
production QA tool -- say that up front, the same way
[DoomFly](https://github.com/nftechie/doomfly) and
[StonkFly](https://github.com/nftechie/stonkfly) do about themselves.

A connectome-driven image QA scanner: the retained MaleCNS v1.0 fly-brain
connectome (~165K neurons, ~25.6M synapses, used **unmodified**) sits
between a task-specific sensory encoder (image -> photoreceptor input) and
a fixed descending-neuron decoder (neuron activity -> pass/fail/flag
verdict). The connectome itself is inert with respect to "defect" -- all
task-specificity lives in the encoder and decoder. See
[`docs/model.md`](docs/model.md) for exactly what the model does and its
known limitations, and Section 5/6 of the build spec for the validation
protocol this project follows.

**No accuracy/precision/recall claim is meaningful until
`scripts/run_validation.py` has actually been run against a held-out
labeled set** -- and even then, report results as "matches this rule on
this calibration set," never "learned to detect defects."

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set `NEUPRINT_API_TOKEN` in a `.env` file (see `.env.example`) -- required,
since the connectome is the actual classifier here, not decoration.

## Fetch the connectome (one-time, or to refresh)

```
python scripts/fetch_connectome.py --output-dir ~/.cache/fly_qa/connectome_export
```

This reads the retained connectome from neuprint (`neuprint-cns.janelia.org`,
dataset `male-cns:v1.0`) via `fetch_traced_adjacencies` plus a supplementary
`fetch_neurons()` pull for `predictedNt` (needed for excitatory/inhibitory
sign assignment, which the bare adjacency export doesn't include), and
writes `neurons.csv`, `neurons_full.csv`, `roi-connections.csv`,
`total-connections.csv`, and a hashed `manifest.json` for reproducibility.
It only reads -- it never writes back to neuprint. Takes 20-40+ minutes
(MaleCNS has ~6.5x more neurons than hemibrain, which this function was
originally sized for).

## Usage

```
fly-qa /path/to/pngs
```

Accepts a single PNG or a directory (scanned recursively). Every image
first goes through the non-neural pre-check layer (Section 4: PNG
validity, dimensions, alpha, duplicates, statistical anomalies) --
findings from this layer are tagged separately from the connectome
decoder's output and are never credited to the "fly." Only images that
pass the pre-check are fed to the connectome simulation. Results are
written to `--output` (default `fly_qa_results.csv`).

**Verdicts are pass / fail / flag only -- never auto-delete, auto-publish,
or auto-block.** `flag` means route to a human reviewer.

## Live dashboard

Running `fly-qa` on a folder auto-launches a local web dashboard (pass
`--no-dashboard` to run headless, or `--port` to pin the port). It shows:
the image currently being processed, a real sampled subset of the actual
MaleCNS v1.0 connectome (input R1-R6/R8 photoreceptors, a sample of
high-degree "hidden" neurons, and the DNp20/DNpe017 outputs) animating
**real simulated activity** as the rate model settles for that image (not
a decorative animation), the running pass/flag/fail tally and progress,
and a results table. Click any row's filename to review that image, its
scores, and a replay of its real neuron activity, even after inference has
moved on to later images.

## Validation protocol

Two separate validation paths -- see [`docs/model.md`](docs/model.md) for
full numbers and what each actually tests:

```
# Section 5's own protocol: generic synthetic PNG defects
python scripts/run_validation.py --good-images-dir <dir-of-known-good-pngs> \
    --connectome-export ~/.cache/fly_qa/connectome_export

# Calibrate/validate against the real TractSeg failure mode instead
python scripts/calibrate_on_real_labels.py examples/ \
    --connectome-export ~/.cache/fly_qa/connectome_export
python scripts/validate_on_real_labels.py examples/study_226 \
    --connectome-export ~/.cache/fly_qa/connectome_export
```

Both write calibrated thresholds to `src/fly_qa/data/decoder_thresholds.json`
(confidence-based -- see below).

**The honest result, on both paths, is still negative: held-out recall on
real QA failures is ~5-15%.** The decoder was fixed to threshold on
`confidence_signal` rather than `defect_score` after a user-reported
accuracy bug traced to a real cause (the original DNp20 left-right
difference readout was measurably *destroying* signal an unsigned sum
preserves -- see `docs/model.md`), which is a genuine ~2x improvement, but
the deeper bottleneck is in the encoder, not the decoder, and remains open.
Report any future runs the same way, including if they're still bad.

The optional plasticity loop (Section 6) is not implemented -- it is
explicitly gated on this validation existing first.

## Development

```
pytest -q
```

Modules kept small and independently testable: `precheck.py` (Section 4),
`connectome_data.py` (loads + signs + normalizes the exported graph),
`encoder.py` / `simulator.py` / `decoder.py` (Sections 2-3),
`validation.py` (Section 5), `viz.py` (picks the dashboard's real
visualization subset), `events.py` / `webapp/` (the live dashboard).
