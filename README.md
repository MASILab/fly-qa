# PNGQAFly (fly_qa)

**EXPERIMENTAL / UNVALIDATED.** This is a research/art exercise, not a
production QA tool -- say that up front, the same way
[DoomFly](https://github.com/nftechie/doomfly) and
[StonkFly](https://github.com/nftechie/stonkfly) do about themselves.

The QA workflow itself -- the `QA.csv` schema (`filename, QA_status, reason,
user, date, duration`), the `yes`/`no`/`maybe` status convention, and both
folder layouts this project reads (flat `{study}/QA.csv` with PNGs directly
in the study folder, and nested `{study}/{process}/QA.csv` per its BIDS
mode) -- is based on
[MASILab/masi-qa](https://github.com/MASILab/masi-qa), a keyboard-driven
tool for rapid *human* QA review. This project automates that same
workflow's output format with a connectome-driven decision instead of (or
alongside) a human reviewer; it does not replace masi-qa's own review UI.

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

## Apples-vs-oranges spike

A simpler sanity-check task than TractSeg QA -- see
[`docs/model.md`](docs/model.md#apples-vs-oranges-spike-scriptsgenerate_fruit_demopy)
for the full story. The encoder fix it drove (confidence-gated
chrominance) was re-measured against real TractSeg data with the full
calibration+held-out protocol and turned out to be a net win there too
(held-out sensitivity roughly doubled, 4.8% -> 9.7%) -- an earlier,
smaller diagnostic had wrongly suggested a regression; see `docs/model.md`
for why.

```
python scripts/generate_fruit_demo.py --output-dir .devtest/fruit_demo --n-per-class 50
python scripts/calibrate_and_validate_folders.py \
    --good-dir .devtest/fruit_demo/apples --bad-dir .devtest/fruit_demo/oranges \
    --connectome-export ~/.cache/fly_qa/connectome_export
```

`calibrate_and_validate_folders.py` is generic (any two-folder good/bad PNG
dataset), not fruit-specific.

## Retuning on your own QA dataset

To calibrate (or re-calibrate) decoder thresholds against any labeled
dataset in the masi-qa format, holding out a study to test on:

```
# 1. Calibrate on every study except the one(s) you hold out.
#    --exclude-study is required (repeatable) -- there is no default, so a
#    held-out study can never be silently included by forgetting the flag.
python scripts/calibrate_on_real_labels.py <path-to-your-dataset>/ \
    --exclude-study "study #1" \
    --connectome-export ~/.cache/fly_qa/connectome_export

# 2. Validate on exactly the study you excluded -- the honest, unbiased number.
python scripts/validate_on_real_labels.py "<path-to-your-dataset>/study #1" \
    --connectome-export ~/.cache/fly_qa/connectome_export
```

Both scripts auto-detect either masi-qa layout per study folder -- flat
(`{study}/QA.csv` + PNGs directly in the study folder) and nested/BIDS mode
(`{study}/{process}/QA.csv`, process folders matched by `--process-glob`,
default `Tractseg_*` for backward compatibility with the original TractSeg
dataset this project was built against -- pass `--process-glob "*"` or your
own pattern for a differently-named pipeline). A study-level `QA.csv` whose
filenames are path-prefixed (e.g. `BRAID/sub-....png`) is treated as an
aggregate rollup of a nested layout, not a flat one, and skipped to avoid
double-counting those images.

Calibration writes to `src/fly_qa/data/decoder_thresholds.json` by default
(`--output-thresholds <path>` to save elsewhere; pass the matching
`--thresholds <path>` to `validate_on_real_labels.py` and to `fly-qa`
itself to use it). `fly-qa` also accepts `--confidence-fail-max` /
`--confidence-flag-max` to override threshold values directly without a
calibration file, for quick experimentation.

**Practical notes, from doing this on real data:** pick a held-out study
with a real mix of both classes -- a study with zero `no`/`maybe` examples
tells you nothing about recall. Runtime is roughly 1 real second per image
through the actual connectome simulation, so a few thousand images across
calibration + held-out is tens of minutes, not seconds. `--max-per-class`
caps the `yes` sample size for runtime (all non-`yes` examples are always
used, since real datasets are typically 95%+ `yes`).

## Validation protocol

The build spec's own Section 5 protocol -- generic synthetic PNG defects
(color shift, banding, corruption, alpha loss, resolution mismatch) rather
than real labeled data -- is a separate path, useful mainly as a sanity
check independent of any specific dataset:

```
python scripts/run_validation.py --good-images-dir <dir-of-known-good-pngs> \
    --connectome-export ~/.cache/fly_qa/connectome_export
```

See [`docs/model.md`](docs/model.md) for full numbers from both protocols.
**The honest result on real QA data is still negative: held-out recall on
real QA failures is ~5-15%** (currently 9.7% on real TractSeg labels after
the confidence-gating encoder fix, up from 4.8% before it -- see
`docs/model.md` for the full before/after). The decoder was fixed to
threshold on `confidence_signal` rather than `defect_score` after a
user-reported accuracy bug traced to a real cause (the original DNp20 left-right
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
