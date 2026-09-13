# PNGQAFly (fly_qa)

**EXPERIMENTAL / UNVALIDATED.** A connectome-driven image QA scanner,
following the same architecture as nftechie/doomfly and nftechie/stonkfly:
the retained MaleCNS v1.0 connectome (~165K neurons, ~25.6M synapses, used
UNMODIFIED) between a task-specific sensory encoder and a fixed
descending-neuron decoder. This is a research/art exercise, not a
production QA tool -- see `docs/model.md` for what it actually does, its
known limitations, and real (currently negative) validation results.

## Code structure
- Use Python
- Run with `fly-qa /path/to/pngs` (a single PNG or a directory, scanned recursively)
- Requires a connectome export first: `python scripts/fetch_connectome.py`
  (writes to `~/.cache/fly_qa/connectome_export` by default; reads from
  `neuprint-cns.janelia.org`, dataset `male-cns:v1.0` -- note this is NOT
  the default `neuprint.janelia.org` server)
- Pipeline per image: non-neural pre-check (`precheck.py`) -> sensory
  encoder (`encoder.py`) -> frozen-weight rate simulation (`simulator.py`,
  `connectome_data.py`) -> descending-neuron decoder (`decoder.py`) ->
  verdict written to a CSV (`--output`, default `fly_qa_results.csv`)
- Verdicts are `pass` / `fail` / `flag` -- never auto-delete, auto-publish,
  or auto-block; `flag` routes to human review
- Auto-launches a live web dashboard (`--no-dashboard` to disable) showing
  the current image, a real sampled subset of the connectome animating
  actual simulated activity (not decorative), and pass/flag/fail results;
  click a row to review that image and replay its activity later
- Validation: `python scripts/run_validation.py --good-images-dir <dir>`
  builds a synthetic-defect calibration/held-out split and reports
  sensitivity/specificity honestly (see `docs/model.md` for the numbers
  from the run already done)
- `examples/study_*` contain real TractSeg tractogram PNGs with historical
  human QA labels (yes/no/maybe) from a prior, now-abandoned heuristic
  approach to this project -- they're incidentally useful as a source of
  real "known-good" images for `scripts/run_validation.py`, not as a QA
  target format this tool still speaks. `study_226` was reserved as a
  held-out set for that abandoned approach; no such reservation applies
  to the current connectome-driven pipeline.

## Project goals
- Reuse the real MaleCNS v1.0 connectome for [MaleCNS v1.0](https://male-cns.janelia.org/download/)
  unmodified as the classifier substrate -- no cropping, no retraining
  connection weights from scratch
- All task-specificity lives in the encoder/decoder, never in the connectome itself
- Never claim accuracy/precision/recall without running the validation
  protocol first, and report results as "matches this rule on this
  calibration set," not "learned to detect defects"
- Ship negative results honestly, the DoomFly way, rather than only shipping the demo
