#!/usr/bin/env python3
"""One-time (or occasional) bulk export of the retained MaleCNS v1.0 connectome.

Ground rule: reuse the graph unmodified -- this script only reads from neuprint,
it never writes back or alters connectivity. Run again only when you deliberately
want to refresh the cached export; the output is hashed so downstream consumers
can tell if it changed.

Usage:
    python scripts/fetch_connectome.py [--output-dir DIR]

Requires NEUPRINT_API_TOKEN in the environment or a .env file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fly_qa.config import NEUPRINT_DATASET, NEUPRINT_SERVER, get_neuprint_token

DEFAULT_OUTPUT_DIR = Path.home() / ".cache" / "fly_qa" / "connectome_export"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    token = get_neuprint_token()
    if token is None:
        print("error: NEUPRINT_API_TOKEN not set (check your .env file)")
        return 1

    from neuprint import Client, NeuronCriteria as NC, fetch_neurons, fetch_traced_adjacencies
    import neuprint

    client = Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, token=token)
    neuprint.set_default_client(client)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching traced adjacencies from {NEUPRINT_SERVER} ({NEUPRINT_DATASET})...")
    print("This reads the retained connectome UNMODIFIED -- no writes to neuprint.")
    print("(This step is slow at MaleCNS scale -- tens of minutes, not the 'few minutes' "
          "quoted for hemibrain, which has ~6.5x fewer neurons.)")

    t0 = time.perf_counter()
    neurons_df, roi_conn_df = fetch_traced_adjacencies(str(args.output_dir), client=client)
    elapsed = time.perf_counter() - t0

    # fetch_traced_adjacencies's own neurons.csv only has bodyId/type/instance -- no
    # predictedNt, which connectome_data.py needs for excitatory/inhibitory sign
    # assignment. A plain fetch_neurons() pull for the same fields is much faster
    # (~1 minute vs tens of minutes) since it doesn't touch the connectivity join.
    print("Fetching full neuron metadata (predictedNt, etc.) for sign assignment...")
    t1 = time.perf_counter()
    full_neurons_df = fetch_neurons(NC(status="Traced"), client=client)[0]
    full_cols = ["bodyId", "type", "instance", "predictedNt", "predictedNtConfidence"]
    full_neurons_path = args.output_dir / "neurons_full.csv"
    full_neurons_df[full_cols].to_csv(full_neurons_path, index=False)
    print(f"  done in {time.perf_counter() - t1:.0f}s")

    neurons_path = args.output_dir / "neurons.csv"
    total_conn_path = args.output_dir / "total-connections.csv"
    roi_conn_path = args.output_dir / "roi-connections.csv"

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "server": NEUPRINT_SERVER,
        "dataset": NEUPRINT_DATASET,
        "n_neurons": len(neurons_df),
        "n_connection_rows": len(roi_conn_df),
        "elapsed_seconds": round(elapsed, 1),
        "neurons_sha256": _hash_file(neurons_path),
        "neurons_full_sha256": _hash_file(full_neurons_path),
        "roi_connections_sha256": _hash_file(roi_conn_path),
        "total_connections_sha256": _hash_file(total_conn_path) if total_conn_path.exists() else None,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Done in {elapsed:.0f}s: {manifest['n_neurons']} neurons, "
          f"{manifest['n_connection_rows']} connection rows")
    print(f"Wrote {neurons_path}, {full_neurons_path}, {roi_conn_path}, {total_conn_path}, {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
