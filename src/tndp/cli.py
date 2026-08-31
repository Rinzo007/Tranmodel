"""Command line interface for TNDP route-network synthesis."""

from __future__ import annotations

import argparse
import json

from .model import NetworkDesignConfig
from .run import run_tndp


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a transit route network from OD demand")
    parser.add_argument("--min-routes", type=int, default=10)
    parser.add_argument("--max-routes", type=int, default=30)
    parser.add_argument("--corridors", type=int, default=300)
    parser.add_argument("--candidates-per-corridor", type=int, default=8)
    parser.add_argument("--surrogate-only", action="store_true", help="Do not run AequilibraE TransitAssignment")
    args = parser.parse_args()
    cfg = NetworkDesignConfig(
        min_routes=args.min_routes,
        max_routes=args.max_routes,
        corridor_top_pairs=args.corridors,
        candidate_limit_per_corridor=args.candidates_per_corridor,
        full_evaluation=not args.surrogate_only,
    )
    print(json.dumps(run_tndp(cfg, full_assignment=not args.surrogate_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
