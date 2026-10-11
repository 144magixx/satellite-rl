"""Logical ND12 loopback tunnel using a frozen physical placement file."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.stage1_nondrl12 import tunnel
from scripts.stage1_nondrl12.placement_transport import Transport


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--placement-file", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    transport = Transport.from_file(args.placement_file)
    old = tunnel.ops
    tunnel.ops = transport
    try:
        # Original lane lock/listener and loopback8943 are retained. Only the
        # SSH connection for logical74 resolves to physical73.
        return tunnel.main(remaining)
    finally:
        tunnel.ops = old


if __name__ == "__main__":
    main()
