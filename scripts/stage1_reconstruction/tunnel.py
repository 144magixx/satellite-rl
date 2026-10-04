"""Visible workbench forwarding, isolated from scientific worker lifetime."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from remote import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_stage1 import tunnel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, choices=(20418,20424), required=True)
    args = parser.parse_args()
    api_port = {20418:8894,20424:8895}[args.port]
    client = connect(args.port)
    try:
        tunnel(client, SimpleNamespace(local_port=api_port, remote_port=api_port))
    finally:
        client.close()


if __name__ == '__main__':main()
