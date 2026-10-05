"""Visible workbench forwarding, isolated from scientific worker lifetime."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from remote import connect, GROUPS, group_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_stage1 import tunnel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, choices=(20418,20424), required=True)
    parser.add_argument('--group', choices=GROUPS, default='s1_03')
    args = parser.parse_args()
    api_port = group_profile(args.group)['api_ports'][args.port]
    client = connect(args.port)
    try:
        tunnel(client, SimpleNamespace(local_port=api_port, remote_port=api_port))
    finally:
        client.close()


if __name__ == '__main__':main()
