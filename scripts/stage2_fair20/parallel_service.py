"""Control-only service with an explicit immutable scientific source checkout."""
import argparse
from pathlib import Path
import subprocess

from satellite_rl.workbench.fair20_guard import executor_port
from satellite_rl.workbench.fair20_source_repair import ORIGIN_COMMIT, readonly_source_path
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.server import WorkbenchServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--scientific-source-root', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    actual = executor_port(args.archive_root)
    if args.port != {20418: 8950, 20474: 8953}[actual]:
        raise ValueError('control API port contradicts its physical executor')
    source = readonly_source_path(args.archive_root.parent, args.scientific_source_root, ORIGIN_COMMIT)
    def git(*parts):
        return subprocess.run(['git', '-C', str(source), *parts], check=True, capture_output=True).stdout
    if git('rev-parse', 'HEAD').decode().strip() != ORIGIN_COMMIT or git('status', '--porcelain', '--', 'src', 'data', 'pyproject.toml').strip():
        raise ValueError('science checkout must be clean at the original 2fc commit')
    app = Application(args.archive_root, source_root=source, python_executable=args.python, max_workers=1)
    server = WorkbenchServer(('127.0.0.1', args.port), app)
    app.scheduler.start()
    print('S2-F20 physical executor ' + str(actual) + '; scientific source ' + ORIGIN_COMMIT, flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        app.scheduler.stop_scheduler()
        server.server_close()


if __name__ == '__main__':
    main()
