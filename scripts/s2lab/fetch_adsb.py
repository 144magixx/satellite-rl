"""远端运行：并行分段下载 adsb.lol 某日 prod-0 日包（大于 2 GB 的分成 .aa/.ab），拼成一个 tar，再抽取飞机位置帧。
  python scripts/s2lab/fetch_adsb.py --dates 2024-06-12 2024-08-14 2024-10-09 --out ~/s2lab/raw4 --frames ~/s2lab/frames4
"""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _common import ROOT  # noqa: F401

BASE = 'https://github.com/adsblol/globe_history_2024/releases/download'
PART = 16 * 2 ** 20


def size_of(url):
    """资产字节数；不存在的资产 GitHub 返回 9 字节占位。网络超时拿不到长度就重试（10-28 曾因一次超时得到 0、拼出空包）。"""
    for _ in range(30):
        out = subprocess.run(['curl', '-sSIL', '-m', '30', url], capture_output=True, text=True).stdout
        lines = [l for l in out.splitlines() if l.lower().startswith('content-length')]
        if lines:
            return int(lines[-1].split(':')[1])
    raise RuntimeError(f'{url} 拿不到大小')


def fetch_range(url, a, b, path):
    for _ in range(20):
        r = subprocess.run(['curl', '-sS', '-L', '-m', '600', '-r', f'{a}-{b}', '-o', str(path), url])
        if r.returncode == 0 and path.exists() and path.stat().st_size == b - a + 1:
            return
    raise RuntimeError(f'{url} {a}-{b} 下载失败')


def fetch(url, dest, threads):
    n = size_of(url)
    parts_dir = dest.parent / (dest.name + '.parts')
    parts_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(url, a, min(a + PART, n) - 1, parts_dir / f'{i:05d}') for i, a in enumerate(range(0, n, PART))]
    todo = [j for j in jobs if not (j[3].exists() and j[3].stat().st_size == j[2] - j[1] + 1)]
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda j: fetch_range(*j), todo))
    with dest.open('wb') as out:
        for *_, p in jobs:
            out.write(p.read_bytes())
    assert dest.stat().st_size == n
    for *_, p in jobs:
        p.unlink()
    parts_dir.rmdir()
    print(f'{dest.name} {n} 字节', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    from satellite_rl.coverage.s1fix.data import extract_day
    for date in a.dates:
        tag = f"v{date.replace('-', '.')}-planes-readsb-prod-0"
        tar = out / f'{tag}.tar'
        if not tar.exists():
            if size_of(f'{BASE}/{tag}/{tag}.tar') > 1000:
                fetch(f'{BASE}/{tag}/{tag}.tar', tar, a.threads)
            else:
                pieces = []
                for s in ('aa', 'ab', 'ac'):
                    url = f'{BASE}/{tag}/{tag}.tar.{s}'
                    if size_of(url) > 1000:
                        p = out / f'{tag}.tar.{s}'
                        if not p.exists():
                            fetch(url, p, a.threads)
                        pieces.append(p)
                with tar.open('wb') as f:
                    for p in pieces:
                        with p.open('rb') as src:
                            while chunk := src.read(64 * 2 ** 20):
                                f.write(chunk)
                for p in pieces:
                    p.unlink()
        if not (Path(a.frames).expanduser() / f'frames_{date}.npz').exists():
            extract_day(date, tar, Path(a.frames).expanduser())


if __name__ == '__main__':
    main()
