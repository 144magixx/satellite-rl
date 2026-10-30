"""从某账号第三轮的 extra_runs.json 里去掉（或加入）若干运行，并重写该账号队列。
  python scripts/s2lab/edit_extra.py --account B --remove r3-dqn-s2-c30 r3-dqn-s3-c30
  python scripts/s2lab/edit_extra.py --account B --add-json runs.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ['S2LAB_ROUND'] = '3'
sys.path.insert(0, str(Path(__file__).parent))
import remote2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--account', required=True)
    ap.add_argument('--remove', nargs='*', default=[])
    ap.add_argument('--add-json')
    a = ap.parse_args()
    p = remote2.paths(a.account)
    path = f"{p['runs']}/extra_runs.json"
    with remote2.connect(remote2.first_port(a.account)) as c:
        with c.open_sftp() as sftp:
            with sftp.open(path) as f:
                runs = json.loads(f.read())
            runs = [r for r in runs if r['run_id'] not in a.remove]
            if a.add_json:
                runs += json.loads(Path(a.add_json).read_text(encoding='utf-8'))
            with sftp.open(path, 'w') as f:
                f.write(json.dumps(runs, ensure_ascii=False))
        out = remote2.run(c, p['env'] + f"{p['py']} scripts/s2lab/write_matrix.py --runs-dir {p['runs']} --round 3 --account {a.account} --extra {path}")
        print('追加运行', [r['run_id'] for r in runs], '矩阵', out.strip())


if __name__ == '__main__':
    main()
