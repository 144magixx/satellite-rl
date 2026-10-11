"""Bounded official source acquisition inside a frozen F20 workbench job."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def acquire_asset(directory, asset, progress=None, stopped=None):
    day = asset['date']
    if day not in ('2024-02-13', '2024-02-14', '2024-02-15'):
        raise ValueError('unregistered source date')
    tag, name, url = asset['release_tag'], asset['asset_name'], asset['url']
    tags = tuple('v' + day.replace('-', '.') + '-planes-readsb-' + mode for mode in ('staging-0', 'prod-0'))
    prefix = 'https://github.com/adsblol/globe_history_2024/releases/download/' + tag + '/'
    if tag not in tags or name != tag + '.tar' or url != prefix + name:
        raise ValueError('official asset identity or URL differs')
    size = asset['expected_bytes']
    if type(asset['asset_id']) is not int or type(size) is not int or not 0 < size < 3 * 1024 ** 3:
        raise ValueError('invalid official metadata dimensions')
    metadata = asset.get('official_metadata', {})
    canonical = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    if canonical != asset.get('official_metadata_canonical_sha256'):
        raise ValueError('frozen official metadata SHA mismatch')
    entries = [a for a in metadata.get('assets', []) if a.get('id') == asset['asset_id']]
    if metadata.get('tag_name') != tag or len(entries) != 1:
        raise ValueError('frozen official release metadata missing')
    official = entries[0]
    if official['name'] != name or official['size'] != size or official['browser_download_url'] != url:
        raise ValueError('asset differs from frozen official metadata')
    root = Path(directory) / 'mother_sources' / 'raw'
    root.mkdir(parents=True, exist_ok=True)
    target, partial, ledger = root / name, root / (name + '.part'), root / (day + '_download.json')
    identity = {k: asset[k] for k in ('date', 'release_tag', 'asset_id', 'asset_name', 'expected_bytes', 'url', 'official_metadata_sha256')}
    if ledger.exists():
        saved = json.loads(ledger.read_text(encoding='utf-8'))
        if saved['asset'] != identity:
            raise ValueError('resumed asset identity changed')
        if not target.exists() and partial.exists() and partial.stat().st_size == size and sha(partial) == saved['sha256']:
            partial.replace(target)
        if target.exists() and target.stat().st_size == size and sha(target) == saved['sha256']:
            return {**asset, 'path': str(target), 'sha256': saved['sha256']}
        raise ValueError('committed source bytes changed')
    if target.exists():
        raise ValueError('archive exists without immutable download ledger')
    failures = []
    for attempt in range(4):
        if stopped and stopped():
            raise InterruptedError('download stopped')
        try:
            start = partial.stat().st_size if partial.exists() else 0
            if start > size:
                raise ValueError('partial exceeds expected official size')
            if start < size:
                headers = {'User-Agent': 'satellite-rl-S2-F20', 'Accept': 'application/octet-stream'}
                if start:
                    headers['Range'] = 'bytes=' + str(start) + '-'
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
                    if start:
                        content_range = response.headers.get('Content-Range', '')
                        if response.status != 206 or not content_range.startswith('bytes ' + str(start) + '-') or not content_range.endswith('/' + str(size)):
                            raise ValueError('resume Content-Range differs')
                    elif response.status != 200:
                        raise ValueError('unexpected initial response')
                    length = response.headers.get('Content-Length')
                    if length is not None and int(length) != size - start:
                        raise ValueError('official response length differs')
                    count, last = start, 0
                    with partial.open('ab' if start else 'wb') as stream:
                        while True:
                            if stopped and stopped():
                                raise InterruptedError('download stopped')
                            chunk = response.read(2 * 1024 * 1024)
                            if not chunk:
                                break
                            count += len(chunk)
                            if count > size:
                                raise ValueError('source exceeds official size')
                            stream.write(chunk)
                            if progress and time.monotonic() - last >= 2:
                                progress({'phase': 'official_download', 'date_utc': day, 'downloaded_bytes': count, 'expected_bytes': size, 'attempt': attempt + 1})
                                last = time.monotonic()
                if count != size:
                    raise OSError('official download incomplete')
            digest = sha(partial)
            upstream = official.get('digest')
            if upstream and upstream.startswith('sha256:') and digest != upstream[7:]:
                raise ValueError('official metadata digest differs')
            save(ledger, {'asset': identity, 'path': str(target), 'sha256': digest, 'bytes': size, 'failures': failures, 'transport': 'official_https_asset_workbench_worker'})
            partial.replace(target)
            return {**asset, 'path': str(target), 'sha256': digest}
        except InterruptedError:
            raise
        except ValueError:
            raise
        except (OSError, TimeoutError) as error:
            failures.append({'attempt': attempt + 1, 'error_type': type(error).__name__})
            save(root / (day + '_download_attempts.json'), {'asset': identity, 'failures': failures, 'partial_bytes': partial.stat().st_size if partial.exists() else 0})
            if attempt < 3:
                time.sleep((1, 2, 4)[attempt])
    raise ConnectionError('official source unavailable after four attempts; same-ID resume or identical-asset transfer required')
