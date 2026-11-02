"""Single-owner, durable records. Unknown inflight records cannot be replayed."""
import hashlib
import json
import os
from pathlib import Path

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf8')

def content_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()

def identity(ref):
    """Scientific identity deliberately excludes a transport locator."""
    if set(('sha256', 'bytes')) - ref.keys():
        raise ValueError('missing content identity')
    if len(ref['sha256']) != 64 or type(ref['bytes']) is not int or ref['bytes'] < 0:
        raise ValueError('invalid content identity')
    int(ref['sha256'], 16)
    return ref['sha256'], ref['bytes']

class Journal:
    def __init__(self, root, limit=134217728):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.lock = self.root / 'owner.lock'
        self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(self.fd, str(os.getpid()).encode())
        os.fsync(self.fd)

    def _path(self, key):
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in key):
            raise ValueError('record key')
        return self.root / (key + '.json')

    def read(self, key):
        return json.loads(self._path(key).read_text(encoding='utf8'))

    def has(self, key):
        return self._path(key).exists()

    def write(self, key, value):
        path = self._path(key)
        raw = encoded(value)
        used = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        if len(raw) > 67108864 or used + len(raw) > self.limit:
            raise RuntimeError('storage budget; preserve existing records')
        temp = path.with_suffix('.pending')
        with temp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        if os.name != 'nt':
            fd = os.open(self.root, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)

    def callback(self, event, payload):
        node = payload.get('node', payload)
        semantic = node.get('semantic_SHA', payload.get('semantic_SHA'))
        key = 'node_' + semantic
        if event == 'entered' and self.has(key):
            raise RuntimeError('existing node: no automatic inflight replay')
        if event == 'alias':
            if not self.has(key) or self.read(key)['event'] != 'completed':
                raise RuntimeError('alias requires durable complete source')
            self.write('path_' + payload['path'], {'event': event, 'payload': payload})
            return
        self.write(key, {'event': event, 'payload': payload})
        # The callback returns only after the whole raw node is durable.

    def close(self):
        os.close(self.fd)
        self.lock.unlink()
