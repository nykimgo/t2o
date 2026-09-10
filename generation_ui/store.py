"""Durable review records. Source assets are snapshotted, never overwritten."""
import json
import sqlite3
import time
import uuid
from pathlib import Path


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'studio.sqlite3'
        with self.connect() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.executescript('''
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, parent TEXT,
                    created REAL NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS records_kind_created ON records(kind, created);
                CREATE INDEX IF NOT EXISTS records_parent ON records(parent);
                CREATE TABLE IF NOT EXISTS imports (fingerprint TEXT PRIMARY KEY, version TEXT NOT NULL);
            ''')

    def connect(self):
        c = sqlite3.connect(self.db, timeout=20)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def unpack(row):
        return {**json.loads(row['data']), 'id': row['id'], 'kind': row['kind'],
                'parent': row['parent'], 'created': row['created']}

    def create(self, kind, data, parent=None):
        ident = uuid.uuid4().hex
        with self.connect() as c:
            c.execute('INSERT INTO records VALUES (?,?,?,?,?)',
                      (ident, kind, parent, time.time(), json.dumps(data, ensure_ascii=False)))
        return self.get(ident)

    def get(self, ident, kind=None):
        with self.connect() as c:
            row = c.execute('SELECT * FROM records WHERE id=?', (ident,)).fetchone()
        if row is None or (kind and row['kind'] != kind):
            raise KeyError(ident)
        return self.unpack(row)

    def list(self, kind, parent=None):
        with self.connect() as c:
            rows = c.execute('SELECT * FROM records WHERE kind=?' +
                             (' AND parent=?' if parent else '') + ' ORDER BY created DESC',
                             (kind, parent) if parent else (kind,)).fetchall()
        return [self.unpack(r) for r in rows]

    def update(self, ident, **changes):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT * FROM records WHERE id=?', (ident,)).fetchone()
            if row is None:
                raise KeyError(ident)
            data = json.loads(row['data'])
            data.update(changes)
            c.execute('UPDATE records SET data=? WHERE id=?',
                      (json.dumps(data, ensure_ascii=False), ident))
        return self.get(ident)

    def folder(self, ident):
        self.get(ident)
        p = self.root / 'files' / ident
        p.mkdir(parents=True, exist_ok=True)
        return p

    def recover(self):
        for job in self.list('job'):
            if job['state'] in ('running', 'cancelling'):
                self.update(job['id'], state='interrupted', ended=time.time(),
                            message='서버 재시작으로 중단되었습니다. 완료된 단계를 보존했습니다.')
                self.update(job['version'], state='interrupted')

        for batch in self.list('batch'):
            if batch['state'] in ('running','cancelling'):
                progress=batch.get('progress') or {}
                progress.update(state='interrupted',ended=time.time())
                for phase in progress.get('phases',[]):
                    if phase['state']=='running':
                        phase.update(state='interrupted',ended=progress.get('updated',time.time()),
                                     seconds=max(0,progress.get('updated',time.time())-phase['started']))
                self.update(batch['id'],state='interrupted',ended=time.time(),progress=progress)
