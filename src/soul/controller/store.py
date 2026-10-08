"""Small durable control-plane ledger; never stores prompts, activations, or KV caches."""

import json
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, model TEXT, status TEXT, created REAL,
                duration REAL DEFAULT 0, input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0, error TEXT);
            CREATE TABLE IF NOT EXISTS deployments (model TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS models (model TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS deleted_models (model TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS node_caches (id TEXT PRIMARY KEY, data TEXT);
        """)
        self.db.execute(
            "UPDATE requests SET status='failed', error='Controller restarted' WHERE status IN ('queued','running')"
        )
        self.db.commit()

    def request(self, id, model):
        self.db.execute(
            "INSERT INTO requests(id, model, status, created) VALUES(?,?,?,?)",
            (id, model, "queued", time.time()),
        )
        self.db.commit()

    def update_request(self, id, status, **fields):
        allowed = {"duration", "input_tokens", "output_tokens", "error"}
        assert set(fields) <= allowed
        values = {"status": status, **fields}
        self.db.execute(
            "UPDATE requests SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?",
            (*values.values(), id),
        )
        self.db.commit()

    def save(self, table, key, data):
        assert table in ("nodes", "deployments", "models", "deleted_models", "node_caches")
        self.db.execute(f"INSERT OR REPLACE INTO {table} VALUES(?,?)", (key, json.dumps(data)))
        self.db.commit()

    def load(self, table):
        assert table in ("nodes", "deployments", "models", "deleted_models", "node_caches")
        return [json.loads(row[0]) for row in self.db.execute(f"SELECT data FROM {table}")]

    def delete_node(self, node_id):
        self.db.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        self.db.commit()

    def forget_model(self, model):
        with self.db:
            self.db.execute("DELETE FROM models WHERE model=?", (model,))
            self.db.execute("DELETE FROM deployments WHERE model=?", (model,))

    def restore_model(self, model):
        with self.db:
            self.db.execute("DELETE FROM deleted_models WHERE model=?", (model,))

    def resume_node(self, node_id):
        with self.db:
            self.db.execute("DELETE FROM node_caches WHERE id=?", (node_id,))

    def stats(self):
        counts = {r[0]: r[1] for r in self.db.execute("SELECT status,count(*) FROM requests GROUP BY status")}
        tokens = self.db.execute("SELECT COALESCE(sum(output_tokens),0) FROM requests").fetchone()[0]
        recent = [dict(r) for r in self.db.execute("SELECT * FROM requests ORDER BY created DESC LIMIT 30")]
        return {"counts": counts, "output_tokens": tokens, "recent": recent}

    def close(self):
        self.db.close()
