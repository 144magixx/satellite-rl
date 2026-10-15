# A1 append-store and workbench API

This is a new isolated A1 runtime. It never modifies the A0 runtime or artifacts.

```
PYTHONPATH=src python -m unittest discover -s tests -p 'test_bottleneck29_a1_stream.py' -v
python -m satellite_rl.bottleneck29.a1_workbench --config /absolute/A1/lane0.json --port 9100
```

Before Python starts, set OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, OMP_NUM_THREADS and NUMEXPR_NUM_THREADS to exactly `1`. Identity receipts include all four actual values and exact UID, boot ID, interpreter and complete GPU UUID output, using the A0 `identity()` schema.

The workbench listens on loopback. GET `/`, `/config`, `/status`, `/logs`; POST `/start/engineering`, `/start/science`, `/stop`. Service and worker ownership use atomic NFS mkdir claims, with PID/birth/boot receipts; stale claims are never automatically deleted. Only the registered root controls audited recovery after verifying old owner exit. Same completed phase cannot be submitted again. Direct worker invocation requires the one-use token of a durable workbench dispatch.

Config fields:

- `task_id`, integer `lane` in 0..3, new `root` (separate from A0), `stop_paths`, `minimum_free_bytes`, `receipt_max_age_seconds`.
- `binding`: exact `source_SHA`, `data_SHA`, `contract_SHA`, `authority_SHA`, `A0_source_SHA`, `A0_summary_SHA`, `actor_SHA`, `Q_status="unavailable_same_node_150000"`.
- `pinned_files`: complete frozen source/input/contract/Actor file path-to-SHA mapping. Receipts bind exact complete config bytes via `config_SHA`; paths may not be edited after freezing.
- `science_limits`: `phy=158925`, `enumeration=128925`, `suffix=30000`, `forward=30015`, `kernel=0`, `prefix=0`, `seconds=3600`; any coefficient-build category must also receive a finite root allocation.
- `engineering_limits`: lane0 `phy=114`, other lanes `phy=113`, each `forward=16`, `kernel=3`, `seconds<=894.7310753585`; these sum to the remaining shared453 PHY/64forward/12kernel/3578.924301s, not a new1024 budget.
- `storage_allocation`: exact root measured `total_working_cap_bytes<=2147483648`, `existing_working_bytes` (all existing A0/A1 cache/report/source working bytes, not archives counted twice), `other_reserved_bytes` (future shared Actor/config/source/ops), four `lane_science_working_bytes`, and four `lane_engineering_working_bytes`. Existing + other + all eight lane caps must fit the same2GiB working allowance. `science_storage_bytes` and `engineering_storage_bytes` must equal this lane's allocated entries; receipt must bind the exact whole allocation. Do not grant A1 a fresh4x512MiB allowance. A-package7GiB peak separately covers required archives/transport/recovery margin.
- `science_admission`, `engineering_admission`: absolute root receipt paths.
- Scientific executor-specific state/input/Actor references are frozen in this same config. Runtime does not generate missing snapshots, input scenes or Critic weights.

Receipt fields: `root_approved=true`, `phase="A1"`, `mode`, `lane`, `generation`, `checked_unix`, exact `binding`, `config_SHA`, `identity`, exact phase `limits`. Science additionally requires `engineering_accepted=true` and conservative `full_loop_seconds_upper_bound<=remaining_science_seconds`. Root must verify shared disk/owner/conflicting tasks/storage/source correctness and full-loop engineering. No engineering evidence automatically authorizes science. Existing failed dispatch needs `recovery_approved=true` and exact `recovery_of_dispatch_SHA`; unledgered preparation is a blocking accounting gap until root records its debit. Never remove a claim or reset a journal to bypass failure.

`StreamStore(root,binding,limits, storage_limit=..., stop_paths=...,minimum_free_bytes=...,active_started=...)` exposes:

- `begin()`: one bounded registered-chunk inventory and SHA verification; no repeated directory scans during work. Start time includes child admission/import/input/recovery preparation.
- `transact(op_id, charges, operation, state=state_id)`: fsync intent and all charge reservations; callback computes once; append one lossless numerical record, fsync; append indexed completion, fsync. Repeated completed IDs return verified saved values. Repeated unknown IDs stop; no free replay.
- `get(op_id)`: indexed readback. `iter_values()`: sequential chunk streaming without per-query files.
- `alias(op_id,{source_SHA,snapshot_SHA,row_id,record_SHA,...},state=state_id)`: zero new PHY, exact immutable original-source provenance. Alias resolution is the executor/reader's responsibility; never relabel A0 numerical values as a new source.
- `close_state(state_id,expected_ids=[...])`: verifies all registered rows, stores completed bitmap and ordered-ID hash, seals chunks, prevents any new row for that state.
- `record_metadata(key,value)`: small finite JSON cursor/evidence only. Numerical probability/logpi may be negative infinity: keep it in the typed numerical record, or supply an explicit tagged JSON value; never epsilon-floor or emit invalid JSON.
- `finish()`: cumulative active seconds. An interrupted unfinished session conservatively reserves its remaining time; normal pause exits at query boundaries and records elapsed. Unknown calls remain charged after pause.

Each binary record uses 8-byte length plus32-byte SHA256 and zlib level1 pickle of exact numerical bytes. Every chunk has at most256 rows. The journal contains row ID/cost/offset/hash, not duplicate capacity arrays. Chunk sealing stores whole-chunk SHA and atomically renames `.open` to `.sealed`; crash between seal and rename fails closed for root repair. Partial rows/chunks/journal are preserved, never truncated automatically. A0 static demands/layout/prefix and kernel are SHA references, never copied per query. Never serialize coefficient matrices or full mutable snapshots per row.

One fixed prefix/FL coefficient block may remain in memory while scanning its legal powers, but each capacity evaluation reserves one PHY. Initial Actor law can be one forward transaction per state. Each suffix step should combine observation+one Actor forward+one capacity into one transaction and save selected probability/logpi/actual action/reward/full capacity once; do not write a whole probability matrix and the capacity result as duplicate records every step. Numerical payloads preserve the original capacity/action/reward components; future rows share static request/prefix refs. Terminal records preserve all N final numeric service and allocation values.

All query storage uses O(1) byte accounting, a2MiB compressed single-record bound, SHA+offset indexes and immutable closed chunks. Admission must independently prove an uncompressed actual-N upper bound including indexes and all A0/A1 working/archives/temp/recovery margin; the code does not assume compression will save an oversized allocation. Status tails only appended journal bytes after the initial read and never scans result directories.

Latest engineering input adapter consumes mother ROW DICTS: `facts.mother(config["mothers"][index])` and `facts.state(ctx,row,prefix)`. Engineering allows at most80 PHY/16forward/0kernel per lane, with old reference calls charged; source and Actor loading remain disabled until workbench admission. Its full-loop numerical mock replaces every physical and model operation.
