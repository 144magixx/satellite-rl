# W29 A0 operational entry

No real physics is executed by importing these modules or running the pure tests.

```
PYTHONPATH=src python -m unittest discover -s tests -p 'test_bottleneck29*.py' -v
python -m satellite_rl.bottleneck29.workbench --config /absolute/lane.json --port 9090
```

HTTP listens only on 127.0.0.1. GET `/`, `/config`, `/status`, `/logs`; POST `/start/engineering`, `/start/science`, `/stop`. Worker CLI requires the token of an existing durable workbench dispatch. Workbench and child survive client disconnection; deploy service with a detached session or service manager. All filenames are registered per lane; no recursive storage scans.

Config schema (JSON):

```
{
  "task_id": "S2-W29-A0-lane0",
  "lane": 0,
  "root": "/absolute/w29/lanes/20655",
  "input_root": "/absolute/w29/shared/transport",
  "mothers": "REPLACE WITH EXACT manifest.rows ARRAY, not this string",
  "binding": {"source_SHA":"...","data_SHA":"...","contract_SHA":"...","authority_SHA":"..."},
  "pinned_files": {"/absolute/frozen/source/member.py":"...", "/absolute/manifest.json":"...", "/absolute/A0_contract_v2.json":"..."},
  "stop_paths": ["/absolute/w29/lanes/20655/stop.json", "/absolute/w29/control/new_user_pause.json"],
  "minimum_free_bytes": 2147483648,
  "max_operation_bytes": 1048576,
  "max_incremental_bytes": 536870912,
  "receipt_max_age_seconds": 300,
  "engineering_admission": "/absolute/w29/lanes/20655/engineering_admission.json",
  "science_admission": "/absolute/w29/lanes/20655/science_admission.json"
}
```

Every `pinned_files` member is verified on each POST and worker start. Root must list the complete frozen source manifest plus input/contract/authority. No mutable config is accepted: root receipt must bind SHA256 of exact config bytes. Do not include receipt file bytes in config pins, to avoid circular hashes.

Admission receipt: `root_approved: true`, `mode: engineering|science`, `lane`, `generation`, `checked_unix`, exact `binding`, `config_SHA`, `identity` containing actual integer `uid`, `/proc/sys/kernel/random/boot_id` `boot_id`, entire stripped `nvidia-smi --query-gpu=uuid --format=csv,noheader` output `gpu_uuid`, resolved interpreter absolute path `interpreter`, and `threads` object with OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/OMP_NUM_THREADS/NUMEXPR_NUM_THREADS all exactly "1". All four must be set before Python imports numerical libraries. Root alone verifies account/resource ownership, shared storage, no conflicting task, independent source correctness, authority and engineering qualification before writing receipt. Science additionally requires `engineering_accepted: true`, `full_loop_seconds_upper_bound <= remaining_science_seconds`. Engineering writes evidence but never self-admits science.

Durable phase dispatch stores task ID, generation, config SHA, one-use token, PID, Linux process birth tick and boot ID. A completed phase rejects restart. A stopped/uncertain phase requires new root receipt `recovery_approved: true` and `recovery_of_dispatch_SHA` equal to exact old dispatch bytes, after old process exit. Pending operation intents still block execution; no implicit retry or count reset exists. Root must resolve unknown charges separately without replacing ledger. Crashed active time reserves entire remaining allowance conservatively.

Journal is fsync append JSONL with per-record SHA, in-memory index and one exclusive lane writer. It uses no SQLite/WAL or mmap. Torn or bad checksums fail closed. Completed numerical objects and every table row are SHA indexed; immutable snapshot saved once per state. Core kernel results are stored once per mother. Results preserve full N numeric capacity, demand, SI and SGM plus per-row actions, prefix and reward numbers; redundant next snapshot is removed. Counters include unknown intents. On resume completed physical arrays are reused to reconstruct pure prefix results with zero new PHY. The active timer starts at ledger construction, including journal recovery; normal exit accounts monotonic elapsed, abnormal exit does not reset the clock.

Science limits per lane: PHY18825, prefix3000/minimum14325/fine1500, kernel15, active900 seconds. Engineering limits PHY256/kernel4/900 seconds. Engineering compares charged old T0 and new implementation on original early prefix, tests exact half-watt hand arithmetic/monotonicity, restore/illegal-zero-call and 16 fixed fine/large-slot probes at each of all three selected prefixes. Mother selection uses all 15 saved-prefix workload proxies before any outcome; original replay reaches 7/8. Maximum N200 budget is 175 new replay +26 old reference +48 probes +2 half-watt =251 PHY, kernel1. Evidence includes exact counts, all row/prefix duration samples, input/kernel timings and wall clock; the proxy is not a mathematical throughput bound. Its max-row conservative extrapolation is evidence, not an automatic throughput acceptance: root must account remaining input/kernel/prefix and representative full-loop overhead before science admission. No model call or training entry exists.

Storage is hard limited to 0.5GiB per science lane (all four 2GiB), with one-object serialized bound 1MiB and atomic temporary peak reserved. The initial writer performs one bounded inventory of fixed operations/rows/snapshots plus registered root artifacts; any missing journal/sentinel, orphan, temporary, changed size or unknown intent fails closed. GET status opens the journal strictly read-only and reports a concurrent partial tail as unavailable. No periodic recursive scans occur.
