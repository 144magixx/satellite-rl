# S2-P19 remote operations

Invoke these entry points from the repository root with `python -m
scripts.s2p19.<module>` so their imports cannot collide with older controllers.
`remote_probe`, `data_inventory`, and `deploy plan` are read-only.
`deploy deploy --commit <40-char SHA>` requires a clean frozen local branch and
clones that exact commit into isolated `sac_routes_s2p19/{20418,20424}/repo`.
`deploy serve --commit <SHA>` starts 8900/8901 only after matching checkout,
DQN/CUDA, and imported package path checks. Neither step submits a job.

Scientific tasks use the instance-local workbench API exclusively. The
preparation route is `POST /api/s2p19/prepare` with a `preparation` object from
`s2p19_data.validate_preparation_config`. Type `mother_sources` consumes the
two pinned Feb01/02 ADSB.lol asset identities and emits a 40-mother CSV record.
Type `dev_validation` consumes the frozen P17 Jan12 validation record and emits
72 development plus 72 validation scenes. Type `final` consumes the completed
40-mother record and emits 880 test scenes. The workbench freezes all small
source CSVs into `job/input_sources` and the DQN worker executes the long work.
Its output record must be imported with `POST /api/scenes/import` in each
instance that will use it. Existing P17 8064 train scenes likewise need a
strict read-only import into each instance; their scientific arrays stay frozen.

For a fresh experiment only, after both services are running from one clean
commit, `python -m scripts.s2p19.plan_builder bootstrap` reads the two
reviewed A-stage API
payloads, fetches the deployed typed catalog, and writes twenty initial tasks:
preparations, two-instance imports, teacher, I0 non-learning gate, C0/C1/I0
pilots, and SHA-bound pilot approvals. Train scene IDs, the final mother record,
teacher manifest/path/SHA and later run IDs are references to **completed API
task results**, never handwritten. The controller automatically appends a
second plan revision when a source-backed retained-arms audit appears at
`controller/gates/retained_arms.json`, both final scene imports are complete,
and each retained arm has a completed, approved full training run. That
revision adds retained model evaluations, the exact frozen S2-P18 V2 anchor,
one shared greedy/rule/random test-baseline task, and same-route final B for
each source. With all C0/C1/I0 arms retained, the final matrix uses exactly
14,960 complete B calls (three arms 7,920, V2 2,640, baselines 4,400).
It remains nonterminal until the formal report and archive receipt are added.

The old frozen 8d controller must **not** append the final 880-scene matrix:
it places C1 timing on 20424 while the other methods run on 20418. Keep the
current training and `pilot_jobs` workers on their frozen source. Before
writing `retained_arms.json`, hand off only the controller to a newly frozen
version containing the common-timing fix, with its plan/state SHA and process
identity checked. The final append requires a fresh, actually observed
`controller/gates/final_timing_isolation_receipt.json`: all retained training
is complete, 20418 serves with one worker and four thread limits at 1, and
neither instance has another active job or external GPU compute process.
Final direct and B jobs then form one serial 20418 chain. A retained C1 final
policy is imported from its SHA-pinned 20424 final pair through the 20418
workbench before timing starts. This adds no training, action repeat or B call.
After the verified single-worker service handoff and source-backed retention
audit, capture this receipt with
`python -m scripts.s2p19.plan_builder capture-final-timing --retained-arms C0,C1,I0`
(replace the list with the actual retained arms, or pass an explicit empty
value if none). Capture reads the two instances' process/thread/GPU/job state
and writes only the local receipt when both are idle. The later append
rechecks the same live identities and requires a receipt no older than ten
minutes. Do not capture it during training or copy a synthetic test fixture.
The new controller rechecks the pinned server/GPU identity and idle jobs/GPU
processes before each timed workbench submission; a changed lane stops dispatch.
External CPU/GPU work starting after a timed worker begins still requires
runtime monitoring and an explicit contention flag in the final time report.

**Current S2-P19 live handoff uses `plan_builder adopt`, never `bootstrap`.**
`adopt-preview` checks the actual run_manifest identities, two workbench run
catalogs, config-file SHA, and scene imports without writing plan/state or
sending POST. `adopt` publishes the exact existing C0/C1/I0/development B,
teacher and R1 mother-source receipts as `adopted_only` tasks with no dispatch
body; the failed 6e8 mother task remains history. Science jobs stay on 47da,
while the sole R1 mother-source preparation may carry 62202. Future final-880
preparation/import, the exact V2 historical model import, and I0 training are
separate typed nodes. I0 training uses request ID `s2p19-i0-train-51001` and
requires a complete, paired raw B scientific audit. The gate reads
`controller/gates/i0_nonlearning_receipt.json`, verifies its audit SHA,
criteria, true per-candidate thresholds, exact four workbench run IDs, and
every listed raw source SHA in bounded batches. A failed audit marks
`gate_failed`/`not_selected` for I0 only; C0/C1 continue. Do not run the
persistent controller until the parent task has reviewed the adopted plan.

After the two services are switched under a separately verified handoff,
`plan_builder harden-live --service-commit <40-char SHA>` records that server
version separately from the fixed 47da science origin. Existing C0/C1 jobs
retain their own 47da `expected_commit`; R1 remains a prepare-only 62202
exception. It also adopts the already submitted final-880 preparation under
its exact request/config SHA and adds a second independent I0 logging-runtime
receipt hold. An I0 first-train POST is impossible until both the raw
development science gate and a source-backed logging equivalence proof pass,
the runtime receipt names the exact request ID, and the active service commit
matches that runtime. Pilot audit receipts are read and their raw sources
rehash checked before any approve API call. C0 continuation waits for **both**
C0 and C1 audits; if either fails, neither arm is silently resumed. A failed
I0 gate marks only I0 `not_selected`, leaving C0/C1 work independent.

`controller --run` requires a frozen
`D:/paper/outputs/experiments/sac_routes_s2p19_20261007/controller/plan.json`.
The plan is an ordered DAG with schema 1, experiment_id S2-P19, exact
scientific_commit, and typed tasks. Task fields: unique `id`, `phase`, SSH
`port`, `kind`, prior `requires` IDs, `body`, `disk_class`, and a unique
`request_id` for jobs. Allowed kinds are `prepare`, `constructive_train`,
`improvement`, `refine`, `import_scene`, `approve_pilot`, and `milestone`.
Imports additionally require `source_record_id` and `expected_count`, enabling
recovery from a lost submit response. A task body may refer to an earlier
completed receipt with `{"$task":{"id":"dev-prep","path":["result","summary",
"outputs","development","path"]}}`; this resolves only from the controller's
verified workbench result. Pilot approval requires a stopped run with
`s2p19_pilot_gate_pending` plus the SHA-bound scientific audit path accepted by
`/api/s2p19/runs/<id>/approve-pilot`. Milestones require a real local receipt
path and SHA; report/archive completion cannot be inferred from training end.

The controller checks both remote Git heads, all request IDs, prerequisites,
and the **archive mount** free space on every poll. `/home` itself is a separate
container overlay; the two archive roots share one NFS mount. Capacity and
running-job reservations are deduplicated by that mount identity. Since NFS
statvfs reports server-wide free space without a verified user quota, a
256 GiB S2-P19 engineering envelope is a **warning to schedule staged
archives**, not a capacity or user-quota claim. A 12 GiB actual-filesystem
reserve remains the stop threshold; 10 MiB per complete B conservatively
forecasts growth. Only crossing the actual reserve requests workbench stop
for active **S2-P19** jobs and exits for diagnosis. It never stops old
experiments, deletes files, silently resumes failures, or approves a pilot
without source-backed audit. `--once` is read-only; `--run` writes controller
receipts and submits one dependent task per poll. Do not start it until the
plan, frozen commit, datasets and bounded gate criteria are complete.

P18's completed SCA cases were inspected read-only by `p18_existing_read.py`;
`p18_final_compare.py` computes corrected G1/G2 gates under the same final B
configuration and scene SHA. This did not create any experiment job.

Disk archive plan for the full proposed 15,680 B evaluations: P18's 1,320-B
jobs each occupy about 9 GB raw; the full S2-P19 budget could exceed 100 GB.
Run final methods in bounded groups under the disk guard. After each completed
group, package exact new-group paths, verify file/member SHA locally and on the
remote, ensure all downstream consumers have read their sources, then have the
main task review any precision cleanup of redundant new-group raw copies.
Compression alone never authorizes deletion. Old P17/P18/G11 assets are outside
the cleanup target. Training weights/checkpoints and source contracts remain
addressable through preserved immutable archives and the workbench ledger.

### Final 880-scene timing lane

`python -m scripts.s2p19.prepare_final_timing --check` is read-only (and is
also the default). It checks the **actual** C0/C1 400000-step endpoints, I0's
400000-step or 12500-episode endpoint, and each unique endpoint B workbench
request with 216 completed method records bound to the same run, checkpoint,
and validation step. A queued, failed, mismatched, or absent job yields a wait
or hard block; controller receipts alone cannot release the switch. The current
final C0/C1 checkpoint SHA is read from each original workbench job's
`checkpoint.pt` and checked against its 400000-step provenance: their status
records do not publish that SHA. I0's status SHA is checked against its file.
The completed pilot B source schemas used for this gate are preserved in
`preflight/final_timing_B_schema_audit.json`. The current
read-only check is saved separately under
`outputs/experiments/sac_routes_s2p19_20261007/preflight/final_timing_readonly_check.json`.

Once the approved jobs and both GPUs are genuinely idle, explicit
`python -m scripts.s2p19.prepare_final_timing --execute` stops only the exact
local ee controller and replaces only the verified 20418:8900 service. The
replacement uses the same clean 13ea checkout and DQN/CUDA/solver packages,
one worker, and four thread environment settings at 1; 20424 is untouched.
It backs up the original service deployment receipt, pins the new PID and
birth time for `controller._service_identity`, and invokes the existing
`capture_final_timing_isolation(["C0", "C1", "I0"])` check. The switch
does not submit the final matrix. An uncertain launch is inspected by exact
service identity before any retry; an already saved same-target switch is a
no-op. The final receipt is
`outputs/experiments/sac_routes_s2p19_20261007/preflight/final_timing_switch/switch_receipt.json`.

### Four-instance core delivery archive

`python -m scripts.s2p19.archive_core --list` reads the frozen local four-lane
plan, its state, the historical plan and run manifest without contacting a
workbench. It lists the exact 12 final task IDs plus the C0/C1/I0 training
run IDs, instance ownership, original unique request IDs, frozen commits and
expected method/interaction counts. The five historical final runs come from
`completed_legacy`; seven four-lane runs come from their current state. A
changed plan SHA or duplicated run/request is rejected.

`python -m scripts.s2p19.archive_core` is a read-only live check: it GETs each
of those 15 workbench runs and reports incomplete or mismatched jobs. It does
not use local controller completion as proof. Only after all 15 live jobs are
completed with the exact run, request, source commit, archive owner, and
budget may an operator run `python -m scripts.s2p19.archive_core --execute`.
The optional `--part-mib` (maximum 8192 MiB), `--max-gib` (maximum 256 GiB),
`--compression-level` (1–9) and `--remote-timeout-seconds` (60–43200) retain
explicit archive.py bounds. The defaults remain 256 MiB, 20 GiB, level 1 and
1200 seconds for smaller jobs. Large results must use values chosen from the
actual inventory and available remote/local disk, rather than assuming a
compression ratio. A volume count over 200 or insufficient local compressed
bytes plus the 8 GiB reserve blocks copying. Changing compression level gives
a distinct immutable package identity; old level-1 packages remain readable.
The archive keeps every original job/source/scene/weight member in verified
ZIP volumes. Only small job/config/status/source-manifest identity files are
unpacked locally. The source index points at SHA-pinned ZIP members for large
results, cases, scenes and training JSONL; the report reads those streams and
checks member SHA without creating plain duplicates. Derived per-beam output
is gzip CSV. The original remote workbench archive remains untouched.
That mode repeats all GET checks before any copy, then calls the existing
completed-only, per-source and per-ZIP-member SHA archive one task at a time.
The shared `analysis/source_index.json` merge has an exclusive file lock and
rejects a task already bound to another run, port or inventory. Earlier smoke
keys are preserved; an older key without a port is enriched only after its
local completed receipt, verified download receipt and ZIP volume SHA match
the same run/port/inventory. Each live GET has its own observation timestamp.
Each verified copy is committed to
`remote_archive/core_archive_ledger.json`; a failed transfer records an error
and stops. A retry rechecks all live jobs and the existing receipts. An
unknown partial remote package is never deleted or overwritten automatically.

This core ledger covers only the 12 final tasks and three training runs.
Preparation, development, pilot and endpoint-B jobs from the historical
controller plan may be copied later in completed-only batches under their
real task/run/port identities; they do not become complete by appearing in
this checklist. Failed preparation, ops, controller and monitoring evidence
remains independent in `analysis/supplemental_archive_plan.json`. Neither
supplemental archive nor formal report milestone is marked complete here.

`python -m scripts.s2p19.archive_waiter --check` (also the default) and
`--list` read only the local four-lane state. Only explicit `--run` starts a
locked persistent waiter: it requires all seven new local task states to be
completed, then rechecks all 15 real workbench runs, records per-job metadata
member names/bytes and remote statvfs free space (not a quota), and checks
256 GiB input, 40,000 files, 200 volumes at 1024 MiB, remote 12 GiB and local
8 GiB reserves. It then calls the serial core archive with compression level
6 and a 43,200-second per-job remote timeout. The original
`2026-10-09T17:51:53Z` deadline stops waiting/gating; once correctly started,
archive time is bounded separately to 72 hours, checked before each job.
Status/phase times, PID, source commit, plan SHA and each inventory are saved
under `remote_archive/`. Limited network retries retain incomplete packages
for review; a capacity or identity failure enters `attention`. The actual ZIP
byte guard in archive.py remains authoritative because compression ratio is
unknown. This waiter still does not copy supplemental evidence or finish the
formal report.
