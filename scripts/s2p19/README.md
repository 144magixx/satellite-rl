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

After both services are running from one clean commit, run `python -m
scripts.s2p19.plan_builder bootstrap`. It reads the two reviewed A-stage API
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
and the shared `/home` free space on every poll. A 12 GiB reserve and a
conservative 10 MiB per complete B prediction block unsafe dispatch; running
jobs reserve future growth so two large jobs cannot both be admitted against
the same initial free-space snapshot. Below reserve, it requests workbench
stop for active **S2-P19** jobs and exits for diagnosis. It never stops old
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
