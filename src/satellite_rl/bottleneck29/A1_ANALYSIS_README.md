# A1 read-only result verification

`a1_analysis.py` is an analysis-only source, frozen separately from the scientific runtime. It imports no Actor loading path and calls no PHY, coefficient build, observation construction, model forward or parameter update. Numerical metrics are checked against already saved capacity arrays.

```
PYTHONPATH=src python -m unittest discover -s tests -p 'test_bottleneck29_a1_analysis.py' -v
python -m satellite_rl.bottleneck29.a1_analysis --configs lane0.original.json lane1.original.json lane2.original.json lane3.original.json --contract A1_contract_v1.json --expected-binding expected_binding.json --path-map local_paths.json --output NEW_ANALYSIS_DIRECTORY
```

Keep the four config files byte-identical to the admitted originals. `expected_binding.json` contains the exact root-accepted binding dict. For copied local results, `local_paths.json` maps original remote root prefixes to their canonical local copy, for example `{"/home/account/project/A1":"D:/paper/outputs/.../collected/A1"}`. Mapping affects access paths only; config SHA and stored A0 alias source references remain original. The reader verifies every pinned source/input/Actor file at its mapped location and requires completed root-admitted dispatch records. Do not edit original configs to make local paths convenient.

The reader rejects missing/incomplete lanes, unresolved intents or sessions, partial/unsealed/unindexed/orphan chunks, bad row/chunk hashes, mismatched source/config/budget/provenance, illegal or incomplete terminal allocations and incorrect saved metrics. It streams one state's chunks into memory at a time and verifies all573000 structural enumeration rows as valid/A0alias/explicit-invalid, with no illegal score substituted by zero. A0 aliases resolve through the accepted bundle and retain their original source/snapshot/record SHA. Committed data is never rewritten.

Initial Actor laws are saved in full: all six nominal slot identities, the expected missing-Q slot, exact reward maximum/ties, mode, all four Actor plus four uniform proposals, deduplication, probabilities and original deterministic RNG keys are reconstructed with pure arithmetic. Each unique candidate must have two complete continuations. Every saved step verifies current state/action/source, full N capacity values, reward components, shared uniform key/value, probability/logpi relationship, legal resources and the final all-N terminal. Ordinary and Q-semantic sampled returns are independently summed, excluding current-candidate entropy.

Suffix full Actor CDFs were deliberately not saved by the frozen runtime. Without new forward calls the reader cannot independently prove that the recorded uniform falls inside the selected action's Actor CDF interval. It verifies recorded key/value, chosen action legality, probability/logpi consistency and arithmetic consequences, and relies on frozen engineering for the actual Actor law implementation. It does not claim complete independent Actor rollout replay.

Aggregation is repeat mean, then three-prefix mother mean, then equal mean across20mothers. Two separate results are reported: reward-max causal single intervention versus original; and hindsight maximum terminal-SGM candidate, breaking ties by that same candidate's SI then earliest slot. SI/SGM/power are never independently cherry-picked. The metric passing its grand-mean effect threshold must itself improve in at least12mothers. The other metric's degradation guards,110%power limit and all hard constraints must also hold.

Outputs are `summary.json`, `states.json` and `terminals.json` in a new directory; existing outputs are never overwritten. Any verification exception prevents a complete/gate-pass artifact. The reader does not dispatch a task or authorize B/training. Reused development states and two-suffix sampling limitations remain in the summary. Oracle-only improvement describes mechanism space, not a deployed policy improvement.
