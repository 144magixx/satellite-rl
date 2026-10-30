# A1 margin evidence reader: prespecified arithmetic only

This analysis source is separate from both frozen runtime93f39b and result reader9fe816c. It has not read A1 scientific outcomes while defining its calculations. The rule is pinned to `A1_to_B_evidence_rule_v1.json` SHA `f4ffcfadb1d4ae2a5145493fd4fa86c584d81e81a89a44b3286a387744de61ce`.

The existing A0 bundle was structurally checked without reading A1 effects: all20kernels contain float64 gain/path and all60snapshots contain rates/starts/counts/powers; frozen physics source contains all required constants. The available structures are sufficient: saved prefix `rates/starts/counts/powers`, complete stable processing indices/groups and positive beam demands; saved `kernel.gain[receiver,transmitter]` and `kernel.path[receiver,slot]`; and recorded kB/temperature/slotHz/rx_gain. Actual readability/alignment is checked later against accepted artifacts. Missing fields are `unavailable`, never zero. The reader does not call an environment, capacity function, physical coefficient or kernel builder, observation constructor, Actor or optimizer.

Direct array formulas are:

- `kappa[i -> j,k] = gain[j,i] * 10**(rx_gain_dBi/10) * path[j,k]`, only for same group parity and different groups.
- `I[j,k]` sums the same saved coupling for every previously assigned transmitter whose actual slot interval contains k, times its `P/count`.
- `n = kB*T*slotHz`; desired signal is excluded from `n+I`. The desired-signal loss parameter does not enter interference.
- X and U sum only slots occupied by each prior receiver. X uses the fixed `exp(-abs(C/D-1))` weight, U uses1. Candidate exposure is `(P/count)*sum_slots(X or U)`; kappa is not divided by candidate count twice.
- SI damage and signed SGM change use only previous receivers and divide by actual full N. SGM beta is all-N mean demand. Current-beam metrics are separate.

All100 exact(l,p) groups are retained for every available state. Nonzero range is strictly `max-min>1e-12`; Spearman uses exact average tie ranks. Constant HX/HU/damage, insufficient frequencies, missing fields and no defined correlation are explicit. Paired rho differences use only the SAME groups where BOTH are defined, followed by state mean, mother defined-state mean and defined-mother mean. The equal-state paired mean is reported separately and is not substituted for equal-mother classification.

Before scientific effect access, root clarified the rule's secondary mass term: affected means the exact saved inequality `Cprime != C`, so even floating-point tiny changes count. Within each action each prior receiver is counted once, in its fixed C/D bin. Report count, sum of corresponding D and fraction of all processed D; all legal actions average to a state, then states average to a mother. With no prior demand, count and sum can be observed0, but the fraction is undefined (`no_prior_demand`), never0. Original/reward-max/registered-oracle actions have separate records. This secondary definition is not a new threshold, and never substitutes for the frozen1e-6 SI/SGM mechanistic-presence test.

S10 source audit reads SHA-pinned source text/AST only. It checks S10 width=true/relations=false and the absence of explicit prior-capacity fields in the original observation builder. It reports current incoming interference/occupancy/request/global context/width fields. It does NOT claim mathematical unrecoverability or prove observation aliasing.

```
PYTHONPATH=src python -m unittest discover -s tests -p 'test_bottleneck29_a1_margin_analysis.py' -v
```

Programmatic API:

`features(prefix, demands, indices, groups, kernel, physics)` returns the fixed100-slot X/U arrays or explicit unavailable.

`analyze_state(state_id, mother, prefix, demands, indices, groups, kernel, physics, queries, selected_points)` consumes every legal original-domain `(f,count,p_W), capacity_vector` pair and verifies domain completeness. `selected_points` has `original`, `reward_max`, `oracle`. It never uses their terminal values in X/U or group correlations.

`analyze(config_paths, rule_path, a1_summary_path, a1_states_path, source_audit_refs=..., accepted_analysis_refs=..., semantics_path=..., kernel_clarification_path=..., path_map=..., output_dir=NEW_DIRECTORY)` requires root-accepted `summary_SHA` and `states_SHA`, exact original config bytes, matching accepted journal SHAs and A0 bundle identities. `source_audit_refs` contains `environment_path`, `schema_path`, `expected_environment_SHA`, `expected_schema_SHA` for frozen S10 source. `path_map` only relocates existing files; it never edits originals. All60states/20mothers remain represented, including unavailable. Diagnostic group vectors are summarized by range/count/rho/degeneracy/SHA with immutable underlying capacities referenced; duplicate huge per-action capacity arrays are not created.

Outputs `margin_summary.json` and `margin_states.json` only to a new directory. Arithmetic/I/O time and zero scientific-call counts are separate. Result gates are read from the independently accepted A1 summary and remain unchanged. A positive conditional B-preparation rationale still requires root coverage/source/limitations review; this program does not authorize training or launch anything. Correlation is descriptive, Q remains unavailable, and no causal missing-information bottleneck or future B improvement is claimed.

The unchanged v1 rule is supplemented by two pre-effect, append-only design clarifications: semantics SHA `c13bb5a1e984808befc4fb018e0fef612d71f52c9d90d705355869bde1621d5a` and saved-kernel arithmetic SHA `69080f186d02189afa63bf5d9fbcf8505ff3c86719a5fb63f2fb00682c842dd4`. All three are mandatory SHA-checked inputs and report references. Group coverage records both-defined, X-only, U-only, neither-defined and unavailable separately by state and mother. Total affected count/mass/share and margin-bin values are both retained.

The margin driver deliberately does NOT call `A0Facts.mother`, `UniformStage2Input`, stage importers or layout/coverage validators, because their default validation can recompute geometry. `readonly_context` only decodes existing typed wire bytes, checks their hashes/parent identity/request conservation, maps the saved beam order/groups, reads accepted kernel arrays and references frozen physical constants. It does not evaluate distances, patterns, footprints or any geometry. Snapshot identity/resource arithmetic remains read-only.
