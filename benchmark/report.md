# Benchmark Report

- Generated at: `2026-06-09T03:04:55`
- Total turns: `65`

## Overall

| dataset | turns | pass_rate | route_ok | hit@5 | recall@5 | diverse_met@5 | forbidden_clean@5 | profile_used_ok | context_reuse_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 65 | 0.985 | 1.000 | 1.000 | 0.872 | 1.000 | 0.982 | 1.000 | 1.000 |

## By Dataset

| dataset | turns | pass_rate | route_ok | hit@5 | recall@5 | diverse_met@5 | forbidden_clean@5 | profile_used_ok | context_reuse_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_dialogues | 5 | 1.000 | 1.000 | - | - | - | - | - | 1.000 |
| cross_scenario | 10 | 1.000 | 1.000 | 1.000 | 0.540 | 1.000 | 1.000 | - | - |
| out_of_catalog | 5 | 1.000 | 1.000 | - | - | - | 1.000 | - | - |
| personalized_coarse | 10 | 1.000 | 1.000 | 1.000 | 0.818 | 1.000 | 1.000 | 1.000 | - |
| retrieval_core | 30 | 0.967 | 1.000 | 1.000 | 1.000 | - | 0.967 | - | - |
| route_boundary | 5 | 1.000 | 1.000 | - | - | - | - | - | - |

## Failures

- `retrieval_core/retrieval_010#t1` expected `recommend`, got `recommend`, products `p_digital_006, p_digital_012, p_digital_023, p_digital_020`
