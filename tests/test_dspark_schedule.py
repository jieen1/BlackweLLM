from runtime.dspark_schedule import (
    SpsAdditiveCostTable,
    SpsCostTable,
    choose_verify_width,
    choose_verify_widths,
    compute_verify_token_budget,
    schedule_verify_lens_topk,
    schedule_verify_widths_topk,
    survival_prefix,
)


def test_survival_prefix_is_cumulative_and_clamped():
    assert survival_prefix([0.9, 0.5, 2.0, -1.0]) == [0.9, 0.45, 0.45, 0.0]


def test_static_policy_verifies_the_full_block():
    assert choose_verify_width([0.1, 0.1, 0.1], max_width=3) == 3


def test_threshold_policy_keeps_minimum_and_stops_at_first_failed_prefix():
    assert choose_verify_width([0.95, 0.9, 0.2, 0.99], min_width=1, survival_threshold=0.5) == 2
    assert choose_verify_width([0.1, 0.99, 0.99], min_width=1, survival_threshold=0.5) == 1


def test_batch_policy_preserves_request_order():
    assert choose_verify_widths([[0.99, 0.99], [0.2, 0.99]], survival_threshold=0.5) == [2, 1]


def test_sps_budget_matches_sglang_objective():
    table = SpsCostTable(
        sample_batch_tokens=(1, 2, 3, 4, 5, 6),
        sample_steps_per_sec=(10.0, 10.0, 10.0, 1.0, 1.0, 1.0),
        max_batch_tokens=6,
    )
    decision = compute_verify_token_budget(
        [survival_prefix([0.9, 0.8, 0.7]), survival_prefix([0.8, 0.4, 0.3])],
        sps_table=table,
    )
    # B=2. One extra candidate keeps the batch at 3 tokens and maximizes
    # tau * SPS; the next candidate crosses the measured knee.
    assert decision.budget == 1
    assert decision.predicted_step_seconds == 1.0 / 10.0


def test_sps_topk_uses_stable_position_then_request_order():
    confidences = [[1.0, 1.0], [1.0, 1.0]]
    assert schedule_verify_lens_topk(confidences, budget=2) == [2, 2]
    assert schedule_verify_widths_topk(confidences, budget=2) == [1, 1]


def test_additive_sps_table_is_supported():
    table = SpsAdditiveCostTable(
        bias_seconds=1.0,
        bs_probes=(1, 2),
        alpha_seconds=(0.0, 0.0),
        m_probes=(1, 4),
        theta_seconds=(1.0, 2.0),
    )
    decision = compute_verify_token_budget([survival_prefix([0.9, 0.8])], sps_table=table)
    assert decision.budget in {0, 1, 2}
    assert decision.predicted_step_seconds is not None
