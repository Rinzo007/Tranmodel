from tndp.optimizer_acceleration import MemoizedEvaluator, cache_stats, screen_then_exact, unique_by_key


def test_unique_by_key_removes_duplicate_networks():
    values = ["a", "b", "a", "c", "b"]
    assert unique_by_key(values, lambda x: x) == ["a", "b", "c"]


def test_memoized_evaluator_reuses_result():
    calls = []
    ev = MemoizedEvaluator(lambda x: calls.append(x) or x * 2, lambda x: x)
    assert ev(3) == 6
    assert ev(3) == 6
    assert calls == [3]
    assert cache_stats(ev) == {"hits": 1, "misses": 1, "size": 1}


def test_screen_then_exact_only_evaluates_top_k():
    exact_calls = []
    result = screen_then_exact(
        [1, 2, 3, 4, 5],
        key_fn=lambda x: x,
        fast_evaluator=lambda x: x,
        exact_evaluator=lambda x: exact_calls.append(x) or x * 10,
        top_k=2,
    )
    assert [x for x, _ in result] == [1, 2]
    assert exact_calls == [1, 2]
