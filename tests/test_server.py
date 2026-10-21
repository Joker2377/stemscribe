import server


def test_queue_position_and_cancel_queued():
    server.jobs.clear()
    server.jobs.update({
        "run": {"status": "running", "stage": "prep", "log": []},
        "q1": {"status": "queued", "stage": "", "log": []},
        "q2": {"status": "queued", "stage": "", "log": []},
        "bad": {"status": "failed", "stage": "", "log": []},
    })
    assert [server.queue_position(s) for s in ("run", "q1", "q2", "bad")] == [None, 1, 2, None]
    assert server.cancel_job("q1") == "removed"
    assert server.queue_position("q2") == 1
    assert server.cancel_job("bad") == "not_queued" and server.cancel_job("nope") == "not_queued"
    server.jobs.clear()
