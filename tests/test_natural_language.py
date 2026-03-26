from retail_analyzer.retail_anomaly_detection import TYPE_MESSAGES


def test_type_messages_nonempty():
    assert len(TYPE_MESSAGES) >= 4
    for msg in TYPE_MESSAGES.values():
        assert len(msg) > 10
