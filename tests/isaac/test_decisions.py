"""Deployment decoding of learned task decisions; no physical success fixtures."""

import pytest

from vla_lego.pretrained.decisions import decode_decision


@pytest.mark.parametrize(
    "values,expected",
    [([0, 0], "continue"), ([1, 0], "complete"), ([0, 1], "abstain"), ([1, 1], "conflict")],
)
def test_task_decision_decoding(values, expected):
    assert decode_decision(values) == expected


def test_invalid_decision_cannot_be_ignored():
    with pytest.raises(ValueError, match="finite"):
        decode_decision([float("nan"), 0])
