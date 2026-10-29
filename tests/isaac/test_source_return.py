"""Empty-jaw returns clear loose pieces without lowering tall-stack retreat routes."""

import pytest

from vla_lego.isaac.assembly_planning import pick_route
from vla_lego.isaac.task import initial_tower_task


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("level", [0, 4])
def test_empty_return_stays_at_source_transfer_height_or_above_tall_stack(side, level):
    task = initial_tower_task()
    route = pick_route(task, level, [0.45, -0.3, 0.83], [0.4, 0], side, 3)
    retreat, lateral = route[-2:]
    assert lateral[0] == 0.4
    if level == 4:
        assert retreat == lateral
    else:
        assert retreat[:2] == [0.4, 0]
    assert lateral[1] > 0 if side == "left" else lateral[1] < 0
    assert retreat[2] == lateral[2] and retreat[2] >= route[1][2]
    assert retreat[2] > route[-3][2]
    if level == 4:
        assert retreat[2] > route[1][2]
