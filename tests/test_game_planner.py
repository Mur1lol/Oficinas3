import chess

from game import planner as planner_module
from game.planner import MovePlanner, square_to_cell


def test_default_returns_home_after_each_move():
    planner = MovePlanner()

    for uci in ("e2e4", "e7e5"):
        plan = planner.plan(chess.Move.from_uci(uci))
        assert plan.steps[0] == (0.0, 0.0, 0)
        assert plan.steps[-1] == (0.0, 0.0, 0)
        assert planner.robot_phys == planner.home_phys()


def test_no_return_home_starts_next_move_at_previous_destination(monkeypatch):
    planner = MovePlanner(return_home=False)
    first = planner.plan(chess.Move.from_uci("e2e4"))
    e4 = square_to_cell(chess.E4)
    previous_position = (e4[0] * 2, e4[1] * 2)

    assert first.steps[0] != (0.0, 0.0, 0)
    assert first.steps[-1] == (*planner.cell_mm(e4), 0)
    assert planner.robot_phys == previous_position

    starts = []
    find_path = planner_module.find_path_physical

    def record_start(board, start, destination):
        starts.append(start)
        return find_path(board, start, destination)

    monkeypatch.setattr(planner_module, "find_path_physical", record_start)
    second = planner.plan(chess.Move.from_uci("e7e5"))

    assert starts[0] == previous_position
    assert second.steps[-1] == (*planner.cell_mm(square_to_cell(chess.E5)), 0)
    assert planner.robot_phys != planner.home_phys()
