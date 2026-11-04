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


def test_electromagnet_overshoot_and_return():
    planner = MovePlanner(return_home=False, overshoot_mm=5.0)
    plan = planner.plan(chess.Move.from_uci("e2e4"))
    e4_x, e4_y = planner.cell_mm(square_to_cell(chess.E4))

    # e2e4 arrives along -X (from 300 to 200 mm), so overshoot subtracts 5mm from X
    assert plan.steps[-2] == (e4_x - 5.0, e4_y, 1)
    # On release (ima=0), it returns to the exact square e4
    assert plan.steps[-1] == (e4_x, e4_y, 0)


def test_overshoot_disabled_when_zero():
    planner = MovePlanner(return_home=False, overshoot_mm=0.0)
    plan = planner.plan(chess.Move.from_uci("e2e4"))
    e4_x, e4_y = planner.cell_mm(square_to_cell(chess.E4))

    assert plan.steps[-2] == (e4_x, e4_y, 1)
    assert plan.steps[-1] == (e4_x, e4_y, 0)


def test_reorganize_returns_black_pieces_home():
    planner = MovePlanner()
    moves = [
        "e2e4", "e7e5",
        "g1f3", "b8c6",
        "f1c4", "g8f6",
        "f3g5", "d7d5",
        "e4d5", "f6d5",
        "g5f7", "e8f7",
        "d1f3", "f7e6",
    ]
    for m in moves:
        planner.plan(chess.Move.from_uci(m))

    plan = planner.plan_reorganize(chess.BLACK)
    assert len(plan.steps) > 0

    # Verify Black cemetery is empty
    black_cem = {k: v for k, v in planner.cem_occ.items() if k[1] in (0, 1)}
    assert len(black_cem) == 0

    # Verify rank 7 has 8 Black pawns
    for sq in range(chess.A7, chess.H7 + 1):
        p = planner.board.piece_at(sq)
        assert p is not None
        assert p.color == chess.BLACK
        assert p.piece_type == chess.PAWN

    # Verify rank 8 has all 8 Black nobles in standard positions
    expected_rank8 = [
        (chess.A8, chess.ROOK), (chess.B8, chess.KNIGHT), (chess.C8, chess.BISHOP),
        (chess.D8, chess.QUEEN), (chess.E8, chess.KING), (chess.F8, chess.BISHOP),
        (chess.G8, chess.KNIGHT), (chess.H8, chess.ROOK),
    ]
    for sq, pt in expected_rank8:
        p = planner.board.piece_at(sq)
        assert p is not None
        assert p.color == chess.BLACK
        assert p.piece_type == pt


def test_reorganize_both_colors_returns_all_pieces_home():
    planner = MovePlanner()
    # Mutual captures
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5g2", "f1g2"]
    for m in moves:
        planner.plan(chess.Move.from_uci(m))

    assert len(planner.cem_occ) > 0  # both sides have captured pieces

    plan = planner.plan_reorganize()  # color=None organizes both
    assert len(plan.steps) > 0

    # Both cemeteries must be completely empty
    assert len(planner.cem_occ) == 0

    # Board must match standard starting setup
    starting = chess.Board()
    for sq in chess.SQUARES:
        assert planner.board.piece_at(sq) == starting.piece_at(sq)


def test_reorganize_prioritizes_non_pawns_before_pawns(monkeypatch):
    planner = MovePlanner()
    moves = ["e2e4", "d7d5", "g1f3", "b8c6", "f1c4", "c8g4"]
    for m in moves:
        planner.plan(chess.Move.from_uci(m))

    carried_destinations = []
    orig_carry = planner._carry

    def record_carry(src, dst):
        carried_destinations.append((src, dst))
        return orig_carry(src, dst)

    monkeypatch.setattr(planner, "_carry", record_carry)
    planner.plan_reorganize()

    # Identify when each final home row (0 or 7 = nobles, 1 or 6 = pawns) was populated
    noble_indices = [i for i, (_, dst) in enumerate(carried_destinations) if dst[0] in (0, 7)]
    pawn_indices = [i for i, (_, dst) in enumerate(carried_destinations) if dst[0] in (1, 6)]

    assert len(noble_indices) > 0
    assert len(pawn_indices) > 0
    # Every misplaced noble is placed on row 0/7 BEFORE any misplaced pawn is placed on row 1/6
    assert max(noble_indices) < min(pawn_indices)


def test_reorganize_with_no_return_home_never_starts_with_magnet_on():
    # Reproduce the exact scenario where --no-return-home left robot_phys on the last moved piece
    planner = MovePlanner(return_home=False)
    moves = ["e2e4", "d7d5", "g1f3", "c8g4", "f1c4", "g4d7"]
    for m in moves:
        planner.plan(chess.Move.from_uci(m))

    batches = planner.plan_reorganize_batches(max_carries_per_batch=2)
    assert len(batches) >= 2

    # First batch must start at (0.0, 0.0) with magnet OFF (0)
    assert batches[0].steps[0] == (0.0, 0.0, 0)

    for b_idx, batch in enumerate(batches):
        assert len(batch.steps) > 0
        # Every batch must start and end with magnet OFF (0)
        assert batch.steps[0][2] == 0, f"Batch {b_idx} started with magnet ON: {batch.steps[0]}"
        assert batch.steps[-1][2] == 0, f"Batch {b_idx} ended with magnet ON: {batch.steps[-1]}"

        # Every pickup (0 -> 1 transition) must happen at the same (x, y) as the preceding ima=0 step
        for i in range(1, len(batch.steps)):
            prev_x, prev_y, prev_ima = batch.steps[i - 1]
            curr_x, curr_y, curr_ima = batch.steps[i]
            if prev_ima == 0 and curr_ima == 1:
                assert (prev_x, prev_y) == (curr_x, curr_y), (
                    f"Magnet turned ON while moving from ({prev_x},{prev_y}) to ({curr_x},{curr_y})"
                )


