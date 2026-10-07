import math

import main


def test_hidden_opcode_bank_roundtrips_all_24_ops():
    ops = set()
    for from_bottom in (False, True):
        for second in (False, True):
            for right in (False, True):
                for action in range(3):
                    op = main.encode_hidden_opcode(
                        from_bottom=from_bottom,
                        second=second,
                        right=right,
                        action=action,
                    )
                    assert main.is_hidden_opcode(op)
                    assert main.decode_hidden_opcode(op) == (
                        from_bottom, second, right, action
                    )
                    ops.add(op)
    assert len(ops) == 24
    assert min(ops) == -151
    assert max(ops) == -128


def test_minus2_captures_pair_and_hidden_read_can_use_it_immediately():
    memory = main.HiddenMemory()
    read_left_top = main.encode_hidden_opcode(
        from_bottom=False, second=False, right=False, action=0
    )
    rule = main.Rule(pattern=[-2, 9, -2], replacement=[read_left_top])
    out, matched, overflow, memory_changed = main.replace_once_cpu(
        [1, 2, 9, 3, 4], rule, 128, memory
    )
    assert matched and not overflow and memory_changed
    assert memory.left == [[1, 2]]
    assert memory.right == [[3, 4]]
    assert out == [1, 2]


def test_minus3_appends_at_bottom_and_pop_delete_selectors_work():
    memory = main.HiddenMemory(left=[[10], [20]], right=[[11], [21]])
    # Add a bottom pair via -3 captures.
    rule = main.Rule(pattern=[-3, 9, -3], replacement=[-1])
    _, matched, overflow, changed = main.replace_once_cpu(
        [30, 9, 31], rule, 128, memory
    )
    assert matched and not overflow and changed
    assert memory.left[-1] == [30]
    assert memory.right[-1] == [31]

    # Pop second item from the bottom of the right list: with 3 items this is [21].
    op = main.encode_hidden_opcode(
        from_bottom=True, second=True, right=True, action=1
    )
    vals, changed = memory.execute(op)
    assert changed and vals == [21]
    assert memory.right == [[11], [31]]

    # Delete the top item of the left list without reading it.
    op = main.encode_hidden_opcode(
        from_bottom=False, second=False, right=False, action=2
    )
    vals, changed = memory.execute(op)
    assert changed and vals == []
    assert memory.left == [[20], [30]]


def test_hidden_rewrite_is_persistent_and_nonexpanding():
    memory = main.HiddenMemory(left=[[1, 2]], right=[[3, 4]])
    out, changed, overflow = main.hidden_rewrite_cpu([1, 2, 7, 1, 2], memory, 128)
    assert not overflow and changed
    assert out == [3, 4, 7, 3, 4]
    # Sweep use itself never consumes memory.
    assert memory.left == [[1, 2]] and memory.right == [[3, 4]]

    # A mapping that would grow the visible state is skipped to preserve the
    # evaluator's hard non-expanding scratch invariant.
    memory = main.HiddenMemory(left=[[1]], right=[[2, 3]])
    out, changed, overflow = main.hidden_rewrite_cpu([1, 1], memory, 128)
    assert out == [1, 1] and not changed and not overflow


def test_cycle_identity_includes_hidden_memory():
    # Visible text is unchanged every round, but -2 keeps changing hidden memory.
    # Old state-only cycle detection would terminate after the first repetition.
    g = main.Genome(
        rules=[main.Rule(pattern=[-2], replacement=[-1])],
        embedding=list(range(main.EMBEDDING_ENTRY_COUNT)),
    )
    _, _, stats, states = main.trajectory_features_cpu([[1, 2, 3, 4]], g, 128)
    expected_limit = int(math.ceil(2.0 * math.sqrt(4)))
    assert stats[0, 0] == expected_limit
    assert stats[0, 3] == 0
    assert states[0] == [1, 2, 3, 4]
