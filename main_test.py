import pytest
from main import Env


def test_env_win_lines():

    win_lines_size_3 = [
        ((0, 0), (0, 1), (0, 2)),
        ((1, 0), (1, 1), (1, 2)),
        ((2, 0), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 1), (1, 1), (2, 1)),
        ((0, 2), (1, 2), (2, 2)),
        ((0, 0), (1, 1), (2, 2)),
        ((0, 2), (1, 1), (2, 0)),
    ]

    env = Env(3)
    result_win_lines = env.win_lines

    assert(len(win_lines_size_3) == len(result_win_lines))

    for win_line in win_lines_size_3:
        expected = sorted(win_line)
        for _win_line in result_win_lines:
            result = sorted(_win_line)
            if expected == result:
                result_win_lines.remove(_win_line)

    assert (0 == len(result_win_lines))
