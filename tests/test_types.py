import numpy as np
import pytest

from sentinel.types import Action, RobotState


def test_robotstate_q_does_not_alias():
    q_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    state = RobotState(q=q_arr, qd=np.zeros(6), t_mono=0.0)
    q_arr[0] = 999.0
    assert state.q[0] != 999.0
    assert not np.shares_memory(state.q, q_arr)


def test_robotstate_qd_does_not_alias():
    qd_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    state = RobotState(q=np.zeros(6), qd=qd_arr, t_mono=0.0)
    qd_arr[0] = 999.0
    assert state.qd[0] != 999.0
    assert not np.shares_memory(state.qd, qd_arr)


def test_robotstate_wrench_does_not_alias():
    wrench_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    state = RobotState(q=np.zeros(6), qd=np.zeros(6), t_mono=0.0, wrench=wrench_arr)
    wrench_arr[0] = 999.0
    assert state.wrench[0] != 999.0
    assert not np.shares_memory(state.wrench, wrench_arr)


def test_action_q_does_not_alias():
    q_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    action = Action(q=q_arr)
    q_arr[0] = 999.0
    assert action.q[0] != 999.0
    assert not np.shares_memory(action.q, q_arr)
