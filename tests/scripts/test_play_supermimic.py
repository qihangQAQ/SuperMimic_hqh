"""Live playback must preserve terminal states and reset policy history."""

from types import SimpleNamespace

import pytest
import torch
from tools import play_supermimic as play


class FakeEnv:
    num_envs = 1

    def __init__(self, *, failed: bool):
        self.failed = failed
        self.supermimic_mdp = SimpleNamespace(
            motion=SimpleNamespace(
                names=["first", "selected"],
                clip_start=torch.tensor([0, 3]),
                clip_len=torch.tensor([3, 3]),
                root_p=torch.arange(6).float().unsqueeze(-1).repeat(1, 3),
                root_R=torch.eye(3).repeat(6, 1, 1),
                q=torch.zeros(6, 29),
            ),
            obs=lambda state, action, context, frames: frames.float().unsqueeze(-1),
            goal=lambda frames, count: frames.float().unsqueeze(-1).repeat(1, count),
        )

    @staticmethod
    def current(frame):
        state = torch.zeros(1, 71)
        state[0, 0] = frame
        state[0, 3] = 1
        return {
            "s": state,
            "f": torch.tensor([frame]),
            "a_prev": torch.zeros(1, 29),
            "xi": torch.zeros(1, 33),
        }

    def reset_to_frames(self, frames):
        self.frame = int(frames[0])
        return {}, {play.SUPERMIMIC_STATE_KEY: self.current(self.frame)}

    def step(self, action):
        assert action.shape == (1, 29)
        before = self.frame
        self.frame += 1
        done = self.frame == 5
        return SimpleNamespace(
            info={
                play.SUPERMIMIC_STATE_KEY: self.current(0 if done else self.frame),
                play.SUPERMIMIC_TRANSITION_KEY: {
                    "s1": self.current(self.frame)["s"],
                    "f": torch.tensor([before]),
                    "r": torch.tensor([play.MAX_STEP_REWARD]),
                    "done": torch.tensor([done]),
                    "term": torch.tensor([done and self.failed]),
                },
            }
        )


@pytest.mark.parametrize("failed", [False, True])
def test_terminal_display_and_loop_history(monkeypatch, failed):
    histories = []

    def policy(history, goal):
        assert torch.is_inference_mode_enabled()
        histories.append(history.clone())
        return torch.zeros(1, 29)

    monkeypatch.setattr(play, "load_policy", lambda *_: policy)
    rollout = play.PolicyRollout(
        FakeEnv(failed=failed), {"args": {"hist": 2, "goal_frames": 2}}, "selected"
    )
    assert rollout.qpos()[0][0] == 3
    rollout.step()
    rollout.step()

    assert rollout.done and rollout.failed == failed
    policy_qpos, reference = rollout.qpos()
    assert policy_qpos[0] == reference[0] == 5  # Not the auto-reset frame 0.
    assert rollout.total_reward / rollout.max_return == pytest.approx(1)
    assert f"failed={failed}" in rollout.report()
    with pytest.raises(RuntimeError, match="Reset"):
        rollout.step()

    rollout.reset()
    assert not rollout.done
    assert rollout.steps == rollout.total_reward == 0
    rollout.step()
    torch.testing.assert_close(histories[-1], torch.tensor([[[3.0], [3.0]]]))
