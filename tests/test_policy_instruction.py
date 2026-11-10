from types import SimpleNamespace

import pytest

from rung.adapters.robolab.client import RungPolicyClient


@pytest.mark.parametrize(
    "override,expected",
    [(None, "native task wording"), ("historical specific wording", "historical specific wording")],
)
def test_actor_wording_override_leaves_native_input_unchanged(override, expected):
    client = RungPolicyClient(
        interface="direct", policy_factory=lambda emb: None, policy_instruction=override
    )
    client.episode_idx = 0
    seen = []

    def start(obs, instruction, env_id):
        seen.append(instruction)
        client._sessions[env_id] = (None, SimpleNamespace(get_action=lambda obs: [0]), None)

    client._start = start
    native_instruction = "native task wording"
    client.infer_batch({}, native_instruction, env_ids=[0])
    assert seen == [expected]
    assert native_instruction == "native task wording"


@pytest.mark.parametrize("text", ["", "  ", 42, "x" * 4097])
def test_invalid_policy_instruction_rejected(text):
    with pytest.raises(ValueError, match="policy instruction"):
        RungPolicyClient(
            interface="direct", policy_factory=lambda emb: None, policy_instruction=text
        )
