"""RoboDojo's per-episode loop for the Rung policy: one observation, one action, repeat.

Runs inside the simulator process (``EvalEnv.eval_one_episode`` imports it). Rung answers
every ``get_action`` with a single action, so this loop never plays an open-loop chunk.
"""


def eval_one_episode(TASK_ENV, model_client):  # noqa: N803 (RoboDojo's argument name)
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        for action in model_client.call(func_name="get_action"):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
