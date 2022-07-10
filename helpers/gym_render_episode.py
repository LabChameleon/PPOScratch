from gym import make

def render_episode(env_name, policy):
    env = make(env_name)
    obs = env.reset()
    env_finished = False
    while not env_finished:
        action = policy.get_action(obs)
        obs, _, env_finished, _ = env.step(action)
        env.render()
    env.close()
