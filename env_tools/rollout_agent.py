enum_2_action = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
    4: 'BOMB',
    5: 'WAIT',
}

def rollout_agent(agent, env, replay_file_dir):
    ob = env.reset()
    replay = ""
    replay += env.render() + "\n"
    # agent = ExpertAgent()
    while True:
        action, _ = agent.get_action(ob)
        next_ob, reward, done, info = env.step(action)
        ob = next_ob
        replay += env.render() + "\n"
        print(enum_2_action[action])
        print(env.render())
        if done:
            with open('replays/replay_file_dir/replay.txt', 'w') as f:
                f.write(replay)
            break