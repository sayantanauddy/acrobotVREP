from gym.envs.registration import register

register(
    id='acrobotVREP-v0',
    entry_point='acrobotVREP.envs:AcrobotVrepEnv',
    max_episode_steps=200,
)
