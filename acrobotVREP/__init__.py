from transformations import euler_matrix
from gym.envs.registration import register

register(
    id='acrobotVREP-v0',
    entry_point='acrobotVREP.envs:AcrobotVrepEnv',
)
