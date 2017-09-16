import gym
from gym import spaces
import numpy as np
from pypot.vrep.io import VrepIO
from acrobotVREP.transformations import euler_matrix

host = '127.0.0.1'
port = 19997
scene='/home/sayantan/computing/repositories/acrobotVREP/vrep_scenes/acrobot.ttt'

class AcrobotVrepEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.max_speed = 10.0
        self.max_torque = 0.3
        self.dt = .01

        # Coordinates of the outer link tip in local (homogeneous) coordinates
        self.outer_tip_local = np.array([[0.0], [0.0], [-0.1], [1.0]])

        # The tip of outer_link should go above this height (z coordinate in the global frame)
        self.threshold = 1.05

        self.vrepio = VrepIO(vrep_host=host,
                             vrep_port=port,
                             scene=scene,
                             start=False)

        #self.vrepio.call_remote_api('simxSynchronous',True)
        self.vrepio.start_simulation()

        print('(AcrobotVrepEnv) initialized')

        obs = np.array([np.inf] * 4)

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(-obs, obs)

    def _self_observe(self):

        self.active_joint_pos = self.vrepio.call_remote_api('simxGetJointPosition',
                                                            self.vrepio.get_object_handle('active_joint'),
                                                            streaming=True)

        # VREP Rempte API does not provide a direct function for retrieving joint velocities
        # Refer to http://www.forum.coppeliarobotics.com/viewtopic.php?f=9&t=2393 for details
        self.active_joint_velocity = self.vrepio.call_remote_api('simxGetObjectFloatParameter',
                                                                 self.vrepio.get_object_handle('active_joint'),
                                                                 2012,
                                                                 streaming=True)

        self.passive_joint_pos = self.vrepio.call_remote_api('simxGetJointPosition',
                                                             self.vrepio.get_object_handle('passive_joint'),
                                                             streaming=True)

        # VREP Rempte API does not provide a direct function for retrieving joint velocities
        # Refer to http://www.forum.coppeliarobotics.com/viewtopic.php?f=9&t=2393 for details
        self.passive_joint_velocity = self.vrepio.call_remote_api('simxGetObjectFloatParameter',
                                                                  self.vrepio.get_object_handle('passive_joint'),
                                                                  2012,
                                                                  streaming=True)

        self.observation = np.array([
            self.active_joint_pos,
            self.active_joint_velocity,
            self.passive_joint_pos,
            self.passive_joint_velocity
        ]).astype('float32')

    def _render(self, mode='human', close=False):
        return

    def _step(self, actions):

        # Advance simulation by 1 step
        #self.vrepio.call_remote_api('simxSynchronousTrigger')

        actions = np.clip(actions, -self.max_torque, self.max_torque)[0]
        apply_torque = actions
        max_vel = 0.0

        # If the torque to be applied is negative, change the sign of the max velocity and then apply a positive torque
        # with the same magnitude

        if apply_torque < 0.0:
            max_vel = -self.max_speed
            apply_torque = -apply_torque
        else:
            max_vel = self.max_speed

        # step
        # Set the target velocity
        self.vrepio.call_remote_api('simxSetJointTargetVelocity',
                                    self.vrepio.get_object_handle('active_joint'),
                                    max_vel,
                                    sending=True)

        # Set the torque of active_joint
        self.vrepio.call_remote_api('simxSetJointForce',
                                    self.vrepio.get_object_handle('active_joint'),
                                    apply_torque,
                                    sending=True)

        # observe again
        self._self_observe()

        # cost
        # If the tip of the outer link is below threshold then reward is -1 else 0

        # First get the position of the outer link frame
        x, y, z = self.vrepio.call_remote_api('simxGetObjectPosition',
                                              self.vrepio.get_object_handle('outer_link'),
                                              -1,
                                              streaming=True)

        # Then calculate the orientation of the outer link frame in Euler angles
        a, b, g = self.vrepio.call_remote_api('simxGetObjectOrientation',
                                              self.vrepio.get_object_handle('outer_link'),
                                              -1,
                                              streaming=True)

        # Compute the homogeneous rotation matrix
        rot_mat = euler_matrix(a, b, g)

        # Insert the position
        rot_mat[0][3] = x
        rot_mat[1][3] = y
        rot_mat[2][3] = z

        # The outer tip of outer_link is at coordinate (0,0,-0.1) in the local frame
        # Convert this into a position in the global frame
        tip_global = np.dot(rot_mat, self.outer_tip_local)

        # Consider the height (z-coordinate) of the tip in the global frame
        tip_height_global = tip_global[2][0]

        # If this height is above the threshold then reward is 0 else -1
        # If height is above threshold, terminate episode
        terminal = False
        if tip_height_global >= self.threshold:
            terminal = True
            reward = 0
        else:
            reward = -1

        return self.observation, reward, terminal, {}

    def _reset(self):
        self.vrepio.stop_simulation()
        self.vrepio.start_simulation()
        self._self_observe()
        return self.observation

    def _destroy(self):
        self.vrepio.stop_simulation()

if __name__ == '__main__':
    env = AcrobotVrepEnv()
    for k in range(5):
        print '==============================='
        observation = env.reset()
        for _ in range(20):
            #   env.render()
            action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)

    print('simulation ended. leaving in 5 seconds...')
