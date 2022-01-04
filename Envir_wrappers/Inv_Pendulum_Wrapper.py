import gym
import numpy as np

class ModifiedInvPendulum(gym.Wrapper):

    def __init__(self,env):

        super(ModifiedInvPendulum,self).__init__(env)



    def reset(self): # Ensure each episode starts with the pendulum in the same upward position


        super().reset()
        self.unwrapped.state = [np.pi,0]

        return self.unwrapped._get_obs()

