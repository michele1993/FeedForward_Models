import gym
import numpy as np

class Parallel_Mod_Pendulum(gym.Wrapper):

    def __init__(self,make_env,num_envs=1):

        super().__init__(make_env())

        self.num_envs = num_envs
        self.envs = [make_env() for env_idx in range(num_envs)]



    def reset(self):

        #return np.asarray([env.reset() for env in self.envs])

        states = []
        #Reset always

        for env in self.envs:

            env.reset()
            env.unwrapped.state = [np.pi,0]
            states.append(env.unwrapped._get_obs())


        return np.asanyarray(states) #.reshape(self.num_envs,-1)




    def step(self,actions):

        next_states, rewards, dones, infos = [],[],[],[]
        actions = actions.reshape(self.num_envs)

        for env, action in zip(self.envs, actions):

            next_state,reward,done,info = env.step([action])
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)


        return np.asarray(next_states), np.asarray(rewards), np.asarray(dones), np.asarray(infos)


    def reset_at(self,indx):

        return self.envs[indx].reset()

