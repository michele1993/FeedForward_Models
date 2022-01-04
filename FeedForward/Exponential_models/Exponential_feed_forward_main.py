import gym
from Envir_wrappers.Inv_Pendulum_Wrapper import ModifiedInvPendulum
from Feed_forward.Exponential_models.Exponential_function import ExpF_agent
#from Feed_forward.Exponential_models.modified_exp_f import ExpF_agent
import torch
import numpy as np

n_exp_fs = 5
n_t_steps = 200
n_eps = 5000
discount = 0.95
t_print = 100
beta_aver = 0.01


b_env = gym.make("Pendulum-v0")
env = ModifiedInvPendulum(b_env)

exp_f = ExpF_agent(n_exps= n_exp_fs,t_steps = n_t_steps)

sum_rwd=[]
best_rwd = []
cum_av_collected_rwd = torch.zeros(n_t_steps)
loss = 0
running_ave = torch.zeros(n_t_steps)
accuracy = []
best_accuracy = []

for ep in range(n_eps):

    env.reset()
    sampled_as = exp_f.sample_actions() #, mean_acts
    rwds = torch.zeros(n_t_steps)
    undis_rwds = torch.empty(n_t_steps)


    for t in range(n_t_steps):


        _, rwd, done, _ = env.step([sampled_as[t].numpy()])

        undis_rwds[t] = torch.from_numpy(rwd)




    sum_rwd.append(torch.mean(undis_rwds))
    best_rwd.append(torch.max(undis_rwds))

    cum_discounted_rwd = exp_f.compute_returns(undis_rwds,discount)

    advantage = cum_discounted_rwd - running_ave
    running_ave += beta_aver * advantage

    #cum_av_collected_rwd_2 = exp_f.compute_returns_2(undis_rwds, discount)


    #advantage = torch.max(undis_rwds)


    loss += exp_f.update(advantage).detach() #




    if ep % t_print == 0 and ep > 1:


        print("loss: ", loss / t_print)
        print(ep,"av rwd", sum(sum_rwd) /t_print)
        print("best rwd ", sum(best_rwd) / t_print)
        #print("av_best_indx ", sum(idx_best_rwd)/200, "\n")

        accuracy.append(sum(sum_rwd) /t_print)

        best_accuracy.append(sum(best_rwd) / t_print)



        sum_rwd = []
        best_rwd = []
        loss = 0


#

np.save("List_actions", sampled_as)
np.save("Accuracy", accuracy)
np.save("Best score", best_accuracy)

# Test:

#env = gym.wrappers.Monitor(env,"record", force=True)
#env.render()
test_t_steps= n_t_steps

env.reset()

with torch.no_grad():
    test_actions = exp_f.deterministic_actions()

print(test_actions)

for t in range(test_t_steps):

    _,rwd,_,_ = env.step([test_actions[t].numpy()])
    print(rwd)


env.close()

np.save("Test_actions", np.array(test_actions))


