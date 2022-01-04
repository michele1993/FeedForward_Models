import gym
from Envir_wrappers.Inv_Pendulum_Wrapper import ModifiedInvPendulum
from Feed_forward.AR_model.Auto_regressive import AR_model
import torch


n_t_steps = 200
n_eps = 30000
discount =  0.95
t_print = 100
backward_t_steps = 200
beta_aver = 0.01
std = 1


b_env = gym.make("Pendulum-v0")
env = ModifiedInvPendulum(b_env)

AR = AR_model(t_steps = backward_t_steps, std = std)

sum_rwd=[]
best_rwd = []
#cum_av_collected_rwd = torch.zeros(n_t_steps)
loss = 0

running_ave = 0


for ep in range(n_eps):

    env.reset()
    rwds = torch.zeros(n_t_steps)
    undis_rwds = torch.empty(n_t_steps)
    p = 0
    done = False
    log_ps = []

    inputs = torch.zeros(backward_t_steps)


    while not done:

        sampled_as, log_p = AR.sample_actions(inputs)
        log_ps.append(log_p)


        for t in range(backward_t_steps):

            _, rwd, done, _ = env.step([sampled_as[t].numpy()])


            undis_rwds[p] = rwd
            p+=1

        inputs = sampled_as


    log_ps = torch.cat(log_ps)

    sum_rwd.append(sum(undis_rwds))
    best_rwd.append(torch.max(undis_rwds))

    #cum_av_collected_rwd = torch.max(rwds)

    cum_discounted_rwd = AR.compute_returns(undis_rwds,discount)

    #cum_discounted_rwd_2 = AR.compute_returns_2(undis_rwds,discount)


    advantage = cum_discounted_rwd - running_ave
    running_ave += beta_aver * advantage



    loss += AR.update(advantage, log_ps).detach() #


    if ep % t_print == 0:


        print("loss: ", loss / t_print)
        print(ep,"av rwd", sum(sum_rwd) /t_print)
        print("best rwd ", sum(best_rwd) / t_print)
        #print("av_best_indx ", sum(idx_best_rwd)/200, "\n")
        sum_rwd = []
        best_rwd = []
        loss = 0



    # if ep % 500 ==0:
    #      #print("actions", sampled_as.view(-1,1))
    #
    #      AR.update_std()