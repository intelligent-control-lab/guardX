# Must import this before torch, otherwise jaxlib will get error: "DLPack tensor is on GPU, but no GPU backend was provided"
from jax import numpy as jp

import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import a2c_core as core
from utils.logx import EpochLogger, setup_logger_kwargs
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from safe_rl_envs.envs.engine import Engine as  safe_rl_envs_Engine
from utils.safe_rl_env_config import configuration
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2CBufferX:
    """
    A buffer for storing trajectories experienced by a A2C agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    
    Important notice! This bufferX assumes only one batch of episodes is collected per epoch.
    """

    def __init__(self, env_num, max_ep_len, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, obs_dim[0])), dtype=torch.float32).to(device)
        self.act_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, act_dim[0])), dtype=torch.float32).to(device)
        self.adv_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.rew_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.ret_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.val_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.logp_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.gamma, self.lam = gamma, lam
        self.ptr = np.zeros(env_num, dtype=np.int16)
        self.path_start_idx = np.zeros(env_num, dtype=np.int16)
        self.max_ep_len = max_ep_len
        self.env_num = env_num

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        All input are env_num batch elements. E.g. shape(obs) = (env_num, obs_shape).
        """
        # all all environments should have run same steps, and buffer has to have room so you can store     
        assert len(set(self.ptr)) == 1
        assert self.ptr[0] < self.max_ep_len
        ptr = self.ptr[0]
        self.obs_buf[:,ptr,:] = obs
        self.act_buf[:,ptr,:] = act
        self.rew_buf[:,ptr] = rew
        self.val_buf[:,ptr] = val
        self.logp_buf[:,ptr] = logp
        self.ptr += 1
        
    def finish_path(self, last_val=None, done=None):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument row entry should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        
        The "done" argument indicates which environment should be considered
        for finish_path.
        """ 
        if np.all(self.path_start_idx == 0) and np.all(self.ptr == self.max_ep_len):
            # simplest case, all enviroment is done at end batch episode, 
            # proceed with batch operation
            if len(last_val.shape) == 1:
                last_val = last_val.unsqueeze(1)
            assert last_val.shape == (self.env_num, 1)
            rews = torch.hstack((self.rew_buf, last_val))
            vals = torch.hstack((self.val_buf, last_val))
            
            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:,:-1] + self.gamma * vals[:,1:] - vals[:,:-1]
            self.adv_buf = torch.from_numpy(core.batch_discount_cumsum(deltas, self.gamma * self.lam).astype(np.float32)).to(device)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf = torch.from_numpy(core.batch_discount_cumsum(rews, self.gamma)[:,:-1].astype(np.float32)).to(device)
            
        else:
            # path slice are different for each environment, 
            # separate treatement is required for each environment
            done_env_idx_all = np.where(done == 1)[0]
            for done_env_idx in done_env_idx_all:
                path_slice = slice(self.path_start_idx[done_env_idx], self.ptr[done_env_idx])
                rews = np.append(self.rew_buf[done_env_idx, path_slice].cpu().numpy(), last_val[done_env_idx].cpu().numpy())
                vals = np.append(self.val_buf[done_env_idx, path_slice].cpu().numpy(), last_val[done_env_idx].cpu().numpy())
                
                # the next two lines implement GAE-Lambda advantage calculation
                deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
                
                self.adv_buf[done_env_idx, path_slice] = torch.from_numpy(core.discount_cumsum(deltas, self.gamma * self.lam).astype(np.float32)).to(device)

                # the next line computes rewards-to-go, to be targets for the value function
                
                self.ret_buf[done_env_idx, path_slice] = torch.from_numpy(core.discount_cumsum(rews, self.gamma)[:-1].astype(np.float32)).to(device)
                
                self.path_start_idx[done_env_idx] = self.ptr[done_env_idx]

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert len(set(self.ptr)) == 1
        assert self.ptr[0] == self.max_ep_len    # buffer has to be full before you can get
        self.ptr = np.zeros(self.env_num, dtype=np.int16)
        self.path_start_idx = np.zeros(self.env_num, dtype=np.int16)
        # the next two lines implement the advantage normalization trick
        def normalized_advantage(adv_buf_instance):
            adv_mean, adv_std = mpi_statistics_scalar(adv_buf_instance)
            adv_buf_instance = (adv_buf_instance - adv_mean) / adv_std
            return adv_buf_instance
        self.adv_buf = torch.from_numpy(np.asarray([normalized_advantage(adv_buf_instance) for adv_buf_instance in self.adv_buf.cpu().numpy()])).to(device)
        
        data = dict(obs=self.obs_buf.view(self.env_num * self.max_ep_len, self.obs_buf.shape[-1]), 
                    act=self.act_buf.view(self.env_num * self.max_ep_len, self.act_buf.shape[-1]),
                    ret=self.ret_buf.view(self.env_num * self.max_ep_len),
                    adv=self.adv_buf.view(self.env_num * self.max_ep_len),
                    logp=self.logp_buf.view(self.env_num * self.max_ep_len),
        )
        return {k: v for k,v in data.items()}



def a2c(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        env_num=100, max_ep_len=1000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, delay=2, lam=0.97,
        logger_kwargs=dict(), save_freq=10, model_save=False):
    """
    Asynchronous Actor-critic (A2C)

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to A2C.

        seed (int): Seed for random number generators.

        env_num (int): Number of environment copies being run in parallel.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    #! TODO: sync the random seed with the environment 
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(max_ep_len * env_num / num_procs())
    buf = A2CBufferX(env_num, max_ep_len, obs_dim, act_dim, gamma, lam)

    #! TODO: make sure max_ep_len of buffer is the same with the max_ep_len setting from environment, error if not


    # Set up function for computing A2C policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        # for i in range(max(1, train_v_iters // delay)):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()

    # reset environment
    o = env.reset()
    # return, length, cost of env_num batch episodes
    ep_ret, ep_len, ep_cost = np.zeros(env_num), np.zeros(env_num, dtype=np.int16), np.zeros(env_num) 
    # cum_cost is the cumulative cost over the training
    cum_cost = 0 
    
    max_ep_len_ret = np.zeros(env_num)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(max_ep_len):

            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, info = env.step(a)
            assert 'cost' in info.keys()
            
            # Track cumulative cost over training
            cum_cost += info['cost'].cpu().numpy().squeeze().sum()

            # update return, length, cost of env_num batch episodes
            ep_ret += r.cpu().numpy().squeeze()
            max_ep_len_ret += r.cpu().numpy().squeeze()
            ep_len += 1
            assert ep_cost.shape == info['cost'].cpu().numpy().squeeze().shape
            ep_cost += info['cost'].cpu().numpy().squeeze()

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v.cpu().numpy())
            
            # Update obs (critical!)
            o = next_o

            timeout = (t + 1) == max_ep_len
            terminal = d.cpu().numpy().any() > 0 or timeout

            if terminal:
                # if trajectory didn't reach terminal state, bootstrap value target
                _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                if timeout:
                    done = np.ones(env_num) # every environment needs to finish path
                    logger.store(EpRet=ep_ret[np.where(ep_len == max_ep_len)],
                                 EpLen=ep_len[np.where(ep_len == max_ep_len)],
                                 EpCost=ep_cost[np.where(ep_len == max_ep_len)])
                    logger.store(MaxEpLenRet=max_ep_len_ret)
                    buf.finish_path(v, done)
                    # reset environment 
                    o = env.reset()
                    ep_ret, ep_len, ep_cost = np.zeros(env_num), np.zeros(env_num, dtype=np.int16), np.zeros(env_num)
                else:
                    # trajectory finished for some environment
                    done = d.cpu().numpy() # finish path for certain environments
                    v[np.where(done == 1)] = torch.zeros(np.where(done == 1)[0].shape[0]).to(device)
                    
                    logger.store(EpRet=ep_ret[np.where(done == 1)], EpLen=ep_len[np.where(done == 1)], EpCost=ep_cost[np.where(done == 1)])
                    ep_ret[np.where(done == 1)], ep_len[np.where(done == 1)], ep_cost[np.where(done == 1)]\
                        =   np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0])
                    
                    buf.finish_path(v, done)
                       
                    # only reset observations for those done environments 
                    o = env.reset_done()


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1) and model_save:
            logger.save_state({'env': env}, None)

        # Perform A2C update!
        update()
        
        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*local_steps_per_epoch)


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('MaxEpLenRet', average_only=True)
        logger.log_tabular('EpCost', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('VVals', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

def create_env(args):
    # env =  safe_rl_envs_Engine(configuration(args.task))
    #! TODO: make engine configurable
    config = {
        'num_envs':args.env_num,
        '_seed':args.seed,
        }
    env = safe_rl_envs_Engine(config)
    return env


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Goal_Point_8Hazards')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--env_num', type=int, default=400)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='a2c')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--delay', type=int, default=2)
    parser.add_argument('--train_v_iters', type=int, default=1000)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    exp_name = args.task + '_' + args.exp_name  \
                        + '_' + 'epochs' + str(args.epochs) + '_' \
                        + 'step' + str(args.max_ep_len * args.env_num)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # whether to save model
    model_save = True if args.model_save else False
    a2c(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, env_num=args.env_num, max_ep_len=args.max_ep_len, epochs=args.epochs,
        logger_kwargs=logger_kwargs, model_save=model_save)