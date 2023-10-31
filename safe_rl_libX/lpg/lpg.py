# Must import this before torch, otherwise jaxlib will get error: "DLPack tensor is on GPU, but no GPU backend was provided"
from jax import numpy as jp

import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
import gym
import time
import copy
import lpg_core as core
from utils.logx import EpochLogger, setup_logger_kwargs, colorize
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from safe_rl_envs.envs.engine import Engine as  safe_rl_envs_Engine
from utils.safe_rl_env_config import configuration
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
class LpgBufferX:
    """
    A buffer for storing trajectories experienced by a LPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    Important notice! This bufferX assumes only one batch of episodes is collected per epoch.
    """
    def __init__(self, env_num, max_ep_len, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, obs_dim[0])), dtype=torch.float32).to(device)
        self.act_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, act_dim[0])), dtype=torch.float32).to(device)
        self.act_safe_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, act_dim[0])), dtype=torch.float32).to(device)
        self.adv_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.rew_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.ret_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.val_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.cost_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.qc_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.targetc_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.logp_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.mu_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, act_dim[0])), dtype=torch.float32).to(device)
        self.logstd_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, act_dim[0])), dtype=torch.float32).to(device)
        self.gamma, self.lam = gamma, lam
        self.ptr = np.zeros(env_num, dtype=np.int16)
        self.path_start_idx = np.zeros(env_num, dtype=np.int16)
        self.max_ep_len = max_ep_len
        self.env_num = env_num
    
    def store(self, obs, act, act_safe, rew, val, logp, mu, logstd, cost, qc):
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
        self.act_safe_buf[:,ptr,:] = act_safe
        self.rew_buf[:,ptr] = rew
        self.val_buf[:,ptr] = val
        self.logp_buf[:,ptr] = logp
        self.mu_buf[:,ptr,:] = mu
        self.logstd_buf[:,ptr,:] = logstd
        self.cost_buf[:,ptr] = cost
        self.qc_buf[:,ptr] = qc
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
            assert last_val.shape == (self.env_num, 1)
            rews = torch.hstack((self.rew_buf, last_val))
            vals = torch.hstack((self.val_buf, last_val))
            
            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:,:-1] + self.gamma * vals[:,1:] - vals[:,:-1]
            self.adv_buf = torch.from_numpy(core.batch_discount_cumsum(deltas, self.gamma * self.lam).astype(np.float32)).to(device)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf = torch.from_numpy(core.batch_discount_cumsum(rews, self.gamma)[:,:-1].astype(np.float32)).to(device)
            
            qcs = torch.hstack((self.qc_buf, torch.zeros(self.env_num, 1)))
            costs = torch.hstack((self.cost_buf, torch.zeros(self.env_num, 1)))
            self.targetc_buf = costs[:,:-1] + self.gamma * qcs[:,1:]
            
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
                
                qcs = np.append(self.qc_buf[done_env_idx, path_slice].cpu().numpy(), 0)
                costs = np.append(self.cost_buf[done_env_idx, path_slice].cpu().numpy(), 0)
                self.targetc_buf[done_env_idx, path_slice] = torch.from_numpy((costs[:-1] + self.gamma * qcs[1:]).astype(np.float32)).to(device)
                
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
                    act_safe=self.act_safe_buf.view(self.env_num * self.max_ep_len, self.act_safe_buf.shape[-1]),
                    ret=self.ret_buf.view(self.env_num * self.max_ep_len),
                    adv=self.adv_buf.view(self.env_num * self.max_ep_len),
                    logp=self.logp_buf.view(self.env_num * self.max_ep_len),
                    mu=self.mu_buf.view(self.env_num * self.max_ep_len, self.mu_buf.shape[-1]),
                    logstd=self.logstd_buf.view(self.env_num * self.max_ep_len, self.logstd_buf.shape[-1]),
                    cost=self.cost_buf.view(self.env_num * self.max_ep_len),
                    targetc=self.targetc_buf.view(self.env_num * self.max_ep_len)
        )
        return {k: v for k,v in data.items()}

def get_net_param_np_vec(net):
    """
        Get the parameters of the network as numpy vector
    """
    return torch.cat([val.flatten() for val in net.parameters()], axis=0).detach().cpu().numpy()

def assign_net_param_from_flat(param_vec, net):
    param_sizes = [np.prod(list(val.shape)) for val in net.parameters()]
    ptr = 0
    for s, param in zip(param_sizes, net.parameters()):
        param.data.copy_(torch.from_numpy(param_vec[ptr:ptr+s]).reshape(param.shape))
        ptr += s

def cg(Ax, b, cg_iters=100):
    x = np.zeros_like(b)
    r = b.copy() # Note: should be 'b - Ax', but for x=0, Ax=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r,r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r,r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
        # early stopping 
        if np.linalg.norm(p) < EPS:
            break
    return x

def auto_grad(objective, net, to_numpy=True):
    """
    Get the gradient of the objective with respect to the parameters of the network
    """
    grad = torch.autograd.grad(objective, net.parameters(), create_graph=True)
    if to_numpy:
        return torch.cat([val.flatten() for val in grad], axis=0).detach().cpu().numpy()
    else:
        return torch.cat([val.flatten() for val in grad], axis=0)

def auto_hession_x(objective, net, x):
    """
    Returns 
    """
    jacob = auto_grad(objective, net, to_numpy=False)
    
    return auto_grad(torch.dot(jacob, x), net, to_numpy=True)

def lpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        env_num=100, max_ep_len=1000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, ccritic_lr=1e-3, train_v_iters=80, train_ccritic_iters=80, lam=0.97, 
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, backtrack_coeff=0.8, backtrack_iters=100, model_save=False):
    """
    LPG
 
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
            you provided to LPG.

        seed (int): Seed for random number generators.

        env_num (int): Number of environment copies being run in parallel.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.
        
        ccritic_lr (float): Learning rate for cost critic.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
            
        train_ccritic_iters (int): Number of gradient descent steps to take on 
            cost critic function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
            
        backtrack_coeff (float): Scaling factor for line search.
        
        backtrack_iters (int): Number of line search steps.
        
        model_save (bool): If saving model.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
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
    buf = LpgBufferX(env_num, max_ep_len, obs_dim, act_dim, gamma, lam)


    def compute_kl_pi(data, cur_pi):
        """
        Return the sample average KL divergence between old and new policies
        """
        obs, mu_old, logstd_old = data['obs'], data['mu'], data['logstd']
        
        # Average KL Divergence  
        average_kl = cur_pi._d_kl(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(mu_old, dtype=torch.float32),
            torch.as_tensor(logstd_old, dtype=torch.float32), device=device)
        
        return average_kl


    def compute_loss_pi(data, cur_pi):
        """
        The reward objective for lpg (lpg policy loss)
        """
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss 
        pi, logp = cur_pi(obs, act)
        # loss_pi = -(logp * adv).mean()
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        
        return loss_pi, pi_info
        

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    def compute_loss_ccritic(data):
        obs, act_safe, targetc = data['obs'], data['act_safe'], data['targetc']

        # Get current C estimate
        obs_act = torch.cat((obs, act_safe), dim=1)
        return ((ac.ccritic(obs_act) - targetc)**2).mean()
    
    # Set up optimizers for policy and value function
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    ccritic_optimizer = Adam(ac.ccritic.parameters(), lr=ccritic_lr)

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)
    
    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, ac.pi)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()


        # SafeLayer policy update core impelmentation 
        loss_pi, pi_info = compute_loss_pi(data, ac.pi)
        g = auto_grad(loss_pi, ac.pi) # get the flatten gradient evaluted at pi old 
        kl_div = compute_kl_pi(data, ac.pi)
        Hx = lambda x: auto_hession_x(kl_div, ac.pi, torch.FloatTensor(x).to(device))
        x_hat    = cg(Hx, g)             # Hinv_g = H \ g
        
        s = x_hat.T @ Hx(x_hat)
        s_ep = s if s < 0. else 1 # log s negative appearence 
            
        x_direction = np.sqrt(2 * target_kl / (s+EPS)) * x_hat
        
        # copy an actor to conduct line search 
        actor_tmp = copy.deepcopy(ac.pi)
        def set_and_eval(step):
            new_param = get_net_param_np_vec(ac.pi) - step * x_direction
            assign_net_param_from_flat(new_param, actor_tmp)
            kl = compute_kl_pi(data, actor_tmp)
            pi_l, _ = compute_loss_pi(data, actor_tmp)
            
            return kl, pi_l
        
        # update the policy such that the KL diveragence constraints are satisfied and loss is decreasing
        for j in range(backtrack_iters):
            try:
                kl, pi_l_new = set_and_eval(backtrack_coeff**j)
            except:
                import ipdb; ipdb.set_trace()
            
            if (kl.item() <= target_kl and pi_l_new.item() <= pi_l_old):
                print(colorize(f'Accepting new params at step %d of line search.'%j, 'green', bold=False))
                # update the policy parameter 
                new_param = get_net_param_np_vec(ac.pi) - backtrack_coeff**j * x_direction
                assign_net_param_from_flat(new_param, ac.pi)
                loss_pi, pi_info = compute_loss_pi(data, ac.pi) # re-evaluate the pi_info for the new policy
                break
            if j==backtrack_iters-1:
                print(colorize(f'Line search failed! Keeping old params.', 'yellow', bold=False))

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
            
        # update the cost critic 
        for i in range(train_ccritic_iters):
            ccritic_optimizer.zero_grad()
            loss_ccritic = compute_loss_ccritic(data)
            loss_ccritic.backward()
            mpi_avg_grads(ac.ccritic)    # average grads across MPI processes
            ccritic_optimizer.step()
        
        
        # Log changes from update        
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     EpochS = s_ep)

    # Prepare for interaction with environment
    start_time = time.time()
    
    # reset environment
    o = env.reset()
    # return, length, cost of env_num batch episodes
    ep_ret, ep_len, ep_cost, ep_cost_ret = np.zeros(env_num), np.zeros(env_num, dtype=np.int16), np.zeros(env_num), np.zeros(env_num) 
    # cum_cost is the cumulative cost over the training
    cum_cost, prev_c = 0, np.zeros(env_num)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(max_ep_len):
            a, v, logp, mu, logstd, qc = ac.step(torch.as_tensor(o, dtype=torch.float32))
            if(t == 0):
                ac.ccritic.store_init(o, a)
                # print("Initial D(x0): ", ac.ccritic.get_Q_init())
            
            # apply safe layer to get corrected action
            warmup_ratio = 1.0/3.0
            # warmup_ratio = 0.
            if epoch > epochs * warmup_ratio:
                a_safe = ac.ccritic.safety_correction(o, a, prev_c)
                assert a_safe is not a
            else:
                a_safe = a 
                    
            # step actions
            next_o, r, d, info = env.step(a_safe)
            assert 'cost' in info.keys()
            
            # Track cumulative cost over training
            cum_cost += info['cost'].cpu().numpy().squeeze().sum()
            
            # update return, length, cost of env_num batch episodes
            ep_ret += r.cpu().numpy().squeeze()
            ep_cost_ret += info['cost'].cpu().numpy().squeeze() * (gamma ** t)
            ep_len += 1
            
            assert ep_cost.shape == info['cost'].cpu().numpy().squeeze().shape
            ep_cost += info['cost'].cpu().numpy().squeeze()
            
            # save and log
            buf.store(o, a, a_safe, r, v, logp, mu, logstd, info['cost'], qc)
            logger.store(VVals=v.cpu().numpy())
            
            # Update obs (critical!)
            o = next_o
            
            timeout = (t + 1) == max_ep_len
            terminal = d.cpu().numpy().any() > 0 or timeout

            if terminal:
                # if trajectory didn't reach terminal state, bootstrap value target
                _, v, _, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                if timeout:
                    done = np.ones(env_num) # every environment needs to finish path
                    # logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                    logger.store(EpRet=ep_ret[np.where(ep_len == max_ep_len)],
                                 EpLen=ep_len[np.where(ep_len == max_ep_len)],
                                 EpCostRet=ep_cost_ret[np.where(ep_len == max_ep_len)],
                                 EpCost=ep_cost[np.where(ep_len == max_ep_len)])
                    buf.finish_path(v, done)
                    # reset environment 
                    o = env.reset()
                    ep_ret, ep_len, ep_cost, prev_c, ep_cost_ret = np.zeros(env_num), np.zeros(env_num, dtype=np.int16), np.zeros(env_num), np.zeros(env_num), np.zeros(env_num)
                else:
                    # trajectory finished for some environment
                    done = d.cpu().numpy() # finish path for certain environments
                    v[np.where(done == 1)] = torch.zeros(np.where(done == 1)[0].shape[0]).to(device)
                    
                    logger.store(EpRet=ep_ret[np.where(done == 1)], EpLen=ep_len[np.where(done == 1)], 
                                 EpCostRet=ep_cost_ret[np.where(done == 1)], EpCost=ep_cost[np.where(done == 1)])
                    ep_ret[np.where(done == 1)], ep_len[np.where(done == 1)], ep_cost[np.where(done == 1)], prev_c[np.where(done==1)], ep_cost_ret[np.where(done == 1)]\
                        =   np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0])
                    
                    buf.finish_path(v, done)

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, None)

        # Perform lpg update!
        update()
        
        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*local_steps_per_epoch)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpCostRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('EpochS', average_only=True)
        logger.dump_tabular()
        
        
def create_env(args):
    # env =  safe_rl_envs_Engine(configuration(args.task))
    #! TODO: make engine configurable
    config = {'num_envs':args.env_num}
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
    parser.add_argument('--exp_name', type=str, default='lpg')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--target_kl', type=float, default=0.02)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    exp_name = args.task + '_' + args.exp_name + '_' + 'kl' + str(args.target_kl) \
                        + '_' + 'epochs' + str(args.epochs) + '_' \
                        + 'step' + str(args.max_ep_len * args.env_num)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # whether to save model
    model_save = True if args.model_save else False
    lpg(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, env_num=args.env_num, max_ep_len=args.max_ep_len, epochs=args.epochs,
        logger_kwargs=logger_kwargs, model_save=model_save, target_kl=args.target_kl)
