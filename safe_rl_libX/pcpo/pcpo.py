# Must import this before torch, otherwise jaxlib will get error: "DLPack tensor is on GPU, but no GPU backend was provided"
from jax import numpy as jp

import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import copy
import pcpo_core as core
from utils.logx import EpochLogger, setup_logger_kwargs, colorize
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from utils.safe_rl_env_config import create_env
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

class PCPOBufferX:
    """
    A buffer for storing trajectories experienced by a PCPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, env_num, max_ep_len, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, obs_dim[0])), dtype=torch.float32).to(device)
        self.act_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, act_dim[0])), dtype=torch.float32).to(device)
        self.adv_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.rew_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.ret_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.val_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.cost_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.cost_ret_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.cost_val_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.adc_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.logp_buf = torch.zeros(env_num, max_ep_len, dtype=torch.float32).to(device)
        self.mu_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, act_dim[0])), dtype=torch.float32).to(device)
        self.logstd_buf = torch.zeros(core.combined_shape(env_num, (max_ep_len, act_dim[0])), dtype=torch.float32).to(device)
        self.gamma, self.lam = gamma, lam
        self.ptr = np.zeros(env_num, dtype=np.int16)
        self.path_start_idx = np.zeros(env_num, dtype=np.int16)
        self.max_ep_len = max_ep_len
        self.env_num = env_num
        
    def store(self, obs, act, rew, val, logp, cost, cost_val, mu, logstd):
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
        self.cost_buf[:,ptr] = cost
        self.cost_val_buf[:,ptr] = cost_val
        self.mu_buf[:,ptr,:] = mu
        self.logstd_buf[:,ptr,:] = logstd
        self.ptr += 1
    
    def finish_path(self, last_val=None, last_cost_val=None, done=None):
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
            if len(last_cost_val.shape) == 1:
                last_cost_val = last_cost_val.unsqueeze(1)
            assert last_cost_val.shape == (self.env_num, 1)
            rews = torch.hstack((self.rew_buf, last_val))
            vals = torch.hstack((self.val_buf, last_val))
            costs = torch.hstack((self.cost_buf, last_cost_val))
            cost_vals = torch.hstack((self.cost_val_buf, last_cost_val))
            
            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:,:-1] + self.gamma * vals[:,1:] - vals[:,:-1]
            self.adv_buf = torch.from_numpy(core.batch_discount_cumsum(deltas, self.gamma * self.lam).astype(np.float32)).to(device)
            
            # cost advantage calculation
            cost_deltas = costs[:,:-1] + self.gamma * cost_vals[:,1:] - cost_vals[:,:-1]
            self.adc_buf = torch.from_numpy(core.batch_discount_cumsum(cost_deltas, self.gamma * self.lam).astype(np.float32)).to(device)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf = torch.from_numpy(core.batch_discount_cumsum(rews, self.gamma)[:,:-1].astype(np.float32)).to(device)
            self.cost_ret_buf = torch.from_numpy(core.batch_discount_cumsum(costs, self.gamma)[:,:-1].astype(np.float32)).to(device)
            
        else:
            # path slice are different for each environment, 
            # separate treatement is required for each environment
            done_env_idx_all = np.where(done == 1)[0]
            for done_env_idx in done_env_idx_all:
                path_slice = slice(self.path_start_idx[done_env_idx], self.ptr[done_env_idx])
                rews = np.append(self.rew_buf[done_env_idx, path_slice].cpu().numpy(), last_val[done_env_idx].cpu().numpy())
                vals = np.append(self.val_buf[done_env_idx, path_slice].cpu().numpy(), last_val[done_env_idx].cpu().numpy())
                costs = np.append(self.cost_buf[done_env_idx, path_slice].cpu().numpy(), last_cost_val[done_env_idx].cpu().numpy())
                cost_vals = np.append(self.cost_val_buf[done_env_idx, path_slice].cpu().numpy(), last_cost_val[done_env_idx].cpu().numpy())
                
                # the next two lines implement GAE-Lambda advantage calculation
                deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
                cost_deltas = costs[:-1] + self.gamma * cost_vals[1:] - cost_vals[:-1]
                
                self.adv_buf[done_env_idx, path_slice] = torch.from_numpy(core.discount_cumsum(deltas, self.gamma * self.lam).astype(np.float32)).to(device)
                self.adc_buf[done_env_idx, path_slice] = torch.from_numpy(core.discount_cumsum(cost_deltas, self.gamma * self.lam).astype(np.float32)).to(device)
                
                # the next line computes rewards-to-go, to be targets for the value function
                self.ret_buf[done_env_idx, path_slice] = torch.from_numpy(core.discount_cumsum(rews, self.gamma)[:-1].astype(np.float32)).to(device)
                self.cost_buf[done_env_idx, path_slice] = torch.from_numpy(core.discount_cumsum(costs, self.gamma)[:-1].astype(np.float32)).to(device)
                
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
        def normalized_cost_advantage(adc_buf_instance):
            adv_mean, _ = mpi_statistics_scalar(adc_buf_instance)
            # center cost advantage, but don't scale
            adc_buf_instance = (adc_buf_instance - adv_mean)
            return adc_buf_instance
        self.adv_buf = torch.from_numpy(np.asarray([normalized_advantage(adv_buf_instance) for adv_buf_instance in self.adv_buf.cpu().numpy()])).to(device)
        self.adc_buf = torch.from_numpy(np.asarray([normalized_cost_advantage(adc_buf_instance) for adc_buf_instance in self.adc_buf.cpu().numpy()])).to(device)
        
        data = dict(obs=self.obs_buf.view(self.env_num * self.max_ep_len, self.obs_buf.shape[-1]), 
                    act=self.act_buf.view(self.env_num * self.max_ep_len, self.act_buf.shape[-1]),
                    ret=self.ret_buf.view(self.env_num * self.max_ep_len),
                    adv=self.adv_buf.view(self.env_num * self.max_ep_len),
                    cost_ret=self.cost_ret_buf.view(self.env_num * self.max_ep_len),
                    adc=self.adc_buf.view(self.env_num * self.max_ep_len),
                    logp=self.logp_buf.view(self.env_num * self.max_ep_len),
                    mu=self.mu_buf.view(self.env_num * self.max_ep_len, self.mu_buf.shape[-1]),
                    logstd=self.logstd_buf.view(self.env_num * self.max_ep_len, self.logstd_buf.shape[-1]),
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

def pcpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        env_num=100, max_ep_len=1000, epochs=50, gamma=0.99, 
        vf_lr=1e-3, vcf_lr=1e-3, train_v_iters=80, train_vc_iters=80, lam=0.97, 
        target_kl=0.01, target_cost = 1.5, logger_kwargs=dict(), save_freq=10, kl_proj=True,
        model_save=False, cost_reduction=0):
    """
    Projection-Based Constrained Policy Optimization, 
 
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
            you provided to PCPO.

        seed (int): Seed for random number generators.

        env_num (int): Number of environment copies running in parallel.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        vf_lr (float): Learning rate for value function optimizer.
        
        vcf_lr (float): Learning rate for cost value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
            
        train_vc_iters (int): Number of gradient descent steps to take on 
            cost value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
            
        target_cost (float): Cost limit that the agent should satisfy
        
        kl_proj (bool): Whether to use the KL divergence projection 

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        
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
    buf = PCPOBufferX(env_num, max_ep_len, obs_dim, act_dim, gamma, lam)

    #! TODO: make sure max_ep_len of buffer is the same with the max_ep_len setting from environment, error if not
   
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
    
    def compute_cost_pi(data, cur_pi):
        """
        Return the suggorate cost for current policy
        """
        obs, act, adc, logp_old = data['obs'], data['act'], data['adc'], data['logp']
        
        # Surrogate cost function 
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        surr_cost = (ratio * adc).mean()
        
        return surr_cost
        
    def compute_loss_pi(data, cur_pi):
        """
        The reward objective for PCPO (PCPO policy loss)
        """
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss 
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = (ratio * adv).mean() # the gradient of PCPO requires is for (maximize J) instead of (minimize -J)
        
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        
        return loss_pi, pi_info
        
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    # Set up function for computing cost loss 
    def compute_loss_vc(data):
        obs, cost_ret = data['obs'], data['cost_ret']
        return ((ac.vc(obs) - cost_ret)**2).mean()

    # Set up optimizers for policy and value function
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vcf_optimizer = Adam(ac.vc.parameters(), lr=vcf_lr)

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # log the loss objective and cost function and value function for old policy
        pi_l_old, pi_info_old = compute_loss_pi(data, ac.pi)
        pi_l_old = pi_l_old.item()
        surr_cost_old = compute_cost_pi(data, ac.pi)
        surr_cost_old = surr_cost_old.item()
        v_l_old = compute_loss_v(data).item()


        # PCPO policy update core impelmentation 
        loss_pi, pi_info = compute_loss_pi(data, ac.pi)
        surr_cost = compute_cost_pi(data, ac.pi)
        
        # get Hessian for KL divergence
        kl_div = compute_kl_pi(data, ac.pi)
        Hx = lambda x: auto_hession_x(kl_div, ac.pi, torch.FloatTensor(x).to(device))
        
        # linearize the loss objective and cost function
        g = auto_grad(loss_pi, ac.pi) # get the loss flatten gradient evaluted at pi old 
        b = auto_grad(surr_cost, ac.pi) # get the cost flatten gradient evaluted at pi old
        
        # get the Episoe cost
        EpLen = logger.get_stats('EpLen')[0]
        EpCost = logger.get_stats('EpCost')[0]
        
        # cost constraint linearization
         
        c = EpCost - target_cost 
        rescale  = EpLen
        c /= (rescale + EPS)
        
        # core calculation for PCPO
        Hinv_g   = cg(Hx, g)             # Hinv_g = H \ g        
        approx_g = Hx(Hinv_g)           # g
        # q        = np.clip(Hinv_g.T @ approx_g, 0.0, None)  # g.T / H @ g
        q        = Hinv_g.T @ approx_g
        Linv_b = cg(Hx, b) if kl_proj else b
        approx_b = Hx(Linv_b) if kl_proj else b # b
        
        # solve QP
        # decide optimization cases (feas/infeas, recovery)
        # Determine optim_case (switch condition for calculation,
        # based on geometry of constrained optimization problem)
        if b.T @ b <= 1e-8 and c < 0:
            Hinv_b, r, s, A, B = 0, 0, 0, 0, 0
            optim_case = 4
        else:
            # cost grad is nonzero: PCPO update!
            Hinv_b = cg(Hx, b)                # H^{-1} b
            r = Hinv_b.T @ approx_g          # b^T H^{-1} g
            s = Hinv_b.T @ Hx(Hinv_b)        # b^T H^{-1} b
            A = q - r**2 / s            # should be always positive (Cauchy-Shwarz)
            B = 2*target_kl - c**2 / s  # does safety boundary intersect trust region? (positive = yes)
    
        # get optimal theta-theta_k direction
        trpo_step = np.sqrt((2*target_kl)/q)
        cpo_step = max(0, (trpo_step * b.T @ Hinv_g + c)/(Linv_b.T @ approx_b))
        x_direction = trpo_step * (Hinv_g) - cpo_step * Linv_b
        
        # copy an actor to conduct line search 
        actor_tmp = copy.deepcopy(ac.pi)
        def set_and_eval(step):
            new_param = get_net_param_np_vec(ac.pi) + step * x_direction
            assign_net_param_from_flat(new_param, actor_tmp)
            kl = compute_kl_pi(data, actor_tmp)
            pi_l, _ = compute_loss_pi(data, actor_tmp)
            surr_cost = compute_cost_pi(data, actor_tmp)
            
            return kl, pi_l, surr_cost
        
        kl, pi_l_new, surr_cost_new = set_and_eval(1)
        # update the policy parameter 
        new_param = get_net_param_np_vec(ac.pi) + 1 * x_direction
        assign_net_param_from_flat(new_param, ac.pi)
        
        loss_pi, pi_info = compute_loss_pi(data, ac.pi) # re-evaluate the pi_info for the new policy
        surr_cost = compute_cost_pi(data, ac.pi) # re-evaluate the surr_cost for the new policy
        
        
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
            
        # Cost value function learning
        for i in range(train_vc_iters):
            vcf_optimizer.zero_grad()
            loss_vc = compute_loss_vc(data)
            loss_vc.backward()
            mpi_avg_grads(ac.vc)    # average grads across MPI processes
            vcf_optimizer.step()

        # Log changes from update        
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossCost=surr_cost_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     DeltaLossCost=(surr_cost.item() - surr_cost_old))

    # Prepare for interaction with environment
    start_time = time.time()
    
    o = env.reset()
    ep_ret, ep_len, ep_cost, ep_cost_ret = np.zeros(env_num), np.zeros(env_num, dtype=np.int16), np.zeros(env_num), np.zeros(env_num)
    cum_cost = 0 
    
    max_ep_len_ret = np.zeros(env_num)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(max_ep_len):
            a, v, vc, logp, mu, logstd = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, info = env.step(a)
            assert 'cost' in info.keys()
              
            # Track cumulative cost over training
            cum_cost += info['cost'].cpu().numpy().squeeze().sum()
            ep_ret += r.cpu().numpy().squeeze()
            max_ep_len_ret += r.cpu().numpy().squeeze()
            ep_cost_ret += info['cost'].cpu().numpy().squeeze() * (gamma ** t)
            ep_len += 1
            
            assert ep_cost.shape == info['cost'].cpu().numpy().squeeze().shape
            ep_cost += info['cost'].cpu().numpy().squeeze()

            # save and log
            buf.store(o, a, r, v, logp, info['cost'], vc, mu, logstd)
            logger.store(VVals=v.cpu().numpy())
            
            # Update obs (critical!)
            o = next_o

            timeout = (t + 1) == max_ep_len
            terminal = d.cpu().numpy().any() > 0 or timeout

            if terminal:
                # if trajectory didn't reach terminal state, bootstrap value target
                _, v, vc, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                if timeout:
                    done = np.ones(env_num) # every environment needs to finish path
                    # no bootstrap for timeout and done environment 
                    v[np.where(done == 1)] = torch.zeros(np.where(done == 1)[0].shape[0]).to(device)
                    vc[np.where(done == 1)] = torch.zeros(np.where(done == 1)[0].shape[0]).to(device)
                    
                    logger.store(EpRet=ep_ret[np.where(ep_len == max_ep_len)],
                                 EpLen=ep_len[np.where(ep_len == max_ep_len)],
                                 EpCostRet=ep_cost_ret[np.where(ep_len == max_ep_len)],
                                 EpCost=ep_cost[np.where(ep_len == max_ep_len)])
                    logger.store(MaxEpLenRet=max_ep_len_ret)
                    buf.finish_path(v, vc, done)
                    # reset environment 
                    o = env.reset()
                    ep_ret, ep_len, ep_cost, ep_cost_ret = np.zeros(env_num), np.zeros(env_num, dtype=np.int16), np.zeros(env_num), np.zeros(env_num)
                    max_ep_len_ret = np.zeros(env_num)
                else:
                    # trajectory finished for some environment
                    done = d.cpu().numpy() # finish path for certain environments
                    v[np.where(done == 1)] = torch.zeros(np.where(done == 1)[0].shape[0]).to(device)
                    vc[np.where(done == 1)] = torch.zeros(np.where(done == 1)[0].shape[0]).to(device)
                    
                    logger.store(EpRet=ep_ret[np.where(done == 1)], 
                                 EpLen=ep_len[np.where(done == 1)],
                                 EpCostRet=ep_cost_ret[np.where(done == 1)], 
                                 EpCost=ep_cost[np.where(done == 1)])
                    ep_ret[np.where(done == 1)], ep_len[np.where(done == 1)], ep_cost[np.where(done == 1)], ep_cost_ret[np.where(done == 1)]\
                        =   np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0]), \
                            np.zeros(np.where(done == 1)[0].shape[0])
                    
                    buf.finish_path(v, vc, done)
                       
                    # only reset observations for those done environments 
                    o = env.reset_done()

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, None)

        # Perform PCPO update!
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
        logger.log_tabular('EpCostRet', average_only=True)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, default='Goal_Point_8Hazards')
    parser.add_argument('--target_cost', type=float, default=0.) # the cost limit for the environment
    parser.add_argument('--target_kl', type=float, default=0.02) # the kl divergence limit for PCPO
    parser.add_argument('--cost_reduction', type=float, default=0.) # the cost_reduction limit when current policy is infeasible
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--env_num', type=int, default=1200)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='pcpo')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--l2_proj', action='store_true')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    kl_proj = False if args.l2_proj else True
    
    exp_name = args.task + '_' + args.exp_name \
                + '_' + 'kl' + str(args.target_kl) \
                + '_' + 'target_cost' + str(args.target_cost) \
                + '_' + 'kl_proj' + str(kl_proj) \
                + '_' + 'epochs' + str(args.epochs) \
                + '_' + 'step' + str(args.max_ep_len * args.env_num)
                
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # whether to save model
    model_save = True if args.model_save else False
    pcpo(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, env_num=args.env_num, max_ep_len=args.max_ep_len, epochs=args.epochs,
        logger_kwargs=logger_kwargs, target_cost=args.target_cost, 
        model_save=model_save, target_kl=args.target_kl, cost_reduction=args.cost_reduction, kl_proj=kl_proj)