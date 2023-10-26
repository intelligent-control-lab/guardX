import numpy as np
import torch 
import scipy 

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def batch_discount_cumsum(x, discount):
    return torch.from_numpy(np.asarray([discount_cumsum(x_row, discount) for x_row in x.cpu().numpy()]))
    
    
def flatten():
    # x = torch.ones([1,4,2], dtype=torch.float32)
    # y = 2*x
    # z = 4*x
    # d = torch.cat((x,y,z), dim=0)
    # d_flatten = d.view(d.shape[0] * d.shape[1], d.shape[-1])
    # print(f"d is {d}")
    # print(f"flatten d is {d_flatten}")
    
    x = torch.from_numpy(np.random.rand(6,3))
    print(f"x is {x}")
    print(f"flatten x is {x.view(6*3)}")
    
    


if __name__ == '__main__':
    # x = np.ones(1000, dtype=np.float32)
    # discount = 0.5 
    # discount_sum = discount_cumsum(x, discount)
    # print(f"the discount sum is {discount_sum}")
    
    
    # batch_x = np.tile(x.reshape(1,-1), (100,1))
    # x_torch = torch.from_numpy(batch_x)
    # discount_sum_batch = batch_discount_cumsum(x_torch, discount)
    # print(f"the discount sum torch is {discount_sum_batch}")
    
    flatten()
    
    
    