import torch

import numpy as np 

class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05, queue_len:int=1024, classes_n: int=10, delta=0.0):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.delta = delta

        self.classes_n = classes_n
        self.queue_len = queue_len
        self.register_buffer(name='logit_queue', tensor=torch.zeros((queue_len, classes_n)), persistent=False)
        self.cur_len = 0 

    def add_to_queue(self, logits: torch.FloatTensor)-> None:
        '''
        :param logits: (N, K)
        '''
        batch_size = logits.size(0)
        classes_n = logits.size(1)
        assert (classes_n == self.classes_n)

        new_queue = torch.concat([logits, self.logit_queue], dim=0)
        self.logit_queue = new_queue[:self.queue_len, :]

        self.cur_len += batch_size
        
        self.cur_len = min(self.cur_len, self.queue_len)

        return 

    def queue_full(self)-> bool:
        
        return self.cur_len == self.queue_len



    @torch.no_grad()
    def forward(self, logits: torch.FloatTensor):
        '''
        :param logits: (N, K)
        '''
        batch_size = logits.size(0)
        all_logits = self.logit_queue 
        
        initial_Q = torch.softmax(all_logits/self.epsilon, dim=1) 
        # Q = torch.exp(logits / self.epsilon).t() # (K, N)
        Q = initial_Q.clone().t() 
        N = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        assert (torch.any(torch.isinf(sum_Q)) == False), "sum_Q is too large"
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            sum_of_rows += self.delta # for numerical stability 
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols 
            Q /= N

        Q *= N  # the colomns must sum to 1 so that Q is an assignment

        batch_assignments = Q.t()[:batch_size, :]
        return batch_assignments, sum_of_rows.squeeze(), sum_of_cols.squeeze()  

# https://github.com/yukimasano/self-label/blob/master/sinkhornknopp.py
def optimize_L_sk(self,PS: np.array, L: np.array, lamb: int=25 ):
    '''
    :param PS: (N, K) probability matrix, K is the number of clusters 
    :param L: (N,) labels 

    '''
    K = PS.size(1)
    N = PS.size(0)
    
    r = np.ones((K, 1), dtype=self.dtype) / K
    c = np.ones((N, 1), dtype=self.dtype) / N
    inv_K = self.dtype(1./K)
    inv_N = self.dtype(1./N)

    PS = PS.T # now it is K x N
    PS **= lamb  # K x N
 
    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (PS @ c)          # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
    # inplace calculations.
    PS *= np.squeeze(c)
    PS = PS.T
    PS *= np.squeeze(r)
    PS = PS.T

    # produce hard labels 
    argmaxes = np.nanargmax(self.PS, 0) # size N
    newL = torch.LongTensor(argmaxes)

    return newL
