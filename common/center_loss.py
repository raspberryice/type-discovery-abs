import torch 
from torch import nn 

class CenterLoss(nn.Module):
    '''
    L2 loss for pushing representations close to their centroid. 
    '''
    def __init__(self, dim_hidden: int , num_classes: int, lambda_c: float = 1.0, alpha: float=1.0, weight_by_prob: bool =False):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.alpha= alpha 
        self.weight_by_prob = weight_by_prob 
      
        self.centers = self.register_buffer('centers', torch.zeros((num_classes, dim_hidden), dtype=torch.float), persistent=False) 

    def _compute_prob(self, distance_centers: torch.FloatTensor, y: torch.LongTensor):
        '''
        compute the probability according to student-t distribution 
        Bug in original RoCORE code, added (-1) to power operation.
        '''

        q = 1.0/(1.0+distance_centers/self.alpha) # (batch_size, num_class)
        q = q**((-1) * (self.alpha+1.0)/2.0) 
        q = q / torch.sum(q, dim=1, keepdim=True)
        prob = q.gather(1, y.unsqueeze(1)).squeeze() # (batch_size)
        return prob 


    def forward(self, y: torch.LongTensor, hidden: torch.FloatTensor) -> torch.FloatTensor:
        '''
        :param y: (batch_size, )
        :param hidden: (batch_size, dim_hidden)
        '''
        batch_size = hidden.size(0)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0) # (num_class, batch_size, hid_dim) => (batch_size, num_class, hid_dim)
        expanded_centers = self.centers.expand(batch_size, -1, -1) # (batch_size, num_class, hid_dim)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1) # (batch_size, num_class, hid_dim) => (batch_size, num_class)
        intra_distances = distance_centers.gather(1, y.unsqueeze(1)).squeeze() # (batch_size, num_class) => (batch_size, 1) => (batch_size)

        if self.weight_by_prob:
            prob = self._compute_prob(distance_centers, y)
            loss = 0.5 * self.lambda_c * torch.mean(intra_distances*prob) # (batch_size) => scalar

        else:
            loss = 0.5 * self.lambda_c * torch.mean(intra_distances) # (batch_size) => scalar
        return loss