'''
File from https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py
'''
from typing import List, Tuple   

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.autograd import Function


from common.layers import MLP

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        # indices = vq(inputs, codebook)
        y = torch.softmax(torch.matmul(inputs, codebook.transpose(0,1)), dim=1) 
        # (batch, K)
        _, indices = torch.max(y, dim=1)

        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


class VQEmbedding(nn.Module):
    def __init__(self, K: int, D: int):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x: torch.FloatTensor):
        logits = torch.matmul(z_e_x, self.embedding.weight.transpose(0,1))
        y = torch.softmax(logits, dim=1) 
        # (batch, K)
        _, indices = torch.max(y, dim=1)
        return logits, indices 

    def straight_through(self, z_e_x):
        z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach())
      
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar = z_q_x_bar_flatten.view_as(z_e_x)
        
        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

class EventVQVAE(nn.Module):
    '''
    This class follows the descriptions in Semi-supervised New Event Type Induction and Event Detection paper.
    '''
    def __init__(self, input_dim: int, dim: int, known_types: int, unknown_types: int, layers_n: int=1, 
        use_vae: bool=False, vae_dim: int=1024) -> None:

        super().__init__()
        self.known_types = known_types 
        self.unknown_types = unknown_types 
        self.types_n = known_types + unknown_types

        # f_c 
        self.encoder = MLP(input_dim, dim, dim, norm=False, layers_n=layers_n)
        self.codebook = VQEmbedding(K=self.types_n, D=dim)
        self.decoder = nn.Linear(in_features=dim, out_features=input_dim)

        self.use_vae = use_vae 
        if use_vae: 
            # f_e 
            self.vae_encoder = MLP(input_dim, vae_dim, 2 * vae_dim, norm=False, layers_n=2)
            # f_r 
            self.vae_decoder = nn.Linear(self.types_n + vae_dim, input_dim) 

        self.apply(weights_init)

    def encode(self, x: torch.FloatTensor) ->Tuple[torch.FloatTensor, torch.LongTensor]:
        '''
        :param x: (batch, input_dim)
        '''
        z_e_x = self.encoder(x) # (batch, dim)
        logits, indexes  = self.codebook(z_e_x) # (batch)
        return logits, indexes
    
    def decode(self, latents: torch.LongTensor) -> torch.FloatTensor:
        z_q_x = self.codebook.embedding(latents)  
        x_tilde = self.decoder(z_q_x)
        return x_tilde 

    def forward(self, x: torch.FloatTensor, known_mask: torch.LongTensor, labels=None):
        z_e_x = self.encoder(x)
        logits, _ = self.codebook(z_e_x)
        # z_q_x is backpropped to embedding 
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        if self.use_vae: 
            mu, logvar = self.vae_encoder(x).chunk(2, dim=1)
            q_z_x = Normal(mu, logvar.mul(.5).exp())
            p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
            kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
            sampled_z_x = q_z_x.rsample()

            y_tilde = torch.softmax(logits, dim=1)
            y = y_tilde.clone() 
            y[known_mask] = F.one_hot(labels[known_mask], num_classes=self.types_n).float() # replace known types with gold labels 
            x_tilde = self.vae_decoder(torch.concat([sampled_z_x, y], dim=1))
        else:
            x_tilde = self.decoder(z_q_x_st)
            kl_div = 0.0
            # for a pure VQ-VQE, the prior is an uniform distribution and thus the kl term is constant 

        return x_tilde, z_e_x, z_q_x, logits, kl_div 
