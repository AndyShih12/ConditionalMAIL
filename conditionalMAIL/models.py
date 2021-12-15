import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim):
        super(Net, self).__init__()

        self.input_dim, self.output_dim, self.latent_dim = input_dim, output_dim, latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Functional Train Network
# len(dimensions) >= 2
class FTNet(nn.Module):
    def __init__(self, dimensions, rank, list_of_nets=None, latents=100, kernel='lin', output_dim=1):
        super(FTNet, self).__init__()

        self.rank = [rank for _ in range(len(dimensions) + 1)]
        self.rank[0] = 1
        self.rank[-1] = output_dim
        print("TT RANK: ", self.rank)

        if list_of_nets is None:
            list_of_nets = self.create_nets(dimensions, self.rank, latents=latents)

        assert(len(list_of_nets) == len(dimensions))
        self.list_of_nets = list_of_nets
        self.dimensions = dimensions
        self.kernel = kernel

        self.net = nn.ModuleList(list_of_nets)

    def create_nets(self, dimensions, rank, latents):
        list_of_nets = [None for _ in dimensions]
        for i, dim in enumerate(dimensions):
            output_dim = rank[i] * rank[i+1]
            #if i == 0 or i == len(dimensions)-1: output_dim = rank
            list_of_nets[i] = Net(dim, output_dim, latents)
        return list_of_nets

    def forward(self, x):
        batch = x.size()[:-1]
        ret = th.ones( *batch, 1, 1 ).to(x.device)
        for i, nt in enumerate(self.list_of_nets):
            dim = self.dimensions[i]
            cur_x, x = th.split(x, split_size_or_sections=[dim, x.size(-1)-dim], dim=-1)
            out = nt(cur_x)

            if i == 0:
                out = F.softmax( out, dim=-1 )

            row, col = self.rank[i], self.rank[i+1]
            #if i == 0: row = 1
            #if i == len(self.list_of_nets) - 1: col = 1
            out = out.reshape( *batch, row, col )

            if self.kernel == 'lin':
                ret = th.matmul(ret, out)
            elif self.kernel == 'rbf':
                ret = self.kernelMatmul(ret, out)
            else:
                assert(False)

        ret = ret.squeeze(-2)
        return ret

    def kernelMatmul(self, m1, m2):
        """
            m1: [batch1,..., batchk, n, p]
            m2: [batch1,..., batchk, p, m]
            output: [batch1,..., batchk, n, m]
        """
        *batch, n, p = m1.size()
        *batch, p, m = m2.size()
        ret = th.zeros( *batch, n, m ).to(m1.device)

        def k(a, b):
            return th.norm(a - b, dim=-1) # can replace with other kernels
            # return th.sum(a * b, dim=-1) # standard inner product

        m2t = m2.transpose(-1,-2) # [batch, m, p]
        for i in range(n):
            retrow = th.narrow(ret, dim=len(batch), start=i, length=1).squeeze(dim=len(batch)) # shares same underlying storage as ret
            m1row = th.narrow(m1, dim=len(batch), start=i, length=1)
            retrow[:] = k(m1row, m2t)   # the [:] is important, otherwise ret is not modified
        return ret


class ModularNet(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, num_partners, continuous=False):
        super(ModularNet, self).__init__()

        self.input_dim, self.output_dim, self.latent_dim = input_dim, output_dim, latent_dim
        self.num_partners = num_partners

        self.main_net = Net(input_dim=self.input_dim, output_dim=self.latent_dim + self.output_dim, latent_dim=self.latent_dim)
        self.partner_nets = nn.ModuleList([
            Net(input_dim=self.latent_dim, output_dim=self.output_dim, latent_dim=latent_dim) for _ in range(self.num_partners)
        ])

        self.continuous = continuous

    def forward(self, x, partner_idx_vec):
        batch = x.size()[:-1]
        batch_prod = np.prod( batch )
        x = x.reshape( batch_prod, self.input_dim )
        partner_idx_vec = partner_idx_vec.reshape( batch_prod )

        latents_and_output = self.main_net(x)
        latents, output_main = th.split(latents_and_output, [self.latent_dim, self.output_dim], dim=-1) # (batch_prod, L), (batch_prod, O)
        output_all_partners = th.stack([self.partner_nets[idx](latents) for idx in range(self.num_partners)]).to(output_main.device) # (P, batch_prod, O)

        if self.continuous:
            output = output_main + output_all_partners[ partner_idx_vec, th.arange(batch_prod).long() ] # (batch_prod, O)
            output = output.reshape( *batch, self.output_dim ) # ( *batch, O )
            output = output / 2

            # for computing marginal regularization
            wass_dist = th.mean(th.sum( th.abs(output_all_partners.mean(dim=0)), dim=1)) # float

            return output, wass_dist # ( *batch, O ) and float

        output = output_main + output_all_partners[ partner_idx_vec, th.arange(batch_prod).long() ] # (batch_prod, O)
        output = output.reshape( *batch, self.output_dim ) # ( *batch, O )

        # for computing marginal regularization
        composed_logits = (output_main.unsqueeze(0) + output_all_partners) # (P, batch_prod, O)
        composed_prob = th.exp(composed_logits - composed_logits.logsumexp(dim=-1, keepdim=True)) # (P, batch_prod, O)
        marginal_prob = composed_prob.mean(dim=0) # (batch_prod, O)
        main_prob = th.exp(output_main - output_main.logsumexp(dim=-1, keepdim=True))  # (batch_prod, O)
        wass_dist = th.mean(th.sum( th.abs(main_prob - marginal_prob), dim=1)) # float

        return output, wass_dist # ( *batch, O ) and float
