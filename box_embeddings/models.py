import torch
import utils
import torch.nn.functional as F

class query2box(torch.nn.Module):

    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, alpha=0.2, norm=1, reg=2, init_scale=1e-3):

        super(query2box, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E_center = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_center = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_offset = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_center.weight.data, 0, init_scale)
        torch.nn.init.normal_(self.R_center.weight.data, 0, init_scale)
        torch.nn.init.uniform_(self.R_offset.weight.data, 0, init_scale*2)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.alpha = alpha
        self.norm = norm

    def forward(self, s, r, o):

        #s_center = self.E_center(s) if s is not None else self.E.weight
        s_center = self.E_center(s)
        r_center = self.R_center(r)
        r_offset = self.R_offset(r)
        o_center = self.E_center(o) if o is not None else self.E_center.weight
        #o_center = self.E_center(o)

        query_center = s_center + r_center
        query_offset = r_offset
        query_max = query_center + F.relu(query_offset)
        query_min = query_center - F.relu(query_offset)


        if not (s_center.shape == r_center.shape == o_center.shape):
            query_max = query_max.unsqueeze(1)
            query_min = query_min.unsqueeze(1)
            query_center = query_center.unsqueeze(1)

        zeros = torch.zeros_like(query_min)

        dist_out = torch.max(o_center - query_max, zeros) + torch.max(query_min - o_center, zeros)
        dist_out = torch.norm(dist_out, p=self.norm, dim=-1)

        dist_in = query_center - torch.min(query_max, torch.max(query_min, o_center))
        dist_in = torch.norm(dist_in, p=self.norm, dim=-1)

        return -(dist_out + self.alpha*dist_in)

    def regularizer(self, s, r, o):

        s_center = self.E_center(s)
        r_center = self.R_center(r)
        r_offset = self.R_offset(r)
        o_center = self.E_center(o)
        if self.reg == 2:
            return (s_center**2 + r_center**2 + r_offset**2 + o_center**2).sum()
        elif self.reg == 3:
            return (s_center.abs()**3 + r_center.abs()**3 + r_offset.abs()**3 + o_center.abs()**3).sum()
        else:
            print("Unknown reg for query2box model")
            assert(False)

class TransE(torch.nn.Module):

    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, alpha=0.2, norm=1, reg=2, init_scale=1e-3):

        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E_center = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_center = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_center.weight.data, 0, init_scale)
        torch.nn.init.normal_(self.R_center.weight.data, 0, init_scale)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.alpha = alpha
        self.norm = norm

    def forward(self, s, r, o):

        #s_center = self.E_center(s) if s is not None else self.E.weight
        s_center = self.E_center(s)
        r_center = self.R_center(r)
        o_center = self.E_center(o) if o is not None else self.E_center.weight
        #o_center = self.E_center(o)

        query_center = s_center + r_center
        if not (s_center.shape == r_center.shape == o_center.shape):
            query_center = query_center.unsqueeze(1)

        return torch.sum(-torch.abs(query_center - o_center), -1)

    def regularizer(self, s, r, o):

        s_center = self.E_center(s)
        r_center = self.R_center(r)
        o_center = self.E_center(o)
        if self.reg == 2:
            return (s_center**2 + r_center**2 + o_center**2).sum()
        elif self.reg == 3:
            return (s_center.abs()**3 + r_center.abs()**3 + o_center.abs()**3).sum()
        else:
            print("Unknown reg for query2box model")
            assert(False)
