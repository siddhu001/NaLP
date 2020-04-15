import torch

class query2box_loss(torch.nn.Module):
    def __init__(self, margin=24.0):
        super(query2box_loss, self).__init__()
        self.margin = margin

    def forward(self, positive, negative_1):
        losses = -1 * torch.log(torch.sigmoid(self.margin + positive)) \
                 - torch.mean(torch.log(torch.sigmoid(-1 * negative_1 - self.margin)), dim=-1)
        return losses.mean()

class softmax_loss(torch.nn.Module):
    def __init__(self):
        super(softmax_loss, self).__init__()
    def forward(self, positive, negative_1):
        max_den_e1 = negative_1.max(dim=1, keepdim=True)[0].detach()
        den_e1 = (negative_1-max_den_e1).exp().sum(dim=-1, keepdim=True)
        losses = ((positive-max_den_e1) - den_e1.log())
        return -losses.mean()


class logistic_loss(torch.nn.Module):
    def __init__(self):
        super(logistic_loss, self).__init__()
    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.ones(1, positive.shape[1]+negative_1.shape[1]).cuda()
        truth[0, 0] = -1
        truth = -truth
        truth = torch.autograd.Variable(truth, requires_grad=False)
        x = torch.log(1+torch.exp(-scores*truth))
        total = x.sum()
        return total/((positive.shape[1]+negative_1.shape[1])*positive.shape[0])

class max_margin_loss(torch.nn.Module):
    def __init__(self):
        super(max_margin_loss, self).__init__()
    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        margin = 1.0 - (scores[:,0][:,None] - scores[:,1:])
        zeros = torch.zeros(margin.shape).cuda()
        zeros = torch.autograd.Variable(zeros, requires_grad=False)
        #cost = torch.min(torch.cat([ones, margin],dim=-1), dim=-1)
        losses = torch.max(margin,zeros)
        return losses.sum()/margin.numel()

class crossentropy_loss(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss, self).__init__()
        self.name = "crossentropy_loss"
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, truth, scores):
        truth = truth.view(truth.shape[0])
        losses = self.loss(scores, truth)
        return losses

class binarycrossentropy_loss(torch.nn.Module):
    def __init__(self):
        super(binarycrossentropy_loss, self).__init__()
        self.name = "binarycrossentropy_loss"
        self.loss = torch.nn.BCELoss(reduction='mean')

    def forward(self, positive, negative):
        target_ones = torch.ones_like(positive)
        target_zeros = torch.zeros_like(negative)
        losses = self.loss(torch.cat([positive, negative],dim=-1), torch.cat([target_ones, target_zeros], dim=-1))
        return losses
