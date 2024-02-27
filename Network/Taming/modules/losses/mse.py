import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


class VQWithMSELoss(nn.Module):
    def __init__(self, codebook_weight):
        self.codebook_weight = codebook_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, codebook_loss, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {
            "{}/total_loss".format(split): loss.clone().detach().mean(),
            "{}/quant_loss".format(split): codebook_loss.detach().mean(),
            "{}/nll_loss".format(split): nll_loss.detach().mean(),
            "{}/rec_loss".format(split): rec_loss.detach().mean(),
        }

        return loss, log
