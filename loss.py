from typing import List
import torch
import torch.nn as nn

from loguru import logger


nINF = -100

class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

        logger.info(f"Initializing TwoWayLoss with Tp: {Tp} and Tn: {Tn}")

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
                torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()


def get_loss(type: str, num_concepts: int, num_samples:int, concept_counts: List[float], cbl_pos_weight:float, cbl_auto_weight: bool=False, tp: float = 4.,device="cuda"):
    if type == "bce":
        logger.info("Using BCE Loss for training CBL...")
        if cbl_auto_weight:
            logger.info(f"Using automatic weighting for positive examples with scale {cbl_pos_weight}")
            pos_count = torch.tensor(concept_counts).to(device)
            neg_count = num_samples - pos_count
            scale = (neg_count / pos_count) * cbl_pos_weight
            logger.info(f"scale mean: {scale.mean()}, scale std: {scale.std()}")
            logger.info(f"scale min: {scale.min()}, scale max: {scale.max()}")
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=scale).to(device)
        else:
            logger.info("Using fixed weighting for positive examples")
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cbl_pos_weight] * num_concepts)).to(device)
    elif type == "twoway":
        logger.info("Using TwoWay Loss for training CBL...")
        loss_fn = TwoWayLoss(Tp=tp)
    else:
        raise NotImplementedError(f"Loss {type} is not implemented")
    

    return loss_fn