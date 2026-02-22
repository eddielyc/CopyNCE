
from loguru import logger

from core.loss.base_loss import BaseLoss


class ScoreRankingLoss(BaseLoss):
    def __init__(self, cfg, model):
        super(ScoreRankingLoss, self).__init__(cfg, model)


        logger.info(self)

    def __repr__(self):
        return f"Build InfoNCE loss with " \
               f"weight: {self.weight}" \
               f"log_scale_init={self.log_scale_init}, " \
               f"gather={self.gather}."

    def calculate(self, outputs, inputs):
        pass
