from functools import partial
from typing import Optional, List

import torch
from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss
from torch import binary_cross_entropy_with_logits, nn, conv2d
from torch.nn import Module


class TverskyBCELoss(Module):
    def __init__(
            self,
            bce_alpha: Optional[float] = None,
            bce_gamma: Optional[float] = 2.0,
            tversky_alpha: Optional[float] = 0.5,
            tversky_beta: Optional[float] = 0.5,
            tversky_gamma: Optional[float] = 0.7,
            smooth_factor: Optional[float] = None,
            ignore_index: Optional[int] = None,
            normalized: bool = False,
            reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss plus tversky

        Args:
            bce_alpha: Prior probability of having positive value in target.
            bce_gamma: Power factor for dampening weight (focal strength).
            tversky_alpha: Weight constant that penalize model for FPs (False Positives)
            tversky_beta: Weight constant that penalize model for FNs (False Negatives)
            tversky_gamma: Constant that squares the error function. Defaults to ``1.0``
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)


        """
        super().__init__()
        self.smooth_factor = smooth_factor

        self.tversky = TverskyLoss(
            "binary", alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma, eps=1e-6
        )
        self.bce = FocalLoss(
            "binary", bce_alpha, bce_gamma, ignore_index, "none", normalized, reduced_threshold
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.smooth_factor is not None:
            soft_targets = y_true * (1 - self.smooth_factor) + 0.5 * self.smooth_factor
        else:
            soft_targets = y_true

        bce = self.bce(y_pred, soft_targets).view(y_true.shape)
        if weights is not None:
            bce = bce * weights

        bce = bce.mean(1).sum()

        tasks = []
        for i in range(y_pred.shape[1]):
            tasks.append(self.tversky(y_pred[:, i], y_true[:, i]))

        tversky = torch.stack(tasks).sum()

        return bce + tversky


class WeightedTverskyLoss(TverskyLoss):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights is None:
            weighted_y_pred = y_pred
        else:
            weighted_y_pred = y_pred * weights
        tasks = []
        for i in range(y_pred.shape[1]):
            tasks.append(super().forward(weighted_y_pred[:, i], y_true[:, i]))

        return torch.stack(tasks).sum()


class WeightedTverskyLossv2(TverskyLoss):
    """Tversky loss for image segmentation task.
    Where FP and FN is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes input is raw logits
        smooth:
        ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``
        label_kernel: Kernel size to determine fill pixels

    Return:
        loss: torch.Tensor

    """

    def __init__(
            self,
            mode: str,
            classes: List[int] = None,
            log_loss: bool = False,
            from_logits: bool = True,
            smooth: float = 0.0,
            ignore_index: Optional[int] = None,
            eps: float = 1e-7,
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = 1.0,
            label_kernel: [tuple, int] = 25,
    ):
        super().__init__(mode, classes, log_loss, from_logits, smooth, ignore_index, eps, alpha, beta, gamma)
        self.label_kernel = (label_kernel, label_kernel) if isinstance(label_kernel, int) else label_kernel

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        is_w = weights is not None
        if is_w:
            # expects that the weight is binary
            weights -= weights.min()
            weight_scale = weights.max()
            weights /= weight_scale

            weights = weights.bool()

        tasks = []
        for i in range(y_pred.shape[1]):
            tasks.append(super().forward(y_pred[:, i], y_true[:, i]))
            if is_w:
                w_ = torch.zeros_like(weights)
                w_[:, i] = weights[:, i]
                tasks.append(
                    weight_scale * super().forward(-y_pred[w_].unsqueeze(-1), 1 - y_true[w_].unsqueeze(-1))
                )

        return torch.stack(tasks).sum()


class WeightedFocalLoss(FocalLoss):

    def __init__(
            self,
            gamma: Optional[float] = 2.0,
            smooth_factor: Optional[float] = None,
            normalized: bool = False,
    ):
        """Compute Focal loss

        Args:
            gamma: Power factor for dampening weight (focal strength).
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        self.smooth_factor = smooth_factor
        self.normalized = normalized
        self.gamma = gamma
        super().__init__("binary", None, gamma, None, "none", normalized, None)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.smooth_factor is not None:
            soft_targets = y_true * (1 - self.smooth_factor) + 0.5 * self.smooth_factor
        else:
            soft_targets = y_true.float()

        logpt = binary_cross_entropy_with_logits(y_pred, soft_targets, reduction=0)

        if weights is not None:
            logpt = logpt * weights

        if self.gamma is not None:
            pt = torch.exp(-logpt)

            # compute the loss
            focal_term = (1.0 - pt).pow(self.gamma)

            loss = focal_term * logpt

            if self.normalized:
                norm_factor = focal_term.sum().clamp_min(1e-6)
                loss /= norm_factor
        else:
            loss = logpt

        return loss.mean(1).mean()


class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Define Sobel kernels for x and y directions
        self.sobel_x = torch.FloatTensor([[1, 0, -1],
                                          [2, 0, -2],
                                          [1, 0, -1]])
        self.sobel_y = torch.FloatTensor([[1, 2, 1],
                                          [0, 0, 0],
                                          [-1, -2, -1]])

        # Reshape kernels to fit PyTorch convolution format
        self.sobel_x = self.sobel_x.view(1, 1, 3, 3)
        self.sobel_y = self.sobel_y.view(1, 1, 3, 3)

        # Convert Sobel kernels to nn.Parameter so that they can be optimized
        self.sobel_x = nn.Parameter(self.sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(self.sobel_y, requires_grad=False)

    def forward(self, input):
        # Compute gradient magnitude along x and y directions using convolution
        gradient_x = conv2d(input, self.sobel_x, padding=1)
        gradient_y = conv2d(input, self.sobel_y, padding=1)

        # Compute gradient magnitude using Pythagorean theorem
        gradient_magnitude = torch.sqrt(gradient_x.pow(2) + gradient_y.pow(2))

        return gradient_magnitude


LOSS_FN = {
    "bce": partial(WeightedFocalLoss, gamma=None, normalized=False, smooth_factor=None),
    "focalbce": partial(WeightedFocalLoss, gamma=2, normalized=False, smooth_factor=None),
    "softfocalbce": partial(WeightedFocalLoss, gamma=2, normalized=False, smooth_factor=0.1),
    "softbce": partial(WeightedFocalLoss, gamma=1, smooth_factor=0.1),
    "focalsoftdice": partial(WeightedTverskyLoss, mode="binary", alpha=0.5, beta=0.5, gamma=0.75, smooth=0.1),
    "tversky": partial(WeightedTverskyLoss, mode="binary", alpha=0.3, beta=0.7),
    "softtversky": partial(WeightedTverskyLoss, mode="binary", alpha=0.3, beta=0.7, smooth=0.1),
    "tversky73": partial(WeightedTverskyLoss, mode="binary", alpha=0.7, beta=0.3),
    "softtversky73": partial(WeightedTverskyLoss, mode="binary", alpha=0.7, beta=0.3, smooth=0.1),
    "focaltversky": partial(WeightedTverskyLoss, mode="binary", alpha=0.3, beta=0.7, gamma=0.75),
    "focalsofttversky": partial(WeightedTverskyLoss, mode="binary", alpha=0.3, beta=0.7, smooth=0.1,
                                gamma=0.75),
    "focaltversky73": partial(WeightedTverskyLoss, mode="binary", alpha=0.7, beta=0.3, gamma=0.75),
    "focalsofttversky73": partial(WeightedTverskyLoss, mode="binary", alpha=0.7, beta=0.3, smooth=0.1,
                                  gamma=0.75),
    "focalsofttversky82": partial(WeightedTverskyLoss, mode="binary", alpha=0.8, beta=0.2, smooth=0.1,
                                  gamma=0.75),
    "focalsofttversky46": partial(WeightedTverskyLoss, mode="binary", alpha=0.4, beta=0.6, smooth=0.1,
                                  gamma=0.75),
    "focalsofttversky91": partial(WeightedTverskyLoss, mode="binary", alpha=0.9, beta=0.1, smooth=0.1,
                                  gamma=0.75),
    "focalsofttverskybce73": partial(
        TverskyBCELoss, tversky_alpha=0.7, tversky_beta=0.3, tversky_gamma=0.75,
        smooth_factor=0.1, bce_gamma=2, normalized=True
    ),
    "softtverskybce73": partial(
        TverskyBCELoss, tversky_alpha=0.7, tversky_beta=0.3, tversky_gamma=1,
        smooth_factor=0.1, bce_gamma=1, normalized=False
    ),
    "focaltverskybce73": partial(
        TverskyBCELoss, tversky_alpha=0.7, tversky_beta=0.3, tversky_gamma=0.75,
        bce_gamma=2, normalized=True
    ),
    "tverskybce73": partial(
        TverskyBCELoss, tversky_alpha=0.7, tversky_beta=0.3, tversky_gamma=1,
        bce_gamma=1, normalized=False
    ),
    "tversky73v2": partial(WeightedTverskyLossv2, mode="binary", alpha=0.7, beta=0.3),
    "focaltversky73v2": partial(WeightedTverskyLossv2, mode="binary", alpha=0.7, beta=0.3, gamma=0.75),
    "softtversky73v2": partial(WeightedTverskyLossv2, mode="binary", alpha=0.7, beta=0.3, smooth=0.1),
    "focalsofttversky73v2": partial(
        WeightedTverskyLossv2, mode="binary", alpha=0.7, beta=0.3, smooth=0.1, gamma=0.75
    ),

}

if __name__ == '__main__':
    torch.random.manual_seed(42)
    y_true = (torch.rand((100, 1, 1)) > 0.8).float()
    y_pred = torch.ones_like(y_true)

    weights = torch.ones_like(y_true)
    weights[40:80] = 100

    tv = WeightedTverskyLoss("binary", alpha=0.7, beta=0.3)
    tv2 = WeightedTverskyLossv2("binary", alpha=0.7, beta=0.3)
    tvbce = TverskyBCELoss(tversky_alpha=0.7, tversky_beta=0.3, normalized=True)
    bce = WeightedFocalLoss("binary", normalized=True)

    print(tv(y_pred, y_true, weights).item())
    print(tv2(y_pred, y_true, weights).item())
    print(tvbce(y_pred, y_true, weights).item())
    print(bce(y_pred, y_true, weights).item())
