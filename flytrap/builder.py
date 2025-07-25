from typing import List

import torch
from mmcv.transforms import TRANSFORMS
from mmengine import Registry


def build_optimizer(cfg, params, **kwargs):
    optimizer_type = cfg['type']
    optimizer_cfg = cfg['params']
    optimizer_cls = OPTIMIZER.get(optimizer_type)
    return optimizer_cls([params], **optimizer_cfg)


def build_scheduler(cfg, optimizer, **kwargs):
    scheduler_type = cfg['type']
    scheduler_cfg = cfg['params']
    scheduler_cls = SCHEDULER.get(scheduler_type)
    return scheduler_cls(optimizer, **scheduler_cfg)


class LossWrapper:
    """Wrapper for loss."""

    def __init__(self, loss_fns: List):
        self.loss_fn = []
        for loss_fn in loss_fns:
            if not callable(loss_fn):
                assert getattr(loss_fn, 'type') is not None, f"{loss_fn} is not callable or a class or config"
                loss_fn = LOSS.build(loss_fn)
            self.loss_fn.append(loss_fn)

        self.loss_dict = {}

    def __call__(self, inputs: dict):
        loss = 0
        for loss_fn in self.loss_fn:
            # attention: avoid in_place operation
            single_loss = loss_fn(inputs)
            loss = loss + single_loss
            self.loss_dict[loss_fn.__class__.__name__] = single_loss.item()
        return loss

    def parse_loss(self):
        return self.loss_dict


class AdaptiveLossWrapper:
    """
    Advanced loss wrapper with automatic weight adjustment.
    
    Supports multiple strategies for dynamic weight adjustment:
    - Real-time loss value adjustment 
    - Uncertainty-based weighting
    - Coefficient of Variations (CV-Weighting)
    
    References:
    - GradNorm: https://arxiv.org/pdf/1711.02257
    - Multi-Loss Weighting with Coefficient of Variations: https://arxiv.org/pdf/2009.01717v1.pdf
    - Strategies for Balancing Multiple Loss Functions: https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0
    """
    
    def __init__(self, loss_fns: List, strategy: str = 'realtime', **kwargs):
        """
        Args:
            loss_fns: List of loss functions
            strategy: Weight adjustment strategy. Options:
                - 'realtime': Real-time loss value adjustment
                - 'uncertainty': Uncertainty-based weighting  
                - 'cv': Coefficient of Variations weighting
                - 'gradnorm': Gradient normalization (requires shared_layer)
            **kwargs: Strategy-specific parameters
        """
        self.loss_fn = []
        for loss_fn in loss_fns:
            if not callable(loss_fn):
                assert getattr(loss_fn, 'type') is not None, f"{loss_fn} is not callable or a class or config"
                loss_fn = LOSS.build(loss_fn)
            self.loss_fn.append(loss_fn)
            
        self.num_losses = len(self.loss_fn)
        self.strategy = strategy
        self.loss_dict = {}
        
        # Initialize weights
        self.weights = torch.ones(self.num_losses, requires_grad=False)
        
        # Strategy-specific initialization
        if strategy == 'uncertainty':
            # Learnable uncertainty parameters
            self.log_vars = torch.nn.Parameter(torch.zeros(self.num_losses))
        elif strategy == 'cv':
            # For CV-weighting
            self.loss_history = [[] for _ in range(self.num_losses)]
            self.loss_means = torch.zeros(self.num_losses)
            self.loss_stds = torch.zeros(self.num_losses)
            self.warmup_steps = kwargs.get('warmup_steps', 10)
            self.step_count = 0
        elif strategy == 'gradnorm':
            # For GradNorm
            self.alpha = kwargs.get('alpha', 1.5)
            self.shared_layer = kwargs.get('shared_layer', None)
            self.initial_losses = None
            
        self.momentum = kwargs.get('momentum', 0.1)
        
    def _update_weights_realtime(self, losses):
        """Real-time loss value adjustment: w_i = 1/L_i(t)"""
        # Avoid division by zero
        eps = 1e-8
        loss_values = torch.stack([loss.detach() for loss in losses])
        new_weights = 1.0 / (loss_values + eps)
        
        # Normalize weights
        new_weights = new_weights / new_weights.sum() * self.num_losses
        
        # Apply momentum
        self.weights = (1 - self.momentum) * self.weights + self.momentum * new_weights
        
    def _update_weights_uncertainty(self, losses):
        """
        Uncertainty-based weighting: w_i = 1/(2*sigma_i^2), 
        with additional regularization term log(1 + sigma_i^2)
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            # Convert log_var to variance
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i] + torch.log(1 + torch.exp(self.log_vars[i]))
            weighted_losses.append(weighted_loss)
        return weighted_losses
        
    def _update_weights_cv(self, losses):
        """Coefficient of Variations weighting: w_i = (std_i/mean_i) / current_loss_ratio_i"""
        self.step_count += 1
        
        # Update loss history
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss.detach().item())
            
        # Need warmup period to establish statistics
        if self.step_count < self.warmup_steps:
            return  # Use equal weights during warmup
            
        # Calculate statistics using Welford's online algorithm
        for i in range(self.num_losses):
            history = torch.tensor(self.loss_history[i])
            self.loss_means[i] = history.mean()
            self.loss_stds[i] = history.std()
            
        # Calculate loss ratios and CV weights
        new_weights = torch.zeros(self.num_losses)
        for i, loss in enumerate(losses):
            if self.loss_means[i] > 1e-8:
                loss_ratio = loss.detach() / self.loss_means[i]
                cv = self.loss_stds[i] / self.loss_means[i] if self.loss_means[i] > 1e-8 else 1.0
                new_weights[i] = cv / (loss_ratio + 1e-8)
            else:
                new_weights[i] = 1.0
                
        # Normalize weights
        if new_weights.sum() > 1e-8:
            new_weights = new_weights / new_weights.sum() * self.num_losses
            # Apply momentum
            self.weights = (1 - self.momentum) * self.weights + self.momentum * new_weights
            
    def _update_weights_gradnorm(self, losses, shared_layer_params):
        """
        GradNorm: Balance gradient norms across tasks
        Requires shared layer parameters for gradient computation
        """
        if shared_layer_params is None:
            print("Warning: GradNorm requires shared_layer parameters. Falling back to equal weights.")
            return
            
        # Initialize with first batch losses
        if self.initial_losses is None:
            self.initial_losses = torch.stack([loss.detach() for loss in losses])
            
        # Calculate gradient norms for each loss
        grad_norms = []
        for i, loss in enumerate(losses):
            # Compute gradients w.r.t shared layer
            grads = torch.autograd.grad(
                outputs=self.weights[i] * loss,
                inputs=shared_layer_params,
                retain_graph=True,
                create_graph=True
            )[0]
            grad_norm = torch.norm(grads, p=2)
            grad_norms.append(grad_norm)
            
        grad_norms = torch.stack(grad_norms)
        
        # Calculate average gradient norm
        avg_grad_norm = grad_norms.mean().detach()
        
        # Calculate relative training rates
        current_losses = torch.stack([loss.detach() for loss in losses])
        loss_ratios = current_losses / self.initial_losses
        relative_rates = loss_ratios / loss_ratios.mean()
        
        # Target gradient norms
        target_grad_norms = avg_grad_norm * (relative_rates ** self.alpha)
        
        # Update weights to minimize L1 loss between actual and target gradient norms
        grad_loss = torch.abs(grad_norms - target_grad_norms).sum()
        
        # Update weights based on gradient of grad_loss
        weight_grads = torch.autograd.grad(grad_loss, self.weights, retain_graph=True)[0]
        
        # Simple gradient descent on weights
        with torch.no_grad():
            self.weights -= 0.01 * weight_grads
            # Re-normalize weights
            self.weights = torch.clamp(self.weights, min=0.01)  # Prevent negative weights
            self.weights = self.weights / self.weights.sum() * self.num_losses

    def __call__(self, inputs: dict, shared_layer_params=None):
        """
        Forward pass with automatic weight adjustment
        
        Args:
            inputs: Input dictionary for loss functions
            shared_layer_params: Parameters of shared layer (needed for GradNorm)
        """
        # Compute individual losses
        losses = []
        for loss_fn in self.loss_fn:
            single_loss = loss_fn(inputs)
            losses.append(single_loss)
            self.loss_dict[loss_fn.__class__.__name__] = single_loss.item()
            
        # Update weights based on strategy
        if self.strategy == 'realtime':
            self._update_weights_realtime(losses)
        elif self.strategy == 'uncertainty':
            # For uncertainty weighting, return the weighted losses directly
            weighted_losses = self._update_weights_uncertainty(losses)
            total_loss = sum(weighted_losses)
            return total_loss
        elif self.strategy == 'cv':
            self._update_weights_cv(losses)
        elif self.strategy == 'gradnorm':
            self._update_weights_gradnorm(losses, shared_layer_params)
            
        # Compute weighted total loss
        total_loss = sum(w * loss for w, loss in zip(self.weights, losses))
        
        return total_loss
        
    def get_current_weights(self):
        """Get current loss weights"""
        return self.weights.clone()
        
    def parse_loss(self):
        """Parse loss dictionary with current weights"""
        loss_info = self.loss_dict.copy()
        loss_info['weights'] = {f'weight_{i}': w.item() for i, w in enumerate(self.weights)}
        return loss_info


MODEL = Registry('model')
APPLYER = Registry('applyer')
PIPELINES = TRANSFORMS
OPTIMIZER = Registry('optimizer', build_func=build_optimizer)
SCHEDULER = Registry('scheduler', build_func=build_scheduler)
METRICS = Registry('metrics')
ENGINE = Registry('engine')
LOSS = Registry('loss')
ALARM = Registry('alarm')
