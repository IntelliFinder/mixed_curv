import torch
from torch.optim import Optimizer
from utils.manifold_utils import project_to_tangent_space, exponential_map, clip_hyperbolic

class RiemannianSGDProductManifoldOptimizer(Optimizer):
    def __init__(self, params, lr=1e-2):
        defaults = dict(lr=lr)
        super(RiemannianSGDProductManifoldOptimizer, self).__init__(params, defaults)

        self.param_manifolds = {}
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'manifold'):
                    self.param_manifolds[id(p)] = p.manifold
                else:
                    raise ValueError(f"Parameter {p} does not have a 'manifold' attribute")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                manifold = self.param_manifolds[id(p)]

                if manifold == 'spherical' or manifold == 'hyperbolic':
                    # Project gradient to tangent space
                    proj_grad = project_to_tangent_space(p.data, grad, manifold)

                    # Scale the projected gradient by the learning rate
                    scaled_proj_grad = -group['lr'] * proj_grad

                    # Apply exponential map to move back to the manifold
                    p.data = exponential_map(p.data, scaled_proj_grad, manifold)

                    # Normalize for numerical stability
                    if manifold == 'spherical':
                        p.data.div_(torch.clamp(torch.norm(p.data, dim=-1, keepdim=True), min=1e-8))
                    elif manifold == 'hyperbolic':
                        p.data = clip_hyperbolic(p.data, max_norm=5)
                elif manifold == 'euclidean':
                    # Standard Euclidean update
                    p.data.add_(-group['lr'] * grad)
