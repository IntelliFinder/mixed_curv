import torch

def scalar_product_hyperbolic(x: torch.Tensor, y: torch.Tensor):
    return -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)

def projection_sphere(base_point: torch.Tensor, v: torch.Tensor):
    return v - torch.sum(v * base_point, dim=-1, keepdim=True) * base_point

def projection_hyperboloid(base_point: torch.Tensor, v: torch.Tensor):
    return v + scalar_product_hyperbolic(v, base_point).unsqueeze(-1) * base_point

def project_to_tangent_space(x, grad, manifold):
    if manifold == 'spherical':
        return projection_sphere(x, grad)
    elif manifold == 'hyperbolic':
        return projection_hyperboloid(x, grad)
    else:
        return grad

def exponential_map(x, v, manifold):
    if manifold == 'spherical':
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        return x * torch.cos(v_norm) + v * torch.sin(v_norm) / torch.clamp(v_norm, min=1e-8)
    elif manifold == 'hyperbolic':
        v_norm = torch.sqrt(torch.clamp(scalar_product_hyperbolic(v, v), min=1e-8)).unsqueeze(-1)
        return x * torch.cosh(v_norm) + v * torch.sinh(v_norm) / v_norm
    else:
        return x + v

def clip_hyperbolic(x, max_norm=0.9):
    with torch.no_grad():
        spatial_norm = torch.norm(x[:, 1:], dim=1, keepdim=True)
        scale = torch.min(
            torch.ones_like(spatial_norm),
            max_norm / torch.clamp(spatial_norm, min=1e-8)
        )
        x_clipped = x.clone()
        x_clipped[:, 1:] = x[:, 1:] * scale
        x_clipped[:, 0] = torch.sqrt(1 + torch.sum(x_clipped[:, 1:]**2, dim=1))
    return x_clipped
