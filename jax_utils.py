import jax
import torch
from jax import numpy as jnp


def pt2jax_array(tensor: torch.Tensor) -> jax.Array:
    np_array = tensor.cpu().numpy()
    jax_array = jnp.array(np_array)

    return jax_array
