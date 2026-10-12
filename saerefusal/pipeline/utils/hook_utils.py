
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []

        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn



def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(
    saved_activation,
    layer,
    vector: Float[Tensor, "d_model"],
    coeff: Float[Tensor, ""],
):
    def hook_fn(module, input):
        nonlocal vector

        # 取 activation
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        # 保证 device / dtype 一致
        vector_ = vector.to(activation)
        activation += coeff * vector_

        # -------------------------
        # 只在 prompt 阶段保存
        # -------------------------
        if activation.dim() == 3 and activation.shape[1] > 1:
            token_number = activation.size(1)

            # [1, seq_len, d_model] → [seq_len, d_model]
            act = activation.squeeze(0)
            if saved_activation is not None and layer is not None:
                # 平均 token
                saved_activation[layer] = (
                    act.sum(dim=0) / token_number
                ).detach()

        # 返回修改后的 activation
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn



def act_norm_get_activation_addition_output_hook(
    saved_activation,
    layer,
    vector: torch.Tensor,
    coeff: torch.Tensor,
):
    def hook_fn(module, input, output):
        
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        # 1️⃣ clone，避免 in-place + view 问题
        activation = activation.clone()

        # 2️⃣ 对齐 dtype & device（这是关键）
        vector_ = vector.to(dtype=activation.dtype, device=activation.device)
        
        
        if activation.shape[1] == 1:
            # decode 阶段
            avg_norm = activation.norm(dim=-1, keepdim=True)
            activation += coeff * avg_norm * vector_
        else:
            # prefill 阶段
            avg_norm = activation[:, 1:].norm(dim=-1, keepdim=True)
            activation[:, 1:] += coeff * avg_norm * vector_
        
        if activation.dim() == 3 and activation.shape[1] > 1:
            token_number = activation.size(1)
            # [1, seq_len, d_model] → [seq_len, d_model]
            act = activation.squeeze(0)
            # 平均 token
            saved_activation[layer] = act

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def save_post_act(saved_activation, layer, ):
    def hook_fn(module, input, output):

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output
        
        if activation.dim() == 3 and activation.shape[1] > 1:
            activation = activation.detach().cpu().squeeze(0)
            assert len(activation.shape) == 2
            saved_activation[layer] = activation
    
    return hook_fn


def build_save_prompt_activation_hook(model_base,
                is_baseline: bool,
                is_sae: bool,
                intervention_layer: int,
                steering_factor: float,
                direction):
    
    if model_base.model_name_or_path == 'google/gemma-3-4b-it' or model_base.model_name_or_path == 'google/gemma-3-12b-it':
        config = model_base.model.language_model.config
    else:
        config = model_base.model.config
    save_activation = {}

    if is_baseline:
        fwd_pre_hooks = []
        fwd_hooks = [
            (model_base.model_block_modules[l],
                save_post_act(saved_activation=save_activation, layer=l))
            for l in range(intervention_layer, config.num_hidden_layers)
        ]
    elif is_sae:
            fwd_pre_hooks = []
            fwd_hooks = [
                (model_base.model_block_modules[intervention_layer],
                        act_norm_get_activation_addition_output_hook(
                        saved_activation=save_activation,
                        layer=intervention_layer,
                        vector=direction,
                        coeff=steering_factor
                    ))
            ]
            fwd_hooks += [
                (model_base.model_block_modules[l],
                    save_post_act(saved_activation=save_activation, layer=l))
                for l in range(intervention_layer +1 , config.num_hidden_layers)
            ]
    else:
        raise NotImplementedError("Post-act steering not implemented")

    return fwd_pre_hooks, fwd_hooks, save_activation



def build_hooks(model_base,
                is_baseline: bool,
                is_sae: bool,
                intervention_layer: int,
                steering_factor: float,
                direction):

    if model_base.model_name_or_path == 'google/gemma-3-4b-it' or model_base.model_name_or_path == 'google/gemma-3-12b-it':
        config = model_base.model.language_model.config
    else:
        config = model_base.model.config
    save_activation = {}

    if is_baseline:
        fwd_pre_hooks = []
        fwd_hooks = [
            (model_base.model_block_modules[l],
                save_post_act(saved_activation=save_activation, layer=l))
            for l in range(config.num_hidden_layers)
        ]
    elif is_sae:
            fwd_pre_hooks = []
            fwd_hooks = [
                (model_base.model_block_modules[intervention_layer],
                        act_norm_get_activation_addition_output_hook(
                        saved_activation=save_activation,
                        layer=intervention_layer,
                        vector=direction,
                        coeff=steering_factor
                    ))
            ]
            fwd_hooks += [
                (model_base.model_block_modules[l],
                    save_post_act(saved_activation=save_activation, layer=l))
                for l in range(intervention_layer +1 , config.num_hidden_layers)
            ]
    else:
        raise NotImplementedError("Post-act steering not implemented")

    return fwd_pre_hooks, fwd_hooks, save_activation