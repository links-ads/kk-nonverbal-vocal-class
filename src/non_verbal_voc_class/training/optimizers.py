import torch.optim

# TODO: Create factory for optimizers
def create_optimizer(model, optimizer_config: object):
    """ Optimizer factory: make optimizer from model and hparams """
    
    optimizer_name = optimizer_config.optimizer
    optim_groups = list(filter(lambda p: p.requires_grad, model.parameters()))

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            optim_groups,
            lr = optimizer_config.lr,
            weight_decay = optimizer_config.weight_decay,
            amsgrad = optimizer_config.amsgrad,
            betas = optimizer_config.betas,
            eps = optimizer_config.eps,
            )
        
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr = optimizer_config.lr,
            weight_decay = optimizer_config.weight_decay,
            amsgrad = optimizer_config.amsgrad,
            betas = optimizer_config.betas,
            eps = optimizer_config.eps,
            )
    
    elif optimizer_name == 'radam':
        optimizer = torch.optim.RAdam(
            optim_groups,
            lr = optimizer_config.lr,
            weight_decay = optimizer_config.weight_decay,
            betas = optimizer_config.betas,
            eps = optimizer_config.eps,
            )
        
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            optim_groups,
            lr = optimizer_config.lr,
            weight_decay = optimizer_config.weight_decay,
            momentum = optimizer_config.momentum,
            dampening = optimizer_config.dampening,
            nesterov = optimizer_config.nesterov,
            )
        
    return optimizer

def get_layers(model):
    submodules = list(model.children())
    if len(submodules) == 0:
        return [model]
    else:
        res = []
        for module in submodules:
            res += get_layers(module)
        return res


def get_optim_groups(model, global_lr):
    optim_groups = []

    for param_type, params in model.named_parameters():  # this skips all layers without parameters (e.g. certain non-parametric fusion modules)
        if global_lr == 0.0:
            params.requires_grad = False

        optim_groups.append(
            {
                "params" : params, 
                "param_name" : param_type
            }
        )

    return optim_groups

def _add_custom_lr(optim_groups, custom_lr={}):
    """
    Adds specified learning rate to the specified optimizer groups.
    NB: custom_lr should be a dict, with keys being the parts of the model to
    which a custom learning rate should be applied, and values being the custom
    learning rate to apply.
    Inspired by: https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L263
    """
    for optim_group in optim_groups:
        name = optim_group["param_name"]
        for k in custom_lr.keys():
            if name == k:
                if custom_lr[k] > 0.0:
                    # apply a custom learning rate to the parameter
                    optim_group["params"].requires_grad = True
                    optim_group["lr"] = custom_lr[k]
                elif custom_lr[k] == 0.0:
                    # set requires_grad to False (computationally faster than setting lr to 0)
                    optim_group["params"].requires_grad = False
    
    return optim_groups

