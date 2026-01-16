
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    param_list = model if isinstance(model, list) else list(model.named_parameters())
    for name, param in param_list:
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1  or 'bn' in name or name.endswith(".bias") or name in skip_list:
            # print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
