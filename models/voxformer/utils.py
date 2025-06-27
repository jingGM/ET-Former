import warnings
import inspect
from typing import Dict, Tuple, Union
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
import torch.utils.checkpoint as cp


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


ACTIVATION_LAYERS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'RReLU': nn.RReLU,
    'ReLU6': nn.ReLU6,
    'ELU': nn.ELU,
    'Tanh': nn.Tanh,
    'PReLU': nn.PReLU, 'Sigmoid': nn.Sigmoid,
}


def build_activation_layer(cfg, *args, **kwargs):
    cfg_ = cfg.copy()
    activation_type = cfg_.pop('type')
    if activation_type not in ACTIVATION_LAYERS.keys():
        raise KeyError(f'Unrecognized padding type {activation_type}.')
    else:
        layer = ACTIVATION_LAYERS[activation_type]
    layer = layer(*args, **kwargs, **cfg_)
    return layer


PADDING_LAYERS = {
    'zero': nn.ZeroPad2d,
    'reflect': nn.ReflectionPad2d,
    'replicate': nn.ReplicationPad2d,
}


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDING_LAYERS.keys():
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = PADDING_LAYERS[padding_type]

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer


PLUGIN_LAYERS = {}


def build_plugin_layer(cfg, postfix='', **kwargs):
    """Build plugin layer.

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify plugin layer type.
            layer args: args needed to instantiate a plugin layer.
        postfix (int, str): appended into norm abbreviation to
            create named layer. Default: ''.

    Returns:
        tuple[str, nn.Module]:
            name (str): abbreviation + postfix
            layer (nn.Module): created plugin layer
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in PLUGIN_LAYERS.keys():
        raise KeyError(f'Unrecognized plugin type {layer_type}')

    plugin_layer = PLUGIN_LAYERS[layer_type]
    abbr = infer_abbr(plugin_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = plugin_layer(**kwargs, **cfg_)

    return name, layer


CONV_LAYERS = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Conv': nn.Conv2d,
}


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS.keys():
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class SyncBatchNorm(nn.SyncBatchNorm):
    def _check_input_dim(self, input):
        super()._check_input_dim(input)


NORM_LAYERS = {
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'SyncBN': SyncBatchNorm,
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'IN': nn.InstanceNorm2d,
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
}


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(cfg, num_features,
                     postfix: Union[int, str] = ''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        (str, nn.Module): The first element is the layer name consisting of
            abbreviation and postfix, e.g., bn1, gn. The second element is the
            created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS.keys():
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS[layer_type]
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
