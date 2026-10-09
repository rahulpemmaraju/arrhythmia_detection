import snntorch.surrogate as surrogate

from .vanilla_snn import OneLayer_SNN, TwoLayer_SNN, FourLayer_SNN
from .hgrsnn import TwoLayer_HGRSNN, ThreeLayer_HGRSNN, FourLayer_HGRSNN
from .hgrsnn_no_flush import TwoLayer_HGRSNN_no_Flush, ThreeLayer_HGRSNN_no_Flush, FourLayer_HGRSNN_no_Flush
from .hgrsnn_no_share import TwoLayer_HGRSNN_no_Share, ThreeLayer_HGRSNN_no_Share, FourLayer_HGRSNN_no_Share
from .eprop_hgrsnn import TwoLayer_EpropHGRSNN, ThreeLayer_EpropHGRSNN, FourLayer_EpropHGRSNN

# model_type -> (class, number of recurrent layers, top cell is the readout).
#
# a model whose top cell is the readout has no separate readout layer, so its width
# IS out_dims and recurrent_dims[-1] must match it. every other model ends in an `fc`
# readout that maps recurrent_dims[-1] -> out_dims, leaving recurrent_dims[-1] free as
# a hidden width. the configs in nested_configs/ all use the four_layer_* types.
MODELS = {
    # recurrent SNN
    'one_layer_snn': (OneLayer_SNN, 1, False),
    'two_layer_snn': (TwoLayer_SNN, 2, False),
    'four_layer_snn': (FourLayer_SNN, 3, False),

    # hgrsnn
    'two_layer_hgrsnn': (TwoLayer_HGRSNN, 2, True),
    'three_layer_hgrsnn': (ThreeLayer_HGRSNN, 3, True),
    'four_layer_hgrsnn': (FourLayer_HGRSNN, 3, False),

    # hgrsnn ablations: boundaries without state flushing / without interlayer sharing
    'two_layer_hgrsnn_no_flush': (TwoLayer_HGRSNN_no_Flush, 2, True),
    'three_layer_hgrsnn_no_flush': (ThreeLayer_HGRSNN_no_Flush, 3, True),
    'four_layer_hgrsnn_no_flush': (FourLayer_HGRSNN_no_Flush, 3, False),

    'two_layer_hgrsnn_no_share': (TwoLayer_HGRSNN_no_Share, 2, True),
    'three_layer_hgrsnn_no_share': (ThreeLayer_HGRSNN_no_Share, 3, True),
    'four_layer_hgrsnn_no_share': (FourLayer_HGRSNN_no_Share, 3, False),

    # e-prop trained hgrsnn
    'two_layer_hgrsnn_eprop': (TwoLayer_EpropHGRSNN, 2, True),
    'three_layer_hgrsnn_eprop': (ThreeLayer_EpropHGRSNN, 3, True),
    'four_layer_hgrsnn_eprop': (FourLayer_EpropHGRSNN, 3, False),
}


def cell_options(params, suffix, adaptive, bias):
    # hyperparameters for one spiking layer, read as alpha_{suffix}, threshold_{suffix}, ...
    options = {
        'alpha': params[f'alpha_{suffix}'],
        'threshold': params[f'threshold_{suffix}'],
        'spike_grad': surrogate.atan(alpha=2),
        'linear_options': {'bias': bias},
    }

    if adaptive:
        options['beta'] = params[f'beta_{suffix}']
        options['rho'] = params[f'rho_{suffix}']

    return options


def create_model(model_configs, config_key='x0'):
    kwargs = model_configs['kwargs']
    params = model_configs[config_key]
    model_type = kwargs['model_type']

    if model_type not in MODELS:
        raise ValueError(f"unknown model_type '{model_type}', expected one of {sorted(MODELS)}")

    model_class, num_layers, top_cell_is_readout = MODELS[model_type]
    bias = kwargs['bias']

    neuron_options = [cell_options(params, i, kwargs['neuron_type'] == 'alif', bias)
                      for i in range(num_layers)]

    # the one-layer model takes a single dict rather than a per-layer list
    if num_layers == 1:
        neuron_options = neuron_options[0]

    out_options = {}

    if not top_cell_is_readout and kwargs['out_type'] in ('lif', 'alif'):
        out_options = cell_options(params, 'out', kwargs['out_type'] == 'alif', bias)

    # out_dims is the number of readout units: 1 for binary (single logit), or
    # the class count for multiclass.
    out_dims = kwargs.get('out_dims', 1)
    recurrent_dims = kwargs['recurrent_dims']

    # for these the top cell is the readout, so a mismatch would silently build a
    # model whose output width is not the class count
    if top_cell_is_readout and recurrent_dims[-1] != out_dims:
        raise ValueError(
            f"model_type '{model_type}' has no separate readout layer, so "
            f'recurrent_dims[-1] ({recurrent_dims[-1]}) must equal out_dims ({out_dims})'
        )

    return model_class(
        in_dims=1,
        out_dims=out_dims,
        recurrent_dims=recurrent_dims,
        neuron_options=neuron_options,
        neuron_type=kwargs['neuron_type'],
        out_type=kwargs['out_type'],
        out_options=out_options,
    )
