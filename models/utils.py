import snntorch.surrogate as surrogate

from .hierarchical_snn import OneLayer_HierarchicalSNN, TwoLayer_HierarchicalSNN, ThreeLayer_HierarchicalSNN, FourLayer_HierarchicalSNN
from .vanilla_snn import TwoLayer_SNN, OneLayer_SNN, FourLayer_SNN
from .multiscale_snn import TwoLayer_MultiscaleSNN, ThreeLayer_MultiscaleSNN, OneLayer_MultiscaleSNN, FourLayer_MultiscaleSNN
from .gated_snn import TwoLayer_GatedSNN, ThreeLayer_GatedSNN, OneLayer_GatedSNN, FourLayer_GatedSNN
from .hg_snn import ThreeLayer_HGSNN, FourLayer_HGSNN
from .bsnn import TwoLayer_BSNN, ThreeLayer_BSNN, FourLayer_BSNN
from .bsnn_no_flush import TwoLayer_BSNN_no_Flush, ThreeLayer_BSNN_no_Flush, FourLayer_BSNN_no_Flush
from .bsnn_no_share import TwoLayer_BSNN_no_Share, ThreeLayer_BSNN_no_Share, FourLayer_BSNN_no_Share
from .eprop_hsnn import TwoLayer_EpropHierarchicalSNN, ThreeLayer_EpropHierarchicalSNN, FourLayer_EpropHierarchicalSNN
from .eprop_bsnn import TwoLayer_EpropBSNN, ThreeLayer_EpropBSNN, FourLayer_EpropBSNN

# these architectures have no separate readout layer -- their top cell emits the
# output directly, so its width IS out_dims and recurrent_dims[-1] must match it.
# every other model type ends in an `fc` readout that maps recurrent_dims[-1] ->
# out_dims, leaving recurrent_dims[-1] free as a hidden width.
TOP_CELL_IS_READOUT = [
    'three_layer_bsnn', 'three_layer_ehsnn', 'three_layer_hsnn', 'three_layer_hgsnn',
    'three_layer_gsnn', 'three_layer_msnn', 'two_layer_bsnn', 'two_layer_ehsnn',
    'one_layer_hsnn', 'one_layer_gsnn', 'one_layer_msnn',
    'three_layer_bsnn_no_flush', 'two_layer_bsnn_no_flush',
    'three_layer_bsnn_no_share', 'two_layer_bsnn_no_share',
    'three_layer_ebsnn', 'two_layer_ebsnn',
]


def create_model(model_configs, config_key='x0'):
    if model_configs['kwargs']['model_type'] in ['one_layer_snn']:
        neuron_options = {
            'alpha': model_configs[config_key]['alpha_0'],
            'threshold': model_configs[config_key]['threshold_0'],
            'spike_grad': surrogate.atan(alpha=2), 
            'linear_options':{'bias': model_configs['kwargs']['bias']}
        }

        if model_configs['kwargs']['neuron_type'] == 'alif':
            neuron_options['beta'] = model_configs[config_key]['beta_0']
            neuron_options['rho'] = model_configs[config_key]['rho_0']
    
    elif model_configs['kwargs']['model_type'] in ['two_layer_bsnn', 'two_layer_bsnn_no_flush', 'two_layer_bsnn_no_share', 'two_layer_ebsnn', 'two_layer_ehsnn', 'one_layer_hsnn', 'one_layer_gsnn', 'one_layer_msnn', 'two_layer_snn', 'two_layer_hsnn', 'two_layer_msnn', 'two_layer_gsnn']:
        neuron_options = [
            {
                'alpha': model_configs[config_key]['alpha_0'],
                'threshold': model_configs[config_key]['threshold_0'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            },
            {
                'alpha': model_configs[config_key]['alpha_1'],
                'threshold': model_configs[config_key]['threshold_1'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            }
        ]

        if model_configs['kwargs']['neuron_type'] == 'alif':
            neuron_options[0]['beta'] = model_configs[config_key]['beta_0']
            neuron_options[0]['rho'] = model_configs[config_key]['rho_0']
            neuron_options[1]['beta'] = model_configs[config_key]['beta_1']
            neuron_options[1]['rho'] = model_configs[config_key]['rho_1']

    elif model_configs['kwargs']['model_type'] in ['three_layer_bsnn', 'three_layer_bsnn_no_flush', 'three_layer_bsnn_no_share', 'three_layer_ebsnn', 'three_layer_ehsnn', 'three_layer_hsnn', 'three_layer_hgsnn', 'three_layer_msnn', 'three_layer_gsnn', 'four_layer_hsnn', 'four_layer_hgsnn', 'four_layer_ehsnn', 'four_layer_gsnn', 'four_layer_bsnn', 'four_layer_bsnn_no_flush', 'four_layer_bsnn_no_share', 'four_layer_ebsnn', 'four_layer_msnn', 'four_layer_snn']:
        neuron_options = [
            {
                'alpha': model_configs[config_key]['alpha_0'],
                'threshold': model_configs[config_key]['threshold_0'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            },
            {
                'alpha': model_configs[config_key]['alpha_1'],
                'threshold': model_configs[config_key]['threshold_1'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            },
            {
                'alpha': model_configs[config_key]['alpha_2'],
                'threshold': model_configs[config_key]['threshold_2'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            }

        ]

        if model_configs['kwargs']['neuron_type'] == 'alif':
            neuron_options[0]['beta'] = model_configs[config_key]['beta_0']
            neuron_options[0]['rho'] = model_configs[config_key]['rho_0']
            neuron_options[1]['beta'] = model_configs[config_key]['beta_1']
            neuron_options[1]['rho'] = model_configs[config_key]['rho_1']
            neuron_options[2]['beta'] = model_configs[config_key]['beta_2']
            neuron_options[2]['rho'] = model_configs[config_key]['rho_2']


    out_options = {}

    if model_configs['kwargs']['model_type'] not in TOP_CELL_IS_READOUT:
        if model_configs['kwargs']['out_type'] == 'lif':
            out_options = {
                'alpha': model_configs[config_key]['alpha_out'],
                'threshold': model_configs[config_key]['threshold_out'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            }

        elif model_configs['kwargs']['out_type'] == 'alif':
            out_options = {
                'alpha': model_configs[config_key]['alpha_out'],
                'beta': model_configs[config_key]['beta_out'],
                'rho': model_configs[config_key]['rho_out'],
                'threshold': model_configs[config_key]['threshold_out'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            }

    # out_dims is the number of readout units: 1 for binary (single logit), or
    # the class count for multiclass.
    out_dims = model_configs['kwargs'].get('out_dims', 1)
    recurrent_dims = model_configs['kwargs']['recurrent_dims']
    model_type = model_configs['kwargs']['model_type']

    # for these the top cell is the readout, so a mismatch would silently build a
    # model whose output width is not the class count
    if model_type in TOP_CELL_IS_READOUT and recurrent_dims[-1] != out_dims:
        raise ValueError(
            f"model_type '{model_type}' has no separate readout layer, so "
            f'recurrent_dims[-1] ({recurrent_dims[-1]}) must equal out_dims ({out_dims})'
        )

    model_params = {
            'in_dims': 1,
            'out_dims': out_dims,
            'recurrent_dims': recurrent_dims,
            'neuron_options': neuron_options,
            'neuron_type': model_configs['kwargs']['neuron_type'],
            'out_type': model_configs['kwargs']['out_type'],
            'out_options': out_options,
    }

    if model_configs['kwargs']['model_type'] == 'one_layer_snn':
        return OneLayer_SNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'one_layer_gsnn':
        return OneLayer_GatedSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'one_layer_hsnn':
        return OneLayer_HierarchicalSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'two_layer_ehsnn':
        return TwoLayer_EpropHierarchicalSNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'two_layer_bsnn':
        return TwoLayer_BSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'one_layer_msnn':
        return OneLayer_MultiscaleSNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'two_layer_snn':
        return TwoLayer_SNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'two_layer_hsnn':
        return TwoLayer_HierarchicalSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'two_layer_msnn':
        return TwoLayer_MultiscaleSNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'two_layer_gsnn':
        return TwoLayer_GatedSNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'three_layer_bsnn':
        return ThreeLayer_BSNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'three_layer_hsnn':
        return ThreeLayer_HierarchicalSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'three_layer_hgsnn':
        return ThreeLayer_HGSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_hgsnn':
        return FourLayer_HGSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_snn':
        return FourLayer_SNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_hsnn':
        return FourLayer_HierarchicalSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_ehsnn':
        return FourLayer_EpropHierarchicalSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_gsnn':
        return FourLayer_GatedSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_bsnn':
        return FourLayer_BSNN(**model_params)

    # bsnn ablations: boundaries without state flushing / without interlayer sharing
    if model_configs['kwargs']['model_type'] == 'two_layer_bsnn_no_flush':
        return TwoLayer_BSNN_no_Flush(**model_params)

    if model_configs['kwargs']['model_type'] == 'three_layer_bsnn_no_flush':
        return ThreeLayer_BSNN_no_Flush(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_bsnn_no_flush':
        return FourLayer_BSNN_no_Flush(**model_params)

    if model_configs['kwargs']['model_type'] == 'two_layer_bsnn_no_share':
        return TwoLayer_BSNN_no_Share(**model_params)

    if model_configs['kwargs']['model_type'] == 'three_layer_bsnn_no_share':
        return ThreeLayer_BSNN_no_Share(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_bsnn_no_share':
        return FourLayer_BSNN_no_Share(**model_params)

    # e-prop trained bsnn
    if model_configs['kwargs']['model_type'] == 'two_layer_ebsnn':
        return TwoLayer_EpropBSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'three_layer_ebsnn':
        return ThreeLayer_EpropBSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_ebsnn':
        return FourLayer_EpropBSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'four_layer_msnn':
        return FourLayer_MultiscaleSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'three_layer_ehsnn':
        return ThreeLayer_EpropHierarchicalSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'three_layer_msnn':
        return ThreeLayer_MultiscaleSNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'three_layer_gsnn':
        return ThreeLayer_GatedSNN(**model_params)

