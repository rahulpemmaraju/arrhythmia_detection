import snntorch.surrogate as surrogate

from .hierarchical_snn import OneLayer_HierarchicalSNN, TwoLayer_HierarchicalSNN, ThreeLayer_HierarchicalSNN
from .vanilla_snn import TwoLayer_SNN, OneLayer_SNN
from .multiscale_snn import TwoLayer_MultiscaleSNN, ThreeLayer_MultiscaleSNN, OneLayer_MultiscaleSNN
from .gated_snn import TwoLayer_GatedSNN, ThreeLayer_GatedSNN, OneLayer_GatedSNN
from .hg_snn import TwoLayer_HGSNN
from .eprop_hsnn import TwoLayer_EpropHierarchicalSNN, ThreeLayer_EpropHierarchicalSNN

def create_model(model_configs):
    if model_configs['kwargs']['model_type'] in ['one_layer_snn']:
        neuron_options = {
            'alpha': model_configs['x0']['alpha_0'],
            'threshold': model_configs['x0']['threshold_0'],
            'spike_grad': surrogate.atan(alpha=2), 
            'linear_options':{'bias': model_configs['kwargs']['bias']}
        }

        if model_configs['kwargs']['neuron_type'] == 'alif':
            neuron_options['beta'] = model_configs['x0']['beta_0']
            neuron_options['rho'] = model_configs['x0']['rho_0']
    
    elif model_configs['kwargs']['model_type'] in ['two_layer_ehsnn', 'one_layer_hsnn', 'one_layer_gsnn', 'one_layer_msnn', 'two_layer_snn', 'two_layer_hsnn', 'two_layer_msnn', 'two_layer_gsnn']:
        neuron_options = [
            {
                'alpha': model_configs['x0']['alpha_0'],
                'threshold': model_configs['x0']['threshold_0'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            },
            {
                'alpha': model_configs['x0']['alpha_1'],
                'threshold': model_configs['x0']['threshold_1'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            }
        ]

        if model_configs['kwargs']['neuron_type'] == 'alif':
            neuron_options[0]['beta'] = model_configs['x0']['beta_0']
            neuron_options[0]['rho'] = model_configs['x0']['rho_0']
            neuron_options[1]['beta'] = model_configs['x0']['beta_1']
            neuron_options[1]['rho'] = model_configs['x0']['rho_1']

    elif model_configs['kwargs']['model_type'] in ['three_layer_ehsnn', 'three_layer_hsnn', 'three_layer_msnn', 'three_layer_gsnn']:
        neuron_options = [
            {
                'alpha': model_configs['x0']['alpha_0'],
                'threshold': model_configs['x0']['threshold_0'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            },
            {
                'alpha': model_configs['x0']['alpha_1'],
                'threshold': model_configs['x0']['threshold_1'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            },
            {
                'alpha': model_configs['x0']['alpha_2'],
                'threshold': model_configs['x0']['threshold_2'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            }

        ]

        if model_configs['kwargs']['neuron_type'] == 'alif':
            neuron_options[0]['beta'] = model_configs['x0']['beta_0']
            neuron_options[0]['rho'] = model_configs['x0']['rho_0']
            neuron_options[1]['beta'] = model_configs['x0']['beta_1']
            neuron_options[1]['rho'] = model_configs['x0']['rho_1']
            neuron_options[2]['beta'] = model_configs['x0']['beta_2']
            neuron_options[2]['rho'] = model_configs['x0']['rho_2']


    out_options = {}

    if model_configs['kwargs']['model_type'] not in ['three_layer_ehsnn', 'three_layer_hsnn', 'three_layer_gsnn', 'three_layer_msnn', 'two_layer_ehsnn', 'one_layer_hsnn', 'one_layer_gsnn', 'one_layer_msnn']:
        if model_configs['kwargs']['out_type'] == 'lif':
            out_options = {
                'alpha': model_configs['x0']['alpha_out'],
                'threshold': model_configs['x0']['threshold_out'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            }

        elif model_configs['kwargs']['out_type'] == 'alif':
            out_options = {
                'alpha': model_configs['x0']['alpha_out'],
                'beta': model_configs['x0']['beta_out'],
                'rho': model_configs['x0']['rho_out'],
                'threshold': model_configs['x0']['threshold_out'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': model_configs['kwargs']['bias']}
            }

    model_params = {
            'in_dims': 1,
            'out_dims': 1,
            'recurrent_dims': model_configs['kwargs']['recurrent_dims'],
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
    
    if model_configs['kwargs']['model_type'] == 'three_layer_hsnn':
        return ThreeLayer_HierarchicalSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'three_layer_ehsnn':
        return ThreeLayer_EpropHierarchicalSNN(**model_params)

    if model_configs['kwargs']['model_type'] == 'three_layer_msnn':
        return ThreeLayer_MultiscaleSNN(**model_params)
    
    if model_configs['kwargs']['model_type'] == 'three_layer_gsnn':
        return ThreeLayer_GatedSNN(**model_params)

if __name__ == '__main__':
    import os
    import torch
    import random
    import yaml
    import numpy as np

    seed = 0
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open('nested_configs/onelayer_hsnn_128_lif_config.yaml', 'r') as file:
        model_configs = yaml.safe_load(file)
    
    model = create_model(model_configs)

    for name, param in model.named_parameters():
        print(f"* Name: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Requires Grad: {param.requires_grad}")
        print(f"  Data: {param.data}\n")