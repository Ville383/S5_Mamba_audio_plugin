import torch
import torch.nn as nn
import json
import numpy as np
from model.mamba import Mamba

def parse_linear_layers(model: Mamba, file_dict):
    """
    Parse a PyTorch model and store it in a JSON file.
    Weights from Linear layers and B, C matrices from MambaBlock are transposed before being stored.
    Complex matrices B and C are split into real and imaginary components.
   
    Args:
        model     (Mamba: nn.Module)
        file_dict (dict)
    """
    if 'layers' not in file_dict:
        file_dict['layers'] = []
   
    last_output_features = None
   
    def get_linear_layer_info(layer: nn.Linear):
        weights = layer.weight.detach().cpu().numpy()
       
        bias = None
        if layer.bias is not None:
            bias = layer.bias.detach().cpu().numpy().tolist()
       
        layer_info = {
            'type': 'dense',
            'activation': '',
            'shape': [None, layer.out_features],
            'weights': [weights.tolist()]
        }
       
        if bias is not None:
            layer_info['weights'].append(bias)
       
        return layer_info, layer.out_features
    
    def get_linear_params(layer):
        weights = layer.weight.detach().cpu().numpy()
        
        params = {
            'weights': weights.tolist(),
            'bias': None,
            'shape': [None, layer.out_features]
        }
        
        if layer.bias is not None:
            params['bias'] = layer.bias.detach().cpu().numpy().tolist()
            
        return params
    
    def get_norm_params(norm_layer):
        params = {
            'weight': norm_layer.weight.detach().cpu().numpy().tolist(),
            'eps': norm_layer.eps
        }
        return params
   
    def get_mamba_parameters(module):
        # Find S5_SSM module
        ssm_module = module.ssm
       
        # Extract and process parameters
        A_real = ssm_module.A_real.detach().cpu().numpy()
        A_imag = ssm_module.A_imag.detach().cpu().numpy()
        B_real = ssm_module.B_real.detach().cpu().numpy()
        B_imag = ssm_module.B_imag.detach().cpu().numpy()
        C_real = ssm_module.C_real.detach().cpu().numpy()
        C_imag = ssm_module.C_imag.detach().cpu().numpy()
        D = ssm_module.D.detach().cpu().numpy()
        inv_dt = ssm_module.inv_dt.detach().cpu().numpy()
        
        # Get linear layer parameters
        in_proj_params = get_linear_params(module.in_proj) if hasattr(module, 'in_proj') else None
        out_proj_params = get_linear_params(module.out_proj) if hasattr(module, 'out_proj') else None
       
        return {
            'A_real': A_real.tolist(),
            'A_imag': A_imag.tolist(),
            'B_real': B_real.tolist(),
            'B_imag': B_imag.tolist(),
            'C_real': C_real.tolist(),
            'C_imag': C_imag.tolist(),
            'D': D.tolist(),
            'inv_dt': inv_dt.tolist(),
            'in_proj': in_proj_params,
            'out_proj': out_proj_params
        }
    
    def get_residual_block_params(module):
        # Get MambaBlock parameters
        mamba_params = get_mamba_parameters(module.mamba)
        
        # Get RMSNorm parameters
        norm_params = get_norm_params(module.norm)
        
        return {
            'type': 'residual',
            'parameters': {
                'mamba': mamba_params,
                'norm': norm_params
            }
        }
   
    def traverse_model(model: Mamba):
        nonlocal last_output_features

        for name, child in model.named_children():
            if isinstance(child, nn.Linear):
                layer_info, output_features = get_linear_layer_info(child)
                file_dict['layers'].append(layer_info)
                last_output_features = output_features
            elif isinstance(child, nn.ModuleList):
                # Handle ModuleList of ResidualBlocks
                for residual_block in child:
                    residual_params = get_residual_block_params(residual_block)
                    file_dict['layers'].append(residual_params)
                    if hasattr(residual_block.mamba, 'out_proj'):
                        last_output_features = residual_block.mamba.out_proj.out_features
            else:
                traverse_model(child)
   
    traverse_model(model)
    return file_dict

def model_2_json(model: Mamba, out_path="model_weights"):
    file_dict = {'in_shape': [None, 1]}
    file_dict = parse_linear_layers(model, file_dict)

    with open(f"{out_path}.json", "w") as json_file: 
        json.dump(file_dict, json_file, indent=1)