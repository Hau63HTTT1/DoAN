
import torch


def interpolate(feat, uv):

    if uv.shape[-1] != 2:
        uv = uv.transpose(1, 2) 
    uv = uv.unsqueeze(2)    
    if int(torch.__version__.split('.')[1]) < 4:
        samples = torch.nn.functional.grid_sample(feat, uv)  
    else:
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) 
    return samples[:, :, :, 0] 
