import yaml
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    import torch
    from denoiser import Denoiser
    from lsm_utils import normalize_16bit_images, compute_norm_range
    
    config = yaml.load(open("model_config_self.yaml", "r"), Loader=yaml.FullLoader)
    print('Computing normalization range for 16-bit data.')
    vmin, vmax, fail_names = compute_norm_range(config['dataset'], ext='tif', percentiles=(1, 99.5), sample_r=0.05)
    if len(fail_names) != 0:
        print('Datasets corrupted: input: {fail_names_i}, target: {fail_neams_t}')
        
    config['norm-range'] = [int(vmin), int(vmax)]
    denoiser = Denoiser(config, False)
    if config['load-weights'] is not None:
        denoiser.backbone.load_state_dict(torch.load(os.path.join('model_weights', config['load-weights'], 'g.pth')))
    denoiser.train(write_log=True, valid_r=1)