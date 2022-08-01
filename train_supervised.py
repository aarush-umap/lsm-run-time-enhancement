import os
import yaml

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    import torch
    from enhancer import Enhancer
    from lsm_utils import compute_norm_range
    
    config = yaml.load(open("model_config.yaml", "r"), Loader=yaml.FullLoader)
    print('Computing normalization range for 16-bit data.')
    vmin_t, vmax_t, fail_names_t = compute_norm_range(config['dataset'] + os.sep+ 'target', ext='tif', percentiles=(1, 99.5), sample_r=0.2)
    vmin_i, vmax_i, fail_names_i = compute_norm_range(config['dataset'] + os.sep+ 'input', ext='tif', percentiles=(1, 99.5), sample_r=0.2)
    if len(fail_names_i + fail_names_t) != 0:
        print('Datasets corrupted: input: {fail_names_i}, target: {fail_neams_t}')
    config['norm-range'] = [int(vmin_i), int(vmax_i)]
    config['norm-range-target'] = [int(vmin_t), int(vmax_t)]
    with open("model_config.yaml", 'w') as file:
        yaml.dump(config, file)
    if config['only-pixel-loss']:
        enhancer = Enhancer(config, scale_factor=config['up-scale-factor'], perceptual_loss=False, adversarial_loss=False)
    else:
        enhancer = Enhancer(config, scale_factor=config['up-scale-factor'], perceptual_loss=True, adversarial_loss=True)

    if config['load-weights']:
        enhancer.backbone.load_state_dict(torch.load(os.path.join('model_weights', config['load-weights'], 'g.pth')))
        if not config['only-pixel-loss']:
            enhancer.discriminator.load_state_dict(torch.load(os.path.join('model_weights', config['load-weights'], 'd.pth')))

    enhancer.train(write_log=True, valid_r=0.1)