import os
import torch


def save_models(network, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(network.coarse.state_dict(), os.path.join(save_dir, f'hyper_coarse.pt'))
    torch.save(network.medium.state_dict(), os.path.join(save_dir, 'hyper_medium.pt'))
    torch.save(network.fine.state_dict(), os.path.join(save_dir, 'hyper_fine.pt'))

