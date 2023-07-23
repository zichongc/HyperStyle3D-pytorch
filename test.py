import os
import argparse
from munch import Munch
import numpy as np
import torch
import torchvision
from StyleSDF.model import Generator
from StyleSDF.options import BaseOptions
from model import HyperModule, TextEncoder
from StyleSDF.utils import (
    generate_camera_params,
    align_volume,
    extract_mesh_with_marching_cubes,
    xyz2mesh,
)
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def views():
    locations = torch.tensor([[0, 0],
                              [-1. * opt.camera.azim, 0],
                              [1. * opt.camera.azim, 0],
                              [0, -1 * opt.camera.elev],
                              [0, 1 * opt.camera.elev]
                              ], device=device)
    fov = opt.camera.fov * torch.ones((locations.shape[0], 1), device=device)
    num_viewdirs = locations.shape[0]

    with torch.no_grad():
        chunk = 8
        sample_z = torch.randn(1, opt.inference.style_dim, device=device).repeat(num_viewdirs, 1)
        sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = \
            generate_camera_params(opt.inference.renderer_output_size, device, batch=num_viewdirs,
                                   locations=locations,  # input_fov=fov,
                                   uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                   elev_range=opt.camera.elev, fov_ang=fov,
                                   dist_radius=opt.camera.dist_radius)
        src_rgb_images = torch.Tensor(0, 3, opt.inference.size, opt.inference.size)  # 1024
        src_rgb_images_thumbs = torch.Tensor(0, 3, opt.inference.renderer_output_size,
                                             opt.inference.renderer_output_size)  # 64
        tgt_rgb_images = torch.Tensor(0, 3, opt.inference.size, opt.inference.size)  # 1024
        tgt_rgb_images_thumbs = torch.Tensor(0, 3, opt.inference.renderer_output_size,
                                             opt.inference.renderer_output_size)  # 64
        for j in range(0, num_viewdirs, chunk):
            out = tgt_g([sample_z[j:j + chunk]],
                        sample_cam_extrinsics[j:j + chunk],
                        sample_focals[j:j + chunk],
                        sample_near[j:j + chunk],
                        sample_far[j:j + chunk],
                        truncation=opt.inference.truncation_ratio,
                        truncation_latent=mean_latent,
                        weights_delta=weights_delta)
            tgt_rgb_images = torch.cat([tgt_rgb_images, out[0].cpu()], 0)
            tgt_rgb_images_thumbs = torch.cat([tgt_rgb_images_thumbs, out[1].cpu()], 0)

            out = src_g([sample_z[j:j + chunk]],
                        sample_cam_extrinsics[j:j + chunk],
                        sample_focals[j:j + chunk],
                        sample_near[j:j + chunk],
                        sample_far[j:j + chunk],
                        truncation=opt.inference.truncation_ratio,
                        truncation_latent=mean_latent)
            src_rgb_images = torch.cat([src_rgb_images, out[0].cpu()], 0)
            src_rgb_images_thumbs = torch.cat([src_rgb_images_thumbs, out[1].cpu()], 0)

    samples = torch.cat([src_rgb_images, tgt_rgb_images], 0)
    target_path = arguments.target_path
    torchvision.utils.save_image(samples, target_path,
                                 nrow=5,
                                 normalize=True,
                                 value_range=(-1, 1),
                                 )
    print(f'Synthesis images saved to {target_path}.')


def degrees(mesh=False):
    locations = torch.tensor([[0, 0]] * len(factors), device=device)
    fov = opt.camera.fov * torch.ones((locations.shape[0], 1), device=device)
    num_viewdirs = locations.shape[0]
    tgt_rgb_images = torch.Tensor(0, 3, opt.inference.size, opt.inference.size)  # 1024

    chunk = 1
    sample_z = torch.randn(1, opt.inference.style_dim, device=device).repeat(num_viewdirs, 1)
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = \
        generate_camera_params(opt.inference.renderer_output_size, device, batch=num_viewdirs,
                               locations=locations,  # input_fov=fov,
                               uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                               elev_range=opt.camera.elev, fov_ang=fov,
                               dist_radius=opt.camera.dist_radius)
    with torch.no_grad():
        for j in range(0, num_viewdirs, chunk):
            out = tgt_g([sample_z[j:j + chunk]],
                        sample_cam_extrinsics[j:j + chunk],
                        sample_focals[j:j + chunk],
                        sample_near[j:j + chunk],
                        sample_far[j:j + chunk],
                        truncation=opt.inference.truncation_ratio,
                        truncation_latent=mean_latent,
                        weights_delta=[w * factors[j] for w in weights_delta])
            tgt_rgb_images = torch.cat([tgt_rgb_images, out[0].cpu()], 0)

        sample_z = torch.randn(1, opt.inference.style_dim, device=device).repeat(num_viewdirs, 1)
        sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = \
            generate_camera_params(opt.inference.renderer_output_size, device, batch=num_viewdirs,
                                   locations=locations,  # input_fov=fov,
                                   uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                   elev_range=opt.camera.elev, fov_ang=fov,
                                   dist_radius=opt.camera.dist_radius)
        for j in range(0, num_viewdirs, chunk):
            out = tgt_g([sample_z[j:j + chunk]],
                        sample_cam_extrinsics[j:j + chunk],
                        sample_focals[j:j + chunk],
                        sample_near[j:j + chunk],
                        sample_far[j:j + chunk],
                        truncation=opt.inference.truncation_ratio,
                        truncation_latent=mean_latent,
                        weights_delta=[w * factors[j] for w in weights_delta])
            tgt_rgb_images = torch.cat([tgt_rgb_images, out[0].cpu()], 0)
            del out

    target_path = arguments.target_path
    torchvision.utils.save_image(tgt_rgb_images, target_path,
                                 nrow=len(factors),
                                 normalize=True,
                                 value_range=(-1, 1),
                                 )
    if mesh:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model.copy()
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(
            device)

        # Load weights to surface extractor
        surface_extractor_dict = surface_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                surface_extractor_dict[k] = v

        surface_g_ema.load_state_dict(surface_extractor_dict)
        with torch.no_grad():
            for j in range(0, num_viewdirs, chunk):
                surface_out = surface_g_ema([sample_z[j:j + chunk]],
                                            sample_cam_extrinsics[j:j + chunk],
                                            sample_focals[j:j + chunk],
                                            sample_near[j:j + chunk],
                                            sample_far[j:j + chunk],
                                            truncation=opt.inference.truncation_ratio,
                                            truncation_latent=mean_latent,
                                            weights_delta=[w * factors[j] for w in weights_delta],
                                            return_sdf=True, return_xyz=True)
                xyz, sdf = surface_out[2], surface_out[3]
                del surface_out

                for k in range(chunk):
                    curr_locations = sample_locations[j:j + chunk]
                    loc_str = '_azim{}_elev{}'.format(int(curr_locations[k, 0] * 180 / np.pi),
                                                      int(curr_locations[k, 1] * 180 / np.pi))
                    depth_mesh = xyz2mesh(xyz[k:k + chunk])

                    if depth_mesh is not None:
                        depth_mesh_filename = os.path.join(arguments.target_depth_path, f'depth{j}.obj')
                        with open(depth_mesh_filename, 'w') as f:
                            depth_mesh.export(f, file_type='obj')
                        if j == 0:
                            try:
                                frostum_aligned_sdf = align_volume(sdf)
                                marching_cubes_mesh = extract_mesh_with_marching_cubes(frostum_aligned_sdf[k:k + chunk])
                            except ValueError:
                                marching_cubes_mesh = None
                                print('Marching cubes extraction failed.')
                                print('Please check whether the SDF values are all larger (or all smaller) than 0.')

                            if marching_cubes_mesh is not None:
                                marching_cubes_mesh_filename = os.path.join(arguments.target_sdf_path, f'mesh{j}.obj')
                                with open(marching_cubes_mesh_filename, 'w') as f:
                                    marching_cubes_mesh.export(f, file_type='obj')


print('Running test for HyperStyle3D...')
print('Initializing configurations...')
# src_txt = {'shape_txt': 'face, face',
#            'attribute_txt': 'hair, face, face',
#            'style_txt': 'photo'}
# # tgt_txt = {'shape_txt': 'fat face, long face',
# #            'attribute_txt': 'silver hair, bearded face, old face',
# #            'style_txt': 'pixar'}
# tgt_txt = {'shape_txt': 'face, face',
#            'attribute_txt': 'hair, face, face',
#            'style_txt': 'pixar'}

src_txt = {'shape_txt': 'face',
           'attribute_txt': 'age',
           'style_txt': 'photograph of human face'}
# tgt_txt = {'shape_txt': 'fat face, long face',
#            'attribute_txt': 'silver hair, bearded face, old face',
#            'style_txt': 'pixar'}
tgt_txt = {'shape_txt': 'face',
           'attribute_txt': 'young age',
           'style_txt': 'photograph of Ukiyo-e human head'}
print("[source text]:")
print(src_txt)
print('[target text]:')
print(tgt_txt)

# factors = [-.3, 0., .3, .6, .9, 1.2]
# factors = [-.4, 0, .4, .8, 1.2, 1.6]
factors = [-.8, -.4, 0, .4, .8]

parser = argparse.ArgumentParser()
mode = 'final'
pt_path = './output/' + mode
parser.add_argument('--coarse_path', type=str, default=os.path.join(pt_path, 'hyper_coarse.pt'))
parser.add_argument('--medium_path', type=str, default=os.path.join(pt_path, 'hyper_medium.pt'))
parser.add_argument('--fine_path', type=str, default=os.path.join(pt_path, 'hyper_fine.pt'))
parser.add_argument('--expname', type=str, default='ffhq1024x1024')
parser.add_argument('--group', type=str, default='333', help='division of coarse, medium and fine. default [3,3,3]')
parser.add_argument('--target_path', type=str, default=os.path.join('./output', f'test_Ukiyo0.png'))
parser.add_argument('--target_depth_path', type=str, default='./output')
parser.add_argument('--target_sdf_path', type=str, default='./output')
arguments = parser.parse_args()

device = "cpu"
opt = BaseOptions().parse()
opt.model.is_test = True
opt.model.freeze_renderer = True
opt.model.size = 1024
opt.rendering.offset_sampling = True
opt.rendering.static_viewdirs = True
opt.rendering.force_background = True
opt.rendering.perturb = 0
opt.inference.size = opt.model.size
opt.inference.camera = opt.camera
opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
opt.inference.style_dim = opt.model.style_dim
opt.inference.project_noise = opt.model.project_noise
opt.inference.return_xyz = opt.rendering.return_xyz
opt.experiment.expname = arguments.expname

print('Loading pre-trained models...')
checkpoints_dir = 'StyleSDF/full_models'
checkpoints_path = os.path.join(checkpoints_dir, opt.experiment.expname + '.pt')

print('---loading generator...', end='')
checkpoint = torch.load(checkpoints_path)
src_g = Generator(opt.model, opt.rendering).to(device)
pretrained_weights_dict = checkpoint['g_ema']
model_dict = src_g.state_dict()
for k, v in pretrained_weights_dict.items():
    if v.size() == model_dict[k].size():
        model_dict[k] = v
src_g.load_state_dict(model_dict)

tgt_g = Generator(opt.model, opt.rendering).to(device)
pretrained_weights_dict = checkpoint['g_ema']
model_dict = tgt_g.state_dict()
for k, v in pretrained_weights_dict.items():
    if v.size() == model_dict[k].size():
        model_dict[k] = v
tgt_g.load_state_dict(model_dict)
print('[done]')

# get the mean latent vector for g_ema
if opt.inference.truncation_ratio < 1:
    with torch.no_grad():
        mean_latent = src_g.mean_latent(opt.inference.truncation_mean, device)

print('Initializing Hyper Network...')
text_encoder = TextEncoder(device=device).to(device)
hyper_network = HyperModule(encoder=text_encoder, g=tgt_g, in_feat=512, group=list(map(int, list(arguments.group))))
hyper_network = hyper_network.to(device)

print('Loading pre-trained models...')
hyper_coarse_checkpoint = torch.load(arguments.coarse_path)
hyper_medium_checkpoint = torch.load(arguments.medium_path)
hyper_fine_checkpoint = torch.load(arguments.fine_path)

print('---loading coarse hyper network...', end='')
model_dict = hyper_network.coarse.state_dict()
for k, v in hyper_coarse_checkpoint.items():
    if v.size() == model_dict[k].size():
        model_dict[k] = v
hyper_network.coarse.load_state_dict(model_dict)
print('[done]')

print('---loading medium hyper network...', end='')
model_dict = hyper_network.medium.state_dict()
for k, v in hyper_medium_checkpoint.items():
    if v.size() == model_dict[k].size():
        model_dict[k] = v
hyper_network.medium.load_state_dict(model_dict)
print('[done]')

print('---loading fine hyper network...', end='')
model_dict = hyper_network.fine.state_dict()
for k, v in hyper_fine_checkpoint.items():
    if v.size() == model_dict[k].size():
        model_dict[k] = v
hyper_network.fine.load_state_dict(model_dict)
print('[done]')

print('Computing parameters offsets...')
delta = hyper_network(src_txt, tgt_txt)
ppl = hyper_network.parameters_per_layer
weights_delta = []
prefix = 0
for i in range(8):
    weights_delta.append(delta[prefix:prefix + ppl[i]])
    prefix += ppl[i]
weights_delta.append(delta[prefix:])

src_g.eval()
tgt_g.eval()

print('Generating...')
views()
# degrees(mesh=False)
print('Done.')
