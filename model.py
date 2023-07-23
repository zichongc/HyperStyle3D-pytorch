import clip
import torch
import torch.nn as nn
from StyleSDF.model import Generator


class TextEncoder(nn.Module):
    def __init__(self, pretrain=r'util/pretrained_models', device='cuda'):
        super().__init__()
        self.encoder, _ = clip.load('ViT-B/32', device=device, download_root=pretrain)
        self.device = device

    def forward(self, text):
        token = clip.tokenize(text).to(self.device)
        return self.encoder.encode_text(token)      # return shape [*, 512]


class HyperNetwork(nn.Module):
    def __init__(self, in_feat, out_group, n_layer=1):
        """
        :param in_feat: embedding feature dim
        :param out_group: a group of parameter volume
        :param n_layer:
        """
        super().__init__()
        self.n_module = len(out_group)
        for i in range(self.n_module):
            layers = []
            for _ in range(n_layer - 1):
                layers += self.create_layer(in_feat, in_feat)
            layers += self.create_layer(in_feat, out_group[i], activation=False)
            self.add_module(f'module_{i}', nn.Sequential(*layers))

    def create_layer(self, in_feat, out_feat, activation=True):
        linear = nn.Linear(in_feat, out_feat)
        return [linear, nn.LeakyReLU(0.2)] if activation else [linear]

    def forward(self, f):
        """
        :param f: input text feature
        :return:
        """
        delta = []
        for i in range(self.n_module):
            delta.append(eval(f'self.module_{i}')(f.float()))
        return torch.cat(delta, dim=1)


class HyperModule(nn.Module):
    def __init__(self, encoder, g: Generator, in_feat=512, group=[3, 3, 3]):
        """
        :param encoder: pretrained text encoder
        :param in_feat: input dim to hyperNetwork, i.e. output dim of encoder
        :param group: division of 9 modulated linear layers
        """
        super().__init__()
        assert sum(group) == 9, "need to match the total 9 modulated linear layers."
        self.in_feat = in_feat
        self.group = group
        self.encoder = encoder
        parameters_per_layer = []
        for i in range(8):
            in_ch, out_ch = g.renderer.network.pts_linears[i].in_channel, g.renderer.network.pts_linears[i].out_channel
         
            parameters_per_layer.append(in_ch * out_ch)
        in_ch, out_ch = g.renderer.network.views_linears.in_channel, g.renderer.network.views_linears.out_channel
        parameters_per_layer.append(in_ch * out_ch)
        
        self.parameters_per_layer = parameters_per_layer
        self.coarse = HyperNetwork(self.in_feat, parameters_per_layer[:self.group[0]])
        self.medium = HyperNetwork(self.in_feat, parameters_per_layer[self.group[0]:sum(self.group[:2])])
        self.fine = HyperNetwork(self.in_feat, parameters_per_layer[sum(self.group[:2]):])

        # initialization, 0.
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.constant(m.weight, 0.)
                nn.init.constant(m.bias, 0.)
        self.coarse.apply(init_weights)
        self.medium.apply(init_weights)
        self.fine.apply(init_weights)

    def predict(self, fc, fm, ff):
        delta_fc = self.coarse(fc)
        delta_fm = self.medium(fm)
        delta_ff = self.fine(ff)
        delta = torch.cat([delta_fc, delta_fm, delta_ff], dim=1)
        delta_mean = torch.mean(delta, dim=0)
        return delta_mean

    def forward(self, src_txt, tgt_txt):
        """
        src_txt, tgt_txt: dict with key {shape_txt, attribute_txt, style_txt}
        """
        # encode texts and images by hierarchical modules
        # and return the feature embeddings with shape [:, 512]
        f_coarse = self.encoder(tgt_txt['shape_txt'])-self.encoder(src_txt['shape_txt'])
        f_medium = self.encoder(tgt_txt['attribute_txt'])-self.encoder(src_txt['attribute_txt'])
        f_fine = self.encoder(tgt_txt['style_txt'])-self.encoder(src_txt['style_txt'])
        # feed the feature embeddings to HyperNetworks respectively
        # to predict the parameter offsets
        delta = self.predict(f_coarse, f_medium, f_fine)

        return delta
