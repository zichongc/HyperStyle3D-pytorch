import os
import clip
from util.arcface.model_irse import Backbone
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


class IDLoss(nn.Module):
    def __init__(self, model_path=r'util/pretrained_models/model_ir_se50.pth'):
        super(IDLoss, self).__init__()
        # print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, predict, target, n_samples):
        pred_feats = self.extract_feats(predict)
        target_feats = self.extract_feats(target).detach()
        loss = 0.
        cnt = 0
        for i in range(n_samples):
            diff = pred_feats[i].dot(target_feats[i])
            loss += 1. - diff
            cnt += 1

        return loss / cnt


class DirectionLoss(torch.nn.Module):
    def __init__(self, loss_type='cosine'):
        super(DirectionLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        return self.loss_func(x, y)


class CLIPLoss(nn.Module):
    def __init__(self, pretrain=r'util/pretrained_models', device='cuda'):
        super().__init__()
        self.model, _ = clip.load('ViT-B/32', device=device, download_root=pretrain)
        self.device = device
        self.sim = nn.CosineSimilarity()
        self.preprocessor = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.direction_loss = DirectionLoss()
        self.delta_text = None

    def get_text_features(self, source_txt, target_txt, norm=True):
        with torch.no_grad():
            src_token = clip.tokenize(source_txt).to(self.device)
            tgt_token = clip.tokenize(target_txt).to(self.device)
            src_txt_feat = self.model.encode_text(src_token).detach()
            tgt_txt_feat = self.model.encode_text(tgt_token).detach()

            if norm:
                src_txt_feat /= src_txt_feat.norm(dim=-1, keepdim=True)
                tgt_txt_feat /= tgt_txt_feat.norm(dim=-1, keepdim=True)

        return src_txt_feat, tgt_txt_feat

    def transform(self, x):
        return Variable(x.to(torch.float32), requires_grad=True).to(self.device)

    def get_image_features(self, source_img, target_img, norm=True):
        tgt_img = self.preprocessor(target_img).unsqueeze(0)
        tgt_img = tgt_img.to(self.device)
        tgt_img_feat = self.model.encode_image(tgt_img)
        if norm:
            tgt_img_feat /= tgt_img_feat.clone().norm(dim=-1, keepdim=True)

        with torch.no_grad():
            src_img = self.preprocessor(source_img).unsqueeze(0)
            src_img = src_img.to(self.device)
            src_img_feat = self.model.encode_image(src_img)
            if norm:
                src_img_feat /= src_img_feat.clone().norm(dim=-1, keepdim=True)

        return src_img_feat, tgt_img_feat

    def compute_text_direction(self, source_txt, target_txt, norm=True):
        src_txt_feat, tgt_txt_feat = self.get_text_features(source_txt, target_txt, norm=norm)
        direction = (tgt_txt_feat - src_txt_feat).mean(axis=0, keepdim=True)
        direction /= direction.norm(dim=-1, keepdim=True)

        return direction

    def forward(self, target_img, source_img, target_text, source_text, n_samples):
        if self.delta_text is None:
            self.delta_text = self.compute_text_direction(source_text, target_text)
        eps = 1e-6
        loss, cnt = 0., 0

        for i in range(n_samples):
            src_img_feat, tgt_img_feat = self.get_image_features(source_img[i], target_img[i])
            delta_image = (tgt_img_feat-src_img_feat)
            if delta_image.sum() == 0:
                tgt_img = target_img[i]+eps
                src_img_feat, tgt_img_feat = self.get_image_features(source_img[i], tgt_img)
                delta_image = (tgt_img_feat-src_img_feat)

            delta_image /= delta_image.clone().norm(dim=-1, keepdim=True)
            loss += self.direction_loss(self.delta_text, delta_image)
            cnt += 1

        return loss / cnt

#
# def REGIONLoss(target_img, source_img):
#     """"""


# if __name__ == '__main__':
#     clip_loss(None, None, None, None)
