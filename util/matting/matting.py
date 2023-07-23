import torch
import torch.nn as nn
from util.matting.unet import UNet
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class Matting:
    def __init__(self, model_path, out_threshold=.5, device='cuda'):
        self.model_path = model_path
        self.out_threshold = out_threshold
        self.device = device
        self.net = UNet(n_channels=3, n_classes=1).to(device)
        self.net.eval()
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.mse = nn.MSELoss()

    def generate_mask(self, img):
        """return a mask that shades the foreground"""
        output = self.net(img)
        probs = torch.sigmoid(output)
        mask = probs < self.out_threshold
        return mask

    def compute_region_loss(self, img1, img2):
        # img2: source
        with torch.no_grad():
            mask1 = self.generate_mask(img1)
            mask2 = self.generate_mask(img2)
        img1_shaded, img2_shaded = img1*mask1, img2*mask2

        return self.mse(img1_shaded, img2_shaded)


if __name__ == '__main__':
    device = 'cpu'
    matte = Matting('./MODEL.pth', device=device)
    x = torch.randn([1, 3, 1024, 1024])
    conv = nn.Conv2d(3, 3, 3, 1, 1)
    noise = torch.randn([1, 3, 1024, 1024])
    xx = conv(noise)
    loss = matte.compute_region_loss(x, xx)
    print(loss)
    loss.backward()
    print([x.grad is not None for x in conv.parameters()])
