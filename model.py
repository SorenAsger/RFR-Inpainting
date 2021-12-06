import torch
import torch.optim as optim
from torchvision.transforms import transforms
from torch import nn

from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.Discriminator import Discriminator
from modules.RFRNet import RFRNet, VGG16FeatureExtractor, EfficientNetFeatureExtractor
import os
import time
import cv2
import numpy as np
# pip install pytorch-msssim
from pytorch_msssim import ssim


class RFRNetModel():
    def __init__(self):
        self.G = None
        self.D = None
        self.optm_D = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.normalizer = nn.Sequential(transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        self.l1_loss_val = 0.0
        self.d_loss_val = 0.0
        self.g_loss_val = 0.0
        self.g_d_loss_val = 0.0

    def initialize_model(self, path=None, train=True):
        self.D_lf = nn.BCELoss()
        self.D = Discriminator()
        self.optm_D = optim.SGD(self.D.parameters(), lr=2e-4)
        self.G = RFRNet()
        self.optm_G = optim.Adam(self.G.parameters(), lr=2e-4)
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr=2e-4)
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0

    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            self.D.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")

    def train(self, train_loader, save_path, finetune=False, iters=450000, image_save_path=None):
        #    writer = SummaryWriter(log_dir="log_info")
        self.G.train(finetune=finetune)
        if finetune:
            self.optm_G = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=5e-5)
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        while self.iter < iters:
            for items in train_loader:
                gt_images, masks = self.__cuda__(*items)
                masked_images = self.normalizer(gt_images) * masks
                # print(gt_images.data.shape)
                # print(masks.data.shape)
                # print(masked_images.data.shape)
                if image_save_path is not None and self.iter % 500 == 0:
                    masksView = torch.cat([masks], dim=1)
                    fake_B, mask = self.G(masked_images, masksView)
                    comp_B = fake_B * (1 - masksView) + gt_images * masksView
                    if not os.path.exists('{:s}/results'.format(image_save_path)):
                        os.makedirs('{:s}/results'.format(image_save_path))
                    for k in range(comp_B.size(0)):
                        grid = make_grid(comp_B[k:k + 1])
                        file_path = '{:s}/results/img_{:d}.png'.format(image_save_path, self.iter)
                        save_image(grid, file_path)

                        grid = make_grid(masked_images[k:k + 1] + 1 - masksView[k:k + 1])
                        file_path = '{:s}/results/masked_img_{:d}.png'.format(image_save_path, self.iter)
                        save_image(grid, file_path)
                self.forward(masked_images, masks, gt_images)
                self.update_parameters()
                self.iter += 1

                if self.iter % 50 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, d_loss:%.4f, G_loss:%.4f, D_G_loss:%.4f,  time_taken:%.2f" % (
                        self.iter, self.l1_loss_val / 50, self.d_loss_val / 50, self.g_loss_val / 50,
                        self.g_d_loss_val / 50, int_time))
                    s_time = time.time()
                    self.l1_loss_val = 0.0
                    self.d_loss_val = 0.0

                if self.iter % 20000 == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    save_ckpt('{:s}/g_{:d}.pth'.format(save_path, self.iter), [('generator', self.G)],
                              [('optimizer_G', self.optm_G)], self.iter)
        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/g_{:s}.pth'.format(save_path, self.iter), [('generator', self.G)],
                      [('optimizer_G', self.optm_G)], self.iter)

    def test(self, test_loader, result_save_path, max_iters):
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False
        count = 1
        l1_hole_losses = []
        l1_unmasked_losses = []
        psnr_losses = []
        ssim_losses = []
        for items in test_loader:
            if count == max_iters:
                break
            gt_images, masks = self.__cuda__(*items)
            masked_images = self.normalizer(gt_images) * masks
            # masks = torch.cat([masks], dim=1)
            fake_B, _ = self.G(masked_images, masks)
            comp_B = fake_B * (1 - masks) + gt_images * masks

            if count % 100 == 0:
                print("Iteration:%d" % count)
                if not os.path.exists('{:s}/results/{:d}'.format(result_save_path, count)):
                    os.makedirs('{:s}/results/{:d}'.format(result_save_path, count))
                for k in range(comp_B.size(0)):
                    # count += 1
                    grid = make_grid(comp_B[k:k + 1])
                    file_path = '{:s}/results/{:d}/img_{:d}.png'.format(result_save_path, count, count)
                    save_image(grid, file_path)

                    grid = make_grid(masked_images[k:k + 1] + 1 - masks[k:k + 1])
                    file_path = '{:s}/results/{:d}/masked_img_{:d}.png'.format(result_save_path, count, count)
                    save_image(grid, file_path)

                    grid = make_grid(fake_B[k:k + 1])
                    file_path = '{:s}/results/{:d}/wtf_img_{:d}.png'.format(result_save_path, count, count)
                    save_image(grid, file_path)

                    grid = make_grid(gt_images[k:k + 1])
                    file_path = '{:s}/results/{:d}/gt_img{:d}.png'.format(result_save_path, count, count)
                    save_image(grid, file_path)

                valid_loss = self.l1_loss(gt_images, fake_B, masks).item()
                hole_loss = self.l1_loss(gt_images, fake_B, (1 - masks)).item()

                # print(gt_images)
                # print(comp_B)
                # print(gt_images.shape)
                # print(comp_B.shape)
                # print(fake_B.shape)
                psnr_loss = self.psnr_loss(gt_images.detach().cpu().numpy(), comp_B.detach().cpu().numpy())
                ssim_loss = self.ssim_loss(gt_images, comp_B).item()

                psnr_losses.append(psnr_loss)
                ssim_losses.append(ssim_loss)
                l1_hole_losses.append(hole_loss)
                l1_unmasked_losses.append(valid_loss)
                with open('{:s}/results/{:d}/loss.txt'.format(result_save_path, count), "w") as f:
                    f.write(str(hole_loss))
                    f.write("\n")
                    f.write(str(valid_loss))
                    f.write("\n")
                    f.write(str(psnr_loss))
                    f.write("\n")
                    f.write(str(ssim_loss))
            count += 1
        print(np.mean(psnr_losses))
        print(np.mean(ssim_losses))
        print(np.mean(l1_hole_losses))
        print(np.mean(l1_unmasked_losses))

    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask

    def update_parameters(self):
        self.update_G()
        self.update_D()

    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()

    def update_D(self):
        self.D.train()
        self.optm_D.zero_grad()
        loss_D = self.get_d_loss()
        loss_D.backward()
        self.optm_D.step()

    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        self.D.eval()
        discriminator_fake = self.D(comp_B)
        # tgt = torch.Tensor(1).to(torch.device('cuda'))
        # loss_D_G = self.D_lf(discriminator_fake, tgt) * 0.01
        loss_D_G = -torch.log(1 - discriminator_fake + 0.0001)
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)

        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats,
                                                                                                  comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))
        loss_G = (tv_loss * 0.1
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 1
                  + hole_loss * 6) + (loss_D_G * 0.1)

        # print(loss_D_G)
        # print(loss_G - loss_D_G)
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        self.g_loss_val += loss_G.detach()
        self.g_d_loss_val += loss_D_G.detach()
        return loss_G

    def get_d_loss(self):
        real_B = self.real_B
        comp_B = self.comp_B.detach()

        discriminator_real = self.D(real_B)
        discriminator_fake = self.D(comp_B)

        # tgt0 = torch.Tensor(0).to(torch.device('cuda'))
        # tgt1 = torch.Tensor(1).to(torch.device('cuda'))
        # d_loss = self.D_lf(discriminator_fake, tgt0) + self.D_lf(discriminator_real, tgt1)

        d_loss = -(torch.log(discriminator_fake + 0.0001) + torch.log(1 - discriminator_real + 0.0001))
        self.d_loss_val += d_loss.detach()
        return d_loss

    def l1_loss(self, f1, f2, mask=1):
        return torch.mean(torch.abs(f1 - f2) * mask)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
