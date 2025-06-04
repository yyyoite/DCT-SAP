import argparse
import os
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import sys

# attacks文件夹的父目录
sys.path.append('/xx/xx')

from data_load.data_loader import get_loader
from DFModels import StarGAN, AttentionGAN, FixedPointModel
from DFModels.HiSD import get_HiSD_config, get_HiSD_model, get_HiSD_img
from DFModels.AttGAN.attgan import get_args, get_AttGAN_model, get_AttGAN_img, get_AttGAN_img_occluded
from tools import *
from DiffJPEG.modules.decompression import decompress_jpeg
from DiffJPEG.modules.compression import compress_jpeg


def occlude_image(image, block_size, row, col, occlusion_value=0.5):
    occluded_image = image.clone()
    occluded_image = (occluded_image + 1) / 2
    occluded_image[:, :, row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size] = occlusion_value
    occluded_image = occluded_image * 2 - 1
    return occluded_image

                        

def Pos_Select(y_dct, cb_dct, cr_dct, y_qt, c_qt, T_gamma, T_lambda, compress, decompress, device):
    dct0 = torch.zeros((8,8))
    spatail = torch.zeros((64, 8, 8))
    dct0 = dct0.to(device)
    spatail = spatail.to(device)
    for i in range(8):
        for j in range(8):
            dct = dct0.clone()
            dct[i, j] = 1
            temp = dct * y_qt
            temp = temp.unsqueeze(0).unsqueeze(0)
            res = decompress.idct(temp)-128
            res = res.squeeze(0).squeeze(0)
            spatail[j + i * 8, :, :] = res

    pos = torch.ones_like(y_dct)
    pos[0, :, 0, 0] = 0


    spa_uq = decompress(y_dct, cb_dct, cr_dct, y_qt, c_qt)
    spa_uq = torch.round(spa_uq)

    
    y_lossy, _, _ = compress.l1(spa_uq)
    y_lossy = compress.l2(y_lossy)

    
    dct_dif = y_dct * y_qt - y_lossy

    
    dis_to_border = torch.zeros_like(dct_dif)
    for i in range(dct_dif.shape[1]):
        dis_to_border[0, i, :, :] = y_qt/2 - torch.abs(dct_dif[0, i, :, :])


    pos[dis_to_border < T_gamma] = 0

    b = decompress.y_dequantize(y_dct, y_qt)
    b = decompress.idct(b)-128
    b = decompress.merging(b, decompress.height, decompress.width)

    m_block = int(b.shape[1] / 8)
    n_block = int(b.shape[2] / 8)

    for m in range(m_block):
        for n in range(n_block):
            for i in range(8):
                for j in range(8):
                    y_temp = b[0, m*8:(m+1)*8, n*8:(n+1)*8] + spatail[j+i*8, :, :]
                    if not (torch.all((y_temp >= -127 + T_lambda) & (y_temp <= 128-T_lambda))):
                        pos[0, m*m_block+n, i, j] = 2


                    y_temp = b[0, m*8:(m+1)*8, n*8:(n+1)*8] - spatail[j+i*8, :, :]
                    if  not (torch.all((y_temp >= -127 + T_lambda) & (y_temp <= 128-T_lambda))):
                        pos[0, m*m_block+n, i, j] = 3
    return pos


class WeightedDCTDis(nn.Module):
    def __init__(self, weight_type='log', epsilon=0.001, alpha=0.1):
        super(WeightedDCTDis, self).__init__()
        self.weight_type = weight_type
        self.epsilon = epsilon
        self.alpha = alpha

    def _generate_weight_matrix(self, qt):
        if self.weight_type == 'log':
            return self._generate_log_weight_matrix(qt)
        elif self.weight_type == 'exp':
            return self._generate_exp_weight_matrix(qt)
        else:
            raise ValueError("Unsupported weight_type. Use 'log' or 'exp'.")
        
    def _generate_log_weight_matrix(self, qt):
        weight_matrix = 1 / torch.log(qt + self.epsilon)
        weight_matrix /= torch.max(weight_matrix)
        return weight_matrix

    def _generate_exp_weight_matrix(self, qt):
        weight_matrix = torch.exp(-self.alpha * qt)
        weight_matrix /= torch.max(weight_matrix)
        return weight_matrix
    
    def forward(self, ori_dct, adv_dct, qt):
        dct_diff = ori_dct - adv_dct

        weight_matrix = self._generate_weight_matrix(qt)
        

        weighted_dct_diff = weight_matrix * torch.abs(dct_diff)
        
        loss = torch.sum(weighted_dct_diff)
        return loss


class CustomLoss(nn.Module):
    def __init__(self, weight1, weight2):
        super(CustomLoss, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2
        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = WeightedDCTDis()

    def forward(self, adv_df, ori_df, adv_dct, ori_dct, qt):
        loss1 = self.loss_fn1(adv_df, ori_df)
        loss2 = self.loss_fn2(ori_dct, adv_dct, qt)
        total_loss = self.weight1 * loss1 - self.weight2 * loss2
        return total_loss


def attack(config, device, data_loader):
    if config.model == 'StarGAN':
        G = StarGAN.Generator(64, 5, 6)
        G = G.to(device)
        load_model_weights(G, config.StarGAN_path)
    elif config.model == 'AttentionGAN':
        G = AttentionGAN.Generator(64, 5, 6)
        G = G.to(device)
        load_model_weights(G, config.AttentionGAN_path)
    elif config.model == 'HiSD':
        HiSD_opts, HiSD_config = get_HiSD_config()
        HiSD_model = get_HiSD_model(HiSD_opts, HiSD_config)
    elif config.model == 'AttGAN':
        attgan_args = get_args()
        attgan = get_AttGAN_model(attgan_args)
    elif config.model == 'Fixed-PointGAN':
        G = FixedPointModel.Generator()
        G = G.to(device)
        G.load_state_dict(torch.load(config.FixedPointGAN_path, map_location=lambda storage, loc: storage))


    print('Loading the victim models')

    eps = config.epsilon
    alpha = config.alpha
    iter = config.iter

    loss_fn = CustomLoss(config.weight1, config.weight2).to(device)

    compress = compress_jpeg()
    decompress = decompress_jpeg(config.image_size, config.image_size)
    compress = compress.to(device)
    decompress = decompress.to(device)

    for i, (x, c_org, filename, x_dct, x_qt) in enumerate(tqdm(data_loader)):
        x_dct = x_dct[0]
        x_qt = x_qt[0]

        filename = str(filename[0])
        x = x.to(device)
        c_trg_list = create_labels(c_org, config.c_dim, 'CelebA', config.selected_attrs)

        y = torch.from_numpy(x_dct['y']).to(torch.float32)
        y = y.view(1, 32 * 32, 8, 8)
        cb = torch.from_numpy(x_dct['cb']).to(torch.float32)
        cb = cb.view(1, 16 * 16, 8, 8)
        cr = torch.from_numpy(x_dct['cr']).to(torch.float32)
        cr = cr.view(1, 16 * 16, 8, 8)

        y = y.clone().detach_().to(device)
        cb = cb.clone().detach_().to(device)
        cr = cr.clone().detach_().to(device)

        y_qt = torch.from_numpy(x_qt[0].astype(np.float32)).to(torch.float32)
        c_qt = torch.from_numpy(x_qt[1].astype(np.float32)).to(torch.float32)
        y_qt = y_qt.clone().detach_().to(device)
        c_qt = c_qt.clone().detach_().to(device)

        if config.model == 'AttGAN':
            c_org = c_org.to(device)

        with torch.no_grad():
            if config.model in ['StarGAN', 'AttentionGAN']:
                ori_df = []
                for c in c_trg_list:
                    df = G(x, c)
                    ori_df.append(df)
                ori_df = torch.cat(ori_df, dim=3)

            elif config.model == 'HiSD':
                ori_df = []
                for c in range(5):
                    if c == 0 or c == 1:
                        steps = [{'type': 'latent-guided', 'tag': c, 'attribute': 0, 'seed': 500}]
                    elif c > 1:
                        steps = [{'type': 'latent-guided', 'tag': 2, 'attribute': c-2, 'seed': 500}]
                    df = get_HiSD_img(HiSD_model, x, steps, HiSD_config)
                    ori_df.append(df)
                ori_df = torch.cat(ori_df, dim=3)
                ori_df = (ori_df - torch.min(ori_df).item()) / (torch.max(ori_df).item() - torch.min(ori_df).item())
                ori_df = ori_df * 2 - 1

            elif config.model == 'AttGAN':
                ori_df = []
                for c in range(len(config.selected_attrs)):
                    df = get_AttGAN_img(attgan, x, c_org, attgan_args, c)
                    ori_df.append(df)

                ori_df = torch.cat(ori_df, dim=3)
                ori_df = ori_df.to(device)
            elif config.model == 'Fixed-PointGAN':
                ori_df = []
                for c_trg in c_trg_list:
                    ori_df.append(torch.tanh(x + G(x, c_trg)))
                ori_df = torch.cat(ori_df, dim=3)

            pos = Pos_Select(y, cb, cr, y_qt, c_qt, config.T_gamma, config.T_lambda, compress, decompress, device)
            pos = pos.to(device)


            change = torch.zeros((32, 32)).to(device)
            occluded_images = []

            for row in range(32):
                for col in range(32):
                    occluded_images.append(occlude_image(x, 8, row, col))

            occluded_images = torch.stack(occluded_images).to(device)
            occluded_images = occluded_images.squeeze(1)
            occluded_dfs = []

            occluded_batch_size = 32

            with torch.no_grad():
                for i in range(0, occluded_images.size(0), occluded_batch_size):
                    batch_occluded_images = occluded_images[i:i + occluded_batch_size]

                    if config.model in ['StarGAN', 'AttentionGAN']:
                        batch_occluded_dfs = []
                        for c in c_trg_list:
                            c_repeated = c.repeat(batch_occluded_images.size(0), 1)
                            batch_occluded_dfs.append(G(batch_occluded_images, c_repeated))
                        batch_occluded_dfs = torch.cat(batch_occluded_dfs, dim=3)
                        occluded_dfs.append(batch_occluded_dfs)
                    
                    elif config.model == 'HiSD':
                        batch_occluded_dfs = []
                        for c in range(5):
                            steps = [{'type': 'latent-guided', 'tag': c if c <= 1 else 2, 'attribute': 0 if c <= 1 else c-2, 'seed': 500}]
                            occluded_df = get_HiSD_img(HiSD_model, batch_occluded_images, steps, HiSD_config)
                            batch_occluded_dfs.append(occluded_df)
                        batch_occluded_dfs = torch.cat(batch_occluded_dfs, dim=3)
                        batch_occluded_dfs = (batch_occluded_dfs - torch.min(batch_occluded_dfs).item()) / (torch.max(batch_occluded_dfs).item() - torch.min(batch_occluded_dfs).item())
                        batch_occluded_dfs = batch_occluded_dfs * 2 - 1
                        occluded_dfs.append(batch_occluded_dfs)

                    elif config.model == 'AttGAN':
                        batch_occluded_dfs = []
                        for c in range(len(config.selected_attrs)):
                            df = get_AttGAN_img_occluded(attgan, batch_occluded_images, c_org, attgan_args, c)
                            batch_occluded_dfs.append(df)
                        batch_occluded_dfs = torch.cat(batch_occluded_dfs, dim=3)
                        occluded_dfs.append(batch_occluded_dfs.to(device))

                    elif config.model == 'Fixed-PointGAN':
                        batch_occluded_dfs = []
                        for c in c_trg_list:
                            c_repeated = c.repeat(batch_occluded_images.size(0), 1)
                            batch_occluded_dfs.append(torch.tanh(batch_occluded_images + G(batch_occluded_images, c_repeated)))
                        batch_occluded_dfs = torch.cat(batch_occluded_dfs, dim=3)
                        occluded_dfs.append(batch_occluded_dfs)
                    
            occluded_dfs = torch.cat(occluded_dfs, dim=0)

            ori_df_repeated = ori_df.repeat(1024, 1, 1, 1)
            mse = F.mse_loss(ori_df_repeated, occluded_dfs, reduction='none').mean(dim=(1,2,3))
            change = mse.view(32, 32)
            threshold = torch.quantile(change, 1 - config.Select2Threshold)
            for row in range(32):
                for col in range(32):
                    if change[row][col] < threshold:
                        pos[:, row*32+col, :, :] = 0
            

            adv = y[pos == 1]
            adv = adv.clone().detach_().to(device) + torch.tensor(np.random.uniform(-eps, eps, adv.shape).astype('float32')).to(device)


        for j in range(iter):
            adv.requires_grad = True

            y_adv = y.clone().detach_()
            y_adv[pos==1] = adv
            adv_image = decompress(y_adv, cb.clone().detach_(), cr.clone().detach_(), y_qt.clone().detach_(), c_qt.clone().detach_())
            adv_image = adv_image/255 * 2 - 1


            if config.model in ['StarGAN', 'AttentionGAN']:
                adv_df = []
                for c in c_trg_list:
                    df = G(adv_image, c)
                    adv_df.append(df)
                adv_df = torch.cat(adv_df, dim=3)

            elif config.model == 'HiSD':
                adv_df = []
                for c in range(5):
                    if c == 0 or c == 1:
                        steps = [{'type': 'latent-guided', 'tag': c, 'attribute': 0, 'seed': 500}]
                    elif c > 1:
                        steps = [{'type': 'latent-guided', 'tag': 2, 'attribute': c-2, 'seed': 500}]
                    df = get_HiSD_img(HiSD_model, adv_image, steps, HiSD_config)
                    adv_df.append(df)
                adv_df = torch.cat(adv_df, dim=3)
                adv_df = (adv_df - torch.min(adv_df).item()) / (torch.max(adv_df).item() - torch.min(adv_df).item())
                adv_df = adv_df * 2 - 1

            elif config.model == 'AttGAN':
                adv_df = []
                for c in range(len(config.selected_attrs)):
                    df = get_AttGAN_img(attgan, adv_image, c_org, attgan_args, c)
                    adv_df.append(df)

                adv_df = torch.cat(adv_df, dim=3)
                adv_df = adv_df.to(device)
            elif config.model == 'Fixed-PointGAN':
                adv_df = []
                for c_trg in c_trg_list:
                    adv_df.append(torch.tanh(adv_image + G(adv_image, c_trg)))
                adv_df = torch.cat(adv_df, dim=3)

            if config.model in ['StarGAN', 'AttentionGAN']:
                G.zero_grad()
            elif config.model == 'HiSD':
                HiSD_model.zero_grad()
            elif config.model == 'FixedPoint-GAN':
                G.zero_grad()

            decompress.zero_grad()

            loss = loss_fn(adv_df, ori_df, y_adv, y, y_qt)
            print('total loss: ', loss)

            loss.backward()
            grad_sign = adv.grad.data.sign()
            adv = adv + grad_sign * alpha
            adv = torch.clamp(adv, min=y[pos==1]-eps, max=y[pos==1]+eps).clone().detach_()
        
        if config.model in ['StarGAN', 'AttentionGAN']:
            G.zero_grad()
        elif config.model == 'HiSD':
            HiSD_model.zero_grad()
        elif config.model == 'FixedPoint-GAN':
                G.zero_grad()

        compress.zero_grad()

        y[pos==1] = adv
        x_adv = decompress(y, cb, cr, y_qt, c_qt)
        x_adv = x_adv/255
        if not config.save_jpeg:
            filename = os.path.splitext(filename)[0] + ".png"
        save_path = os.path.join(config.result_dir, filename)
        torchvision.utils.save_image(x_adv, save_path)
        torch.cuda.empty_cache()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--iter', type=int, default=20, help='attack iters')
    parser.add_argument('--epsilon', type=float, default=1, help='epsilon')
    parser.add_argument('--alpha', type=float, default=0.06, help='learning rate')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels')
    parser.add_argument('--save_jpeg', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--T_gamma', type=float, default=0)
    parser.add_argument('--T_lambda', type=float, default=0)
    parser.add_argument('--Select2Threshold', type=float, default=0.7)
    
    # 路径
    parser.add_argument('--data_path', type=str, default='/xx', help='dataset')
    parser.add_argument('--attr_path', type=str, default='/xx', help='attr')
    parser.add_argument('--result_dir', type=str, default='/xx', help='result')
    
    # 模型
    parser.add_argument('--model', type=str, choices=['StarGAN', 'AttentionGAN', 'AttGAN', 'HiSD', 'Fixed-PointGAN'], default='StarGAN')
    parser.add_argument('--StarGAN_path', type=str, default='/xx', help='StarGAN')
    parser.add_argument('--AttentionGAN_path', type=str, default='/xx', help='AttentionGAN')
    parser.add_argument('--HiSD_path', type=str, default='/xx')
    parser.add_argument('--AttGAN_path', type=str, default='/xx')
    parser.add_argument('--FixedPointGAN_path', type=str, default='/xx')

    parser.add_argument('--weight1', type=float)
    parser.add_argument('--weight2', type=float)

    config = parser.parse_args()

    os.makedirs(config.result_dir, exist_ok=True)
    # 加载数据
    celeba_loader = get_loader(config.data_path, config.attr_path, config.selected_attrs)
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    attack(config, device, celeba_loader)


if __name__ == '__main__':
    main()
