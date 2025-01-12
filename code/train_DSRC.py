import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.BCP_utils import context_mask, context_mask1,concate_mask
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataloaders.dataset import *
from networks.VNet import VNet
from utils import ramps, losses, test_patch
from skimage.measure import label
#rgparse是python用于解析命令行参数和选项的标准模块
#首先导入该模块；然后创建一个解析对象；
#然后向该对象中添加你要关注的命令行参数和选项
#每一个add_argument方法对应一个你要关注的参数或选项；最后调用parse_args()方法进行解析；
#解析成功之后即可使用。
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str,  default='Pancreas_CT', help='dataset_name')
parser.add_argument('--model', type=str,  default='DSRC', help='model_name')
parser.add_argument('--exp', type=str,  default='DSRC', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=16000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--labelnum', type=int,  default=12, help='label num')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
parser.add_argument('--mask_ratio1', type=float, default=4/5, help='ratio of mask1/image')
parser.add_argument('--block_size', type=float, default=8, help='size of mask block')
args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum, args.model)
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path+'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path+'data/Pancreas_h5'
    args.max_samples = 62
train_data_path = args.root_path
torch.cuda.is_available()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)# 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(args.seed) # 为所有的GPU设置种子，以使得结果是确定的
num_classes = 2
# 绘制熵的指定切片
def plot_entropy_map_fixed_slice1(entropy_map, slice_index, title='Entropy Map (Fixed Slice)', save_path=None):
    """
    绘制熵值的指定切片，并保存为图片。
    :param entropy_map: 熵值张量，形状应为[Depth, Height, Width]。
    :param slice_index: 指定的切片索引。
    :param title: 图表标题。
    :param save_path: 保存图像的路径。如果为 None，则不会保存图像。
    """
    # 提取指定切片，形状为 [Height, Width]
    middle_slice = entropy_map[slice_index, :, :]  
    # 转换为 numpy 数组
    middle_slice = middle_slice.cpu().detach().numpy()
    # 绘制图像
    plt.figure(figsize=(100, 100))  # 调整图片大小，适应大尺寸图像
    plt.imshow(middle_slice, cmap='Blues', interpolation='nearest')  # 使用单一颜色映射
    plt.colorbar()  # 添加颜色条
    plt.title(title, fontsize=16)  # 设置标题
    # 在网格上标注数值
    for i in range(middle_slice.shape[0]):
        for j in range(middle_slice.shape[1]):
            plt.text(j, i, f"{middle_slice[i, j]:.2f}", ha='center', va='center', color="black", fontsize=6)
    # 添加网格线
    plt.xticks(np.arange(-0.5, middle_slice.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, middle_slice.shape[0], 1), [])
    plt.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    plt.gca().set_xticks(np.arange(-0.5, middle_slice.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, middle_slice.shape[0], 1), minor=True)

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到：{save_path}")
    # 显示图像
    plt.show()

# 计算并绘制熵的概率映射图并保存为文件
def plot_entropy_map_fixed_slice(entropy_map, slice_index, title='Entropy Map (Fixed Slice)', save_path=None):
    """
    绘制熵值的指定切片，并保存为图片。
    :param entropy_map: 熵值张量，形状应为[Depth, Height, Width]。
    :param slice_index: 指定的切片索引。
    :param title: 图表标题。
    :param save_path: 保存图像的路径。如果为 None，则不会保存图像。
    """
    # 提取指定切片
    middle_slice = entropy_map[slice_index, :, :]  # 提取指定切片，形状为 [Height, Width]
    # 转换为 numpy 数组
    middle_slice = middle_slice.cpu().detach().numpy()
    # 绘制图像
    plt.figure(figsize=(6, 6))  # 设置图片大小
    plt.imshow(middle_slice, cmap='Blues', interpolation='nearest')  # 单一颜色映射 'Blues'
    plt.colorbar()
    plt.title(title)
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到：{save_path}")
    # 显示图像
    plt.show()

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()
def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                        split='train',
                        transform = transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                            ]))
        db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
        db_test = Pancreas(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    #labeled_idxs = list(range(16))
    #unlabeled_idxs = list(range(16, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    best_dice = 0
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            img_a, img_b = volume_batch[:1], volume_batch[1:2]
            lab_a, lab_b = label_batch[:1], label_batch[1:2]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:3], volume_batch[3:]
            with torch.no_grad():
                unoutput_a = ema_model(unimg_a)
                unoutput_b = ema_model(unimg_b)
                output_c = ema_model(img_a)
                img_mask = get_cut_mask(output_c)
                # img_mask, loss_mask = concate_mask(img_a)
                # img_mask1, loss_mask1 = concate_mask(unimg_a)
                img_mask1 = get_cut_mask(unoutput_a, nms=1)
            mixl_img = img_a * img_mask1 + unimg_a * (1 - img_mask1)
            mixu_img = unimg_b * img_mask1 + img_b * (1 - img_mask1)
            # mixl_lab = lab_a * img_mask1 + get_cut_mask(unoutput_a, nms=1) * (1 - img_mask1)
            # mixu_lab = get_cut_mask(unoutput_b, nms=1) * img_mask1 + lab_b * (1 - img_mask1)

            # 使用 torch.cat 在第 0 维（batch维度）拼接
            unlabeled_volume_batch1 = torch.cat((mixl_img, mixu_img), dim=0)
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            noise1 = torch.clamp(torch.randn_like(unlabeled_volume_batch1) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            ema1_inputs = unlabeled_volume_batch1 + noise1
            volume_batch1 = img_a * img_mask + img_b * (1 - img_mask)
            label_batch1 = lab_a * img_mask + lab_b * (1 - img_mask)
            outputs =model(volume_batch)
            outputs1 = model(volume_batch1)
            outputs_l = model(unlabeled_volume_batch1)
              # 计算错误预测体素数量
            incorrect_voxels = torch.sum(torch.abs(outputs.argmax(dim=1) - label_batch))  # 假设标签是0/1二值的

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output1 = ema_model(ema1_inputs)
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, 96, 96, 96]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, 96, 96, 96)
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)
            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5*(loss_seg+loss_seg_dice)
             ## calculate the loss1
            loss_seg1 = F.cross_entropy(outputs1[:labeled_bs], label_batch1[:labeled_bs])
            outputs1_soft = F.softmax(outputs1, dim=1)
            loss_seg_dice1 = losses.dice_loss(outputs1_soft[:labeled_bs, 1, :, :, :], label_batch1[:labeled_bs] == 1)
            supervised_loss1 = 0.5*(loss_seg1+loss_seg_dice1)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist1 = consistency_criterion(outputs_l, ema_output1)
            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output) #(batch, 2, 112,112,80)
            consistency_dist2 = consistency_criterion((outputs_l[0, 1, :, :, :].float().unsqueeze(0))*img_mask1, lab_a * img_mask1.float())
            # # 假设 probabilities 是 Softmax 后的概率分布
            # Step 1: 计算 outputs_l 和 unoutput_a 的不确定性（基于熵）
            outputs_l_softmax = F.softmax(outputs_l[0, :, :, :, :].float() * (1 - img_mask1), dim=0)
            unoutput_a_softmax = F.softmax(unoutput_a.float() * (1 - img_mask1), dim=1)
            unoutput_a_softmax = unoutput_a_softmax.squeeze(0) 
            # 计算熵：- sum(p * log(p))
            entropy_outputs_l = -torch.sum(outputs_l_softmax * torch.log(outputs_l_softmax + 1e-8), dim=0)
            entropy_unoutput_a = -torch.sum(unoutput_a_softmax * torch.log(unoutput_a_softmax + 1e-8), dim=0)
            # Step 2: 比较熵，选择更低不确定性的结果作为伪标签
            mask_better_outputs_l = (entropy_outputs_l < entropy_unoutput_a).float()  # outputs_l 更可信
            mask_better_unoutput_a = (1 - mask_better_outputs_l)  # unoutput_a 更可信

            # print("outputs_l[1, :, :, :, :].shape:", outputs_l[1, :, :, :, :].shape)
            # 生成伪标签 P_conf
            P_conf = mask_better_outputs_l.unsqueeze(0) * outputs_l[0, :, :, :, :].float() * (1 - img_mask1) + \
                    mask_better_unoutput_a.unsqueeze(0) * unoutput_a.float() * (1 - img_mask1)
            P_conf = P_conf.squeeze(0)  # 去掉维度1，变成[2, 112, 112, 80]
            # Step 3: 计算 consistency_criterion
            consistency_dist2_1 = consistency_criterion(
                outputs_l[0, :, :, :, :].float() * (1 - img_mask1), P_conf * (1 - img_mask1)
            )
            unoutput_a = unoutput_a.squeeze(0)
            consistency_dist2_2 = consistency_criterion(
                unoutput_a.float() * (1 - img_mask1), P_conf * (1 - img_mask1)
            )
            # 累加 consistency_dist2
            consistency_dist2 = consistency_dist2 + consistency_dist2_1 + consistency_dist2_2
            # # consistency_dist2 = consistency_dist2 + consistency_criterion(outputs_l[0, :, :, :, :].float()*(1 - img_mask1), ((unoutput_a.float().squeeze())* (1 - img_mask1)))
            # Step 1: 计算 outputs_l 和 unoutput_a 的不确定性（基于熵）
            outputs_u_softmax = F.softmax(outputs_l[1, :, :, :, :].float() * img_mask1, dim=0)
            unoutput_b_softmax = F.softmax(unoutput_b.float() * img_mask1, dim=1)
            unoutput_b_softmax = unoutput_b_softmax.squeeze(0) 
            # 计算熵：- sum(p * log(p))
            entropy_outputs_u = -torch.sum(outputs_u_softmax * torch.log(outputs_u_softmax + 1e-8), dim=0)
            entropy_unoutput_b = -torch.sum(unoutput_b_softmax * torch.log(unoutput_b_softmax + 1e-8), dim=0)
            # 确定统一的切片索引
            # 比如选择两者深度的中间索引
            depth_outputs_u = entropy_outputs_u.shape[0]
            depth_unoutput_b = entropy_unoutput_b.shape[0]
            slice_index = min(depth_outputs_u, depth_unoutput_b) // 2  # 统一选择中间切片索引
            #Step 2: 比较熵，选择更低不确定性的结果作为伪标签
            mask_better_outputs_u = (entropy_outputs_u < entropy_unoutput_b).float()  # outputs_l 更可信
            mask_better_unoutput_b = (1 - mask_better_outputs_u)  # unoutput_a 更可信
            # print("outputs_l[1, :, :, :, :].shape:", outputs_l[1, :, :, :, :].shape)
            # 生成伪标签 P_conf
            P_conf2 = mask_better_outputs_u.unsqueeze(0) * outputs_l[1, :, :, :, :].float() * img_mask1 + \
                    mask_better_unoutput_b.unsqueeze(0) * unoutput_b.float() * img_mask1
            P_conf2 = P_conf2.squeeze(0)  # 去掉维度1，变成[2, 112, 112, 80]
            # Step 3: 计算 consistency_criterion
            consistency_dist2_3 = consistency_criterion(outputs_l[1, :, :, :, :].float() * img_mask1, P_conf2 *img_mask1 )
            unoutput_b = unoutput_b.squeeze(0)
            consistency_dist2_4 = consistency_criterion(unoutput_b.float() * img_mask1, P_conf2 * img_mask1)
            # 累加 consistency_dist2
            consistency_dist2 = consistency_dist2 + consistency_dist2_3 + consistency_dist2_4
            # consistency_dist2 = consistency_dist2 + consistency_criterion(outputs_l[1, :, :, :, :].float()*img_mask1, ((unoutput_b.float().squeeze())*img_mask1).squeeze())
            consistency_dist2 = consistency_dist2 + consistency_criterion((outputs_l[1, 1, :, :, :].float().unsqueeze(0))*(1 - img_mask1),lab_b.float()*(1 - img_mask1))
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            mask = (uncertainty<threshold).float()
            consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            consistency_dist1 = torch.sum(mask*consistency_dist1)/(2*torch.sum(mask)+1e-16)
            consistency_dist2 = torch.sum(mask*consistency_dist2)/(2*torch.sum(mask)+1e-16)
            # print("supervised_loss1",supervised_loss1)
            consistency_loss = consistency_weight *(consistency_dist+ consistency_dist1+ consistency_dist2)
            loss = supervised_loss + consistency_loss + supervised_loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('loss/supervised_loss1', supervised_loss1, iter_num)
            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('loss/consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('loss/consistency_dist1', consistency_dist1, iter_num)
            writer.add_scalar('loss/consistency_dist2', consistency_dist2, iter_num)
                 # 记录错误预测体素数量到TensorBoard
            writer.add_scalar('Incorrect_Voxels', incorrect_voxels, iter_num)
  
            if iter_num % 2500 == 0:
                lr_ = base_lr
               #lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num >= 800 and iter_num % 100 == 0:
                model.eval()       
                if args.dataset_name =="LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name =args.dataset_name)
                if args.dataset_name =="Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name =args.dataset_name)
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.exp))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
