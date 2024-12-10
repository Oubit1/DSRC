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
from utils.BCP_utils import context_mask
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataloaders.dataset import *
from networks.VNet import UAMT
from utils import ramps, losses, test_patch
from skimage.measure import label
#rgparse是python用于解析命令行参数和选项的标准模块
#首先导入该模块；然后创建一个解析对象；
#然后向该对象中添加你要关注的命令行参数和选项
#每一个add_argument方法对应一个你要关注的参数或选项；最后调用parse_args()方法进行解析；
#解析成功之后即可使用。
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--model', type=str,  default='DMSC', help='model_name')
parser.add_argument('--exp', type=str,  default='DMSC', help='model_name')
# parser.add_argument('--model', type=str,  default='mcnet3d_v1', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=16000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
### costs
parser.add_argument('--labelnum', type=int,  default=9, help='label num')
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
patch_size = (112, 112, 80)

def context_mask1(img, mask_ratio, block_size):
    # 获取输入图像的形状信息
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    
    # 创建一个全为1的遮罩张量，该遮罩将在后续被部分置为0
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    
    # 创建一个全为1的初始遮罩，之后将部分区域置为0
    mask = torch.ones(img_x, img_y, img_z).cuda()
    
    # 计算要遮罩的区域的块数目
    num_blocks_x = int(img_x * mask_ratio / block_size)
    num_blocks_y = int(img_y * mask_ratio / block_size)
    num_blocks_z = int(img_z * mask_ratio / block_size)
    
    # 随机选择块的位置
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            for k in range(num_blocks_z):
                # 计算块的起始位置
                w = i * block_size
                h = j * block_size
                z = k * block_size
                
                # 将选定块的像素值置为0，即生成网状遮罩
                mask[w:w+block_size, h:h+block_size, z:z+block_size] = 0
                
                # 同样将loss_mask中对应位置的像素值置为0，用于后续计算损失
                loss_mask[:, w:w+block_size, h:h+block_size, z:z+block_size] = 0
    
    # 返回生成的遮罩和损失遮罩
    return mask.long(), loss_mask.long()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def get_cut_mask1(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    block_size = 8
    mask_ratio = 2/3
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    # 计算要遮罩的区域的块数目
    num_blocks_x = int(masks.shape[1] * mask_ratio / block_size)
    num_blocks_y = int(masks.shape[2] * mask_ratio / block_size)
    num_blocks_z = int( masks.shape[3] * mask_ratio / block_size)
    
    # 随机选择块的位置
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            for k in range(num_blocks_z):
                # 计算块的起始位置
                w = i * block_size
                h = j * block_size
                z = k * block_size
                
                # 将选定块的像素值置为0，即生成网状遮罩
                masks[w:w+block_size, h:h+block_size, z:z+block_size] = 0

    return masks

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
        net = UAMT(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
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
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            with torch.no_grad():
                unoutput_a = ema_model(unimg_a)
                unoutput_b = ema_model(unimg_b)
                output_c = ema_model(img_a)
                print()
                img_mask = get_cut_mask(output_c)
                # img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
                # img_mask1, loss_mask1 = context_mask1(img_a, args.mask_ratio,args.block_size)
                img_mask1 = get_cut_mask(unoutput_a, nms=1)
            """Mix Input"""
            mixl_img = img_a * img_mask1 + unimg_a * (1 - img_mask1)
            mixu_img = unimg_b * img_mask1 + img_b * (1 - img_mask1)
            mixl_lab = lab_a * img_mask1 + get_cut_mask(unoutput_a, nms=1) * (1 - img_mask1)
            mixu_lab = get_cut_mask(unoutput_b, nms=1) * img_mask1 + lab_b * (1 - img_mask1)
            outputs_l = model(mixl_img)
            outputs_u = model(mixu_img)
            mix_outputs_l = lab_a * img_mask1+ get_cut_mask(outputs_l, nms=1)* (1 - img_mask1)
            mix_outputs_u=get_cut_mask(outputs_u, nms=1) * img_mask1 +lab_b * (1 - img_mask1)
            mix_outputs_l2 =get_cut_mask(outputs_l, nms=1)* img_mask1
            mix_outputs_u2=get_cut_mask(outputs_u, nms=1) * (1 - img_mask1)
            volume_batch1 = img_a * img_mask + img_b * (1 - img_mask)
            label_batch1 = lab_a * img_mask + lab_b * (1 - img_mask)
            
            outputs = model(volume_batch)
            outputs1 = model(volume_batch1)
            consistency_weight1 = get_current_consistency_weight(iter_num//150)
            consistency_dist1 = consistency_criterion(mix_outputs_l, mixl_lab).sum() + consistency_criterion(mix_outputs_u, mixu_lab).sum()+ consistency_criterion(lab_a* img_mask1,mix_outputs_l2).sum()+ consistency_criterion(lab_b* (1 - img_mask1),mix_outputs_u2).sum()

            consistency_loss1 = consistency_weight1 * consistency_dist1
            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5*(loss_seg+loss_seg_dice)

             ## calculate the loss1
            loss_seg1 = F.cross_entropy(outputs1[:labeled_bs], label_batch1[:labeled_bs])
            outputs1_soft = F.softmax(outputs1, dim=1)
            loss_seg_dice1 = losses.dice_loss(outputs1_soft[:labeled_bs, 1, :, :, :], label_batch1[:labeled_bs] == 1)
            supervised_loss1 = 1.2*(loss_seg1+loss_seg_dice1)
            loss = supervised_loss + supervised_loss1 + consistency_loss1
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
          
            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss1, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight1, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist1, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist1.item(), consistency_weight1))    
            if iter_num % 2500 == 0:
                lr_ = base_lr
               #lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
            #    if args.dataset_name =="LA":
                dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name =args.dataset_name)
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
