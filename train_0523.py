import sys, os
if "ck696" in os.getcwd():
    sys.path.append("/share/hariharan/ck696/allclear")
else:
    sys.path.append("/share/hariharan/cloud_removal/allclear")
    
os.environ["WANDB_API_KEY"] = "5f04d2ce100707f23b71379f67f28901d496edda"
# os.environ["WANDB_MODE"] = "disabled"

import warnings
from utils import set_requires_grad, get_rgb, GANLoss
from utils import *
from tensorboardX import SummaryWriter
import tqdm
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_new import Sen2_MTC
from model.fe import FeatureExtractor
from model.pmaa import PMAA
from model.discriminator import Discriminator
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb

# use logger 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser()
"""Path"""
# parser.add_argument("--root", type=str, default='/share/hariharan/ck696',
parser.add_argument("--root", type=str, default= '/share/hariharan/cloud_removal', 
                    choices=['/scratch/allclear', '/share/hariharan/ck696'],
                    help="Path to dataset")
parser.add_argument("--root_file", type=str, default='/share/hariharan/ck696/allclear/baselines/PMAA/data', 
                    help="Path to dataset")
parser.add_argument("--cloud_model_path", type=str,
                    default='./data/Feature_Extrator_FS2.pth', help="path to feature extractor model")
parser.add_argument("--save_model_path", type=str,
                    default='./checkpoints0515', help="Path to save model")
parser.add_argument("--dataset_name", type=str, choices=["CTGAN_Sen2_MTC", "AllClear"],
                    default='AllClear', help="name of the dataset")
parser.add_argument("--load_gen", type=str, default='',
                    help="path to the model of generator")
parser.add_argument("--load_dis", type=str, default='',
                    help="path to the model of discriminator")

"""Parameters"""
parser.add_argument("--n_epochs", type=int,
                    default=100, help="Number of epochs")
parser.add_argument("--gan_mode", type=str, default='lsgan',
                    help="Which gan mode(lsgan/vanilla)")
parser.add_argument("--optimizer", type=str, default='AdamW',
                    help="optimizer you want to use(AdamW/SGD)")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--workers", type=int, default=2,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int,
                    default=4, help="size of the batches")
parser.add_argument('--lambda_L1', type=float,
                    default=100.0, help='weight for L1 loss')
parser.add_argument('--lambda_aux', type=float,
                    default=50.0, help='weight for aux loss')
parser.add_argument("--in_channel", type=int, default=4,
                    help="the number of input channels")
parser.add_argument("--out_channel", type=int, default=4,
                    help="the number of output channels")
parser.add_argument("--image_size", type=int,
                    default=256, help="crop size")
parser.add_argument("--aux_loss", action='store_true',
                    help="whether use auxiliary loss(1/0)")
parser.add_argument("--label_noise", action='store_true',
                    help="whether to add noise on the label of gan training")
parser.add_argument("--masked_L1", type=int, default=0,
                    help="For AllClear, use masked the lsos by the target cloud mask")
parser.add_argument("--pred_cloud", type=int, default=0,
                    help="For AllClear, use masked the lsos by the target cloud mask")

"""base_options"""
parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
parser.add_argument("--manual_seed", type=int,
                    default=2022, help="random_seed you want")

opt, _ = parser.parse_known_args()
print(opt)
logger.info(opt)

if opt.dataset_name == "CTGAN_Sen2_MTC": dataset_name_ = "SEN2MTC"
if opt.dataset_name == "AllClear": dataset_name_ = "AllClear"   
run_name = f"PMAA_{dataset_name_}_lm{int(opt.lambda_L1)}_la{int(opt.lambda_aux)}_bs{opt.batch_size}_lr{opt.lr}_0523"
if opt.dataset_name == "AllClear": 
    if opt.masked_L1 == 0:
        run_name += "_AddCldMask"
    elif opt.masked_L1 == 1:
        run_name += "_AddCldMaskedL1"
    if opt.pred_cloud == 1:
        run_name += "_PredCloud"
wandb.init(project="allclear-pmaa-v1", name=run_name, config=opt)

os.makedirs(os.path.join(opt.save_model_path,
            run_name), exist_ok=True)
fixed_seed(opt.manual_seed)

if opt.dataset_name == "AllClear":

    from dataset.dataloader_v1 import CRDataset
    from torch.utils.data import DataLoader

    import json
    with open('/scratch/allclear/metadata/v3/test_tx3_v1.json') as f:
        metadata = json.load(f)
    #len=3168
    train_data = CRDataset(metadata, 
                        selected_rois="all", 
                        main_sensor="s2_toa", 
                        aux_sensors=[],
                        aux_data=["cld_shdw"],
                        format="stp",
                        target="s2p",
                        tx=3)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    
elif opt.dataset_name == "CTGAN_Sen2_MTC":
    # len=1190
    train_data = Sen2_MTC(opt, 'train')
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.workers, drop_last=True, pin_memory=True, persistent_workers=True)

val_data = Sen2_MTC(opt, mode='val')
val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.workers, drop_last=False, pin_memory=True, persistent_workers=True)

# assert 0 == 1
print('Load cloud_detection_model')
cloud_detection_model = FeatureExtractor()
cloud_detection_model.load_state_dict(torch.load(opt.cloud_model_path))
cloud_detection_model.eval()
set_requires_grad(cloud_detection_model, False)

print('Load ours model')
logger.info('Load ours model')
GEN = PMAA(32, 4)

def replace_batchnorm(model):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            child: torch.nn.BatchNorm2d = child
            setattr(model, name, torch.nn.InstanceNorm2d(child.num_features))
        else:
            replace_batchnorm(child)
replace_batchnorm(GEN)
DIS = Discriminator()

if opt.load_gen and opt.load_dis:
    print('loading pre-trained model')
    GEN.load_state_dict(torch.load(opt.load_gen))
    DIS.load_state_dict(torch.load(opt.load_dis))

if opt.optimizer == 'AdamW':
    optimizer_G = torch.optim.AdamW(
        GEN.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=5e-4)
    optimizer_D = torch.optim.AdamW(
        DIS.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=5e-4)
if opt.optimizer == 'SGD':
    optimizer_G = torch.optim.SGD(
        GEN.parameters(), lr=opt.lr, momentum=0.9, nesterov=True)
    optimizer_D = torch.optim.SGD(
        DIS.parameters(), lr=opt.lr, momentum=0.9, nesterov=True)

model_GEN = GEN
model_DIS = DIS

def preprocess_images(opt, batch, pmaa_bands=(3,2,1,7), mode="train"):
    if opt.dataset_name == "CTGAN_Sen2_MTC" or mode=="validation":
        real_A, real_B, _ = batch
        with torch.no_grad():
            M0, _, _ = cloud_detection_model(real_A[0].cuda())
            M1, _, _ = cloud_detection_model(real_A[1].cuda())
            M2, _, _ = cloud_detection_model(real_A[2].cuda())
        M = [M0, M1, M2]

        return real_A[0].cuda(), real_A[1].cuda(), real_A[2].cuda(), real_B.cuda(), M
    
    elif opt.dataset_name == "AllClear":        
        real_As = batch["input_images"] * 2 - 1
        real_Bs = batch["target"] * 2 - 1
        M = batch["input_cld_shdw"].max(dim=1).values
        M = [M[:,t:t+1].cuda() for t in range(3)]
        return real_As[:,pmaa_bands,0].cuda(), real_As[:,pmaa_bands,1].cuda(), real_As[:,pmaa_bands,2].cuda(), real_Bs[:,pmaa_bands,0].cuda(), M

def train(opt, model_GEN, model_DIS, cloud_detection_model, optimizer_G, optimizer_D, train_loader, val_loader):
    writer = SummaryWriter('runs29/%s' % opt.dataset_name)

    noise = opt.label_noise
    criterionGAN = GANLoss(opt.gan_mode)
    criterionL1 = torch.nn.L1Loss()
    criterionMSE = nn.MSELoss()

    if cuda:
        criterionGAN = criterionGAN.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        cloud_detection_model = cloud_detection_model.cuda()
        model_GEN = model_GEN.cuda()
        model_DIS = model_DIS.cuda()

    """lr_scheduler"""
    scheduler_G = CosineAnnealingLR(
        optimizer_G, T_max=opt.n_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(
        optimizer_D, T_max=opt.n_epochs, eta_min=1e-6)

    """training"""
    train_update = 0
    psnr_max = 0.
    ssim_max = 0.

    print('Start training!')
    for epoch in range(opt.n_epochs):
        model_GEN.train()
        model_DIS.train()

        pbar = tqdm.tqdm(total=len(train_loader), ncols=0,
                         desc="Train[%d/%d]" % (epoch, opt.n_epochs), unit=" step")

        lr = optimizer_G.param_groups[0]['lr']
        print('\nlearning rate = %.7f' % lr)

        L1_total = 0
        for batch_ids, batch in enumerate(train_loader): 
            
            if (batch_ids+1) >= 2380 // opt.batch_size: break
                
            real_A = [[], [], []]
            real_A[0], real_A[1], real_A[2], real_B, M = preprocess_images(opt, batch)

            real_A_combined = torch.cat(
                (real_A[0], real_A[1], real_A[2]), 1).cuda()
            real_A_input = torch.stack(
                (real_A[0], real_A[1], real_A[2]), 1).cuda()

            """forward generator"""
            fake_B, cloud_mask, aux_pred = model_GEN(real_A_input)

            """update Discriminator"""
            set_requires_grad(model_DIS, True)
            optimizer_D.zero_grad()

            fake_AB = torch.cat((real_A_combined, fake_B), 1)
            pred_fake = model_DIS(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False, noise)

            real_AB = torch.cat((real_A_combined, real_B), 1)
            pred_real = model_DIS(real_AB)
            loss_D_real = criterionGAN(pred_real, True, noise)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()

            """update generator"""
            optimizer_G.zero_grad()
            set_requires_grad(model_DIS, False)

            fake_AB = torch.cat((real_A_combined, fake_B), 1)
            pred_fake = model_DIS(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True, noise)

            if opt.dataset_name == "AllClear" and opt.masked_L1 == 1:
                target_mask = 1 - batch["target_cld_shdw"].max(dim=1).values.cuda()
                loss_G_L1 = criterionL1(fake_B * target_mask, real_B * target_mask) * opt.lambda_L1
            else:
                loss_G_L1 = criterionL1(fake_B, real_B) * opt.lambda_L1
            L1_total += loss_G_L1.item()

            loss_g_att = 0
            for i in range(len(cloud_mask)):
                
                if opt.dataset_name == "AllClear" and opt.pred_cloud == 1:
                    loss_g_att += criterionMSE(cloud_mask[i]
                                               [:, 0, :, :], 1-M[i][:, 0, :, :])                   
                else:
                    loss_g_att += criterionMSE(cloud_mask[i]
                                               [:, 0, :, :], M[i][:, 0, :, :])

            if opt.aux_loss:
                loss_G_aux = (criterionL1(aux_pred[0], real_B) + criterionL1(
                    aux_pred[1], real_B) + criterionL1(aux_pred[2], real_B)) * opt.lambda_aux
                loss_G = loss_G_GAN + loss_G_L1 + loss_g_att + loss_G_aux
            else:
                loss_G = loss_G_GAN + loss_G_L1 + loss_g_att
            loss_G.backward()
            optimizer_G.step()

            writer.add_scalar('training_G_GAN', loss_G_GAN, train_update)
            writer.add_scalar('training_G_L1', loss_G_L1, train_update)
            writer.add_scalar('training_D_real', loss_D_real, train_update)
            writer.add_scalar('training_D_fake', loss_D_fake, train_update)
            writer.add_scalar('training_D_fake', loss_g_att, train_update)
            
            
            wandb.log({
                'training_G_GAN': loss_G_GAN,
                'training_G_L1': loss_G_L1,
                'training_D_real': loss_D_real,
                'training_D_fake': loss_D_fake,
                'training_g_att': loss_g_att
            }, step=train_update)
            

            pbar.update()
            pbar.set_postfix(
                G_GAN=f"{loss_G_GAN:.4f}",
                G_L1=f"{loss_G_L1:.4f}",
                G_L1_total=f"{L1_total:.4f}",
                D_real=f"{loss_D_real:.4f}",
                D_fake=f"{loss_D_fake:.4f}"
            )
            train_update += 1
            
        pbar.close()
        """validation"""
        psnr, ssim = valid(opt, model_GEN, val_loader,
                           criterionL1, writer, epoch)

        wandb.log({"val_psnr": psnr}, step=train_update)
        wandb.log({"val_ssim": ssim}, step=train_update)
        # wandb.log({"val_psnr (ep)": psnr, "val_ssim (ep)": ssim}, step=epoch)

        if psnr_max < psnr:
            psnr_max = psnr
            torch.save(model_GEN.state_dict(), os.path.join(
                opt.save_model_path, run_name, f'EP{epoch}_G_best_PSNR_{psnr:.3f}_SSIM_{ssim:.3f}.pth'))

        if ssim_max < ssim:
            ssim_max = ssim
            torch.save(model_GEN.state_dict(), os.path.join(
                opt.save_model_path, run_name, f'EP{epoch}_G_best_SSIM_{ssim:.3f}_PNSR_{psnr:.3f}.pth'))

        scheduler_D.step()
        scheduler_G.step()

    print('Best PSNR: %.3f | Best SSIM: %.3f' % (psnr_max, ssim_max))

def valid(opt, model_GEN, val_loader, criterionL1, writer, epoch):
    model_GEN.eval()

    psnr_list = []
    ssim_list = []
    total_loss = 0

    pbar = tqdm.tqdm(total=len(val_loader), ncols=0,
                     desc="Valid[%d/%d]" % (epoch, opt.n_epochs), unit=" step")
    with torch.no_grad():
        for batch in val_loader:
            
            real_A = [[], [], []]
            real_A[0], real_A[1], real_A[2], real_B, _ = preprocess_images(opt, batch, mode="validation")

            real_A_input = torch.stack(
                (real_A[0], real_A[1], real_A[2]), 1).cuda()
            fake_B, _, _ = model_GEN(real_A_input)

            loss = criterionL1(fake_B, real_B)

            for batch_idx in range(len(real_B)):
                output, label = fake_B[batch_idx], real_B[batch_idx]
                output_rgb, label_rgb = get_rgb(output), get_rgb(label)

                psnr, ssim = psnr_ssim_cal(label_rgb, output_rgb)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

            total_loss += loss.item()
            pbar.update()
            pbar.set_postfix(
                loss_val=f"{total_loss:.4f}"
            )
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    psnr = np.mean(psnr_list)
    ssim = np.mean(ssim_list)

    writer.add_scalar('validation_PSNR', psnr, epoch)
    writer.add_scalar('validation_SSIM', ssim, epoch)
    pbar.set_postfix(loss_val=f"{total_loss:.4f}",
                     psnr=f"{psnr:.3f}", ssim=f"{ssim:.3f}")

    pbar.close()
    
    return psnr, ssim

train(opt, GEN, DIS, cloud_detection_model, optimizer_G,
      optimizer_D, train_loader, val_loader)