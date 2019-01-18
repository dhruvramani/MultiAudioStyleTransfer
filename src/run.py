import os
import gc
import torch
import argparse
import librosa
import matplotlib
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import *
from feature import *
from dataset import *
#from vctk import VCTK
from utils import progress_bar

# NOTE : Change the code to run on background_ds
# TODO : Combine audio after style transfer
# TODO : Implement test
parser = argparse.ArgumentParser(description='PyTorch Audio Style Transfer')
parser.add_argument('--epoch', '-e', type=int, default=4, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--resume1', '-r1', type=int, default=0, help='resume transform1 from checkpoint')
parser.add_argument('--lr1', default=0.001, type=float, help='learning rate for transform1') 
parser.add_argument('--resume2', '-r2', type=int, default=0, help='resume transform2 from checkpoint')
parser.add_argument('--lr2', default=0.001, type=float, help='learning rate for transform2') 
# Loss network trainer
parser.add_argument('--lresume1', type=int, default=0, help='Resume network from checkpoint for loss network 1')
parser.add_argument('--loss_lr1', type=float, default=1e-4, help='The rearning rate for loss network 1')
parser.add_argument('--lresume2', type=int, default=0, help='resume network from checkpoint for loss network 2')
parser.add_argument('--loss_lr2', type=float, default=1e-4, help='The learning rate for loss network 2')

parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=1e-5, help='Weight decay (L2 penalty).')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = torch.nn.MSELoss() # MaskedMSE()

def train_lossn(network_params):
    ida, epoch, lstep, lr, dataset = network_params["id"], network_params["epoch"], network_params["step"], network_params["lr"], network_params["dataset"]
    encoder, decoder = network_params["network"]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)

    print('\n=> Loss {} Epoch: {}'.format(ida, epoch))
    train_loss, total = 0, 0
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=args.decay)
    
    for i in range(lstep, len(dataloader)):
        (audios, captions) = next(dataloader)
        if(type(audios) == int):
            print("=> Loss {} Network : Chucked Sample".format(ida))
            continue
        
        del captions
        audios = (audios[:, :, :, 0:500].to(device), audios[:, :, :, 500:1000].to(device))
        # Might have to remove the loop,, memory
        for audio in audios:
            latent_space = encoder(audio)
            output = decoder(latent_space)
            optimizer.zero_grad()
            loss = loss_fn(output, audio[:, :, :, :-3])
            loss.backward()
            optimizer.step()

        del audios
        train_loss += loss.item()
        print(train_loss)
        with open("../save/loss{}/logs/lossn_train_loss.log".format(ida), "a+") as lfile:
            lfile.write("{}\n".format(train_loss / (i - lstep +1)))

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(encoder.state_dict(), '../save/loss{}/loss_encoder.ckpt'.format(ida))
        torch.save(decoder.state_dict(), '../save/loss{}/loss_decoder.ckpt'.format(ida))

        with open("../save/loss{}/info.txt".format(ida), "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: %.3f' % (train_loss / (i - lstep + 1)))

    lstep = 0
    #del dataloader
    #del vdataset
    print('=> Loss {} Network : Epoch [{}/{}], Loss:{:.4f}'.format(ida, epoch + 1, 5, train_loss / len(dataloader)))
    network_params["epoch"], network_params["step"] = epoch, lstep 
    network_params["network"] = [encoder, decoder]
    del dataloader
    return network_params

def train_transformation(network_params):
    ida, epoch, tstep, lr, dataset = network_params["id"], network_params["epoch"], network_params["step"], network_params["lr"], network_params["dataset"]
    t_net, encoder = network_params["network"]
    print('\n=> Transformation {} Epoch: {}'.format(ida, epoch))
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)

    train_loss, tr_con, tr_sty = 0, 0, 0

    params = t_net.parameters()     
    optimizer = torch.optim.Adam(params, lr=lr) 

    l_list = list(encoder.children())
    l_list = list(l_list[0].children())
    conten_activ = torch.nn.Sequential(*l_list[:-1]) # Not having batchnorm

    for param in conten_activ.parameters():
        param.requires_grad = False

    # TODO : might have diff hyperparams for diff transformation
    alpha, beta = 200, 100000 # TODO : CHANGEd from 7.5, 100 
    gram = GramMatrix()

    style_audio = get_style(network_params["style"])

    for i in range(tstep, len(dataloader)):
        try :
            (audios, captions) = next(dataloader)
        except ValueError:
            break
        if(type(audios) == int):
            print("=> Transformation {} Network : Chucked Sample".format(ida))
            continue

        audios = (audios[:, :, :, 0:300].to(device), audios[:, :, :, 300:600].to(device), audios[:, :, :, 600:900].to(device))
        for audio in audios : # LOL - splitting coz GPU
            optimizer.zero_grad()
            y_t = t_net(audio)

            content = conten_activ(audio)
            y_c = conten_activ(y_t)

            c_loss = loss_fn(y_c, content)

            s_loss = 0
            sty_aud = []
            for k in range(audio.size()[0]): # No. of style audio == batch_size
                sty_aud.append(style_audio)
            sty_aud = torch.stack(sty_aud).to(device)

            for st_i in range(2, len(l_list)-4, 3): # NOTE : gets relu of 1, 2, 3
                st_activ = torch.nn.Sequential(*l_list[:st_i])
                for param in st_activ.parameters():
                    param.requires_grad = False

                y_s = gram(st_activ(y_t))
                style = gram(st_activ(sty_aud))
		
                s_loss += loss_fn(y_s, style)
            
            del sty_aud	
            loss = alpha * c_loss + beta * s_loss 

            train_loss = loss.item()
            tr_con = c_loss.item()
            tr_sty = s_loss.item()
            
            for param in encoder.parameters():
                param.requires_grad = False
        
            loss.backward()
            optimizer.step()

        del audios

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(t_net.state_dict(), '../save/transform{}/trans_model.ckpt'.format(ida))
        with open("../save/transform{}/info.txt".format(ida), "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/transform{}/logs/transform_train_loss.log".format(ida), "a+") as lfile:
            lfile.write("{}\n".format(train_loss))

        progress_bar(i, len(dataloader), 'Loss: {}, Con Loss: {}, Sty Loss: {} '.format(train_loss, tr_con, tr_sty))

    tstep = 0
    del dataloader
    print('=> Transformation {} Network, {} : Epoch [{}/{}], Loss:{:.4f}'.format(ida ,epoch + 1, args.epochs, train_loss))

    network_params["epoch"], network_params["step"] = epoch, lstep 
    network_params["network"] = [t_net, encoder]
    return network_params

def train_multiast():
    # TODO : Change plis
    print('==> Preparing data..')
    combined_ds = CombinedDataset()
    foreground_ds, background_ds = VocalDataset(), BackgroundDataset(n_splits=1)
    del combined_ds
    
    print('==> Creating networks..')
    best_acc, tsepoch1, tstep1, lsepoch1, lstep1, tsepoch2, tstep2, lsepoch2, lstep2 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    t_net1, t_net2 = Transformation(netno=1).to(device), Transformation(netno=2).to(device)
    enc1, enc2 = Encoder().to(device), Encoder().to(device)
    dec1, dec2 = Decoder().to(device), Decoder().to(device)

    if(args.lresume1):
        if(os.path.isfile('../save/loss1/loss_encoder.ckpt')):
            enc1.load_state_dict(torch.load('../save/loss1/loss_encoder.ckpt'))
            dec1.load_state_dict(torch.load('../save/loss1/loss_decoder.ckpt'))
            print("=> Loss Network 1 : loaded")

        if(os.path.isfile("../save/loss1/info.txt")):
            with open("../save/loss1/info.txt", "r") as f:
                lsepoch1, lstep1 = (int(i) for i in str(f.read()).split(" "))
                print("=> Loss Network 1 : prev epoch found")

        with open("../save/loss1/logs/transform_train_loss.log", "w+") as f:
            pass 

    if(args.lresume2):
        if(os.path.isfile('../save/loss2/loss_encoder.ckpt')):
            enc2.load_state_dict(torch.load('../save/loss2/loss_encoder.ckpt'))
            dec2.load_state_dict(torch.load('../save/loss2/loss_decoder.ckpt'))
            print("=> Loss Network 2 : loaded")

        if(os.path.isfile("../save/loss2/info.txt")):
            with open("../save/loss2/info.txt", "r") as f:
                lsepoch2, lstep2 = (int(i) for i in str(f.read()).split(" "))
                print("=> Loss Network 2 : prev epoch found")

        with open("../save/loss2/logs/transform_train_loss.log", "w+") as f:
            pass 

    if(args.resume1):
        if(os.path.isfile('../save/transform1/trans_model.ckpt')):
            t_net1.load_state_dict(torch.load('../save/transform1/trans_model.ckpt'))
            print('==> Transformation network 1 : loaded')

        if(os.path.isfile("../save/transform1/info.txt")):
            with open("../save/transform1/info.txt", "r") as f:
                tsepoch1, tstep1 = (int(i) for i in str(f.read()).split(" "))
            print("=> Transformation network 1 : prev epoch found")

            # To get logs of current run only
        with open("../save/transform1/logs/transform_train_loss.log", "w+") as f:
            pass 

    if(args.resume2):
        if(os.path.isfile('../save/transform2/trans_model.ckpt')):
            t_net2.load_state_dict(torch.load('../save/transform2/trans_model.ckpt'))
            print('==> Transformation network 2 : loaded')

        if(os.path.isfile("../save/transform2/info.txt")):
            with open("../save/transform2/info.txt", "r") as f:
                tsepoch2, tstep2 = (int(i) for i in str(f.read()).split(" "))
            print("=> Transformation network 2 : prev epoch found")

            # To get logs of current run only
        with open("../save/transform2/logs/transform_train_loss.log", "w+") as f:
            pass 

    # TODO : Change the params
    network_dict = {"t1" : {"id" : 1, "network" : [t_net1, enc1], "lr" : args.lr1, "epoch" : tsepoch1, "step" : tstep1, "dataset" : foreground_ds, "style" : "lady.mp3"}, 
                    "t2" : {"id" : 2, "network" : [t_net2, enc2], "lr" : args.lr2, "epoch" : tsepoch2, "step" : tstep2, "dataset" : background_ds, "style" : "guy.mp3"},
                    "l1" : {"id" : 1, "network" : [enc1, dec1], "lr" : args.loss_lr1, "epoch" : lsepoch1, "step" : lstep1, "dataset" : foreground_ds},
                    "l2" : {"id" : 2, "network" : [enc2, dec2], "lr" : args.loss_lr2, "epoch" : lsepoch2, "step" : lstep2, "dataset" : background_ds}}
    for epoch in range(lsepoch1, lsepoch1 + args.epoch):
        network_dict["l1"] = train_lossn(network_dict["l1"])

    # NOTE : Uncomment    
    #for epoch in range(lsepoch2, lsepoch1 + args.epoch):
    #    network_dict["l2"] = train_lossn(network_dict["l2"])

    for epoch in range(tsepoch1, tsepoch1 + args.epochs):
        network_dict["t1"] = train_transformation(network_dict["t1"])

   # NOTE : Uncomment  
   # for epoch in range(tsepoch2, tsepoch2 + args.epochs):
   #     network_dict["t2"] = train_transformation(network_dict["t2"])

def test():
    # TODO : Change completely
    global t_net
    t_net.load_state_dict(torch.load('../save/transform/trans_model.ckpt'))
    #vdataset = VCTK('/home/nevronas/dataset/', download=False)
    #dataloader = DataLoader(vdataset, batch_size=1)
    #audio, _ = next(iter(dataloader))
    audio, fs = load_audio('/home/nevronas/dataset/vctk/raw/p225_308.wav')
    audio = torch.Tensor(audio)
    audio, phase = test_transform(audio)
    audio = audio.to(device)
    out = t_net(audio)
    out = out[0].detach().cpu().numpy()
    audio = audio[0].cpu().numpy()
    matplotlib.image.imsave('../save/plots/input/audio.png', audio[0])
    matplotlib.image.imsave('../save/plots/output/stylized_audio.png', out[0])
    aud_res = reconstruction(audio[0], phase)
    out_res = reconstruction(out[0], phase[:, :-3])
    librosa.output.write_wav("../save/plots/input/raw_audio.wav", aud_res, fs)
    librosa.output.write_wav("../save/plots/output/raw_output.wav", out_res, fs)
    print("Testing Finished")

if __name__ == '__main__':
    train_multiast()
