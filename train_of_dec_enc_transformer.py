import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from torch import optim

from data_loader import get_loader
from build_vocab import Vocabulary
from torchvision import transforms
# from resnet_backbone.resnet101 import Encoder
from resnet_backbone.resnet2 import Encoder
# from Transformer.models import Transformer
# from Transformer.pos_models import Transformer
# from Transformer.pos_model_learn import Transformer
# from Transformer.pos_models_learnxy import Transformer
# from Transformer.pos_models_rel import Transformer
# from Transformer.pos_model_learn_add import Transformer
# from Transformer.pos_model_learn_add import Transformer
# from Transformer.pos_1D_com import Transformer
# from Transformer.pos_strength import Transformer
# from Transformer. D2y_model import Transformer
# from Transformer.Dys_model import Transformer
from Transformer.Dy1_model import Transformer

# from Transformer.PGEmod import Transformer
# from Transformer.PGEmod import Transformer
# from Transformer.poscom_3D import Transformer
# # from torchstat import stat
# from thop import profile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cuda:1')
# class FNet(nn.Module):
#     def __init__(self,tgt_vocab_size):
#         super(FNet,self).__init__()
#
#         self.encoder=Encoder(encoded_image_size=14)
#         self.decoder=Transformer(n_layers_dec=4, n_layers_enc=12, d_k=64, d_v=64, d_model=2048, d_ff=2048, n_heads=8,max_seq_len=50, tgt_vocab_size=tgt_vocab_size,
#                  dropout=0.1 )
#         self.encoder.requires_grad_(False)
#     def forward(self,image,cap_len,captions):
#         features = self.encoder(image)
#         output, _ =self.decoder(features, captions, cap_len)
#         return output

def main(args):
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose(
        [transforms.RandomCrop(args.crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_eval=vocab
    data_loader_train = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size, shuffle=True,
                             num_workers=args.num_workers)
    data_loader_eval = get_loader(args.image_dir_eval, args.caption_path_eval, vocab_eval, transform, args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)

    # tgt_vocab_size=len(vocab)
    encoder=Encoder(encoded_image_size=14).to(device)

    decoder = Transformer(n_layers_dec=3, n_layers_enc=9, d_k=64, d_v=64, d_model=2048, d_ff=2048, n_heads=8,max_seq_len=50, tgt_vocab_size=len(vocab),
                 dropout=0.1 ).to(device)

    # input=torch.randn(20,144,2048)
    # flops,params=profile(decoder,(input))
    criterion = nn.CrossEntropyLoss(ignore_index=0)                     ##dy best= 0.00003 + 0.000001  + 0.0000005
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0000075)  #1~7 lr=0.00003/  8,9:add w_d=0.5 / 10,11 lr=0.0000075 ,12,13 w_d=0.5  ##weight_decay=0.00000125
    # decoder_optimizer = optim.AdamW(decoder.parameters(),lr=0.001,weight_decay=0.001)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=decoder_optimizer, last_epoch=-1, step_size=5, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(decoder_optimizer,T_0=4,eta_min=0.000001)
    total_step_train = len(data_loader_train)
    total_step_eval=len(data_loader_eval)
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))
    encoder.eval()
    train_loss=[]
    eval_loss=[]
    for epoch in range(args.num_epochs):  ##时间步
        avg_loss=train(encoder,decoder,data_loader_train,criterion,decoder_optimizer,epoch,total_step_train)
        train_loss.append(avg_loss)

        avg_loss_eval=eval(encoder, decoder, data_loader_eval, criterion, epoch, total_step_eval)
        eval_loss.append(avg_loss_eval)



        torch.save(decoder.state_dict(),
                   os.path.join(args.model_path, 'decoder2-{}.ckpt'.format(epoch + 1)))
        torch.save(encoder.state_dict(),
                   os.path.join(args.model_path, 'encoder2-{}.ckpt'.format(epoch + 1)))
    # torch.save(decoder.state_dict(),os.path.join(args.model_path, 'full_d_resnet4.ckpt'))
    # torch.save(encoder.state_dict(),os.path.join(args.model_path, 'full_e_resnet4.ckpt'))
    plt.title("train_avg_loss/epoch", fontsize=20)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.plot(range(1,args.num_epochs+1), train_loss, label="train_avg_loss", color='red')
    plt.legend(loc='best')
    picture_loss = plt.gcf()
    picture_loss.savefig(r'loss_picture/pos22e2.png')
    plt.show()
    plt.title("eval_avg_loss/epoch", fontsize=20)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.plot(range(1, args.num_epochs + 1), eval_loss, label="train_avg_loss", color='red')
    # plt.plot(range(81, 81+len(eval_loss)), eval_loss, label="eval_avg_loss", color='blue')
    plt.legend(loc='best')
    picture_loss = plt.gcf()
    picture_loss.savefig(r'loss_picture/pos22d2.png')
    plt.show()
def eval(encoder,decoder,data_loader,criterion,epoch,total_step):
    total_loss=0.0
    decoder.eval()
    encoder.eval()
    print("Now eval")
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            y= encoder(images)
            features = y.to(device)
            dec_lengths = torch.tensor(lengths).to(device)
            dec_lengths = (dec_lengths - 1).to(device)
            dec_inputs = (captions[:, :-1]).to(device)  ##
            output,_= decoder(features,dec_inputs,dec_lengths)
            dec_targets = (captions[:, 1:]).to(device)
            loss = criterion(output, dec_targets.contiguous().view(-1))
            total_loss+=loss.item()
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}],LOSS:{:.4f}'.format(epoch + 1, args.num_epochs,i,total_step,loss))
    avg_loss = total_loss / total_step
    return round(avg_loss, 4)


def train(encoder,decoder,data_loader,criterion,decoder_optimizer,epoch,total_step):
    encoder.eval()
    decoder.train()
    total_loss=0.0
    print("Now train")
    # total_step=1250
    for i, (images, captions, lengths) in enumerate(data_loader):
        images = images.to(device)
        dec_lengths = torch.tensor(lengths).to(device)
        dec_lengths = (dec_lengths - 1).to(device)
        dec_inputs = (captions[:, :-1]).to(device)  ##
        y = encoder(images)
        features = y.to(device)
        output,_= decoder(features,dec_inputs,dec_lengths)
        dec_targets = (captions[:, 1:]).to(device)
        loss = criterion(output, dec_targets.contiguous().view(-1))
        decoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()
        total_loss+=loss.item()
        if i % args.log_step == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch + 1, args.num_epochs,
                                                                                        i,
                                                                                        total_step, loss.item(),
                                                                                        np.exp(loss.item())))

    avg_loss=total_loss/total_step

    return round(avg_loss,4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ##~~~initial
    parser.add_argument('--model_path', type=str, default='/media/huashuo/mydisk/hssavemodel/dy3', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=384, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab2.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/media/huashuo/mydisk/cocodataset/coco2014/384_resized/train2014',    ##384_resized/train2014
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='/media/huashuo/mydisk/cocodataset/coco2014/annotations_trainval2014/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')

    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--caption_path_eval', type=str, default='/media/huashuo/mydisk/cocodataset/coco2014/annotations_trainval2014/annotations/captions_val2014.json',
                        help='path for train annotation json file')  ###new
    parser.add_argument('--image_dir_eval', type=str, default='/media/huashuo/mydisk/cocodataset/coco2014/384_resized/val2014',    ##images_resized
                        help='directory for resized images')  ##new
    parser.add_argument('--encoder', type=str, default='/media/huashuo/mydisk/hssavemodel/dy3/encoder-3.ckpt',  # default .pkl  ##poswh /hs_save_model_poshw/encoder-4.ckpt
                        help='path for trained encoder')
    parser.add_argument('--decoder', type=str, default='/media/huashuo/mydisk/hssavemodel/dy3/decoder-3.ckpt',  # default.pkl  4-4
                        help='path for trained decoder')
    args = parser.parse_args()
    print(args)
    main(args)
