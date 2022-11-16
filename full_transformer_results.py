import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
from data_loader2 import get_loader
from build_vocab import Vocabulary
from torchvision import transforms
# from resnet_backbone.resnet101 import Encoder
from resnet_backbone.resnet2 import Encoder
# import time

# from Transformer.models import Transformer
# from Transformer.pos_hw_models import Transformer
# from Transformer.pos_models import Transformer
# from Transformer.pos_model_learn import Transformer
# from Transformer.pos_models_learnxy import Transformer
# from Transformer.pos_strength_com import Transformer
# from Transformer.pos_models_rel import Transformer
# from Transformer.pos_model_com_add import Transformer
# from Transformer.pos_model_learn_add import Transformer
# from Transformer.pos_1D_com import Transformer
# from Transformer.pos_strength import Transformer
# from Transformer.poscom_3D import Transformer
# from Transformer.PGEmod import Transformer
# from Transformer.complex_order import Transformer
# from Transformer.D2y_model import Transformer
# from Transformer.Dys_model import Transformer
from Transformer.Dy1_model import  Transformer
import json

from pytorch_ranger import Ranger
from nltk.translate.bleu_score import corpus_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose(
    [transforms.RandomCrop(args.crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    data_loader_eval = get_loader(args.image_dir_eval, args.caption_path_eval, vocab, transform, 1,
                                  shuffle=False,
                                  num_workers=args.num_workers)
    encoder=Encoder(encoded_image_size=14).to(device)

    decoder = Transformer(n_layers_dec=3, n_layers_enc=9, d_k=64, d_v=64, d_model=2048, d_ff=2048, n_heads=8,
                          max_seq_len=50, tgt_vocab_size=len(vocab),
                          dropout=0.1).to(device)
    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    prob_proj = nn.LogSoftmax(dim=-1)
    json_file = open(args.result_path, mode='w')
    save_json_content = []
    flag=[]
    total_step=len(data_loader_eval)
    for i, (images, captions, lengths,img_id) in enumerate(data_loader_eval):
        # begin=time.clock()
        print('Step [{}/{}]'.format(i,total_step))
        img_id=img_id.numpy()[0]
        if img_id not in flag:
            images = images.to(device)
            y = encoder(images)
            enc_outputs=y.to(device)
            enc_outputs, _ = decoder.encode(enc_outputs)
            beam_size=5
            k_prev_words = torch.LongTensor([[vocab.word2idx['<start>']]]* beam_size).to(device)
            top_k_scores = torch.zeros(beam_size, 1).to(device)
            complete_seqs = list()
            complete_seqs_scores = list()
            for step in range(args.max_decode_step):
                len_dec_seq = step+1
                dec_partial_inputs_len = torch.tensor([len_dec_seq]*beam_size).long().to(device)
                enc_output = enc_outputs.repeat(1, beam_size, 1).view(
                    beam_size, enc_outputs.size(1), enc_outputs.size(2))
                dec_out,_,_=decoder.decode(k_prev_words,dec_partial_inputs_len,enc_output)
                scores=decoder.tgt_proj(dec_out)
                scores=prob_proj(scores)

                for t in range(scores.size(0)):
                    scores[t,-1,:]+=top_k_scores[t,]
                if step==0:
                    top_k_scores, top_k_words = scores[0,-1,].topk(beam_size, 0, True, True)
                else:
                    scores=scores[:,-1,:]
                    top_k_scores, top_k_words = scores.reshape(-1).topk(beam_size, 0, True, True)
                prev_word_inds = top_k_words / len(vocab) # (s)
                next_word_inds = top_k_words % len(vocab)
                p = prev_word_inds.type(torch.LongTensor)
                prev_word_inds = p
                k_prev_words = torch.cat([k_prev_words[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != vocab.word2idx['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                if len(complete_inds) > 0:
                    complete_seqs.extend(k_prev_words[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                beam_size -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if beam_size == 0:
                    break
                k_prev_words = k_prev_words[incomplete_inds]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                # end=time.clock()
                # print(end-begin)
                # q=t
            sampled_re=[]
            m = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[m]
            for word_id in seq:
                if word_id not in {vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>'],vocab.word2idx['.']}:
                    word = vocab.idx2word[word_id]
                    sampled_re.append(word)
            sentence = ' '.join(sampled_re)
            # result.append({'img_id':img_id,'caption':sentence})
            result_json = {
                "image_id": int(img_id),
                "caption": sentence,
            }
            save_json_content.append(result_json)

            flag.append(img_id)
    json.dump(save_json_content, json_file)
    json_file.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##~~~initial

    parser.add_argument('--crop_size', type=int, default=384, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab2.pkl', help='path for vocabulary wrapper')


    parser.add_argument('--image_dir_eval', type=str, default='/media/huashuo/mydisk/cocodataset/coco2014/384_resized/val2014',   ##images_resized
                        help='directory for resized images')  ##new

    parser.add_argument('--caption_path_eval', type=str, default='/media/huashuo/mydisk/cocodataset/coco2014/annotations_trainval2014/annotations/captions_val2014.json',
                        help='path for train annotation json file')
    # parser.add_argument('--image_dir_eval', type=str,
    #                     default='data/image_datang_resized/val2017',
    #                     help='directory for resized images')  ##new
    #
    # parser.add_argument('--caption_path_eval', type=str,
    #                     default='/media/huashuo/mydisk/cocodataset/coco2017/annotations/captions_val2017.json',
    #                     help='path for train annotation json file')
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--encoder_path', type=str, default='/media/huashuo/mydisk/hssavemodel/dy3/encoder-4.ckpt',  # default .pkl  3-3
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/media/huashuo/mydisk/hssavemodel/dy3/decoder-4.ckpt',  # default.pkl
                        help='path for trained decoder')

    # parser.add_argument('--encoder_path', type=str, default='save_model_pos/best/encoder3-3.ckpt',  # default .pkl
    #                     help='path for trained encoder')
    # parser.add_argument('--decoder_path', type=str, default='save_model_pos/best/decoder3-3.ckpt',  # default.pkl
    #                     help='path for trained decoder')

    parser.add_argument('--result_path', type=str, default='result/d1y4.json',       ###pos_3D_ver11_3 best\dy9\dy17\dy18   d2y1
                        help='path for trained decoder')
    parser.add_argument('-max_decode_step', type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args)