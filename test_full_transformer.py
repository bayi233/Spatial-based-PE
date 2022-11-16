import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
# from resnet_backbone.resnet101 import Encoder
from resnet_backbone.resnet2 import Encoder
# from Transformer.models import Transformer
# from Transformer.pos_models import Transformer
# from Transformer.pos_1D_com import Transformer
# from Transformer.pos_strength import Transformer
from Transformer.poscom_3D import Transformer
from PIL import Image
import json
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Image preprocessing
    transform = transforms.Compose(
        [transforms.Resize(args.crop_size), transforms.CenterCrop(args.crop_size), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transform.Resize insteads of transforms.Scale()
    ##centecrop裁图片
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    args.src_vocab_size = len(vocab)
    args.tgt_vocab_size = len(vocab)
    # print(len(vocab))
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    encoder=Encoder(encoded_image_size=14).to(device)

    decoder = Transformer(n_layers_dec=3, n_layers_enc=9, d_k=64, d_v=64, d_model=2048, d_ff=2048, n_heads=8,
                          max_seq_len=50, tgt_vocab_size=len(vocab),
                          dropout=0.1).to(device)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    encoder.eval()
    decoder.eval()
    image_dir = args.image_dir
    images = os.listdir(image_dir)
    img_s = {}
    img_dict = {}
    prob_proj = nn.LogSoftmax(dim=-1)
    for i, image in enumerate(images):
        img_path = os.path.join(image_dir, image)
        # img_dict[image] = str(img_path)
        with open(img_path, 'r+b') as f:
            with Image.open(f) as img:
                # resize image
                img = transform(img).unsqueeze(0)
                img_dict[image] = img
    for img_key, img in img_dict.items():
        img_var = Variable(img)
        img_var = img_var.to(device)
        y = encoder(img_var)
        enc_outputs = y.to(device)
        enc_outputs,_=decoder.encode(enc_outputs)

        beam_size = 3
        k_prev_words = torch.LongTensor([[vocab.word2idx['<start>']]] * beam_size).to(device)
        top_k_scores = torch.zeros(beam_size, 1).to(device)
        complete_seqs = list()
        complete_seqs_scores = list()
        for step in range(args.max_decode_step):
            len_dec_seq = step + 1
            dec_partial_inputs_len = torch.tensor([len_dec_seq] * beam_size).long().to(device)
            enc_output = enc_outputs.repeat(1, beam_size, 1).view(
                beam_size, enc_outputs.size(1), enc_outputs.size(2))
            # dec_partial_inputs_len = torch.LongTensor(n_remaining_sents, ).fill_(len_dec_seq)
            # dec_partial_inputs_len = dec_partial_inputs_len.repeat(beam_size)
            dec_out,_,_= decoder.decode(k_prev_words, dec_partial_inputs_len, enc_output)
            scores=decoder.tgt_proj(dec_out)
            scores = prob_proj(scores)

            for t in range(scores.size(0)):
                scores[t, -1, :] += top_k_scores[t,]
            if step == 0:
                top_k_scores, top_k_words = scores[0, -1,].topk(beam_size, 0, True, True)
            else:
                scores = scores[:, -1, :]
                top_k_scores, top_k_words = scores.reshape(-1).topk(beam_size, 0, True, True)
            prev_word_inds = top_k_words / len(vocab)  # (s)
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
        m = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[m]
        sampled_caption = []
        for word_id in seq:
            word = vocab.idx2word[int(word_id)]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)  ##拼接
        img_s[img_key] = sentence
    json_dic = json.dumps(img_s, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=True)
    print(json_dic)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##~~~initial

    parser.add_argument('--image_dir', type=str, default='data/test',
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='hssavemodel/pool/beste.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='hssavemodel/pool/bestd.ckpt',      ##hssavemodel/strength/decoder2-1   #save_model_pos/best/   ##hssavemodel/1DCOM/
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab2.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=384, help='size for center cropping images')
    parser.add_argument('--max_decode_step', type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args)