import torch
import torch.nn as nn
import math
class Poslearned(nn.Module):
    def __init__(self, w,h,features):
        super(Poslearned, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, w*h, features))
        self.h=h
        self.w=w
    def forward(self,batch):
        out=self.pos_embedding.repeat(batch,1,1)
        return out



# p=PosColumn(3,3,1)
# q=p(2)
# print(q.shape)


# class PositionEmbeddingSineHW(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on images.
#     """
#     def __init__(self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperatureH = temperatureH
#         self.temperatureW = temperatureW
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale
#
#     def forward(self, batch_size,picture_size):
#         # x = tensor_list.tensors
#         # mask = tensor_list.mask
#         # assert mask is not None
#         # not_mask = ~mask
#         pos_emb=torch.ones(batch_size,picture_size,picture_size)
#         y_embed = pos_emb.cumsum(1, dtype=torch.float32)
#         x_embed = pos_emb.cumsum(2, dtype=torch.float32)
#
#         # import ipdb; ipdb.set_trace()
#
#         if self.normalize:
#             eps = 1e-6
#             y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
#             x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
#
#         dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32)
#         dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
#         pos_x = x_embed[:, :, :, None] / dim_tx
#
#         dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32)
#         dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
#         pos_y = y_embed[:, :, :, None] / dim_ty
#
#         pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         # pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#         pos = torch.cat((pos_y, pos_x), dim=3)
#         pos=pos.reshape((pos.shape[0],-1,pos.shape[3]))
#         # import ipdb; ipdb.set_trace()
#         return pos


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, batch_size,picture_size):
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask
        pos_emb = torch.ones(batch_size, picture_size, picture_size)
        y_embed = pos_emb.cumsum(1, dtype=torch.float32)
        x_embed = pos_emb.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)   #device=x.device
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = pos_y+pos_x
        pos = pos.reshape((pos.shape[0], -1, pos.shape[3]))
        return pos
class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        nn.init.xavier_uniform_(self.row_embed.weight)   #
        nn.init.xavier_uniform_(self.col_embed.weight)   #
    def forward(self,batch, w,h):
        i = torch.arange(w).cuda()
        j = torch.arange(h).cuda()
        # i = torch.arange(w)
        # j = torch.arange(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos=(x_emb.unsqueeze(0).repeat(h, 1, 1)+y_emb.unsqueeze(1).repeat(1, w, 1)).unsqueeze(0).repeat(batch, 1, 1, 1)

        pos = pos.reshape((pos.shape[0], -1, pos.shape[3]))
        return pos
# test》LearnedPositionalEncoding
# pos=LearnedPositionalEncoding(num_feats=1024,row_num_embed=50,
#                  col_num_embed=50)
# batch=4
# picture=14
# pos_out=pos(batch,picture,picture)
# print(pos_out.shape)
# class LearnedPositionalxy(nn.Module):
#     """Position embedding with learnable embedding weights.
#     Args:
#         num_feats (int): The feature dimension for each position
#             along x-axis or y-axis. The final returned dimension for
#             each position is 2 times of this value.
#         row_num_embed (int, optional): The dictionary size of row embeddings.
#             Default 50.
#         col_num_embed (int, optional): The dictionary size of col embeddings.
#             Default 50.
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#     """
#
#     def __init__(self,
#                  num_feats,
#                  embed=50,):
#         super(LearnedPositionalxy, self).__init__()
#         self.embed = nn.Embedding(embed, num_feats)
#
#         self.num_feats = num_feats
#
#
#     def forward(self,batch, w,h):
#         i = torch.arange(w).cuda()
#         j = torch.arange(h).cuda()
#         x_emb = self.embed(i)
#         y_emb = self.embed(j)
#         pos = torch.cat([
#             x_emb.unsqueeze(0).repeat(h, 1, 1),
#             y_emb.unsqueeze(1).repeat(1, w, 1),
#         ], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
#         pos = pos.reshape((pos.shape[0], -1, pos.shape[3]))
#         return pos