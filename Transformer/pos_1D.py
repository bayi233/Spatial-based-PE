import torch
import torch.nn as nn
import math
class PositionEmbeddingSine1D(nn.Module):
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

    def forward(self, batch_size,line_size):
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask
        pos_emb = torch.ones(batch_size, line_size)
        y_embed = pos_emb.cumsum(1, dtype=torch.float32)
        # x_embed = pos_emb.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)   #device=x.device
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :,  None] / dim_t
        # pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :,  0::2].sin(), pos_y[:, :,  1::2].cos()), dim=3).flatten(2)
        # pos = torch.cat((pos_y, pos_x), dim=3)
        # pos = pos.reshape((pos.shape[0], -1, pos.shape[3]))
        return pos_y
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
                 ):   #col_num_embed=50,
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        # self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        # self.col_num_embed = col_num_embed
        nn.init.xavier_uniform_(self.row_embed.weight)   #
        # nn.init.xavier_uniform_(self.col_embed.weight)   #
    def forward(self,batch, line):
        # i = torch.arange(line)
        j = torch.arange(line).cuda()
        # x_emb = self.col_embed(i)

        y_emb = self.row_embed(j)
        pos = y_emb.repeat(batch, 1, 1)
        # pos = torch.cat([
        #     x_emb.unsqueeze(0).repeat(h, 1, 1),
        #     y_emb.unsqueeze(1).repeat(1, w, 1),
        # ], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
        # pos = pos.reshape((pos.shape[0], -1, pos.shape[3]))
        return pos


# test》PositionEmbeddingSine
# pos=PositionEmbeddingSine(num_pos_feats=2, temperature=10000, normalize=False, scale=None)
# batch=1
# picture=8
# pos_out=pos(batch,picture)
# print(pos_out)
# test》LearnedPositionalEncoding
# pos=LearnedPositionalEncoding(num_feats=1024, row_num_embed=196)
# batch=4
# picture=196
# pos_out=pos(batch,picture)
# print(pos_out.shape)