import torch
import torch.nn as nn
import math
#DSPE

class StrengthenSpatialPosEncoding(nn.Module):
    "ATTENTION:The num_feats should be defined as the double of the min(w,h) minus one"
    def __init__(self,
                 num_feats,
                 embed_dim):
        super(StrengthenSpatialPosEncoding, self).__init__()
        self.embed = nn.Embedding(embed_dim, num_feats)
        self.num_feats = num_feats
        self.embed_dim = embed_dim
        nn.init.uniform_(self.embed.weight)

    def forward(self,batch, w,h):
        emb=[]
        for i in range(0, h):
            for p in range(i, i + w):
                emb.append(p)
        emb=torch.tensor(emb).cuda()
        pos=self.embed(emb)
        pos = pos.repeat(batch,1,1)
        return pos
class StrengthenSpatialPosEncodingcom(nn.Module):
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
        pos_emb = torch.ones(picture_size*2-1)
        embed = pos_emb.cumsum(0, dtype=torch.float32)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32) #device=x.device
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos = embed[:, None] / dim_t
        pos = torch.stack((pos[:, 0::2].sin(), pos[:, 1::2].cos()), dim=2).flatten(1)
        pos_fin=pos[0,:].unsqueeze(0)
        for i in range(0, picture_size):
            for p in range(i, i + picture_size):
                if not (i==0 and p==0):
                    pos_mid=pos[p,:].unsqueeze(0)
                    pos_fin=torch.cat((pos_fin, pos_mid), dim=0)
        pos = pos_fin.repeat(batch_size,1,1)
        return pos
##test =>  StrengthenSpatialPosEncoding
# pos=StrengthenSpatialPosEncoding(2048,23)
# batch=6
# w=12
# h=12
# pos_emb=pos(batch,w,h)
# print(pos_emb.shape)
