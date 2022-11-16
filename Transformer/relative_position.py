import torch
import torch.nn as nn
class RelativePosition(nn.Module):

    def __init__(self, feature, max_relative_position):
        super().__init__()
        # self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_tablex = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, feature))
        self.embeddings_tabley=  nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, feature))
        nn.init.xavier_uniform_(self.embeddings_tablex)
        nn.init.xavier_uniform_(self.embeddings_tabley)

    def forward(self,batch, length_h, length_w):
        range_vec_q = torch.arange(length_h)
        range_vec_k = torch.arange(length_w)
        distance_matx = range_vec_k[None, :] - range_vec_q[:, None]
        distance_maty = range_vec_q[:, None] - range_vec_k[None,:]
        # distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_matx = distance_matx + self.max_relative_position
        final_maty= distance_maty + self.max_relative_position
        final_matx = torch.LongTensor(final_matx).cuda()
        final_maty= torch.LongTensor(final_maty).cuda()
        pos_x = self.embeddings_tablex[final_matx].cuda()
        pos_y = self.embeddings_tablex[final_maty].cuda()
        pos=torch.cat((pos_x,pos_y),dim=2).repeat(batch,1,1,1)
        pos=pos.reshape((pos.shape[0],-1,pos.shape[3]))
        return pos


# pos=RelativePosition(feature=1,max_relative_position=3)
# x=pos(2,2,2)
# print(x.shape)
# self.relative_position_k = RelativePosition(i, self.d_k, max_relative_position)
# self.relative_position_v = RelativePosition(i, self.d_v, max_relative_position)
#
# r_q = q.permute(2, 0, 1, 3).contiguous().view(len_q, sz_b*n_head, d_k)
# r_k = self.relative_position_k(len_q, len_k)
# attn_2 = torch.matmul(r_q, r_k.transpose(1, 2)).transpose(0, 1)
# attn_2 = attn_2.contiguous().view(sz_b, self.n_head, len_k, len_k)
#
# r_v = self.relative_position_v(len_q, len_v)
# weight = attn.permute(2, 0, 1, 3).contiguous().view(len_q, sz_b*n_head, len_k)
# weight = torch.matmul(weight, r_v)
# weight = weight.transpose(0, 1).contiguous().view(sz_b, self.n_head, len_q, d_v)
