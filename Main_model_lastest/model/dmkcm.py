from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer, MultiheadAttention
import math
import torch.nn.functional as F
from torch_scatter import scatter_add


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class DMKCM(nn.Module):
    def __init__(self,
                 config,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(DMKCM, self).__init__()
        self.config = config
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        # 1hop
        if config.only_hop1 or config.memory:
            self.multihead_attn = MultiheadAttention(emb_size, nhead, dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.alpha_linear = nn.Linear(dim_feedforward, 1)

        if config.memory:
            self.attn = nn.Linear(emb_size * 2, emb_size)
            self.v = nn.Parameter(torch.FloatTensor(emb_size))
            self.multihead_attn_m = MultiheadAttention(emb_size, nhead, dropout)
            self.dropout_m = nn.Dropout(dropout)

        # 2hop
        if config.hop2:
            self.relation_embd = nn.Embedding(40, dim_feedforward)
            self.hop_number = 1
            self.W_s = nn.ModuleList([nn.Linear(dim_feedforward, dim_feedforward, bias=False) for _ in range(self.hop_number)])
            self.W_n = nn.ModuleList([nn.Linear(dim_feedforward, dim_feedforward, bias=False) for _ in range(self.hop_number)])
            self.W_r = nn.ModuleList([nn.Linear(dim_feedforward, dim_feedforward, bias=False) for _ in range(self.hop_number)])
            self.gate_linear = nn.Linear(dim_feedforward, 1)

    def forward(self,
                his, his_mask, his_len, src, src_mask, src_len, trg, tgt_mask,
                src_padding_mask, tgt_padding_mask, 
                inner_info, inner_info_mask,
                memory, memory_mask, turn,  # memory
                concept_ids, concept_label, distance, relation, head, tail, triple_label, vocab_map, vocab_mask,
                ):
        src = src
        trg = trg
        src_mask = src_mask
        tgt_mask = tgt_mask

        # encoding
        if self.config.encoding_mode == 'no_his':
            src = src
            src_mask = src_mask
        elif self.config.encoding_mode == 'inp_cat_2':
            src = torch.cat([his, src], 0)
            src_seq_len = src.shape[0]
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
            src_padding_mask = (src == 1).transpose(0, 1)
        elif self.config.encoding_mode == 'his_src_inp':
            his_src_max_len = his_len.max().item() + src_len.max().item() - 2
            his_src_inp = []
            for idx, b in enumerate(his.transpose(0, 1)):
                new_b = torch.cat([b[:his_len[idx].item() - 1], src.transpose(0, 1)[idx][1:src_len[idx].item()]], -1)
                if new_b.size(0) < his_src_max_len:
                    new_b = torch.cat([new_b, torch.ones(his_src_max_len - new_b.size(0)).long().to(his.device)], -1)
                his_src_inp.append(new_b.unsqueeze(0))
            src = torch.cat(his_src_inp).to(src.device).transpose(0, 1)
            src_seq_len = src.shape[0]
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
            src_padding_mask = (src == 1).transpose(0, 1)

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        encoded = self.transformer.encoder(src_emb, src_mask, src_padding_mask)

        # 1hop
        if self.config.only_hop1:
            encoded_inner_list = []
            inner_key_padding_mask_list = []
            for inner in inner_info:
                inner_len = inner.size(0)
                inner_mask = torch.zeros((inner_len, inner_len), device=src.device).type(torch.bool)
                inner_out = self.transformer.encoder(self.positional_encoding(self.src_tok_emb(inner)), inner_mask)
                inner_key_padding_mask = (inner == 1).transpose(0, 1)
                inner_key_padding_mask_list.append(inner_key_padding_mask)
                encoded_inner_list.append(inner_out)
            encoded_inner = torch.cat(encoded_inner_list, 0).to(src.device)
            inner_key_padding_mask = torch.cat(inner_key_padding_mask_list, 1).to(src.device)
            o_in = self.dropout1(self.multihead_attn(encoded, encoded_inner, encoded_inner,
                                                         key_padding_mask=inner_key_padding_mask)[0])
            alpha = torch.sigmoid(self.alpha_linear(encoded))
            encoded = alpha * o_in + encoded

        if self.config.memory:
            # 4. Knowledge selector
            encoded_inner_list = []
            inner_key_padding_mask_list = []
            for inner in inner_info:
                inner_len = inner.size(0)
                inner_mask = torch.zeros((inner_len, inner_len), device=src.device).type(torch.bool)
                inner_out = self.transformer.encoder(self.positional_encoding(self.src_tok_emb(inner)), inner_mask)
                inner_key_padding_mask = (inner == 1).transpose(0, 1)
                inner_key_padding_mask_list.append(inner_key_padding_mask)
                encoded_inner_list.append(inner_out)
            encoded_inner = torch.cat(encoded_inner_list, 0).to(src.device)
            inner_key_padding_mask = torch.cat(inner_key_padding_mask_list, 1).to(src.device)
            o_in = self.dropout1(self.multihead_attn(encoded, encoded_inner, encoded_inner,
                                                     key_padding_mask=inner_key_padding_mask)[0])
            alpha = torch.sigmoid(self.alpha_linear(encoded))
            if turn != 0:
                attn_weights_list = []
                # encoded_post, _ = self.encode(src, src_mask)
                encoded_post = self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)
                post_mask = (src == 1).transpose(0, 1)
                for len_i in range(encoded_post.size(0)):
                    energy = self.attn(torch.cat((encoded_post[len_i, :, :].unsqueeze(0).expand(memory.size(0), -1, -1), memory), 2)).tanh()
                    attn_energies = torch.sum(self.v * energy, dim=2)
                    attn_energies = attn_energies.t()
                    attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
                    attn_weights_list.append(attn_weights)

                attn_weights = torch.cat(attn_weights_list, 1)
                memory = attn_weights.bmm(memory.transpose(0, 1)).transpose(0, 1)
                o_m = self.dropout_m(self.multihead_attn_m(encoded, memory, memory, key_padding_mask=post_mask)[0])
                encoded = alpha * o_in + (1 - alpha) * o_m + encoded
                memory = torch.cat([memory, encoded_inner], 0)
            else:
                encoded = alpha * o_in + encoded  # 将选择到的inner和encoded加起来
                memory = encoded_inner  # 记录turn 0的历史inner知识

        # decoding
        encoded_key_padding_mask = src_padding_mask
        decoded = self.transformer.decoder(tgt_emb, encoded, tgt_mask=tgt_mask, memory_mask=None,
                                           tgt_key_padding_mask=tgt_padding_mask,
                                           memory_key_padding_mask=encoded_key_padding_mask)
        # outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
        #                         src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        out = self.generator(decoded)

        # 2hop
        if self.config.hop2:
            concept_ids = self.src_tok_emb(concept_ids)  # (b,l,h)
            rel_repr = self.relation_embd(relation)  # (b,l,h)
            # concept_ids, concept_label: 400
            # head, tail, rel_repr, triple_label: 800
            node_repr, rel_repr = self.multi_layer_comp_gcn(concept_ids, rel_repr, head, tail, concept_label,
                                                            triple_label, layer_number=self.hop_number)
            # node_repr: 400
            # rel_repr: 800
            node_repr = node_repr.transpose(0, 1)
            rel_repr = rel_repr.transpose(0, 1)
            head = head.transpose(0, 1)
            tail = tail.transpose(0, 1)
            head_repr = torch.gather(node_repr, 1,
                                     head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
            tail_repr = torch.gather(node_repr, 1,
                                     tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
            encoded = encoded.transpose(0, 1)
            triple_weight = torch.softmax(torch.matmul(encoded, ((head_repr + rel_repr) / 2).transpose(1, 2)), -1)  # (b, en_l, head_l)
            triple_repr = torch.matmul(triple_weight, tail_repr)  # (b, en_l, h)
            # out
            decoded = decoded.transpose(0, 1)
            out = out.transpose(0, 1)
            vocab_map = vocab_map.transpose(0, 1)
            vocab_mask = vocab_mask.transpose(0, 1)
            out = F.softmax(out, dim=2)

            # old
            # triple_de_weight = torch.softmax(torch.matmul(decoded, triple_repr.transpose(1, 2)), -1)
            # triple_de_repr = torch.matmul(triple_de_weight, triple_repr)

            # new
            triple_de_weight_list = []
            for t_idx in range(decoded.size(1)):
                triple_de_weight = F.softmax(torch.sum(decoded[:, t_idx, :].unsqueeze(1) * triple_repr, -1), 1).unsqueeze(1)
                triple_de_weight_list.append(triple_de_weight)
            triple_de_weight = torch.cat(triple_de_weight_list, 1)
            triple_de_repr = triple_de_weight.bmm(triple_repr)
            #

            cpt_prob = F.softmax(triple_de_repr, dim=-1)  # (b, de_l, l)
            cpt_probs_vocab = cpt_prob.gather(2, vocab_map.unsqueeze(1).expand(cpt_prob.size(0), cpt_prob.size(1), -1))
            cpt_probs_vocab.masked_fill_((vocab_mask == 0).unsqueeze(1), 0)
            # cpt_prob: (8,50,vocab)
            # out_prob: (8,24,vocab)
            gate = torch.sigmoid(self.gate_linear(decoded))
            out = cpt_probs_vocab * (1 - gate) + gate * out
            out = out.clamp(min=1e-5).log().transpose(0, 1)

        return out, memory, None
        # return self.generator(decoded), None, None

    def encode(self, src: Tensor, src_mask: Tensor, inner_info=None, memory=None, turn=None):
        encoded = self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)
        o_in_attn, o_m_attn = None, None
        if self.config.only_hop1:
            inner_info_cat = torch.cat(inner_info, 0)
            inner_info_len = inner_info_cat.size(0)
            inner_info_mask = torch.zeros((inner_info_len, inner_info_len), device=src.device).type(torch.bool)
            inner_info_out = self.transformer.encoder(self.positional_encoding(self.src_tok_emb(inner_info_cat)),
                                                      inner_info_mask)
            memory_key_padding_mask = (inner_info_cat == 1).transpose(0, 1)
            o_in_out, o_in_attn = self.multihead_attn(encoded, inner_info_out, inner_info_out,
                                                     key_padding_mask=memory_key_padding_mask)
            o_in = self.dropout1(o_in_out)
            alpha = torch.sigmoid(self.alpha_linear(encoded))
            encoded = alpha * o_in + encoded

        if self.config.memory:
            # 4. Knowledge selector
            encoded_inner_list = []
            inner_key_padding_mask_list = []
            for inner in inner_info:
                inner_len = inner.size(0)
                inner_mask = torch.zeros((inner_len, inner_len), device=src.device).type(torch.bool)
                inner_out = self.transformer.encoder(self.positional_encoding(self.src_tok_emb(inner)), inner_mask)
                inner_key_padding_mask = (inner == 1).transpose(0, 1)
                inner_key_padding_mask_list.append(inner_key_padding_mask)
                encoded_inner_list.append(inner_out)
            encoded_inner = torch.cat(encoded_inner_list, 0).to(src.device)
            inner_key_padding_mask = torch.cat(inner_key_padding_mask_list, 1).to(src.device)

            o_in_out, o_in_attn = self.multihead_attn(encoded, encoded_inner, encoded_inner,
                                                      key_padding_mask=inner_key_padding_mask)
            o_in = self.dropout1(o_in_out)
            # o_in = self.dropout1(self.multihead_attn(encoded, encoded_inner, encoded_inner,
            #                                          key_padding_mask=inner_key_padding_mask)[0])
            alpha = torch.sigmoid(self.alpha_linear(encoded))
            if turn != 0:
                attn_weights_list = []
                encoded_post = self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)
                post_mask = (src == 1).transpose(0, 1)
                for len_i in range(encoded_post.size(0)):
                    energy = self.attn(torch.cat((encoded_post[len_i, :, :].unsqueeze(0).expand(memory.size(0), -1, -1), memory), 2)).tanh()
                    attn_energies = torch.sum(self.v * energy, dim=2)
                    attn_energies = attn_energies.t()
                    attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
                    attn_weights_list.append(attn_weights)

                attn_weights = torch.cat(attn_weights_list, 1)
                memory = attn_weights.bmm(memory.transpose(0, 1)).transpose(0, 1)
                o_m_out, o_m_attn = self.multihead_attn_m(encoded, memory, memory, key_padding_mask=post_mask)
                o_m = self.dropout1(o_in_out)
                # o_m = self.dropout_m(self.multihead_attn_m(encoded, memory, memory, key_padding_mask=post_mask)[0])
                encoded = alpha * o_in + (1 - alpha) * o_m + encoded
                memory = torch.cat([memory, encoded_inner], 0)
            else:
                encoded = alpha * o_in + encoded  # 将选择到的inner和encoded加起来
                memory = encoded_inner  # 记录turn 0的历史inner知识
        return encoded, memory, o_in_attn, o_m_attn

    def decode(self, tgt: Tensor, encoded: Tensor, tgt_mask: Tensor,
               concept_ids=None, concept_label=None, distances=None, relation=None, head=None, tail=None,
               triple_label=None, vocab_map=None, vocab_mask=None):
        decoded = self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), encoded,
            tgt_mask)
        out = decoded
        triple_de_weight, gate = None, None
        if self.config.hop2:
            concept_ids = self.src_tok_emb(concept_ids)  # (b,l,h)
            rel_repr = self.relation_embd(relation)  # (b,l,h)
            node_repr, rel_repr = self.multi_layer_comp_gcn(concept_ids, rel_repr, head, tail, concept_label,
                                                            triple_label, layer_number=self.hop_number)
            # concept_ids, concept_label: 117
            # head, tail, rel_repr, triple_label: 246

            node_repr = node_repr.transpose(0, 1)
            rel_repr = rel_repr.transpose(0, 1)
            head = head.transpose(0, 1)
            tail = tail.transpose(0, 1)
            head_repr = torch.gather(node_repr, 1,
                                     head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
            tail_repr = torch.gather(node_repr, 1,
                                     tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
            # head_repr, tail_repr: 246
            encoded = encoded.transpose(0, 1)
            triple_weight = torch.softmax(torch.matmul(encoded, ((head_repr + rel_repr) / 2).transpose(1, 2)), -1)  # (b, de_l, head_l)
            triple_repr = torch.matmul(triple_weight, tail_repr)  # (b, en_l, h)
            # out
            decoded = decoded.transpose(0, 1)
            out = self.generator(out.transpose(0, 1))
            vocab_map = vocab_map.transpose(0, 1)
            vocab_mask = vocab_mask.transpose(0, 1)
            out = F.softmax(out, dim=2)

            # old
            # triple_de_weight = torch.softmax(torch.matmul(decoded, triple_repr.transpose(1, 2)), -1)
            # triple_de_repr = torch.matmul(triple_de_weight, triple_repr)

            # new
            triple_de_weight_list = []
            for t_idx in range(decoded.size(1)):
                triple_de_weight = F.softmax(torch.sum(decoded[:, t_idx, :].unsqueeze(1) * triple_repr, -1),
                                             1).unsqueeze(1)
                triple_de_weight_list.append(triple_de_weight)
            triple_de_weight = torch.cat(triple_de_weight_list, 1)
            triple_de_repr = triple_de_weight.bmm(triple_repr)
            #

            cpt_prob = F.softmax(triple_de_repr, dim=-1)  # (b, 50, l)
            cpt_probs_vocab = cpt_prob.gather(2, vocab_map.unsqueeze(1).expand(cpt_prob.size(0), cpt_prob.size(1), -1))
            cpt_probs_vocab.masked_fill_((vocab_mask == 0).unsqueeze(1), 0)
            # cpt_prob: (8,50,vocab)
            # out_prob: (8,24,vocab)
            gate = torch.sigmoid(self.gate_linear(decoded))
            out = cpt_probs_vocab * (1 - gate) + gate * out
            out = out.clamp(min=1e-5).log().transpose(0, 1)
        return out, triple_de_weight, gate

    def multi_layer_comp_gcn(self, concept_hidden, relation_hidden, head, tail, concept_label, triple_label,
                             layer_number=2):
        for i in range(layer_number):
            concept_hidden, relation_hidden = self.comp_gcn(concept_hidden, relation_hidden, head, tail, concept_label,
                                                            triple_label, i)
        return concept_hidden, relation_hidden

    def comp_gcn(self, concept_hidden, relation_hidden, head, tail, concept_label, triple_label, layer_idx):
        '''
        concept_hidden: mem x bsz x hidden
        relation_hidden: mem_t x bsz x hidden
        '''
        bsz = head.size(1)
        mem_t = head.size(0)
        mem = concept_hidden.size(0)
        hidden_size = concept_hidden.size(2)

        update_node = torch.zeros_like(concept_hidden).to(concept_hidden.device).float()
        count = torch.ones_like(head).to(head.device).masked_fill_(triple_label == -1, 0).float()
        # count_out = torch.zeros(bsz, mem).to(head.device).float()
        count_out = torch.zeros(mem, bsz).to(head.device).float()

        o = concept_hidden.gather(0, head.unsqueeze(2).expand(mem_t, bsz, hidden_size))
        # o = concept_hidden.gather(1, head.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        # scatter_add(o, tail, dim=1, out=update_node)  # 只能在gpu下正常运行
        scatter_add(o, tail, dim=0, out=update_node)  # 只能在gpu下正常运行
        # o.shape
        # Out[10]: torch.Size([16, 800, 768])
        # tail.shape
        # Out[11]: torch.Size([16, 800])
        # update_node.shape
        # Out[12]: torch.Size([16, 400, 768])
        scatter_add(- relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), tail, dim=0, out=update_node)
        scatter_add(count, tail, dim=0, out=count_out)
        o = concept_hidden.gather(0, tail.unsqueeze(2).expand(mem_t, bsz, hidden_size))
        # o = concept_hidden.gather(0, tail.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=0, out=update_node)
        scatter_add(- relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), head, dim=0, out=update_node)
        scatter_add(count, head, dim=0, out=count_out)

        act = nn.ReLU()
        update_node = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_node) / count_out.clamp(
            min=1).unsqueeze(2)
        update_node = act(update_node)

        return update_node, self.W_r[layer_idx](relation_hidden)
