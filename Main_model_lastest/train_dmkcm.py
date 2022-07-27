# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from Main_model_lastest.model.dmkcm import DMKCM
from Main_model_lastest.configs_manage import DVKCM_config
from Main_model_lastest.model.utils_dmkcm import *
from tqdm import tqdm
from torchtext.data import Field
import time
import math

def pad(tensor, pad_len=50):
    batch = tensor.size(1)
    tensor_len = tensor.size(0)
    pad_tensor = torch.ones(pad_len - tensor_len, batch).to(tensor.device).long()
    padded_tensor = torch.cat([tensor, pad_tensor], 0)
    return padded_tensor


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


# count_parameters
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_vocab_map(concepts_tensor, vocab):
    new_vocab_map_tensor = []
    new_vocab_mask_tensor = []
    for batch_con in concepts_tensor.transpose(0, 1):
        concepts = batch_con[batch_con!=1].tolist()
        vocab_map = []
        map_mask = []
        for idx in vocab.keys():  # 遍历词表的索引值
            try:
                pos = concepts.index(idx)  # _concept_ids中词的位置
                vocab_map.append(pos)
                map_mask.append(1)
            except:
                vocab_map.append(0)
                map_mask.append(0)
        new_vocab_map_tensor.append(torch.tensor(vocab_map).unsqueeze(1))
        new_vocab_mask_tensor.append(torch.tensor(map_mask).unsqueeze(1))

    return torch.cat(new_vocab_map_tensor, 1).long().to(device), torch.cat(new_vocab_mask_tensor, 1).long().to(device)


def train(trans_config, model, iterator, transformer_optimizer, criterion, clip, itos):
    if trans_config.paral:
        model.module.train()
    else:
        model.train()
    epoch_loss = 0
    global best_train_loss
    for i, batch in enumerate(tqdm(iterator)):
        if trans_config.hop2:
            batch_data = [
                [batch.h1, batch.p1, batch.r1,
                 batch.one_hop1_1, batch.one_hop1_2, batch.one_hop1_3, batch.one_hop1_4, batch.one_hop1_5,
                 batch.concepts1, batch.labels1, batch.distances1, batch.relations1, batch.head_ids1, batch.tail_ids1
                    , batch.triple_labels1],
                [batch.h2, batch.p2, batch.r2,
                 batch.one_hop2_1, batch.one_hop2_2, batch.one_hop2_3, batch.one_hop2_4, batch.one_hop2_5,
                 batch.concepts2, batch.labels2, batch.distances2, batch.relations2, batch.head_ids2, batch.tail_ids2,
                 batch.triple_labels2],
                [batch.h3, batch.p3, batch.r3,
                 batch.one_hop3_1, batch.one_hop3_2, batch.one_hop3_3, batch.one_hop3_4, batch.one_hop3_5,
                 batch.concepts3, batch.labels3, batch.distances3, batch.relations3, batch.head_ids3, batch.tail_ids3,
                 batch.triple_labels3],
                [batch.h4, batch.p4, batch.r4,
                 batch.one_hop4_1, batch.one_hop4_2, batch.one_hop4_3, batch.one_hop4_4, batch.one_hop4_5,
                 batch.concepts4, batch.labels4, batch.distances4, batch.relations4, batch.head_ids4, batch.tail_ids4,
                 batch.triple_labels4]]
            loss = 0
            turn_num = 0
            memory = None
            memory_mask = None
            for line in batch_data:
                his_con, his_con_len = line[0]
                post, post_len = line[1]
                response, response_len = line[2]
                one_hop_1, _ = line[3]
                one_hop_2, _ = line[4]
                one_hop_3, _ = line[5]
                one_hop_4, _ = line[6]
                one_hop_5, _ = line[7]
                concepts = line[8]
                labels = line[9]
                distances = line[10]
                relations = line[11]
                head_ids = line[12]
                tail_ids = line[13]
                triple_labels = line[14]
                vocab_map, vocab_mask = get_vocab_map(concepts, itos)

                inner_list = [one_hop_1, one_hop_2, one_hop_3, one_hop_4, one_hop_5]
                inner = []
                inner_maxlen = 0
                for inn in inner_list:
                    if len(inn) > inner_maxlen:
                        inner_maxlen = len(inn)
                for inn in inner_list:
                    inner.append(pad(inn, inner_maxlen))
                inner_mask = [(in_mask != 1).unsqueeze(1).unsqueeze(1) for in_mask in inner]

                his_mask = (his_con != 1).to(device).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words)

                transformer_optimizer.zero_grad()
                # transformer_optimizer.optimizer.zero_grad()

                # Move to device
                question = post.to(device)
                reply = response.to(device)

                # Prepare Target Data
                # reply_input = reply[:, :-1]
                # reply_target = reply[:, 1:]
                reply_input, reply_target = reply[:-1, :], reply[1:, :]

                # Create mask and add dimensions
                # question_mask, reply_input_mask, reply_target_mask = create_masks(device, question, reply_input, reply_target)

                question_mask, reply_input_mask, src_padding_mask, tgt_padding_mask = create_masks(device, question,
                                                                                                   reply_input,
                                                                                                   reply_target)

                # print('===============================')
                # print('His: ', tensor2str(his_con, itos))
                # print('Post: ', tensor2str(question, itos))
                # print('Reply: ', tensor2str(reply_input, itos))

                # Get the transformer outputs
                out, memory, memory_mask = model(his_con, his_mask, his_con_len, question, question_mask, post_len, reply_input, reply_input_mask,
                                    src_padding_mask, tgt_padding_mask, inner, inner_mask,
                                    memory, memory_mask, turn_num,
                                    concepts, labels, distances, relations, head_ids, tail_ids, triple_labels,
                                    vocab_map, vocab_mask
                                    )
                turn_num += 1

                # Compute the loss
                reply_target_mask = reply_target != 1
                loss += criterion(out.transpose(0, 1), reply_target.transpose(0, 1), reply_target_mask.transpose(0, 1))

                if trans_config.print_train:
                    batch_num = 0
                    print('\n===============================')
                    print('His: ', tensor2str(his_con[batch_num], itos))
                    print('Post: ', tensor2str(question[batch_num], itos))
                    print('Reply_lable: ', tensor2str(reply_input[batch_num], itos))
                    print('Generated_reply: ', tensor2str(out[batch_num].argmax(1), itos))
                ###

            # loss = loss / 4
            if i % 50 == 0:
                print("Epoch", epoch + 1, "loss: ", loss.item())
                # global best_train_loss
                if best_train_loss > loss.item():
                    best_train_loss = loss.item()
                    if trans_config.save_ckp:
                        torch.save(model.state_dict(), params_path)
                        print('已保存参数文件')
                        with open(trans_config.log_path, 'a+') as f:
                            f.write("\nEpoch" + str(epoch + 1) + "\tloss" + str(loss.item()))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            transformer_optimizer.step()

            epoch_loss += loss.item()

        else:
            batch_data = [
                [batch.h1, batch.p1, batch.r1,
                 batch.one_hop1_1, batch.one_hop1_2, batch.one_hop1_3, batch.one_hop1_4, batch.one_hop1_5],
                [batch.h2, batch.p2, batch.r2,
                 batch.one_hop2_1, batch.one_hop2_2, batch.one_hop2_3, batch.one_hop2_4, batch.one_hop2_5],
                [batch.h3, batch.p3, batch.r3,
                 batch.one_hop3_1, batch.one_hop3_2, batch.one_hop3_3, batch.one_hop3_4, batch.one_hop3_5],
                [batch.h4, batch.p4, batch.r4,
                 batch.one_hop4_1, batch.one_hop4_2, batch.one_hop4_3, batch.one_hop4_4, batch.one_hop4_5]]
            loss = 0
            turn_num = 0
            memory = None
            memory_mask = None
            for line in batch_data:
                his_con, his_con_len = line[0]
                post, post_len = line[1]
                response, response_len = line[2]
                one_hop_1, _ = line[3]
                one_hop_2, _ = line[4]
                one_hop_3, _ = line[5]
                one_hop_4, _ = line[6]
                one_hop_5, _ = line[7]

                inner_list = [one_hop_1, one_hop_2, one_hop_3, one_hop_4, one_hop_5]
                inner = []
                inner_maxlen = 0
                for inn in inner_list:
                    if len(inn) > inner_maxlen:
                        inner_maxlen = len(inn)
                for inn in inner_list:
                    inner.append(pad(inn, inner_maxlen))
                inner_mask = [(in_mask != 1).unsqueeze(1).unsqueeze(1) for in_mask in inner]

                his_mask = (his_con != 1).to(device).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words)

                transformer_optimizer.zero_grad()
                # transformer_optimizer.optimizer.zero_grad()

                # Move to device
                question = post.to(device)
                reply = response.to(device)

                # Prepare Target Data
                # reply_input = reply[:, :-1]
                # reply_target = reply[:, 1:]
                # reply_input, reply_target = reply, reply
                reply_input, reply_target = reply[:-1, :], reply[1:, :]

                # Create mask and add dimensions
                # question_mask, reply_input_mask, reply_target_mask = create_masks(device, question, reply_input, reply_target)

                question_mask, reply_input_mask, src_padding_mask, tgt_padding_mask = create_masks(device, question,
                                                                                                   reply_input,
                                                                                                   reply_target)

                # print('===============================')
                # print('His: ', tensor2str(his_con, itos))
                # print('Post: ', tensor2str(question, itos))
                # print('Reply: ', tensor2str(reply_input, itos))

                concepts, labels, distances, relations, head_ids, tail_ids, triple_labels, vocab_map, vocab_mask = \
                    None, None, None, None, None, None, None, None, None

                # Get the transformer outputs
                out, memory, memory_mask = model(his_con, his_mask, his_con_len, question, question_mask, post_len, reply_input, reply_input_mask,
                                    src_padding_mask, tgt_padding_mask, inner, inner_mask,
                                    memory, memory_mask, turn_num,
                                    concepts, labels, distances, relations, head_ids, tail_ids, triple_labels,
                                    vocab_map, vocab_mask
                                    )
                turn_num += 1

                # Compute the loss
                loss += criterion(out.reshape(-1, out.shape[-1]), reply_target.reshape(-1))
                # loss += criterion(out, reply_target, reply_target_mask)

                ###

            # loss = loss / 4
            if i % 50 == 0:
                print("Epoch", epoch + 1, "loss: ", loss.item())
                # global best_train_loss
                if best_train_loss > loss.item():
                    best_train_loss = loss.item()
                    if trans_config.save_ckp:
                        torch.save(model.state_dict(), params_path)
                        print('已保存参数文件')
                        with open(trans_config.log_path, 'a+') as f:
                            f.write("\nEpoch" + str(epoch + 1) + "\tloss" + str(loss.item()))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            transformer_optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(trans_config, model, iterator, criterion, itos):
    if trans_config.paral:
        model.module.eval()
    else:
        model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            if trans_config.hop2:
                batch_data = [
                    [batch.h1, batch.p1, batch.r1,
                     batch.one_hop1_1, batch.one_hop1_2, batch.one_hop1_3, batch.one_hop1_4, batch.one_hop1_5,
                     batch.concepts1, batch.labels1, batch.distances1, batch.relations1, batch.head_ids1,
                     batch.tail_ids1
                        , batch.triple_labels1],
                    [batch.h2, batch.p2, batch.r2,
                     batch.one_hop2_1, batch.one_hop2_2, batch.one_hop2_3, batch.one_hop2_4, batch.one_hop2_5,
                     batch.concepts2, batch.labels2, batch.distances2, batch.relations2, batch.head_ids2,
                     batch.tail_ids2,
                     batch.triple_labels2],
                    [batch.h3, batch.p3, batch.r3,
                     batch.one_hop3_1, batch.one_hop3_2, batch.one_hop3_3, batch.one_hop3_4, batch.one_hop3_5,
                     batch.concepts3, batch.labels3, batch.distances3, batch.relations3, batch.head_ids3,
                     batch.tail_ids3,
                     batch.triple_labels3],
                    [batch.h4, batch.p4, batch.r4,
                     batch.one_hop4_1, batch.one_hop4_2, batch.one_hop4_3, batch.one_hop4_4, batch.one_hop4_5,
                     batch.concepts4, batch.labels4, batch.distances4, batch.relations4, batch.head_ids4,
                     batch.tail_ids4,
                     batch.triple_labels4]]
                loss = 0
                turn_num = 0
                memory = None
                memory_mask = None
                for line in batch_data:
                    his_con, his_con_len = line[0]
                    post, post_len = line[1]
                    response, response_len = line[2]
                    one_hop_1, _ = line[3]
                    one_hop_2, _ = line[4]
                    one_hop_3, _ = line[5]
                    one_hop_4, _ = line[6]
                    one_hop_5, _ = line[7]
                    concepts = line[8]
                    labels = line[9]
                    distances = line[10]
                    relations = line[11]
                    head_ids = line[12]
                    tail_ids = line[13]
                    triple_labels = line[14]
                    vocab_map, vocab_mask = get_vocab_map(concepts, itos)

                    inner_list = [one_hop_1, one_hop_2, one_hop_3, one_hop_4, one_hop_5]
                    inner = []
                    inner_maxlen = 0
                    for inn in inner_list:
                        if len(inn) > inner_maxlen:
                            inner_maxlen = len(inn)
                    for inn in inner_list:
                        inner.append(pad(inn, inner_maxlen))
                    inner_mask = [(in_mask != 1).unsqueeze(1).unsqueeze(1) for in_mask in inner]

                    his_mask = (his_con != 1).to(device).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words)

                    transformer_optimizer.zero_grad()
                    # transformer_optimizer.optimizer.zero_grad()

                    # Move to device
                    question = post.to(device)
                    reply = response.to(device)

                    # Prepare Target Data
                    # reply_input = reply[:, :-1]
                    # reply_target = reply[:, 1:]
                    # reply_input, reply_target = reply, reply
                    reply_input, reply_target = reply[:-1, :], reply[1:, :]

                    # Create mask and add dimensions
                    # question_mask, reply_input_mask, reply_target_mask = create_masks(device, question, reply_input,
                    #                                                                   reply_target)

                    question_mask, reply_input_mask, src_padding_mask, tgt_padding_mask = create_masks(device, question,
                                                                                                       reply_input,
                                                                                                       reply_target)

                    # print('===============================')
                    # print('His: ', tensor2str(his_con, itos))
                    # print('Post: ', tensor2str(question, itos))
                    # print('Reply: ', tensor2str(reply_input, itos))

                    # Get the transformer outputs
                    out, memory, memory_mask = model(his_con, his_mask, his_con_len, question, question_mask, post_len, reply_input, reply_input_mask,
                                    src_padding_mask, tgt_padding_mask, inner, inner_mask,
                                    memory, memory_mask, turn_num,
                                    concepts, labels, distances, relations, head_ids, tail_ids, triple_labels,
                                    vocab_map, vocab_mask
                                    )
                    turn_num += 1

                    # Compute the loss
                    reply_target_mask = reply_target != 1
                    loss += criterion(out.transpose(0, 1), reply_target.transpose(0, 1), reply_target_mask.transpose(0, 1))

                    ###
            else:
                batch_data = [
                    [batch.h1, batch.p1, batch.r1,
                     batch.one_hop1_1, batch.one_hop1_2, batch.one_hop1_3, batch.one_hop1_4, batch.one_hop1_5],
                    [batch.h2, batch.p2, batch.r2,
                     batch.one_hop2_1, batch.one_hop2_2, batch.one_hop2_3, batch.one_hop2_4, batch.one_hop2_5],
                    [batch.h3, batch.p3, batch.r3,
                     batch.one_hop3_1, batch.one_hop3_2, batch.one_hop3_3, batch.one_hop3_4, batch.one_hop3_5],
                    [batch.h4, batch.p4, batch.r4,
                     batch.one_hop4_1, batch.one_hop4_2, batch.one_hop4_3, batch.one_hop4_4, batch.one_hop4_5]]
                loss = 0
                turn_num = 0
                memory = None
                memory_mask = None
                for line in batch_data:
                    his_con, his_con_len = line[0]
                    post, post_len = line[1]
                    response, response_len = line[2]
                    one_hop_1, _ = line[3]
                    one_hop_2, _ = line[4]
                    one_hop_3, _ = line[5]
                    one_hop_4, _ = line[6]
                    one_hop_5, _ = line[7]

                    inner_list = [one_hop_1, one_hop_2, one_hop_3, one_hop_4, one_hop_5]
                    inner = []
                    inner_maxlen = 0
                    for inn in inner_list:
                        if len(inn) > inner_maxlen:
                            inner_maxlen = len(inn)
                    for inn in inner_list:
                        inner.append(pad(inn, inner_maxlen))
                    inner_mask = [(in_mask != 1).unsqueeze(1).unsqueeze(1) for in_mask in inner]

                    his_mask = (his_con != 1).to(device).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words)

                    transformer_optimizer.zero_grad()
                    # transformer_optimizer.optimizer.zero_grad()

                    # Move to device
                    question = post.to(device)
                    reply = response.to(device)

                    # Prepare Target Data
                    # reply_input = reply[:, :-1]
                    # reply_target = reply[:, 1:]
                    # reply_input, reply_target = reply, reply
                    reply_input, reply_target = reply[:-1, :], reply[1:, :]

                    # Create mask and add dimensions
                    # question_mask, reply_input_mask, reply_target_mask = create_masks(device, question, reply_input,
                    #                                                                   reply_target)
                    question_mask, reply_input_mask, src_padding_mask, tgt_padding_mask = create_masks(device, question,
                                                                                                       reply_input,
                                                                                                       reply_target)

                    # print('===============================')
                    # print('His: ', tensor2str(his_con, itos))
                    # print('Post: ', tensor2str(question, itos))
                    # print('Reply: ', tensor2str(reply_input, itos))

                    concepts, labels, distances, relations, head_ids, tail_ids, triple_labels, vocab_map, vocab_mask = \
                        None, None, None, None, None, None, None, None, None

                    # Get the transformer outputs
                    out, memory, memory_mask = model(his_con, his_mask, his_con_len, question, question_mask, post_len, reply_input, reply_input_mask,
                                    src_padding_mask, tgt_padding_mask, inner, inner_mask,
                                    memory, memory_mask, turn_num,
                                    concepts, labels, distances, relations, head_ids, tail_ids, triple_labels,
                                    vocab_map, vocab_mask
                                    )
                    turn_num += 1

                    # Compute the loss
                    loss += criterion(out.reshape(-1, out.shape[-1]), reply_target.reshape(-1))
                    # loss += criterion(out, reply_target, reply_target_mask)

                    ###
            epoch_loss += loss.item()

        batch_num = 0
        print('\n===============================')
        print('His: ', tensor2str(his_con[:,batch_num], itos))
        print('Post: ', tensor2str(question[:,batch_num], itos))
        print('Reply_lable: ', tensor2str(reply_target[:,batch_num], itos))
        print('Generated_reply: ', tensor2str(out[:,batch_num].argmax(1), itos))
        with open(trans_config.log_path, 'a+') as f:
            # f.write("\n===============================")
            f.write("\nHis: " + str(tensor2str(his_con[:,batch_num], itos)))
            f.write("\nPost: " + str(tensor2str(question[:,batch_num], itos)))
            f.write("\nReply_lable: " + str(tensor2str(reply_target[:,batch_num], itos)))
            f.write("\nGenerated_reply: " + str(tensor2str(out[:,batch_num].argmax(1), itos)))

    return epoch_loss / len(iterator)


if __name__ == '__main__':
    # personaChat、dailyDialog
    trans_config = DVKCM_config()
    print('\n'.join(['%s:%s' % item for item in trans_config.__dict__.items()]))
    with open(trans_config.log_path, 'a+') as f:
        f.write('\n'.join(['%s:%s' % item for item in trans_config.__dict__.items()]))
    POST = Field(tokenize=None,
                 init_token='<sos>',
                 eos_token='<eos>',
                 include_lengths=True,
                 batch_first=False)
    HOP1 = None
    CONCEPTS = Field(use_vocab=False,
                     fix_length=400,
                     pad_token=1,
                     batch_first=False
                     )
    LABELS = Field(use_vocab=False,
                   fix_length=400,
                   pad_token=-1,
                   batch_first=False)
    DISTENCES = Field(use_vocab=False,
                      fix_length=400,
                      pad_token=0,
                      batch_first=False)
    H_T_R = Field(use_vocab=False,
                  fix_length=800,
                  pad_token=0,
                  batch_first=False)
    TRIPLE_LABELS = Field(use_vocab=False,
                          fix_length=800,
                          pad_token=-1,
                          batch_first=False)
    # include_lengths: 是否返回一个已经补全的最小batch的元组和和一个包含每条数据长度的列表 . 默认值: False.
    # pad_token: 用于补全的字符. 默认值: “”.
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    POST, train_iterator, valid_iterator, device, embed = DataLoader(trans_config, device, POST, HOP1, CONCEPTS, LABELS, DISTENCES, H_T_R,
                                                              TRIPLE_LABELS)

    # model parameter
    print(device)
    print('正在使用显卡设备: ', torch.cuda.current_device())

    vocab = json.load(open(trans_config.word_dict_path))
    word_map = len(vocab)  # INPUT_DIM
    itos = {value: key for key, value in vocab.items()}

    print('vocab size: ', word_map)

    d_model = 512
    heads = 8
    num_layers = 6

    SRC_VOCAB_SIZE = word_map
    TGT_VOCAB_SIZE = word_map
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    # BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    model = DMKCM(trans_config, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(device)

    if trans_config.paral:
        model = nn.DataParallel(model, device_ids=trans_config.gpu)

    # judge if exist pre_model_param
    params_path = trans_config.params_path
    model_path = trans_config.model_path
    start_epoch = 0
    if trans_config.load_ckp:
        '''
        if os.path.exists(params_path):
            model.load_state_dict(torch.load(params_path))
            print("load params from {}".format(params_path))
        elif os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print("load epoch-model")
        '''
        epoch_list = os.listdir(model_path)
        start_epoch = max([int(x[x.index('h') + 1:]) for x in epoch_list]) + 1
    else:
        # Initial weight
        model.apply(init_weights)
        print("checkpoint not exists,need initial weight")

    # initial word embedding
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Loss
    transformer_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    if trans_config.hop2:
        criterion = LossWithLS(word_map, 0.1)
        # criterion = nn.NLLLoss(ignore_index=1, reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=POST.vocab.stoi[POST.pad_token])

    # initial train param
    N_EPOCHS = trans_config.N_EPOCHS
    CLIP = trans_config.CLIP

    best_valid_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = -1

    # train process
    for epoch in range(start_epoch, N_EPOCHS):

        start_time = time.time()

        train_loss = train(trans_config, model, train_iterator, transformer_optimizer, criterion, CLIP, itos)
        valid_loss = evaluate(trans_config, model, valid_iterator, criterion, itos)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        with open(trans_config.log_path, 'a+') as f:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                if trans_config.save_ckp:
                    torch.save(model.state_dict(),
                               model_path + '/dmkcm_' + trans_config.name + '_epoch' + str(epoch))
                    print("已保存best epoch {}的参数文件".format(epoch + 1))
                    f.write("\n已保存best epoch {}的参数文件".format(epoch + 1))

            elif epoch % 5 == 0 and epoch > 20:
                torch.save(model.state_dict(), model_path + '/dmkcm_' + trans_config.name + '_epoch' + str(epoch))
                print("已保存epoch {}的参数文件".format(epoch + 1))
                f.write("\n已保存epoch {}的参数文件".format(epoch + 1))

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:5.2f} | Train PPL: {math.exp(train_loss):8.2f}')
            print(f'\t Val. Loss: {valid_loss:5.2f} |  Val. PPL: {math.exp(valid_loss):8.2f}')
            f.write('\nEpoch: ' + str(epoch + 1) + '| Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
            f.write('\nTrain Loss: ' + str(train_loss) + ' | Train PPL: ' + str(math.exp(train_loss)))
            f.write('\nVal. Loss: ' + str(valid_loss) + ' |  Val. PPL: ' + str(math.exp(valid_loss)))

            print('best_train_loss: ', best_train_loss)
            print('best_valid_loss: ', best_valid_loss)
            print('best_epoch: ', best_epoch + 1)
            f.write('\nbest_train_loss: ' + str(best_train_loss))
            f.write('\nbest_valid_loss: ' + str(best_valid_loss))
            f.write('\nbest_epoch: ' + str(best_epoch + 1))
            f.write("\n===============================")

