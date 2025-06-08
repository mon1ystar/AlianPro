
# import new Network name here and add in model_class args
from email.mime import base
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# from .Network import MYNET



from utils import *
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertModel
import numpy as np
import torch

class MYLoss(nn.Module):
    def __init__(self):
        super(MYLoss, self).__init__()
        

    def forward(self, logits, labels):
        # logits: [batch_size, num_classes]
        # labels: [batch_size]

        # 提取每个样本对应标签的logit值
        batch_size = logits.size(0)  # 获取批次大小
        selected_logits = logits[torch.arange(batch_size), labels]
        #loss = 0.5 * (torch.acos(selected_logits) **2)
        loss = 0.5 * (selected_logits - 1) ** 2
        weighted_loss = loss
        # 计算损失的平均值
        final_loss = torch.mean(weighted_loss)
        return 10 * final_loss

class MYSmothLoss(nn.Module):
    def __init__(self):
        super(MYSmothLoss, self).__init__()

    def forward(self, logits, labels):
        # logits: [batch_size, num_classes]
        # labels: [batch_size]
        batch_size = logits.size(0)

        # 提取每个样本对应标签的logit值（即余弦相似度）
        selected_logits = logits[torch.arange(batch_size), labels]

        # 最大余弦对齐：1 - cos(θ)
        loss = 1.0 - selected_logits  # 越接近1越好

        # 平均后返回
        final_loss = torch.mean(loss)
        return 10 * final_loss  # 可调 scale

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    features: Tensor of shape [B, D] - hidden_embedding
    labels: Tensor of shape [B] - class ids
    """
    features = F.normalize(features, dim=1)  # cosine 相似度前要归一化
    batch_size = features.shape[0]

    similarity_matrix = torch.matmul(features, features.T)  # [B, B] 相似度矩阵
    logits = similarity_matrix / temperature

    labels = labels.contiguous().view(-1, 1)  # [B, 1]
    mask = torch.eq(labels, labels.T).float().cuda()  # [B, B] - 正样本 mask

    # 去掉自己和自己的匹配（对角线）
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size).cuda()
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    # 计算每个 anchor 的 loss（正样本对数概率平均）
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

    # 损失：负的对数概率
    loss = -mean_log_prob_pos
    return loss.mean()
    
def orthogonality_loss(vcls):
    """
    vcls: Tensor of shape [b, 10, 768]
    Returns: A scalar loss encouraging orthogonality among the 10 vectors
    """
    # Normalize vectors along the feature dimension (dim=2)
    vcls_normalized = torch.nn.functional.normalize(vcls, dim=2)
    
    # Compute pairwise cosine similarity: [b, 10, 10]
    similarity_matrix = torch.matmul(vcls_normalized, vcls_normalized.transpose(1, 2))
    
    # Mask the diagonal elements (self-similarity)
    identity_matrix = torch.eye(similarity_matrix.size(1), device=similarity_matrix.device).unsqueeze(0)
    
    # Remove diagonal (self-similarity) and compute sum of squared off-diagonal elements
    off_diagonal_loss = ((similarity_matrix - identity_matrix) ** 2).sum(dim=(1, 2))/( similarity_matrix.size(1) * similarity_matrix.size(2))
    
    # Return the mean loss across the batch
    return  1 * off_diagonal_loss.mean()




def logits_to_zero_loss(logits):
    """
    Loss function to minimize the distance of logits to zero.
    
    Args:
        logits: Tensor of shape [b, c], where b is batch size and c is number of classes.
    
    Returns:
        loss: A scalar tensor representing the loss.
    """
    loss = torch.mean( 0.5 * (logits ** 2))  # Mean squared loss over all logits
    return 10 * loss
def pairwise_cosine_loss(cls_embedding, cls_embedding_v):
    """
    计算 cls_embedding 和 cls_embedding_v 中所有向量两两之间的余弦相似度，并设计损失。

    Args:
        cls_embedding (torch.Tensor): 形状为 [b, 768] 的张量。
        cls_embedding_v (torch.Tensor): 形状为 [b, 768] 的张量。

    Returns:
        torch.Tensor: 余弦相似度最小化的损失。
    """
    # 对每个向量进行 L2 归一化
    cls_embedding_norm = F.normalize(cls_embedding, dim=1)  # 形状 [b, 768]
    cls_embedding_v_norm = F.normalize(cls_embedding_v, dim=1)  # 形状 [b, 768]
    
    # 计算两两余弦相似度，结果形状为 [b, b]
    cos_sim_matrix = torch.matmul(cls_embedding_norm, cls_embedding_v_norm.T)  # [b, b]
    
    # 损失为相似度的均值（可以根据需求调整）
    loss = torch.mean(cos_sim_matrix** 2)  # 越小越好
    return 5 * loss

def compute_averagecos(prompt):
    # 归一化每个 prompt，使其模为 1
    normalized_prompt = F.normalize(prompt, dim=1)  # shape: (10, 768)

    # 计算余弦相似度矩阵
    cosine_similarity_matrix = torch.matmul(normalized_prompt, normalized_prompt.T)
    # 创建掩码（对角线为 True，其它为 False）
    batch_size = cosine_similarity_matrix.size(0)
    mask = torch.eye(batch_size, dtype=torch.bool)

    # 提取非对角元素
    non_diag_elements = cosine_similarity_matrix[~mask]  # 提取所有非对角元素

    # 计算非对角元素的平均值
    non_diag_mean = (non_diag_elements**2).mean()
    
    return 10 * non_diag_mean

def build_label_embedding(train_set,session,Bert_model,tokenizer,word_info, args):
    if args.dataset == "cifar100":
        classes = np.unique(train_set.classes)
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes[classes_int]))
    elif args.dataset == "mini_imagenet":
        classes = np.unique(train_set.wnids)
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes[classes_int]))
    elif args.dataset == "cub200" or args.dataset == "air":
        classes = np.unique(np.array(train_set.labels)[train_set.targets])
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes))
        
    else:
        raise KeyError
    
    words_embed = []
    with torch.no_grad():
        Bert_model.eval()
        if args.dataset in ['cifar100', 'mini_imagenet']:
            for cls in classes[classes_int]:
                if args.pret_clip:
                    encoded_input = Bert_model.tokenizer(f'a photo of {cls}')
                    output = Bert_model.text_encoder.encode_text(encoded_input.cuda())
                    # words_embed.append(bert_map(output))
                    words_embed.append(output)
                    word_info["label_text"] = np.append(word_info["label_text"], cls)
                else:
                    encoded_input = tokenizer(f'a photo of {cls}', return_tensors='pt')
                    output = Bert_model(**encoded_input)
                    # words_embed.append(bert_map(output.pooler_output))
                    words_embed.append(output.pooler_output)
                    word_info["label_text"] = np.append(word_info["label_text"], cls)
        elif args.dataset in ['cub200', 'air']:
            for cls in classes:
                if args.pret_clip:
                    encoded_input = Bert_model.tokenizer(f'a photo of {cls}')
                    output = Bert_model.text_encoder.encode_text(encoded_input.cuda())
                    words_embed.append(output)
                    word_info["label_text"] = np.append(word_info["label_text"], f'a photo of {cls}')
                else:
                    encoded_input = tokenizer(f'a photo of {cls}', return_tensors='pt')
                    output = Bert_model(**encoded_input)
                    # words_embed.append(bert_map(output.pooler_output))
                    words_embed.append(output.pooler_output)
                    word_info["label_text"] = np.append(word_info["label_text"], f'a photo of {cls}')
        else:
            raise KeyError
        
    words_embed = torch.cat(words_embed,dim=0)
    
    if word_info["embed"] == None:
        word_info["embed"] = words_embed.cpu()
    else:
        word_info["embed"] = torch.cat([word_info["embed"].cpu(),words_embed.cpu()],dim=0)
        
    word_info["cur_embed"] = words_embed.cpu()
    word_info["cur_label"] = torch.tensor(classes_int).cpu()


def replace_base_fc(trainset, transform, model, args):
    print("[Replace Base FC - Original]")
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=4, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            if args.pret_clip:
                embedding = model([data, label], query=True)
            
            else:
                logit,  cls_embedding = model(data, base=True)
            embedding_list.append(cls_embedding.cpu())
            label_list.append(label.cpu())
        
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.classifiers[0].weight.data = proto_list.cuda()

    return model

def cross_entropy(preds, targets, reduction='none'):
    labels = torch.arange(targets.shape[0]).cuda()
    loss = F.cross_entropy(preds,labels, reduction='none')
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
def cosine_distillation_loss(student_features, teacher_features):
    """
    使用余弦相似度定义的蒸馏损失。
    Args:
        student_features (torch.Tensor): 学生模型的特征，形状为 [N, D]。
        teacher_features (torch.Tensor): 教师模型的特征，形状为 [N, D]。
    Returns:
        torch.Tensor: 余弦相似度蒸馏损失。
    """
    # 对特征进行归一化
    student_features = torch.nn.functional.normalize(student_features, dim=-1)
    teacher_features = torch.nn.functional.normalize(teacher_features, dim=-1)
    
    # 计算余弦相似度
    cosine_similarity = (student_features * teacher_features).sum(dim=-1)
    
    # 蒸馏损失为 1 - 平均余弦相似度
    loss = ((1 - cosine_similarity)**2).mean()
    
    return  0.4*loss

    


def map_range(tensor, old_min, old_max, new_min, new_max):
    return ((tensor - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min


def base_train(model, trainloader, optimizer, scheduler, epoch, word_info, query_info, class_list, args, loss_curve):
    print("[Base Train]")
    base_mode = model.module.mode
    
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader, mininterval=1.0)
    myloss = MYLoss()
    myloss1 = MYSmothLoss()
    model.module.mode = "encoder"
    
    
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
       
        logits = model(data)

        criterion = nn.CrossEntropyLoss()

        loss = criterion(logits, train_label)
        
        acc = count_acc(logits, train_label)
        total_loss = loss
        
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        ta.add(acc, len(train_label))
        
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} ,acc={:.4f}'.format(epoch, lrc, total_loss.item(),ta.item()))
        
        optimizer.zero_grad()
        total_loss.backward()
        
        
        
        optimizer.step()
        
    tl = tl.item()
    ta = ta.item()
    
    model.module.mode = base_mode
    return tl, ta


def triplet(cls_embed, vision_embed, query_info, train_label,loss_curve):
    P_head = query_info['proto'].clone().cuda()
    
    cls_logit = F.linear(cls_embed, P_head)
    cls_gt = F.cross_entropy(cls_logit, train_label, reduction='none')   #* B
    vis_logit = F.linear(vision_embed, P_head)
    vis_gt = F.cross_entropy(vis_logit, train_label, reduction='none')   #* B
    
    idx = torch.arange(vis_logit.shape[0])
    
    cls_logit[idx, train_label]=0.
    vis_logit[idx, train_label]=0.
    
    l_kl = F.kl_div(F.log_softmax(vis_logit,dim=1), F.softmax(cls_logit,dim=1), reduction='batchmean')
    l_ent = vis_gt.mean() + cls_gt.mean()
    
    loss_tri = ((l_ent/l_kl)+1).log()
    return loss_tri

def knowledge_boosting(lang_embed, word_embed, query_info, train_label, loss_curve):
    T = 2.
    idx= torch.arange(len(train_label))
    #* Original
    P_head = query_info['proto'].clone().cuda()
    
    #* =======================================================================
    lang_logit = F.linear(lang_embed, P_head)    #* Soft pred
    loss_seman = F.cross_entropy(lang_logit, train_label)
    #* KL Feature
    loss_kd = F.kl_div(F.log_softmax(lang_embed/T,dim=1), F.softmax(word_embed[train_label]/T,dim=1), reduction='batchmean')
    
    loss = loss_kd + 0.2*loss_seman
    return 0.1*loss


def test(model, testloader, epoch, args, session, word_info):
    #todo Test시 Prompt Selection is needed..
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager_Loss()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
        # 初始化累积列表
    all_true_labels = []
    all_pred_labels = []
    print("\t\t\t[Test Phase] Session: {}".format(session))
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            
            
            logits = model(data)
            
            logits = logits[:, :test_class]
            
            loss = F.cross_entropy(logits, test_label)
            
            # 累积真实标签和预测标签
            _, preds = torch.max(logits, 1)
            all_true_labels.append(test_label)
            all_pred_labels.append(preds)
            
            acc = count_acc(logits, test_label)

            base_idxs = test_label < args.base_class
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs])
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs])
                va_base.add(acc_base, len(test_label[base_idxs]))
                va_base_given_new.add(acc_base_given_new, len(test_label[base_idxs]))


            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class)
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs])
                va_new.add(acc_new, len(test_label[new_idxs]))
                va_new_given_base.add(acc_new_given_base, len(test_label[new_idxs]))

            vl.add(loss.item(), len(test_label))
            va.add(acc, len(test_label))
            
        

        vl = vl.item()
        va = va.item()

        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    print('base only accuracy: {:.4f}, new only accuracy: {:.4f}'.format(va_base, va_new))
    print('base acc given new : {:.4f}'.format(va_base_given_new))
    print('new acc given base : {:.4f}'.format(va_new_given_base))

    logs = dict(num_session=session + 1, acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new,
                new_acc_given_base=va_new_given_base)

    return vl, va, logs


def test_my(model, testloader, epoch, args, session, word_info):
    #todo Test시 Prompt Selection is needed..
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager_Loss()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
        # 初始化累积列表
    all_true_labels = []
    all_pred_labels = []
    logits_base_correct = []
    logits_new_misclassified_to_base = []       
    all_features = []
    all_labels = []
    top5_logits_base = []
    top5_logits_novel = []
    wrong_novel_top5 = []
    correct_base_top5 = []

    print("\t\t\t[Test Phase] Session: {}".format(session))
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits, cls_embedding,hidden_embedding,vlogits,vcls,= model(data)
            all_features.append(hidden_embedding.cpu())
            all_labels.append(test_label.cpu())

            logits = logits[:, :test_class]
            
            loss = F.cross_entropy(logits, test_label)

            # 找出新类样本（真实标签在 base_class 之后）
            novel_mask = test_label >= args.base_class

            # 找出预测错误且预测为基类的样本
            pred_label = logits.argmax(dim=1)
            pred_base_mask = pred_label < args.base_class

            # 组合：新类 + 被错分为基类
            wrong_novel_mask = novel_mask & pred_base_mask

            if wrong_novel_mask.any():
                # 提取这些错分样本的 logits
                top5_values, _ = torch.topk(logits[wrong_novel_mask], k=5, dim=1)
                wrong_novel_top5.append(top5_values.detach().cpu())

            # ==== 被正确分类的基类样本 ====
            base_mask = test_label < args.base_class
            correct_mask = test_label == pred_label
            correct_base_mask = base_mask & correct_mask

            if correct_base_mask.any():
                top5_values_correct, _ = torch.topk(logits[correct_base_mask], k=5, dim=1)
                correct_base_top5.append(top5_values_correct.detach().cpu())
            
            # 累积真实标签和预测标签
            _, preds = torch.max(logits, 1)
            # threshold = 0.2 # 你可以自由设定或 tune 这个阈值
            
            # logit_values, raw_preds = torch.max(logits, dim=1)
            # final_preds = raw_preds.clone()  # 复制一份作为最终预测

            # for i in range(len(test_label)):
            #     pred_class = raw_preds[i].item()
            #     logit_score = logit_values[i].item()

            #     # 被预测为基类，且 logit 低于阈值 → 放弃基类预测
            #     if pred_class < args.base_class and logit_score < threshold:
            #         # 仅在新类 logits 中重新预测
            #         new_logits = logits[i, args.base_class:]
            #         new_pred = torch.argmax(new_logits) + args.base_class  # 注意偏移
            #         final_preds[i] = new_pred


            # all_true_labels.append(test_label)
            # all_pred_labels.append(final_preds)
            
            acc = (preds == test_label).sum().item()

            # # 1️⃣ 基类中被正确分类的样本
            # base_correct_mask = (test_label < args.base_class) & (final_preds == test_label)
            # if torch.any(base_correct_mask):
            #     correct_logits = logits[base_correct_mask, test_label[base_correct_mask]]
            #     logits_base_correct.append(correct_logits.detach().cpu())

            # # 2️⃣ 新类被错误预测为基类的样本
            # new_to_base_mask = (test_label >= args.base_class) & (final_preds < args.base_class)
            # if torch.any(new_to_base_mask):
            #     wrong_logits = logits[new_to_base_mask, final_preds[new_to_base_mask]]
            #     logits_new_misclassified_to_base.append(wrong_logits.detach().cpu())

            base_idxs = test_label < args.base_class
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs])
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs])
                va_base.add(acc_base, len(test_label[base_idxs]))
                va_base_given_new.add(acc_base_given_new, len(test_label[base_idxs]))


            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class)
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs])
                va_new.add(acc_new, len(test_label[new_idxs]))
                va_new_given_base.add(acc_new_given_base, len(test_label[new_idxs]))

            vl.add(loss.item(), len(test_label))
            va.add(acc, len(test_label))
        # 最后拼接为一个大的 tensor，方便画图或统计
        wrong_novel_top5 = torch.cat(wrong_novel_top5, dim=0)  # shape: (N_base, 5)
        correct_base_top5 = torch.cat(correct_base_top5, dim=0)  # shape: (N_novel, 5)
        save_top5_logits_to_csv(wrong_novel_top5, correct_base_top5, "top5_logits_{}.csv".format(session))
        # all_features = torch.cat(all_features, dim=0).numpy()  # shape: (N, 768)
        # all_labels = torch.cat(all_labels, dim=0).numpy()
            
        # plot_tsne(all_features,all_labels,session)

        vl = vl.item()
        va = va.item()

        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
    logits_base_correct = torch.cat(logits_base_correct, dim=0) if logits_base_correct else torch.tensor([])
    logits_new_misclassified_to_base = torch.cat(logits_new_misclassified_to_base, dim=0) if logits_new_misclassified_to_base else torch.tensor([])

    print(f"\n[统计] 基类正确预测时的 logit 平均值: {logits_base_correct.mean().item():.4f} ± {logits_base_correct.std().item():.4f}")
    print(f"[统计] 新类错误预测到基类时的 logit 平均值: {logits_new_misclassified_to_base.mean().item():.4f} ± {logits_new_misclassified_to_base.std().item():.4f}")
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    print('base only accuracy: {:.4f}, new only accuracy: {:.4f}'.format(va_base, va_new))
    print('base acc given new : {:.4f}'.format(va_base_given_new))
    print('new acc given base : {:.4f}'.format(va_new_given_base))

    logs = dict(num_session=session + 1, acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new,
                new_acc_given_base=va_new_given_base)

    return vl, va, logs

def plot_tsne(features, labels, session,title="t-SNE of Features", perplexity=30, random_state=42):
    """
    使用 t-SNE 可视化特征

    Args:
        features (Tensor or ndarray): shape [N, D] 的特征向量
        labels (Tensor or ndarray): shape [N,] 的类别标签
        title (str): 图标题
        perplexity (int): t-SNE 参数，平衡局部/全局结构，建议 5~50
        random_state (int): 随机种子
    """
    # Tensor 转 numpy（如果需要）
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 执行 t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=random_state)
    features_2d = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", np.unique(labels).shape[0])
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=labels, palette=palette, legend='full', s=12, alpha=0.8
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f'hidden_tsne/{session}_tsne_features.png', dpi=600, bbox_inches="tight")

def build_base_proto(train_loader, model, args):
    model = model.eval()
    
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            data, label = [_.cuda() for _ in batch]
            
            model.module.mode = 'encoder'
            if args.pret_clip:
                embedding = model([data, label], query=True)
            else:
                embedding = model(data, query=True)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
            
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0) #* num_base, feat_dim
    
    model.module.mode = args.base_mode
    model = model.train()
    
    return proto_list


def plot_confusion_matrix(true_labels, pred_labels, num_classes, va,figsize=(20, 16), normalize=False, title='Confusion Matrix'):
    """
    绘制混淆矩阵的函数。

    参数:
        true_labels (np.array or torch.Tensor): 真实标签。
        pred_labels (np.array or torch.Tensor): 预测标签。
        num_classes (int): 类别数量。
        figsize (tuple): 图像大小，默认为 (20, 16)。
        normalize (bool): 是否对混淆矩阵进行归一化，默认为 False。
        title (str): 图像标题，默认为 'Confusion Matrix'。
    """
    # 如果输入是 PyTorch 张量，转换为 NumPy 数组
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 如果归一化，计算比例
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘制混淆矩阵
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=np.arange(num_classes),
                yticklabels=np.arange(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig('confusion_matrix_{}.png'.format(num_classes))
    
def mixup(data, labels, alpha=0.2, new_class_offset=100):
    device = data.device
    batch_size = data.size(0)
    
    # 确保输入标签在 0~99 范围内
    assert labels.max() < 100, "原始标签应属于 0~99"
    
    # 为每个样本随机选择不同类别的配对样本
    rand_index = torch.zeros_like(labels)
    for i in range(batch_size):
        # 找到所有与当前样本不同类别的候选索引
        valid_indices = torch.where(labels != labels[i])[0]
        if len(valid_indices) > 0:
            rand_index[i] = valid_indices[torch.randint(0, len(valid_indices), (1,))]
        else:
            rand_index[i] = i  # 如果没有不同类别，则配对自身（不推荐）
    
    # 生成混合系数 lam ~ Beta(alpha, alpha)
    lam = torch.from_numpy(np.random.beta(alpha, alpha, batch_size)).float().to(device)
    lam = lam.view(-1, 1, 1, 1)  # 扩展维度以匹配数据形状
    
    # 混合数据
    mixed_data = lam * data + (1 - lam) * data[rand_index, :]
    
    # 生成新类别ID：原类别ID + 配对类别ID + 基数100
    new_labels = (labels + labels[rand_index] + new_class_offset).clamp(max=new_class_offset+99)
    
    return mixed_data, new_labels

def save_top5_logits_to_csv(wrong_tensor: torch.Tensor, correct_tensor: torch.Tensor, filepath: str = "top5_logits_analysis.csv"):
    
    # 转为 DataFrame
    df_wrong = pd.DataFrame(wrong_tensor.numpy(), columns=[f"logit_{i+1}" for i in range(5)])
    df_correct = pd.DataFrame(correct_tensor.numpy(), columns=[f"logit_{i+1}" for i in range(5)])

    # 添加分组标记
    df_wrong["group"] = "wrong_novel"
    df_correct["group"] = "correct_base"

    # 合并并保存
    df_all = pd.concat([df_wrong, df_correct], ignore_index=True)
    df_all.to_csv(filepath, index=False)
    print(f"✅ Top-5 logits 已保存到: {filepath}")

#* ===============================================================================================================
# import gc
# import cv2
# from torchvision.transforms import transforms

# class VITAttentionRollout:
#     def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.8, attn_score=False):
#         #*self.model = model.module.encoder.cuda()
#         self.model = model.module
#         self.model.eval()
#         self.model.mode = "encoder"
#         self.head_fusion = head_fusion
#         self.discard_ratio = discard_ratio
#         self.attn_score = attn_score
#         self.handles = []
        
#         for name, module in self.model.named_modules():
#             if attention_layer_name in name:
#                 self.handles.append(module.register_forward_hook(self.get_attention))

#         self.attentions = []

#     def get_attention(self, module, input, output):
#         print("[get_attention]output:",output.shape)
#         if output.size(-1)>197:
#             sep = output.size(-1)-197
#             self.attentions.append(output[:,:,:,1+sep:].cpu())
#         else:
#             self.attentions.append(output.cpu())
        
#     def __call__(self, input_tensor):
#         self.attentions = []
#         with torch.no_grad():
#             cls_feat, prompt_feat = self.model.prompt_encode(input_tensor ,prompt_feat=True, B_tuning=True)

#         if self.attn_score:
#             return self.attentions
#         else:
#             return rollout(self.attentions, self.discard_ratio, self.head_fusion)

# def rollout(attentions, discard_ratio, head_fusion):
#     result = torch.eye(attentions[0].size(-1))
#     with torch.no_grad():
#         for attention in attentions:
#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(axis=1)
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(axis=1)[0]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(axis=1)[0]
#             else:
#                 raise "Attention head fusion type Not supported"

#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
#             indices = indices[indices != 0]
#             flat[0, indices] = 0

#             I = torch.eye(attention_heads_fused.size(-1))
#             a = (attention_heads_fused + 1.0*I)/2
#             a = a / a.sum(dim=-1)

#             result = torch.matmul(a, result)

#     mask = result[0, 0 , 1 :]
#     width = int(mask.size(-1)**0.5)
#     mask = mask.reshape(width, width).numpy()
#     mask = mask / np.max(mask)
#     return mask

# def imshow(inp, title=None):
#     """Display image for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.axis('off')
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# def denormalize(img):
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
#     img = np.clip(255.0 * (img * IMAGENET_STD + IMAGENET_MEAN), 0, 255)
#     return img

# def show_mask_on_image(img, mask):
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam), heatmap

# import copy
# def visualize_attn_map(model, loader, train_set, args, path, attn_score=False):
#     #todo Numpy로 모든 이미지 Attn Map 생성
#     #todo label & Task ID와 매칭하여 폴더별로 저장
#     #todo Cherry picking
#     validate_path(path)
#     if args.dataset == "cifar100":
#         classes = np.unique(train_set.classes)
#         classes_int = np.unique(train_set.targets)
#     elif args.dataset == "mini_imagenet":
#         classes = np.unique(train_set.wnids)
#         classes_int = np.unique(train_set.targets)
#     elif args.dataset == "cub200":
#         classes = np.unique(np.array(train_set.labels)[train_set.targets])
#         classes_int = np.unique(train_set.targets)
        
#     rollout_model = VITAttentionRollout(model,head_fusion='mean',discard_ratio=0.8, attn_score=attn_score)
#     with torch.no_grad():
#         for i, batch in enumerate(loader, 1):
#             #! Original
#             #! data, test_label = [_.cuda() for _ in batch]
#             data, test_label = [_.cuda() for _ in batch]
#             # data = data[:98]
#             for i in range(data.shape[0]):
#                 img = data[i]
#                 img_w, img_h = img.shape[-2], img.shape[-1]
#                 #! rollout_model = VITAttentionRollout(model,head_fusion='mean',discard_ratio=0.8)
#                 mask = rollout_model(img.unsqueeze(0))
#                 # print("Attn Score:", mask.shape)
#                 if attn_score:
#                     return mask
#                 #* img = transforms.Resize((32,32))(img).detach().cpu().numpy()
#                 img = img.detach().cpu().numpy()
#                 img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]
#                 img = denormalize(img) # *255 or IMAGENET denorm 방법
#                 np_img = np.array(img)[:,:,::-1]
#                 mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
#                 mask,heatmap = show_mask_on_image(np_img, mask)
#                 resize_mask = cv2.resize(mask, (512, 512), fx=0.3, fy=0.7, interpolation=cv2.INTER_LANCZOS4)

#                 # cv2.imwrite(f'/content/Viz/Mask_cifar100_sample_{i}_{class_names[ori_targets[i]]}.png',resize_mask)
#                 #* cv2_imshow(resize_mask)
#                 img = img.astype(np.uint8).copy() # np.float32 -> np.uint8
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB 채널

#                 resize_img = cv2.resize(img, (512, 512), fx=0.3, fy=0.7, interpolation=cv2.INTER_LANCZOS4)
#                 # cv2.imwrite(f'/content/Viz/Img_cifar100_sample_{i}_{class_names[ori_targets[i]]}.png',resize_img)
#                 # cv2_imshow(resize_img)
                
#                 img_cat = cv2.hconcat([resize_img, resize_mask])
#                 cv2.imwrite(f'{path}/hcat_{args.dataset}_sample_{i}_{classes[test_label[i]]}.png',img_cat)
        
#         for handle in rollout_model.handles:
#             handle.remove()

# def count_wise_acc(logits, label, cls_mat, cls_samples):
#     pred = torch.argmax(logits, dim=1)
#     for idx, gt in enumerate(label):
#         cls_samples[gt] += 1.
#         if pred[idx] == gt:
#             cls_mat[gt]+=1.
    
#     return cls_mat, cls_samples


# def class_wise_test(model, testloader, epoch, args, session, word_info):
#     test_classes = ['Nighthaw', 'Least_Aukle', 'Western_Wood_Pewe', 'Warbling_Vire', 'Common_Ter', 'Pigeon_Guillemo',
#                     'House_Wre', 'Baird_Sparro', 'Rufous_Hummingbir', 'Le_Conte_Sparro']
#     test_idx = []
#     for test_cls in test_classes:
#         test_idx.append(test_classes.index(test_cls))
#     test_idx = np.array(test_idx)
#     #todo Test시 Prompt Selection is needed..
#     if args.dataset == "cub200":
#         classes = np.unique(np.array(testloader.dataset.labels))
#     elif args.dataset == "cifar100":
#         classes = np.unique(testloader.dataset.classes)
#     else:
#         print("SOMETHING IS WEIRD!!")
#         return
    
#     cls_mat = torch.tensor([0. for _ in range(len(classes))])   #* Correct Count
#     cls_samples = torch.tensor([0. for _ in range(len(classes))])   #* Sample count
    
    
#     test_class = args.base_class + session * args.way
#     model = model.eval()
#     vl = Averager_Loss()
#     va = Averager()
#     va_base = Averager()
#     va_new = Averager()
#     va_base_given_new = Averager()
#     va_new_given_base = Averager()
#     with torch.no_grad():
#         tqdm_gen = tqdm(testloader)
#         for i, batch in enumerate(tqdm_gen, 1):
#             #! Original
#             #! data, test_label = [_.cuda() for _ in batch]
#             data, test_label = [_.cuda() for _ in batch]
            
#             if args.pret_clip:
#                 out = model([data, test_label],word_info=word_info)
#                 logits = out['logit_pred']
#             else:
#                 #! B-Tuning해야 B-Prompt들어가지?
#                 logits = model(data, B_tuning=True)
#                 logits = logits[:, :test_class]
            
#             cls_mat, cls_samples = count_wise_acc(logits, test_label, cls_mat, cls_samples)
        
#         top_val, top_idx = torch.topk(cls_mat, k=200)
#         bot_val, bot_idx = torch.topk(cls_mat, k=200, largest=False)
        
#         print('Experiment Classes:', classes)
#         accs = (cls_mat/cls_samples)*100.
#         print('Bottom-ACC:', accs)
#         validate_path(f"/data/pgh2874/FSCIL/Ours/Class_Wise_ACC/{args.out}_CUB200")
#         np.save(f"/data/pgh2874/FSCIL/Ours/Class_Wise_ACC/{args.out}_CUB200/Seed{args.seed}_Classes.npy", classes)
#         np.save(f"/data/pgh2874/FSCIL/Ours/Class_Wise_ACC/{args.out}_CUB200/Seed{args.seed}_accs.npy", accs.detach().cpu().numpy())

# from sklearn.manifold import TSNE
# import os

# def validate_path(path):
#     if os.path.exists(path):
#         pass
#     else:
#         print('create folder:', path)
#         os.makedirs(path)
    
# def visualization(train_set, test_loader, model, args):
#     classes = np.unique(np.array(train_set.labels)[train_set.targets])
    
#     model.eval()
#     ori_mode = model.module.mode
#     model.module.mode = 'encoder'
#     embedding_list = []
#     label_list = []
#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):
#             data, test_label = [_.cuda() for _ in batch]
#             cls_embed, prompt_embed = model(data, prompt_feat=True, B_tuning=True)
#             embedding = 0.5*(prompt_embed['Vision']+cls_embed).cpu()
#             embedding_list.append(embedding)
#             label_list.append(test_label.cpu())
        
#     embedding_list = torch.cat(embedding_list, dim=0)
#     label_list = torch.cat(label_list, dim=0)
#     print("embedding_list:",embedding_list.shape)
#     print("label_list:",label_list.shape)    
    
    
    
    
#     path = f'TSNE_VIZ/{args.out}_train/' #? Epoch마다 찍자.. 
#     validate_path(path)
#     perplexties = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#     for trial in range(10):
#         tsne = TSNE(n_components=2, perplexity=25, random_state=0, learning_rate=perplexties[trial], n_iter=10000, init='pca') #todo Inc Session에서 사용할 경우 Perplexity 수정 필요
#         # tsne = TSNE(n_components=2, random_state=0) #todo Inc Session에서 사용할 경우 Perplexity 수정 필요
        
#         # select_cls = torch.randperm(args.base_class)[:10]
#         #* 60, 40, 14, 36, 5
#         select_cls = torch.arange(args.base_class)[trial*10:(trial+1)*10]
#         print("select_classes:", select_cls)
        
#         marker=['o','^','*','s','p','P','h','+','x','D']
#         color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
#                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
#         plt.figure(figsize=(8, 8))
#         for c_idx, cls_id in enumerate(select_cls):
#             data_index = np.where(label_list == cls_id)
            
#             cls_prompt_embed = embedding_list[data_index[0]].detach().cpu()
#             print("class:",cls_id)
#             print('idx:',data_index[0].shape)
#             print("feature_embed:", cls_prompt_embed.shape)
#             emb = np.array(tsne.fit_transform(np.array(cls_prompt_embed)))
#             print("TSNE feature:", emb.shape)
#             plt.scatter(emb[:, 0], emb[:, 1], label=classes[cls_id.item()])
            
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(path+f'Seed_{args.seed}_Trial_{trial}_Base_class_10.png', dpi=600)
        

# def visualize_ED_feature(train_set, test_loader, model, args):
#     classes = np.unique(np.array(train_set.labels)[train_set.targets])
    
#     model.eval()
#     ori_mode = model.module.mode
#     model.module.mode = 'encoder'
#     cls_embedding_list=[]
#     vis_embedding_list=[]
#     embedding_list = []
#     label_list = []
#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):
#             data, test_label = [_.cuda() for _ in batch]
#             cls_embed, prompt_embed = model(data, prompt_feat=True, B_tuning=True)
#             embedding = 0.5*(prompt_embed['Vision']+cls_embed).cpu()
#             embedding_list.append(embedding)
#             cls_embedding_list.append(cls_embed.cpu())
#             vis_embedding_list.append(prompt_embed['Vision'].cpu())
            
#             label_list.append(test_label.cpu())
        
#     embedding_list = torch.cat(embedding_list, dim=0)
#     cls_embedding_list = torch.cat(cls_embedding_list, dim=0)
#     vis_embedding_list = torch.cat(vis_embedding_list, dim=0)
#     label_list = torch.cat(label_list, dim=0)
#     print("embedding_list:",embedding_list.shape)
#     print("label_list:",label_list.shape)    
    
#     path = f'TSNE_VIZ/ED_Loss_Token_Feature/{args.out}_train/' #? Epoch마다 찍자.. 
#     validate_path(path)
#     perplexties = [15, 25, 31, 32, 33, 34, 35, 36, 37, 38]
#     for trial in range(10):
#         tsne = TSNE(n_components=2, perplexity=25, random_state=0, learning_rate=perplexties[trial], n_iter=10000, init='pca') #todo Inc Session에서 사용할 경우 Perplexity 수정 필요
#         # tsne = TSNE(n_components=2, random_state=0) #todo Inc Session에서 사용할 경우 Perplexity 수정 필요
        
#         # select_cls = torch.randperm(args.base_class)[:10]
#         #* 60, 40, 14, 36, 5
#         select_cls = torch.arange(args.base_class)[trial*10:(trial+1)*10]
#         print("select_classes:", select_cls)
        
#         marker=['o','^','*','s','p','P','h','+','x','D']
#         color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
#                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
#         fig, ax = plt.subplots(figsize=(8, 8))
#         for c_idx, cls_id in enumerate(select_cls):
#             data_index = np.where(label_list == cls_id)
#             #*cls_prompt_embed = embedding_list[data_index[0]].detach().cpu()
#             cls_emb = cls_embedding_list[data_index[0]].detach().cpu()
#             vis_emb = vis_embedding_list[data_index[0]].detach().cpu()
#             print("class:",cls_id)
#             print('idx:',data_index[0].shape)
#             #*emb = np.array(tsne.fit_transform(np.array(cls_prompt_embed)))
#             cls_emb = np.array(tsne.fit_transform(np.array(cls_emb)))
#             vis_emb = np.array(tsne.fit_transform(np.array(vis_emb)))
            
            
#             if (c_idx+1)==len(select_cls):
#                 plt.scatter(cls_emb[:, 0], cls_emb[:, 1], marker='o', alpha=0.3, color=color[c_idx], label=classes[cls_id.item()])
#                 plt.scatter(vis_emb[:, 0], vis_emb[:, 1], marker='^', alpha=0.3, color=color[c_idx])
#             else:
#                 plt.scatter(cls_emb[:, 0], cls_emb[:, 1], alpha=0.3, marker='o', color=color[c_idx], label=classes[cls_id.item()])
#                 plt.scatter(vis_emb[:, 0], vis_emb[:, 1], alpha=0.3, marker='^', color=color[c_idx])
            
#         leg1 = plt.legend(loc='lower left', bbox_to_anchor=(1.01,0.2), fontsize=10)
#         # leg1 = plt.legend(bbox_to_anchor=(1.03,0), fontsize=10)
#         ax.add_artist(leg1)
#         h = [plt.plot([],[], color="gray", marker="o", ls="", label='[CLS] Token')[0], plt.plot([],[], color="gray", marker="^", ls="", label='Vis Token')[0]]
#         leg2 = plt.legend(handles=h, loc="lower left", bbox_to_anchor=(1.01,0.8), fontsize=10)
#         plt.tight_layout()
#         plt.savefig(path+f'Vis_CLS_Seed_{args.seed}_Trial_{trial}_Base_class_10.png', dpi=600, bbox_inches="tight")                
#         #todo ========================
        
# import csv
# def check_lamda(model, inputs, attn_score=False):
#     #todo Numpy로 모든 이미지 Attn Map 생성
#     #todo label & Task ID와 매칭하여 폴더별로 저장
#     #todo Cherry picking
#     rollout_model = VITAttentionRollout(model, head_fusion='mean',discard_ratio=0.8, attn_score=attn_score)
#     with torch.no_grad():
#         attn_score = rollout_model(inputs)
#         attn_score = torch.stack(attn_score)
    
#     print("Attn Score:", attn_score.shape)  #* Tuning Layer, B, Head, Q-tkn (1+2+196), Key+ B-Prompt (1[prefix]+1[cls]+2[vl]+196)
#     #todo lamda / image 각각 Scalar로 계산해서 저장
#     #todo Query: CLS, V-L, Image 
#     #todo Key: Basic, CLS, V-L, Image 
    
#     #todo 1. With VL
#     lamda_h_q = attn_score[:,:,:,3:,:4] #* Tuning Layer, B, Head, input-query, 4 (prompt)
#     lamda_h_q =lamda_h_q.mean(dim=2)     #* Tuning Layer, B, input_query,4
#     lamda_h_q =lamda_h_q.mean(dim=1)    #* Tuning Layer, input-query, 4
#     lamda_h_q =lamda_h_q.mean(dim=1)    #* Tuning Layer, 4
    
#     lamda_p_vl = attn_score[:,:,:,1:3,:4] #* Tuning Layer, B, Head, VL-query, 4 (prompt)
#     lamda_p_vl =lamda_p_vl.mean(dim=2)    #* Tuning Layer, B, VL-query,4
#     lamda_p_vl =lamda_p_vl.mean(dim=1)   #* Tuning Layer, input-query, 4
#     lamda_p_vl =lamda_p_vl.mean(dim=1)    #* Tuning Layer, 4
    
#     if not os.path.exists("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_h_q.csv"):
#         print("Create CSV File..")
#         f1 = open("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_h_q.csv",'w',newline='')
#         wr1 = csv.writer(f1)
#         wr1.writerow(['PKT','Layer 1-prefix', 'Layer 1-cls', 'Layer 1-vision', 'Layer 1-Language',  'Layer 2-prefix', 'Layer 2-cls', 'Layer 2-vision', 'Layer 2-Language'])
#         wr1.writerow(['   ',lamda_h_q[0,0].item(),lamda_h_q[0,1].item(),lamda_h_q[0,2].item(),lamda_h_q[0,3].item(), lamda_h_q[1,0].item(),lamda_h_q[1,1].item(),lamda_h_q[1,2].item(),lamda_h_q[1,3].item()])
        
#         f2 = open("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_p_vl.csv",'w',newline='')
#         wr2 = csv.writer(f2)
#         wr2.writerow(['PKT','Layer 1-prefix', 'Layer 1-cls', 'Layer 1-vision', 'Layer 1-Language',  'Layer 2-prefix', 'Layer 2-cls', 'Layer 2-vision', 'Layer 2-Language'])
#         wr2.writerow(['   ',lamda_p_vl[0,0].item(),lamda_p_vl[0,1].item(),lamda_p_vl[0,2].item(),lamda_p_vl[0,3].item(), lamda_p_vl[1,0].item(),lamda_p_vl[1,1].item(),lamda_p_vl[1,2].item(),lamda_p_vl[1,3].item()])
        
#     else:
#         print("Continue to write CSV File..")
#         f1 = open("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_h_q.csv",'a', newline='')
#         wr1 = csv.writer(f1)
#         wr1.writerow(['   ',lamda_h_q[0,0].item(),lamda_h_q[0,1].item(),lamda_h_q[0,2].item(),lamda_h_q[0,3].item(), lamda_h_q[1,0].item(),lamda_h_q[1,1].item(),lamda_h_q[1,2].item(),lamda_h_q[1,3].item()])
        
#         f2 = open("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_p_vl.csv",'a', newline='')
#         wr2 = csv.writer(f2)
#         wr2.writerow(['   ',lamda_p_vl[0,0].item(),lamda_p_vl[0,1].item(),lamda_p_vl[0,2].item(),lamda_p_vl[0,3].item(), lamda_p_vl[1,0].item(),lamda_p_vl[1,1].item(),lamda_p_vl[1,2].item(),lamda_p_vl[1,3].item()])
        
#     f1.close()
#     f2.close()
#     for handle in rollout_model.handles:
#         handle.remove()
#     del rollout_model
            