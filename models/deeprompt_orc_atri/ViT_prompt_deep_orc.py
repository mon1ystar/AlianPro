from typing_extensions import Self
import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_cifar import resnet18_cifar
from utils import identify_importance
import numpy as np
import copy
from utils import *
# from .helper import *
import timm
# from timm.models import vit_base_patch16_224_in21k
from models.vision_transformer import VisionTransformer
#todo PKT for domain specific knowledge learning..
#todo PKT with B-Prompt ==> Prefix Tuning 
#todo Need Something to focus on domain specific knowledge learning 
#todo finc inciteness from the Novel Category Discovery 
import open_clip as clip
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.init as init


class PseudoTargetClassifier(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        
        self.args = args
        self.num_features = num_features        # Input dimension for all classifiers

        # Classifier for the base classes
        self.base_fc = nn.Linear(self.num_features, self.args.base_class, bias=False)       # Note the entire number of classes are already added

        # Set of all classifiers
        self.classifiers = nn.Sequential(self.base_fc)

        # Register buffer for the pseudo targets. Assume the total number of classes
        self.num_classes = self.args.num_classes
        self.n_inc_classes = self.args.num_classes - self.args.base_class

        # Number of generated pseudo targets
        
        self.reserve_vector_count = self.num_classes
        

        # Storing the generated pseudo targets (reserved vectors)等待分配的目标
        self.register_buffer("rv", torch.randn(self.reserve_vector_count, self.num_features))

        self.temperature = 1.0

    def compute_angles(self, vectors):
        '''
        avg_angle-所有向量之间的平均夹角
        avg_angle_close-每个向量与其最近邻的平均夹角
        
        '''
        proto = vectors.cpu().numpy()
        dot = np.matmul(proto, proto.T)
        dot = dot.clip(min=0, max=1)
        theta = np.arccos(dot)
        np.fill_diagonal(theta, np.nan)
        theta = theta[~np.isnan(theta)].reshape(theta.shape[0], theta.shape[1] - 1)
        
        avg_angle_close = theta.min(axis = 1).mean()
        avg_angle = theta.mean()

        return np.rad2deg(avg_angle), np.rad2deg(avg_angle_close)

    def get_assignment(self, cost):
        """Tak array with cosine scores and return the output col ind """
        _, col_ind = linear_sum_assignment(cost, maximize = True)
        return col_ind

    def get_classifier_weights(self, uptil = -1):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            if uptil >= 0 and uptil < i + 1:
                break
            output.append(cls.weight.data)
        return torch.cat(output, axis = 0)

    def assign_base_classifier(self,):
        
        col_ind = np.arange(self.args.base_class)
        new_fc_tensor = self.rv[col_ind]

        # Create fixed linear layer
        self.classifiers[0].weight.data = new_fc_tensor#已有向量中选择相应的向量作为新的线性层的权重
        self.classifiers[0].weight.requires_grad_(False)

        # Remove from the final rv避免重复分配
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]
        
    def assign_all_classifier(self,):
        
        col_ind = np.arange(self.num_classes)
        new_fc_tensor = self.rv[col_ind]

        # Create fixed linear layer
        self.classifiers[0].weight.data = new_fc_tensor#已有向量中选择相应的向量作为新的线性层的权重
        self.classifiers[0].weight.requires_grad_(False)

        # Remove from the final rv避免重复分配
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]

    def assign_novel_classifier(self):  
        # Normalise incoming prototypes
        col_ind = np.arange(self.args.way)
        new_fc_tensor = self.rv[col_ind]

        # Creating and appending a new classifier from the given reserved vectors
        new_fc = nn.Linear(new_fc_tensor.shape[1], new_fc_tensor.shape[0], bias=False).cuda()
        new_fc.weight.data.copy_(new_fc_tensor)
        #self.classifiers = nn.Sequential(self.classifiers,new_fc.cuda())
        #self.classifiers.append(new_fc.cuda())
        new_classifiers = nn.Sequential()

        # 复制旧的层到新的 Sequential 容器中
        for name, module in self.classifiers.named_children():
            new_classifiers.add_module(name, module)
            
        new_fc.weight.requires_grad_(False)

        # 添加新的层到新的 Sequential 容器中
        new_classifiers.add_module(f'fc{len(new_classifiers)}', new_fc.cuda())

        # 更新模型中的 classifiers
        self.classifiers = new_classifiers

        # Maintaining the pseudo targets. Self.rv contains only the unassigned vectors
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]


    def find_reseverve_vectors_all(self):
        """
        生成正交伪目标
        """
        points = torch.randn(self.reserve_vector_count, self.num_features).cuda()
        points = normalize(points)
        points = torch.nn.Parameter(points)

        opt = torch.optim.SGD([points], lr=1)
        
        best_angle = 0
        tqdm_gen = tqdm(range(1000))

        for _ in tqdm_gen:
            # Compute the cosine similarity.
            sim = F.cosine_similarity(points[None,:,:], points[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / points.shape[0]
            
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = self.compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle

            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data

    def forward(self, x):
        return self.get_logits(x)
        
    def get_logits(self, encoding,):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            out = F.linear(F.normalize(encoding, p=2, dim=-1), F.normalize(cls.weight, p=2, dim=-1))
            out = out / self.temperature
            output.append(out)
        output = torch.cat(output, axis = 1)
        
        return output
    
    
class ViT_DEEP_ORC_A(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            self.num_features = 768
            self.prompt_length = 10
            self.vcls_length = 2
        if self.args.dataset in ['mini_imagenet']:
            self.num_features = 768
            self.prompt_length = 1
            self.vcls_length = 2
        if self.args.dataset == 'cub200' or self.args.dataset == 'air':
            self.num_features = 768
            self.prompt_length = 5
            self.vcls_length = 2
        if self.args.dataset in ['classroom']:
            self.num_features = 768
            self.prompt_length = 10
            self.vcls_length = 2

        self.projc_features = 768
        
        
        if args.scratch:
            self.encoder = timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=args.num_classes,
                                drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
        else:
            self.encoder = timm.create_model("vit_base_patch16_224",pretrained=True,num_classes=args.num_classes,
                                drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
        
        
        #* Prompt
        #todo Head 토큰 없애고 Vision으로 Pool
         
        
        self.prompt = nn.Parameter(torch.randn(self.prompt_length,self.num_features))   #* VL
        self.deep_prompt_embeddings = nn.Parameter(torch.randn(
                    11, self.prompt_length, self.num_features))          #deeprompt
        self.vcls = nn.Parameter(torch.randn(self.vcls_length,self.num_features))
        
        self.scale_params = nn.Parameter(torch.ones(args.num_classes))
                
        
        nn.init.uniform_(self.prompt, -1, 1)
        nn.init.uniform_(self.deep_prompt_embeddings, -1, 1)
        nn.init.uniform_(self.vcls, -1, 1)
        
        #*------------------------------------------------------
        self.num_tokens = 197
        
        self.projector = nn.Sequential(
                nn.Linear(self.num_features, self.projc_features),
            )
        
        
        self.fc = PseudoTargetClassifier(self.args, self.projc_features)
        
        self.hc=[]
        
        self.vmask = nn.Parameter(torch.randn(196,self.num_features))
        
        self.seen_classes = args.base_class
    #todo =======================================================================================
    
    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.mask = torch.zeros(self.args.num_classes,device='cuda')
        self.mask[:self.seen_classes]=-torch.inf
        self.seen_classes += len(new_classes)
    
    
    
    def encode(self, x):
        x = self.encoder.forward_features(x)[:,0]
        return x
    
    def ft_encode(self, x):
        
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        #*==============================================================
        #! VL-Prompt tuning
        # prompting_tkn = self.prompt

        # pos_tkn = prompting_tkn + self.encoder.pos_embed[:,0].expand(self.prompt_length, -1)
        # pos_tkn = pos_tkn.expand(x.shape[0],-1,-1)
        # x = torch.cat([x[:,0].unsqueeze(1), pos_tkn, x[:,1:]],dim=1)#
        # #!=============================================================
        # #* prefix for B-Prompt (Original)
        # x = self.deeprompt_encode(x)
        x = self.encoder.blocks(x)
        
        cls_embed = x[:,0]
        meiyong = x[:,1:3]
        
        return cls_embed,meiyong
    
    def prompt_encode(self, x):
        
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        #*==============================================================
        #! VL-Prompt tuning
        prompting_tkn = self.prompt

        pos_tkn = prompting_tkn + self.encoder.pos_embed[:,0].expand(self.prompt_length, -1)
        pos_tkn = pos_tkn.expand(x.shape[0],-1,-1)
        x = torch.cat([x[:,0].unsqueeze(1), pos_tkn, x[:,1:]],dim=1)#
        
        x = self.deeprompt_encode(x)
        #x = self.encoder.blocks(x)
        
        cls_embed = x[:,0]
        noused = x[:,1:3]
        
        return cls_embed,noused
    
    def deeprompt_encode(self, x):
        
        for block_idx, block in enumerate(self.encoder.blocks):
            if block_idx == 0:
                latent_feat = block(x)
            else:
                prompt_drop = self.deep_prompt_embeddings[block_idx-1]
                current_prompt = self.encoder.pos_drop(prompt_drop.expand(x.shape[0],-1,-1))
                latent_feat = torch.cat((
                        latent_feat[:, :1, :],
                        current_prompt,
                        latent_feat[:, (1+self.prompt_length):, :]
                    ), dim=1)
                latent_feat = block(latent_feat)
        feat = self.encoder.norm(latent_feat)
        return feat
    
    
    def vcls_encode(self, x):
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        vcls = self.vcls.expand(x.shape[0], -1, -1)
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = torch.cat([x[:,:1],vcls,x[:,1:]],dim=1)
        #*==============================================================
        #! VL-Prompt tuning
        prompting_tkn = self.prompt
        
        pos_tkn = prompting_tkn + self.encoder.pos_embed[:,0].expand(self.prompt_length, -1)
        pos_tkn = pos_tkn.expand(x.shape[0],-1,-1)
        x = torch.cat([x[:,:1+self.vcls_length], pos_tkn, x[:,1+self.vcls_length:]],dim=1)#
        #!=============================================================
        #* prefix for B-Prompt (Original)
        for block_idx, block in enumerate(self.encoder.blocks):
            if block_idx == 0:
                latent_feat = block(x)
                
            else:
                prompt_drop = self.deep_prompt_embeddings[block_idx-1]
                # noise = torch.randn_like(prompt_drop) * 0.1
                # prompt_drop = prompt_drop + noise
                current_prompt = self.encoder.pos_drop(prompt_drop.expand(x.shape[0],-1,-1))
                latent_feat = torch.cat((
                        latent_feat[:, :1+self.vcls_length, :],
                        current_prompt,
                        latent_feat[:, (1+self.vcls_length+self.prompt_length):, :]
                    ), dim=1)
                latent_feat = block(latent_feat)
                
                
        feat = self.encoder.norm(latent_feat)
        
        h_cls =feat[:,0]
        vcls = feat[:,1:1+self.vcls_length]
        
        
        return h_cls,vcls
    
    def adaptive_encode(self, x):
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        vcls = self.vcls.expand(x.shape[0], -1, -1)
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = torch.cat([x[:,:1],vcls,x[:,1:]],dim=1)
        #*==============================================================
        #! VL-Prompt tuning
        prompting_tkn = self.prompt
        pos_tkn = prompting_tkn + self.encoder.pos_embed[:,0].expand(self.prompt_length, -1)
        pos_tkn = pos_tkn.expand(x.shape[0],-1,-1)
        x = torch.cat([x[:,:1+self.vcls_length], pos_tkn, x[:,1+self.vcls_length:]],dim=1)#
        #!=============================================================
        #* prefix for B-Prompt (Original)
        for block_idx, block in enumerate(self.encoder.blocks):
            if block_idx == 0:
                latent_feat = block(x)
                
            elif block_idx in [1,2,3,4,5]:
                prompt_drop = self.deep_prompt_embeddings[block_idx-1]
                current_prompt = self.encoder.pos_drop(prompt_drop.expand(x.shape[0],-1,-1))
                latent_feat = torch.cat((
                        latent_feat[:, :1+self.vcls_length, :],
                        current_prompt,
                        latent_feat[:, (1+self.vcls_length+self.prompt_length):, :]
                    ), dim=1)
                latent_feat = block(latent_feat)
                
            elif block_idx == 6:
                prompt_drop = self.deep_prompt_embeddings[block_idx-1]
                
                current_prompt = self.encoder.pos_drop(prompt_drop.expand(x.shape[0],-1,-1))
                latent_feat = torch.cat((
                        latent_feat[:, :1+self.vcls_length, :],
                        current_prompt,
                        latent_feat[:, -196:, :]
                    ), dim=1)
                latent_feat = block(latent_feat)

            else:
                prompt_drop =  self.deep_prompt_embeddings[block_idx-1] 
                
                current_prompt = self.encoder.pos_drop(prompt_drop.expand(x.shape[0],-1,-1))
                latent_feat = torch.cat((
                        latent_feat[:, :1+self.vcls_length, :],
                        current_prompt,
                        latent_feat[:, -196:, :]
                    ), dim=1)
                latent_feat = block(latent_feat)
                
                
        feat = self.encoder.norm(latent_feat)
        
        h_cls =feat[:,0]
        
        vcls = feat[:,1:1+self.vcls_length]
        
        
        return h_cls,vcls
    
        

    def forward(self, x):
        
            
        
        if self.args.dataset not in ['classroom']:
            
            x = self.encoder.patch_embed(x)
        
        # hidden_embedding,vcls = self.vcls_encode(x,)
        # hidden_embedding,vcls = self.adaptive_encode(x,)
        hidden_embedding,vcls = self.prompt_encode(x,)
        # hidden_embedding,vcls = self.ft_encode(x,)
        # hidden_embedding = self.encoder.forward_features(x)[:,0]
        
        
        cls_embedding = self.projector(hidden_embedding)
        v_embedding = self.projector(vcls)
        
        
        
        vv_embedding = v_embedding.reshape(-1, 768)
        
        
        logits = self.fc.get_logits(cls_embedding)
        # logits = logits * self.scale_params
        
        vlogits = self.fc.get_logits(vv_embedding)
        # vlogits = vlogits * self.scale_params
        
        
        
        
        
        
        return logits, cls_embedding,hidden_embedding,vlogits,v_embedding,
        
        
            
           
        
            

   
    def train_inc(self, dataloader, epochs, session, class_list,args):
        print("[Session: {}]".format(session))
        # self.fc.assign_novel_classifier()
        self.update_fc_avg(dataloader, class_list, session)#计算hc和新类原型
        # self.reinitialize_projection_layer()
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr_new)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        for epoch in range(epochs):
            self.train()
            
            hidden_embedding = torch.stack(self.hc, dim=0).cuda()
            cls_embedding = self.projector(hidden_embedding)
            logits = self.fc.get_logits(cls_embedding)
            joint_label = torch.arange(args.base_class+args.way*(session)).cuda()
            
            loss_angle =  self.myloss(logits, joint_label)

            acc = count_acc(logits, joint_label)
            optim.zero_grad()
            loss_angle.backward()
            
            optim.step()
            scheduler.step()
            pred = torch.argmax(logits, dim=1)
            acc = (pred == joint_label).sum().item()/joint_label.shape[0]*100.
            
            print(f"[{epoch}/{epochs}] Loss_CE:{loss_angle.item():.4f} ACC: {acc}")
            
    def train_inc_5shot(self, dataloader, epochs, session, class_list,args):
        print("[Session: {}]".format(session))
        # self.fc.assign_novel_classifier()
        
        #self.update_fc_avg(dataloader, class_list, session)#fc层换成原型
    
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr_new)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        for batch in dataloader:
            data_imgs, label = [_.cuda() for _ in batch]
            logits, cls_new,hidden_new,_,_=self.forward(data_imgs)
        
        for epoch in range(epochs):
            self.train()
            base_proto = torch.stack(self.hc, dim=0).cuda()             # [num_base_class, d]
            base_proto = base_proto.repeat_interleave(5, dim=0)         # => [num_base_class * 5, d]
           
            hidden_embedding = torch.cat((base_proto, hidden_new), dim=0)  # 拼上 novel 类 5 个样本
            
            cls_embedding = self.projector(hidden_embedding)
            logits = self.fc.get_logits(cls_embedding)

            num_base = args.base_class + args.way * (session - 1)
            old_label = torch.arange(num_base).repeat_interleave(5).cuda()  # [num_base * 5]
            joint_label = torch.cat((old_label, label), dim=0).cuda()
            
           
            loss_angle =  self.myloss(logits, joint_label)

            acc = count_acc(logits, joint_label)
            optim.zero_grad()
            loss_angle.backward(retain_graph=True)
            
            optim.step()
            scheduler.step()
            pred = torch.argmax(logits, dim=1)
            acc = (pred == joint_label).sum().item()/joint_label.shape[0]*100.
            
            print(f"[{epoch}/{epochs}] Loss_CE:{loss_angle.item():.4f} ACC: {acc}")
            
        self.compute_hc_avg(dataloader, class_list, session)

    def train_inc_5shot_samples(self, dataloader, epochs, session, class_list,args):
        print("[Session: {}]".format(session))
    
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr_new)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        # Step 1: 准备新类特征
        for batch in dataloader:
            data_imgs, label = [_.cuda() for _ in batch]
            logits, cls_new, hidden_new, _, _ = self.forward(data_imgs)

        # Step 2: 扩充旧类特征，每类复制 5 次
        hidden_old = torch.stack(self.hc, dim=0).cuda()  # [num_old_class, d]
        hidden_old_expanded = hidden_old.unsqueeze(1).repeat(1, 5, 1)  # [num_old_class, 5, d]
        hidden_old_expanded = hidden_old_expanded.view(-1, hidden_old.shape[-1])  # [num_old_class*5, d]

        # Step 3: 拼接新旧特征
        hidden_all = torch.cat((hidden_old_expanded, hidden_new), dim=0)  # shape: [num_total, d]
        cls_embedding = self.projector(hidden_all)

        # Step 4: 构建标签
        old_label = torch.arange(args.base_class + args.way * (session - 1)).cuda()  # [num_old_class]
        old_label = old_label.unsqueeze(1).repeat(1, 5).view(-1)  # 每类复制5次 → [num_old_class * 5]
        joint_label = torch.cat((old_label, label), dim=0)  # shape: [num_total]

        # Step 5: 分 batch 训练
        batch_size = 128
        num_samples = cls_embedding.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size  # ceil

        for epoch in range(epochs):
            self.train()
            
            indices = torch.randperm(num_samples).cuda()  # 打乱
            cls_embedding_shuffled = cls_embedding[indices]
            joint_label_shuffled = joint_label[indices]

            epoch_loss = 0.
            epoch_acc = 0.

            for b in range(num_batches):
                start = b * batch_size
                end = min((b + 1) * batch_size, num_samples)

                logits_batch = self.fc.get_logits(cls_embedding_shuffled[start:end])
                label_batch = joint_label_shuffled[start:end]

                loss = self.myloss(logits_batch, label_batch)
                acc = count_acc(logits_batch, label_batch)

                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()

                epoch_loss += loss.item()
                epoch_acc += acc 

            scheduler.step()
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_samples * 100.

            print(f"[{epoch}/{epochs}] Loss_CE: {avg_loss:.4f}  ACC: {avg_acc:.2f}%")

            
        self.compute_hc_avg(dataloader, class_list, session)


    def update_fc_avg(self,dataloader,class_list,session):
        self.eval()
        
        
        with torch.no_grad():
            for batch in dataloader:
                data_imgs, label = [_.cuda() for _ in batch]
                # logits, cls_embedding,hidden_embedding,vlogits,vcls=self.forward(data_imgs,base=True)
                logits, cls_embedding,hidden_embedding,vlogits,vcls=self.forward(data_imgs)
                cls_embed=cls_embedding.detach()
                hidden_embed=hidden_embedding.detach().cpu()
                # cls_embed=self.prompt_encode(data_imgs,B_tuning=True).detach()
            max_cos_list = []
            for class_index in class_list:
                data_index=(label==class_index).nonzero().squeeze(-1)
                
                embedding = cls_embed[data_index]
                
                
                hc = hidden_embed[data_index]
                # hc = hc[clean_indices].mean(0)
                hc = hc.mean(0)
                self.hc.append(hc)
                
                proto=embedding.mean(0)
                max_cos , base_vector = self.get_most_similar_weight(proto,self.fc.classifiers[0].weight.data)
                max_cos_list.append(max_cos)
                # classifier_weights = self.update_classifier_weights(proto,self.fc.classifiers[0].weight.data)
                # self.fc.classifiers[0].weight.data[class_index] =   proto/torch.norm(proto)
            
            average_max_cos = sum(max_cos_list) / len(max_cos_list)
            print(average_max_cos)
        self.train()
    
    def remove_noise_samples(self,embedding, threshold=0.4):
        """
        通过余弦距离剔除噪声样本，返回没有噪声的样本。
        
        参数：
        - embedding: 形状为 [num_samples, feature_dim] 的 tensor表示样本的特征。
        - threshold: 余弦相似度的阈值，低于该值的样本被认为是噪声。
        
        返回：
        - clean_embedding: 形状为 [num_samples', feature_dim] 的 tensor表示去除噪声后的特征。
        """
        # 计算余弦相似度矩阵
        cosine_sim = F.cosine_similarity(embedding.unsqueeze(0), embedding.unsqueeze(1), dim=2)  # [num_samples, num_samples]
        
        # 去除对角线元素（每个样本与自己的相似度）
        mask = torch.eye(cosine_sim.size(0), device=cosine_sim.device)  # 创建一个对角线为1的mask
        cosine_sim = cosine_sim.masked_fill(mask.bool(), 0)  # 将对角线元素置为0
        
        # 计算每个样本与其他所有样本的平均相似度
        avg_cosine_sim = cosine_sim.mean(dim=1)  # 计算每个样本与其他所有样本的平均余弦相似度
        
        # 选择那些与其他样本相似度较高的样本（平均相似度大于阈值）
        clean_indices = avg_cosine_sim > threshold  # 保留平均余弦相似度大于阈值的样本
        
        # 返回去除噪声后的样本
        clean_embedding = embedding[clean_indices]
        
        return clean_indices
    
    def compute_hc_avg(self,dataloader,class_list,session):
        self.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for batch in dataloader:
                data_imgs, label = [_.cuda() for _ in batch]
                logits, cls_embedding,hidden_embedding,_,_=self.forward(data_imgs)
                embedding_list.append(hidden_embedding.detach().cpu())
                label_list.append(label.cpu())
                # cls_embed=self.prompt_encode(data_imgs,B_tuning=True).detach()
                
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
               
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero()
                embedding_this = embedding_list[data_index.squeeze(-1)]
                #验证高斯分布
                # self.pca_anlysis(embedding_this,class_index)
                embedding_this = embedding_this.mean(0)
                self.hc.append(embedding_this)
        
        self.train()
        
    def pca_anlysis(self,embedding,index):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embedding)

        plt.figure(figsize=(6, 4))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=50)
        plt.title('PCA Projection of Class Features')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.savefig('pca_anly/pca_{}.png'.format(index))

    def init_base_fc(self,query,class_list):
        self.eval()
        with torch.no_grad():
            for class_index in class_list:
                self.fc.weight.data[class_index] = query[class_index]
    
    def get_logits(self,x, fc):
        return fc(x)
    
    def myloss(self, logits, labels):
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
    
    def compute_averagecos(self,prompt):
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
        
        return non_diag_mean
    
    def batch_token_level_mixup(self,tokens, alpha=0.2):
        """
        Perform token-level mixup within a batch by shuffling the batch.
        
        Args:
            tokens: Tensor of shape [batch_size, num_tokens, embed_dim].
            alpha: Mixup hyperparameter controlling the Beta distribution.
        
        Returns:
            mixed_tokens: Mixed tokens of shape [batch_size, num_tokens, embed_dim].
            mixup_ratios: Mixing ratios (lambda values) for each sample.
        """
        batch_size = tokens.size(0)
        
        # Shuffle the batch indices
        shuffle_indices = torch.randperm(batch_size).to(tokens.device)
        shuffled_tokens = tokens[shuffle_indices]  # Shuffle the tokens
        
        # Sample lambda for each example
        lambda_ = np.random.beta(alpha, alpha, size=batch_size)
        lambda_tensor = torch.tensor(lambda_, dtype=torch.float32).to(tokens.device)
        lambda_tensor = lambda_tensor.view(batch_size, 1, 1)  # Reshape for broadcasting
        
        # Perform Mixup
        mixed_tokens = lambda_tensor * tokens + (1 - lambda_tensor) * shuffled_tokens
        
        return mixed_tokens
    
    def rand_patch_bbox(self,num_patches, lam):
        """
        Generate a random bounding box in patch space for CutMix.
        
        Args:
            num_patches: Number of patches along one dimension (H or W).
            lam: Lambda value determining the mix ratio.
            
        Returns:
            bbx1, bby1, bbx2, bby2: Coordinates of the patch bounding box.
        """
        cut_ratio = np.sqrt(1.0 - lam)
        cut_size = int(num_patches * cut_ratio)
        
        cx = np.random.randint(num_patches)  # Center of the bounding box
        cy = np.random.randint(num_patches)
        
        bbx1 = np.clip(cx - cut_size // 2, 0, num_patches)
        bby1 = np.clip(cy - cut_size // 2, 0, num_patches)
        bbx2 = np.clip(cx + cut_size // 2, 0, num_patches)
        bby2 = np.clip(cy + cut_size // 2, 0, num_patches)
        
        return bbx1, bby1, bbx2, bby2

    def patch_level_cutmix_no_labels(self,patches, alpha=1.0):
        """
        Apply CutMix at the patch level for Vision Transformer input tokens without returning labels.
        
        Args:
            patches: Tensor of shape [B, N, D], where B is batch size, N is the number of patches, and D is embedding dim.
            alpha: Hyperparameter for the Beta distribution.
        
        Returns:
            mixed_patches: Mixed patch tokens of shape [B, N, D].
        """
        batch_size, num_patches, embed_dim = patches.size()
        lam = np.random.beta(alpha, alpha)
        shuffle_indices = torch.randperm(batch_size).to(patches.device)
        shuffled_patches = patches[shuffle_indices]
        
        # Compute patch dimensions (assuming a square grid of patches)
        num_patch_dim = int(np.sqrt(num_patches))
        assert num_patch_dim ** 2 == num_patches, "Number of patches must be a perfect square."
        
        # Generate a random patch-level bounding box
        bbx1, bby1, bbx2, bby2 = self.rand_patch_bbox(num_patch_dim, lam)
        
        # Convert bounding box to 1D patch indices
        patch_mask = torch.ones((num_patch_dim, num_patch_dim), device=patches.device)
        patch_mask[bbx1:bbx2, bby1:bby2] = 0
        patch_mask = patch_mask.flatten()  # Flatten to match patch token indices
        
        # Apply the patch mask for CutMix
        mixed_patches = patches.clone()
        for i in range(batch_size):
            mixed_patches[i, patch_mask == 0] = shuffled_patches[i, patch_mask == 0]
        
        return mixed_patches
    
    def forward_with_intermediate_inputs(self, x):

        block_inputs = []  # 用于存储每个 block 的输入
        x = self.encoder.patch_embed(x)  # 计算初始 embeddings
        for block in self.encoder.blocks:
            block_inputs.append(x)  # 在每个 block 前存储输入
            x = block(x)  # 通过当前 block
        return x, block_inputs

    def compute_token_prototypes(self, dataloader):
        # 初始化 prototypes, 假设 seq_len = 196，feature_dim = 768
        prototypes = torch.zeros(12, 196, 768).cuda()  # 将初始张量直接放到 GPU 上
        count = 0

        for batch in dataloader:
            data, train_label = [_.cuda() for _ in batch]  # 数据和标签放到 GPU 上

            with torch.no_grad():
                _, block_inputs = self.forward_with_intermediate_inputs(data)

            # 累加特征
            for i, inputs in enumerate(block_inputs):
                mean_features = inputs.mean(dim=0)  # 对 batch 维度求平均 -> (seq_len, feature_dim)
                prototypes[i] += mean_features  # 排除 class token（第 0 个）

            count += 1

        # 取平均
        prototypes /= count
        
        # 平均池化，将每个 block 的 prototypes 从 [196, 768] 变为 [5, 768]
        pooled_prototypes = torch.zeros(12, 5, 768).cuda()  # 初始化池化后的张量
        avg_pool = nn.AvgPool1d(kernel_size=39, stride=39, ceil_mode=False)  # 平均池化

        for i in range(12):  # 遍历每个 block
            # 将 [196, 768] 转为 [768, 196]，以便使用 1D 池化
            block_prototype = prototypes[i].permute(1, 0)  # 变为 [768, 196]
            pooled = avg_pool(block_prototype.unsqueeze(0))  # 添加 batch 维度，应用池化
            pooled_prototypes[i] = pooled.squeeze(0).permute(1, 0)  # 恢复形状为 [5, 768]
            
        with torch.no_grad():  # 禁止梯度追踪，直接更新值
            self.prompt.data.copy_(pooled_prototypes[0])  # 确保形状匹配    
            self.deep_prompt_embeddings.data.copy_(pooled_prototypes[1:])
            
    def promptdropout(self, x, dropout_prob=0.5):
        # 随机生成一个掩码，掩码的形状与提示的数量相同
        mask = torch.rand(self.prompt_length).bernoulli(p=dropout_prob).bool().cuda()
        # 使用掩码失活部分提示
        x = x * mask.unsqueeze(-1).float()  # 失活的提示变为 0
        return x
    
    def get_most_similar_weight(self,proto, fc_weights):
        """
        计算 proto 和 fc_weights 之间的余弦相似度，并返回最大相似度对应的权重向量。
        
        参数：
        proto: [in_features] 的向量，表示提示向量
        fc_weights: [out_features, in_features] 的矩阵，表示分类器权重
        
        返回：
        most_similar_weight: 与 proto 余弦相似度最大的一行权重向量，形状为 [in_features]
        """
        # 扩展 proto 维度使其形状与权重矩阵匹配
        proto = proto.unsqueeze(0)  # 将 proto 从 [in_features] 变为 [1, in_features]
        
        # 计算每个权重向量与 proto 之间的余弦相似度
        cosine_similarities = F.cosine_similarity(fc_weights, proto, dim=1)
        
        # 找到余弦相似度最大的那个类别的索引
        max_cos, max_idx = torch.max(cosine_similarities, dim=0)
        
        # 返回与 proto 最相似的权重向量
        most_similar_weight = fc_weights[max_idx]
        
        return max_cos , most_similar_weight
    
    def update_classifier_weights(self, proto, classifier_weights):
        """
        计算 proto 和 classifier_weights 之间的余弦相似度，选择前五个相似度最大的权重，使用 softmax 作为系数加权并返回新的分类器权重向量。

        参数：
        proto: 形状为 [in_features] 的提示向量
        classifier_weights: 形状为 [out_features, in_features] 的分类器权重矩阵

        返回：
        updated_weight: 加权后的新的分类器权重向量（形状为 [in_features]
        """
        # 计算 proto 和每个权重向量的余弦相似度
        cosine_similarities = F.cosine_similarity(classifier_weights, proto.unsqueeze(0), dim=1)

        # 获取前五个相似度最大的索引
        top_k_values, top_k_indices = torch.topk(cosine_similarities, k=5)

        # 选择前五个权重向量
        top_k_weights = classifier_weights[top_k_indices]

        # 对前五个相似度值应用 softmax
        softmax_weights = F.softmax(top_k_values, dim=0)

        # 每个权重向量乘以对应的 softmax 权重并求和
        weighted_weights = top_k_weights.T @ softmax_weights  # 计算加权后的向量

        return weighted_weights
    
    def compute_prompt_loss(self, prompt,lambda_reg=0.001):
        """
        计算 prompt 的 L2 正则化损失并返回。
        
        参数：
        - lambda_reg: 正则化损失的权重系数，控制正则化的强度
        
        返回：
        - prompt_loss: 正则化损失
        """
        # 获取 prompt 的形状，假设为 [12, 5, 768]（12层，每层5个提示，768维）
        
        
        # 计算每个提示的 L2 范数（即每个提示的欧氏距离）
        # prompt.norm(p=2, dim=-1) 计算每个提示的 L2 范数，结果形状为 [12, 5]
        prompt_norms = prompt.norm(p=2, dim=-1)
        
        # 计算总的正则化损失，sum(prompt_norms) 会得到每层所有提示的 L2 范数之和
        # 然后对每层的正则化损失加权
        prompt_loss = prompt_norms.sum()  # [12, 5] -> 每层所有提示的 L2 范数之和
        prompt_loss = prompt_loss * lambda_reg  # 使用 lambda_reg 进行加权
        
        return prompt_loss
    
    def dropout_prompt_token(self,prompt_embeds, drop_prob=0.5):
        """
        prompt_embeds: shape [num_prompts, dim] = [5, 768]
        drop_prob: probability of dropping each prompt token
        """
        if drop_prob <= 0.0:
            return prompt_embeds
        if drop_prob >= 1.0:
            return torch.zeros_like(prompt_embeds)
        
        # shape: [5, 1], 1 表示保留，0 表示丢弃
        keep_mask = (torch.rand(prompt_embeds.size(0), 1, device=prompt_embeds.device) > drop_prob).float()
        
        return prompt_embeds * keep_mask  # shape [5, 768]

    # 重新初始化投影层的参数
    def reinitialize_projection_layer(self):
        # 遍历模型中的投影层，重新初始化权重和偏置
        for m in self.projector:
            if isinstance(m, nn.Linear):
                # 例如：使用 Xavier 初始化
                init.xavier_uniform_(m.weight)  # 使用均匀分布进行 Xavier 初始化
                if m.bias is not None:
                    init.zeros_(m.bias)  # 将偏置初始化为零


