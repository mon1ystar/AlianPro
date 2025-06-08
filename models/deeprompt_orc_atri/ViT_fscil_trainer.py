



from models.deeprompt_orc_atri.ViT_prompt_deep_orc import ViT_DEEP_ORC_A
from .base import Trainer
import os.path as osp
import torch.nn as nn
from .parallel import DataParallelModel, DataParallelCriterion
import copy
from copy import deepcopy
import pandas as pd
from os.path import exists as is_exists

from .helper import *
from utils import *
from dataloader.data_utils import *
from models.switch_module import switch_module
from dataloader.data_manager import DataManager

class ViT_FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.set_log_path()

        self.args = set_up_datasets(self.args)
        self.model = ViT_DEEP_ORC_A(self.args, mode=self.args.base_mode)
        
        if self.args.LT:
            print("Tuning Layer!!")  
            for p in self.model.encoder.parameters():
                p.requires_grad=False
            
            num_layer = [l for l in range(args.taskblock)] 
            for idx, block in enumerate(self.model.encoder.blocks):
                if idx in num_layer:
                    for p in block.parameters():
                        p.requires_grad=True
        elif self.args.scratch:
            print("Scratch Model!!")
        elif self.args.ft:
            print("ft Model!!")
        elif self.args.lp:
            pass
        else:
            for p in self.model.encoder.parameters():
                p.requires_grad=False
            print("No Tuning Layer!!")
        
        
        
        self.val_model = ViT_DEEP_ORC_A(self.args, mode=self.args.base_mode)

        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.val_model = nn.DataParallel(self.val_model, list(range(self.args.num_gpu)))
        self.val_model = self.val_model.cuda()
        
        self.word_info = {}
        self.word_info["embed"] = None
        self.word_info["cur_embed"] = None
        self.word_info["label_text"] = np.array([])
        
        self.query_info={}
        self.query_info["proto"] = None
        
        self.loss_curve={}
        self.loss_curve['ACC'] = []
        self.loss_curve['CE_loss'] = []
        self.loss_curve['ED_loss']=[]
        self.loss_curve['ED_ce']=[]
        self.loss_curve['ED_kl']=[]
        self.loss_curve['ED_loss']=[]
        self.loss_curve['SKD_loss']=[]
        self.loss_curve['SKD_kd']=[]
        self.loss_curve['SKD_ce']=[]
        self.loss_curve['total_loss']=[]
        self.loss_curve['grad_list'] = []
        
        self.loss_curve['attn_score']=[]
        
        #* Bert_model for ViT-B
        self.tokenizer = BertTokenizer.from_pretrained('/amax/2020/qyl/PriViLege/bertmodel')
        self.Bert_model = BertModel.from_pretrained("/amax/2020/qyl/PriViLege/bertmodel")
        self.Bert_model = nn.DataParallel(self.Bert_model, list(range(self.args.num_gpu)))
        self.Bert_model = self.Bert_model.cuda()
        
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            
            

        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())
    
    
        print("#"*50)
        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        self.init_params = sum(param.numel() for param in self.model.parameters())
        print('total parameters:',self.init_params)
        print('trainable parameters:',trainable_params)
        print("#"*50)

    def get_optimizer_base(self):
        # ! Original
        # optimizer = torch.optim.Adam([
        #         {'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
        #         {'params': self.model.module.projector.parameters(), 'lr': 0.1}
        #     ])
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr_base,)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def get_standard_dataloader(self, session, data_manager):
        # import data_manager
        batchsize=128
        num_cls=10
        train_dataset = data_manager.get_dataset(
            np.arange(session*num_cls, (session+1)*num_cls),
            source="train",
            mode="train",
            # appendent=self._get_memory(),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batchsize, shuffle=True, num_workers=4
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, (session+1)*num_cls), source="test", mode="test"
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batchsize, shuffle=False, num_workers=4
        )
        return data_manager.idata.train_set, train_loader, test_loader


    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        columns = ['num_session', 'acc', 'base_acc', 'new_acc', 'base_acc_given_new', 'new_acc_given_base']
        acc_df = pd.DataFrame(columns=columns)
        print("[Start Session: {}] [Sessions: {}]".format(args.start_session, args.sessions))
        
        for session in range(args.start_session, args.sessions):
            
            train_set, trainloader, testloader = self.get_dataloader(session)
            print(f"Session: {session} Data Config")
            print(len(train_set.targets))
            if session > 0:
                print("Freeze parameters of the encoder.. ")
                if args.pret_clip:
                    for idx, block in enumerate(self.model.module.encoder.transformer.resblocks):
                        for p in block.parameters():
                            p.requires_grad=False
                    for p in self.model.module.encoder.transformer.key_comp.parameters():
                        p.requires_grad = False
                    self.model.module.encoder.expert_prompt.requires_grad=False
                elif args.ft:
                    for p in self.model.module.encoder.parameters():
                        p.requires_grad=False
                elif args.lp:
                    for p in self.model.module.parameters():
                        p.requires_grad=False
                    
                    for p in self.model.module.fc.parameters():
                        p.requires_grad=True
                else:
                    for p in self.model.module.encoder.parameters():
                        p.requires_grad=False
                    self.model.module.prompt.requires_grad=False
                    self.model.module.deep_prompt_embeddings.requires_grad=False
                    self.model.module.vcls.requires_grad=False
                    #* Pointwise_Compressor
                    
            #todo ===============================================
            if session == 0:  # load base class train img label
                
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                
                
               
                print("[Base Session Training]")
                print("#"*50)
                trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
                print('[Session {}] Trainable parameters: {}'.format(session,trainable_params))
                print("#"*50)
                # 一次全部生成 or 分任务生成
                print('生成正交分类器，分配基类')
                self.model.module.fc.find_reseverve_vectors_all()
                # self.model.module.fc.assign_base_classifier()
                self.model.module.fc.assign_all_classifier()
                print("#"*50)
                print('token初始化')
                # self.model.module.compute_token_prototypes(trainloader)
                print("#"*50)
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, self.word_info, self.query_info, np.unique(train_set.targets), args, self.loss_curve)
                    tsl, tsa, logs = test(self.model, testloader, epoch, args, session,self.word_info)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        # torch.save(dict(params=self.model.state_dict()), 'best_model.pth')
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                        self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                            '\nstill need around %.2f mins to finish this session' % (
                                    (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_last_epoch.pth')
                #torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.model.load_state_dict(self.best_model_dict)
                
                
                print("#"*50)
                print('计算基类hc')

                self.model.module.compute_hc_avg(trainloader, np.unique(train_set.targets), session)
                #self.best_model_dict = deepcopy(self.model.state_dict())
                #*=======================================================================================
                # if not args.not_data_init:
                #     self.model.load_state_dict(self.best_model_dict)
                #     if not args.pret_clip:
                #         self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                #         best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc_replace_head.pth')
                #         print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                #         # torch.save(dict(params=self.model.state_dict()), best_model_dir)
                #         self.best_model_dict = deepcopy(self.model.state_dict())

                #     self.model.module.mode = 'avg_cos'
                #     tsl, tsa, logs = test(self.model, testloader, 0, args, session,self.word_info)
                #     self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                #     print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
            
            else:  # incremental learning sessions
                print("Incremental session: [%d]" % session)
                print("#"*50)
                trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
                print('[Session {}] Trainable parameters: {}'.format(session,trainable_params))
                print("#"*50)
                
                self.model.module.update_seen_classes(np.unique(train_set.targets))

                self.model.module.mode = self.args.new_mode
                self.model.train()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.train_inc(trainloader, self.args.epochs_new, session, np.unique(train_set.targets), args)
                self.model.eval()
                self.model.module.mode = 'avg_cos'
                tsl, tsa, logs = test(self.model, testloader, 0, args, session,self.word_info)
                acc_df = acc_df.append(logs, ignore_index=True)
                
                print("Build Vision ProtoType")

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
            
        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        
        # end_params = 0.
        end_params = sum(param.numel() for param in self.model.module.parameters())
        print('[Begin] Total parameters: {}'.format(self.init_params))
        print('[END] Total parameters: {}'.format(end_params))
        
    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        if self.args.vit:
            self.args.save_path = self.args.save_path + '%s/' % (self.args.project+'_ViT_Ours')
        else:
            self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        else:
            self.args.save_path = self.args.save_path + 'tsne-Epobase_%d-Eponew_%d-Lr_%.4f-COS_%d-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base,self.args.epochs_new, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join(f'checkpoint/{self.args.out}', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

    def set_log_path(self):
        if self.args.model_dir is not None:
            self.args.save_log_path = '%s/' % self.args.project
            self.args.save_log_path = self.args.save_log_path + '%s' % self.args.dataset
            if 'avg' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_prototype_' + self.args.model_dir.split('/')[-2][:7] + '/'
            if 'ft' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_WaRP_' + 'lr_new_%.3f-epochs_new_%d-keep_frac_%.2f/' % (
                    self.args.lr_new, self.args.epochs_new, self.args.fraction_to_keep)
            self.args.save_log_path = os.path.join('acc_logs', self.args.save_log_path)
            ensure_path(self.args.save_log_path)
            self.args.save_log_path = self.args.save_log_path + self.args.model_dir.split('/')[-2] + '.csv'

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)