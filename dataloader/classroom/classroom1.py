from distutils.log import error
from math import fabs
from re import X
from torch.utils.data import Dataset, DataLoader
import torch
import random
import time
from PIL import Image
import os
from torchvision import transforms
import cv2

# random.seed(42)
classes = [
    "Listening",
    "reading",
    "Using_phone",
    "Using_pad",
    "Using_computers",
    "Scratching_head",
    "writing",
    "talking",
    "standing",
    "Sleeping",
    "teaching",
    "yawning",
    "walking",
    "relaxing",
    "analyzing",
    "Taking_bottle",
    "Gathering_up_bag",
    "drinking",
    "Taking_photos",
    "Listening_to_music",
    "discussing",
    "Setting_equipment",
    "Taking_bag",
    "Blackboard_writing",
    "Blackboard_wiping",
    "Taking_off",
    "Student_demonstrating",
    "eating",
    "reviewing",
    "Hands_up",
    "speaking",
    "Picking_up_computers"
]

#不同 小样本 增量类型
fscil_class = [
    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [23, 25, 21, 22],
        [30, 26, 28, 24],
        [17, 18, 29, 31],
        [20, 27, 16, 19],   ], # a1

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [27, 25, 22],
        [28, 29, 26],
        [31, 23, 24],
        [20, 21, 30],      ], # a2

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],  ], # b1

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22],
        [23, 24, 25],
        [26, 27, 28],
        [29, 30, 31],      ], # b2
]

#不同 常规连续学习 增量类型
cil_class = [
    [   [0, 1, 2, 3, 4, 5, 6, 7],
        [11, 19, 12, 16],
        [10, 13, 22, 21],
        [24, 30, 18, 14],
        [25, 26, 15, 17],
        [28, 31, 29, 27],
        [9,  20, 23,  8],    ], # a1

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [31, 17, 26, 16],
        [21, 25, 27, 30],
        [18, 24, 29, 22],
        [13, 23, 14, 28],
        [19, 20, 12, 15],    ], # a2

    [   [0, 1, 2, 3, 4, 5, 6, 7],
        [8,  9,  10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],     ], # b1

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],    ], # b2
]


class ClassRoom(Dataset):
    """
    一个用于加载不同类型小样本和连续学习数据集的类。
    """
    
    def __init__(self, fscil = True, train = True, classroom_type = 'resnet_a1', session = 0, f_shot = 5):
        
        # 根据参数选择预处理模型和增量类型
        self.pre_model, cil_type = classroom_type.strip().rsplit('_', 1)           
        self.curr_class = []
        self.kown_class = 0
        self.num_max = 500 #设置训练阶段样本数量
        self.classes = classes
        
        #输入检查
        assert self.pre_model in ['resnet', 'vit', 'clipvit']
        assert cil_type in ['a1', 'a2', 'b1', 'b2']
        
        if train==True:
            self.transform = transforms.Compose([
                            transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),  # 随机裁剪
                            transforms.RandomHorizontalFlip(),                          # 随机水平翻转
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
                            transforms.RandomRotation(degrees=15),                     # 随机旋转
                            transforms.ToTensor(),                                     # 转换为Tensor
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
                        ])
        else:
            self.transform=transforms.Compose([
                                            transforms.Resize((224, 224)),  # 调整图像大小为224x224
                                            transforms.ToTensor(),          # 将图像转换为PyTorch张量，并归一化到[0, 1]范围
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

        #根据是否小样本，设置加载数据上限
        if fscil == True:
            if session > 0:
                self.num_max = f_shot
            class_list = fscil_class
        else:
            class_list = cil_class
        
        self.output_base_dir = '/amax/2020/ckl/instances_all/origin_img'
        #根据选择预处理模型类型，设置不同的样本根目录
        root_base = '/media/dataset/dataset_36453_raw/image_features'
        self.root = os.path.join(root_base, self.pre_model)

            
        #根据不同的加载数据类设置，计算当前seesion的旧类数量
        #根据是否 train ，设置 txt 的文件目录
        self.txt_path = '/amax/2020/gkl/CIL/classroom_data/data/index_list/classroom/index'
        index = ['a1', 'a2', 'b1', 'b2'].index(cil_type)
        self.curr_class = class_list[index][session]
        if train:
            self.end = 'train/train_'
            if session > 0:
                if fscil == False and cil_type in ['a1', 'b1']:
                    self.kown_class = 8 + (session-1)*len(self.curr_class)
                if fscil == False and cil_type in ['a2', 'b2']:
                    self.kown_class = 12 + (session-1)*len(self.curr_class)
                if fscil == True and cil_type in ['a1', 'b1']:
                    self.kown_class = 16 + (session-1)*len(self.curr_class)
                if fscil == True and cil_type in ['a2', 'b2']:
                    self.kown_class = 20 + (session-1)*len(self.curr_class)
        else:
            if session != 0:
                temp = []
                for i in range(0, session):
                    temp += class_list[index][i]
                self.curr_class = temp + self.curr_class   
            self.end = 'test/test_'    
            self.num_max = 100
                  
                  
        # 当前增量阶段，单独保存每个类的数据路径和标签
        self.class_data_path = [[] for i in range(len(self.curr_class))]
        self.class_label = [[] for i in range(len(self.curr_class))]
        self.get_every_class_data()       
         
        # 一个增量阶段的总的数据路径和标签
        self.data_path = []
        self.targets = []
        self.get_all_class_data()
        pass


    def get_every_class_data(self,):
        """
        分别读取当前阶段每个类的数据和标签
        """
        for class_index in self.curr_class:
            #存储每个类的 txt 文件路径
            class_path = os.path.join(self.txt_path , (self.end + f'{class_index}.txt'))
            with open(class_path, 'r') as file:
                lines = file.readlines()
            unordered_index = list(range(len(lines)))# 将读取的数据打乱，防止顺序加载    
            random.shuffle(unordered_index)
            
            index = self.curr_class.index(class_index)        
                
            for i in unordered_index:
                if len(self.class_data_path[index]) < self.num_max:#当前数据量小于设定值, 继续向各类别列表中添加
                    
                    data_path, label = lines[i].strip().rsplit(' ', 1) # 去掉行末的换行符并分割
                    data_path = os.path.join(self.root, data_path)
                    label = int(label)
                    
                    if(self.pre_model == 'clipvit'):
                        img_path = data_path.replace('.pkl', '_img.pkl')
                        text_path = data_path.replace('.pkl', '_text.pkl')
                        data_path = (img_path, text_path)
                                      
                        
                    self.class_data_path[index].append(data_path)
                    self.class_label[index].append(index + self.kown_class)
                    
                else:
                    break
            
              
    def get_all_class_data(self, ):
        """
        整合当前增量阶段所有类的数据和标签
        """
        for class_index in range(len(self.curr_class)):
            for data in zip( self.class_data_path[class_index], self.class_label[class_index] ):
                data_path, label = data
                self.data_path.append(data_path)  
                self.targets.append(label)
                
                
    def __len__(self):
        return len(self.data_path)


    def _load_data(self, path):
        with open(path, 'rb') as file:
            data =  torch.squeeze(torch.load(file, map_location=torch.device('cpu'))).detach()
        if self.transform is not None:
            data = self.transform(data)
        return data


    def __getitem__(self, i):
        path, targets = self.data_path[i], self.targets[i]
        
        path = path.split("/")
        filename = f"{self.output_base_dir}/{path[-3]}/{path[-2]}/{path[-1]}"
        new_filename = filename.replace('.pkl', '.jpg')
        data = Image.open(new_filename)
        data = self.transform(data)
        
        
        return data, targets
        
        # if self.pre_model == 'clipvit':
        #     img_data = self._load_data(path[0])
        #     text_data = self._load_data(path[1])
        #     return (img_data, text_data), targets
        # else:
        #     data = self._load_data(path)
        #       return data, targets



# 使用样例
if __name__ == '__main__':

    #train
    
    fs_trainset = ClassRoom(fscil=True, train=True, classroom_type = 'vit_a2', session = 0, f_shot= 5)
    print(len(fs_trainset))
    trainloader = DataLoader(dataset=fs_trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=False)
    for sample in trainloader:   
        img_data , text_label = sample

    # test
    
    fs_testset = ClassRoom(fscil=True, train=False, classroom_type = 'vit_a2', session = 0, f_shot= 5)
    print(len(fs_testset))
    testloader = DataLoader(dataset=fs_testset, batch_size=64, shuffle=True, num_workers=8)
                            
    for sample in testloader:
        img_data , text_label = sample
    
        
        
    