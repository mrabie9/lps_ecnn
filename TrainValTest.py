import random
import pickle
import math
import os
import sys
import numpy as np
import torch
import time
np.set_printoptions(threshold=sys.maxsize)

from testers import *
from utils import *
import DataGenerator as DG
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn import preprocessing
from dst import set_valued_predict_from_masses

class CVTrainValTest():
    def __init__(self, base_path,save_path, num_tasks=None):

        self.base_path = base_path
        self.save_path = save_path
        print(base_path)
        print(save_path)

        if num_tasks is not None:
            self.classes_per_task = np.zeros(num_tasks)

    def load_data_cifar(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))
        
        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.training_set = DG.CifarDataGenerator(self.x_train, self.y_train)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.CifarDataGenerator(self.x_test, self.y_test)
        DataParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        
        return self.train_generator

    def load_data_mnist(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))
        
        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.training_set = DG.MnistDataGenerator(self.x_train, self.y_train)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.MnistDataGenerator(self.x_test, self.y_test)
        DataParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        
        return self.train_generator
    
    def load_data_rfmls(self, input_size, batch_size, num_classes, task_num):

        ### Load data 
        # if not os.path.isdir('task0'):
        #     csv_file = './IQ_Data_with_Labels.csv'
        #     # csv_file = './data_sample.txt'
        #     print("Processing IQ data from CSV file:", csv_file)
        #     self.process_iq_data(csv_file)

        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))[:,2:]
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy"))) - 30#- (num_classes*task_num) # device ID offset
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))[:,2:]
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy"))) - 30 #- (num_classes*task_num) # device ID offset

        # print(self.x_train[:6,:6])
        # print(np.unique(self.x_train[:,1]))
        # print("pre", self.y_train[:6])
        
        self.y_train, self.x_train = self.split_features_into_samples(self.x_train, self.y_train, sample_size=input_size)
        self.y_test, self.x_test = self.split_features_into_samples(self.x_test, self.y_test, sample_size=input_size)
        # print(self.y_test.shape, np.sum(self.y_test), self.x_test.shape, self.x_train.shape)
        # from sklearn.utils import shuffle
        # self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        
        # print("post", self.y_train[:6])

        # print(self.y_train.size, self.y_test.size, self.x_train.size/input_size, self.x_test.size/input_size)
        # print("Expecting 14400.... ", np.sum(temp_y[14400:]), self.y_train.size, temp_y.size)

        # self.x_train = self.x_train[:,:input_size]
        # self.x_test = self.x_test[:,:input_size]
        
        scalar_train = preprocessing.StandardScaler().fit(self.x_train)
        scalar_test = preprocessing.StandardScaler().fit(self.x_test)
        self.x_train = scalar_train.transform(self.x_train)
        self.x_test = scalar_test.transform(self.x_test)

        ### 
        self.training_set = DG.IQDataGenerator(self.x_train, self.y_train)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.IQDataGenerator(self.x_test, self.y_test)
        DataParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        
        return self.train_generator
    
    def load_data_dronerc(self, batch_size, args = None, offset = None, mixed_snrs=False, data=None):
        from sklearn.model_selection import train_test_split
        def remap_labels(labels, offset):
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            labels = labels.to(torch.long)  # ensure it's integer type
            unique_labels = torch.unique(labels)
            mapping = {orig.item(): (new + offset) for new, orig in enumerate(unique_labels)}
            remapped = torch.tensor([mapping[int(l)] for l in labels], device=labels.device)
            return remapped
        
        def remove_class_zero(x, y):
                # Find indices where y != 0
                mask = y != 0

                # Apply masks to remove class 0
                x_filtered = x[mask]
                y_filtered = y[mask] - 1

                return x_filtered, y_filtered

        ### Load data 
        # if not os.path.isdir('task0'):
        #     csv_file = './IQ_Data_with_Labels.csv'
        #     # csv_file = './data_sample.txt'
        #     print("Processing IQ data from CSV file:", csv_file)
        #     self.process_iq_data(csv_file)

        if mixed_snrs:
            self.x_train = np.asarray(np.load(data)['xte'])
            self.y_train = np.asarray(np.load(data)['yte'])
            print(data)
            self.x_test = np.asarray(np.load(data)['xte'])
            self.y_test = np.asarray(np.load(data)['yte'])
            print(self.x_test.shape)
            scalar_train = preprocessing.StandardScaler().fit(self.x_train)
            scalar_test = preprocessing.StandardScaler().fit(self.x_test)
            self.x_train = scalar_train.transform(self.x_train)
            self.x_test = scalar_test.transform(self.x_test)
            # self.x_train, _, self.y_train, _ = train_test_split(self.x_train, self.y_train, test_size=0.95, random_state=42, stratify=self.y_train)
            # self.x_test, _, self.y_test, _ = train_test_split(self.x_test, self.y_test, test_size=0.95, random_state=42, stratify=self.y_test)
        elif "radar" in self.base_path.lower():
            self.x_train = np.asarray(np.load(os.path.join(self.base_path, "radar_dataset.npz"))['xtr'])
            self.y_train = np.asarray(np.load(os.path.join(self.base_path, "radar_dataset.npz"))['ytr'])
            self.x_test = np.asarray(np.load(os.path.join(self.base_path, "radar_dataset.npz"))['xte'])
            self.y_test = np.asarray(np.load(os.path.join(self.base_path, "radar_dataset.npz"))['yte'])

            from sklearn.model_selection import train_test_split
            # if args.current_task == 0:
                # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_test, self.y_test, test_size=0.65, random_state=42, stratify=self.y_test)
            self.x_train, _, self.y_train, _ = train_test_split(self.x_train, self.y_train, test_size=0.4, random_state=42, stratify=self.y_train)
            self.x_test, _, self.y_test, _ = train_test_split(self.x_test, self.y_test, test_size=0.4, random_state=42, stratify=self.y_test)
            print(self.x_train.shape)
            # self.y_test = remap_labels(self.y_test, offset)
            # self.y_train = remap_labels(self.y_train, offset)
            # tr_mask = ~np.isin(self.y_train, [11,12,13,14,15,16,17,18])
            # te_mask = ~np.isin(self.y_test, [11,12,13,14,15,16,17,18])
            # self.x_train = self.x_train[tr_mask]
            # self.y_train = self.y_train[tr_mask]
            # self.x_test = self.x_test[te_mask]
            # self.y_test = self.y_test[te_mask]

            print(np.mean(self.x_train), np.std(self.x_train))
            scalar_train = preprocessing.StandardScaler().fit(self.x_train)
            scalar_test = preprocessing.StandardScaler().fit(self.x_test)
            self.x_train = scalar_train.transform(self.x_train)
            self.x_test = scalar_test.transform(self.x_test)
            print(np.mean(self.x_train), np.std(self.x_train))
            
            

            
            print("Labels Remapped to: ", np.unique(self.y_train))
            # print(f"{np.count_nonzero(self.y_test)}/{len(self.y_test) - np.count_nonzero(self.y_test)}")
            print(np.unique_counts(self.y_train))
            print(np.unique_counts(self.y_test))

        elif "usrp" in self.base_path.lower():
            self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train.npz"))['X'])
            self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train.npz"))['y'])
            self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test.npz"))['X'])
            self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test.npz"))['y'])

            print(np.mean(self.x_test), np.std(self.x_test))
            # self.x_train, _, self.y_train, _ = train_test_split(self.x_train, self.y_train, test_size=0.5, random_state=42, stratify=self.y_train)
            # self.x_test, _, self.y_test, _ = train_test_split(self.x_test, self.y_test, test_size=0.5, random_state=42, stratify=self.y_test)
            print(np.unique(self.y_train))

        elif "rfmls" in self.base_path.lower():
            self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train.npz"))['X'])
            self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train.npz"))['y'])
            self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test.npz"))['X'])
            self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test.npz"))['y'])

            self.y_train, self.x_train = self.split_features_into_samples(self.x_train, self.y_train, sample_size=1024)
            self.y_test, self.x_test = self.split_features_into_samples(self.x_test, self.y_test, sample_size=1024)

        else :# "drone" in self.base_path.lower():
            self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train.npz"))['X'])
            self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train.npz"))['y'])
            self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test.npz"))['X'])
            self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test.npz"))['y'])
            # print(np.mean(self.x_train), np.std(self.x_train))
            # scalar_train = preprocessing.StandardScaler().fit(self.x_train)
            # scalar_test = preprocessing.StandardScaler().fit(self.x_test)
            # self.x_train = scalar_train.transform(self.x_train)
            # self.x_test = scalar_test.transform(self.x_test)

            # self.x_train, _, self.y_train, _ = train_test_split(self.x_train, self.y_train, test_size=0.5, random_state=42, stratify=self.y_train)
            # self.x_test, _, self.y_test, _ = train_test_split(self.x_test, self.y_test, test_size=0.5, random_state=42, stratify=self.y_test)
            print(np.mean(self.x_train), np.std(self.x_train))
            print(np.unique(self.y_train))
            
        
        # self.y_train, self.x_train = self.split_features_into_samples(self.x_train, self.y_train, sample_size=1024)
        # self.y_test, self.x_test = self.split_features_into_samples(self.x_test, self.y_test, sample_size=1024)
        # self.x_train, self.y_train = remove_class_zero(self.x_train, self.y_train)
        # self.x_test, self.y_test = remove_class_zero(self.x_test, self.y_test)
        # self.y_train[self.y_train !=0] = 1
        # self.y_test[self.y_test !=0] = 1

        if offset is not None:
            self.y_test = remap_labels(self.y_test, offset) if not args.multi_head else remap_labels(self.y_test, 0)
            self.y_train = remap_labels(self.y_train, offset) if not args.multi_head else remap_labels(self.y_train, 0)
            print("Labels Remapped to: ", np.unique(self.y_train))
        
        # self.y_test = remap_labels(self.y_test, offset=0)
        # self.y_train = remap_labels(self.y_train, offset=0)
        # print("Labels Remapped to: ", np.unique(self.y_train))
        # print(self.x_train.shape)
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(self.x_test, self.y_test, test_size=0.7, random_state=42, stratify=self.y_test)
        # print(np.mean(self.x_train), np.std(self.x_train))
        # scalar_train = preprocessing.StandardScaler().fit(self.x_train)
        # scalar_test = preprocessing.StandardScaler().fit(self.x_test)
        # self.x_train = scalar_train.transform(self.x_train)
        # self.x_test = scalar_test.transform(self.x_test)
        # print(np.mean(self.x_train), np.std(self.x_train))

        ### 
        self.training_set = DG.IQDataGenerator(self.x_train, self.y_train)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.val_set = DG.IQDataGenerator(self.x_val, self.y_val)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
        self.val_generator = DataLoader(self.val_set, **DataParams)

        self.test_set = DG.IQDataGenerator(self.x_test, self.y_test)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        
        return self.train_generator, len(np.unique(self.y_test))
    
    def test_loader(self):
        return self.test_generator

    def split_features_into_samples(self, data, labels, sample_size=2048):
        labels = labels # Extract labels
        features = data # Extract feature columns
        
        num_features = features.shape[1]
        num_splits = num_features // sample_size  # Ensure it's divisible
        
        # print(features[:6,:6])
        # print("features", features[:6])

        assert num_features % sample_size == 0, "Number of features must be divisible by sample size" + str(num_features) + str(sample_size)
        
        # Reshape features into smaller chunks
        reshaped_features = features.reshape(-1, sample_size)  # Reshape while keeping all rows
        
        # Repeat labels to match the new rows
        repeated_labels = np.repeat(labels, num_splits).reshape(-1, 1)
        repeated_labels = np.hstack(repeated_labels)
        
        return repeated_labels, reshaped_features

    def process_iq_data(self, csv_file, output_dir=""):
        """
        Processes IQ data from a CSV file, splits into tasks, and saves in .npy format.

        Parameters:
        - csv_file (str): Path to the CSV file containing IQ data.
        - output_dir (str): Directory where processed data will be saved.
        """
        from ast import literal_eval
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(csv_file)

        # Extract labels from the first column
        labels = df.iloc[:, 0].values

        # Extract IQ data (starting from the second column)
        iq_data = df.iloc[:, 1:].astype(str)  # Ensure it's treated as string before conversion

        # Convert complex string to interleaved floats
        def parse_complex(row):
            return np.hstack([(literal_eval(c).real, literal_eval(c).imag) for c in row])

        iq_floats = np.array([parse_complex(row) for row in iq_data.values])

        # Get unique devices (assuming each label corresponds to a unique device)
        unique_labels = np.unique(labels)
        unique_labels.sort()
        
        # Prepare task groupings (5 tasks, 2 devices each)
        devices_per_task = 2
        tasks = [unique_labels[i:i + devices_per_task] for i in range(0, len(unique_labels), devices_per_task)]

        # os.makedirs(output_dir, exist_ok=True)

        # Process data for each task
        for task_idx, task_labels in enumerate(tasks):
            task_dir = f"task{task_idx}/"
            
            for split in ["train", "test"]:
                os.makedirs(os.path.join(task_dir, split), exist_ok=True)

            X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []

            for label in task_labels:
                label_mask = labels == label
                label_iq = iq_floats[label_mask]
                label_data = labels[label_mask]

                # Split into train/test (75:25)
                X_train, X_test, y_train, y_test = train_test_split(
                    label_iq, label_data, test_size=0.25, random_state=42, stratify=label_data
                )

                # Store data
                X_train_list.append(X_train)
                y_train_list.append(y_train)
                X_test_list.append(X_test)
                y_test_list.append(y_test)


            # Save as .npy
            np.save(os.path.join(task_dir, "train", "X.npy"), np.vstack(X_train_list))
            np.save(os.path.join(task_dir, "train", "y.npy"), np.concatenate(y_train_list))
            np.save(os.path.join(task_dir, "test", "X.npy"), np.vstack(X_test_list))
            np.save(os.path.join(task_dir, "test", "y.npy"), np.concatenate(y_test_list))

        print(f"IQ data processing complete.")

    def load_data_mixture(self, params):
        '''
        Mixture dataset contains 5 tasks, [mnist,cifar,mnist,cifar,mnist]
        Mnist > Cifar => subsample mnist
        # Mnist: 60000
        # Cifar: 5000
        '''
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))
        
        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

        # scale number of training sample
        scale = 1
        trigger = False
        if len(self.y_train) > 5000:
            trigger=True
            params.epochs = 50
            params.epochs_prune = 30
            params.epochs_mask_retrain = 50
            print('Sample {} examples in each training epoch.'.format(int(len(self.y_train)*scale)))
        else:
            params.epochs = 300
            params.epochs_prune = 200
            params.epochs_mask_retrain = 300
            
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.training_set = DG.MixtureDataGenerator(self.x_train, self.y_train, scale=scale, trigger=trigger)
        DataParams = {'batch_size': params.batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.MixtureDataGenerator(self.x_test, self.y_test, trigger=trigger)
        DataParams = {'batch_size': params.batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        
        return params, self.train_generator
    
    def train_model(self, args, model, masks, train_loader, criterion, optimizer, scheduler, epoch):
        atch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        omegas = AverageMeter()
        idx_loss_dict = {}
        
        #if masks:
        #    test_sparsity_mask(args,masks)
        model.train()
        all_preds = []
        tr_acc = []
        all_labels = []
        for i, (input, target) in enumerate(train_loader):
            input = input.float().cuda()
            target = target.long().cuda()
            optimizer.step()
            # scheduler.step()
            # compute output
            output, _, omega, beliefs = model(input, args.current_task, return_features = True) if args.multi_head else model(input, return_features = True) 
            # belief = output
            output = output.float()
            e_utils = output[:, :-1]
            # omega = output[:,-1]

            # if epoch < 5:
            #     criterion = EvidentialLoss(num_classes=args.classes).cuda()
            #     optimizer.param_groups[0]['lr'] = 1e-5
            # else:
            #     optimizer.param_groups[0]['lr'] = 1e-4  
            #     criterion = EvidentialLoss(num_classes=args.classes).cuda()
            
            ce_loss = criterion(e_utils, target, beliefs)
            # print("omega min/max:", omega.min().item(), omega.max().item())

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad norm = {param.grad.norm()}")

                # optimizer.param_groups[0]['lr'] = 1e-3
            # criterion = EvidentialLoss(num_classes=args.classes).cuda()
            # # print(output[0])
            # ce_loss = criterion(belief, target)

            # measure accuracy and record loss
            max_utils, preds = torch.max(e_utils, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.append(torch.from_numpy(target.cpu().numpy()))
            
            omega = 1-max_utils.float()
            # measure accuracy and record loss
            losses.update(ce_loss.item(), input.size(0))
            omegas.update(omega.mean().item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            ce_loss.backward()
            
            if masks:
                with torch.no_grad():
                    for name, W in (model.named_parameters()):
                        # fixed-layers are shared layers for multi-tasks, it should not be trained besides the first task
                        if name in args.fixed_layer:
                            W.grad *= 0
                            continue
                        if name in masks and name in args.pruned_layer:
                            W.grad *= 1-masks[name].cuda()
            
            if scheduler is not None:
                scheduler.step()

            if i % 100 == 0:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                # print('({0}) lr:[{1:.5f}]  '
                #       'Epoch: [{2}][{3}/{4}]\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                #       .format('adam', current_lr,
                #        epoch, i, len(train_loader), loss=losses, top1=top1))
            if i % 100 == 0:
                idx_loss_dict[i] = losses.avg
        
        all_labels = torch.cat(all_labels)  # Combine all label tensors
        weighted_recall = accuracy(all_labels, all_preds)
        tr_acc.append(weighted_recall)
        top1=weighted_recall*100
        print('({0}) lr:[{1:.5f}]  '
                      'Epoch: [{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1:.3f}% ({omegas:.3f}%)\t'
                      .format('adam', current_lr,
                       epoch, len(train_loader), loss=losses, top1=top1, omegas=omegas.avg*100))
        print("Training recall:", accuracy(all_labels, all_preds, weight=None))
        # print(np.unique_counts(all_labels), np.unique_counts(all_preds))
        return model, losses.avg, tr_acc

    def test_model(self, args, model, mask="", test=True, cm=False, ds_layer=None, enable_diagnostics=False, label_names=None, mixed_snrs=False):
        """
        Run evaluation. Returns predictions and metrics. If cm=True, returns all prediction data.
        """
        batch_time = AverageMeter()
        top1 = AverageMeter()
        omega_meter = AverageMeter()

        if mask:
            set_model_mask(model, mask)
        model.eval()

        generator = self.test_generator if test else self.val_generator

        all_preds = []
        all_labels = []
        pred = []
        omegas = []
        beliefs_list = []

        end = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(generator):
                input = input.float().cuda()
                target = target.long().cuda()
                output, features, omega_batch, beliefs = model(input, args.current_task, return_features = True) if args.multi_head else model(input, return_features = True) 
                # beliefs: [B,K], omegas: [B]
                sets, EU, acts = set_valued_predict_from_masses(
                    beliefs, omega_batch, gamma=0.8, nu=0.9,
                    acts="singletons+pairs+Omega", top_pairs_per_sample=1
                )

                # Example: turn chosen sets into readable labels
                def set_to_label(A, idx2name):
                    if A == ("Omega",): return "{all}"
                    return "{" + ",".join(idx2name[i] for i in A) + "}"

                # labels = [set_to_label(A, idx2name) for A in sets]
                output = output.float()
                expected_util = output[:, :-1]
                
                # omega_batch = output[:, -1]

                max_utils, preds = torch.max(expected_util, dim=1)
                omega_batch = (1-max_utils).float()
                omega_meter.update(omega_batch.mean().item(), input.size(0))
                all_preds.extend(preds.cpu().numpy())
                all_labels.append(target.cpu())
                pred.append(preds.cpu())
                omegas.extend(omega_batch.cpu())
                beliefs_list.append(expected_util.cpu())

        # Collate all tensors
        all_labels = torch.cat(all_labels)
        omegas_tensor = torch.stack(omegas)
        pred_tensor = torch.cat(pred)
        beliefs_tensor = torch.cat(beliefs_list)

        unique_labels = torch.unique(all_labels)
        weighted_recall = accuracy(all_labels, all_preds)
        acc = weighted_recall*100
        print('Testing Prec@1 {weighted_recall:.3f}, {omegas:.3f}%'.format(weighted_recall=acc, omegas=omega_meter.avg*100))
        print("Testing recall:", accuracy(all_labels, all_preds, weight=None))


        # Handle diagnostics (either by explicit flag or fallback when accuracy is low)
        if enable_diagnostics :#or acc < 30.0:
            print("\nRunning Dempster-Shafer diagnostics...")
            from diagnostics import analyze_per_class_belief, visualize_prototype_distances, inspect_beta_matrix

            analyze_per_class_belief(model, generator, num_classes=args.classes if args else 17)
            if ds_layer is not None and features is not None:
                visualize_prototype_distances(ds_layer, features, all_labels, class_names=label_names)
                inspect_beta_matrix(ds_layer)

        if cm or enable_diagnostics:
            unique_labels = torch.unique(all_labels)
            return all_labels, all_preds, unique_labels, omegas_tensor, pred_tensor, beliefs_tensor
        elif mixed_snrs:
            return acc, omega_meter.avg*100

        return acc

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(target, output, weight='macro', zero_div=0.0):

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        recall = recall_score(target, output, average=weight, zero_division=zero_div)
        return recall

import torch
import os

def test_retrained_models(args, num_tasks, model_loader, CVTrainValTest):
    """ Load retrained models from each task and test on its respective test data """
    for task in range(num_tasks):
        print(f"\nTesting retrained model for Task {task}...")

        # Load the model
        model = model_loader(args)  # Initialize a new model instance
        save_path = os.path.join(args.save_path_exp, f'task{task}', 'cumu_model.pt')
        state_dict = torch.load(save_path, map_location='cpu')

        # Remove unexpected keys before loading
        # filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.w_mask')}
        model.load_state_dict(state_dict, strict=False)  # Allow missing keys

        # Load test data
        base_path = os.path.join(args.base_path, f'task{task}')
        pipeline = CVTrainValTest(base_path=base_path, save_path=args.save_path_exp)
        pipeline.load_data_mnist(args.batch_size)

        # Test the model
        test_accuracy = pipeline.test_model(args, model)
        print(f"Task {task} Test Accuracy: {test_accuracy:.2f}%")

def run_ds_diagnostics(beliefs, omegas, labels, preds, enabled=True):
    if not enabled:
        return

    print("\n DS Diagnostic Summary")
    print("="*35)

    belief_sum = beliefs.sum(dim=1)
    print(f"Average belief sum (should be < 1): {belief_sum.mean().item():.4f}")
    print(f"Min belief sum: {belief_sum.min().item():.4f}")
    print(f"Max belief sum: {belief_sum.max().item():.4f}")

    print(f"\nAverage Omega (uncertainty): {omegas.mean().item():.4f}")
    print(f"Min Omega: {omegas.min().item():.4f}")
    print(f"Max Omega: {omegas.max().item():.4f}")

    correct = preds == labels
    acc_per_class = {}
    for cls in torch.unique(labels):
        cls_mask = labels == cls
        acc = correct[cls_mask].float().mean().item()
        avg_omega = omegas[cls_mask].mean().item()
        acc_per_class[int(cls.item())] = (acc, avg_omega)

    print("\nPer-Class Accuracy and Avg Omega:")
    for cls, (acc, om) in sorted(acc_per_class.items()):
        print(f"  Class {cls:2d}: Acc={acc*100:.2f}%, Omega={om:.3f}")
    print("="*35 + "\n")