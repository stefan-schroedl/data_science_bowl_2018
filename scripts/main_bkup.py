#%pycat main.py
import cv2
import os
import shutil
import torch
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from KNN import *

def save_model(
        model,
        optimizer,
        stats,
        it,
        is_best=False,
        fname='model_save.pth.tar'):
    m_state_dict = model.state_dict()
    o_state_dict = optimizer.state_dict()

    torch.save({
        'it': it,
        'model_state_dict': m_state_dict,
        'optimizer_state_dict': o_state_dict,
        'stats': stats},
        fname)
    if is_best:
        print('new best: ', stats[-1])
        shutil.copyfile(fname, 'model_best.pth.tar')


def load_model(model, optimizer, fname='model_best.pth.tar'):

    if os.path.isfile(fname):
        checkpoint = torch.load(fname)
        it = checkpoint['it']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stats = checkpoint['stats']
        print(
            "=> loaded checkpoint '{}' (iteration {})".format(
                fname, checkpoint['it']))
        return it, stats
    else:
        print("=> no checkpoint found at '{}'".format(fname))
        return None, None

def validate_knn(model, loader, criterion):
    running_loss = 0.0
    cnt = 0
    for i, (img, (labels, labels_seg)) in enumerate(loader):
        p_img,p_seg,p_boundary,p_blend=model.predict(img)
        torch_p_seg = torch.from_numpy(p_seg[None,:,:].astype(np.float)/255).float()
        #torch_p_boundary = torch.from_numpy(p_boundary[None,:,:].astype(np.float)/255).float()
        #torch_p_blend = torch.from_numpy(p_blend[None,:,:].astype(np.float)/255).float()
        cv2.imshow('pim',p_img)
        cv2.imshow('pseg',p_seg)
        cv2.imshow('pbound',p_boundary)
        cv2.imshow('pblend',p_blend)
        cv2.waitKey(40000)
        loss = criterion(Variable(torch_p_seg), Variable(labels_seg.float()))
        running_loss += loss.data[0]
        cnt = cnt + 1
    l = running_loss / cnt
    return l

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    cnt = 0
    for i, (img, (labels, labels_seg)) in enumerate(loader):
        img, labels_seg = Variable(img), Variable(labels_seg)
        outputs = model(img)
        loss = criterion(outputs, labels_seg)
        running_loss += loss.data[0]
        cnt = cnt + 1
    l = running_loss / cnt
    return l

def train_knn(
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        stats,
        epoch,
        eval_every,
        print_every,
        save_every):
    running_loss = 0.0
    cnt = 0
    global it, best_it, best_loss
    for it, (img, (labels, labels_seg)) in tqdm(enumerate(train_loader, it + 1)):
        model.prepare_fit(img,labels,labels_seg)

        cnt = cnt + 1

        if cnt > 0 and it % eval_every == 0:
            model.fit()
            l = validate_knn(model, valid_loader, criterion)
            img,mask,boundary,blend=model.predict(img)
            stats.append((it, running_loss / cnt, l))
            running_loss = 0.0

            if cnt > 0 and it % print_every == 0:
                print('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f' %
                      (epoch, it, stats[-1][1], stats[-1][2]))
            if it % save_every == 0:
                is_best = False
                if best_loss > l:
                    best_loss = l
                    best_it = it
                    is_best = True
    return it, best_loss, best_it

def train(
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        stats,
        epoch,
        eval_every,
        print_every,
        save_every):
    running_loss = 0.0
    cnt = 0
    global it, best_it, best_loss
    for it, (img, (labels, labels_seg)) in tqdm(enumerate(train_loader, it + 1)):
        img, labels_seg = Variable(img), Variable(labels_seg)
        
        model.train(True)

        outputs = model(img)
        loss = criterion(outputs, labels_seg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt = cnt + 1

        running_loss += loss.data[0]
        if cnt > 0 and it % eval_every == 0:

            l = validate(model, valid_loader, criterion)
            stats.append((it, running_loss / cnt, l))
            running_loss = 0.0

            if cnt > 0 and it % print_every == 0:
                print('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f' %
                      (epoch, it, stats[-1][1], stats[-1][2]))
            if it % save_every == 0:
                is_best = False
                if best_loss > l:
                    best_loss = l
                    best_it = it
                    is_best = True
                    save_model(
                        model,
                        optimizer,
                        stats,
                        it,
                        is_best,
                        'model_save.pth.tar')
    return it, best_loss, best_it

def baseline(
        train_loader,
        valid_loader,
        criterion,
        it):
    
    m = 0.0
    cnt = 0.0
    for i, (img, (labels, labels_seg)) in enumerate(train_loader):
        if i > it:
            break
        m += labels_seg[0].sum()
        cnt += labels_seg[0].numel()
    m = m / cnt

    running_loss = 0.0
    
    cnt = 0
    for i, (img, (labels,labels_seg)) in enumerate(valid_loader):
        outputs = labels_seg.clone()        
        outputs = torch.clamp(outputs, m, m)
        img, labels = Variable(img), Variable(labels_seg)
        outputs = Variable(outputs)
    
        loss = criterion(outputs, labels_seg)

        running_loss += loss.data[0]
        cnt += 1

    return running_loss/cnt, m


def adjust_learning_rate(optimizer, epoch, lr0):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

from torch import optim
from architectures import *
from main import *
from dataset import *

import transform
from transform import random_rotate90_transform2
import torchvision
from torchvision.transforms import ToTensor, ToPILImage

#Note: for image and mask, there is no compatible solution that can use transforms.Compse(), see https://github.com/pytorch/vision/issues/9
#transformations = transforms.Compose([random_rotate90_transform2(),transforms.ToTensor(),])

def train_transform(img, mask, mask_seg):
    # HACK (make dims consistent, first one is channels)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, 2)
    if len(mask_seg.shape) == 2:
        mask_seg = np.expand_dims(mask_seg, 2)
    img, mask, mask_seg = random_rotate90_transform2(0.5, img, mask, mask_seg)
    img = ToTensor()(img)
    mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float()
    mask_seg = torch.from_numpy(np.transpose(mask_seg, (2, 0, 1))).float()
    return img, mask, mask_seg


#dsb_data_dir = os.path.join('..', '..', 'input')

dsb_data_dir = os.path.join('..', 'explore')
stage_name = 'stagesmall'
dset = NucleusDataset(dsb_data_dir, stage_name,transform=train_transform)
dset.data_df = dset.data_df[dset.data_df['size'] != (1388, 1040)]

stratify = dset.data_df['images'].map(lambda x: '{}'.format(x.size))
train_dset, valid_dset = dset.train_test_split(test_size=0.1, random_state=1, shuffle=True) #, stratify=stratify)

print_every = 10
save_every = 10
eval_every = 10
    
epochs = 10

model = CNN()
knn_model = KNN()
it = 0
best_loss = 1e20
best_it = 0
stats = []


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-4)

train_loader = DataLoader(train_dset, batch_size=1,shuffle=True)
valid_loader = DataLoader(valid_dset, batch_size=1,shuffle=True)

for epoch in range(epochs):
    # adjust_learning_rate(optimizer, epoch)
    it, best_loss, best_it = train_knn(train_loader, valid_loader, knn_model, criterion, None, stats, epoch, eval_every, print_every, save_every)
    #it, best_loss, best_it = train(train_loader, valid_loader, model, criterion, optimizer, stats, epoch, eval_every, print_every, save_every)
print it, best_loss, best_it
