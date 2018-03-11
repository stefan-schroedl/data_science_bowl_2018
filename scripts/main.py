#%pycat main.py

import os
import shutil
import torch
from torch.autograd import Variable
from tqdm import tqdm

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
