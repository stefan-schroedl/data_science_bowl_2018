#%pycat main.py

import os
import shutil
import torch
from torch.autograd import Variable


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
    for i, (img, labels) in enumerate(loader):
        img, labels = Variable(img), Variable(labels)
        outputs = model(img)
        loss = criterion(outputs, labels)
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
        best_loss,
        best_it,
        epoch,
        it,
        eval_every,
        print_every,
        save_every):
    running_loss = 0.0
    cnt = 0

    for i, (img, labels) in enumerate(train_loader, it + 1):
        img, labels = Variable(img), Variable(labels)
        
        model.train(True)

        outputs = model(img)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt = cnt + 1

        running_loss += loss.data[0]
        if cnt > 0 and i % eval_every == 0:

            l = validate(model, valid_loader, criterion)
            stats.append((i, running_loss / cnt, l))
            running_loss = 0.0

            if cnt > 0 and i % print_every == 0:
                print('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f' %
                      (epoch, i, stats[-1][1], stats[-1][2]))
            if i % save_every == 0:
                is_best = False
                if best_loss > l:
                    best_loss = l
                    best_it = i
                    is_best = True
                    save_model(
                        model,
                        optimizer,
                        stats,
                        i,
                        is_best,
                        'model_save.pth.tar')
    return i, best_loss, best_it

def baseline(
        loader,
        criterion,
        it):
    
    m = 0.0
    cnt = 0.0
    for i, (img, labels) in enumerate(loader):
        if i > it:
            break
        m += labels[0].sum()
        cnt += labels[0].numel()
    m = m / cnt

    running_loss = 0.0
    
    for i, (img, labels) in enumerate(loader):
        if i > it:
            break
        outputs = labels.clone()        
        outputs = torch.clamp(outputs, m, m)
        img, labels = Variable(img), Variable(labels)
        outputs = Variable(outputs)
    
        loss = criterion(outputs, labels)

        running_loss += loss.data[0]

    return running_loss/it, m


def adjust_learning_rate(optimizer, epoch, lr0):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
