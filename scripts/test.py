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
stage_name = 'stage1'
dset = NucleusDataset(dsb_data_dir, stage_name,transform=train_transform)
dset.transform=train_transform
#dset_save = dset
#train_idx, valid_idx = train_valid_split(dset,test_size=0.05,random_seed=1,shuffle=True)
#train_sampler = SubsetRandomSampler(train_idx)
#valid_sampler = SubsetRandomSampler(valid_idx)
#train_loader = DataLoader(dset,batch_size=1,sampler=train_sampler,num_workers=4)
#valid_loader = DataLoader(dset,batch_size=1,sampler=valid_sampler,num_workers=4)

# hack: this image format (1388, 1040) occurs only ones, stratify complains .. 
dset.data_df = dset.data_df[dset.data_df['size'] != (1388, 1040)]

stratify = dset.data_df['images'].map(lambda x: '{}'.format(x.size))
train_dset, valid_dset = dset.train_test_split(test_size=0.05, random_state=1, shuffle=True, stratify=stratify)

print_every = 10
save_every = 10
eval_every = 10
    
epochs = 10

model = CNN()
it = 0
best_loss = 1e20
best_it = 0
stats = []


criterion = nn.MSELoss()
#criterion = nn.BCEWithLogitsLoss()
#criterion = loss.DiceLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-4)


train_loader = DataLoader(train_dset, batch_size=1,shuffle=True)
valid_loader = DataLoader(valid_dset, batch_size=1,shuffle=True)

for epoch in range(epochs):
    # adjust_learning_rate(optimizer, epoch)
    it, best_loss, best_it = train(train_loader, valid_loader, model, criterion, optimizer, stats, epoch, eval_every, print_every, save_every)
print it, best_loss, best_it
