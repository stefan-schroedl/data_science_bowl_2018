import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
#from post_process import parametric_pipeline

from utils import add_contour

## Eval metrics

def precision_at(overlap, thresh):
    matches = (overlap > thresh).astype(int)
    matches_by_pred = np.sum(matches, axis=0)
    matches_by_target = np.sum(matches, axis=1)
    true_positives = (matches_by_target == 1).astype(int)   # Correct objects
    false_positives = (matches_by_pred == 0).astype(int)  # Extra objects
    false_negatives = (matches_by_target == 0).astype(int)  # Missed objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn, matches_by_pred, matches_by_target


def union_intersection(labels, y_pred, exclude_bg=True):

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    #np.set_printoptions(threshold=np.nan)
    #print intersection

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    if exclude_bg:
        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        area_true = area_true[1:,]
        area_pred = area_pred[:,1:]

    union[union == 0] = 1e-9

    return union, intersection, area_true, area_pred


def iou_metric(labels, y_pred, print_table=False):

    if labels.max() == 0:
        return 0.0

    union, intersection, _, _ = union_intersection(labels, y_pred)

    # Compute the intersection over union
    iou = intersection.astype(float) / union

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn, _, _ = precision_at(iou, t)

        if (tp + fp + fn) > 0:
            p = 1.0 * tp / (tp + fp + fn)
        else:
            p = 0.0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def print_diag(p, p_loc, mean_prec, mean_rec, missed_rate, extra_rate, oseg, useg):
    s = 'average precision: %.1f %%; max score improvment without mislocations: %.1f %%;' % (100*p, 100*p_loc)
    if missed_rate > 0.0:
        s = s + ' missed %.1f %% of positives;' % (100.0 * missed_rate)
    if extra_rate > 0.0:
        s = s + ' predicted %.1f %% false positives;' % (100.0 * extra_rate)
    if oseg > 0.0:
        s = s + '  %.1f %% of objects predicted multiple times;' % (100.0 * oseg)
    if useg > 0.0:
        s = s + '  %.1f %% of predictions covering multiple objects;' % (100.0 * useg)

    if mean_prec > mean_rec:
        s = s + ' segments tend to be too small:'
    else:
        s = s + ' segments tend to be too large:'
    s = s + ' pixel precision: %.1f %%, pixel recall: %.1f %%' % (100.0 * mean_prec, 100.0 * mean_rec)
    print(s)


# see the SDS paper for motivation and discussion
def diagnose_errors(labels, y_pred, threshold=.5, print_message=True):

    union, intersection, area_true, area_pred = union_intersection(labels, y_pred)

    # Compute the intersection over union
    iou = intersection.astype(float) / union

    tp, fp, fn, matches_by_pred, matches_by_target = precision_at(iou, threshold)

    denom = 1.0 * (tp + fp + fn)
    if denom <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    p = tp / denom

    # what is the best possible score when loosely overlapping locations were fixed?

    # sort newly matched indices by iou
    # assign less stringent matches greedily if both pred and target haven't been matched
    matches0 = np.where(iou > 0.1)
    matches0 = sorted([(x,y,iou[x,y]) for x,y in zip(matches0[0], matches0[1])], key = lambda x: -x[2])

    iou_loc = np.copy(iou)
    for x,y,_ in matches0:
        if matches_by_target[x] == 0 and matches_by_pred[y] == 0 and iou[x,y] >= np.max(iou[x,:]):
            iou_loc[:,y] = 0.0
            iou_loc[x,y] = 1.0
            matches_by_target[x] = 1
            matches_by_pred[y] = 1

    tp_loc, fp_loc, fn_loc, matches_by_pred_loc, matches_by_target_loc = precision_at(iou_loc, threshold)

    p_loc = 0.0
    denom_loc = 1.0 * (tp_loc + fp_loc + fn_loc)
    if denom_loc > 0:
        p_loc = tp_loc / denom_loc - p

    missed_rate = np.sum((np.sum(((iou > 0.1).astype(int)), axis=1) == 0).astype(int)) / denom
    extra_rate = np.sum((np.sum(((iou > 0.1).astype(int)), axis=0) == 0).astype(int)) / denom

    prec_thresh = 0.67

    # precision measure
    prec = intersection.astype(float) / np.tile(area_pred, (intersection.shape[0], 1))
    # Objects predicted multiple times
    oseg = np.sum((np.sum(prec>prec_thresh, axis=1) > 1).astype(int))  / denom

    # recall measure
    rec = intersection.astype(float) / np.tile(area_true, (1, intersection.shape[1]))
    # Predictions overlapping multiple objects
    useg = np.sum((np.sum((rec>prec_thresh).astype(int), axis=0) > 1).astype(int)) / denom

    # pixel precision and recall for existing match
    mean_prec = np.mean(prec[(iou > threshold)])
    mean_rec = np.mean(rec[(iou > threshold)])

    if print_message:
        print_diag(p, p_loc, mean_prec, mean_rec, missed_rate, extra_rate, oseg, useg)
    return p, p_loc, mean_prec, mean_rec, missed_rate, extra_rate, oseg, useg


def show_compare_gt(img, masks, thresh=0.5, **opts):
    pred_masks = parametric_pipeline(img, **opts)
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.grid(None)
    ax.imshow(img)
    add_contour(masks, ax, 'green')
    add_contour(pred_masks, ax, 'red')
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return diagnose_errors(masks, pred_masks, thresh, print_message=True)

    
#####

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        prediction = self.sigmoid(output)
        return 1 - 2 * torch.sum(prediction * target) / (torch.sum(prediction) + torch.sum(target) + 1e-7)


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        prediction = self.sigmoid(output)
        prod = torch.sum(prediction * target)
        return 1 - 2 * (prod + 1.0) / (torch.sum(prediction) + torch.sum(target) - prod + 1.0)

    
##  http://geek.csdn.net/news/detail/126833
def weighted_binary_cross_entropy_with_logits(logits, labels, weights):

    loss = weights*(logits.clamp(min=0) - logits*labels + torch.log(1 + torch.exp(-logits.abs())))
    loss = loss.sum()/(weights.sum()+1e-12)

    return loss

def segmentation_loss(output, target):
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    return bce(output, target) + dice(output, target)


def cross_entropy(output, target, squeeze=False):
    if squeeze:
        target = target.squeeze(1)
    return F.nll_loss(output, target)


def mse(output, target, squeeze=False):
    if squeeze:
        target = target.squeeze(1)
    return F.mse_loss(output, target)


def multi_output_cross_entropy(outputs, targets):
    losses = []
    for output, target in zip(outputs, targets):
        loss = cross_entropy(output, target)
        losses.append(loss)
    return sum(losses) / len(losses)


def score_model(model, loss_function, datagen):
    batch_gen, steps = datagen
    partial_batch_losses = {}
    for batch_id, data in enumerate(batch_gen):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X, volatile=True).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor, volatile=True).cuda())
        else:
            X = Variable(X, volatile=True)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor, volatile=True))

        outputs = model(X)
        if len(loss_function) == 1:
            for (name, loss_function_one, weight), target in zip(loss_function, targets_var):
                loss_sum = loss_function_one(outputs, target) * weight
        else:
            batch_losses = []
            for (name, loss_function_one, weight), output, target in zip(loss_function, outputs, targets_var):
                loss = loss_function_one(output, target) * weight
                batch_losses.append(loss)
                partial_batch_losses.setdefault(name, []).append(loss)
            loss_sum = sum(batch_losses)
        partial_batch_losses.setdefault('sum', []).append(loss_sum)
        if batch_id == steps:
            break

    average_losses = {name: sum(losses) / steps for name, losses in partial_batch_losses.items()}
    return average_losses


def torch_acc_score(output, target):
    output = output.data.cpu().numpy()
    y_true = target.numpy()
    y_pred = output.argmax(axis=1)

    return accuracy_score(y_true, y_pred)


def torch_acc_score_multi_output(outputs, targets, take_first=None):
    accuracies = []
    for i, (output, target) in enumerate(zip(outputs, targets)):
        if i == take_first:
            break
        accuracy = torch_acc_score(output, target)
        accuracies.append(accuracy)
    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy

