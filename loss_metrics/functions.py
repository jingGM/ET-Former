import torch
import torch.nn as nn
import torch.nn.functional as F


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]
    # print(torch.sum(mask))
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

def precision_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
    )

def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights, use_mean=False):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    if use_mean:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction="mean")
        loss = criterion(pred, target.long())
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction="none")
        loss = criterion(pred, target.long())
        loss_valid = loss[target != 255]
        loss = torch.mean(loss_valid)
    return loss


def compute_super_CP_multilabel_loss(pred_logits, CP_mega_matrices):
    logits = []
    labels = []
    bs, n_relations, _, _ = pred_logits.shape
    for i in range(bs):
        pred_logit = pred_logits[i, :, :, :].permute(
            0, 2, 1
        )  # n_relations, N, n_mega_voxels
        CP_mega_matrix = CP_mega_matrices[i]  # n_relations, N, n_mega_voxels
        logits.append(pred_logit.reshape(n_relations, -1))
        labels.append(CP_mega_matrix.reshape(n_relations, -1))

    logits = torch.cat(logits, dim=1).T  # M, 4
    labels = torch.cat(labels, dim=1).T  # M, 4

    cnt_neg = (labels == 0).sum(0)
    cnt_pos = labels.sum(0)
    pos_weight = cnt_neg / cnt_pos
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_bce = criterion(logits, labels.float())
    return loss_bce

def BCE_ssc_loss(pred, target, class_weights, alpha):

    class_weights[0] = 1-alpha    # empty
    class_weights[1] = alpha    # occupied

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction="none")
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)

    return loss_valid_mean