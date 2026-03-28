import os, sys, json, time

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

_n_threads = 8
os.environ['OMP_NUM_THREADS'] = str(_n_threads)
os.environ['MKL_NUM_THREADS'] = str(_n_threads)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from car_color_utils import (
    load_dvm_dataset,
    CarColorDataset,
    CustomResNet,
    make_pretrained_resnet18,
    train_transform,
    train_transform_strong,
    test_transform,
)

DATA_DIR              = 'confirmed_fronts'
OUTPUT_DIR            = 'training_output'
MIN_SAMPLES_PER_CLASS = 500
TRAIN_SUBSAMPLE       = 1.0
BATCH_SIZE            = 64
NUM_WORKERS           = 4
NUM_EPOCHS            = 50
EARLY_STOPPING        = 12
LEARNING_RATE         = 1e-3
SEED                  = 42


def _bar(iterable, desc):
    return tqdm(iterable, desc=desc, mininterval=2,
                file=sys.stdout, dynamic_ncols=False, ncols=100)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = total = 0
    for images, labels in _bar(loader, '  train'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for images, labels in _bar(loader, '  val  '):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    loss = running_loss / len(loader)
    acc  = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / n
    f1   = f1_score(all_labels, all_preds, average='macro')

    return loss, acc, f1, all_preds, all_labels

def train_model(model, train_loader, val_loader, criterion,
                optimizer, scheduler, num_epochs, device,
                early_stopping_patience=7):
    best_f1 = 0.0
    best_sd = None
    wait = 0
    history = dict(train_loss=[], train_acc=[],
                   val_loss=[], val_acc=[], val_f1=[])

    for epoch in range(num_epochs):
        log(f"\n{'─'*50}  Epoch {epoch+1}/{num_epochs}  {'─'*10}")

        tl, ta = train_epoch(model, train_loader, criterion,
                             optimizer, device)
        vl, va, vf, _, _ = validate(model, val_loader, criterion, device)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(vf)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        history['val_f1'].append(vf)

        log(f"  train  loss={tl:.4f}  acc={ta:.2f}%")
        log(f"  val    loss={vl:.4f}  acc={va:.2f}%  F1={vf:.4f}  lr={current_lr:.1e}")

        if vf > best_f1:
            best_f1 = vf
            wait = 0
            best_sd = {k: v.cpu().clone()
                       for k, v in model.state_dict().items()}
            log(f" new best F1 = {best_f1:.4f}")
        else:
            wait += 1
            log(f" … no improvement ({wait}/{early_stopping_patience})")

            if early_stopping_patience and wait >= early_stopping_patience:
                log(f"\n  Early stopping")
                break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return history, best_f1

def log(msg):
    print(msg, flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(_n_threads)
    log(f"Device: {device},  threads: {_n_threads}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    (image_paths, labels, color_labels,
     metadata, color_to_idx, color_names) = load_dvm_dataset(
        DATA_DIR, min_samples_per_class=MIN_SAMPLES_PER_CLASS)
    num_classes = len(color_names)

    train_p, test_p, train_l, test_l = train_test_split(
        image_paths, labels, test_size=0.2,
        random_state=SEED, stratify=labels)
    train_p, val_p, train_l, val_l = train_test_split(
        train_p, train_l, test_size=0.2,
        random_state=SEED, stratify=train_l)

    if TRAIN_SUBSAMPLE < 1.0:
        train_p, _, train_l, _ = train_test_split(
            train_p, train_l, train_size=TRAIN_SUBSAMPLE,
            random_state=SEED, stratify=train_l)
        log(f"Train subsample {TRAIN_SUBSAMPLE*100:.0f}%: "
            f"{len(train_p)} images")

    log(f"Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    split = dict(train_paths=train_p, train_labels=train_l,
                 val_paths=val_p, val_labels=val_l,
                 test_paths=test_p, test_labels=test_l)
    with open(os.path.join(OUTPUT_DIR, 'split.json'), 'w') as f:
        json.dump(split, f)

    use_cuda = torch.cuda.is_available()
    lkw = dict(batch_size=BATCH_SIZE,
               num_workers=NUM_WORKERS if use_cuda else 0,
               pin_memory=use_cuda)

    val_loader = DataLoader(
        CarColorDataset(val_p, val_l, test_transform),
        shuffle=False, **lkw)

    log("\n" + "="*60)
    log("  SCRATCH (CustomResNet v2)")
    log("="*60)

    train_counts = Counter(train_l)
    n_train = len(train_l)
    class_weights = torch.zeros(num_classes)
    for i in range(num_classes):
        class_weights[i] = n_train / (num_classes * train_counts[i])
    class_weights = class_weights.to(device)

    log(f"  Class weights:")
    for i, name in enumerate(color_names):
        log(f"    {name:>8s}: {class_weights[i]:.2f}  "
            f"({train_counts[i]} samples)")

    criterion_scratch = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.1)

    train_loader_scratch = DataLoader(
        CarColorDataset(train_p, train_l, train_transform_strong),
        shuffle=True, **lkw)

    model_s = CustomResNet(num_classes=num_classes, dropout=0.5).to(device)
    n_params = sum(p.numel() for p in model_s.parameters())
    log(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")

    opt_s = optim.AdamW(model_s.parameters(),
                        lr=LEARNING_RATE, weight_decay=5e-4)

    sch_s = optim.lr_scheduler.ReduceLROnPlateau(
        opt_s, mode='max', factor=0.3, patience=3, min_lr=1e-6)

    hist_s, best_s = train_model(
        model_s, train_loader_scratch, val_loader, criterion_scratch,
        opt_s, sch_s, NUM_EPOCHS, device, EARLY_STOPPING)

    torch.save(model_s.state_dict(),
               os.path.join(OUTPUT_DIR, 'model_scratch.pth'))
    with open(os.path.join(OUTPUT_DIR, 'history_scratch.json'), 'w') as f:
        json.dump(hist_s, f)
    log(f"\n  Scratch saved (best F1={best_s:.4f})")

    log("\n" + "="*60)
    log("  PRETRAINED (ResNet-18)")
    log("="*60)

    train_loader_pretrained = DataLoader(
        CarColorDataset(train_p, train_l, train_transform),
        shuffle=True, **lkw)

    criterion_pretrained = nn.CrossEntropyLoss()

    model_p = make_pretrained_resnet18(num_classes).to(device)
    log(f"  Params: {sum(p.numel() for p in model_p.parameters()):,}")

    opt_p = optim.Adam([
        {'params': model_p.conv1.parameters(),  'lr': LEARNING_RATE * 0.01},
        {'params': model_p.layer1.parameters(), 'lr': LEARNING_RATE * 0.01},
        {'params': model_p.layer2.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model_p.layer3.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model_p.layer4.parameters(), 'lr': LEARNING_RATE * 0.5},
        {'params': model_p.fc.parameters(),     'lr': LEARNING_RATE},
    ], weight_decay=1e-4)
    
    sch_p = optim.lr_scheduler.ReduceLROnPlateau(
        opt_p, mode='max', factor=0.5, patience=5)

    hist_p, best_p = train_model(
        model_p, train_loader_pretrained, val_loader, criterion_pretrained,
        opt_p, sch_p, NUM_EPOCHS, device, EARLY_STOPPING)

    torch.save(model_p.state_dict(),
               os.path.join(OUTPUT_DIR, 'model_pretrained.pth'))
    
    with open(os.path.join(OUTPUT_DIR, 'history_pretrained.json'), 'w') as f:
        json.dump(hist_p, f)

    log(f"\n  Pretrained saved (best F1={best_p:.4f})")

    meta = dict(
        color_names=color_names,
        color_to_idx=color_to_idx,
        num_classes=num_classes,
        best_f1_scratch=best_s,
        best_f1_pretrained=best_p,
        num_train=len(train_p),
        num_val=len(val_p),
        num_test=len(test_p),
    )

    with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    log("\n" + "="*60)
    log(f"  Scratch     best val F1 = {best_s:.4f}")
    log(f"  Pretrained  best val F1 = {best_p:.4f}")
    log(f"  Saved to {OUTPUT_DIR}/")
    log("="*60)


if __name__ == '__main__':
    main()