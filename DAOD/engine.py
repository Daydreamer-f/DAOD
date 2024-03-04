import time
import datetime
import json
import torch
from torch.utils.data import DataLoader

from datasets.coco_style_dataset import DataPreFetcher
from datasets.coco_eval import CocoEvaluator

from models.criterion import post_process, get_pseudo_labels
from utils.distributed_utils import is_main_process
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from collections import defaultdict
from typing import List


def train_one_epoch_standard(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    model.train()
    criterion.train()
    fetcher = DataPreFetcher(data_loader, device=device)
    images, masks, annotations = fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(len(data_loader)):
        # Forward
        out = model(images, masks)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        images, masks, annotations = fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= len(data_loader)
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= len(data_loader)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict

def train_one_epoch_fa(model: torch.nn.Module,
                             criterion_train: torch.nn.Module,
                             criterion_target: torch.nn.Module,
                             data_loader: list,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True,
                            coef_target: float = 1.0):
    start_time = time.time()
    model.train()
    criterion_train.train()
    criterion_target.train()
    source_loader = data_loader[0]
    target_loader = data_loader[1]
    fetcher_source = DataPreFetcher(source_loader, device=device)
    fetcher_target = DataPreFetcher(target_loader, device=device)
    num_of_rounds = len(source_loader)
    images_s, masks_s, annotations_s = fetcher_source.next()
    images_t, masks_t, _ = fetcher_target.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(num_of_rounds):
        # Forward
        out_s = model(images_s, masks_s, domain_label=0)
        out_t = model(images_t, masks_t, domain_label=1)
        # Loss
        loss_source, loss_dict = criterion_train(out_s, annotations_s, domain_label=0)
        loss_target, loss_dict_t = criterion_target(out_t, domain_label=1)
        loss = loss_source + loss_target * coef_target
        loss_dict['target_domain_loss'] = loss_target
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        images_s, masks_s, annotations_s = fetcher_source.next()
        images_t, masks_t, _ = fetcher_target.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(source_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= num_of_rounds
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= num_of_rounds
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


def train_one_epoch_teaching(student_model: torch.nn.Module,
                             teacher_model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             criterion_pseudo: torch.nn.Module,
                             source_loader: DataLoader,
                             target_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             thresholds: List[float],
                             coef_target: float,
                             mask_ratio: float,
                             alpha_ema: float,
                             device: torch.device,
                             epoch: int,
                             enable_mae: bool = False,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion.train()
    criterion_pseudo.train()
    source_fetcher = DataPreFetcher(source_loader, device=device)
    target_fetcher = DataPreFetcher(target_loader, device=device)
    source_images, source_masks, source_annotations = source_fetcher.next()
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    # Training data statistics
    epoch_source_loss_dict = defaultdict(float)
    epoch_target_loss_dict = defaultdict(float)
    total_iters = min(len(source_loader), len(target_loader))
    for i in range(total_iters):
        # Source forward
        source_out = student_model(source_images, source_masks)
        source_loss, source_loss_dict = criterion(source_out, source_annotations, teacher_label=0)
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logits_all'][-1], teacher_out['boxes_all'][-1], thresholds)
        # Target student forward
        target_student_out = student_model(target_student_images, target_masks, enable_mae, mask_ratio)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels, None, 1, enable_mae)
        # Backward
        optimizer.zero_grad()
        loss = source_loss + coef_target * target_loss
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()
        # Record epoch losses
        epoch_loss += loss
        # update loss_dict
        for k, v in source_loss_dict.items():
            epoch_source_loss_dict[k] += v.detach().cpu().item()
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()
        # EMA update teacher
        with torch.no_grad():
            state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
            for key, value in state_dict.items():
                state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
            teacher_model.load_state_dict(state_dict)
        # Data pre-fetch
        source_images, source_masks, source_annotations = source_fetcher.next()
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_source_loss_dict.items():
        epoch_source_loss_dict[k] /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_source_loss_dict, epoch_target_loss_dict


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             flush: bool = False):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        dataset_annotations = [[] for _ in range(len(coco_data['images']))]
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images, masks)
        logits_all, boxes_all = out['logits_all'], out['boxes_all']
        # Get pseudo labels
        if output_result_labels:
            results = get_pseudo_labels(logits_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
            for anno, res in zip(annotations, results):
                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    pseudo_anno = {
                        'id': 0,
                        'image_id': image_id,
                        'category_id': label,
                        'iscrowd': 0,
                        'area': box[-2] * box[-1],
                        'bbox': box
                    }
                    dataset_annotations[image_id].append(pseudo_anno)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logits_all[-1], boxes_all[-1], orig_image_sizes, 100)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    # Save results
    if output_result_labels:
        dataset_annotations_return = []
        id_cnt = 0
        for image_anno in dataset_annotations:
            for box_anno in image_anno:
                box_anno['id'] = id_cnt
                id_cnt += 1
                dataset_annotations_return.append(box_anno)
        coco_data['annotations'] = dataset_annotations_return
        return aps, epoch_loss / len(data_loader_val), coco_data
    return aps, epoch_loss / len(data_loader_val)
