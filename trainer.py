
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, Mult_Out
from torchvision import transforms
from datasets.dataset_prostate import Prostate_dataset
from SSIM import SSIM


def muti_loss(CE, loss_fn, x, d1, d2, d3, d4, gt):
    if CE:
        gt = gt[:].long()
        loss0 = loss_fn(x, gt)
        loss1 = loss_fn(d1, gt)
        loss2 = loss_fn(d2, gt)
        loss3 = loss_fn(d3, gt)
        loss4 = loss_fn(d4, gt)
    else:
        loss0 = loss_fn(x, gt, softmax=True)
        loss1 = loss_fn(d1, gt, softmax=True)
        loss2 = loss_fn(d2, gt, softmax=True)
        loss3 = loss_fn(d3, gt, softmax=True)
        loss4 = loss_fn(d4, gt, softmax=True)
    loss = 0.6 * loss0 + 0.1 * loss1 + 0.1 * loss2 + 0.1 * loss3 + 0.1 * loss4
    return loss, loss0


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    # batch_size = args.batch_size * args.n_gpu
    batch_size = args.batch_size
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(),lr=0.001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs, side = model(image_batch)
            loss_ce_1 = ce_loss(outputs, label_batch[:].long())
            loss_ce_2 = ce_loss(side, label_batch[:].long())
            loss_dice_1 = dice_loss(outputs, label_batch, softmax=True)
            loss_dice_2 = dice_loss(side, label_batch, softmax=True)
            loss = 0.4 * (0.6 * loss_ce_1 + 0.4 * loss_ce_2) + 0.6 * (0.6 * loss_dice_1 + 0.4 * loss_dice_2)
            # outputs, d1, d2, d3, d4 = model(image_batch)
            # loss_ce, loss_ce_1 = muti_loss(True, ce_loss, outputs, d1, d2, d3, d4, label_batch)
            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice, loss_dice_1 = muti_loss(False, dice_loss, outputs, d1, d2, d3, d4, label_batch)
            # loss_ssim_1 = 1 - ssim_loss(outputs, label_batch)
            # loss_ssim_2 = 1 - ssim_loss(d4, label_batch)
            # loss_ssim = 0.6 * loss_ssim_1 + 0.4 * loss_ssim_2
            # loss = 0.4 * loss_ce  + 0.6 * loss_dice 
            loss_all = 0.4 * loss_ce_1 + 0.6 * loss_dice_1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss_all, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice_1, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            logging.info('iteration %d : loss_dice : %f, loss_ce : %f, loss_muti : %f, loss : %f' % (iter_num, loss_dice_1.item(), loss_ce_1.item(), loss.item(), loss_all.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 25  # int(max_epoch/6)
        if epoch_num > 50 and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_prostate(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size 
    max_iterations = args.max_iterations
    db_train = Prostate_dataset(patients_path=args.root_path,NVB=False,DVC=False)
    # db_train = Prostate_dataset(patients_path=args.root_path)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    ssim_loss = SSIM(window_size=11,size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            gt_batch = label_batch.unsqueeze(1)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # result, outputs = model(image_batch)
            outputs, d1, d2, d3, d4 = model(image_batch)
            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_ce_2 = ce_loss(outputs, label_batch[:].long())
            # loss_dice_2 = dice_loss(outputs, label_batch, softmax=True)
            loss_ce, loss_ce_1 = muti_loss(True, ce_loss, outputs, d1, d2, d3, d4, label_batch)
            loss_dice, loss_dice_1 = muti_loss(False, dice_loss, outputs, d1, d2, d3, d4, label_batch)
            loss_ssim_1 = 1 - ssim_loss(outputs, label_batch)
            loss_ssim_2 = 1 - ssim_loss(d4, label_batch)
            loss_ssim = 0.6 * loss_ssim_1 + 0.4 * loss_ssim_2
            loss = 0.5 * ( 0.4 * loss_ce  + 0.6 * loss_dice ) + 0.5 * loss_ssim
            loss_all = 0.4 * loss_ce_1 + 0.6 * loss_dice_1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss_all, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce_1, iter_num)
            # writer.add_scalar('info/loss_muti', loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_muti: %f, loss_ssim: %f' % (iter_num, loss_all.item(), loss_ce_1.item(), loss.item(), loss_ssim.item()))
            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))


            if iter_num % 20 == 0:
                # print(image_batch.shape)
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
            
            del outputs, d1, d2, d3, d4, loss, loss_ce, loss_dice

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > 98 and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"