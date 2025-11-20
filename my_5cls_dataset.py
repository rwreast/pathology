import argparse
import os
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
from dataset import (
    get_train_test_ds_MultiCenter_region_trainwithTCGA,
    TumorRegion_PathologyType_Feat,
)
import datetime
import utliz
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import task_pathocls_prediction  # 五分类


def filter_dataset_by_label(dataset, label_index, keep_value):
    """
    根据指定的标签值过滤数据集。

    Args:
        dataset (list of lists): 要过滤的数据集，格式类似 InternalTrain。
                                 每个子列表代表一种数据（如特征、标签等）。
        label_index (int): 用于过滤的标签所在的子列表的索引。
                            0:data 1:label
                            2:name_slide 3:name_patch
                            4:patho_label 5:nuclearLevel_label
                            6:prognosis_label 7:isup_label
                            8:size_label 9:tnm_label
                            10:necrosis_label 11:yuHouFenZu_label
                            12:zhuyuanhao
        keep_value: 要保留的标签值 4-1是乳头细胞。

    Returns:
        list of lists: 过滤后的数据集。
    """

    if not dataset or not isinstance(dataset, list) or not isinstance(dataset[0], list):
        print("Warning: Input dataset is not in the expected format (list of lists).")
        return dataset

    labels_to_filter = np.array(dataset[label_index])
    indices_to_keep = np.where(labels_to_filter == keep_value)[0]

    # 使用索引过滤 dataset 中的每个列表
    filtered_dataset = [
        np.array(data_list, dtype=object)[indices_to_keep].tolist()
        for data_list in dataset
    ]

    print(
        f"Filtered dataset. Kept {len(indices_to_keep)} items where label at index {label_index} is {keep_value}."
    )
    return filtered_dataset


def datasetprogress(args):

    # 定义缓存目录和文件名
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = os.path.join(cache_dir, f"5CLS_dataset_cache_downsample_{args.downsample}.pth")

    # 检查缓存文件是否存在
    if os.path.exists(cache_filename):
        print(f"--- Loading dataset from cache: {cache_filename} ---")
        InternalTrain, InternalTest, ExternalTest, huadong = torch.load(cache_filename)
        print("--- Dataset loaded from cache successfully. ---")
    else:
        print(f"--- Cache file not found. Loading dataset from source... ---")
        InternalTrain, InternalTest, ExternalTest, huadong = (
            get_train_test_ds_MultiCenter_region_trainwithTCGA(downsample=args.downsample)
        )
        print(f"--- Saving dataset to cache file: {cache_filename} ---")
        # 使用 torch.save 保存数据
        torch.save((InternalTrain, InternalTest, ExternalTest, huadong), cache_filename)
        print("--- Dataset cached successfully. ---")

    # 根据 --filter 参数决定是否执行过滤
    if args.filter:
        print("Filtering enabled: Keeping only patho_label == 1 (乳头状癌).")
        InternalTrain = filter_dataset_by_label(
            dataset=InternalTrain, label_index=4, keep_value=1
        )
        InternalTest = filter_dataset_by_label(
            dataset=InternalTest, label_index=4, keep_value=1
        )
        ExternalTest = filter_dataset_by_label(
            dataset=ExternalTest, label_index=4, keep_value=1
        )
        huadong = filter_dataset_by_label(dataset=huadong, label_index=4, keep_value=1)

    print(
        "Num of slides in InternalTrain/InternalTest/ExternalTest: {} / {} / {} / {}".format(
            len(InternalTrain[0]),
            len(InternalTest[0]),
            len(ExternalTest[0]),
            len(huadong[0]),
        )
    )
    # 加载为类
    train_ds = TumorRegion_PathologyType_Feat(InternalTrain, return_bag=False)
    InternalVal_ds = TumorRegion_PathologyType_Feat(InternalTest, return_bag=False)
    ExternalVal_ds = TumorRegion_PathologyType_Feat(ExternalTest, return_bag=False)
    ExternalHDVal_ds = TumorRegion_PathologyType_Feat(huadong, return_bag=False)

    print(
        "Num of patches in InternalTrain/InternalTest/ExternalTest: {} / {} / {} / {}".format(
            len(train_ds),
            len(InternalVal_ds),
            len(ExternalVal_ds),
            len(ExternalHDVal_ds),
        )
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )
    InternalVal_loader = torch.utils.data.DataLoader(
        InternalVal_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )
    ExternalVal_loader = torch.utils.data.DataLoader(
        ExternalVal_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )
    ExternalHDVal_loader = torch.utils.data.DataLoader(
        ExternalHDVal_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )

    print("[Data] {} training samples".format(len(train_ds)))
    print("[Data] {} Internal evaluating samples".format(len(InternalVal_ds)))
    print("[Data] {} External evaluating samples".format(len(ExternalVal_ds)))
    print("[Data] {} HuaDong evaluating samples".format(len(ExternalHDVal_ds)))

    return train_loader, InternalVal_loader, ExternalVal_loader, ExternalHDVal_loader


if __name__ == "__main__":

    # 创建一个独立的解析器来处理 --filter 参数
    filter_parser = argparse.ArgumentParser(
        description="Filter control for my_5cls.py", add_help=False
    )
    filter_parser.add_argument(
        "--filter",
        action="store_true",
        help="Enable filtering to keep only patho_label == 1.",
    )
    # parse_known_args() 只解析它认识的参数，忽略其他参数
    filter_args, _ = filter_parser.parse_known_args()

    # 从 task_pathocls_prediction 获取其他所有参数
    args = task_pathocls_prediction.get_parser()

    # 如果 resume 参数是 task_pathocls_prediction.py 中的默认值，则将其重置为空字符串
    # 不用多中心训练过的
    default_resume_path = "/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/results_ckpt_patho/InternalTest_Epoch3_AUC_0.9902"
    if args.resume == default_resume_path:
        args.resume = ""

    train_loader, InternalVal_loader, ExternalVal_loader, ExternalHDVal_loader = (
        datasetprogress(args.filter)
    )

    dev = "cuda:0"
    model = task_pathocls_prediction.prediction_head()
    model.to(dev)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    start_epoch = 0
    if args.resume != "":
        model, start_epoch, threshold_dict = task_pathocls_prediction.load_ckpt(
            model, args.resume
        )
        start_epoch = start_epoch + 1
        print("Load from {}".format(args.resume))

    name = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    ) + "_Bs{}_lr{}_downsample{}".format(args.batch_size, args.lr, args.downsample)
    writer = SummaryWriter("./runs_RegionPathology_feat/%s" % name)

    threshold_dict = None
    for epoch in range(start_epoch, args.epochs):
        thresholds_dict = task_pathocls_prediction.train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            dev=dev,
            writer=writer,
        )
        if threshold_dict != None:
            task_pathocls_prediction.eval(
                epoch=epoch,
                model=model,
                loader=[InternalVal_loader, ExternalVal_loader, ExternalHDVal_loader],
                dev=dev,
                writer=writer,
                threshold_dict=threshold_dict,
            )
