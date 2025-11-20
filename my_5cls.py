import argparse
import torch
import torch.optim
from tensorboardX import SummaryWriter
import datetime
import task_pathocls_prediction  # 五分类
import my_5cls_dataset  # 五分类数据处理带过滤


if __name__ == "__main__":

    ###############命令行读取###############

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

    ###############数据集加载###############
    train_loader, InternalVal_loader, ExternalVal_loader, ExternalHDVal_loader = (
        my_5cls_dataset.datasetprogress(args)
    )
    ###############模型加载###############
    dev = "cuda:0"
    model = task_pathocls_prediction.prediction_head()
    model.to(dev)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    ###############checkpoints加载###############
    start_epoch = 0
    if args.resume != "":
        model, start_epoch, threshold_dict = task_pathocls_prediction.load_ckpt(
            model, args.resume
        )
        start_epoch = start_epoch + 1
        print("Load from {}".format(args.resume))
    ###############结果保存###############
    name = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    ) + "_Bs{}_lr{}_downsample{}".format(args.batch_size, args.lr, args.downsample)
    writer = SummaryWriter("./runs_RegionPathology_feat/%s" % name)

    threshold_dict = None
    ###############正式训练###############
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
