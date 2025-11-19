
import shutil
import wandb
import os


def save_files(args):
    print('copying files ...')

    files_path = os.path.join(args.logdir, 'files/')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    shutil.copy(f'./train.py', f'{files_path}train.py')
    shutil.copy(f'./main_mri.py', f'{files_path}main_mri.py')
    shutil.copy(f'./dataloader.py', f'{files_path}dataloader.py')
    shutil.copy(f'./train_one_epoch_cox.py', f'{files_path}train_one_epoch_cox.py')

    shutil.copy(f'./model_init.py', f'{files_path}model_init.py')
    # shutil.copy(f'./models/mymodel_3d.py', f'{files_path}mymodel_3d.py')
    shutil.copy(f'./util/utils.py', f'{files_path}utils.py')
    shutil.copy(f'./util/losses.py', f'{files_path}losses.py')
    shutil.copy(f'./models/resnet_pretrained1.py', f'{files_path}resnet_pretrained1.py')
    shutil.copy(f'./configs.py', f'{files_path}config.py')
    print('copying files successfully!')


def save_files_seq(args):
    print('copying files ...')
    
    files_path = os.path.join(args.logdir, 'files/')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    shutil.copy(f'./main_all.py', f'{files_path}main_all.py')
    shutil.copy(f'./train.py', f'{files_path}train.py')
    shutil.copy(f'./dataloader_ensemble.py', f'{files_path}dataloader_ensemble.py')
    shutil.copy(f'./train_one_epoch_ensemble.py', f'{files_path}train_one_epoch_ensemble.py')
    shutil.copy(f'./model_init_ensemble.py', f'{files_path}model_init_ensemble.py')
    
    shutil.copy(f'./models/seqmodel.py', f'{files_path}seqmodel.py')
    shutil.copy(f'./util/utils.py', f'{files_path}utils.py')
    shutil.copy(f'./util/losses.py', f'{files_path}losses.py')
    print('copying files successfully!')
