'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from tools import *
from ECAPAModel import ECAPAModel
from CNCeleb2 import *

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=400,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=15,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path

## modify these paths on your own device
parser.add_argument('--train_list', type=str,   default="/root/autodl-tmp/CN-Celeb2_flac/train_files.txt",     help='The path of the training list')
parser.add_argument('--root_dir', type=str,   default="/root/autodl-tmp/CN-Celeb2_flac/data",                    help='The path of the training data')
parser.add_argument('--eval_list',  type=str,   default="/root/autodl-tmp/CN-Celeb2_flac/test_pairs.txt", help='The path of the evaluation list')
parser.add_argument('--eval_path',  type=str,   default="/root/autodl-tmp/CN-Celeb2_flac/data",                    help='The path of the evaluation data')
parser.add_argument('--musan_path', type=str,   default="/root/autodl-tmp/musan",                    help='The path to the MUSAN set')
parser.add_argument('--rir_path',   type=str,   default="/root/autodl-tmp/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set');
## modify these paths on your own device
parser.add_argument('--save_path',  type=str,   default="exps/exp1",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=1996,   help='Number of speakers')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

# ## Define the data loader
dataset = CNCeleb2Dataset(**vars(args))
trainLoader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

EERs = []
score_file = open(args.score_save_path, "a+")
writer = SummaryWriter(args.save_path)

while(1):
	## Training for one epoch
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

	# 将训练损失、学习率和准确率记录到TensorBoard
	writer.add_scalar('Training/Loss', loss, epoch)
	writer.add_scalar('Training/Learning Rate', lr, epoch)
	writer.add_scalar('Training/Accuracy', acc, epoch)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
		score_file.flush()

		# 将EER记录到TensorBoard
		writer.add_scalar('Validation/EER', EERs[-1], epoch)

	if epoch >= args.max_epoch:
		quit()

	epoch += 1

writer.close()