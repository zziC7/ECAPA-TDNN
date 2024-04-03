'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''

import os, numpy, torch, itertools, random, math
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F

def init_args(args):
    args.score_save_path = os.path.join(args.save_path, 'score.txt')
    args.model_save_path = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok=True)
    return args

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
	
	fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
	fnr = 1 - tpr
	tunedThreshold = [];
	if target_fr:
		for tfr in target_fr:
			idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
			tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	for tfa in target_fa:
		idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
		tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
	eer  = max(fpr[idxE],fnr[idxE])*100
	
	return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res

def GenerateList(root_dir):
    # 读取所有音频文件的路径和标签
    all_files_labels = []
    for speaker in os.listdir(root_dir):
        speaker_dir = os.path.join(root_dir, speaker)
        if os.path.isdir(speaker_dir):
            files = [os.path.join(speaker, f) for f in os.listdir(speaker_dir) if f.endswith('.flac')]
            for file in files:
                all_files_labels.append((file, speaker))

    # 打乱并分割数据集
    random.shuffle(all_files_labels)
    split_point = int(len(all_files_labels) * 0.8)
    train_files_labels = all_files_labels[:split_point]
    test_files_labels = all_files_labels[split_point:]

    # 生成测试集的音频对
    # 计算测试集中可以生成的样本对的最大数量
    max_pairs = 10000
    positive_pairs = []
    negative_pairs = []
    speakers_in_test = set(label for _, label in test_files_labels)

    # 生成正样本对
    for speaker in speakers_in_test:
        speaker_files_labels = [(file, label) for file, label in test_files_labels if label == speaker]
        for (file1, _), (file2, _) in itertools.combinations(speaker_files_labels, 2):
            positive_pairs.append(f"1 {file1} {file2}")
            if len(positive_pairs) >= max_pairs:
                break
        if len(positive_pairs) >= max_pairs:
            break

    # 生成负样本对
    while len(negative_pairs) < len(positive_pairs):
        file1, speaker1 = random.choice(test_files_labels)
        file2, speaker2 = random.choice(test_files_labels)
        if speaker1 != speaker2:
            negative_pairs.append(f"0 {file1} {file2}")

    # 合并正负样本对并打乱
    test_pairs = positive_pairs + negative_pairs
    random.shuffle(test_pairs)

    parent_directory = os.path.dirname(root_dir)
    # 写入训练集文件
    train_file_path = os.path.join(parent_directory, "train_files.txt")
    with open(train_file_path, 'w') as f:
        for file, label in train_files_labels:
            f.write(f"{label} {file} \n")

    # 写入测试集文件
    test_file_path = os.path.join(parent_directory, "test_pairs.txt")
    with open(test_file_path, 'w') as f:
        for pair in test_pairs:
            f.write(pair + '\n')

    print(f"训练文件已保存到 {train_file_path}")
    print(f"测试对已保存到 {test_file_path}")

if __name__ == '__main__':
    root_dir = "/root/autodl-tmp/CN-Celeb2_flac/data"
    GenerateList(root_dir)
