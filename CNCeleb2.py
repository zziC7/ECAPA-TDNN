import glob
import os
import random
from scipy import signal
import numpy
import soundfile
import torch
import torchaudio
from torch.utils.data import Dataset

class CNCeleb2Dataset(Dataset):
    def __init__(self, train_list, root_dir, num_frames, musan_path, rir_path, **kwargs):
        """
        初始化数据集
        :param train_list: 包含训练集信息的文本文件路径
        :param root_dir: 数据集的根目录
        :param num_frames: 指定的帧数
        :param musan_path: musan数据集的目录 用于数据增强
        :param rir_path: rir混响数据集的目录 用于数据增强
        """
        self.root_dir = root_dir
        self.num_frames = num_frames

        # Load and configure augmentation files
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        # 初始化字典来存储不同类型的音频文件
        self.noiselist = {'noise': [], 'speech': [], 'music': []}
        # 读取所有音频文件的路径
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        # 遍历所有文件，并根据它们的类型添加到相应的列表中
        for file in augment_files:
            # 从文件路径中获取音频类型
            noise_type = file.split('/')[-3]
            if noise_type in self.noiselist:
                self.noiselist[noise_type].append(file)

        # 读取混响音频文件用于数据增强
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(root_dir, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个样本
        :param idx: 样本的索引
        """
        # Read the utterance and randomly select the segment
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[idx])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Data Augmentation
        augtype = random.randint(0, 5)
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return torch.FloatTensor(audio[0]), self.data_label[idx]

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


if __name__ == '__main__':
    train_list = '/root/autodl-tmp/CN-Celeb2_flac/train_files.txt'
    root_dir = '/root/autodl-tmp/CN-Celeb2_flac/data'
    musan_path = "/root/autodl-tmp/musan"
    rir_path = "/root/autodl-tmp/RIRS_NOISES/simulated_rirs"
    dataset = CNCeleb2Dataset(train_list, root_dir, 200, musan_path, rir_path)
    audio_sample, label = dataset.__getitem__(0)  # 获取第一个样本
    # 打印形状
    print(f"Audio sample shape: {audio_sample.shape}")
    print(f"Label shape: {label.shape if isinstance(label, torch.Tensor) else label}")
    print(f"Label: {label}")