# dataset.py - 修改后的完整代码

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
from my_tokenizer import PianoRollTokenizer


def encode_measure_tokens(
    measure: np.ndarray,
    tokenizer: PianoRollTokenizer,
) -> Tuple[List[int], List[int]]:
    """
    将单个小节编码为高、低声部的压缩token序列。

    Args:
        measure: (4, 88, t)，前两通道为高声部，后两通道为低声部
        tokenizer: PianoRollTokenizer实例

    Returns:
        part0_tokens: 高声部压缩后的token序列
        part1_tokens: 低声部压缩后的token序列
    """
    part0 = measure[:2]
    part1 = measure[2:]

    tokens_0 = tokenizer.image_to_patch_tokens(part0, strict_mode=True)
    compressed_tokens_0 = tokenizer.compress_tokens(tokens_0, end_marker=tokenizer.end_marker_part0)

    tokens_1 = tokenizer.image_to_patch_tokens(part1, strict_mode=True)
    compressed_tokens_1 = tokenizer.compress_tokens(tokens_1, end_marker=tokenizer.end_marker_part1)

    return compressed_tokens_0.tolist(), compressed_tokens_1.tolist()


def encode_bpm(bpm):
        if bpm is None:
            return 3  # UNK token
        bpm = int(bpm)
        if bpm < 90:
            return 0  # 慢速
        elif bpm <= 200:
            return 1  # 中速
        else:
            return 2  # 快速
        


class PianoDataset(Dataset):
    """支持长度感知的数据集"""

    def __init__(self, data_dir, config, cache_lengths=True, mode='train',
                 test_split_ratio=0.05, random_seed=42):
        """
        Args:
            data_dir: 数据目录
            config: 模型配置
            cache_lengths: 是否使用长度缓存
            mode: 'train' 或 'test'，决定使用训练集还是测试集
            test_split_ratio: 测试集划分比例（0-1之间）
            random_seed: 随机种子，用于可重复的数据集划分
        """
        self.root_dir = data_dir
        self.patch_h = config.patch_h
        self.patch_w = config.patch_w
        self.max_seq_len = config.train_cutoff_len
        self.pad_token = config.pad_token_id
        self.bos_token = config.bos_token_id
        self.eos_token = config.eos_token_id
        self.bar_token = config.bar_token_id
        self.time_sig_offset_id = config.time_sig_offset_id
        self.bpm_offset_id = config.bpm_offset_id
        self.mode = mode
        self.test_split_ratio = test_split_ratio
        self.random_seed = random_seed
        self.measure_package_size = 40
        self.max_measures = config.max_measures
        self.measure_eos_token = config.eos_token_id
        # 创建tokenizer实例
        self.tokenizer = PianoRollTokenizer(
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            marker_offset=81,
            measures_length=88,
            end_marker_part0=170,
            end_marker_part1=171,
            empty_marker=169,
            img_h=88
        )
        self.part1_end_marker = self.tokenizer.end_marker_part1
        self.part1_empty_marker = self.tokenizer.empty_marker

        self.data_files = [f for f in os.listdir(self.root_dir) if f.endswith('.npz')]
        print(f"找到 {len(self.data_files)} 个有效的npz文件")

        # 预计算长度信息
        cache_file = os.path.join(data_dir, '.lengths_cache_measure.pkl')

        if cache_lengths and os.path.exists(cache_file):
            print("加载长度缓存...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # 验证patch参数是否匹配
            if (cache_data['patch_h'] != self.patch_h or 
                cache_data['patch_w'] != self.patch_w):
                raise ValueError(
                    f"缓存的patch参数({cache_data['patch_h']}x{cache_data['patch_w']}) "
                    f"与配置({self.patch_h}x{self.patch_w})不匹配，请重新运行precompute_lengths.py"
                )
            
            self.data_files = cache_data['data_files']
            self.file_lengths = cache_data['lengths']
            self.sorted_indices = cache_data['sorted_indices']
            
            print(f"加载 {len(self.data_files)} 个文件的长度信息")
            
        elif cache_lengths:
            raise FileNotFoundError(
                f"长度缓存不存在: {cache_file}\n"
                f"请先运行: python precompute_lengths.py"
            )
        else:
            # 不使用缓存，传统方式
            self.file_lengths = None
            self.sorted_indices = None
            print(f"找到 {len(self.data_files)} 个文件（未使用长度缓存）")

        # 划分训练集和测试集
        self._split_train_test()

    def _split_train_test(self):
        """根据mode参数划分训练集和测试集"""
        total_files = len(self.data_files)

        # 设置随机种子以确保可重复性
        np.random.seed(self.random_seed)

        # 创建索引数组并打乱
        indices = np.arange(total_files)
        np.random.shuffle(indices)

        # 计算测试集大小
        test_size = int(total_files * self.test_split_ratio)
        train_size = total_files - test_size

        if self.mode == 'train':
            # 使用前train_size个样本作为训练集
            selected_indices = indices[:train_size]
            print(f"使用训练集: {len(selected_indices)} 个文件 ({train_size}/{total_files})")
        elif self.mode == 'test':
            # 使用后test_size个样本作为测试集
            selected_indices = indices[train_size:]
            print(f"使用测试集: {len(selected_indices)} 个文件 ({test_size}/{total_files})")
        else:
            raise ValueError(f"mode必须是'train'或'test'，当前为: {self.mode}")

        # 更新data_files和相关索引
        self.data_files = [self.data_files[i] for i in selected_indices]

        # 如果使用了长度缓存，也需要更新相关信息
        if self.file_lengths is not None:
            self.file_lengths = [self.file_lengths[i] for i in selected_indices]

            # 重新创建sorted_indices（在新的子集中的排序）
            self.sorted_indices = sorted(
                range(len(self.file_lengths)),
                key=lambda i: self.file_lengths[i]
            )

    def __len__(self):
        return len(self.data_files)
    
    def _find_next_bar_token(self, sequence, start_pos, target_len):
        """
        从指定位置开始查找下一个bar_token，并返回从该位置开始的target_len长度序列
        
        Args:
            sequence: 原始token序列
            start_pos: 开始搜索的位置
            target_len: 目标序列长度
        
        Returns:
            从找到的bar_token开始的序列切片
        """
        search_end = min(len(sequence), start_pos + target_len)
        
        for i in range(start_pos, search_end):
            if sequence[i] == self.bar_token:
                return sequence[i:i + target_len]
        
        # 找不到bar_token时，从原始位置截取（降级策略）
        return sequence[start_pos:start_pos + target_len]

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.data_files[idx])

        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()
        num_measures = metadata['num_measures']
        shift = 0
        
        if np.random.random() < 0.7:
            shift = np.random.randint(-5, 6)

        measure_inputs: List[List[int]] = []
        measure_labels: List[List[int]] = []
        for i in range(num_measures):
            if self.max_measures is not None and len(measure_inputs) >= self.max_measures:
                break

            measure = save_dict[f'measure_{i}']

            if shift != 0:
                measure = np.roll(measure, shift, axis=1)
                if shift > 0:
                    measure[:, :shift, :] = 0
                else:
                    measure[:, shift:, :] = 0

            part0_tokens, part1_tokens = encode_measure_tokens(measure, tokenizer=self.tokenizer)
            
            part0_tokens = part0_tokens + [self.measure_eos_token]
            truncated_tokens0 = part0_tokens[:self.measure_package_size]
            
            pad_len = self.measure_package_size - len(truncated_tokens0)

            if pad_len > 0:
                input_row = truncated_tokens0 + [self.pad_token] * pad_len
                label_row = truncated_tokens0 + [-100] * pad_len
            else:
                input_row = list(truncated_tokens0)
                label_row = list(truncated_tokens0)

            measure_inputs.append(input_row)
            measure_labels.append(label_row)


            part1_tokens = part1_tokens + [self.measure_eos_token]
            truncated_tokens1 = part1_tokens[:self.measure_package_size]

            pad_len = self.measure_package_size - len(truncated_tokens1)

            if pad_len > 0:
                input_row = truncated_tokens1 + [self.pad_token] * pad_len
                label_row = truncated_tokens1 + [-100] * pad_len
            else:
                input_row = list(truncated_tokens1)
                label_row = list(truncated_tokens1)

            measure_inputs.append(input_row)
            measure_labels.append(label_row)

        
        compressed_input = torch.tensor(measure_inputs, dtype=torch.long)
        compressed_labels = torch.tensor(measure_labels, dtype=torch.long)
        return {
            'input_ids': compressed_input,
            'labels': compressed_labels,
        }


class BucketBatchSampler(Sampler):
    """长度感知的批采样器"""
    
    def __init__(self, dataset, batch_size=16, bucket_size=100, shuffle=True):
        """
        Args:
            dataset: PianoDataset实例
            batch_size: 实际训练的batch大小
            bucket_size: 每个长度bucket的大小
            shuffle: 是否随机化
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        
        if dataset.sorted_indices is None:
            raise ValueError("Dataset需要启用cache_lengths=True")
        
        self._create_buckets()
    
    def _create_buckets(self):
        """将相近长度的样本分组到buckets中"""
        self.buckets = []
        sorted_indices = self.dataset.sorted_indices
        
        # 将排序后的索引分割成buckets
        for i in range(0, len(sorted_indices), self.bucket_size):
            bucket = sorted_indices[i:i + self.bucket_size]
            self.buckets.append(bucket)
        
        print(f"创建了 {len(self.buckets)} 个长度buckets")
    
    def __iter__(self):
        """生成batch索引"""
        # 随机打乱buckets的顺序
        if self.shuffle:
            bucket_order = np.random.permutation(len(self.buckets))
        else:
            bucket_order = range(len(self.buckets))
        
        for bucket_idx in bucket_order:
            bucket = self.buckets[bucket_idx].copy()
            
            # 在bucket内部随机打乱
            if self.shuffle:
                np.random.shuffle(bucket)
            
            # 从bucket中生成batches
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) > 0:  # 确保不是空batch
                    yield batch
    
    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets) // self.batch_size


@dataclass
class DataCollatorForVariableLengthLM:
    """数据整理器，支持动态padding"""
    
    def __init__(self, config):
        self.pad_token_id = config.pad_token_id
        self.max_length = config.max_measures
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        measure_size = features[0]["input_ids"].shape[-1]
        max_measures = self.max_length
        max_measures_in_batch = min(
            max(feature["input_ids"].shape[0] for feature in features),
            max_measures
        )

        batch_input_ids = []
        batch_labels = []
        batch_attention = []

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]

            input_ids = input_ids[:max_measures_in_batch]
            labels = labels[:max_measures_in_batch]

            current_measures = input_ids.shape[0]

            if current_measures < max_measures_in_batch:
                pad_measures = max_measures_in_batch - current_measures
                pad_inputs = torch.full(
                    (pad_measures, measure_size),
                    self.pad_token_id,
                    dtype=torch.long
                )
                pad_labels = torch.full(
                    (pad_measures, measure_size),
                    -100,
                    dtype=torch.long
                )
                input_ids = torch.cat([input_ids, pad_inputs], dim=0)
                labels = torch.cat([labels, pad_labels], dim=0)

            attention_mask = (input_ids != self.pad_token_id).long()

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention.append(attention_mask)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention),
        }
