import random
import collections

import torch


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def read_data(file_path):
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # 去掉空行 和 -
            if line.strip() and not line.startswith('-'):
                # 补上'\n' 使模型能够学会换行
                corpus += list(line + '\n')
    return corpus


def count_corpus(tokens):
    """统计词频"""
    if len(tokens) != 0 and isinstance(tokens[0], list):
        # 展平
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(index) for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


# 采样生成训练数据
# 随机采样
def seq_data_iter_random(corpus, batch_size, num_steps):
    # 随机语料的起始位置，使能够充分利用不能整除的数据
    corpus = corpus[random.randint(0, num_steps-1): ]
    num_subseqs = (len(corpus)-1) // num_steps
    start_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(start_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batchs = num_subseqs // batch_size
    for i in range(0, batch_size * num_batchs, batch_size):
        start_indices_per_batch = start_indices[i: i+batch_size]
        x = [data(j) for j in start_indices_per_batch]
        y = [data(j+1) for j in start_indices_per_batch]
        yield torch.tensor(x), torch.tensor(y)


# 顺序采样
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps-1):]
    num_tokens = ((len(corpus) - 1) // batch_size) * batch_size
    xs = torch.tensor(corpus[0: num_tokens])
    ys = torch.tensor(corpus[1: 1+num_tokens])
    xs, ys = xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    num_batchs = xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batchs, num_steps):
        x = xs[:, i: i+num_steps]
        y = ys[:, i: i+num_steps]
        yield x, y

