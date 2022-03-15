import torch
import torch.nn as nn
import torch.nn.functional as F

from data_process import *


class MyModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, dropout):
        super(MyModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.dropout = dropout
        self.gru = nn.GRU(self.vocab_size, self.num_hiddens)
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)

    def forward(self, inputs, h0):
        inputs = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)
        gru_out, hn = self.gru(inputs, h0)
        linear_out = self.linear(gru_out)
        linear_out = linear_out.reshape(-1, self.vocab_size)
        return linear_out, hn

    def begin_state(self, batch_size, device):
        return torch.zeros((1, batch_size, self.num_hiddens), device=device)


if __name__ == '__main__':
    corpus = read_data('data/input.txt')
    vocab = Vocab(corpus)
    corpus = [vocab[c] for c in corpus]

    device = try_gpu()
    vocab_size = len(vocab)
    batch_size = 32
    num_steps = 10
    random_train = True
    num_hiddens = 128
    dropout = 0     # 层数为1，无效参数
    num_epochs = 200
    lr = 1
    weight_decay = 0
    if random_train:
        train_dataset = seq_data_iter_random
    else:
        train_dataset = seq_data_iter_sequential

    model = MyModel(vocab_size, num_hiddens, dropout=dropout)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        state = None
        total_loss = []
        for x, y in train_dataset(corpus, batch_size, num_steps):
            if state is None or random_train:
                state = model.begin_state(batch_size=x.shape[0], device=device)
            else:
                # 减少计算量，状态保留，梯度不保留
                state.detach_()
            y = y.T.reshape(-1)
            x, y = x.to(device), y.to(device)
            y_hat, state = model(x, state)
            l = loss(y_hat, y.long()).mean()
            l.backward()
            # loss 求过平均，所以此处batch_size为1
            opt.step()
            opt.zero_grad()
            total_loss.append(l.item())
        print(f'{epoch} mean loss {sum(total_loss) / len(total_loss)}')

    while True:
        prefix = input('请输入开头，如：我在哭泣，：')
        num_preds = input('请输入生成长度，默认200，：')
        try:
            num_preds = int(num_preds)
        except:
            num_preds = 200
        state = model.begin_state(batch_size=1, device=device)
        outputs = [vocab[prefix[0]]]
        # 预热期
        get_input = lambda : torch.tensor([outputs[-1]], device=device).reshape((1, 1))
        for y in prefix[1:]:
            _, state = model(get_input(), state)
            outputs.append(vocab[y])
        for _ in range(num_preds):
            y, state = model(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        outputs_words = ''.join([vocab.idx_to_token[i] for i in outputs])
        print('*' * 20 + '模型输出' + '*' * 20)
        print(outputs_words)
        print('*' * 50)