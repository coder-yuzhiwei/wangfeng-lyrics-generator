
import torch.nn.functional as F
import torch.nn as nn

from data_process import *


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    w_xz, w_hz, b_z = three()
    w_xr, w_hr, b_r = three()
    w_xh, w_hh, b_h = three()
    w_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device)


def gru(inputs, state, params):
    w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q = params
    h = state
    outputs = []
    for x in inputs:
        z = torch.sigmoid(x @ w_xz + h @ w_hz + b_z)
        r = torch.sigmoid(x @ w_xr + h @ w_hr + b_r)
        h_hat = torch.tanh(x @ w_xh + (r * h) @ w_hh + b_h)
        h = z * h + (1 - z) * h_hat
        q = h @ w_hq + b_q
        outputs.append(q)
    return torch.cat(outputs, dim=0), h


class MyModel():
    def __init__(self, vocab_size, num_hiddens, get_params, init_state, forward_fn, device):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, x, state):
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device=device)


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def grad_clipping(model, theta):
    params = model.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


if __name__ == '__main__':
    corpus = read_data('data/input.txt')
    # vocab_size 1758
    vocab = Vocab(corpus)
    corpus = [vocab[word] for word in corpus]

    device = try_gpu()
    vocab_size = len(vocab)
    batch_size = 128
    num_steps = 10
    random_train = True
    num_hiddens = 256
    num_epochs = 200
    lr = 4

    model = MyModel(vocab_size, num_hiddens, get_params, init_gru_state, gru, device)
    loss = nn.CrossEntropyLoss()

    if random_train:
        train_dataset = seq_data_iter_random
    else:
        train_dataset = seq_data_iter_sequential

    # 训练
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
            grad_clipping(model, 1)
            # loss 求过平均，所以此处batch_size为1
            sgd(model.params, lr, 1)
            total_loss.append(l.item())
        print(f'{epoch} mean loss {sum(total_loss) / len(total_loss)}')

    # 预测
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







