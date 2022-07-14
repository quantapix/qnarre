import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncLayer
from io import open

from ..utils import Corpus

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="./model.pt")
parser.add_argument("--outf", type=str, default="generated.txt")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--words", type=int, default="1000")
parser.add_argument("--bptt", type=int, default=35)
parser.add_argument("--clip", type=float, default=0.25)
parser.add_argument("--data", type=str, default="./data/wikitext-2")
parser.add_argument("--drop", type=float, default=0.2)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--emsize", type=int, default=200)
parser.add_argument("--log-interval", type=int, default=200, metavar="N")
parser.add_argument("--nhead", type=int, default=2)
parser.add_argument("--nhid", type=int, default=200)
parser.add_argument("--nlayers", type=int, default=2)
parser.add_argument("--save", type=str, default="model.pt")
parser.add_argument("--tied", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


corpus = Corpus(args.data)


class RNN(qc.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, drop=0.5, tie_weights=False):
        super().__init__()
        self.ntoken = ntoken
        self.drop = qc.Dropout(drop)
        self.encoder = qc.Embed(ntoken, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, drop=drop)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, drop=drop)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError("When using the tied flag, nhid must be equal to emsize")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class PositionalEncoding(qc.Module):
    def __init__(self, d_hidden, drop=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.drop = qc.Dropout(p=drop)

        pe = torch.zeros(max_len, d_hidden)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hidden, 2).float() * (-math.log(10000.0) / d_hidden))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.drop(x)


class Transformer(qc.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, drop=0.5):
        super().__init__()
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, drop)
        n_enc_lays = TransformerEncLayer(ninp, nhead, nhid, drop)
        self.transformer_encoder = TransformerEncoder(n_enc_lays, nlayers)
        self.encoder = qc.Embed(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


train_data = batchify(corpus.train, args.train_batch_size)
eval_data = batchify(corpus.eval, args.eval_batch_size)
test_data = batchify(corpus.test, args.eval_batch_size)

ntokens = len(corpus.dictionary)
if args.model_name == "Transformer":
    model = Transformer(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.drop).to(
        device
    )
else:
    model = RNN(
        args.model_name, ntokens, args.emsize, args.nhid, args.nlayers, args.drop, args.tied
    ).to(device)

criterion = nn.NLLLoss()


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source):
    model.eval()
    total_loss = 0.0
    ntokens = len(corpus.dictionary)
    if args.model_name != "Transformer":
        hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model_name == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model_name != "Transformer":
        hidden = model.init_hidden(args.train_batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        if args.model_name == "Transformer":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.bptt,
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


lr = args.lr
best_val_loss = None

try:
    for epoch in range(1, args.train_epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(eval_data)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        print("-" * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, "wb") as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")

with open(args.save, "rb") as f:
    model = torch.load(f)
    if args.model_name in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
        model.rnn.flatten_parameters()

test_loss = evaluate(test_data)
print("=" * 89)
print(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, math.exp(test_loss)
    )
)
print("=" * 89)


if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

with open(args.checkpoint, "rb") as f:
    model = torch.load(f, map_location=device)
model.eval()

corpus = Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, "model_type") and model.model_type == "Transformer"
if not is_transformer_model:
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, "w") as outf:
    with torch.no_grad():
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ("\n" if i % 20 == 19 else " "))

            if i % args.log_interval == 0:
                print("| Generated {}/{} words".format(i, args.words))


"""
python main.py --cuda --train_epochs 6
python main.py --cuda --train_epochs 6 --tied
python main.py --cuda --tied
python main.py --cuda --train_epochs 6 --model Transformer --lr 5   
python generate.py
python generate.py --cuda --model Transformer
"""
