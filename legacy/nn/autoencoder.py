# Heavily brought from Sean Robertson's pytorch tutorial on seq2seq

from __future__ import unicode_literals, print_function, division
from io import open
import random
import time
from legacy import logic_math
import os.path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 100


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def indicesFromExpression(self, string_expression):
        indices = [self.word2index[word] for word in string_expression.split(' ')]
        return indices

    def tensorFromExpression(self, string_expression):
        indices = self.indicesFromExpression(string_expression)
        indices.append(EOS_token)
        return torch.LongTensor(indices, device=device).view(-1, 1)

    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromExpression(pair[0])
        target_tensor = self.tensorFromExpression(pair[1])
        return (input_tensor, target_tensor)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def readLangs(file_path):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('\t')] for l in lines]

    return pairs


def prepareData(lang_name):
    lang = Lang(lang_name)
    pairs = readLangs(file_path=data_path)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        lang.addSentence(pair[0])
        lang.addSentence(pair[1])
    print("Counted words:")
    print(lang.n_words)
    return lang, pairs


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = logic_math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, total_epoch, print_every=1000, plot_every=100, learning_rate=0.01,
               encoder_optimizer=None, decoder_optimizer=None):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    if not encoder_optimizer:
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    if not decoder_optimizer:
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [lang.tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        # print("EPOCH {}".format(iter))
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            state = {
                'epoch': total_epoch - n_iters + iter, 'encoder': encoder, 'decoder': decoder,
                'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer,
            }
            torch.save(state, "../checkpoints/autoencoder_checkpoint.pt")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    return plot_losses


def evaluate(encoder, decoder, lang, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = lang.tensorFromExpression(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, lang, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
hidden_size = 128
NUM_LAYERS = 10
data_path = "../data/logic_statement.txt"
EPOCH = 100000

if __name__ == "__main__":
    lang, pairs = prepareData('Math')
    encoder1 = EncoderRNN(lang.n_words, hidden_size, num_layers=NUM_LAYERS).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, lang.n_words, num_layers=NUM_LAYERS, dropout_p=0.1).to(device)
    print(encoder1)
    print(attn_decoder1)
    trained_iter = 0

    while trained_iter < EPOCH:
        if not os.path.isfile("../checkpoints/autoencoder_checkpoint.pt"):
            trainIters(encoder1, attn_decoder1, n_iters=1, total_epoch=1, print_every=1)
        try:
            state = torch.load("../checkpoints/autoencoder_checkpoint.pt")
            trained_iter = state["epoch"]
            losses = trainIters(encoder=state["encoder"], decoder=state["decoder"],
                                encoder_optimizer=state["encoder_optimizer"],
                                decoder_optimizer=state["decoder_optimizer"],
                                n_iters=EPOCH + 1 - trained_iter, total_epoch=EPOCH + 1)
        except RuntimeError:
            continue
    # pickle.dump(losses, open("logic_losses.p", "wb"))
    torch.save(encoder1, "../pt_models/logic_statements_autoencoder_encoder{}.pt".format(device))
    torch.save(attn_decoder1, "../pt_models/logic_statements_autoencoder_decoder{}.pt".format(device))
