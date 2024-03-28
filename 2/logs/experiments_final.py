import re
import random
import time
import math
import numpy as np
import nltk
import matplotlib.pyplot as plt
plt.style.use('default')
import pickle
import torch
from torch import nn
from torch import optim
from collections import namedtuple
from collections import Counter, OrderedDict, defaultdict
from nltk import Tree
print("Using torch", torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################### Set seeding function #####################
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

##################### Load data function #####################
def filereader(path):
  with open(path, mode="r", encoding="utf-8") as f:
    for line in f:
      yield line.strip().replace("\\","")

def tokens_from_treestring(s):
  """extract the tokens from a sentiment tree"""
  return re.sub(r"\([0-9] |\)", "", s).split()

SHIFT = 0
REDUCE = 1

def transitions_from_treestring(s):
  s = re.sub("\([0-5] ([^)]+)\)", "0", s)
  s = re.sub("\)", " )", s)
  s = re.sub("\([0-4] ", "", s)
  s = re.sub("\([0-4] ", "", s)
  s = re.sub("\)", "1", s)
  return list(map(int, s.split()))

##################### Example reader function #####################

Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])

def examplereader(path, lower=False):
  """Returns all examples in a file one by one."""
  for line in filereader(path):
    line = line.lower() if lower else line
    tokens = tokens_from_treestring(line)
    tree = Tree.fromstring(line)  # use NLTK's Tree
    label = int(line[1])
    trans = transitions_from_treestring(line)
    yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)


##################### Subtree extraction function #####################
def extract_subtrees(treestring):
  """Given a treestring, returns a list of subtreestrings."""
  subtrees = [treestring]
  counter = 0 # add 1 for every '(', subtract 1 for every ')'
  for i in range(1, len(treestring)):
    if treestring[i] == '(':
      counter += 1
      if counter == 1:
        begin = i # found first position of new child
    elif treestring[i] == ')':
      counter -= 1
      if counter == -1:
        return subtrees # no more subtrees
      if counter == 0:
        end = i + 1 # found final position of new child, add subtrees of child
        subtrees.extend(extract_subtrees(treestring[begin:end]))
  return subtrees


##################### Subtree reader function #####################
def subtreereader(path, lower=False):
  """Returns all examples and their subtrees in a file one by one."""
  for line in filereader(path):
    line = line.lower() if lower else line
    for substring in extract_subtrees(line):
      tokens = tokens_from_treestring(substring)
      tree = Tree.fromstring(substring)  # use NLTK's Tree
      label = int(substring[1])
      trans = transitions_from_treestring(substring)
      yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)


##################### Dictionary function #####################
class OrderedCounter(Counter, OrderedDict):
  """Counter that remembers the order elements are first seen"""
  def __repr__(self):
    return '%s(%r)' % (self.__class__.__name__,
                      OrderedDict(self))
  def __reduce__(self):
    return self.__class__, (OrderedDict(self),)


class Vocabulary:
  """A vocabulary, assigns IDs to tokens"""

  def __init__(self):
    self.freqs = OrderedCounter()
    self.w2i = {}
    self.i2w = []

  def count_token(self, t):
    self.freqs[t] += 1

  def add_token(self, t):
    self.w2i[t] = len(self.w2i)
    self.i2w.append(t)

  def build(self, min_freq=0):
    '''
    min_freq: minimum number of occurrences for a word to be included
              in the vocabulary
    '''
    self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
    self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

    tok_freq = list(self.freqs.items())
    tok_freq.sort(key=lambda x: x[1], reverse=True)
    for tok, freq in tok_freq:
      if freq >= min_freq:
        self.add_token(tok)

##################### Models #####################
        
##################### BOW model #####################
class BOW(nn.Module):
  """A simple bag-of-words model"""

  def __init__(self, vocab_size, embedding_dim, vocab):
    super(BOW, self).__init__()
    self.vocab = vocab

    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)

    # this is a trainable bias term
    self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)

    # the output is the sum across the time dimension (1)
    # with the bias term added
    logits = embeds.sum(1) + self.bias

    return logits


##################### CBOW model #####################
class CBOW(nn.Module):
  """A continuous bag-of-words model"""

  def __init__(self, vocab_size, embedding_dim, output_dim, vocab):
    super(CBOW, self).__init__()
    self.vocab = vocab

    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)

    # parameter matrix W and bias term
    self.W = nn.Linear(embedding_dim, output_dim)

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)

    # the output is the sum across the time dimension (1)
    # followed by the linear layer
    logits = self.W(embeds.sum(1))

    return logits
  

##################### DeepCBOW model #####################
class DeepCBOW(nn.Module):
  """A deep continuous bag-of-words model"""

  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
    super(DeepCBOW, self).__init__()
    self.vocab = vocab

    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)

    # linear transformations and non-linear activation functions
    self.output_layer = nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)

    # the output is the sum across the time dimension (1)
    # followed by the output layers
    logits = self.output_layer(embeds.sum(1))

    return logits
  
##################### Pretrained CBOW model #####################
class PTDeepCBOW(DeepCBOW):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
    super(PTDeepCBOW, self).__init__(
        vocab_size, embedding_dim, hidden_dim, output_dim, vocab)
    
##################### LSTM model #####################
class MyLSTMCell(nn.Module):
  """Our own LSTM cell"""

  def __init__(self, input_size, hidden_size, bias=True):
    """Creates the weights for this LSTM"""
    super(MyLSTMCell, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    self.W_h = nn.Linear(self.hidden_size, self.hidden_size * 4, self.bias)
    self.W_i = nn.Linear(self.input_size, self.hidden_size * 4, self.bias)

    self.reset_parameters()

  def reset_parameters(self):
    """This is PyTorch's default initialization method"""
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, input_, hx, mask=None):
    """
    input is (batch, input_size)
    hx is ((batch, hidden_size), (batch, hidden_size))
    """
    prev_h, prev_c = hx

    # project input and prev state
    input_proj = self.W_i(input_)
    prev_h_proj = self.W_h(prev_h)

    # main LSTM computation

    i, f, g, o = torch.chunk(input_proj + prev_h_proj, 4, -1)
    i = i.sigmoid()
    f = f.sigmoid()
    g = g.tanh()
    o = o.sigmoid()

    c = f * prev_c + i * g
    h = o * c.tanh()

    return h, c

  def __repr__(self):
    return "{}({:d}, {:d})".format(
        self.__class__.__name__, self.input_size, self.hidden_size)
  
class LSTMClassifier(nn.Module):
  """Encodes sentence with an LSTM and projects final hidden state"""

  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
    super(LSTMClassifier, self).__init__()
    self.vocab = vocab
    self.hidden_dim = hidden_dim
    self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
    self.rnn = MyLSTMCell(embedding_dim, hidden_dim)

    self.output_layer = nn.Sequential(
        nn.Dropout(p=0.5),  # explained later
        nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, x):

    B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
    T = x.size(1)  # timesteps (the number of words in the sentence)

    input_ = self.embed(x)

    # here we create initial hidden states containing zeros
    # we use a trick here so that, if input is on the GPU, then so are hx and cx
    hx = input_.new_zeros(B, self.rnn.hidden_size)
    cx = input_.new_zeros(B, self.rnn.hidden_size)

    # process input sentences one word/timestep at a time
    # input is batch-major (i.e., batch size is the first dimension)
    # so the first word(s) is (are) input_[:, 0]
    outputs = []
    for i in range(T):
      hx, cx = self.rnn(input_[:, i], (hx, cx))
      outputs.append(hx)

    # if we have a single example, our final LSTM state is the last hx
    if B == 1:
      final = hx
    else:
      #
      # This part is explained in next section, ignore this else-block for now.
      #
      # We processed sentences with different lengths, so some of the sentences
      # had already finished and we have been adding padding inputs to hx.
      # We select the final state based on the length of each sentence.

      # two lines below not needed if using LSTM from pytorch
      outputs = torch.stack(outputs, dim=0)           # [T, B, D]
      outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]

      # to be super-sure we're not accidentally indexing the wrong state
      # we zero out positions that are invalid
      pad_positions = (x == 1).unsqueeze(-1)

      outputs = outputs.contiguous()
      outputs = outputs.masked_fill_(pad_positions, 0.)

      mask = (x != 1)  # true for valid positions [B, T]
      lengths = mask.sum(dim=1)                 # [B, 1]

      indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
      final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]

    # we use the last hidden state to classify the sentence
    logits = self.output_layer(final)
    return logits
  
class TreeLSTMCell(nn.Module):
  """A Binary Tree LSTM cell"""

  def __init__(self, input_size, hidden_size, bias=True):
    """Creates the weights for this LSTM"""
    super(TreeLSTMCell, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    self.reduce_layer = nn.Linear(2 * hidden_size, 5 * hidden_size)
    self.dropout_layer = nn.Dropout(p=0.25)

    self.reset_parameters()

  def reset_parameters(self):
    """This is PyTorch's default initialization method"""
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, hx_l, hx_r, mask=None):
    """
    hx_l is ((batch, hidden_size), (batch, hidden_size))
    hx_r is ((batch, hidden_size), (batch, hidden_size))
    """
    prev_h_l, prev_c_l = hx_l  # left child
    prev_h_r, prev_c_r = hx_r  # right child

    B = prev_h_l.size(0)

    # we concatenate the left and right children
    # you can also project from them separately and then sum
    children = torch.cat([prev_h_l, prev_h_r], dim=1)

    # project the combined children into a 5D tensor for i,fl,fr,g,o
    # this is done for speed, and you could also do it separately
    proj = self.reduce_layer(children)  # shape: B x 5D

    # each shape: B x D
    i, f_l, f_r, g, o = torch.chunk(proj, 5, dim=-1)

    # main Tree LSTM computation

    # The shape of each of these is [batch_size, hidden_size]

    i = i.sigmoid()
    f_l = f_l.sigmoid()
    f_r = f_r.sigmoid()
    g = g.tanh()
    o = o.sigmoid()

    c = i * g + f_l * prev_c_l + f_r * prev_c_r
    h = o * c.tanh()

    return h, c

  def __repr__(self):
    return "{}({:d}, {:d})".format(
        self.__class__.__name__, self.input_size, self.hidden_size)
  

##################### TreeLSTM model #####################
def batch(states):
  """
  Turns a list of states into a single tensor for fast processing.
  This function also chunks (splits) each state into a (h, c) pair"""
  return torch.cat(states, 0).chunk(2, 1)

def unbatch(state):
  """
  Turns a tensor back into a list of states.
  First, (h, c) are merged into a single state.
  Then the result is split into a list of sentences.
  """
  return torch.split(torch.cat(state, 1), 1, 0)

class TreeLSTM(nn.Module):
  """Encodes a sentence using a TreeLSTMCell"""

  def __init__(self, input_size, hidden_size, bias=True):
    """Creates the weights for this LSTM"""
    super(TreeLSTM, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    self.reduce = TreeLSTMCell(input_size, hidden_size)

    # project word to initial c
    self.proj_x = nn.Linear(input_size, hidden_size)
    self.proj_x_gate = nn.Linear(input_size, hidden_size)

    self.buffers_dropout = nn.Dropout(p=0.5)

  def forward(self, x, transitions):
    """
    WARNING: assuming x is reversed!
    :param x: word embeddings [B, T, E]
    :param transitions: [2T-1, B]
    :return: root states
    """

    B = x.size(0)  # batch size
    T = x.size(1)  # time

    # compute an initial c and h for each word
    # Note: this corresponds to input x in the Tai et al. Tree LSTM paper.
    # We do not handle input x in the TreeLSTMCell itself.
    buffers_c = self.proj_x(x)
    buffers_h = buffers_c.tanh()
    buffers_h_gate = self.proj_x_gate(x).sigmoid()
    buffers_h = buffers_h_gate * buffers_h

    # concatenate h and c for each word
    buffers = torch.cat([buffers_h, buffers_c], dim=-1)

    D = buffers.size(-1) // 2

    # we turn buffers into a list of stacks (1 stack for each sentence)
    # first we split buffers so that it is a list of sentences (length B)
    # then we split each sentence to be a list of word vectors
    buffers = buffers.split(1, dim=0)  # Bx[T, 2D]
    buffers = [list(b.squeeze(0).split(1, dim=0)) for b in buffers]  # BxTx[2D]

    # create B empty stacks
    stacks = [[] for _ in buffers]

    # t_batch holds 1 transition for each sentence
    for t_batch in transitions:

      child_l = []  # contains the left child for each sentence with reduce action
      child_r = []  # contains the corresponding right child

      # iterate over sentences in the batch
      # each has a transition t, a buffer and a stack
      for transition, buffer, stack in zip(t_batch, buffers, stacks):
        if transition == SHIFT:
          stack.append(buffer.pop())
        elif transition == REDUCE:
          assert len(stack) >= 2, \
            "Stack too small! Should not happen with valid transition sequences"
          child_r.append(stack.pop())  # right child is on top
          child_l.append(stack.pop())

      # if there are sentences with reduce transition, perform them batched
      if child_l:
        reduced = iter(unbatch(self.reduce(batch(child_l), batch(child_r))))
        for transition, stack in zip(t_batch, stacks):
          if transition == REDUCE:
            stack.append(next(reduced))

    final = [stack.pop().chunk(2, -1)[0] for stack in stacks]
    final = torch.cat(final, dim=0)  # tensor [B, D]

    return final
  
class TreeLSTMClassifier(nn.Module):
  """Encodes sentence with a TreeLSTM and projects final hidden state"""

  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
    super(TreeLSTMClassifier, self).__init__()
    self.vocab = vocab
    self.hidden_dim = hidden_dim
    self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
    self.treelstm = TreeLSTM(embedding_dim, hidden_dim)
    self.output_layer = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(hidden_dim, output_dim, bias=True)
    )

  def forward(self, x):

    # x is a pair here of words and transitions; we unpack it here.
    # x is batch-major: [B, T], transitions is time major [2T-1, B]
    x, transitions = x
    emb = self.embed(x)

    # we use the root/top state of the Tree LSTM to classify the sentence
    root_states = self.treelstm(emb, transitions)

    # we use the last hidden state to classify the sentence
    logits = self.output_layer(root_states)
    return logits
  
##################### ChildSum TreeLSTM model #####################
class ChildSumCell(nn.Module):
  """A Child-Sum Tree LSTM cell"""

  def __init__(self, input_size, hidden_size, bias=True):
    """Creates the weights for this LSTM"""
    super(ChildSumCell, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    self.reduce_layer = nn.Linear(hidden_size, 3 * hidden_size)
    self.dropout_layer = nn.Dropout(p=0.25)

    self.reduce_layer_f = nn.Linear(hidden_size, hidden_size)
    self.dropout_layer = nn.Dropout(p=0.25)

    self.reset_parameters()

  def reset_parameters(self):
    """This is PyTorch's default initialization method"""
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, hx_l, hx_r, mask=None):
    """
    hx_l is ((batch, hidden_size), (batch, hidden_size))
    hx_r is ((batch, hidden_size), (batch, hidden_size))
    """
    prev_h_l, prev_c_l = hx_l  # left child
    prev_h_r, prev_c_r = hx_r  # right child

    B = prev_h_l.size(0)

    #children = torch.cat([prev_h_l, prev_h_r], dim=1)
    child_sum = prev_h_l + prev_h_r
    children = torch.cat([prev_h_l, prev_h_r], dim=0)

    proj = self.reduce_layer(child_sum)  # shape: B x 3D
    proj_f = self.reduce_layer_f(children) # shape: 2B x D

    # each shape: B x D
    i, g, o = torch.chunk(proj, 3, dim=-1)
    f_l, f_r = torch.chunk(proj_f, 2, dim=0)

    # main Tree LSTM computation

    # The shape of each of these is [batch_size, hidden_size]

    i = i.sigmoid()
    f_l = f_l.sigmoid()
    f_r = f_r.sigmoid()
    g = g.tanh()
    o = o.sigmoid()

    c = i * g + f_l * prev_c_l + f_r * prev_c_r
    h = o * c.tanh()

    return h, c

  def __repr__(self):
    return "{}({:d}, {:d})".format(
        self.__class__.__name__, self.input_size, self.hidden_size)
  
class ChildSumLSTM(nn.Module):
  """Encodes a sentence using a ChildSumCell"""

  def __init__(self, input_size, hidden_size, bias=True):
    """Creates the weights for this LSTM"""
    super(ChildSumLSTM, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    self.reduce = ChildSumCell(input_size, hidden_size)

    # project word to initial c
    self.proj_x = nn.Linear(input_size, hidden_size)
    self.proj_x_gate = nn.Linear(input_size, hidden_size)

    self.buffers_dropout = nn.Dropout(p=0.5)

  def forward(self, x, transitions):
    """
    WARNING: assuming x is reversed!
    :param x: word embeddings [B, T, E]
    :param transitions: [2T-1, B]
    :return: root states
    """

    B = x.size(0)  # batch size
    T = x.size(1)  # time

    # compute an initial c and h for each word
    # Note: this corresponds to input x in the Tai et al. Tree LSTM paper.
    # We do not handle input x in the TreeLSTMCell itself.
    buffers_c = self.proj_x(x)
    buffers_h = buffers_c.tanh()
    buffers_h_gate = self.proj_x_gate(x).sigmoid()
    buffers_h = buffers_h_gate * buffers_h

    # concatenate h and c for each word
    buffers = torch.cat([buffers_h, buffers_c], dim=-1)

    D = buffers.size(-1) // 2

    # we turn buffers into a list of stacks (1 stack for each sentence)
    # first we split buffers so that it is a list of sentences (length B)
    # then we split each sentence to be a list of word vectors
    buffers = buffers.split(1, dim=0)  # Bx[T, 2D]
    buffers = [list(b.squeeze(0).split(1, dim=0)) for b in buffers]  # BxTx[2D]

    # create B empty stacks
    stacks = [[] for _ in buffers]

    # t_batch holds 1 transition for each sentence
    for t_batch in transitions:

      child_l = []  # contains the left child for each sentence with reduce action
      child_r = []  # contains the corresponding right child

      # iterate over sentences in the batch
      # each has a transition t, a buffer and a stack
      for transition, buffer, stack in zip(t_batch, buffers, stacks):
        if transition == SHIFT:
          stack.append(buffer.pop())
        elif transition == REDUCE:
          assert len(stack) >= 2, \
            "Stack too small! Should not happen with valid transition sequences"
          child_r.append(stack.pop())  # right child is on top
          child_l.append(stack.pop())

      # if there are sentences with reduce transition, perform them batched
      if child_l:
        reduced = iter(unbatch(self.reduce(batch(child_l), batch(child_r))))
        for transition, stack in zip(t_batch, stacks):
          if transition == REDUCE:
            stack.append(next(reduced))

    final = [stack.pop().chunk(2, -1)[0] for stack in stacks]
    final = torch.cat(final, dim=0)  # tensor [B, D]

    return final


class ChildSumClassifier(nn.Module):
  """Encodes sentence with a Child-Sum TreeLSTM and projects final hidden state"""

  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
    super(ChildSumClassifier, self).__init__()
    self.vocab = vocab
    self.hidden_dim = hidden_dim
    self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
    self.treelstm = ChildSumLSTM(embedding_dim, hidden_dim)
    self.output_layer = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(hidden_dim, output_dim, bias=True)
    )

  def forward(self, x):

    # x is a pair here of words and transitions; we unpack it here.
    # x is batch-major: [B, T], transitions is time major [2T-1, B]
    x, transitions = x
    emb = self.embed(x)

    # we use the root/top state of the Tree LSTM to classify the sentence
    root_states = self.treelstm(emb, transitions)

    # we use the last hidden state to classify the sentence
    logits = self.output_layer(root_states)
    return logits


##################### Utils functions #####################
def pad(tokens, length, pad_value=1):
  """add padding 1s to a sequence to that it has the desired length"""
  return tokens + [pad_value] * (length - len(tokens))


def get_examples(data, shuffle=True, **kwargs):
  """Shuffle data set and return 1 example at a time (until nothing left)"""
  if shuffle:
    print("Shuffling training data")
    random.shuffle(data)  # shuffle training data each epoch
  for example in data:
    yield example

def get_minibatch(data, batch_size=25, shuffle=True):
  """Return minibatches, optional shuffling"""

  if shuffle:
    print("Shuffling training data")
    random.shuffle(data)  # shuffle training data each epoch

  batch = []

  # yield minibatches
  for example in data:
    batch.append(example)

    if len(batch) == batch_size:
      yield batch
      batch = []

  # in case there is something left
  if len(batch) > 0:
    yield batch


def prepare_example(example, vocab):
  """
  Map tokens to their IDs for a single example
  """

  # vocab returns 0 if the word is not there (i2w[0] = <unk>)
  x = [vocab.w2i.get(t, 0) for t in example.tokens]

  x = torch.LongTensor([x])
  x = x.to(device)

  y = torch.LongTensor([example.label])
  y = y.to(device)

  return x, y

def prepare_minibatch(mb, vocab):
  """
  Minibatch is a list of examples.
  This function converts words to IDs and returns
  torch tensors to be used as input/targets.
  """
  batch_size = len(mb)
  maxlen = max([len(ex.tokens) for ex in mb])

  # vocab returns 0 if the word is not there
  x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

  x = torch.LongTensor(x)
  x = x.to(device)

  y = [ex.label for ex in mb]
  y = torch.LongTensor(y)
  y = y.to(device)

  return x, y

def prepare_treelstm_minibatch(mb, vocab):
  """
  Returns sentences reversed (last word first)
  Returns transitions together with the sentences.
  """
  batch_size = len(mb)
  maxlen = max([len(ex.tokens) for ex in mb])

  # vocab returns 0 if the word is not there
  # NOTE: reversed sequence!
  x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]

  x = torch.LongTensor(x)
  x = x.to(device)

  y = [ex.label for ex in mb]
  y = torch.LongTensor(y)
  y = y.to(device)

  maxlen_t = max([len(ex.transitions) for ex in mb])
  transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
  transitions = np.array(transitions)
  transitions = transitions.T  # time-major

  return (x, transitions), y


def simple_evaluate(model, data, prep_fn=prepare_example, **kwargs):
  """Accuracy of a model on given data set."""
  correct = 0
  total = 0
  model.eval()  # disable dropout

  for example in data:

    # convert the example input and label to PyTorch tensors
    x, target = prep_fn(example, model.vocab)

    # forward pass without backpropagation (no_grad)
    # get the output from the neural network for input x
    with torch.no_grad():
      logits = model(x)

    # get the prediction
    prediction = logits.argmax(dim=-1)

    # add the number of correct predictions to the total correct
    correct += (prediction == target).sum().item()
    total += 1

  return correct, total, correct / float(total)

def evaluate(model, data,
             batch_fn=get_minibatch, prep_fn=prepare_minibatch,
             batch_size=16):
  """Accuracy of a model on given data set (using mini-batches)"""
  correct = 0
  total = 0
  model.eval()  # disable dropout

  for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
    x, targets = prep_fn(mb, model.vocab)
    with torch.no_grad():
      logits = model(x)

    predictions = logits.argmax(dim=-1).view(-1)

    # add the number of correct predictions to the total correct
    correct += (predictions == targets.view(-1)).sum().item()
    total += targets.size(0)

  return correct, total, correct / float(total)

LOWER = False  # we will keep the original casing
train_data = list(examplereader("trees/train.txt", lower=LOWER))
dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
test_data = list(examplereader("trees/test.txt", lower=LOWER))

def evaluate_models(models, data_sets, eval_fn=simple_evaluate,
                    batch_fn=get_minibatch, prep_fn=prepare_example,
                    batch_size=16):
  """
  Evaluates each model on each data set.
  Returns a list of lists where each list contains the accuracies of one model.
  """
  accuracies = []
  for model in models:
    model_accs = []
    for data_set in data_sets:
      _, _, accuracy = eval_fn(model, data_set, batch_size=batch_size,
                               batch_fn=batch_fn, prep_fn=prep_fn)
      model_accs.append(accuracy)
    accuracies.append(model_accs)
  return accuracies

def train_model(model, optimizer,
                num_iterations=10000,
                print_every=1000, eval_every=1000,
                batch_fn=get_examples,
                prep_fn=prepare_example,
                eval_fn=simple_evaluate,
                batch_size=1, eval_batch_size=None,
                data_train=train_data):
  """Train a model."""
  data_train = data_train.copy()
  iter_i = 0
  train_loss = 0.
  print_num = 0
  start = time.time()
  criterion = nn.CrossEntropyLoss() # loss function
  best_eval = 0.
  best_iter = 0

  # store train loss and validation accuracy during training
  # so we can plot them afterwards
  losses = []
  dev_accs = []

  if eval_batch_size is None:
    eval_batch_size = batch_size

  while True:  # when we run out of examples, shuffle and continue
    for batch in batch_fn(data_train, batch_size=batch_size):

      # forward pass
      model.train()
      x, targets = prep_fn(batch, model.vocab)
      logits = model(x)

      B = targets.size(0)  # later we will use B examples per update

      # compute cross-entropy loss (our criterion)
      # note that the cross entropy loss function computes the softmax for us
      loss = criterion(logits.view([B, -1]), targets.view(-1))
      train_loss += loss.item()

      # backward pass (tip: check the Introduction to PyTorch notebook)

      # erase previous gradients
      #raise NotImplementedError("Implement this")
      # YOUR CODE HERE
      optimizer.zero_grad()

      # compute gradients
      # YOUR CODE HERE
      loss.backward()

      # update weights - take a small step in the opposite dir of the gradient
      # YOUR CODE HERE
      optimizer.step()

      print_num += 1
      iter_i += 1

      # print info
      if iter_i % print_every == 0:
        train_loss = train_loss / print_every
        print("Iter %r: loss=%.4f, time=%.2fs" %
              (iter_i, train_loss, time.time()-start))
        losses.append(train_loss)
        print_num = 0
        train_loss = 0.

      # evaluate
      if iter_i % eval_every == 0:
        _, _, accuracy = eval_fn(model, dev_data, batch_size=eval_batch_size,
                                 batch_fn=batch_fn, prep_fn=prep_fn)
        dev_accs.append(accuracy)
        print("iter %r: dev acc=%.4f" % (iter_i, accuracy))

        # save best model parameters
        if accuracy > best_eval:
          print("new highscore")
          best_eval = accuracy
          best_iter = iter_i
          path = "{}.pt".format(model.__class__.__name__)
          ckpt = {
              "state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "best_eval": best_eval,
              "best_iter": best_iter
          }
          torch.save(ckpt, path)

      # done training
      if iter_i == num_iterations:
        print("Done training")

        # evaluate on train, dev, and test with best model
        print("Loading best model")
        path = "{}.pt".format(model.__class__.__name__)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["state_dict"])

        #_, _, train_acc = eval_fn(
        #    model, data_train, batch_size=eval_batch_size,
        #    batch_fn=batch_fn, prep_fn=prep_fn)
        _, _, dev_acc = eval_fn(
            model, dev_data, batch_size=eval_batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn)
        _, _, test_acc = eval_fn(
            model, test_data, batch_size=eval_batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn)

        print("best model iter {:d}: "
              "dev acc={:.4f}, test acc={:.4f}".format(
                  best_iter, dev_acc, test_acc))

        return losses, dev_accs, test_acc
      

def train_with_seeds(seeds, model_class, lr, *model_args,
                     embeddings=None, finetune=False, **train_args):
  """
  For each seed, trains the model and returns the results.

  Args:
    seeds: A list of seeds.
    model_class: Class of the model to be trained.
    lr: Learning rate.
    *model_args: Arguments needed for initializing the model.
    embeddings: Pre-trained word embeddings (should NOT contain vectors for <unk> and <pad>).
    finetune: Fine-tunes the pre-trained word embeddings if True, keeps them frozen if False.
    **train_args: Optional arguments for the train_model function.

  Returns:
    models: A list of the trained models.
    losses: A list of lists of training losses during training (one list for every seed).
    dev_accs: A list of lists of development accuracies during training (one list for every seed).
    test_accs: A list of test accuracies of the best model after training (one value for every seed).
  """
  models = []
  losses = []
  dev_accs = []
  test_accs = []

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  for seed in seeds:
    set_seed(seed)
    model = model_class(*model_args)

    if embeddings is not None:
      # vectors for <unk> and <pad>
      unk_pad = [np.random.normal(0., 1., 300), np.random.normal(0., 1., 300)]
      full_embeddings = np.vstack((unk_pad, embeddings))
      with torch.no_grad():
        model.embed.weight.data.copy_(torch.from_numpy(full_embeddings))
        model.embed.weight.requires_grad = finetune

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    results = train_model(model, optimizer, **train_args)
    models.append(model)
    losses.append(results[0])
    dev_accs.append(results[1])
    test_accs.append(results[2])
  return models, losses, dev_accs, test_accs

def means_and_stds(inputs):
  """Return means and standard deviations along axis 0."""
  inputs = np.array(inputs)
  means = np.mean(inputs, axis=0)
  stds = np.std(inputs, axis=0)
  return means, stds



##################### Main function #####################

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    LOWER = False  # we will keep the original casing
    train_data = list(examplereader("trees/train.txt", lower=LOWER))
    dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
    test_data = list(examplereader("trees/test.txt", lower=LOWER))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    s = next(filereader("trees/dev.txt"))
    print(s)
    extract_subtrees(s)

    LOWER = False  # we will keep the original casing
    train_extra = list(subtreereader("trees/train.txt", lower=LOWER))
    print("train data with subtrees length:", len(train_extra))

    v_train = Vocabulary()
    for data_set in (train_data,):
        for ex in data_set:
            for token in ex.tokens:
                v_train.count_token(token)

    v_train.build()
    print("Vocabulary size:", len(v_train.w2i))

    v_glove = Vocabulary()
    vectors_glove = []

    file_name = "glove.840B.300d.sst.txt"
    with open(file_name, "r") as f:
        for line in f.read().splitlines():
            line = line.split()
            v_glove.count_token(line[0])
            vectors_glove.append(np.array(line[1:], dtype=float))

    v_glove.build()
    # vectors for <unk> and <pad> will be initialized later, because dependent on seed
    vectors_glove = np.stack(vectors_glove, axis=0)
    print("GloVe vocabulary size:", len(v_glove.w2i))

    #v_w2v = Vocabulary()
    #vectors_w2v = []

    """ file_name = "word2vec.sst.txt"
    with open(file_name, "r") as f:
        for line in f.read().splitlines():
            line = line.split()
            v_w2v.count_token(line[0])
            vectors_w2v.append(np.array(line[1:], dtype=float))

    v_w2v.build()
    # vectors for <unk> and <pad> will be initialized later, because dependent on seed
    vectors_w2v = np.stack(vectors_w2v, axis=0)
    print("Vocabulary size:", len(v_w2v.w2i)) """

    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})


    lengths = [len(ex.tokens) for ex in test_data]
    print('Median test sample length:', np.median(lengths))

    s = 0
    m = 0
    l = 0

    for ex in test_data:
        if len(ex.tokens) < 18:
            s += 1
        elif len(ex.tokens) == 18:
            m += 1
        else:
            l += 1

    print('Number of test samples shorter than 18:', s)
    print('Number of test samples of length 18:', m)
    print('Number of test samples longer than 18:', l)

    test_short = [ex for ex in test_data if len(ex.tokens) <= 18]
    test_long = [ex for ex in test_data if len(ex.tokens) > 18]

    print('Number of short test samples:', len(test_short))
    print('Number of long test samples:', len(test_long))

    print('Mean length of all test samples', np.mean([len(ex.tokens) for ex in test_data]))
    print('Mean length of short test samples', np.mean([len(ex.tokens) for ex in test_short]))
    print('Mean length of long test samples', np.mean([len(ex.tokens) for ex in test_long]))



    seeds = [0, 1]
    models, losses, dev_accs, test_accs = train_with_seeds(
        seeds, BOW, 0.0005, len(v_train.w2i), len(t2i), v_train,
        num_iterations=30000, print_every=1000, eval_every=1000)

    loss_means, loss_stds = means_and_stds(losses)

    print("losses :", losses)
    dev_acc_means, dev_acc_stds = means_and_stds(dev_accs)
    test_acc_mean, test_acc_std = means_and_stds(test_accs)

    short_long_accs = evaluate_models(models, [test_short, test_long])
    short_long_means, short_long_stds = means_and_stds(short_long_accs)

    bow_results = (loss_means, loss_stds, dev_acc_means, dev_acc_stds,
                test_acc_mean, test_acc_std, short_long_means, short_long_stds)
    with open('BOW_results1.pkl', 'wb') as f:
        pickle.dump(bow_results, f)


main()

