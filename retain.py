# -*- coding: utf-8 -*-
#################################################################
# Original Theano RETAIN by Edward Choi (mp2893@gatech.edu)
# Refactored to PyTorch + Python 3.x (corrected/faithful version)
#################################################################

import sys
import random
import pickle
import argparse
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1


def load_embedding(infile):
    with open(infile, 'rb') as f:
        return np.array(pickle.load(f)).astype(np.float32)  # (inputDim, embDim)


class RETAIN(nn.Module):
    """
    Faithful PyTorch implementation of RETAIN.

    Differences vs Theano handled:
    - α/β GRU outputs are scaled by 0.5 (as in original).
    - External embedding matrix is transposed when loaded (Linear stores weight as (out,in)).
    - Time channel (if used) is concatenated only for GRU inputs; context sum uses emb only.
    - O(T^2) computation mirrored: for each i, run GRUs over reversed prefix [:i+1].
    """

    def __init__(
        self,
        inputDimSize: int,
        embDimSize: int,
        alphaHiddenDimSize: int,
        betaHiddenDimSize: int,
        numClass: int,
        use_time: bool,
        keep_prob_emb: float,
        keep_prob_context: float,
    ):
        super().__init__()
        self.inputDimSize = inputDimSize
        self.embDimSize = embDimSize
        self.alphaHiddenDimSize = alphaHiddenDimSize
        self.betaHiddenDimSize = betaHiddenDimSize
        self.numClass = numClass
        self.use_time = use_time

        # Visit embedding: x (one-hot/bow over codes) → E
        self.embedding = nn.Linear(inputDimSize, embDimSize, bias=False)
        self.dropout_emb = nn.Dropout(p=1.0 - keep_prob_emb)

        gru_input_size = embDimSize + (1 if use_time else 0)
        self.gru_alpha = nn.GRU(gru_input_size, alphaHiddenDimSize, batch_first=True)
        self.alpha_fc = nn.Linear(alphaHiddenDimSize, 1)

        self.gru_beta = nn.GRU(gru_input_size, betaHiddenDimSize, batch_first=True)
        self.beta_fc = nn.Linear(betaHiddenDimSize, embDimSize)

        self.dropout_context = nn.Dropout(p=1.0 - keep_prob_context)
        self.output_fc = nn.Linear(embDimSize, numClass)

        self._init_like_theano()

    def _init_like_theano(self):
        # Match the original uniform init [-0.1, 0.1]
        def _u(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.GRU):
                for name, p in m.named_parameters(recurse=False):
                    if "weight" in name:
                        nn.init.uniform_(p, -0.1, 0.1)
                    elif "bias" in name:
                        nn.init.zeros_(p)

        self.apply(_u)

    def forward(self, x, t=None, lengths=None):
        """
        x: (B, T, inputDim)
        t: (B, T) or None (only used if self.use_time)
        lengths: (B,) int tensor or list (unused inside forward; used by caller to pick last step)
        Returns:
          logits: (B, T) real-valued scores (pre-sigmoid)
        """
        # (B, T, E)
        emb = self.embedding(x)
        emb = self.dropout_emb(emb)

        if self.use_time:
            if t is None:
                raise ValueError("Time tensor t must be provided when use_time=True")
            temb = torch.cat((emb, t.unsqueeze(2)), dim=2)  # (B, T, E+1)
        else:
            temb = emb  # (B, T, E)

        B, T, _ = temb.shape
        c_t_list = []

        # For each timestep i, run GRUs over reversed prefix [:i+1]
        for i in range(T):
            # (B, i+1, D) reversed in time
            subseq_rev = torch.flip(temb[:, : i + 1, :], dims=[1])

            # α path
            h_a_rev, _ = self.gru_alpha(subseq_rev)       # (B, i+1, Ha)
            h_a = torch.flip(h_a_rev, dims=[1]) * 0.5     # reverse back and scale 0.5
            pre_alpha = self.alpha_fc(h_a)                # (B, i+1, 1)
            alpha = F.softmax(pre_alpha, dim=1)           # softmax over time positions

            # β path
            h_b_rev, _ = self.gru_beta(subseq_rev)        # (B, i+1, Hb)
            h_b = torch.flip(h_b_rev, dims=[1]) * 0.5
            beta = torch.tanh(self.beta_fc(h_b))          # (B, i+1, E)

            # Context uses 'emb' (no time channel), prefix [:i+1]
            c_t = (alpha * beta * emb[:, : i + 1, :]).sum(dim=1)  # (B, E)
            c_t_list.append(c_t)

        # (B, T, E) → dropout → (B, T) logits
        context = torch.stack(c_t_list, dim=1)
        context = self.dropout_context(context)
        logits = self.output_fc(context).squeeze(-1)  # (B, T)

        return logits


# ---------- Data utilities ----------

def pad_matrix_with_time(seqs, times, inputDimSize, useLogTime):
    """
    seqs: list of length B, each element is a list of visits, each visit is list of code indices.
    times: list of length B, each element is list[float] (durations between visits).
    Returns x:(B,T,inputDim), t:(B,T), lengths:(B,) int
    """
    lengths = np.array([len(seq) for seq in seqs], dtype=np.int32)
    B = len(seqs)
    T = int(np.max(lengths)) if B > 0 else 0

    x = np.zeros((B, T, inputDimSize), dtype=np.float32)
    t = np.zeros((B, T), dtype=np.float32)

    for b, (visits, time_list) in enumerate(zip(seqs, times)):
        for i, visit in enumerate(visits):
            for code in visit:
                x[b, i, code] = 1.0
        t[b, : lengths[b]] = np.asarray(time_list, dtype=np.float32)

    if useLogTime:
        t = np.log(t + 1.0, dtype=np.float32)

    return torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(lengths)


def pad_matrix_without_time(seqs, inputDimSize):
    lengths = np.array([len(seq) for seq in seqs], dtype=np.int32)
    B = len(seqs)
    T = int(np.max(lengths)) if B > 0 else 0

    x = np.zeros((B, T, inputDimSize), dtype=np.float32)
    for b, visits in enumerate(seqs):
        for i, visit in enumerate(visits):
            for code in visit:
                x[b, i, code] = 1.0

    return torch.from_numpy(x), torch.from_numpy(lengths)


def load_data_simple(seqFile, labelFile, timeFile=''):
    with open(seqFile, 'rb') as f:
        sequences = np.array(pickle.load(f), dtype=object)
    with open(labelFile, 'rb') as f:
        labels = np.array(pickle.load(f))

    if len(timeFile) > 0:
        with open(timeFile, 'rb') as f:
            times = np.array(pickle.load(f), dtype=object)

    dataSize = len(labels)
    np.random.seed(0)
    ind = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = valid_set_t = test_set_t = None

    if len(timeFile) > 0:
        train_set_t = times[train_indices]
        test_set_t = times[test_indices]
        valid_set_t = times[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    valid_sorted_index = len_argsort(valid_set_x)
    test_sorted_index = len_argsort(test_set_x)

    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)
    return train_set, valid_set, test_set


def load_data(seqFile, labelFile, timeFile):
    with open(seqFile + '.train', 'rb') as f:
        train_set_x = pickle.load(f)
    with open(seqFile + '.valid', 'rb') as f:
        valid_set_x = pickle.load(f)
    with open(seqFile + '.test', 'rb') as f:
        test_set_x = pickle.load(f)
    with open(labelFile + '.train', 'rb') as f:
        train_set_y = pickle.load(f)
    with open(labelFile + '.valid', 'rb') as f:
        valid_set_y = pickle.load(f)
    with open(labelFile + '.test', 'rb') as f:
        test_set_y = pickle.load(f)

    train_set_t = valid_set_t = test_set_t = None
    if len(timeFile) > 0:
        with open(timeFile + '.train', 'rb') as f:
            train_set_t = pickle.load(f)
        with open(timeFile + '.valid', 'rb') as f:
            valid_set_t = pickle.load(f)
        with open(timeFile + '.test', 'rb') as f:
            test_set_t = pickle.load(f)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    valid_sorted_index = len_argsort(valid_set_x)
    test_sorted_index = len_argsort(test_set_x)

    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)
    return train_set, valid_set, test_set


# ---------- Eval / Logging ----------

def calculate_auc(model, dataset, options, device):
    model.eval()
    batchSize = options['batchSize']
    useTime = options['useTime']
    inputDimSize = options['inputDimSize']
    useLogTime = options['useLogTime']

    scoreVec = []
    labelsVec = []

    with torch.no_grad():
        n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
        for index in range(n_batches):
            batchX = dataset[0][index * batchSize:(index + 1) * batchSize]
            batchY = dataset[1][index * batchSize:(index + 1) * batchSize]

            if useTime:
                batchT = dataset[2][index * batchSize:(index + 1) * batchSize]
                x, t, lengths = pad_matrix_with_time(batchX, batchT, inputDimSize, useLogTime)
                x, t = x.to(device), t.to(device)
            else:
                x, lengths = pad_matrix_without_time(batchX, inputDimSize)
                x = x.to(device)
                t = None

            logits = model(x, t, lengths)             # (B, T)
            scores = torch.sigmoid(logits)            # (B, T)

            lens = lengths.cpu().tolist()
            last_idxs = torch.tensor([l - 1 for l in lens], device=scores.device, dtype=torch.long).unsqueeze(1)
            last_scores = scores.gather(1, last_idxs).squeeze(1).tolist()

            scoreVec.extend(last_scores)
            labelsVec.extend([float(y) for y in batchY])

    return roc_auc_score(labelsVec, scoreVec)


def print2file(buf, outFile):
    with open(outFile, 'a') as outfd:
        outfd.write(buf + '\n')


# ---------- Training ----------

def train_RETAIN(
    seqFile='seqFile.txt',
    inputDimSize=20000,
    labelFile='labelFile.txt',
    numClass=1,
    outFile='outFile.txt',
    timeFile='',
    modelFile='',
    useLogTime=True,
    embFile='',
    embDimSize=128,
    embFineTune=True,
    alphaHiddenDimSize=128,
    betaHiddenDimSize=128,
    batchSize=100,
    max_epochs=10,
    L2_output=0.001,
    L2_emb=0.001,
    L2_alpha=0.001,
    L2_beta=0.001,
    keepProbEmb=0.5,
    keepProbContext=0.5,
    logEps=1e-8,  # not used with BCEWithLogitsLoss, kept for API parity
    solver='adadelta',
    simpleLoad=False,
    verbose=False,
):
    options = locals().copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    useTime = len(timeFile) > 0
    options['useTime'] = useTime

    print('Initializing the model ...')
    model = RETAIN(
        inputDimSize,
        embDimSize,
        alphaHiddenDimSize,
        betaHiddenDimSize,
        numClass,
        useTime,
        keepProbEmb,
        keepProbContext,
    ).to(device)

    # External embedding
    if len(embFile) > 0:
        print('Using external code embedding')
        emb_weights = load_embedding(embFile)  # (inputDim, embDim)
        assert emb_weights.shape == (inputDimSize, embDimSize), \
            f"embed_file shape {emb_weights.shape} != ({inputDimSize}, {embDimSize})"
        with torch.no_grad():
            # Linear weight is (out_features, in_features)
            model.embedding.weight.copy_(torch.from_numpy(emb_weights).T)

    # Fine-tune switch
    model.embedding.weight.requires_grad = bool(embFineTune)

    # Optionally warm start from a saved PyTorch checkpoint
    if len(modelFile) > 0:
        print(f'Loading previous model from {modelFile}')
        sd = torch.load(modelFile, map_location=device)
        model.load_state_dict(sd, strict=True)
        # Respect current finetune flag even after load
        model.embedding.weight.requires_grad = bool(embFineTune)

    print('Building the model ...')
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    if solver == 'adadelta':
        optimizer = torch.optim.Adadelta(params_to_update, lr=1.0, rho=0.95, eps=1e-6)
    elif solver == 'adam':
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
    else:
        raise ValueError("solver must be 'adadelta' or 'adam'")

    print('Loading data ...')
    if simpleLoad:
        trainSet, validSet, testSet = load_data_simple(seqFile, labelFile, timeFile)
    else:
        trainSet, validSet, testSet = load_data(seqFile, labelFile, timeFile)
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    print('done')

    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    logFile = outFile + '.log'

    print('Optimization start !!')
    for epoch in range(max_epochs):
        model.train()
        iteration = 0
        costVector = []

        # Shuffle mini-batch order (over batch *indices*)
        shuffled_indices = random.sample(range(n_batches), n_batches)

        for index in shuffled_indices:
            batchX = trainSet[0][index * batchSize:(index + 1) * batchSize]
            batchY = torch.tensor(
                trainSet[1][index * batchSize:(index + 1) * batchSize],
                dtype=torch.float32,
                device=device,
            )

            if useTime:
                batchT = trainSet[2][index * batchSize:(index + 1) * batchSize]
                x, t, lengths = pad_matrix_with_time(batchX, batchT, inputDimSize, useLogTime)
                x, t = x.to(device), t.to(device)
            else:
                x, lengths = pad_matrix_without_time(batchX, inputDimSize)
                x = x.to(device)
                t = None

            optimizer.zero_grad()
            logits = model(x, t, lengths)  # (B, T)

            # Select last valid step per sequence using gather (GPU-safe)
            lens = lengths.cpu().tolist()
            last_idxs = torch.tensor([l - 1 for l in lens], device=logits.device, dtype=torch.long).unsqueeze(1)
            last_logits = logits.gather(1, last_idxs).squeeze(1)  # (B,)

            loss = criterion(last_logits, batchY)

            # L2 regularization terms
            l2_reg = torch.tensor(0.0, device=device)
            if L2_output > 0:
                l2_reg = l2_reg + L2_output * torch.sum(model.output_fc.weight ** 2)
            if embFineTune and L2_emb > 0:
                l2_reg = l2_reg + L2_emb * torch.sum(model.embedding.weight ** 2)
            if L2_alpha > 0:
                l2_reg = l2_reg + L2_alpha * torch.sum(model.alpha_fc.weight ** 2)
            if L2_beta > 0:
                l2_reg = l2_reg + L2_beta * torch.sum(model.beta_fc.weight ** 2)

            (loss + l2_reg).backward()
            optimizer.step()

            costVector.append((loss + l2_reg).item())
            if (iteration % 10 == 0) and verbose:
                print(f'Epoch:{epoch}, Iteration:{iteration}/{n_batches}, Train_Cost:{costVector[-1]:.6f}')
            iteration += 1

        trainCost = float(np.mean(costVector)) if costVector else 0.0
        validAuc = calculate_auc(model, validSet, options, device)
        buf = f'Epoch:{epoch}, Train_cost:{trainCost:.6f}, Validation_AUC:{validAuc:.6f}'
        print(buf)
        print2file(buf, logFile)

        if validAuc > bestValidAuc:
            bestValidAuc = validAuc
            bestValidEpoch = epoch
            bestTestAuc = calculate_auc(model, testSet, options, device)
            buf = f'Currently the best validation AUC found. Test AUC:{bestTestAuc:.6f} at epoch:{epoch}'
            print(buf)
            print2file(buf, logFile)
            save_path = f"{outFile}.{epoch}.pt"
            torch.save(model.state_dict(), save_path)

    buf = f'The best validation & test AUC:{bestValidAuc:.6f}, {bestTestAuc:.6f} at epoch:{bestValidEpoch}'
    print(buf)
    print2file(buf, logFile)


# ---------- CLI ----------

def parse_arguments(parser):
    parser.add_argument('seq_file', type=str, metavar='<visit_file>',
                        help='Pickle file containing visit information of patients')
    parser.add_argument('n_input_codes', type=int, metavar='<n_input_codes>',
                        help='Number of unique input medical codes')
    parser.add_argument('label_file', type=str, metavar='<label_file>',
                        help='Pickle file containing labels')
    parser.add_argument('out_file', metavar='<out_file>',
                        help='Output prefix; checkpoints will be saved as <out_file>.<epoch>.pt')
    parser.add_argument('--time_file', type=str, default='',
                        help='Pickle file containing durations between visits; omit to disable time')
    parser.add_argument('--model_file', type=str, default='',
                        help='Path to a saved PyTorch state_dict to warm-start training (.pt/.pth)')
    parser.add_argument('--use_log_time', type=int, default=1, choices=[0, 1],
                        help='Use log(1 + t) for time (default: 1)')
    parser.add_argument('--embed_file', type=str, default='',
                        help='Pickle file with code embedding matrix of shape (inputDim, embDim)')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Visit embedding size (default: 128)')
    parser.add_argument('--embed_finetune', type=int, default=1, choices=[0, 1],
                        help='Fine-tune embeddings if 1 (default: 1)')
    parser.add_argument('--alpha_hidden_dim_size', type=int, default=128,
                        help='Hidden size for GRU generating alpha (default: 128)')
    parser.add_argument('--beta_hidden_dim_size', type=int, default=128,
                        help='Hidden size for GRU generating beta (default: 128)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Mini-batch size (default: 100)')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Training epochs (default: 10)')
    parser.add_argument('--L2_output', type=float, default=0.001,
                        help='L2 for output classifier weight (default: 0.001)')
    parser.add_argument('--L2_emb', type=float, default=0.001,
                        help='L2 for embedding weight (default: 0.001)')
    parser.add_argument('--L2_alpha', type=float, default=0.001,
                        help='L2 for alpha generator weight (default: 0.001)')
    parser.add_argument('--L2_beta', type=float, default=0.001,
                        help='L2 for beta generator weight (default: 0.001)')
    parser.add_argument('--keep_prob_emb', type=float, default=0.5,
                        help='Keep probability for dropout on embeddings (default: 0.5)')
    parser.add_argument('--keep_prob_context', type=float, default=0.5,
                        help='Keep probability for dropout on context (default: 0.5)')
    parser.add_argument('--log_eps', type=float, default=1e-8,
                        help='(unused) Kept for API parity (default: 1e-8)')
    parser.add_argument('--solver', type=str, default='adadelta', choices=['adadelta', 'adam'],
                        help='Optimizer (default: adadelta)')
    parser.add_argument('--simple_load', action='store_true',
                        help='Randomly split a single dataset into train/valid/test')
    parser.add_argument('--verbose', action='store_true',
                        help='Print more frequently during training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    train_RETAIN(
        seqFile=args.seq_file,
        inputDimSize=args.n_input_codes,
        labelFile=args.label_file,
        outFile=args.out_file,
        timeFile=args.time_file,
        modelFile=args.model_file,
        useLogTime=(args.use_log_time == 1),
        embFile=args.embed_file,
        embDimSize=args.embed_size,
        embFineTune=(args.embed_finetune == 1),
        alphaHiddenDimSize=args.alpha_hidden_dim_size,
        betaHiddenDimSize=args.beta_hidden_dim_size,
        batchSize=args.batch_size,
        max_epochs=args.n_epochs,
        L2_output=args.L2_output,
        L2_emb=args.L2_emb,
        L2_alpha=args.L2_alpha,
        L2_beta=args.L2_beta,
        keepProbEmb=args.keep_prob_emb,
        keepProbContext=args.keep_prob_context,
        logEps=args.log_eps,
        solver=args.solver,
        simpleLoad=args.simple_load,
        verbose=args.verbose,
    )