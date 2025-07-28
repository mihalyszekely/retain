# -*- coding: utf-8 -*-
#################################################################
# Original Theano RETAIN by Edward Choi (mp2893@gatech.edu)
# PyTorch evaluation/refactor (faithful to Theano eval and
# compatible with the corrected training script)
#################################################################

import argparse
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Utilities --------------------

def load_embedding(infile):
    with open(infile, 'rb') as f:
        return np.array(pickle.load(f)).astype(np.float32)  # (inputDim, embDim)

def pad_matrix_with_time(seqs, times, inputDimSize, useLogTime):
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
        t = np.log(t + 1.0).astype(np.float32)

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

def load_data(dataFile, labelFile, timeFile):
    with open(dataFile, 'rb') as f:
        test_set_x = np.array(pickle.load(f), dtype=object)
    with open(labelFile, 'rb') as f:
        test_set_y = np.array(pickle.load(f))

    test_set_t = None
    if len(timeFile) > 0:
        with open(timeFile, 'rb') as f:
            test_set_t = np.array(pickle.load(f), dtype=object)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in sorted_index]
    test_set_y = [test_set_y[i] for i in sorted_index]
    if len(timeFile) > 0:
        test_set_t = [test_set_t[i] for i in sorted_index]

    return (test_set_x, test_set_y, test_set_t)

def print2file(buf, outFile):
    with open(outFile, 'a') as outfd:
        outfd.write(buf + '\n')

# -------------------- Model --------------------

class RETAIN(nn.Module):
    """Same module structure as the training script; used to load checkpoint."""
    def __init__(self, inputDimSize, embDimSize, alphaHiddenDimSize, betaHiddenDimSize, numClass, use_time):
        super().__init__()
        self.inputDimSize = inputDimSize
        self.embDimSize = embDimSize
        self.alphaHiddenDimSize = alphaHiddenDimSize
        self.betaHiddenDimSize = betaHiddenDimSize
        self.numClass = numClass
        self.use_time = use_time

        self.embedding = nn.Linear(inputDimSize, embDimSize, bias=False)
        gru_input_size = embDimSize + (1 if use_time else 0)

        self.gru_alpha = nn.GRU(gru_input_size, alphaHiddenDimSize, batch_first=True)
        self.alpha_fc = nn.Linear(alphaHiddenDimSize, 1)

        self.gru_beta = nn.GRU(gru_input_size, betaHiddenDimSize, batch_first=True)
        self.beta_fc = nn.Linear(betaHiddenDimSize, embDimSize)

        self.output_fc = nn.Linear(embDimSize, numClass)

    @torch.no_grad()
    def alpha_beta_final(self, x, t=None):
        """
        Compute α, β for the final prediction (Theano eval style):
        reverse whole sequence once → GRUs → reverse back → scale 0.5 → α softmax, β tanh.
        x: (B, T, inputDim)
        t: (B, T) or None
        Returns:
          emb   : (B, T, E)
          alpha : (B, T, 1)  (time-softmax over T)
          beta  : (B, T, E)
        """
        emb = self.embedding(x)  # (B, T, E)

        temb = torch.cat((emb, t.unsqueeze(2)), dim=2) if self.use_time else emb  # (B,T,E[+1])
        rev = torch.flip(temb, dims=[1])

        h_a_rev, _ = self.gru_alpha(rev)
        h_b_rev, _ = self.gru_beta(rev)

        h_a = torch.flip(h_a_rev, dims=[1]) * 0.5  # scale as in Theano
        h_b = torch.flip(h_b_rev, dims=[1]) * 0.5

        pre_alpha = self.alpha_fc(h_a)                  # (B, T, 1)
        alpha = F.softmax(pre_alpha, dim=1)            # softmax over time
        beta = torch.tanh(self.beta_fc(h_b))           # (B, T, E)
        return emb, alpha, beta

# -------------------- Evaluation --------------------

def test_RETAIN(
    modelFile='model.pt',
    seqFile='seq.pkl',
    labelFile='label.pkl',
    outFile='out.txt',
    timeFile='',
    typeFile='types.pkl',
    useLogTime=True,
    embFile=''
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load state dict and infer shapes
    state = torch.load(modelFile, map_location=device)
    embDimSize = state['embedding.weight'].shape[0]
    inputDimSize = state['embedding.weight'].shape[1]
    alphaHiddenDimSize = state['alpha_fc.weight'].shape[1]
    betaHiddenDimSize = state['beta_fc.weight'].shape[1]
    numClass = state['output_fc.weight'].shape[0]
    useTime = len(timeFile) > 0

    # Build model skeleton and load weights
    model = RETAIN(inputDimSize, embDimSize, alphaHiddenDimSize, betaHiddenDimSize, numClass, useTime).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Optionally override embedding with external matrix (transpose needed)
    if embFile:
        emb_weights = load_embedding(embFile)  # (inputDim, embDim)
        assert emb_weights.shape == (inputDimSize, embDimSize), \
            f"embed_file shape {emb_weights.shape} != ({inputDimSize}, {embDimSize})"
        with torch.no_grad():
            model.embedding.weight.copy_(torch.from_numpy(emb_weights).T)

    # Load data
    testSet = load_data(seqFile, labelFile, timeFile)

    # Load types mapping and reverse it: id(int) -> code(str)
    with open(typeFile, 'rb') as f:
        types = pickle.load(f)
    rtypes = {v: k for k, v in types.items()}

    print('Contribution calculation start!!')
    count = 0
    with open(outFile, 'w') as outfd:
        for index in range(len(testSet[0])):
            if count % 100 == 0:
                print(f'processed {count} patients')
            count += 1

            batchX = [testSet[0][index]]
            label = int(testSet[1][index])

            if useTime:
                batchT = [testSet[2][index]]
                x, t, lengths = pad_matrix_with_time(batchX, batchT, inputDimSize, useLogTime)
                x, t = x.to(device), t.to(device)
            else:
                x, lengths = pad_matrix_without_time(batchX, inputDimSize)
                x = x.to(device)
                t = None

            L = int(lengths[0])

            with torch.no_grad():
                emb, alpha, beta = model.alpha_beta_final(x, t)  # emb:(1,T,E), alpha:(1,T,1), beta:(1,T,E)
                emb_mat = emb[0, :L, :]           # (L, E)
                alpha_vec = alpha[0, :L, 0]       # (L,)
                beta_mat = beta[0, :L, :]         # (L, E)

                # Final-step context and score
                ct = (alpha_vec.unsqueeze(1) * beta_mat * emb_mat).sum(dim=0)           # (E,)
                w = model.output_fc.weight.squeeze(0)                                   # (E,)
                b = model.output_fc.bias.squeeze() if model.output_fc.bias is not None else torch.tensor(0., device=w.device)
                y = torch.sigmoid(ct @ w + b).item()

            # Write per-visit, per-code contributions using code embedding (not whole visit)
            lines = []
            patient = batchX[0]
            for i, visit in enumerate(patient):
                lines.append('-------------- visit_index:%d ---------------' % i)
                a_i = float(alpha_vec[i].item())
                b_i = beta_mat[i]  # (E,)
                for code in visit:
                    code_emb = model.embedding.weight[:, code]  # (E,)
                    contrib = torch.dot(w, a_i * b_i * code_emb).item()
                    lines.append('%s:%f  ' % (rtypes[code], contrib))
                lines.append('------------------------------------')
            lines.append('patient_index:%d, label:%d, score:%f' % (index, label, y))
            outfd.write('\n'.join(lines) + '\n')

# -------------------- CLI --------------------

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('model_file', type=str, metavar='<model_file>',
                   help='Path to the PyTorch checkpoint (from training), e.g., outFile.<epoch>.pt')
    p.add_argument('seq_file', type=str, metavar='<visit_file>',
                   help='Pickle file with visit sequences')
    p.add_argument('label_file', type=str, metavar='<label_file>',
                   help='Pickle file with labels')
    p.add_argument('type_file', type=str, metavar='<type_file>',
                   help='Pickle dict mapping medical code string -> int')
    p.add_argument('out_file', type=str, metavar='<out_file>',
                   help='Where to write contributions')
    p.add_argument('--time_file', type=str, default='',
                   help='Pickle file with durations between visits (optional)')
    p.add_argument('--use_log_time', type=int, default=1, choices=[0, 1],
                   help='Use log(1 + t) (default: 1)')
    p.add_argument('--embed_file', type=str, default='',
                   help='Optional external embedding (shape: inputDim x embDim); will override checkpoint')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    test_RETAIN(
        modelFile=args.model_file,
        seqFile=args.seq_file,
        labelFile=args.label_file,
        typeFile=args.type_file,
        outFile=args.out_file,
        timeFile=args.time_file,
        useLogTime=(args.use_log_time == 1),
        embFile=args.embed_file
    )