
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1

def numpy_floatX(data):
    return np.asarray(data, dtype=np.float32)

def load_embedding(infile):
    return np.array(pickle.load(open(infile, 'rb'))).astype(np.float32)

class RETAIN(nn.Module):
    def __init__(self, inputDimSize, embDimSize, alphaHiddenDimSize, betaHiddenDimSize, numClass, use_time, keep_prob_emb, keep_prob_context):
        super(RETAIN, self).__init__()
        self.inputDimSize = inputDimSize
        self.embDimSize = embDimSize
        self.alphaHiddenDimSize = alphaHiddenDimSize
        self.betaHiddenDimSize = betaHiddenDimSize
        self.numClass = numClass
        self.use_time = use_time
        
        self.embedding = nn.Linear(inputDimSize, embDimSize, bias=False)
        self.dropout_emb = nn.Dropout(p=1.0 - keep_prob_emb)
        
        gru_input_size = embDimSize
        if self.use_time:
            gru_input_size += 1

        self.gru_alpha = nn.GRU(gru_input_size, alphaHiddenDimSize, batch_first=True)
        self.alpha_fc = nn.Linear(alphaHiddenDimSize, 1)

        self.gru_beta = nn.GRU(gru_input_size, betaHiddenDimSize, batch_first=True)
        self.beta_fc = nn.Linear(betaHiddenDimSize, embDimSize)
        
        self.dropout_context = nn.Dropout(p=1.0 - keep_prob_context)
        self.output_fc = nn.Linear(embDimSize, numClass)

    def forward(self, x, t, lengths):
        n_timesteps = x.size(1)
        emb = self.embedding(x)
        emb = self.dropout_emb(emb)

        if self.use_time:
            temb = torch.cat((emb, t.unsqueeze(2)), 2)
        else:
            temb = emb
        
        c_t_list = []
        for i in range(n_timesteps):
            # for each timestep, we want to compute context vector using visits up to that timestep
            # this is O(T^2) and is inherited from the original RETAIN paper
            
            # Get reverse sequence for each sample in the batch
            reverse_emb_t_list = []
            for j in range(x.size(0)):
                if i < lengths[j]:
                    reverse_emb_t_list.append(torch.flip(temb[j, :i+1, :], dims=[0]))
                else:
                    # if length is shorter than i, we use a zero tensor
                    reverse_emb_t_list.append(torch.zeros_like(temb[j, :i+1, :]))
            
            # Pad the reversed sequences
            packed_input = nn.utils.rnn.pad_sequence(reverse_emb_t_list, batch_first=True)
            
            # Alpha (attention)
            alpha_output, _ = self.gru_alpha(packed_input)
            pre_alpha = self.alpha_fc(alpha_output)
            alpha = F.softmax(pre_alpha, dim=1) # (batch, seq_len, 1)

            # Beta (contribution)
            beta_output, _ = self.gru_beta(packed_input)
            beta = torch.tanh(self.beta_fc(beta_output)) # (batch, seq_len, embDim)
            
            # Unpad alpha and beta
            alpha_unpadded_list = []
            beta_unpadded_list = []
            emb_unpadded_list = []

            for j in range(x.size(0)):
                if i < lengths[j]:
                    # get the original length of reversed sequence
                    original_len = i + 1
                    alpha_unpadded_list.append(alpha[j, :original_len, :])
                    beta_unpadded_list.append(beta[j, :original_len, :])
                    emb_unpadded_list.append(torch.flip(emb[j, :i+1, :], dims=[0])) # Original embedding, not time-augmented
                else:
                    # placeholder, these won't be used to compute context vector
                    alpha_unpadded_list.append(torch.zeros(1, 1, device=x.device))
                    beta_unpadded_list.append(torch.zeros(1, self.embDimSize, device=x.device))
                    emb_unpadded_list.append(torch.zeros(1, self.embDimSize, device=x.device))

            c_t_batch = []
            for j in range(x.size(0)):
                if i < lengths[j]:
                    c_t = (alpha_unpadded_list[j] * beta_unpadded_list[j] * emb_unpadded_list[j]).sum(0)
                    c_t_batch.append(c_t)
                else:
                    c_t_batch.append(torch.zeros(self.embDimSize, device=x.device))

            c_t_list.append(torch.stack(c_t_batch))

        context_vectors = torch.stack(c_t_list, dim=1) # (batch, seq_len, embDim)
        context_vectors = self.dropout_context(context_vectors)
        
        logits = self.output_fc(context_vectors) # (batch, seq_len, numClass)
        return logits.squeeze(-1)


def pad_matrix_with_time(seqs, times, inputDimSize, useLogTime):
    lengths = np.array([len(seq) for seq in seqs]).astype('int32')
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen, inputDimSize)).astype(np.float32)
    t = np.zeros((n_samples, maxlen)).astype(np.float32)

    for idx, (seq,time) in enumerate(zip(seqs,times)):
        for i, visit in enumerate(seq):
            for code in visit:
                x[idx, i, code] = 1.
        t[idx, :lengths[idx]] = time

    if useLogTime: t = np.log(t + 1.)
    
    return torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(lengths)


def pad_matrix_without_time(seqs, inputDimSize):
    lengths = np.array([len(seq) for seq in seqs]).astype('int32')
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen, inputDimSize)).astype(np.float32)
    for idx, seq in enumerate(seqs):
        for i, visit in enumerate(seq):
            for code in visit:
                x[idx, i, code] = 1.

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
    valid_indices = ind[nTest:nTest+nValid]
    train_indices = ind[nTest+nValid:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = None
    test_set_t = None
    valid_set_t = None

    if len(timeFile) > 0:
        train_set_t = times[train_indices]
        test_set_t = times[test_indices]
        valid_set_t = times[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
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
    with open(seqFile+'.train', 'rb') as f: train_set_x = pickle.load(f)
    with open(seqFile+'.valid', 'rb') as f: valid_set_x = pickle.load(f)
    with open(seqFile+'.test', 'rb') as f: test_set_x = pickle.load(f)
    with open(labelFile+'.train', 'rb') as f: train_set_y = pickle.load(f)
    with open(labelFile+'.valid', 'rb') as f: valid_set_y = pickle.load(f)
    with open(labelFile+'.test', 'rb') as f: test_set_y = pickle.load(f)
    
    train_set_t = None
    valid_set_t = None
    test_set_t = None

    if len(timeFile) > 0:
        with open(timeFile+'.train', 'rb') as f: train_set_t = pickle.load(f)
        with open(timeFile+'.valid', 'rb') as f: valid_set_t = pickle.load(f)
        with open(timeFile+'.test', 'rb') as f: test_set_t = pickle.load(f)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
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
            batchX = dataset[0][index*batchSize:(index+1)*batchSize]
            batchY = dataset[1][index*batchSize:(index+1)*batchSize]
            
            if useTime:
                batchT = dataset[2][index*batchSize:(index+1)*batchSize]
                x, t, lengths = pad_matrix_with_time(batchX, batchT, inputDimSize, useLogTime)
                x, t = x.to(device), t.to(device)
            else:
                x, lengths = pad_matrix_without_time(batchX, inputDimSize)
                x = x.to(device)
                t = None

            logits = model(x, t, lengths)
            scores = torch.sigmoid(logits)
            
            last_scores = []
            for i in range(len(lengths)):
                last_scores.append(scores[i, lengths[i]-1].item())
            
            scoreVec.extend(last_scores)
            labelsVec.extend(batchY)

    auc = roc_auc_score(labelsVec, scoreVec)
    return auc

def print2file(buf, outFile):
    with open(outFile, 'a') as outfd:
        outfd.write(buf + '\n')

def train_RETAIN(
    seqFile='seqFile.txt',
    inputDimSize=20000,
    labelFile='labelFile.txt',
    numClass=1,
    outFile='outFile.txt',
    timeFile='',
    modelFile='model.npz',
    useLogTime=True,
    embFile='embFile.txt',
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
    logEps=1e-8,
    solver='adadelta',
    simpleLoad=False,
    verbose=False
):
    options = locals().copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    useTime = len(timeFile) > 0
    options['useTime'] = useTime
    
    print('Initializing the model ... ')
    model = RETAIN(inputDimSize, embDimSize, alphaHiddenDimSize, betaHiddenDimSize, numClass, useTime, keepProbEmb, keepProbContext).to(device)

    if len(embFile) > 0: 
        print('using external code embedding')
        emb_weights = load_embedding(embFile)
        model.embedding.weight.data.copy_(torch.from_numpy(emb_weights))
    
    if not embFineTune:
        model.embedding.weight.requires_grad = False
    
    if len(modelFile) > 0:
        print('Loading previous model')
        model.load_state_dict(torch.load(modelFile))

    print('Building the model ... ')
    
    criterion = nn.BCEWithLogitsLoss()
    
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    if solver == 'adadelta':
        optimizer = torch.optim.Adadelta(params_to_update, lr=1.0, rho=0.95, eps=1e-6)
    elif solver == 'adam':
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)

    print('Loading data ... ')
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
        
        # Shuffle training data
        shuffled_indices = random.sample(range(n_batches), n_batches)
        
        for index in shuffled_indices:
            batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
            batchY = torch.tensor(trainSet[1][index*batchSize:(index+1)*batchSize], dtype=torch.float32).to(device)

            if useTime:
                batchT = trainSet[2][index*batchSize:(index+1)*batchSize]
                x, t, lengths = pad_matrix_with_time(batchX, batchT, inputDimSize, useLogTime)
                x, t = x.to(device), t.to(device)
            else:
                x, lengths = pad_matrix_without_time(batchX, inputDimSize)
                x = x.to(device)
                t = None

            optimizer.zero_grad()
            logits = model(x, t, lengths)
            
            last_logits = []
            for i in range(len(lengths)):
                last_logits.append(logits[i, lengths[i]-1])
            
            loss = criterion(torch.stack(last_logits), batchY)
            
            # Add L2 regularization
            l2_reg = torch.tensor(0., device=device)
            if L2_output > 0: l2_reg += L2_output * torch.sum(model.output_fc.weight**2)
            if embFineTune and L2_emb > 0: l2_reg += L2_emb * torch.sum(model.embedding.weight**2)
            if L2_alpha > 0: l2_reg += L2_alpha * torch.sum(model.alpha_fc.weight**2)
            if L2_beta > 0: l2_reg += L2_beta * torch.sum(model.beta_fc.weight**2)
            
            loss += l2_reg
            
            loss.backward()
            optimizer.step()

            costVector.append(loss.item())
            if (iteration % 10 == 0) and verbose: 
                print(f'Epoch:{epoch}, Iteration:{iteration}/{n_batches}, Train_Cost:{loss.item()}')
            iteration += 1

        trainCost = np.mean(costVector)
        validAuc = calculate_auc(model, validSet, options, device)
        buf = f'Epoch:{epoch}, Train_cost:{trainCost}, Validation_AUC:{validAuc}'
        print(buf)
        print2file(buf, logFile)
        
        if validAuc > bestValidAuc: 
            bestValidAuc = validAuc
            bestValidEpoch = epoch
            bestTestAuc = calculate_auc(model, testSet, options, device)
            buf = f'Currently the best validation AUC found. Test AUC:{bestTestAuc} at epoch:{epoch}'
            print(buf)
            print2file(buf, logFile)
            torch.save(model.state_dict(), outFile + '.' + str(epoch))
            
    buf = f'The best validation & test AUC:{bestValidAuc}, {bestTestAuc} at epoch:{bestValidEpoch}'
    print(buf)
    print2file(buf, logFile)

def parse_arguments(parser):
    parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The path to the Pickled file containing visit information of patients')
    parser.add_argument('n_input_codes', type=int, metavar='<n_input_codes>', help='The number of unique input medical codes')
    parser.add_argument('label_file', type=str, metavar='<label_file>', help='The path to the Pickled file containing label information of patients')
    parser.add_argument('out_file', metavar='<out_file>', help='The path to the output models. The models will be saved after every epoch')
    parser.add_argument('--time_file', type=str, default='', help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='', help='The path to the PyTorch model file. Use this option if you want to re-train an existing model')
    parser.add_argument('--use_log_time', type=int, default=1, choices=[0,1], help='Use logarithm of time duration to dampen the impact of the outliers (0 for false, 1 for true) (default value: 1)')
    parser.add_argument('--embed_file', type=str, default='', help='The path to the Pickled file containing the representation vectors of medical codes. If you are not using medical code representations, do not use this option')
    parser.add_argument('--embed_size', type=int, default=128, help='The size of the visit embedding. If you are not providing your own medical code vectors, you can specify this value (default value: 128)')
    parser.add_argument('--embed_finetune', type=int, default=1, choices=[0,1], help='If you are using randomly initialized code representations, always use this option. If you are using an external medical code representations, and you want to fine-tune them as you train RETAIN, use this option (0 for false, 1 for true) (default value: 1)')
    parser.add_argument('--alpha_hidden_dim_size', type=int, default=128, help='The size of the hidden layers of the GRU responsible for generating alpha weights (default value: 128)')
    parser.add_argument('--beta_hidden_dim_size', type=int, default=128, help='The size of the hidden layers of the GRU responsible for generating beta weights (default value: 128)')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--n_epochs', type=int, default=10, help='The number of training epochs (default value: 10)')
    parser.add_argument('--L2_output', type=float, default=0.001, help='L2 regularization for the final classifier weight w (default value: 0.001)')
    parser.add_argument('--L2_emb', type=float, default=0.001, help='L2 regularization for the input embedding weight W_emb (default value: 0.001)')
    parser.add_argument('--L2_alpha', type=float, default=0.001, help='L2 regularization for the alpha generating weight w_alpha (default value: 0.001).')
    parser.add_argument('--L2_beta', type=float, default=0.001, help='L2 regularization for the input embedding weight W_beta (default value: 0.001)')
    parser.add_argument('--keep_prob_emb', type=float, default=0.5, help='Decides how much you want to keep during the dropout between the embedded input and the alpha & beta generation process (default value: 0.5)')
    parser.add_argument('--keep_prob_context', type=float, default=0.5, help='Decides how much you want to keep during the dropout between the context vector c_i and the final classifier (default value: 0.5)')
    parser.add_argument('--log_eps', type=float, default=1e-8, help='A small value to prevent log(0) (default value: 1e-8)')
    parser.add_argument('--solver', type=str, default='adadelta', choices=['adadelta','adam'], help='Select which solver to train RETAIN: adadelta, or adam. (default: adadelta)')
    parser.add_argument('--simple_load', action='store_true', help='Use an alternative way to load the dataset. Instead of you having to provide a trainign set, validation set, test set, this will automatically divide the dataset. (default false)')
    parser.add_argument('--verbose', action='store_true', help='Print output after every 10 mini-batches (default false)')
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
        useLogTime=args.use_log_time == 1,
        embFile=args.embed_file, 
        embDimSize=args.embed_size, 
        embFineTune=args.embed_finetune == 1, 
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
        verbose=args.verbose
    )
