#################################################################
# Code written by Edward Choi (mp2893@gatech.edu)
# For bug report, please contact author using the email address
#################################################################

import sys
import random
import numpy as np
import pickle
from collections import OrderedDict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def load_embedding(infile):
	return np.array(pickle.load(open(infile, 'rb')))

class RETAIN(nn.Module):
	def __init__(self, inputDimSize, embDimSize, alphaHiddenDimSize, betaHiddenDimSize, numClass, use_time):
		super(RETAIN, self).__init__()
		self.inputDimSize = inputDimSize
		self.embDimSize = embDimSize
		self.alphaHiddenDimSize = alphaHiddenDimSize
		self.betaHiddenDimSize = betaHiddenDimSize
		self.numClass = numClass
		self.use_time = use_time
		
		self.embedding = nn.Linear(inputDimSize, embDimSize, bias=False)
		
		gru_input_size = embDimSize
		if self.use_time:
			gru_input_size += 1

		self.gru_alpha = nn.GRU(gru_input_size, alphaHiddenDimSize, batch_first=True)
		self.alpha_fc = nn.Linear(alphaHiddenDimSize, 1)

		self.gru_beta = nn.GRU(gru_input_size, betaHiddenDimSize, batch_first=True)
		self.beta_fc = nn.Linear(betaHiddenDimSize, embDimSize)

		self.output_fc = nn.Linear(embDimSize, numClass)

	def forward(self, x, t, lengths):
		n_timesteps = x.size(1)
		emb = self.embedding(x)

		if self.use_time:
			temb = torch.cat((emb, t.unsqueeze(2)), 2)
		else:
			temb = emb

		alpha_list = []
		beta_list = []

		for i in range(n_timesteps):
			reverse_emb_t_list = []
			for j in range(x.size(0)):
				if i < lengths[j]:
					reverse_emb_t_list.append(torch.flip(temb[j, :i+1, :], dims=[0]))
				else:
					reverse_emb_t_list.append(torch.zeros_like(temb[j, :i+1, :]))
			
			padded_reverse_emb_t = torch.nn.utils.rnn.pad_sequence(reverse_emb_t_list, batch_first=True)
			
			alpha_output, _ = self.gru_alpha(padded_reverse_emb_t)
			pre_alpha = self.alpha_fc(alpha_output)
			alpha = F.softmax(pre_alpha, dim=1)
			alpha_list.append(alpha)
			
			beta_output, _ = self.gru_beta(padded_reverse_emb_t)
			beta = torch.tanh(self.beta_fc(beta_output))
			beta_list.append(beta)
		
		return alpha_list, beta_list

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
	
	test_set = (test_set_x, test_set_y, test_set_t)

	return test_set

def print2file(buf, outFile):
	with open(outFile, 'a') as outfd:
		outfd.write(buf + '\n')

def test_RETAIN(
	modelFile='model.npz',
	seqFile='seqFile.txt',
	labelFile='labelFile.txt',
	outFile='outFile.txt',
	timeFile='timeFile.txt',
	typeFile='types.txt',
	useLogTime=True,
	embFile='',
	logEps=1e-8
):
	options = locals().copy()
	useTime = len(timeFile) > 0
	options['useTime'] = useTime

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	params = torch.load(modelFile, map_location=lambda storage, loc: storage)
	
	alphaHiddenDimSize = params['alpha_fc.weight'].shape[1]
	betaHiddenDimSize = params['beta_fc.weight'].shape[1]
	embDimSize = params['embedding.weight'].shape[0]
	inputDimSize = params['embedding.weight'].shape[1]
	num_class = params['output_fc.weight'].shape[0]

	model = RETAIN(inputDimSize, embDimSize, alphaHiddenDimSize, betaHiddenDimSize, num_class, useTime).to(device)
	model.load_state_dict(params)
	model.eval()
	
	if embFile:
		embedding_weights = load_embedding(embFile)
		model.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))

	testSet = load_data(seqFile, labelFile, timeFile)
	
	with open(typeFile, 'rb') as f:
		types = pickle.load(f)
	rtypes = dict([(v,k) for k,v in types.items()])

	print('Contribution calculation start!!')
	count = 0
	with open(outFile, 'w') as outfd:
		for index in range(len(testSet[0])):
			if count % 100 == 0: print('processed %d patients' % count)
			count += 1
			batchX = [testSet[0][index]]
			label = testSet[1][index]
			
			if useTime: 
				batchT = [testSet[2][index]]
				x, t, lengths = pad_matrix_with_time(batchX, batchT, inputDimSize, useLogTime)
				x, t = x.to(device), t.to(device)
			else:
				x, lengths = pad_matrix_without_time(batchX, inputDimSize)
				x = x.to(device)
				t = None

			with torch.no_grad():
				emb = model.embedding(x)
				alpha, beta = model(x, t, lengths)
			
				ct_list = []
				for i in range(lengths[0]):
					alpha_i_unpadded = alpha[i][0, :i+1]
					beta_i_unpadded = beta[i][0, :i+1]
					emb_i_unpadded = emb[0, :i+1]
					
					ct = (alpha_i_unpadded * beta_i_unpadded * emb_i_unpadded).sum(0)
					ct_list.append(ct)
				
				context_vectors = torch.stack(ct_list)
				
				logits = torch.matmul(context_vectors, model.output_fc.weight.t()) + model.output_fc.bias
				y_t = sigmoid(logits.cpu().detach().numpy())

				buf = ''
				patient = batchX[0]
				for i in range(len(patient)):
					visit = patient[i]
					buf += '-------------- visit_index:%d ---------------\n' % i
					for j in range(len(visit)):
						code = visit[j]
						
						original_visit_emb = emb[0, i, :]
						
						alpha_i_unpadded = alpha[i][0, :i+1]
						beta_i_unpadded = beta[i][0, :i+1]
						
						unpadded_emb_for_visit_i = emb[0, :i+1]
						
						alpha_i_contrib = alpha_i_unpadded[i].item()
						beta_i_contrib = beta_i_unpadded[i]
						
						contribution = torch.dot(model.output_fc.weight.squeeze(), (alpha_i_contrib * beta_i_contrib * original_visit_emb))
						buf += '%s:%f  ' % (rtypes[code], contribution.item())
					buf += '\n------------------------------------\n'
				buf += 'patient_index:%d, label:%d, score:%f\n\n' % (index, label, y_t[lengths[0]-1][0])
				outfd.write(buf + '\n')
	
def parse_arguments(parser):
	parser.add_argument('model_file', type=str, metavar='<model_file>', help='The path to the PyTorch model file.')
	parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The path to the cPickled file containing visit information of patients')
	parser.add_argument('label_file', type=str, metavar='<label_file>', help='The path to the cPickled file containing label information of patients')
	parser.add_argument('type_file', type=str, metavar='<type_file>', help='The path to the cPickled dictionary for mapping medical code strings to integers')
	parser.add_argument('out_file', metavar='<out_file>', help='The path to the output models. The models will be saved after every epoch')
	parser.add_argument('--time_file', type=str, default='', help='The path to the cPickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
	parser.add_argument('--use_log_time', type=int, default=1, choices=[0,1], help='Use logarithm of time duration to dampen the impact of the outliers (0 for false, 1 for true) (default value: 1)')
	parser.add_argument('--embed_file', type=str, default='', help='The path to the cPickled file containing the representation vectors of medical codes. If you are not using medical code representations, do not use this option')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)

	test_RETAIN(
		modelFile=args.model_file,
		seqFile=args.seq_file, 
		labelFile=args.label_file, 
		typeFile=args.type_file, 
		outFile=args.out_file, 
		timeFile=args.time_file, 
		useLogTime=args.use_log_time == 1,
		embFile=args.embed_file
	)
