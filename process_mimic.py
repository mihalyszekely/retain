# This script processes MIMIC-III dataset and builds longitudinal diagnosis records for patients with at least two visits.
# The output data are pickled and suitable for training Doctor AI or RETAIN
# Written by Edward Choi (mp2893@gatech.edu)
# Usage: Put this script to the folder where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv <output file> 

# Output files
# <output file>.pids: List of unique Patient IDs. Used for intermediate processing
# <output file>.morts: List of binary values indicating the mortality of each patient
# <output file>.dates: List of List of Python datetime objects. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.seqs: List of List of List of integer diagnosis codes. The outer List is for each patient. The middle List contains visits made by each patient. The inner List contains the integer diagnosis codes that occurred in each visit
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.

import sys
import pickle
import argparse
from datetime import datetime

def convert_to_icd9(dxStr):
	if dxStr.startswith('E'):
		if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
		else: return dxStr
	else:
		if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
		else: return dxStr
	
def convert_to_3digit_icd9(dxStr):
	if dxStr.startswith('E'):
		if len(dxStr) > 4: return dxStr[:4]
		else: return dxStr
	else:
		if len(dxStr) > 3: return dxStr[:3]
		else: return dxStr

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process MIMIC-III dataset and build longitudinal diagnosis records.')
    parser.add_argument('admissions_file', help='Path to the ADMISSIONS.csv file.')
    parser.add_argument('diagnoses_file', help='Path to the DIAGNOSES_ICD.csv file.')
    parser.add_argument('patients_file', help='Path to the PATIENTS.csv file.')
    parser.add_argument('output_file', help='Path to the output file.')
    return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	admissionFile = args.admissions_file
	diagnosisFile = args.diagnoses_file
	patientsFile = args.patients_file
	outFile = args.output_file

	print('Collecting mortality information')
	pidDodMap = {}
	with open(patientsFile, 'r') as infd:
		infd.readline()
		for line in infd:
			tokens = line.strip().split(',')
			pid = int(tokens[1])
			dod_hosp = tokens[5]
			if len(dod_hosp) > 0:
				pidDodMap[pid] = 1
			else:
				pidDodMap[pid] = 0

	print('Building pid-admission mapping, admission-date mapping')
	pidAdmMap = {}
	admDateMap = {}
	with open(admissionFile, 'r') as infd:
		infd.readline()
		for line in infd:
			tokens = line.strip().split(',')
			pid = int(tokens[1])
			admId = int(tokens[2])
			admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
			admDateMap[admId] = admTime
			if pid in pidAdmMap: pidAdmMap[pid].append(admId)
			else: pidAdmMap[pid] = [admId]

	print('Building admission-dxList mapping')
	admDxMap = {}
	admDxMap_3digit = {}
	with open(diagnosisFile, 'r') as infd:
		infd.readline()
		for line in infd:
			tokens = line.strip().split(',')
			admId = int(tokens[2])
			dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
			dxStr_3digit = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])

			if admId in admDxMap: 
				admDxMap[admId].append(dxStr)
			else: 
				admDxMap[admId] = [dxStr]

			if admId in admDxMap_3digit: 
				admDxMap_3digit[admId].append(dxStr_3digit)
			else: 
				admDxMap_3digit[admId] = [dxStr_3digit]

	print('Building pid-sortedVisits mapping')
	pidSeqMap = {}
	pidSeqMap_3digit = {}
	for pid, admIdList in pidAdmMap.items():
		if len(admIdList) < 2: continue

		sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
		pidSeqMap[pid] = sortedList

		sortedList_3digit = sorted([(admDateMap[admId], admDxMap_3digit[admId]) for admId in admIdList])
		pidSeqMap_3digit[pid] = sortedList_3digit
	
	print('Building pids, dates, mortality_labels, strSeqs')
	pids = []
	dates = []
	seqs = []
	morts = []
	for pid, visits in pidSeqMap.items():
		pids.append(pid)
		morts.append(pidDodMap[pid])
		seq = []
		date = []
		for visit in visits:
			date.append(visit[0])
			seq.append(visit[1])
		dates.append(date)
		seqs.append(seq)
	
	print('Building pids, dates, strSeqs for 3digit ICD9 code')
	seqs_3digit = []
	for pid, visits in pidSeqMap_3digit.items():
		seq = []
		for visit in visits:
			seq.append(visit[1])
		seqs_3digit.append(seq)
	
	print('Converting strSeqs to intSeqs, and making types')
	types = {}
	newSeqs = []
	for patient in seqs:
		newPatient = []
		for visit in patient:
			newVisit = []
			for code in visit:
				if code in types:
					newVisit.append(types[code])
				else:
					types[code] = len(types)
					newVisit.append(types[code])
			newPatient.append(newVisit)
		newSeqs.append(newPatient)
	
	print('Converting strSeqs to intSeqs, and making types for 3digit ICD9 code')
	types_3digit = {}
	newSeqs_3digit = []
	for patient in seqs_3digit:
		newPatient = []
		for visit in patient:
			newVisit = []
			for code in set(visit):
				if code in types_3digit:
					newVisit.append(types_3digit[code])
				else:
					types_3digit[code] = len(types_3digit)
					newVisit.append(types_3digit[code])
			newPatient.append(newVisit)
		newSeqs_3digit.append(newPatient)

	with open(outFile+'.pids', 'wb') as f: pickle.dump(pids, f)
	with open(outFile+'.dates', 'wb') as f: pickle.dump(dates, f)
	with open(outFile+'.morts', 'wb') as f: pickle.dump(morts, f)
	with open(outFile+'.seqs', 'wb') as f: pickle.dump(newSeqs, f)
	with open(outFile+'.types', 'wb') as f: pickle.dump(types, f)
	with open(outFile+'.3digitICD9.seqs', 'wb') as f: pickle.dump(newSeqs_3digit, f)
	with open(outFile+'.3digitICD9.types', 'wb') as f: pickle.dump(types_3digit, f)
