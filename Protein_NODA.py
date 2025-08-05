import pandas as pd
import numpy as np
import glob
import os
import sys
import itertools as it
import Protein_NODA_CSP as CSP
import Protein_NODA_PRE as PRE
import Protein_NODA_Correlations as Corelation
import Protein_NODA_SlowEx as SlowEx
import Protein_NODA_Intensity




########################################################################################################################
replacements = {'N1-': '   N1-', 'N3-': '   N3-', 'N7-': '   N7-', 'N9-': '   N9-',
				'C6-': '   C6-',  'C7-': '   C7-', 'C8-': '   C8-', "C1'-": "   C1'-", 'C5-': '   C5-', 'C2-': '   C2-',
				"C2'-": "   C2'-", "C3'-": "   C3'-","C4'-": "   C4'-","C5'-": "   C5'-", 'P-': '   P-', "P-   C4'":"P-C4'",
				 "P-   C5'":"P-C5'"}
Inheader = 'place holder'
nmrPipe_header = "INDEX   X_AXIS    Y_AXIS     DX     DY    w1     w2  X_HZ        Y_HZ      XW      YW     lw1    lw2   X1   X3    Y1   Y3    Intensity      DHEIGHT         Volume        PCHI2  TYPE Group Atom CLUSTID MEMCNT \n"
Sort_dict = {'nmrPipe': [nmrPipe_header, 'FORMAT'], 'sparky': [Inheader, 'Assignment']}
Format_Name = {'uM': r'$\mu$M', 'MnCl2': r'MnCl$_{2}$', 'MgCl2': r'MgCl$_{2}$', 'CaCl2': r'CaCl$_{2}$', 'ZnCl2': r'ZnCl$_{2}$', 'BaCl2': r'BaCl$_{2}$', "H2O": r"H$_{2}$O", "D2O": r"D$_{2}$O"}

CurDir = os.getcwd()

	
# def Main():
if "run" in sys.argv[1].lower():
	inputParPath = os.path.join(CurDir, sys.argv[2])
	with open(inputParPath, 'r') as inputfile:
		rawInputData = []
		# read the input file and create list of input params with '+' spearating different runs
		for key, group in it.groupby(inputfile, lambda line: line.startswith('+')):
			if not key:
				rawInputData.append(list(group))
	rawInputData = [[x.strip() for x in y if "#" not in x and len(x) != 1] for y in rawInputData[1:]]
	RunParams = [x.strip().split() for x in rawInputData[0]]
	# Assign variables used in exicution
	# globals()[RunParams[0][0]] = RunParams[0][1]
	# globals()[RunParams[1][0]] = RunParams[1][1]
	for i in range(len(RunParams)):
		if len(RunParams[i]) == 2:
			# globals()[RunParams[i][0]] = RunParams[i][1]
			exec(RunParams[i][0] + '= RunParams[i][1]')
		else:
			# globals()[RunParams[i][0]] = RunParams[i][1:len(RunParams[i])]
			exec(RunParams[i][0] + '= RunParams[i][1:len(RunParams[i])]')
	MasterList = [x.strip().split(', ') for x in rawInputData[1]]

	if Analysis == 'CSP':
		outpath = os.path.join(CurDir, OutDir + '/')
		#cvs_path = os.path.join(outpath, 'CSV_files/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		CSP.Plot_CSP_data(OutName,Sequence,Start_Index,Residues,Data_Types,SDM,Show_Labels,Intensity_Norm,Common_Scale,input_path,PDB, MasterList, outpath,CS_min_max,Intensity_min_max)
	if Analysis == 'Intensity':
		outpath = os.path.join(CurDir, OutDir + '/')
		#cvs_path = os.path.join(outpath, 'CSV_files/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		Protein_NODA_Intensity.Plot_Intensiy_data(RunParams, MasterList, outpath)

	if Analysis == 'SlowEx':
		outpath = os.path.join(CurDir, OutDir + '/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		ConcentrationsList = []
		unit = MasterList[0][1].split()[1]
		for DataSet in MasterList:
			ConcentrationsList.append(float(DataSet[1].split()[0]))  # storing the ligand concentration for use in fitting
		SlowEx.Fitdata(RunParams, MasterList, outpath)
		SlowEx.Plot(outpath, RunParams, ConcentrationsList, unit)
	
	if Analysis == 'Titration':
		outpath = os.path.join(CurDir, OutDir + '/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		ConcentrationsList = []
		for DataSet, x in zip(MasterList, range(len(MasterList))):
			ConcentrationsList.append(float(DataSet[1].split()[0]))  # storing the ligand concentration for use in fitting
		titration.Fit_Data(RunParams, MasterList, outpath)
		titration.Plot(outpath, RunParams, MasterList, ConcentrationsList)
	if Analysis == 'Titration_Ex':
		outpath = os.path.join(CurDir, OutDir + '/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		ConcentrationsList = []
		for DataSet, x in zip(MasterList, range(len(MasterList))):
			ConcentrationsList.append(float(DataSet[1].split()[0]))  # storing the ligand concentration for use in fitting
		titration_Ex.Fit_Data(RunParams, MasterList, outpath)
		titration_Ex.Plot(outpath, RunParams, MasterList, ConcentrationsList)
	if Analysis == 'Titration_Ex_hill':
		outpath = os.path.join(CurDir, OutDir + '/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		ConcentrationsList = []
		for DataSet, x in zip(MasterList, range(len(MasterList))):
			if DataSet[1].split()[2] in Format_Name.keys():
				Ligand = Format_Name[DataSet[1].split()[2]]
			else:
				Ligand = DataSet[1].split()[2]
			ConcentrationsList.append(float(DataSet[1].split()[0]))  # storing the ligand concentration for use in fitting
		titration_Ex_hill.Fit_Data(RunParams, MasterList, outpath)
		titration_Ex_hill.Plot(outpath, RunParams, MasterList, ConcentrationsList)
	if Analysis == 'PRE':
		outpath = os.path.join(CurDir, OutDir + '/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		PRE.Plot_PRE_data(RunParams, MasterList, outpath)


if "plot" in sys.argv[1].lower():
	inputParPath = os.path.join(CurDir, sys.argv[2])
	with open(inputParPath, 'r') as inputfile:
		rawInputData = []
		# read the input file and create list of input params with '+' spearating different runs
		for key, group in it.groupby(inputfile, lambda line: line.startswith('+')):
			if not key:
				rawInputData.append(list(group))
	rawInputData = [[x.strip() for x in y if "#" not in x and len(x) != 1] for y in rawInputData[1:]]
	RunParams = [x.strip().split() for x in rawInputData[0]]
	# Assign variables used in exicution
	for i in range(len(RunParams)):
		if len(RunParams[i]) == 2:
			exec(RunParams[i][0] + '= RunParams[i][1]')
		else:
			exec(RunParams[i][0] + '= RunParams[i][1:len(RunParams[i])]')
	MasterList = [x.strip().split(', ') for x in rawInputData[1]]
	if Analysis == 'Titration':
		outpath = os.path.join(CurDir, OutDir + '/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		ConcentrationsList = []
		for DataSet, x in zip(MasterList, range(len(MasterList))):
			ConcentrationsList.append(float(DataSet[1].split()[0]))  # storing the ligand concentration for use in fitting
		titration.Plot(outpath, RunParams, MasterList, ConcentrationsList)
	if Analysis == 'Titration_Ex':
		outpath = os.path.join(CurDir, OutDir + '/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		ConcentrationsList = []
		for DataSet, x in zip(MasterList, range(len(MasterList))):
			ConcentrationsList.append(float(DataSet[1].split()[0]))  # storing the ligand concentration for use in fitting
		titration_Ex.Plot(outpath, RunParams, MasterList, ConcentrationsList)
	if Analysis == 'Titration_Ex_hill':
		outpath = os.path.join(CurDir, OutDir + '/')
		if not os.path.exists(outpath):
			os.makedirs(outpath)
		ConcentrationsList = []
		for DataSet, x in zip(MasterList, range(len(MasterList))):
			ConcentrationsList.append(float(DataSet[1].split()[0]))  # storing the ligand concentration for use in fitting
		titration_Ex_hill.Plot(outpath, RunParams, MasterList, ConcentrationsList)

	# for DataSet in DataSetList:
	# 	os.system("rm %s" % DataSet)

