import pandas as pd
import numpy as np
import csv
import glob
import sys
import os
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.markers as mmarkers
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.stats import norm
from scipy.stats import trim_mean
from matplotlib.pyplot import gca
import itertools as it
#pd.set_option('precision',4)


####----------------------------------------------------------------------------------------------####
## 		Dictionaries for sorting input data into pandas Data Frame, and formatting of supper and 	##
## subscript of salts, and primes for sugars, and some other probable things 						##
####----------------------------------------------------------------------------------------------####

CS_cols = ['C6C8','C1p','C2','C5','NImino','N7','H6H8','H1p','H2','H5','HImino','N1N9','AN1','AN3','C2p','C3p','C4p','C5p','H2p','H3p','H4p','H5p','H5pp','P']
replacements = {'X_PPM': 'X_PPM', 'Y_PPM': 'w2', 'XW_HZ': 'lw1', 'YW_HZ': 'lw2', 'HEIGHT': 'Intensity',  'VOL': 'Volume', 'ASS': 'Group     Atom ', 'Assignment': "Group     Atom ",
				" (hz)": " ", 'Data Height': 'Intensity'}
DataTypes_dict = {'CS':['w1','w2'],'Intensity':['Intensity'],'Volume':['Volume'],'LW':['lw1','lw2']}
Sort_dic_full = {"C6-H6":[[['C6C8', 'w1'], ['H6H8', 'w2']], [['C6C8', 'lw1'], ['H6H8', 'lw2']], [['C6C8', 'Intensity']], [['C6C8', 'Volume']]],
				 "C8-H8":[[['C6C8', 'w1'], ['H6H8', 'w2']], [['C6C8', 'lw1'], ['H6H8', 'lw2']], [['C6C8', 'Intensity']], [['C6C8', 'Volume']]],
				 "C7-H7":[[['C7', 'w1'], ['H7', 'w2']], [['C7', 'lw1'], ['H7', 'lw2']], [['C7', 'Intensity']], [['C7', 'Volume']]],
				 "C1'-H1'":[[['C1p', 'w1'], ['H1p', 'w2']], [['C1p', 'lw1'], ['H1p', 'lw2']], [['C1p', 'Intensity']], [['C1p', 'Volume']]],
				 "C2-H2":[[['C2', 'w1'], ['H2', 'w2']], [['C2', 'lw1'], ['H2', 'lw2']], [['C2', 'Intensity']], [['C2', 'Volume']]],
				 "C5-H5":[[['C5', 'w1'], ['H5', 'w2']], [['C5', 'lw1'], ['H5', 'lw2']], [['C5', 'Intensity']], [['C5', 'Volume']]],
				 "N1-H1":[[['NImino', 'w1'], ['HImino', 'w2']], [['NImino', 'lw1'], ['HImino', 'lw2']], [['NImino', 'Intensity']], [['NImino', 'Volume']]],
				 "N3-H3":[[['NImino', 'w1'], ['HImino', 'w2']], [['NImino', 'lw1'], ['HImino', 'lw2']], [['NImino', 'Intensity']], [['NImino', 'Volume']]],
				 "N7-H8":[[['N7', 'w1']], [['N7', 'lw1']], [['N7', 'Intensity']], [['N7', 'Volume']]],
				 "N1-H2":[[['AN1', 'w1']], [['AN1', 'lw1']], [['AN1', 'Intensity']], [['AN1', 'Volume']]],
				 "N3-H2":[[['AN3', 'w1']], [['AN3', 'lw1']], [['AN3', 'Intensity']], [['AN3', 'Volume']]],
				 "N9-H8":[[['N1N9', 'w1']], [['N1N9', 'lw1']], [['N1N9', 'Intensity']], [['N1N9', 'Volume']]],
				 "N9-H1'":[[['N1N9', 'w1']], [['N1N9', 'lw1']], [['N1N9', 'Intensity']], [['N1N9', 'Volume']]],
				 "N1-H6":[[['N1N9', 'w1']], [['N1N9', 'lw1']], [['N1N9', 'Intensity']], [['N1N9', 'Volume']]],
				 "N1-H1'":[[['N1N9', 'w1']], [['N1N9', 'lw1']], [['N1N9', 'Intensity']], [['N1N9', 'Volume']]],
				 "C2'-H2'":[[['C2p', 'w1'], ['H2p', 'w2']], [['C2p', 'lw1'], ['H2p', 'lw2']], [['C2p', 'Intensity']], [['C2p', 'Volume']]],
				 "C2'-H2''":[[['C2p', 'w1'],['H2pp', 'w2']], [['C2p', 'lw1'], ['H2pp', 'lw2']], [['C2p', 'Intensity']], [['H2pp', 'Volume']]],
				 "C3'-H3'":[[['C3p', 'w1'], ['H3p', 'w2']], [['C3p', 'lw1'], ['H3p', 'lw2']], [['C3p', 'Intensity']], [['C3p', 'Volume']]],
				 "C4'-H4'":[[['C4p', 'w1'], ['H4p', 'w2']], [['C4p', 'lw1'], ['H4p', 'lw2']], [['C4p', 'Intensity']], [['C4p', 'Volume']]],
				 "C5'-H5'":[[['C5p', 'w1'], ['H5p', 'w2']], [['C5p', 'lw1'], ['H5p', 'lw2']], [['C5p', 'Intensity']], [['C5p', 'Volume']]],
				 "C5'-H5''":[[['C5p', 'w1'],['H5pp', 'w2']], [['C5p', 'lw1'], ['H5pp', 'lw2']], [['H5pp', 'Intensity']], [['H5pp', 'Volume']]],
				 "P-C4'-H4'": [[['P', 'w1'], ['C4p', 'w2'], ['H4p', 'w3']]], "P-C5'-H5'": [[['P', 'w1'], ['C5p', 'w2'], ['H5p', 'w3']]], "P-C5'-H5''": [[['P', 'w1'], ['C5pp', 'w2'], ['H5pp', 'w3']]],
			     "H1'-C1'-H1'": [[['H1p', 'w1'], ['C1p', 'w2'], ['H1p', 'w3']]], "H1'-C1'-H2'": [[['H1p', 'w1'], ['C1p', 'w2'], ['H2p', 'w3']]], "H1'-C2'-H1'": [[['H1p', 'w1'], ['C2p', 'w2'], ['H1p', 'w3']]],
			     "H1'-C2'-H2'": [[['H1p', 'w1'], ['C2p', 'w2'], ['H2p', 'w3']]], "H1'-C2'-H3'": [[['H1p', 'w1'], ['C2p', 'w2'], ['H3p', 'w3']]], "H1'-C3'-H2'": [[['H1p', 'w1'], ['C3p', 'w2'], ['H2p', 'w3']]],
			     "H1'-C3'-H3'": [[['H1p', 'w1'], ['C3p', 'w2'], ['H3p', 'w3']]], "H1'-C3'-H4'": [[['H1p', 'w1'], ['C3p', 'w2'], ['H4p', 'w3']]], "H1'-C4'-H3'": [[['H1p', 'w1'], ['C4p', 'w2'], ['H3p', 'w3']]],
				 "H1'-C4'-H4'": [[['H1p', 'w1'], ['C4p', 'w2'], ['H4p', 'w3']]], "H1'-C5'-H4'": [[['H1p', 'w1'], ['C5p', 'w2'], ['H4p', 'w3']]], "H2'-C1'-H1'": [[['H2p', 'w1'], ['C1p', 'w2'], ['H1p', 'w3']]],
			     "H2'-C1'-H2'": [[['H2p', 'w1'], ['C1p', 'w2'], ['H2p', 'w3']]], "H2'-C2'-H1'": [[['H2p', 'w1'], ['C2p', 'w2'], ['H1p', 'w3']]], "H2'-C2'-H2'": [[['H2p', 'w1'], ['C2p', 'w2'], ['H2p', 'w3']]],
			     "H2'-C2'-H3'": [[['H2p', 'w1'], ['C2p', 'w2'], ['H3p', 'w3']]], "H2'-C3'-H2'": [[['H2p', 'w1'], ['C3p', 'w2'], ['H2p', 'w3']]], "H2'-C3'-H3'": [[['H2p', 'w1'], ['C3p', 'w2'], ['H3p', 'w3']]],
			     "H2'-C3'-H4'": [[['H2p', 'w1'], ['C3p', 'w2'], ['H4p', 'w3']]], "H2'-C4'-H3'": [[['H2p', 'w1'], ['C4p', 'w2'], ['H3p', 'w3']]], "H2'-C4'-H4'": [[['H2p', 'w1'], ['C4p', 'w2'], ['H4p', 'w3']]],
			     "H2'-C5'-H4'": [[['H2p', 'w1'], ['C5p', 'w2'], ['H4p', 'w3']]], "H3'-C1'-H1'": [[['H2p', 'w1'], ['C1p', 'w2'], ['H1p', 'w3']]], "H3'-C1'-H2'": [[['H3p', 'w1'], ['C1p', 'w2'], ['H2p', 'w3']]],
			     "H3'-C2'-H1'": [[['H3p', 'w1'], ['C2p', 'w2'], ['H1p', 'w3']]], "H3'-C2'-H2'": [[['H3p', 'w1'], ['C2p', 'w2'], ['H2p', 'w3']]], "H3'-C2'-H3'": [[['H3p', 'w1'], ['C2p', 'w2'], ['H3p', 'w3']]],
			     "H3'-C3'-H1'": [[['H3p', 'w1'], ['C3p', 'w2'], ['H1p', 'w3']]], "H3'-C3'-H2'": [[['H3p', 'w1'], ['C3p', 'w2'], ['H2p', 'w3']]], "H3'-C3'-H3'": [[['H3p', 'w1'], ['C3p', 'w2'], ['H3p', 'w3']]],
			     "H3'-C4'-H3'": [[['H3p', 'w1'], ['C4p', 'w2'], ['H3p', 'w3']]], "H3'-C4'-H4'": [[['H3p', 'w1'], ['C4p', 'w2'], ['H4p', 'w3']]], "H3'-C5'-H4'": [[['H3p', 'w1'], ['C5p', 'w2'], ['H4p', 'w3']]],
			     "H4'-C1'-H1'": [[['H4p', 'w1'], ['C1p', 'w2'], ['H1p', 'w3']]], "H4'-C1'-H2'": [[['H4p', 'w1'], ['C1p', 'w2'], ['H2p', 'w3']]], "H4'-C2'-H1'": [[['H4p', 'w1'], ['C2p', 'w2'], ['H1p', 'w3']]],
			     "H4'-C2'-H2'": [[['H4p', 'w1'], ['C2p', 'w2'], ['H2p', 'w3']]], "H4'-C2'-H3'": [[['H4p', 'w1'], ['C2p', 'w2'], ['H3p', 'w3']]], "H4'-C3'-H1'": [[['H4p', 'w1'], ['C3p', 'w2'], ['H1p', 'w3']]],
			     "H4'-C3'-H2'": [[['H4p', 'w1'], ['C3p', 'w2'], ['H2p', 'w3']]], "H4'-C3'-H3'": [[['H4p', 'w1'], ['C3p', 'w2'], ['H3p', 'w3']]], "H4'-C4'-H3'": [[['H4p', 'w1'], ['C4p', 'w2'], ['H3p', 'w3']]],
			     "H4'-C4'-H4'": [[['H4p', 'w1'], ['C4p', 'w2'], ['H4p', 'w3']]], "H4'-C5'-H4'": [[['H4p', 'w1'], ['C4p', 'w2'], ['H4p', 'w3']]]}
Sort_dic_index = {"CS":0,"LW":1,"Intensity":2,"Volume":3}
Format_Name = {'uM': r'$\mu$M', 'MnCl2': r'MnCl$_{2}$', 'MgCl2': 'MgCl$_{2}$', 'CaCl2': r'CaCl$_{2}$', 'ZnCl2': r'ZnCl$_{2}$', 'BaCl2': r'BaCl$_{2}$', "H2O": r"H$_{2}$O", "D2O": r"D$_{2}$O", 
				"C1p": r"C1'", "H1p": r"H1'", "C2p": r"C2'", "H2p": r"H2'", "C3p": r"C3'", "H3p": r"H3'", "C4p": r"C4'", "H4p": r"H4'", "C5p": r"C5'", "H5p": r"H5'", "H5pp": r"H5'"}


def Plot_CSP_data(RunParams, MasterList, outpath):

####----------------------------------------------------------------------------------------------####
## 		Generates all the run parameters need, but lets automate it so we don't have to worry about ##
## missing any thing. 																				##
## RunParams for CSP :Data_Types, Intensity_Norm, ResIDList, outpath, OutName, ColorsDict,			##
## 					  Legend_dict, Title_dict 														##
####----------------------------------------------------------------------------------------------####

	for i in range(len(RunParams)):
		if RunParams[i][0] != 'Sample':
			if len(RunParams[i]) == 2:
				exec(RunParams[i][0] + '= RunParams[i][1]')
			else:
				exec(RunParams[i][0] + '= RunParams[i][1:len(RunParams[i])]')
		if RunParams[i][0] == 'Sample':
			exec(RunParams[i][0] + '= RunParams[i][1:]')
	if Data_Types[0] == 'C' or Data_Types[0] == 'L' or Data_Types[0] == 'I' or Data_Types[0] == 'V':
		Data_Types = [ Data_Types ]
	with open(ResIDList, 'r') as f:
		ResIDlist = [line.rstrip() for line in f]

	Title_dict = {}	
	if len(Sample) != 0:
		sample = Sample[0]
		for samp in Sample[1:]:
			sample = sample + ' ' + samp
		for src, target in Format_Name.iteritems():
			title = sample.replace(src, target)
		Title_dict[OutName]= title


####----------------------------------------------------------------------------------------------####
##																									##
## Using the MasterList information generate list of DataSet names (DataSetList) which serve as 	##
## column names in pandas dataframe and keys in dictionaries for plotting.  						##
## Formatted titles and legend entries and storing them in dictionaries with DataSet as keys 		##
## Finally assigning uniqu color to each dataset in the ColorsDict.									##
##																									##
####----------------------------------------------------------------------------------------------####


	ColorsDict = {}
	Legend_dict = {}
	Comparisons = []
	DataSetList = []
	DF_columns = ['resid', 'atom', 'atom_sort']	
	for DataSet in MasterList:
		DataSetList.append(DataSet[0])
		DF_columns.append(DataSet[0])
		xDataSet = DataSet[0]
		if len(DataSet) > 3:
			if DataSet[3][0] != "[":
				ColorsDict[DataSet[0]] = DataSet[3]
			if DataSet[3][0] == "[":
				color_temp = eval(DataSet[3])
				if color_temp[0] > 1.0 or color_temp[1] > 1.0 or color_temp[2] > 1.0:
					for i in range(len(color_temp)):
						val = color_temp[i]
						color_temp[i] = np.round(val/255.000,3)
			ColorsDict[DataSet[0]]=color_temp
		if len(DataSet) > 4:
			for i in range(len(DataSet))[4:]:
				comp = [xDataSet, DataSet[i]]
				Comparisons.append(comp)
		legend = DataSet[1]
		for src, target in Format_Name.iteritems():
			legend = legend.replace(src, target)
		Legend_dict[DataSet[0]] = legend

		if len(Sample) != 0:
			Title_dict[DataSet[0]] = title + ' ' + legend
		else: 
			Title_dict[DataSet[0]] = legend
## Remove reference data sets from plot list. Getting rid of white space. 
	Plot_DataSetList = []
	for DataSet in DataSetList:
		if DataSet not in Comparisons:
			Plot_DataSetList.append(DataSet)

## Make a list all possible combinations of resID and atoms to serve as index in DataFrame 
	SortList = []
	for res in ResIDlist:
		for atom in CS_cols:
			SortList.append(res + '_' + atom)

####----------------------------------------------------------------------------------------------####
##								Sorting Input Data in to pandas DataFrame 							##
##																									##
## 		For each type of data requested a separate pandas DataFrame object DF_DataType with index   ##
## set = ResID_atom and one column for each sample/condition (dataset). 							##
## The Sort_dic[key]=Sort_dic_full[key][Sort_dic_index[type]]statement selects the appropriate		##
## column from input file.																			##
## ResID, atom and atom_sort columns are used for plotting and sorting purposes  					##
####----------------------------------------------------------------------------------------------####


	for dtype in Data_Types:
		exec('DF_' +dtype + ' = pd.DataFrame(index=SortList)')

	for DataSet in MasterList:
		InputFiles = glob.glob(os.path.join(input_path + DataSet[2].strip()))
		for file in InputFiles:
			temp_file = open('NODA_temp_Data_input','w')
			Inheader = open(file).readline().rstrip()
			for src, target in replacements.iteritems():
				Inheader = Inheader.replace(src, target)
			Header = Inheader + ' \n'
			temp_file.write(Header)
			infile = open(file).readlines()
			for line in infile:
				if str('FORMAT') not in line or str('Assignment') not in line:
					for res in ResIDlist:
						line = line.replace(res, res + '   ')
					if len(line.rstrip().split()) <= len(Header.rstrip().split()):
						temp_file.write(line)
			temp_file.close()
			if 'lw3' in Header:
				Data_Types = ['CS'] 
			else: Data_Types = Data_Types
			for dtype in Data_Types:
				df = eval('DF_' +dtype)
				Sort_dic = {}
				for key in Sort_dic_full:
					if  len(Sort_dic_full[key]) >=	Sort_dic_index[dtype]:
						Sort_dic[key]=Sort_dic_full[key][Sort_dic_index[dtype]]
				in_df = pd.read_table('NODA_temp_Data_input', delim_whitespace=True)
				for i, row in in_df.iterrows():
	# 				## Since 1H chemcal shift can be repeated across multiple specra, not all specra are used to read in the 1H value, so
	# 				## some will only provide 15N or 13C informaiton. 
					if row['Atom'] in Sort_dic.keys():
						for y in range(len(Sort_dic[row['Atom']])):
							## Set the Row with index = ResID_Atom 
							df.loc[df.index == row['Group'] + '_' + Sort_dic[row['Atom']][y][0], 'resid'] = row['Group']
							df.loc[df.index == row['Group'] + '_' + Sort_dic[row['Atom']][y][0], 'atom'] = row['Atom'].split("-")[y]
							df.loc[df.index == row['Group'] + '_' + Sort_dic[row['Atom']][y][0], 'atom_sort'] = Sort_dic[row['Atom']][y][0]
							df.loc[df.index == row['Group'] + '_' + Sort_dic[row['Atom']][y][0], DataSet[0]] = np.around(row[Sort_dic[row['Atom']][y][1]],3)
	for dtype in Data_Types:
		df = eval('DF_'+dtype)
		df.dropna(axis=0,subset=[DataSetList[0]],inplace=True)
		print(dtype + ' Data sorted into DataFrame')

	DF_CS_original = DF_CS.copy(deep = True)
## Make a copy of the CS DataFrame to hold onto and use latter for assignments table, and correlation plots 

####----------------------------------------------------------------------------------------------####
##																									##
## 	  Calculate relative line width for all data sets relative to the first (reference) data set    ##
##																									##
####----------------------------------------------------------------------------------------------####
	for val in Data_Types:
		if val == 'LW':
			Ref_LW=eval('DF_LW').copy(deep=True)
			LW=eval('DF_LW')
			for i in range(len(DataSetList)):
				DataSet = DataSetList[i]
				Ref_DS = Comparisons[i]
				LW[DataSet]=LW[DataSet] / Ref_LW[Ref_DS]

####----------------------------------------------------------------------------------------------####
##																									##
##						Dictionary used for Internal Intensity/Volume normalization 				##
##  Peak intensity/volume normalization is done based on residue type (purine or pyrimidine) and 	##
##  atom identity. All purine (G's/ A's ) atoms are normalized relative to helical G specified in 	##
##  Intenstiy_Norm. With the exception of A-C2/H2 and A-N1/N3 which are normalized relative to the  ##
##  specified helical A. All pyrimidines (C's, U's, and T's) normalized relative to the specified 	##
##  helical U. 																						##
##																									##
##  	Data is normalized independently for each DataSet and not relative to reference DataSet 	##
## 																									##
####----------------------------------------------------------------------------------------------####

	if 'Intensity' in Data_Types:
		GA_Norm = {'C6C8':Intensity_Norm[0] + '_C6C8', 'C1p':Intensity_Norm[0] + '_C1p', 'C2':Intensity_Norm[2] + '_C2', 'NImino':Intensity_Norm[0] + '_NImino',
				   'N1N9':Intensity_Norm[0] + '_N1N9', 'N7':Intensity_Norm[0] + '_N7', 'AN1':Intensity_Norm[2] + '_AN1', 'AN3':Intensity_Norm[2] + '_AN3',
				   'C2p':Intensity_Norm[0] + '_C2p', 'C3p':Intensity_Norm[0] + '_C3p','C4p':Intensity_Norm[0] + '_C4p','C5p':Intensity_Norm[0] + '_C5p'}
		TCU_Norm = {'C6C8':Intensity_Norm[1] + '_C6C8','C1p':Intensity_Norm[1] + '_C1p','C5':Intensity_Norm[1] + '_C5','NImino':Intensity_Norm[1] + '_NImino','N1N9':Intensity_Norm[1] + '_N1N9',
				   'C2p':Intensity_Norm[1] + '_C2p','C3p':Intensity_Norm[1] + '_C3p','C4p':Intensity_Norm[1] + '_C4p','C5p':Intensity_Norm[1] + '_C5p'}
		Norm_Int_dict={'A':GA_Norm,'G':GA_Norm,'C':TCU_Norm, 'U':TCU_Norm, 'T':TCU_Norm}


		if val == 'Intensity' or val == 'Volume':
			Ref_Int=eval('DF_' + val).copy(deep=True)
			Int_df=eval('DF_' + val)
			for DataSet in DataSetList:
				for row in Int_df.index.tolist():
					res=row.split('_')[0]
					atom=row.split('_')[1]
					if atom in Norm_Int_dict[res[0]].keys():
						Norm=Norm_Int_dict[res[0]][atom]
						refval=Ref_Int.loc[Norm,DataSet] * 1.00
						Int_df.loc[row,DataSet] = Int_df.loc[row,DataSet] /refval
####----------------------------------------------------------------------------------------------####
##																									##
## 		Generating dictionaries to control axis limits and titles in scatter and bar plots 			##
## 		based on current data. This will keep everything plotted on the same relative scale			##
##																									##
####----------------------------------------------------------------------------------------------####


	Delta_ppm_dict = {'H':r"($^{1}$H, ppm)",'C':r"($^{13}$C, ppm)",'N':r"($^{15}$N, ppm)",'A':r"($^{15}$N, ppm)"}

	Summary_Plot_Dict = {}
	nuc_list = ['H','C','N']
	for val in Data_Types:
		for nuc in nuc_list:
			nuc_lsit_2, temp_min, temp_max=[], [], []
			df1 = eval('DF_' + val)
			for row in df1.index.tolist():
				if df1.loc[row, 'atom'][0] == nuc:
					nuc_lsit_2.append(row)
			if len(nuc_lsit_2) > 0:
				df2=eval('DF_'+ val).ix[nuc_lsit_2]
				for DataSet in DataSetList[1:]: 
					df2.fillna(value=0.0,axis=0,inplace=True)
					if len(df2[DataSet]) != 0:
						temp_min.append((min(df2[DataSet].dropna(axis=0,how='all')) *11.00))
						temp_max.append((max(df2[DataSet].dropna(axis=0,how='all')) *11.00))
				if len(temp_min) != 0:		
					if val == 'CS':
						Summary_Plot_Dict[nuc + val]=[[np.round((min(temp_min)),1)/ 10.0, np.round((max(temp_max)),1) / 10.0], r"$\Delta\omega$"+ Delta_ppm_dict[nuc]]
					elif val == 'LW':
						Summary_Plot_Dict[nuc + val]=[[0.0, np.round((max(temp_max)),1) / 10.0], r"LW/LW$_{0}$ "]
					elif val == 'Intensity':
						Summary_Plot_Dict[nuc + val]=[[0.0, np.round((max(temp_max)),1) / 10.0], r"$\bar I$ (au)"]
					elif val == 'Volume':
						Summary_Plot_Dict[nuc + val]=[[0.0, np.round((max(temp_max)),1) / 10.0], r"$\bar V$ (au)"]
	Bar_plot_dict = {}
	for val in Data_Types:
		df = eval('DF_' + val)
		AtomsList=df['atom_sort'].unique().tolist()
		for atom in AtomsList:
			if atom in Format_Name.keys():
				fatom = Format_Name[atom]
			else: fatom = atom 
			temp_min, temp_max=[], []
			df3 = df[df.atom_sort == atom].copy(deep=True)
			for DataSet in DataSetList[1:]:
				df3.fillna(value=0.0,axis=0,inplace=True)
				if len(df3[DataSet]) != 0:
					temp_min.append((min(df3[DataSet].dropna(axis=0,how='all')) *11.00))
					temp_max.append((max(df3[DataSet].dropna(axis=0,how='all')) *11.00))
			## only for populated list create dictionary entry 
			if len(temp_min) != 0:	
				if val == 'CS':
					Bar_plot_dict[atom+val]=[[np.round((min(temp_min)),1)/ 10.0,np.round((max(temp_max)),1) / 10.0], r"$\Delta$" + fatom + ' '+ Delta_ppm_dict[atom[0]]]
				elif val == 'LW':
					Bar_plot_dict[atom+val]=[[0.0,np.round((max(temp_max)),1) / 10.0], r"LW/LW$_{0}$ "+ fatom + " (au)"]
				elif val == 'Intensity':
					Bar_plot_dict[atom+val]=[[0.0,np.round((max(temp_max)),1) / 10.0], r"$\bar I$ "+ fatom + " (au)"]
				elif val == 'Volume':
					Bar_plot_dict[atom+val]=[[0.0,np.round((max(temp_max)),1) / 10.0], r"$\bar V$ "+ fatom + " (au)"]

####----------------------------------------------------------------------------------------------####
##																									##
##			 	Setting controlling plot appearance: Font, line widths, ticks, and legend  			##
##																									##
####----------------------------------------------------------------------------------------------####

	mpl.rcParams['pdf.fonttype']=42
	mpl.rcParams['ps.fonttype'] = 42
	mpl.rcParams['font.sans-serif']='arial'
	mpl.rcParams['font.size']=10
	mpl.rcParams['axes.linewidth']=2
	mpl.rcParams['xtick.direction']=mpl.rcParams['ytick.direction']='out'
	mpl.rcParams['xtick.labelsize']=mpl.rcParams['ytick.labelsize']=10
	mpl.rcParams['xtick.major.size']=mpl.rcParams['ytick.major.size']=6
	mpl.rcParams['xtick.major.width']=mpl.rcParams['ytick.major.width']=2
	mpl.rcParams['xtick.minor.size']=4
	mpl.rcParams['xtick.minor.width']=1
	mpl.rcParams['axes.spines.right']=False
	mpl.rcParams['axes.spines.top']=False
	mpl.rcParams['legend.fontsize']=10
	mpl.rcParams['legend.loc']='best'
	mpl.rcParams['legend.borderpad']=0.01
	mpl.rcParams['legend.frameon']=False
	mpl.rcParams['legend.handlelength']=0
	mpl.rcParams['legend.scatterpoints']=1
	mpl.rcParams['xtick.major.bottom']=mpl.rcParams['ytick.major.left'] = True
	mpl.rcParams['xtick.major.top']=mpl.rcParams['ytick.major.right'] = True
	mpl.rcParams['mathtext.fontset']= 'cm'
	mpl.rcParams['mathtext.sf']= 'sans\\-serif'

	AtomsList = DF_CS['atom_sort'].unique().tolist()
	for Comp in Comparisons: 
		if Comp[0] != Comp[1]:
			pdf = PdfPages(outpath + Comp[0].replace(' ','_') + '_' + Comp[1].replace(' ','_')  + '_correlation_plot.pdf')
			for atom in AtomsList:
				x1 = np.array([])
				y1 = np.array([])
				df1 = DF_CS[DF_CS['atom_sort'] == atom].copy(deep=True)
				reslist = df1.dropna(axis=0,how='any').index.tolist()
				if len(reslist) != 0:
					for res in reslist:
						x1 = np.append(x1,df1.loc[res,Comp[0]])
						y1 = np.append(y1,df1.loc[res,Comp[1]])
					fig=plt.figure(figsize=(4,4))
					ax = fig.add_subplot(1,1,1)
					ax.plot(np.array(df1[Comp[0]]),np.array(df1[Comp[0]]),linewidth = 2, color = [0,0,0], label = None, zorder = 1)
					ax.scatter(np.array(df1[Comp[0]]),np.array(df1[Comp[1]]), color=ColorsDict[Comp[0]], marker='o', s=30, label = None, clip_on=False, zorder = 2)
					df1.dropna(subset=[Comp[0],Comp[1]])
					axismin = math.floor(min([df1[Comp[0]].min(),df1[Comp[1]].min()]))
					axismax = math.ceil(max([df1[Comp[0]].max(),df1[Comp[1]].max()]))
					if (axismax - axismin) >= 8.0:
						ax.set_xlim(np.round(axismin,0) -0.5, np.round(axismax,0) + 0.5)
						ax.set_ylim(np.round(axismin,0) -0.5, np.round(axismax,0) + 0.5)
						ax.xaxis.set_ticks(np.arange(np.round(axismin,0), np.round(axismax,0), 2.0))
						ax.yaxis.set_ticks(np.arange(np.round(axismin,0), np.round(axismax,0), 2.0))
					if (axismax - axismin) <= 5 and (axismax - axismin) > 2.5:
						ax.set_xlim(np.round(axismin,0) -0.5, np.round(axismax,0) + 0.5)
						ax.set_ylim(np.round(axismin,0) -0.5, np.round(axismax,0) + 0.5)
						ax.xaxis.set_ticks(np.arange(np.round(axismin,0), np.round(axismax,0), 1.0))
						ax.yaxis.set_ticks(np.arange(np.round(axismin,0), np.round(axismax,0), 1.0))
					if atom[0] == 'H':
						ax.set_xlim(np.round(axismin,0) -0.2, np.round(axismax,0) + 0.2)
						ax.set_ylim(np.round(axismin,0) -0.2, np.round(axismax,0) + 0.2)
						ax.xaxis.set_ticks(np.arange(np.round(axismin,0), np.round(axismax,0), 0.2))
						ax.yaxis.set_ticks(np.arange(np.round(axismin,0), np.round(axismax,0), 0.2))
					ax.set_xlabel(atom.replace('p',"'") + " " + Delta_ppm_dict[atom[0]] + '\n' + Legend_dict[Comp[0]])
					ax.set_ylabel(Legend_dict[Comp[1]] + '\n' +  atom.replace('p',"'") + " " + Delta_ppm_dict[atom[0]])
					L = np.dot(x1,y1) / np.dot(y1,y1)
					correlation = np.round(np.corrcoef(y1,x1)[0,1],5)
					chi2 = sum((L*y1-x1)**2)
					rmsd = np.round(np.sqrt(chi2/len(y1)),3)
					q = np.sqrt(chi2) / np.sqrt(sum(x1**2))
					mytext = "RMSD = "+str(rmsd)+' ppm'+'\n'+"R ="+str(correlation)
					ax.text(0.05,0.90, mytext, verticalalignment='center', fontsize=10, transform=ax.transAxes)
					plt.axis('equal')
					for i in range(len(df1['resid'].tolist())):
						if atom[0] == 'C':
							ax.annotate(df1['resid'].tolist()[i],(np.array(df1[Comp[0]])[i] + 0.2 ,np.array(df1[Comp[1]])[i] - 0.1),fontsize = 8)
						if atom[0] == 'H':
							ax.annotate(df1['resid'].tolist()[i],(np.array(df1[Comp[0]])[i] + 0.01 ,np.array(df1[Comp[1]])[i] - 0.01),fontsize = 8)
						if atom[0] == 'N':
							ax.annotate(df1['resid'].tolist()[i],(np.array(df1[Comp[0]])[i] + 0.2 ,np.array(df1[Comp[1]])[i]),fontsize = 8)
					fig.tight_layout()
					pdf.savefig()
					plt.close()
			pdf.close()	
	
	Correlation_Combinations = [['C6C8','C1p'],['C6C8','N7'],['C1p','C4p'],['C6C8','N1N9']]
	AtomsList = DF_CS['atom_sort'].unique().tolist()
	df = DF_CS
	for pair in Correlation_Combinations: 
		if pair[0] in AtomsList and pair[1] in AtomsList:
			pdf = PdfPages(outpath + OutName + "_" + pair[0] + '_' + pair[1] + '_correlation_plot.pdf')
			temp1 , temp2 = [], []
			for res in ResIDlist:
				temp1.append(res + "_" + pair[0])
				temp2.append(res + "_" + pair[1])
			df1 = df.ix[temp1]
			df2 = df.ix[temp2]
			x_max, y_max = [], []
			x_min, y_min = [], []
			fig=plt.figure(figsize=(4,4))
			ax = fig.add_subplot(1,1,1)
			colors = []
			for DataSet in DataSetList:
				colors.append(ColorsDict[DataSet])
				ax.scatter(np.array(df1[DataSet]),np.array(df2[DataSet]),color=ColorsDict[DataSet],marker='o',s=30, label = Legend_dict[DataSet],clip_on=False)
				df1.dropna(subset=[DataSet])
				df2.dropna(subset=[DataSet])
				x_max.append(df1[DataSet].max())
				y_max.append(df2[DataSet].max())
				x_min.append(df1[DataSet].min())
				y_min.append(df2[DataSet].min())
			legend = ax.legend(loc='best', frameon=False, markerscale=0.000001)
			for color,text in zip(colors, legend.get_texts()):
				text.set_color(color)
			ax.set_xlim(min(x_min) - 0.2, max(x_max) + 0.2 )
			ax.set_ylim(min(y_min) - 0.2, max(y_max) + 0.2)
			ax.set_xlabel(pair[0].replace('p',"'") + " " + Delta_ppm_dict[pair[0][0]])
			ax.set_ylabel(pair[1].replace('p',"'") + " " + Delta_ppm_dict[pair[1][0]])
			ax.set_title(Title_dict[OutName])
			plt.axis('equal')
			for i in range(len(df1['resid'].tolist())):
				ax.annotate(df1['resid'].tolist()[i],(np.array(df1[DataSet])[i] + 0.05 ,np.array(df2[DataSet])[i] + 0.05),fontsize = 8)
			fig.tight_layout()
			pdf.savefig()
			plt.close()
			pdf.close()	

	for DataSet in DataSetList:
			os.system("rm %s" % DataSet)
	print("Finished")