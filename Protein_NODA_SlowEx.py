import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import trim_mean
import matplotlib.ticker as ticker 
from scipy.stats import tstd
from scipy.stats import norm
import re
import math
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy.optimize import leastsq

##########################################################################################
## Developed by Mary C Clay 
## e-mail: mary.clay@stjude.org
## St Jude Children's Research Hospital 
## Department of Structural Biology
##
## Last updated: November 24, 2020
## 
########################################################################################## 

########################################################################################## 
## Dictionaries for sorting input data into pandas Data Frame, and formatting of supper 
## and subscript, and primes for sugars
########################################################################################## 
Allowed_atoms = {
 'A': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CMe', 'HB':'HMe'},
 'C': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB'},
 'D': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB'},
 'E': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG2':'HG', 'HG3':'HG'},
 'F': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CD1':'CDAro', 'HD1':'HDAro', 'CD2':'CDAro', 'HD2':'HDAro', 'CE1':'CEAro', 'HE1':'HEAro', 'CE2':'CEAro', 'HE2':'HEAro', 'CZ':'CZAro', 'HZ':'HZAro'},
 'G': {'H':'H', 'N':'N', 'CA':'CA', 'HA2':'HA', 'HA3':'HA'},
 'H': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'ND1':'Nside', 'HD1':'Hside', 'CD2':'CAro', 'HD2':'HDAro', 'CE1':'CAro', 'HE1':'HEAro', 'NE2':'Nside', 'HE2':'Hside'},
 'I': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA','HB':'HB', 'CB':'CB', 'CG1':'CG', 'HG12':'HG', 'HG13':'HG', 'CG2':'CMe', 'HG2':'HMe', 'CD1':'CMe', 'HD1':'HMe'},
 'K': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG2':'HG', 'HG3':'HG', 'CD':'CD','HD2':'HD', 'HD3':'HD', 'CE':'CE', 'HE2':'HE', 'HE3':'HE', 'NZ':'Nside', 'HZ':'Hside'},
 'L': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG':'HG', 'CD1':'CMe', 'HD1':'HMe', 'CD2':'CMe', 'HD2':'HMe'},
 'M': {'H':'H', 'N':'N', 'HA':'HA', 'HB2':'HB', 'HB3':'HB', 'HG2':'HG', 'HG3':'HG', 'HE':'HMe', 'CA':'CA', 'CB':'CB', 'CE':'CMe', 'CG':'CG'},
 'N': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'ND2':'Nside', 'HD21':'Hside', 'HD22':'Hside'},
 'P': {'CA':'CA','HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG2':'HG', 'HG3':'HG', 'CD':'CD','HD2':'HD', 'HD3':'HD'},
 'Q': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG2':'HG', 'HG3':'HG', 'NE2':'Nside', 'HE21':'Hside', 'HE22':'Hside'},
 'R': {'H':'H', 'N':'N', 'HA':'HA', 'HB2':'HB', 'HB3':'HB', 'HD2':'HD', 'HD3':'HD', 'CG':'CG', 'HG2':'HG', 'HG3':'HG','NH1':'Nside', 'HH11':'Hside', 'HH12':'Hside', 'NH2':'Nside', 'HH21':'Hside', 'HH22':'Hside',  'CA':'CA', 'CB':'CB', 'CD':'CD'},
 'S': {'H':'H', 'N':'N', 'CA':'CA','HA':'HA','CB':'CB', 'HB2':'HB', 'HB3':'HB'},
 'T': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB':'HB', 'CG2':'CMe', 'HG2':'HMe'},
 'V': {'H':'H', 'N':'N', 'CA':'CA','HA':'HA', 'CB':'CB', 'HB':'HB', 'CG1':'CMe', 'HG1':'HMe', 'CG2':'CMe', 'HG2':'HMe'},
 'W': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA','CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CD1':'CDAro', 'HD1':'HDAro', 'NE1':'Nside', 'HE1':'Hside', 'CE3':'CEAro', 'HE3':'HEAro', 'CH2':'CHAro', 'HH2':'HEAro', 'CZ2':'CZAro', 'HZ2':'HZAro', 'CZ3':'CZAro', 'HZ3':'HZAro'},
 'Y': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CD1':'CDAro', 'HD1':'HDAro', 'CD2':'CDAro', 'HD2':'HDAro', 'CE1':'CEAro', 'HE1':'HEAro', 'CE2':'CEAro', 'HE2':'HEAro'}
 }

Format_Name = {'uM':r'$\mu$M','MnCl2':r'MnCl$_{2}$','MgCl2':r'MgCl$_{2}$','CaCl2':r'CaCl$_{2}$','ZnCl2':r'ZnCl$_{2}$','BaCl2':r'BaCl$_{2}$',"H2O":r"H$_{2}$O","D2O":r"D$_{2}$O", 'alpha':r"$\alpha$", 'beta':r"$\beta$",'delta':r"$\Delta$"}

####----------------------------------------------------------------------------------------------####
##										Fitting Equation											##
####----------------------------------------------------------------------------------------------####

def Fit_Slow_dimer_kd(x,Kd,a):

	return a*(((4*x + Kd)-np.sqrt(pow(4*x + Kd,2)-16*x*x))/(4*x))

def Fit_Slow_mono_kd(x,Kd,a):

	return (1 - a*(((4*x + Kd)-np.sqrt(pow(4*x + Kd,2)-16*x*x))/(4*x)))


def Fitdata(RunParams, MasterList, outpath):

####----------------------------------------------------------------------------------------------####
##																									##
## 	Generates all the run parameters from the *_input.txt file									 	##
##	RunParams for CSP : OutDir, OutName, Sequence, Start_Index, Residues, Data_Types, SDM, 			##
##						Show_Lables,Intensity_Norm(s), input_path									##
##																									##
####----------------------------------------------------------------------------------------------####

	for i in range(len(RunParams)):
		if len(RunParams[i]) == 2:
			exec(RunParams[i][0] + '= RunParams[i][1]')
		else:
			exec(RunParams[i][0] + '= RunParams[i][1:len(RunParams[i])]')
	# Just in case only one type of analysis is requested. 
	if isinstance(Data_Types, str): Data_Types = [ Data_Types ]

### Read in the sequence file in any formate as long as it is not numbered and generates a numbered sequence (Seq) with single letter abbreviations for the amino acid types
	sequence = ''
	Seq = []
	with open(Sequence, 'r') as my_seq:
		for line in my_seq:
			if ">" not in line:
				if len(line.rstrip()) == 3 and line.rstrip() in L3_to_1L.keys():
					line2 = L3_to_1L[line.rstrip()]
					sequence = sequence + str(line2) 
				else:
					line2 = line.rstrip()
					sequence = sequence + str(line2) 
	for i in range(len(sequence)):
		Seq.append(sequence[i]+str(int(Start_Index) + int(i)))
## This pulse out the residues the user has indicated that they want to use in the Residues entry of the input file
	ResIDlist = []
	for i in range(len(Seq)): 
		if int(Seq[i][1:])>= int(Residues.split('-')[0]) and int(Seq[i][1:]) <= int(Residues.split('-')[1]):
			ResIDlist.append(Seq[i])

####----------------------------------------------------------------------------------------------####
##																									##
##								Sorting Input Data in to pands DataFrame 							##
##																									##
## 		For each type of data requested a separate pandas DataFrame object DF_DataType with index   ##
## set = ResID_atom and one column for each sample/condition (dataset). 							##
## The Sort_dic[key]=Sort_dic_full[key][Sort_dic_index[type]]statement selects the 					##
## appropriate column from input file.																##
## resid, atom and atom_sort columns are used for plotting and sorting purposes  					##
## 																									##
####----------------------------------------------------------------------------------------------####
	## creating empty data frame to store results in, must exist before use
	DF_columns = ['resid', 'atom', 'atom_sort', 'kd', 'kd_err','scale','scale_err', 'chi2']
	for dtype in Data_Types:
		exec('DF_temp' + dtype + " = pd.DataFrame(columns=['resid', 'resi', 'name', 'nuc','atom_sort','state', 'kd', 'kd_err','scale','scale_err','chi2'])")
	DataSetList = []
	ConcentrationsList = []
	## Pars the input sparky list and sort data into approreate DataFrame 
	for DataSet in MasterList:
		InputFiles = glob.glob(os.path.join(input_path + DataSet[2].strip()))
		DataSetList.append(DataSet[0])
		ConcentrationsList.append(float(DataSet[1].split()[0]))
		for file in InputFiles:
			print("Reading in {:}".format(file.split('/')[-1]))
			in_df = pd.read_csv(file,sep=r'\s*',names=open(file).readline().rstrip().replace('Data Height', "Intensity").replace('Volume', "Volume  fit").replace('(hz)', "").split(),skiprows=[0,1],engine='python')
			if in_df.shape[0] < 2: print('incorrect file path {:}'.format(file.split('/')[-1]))
			for i, row in in_df.iterrows():
				if '?' not in row['Assignment']:
					if re.findall('[A-Z]([0-9]*)[a-z][A-Z]', row['Assignment']):
						group = re.search('[A-Z]([0-9]*)[a-z]', row['Assignment']).group()
						resid = group[:-1]
					if re.findall('[A-Z]([0-9]*)[A-Z]', row['Assignment']) and not re.findall('[A-Z]([0-9]*)[a-z][A-Z]', row['Assignment']):
						group = re.search('[A-Z]([0-9]*)', row['Assignment']).group()
						resid = group
					Assignment =  row['Assignment'].replace(group, group + '_')
					in_df.loc[i, 'Group'] = Assignment.split('_')[0]
					in_df.loc[i, 'Atoms'] = Assignment.split('_')[1]
					in_df.loc[i, 'resid'] = resid
					in_df.loc[i, 'resi'] = int(resid[1:])
					if re.search('[a-z]',group):
						in_df.loc[i, 'state'] = group[-1]
			for dtype in Data_Types:
				df = eval('DF_temp' +dtype)
				for i, row in in_df.iterrows():
					Sorting_dict = Allowed_atoms[row['Group'][0]]
					if dtype == 'CS' or dtype == 'LW':
						for y in range(len(row['Atoms'].split('-'))):
							if row['Atoms'].split('-')[y] in Sorting_dict.keys():
								dfidx = row['Group'] + '_' + row['Atoms'].split("-")[y]
								df.loc[dfidx , 'resid'] = row['resid']
								df.loc[dfidx , 'resi'] = row['resi']
								if 'state' in in_df.columns.tolist(): df.loc[dfidx , 'state'] = row['state']
								df.loc[dfidx , 'nuc'] = row['Atoms'].split("-")[y][0]
								df.loc[dfidx , 'name'] = row['Atoms'].split("-")[y]
								df.loc[dfidx , 'atom_sort'] = Sorting_dict[row['Atoms'].split('-')[y]]
								df.loc[dfidx , DataSet[0]] = np.around(row[dtype.replace('CS','w').replace('LW','lw')+ str(y+1)],3)
					if dtype == 'Intensity' or dtype == 'Volume':
						if row['Atoms'].split('-')[0] in Sorting_dict.keys():
							dfidx = row['Group'] + '_' + row['Atoms'].split("-")[0]
							df.loc[dfidx , 'resid'] = row['resid']
							df.loc[dfidx , 'resi'] = row['resi']
							if 'state' in in_df.columns.tolist(): df.loc[dfidx , 'state'] = row['state']
							df.loc[dfidx , 'nuc'] = row['Atoms'].split("-")[0][0]
							df.loc[dfidx , 'name'] = row['Atoms'].split("-")[0]
							df.loc[dfidx , 'atom_sort'] = Sorting_dict[row['Atoms'].split('-')[0]]
							df.loc[dfidx , DataSet[0]] = np.around(row[dtype],3)

	## Creating a clean copy of the data frame omitting my rows that are not populated by the reference (first) data set 
	for dtype in Data_Types:
		df = eval('DF_temp'+ dtype)
		df.sort_values(by=['resi','state'],inplace=True)
		exec('DF_' + dtype + ' = df')
		df_used = eval('DF_' + dtype)
		print(dtype + ' Data sorted into DataFrame')

####----------------------------------------------------------------------------------------------####
##																									##
##									Intensity/Volume normalization 									##
##  Peak intensity/volume normalization is done based on trimed mean of the observed values 		##
##  Trimed mean exclueds the lower and upper 10% of the gusian distribution of the observed values	##
##																									##
##  	Data is normalized independently for each DataSet and not relative to reference DataSet 	##
## 																									##
####----------------------------------------------------------------------------------------------####
	Normaliz = {}
	for dtype in Data_Types:
		if dtype == 'Intensity' or dtype == 'Volume': 
			df = eval('DF_' +dtype)
			for nuc in df['nuc'].unique().tolist():
				df1 = df[df['nuc']== nuc ].copy(deep = True)
				for DataSet in DataSetList:
					values = [np.round(abs(val),4) for val in df1[DataSet].dropna().tolist() if val != 0.0]
					Normaliz[DataSet+'_'+nuc]=trim_mean(values,0.1)
			for i, row in df.iterrows():
				for DataSet in DataSetList:
					df.loc[i,DataSet] = df.loc[i,DataSet] / Normaliz[DataSet+'_'+row['nuc']]
			idxlist = df.index.tolist()
			idxlist.append(idxlist[-1])
			for DataSet in DataSetList:
				for x in range(len(idxlist))[:-1]:
					if (df.loc[idxlist[x],'resi'] == df.loc[idxlist[x+1],'resi']) and (df.loc[idxlist[x],'name'] == df.loc[idxlist[x+1],'name']) and (idxlist[x] != idxlist[x+1]):
						if re.search('[a-z]',idxlist[x].split('_')[0][-1]) and re.search('[a-z]',idxlist[x+1].split('_')[0][-1]):
							df.loc[df.loc[idxlist[x],'resid']+'_'+df.loc[idxlist[x],'name'], 'resid'] = df.loc[idxlist[x],'resid']
							df.loc[df.loc[idxlist[x],'resid']+'_'+df.loc[idxlist[x],'name'], 'resi'] = df.loc[idxlist[x],'resi']
							df.loc[df.loc[idxlist[x],'resid']+'_'+df.loc[idxlist[x],'name'], 'name'] = df.loc[idxlist[x],'name']
							df.loc[df.loc[idxlist[x],'resid']+'_'+df.loc[idxlist[x],'name'], 'atom_sort'] = df.loc[idxlist[x],'atom_sort']
							if not pd.isnull(df.loc[idxlist[x],DataSet]) and not pd.isnull(df.loc[idxlist[x+1],DataSet]):
								df.loc[df.loc[idxlist[x],'resid']+'_'+df.loc[idxlist[x],'name'], DataSet] = df.loc[idxlist[x],DataSet] + df.loc[idxlist[x+1],DataSet]
								df.loc[idxlist[x],DataSet] = df.loc[idxlist[x],DataSet]/ df.loc[df.loc[idxlist[x],'resid']+'_'+df.loc[idxlist[x],'name'], DataSet]
								df.loc[idxlist[x+1],DataSet] = df.loc[idxlist[x+1],DataSet]/ df.loc[df.loc[idxlist[x],'resid']+'_'+df.loc[idxlist[x],'name'], DataSet]
	for dtype in Data_Types:
		if dtype == 'Intensity' or dtype == 'Volume': 
			df = eval('DF_' +dtype)
			for res in df.index.tolist():
				Yvals = []
				yvals = np.array(df.ix[res]).tolist()[11:]
				for i in range(len(yvals)):
					if not np.isnan(yvals[i]):
						Yvals.append(yvals[i])
				if max(Yvals) > 1.0:
					refval = max(Yvals)
					for DataSet in DataSetList:
						df.loc[res,DataSet] = df.loc[res,DataSet]/refval

		print('Finished {:} Analysis'.format(dtype))
####----------------------------------------------------------------------------------------------####
##																									##
##												Fitting Data 										##
##	If the residue as two states then the data will be fit. The fitting equation (func) is chosen 	##
##	based on the trend observed in the intensity changes. If the last value is larger than the 		##
##	inital value then it is assumed that this represents dimer formation							##
##	Scipy.curve_fit uses a least suares method the fit the data, and provides the inial gess for 	##
##	Monte Carlo error analysis where the standard deviation in the fit residuals is used to perfom	##
##	random noise corruption of the data to determine the error in the fit. Chi suared is calaulated	##
##	using the resulting value of kD. 																##
##																									##
####----------------------------------------------------------------------------------------------####
	for dtype in Data_Types:
		FitList = []
		Residual= []
		df = eval('DF_'+dtype)
		for res in df.index.tolist():
			if not pd.isnull(df.loc[res, 'state']):
				## Convert concentration and intensity list to numpy array for fitting 
				Yvals, xvals = [], []
				yvals = np.array(df.ix[res]).tolist()[11:]
				for i in range(len(yvals)):
					if not np.isnan(yvals[i]):
						Yvals.append(yvals[i])
						xvals.append(ConcentrationsList[i])
				Xvals = np.array(xvals) 
				FitList.append(res)
				if Yvals[-1] > Yvals[0]:
					func = Fit_Slow_dimer_kd
				if Yvals[-1] < Yvals[0]:
					func = Fit_Slow_mono_kd
				## Fit_bound = [[low_kd,low_scale],[ high_kd,high_scale]]
				fit_bounds=[[0.000000001,0.1*min(Yvals)],[2*max(Xvals),10*max(Yvals)]]
				fit, matcov = curve_fit(func,Xvals,Yvals,bounds= fit_bounds,method='trf')
				df.loc[res,'kd']= np.round(fit[0],3)
				df.loc[res,'scale']= np.round(fit[1],3)
				residuals = (Yvals - func(Xvals,fit[0],fit[1]))
				for val in residuals:
					Residual.append(val)
		# Residual = [x for x in Residual if abs(x) <= 2 *trim_mean(abs(np.array(Residual)),0.01)]
		print("\nResiduals Calculated {:}".format(dtype))
		(mu, sigma) = norm.fit(Residual)
		for res in FitList: 
			mc_kd, mc_scale = [], []
			Yvals, xvals = [], []
			yvals = np.array(df.ix[res]).tolist()[11:]
			for i in range(len(yvals)):
				if not np.isnan(yvals[i]):
					Yvals.append(yvals[i])
					xvals.append(ConcentrationsList[i])
			Xvals = np.array(xvals) 
			MC_bounds = [[df.loc[res,'kd']/20,df.loc[res,'scale']/20], [20*abs(df.loc[res,'kd']),20*abs(df.loc[res,'scale'])]]
			if Yvals[-1] > Yvals[0]:
				func = Fit_Slow_dimer_kd
			if Yvals[-1] < Yvals[0]:
				func = Fit_Slow_mono_kd
			for y in range(300):
				val_noise =np.ndarray.tolist(Yvals + np.random.randn(len(Yvals)) * sigma)
				MCfit, matcov = curve_fit(func, Xvals, val_noise, p0=[df.loc[res,'kd'],df.loc[res,'scale']],bounds=MC_bounds)
				mc_kd.append(MCfit[0])
				mc_scale.append(MCfit[1])
			df.loc[res, 'kd'] = np.round(np.average(mc_kd),3)
			df.loc[res, 'kd_err'] = np.round(np.std(mc_kd),3)
			df.loc[res, 'scale'] = np.round(np.average(mc_scale),3)
			df.loc[res, 'scale_err'] = np.round(np.std(mc_scale),3)
			(chi2, p) = chisquare(Yvals,func(Xvals,df.loc[res,'kd'],df.loc[res,'scale']))
			df.loc[res,'chi2'] = np.round(chi2,3)
			print("{:}\nkD =  {:7.3f} +/- {:1.3f}\nscale =  {:7.3f} +/- {:1.3f}\nchi2 {:1.3f}".format(res, df.loc[res,'kd'], df.loc[res, 'kd_err'], df.loc[res,'scale'], df.loc[res, 'scale_err'], df.loc[res,'chi2']))


####----------------------------------------------------------------------------------------------####
##					 Save these results to a CSV file for each Data Type 							##
####----------------------------------------------------------------------------------------------####
	out_Columns = ['state', 'kd', 'kd_err','scale','scale_err', 'chi2']
	out_Columns.extend(DataSetList)
	for dtype in Data_Types:
		DF = eval('DF_' + dtype)
		DF = DF.sort_values(by=['resi','state'])
		if not os.path.exists(outpath + 'CSV_Files/'):
			os.makedirs(outpath + 'CSV_Files/')
		DF.to_csv(outpath + 'CSV_Files/' +  OutName + '_' + dtype + '_results.csv', columns = out_Columns)



def Plot(outpath, RunParams, ConcentrationsList, unit):

	for i in range(len(RunParams)):
		if len(RunParams[i]) == 2:
			exec(RunParams[i][0] + '= RunParams[i][1]')
		else:
			exec(RunParams[i][0] + '= RunParams[i][1:len(RunParams[i])]')
	# Just in case only one type of analysis is requested. 
	if isinstance(Data_Types, str): Data_Types = [ Data_Types ]

### Read in the sequence file in any formate as long as it is not numbered and generates a numbered sequence (Seq) with single letter abbreviations for the amino acid types
	sequence = ''
	Seq = []
	with open(Sequence, 'r') as my_seq:
		for line in my_seq:
			if ">" not in line:
				if len(line.rstrip()) == 3 and line.rstrip() in L3_to_1L.keys():
					line2 = L3_to_1L[line.rstrip()]
					sequence = sequence + str(line2) 
				else:
					line2 = line.rstrip()
					sequence = sequence + str(line2) 
	for i in range(len(sequence)):
		Seq.append(sequence[i]+str(int(Start_Index) + int(i)))
## This pulse out the residues the user has indicated that they want to use in the Residues entry of the input file
	ResIDlist = []
	for i in range(len(Seq)): 
		if int(Seq[i][1:])>= int(Residues.split('-')[0]) and int(Seq[i][1:]) <= int(Residues.split('-')[1]):
			ResIDlist.append(Seq[i])

####----------------------------------------------------------------------------------------------####
##																									##
##			 	Setting controlling plot appearance: Font, line widths, ticks, and legend  			##
##																									##
####----------------------------------------------------------------------------------------------####
	linewidths = 1.5
	mpl.rcParams['pdf.fonttype'] = 42
	mpl.rcParams['font.sans-serif'] = 'arial'
	mpl.rcParams['font.size'] = 10
	mpl.rcParams['axes.linewidth'] = linewidths
	mpl.rcParams['xtick.direction'] = mpl.rcParams['ytick.direction']='out'
	mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize']=10
	mpl.rcParams['xtick.major.size'] = 6
	mpl.rcParams['ytick.major.size'] = 5
	mpl.rcParams['xtick.major.width'] = mpl.rcParams['ytick.major.width']=linewidths
	mpl.rcParams['xtick.minor.size'] = 3
	mpl.rcParams['xtick.minor.width'] = linewidths
	mpl.rcParams['axes.spines.right'] = False
	mpl.rcParams['axes.spines.top'] = False
	mpl.rcParams['legend.fontsize'] = 10
	mpl.rcParams['legend.loc'] = 'best'
	mpl.rcParams['legend.borderpad'] = 0.01
	mpl.rcParams['legend.frameon'] = False
	mpl.rcParams['legend.handlelength'] = 0
	mpl.rcParams['legend.scatterpoints'] = 1
	mpl.rcParams['xtick.major.bottom'] = mpl.rcParams['ytick.major.left'] = True
	mpl.rcParams['xtick.major.top'] = mpl.rcParams['ytick.major.right'] = True
	mpl.rcParams['mathtext.fontset'] = 'cm'
	mpl.rcParams['mathtext.sf'] = 'sans\\-serif'
	plt.rcParams['mathtext.default'] = 'regular'
	mpl.rcParams['xtick.major.pad']=mpl.rcParams['ytick.major.pad']= 2
	mpl.rcParams['axes.labelpad'] = 3
	mpl.mathtext.FontConstantsBase.sup1 = 0.25

####----------------------------------------------------------------------------------------------####
## 			Bar plots showing CSP  for each atom observed, with a separate file for each 			##
##	individual dataset. Basically breaking 	Name_data_type_bar_plot.pdf file up. 					##
##	Saved as DataSet_data_type_bar_plot.pdf 														##																								##
####----------------------------------------------------------------------------------------------####
	plot_size = [3, 3]
	PCons = np.array(ConcentrationsList)
	for dtype in Data_Types:
		pdf = PdfPages(outpath+OutName +'_'+dtype+'_all_traj_log.pdf')
		df = pd.read_csv(outpath+'CSV_Files/' + OutName + '_' + dtype + '_results.csv',index_col=0)
		Plot_list = df.index.tolist()
		num_rows = math.ceil(len(Plot_list)/5.00)
		fig = plt.figure(figsize=(5*plot_size[0],num_rows*plot_size[1]))
		simcons=np.linspace(0.0001, 2*max(PCons),10000)
		for i in range(len(Plot_list)):
			gridloc = 0
			gridloc = i+1
			ax = fig.add_subplot(num_rows,5,gridloc)
			Yvals, xvals = [], []
			yvals = df.ix[Plot_list[i]].tolist()[6:]
			for x in range(len(yvals)):
				if not np.isnan(yvals[x]):
					Yvals.append(yvals[x])
					xvals.append(ConcentrationsList[x])
			Xvals = np.array(xvals) 
			Yvals = np.array(Yvals)
			if Yvals[-1] > Yvals[0]:
				func = Fit_Slow_dimer_kd
			if Yvals[-1] < Yvals[0]:
				func = Fit_Slow_mono_kd
			if df.loc[Plot_list[i],'kd'] > 0:
				ax.plot(simcons,func(simcons, df.loc[Plot_list[i],'kd'],df.loc[Plot_list[i],'scale']), '-',color=[1,0,0],linewidth = 2)
				ax.fill_between(simcons,func(simcons,df.loc[Plot_list[i],'kd']+df.loc[Plot_list[i],'kd_err'],df.loc[Plot_list[i],'scale']+df.loc[Plot_list[i],'scale_err']),
							func(simcons,df.loc[Plot_list[i],'kd']-df.loc[Plot_list[i],'kd_err'],df.loc[Plot_list[i],'scale']-df.loc[Plot_list[i],'scale_err']),
							facecolor='deepskyblue', alpha=0.5,edgecolor='None')
				ax.scatter(Xvals,Yvals,marker='o',s=20,color=[0,0,0])
				mytext = (r"k$_{D}$ =%3.1f $\pm$%2.1f %s"+'\n'+ r"$\chi$$^{2}$ %1.3f")\
					    %(df.loc[Plot_list[i],'kd'], df.loc[Plot_list[i],'kd_err'],Format_Name[unit],df.loc[Plot_list[i],'chi2'])
				ax.text(0.05,0.90, mytext, verticalalignment='center', fontsize=8, transform=ax.transAxes)
			else:
				ax.scatter(Xvals,Yvals,marker='o',s=20,color=[0,0,0])
			ax.set_xscale('log')
			ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
			ax.set_title(Plot_list[i].replace('_','-'))
			ax.set_xlabel(r"Total Protein Concentration (%s)"%(Format_Name[unit]))
			ax.set_ylabel("Norm %s" %(dtype))
			if max(Yvals) > 1.0: 
				ymax = np.round(max(Yvals),1)+0.1
				ymin = np.round(max(Yvals),1)-1.0
			if max(Yvals) <= 1.0:
				ymax = 1.1
				ymin = 0.0
			ax.set_ylim(ymin, ymax)
			ax.set_xlim(0.5*min(PCons), 2* max(PCons))
			ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
		fig.set_tight_layout(True)
		pdf.savefig()
		plt.close()
		pdf.close()

	plot_size = [3, 3]
	PCons = np.array(ConcentrationsList)
	for dtype in Data_Types:
		pdf = PdfPages(outpath+OutName +'_'+dtype+'_all_traj_linear.pdf')
		df = pd.read_csv(outpath+'CSV_Files/' + OutName + '_' + dtype + '_results.csv',index_col=0)
		Plot_list = df.index.tolist()
		num_rows = math.ceil(len(Plot_list)/5.00)
		fig = plt.figure(figsize=(5*plot_size[0],num_rows*plot_size[1]))
		simcons=np.linspace(0.0001, max(PCons) + max(PCons),10000)
		for i in range(len(Plot_list)):
			gridloc = 0
			gridloc = i+1
			ax = fig.add_subplot(num_rows,5,gridloc)
			Yvals, xvals = [], []
			yvals = df.ix[Plot_list[i]].tolist()[6:]
			for x in range(len(yvals)):
				if not np.isnan(yvals[x]):
					Yvals.append(yvals[x])
					xvals.append(ConcentrationsList[x])
			Xvals = np.array(xvals) 
			Yvals = np.array(Yvals)
			if Yvals[-1] > Yvals[0]:
				func = Fit_Slow_dimer_kd
			if Yvals[-1] < Yvals[0]:
				func = Fit_Slow_mono_kd
			if df.loc[Plot_list[i],'kd'] > 0:
				ax.plot(simcons,func(simcons, df.loc[Plot_list[i],'kd'],df.loc[Plot_list[i],'scale']), '-',color=[1,0,0],linewidth = 2)
				ax.fill_between(simcons,func(simcons,df.loc[Plot_list[i],'kd']+df.loc[Plot_list[i],'kd_err'], df.loc[Plot_list[i],'scale']+df.loc[Plot_list[i],'scale_err']),
							func(simcons,df.loc[Plot_list[i],'kd']-df.loc[Plot_list[i],'kd_err'], df.loc[Plot_list[i],'scale']-df.loc[Plot_list[i],'scale_err']),
							facecolor='deepskyblue', alpha=0.5,edgecolor='None')
				ax.scatter(Xvals,Yvals,marker='o',s=20,color=[0,0,0])
				mytext = (r"k$_{D}$ ={:3.1f} $\pm${:2.1f} {:}"+'\n'+ r"$\chi$$^{2}$ {:1.3f}".format(df.loc[Plot_list[i],'kd'], df.loc[Plot_list[i],'kd_err'],Format_Name[unit],df.loc[Plot_list[i],'chi2']))
				ax.text(0.05,0.90, mytext, verticalalignment='center', fontsize=8, transform=ax.transAxes)
			else:
				ax.scatter(Xvals,Yvals,marker='o',s=20,color=[0,0,0])
			#ax.set_xscale('log')
			ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
			ax.set_title(Plot_list[i].replace('_','-'))
			ax.set_xlabel(r"Total Protein Concentration (%s)"%(Format_Name[unit]))
			ax.set_ylabel("Norm %s" %(dtype))
			if max(Yvals) > 1.0: 
				ymax = np.round(max(Yvals),1)+0.1
				ymin = np.round(max(Yvals),1)-1.0
			if max(Yvals) <= 1.0:
				ymax = 1.1
				ymin = 0.0
			ax.set_ylim(ymin, ymax)
			ax.set_xlim(0.0, 1.1* max(PCons))
			ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
		fig.set_tight_layout(True)
		pdf.savefig()
		plt.close()
		pdf.close()
	print("Finished")
