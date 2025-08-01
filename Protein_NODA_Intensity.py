import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker 
import re
from scipy.stats import trim_mean

##########################################################################################
## Developed by Mary C Clay 
## e-mail: mary.clay@stjude.org
## St Jude Children's Research Hospital 
## Department of Structural Biology
##
## Last updated: November 20, 2020
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
##																									##
##	Function for placing labels on graphs for data points with absolute values greater than the 	##
##	significance cutoff values 																		##
##																									##
####----------------------------------------------------------------------------------------------####

def Plot_Intensiy_data(RunParams, MasterList, outpath):

####----------------------------------------------------------------------------------------------####
##																									##
## 	Generates all the run parameters from the *_input.txt file									 	##
##	RunParams for CSP : OutDir, OutName, Sequence, Start_Index, Residues, Data_Types, SDM, 			##
##						Show_Lables,Intensity_Norm(s), input_path									##
##																									##
####----------------------------------------------------------------------------------------------####
	ResIDlist = []
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
## Using the MasterList information generate list of DataSet names (DataSetList) which serve as 	##
## column names in pandas dataframe and keys in dictionaries for plotting.  						##
## Formatted titles and legend entries and storing them in dictionaries with DataSet as keys 		##
## Finally assigning uniqu color to each dataset in the ColorsDict.									##
##																									##
####----------------------------------------------------------------------------------------------####
	ColorsDict = {}
	Legend_dict = {}
	Title_dict = {}
	References = []
	DataSetList = []
	DF_columns = ['resid', 'atom', 'atom_sort']	
	for DataSet in MasterList:
		DataSetList.append(DataSet[0])
		DF_columns.append(DataSet[0])
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
		if len(DataSet) == 5:
			References.append(DataSet[4])
		if len(DataSet) == 4:
			References.append(MasterList[0][0])
		legend = DataSet[1]
		for src, target in Format_Name.iteritems():
			legend = legend.replace(src, target)
		Legend_dict[DataSet[0]] = legend
		Title_dict[DataSet[0]] = legend

	Plot_DataSetList =  DataSetList

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
	for dtype in Data_Types:
		exec('DF_' + dtype + ' = pd.DataFrame()')
		exec('DF_' + dtype + "_plot = pd.DataFrame()")

	## Pars the input sparky list and sort data into approreate DataFrame 
	for DataSet in MasterList:
		InputFiles = glob.glob(os.path.join(input_path + DataSet[2].strip()))
		if len(InputFiles) == 0: 
			print('No Data found check input file path %s%s' % (input_path, DataSet[2]))
			exit()
		if DataSet[0] not in DataSetList:
			DataSetList.append(DataSet[0])
		for file in InputFiles:
			print("Reading in %s" % file.split('/')[-1])
			in_df = pd.read_csv(file,sep=r'\s*',names=open(file).readline().rstrip().replace('Data Height', "Intensity").replace('Volume', "Volume  fit").replace('(hz)', "").split(),skiprows=[0,1],engine='python')
			for i, row in in_df.iterrows():
				if '?' not in row['Assignment']:
					if re.findall('[A-Z]([0-9]*)[A-Z]', row['Assignment']) and not re.findall('[A-Z]([0-9]*)[a-z][A-Z]', row['Assignment']):
						group = re.search('[A-Z]([0-9]*)', row['Assignment']).group()
						resid = group
					Assignment =  row['Assignment'].replace(group, group + '_')
					in_df.loc[i, 'Group'] = Assignment.split('_')[0]
					in_df.loc[i, 'Atoms'] = Assignment.split('_')[1]
					in_df.loc[i, 'resid'] = resid
					in_df.loc[i, 'resi'] = int(resid[1:])
			for dtype in Data_Types:
				df = eval('DF_' + dtype)
				for i, row in in_df.dropna(axis=0, subset=['resid']).iterrows():
					Sorting_dict = Allowed_atoms[row['Group'][0]]
					if row['Atoms'].split('-')[0] in Sorting_dict.keys():
						dfidx = row['Group'] + '_' + row['Atoms'].split("-")[0]
						df.loc[dfidx , 'resid'] = row['resid']
						df.loc[dfidx , 'resi'] = row['resi']
						df.loc[dfidx , 'nuc'] = row['Atoms'].split("-")[0][0]
						df.loc[dfidx , 'name'] = row['Atoms'].split("-")[0]
						df.loc[dfidx , 'atom_sort'] = Sorting_dict[row['Atoms'].split('-')[0]]
						df.loc[dfidx , DataSet[0]] = np.around(row[dtype],3)

####----------------------------------------------------------------------------------------------####
##																									##
##									Intensity/Volume normalization 									##
##  Peak intensity/volume normalization is done based on trimed mean of the observed values 		##
##  Trimed mean exclueds the lower and upper 10% of the gusian distribution of the observed values	##
##																									##
##  	Data is normalized independently for each DataSet and not relative to reference DataSet 	##
## 																									##
####----------------------------------------------------------------------------------------------####
	Normalize = {}
	for dtype in Data_Types:
		if dtype == 'Intensity' or dtype == 'Volume': 
			df = eval('DF_' +dtype)
			for nuc in df['nuc'].unique().tolist():
				df1 = df[df['nuc']== nuc ].copy(deep = True)
				for DataSet in DataSetList:
					values = [np.round(abs(val),4) for val in df1[DataSet].dropna().tolist() if val != 0.0]
					# print('NP Average '+'{:10.0f}'.format(np.average(values)))
					# print('Trimed Mean '+'{:10.0f}'.format(trim_mean(values,0.1)))
					Normalize[DataSet+'_'+nuc]=trim_mean(values,0.1)
			for i, row in df.iterrows():
				for DataSet in DataSetList:
					df.loc[i,DataSet] = df.loc[i,DataSet] / Normalize[DataSet+'_'+row['nuc']]
	print('Finished %s Analysis' %(dtype))
####----------------------------------------------------------------------------------------------####
##					 Save these results to a CSV file for each Data Type 							##
####----------------------------------------------------------------------------------------------####
	out_Columns = ['resid']
	for DataSet in Plot_DataSetList:
		out_Columns.append(DataSet)
	for dtype in Data_Types:
		DF = eval('DF_' + dtype)
		DF = DF.sort_values(by=['atom_sort', 'resi'])
		if not os.path.exists(outpath + 'CSV_Files/'):
			os.makedirs(outpath + 'CSV_Files/')
		DF.to_csv(outpath + 'CSV_Files/' +  OutName + '_' + dtype + '_results.csv', columns = out_Columns)

####----------------------------------------------------------------------------------------------####
##																									##
##		For geminal pairs select the entry with the largest perturbation and store it in a new		##
##	data frame df_dtype_plot which will be used for plotting the data and generating PyMol scripts 	##
##	This leaves the full list intact, but simplifies plots.											##
##																									##
####----------------------------------------------------------------------------------------------####

	for dtype in Data_Types:
		df = eval('DF_' +dtype)
		df_plot = eval('DF_' +dtype + '_plot')
		for atom in df['atom_sort'].unique().tolist():
			df_atoms = df[df['atom_sort']== atom ].sort_values(by=['resi']).index.tolist()
			df_atoms.append(df_atoms[-1])
			# print(df_atoms)
			for DataSet in Plot_DataSetList:
				used = []
				if df_atoms[i] not in used:
					for i in range(len(df_atoms))[:-1]:
						if df.loc[df_atoms[i],'resid'] != df.loc[df_atoms[i+1],'resid']:
							df_plot.loc[df.loc[df_atoms[i],'resid']+'_'+atom, 'resid'] = df.loc[df_atoms[i],'resid']
							df_plot.loc[df.loc[df_atoms[i],'resid']+'_'+atom, 'name'] = df.loc[df_atoms[idx],'name']
							df_plot.loc[df.loc[df_atoms[i],'resid']+'_'+atom, 'nuc'] = df.loc[df_atoms[i],'nuc']
							df_plot.loc[df.loc[df_atoms[i],'resid']+'_'+atom, 'atom_sort'] = atom
							df_plot.loc[df.loc[df_atoms[i],'resid']+'_'+atom, DataSet] = df.loc[df_atoms[i],DataSet]
							used.append(df_atoms[i])
						elif df.loc[df_atoms[i],'resid'] == df.loc[df_atoms[i+1],'resid']:
							values = [abs(df.loc[df_atoms[i],DataSet]),abs(df.loc[df_atoms[i+1],DataSet])]
							idx = i + values.index(max(values))
							df_plot.loc[df.loc[df_atoms[idx],'resid']+'_'+atom, 'resid'] = df.loc[df_atoms[idx],'resid']
							df_plot.loc[df.loc[df_atoms[idx],'resid']+'_'+atom, 'name'] = df.loc[df_atoms[idx],'name']
							df_plot.loc[df.loc[df_atoms[idx],'resid']+'_'+atom, 'nuc'] = df.loc[df_atoms[idx],'nuc']
							df_plot.loc[df.loc[df_atoms[idx],'resid']+'_'+atom, 'atom_sort'] = atom
							df_plot.loc[df.loc[df_atoms[idx],'resid']+'_'+atom, DataSet] = df.loc[df_atoms[idx],DataSet]
							used.extend((df_atoms[i],df_atoms[i+1]))
		df_plot.to_csv(outpath + 'CSV_Files/' +  OutName + '_' + dtype + '_plot.csv')

####----------------------------------------------------------------------------------------------####
##																									##
## 		Generating dictionaries to control axis limits and titles in scatter and bar plots 			##
## 		based on current data. This will keep everything plotted on the same relative scale			##
##		This also also done for the original chemical shifts used on correlation plots for 			##
##		assessing sugar pucker 																		##
##																									##
####----------------------------------------------------------------------------------------------####

	Delta_ppm_dict = {'H':r" ($^{1}$H, ppm)",'C':r" ($^{13}$C, ppm)",'N':r" ($^{15}$N, ppm)"}
	Delta_wavg_dict = {'C':r"$\Delta\delta$$_{avg}$ (HC, ppm)", 'N': r"$\Delta\delta$$_{avg}$ (HN, ppm)" }
	Title_label = {'N' : "Backbone",'H' : "Backbone", 'CMe': r"Methyl", "HMe": r"Methyl", "Me": "Methyl Intensity", "NE1":"W-NE1", "HE1":"W-HE1", 'Nside':'Nside'}
	Plot_Dict = {}
	for dtype in Data_Types:
		df = eval('DF_' + dtype + '_plot')
		for nuc in df['nuc'].unique().tolist():
			df1 = df[df['nuc'] == nuc].copy(deep=True)
			if np.max(df1.max(skipna=True, numeric_only=True)) > 0.02:
				vmax = (np.max(df1.max(skipna=True, numeric_only=True)) * 12.00) / 10.00
			else: vmax = 0.02
			if np.min(df1.min(skipna=True, numeric_only=True)) < -0.02:
				vmin = (np.min(df1.min(skipna=True, numeric_only=True)) * 12.00 )/ 10.00
			else: vmin = -0.0
			if dtype == 'Intensity':
				# if vmin < -1.0: vmin = -1.0
				if vmax < 1.0: vmax = 1.1
				Plot_Dict[nuc + dtype]=[[0.0, vmax], r"$\bar I$"]
			elif dtype == 'Volume':
				Plot_Dict[nuc + dtype]=[[0.0, vmax], r"$\bar V$"]



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
	mpl.rcParams['xtick.major.size'] = mpl.rcParams['ytick.major.size'] = 5
	mpl.rcParams['xtick.major.width'] = mpl.rcParams['ytick.major.width']=linewidths
	mpl.rcParams['xtick.minor.size'] = 2
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
	DataSets = Plot_DataSetList
	for x in range(len(DataSets)):
		colors = []
		colors.append(ColorsDict[DataSets[x]])
		pdf = PdfPages(outpath + DataSets[x] + '_plot.pdf')
		for dtype in Data_Types:
			Xval = range(int(Residues.split('-')[0]), int(Residues.split('-')[1])+1)
			df = eval('DF_' + dtype + '_plot')
			AtomsList=df['atom_sort'].unique().tolist()
			for atom in AtomsList:
				temp=[]
				for res in ResIDlist:
					temp.append(res +'_' +atom)
				df2 = df.reindex(temp)
				fig=plt.figure(figsize=(7.28,3))
				ax = fig.add_subplot(111)
				width = 0.9
				ax.bar(Xval, df2[DataSets[x]], width, color=ColorsDict[DataSets[x]], edgecolor='none', label = Legend_dict[DataSets[x]])
				ax.set_ylim(Plot_Dict[atom[0] + dtype][0])
				ax.set_ylabel(Plot_Dict[atom[0] + dtype][1])
				ax.set_xlim(int(Residues.split('-')[0]) - 2, int(Residues.split('-')[1]) + 2)
				if len(Xval) < 120:
					ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
					ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
				if len(Xval) >= 120:
					ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
					ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
				ax.set_xlabel('Residue Number')
				# l, b, w, h = ax.get_position().bounds
				# ax.set_position([ll, b + 0.1*h, ww, h*0.8])
				if Plot_Dict[atom[0] + dtype][0][1] <= 0.05:
					ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
				ax.yaxis.set_major_formatter(FormatStrFormatter('%4.2f'))
				ax.set_title(Title_label[atom])
				# legend = ax.legend(loc='upper left', frameon=False, markerscale=0.000001)
				# for color,text in zip(colors, legend.get_texts()):
				# 	text.set_color(color)
				plt.text(int(Residues.split('-')[0]),0.9*(Plot_Dict[atom[0] + dtype][0][1]), Legend_dict[DataSets[x]] , {'color': ColorsDict[DataSets[x]], 'fontsize': 10})
				plt.tight_layout(pad = 0.4, w_pad = 0.4, h_pad = 0.4)
				pdf.savefig()
				plt.close()
		pdf.close()

	print("Finished Individual Bar Plots")

####----------------------------------------------------------------------------------------------####
##	Create a separate bar plot for each type of data analyzed and atom observed, with all datasets	##
##	shown in same plot. Saved as Name_data_type_bar_plot.pdf.										##
####----------------------------------------------------------------------------------------------####
	if len(DataSets) > 1:
		for dtype in Data_Types:
			colors = []
			pdf = PdfPages(outpath + OutName + "_" + dtype + '_Summary_plot.pdf')
			df = eval('DF_' + dtype + '_plot')
			AtomsList=df['atom_sort'].unique().tolist()
			Xval = np.arange(int(Residues.split('-')[0]), int(Residues.split('-')[1])+1)
			for atom in AtomsList:
				temp=[]
				for res in ResIDlist:
					temp.append(res +'_' +atom)
				df2 = df.reindex(temp)
				fig_height = 2.0 * len(DataSets) 
				if fig_height <= 2.0: 
					fig_height = 3.0
				fig=plt.figure(figsize=(7.28,fig_height))
				resid = df2.resid.tolist()
				width = 0.9
				for x in range(len(DataSets)):
					colors.append(ColorsDict[DataSets[x]])
					ax = fig.add_subplot(int(len(DataSets)),1,x+1)
					rects = ax.bar(Xval, df2[DataSets[x]], width, color=ColorsDict[DataSets[x]], edgecolor='none', label = Legend_dict[DataSets[x]])
					if dtype == 'CS_wavg':
						ax.axhline(y = Cutoff_dict[DataSets[x]+atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
					if dtype != 'CS_wavg':
						ax.axhline(y = Cutoff_dict[DataSets[x]+atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
						ax.axhline(y = -1* Cutoff_dict[DataSets[x]+atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
						if Plot_Dict[atom[0] + dtype][0][0] != 0.0: 
							ax.axhline(y = 0.0, color = [0,0,0])
					if Show_Labels.upper() == 'Y':
						autolabel(rects, df2['resid'].tolist(), Cutoff_dict[DataSets[x]+atom[0]], ax)
					ax.set_ylim(Plot_Dict[atom[0] + dtype][0])
					ax.set_ylabel(Plot_Dict[atom[0] + dtype][1])
					ax.set_xlim(int(Residues.split('-')[0]) - 2, int(Residues.split('-')[1]) + 2)
					if len(Xval) < 120:
						ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
						ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
					if len(Xval) >= 120:
						ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
						ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
					if Plot_Dict[atom[0] + dtype][0][1] <= 0.05:
						ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
					plt.text(int(Residues.split('-')[0]),0.9*(Plot_Dict[atom[0] + dtype][0][1]), Legend_dict[DataSets[x]] , {'color': ColorsDict[DataSets[x]], 'fontsize': 10})
					ax.yaxis.set_major_formatter(FormatStrFormatter('%4.2f'))
				ax.set_title(Title_label[atom])
				ax.set_xlabel('Residue Number')
				plt.tight_layout(pad = 0.4, w_pad = 0.4, h_pad = 0.4)
				pdf.savefig(transparent=True)
				plt.close()
			pdf.close()
		print("Finished Summary Bar Plots")


####----------------------------------------------------------------------------------------------####
##																									##
## 											PyMol Scripts 											##
##	For each non reference data set generate a PyMol script to color code the CSP/intensity changes	##
##	According to magnitude of change and the significance cutoff. 									##
##	The absolute value of the reported value is stored according to the following binds, which are  ##
##	generated on the fly using the Cutoff value and maximum change observed for all combinations of	##
##	data set and data types 																		##
##			color0 if Cutoff < val < 2*Cutoff 														##
##			color1 if 2*Cutoff < val < 3*Cutoff 													##
##			color2 if 3*Cutoff < val < 4*Cutoff 													##
##			color3 if 4*Cutoff < val < 5*Cutoff 													##
##			color4 if 5*Cutoff < val < 6*Cutoff 													##
##			color5 if 6*Cutoff < val < max 															##
##																									##
##	The results are saved in a separate txt file for each non reference data set, with a section 	##
##	for each data type analyzed. After loading the appropriate pdb file user can copy and past the 	##
##	content to render and image																		##
####----------------------------------------------------------------------------------------------####
# 	pymol_lable_str = '''one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
# 'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
# 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
# 'GLY':'G', 'PRO':'P', 'CYS':'C'}'''

# 	outstr = '''set ray_opaque_background, 0
# set depth_cue, off
# bg_color white
# set label_color, black
# set ray_shadows, 0
# set orthoscopic, on
# hide everything, all
# show cartoon, chain A
# color gray70, all
# set sphere_scale, 0.6
# set cartoon_color, gray70, chain A
# '''
# 	C_color = '''set_color C-color0 = [254,217,118]
# set_color C-color1 = [254,178,76]
# set_color C-color2 = [253,141,60]
# set_color C-color3 = [252,78,42]
# set_color C-color4 = [189,0,38]
# set_color C-color5 = [150,0,0]
# '''
# 	N_color = '''set_color N-color0 = [151,212,230]
# set_color N-color1 = [81,183,214]
# set_color N-color2 = [65,143,190]
# set_color N-color3 = [50,101,166]
# set_color N-color4 = [33,54,141]
# set_color N-color5 = [20,30,100]
# '''
# 	H_color = '''set_color H-color0 = [195, 235, 150]
# set_color H-color1 = [75,195,130]
# set_color H-color2 = [0,135,100]
# set_color H-color3 = [0,115,90]
# set_color H-color4 = [0,95,80]
# set_color H-color5 = [0,75,70]
# '''
# 	print 'Generating PyMol Scripts'
# 	for DataSet in Plot_DataSetList:
# 		if not os.path.exists(outpath + 'Pymol_Scripts/'):
# 			os.makedirs(outpath + 'Pymol_Scripts/')
# 		pymol_script = open(outpath + 'Pymol_Scripts/' + DataSet +'_Pymol.txt','w')
# 		for dtype in Data_Types:
# 			df = eval('DF_' + dtype + '_plot')
# 			AtomsList=df['atom_sort'].unique().tolist()
# 			for atom in AtomsList:
# 				select_name = "%s_%s-%s" % (DataSet,dtype,atom)
# 				Pymol_dict = {'CMe':['C-color', "show spheres," + select_name + " and resn VAL+THR and name CG2\nshow spheres," + select_name + " and resn LEU+ILE and name CD1\nshow spheres," + select_name + " and resn MET and name CE\nshow spheres," + select_name + " and resn ALA and name CB\n"],
# 				 'HMe':['H-color', "show spheres," + select_name + " and resn VAL+THR and name HG2\nshow spheres," + select_name + " and resn LEU+ILE and name HD1\nshow spheres," + select_name + " and resn MET and name HE\nshow spheres," + select_name + " and resn ALA and name HB\n"],
# 				 'N':['N-color', "show spheres," + select_name + " and name N\n"], 'H':['H-color',"show spheres," + select_name + " and name H\n"],
# 				  'NE1':['N-color',"show spheres," + select_name + " and name NE1\n"], 'HE1':['H-color',"show spheres," + select_name + " and name HE1\n"]}
# 				df2 = df[(df['atom_sort'] == atom)].copy(deep=True)
# 				Bins = []
# 				if cutoff > 6* df2[DataSet].abs().max(skipna=True):
# 					cutoff = cutoff/int(SDM)
# 				for i in range(6):
# 					# Bins.append(float("{0:.3f}".format(cutoff + i*((df2[DataSet].abs().max(skipna=True))/6.0))))
# 					Bins.append(float("{0:.3f}".format(cutoff + i*cutoff)))
# 				Bins.append(float("{0:.3f}".format((df2[DataSet].abs().max(skipna=True))))) 
# 				pymol_script.write("#########################################################\n## %s_%s Color Coding\n## Bins %s\n#########################################################\n\n" %(dtype, atom, Bins))
# 				pymol_script.write(str(outstr))
# 				color_str = eval(atom[0]+'_color')
# 				pymol_script.write(str(color_str))
# 				select_line = 'create %s_%s-%s, chain A and resi ' % (DataSet,dtype,atom)
# 				for x in range(len(Bins)-1): 
# 					for row in df2.dropna(subset=[DataSet]).index.tolist():
# 						if float("{0:.3f}".format(abs(df.loc[row,DataSet]))) >= Bins[x] and float("{0:.3f}".format(abs(df.loc[row,DataSet]))) <= Bins[x+1]:
# 							select_line = select_line + df2.loc[row,'resid'][1:] + "+"
# 				pymol_script.write(select_line[:-1] + '\n')
# 				pymol_script.write('hide cartoon, '+ select_name + '\n')
# 				print '%s %s %s Bins ' %(DataSet, atom, dtype)
# 				print Bins
# 				for x in range(len(Bins)-1): 
# 					outline = 'color ' + Pymol_dict[atom][0] + str(x) + ', '+ select_name + ' and resi '
# 					for row in df2.dropna(subset=[DataSet]).index.tolist():
# 						if float("{0:.3f}".format(abs(df.loc[row,DataSet]))) >= Bins[x] and float("{0:.3f}".format(abs(df.loc[row,DataSet]))) <= Bins[x+1]:
# 							outline = outline + df2.loc[row,'resid'][1:] + "+"
# 							select_line = select_line + df2.loc[row,'resid'][1:] + "+"
# 					pymol_script.write(outline[:-1] + '\n')
# 				pymol_script.write(Pymol_dict[atom][1])
# 				pymol_script.write('\n\n')
# 				# pymol_script.write('label (' + select_name + ' and name CA), "%s%s" %(one_letter[resn],resi)\n\n\n\n')
# 		pymol_script.close()	

	print("Finished")