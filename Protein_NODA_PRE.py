import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import trim_mean
from scipy.stats import tstd
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker 


# pd.set_option('precision',4)


CurDir = os.getcwd()
########################################################################################## 
## Dictionaries for sorting input data into pandas Data Frame, and formatting of supper 
## and subscript, and primes for sugars
########################################################################################## 

replacements = {'X_PPM': 'w1', 'Y_PPM': 'w2', 'XW_HZ': 'lw1', 'YW_HZ': 'lw2', 'HEIGHT': 'Intensity',  'VOL': 'Volume', 'ASS': 'Group     Atom ', 'Assignment': "Group     Atom ",
				" (hz)": " ", 'Data Height': 'Intensity'}
Sort_dic_full = {"N-H":[[['N', 'w1'], ['H', 'w2']], [['N', 'lw1'], ['H', 'lw2']], [['N', 'Intensity']], [['N', 'Volume']]],
				 "CB-HB":[[['CMe1', 'w1'], ['HMe1', 'w2']], [['CMe1', 'lw1'], ['HMe1', 'lw2']], [['CMe1', 'Intensity']], [['CMe1', 'Volume']]],
				 "CE-HE":[[['CMe1', 'w1'], ['HMe1', 'w2']], [['CMe1', 'lw1'], ['HMe1', 'lw2']], [['CMe1', 'Intensity']], [['CMe1', 'Volume']]],
				 "CD1-HD1":[[['CMe1', 'w1'], ['HMe1', 'w2']], [['CMe1', 'lw1'], ['HMe1', 'lw2']], [['CMe1', 'Intensity']], [['CMe1', 'Volume']]],
				 "CD2-HD2":[[['CMe2', 'w1'], ['HMe2', 'w2']], [['CMe2', 'lw1'], ['HMe2', 'lw2']], [['CMe2', 'Intensity']], [['CMe2', 'Volume']]],
				 "CG1-HG1":[[['CMe1', 'w1'], ['HMe1', 'w2']], [['CMe1', 'lw1'], ['HMe1', 'lw2']], [['CMe1', 'Intensity']], [['CMe1', 'Volume']]],
				 "CG2-HG2":[[['CMe2', 'w1'], ['HMe2', 'w2']], [['CMe2', 'lw1'], ['HMe2', 'lw2']], [['CMe2', 'Intensity']], [['CMe2', 'Volume']]],
				 "NE-HE":[[['NE', 'w1'], ['HE', 'w2']], [['NE', 'lw1'], ['HE', 'lw2']], [['NE', 'Intensity']], [['NE', 'Volume']]],
				 }
Sort_dic_index = {"CS":0, "LW":1, "Intensity":2, "Volume":3}
Format_Name = {'uM':r'$\mu$M','MnCl2':r'MnCl$_{2}$','MgCl2':r'MgCl$_{2}$','CaCl2':r'CaCl$_{2}$','ZnCl2':r'ZnCl$_{2}$','BaCl2':r'BaCl$_{2}$',"H2O":r"H$_{2}$O","D2O":r"D$_{2}$O", 'alpha':r"$\alpha$", 'beta':r"$\beta$",'delta':r"$\Delta$"}

####----------------------------------------------------------------------------------------------####
##																									##
## 	Generates all the run parameters need, but lets automate it so we don't have to worry about 	##
## 	missing any thing.																				##
##	RunParams for CSP : input_path, OutDir, OutName, Data_Types, Intensity_Norm, Sequence, 			##
##						Residues , sample    														##
##																									##
####----------------------------------------------------------------------------------------------##### 


def Plot_PRE_data(RunParams, MasterList, outpath):

##########################################################################################
## Generates all the run parameters need, but lets automate it so we don't have to worry about missing any thing
## RunParams for CSP : input_path, Data_Types, Intensity_Norm, ResIDList, Name , Sample
## All RunParams but Sample are entented to be treated as name = valure or name = list of values
##########################################################################################

	for i in range(len(RunParams)):
		if len(RunParams[i]) == 2:
			exec(RunParams[i][0] + '= RunParams[i][1]')
		else:
			exec(RunParams[i][0] + '= RunParams[i][1:len(RunParams[i])]')
	# Just in case only one type of analysis is requested. 
	if isinstance(Data_Types, str): Data_Types = [ Data_Types ]
	if isinstance(Intensity_Norm, str): Intensity_Norm = [ Intensity_Norm ]
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
### Make a list of only the residues the user has indicated that they want to use in the Residues entry of the input file
	ResIDlist = []
	for i in range(len(Seq)): 
		if int(Seq[i][1:])>= int(Residues.split('-')[0]) and int(Seq[i][1:]) <= int(Residues.split('-')[1]):
			ResIDlist.append(Seq[i])
	
### Make groups of residies to make veiwing easier for large systems. 
	Groups = [[1, 99]]
	for i in range(1,(len(ResIDlist)/100) ,1):
		start_i = i*100
		end_i = start_i +99
		Groups.append([start_i ,end_i])
	Groups.append([(len(ResIDlist)/100) * 100 ,len(ResIDlist)])
	if 'Intensity' in Data_Types:
		for i in range(len(Intensity_Norm)):
			if Intensity_Norm[i].split('-')[1][0] == 'N':
				N_Norm = {'N':Intensity_Norm[i].replace('-','_')}
			if Intensity_Norm[i].split('-')[1][0] == 'C':
				Me_ref = Intensity_Norm[i].replace('-CB','_CMe1').replace('-CG1','_CMe1').replace('-CD1','_CMe1').replace('-CG2','_CMe2').replace('-CD2','_CMe2').replace('-CE','_CMe1')
				Me_Norm = {'CMe1':Me_ref, 'CMe2':Me_ref}

####----------------------------------------------------------------------------------------------####
##																									##
## Using the MasterList information generate list of DataSet names (DataSetList) which serve as 	##
## column names in pandas dataframe and keys in dictionaries for plotting.  						##
## Formatted titles and legend entries and storing them in dictionaries with DataSet as keys 		##
## Finally assigning uniqu color to each dataset in the ColorsDict.									##
##																									##
####----------------------------------------------------------------------------------------------####
	DF_columns = ['resid', 'atom', 'nuc','atom_sort']	
	ColorsDict = {}
	Legend_dict = {}
	Title_dict = {}
	DataSetList = []
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
		legend = DataSet[1]
		for src, target in Format_Name.iteritems():
			legend = legend.replace(src, target)
		Legend_dict[DataSet[0]] = legend 
		Title_dict[DataSet[0]] = legend

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

	for dtype in Data_Types:
		## creating empty data frame to store results in, must exist before use
		exec('DF_temp' +dtype + ' = pd.DataFrame()')
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
				if not line.startswith('      Assignment') or line.startswith('FORMAT') or line.startswith('REMARK'):
					line = line.replace("N-", "   N-").replace("CB-", "   CB-").replace("CD1-", "   CD1-").replace("CD2-", "   CD2-").replace("CE-", "   CE-").replace("CG1-", "   CG1-").replace("CG2-", "   CG2-")
					if 'T' in line:
						line = line.replace('CG2-HG2', 'CG1-HG1')
					temp_file.write(line)
			temp_file.close()
			if 'lw3' in Header:
				Data_Types = ['CS'] 
			else: Data_Types = Data_Types
			for dtype in Data_Types:
				df = eval('DF_temp' + dtype)
				Sort_dic = {}
				in_df = pd.read_table('NODA_temp_Data_input', delim_whitespace=True)
				for key in in_df['Atom'].tolist():
					if key in Sort_dic_full.keys():
						if  len(Sort_dic_full[key]) >=	Sort_dic_index[dtype]:
							Sort_dic[key]=Sort_dic_full[key][Sort_dic_index[dtype]]
				for i, row in in_df.iterrows():
					if row['Atom'] in Sort_dic.keys():
						for y in range(len(Sort_dic[row['Atom']])):
							## Set the Row with index = ResID_Atom 
							if row['Group'] in ResIDlist:
								df.loc[row['Group'] + '_' + Sort_dic[row['Atom']][y][0], 'resid'] = row['Group']
								df.loc[row['Group'] + '_' + Sort_dic[row['Atom']][y][0], 'atom'] = row['Atom'].split("-")[y]
								df.loc[row['Group'] + '_' + Sort_dic[row['Atom']][y][0], 'nuc'] = row['Atom'].split("-")[y][0]
								df.loc[row['Group'] + '_' + Sort_dic[row['Atom']][y][0], 'atom_sort'] = Sort_dic[row['Atom']][y][0]
								df.loc[row['Group'] + '_' + Sort_dic[row['Atom']][y][0], DataSet[0]] = np.around(row[Sort_dic[row['Atom']][y][1]],3)
	for dtype in Data_Types:
		df = eval('DF_temp'+ dtype)
		exec('DF_' + dtype + ' = df.dropna(axis = 0, subset=[DataSetList[0]]).copy(deep=True)')
		df_used = eval('DF_' + dtype)
		print(dtype + ' Data sorted into DataFrame')
	DF_CS_original = DF_CS.copy(deep = True)

####----------------------------------------------------------------------------------------------####
##																									##
## 	 Calculate CSP change in LW and Intensity/Volume relative to the first (reference) data set  	##
##																									##
####----------------------------------------------------------------------------------------------####

####----------------------------------------------------------------------------------------------####
##																									##
## 	 	 Calculate delta omega for all data sets relative to the first (reference) data set  		##
##																									##
####----------------------------------------------------------------------------------------------####
	if 'CS' in Data_Types:
		Ref_CS = DF_CS.copy(deep=True)
		CS_init = DF_CS
		for DataSet in DataSetList:
			CS_init[DataSet] = CS_init[DataSet] - Ref_CS[DataSetList[0]]
		DF_CS = CS_init.copy(deep = True)
####----------------------------------------------------------------------------------------------####
##																									##
## 	 	 Calculate weighted chemical shift difference for all data sets based on delta omega		##
##																									##
####----------------------------------------------------------------------------------------------####
		Delta_CS_H = DF_CS[DF_CS['nuc'] == 'H'].copy(deep = True)
		Delta_CS_X =DF_CS[DF_CS['nuc'] != 'H'].copy(deep = True)
		CS_wavg = DF_CS[DF_CS['nuc'] != 'H'].copy(deep = True)

		for DataSet in DataSetList[1:]:
			for x in range(len(Delta_CS_H.index.tolist())): 
				if CS_wavg.loc[Delta_CS_X.index.tolist()[x],'nuc'] == 'C':
					CS_wavg.loc[Delta_CS_X.index.tolist()[x], DataSet] = np.sqrt((Delta_CS_H.loc[Delta_CS_H.index.tolist()[x], DataSet])**2 + 0.3 * (Delta_CS_X.loc[Delta_CS_X.index.tolist()[x],DataSet])**2)
				if CS_wavg.loc[Delta_CS_X.index.tolist()[x],'nuc'] == 'N':
					CS_wavg.loc[Delta_CS_X.index.tolist()[x], DataSet] = np.sqrt((Delta_CS_H.loc[Delta_CS_H.index.tolist()[x], DataSet])**2 + 0.15 * (Delta_CS_X.loc[Delta_CS_X.index.tolist()[x],DataSet])**2)
		DF_CS_wavg = CS_wavg.copy(deep=True)
		Data_Types.append('CS_wavg')
####----------------------------------------------------------------------------------------------####
##																									##
## 	  Calculate relative line width for all data sets relative to the first (reference) data set    ##
##																									##
####----------------------------------------------------------------------------------------------####
	if 'LW' in Data_Types:
		Ref_LW=DF_LW.copy(deep=True)
		LW=DF_LW
		for DataSet in DataSetList:
			LW[DataSet]=LW[DataSet] / Ref_LW[DataSetList[0]]
		print('Finished LW Analysis')
####----------------------------------------------------------------------------------------------####
##																									##
##						Dictionary used for Internal Intensity/Volume normalization 				##
##  Peak intensity/volume normalization is done based on residue type for methyls and based on a	##
##  specificed residues for amid peaks. All Methionies CE-HE are normalized relative to the same 	##
##  specificed Me residue, and so on. 																##																					##
##																									##
##  	Data is normalized independently for each DataSet and not relative to reference DataSet 	##
## 																									##
####----------------------------------------------------------------------------------------------####
	if 'Intensity' in Data_Types:
		Ref_Int = DF_Intensity.fillna(value=0.01,axis=0).copy(deep = True)
		Int_df = DF_Intensity.fillna(value=0.01,axis=0).copy(deep = True)
		for DataSet in DataSetList:
			Int_df[DataSet] = 1.0 - Int_df[DataSet] / Ref_Int[DataSetList[0]]
		DF_Intensity = Int_df
		print('Finished Intensity Analysis')
	if 'Volume' in Data_Types:
		Ref_Int = DF_Volume.fillna(value=0.01,axis=0).copy(deep = True)
		Int_df = DF_Volume.fillna(value=0.01,axis=0).copy(deep = True)
		for DataSet in DataSetList:
			Int_df[DataSet] = 1.00 - Int_df[DataSet] / Ref_Int[DataSetList[0]]
		DF_Volume = Int_df
		print('Finished Volume Analysis')
####----------------------------------------------------------------------------------------------####
##																									##
## 									Calculate significance cut off									##
## A separate cut off is calculated for 1H, 13C, and 15N 											##
## The cut off is = 2* std(Trimmed data Set)														##
## All values that are less than 0.7% of maximum observed CSP are used so that an inordinately 		##
## std is not obtained due to out lier in the normal spread of the data. 							##
## Essentially excluding outliers in the dw spread. 												##
##																									##
####----------------------------------------------------------------------------------------------####

	for dtype in Data_Types:
		exec('Cutoff_' + dtype + '_dict = {}')
		Cutoff_dict = eval('Cutoff_' + dtype + '_dict')
		df = eval('DF_' + dtype)
		df = df.fillna(value=0.00,axis=0)
		for nuc in df['nuc'].unique().tolist():
			df1 = df[df['nuc']== nuc ].copy(deep = True)
			Values = []
			for DataSet in DataSetList[1:]:
				for val in df1.index.tolist():
						Values.append(abs(df1.loc[val,DataSet]))
			cutoff_loop = False
			print('Starting Number of Values = {:}'.format(len(Values)))
			while not cutoff_loop:
				sigma = np.std(Values)
				Trimmed_values = [Values[i] for i in range(len(Values)) if Values[i] < 3* sigma]
				if len(Trimmed_values) != len(Values):
					Values = Trimmed_values
				if len(Trimmed_values) == len(Values):
					cutoff_loop = True
					sigma_good = np.std(Trimmed_values)
					print("Final number of values used in cutoff calc %s" %(len(Trimmed_values)))
			Cutoff_dict[nuc] = sigma_good
			
			print('{:} {:} {:0.4f}'.format(dtype, nuc, sigma_good))


####----------------------------------------------------------------------------------------------####
##																									##
## 		Generating dictionaries to control axis limits and titles in scatter and bar plots 			##
## 		based on current data. This will keep everything plotted on the same relative scale			##
##		This also also done for the original chemical shifts used on correlation plots for 			##
##		assessing sugar pucker 																		##
##																									##
####----------------------------------------------------------------------------------------------####

	Delta_ppm_dict = {'H':r"($^{1}$H, ppm)",'C':r"($^{13}$C, ppm)",'N':r"($^{15}$N, ppm)"}
	Delta_wavg_dict = {'C':r"($\Delta$H$^{2}$ + 0.3$\Delta$C$^{2}$)$^{1/2}$", 'N': r"($\Delta$H$^{2}$ + 0.15$\Delta$N$^{2}$)$^{1/2}$" }
	Title_label = {'N' : "Backbone amide-N",'H' : "Backbone amide-H", 'CMe1': "M-CE, A-CB, I-CD1, T-CG2, V-CG1, L-CD1", "CMe2": "V-CG2, L-CD2", "HMe1": "M-HE, A-HB, I-HD1, T-HG2, V-HG1, L-HD1", "HMe2": "V-HG2, L-HD2"}
	Plot_Dict = {}
	for dtype in Data_Types:
		df = eval('DF_' + dtype)
		for nuc in df['nuc'].unique().tolist():
			df1 = df[df['nuc'] == nuc].copy(deep=True)
			# if np.max(df1.max(skipna=True, numeric_only=True)) > 0.05:
			# 	vmax = (np.max(df1.max(skipna=True, numeric_only=True)) * 11.00) / 10.00
			# else: vmax = 0.05
			# if np.min(df1.min(skipna=True, numeric_only=True)) < -0.05:
			# 	vmin = (np.min(df1.min(skipna=True, numeric_only=True)) * 11.00 )/ 10.00
			# else: vmin = -0.05
			if dtype == 'CS':
				Plot_Dict[nuc + dtype]=[[(np.min(df1.min(skipna=True, numeric_only=True)) * 11.00 )/ 10.00, (np.max(df1.max(skipna=True, numeric_only=True)) * 11.00) / 10.00], r"$\Delta\delta$"+ Delta_ppm_dict[nuc]]
			elif dtype == 'CS_wavg':
				Plot_Dict[nuc + dtype]=[[0.0, (np.max(df1.max(skipna=True, numeric_only=True)) * 11.00) / 10.00], Delta_wavg_dict[nuc]]
			elif dtype == 'LW':
				Plot_Dict[nuc + dtype]=[[0.0, (np.max(df1.max(skipna=True, numeric_only=True)) * 11.00) / 10.00], r"LW/LW$_{0}$ "]
			elif dtype == 'Intensity':
				 Plot_Dict[nuc + dtype]=[[0.0, 1.1 ], r"1 - $\bar I$$_{0}$/$\bar I$"]
			elif dtype == 'Volume':
				Plot_Dict[nuc + dtype]=[[0.0, (np.max(df1.max(skipna=True, numeric_only=True)) * 11.00) / 10.00], r"1 - $\bar V$$_{0}$/$\bar V$"]

####----------------------------------------------------------------------------------------------####
##			 	Setting controlling plot appearance: Font, line widths, ticks, and legend  			##
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

####----------------------------------------------------------------------------------------------####
## 			Bar plots showing CSP  for each atom observed, with a separate file for each 			##
##	individual dataset. Basically breaking 	Name_data_type_bar_plot.pdf file up. 					##
##	Saved as DataSet_data_type_bar_plot.pdf 														##																								##
####----------------------------------------------------------------------------------------------####
	for dtype in Data_Types:
		print(dtype)
		Cutoff_dict = eval('Cutoff_' + dtype + '_dict')
		Xval = range(int(Residues.split('-')[0]), int(Residues.split('-')[1])+1)
		df = eval('DF_' + dtype)
		DataSets = DataSetList[1:]
		for x in range(len(DataSets)):
			colors = []
			colors.append(ColorsDict[DataSets[x]])
			pdf = PdfPages(outpath + DataSets[x] + "_" + dtype + '_plot.pdf')
			AtomsList=df['atom_sort'].unique().tolist()
			for atom in AtomsList:
				temp=[]
				for res in ResIDlist:
					temp.append(res +'_' +atom)
				df2 = df.reindex(temp)
				entry_width = 6.5/len(ResIDlist)
				fig_width = 0.78 + entry_width *len(df2.resid.tolist())
				fig=plt.figure(figsize=(fig_width,5))
				ax = fig.add_subplot(111)
				resid = np.arange(len(df2.resid.tolist()))
				width = 0.9
				ax.bar(Xval, df2[DataSets[x]], width, color=ColorsDict[DataSets[x]], edgecolor='none', label = Legend_dict[DataSets[x]])
				if dtype == 'CS_wavg':
					ax.axhline(y = Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
				if dtype == 'CS':
					ax.axhline(y = Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
					ax.axhline(y = -1* Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
					ax.axhline(y = 0.0, color = [0,0,0])
				ax.set_ylim(Plot_Dict[atom[0] + dtype][0])
				ax.set_ylabel(Plot_Dict[atom[0] + dtype][1])
				ax.set_xlim(int(Residues.split('-')[0]) - 2, int(Residues.split('-')[1]) + 2)
				ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
				ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
				ax.set_xlabel('Residue Number')
				#ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
				ax.set_title(Title_label[atom])
				legend = ax.legend(loc='best', frameon=False, markerscale=0.000001)
				for color,text in zip(colors, legend.get_texts()):
					text.set_color(color)
				plt.tight_layout(pad = 0.4, w_pad = 0.4, h_pad = 0.4)
				pdf.savefig()
				plt.close()
			pdf.close()
	print("Finished Individual Bar Plots")

####----------------------------------------------------------------------------------------------####
##	Create a separate bar plot for each type of data analyzed and atom observed, with all datasets	##
##	shown in same plot. Saved as Name_data_type_bar_plot.pdf.										##
####----------------------------------------------------------------------------------------------####

	for dtype in Data_Types:
		Cutoff_dict = eval('Cutoff_' + dtype + '_dict')
		colors = []
		pdf = PdfPages(outpath + OutName + "_" + dtype + '_Summary_plot.pdf')
		df = eval('DF_' + dtype)
		AtomsList=df['atom_sort'].unique().tolist()
		Xval = np.arange(int(Residues.split('-')[0]), int(Residues.split('-')[1])+1)
		for atom in AtomsList:
			temp=[]
			for res in ResIDlist:
				temp.append(res +'_' +atom)
			df2 = df.reindex(temp)
			DataSets = DataSetList[1:]
			entry_width = 8.0/len(ResIDlist)
			fig_width = 0.78 + entry_width *len(df2.resid.tolist())
			if fig_width < 3.0: 
				fig_width = 3.0 
			fig_height = 2.0 * len(DataSets) 
			if fig_height < 5.0: 
				fig_height = 5.0
			fig=plt.figure(figsize=(fig_width,fig_height))
			resid = np.arange(len(df2.resid.tolist()))
			width = 0.9
			for x in range(len(DataSets)):
				colors.append(ColorsDict[DataSets[x]])
				ax = fig.add_subplot(int(len(DataSets)),1,x+1)
				ax.bar(Xval, df2[DataSets[x]], width, color=ColorsDict[DataSets[x]], edgecolor='none', label = Legend_dict[DataSets[x]])
				if dtype == 'CS_wavg':
					ax.axhline(y = Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
				if dtype == 'CS':
					ax.axhline(y = Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
					ax.axhline(y = -1* Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
					ax.axhline(y = 0.0, color = [0,0,0])
				ax.set_ylim(Plot_Dict[atom[0] + dtype][0])
				ax.set_ylabel(Plot_Dict[atom[0] + dtype][1])
				ax.set_xlim(int(Residues.split('-')[0]) - 2, int(Residues.split('-')[1]) + 2)
				if len(Xval) < 100:
					ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
					ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
				if len(Xval) >= 100:
					ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
					ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
				legend = ax.legend(loc='best', frameon=False, markerscale=0.000001)
				for text in legend.get_texts():
					text.set_color(ColorsDict[DataSets[x]])
				#ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
			ax.set_title(Title_label[atom])
			ax.set_xlabel('Residue Number')
			plt.tight_layout(pad = 0.4, w_pad = 0.4, h_pad = 0.4)
			pdf.savefig()
			plt.close()
		pdf.close()
	print("Finished Summary Bar Plots")
####----------------------------------------------------------------------------------------------####
##	Create a separate scatter plot for each type of data analyzed and atom observed,with all		##
##	datasets shown in same plot. Saved as Name_data_type_bar_plot.pdf.								##
####----------------------------------------------------------------------------------------------####
	for dtype in Data_Types:
		Cutoff_dict = eval('Cutoff_' + dtype + '_dict')
		colors = []
		pdf = PdfPages(outpath + OutName + "_" + dtype + '_Scatter_plot.pdf')
		df = eval('DF_' + dtype)
		AtomsList=df['atom_sort'].unique().tolist()
		Xval = np.arange(int(Residues.split('-')[0]), int(Residues.split('-')[1])+1)
		for atom in AtomsList:
			temp=[]
			for res in ResIDlist:
				temp.append(res +'_' +atom)
			df2 = df.reindex(temp)
			entry_width = 6.5/len(ResIDlist)
			fig_width = 0.78 + entry_width *len(df2.resid.tolist())
			if fig_width < 3.0: 
				fig_width = 3.0
			fig=plt.figure(figsize=(fig_width,5))
			ax = fig.add_subplot(111)
			DataSets = DataSetList[1:]
			for x in range(len(DataSets)):
				colors.append(ColorsDict[DataSets[x]])
				ax.scatter(np.array(Xval),np.array(df2[DataSets[x]]),color=ColorsDict[DataSets[x]],marker='o',s=30,clip_on=False,label = Legend_dict[DataSets[x]])
				ax.plot(np.array(Xval),np.array(df2[DataSets[x]]),color=ColorsDict[DataSets[x]])
			if dtype == 'CS_wavg':
				ax.axhline(y = Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
			if dtype == 'CS':
				ax.axhline(y = Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
				ax.axhline(y = -1* Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
				ax.axhline(y = 0.0, color = [0,0,0])
			ax.set_ylim(Plot_Dict[atom[0] + dtype][0])
			ax.set_ylabel(Plot_Dict[atom[0] + dtype][1])
			ax.set_xlim(int(Residues.split('-')[0]) - 2, int(Residues.split('-')[1]) + len(DataSets) + 2)
			ax.set_xlabel('Residue Number')
			if len(Xval) < 100:
				ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
				ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
			if len(Xval) >= 100:
				ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
				ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
			ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
			ax.set_title(Title_label[atom])
			legend = ax.legend(loc='best', frameon=False, markerscale=0.000001 )
			for color,text in zip(colors, legend.get_texts()):
				text.set_color(color)
			plt.tight_layout(pad = 0.4, w_pad = 0.4, h_pad = 0.4)
			pdf.savefig()
			plt.close()
		pdf.close()
	print("Finished Summary Scater Plots")
####----------------------------------------------------------------------------------------------####
##	Create a separate bar plot for each type fo data analized and atom observed, with all datasets	##
##	shown in same plot. Saved as Name_data_type_bar_plot.pdf, But this time plot only 100 resudues	##
##	per plot, to break up larger plots and make identification of residue number easier.			##
####----------------------------------------------------------------------------------------------####
	for dtype in Data_Types:
		Cutoff_dict = eval('Cutoff_' + dtype + '_dict')
		colors = []
		pdf = PdfPages(outpath + OutName + "_" + dtype + '_Summary_plot_v2.pdf')
		df = eval('DF_' + dtype)
		AtomsList=df['atom_sort'].unique().tolist()
		Xval = np.arange(int(Residues.split('-')[0]), int(Residues.split('-')[1])+1)
		for atom in AtomsList:
			for group in Groups:
				shortlist = ResIDlist[group[0]-1:group[1]]
				temp=[]
				for res in shortlist:
					temp.append(res +'_' +atom)
				df2 = df.reindex(temp)
				xval = Xval[group[0]-1:group[1]]
				fig_width = 0.78 + 0.065 *len(df2.resid.tolist())
				if fig_width < 3.0: 
					fig_width = 3.0
				fig=plt.figure(figsize=(fig_width,5))
				ax = fig.add_subplot(111)
				DataSets = DataSetList[1:]
				width = 0.9/len(DataSets)
				for x in range(len(DataSets)):
					colors.append(ColorsDict[DataSets[x]])
					ax.bar(xval + x * width, df2[DataSets[x]], width, color=ColorsDict[DataSets[x]], edgecolor='none', label = Legend_dict[DataSets[x]])
				if dtype == 'CS_wavg':
					ax.axhline(y = Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
				if dtype == 'CS':
					ax.axhline(y = Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
					ax.axhline(y = -1* Cutoff_dict[atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
					ax.axhline(y = 0.0, color = [0,0,0])
				#ax.set_xticks(xval + (width * (len(DataSets)/2.0)))
				ax.set_ylim(Plot_Dict[atom[0] + dtype][0])
				ax.set_ylabel(Plot_Dict[atom[0] + dtype][1])
				ax.set_xlim(min(xval) - 2, max(xval) + len(DataSets) + 2)
				ax.set_xlabel('Residue Number')
				ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
				ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
				ax.set_title(Title_label[atom])
				ax.yaxis.set_major_formatter(FormatStrFormatter('%3.2f'))
				legend = ax.legend(loc='best', frameon=False, markerscale=0.000001)
				for color,text in zip(colors, legend.get_texts()):
					text.set_color(color)
				plt.tight_layout(pad = 0.4, w_pad = 0.4, h_pad = 0.4)
				pdf.savefig()
				plt.close()
		pdf.close()
	print("Finished Summary Bar Plots v2 ")

# ### Generating bins for assigning colors in pymol scripts, using the range of CSP observed, and 8 bins total. 	
# 	for dtype in Data_Types:
# 		df = eval('DF_' + dtype)
# 		nuc_list = df.nuc.unique().tolist()
# 		for nuc in nuc_list:
# 			exec(nuc +  '_' + dtype + '_Bins = [0.0]')
# 			temp = eval(nuc + '_' + dtype + '_Bins')  
# 			val_list = []
# 			df1 = df[df['nuc'] == nuc].copy(deep=True)
# 			for res in df1.index.tolist():
# 				for dataset in DataSetList[1:]:
# 					if abs(df1.loc[res,dataset]) > 0.01 :
# 						val_list.append(abs(df1.loc[res,dataset]))
# 			mean = np.round(trim_mean(val_list,0.1),2)
# 			for i in range(6)[1:]:
# 				temp.append(1.00* i*mean)
# 			temp.sort()
# 			print nuc
# 			print mean
# 			print temp

# 	outstr = '''set ray_opaque_background, 0
# set depth_cue, off
# set ray_opaque_background, 0
# set ray_shadows, 0
# set orthoscopic, on
# hide all
# color gray70, all
# show cartoon
# set sphere_scale, 0.4
# set_color color0 = [1.00, 0.85, 0.46]
# set_color color1 = [1.00, 0.70, 0.30]
# set_color color2 = [0.99, 0.55, 0.24]
# set_color color3 = [0.98, 0.31, 0.165]
# set_color color4 = [0.74, 0.00, 0.15]
# set_color color5 = [0.51, 0.00, 0.156]
# '''
# 	for dtype in ['CS_wavg']:
# 		df = eval('DF_' + dtype)
# 		for DataSet in DataSetList[1:]:
# 			if not os.path.exists(outpath + 'Pymol_Scripts/'):
# 				os.makedirs(outpath + 'Pymol_Scripts/')
# 			pymol_script = open(outpath + 'Pymol_Scripts/' + DataSet + '_' + dtype +'_Pymol.txt','w')
# 			pymol_script.write(str(outstr))
# 			for row  in df.dropna(subset=[DataSet]).index.tolist():
# 				bins = eval(df.loc[row,'nuc'] + '_' + dtype + '_Bins')
# 				for i in range(len(bins)):
# 					if df.loc[row,DataSet] >= bins[i]:
# 						color = 'color' + str(i)
# 					elif df.loc[row,DataSet] == np.nan: 
# 						break
# 				pymol_script.write('color '+ color +', resi '+str(df.loc[row,'resid'][1:]) + '\n')

# 			pymol_script.close()
	
	os.system("rm NODA_temp_Data_input")

####----------------------------------------------------------------------------------------------####
##					 Save these results to a CSV file for each Data Type 							##
####----------------------------------------------------------------------------------------------####

	if not os.path.exists(outpath + 'CSV_Files/'):
		os.makedirs(outpath + 'CSV_Files/')
	for dtype in Data_Types:
		Cutoff_dict = eval('Cutoff_' + dtype + '_dict')
		DF = eval('DF_' +dtype)
		DF.to_csv(outpath + 'CSV_Files/' +  OutName + '_' + dtype + '_results.csv')
		f = open(outpath + 'CSV_Files/' +  OutName + '_' + dtype + '_results.csv','a')
		for nuc in df['nuc'].unique().tolist():
			# Bins = eval(nuc + '_' + dtype + '_Bins') 
			f.write("%s cutoff,  %0.4f\n" % (nuc, Cutoff_dict[nuc]))
			# f.write("%s Pymol Bins, % " (nuc, Bins))
		f.close()


	print("Finished")