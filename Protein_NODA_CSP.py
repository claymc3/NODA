import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import trim_mean
from scipy.stats import tstd
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker 
import matplotlib.colors
import re



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
 'F': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CD1':'CDAro', 'HD1':'HDAro', 'CD2':'CDAro', 'HD2':'HDAro', 'CE1':'CEAro','CE':'CEAro','HE':'HEAro', 'HE1':'HEAro','CE':'CEAro', 'HE':'HEAro', 'CE2':'CEAro', 'HE2':'HEAro', 'CZ':'CZAro', 'HZ':'HZAro'},
 'G': {'H':'H', 'N':'N', 'CA':'CA', 'HA2':'HA', 'HA3':'HA'},
 'H': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'ND1':'Nside', 'HD1':'Hside', 'CD2':'CDAro', 'HD2':'HDAro', 'CE1':'CEAro', 'HE1':'HEAro', 'NE2':'Nside', 'HE2':'Hside'},
 'I': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA','HB':'HB', 'CB':'CB', 'CG1':'CG', 'HG12':'HG', 'HG13':'HG', 'CG2':'CMe', 'HG2':'HMe', 'CD1':'CMe', 'HD1':'HMe'},
 'K': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG2':'HG', 'HG3':'HG', 'CD':'CD','HD2':'HD', 'HD3':'HD', 'CE':'CE', 'HE2':'HE', 'HE3':'HE', 'NZ':'Nside', 'HZ':'Hside'},
 'L': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG':'HG', 'CD1':'CMe', 'HD1':'HMe', 'CD2':'CMe', 'HD2':'HMe'},
 'M': {'H':'H', 'N':'N', 'HA':'HA', 'HB2':'HB', 'HB3':'HB', 'HG2':'HG', 'HG3':'HG', 'HE':'HMe', 'CA':'CA', 'CB':'CB', 'CE':'CMe', 'CG':'CG'},
 'N': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'ND2':'Nside', 'HD21':'Hside', 'HD22':'Hside'},
 'P': {'CA':'CA','HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG2':'HG', 'HG3':'HG', 'CD':'CD','HD2':'HD', 'HD3':'HD'},
 'Q': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CG':'CG', 'HG2':'HG', 'HG3':'HG', 'NE2':'Nside', 'HE21':'Hside', 'HE22':'Hside'},
 'R': {'H':'H', 'N':'N', 'HA':'HA', 'HB2':'HB', 'HB3':'HB', 'HD2':'HD', 'HD3':'HD', 'CG':'CG', 'HG2':'HG', 'HG3':'HG','NE':'Nside','HE':'Hside','NH1':'Nside', 'HH11':'Hside', 'HH12':'Hside', 'NH2':'Nside', 'HH21':'Hside', 'HH22':'Hside',  'CA':'CA', 'CB':'CB', 'CD':'CD'},
 'S': {'H':'H', 'N':'N', 'CA':'CA','HA':'HA','CB':'CB', 'HB2':'HB', 'HB3':'HB'},
 'T': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB':'HB', 'CG2':'CMe', 'HG2':'HMe'},
 'V': {'H':'H', 'N':'N', 'CA':'CA','HA':'HA', 'CB':'CB', 'HB':'HB', 'CG1':'CMe', 'HG1':'HMe', 'CG2':'CMe', 'HG2':'HMe'},
 'W': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA','CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CD1':'CDAro', 'HD1':'HDAro', 'NE1':'Nside', 'HE1':'Hside', 'CE3':'CEAro', 'HE3':'HEAro', 'CH2':'CHAro', 'HH2':'HEAro', 'CZ2':'CZAro', 'HZ2':'HZAro', 'CZ3':'CZAro', 'HZ3':'HZAro'},
 'Y': {'H':'H', 'N':'N', 'CA':'CA', 'HA':'HA', 'CB':'CB', 'HB2':'HB', 'HB3':'HB', 'CD1':'CDAro', 'HD1':'HDAro', 'CD2':'CDAro', 'HD2':'HDAro', 'CE1':'CEAro', 'HE1':'HEAro', 'CE2':'CEAro', 'HE2':'HEAro','CE':'CEAro', 'HE':'HEAro'}
 }

Format_Name = {'uM':r'$\mu$M','MnCl2':r'MnCl$_{2}$','MgCl2':r'MgCl$_{2}$','CaCl2':r'CaCl$_{2}$','ZnCl2':r'ZnCl$_{2}$','BaCl2':r'BaCl$_{2}$',"H2O":r"H$_{2}$O","D2O":r"D$_{2}$O", 'alpha':r"$\alpha$", 'beta':r"$\beta$",'delta':r"$\Delta$"}


####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##  Function for placing labels on graphs for data points with absolute values greater than the     ##
##  significance cutoff values                                                                      ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####
def autolabel(rects,labels,cutoff,ax):
  for idx,rect in enumerate(rects):
    height = rect.get_height()
    if abs(height) > cutoff:
      if height > 0:
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, labels[idx], ha='center', va='bottom', rotation=90, fontsize=6)
      if height < 0:
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, labels[idx], ha='center', va='top', rotation=90, fontsize=6)
def Plot_CSP_data(OutName,Sequence,Start_Index,Residues,Data_Types,SDM,Show_Labels,Intensity_Norm,Common_Scale,input_path,PDB, MasterList, outpath,CS_min_max,Intensity_min_max):

  DF_Volume = pd.DataFrame()
  DF_Intensity = pd.DataFrame()
  DF_CS = pd.DataFrame()
  DF_LW = pd.DataFrame()
####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##  Generates all the run parameters from the *_input.txt file                                      ##
##  RunParams for CSP : OutDir, OutName, Sequence, Start_Index, Residues, Data_Types, SDM,          ##
##                      Show_Lables,Intensity_Norm(s), input_path                                   ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####
  CXC_limits = {}
  sequence = ''
  Seq = []
  ResIDlist = []
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
## This pulse out the residues the user has indicated that they want to use in the Residues entry of the input file
  ResIDlist = []
  for i in range(len(Seq)): 
    if int(Seq[i][1:])>= int(Residues.split('-')[0]) and int(Seq[i][1:]) <= int(Residues.split('-')[1]):
      ResIDlist.append(Seq[i])
### Make groups of residies to make veiwing easier for large systems. 
  Int_Norm,Int_lim ={},[]
  if 'Intensity' in Data_Types or 'Volume' in Data_Types:

    for i in range(len(Intensity_Norm)):
      Int_Norm[Intensity_Norm[i].split('-')[1][0]]= Intensity_Norm[i].replace('-','_')

  if 'CS' in Data_Types:
    CXC_limits['CS_min'] = float(CS_min_max[0])
    CXC_limits['CS_max'] = float(CS_min_max[1])
    CXC_limits['CS_wavg_min'] = float(CS_min_max[0])
    CXC_limits['CS_wavg_max'] = float(CS_min_max[1])
  if 'Intensity' in Data_Types:
    CXC_limits['Intensity_min'] = float(Intensity_min_max[0])
    CXC_limits['Intensity_max'] = float(Intensity_min_max[1])
  if 'Volume' in Data_Types:
    CXC_limits['Volume_min'] = float(Intensity_min_max[0])
    CXC_limits['Volume_max'] = float(Intensity_min_max[1])
####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
## Using the MasterList information generate list of DataSet names (DataSetList) which serve as     ##
## column names in pandas dataframe and keys in dictionaries for plotting.                          ##
## Formatted titles and legend entries and storing them in dictionaries with DataSet as keys        ##
## Finally assigning uniqu color to each dataset in the ColorsDict.                                 ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####
  ColorsDict, Legend_dict, Title_dict, References= {}, {}, {}, {}
  Refs = []
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
      References[DataSet[0]] = DataSet[4]
      Refs.append(DataSet[4])
    if len(DataSet) == 4:
      References[DataSet[0]]= MasterList[0][0]
      Refs.append(MasterList[0][0])
    legend = DataSet[1]
    for src, target in Format_Name.items():
      legend = legend.replace(src, target)
    Legend_dict[DataSet[0]] = legend
    Title_dict[DataSet[0]] = legend

## Remove reference data sets from plot list. Getting rid of white space. 
  Plot_DataSetList = []
  for DataSet in DataSetList:
    if DataSet not in Refs:
      Plot_DataSetList.append(DataSet)
## Make a list all possible combinations of resID and atoms to serve as index in DataFrame 

####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##                              Sorting Input Data in to pands DataFrame                            ##
##                                                                                                  ##
##      For each type of data requested a separate pandas DataFrame object DF_DataType with index   ##
## set = ResID_atom and one column for each sample/condition (dataset).                             ##
## The Sort_dic[key]=Sort_dic_full[key][Sort_dic_index[type]]statement selects the                  ##
## appropriate column from input file.                                                              ##
## resid, atom and atom_sort columns are used for plotting and sorting purposes                     ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####
  ## creating empty data frame to store results in, must exist before use
  dataframs,H2X = {},{}
  for dtype in Data_Types:
    dataframs['DF_' + dtype] = pd.DataFrame()
    dataframs['DF_' + dtype + '_plot'] = pd.DataFrame()
  ## Pars the input sparky list and sort data into approreate DataFrame 
  for DataSet in MasterList:
    InputFiles = glob.glob(os.path.join(input_path + DataSet[2].strip()))
    if len(InputFiles) == 0: 
      print('No Data found check input file path {:}{:}'.format(input_path, DataSet[2]))
      exit()
    if DataSet[0] not in DataSetList:
      DataSetList.append(DataSet[0])
    for file in InputFiles:
      print("Reading in {:}".format(file.split('/')[-1]))
      in_df = pd.read_csv(file,sep=r'\s+',names=open(file).readline().rstrip().replace('Data Height', "Intensity").replace('Volume', "Volume  fit").replace('(hz)', "").split(),skiprows=[0,1],engine='python')
      for i, row in in_df.iterrows():
        if '?' not in row['Assignment']:
          if re.findall('[A-Z]([0-9]*)[A-Z]', row['Assignment']):
            if not re.findall('[a-z]', row['Assignment']):
              group = re.search('[A-Z]([0-9]*)', row['Assignment']).group()
              resid = group
              Assignment =  row['Assignment'].replace(group, group + '_')
              in_df.loc[i, 'Group'] = Assignment.split('_')[0]
              in_df.loc[i, 'Atoms'] = Assignment.split('_')[1]
              in_df.loc[i, 'resid'] = resid
              in_df.loc[i, 'resi'] = int(resid[1:])
            if re.findall('[a-z]', row['Assignment']):
              print("bad assignment format {:}".format(row['Assignment']))
      for dtype in Data_Types:
        df = dataframs['DF_' + dtype]
        for i, row in in_df.dropna(axis=0, subset=['resid']).iterrows():
          if in_df.loc[i,'resi'] >= int(Residues.split('-')[0]) and in_df.loc[i,'resi'] <= int(Residues.split('-')[1]):
            Sorting_dict = Allowed_atoms[row['Group'][0]]
            if dtype == 'CS' or dtype == 'LW':
              if row['Group'] + '_' + row['Atoms'].split("-")[1] not in H2X.keys():
                H2X[row['Group'] + '_' + row['Atoms'].split("-")[1]] = row['Group'] + '_' + row['Atoms'].split("-")[0]
              for y in range(len(row['Atoms'].split('-'))):
                if row['Atoms'].split('-')[y] in Sorting_dict.keys():
                  dfidx = row['Group'] + '_' + row['Atoms'].split("-")[y]
                  df.loc[dfidx , 'resid'] = row['resid']
                  df.loc[dfidx , 'resi'] = row['resi']
                  df.loc[dfidx , 'nuc'] = row['Atoms'].split("-")[y][0]
                  df.loc[dfidx , 'name'] = row['Atoms'].split("-")[y]
                  df.loc[dfidx , 'atom_sort'] = Sorting_dict[row['Atoms'].split('-')[y]]
                  df.loc[dfidx , DataSet[0]] = np.around(row[dtype.replace('CS','w').replace('LW','lw')+ str(y+1)],3)
                if row['Atoms'].split('-')[y] not in Sorting_dict.keys():
                  print('Assignment atom name error {:}'.format(row['Assignment']))
            if dtype == 'Intensity' or dtype == 'Volume':
              if row['Atoms'].split('-')[0] in Sorting_dict.keys():
                dfidx = row['Group'] + '_' + row['Atoms'].split("-")[0]
                df.loc[dfidx , 'resid'] = row['resid']
                df.loc[dfidx , 'resi'] = row['resi']
                df.loc[dfidx , 'nuc'] = row['Atoms'].split("-")[0][0]
                df.loc[dfidx , 'name'] = row['Atoms'].split("-")[0]
                df.loc[dfidx , 'atom_sort'] = Sorting_dict[row['Atoms'].split('-')[0]]
                df.loc[dfidx , DataSet[0]] = np.around(row[dtype],3)
        # print(df)
  ## Creating a clean copy of the data frame omitting my rows that are not populated by the reference (first) data set 
  for dtype in Data_Types:
    df = dataframs['DF_' + dtype]
    df2 = df.dropna(axis = 0, subset=[DataSetList[1]]).copy(deep=True)
  DF_CS_original = DF_CS.copy(deep = True)

####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##       Calculate delta omega for all data sets relative to the first (reference) data set         ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####


  if 'CS' in Data_Types:
    Ref_CS = dataframs['DF_CS'].copy(deep=True)
    CS_init = dataframs['DF_CS']
    # print(Ref_CS)
    for DataSet in DataSetList:
      Ref_DS = References[DataSet]
      CS_init[DataSet] = CS_init[DataSet] - Ref_CS[Ref_DS]
    
    DF_CS = CS_init.copy(deep = True)
    print('Finished CS Analysis')

####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##       Calculate weighted chemical shift difference for all data sets relative to their           ##
##      specified reference data set. Reference data sets will not be plotted                       ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####
    DF_CS = dataframs['DF_CS']
    Delta_CS_H = DF_CS[DF_CS['nuc'] == 'H'].copy(deep = True)
    Delta_CS_X =DF_CS[DF_CS['nuc'] != 'H'].copy(deep = True)
    CS_wavg = DF_CS[DF_CS['nuc'] != 'H'].copy(deep = True)
    for DataSet in DataSetList:
      for prot in H2X.keys():
        if CS_wavg.loc[H2X[prot],'nuc'] == 'C':
          CS_wavg.loc[H2X[prot], DataSet] = np.sqrt((DF_CS.loc[prot, DataSet])**2 + 0.3 * (DF_CS.loc[H2X[prot],DataSet])**2)
        if CS_wavg.loc[H2X[prot],'nuc'] == 'N':
          CS_wavg.loc[H2X[prot], DataSet] = np.sqrt((DF_CS.loc[prot, DataSet])**2 + 0.15 * (DF_CS.loc[H2X[prot],DataSet])**2)
    DF_CS_wavg = CS_wavg.copy(deep=True)
    dataframs['DF_CS_wavg'] = DF_CS_wavg
    DF_CS_wavg_plot = pd.DataFrame(columns=['atom_sort'])
    dataframs['DF_CS_wavg_plot'] = DF_CS_wavg_plot
    Data_Types.insert(Data_Types.index('CS') +1, 'CS_wavg')
    print('Finished CS_wavg Analysis')
####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##    Calculate relative line width for all data sets relative to the first (reference) data set    ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####
  if 'LW' in Data_Types:
    DF_LW = dataframs['DF_LW']
    Ref_LW=DF_LW.copy(deep=True)
    LW=DF_LW
    for DataSet in DataSetList:
      Ref_DS = References[DataSet]
      LW[DataSet]=LW[DataSet] / Ref_LW[Ref_DS]
    print('Finished LW Analysis')
####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##                               Internal Intensity/Volume normalization                            ##
##  Peak intensity/volume normalization is done based specified residue in input file for each      ##
##  for methyls and amid peaks.                                                                     ##
##                                                                                                  ##
##      Data is normalized independently for each DataSet relative to the specified residue. Then   ##
##  each each residue in each data set is compared to its counter part in the specified reference   ##
##  dataset(s)                                                                                      ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####

  if 'Intensity' in Data_Types:
    DF_Intensity = dataframs['DF_Intensity']
    Ref_Int = DF_Intensity.copy(deep=True)
    Int_df = DF_Intensity
    ## Performing intral normalization
    if len(Int_Norm) != 0:
      for DataSet in DataSetList:
        for row in Int_df.index.tolist():
          Norm = Int_Norm[row.split('_')[1][0]]
          refval = Ref_Int.loc[Norm, DataSet] * 1.00
          Int_df.loc[row,DataSet] = Int_df.loc[row,DataSet] /refval
    Ref_Intensity = DF_Intensity.fillna(value=0.01,axis=0).copy(deep = True)
    Int_df = DF_Intensity.fillna(value=0.01,axis=0).copy(deep = True)
    for DataSet in DataSetList:
      Int_df[DataSet] = 1.00 - Int_df[DataSet] / Ref_Intensity[References[DataSet]]
    DF_Intensity = Int_df
    print('Finished Intensity Analysis')


  if 'Volume' in Data_Types:
    DF_Volume = dataframs['DF_Volume']
    Ref_Vol = DF_Volume.copy(deep=True)
    Vol_df = DF_Volume
    if len(Int_Norm) != 0:
      for DataSet in DataSetList:
        for row in Vol_df.index.tolist():
          Norm = Int_Norm[row.split('_')[1][0]]
          refval = Ref_Vol.loc[Norm, DataSet] * 1.00
          Vol_df.loc[row,DataSet] = Vol_df.loc[row,DataSet] /refval
    Ref_Volume = DF_Volume.fillna(value=0.01,axis=0).copy(deep = True)
    Vol_df = DF_Volume.fillna(value=0.01,axis=0).copy(deep = True)
    for DataSet in DataSetList:
      Vol_df[DataSet] = 1.00 - Vol_df[DataSet] / Ref_Volume[References[DataSet]]
    DF_Volume = Vol_df
    print('Finished Volume Analysis')
####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##                                  Calculate significance cut off                                  ##
## A separate cut off is calculated for 1H, 13C, and 15N                                            ##
## The cut off is = 2* std(Trimmed data Set)                                                        ##
## All values that are less than 0.7% of maximum observed CSP are used so that an inordinately      ##
## std is not obtained due to out liers in the normal spread of the data.                           ##
## Essentially excluding outliers in the dw spread.                                                 ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####
  MasterDict = {}
  for dtype in Data_Types:
    MasterDict['Cutoff_' + dtype + '_dict'] = {}
    Cutoff_dict = MasterDict['Cutoff_' + dtype + '_dict']
    df = dataframs['DF_' + dtype]
    for nuc in df['nuc'].unique().tolist():
      df1 = df[df['nuc']== nuc ].copy(deep = True)
      Values2 = []
      for DataSet in Plot_DataSetList:
        Values = [np.round(abs(val),4) for val in df1[DataSet].dropna().tolist() if val != 0.0]
        Values2.extend(Values)
        Trimmed_values_1 = [Values[i] for i in range(len(Values)) if Values[i] < 3 * np.std(Values)]
        if len(Trimmed_values_1) != 0:
          Trimmed_values = [Trimmed_values_1[i] for i in range(len(Trimmed_values_1)) if Trimmed_values_1[i] < 2 * np.std(Trimmed_values_1)]
          if len(Trimmed_values) != 0:
            Cutoff_dict[DataSet+nuc] = int(SDM) * np.std(Trimmed_values)
          else: Cutoff_dict[DataSet+nuc] = int(SDM) * np.std(Trimmed_values_1)
        elif len(Trimmed_values_1) == 0:
          Cutoff_dict[DataSet+nuc] = int(SDM) * np.std(Values)
      Trimmed_values_2 = [Values2[i] for i in range(len(Values2)) if Values2[i] < 3 * np.std(Values2)]
      if len(Trimmed_values_2) != 0:
        Trimmed_values2 = [Trimmed_values_2[i] for i in range(len(Trimmed_values_2)) if Trimmed_values_2[i] < 2 * np.std(Trimmed_values_2)]
        if len(Trimmed_values2) != 0:
          Cutoff_dict['Common_'+nuc] = int(SDM) * np.std(Trimmed_values2)
        else: Cutoff_dict['Common_'+nuc] = int(SDM) * np.std(Trimmed_values_2)
      elif len(Trimmed_values_2) == 0:
        Cutoff_dict['Common_'+nuc] = int(SDM) * np.std(Values2)

####----------------------------------------------------------------------------------------------####
##                   Save these results to a CSV file for each Data Type                            ##
####----------------------------------------------------------------------------------------------####
  out_Columns = ['resid']
  for DataSet in Plot_DataSetList:
    out_Columns.append(DataSet)
  for dtype in Data_Types:
    Cutoff_dict = MasterDict['Cutoff_' + dtype + '_dict']
    DF = dataframs['DF_' + dtype]
    DF = DF.sort_values(by=['atom_sort', 'resi'])
    for nuc in df['nuc'].unique().tolist():
      for DataSet in Plot_DataSetList:
        DF.loc['Cutoff-' + nuc, DataSet] = np.round(Cutoff_dict[DataSet+nuc],4)
    if not os.path.exists(outpath + 'CSV_Files/'):
      os.makedirs(outpath + 'CSV_Files/')
    DF.to_csv(outpath + 'CSV_Files/' +  OutName + '_' + dtype + '_results.csv', columns = out_Columns)

####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##      For geminal pairs select the entry with the largest perturbation and store it in a new      ##
##  data frame df_dtype_plot which will be used for plotting the data and generating PyMol scripts  ##
##  This leaves the full list intact, but simplifies plots.                                         ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####

  for dtype in Data_Types:
    # print(dtype)
    df = dataframs['DF_' + dtype]
    df_plot = dataframs['DF_' +dtype + '_plot']
    for atom in df['atom_sort'].unique().tolist():
      df_atoms = df[df['atom_sort']== atom ].sort_values(by=['resi']).index.tolist()
      df_atoms.append(df_atoms[-1])
      for DataSet in Plot_DataSetList:
        used = []
        for i in range(len(df_atoms))[:-1]:
          if df_atoms[i] not in used:
            if df.loc[df_atoms[i],'resid'] != df.loc[df_atoms[i+1],'resid']:
              df_plot.loc[df.loc[df_atoms[i],'resid']+'_'+atom, 'resid'] = df.loc[df_atoms[i],'resid']
              df_plot.loc[df.loc[df_atoms[i],'resid']+'_'+atom, 'name'] = df.loc[df_atoms[i],'name']
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
##                                                                                                  ##
##      Generating dictionaries to control axis limits and titles in scatter and bar plots          ##
##      based on current data. This will keep everything plotted on the same relative scale         ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####

  Delta_ppm_dict = {'H':r" ($^{1}$H, ppm)",'C':r" ($^{13}$C, ppm)",'N':r" ($^{15}$N, ppm)"}
  nuc_dict = {'H':r" ($^{1}$H)",'C':r" ($^{13}$C)",'N':r" ($^{15}$N)"}
  Delta_wavg_dict = {'C':r"$\Delta\delta$$_{avg}$ (HC, ppm)", 'N': r"$\Delta\delta$$_{avg}$ (HN, ppm)" }
  Title_label = {'N' : "Backbone",'H' : "Backbone", 'CMe': r"Methyl", "HMe": r"Methyl", "Me": "Methyl Intensity", "NE1":"W-NE1", "HE1":"W-HE1",
           "Nside":'Side Chain N', "Hside":'Side Chain H', 'HB':'Side Chain HB2/HB3', 'HA':'Side Chain HA2/HA3', 'HG':'Side Chain HG2/HG3','CB':'Side Chain CB', 'CA':'Side Chain CA', 'CG':'Side Chain CG', 'CAro':'F/W CE1/CE2', 'CDAro':'F-CD1/2 Y-CD1/2 H-CD2','HDAro': 'F-HD1/2 Y-HD1/2 H-HD2', 'HEAro':'F-HE1/2 Y-HE1/2 H-HE1','CEAro':'F-CE1/2 Y-CE1/2 H-HE1'}
  Plot_Dict = {}
  for dtype in Data_Types:
    df = dataframs['DF_' + dtype + '_plot']
    for nuc in df['nuc'].unique().tolist():
      df1 = df[df['nuc'] == nuc].copy(deep=True)
      Values = []
      for DataSet in Plot_DataSetList:
        Values.extend([np.round(abs(val),4) for val in df1[DataSet].dropna().tolist() if val != 0.0])
      # print(Values)
      # print(trim_mean(Values, 0.1))
      bmax = trim_mean(Values, 0.1)
      if np.max(df1.max(skipna=True, numeric_only=True)) > 0.02:
        vmax = (np.max(df1.max(skipna=True, numeric_only=True)) * 12.00) / 10.00
      else: vmax = 0.02
      if abs(np.min(df1.min(skipna=True, numeric_only=True))) > 0.02:
        vmin = (np.min(df1.min(skipna=True, numeric_only=True)) * 12.00 )/ 10.00
      else: vmin = 0.0
      if dtype == 'CS':
        Plot_Dict[nuc + dtype]=[[vmin, vmax], r"$\Delta\delta$"+ Delta_ppm_dict[nuc],bmax]
      elif dtype == 'CS_wavg':
        Plot_Dict[nuc + dtype]=[[0.0, vmax], Delta_wavg_dict[nuc],bmax]
      elif dtype == 'LW':
        Plot_Dict[nuc + dtype]=[[0.0, vmax], nuc_dict[nuc] + r" LW/LW$_{0}$ ",1.0]
      elif dtype == 'Intensity':
        # if vmin < -1.0: vmin = -1.0
        if vmax < 1.0: vmax = 1.1
        Plot_Dict[nuc + dtype]=[[vmin, vmax], r"1 - $\bar I$/$\bar I$$_{0}$",1.0]
      elif dtype == 'Volume':
        Plot_Dict[nuc + dtype]=[[0.0, (np.max(df1.max(skipna=True, numeric_only=True)) * 11.00) / 10.00], r"1 - $\bar V$/$\bar V$$_{0}$",1.0]



####----------------------------------------------------------------------------------------------####
##                                                                                                  ##
##              Setting controlling plot appearance: Font, line widths, ticks, and legend           ##
##                                                                                                  ##
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
  # mpl.mathtext.FontConstantsBase.sup1 = 0.25

####----------------------------------------------------------------------------------------------####
##          Bar plots showing CSP  for each atom observed, with a separate file for each            ##
##  individual dataset. Basically breaking  Name_data_type_bar_plot.pdf file up.                    ##
##  Saved as DataSet_data_type_bar_plot.pdf                                                         ##                                                                                              ##
####----------------------------------------------------------------------------------------------####
  DataSets = Plot_DataSetList
  for x in range(len(DataSets)):
    colors = []
    colors.append(ColorsDict[DataSets[x]])
    pdf = PdfPages(outpath + DataSets[x] + '_plot.pdf')
    for dtype in Data_Types:
      Cutoff_dict = MasterDict['Cutoff_' + dtype + '_dict']
      Xval = range(int(Residues.split('-')[0]), int(Residues.split('-')[1])+1)
      df = dataframs['DF_' + dtype + '_plot']
      AtomsList=df['atom_sort'].unique().tolist()
      for atom in AtomsList:
        temp=[]
        for res in ResIDlist:
          temp.append(res +'_' +atom)
        df2 = df.reindex(temp)
        fig=plt.figure(figsize=(7.28,3))
        ax = fig.add_subplot(111)
        width = 0.9
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
        if Common_Scale.upper() == 'N':
          ymax = (df2[DataSets[x]].max(skipna=True) * 12.0) / 10.00
          if ymax < 0.02: ymax = 0.02
          if abs((df2[DataSets[x]].min(skipna=True) * 12.0) / 10.00) > 0.02:
            ymin = (df2[DataSets[x]].min(skipna=True) * 12.0) / 10.00
          else: ymin = 0.0
          ax.set_ylim([ymin, ymax])
          ylimit = [ymin, ymax]
        if Common_Scale.upper() == 'Y':
          ylimit = Plot_Dict[atom[0] + dtype][0]
        ax.set_ylim(ylimit)
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
        #   text.set_color(color)

        plt.text(int(Residues.split('-')[0]),0.9*(ylimit[1]), Legend_dict[DataSets[x]] , {'color': ColorsDict[DataSets[x]], 'fontsize': 10})
        plt.tight_layout(pad = 0.4, w_pad = 0.4, h_pad = 0.4)
        pdf.savefig()
        plt.close()
    pdf.close()

  print("Finished Individual Bar Plots")

####----------------------------------------------------------------------------------------------####
##  Create a separate bar plot for each type of data analyzed and atom observed, with all datasets  ##
##  shown in same plot. Saved as Name_data_type_bar_plot.pdf.                                       ##
####----------------------------------------------------------------------------------------------####
  if len(DataSets) > 1:
    for dtype in Data_Types:
      Cutoff_dict = MasterDict['Cutoff_' + dtype + '_dict']
      colors = []
      pdf = PdfPages(outpath + OutName + "_" + dtype + '_Summary_plot.pdf')
      df = dataframs['DF_' + dtype + '_plot']
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
            ax.axhline(y = Cutoff_dict['Common_'+atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
          if dtype != 'CS_wavg':
            ax.axhline(y = Cutoff_dict['Common_'+atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
            ax.axhline(y = -1* Cutoff_dict['Common_'+atom[0]], color = [0.5,0.5,0.5], alpha = 0.5)
            if Plot_Dict[atom[0] + dtype][0][0] != 0.0: 
              ax.axhline(y = 0.0, color = [0,0,0])
          if Show_Labels.upper() == 'Y':
            autolabel(rects, df2['resid'].tolist(), Cutoff_dict['Common_'+atom[0]], ax)
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
##                                                                                                  ##
##                                          Chimera Scripts                                         ##
##  For each non reference weighted average CSP data set generate a cmx script to color             ##
##   code the CSP changes                                                                           ##
##                                                                                                  ##
##  The results are saved in a separate txt files for each non reference data set, with a separate  ##
##  file for each data type analyzed. The script atomatically only loads a single model and the     ##
##  the user can open each one individually to redenr and image.                                    ##
##                                                                                                  ##
##                                                                                                  ##
## Resonances that were not oberbserved in the apo state are colored gray                           ##
## Resonances that were not found again in the perturbrd sameple are colored purple                 ##
##                                                                                                  ##
####----------------------------------------------------------------------------------------------####
####----------------------------------------------------------------------------------------------####
  Key_text={  'HCS':"2dlab text '\u0394\u03b4 (\u00B9H ppm)' size 24 xpos 0.80 ypos 0.067\n",
        'CCS':"2dlab text '\u0394\u03b4 (\u00B9\u00B3C ppm)' size 24 xpos 0.80 ypos 0.067\n",
        'NCS':"2dlab text '\u0394\u03b4 (\u00B9\u2075N ppm)' size 24 xpos 0.80 ypos 0.067\n",
        'NCS_wavg':"2dlab text '\u0394\u03b4\u2090\u1D65\u2091 (HN ppm)' size 24 xpos 0.79 ypos 0.067\n",
        'CCS_wavg':"2dlab text '\u0394\u03b4\u2090\u1D65\u2091 (HC ppm)' size 24 xpos 0.79 ypos 0.067\n",
        'CIntensity':"2dlab text '1 \u2013 I \u2215 I\u2080' size 24 xpos 0.82 ypos 0.066 margin 6\n",
        'NIntensity':"2dlab text '1 \u2013 I \u2215 I\u2080' size 24 xpos 0.82 ypos 0.066 margin 6\n",
        'CVolume':"2dlab text '1 \u2013 V \u2215 V\u2080' ssize 24 xpos 0.82 ypos 0.066 margin 6\n",
        'NVolume':"2dlab text '1 \u2013 V \u2215 V\u2080' size 24 xpos 0.82 ypos 0.066 margin 6\n",
        'CLW':"2dlab text 'LW \u2215 LW\u2080' size 24 xpos 0.80 ypos 0.067\n",
        'NLW':"2dlab text 'LW \u2215 LW\u2080' size 24 xpos 0.80 ypos 0.067\n",
        'HLW':"2dlab text 'LW \u2215 LW\u2080' size 24 xpos 0.80 ypos 0.067\n"}
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom",['#FED976','#FD8C3C','#FF3700','#800026'])
  # cmap = mpl.colormaps['YlOrRd']
  pdbname = PDB.split('/')[-1].split('.')[0]
  print('Generating ChimeraX Scripts')
  if not os.path.exists(outpath + 'Chimera_Scripts/'):
    os.makedirs(outpath + 'Chimera_Scripts/')

  for DataSet in Plot_DataSetList:
    for dtype in Data_Types:
      if dtype != 'CS':
        vmin = CXC_limits[dtype+'_min']
        vmax = CXC_limits[dtype+'_max']
        df = dataframs['DF_' + dtype + '_plot']
        Cutoff_dict = MasterDict['Cutoff_' + dtype + '_dict']
        AtomsList=df['atom_sort'].unique().tolist()
        for atom in AtomsList:
          mn = 3
          cmxlable = 'label #3:'
          df2 = df[(df['atom_sort'] == atom)].copy(deep=True)
          if Common_Scale.upper() == 'Y':
            cutoff = Cutoff_dict['Common_'+atom[0]]
          if Common_Scale.upper() == 'N':
            cutoff = Cutoff_dict[DataSet+atom[0]]
          print('Processing {:} {:} {:}'.format(DataSet, atom, dtype))
          showlist = 'show #3:'
          outline = ''
          missingdat = []
          ## If the list is occupied make a cxc file
          if len(df2.dropna(subset=[DataSet]).index.tolist()) > 0:
            used = []
            cmx_script = open("{:}Chimera_Scripts/{:}_{:}_{:}.cxc".format(outpath,DataSet,atom,dtype),'w')
            cmx_script.write("#########################################################\n## {:}_{:} Color Coding\n## Range{:} - {:}\n#########################################################\n".format(dtype, atom, vmin, vmax))
            cmx_script.write("key rgb({:3.0f},{:3.0f},{:3.0f}):0.0 rgb({:3.0f},{:3.0f},{:3.0f}):1.0 rgb({:3.0f},{:3.0f},{:3.0f}):2.0  rgb({:3.0f},{:3.0f},{:3.0f}):3.0  rgb({:3.0f},{:3.0f},{:3.0f}):4.0 fontsize 1 colorTreatment distinct numericLabelSpacing equal size 0.20000,0.03000 pos 0.75,0.03\nkey colorTreatment blended\n".format(255*cmap(0.0)[0],255*cmap(0.0)[1],255*cmap(0.0)[2],255*cmap(0.25)[0],255*cmap(0.25)[1],255*cmap(0.25)[2],255*cmap(0.5)[0],255*cmap(0.5)[1],255*cmap(0.5)[2],255*cmap(0.75)[0],255*cmap(0.75)[1],255*cmap(0.75)[2],255*cmap(1.0)[0],255*cmap(1.0)[1],255*cmap(1.0)[2]))
            cmx_script.write(Key_text[atom[0]+dtype])
            cmx_script.write("2dlab text '{:3.2f}' size 20 xpos 0.75 ypos 0.01\n2dlab text '\u2265{:3.2f}' size 20 xpos 0.925 ypos 0.01\n".format(vmin,vmax))
            cmx_script.write('open {:} maxModels 1\nhide #3 target a\n'.format(PDB))
            cmx_script.write('rename #3 {:}_{:}_{:}\ncolor #{:} gray(150)\n'.format(DataSet,dtype,atom,mn))
            cmx_script.write("show nucleic\nhide #{:}: protein|solvent|H target as\nsurface hide\nstyle (protein|nucleic|solvent) & @@draw_mode=0 stick\ncartoon style modeHelix tube sides 20\ngraphics silhouettes tru\ncartoon style radius 1.5\nset bgColor white\n".format(mn))
            for row in df2.index.tolist():
              if np.isnan(df2.loc[row,DataSet]):
                missingdat.append(df2.loc[row,'resid'][1:])
                used.append(row)
            for row in df2.index.tolist():
              if row not in used:
                if abs(df.loc[row,DataSet]) >= cutoff:
                  cmxlable = cmxlable + df2.loc[row,'resid'][1:] + ","
                norm = (abs(df2.loc[row,DataSet])-vmin)/(vmax-vmin)
                if abs(df.loc[row,DataSet]) >= vmin:
                  if atom in ['CMe','HMe','CDAro','HDAro', 'CEAro', 'HEAro']:
                    cmx_script.write("## {:} {:4.3f}\n".format(row,df2.loc[row,DataSet]))
                    cmx_script.write("color #{:}:{:} rgb({:3.0f},{:3.0f},{:3.0f}) target a\n".format(mn,df2.loc[row,'resid'][1:],255*cmap(norm)[0],255*cmap(norm)[1],255*cmap(norm)[2]))
                    showlist = showlist + df2.loc[row,'resid'][1:] + ','
                  else:
                    cmx_script.write("## {:} {:4.3f}\n".format(row,df2.loc[row,DataSet]))
                    cmx_script.write("color #{:}:{:} rgb({:3.0f},{:3.0f},{:3.0f}) target c\n ".format(mn,df2.loc[row,'resid'][1:],255*cmap(norm)[0],255*cmap(norm)[1],255*cmap(norm)[2]))
            if len(missingdat) > 0:
              cmx_missing_outline = 'color #{:}:'.format(mn)
              for res in missingdat:
                cmx_missing_outline = cmx_missing_outline + '{:},'.format(res)
                if atom in ['CMe','HMe','CDAro','HDAro', 'CEAro', 'HEAro']:
                  showlist = showlist + df2.loc[row,'resid'][1:] + ','
              cmx_missing_outline = cmx_missing_outline[:-1] + ' color purple\n'
              cmx_script.write(cmx_missing_outline)
              # if atom in ['CMe','HMe']:
              #     cmx_script.write('size stickRadius 0.27\nsize atomRadius 2.7\nsize :*@CA,CB,O*,N,C atomRadius 0.7\nsize :ala@CB atomRadius 2.7\nsize :met@CG,SD atomRadius 0.7\nsize :leu,ile@CG* atomRadius 0.7\nsize :phe,try@CD*,CG*,CZ* atomRadius 0.7\n')
              #     cmx_script.write(cmxlable[:-1].replace('label','show') + ' target a\n')
              # if atom in ['CDAro','HDAro', 'CEAro', 'HEAro']:
              #     cmx_script.write('size stickRadius 0.27\nsize atomRadius 2.7\nsize :*@CA,CB,O*,N,C atomRadius 0.7\nsize :ala@CB atomRadius 2.7\nsize :met@CG,SD atomRadius 0.7\nsize :leu,ile@CG* atomRadius 0.7\nsize :phe,try@CD*,CG*,CZ* atomRadius 0.7\n')
              #     cmx_script.write(cmxlable[:-1].replace('label','show') + ' target a\n')
            cmx_script.write(cmxlable[:-1] + ' text "{0.label_one_letter_code}{0.number}"\n')
            cmx_script.write(showlist[:-1]+'\n')
            cmx_script.write('hide H\nlighting gentle\n')
            cmx_script.write('#save {:}_{:}_{:}.tiff format tiff transparentBackground true\n'.format(DataSet,dtype, atom))
            cmx_script.write('\n\n')
            
            # cmx_script.write('size atomRadius 2\nhide :ala@CA\nhide :thr@CA,CB,OG1\nhide :leu,ile@CA,CB,CG*\nhide :met@CA,CB,CG,SD\nhide :val@CA,CB\n')
            cmx_script.close()

  print("Finished")
