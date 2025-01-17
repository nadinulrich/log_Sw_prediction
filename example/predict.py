from datetime import datetime
import deepchem as dc
import numpy as np 
import pandas as pd
import csv
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models import KerasModel
from pathlib import Path 
from classes.mygraphconvmodel import MyGraphConvModel

#functions
def datasets(Extdataset_csv,ListPredicts, result_dir):
    now = datetime.now()                                          
    filename = "Result_"+now.strftime("%d%m%Y%H%M%S")   
    df1      = pd.DataFrame
    result = pd.DataFrame
    df1 = pd.read_csv(Extdataset_csv, encoding='utf-8', header=None)

    DataFrameList = []
    for ExtPredicts_csv in ListPredicts:
       DataFrameList.append( pd.read_csv(ExtPredicts_csv, encoding='utf-8', header=None))

    prd = pd.concat( DataFrameList, axis=1)
    df_median = prd.mean(axis=1)  
    df_std =  prd.std(axis=1,ddof=0)  
  
    result = pd.concat([df1, df_median,df_std ], axis=1)

    try:
      result.to_excel(result_dir+filename+'.xlsx', index = False, header = False)      
    except:
        print("Error writing Excel file")

def read_data(input_file_path, output_file_path):
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=prediction_tasks, feature_field="SMILES", featurizer=featurizer)
    dataset = loader.featurize(input_file_path, shard_size=8192)
    
    trainarray=[]
    smiles = dataset.ids
    
    for i in range(dataset.ids.size):
      entry=[str(smiles[i])]
      trainarray.append(entry)

    with open(output_dir+output_file_path,mode='w') as csv_file:
        wr=csv.writer(csv_file,quoting=csv.QUOTE_NONE)
        wr.writerows(trainarray)

    return dataset

def make_predictions(model,dataset_ext,batch_size, read_out_string ):
  predictions = model.predict_on_generator(data_generator(dataset_ext,batch_size=batch_size))
  write_predictions(predictions, 'Prediction'+read_out_string+'.csv')
 
def data_generator(dataset,batch_size,epochs=1,modelX = None):
  for epoch in range(epochs):
    if modelX is not None:
      print("start epoch "+ str(epoch+1))
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size,deterministic=True, pad_batches=True)):
      multiConvMol = ConvMol.agglomerate_mols(X_b)
      inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
      for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
      labels = [y_b]
      weights = [w_b]
      yield (inputs, labels, weights)
   
def write_predictions(dataset, file_name):
  with open(output_dir+file_name,mode='w') as csv_file:
    wr=csv.writer(csv_file,quoting=csv.QUOTE_ALL)
    wr.writerows(dataset)

#Constants
prediction_tasks = []
basis_dir =  str(Path(__file__).parent.absolute())
model_dir_basis = basis_dir+'/models/split'
output_dir = basis_dir+'/working/'
result_dir = basis_dir+'/result/'
input_dir = basis_dir+'./input/'

ext_dataset = read_data(input_dir+'delaney.csv','inputDataset.csv')

#model parameters
batch_size = 2
neuronslayer1=64
neuronslayer2=128
dropout=0.1

saved_predictions = []

for i in range(1, 6):  
  model_dir = model_dir_basis+str(i)+"/"
  print('---start predictions---')
  model = KerasModel(MyGraphConvModel(batch_size=batch_size,neuronslayer1=neuronslayer1,neuronslayer2=neuronslayer2,dropout=dropout), loss=dc.models.losses.L1Loss(), model_dir=model_dir)
  print('---load model---')
  model.restore()
  make_predictions(model=model,dataset_ext= ext_dataset,batch_size=batch_size,read_out_string=str(i))
  saved_predictions.append(output_dir+'Prediction'+str(i)+'.csv')
  print('---END!---')
  
#writing result
datasets(output_dir+'inputDataset.csv',saved_predictions,result_dir)
