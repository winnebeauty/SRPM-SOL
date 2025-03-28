import os
import json
import torch
import random
import numpy as np
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader
)
import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import torch.nn.functional as F
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from data.ConvertSeq import sequence_to_index
from torch.utils.data import random_split

SEQ_MAX_LENGTH=128


def calculate_gravy(sequence):
    """
    使用 Koyté 方法计算GRAVY（Grand Average of Hydropathy）指数
    """
    analysis = ProteinAnalysis(sequence)
    gravy = analysis.gravy()  # 计算 GRAVY 值
    return gravy


class LoadData(nn.Module):
    def __init__(self, file_path,pdb_folder,type,pdb_index,sampling_num):
        super(LoadData, self).__init__()
        self.file_path = file_path
        self.sampling_num = sampling_num
        self.type=type
        self.pdb_folder=pdb_folder
        self.pdb_index=pdb_index
        
    def load_file_list(self):
        file_list = []
        for root, dirs, files in os.walk(self.file_path):
            for file in files:
                if file.endswith('.json'):
                    file_list.append(os.path.join(root, file))
        return file_list
        
    def load_data(self):
        file_list = self.load_file_list()
        
        train_data = []
        valid_data = []
        test_data = []
        all_data=[]
        for file in file_list:
            dataset = None
            if 'train' in file.lower():
                fold_name='train'
                dataset = train_data
                num=self.sampling_num['train']
            elif 'valid' in file.lower():
                fold_name='valid'
                dataset = valid_data
                num=self.sampling_num['valid']
            elif 'test' in file.lower():
                fold_name='test'
                dataset = test_data
                num=self.sampling_num['test']
            
            if dataset is not None:
                try:
                    with open(file, 'r') as f:
                        json_data = json.load(f)  
                        
                        random.shuffle(json_data)     
                        
                        for record in json_data:
                            if len(dataset) >= num:
                                break
                            name=record.get('name', '')  
                            sequence = record.get('sequence', '')  
                            solubility = record.get('solubility', None)  
                            if sequence and solubility is not None:
                                if self.type != 'esm3':
                                    gravy=calculate_gravy(sequence)
                                    dataset.append((sequence,gravy, solubility))
                                else:
                                    pdb_file_path = os.path.join(self.pdb_folder, fold_name)
                                    pdb_file = os.path.join(pdb_file_path, f"{name}.pdb")
                                    with open(self.pdb_index, "r") as f:
                                        pdb_file_map=json.load(f)
                                    pdb_file_map = {int(k): v for k, v in pdb_file_map.items()}
                                    for id,path in pdb_file_map.items():
                                        if path==pdb_file:
                                            pdb_file_id=id
                                            break
                                    if os.path.exists(pdb_file):
                                        try:
                                            gravy=calculate_gravy(sequence)
                                            structure = load_structure(pdb_file)  
                                            coords, seq_from_pdb = extract_coords_from_structure(structure)  
                                            if len(sequence)==coords.shape[0]:  
                                                dataset.append((sequence, torch.Tensor(coords),gravy,pdb_file_id, solubility))
                                        except Exception as e:
                                            print(f"Error processing {pdb_file}: {e}")
                                    else:
                                        print(f"PDB file {pdb_file} not found. Skipping.")
                                    
                except Exception as e:
                    print(f"{e}")
        all_data.extend(train_data)
        all_data.extend(valid_data)
        all_data.extend(test_data)
            

        return all_data




def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq

def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def esmprotein_collate_fn(batch):
        proteins = []
        coordinates = []
        gravys=[]
        pdb_files_ids=[]
        labels = []
        original_lengths=[]
        max_length = SEQ_MAX_LENGTH


        for seq,coords,gravy,pdb_file, label in batch:
            seq_id=sequence_to_index(seq)
            pad_protein=F.pad(torch.tensor(seq_id),(0,max_length-torch.tensor(seq_id).shape[0]),value=1)  
            proteins.append(pad_protein.unsqueeze(0))


            original_length = coords.shape[0]
            original_lengths.append(original_length)

            pad_coords=F.pad(coords,(0,0,0,0,0,max_length-coords.shape[0]),value=0) 
            coordinates.append(pad_coords.unsqueeze(0))
            gravys.append(gravy)
            pdb_files_ids.append(pdb_file)    
            labels.append(label)
            
        return torch.cat(proteins,dim=0),torch.cat(coordinates,dim=0),torch.tensor(original_lengths),torch.tensor(gravys),torch.tensor(pdb_files_ids), torch.tensor(labels)
        



class DataModule(nn.Module):
    def __init__(self, file_path,pdb_folder,type, sampling_num,  batch_size, num_workers,seq_max_length,pdb_index, train_ratio=0.8, valid_ratio=0.1):
        super(DataModule, self).__init__()
        self.file_path = file_path
        self.sampling_num = sampling_num
        self.type=type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pdb_folder=pdb_folder
        self.pdb_index=pdb_index
        self.seq_max_length=seq_max_length
        all_data = LoadData(file_path=file_path,pdb_folder=self.pdb_folder,type=self.type, pdb_index=pdb_index, sampling_num=sampling_num).load_data()

        total_size = len(all_data)
        print(f"Total data size: {total_size}")
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)
        test_size = total_size - train_size - valid_size  
        self.train_data, self.valid_data, self.test_data = random_split(all_data, [train_size, valid_size, test_size])
        
                
        self.type=type
        if self.type=='esm3':
            self.collate_fn=esmprotein_collate_fn
        else:
            self.collate_fn=None
    
    
    def forward(self):
        if len(self.train_data) > 0:
            self.train_dataset = DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        else:
            self.train_dataset = None

        if len(self.valid_data) > 0:
            self.valid_dataset = DataLoader(
                self.valid_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        else:
            self.valid_dataset = None

        if len(self.test_data) > 0:
            self.test_dataset = DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        else:
            self.test_dataset = None

        return self.train_dataset, self.valid_dataset, self.test_dataset
        
