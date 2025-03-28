import os
import sys
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Sequence, Tuple, List
import biotite
import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
from esm.data import BatchConverter, Alphabet

sys.path.append('../')
sys.path.append('../../')


class LoadData:

    def __init__(self,
                 file_path,
                 foldseek,
                 sampling_num,
                 run_mode,
                 label_mapping,
                 use_sslm,
                 use_gvp,
                 use_plm,
                 plm_type,
                 AA_max_length):

        assert isinstance(file_path,
                          str), 'file_path should be a string that points to the folder containing pdb files.'
        
        self.pdb_path = file_path + run_mode
        self.label_mapping = label_mapping
        self.pdb_files = self.sample_pdb_files(self.get_file_name_list(self.pdb_path))
        
        if use_gvp:
            assert AA_max_length != None, "The maximum length of `coords_max_shape` should be provided."
            self.coords_max_shape = [AA_max_length, 3, 3]
            self.confidence_max_shape = [AA_max_length]

        self.use_sslm = use_sslm
        self.use_gvp = use_gvp
        self.use_plm = use_plm
        self.plm_type = plm_type
        self.file_path = file_path
        self.foldseek = foldseek

    def load(self):
        data = []
        if self.use_sslm and self.use_gvp:
            coords, coord_mask, padding_mask, confidence, struct_seqs = self.get_structure_from_pdb(self.pdb_path,
                                                                                                    self.pdb_files)
            for i in range(coords.size()[0]):
                label = self.get_label(self.pdb_files[i])
                # print(f'{self.pdb_files[i]}--label: {label}')
                data.append((struct_seqs[i], coords[i], coord_mask[i], padding_mask[i], confidence[i], label))

        elif self.use_sslm:
            with tqdm(total=len(self.pdb_files)) as pbar:
                pbar.set_description("Loading 'structure-sequence' using foldseek from {}".format(self.pdb_path))
                for file in self.pdb_files:
                    struct_seq = self.get_struct_seq(self.foldseek, os.path.join(self.pdb_path, file))["A"]
                    data.append((struct_seq, self.get_label(file)))
                    pbar.update(1)
            print('========= Loaded ==========')

        elif self.use_plm:
            if self.plm_type == 'esm' or self.plm_type == 'prot_bert':
                coords, coord_mask, padding_mask, confidence, aa_seqs = self.get_structure_from_pdb(self.pdb_path,
                                                                                                self.pdb_files)
                for i in range(coords.size()[0]):
                    label = self.get_label(self.pdb_files[i])
                    data.append((aa_seqs[i], coords[i], coord_mask[i], padding_mask[i], confidence[i], label))
            elif self.plm_type == 'saprot':
                with tqdm(total=len(self.pdb_files)) as pbar:
                    pbar.set_description("Loading 'Structure Aware' sequences using from {}".format(self.pdb_path))
                    for file in self.pdb_files:
                        combined_seq = self.get_struct_seq(
                            self.foldseek, os.path.join(self.pdb_path, file), combine=True)["A"]
                        data.append((combined_seq, self.get_label(file)))
                        pbar.update(1)
                print('========= Loaded ==========')
            else:
                raise ValueError("plm_type should be either 'esm' or 'prot_bert' or 'saprot'")
            

        elif self.use_gvp:
            self.pdb_files = self.sample_pdb_files(self.get_file_name_list(self.pdb_path))
            coords, coord_mask, padding_mask, confidence = self.get_structure_from_pdb(self.pdb_path, self.pdb_files)
            for i in range(coords.size()[0]):
                data.append(
                    (coords[i], coord_mask[i], padding_mask[i], confidence[i], self.get_label(self.pdb_files[i])))
        return data

    def get_file_name_list(self, path):
        file_name_list = []
        for file in os.listdir(path):
            file_name_list.append(file)
        return file_name_list

    def sample_pdb_files(self, pdb_files):
        grouped_files = {label: [] for label, _ in self.label_mapping.items()}
        for pdb_file in pdb_files:
            for label, _ in self.label_mapping.items():
                if f"-{label}" in pdb_file:
                    if len(grouped_files[label]) == self.each_class_num: continue
                    grouped_files[label].append(pdb_file)
                    break
        print("grouped_files: ", grouped_files)
        resampled_files = []
        for _, files in grouped_files.items():
            resampled_files.extend(files)
        return resampled_files

    def get_struct_seq(self, foldseek, path, chains: list = ["A"], process_id: int = 0, combine: bool = True):
        """
        Args:
            foldseek: Binary executable file of foldseek
            path: Path to pdb file
            chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
            process_id: Process ID for temporary files. This is used for parallel processing.

        Returns: sequence dictionary: {chain: sequence}
        """
        assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
        assert os.path.exists(path), f"Pdb file not found: {path}"

        tmp_save_path = f"get_struc_seq_{process_id}.tsv"
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
        os.system(cmd)

        seq_dict = {}
        name = os.path.basename(path)
        with open(tmp_save_path, "r") as r:
            for line in r:
                desc, seq, struct_seq = line.split("\t")[:3]
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        if combine:
                            combined_seq = "".join([a + b.lower() for a, b in zip(seq, struct_seq)])
                        seq_dict[chain] = combined_seq if combine else struct_seq

        os.remove(tmp_save_path)
        os.remove(tmp_save_path + ".dbtype")
        return seq_dict

    def get_label(self, file_name):
        for key, value in self.label_mapping.items():
            if key in file_name: return value
        return ValueError("No corresponding label found for {}'s data.".format(file_name))

    def get_structure_from_pdb(self, path, file_name):
        '''
        Load protein structure from pdb file
        :param file_name:
        :return: structure
        '''
        if self.use_gvp and not self.use_sslm: self.sampling_num = len(file_name) if len(
            file_name) < self.sampling_num else self.sampling_num
        confident = None
        raw_batch = []
        with tqdm(total=self.sampling_num) as pbar:
            if self.use_sslm:
                struct_seqs = []
                pbar.set_description("Loading structures and 'structure-sequence' from {}".format(path))
            if self.use_plm:
                aa_seqs = []
                pbar.set_description("Loading structures and AA sequence from {}".format(path))
            else:
                pbar.set_description("Loading structures from {}".format(path))
            for index in range(self.sampling_num):
                file_path = os.path.join(path, file_name[index])
                if self.use_sslm:
                    struct_seqs.append(self.get_struct_seq(self.foldseek, file_path)["A"])
                coords, seq = self.load_coords(file_path, ["A"])
                if self.use_plm: aa_seqs.append(seq)
                raw_batch.append((coords, confident, seq))
                pbar.update(1)
        print('========= Loaded ==========')
        alphabet = Alphabet.from_architecture("invariant_gvp")
        converter = CoordBatchConverter(alphabet)
        coords, coord_mask, padding_mask, confidence = converter(
            raw_batch=raw_batch,
            coords_max_shape=self.coords_max_shape,
            confidence_max_shape=self.confidence_max_shape)
        if self.use_gvp and self.use_sslm:
            return coords, coord_mask, padding_mask, confidence, struct_seqs
        if self.use_gvp and self.use_plm:
            return coords, coord_mask, padding_mask, confidence, aa_seqs
        return coords, coord_mask, padding_mask, confidence

    def load_structure(self, fpath, chain=None):
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

    def get_atom_coords_residuewise(self, atoms: List[str], struct: biotite.structure.AtomArray):
        """
        Args:
            atoms: List of atom names to extract coordinates for, e.g., ["N", "CA", "C"].
            struct: A biotite AtomArray representing the molecular structure.
        """

        def filterfn(s, axis=None):
            """
            Args:
                s: biotite AtomArray
                axis: None
            Returns: coords
            """
            filters = np.stack([s.atom_name == name for name in atoms], axis=1)
            sum_check = filters.sum(0)
            if not np.all(sum_check <= np.ones(filters.shape[1])):
                raise RuntimeError("The structure contains multiple atoms with the same name.")
            first_occurrence_index = filters.argmax(0)
            coords = s[first_occurrence_index].coord
            coords[sum_check == 0] = float("nan")

            return coords

        return biotite.structure.apply_residue_wise(struct, struct, filterfn)

    def extract_coords_from_structure(self, structure: biotite.structure.AtomArray):
        """
        Args:
            structure: An instance of biotite AtomArray
        Returns:
            Tuple (coords, seq)
                - coords is an L x 3 x 3 array for N, CA, C coordinates
                - seq is the extracted sequence
        """
        coords = self.get_atom_coords_residuewise(["N", "CA", "C"], structure)
        residue_identities = get_residues(structure)[1]
        seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
        return coords, seq

    def load_coords(self, fpath, chain):
        """
        Args:
            fpath: filepath to either pdb or cif file
            chain: the chain id
        Returns:
            Tuple (coords, seq)
                - coords is an L x 3 x 3 array for N, CA, C coordinates
                - seq is the extracted sequence
        """
        structure = self.load_structure(fpath, chain)
        return self.extract_coords_from_structure(structure)


class CoordBatchConverter(BatchConverter):

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], coords_max_shape, confidence_max_shape,
                 device=None, ):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))
        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan, max_shape=coords_max_shape)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1., max_shape=confidence_max_shape)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, coord_mask, padding_mask, confidence

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v, max_shape=None):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        # max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            slices = tuple(slice(0, min(dim, max_dim)) for dim, max_dim in zip(t.shape, max_shape))
            result_i[slices] = t[slices]

        return result


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# test
if __name__ == '__main__':
    seed_everything(42)
    file_path = '/home/lmc/gwr/CPE-Pro-dataset/C-A_pdb/'
    foldseek = '/home/lmc/gwr/foldseek/bin/foldseek'
    sampling_num = {'train': 4, 'valid': 10, 'test': 10}
    label_mapping = {"crystal": 0, "alphafold": 1}
    use_sslm = False
    use_gvp = True
    use_plm = False
    max_length = 100
    run_mode = 'train'

    data_loader = LoadData(file_path, foldseek, sampling_num, 'train', label_mapping, use_sslm, use_gvp, use_plm, 'esm', 256)
    data = data_loader.load()
    print(data[0])