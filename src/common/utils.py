import string
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from skbio.stats.distance import DistanceMatrix as SKDM
from skbio.tree import nj as skbio_nj
from io import StringIO
from scipy.spatial.distance import pdist, squareform
from Bio import Phylo
import torch
from tqdm import tqdm
import torch
import math
import torch.nn.functional as F

DeletionMatrix = Sequence[Sequence[int]]

# 'm' for mask, 'b' for bos, 'u' for unk.
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V', "X", "m", "b", "eos", "u"
]

restype_order = {restype: i for i, restype in enumerate(restypes)}

order_restype = {i: restype for i, restype in enumerate(restypes)}

def _build_aa_lookup(restype_order: dict):
    lut = np.full(256, restype_order["X"], dtype=np.int64)
    for aa, idx in restype_order.items():
        if isinstance(aa, str) and len(aa) == 1:
            lut[ord(aa)] = int(idx)
    return lut

def read_fasta(file_path):
    names = []
    sequences = []
    
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
                names.append(line[1:])
            else:
                current_seq.append(line)

        if current_seq:
            sequences.append("".join(current_seq))
            
    return names, sequences



def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
  """Parses FASTA string and returns list of strings with amino-acid sequences.

  Arguments:
    fasta_string: The string contents of a FASTA file.

  Returns:
    A tuple of two lists:
    * A list of sequences.
    * A list of sequence descriptions taken from the comment lines. In the
      same order as the sequences.
  """
  sequences = []
  descriptions = []
  index = -1
  for line in fasta_string.splitlines():
    line = line.strip()
    if line.startswith('>'):
      index += 1
      descriptions.append(line[1:])  # Remove the '>' at the beginning.
      sequences.append('')
      continue
    elif not line:
      continue  # Skip blank lines.
    sequences[index] += line

  return sequences, descriptions


def parse_a3m(a3m_string: str) -> Tuple[Sequence[str], DeletionMatrix]:
  """Parses sequences and deletion matrix from a3m format alignment.

  Args:
    a3m_string: The string contents of a a3m file. The first sequence in the
      file should be the query sequence.

  Returns:
    A tuple of:
      * A list of sequences that have been aligned to the query. These
        might contain duplicates.
      * The deletion matrix for the alignment as a list of lists. The element
        at `deletion_matrix[i][j]` is the number of residues deleted from
        the aligned sequence i at residue position j.
  """
  sequences, _ = parse_fasta(a3m_string)
  deletion_matrix = []
  for msa_sequence in sequences:
    deletion_vec = []
    deletion_count = 0
    for j in msa_sequence:
      if j.islower():
        deletion_count += 1
      else:
        deletion_vec.append(deletion_count)
        deletion_count = 0
    deletion_matrix.append(deletion_vec)

  # Make the MSA matrix out of aligned (deletion-free) sequences.
  deletion_table = str.maketrans('', '', string.ascii_lowercase)
  aligned_sequences = [s.translate(deletion_table) for s in sequences]
  return aligned_sequences, deletion_matrix


def read_msa(msas):
    with open(msas, 'r') as f:
        msa_content = f.read()
    input_msa, input_dlm = parse_a3m(msa_content)
    return input_msa, input_dlm


def create_feature_list(seq_list, num_prefix=64, device="cpu"):
    """
    Args:
        seq_list: sequene list
        num_prefix: num of prefix tokens
        device: cuda
    """
    save_list = []
    
    substitution_map = str.maketrans("UB", "CD")

    for sequence in tqdm(seq_list, desc="Processing sequences"):
        
        sequence = sequence.translate(substitution_map)

        seq_len = len(sequence)
        
        raw_features = {
            "seq_len": seq_len,
            "aatype": np.array([restype_order.get(res, restype_order["X"]) for res in sequence]),
            "residue_index": np.arange(1, seq_len + 1)
        }
        
        data_dict = feature_generate(raw_features, num_prefix_tokens=num_prefix, max_len=1024)
        
        tensor_dict = {
            k: torch.as_tensor(v[None, :], dtype=torch.long, device=device) 
            for k, v in data_dict.items()
        }
        
        save_list.append(tensor_dict)
        
    return save_list


def feature_generate(feature, num_prefix_tokens=16, max_len=1024,mask=False):
    """
    Process protein features for model input/output.
    
    Args:
        feature: Dictionary containing protein features
        num_prefix_tokens: Number of prefix tokens to reserve
        max_len: Maximum sequence length
    
    Returns:
        Dictionary with processed data for both structure and sequence
    """
    # Constants
    ###4.18 other aa : 0-20 (include X)  mask:21 BOS:22 EOS:23  UNK:24
    
    ###4.16 other aa : 0-20 (include X)  gap/mask:21   BOS:22 EOS:23  UNK:24
    EOS_TOKEN = { "aatype": 23}
    BOS_TOKEN = { "aatype": 22}
    
    seq_len = feature["seq_len"]
    
    # Determine sequence handling approach based on length
    if seq_len + num_prefix_tokens + 2 > max_len:
         data_dict_AA = process_long_sequence(
            feature, seq_len, max_len, num_prefix_tokens, BOS_TOKEN, EOS_TOKEN,mask=mask)
    else:
        data_dict_AA = process_normal_sequence(
            feature, seq_len, max_len, num_prefix_tokens, BOS_TOKEN, EOS_TOKEN,mask=mask)
    
    return data_dict_AA

def process_long_sequence(feature, seq_len, max_len, num_prefix_tokens, BOS_TOKEN, EOS_TOKEN, mask=False):
    """Process sequences that exceed the max length and need to be cropped."""
    
    # Calculate effective sequence length after cropping
    seq_len_crop = max_len - num_prefix_tokens - 2
    
    # Extract cropped features
    encoder_code_AA = feature["aatype"][:seq_len_crop]
    if mask:
        encoder_code_AA_ref = feature["ref_aa"][:seq_len_crop]
    encoder_residue_index = feature["residue_index"][:seq_len_crop]
    
    # Add special tokens
    encoder_code_AA = add_special_tokens(encoder_code_AA, BOS_TOKEN["aatype"], EOS_TOKEN["aatype"])
    
    if mask:
        encoder_code_AA_ref = add_special_tokens(
            encoder_code_AA_ref, BOS_TOKEN["aatype"], EOS_TOKEN["aatype"])
    
    # Handle residue index
    encoder_residue_index = add_special_residue_indices(
        encoder_residue_index, is_padding=False)
    
    # Create encoder mask
    encoder_mask = np.full((max_len - num_prefix_tokens,), 1)
    
    # Set up decoder components
    decoder_code_AA = encoder_code_AA
    
    if mask:
        decoder_code_AA = encoder_code_AA_ref
        
    decoder_residue_index = encoder_residue_index
    
    # Create label mask
    label_mask = np.full((max_len,), 0)
    label_mask[num_prefix_tokens:num_prefix_tokens+seq_len_crop+1] = 1
    
    # Create labels
    
    label_AA = np.full((max_len,), 0)
    label_AA[num_prefix_tokens:num_prefix_tokens+seq_len_crop+1] = encoder_code_AA[1:seq_len_crop+2]

    if mask:
        label_AA[num_prefix_tokens:num_prefix_tokens+seq_len_crop+1] = encoder_code_AA_ref[1:seq_len_crop+2]
    
    # Create data dictionaries
    
    data_dict_AA = create_data_dict(encoder_code_AA, encoder_mask, encoder_residue_index, 
                                  decoder_code_AA, decoder_residue_index, label_mask, label_AA)
    
    return data_dict_AA

def process_normal_sequence(feature, seq_len, max_len, num_prefix_tokens, BOS_TOKEN, EOS_TOKEN, mask=False):
    """Process sequences that don't exceed max length and need padding."""
    
    # Calculate padding length
    padding_len = max_len - 1 - seq_len - num_prefix_tokens
    
    # Extract features
    encoder_code_AA = feature["aatype"]
    if mask:
        encoder_code_AA_ref = feature["ref_aa"]
    encoder_residue_index = feature["residue_index"]
    
    # Add special tokens and padding
    encoder_code_AA = add_special_tokens_with_padding(
        encoder_code_AA, BOS_TOKEN["aatype"], EOS_TOKEN["aatype"], padding_len)

    if mask:
        encoder_code_AA_ref = add_special_tokens_with_padding(
            encoder_code_AA_ref, BOS_TOKEN["aatype"], EOS_TOKEN["aatype"], padding_len)
    
    # Handle residue index with padding

    encoder_residue_index = add_special_residue_indices_with_padding(
        encoder_residue_index, padding_len)
    
    # Create encoder mask (1 for real tokens, 0 for padding)
    encoder_mask = np.concatenate((np.full((seq_len+2,), 1), np.full((padding_len-1,), 0)))
    
    # Set up decoder components
    decoder_code_AA = encoder_code_AA
    
    if mask:
        decoder_code_AA = encoder_code_AA_ref
        
    decoder_residue_index = encoder_residue_index
    
    # Create label mask
    label_mask = np.full((max_len,), 0)
    label_mask[num_prefix_tokens:num_prefix_tokens+seq_len+1] = 1
    
    # Create labels
    
    label_AA = np.full((max_len,), 0)
    label_AA[num_prefix_tokens:num_prefix_tokens+seq_len+1] = encoder_code_AA[1:seq_len+2]
    
    if mask:
        label_AA[num_prefix_tokens:num_prefix_tokens+seq_len+1] = encoder_code_AA_ref[1:seq_len+2]
    
    # Create data dictionaries
    
    data_dict_AA = create_data_dict(encoder_code_AA, encoder_mask, encoder_residue_index, 
                                  decoder_code_AA, decoder_residue_index, label_mask, label_AA)
    
    return data_dict_AA

def add_special_tokens(sequence, bos_token, eos_token):
    """Add BOS and EOS tokens to a sequence."""
    bos = np.full((1,), bos_token)
    eos = np.full((1,), eos_token)
    return np.concatenate((bos, sequence, eos))

def add_special_tokens_with_padding(sequence, bos_token, eos_token, padding_len):
    """Add BOS, EOS tokens and padding to a sequence."""
    bos = np.full((1,), bos_token)
    padding = np.full((padding_len,), eos_token)
    return np.concatenate((bos, sequence, padding))

def add_special_residue_indices(residue_index, is_padding=False):
    """Add special indices for BOS and EOS tokens."""
    bos_index = np.full((1,), residue_index[0]-1)
    eos_index = np.full((1,), residue_index[-1]+1)
    return np.concatenate((bos_index, residue_index, eos_index))

def add_special_residue_indices_with_padding(residue_index, padding_len):
    """Add special indices for BOS, EOS tokens and padding."""
    bos_index = np.full((1,), residue_index[0]-1)
    padding_indices = np.arange(residue_index[-1] + 1, residue_index[-1] + 1 + padding_len)
    return np.concatenate((bos_index, residue_index, padding_indices))

def create_data_dict(encoder_tokens, encoder_mask, encoder_residue_index, 
                   decoder_tokens, decoder_residue_index, label_mask, label):
    """Create a data dictionary with all required fields."""
    return {
        "encoder_protokens": encoder_tokens,
        "encoder_mask": encoder_mask,
        "encoder_residue_index": encoder_residue_index,
        "decoder_protokens": decoder_tokens,
        "decoder_residue_index": decoder_residue_index,
        "label_mask": label_mask,
        "label": label
    }

    
def build_feature(
    seq_list,
    restype_order: dict,
    num_prefix: int = 64,
    max_len: int = 1024,
    mask: bool = False,
    ref_seq_list=None,
    return_torch: bool = True,
    torch_device: str | torch.device = "cpu",
):
    EOS = 23
    BOS = 22

    n = len(seq_list)
    enc_len = max_len - num_prefix
    max_seq_payload = max_len - num_prefix - 2  

    encoder_protokens = np.full((n, enc_len), EOS, dtype=np.int64)
    decoder_protokens = np.full((n, enc_len), EOS, dtype=np.int64)

    encoder_mask = np.zeros((n, enc_len), dtype=np.int64)

    encoder_residue_index = np.tile(np.arange(enc_len, dtype=np.int64), (n, 1))
    decoder_residue_index = encoder_residue_index.copy()

    label_mask = np.zeros((n, max_len), dtype=np.int64)
    label = np.zeros((n, max_len), dtype=np.int64)

    trans = str.maketrans("UB", "CD")

    # aa lookup
    aa_lut = _build_aa_lookup(restype_order)

    if mask:
        if ref_seq_list is None:
            raise ValueError("mask=True need ref_seq_list or ref_aa ")
        if len(ref_seq_list) != n:
            raise ValueError("ref_seq_list must equal seq_list")

    for i, seq in enumerate(seq_list):
        seq = seq.translate(trans)
        L_raw = len(seq)
        L_use = min(L_raw, max_seq_payload)

        b = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)[:L_use]
        aatype = aa_lut[b]  # (L_use,)

        encoder_protokens[i, 0] = BOS
        encoder_protokens[i, 1:1 + L_use] = aatype
        encoder_protokens[i, 1 + L_use] = EOS  

        if not mask:
            decoder_protokens[i] = encoder_protokens[i]
        else:
            ref_seq = ref_seq_list[i].translate(trans)
            rb = np.frombuffer(ref_seq.encode("ascii", "ignore"), dtype=np.uint8)[:L_use]
            ref_aatype = aa_lut[rb]
            decoder_protokens[i, 0] = BOS
            decoder_protokens[i, 1:1 + L_use] = ref_aatype
            decoder_protokens[i, 1 + L_use] = EOS

        is_long = (L_raw + num_prefix + 2 > max_len)
        if is_long:
            encoder_mask[i, :] = 1
        else:
            encoder_mask[i, :L_use + 2] = 1

        lm_start = num_prefix
        lm_end = num_prefix + L_use + 1
        label_mask[i, lm_start:lm_end] = 1

        label[i, lm_start:lm_end] = encoder_protokens[i, 1:1 + (L_use + 1)]

    out = {
        "encoder_protokens": encoder_protokens,
        "encoder_mask": encoder_mask,
        "encoder_residue_index": encoder_residue_index,
        "decoder_protokens": decoder_protokens,
        "decoder_residue_index": decoder_residue_index,
        "label_mask": label_mask,
        "label": label,
    }

    if not return_torch:
        return out

    out_t = {k: torch.from_numpy(v) for k, v in out.items()}

    if torch_device != "cpu":
        out_t = {k: t.to(torch_device, non_blocking=True) for k, t in out_t.items()}

    return out_t

def decode_to_seq(decode_result: torch.Tensor) -> str:

    result = decode_result.detach().cpu().numpy().flatten()
    seq = "".join(order_restype.get(int(i), "X") for i in result[1:])  
    return seq.replace("eos", "")

def default_embed_init(tensor):
    fan_in = tensor.size(-2)  # Assume `out_axis=0` corresponds to the first axis in PyTorch
    std = math.sqrt(1.0 / fan_in)  # Scale for `fan_in` and `normal`
    with torch.no_grad():
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    
def l2_normalize(x, mask = None):
    _dtype = x.dtype
    x = x.to(torch.float32)
    # x = x + (1.0 - mask).unsqueeze(-1).to(torch.float32) * 1e-6  ##### prevent nan bug
    x = x / torch.maximum(torch.linalg.norm(x, dim=-1, keepdim=True), torch.tensor(1e-6))
    x = x.to(_dtype)
    return x


class DecoderUtils:
    @staticmethod
    def gnmt_lp(length: int, alpha: float = 0.9):
        return ((5.0 + length) / 6.0) ** alpha

    @staticmethod
    def apply_eos_bias(logits, step, min_len, warmup, bias, eos_id):
        if step < min_len:
            logits[:, eos_id] = -float('inf')
        elif warmup > 0 and step < (min_len + warmup):
            frac = (min_len + warmup - step) / float(warmup)
            logits[:, eos_id] -= bias * max(0.0, min(1.0, frac))

    @staticmethod
    def top_p_k_filter(logits, top_p=1.0, top_k=0):
        v = logits.clone()
        if top_k > 0:
            indices_to_remove = v < torch.topk(v, top_k)[0][..., -1, None]
            v[indices_to_remove] = -float('inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(v, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_idx_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            idx_remove = sorted_idx_remove.scatter(1, sorted_indices, sorted_idx_remove)
            v[idx_remove] = -float('inf')
        return v
    


def _format_beam_results(B, num_beams, nrs, hyps, final_inputs, final_scores, eos_id):
    results = []
    for b in range(B):
        for k in range(num_beams):
            seq = final_inputs[b*num_beams + k]
            if seq[-1] != eos_id: seq = torch.cat([seq, torch.tensor([eos_id], device=seq.device)])
            hyps[b].append({'seq': seq, 'logp': float(final_scores[b, k])})
        
        for h in hyps[b]:
            h['final_score'] = h['logp'] / DecoderUtils.gnmt_lp(len(h['seq'])-1)
        
        sorted_hyps = sorted(hyps[b], key=lambda x: x['final_score'], reverse=True)
        take = min(nrs, len(sorted_hyps))
        results.append({
            'best_sequence': sorted_hyps[0]['seq'].unsqueeze(0),
            'all_candidates': [{'sequence': sorted_hyps[i]['seq'].unsqueeze(0), 
                                'score': sorted_hyps[i]['final_score'], 
                                'logp': sorted_hyps[i]['logp']} for i in range(take)]
        })
    return results

def _format_sample_results(B, nrs, inputs, sum_logp, eos_id):
    results = []
    for b in range(B):
        candidates = []
        for r in range(b*nrs, (b+1)*nrs):
            seq = inputs[r]
            eos_idx = (seq == eos_id).nonzero()
            if len(eos_idx) > 0: seq = seq[:eos_idx[0].item() + 1]
            elif seq[-1] != eos_id: seq = torch.cat([seq, torch.tensor([eos_id], device=seq.device)])
            
            score = float(sum_logp[r] / DecoderUtils.gnmt_lp(len(seq)-1, 1.2))
            candidates.append({'sequence': seq.unsqueeze(0), 'score': score})
            
        candidates.sort(key=lambda x: x['score'], reverse=True)
        results.append({'best_sequence': candidates[0]['sequence'], 'all_candidates': candidates})
    return results