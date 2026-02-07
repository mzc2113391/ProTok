import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from Bio import Phylo
import string
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from skbio.stats.distance import DistanceMatrix as SKDM
from skbio.tree import nj as skbio_nj
from io import StringIO
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from ete3 import Tree

def sperman_rank_corr(a,b): # spearman rank correlation between 2 distance matrices - in numpy array.
    # check the input type and shape
    assert type(a)==np.ndarray and type(b)==np.ndarray, "Input must be numpy array " #pd.DataFrame
    assert a.shape == b.shape, "Arrays are of different shape"

    # get the upper half of the matrix only
    mask =  np.triu_indices(a.shape[0], k=1)
    
    return stats.spearmanr(a[mask],b[mask])

def pearson_corr(a,b): # spearman rank correlation between 2 distance matrices - in numpy array.
    # check the input type and shape
    assert type(a)==np.ndarray and type(b)==np.ndarray, "Input must be numpy array " #pd.DataFrame
    assert a.shape == b.shape, "Arrays are of different shape"

    # get the upper half of the matrix only
    mask =  np.triu_indices(a.shape[0], k=1)
    
    return pearsonr(a[mask],b[mask])

def cal_correlation(phylo_embedding_dict, ref_embedding):

    cosim_cutoff_1 = 0.90

    cosim_cutoff_2 = 0.61

    phylum_list = list(sorted(set(phylo_embedding_dict.keys()).intersection(ref_embedding.keys())))

    seq_embedding_list = [phylo_embedding_dict[i] for i in phylum_list]

    seq_embedding_ref = [ref_embedding[i] for i in phylum_list]

    embeddings = np.array(seq_embedding_list)
    diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]  # shape: (N, N, 768)
    distance_matrix = np.linalg.norm(diff, axis=2)

    embeddings_ref = np.array(seq_embedding_ref)
    diff_ref = embeddings_ref[:, np.newaxis, :] - embeddings_ref[np.newaxis, :, :]  # shape: (N, N, 768)
    distance_matrix_ref = np.linalg.norm(diff_ref, axis=2)

    r, rho = pearson_corr(distance_matrix,distance_matrix_ref), sperman_rank_corr(distance_matrix,distance_matrix_ref)

    return seq_embedding_list, phylum_list, seq_embedding_ref, r, rho

def parse_fasta_to_dict(file_name):
    temp_data = {}
    all_protein_names = []

    with open(file_name, 'r') as f:
        current_species = None
        current_protein = None
        current_seq = []

        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith(">"):
                if current_species and current_protein:
                    temp_data[current_species][current_protein] = "".join(current_seq)
                
                header_content = line[1:]
                if "@" in header_content:
                    species, protein = header_content.split("@", 1)
                    
                    if species not in temp_data:
                        temp_data[species] = {}
                    if protein not in all_protein_names:
                        all_protein_names.append(protein)
                    
                    current_species = species
                    current_protein = protein
                    current_seq = []
                else:
                    continue
            else:
                current_seq.append(line)
        
        if current_species and current_protein:
            temp_data[current_species][current_protein] = "".join(current_seq)

    standard_order = all_protein_names 

    final_dict = {}
    for species, proteins in temp_data.items():
        final_dict[species] = [proteins.get(p, "") for p in standard_order]
            
    return final_dict

def run_incongruence_pipeline(org_df, ref_embedding, gene_embedding_dict, top_n_display=50):

    global_conserve_score = 0.7

    cross_class = 0.61

    sim_threshold = 0.9

    name_to_phylum = pd.Series(org_df["phylum"].values, index=org_df["genome_name"]).to_dict()

    common_species = sorted(list(set(gene_embedding_dict.keys()) & set(ref_embedding.keys())))
    species_list = np.array(common_species)
    n_species = len(species_list)
    phyla = np.array([name_to_phylum.get(s, "Unknown") for s in species_list])
    
    gene_vecs = np.array([gene_embedding_dict[s] for s in species_list])
    ref_vecs = np.array([ref_embedding[s] for s in species_list])

    sim_gene = 1 - cdist(gene_vecs, gene_vecs, metric='cosine')
    sim_ref = 1 - cdist(ref_vecs, ref_vecs, metric='cosine')

    phyla_i = phyla[:, np.newaxis]
    phyla_j = phyla[np.newaxis, :]
    inter_phylum_mask = (phyla_i != phyla_j) & (phyla_i != "Unknown") & (phyla_j != "Unknown")
    
    global_inter_sims = sim_gene[inter_phylum_mask]
    
    gene_global_median_sim = np.median(global_inter_sims)

    is_globally_conserved = gene_global_median_sim > global_conserve_score


    criteria_mask = (sim_gene > sim_threshold) & (sim_ref < cross_class) & inter_phylum_mask 
    rows, cols = np.where(criteria_mask)
    
    results = []
    if len(rows) > 0:
        for i, j in zip(rows, cols):
            target_inter_mask = (phyla != phyla[i]) & (phyla != "Unknown")
            target_background_sim = np.median(sim_gene[i][target_inter_mask])
            
            i_gene_ranks = np.argsort(np.argsort(-sim_gene[i]))
            i_ref_ranks = np.argsort(np.argsort(-sim_ref[i]))
            
            gene_rank = i_gene_ranks[j] + 1
            ref_rank = i_ref_ranks[j] + 1
            rel_shift = (ref_rank - gene_rank) / n_species
            

            outlier_diff = sim_gene[i, j] - target_background_sim
            
            results.append({
                "Species_A": species_list[i], "Species_B": species_list[j],
                "Phylum_A": phyla[i], "Phylum_B": phyla[j],
                "Gene_Sim": sim_gene[i, j], "Ref_Sim": sim_ref[i, j],
                "Rel_Shift": rel_shift,
                "Target_Background_Sim": target_background_sim,
                "Outlier_Diff": outlier_diff,
                "HGT_Score": sim_gene[i, j] - sim_ref[i, j]
            })

    df = pd.DataFrame(results)

    def judge_confidence(row):
        if is_globally_conserved:
            return "None"
        
        if row['Rel_Shift'] > 0.7:
            return "High"
        
        if row['Rel_Shift'] > 0.4:
            return "Potential"
        
        return "None"

    if not df.empty:
        df['Confidence'] = df.apply(judge_confidence, axis=1)
        df['is_HGT'] = df['Confidence'] != "None"
        df['is_High'] = df['Confidence'] == "High"
        
        is_HGT_detected = (df['is_HGT']).any()
        is_High_detected = (df['is_High']).any()
        
        df = df.sort_values("HGT_Score", ascending=False).reset_index(drop=True)
    else:
        is_HGT_detected = False
        is_High_detected = False

    if is_HGT_detected:
        print("\n" + "!"*30 + " EVOLUTIONARY ANOMALY DETECTED " + "!"*30)
        print(f"Total abnormal pairs found: {len(df[df['is_HGT']])}")
        print(f"{'RANK':<4} | {'SPECIES A':<20} | {'SPECIES B':<20} | {'G-SIM':<6} | {'R-SIM':<6} | {'SHIFT'}")
        print("-" * 100)
        
        count = 0
        for _, row in df[df['is_HGT']].iterrows():
            count += 1
            print(f"{count:<4} | {row['Species_A']:<20} | {row['Species_B']:<20} | {row['Gene_Sim']:<6.3f} | {row['Ref_Sim']:<6.3f} | {row['Rel_Shift']:<6.2%}")
            print(f"     └─ Taxonomy: {row['Phylum_A']} <-> {row['Phylum_B']}")
            if count >= top_n_display: break
    else:
        print(f"\n>>> Gene {n_species} species: No significant HGT signal detected.")

    return df, is_HGT_detected, is_High_detected


def rf_distance(tree1_str, tree2_str):
    """
    Calculates Robinson-Foulds distance between two trees
    Input: (str) Newick string of tree 1
           (str) Newick string of tree 2
    Output: (int) output Robinson-Foulds distance
    """
    try:
    
        # Remove branch distances from the Newick strings of the predicted and reference tree
        def remove_branch_distances(tree_str):

            # Set branch lengths in tree to zero
            phylo_tree = Phylo.read(StringIO(tree_str), "newick")
            for i in phylo_tree.get_nonterminals():
                i.branch_length=None
            for i in phylo_tree.get_terminals():
                i.branch_length=None

            # Convert edited tree to Newick string
            new_str_obj = StringIO()
            Phylo.write(phylo_tree, new_str_obj, "newick")
            new_str_obj.seek(0)
            new_str = new_str_obj.getvalue()

            # Remove distances from edited tree string
            dist_decimals = 8   # To remove the distance value of ":0.00000"
            while True:
                try:
                    curr_index = new_str.index(":")
                    new_str = new_str[:curr_index] + new_str[curr_index+dist_decimals:]
                except:
                    return new_str

        tree1_str_nodist = remove_branch_distances(tree1_str)
        tree2_str_nodist = remove_branch_distances(tree2_str)

        # Calculate tree comparison metrics
        t1 = Tree(tree1_str_nodist)
        t2 = Tree(tree2_str_nodist)
        result = t1.compare(t2, unrooted=True)
        rf = int(result["rf"])
        max_rf = int(result["max_rf"])
        norm_rf = result["norm_rf"]

    except:
        print("Tree formats are invalid, skipping this sample")
        return {"rf": None,
            "max_rf": None,
            "norm_rf": None}

    return {"rf": rf,
            "max_rf": max_rf,
            "norm_rf": norm_rf}

def create_tree(seq_embedding_list, phylum_list, return_str = False):

    X = np.stack(seq_embedding_list, axis=0)
    
    distance_matrix = squareform(pdist(X, metric="cosine")).astype(float)
    
    np.fill_diagonal(distance_matrix, 0.0)
    
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-8):
        raise ValueError("distance_matrix must be symmetric")

    assert len(phylum_list) == distance_matrix.shape[0] == distance_matrix.shape[1]

    sk_dm = SKDM(distance_matrix, ids=list(phylum_list))

    sk_tree = skbio_nj(sk_dm)

    if return_str:
## return unrooted trees
        tree_str = sk_tree.__str__()

        return tree_str

    buf = StringIO()

    sk_tree.write(buf) 

    newick_str = buf.getvalue()

    bio_tree_nj = Phylo.read(StringIO(newick_str), "newick")

    bio_tree_nj.ladderize()
    
    return bio_tree_nj