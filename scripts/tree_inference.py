import os
import torch
import argparse
import lightning as L
import matplotlib.pyplot as plt
from Bio import Phylo
from net.ProTok import ProTok_lightning
from src.common.utils import create_feature_list, read_fasta
from src.common.phylo import create_tree

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = ProTok_lightning.load_from_checkpoint(args.checkpoint, strict=False)
    model.to(device)
    model.eval()

    print(f"Reading FASTA file: {args.input}")
    names, seqs = read_fasta(args.input)
    input_list = create_feature_list(seqs, device=device)

    print("Extracting embeddings...")
    seq_embedding_list = []
    with torch.no_grad():
        for i, _ in enumerate(seqs):
            output, _ = model.encode(input_list[i])
            quantized = output.reshape(-1).cpu().numpy()
            seq_embedding_list.append(quantized)

    print("Generating phylogenetic tree...")
    bio_tree_nj = create_tree(seq_embedding_list, names)

    os.makedirs(os.path.dirname(args.output_nwk) or '.', exist_ok=True)
    Phylo.write(bio_tree_nj, args.output_nwk, "newick")
    print(f"Newick tree saved to: {args.output_nwk}")

    os.makedirs(os.path.dirname(args.output_png) or '.', exist_ok=True)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(
        bio_tree_nj,
        axes=ax,
        show_confidence=False,
        label_func=lambda c: c.name if c.name else "",
        do_show=False,
    )
    plt.tight_layout()
    plt.savefig(args.output_png)
    print(f"Tree image saved to: {args.output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProTok Phylogenetic Tree Generation Tool")

    # settings
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/ProTok_evo.ckpt",
                        help="Path to the model checkpoint (.ckpt)")
    parser.add_argument("--input", type=str, default="./example/TB2:S15435Taxa1_processed.fasta",
                        help="Path to the input FASTA file")
    parser.add_argument("--output_nwk", type=str, default="./example/tree.nwk",
                        help="Path to save the output Newick file")
    parser.add_argument("--output_png", type=str, default="./example/tree.png",
                        help="Path to save the output visualization image")

    args = parser.parse_args()
    run_inference(args)
