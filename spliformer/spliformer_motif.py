import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from spliformer.Network_motif import *
from pyfaidx import Fasta
import logging
import pysam
import seaborn as sns
import os
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
#This script was adpated from SpliceAI's prediction script
class Annotator:
    def __init__(self, ref_fasta, annotations):

        df = pd.read_csv(annotations, sep='\t', dtype={'CHR': object})
        self.genes = df['GENE'].to_numpy()
        self.chroms = df['CHR'].to_numpy()
        self.strands = df['STRAND'].to_numpy()
        self.tx_starts = df['GENE_START'].to_numpy() + 1
        self.tx_ends = df['GENE_END'].to_numpy()
        self.ref_fasta = Fasta(ref_fasta, rebuild=False)

    def get_name_and_strand(self, chrom, pos):

        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                              np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                                             np.nonzero(pos <= self.tx_ends)[0]))

        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs
        else:
            return [], [], []

    def get_pos_data(self, idx, pos):

        dist_tx_start = self.tx_starts[idx] - pos
        dist_tx_end = self.tx_ends[idx] - pos

        dist_ann = (dist_tx_start, dist_tx_end)

        return dist_ann


def one_hot(seqs, strand):
    # {'N':0, A': 1, 'C': 2, 'G': 3, 'T': 4}
    def preprocess_inputs(seq):
        fun = lambda seq: [token2int[x] for x in seq]
        return np.transpose(np.array([[fun(seq)]]), (0, 2, 1))

    if strand == '+':
        token2int = {x: i for i, x in enumerate('NACGT')}
        seqs = preprocess_inputs(seqs)
    elif strand == '-':
        token2int = {x: i for i, x in enumerate('NTGCA')}
        seqs = preprocess_inputs(seqs[::-1])  # 取反后互换
        # seqs = preprocess_inputs(seqs)

    return seqs

def data_normalise(orign_data):
    
    d_min = orign_data.min().min()
    if d_min < 0:
        orign_data +=torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max().max()
    dst = d_max - d_min
    norm_data = (orign_data-d_min)/(dst)
    return norm_data

def normalise_chrom(source, target): #source=user's; target=genome table
    def has_prefix(x):
        return x.startswith('chr')

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')
    elif not has_prefix(source) and has_prefix(target):
        return 'chr' + source

    return source

def draw_picture(label,result,gene,seqtype,distance):
    foldername=gene+'_motif_aws'
    if not os.path.exists('./motif_result/'+foldername):
            os.makedirs('./motif_result/'+foldername)    
    result=result.T
    name=label
    namelist=[]
    for n in range(len(name)-12):
        namelist.append(name[6+n-6:6+n+7])
    plt.figure(figsize=(95,((distance*2)+5)),tight_layout=True)
#     plt.figure(figsize=(95,85))
    sns.set(font_scale=4.25)
    sns.heatmap(result[40-distance:40+distance+1],cbar=True,
                vmin=0, vmax=1,
                     cmap="Reds",
                     annot=True,
                     fmt='.2f',
                     xticklabels=namelist,
                     yticklabels=namelist[40-distance:40+distance+1],
                     annot_kws={'size': 18},
                     linewidths=0.3)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.savefig('./motif_result/{}/{}_motif_{}.png'.format(foldername,gene,seqtype), bbox_inches='tight')

def predict(record, ann, model, distance):


    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)

    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    seq = ann.ref_fasta[chrom][(record.pos - 46 - 1):(record.pos + 46 + 40)].seq  #the first nucleictide won't extract

    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue

            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            if len(record.ref) > 1 and len(record.alts[j]) > 1:
                continue
                
            if len(record.ref) > 40 or len(record.alts[j]) > 40:
                continue
                
            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len - alt_len, 0)

            x_ref = seq.upper()
            x_alt = x_ref[:93// 2] + str(record.alts[j]) + x_ref[93// 2 + ref_len:]
            x_ref=x_ref[0:93]
            x_alt=x_alt[0:93]
            label_ref=x_ref.replace('T','U')
            label_alt=x_alt.replace('T','U')
            if strands[i] == '-':  
                label_ref=label_ref[::-1].replace('A','a').replace('U','t').replace('C','c').replace('G','g').replace('n','N')
                label_ref=label_ref.replace('a','U').replace('t','A').replace('c','G').replace('g','C')
                label_alt=label_alt[::-1].replace('A','a').replace('U','t').replace('C','c').replace('G','g').replace('n','N')
                label_alt=label_alt.replace('a','U').replace('t','A').replace('c','G').replace('g','C')
            x_ref = torch.tensor(one_hot(x_ref, strands[i]))
            x_alt = torch.tensor(one_hot(x_alt, strands[i]))
            if torch.cuda.is_available():
                x_ref = x_ref.to(torch.device("cuda"))
                x_alt = x_alt.to(torch.device("cuda"))

            with torch.no_grad():
                y_ref_aws = model(x_ref)
                y_alt_aws = model(x_alt)
#             if strands[i] == '-':  
#                 y_ref_aws = y_ref_aws[::-1, ::-1]
#                 y_alt_aws = y_alt_aws[::-1, ::-1]
            ref_alt_aws=np.vstack((y_ref_aws, y_alt_aws))
            ref_alt_aws=data_normalise(ref_alt_aws)
            ref_aws=ref_alt_aws[0:81,:]
            alt_aws=ref_alt_aws[81:,:]
            draw_picture(label_ref,ref_aws,genes[i],'wt',distance)
            draw_picture(label_alt,alt_aws,genes[i],'vt',distance)




def spliformer_motif_predict(argsI, argsR, argsA, argsN, argsG):

    try:
        vcf = pysam.VariantFile(argsI)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    os.environ["CUDA_VISIBLE_DEVICES"] =  argsG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Spliformer_motif(5, 3, 64, 8, 1024, 6, 0.1).to(device)

    Model = torch.load(resource_filename(__name__,"weights/spliformer-motif.ckpt"),map_location=torch.device('cpu'))
    model.load_state_dict(Model['model_state_dict'])
    model.eval()
    if not os.path.exists('motif_result'):
            os.makedirs('motif_result')
    ann = Annotator(argsR, argsA)
    for mutation in vcf:
        predict(mutation, ann, model, argsN)

    vcf.close()
