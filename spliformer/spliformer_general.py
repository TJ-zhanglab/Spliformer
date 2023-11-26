import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from spliformer.Network_general import *
from pyfaidx import Fasta
import logging
import pysam
from pkg_resources import resource_filename
import os

#This script was adpated from SpliceAI's prediction script,https://github.com/Illumina/SpliceAI/blob/master/spliceai/utils.py
class Annotator:
    def __init__(self, ref_fasta, annotations):

        df = pd.read_csv(annotations, sep='\t', dtype={'CHR': object})
        self.genes = df['GENE'].to_numpy()
        self.chroms = df['CHR'].to_numpy()
        self.strands = df['STRAND'].to_numpy()
        self.tx_starts = df['GENE_START'].to_numpy() + 1
        self.tx_ends = df['GENE_END'].to_numpy()
        self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i])
                            for c in df['EXON_START'].to_numpy()]
        self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                          for c in df['EXON_END'].to_numpy()]


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
        dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx]) - pos, key=abs)
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)

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
        seqs = preprocess_inputs(seqs[::-1])  

    return seqs


def normalise_chrom(source, target): #source=user's; target=annotation table
    def has_prefix(x):
        return x.startswith('chr')

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')
    elif not has_prefix(source) and has_prefix(target):
        return 'chr' + source

    return source


def predict(record, ann, dist_var, mask, model):

    middle = 20001
    distance = 5000 - dist_var
    delta_scores = []

    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Please check your variant input: {}'.format(record))
        delta_scores.append('"Input error, check your input info"')
        return delta_scores

    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        logging.warning('this position is not included in gene annotation transcript: {}'.format(record))
        delta_scores.append('"This position is not included in gene annotation transcript"')
        return delta_scores
    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][
              record.pos - middle // 2 - 1:record.pos + middle // 2].seq  # get seqs,range pos-5001:pos+5000,the first nucleictide won't extract
    except (IndexError, ValueError):
        logging.warning('Cannot be extracted from fastafile: {}'.format(record))
        delta_scores.append('"Cannot be extracted from fastafile"')
        return delta_scores

    if seq[middle // 2:middle // 2 + len(record.ref)].upper() != record.ref:  # ensure the ref base is right
        logging.warning('The REF is not same to reference genome: {}'.format(record))
        delta_scores.append('"The REF is not same to reference genome"')
        return delta_scores

    if len(seq) != middle:
        logging.warning('The variant is near chromosome end): {}'.format(record))
        delta_scores.append('"The variant is near chromosome end"')
        return delta_scores

    if len(record.ref) > dist_var:
        logging.warning('The REF is too long: {}'.format(record))  # ref base can't be longer than dis
        delta_scores.append('"The ref is too long"')
        return delta_scores

    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue

            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            if len(record.ref) > 1 and len(record.alts[j]) > 1:
                delta_scores.append('"please check your REF and ALT format"')
                continue
            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len - alt_len, 0)

            x_ref = seq.upper()
            x_alt = x_ref[:middle // 2] + str(record.alts[j]) + x_ref[middle // 2 + ref_len:]


            x_ref = torch.tensor(one_hot(x_ref, strands[i]))
            x_alt = torch.tensor(one_hot(x_alt, strands[i]))
            if torch.cuda.is_available():
                x_ref = x_ref.to(torch.device("cuda"))
                x_alt = x_alt.to(torch.device("cuda"))

            with torch.no_grad():
                y_ref = model(x_ref).cpu().numpy()
                y_alt = model(x_alt).cpu().numpy()

            if strands[i] == '-':  
                y_ref = y_ref[:, ::-1]
                y_alt = y_alt[:, ::-1]
            if ref_len > 1 and alt_len == 1:    #del
                y_alt = np.concatenate([
                    y_alt[:, distance:distance + dist_var + alt_len],
                    np.zeros((1, del_len, 3)),
                    y_alt[:, distance + dist_var + alt_len:-distance]],
                    axis=1)
            elif ref_len == 1 and alt_len > 1: #ins
                y_alt = np.concatenate([
                    y_alt[:, distance:distance+dist_var],
                    np.max(y_alt[:, distance+dist_var:distance+dist_var + alt_len], axis=1)[:, None, :],
                    y_alt[:, distance+dist_var + alt_len:-distance]],
                    axis=1)

            else:
                y_alt=y_alt[:, distance:-distance]
            
            y = np.concatenate([y_ref[:, distance:-distance], y_alt[:, :]])  #predicted range
           
            idx_pa = (y[1, :, 0] - y[0, :, 0]).argmax()  # alt.acc-ref.acc
            idx_na = (y[0, :, 0] - y[1, :, 0]).argmax()  # ref.acc - alt.acc
            idx_pd = (y[1, :, 1] - y[0, :, 1]).argmax()  # alt.don - ref.don
            idx_nd = (y[0, :, 1] - y[1, :, 1]).argmax()  # ref.don- alt.don

            mask_pa = np.logical_and((idx_pa - dist_var == dist_ann[2]), mask) #mask/not
            mask_na = np.logical_and((idx_na - dist_var != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd - dist_var == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd - dist_var != dist_ann[2]), mask)
            delta_scores.append("{}>{}|{}|{}:{:.2f}|{}:{:.2f}|{}:{:.2f}|{}:{:.2f}".format(
                record.ref,
                record.alts[j],
                genes[i],
                idx_pa - dist_var,#acceptor gain distance
                (y[1, idx_pa, 0] - y[0, idx_pa, 0]) * (1 - mask_pa),#acceptor gain
                idx_na - dist_var,#acceptor loss distance
                (y[0, idx_na, 0] - y[1, idx_na, 0]) * (1 - mask_na),#acceptor loss
                idx_pd - dist_var,#donor gain distance
                (y[1, idx_pd, 1] - y[0, idx_pd, 1]) * (1 - mask_pd),#donor gain
                idx_nd - dist_var,#donor loss distance
                (y[0, idx_nd, 1] - y[1, idx_nd, 1]) * (1 - mask_nd)#donor loss
                ))

    return delta_scores




def spliformer_predict(argsI, argsO, argsR, argsA, argsD, argsM, argsG):

    try:
        vcf = pysam.VariantFile(argsI)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    header = vcf.header
    header.add_line('##INFO=<ID=Spliformer,Number=.,Type=String,Description="Spliformer1.0 version '
                    'Format: Ref>Alt|gene|Increased acceptor dis: score|Decreased acceptor dis: score|Increased donor dis: score|Decreased donor dis: score">')

    try:
        output = pysam.VariantFile(argsO, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] =  argsG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Spliformer(5, 3, 64, 8, 1024,
                           4, 50, 29, [11, 21, 41],
                           0.1).to(device)
    
    Model = torch.load(resource_filename(__name__,"weights/spliformer.ckpt"),map_location=torch.device('cpu'))
    model.load_state_dict(Model['model_state_dict'])
    model.eval()

    ann = Annotator(argsR, argsA)
    for mutation in vcf:
        scores = predict(mutation, ann, argsD, argsM, model)
        if len(scores) > 0:
            mutation.info['Spliformer'] = scores
        output.write(mutation)

    vcf.close()
    output.close()

