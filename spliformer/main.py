from spliformer.spliformer_motif import spliformer_motif_predict
from spliformer.spliformer_general import spliformer_predict
import argparse
import logging
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', metavar='input', nargs='?',
                        help='The input VCF file path')
    parser.add_argument('-O', metavar='output', nargs='?',
                        help='The output VCF file path')
    parser.add_argument('-R', metavar='reference', default='./reference/hg19.fa',
                        help='The reference genome fasta file path')
    parser.add_argument('-A', metavar='annotation', default='./reference/hg19annotation.txt',
                        help='annotation file converted from GENCODE V39 gtf,default=./reference/hg19anno.txt')
    parser.add_argument('-D', metavar='distance', default=50,
                        type=int, choices=range(0, 5000),
                        help='maximum distance between the variant and gained/lost splice '
                             'site, defaults to 50')
    parser.add_argument('-N', metavar='number', default=10, type=int, choices=range(0, 41),
                        help='Number of target motifs on each side of the variant in heatmap, only can be used in motif model')
    parser.add_argument('-M', metavar='mask', default=0,
                        type=int, choices=[0, 1],
                        help='mask scores representing annotated acceptor/donor gain and '
                             'unannotated acceptor/donor loss, defaults to 0')
    parser.add_argument('-G', metavar='gpu', default='0',
                        help='choose the gpu need to use')
    parser.add_argument('-T', metavar='Tool', default='general', choices=['general', 'motif'],
                        help='choose which prediction tool to use')
    args = parser.parse_args()
    
    if args.T=='general':
        print('Starting the prediction by Spliformer-general tool')
        spliformer_predict(args.I, args.O, args.R, args.A, args.D, args.M, args.G)
        
    elif args.T=='motif':
        print('Starting the prediction by Spliformer-motif tool')
        spliformer_motif_predict(args.I, args.R, args.A, args.N, args.G)
        
    else:
        print('please choose the right precition tool!')
        
if __name__ == '__main__':
    main()