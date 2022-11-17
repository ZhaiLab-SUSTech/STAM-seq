import pysam
import gzip
import os
import click
import numpy as np
import pandas as pd
import pickle
from loguru import logger


'''
## Make reference into pickle:

from Bio import SeqIO

arabidopsis_genome = {}
arabidopsis_genome_fa = '/data/Zhaijx/mowp/db/col-PEK/dna/file1.Col-PEK1.5-Chr1-5_20220523.fasta'
for read in SeqIO.parse(arabidopsis_genome_fa, 'fasta'):
    arabidopsis_genome[read.id] = np.array(read.seq)
# save pickle
arabidopsis_genome_pkl = open("/data/Zhaijx/mowp/db/col-PEK/dna/Col-PEK1.5-Chr1-5.pkl", "wb")
pickle.dump(arabidopsis_genome, arabidopsis_genome_pkl)
arabidopsis_genome_pkl.close()

'''


def load_genome_pkl(infile: str) -> dict:
    genome_pkl = open(infile, 'rb')
    genome = pickle.load(genome_pkl)
    genome_pkl.close()

    return genome


def load_mod_file(infile: str) -> dict:
    '''It takes a file with the results of the modified base calling and returns a dictionary with the read
    id as the key and the read position and probability of modification as the value
    
    Parameters
    ----------
    infile : str
        the file that contains the results of the modified base calling
    
    Returns
    -------
        A dictionary with the read_id as the key and a tuple of the read_pos and prob_m as the value.
    
    '''
    mod_results = {}
    if os.path.splitext(infile)[-1] == '.gz':
        f = gzip.open(infile, 'rt')
    else:
        f = open(infile, 'r')

    next(f)
    for i, line in enumerate(f):
        read_id, readlen, Acount, mAcount, methylated_pos, unmethylated_pos = line.split('\t')
        unmethylated_pos = unmethylated_pos.rstrip()
        methylated_pos = np.fromstring(methylated_pos, sep=',', dtype=int) 
        unmethylated_pos = np.fromstring(unmethylated_pos, sep=',', dtype=int)
        mod_results[read_id] = (methylated_pos, unmethylated_pos)

        if i % 1_000_000 == 0:
            logger.info(f'{i} reads processed')

    return mod_results


def parse_methylation_output(infile: str, mod_results: dict, genome: dict, outfile: str):
    '''For each read, if it is mapped to the genome, then find the positions of the methylated A's in the
    read, and then find the corresponding positions in the genome
    
    Parameters
    ----------
    infile : str
        the path to the bam file
    mod_results : dict
        a dictionary with read_id as key and a list of two numpy arrays as value. The first array is the
    position of the modified base in the read, and the second array is the probability of the modified
    base being methylated.
    genome : dict
        a dict of {chrom: str}
    outfile : str
        the output file name
    
    '''
    out = open(outfile, 'w')
    print('read_id\tmod_type\tchrom\tstart\tend\tstrand\tmethylated_pos\tunmethylated_pos', file=out)

    with pysam.AlignmentFile(infile, 'rb') as bam:
        for read in bam:
            if read.is_unmapped or read.is_supplementary or read.mapping_quality < 1:
                continue

            _genome_pos = np.array(read.get_reference_positions(full_length=True))
            _metylated_pos = mod_results[read.query_name][0]
            _unmetylated_pos = mod_results[read.query_name][1]

            chrom = read.reference_name
            strand = '-' if read.is_reverse else '+'
            start = read.reference_start
            end = read.reference_end
            query_length = read.query_length

            if strand == '+':
                boo1 = ~pd.isnull(_genome_pos[_metylated_pos])  # 被鉴定位mA，且能比对到基因组的位置
                boo2 = genome[chrom][list(_genome_pos[_metylated_pos][boo1])] == 'A' # 被鉴定位mA，且基因组上位置也是A的位置
                methylated_pos = _genome_pos[_metylated_pos[boo1][boo2]]

                boo1 = ~pd.isnull(_genome_pos[_unmetylated_pos])
                boo2 = genome[chrom][list(_genome_pos[_unmetylated_pos][boo1])] == 'A'
                unmethylated_pos = _genome_pos[_unmetylated_pos[boo1][boo2]]

            else:
                boo1 = ~pd.isnull(_genome_pos[query_length - _metylated_pos - 1])  # 被鉴定位mA，且能比对到基因组的位置
                boo2 = genome[chrom][list(_genome_pos[query_length - _metylated_pos - 1][boo1])] == 'T' # 被鉴定位mA，且基因组上位置也是A的位置
                methylated_pos = _genome_pos[query_length - _metylated_pos[boo1][boo2] - 1]
                
                boo1 = ~pd.isnull(_genome_pos[query_length - _unmetylated_pos - 1])
                boo2 = genome[chrom][list(_genome_pos[query_length - _unmetylated_pos - 1][boo1])] == 'T'
                unmethylated_pos = _genome_pos[query_length - _unmetylated_pos[boo1][boo2] - 1]
                
            
            methylated_pos = ','.join(map(str, methylated_pos))  # 0-based leftmost coordinate
            unmethylated_pos = ','.join(map(str, unmethylated_pos))
            
            print(f'{read.query_name}\tmA\t{chrom}\t{start}\t{end}\t{strand}\t{methylated_pos}\t{unmethylated_pos}', file=out)
    
    out.close()


@click.command()
@click.option('--inbam', required=True)
@click.option('--inmod', required=True)
@click.option('--genome_pkl', required=True)
@click.option('--outfile', required=True)
def main(inbam, inmod, genome_pkl, outfile):
    genome = load_genome_pkl(genome_pkl)
    mod_results = load_mod_file(inmod)
    parse_methylation_output(inbam, mod_results, genome, outfile)


if __name__ == '__main__':
    main()