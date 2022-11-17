import pysam
import gzip
from collections import defaultdict
import os
import click
from loguru import logger


seq2motif = {
    'CGN': 'CGN',
    'CGG': 'CGN',
    'CGC': 'CGN',
    'CGA': 'CGN',
    'CGT': 'CGN',
    'CCG': 'CHG',
    'CTG': 'CHG',
    'CAG': 'CHG',
    'CHG': 'CHG',
    'CAT': 'CHH',
    'CHH': 'CHH',
    'CTA': 'CHH',
    'CAC': 'CHH',
    'CTC': 'CHH',
    'CCT': 'CHH',
    'CAA': 'CHH',
    'CCA': 'CHH',
    'CCC': 'CHH',
    'CTT': 'CHH'
}


def load_mod_file(infile: str) -> dict:
    '''It reads in the output of the `deepsignal-plant` and returns two dictionaries, one for methylated
    positions and one for unmethylated positions for each read.
    
    Parameters
    ----------
    infile : str
        the file containing the methylation calls
    
    Returns
    -------
        methylated and unmethylated positions for each read.
    
    '''
    methy_results = defaultdict(lambda: {'CG': [], 'CHG': [], 'CHH': [], '5mC': []})
    unmethy_results = defaultdict(lambda: {'CG': [], 'CHG': [], 'CHH': [], '5mC': []})
    if os.path.splitext(infile)[-1] == '.gz':
        f = gzip.open(infile, 'rt')
    else:
        f = open(infile, 'r')
    
    for i, line in enumerate(f):
        chrom, pos, strand, _, read_id, read_strand, prob_unm, prob_m, called_label, kmer = line.rstrip().split('\t')
        called_label = int(called_label)

        genome_pos = pos
        # for 5mC motif
        kmer_cenpos = len(kmer)//2
        mod_type = seq2motif[kmer[kmer_cenpos:(kmer_cenpos+3)]]
        if mod_type.startswith('CG'):
            mod_type = 'CG'
        if called_label:
            methy_results[read_id][mod_type].append(genome_pos)
            methy_results[read_id]['5mC'].append(genome_pos)
        else:
            unmethy_results[read_id][mod_type].append(genome_pos)
            unmethy_results[read_id]['5mC'].append(genome_pos)
        
        if i % 100_000_000 == 0:
            logger.info(f'Processed {i} lines')
    
    return methy_results, unmethy_results


def parse_methylation_output(infile: str, methy_results: dict, unmethy_results:dict, outfile_prefix: str) -> dict:
    outfile = {}
    for mod_type in ('CG', 'CHG', 'CHH', '5mC'):
        outfile[mod_type] = open(outfile_prefix + '.' + mod_type + '_genome_pos' + '.tsv', 'w')
        outfile[mod_type].write('read_id\tmod_type\tchrom\tstart\tend\tstrand\tmetylated_pos\tunmethylated_pos\n')

    with pysam.AlignmentFile(infile, 'rb') as bam:
        for read in bam:
            if read.is_unmapped or read.is_supplementary or read.mapping_quality < 1:
                continue

            chrom = read.reference_name
            strand = '-' if read.is_reverse else '+'
            start = read.reference_start
            end = read.reference_end

            for mod_type in ('CG', 'CHG', 'CHH', '5mC'):
                if read.query_name in methy_results:
                    methylated_pos = ','.join(methy_results[read.query_name][mod_type])
                else:
                    methylated_pos = ''
                if read.query_name in unmethy_results:
                    unmethylated_pos = ','.join(unmethy_results[read.query_name][mod_type])
                else:
                    unmethylated_pos = ''
                outfile[mod_type].write(f'{read.query_name}\t{mod_type}\t{chrom}\t{start}\t{end}\t{strand}\t{methylated_pos}\t{unmethylated_pos}\n')
    
    for mod_type in ('CG', 'CHG', 'CHH', '5mC'):
        outfile[mod_type].close()


@click.command()
@click.option('--inbam', required=True)
@click.option('--inmod', required=True)
@click.option('--outprefix', required=True)
def main(inbam, inmod, outprefix):
    methy_results, unmethy_results = load_mod_file(inmod)
    parse_methylation_output(inbam, methy_results, unmethy_results, outprefix)


if __name__ == '__main__':
    main()