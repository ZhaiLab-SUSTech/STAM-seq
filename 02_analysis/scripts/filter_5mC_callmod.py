import pysam
import gzip
import click


@click.command()
@click.option('--bam', required=True, help='BAM file')
@click.option('--tsv', required=True, help='5mC call mod file')
@click.option('--out', required=True, help='Output file')
def main(bam, tsv, out):
    '''
    keep unique mapped reads within 5mC call mod file
    
    Parameters
    ----------
    bam
        The input bam file
    tsv
        C.call_mods.tsv file from deepsignal plant
    out
        the output file name
    
    '''
    read_id_set = set()
    with pysam.AlignmentFile(bam, 'rb') as inbam:
        for read in inbam:
            read_id_set.add(read.query_name)
    
    with gzip.open(tsv, 'rt') as instv, gzip.open(out, 'wt') as outtsv:
        for line in instv:
            _line = line.strip().split('\t')
            read_id = _line[4]
            if read_id in read_id_set:
                outtsv.write(line)


if __name__ == '__main__':
    main()