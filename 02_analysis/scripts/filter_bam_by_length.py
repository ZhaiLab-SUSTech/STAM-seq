import pysam
import click


def filter_bam(infile: str, outfile: str, min_len: int=1000):
    with pysam.AlignmentFile(infile, "rb") as inbam, pysam.AlignmentFile(outfile, "wb", template=inbam) as outbam:
        for read in inbam:
            if read.query_length >= min_len:
                outbam.write(read)
    
    pysam.index(outfile, '-@ 10')


@click.command()
@click.option("--infile", "-i", type=str, required=True)
@click.option("--outfile", "-o", type=str, required=True)
@click.option("--min_len", "-l", type=int, default=1000)
def main(infile, outfile, min_len):
    filter_bam(infile, outfile, min_len)


if __name__ == "__main__":
    main()