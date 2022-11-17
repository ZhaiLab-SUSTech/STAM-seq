from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import click
from itertools import repeat


qthreshInt=10
qthresh = chr(qthreshInt+33)
mAthresh1=252 #255*0.99
mAthresh2=250 #255*0.98
mAthresh3=242 #255*0.95
mAthresh4=230 #255*0.9
mAthresh5=204 #255*0.8
mAthresh6=128 #255*0.5

totA=0
totbases=0

def processFile(fast5_filepath, mAthresh=128):
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        results = []
        for read_id in f5.get_read_ids():
            read = f5.get_read(read_id)
            latest_basecall = read.get_latest_analysis('Basecall_1D')
            mod_base_table = np.array(read.get_analysis_dataset(latest_basecall, 'BaseCalled_template/ModBaseProbs').transpose()[1,:])
            name, sequence, _, qstring = read.get_analysis_dataset(latest_basecall,'BaseCalled_template/Fastq').strip().split('\n')
            Athreshvec = np.logical_and(np.array(tuple(qstring)) >= qthresh, np.array(tuple(sequence)) == 'A')  # basecalling calling 出来是A的
            mAthreshvec = np.logical_and(Athreshvec, mod_base_table >= mAthresh)  # decide threshold here, A为6mA的概率
            unmAthreshvec = np.logical_and(Athreshvec, mod_base_table < mAthresh)
            m6A_pos = ','.join(map(str, np.flatnonzero(mAthreshvec)))  # 6mA的位置
            unm6A_pos = ','.join(map(str, np.flatnonzero(unmAthreshvec)))
            prob_m = ','.join(map(str, (mod_base_table[mAthreshvec]/255).round(4)))  # 6mA的概率
            readlen = len(sequence)
            Acount = np.sum(Athreshvec)
            mAcount = np.sum(np.logical_and(Athreshvec, mod_base_table >= mAthresh))

            outputstring = f'{read_id}\t{readlen}\t{Acount}\t{mAcount}\t{m6A_pos}\t{unm6A_pos}'
            results.append(outputstring)
    return results


@click.command()
@click.option('-i', '--fast5_path', required=True)
@click.option('-o', '--outfile', default='output_methylAthresh.txt')
@click.option('-t', '--threads', default=None)
@click.option('-m', '--mthresh', default=.5)
def main(fast5_path, outfile, threads, mthresh):
    files = list(glob.glob(fast5_path+'/*.fast5'))
    if threads is None:
        threads = multiprocessing.cpu_count()
    if threads > len(files):
        threads = len(files)
    mAthresh = 255*mthresh

    with ProcessPoolExecutor(max_workers=threads) as e:
        chunksize = int(len(files) / threads)
        results = e.map(
            processFile,
            files,
            repeat(mAthresh),
            chunksize=chunksize)

    with open(outfile, 'w') as o:
        print('read_id\treadlen\tAcount\tmAcount\tmetylated_pos\tunmethylated_pos', file=o)
        for res in results:
            for line in res:
                print(line, file=o)


if __name__ == '__main__':
    main()