#BSUB -J 01_guppy
#BSUB -n 20
#BSUB -o %J.stdout
#BSUB -e %J.stderr
#BSUB -R span[hosts=1]
#BSUB -q gpu


export PATH=/work/bio-mowp/software/ont-guppy-6.0.0/ont-guppy/bin:$PATH

guppy_basecaller -i fast5 -s guppy_out_col_CEN -c res_dna_r941_min_modbases-all-context_v001.cfg --bam_out --bam_methylation_threshold 0 --fast5_out --device "cuda:all:100%" --num_callers 16 --gpu_runners_per_device 24 --chunks_per_runner 1024 --chunk_size 2000 -a /work/bio-mowp/db/col_CEN/Col-CEN_v1.2.fasta --num_alignment_threads 30

python scripts/extract_basemod_calls.py -i guppy_out_col_PEK/workspace -o 20220524_col_m6A.6mA_read_pos.tsv
