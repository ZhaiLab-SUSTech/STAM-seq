from Bio import SeqIO
import numpy as np
import pickle

arabidopsis_genome = {}
arabidopsis_genome_fa = '/data/Zhaijx/mowp/db/col-CEN/dna/Col-CEN_v1.2.fasta'
for read in SeqIO.parse(arabidopsis_genome_fa, 'fasta'):
    arabidopsis_genome[read.id] = np.array(read.seq)
# save pickle
arabidopsis_genome_pkl = open("/data/Zhaijx/mowp/db/col-CEN/dna/Col-CEN_v1.2.pkl", "wb")
pickle.dump(arabidopsis_genome, arabidopsis_genome_pkl)
arabidopsis_genome_pkl.close()