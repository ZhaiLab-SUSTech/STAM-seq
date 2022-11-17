#BSUB -J 02_resquiggle
#BSUB -n 4
#BSUB -o %J.stdout
#BSUB -e %J.stderr
#BSUB -R span[hosts=1]
#BSUB -q ser

DATE=$(date +%Y%m%d)

echo "`date "+%Y-%m-%d %H:%M:%S"` | multi_to_single_fast5 start"  >> ${DATE}".02_resquiggle.log"
multi_to_single_fast5 -i guppy_out_col_CEN/workspace -s single_fast5_col_CEN -t 40 --recursive
echo "`date "+%Y-%m-%d %H:%M:%S"` | multi_to_single_fast5 finished"  >> ${DATE}".02_resquiggle.log"

echo "`date "+%Y-%m-%d %H:%M:%S"` | resquiggle start"  >> ${DATE}".02_resquiggle.log"
tombo resquiggle   single_fast5_col_PEK/   /work/bio-mowp/db/col_CEN/Col-CEN_v1.2.fasta   --corrected-group RawGenomeCorrected_000   --basecall-group Basecall_1D_000 --overwrite --processes 40
echo "`date "+%Y-%m-%d %H:%M:%S"` | resquiggle finished"  >> ${DATE}".02_resquiggle.log"
