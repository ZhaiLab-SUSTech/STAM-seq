#BSUB -J 03_deepsignal_plant
#BSUB -n 20
#BSUB -o %J.stdout
#BSUB -e %J.stderr
#BSUB -R span[hosts=1]
#BSUB -q gpu

module load gcc/9.3.0

DATE=$(date +%Y%m%d)

echo "`date "+%Y-%m-%d %H:%M:%S"` | call call_mods start"  >> ${DATE}".03_deepsignal_plant.log"
CUDA_VISIBLE_DEVICES=0,1 deepsignal_plant call_mods --input_path single_fast5_col_CEN   --model_path /work/bio-mowp/software/deepsignal_plant/model.dp2.CNN.arabnrice2-1_120m_R9.4plus_tem.bn13_sn16.both_bilstm.epoch6.ckpt   --result_file 20221029_col_m6A.C.call_mods.tsv   --corrected_group RawGenomeCorrected_000   --motifs C --nproc 30 --nproc_gpu 4
echo "`date "+%Y-%m-%d %H:%M:%S"` | call call_mods finished"  >> ${DATE}".03_deepsignal_plant.log"

echo "`date "+%Y-%m-%d %H:%M:%S"` | call modifications finished"  >> ${DATE}".03_deepsignal_plant.log"

echo "`date "+%Y-%m-%d %H:%M:%S"` | call call_freq start"  >> ${DATE}".03_deepsignal_plant.log"
deepsignal_plant call_freq --input_path 20221029_col_m6A.C.call_mods.tsv --result_file 20221029_col_m6A.C.call_mods.frequency.tsv

python /work/bio-mowp/software/deepsignal_plant/split_freq_file_by_5mC_motif.py --freqfile 20221029_col_m6A.C.call_mods.frequency.tsv
echo "`date "+%Y-%m-%d %H:%M:%S"` | call call_freq end"  >> ${DATE}".03_deepsignal_plant.log"
