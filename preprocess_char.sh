#!/bin/bash

#SBATCH --job-name=char_preprocess-tfm
#SBATCH --output=slurm_logs/preprocess-char_%j.out
#SBATCH --error=slurm_logs/preprocess-char_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00


load_config_amd_0102() {
        module load gcc/10.2.0 rocm/5.1 intel/2018.4 mkl/2018.4 python/3.7.4

        export LD_LIBRARY_PATH=/gpfs/projects/bsc88/projects/bne/eval_amd/scripts_to_run/external-lib:$LD_LIBRARY_PATH

        source /gpfs/projects/bsc88/projects/bne/eval_amd/ksenia/venv-fairseq/bin/activate

        echo $SLURM_JOB_NODELIST
}

load_config_amd_0102
umask 007

preprocess() {

    mkdir -p $dest_dir

    python $(which fairseq-preprocess) --source-lang $src \
        --target-lang $tgt \
        --trainpref "${data_dir}/train.${model_type}" \
        --validpref "${data_dir}/val.${model_type}" \
        --testpref "${data_dir}/test.${model_type}" \
        --destdir $dest_dir \
        --workers 128 \
        --joined-dictionary \
        --dataset-impl lazy
}




src='en'
tgt='ca'
data_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data'
dest_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-char'
input_size=3000000
model_type='char'

mkdir -p $data_dir
mkdir -p $dest_dir

for lang in $src $tgt 
do
    python char_split.py < ${data_dir}/train.${lang} > ${data_dir}/train.${model_type}.${lang}
    python char_split.py < ${data_dir}/val.${lang} > ${data_dir}/val.${model_type}.${lang}
    python char_split.py < ${data_dir}/test.${lang} > ${data_dir}/test.${model_type}.${lang}
done

#Binarize data
preprocess
