#!/bin/bash

#SBATCH --job-name=preprocess-tfm
#SBATCH --output=slurm_logs/preprocess-aud-bpe32_%j.out
#SBATCH --error=slurm_logs/preprocess-aud-bpe32_%j.err
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

pip install --no-index --find-links /gpfs/projects/bsc88/wheels subword-nmt

preprocess() {

    mkdir -p $dest_dir

    python $(which fairseq-preprocess) --source-lang $src \
        --target-lang $tgt \
        --trainpref "${data_dir}/train.${model_type}.${merges}" \
        --validpref "${data_dir}/val.${model_type}.${merges}" \
        --testpref "${data_dir}/test.${model_type}.${merges}" \
        --destdir $dest_dir \
        --workers 128 \
        --joined-dictionary \
        --dataset-impl lazy
}




src='en'
tgt='ca'
data_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data'
dest_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-bpe32'
merges=32000
input_size=12000000
model_type='bpe'

mkdir -p $data_dir
mkdir -p $dest_dir

# input, prefix, merges
echo 'Compute bpe vocabulary'

subword-nmt learn-joint-bpe-and-vocab --input ${data_dir}/train.${src} ${data_dir}/train.${tgt} -s ${merges} -o codes_${model_type}${merges}.${src}${tgt} --write-vocabulary vocab_${model_type}${merges}.${src} vocab_${model_type}${merges}.${tgt}

# Subword tokenization 
for lang in $src $tgt 
do
    subword-nmt apply-bpe -c codes_${model_type}${merges}.${src}${tgt} --vocabulary vocab_${model_type}${merges}.${src} --vocabulary-threshold 50 < ${data_dir}/train.${lang} > ${data_dir}/train.${model_type}.${merges}.${lang}
    subword-nmt apply-bpe -c codes_${model_type}${merges}.${src}${tgt} --vocabulary vocab_${model_type}${merges}.${src} --vocabulary-threshold 50 < ${data_dir}/test.${lang} > ${data_dir}/test.${model_type}.${merges}.${lang}
    subword-nmt apply-bpe -c codes_${model_type}${merges}.${src}${tgt} --vocabulary vocab_${model_type}${merges}.${src} --vocabulary-threshold 50 < ${data_dir}/val.${lang} > ${data_dir}/val.${model_type}.${merges}.${lang}
done

#Binarize data
preprocess