#!/bin/bash

#SBATCH --job-name=preprocess-tfm
#SBATCH --output=slurm_logs/preprocess-aud-uni32_%j.out
#SBATCH --error=slurm_logs/preprocess-aud-uni32_%j.err
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
        --trainpref "${data_dir}/train.${model_type}.${vocab_size}" \
        --validpref "${data_dir}/val.${model_type}.${vocab_size}" \
        --testpref "${data_dir}/test.${model_type}.${vocab_size}" \
        --destdir $dest_dir \
        --workers 128 \
        --joined-dictionary \
        --dataset-impl lazy
}




src='en'
tgt='ca'
data_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data'
dest_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-uni32'
vocab_size=32000
input_size=12000000
model='model'
model_type='uni'

mkdir -p $data_dir
mkdir -p $dest_dir

cat ${data_dir}/train.${src} ${data_dir}/train.${tgt} > $data_dir/train_data.${src}${tgt}.txt

# input, prefix, vocab_size
echo 'Compute spm vocabulary'
python get_vocabulary.py $data_dir/train_data.${src}${tgt}.txt $data_dir/${model} ${vocab_size} ${input_size}


# Subword tokenization with SentencePiece
for lang in ${src} ${tgt} 
do
    python spm_encode.py --model $data_dir/${model}.model --inputs ${data_dir}/train.${lang} --outputs ${data_dir}/train.${model_type}.${vocab_size}.${lang}
    python spm_encode.py --model $data_dir/${model}.model --inputs ${data_dir}/val.${lang} --outputs ${data_dir}/val.${model_type}.${vocab_size}.${lang}
    python spm_encode.py --model $data_dir/${model}.model --inputs ${data_dir}/test.${lang} --outputs ${data_dir}/test.${model_type}.${vocab_size}.${lang}
done

#Binarize data
preprocess
