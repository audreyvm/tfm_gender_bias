#!/bin/bash
#SBATCH --job-name=tests-preprocess
#SBATCH --output=slurm_logs/preprocess-all_tests_%j.out
#SBATCH --error=slurm_logs/preprocess-all_tests_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --qos=debug


load_config_amd_0102() {
        module load gcc/10.2.0 rocm/5.1 intel/2018.4 mkl/2018.4 python/3.7.4

        export LD_LIBRARY_PATH=/gpfs/projects/bsc88/projects/bne/eval_amd/scripts_to_run/external-lib:$LD_LIBRARY_PATH

        source /gpfs/projects/bsc88/projects/bne/eval_amd/ksenia/venv-fairseq/bin/activate

        echo $SLURM_JOB_NODELIST
}

preprocess() {

    mkdir -p $dest_dir

    python $(which fairseq-preprocess) --source-lang $src_lang \
        --target-lang $tgt_lang \
        --testpref "${data_path}/test.${model_type}" \
        --destdir $dest_dir \
        --workers 128 \
        --joined-dictionary \
        --srcdict $dict_dir \
        --dataset-impl lazy
}




load_config_amd_0102
umask 007


#Copy test that not follow the prefix.language pattern

flores_dev_dir='/gpfs/projects/bsc88/corpora/flores_101_multilingual/v1/s1/flores101_dataset/dev'
data='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/test_data/flores_dev'
mkdir -p $data
cp ${flores_dev_dir}/cat.dev ${data}/test.ca
cp ${flores_dev_dir}/eng.dev ${data}/test.en
flores_dev_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/test_data/flores_dev'

flores_devtest_dir='/gpfs/projects/bsc88/corpora/flores_101_multilingual/v1/s1/flores101_dataset/devtest'
data='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/test_data/flores_devtest'
mkdir -p $data
cp ${flores_devtest_dir}/cat.devtest ${data}/test.ca
cp ${flores_devtest_dir}/eng.devtest ${data}/test.en
flores_devtest_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/test_data/flores_devtest'

must_she_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/must_she_data'
data='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/test_data/must_she_data'
mkdir -p $data
cp ${must_she_dir}/cat.test ${data}/test.ca
cp ${must_she_dir}/eng.test ${data}/test.en
must_she_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/test_data/must_she_data'


# Subword tokenization with BPE
merges=32000
input_size=12000000
model_type='bpe'

dict='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-bpe32/dict.en.txt'

for data_dir in $flores_dev_dir $flores_devtest_dir $must_she_dir
do
    subword-nmt apply-bpe -c codes_${model_type}${merges}.enca --vocabulary vocab_${model_type}${merges}.en --vocabulary-threshold 50 < ${data_dir}/test.ca > ${data_dir}/test.${model_type}.ca
    subword-nmt apply-bpe -c codes_${model_type}${merges}.enca --vocabulary vocab_${model_type}${merges}.en --vocabulary-threshold 50 < ${data_dir}/test.en > ${data_dir}/test.${model_type}.en

done

# Fairseq preprocess

src_lang='en'
tgt_lang='ca'

#Flores dev
dest_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-tests/flores_dev_bpe'
rm -r $dest_dir
mkdir -p $dest_dir
dict_dir=$dict
data_path=$flores_dev_dir
preprocess

#Flores devtest
dest_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-tests/flores_devtest_bpe'
rm -r $dest_dir
mkdir -p $dest_dir
dict_dir=$dict
data_path=$flores_devtest_dir
preprocess

#Must_She
dest_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-tests/must_she_data_bpe'
rm -r $dest_dir
mkdir -p $dest_dir
dict_dir=$dict
data_path=$must_she_dir
preprocess


