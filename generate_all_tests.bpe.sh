#!/bin/bash
#SBATCH --job-name=tfm_generate_tests
#SBATCH --output=slurm_logs/tfm_generate_tests_%j.out
#SBATCH --error=slurm_logs/tfm_generate_tests_%j.err
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

load_config_amd_0102
umask 007

src='en'
tgt='ca'
model='bpe32'

generate() {

    python $(which fairseq-generate) $data \
        --path $checkpoint \
        --beam 8 --batch-size 32 \
        --task translation  \
        -s $src -t $tgt \
         > $translation.log
    
    cat $translation.log | sed 's/@@ //g' | grep -P "^H" |sort -V |cut -f 3- > $translation 

    sacrebleu $reference -i $translation --metrics {bleu,chrf,ter} >> $scores


}


#Output directory
out_dir='results'
scores="${out_dir}/scores_enca_${model}.txt"
mkdir -p $out_dir


#Path to reference data

flores_dev_ref='/gpfs/projects/bsc88/corpora/flores_101_multilingual/v1/s1/flores101_dataset/dev/cat.dev'
flores_devtest_ref='/gpfs/projects/bsc88/corpora/flores_101_multilingual/v1/s1/flores101_dataset/devtest/cat.devtest'
must_she_ref='must_she_data/cat.test'


#model

checkpoint="checkpoints/enca-${model}/checkpoint_best.pt"

#Path to the binarized data
flores_dev='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-tests/flores_dev_bpe'
flores_devtest='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-tests/flores_devtest_bpe'
must_she='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-tests/must_she_data_bpe'


echo "BLEU scores TFM data" > $scores

echo "" >> $scores
echo "Flores Dev" >> $scores
echo "" >> $scores
reference=$flores_dev_ref
data=$flores_dev
translation="${out_dir}/flores_dev_enca_${model}.ca"
generate

echo "" >> $scores
echo "Flores DevTest" >> $scores
echo "" >> $scores
reference=$flores_devtest_ref
data=$flores_devtest
translation="${out_dir}/flores_devtest_enca_${model}.ca"
generate

echo "" >> $scores
echo "MuST SHE" >> $scores
echo "" >> $scores
reference=$must_she_ref
data=$must_she
translation="${out_dir}/mustshe_enca_${model}.ca"
generate


