#!/bin/bash
#SBATCH --job-name=train_mt_ca_en_morf
#SBATCH --output=slurm_logs/train_mt_en_ca_morf_%j.out
#SBATCH --error=slurm_logs/train_mt_en_ca_morf_%j.err
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --time=2-00:00:00


load_config_amd_0102() {
	module load gcc/10.2.0 rocm/5.1 intel/2018.4 mkl/2018.4 python/3.7.4

	export LD_LIBRARY_PATH=/gpfs/projects/bsc88/projects/bne/eval_amd/scripts_to_run/external-lib:$LD_LIBRARY_PATH

	source /gpfs/projects/bsc88/projects/bne/eval_amd/ksenia/venv-fairseq/bin/activate

	echo $SLURM_JOB_NODELIST
}

load_config_amd_0102

umask 007

data_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/data-bin-morf_j_sample'
save_dir='/gpfs/scratch/bsc88/bsc88400/tfm_gender_bias/checkpoints/enca-morf/'
src='en'
tgt='ca'


echo $data_dir

fairseq-train $data_dir \
    --save-dir $save_dir \
    -s $src -t $tgt \
    --arch transformer \
    --share-decoder-input-output-embed --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 3000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --encoder-normalize-before --decoder-normalize-before \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3072 \
    --max-update 250000 \
    --update-freq  8 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --distributed-no-spawn --save-interval-updates 1000 \
    --keep-interval-updates 10 --no-epoch-checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --dataset-impl lazy \
    --fp16 
