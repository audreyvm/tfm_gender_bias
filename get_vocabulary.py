import sys
import sentencepiece as spm


spm.SentencePieceTrainer.train('--input={} --model_prefix={} --vocab_size={}  --input_sentence_size={} --shuffle_input_sentence=true'.format(sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4]))






