import sys
import os
import morfessor

io = morfessor.MorfessorIO()
model = io.read_binary_model_file('morfessor.model')

input_data=open('{}'.format(sys.argv[1]), 'r')
output_data=open('{}'.format(sys.argv[2]), 'w')

for line in input_data:
    tokens = []
    words = line.split()
    for word in words:
        segments = model.viterbi_segment(word)[0]
        segments[0]= '▁'+segments[0]
        tokens.append(' '.join(segments))
    sentence = ' '.join(tokens)
    print(sentence, file=output_data)




