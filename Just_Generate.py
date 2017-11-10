from __future__ import absolute_import, division, print_function

import os
import pickle
import tflearn
import tensorflow as tf
from tflearn.data_utils import *
import re
from builtins import any as b_any

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ID = "TrumpGen_JG"
char_idx_file = 'char_idx.pickle'
path = "./Trump_fix.txt"

maxlen = 25
char_idx = None
if os.path.isfile(char_idx_file):
    print('Loading previous char_idx')
    char_idx = pickle.load(open(char_idx_file, 'rb'))

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(
        path, seq_maxlen=maxlen, redun_step=1)
pickle.dump(char_idx, open(char_idx_file, 'wb'))

# Instantiating checkpoint finder
checkpoint = False
list_of_files = os.listdir()
checkpoint_type = ".data-00000-of-00001"
if b_any(checkpoint_type in x for x in list_of_files):
    checkpoint = True

    def extract_number(f):
        s = re.findall("(\d+).data-00000-of-00001", f)
        return (int(s[0]) if s else -1, f)
    target = (max(list_of_files, key=extract_number))
    target = target.split('.')
    target = target[0]

# Begin Main loop
with tf.device('/cpu:0'):
    # Launch tensorboard (This is disabled as it causes Python to crash)
    #os.spawnl(os.P_NOWAIT, "tensorboard --logdir='/tmp/tflearn_logs/" + ID + "'")
    #os.spawnl(os.P_NOWAIT, "start \"\" http://localhost:6006")
    # Building layers in network
    g = tflearn.input_data([None, maxlen, len(char_idx)])
    g = tflearn.lstm(g, 256, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 256, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 256)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                           learning_rate=0.001)

    # stating model is to be used in tflearns sequence generator template
    m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                                  seq_maxlen=maxlen,
                                  clip_gradients=5.0,
                                  checkpoint_path='model_trump_Gen',
                                  max_checkpoints=1)
    # checking if checkpoint
    if checkpoint is True:
        m.load(target)
    seed = random_sequence_from_textfile(path, maxlen)

# Create 1 tweet length message
the_Trump_file = open('Trumpish_Snippet.txt', 'w')
Trumping = m.generate(250, temperature=.5,
                      seq_seed=seed)  # random sentence
the_Trump_file.write("\r%s\n" % Trumping + '...')
print('One line, coming up')
