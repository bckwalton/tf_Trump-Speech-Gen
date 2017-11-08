from __future__ import absolute_import, division, print_function

import os
import pickle
import tflearn
import tensorflow as tf
from tflearn.data_utils import *
import re
from builtins import any as b_any

tf.logging.set_verbosity(tf.logging.ERROR)

ID = "TrumpGen"
path = "./Trump.txt"
char_idx_file = 'char_idx.pickle'

#Text fixer
with open(path, 'rb') as f:
    lines = [x.decode('utf8').strip() for x in f.readlines()]
    fix_path = open("./Trump_fix.txt", 'w')
    for line in lines:
        fix_path.write(line)
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
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

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
with tf.device('/gpu:0'):
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
                           learning_rate=0.01) # changed from 0.001

    # stating model is to be used in tflearns sequence generator template
    m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                                  seq_maxlen=maxlen,
                                  clip_gradients=5.0,
                                  checkpoint_path='model_trump',
                                  max_checkpoints=10, tensorboard_verbose=3)
    # checking if checkpoint
    if checkpoint is True:
        m.load(target)
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=100, run_id='Trumpish')

# Create ten sentences and add them to a file
the_Trump_file = open('Trumpish.txt', 'w')
i = 0
for i in range(10):
    Trumping = m.generate(600, temperature=1.0,
                          seq_seed=seed)  # random sentence
    the_Trump_file.write("\r%s\n" % Trumping)
    print('Line: ')
    i = i + 1
    print()
