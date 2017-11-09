#Tensorflow Text generator

Derrivative work of: https://github.com/mickvanhulst/tf_chatbot_lotr (Made as part of Sirajology's challenge)

Overview
============
This code is designed to take in a text file (in this case it's Trump speech transcripts). After sufficent training (I was able to get it to a loss of 0.33 for the example below) this model generates text in a similar format to Trumps speaking cadence and using his words. But similar to mickvanhulst's findings, the results are... creative. This gets better with more training, but at 0.22 the results are not always english. 


Dependencies
============
* numpy
* scipy 
* tensorflow (https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html)
* pickle
* tflearn

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies


Usage
===========

Simply run 'python3 main.py' and your sentences are on the way!
To run tensorboard for that juicy play by play, run:
1. tensorboard --logdir='/tmp/tflearn_logs/Trumpish'
2. open localhost:6006 in your browser
3. profit

Result
===========
Output (using 128 Neurons per layer) Loss 0.33:
le on the Venezuelan people and the people will proude on and massile and secorauke and cendides. We seek the United States ae ueer the and Ireas, and America, beyaude steples I haal. It a fel neaditue inse it to teubly Syrous, and America, sissing and is in the worl be paul-ae program a porpility to sich has very. I selure of the people resware is a gratiens that presenukl action. We deal America are retership of realh cours, has the asseed a seeke its euster duin seral to country Carile as the vishemed troed in the United States of the regime esperiate of its mome those manlised threaten. In Syrieve thee fal in thei

le on the Venezuelan people. We are fortune head. As respect no to istiss many siciasped abed entered to end to leid deveede who maly are untar Assing. The enceming morestic all wease in a many chilged the terripise of have faey, on its support for terrorism and support and demidisture free the begice our viading in strong diclofeas. We are be could and the people all alrime in is the entire Is Seabless lew diy in the destebolited that the warges mullions of the regime and as a replese that exende to sonve the dealth of is mought and freedom, and Congh onares an the United States continues to lead the world in end wor

The rest of the output is in Trumpish.txt

Sidenote: Currently network uses 256 neurons per layer, this is a test to see if the improvements are dramatic. If they are, then output will be updated.

Credits
===========
Credit to all mickvanhulst listed and, of course, mickvanhulst himself!
