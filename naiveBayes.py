
import pandas as pd
from math import log
from getProbabilities import message_preprocessing

hate = pd.read_csv('datasets/hate_words_prob.csv')
hate.columns = ['word','count']

no_hate = pd.read_csv('datasets/no_hate_words_prob.csv')
no_hate.columns = ['word','count']

total_hate = len(hate)
total_no_hate = len(no_hate)
total = total_hate + total_no_hate

p_hate = total_hate/total
p_no_hate = total_no_hate/total

message = 'billionaire chair of trump says he hopes obama dies and michelle goes back to being a male ape in africa.'
print(message)
#message_processed = message_preprocessing(message)
message_processed = message
sum = 0
for i in message_processed.split(' '):
    if(i in no_hate['word'].values.tolist()):
        count = no_hate[no_hate['word'] == i]['count'].values[0]
        sum += log(count/total_no_hate)*count
p_no_hate_message = log(p_no_hate) + sum

sum = 1
for i in message_processed.split(' '):
    if(i in hate['word'].values.tolist()):
        count = hate[hate['word'] == i]['count'].values[0]
        sum += log(count/total_no_hate)*count
p_hate_message = log(p_hate) + sum

if(p_hate_message < p_no_hate_message):
    print("Es un tweet de odio")
else:
    print("No es un tweet de odio")