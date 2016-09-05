# -*- coding: utf-8 -*-
import re
import codecs

reader = codecs.open('corpus.txt', 'r', 'utf8')
writer = codecs.open('res.txt', 'w', 'utf8')

date_pattern = []
phone_pattern = []
link_pattern = []
currency_pattern = []
emoticon_pattern = []

date_string = ' date '
phone_string = ' phone '
link_string = ' link '
currency_string = ' currency '
emoticon_string = ' emoticon '


date_pattern.append(ur'\d{1,2}/\d{1,2}/\d{2,4}')
date_pattern.append(ur'\d{1,2}-\d{1,2}-\d{2,4}')
date_pattern.append(ur'\+\d{1,2}:\d{1,4}')
date_pattern.append(ur'\d{1,2}/\d{1,4}')
date_pattern.append(ur'\d{1,2}-\d{1,4}')
date_pattern.append(ur'\d{1,2}h\d{0,2}')

phone_pattern.append(ur'\+\d{10,12}')
phone_pattern.append(ur'\d{3,5}\.\d{3,4}\.\d{3,5}')
phone_pattern.append(ur'\d{8,12}')
phone_pattern.append(ur'1800\d{4}')
phone_pattern.append(ur'195')
phone_pattern.append(ur'900')
phone_pattern.append(ur'1342')
phone_pattern.append(ur'191')
phone_pattern.append(ur'888')
phone_pattern.append(ur'333')
phone_pattern.append(ur'1414')
phone_pattern.append(ur'1576')
phone_pattern.append(ur'8170')
phone_pattern.append(ur'9123')
phone_pattern.append(ur'9118')
phone_pattern.append(ur'266')
phone_pattern.append(ur'153')
phone_pattern.append(ur'199')
phone_pattern.append(ur'9029')
phone_pattern.append(ur'8049')
phone_pattern.append(ur'1560')
phone_pattern.append(ur'9191')

link_pattern.append(ur'www\..*')
link_pattern.append(ur'http://.*')

currency_pattern.append(ur'[0-9|\,\.]{3,}VND')
currency_pattern.append(ur'[0-9|\.]{3,}VND')
currency_pattern.append(ur'[0-9|\.]{3,}d')
currency_pattern.append(ur'[0-9|\.]{3,}đ')
currency_pattern.append(ur'[0-9|\.]{3,}tr')
currency_pattern.append(ur'[0-9|\.]{3,}Tr')
currency_pattern.append(ur'[0-9|\.]{3,}TR')

emoticon_pattern.append(ur'o.O')
emoticon_pattern.append(ur'O.o')
emoticon_pattern.append(ur'\(y\)')
emoticon_pattern.append(ur'\(Y\)')
emoticon_pattern.append(ur':v')
emoticon_pattern.append(ur':V')
emoticon_pattern.append(ur':3')
emoticon_pattern.append(ur'-_-')
emoticon_pattern.append(ur'\^_\^')
emoticon_pattern.append(ur'<3')
emoticon_pattern.append(ur':-\*')
emoticon_pattern.append(ur':\*')
emoticon_pattern.append(ur":'\(")
emoticon_pattern.append(ur':p ')
emoticon_pattern.append(ur':P')
emoticon_pattern.append(ur':d')
emoticon_pattern.append(ur':D')
emoticon_pattern.append(ur':-\?')
emoticon_pattern.append(ur'>\.<')
emoticon_pattern.append(ur'><')
emoticon_pattern.append(ur':-\w ')
emoticon_pattern.append(ur':\)\)')
emoticon_pattern.append(ur';\)\)')
emoticon_pattern.append(ur'=\)\)')
emoticon_pattern.append(ur':-\)')
emoticon_pattern.append(ur':\)')
emoticon_pattern.append(ur':\]')
emoticon_pattern.append(ur'=\)')
emoticon_pattern.append(ur':-\(')
emoticon_pattern.append(ur':\(')
emoticon_pattern.append(ur':\[')
emoticon_pattern.append(ur'=\(')

# for ex in regex_dict:
#     print ex, ' -> ', regex_dict[ex]
for line in reader.readlines():
    line = line.replace('\n', '')
    temp = line.split('\t')
    label = temp[0]
    line = temp[1]
    #print line
    sentence = ''
    for word in line.split(' '):
        for date_pat in date_pattern:
            word = re.sub(date_pat, date_string, word)
        for currency_pat in currency_pattern:
            word = re.sub(currency_pat, currency_string, word)
        for phone_pat in phone_pattern:
            word = re.sub(phone_pat, phone_string, word)
        for link_pat in link_pattern:
            word = re.sub(link_pat, link_string, word)
        for emoticon_pat in emoticon_pattern:
            word = re.sub(emoticon_pat, emoticon_string, word)
        # word = re.sub(r'.*/.*/.*', '__date__', word)
        sentence += word + ' '
    #print sentence.strip()
    writer.write(label + '\t' + sentence.strip() + '\n')
reader.close()
writer.close()

f1 = codecs.open('res.txt', 'r', 'utf-8')
f2 = codecs.open('corpus_train.txt', 'w', 'utf-8')
list_label = []
list_content = []
for line in f1:
    line = line.split('\t')
    list_label.append(line[0])
    temp = line[1]
    stop_list = ['.', ',', '/', '?', ';', ':', '&', '@', '!', '`', "'", '"', '>', '<', '*', '%', '#', '(', ')', '[',
                 ']', '-', '_', '=', '+', '{', '}', '~', '$', '^', '*', '|', '\\']
    for item in stop_list:
        temp = temp.replace(item, ' ')
    list_content.append(temp.strip('\n').lower())
count = 0
for line in list_content:
    sentence = ''
    line = line.split()
    for word in line:
        if word.isdigit():
            word = ' number '
        sentence += word + ' '
    f2.write(list_label[count] + '\t' + sentence.strip() + '\n')
    count += 1
f1.close()
f2.close()
