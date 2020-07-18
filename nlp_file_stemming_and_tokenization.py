#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 20:34:16 2020

@author: divyansh
"""

import nltk
par = """Once the crucial goal of suffrage had been achieved
 the feminist movement virtually collapsed in both Europe and the United States.
Lacking an ideology beyond the achievement of the vote, feminism
fractured into a dozen splinter groups: the Women’s Joint Congressional
Committee, a lobbying group, fought for legislation to promote
education and maternal and infant health care; the League of Women
Voters organized voter registration and education drives; and
the Women’s Trade Union League launched a campaign for protective
labour legislation for women.
Each of these groups offered some civic contribution, but none was
specifically feminist in nature. Filling the vacuum, the National
Woman’s Party, led by Paul, proposed a new initiative meant to
remove discrimination from American laws and move women closer to
equality through an Equal Rights Amendment (ERA) that would ban any
government-sanctioned discrimination based on sex. Infighting began
because many feminists were not looking for strict equality; they were
fighting for laws that would directly benefit women. Paul, however,
argued that protective legislation—such as laws mandating maximum
eight-hour shifts for female factory workers—actually closed the door of
opportunity on women by imposing costly rules on employers, who
would then be inclined to hire fewer women."""

#print(par)
nltk.download('punkt')


sentences = nltk.sent_tokenize(par)

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

from nltk.corpus import stopwords

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]=' '.join(words)
    

