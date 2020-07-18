#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: divyansh
"""

import nltk
par = """Football game is very useful to all of us if played regularly. It benefits us in many ways. It is an interesting outdoor game played by two teams having 11 players in each. It is a game of good physical exercise which teaches players about harmony, discipline and sportsmanship. It is a popular game all over the world and played for years in various cities and towns of many countries.

Origin of Football Game

Historically, the football game has been 700-800 years old however became the world’s favorite game for more than 100 years. It was brought to the Britain by the Romans. It was first started playing in England in 1863. Football Association was formed in England as the first governing body to govern this sport. Earlier, people were playing it simply by kicking the ball with their foot which later became an interesting game. Slowly, this game got much popularity and started to be played with rules on a rectangular field which marked by the boundary lines and a centre line. It is not an expensive and also called as soccer. The Laws of this game were originally arranged in a systematic code by the Football Association, England in 1863 which is governed internationally by the FIFA. It organizes the FIFA World Cup after every four years.

Rules of Playing Football

Rules of playing the football game are officially called as Laws of the Game. There are almost 17 rules of playing this game under two teams:

It is played in a rectangular field with two long sides (touch lines) and two shorter sides (goal lines).
It is played in a field divided by halfway line.Football must be round in shape (made of leather) with 68-70 cm in circumference and filled with air.
It has two teams of 11 players in each. Once cannot start this game if any team has less than 7 players.
There should be a referee and 2 assistant referees to ensure the Laws of game.
 Assistant Referees.
This game is of 90 minutes duration with 2 halves of 45 minutes each. Interval should not exceed more than 15 minutes.
A ball becomes in play all time however becomes out of play whenever a team has scored a goal or referee has stopped the game.
There is a goal kick to restart the play after a goal is scored.
Conclusion

Football is a most popular game all over the world. It is an inexpensive game, played in almost all the countries with much interest. Players, who practice it regularly, get benefited in many ways. It provides lots of benefits to the physical and mental health.
"""

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
    

