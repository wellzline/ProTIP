from spellchecker import SpellChecker
import pandas as pd

spell = SpellChecker()

str1 = "A sunny beach with a pXalm tree and a surgfboard lying in the sand"
str2 = str1.split()
misspelled = spell.unknown(str1.split())  

corrected_sentence = " ".join(spell.correction(word) if word in misspelled else word for word in str2)

print(corrected_sentence)


