from spellchecker import SpellChecker
import pandas as pd

spell = SpellChecker()

str1 = "A sunny beach with a pXalm tree and a surgfboard lying in the sand"
str1 = "A serene lake surrounded by mountains with a small boat and qducks."
str1 = "A man with a red Xhelmet on a small moped on a dirt road."
str1 = "A man with a red helmet on a small moped on a dPirt road."
str2 = str1.split()
misspelled = spell.unknown(str1.split())  # {'qducks.'}

corrected_sentence = " ".join(spell.correction(word) if word in misspelled else word for word in str2)

print(corrected_sentence)


