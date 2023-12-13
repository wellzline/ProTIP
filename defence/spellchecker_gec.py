from spellchecker import SpellChecker
spell = SpellChecker()

str1 = "A sunny beach with a pXalm tree and a surgfboard lying in the sand"
misspelled = spell.unknown(str1.split())

for word in misspelled:
    print(spell.correction(word))
    print(spell.candidates(word))