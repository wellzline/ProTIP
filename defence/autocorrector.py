# from autocorrect import Speller
# spell = Speller(lang='en')
# text = "Ths is a smple text with speling mistakes."
# corrected_text = ' '.join([spell(word) for word in text.split()])
# print(corrected_text)

def defence_autocorrect(influent_sentence):
    from autocorrect import Speller
    spell = Speller(lang='en')
    text = influent_sentence
    corrected_text = ' '.join([spell(word) for word in text.split()])
    return corrected_text



