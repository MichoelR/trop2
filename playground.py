import unicodedata

unicodedata.name('֖')
def unicode_name(char):
    name = " ".join(unicodedata.name(char).split()[2:])
    return name

