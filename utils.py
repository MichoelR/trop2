import unicodedata

# list of unicode trope symbols, and corresponding names (minus "HEBREW ACCENT")
# includes SOF PASUQ (1473), meaning the :, not the meteg inside the word
trops = [chr(x) for x in range(1425, 1455)] + [chr(1475)]
trop_names = {x: " ".join(unicodedata.name(x).split(" ")[2:]) for x in trops}


