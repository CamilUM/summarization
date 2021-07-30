from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("spanish")) - {"no"}
SPECIAL = {".", ",", ":", ";", "...", "(", ")", "[", "]", "{", "}", "-", "--", "_", "+", "*", "/", "\\", "\"", "'", "¿", "?", "¡", "!", "=", "~", "·", "¬", "&", "%", "$", "€", "º", "ª"}
