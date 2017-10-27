from support_functions import *
stop_word_file = 'stop_words.txt'
stop_words = read_words_in_file(stop_word_file)
stop_words_unique = remove_duplicate(words)
write_to_file(stop_words_unique, stop_word_file)