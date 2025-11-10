import pandas as pd
import re

def get_seq_phn(a,n):
    '''
    input : a : list of phonemes
            n : maximum number of phonemes in a sequence
    output : list of n-grams of phonemes
    '''
    return list(zip(*[a[i:] for i in range(n)]))


def sampa2ipa(str, path_IPA_SAMPA = 'C:/Users/D-CAP/Documents/GitHub/witching-star/clement/TTS1-fr-FR.csv'):
    '''
    from Clément Sauvage's code
    from a SAMPA phoneme, returns the corresponding IPA phoneme
    input : str : SAMPA phoneme 
    output : str : IPA phoneme
    '''
    df_IPA_SAMPA = pd.read_csv(path_IPA_SAMPA, encoding='utf-16')
    row_indices = df_IPA_SAMPA.index[df_IPA_SAMPA['modified_SAMPA'] == str].tolist()
    if row_indices:
        # If there are matching rows, return the IPA from the first match
        return df_IPA_SAMPA.at[row_indices[0], 'modified_IPA'], False  # second output is here to test if the str ph is absent from df_IPA_SAMPA
    else:
        # Handle the case where no match was found
        #print("No matching found, probably because the actual phoneme was an english one.\n Is it ? : ", str)
        return str, True  # second output is here to test if the str ph is absent from df_IPA_SAMPA

def partition_list(lst, parts):
    # Helper function to generate partitions
    def generate_partitions(lst, parts):
        if parts == 1:
            yield [lst]
        else:
            for i in range(1, len(lst) - parts + 2):
                first_part = lst[:i]
                for remaining in generate_partitions(lst[i:], parts - 1):
                    yield [first_part] + remaining

    return list(generate_partitions(lst, parts))
    # Filter partitions to keep only those with unique lengths of parts
    #def filter_partitions(partitions):
    #    for partition in partitions:
    #        if len(set(map(len, partition))) == parts:  # Check if all parts have different lengths
    #            yield partition

    #return list(filter_partitions(generate_partitions(lst, parts)))

def get_alphabet(text, regex_val = '[^a-zA-Z]'):
    regex = re.compile(regex_val)
    #First parameter is the replacement, second parameter is your input string
    alphabet = regex.sub('', text)
    return alphabet

def remove_duplicates(s):
    return ''.join(dict.fromkeys(s))