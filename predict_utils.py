import torch
import pandas as pd
import numpy as np
from transformers import CamembertForMaskedLM, CamembertTokenizer, CamembertForCausalLM, AutoTokenizer, AutoConfig
# Custom import
import predict_utils as predict

tokenizer = CamembertTokenizer.from_pretrained("camembert-base", truncation_side='left')
camembert = CamembertForMaskedLM.from_pretrained("camembert-base")
camembert.eval()
config = AutoConfig.from_pretrained("almanach/camembert-base")
config.is_decoder = True


bad_index_file = 'C:/Users/D-CAP/Documents/GitHub/witching-star/lexique_dataframe/index_foireux.pkl'
bad_index = pd.read_pickle(bad_index_file)
bad_index.remove(7)
bad_index.remove(9)

def predict_from_context(context_words, model, tokenizer, mask = False,handle_bad_index = True):
    """
    Predict the next token in a sequence using a mask-based language model.
    :param context: Input sequence, word with mask token
    :param model: Language model
    :param tokenizer: Tokenizer
    :param mask: Whether to use a mask token or just predict the next token
    :return: Predicted token ID
    """
    
    inputs = tokenizer(context_words, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).logits
    
    if mask:
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_logits = output[0, mask_token_index][0]
        output = output[0, mask_token_index, :][0]
    else:
        predicted_token_logits = output[0, -1]
        predicted_token_logits[2] = -float('inf')
        output = output[0, -1, :]

    if handle_bad_index:
        predicted_token_logits = force_value_index(predicted_token_logits, bad_index, -float('inf'))
    predicted_token_id = predicted_token_logits.argmax()
    most_likely_output = tokenizer.decode(predicted_token_id)

    return output

def predict_word_from_context(context_words, next_word, model, tokenizer, handle_bad_index = True):
    if not '<mask>' in context_words:
        context_words += ' <mask>'
    next_tokenized_word = clean_token(tokenizer.encode(next_word, return_tensors='pt')[0],[5,6])
    proba_word = 1
    new_context = context_words
    for token_id in next_tokenized_word:
        prediction_m = predict_from_context(new_context, model, tokenizer, mask = True)
        prediction_m = normalized_probabilities(prediction_m, handle_bad_index = True)
        predict_token = tokenizer.decode(token_id)
        probability_m = prediction_m[token_id].tolist()
        proba_word *= probability_m
        new_context = new_context.replace('<mask>',predict_token) + '<mask>'
    return proba_word



def clean_token(token_ids, bad_index):
    new_tensor = torch.tensor(list(filter(lambda a: not a in bad_index, token_ids.tolist())))
    return new_tensor

def clean_tensor(tensor, bad_index):
    new_tensor = torch.tensor(list(filter(lambda a: not a in bad_index, tensor.tolist())))
    return new_tensor

def force_value_index(tensor, index_list, value):
    for index in index_list:
        tensor[index] = value
    return tensor

def normalized_probabilities(logit, handle_bad_index = True):
    proba = torch.softmax(logit, dim=-1)
    if handle_bad_index:
        proba = force_value_index(proba, bad_index, 0)
    return proba/torch.sum(proba)

def get_probability_token(norm_proba, token_id, logit = True):
    if logit:
        return torch.softmax(norm_proba, dim=-1)[token_id].item()
    else:
        return norm_proba[token_id].item()


def predict_mask(context_words):
    inputs = tokenizer(context_words, return_tensors="pt")
    inputs['attention_mask'] = inputs['attention_mask'][:,-510:]
    inputs['input_ids'] = inputs['input_ids'][:,-510:]
    with torch.no_grad():
        outputs = camembert(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_logits = outputs[0, mask_token_index][0]
    output = outputs[0, mask_token_index, :][0]
    output = output.numpy()
    return output

def advance_mask(context, token, connector = ' '):
    mask_position = context.index('<mask>')
    new_context = context[:mask_position] + token + connector +'<mask>'
    return new_context

def get_phn_token(next_token, t2p):
    a = list(t2p.T.index)
    possible_tokens = []
    for i in range(len(a)):
        if next_token in a[i]:
            possible_tokens.append(a[i])
    token_length = np.inf
    best_token = ''
    for token in possible_tokens:
        if len(token) < token_length:
            token_length = len(token)
            best_token = token
    list_phn = t2p[best_token]
    return list_phn

def get_syl_token(next_token, t2s):
    a = list(t2s.T.index)
    possible_tokens = []
    for i in range(len(a)):
        if next_token in a[i]:
            possible_tokens.append(a[i])
    token_length = np.inf
    best_token = ''
    for token in possible_tokens:
        if len(token) < token_length:
            token_length = len(token)
            best_token = token
    list_syl = t2s[best_token]
    return list_syl

def phonemes_possibilities(list_phn,t2p, all_phn):
    encode_position_dict = dict()
    list_prior_tokens = list(t2p.columns)
    for position,phn in enumerate(list_phn):
        if not pd.isnull(phn):
            encode_position_dict[position] = {'true': {'possibilities':[], 'phoneme':phn}, 'all': {'possibilities':{key: [] for key in all_phn}, 'phoneme':all_phn}}
            list_possible_tokens = []
            for token in list_prior_tokens:
                observed_phn = t2p[token][position]
                if not pd.isnull(observed_phn):
                    encode_position_dict[position]['all']['possibilities'][observed_phn].append(tokenizer.encode(token, return_tensors='pt').numpy()[0][1])
                if observed_phn == phn:
                    list_possible_tokens.append(token)
                    encode_position_dict[position]['true']['possibilities'].append(tokenizer.encode(token, return_tensors='pt').numpy()[0][1])
            list_prior_tokens = list_possible_tokens
    return encode_position_dict, list_prior_tokens

def diphones_possibilities(list_diphone,t2d, all_diphone):
    encode_position_dict = dict()
    list_prior_tokens = list(t2d.columns)
    for position,diphone in enumerate(list_diphone):
        if not pd.isnull(diphone):
            encode_position_dict[position] = {'true': {'possibilities':[], 'diphone':diphone}, 'all': {'possibilities':{key: [] for key in all_diphone}, 'diphone':all_diphone}}
            list_possible_tokens = []
            for token in list_prior_tokens:
                observed_diphone = t2d[token][position]
                if not pd.isnull(observed_diphone):
                    encode_position_dict[position]['all']['possibilities'][observed_diphone].append(tokenizer.encode(token, return_tensors='pt').numpy()[0][1])
                if observed_diphone == diphone:
                    list_possible_tokens.append(token)
                    encode_position_dict[position]['true']['possibilities'].append(tokenizer.encode(token, return_tensors='pt').numpy()[0][1])
            list_prior_tokens = list_possible_tokens
    return encode_position_dict, list_prior_tokens


def syllables_possibilities(list_syl,t2s, all_syl):
    encode_position_dict = dict()
    list_prior_tokens = list(t2s.columns)
    for position,syl in enumerate(list_syl):
        if not pd.isnull(syl):
            encode_position_dict[position] = {'true': {'possibilities':[], 'syllable':syl}, 'all': {'possibilities':{key: [] for key in all_syl}, 'phoneme':all_syl}}
            list_possible_tokens = []
            for token in list_prior_tokens:
                observed_syl = t2s[token][position]
                if not pd.isnull(observed_syl):
                    encode_position_dict[position]['all']['possibilities'][observed_syl].append(tokenizer.encode(token, return_tensors='pt').numpy()[0][1])
                if observed_syl == syl:
                    list_possible_tokens.append(token)
                    encode_position_dict[position]['true']['possibilities'].append(tokenizer.encode(token, return_tensors='pt').numpy()[0][1])
            list_prior_tokens = list_possible_tokens
    return encode_position_dict, list_prior_tokens

def log2normproba(logits):
    unnormalized_probs = np.exp(logits)
    normalized_probs = unnormalized_probs / unnormalized_probs.sum()
    return normalized_probs

def logit2proba(logits):
    proba = torch.softmax(torch.from_numpy(logits), dim = -1)
    return proba.numpy()
    