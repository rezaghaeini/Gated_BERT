# -*- coding: utf-8 -*-
import json
import argparse
import numpy as np
import time, sys, os
from random import randrange
from nltk.corpus import wordnet
from .demo_model_bridge import Bridge
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, disconnect

# -----------------------------------------------------------------------------------
# nltk.download('wordnet')
# -----------------------------------------------------------------------------------

ngram_distribution = None
mask_spec_chars = False
head_count, layer_count = 0, 0
args, model_bridge = None, None
model_width = 1210
model_height = 5280

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)
model_type = 'player_norm'

# -----------------------------------------------------------------------------------

def extract_ngrams():
    def add_to_dict(key_str, _dict):
        if key_str in _dict:
            _dict[key_str] += 1
        else:
            _dict[key_str] = 1
        return _dict
    def add_ngram(key_cat, token, _dict):
        if key_cat in _dict:
            _dict[key_cat] = add_to_dict(token, _dict[key_cat])
        else:
            _dict[key_cat] = add_to_dict(token, {})
        return _dict

    ngram_dict = {'2gram': {}, '3gram': {}, 'vocab': {}}
    for line in open(args.ngram_source):
        tokens = line.rstrip().lower().split(" ")
        if len(tokens) == 1:
            if ('-' not in tokens[0]) and ('_' not in tokens[0]):
                ngram_dict['vocab'] = add_to_dict(tokens[0], ngram_dict['vocab'])
            ngram_dict['2gram'] = add_ngram("b_", tokens[0], ngram_dict['2gram'])
            ngram_dict['2gram'] = add_ngram("a_", tokens[0], ngram_dict['2gram'])
            ngram_dict['3gram'] = add_ngram("_", tokens[0], ngram_dict['3gram'])
        else:
            for i in range(len(tokens)):
                if ('-' not in tokens[i]) and ('_' not in tokens[i]):
                    ngram_dict['vocab'] = add_to_dict(tokens[i], ngram_dict['vocab'])
                if i == 0:
                    ngram_dict['2gram'] = add_ngram("a_", tokens[0], ngram_dict['2gram'])
                    ngram_dict['2gram'] = add_ngram("b_%s"%tokens[1], tokens[0], ngram_dict['2gram'])
                    ngram_dict['3gram'] = add_ngram("_%s"%tokens[1], tokens[0], ngram_dict['3gram'])
                elif i == len(tokens)-1:
                    ngram_dict['2gram'] = add_ngram("a_%s"%tokens[i-1], tokens[i], ngram_dict['2gram'])
                    ngram_dict['2gram'] = add_ngram("b_", tokens[i], ngram_dict['2gram'])
                    ngram_dict['3gram'] = add_ngram("%s_"%tokens[i-1], tokens[i], ngram_dict['3gram'])
                else:
                    ngram_dict['2gram'] = add_ngram("a_%s"%tokens[i-1], tokens[i], ngram_dict['2gram'])
                    ngram_dict['2gram'] = add_ngram("b_%s"%tokens[i+1], tokens[i], ngram_dict['2gram'])
                    ngram_dict['3gram'] = add_ngram("%s_%s"%(tokens[i-1], tokens[i+1]), tokens[i], ngram_dict['3gram'])

    with open(args.ngram_distribution ,'w') as f:
        json.dump(ngram_dict, f)

# -----------------------------------------------------------------------------------

def mask_special_chars(data, idx, mask=False):
    if mask:
        if data.min() < 0:
            data = data - data.min()
        if len(data.shape) == 1:
            for i in idx:
                data[i] = 0
        elif len(data.shape) == 2:
            for i in idx:
                data[i,:] = 0
                data[:,i] = 0
        return data
    else:
        return data

def normalization(v, ax=None, zero_one=True, doAbs=False):
    if v is None:
        raise RuntimeError("array is None!")
    v = np.array(v, dtype='float32')
    assert len(v.shape) <= 2
    if doAbs:
        v = np.abs(v)
    max_value = np.max(v, axis=ax)
    min_value = np.min(v, axis=ax)
    if ax is None:
        det = (max_value - min_value)
        det = det if det > 0 else 1
        v = (v - min_value) / det
    elif ax == 1 or ax == -1:
        det = (max_value - min_value)[:,None]
        det = np.where(det==0, 1, det)
        v = (v - min_value[:,None]) / det
    else:
        det = (max_value - min_value)[None,:]
        det = np.where(det==0, 1, det)
        v = (v - min_value[None,:]) / det
    if not zero_one:
        v = (2 * v) - 1
    return v.flatten().tolist()

# -----------------------------------------------------------------------------------
# General run of the model

def interpretation_extraction(inp1, inp2, pairwise, task, user, mask_special=None):
    if mask_special is None:
        mask_special = mask_spec_chars
    data_list =  [inp1, inp2] if pairwise else [inp1]
    data_batch, input_text = model_bridge.parse(data_list, task)
    special_idx = [0]
    for i in range(len(input_text)):
        if input_text[i] == '[SEP]':
            special_idx.append(i)
    model_info = model_bridge._demo_run(task, data_batch, user)

    max_word_len = (max([len(w) for w in input_text]))
    json_dict = {
        "head_names": ["Head %d"%i for i in range(head_count)],
        "layer_names": ["Layer %d"%i for i in range(layer_count)],
        "head_count": head_count,
        "y_margin": 9*max_word_len,
        "x_margin": int(5.5*max_word_len),
        "len": len(input_text),
        "x": ["%d_%s"%(i, input_text[i]) for i in range(len(input_text))],
        "classes": model_bridge.get_class_names(task),
        "logit": normalization(model_info['logit']),
        "prediction": model_bridge.get_prediction_string(task, model_info["prediction"]),
        "layers": [],
        "layers_impact_W": normalization(model_info['layer_weight_impact']['w']),
        "layers_impact_G": normalization(model_info['layer_weight_impact']['g']),
        "layers_impact_T": normalization(np.multiply(model_info['layer_weight_impact']['w'],
                                                     model_info['layer_weight_impact']['g'])),
        "embedding_W_main": normalization(mask_special_chars(
            np.abs(model_info['embedding']['w']).sum(axis=-1), special_idx, mask_special)),
        "embedding_G_main": normalization(mask_special_chars(
            np.abs(model_info['embedding']['g']).sum(axis=-1), special_idx, mask_special)),
        "embedding_T_main": normalization(mask_special_chars(
            np.abs(np.multiply(model_info['embedding']['w'],
                        model_info['embedding']['g'])).sum(axis=-1), special_idx, mask_special)),
        "sub_embedding_WG": []
    }
    if user == "Developer":
        for i in range(layer_count):
            layer_dict = {
                            "idx": i,
                            "W_output": normalization(mask_special_chars(
                                np.abs(model_info['attetion_layer_%d'%i]['output']['w']).sum(axis=-1), special_idx, mask_special)),
                            "G_output": normalization(mask_special_chars(
                                np.abs(model_info['attetion_layer_%d'%i]['output']['g']).sum(axis=-1), special_idx, mask_special)),
                            "T_output": normalization(mask_special_chars(
                                np.abs(np.multiply(model_info['attetion_layer_%d'%i]['output']['w'],
                                            model_info['attetion_layer_%d'%i]['output']['g'])).sum(axis=-1), special_idx, mask_special)),
                            "W_Head": [],
                            "G_Head": [],
                            "T_Head": []
                         }
            W_impact, G_impact, T_impact = [], [], []
            for j in range(head_count):
                layer_dict["W_Head"].append(normalization(mask_special_chars(
                    np.abs(model_info['attetion_layer_%d'%i]['head_%d_probs'%j]['w']), special_idx, mask_special)))
                W_impact.append(mask_special_chars(model_info['attetion_layer_%d'%i]['head_%d_probs'%j]['w'], special_idx, mask_special).sum())
                layer_dict["G_Head"].append(normalization(mask_special_chars(
                    np.abs(model_info['attetion_layer_%d'%i]['head_%d_probs'%j]['g']), special_idx, mask_special)))
                G_impact.append(mask_special_chars(model_info['attetion_layer_%d'%i]['head_%d_probs'%j]['g'], special_idx, mask_special).sum())
                layer_dict["T_Head"].append(normalization(mask_special_chars(
                    np.abs(np.multiply(model_info['attetion_layer_%d'%i]['head_%d_probs'%j]['w'],
                                model_info['attetion_layer_%d'%i]['head_%d_probs'%j]['g'])), special_idx, mask_special)))
                T_impact.append(mask_special_chars(
                    np.abs(np.multiply(model_info['attetion_layer_%d'%i]['head_%d_probs'%j]['w'],
                                model_info['attetion_layer_%d'%i]['head_%d_probs'%j]['g'])), special_idx, mask_special).sum())
            layer_dict["W_impact"] = normalization(np.array(W_impact))
            layer_dict["G_impact"] = normalization(np.array(G_impact))
            layer_dict["T_impact"] = normalization(np.array(T_impact))
            json_dict["layers"].append(layer_dict)

        json_dict["sub_embedding_WG"] = [
                                    {
                                        "name": "Word",
                                        "W": normalization(mask_special_chars(
                                            np.abs(model_info['words_embedding']['w']).sum(axis=-1), special_idx, mask_special)),
                                        "G": normalization(mask_special_chars(
                                            np.abs(model_info['words_embedding']['g']).sum(axis=-1), special_idx, mask_special)),
                                        "T": normalization(mask_special_chars(
                                            np.abs(np.multiply(model_info['words_embedding']['w'],
                                                        model_info['words_embedding']['g'])).sum(axis=-1), special_idx, mask_special))
                                    },
                                    {
                                        "name": "Position",
                                        "W": normalization(mask_special_chars(
                                            np.abs(model_info['position_embedding']['w']).sum(axis=-1), special_idx, mask_special)),
                                        "G": normalization(mask_special_chars(
                                            np.abs(model_info['position_embedding']['g']).sum(axis=-1), special_idx, mask_special)),
                                        "T": normalization(mask_special_chars(
                                            np.abs(np.multiply(model_info['position_embedding']['w'],
                                                        model_info['position_embedding']['g'])).sum(axis=-1), special_idx, mask_special))
                                    },
                                    {
                                        "name": "Type",
                                        "W": normalization(mask_special_chars(
                                            np.abs(model_info['token_type_embedding']['w']).sum(axis=-1), special_idx, mask_special)),
                                        "G": normalization(mask_special_chars(
                                            np.abs(model_info['token_type_embedding']['g']).sum(axis=-1), special_idx, mask_special)),
                                        "T": normalization(mask_special_chars(
                                            np.abs(np.multiply(model_info['token_type_embedding']['w'],
                                                        model_info['token_type_embedding']['g'])).sum(axis=-1), special_idx, mask_special))
                                    }
                                   ]
    return json_dict

# -----------------------------------------------------------------------------------
# Automatic word modifications

def wordnet_token(org_input, idx):
    synonyms = []
    for syn in wordnet.synsets(org_input[idx].lower()):
        for l in syn.lemmas():
            w = l.name()
            if w not in synonyms and (w.lower()!=org_input[idx].lower()) and (len(w.split(' ')) == 1) and (len(w.split('_')) == 1) and (len(w.split('-')) == 1):
                synonyms.append(w.lower())
    if len(synonyms) == 0:
        return org_input[idx]
    random_value = randrange(len(synonyms))
    return synonyms[random_value]

def sampling_token(org_input, idx):
    global ngram_distribution
    if ngram_distribution is None:
        with open(args.ngram_distribution ,'r') as f:
            ngram_distribution = json.load(f)
    previous_word, next_word = '', ''
    if (idx > 0) and (org_input[idx-1] != '[CLS]') and (org_input[idx-1] != '[SEP]'):
        previous_word = org_input[idx-1]
    if (idx < len(org_input)) and (org_input[idx+1] != '[CLS]') and (org_input[idx+1] != '[SEP]'):
        next_word = org_input[idx+1]
    candidate_list = {}
    if "a_%s"%previous_word in ngram_distribution['2gram']:
        candidate_list = ngram_distribution['2gram']["a_%s"%previous_word]
        if "b_%s"%next_word in ngram_distribution['2gram']:
            tmp = ngram_distribution['2gram']["b_%s"%next_word]
            for k in tmp.keys():
                if k in candidate_list:
                    candidate_list[k] += tmp[k]
                else:
                    candidate_list[k] = tmp[k]
        if "%s_%s"%(previous_word,next_word) in ngram_distribution:
             tmp = ngram_distribution['3gram']["%s_%s"%(previous_word,next_word)]
             for k in tmp.keys():
                 if k in candidate_list:
                     candidate_list[k] += tmp[k]
                 else:
                     candidate_list[k] = tmp[k]
    elif "b_%s"%next_word in ngram_distribution['2gram']:
        candidate_list = ngram_distribution['2gram']["b_%s"%next_word]
    if org_input[idx] in candidate_list:
        del candidate_list[org_input[idx]]
    elim_list = []
    for k in candidate_list.keys():
        if ('-' in k) or ('_' in k):
            elim_list.append(k)
    for k in elim_list:
        del candidate_list[k]
    if len(candidate_list) > 0:
        freq_sum = 0
        for k in candidate_list.keys():
            freq_sum += candidate_list[k]
        random_value = randrange(freq_sum) + 1
        counter = 0
        for k in candidate_list.keys():
            counter += candidate_list[k]
            if counter >= random_value:
                return k
    else:
        freq_sum = 0
        for k in ngram_distribution['vocab'].keys():
            freq_sum += ngram_distribution['vocab'][k]
        random_value = randrange(freq_sum) + 1
        counter = 0
        for k in ngram_distribution['vocab'].keys():
            counter += ngram_distribution['vocab'][k]
            if counter >= random_value:
                return k

def _word_modification(method, org_input, idx):
    if method == 'Remove':
        return '[REMOVED]'
    elif method == 'Zero Out':
        return '[ZERO]'
    elif method == 'Unknown':
        return '[UNK]'
    elif method == 'Wordnet':
        return wordnet_token(org_input, idx)
    elif method == 'Sampling':
        return sampling_token(org_input, idx)
    else:
        raise RuntimeError('The modification method is not defined.')
    return org_input[idx]

def word_modification_process(inp1, inp2, pairwise, task, method=None, modif_inp1=None, modif_inp2=None):
    data_list =  [inp1, inp2] if pairwise else [inp1]
    word_modification = True if (method is not None) and (method in ['Remove', 'Zero Out', 'Unknown']) else False
    modif_data_list = None
    if word_modification:
        modif_data_list = [modif_inp1, modif_inp2] if pairwise else [modif_inp1]
    elif method is not None:
        data_list = [modif_inp1, modif_inp2] if pairwise else [modif_inp1]
    data_batch, input_text = model_bridge.parse(data_list, task, word_modification, modif_data_list, word_analyses=True)
    if word_modification:
        input_text = ['[CLS]'] + ((modif_inp1.split(' ') + ['[SEP]'] + modif_inp2.split(' ')) if pairwise else modif_inp1.split(' ')) + ['[SEP]']
    model_info = model_bridge._demo_word_change_run(task, data_batch)
    return input_text, model_info['prediction'], normalization(model_info['logit'])

# -----------------------------------------------------------------------------------
# Structure modification

def structure_modification_process(inp1, inp2, pairwise, task, head_mask=None, layer_mask=None):
    data_list =  [inp1, inp2] if pairwise else [inp1]
    data_batch, input_text = model_bridge.parse(data_list, task)
    model_info = model_bridge._demo_structure_change_run(task, data_batch, head_mask=head_mask, layer_mask=layer_mask)
    return input_text, model_info['prediction'], normalization(model_info['logit'])

# -----------------------------------------------------------------------------------
# Main Templates

@app.route("/")
def index():
    info = {
        "task_set": model_bridge.task_list,
        "task_pair": model_bridge.task_pair_list,
        "task_count": len(model_bridge.task_list),
        "selected_task_id": 0,
        "selected_user": "Developer",
        "input01": "",
        "input02": ""
    }
    return render_template("async_demo.html", info=info)

@app.route("/", methods=['POST'])
def my_from_post():
    inp1 = request.form['input01'].lower()
    inp2 = request.form['input02'].lower()
    task = request.form['taskcombo']
    user = request.form['usercombo']
    pairwise = (model_bridge.task_pair_list[model_bridge.task_list.index(task)] == "1")
    if request.form['submit'] == 'Submit':
        json_dict = interpretation_extraction(inp1, inp2, pairwise, task, user)
        info = {
            "task_set": model_bridge.task_list,
            "task_pair": model_bridge.task_pair_list,
            "task_count": len(model_bridge.task_list),
            "selected_task_id": model_bridge.task_list.index(task),
            "selected_user": user,
            "input01": inp1,
            "input02": inp2,
            "prediction": json_dict["prediction"],
            "head_count": head_count if len(json_dict["layers"]) > 0 else 0,
            "layer_idx": range(len(json_dict["layers"])-1, -1, -1),
            "sub_embedding_WG": ["Word", "Position", "Type"] if len(json_dict["sub_embedding_WG"]) > 0 else [],
            "json": json_dict
        }
        return render_template("async_lazy_response_d3.html", info=info)
    elif request.form['submit'] == 'Word Analyses':
        token_list, prediction, logit = word_modification_process(inp1, inp2, pairwise, task)
        info = {
            "task": task,
            "pairwise": pairwise,
            "token_list": token_list,
            "token_list_len": len(token_list),
            "token_list_cat": ['static' if (x == '[CLS]' or x == '[SEP]') else 'multi' for x in token_list],
            "original_input": ' '.join(token_list),
            "input01": inp1,
            "input02": inp2,
            "classes": model_bridge.get_class_names(task),
            "org_prediction": model_bridge.get_prediction_string(task, prediction),
            "org_logit_vector": logit
        }
        return render_template("async_word_analyze_d3.html", info=info)
    elif request.form['submit'] == 'Layer and Attention Head Analyses':
        token_list, prediction, logit = structure_modification_process(inp1, inp2, pairwise, task)
        info = {
            "model_width": model_width,
            "model_height": model_height,
            "task": task,
            "classes": model_bridge.get_class_names(task),
            "original_input": ' '.join(token_list),
            "org_prediction": model_bridge.get_prediction_string(task, prediction),
            "org_logit_vector": logit
        }
        return render_template("async_structure_analyze_d3.html", info=info)

# -----------------------------------------------------------------------------------
# Word Analyses Template

@socketio.on('change_modification_type', namespace='/word_analyze')
def change_modification_type_message(message):
    task = message['task']
    method = message['type']
    org_input = message['org_input'].split(' ')
    cur_input = message['cur_input'].split(' ')
    pairwise = (model_bridge.task_pair_list[model_bridge.task_list.index(task)] == "1")
    inps, modif_inps, idx = [[], []], [[], []], 0
    for i, [ow, cw] in enumerate(zip(org_input, cur_input)):
        if ow != '[CLS]' and ow != '[SEP]':
            if ow == cw:
                inps[idx].append(ow)
                modif_inps[idx].append(ow)
            else:
                inps[idx].append(ow)
                modif_inps[idx].append(_word_modification(method, org_input, i))
        elif ow == '[SEP]':
            idx += 1
    token_list, prediction, logit = word_modification_process(' '.join(inps[0]), ' '.join(inps[1]), pairwise,
                                        task, method, ' '.join(modif_inps[0]), ' '.join(modif_inps[1]))
    response = {'text': ' '.join(token_list),
                'prediction': model_bridge.get_prediction_string(task, prediction),
                'logit': logit}
    emit('auto_response', response)

@socketio.on('change_words', namespace='/word_analyze')
def change_words_message(message):
    task = message['task']
    method = message['type']
    word_idx = int(message['word_idx'].split('_')[1])
    org_input = message['org_input'].split(' ')
    cur_input = message['cur_input'].split(' ')
    pairwise = (model_bridge.task_pair_list[model_bridge.task_list.index(task)] == "1")
    inps, modif_inps, idx = [[], []], [[], []], 0
    for i, [ow, cw] in enumerate(zip(org_input, cur_input)):
        if cw != '[CLS]' and cw != '[SEP]':
            if i != word_idx:
                inps[idx].append(ow)
                modif_inps[idx].append(cw)
            else:
                inps[idx].append(ow)
                if ow == cw:
                    modif_inps[idx].append(_word_modification(method, org_input, i))
                else:
                    modif_inps[idx].append(ow)
        elif cw == '[SEP]':
            idx += 1
    token_list, prediction, logit = word_modification_process(' '.join(inps[0]), ' '.join(inps[1]), pairwise,
                                        task, method, ' '.join(modif_inps[0]), ' '.join(modif_inps[1]))
    response = {'text': ' '.join(token_list),
                'prediction': model_bridge.get_prediction_string(task, prediction),
                'logit': logit}
    emit('auto_response', response)

@socketio.on('new_input', namespace='/word_analyze')
def new_input_message(message):
    task = message['task']
    pairwise = (model_bridge.task_pair_list[model_bridge.task_list.index(task)] == "1")
    inp1 = message['input01'].lower()
    inp2 = message['input02'].lower() if pairwise else ""
    _, prediction, logit = word_modification_process(inp1, inp2, pairwise, task)
    response = {'prediction': model_bridge.get_prediction_string(task, prediction),
                'logit': logit}
    emit('manual_response', response)

# -----------------------------------------------------------------------------------
# Structure Analyses Template

@socketio.on('connect', namespace='/structure_analyze')
def structure_analyze_connect():
    graph = model_bridge.get_model_graph()
    emit('connect_response', graph)

@socketio.on('change_structure', namespace='/structure_analyze')
def structure_change_message(message):
    task = message['task']
    _input = message['input'].split(' ')
    pairwise = (model_bridge.task_pair_list[model_bridge.task_list.index(task)] == "1")
    inps, idx = [[], []], 0
    if pairwise:
        for w in _input:
            if w != '[CLS]' and w != '[SEP]':
                inps[idx].append(w)
            elif w == '[SEP]':
                idx += 1
    else:
        inps[0] = _input[1:-1]
    head_status = message['head_status']
    head_mask = [[1]*head_count]*layer_count
    active_heads = True
    for i in range(layer_count):
        for j in range(head_count):
            if ("Layer_%d_Head_%d"%(i,j) in head_status) and (head_status["Layer_%d_Head_%d"%(i,j)] == 0):
                head_mask[i][j] = 0
                active_heads = False
    if active_heads:
        head_mask = None
    layer_status = message['layer_status']
    layer_mask = []
    active_layers = True
    for i in range(layer_count):
        if ("Layer_%d_Collector"%(i) in layer_status) and (layer_status["Layer_%d_Collector"%(i)] == 0):
            layer_mask.append(0.)
            active_layers = False
        else:
            layer_mask.append(1.)
    if active_layers:
        layer_mask = None
    _, prediction, logit = structure_modification_process(' '.join(inps[0]), ' '.join(inps[1]), pairwise, task, head_mask, layer_mask)
    response = {'prediction': model_bridge.get_prediction_string(task, prediction),
                'logit': logit}
    emit('change_response', response)

# -----------------------------------------------------------------------------------

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def main(inp_args):
    global model_bridge, args, head_count, layer_count
    args = inp_args
    if args.ngram_extraction:
        extract_ngrams()
    else:
        if args.model_type != "":
            model_type = args.model_type
        model_bridge = Bridge(model_type)
        head_count = model_bridge.head_count
        layer_count = model_bridge.layer_count
        socketio.run(app, host=args.ip, port=args.port)
