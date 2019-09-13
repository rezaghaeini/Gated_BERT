import re
import torch
import numpy as np
from run_tasks import (compute_metrics, convert_examples_to_features,
                        output_modes, processors, InputExample, InputFeatures)
from pyencoder import (BertConfig, BertForSequenceClassification, BertForMultiSequenceClassification)
from pytorch_transformers import (WEIGHTS_NAME, BertTokenizer)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

class Bridge(object):

    def __init__(self, model_type):
        self.task_ids = {"MNLI": 0, "QNLI": 1, "MRPC": 2, "CoLA": 3, "RTE": 4, "SST-2": 5}
        self.task_list = ["MNLI", "QNLI", "MRPC", "CoLA", "RTE", "SST-2"]
        self.task_pair_list = ["1", "1", "1", "0", "1", "0"]
        self.head_count = 16
        self.layer_count = 24
        self.model_list = {}
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
        for task in self.task_list:
            model_path = self.get_model_path(task)
            config = config_class.from_pretrained(model_path, finetuning_task=task)
            config.keep_part_grads = True
            tokenizer_address = model_path if model_path.split('/')[-1][:10]!='checkpoint' else '/'.join(model_path.split('/')[:-1])
            tokenizer = tokenizer_class.from_pretrained(tokenizer_address, do_lower_case=True) 
            model = model_class.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=config)
            model.to(self.device)
            self.model_list[task] = [config, tokenizer, model]

    def get_model_path(self, task):
        if self.model_type == 'baseline':
            if task == "MRPC":
                return "./script/demeo/MRPC_baseline/checkpoint-best-acc_and_f1"
            elif task == "RTE":
                return "./script/demo/RTE_baseline/checkpoint-best-acc"
            elif task == "STS-B":
                return "./script/demo/STS-B_baseline/checkpoint-best-pearson"
            elif task == "CoLA":
                return "./script/demo/CoLA_baseline/checkpoint-best-mcc"
            elif task == "SST-2":
                return "./script/demo/SST-2_baseline/checkpoint-best-acc"
            elif task == "QNLI":
                return "./script/demo/QNLI_baseline/checkpoint-best-acc"
            elif task == "MNLI":
                return "./script/demo/MNLI_baseline/checkpoint-best-acc"
        else:
            if task == "MRPC":
                return "./script/demo/MRPC/player_norm/checkpoint-best-acc_and_f1"
            elif task == "RTE":
                return "./script/demo/RTE/player_norm/checkpoint-best-acc"
            elif task == "STS-B":
                return "./script/demo/STS-B/player_norm/checkpoint-best-pearson"
            elif task == "CoLA":
                return "./script/demo/CoLA/player_norm/checkpoint-best-mcc"
            elif task == "SST-2":
                return "./script/demo/SST-2/player_norm/checkpoint-best-acc"
            elif task == "QNLI":
                return "./script/demo/QNLI/player_norm/checkpoint-best-acc"
            elif task == "MNLI":
                return "./script/demo/MNLI/player_norm/checkpoint-best-acc"

    def get_class_names(self, task):
        if task in ["MNLI", "MNLI"]:
            return ["Contradiction", "Entailment", "Neutral"]
        elif task in ["MRPC"]:
            return ["non-Paraphrase", "Paraphrase"]
        elif task == "CoLA":
            return ["Incorrect", "Correct"]
        elif task == "SST-2":
            return ["Negative", "Positive"]
        elif task in ["QNLI", "RTE"]:
            return ["Entailment", "not-Entailment"]

    def get_prediction_string(self, task, prediction):
        class_list = self.get_class_names(task)
        if class_list is None:
            return prediction
        else:
            return class_list[prediction]


    # --------------------- DATA HANDELING ----------------------------------

    def pre_process_modif(self, main_text, modif_text):
        main_text = main_text.split(' ')
        modif_text = modif_text.split(' ')
        assert len(main_text) == len(modif_text)
        for i in range(len(main_text)):
            if modif_text[i] == '[REMOVED]':
                main_text[i] = ''
        text = ' '.join(main_text)
        text = re.sub(r"  +", r" ", text)
        return text

    def parse(self, data_list, task, word_modification=False, modif_data_list=None, word_analyses=False):
        if word_modification:
            data_list[0] = self.pre_process_modif(data_list[0], modif_data_list[0])
            if len(data_list)>1:
                data_list[1] = self.pre_process_modif(data_list[1], modif_data_list[1])
        examples = [InputExample(guid=0, text_a=data_list[0], text_b=(data_list[1] if len(data_list)>1 else None), label="0")]
        tokenizer = self.model_list[task][1]
        token_count = 4 + len(data_list[0].split(' ')) + ((1 + len(data_list[1].split(' '))) if len(data_list)>1 else 0)
        features, tokens = convert_examples_to_features(examples, [], -1,
                                                    tokenizer, "regression",
                                                    cls_token_at_end=False,
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=0,
                                                    pad_on_left=False,
                                                    pad_token_segment_id=0,
                                                    pass_text=True, only_split=word_analyses)
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if word_modification:
            modif_tokens = ['[CLS]'] + ((modif_data_list[0].split(' ') + ['[SEP]'] + modif_data_list[1].split(' ')) if len(modif_data_list)>1 else modif_data_list[0].split(' ')) + ['[SEP]']
            if '[ZERO]' in modif_tokens:
                for i in range(len(modif_tokens)):
                    if modif_tokens[i] == '[ZERO]':
                        all_input_mask[0][i] = 0
                        all_input_ids[0][i] = tokenizer._convert_token_to_id(tokenizer.mask_token)
            elif '[UNK]' in modif_tokens:
                for i in range(len(modif_tokens)):
                    if modif_tokens[i] == '[UNK]':
                        all_input_ids[0][i] = tokenizer._convert_token_to_id(tokenizer.unk_token)

        batch = [all_input_ids, all_input_mask, all_segment_ids]
        return batch, tokens

    # --------------------- RUN MODELS ----------------------------------

    def _demo_run(self, task, batch, user):
        self.model_list[task][2].eval()
        batch = tuple(t.to(self.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         None}
        outputs = self.model_list[task][2](**inputs)
        logits = outputs[0]
        pred = logits.detach().cpu().numpy()[0]
        dy_dl = torch.ones((1,1)).to(self.device)
        tmp_model = self.model_list[task][2].module if hasattr(self.model_list[task][2], 'module') else self.model_list[task][2]
        # compute the gradient of the output respect to desired units and components of the model
        if task != "STS-B":
            tmp_model.prediction_vals[:,pred[0]].backward(dy_dl)
        else:
            tmp_model.prediction_vals.backward(dy_dl)
        info = {'prediction': pred[0],
                'logit': logits.detach().cpu().numpy()[0],
                'embedding': {
                                'w': tmp_model.bert.embeddings.embeddig_list[3].detach().cpu().numpy()[0],
                                'g': tmp_model.bert.embeddings.embeddig_list[3].grad.cpu().numpy()[0]
                             }
               }
        if tmp_model.bert.layer_mask_weight is None:
            info['layer_weight_impact'] = {'w': np.array([0]*(self.layer_count-1)+[1]),
                                           'g': np.array([0]*(self.layer_count-1)+[1])}
        else:
            info['layer_weight_impact'] = {'w': tmp_model.bert.layer_mask_weight.detach().cpu().numpy(),
                                           'g': tmp_model.bert.layer_mask_weight.grad.cpu().numpy()}
        if user == 'Developer':
            info['words_embedding'] = {
                            'w': tmp_model.bert.embeddings.embeddig_list[0].detach().cpu().numpy()[0],
                            'g': tmp_model.bert.embeddings.embeddig_list[0].grad.cpu().numpy()[0]
                         }
            info['position_embedding'] = {
                            'w': tmp_model.bert.embeddings.embeddig_list[1].detach().cpu().numpy()[0],
                            'g': tmp_model.bert.embeddings.embeddig_list[1].grad.cpu().numpy()[0]
                         }
            info['token_type_embedding'] = {
                            'w': tmp_model.bert.embeddings.embeddig_list[2].detach().cpu().numpy()[0],
                            'g': tmp_model.bert.embeddings.embeddig_list[2].grad.cpu().numpy()[0]
                         }
            for _layer in range(self.layer_count):
                info['attetion_layer_%d'%_layer] = {}
                info['attetion_layer_%d'%_layer]['output'] = {
                                                            'w': tmp_model.bert.encoder.layer[_layer].attention.self.context_output.detach().cpu().numpy()[0],
                                                            'g': tmp_model.bert.encoder.layer[_layer].attention.self.context_output.grad.cpu().numpy()[0]
                                                        }
                att_probs = tmp_model.bert.encoder.layer[_layer].attention.self.att_probs.detach().cpu().numpy()[0]
                att_probs_grad = tmp_model.bert.encoder.layer[_layer].attention.self.att_probs.grad.cpu().numpy()[0]
                for _head in range(self.head_count):
                    info['attetion_layer_%d'%_layer]['head_%d_probs'%_head] = {
                                                                'w': att_probs[_head],
                                                                'g': att_probs_grad[_head]
                                                            }
        return info

    def _demo_word_change_run(self, task, batch):
        self.model_list[task][2].eval()
        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         None}
            outputs = self.model_list[task][2](**inputs)
            logits = outputs[0]
            pred = np.squeeze(logits.detach().cpu().numpy())

            info = {'prediction': pred,
                    'logit': logits.detach().cpu().numpy()[0],}

            return info

    def _demo_structure_change_run(self, task, batch, head_mask=None, layer_mask=None):
        self.model_list[task][2].eval()
        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         None,
                      'head_mask':      None if head_mask is None else torch.FloatTensor(head_mask).to(device=self.device),
                      'layer_mask':     None if layer_mask is None else layer_mask}
            outputs = self.model_list[task][2](**inputs)
            logits = outputs[0]
            pred = np.squeeze(logits.detach().cpu().numpy())

            info = {'prediction': pred,
                    'logit': logits.detach().cpu().numpy()[0],}

            return info


    # --------------------- GET MODEL STRUCTURE ----------------------------------

    def get_baseline_model(self):
        def adjust_link_x(x, w):
            return x+(w/2)
        def adjust_link_y(y, h):
            return y+h

        svg_w, svg_h = 10+int(self.head_count*62.5), 30+(self.layer_count+1)*210
        sub_embd_w, embd_w, sub_embd_x_pad = 200, 400, 50
        head_w, layer_w, prd_w, comp_h,  = 50, 400, 400, 30
        y_pad = (svg_h - (comp_h*(2*self.layer_count+3))) / (2*self.layer_count+2)
        x_head_pad = (svg_w - 10 - (head_w*head_count)) / (head_count-1)
        curr_y = svg_h-comp_h
        w_x, p_x, t_x = (svg_w/2)-(3*sub_embd_w/2 + sub_embd_x_pad), (svg_w/2)-(sub_embd_w/2), (svg_w/2)+(sub_embd_w/2 + sub_embd_x_pad)
        source_link_x = adjust_link_x((svg_w/2)-(embd_w/2), embd_w)
        source_link_y = adjust_link_y(curr_y-(y_pad+comp_h), comp_h)
        source_x, source_y = (svg_w/2)-(embd_w/2), curr_y-(y_pad+comp_h)
        graph = {
            'comp_h': comp_h,
            'nodes': [
                {"id": "Word_Embeddings", "text": "Word Embeddings", "group": 2, "x": w_x , "y": curr_y, "w": sub_embd_w},
                {"id": "Position_Embeddings", "text": "Position Embeddings", "group": 2, "x": p_x , "y": curr_y, "w": sub_embd_w},
                {"id": "Type_Embeddings", "text": "Type Embeddings", "group": 2, "x":  t_x, "y": curr_y, "w": sub_embd_w},
                {"id": "Embedding_Collector", "text": "Embeddings", "group": 2, "x":  source_x, "y": source_y, "w": embd_w }
            ],
            'links': [
                {"source": "Word_Embeddings", "target": "Embedding_Collector", "value": 1, "x1": adjust_link_x(w_x, sub_embd_w), "y1": curr_y, "x2": source_link_x, "y2": source_link_y},
                {"source": "Position_Embeddings", "target": "Embedding_Collector", "value": 1, "x1": adjust_link_x(p_x, sub_embd_w), "y1": curr_y, "x2": source_link_x, "y2": source_link_y},
                {"source": "Type_Embeddings", "target": "Embedding_Collector", "value": 1, "x1": adjust_link_x(t_x, sub_embd_w), "y1": curr_y, "x2": source_link_x, "y2": source_link_y}
            ]
        }
        curr_y -= (y_pad+comp_h)
        source = "Embedding_Collector"
        for i in range(self.layer_count):
            curr_y -= (y_pad+comp_h)
            cur_x = 5
            for j in range(self.head_count):
                graph['nodes'].append({"id": "Layer_%d_Head_%d"%(i,j),
                                       "text": "H. %d"%j,
                                       "group": 1,
                                       "x": cur_x,
                                       "y": curr_y,
                                       "w": head_w})
                graph['links'].append({"source": source,
                                       "target": "Layer_%d_Head_%d"%(i,j),
                                       "value": 1,
                                       "x1": source_link_x,
                                       "y1": source_y,
                                       "x2": adjust_link_x(cur_x, head_w),
                                       "y2": adjust_link_y(curr_y, comp_h)})
                cur_x += (x_head_pad+head_w)
            curr_y -= (y_pad+comp_h)
            source_x, source_y = (svg_w/2)-(layer_w/2), curr_y
            source_link_x = adjust_link_x(source_x, layer_w)
            source_link_y = adjust_link_y(source_y, comp_h)
            graph['nodes'].append({"id": "Layer_%d_Collector"%i,
                                   "text": "Layer %d Output"%i,
                                   "group": 2,
                                   "x": source_x,
                                   "y": source_y,
                                   "w": layer_w})
            for j in range(self.head_count):
                graph['links'].append({"source": "Layer_%d_Head_%d"%(i,j),
                                       "target": "Layer_%d_Collector"%i,
                                       "value": 1,
                                       "x1": adjust_link_x(graph['nodes'][j-(self.head_count+1)]["x"], head_w),
                                       "y1": graph['nodes'][j-(self.head_count+1)]["y"],
                                       "x2": source_link_x,
                                       "y2": source_link_y})
            source = "Layer_%d_Collector"%i
        graph['nodes'].append({"id": "Prediction",
                               "text": "Prediction",
                               "group": -1,
                               "x": (svg_w/2)-(prd_w/2),
                               "y": 0,
                               "w": prd_w})
        graph['links'].append({"source": source,
                               "target": "Prediction",
                               "value": 1,
                               "x1": source_link_x,
                               "y1": source_y,
                               "x2": adjust_link_x((svg_w/2)-(prd_w/2), prd_w),
                               "y2": comp_h})
        return graph

    def get_player_model(self):
        def adjust_link_x(x, w):
            return x+(w/2)
        def adjust_link_y(y, h):
            return y+h

        svg_w, svg_h = 210+int(self.head_count*62.5), (self.layer_count+1)*210-75
        sub_embd_w, embd_w, sub_embd_x_pad = 200, 400, 50
        head_w, layer_w, prd_w, comp_h,  = 50, 400, 400, 30
        pool_part, pool_link = 100, 70
        y_pad = (svg_h - (comp_h*(2*self.layer_count+2))) / (2*self.layer_count+1)
        x_head_pad = (svg_w - 10 - (2*pool_part) - (head_w*self.head_count)) / (self.head_count-1)
        curr_y = svg_h-comp_h
        w_x = ((svg_w- (2*pool_part))/2)-(3*sub_embd_w/2 + sub_embd_x_pad)
        p_x = ((svg_w- (2*pool_part))/2)-(sub_embd_w/2)
        t_x = ((svg_w- (2*pool_part))/2)+(sub_embd_w/2 + sub_embd_x_pad)
        source_link_x = adjust_link_x(((svg_w- (2*pool_part))/2)-(embd_w/2), embd_w)
        source_link_y = adjust_link_y(curr_y-(y_pad+comp_h), comp_h)
        source_x, source_y = ((svg_w- (2*pool_part))/2)-(embd_w/2), curr_y-(y_pad+comp_h)
        graph = {
            'comp_h': comp_h,
            'nodes': [
                {"id": "Word_Embeddings", "text": "Word Embeddings", "group": 2, "x": w_x , "y": curr_y, "w": sub_embd_w},
                {"id": "Position_Embeddings", "text": "Position Embeddings", "group": 2, "x": p_x , "y": curr_y, "w": sub_embd_w},
                {"id": "Type_Embeddings", "text": "Type Embeddings", "group": 2, "x":  t_x, "y": curr_y, "w": sub_embd_w},
                {"id": "Embedding_Collector", "text": "Embeddings", "group": 2, "x":  source_x, "y": source_y, "w": embd_w }
            ],
            'links': [
                {"source": "Word_Embeddings", "target": "Embedding_Collector", "value": 1, "x1": adjust_link_x(w_x, sub_embd_w), "y1": curr_y, "x2": source_link_x, "y2": source_link_y},
                {"source": "Position_Embeddings", "target": "Embedding_Collector", "value": 1, "x1": adjust_link_x(p_x, sub_embd_w), "y1": curr_y, "x2": source_link_x, "y2": source_link_y},
                {"source": "Type_Embeddings", "target": "Embedding_Collector", "value": 1, "x1": adjust_link_x(t_x, sub_embd_w), "y1": curr_y, "x2": source_link_x, "y2": source_link_y}
            ]
        }
        curr_y -= (y_pad+comp_h)
        source = "Embedding_Collector"
        for i in range(self.layer_count):
            curr_y -= (y_pad+comp_h)
            cur_x = 5
            for j in range(self.head_count):
                graph['nodes'].append({"id": "Layer_%d_Head_%d"%(i,j),
                                       "text": "H. %d"%j,
                                       "group": 1,
                                       "x": cur_x,
                                       "y": curr_y,
                                       "w": head_w})
                graph['links'].append({"source": source,
                                       "target": "Layer_%d_Head_%d"%(i,j),
                                       "value": 1,
                                       "x1": source_link_x,
                                       "y1": source_y,
                                       "x2": adjust_link_x(cur_x, head_w),
                                       "y2": adjust_link_y(curr_y, comp_h)})
                cur_x += (x_head_pad+head_w)
            curr_y -= (y_pad+comp_h)
            source_x, source_y = ((svg_w- (2*pool_part))/2)-(layer_w/2), curr_y
            source_link_x = adjust_link_x(source_x, layer_w)
            source_link_y = adjust_link_y(source_y, comp_h)
            graph['nodes'].append({"id": "Layer_%d_Collector"%i,
                                   "text": "Layer %d Output"%i,
                                   "group": 2,
                                   "x": source_x,
                                   "y": source_y,
                                   "w": layer_w})
            graph['links'].append({"source": "Layer_%d_Collector"%i,
                                   "target": "Pooler",
                                   "value": 1,
                                   "x1": source_x+layer_w,
                                   "y1": source_y + comp_h/2,
                                   "x2": svg_w-pool_part-comp_h-5,
                                   "y2": source_y + comp_h/2})
            for j in range(self.head_count):
                graph['links'].append({"source": "Layer_%d_Head_%d"%(i,j),
                                       "target": "Layer_%d_Collector"%i,
                                       "value": 1,
                                       "x1": adjust_link_x(graph['nodes'][j-(self.head_count+1)]["x"], head_w),
                                       "y1": graph['nodes'][j-(self.head_count+1)]["y"],
                                       "x2": source_link_x,
                                       "y2": source_link_y})
            source = "Layer_%d_Collector"%i
        pooler_h = 2*(self.layer_count-1)*(comp_h+y_pad) + comp_h
        graph['nodes'].append({"id": "Pooler",
                               "text": "+",
                               "group": -1,
                               "x": svg_w-pool_part-comp_h-5,
                               "y": 0,
                               "h": pooler_h,
                               "w": comp_h})
        graph['nodes'].append({"id": "Prediction",
                               "text": "",
                               "group": -1,
                               "x": svg_w-comp_h-5,
                               "y": 0,
                               "w": comp_h})
        graph['links'].append({"source": "Pooler",
                               "target": "Prediction",
                               "value": 1,
                               "x1": svg_w-pool_part-5,
                               "y1": comp_h/2,
                               "x2": svg_w-comp_h-5,
                               "y2": comp_h/2})
        return graph


    def get_model_graph(self):
        if self.model_type == 'baseline':
            return self.get_baseline_model()
        else:
            return self.get_gbert_model()