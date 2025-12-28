# -*- coding: utf-8 -*-
"""
@project: custom words similarity (transformers本地版) 
@author: LiZhiwen
@time: 2025/11/15 16:45
@description: 已修改为文件交互模式 (File I/O)，去除 Flask 和 MySQL。
"""

import os
import json
import re
import time
import jieba
import numpy as np
import torch
import sys
from transformers import AutoTokenizer, AutoModel



class Utils(object):
    def __init__(self):
        pass

    def fetch_chinese(self, word):
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese = re.sub(pattern, '', word)
        return chinese

    def filter_elements(self, param):
        if "param4" not in param or not isinstance(param["param4"], dict):
            param["param4"] = {}
        if "param5" not in param or not isinstance(param["param5"], dict):
            param["param5"] = {}
        if "param4" not in param["param5"] or not isinstance(param["param5"]["param4"], dict):
            param["param5"]["param4"] = {}
        if "param2" not in param["param5"] or not isinstance(param["param5"]["param2"], dict):
            param["param5"]["param2"] = {}
        if "param2" not in param or not isinstance(param["param2"], list):
            param["param2"] = []
        
        for key, value in param["param4"].items():
            for i, word in enumerate(value):
                if "：" in word:
                    split_word = word.split("：")[1] if len(word.split("：")[1]) > 0 else word
                    value[i] = split_word
                    if word in param["param5"]["param4"]:
                        param["param5"]["param4"][split_word] = param["param5"]["param4"][word]
                    else:
                        param["param5"]["param4"][split_word] = 1
            param["param4"][key] = value

        for i, word in enumerate(param["param2"]):
            if "：" in word:
                split_word = word.split("：")[1] if len(word.split("：")[1]) > 0 else word
                param["param2"][i] = split_word
                if word in param["param5"]["param2"]:
                    param["param5"]["param2"][split_word] = param["param5"]["param2"][word]
                else:
                    param["param5"]["param2"][split_word] = 1
        return param

class SortParams(object):
    def __init__(self):
        pass

    def main_word_resort(self, param, param_freq):
        new_param = {}
        for key, value in param.items():
            frequence = 0
            dict_val = {}
            key_word = key
            frequence = param_freq.get(key, 0)
            for word in value:
                freq = param_freq.get(word, 0)
                if int(freq) > int(frequence):
                    dict_val[key_word] = frequence
                    key_word = word
                    frequence = freq
                else:
                    dict_val[word] = freq
            if len(value) > 0:
                new_param[key_word] = dict_val
            else:
                new_param[key_word] = {}
        return new_param

    def via_frequency_resort(self, new_param):
        sorted_dict = {}
        for key, dict_val in new_param.items():
            sorted_value = sorted(dict_val.items(), key=lambda kv: kv[1], reverse=True) if len(dict_val) > 0 else {}
            if len(sorted_value) > 0:
                for val in sorted_value:
                    sorted_dict.setdefault(key, []).append(val[0])
            else:
                sorted_dict[key] = []
        return sorted_dict

class Thesaurus(object):
    def __init__(self, freq):
        self.utils = Utils()
        self.fres = freq

    def user_synonyms_features(self, user_synonyms, threshold):
        all_lists = []
        all_keys = []
        for key, value in user_synonyms.items():
            value.insert(0, key)
            lists = []
            for word in value:
                word = self.utils.fetch_chinese(word)
                w_list = jieba.lcut(word, cut_all=False)
                lists.extend(w_list)
            all_lists.append(lists)
            all_keys.append(key)
        keys = []
        keywords = []
        for x, group in enumerate(all_lists):
            res = {}
            new_dict = {}
            grouplen = len(group)
            if grouplen == 0:
                continue
            for i in range(len(group)):
                if group[i] in res:
                    res[group[i]] += 1
                else:
                    res[group[i]] = 1
            new_data = {k: v for k, v in res.items() if v / grouplen > threshold}
            new_dict[all_keys[x]] = new_data
            keywords.append(new_dict)
            keys.extend(list(new_data.keys()))
        return keywords

    def train_synonyms_features(self, trainSyn):
        trainKeywordsFre = {}
        for key, value in trainSyn.items():
            value.insert(0, key)
            res = {}
            for word in value:
                freq = self.fres.get(word, 0)
                word = self.utils.fetch_chinese(word)
                w_list = jieba.lcut(word, cut_all=False)
                for w in w_list:
                    if w in res:
                        res[w] += int(freq)
                    else:
                        res[w] = int(freq)
            trainKeywordsFre[key] = res
        return trainKeywordsFre

    def features_param2(self, features, param2):
        param2_features = {}
        unknow = []
        for par in param2:
            par1 = self.utils.fetch_chinese(par)
            w_list = jieba.lcut(par1, cut_all=False)
            max_score = 0
            maxKw = 0
            have = False
            for kw, feature in features.items():
                number = 0
                leng = 0
                for feak, feav in feature.items():
                    leng += feav
                    if feak in w_list:
                        number += feav
                if leng > 0:
                    score = number / leng
                else:
                    score = 0
                if score > max_score and score > 0.1:
                    have = True
                    max_score = score
                    maxKw = kw
            if have is True:
                param2_features.setdefault(maxKw, []).append(par)
            else:
                unknow.append(par)
        return param2_features, unknow

class PendingWord(object):
    def __init__(self, model_path):
        self.utils = Utils()
        print(f" 正在加载模型: {model_path}")
        if not os.path.exists(model_path):
            print(f"[严重错误] 模型路径不存在: {model_path}")
            sys.exit(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- 将模型加载到: {self.device} ---")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        print(" 模型加载完成。")

    def containAny(self, seq, aset):
        for c in seq:
            if c in aset:
                return True
        return False

    def negative_words(self, group):
        negatives = {'否', '非', '不'}
        negativeDict = {}
        dels = []
        for key, value in list(group.items()):
            negativeValue = []
            zhu = True
            if self.containAny(key, negatives):
                negativeDict[key] = []
                zhu = False
            
            for i, val in enumerate(value[:]):
                if self.containAny(val, negatives):
                    group[key].remove(val)
                    negativeValue.append(val)
            
            if zhu == False:
                negativeDict[key] = negativeValue
                newValue = group[key]
                if len(newValue) > 0:
                    negativeDict[newValue[-1]] = newValue
                dels.append(key)
            else:
                if len(negativeValue) > 0:
                    negativeDict[negativeValue[-1]] = negativeValue[:-1]
        
        for x in dels:
            if x in group:
                del group[x]
        
        group.update(negativeDict)
        return group

    def cluster_unknow_by_BertEmbedding(self, renew, threshold_value):
        unique_renew = sorted(list(set(renew)), key=renew.index)
        if len(unique_renew) == 0:
            return {}
        
        inputs = self.tokenizer(
            unique_renew, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state 
            mask = inputs['attention_mask']
            expanded_mask = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * expanded_mask, 1)
            sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        
        embeddings = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_embeddings = embeddings / norms
        
        similarity = np.dot(normalized_embeddings, normalized_embeddings.T)
        np.fill_diagonal(similarity, 0)
        
        group = {}
        words_to_cluster = unique_renew[:]
        processed_indices = set()

        for i, query in enumerate(words_to_cluster):
            if i in processed_indices:
                continue
            
            group[query] = []
            processed_indices.add(i)
            scores = similarity[i]
            similar_indices = np.where(scores > threshold_value)[0]
            
            for idx in similar_indices:
                if idx not in processed_indices:
                    group[query].append(words_to_cluster[idx])
                    processed_indices.add(idx)
        
        final_group = {k: v for k, v in group.items() if v}
        return final_group

class MergeSynonyms(object):
    def __init__(self, param4_fre, fres):
        self.param4_fre = param4_fre
        self.fres = fres
        self.all_mainwords = []
        self.keys_mainwords_nums = []
        self.utils = Utils()

    def co_occurrence_keywords(self):
        all_lists = []
        for mainword, value in self.param4_fre.items():
            lists = []
            if isinstance(value, dict):
                for word, pin in value.items():
                    word_cn = self.utils.fetch_chinese(word)
                    w_list = jieba.lcut(word_cn, cut_all=False)
                    lists.extend([x + '+' + str(pin) for x in w_list])
            
            pin = self.fres.get(mainword, 0)
            word_cn = self.utils.fetch_chinese(mainword)
            w_list = jieba.lcut(word_cn, cut_all=False)
            lists.extend([x + '+' + str(pin) for x in w_list])
            all_lists.append(lists)
            self.all_mainwords.append(mainword)
        
        keys = []
        for x, group in enumerate(all_lists):
            res = {}
            grouplen = 0
            for i in range(len(group)):
                group_list = group[i].split('+')
                if len(group_list) < 2:
                    continue
                grouplen += int(group_list[1])
                if group_list[0] in res:
                    res[group_list[0]] += int(group_list[1])
                else:
                    res[group_list[0]] = int(group_list[1])
            
            if grouplen == 0:
                new_data = {}
            else:
                new_data = {k: v for k, v in res.items() if v / grouplen > 0.05}
            
            keys.extend(list(new_data.keys()))
            self.keys_mainwords_nums.append(new_data)
        
        ckeys = set([key for key in keys if keys.count(key) > 1])
        singleKeys = set(keys) - ckeys
        ckeys_mainwords = {}
        ckeys_mainwords_list = []
        for key in ckeys:
            for num, kw in enumerate(self.keys_mainwords_nums):
                if key in kw:
                    ckeys_mainwords.setdefault(key, []).append(self.all_mainwords[num])
                    ckeys_mainwords_list.append(self.all_mainwords[num])
        single_mainwords = list(set(self.all_mainwords) - set(ckeys_mainwords_list))
        return ckeys_mainwords, single_mainwords

    def de_duplication(self, ckeys_mainwords):
        strs = []
        new_ckeys_mainwords = {}
        for key, main_words in ckeys_mainwords.items():
            sorted_str = ''.join(sorted(main_words))
            if sorted_str not in strs:
                new_ckeys_mainwords[key] = main_words
                strs.append(sorted_str)
        return new_ckeys_mainwords

    def merge_synonyms_utils(self, new_ckeys_mainwords):
        mains = []
        for mainwords in new_ckeys_mainwords.values():
            mains.extend(mainwords)
        compete_mainwords = set([main for main in mains if mains.count(main) > 1])
        mainwords_ckeys_1_n = {}
        for main in compete_mainwords:
            for kw, mainwords in new_ckeys_mainwords.items():
                if main in mainwords:
                    mainwords_ckeys_1_n.setdefault(main, []).append(kw)
        return compete_mainwords, mainwords_ckeys_1_n

    def compete_keywords_merge_synonyms(self, compete_mainwords, mainwords_ckeys_1_n, new_ckeys_mainwords):
        new_ckeys_mainwords_copy = new_ckeys_mainwords.copy()
        for key, main_list in new_ckeys_mainwords.items():
            for mainw in main_list[:]:
                if mainw in compete_mainwords:
                    repeat_list = mainwords_ckeys_1_n[mainw]
                    max_fre = -1
                    max_key = ""
                    try:
                        mainw_index = self.all_mainwords.index(mainw)
                        kws = self.keys_mainwords_nums[mainw_index]
                    except (ValueError, IndexError):
                        continue
                    
                    for li in repeat_list:
                        current_fre = kws.get(li, 0)
                        if current_fre > max_fre:
                            max_fre = current_fre
                            max_key = li
                    
                    if not max_key:
                        continue

                    for key2, value in new_ckeys_mainwords_copy.items():
                            if max_key != key2 and mainw in value:
                                value.remove(mainw)
        
        final_ckeys_mainwords = {k: v for k, v in new_ckeys_mainwords_copy.items() if v}
        return final_ckeys_mainwords

    def merge_synonyms(self, new_ckeys_mainwords, single_mainwords):
        new_param4 = {}
        for mainwords in new_ckeys_mainwords.values():
            if not mainwords:
                continue
            
            frequence = -1
            key_word = mainwords[0]
            
            for mainword in mainwords:
                freq = self.fres.get(mainword, 0)
                if int(freq) > int(frequence):
                    key_word = mainword
                    frequence = freq
            
            new_param4.setdefault(key_word, [])
            
            for mainword in mainwords:
                if mainword in self.param4_fre:
                    sub_words = list(self.param4_fre[mainword].keys())
                    new_param4[key_word].extend(sub_words)
                
                if key_word != mainword:
                    new_param4[key_word].append(mainword)
        
        for mainword in single_mainwords:
            if mainword not in new_param4:
                if mainword in self.param4_fre and self.param4_fre[mainword]:
                    new_param4[mainword] = list(self.param4_fre[mainword].keys())
                else:
                    new_param4[mainword] = []
        
        for key in new_param4:
            new_param4[key] = sorted(list(set(new_param4[key])), key=new_param4[key].index)
        
        return new_param4



def process_task(param, threshold_value, pw_model, output_file):
    print(" 开始数据处理...")
    util = Utils()
    param = util.filter_elements(param=param)
    result = {}
    param_fres = {}
    
    param5 = param.get("param5", {})
    param_fres.update(param5.get("param4", {}))
    param_fres.update(param5.get("param2", {}))
    
    pw = pw_model
    sort = SortParams()
    thesaurus = Thesaurus(freq=param_fres)
    
    param2 = param.get("param2", [])
    param4 = param.get("param4", {})
    
    group = {}
    if len(param2) > 0:
        print(" 步骤 1/5: 训练特征...")
        trainKeywordsFre = thesaurus.train_synonyms_features(trainSyn=param4)
        result, unknow = thesaurus.features_param2(features=trainKeywordsFre, param2=param2)
        
        print(f" 步骤 2/5: 向量聚类未知词... (数量: {len(unknow)})")
        group = pw.cluster_unknow_by_BertEmbedding(renew=unknow, threshold_value=threshold_value)
    
    print(" 步骤 3/5: 合并与排序...")
    if len(param4) > 0:
        for key, value in param4.items():
            result.setdefault(key, []).extend(value)
        for key, value in group.items():
            result.setdefault(key, []).extend(value)
        
        new_result = sort.main_word_resort(param=result, param_freq=param_fres)
        merge = MergeSynonyms(param4_fre=new_result, fres=param_fres)
        ckeys_mainwords, single_mainwords = merge.co_occurrence_keywords()
        new_ckeys_mainwords = merge.de_duplication(ckeys_mainwords=ckeys_mainwords)
        compete_mainwords, mainwords_ckeys_1_n = merge.merge_synonyms_utils(new_ckeys_mainwords=new_ckeys_mainwords)
        new_new_ckeys_mainwords = merge.compete_keywords_merge_synonyms(
            compete_mainwords=compete_mainwords,
            mainwords_ckeys_1_n=mainwords_ckeys_1_n,
            new_ckeys_mainwords=new_ckeys_mainwords
        )
        result = merge.merge_synonyms(new_ckeys_mainwords=new_new_ckeys_mainwords, single_mainwords=single_mainwords)
    else:
        for key, value in group.items():
            result.setdefault(key, []).extend(value)
    
    print(" 步骤 4/5: 处理否定词并最终排序...")
    result = pw.negative_words(group=result)
    new_param4_s = sort.main_word_resort(param=result, param_freq=param_fres)
    new_param4_s1 = sort.via_frequency_resort(new_param=new_param4_s)
    
    print(f" 步骤 5/5: 写入结果到文件: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_param4_s1, f, ensure_ascii=False, indent=4)
        print(" 成功写入结果文件。")
        return 0
    except Exception as e:
        print(f" 写入文件失败: {e}")
        return 1





BERT_MODEL_PATH = '/media/lzw/新加卷/LZW/models/Qwen3-Embedding-4B'

if __name__ == '__main__':
    start_time = time.time()

    # 参数检查
    if len(sys.argv) != 3:
        print("Usage: python your_script_name.py <输入JSON参数文件> <输出结果JSON文件>")
        print("Input Format (JSON): {'param4': {...}, 'param2': [...], 'param5': {...}, 'threshold': 0.9}")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # 1. 加载输入
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 支持 json 加载或 eval 加载 
            try:
                input_data = json.loads(content)
            except json.JSONDecodeError:
                input_data = eval(content) # 兼容参考代码的 eval 写法
            
            # 提取参数
            param = {
                "param4": input_data.get("param4", {}), # 现有同义词库
                "param2": input_data.get("param2", []), # 待处理词
                "param5": input_data.get("param5", {"param2": {}, "param4": {}}) # 词频
            }
            threshold = float(input_data.get("threshold", 0.9)) # 阈值，默认0.9

        print(f"文件读取成功，待处理词数: {len(param['param2'])}")
        end_time_read = time.time()
        print(f"文件读取耗时: {end_time_read - start_time:.4f}s")

    except Exception as e:
        print(f"[错误] 读取输入文件失败: {e}")
        sys.exit(1)

    # 2. 加载模型
    try:
        pw_model = PendingWord(BERT_MODEL_PATH)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        sys.exit(1)

    # 3. 执行处理
    try:
        return_code = process_task(param, threshold, pw_model, output_file)
        
        end_time_process = time.time()
        print(f"处理耗时: {end_time_process - end_time_read:.4f}s")
        print(f"总耗时: {end_time_process - start_time:.4f}s")
        print(f"程序返回码: {return_code}")
        sys.exit(return_code)
        
    except Exception as e:
        print(f"[严重错误] 程序运行出错: {e}")
        sys.exit(1)