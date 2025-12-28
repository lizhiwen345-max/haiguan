# -*- coding: utf-8 -*-
"""
@project: custom words similarity (本地版)
@author: LiZhiwen 
@time: 2025/11/13 -> optimized 2025/12
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


DEFAULT_MODEL_PATH = '/media/lzw/新加卷/LZW/models/chinese-roberta-wwm-ext'
BERT_MODEL_PATH = os.environ.get('BERT_MODEL_PATH', DEFAULT_MODEL_PATH)
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))


class Utils(object):
    def __init__(self):
        pass

    def get_datas(self, DIR):
        try:
            with open(DIR, 'r', encoding='utf-8') as load_f:
                dic = json.load(load_f)
            return dic
        except Exception as e:
            raise ValueError(f"读取文件失败: {DIR}, 错误: {str(e)}")

    def norm_word(self, word: str) -> str:
        """统一规范化：strip + 删除中间多余空白 + 全角转半角 + 特殊字符处理"""
        if not isinstance(word, str):
            return word
        # 去除首尾空白
        word = word.strip()
        # 全角空格转半角
        word = word.replace('\u3000', ' ')
        # 删除所有空白字符
        word = re.sub(r"\s+", "", word)
        # 删除零宽字符
        word = re.sub(r"[\u200b-\u200d\ufeff]", "", word)
        return word

    def fetch_chinese(self, word: str) -> str:
        # 先规范化再抽取中文
        word = self.norm_word(word)
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese = re.sub(pattern, '', str(word))
        return chinese

    def write_file(self, result_address: str, group: dict):
        try:
            with open(r'{}'.format(result_address), 'w', encoding='utf-8') as f:
                qd = json.dumps(group, ensure_ascii=False, indent=4)
                f.write(qd)
            print(f" 结果已保存至: {os.path.abspath(result_address)}")
        except Exception as e:
            raise RuntimeError(f"保存文件失败: {result_address}, 错误: {str(e)}")

    def filter_elements(self, param: dict) -> dict:
        # 保护性处理，确保 key 存在且类型正确
        param.setdefault('param4', {})
        param.setdefault('param2', [])
        param.setdefault('param5', {})
        param['param5'].setdefault('param4', {})
        param['param5'].setdefault('param2', {})

        new_p4 = {}
        for k, v in param['param5'].get('param4', {}).items():
            nk = self.norm_word(k)
            new_p4[nk] = max(int(v), int(new_p4.get(nk, 0)))
        param['param5']['param4'] = new_p4

        new_p2 = {}
        for k, v in param['param5'].get('param2', {}).items():
            nk = self.norm_word(k)
            new_p2[nk] = max(int(v), int(new_p2.get(nk, 0)))
        param['param5']['param2'] = new_p2

        new_param4 = {}
        for key, value in list(param['param4'].items()):
            k_norm = self.norm_word(key)
            if not isinstance(value, list):
                new_param4[k_norm] = []
                continue
            new_list = []
            for word in value:
                if isinstance(word, str) and '：' in word:
                    split_word = word.split('：', 1)[1] if len(word.split('：', 1)[1]) > 0 else word
                else:
                    split_word = word
                # 规范化
                normed = self.norm_word(split_word)
                new_list.append(normed)
                # nothing else needed here because param5 keys already normalized
            new_param4[k_norm] = new_list
        param['param4'] = new_param4

        # 3) 规范化 param2
        new_param2 = []
        for word in list(param['param2']):
            if isinstance(word, str) and '：' in word:
                split_word = word.split('：', 1)[1] if len(word.split('：', 1)[1]) > 0 else word
            else:
                split_word = word
            normed = self.norm_word(split_word)
            new_param2.append(normed)
        param['param2'] = new_param2

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
            if not isinstance(value, list):
                value = []
            for word in value:
                freq = param_freq.get(word, 0)
                if int(freq) > int(frequence):
                    dict_val[key_word] = frequence
                    key_word = word
                    frequence = freq
                else:
                    dict_val[word] = freq
            new_param[key_word] = dict_val if len(value) > 0 else {}
        return new_param

    def via_frequency_resort(self, new_param):
        sorted_dict = {}
        for key, dict_val in new_param.items():
            if isinstance(dict_val, dict) and len(dict_val) > 0:
                sorted_value = sorted(dict_val.items(), key=lambda kv: kv[1], reverse=True)
                sorted_dict[key] = [val[0] for val in sorted_value]
            else:
                sorted_dict[key] = []
        return sorted_dict


class Thesaurus(object):
    def __init__(self, freq):
        self.utils = Utils()
        self.fres = freq or {}

    def train_synonyms_features(self, trainSyn):
        trainKeywordsFre = {}
        for key, value in trainSyn.items():
            if not isinstance(value, list):
                continue
            # 插入主词作为第0位
            words = [key] + value
            res = {}
            for word in words:
                freq = int(self.fres.get(word, 0))
                word_cn = self.utils.fetch_chinese(word)
                w_list = jieba.lcut(word_cn, cut_all=False)
                for w in w_list:
                    res[w] = res.get(w, 0) + freq
            trainKeywordsFre[key] = res
        return trainKeywordsFre

    def features_param2(self, features, param2):
        param2_features = {}
        unknow = []
        for par in param2:
            par1 = self.utils.fetch_chinese(par)
            w_list = jieba.lcut(par1, cut_all=False)
            max_score = 0.0
            maxKw = None
            have = False
            for kw, feature in features.items():
                number = 0
                leng = 0
                for feak, feav in feature.items():
                    leng += feav
                    if feak in w_list:
                        number += feav
                score = (number / leng) if leng > 0 else 0
                if score > max_score and score > 0.1:
                    have = True
                    max_score = score
                    maxKw = kw
            if have:
                param2_features.setdefault(maxKw, []).append(par)
            else:
                unknow.append(par)
        return param2_features, unknow


class PendingWord(object):
    def __init__(self, model_path=BERT_MODEL_PATH, batch_size=BATCH_SIZE):
        self.utils = Utils()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"加载模型到设备: {self.device} -> {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = int(batch_size)
        print("模型加载完成。")

    def containAny(self, seq, aset):
        return any((c in aset) for c in seq)

    def negative_words(self, group):

        negatives = {'否', '非', '不'}
        neg_prefixes = ['不', '非', '否', '未', '无', '没有', '没']
        
        new_group = {}
        neg_collected = {}  # 收集所有否定词
        
        for key, values in list(group.items()):
            if not isinstance(values, list):
                values = []
            
            key_has_neg = self._is_negative_word(key, negatives, neg_prefixes)
            
            # 分离否定词和非否定词
            neg_values = []
            pos_values = []
            for v in values:
                if self._is_negative_word(v, negatives, neg_prefixes):
                    neg_values.append(v)
                else:
                    pos_values.append(v)
            
            if key_has_neg:
                neg_collected.setdefault(key, []).extend(neg_values)
                for pv in pos_values:
                    if pv not in new_group:
                        new_group[pv] = []
            else:
                if key in new_group:
                    new_group[key].extend(pos_values)
                else:
                    new_group[key] = pos_values
                for nv in neg_values:
                    neg_collected.setdefault(nv, [])
        
        # 合并相似的否定词组
        merged_neg = self._merge_similar_neg_groups(neg_collected)
        
        # 将否定词组加入结果
        for neg_key, neg_vals in merged_neg.items():
            unique_vals = []
            for nv in neg_vals:
                if nv not in unique_vals and nv != neg_key:
                    unique_vals.append(nv)
            new_group[neg_key] = unique_vals
        
        return new_group
    
    def _merge_similar_neg_groups(self, neg_collected):
        if not neg_collected:
            return {}
        
        # 提取每个否定词的词根
        neg_prefixes = ['不', '非', '否', '未', '无', '没有', '没']
        
        def get_root(word):
            for prefix in sorted(neg_prefixes, key=len, reverse=True):
                if word.startswith(prefix):
                    return word[len(prefix):]
            return word
        
        # 按词根分组
        root_to_negs = {}
        for neg_word in neg_collected.keys():
            root = get_root(neg_word)
            if root:  # 只有有效词根才分组
                root_to_negs.setdefault(root, []).append(neg_word)
        
        # 合并同词根的否定词
        merged = {}
        processed = set()
        
        for neg_word, vals in neg_collected.items():
            if neg_word in processed:
                continue
            
            root = get_root(neg_word)
            if root and root in root_to_negs and len(root_to_negs[root]) > 1:
                # 有多个同词根的否定词，选择第一个作为主词
                same_root_negs = root_to_negs[root]
                main_neg = same_root_negs[0]
                all_vals = []
                for sn in same_root_negs:
                    if sn != main_neg:
                        all_vals.append(sn)
                    all_vals.extend(neg_collected.get(sn, []))
                    processed.add(sn)
                merged[main_neg] = all_vals
            else:
                merged[neg_word] = vals
                processed.add(neg_word)
        
        return merged
    
    def _is_negative_word(self, word, negatives, neg_prefixes):
        if not isinstance(word, str):
            return False
        # 检查是否包含否定字符
        if self.containAny(word, negatives):
            return True
        # 检查是否以否定前缀开头
        for prefix in neg_prefixes:
            if word.startswith(prefix):
                return True
        return False

    def _compute_embeddings(self, texts):
        # batch 计算 mean pooling
        all_embeddings = []
        n = len(texts)
        for i in range(0, n, self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state  # [B, L, H]
                mask = inputs.get('attention_mask')  # [B, L]
                if mask is None:
                    # fallback to mean over all tokens
                    emb = last_hidden.mean(dim=1)
                else:
                    mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * mask, dim=1)
                    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                    emb = sum_embeddings / sum_mask
                all_embeddings.append(emb.cpu().numpy())
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            # fallback shape
            hidden_size = getattr(self.model.config, 'hidden_size', 768)
            return np.zeros((0, hidden_size))

    def cluster_unknow_by_BertEmbedding(self, renew, threshold_value):
        # 规范化并去重保序
        normed = []
        seen = set()
        for x in renew:
            nx = self.utils.norm_word(x) if isinstance(x, str) else x
            if nx not in seen:
                seen.add(nx)
                normed.append(nx)

        unique_renew = normed

        if len(unique_renew) == 0:
            return {}

        embeddings = self._compute_embeddings(unique_renew)
        if embeddings.size == 0:
            return {}

        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = embeddings / norms

        # 余弦相似度矩阵
        similarity = np.dot(normalized, normalized.T)
        np.fill_diagonal(similarity, 0.0)

        group = {}
        processed = set()
        n = len(unique_renew)
        for i in range(n):
            if i in processed:
                continue
            query = unique_renew[i]
            group[query] = []
            processed.add(i)
            scores = similarity[i]
            sim_idx = np.where(scores > threshold_value)[0]
            for idx in sim_idx:
                if idx not in processed:
                    group[query].append(unique_renew[idx])
                    processed.add(idx)
        # 仅保留非空分组
        final_group = {k: v for k, v in group.items() if v}
        return final_group


class MergeSynonyms(object):
    def __init__(self, param4_fre, fres):
        self.param4_fre = param4_fre or {}
        self.fres = fres or {}
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
            pin = int(self.fres.get(mainword, 0))
            word_cn = self.utils.fetch_chinese(mainword)
            w_list = jieba.lcut(word_cn, cut_all=False)
            lists.extend([x + '+' + str(pin) for x in w_list])
            all_lists.append(lists)
            self.all_mainwords.append(mainword)

        keys = []
        for x, group in enumerate(all_lists):
            res = {}
            grouplen = 0
            for item in group:
                parts = item.rsplit('+', 1)
                if len(parts) < 2:
                    continue
                token, pin_str = parts
                try:
                    pin = int(pin_str)
                except ValueError:
                    continue
                grouplen += pin
                res[token] = res.get(token, 0) + pin
            if grouplen == 0:
                new_data = {}
            else:
                new_data = {k: v for k, v in res.items() if v / grouplen > 0.05}
            keys.extend(list(new_data.keys()))
            self.keys_mainwords_nums.append(new_data)

        ckeys = set([key for key in keys if keys.count(key) > 1])
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
        strs = set()
        new_ckeys_mainwords = {}
        for key, main_words in ckeys_mainwords.items():
            sorted_str = ''.join(sorted(main_words))
            if sorted_str not in strs:
                new_ckeys_mainwords[key] = main_words
                strs.add(sorted_str)
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
        new_ckeys_mainwords_copy = {k: v[:] for k, v in new_ckeys_mainwords.items()}
        for mainw in list(compete_mainwords):
            try:
                mainw_index = self.all_mainwords.index(mainw)
                kws = self.keys_mainwords_nums[mainw_index]
            except (ValueError, IndexError):
                continue
            repeat_list = mainwords_ckeys_1_n.get(mainw, [])
            max_fre = -1
            max_key = None
            for li in repeat_list:
                current_fre = kws.get(li, 0)
                if current_fre > max_fre:
                    max_fre = current_fre
                    max_key = li
            if not max_key:
                continue
            for key2, val in new_ckeys_mainwords_copy.items():
                if max_key != key2 and mainw in val:
                    try:
                        val.remove(mainw)
                    except ValueError:
                        pass
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
                freq = int(self.fres.get(mainword, 0))
                if freq > frequence:
                    key_word = mainword
                    frequence = freq
            new_param4.setdefault(key_word, [])
            for mainword in mainwords:
                if mainword in self.param4_fre and isinstance(self.param4_fre[mainword], dict):
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
        for key in list(new_param4.keys()):
            new_param4[key] = list(dict.fromkeys(new_param4[key]))
        return new_param4


def merge_duplicate_outputs(result, param_fres):
    if not result:
        return result
    
    content_to_keys = {}
    for key, vals in result.items():
        if not isinstance(vals, list):
            vals = []
        # 使用排序后的元组作为key，只合并完全相同的列表
        vals_key = tuple(sorted(vals))
        
        if vals_key in content_to_keys:
            content_to_keys[vals_key].append(key)
        else:
            content_to_keys[vals_key] = [key]
    
    # 重新构建结果
    new_result = {}
    for vals_tuple, keys in content_to_keys.items():
        vals_list = list(vals_tuple)
        
        if len(keys) == 1:
            # 只有一个key，直接保留
            new_result[keys[0]] = vals_list
        else:
            # 多个key有相同的同义词列表，选择频率最高的作为主词
            best_key = keys[0]
            best_freq = int(param_fres.get(best_key, 0))
            for k in keys[1:]:
                freq = int(param_fres.get(k, 0))
                if freq > best_freq:
                    best_freq = freq
                    best_key = k
            
            # 其他key加入同义词列表
            all_synonyms = list(vals_list)
            for k in keys:
                if k != best_key and k not in all_synonyms:
                    all_synonyms.append(k)
            
            # 按频率排序
            all_synonyms.sort(key=lambda x: int(param_fres.get(x, 0)), reverse=True)
            new_result[best_key] = all_synonyms
    
    return new_result


def main(param, threshold_value, result_path):
    util = Utils()
    param = util.filter_elements(param=param)
    result = {}
    param_fres = {}
    # param_fres 已在 filter_elements 中规范化
    param_fres.update(param['param5']['param4'])
    param_fres.update(param['param5']['param2'])

    pw = PendingWord()
    sort = SortParams()
    theaurus = Thesaurus(freq=param_fres)

    group = {}
    if len(param['param2']) > 0:
        trainKeywordsFre = theaurus.train_synonyms_features(trainSyn=param['param4'])
        result, unknow = theaurus.features_param2(features=trainKeywordsFre, param2=param['param2'])
        group = pw.cluster_unknow_by_BertEmbedding(renew=unknow, threshold_value=threshold_value)

    if len(param['param4']) > 0:
        for key, value in param['param4'].items():
            k_norm = util.norm_word(key)
            # value 已经在 filter_elements 里规范化
            result.setdefault(k_norm, []).extend(value)

        # group keys/values 已被 PendingWord 规范化
        for key, value in group.items():
            k_norm = util.norm_word(key)
            result.setdefault(k_norm, []).extend(value)

        # 先用 main_word_resort 得到 new_result（freq 决定主词）
        new_result = sort.main_word_resort(param=result, param_freq=param_fres)
        result = {}
        for k, v in new_result.items():
            nk = util.norm_word(k)
            if isinstance(v, dict):
                # 提取同义词列表并去重
                synonyms = list(v.keys())
                result[nk] = synonyms
            else:
                result[nk] = []
    else:
        for key, value in group.items():
            k_norm = util.norm_word(key)
            result.setdefault(k_norm, []).extend(value)

    # 否定词处理
    result = pw.negative_words(group=result)

    # 重新排序/生成预输出
    new_param4_s = sort.main_word_resort(param=result, param_freq=param_fres)
    new_param4_s1 = sort.via_frequency_resort(new_param=new_param4_s)

    final_result = {}
    for k, vals in new_param4_s1.items():
        nk = util.norm_word(k) if isinstance(k, str) else k
        clean_vals = []
        for v in vals:
            nv = util.norm_word(v) if isinstance(v, str) else v
            # 避免将 key 自身加入到值列表
            if nv == nk:
                continue
            if nv not in clean_vals:
                clean_vals.append(nv)
        final_result[nk] = clean_vals
    
    final_result = merge_duplicate_outputs(final_result, param_fres)
    
    cleaned_result = {}
    for k, vals in final_result.items():
        clean_vals = []
        for v in vals:
            if v != k and v not in clean_vals:
                clean_vals.append(v)
        cleaned_result[k] = clean_vals

    util.write_file(result_address=result_path, group=cleaned_result)


if __name__ == '__main__':
    start_time = time.time()
    if len(sys.argv) != 4:
        print("Usage: python run_transformers_local.py <params_file> <待处理词文件> <结果文件>")
        sys.exit(1)

    params_file = sys.argv[1]
    待处理词文件 = sys.argv[2]
    result_path = sys.argv[3]

    util = Utils()
    try:
        content = util.get_datas(params_file)
        threshold_value = float(content.get('param3', 0.8))
        print(f"文件读取成功，待处理词数: {len(content.get('param2', []))}")
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        sys.exit(1)

    try:
        main(param=content, threshold_value=threshold_value, result_path=result_path)
        end_time = time.time()
        print(f"运行完成，总耗时: {end_time - start_time:.2f}s")
        sys.exit(0)
    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)
