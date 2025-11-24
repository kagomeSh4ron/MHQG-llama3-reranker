#
#   用于文段重排部分准确率的测试
#
import json
import nltk
from nltk import word_tokenize
nltk.download('punkt')



# 将句子拆成词语
def sent2text(sent):
    texts = [word for word in word_tokenize(sent)]
    return texts

# 将文段拆成词语
def sents2text(sents):
    texts = [[word for word in word_tokenize(sent)] for sent in sents]
    return texts

# 建立语料库
def corpus_builder(sents):
    texts = sents2text(sents)
    all_list = []
    for text in texts:
        all_list += text
    corpus = set(all_list)
    #print(corpus)
    corpus_dict = dict(zip(corpus, range(len(corpus))))
    return corpus_dict

# 建立文本的向量表示
def vector_rep(text, corpus_dict):
    vec = []
    for key in corpus_dict.keys():
        if key in text:
            vec.append((corpus_dict[key], text.count(key)))
        else:
            vec.append((corpus_dict[key], 0))

    vec = sorted(vec, key= lambda x: x[0])
    return vec

# 计算余弦相似性（两个向量长度相同，做了运算简化）
def similarity_with_2_sents(vec1, vec2):
    inner_product = 0
    square_length_vec1 = len(vec1)
    for tup1, tup2 in zip(vec1, vec2):
        inner_product += tup1[1]*tup2[1]*1000 # 避免精度丢失
    return (inner_product/square_length_vec1)

# 返回检索到的关键信息
def rerank(context,answer):
    sents = []
    key_contexts=''

    sents.append(answer)
    for con in context:
        sents.append(con[0])
        for i in con[1]:
            sents.append(i)
    
    corpus_dict = corpus_builder(sents)
             
    # 第一次检索
    vec_answer = vector_rep(sent2text(answer),corpus_dict) # 查询文本向量
    max_index = -1  # 第一次查询产生的匹配文本下标
    max_sim = -1    # 最大余弦相似性
    index = 0   # 遍历下标
    for con in context:
        vec_con = vector_rep(sent2text(con[0]),corpus_dict)
        score = similarity_with_2_sents(vec_con,vec_answer)
        
        if(score>max_sim):
            max_index = index
            max_sim = score

        for i in con[1]:
            vec_con = vector_rep(sent2text(i),corpus_dict)
            score = similarity_with_2_sents(vec_con,vec_answer)
            if(score>max_sim):
                max_index = index
                max_sim = score
        
        index = index + 1

    # 第二次检索
    vec_answer = vector_rep(sent2text(answer + context[max_index][0]),corpus_dict)
    max_index2 = -1 # 第二次查询产生的匹配文本下标
    max_sim = -1
    index = 0
    for con in context:
        # 避免重复
        if index == max_index:
            index = index + 1
            continue
        vec_con = vector_rep(sent2text(con[0]),corpus_dict)
        score = similarity_with_2_sents(vec_con,vec_answer)
        
        if(score>max_sim):
            max_index2 = index
            max_sim = score
            
        for i in con[1]:
            vec_con = vector_rep(sent2text(i),corpus_dict)
            score = similarity_with_2_sents(vec_con,vec_answer)
            if(score>max_sim):
                max_index2 = index
                max_sim = score
        
        index = index + 1

    # 检索到的supporting_fact（这个代码仅用于测试，没有保存对应文段）
    keys = [context[max_index][0],context[max_index2][0]]

    return keys


with open('./dataset/dev.json', 'r',encoding='utf-8') as f:
    dataset=json.load(f)

acc_acount1 = 0
acc_acount2 = 0
yesno_count = 0
for item in dataset:
    flag = 0
    if item['answer'] == 'yes' or item['answer'] == 'no':
        yesno_count = yesno_count + 1
        continue
    # 进行重排
    keys=rerank(item['context'],item['answer'])
    list_cmp=set()
    for sf in item['supporting_facts']:
        if(sf[0]in list_cmp):
            continue
        else:
            list_cmp.add(sf[0])
        if sf[0] in keys:
            flag = 1
    
    if flag:
        acc_acount1 = acc_acount1 + 1

    set_keys = set(keys)
    if list_cmp == set_keys:
        acc_acount2 = acc_acount2 + 1
        #print('2,',acc_acount2,list_cmp,set_keys)
    print('1,',acc_acount1)


total = len(dataset) - yesno_count
acc1 = acc_acount1/total
acc2 = acc_acount2/total
print(total,yesno_count)
print(f'acc1:{acc1},acc2:{acc2}')
