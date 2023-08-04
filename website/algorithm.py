from flask import Blueprint, render_template, request, flash

algorithm = Blueprint('algorithm', __name__)

@algorithm.route('/textrank-word', methods=['GET', 'POST'])
def text_rank_word(keywords=None):
    if request.method == 'POST':
        text = request.form['txt']
        keywords = kw_extract(text)
        return render_template("textrank_word.html", keywords=keywords)
    
    elif request.method == 'GET':
        return render_template("textrank_word.html", keywords=keywords)
    
@algorithm.route('/textrank-sent', methods=['GET', 'POST'])
def text_rank_sent(keywords=None):
    if request.method == 'POST':
        text = request.form['txt']
        keywords = keysentence_Extractor(text) # summary 문장 추출
        return render_template("textrank_sent.html", keywords=keywords)
    
    elif request.method == 'GET':
        return render_template("textrank_sent.html", keywords=keywords)
    
@algorithm.route('/yake', methods=['GET', 'POST'])
def yake(keywords=None):
    if request.method == 'POST':
        text = request.form['txt']
        keywords = yake_KwExtractor(text)
        return render_template("yake.html", keywords=keywords)
    
    elif request.method == 'GET':
        return render_template("yake.html", keywords=keywords)
    
@algorithm.route('/keybert', methods=['GET', 'POST'])
def keybert(keywords=None):
    if request.method == 'POST':
        text = request.form['txt']
        keywords = bert_KwExtractor(text)
        return render_template("keybert.html", keywords=keywords)
    
    elif request.method == 'GET':
        return render_template("keybert.html", keywords=keywords)


##############################################################################################################
    ################################## TextRank 키워드 추출 및 전처리 함수 #####################################
##############################################################################################################


from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
from collections import Counter
from collections import defaultdict
import networkx as nx

def kw_extract(text):
    processed_text = [preprocess(s) for s in PunktSentenceTokenizer().tokenize(text)]
    sents = [sent.split() for sent in processed_text]

    idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, min_count=2)
    graph = cooccurrence(sents, vocab_to_idx, window=8, min_cooccurrence=2)
    scores = calculate_score(graph)

    R = sorted(scores.items(), key=lambda x: -x[1])[:30]
    KeyWords = ', '.join([idx_to_vocab[idx] for idx, _ in R])
    if KeyWords.isspace() or len(KeyWords.split()) < 1:
        KeyWords = '해당 텍스트의 키워드는 존재하지 않습니다. (ex. 지나치게 짧은 텍스트 등)'

    return KeyWords

def preprocess(sentence):
    sentence = sentence.lower() 
    sentence = ''.join([c for c in sentence if c.isalnum() or c.isspace()])
    
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    filtered_words = [word for word in words if word not in stop_words and word]
    
    pos_tags = pos_tag(filtered_words)
    pos_tags = [(word, 'a') if pos.startswith('J') else (word, pos.lower()) for word, pos in pos_tags]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos=tag[0]) if tag[0] in ['n', 'v', 'a', 'r', ] else word for word, tag in pos_tags]
    
    return ' '.join(words)

def scan_vocabulary(sents, min_count=2):
    counter = Counter(w for sent in sents for w in sent)
    counter = {w:c for w,c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2):
    counter = defaultdict(int)
    for s, tokens_i in enumerate(tokens):
        vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
        n = len(vocabs)
        for i, v in enumerate(vocabs):
            if window <= 0:
                b, e = 0, n
            else:
                b = max(0, i - window)
                e = min(i + window, n)
            for j in range(b, e):
                if i == j:
                    continue
                counter[(v, vocabs[j])] += 1
                counter[(vocabs[j], v)] += 1
    counter = {k:v for k,v in counter.items() if v >= min_cooccurrence}
    
    nx_graph = nx.Graph()
    for (word1, word2), count in counter.items():
        nx_graph.add_edge(word1, word2, weight=count)
        
    return nx_graph

def calculate_score(graph):
    scores = nx.pagerank(graph)
    return scores

##############################################################################################################
    ################################## TextRank 핵심문장 추출 및 전처리 함수 #####################################
##############################################################################################################

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def keysentence_Extractor(text):
    glove_dict = dict()
    f = open('/Users/minjaelee/ToyProjects/kw_extraction/website/glove.6B.100d.txt', encoding="utf-8") # 100차원 GloVe 벡터 사용

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        glove_dict[word] = word_vector_arr
    f.close()

    embedding_dim = 100
    zero_vector = np.zeros(embedding_dim)

    # 단어 벡터의 평균으로부터 문장 벡터를 계산 -> 중첩 함수
    def calculate_sentence_vector(sentence):
        return sum([glove_dict.get(word, zero_vector) for word in sentence])/len(sentence)

    def sentences_to_vectors(sentences):
        return [calculate_sentence_vector(sentence) for sentence in sentences]
    
    # 문장 간 코사인 유사도 계산
    def similarity_matrix(sentence_embedding):
        sim_mat = np.zeros([len(sentence_embedding), len(sentence_embedding)])
        
        for i in range(len(sentence_embedding)):
            for j in range(len(sentence_embedding)):
                sim_mat[i][j] = cosine_similarity(sentence_embedding[i].reshape(1, embedding_dim), sentence_embedding[j].reshape(1, embedding_dim))[0,0]
        
        return sim_mat
    
    # page-rank 기반 점수 산정
    def calculate_score(sim_matrix):
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        return scores

    processed_text = [preprocess(s) for s in PunktSentenceTokenizer().tokenize(text)]
    sentences = [sent.split() for sent in processed_text]
    SentenceEmbedding = sentences_to_vectors(sentences)
    SimMatrix = similarity_matrix(SentenceEmbedding)
    scores = calculate_score(SimMatrix).items()
    top_3_sent_idx = sorted(scores, key=lambda x: -x[1])[:3]

    original_sentences = PunktSentenceTokenizer().tokenize(text)
    summary = ""
    for idx, _ in top_3_sent_idx:
        summary += original_sentences[idx]+'\n\n'

    return summary

    
##############################################################################################################
    ################################## yake 기반 키워드 추출 및 전처리 함수 #####################################
##############################################################################################################

def yake_KwExtractor(text):
    import yake
    doc = ' '.join(PunktSentenceTokenizer().tokenize(text))

    params = {
    "lan" : 'en',
    "n" : 3, # n-gram 설정
    "dedupLim": 0.9, # 중복 제거 관련 파라미터
    "dedupFunc" : 'seqm', # 유사도 계산 알고리즘
    "windowsSize" : 1,
    "top" :10,
    "features":None,
    "stopwords":None
    }

    custom_kw_extractor = yake.KeywordExtractor(**params)
    keywords = custom_kw_extractor.extract_keywords(doc)

    return ' /// '.join([k for k,s in keywords])

##############################################################################################################
    ################################## keyBERT 기반 키워드 추출 및 전처리 함수 #####################################
##############################################################################################################

def bert_KwExtractor(text):
    from keybert import KeyBERT
    kw_model = KeyBERT(model='all-MiniLM-L6-v2') 
    processed_text = [preprocess(s) for s in PunktSentenceTokenizer().tokenize(text)]
    doc = ' '.join(processed_text)

    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), top_n=20, use_mmr=True, diversity=0.7)
    keywords = ' /// '.join([k for k,s in keywords])

    return keywords

