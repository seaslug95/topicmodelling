import string
import random
from gensim.models import CoherenceModel

def simu_topics(n_topics, n_tokens, seed) -> dict:
    """
    Returns dict containing topics with
    their respective token and token frequency.
    Token frequency follows uniform distri.
    """
    random.seed(seed)
    topics = {}
    letters = list(string.ascii_lowercase)[:n_topics]
    for i in range(n_topics):
        topics['topic_'+str(i)] = {}
        topics['topic_'+str(i)]['tokens'] = [letters[i]+str(j) for j in range(n_tokens)]
        freqs = [random.uniform(a=0,b=1) for j in range(n_tokens)]
        freqs = [f/sum(freqs) for f in freqs]
        topics['topic_'+str(i)]['freqs'] = freqs
    return topics

def simu_doc(n_tokens, topics, distri_topics, seed) -> list:
    """
    Returns a doc which is a list of token.
    Token are randomly sampled in function of
    topics and token frequencies.
    """
    random.seed(seed)
    doc = []
    lists_tokens = [topics['topic_'+str(i)]['tokens'] for i in range(len(topics))]
    freqs_tokens = [topics['topic_'+str(i)]['freqs'] for i in range(len(topics))]
    freqs_topics = [distri_topics['topic_'+str(i)] for i in range(len(topics))]
    tokens = [token for sublist in lists_tokens for token in sublist]
    freqs = [f*ftop for ftok, ftop in zip(freqs_tokens, freqs_topics) for f in ftok]
    doc = random.choices(tokens, weights=freqs, k=n_tokens)
    return doc

def simu_distridocs(n_docs, n_topics, seed) -> dict:
    """
    Returns a dict containing the topics distri
    of all docs. It follows an exponential distri.
    """
    random.seed(seed)
    distri_docs = {}
    for d in range(n_docs):
        distri_docs['doc_'+str(d)] = {}
        freqs = [random.expovariate(lambd=1) for t in range(n_topics)]
        freqs = [f/sum(freqs) for f in freqs]
        for t in range(n_topics):
            distri_docs['doc_'+str(d)]['topic_'+str(t)] = freqs[t]
    return distri_docs

def simu_corpus(n_docs, n_topics, n_tokens, seed) -> list:
    """
    Returns a corpus which is a list of lists of tokens.
    n_docs is the number of docs.
    n_topics is the number of topics generating docs.
    n_tokens is the number of token per doc.
    Topics distri follows an exponential.
    Tokens distri within topics follows an uniform.
    """
    topics = simu_topics(n_topics=n_topics, n_tokens=n_tokens, seed=seed)
    distri_docs = simu_distridocs(n_docs=n_docs, n_topics=n_topics, seed=seed)
    corpus = [simu_doc(n_tokens=n_tokens, topics=topics, distri_topics=distri_docs[dtop], seed=seed) for dtop in distri_docs]
    return corpus

def coherence_umass(model, corpus_bow) -> float:
    """
    Returns u_mass coherence.   
    """
    cm = CoherenceModel(model=model, corpus=corpus_bow, coherence='u_mass')
    res = cm.get_coherence()
    return res

def coherence_cv(model, corpus, dictionary) -> float:
    """
    Returns c_v coherence. 
    """
    cm = CoherenceModel(model=model, texts=corpus, dictionary=dictionary, coherence='c_v')
    res = cm.get_coherence()
    return res
