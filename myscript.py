import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
from gensim.models.nmf import Nmf
import mymodule

def main():
    # Select parameters
    n_topics = 4                 # True number of topics (ex topic: 0.5*'a0' + 0.3*'a1' + 0.2*'a2')
    n_tokens = 8                 # Number of tokens per topics (n_tokens = 3 in ex above)
    n_docs = 500                 # Number of documents (list of tokens) in corpus (list of lists of tokens) 
    seed = 2                     # Replicability
    topics_test = range(2, 8, 1) # Number of topics to test (to see which one is optimal)

    # Compute models parameters
    corpus = mymodule.simu_corpus(n_docs=n_docs, n_topics=n_topics, n_tokens=n_tokens, seed=seed)
    dictionary = Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]

    list_nmf_umass = []
    list_nmf_cv = []
    list_lda_umass = []
    list_lda_cv = []

    for num_topics in topics_test:
        # Compute models
        model_nmf = Nmf(corpus=corpus_bow, id2word=dictionary, num_topics=num_topics, random_state=seed)
        model_lda = LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=num_topics, random_state=seed)
        
        # Compute coherences scores
        coherence_nmf_umass = mymodule.coherence_umass(model=model_nmf, corpus_bow=corpus_bow)
        coherence_nmf_cv = mymodule.coherence_cv(model=model_nmf, corpus=corpus, dictionary=dictionary)
        coherence_lda_umass = mymodule.coherence_umass(model=model_lda, corpus_bow=corpus_bow)
        coherence_lda_cv = mymodule.coherence_cv(model=model_lda, corpus=corpus, dictionary=dictionary)

        # Store coherence results
        list_nmf_umass.append(coherence_nmf_umass)
        list_nmf_cv.append(coherence_nmf_cv)
        list_lda_umass.append(coherence_lda_umass)
        list_lda_cv.append(coherence_lda_cv)

    # Display results
    data = {
        'topics': topics_test,
        'nmf_umass': list_nmf_umass,
        'nmf_cv': list_nmf_cv,
        'lda_umass': list_lda_umass,
        'lda_cv': list_lda_cv,
    }
    df = pd.DataFrame.from_dict(data)
    best_nmf_umass = df[df['nmf_umass'] == max(df['nmf_umass'])]['topics']
    best_nmf_cv = df[df['nmf_cv'] == max(df['nmf_cv'])]['topics']
    best_lda_umass = df[df['lda_umass'] == max(df['lda_umass'])]['topics']
    best_lda_cv = df[df['lda_cv'] == max(df['lda_cv'])]['topics']

    print(df)
    print('Optimal number of topics for NMF with UMASS: ', best_nmf_umass.values)
    print('Optimal number of topics for NMF with CV: ', best_nmf_cv.values)
    print('Optimal number of topics for LDA with UMASS: ', best_lda_umass.values)
    print('Optimal number of topics for LDA with CV: ', best_lda_cv.values)
    
if __name__ == '__main__':
    main()
