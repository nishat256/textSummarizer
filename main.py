from nltk.corpus import brown, stopwords
from string import punctuation
from nltk.cluster.util import cosine_distance
import numpy as np
from operator import itemgetter
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

def sent_similarity(sent_1,sent_2):
    stop_words = stopwords.words('english')+list(punctuation)
    sent_1=[lemma.lemmatize(word,pos='v') for word in word_tokenize(sent_1.lower())]
    sent_2=[lemma.lemmatize(word,pos='v') for word in word_tokenize(sent_2.lower())]
    all_words=list(set(sent_1+sent_2))
    vector_1=[0]*len(all_words)
    vector_2=[0]*len(all_words)
    for w in sent_1:
        if w in stop_words:
            continue
        vector_1[all_words.index(w)]+=1
    for w in sent_2:
        if w in stop_words:
            continue
        vector_2[all_words.index(w)]+=1
    return 1- cosine_distance(vector_1,vector_2)


def build_similarity_matrix(sentences):
    S=np.zeros((len(sentences),len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i==j:
                continue
            S[i][j]=sent_similarity(sentences[i],sentences[j])
    #normaize the matrix row-wise
    for i in range(len(S)):
        if S[i].sum() != 0:
            S[i]/=S[i].sum()
    return S



def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P

def return_summary(bigText,SUMMARY_SIZE=5):
    sentences=sent_tokenize(bigText)
    if len(sentences) < SUMMARY_SIZE:
        return (False,'Number of sentences in your text is less than summarized sentences you want .<br/>Please specify lower number.')
    S=build_similarity_matrix(sentences)
    sentence_ranks = pagerank(S)
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    SELECTED_SENTENCES = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])
    summary = itemgetter(*SELECTED_SENTENCES)(sentences)
    return (True,summary)

if __name__=='__main__':
    bigText="""
              With just 20 days left for Karnataka to submit its revised recommendations on the demarcation of the Eco-Sensitive Zone (ESZ) in the Western Ghats in the State, an MP representing Western Ghat areas in Karnataka are growing anxious.

According to Nalin Kumar Kateel, BJP MP from Dakshina Kannada, if Karnataka fails to respond by August 25 — the deadline fixed by the Centre — the demarcation recommended by the Kasturirangan Committee would be deemed as final.

According to Mr. Kateel, the alleged delay is likely have an impact on hundreds of villages that come under the purview of the ESZ in the Western Ghats section in Karnataka.

As per the draft notification, based on the report of Kasturirangan Committee, 20,668 sq. km in Karnataka would come under ESZ, which includes 1,576 villages in eight districts of Karnataka.

The neighbouring Kerala government submitted to the Centre its revised recommendations on the subject in June. The Centre had sought the views of Karnataka, Kerala, Maharashtra, Goa, Gujarat, and Tamil Nadu.

The six States that share the range are trying to bring down the area under the proposed ESAs in the Western Ghat ranges. Once the ESZ is declared, restrictions would be imposed on human activity and development projects in villages in the forest periphery.

“I am writing a detailed letter to Chief Minister H.D. Kumaraswamy appealing to him to hasten the process,” said Mr. Kateel.
              """
    summary=return_summary(bigText)
    # Print the actual summary
    for sentence in summary:
        print(sentence.encode('utf-8'))
    

