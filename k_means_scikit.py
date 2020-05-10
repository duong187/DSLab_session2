import numpy as np
from collections import defaultdict
def get_tf_idf(data_path,data_path_2):
    with open('D:/Python/machine_learning/dslab_session_2/word_idf.txt') as f:
        words_idfs=[(line.split('<fff>')[0],float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        words_IDs=dict([word,index] for index,(word,idf) in enumerate(words_idfs))
        idfs=dict(words_idfs)
        
    with open (data_path) as f:
        documents=[(int(line.split('<fff>')[0]),int(line.split('<fff>')[1]),line.split('<fff>')[2])
        for line in f.read().splitlines()]
        data_tf_idf=[]
    for document in documents:
        label,word_id,text=document
        words=[word for word in text.split()if word in idfs]
        words_set=list(set(words))
        max_term_freq=max(words.count(word)for word in words_set)
        sum_square=0.0
        words_tf_idf=[]
        for word in words_set:
            freq=words.count(word)
            tf_idf_value=freq*1./max_term_freq*idfs[word]
            sum_square+=tf_idf_value**2
            words_tf_idf.append((words_IDs[word],tf_idf_value))
        tf_idf_normalized=[(str(index)+':'+str(value*1./sum_square))for index,value in words_tf_idf]
        sparse_rep=' '.join(tf_idf_normalized)
        data_tf_idf.append((label,word_id,sparse_rep))
    with open(data_path_2,'w') as f:
        f.write('\n'.join([str(label)+'<fff>'+str(word_id)+'<fff>'+sparse_rep for label,word_id,sparse_rep in data_tf_idf]))

def load_data(data_path):
    def sparse_to_dense(sparse_r_d,vocab_size):
        r_d=[0.0 for _ in range(vocab_size)]
        indices_tfidfs=sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index=int(index_tfidf.split(":")[0])
            tfidf=float(index_tfidf.split(":")[1])
            r_d[index]=tfidf
        return np.array(r_d)
    with open(data_path) as f:
        d_lines=f.read().splitlines()
    with open("D:/Python/machine_learning/dslab_session_2/word_idf.txt") as f:
        vocab_size=len(f.read().splitlines())
    data=[] #list of Members
    labels=[]
    label_count=defaultdict(int)
    for data_id,d in enumerate(d_lines):
        feature=d.split('<fff>')
        label,doc_id=feature[0],feature[1]
        labels.append(label)
        label_count[label]+=1
        r_d=sparse_to_dense(sparse_r_d=feature[2],vocab_size=vocab_size)
        data.append(r_d)
    return data,labels


def clustering_with_KMeans():
    data, labels=load_data('D:/Python/machine_learning/dslab_session_2/20news-full-tfidf.txt')
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    X=csr_matrix(data)
    print('========')
    kmeans=KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2018
    ).fit(X)
    labels=kmeans.labels_

if __name__ == '__main__':
    get_tf_idf('D:/Python/machine_learning/dslab_session_2/20news-bydate-full-procceed.txt','D:/Python/machine_learning/dslab_session_2/20news-full-tfidf.txt')
    clustering_with_KMeans()