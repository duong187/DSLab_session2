from collections import defaultdict
import numpy as np
import random
class Member:
    def __init__(self,r_d,label=None,doc_id=None):
        self._r_d=r_d
        self._label=label
        self._doc_id=doc_id

class Cluster:
    def __init__(self):
        self._centroid=None
        self._members=[]
    def reset_members(self):
        self._members=[]
    def add_members(self,member):
        self._members.append(member)

class Kmeans:
    def __init__(self,num_clusters):
        self._num_clusters=num_clusters
        self._clusters=[Cluster() for _ in range(self._num_clusters)]
        self._E=[] #list of centroids
        self._S=0.
    #Tải dữ liệu vào self._data
    def load_data(self,data_path):
        def sparse_to_dense(sparse_r_d,vocab_size):
            r_d=[0.0 for _ in range(vocab_size)]
            indices_tfidfs=sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index=int(index_tfidf.split(':')[0])
                tfidf=float(index_tfidf.split(':')[1])
                r_d[index]=tfidf
            return np.array(r_d)
        with open(data_path) as f:
            d_lines=f.read().splitlines()
        with open("D:/Python/machine_learning/dslab_session_2/word_idf.txt") as f:
            vocab_size=len(f.read().splitlines())
        self._data=[] #list of Members
        self._label_count=defaultdict(int)
        for data_id,d in enumerate(d_lines):
            feature=d.split('<fff>')
            label,doc_id=int(feature[0]),int(feature[1])
            self._label_count[label]+=1
            r_d=sparse_to_dense(sparse_r_d=feature[2],vocab_size=vocab_size)
            self._data.append(Member(r_d=r_d,label=label,doc_id=doc_id))
            print("Line ",data_id," loaded!!!")
    #tính cosine similarity
    def compute_similarity(self,member,centroid):
        return np.dot(member._r_d, centroid)*(1.0/(np.linalg.norm(member._r_d)*np.linalg.norm(centroid)))
    #Chọn cluster cho một member
    def select_cluster_for(self,member):
        best_fit_cluster=None
        max_similarity=-1
        for cluster in self._clusters:
            current_similarity=self.compute_similarity(member,cluster._centroid)
            if current_similarity>max_similarity:
                max_similarity=current_similarity
                best_fit_cluster=cluster
        best_fit_cluster.add_members(member)
        return max_similarity
    #tính lại centroid của một cluster dựa trên các member mới
    def update_centroid_of(self,cluster):
        # member_rds = [member._r_d for member in cluster._members]
        # aver_rd = np.mean(member_rds, axis = 0)
        # sqrt_sum_sqr = np.sqrt(np.sum(aver_rd ** 2))
        # new_centroid = np.array([val/sqrt_sum_sqr for val in aver_rd])  # normalized centroid
        
        # cluster._centroid = new_centroid
        members_rds=[member._r_d for member in cluster._members]
        avg_r_d=np.mean(members_rds,axis=0)
        norm_r_d=np.sqrt(np.sum(avg_r_d**2))
        new_r_d=np.array([value/norm_r_d for value in avg_r_d])
        cluster._centroid=new_r_d
    #điều kiện dừng
    def stopping_condition(self,criterion, threshold):
        criteria=["centroid","similarity","max_iters"]
        assert criterion in criteria
        if criterion=="max_iters":
            if self._iteration>threshold:
                return True
            else:
                return False
        elif criterion=="centroid":
            E_new=[list(cluster._centroid)for cluster in self._clusters]
            E_new_minus_E=[centroid for centroid in E_new if centroid not in self._E]
            self._E=E_new
            if len(E_new_minus_E)<=threshold:
                return True
            else:
                return False
        else:
            S_new_minus_S=self._new_S-self._S
            self._S=self._new_S
            if S_new_minus_S<=threshold:
                return True
            else:
                return False

   #khởi tạo centroid ban đầu cho các cluster
    def random_init(self,seed_value):
        assert seed_value==self._num_clusters

        sample=random.sample(self._data,seed_value)
        for index,cluster in enumerate(self._clusters):
            cluster._centroid=sample[index-1]._r_d
    #chạy chương trình
    def run(self,seed_value,criterion,threshold):
        self.random_init(seed_value)
        self._iteration=0
        while True:
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S=0
            for member in self._data:
                max_S=self.select_cluster_for(member)
                self._new_S+=max_S
            print("Current S: ",self._new_S)
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            self._iteration+=1
            if self.stopping_condition(criterion,threshold):
                break
    #tính purity 
    def compute_purity(self):
        majority_sum=0.
        for cluster in self._clusters:
            members_labels=[member._label for member in cluster._members]
            max_count=max([members_labels.count(label)for label in range(20)])
            majority_sum+=max_count
        return majority_sum*1/len(self._data)
    # tính NMI
    def compute_NMI(self):
        I_value,H_omega,H_C,N=0.0,0.0,0.0,len(self._data)
        for cluster in self._clusters:
            wk=len(cluster._members)*1.
            H_omega+=-wk/N*np.log10(wk/N)
            member_labels=[member._label for member in cluster._members]
            for label in range(20):
                wk_cj=member_labels.count(label)*1.
                cj=self._label_count[label]
                I_value+=wk_cj/N*np.log10(N*wk_cj/(wk*cj)+1e-12)
        for label in range(20):
            cj=self._label_count[label]*1.
            H_C+=-cj/N*np.log10(cj/N)
        return I_value*2./(H_omega+H_C)

if __name__=='__main__':
    implement = Kmeans(num_clusters = 20)
    implement.load_data(data_path = 'D:/Python/machine_learning/dslab_session_2/data_tf_idf.txt')
    implement.run(seed_value=20,criterion = 'max_iters', threshold = 20)
    print('Purity = ', implement.compute_purity())
    print('NMI = ', implement.compute_NMI())