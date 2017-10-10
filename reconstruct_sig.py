import numpy as np 
import pandas as pd
import collections
import sys
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
# import numpy_indexed as npi
def bootstrap(arr):
	#Randomly sample n rows with replacement 
	rows,cols = arr.shape
	idx = np.random.randint(rows, size=rows)
	new_arr = arr[idx,:]
	#Not using the part two of the bootstrap process i.e. sampling based on probability distribution
	return np.array(new_arr) 
	final_arr = []
	for i in xrange(0, new_arr.shape[0]):
		total = np.sum(new_arr[i,:])
		prob = [x*1.0/total for x in new_arr[i,:]]
		prob_vec = list(np.random.choice(len(prob), total, prob))
		count = collections.Counter(prob_vec)
		val = np.zeros(96)
		for j in count.keys():
			val[j] = count[j]
		final_arr.append(val)	
	f = np.array(final_arr)
	return f
	
def clustering(H,W, no):
	concatH = np.concatenate(H)
	kcenter = KMeans(n_clusters=no)
	kcenter.fit(concatH)
	if any( (no*(no-1)/2) != a for a in [sum(kcenter.labels_[i:i+no]) for i in range(0, len(kcenter.labels_), no)]) :
		print "Failed clustering: A cluster has more than one signature from same iteration"
		#Do something to handle this
		print [sum(kcenter.labels_[i:i+no]) for i in range(0, len(kcenter.labels_), no)]
	concatW = np.concatenate(W)
	weight = []
	for i in xrange(0 , no):
		new = concatW[kcenter.labels_==i]
		weight.append(np.mean(new,axis=0))
	return kcenter.cluster_centers_,weight

def nmf(arr,no,no_of_iter):
	aggregateH = []
	aggregateW = []
	for i in xrange(0, no_of_iter):
		model = NMF(n_components=no, init='random',solver='mu', beta_loss='frobenius', max_iter=10000)
		bootstraparr = bootstrap(arr)
		W = model.fit_transform(bootstraparr)
		H = model.components_
		aggregateH.append(H)
		aggregateW.append(W)
	
	signature, exposure = clustering(aggregateH, aggregateW, no)
	return signature, exposure
	#TODO: silhoette score and automate calculating no.of.iterations based on model reconstruction error

def nmf_submit(arr,no,no_of_iter):
	aggregateH = []
	aggregateW = []
	for i in xrange(0, no_of_iter):
		model = NMF(n_components=no, init='random',solver='mu', beta_loss='frobenius', max_iter=10000)
		bootstraparr = bootstrap(arr)
		W = model.fit_transform(bootstraparr)
		H = model.components_
		aggregateH.append(H)
		aggregateW.append(W)
	return aggregateH, aggregateW


def example_data():
	mutation = pd.read_table('example-mutation-counts.tsv')
	n = mutation.set_index('Unnamed: 0')
	arr = n.as_matrix(columns=None)
	#TODO Remove columns (mutation column) that are observed in less that 1% of the samples
	signature, exposure = nmf(arr,5,500)
	true = np.load('example-signatures.npy')
	print cosine_similarity(signature,true)

def read_values(signature_num):
	aggsig = []
	aggexp = []
	for i in xrange(0, 10):
		if os.path.isfile('signature'+str(signature_num)+'_'+str(i)+'.npy')  and  os.path.isfile('exposure'+str(signature_num)+'_'+str(i)+'.npy'):
			a = np.load('signature'+str(signature_num)+'_'+str(i)+'.npy')
			b = np.load('exposure'+str(signature_num)+'_'+str(i)+'.npy')
			aggsig.append(a)
			aggexp.append(b)
	aggsigarr = np.array(aggsig)
	aggexparr = np.array(aggexp)
	print aggsigarr.shape, aggexparr.shape
	sig,exp = clustering(np.concatenate(aggsigarr), np.concatenate(aggexparr), signature_num)
	return sig, exp
def submit_nmf_jobs():
	real_data = pd.read_table('/cbcb/project2-scratch/nidhi/mutation/combined_mutation.txt', delimiter='\t')
	transpose = real_data.T
	arr = transpose.iloc[1:,0:].as_matrix(columns=None)
	sig_num = int(sys.argv[1])
	signature, exposure = nmf_submit(arr,sig_num,50)
	np.save('signature'+str(sig_num)+'_'+str(sys.argv[2]), signature)
	np.save('exposure'+str(sig_num)+'_'+str(sys.argv[2]), exposure)

def clustering_signatures():
	sig_num = int(sys.argv[1])
	signature,exposure = read_values(sig_num)
	print len(signature), len( exposure)
	np.save('final_signature',signature)
	np.save('final_exposure', exposure)
	true = np.load('/cbcb/project2-scratch/nidhi/mutation/true_sig.npy')
	cosine_sim = cosine_similarity(signature,true)
	np.save('cos_sim', cosine_sim)
def main():	
	'''Run on simulated data'''
	# example_data()

	'''Submit multiple jobs of nmf decomposition '''
	#submit_nmf_jobs()
	
	'''Cluster the signatures generated in multiple nmf iterations'''
	clustering_signatures()

if __name__ == "__main__":
	main()
