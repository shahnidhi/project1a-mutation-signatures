import numpy as np 
import pandas as pd
import sys
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
# import numpy_indexed as npi
def bootstrap(arr):
	return arr
	#TODO Randomly sample n rows with replacement 

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
		model = NMF(n_components=no, init='random',solver='mu', beta_loss='frobenius', max_iter=1000)
		bootstraparr = bootstrap(arr)
		W = model.fit_transform(bootstraparr)
		H = model.components_
		aggregateH.append(H)
		aggregateW.append(W)

	signature, exposure = clustering(aggregateH, aggregateW, no)
	return signature, exposure
	#TODO: silhoette score and automate calculating no.of.iterations based on model reconstruction error


def main():	
	mutation = pd.read_table('example-mutation-counts.tsv')
	n = mutation.set_index('Unnamed: 0')
	arr = n.as_matrix(columns=None)
	#TODO Remove columns (mutation column) that are observed in less that 1% of the samples
	signature, exposure = nmf(arr,5,500)
	true = np.load('example-signatures.npy')
	print cosine_similarity(signature,true)

if __name__ == "__main__":
	main()
