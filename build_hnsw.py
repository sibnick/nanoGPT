import os
import numpy as np
import hnswlib

dim = 768
num_elements = 50304

# Generating sample data
# data = np.float32(np.random.random((num_elements, dim)))
import torch
embeddings = torch.load("/data/nikolay/flibusta-prj/static-emb/data_openweb/glove-768-8.pt")
data = embeddings["_orig_mod._context_embeddings.weight"] + embeddings["_orig_mod._focal_embeddings.weight"]
print(data.shape)
data = data.cpu().numpy()
# Declaring index
p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip

# Initiating index
# max_elements - the maximum number of elements, should be known beforehand
#     (probably will be made optional in the future)
#
# ef_construction - controls index search speed/build speed tradeoff
# M - is tightly connected with internal dimensionality of the data
#     strongly affects the memory consumption

p.init_index(max_elements=num_elements, ef_construction=100, M=64)
# Controlling the recall by setting ef:
# higher ef leads to better accuracy, but slower search
p.set_ef(50)
p.set_num_threads(4)  # by default using all available cores
# We split the data in two batches:
print("Adding first batch of %d elements" % (len(data)))
p.add_items(data)
# Serializing and deleting the index:
index_path = 'index.bin'
# print("Saving index to '%s'" % index_path)
p.save_index(index_path)

# p = hnswlib.Index(space='cosine', dim=dim)
# p.load_index(index_path)
# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(data, k=3)
for x in zip(labels[:10], distances[:10]):
    print(x[0], x[1])

#input()
#os.remove(index_path)
