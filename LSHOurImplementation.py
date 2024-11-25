#%%
import mmh3
from collections import defaultdict
from Preprocessing import df

#%%
def listhash(l, seed):
    """Given a list of strings, return a hash of the list"""
    val = 0
    for e in l:
        val = val ^ mmh3.hash(e, seed)
    return val


def shingle(string, q):
    """Given a string and a shingle size q, return the set of shingles"""
    shingles = []
    for i in range(len(string) - q + 1):
        shingles.append(string[i : i + q])
    return set(shingles)


def minhash(l, seed):
    """Given a list of strings and a seed, return the minhash of the list"""
    return min([listhash(l, seed) for l in l])


def minhashes(s, k):
    """Given a set of shingles and a number of hashes k, return the min-hash signatures in a list"""
    return [minhash(s, i) for i in range(k)]


def signature(docs, q, k):
    """Given a dictionary of documents and a shingle size q, return the min-hash signatures of the documents"""
    res = dict()
    for key, value in docs.items():
        res[key] = minhashes(shingle(value, q), k)
    return res


def jaccard_approximated_sig(signature1, signature2):
    """Given two min-hash signatures, return the approximated Jaccard similarity"""
    union = set(signature1 + signature2)
    sum = 0
    for e in union:
        if e in signature1 and e in signature2:
            sum += 1
    return sum / len(union)


def lsh_signature_bands(signature, bands, rows_per_band):
    """Given a min-hash signature, the number of bands and the number of rows per band, return a list"""
    assert (
        len(signature) == bands * rows_per_band
    ), "Signature length must be equal to bands * rows_per_band"

    # Create bands from the signature
    bands_list = []
    for i in range(bands):
        start = i * rows_per_band
        # Convert the rows of the band into a tuple (hashable)
        band = tuple(signature[start : start + rows_per_band])
        bands_list.append(band)

    return bands_list


def lsh(docs, threshold, q, k, bands, rows_per_band):
    """Given a dictionary of documents, a similarity threshold, a shingle size q, the number of hashes k,
    the number of bands and the number of rows per band, return a set of similar documents
    """
    sigs = signature(docs, q, k)
    buckets = defaultdict(list)
    doc_names = list(sigs.keys())

    # Hash each document's bands into buckets
    for doc_name in doc_names:
        sig = sigs[doc_name]
        bands_list = lsh_signature_bands(sig, bands, rows_per_band)

        # For each band, hash the band into a bucket
        for i, band in enumerate(bands_list):
            buckets[(i, band)].append(doc_name)

        print("Hashed", doc_name)

    # Compare only documents in the same bucket
    # similar_docs = set()
    # print("\n\n Starting comparison\n\n")
    # for bucket_docs in buckets.values():
    #     if (
    #         len(bucket_docs) > 1
    #     ):  # Only compare if more than one document is in the bucket
    #         for i in range(len(bucket_docs)):
    #             for j in range(i + 1, len(bucket_docs)):
    #                 doc1 = bucket_docs[i]
    #                 doc2 = bucket_docs[j]
    #                 sig1 = sigs[doc1]
    #                 sig2 = sigs[doc2]
    #                 if jaccard_approximated_sig(sig1, sig2) >= threshold:
    #                     similar_docs.add((doc1, doc2))

    # return similar_docs
    similar_docs_list = []
    for bucket_docs in buckets.values():
        similar_doc = set()
        if len(bucket_docs) > 1:
            for i in range(len(bucket_docs)):
                for j in range(i + 1, len(bucket_docs)):
                    doc1 = bucket_docs[i]
                    doc2 = bucket_docs[j]
                    sig1 = sigs[doc1]
                    sig2 = sigs[doc2]
                    x = jaccard_approximated_sig(sig1, sig2)
                    if x >= threshold:
                        if x >= 0.50:
                            pass # probabily the same
                        else:
                            similar_doc.add((doc1, doc2))
        similar_docs_list.append(similar_doc)
    return similar_docs_list

#%%

threshold = 0.05
q = 5
k = 120
bands = 12
rows_per_band = 10

docs = {}

for i, tweet in enumerate(df["tweet"]):
    docs[i] = tweet


# get the first 10k tweets
docs = dict(list(docs.items())[0:10000])


similar_tweets = lsh(docs, threshold, q, k, bands, rows_per_band)
# Remove empty sets
similar_tweets = [x for x in similar_tweets if x]
print(similar_tweets)
# for tweet1, tweet2 in similar_tweets:
#     print(f"Tweets {tweet1} and {tweet2} are similar")


#%%
print(docs[8817])
print()
print(docs[5295])

# %%
