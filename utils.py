from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from persim import PersistenceImager

def get_most_frequent_words(text, num_top_words=300):
    words_data = dict(Counter(text.split()))
    top_words = sorted(words_data.keys(), key=lambda x: -words_data[x])[:num_top_words]
    return top_words

def words_set_to_cloud(words, vectorize):
    lost_words_count = 0
    size = len(words)
    cloud = []
    for i in range(len(words)):
        word = words[i]
        try:
            vector = vectorize(word)
            cloud.append(vector)
        except:
            pass

    cloud = np.array(cloud)
    
    norms = np.linalg.norm(cloud, axis=1, keepdims=True)
    cloud = cloud / norms

    return cloud
    
def get_text_point_cloud(text, vectorize, num_top_words=300):
    most_frequent_words = get_most_frequent_words(text, num_top_words)
    return words_set_to_cloud(most_frequent_words, vectorize)

def plot_PIs_for_params(h1, pixel_size=0.015, sigma=0.001):
    pimgr = PersistenceImager()

    pimgr.pixel_size = pixel_size
    pimgr.kernel_params = {'sigma': sigma}

    pimgr.fit(h1)
    imgs = pimgr.transform(h1)

    plot_PIs(imgs, pimgr)

def plot_PIs(imgs, pimgr):
    plt.figure(figsize=(20, 4))
    
    for i in range(4):
        ax = plt.subplot(240+i+1)
        pimgr.plot_image(imgs[i], ax)
        plt.title("PI of Human Text")
    
    for i in range(4):
        ax = plt.subplot(240+i+5)
        pimgr.plot_image(imgs[-(i+1)], ax)
        plt.title("PI of AI Text")