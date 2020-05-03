import numpy as np 
import matplotlib.pyplot as plt 

def load_data(filename):
    return np.loadtxt(filename)


def euclidian(x,y):
    return np.linalg.norm(x-y)


def Draw(centroids,min_position):
    colors = ['r','g']
    fig, ax = plt.subplots() 
    dataset = load_data('./k-means_clustering/data2.txt')
#------------------------------------------------------------------------------------------
#画出几组不同的点
    for index in range(dataset.shape[1]):
        mid_position = [i for i in range(len(min_position)) if min_position[i] == index]
        for ele in mid_position:
            ax.plot(dataset[ele][0],dataset[ele][1],(colors[index]+'o'))

#------------------------------------------------------------------------------------------
# 画出中心点的变化过程   
    points = []
    for index, mid_points in enumerate(centroids):
        for inner, content in enumerate(mid_points):
            if index == 0:
                points.append(ax.plot(content[0],content[1],'bo')[0])
            else:
                points[inner].set_data(content[0],content[1])
                print("centroids {} {}".format(index,content))
                plt.pause(0.8)



def Kmeans(k, elision=0, distance='euclidian'):
    centroids = []
    if distance == 'euclidian':
        dist_method = euclidian 
    dataset = load_data('./k-means_clustering/data2.txt')
    num_instances, num_features = dataset.shape
    samples = dataset[np.random.randint(0,num_instances-1,size=k)]
    centroids.append(samples)
    old_samples = np.zeros(samples.shape)
    min_position = np.zeros((num_instances,1))
    dist = dist_method(samples,old_samples)
    num = 0
    while dist > elision:
        num += 1
        dist = dist_method(samples,old_samples)
        old_samples = samples
#-----------------------------------------------------------------------------------------
#calculate the distances between samples and dateset and record the min value's position
        for index, instances in enumerate(dataset):
            dist_list = np.zeros((k,1))
            for numbers, element in enumerate(samples):
                dist_list[numbers] = dist_method(instances,element)
            min_position[index,0] = np.argmin(dist_list)
        tem_result = np.zeros((k, num_features))
#-----------------------------------------------------------------------------------------
#calculate the mean value of different groups and update the samples
        for index_samples in range(len(samples)):
            mid_position = [i for i in range(len(min_position)) if min_position[i] == index_samples]
            sample = np.mean(dataset[mid_position], axis=0)
            tem_result[index_samples, :] = sample
        samples = tem_result
        centroids.append(tem_result)

    return samples, centroids, min_position    


if __name__ == "__main__":

    samples, centroids, min_position = Kmeans(2)
    Draw(centroids, min_position)


    