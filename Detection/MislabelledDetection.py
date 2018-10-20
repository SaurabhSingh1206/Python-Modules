def wKNN(X,y,z,s,c):
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    nn=NearestNeighbors(n_neighbors=z)
    nn.fit(X)
    l=[]
    sigma=s*np.var((np.sqrt(np.sum(np.square(X),axis=1))),axis=0,dtype=np.float64)
    gamma=(1/(2*sigma*sigma))
    from sklearn.metrics.pairwise import rbf_kernel
    W=rbf_kernel(X,gamma=gamma)
    for i in range(0,X.shape[0]):
        knei=nn.kneighbors(X[i,:].reshape(1,-1),return_distance=False)
        den=0
        num=0
        for j in range(1,knei.shape[1]):
            o=knei[0,j]
            den=den+W[i,o]
            num=num+y[o]*W[i,o]
        y_hat=num/den
        delta=y[i]*y_hat
        l.append(delta)
    output=np.array(l)
    indices = [i for i in range(output.shape[0]) if output[i]<0]
    store = [output[i] for i in range(output.shape[0]) if output[i]<0]
    print('Fraction Mislabelling found in sample : '+str(len(indices)/X.shape[0]))
    item=int(c*len(indices))
    returning_value=np.argsort(store)[:item]
    result=[indices[returning_value[i]] for i in range(returning_value.shape[0])]
    return result
