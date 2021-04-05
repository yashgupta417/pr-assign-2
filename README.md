# pr-assign-2

Applying k-means clustering on iris dataset. Used Elbow method to find out optimal K.

## To Run
To run the model locally, fist clone the repository using following command in your terminal.
```
https://github.com/yashgupta417/pr-assign-2.git
```
Then, open the `model.ipynb` file using jupyter notebook and press run.

##Code Summary

In this code snippet, we are reading the dataset
```python
df=pd.read_csv('./iris.data',header=0,names=['sepal_length','sepal_width','petal_length','petal_width','class'])
```

Now, splitting dataset into train and test
```python
#Splitting dataset in 70:30 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=66)
```

Training on different values of K
```python
#iterating over different values of K
for k in range(1,K):
    
    #applying K-means clustering
    kmeans=KMeans(n_clusters=k).fit(X_train)
    
    #getting train labels
    labels[k]=kmeans.labels_
    
    #generating mapping
    mapping=generate_mapping(labels[k],Y_train,k)
    
    #mapping training labels
    mapped_labels=map_labels(labels[k],mapping)
    
    #calculating training accuracy
    acc=accuracy_score(Y_train, mapped_labels)
    
    
    
    #evaluating test data
    Y_pred=kmeans.predict(np.array(X_test))
    
    #mapped test preds
    mapped_preds=map_labels(Y_pred,mapping)
    
    #calculating validation accuracy
    val_acc=accuracy_score(Y_test, mapped_preds)
    
    
    
    print(f"k={k} Training Accuracy: {acc}\t Validation Accuracy: {val_acc}")
    
    iner[k]=kmeans.inertia_
    
    models[k]=kmeans
    
    mappings[k]=mapping
```

Plotting inertia vs no_of_clusters curve
```python
plt.figure()
plt.plot(list(iner.keys()), list(iner.values()))
plt.xlabel("Number of cluster")
plt.ylabel("Error")
plt.show()
```

