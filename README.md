# cs-project-ml

Several distance-based learning algorithms, including our study topic TransD.

## Motivation

Pre-specified features often restrict performance of various algorithms. Distance-based features thus provide an alternative to perform learning,
especially in the situation where similarity relation is easier to get or analyze, such as computer vision, bioinformatics natural language processing etc.

## Goal

1. Transform data into a “neat” distribution, by pulling or pushing each pair of points.
2. Use simple distance-based algorithm to get the final prediction!
![data pulling](https://i.imgur.com/lznsj1b.png)

## TransD

Semi-supervised: Train unlabeled data with labeled data.

<pre>
for each <b>round</b>:
  determine label for unlabeled data
  for each pair of data:
    if(pass the <b>conditions</b>):
      <b>adjust</b> their distance
    if(data is <b>neat enough</b>): end
</pre>
        
**round**: maximum of 20 rounds

**conditions**:
  * 𝑐𝑖, 𝑐𝑗 are calculated in the Bayesian KNN.
  * If random 𝑟 >= 𝜉𝑖𝑗 , transform to new distance. else keep it.
  
**neat enough**:
  * Consensus of 1-nn and 1-mi algorithm.

**adjust**:  
![adjust formula](https://i.imgur.com/DtQqNkb.png)

## Bayesian KNN

We have k hypothesis : 1-NN, 2-NN, …, K-NN.  
![Bayesian](https://i.imgur.com/GwsIf7Y.png)

## Linear Transform Approximation

![Inverse](https://i.imgur.com/TyNdvnI.png)
* Our model becomes a single linear transform matrix 𝑇!

## Improve and Experiments

### **Other transformation approximation**:  
Use feature space extension method. Result for quadratic transformation:  
**Linear**  
![Linear Result](https://i.imgur.com/KmSL2ZU.png)  
**Quadratic**  
![Quadratic Result](https://i.imgur.com/skoAwox.png)  
*Some significant good, some significant bad. We can treat different transformation as learning parameter, tune for a specific dataset.*

### **Clustering Preprocessing**:  
* ***Improving accuracy***:  
Result: We can increase accuracy in some dataset using clustering preprocessing, however, the overhead time isn’t worthy.  
* ***Compressing data***:  
Result: Compress unlabeled data into 1/5 or even 1/10 with same accuracy (No significant bad), thus saving a lot of time performing TransD.  

### **Randomness Adjustment**:  
Randomly return class based on the weight.
The randomness will decrease after every iteration.  
Result: not significant, need more experiments!

## Further Issue
1. We need more experiments on big data.
2. Further improve time and space complexity.
3. Implement the algorithm on CUDA (run on GPU).
4. Other ways to compress data—Fewer data but higher dimension?

## Reference
Yuh-Jyh Hu, Min-Che Yu, Hsiang-An Wang, and Zih-Yun Ting, “A Similarity-Based Learning Algorithm Using Distance Transformation,” IEEE TKDE., vol. 27, no. 6, pp. June. 2015.
