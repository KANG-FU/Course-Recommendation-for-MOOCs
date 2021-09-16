project name: MOOC Course Recommendation
Group member: Chenkai Wang, Zhijie Song, Kang Fu

First, we have done the exploratory analysis of our dataset. 
You can follow the instrustions in the Exploratory_Analysis.ipynb

Then we have implemented 3 models: baseline (non-peronalized recommendation), Bayesian Personal Ranking (BPR) and Self-Attentive Sequential Recommendation model (SASRec).

The baseline model is actually non-peronalized recommendation. We recommend the most 10 popular courses to every student.
The evaluation of this model is based on hit rate and NDCG. The test data is the most recent course that a student chooses.
This part is implemented in baseline.ipynb.

The BPR model is personalized recommendation. It can provide users with item recommendations of a ranked list of items. 
The ranked list of items is calculated from the users’implicit behavior.  BPR is based on matrix factorization.
We  split the dataset into two parts: training set and test set. 
The last item of each user is selected to be the test set.  All the remaining items are taken as the training set.
We randomly sampled n_batch users in each batch during training and the algorithm is optimized by stochastic gradient descent.
The evaluation of this model is based on hit rate and NDCG.  
This part is implemented in BPR.ipynb.

The SASRec models is in the folder called SASRec. 
data: The processed data, which was extracted from the original dataset. It consists of two columns: userid and itemid.The itemid has been added by 1 because the padding id is 0.
There are three python files.
main.py:  The main function to run the models
model.py: The definition of the models
utils: The utitlity functions for data partition, prediction and output
The SASRec models can be implemented by implementing "python main.py". You need to go to the folder called SASRec.
The expriments can be carried out by modifying the values of the correspoding variables.
After obtaining the csv file, we used the jupyter notebook in the folder "post_process" to do the analysis and plotting.
 
