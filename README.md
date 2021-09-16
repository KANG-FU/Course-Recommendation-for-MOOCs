# Course-Recommendation-for-MOOCs
Author: Kang Fu, Chenkai Wang, Zhijie Song
# Description
This is the project of EPFL CS-421 Machine Learning for Behavior Data. Our project is to conduct course recommendations based on the open MOOC site, XuetangX. Our goal is to predict the next course to recommend to each student, based on their needs or interests.
# Structure
Codebase: contains all the data and codes

Presentation.pptx: the slides for the final presentation

Report.pdf: the final report for our project
# Models
1. Non-Personalized Recommendation Model: 

This is our baseline that recommends courses based on courses' popularity. Since it is non-personalized, all the students will receive the identical course recommendations.

2. Bayesian Personal Ranking (BPR) Model: 

BPR is a classic method for learning personalized rankings from implicit feedback, based on matrix factorization.

3. The Self-attentive Sequential Recommendation Model (SASRec)}: 

In 2017, a new sequential model Transfomer in Natural Language Processing achieved remarkable performance and efﬁciency for machine translation tasks. Inspired by this method, we follow the experimental steps from Wang. We apply the self-attention mechanisms to sequential recommendation problems. From former research, the SASRec is proven to perform significantly better than MC/CNN/RNN-based sequential recommendation methods.

# Results
The SASRec model achieve the best performance among the three models.
