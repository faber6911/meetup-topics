<h1>Scripts description</h1>

In order to reproduce the project pipeline, after having installed the requirements via

> pip install requirements.txt

the needed scripts are described below and are expected to be executed in the following order:
<ul>
<li><strong>Data_Exploration_and_Preprocessing.ipynb</strong> : exploration of the initial dataset and preprocessing operations</li>

<li><strong>LDA_unsupervised_clusterization_after_*.ipynb</strong>: LDA analysis for the stopwords list expansion with  our "badwords" (an example of LDA-viz dashboard is the file "lda.html")</li>

<li><strong>Text_Classification_Bag_of_Words_Exploration.ipynb</strong>: initial exploration and experiments with BOW approach</li>

<li><strong>Text_Classification_Bag_of_Words_Exploration_SVD_analysis.ipynb</strong>: sparsity analysis and optimal max_feature cut-off threshold</li>

<li><strong>Text_Classification_Bag_of_Words_Exploration_lemm_vs_stemm_rf.ipynb</strong>: comparison among preprocessing methods (lemmatization vs stemming vs same + badwords removal) for BOW methods w/ Random Forest</li>

<li><strong>Text_Classification_Bag_of_Words_CV.ipynb</strong>: final CV results w/ both RF and NN over the best selected methods</li>
<li><strong>Text_Classification_Word_Embedding.ipynb</strong>: comparison among preprocessing methods (lemmatization vs stemming vs same + badwords removal) for Word Embedding methods w/ Random Forest and Neural Network together w/ the final CV results w/ both RF and NN over the best selected methods</li>
<li><strong>Best_Model_Optimization.ipynb</strong>: Bayesian optimization for the best model combined with the best preprocessing and text representation</li>
</ul>
