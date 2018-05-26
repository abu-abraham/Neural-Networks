# ReadMe


> This is a relatively old concept. Not used in most neural networks now. However, concepts such as DropOut is very similar to this. 

Determining the exact number of hidden neurons by looking at a data set is a close to impossible task. There is no method to clearly identify and determine an exact number of hidden units for an optimal network. Various techniques such as relevance, contribution, sensitivity, distinctiveness and badness has been used to prune neural networks. Of the aforementioned techniques, this repository contains the distinctiveness technique. The method of distinctiveness is applied by determining the similarity between each hidden unit with all other hidden units. Based on the results of distinctiveness, the network is then pruned to remove the redundant hidden neurons[1]  This form of pruning resulted in a network with fewer hidden neurons. Since only similar hidden neurons were removed, no further training on the pruned network is required.

To remove the hidden layers, without reconstructing the network, we remove only those layers deemed as similar *(model.py)*.  As mentioned earlier, this is not a very efficent method as it is a computationally heavy process. However, it was a good place to get into the pytorch basics. With this method, prediction is made on the Diabetic Retinopahy dataset. 

[1] T. Gedeon et al, "Network Reduction Techniques”