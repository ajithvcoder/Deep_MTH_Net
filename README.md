## Topic : Person Attribute Recognition

**Objective** : To improve the base line code and achieve better results than current SOTA(please see this paper for the accuracy that we have to achieve "Rethinking of Pedestrian Attribute Recognition:
Realistic Datasets and A Strong Baseline" - https://arxiv.org/pdf/2005.11909.pdf )

### Important links :

**Base line code** - https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition

**Base line codes paper** - "Rethinking of Pedestrian Attribute Recognition: Realistic Datasets and A Strong Baseline" https://arxiv.org/pdf/2005.11909.pdf

**All Datasets link** - https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List

**Literature survey** - All papers related to it - https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List

**Another base line code (but its not working with Rapv2 dataset)** - https://github.com/dangweili/pedestrian-attribute-recognition-pytorch





**Current Accuracies:**
(Taken from https://arxiv.org/pdf/2005.11909.pdf)
![Types of models](baselinepaper.PNG)


**Methods or different models available :**
Example :
![Types of models](methods.PNG)
https://arxiv.org/pdf/2005.11576.pdf

**Datasets that we are going to focus on :**
PA100k 
RAPv2
PETA
RAPv1
The school of AI dataset - will send it later 
Market 1501 (optional)



**Project Schedule :** (Even if planned schedule didnt work its not a problem we can try to achieve it else we can reschedule)

Since we have a base line code and datasets already we need to improve the model alone thats enough . Each one can take a dataset and work on improving a model for that if any one gets a improvement in any one dataset we can proceed with that model .


Current baseline is on Resnet50 so we can think of using inception ,densenet or any other suitable network .

Literature survey 4 days 

I think 12 days is enought for find a new model . 
so with in  August 2 lets try to find a model that better than current model

So in august month 1- 15 we can write our paper .

July 17 - July 20 - Literature survey

July 21 - Aug 2 - Model preparation

Aug 2 - Aug 20 - Training part

Aug 20 - Aug 30 - paper writing

**Accuracy to be achieved :**

Name   - Precision

Pa100k - 89.41

Rapv2- 81.99

PETA - 86.99

Rapv1 - 82.84

MARKET-1501 - (optional)

TSAI - (Any thing is okay as along as above is satisfied)

**Works done :**

####### Coding part starts ###########################

19-07-2020 - Attached the notebooks to run the file on pa100k and rapv2 dataset with baseline code 

26-07-2020 - Added Vgg models code to run pa100k - (Vgg_valencebond_PA100k_notebook (1).ipynb)

30-07-2020 - Added Densenet models code to run pa100k - (Dense_net_PA100k_notebook.ipynb)  - modified the input channel and batch size

30-07-2020 - Added Mobilenet models code to run pa100k - (Mobile_net_PA100k_notebook.ipynb) - modified input channel

30-07-2020 - Added MNAS net - (MNAS_net_PA100k_notebook.ipynb)

30-07-2020 - Added Squeeze net - (Squeze_net_PA100k_notebook.ipynb)

30-07-2020 - Added Alex net - (Alexnet_net_PA100k_notebook.ipynb)

30-07-2020 - Added Inception net - (Inception_net_PA100k_notebook.ipynb)

30-07-2020 - Added SE_resnet net - (se_resnet_net_PA100k_notebook.ipynb)

######### Coding part Completed ###########################

Authors of the base paper have not released original pkl file or dataset so we are going with existing methods

######### Hyper tunning and training part Start ###########################

### Pa100k 

Dataset link - 

| Model name | Ckpt file | logs file | png file of final accuracy | epochs trained | number of trainval and test data |
| --- | --- | --- | --- | --- | --- | 
| Resnet50
| Densenet121
| Alexnet
| mnasnet
| shufflenetv2
| squeezenet
| vgg
| inception (optional if it works)



### Rapv2 

Dataset link - 

| Model name | Ckpt file | logs file | snap -png file of final accuracy | epochs trained | number of trainval and test data |
| --- | --- | --- | --- | --- | --- | 
| Resnet50
| Densenet121
| Alexnet
| mnasnet
| shufflenetv2
| squeezenet
| vgg
| inception (optional if it works)



### PETA

Dataset link - 

| Model name | Ckpt file | logs file | png file of final accuracy | epochs trained | number of trainval and test data |
| --- | --- | --- | --- | --- | --- | 
| Resnet50
| Densenet121
| Alexnet
| mnasnet
| shufflenetv2
| squeezenet
| vgg
| inception (optional if it works)





