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
| Resnet50 | [drive](https://drive.google.com/file/d/1_F9gPOwwPrQUDfJyfAFVNnwFfiOXSe-M/view?usp=sharing) | [drive](https://docs.google.com/document/d/13YE5wJRlxgLH6xaUtkjdYl2kd2tzSM8-E0qy602z7Xg/edit?usp=sharing)| ma: 0.7992, Acc: 0.7853, Prec: 0.8718, Rec: 0.8667 | 50 | trainval set: 90000, test set: 10000, attr_num : 26 |
| Densenet121 | [drive](https://drive.google.com/file/d/1uR6weiL5cy7-GTtZfQJ6MeMuJlETBtHL/view?usp=sharing) | [drive](https://docs.google.com/document/d/1BAaB6z8y5hgv0d1bO2qAohJMlD6sJu-u968DpLcbcw4/edit?usp=sharing)|  ma: 0.7579, Acc: 0.7448, Prec: 0.8460, Rec: 0.8358 | 50 | trainval set: 90000, test set: 10000, attr_num : 26 |
| Alexnet | [drive](https://drive.google.com/file/d/16xo6nyM7sZ8rZEKN4rtq2lO0LJ2CVMZi/view?usp=sharing) | [drive](https://docs.google.com/document/d/10_MCfFhN2Lkbj2oGnO0yyyPFgz9X0gsKM26vfKiNzq8/edit?usp=sharing) | ma: 0.7584, Acc: 0.7472, Prec: 0.8461, Rec: 0.8384 | 50 | trainval set: 90000, test set: 10000, attr_num : 26 |
| mnasnet
| shufflenetv2
| squeezenet | [drive](https://drive.google.com/file/d/1cOuE9a5PxISjVtNeQGPLlwovjysePFeP/view?usp=sharing) | [drive](https://docs.google.com/document/d/19F-q4nXUmXywYa-CZUvI5Tu6kYjwkE-_fqRbaUmP-QA/edit?usp=sharing) | ma: 0.7203, Acc: 0.7016, Prec: 0.8234, Rec: 0.8039 | 50 | trainval set: 90000, test set: 10000, attr_num : 26 |
| vgg | [drive](https://drive.google.com/file/d/1F4DMX5tuqfwTig-L7J0PWZUd2JIMHfBn/view?usp=sharing) | [drive](https://docs.google.com/document/d/1m686ZghKw0QIqgg_pErDXuPmtO1ayuZKuIxKYKi78WU/edit?usp=sharing) |  ma: 0.7711, Acc: 0.7333, Prec: 0.8391, Rec: 0.8303 | 50 | trainval set: 90000, test set: 10000, attr_num : 26 |
| inception (optional if it works)
| seresnet 


### Rapv2 

Dataset link - 

| Model name | Ckpt file | logs file | snap -png file of final accuracy | epochs trained | number of trainval and test data |
| --- | --- | --- | --- | --- | --- | 
| Resnet50 | [drive](https://drive.google.com/file/d/17Mxm90621He58LzAoQyERdJ98NJ6C_cJ/view?usp=sharing) | [drive](https://docs.google.com/document/d/1rM57imAnr_PZAhINtqFFlFnm0Zar9T3qSpc1mSJSwmA/edit?usp=sharing) | ma: 0.7862, Acc: 0.6640, Prec: 0.7794, Rec: 0.7974 | 50 | trainval set: 67943, test set: 16985, attr_num : 54 |
| Densenet121 | [drive](https://drive.google.com/file/d/1Tvw98ZBc73mcLLuwUpr6debnfKlCTSs-/view?usp=sharing)| [drive](https://docs.google.com/document/d/1vLNnuoh7uIXG7zLeM6TW_BEZE_35DlnizPPd2x2OQiA/edit?usp=sharing) |  ma: 0.7666, Acc: 0.6488, Prec: 0.7729, Rec: 0.7801 | 50 |
trainval set: 67943, test set: 16985, attr_num : 54 |
| Alexnet | [drive](https://drive.google.com/file/d/1g3C2eMJNGqSDFV36khL4HQO2QQKAFZPe/view?usp=sharing) | [drive](https://docs.google.com/document/d/1cFnSCNkDgtSHUXQxLRU1acf4oySrV0uUyFiBoKUzQ3Q/edit?usp=sharing)| ma: 0.7252, Acc: 0.6150, Prec: 0.7618, Rec: 0.7431, F1: 0.7468| 50| trainval set: 67943, test set: 16985, attr_num : 54 |
| mnasnet
| shufflenetv2
| squeezenet | [drive](https://drive.google.com/file/d/1MRxfnImg82DKpgm7w48uCHbchByKUsu-/view?usp=sharing) | [drive](https://docs.google.com/document/d/1ny7gBsiQb9FIvm1BqHGDY68_cu2e1glqhraWUnC-7HM/edit?usp=sharing) | ma: 0.7407, Acc: 0.6270, Prec: 0.7569, Rec: 0.7674 | 50 | trainval set: 67943, test set: 16985, attr_num : 54
| vgg | [drive](https://drive.google.com/file/d/1XCUMdN0TjZJGUFOQ6A71cMYJvv0Uv9y6/view?usp=sharing) | [drive](https://docs.google.com/document/d/1fK3sj1Pos3tjVsjTggpSp0nK8uwleVlZ3YuywggpeW8/edit?usp=sharing)| ma: 0.7407, Acc: 0.6270, Prec: 0.7569, Rec: 0.7674, | 50 | trainval set: 67943, test set: 16985, attr_num : 54
| inception (optional if it works)
| seresnet 


### PETA

Dataset link - 


| Model name | Ckpt file | logs file | png file of final accuracy | epochs trained | number of trainval and test data |
| --- | --- | --- | --- | --- | --- | 
| Resnet50  | drive | drive | ma: 0.8529, Acc: 0.7912 | 50 | trainset -11400, test set: 7600
| Densenet121 | drive | drive | ma: 0.8113, Acc: 0.7465 | 50 | trainset -11400, test set: 7600
| Alexnet | drive | drive | ma: 0.7804,acc: 0.7109 | 50 | trainset -11400, test set: 7600
| mnasnet | drive | drive | ma: 0.7476,acc : 0.6688 | 50 | trainset -11400, test set: 7600
| shufflenetv2 | drive | drive | ma: 7919, Acc: 0.7234 | 50 | trainset -11400, test set: 7600
| squeezenet | drive | drive | ma: 0.6780, Acc: 0.5575 | 50 | trainset -11400, test set: 7600
| vgg | drive | drive | ma: 0.7836, Acc: 0.7057 | 50 | trainset -11400, test set: 7600
| inception (optional if it works) |
| se_resnet  | drive | drive | ma: 0.8129, Acc: 0.7448 | 50 | trainset -11400, test set: 7600
| mobilenet  | drive | drive | ma:0.8065, acc:0.7278 | 50 | trainset -11400, test set: 7600


Resnet results - https://drive.google.com/drive/u/1/folders/1-FlKmYoj7wE0TsAgEjjFUqzAl_6-u6Eu 
all notebooks - https://drive.google.com/file/d/1Nd4Zy2vFKdenHZM5V5docBwnwlhAQlOL/view?usp=sharing

### HA-13.5k-TSAI


Dataset link - 

Attributes - 27

| Model name | Ckpt file | logs file | snap -png file of final accuracy | epochs trained | number of trainval and test data |
| --- | --- | --- | --- | --- | --- | 
| Resnet50  | drive | drive | ma: 0.6868,Acc: 0.5437 | 50 | trainset -12150, test set: 1350
| Densenet121 | drive | drive | ma: 0.6647, Acc: 0.5271 | 50 | trainset -12150, test set: 1350
| Alexnet | drive | drive | ma: 0.6451, Acc: 0.5035 | 50 | trainset -12150, test set: 1350
| mnasnet | drive | drive | ma: 0.50, Acc:0.2980 | 50 | trainset -12150, test set: 1350
| shufflenetv2 | drive | drive | ma: 0.6533,Acc: 0.5086 | 50 | trainset -12150, test set: 1350
| squeezenet | drive | drive | ma: 0.5943,Acc: 0.4573 | 50 | trainset -12150, test set: 1350
| vgg | drive | drive | ma: 0.5928,Acc: 0.4677 | 50 | trainset -12150, test set: 1350
| inception (optional if it works) |
| se_resnet  | drive | drive | ma: 0.6905,Acc: 0.5427 | 50 | trainset -12150, test set: 1350
| mobilenet  | drive | drive | ma: 0.6572,Acc: 0.5026 | 50 | trainset -12150, test set: 1350

All notebooks/drive links - https://drive.google.com/file/d/1Nd4Zy2vFKdenHZM5V5docBwnwlhAQlOL/view?usp=sharing

TSAI Accuracy :

Alexnet :

TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.8748148148148148
Image quality accuracy  -  0.5170370370370371
age accuracy  -  0.4274074074074074
weight accuracy  -  0.6296296296296297
carryingbag accuracy  -  0.6585185185185185
footwear accuracy  -  0.6837037037037037
emotion accuracy  -  0.697037037037037
bodypose accuracy  -  0.845925925925926


Densenet :


TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.9044444444444445
Image quality accuracy  -  0.534074074074074
age accuracy  -  0.43851851851851853
weight accuracy  -  0.6703703703703704
carryingbag accuracy  -  0.6644444444444444
footwear accuracy  -  0.6844444444444444
emotion accuracy  -  0.6777777777777778
bodypose accuracy  -  0.86

Mobilenet:

TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.8837037037037037
Image quality accuracy  -  0.5111111111111111
age accuracy  -  0.42962962962962964
weight accuracy  -  0.6037037037037037
carryingbag accuracy  -  0.6533333333333333
footwear accuracy  -  0.662962962962963
emotion accuracy  -  0.6688888888888889
bodypose accuracy  -  0.8540740740740741

Se_resnet :

TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.9340740740740741
Image quality accuracy  -  0.5274074074074074
age accuracy  -  0.4488888888888889
weight accuracy  -  0.6362962962962962
carryingbag accuracy  -  0.7266666666666667
footwear accuracy  -  0.6881481481481482
emotion accuracy  -  0.6681481481481482
bodypose accuracy  -  0.8718518518518519

Resnet :

TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.9244444444444444
Image quality accuracy  -  0.542962962962963
age accuracy  -  0.44296296296296295
weight accuracy  -  0.6451851851851852
carryingbag accuracy  -  0.7340740740740741
footwear accuracy  -  0.6977777777777778
emotion accuracy  -  0.6644444444444444
bodypose accuracy  -  0.88

Shuffle net:

TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.88
Image quality accuracy  -  0.5185185185185185
age accuracy  -  0.42074074074074075
weight accuracy  -  0.6444444444444445
carryingbag accuracy  -  0.6377777777777778
footwear accuracy  -  0.6807407407407408
emotion accuracy  -  0.705925925925926
bodypose accuracy  -  0.8503703703703703

Squeezenet :

TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.7814814814814814
Image quality accuracy  -  0.5244444444444445
age accuracy  -  0.3985185185185185
weight accuracy  -  0.6340740740740741
carryingbag accuracy  -  0.6125925925925926
footwear accuracy  -  0.6592592592592592
emotion accuracy  -  0.7466666666666667
bodypose accuracy  -  0.7348148148148148

Vgg:

TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.8044444444444444
Image quality accuracy  -  0.5370370370370371
age accuracy  -  0.40296296296296297
weight accuracy  -  0.6296296296296297
carryingbag accuracy  -  0.6177777777777778
footwear accuracy  -  0.6748148148148149
emotion accuracy  -  0.7518518518518519
bodypose accuracy  -  0.7333333333333333

MNASnet :

TSAI accuracy metric
shape gt label (1350, 27)
pred prob shape (1350, 27)
gender accuracy  -  0.43555555555555553
Image quality accuracy  -  0.5481481481481482
age accuracy  -  0.4192592592592593
weight accuracy  -  0.6525925925925926
carryingbag accuracy  -  0.32222222222222224
footwear accuracy  -  0.43037037037037035
emotion accuracy  -  0.7622222222222222
bodypose accuracy  -  0.20296296296296296


papers to read before writing :

For ajith :
An Attention-Based Deep Learning Model for Multiple Pedestrian
Rethinking of Pedestrain Attribute Recognition

For hammad:
skiming - Clothes key point localization and attribute recognition via prior knowledge
Hierarchial Feature Embedding for Attribute recogntion
Texture and shape biased two-steam networks for clothing classification and attribute recognition
Rethinking of Pedestrain Attribute Recognition


Draft paper topics :

Abstract   - Ajith
1. Introduction - Hammad
2. Related Work - Hammad
3. Proposed method - Hammad
    3.1 -
    3.2 - loss
4. Experiments - Ajith
    4.1 Datasets
    4.2 Evaluations
    4.3 Implementation Details
    4.4 Experiments on pedestrian attribute dataset
5. Conclusion - Ajith
6. Acknowledgements - Ajith 
