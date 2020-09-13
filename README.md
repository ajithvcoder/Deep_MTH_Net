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

HA-TSAI Accuracy on Strong base line (Attenation based):

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

HA-TSAI on TSAI_net :
age_output_acc : 44.5565
bag_output_acc : 68.0444
emotion_output_acc : 70.6149
footwear_output_acc : 67.7923
gender_output_acc : 91.1794
image_quality_output_acc : 60.2319
pose_output_acc : 87.5
weight_output_acc : 65.2722


### HA-TSAI on Dangweili's Deep Multi attribute recognition network :

https://github.com/dangweili/pedestrian-attribute-recognition-pytorch.git 
Epochs : 50 
TSAI accuracy metric
('shape gt label', (2700, 27))
('pred prob shape', (2700, 27))
('gender', 'accuracy  - ', 0.9274074074074075)
('Image quality', 'accuracy  - ', 0.5648148148148148)
('age', 'accuracy  - ', 0.37333333333333335)
('weight', 'accuracy  - ', 0.644074074074074)
('carryingbag', 'accuracy  - ', 0.7122222222222222)
('footwear', 'accuracy  - ', 0.6688888888888889)
('emotion', 'accuracy  - ', 0.6133333333333333)
('bodypose', 'accuracy  - ', 0.8814814814814815)

Evaluation on test set:
Label-based evaluation: 
 mA: 0.6678
Instance-based evaluation: 
 Acc: 0.5361, Prec: 0.6975, Rec: 0.6676, F1: 0.6822


### HA-TSAI on Visuval attenation model :
https://github.com/hguosc/visual_attention_consistency.git

Epochs : 30 
epoch: 29, train step: 0, Loss: 9.320457
	cls loss: 7.4327;	flip_loss_l: 0.4751 flip_loss_s: 0.4202;	scale_loss: 0.9925
epoch: 29, train step: 200, Loss: 10.829878
	cls loss: 8.8861;	flip_loss_l: 0.4748 flip_loss_s: 0.4420;	scale_loss: 1.0269
Epoch time:  424.5598879999998
Saving model to /content/gdrive/My Drive/vi_ac_results//model_resnet50_epoch29.pth
testing ... 
prediction finished ....
>>>>>>>>>>>>>>>>>>>>>>>> Average for Each Attribute >>>>>>>>>>>>>>>>>>>>>>>>>>>
APs
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1.]
precision scores
[0.94046592 0.92587859 0.61413778 0.51890034 0.44923077 0.44927536
 0.49478881 0.3823934  0.368      0.63265306 0.69987119 0.34090909
 0.4953271  0.27868852 0.72125    0.60869565 0.74972973 0.81459566
 0.40291262 0.67909801 0.32098765 0.3        0.74576271 0.
 0.94910941 0.90832396 0.76818951]
recall scores
[0.91905565 0.95706737 0.86768448 0.32683983 0.43843844 0.26552463
 0.80177778 0.4082232  0.16312057 0.2137931  0.93409742 0.09375
 0.3392     0.1        0.64831461 0.21052632 0.89831606 0.79652845
 0.3487395  0.65964617 0.08813559 0.13003096 0.97384306 0.
 0.86342593 0.95618709 0.78411054]
f1 scores
[0.92963753 0.94121468 0.71921962 0.40106242 0.443769   0.33378197
 0.6119403  0.39488636 0.22604423 0.31958763 0.80019637 0.14705882
 0.40265907 0.14718615 0.68284024 0.31284916 0.81732469 0.80546075
 0.37387387 0.66923077 0.13829787 0.18142549 0.84467714 0.
 0.90424242 0.93164119 0.77606838]

AP: 1.0
F1-C: 0.528006521970672
P-C: 0.5762657355237963
R-C: 0.5254213600261832

>>>>>>>>>>>>>>>>>>>>>>>> Overall Sample-Label Pairs >>>>>>>>>>>>>>>>>>>>>>>>>>>
F1_O: 0.7160387805374133
P_O: 0.6994569296657689
R_O: 0.7334259259259259


mA scores
[0.93674051 0.92962981 0.55396635 0.63214199 0.63121529 0.59872738
 0.60850794 0.59291794 0.56522447 0.60337405 0.60108012 0.54116634
 0.61755181 0.54130435 0.76255509 0.59786792 0.74889852 0.84173987
 0.61906399 0.70754945 0.53263329 0.54440126 0.52343838 0.49980814
 0.92730379 0.89748029 0.85975918]
mean mA
0.6672610188954403


test time:  3.703718000000663
Current learning rate is: 0.00000
epoch: 30, train step: 0, Loss: 10.257829
	cls loss: 8.4541;	flip_loss_l: 0.4635 flip_loss_s: 0.3773;	scale_loss: 0.9629
epoch: 30, train step: 200, Loss: 11.067913
	cls loss: 9.1150;	flip_loss_l: 0.5018 flip_loss_s: 0.4119;	scale_loss: 1.0392
Epoch time:  424.0990129999991
Saving model to /content/gdrive/My Drive/vi_ac_results//model_resnet50_epoch30.pth
testing ... 
prediction finished ....
>>>>>>>>>>>>>>>>>>>>>>>> Average for Each Attribute >>>>>>>>>>>>>>>>>>>>>>>>>>>
APs
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1.]
precision scores
[0.92737896 0.94044503 0.61513761 0.51428571 0.44622093 0.45547945
 0.50028986 0.3837037  0.38095238 0.64285714 0.70565046 0.35294118
 0.49673203 0.24675325 0.6782988  0.57407407 0.77333333 0.82692308
 0.40731707 0.67266776 0.30952381 0.31914894 0.74450019 0.
 0.94264339 0.90985915 0.77624785]
recall scores
[0.93676223 0.94914135 0.85305344 0.35064935 0.46096096 0.28479657
 0.76711111 0.38032305 0.17021277 0.24827586 0.92320917 0.1125
 0.3648     0.11176471 0.6988764  0.23308271 0.86398964 0.78784957
 0.35084034 0.69250211 0.08813559 0.13931889 0.97032193 0.
 0.875      0.95618709 0.77892919]
f1 scores
[0.93204698 0.94477318 0.71481876 0.41698842 0.4534712  0.35046113
 0.60561404 0.3820059  0.23529412 0.35820896 0.7999007  0.17061611
 0.42066421 0.15384615 0.68843387 0.3315508  0.81615173 0.80691358
 0.37697517 0.68244085 0.13720317 0.19396552 0.84254204 0.
 0.90756303 0.93244804 0.77758621]

AP: 1.0
F1-C: 0.5345364381373553
P-C: 0.5756801905042619
R-C: 0.5314294078065425

>>>>>>>>>>>>>>>>>>>>>>>> Overall Sample-Label Pairs >>>>>>>>>>>>>>>>>>>>>>>>>>>
F1_O: 0.7168359843428289
P_O: 0.7010222595919813
R_O: 0.7333796296296297


mA scores
[0.93964928 0.93620642 0.55462955 0.64114237 0.63682266 0.60679596
 0.60990476 0.58714023 0.56897735 0.62022404 0.60977212 0.54975394
 0.62673735 0.5444199  0.76794649 0.60709189 0.76287717 0.84251167
 0.62078887 0.71406335 0.53200958 0.54946592 0.52027333 0.49980814
 0.93242945 0.89896397 0.85881867]
mean mA
0.671823127569605


test time:  3.8747050000001764
Saving model to /content/gdrive/My Drive/vi_ac_results//model_resnet50_final.pth
testing ... 
prediction finished ....
>>>>>>>>>>>>>>>>>>>>>>>> Average for Each Attribute >>>>>>>>>>>>>>>>>>>>>>>>>>>
APs
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1.]
precision scores
[0.92737896 0.94044503 0.61513761 0.51428571 0.44622093 0.45547945
 0.50028986 0.3837037  0.38095238 0.64285714 0.70565046 0.35294118
 0.49673203 0.24675325 0.6782988  0.57407407 0.77333333 0.82692308
 0.40731707 0.67266776 0.30952381 0.31914894 0.74450019 0.
 0.94264339 0.90985915 0.77624785]
recall scores
[0.93676223 0.94914135 0.85305344 0.35064935 0.46096096 0.28479657
 0.76711111 0.38032305 0.17021277 0.24827586 0.92320917 0.1125
 0.3648     0.11176471 0.6988764  0.23308271 0.86398964 0.78784957
 0.35084034 0.69250211 0.08813559 0.13931889 0.97032193 0.
 0.875      0.95618709 0.77892919]
f1 scores
[0.93204698 0.94477318 0.71481876 0.41698842 0.4534712  0.35046113
 0.60561404 0.3820059  0.23529412 0.35820896 0.7999007  0.17061611
 0.42066421 0.15384615 0.68843387 0.3315508  0.81615173 0.80691358
 0.37697517 0.68244085 0.13720317 0.19396552 0.84254204 0.
 0.90756303 0.93244804 0.77758621]

AP: 1.0
F1-C: 0.5345364381373553
P-C: 0.5756801905042619
R-C: 0.5314294078065425

>>>>>>>>>>>>>>>>>>>>>>>> Overall Sample-Label Pairs >>>>>>>>>>>>>>>>>>>>>>>>>>>
F1_O: 0.7168359843428289
P_O: 0.7010222595919813
R_O: 0.7333796296296297


mA scores
[0.93964928 0.93620642 0.55462955 0.64114237 0.63682266 0.60679596
 0.60990476 0.58714023 0.56897735 0.62022404 0.60977212 0.54975394
 0.62673735 0.5444199  0.76794649 0.60709189 0.76287717 0.84251167
 0.62078887 0.71406335 0.53200958 0.54946592 0.52027333 0.49980814
 0.93242945 0.89896397 0.85881867]
mean mA
0.671823127569605


papers to read before writing :

https://github.com/ajithvallabai/person_attribute_recognition/tree/master/basepaperstobe_read

if we want further we can refer papers published in 2020 and 2019 from here 
https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List


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
