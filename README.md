# Seedtag NLP test

In order to run the code you have to follow next steps.

1.- First go to to the path with the code:
```
cd codetest/NLP/code
```

2.- To train the model with data in directory dataset you have to run the following sentence:
```
python train.py '../../dataset'
```

3.- To test the model and classify new documents you have to run the following sentence:
```
python classify.py model_path document_path_1 document_path_2 document_path_3 ...
```
Here you can fin an example:
```
python classify.py '../../pickle_model/predictive_model.pkl' '../../dataset/exploration/59497' '../../dataset/exploration/59873' '../../dataset/intelligence/176870' '../../dataset/politics/100521' '../../dataset/weapons/53328'

```

For further information over the analysis and the approach of the test you can take a look to the notebook in notebook directory
