# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

    - Author: Luis Melgar
    - Model date: 2021-11-10
    - Model version: 1.0.0
    - Model Type: Random Forest Classifier

## Intended Use

    - Udacity MLOps Course - module 4.

## Training Data

    - The model was trained on 80% of the data. Some pre-process steps have been implemented.

## Evaluation Data

    - The test (evaluation) set includes the remaining 20% of the data

## Metrics



## Ethical Considerations

    - No ethical issues or intrinsic biases have been considered as this project serves educational purposes only.  

## Caveats and Recommendations

    - One thing to consider are categorical variables that posses large cardinality, as the one hot encoding method is not the best approach in that case. Also, some categories might have very few examples attached to them and it could make sense to combine them with others to create meaningful and larger groups.