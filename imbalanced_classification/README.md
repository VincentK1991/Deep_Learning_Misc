# Imbalanced Classification Training with Weighted Loss Function

## Challenge

imbalanced classification creates a challenge for prediction, because the model tends to maximize performance of the major class and ignore the error of minor class. The accuracy score can look good simply because the base rate is high. The problem is severe when the class distribution is highly skewed as found in many real-world problems such as fraud detection, spam detection, etc. 

There are many ways to handle the problem such as

- Resampling the data. Either over-sampling the minor class or under-sampling the major class. The idea is to make sure that the predictive model is trained on a balanced data.

- Penalizing the mistakes on the minor class. The idea is that by imposing higher cost on the mistakes on the minor class, we make the model more sensitive to the imbalanced distribution in the data.

In this work, I choose the second approach. I will apply it to solve a well-known credit card fraud detection dataset.

---

## About the dataset

This dataset is a good tutorial for dealing with highly imbalanced data. We have 284807 transaction samples. Of these 492 are fraudulent (0.1 %). Besides time and amount of transaction, we have 28 features which doesn't really represent anything physical, but are features resulting from PCA transformation. This is done for privacy protection. I split the dataset 9:1 ratio for training and validation.


## Model

This model is a plain vanilla feed forward multi-layer perceptron. I use Pytorch implementation which is quick and easy. I add dropout layer after each hidden layer. But for the most part, very simple model for a simple task.


## Loss Function

The real work here is in the loss function, which is weighted according to the class distribution. Since the class distribution between the good vs fraud transaction is 0.9982 and 0.0018; we apply the opposite weight to the binary classification loss. Now for each error of the fraud transaction class (i.e. missing the fraud class), it is weighted about 555 times more than the error coming from the good transaction class. By doing this, we create the objective function that is equally sensitive to the good class and the fraud class.

This little tweaking is very easy to do in pytorch, because we can just multiply the weight directly to the loss before backward propagation.

## Result

We monitor 4 measurements over the course of training

1. Weighted binary classification loss. this is the objective function the model trying to minimize.
2. Precision. ratio of true positive over total predicted positive. In the beginning, we see this score low because the true positive is small, while initially the model probably performs at chance level. The score improves very quickly as the model is trained.
3. Recall. ratio of true positive over total positive (actual fraud).
4. F1 score. This is the harmonic means between precision and recall score.



