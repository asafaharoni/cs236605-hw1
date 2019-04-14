r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
By increasing k we get more examples that are similar to the test object. This can improve the performance, but when we
use big k values, which include farther neighbors - we hurt the generalization, and increase the bias. If we look at the
limit, we can see that when k=train_size, we get the same value (the most common label) for any test object, which means
generalization failed. 
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
$\Delta$ is suppose to be the permissible margin between two classes samples.
In our case we allow soft margins so in case we choose a bigger $\Delta$ then the weights will be bigger as well.
We note that also $\Delta$ do not takes place in the process of weights update because the gradient of $L(W)$ is not dependent on $\Delta$.
"""

part3_q2 = r"""
1. The model learns the main regions that are constantly used to mark each number. 
In the figure "0" one never uses the inside region of the number thus, this region is black.
It is noticeable that some figures that mark the number "1" are italic to the left and the others are italic to the right.
Some of the mistakes stemmed from a strange form of number writing or, from digits looking like other digits.

2. A KNN model predicts an answer based on few closest samples while the SVM model tries to find correct choice of weights based on the whole data set.
"""

part3_q3 = r"""
**Your answer:**
1. The learning rate is good.  
It goes down pretty quick together with the validation set loss and it seems to be converging after 30 epochs.
If the learning rate was lower, then it probably would not have reached convergence. 
If the learning rate was faster we would probably see a phenomenon of overfitting.

2. 
Based on the graph of the training and test set accuracy, would you say that the model is sightly overfitted to the training set.
The training set accuracy reach a higher value (~91.5%) while the test-set accuracy after training is 88.1%.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal pattern would look like a straight line on $e^{(i)} = y^{(i)} - \hat{y}^{(i)} = 0$, which would mean that 
there are no erros in prediction. We can see that as we proceeded with this part, we saw a decrease in the variance of
$e^{(i)}$, which meant the fitness was better, especially when we compare it to the top-5 features.
"""

part4_q2 = r"""
1. We are combing for the best lambda, without even knowing what scale to use. We can see a great variance in score over
different scales. to fine tune, we can add another level of cv, but there is no need.
2. The answer is the product of all values sets sizes for the params, times the number of folds:
$$
n_{folds} \times \prod_{values \in ranges} \left( len(values) \right)
$$
"""

# ==============
