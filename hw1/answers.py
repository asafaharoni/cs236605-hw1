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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
