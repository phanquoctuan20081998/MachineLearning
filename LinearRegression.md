
# Linear Regression

## Problem

The goal of this task is to predict the price of a house based on the number of rooms it has.

**Input:**  $x_p \in \mathbb{N}$ - the number of rooms in the house  
**Output:**  $y_p$ - the price of the house

## Dataset

We are given a dataset of $n$ data points: $X={(x_1, y_1),(x_2, y_2), \ldots ,(x_n, y_n)}$. Here, $x_1, x_2,\ldots, x_n$ represent the number of rooms, while $y_1, y_2,\ldots, y_n$ represent the corresponding house prices (**ground truth**).

## Solution

We can find a straight line to map the  **input**  and  **output**  variables. This line can be represented by the function $\hat{y}=wx + b$, where $\hat{y}$ is the predicted value, $w$ is the slope, and $b$ is the intercept.

We need to find the values of $w$ and $b$ that best fit our dataset.

## Loss function

To measure how well our model fits the data, we need to define a loss function that quantifies the difference between our predicted values and the ground truth. We use the  **mean squared error**  as our loss function:

$$\mathcal{L}(w, b)=\frac{1}{n} \sum_{i=1}^m\left(y_i-\hat{y}_i\right)^2$$

# Gradient descent

Our goal is to minimize the loss function $\mathcal{L}(w, b)$ by finding the values of $w$ and $b$ that result in the smallest possible value for the loss. We use gradient descent to optimize the values of $w$ and $b$:

$$ \left\{\begin{array}{l}w:=w-\alpha\frac{\partial \mathcal{L}(w, b)}{\partial w} \\ b:=b-\alpha\frac{\partial \mathcal{L}(w, b)}{\partial b}\end{array}\right. $$

Here, $\alpha$ is the learning rate, which controls how large the updates to $w$ and $b$ are on each iteration. We need to compute the partial derivatives of the loss function with respect to $w$ and $b$:

$$\begin{aligned} \frac{\partial \mathcal{L}(w, b)}{\partial w} & =\frac{1}{2 n} \sum_{i=1}^{n} 2\left(y_{i}-\hat{y}_{i}\right) \frac{\partial \hat{y}_{i}}{\partial w} \\ & =\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right) x_{i}\end{aligned}$$

$$\begin{aligned} \frac{\partial L(w, b)}{\partial b} & =\frac{1}{2 n} \sum_{i=1}^{n} 2\left(y-\hat{y}_{i}\right) \frac{\partial \hat{y}_{i}}{\partial b} \\ & =\frac{1}{n} \sum_{i=1}^{n}\left(y-\hat{y}_{i}\right)\end{aligned}$$

We can then use these partial derivatives in the update rule for $w$ and $b$ to iteratively minimize the loss function and find the best values for $w$ and $b$.
