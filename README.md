
# Linear Regression

## Problem

The goal of this task is to predict the price of a house based on the number of rooms it has.

**Input:** $x_p \in \mathbb{N}$ - a natural number of rooms  
**Output:** $y_p$ - the house price

## Dataset

We are given a dataset of $n$ datapoints: $X=\left\{\left(x_1, y_1\right),\left(x_2, y_2\right), \ldots ,\left(x_n, y_n\right)\right\}$.
 Here, $x_1, x_2,\ldots, x_n$ represent the number of rooms, while $y_1, y_2,\ldots, y_n$ represent the corresponding house prices (**ground truth**).

## Solution

We can find a straight line to map the **input** and **output** variables. This line can be represented by the function $\hat{y}=wx + b$, where $\hat{y}$ is the predicted value, $w$ is the weight, and $b$ is the bias.

We need to find the values of $w$ and $b$  based on our dataset.

## Loss function

We want to find $w$ and $b$ to make predicted value $\hat{y}$ is close at ground truth $y$ as much as possible.
To calculate different between $\hat{y}$ and $y$ we define a loss function using **Mean Squared Error**

$$\mathcal{L}(w, b)=\frac{1}{n} \sum_{i=1}^m\left(y_i-\hat{y}_i\right)^2$$

# Gradient descent 
Now our goal is find $w$ and $b$ to make $\mathcal{L}(w, b)$ as close as zero as possible.
