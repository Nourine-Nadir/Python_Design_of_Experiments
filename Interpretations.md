# Anova
## Type 1 (Sequential)
* **Order-dependant :** Each predictor is tested **in the order it appears** in the model.
* The sum of squares (sum_sq) for each variable **includes only the variance explained by trhat variable, given te ones before it**
* Variables appearing later  in the fornmula have less 'exclusive' variance to explain because earlier ones have already taken a share 
## Type 2 (Marginal Contribution)
* **Order-independant :** Tests each predictor **while considering all other predictors already in the model.**
* **Does not include interactions** when calculating main effects.
* Each predictor's ` sum_sq ` reflects its unique contribution **after accounting for other main effects.**  

## Results :
### Reflectance : 
| Variable | df  | sum_sq | mean_sq | F-value | P-value | Significance |
|----------|------|---------|----------|----------|----------|--------------|
| x0       | 1.0  | 310.38  | 310.38   | 114.73   | 0.00     | S            |
| x1       | 1.0  | 343.48  | 343.48   | 126.97   | 0.00     | S            |
| x2       | 1.0  | 54.55   | 54.55    | 20.16    | 0.01     | S            |
| x0^2     | 1.0  | 3.32    | 3.32     | 1.23     | 0.32     | NS           |
| x0_x1    | 1.0  | 77.70   | 77.70    | 28.72    | 0.00     | S            |
| x0_x2    | 1.0  | 0.15    | 0.15     | 0.06     | 0.82     | NS           |
| x1^2     | 1.0  | 37.64   | 37.64    | 13.91    | 0.01     | S            |
| x1_x2    | 1.0  | 40.51   | 40.51    | 14.98    | 0.01     | S            |
| x2^2     | 1.0  | 2.01    | 2.01     | 0.74     | 0.43     | NS           |
| Residual | 5.0  | 13.53   | 2.71     | NaN      | NaN      | NS           |

### DSL : 
| Variable | df  | sum_sq  | mean_sq  | F-value | P-value | Significance |
|----------|------|----------|-----------|----------|----------|--------------|
| x0       | 1.0  | 2433.06  | 2433.06   | 102.35   | 0.00     | S            |
| x1       | 1.0  | 9591.18  | 9591.18   | 403.46   | 0.00     | S            |
| x2       | 1.0  | 972.40   | 972.40    | 40.91    | 0.00     | S            |
| x0^2     | 1.0  | 2182.27  | 2182.27   | 91.80    | 0.00     | S            |
| x0_x1    | 1.0  | 2.58     | 2.58      | 0.11     | 0.76     | NS           |
| x0_x2    | 1.0  | 276.28   | 276.28    | 11.62    | 0.02     | S            |
| x1^2     | 1.0  | 423.29   | 423.29    | 17.81    | 0.01     | S            |
| x1_x2    | 1.0  | 27.41    | 27.41     | 1.15     | 0.33     | NS           |
| x2^2     | 1.0  | 483.91   | 483.91    | 20.36    | 0.01     | S            |
| Residual | 5.0  | 118.86   | 23.77     | NaN      | NaN      | NS           |

## Terms significance 
* **Variables :** are the equations terms that refer to the parameters used in the experience
* **Sum of Squares :**  Represents the variation explained by each term (predictor or interaction) in the model
* **df (Degrees of Freedom) :** This indicates the number of independent pieces of information used to estimate the variance for each term
* **F (F-statistic) :** The ratio of variance explained by the term over the residual variance. A larger F-value suggests that the term contributes significantly to explaining the variance in the data.
* **P-value (PR(>F)):** The probability of observing the given F-statistic, or one more extreme, under the null hypothesis. If the P-value is less than a significance threshold (typically 0.05), you can reject the null hypothesis and conclude that the corresponding term is statistically significant.
* **Significance :** P-value threshold is set to 0.05 to assess that the predictor or interaction is significant.

## Computing pred on the same set
### ✅ Why is it okay here?

* You’re calculating **Adequate Precision, Standard Deviation, and Coefficient of Variation (CV)**, which describe how well the model fits the given dataset.
* These metrics do **not** require a test set; they focus on the **model’s behavior on the training set**.
* **ANOVA is an explanatory analysis,** meaning we want to understand how the independent variables affect the dependent variable in the **given dataset**, not necessarily in unseen data.
