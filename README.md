# README

## Summary of Findings
This project aims to compare the performance of four machine learning classifiers (Logistic Regression, K-Nearest Neighbors, Decision Trees, and Support Vector Machines) on a dataset related to marketing bank products over the telephone. The data comes from a Portuguese banking institution and is sourced from the UCI Machine Learning repository.

The goal is to predict whether a client will subscribe to a term deposit (target variable `y`) based on demographic, financial, and campaign-related features. The findings will help improve telemarketing campaigns by focusing efforts on clients most likely to respond positively.

### Key Highlights:
- **Baseline Accuracy**: 89% of clients in the dataset did not subscribe to a term deposit.
- **Best Performing Model**: Logistic Regression and Support Vector Machines achieved the highest test accuracy (88.6%), while Decision Tree had the highest training accuracy (91.7%), indicating potential overfitting.
- **Efficiency**: KNN had the shortest training time, while SVM took significantly longer.

### Link to Notebook
[Link to Prompt 3 Notebook](./prompt_III[1]-Final-Copy1.ipynb)



---




---

## Business Understanding
The business goal is to identify which clients are most likely to subscribe to a term deposit. This will:
- Improve telemarketing efficiency.
- Save resources by focusing on high-potential clients.
- Increase the overall success rate of term deposit subscriptions.

The analysis also highlights which factors most influence a client’s decision, enabling data-driven campaign strategies.

---

## Notebook Highlights
- **Data Cleaning**: Missing values were handled, and categorical variables were encoded using one-hot encoding and label encoding.
- **Baseline Model**: Majority class prediction ("no" for all clients) served as the baseline, achieving 89% accuracy.
- **Model Comparison**: Default hyperparameters were used for all classifiers to evaluate training time and accuracy on training and test datasets.
- **Visualizations**: ROC-AUC curves, confusion matrices, and a combined accuracy vs. training time plot were included for performance evaluation.

---

## Findings
### Key Metrics:
| Model                  | Training Accuracy | Test Accuracy | Training Time (s) |
|------------------------|-------------------|---------------|--------------------|
| Logistic Regression    | 88.8%            | 88.6%         | 0.544             |
| K-Nearest Neighbors    | 89.0%            | 87.3%         | 0.008             |
| Decision Tree          | 91.7%            | 86.5%         | 0.469             |
| Support Vector Machine | 88.9%            | 88.6%         | 145.418           |

![Model Comparison Plot](model_comparison_plot.png)



### Business Recommendations:
- Logistic Regression and Support Vector Machines provide the most balanced performance and should be used for client targeting.
- Consider KNN if model training time is a critical factor.
- Improve models by addressing class imbalance using oversampling or SMOTE to improve recall for minority class (subscribed clients).

---

## Next Steps
- Perform hyperparameter tuning (e.g., grid search) for all classifiers to optimize performance.
- Explore feature importance to identify the most influential factors.
- Use advanced models such as ensemble methods (Random Forest, Gradient Boosting).
- Conduct a cost-benefit analysis to ensure predictions lead to tangible business improvements.
- Deploy the best-performing model for real-time predictions.

---

## Recommendations
1. **Target High-Probability Clients**: Focus marketing campaigns on clients with characteristics identified by the models as indicative of high likelihood to subscribe.
2. **Address Class Imbalance**: Implement techniques to improve recall for subscribed clients to ensure the model does not miss potential customers.
3. **Invest in Efficient Models**: While Logistic Regression is computationally efficient, SVM’s performance can be enhanced with further tuning.

By implementing these strategies, the bank can achieve a more effective and resource-efficient marketing campaign, resulting in higher conversion rates for term deposits.

