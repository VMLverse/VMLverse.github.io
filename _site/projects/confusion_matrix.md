# Remembering the Confusion Matrix
The confusion matrix or error matrix is a table for summarizing the performance of a classification algorithm.

Remember the Confusion Matrix might be well 'confusing'. One easy way to understand the confusion matrix is to practically associate each section of the confusion matrix to a real event and a prediction. The real event in this case is having a fire vs not having a fire. The predicted event in this case will be alarm ringing vs no alarm. This can be represented as a confusion matrix as below:
![Alt text](/images/positive_negative.png)

## Precision & Recall
Precision and recall are two commonly used performance metrics in machine learning to evaluate the performance of a classification model.

### Precision
Precision is the fraction of correctly predicted positive instances out of all predicted positive instances. In other words, it measures how often the model correctly predicts positive instances. Precision is calculated as:

Precision = True Positives / (True Positives + False Positives)

![Alt text](/images/precision.png)

### Recall
Recall is the fraction of correctly predicted positive instances out of all actual positive instances. In other words, it measures how often the model correctly identifies positive instances. Recall is calculated as:

Recall = True Positives / (True Positives + False Negatives)

![Alt text](/images/recall.png)

Where:
- True Positives (TP): the number of correctly predicted positive instances
- False Positives (FP): the number of instances that were predicted positive but actually negative
-  False Negatives (FN): the number of instances that were predicted negative but actually positive

Precision and recall are often used together to evaluate the performance of a model since they provide complementary information. A model with high precision is good at identifying the positive instances but may miss some of them, while a model with high recall is good at identifying all the positive instances but may also misclassify some negative instances. The ideal model would have both high precision and high recall.


## Type I and II Errors
<table>
<thead>
  <tr>
    <th>Type I Error</th>
    <th>Type II Error</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>False Positive</td>
    <td>False Negative</td>
  </tr>
  <tr>
    <td>eg. Prediction = Positive (Alarm Rings)<br />But Reality = False (Opposite, i.e., no Fire)</td>
    <td>eg. Prediction = Negative (No Alarm)<br />But Reality = False (Opposite) ie., Fire</td>
  </tr>
  <tr>
    <td>Useful in Spam Domain: False Positive in Spam.<br />
     Prediction = Spam, but reality is a genuine mail (False).<br />
     Impact: User misses out an important email as it gets classified as spam</td>
    <td>Useful in Fire Alarm or Medical: False Negative in Medical.<br />
     Prediction = No Disease or Fire Actual = Opposite (Disease or Fire)<br />
     Impact: User will not be treated for harmful disease or User will not be alert to Fire.</td>
  </tr>
</tbody>
</table>
