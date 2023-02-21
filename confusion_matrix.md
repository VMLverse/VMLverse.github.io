# Confusion Matrix | Machine Learning

## Remembering the confusion Matrix
The confusion matrix or error matrix is a table for summarizing the performance of a classification algorithm.

Remember the Confusion Matrix might be well 'confusing'. One easy way to understand the confusion matrix is to practically associate each section of the confusion matrix to a real event and a prediction. The real event in this case is having a fire vs not having a fire. The predicted event in this case will be alarm ringing vs no alarm. This can be represented as a confusion matrix as below:
![Alt text](/images/positive_negative.png)

## Precision
![Alt text](/images/precision.png)

## Recall
![Alt text](/images/recall.png)

## Type I and II Errors
| Type I Error                                                                                                                                                                        | Type II Error                                                                                                                                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| False Positive                                                                                                                                                                      | False Negative                                                                                                                                                                                                           |
| eg. Prediction = Positive (Alarm Rings) But Reality = False (Opposite, i.e., no Fire)                                                                                               | eg. Prediction = Negative (No Alarm) But Reality = False (Opposite) ie., Fire                                                                                                                                            |
| Useful in Spam Domain: False Positive in Spam. Prediction = Spam, but reality is a  genuine mail (False). Impact: User misses out an important email  as it gets classified as spam | Useful in Fire Alarm or Medical: False Negative in Medical. Prediction = No Disease or Fire Actual = Opposite (Disease or Fire) Impact: User will not be treated for harmful  disease or User will not be alert to Fire. |
