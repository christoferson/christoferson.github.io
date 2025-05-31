Cost Functions vs. Evaluation Metrics: Key Differences
Cost Functions (Loss Functions)
Purpose: Guide the model's learning process during training
When used: During the optimization/training phase
Requirements: Must be mathematically differentiable and smooth
Goal: Provide gradients that help the algorithm adjust model parameters
Evaluation Metrics
Purpose: Assess model performance for human interpretation
When used: After training, during validation/testing
Requirements: Should be intuitive and meaningful to stakeholders
Goal: Help us understand how well the model performs in real-world terms
Why All Cost Functions Can Be Evaluation Metrics
Every cost function provides some measure of model performance, so they can technically serve as evaluation metrics. However, they're not always intuitive:

Example - Log Loss:

Log Loss = -[y*log(p) + (1-y)*log(1-p)]

A log loss of 0.693 vs 0.347 - which is better? (Lower is better, but the numbers aren't immediately meaningful)
Compare to accuracy: 85% vs 92% - immediately understandable
Why Some Evaluation Metrics Can't Be Cost Functions
1. Computational Complexity
Some metrics are too complex or subjective to optimize directly:

"Dog-likeness" - How do you quantify this mathematically?
User satisfaction - Requires human judgment
Aesthetic quality - Highly subjective
2. Non-Smooth Functions
Cost functions need to be differentiable for gradient-based optimization:

Accuracy Example:

# Accuracy is step-like - small parameter changes often don't change accuracy
# Model A: 84.7% accuracy
# Model A (slightly adjusted): 84.7% accuracy
# Model A (more adjustment): 84.7% accuracy
# Model A (significant change): 85.2% accuracy

The gradient is often zero, providing no learning signal for small improvements.

3. Threshold-Dependent Metrics
ROC Curve Issue:

ROC analysis requires testing multiple classification thresholds (0.1, 0.2, 0.3... 0.9)
But in practice, your model uses one fixed threshold (typically 0.5)
You can't optimize for "all possible thresholds" during training
Practical Examples
Good Cost Function + Evaluation Metric Pairs:
Binary Classification:

Cost Function: Binary Cross-Entropy (Log Loss)
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
Regression:

Cost Function: Mean Squared Error (MSE)
Evaluation Metrics: MAE, RMSE, R², MAPE
Why This Separation Matters:
Training Efficiency: Smooth, differentiable cost functions enable effective learning
Business Understanding: Intuitive metrics help stakeholders make decisions
Model Selection: You might choose a model with slightly higher training loss but better business metrics
Key Takeaway
Think of cost functions as the "language" your algorithm uses to learn, while evaluation metrics are the "language" you use to communicate model performance to humans and make business decisions. The best approach often involves optimizing one function while monitoring several others.