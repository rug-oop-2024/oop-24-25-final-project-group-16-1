DSC-0001: Using Abstract Base Class for Model Interface
=======================================================

**Date:** 2024-11-08  
**Decision:** Defining a base `Model` class as an abstract base class (ABC).  
**Status:** Accepted  

**Motivation:**  
Provide a consistent interface for machine learning models that enforce the implementation of `fit` and `predict` methods.

**Reason:**  
By using an ABC, we ensure that each model subclass promotes consistency across models and provides a standard base.

**Limitations:**  
Requires each subclass to override methods.

**Alternatives:**  
Use a standard class without enforced method overrides.

**Location:**  
`autoop/core/ml/model/model.py`


DSC-0002: Use Base64 Encoding for Data Serialization
====================================================

**Date:** 2024-11-08  
**Decision:** Use Base64 encoding to serialize artifact data.  
**Status:** Accepted  

**Motivation:**  
Ensures data can be saved as strings and stored.

**Reason:**  
Base64 encoding provides compatibility with JSON and other text-based storage formats.

**Limitations:**  
Increases data size.

**Alternatives:**  
Use binary storage or implement a custom serialization format.

**Location:**  
`autoop/core/ml/artifact.py`


DSC-0003: Use Unique ID for Artifact Versioning
===============================================

**Date:** 2024-11-08  
**Decision:** Generating a unique ID for each artifact using its name and version.  
**Status:** Accepted  

**Motivation:**  
Enable version control and facilitate artifact tracking.

**Reason:**  
Combining name and version provides a clear and unique identifier for each artifact.

**Limitations:**  
Requires versioning management to avoid conflicts.

**Alternatives:**  
Use a hashing function for a unique identifier.

**Location:**  
`autoop/core/ml/artifact.py`


DSC-0004: Extend Artifact Class for Dataset Handling
====================================================

**Date:** 2024-11-08  
**Decision:** Inherit the `Dataset` class from the `Artifact` class to enable serialization and metadata handling.  
**Status:** Accepted  

**Motivation:**  
Reusing existing functionality for metadata and data storage by extending `Artifact`.

**Reason:**  
Extending `Artifact` simplifies dataset management and allows datasets to be treated as artifacts.

**Limitations:**  
Dataset class must align with `Artifact` data handling.

**Alternatives:**  
Create a separate standalone `Dataset` class.

**Location:**  
`autoop/core/ml/dataset.py`


DSC-0005: Use Base64 Encoding for Dataset Serialization
=======================================================

**Date:** 2024-11-08  
**Decision:** Store dataset data as Base64-encoded CSV to allow text-based serialization.  
**Status:** Accepted  

**Motivation:**  
Ensure datasets can be easily serialized, saved, and loaded from a text-based format.

**Reason:**  
Base64 encoding enables compatibility with JSON or database storage.

**Limitations:**  
Increases data size.

**Alternatives:**  
Store data as raw binary.

**Location:**  
`autoop/core/ml/dataset.py`


DSC-0006: Encapsulation for Feature Attributes
==============================================

**Date:** 2024-11-08  
**Decision:** Use private attributes with getters and setters for feature properties.  
**Status:** Accepted  

**Motivation:**  
Control access for feature properties such as name, type, and values.

**Reason:**  
Encapsulation allows and manages validation checks.

**Limitations:**  
Increases code complexity with additional methods.

**Alternatives:**  
Directly use public attributes without validation.

**Location:**  
`autoop/core/ml/feature.py`


DSC-0007: Provide Statistics Calculation for Numeric Features
=============================================================

**Date:** 2024-11-08  
**Decision:** Implement `get_statistics` method to calculate statistics for NUMERIC features.  
**Status:** Accepted  

**Motivation:**  
Allows easy retrieval of basic statistics like mean, std, min, and max for numeric data.

**Reason:**  
Statistics are commonly needed in data processing and model building.

**Limitations:**  
Only applicable for numeric features; raises error if called on categorical data.

**Alternatives:**  
Move statistics calculation to an external function.

**Location:**  
`autoop/core/ml/feature.py`


DSC-0008: Use Abstract Base Class for Metric Interface
======================================================

**Date:** 2024-11-08  
**Decision:** Define a base `Metric` class as an abstract base class (ABC).  
**Status:** Accepted  

**Motivation:**  
Enforce a standard interface for all metrics by requiring implementation of the `evaluate` method.

**Reason:**  
This approach ensures all metric subclasses provide a method to calculate the metric.

**Limitations:**  
Each subclass must implement its own `evaluate` method.

**Alternatives:**  
Use a simple base class without enforced abstract methods.

**Location:**  
`autoop/core/ml/metric.py`


DSC-0009: Factory Function for Metric Retrieval
===============================================

**Date:** 2024-11-08  
**Decision:** Implement a factory function, `get_metric`, to retrieve metric instances by name.  
**Status:** Accepted  

**Motivation:**  
Simplifies metric retrieval and enables dynamic selection of metrics.

**Reason:**  
The factory pattern provides a single access point for all metric types.

**Limitations:**  
Requires updates when new metrics are added.

**Alternatives:**  
Directly instantiate metrics without a factory function.

**Location:**  
`autoop/core/ml/metric.py`


DSC-0010: Validation of Model Type Based on Target Feature Type
===============================================================

**Date:** 2024-11-08  
**Decision:** Validate that the model type corresponds with the target feature type.  
**Status:** Accepted  

**Motivation:**  
Ensure that models are compatible with target feature types.

**Reason:**  
Prevents runtime errors by enforcing correct model-target type pairing.

**Limitations:**  
Additional model types will require validation updates.

**Alternatives:**  
Use a less strict validation and allow flexibility.

**Location:**  
`autoop/core/ml/pipeline.py`


DSC-0011: Artifact Registration System
======================================

**Date:** 2024-11-08  
**Decision:** Implement an artifact registration system to track feature transformations.  
**Status:** Accepted  

**Motivation:**  
Store data transformations for model reproducibility.

**Reason:**  
Essential for model tracking and reproducibility in ML pipelines.

**Limitations:**  
Large data could lead to increased storage requirements.

**Alternatives:**  
Use logging instead of in-memory tracking.

**Location:**  
`autoop/core/ml/pipeline.py`


DSC-0012: Split Data by Configurable Ratio
==========================================

**Date:** 2024-11-08  
**Decision:** Implement data splitting using a configurable ratio (default 0.8).  
**Status:** Accepted  

**Motivation:**  
Allow flexible train-test splitting as per model requirements.

**Reason:**  
Improves control over model training/testing phases.

**Limitations:**  
Fixed ratio may not be suitable for all dataset sizes.

**Alternatives:**  
Use cross-validation or k-folds.

**Location:**  
`autoop/core/ml/pipeline.py`


DSC-0013: Use JSON Serialization for Data Persistence
=====================================================

**Date:** 2024-11-08  
**Decision:** Store data as JSON strings in a specified storage backend.  
**Status:** Accepted  

**Motivation:**  
JSON provides a human-readable version of structured data.

**Reason:**  
It is widely compatible and human-readable.

**Limitations:**  
JSON is not ideal for very large datasets or complex data.

**Alternatives:**  
SQLite or NoSQL databases for larger data needs.

**Location:**  
`autoop/core/database.py`


DSC-0014: Automatic Detection of Feature Types
==============================================

**Date:** 2024-11-08  
**Decision:** Automatically detect feature types in the dataset as either 'categorical' or 'numeric'.  
**Status:** Accepted  

**Motivation:**  
To streamline the process of identifying feature types for ML tasks.

**Reason:**  
Provides an automated way to classify features without manual input, reducing human error.

**Limitations:**  
May not work for complex data types or features that require custom classification.

**Alternatives:**  
Allow manual feature type input or use a more feature type detection library.

**Location:**  
`autoop/core/functional/feature.py`

DSC-0015: Preprocessing Features with Scikit-Learn Encoders
===========================================================

**Date:** 2024-11-08  
**Decision:** Use Scikit-Learn's OneHotEncoder for categorical features and StandardScaler for numeric features to preprocess data.  
**Status:** Accepted  

**Motivation:**  
Scikit-Learn provides reliable and efficient tools for encoding and scaling.

**Reason:**  
It simplifies the preprocessing process and ensures compatibility with common machine learning models.

**Limitations:**  
Limited to one-hot encoding and standard scaling.

**Alternatives:**  
Custom encoders, MinMaxScaler for numeric scaling, or other libraries.

**Location:**  
`autoop/core/functional/preprocessing.py`

DSC-0016: Model Selection with Factory Method
=============================================

**Date:** 2024-11-08  
**Decision:** Implement a factory function for model selection.  
**Status:** Accepted  

**Motivation:**  
Simplify model selection and initialization.

**Reason:**  
Reduces code complexity by centralizing model instantiation.

**Limitations:**  
Requires updating `models_map` for new models.

**Alternatives:**  
Use individual import and instantiation in each script.

**Location:**  
`autoop/core/ml/model/__init__.py`

DSC-0017: Lasso Regression Model Implementation
===============================================

**Date:** 2024-11-09  
**Decision:** Implement Lasso regression as a model subclass with configurable regularization.  
**Status:** Accepted  

**Motivation:**  
Include a regularized regression option that penalizes model complexity.

**Reason:**  
Lasso regression encourages sparsity in features, beneficial for high-dimensional data.

**Limitations:**  
Regularization strength must be manually tuned for optimal performance.

**Alternatives:**  
Ridge regression, Elastic Net.

**Location:**  
`autoop/core/ml/model/regression/lasso_regression.py`


DSC-0018: Linear Regression with Gradient Descent
=================================================

**Date:** 2024-11-09  
**Decision:** Implement linear regression using gradient descent as an iterative optimization.  
**Status:** Accepted  

**Motivation:**  
Provide a foundational linear model with a configurable learning rate and iteration count.

**Reason:**  
This approach allows control over model convergence and flexibility for small to moderate datasets.

**Limitations:**  
Gradient descent can be slow for large datasets and may require tuning.

**Alternatives:**  
Use libraries like scikit-learn for optimized, built-in linear regression.

**Location:**  
`autoop/core/ml/model/regression/linear_regression.py`


DSC-0019: Use LinearRegressionModel for Multiple Linear Regression
==================================================================

**Date:** 2024-11-09  
**Decision:** Use LinearRegressionModel as the base model for implementing multiple linear regression.  
**Status:** Accepted  

**Motivation:**  
Utilize the existing linear regression implementation for multi-feature support.

**Reason:**  
Reduces code redundancy and leverages the tested gradient descent approach.

**Limitations:**  
Inherits all limitations of the LinearRegressionModel, such as slow convergence with large datasets.

**Alternatives:**  
Implement multiple linear regression separately or use a third-party library.

**Location:**  
`autoop/core/ml/model/regression/multiple_linear_regression.py`


DSC-0020: Use DecisionTreeClassifier from Scikit-Learn for Decision Tree Model
==============================================================================

**Date:** 2024-11-09  
**Decision:** Use Scikit-Learn's DecisionTreeClassifier as the core model for implementing decision tree classification.  
**Status:** Accepted  

**Motivation:**  
Leverage a robust, well-optimized library implementation for decision trees.

**Reason:**  
Reduces implementation time and provides a reliable, tested model.

**Limitations:**  
The Scikit-Learn model may have limitations for very large datasets or highly customized use cases.

**Alternatives:**  
Implement a custom decision tree algorithm or use other libraries.

**Location:**  
`autoop/core/ml/model/classification/decision_tree_classifier.py`


DSC-0021: Implement K-Nearest Neighbors (KNN) Using Custom Distance Calculations
================================================================================

**Date:** 2024-11-08  
**Decision:** Implement KNN algorithm manually to allow control over distance metric and neighbor selection.  
**Status:** Accepted  

**Motivation:**  
Provides flexibility for experimenting with different distance metrics.

**Reason:**  
Offers deeper insights into model behavior compared to using an existing library implementation.

**Limitations:**  
Slower for large datasets due to O(n) complexity for each prediction.

**Alternatives:**  
Use Scikit-Learn's KNeighborsClassifier.

**Location:**  
`autoop/core/ml/model/classification/k_nearest_neighbors.py`


DSC-0022: Implement Random Forest Classifier Using Scikit-Learn
===============================================================

**Date:** 2024-11-08  
**Decision:** Use scikit-learn's RandomForestClassifier to enable classification tasks.  
**Status:** Accepted  

**Motivation:**  
Provides a powerful ensemble learning method with easy implementation.

**Reason:**  
The built-in RandomForestClassifier has optimized performance and parameter tuning capabilities.

**Limitations:**  
May be computationally expensive for large datasets due to multiple decision trees.

**Alternatives:**  
Implement custom random forest logic or use alternative ensemble models.

**Location:**  
`autoop/core/ml/model/classification/random_forest.py`
