# Managing Disagreement in Interactive Learning: A Counterfactual Explanation Approach

## Project Overview
This project explores methods for **managing disagreement between humans and AI models** in **interactive learning** environments through the use of **counterfactual explanations**.  
It was developed as part of the **Master’s Degree in Data Science and Business Informatics**, at the University of Pisa.

## Context and Motivation
In many real-world applications, users should not passively accept AI predictions but **interact with models**, provide feedback, and **influence the learning process** over time.  
This human-centered approach, referred to as **Human-in-the-Loop (HTL)**, enables bidirectional interaction — combining machine learning outputs with human expertise and skepticism.

## Objectives
- Introduce a framework to **model, quantify, and manage disagreement** (skepticism) between humans and AI systems.  
- Employ **counterfactual explanations** to provide interpretable insights and guide adaptive model corrections.  
- Evaluate the effectiveness of different counterfactual methods and their impact on skepticism and model performance.

## Framework and Methodology

### Key Components
- **FRANK Framework**:  
  Models user skepticism based on the difference between model confidence and user confidence.  
  - *Skepticism = (Model Confidence × Model Probability) – (User Confidence × User Probability)*  

- **Counterfactual Explanation Methods**:
  - **DICE**
  - **Growing Spheres**
  - **LORE** (Local Rule-based Explanations)
  - **Prototypes**

- **Adaptive Correction**:  
  LORE is used to generate a local neighborhood around an instance to build a surrogate model and propose meaningful counterfactuals for correction.

### Counterfactual Properties Evaluated
- **Sparsity**
- **Validity**
- **Proximity**
- **Plausibility**

### User Modeling
Four simulated user profiles were defined based on two key attributes:
- **Expertise** (decision-making competence)
- **Believability** (trust in model predictions)


## Experimental Setup
Experiments were conducted on several datasets to evaluate:
- Counterfactual explanation quality  
- Skepticism behavior across user profiles  
- Model performance before and after adaptive correction  

### Datasets
- **Phishing**
- **Elec2**
- **HTTP**
- **CreditCard**

### Evaluation Metrics
- **Proximity**
- **Sparsity**
- **Plausibility**
- **Execution time**
- **Skepticism index**
- **Model accuracy**

