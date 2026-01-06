# Portfolio
Selected projects in data science, machine learning and NLP

<hr style="border: 1px solid #eaecef;">

## Work Experience
**Data Science Intern @ Botrista (_August 2025 - Present_)**
- Data team for office of the CEO.

**Data Science Intern @ Splashtop (_May 2024 - August 2024_)**
- Churn Prediction, Customer 360.

**Data Science Intern @ Ozaru (_February 2024 - May 2024_)**
- Skydeck Batch 17 Investor Exhibition.

<hr style="border: 1px solid #eaecef;">

## Projects

<hr style="border: 1px solid #eaecef;">

### LLM Persona Fine-Tuning: The "Wednesday Addams" Chatbot

Developed a reusable fine-tuning pipeline to adapt open-source Large Language Models (LLMs) to specific personas or domain requirements. Tested with GPT and Llama. As a proof of concept, I engineered a chatbot with the distinct personality of Wednesday Addams (Netflix). Since no public dataset existed, I manually curated and formatted a custom dialogue dataset from the show's scripts. I utilized **QLoRA** (Quantized Low-Rank Adaptation) and **4-bit quantization** to fine-tune the model efficiently on my laptop with minimal computational cost.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![QLoRA](https://img.shields.io/badge/QLoRA-Quantized_Fine_Tuning-007ACC?style=flat)

[View Notebook on Github](LINK_TO_NOTEBOOK)

**Key Technical Achievements:**
* **Custom Data Engineering:** Scraped and structured raw script data into instruction-response pairs for supervised fine-tuning.
* **Efficient Training:** Implemented 4-bit quantization (`bitsandbytes`) and LoRA adapters (`PEFT`) to reduce memory usage by ~70% while maintaining performance.
* **Scalable Workflow:** Created a modular pipeline applicable to any open-source model (e.g., Llama 2, Mistral) for domain-specific tasks.

<hr style="border: 1px solid #eaecef;">

### Ensemble Machine Learning for Sentiment Analysis on Amazon Reviews

Predicting customer satisfaction from text is a core challenge in e-commerce. In this project, I formulated star-rating prediction as a binary classification task, distinguishing 5-star reviews from lower ratings using Natural Language Processing (NLP). I created a custom ensemble ("Blending") model that stacked Logistic Regression, CART, Random Forest, and Gradient Boosting classifiers, optimizing their weights via OLS regression to achieve a 16% accuracy improvement over the baseline.

**Selected Project Figures:**

<table width="100%" style="border: none; background-color: transparent;">
  <tr style="border: none; background-color: transparent;">
    <td width="50%" align="center" style="border: none;">
      <img src="https://raw.githubusercontent.com/Nchen29/portfolio/main/assets/img/logistic%20regression%20roc.png" width="100%" alt="ROC Curve"/>
      <br>
      <b><small>Figure 1. Logistic Regression ROC Curve</small></b>
    </td>
    <td width="50%" align="center" style="border: none;">
      <img src="https://raw.githubusercontent.com/Nchen29/portfolio/main/assets/img/random%20forest%20bar.png" width="100%" alt="Feature Importance"/>
      <br>
      <b><small>Figure 2: Random Forest Most Important Features</small></b>
    </td>
  </tr>
</table>

<p align="center">
  <img src="https://raw.githubusercontent.com/Nchen29/portfolio/main/assets/img/model%20comparisons.png" width="85%" alt="Model Comparison Table"/>
  <br>
  <b><small>Figure 3: Model Comparison</small></b>
</p>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-306998?style=flat&logo=python&logoColor=white)
![Ensemble Learning](https://img.shields.io/badge/Ensemble_Learning-FF6F00?style=flat)

[View Notebook on GitHub](https://github.com/Nchen29/portfolio/blob/main/projects/amazon_sentiment/Ensemble_learning_Amazon_Reviews.ipynb)


<hr style="border: 1px solid #eaecef;">

### 7 Million Demand Elasticities: Visualizing the "Retail Apocalypse"

Collaborated with researchers from the San Francisco Fed and UChicago on the "Seven Million Demand Elasticities" (7MDE) project to investigate if broadband proliferation caused a decline in brick-and-mortar retail. Developed an interactive **R Shiny** dashboard to visualize over 7.5 million price elasticity estimates across 14,000 products. The analysis revealed no negative correlation between broadband usage and in-store consumer activity, challenging the retail apocalypse hypothesis.

**Project Visualizations:**
<p align="center">
  <img src="https://raw.githubusercontent.com/Nchen29/Broadband-R-Shiny/main/img/7mde_figure_1.png" width="49%" />
  <img src="https://raw.githubusercontent.com/Nchen29/Broadband-R-Shiny/main/img/7mde_figure_2_3.png" width="49%" />
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/Nchen29/Broadband-R-Shiny/main/img/7mde_web_app_design.png" width="75%" />
</p>

![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![R Shiny](https://img.shields.io/badge/Shiny-007BC2?style=flat&logo=rstudio&logoColor=white)
![GIS](https://img.shields.io/badge/GIS-Geographic_Information_Systems-228B22?style=flat&logo=openstreetmap&logoColor=white)

[View Project Repository](https://github.com/Nchen29/Broadband-R-Shiny)<br> 
[View Official Poster](https://cdss.berkeley.edu/project/seven-million-demand-elasticities)

<hr style="border: 1px solid #eaecef;">

