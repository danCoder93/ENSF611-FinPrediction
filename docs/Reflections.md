# Reflections

---

a. Why did you select this problem to solve?

[Ans] I have been involved in financial market prediction since 2016, approaching it both as an academic pursuit and as a practical extension of my investment interests. Financial time series offer a uniquely challenging environment: the data are noisy, often indistinguishable from randomness, and resistant to stable pattern extraction. This inherent complexity has continually drawn my attention, motivating my exploration of advanced modelling techniques and hybrid architectures capable of navigating such uncertainty. The difficulty of the problem is precisely what makes it intellectually compelling and practically significant, and it remains a driving reason for my continued engagement with this domain.

b. How did you contribute to this project?

[Ans] I contributed to this project across the full research and development pipeline, beginning with the formulation of the initial academically informed proposal and the subsequent refinement of the hybrid modeling strategy. My first responsibilities centered on data acquisition and feature engineering, ensuring that the inputs feeding into the models were both theoretically grounded and empirically robust. Building on this foundation, I redesigned the hybrid architecture to integrate ARIMA preprocessing, TCN-based feature extraction, and TFT based sequence modeling into a coherent multistage framework. I implemented the TCN and TFT components, validated their interaction within the pipeline, and adapted the workflow to support systematic experimentation. These contributions established a strong methodological baseline and enabled a rigorous evaluation of hybrid deep learning approaches for financial time series forecasting.

c. What did you find difficult about this project? What did you find easy? What did
you learn?

[Ans] I found some aspects of this project particularly challenging, especially the acquisition and alignment of fundamental data, given its quarterly frequency and the need to prevent any form of temporal leakage into the training process. Identifying suitable GPU-compatible libraries for TCN and TFT implementation also required careful exploration, as did ensuring that each component of the hybrid pipeline interacted correctly without inadvertently sharing future information. In contrast, tasks such as pandas-based data manipulation and implementing the XGBoost models felt more straightforward, largely because these tools and workflows were already familiar from prior coursework. Through this project, I gained a deeper understanding of how to construct and manage more advanced hybrid modeling pipelines, how to structure time-series experiments responsibly, and how to navigate the quirks and constraints of financial forecasting data.

d. Reflect on your experience during the demo session. Describe what it was like to
both present your project and check out the projects of your classmates.

[Ans] Presenting the project during the demo session was a genuinely rewarding experience, as it allowed me to share work that I am deeply invested in and engage in thoughtful technical discussions with the invigilator. The opportunity to articulate the motivation, methodology, and challenges behind the hybrid modeling pipeline helped reinforce my own understanding of the project’s strengths and limitations. At the same time, exploring the work of classmates offered a glimpse into the diverse approaches others took toward their problem domains. The session would have been even more enriching had every group presented to the full class, as a shared forum would have provided a broader sense of the cohort’s collective efforts and encouraged cross-project exchange of ideas.

e. How can you apply what you learned in this course to your desired career?

[Ans] The material I learned in this course aligns closely with my goal of becoming a quantitative researcher in the financial markets. Working through the full lifecycle of a financial prediction project: data acquisition, feature engineering, leakage free experimentation, and the construction of hybrid deep learning models mirrors the type of technical responsibilities found in quantitative finance roles. Developing and refining complex architectures such as TCN and TFT based pipelines strengthened my ability to navigate the challenges inherent to market data, including irregular sampling, noisy signals, and structural regime shifts. This experience not only deepened my understanding of how advanced machine learning methods can be applied to real financial time series, but also helped build the analytical discipline and modeling intuition essential for a career in quantitative research.
