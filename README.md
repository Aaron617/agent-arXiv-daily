# Agent arXiv Daily

**Last Updated:** 2025-10-15 02:43:31

**Total Papers:** 75

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System</strong> - Qixin Wang, Dawei Wang, Kun Chen, Yaowei Hu, Puneet Girdhar, Ruoteng Wang, Aadesh Gupta, Chaitanya Devella, Wenlai Guo, Shangwen Huang, Bachir Aoun, Greg Hayworth, Han Li, Xintao Wu - [[pdf]](https://arxiv.org/pdf/2508.13423)</summary>

**Abstract:** In recent years, recommendation systems have evolved from providing a single list of recommendations to offering a comprehensive suite of topic focused services. To better accomplish this task, conversational recommendation systems (CRS) have progressed from basic retrieval augmented LLM generation to agentic systems with advanced reasoning and self correction capabilities. However, agentic systems come with notable response latency, a longstanding challenge for conversational recommendation systems. To balance the trade off between handling complex queries and minimizing latency, we propose AdaptJobRec, the first conversational job recommendation system that leverages autonomous agent to integrate personalized recommendation algorithm tools. The system employs a user query complexity identification mechanism to minimize response latency. For straightforward queries, the agent directly selects the appropriate tool for rapid responses. For complex queries, the agent uses the memory processing module to filter chat history for relevant content, then passes the results to the intelligent task decomposition planner, and finally executes the tasks using personalized recommendation tools. Evaluation on Walmart's real world career recommendation scenarios demonstrates that AdaptJobRec reduces average response latency by up to 53.3% compared to competitive baselines, while significantly improving recommendation accuracy.

**arXiv ID:** 2508.13423
</details>

<details>
<summary><strong>A Longitudinal Randomized Control Study of Companion Chatbot Use: Anthropomorphism and Its Mediating Role on Social Impacts</strong> - Rose E. Guingrich, Michael S. A. Graziano - [[pdf]](https://arxiv.org/pdf/2509.19515)</summary>

**Abstract:** Many Large Language Model (LLM) chatbots are designed and used for companionship, and people have reported forming friendships, mentorships, and romantic partnerships with them. Concerns that companion chatbots may harm or replace real human relationships have been raised, but whether and how these social consequences occur remains unclear. In the present longitudinal study ($N = 183$), participants were randomly assigned to a chatbot condition (text chat with a companion chatbot) or to a control condition (text-based word games) for 10 minutes a day for 21 days. Participants also completed four surveys during the 21 days and engaged in audio recorded interviews on day 1 and 21. Overall, social health and relationships were not significantly impacted by companion chatbot interactions across 21 days of use. However, a detailed analysis showed a different story. People who had a higher desire to socially connect also tended to anthropomorphize the chatbot more, attributing humanlike properties to it; and those who anthropomorphized the chatbot more also reported that talking to the chatbot had a greater impact on their social interactions and relationships with family and friends. Via a mediation analysis, our results suggest a key mechanism at work: the impact of human-AI interaction on human-human social outcomes is mediated by the extent to which people anthropomorphize the AI agent, which is in turn motivated by a desire to socially connect. In a world where the desire to socially connect is on the rise, this finding may be cause for concern.

**arXiv ID:** 2509.19515
</details>

<details>
<summary><strong>VizCopilot: Fostering Appropriate Reliance on Enterprise Chatbots with Context Visualization</strong> - Sam Yu-Te Lee, Jingya Chen, Albert Calzaretto, Richard Lee, Alice Ferng, Mihaela Vorvoreanu - [[pdf]](https://arxiv.org/pdf/2510.11954)</summary>

**Abstract:** Enterprise chatbots show promise in supporting knowledge workers in information synthesis tasks by retrieving context from large, heterogeneous databases before generating answers. However, when the retrieved context misaligns with user intentions, the chatbot often produces "irrelevantly right" responses that provide little value. In this work, we introduce VizCopilot, a prototype that incorporates visualization techniques to actively involve end-users in context alignment. By combining topic modeling with document visualization, VizCopilot enables human oversight and modification of retrieved context while keeping cognitive overhead manageable. We used VizCopilot as a design probe in a Research-through-Design study to evaluate the role of visualization in context alignment and to surface future design opportunities. Our findings show that visualization not only helps users detect and correct misaligned context but also encourages them to adapt their prompting strategies, enabling the system to retrieve more relevant context from the outset. At the same time, the study reveals limitations in verification support regarding close-reading and trust in AI summaries. We outline future directions for visualization-enhanced chatbots, focusing on personalization, proactivity, and sustainable human-AI collaboration.

**arXiv ID:** 2510.11954
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation</strong> - Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary S Siegel, Boyi Wei, Tianci Xue, Ziru Chen, Felix Chen, Saiteja Utpala, Franck Ndzomga, Dheeraj Oruganty, Sophie Luskin, Kangheng Liu, Botao Yu, Amit Arora, Dongyoon Hahm, Harsh Trivedi, Huan Sun, Juyong Lee, Tengjun Jin, Yifan Mai, Yifei Zhou, Yuxuan Zhu, Rishi Bommasani, Daniel Kang, Dawn Song, Peter Henderson, Yu Su, Percy Liang, Arvind Narayanan - [[pdf]](https://arxiv.org/pdf/2510.11977)</summary>

**Abstract:** AI agents have been developed for complex real-world tasks from coding to customer service. But AI agent evaluations suffer from many challenges that undermine our understanding of how well agents really work. We introduce the Holistic Agent Leaderboard (HAL) to address these challenges. We make three main contributions. First, we provide a standardized evaluation harness that orchestrates parallel evaluations across hundreds of VMs, reducing evaluation time from weeks to hours while eliminating common implementation bugs. Second, we conduct three-dimensional analysis spanning models, scaffolds, and benchmarks. We validate the harness by conducting 21,730 agent rollouts across 9 models and 9 benchmarks in coding, web navigation, science, and customer service with a total cost of about $40,000. Our analysis reveals surprising insights, such as higher reasoning effort reducing accuracy in the majority of runs. Third, we use LLM-aided log inspection to uncover previously unreported behaviors, such as searching for the benchmark on HuggingFace instead of solving a task, or misusing credit cards in flight booking tasks. We share all agent logs, comprising 2.5B tokens of language model calls, to incentivize further research into agent behavior. By standardizing how the field evaluates agents and addressing common pitfalls in agent evaluation, we hope to shift the focus from agents that ace benchmarks to agents that work reliably in the real world.

**arXiv ID:** 2510.11977
</details>

<details>
<summary><strong>AI Agents as Universal Task Solvers</strong> - Alessandro Achille, Stefano Soatto - [[pdf]](https://arxiv.org/pdf/2510.12066)</summary>

**Abstract:** AI reasoning agents are already able to solve a variety of tasks by deploying tools, simulating outcomes of multiple hypotheses and reflecting on them. In doing so, they perform computation, although not in the classical sense -- there is no program being executed. Still, if they perform computation, can AI agents be universal? Can chain-of-thought reasoning solve any computable task? How does an AI Agent learn to reason? Is it a matter of model size? Or training dataset size?
In this work, we reinterpret the role of learning in the context of AI Agents, viewing them as compute-capable stochastic dynamical systems, and highlight the role of time in a foundational principle for learning to reason. In doing so, we propose a shift from classical inductive learning to transductive learning -- where the objective is not to approximate the distribution of past data, but to capture their algorithmic structure to reduce the time needed to find solutions to new tasks.
Transductive learning suggests that, counter to Shannon's theory, a key role of information in learning is about reduction of time rather than reconstruction error. In particular, we show that the optimal speed-up that a universal solver can achieve using past data is tightly related to their algorithmic information. Using this, we show a theoretical derivation for the observed power-law scaling of inference time versus training time. We then show that scaling model size can lead to behaviors that, while improving accuracy on benchmarks, fail any reasonable test of intelligence, let alone super-intelligence: In the limit of infinite space and time, large models can behave as savants, able to brute-force through any task without any insight. Instead, we argue that the key quantity to optimize when scaling reasoning models is time, whose critical role in learning has so far only been indirectly considered.

**arXiv ID:** 2510.12066
</details>

<details>
<summary><strong>AwareCompiler: Agentic Context-Aware Compiler Optimization via a Synergistic Knowledge-Data Driven Framework</strong> - Hongyu Lin, Haolin Pan, Haoran Luo, Yuchen Li, Kaichun Yao, Libo Zhang, Mingjie Xing, Yanjun Wu - [[pdf]](https://arxiv.org/pdf/2510.11759)</summary>

**Abstract:** Compiler optimization is crucial for enhancing program performance by transforming the sequence of optimization passes while maintaining correctness. Despite the promising potential of large language models (LLMs)-based agent for software optimization, automating compiler optimization remains challenging due to: (1) semantic misalignment between abstract program representations and concrete optimization passes, (2) inefficient interaction mechanisms between agents and compiler environments, and (3) reward sparsity from the extensive decision-making process within large optimization spaces. This paper introduces \textbf{AwareCompiler}, an agentic framework for compiler optimization that addresses these challenges through three key innovations: structured knowledge integration and dataset construction, knowledge-driven adaptive pass generation, and data-driven hybrid training pipeline. Experimental results on standard benchmarks demonstrate that AwareCompiler significantly outperforms existing baselines in both performance and efficiency, highlighting the effectiveness of our synergistic knowledge-data-driven approach. Our code is publicly available at this https URL.

**arXiv ID:** 2510.11759
</details>

<details>
<summary><strong>DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving</strong> - Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, Zhaoxiang Zhang - [[pdf]](https://arxiv.org/pdf/2510.12796)</summary>

**Abstract:** Scaling Vision-Language-Action (VLA) models on large-scale data offers a promising path to achieving a more generalized driving intelligence. However, VLA models are limited by a ``supervision deficit'': the vast model capacity is supervised by sparse, low-dimensional actions, leaving much of their representational power underutilized. To remedy this, we propose \textbf{DriveVLA-W0}, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment. We showcase the paradigm's versatility by instantiating it for two dominant VLA archetypes: an autoregressive world model for VLAs that use discrete visual tokens, and a diffusion world model for those operating on continuous visual features. Building on the rich representations learned from world modeling, we introduce a lightweight action expert to address the inference latency for real-time deployment. Extensive experiments on the NAVSIM v1/v2 benchmark and a 680x larger in-house dataset demonstrate that DriveVLA-W0 significantly outperforms BEV and VLA baselines. Crucially, it amplifies the data scaling law, showing that performance gains accelerate as the training dataset size increases.

**arXiv ID:** 2510.12796
</details>

<details>
<summary><strong>Causal Agent based on Large Language Model</strong> - Kairong Han, Kun Kuang, Ziyu Zhao, Junjian Ye, Fei Wu - [[pdf]](https://arxiv.org/pdf/2408.06849)</summary>

**Abstract:** The large language model (LLM) has achieved significant success across various domains. However, the inherent complexity of causal problems and causal theory poses challenges in accurately describing them in natural language, making it difficult for LLM to comprehend and use them effectively. Causal methods are not easily conveyed through natural language, which hinders LLM's ability to apply them accurately. Additionally, causal datasets are typically tabular, while LLM excels in handling natural language data, creating a structural mismatch that impedes effective reasoning with tabular data. To address these challenges, we have equipped the LLM with causal tools within an agent framework, named the Causal Agent, enabling it to tackle causal problems. The causal agent comprises tools, memory, and reasoning modules. In the tool module, the causal agent calls Python code and uses the encapsulated causal function module to align tabular data with natural language. In the reasoning module, the causal agent performs reasoning through multiple iterations with the tools. In the memory module, the causal agent maintains a dictionary instance where the keys are unique names and the values are causal graphs. To verify the causal ability of the causal agent, we established a Causal Tabular Question Answer (CausalTQA) benchmark consisting of four levels of causal problems: variable level, edge level, causal graph level, and causal effect level. CausalTQA consists of about 1.4K for these four levels questions. Causal agent demonstrates remarkable efficacy on the four-level causal problems, with accuracy rates all above 80\%. Through verification on the real-world dataset QRData, the causal agent is 6\% higher than the original SOTA. For further insights and implementation details, our code is accessible via the GitHub repository this https URL.

**arXiv ID:** 2408.06849
</details>

<details>
<summary><strong>EvolveNav: Empowering LLM-Based Vision-Language Navigation via Self-Improving Embodied Reasoning</strong> - Bingqian Lin, Yunshuang Nie, Khun Loun Zai, Ziming Wei, Mingfei Han, Rongtao Xu, Minzhe Niu, Jianhua Han, Hanwang Zhang, Liang Lin, Bokui Chen, Cewu Lu, Xiaodan Liang - [[pdf]](https://arxiv.org/pdf/2506.01551)</summary>

**Abstract:** Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for enhancing vision-language navigation (VLN) performance, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches predominantly adopt straightforward input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. To address these issues, we propose EvolveNav, a novel sElf-improving embodied reasoning paradigm that realizes adaptable and generalizable navigational reasoning for boosting LLM-based vision-language Navigation. Specifically, EvolveNav involves a two-stage training process: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with curated formalized CoT labels to first activate the model's navigational reasoning capabilities, and simultaneously increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also designed to encourage the model to learn correct reasoning patterns by contrasting with wrong ones. Experimental results under both task-specific and cross-task training paradigms demonstrate the consistent superiority of EvolveNav over previous LLM-based VLN approaches on various popular benchmarks, including R2R, REVERIE, CVDN, and SOON. Code is available at this https URL.

**arXiv ID:** 2506.01551
</details>

<details>
<summary><strong>Abmax: A JAX-based Agent-based Modeling Framework</strong> - Siddharth Chaturvedi, Ahmed El-Gazzar, Marcel van Gerven - [[pdf]](https://arxiv.org/pdf/2508.16508)</summary>

**Abstract:** Agent-based modeling (ABM) is a principal approach for studying complex systems. By decomposing a system into simpler, interacting agents, agent-based modeling (ABM) allows researchers to observe the emergence of complex phenomena. High-performance array computing libraries like JAX can help scale such computational models to a large number of agents by using automatic vectorization and just-in-time (JIT) compilation. One of the caveats of using JAX to achieve such scaling is that the shapes of arrays used in the computational model should remain immutable throughout the simulation. In the context of agent-based modeling (ABM), this can pose constraints on certain agent manipulation operations that require flexible data structures. A subset of which is represented by the ability to update a dynamically selected number of agents by applying distinct changes to them during a simulation. To this effect, we introduce Abmax, an ABM framework based on JAX that implements multiple just-in-time (JIT) compilable algorithms to provide this functionality. On the canonical predation model benchmark, Abmax achieves runtime performance comparable to state-of-the-art implementations. Further, we show that this functionality can also be vectorized, making it possible to run many similar agent-based models in parallel. We also present two examples in the form of a traffic-flow model and a financial market model to show the use case of Abmax.

**arXiv ID:** 2508.16508
</details>

<details>
<summary><strong>AgentAda: Skill-Adaptive Data Analytics for Tailored Insight Discovery</strong> - Amirhossein Abaskohi, Amrutha Varshini Ramesh, Shailesh Nanisetty, Chirag Goel, David Vazquez, Christopher Pal, Spandana Gella, Giuseppe Carenini, Issam H. Laradji - [[pdf]](https://arxiv.org/pdf/2504.07421)</summary>

**Abstract:** We introduce AgentAda, the first LLM-powered analytics agent that can learn and use new analytics skills to extract more specialized insights. Unlike existing methods that require users to manually decide which data analytics method to apply, AgentAda automatically identifies the skill needed from a library of analytical skills to perform the analysis. This also allows AgentAda to use skills that existing LLMs cannot perform out of the box. The library covers a range of methods, including clustering, predictive modeling, and NLP techniques like BERT, which allow AgentAda to handle complex analytics tasks based on what the user needs. AgentAda's dataset-to-insight extraction strategy consists of three key steps: (I) a question generator to generate queries relevant to the user's goal and persona, (II) a hybrid Retrieval-Augmented Generation (RAG)-based skill matcher to choose the best data analytics skill from the skill library, and (III) a code generator that produces executable code based on the retrieved skill's documentation to extract key patterns. We also introduce KaggleBench, a benchmark of curated notebooks across diverse domains, to evaluate AgentAda's performance. We conducted a human evaluation demonstrating that AgentAda provides more insightful analytics than existing tools, with 48.78% of evaluators preferring its analyses, compared to 27.67% for the unskilled agent. We also propose a novel LLM-as-a-judge approach that we show is aligned with human evaluation as a way to automate insight quality evaluation at larger scale.

**arXiv ID:** 2504.07421
</details>

<details>
<summary><strong>DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments</strong> - Chiyu Zhang, Marc-Alexandre Cote, Michael Albada, Anush Sankaran, Jack W. Stokes, Tong Wang, Amir Abdi, William Blum, Muhammad Abdul-Mageed - [[pdf]](https://arxiv.org/pdf/2506.00739)</summary>

**Abstract:** Large language model (LLM) agents have shown impressive capabilities in human language comprehension and reasoning, yet their potential in cybersecurity remains underexplored. We introduce DefenderBench, a practical, open-source toolkit for evaluating language agents across offense, defense, and cybersecurity knowledge-based tasks. DefenderBench includes environments for network intrusion, malicious content detection, code vulnerability analysis, and cybersecurity knowledge assessment. It is intentionally designed to be affordable and easily accessible for researchers while providing fair and rigorous assessment. We benchmark several state-of-the-art (SoTA) and popular LLMs, including both open- and closed-weight models, using a standardized agentic framework. Our results show that Claude-3.7-sonnet performs best with a DefenderBench score of 81.65, followed by Claude-3.7-sonnet-think with 78.40, while the best open-weight model, Llama 3.3 70B, is not far behind with a DefenderBench score of 71.81. DefenderBench's modular design allows seamless integration of custom LLMs and tasks, promoting reproducibility and fair comparisons. An anonymized version of DefenderBench is available at this https URL.

**arXiv ID:** 2506.00739
</details>

<details>
<summary><strong>AGENTIQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation</strong> - Omid Reza Heidari, Siobhan Reid, Yassine Yaakoubi - [[pdf]](https://arxiv.org/pdf/2510.10661)</summary>

**Abstract:** LLMs have advanced text-to-SQL generation, yet monolithic architectures struggle with complex reasoning and schema diversity. We propose AGENTIQL, an agent-inspired multi-expert framework that combines a reasoning agent for question decomposition, a coding agent for sub-query generation, and a refinement step for column selection. An adaptive router further balances efficiency and accuracy by selecting between our modular pipeline and a baseline parser. Several steps in the pipeline can be executed in parallel, making the framework scalable to larger workloads. Evaluated on the Spider benchmark, AGENTIQL improves execution accuracy and interpretability and achieves up to 86.07% EX with 14B models using the Planner&Executor merging strategy. The attained performance is contingent upon the efficacy of the routing mechanism, thereby narrowing the gap to GPT-4-based SOTA (89.65% EX) while using much smaller open-source LLMs. Beyond accuracy, AGENTIQL enhances transparency by exposing intermediate reasoning steps, offering a robust, scalable, and interpretable approach to semantic parsing.

**arXiv ID:** 2510.10661
</details>

<details>
<summary><strong>HEAL: An Empirical Study on Hallucinations in Embodied Agents Driven by Large Language Models</strong> - Trishna Chakraborty, Udita Ghosh, Xiaopan Zhang, Fahim Faisal Niloy, Yue Dong, Jiachen Li, Amit K. Roy-Chowdhury, Chengyu Song - [[pdf]](https://arxiv.org/pdf/2506.15065)</summary>

**Abstract:** Large language models (LLMs) are increasingly being adopted as the cognitive core of embodied agents. However, inherited hallucinations, which stem from failures to ground user instructions in the observed physical environment, can lead to navigation errors, such as searching for a refrigerator that does not exist. In this paper, we present the first systematic study of hallucinations in LLM-based embodied agents performing long-horizon tasks under scene-task inconsistencies. Our goal is to understand to what extent hallucinations occur, what types of inconsistencies trigger them, and how current models respond. To achieve these goals, we construct a hallucination probing set by building on an existing benchmark, capable of inducing hallucination rates up to 40x higher than base prompts. Evaluating 12 models across two simulation environments, we find that while models exhibit reasoning, they fail to resolve scene-task inconsistencies-highlighting fundamental limitations in handling infeasible tasks. We also provide actionable insights on ideal model behavior for each scenario, offering guidance for developing more robust and reliable planning strategies.

**arXiv ID:** 2506.15065
</details>

<details>
<summary><strong>ManiAgent: An Agentic Framework for General Robotic Manipulation</strong> - Yi Yang, Kefan Gu, Yuqing Wen, Hebei Li, Yucheng Zhao, Tiancai Wang, Xudong Liu - [[pdf]](https://arxiv.org/pdf/2510.11660)</summary>

**Abstract:** While Vision-Language-Action (VLA) models have demonstrated impressive capabilities in robotic manipulation, their performance in complex reasoning and long-horizon task planning is limited by data scarcity and model capacity. To address this, we introduce ManiAgent, an agentic architecture for general manipulation tasks that achieves end-to-end output from task descriptions and environmental inputs to robotic manipulation actions. In this framework, multiple agents involve inter-agent communication to perform environmental perception, sub-task decomposition and action generation, enabling efficient handling of complex manipulation scenarios. Evaluations show ManiAgent achieves an 86.8% success rate on the SimplerEnv benchmark and 95.8% on real-world pick-and-place tasks, enabling efficient data collection that yields VLA models with performance comparable to those trained on human-annotated datasets. The project webpage is available at this https URL.

**arXiv ID:** 2510.11660
</details>

<details>
<summary><strong>Image Quality Assessment for Embodied AI</strong> - Chunyi Li, Jiaohao Xiao, Jianbo Zhang, Farong Wen, Zicheng Zhang, Yuan Tian, Xiangyang Zhu, Xiaohong Liu, Zhengxue Cheng, Weisi Lin, Guangtao Zhai - [[pdf]](https://arxiv.org/pdf/2505.16815)</summary>

**Abstract:** Embodied AI has developed rapidly in recent years, but it is still mainly deployed in laboratories, with various distortions in the Real-world limiting its application. Traditionally, Image Quality Assessment (IQA) methods are applied to predict human preferences for distorted images; however, there is no IQA method to assess the usability of an image in embodied tasks, namely, the perceptual quality for robots. To provide accurate and reliable quality indicators for future embodied scenarios, we first propose the topic: IQA for Embodied AI. Specifically, we (1) based on the Mertonian system and meta-cognitive theory, constructed a perception-cognition-decision-execution pipeline and defined a comprehensive subjective score collection process; (2) established the Embodied-IQA database, containing over 36k reference/distorted image pairs, with more than 5m fine-grained annotations provided by Vision Language Models/Vision Language Action-models/Real-world robots; (3) trained and validated the performance of mainstream IQA methods on Embodied-IQA, demonstrating the need to develop more accurate quality indicators for Embodied AI. We sincerely hope that through evaluation, we can promote the application of Embodied AI under complex distortions in the Real-world. Project page: this https URL

**arXiv ID:** 2505.16815
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>Empowering LLM Agents with Geospatial Awareness: Toward Grounded Reasoning for Wildfire Response</strong> - Yiheng Chen, Lingyao Li, Zihui Ma, Qikai Hu, Yilun Zhu, Min Deng, Runlong Yu - [[pdf]](https://arxiv.org/pdf/2510.12061)</summary>

**Abstract:** Effective disaster response is essential for safeguarding lives and property. Existing statistical approaches often lack semantic context, generalize poorly across events, and offer limited interpretability. While Large language models (LLMs) provide few-shot generalization, they remain text-bound and blind to geography. To bridge this gap, we introduce a Geospatial Awareness Layer (GAL) that grounds LLM agents in structured earth data. Starting from raw wildfire detections, GAL automatically retrieves and integrates infrastructure, demographic, terrain, and weather information from external geodatabases, assembling them into a concise, unit-annotated perception script. This enriched context enables agents to produce evidence-based resource-allocation recommendations (e.g., personnel assignments, budget allocations), further reinforced by historical analogs and daily change signals for incremental updates. We evaluate the framework in real wildfire scenarios across multiple LLM models, showing that geospatially grounded agents can outperform baselines. The proposed framework can generalize to other hazards such as floods and hurricanes.

**arXiv ID:** 2510.12061
</details>

<details>
<summary><strong>GOAT: A Training Framework for Goal-Oriented Agent with Tools</strong> - Hyunji Min, Sangwon Jung, Junyoung Sung, Dosung Lee, Leekyeung Han, Paul Hongsuck Seo - [[pdf]](https://arxiv.org/pdf/2510.12218)</summary>

**Abstract:** Large language models (LLMs) have recently been extended beyond traditional text generation to serve as interactive agents capable of using external tools based on user intent. However, current LLM agents still show limited ability to handle goal-oriented queries, which require decomposing a high-level objective into multiple interdependent API calls with correct planning and execution. Current approaches mainly rely on zero-shot evaluation due to the absence of training data. While proprietary closed-source models such as GPT-4 demonstrate strong reasoning abilities, smaller open-source models struggle to perform complex tool use effectively. Thus, we propose a novel training framework GOAT, which enables fine-tuning of LLM agents in a human annotation-free setting. GOAT automatically constructs synthetic datasets of goal-oriented API execution tasks directly from given API documents, equipping models with the ability to reason over interdependent calls and generate coherent responses. Through extensive experiments, we show that GOAT-trained agents achieve state-of-the-art performance across multiple existing goal-oriented benchmarks. In addition, we introduce GOATBench, a new goal-oriented API execution benchmark, and demonstrate that agents trained with GOAT also excel in this setting. These results highlight GOAT as a practical path toward building robust open-source LLM agents capable of complex reasoning and tool use.

**arXiv ID:** 2510.12218
</details>

<details>
<summary><strong>Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning</strong> - Dongrong Yang, Xin Wu, Yibo Xie, Xinyi Li, Qiuwen Wu, Jackie Wu, Yang Sheng - [[pdf]](https://arxiv.org/pdf/2510.11754)</summary>

**Abstract:** Radiation therapy treatment planning is an iterative, expertise-dependent process, and the growing burden of cancer cases has made reliance on manual planning increasingly unsustainable, underscoring the need for automation. In this study, we propose a workflow that leverages a large language model (LLM)-based agent to navigate inverse treatment planning for intensity-modulated radiation therapy (IMRT). The LLM agent was implemented to directly interact with a clinical treatment planning system (TPS) to iteratively extract intermediate plan states and propose new constraint values to guide inverse optimization. The agent's decision-making process is informed by current observations and previous optimization attempts and evaluations, allowing for dynamic strategy refinement. The planning process was performed in a zero-shot inference setting, where the LLM operated without prior exposure to manually generated treatment plans and was utilized without any fine-tuning or task-specific training. The LLM-generated plans were evaluated on twenty head-and-neck cancer cases against clinical manual plans, with key dosimetric endpoints analyzed and reported. The LLM-generated plans achieved comparable organ-at-risk (OAR) sparing relative to clinical plans while demonstrating improved hot spot control (Dmax: 106.5% vs. 108.8%) and superior conformity (conformity index: 1.18 vs. 1.39 for boost PTV; 1.82 vs. 1.88 for primary PTV). This study demonstrates the feasibility of a zero-shot, LLM-driven workflow for automated IMRT treatment planning in a commercial TPS. The proposed approach provides a generalizable and clinically applicable solution that could reduce planning variability and support broader adoption of AI-based planning strategies.

**arXiv ID:** 2510.11754
</details>

<details>
<summary><strong>Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design</strong> - Junhua Liu, Fanfan Lin, Xinze Li, Kwan Hui Lim, Shuai Zhao - [[pdf]](https://arxiv.org/pdf/2411.14214)</summary>

**Abstract:** LLM-based autonomous agents have recently shown strong capabilities in solving complex industrial design tasks. However, in domains aiming for carbon neutrality and high-performance renewable energy systems, current AI-assisted design automation methods face critical challenges in explainability, scalability, and practical usability. To address these limitations, we introduce PHIA (Physics-Informed Autonomous Agent), an LLM-driven system that automates modulation design for power converters in Power Electronics Systems with minimal human intervention. In contrast to traditional pipeline-based methods, PHIA incorporates an LLM-based planning module that interactively acquires and verifies design requirements via a user-friendly chat interface. This planner collaborates with physics-informed simulation and optimization components to autonomously generate and iteratively refine modulation designs. The interactive interface also supports interpretability by providing textual explanations and visual outputs throughout the design process. Experimental results show that PHIA reduces standard mean absolute error by 63.2% compared to the second-best benchmark and accelerates the overall design process by over 33 times. A user study involving 20 domain experts further confirms PHIA's superior design efficiency and usability, highlighting its potential to transform industrial design workflows in power electronics.

**arXiv ID:** 2411.14214
</details>

<details>
<summary><strong>Scaling Long-Horizon LLM Agent via Context-Folding</strong> - Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, Jiecao Chen - [[pdf]](https://arxiv.org/pdf/2510.11967)</summary>

**Abstract:** Large language model (LLM) agents are fundamentally constrained by context length on long-horizon tasks. We introduce Context-Folding, a framework that empowers agents to actively manage their working context. An agent can procedurally branch into a sub-trajectory to handle a subtask and then fold it upon completion, collapsing the intermediate steps while retaining a concise summary of the outcome. To make this behavior learnable, we develop an end-to-end reinforcement learning framework FoldGRPO with specific process rewards to encourage effective task decomposition and context management. On complex long-horizon tasks (Deep Research and SWE), our folding agent matches or outperforms the ReAct baselines while using an active context 10$\times$ smaller and significantly outperforms models that rely on summarization-based context management.

**arXiv ID:** 2510.11967
</details>

<details>
<summary><strong>HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities</strong> - Xiaoxue Ren, Penghao Jiang, Kaixin Li, Zhiyong Huang, Xiaoning Du, Jiaojiao Jiang, Zhenchang Xing, Jiamou Sun, Terry Yue Zhuo - [[pdf]](https://arxiv.org/pdf/2510.12200)</summary>

**Abstract:** Web applications are prime targets for cyberattacks as gateways to critical services and sensitive data. Traditional penetration testing is costly and expertise-intensive, making it difficult to scale with the growing web ecosystem. While language model agents show promise in cybersecurity, modern web applications demand visual understanding, dynamic content handling, and multi-step interactions that only computer-use agents (CUAs) can perform. Yet, their ability to discover and exploit vulnerabilities through graphical interfaces remains largely unexplored. We present HackWorld, the first framework for systematically evaluating CUAs' capabilities to exploit web application vulnerabilities via visual interaction. Unlike sanitized benchmarks, HackWorld includes 36 real-world applications across 11 frameworks and 7 languages, featuring realistic flaws such as injection vulnerabilities, authentication bypasses, and unsafe input handling. Using a Capture-the-Flag (CTF) setup, it tests CUAs' capacity to identify and exploit these weaknesses while navigating complex web interfaces. Evaluation of state-of-the-art CUAs reveals concerning trends: exploitation rates below 12% and low cybersecurity awareness. CUAs often fail at multi-step attack planning and misuse security tools. These results expose the current limitations of CUAs in web security contexts and highlight opportunities for developing more security-aware agents capable of effective vulnerability detection and exploitation.

**arXiv ID:** 2510.12200
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (24 papers)</h2></summary>

<details>
<summary><strong>EmboMatrix: A Scalable Training-Ground for Embodied Decision-Making</strong> - Zixing Lei, Sheng Yin, Yichen Xiong, Yuanzhuo Ding, Wenhao Huang, Yuxi Wei, Qingyao Xu, Yiming Li, Weixin Li, Yunhong Wang, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2510.12072)</summary>

**Abstract:** Embodied decision-making enables agents to translate high-level goals into executable actions through continuous interactions within the physical world, forming a cornerstone of general-purpose embodied intelligence. Large language models (LLMs), with their general decision-making capabilities, offer a promising path to realize this potential; however, LLMs trained solely on language lack exposure to physical environments, limiting their true embodied understanding. To bridge this gap, we propose the concept of a training ground: a comprehensive infrastructure that provides task and scene simulation, embodied interaction, and feedback signals, offering a one-stop solution for LLM acquire genuine embodied decision-making skills. In this work, we present EmboMatrix, the first training ground of its kind, providing massive and diverse tasks with efficient simulation and precise rewards. EmboMatrix incorporates a series of novel techniques: a multi-agent data engine for large-scale task and scene generation, a distributed heterogeneous-hardware system for scalable simulation, and a multi-level reward architecture for precise supervision. Leveraging EmboMatrix, we cultivate EmboBrain, an LLM whose embodied decision-making abilities emerge from extensive embodied interactions. Experiments show that EmboBrain-7B surpasses the 671B DeepSeek-R1 baseline by 9.5\% on two challenging embodied decision-making benchmarks, demonstrating the power of interactive, environment-grounded learning for building truly intelligent embodied agents.

**arXiv ID:** 2510.12072
</details>

<details>
<summary><strong>ToPolyAgent: AI Agents for Coarse-Grained Topological Polymer Simulations</strong> - Lijie Ding, Jan-Michael Carrillo, Changwoo Do - [[pdf]](https://arxiv.org/pdf/2510.12091)</summary>

**Abstract:** We introduce ToPolyAgent, a multi-agent AI framework for performing coarse-grained molecular dynamics (MD) simulations of topological polymers through natural language instructions. By integrating large language models (LLMs) with domain-specific computational tools, ToPolyAgent supports both interactive and autonomous simulation workflows across diverse polymer architectures, including linear, ring, brush, and star polymers, as well as dendrimers. The system consists of four LLM-powered agents: a Config Agent for generating initial polymer-solvent configurations, a Simulation Agent for executing LAMMPS-based MD simulations and conformational analyses, a Report Agent for compiling markdown reports, and a Workflow Agent for streamlined autonomous operations. Interactive mode incorporates user feedback loops for iterative refinements, while autonomous mode enables end-to-end task execution from detailed prompts. We demonstrate ToPolyAgent's versatility through case studies involving diverse polymer architectures under varying solvent condition, thermostats, and simulation lengths. Furthermore, we highlight its potential as a research assistant by directing it to investigate the effect of interaction parameters on the linear polymer conformation, and the influence of grafting density on the persistence length of the brush polymer. By coupling natural language interfaces with rigorous simulation tools, ToPolyAgent lowers barriers to complex computational workflows and advances AI-driven materials discovery in polymer science. It lays the foundation for autonomous and extensible multi-agent scientific research ecosystems.

**arXiv ID:** 2510.12091
</details>

<details>
<summary><strong>Inclusive Fitness as a Key Step Towards More Advanced Social Behaviors in Multi-Agent Reinforcement Learning Settings</strong> - Andries Rosseau, Raphaël Avalos, Ann Nowé - [[pdf]](https://arxiv.org/pdf/2510.12555)</summary>

**Abstract:** The competitive and cooperative forces of natural selection have driven the evolution of intelligence for millions of years, culminating in nature's vast biodiversity and the complexity of human minds. Inspired by this process, we propose a novel multi-agent reinforcement learning framework where each agent is assigned a genotype and where reward functions are modelled after the concept of inclusive fitness. An agent's genetic material may be shared with other agents, and our inclusive reward function naturally accounts for this. We study the resulting social dynamics in two types of network games with prisoner's dilemmas and find that our results align with well-established principles from biology, such as Hamilton's rule. Furthermore, we outline how this framework can extend to more open-ended environments with spatial and temporal structure, finite resources, and evolving populations. We hypothesize the emergence of an arms race of strategies, where each new strategy is a gradual improvement over earlier adaptations of other agents, effectively producing a multi-agent autocurriculum analogous to biological evolution. In contrast to the binary team-based structures prevalent in earlier research, our gene-based reward structure introduces a spectrum of cooperation ranging from full adversity to full cooperativeness based on genetic similarity, enabling unique non team-based social dynamics. For example, one agent having a mutual cooperative relationship with two other agents, while the two other agents behave adversarially towards each other. We argue that incorporating inclusive fitness in agents provides a foundation for the emergence of more strategically advanced and socially intelligent agents.

**arXiv ID:** 2510.12555
</details>

<details>
<summary><strong>Multi-Agent Debate for LLM Judges with Adaptive Stability Detection</strong> - Tianyu Hu, Zhen Tan, Song Wang, Huaizhi Qu, Tianlong Chen - [[pdf]](https://arxiv.org/pdf/2510.12697)</summary>

**Abstract:** With advancements in reasoning capabilities, Large Language Models (LLMs) are increasingly employed for automated judgment tasks. While LLMs-as-Judges offer promise in automating evaluations, current approaches often rely on simplistic aggregation methods (e.g., majority voting), which can fail even when individual agents provide correct answers. To address this, we propose a multi-agent debate judge framework where agents collaboratively reason and iteratively refine their responses. We formalize the debate process mathematically, analyzing agent interactions and proving that debate amplifies correctness compared to static ensembles. To enhance efficiency, we introduce a stability detection mechanism that models judge consensus dynamics via a time-varying Beta-Binomial mixture, with adaptive stopping based on distributional similarity (Kolmogorov-Smirnov test). This mechanism models the judges' collective correct rate dynamics using a time-varying mixture of Beta-Binomial distributions and employs an adaptive stopping criterion based on distributional similarity (Kolmogorov-Smirnov statistic). Experiments across multiple benchmarks and models demonstrate that our framework improves judgment accuracy over majority voting while maintaining computational efficiency.

**arXiv ID:** 2510.12697
</details>

<details>
<summary><strong>Ax-Prover: A Deep Reasoning Agentic Framework for Theorem Proving in Mathematics and Quantum Physics</strong> - Marco Del Tredici, Jacob McCarran, Benjamin Breen, Javier Aspuru Mijares, Weichen Winston Yin, Jacob M. Taylor, Frank Koppens, Dirk Englund - [[pdf]](https://arxiv.org/pdf/2510.12787)</summary>

**Abstract:** We present Ax-Prover, a multi-agent system for automated theorem proving in Lean that can solve problems across diverse scientific domains and operate either autonomously or collaboratively with human experts. To achieve this, Ax-Prover approaches scientific problem solving through formal proof generation, a process that demands both creative reasoning and strict syntactic rigor. Ax-Prover meets this challenge by equipping Large Language Models (LLMs), which provide knowledge and reasoning, with Lean tools via the Model Context Protocol (MCP), which ensure formal correctness. To evaluate its performance as an autonomous prover, we benchmark our approach against frontier LLMs and specialized prover models on two public math benchmarks and on two Lean benchmarks we introduce in the fields of abstract algebra and quantum theory. On public datasets, Ax-Prover is competitive with state-of-the-art provers, while it largely outperform them on the new benchmarks. This shows that, unlike specialized systems that struggle to generalize, our tool-based agentic theorem prover approach offers a generalizable methodology for formal verification across diverse scientific domains. Furthermore, we demonstrate Ax-Prover's assistant capabilities in a practical use case, showing how it enabled an expert mathematician to formalize the proof of a complex cryptography theorem.

**arXiv ID:** 2510.12787
</details>

<details>
<summary><strong>Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning</strong> - Simin Li, Zihao Mao, Hanxiao Li, Zonglei Jing, Zhuohang bian, Jun Guo, Li Wang, Zhuoran Han, Ruixiao Xu, Xin Yu, Chengdong Ma, Yuqing Ma, Bo An, Yaodong Yang, Weifeng Lv, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2510.11824)</summary>

**Abstract:** In cooperative Multi-Agent Reinforcement Learning (MARL), it is a common practice to tune hyperparameters in ideal simulated environments to maximize cooperative performance. However, policies tuned for cooperation often fail to maintain robustness and resilience under real-world uncertainties. Building trustworthy MARL systems requires a deep understanding of robustness, which ensures stability under uncertainties, and resilience, the ability to recover from disruptions--a concept extensively studied in control systems but largely overlooked in MARL. In this paper, we present a large-scale empirical study comprising over 82,620 experiments to evaluate cooperation, robustness, and resilience in MARL across 4 real-world environments, 13 uncertainty types, and 15 hyperparameters. Our key findings are: (1) Under mild uncertainty, optimizing cooperation improves robustness and resilience, but this link weakens as perturbations intensify. Robustness and resilience also varies by algorithm and uncertainty type. (2) Robustness and resilience do not generalize across uncertainty modalities or agent scopes: policies robust to action noise for all agents may fail under observation noise on a single agent. (3) Hyperparameter tuning is critical for trustworthy MARL: surprisingly, standard practices like parameter sharing, GAE, and PopArt can hurt robustness, while early stopping, high critic learning rates, and Leaky ReLU consistently help. By optimizing hyperparameters only, we observe substantial improvement in cooperation, robustness and resilience across all MARL backbones, with the phenomenon also generalizing to robust MARL methods across these backbones. Code and results available at this https URL .

**arXiv ID:** 2510.11824
</details>

<details>
<summary><strong>Diffusion Models for Reinforcement Learning: Foundations, Taxonomy, and Development</strong> - Changfu Xu, Jianxiong Guo, Yuzhu Liang, Haiyang Huang, Haodong Zou, Xi Zheng, Shui Yu, Xiaowen Chu, Jiannong Cao, Tian Wang - [[pdf]](https://arxiv.org/pdf/2510.12253)</summary>

**Abstract:** Diffusion Models (DMs), as a leading class of generative models, offer key advantages for reinforcement learning (RL), including multi-modal expressiveness, stable training, and trajectory-level planning. This survey delivers a comprehensive and up-to-date synthesis of diffusion-based RL. We first provide an overview of RL, highlighting its challenges, and then introduce the fundamental concepts of DMs, investigating how they are integrated into RL frameworks to address key challenges in this research field. We establish a dual-axis taxonomy that organizes the field along two orthogonal dimensions: a function-oriented taxonomy that clarifies the roles DMs play within the RL pipeline, and a technique-oriented taxonomy that situates implementations across online versus offline learning regimes. We also provide a comprehensive examination of this progression from single-agent to multi-agent domains, thereby forming several frameworks for DM-RL integration and highlighting their practical utility. Furthermore, we outline several categories of successful applications of diffusion-based RL across diverse domains, discuss open research issues of current methodologies, and highlight key directions for future research to advance the field. Finally, we summarize the survey to identify promising future development directions. We are actively maintaining a GitHub repository (this https URL) for papers and other related resources to apply DMs for RL.

**arXiv ID:** 2510.12253
</details>

<details>
<summary><strong>Scaling Multi-Agent Epistemic Planning through GNN-Derived Heuristics</strong> - Giovanni Briglia, Francesco Fabiano, Stefano Mariani - [[pdf]](https://arxiv.org/pdf/2508.12840)</summary>

**Abstract:** Multi-agent Epistemic Planning (MEP) is an autonomous planning framework for reasoning about both the physical world and the beliefs of agents, with applications in domains where information flow and awareness among agents are critical. The richness of MEP requires states to be represented as Kripke structures, i.e., directed labeled graphs. This representation limits the applicability of existing heuristics, hindering the scalability of epistemic solvers, which must explore an exponential search space without guidance, resulting often in intractability. To address this, we exploit Graph Neural Networks (GNNs) to learn patterns and relational structures within epistemic states, to guide the planning process. GNNs, which naturally capture the graph-like nature of Kripke models, allow us to derive meaningful estimates of state quality -- e.g., the distance from the nearest goal -- by generalizing knowledge obtained from previously solved planning instances. We integrate these predictive heuristics into an epistemic planning pipeline and evaluate them against standard baselines, showing improvements in the scalability of multi-agent epistemic planning.

**arXiv ID:** 2508.12840
</details>

<details>
<summary><strong>ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery</strong> - Junda Wang, Zonghai Yao, Lingxi Li, Junhui Qian, Zhichao Yang, Hong Yu - [[pdf]](https://arxiv.org/pdf/2508.20996)</summary>

**Abstract:** Substance use disorders (SUDs) affect millions of people, and relapses are common, requiring multi-session treatments. Access to care is limited, which contributes to the challenge of recovery support. We present \textbf{ChatThero}, an innovative low-cost, multi-session, stressor-aware, and memory-persistent autonomous \emph{language agent} designed to facilitate long-term behavior change and therapeutic support in addiction recovery. Unlike existing work that mostly finetuned large language models (LLMs) on patient-therapist conversation data, ChatThero was trained in a multi-agent simulated environment that mirrors real therapy. We created anonymized patient profiles from recovery communities (e.g., Reddit). We classify patients as \texttt{easy}, \texttt{medium}, and \texttt{difficult}, three scales representing their resistance to recovery. We created an external environment by introducing stressors (e.g., social determinants of health) to simulate real-world situations. We dynamically inject clinically-grounded therapeutic strategies (motivational interview and cognitive behavioral therapy). Our evaluation, conducted by both human (blinded clinicians) and LLM-as-Judge, shows that ChatThero is superior in empathy and clinical relevance. We show that stressor simulation improves robustness of ChatThero. Explicit stressors increase relapse-like setbacks, matching real-world patterns. We evaluate ChatThero with behavioral change metrics. On a 1--5 scale, ChatThero raises \texttt{motivation} by $+1.71$ points (from $2.39$ to $4.10$) and \texttt{confidence} by $+1.67$ points (from $1.52$ to $3.19$), substantially outperforming GPT-5. On \texttt{difficult} patients, ChatThero reaches the success milestone with $26\%$ fewer turns than GPT-5.

**arXiv ID:** 2508.20996
</details>

<details>
<summary><strong>MapAgent: A Hierarchical Agent for Geospatial Reasoning with Dynamic Map Tool Integration</strong> - Md Hasebul Hasan, Mahir Labib Dihan, Tanzima Hashem, Mohammed Eunus Ali, Md Rizwan Parvez - [[pdf]](https://arxiv.org/pdf/2509.05933)</summary>

**Abstract:** Agentic AI has significantly extended the capabilities of large language models (LLMs) by enabling complex reasoning and tool use. However, most existing frameworks are tailored to domains such as mathematics, coding, or web automation, and fall short on geospatial tasks that require spatial reasoning, multi-hop planning, and real-time map interaction. To address these challenges, we introduce MapAgent, a hierarchical multi-agent plug-and-play framework with customized toolsets and agentic scaffolds for map-integrated geospatial reasoning. Unlike existing flat agent-based approaches that treat tools uniformly-often overwhelming the LLM when handling similar but subtly different geospatial APIs-MapAgent decouples planning from execution. A high-level planner decomposes complex queries into subgoals, which are routed to specialized modules. For tool-heavy modules-such as map-based services-we then design a dedicated map-tool agent that efficiently orchestrates related APIs adaptively in parallel to effectively fetch geospatial data relevant for the query, while simpler modules (e.g., solution generation or answer extraction) operate without additional agent overhead. This hierarchical design reduces cognitive load, improves tool selection accuracy, and enables precise coordination across similar APIs. We evaluate MapAgent on four diverse geospatial benchmarks-MapEval-Textual, MapEval-API, MapEval-Visual, and MapQA-and demonstrate substantial gains over state-of-the-art tool-augmented and agentic baselines. We open-source our framwork at this https URL.

**arXiv ID:** 2509.05933
</details>

<details>
<summary><strong>Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining</strong> - Crystal Qian, Kehang Zhu, John Horton, Benjamin S. Manning, Vivian Tsai, James Wexler, Nithum Thain - [[pdf]](https://arxiv.org/pdf/2509.09071)</summary>

**Abstract:** As large language models (LLMs) are increasingly embedded in collaborative human activities such as business negotiations and group coordination, it becomes critical to evaluate both the performance gains they can achieve and how they interact in dynamic, multi-agent environments. Unlike traditional statistical agents such as Bayesian models, which may excel under well-specified conditions, large language models (LLMs) can generalize across diverse, real-world scenarios, raising new questions about how their strategies and behaviors compare to those of humans and other agent types. In this work, we compare outcomes and behavioral dynamics across humans (N = 216), LLMs (GPT-4o, Gemini 1.5 Pro), and Bayesian agents in a dynamic negotiation setting under identical conditions. Bayesian agents extract the highest surplus through aggressive optimization, at the cost of frequent trade rejections. Humans and LLMs achieve similar overall surplus, but through distinct behaviors: LLMs favor conservative, concessionary trades with few rejections, while humans employ more strategic, risk-taking, and fairness-oriented behaviors. Thus, we find that performance parity -- a common benchmark in agent evaluation -- can conceal fundamental differences in process and alignment, which are critical for practical deployment in real-world coordination tasks. By establishing foundational behavioral baselines under matched conditions, this work provides a baseline for future studies in more applied, variable-rich environments.

**arXiv ID:** 2509.09071
</details>

<details>
<summary><strong>L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)</strong> - Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu - [[pdf]](https://arxiv.org/pdf/2510.07363)</summary>

**Abstract:** The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.

**arXiv ID:** 2510.07363
</details>

<details>
<summary><strong>AgentBreeder: Mitigating the AI Safety Risks of Multi-Agent Scaffolds via Self-Improvement</strong> - J Rosser, Jakob Foerster - [[pdf]](https://arxiv.org/pdf/2502.00757)</summary>

**Abstract:** Scaffolding Large Language Models (LLMs) into multi-agent systems often improves performance on complex tasks, but the safety impact of such scaffolds has not been thoroughly explored. We introduce AgentBreeder, a framework for multi-objective self-improving evolutionary search over scaffolds. We evaluate discovered scaffolds on widely recognized reasoning, mathematics, and safety benchmarks and compare them with popular baselines. In "blue" mode, we see a 79.4% average uplift in safety benchmark performance while maintaining or improving capability scores. In "red" mode, we find adversarially weak scaffolds emerging concurrently with capability optimization. Our work demonstrates the risks of multi-agent scaffolding and provides a framework for mitigating them. Code is available at this https URL.

**arXiv ID:** 2502.00757
</details>

<details>
<summary><strong>Multi-Agent Autonomous Driving Systems with Large Language Models: A Survey of Recent Advances</strong> - Yaozu Wu, Dongyuan Li, Yankai Chen, Renhe Jiang, Henry Peng Zou, Wei-Chieh Huang, Yangning Li, Liancheng Fang, Zhen Wang, Philip S. Yu - [[pdf]](https://arxiv.org/pdf/2502.16804)</summary>

**Abstract:** Autonomous Driving Systems (ADSs) are revolutionizing transportation by reducing human intervention, improving operational efficiency, and enhancing safety. Large Language Models (LLMs) have been integrated into ADSs to support high-level decision-making through their powerful reasoning, instruction-following, and communication abilities. However, LLM-based single-agent ADSs face three major challenges: limited perception, insufficient collaboration, and high computational demands. To address these issues, recent advances in LLM-based multi-agent ADSs leverage language-driven communication and coordination to enhance inter-agent collaboration. This paper provides a frontier survey of this emerging intersection between NLP and multi-agent ADSs. We begin with a background introduction to related concepts, followed by a categorization of existing LLM-based methods based on different agent interaction modes. We then discuss agent-human interactions in scenarios where LLM-based agents engage with humans. Finally, we summarize key applications, datasets, and challenges to support future research.

**arXiv ID:** 2502.16804
</details>

<details>
<summary><strong>Heterogeneous RBCs via deep multi-agent reinforcement learning</strong> - Federico Gabriele, Aldo Glielmo, Marco Taboga - [[pdf]](https://arxiv.org/pdf/2510.12272)</summary>

**Abstract:** Current macroeconomic models with agent heterogeneity can be broadly divided into two main groups. Heterogeneous-agent general equilibrium (GE) models, such as those based on Heterogeneous Agents New Keynesian (HANK) or Krusell-Smith (KS) approaches, rely on GE and 'rational expectations', somewhat unrealistic assumptions that make the models very computationally cumbersome, which in turn limits the amount of heterogeneity that can be modelled. In contrast, agent-based models (ABMs) can flexibly encompass a large number of arbitrarily heterogeneous agents, but typically require the specification of explicit behavioural rules, which can lead to a lengthy trial-and-error model-development process. To address these limitations, we introduce MARL-BC, a framework that integrates deep multi-agent reinforcement learning (MARL) with Real Business Cycle (RBC) models. We demonstrate that MARL-BC can: (1) recover textbook RBC results when using a single agent; (2) recover the results of the mean-field KS model using a large number of identical agents; and (3) effectively simulate rich heterogeneity among agents, a hard task for traditional GE approaches. Our framework can be thought of as an ABM if used with a variety of heterogeneous interacting agents, and can reproduce GE results in limit cases. As such, it is a step towards a synthesis of these often opposed modelling paradigms.

**arXiv ID:** 2510.12272
</details>

<details>
<summary><strong>Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs</strong> - Yurun Chen, Xavier Hu, Yuhan Liu, Ziqi Wang, Zeyi Liao, Lin Chen, Feng Wei, Yuxi Qian, Bo Zheng, Keting Yin, Shengyu Zhang - [[pdf]](https://arxiv.org/pdf/2510.00507)</summary>

**Abstract:** As multimodal LLM-driven agents continue to advance in autonomy and generalization, evaluation based on static datasets can no longer adequately assess their true capabilities in dynamic environments and diverse tasks. Existing LLM-based synthetic data methods are largely designed for LLM training and evaluation, and thus cannot be directly applied to agent tasks that require tool use and interactive capabilities. While recent studies have explored automatic agent task generation with LLMs, most efforts remain limited to text or image analysis, without systematically modeling multi-step interactions in web environments. To address these challenges, we propose Graph2Eval, a knowledge graph-based framework that automatically generates both multimodal document comprehension tasks and web interaction tasks, enabling comprehensive evaluation of agents' reasoning, collaboration, and interactive capabilities. In our approach, knowledge graphs constructed from multi-source external data serve as the task space, where we translate semantic relations into structured multimodal tasks using subgraph sampling, task templates, and meta-paths. A multi-stage filtering pipeline based on node reachability, LLM scoring, and similarity analysis is applied to guarantee the quality and executability of the generated tasks. Furthermore, Graph2Eval supports end-to-end evaluation of multiple agent types (Single-Agent, Multi-Agent, Web Agent) and measures reasoning, collaboration, and interaction capabilities. We instantiate the framework with Graph2Eval-Bench, a curated dataset of 1,319 tasks spanning document comprehension and web interaction scenarios. Experiments show that Graph2Eval efficiently generates tasks that differentiate agent and model performance, revealing gaps in reasoning, collaboration, and web interaction across different settings and offering a new perspective for agent evaluation.

**arXiv ID:** 2510.00507
</details>

<details>
<summary><strong>Autonomous vehicles need social awareness to find optima in multi-agent reinforcement learning routing games</strong> - Anastasia Psarou, Łukasz Gorczyca, Dominik Gaweł, Rafał Kucharski - [[pdf]](https://arxiv.org/pdf/2510.11410)</summary>

**Abstract:** Previous work has shown that when multiple selfish Autonomous Vehicles (AVs) are introduced to future cities and start learning optimal routing strategies using Multi-Agent Reinforcement Learning (MARL), they may destabilize traffic systems, as they would require a significant amount of time to converge to the optimal solution, equivalent to years of real-world commuting.
We demonstrate that moving beyond the selfish component in the reward significantly relieves this issue. If each AV, apart from minimizing its own travel time, aims to reduce its impact on the system, this will be beneficial not only for the system-wide performance but also for each individual player in this routing game.
By introducing an intrinsic reward signal based on the marginal cost matrix, we significantly reduce training time and achieve convergence more reliably. Marginal cost quantifies the impact of each individual action (route-choice) on the system (total travel time). Including it as one of the components of the reward can reduce the degree of non-stationarity by aligning agents' objectives. Notably, the proposed counterfactual formulation preserves the system's equilibria and avoids oscillations.
Our experiments show that training MARL algorithms with our novel reward formulation enables the agents to converge to the optimal solution, whereas the baseline algorithms fail to do so. We show these effects in both a toy network and the real-world network of Saint-Arnoult. Our results optimistically indicate that social awareness (i.e., including marginal costs in routing decisions) improves both the system-wide and individual performance of future urban systems with AVs.

**arXiv ID:** 2510.11410
</details>

<details>
<summary><strong>Optimistic Multi-Agent Policy Gradient</strong> - Wenshuai Zhao, Yi Zhao, Zhiyuan Li, Juho Kannala, Joni Pajarinen - [[pdf]](https://arxiv.org/pdf/2311.01953)</summary>

**Abstract:** *Relative overgeneralization* (RO) occurs in cooperative multi-agent learning tasks when agents converge towards a suboptimal joint policy due to overfitting to suboptimal behavior of other agents. No methods have been proposed for addressing RO in multi-agent policy gradient (MAPG) methods although these methods produce state-of-the-art results. To address this gap, we propose a general, yet simple, framework to enable optimistic updates in MAPG methods that alleviate the RO problem. Our approach involves clipping the advantage to eliminate negative values, thereby facilitating optimistic updates in MAPG. The optimism prevents individual agents from quickly converging to a local optimum. Additionally, we provide a formal analysis to show that the proposed method retains optimality at a fixed point. In extensive evaluations on a diverse set of tasks including the *Multi-agent MuJoCo* and *Overcooked* benchmarks, our method outperforms strong baselines on 13 out of 19 tested tasks and matches the performance on the rest.

**arXiv ID:** 2311.01953
</details>

<details>
<summary><strong>Stronger Together: On-Policy Reinforcement Learning for Collaborative LLMs</strong> - Yujie Zhao, Lanxiang Hu, Yang Wang, Minmin Hou, Hao Zhang, Ke Ding, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2510.11062)</summary>

**Abstract:** Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models.
We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: this https URL.

**arXiv ID:** 2510.11062
</details>

<details>
<summary><strong>EmoDebt: Bayesian-Optimized Emotional Intelligence for Strategic Agent-to-Agent Debt Recovery</strong> - Yunbo Long, Yuhan Liu, Liming Xu, Alexandra Brintrup - [[pdf]](https://arxiv.org/pdf/2503.21080)</summary>

**Abstract:** The emergence of autonomous Large Language Model (LLM) agents has created a new ecosystem of strategic, agent-to-agent interactions. However, a critical challenge remains unaddressed: in high-stakes, emotion-sensitive domains like debt collection, LLM agents pre-trained on human dialogue are vulnerable to exploitation by adversarial counterparts who simulate negative emotions to derail negotiations. To fill this gap, we first contribute a novel dataset of simulated debt recovery scenarios and a multi-agent simulation framework. Within this framework, we introduce EmoDebt, an LLM agent architected for robust performance. Its core innovation is a Bayesian-optimized emotional intelligence engine that reframes a model's ability to express emotion in negotiation as a sequential decision-making problem. Through online learning, this engine continuously tunes EmoDebt's emotional transition policies, discovering optimal counter-strategies against specific debtor tactics. Extensive experiments on our proposed benchmark demonstrate that EmoDebt achieves significant strategic robustness, substantially outperforming non-adaptive and emotion-agnostic baselines across key performance metrics, including success rate and operational efficiency. By introducing both a critical benchmark and a robustly adaptive agent, this work establishes a new foundation for deploying strategically robust LLM agents in adversarial, emotion-sensitive debt interactions.

**arXiv ID:** 2503.21080
</details>

<details>
<summary><strong>MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems</strong> - Xuanming Zhang, Yuxuan Chen, Samuel Yeh, Sharon Li - [[pdf]](https://arxiv.org/pdf/2505.18943)</summary>

**Abstract:** Human social interactions depend on the ability to infer others' unspoken intentions, emotions, and beliefs-a cognitive skill grounded in the psychological concept of Theory of Mind (ToM). While large language models (LLMs) excel in semantic understanding tasks, they struggle with the ambiguity and contextual nuance inherent in human communication. To bridge this gap, we introduce MetaMind, a multi-agent framework inspired by psychological theories of metacognition, designed to emulate human-like social reasoning. MetaMind decomposes social understanding into three collaborative stages: (1) a Theory-of-Mind Agent generates hypotheses about user mental states (e.g., intent, emotion), (2) a Moral Agent refines these hypotheses using cultural norms and ethical constraints, and (3) a Response Agent generates contextually appropriate responses while validating alignment with inferred intent. Our framework achieves state-of-the-art performance across three challenging benchmarks, with 35.7% improvement in real-world social scenarios and 6.2% gain in ToM reasoning. Notably, it enables LLMs to match human-level performance on key ToM tasks for the first time. Ablation studies confirm the necessity of all components, which showcase the framework's ability to balance contextual plausibility, social appropriateness, and user adaptation. This work advances AI systems toward human-like social intelligence, with applications in empathetic dialogue and culturally sensitive interactions. Code is available at this https URL.

**arXiv ID:** 2505.18943
</details>

<details>
<summary><strong>DoctorAgent-RL: A Multi-Agent Collaborative Reinforcement Learning System for Multi-Turn Clinical Dialogue</strong> - Yichun Feng, Jiawei Wang, Lu Zhou, Zhen Lei, Yixue Li - [[pdf]](https://arxiv.org/pdf/2505.19630)</summary>

**Abstract:** Large language models (LLMs) have demonstrated excellent capabilities in the field of biomedical question answering, but their application in real-world clinical consultations still faces core challenges. Single-round consultation systems require patients to describe all symptoms upfront, leading to vague diagnosis with unclear complaints. Traditional multi-turn dialogue models, constrained by static supervised learning, lack flexibility and fail to intelligently extract key clinical information. To address these limitations, we propose \Ours{}, a reinforcement learning (RL)-based multi-agent collaborative framework that models medical consultations as a dynamic decision-making process under uncertainty. The doctor agent continuously optimizes its questioning strategy within the RL framework through multi-turn interactions with the patient agent, dynamically adjusting its information-gathering path based on comprehensive rewards from the Consultation Evaluator. This RL fine-tuning mechanism enables LLMs to autonomously develop interaction strategies aligned with clinical reasoning logic, rather than superficially imitating patterns in existing dialogue data. Notably, we constructed MTMedDialog, the first English multi-turn medical consultation dataset capable of simulating patient interactions. Experiments demonstrate that \Ours{} outperforms existing models in both multi-turn reasoning capability and final diagnostic performance. This approach shows immense practical value by reducing misdiagnosis risks in time-pressured settings, freeing clinicians for complex cases, and pioneering a strategy to optimize medical resource allocation and alleviate workforce shortages. Code and data are available at this https URL

**arXiv ID:** 2505.19630
</details>

<details>
<summary><strong>MooseAgent: A LLM Based Multi-agent Framework for Automating Moose Simulation</strong> - Tao Zhang, Zhenhai Liu, Yong Xin, Yongjun Jiao - [[pdf]](https://arxiv.org/pdf/2504.08621)</summary>

**Abstract:** The Finite Element Method (FEM) is widely used in engineering and scientific computing, but its pre-processing, solver configuration, and post-processing stages are often time-consuming and require specialized knowledge. This paper proposes an automated solution framework, MooseAgent, for the multi-physics simulation framework MOOSE, which combines large-scale pre-trained language models (LLMs) with a multi-agent system. The framework uses LLMs to understand user-described simulation requirements in natural language and employs task decomposition and multi-round iterative verification strategies to automatically generate MOOSE input files. To improve accuracy and reduce model hallucinations, the system builds and utilizes a vector database containing annotated MOOSE input cards and function documentation. We conducted experimental evaluations on several typical cases, including heat transfer, mechanics, phase field, and multi-physics coupling. The results show that MooseAgent can automate the MOOSE simulation process to a certain extent, especially demonstrating a high success rate when dealing with relatively simple single-physics problems. The main contribution of this research is the proposal of a multi-agent automated framework for MOOSE, which validates its potential in simplifying finite element simulation processes and lowering the user barrier, providing new ideas for the development of intelligent finite element simulation software. The code for the MooseAgent framework proposed in this paper has been open-sourced and is available at this https URL

**arXiv ID:** 2504.08621
</details>

<details>
<summary><strong>PEAR: Planner-Executor Agent Robustness Benchmark</strong> - Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing - [[pdf]](https://arxiv.org/pdf/2510.07505)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

**arXiv ID:** 2510.07505
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>Characterizing Agent-Based Model Dynamics via $ε$-Machines and Kolmogorov-Style Complexity</strong> - Roberto Garrone - [[pdf]](https://arxiv.org/pdf/2510.12729)</summary>

**Abstract:** We propose a two-level information-theoretic framework for characterizing the informational organization of Agent-Based Model (ABM) dynamics within the broader paradigm of Complex Adaptive Systems (CAS). At the macro level, a pooled $\epsilon$-machine is reconstructed as a reference model that summarizes the system-wide informational regime. At the micro level, $\epsilon$-machines are reconstructed for each caregiver-elder dyad and variable, and are complemented with algorithm-agnostic Kolmogorov-style measures, including normalized LZ78 complexity and bits per symbol from lossless compression. The resulting feature set $\{h_{\mu}, C_{\mu}, E, \mathrm{LZ78}, \mathrm{bps}\}$ enables distributional analysis, stratified comparisons, and unsupervised clustering across agents and scenarios. This dual-scale design preserves agent heterogeneity while providing an interpretable macro-level baseline, aligning ABM practice with CAS principles of emergence, feedback, and adaptation. A case study on caregiver-elder interactions illustrates the framework's implementation; the results and discussion will be completed following final simulation runs.

**arXiv ID:** 2510.12729
</details>

<details>
<summary><strong>R-WoM: Retrieval-augmented World Model For Computer-use Agents</strong> - Kai Mei, Jiang Guo, Shuaichen Chang, Mingwen Dong, Dongkyu Lee, Xing Niu, Jiarong Jiang - [[pdf]](https://arxiv.org/pdf/2510.11892)</summary>

**Abstract:** Large Language Models (LLMs) can serve as world models to enhance agent decision-making in digital environments by simulating future states and predicting action outcomes, potentially eliminating costly trial-and-error exploration. However, this capability is fundamentally limited by LLMs' tendency toward hallucination and their reliance on static training knowledge, which can lead to compounding errors that inhibit long-horizon simulations. To systematically investigate whether LLMs are appropriate for world modeling, we probe two core capabilities of world models--future state prediction and reward estimation--through three tasks: next-state identification, full-procedure planning alignment, and milestone transition recognition. Our analysis shows that while LLMs effectively capture immediate next states and identify meaningful state transitions, their performance rapidly degrades in full-procedure planning. This highlights LLMs' limitations in reliably modeling environment dynamics over long horizons. To address these limitations, we propose the Retrieval-augmented World Model (R-WoM), which grounds LLM simulations by incorporating factual, up-to-date knowledge retrieved from external tutorials. Experiments show that R-WoM achieves substantial improvements of up to 25.3% (OSWorld) and 18.1% (WebArena) compared to baselines, with particular advantages in longer-horizon simulations.

**arXiv ID:** 2510.11892
</details>

<details>
<summary><strong>BrowserAgent: Building Web Agents with Human-Inspired Web Browsing Actions</strong> - Tao Yu, Zhengbo Zhang, Zhiheng Lyu, Junhao Gong, Hongzhu Yi, Xinming Wang, Yuxuan Zhou, Jiabing Yang, Ping Nie, Yan Huang, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2510.10666)</summary>

**Abstract:** Efficiently solving real-world problems with LLMs increasingly hinges on their ability to interact with dynamic web environments and autonomously acquire external information. While recent research like Search-R1 and WebDancer demonstrates strong performance in solving web tasks, they heavily rely on additional tools to convert the interactive web environment into static text content. This is in contrast to human browsing behaviors, which involve diverse interactions with the browser, such as scrolling, clicking, and typing. In this paper, we propose BrowserAgent, a more interactive agent that solves complex tasks through human-inspired browser actions. BrowserAgent operates directly on raw web pages via Playwright through a set of predefined browser actions. We adopt a two-stage training (Supervised Fine-Tuning (SFT) and Rejection Fine-Tuning (RFT)) to improve the model's generalization abilities. Despite using significantly less training data than Search-R1, BrowserAgent achieves more competitive results across different Open-QA tasks. Additionally, we introduce an explicit memory mechanism to store key conclusions across steps, further enhancing the model's reasoning capabilities for long-horizon tasks. Notably, BrowserAgent-7B can achieve around 20\% improvement over Search-R1 on multi-hop QA tasks like HotpotQA, 2Wiki, and Bamboogle. These results indicate that BrowserAgent can serve as a more advanced framework for more interactive and scalable web agents.

**arXiv ID:** 2510.10666
</details>

<details>
<summary><strong>Embodied Natural Language Interaction (NLI): Speech Input Patterns in Immersive Analytics</strong> - Hyemi Song, Matthew Johnson, Kirsten Whitley, Eric Krokos, Amitabh Varshney - [[pdf]](https://arxiv.org/pdf/2510.12156)</summary>

**Abstract:** Embodiment shapes how users verbally express intent when interacting with data through speech interfaces in immersive analytics. Despite growing interest in Natural Language Interaction (NLI) for visual analytics in immersive environments, users' speech patterns and their use of embodiment cues in speech remain underexplored. Understanding their interplay is crucial to bridging the gap between users' intent and an immersive analytic system. To address this, we report the results from 15 participants in a user study conducted using the Wizard of Oz method. We performed axial coding on 1,280 speech acts derived from 734 utterances, examining how analysis tasks are carried out with embodiment and linguistic features. Next, we measured speech input uncertainty for each analysis task using the semantic entropy of utterances, estimating how uncertain users' speech inputs appear to an analytic system. Through these analyses, we identified five speech input patterns, showing that users dynamically blend embodied and non-embodied speech acts depending on data analysis tasks, phases, and embodiment reliance driven by the counts and types of embodiment cues in each utterance. We then examined how these patterns align with user reflections on factors that challenge speech interaction during the study. Finally, we propose design implications aligned with the five patterns.

**arXiv ID:** 2510.12156
</details>

<details>
<summary><strong>AgentBuilder: Exploring Scaffolds for Prototyping User Experiences of Interface Agents</strong> - Jenny T. Liang, Titus Barik, Jeffrey Nichols, Eldon Schoop, Ruijia Cheng - [[pdf]](https://arxiv.org/pdf/2510.04452)</summary>

**Abstract:** Interface agents powered by generative AI models (referred to as "agents") can automate actions based on user commands. An important aspect of developing agents is their user experience (i.e., agent experience). There is a growing need to provide scaffolds for a broader set of individuals beyond AI engineers to prototype agent experiences, since they can contribute valuable perspectives to designing agent experiences. In this work, we explore the affordances agent prototyping systems should offer by conducting a requirements elicitation study with 12 participants with varying experience with agents. We identify key activities in agent experience prototyping and the desired capabilities of agent prototyping systems. We instantiate those capabilities in the AgentBuilder design probe for agent prototyping. We conduct an in situ agent prototyping study with 14 participants using AgentBuilder to validate the design requirements and elicit insights on how developers prototype agents and what their needs are in this process.

**arXiv ID:** 2510.04452
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (24 papers)</h2></summary>

<details>
<summary><strong>AI Agents for the Dhumbal Card Game: A Comparative Study</strong> - Sahaj Raj Malla - [[pdf]](https://arxiv.org/pdf/2510.11736)</summary>

**Abstract:** This study evaluates Artificial Intelligence (AI) agents for Dhumbal, a culturally significant multiplayer card game with imperfect information, through a systematic comparison of rule-based, search-based, and learning-based strategies. We formalize Dhumbal's mechanics and implement diverse agents, including heuristic approaches (Aggressive, Conservative, Balanced, Opportunistic), search-based methods such as Monte Carlo Tree Search (MCTS) and Information Set Monte Carlo Tree Search (ISMCTS), and reinforcement learning approaches including Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), and a random baseline. Evaluation involves within-category tournaments followed by a cross-category championship. Performance is measured via win rate, economic outcome, Jhyap success, cards discarded per round, risk assessment, and decision efficiency. Statistical significance is assessed using Welch's t-test with Bonferroni correction, effect sizes via Cohen's d, and 95% confidence intervals (CI). Across 1024 simulated rounds, the rule-based Aggressive agent achieves the highest win rate (88.3%, 95% CI: [86.3, 90.3]), outperforming ISMCTS (9.0%) and PPO (1.5%) through effective exploitation of Jhyap declarations. The study contributes a reproducible AI framework, insights into heuristic efficacy under partial information, and open-source code, thereby advancing AI research and supporting digital preservation of cultural games.

**arXiv ID:** 2510.11736
</details>

<details>
<summary><strong>CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing</strong> - Chathurangi Shyalika, Aryaman Sharma, Fadi El Kalach, Utkarshani Jaimini, Cory Henson, Ramy Harik, Amit Sheth - [[pdf]](https://arxiv.org/pdf/2510.12033)</summary>

**Abstract:** Modern manufacturing environments demand not only accurate predictions but also interpretable insights to process anomalies, root causes, and potential interventions. Existing AI systems often function as isolated black boxes, lacking the seamless integration of prediction, explanation, and causal reasoning required for a unified decision-support solution. This fragmentation limits their trustworthiness and practical utility in high-stakes industrial environments. In this work, we present CausalTrace, a neurosymbolic causal analysis module integrated into the SmartPilot industrial CoPilot. CausalTrace performs data-driven causal analysis enriched by industrial ontologies and knowledge graphs, including advanced functions such as causal discovery, counterfactual reasoning, and root cause analysis (RCA). It supports real-time operator interaction and is designed to complement existing agents by offering transparent, explainable decision support. We conducted a comprehensive evaluation of CausalTrace using multiple causal assessment methods and the C3AN framework (i.e. Custom, Compact, Composite AI with Neurosymbolic Integration), which spans principles of robustness, intelligence, and trustworthiness. In an academic rocket assembly testbed, CausalTrace achieved substantial agreement with domain experts (ROUGE-1: 0.91 in ontology QA) and strong RCA performance (MAP@3: 94%, PR@2: 97%, MRR: 0.92, Jaccard: 0.92). It also attained 4.59/5 in the C3AN evaluation, demonstrating precision and reliability for live deployment.

**arXiv ID:** 2510.12033
</details>

<details>
<summary><strong>ResearStudio: A Human-Intervenable Framework for Building Controllable Deep-Research Agents</strong> - Linyi Yang, Yixuan Weng - [[pdf]](https://arxiv.org/pdf/2510.12194)</summary>

**Abstract:** Current deep-research agents run in a ''fire-and-forget'' mode: once started, they give users no way to fix errors or add expert knowledge during execution. We present ResearStudio, the first open-source framework that places real-time human control at its core. The system follows a Collaborative Workshop design. A hierarchical Planner-Executor writes every step to a live ''plan-as-document,'' a fast communication layer streams each action, file change, and tool call to a web interface. At any moment, the user can pause the run, edit the plan or code, run custom commands, and resume -- switching smoothly between AI-led, human-assisted and human-led, AI-assisted modes. In fully autonomous mode, ResearStudio achieves state-of-the-art results on the GAIA benchmark, surpassing systems like OpenAI's DeepResearch and Manus. These results show that strong automated performance and fine-grained human control can coexist. The full code, protocol, and evaluation scripts are available at this https URL. We will continue to update the repository to encourage further work on safe and controllable research agents. Our live demo is publicly accessible at this http URL. We support the development of DeepScientist, which can be accessed at this https URL.

**arXiv ID:** 2510.12194
</details>

<details>
<summary><strong>$\mathbf{T^3}$: Reducing Belief Deviation in Reinforcement Learning for Active Reasoning</strong> - Deyu Zou, Yongqiang Chen, Jianxiang Wang, Haochen Yang, Mufei Li, James Cheng, Pan Li, Yu Gong - [[pdf]](https://arxiv.org/pdf/2510.12264)</summary>

**Abstract:** Active reasoning requires large language models (LLMs) to interact with external sources and strategically gather information to solve problems. Central to this process is belief tracking: maintaining a coherent understanding of the problem state and the missing information toward the solution. However, due to limited reasoning capabilities, LLM-based agents often suffer from belief deviation: they struggle to correctly model beliefs, lose track of problem states, and fall into uninformative or repetitive actions. Once this happens, errors compound and reinforcement learning (RL) training fails to properly credit the crucial exploratory steps. To address this issue, we propose to track the deviation of model beliefs and develop $\mathbf{T^3}$, a simple yet effective method that detects excessive belief deviation and truncates trajectories during training to remove uninformative tails. By preserving credit for informative prefixes, $\mathbf{T^3}$ systematically improves policy optimization. Across 5 challenging tasks, $\mathbf{T^3}$ consistently enhances training stability, token efficiency, and final performance, achieving up to 30% gains while cutting rollout tokens by roughly 25%. These results highlight belief control as a key principle for developing robust and generalizable LLM-based active reasoners.

**arXiv ID:** 2510.12264
</details>

<details>
<summary><strong>Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks</strong> - Yuxiang Zhang, Jiangming Shu, Ye Ma, Xueyuan Lin, Shangxi Wu, Jitao Sang - [[pdf]](https://arxiv.org/pdf/2510.12635)</summary>

**Abstract:** Large Language Models face challenges in long-horizon agentic tasks as their constrained memory is easily overwhelmed by distracting or irrelevant context. Existing working memory methods typically rely on external, heuristic mechanisms that are decoupled from the agent's core policy. In this work, we reframe working memory management as a learnable, intrinsic capability. We propose a novel framework, Memory-as-Action, where an agent actively manages its working memory by executing explicit editing operations as part of a unified policy. This formulation allows an agent, trained via reinforcement learning, to balance memory curation against long-term task objectives under given resource constraints. However, such memory editing actions break the standard assumption of a continuously growing prefix in LLM interactions, leading to what we call trajectory fractures. These non-prefix changes disrupt the causal continuity required by standard policy gradient methods, making those methods inapplicable. To address this, we propose a new algorithm, Dynamic Context Policy Optimization, which enables stable end-to-end reinforcement learning by segmenting trajectories at memory action points and applying trajectory-level advantages to the resulting action segments. Our results demonstrate that jointly optimizing for task reasoning and memory management in an end-to-end fashion not only reduces overall computational consumption but also improves task performance, driven by adaptive context curation strategies tailored to the model's intrinsic capabilities.

**arXiv ID:** 2510.12635
</details>

<details>
<summary><strong>ERA: Transforming VLMs into Embodied Agents via Embodied Prior Learning and Online Reinforcement Learning</strong> - Hanyang Chen, Mark Zhao, Rui Yang, Qinwei Ma, Ke Yang, Jiarui Yao, Kangrui Wang, Hao Bai, Zhenhailong Wang, Rui Pan, Mengchao Zhang, Jose Barreiros, Aykut Onol, ChengXiang Zhai, Heng Ji, Manling Li, Huan Zhang, Tong Zhang - [[pdf]](https://arxiv.org/pdf/2510.12693)</summary>

**Abstract:** Recent advances in embodied AI highlight the potential of vision language models (VLMs) as agents capable of perception, reasoning, and interaction in complex environments. However, top-performing systems rely on large-scale models that are costly to deploy, while smaller VLMs lack the necessary knowledge and skills to succeed. To bridge this gap, we present \textit{Embodied Reasoning Agent (ERA)}, a two-stage framework that integrates prior knowledge learning and online reinforcement learning (RL). The first stage, \textit{Embodied Prior Learning}, distills foundational knowledge from three types of data: (1) Trajectory-Augmented Priors, which enrich existing trajectory data with structured reasoning generated by stronger models; (2) Environment-Anchored Priors, which provide in-environment knowledge and grounding supervision; and (3) External Knowledge Priors, which transfer general knowledge from out-of-environment datasets. In the second stage, we develop an online RL pipeline that builds on these priors to further enhance agent performance. To overcome the inherent challenges in agent RL, including long horizons, sparse rewards, and training instability, we introduce three key designs: self-summarization for context management, dense reward shaping, and turn-level policy optimization. Extensive experiments on both high-level planning (EB-ALFRED) and low-level control (EB-Manipulation) tasks demonstrate that ERA-3B surpasses both prompting-based large models and previous training-based baselines. Specifically, it achieves overall improvements of 8.4\% on EB-ALFRED and 19.4\% on EB-Manipulation over GPT-4o, and exhibits strong generalization to unseen tasks. Overall, ERA offers a practical path toward scalable embodied intelligence, providing methodological insights for future embodied AI systems.

**arXiv ID:** 2510.12693
</details>

<details>
<summary><strong>CAMNet: Leveraging Cooperative Awareness Messages for Vehicle Trajectory Prediction</strong> - Mattia Grasselli, Angelo Porrello, Carlo Augusto Grazia - [[pdf]](https://arxiv.org/pdf/2510.12703)</summary>

**Abstract:** Autonomous driving remains a challenging task, particularly due to safety concerns. Modern vehicles are typically equipped with expensive sensors such as LiDAR, cameras, and radars to reduce the risk of accidents. However, these sensors face inherent limitations: their field of view and line of sight can be obstructed by other vehicles, thereby reducing situational awareness. In this context, vehicle-to-vehicle communication plays a crucial role, as it enables cars to share information and remain aware of each other even when sensors are occluded. One way to achieve this is through the use of Cooperative Awareness Messages (CAMs). In this paper, we investigate the use of CAM data for vehicle trajectory prediction. Specifically, we design and train a neural network, Cooperative Awareness Message-based Graph Neural Network (CAMNet), on a widely used motion forecasting dataset. We then evaluate the model on a second dataset that we created from scratch using Cooperative Awareness Messages, in order to assess whether this type of data can be effectively exploited. Our approach demonstrates promising results, showing that CAMs can indeed support vehicle trajectory prediction. At the same time, we discuss several limitations of the approach, which highlight opportunities for future research.

**arXiv ID:** 2510.12703
</details>

<details>
<summary><strong>GAR: Generative Adversarial Reinforcement Learning for Formal Theorem Proving</strong> - Ruida Wang, Jiarui Yao, Rui Pan, Shizhe Diao, Tong Zhang - [[pdf]](https://arxiv.org/pdf/2510.11769)</summary>

**Abstract:** Solving math problems through verifiable languages such as Lean has significantly impacted both the mathematics and computer science communities. Current state-of-the-art models are often trained with expensive online Reinforcement Learning (RL) or expert iteration. However, these approaches rely on fixed problem sets, which causes inefficient training and limits the model to tackle complex problems. To overcome these limitations, we propose GAR: Generative Adversarial Reinforcement learning, a comprehensive RL training framework that jointly trains the problem composer and solver in an adversarial loop. GAR introduces an implicit curriculum learning mechanism, which aligns task difficulty with the prover's evolving capability. It thereby improves the training efficiency and enables stronger performance of proving advanced theorems. Experiments show that with GAR training, Goedel-Prover-V2-8B and DeepSeek-Prover-V2-7B achieve an average relative improvement in pass@32 of 4.20% on MiniF2F-Test benchmark, while DeepSeek-Prover-V2's pass@32 on ProofNet-Test increases from 22.58% to 25.81%. Beyond formal proving, GAR establishes a general RL paradigm for co-evolution of problem generation and solving under verifiable environments.

**arXiv ID:** 2510.11769
</details>

<details>
<summary><strong>Agent Learning via Early Experience</strong> - Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Xiaohan Fu, Jian Xie, Yuxuan Sun, Boyu Gou, Qi Qi, Zihang Meng, Jianwei Yang, Ning Zhang, Xian Li, Ashish Shah, Dat Huynh, Hengduo Li, Zi Yang, Sara Cao, Lawrence Jang, Shuyan Zhou, Jiacheng Zhu, Huan Sun, Jason Weston, Yu Su, Yifan Wu - [[pdf]](https://arxiv.org/pdf/2510.08558)</summary>

**Abstract:** A long-term goal of language agents is to learn and improve through their own experience, ultimately outperforming humans in complex, real-world tasks. However, training agents from experience data with reinforcement learning remains difficult in many environments, which either lack verifiable rewards (e.g., websites) or require inefficient long-horizon rollouts (e.g., multi-turn tool use). As a result, most current agents rely on supervised fine-tuning on expert data, which is challenging to scale and generalizes poorly. This limitation stems from the nature of expert demonstrations: they capture only a narrow range of scenarios and expose the agent to limited environment diversity. We address this limitation with a middle-ground paradigm we call early experience: interaction data generated by the agent's own actions, where the resulting future states serve as supervision without reward signals. Within this paradigm we study two strategies of using such data: (1) Implicit world modeling, which uses collected states to ground the policy in environment dynamics; and (2) Self-reflection, where the agent learns from its suboptimal actions to improve reasoning and decision-making. We evaluate across eight diverse environments and multiple model families. Our approaches consistently improve effectiveness and out-of-domain generalization, highlighting the value of early experience. Moreover, in environments with verifiable rewards, our results provide promising signals that early experience offers a strong foundation for subsequent reinforcement learning, positioning it as a practical bridge between imitation learning and fully experience-driven agents.

**arXiv ID:** 2510.08558
</details>

<details>
<summary><strong>VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents</strong> - Sam Yu-Te Lee, Chenyang Ji, Shicheng Wen, Lifu Huang, Dongyu Liu, Kwan-Liu Ma - [[pdf]](https://arxiv.org/pdf/2506.21582)</summary>

**Abstract:** Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems.

**arXiv ID:** 2506.21582
</details>

<details>
<summary><strong>A Cooperative Approach for Knowledge-based Business Process Design in a Public Authority</strong> - Mohammad Azarijafari, Luisa Mich, Michele Missikoff, Oleg Missikoff - [[pdf]](https://arxiv.org/pdf/2507.19842)</summary>

**Abstract:** Enterprises are currently undergoing profound transformations due to the unpostponable digital transformation. Then, to remain competitive, enterprises must adapt digital solutions, transforming their organisational structures and operations. This organisational shift is also important for small and medium-sized enterprises. A key innovation frontier is the adoption of process-oriented production models. This paper presents a knowledge-based method to support business experts in designing business processes. The method requires no prior expertise in Knowledge Engineering and guides designers through a structured sequence of steps to produce a diagrammatic workflow of the target process. The construction of the knowledge base starts from simple, text-based, knowledge artefacts and then progresses towards more structured, formal representations. The approach has been conceived to allow a shared approach for all stakeholders and actors who participate in the BP design.

**arXiv ID:** 2507.19842
</details>

<details>
<summary><strong>SAGE: A Top-Down Bottom-Up Knowledge-Grounded User Simulator for Multi-turn AGent Evaluation</strong> - Ryan Shea, Yunan Lu, Liang Qiu, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2510.11997)</summary>

**Abstract:** Evaluating multi-turn interactive agents is challenging due to the need for human assessment. Evaluation with simulated users has been introduced as an alternative, however existing approaches typically model generic users and overlook the domain-specific principles required to capture realistic behavior. We propose SAGE, a novel user Simulation framework for multi-turn AGent Evaluation that integrates knowledge from business contexts. SAGE incorporates top-down knowledge rooted in business logic, such as ideal customer profiles, grounding user behavior in realistic customer personas. We further integrate bottom-up knowledge taken from business agent infrastructure (e.g., product catalogs, FAQs, and knowledge bases), allowing the simulator to generate interactions that reflect users' information needs and expectations in a company's target market. Through empirical evaluation, we find that this approach produces interactions that are more realistic and diverse, while also identifying up to 33% more agent errors, highlighting its effectiveness as an evaluation tool to support bug-finding and iterative agent improvement.

**arXiv ID:** 2510.11997
</details>

<details>
<summary><strong>DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL</strong> - Rui Lu, Zhenyu Hou, Zihan Wang, Hanchen Zhang, Xiao Liu, Yujiang Li, Shi Feng, Jie Tang, Yuxiao Dong - [[pdf]](https://arxiv.org/pdf/2509.10446)</summary>

**Abstract:** Augmenting large language models (LLMs) with browsing tools substantially improves their potential as deep search agents to solve complex, real-world tasks. Yet, open LLMs still perform poorly in such settings due to limited long-horizon reasoning capacity with browsing tools and the lack of sufficiently difficult supervised data. To address these challenges, we present DeepDive to advance deep search agents. First, we propose a strategy to automatically synthesize complex, difficult, and hard-to-find questions from open knowledge graphs. Second, we apply end-to-end multi-turn reinforcement learning (RL) to enhance LLMs' long-horizon reasoning with deep search. To encourage diversity and reduce redundancy, we design a redundancy penalty that discourages repeated similar queries. Experiments show that DeepDive-32B achieves a new open-source competitive result on BrowseComp, outperforming WebSailor, DeepSeek-R1-Browse, and Search-o1. We demonstrate that multi-turn RL training improves deep search ability and significantly contributes to the performance improvements across multiple benchmarks. We observe that DeepDive enables test-time scaling of tool calls and parallel sampling. All datasets, models, and code are publicly available at this https URL.

**arXiv ID:** 2509.10446
</details>

<details>
<summary><strong>Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning</strong> - Xue Zhang, Yunlong Liang, Fandong Meng, Songming Zhang, Kaiyu Huang, Yufeng Chen, Jinan Xu, Jie Zhou - [[pdf]](https://arxiv.org/pdf/2510.07300)</summary>

**Abstract:** Large Reasoning Models (LRMs) have achieved remarkable performance on complex reasoning tasks by adopting the "think-then-answer" paradigm, which enhances both accuracy and interpretability. However, current LRMs exhibit two critical limitations when processing non-English languages: (1) They often struggle to maintain input-output language consistency; (2) They generally perform poorly with wrong reasoning paths and lower answer accuracy compared to English. These limitations significantly degrade the user experience for non-English speakers and hinder the global deployment of LRMs. To address these limitations, we propose M-Thinker, which is trained by the GRPO algorithm that involves a Language Consistency (LC) reward and a novel Cross-lingual Thinking Alignment (CTA) reward. Specifically, the LC reward defines a strict constraint on the language consistency between the input, thought, and answer. Besides, the CTA reward compares the model's non-English reasoning paths with its English reasoning path to transfer its own reasoning capability from English to non-English languages. Through an iterative RL procedure, our M-Thinker-1.5B/7B models not only achieve nearly 100% language consistency and superior performance on two multilingual benchmarks (MMATH and PolyMath), but also exhibit excellent generalization on out-of-domain languages.

**arXiv ID:** 2510.07300
</details>

<details>
<summary><strong>DSPO: Stable and Efficient Policy Optimization for Agentic Search and Reasoning</strong> - Chenyang Gu, Yewen Pu, Bruce Yang, Xiaofan Li, Huan Gao - [[pdf]](https://arxiv.org/pdf/2510.09255)</summary>

**Abstract:** Enhancing LLMs with the ability to actively search external knowledge is crucial for complex and real-world tasks. Current approaches either rely on prompting to elicit the model's innate agent capabilities, or suffer from performance ceilings and collapse when applying RL to complex interactive tasks, leaving their true agentic potential untapped. To address this, we introduce \textbf{D}ynamic-filter \textbf{S}equence-level \textbf{P}olicy \textbf{O}ptimization (DSPO), an improved RL algorithm designed for robust agent training through sequence-level optimization and dynamic sample filtering. We train our model purely through RL to interleave multi-turn search and reasoning, obviating the need for supervised demonstration data. Across multiple QA benchmarks, our 7B model improves over a comparable previous work by \textbf{34.1\%}, and even outperforms the 14B model from previous work in complex multihop QA such as HotpotQA by nearly \textbf{9\% relative}, maintaining exceptional training stability.

**arXiv ID:** 2510.09255
</details>

<details>
<summary><strong>Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling</strong> - Xiaohang Tang, Zhuowen Cheng, Satyabrat Kumar - [[pdf]](https://arxiv.org/pdf/2510.11877)</summary>

**Abstract:** The Transformer, a highly expressive architecture for sequence modeling, has recently been adapted to solve sequential decision-making, most notably through the Decision Transformer (DT), which learns policies by conditioning on desired returns. Yet, the adversarial robustness of reinforcement learning methods based on sequence modeling remains largely unexplored. Here we introduce the Conservative Adversarially Robust Decision Transformer (CART), to our knowledge the first framework designed to enhance the robustness of DT in adversarial stochastic games. We formulate the interaction between the protagonist and the adversary at each stage as a stage game, where the payoff is defined as the expected maximum value over subsequent states, thereby explicitly incorporating stochastic state transitions. By conditioning Transformer policies on the NashQ value derived from these stage games, CART generates policy that are simultaneously less exploitable (adversarially robust) and conservative to transition uncertainty. Empirically, CART achieves more accurate minimax value estimation and consistently attains superior worst-case returns across a range of adversarial stochastic games.

**arXiv ID:** 2510.11877
</details>

<details>
<summary><strong>Efficient Restarts in Non-Stationary Model-Free Reinforcement Learning</strong> - Hiroshi Nonaka, Simon Ambrozak, Sofia R. Miskala-Dinc, Amedeo Ercole, Aviva Prins - [[pdf]](https://arxiv.org/pdf/2510.11933)</summary>

**Abstract:** In this work, we propose three efficient restart paradigms for model-free non-stationary reinforcement learning (RL). We identify two core issues with the restart design of Mao et al. (2022)'s RestartQ-UCB algorithm: (1) complete forgetting, where all the information learned about an environment is lost after a restart, and (2) scheduled restarts, in which restarts occur only at predefined timings, regardless of the incompatibility of the policy with the current environment dynamics. We introduce three approaches, which we call partial, adaptive, and selective restarts to modify the algorithms RestartQ-UCB and RANDOMIZEDQ (Wang et al., 2025). We find near-optimal empirical performance in multiple different environments, decreasing dynamic regret by up to $91$% relative to RestartQ-UCB.

**arXiv ID:** 2510.11933
</details>

<details>
<summary><strong>Rethinking the Role of Dynamic Sparse Training for Scalable Deep Reinforcement Learning</strong> - Guozheng Ma, Lu Li, Zilin Wang, Haoyu Wang, Shengchao Hu, Leszek Rutkowski, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2510.12096)</summary>

**Abstract:** Scaling neural networks has driven breakthrough advances in machine learning, yet this paradigm fails in deep reinforcement learning (DRL), where larger models often degrade performance due to unique optimization pathologies such as plasticity loss. While recent works show that dynamically adapting network topology during training can mitigate these issues, existing studies have three critical limitations: (1) applying uniform dynamic training strategies across all modules despite encoder, critic, and actor following distinct learning paradigms, (2) focusing evaluation on basic architectures without clarifying the relative importance and interaction between dynamic training and architectural improvements, and (3) lacking systematic comparison between different dynamic approaches including sparse-to-sparse, dense-to-sparse, and sparse-to-dense. Through comprehensive investigation across modules and architectures, we reveal that dynamic sparse training strategies provide module-specific benefits that complement the primary scalability foundation established by architectural improvements. We finally distill these insights into Module-Specific Training (MST), a practical framework that further exploits the benefits of architectural improvements and demonstrates substantial scalability gains across diverse RL algorithms without algorithmic modifications.

**arXiv ID:** 2510.12096
</details>

<details>
<summary><strong>Expert or not? assessing data quality in offline reinforcement learning</strong> - Arip Asadulaev, Fakhri Karray, Martin Takac - [[pdf]](https://arxiv.org/pdf/2510.12638)</summary>

**Abstract:** Offline reinforcement learning (RL) learns exclusively from static datasets, without further interaction with the environment. In practice, such datasets vary widely in quality, often mixing expert, suboptimal, and even random trajectories. The choice of algorithm therefore depends on dataset fidelity. Behavior cloning can suffice on high-quality data, whereas mixed- or low-quality data typically benefits from offline RL methods that stitch useful behavior across trajectories. Yet in the wild it is difficult to assess dataset quality a priori because the data's provenance and skill composition are unknown. We address the problem of estimating offline dataset quality without training an agent. We study a spectrum of proxies from simple cumulative rewards to learned value based estimators, and introduce the Bellman Wasserstein distance (BWD), a value aware optimal transport score that measures how dissimilar a dataset's behavioral policy is from a random reference policy. BWD is computed from a behavioral critic and a state conditional OT formulation, requiring no environment interaction or full policy optimization. Across D4RL MuJoCo tasks, BWD strongly correlates with an oracle performance score that aggregates multiple offline RL algorithms, enabling efficient prediction of how well standard agents will perform on a given dataset. Beyond prediction, integrating BWD as a regularizer during policy optimization explicitly pushes the learned policy away from random behavior and improves returns. These results indicate that value aware, distributional signals such as BWD are practical tools for triaging offline RL datasets and policy optimization.

**arXiv ID:** 2510.12638
</details>

<details>
<summary><strong>Pretraining in Actor-Critic Reinforcement Learning for Robot Motion Control</strong> - Jiale Fan, Andrei Cramariuc, Tifanny Portela, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2510.12363)</summary>

**Abstract:** The pretraining-finetuning paradigm has facilitated numerous transformative advancements in artificial intelligence research in recent years. However, in the domain of reinforcement learning (RL) for robot motion control, individual skills are often learned from scratch despite the high likelihood that some generalizable knowledge is shared across all task-specific policies belonging to a single robot embodiment. This work aims to define a paradigm for pretraining neural network models that encapsulate such knowledge and can subsequently serve as a basis for warm-starting the RL process in classic actor-critic algorithms, such as Proximal Policy Optimization (PPO). We begin with a task-agnostic exploration-based data collection algorithm to gather diverse, dynamic transition data, which is then used to train a Proprioceptive Inverse Dynamics Model (PIDM) through supervised learning. The pretrained weights are loaded into both the actor and critic networks to warm-start the policy optimization of actual tasks. We systematically validated our proposed method on seven distinct robot motion control tasks, showing significant benefits to this initialization strategy. Our proposed approach on average improves sample efficiency by 40.1% and task performance by 7.5%, compared to random initialization. We further present key ablation studies and empirical analyses that shed light on the mechanisms behind the effectiveness of our method.

**arXiv ID:** 2510.12363
</details>

<details>
<summary><strong>CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving</strong> - Xiaoji Zheng, Ziyuan Yang, Yanhao Chen, Yuhang Peng, Yuanrong Tang, Gengyuan Liu, Bokui Chen, Jiangtao Gong - [[pdf]](https://arxiv.org/pdf/2510.12560)</summary>

**Abstract:** End-to-end autonomous driving models trained solely with imitation learning (IL) often suffer from poor generalization. In contrast, reinforcement learning (RL) promotes exploration through reward maximization but faces challenges such as sample inefficiency and unstable convergence. A natural solution is to combine IL and RL. Moving beyond the conventional two-stage paradigm (IL pretraining followed by RL fine-tuning), we propose CoIRL-AD, a competitive dual-policy framework that enables IL and RL agents to interact during training. CoIRL-AD introduces a competition-based mechanism that facilitates knowledge exchange while preventing gradient conflicts. Experiments on the nuScenes dataset show an 18% reduction in collision rate compared to baselines, along with stronger generalization and improved performance on long-tail scenarios. Code is available at: this https URL.

**arXiv ID:** 2510.12560
</details>

<details>
<summary><strong>A Task-Efficient Reinforcement Learning Task-Motion Planner for Safe Human-Robot Cooperation</strong> - Gaoyuan Liu, Joris de Winter, Kelly Merckaert, Denis Steckelmacher, Ann Nowe, Bram Vanderborght - [[pdf]](https://arxiv.org/pdf/2510.12477)</summary>

**Abstract:** In a Human-Robot Cooperation (HRC) environment, safety and efficiency are the two core properties to evaluate robot performance. However, safety mechanisms usually hinder task efficiency since human intervention will cause backup motions and goal failures of the robot. Frequent motion replanning will increase the computational load and the chance of failure. In this paper, we present a hybrid Reinforcement Learning (RL) planning framework which is comprised of an interactive motion planner and a RL task planner. The RL task planner attempts to choose statistically safe and efficient task sequences based on the feedback from the motion planner, while the motion planner keeps the task execution process collision-free by detecting human arm motions and deploying new paths when the previous path is not valid anymore. Intuitively, the RL agent will learn to avoid dangerous tasks, while the motion planner ensures that the chosen tasks are safe. The proposed framework is validated on the cobot in both simulation and the real world, we compare the planner with hard-coded task motion planning methods. The results show that our planning framework can 1) react to uncertain human motions at both joint and task levels; 2) reduce the times of repeating failed goal commands; 3) reduce the total number of replanning requests.

**arXiv ID:** 2510.12477
</details>

<details>
<summary><strong>Autonomous Legged Mobile Manipulation for Lunar Surface Operations via Constrained Reinforcement Learning</strong> - Alvaro Belmonte-Baeza, Miguel Cazorla, Gabriel J. García, Carlos J. Pérez-Del-Pulgar, Jorge Pomares - [[pdf]](https://arxiv.org/pdf/2510.12684)</summary>

**Abstract:** Robotics plays a pivotal role in planetary science and exploration, where autonomous and reliable systems are crucial due to the risks and challenges inherent to space environments. The establishment of permanent lunar bases demands robotic platforms capable of navigating and manipulating in the harsh lunar terrain. While wheeled rovers have been the mainstay for planetary exploration, their limitations in unstructured and steep terrains motivate the adoption of legged robots, which offer superior mobility and adaptability. This paper introduces a constrained reinforcement learning framework designed for autonomous quadrupedal mobile manipulators operating in lunar environments. The proposed framework integrates whole-body locomotion and manipulation capabilities while explicitly addressing critical safety constraints, including collision avoidance, dynamic stability, and power efficiency, in order to ensure robust performance under lunar-specific conditions, such as reduced gravity and irregular terrain. Experimental results demonstrate the framework's effectiveness in achieving precise 6D task-space end-effector pose tracking, achieving an average positional accuracy of 4 cm and orientation accuracy of 8.1 degrees. The system consistently respects both soft and hard constraints, exhibiting adaptive behaviors optimized for lunar gravity conditions. This work effectively bridges adaptive learning with essential mission-critical safety requirements, paving the way for advanced autonomous robotic explorers for future lunar missions.

**arXiv ID:** 2510.12684
</details>

<details>
<summary><strong>Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control</strong> - Se Hwan Jeon, Ho Jae Lee, Seungwoo Hong, Sangbae Kim - [[pdf]](https://arxiv.org/pdf/2510.12717)</summary>

**Abstract:** Model Predictive Control (MPC) provides interpretable, tunable locomotion controllers grounded in physical models, but its robustness depends on frequent replanning and is limited by model mismatch and real-time computational constraints. Reinforcement Learning (RL), by contrast, can produce highly robust behaviors through stochastic training but often lacks interpretability, suffers from out-of-distribution failures, and requires intensive reward engineering. This work presents a GPU-parallelized residual architecture that tightly integrates MPC and RL by blending their outputs at the torque-control level. We develop a kinodynamic whole-body MPC formulation evaluated across thousands of agents in parallel at 100 Hz for RL training. The residual policy learns to make targeted corrections to the MPC outputs, combining the interpretability and constraint handling of model-based control with the adaptability of RL. The model-based control prior acts as a strong bias, initializing and guiding the policy towards desirable behavior with a simple set of rewards. Compared to standalone MPC or end-to-end RL, our approach achieves higher sample efficiency, converges to greater asymptotic rewards, expands the range of trackable velocity commands, and enables zero-shot adaptation to unseen gaits and uneven terrain.

**arXiv ID:** 2510.12717
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
