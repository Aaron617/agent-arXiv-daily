# Agent arXiv Daily

**Last Updated:** 2025-09-15 02:06:57

**Total Papers:** 47

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
<summary><strong>Vibe Check: Understanding the Effects of LLM-Based Conversational Agents' Personality and Alignment on User Perceptions in Goal-Oriented Tasks</strong> - Hasibur Rahman, Smit Desai - [[pdf]](https://arxiv.org/pdf/2509.09870)</summary>

**Abstract:** Large language models (LLMs) enable conversational agents (CAs) to express distinctive personalities, raising new questions about how such designs shape user perceptions. This study investigates how personality expression levels and user-agent personality alignment influence perceptions in goal-oriented tasks. In a between-subjects experiment (N=150), participants completed travel planning with CAs exhibiting low, medium, or high expression across the Big Five traits, controlled via our novel Trait Modulation Keys framework. Results revealed an inverted-U relationship: medium expression produced the most positive evaluations across Intelligence, Enjoyment, Anthropomorphism, Intention to Adopt, Trust, and Likeability, significantly outperforming both extremes. Personality alignment further enhanced outcomes, with Extraversion and Emotional Stability emerging as the most influential traits. Cluster analysis identified three distinct compatibility profiles, with "Well-Aligned" users reporting substantially positive perceptions. These findings demonstrate that personality expression and strategic trait alignment constitute optimal design targets for CA personality, offering design implications as LLM-based CAs become increasingly prevalent.

**arXiv ID:** 2509.09870
</details>

<details>
<summary><strong>Scaling Arabic Medical Chatbots Using Synthetic Data: Enhancing Generative AI with Synthetic Patient Records</strong> - Abdulrahman Allam, Seif Ahmed, Ali Hamdi, Khaled Shaban - [[pdf]](https://arxiv.org/pdf/2509.10108)</summary>

**Abstract:** The development of medical chatbots in Arabic is significantly constrained by the scarcity of large-scale, high-quality annotated datasets. While prior efforts compiled a dataset of 20,000 Arabic patient-doctor interactions from social media to fine-tune large language models (LLMs), model scalability and generalization remained limited. In this study, we propose a scalable synthetic data augmentation strategy to expand the training corpus to 100,000 records. Using advanced generative AI systems ChatGPT-4o and Gemini 2.5 Pro we generated 80,000 contextually relevant and medically coherent synthetic question-answer pairs grounded in the structure of the original dataset. These synthetic samples were semantically filtered, manually validated, and integrated into the training pipeline. We fine-tuned five LLMs, including Mistral-7B and AraGPT2, and evaluated their performance using BERTScore metrics and expert-driven qualitative assessments. To further analyze the effectiveness of synthetic sources, we conducted an ablation study comparing ChatGPT-4o and Gemini-generated data independently. The results showed that ChatGPT-4o data consistently led to higher F1-scores and fewer hallucinations across all models. Overall, our findings demonstrate the viability of synthetic augmentation as a practical solution for enhancing domain-specific language models in-low resource medical NLP, paving the way for more inclusive, scalable, and accurate Arabic healthcare chatbot systems.

**arXiv ID:** 2509.10108
</details>

<details>
<summary><strong>Testing chatbots on the creation of encoders for audio conditioned image generation</strong> - Jorge E. León, Miguel Carrasco - [[pdf]](https://arxiv.org/pdf/2509.09717)</summary>

**Abstract:** On one hand, recent advances in chatbots has led to a rising popularity in using these models for coding tasks. On the other hand, modern generative image models primarily rely on text encoders to translate semantic concepts into visual representations, even when there is clear evidence that audio can be employed as input as well. Given the previous, in this work, we explore whether state-of-the-art conversational agents can design effective audio encoders to replace the CLIP text encoder from Stable Diffusion 1.5, enabling image synthesis directly from sound. We prompted five publicly available chatbots to propose neural architectures to work as these audio encoders, with a set of well-explained shared conditions. Each valid suggested encoder was trained on over two million context related audio-image-text observations, and evaluated on held-out validation and test sets using various metrics, together with a qualitative analysis of their generated images. Although almost all chatbots generated valid model designs, none achieved satisfactory results, indicating that their audio embeddings failed to align reliably with those of the original text encoder. Among the proposals, the Gemini audio encoder showed the best quantitative metrics, while the Grok audio encoder produced more coherent images (particularly, when paired with the text encoder). Our findings reveal a shared architectural bias across chatbots and underscore the remaining coding gap that needs to be bridged in future versions of these models. We also created a public demo so everyone could study and try out these audio encoders. Finally, we propose research questions that should be tackled in the future, and encourage other researchers to perform more focused and highly specialized tasks like this one, so the respective chatbots cannot make use of well-known solutions and their creativity/reasoning is fully tested.

**arXiv ID:** 2509.09717
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>Compartmentalised Agentic Reasoning for Clinical NLI</strong> - Maël Jullien, Lei Xu, Marco Valentino, André Freitas - [[pdf]](https://arxiv.org/pdf/2509.10222)</summary>

**Abstract:** A common assumption holds that scaling data and parameters yields increasingly structured, generalisable internal representations. We interrogate this assumption in clinical natural language inference (NLI) by adopting a benchmark decomposed into four reasoning families, Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction, and introducing CARENLI, a Compartmentalised Agentic Reasoning for Clinical NLI that separates knowledge access from principled inference. CARENLI routes each premise, statement pair to a family specific solver and enforces auditable procedures via a planner, verifier, and refiner.
Across four LLMs, CARENLI improves fidelity by up to 42 points, reaching 98.0% in Causal Attribution and 81.2% in Risk State Abstraction. Verifiers flag violations with near-ceiling reliability, while refiners correct a substantial share of epistemic errors. Remaining failures cluster in routing, identifying family classification as the main bottleneck. These results show that LLMs often retain relevant facts but default to heuristics when inference is underspecified, a dissociation CARENLI makes explicit while offering a framework for safer, auditable reasoning.

**arXiv ID:** 2509.10222
</details>

<details>
<summary><strong>TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation</strong> - Keunwoo Choi, Seungheon Doh, Juhan Nam - [[pdf]](https://arxiv.org/pdf/2509.09685)</summary>

**Abstract:** We present TalkPlayData 2, a synthetic dataset for multimodal conversational music recommendation generated by an agentic data pipeline. In TalkPlayData 2 pipeline, multiple large language model (LLM) agents are created under various roles with specialized prompts and access to different parts of information, and the chat data is acquired by logging the conversation between the Listener LLM and the Recsys LLM. To cover various conversation scenarios, for each conversation, the Listener LLM is conditioned on a finetuned conversation goal. Finally, all the LLMs are multimodal with audio and images, allowing a simulation of multimodal recommendation and conversation. In the LLM-as-a-judge and subjective evaluation experiments, TalkPlayData 2 achieved the proposed goal in various aspects related to training a generative recommendation model for music. TalkPlayData 2 and its generation code are open-sourced at this https URL.

**arXiv ID:** 2509.09685
</details>

<details>
<summary><strong>MITS: A Large-Scale Multimodal Benchmark Dataset for Intelligent Traffic Surveillance</strong> - Kaikai Zhao, Zhaoxiang Liu, Peng Wang, Xin Wang, Zhicheng Ma, Yajun Xu, Wenjing Zhang, Yibing Nan, Kai Wang, Shiguo Lian - [[pdf]](https://arxiv.org/pdf/2509.09730)</summary>

**Abstract:** General-domain large multimodal models (LMMs) have achieved significant advances in various image-text tasks. However, their performance in the Intelligent Traffic Surveillance (ITS) domain remains limited due to the absence of dedicated multimodal datasets. To address this gap, we introduce MITS (Multimodal Intelligent Traffic Surveillance), the first large-scale multimodal benchmark dataset specifically designed for ITS. MITS includes 170,400 independently collected real-world ITS images sourced from traffic surveillance cameras, annotated with eight main categories and 24 subcategories of ITS-specific objects and events under diverse environmental conditions. Additionally, through a systematic data generation pipeline, we generate high-quality image captions and 5 million instruction-following visual question-answer pairs, addressing five critical ITS tasks: object and event recognition, object counting, object localization, background analysis, and event reasoning. To demonstrate MITS's effectiveness, we fine-tune mainstream LMMs on this dataset, enabling the development of ITS-specific applications. Experimental results show that MITS significantly improves LMM performance in ITS applications, increasing LLaVA-1.5's performance from 0.494 to 0.905 (+83.2%), LLaVA-1.6's from 0.678 to 0.921 (+35.8%), Qwen2-VL's from 0.584 to 0.926 (+58.6%), and Qwen2.5-VL's from 0.732 to 0.930 (+27.0%). We release the dataset, code, and models as open-source, providing high-value resources to advance both ITS and LMM research.

**arXiv ID:** 2509.09730
</details>

<details>
<summary><strong>MCP-AgentBench: Evaluating Real-World Language Agent Performance with MCP-Mediated Tools</strong> - Zikang Guo, Benfeng Xu, Chiwei Zhu, Wentao Hong, Xiaorui Wang, Zhendong Mao - [[pdf]](https://arxiv.org/pdf/2509.09734)</summary>

**Abstract:** The Model Context Protocol (MCP) is rapidly emerging as a pivotal open standard, designed to enhance agent-tool integration and interoperability, and is positioned to unlock a new era of powerful, interconnected, and genuinely utilitarian agentic AI. However, despite MCP's growing adoption, existing benchmarks often fail to capture real-world agent performance within this new paradigm, leading to a distorted perception of their true operational value and an inability to reliably differentiate proficiencies. To bridge this critical evaluation gap, we introduce MCP-AgentBench -- a comprehensive benchmark specifically engineered to rigorously assess language agent capabilities in MCP-mediated tool interactions. Core contributions of MCP-AgentBench include: the establishment of a robust MCP testbed comprising 33 operational servers with 188 distinct tools; the development of a benchmark featuring 600 systematically designed queries distributed across 6 distinct categories of varying interaction complexity; and the introduction of MCP-Eval, a novel outcome-oriented evaluation methodology prioritizing real-world task success. Through extensive empirical evaluation of leading language agents, we provide foundational insights. MCP-AgentBench aims to equip the research community with a standardized and reliable framework to build, validate, and advance agents capable of fully leveraging MCP's transformative benefits, thereby accelerating progress toward truly capable and interoperable AI systems.

**arXiv ID:** 2509.09734
</details>

<details>
<summary><strong>Securing LLM-Generated Embedded Firmware through AI Agent-Driven Validation and Patching</strong> - Seyed Moein Abtahi, Akramul Azim - [[pdf]](https://arxiv.org/pdf/2509.09970)</summary>

**Abstract:** Large Language Models (LLMs) show promise in generating firmware for embedded systems, but often introduce security flaws and fail to meet real-time performance constraints. This paper proposes a three-phase methodology that combines LLM-based firmware generation with automated security validation and iterative refinement in a virtualized environment. Using structured prompts, models like GPT-4 generate firmware for networking and control tasks, deployed on FreeRTOS via QEMU. These implementations are tested using fuzzing, static analysis, and runtime monitoring to detect vulnerabilities such as buffer overflows (CWE-120), race conditions (CWE-362), and denial-of-service threats (CWE-400). Specialized AI agents for Threat Detection, Performance Optimization, and Compliance Verification collaborate to improve detection and remediation. Identified issues are categorized using CWE, then used to prompt targeted LLM-generated patches in an iterative loop. Experiments show a 92.4\% Vulnerability Remediation Rate (37.3\% improvement), 95.8\% Threat Model Compliance, and 0.87 Security Coverage Index. Real-time metrics include 8.6ms worst-case execution time and 195{\mu}s jitter. This process enhances firmware security and performance while contributing an open-source dataset for future research.

**arXiv ID:** 2509.09970
</details>

<details>
<summary><strong>Building Self-Evolving Agents via Experience-Driven Lifelong Learning: A Framework and Benchmark</strong> - Yuxuan Cai, Yipeng Hao, Jie Zhou, Hang Yan, Zhikai Lei, Rui Zhen, Zhenhua Han, Yutao Yang, Junsong Li, Qianjun Pan, Tianyu Huai, Qin Chen, Xin Li, Kai Chen, Bo Zhang, Xipeng Qiu, Liang He - [[pdf]](https://arxiv.org/pdf/2508.19005)</summary>

**Abstract:** As AI advances toward general intelligence, the focus is shifting from systems optimized for static tasks to creating open-ended agents that learn continuously. In this paper, we introduce Experience-driven Lifelong Learning (ELL), a framework for building self-evolving agents capable of continuous growth through real-world interaction. The framework is built on four core principles: (1) Experience Exploration: Agents learn through continuous, self-motivated interaction with dynamic environments, navigating interdependent tasks and generating rich experiential trajectories. (2) Long-term Memory: Agents preserve and structure historical knowledge, including personal experiences, domain expertise, and commonsense reasoning, into a persistent memory system. (3) Skill Learning: Agents autonomously improve by abstracting recurring patterns from experience into reusable skills, which are actively refined and validated for application in new tasks. (4) Knowledge Internalization: Agents internalize explicit and discrete experiences into implicit and intuitive capabilities as "second nature".
We also introduce StuLife, a benchmark dataset for ELL that simulates a student's holistic college journey, from enrollment to academic and personal development, across three core phases and ten detailed sub-scenarios. StuLife is designed around three key paradigm

**arXiv ID:** 2508.19005
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>HypoGeneAgent: A Hypothesis Language Agent for Gene-Set Cluster Resolution Selection Using Perturb-seq Datasets</strong> - Ying Yuan, Xing-Yue Monica Ge, Aaron Archer Waterman, Tommaso Biancalani, David Richmond, Yogesh Pandit, Avtar Singh, Russell Littman, Jin Liu, Jan-Christian Huetter, Vladimir Ermakov - [[pdf]](https://arxiv.org/pdf/2509.09740)</summary>

**Abstract:** Large-scale single-cell and Perturb-seq investigations routinely involve clustering cells and subsequently annotating each cluster with Gene-Ontology (GO) terms to elucidate the underlying biological programs. However, both stages, resolution selection and functional annotation, are inherently subjective, relying on heuristics and expert curation. We present HYPOGENEAGENT, a large language model (LLM)-driven framework, transforming cluster annotation into a quantitatively optimizable task. Initially, an LLM functioning as a gene-set analyst analyzes the content of each gene program or perturbation module and generates a ranked list of GO-based hypotheses, accompanied by calibrated confidence scores. Subsequently, we embed every predicted description with a sentence-embedding model, compute pair-wise cosine similarities, and let the agent referee panel score (i) the internal consistency of the predictions, high average similarity within the same cluster, termed intra-cluster agreement (ii) their external distinctiveness, low similarity between clusters, termed inter-cluster separation. These two quantities are combined to produce an agent-derived resolution score, which is maximized when clusters exhibit simultaneous coherence and mutual exclusivity. When applied to a public K562 CRISPRi Perturb-seq dataset as a preliminary test, our Resolution Score selects clustering granularities that exhibit alignment with known pathway compared to classical metrics such silhouette score, modularity score for gene functional enrichment summary. These findings establish LLM agents as objective adjudicators of cluster resolution and functional annotation, thereby paving the way for fully automated, context-aware interpretation pipelines in single-cell multi-omics studies.

**arXiv ID:** 2509.09740
</details>

<details>
<summary><strong>SciML Agents: Write the Solver, Not the Solution</strong> - Saarth Gaonkar, Xiang Zheng, Haocheng Xi, Rishabh Tiwari, Kurt Keutzer, Dmitriy Morozov, Michael W. Mahoney, Amir Gholami - [[pdf]](https://arxiv.org/pdf/2509.09936)</summary>

**Abstract:** Recent work in scientific machine learning aims to tackle scientific tasks directly by predicting target values with neural networks (e.g., physics-informed neural networks, neural ODEs, neural operators, etc.), but attaining high accuracy and robustness has been challenging. We explore an alternative view: use LLMs to write code that leverages decades of numerical algorithms. This shifts the burden from learning a solution function to making domain-aware numerical choices. We ask whether LLMs can act as SciML agents that, given a natural-language ODE description, generate runnable code that is scientifically appropriate, selecting suitable solvers (stiff vs. non-stiff), and enforcing stability checks. There is currently no benchmark to measure this kind of capability for scientific computing tasks. As such, we first introduce two new datasets: a diagnostic dataset of adversarial "misleading" problems; and a large-scale benchmark of 1,000 diverse ODE tasks. The diagnostic set contains problems whose superficial appearance suggests stiffness, and that require algebraic simplification to demonstrate non-stiffness; and the large-scale benchmark spans stiff and non-stiff ODE regimes. We evaluate open- and closed-source LLM models along two axes: (i) unguided versus guided prompting with domain-specific knowledge; and (ii) off-the-shelf versus fine-tuned variants. Our evaluation measures both executability and numerical validity against reference solutions. We find that with sufficient context and guided prompts, newer instruction-following models achieve high accuracy on both criteria. In many cases, recent open-source systems perform strongly without fine-tuning, while older or smaller models still benefit from fine-tuning. Overall, our preliminary results indicate that careful prompting and fine-tuning can yield a specialized LLM agent capable of reliably solving simple ODE problems.

**arXiv ID:** 2509.09936
</details>

<details>
<summary><strong>Exploring a Gamified Personality Assessment Method through Interaction with LLM Agents Embodying Different Personalities</strong> - Baiqiao Zhang, Xiangxian Li, Chao Zhou, Xinyu Gai, Juan Liu, Xue Yang, Xiaojuan Ma, Yong-jin Liu, Yulong Bian - [[pdf]](https://arxiv.org/pdf/2507.04005)</summary>

**Abstract:** The low-intrusion and automated personality assessment is receiving increasing attention in psychology and human-computer interaction fields. This study explores an interactive approach for personality assessment, focusing on the multiplicity of personality representation. We propose a framework of Gamified Personality Assessment through Multi-Personality Representations (Multi-PR GPA). The framework leverages Large Language Models to empower virtual agents with different personalities. These agents elicit multifaceted human personality representations through engaging in interactive games. Drawing upon the multi-type textual data generated throughout the interaction, it achieves two modes of personality assessment (i.e., Direct Assessment and Questionnaire-based Assessment) and provides interpretable insights. Grounded in the classic Big Five personality theory, we developed a prototype system and conducted a user study to evaluate the efficacy of Multi-PR GPA. The results affirm the effectiveness of our approach in personality assessment and demonstrate its superior performance when considering the multiplicity of personality representation.

**arXiv ID:** 2507.04005
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (9 papers)</h2></summary>

<details>
<summary><strong>GAMA: A General Anonymizing Multi-Agent System for Privacy Preservation Enhanced by Domain Rules and Disproof Method</strong> - Hailong Yang, Renhuo Zhao, Guanjin Wang, Zhaohong Deng - [[pdf]](https://arxiv.org/pdf/2509.10018)</summary>

**Abstract:** With the rapid advancement of Large Language Model (LLM), LLM-based agents exhibit exceptional abilities in understanding and generating natural language, facilitating human-like collaboration and information transmission in LLM-based Multi-Agent System (MAS). High-performance LLMs are often hosted on remote servers in public spaces. When tasks involve privacy data, MAS cannot securely utilize these LLMs without implementing privacy-preserving mechanisms. To address this challenge, we propose a General Anonymizing Multi-Agent system (GAMA), which divides the agents' workspace into private and public spaces and protects privacy through the anonymizing mechanism. In the private space, agents handle sensitive data, while in the public space, only anonymized data is utilized. GAMA incorporates two key modules to mitigate semantic loss caused by anonymization: Domain-Rule-based Knowledge Enhancement (DRKE) and Disproof-based Logic Enhancement (DLE). We evaluate GAMA on two public question-answering datasets: Trivia Creative Writing and Logic Grid Puzzle. The results demonstrate that GAMA has superior performance compared to the state-of-the-art models. To further assess its privacy-preserving capabilities, we designed two new datasets: Knowledge Privacy Preservation and Logic Privacy Preservation. The final results highlight GAMA's exceptional effectiveness in both task processing and privacy preservation.

**arXiv ID:** 2509.10018
</details>

<details>
<summary><strong>XAgents: A Unified Framework for Multi-Agent Cooperation via IF-THEN Rules and Multipolar Task Processing Graph</strong> - Hailong Yang, Mingxian Gu, Jianqi Wang, Guanjin Wang, Zhaohong Deng - [[pdf]](https://arxiv.org/pdf/2509.10054)</summary>

**Abstract:** The rapid advancement of Large Language Models (LLMs) has significantly enhanced the capabilities of Multi-Agent Systems (MAS) in supporting humans with complex, real-world tasks. However, MAS still face challenges in effective task planning when handling highly complex tasks with uncertainty, often resulting in misleading or incorrect outputs that hinder task execution. To address this, we propose XAgents, a unified multi-agent cooperative framework built on a multipolar task processing graph and IF-THEN rules. XAgents uses the multipolar task processing graph to enable dynamic task planning and handle task uncertainty. During subtask processing, it integrates domain-specific IF-THEN rules to constrain agent behaviors, while global rules enhance inter-agent collaboration. We evaluate the performance of XAgents across three distinct datasets, demonstrating that it consistently surpasses state-of-the-art single-agent and multi-agent approaches in both knowledge-typed and logic-typed question-answering tasks. The codes for XAgents are available at: this https URL.

**arXiv ID:** 2509.10054
</details>

<details>
<summary><strong>Towards Fully Automated Molecular Simulations: Multi-Agent Framework for Simulation Setup and Force Field Extraction</strong> - Marko Petković, Vlado Menkovski, Sofía Calero - [[pdf]](https://arxiv.org/pdf/2509.10210)</summary>

**Abstract:** Automated characterization of porous materials has the potential to accelerate materials discovery, but it remains limited by the complexity of simulation setup and force field selection. We propose a multi-agent framework in which LLM-based agents can autonomously understand a characterization task, plan appropriate simulations, assemble relevant force fields, execute them and interpret their results to guide subsequent steps. As a first step toward this vision, we present a multi-agent system for literature-informed force field extraction and automated RASPA simulation setup. Initial evaluations demonstrate high correctness and reproducibility, highlighting this approach's potential to enable fully autonomous, scalable materials characterization.

**arXiv ID:** 2509.10210
</details>

<details>
<summary><strong>Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems</strong> - Alva West, Yixuan Weng, Minjun Zhu, Zhen Lin, Yue Zhang - [[pdf]](https://arxiv.org/pdf/2509.10401)</summary>

**Abstract:** Failure attribution in multi-agent systems -- pinpointing the exact step where a decisive error occurs -- is a critical yet unsolved challenge. Current methods treat this as a pattern recognition task over long conversation logs, leading to critically low step-level accuracy (below 17\%), which renders them impractical for debugging complex systems. Their core weakness is a fundamental inability to perform robust counterfactual reasoning: to determine if correcting a single action would have actually averted the task failure. To bridge this counterfactual inference gap, we introduce Abduct-Act-Predict (A2P) Scaffolding, a novel agent framework that transforms failure attribution from pattern recognition into a structured causal inference task. A2P explicitly guides a large language model through a formal three-step reasoning process within a single inference pass: (1) Abduction, to infer the hidden root causes behind an agent's actions; (2) Action, to define a minimal corrective intervention; and (3) Prediction, to simulate the subsequent trajectory and verify if the intervention resolves the failure. This structured approach leverages the holistic context of the entire conversation while imposing a rigorous causal logic on the model's analysis. Our extensive experiments on the Who\&When benchmark demonstrate its efficacy. On the Algorithm-Generated dataset, A2P achieves 47.46\% step-level accuracy, a 2.85$\times$ improvement over the 16.67\% of the baseline. On the more complex Hand-Crafted dataset, it achieves 29.31\% step accuracy, a 2.43$\times$ improvement over the baseline's 12.07\%. By reframing the problem through a causal lens, A2P Scaffolding provides a robust, verifiable, and significantly more accurate solution for automated failure attribution.

**arXiv ID:** 2509.10401
</details>

<details>
<summary><strong>A Holistic Architecture for Monitoring and Optimization of Robust Multi-Agent Path Finding Plan Execution</strong> - David Zahrádka, Denisa Mužíková, David Woller, Miroslav Kulich, Jiří Švancara, Roman Barták - [[pdf]](https://arxiv.org/pdf/2509.10284)</summary>

**Abstract:** The goal of Multi-Agent Path Finding (MAPF) is to find a set of paths for a fleet of agents moving in a shared environment such that the agents reach their goals without colliding with each other. In practice, some of the robots executing the plan may get delayed, which can introduce collision risk. Although robust execution methods are used to ensure safety even in the presence of delays, the delays may still have a significant impact on the duration of the execution. At some point, the accumulated delays may become significant enough that instead of continuing with the execution of the original plan, even if it was optimal, there may now exist an alternate plan which will lead to a shorter execution. However, the problem is how to decide when to search for the alternate plan, since it is a costly procedure. In this paper, we propose a holistic architecture for robust execution of MAPF plans, its monitoring and optimization. We exploit a robust execution method called Action Dependency Graph to maintain an estimate of the expected execution duration during the plan's execution. This estimate is used to predict the potential that finding an alternate plan would lead to shorter execution. We empirically evaluate the architecture in experiments in a real-time simulator which we designed to mimic our real-life demonstrator of an autonomous warehouse robotic fleet.

**arXiv ID:** 2509.10284
</details>

<details>
<summary><strong>DECAMP: Towards Scene-Consistent Multi-Agent Motion Prediction with Disentangled Context-Aware Pre-Training</strong> - Jianxin Shi, Zengqi Peng, Xiaolong Chen, Tianyu Wo, Jun Ma - [[pdf]](https://arxiv.org/pdf/2509.10426)</summary>

**Abstract:** Trajectory prediction is a critical component of autonomous driving, essential for ensuring both safety and efficiency on the road. However, traditional approaches often struggle with the scarcity of labeled data and exhibit suboptimal performance in multi-agent prediction scenarios. To address these challenges, we introduce a disentangled context-aware pre-training framework for multi-agent motion prediction, named DECAMP. Unlike existing methods that entangle representation learning with pretext tasks, our framework decouples behavior pattern learning from latent feature reconstruction, prioritizing interpretable dynamics and thereby enhancing scene representation for downstream prediction. Additionally, our framework incorporates context-aware representation learning alongside collaborative spatial-motion pretext tasks, which enables joint optimization of structural and intentional reasoning while capturing the underlying dynamic intentions. Our experiments on the Argoverse 2 benchmark showcase the superior performance of our method, and the results attained underscore its effectiveness in multi-agent motion forecasting. To the best of our knowledge, this is the first context autoencoder framework for multi-agent motion forecasting in autonomous driving. The code and models will be made publicly available.

**arXiv ID:** 2509.10426
</details>

<details>
<summary><strong>A Role-Aware Multi-Agent Framework for Financial Education Question Answering with LLMs</strong> - Andy Zhu, Yingjun Du - [[pdf]](https://arxiv.org/pdf/2509.09727)</summary>

**Abstract:** Question answering (QA) plays a central role in financial education, yet existing large language model (LLM) approaches often fail to capture the nuanced and specialized reasoning required for financial problem-solving. The financial domain demands multistep quantitative reasoning, familiarity with domain-specific terminology, and comprehension of real-world scenarios. We present a multi-agent framework that leverages role-based prompting to enhance performance on domain-specific QA. Our framework comprises a Base Generator, an Evidence Retriever, and an Expert Reviewer agent that work in a single-pass iteration to produce a refined answer. We evaluated our framework on a set of 3,532 expert-designed finance education questions from this http URL, an online learning platform. We leverage retrieval-augmented generation (RAG) for contextual evidence from 6 finance textbooks and prompting strategies for a domain-expert reviewer. Our experiments indicate that critique-based refinement improves answer accuracy by 6.6-8.3% over zero-shot Chain-of-Thought baselines, with the highest performance from Gemini-2.0-Flash. Furthermore, our method enables GPT-4o-mini to achieve performance comparable to the finance-tuned FinGPT-mt_Llama3-8B_LoRA. Our results show a cost-effective approach to enhancing financial QA and offer insights for further research in multi-agent financial LLM systems.

**arXiv ID:** 2509.09727
</details>

<details>
<summary><strong>Federated Multi-Agent Reinforcement Learning for Privacy-Preserving and Energy-Aware Resource Management in 6G Edge Networks</strong> - Francisco Javier Esono Nkulu Andong, Qi Min - [[pdf]](https://arxiv.org/pdf/2509.10163)</summary>

**Abstract:** As sixth-generation (6G) networks move toward ultra-dense, intelligent edge environments, efficient resource management under stringent privacy, mobility, and energy constraints becomes critical. This paper introduces a novel Federated Multi-Agent Reinforcement Learning (Fed-MARL) framework that incorporates cross-layer orchestration of both the MAC layer and application layer for energy-efficient, privacy-preserving, and real-time resource management across heterogeneous edge devices. Each agent uses a Deep Recurrent Q-Network (DRQN) to learn decentralized policies for task offloading, spectrum access, and CPU energy adaptation based on local observations (e.g., queue length, energy, CPU usage, and mobility). To protect privacy, we introduce a secure aggregation protocol based on elliptic curve Diffie Hellman key exchange, which ensures accurate model updates without exposing raw data to semi-honest adversaries. We formulate the resource management problem as a partially observable multi-agent Markov decision process (POMMDP) with a multi-objective reward function that jointly optimizes latency, energy efficiency, spectral efficiency, fairness, and reliability under 6G-specific service requirements such as URLLC, eMBB, and mMTC. Simulation results demonstrate that Fed-MARL outperforms centralized MARL and heuristic baselines in task success rate, latency, energy efficiency, and fairness, while ensuring robust privacy protection and scalability in dynamic, resource-constrained 6G edge networks.

**arXiv ID:** 2509.10163
</details>

<details>
<summary><strong>Robot guide with multi-agent control and automatic scenario generation with LLM</strong> - Elizaveta D. Moskovskaya, Anton D. Moscowsky - [[pdf]](https://arxiv.org/pdf/2509.10317)</summary>

**Abstract:** The work describes the development of a hybrid control architecture for an anthropomorphic tour guide robot, combining a multi-agent resource management system with automatic behavior scenario generation based on large language models. The proposed approach aims to overcome the limitations of traditional systems, which rely on manual tuning of behavior scenarios. These limitations include manual configuration, low flexibility, and lack of naturalness in robot behavior. The process of preparing tour scenarios is implemented through a two-stage generation: first, a stylized narrative is created, then non-verbal action tags are integrated into the text. The multi-agent system ensures coordination and conflict resolution during the execution of parallel actions, as well as maintaining default behavior after the completion of main operations, contributing to more natural robot behavior. The results obtained from the trial demonstrate the potential of the proposed approach for automating and scaling social robot control systems.

**arXiv ID:** 2509.10317
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>LLMs as Agentic Cooperative Players in Multiplayer UNO</strong> - Yago Romano Matinez, Jesse Roberts - [[pdf]](https://arxiv.org/pdf/2509.09867)</summary>

**Abstract:** LLMs promise to assist humans -- not just by answering questions, but by offering useful guidance across a wide range of tasks. But how far does that assistance go? Can a large language model based agent actually help someone accomplish their goal as an active participant? We test this question by engaging an LLM in UNO, a turn-based card game, asking it not to win but instead help another player to do so. We built a tool that allows decoder-only LLMs to participate as agents within the RLCard game environment. These models receive full game-state information and respond using simple text prompts under two distinct prompting strategies. We evaluate models ranging from small (1B parameters) to large (70B parameters) and explore how model scale impacts performance. We find that while all models were able to successfully outperform a random baseline when playing UNO, few were able to significantly aid another player.

**arXiv ID:** 2509.09867
</details>

<details>
<summary><strong>AEGIS: An Agent for Extraction and Geographic Identification in Scholarly Proceedings</strong> - Om Vishesh, Harshad Khadilkar, Deepak Akkil - [[pdf]](https://arxiv.org/pdf/2509.09470)</summary>

**Abstract:** Keeping pace with the rapid growth of academia literature presents a significant challenge for researchers, funding bodies, and academic societies. To address the time-consuming manual effort required for scholarly discovery, we present a novel, fully automated system that transitions from data discovery to direct action. Our pipeline demonstrates how a specialized AI agent, 'Agent-E', can be tasked with identifying papers from specific geographic regions within conference proceedings and then executing a Robotic Process Automation (RPA) to complete a predefined action, such as submitting a nomination form. We validated our system on 586 papers from five different conferences, where it successfully identified every target paper with a recall of 100% and a near perfect accuracy of 99.4%. This demonstration highlights the potential of task-oriented AI agents to not only filter information but also to actively participate in and accelerate the workflows of the academic community.

**arXiv ID:** 2509.09470
</details>

<details>
<summary><strong>We Need a New Ethics for a World of AI Agents</strong> - Iason Gabriel, Geoff Keeling, Arianna Manzini, James Evans - [[pdf]](https://arxiv.org/pdf/2509.10289)</summary>

**Abstract:** 

**arXiv ID:** 2509.10289
</details>

<details>
<summary><strong>Web3 x AI Agents: Landscape, Integrations, and Foundational Challenges</strong> - Yiming Shen, Jiashuo Zhang, Zhenzhe Shao, Wenxuan Luo, Yanlin Wang, Ting Chen, Zibin Zheng, Jiachi Chen - [[pdf]](https://arxiv.org/pdf/2508.02773)</summary>

**Abstract:** The convergence of Web3 technologies and AI agents represents a rapidly evolving frontier poised to reshape decentralized ecosystems. This paper presents the first and most comprehensive analysis of the intersection between Web3 and AI agents, examining five critical dimensions: landscape, economics, governance, security, and trust mechanisms. Through an analysis of 133 existing projects, we first develop a taxonomy and systematically map the current market landscape (RQ1), identifying distinct patterns in project distribution and capitalization. Building upon these findings, we further investigate four key integrations: (1) the role of AI agents in participating in and optimizing decentralized finance (RQ2); (2) their contribution to enhancing Web3 governance mechanisms (RQ3); (3) their capacity to strengthen Web3 security via intelligent vulnerability detection and automated smart contract auditing (RQ4); and (4) the establishment of robust reliability frameworks for AI agent operations leveraging Web3's inherent trust infrastructure (RQ5). By synthesizing these dimensions, we identify key integration patterns, highlight foundational challenges related to scalability, security, and ethics, and outline critical considerations for future research toward building robust, intelligent, and trustworthy decentralized systems with effective AI agent interactions.

**arXiv ID:** 2508.02773
</details>

<details>
<summary><strong>Agentic Vehicles for Human-Centered Mobility Systems</strong> - Jiangbo Yu - [[pdf]](https://arxiv.org/pdf/2507.04996)</summary>

**Abstract:** Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Autonomous vehicles (AuVs) are therefore understood as systems that perceive their environment and execute pre-programmed tasks independently of external input, consistent with the SAE levels of automated driving. Yet recent research and real-world deployments have begun to showcase vehicles that exhibit behaviors outside the scope of this definition. These include natural language interaction with humans, goal adaptation, contextual reasoning, external tool use, and the handling of unforeseen ethical dilemmas, enabled in part by multimodal large language models (LLMs). These developments highlight not only a gap between technical autonomy and the broader cognitive and social capacities required for human-centered mobility, but also the emergence of a form of vehicle intelligence that currently lacks a clear designation. To address this gap, the paper introduces the concept of agentic vehicles (AgVs): vehicles that integrate agentic AI systems to reason, adapt, and interact within complex environments. It synthesizes recent advances in agentic systems and suggests how AgVs can complement and even reshape conventional autonomy to ensure mobility services are aligned with user and societal needs. The paper concludes by outlining key challenges in the development and governance of AgVs and their potential role in shaping future agentic transportation systems.

**arXiv ID:** 2507.04996
</details>

<details>
<summary><strong>Acetrans: An Autonomous Corridor-Based and Efficient UAV Suspended Transport System</strong> - Weiyan Lu, Huizhe Li, Yuhao Fang, Zhexuan Zhou, Junda Wu, Yude Li, Youmin Gong, Jie Mei - [[pdf]](https://arxiv.org/pdf/2509.10349)</summary>

**Abstract:** Unmanned aerial vehicles (UAVs) with suspended payloads offer significant advantages for aerial transportation in complex and cluttered environments. However, existing systems face critical limitations, including unreliable perception of the cable-payload dynamics, inefficient planning in large-scale environments, and the inability to guarantee whole-body safety under cable bending and external disturbances. This paper presents Acetrans, an Autonomous, Corridor-based, and Efficient UAV suspended transport system that addresses these challenges through a unified perception, planning, and control framework. A LiDAR-IMU fusion module is proposed to jointly estimate both payload pose and cable shape under taut and bent modes, enabling robust whole-body state estimation and real-time filtering of cable point clouds. To enhance planning scalability, we introduce the Multi-size-Aware Configuration-space Iterative Regional Inflation (MACIRI) algorithm, which generates safe flight corridors while accounting for varying UAV and payload geometries. A spatio-temporal, corridor-constrained trajectory optimization scheme is then developed to ensure dynamically feasible and collision-free trajectories. Finally, a nonlinear model predictive controller (NMPC) augmented with cable-bending constraints provides robust whole-body safety during execution. Simulation and experimental results validate the effectiveness of Acetrans, demonstrating substantial improvements in perception accuracy, planning efficiency, and control safety compared to state-of-the-art methods.

**arXiv ID:** 2509.10349
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (20 papers)</h2></summary>

<details>
<summary><strong>The (R)evolution of Scientific Workflows in the Agentic AI Era: Towards Autonomous Science</strong> - Woong Shin, Renan Souza, Daniel Rosendo, Frédéric Suter, Feiyi Wang, Prasanna Balaprakash, Rafael Ferreira da Silva - [[pdf]](https://arxiv.org/pdf/2509.09915)</summary>

**Abstract:** Modern scientific discovery increasingly requires coordinating distributed facilities and heterogeneous resources, forcing researchers to act as manual workflow coordinators rather than scientists. Advances in AI leading to AI agents show exciting new opportunities that can accelerate scientific discovery by providing intelligence as a component in the ecosystem. However, it is unclear how this new capability would materialize and integrate in the real world. To address this, we propose a conceptual framework where workflows evolve along two dimensions which are intelligence (from static to intelligent) and composition (from single to swarm) to chart an evolutionary path from current workflow management systems to fully autonomous, distributed scientific laboratories. With these trajectories in mind, we present an architectural blueprint that can help the community take the next steps towards harnessing the opportunities in autonomous science with the potential for 100x discovery acceleration and transformational scientific workflows.

**arXiv ID:** 2509.09915
</details>

<details>
<summary><strong>Virtual Agent Economies</strong> - Nenad Tomasev, Matija Franklin, Joel Z. Leibo, Julian Jacobs, William A. Cunningham, Iason Gabriel, Simon Osindero - [[pdf]](https://arxiv.org/pdf/2509.10147)</summary>

**Abstract:** The rapid adoption of autonomous AI agents is giving rise to a new economic layer where agents transact and coordinate at scales and speeds beyond direct human oversight. We propose the "sandbox economy" as a framework for analyzing this emergent system, characterizing it along two key dimensions: its origins (emergent vs. intentional) and its degree of separateness from the established human economy (permeable vs. impermeable). Our current trajectory points toward a spontaneous emergence of a vast and highly permeable AI agent economy, presenting us with opportunities for an unprecedented degree of coordination as well as significant challenges, including systemic economic risk and exacerbated inequality. Here we discuss a number of possible design choices that may lead to safely steerable AI agent markets. In particular, we consider auction mechanisms for fair resource allocation and preference resolution, the design of AI "mission economies" to coordinate around achieving collective goals, and socio-technical infrastructure needed to ensure trust, safety, and accountability. By doing this, we argue for the proactive design of steerable agent markets to ensure the coming technological shift aligns with humanity's long-term collective flourishing.

**arXiv ID:** 2509.10147
</details>

<details>
<summary><strong>Mutual Information Tracks Policy Coherence in Reinforcement Learning</strong> - Cameron Reid, Wael Hafez, Amirhossein Nazeri - [[pdf]](https://arxiv.org/pdf/2509.10423)</summary>

**Abstract:** Reinforcement Learning (RL) agents deployed in real-world environments face degradation from sensor faults, actuator wear, and environmental shifts, yet lack intrinsic mechanisms to detect and diagnose these failures. We present an information-theoretic framework that reveals both the fundamental dynamics of RL and provides practical methods for diagnosing deployment-time anomalies. Through analysis of state-action mutual information patterns in a robotic control task, we first demonstrate that successful learning exhibits characteristic information signatures: mutual information between states and actions steadily increases from 0.84 to 2.83 bits (238% growth) despite growing state entropy, indicating that agents develop increasingly selective attention to task-relevant patterns. Intriguingly, states, actions and next states joint mutual information, MI(S,A;S'), follows an inverted U-curve, peaking during early learning before declining as the agent specializes suggesting a transition from broad exploration to efficient exploitation. More immediately actionable, we show that information metrics can differentially diagnose system failures: observation-space, i.e., states noise (sensor faults) produces broad collapses across all information channels with pronounced drops in state-action coupling, while action-space noise (actuator faults) selectively disrupts action-outcome predictability while preserving state-action relationships. This differential diagnostic capability demonstrated through controlled perturbation experiments enables precise fault localization without architectural modifications or performance degradation. By establishing information patterns as both signatures of learning and diagnostic for system health, we provide the foundation for adaptive RL systems capable of autonomous fault detection and policy adjustment based on information-theoretic principles.

**arXiv ID:** 2509.10423
</details>

<details>
<summary><strong>Meta-Learning Reinforcement Learning for Crypto-Return Prediction</strong> - Junqiao Wang, Zhaoyang Guan, Guanyu Liu, Tianze Xia, Xianzhi Li, Shuo Yin, Xinyuan Song, Chuhan Cheng, Tianyu Shi, Alex Lee - [[pdf]](https://arxiv.org/pdf/2509.09751)</summary>

**Abstract:** Predicting cryptocurrency returns is notoriously difficult: price movements are driven by a fast-shifting blend of on-chain activity, news flow, and social sentiment, while labeled training data are scarce and expensive. In this paper, we present Meta-RL-Crypto, a unified transformer-based architecture that unifies meta-learning and reinforcement learning (RL) to create a fully self-improving trading agent. Starting from a vanilla instruction-tuned LLM, the agent iteratively alternates between three roles-actor, judge, and meta-judge-in a closed-loop architecture. This learning process requires no additional human supervision. It can leverage multimodal market inputs and internal preference feedback. The agent in the system continuously refines both the trading policy and evaluation criteria. Experiments across diverse market regimes demonstrate that Meta-RL-Crypto shows good performance on the technical indicators of the real market and outperforming other LLM-based baselines.

**arXiv ID:** 2509.09751
</details>

<details>
<summary><strong>Revisiting Actor-Critic Methods in Discrete Action Off-Policy Reinforcement Learning</strong> - Reza Asad, Reza Babanezhad, Sharan Vaswani - [[pdf]](https://arxiv.org/pdf/2509.09838)</summary>

**Abstract:** Value-based approaches such as DQN are the default methods for off-policy reinforcement learning with discrete-action environments such as Atari. Common policy-based methods are either on-policy and do not effectively learn from off-policy data (e.g. PPO), or have poor empirical performance in the discrete-action setting (e.g. SAC). Consequently, starting from discrete SAC (DSAC), we revisit the design of actor-critic methods in this setting. First, we determine that the coupling between the actor and critic entropy is the primary reason behind the poor performance of DSAC. We demonstrate that by merely decoupling these components, DSAC can have comparable performance as DQN. Motivated by this insight, we introduce a flexible off-policy actor-critic framework that subsumes DSAC as a special case. Our framework allows using an m-step Bellman operator for the critic update, and enables combining standard policy optimization methods with entropy regularization to instantiate the resulting actor objective. Theoretically, we prove that the proposed methods can guarantee convergence to the optimal regularized value function in the tabular setting. Empirically, we demonstrate that these methods can approach the performance of DQN on standard Atari games, and do so even without entropy regularization or explicit exploration.

**arXiv ID:** 2509.09838
</details>

<details>
<summary><strong>SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness Under Resource Constraints</strong> - Zhiyu Fan, Kirill Vasilevski, Dayi Lin, Boyuan Chen, Yihao Chen, Zhiqing Zhong, Jie M. Zhang, Pinjia He, Ahmed E. Hassan - [[pdf]](https://arxiv.org/pdf/2509.09853)</summary>

**Abstract:** The advancement of large language models (LLMs) and code agents has demonstrated significant potential to assist software engineering (SWE) tasks, such as autonomous issue resolution and feature addition. Existing AI for software engineering leaderboards (e.g., SWE-bench) focus solely on solution accuracy, ignoring the crucial factor of effectiveness in a resource-constrained world. This is a universal problem that also exists beyond software engineering tasks: any AI system should be more than correct - it must also be cost-effective. To address this gap, we introduce SWE-Effi, a set of new metrics to re-evaluate AI systems in terms of holistic effectiveness scores. We define effectiveness as the balance between the accuracy of outcome (e.g., issue resolve rate) and the resources consumed (e.g., token and time). In this paper, we specifically focus on the software engineering scenario by re-ranking popular AI systems for issue resolution on a subset of the SWE-bench benchmark using our new multi-dimensional metrics. We found that AI system's effectiveness depends not just on the scaffold itself, but on how well it integrates with the base model, which is key to achieving strong performance in a resource-efficient manner. We also identified systematic challenges such as the "token snowball" effect and, more significantly, a pattern of "expensive failures". In these cases, agents consume excessive resources while stuck on unsolvable tasks - an issue that not only limits practical deployment but also drives up the cost of failed rollouts during RL training. Lastly, we observed a clear trade-off between effectiveness under the token budget and effectiveness under the time budget, which plays a crucial role in managing project budgets and enabling scalable reinforcement learning, where fast responses are essential.

**arXiv ID:** 2509.09853
</details>

<details>
<summary><strong>Reinforcement learning for spin torque oscillator tasks</strong> - Jakub Mojsiejuk, Sławomir Ziętek, Witold Skowroński - [[pdf]](https://arxiv.org/pdf/2509.10057)</summary>

**Abstract:** We address the problem of automatic synchronisation of the spintronic oscillator (STO) by means of reinforcement learning (RL). A numerical solution of the macrospin Landau-Lifschitz-Gilbert-Slonczewski equation is used to simulate the STO and we train the two types of RL agents to synchronise with a target frequency within a fixed number of steps. We explore modifications to this base task and show an improvement in both convergence and energy efficiency of the synchronisation that can be easily achieved in the simulated environment.

**arXiv ID:** 2509.10057
</details>

<details>
<summary><strong>Generalizing Beyond Suboptimality: Offline Reinforcement Learning Learns Effective Scheduling through Random Data</strong> - Jesse van Remmerden, Zaharah Bukhsh, Yingqian Zhang - [[pdf]](https://arxiv.org/pdf/2509.10303)</summary>

**Abstract:** The Job-Shop Scheduling Problem (JSP) and Flexible Job-Shop Scheduling Problem (FJSP), are canonical combinatorial optimization problems with wide-ranging applications in industrial operations. In recent years, many online reinforcement learning (RL) approaches have been proposed to learn constructive heuristics for JSP and FJSP. Although effective, these online RL methods require millions of interactions with simulated environments that may not capture real-world complexities, and their random policy initialization leads to poor sample efficiency. To address these limitations, we introduce Conservative Discrete Quantile Actor-Critic (CDQAC), a novel offline RL algorithm that learns effective scheduling policies directly from historical data, eliminating the need for costly online interactions, while maintaining the ability to improve upon suboptimal training data. CDQAC couples a quantile-based critic with a delayed policy update, estimating the return distribution of each machine-operation pair rather than selecting pairs outright. Our extensive experiments demonstrate CDQAC's remarkable ability to learn from diverse data sources. CDQAC consistently outperforms the original data-generating heuristics and surpasses state-of-the-art offline and online RL baselines. In addition, CDQAC is highly sample efficient, requiring only 10-20 training instances to learn high-quality policies. Surprisingly, we find that CDQAC performs better when trained on data generated by a random heuristic than when trained on higher-quality data from genetic algorithms and priority dispatching rules.

**arXiv ID:** 2509.10303
</details>

<details>
<summary><strong>A Conflicts-free, Speed-lossless KAN-based Reinforcement Learning Decision System for Interactive Driving in Roundabouts</strong> - Zhihao Lin, Zhen Tian, Jianglin Lan, Qi Zhang, Ziyang Ye, Hanyang Zhuang, Xianxian Zhao - [[pdf]](https://arxiv.org/pdf/2408.08242)</summary>

**Abstract:** Safety and efficiency are crucial for autonomous driving in roundabouts, especially mixed traffic with both autonomous vehicles (AVs) and human-driven vehicles. This paper presents a learning-based algorithm that promotes safe and efficient driving across varying roundabout traffic conditions. A deep Q-learning network is used to learn optimal strategies in complex multi-vehicle roundabout scenarios, while a Kolmogorov-Arnold Network (KAN) improves the AVs' environmental understanding. To further enhance safety, an action inspector filters unsafe actions, and a route planner optimizes driving efficiency. Moreover, model predictive control ensures stability and precision in execution. Experimental results demonstrate that the proposed system consistently outperforms state-of-the-art methods, achieving fewer collisions, reduced travel time, and stable training with smooth reward convergence.

**arXiv ID:** 2408.08242
</details>

<details>
<summary><strong>AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning</strong> - Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, Tongkai Yang, Binhang Yuan, Yi Wu - [[pdf]](https://arxiv.org/pdf/2505.24298)</summary>

**Abstract:** Reinforcement learning (RL) has become a dominant paradigm for training large language models (LLMs), particularly for reasoning tasks. Effective RL for LLMs requires massive parallelization and poses an urgent need for efficient training systems. Most existing large-scale RL systems for LLMs are synchronous, alternating generation and training in a batch setting where rollouts in each training batch are generated by the same model. This approach stabilizes RL training but suffers from severe system-level inefficiency: generation must wait until the longest output in the batch is completed before model updates, resulting in GPU underutilization. We present AReaL, a fully asynchronous RL system that completely decouples generation from training. Rollout workers in AReaL continuously generate new outputs without waiting, while training workers update the model whenever a batch of data is collected. AReaL also incorporates a collection of system-level optimizations, leading to substantially higher GPU utilization. To stabilize RL training, AReaL balances the workload of rollout and training workers to control data staleness, and adopts a staleness-enhanced PPO variant to better handle outdated training samples. Extensive experiments on math and code reasoning benchmarks show that AReaL achieves up to 2.77$\times$ training speedup compared to synchronous systems with the same number of GPUs and matched or improved final performance. The code of AReaL is available at this https URL.

**arXiv ID:** 2505.24298
</details>

<details>
<summary><strong>HiLight: A Hierarchical Reinforcement Learning Framework with Global Adversarial Guidance for Large-Scale Traffic Signal Control</strong> - Yaqiao Zhu, Hongkai Wen, Geyong Min, Man Luo - [[pdf]](https://arxiv.org/pdf/2506.14391)</summary>

**Abstract:** Efficient traffic signal control (TSC) is essential for mitigating urban congestion, yet existing reinforcement learning (RL) methods face challenges in scaling to large networks while maintaining global coordination. Centralized RL suffers from scalability issues, while decentralized approaches often lack unified objectives, resulting in limited network-level efficiency. In this paper, we propose HiLight, a hierarchical reinforcement learning framework with global adversarial guidance for large-scale TSC. HiLight consists of a high-level Meta-Policy, which partitions the traffic network into subregions and generates sub-goals using a Transformer-LSTM architecture, and a low-level Sub-Policy, which controls individual intersections with global awareness. To improve the alignment between global planning and local execution, we introduce an adversarial training mechanism, where the Meta-Policy generates challenging yet informative sub-goals, and the Sub-Policy learns to surpass these targets, leading to more effective coordination. We evaluate HiLight across both synthetic and real-world benchmarks, and additionally construct a large-scale Manhattan network with diverse traffic conditions, including peak transitions, adverse weather, and holiday surges. Experimental results show that HiLight exhibits significant advantages in large-scale scenarios and remains competitive across standard benchmarks of varying sizes.

**arXiv ID:** 2506.14391
</details>

<details>
<summary><strong>OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning</strong> - Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan - [[pdf]](https://arxiv.org/pdf/2509.09332)</summary>

**Abstract:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible. To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: this https URL

**arXiv ID:** 2509.09332
</details>

<details>
<summary><strong>Topic-Guided Reinforcement Learning with LLMs for Enhancing Multi-Document Summarization</strong> - Chuyuan Li, Austin Xu, Shafiq Joty, Giuseppe Carenini - [[pdf]](https://arxiv.org/pdf/2509.09852)</summary>

**Abstract:** A key challenge in Multi-Document Summarization (MDS) is effectively integrating information from multiple sources while maintaining coherence and topical relevance. While Large Language Models have shown impressive results in single-document summarization, their performance on MDS still leaves room for improvement. In this paper, we propose a topic-guided reinforcement learning approach to improve content selection in MDS. We first show that explicitly prompting models with topic labels enhances the informativeness of the generated summaries. Building on this insight, we propose a novel topic reward within the Group Relative Policy Optimization (GRPO) framework to measure topic alignment between the generated summary and source documents. Experimental results on the Multi-News and Multi-XScience datasets demonstrate that our method consistently outperforms strong baselines, highlighting the effectiveness of leveraging topical cues in MDS.

**arXiv ID:** 2509.09852
</details>

<details>
<summary><strong>DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL</strong> - Rui Lu, Zhenyu Hou, Zihan Wang, Hanchen Zhang, Xiao Liu, Yujiang Li, Shi Feng, Jie Tang, Yuxiao Dong - [[pdf]](https://arxiv.org/pdf/2509.10446)</summary>

**Abstract:** Augmenting large language models (LLMs) with browsing tools substantially improves their potential as deep search agents to solve complex, real-world tasks. Yet, open LLMs still perform poorly in such settings due to limited long-horizon reasoning capacity with browsing tools and the lack of sufficiently difficult supervised data. To address these challenges, we present DeepDive to advance deep search agents. First, we propose a strategy to automatically synthesize complex, difficult, and hard-to-find questions from open knowledge graphs. Second, we apply end-to-end multi-turn reinforcement learning (RL) to enhance LLMs' long-horizon reasoning with deep search. Experiments show that DeepDive-32B achieves a new open-source competitive result on BrowseComp, outperforming WebSailor, DeepSeek-R1-Browse, and Search-o1. We demonstrate that multi-turn RL training improves deep search ability and significantly contributes to the performance improvements across multiple benchmarks. We observe that DeepDive enables test-time scaling of tool calls and parallel sampling. All datasets, models, and code are publicly available at this https URL.

**arXiv ID:** 2509.10446
</details>

<details>
<summary><strong>Parallel-R1: Towards Parallel Thinking via Reinforcement Learning</strong> - Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, Dong Yu - [[pdf]](https://arxiv.org/pdf/2509.07980)</summary>

**Abstract:** Parallel thinking has emerged as a novel approach for enhancing the reasoning capabilities of large language models (LLMs) by exploring multiple reasoning paths concurrently. However, activating such capabilities through training remains challenging, as existing methods predominantly rely on supervised fine-tuning (SFT) over synthetic data, which encourages teacher-forced imitation rather than exploration and generalization. Different from them, we propose \textbf{Parallel-R1}, the first reinforcement learning (RL) framework that enables parallel thinking behaviors for complex real-world reasoning tasks. Our framework employs a progressive curriculum that explicitly addresses the cold-start problem in training parallel thinking with RL. We first use SFT on prompt-generated trajectories from easier tasks to instill the parallel thinking ability, then transition to RL to explore and generalize this skill on harder problems. Experiments on various math benchmarks, including MATH, AMC23, and AIME, show that Parallel-R1 successfully instills parallel thinking, leading to 8.4% accuracy improvements over the sequential thinking model trained directly on challenging tasks with RL. Further analysis reveals a clear shift in the model's thinking behavior: at an early stage, it uses parallel thinking as an exploration strategy, while in a later stage, it uses the same capability for multi-perspective verification. Most significantly, we validate parallel thinking as a \textbf{mid-training exploration scaffold}, where this temporary exploratory phase unlocks a higher performance ceiling after RL, yielding a 42.9% improvement over the baseline on AIME25. Our model, data, and code will be open-source at this https URL.

**arXiv ID:** 2509.07980
</details>

<details>
<summary><strong>Hybrid Adaptive Conformal Offline Reinforcement Learning for Fair Population Health Management</strong> - Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, Rajaie Batniji - [[pdf]](https://arxiv.org/pdf/2509.09772)</summary>

**Abstract:** Population health management programs for Medicaid populations coordinate longitudinal outreach and services (e.g., benefits navigation, behavioral health, social needs support, and clinical scheduling) and must be safe, fair, and auditable. We present a Hybrid Adaptive Conformal Offline Reinforcement Learning (HACO) framework that separates risk calibration from preference optimization to generate conservative action recommendations at scale. In our setting, each step involves choosing among common coordination actions (e.g., which member to contact, by which modality, and whether to route to a specialized service) while controlling the near-term risk of adverse utilization events (e.g., unplanned emergency department visits or hospitalizations). Using a de-identified operational dataset from Waymark comprising 2.77 million sequential decisions across 168,126 patients, HACO (i) trains a lightweight risk model for adverse events, (ii) derives a conformal threshold to mask unsafe actions at a target risk level, and (iii) learns a preference policy on the resulting safe subset. We evaluate policies with a version-agnostic fitted Q evaluation (FQE) on stratified subsets and audit subgroup performance across age, sex, and race. HACO achieves strong risk discrimination (AUC ~0.81) with a calibrated threshold ( {\tau} ~0.038 at {\alpha} = 0.10), while maintaining high safe coverage. Subgroup analyses reveal systematic differences in estimated value across demographics, underscoring the importance of fairness auditing. Our results show that conformal risk gating integrates cleanly with offline RL to deliver conservative, auditable decision support for population health management teams.

**arXiv ID:** 2509.09772
</details>

<details>
<summary><strong>Off Policy Lyapunov Stability in Reinforcement Learning</strong> - Sarvan Gill, Daniela Constantinescu - [[pdf]](https://arxiv.org/pdf/2509.09863)</summary>

**Abstract:** Traditional reinforcement learning lacks the ability to provide stability guarantees. More recent algorithms learn Lyapunov functions alongside the control policies to ensure stable learning. However, the current self-learned Lyapunov functions are sample inefficient due to their on-policy nature. This paper introduces a method for learning Lyapunov functions off-policy and incorporates the proposed off-policy Lyapunov function into the Soft Actor Critic and Proximal Policy Optimization algorithms to provide them with a data efficient stability certificate. Simulations of an inverted pendulum and a quadrotor illustrate the improved performance of the two algorithms when endowed with the proposed off-policy Lyapunov function.

**arXiv ID:** 2509.09863
</details>

<details>
<summary><strong>Quantum-Enhanced Forecasting for Deep Reinforcement Learning in Algorithmic Trading</strong> - Jun-Hao Chen, Yu-Chien Huang, Yun-Cheng Tsai, Samuel Yen-Chi Chen - [[pdf]](https://arxiv.org/pdf/2509.09176)</summary>

**Abstract:** The convergence of quantum-inspired neural networks and deep reinforcement learning offers a promising avenue for financial trading. We implemented a trading agent for USD/TWD by integrating Quantum Long Short-Term Memory (QLSTM) for short-term trend prediction with Quantum Asynchronous Advantage Actor-Critic (QA3C), a quantum-enhanced variant of the classical A3C. Trained on data from 2000-01-01 to 2025-04-30 (80\% training, 20\% testing), the long-only agent achieves 11.87\% return over around 5 years with 0.92\% max drawdown, outperforming several currency ETFs. We detail state design (QLSTM features and indicators), reward function for trend-following/risk control, and multi-core training. Results show hybrid models yield competitive FX trading performance. Implications include QLSTM's effectiveness for small-profit trades with tight risk and future enhancements. Key hyperparameters: QLSTM sequence length$=$4, QA3C workers$=$8. Limitations: classical quantum simulation and simplified strategy. \footnote{The views expressed in this article are those of the authors and do not represent the views of Wells Fargo. This article is for informational purposes only. Nothing contained in this article should be construed as investment advice. Wells Fargo makes no express or implied warranties and expressly disclaims all legal, tax, and accounting implications related to this article.

**arXiv ID:** 2509.09176
</details>

<details>
<summary><strong>Integrating Diffusion-based Multi-task Learning with Online Reinforcement Learning for Robust Quadruped Robot Control</strong> - Xinyao Qin, Xiaoteng Ma, Yang Qi, Qihan Liu, Chuanyi Xue, Ning Gui, Qinyu Dong, Jun Yang, Bin Liang - [[pdf]](https://arxiv.org/pdf/2507.05674)</summary>

**Abstract:** Recent research has highlighted the powerful capabilities of imitation learning in robotics. Leveraging generative models, particularly diffusion models, these approaches offer notable advantages such as strong multi-task generalization, effective language conditioning, and high sample efficiency. While their application has been successful in manipulation tasks, their use in legged locomotion remains relatively underexplored, mainly due to compounding errors that affect stability and difficulties in task transition under limited data. Online reinforcement learning (RL) has demonstrated promising results in legged robot control in the past years, providing valuable insights to address these challenges. In this work, we propose DMLoco, a diffusion-based framework for quadruped robots that integrates multi-task pretraining with online PPO finetuning to enable language-conditioned control and robust task transitions. Our approach first pretrains the policy on a diverse multi-task dataset using diffusion models, enabling language-guided execution of various skills. Then, it finetunes the policy in simulation to ensure robustness and stable task transition during real-world deployment. By utilizing Denoising Diffusion Implicit Models (DDIM) for efficient sampling and TensorRT for optimized deployment, our policy runs onboard at 50Hz, offering a scalable and efficient solution for adaptive, language-guided locomotion on resource-constrained robotic platforms.

**arXiv ID:** 2507.05674
</details>

<details>
<summary><strong>CTBC: Contact-Triggered Blind Climbing for Wheeled Bipedal Robots with Instruction Learning and Reinforcement Learning</strong> - Rankun Li, Hao Wang, Qi Li, Zhuo Han, Yifei Chu, Linqi Ye, Wende Xie, Wenlong Liao - [[pdf]](https://arxiv.org/pdf/2509.02986)</summary>

**Abstract:** In recent years, wheeled bipedal robots have gained increasing attention due to their advantages in mobility, such as high-speed locomotion on flat terrain. However, their performance on complex environments (e.g., staircases) remains inferior to that of traditional legged robots. To overcome this limitation, we propose a general contact-triggered blind climbing (CTBC) framework for wheeled bipedal robots. Upon detecting wheel-obstacle contact, the robot triggers a leg-lifting motion to overcome the obstacle. By leveraging a strongly-guided feedforward trajectory, our method enables the robot to rapidly acquire agile leg-lifting skills, significantly enhancing its capability to traverse unstructured terrains. The approach has been experimentally validated and successfully deployed on LimX Dynamics' wheeled bipedal robot, Tron1. Real-world tests demonstrate that Tron1 can reliably climb obstacles well beyond its wheel radius using only proprioceptive feedback.

**arXiv ID:** 2509.02986
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
