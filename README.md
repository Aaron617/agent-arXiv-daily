# Agent arXiv Daily

**Last Updated:** 2025-11-11 02:50:36

**Total Papers:** 49

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
<summary><strong>HugAgent: Benchmarking LLMs for Simulation of Individualized Human Reasoning</strong> - Chance Jiajie Li, Zhenze Mo, Yuhan Tang, Ao Qu, Jiayi Wu, Kaiya Ivy Zhao, Yulu Gan, Jie Fan, Jiangbo Yu, Hang Jiang, Paul Pu Liang, Jinhua Zhao, Luis Alberto Alonso Pastor, Kent Larson - [[pdf]](https://arxiv.org/pdf/2510.15144)</summary>

**Abstract:** Simulating human reasoning in open-ended tasks has long been a central aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), which rethinks human reasoning simulation along three dimensions: (i) from averaged to individualized reasoning, (ii) from behavioral mimicry to cognitive alignment, and (iii) from vignette-based to open-ended data. The benchmark evaluates whether a model can predict a specific person's behavioral responses and the underlying reasoning dynamics in out-of-distribution scenarios, given partial evidence of their prior views. HugAgent adopts a dual-track design: a human track that automates and scales the think-aloud method to collect ecologically valid human reasoning data, and a synthetic track for further scalability and systematic stress testing. This architecture enables low-cost, extensible expansion to new tasks and populations. Experiments with state-of-the-art language models reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. The benchmark, along with its complete data collection pipeline and companion chatbot, is open-sourced as HugAgent (this https URL) and TraceYourThinking (this https URL).

**arXiv ID:** 2510.15144
</details>

<details>
<summary><strong>AURA: A Reinforcement Learning Framework for AI-Driven Adaptive Conversational Surveys</strong> - Jinwen Tang, Yi Shang - [[pdf]](https://arxiv.org/pdf/2510.27126)</summary>

**Abstract:** Conventional online surveys provide limited personalization, often resulting in low engagement and superficial responses. Although AI survey chatbots improve convenience, most are still reactive: they rely on fixed dialogue trees or static prompt templates and therefore cannot adapt within a session to fit individual users, which leads to generic follow-ups and weak response quality. We address these limitations with AURA (Adaptive Understanding through Reinforcement Learning for Assessment), a reinforcement learning framework for AI-driven adaptive conversational surveys. AURA quantifies response quality using a four-dimensional LSDE metric (Length, Self-disclosure, Emotion, and Specificity) and selects follow-up question types via an epsilon-greedy policy that updates the expected quality gain within each session. Initialized with priors extracted from 96 prior campus-climate conversations (467 total chatbot-user exchanges), the system balances exploration and exploitation across 10-15 dialogue exchanges, dynamically adapting to individual participants in real time. In controlled evaluations, AURA achieved a +0.076 mean gain in response quality and a statistically significant improvement over non-adaptive baselines (p=0.044, d=0.66), driven by a 63% reduction in specification prompts and a 10x increase in validation behavior. These results demonstrate that reinforcement learning can give survey chatbots improved adaptivity, transforming static questionnaires into interactive, self-improving assessment systems.

**arXiv ID:** 2510.27126
</details>

<details>
<summary><strong>Building Specialized Software-Assistant ChatBot with Graph-Based Retrieval-Augmented Generation</strong> - Mohammed Hilel, Yannis Karmim, Jean De Bodinat, Reda Sarehane, Antoine Gillon - [[pdf]](https://arxiv.org/pdf/2511.05297)</summary>

**Abstract:** Digital Adoption Platforms (DAPs) have become essential tools for helping employees navigate complex enterprise software such as CRM, ERP, or HRMS systems. Companies like LemonLearning have shown how digital guidance can reduce training costs and accelerate onboarding. However, building and maintaining these interactive guides still requires extensive manual effort. Leveraging Large Language Models as virtual assistants is an appealing alternative, yet without a structured understanding of the target software, LLMs often hallucinate and produce unreliable answers. Moreover, most production-grade LLMs are black-box APIs, making fine-tuning impractical due to the lack of access to model weights. In this work, we introduce a Graph-based Retrieval-Augmented Generation framework that automatically converts enterprise web applications into state-action knowledge graphs, enabling LLMs to generate grounded and context-aware assistance. The framework was co-developed with the AI enterprise RAKAM, in collaboration with Lemon Learning. We detail the engineering pipeline that extracts and structures software interfaces, the design of the graph-based retrieval process, and the integration of our approach into production DAP workflows. Finally, we discuss scalability, robustness, and deployment lessons learned from industrial use cases.

**arXiv ID:** 2511.05297
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (8 papers)</h2></summary>

<details>
<summary><strong>Learning to reason about rare diseases through retrieval-augmented agents</strong> - Ha Young Kim, Jun Li, Ana Beatriz Solana, Carolin M. Pirkl, Benedikt Wiestler, Julia A. Schnabel, Cosmin I. Bercea - [[pdf]](https://arxiv.org/pdf/2511.04720)</summary>

**Abstract:** Rare diseases represent the long tail of medical imaging, where AI models often fail due to the scarcity of representative training data. In clinical workflows, radiologists frequently consult case reports and literature when confronted with unfamiliar findings. Following this line of reasoning, we introduce RADAR, Retrieval Augmented Diagnostic Reasoning Agents, an agentic system for rare disease detection in brain MRI. Our approach uses AI agents with access to external medical knowledge by embedding both case reports and literature using sentence transformers and indexing them with FAISS to enable efficient similarity search. The agent retrieves clinically relevant evidence to guide diagnostic decision making on unseen diseases, without the need of additional training. Designed as a model-agnostic reasoning module, RADAR can be seamlessly integrated with diverse large language models, consistently improving their rare pathology recognition and interpretability. On the NOVA dataset comprising 280 distinct rare diseases, RADAR achieves up to a 10.2% performance gain, with the strongest improvements observed for open source models such as DeepSeek. Beyond accuracy, the retrieved examples provide interpretable, literature grounded explanations, highlighting retrieval-augmented reasoning as a powerful paradigm for low-prevalence conditions in medical imaging.

**arXiv ID:** 2511.04720
</details>

<details>
<summary><strong>SmartSecChain-SDN: A Blockchain-Integrated Intelligent Framework for Secure and Efficient Software-Defined Networks</strong> - Azhar Hussain Mozumder, M. John Basha, Chayapathi A. R - [[pdf]](https://arxiv.org/pdf/2511.05156)</summary>

**Abstract:** With more and more existing networks being transformed to Software-Defined Networking (SDN), they need to be more secure and demand smarter ways of traffic control. This work, SmartSecChain-SDN, is a platform that combines machine learning based intrusion detection, blockchain-based storage of logs, and application-awareness-based priority in SDN networks. To detect network intrusions in a real-time, precision and low-false positives setup, the framework utilizes the application of advanced machine learning algorithms, namely Random Forest, XGBoost, CatBoost, and CNN-BiLSTM. SmartSecChain-SDN is based on the Hyperledger Fabric, which is a permissioned blockchain technology, to provide secure, scalable, and privacy-preserving storage and, thus, guarantee that the Intrusion Detection System (IDS) records cannot be altered and can be analyzed comprehensively. The system also has Quality of Service (QoS) rules and traffic shaping based on applications, which enables prioritization of critical services, such as VoIP, video conferencing, and business applications, as well as de-prioritization of non-essential traffic, such as downloads and updates. Mininet can simulate real-time SDN scenarios because it is used to prototype whole architectures. It is also compatible with controllers OpenDaylight and Ryu. It has tested the framework using the InSDN dataset and proved that it can identify different kinds of cyberattacks and handle bandwidth allocation efficiently under circumstances of resource constraints. SmartSecChain-SDN comprehensively addresses SDN system protection, securing and enhancing. The proposed study offers an innovative, extensible way to improve cybersecurity, regulatory compliance, and the administration of next-generation programmable networks.

**arXiv ID:** 2511.05156
</details>

<details>
<summary><strong>SWE-Compass: Towards Unified Evaluation of Agentic Coding Abilities for Large Language Models</strong> - Jingxuan Xu, Ken Deng, Weihao Li, Songwei Yu, Huaixi Tang, Haoyang Huang, Zhiyi Lai, Zizheng Zhan, Yanan Wu, Chenchen Zhang, Kepeng Lei, Yifan Yao, Xinping Lei, Wenqiang Zhu, Zongxian Feng, Han Li, Junqi Xiong, Dailin Li, Zuchen Gao, Kun Wu, Wen Xiang, Ziqi Zhan, Yuanxing Zhang, Wuxuan Gong, Ziyuan Gao, Guanxiang Wang, Yirong Xue, Xiaojiang Zhang, Jinghui Wang, Huiming Wang, Wenhao Zhuang, Zhaoxiang Zhang, Yuqun Zhang, Haotian Zhang, Bin Chen, Jiaheng Liu - [[pdf]](https://arxiv.org/pdf/2511.05459)</summary>

**Abstract:** Evaluating large language models (LLMs) for software engineering has been limited by narrow task coverage, language bias, and insufficient alignment with real-world developer workflows. Existing benchmarks often focus on algorithmic problems or Python-centric bug fixing, leaving critical dimensions of software engineering underexplored. To address these gaps, we introduce SWE-Compass1, a comprehensive benchmark that unifies heterogeneous code-related evaluations into a structured and production-aligned framework. SWE-Compass spans 8 task types, 8 programming scenarios, and 10 programming languages, with 2000 high-quality instances curated from authentic GitHub pull requests and refined through systematic filtering and validation. We benchmark ten state-of-the-art LLMs under two agentic frameworks, SWE-Agent and Claude Code, revealing a clear hierarchy of difficulty across task types, languages, and scenarios. Moreover, by aligning evaluation with real-world developer practices, SWE-Compass provides a rigorous and reproducible foundation for diagnosing and advancing agentic coding capabilities in large language models.

**arXiv ID:** 2511.05459
</details>

<details>
<summary><strong>Open Agent Specification (Agent Spec): A Unified Representation for AI Agents</strong> - Soufiane Amini, Yassine Benajiba, Cesare Bernardis, Paul Cayet, Hassan Chafi, Abderrahim Fathan, Louis Faucon, Damien Hilloulin, Sungpack Hong, Ingo Kossyk, Tran Minh Son Le, Rhicheek Patra, Sujith Ravi, Jonas Schweizer, Jyotika Singh, Shailender Singh, Weiyi Sun, Kartik Talamadupula, Jerry Xu - [[pdf]](https://arxiv.org/pdf/2510.04173)</summary>

**Abstract:** The proliferation of agent frameworks has led to fragmentation in how agents are defined, executed, and evaluated. Existing systems differ in their abstractions, data flow semantics, and tool integrations, making it difficult to share or reproduce workflows. We introduce Open Agent Specification (Agent Spec), a declarative language that defines AI agents and agentic workflows in a way that is compatible across frameworks, promoting reusability, portability and interoperability of AI agents. Agent Spec defines a common set of components, control and data flow semantics, and schemas that allow an agent to be defined once and executed across different runtimes. Agent Spec also introduces a standardized Evaluation harness to assess agent behavior and agentic workflows across runtimes - analogous to how HELM and related harnesses standardized LLM evaluation - so that performance, robustness, and efficiency can be compared consistently across frameworks. We demonstrate this using four distinct runtimes (LangGraph, CrewAI, AutoGen, and WayFlow) evaluated over three different benchmarks (SimpleQA Verified, $\tau^2$-Bench and BIRD-SQL). We provide accompanying toolsets: a Python SDK (PyAgentSpec), a reference runtime (WayFlow), and adapters for popular frameworks (e.g., LangGraph, AutoGen, CrewAI). Agent Spec bridges the gap between model-centric and agent-centric standardization & evaluation, laying the groundwork for reliable, reusable, and portable agentic systems.

**arXiv ID:** 2510.04173
</details>

<details>
<summary><strong>A Proprietary Model-Based Safety Response Framework for AI Agents</strong> - Qi Li, Jianjun Xu, Pingtao Wei, Jiu Li, Peiqiang Zhao, Jiwei Shi, Xuan Zhang, Yanhui Yang, Xiaodong Hui, Peng Xu, Wenqin Shao - [[pdf]](https://arxiv.org/pdf/2511.03138)</summary>

**Abstract:** With the widespread application of Large Language Models (LLMs), their associated security issues have become increasingly prominent, severely constraining their trustworthy deployment in critical domains. This paper proposes a novel safety response framework designed to systematically safeguard LLMs at both the input and output levels. At the input level, the framework employs a supervised fine-tuning-based safety classification model. Through a fine-grained four-tier taxonomy (Safe, Unsafe, Conditionally Safe, Focused Attention), it performs precise risk identification and differentiated handling of user queries, significantly enhancing risk coverage and business scenario adaptability, and achieving a risk recall rate of 99.3%. At the output level, the framework integrates Retrieval-Augmented Generation (RAG) with a specifically fine-tuned interpretation model, ensuring all responses are grounded in a real-time, trustworthy knowledge base. This approach eliminates information fabrication and enables result traceability. Experimental results demonstrate that our proposed safety control model achieves a significantly higher safety score on public safety evaluation benchmarks compared to the baseline model, TinyR1-Safety-8B. Furthermore, on our proprietary high-risk test set, the framework's components attained a perfect 100% safety score, validating their exceptional protective capabilities in complex risk scenarios. This research provides an effective engineering pathway for building high-security, high-trust LLM applications.

**arXiv ID:** 2511.03138
</details>

<details>
<summary><strong>AI Agentic Vulnerability Injection And Transformation with Optimized Reasoning</strong> - Amine Lbath, Massih-Reza Amini, Aurelien Delaitre, Vadim Okun - [[pdf]](https://arxiv.org/pdf/2508.20866)</summary>

**Abstract:** The increasing complexity of software systems and the sophistication of cyber-attacks have underscored the critical need for effective automated vulnerability detection and repair systems. Data-driven approaches using deep learning models show promise but critically depend on the availability of large, accurately labeled datasets. Yet existing datasets either suffer from noisy labels, limited range of vulnerabilities, or fail to reflect vulnerabilities as they occur in real-world software. This also limits large-scale benchmarking of such solutions. Automated vulnerability injection provides a way to directly address these dataset limitations, but existing techniques remain limited in coverage, contextual fidelity, or injection success rates. In this paper, we present AVIATOR, the first AI-agentic vulnerability injection workflow. It automatically injects realistic, category-specific vulnerabilities for high-fidelity, diverse, large-scale vulnerability dataset generation. Unlike prior monolithic approaches, AVIATOR orchestrates specialized AI agents, function agents and traditional code analysis tools that replicate expert reasoning. It combines semantic analysis, injection synthesis enhanced with LoRA-based fine-tuning and Retrieval-Augmented Generation, as well as post-injection validation via static analysis and LLM-based discriminators. This modular decomposition allows specialized agents to focus on distinct tasks, improving robustness of injection and reducing error propagation across the workflow. Evaluations across three distinct benchmarks demonstrate that AVIATOR achieves 91%-95% injection success rates, significantly surpassing existing automated dataset generation techniques in both accuracy and scope of software vulnerabilities.

**arXiv ID:** 2508.20866
</details>

<details>
<summary><strong>Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents</strong> - Gauri Kholkar, Ratinder Ahuja - [[pdf]](https://arxiv.org/pdf/2509.23994)</summary>

**Abstract:** As autonomous AI agents are used in regulated and safety-critical settings, organizations need effective ways to turn policy into enforceable controls. We introduce a regulatory machine learning framework that converts unstructured design artifacts (like PRDs, TDDs, and code) into verifiable runtime guardrails. Our Policy as Prompt method reads these documents and risk controls to build a source-linked policy tree. This tree is then compiled into lightweight, prompt-based classifiers for real-time runtime monitoring. The system is built to enforce least privilege and data minimization. For conformity assessment, it provides complete provenance, traceability, and audit logging, all integrated with a human-in-the-loop review process. Evaluations show our system reduces prompt-injection risk, blocks out-of-scope requests, and limits toxic outputs. It also generates auditable rationales aligned with AI governance frameworks. By treating policies as executable prompts (a policy-as-code for agents), this approach enables secure-by-design deployment, continuous compliance, and scalable AI safety and AI security assurance for regulatable ML.

**arXiv ID:** 2509.23994
</details>

<details>
<summary><strong>Do intelligent tutoring systems benefit K-12 students? A meta-analysis and evaluation of heterogeneity of treatment effects in the U.S</strong> - Walter L. Leite, Huibin Zhang, Shibani Rana, Yide Hao, Amber D. Hatch, Lingchen Kong, Huan Kuang - [[pdf]](https://arxiv.org/pdf/2511.04997)</summary>

**Abstract:** To expand the use of intelligent tutoring systems (ITS) in K-12 schools, it is essential to understand the conditions under which their use is most beneficial. This meta-analysis evaluated the heterogeneity of ITS effects across studies focusing on elementary, middle, and high schools in the U.S. It included 18 studies with 77 effect sizes across 11 ITS. Overall, there was a significant positive effect size of ITS on U.S. K-12 students' learning outcomes (g=0.271, SE=0.011, p=0.001). Furthermore, effect sizes were similar across elementary and middle schools, and for low-achieving students, but were lower in studies including rural schools. A MetaForest analysis showed that providing worked-out examples, intervention duration, intervention condition, type of learning outcome, and immediate measurement were the most important moderators of treatment effects.

**arXiv ID:** 2511.04997
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance</strong> - Valeriu Dimidov, Faisal Hawlader, Sasan Jafarnejad, Raphaël Frank - [[pdf]](https://arxiv.org/pdf/2511.05311)</summary>

**Abstract:** Economic constraints, limited availability of datasets for reproducibility and shortages of specialized expertise have long been recognized as key challenges to the adoption and advancement of predictive maintenance (PdM) in the automotive sector. Recent progress in large language models (LLMs) presents an opportunity to overcome these barriers and speed up the transition of PdM from research to industrial practice. Under these conditions, we explore the potential of LLM-based agents to support PdM cleaning pipelines. Specifically, we focus on maintenance logs, a critical data source for training well-performing machine learning (ML) models, but one often affected by errors such as typos, missing fields, near-duplicate entries, and incorrect dates. We evaluate LLM agents on cleaning tasks involving six distinct types of noise. Our findings show that LLMs are effective at handling generic cleaning tasks and offer a promising foundation for future industrial applications. While domain-specific errors remain challenging, these results highlight the potential for further improvements through specialized training and enhanced agentic capabilities.

**arXiv ID:** 2511.05311
</details>

<details>
<summary><strong>Internal World Models as Imagination Networks in Cognitive Agents</strong> - Saurabh Ranjan, Brian Odegaard - [[pdf]](https://arxiv.org/pdf/2510.04391)</summary>

**Abstract:** What is the computational objective of imagination? While classical interpretations suggest imagination is useful for maximizing rewards, recent findings challenge this view. In this study, we propose that imagination serves to access an internal world model (IWM) and use psychological network analysis to explore IWMs in humans and large language models (LLMs). Specifically, we assessed imagination vividness ratings using two questionnaires and constructed imagination networks from these reports. Imagination networks from human groups showed correlations between different centrality measures, including expected influence, strength, and closeness. However, imagination networks from LLMs showed a lack of clustering and lower correlations between centrality measures under different prompts and conversational memory conditions. Together, these results indicate a lack of similarity between IWMs in human and LLM agents. Overall, our study offers a novel method for comparing internally-generated representations in humans and AI, providing insights for developing human-like imagination in artificial intelligence.

**arXiv ID:** 2510.04391
</details>

<details>
<summary><strong>AgentExpt: Automating AI Experiment Design with LLM-based Resource Retrieval Agent</strong> - Yu Li, Lehui Li, Qingmin Liao, Fengli Xu, Yong Li - [[pdf]](https://arxiv.org/pdf/2511.04921)</summary>

**Abstract:** Large language model agents are becoming increasingly capable at web-centric tasks such as information retrieval, complex reasoning. These emerging capabilities have given rise to surge research interests in developing LLM agent for facilitating scientific quest. One key application in AI research is to automate experiment design through agentic dataset and baseline retrieval. However, prior efforts suffer from limited data coverage, as recommendation datasets primarily harvest candidates from public portals and omit many datasets actually used in published papers, and from an overreliance on content similarity that biases model toward superficial similarity and overlooks experimental suitability. Harnessing collective perception embedded in the baseline and dataset citation network, we present a comprehensive framework for baseline and dataset recommendation. First, we design an automated data-collection pipeline that links roughly one hundred thousand accepted papers to the baselines and datasets they actually used. Second, we propose a collective perception enhanced retriever. To represent the position of each dataset or baseline within the scholarly network, it concatenates self-descriptions with aggregated citation contexts. To achieve efficient candidate recall, we finetune an embedding model on these representations. Finally, we develop a reasoning-augmented reranker that exact interaction chains to construct explicit reasoning chains and finetunes a large language model to produce interpretable justifications and refined rankings. The dataset we curated covers 85\% of the datasets and baselines used at top AI conferences over the past five years. On our dataset, the proposed method outperforms the strongest prior baseline with average gains of +5.85\% in Recall@20, +8.30\% in HitRate@5. Taken together, our results advance reliable, interpretable automation of experimental design.

**arXiv ID:** 2511.04921
</details>

<details>
<summary><strong>Grounded Test-Time Adaptation for LLM Agents</strong> - Arthur Chen, Zuxin Liu, Jianguo Zhang, Akshara Prabhakar, Zhiwei Liu, Shelby Heinecke, Silvio Savarese, Victor Zhong, Caiming Xiong - [[pdf]](https://arxiv.org/pdf/2511.04847)</summary>

**Abstract:** Large language model (LLM)-based agents struggle to generalize to novel and complex environments, such as unseen websites or new sets of functions, due to a fundamental mismatch between their pre-training and test-time conditions. This challenge stems from two distinct failure modes: a syntactic misunderstanding of environment-specific components like observation formats, and a semantic misunderstanding of state-transition dynamics, which are only revealed at test time. To address these issues, we propose two distinct and complementary strategies for adapting LLM agents by leveraging environment-specific information available during deployment. First, an online distributional adaptation method parameterizes environmental nuances by learning a lightweight adaptation vector that biases the model's output distribution, enabling rapid alignment with an environment response format. Second, a deployment-time dynamics grounding method employs a persona-driven exploration phase to systematically probe and learn the environment's causal dynamics before task execution, equipping the agent with a nonparametric world model. We evaluate these strategies across diverse agentic benchmarks, including function calling and web navigation. Our empirical results show the effectiveness of both strategies across all benchmarks with minimal computational cost. We find that dynamics grounding is particularly effective in complex environments where unpredictable dynamics pose a major obstacle, demonstrating a robust path toward more generalizable and capable LLM-based agents. For example, on the WebArena multi-site split, this method increases the agent's success rate from 2% to 23%.

**arXiv ID:** 2511.04847
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (11 papers)</h2></summary>

<details>
<summary><strong>DeepForgeSeal: Latent Space-Driven Semi-Fragile Watermarking for Deepfake Detection Using Multi-Agent Adversarial Reinforcement Learning</strong> - Tharindu Fernando, Clinton Fookes, Sridha Sridharan - [[pdf]](https://arxiv.org/pdf/2511.04949)</summary>

**Abstract:** Rapid advances in generative AI have led to increasingly realistic deepfakes, posing growing challenges for law enforcement and public trust. Existing passive deepfake detectors struggle to keep pace, largely due to their dependence on specific forgery artifacts, which limits their ability to generalize to new deepfake types. Proactive deepfake detection using watermarks has emerged to address the challenge of identifying high-quality synthetic media. However, these methods often struggle to balance robustness against benign distortions with sensitivity to malicious tampering. This paper introduces a novel deep learning framework that harnesses high-dimensional latent space representations and the Multi-Agent Adversarial Reinforcement Learning (MAARL) paradigm to develop a robust and adaptive watermarking approach. Specifically, we develop a learnable watermark embedder that operates in the latent space, capturing high-level image semantics, while offering precise control over message encoding and extraction. The MAARL paradigm empowers the learnable watermarking agent to pursue an optimal balance between robustness and fragility by interacting with a dynamic curriculum of benign and malicious image manipulations simulated by an adversarial attacker agent. Comprehensive evaluations on the CelebA and CelebA-HQ benchmarks reveal that our method consistently outperforms state-of-the-art approaches, achieving improvements of over 4.5% on CelebA and more than 5.3% on CelebA-HQ under challenging manipulation scenarios.

**arXiv ID:** 2511.04949
</details>

<details>
<summary><strong>Multi-agent Coordination via Flow Matching</strong> - Dongsu Lee, Daehee Lee, Amy Zhang - [[pdf]](https://arxiv.org/pdf/2511.05005)</summary>

**Abstract:** This work presents MAC-Flow, a simple yet expressive framework for multi-agent coordination. We argue that requirements of effective coordination are twofold: (i) a rich representation of the diverse joint behaviors present in offline data and (ii) the ability to act efficiently in real time. However, prior approaches often sacrifice one for the other, i.e., denoising diffusion-based solutions capture complex coordination but are computationally slow, while Gaussian policy-based solutions are fast but brittle in handling multi-agent interaction. MAC-Flow addresses this trade-off by first learning a flow-based representation of joint behaviors, and then distilling it into decentralized one-step policies that preserve coordination while enabling fast execution. Across four different benchmarks, including $12$ environments and $34$ datasets, MAC-Flow alleviates the trade-off between performance and computational cost, specifically achieving about $\boldsymbol{\times14.5}$ faster inference compared to diffusion-based MARL methods, while maintaining good performance. At the same time, its inference speed is similar to that of prior Gaussian policy-based offline multi-agent reinforcement learning (MARL) methods.

**arXiv ID:** 2511.05005
</details>

<details>
<summary><strong>TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems</strong> - Ishan Kavathekar, Hemang Jain, Ameya Rathod, Ponnurangam Kumaraguru, Tanuja Ganu - [[pdf]](https://arxiv.org/pdf/2511.05269)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents through tool use, planning, and decision-making abilities, leading to their widespread adoption across diverse tasks. As task complexity grows, multi-agent LLM systems are increasingly used to solve problems collaboratively. However, safety and security of these systems remains largely under-explored. Existing benchmarks and datasets predominantly focus on single-agent settings, failing to capture the unique vulnerabilities of multi-agent dynamics and co-ordination. To address this gap, we introduce $\textbf{T}$hreats and $\textbf{A}$ttacks in $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{S}$ystems ($\textbf{TAMAS}$), a benchmark designed to evaluate the robustness and safety of multi-agent LLM systems. TAMAS includes five distinct scenarios comprising 300 adversarial instances across six attack types and 211 tools, along with 100 harmless tasks. We assess system performance across ten backbone LLMs and three agent interaction configurations from Autogen and CrewAI frameworks, highlighting critical challenges and failure modes in current multi-agent deployments. Furthermore, we introduce Effective Robustness Score (ERS) to assess the tradeoff between safety and task effectiveness of these frameworks. Our findings show that multi-agent systems are highly vulnerable to adversarial attacks, underscoring the urgent need for stronger defenses. TAMAS provides a foundation for systematically studying and improving the safety of multi-agent LLM systems.

**arXiv ID:** 2511.05269
</details>

<details>
<summary><strong>From Observability Data to Diagnosis: An Evolving Multi-agent System for Incident Management in Cloud Systems</strong> - Yu Luo, Jiamin Jiang, Jingfei Feng, Lei Tao, Qingliang Zhang, Xidao Wen, Yongqian Sun, Shenglin Zhang, Dan Pei - [[pdf]](https://arxiv.org/pdf/2510.24145)</summary>

**Abstract:** Incident management (IM) is central to the reliability of large-scale cloud systems. Yet manual IM, where on-call engineers examine metrics, logs, and traces is labor-intensive and error-prone in the face of massive and heterogeneous observability data. Existing automated IM approaches often struggle to generalize across systems, provide limited interpretability, and incur high deployment costs, which hinders adoption in practice. In this paper, we present OpsAgent, a lightweight, self-evolving multi-agent system for IM that employs a training-free data processor to convert heterogeneous observability data into structured textual descriptions, along with a multi-agent collaboration framework that makes diagnostic inference transparent and auditable. To support continual capability growth, OpsAgent also introduces a dual self-evolution mechanism that integrates internal model updates with external experience accumulation, thereby closing the deployment loop. Comprehensive experiments on the OPENRCA benchmark demonstrate state-of-the-art performance and show that OpsAgent is generalizable, interpretable, cost-efficient, and self-evolving, making it a practically deployable and sustainable solution for long-term operation in real-world cloud systems.

**arXiv ID:** 2510.24145
</details>

<details>
<summary><strong>Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment</strong> - Xubin Wang, Qing Li, Weijia Jia - [[pdf]](https://arxiv.org/pdf/2501.03265)</summary>

**Abstract:** This article surveys Cognitive Edge Computing as a practical and methodical pathway for deploying reasoning-capable Large Language Models (LLMs) and autonomous AI agents on resource-constrained devices at the network edge. We present a unified, cognition-preserving framework spanning: (1) model optimization (quantization, sparsity, low-rank adaptation, distillation) aimed at retaining multi-step reasoning under tight memory/compute budgets; (2) system architecture (on-device inference, elastic offloading, cloud-edge collaboration) that trades off latency, energy, privacy, and capacity; and (3) adaptive intelligence (context compression, dynamic routing, federated personalization) that tailors computation to task difficulty and device constraints. We synthesize advances in efficient Transformer design, multimodal integration, hardware-aware compilation, privacy-preserving learning, and agentic tool use, and map them to edge-specific operating envelopes. We further outline a standardized evaluation protocol covering latency, throughput, energy per token, accuracy, robustness, privacy, and sustainability, with explicit measurement assumptions to enhance comparability. Remaining challenges include modality-aware reasoning benchmarks, transparent and reproducible energy reporting, edge-oriented safety/alignment evaluation, and multi-agent testbeds. We conclude with practitioner guidelines for cross-layer co-design of algorithms, runtime, and hardware to deliver reliable, efficient, and privacy-preserving cognitive capabilities on edge devices.

**arXiv ID:** 2501.03265
</details>

<details>
<summary><strong>Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration</strong> - Songyuan Sui, Hongyi Liu, Serena Liu, Li Li, Soo-Hyun Choi, Rui Chen, Xia Hu - [[pdf]](https://arxiv.org/pdf/2508.15809)</summary>

**Abstract:** Table understanding requires structured, multi-step reasoning. Large Language Models (LLMs) struggle with it due to the structural complexity of tabular data. Recently, multi-agent frameworks for SQL generation have shown promise in tackling the challenges of understanding tabular data, but existing approaches often suffer from limitations such as the inability to comprehend table structure for reliable SQL generation, error propagation that results in invalid queries, and over-reliance on execution correctness. To address these issues, we propose Chain-of-Query (CoQ), a novel multi-agent framework for SQL-aided table understanding. CoQ adopts natural-language-style representations of table schemas to abstract away structural noise and enhance understanding. It employs a clause-by-clause SQL generation strategy to improve query quality and introduces a hybrid reasoning division that separates SQL-based mechanical reasoning from LLM-based logical inference, thereby reducing reliance on execution outcomes. Extensive experiments across four models and five widely used benchmarks demonstrate that CoQ achieves substantial accuracy improvements and significantly lowers invalid SQL rates compared to prior generic LLM-based, SQL-aided, and hybrid baselines, confirming its superior effectiveness in table understanding. The code is available at this https URL.

**arXiv ID:** 2508.15809
</details>

<details>
<summary><strong>Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics</strong> - Akshara Prabhakar, Roshan Ram, Zixiang Chen, Silvio Savarese, Frank Wang, Caiming Xiong, Huan Wang, Weiran Yao - [[pdf]](https://arxiv.org/pdf/2510.17797)</summary>

**Abstract:** As information grows exponentially, enterprises face increasing pressure to transform unstructured data into coherent, actionable insights. While autonomous agents show promise, they often struggle with domain-specific nuances, intent alignment, and enterprise integration. We present Enterprise Deep Research (EDR), a multi-agent system that integrates (1) a Master Planning Agent for adaptive query decomposition, (2) four specialized search agents (General, Academic, GitHub, LinkedIn), (3) an extensible MCP-based tool ecosystem supporting NL2SQL, file analysis, and enterprise workflows, (4) a Visualization Agent for data-driven insights, and (5) a reflection mechanism that detects knowledge gaps and updates research direction with optional human-in-the-loop steering guidance. These components enable automated report generation, real-time streaming, and seamless enterprise deployment, as validated on internal datasets. On open-ended benchmarks including DeepResearch Bench and DeepConsult, EDR outperforms state-of-the-art agentic systems without any human steering. We release the EDR framework and benchmark trajectories to advance research on multi-agent reasoning applications.
Code at this https URL and Dataset at this https URL

**arXiv ID:** 2510.17797
</details>

<details>
<summary><strong>Multi-Agent Craftax: Benchmarking Open-Ended Multi-Agent Reinforcement Learning at the Hyperscale</strong> - Bassel Al Omari, Michael Matthews, Alexander Rutherford, Jakob Nicolaus Foerster - [[pdf]](https://arxiv.org/pdf/2511.04904)</summary>

**Abstract:** Progress in multi-agent reinforcement learning (MARL) requires challenging benchmarks that assess the limits of current methods. However, existing benchmarks often target narrow short-horizon challenges that do not adequately stress the long-term dependencies and generalization capabilities inherent in many multi-agent systems. To address this, we first present \textit{Craftax-MA}: an extension of the popular open-ended RL environment, Craftax, that supports multiple agents and evaluates a wide range of general abilities within a single environment. Written in JAX, \textit{Craftax-MA} is exceptionally fast with a training run using 250 million environment interactions completing in under an hour. To provide a more compelling challenge for MARL, we also present \textit{Craftax-Coop}, an extension introducing heterogeneous agents, trading and more mechanics that require complex cooperation among agents for success. We provide analysis demonstrating that existing algorithms struggle with key challenges in this benchmark, including long-horizon credit assignment, exploration and cooperation, and argue for its potential to drive long-term research in MARL.

**arXiv ID:** 2511.04904
</details>

<details>
<summary><strong>ConVerse: Benchmarking Contextual Safety in Agent-to-Agent Conversations</strong> - Amr Gomaa, Ahmed Salem, Sahar Abdelnabi - [[pdf]](https://arxiv.org/pdf/2511.05359)</summary>

**Abstract:** As language models evolve into autonomous agents that act and communicate on behalf of users, ensuring safety in multi-agent ecosystems becomes a central challenge. Interactions between personal assistants and external service providers expose a core tension between utility and protection: effective collaboration requires information sharing, yet every exchange creates new attack surfaces. We introduce ConVerse, a dynamic benchmark for evaluating privacy and security risks in agent-agent interactions. ConVerse spans three practical domains (travel, real estate, insurance) with 12 user personas and over 864 contextually grounded attacks (611 privacy, 253 security). Unlike prior single-agent settings, it models autonomous, multi-turn agent-to-agent conversations where malicious requests are embedded within plausible discourse. Privacy is tested through a three-tier taxonomy assessing abstraction quality, while security attacks target tool use and preference manipulation. Evaluating seven state-of-the-art models reveals persistent vulnerabilities; privacy attacks succeed in up to 88% of cases and security breaches in up to 60%, with stronger models leaking more. By unifying privacy and security within interactive multi-agent contexts, ConVerse reframes safety as an emergent property of communication.

**arXiv ID:** 2511.05359
</details>

<details>
<summary><strong>Story Arena: A Multi-Agent Environment for Envisioning the Future of Software Engineering</strong> - Justin D. Weisz, Michael Muller, Kush R. Varshney - [[pdf]](https://arxiv.org/pdf/2511.05410)</summary>

**Abstract:** What better way to understand the impact of AI on software engineering than to ask AI itself? We constructed Story Arena, a multi-agent "writer's room" in which multiple AI agents, independently imbued with a position statement on the future of software engineering, converse with each other to develop a shared vision. They then use this shared vision to collaboratively construct a design fiction that depicts this vision in narrative form. We present "The Code of Trust," a short fiction that investigates themes of human comprehension, trust, content ownership, augmentation vs. replacement, and uncertain futures in human-AI co-creation.

**arXiv ID:** 2511.05410
</details>

<details>
<summary><strong>A Composable Agentic System for Automated Visual Data Reporting</strong> - Péter Ferenc Gyarmati, Dominik Moritz, Torsten Möller, Laura Koesten - [[pdf]](https://arxiv.org/pdf/2509.05721)</summary>

**Abstract:** To address the brittleness of monolithic AI agents, our prototype for automated visual data reporting explores a Human-AI Partnership model. Its hybrid, multi-agent architecture strategically externalizes logic from LLMs to deterministic modules, leveraging the rule-based system Draco for principled visualization design. The system delivers a dual-output: an interactive Observable report with Mosaic for reader exploration, and executable Marimo notebooks for deep, analyst-facing traceability. This granular architecture yields a fully automatic yet auditable and steerable system, charting a path toward a more synergistic partnership between human experts and AI. For reproducibility, our implementation and examples are available at this https URL.

**arXiv ID:** 2509.05721
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>ORCHID: Orchestrated Retrieval-Augmented Classification with Human-in-the-Loop Intelligent Decision-Making for High-Risk Property</strong> - Maria Mahbub, Vanessa Lama, Sanjay Das, Brian Starks, Christopher Polchek, Saffell Silvers, Lauren Deck, Prasanna Balaprakash, Tirthankar Ghosal - [[pdf]](https://arxiv.org/pdf/2511.04956)</summary>

**Abstract:** High-Risk Property (HRP) classification is critical at U.S. Department of Energy (DOE) sites, where inventories include sensitive and often dual-use equipment. Compliance must track evolving rules designated by various export control policies to make transparent and auditable decisions. Traditional expert-only workflows are time-consuming, backlog-prone, and struggle to keep pace with shifting regulatory boundaries. We demo ORCHID, a modular agentic system for HRP classification that pairs retrieval-augmented generation (RAG) with human oversight to produce policy-based outputs that can be audited. Small cooperating agents, retrieval, description refiner, classifier, validator, and feedback logger, coordinate via agent-to-agent messaging and invoke tools through the Model Context Protocol (MCP) for model-agnostic on-premise operation. The interface follows an Item to Evidence to Decision loop with step-by-step reasoning, on-policy citations, and append-only audit bundles (run-cards, prompts, evidence). In preliminary tests on real HRP cases, ORCHID improves accuracy and traceability over a non-agentic baseline while deferring uncertain items to Subject Matter Experts (SMEs). The demonstration shows single item submission, grounded citations, SME feedback capture, and exportable audit artifacts, illustrating a practical path to trustworthy LLM assistance in sensitive DOE compliance workflows.

**arXiv ID:** 2511.04956
</details>

<details>
<summary><strong>How Do AI Agents Do Human Work? Comparing AI and Human Workflows Across Diverse Occupations</strong> - Zora Zhiruo Wang, Yijia Shao, Omar Shaikh, Daniel Fried, Graham Neubig, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2510.22780)</summary>

**Abstract:** AI agents are continually optimized for tasks related to human work, such as software engineering and professional writing, signaling a pressing trend with significant impacts on the human workforce. However, these agent developments have often not been grounded in a clear understanding of how humans execute work, to reveal what expertise agents possess and the roles they can play in diverse workflows. In this work, we study how agents do human work by presenting the first direct comparison of human and agent workers across multiple essential work-related skills: data analysis, engineering, computation, writing, and design. To better understand and compare heterogeneous computer-use activities of workers, we introduce a scalable toolkit to induce interpretable, structured workflows from either human or agent computer-use activities. Using such induced workflows, we compare how humans and agents perform the same tasks and find that: (1) While agents exhibit promise in their alignment to human workflows, they take an overwhelmingly programmatic approach across all work domains, even for open-ended, visually dependent tasks like design, creating a contrast with the UI-centric methods typically used by humans. (2) Agents produce work of inferior quality, yet often mask their deficiencies via data fabrication and misuse of advanced tools. (3) Nonetheless, agents deliver results 88.3% faster and cost 90.4-96.2% less than humans, highlighting the potential for enabling efficient collaboration by delegating easily programmable tasks to agents.

**arXiv ID:** 2510.22780
</details>

<details>
<summary><strong>The Less Intelligent the Elements, the More Intelligent the Whole. Or, Possibly Not?</strong> - Guido Fioretti - [[pdf]](https://arxiv.org/pdf/2012.12689)</summary>

**Abstract:** The agent-based modelling community has a debate on how ``intelligent'' artificial agents should be, and in what ways their local intelligence relates to the emergence of a collective intelligence. I approach this debate by endowing the preys and predators of the Lotka-Volterra model with behavioral algorithms characterized by different levels of sophistication. The main finding is that by endowing both preys and predators with the capability of making predictions based on linear extrapolation a novel sort of dynamic equilibrium appears, where both species co-exist while both populations grow indefinitely. While this broadly confirms that, in general, relatively simple agents favor the emergence of complex collective behavior, it also suggests that one fundamental mechanism is that the capability of individuals to take first-order derivatives of one other's behavior can allow the collective computation of derivatives of any order.

**arXiv ID:** 2012.12689
</details>

<details>
<summary><strong>CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents</strong> - Zesen Liu, Zhixiang Zhang, Yuchong Xie, Dongdong She - [[pdf]](https://arxiv.org/pdf/2510.22963)</summary>

**Abstract:** LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.

**arXiv ID:** 2510.22963
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (19 papers)</h2></summary>

<details>
<summary><strong>Autonomous generation of different courses of action in mechanized combat operations</strong> - Johan Schubert, Patrik Hansen, Pontus Hörling, Ronnie Johansson - [[pdf]](https://arxiv.org/pdf/2511.05182)</summary>

**Abstract:** In this paper, we propose a methodology designed to support decision-making during the execution phase of military ground combat operations, with a focus on one's actions. This methodology generates and evaluates recommendations for various courses of action for a mechanized battalion, commencing with an initial set assessed by their anticipated outcomes. It systematically produces thousands of individual action alternatives, followed by evaluations aimed at identifying alternative courses of action with superior outcomes. These alternatives are appraised in light of the opponent's status and actions, considering unit composition, force ratios, types of offense and defense, and anticipated advance rates. Field manuals evaluate battle outcomes and advancement rates. The processes of generation and evaluation work concurrently, yielding a variety of alternative courses of action. This approach facilitates the management of new course generation based on previously evaluated actions. As the combat unfolds and conditions evolve, revised courses of action are formulated for the decision-maker within a sequential decision-making framework.

**arXiv ID:** 2511.05182
</details>

<details>
<summary><strong>Real-Time Reasoning Agents in Evolving Environments</strong> - Yule Wen, Yixin Ye, Yanzhe Zhang, Diyi Yang, Hao Zhu - [[pdf]](https://arxiv.org/pdf/2511.04898)</summary>

**Abstract:** Agents in the real world must make not only logical but also timely judgments. This requires continuous awareness of the dynamic environment: hazards emerge, opportunities arise, and other agents act, while the agent's reasoning is still unfolding. Despite advances in language model reasoning, existing approaches fail to account for this dynamic nature. We introduce real-time reasoning as a new problem formulation for agents in evolving environments and build Real-Time Reasoning Gym to demonstrate it. We study two paradigms for deploying language models in agents: (1) reactive agents, which employ language models with bounded reasoning computation for rapid responses, and (2) planning agents, which allow extended reasoning computation for complex problems. Our experiments show that even state-of-the-art models struggle with making logical and timely judgments in either paradigm. To address this limitation, we propose AgileThinker, which simultaneously engages both reasoning paradigms. AgileThinker consistently outperforms agents engaging only one reasoning paradigm as the task difficulty and time pressure rise, effectively balancing reasoning depth and response latency. Our work establishes real-time reasoning as a critical testbed for developing practical agents and provides a foundation for research in temporally constrained AI systems, highlighting a path toward real-time capable agents.

**arXiv ID:** 2511.04898
</details>

<details>
<summary><strong>Simulating Misinformation Vulnerabilities With Agent Personas</strong> - David Farr, Lynnette Hui Xian Ng, Stephen Prochaska, Iain J. Cruickshank, Jevin West - [[pdf]](https://arxiv.org/pdf/2511.04697)</summary>

**Abstract:** Disinformation campaigns can distort public perception and destabilize institutions. Understanding how different populations respond to information is crucial for designing effective interventions, yet real-world experimentation is impractical and ethically challenging. To address this, we develop an agent-based simulation using Large Language Models (LLMs) to model responses to misinformation. We construct agent personas spanning five professions and three mental schemas, and evaluate their reactions to news headlines. Our findings show that LLM-generated agents align closely with ground-truth labels and human predictions, supporting their use as proxies for studying information responses. We also find that mental schemas, more than professional background, influence how agents interpret misinformation. This work provides a validation of LLMs to be used as agents in an agent-based model of an information network for analyzing trust, polarization, and susceptibility to deceptive content in complex social systems.

**arXiv ID:** 2511.04697
</details>

<details>
<summary><strong>An End-to-End Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drones</strong> - Taihelong Zeng, Yun Lin, Yuhe Shi, Yan Li, Zhiqing Wei, Xuanru Ji - [[pdf]](https://arxiv.org/pdf/2511.05265)</summary>

**Abstract:** The emergence of truck-drone collaborative systems in last-mile logistics has positioned the Traveling Salesman Problem with Drones (TSP-D) as a pivotal extension of classical routing optimization, where synchronized vehicle coordination promises substantial operational efficiency and reduced environmental impact, yet introduces NP-hard combinatorial complexity beyond the reach of conventional optimization paradigms. Deep reinforcement learning offers a theoretically grounded framework to address TSP-D's inherent challenges through self-supervised policy learning and adaptive decision-making. This study proposes a hierarchical Actor-Critic deep reinforcement learning framework for solving the TSP-D problem. The architecture consists of two primary components: a Transformer-inspired encoder and an efficient Minimal Gated Unit decoder. The encoder incorporates a novel, optimized k-nearest neighbors sparse attention mechanism specifically for focusing on relevant spatial relationships, further enhanced by the integration of global node features. The Minimal Gated Unit decoder processes these encoded representations to efficiently generate solution sequences. The entire framework operates within an asynchronous advantage actor-critic paradigm. Experimental results show that, on benchmark TSP-D instances of various scales (N=10 to 100), the proposed model can obtain competitive or even superior solutions in shorter average computation times compared to high-performance heuristic algorithms and existing reinforcement learning methods. Moreover, compared to advanced reinforcement learning algorithm benchmarks, the proposed framework significantly reduces the total training time required while achieving superior final performance, highlighting its notable advantage in training efficiency.

**arXiv ID:** 2511.05265
</details>

<details>
<summary><strong>DeepEyesV2: Toward Agentic Multimodal Model</strong> - Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, Xing Yu - [[pdf]](https://arxiv.org/pdf/2511.05271)</summary>

**Abstract:** Agentic multimodal models should not only comprehend text and images, but also actively invoke external tools, such as code execution environments and web search, and integrate these operations into reasoning. In this work, we introduce DeepEyesV2 and explore how to build an agentic multimodal model from the perspectives of data construction, training methods, and model evaluation. We observe that direct reinforcement learning alone fails to induce robust tool-use behavior. This phenomenon motivates a two-stage training pipeline: a cold-start stage to establish tool-use patterns, and reinforcement learning stage to further refine tool invocation. We curate a diverse, moderately challenging training dataset, specifically including examples where tool use is beneficial. We further introduce RealX-Bench, a comprehensive benchmark designed to evaluate real-world multimodal reasoning, which inherently requires the integration of multiple capabilities, including perception, search, and reasoning. We evaluate DeepEyesV2 on RealX-Bench and other representative benchmarks, demonstrating its effectiveness across real-world understanding, mathematical reasoning, and search-intensive tasks. Moreover, DeepEyesV2 exhibits task-adaptive tool invocation, tending to use image operations for perception tasks and numerical computations for reasoning tasks. Reinforcement learning further enables complex tool combinations and allows model to selectively invoke tools based on context. We hope our study can provide guidance for community in developing agentic multimodal models.

**arXiv ID:** 2511.05271
</details>

<details>
<summary><strong>TeaRAG: A Token-Efficient Agentic Retrieval-Augmented Generation Framework</strong> - Chao Zhang, Yuhao Wang, Derong Xu, Haoxin Zhang, Yuanjie Lyu, Yuhao Chen, Shuochen Liu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2511.05385)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) utilizes external knowledge to augment Large Language Models' (LLMs) reliability. For flexibility, agentic RAG employs autonomous, multi-round retrieval and reasoning to resolve queries. Although recent agentic RAG has improved via reinforcement learning, they often incur substantial token overhead from search and reasoning processes. This trade-off prioritizes accuracy over efficiency. To address this issue, this work proposes TeaRAG, a token-efficient agentic RAG framework capable of compressing both retrieval content and reasoning steps. 1) First, the retrieved content is compressed by augmenting chunk-based semantic retrieval with a graph retrieval using concise triplets. A knowledge association graph is then built from semantic similarity and co-occurrence. Finally, Personalized PageRank is leveraged to highlight key knowledge within this graph, reducing the number of tokens per retrieval. 2) Besides, to reduce reasoning steps, Iterative Process-aware Direct Preference Optimization (IP-DPO) is proposed. Specifically, our reward function evaluates the knowledge sufficiency by a knowledge matching mechanism, while penalizing excessive reasoning steps. This design can produce high-quality preference-pair datasets, supporting iterative DPO to improve reasoning conciseness. Across six datasets, TeaRAG improves the average Exact Match by 4% and 2% while reducing output tokens by 61% and 59% on Llama3-8B-Instruct and Qwen2.5-14B-Instruct, respectively. Code is available at this https URL.

**arXiv ID:** 2511.05385
</details>

<details>
<summary><strong>Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction</strong> - Yiting He, Zhishuai Liu, Weixin Wang, Pan Xu - [[pdf]](https://arxiv.org/pdf/2511.05396)</summary>

**Abstract:** Off-dynamics reinforcement learning (RL), where training and deployment transition dynamics are different, can be formulated as learning in a robust Markov decision process (RMDP) where uncertainties in transition dynamics are imposed. Existing literature mostly assumes access to generative models allowing arbitrary state-action queries or pre-collected datasets with a good state coverage of the deployment environment, bypassing the challenge of exploration. In this work, we study a more realistic and challenging setting where the agent is limited to online interaction with the training environment. To capture the intrinsic difficulty of exploration in online RMDPs, we introduce the supremal visitation ratio, a novel quantity that measures the mismatch between the training dynamics and the deployment dynamics. We show that if this ratio is unbounded, online learning becomes exponentially hard. We propose the first computationally efficient algorithm that achieves sublinear regret in online RMDPs with $f$-divergence based transition uncertainties. We also establish matching regret lower bounds, demonstrating that our algorithm achieves optimal dependence on both the supremal visitation ratio and the number of interaction episodes. Finally, we validate our theoretical results through comprehensive numerical experiments.

**arXiv ID:** 2511.05396
</details>

<details>
<summary><strong>TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning</strong> - Junwen Pan, Qizhe Zhang, Rui Zhang, Ming Lu, Xin Wan, Yuan Zhang, Chang Liu, Qi She - [[pdf]](https://arxiv.org/pdf/2511.05489)</summary>

**Abstract:** Temporal search aims to identify a minimal set of relevant frames from tens of thousands based on a given query, serving as a foundation for accurate long-form video understanding. Existing works attempt to progressively narrow the search space. However, these approaches typically rely on a hand-crafted search process, lacking end-to-end optimization for learning optimal search strategies. In this paper, we propose TimeSearch-R, which reformulates temporal search as interleaved text-video thinking, seamlessly integrating searching video clips into the reasoning process through reinforcement learning (RL). However, applying RL training methods, such as Group Relative Policy Optimization (GRPO), to video reasoning can result in unsupervised intermediate search decisions. This leads to insufficient exploration of the video content and inconsistent logical reasoning. To address these issues, we introduce GRPO with Completeness Self-Verification (GRPO-CSV), which gathers searched video frames from the interleaved reasoning process and utilizes the same policy model to verify the adequacy of searched frames, thereby improving the completeness of video reasoning. Additionally, we construct datasets specifically designed for the SFT cold-start and RL training of GRPO-CSV, filtering out samples with weak temporal dependencies to enhance task difficulty and improve temporal search capabilities. Extensive experiments demonstrate that TimeSearch-R achieves significant improvements on temporal search benchmarks such as Haystack-LVBench and Haystack-Ego4D, as well as long-form video understanding benchmarks like VideoMME and MLVU. Notably, TimeSearch-R establishes a new state-of-the-art on LongVideoBench with 4.1% improvement over the base model Qwen2.5-VL and 2.0% over the advanced video reasoning model Video-R1. Our code is available at this https URL.

**arXiv ID:** 2511.05489
</details>

<details>
<summary><strong>Outbidding and Outbluffing Elite Humans: Mastering Liar's Poker via Self-Play and Reinforcement Learning</strong> - Richard Dewey, Janos Botyanszki, Ciamac C. Moallemi, Andrew T. Zheng - [[pdf]](https://arxiv.org/pdf/2511.03724)</summary>

**Abstract:** AI researchers have long focused on poker-like games as a testbed for environments characterized by multi-player dynamics, imperfect information, and reasoning under uncertainty. While recent breakthroughs have matched elite human play at no-limit Texas hold'em, the multi-player dynamics are subdued: most hands converge quickly with only two players engaged through multiple rounds of bidding. In this paper, we present Solly, the first AI agent to achieve elite human play in reduced-format Liar's Poker, a game characterized by extensive multi-player engagement. We trained Solly using self-play with a model-free, actor-critic, deep reinforcement learning algorithm. Solly played at an elite human level as measured by win rate (won over 50% of hands) and equity (money won) in heads-up and multi-player Liar's Poker. Solly also outperformed large language models (LLMs), including those with reasoning abilities, on the same metrics. Solly developed novel bidding strategies, randomized play effectively, and was not easily exploitable by world-class human players.

**arXiv ID:** 2511.03724
</details>

<details>
<summary><strong>Tactical Decision Making for Autonomous Trucks by Deep Reinforcement Learning with Total Cost of Operation Based Reward</strong> - Deepthi Pathare, Leo Laine, Morteza Haghir Chehreghani - [[pdf]](https://arxiv.org/pdf/2403.06524)</summary>

**Abstract:** We develop a deep reinforcement learning framework for tactical decision making in an autonomous truck, specifically for Adaptive Cruise Control (ACC) and lane change maneuvers in a highway scenario. Our results demonstrate that it is beneficial to separate high-level decision-making processes and low-level control actions between the reinforcement learning agent and the low-level controllers based on physical models. In the following, we study optimizing the performance with a realistic and multi-objective reward function based on Total Cost of Operation (TCOP) of the truck using different approaches; by adding weights to reward components, by normalizing the reward components and by using curriculum learning techniques.

**arXiv ID:** 2403.06524
</details>

<details>
<summary><strong>Ethics-Aware Safe Reinforcement Learning for Rare-Event Risk Control in Interactive Urban Driving</strong> - Dianzhao Li, Ostap Okhrin - [[pdf]](https://arxiv.org/pdf/2508.14926)</summary>

**Abstract:** Autonomous vehicles hold great promise for reducing traffic fatalities and improving transportation efficiency, yet their widespread adoption hinges on embedding credible and transparent ethical reasoning into routine and emergency maneuvers, particularly to protect vulnerable road users (VRUs) such as pedestrians and cyclists. Here, we present a hierarchical Safe Reinforcement Learning (Safe RL) framework that augments standard driving objectives with ethics-aware cost signals. At the decision level, a Safe RL agent is trained using a composite ethical risk cost, combining collision probability and harm severity, to generate high-level motion targets. A dynamic, risk-sensitive Prioritized Experience Replay mechanism amplifies learning from rare but critical, high-risk events. At the execution level, polynomial path planning coupled with Proportional-Integral-Derivative (PID) and Stanley controllers translates these targets into smooth, feasible trajectories, ensuring both accuracy and comfort. We train and validate our approach on closed-loop simulation environments derived from large-scale, real-world traffic datasets encompassing diverse vehicles, cyclists, and pedestrians, and demonstrate that it outperforms baseline methods in reducing risk to others while maintaining ego performance and comfort. This work provides a reproducible benchmark for Safe RL with explicitly ethics-aware objectives in human-mixed traffic scenarios. Our results highlight the potential of combining formal control theory and data-driven learning to advance ethically accountable autonomy that explicitly protects those most at risk in urban traffic environments. Across two interactive benchmarks and five random seeds, our policy decreases conflict frequency by 25-45% compared to matched task successes while maintaining comfort metrics within 5%.

**arXiv ID:** 2508.14926
</details>

<details>
<summary><strong>Explore Data Left Behind in Reinforcement Learning for Reasoning Language Models</strong> - Chenxi Liu, Junjie Liang, Yuqi Jia, Bochuan Cao, Yang Bai, Heng Huang, Xun Chen - [[pdf]](https://arxiv.org/pdf/2511.04800)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective approach for improving the reasoning abilities of large language models (LLMs). The Group Relative Policy Optimization (GRPO) family has demonstrated strong performance in training LLMs with RLVR. However, as models train longer and scale larger, more training prompts become residual prompts, those with zero variance rewards that provide no training signal. Consequently, fewer prompts contribute to training, reducing diversity and hindering effectiveness. To fully exploit these residual prompts, we propose the Explore Residual Prompts in Policy Optimization (ERPO) framework, which encourages exploration on residual prompts and reactivates their training signals. ERPO maintains a history tracker for each prompt and adaptively increases the sampling temperature for residual prompts that previously produced all correct responses. This encourages the model to generate more diverse reasoning traces, introducing incorrect responses that revive training signals. Empirical results on the Qwen2.5 series demonstrate that ERPO consistently surpasses strong baselines across multiple mathematical reasoning benchmarks.

**arXiv ID:** 2511.04800
</details>

<details>
<summary><strong>Minority-Aware Satisfaction Estimation in Dialogue Systems via Preference-Adaptive Reinforcement Learning</strong> - Yahui Fu, Zi Haur Pang, Tatsuya Kawahara - [[pdf]](https://arxiv.org/pdf/2511.05407)</summary>

**Abstract:** User satisfaction in dialogue systems is inherently subjective. When the same response strategy is applied across users, minority users may assign different satisfaction ratings than majority users due to variations in individual intents and preferences. However, existing alignment methods typically train one-size-fits-all models that aim for broad consensus, often overlooking minority perspectives and user-specific adaptation. We propose a unified framework that models both individual- and group-level preferences for user satisfaction estimation. First, we introduce Chain-of-Personalized-Reasoning (CoPeR) to capture individual preferences through interpretable reasoning chains. Second, we propose an expectation-maximization-based Majority-Minority Preference-Aware Clustering (M2PC) algorithm that discovers distinct user groups in an unsupervised manner to learn group-level preferences. Finally, we integrate these components into a preference-adaptive reinforcement learning framework (PAda-PPO) that jointly optimizes alignment with both individual and group preferences. Experiments on the Emotional Support Conversation dataset demonstrate consistent improvements in user satisfaction estimation, particularly for underrepresented user groups.

**arXiv ID:** 2511.05407
</details>

<details>
<summary><strong>Neural at ArchEHR-QA 2025: Agentic Prompt Optimization for Evidence-Grounded Clinical Question Answering</strong> - Sai Prasanna Teja Reddy Bogireddy, Abrar Majeedi, Viswanatha Reddy Gajjala, Zhuoyan Xu, Siddhant Rai, Vaishnav Potlapalli - [[pdf]](https://arxiv.org/pdf/2506.10751)</summary>

**Abstract:** Automated question answering (QA) over electronic health records (EHRs) can bridge critical information gaps for clinicians and patients, yet it demands both precise evidence retrieval and faithful answer generation under limited supervision. In this work, we present Neural, the runner-up in the BioNLP 2025 ArchEHR-QA shared task on evidence-grounded clinical QA. Our proposed method decouples the task into (1) sentence-level evidence identification and (2) answer synthesis with explicit citations. For each stage, we automatically explore the prompt space with DSPy's MIPROv2 optimizer, jointly tuning instructions and few-shot demonstrations on the development set. A self-consistency voting scheme further improves evidence recall without sacrificing precision. On the hidden test set, our method attains an overall score of 51.5, placing second stage while outperforming standard zero-shot and few-shot prompting by over 20 and 10 points, respectively. These results indicate that data-driven prompt optimization is a cost-effective alternative to model fine-tuning for high-stakes clinical QA, advancing the reliability of AI assistants in healthcare.

**arXiv ID:** 2506.10751
</details>

<details>
<summary><strong>Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward</strong> - Guanhua Huang, Tingqiang Xu, Mingze Wang, Qi Yi, Xue Gong, Siheng Li, Ruibin Xiong, Kejiao Li, Yuhao Jiang, Bo Zhou - [[pdf]](https://arxiv.org/pdf/2510.03222)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has propelled Large Language Models in complex reasoning, yet its scalability is often hindered by a training bottleneck where performance plateaus as policy entropy collapses, signaling a loss of exploration. Previous methods typically address this by maintaining high policy entropy, yet the precise mechanisms that govern meaningful exploration have remained underexplored. Our analysis suggests that an unselective focus on entropy risks amplifying irrelevant tokens and destabilizing training. This paper investigates the exploration dynamics within RLVR and identifies a key issue: the gradual elimination of valuable low-probability exploratory tokens, which we term \textbf{\textit{reasoning sparks}}. We find that while abundant in pre-trained models, these sparks are systematically extinguished during RLVR due to over-penalization, leading to a degeneracy in exploration. To address this, we introduce Low-probability Regularization (Lp-Reg). Its core mechanism regularizes the policy towards a heuristic proxy distribution. This proxy is constructed by filtering out presumed noise tokens and re-normalizing the distribution over the remaining candidates. The result is a less-noisy proxy where the probability of \textit{reasoning sparks} is amplified, which then serves as a soft regularization target to shield these valuable tokens from elimination via KL divergence. Experiments show that Lp-Reg enables stable on-policy RL, sustaining continuous scaling across $3,000$ training steps and $81,204$ GPU-hours, where baseline entropy-control methods collapse. This sustained exploration leads to state-of-the-art performance, achieving a $60.17\%$ average accuracy on five math benchmarks, an improvement of $2.66\%$ over prior methods. Code is available at this https URL.

**arXiv ID:** 2510.03222
</details>

<details>
<summary><strong>Quantum Boltzmann Machines for Sample-Efficient Reinforcement Learning</strong> - Thore Gerlach, Michael Schenk, Verena Kain - [[pdf]](https://arxiv.org/pdf/2511.04856)</summary>

**Abstract:** We introduce theoretically grounded Continuous Semi-Quantum Boltzmann Machines (CSQBMs) that supports continuous-action reinforcement learning. By combining exponential-family priors over visible units with quantum Boltzmann distributions over hidden units, CSQBMs yield a hybrid quantum-classical model that reduces qubit requirements while retaining strong expressiveness. Crucially, gradients with respect to continuous variables can be computed analytically, enabling direct integration into Actor-Critic algorithms. Building on this, we propose a continuous Q-learning framework that replaces global maximization by efficient sampling from the CSQBM distribution, thereby overcoming instability issues in continuous control.

**arXiv ID:** 2511.04856
</details>

<details>
<summary><strong>FoodRL: A Reinforcement Learning Ensembling Framework For In-Kind Food Donation Forecasting</strong> - Esha Sharma, Lauren Davis, Julie Ivy, Min Chi - [[pdf]](https://arxiv.org/pdf/2511.04865)</summary>

**Abstract:** Food banks are crucial for alleviating food insecurity, but their effectiveness hinges on accurately forecasting highly volatile in-kind donations to ensure equitable and efficient resource distribution. Traditional forecasting models often fail to maintain consistent accuracy due to unpredictable fluctuations and concept drift driven by seasonal variations and natural disasters such as hurricanes in the Southeastern U.S. and wildfires in the West Coast. To address these challenges, we propose FoodRL, a novel reinforcement learning (RL) based metalearning framework that clusters and dynamically weights diverse forecasting models based on recent performance and contextual information. Evaluated on multi-year data from two structurally distinct U.S. food banks-one large regional West Coast food bank affected by wildfires and another state-level East Coast food bank consistently impacted by hurricanes, FoodRL consistently outperforms baseline methods, particularly during periods of disruption or decline. By delivering more reliable and adaptive forecasts, FoodRL can facilitate the redistribution of food equivalent to 1.7 million additional meals annually, demonstrating its significant potential for social impact as well as adaptive ensemble learning for humanitarian supply chains.

**arXiv ID:** 2511.04865
</details>

<details>
<summary><strong>Self-Interest and Systemic Benefits: Emergence of Collective Rationality in Mixed Autonomy Traffic Through Deep Reinforcement Learning</strong> - Di Chen, Jia Li, Michael Zhang - [[pdf]](https://arxiv.org/pdf/2511.04883)</summary>

**Abstract:** Autonomous vehicles (AVs) are expected to be commercially available in the near future, leading to mixed autonomy traffic consisting of both AVs and human-driven vehicles (HVs). Although numerous studies have shown that AVs can be deployed to benefit the overall traffic system performance by incorporating system-level goals into their decision making, it is not clear whether the benefits still exist when agents act out of self-interest -- a trait common to all driving agents, both human and autonomous. This study aims to understand whether self-interested AVs can bring benefits to all driving agents in mixed autonomy traffic systems. The research is centered on the concept of collective rationality (CR). This concept, originating from game theory and behavioral economics, means that driving agents may cooperate collectively even when pursuing individual interests. Our recent research has proven the existence of CR in an analytical game-theoretical model and empirically in mixed human-driven traffic. In this paper, we demonstrate that CR can be attained among driving agents trained using deep reinforcement learning (DRL) with a simple reward design. We examine the extent to which self-interested traffic agents can achieve CR without directly incorporating system-level objectives. Results show that CR consistently emerges in various scenarios, which indicates the robustness of this property. We also postulate a mechanism to explain the emergence of CR in the microscopic and dynamic environment and verify it based on simulation evidence. This research suggests the possibility of leveraging advanced learning methods (such as federated learning) to achieve collective cooperation among self-interested driving agents in mixed-autonomy systems.

**arXiv ID:** 2511.04883
</details>

<details>
<summary><strong>Diverse Mini-Batch Selection in Reinforcement Learning for Efficient Chemical Exploration in de novo Drug Design</strong> - Hampus Gummesson Svensson, Ola Engkvist, Jon Paul Janet, Christian Tyrchan, Morteza Haghir Chehreghani - [[pdf]](https://arxiv.org/pdf/2506.21158)</summary>

**Abstract:** In many real-world applications, evaluating the quality of instances is costly and time-consuming, e.g., human feedback and physics simulations, in contrast to proposing new instances. In particular, this is even more critical in reinforcement learning, since it relies on interactions with the environment (i.e., new instances) that must be evaluated to provide a reward signal for learning. At the same time, performing sufficient exploration is crucial in reinforcement learning to find high-rewarding solutions, meaning that the agent should observe and learn from a diverse set of experiences to find different solutions. Thus, we argue that learning from a diverse mini-batch of experiences can have a large impact on the exploration and help mitigate mode this http URL this paper, we introduce mini-batch diversification for reinforcement learning and study this framework in the context of a real-world problem, namely, drug discovery. We extensively evaluate how our proposed framework can enhance the effectiveness of chemical exploration in de novo drug design, where finding diverse and high-quality solutions is crucial. Our experiments demonstrate that our proposed diverse mini-batch selection framework can substantially enhance the diversity of solutions while maintaining high-quality solutions. In drug discovery, such an outcome can potentially lead to fulfilling unmet medical needs faster.

**arXiv ID:** 2506.21158
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
