# Agent arXiv Daily

**Last Updated:** 2025-10-30 02:48:56

**Total Papers:** 68

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (6 papers)</h2></summary>

<details>
<summary><strong>GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning</strong> - Jiaqi Wu, Qinlao Zhao, Zefeng Chen, Kai Qin, Yifei Zhao, Xueqian Wang, Yuhang Yao - [[pdf]](https://arxiv.org/pdf/2510.25320)</summary>

**Abstract:** Autonomous agents powered by large language models (LLMs) have shown impressive capabilities in tool manipulation for complex task-solving. However, existing paradigms such as ReAct rely on sequential reasoning and execution, failing to exploit the inherent parallelism among independent sub-tasks. This sequential bottleneck leads to inefficient tool utilization and suboptimal performance in multi-step reasoning scenarios. We introduce Graph-based Agent Planning (GAP), a novel framework that explicitly models inter-task dependencies through graph-based planning to enable adaptive parallel and serial tool execution. Our approach trains agent foundation models to decompose complex tasks into dependency-aware sub-task graphs, autonomously determining which tools can be executed in parallel and which must follow sequential dependencies. This dependency-aware orchestration achieves substantial improvements in both execution efficiency and task accuracy. To train GAP, we construct a high-quality dataset of graph-based planning traces derived from the Multi-Hop Question Answering (MHQA) benchmark. We employ a two-stage training strategy: supervised fine-tuning (SFT) on the curated dataset, followed by reinforcement learning (RL) with a correctness-based reward function on strategically sampled queries where tool-based reasoning provides maximum value. Experimental results on MHQA datasets demonstrate that GAP significantly outperforms traditional ReAct baselines, particularly on multi-step retrieval tasks, while achieving dramatic improvements in tool invocation efficiency through intelligent parallelization. The project page is available at: this https URL.

**arXiv ID:** 2510.25320
</details>

<details>
<summary><strong>Cardi-GPT: An Expert ECG-Record Processing Chatbot</strong> - Koustav Mallick, Neel Singh, Mohammedreza Hajiarbabi - [[pdf]](https://arxiv.org/pdf/2510.24737)</summary>

**Abstract:** Interpreting and communicating electrocardiogram (ECG) findings are crucial yet challenging tasks in cardiovascular diagnosis, traditionally requiring significant expertise and precise clinical communication. This paper introduces Cardi-GPT, an advanced expert system designed to streamline ECG interpretation and enhance clinical communication through deep learning and natural language interaction. Cardi-GPT employs a 16-residual-block convolutional neural network (CNN) to process 12-lead ECG data, achieving a weighted accuracy of 0.6194 across 24 cardiac conditions. A novel fuzzification layer converts complex numerical outputs into clinically meaningful linguistic categories, while an integrated chatbot interface facilitates intuitive exploration of diagnostic insights and seamless communication between healthcare providers.
The system was evaluated on a diverse dataset spanning six hospitals across four countries, demonstrating superior performance compared to baseline models. Additionally, Cardi-GPT achieved an impressive overall response quality score of 73\%, assessed using a comprehensive evaluation framework that measures coverage, grounding, and coherence. By bridging the gap between intricate ECG data interpretation and actionable clinical insights, Cardi-GPT represents a transformative innovation in cardiovascular healthcare, promising to improve diagnostic accuracy, clinical workflows, and patient outcomes across diverse medical settings.

**arXiv ID:** 2510.24737
</details>

<details>
<summary><strong>Do Chatbots Walk the Talk of Responsible AI?</strong> - Susan Ariel Aaronson, Michael Moreno - [[pdf]](https://arxiv.org/pdf/2510.24823)</summary>

**Abstract:** This study examines whether leading AI chatbot companies implement the responsible AI principles they publicly advocate. The authors used a mixed-methods approach analyzing four major chatbots (ChatGPT, Gemini, DeepSeek, and Grok) across company websites, technical documentation, and direct chatbot evaluations. We found significant gaps between corporate rhetoric and practice.

**arXiv ID:** 2510.24823
</details>

<details>
<summary><strong>SCOUT: A Lightweight Framework for Scenario Coverage Assessment in Autonomous Driving</strong> - Anil Yildiz, Sarah M. Thornton, Carl Hildebrandt, Sreeja Roy-Singh, Mykel J. Kochenderfer - [[pdf]](https://arxiv.org/pdf/2510.24949)</summary>

**Abstract:** Assessing scenario coverage is crucial for evaluating the robustness of autonomous agents, yet existing methods rely on expensive human annotations or computationally intensive Large Vision-Language Models (LVLMs). These approaches are impractical for large-scale deployment due to cost and efficiency constraints. To address these shortcomings, we propose SCOUT (Scenario Coverage Oversight and Understanding Tool), a lightweight surrogate model designed to predict scenario coverage labels directly from an agent's latent sensor representations. SCOUT is trained through a distillation process, learning to approximate LVLM-generated coverage labels while eliminating the need for continuous LVLM inference or human annotation. By leveraging precomputed perception features, SCOUT avoids redundant computations and enables fast, scalable scenario coverage estimation. We evaluate our method across a large dataset of real-life autonomous navigation scenarios, demonstrating that it maintains high accuracy while significantly reducing computational cost. Our results show that SCOUT provides an effective and practical alternative for large-scale coverage analysis. While its performance depends on the quality of LVLM-generated training labels, SCOUT represents a major step toward efficient scenario coverage oversight in autonomous systems.

**arXiv ID:** 2510.24949
</details>

<details>
<summary><strong>CGM-Led Multimodal Tracking with Chatbot Support: An Autoethnography in Sub-Health</strong> - Dongyijie Primo Pan, Lan Luo, Yike Wang, Pan Hui - [[pdf]](https://arxiv.org/pdf/2510.25381)</summary>

**Abstract:** Metabolic disorders present a pressing global health challenge, with China carrying the world's largest burden. While continuous glucose monitoring (CGM) has transformed diabetes care, its potential for supporting sub-health populations -- such as individuals who are overweight, prediabetic, or anxious -- remains underexplored. At the same time, large language models (LLMs) are increasingly used in health coaching, yet CGM is rarely incorporated as a first-class signal. To address this gap, we conducted a six-week autoethnography, combining CGM with multimodal indicators captured via common digital devices and a chatbot that offered personalized reflections and explanations of glucose fluctuations. Our findings show how CGM-led, data-first multimodal tracking, coupled with conversational support, shaped everyday practices of diet, activity, stress, and wellbeing. This work contributes to HCI by extending CGM research beyond clinical diabetes and demonstrating how LLM-driven agents can support preventive health and reflection in at-risk populations.

**arXiv ID:** 2510.25381
</details>

<details>
<summary><strong>Small Talk, Big Impact? LLM-based Conversational Agents to Mitigate Passive Fatigue in Conditional Automated Driving</strong> - Lewis Cockram, Yueteng Yu, Jorge Pardo, Xiaomeng Li, Andry Rakotonirainy, Jonny Kuo, Sebastien Demmel, Mike Lenné, Ronald Schroeter - [[pdf]](https://arxiv.org/pdf/2510.25421)</summary>

**Abstract:** Passive fatigue during conditional automated driving can compromise driver readiness and safety. This paper presents findings from a test-track study with 40 participants in a real-world rural automated driving scenario. In this scenario, a Large Language Model (LLM) based conversational agent (CA) was designed to check in with drivers and re-engage them with their surroundings. Drawing on in-car video recordings, sleepiness ratings and interviews, we analysed how drivers interacted with the agent and how these interactions shaped alertness. Users found the CA helpful for supporting vigilance during passive fatigue. Thematic analysis of acceptability further revealed three user preference profiles that implicate future intention to use CAs. Positioning empirically observed profiles within existing CA archetype frameworks highlights the need for adaptive design sensitive to diverse user groups. This work underscores the potential of CAs as proactive Human-Machine Interface (HMI) interventions, demonstrating how natural language can support context-aware interaction during automated driving.

**arXiv ID:** 2510.25421
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>TheraMind: A Strategic and Adaptive Agent for Longitudinal Psychological Counseling</strong> - He Hu, Yucheng Zhou, Chiyuan Ma, Qianning Wang, Zheng Zhang, Fei Ma, Laizhong Cui, Qi Tian - [[pdf]](https://arxiv.org/pdf/2510.25758)</summary>

**Abstract:** Large language models (LLMs) in psychological counseling have attracted increasing attention. However, existing approaches often lack emotional understanding, adaptive strategies, and the use of therapeutic methods across multiple sessions with long-term memory, leaving them far from real clinical practice. To address these critical gaps, we introduce TheraMind, a strategic and adaptive agent for longitudinal psychological counseling. The cornerstone of TheraMind is a novel dual-loop architecture that decouples the complex counseling process into an Intra-Session Loop for tactical dialogue management and a Cross-Session Loop for strategic therapeutic planning. The Intra-Session Loop perceives the patient's emotional state to dynamically select response strategies while leveraging cross-session memory to ensure continuity. Crucially, the Cross-Session Loop empowers the agent with long-term adaptability by evaluating the efficacy of the applied therapy after each session and adjusting the method for subsequent interactions. We validate our approach in a high-fidelity simulation environment grounded in real clinical cases. Extensive evaluations show that TheraMind outperforms other methods, especially on multi-session metrics like Coherence, Flexibility, and Therapeutic Attunement, validating the effectiveness of its dual-loop design in emulating strategic, adaptive, and longitudinal therapeutic behavior. The code is publicly available at this https URL.

**arXiv ID:** 2510.25758
</details>

<details>
<summary><strong>Process-Level Trajectory Evaluation for Environment Configuration in Software Engineering Agents</strong> - Jiayi Kuang, Yinghui Li, Xin Zhang, Yangning Li, Di Yin, Xing Sun, Ying Shen, Philip S. Yu - [[pdf]](https://arxiv.org/pdf/2510.25694)</summary>

**Abstract:** Large language model-based agents show promise for software engineering, but environment configuration remains a bottleneck due to heavy manual effort and scarce large-scale, high-quality datasets. Existing benchmarks assess only end-to-end build/test success, obscuring where and why agents succeed or fail. We introduce the Environment Configuration Diagnosis Benchmark, Enconda-bench, which provides process-level trajectory assessment of fine-grained agent capabilities during environment setup-planning, perception-driven error diagnosis, feedback-driven repair, and action to execute final environment configuration. Our task instances are automatically constructed by injecting realistic README errors and are validated in Docker for scalable, high-quality evaluation. Enconda-bench combines process-level analysis with end-to-end executability to enable capability assessments beyond aggregate success rates. Evaluations across state-of-the-art LLMs and agent frameworks show that while agents can localize errors, they struggle to translate feedback into effective corrections, limiting end-to-end performance. To our knowledge, Enconda-bench is the first framework to provide process-level internal capability assessment for environment configuration, offering actionable insights for improving software engineering agents.

**arXiv ID:** 2510.25694
</details>

<details>
<summary><strong>The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution</strong> - Junlong Li, Wenshuo Zhao, Jian Zhao, Weihao Zeng, Haoze Wu, Xiaochen Wang, Rui Ge, Yuxuan Cao, Yuzhen Huang, Wei Liu, Junteng Liu, Zhaochen Su, Yiyang Guo, Fan Zhou, Lueyang Zhang, Juan Michelini, Xingyao Wang, Xiang Yue, Shuyan Zhou, Graham Neubig, Junxian He - [[pdf]](https://arxiv.org/pdf/2510.25726)</summary>

**Abstract:** Real-world language agents must handle complex, multi-step workflows across diverse Apps. For instance, an agent may manage emails by coordinating with calendars and file systems, or monitor a production database to detect anomalies and generate reports following an operating manual. However, existing language agent benchmarks often focus on narrow domains or simplified tasks that lack the diversity, realism, and long-horizon complexity required to evaluate agents' real-world performance. To address this gap, we introduce the Tool Decathlon (dubbed as Toolathlon), a benchmark for language agents offering diverse Apps and tools, realistic environment setup, and reliable execution-based evaluation. Toolathlon spans 32 software applications and 604 tools, ranging from everyday platforms such as Google Calendar and Notion to professional ones like WooCommerce, Kubernetes, and BigQuery. Most of the tools are based on a high-quality set of Model Context Protocol (MCP) servers that we may have revised or implemented ourselves. Unlike prior works, which primarily ensure functional realism but offer limited environment state diversity, we provide realistic initial environment states from real software, such as Canvas courses with dozens of students or real financial spreadsheets. This benchmark includes 108 manually sourced or crafted tasks in total, requiring interacting with multiple Apps over around 20 turns on average to complete. Each task is strictly verifiable through dedicated evaluation scripts. Comprehensive evaluation of SOTA models highlights their significant shortcomings: the best-performing model, Claude-4.5-Sonnet, achieves only a 38.6% success rate with 20.2 tool calling turns on average, while the top open-weights model DeepSeek-V3.2-Exp reaches 20.1%. We expect Toolathlon to drive the development of more capable language agents for real-world, long-horizon task execution.

**arXiv ID:** 2510.25726
</details>

<details>
<summary><strong>Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine</strong> - Wenyi Wang, Piotr Piękos, Li Nanbo, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, Jürgen Schmidhuber - [[pdf]](https://arxiv.org/pdf/2510.21614)</summary>

**Abstract:** Recent studies operationalize self-improvement through coding agents that edit their own codebases. They grow a tree of self-modifications through expansion strategies that favor higher software engineering benchmark performance, assuming that this implies more promising subsequent self-modifications. However, we identify a mismatch between the agent's self-improvement potential (metaproductivity) and its coding benchmark performance, namely the Metaproductivity-Performance Mismatch. Inspired by Huxley's concept of clade, we propose a metric ($\mathrm{CMP}$) that aggregates the benchmark performances of the descendants of an agent as an indicator of its potential for self-improvement. We show that, in our self-improving coding agent development setting, access to the true $\mathrm{CMP}$ is sufficient to simulate how the Gödel Machine would behave under certain assumptions. We introduce the Huxley-Gödel Machine (HGM), which, by estimating $\mathrm{CMP}$ and using it as guidance, searches the tree of self-modifications. On SWE-bench Verified and Polyglot, HGM outperforms prior self-improving coding agent development methods while using fewer allocated CPU hours. Last but not least, HGM demonstrates strong transfer to other coding datasets and large language models. The agent optimized by HGM on SWE-bench Verified with GPT-5-mini and evaluated on SWE-bench Lite with GPT-5 achieves human-level performance, matching the best officially checked results of human-engineered coding agents. Our code is publicly available at this https URL.

**arXiv ID:** 2510.21614
</details>

<details>
<summary><strong>Evaluation of Safety Cognition Capability in Vision-Language Models for Autonomous Driving</strong> - Enming Zhang, Peizhe Gong, Xingyuan Dai, Min Huang, Yisheng Lv, Qinghai Miao - [[pdf]](https://arxiv.org/pdf/2503.06497)</summary>

**Abstract:** Ensuring the safety of vision-language models (VLMs) in autonomous driving systems is of paramount importance, yet existing research has largely focused on conventional benchmarks rather than safety-critical evaluation. In this work, we present SCD-Bench (Safety Cognition Driving Benchmark) a novel framework specifically designed to assess the safety cognition capabilities of VLMs within interactive driving scenarios. To address the scalability challenge of data annotation, we introduce ADA (Autonomous Driving Annotation), a semi-automated labeling system, further refined through expert review by professionals with domain-specific knowledge in autonomous driving. To facilitate scalable and consistent evaluation, we also propose an automated assessment pipeline leveraging large language models, which demonstrates over 98% agreement with human expert judgments. In addressing the broader challenge of aligning VLMs with safety cognition in driving environments, we construct SCD-Training, the first large-scale dataset tailored for this task, comprising 324.35K high-quality samples. Through extensive experiments, we show that models trained on SCD-Training exhibit marked improvements not only on SCD-Bench, but also on general and domain-specific benchmarks, offering a new perspective on enhancing safety-aware interactions in vision-language systems for autonomous driving.

**arXiv ID:** 2503.06497
</details>

<details>
<summary><strong>SeeingEye: Agentic Information Flow Unlocks Multimodal Reasoning In Text-only LLMs</strong> - Weijia Zhang, Zijia Liu, Haoru Li, Haoqi Chen, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2510.25092)</summary>

**Abstract:** Recent advances in text-only large language models (LLMs), such as DeepSeek-R1, demonstrate remarkable reasoning ability. However, these models remain fragile or entirely incapable when extended to multi-modal tasks. Existing approaches largely rely on single-form captions, which lack diversity and often fail to adapt across different types of Visual Question Answering (VQA) benchmarks. As a result, they provide no principled or efficient channel for transmitting fine-grained visual information. We introduce Seeing Eye, a modular framework that unlocks multimodal reasoning in text-only LLMs through an agent-based small VLM translator. This translator acts as a perception agent: it can invoke specialized tools (e.g., OCR and crop) and iteratively distill multimodal inputs into structured intermediate representations (SIRs) tailored to the question. These SIRs are then passed to the text-only LLM, which serves as a reasoning agent. Crucially, the translator and reasoner engage in multi-round feedback and interaction, enabling the extraction of targeted visual details and yielding more confident answers. Experiments on knowledge-intensive VQA benchmarks, including MMMU and MIA-Bench, demonstrate that Seeing Eye not only reduces inference cost but also surpasses much larger end-to-end VLMs. For example, an instantiation combining a 3B-parameter vision translator with an 8B-parameter language reasoner outperforms a monolithic 32B VLM on challenging knowledge-based questions. Our results highlight that decoupling perception from reasoning via agent information flow offers a scalable and plug-and-play pathway to multimodal reasoning, allowing strong text-only LLMs to fully leverage their reasoning capabilities. Code is available at: this https URL

**arXiv ID:** 2510.25092
</details>

<details>
<summary><strong>CRMWeaver: Building Powerful Business Agent via Agentic RL and Shared Memories</strong> - Yilong Lai, Yipin Yang, Jialong Wu, Fengran Mo, Zhenglin Wang, Ting Liang, Jianguo Lin, Keping Yang - [[pdf]](https://arxiv.org/pdf/2510.25333)</summary>

**Abstract:** Recent years have witnessed the rapid development of LLM-based agents, which shed light on using language agents to solve complex real-world problems. A prominent application lies in business agents, which interact with databases and internal knowledge bases via tool calls to fulfill diverse user requirements. However, this domain is characterized by intricate data relationships and a wide range of heterogeneous tasks, from statistical data queries to knowledge-based question-answering. To address these challenges, we propose CRMWeaver, a novel approach that enhances business agents in such complex settings. To acclimate the agentic model to intricate business environments, we employ a synthesis data generation and RL-based paradigm during training, which significantly improves the model's ability to handle complex data and varied tasks. During inference, a shared memories mechanism is introduced, prompting the agent to learn from task guidelines in similar problems, thereby further boosting its effectiveness and generalization, especially in unseen scenarios. We validate the efficacy of our approach on the CRMArena-Pro dataset, where our lightweight model achieves competitive results in both B2B and B2C business scenarios, underscoring its practical value for real-world applications.

**arXiv ID:** 2510.25333
</details>

<details>
<summary><strong>UrduFactCheck: An Agentic Fact-Checking Framework for Urdu with Evidence Boosting and Benchmarking</strong> - Sarfraz Ahmad, Hasan Iqbal, Momina Ahsan, Numaan Naeem, Muhammad Ahsan Riaz Khan, Arham Riaz, Muhammad Arslan Manzoor, Yuxia Wang, Preslav Nakov - [[pdf]](https://arxiv.org/pdf/2505.15063)</summary>

**Abstract:** The rapid adoption of Large Language Models (LLMs) has raised important concerns about the factual reliability of their outputs, particularly in low-resource languages such as Urdu. Existing automated fact-checking systems are predominantly developed for English, leaving a significant gap for the more than 200 million Urdu speakers worldwide. In this work, we present UrduFactBench and UrduFactQA, two novel hand-annotated benchmarks designed to enable fact-checking and factual consistency evaluation in Urdu. While UrduFactBench focuses on claim verification, UrduFactQA targets the factuality of LLMs in question answering. These resources, the first of their kind for Urdu, were developed through a multi-stage annotation process involving native Urdu speakers. To complement these benchmarks, we introduce UrduFactCheck, a modular fact-checking framework that incorporates both monolingual and translation-based evidence retrieval strategies to mitigate the scarcity of high-quality Urdu evidence. Leveraging these resources, we conduct an extensive evaluation of twelve LLMs and demonstrate that translation-augmented pipelines consistently enhance performance compared to monolingual ones. Our findings reveal persistent challenges for open-source LLMs in Urdu and underscore the importance of developing targeted resources. All code and data are publicly available at this https URL.

**arXiv ID:** 2505.15063
</details>

<details>
<summary><strong>OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents</strong> - Thomas Kuntz, Agatha Duzan, Hao Zhao, Francesco Croce, Zico Kolter, Nicolas Flammarion, Maksym Andriushchenko - [[pdf]](https://arxiv.org/pdf/2506.14866)</summary>

**Abstract:** Computer use agents are LLM-based agents that can directly interact with a graphical user interface, by processing screenshots or accessibility trees. While these systems are gaining popularity, their safety has been largely overlooked, despite the fact that evaluating and understanding their potential for harmful behavior is essential for widespread adoption. To address this gap, we introduce OS-Harm, a new benchmark for measuring safety of computer use agents. OS-Harm is built on top of the OSWorld environment and aims to test models across three categories of harm: deliberate user misuse, prompt injection attacks, and model misbehavior. To cover these cases, we create 150 tasks that span several types of safety violations (harassment, copyright infringement, disinformation, data exfiltration, etc.) and require the agent to interact with a variety of OS applications (email client, code editor, browser, etc.). Moreover, we propose an automated judge to evaluate both accuracy and safety of agents that achieves high agreement with human annotations (0.76 and 0.79 F1 score). We evaluate computer use agents based on a range of frontier models - such as o4-mini, Claude 3.7 Sonnet, Gemini 2.5 Pro - and provide insights into their safety. In particular, all models tend to directly comply with many deliberate misuse queries, are relatively vulnerable to static prompt injections, and occasionally perform unsafe actions. The OS-Harm benchmark is available at this https URL.

**arXiv ID:** 2506.14866
</details>

<details>
<summary><strong>Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles</strong> - Ian Nell, Shane Gilroy - [[pdf]](https://arxiv.org/pdf/2509.09349)</summary>

**Abstract:** Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behaviour classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviours such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioural analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions.

**arXiv ID:** 2509.09349
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>From Narrative to Action: A Hierarchical LLM-Agent Framework for Human Mobility Generation</strong> - Qiumeng Li, Chunhou Ji, Xinyue Liu - [[pdf]](https://arxiv.org/pdf/2510.24802)</summary>

**Abstract:** Understanding and replicating human mobility requires not only spatial-temporal accuracy but also an awareness of the cognitive hierarchy underlying real-world travel decisions. Traditional agent-based or deep learning models can reproduce statistical patterns of movement but fail to capture the semantic coherence and causal logic of human behavior. Large language models (LLMs) show potential, but struggle to balance creative reasoning with strict structural compliance. This study proposes a Hierarchical LLM-Agent Framework, termed Narrative-to-Action, that integrates high-level narrative reasoning, mid-level reflective planning, and low-level behavioral execution within a unified cognitive hierarchy. At the macro level, one agent is employed as a "creative writer" to produce diary-style narratives rich in motivation and context, then uses another agent as a "structural parser" to convert narratives into machine-readable plans. A dynamic execution module further grounds agents in geographic environments and enables adaptive behavioral adjustments guided by a novel occupation-aware metric, Mobility Entropy by Occupation (MEO), which captures heterogeneous schedule flexibility across different occupational personalities. At the micro level, the agent executes concrete actions-selecting locations, transportation modes, and time intervals-through interaction with an environmental simulation. By embedding this multi-layer cognitive process, the framework produces not only synthetic trajectories that align closely with real-world patterns but also interpretable representations of human decision logic. This research advances synthetic mobility generation from a data-driven paradigm to a cognition-driven simulation, providing a scalable pathway for understanding, predicting, and synthesizing complex urban mobility behaviors through hierarchical LLM agents.

**arXiv ID:** 2510.24802
</details>

<details>
<summary><strong>StorageXTuner: An LLM Agent-Driven Automatic Tuning Framework for Heterogeneous Storage Systems</strong> - Qi Lin, Zhenyu Zhang, Viraj Thakkar, Zhenjie Sun, Mai Zheng, Zhichao Cao - [[pdf]](https://arxiv.org/pdf/2510.25017)</summary>

**Abstract:** Automatically configuring storage systems is hard: parameter spaces are large and conditions vary across workloads, deployments, and versions. Heuristic and ML tuners are often system specific, require manual glue, and degrade under changes. Recent LLM-based approaches help but usually treat tuning as a single-shot, system-specific task, which limits cross-system reuse, constrains exploration, and weakens validation. We present StorageXTuner, an LLM agent-driven auto-tuning framework for heterogeneous storage engines. StorageXTuner separates concerns across four agents - Executor (sandboxed benchmarking), Extractor (performance digest), Searcher (insight-guided configuration exploration), and Reflector (insight generation and management). The design couples an insight-driven tree search with layered memory that promotes empirically validated insights and employs lightweight checkers to guard against unsafe actions. We implement a prototype and evaluate it on RocksDB, LevelDB, CacheLib, and MySQL InnoDB with YCSB, MixGraph, and TPC-H/C. Relative to out-of-the-box settings and to ELMo-Tune, StorageXTuner reaches up to 575% and 111% higher throughput, reduces p99 latency by as much as 88% and 56%, and converges with fewer trials.

**arXiv ID:** 2510.25017
</details>

<details>
<summary><strong>Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry</strong> - Run Peng, Ziqiao Ma, Amy Pang, Sikai Li, Zhang Xi-Jia, Yingzhuo Yu, Cristian-Paul Bara, Joyce Chai - [[pdf]](https://arxiv.org/pdf/2510.25595)</summary>

**Abstract:** While Large Language Model (LLM) agents are often approached from the angle of action planning/generation to accomplish a goal (e.g., given by language descriptions), their abilities to collaborate with each other to achieve a joint goal are not well explored. To address this limitation, this paper studies LLM agents in task collaboration, particularly under the condition of information asymmetry, where agents have disparities in their knowledge and skills and need to work together to complete a shared task. We extend Einstein Puzzles, a classical symbolic puzzle, to a table-top game. In this game, two LLM agents must reason, communicate, and act to satisfy spatial and relational constraints required to solve the puzzle. We apply a fine-tuning-plus-verifier framework in which LLM agents are equipped with various communication strategies and verification signals from the environment. Empirical results highlight the critical importance of aligned communication, especially when agents possess both information-seeking and -providing capabilities. Interestingly, agents without communication can still achieve high task performance; however, further analysis reveals a lack of true rule understanding and lower trust from human evaluators. Instead, by integrating an environment-based verifier, we enhance agents' ability to comprehend task rules and complete tasks, promoting both safer and more interpretable collaboration in AI systems. this https URL

**arXiv ID:** 2510.25595
</details>

<details>
<summary><strong>Emergent Coordinated Behaviors in Networked LLM Agents: Modeling the Strategic Dynamics of Information Operations</strong> - Gian Marco Orlando, Jinyi Ye, Valerio La Gatta, Mahdi Saeedi, Vincenzo Moscato, Emilio Ferrara, Luca Luceri - [[pdf]](https://arxiv.org/pdf/2510.25003)</summary>

**Abstract:** Generative agents are rapidly advancing in sophistication, raising urgent questions about how they might coordinate when deployed in online ecosystems. This is particularly consequential in information operations (IOs), influence campaigns that aim to manipulate public opinion on social media. While traditional IOs have been orchestrated by human operators and relied on manually crafted tactics, agentic AI promises to make campaigns more automated, adaptive, and difficult to detect. This work presents the first systematic study of emergent coordination among generative agents in simulated IO campaigns. Using generative agent-based modeling, we instantiate IO and organic agents in a simulated environment and evaluate coordination across operational regimes, from simple goal alignment to team knowledge and collective decision-making. As operational regimes become more structured, IO networks become denser and more clustered, interactions more reciprocal and positive, narratives more homogeneous, amplification more synchronized, and hashtag adoption faster and more sustained. Remarkably, simply revealing to agents which other agents share their goals can produce coordination levels nearly equivalent to those achieved through explicit deliberation and collective voting. Overall, we show that generative agents, even without human guidance, can reproduce coordination strategies characteristic of real-world IOs, underscoring the societal risks posed by increasingly automated, self-organizing IOs.

**arXiv ID:** 2510.25003
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (22 papers)</h2></summary>

<details>
<summary><strong>Agentic Moderation: Multi-Agent Design for Safer Vision-Language Models</strong> - Juan Ren, Mark Dras, Usman Naseem - [[pdf]](https://arxiv.org/pdf/2510.25179)</summary>

**Abstract:** Agentic methods have emerged as a powerful and autonomous paradigm that enhances reasoning, collaboration, and adaptive control, enabling systems to coordinate and independently solve complex tasks. We extend this paradigm to safety alignment by introducing Agentic Moderation, a model-agnostic framework that leverages specialised agents to defend multimodal systems against jailbreak attacks. Unlike prior approaches that apply as a static layer over inputs or outputs and provide only binary classifications (safe or unsafe), our method integrates dynamic, cooperative agents, including Shield, Responder, Evaluator, and Reflector, to achieve context-aware and interpretable moderation. Extensive experiments across five datasets and four representative Large Vision-Language Models (LVLMs) demonstrate that our approach reduces the Attack Success Rate (ASR) by 7-19%, maintains a stable Non-Following Rate (NF), and improves the Refusal Rate (RR) by 4-20%, achieving robust, interpretable, and well-balanced safety performance. By harnessing the flexibility and reasoning capacity of agentic architectures, Agentic Moderation provides modular, scalable, and fine-grained safety enforcement, highlighting the broader potential of agentic systems as a foundation for automated safety governance.

**arXiv ID:** 2510.25179
</details>

<details>
<summary><strong>FELA: A Multi-Agent Evolutionary System for Feature Engineering of Industrial Event Log Data</strong> - Kun ouyang, Haoyu Wang, Dong Fang - [[pdf]](https://arxiv.org/pdf/2510.25223)</summary>

**Abstract:** Event log data, recording fine-grained user actions and system events, represent one of the most valuable assets for modern digital services. However, the complexity and heterogeneity of industrial event logs--characterized by large scale, high dimensionality, diverse data types, and intricate temporal or relational structures--make feature engineering extremely challenging. Existing automatic feature engineering approaches, such as AutoML or genetic methods, often suffer from limited explainability, rigid predefined operations, and poor adaptability to complicated heterogeneous data. In this paper, we propose FELA (Feature Engineering LLM Agents), a multi-agent evolutionary system that autonomously extracts meaningful and high-performing features from complex industrial event log data. FELA integrates the reasoning and coding capabilities of large language models (LLMs) with an insight-guided self-evolution paradigm. Specifically, FELA employs specialized agents--Idea Agents, Code Agents, and Critic Agents--to collaboratively generate, validate, and implement novel feature ideas. An Evaluation Agent summarizes feedback and updates a hierarchical knowledge base and dual-memory system to enable continual improvement. Moreover, FELA introduces an agentic evolution algorithm, combining reinforcement learning and genetic algorithm principles to balance exploration and exploitation across the idea space. Extensive experiments on real industrial datasets demonstrate that FELA can generate explainable, domain-relevant features that significantly improve model performance while reducing manual effort. Our results highlight the potential of LLM-based multi-agent systems as a general framework for automated, interpretable, and adaptive feature engineering in complex real-world environments.

**arXiv ID:** 2510.25223
</details>

<details>
<summary><strong>Retrieval Augmented Generation (RAG) for Fintech: Agentic Design and Evaluation</strong> - Thomas Cook, Richard Osuagwu, Liman Tsatiashvili, Vrynsia Vrynsia, Koustav Ghosal, Maraim Masoud, Riccardo Mattivi - [[pdf]](https://arxiv.org/pdf/2510.25518)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) systems often face limitations in specialized domains such as fintech, where domain-specific ontologies, dense terminology, and acronyms complicate effective retrieval and synthesis. This paper introduces an agentic RAG architecture designed to address these challenges through a modular pipeline of specialized agents. The proposed system supports intelligent query reformulation, iterative sub-query decomposition guided by keyphrase extraction, contextual acronym resolution, and cross-encoder-based context re-ranking. We evaluate our approach against a standard RAG baseline using a curated dataset of 85 question--answer--reference triples derived from an enterprise fintech knowledge base. Experimental results demonstrate that the agentic RAG system outperforms the baseline in retrieval precision and relevance, albeit with increased latency. These findings suggest that structured, multi-agent methodologies offer a promising direction for enhancing retrieval robustness in complex, domain-specific settings.

**arXiv ID:** 2510.25518
</details>

<details>
<summary><strong>Counterfactual-based Agent Influence Ranker for Agentic AI Workflows</strong> - Amit Giloni, Chiara Picardi, Roy Betser, Shamik Bose, Aishvariya Priya Rathina Sabapathy, Roman Vainshtein - [[pdf]](https://arxiv.org/pdf/2510.25612)</summary>

**Abstract:** An Agentic AI Workflow (AAW), also known as an LLM-based multi-agent system, is an autonomous system that assembles several LLM-based agents to work collaboratively towards a shared goal. The high autonomy, widespread adoption, and growing interest in such AAWs highlight the need for a deeper understanding of their operations, from both quality and security aspects. To this day, there are no existing methods to assess the influence of each agent on the AAW's final output. Adopting techniques from related fields is not feasible since existing methods perform only static structural analysis, which is unsuitable for inference time execution. We present Counterfactual-based Agent Influence Ranker (CAIR) - the first method for assessing the influence level of each agent on the AAW's output and determining which agents are the most influential. By performing counterfactual analysis, CAIR provides a task-agnostic analysis that can be used both offline and at inference time. We evaluate CAIR using an AAWs dataset of our creation, containing 30 different use cases with 230 different functionalities. Our evaluation showed that CAIR produces consistent rankings, outperforms baseline methods, and can easily enhance the effectiveness and relevancy of downstream tasks.

**arXiv ID:** 2510.25612
</details>

<details>
<summary><strong>Dingtalk DeepResearch: A Unified Multi Agent Framework for Adaptive Intelligence in Enterprise Environments</strong> - Mengyuan Chen, Chengjun Dai, Xinyang Dong, Chengzhe Feng, Kewei Fu, Jianshe Li, Zhihan Peng, Yongqi Tong, Junshao Zhang, Hong Zhu - [[pdf]](https://arxiv.org/pdf/2510.24760)</summary>

**Abstract:** We present Dingtalk DeepResearch, a unified multi agent intelligence framework for real world enterprise environments, delivering deep research, heterogeneous table reasoning, and multimodal report generation.

**arXiv ID:** 2510.24760
</details>

<details>
<summary><strong>MASPRM: Multi-Agent System Process Reward Model</strong> - Milad Yazdani, Mahdi Mostajabdaveh, Zirui Zhou, Ying Xiong - [[pdf]](https://arxiv.org/pdf/2510.24803)</summary>

**Abstract:** Practical deployment of Multi-Agent Systems (MAS) demands strong test-time performance, motivating methods that guide inference-time search and selectively spend compute to improve quality. We present the Multi-Agent System Process Reward Model (MASPRM). It assigns per-action, per-agent values to partial inter-agent transcripts and acts as an inference-time controller. MASPRM is trained from multi-agent Monte Carlo Tree Search (MCTS) rollouts without requiring step-level human annotations, by propagating returns to local targets. At inference, MASPRM guides step-level beam search and MCTS, focusing computation on promising branches and pruning early. On GSM8K and MATH, MASPRM-guided decoding with an outcome reward model (ORM) applied to the final answer, improves exact match (EM) over a single straight-through MAS pass by $+30.7$ and $+22.9$ points, respectively. A MASPRM trained on GSM8K transfers zero-shot to MATH without retraining, adding $8.4$ EM points at the same budget. MASPRM is a plug-in value model that estimates per-agent progress and complements verifier-style decoders, enabling more reliable, compute-aware multi-agent reasoning. Code: this https URL

**arXiv ID:** 2510.24803
</details>

<details>
<summary><strong>Trust Dynamics in Strategic Coopetition: Computational Foundations for Requirements Engineering in Multi-Agent Systems</strong> - Vik Pant, Eric Yu - [[pdf]](https://arxiv.org/pdf/2510.24909)</summary>

**Abstract:** Requirements engineering increasingly occurs in multi-stakeholder environments where organizations simultaneously cooperate and compete, creating coopetitive relationships in which trust evolves dynamically based on observed behavior over repeated interactions. While conceptual modeling languages like i* represent trust relationships qualitatively, they lack computational mechanisms for analyzing how trust changes with behavioral evidence. Conversely, computational trust models from multi-agent systems provide algorithmic updating but lack grounding in requirements engineering contexts and conceptual models. This technical report bridges this gap by developing a computational trust model that extends game-theoretic foundations for strategic coopetition with dynamic trust evolution. We introduce trust as a two-layer system with immediate trust responding to current behavior and reputation tracking violation history. Trust evolves through asymmetric updating where cooperation builds trust gradually while violations erode it sharply, creating hysteresis effects and trust ceilings that constrain relationship recovery. We develop a structured translation framework enabling requirements engineers to instantiate computational trust models from i* dependency networks and organizational contexts. Comprehensive experimental validation across 78,125 parameter configurations establishes robust emergence of negativity bias, hysteresis effects, and cumulative damage amplification. Empirical validation using the Renault-Nissan Alliance case study (1999-2025) achieves 49 out of 60 validation points (81.7%), successfully reproducing documented trust evolution across five distinct relationship phases including crisis and recovery periods. This technical report builds upon its foundational companion work in arXiv:2510.18802.

**arXiv ID:** 2510.24909
</details>

<details>
<summary><strong>Multi-party Agent Relation Sampling for Multi-party Ad Hoc Teamwork</strong> - Beiwen Zhang, Yongheng Liang, Hejun Wu - [[pdf]](https://arxiv.org/pdf/2510.25340)</summary>

**Abstract:** Multi-agent reinforcement learning (MARl) has achieved strong results in cooperative tasks but typically assumes fixed, fully controlled teams. Ad hoc teamwork (AHT) relaxes this by allowing collaboration with unknown partners, yet existing variants still presume shared conventions. We introduce Multil-party Ad Hoc Teamwork (MAHT), where controlled agents must coordinate with multiple mutually unfamiliar groups of uncontrolled teammates. To address this, we propose MARs, which builds a sparse skeleton graph and applies relational modeling to capture cross-group dvnamics. Experiments on MPE and starCralt ll show that MARs outperforms MARL and AHT baselines while converging faster.

**arXiv ID:** 2510.25340
</details>

<details>
<summary><strong>Task Completion Agents are Not Ideal Collaborators</strong> - Shannon Zejiang Shen, Valerie Chen, Ken Gu, Alexis Ross, Zixian Ma, Jillian Ross, Alex Gu, Chenglei Si, Wayne Chi, Andi Peng, Jocelyn J Shen, Ameet Talwalkar, Tongshuang Wu, David Sontag - [[pdf]](https://arxiv.org/pdf/2510.25744)</summary>

**Abstract:** Current evaluations of agents remain centered around one-shot task completion, failing to account for the inherently iterative and collaborative nature of many real-world problems, where human goals are often underspecified and evolve. We argue for a shift from building and assessing task completion agents to developing collaborative agents, assessed not only by the quality of their final outputs but by how well they engage with and enhance human effort throughout the problem-solving process. To support this shift, we introduce collaborative effort scaling, a framework that captures how an agent's utility grows with increasing user involvement. Through case studies and simulated evaluations, we show that state-of-the-art agents often underperform in multi-turn, real-world scenarios, revealing a missing ingredient in agent design: the ability to sustain engagement and scaffold user understanding. Collaborative effort scaling offers a lens for diagnosing agent behavior and guiding development toward more effective interactions.

**arXiv ID:** 2510.25744
</details>

<details>
<summary><strong>Integrating Counterfactual Simulations with Language Models for Explaining Multi-Agent Behaviour</strong> - Bálint Gyevnár, Christopher G. Lucas, Stefano V. Albrecht, Shay B. Cohen - [[pdf]](https://arxiv.org/pdf/2505.17801)</summary>

**Abstract:** Autonomous multi-agent systems (MAS) are useful for automating complex tasks but raise trust concerns due to risks such as miscoordination or goal misalignment. Explainability is vital for users' trust calibration, but explainable MAS face challenges due to complex environments, the human factor, and non-standardised evaluation. Leveraging the counterfactual effect size model and LLMs, we propose Agentic eXplanations via Interrogative Simulation (AXIS). AXIS generates human-centred action explanations for multi-agent policies by having an LLM interrogate an environment simulator using prompts like 'whatif' and 'remove' to observe and synthesise counterfactual information over multiple rounds. We evaluate AXIS on autonomous driving across ten scenarios for five LLMs with a comprehensive methodology combining robustness, subjective preference, correctness, and goal/action prediction with an external LLM as evaluator. Compared to baselines, AXIS improves perceived explanation correctness by at least 7.7% across all models and goal prediction accuracy by 23% for four models, with comparable action prediction accuracy, achieving the highest scores overall. Our code is open-sourced at this https URL.

**arXiv ID:** 2505.17801
</details>

<details>
<summary><strong>HAMLET: Hyperadaptive Agent-based Modeling for Live Embodied Theatrics</strong> - Sizhou Chen, Shufan Jiang, Chi Zhang, Xiao-Lei Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2507.15518)</summary>

**Abstract:** Creating an immersive and interactive theatrical experience is a long-term goal in the field of interactive narrative. The emergence of large language model (LLM) is providing a new path to achieve this goal. However, existing LLM-based drama generation methods often result in agents that lack initiative and cannot interact with the physical scene. Furthermore, these methods typically require detailed user input to drive the drama. These limitations reduce the interactivity and immersion of online real-time performance. To address the above challenges, we propose HAMLET, a multi-agent framework focused on drama creation and online performance. Given a simple topic, the framework generates a narrative blueprint, guiding the subsequent improvisational performance. During the online performance, each actor is given an autonomous mind. This means that actors can make independent decisions based on their own background, goals, and emotional state. In addition to conversations with other actors, their decisions can also change the state of scene props through actions such as opening a letter or picking up a weapon. The change is then broadcast to other related actors, updating what they know and care about, which in turn influences their next action. To evaluate the quality of drama performance generated by HAMLET, we designed an evaluation method to assess three primary aspects, including character performance, narrative quality, and interaction experience. The experimental evaluation shows that HAMLET can create expressive and coherent theatrical experiences.

**arXiv ID:** 2507.15518
</details>

<details>
<summary><strong>Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</strong> - Ashmi Banerjee, Fitri Nur Aisyah, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo - [[pdf]](https://arxiv.org/pdf/2508.15030)</summary>

**Abstract:** We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems.

**arXiv ID:** 2508.15030
</details>

<details>
<summary><strong>VDSAgents: A PCS-Guided Multi-Agent System for Veridical Data Science Automation</strong> - Yunxuan Jiang, Silan Hu, Xiaoning Wang, Yuanyuan Zhang, Xiangyu Chang - [[pdf]](https://arxiv.org/pdf/2510.24339)</summary>

**Abstract:** Large language models (LLMs) become increasingly integrated into data science workflows for automated system design. However, these LLM-driven data science systems rely solely on the internal reasoning of LLMs, lacking guidance from scientific and theoretical principles. This limits their trustworthiness and robustness, especially when dealing with noisy and complex real-world datasets. This paper provides VDSAgents, a multi-agent system grounded in the Predictability-Computability-Stability (PCS) principles proposed in the Veridical Data Science (VDS) framework. Guided by PCS principles, the system implements a modular workflow for data cleaning, feature engineering, modeling, and evaluation. Each phase is handled by an elegant agent, incorporating perturbation analysis, unit testing, and model validation to ensure both functionality and scientific auditability. We evaluate VDSAgents on nine datasets with diverse characteristics, comparing it with state-of-the-art end-to-end data science systems, such as AutoKaggle and DataInterpreter, using DeepSeek-V3 and GPT-4o as backends. VDSAgents consistently outperforms the results of AutoKaggle and DataInterpreter, which validates the feasibility of embedding PCS principles into LLM-driven data science automation.

**arXiv ID:** 2510.24339
</details>

<details>
<summary><strong>HyperMARL: Adaptive Hypernetworks for Multi-Agent RL</strong> - Kale-ab Abebe Tessera, Arrasy Rahman, Amos Storkey, Stefano V. Albrecht - [[pdf]](https://arxiv.org/pdf/2412.04233)</summary>

**Abstract:** Adaptive cooperation in multi-agent reinforcement learning (MARL) requires policies to express homogeneous, specialised, or mixed behaviours, yet achieving this adaptivity remains a critical challenge. While parameter sharing (PS) is standard for efficient learning, it notoriously suppresses the behavioural diversity required for specialisation. This failure is largely due to cross-agent gradient interference, a problem we find is surprisingly exacerbated by the common practice of coupling agent IDs with observations. Existing remedies typically add complexity through altered objectives, manual preset diversity levels, or sequential updates -- raising a fundamental question: can shared policies adapt without these intricacies? We propose a solution built on a key insight: an agent-conditioned hypernetwork can generate agent-specific parameters and decouple observation- and agent-conditioned gradients, directly countering the interference from coupling agent IDs with observations. Our resulting method, HyperMARL, avoids the complexities of prior work and empirically reduces policy gradient variance. Across diverse MARL benchmarks (22 scenarios, up to 30 agents), HyperMARL achieves performance competitive with six key baselines while preserving behavioural diversity comparable to non-parameter sharing methods, establishing it as a versatile and principled approach for adaptive MARL. The code is publicly available at this https URL.

**arXiv ID:** 2412.04233
</details>

<details>
<summary><strong>Redistributing Rewards Across Time and Agents for Multi-Agent Reinforcement Learning</strong> - Aditya Kapoor, Kale-ab Tessera, Mayank Baranwal, Harshad Khadilkar, Jan Peters, Stefano Albrecht, Mingfei Sun - [[pdf]](https://arxiv.org/pdf/2502.04864)</summary>

**Abstract:** Credit assignmen, disentangling each agent's contribution to a shared reward, is a critical challenge in cooperative multi-agent reinforcement learning (MARL). To be effective, credit assignment methods must preserve the environment's optimal policy. Some recent approaches attempt this by enforcing return equivalence, where the sum of distributed rewards must equal the team reward. However, their guarantees are conditional on a learned model's regression accuracy, making them unreliable in practice. We introduce Temporal-Agent Reward Redistribution (TAR$^2$), an approach that decouples credit modeling from this constraint. A neural network learns unnormalized contribution scores, while a separate, deterministic normalization step enforces return equivalence by construction. We demonstrate that this method is equivalent to a valid Potential-Based Reward Shaping (PBRS), which guarantees the optimal policy is preserved regardless of model accuracy. Empirically, on challenging SMACLite and Google Research Football (GRF) benchmarks, TAR$^2$ accelerates learning and achieves higher final performance than strong baselines. These results establish our method as an effective solution for the agent-temporal credit assignment problem.

**arXiv ID:** 2502.04864
</details>

<details>
<summary><strong>Partially Observable Multi-Agent Reinforcement Learning with Information Sharing</strong> - Xiangyu Liu, Kaiqing Zhang - [[pdf]](https://arxiv.org/pdf/2308.08705)</summary>

**Abstract:** We study provable multi-agent reinforcement learning (RL) in the general framework of partially observable stochastic games (POSGs). To circumvent the known hardness results and the use of computationally intractable oracles, we advocate leveraging the potential \emph{information-sharing} among agents, a common practice in empirical multi-agent RL, and a standard model for multi-agent control systems with communication. We first establish several computational complexity results to justify the necessity of information-sharing, as well as the observability assumption that has enabled quasi-polynomial time and sample single-agent RL with partial observations, for tractably solving POSGs. Inspired by the inefficiency of planning in the ground-truth model, we then propose to further \emph{approximate} the shared common information to construct an approximate model of the POSG, in which an approximate \emph{equilibrium} (of the original POSG) can be found in quasi-polynomial-time, under the aforementioned assumptions. Furthermore, we develop a partially observable multi-agent RL algorithm whose time and sample complexities are \emph{both} quasi-polynomial. Finally, beyond equilibrium learning, we extend our algorithmic framework to finding the \emph{team-optimal solution} in cooperative POSGs, i.e., decentralized partially observable Markov decision processes, a more challenging goal. We establish concrete computational and sample complexities under several structural assumptions of the model. We hope our study could open up the possibilities of leveraging and even designing different \emph{information structures}, a well-studied notion in control theory, for developing both sample- and computation-efficient partially observable multi-agent RL.

**arXiv ID:** 2308.08705
</details>

<details>
<summary><strong>DEBATE: A Large-Scale Benchmark for Role-Playing LLM Agents in Multi-Agent, Long-Form Debates</strong> - Yun-Shiuan Chuang, Ruixuan Tu, Chengtao Dai, Smit Vasani, Binwei Yao, Michael Henry Tessler, Sijia Yang, Dhavan Shah, Robert Hawkins, Junjie Hu, Timothy T. Rogers - [[pdf]](https://arxiv.org/pdf/2510.25110)</summary>

**Abstract:** Accurately modeling opinion change through social interactions is crucial for addressing issues like misinformation and polarization. While role-playing large language models (LLMs) offer a promising way to simulate human-like interactions, existing research shows that single-agent alignment does not guarantee authentic multi-agent group dynamics. Current LLM role-play setups often produce unnatural dynamics (e.g., premature convergence), without an empirical benchmark to measure authentic human opinion trajectories. To bridge this gap, we introduce DEBATE, the first large-scale empirical benchmark explicitly designed to evaluate the authenticity of the interaction between multi-agent role-playing LLMs. DEBATE contains 29,417 messages from multi-round debate conversations among over 2,792 U.S.-based participants discussing 107 controversial topics, capturing both publicly-expressed messages and privately-reported opinions. Using DEBATE, we systematically evaluate and identify critical discrepancies between simulated and authentic group dynamics. We further demonstrate DEBATE's utility for aligning LLMs with human behavior through supervised fine-tuning, achieving improvements in surface-level metrics (e.g., ROUGE-L and message length) while highlighting limitations in deeper semantic alignment (e.g., semantic similarity). Our findings highlight both the potential and current limitations of role-playing LLM agents for realistically simulating human-like social dynamics.

**arXiv ID:** 2510.25110
</details>

<details>
<summary><strong>AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System</strong> - Hui Yi Leong, Yuheng Li, Yuqing Wu, Wenwen Ouyang, Wei Zhu, Jiechao Gao, Wei Han - [[pdf]](https://arxiv.org/pdf/2510.01617)</summary>

**Abstract:** Although large language models (LLMs) have revolutionized natural language processing capabilities, their practical implementation as autonomous multi-agent systems (MAS) for industrial problem-solving encounters persistent barriers. Conventional MAS architectures are fundamentally restricted by inflexible, hand-crafted graph topologies that lack contextual responsiveness, resulting in diminished efficacy across varied academic and commercial workloads. To surmount these constraints, we introduce AMAS, a paradigm-shifting framework that redefines LLM-based MAS through a novel dynamic graph designer. This component autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation, eliminating the reliance on monolithic, universally applied structural templates. Instead, AMAS exploits the intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways. Rigorous validation across question answering, mathematical deduction, and code generation benchmarks confirms that AMAS systematically exceeds state-of-the-art single-agent and multi-agent approaches across diverse LLM architectures. Our investigation establishes that context-sensitive structural adaptability constitutes a foundational requirement for high-performance LLM MAS deployments.

**arXiv ID:** 2510.01617
</details>

<details>
<summary><strong>MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs</strong> - Yucheng Ning, Xixun Lin, Fang Fang, Yanan Cao - [[pdf]](https://arxiv.org/pdf/2510.22967)</summary>

**Abstract:** The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains.

**arXiv ID:** 2510.22967
</details>

<details>
<summary><strong>Incorporating Social Awareness into Control of Unknown Multi-Agent Systems: A Real-Time Spatiotemporal Tubes Approach</strong> - Siddhartha Upadhyay, Ratnangshu Das, Pushpak Jagtap - [[pdf]](https://arxiv.org/pdf/2510.25597)</summary>

**Abstract:** This paper presents a decentralized control framework that incorporates social awareness into multi-agent systems with unknown dynamics to achieve prescribed-time reach-avoid-stay tasks in dynamic environments. Each agent is assigned a social awareness index that quantifies its level of cooperation or self-interest, allowing heterogeneous social behaviors within the system. Building on the spatiotemporal tube (STT) framework, we propose a real-time STT framework that synthesizes tubes online for each agent while capturing its social interactions with others. A closed-form, approximation-free control law is derived to ensure that each agent remains within its evolving STT, thereby avoiding dynamic obstacles while also preventing inter-agent collisions in a socially aware manner, and reaching the target within a prescribed time. The proposed approach provides formal guarantees on safety and timing, and is computationally lightweight, model-free, and robust to unknown disturbances. The effectiveness and scalability of the framework are validated through simulation and hardware experiments on a 2D omnidirectional

**arXiv ID:** 2510.25597
</details>

<details>
<summary><strong>Taxonomy and Trends in Reinforcement Learning for Robotics and Control Systems: A Structured Review</strong> - Kumater Ter, Ore-Ofe Ajayi, Daniel Udekwe - [[pdf]](https://arxiv.org/pdf/2510.21758)</summary>

**Abstract:** Reinforcement learning (RL) has become a foundational approach for enabling intelligent robotic behavior in dynamic and uncertain environments. This work presents an in-depth review of RL principles, advanced deep reinforcement learning (DRL) algorithms, and their integration into robotic and control systems. Beginning with the formalism of Markov Decision Processes (MDPs), the study outlines essential elements of the agent-environment interaction and explores core algorithmic strategies including actor-critic methods, value-based learning, and policy gradients. Emphasis is placed on modern DRL techniques such as DDPG, TD3, PPO, and SAC, which have shown promise in solving high-dimensional, continuous control tasks. A structured taxonomy is introduced to categorize RL applications across domains such as locomotion, manipulation, multi-agent coordination, and human-robot interaction, along with training methodologies and deployment readiness levels. The review synthesizes recent research efforts, highlighting technical trends, design patterns, and the growing maturity of RL in real-world robotics. Overall, this work aims to bridge theoretical advances with practical implementations, providing a consolidated perspective on the evolving role of RL in autonomous robotic systems.

**arXiv ID:** 2510.21758
</details>

<details>
<summary><strong>OrchVis: Hierarchical Multi-Agent Orchestration for Human Oversight</strong> - Jieyu Zhou - [[pdf]](https://arxiv.org/pdf/2510.24937)</summary>

**Abstract:** We introduce OrchVis, a multi-agent orchestration framework that visualizes, verifies, and coordinates goal-driven collaboration among LLM-based agents. Through hierarchical goal alignment, task assignment, and conflict resolution, OrchVis enables humans to supervise complex multi-agent workflows without micromanaging each step. The system parses user intent into structured goals, monitors execution via automated verification, and exposes inter-agent dependencies through an interactive planning panel. When conflicts arise, users can explore system-proposed alternatives and selectively replan. OrchVis advances human-centered design for multi-agent systems by combining transparent visualization with adaptive autonomy.

**arXiv ID:** 2510.24937
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Agentic AI: A Comprehensive Survey of Architectures, Applications, and Future Directions</strong> - Mohamad Abou Ali, Fadi Dornaika - [[pdf]](https://arxiv.org/pdf/2510.25445)</summary>

**Abstract:** Agentic AI represents a transformative shift in artificial intelligence, but its rapid advancement has led to a fragmented understanding, often conflating modern neural systems with outdated symbolic models -- a practice known as conceptual retrofitting. This survey cuts through this confusion by introducing a novel dual-paradigm framework that categorizes agentic systems into two distinct lineages: the Symbolic/Classical (relying on algorithmic planning and persistent state) and the Neural/Generative (leveraging stochastic generation and prompt-driven orchestration). Through a systematic PRISMA-based review of 90 studies (2018--2025), we provide a comprehensive analysis structured around this framework across three dimensions: (1) the theoretical foundations and architectural principles defining each paradigm; (2) domain-specific implementations in healthcare, finance, and robotics, demonstrating how application constraints dictate paradigm selection; and (3) paradigm-specific ethical and governance challenges, revealing divergent risks and mitigation strategies. Our analysis reveals that the choice of paradigm is strategic: symbolic systems dominate safety-critical domains (e.g., healthcare), while neural systems prevail in adaptive, data-rich environments (e.g., finance). Furthermore, we identify critical research gaps, including a significant deficit in governance models for symbolic systems and a pressing need for hybrid neuro-symbolic architectures. The findings culminate in a strategic roadmap arguing that the future of Agentic AI lies not in the dominance of one paradigm, but in their intentional integration to create systems that are both adaptable and reliable. This work provides the essential conceptual toolkit to guide future research, development, and policy toward robust and trustworthy hybrid intelligent systems.

**arXiv ID:** 2510.25445
</details>

<details>
<summary><strong>DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</strong> - Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2510.14205)</summary>

**Abstract:** The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF). DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences. We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews. DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios. Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.

**arXiv ID:** 2510.14205
</details>

<details>
<summary><strong>Defect Mitigation for Robot Arm-based Additive Manufacturing Utilizing Intelligent Control and IOT</strong> - Matsive Ali, Blake Gassen, Sen Liu - [[pdf]](https://arxiv.org/pdf/2510.24994)</summary>

**Abstract:** This paper presents an integrated robotic fused deposition modeling additive manufacturing system featuring closed-loop thermal control and intelligent in-situ defect correction using a 6-degree of freedom robotic arm and an Oak-D camera. The robot arm end effector was modified to mount an E3D hotend thermally regulated by an IoT microcontroller, enabling precise temperature control through real-time feedback. Filament extrusion system was synchronized with robotic motion, coordinated via ROS2, ensuring consistent deposition along complex trajectories. A vision system based on OpenCV detects layer-wise defects position, commanding autonomous re-extrusion at identified sites. Experimental validation demonstrated successful defect mitigation in printing operations. The integrated system effectively addresses challenges real-time quality assurance. Inverse kinematics were used for motion planning, while homography transformations corrected camera perspectives for accurate defect localization. The intelligent system successfully mitigated surface anomalies without interrupting the print process. By combining real-time thermal regulation, motion control, and intelligent defect detection & correction, this architecture establishes a scalable and adaptive robotic additive manufacturing framework suitable for aerospace, biomedical, and industrial applications.

**arXiv ID:** 2510.24994
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (23 papers)</h2></summary>

<details>
<summary><strong>Scheduling Your LLM Reinforcement Learning with Reasoning Trees</strong> - Hong Wang, Zhezheng Hao, Jian Luo, Chenxing Wei, Yao Shu, Lei Liu, Qiang Lin, Hande Dong, Jiawei Chen - [[pdf]](https://arxiv.org/pdf/2510.24832)</summary>

**Abstract:** Using Reinforcement Learning with Verifiable Rewards (RLVR) to optimize Large Language Models (LLMs) can be conceptualized as progressively editing a query's `Reasoning Tree'. This process involves exploring nodes (tokens) and dynamically modifying the model's policy at each node. When combined with data scheduling, this process yields further gains in data efficiency and accuracy. However, existing RLVR data scheduling methods typically rely on path-based metrics to rank queries, overlooking the reasoning tree structures of these queries. In this paper, we introduce a novel metric, namely Reasoning Score (r-score), which measures the query's learning difficulty based on the structure of its reasoning tree. Based on the r-score, we propose the Reasoning Tree Schedule (Re-Schedule), a scheduling algorithm that constructs a curriculum progressing from structurally simple (high r-score) to complex (low r-score) queries. Experiments on six math-reasoning benchmarks show that Re-Schedule significantly improves average accuracy, achieving gains of up to 3.2%. These strong results validate our approach and demonstrate that a structural understanding of the reasoning tree provides a more powerful and principled foundation for RLVR data scheduling.

**arXiv ID:** 2510.24832
</details>

<details>
<summary><strong>KnowCoder-A1: Incentivizing Agentic Reasoning Capability with Outcome Supervision for KBQA</strong> - Zhuo Chen, Fei Wang, Zixuan Li, Zhao Zhang, Weiwei Ding, Chuanguang Yang, Yongjun Xu, Xiaolong Jin, Jiafeng Guo - [[pdf]](https://arxiv.org/pdf/2510.25101)</summary>

**Abstract:** Knowledge Base Question Answering (KBQA) aims to answer natural-language questions over a structured Knowledge Base (KB). Recent work improves KBQA by adopting an agentic reasoning paradigm, in which Large Language Models (LLMs) iteratively decompose a question, generate its corresponding logical queries, and interact with the KB to derive the answer. However, these methods typically fine-tune LLMs on reasoning trajectories synthesized via process supervision, which offers weak incentives for exploration and thus fails to strengthen the agentic reasoning ability. In this paper, we propose KnowCoder-A1, an LLM that can autonomously perform agentic reasoning on KBs to obtain answers. To incentivize autonomous exploration, KnowCoder-A1 trains the LLM under outcome-only supervision via a multi-stage curriculum reinforcement learning with an easy-to-hard curriculum. To establish foundational agentic capabilities, KnowCoder-A1 first fine-tunes the LLM on a small set of high-quality trajectories obtained through outcome-based rejection sampling. Then, to alleviate the reward sparsity inherent in outcome-only supervision, it applies multi-stage curriculum RL with reward schedules that progress from easy to hard. Trained with outcome-only supervision, KnowCoder-A1 exhibits powerful reasoning behaviors and consistently outperforms prior approaches across three mainstream datasets. Notably, on the zero-shot subset of GrailQA, KnowCoder-A1 achieves up to an 11.1% relative improvement while using only one-twelfth of the training data, demonstrating strong agentic reasoning capabilities.

**arXiv ID:** 2510.25101
</details>

<details>
<summary><strong>Energy-Efficient Autonomous Driving with Adaptive Perception and Robust Decision</strong> - Yuyang Xia, Zibo Liang, Liwei Deng, Yan Zhao, Han Su, Kai Zheng - [[pdf]](https://arxiv.org/pdf/2510.25205)</summary>

**Abstract:** Autonomous driving is an emerging technology that is expected to bring significant social, economic, and environmental benefits. However, these benefits come with rising energy consumption by computation engines, limiting the driving range of vehicles, especially electric ones. Perception computing is typically the most power-intensive component, as it relies on largescale deep learning models to extract environmental features. Recently, numerous studies have employed model compression techniques, such as sparsification, quantization, and distillation, to reduce computational consumption. However, these methods often result in either a substantial model size or a significant drop in perception accuracy compared to high-computation models. To address these challenges, we propose an energy-efficient autonomous driving framework, called EneAD. In the adaptive perception module, a perception optimization strategy is designed from the perspective of data management and tuning. Firstly, we manage multiple perception models with different computational consumption and adjust the execution framerate dynamically. Then, we define them as knobs and design a transferable tuning method based on Bayesian optimization to identify promising knob values that achieve low computation while maintaining desired accuracy. To adaptively switch the knob values in various traffic scenarios, a lightweight classification model is proposed to distinguish the perception difficulty in different scenarios. In the robust decision module, we propose a decision model based on reinforcement learning and design a regularization term to enhance driving stability in the face of perturbed perception results. Extensive experiments evidence the superiority of our framework in both energy consumption and driving performance. EneAD can reduce perception consumption by 1.9x to 3.5x and thus improve driving range by 3.9% to 8.5%

**arXiv ID:** 2510.25205
</details>

<details>
<summary><strong>MTIR-SQL: Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL</strong> - Zekun Xu, Siyu Xia, Chuhuai Yue, Jiajun Chai, Mingxue Tian, Xiaohan Wang, Wei Lin, Haoxuan Li, Guojun Yin - [[pdf]](https://arxiv.org/pdf/2510.25510)</summary>

**Abstract:** As large language models (LLMs) are increasingly used in Text-to-SQL tasks, Reinforcement Learning (RL) has become a common method for improving performance. Existing methods primarily rely on static execution feedback, which restricts real-time error correction. However, integrating multi-turn tool invocation along with dynamic feedback could significantly improve adaptability and robustness, ultimately enhancing model performance. To address these issues, we propose MTIR-SQL, an innovative Multi-turn Tool-Integrated Reasoning reinforcement learning framework for Text-to-SQL. Our approach introduces an execution-aware multi-turn reasoning paradigm that seamlessly incorporates database execution feedback at each reasoning step, enabling context-sensitive query generation and progressive refinement throughout the reasoning process. The framework extends the GRPO algorithm to accommodate complex multi-turn interaction scenarios. Considering the training instability characteristics of MTIR and the potential for significant Deviation of model distribution from the initial model, we enhance the GRPO algorithm by adding a trajectory filtering mechanism and removing KL loss constraints. Experimental results demonstrate that MTIR-SQL, with 4B parameters, achieves \textbf{64.4}\% accuracy in the BIRD Dev and 84.6% execution accuracy in the SPIDER Dev, significantly outperforming existing approaches.

**arXiv ID:** 2510.25510
</details>

<details>
<summary><strong>Zero Reinforcement Learning Towards General Domains</strong> - Yuyuan Zeng, Yufei Huang, Can Xu, Qingfeng Sun, Jianfeng Yan, Guanghui Xu, Tao Yang, Fengzong Lian - [[pdf]](https://arxiv.org/pdf/2510.25528)</summary>

**Abstract:** Zero Reinforcement Learning (Zero-RL) has proven to be an effective approach for enhancing the reasoning capabilities of large language models (LLMs) by directly applying reinforcement learning with verifiable rewards on pretrained models, without the need for a supervised fine-tuning phase. However, current research on zero-RL primarily focuses on domains with easily verifiable reward signals, such as mathematics, programming, and other reasoning tasks. The challenge of eliciting reasoning abilities in more diverse scenarios, where verification is not straightforward, remains underexplored. To address this gap, we propose a novel zero-RL paradigm designed to improve a model's reasoning ability across both verifiable and non-verifiable domains. By combining verifiable rewards with a generative reward model, we conduct multi-task zero-RL training across both domains, facilitating the transfer of reasoning capabilities between them. Furthermore, to mitigate reward hacking in the generative reward model, we design a smooth length penalty that encourages the generation of more comprehensive thinking tokens in general domains. Experimental results on Qwen3-8B-Base and Qwen3-14B-Base demonstrate that our approach achieves superior reasoning performance, not only on tasks requiring extensive reasoning but also on more general tasks.

**arXiv ID:** 2510.25528
</details>

<details>
<summary><strong>Off-policy Reinforcement Learning with Model-based Exploration Augmentation</strong> - Likun Wang, Xiangteng Zhang, Yinuo Wang, Guojian Zhan, Wenxuan Wang, Haoyu Gao, Jingliang Duan, Shengbo Eben Li - [[pdf]](https://arxiv.org/pdf/2510.25529)</summary>

**Abstract:** Exploration is fundamental to reinforcement learning (RL), as it determines how effectively an agent discovers and exploits the underlying structure of its environment to achieve optimal performance. Existing exploration methods generally fall into two categories: active exploration and passive exploration. The former introduces stochasticity into the policy but struggles in high-dimensional environments, while the latter adaptively prioritizes transitions in the replay buffer to enhance exploration, yet remains constrained by limited sample diversity. To address the limitation in passive exploration, we propose Modelic Generative Exploration (MoGE), which augments exploration through the generation of under-explored critical states and synthesis of dynamics-consistent experiences through transition models. MoGE is composed of two components: (1) a diffusion-based generator that synthesizes critical states under the guidance of a utility function evaluating each state's potential influence on policy exploration, and (2) a one-step imagination world model for constructing critical transitions based on the critical states for agent learning. Our method adopts a modular formulation that aligns with the principles of off-policy learning, allowing seamless integration with existing algorithms to improve exploration without altering their core structures. Empirical results on OpenAI Gym and DeepMind Control Suite reveal that MoGE effectively bridges exploration and policy learning, leading to remarkable gains in both sample efficiency and performance across complex control tasks.

**arXiv ID:** 2510.25529
</details>

<details>
<summary><strong>ALDEN: Reinforcement Learning for Active Navigation and Evidence Gathering in Long Documents</strong> - Tianyu Yang, Terry Ruas, Yijun Tian, Jan Philip Wahle, Daniel Kurzawe, Bela Gipp - [[pdf]](https://arxiv.org/pdf/2510.25668)</summary>

**Abstract:** Vision-language models (VLMs) excel at interpreting text-rich images but struggle with long, visually complex documents that demand analysis and integration of information spread across multiple pages. Existing approaches typically rely on fixed reasoning templates or rigid pipelines, which force VLMs into a passive role and hinder both efficiency and generalization. We present Active Long-DocumEnt Navigation (ALDEN), a multi-turn reinforcement learning framework that fine-tunes VLMs as interactive agents capable of actively navigating long, visually rich documents. ALDEN introduces a novel fetch action that directly accesses the page by index, complementing the classic search action and better exploiting document structure. For dense process supervision and efficient training, we propose a rule-based cross-level reward that provides both turn- and token-level signals. To address the empirically observed training instability caused by numerous visual tokens from long documents, we further propose a visual-semantic anchoring mechanism that applies a dual-path KL-divergence constraint to stabilize visual and textual representations separately during training. Trained on a corpus constructed from three open-source datasets, ALDEN achieves state-of-the-art performance on five long-document benchmarks. Overall, ALDEN marks a step beyond passive document reading toward agents that autonomously navigate and reason across long, visually rich documents, offering a robust path to more accurate and efficient long-document understanding.

**arXiv ID:** 2510.25668
</details>

<details>
<summary><strong>Navigation in a Three-Dimensional Urban Flow using Deep Reinforcement Learning</strong> - Federica Tonti, Ricardo Vinuesa - [[pdf]](https://arxiv.org/pdf/2510.25679)</summary>

**Abstract:** Unmanned Aerial Vehicles (UAVs) are increasingly populating urban areas for delivery and surveillance purposes. In this work, we develop an optimal navigation strategy based on Deep Reinforcement Learning. The environment is represented by a three-dimensional high-fidelity simulation of an urban flow, characterized by turbulence and recirculation zones. The algorithm presented here is a flow-aware Proximal Policy Optimization (PPO) combined with a Gated Transformer eXtra Large (GTrXL) architecture, giving the agent richer information about the turbulent flow field in which it navigates. The results are compared with a PPO+GTrXL without the secondary prediction tasks, a PPO combined with Long Short Term Memory (LSTM) cells and a traditional navigation algorithm. The obtained results show a significant increase in the success rate (SR) and a lower crash rate (CR) compared to a PPO+LSTM, PPO+GTrXL and the classical Zermelo's navigation algorithm, paving the way to a completely reimagined UAV landscape in complex urban environments.

**arXiv ID:** 2510.25679
</details>

<details>
<summary><strong>FT-ARM: Fine-Tuned Agentic Reflection Multimodal Language Model for Pressure Ulcer Severity Classification with Reasoning</strong> - Reza Saadati Fard, Emmanuel Agu, Palawat Busaranuvong, Deepak Kumar, Shefalika Gautam, Bengisu Tulu, Diane Strong, Lorraine Loretz - [[pdf]](https://arxiv.org/pdf/2510.24980)</summary>

**Abstract:** Pressure ulcers (PUs) are a serious and prevalent healthcare concern. Accurate classification of PU severity (Stages I-IV) is essential for proper treatment but remains challenging due to subtle visual distinctions and subjective interpretation, leading to variability among clinicians. Prior AI-based approaches using Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) achieved promising accuracy but offered limited interpretability. We present FT-ARM (Fine-Tuned Agentic Reflection Multimodal model), a fine-tuned multimodal large language model (MLLM) with an agentic self-reflection mechanism for pressure ulcer severity classification. Inspired by clinician-style diagnostic reassessment, FT-ARM iteratively refines its predictions by reasoning over visual features and encoded clinical knowledge from text, enhancing both accuracy and consistency. On the publicly available Pressure Injury Image Dataset (PIID), FT-ARM, fine-tuned from LLaMA 3.2 90B, achieved 85% accuracy in classifying PU stages I-IV, surpassing prior CNN-based models by +4%. Unlike earlier CNN/ViT studies that relied solely on offline evaluations, FT-ARM is designed and tested for live inference, reflecting real-time deployment conditions. Furthermore, it produces clinically grounded natural-language explanations, improving interpretability and trust. By integrating fine-tuning and reflective reasoning across multimodal inputs, FT-ARM advances the reliability, transparency, and clinical applicability of automated wound assessment systems, addressing the critical need for consistent and explainable PU staging to support improved patient care.

**arXiv ID:** 2510.24980
</details>

<details>
<summary><strong>Dense and Diverse Goal Coverage in Multi Goal Reinforcement Learning</strong> - Sagalpreet Singh, Rishi Saket, Aravindan Raghuveer - [[pdf]](https://arxiv.org/pdf/2510.25311)</summary>

**Abstract:** Reinforcement Learning algorithms are primarily focused on learning a policy that maximizes expected return. As a result, the learned policy can exploit one or few reward sources. However, in many natural situations, it is desirable to learn a policy that induces a dispersed marginal state distribution over rewarding states, while maximizing the expected return which is typically tied to reaching a goal state. This aspect remains relatively unexplored. Existing techniques based on entropy regularization and intrinsic rewards use stochasticity for encouraging exploration to find an optimal policy which may not necessarily lead to dispersed marginal state distribution over rewarding states. Other RL algorithms which match a target distribution assume the latter to be available apriori. This may be infeasible in large scale systems where enumeration of all states is not possible and a state is determined to be a goal state only upon reaching it. We formalize the problem of maximizing the expected return while uniformly visiting the goal states as Multi Goal RL in which an oracle classifier over the state space determines the goal states. We propose a novel algorithm that learns a high-return policy mixture with marginal state distribution dispersed over the set of goal states. Our algorithm is based on optimizing a custom RL reward which is computed - based on the current policy mixture - at each iteration for a set of sampled trajectories. The latter are used via an offline RL algorithm to update the policy mixture. We prove performance guarantees for our algorithm, showing efficient convergence bounds for optimizing a natural objective which captures the expected return as well as the dispersion of the marginal state distribution over the goal states. We design and perform experiments on synthetic MDPs and standard RL environments to evaluate the effectiveness of our algorithm.

**arXiv ID:** 2510.25311
</details>

<details>
<summary><strong>Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models</strong> - Jiaqi Wang, Kevin Qinghong Lin, James Cheng, Mike Zheng Shou - [[pdf]](https://arxiv.org/pdf/2505.16854)</summary>

**Abstract:** Reinforcement Learning (RL) has proven to be an effective post-training strategy for enhancing reasoning in vision-language models (VLMs). Group Relative Policy Optimization (GRPO) is a recent prominent method that encourages models to generate complete reasoning traces before answering, leading to increased token usage and computational cost. Inspired by the human-like thinking process-where people skip reasoning for easy questions but think carefully when needed-we explore how to enable VLMs to first decide when reasoning is necessary. To realize this, we propose TON, a two-stage training strategy: (i) a supervised fine-tuning (SFT) stage with a simple yet effective 'thought dropout' operation, where reasoning traces are randomly replaced with empty thoughts. This introduces a think-or-not format that serves as a cold start for selective reasoning; (ii) a GRPO stage that enables the model to freely explore when to think or not, while maximizing task-aware outcome rewards. Experimental results show that TON can reduce the completion length by up to 90% compared to vanilla GRPO, without sacrificing performance or even improving it. Further evaluations across LLM (GSM8K), VLM (CLEVR, Super-CLEVR, GeoQA), and Agentic (AITZ) tasks-covering a range of reasoning difficulties under both 3B and 7B models-consistently reveal that the model progressively learns to bypass unnecessary reasoning steps as training advances. These findings shed light on the path toward human-like reasoning patterns in RL approaches. Our code is available at this https URL.

**arXiv ID:** 2505.16854
</details>

<details>
<summary><strong>Pentest-R1: Towards Autonomous Penetration Testing Reasoning Optimized via Two-Stage Reinforcement Learning</strong> - He Kong, Die Hu, Jingguo Ge, Liangxiong Li, Hui Li, Tong Li - [[pdf]](https://arxiv.org/pdf/2508.07382)</summary>

**Abstract:** Automating penetration testing is crucial for enhancing cybersecurity, yet current Large Language Models (LLMs) face significant limitations in this domain, including poor error handling, inefficient reasoning, and an inability to perform complex end-to-end tasks autonomously. To address these challenges, we introduce Pentest-R1, a novel framework designed to optimize LLM reasoning capabilities for this task through a two-stage reinforcement learning pipeline. We first construct a dataset of over 500 real-world, multi-step walkthroughs, which Pentest-R1 leverages for offline reinforcement learning (RL) to instill foundational attack logic. Subsequently, the LLM is fine-tuned via online RL in an interactive Capture The Flag (CTF) environment, where it learns directly from environmental feedback to develop robust error self-correction and adaptive strategies. Our extensive experiments on the Cybench and AutoPenBench benchmarks demonstrate the framework's effectiveness. On AutoPenBench, Pentest-R1 achieves a 24.2\% success rate, surpassing most state-of-the-art models and ranking second only to Gemini 2.5 Flash. On Cybench, it attains a 15.0\% success rate in unguided tasks, establishing a new state-of-the-art for open-source LLMs and matching the performance of top proprietary models. Ablation studies confirm that the synergy of both training stages is critical to its success.

**arXiv ID:** 2508.07382
</details>

<details>
<summary><strong>The Landscape of Agentic Reinforcement Learning for LLMs: A Survey</strong> - Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Francisco Piedrahita-Velez, Yue Liao, Hongru Wang, Mengyue Yang, Heng Ji, Michael Littman, Jun Wang, Shuicheng Yan, Philip Torr, Lei Bai - [[pdf]](https://arxiv.org/pdf/2509.02547)</summary>

**Abstract:** The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.

**arXiv ID:** 2509.02547
</details>

<details>
<summary><strong>ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation</strong> - Ziyi Liu, Bahar Sarrafzadeh, Pei Zhou, Longqi Yang, Jieyu Zhao, Ashish Sharma - [[pdf]](https://arxiv.org/pdf/2510.25224)</summary>

**Abstract:** While Large Language Models (LLMs) are increasingly used in agentic frameworks to assist individual users, there is a growing need for agents that can proactively manage complex, multi-party collaboration. Systematic evaluation methods for such proactive agents remain scarce, limiting progress in developing AI that can effectively support multiple people together. Negotiation offers a demanding testbed for this challenge, requiring socio-cognitive intelligence to navigate conflicting interests between multiple participants and multiple topics and build consensus. Here, we present ProMediate, the first framework for evaluating proactive AI mediator agents in complex, multi-topic, multi-party negotiations. ProMediate consists of two core components: (i) a simulation testbed based on realistic negotiation cases and theory-driven difficulty levels (ProMediate-Easy, ProMediate-Medium, and ProMediate-Hard), with a plug-and-play proactive AI mediator grounded in socio-cognitive mediation theories, capable of flexibly deciding when and how to intervene; and (ii) a socio-cognitive evaluation framework with a new suite of metrics to measure consensus changes, intervention latency, mediator effectiveness, and intelligence. Together, these components establish a systematic framework for assessing the socio-cognitive intelligence of proactive AI agents in multi-party settings. Our results show that a socially intelligent mediator agent outperforms a generic baseline, via faster, better-targeted interventions. In the ProMediate-Hard setting, our social mediator increases consensus change by 3.6 percentage points compared to the generic baseline (10.65\% vs 7.01\%) while being 77\% faster in response (15.98s vs. 3.71s). In conclusion, ProMediate provides a rigorous, theory-grounded testbed to advance the development of proactive, socially intelligent agents.

**arXiv ID:** 2510.25224
</details>

<details>
<summary><strong>ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards</strong> - Shiyu Li, Yang Tang, Yifan Wang, Peiming Li, Xi Chen - [[pdf]](https://arxiv.org/pdf/2510.00568)</summary>

**Abstract:** Search agents powered by Large Language Models (LLMs) have demonstrated significant potential in tackling knowledge-intensive tasks. Reinforcement learning (RL) has emerged as a powerful paradigm for training these agents to perform complex, multi-step reasoning. However, prior RL-based methods often rely on sparse or rule-based rewards, which can lead agents to commit to suboptimal or erroneous reasoning paths without the ability to recover. To address these limitations, we propose ReSeek, a novel self-correcting framework for training search agents. Our framework introduces a self-correction mechanism that empowers the agent to dynamically identify and recover from erroneous search paths during an episode. By invoking a special JUDGE action, the agent can judge the information and re-plan its search strategy. To guide this process, we design a dense, instructive process reward function, which decomposes into a correctness reward for retrieving factual information and a utility reward for finding information genuinely useful for the query. Furthermore, to mitigate the risk of data contamination in existing datasets, we introduce FictionalHot, a new and challenging benchmark with recently curated questions requiring complex reasoning. Being intuitively reasonable and practically simple, extensive experiments show that agents trained with ReSeek significantly outperform SOTA baselines in task success rate and path faithfulness.

**arXiv ID:** 2510.00568
</details>

<details>
<summary><strong>Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards</strong> - Shangyu Xing, Siyuan Wang, Chenyuan Yang, Xinyu Dai, Xiang Ren - [[pdf]](https://arxiv.org/pdf/2510.24302)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR), particularly with algorithms like Group Relative Policy Optimization (GRPO), has proven highly effective in enhancing the reasoning capabilities of large language models. However, a critical bottleneck in current pipelines lies in the limited diversity of sampled trajectories during group rollouts. Homogeneous trajectories and their associated rewards would diminish the return signals for policy updates, thereby hindering effective policy learning. This lack of diversity stems primarily from token-level stochastic sampling, where local variations are likely to collapse into near-identical reasoning paths. To address this limitation, we propose Lookahead Tree-Based Rollouts (LATR), a novel rollout strategy designed to explicitly promotes trajectory-level diversity by enforcing branching into different candidate tokens likely to yield distinct continuations. Specifically, LATR iteratively operates in three stages: (1) branching at high-uncertainty generation steps, (2) performing lookahead simulation for each new branch, and (3) pruning branches that exhibits prolonged similarity during simulation. Compared with stochastic Sampling, LATR accelerates policy learning by 131% on average and improves final pass@1 performance by 4.2% on both GRPO and Dynamic sAmpling Policy Optimization (DAPO) algorithms across different reasoning tasks. Our code and data are publicly available at this https URL.

**arXiv ID:** 2510.24302
</details>

<details>
<summary><strong>OpenReward: Learning to Reward Long-form Agentic Tasks via Reinforcement Learning</strong> - Ziyou Hu, Zhengliang Shi, Minghang Zhu, Haitao Li, Teng Sun, Pengjie Ren, Suzan Verberne, Zhaochun Ren - [[pdf]](https://arxiv.org/pdf/2510.24636)</summary>

**Abstract:** Reward models (RMs) have become essential for aligning large language models (LLMs), serving as scalable proxies for human evaluation in both training and inference. However, existing RMs struggle on knowledge-intensive and long-form tasks, where evaluating correctness requires grounding beyond the model's internal knowledge. This limitation hinders them from reliably discriminating subtle quality differences, especially when external evidence is necessary. To address this, we introduce OpenRM, a tool-augmented long-form reward model that systematically judges open-ended responses by invoking external tools to gather relevant evidence. We train OpenRM with Group Relative Policy Optimization (GRPO) on over 27K synthesized pairwise examples generated through a controllable data synthesis framework. The training objective jointly supervises intermediate tool usage and final outcome accuracy, incentivizing our reward model to learn effective evidence-based judgment strategies. Extensive experiments on three newly-collected datasets and two widely-used benchmarks demonstrate that OpenRM substantially outperforms existing reward modeling approaches. As a further step, we integrate OpenRM into both inference-time response selection and training-time data selection. This yields consistent gains in downstream LLM alignment tasks, highlighting the potential of tool-augmented reward models for scaling reliable long-form evaluation.

**arXiv ID:** 2510.24636
</details>

<details>
<summary><strong>Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems</strong> - Christian Walder, Deep Karkhanis - [[pdf]](https://arxiv.org/pdf/2505.15201)</summary>

**Abstract:** Reinforcement Learning (RL) algorithms sample multiple n>1 solution attempts for each problem and reward them independently. This optimizes for pass@1 performance and prioritizes the strength of isolated samples at the expense of the diversity and collective utility of sets of samples. This under-utilizes the sampling capacity, limiting exploration and eventual improvement on harder examples. As a fix, we propose Pass-at-k Policy Optimization (PKPO), a transformation on the final rewards which leads to direct optimization of pass@k performance, thus optimizing for sets of samples that maximize reward when considered jointly. Our contribution is to derive novel low variance unbiased estimators for pass@k and its gradient, in both the binary and continuous reward settings. We show optimization with our estimators reduces to standard RL with rewards that have been jointly transformed by a stable and efficient transformation function.
While previous efforts are restricted to k=n, ours is the first to enable robust optimization of pass@k for any arbitrary k <= n. Moreover, instead of trading off pass@1 performance for pass@k gains, our method allows annealing k during training, optimizing both metrics and often achieving strong pass@1 numbers alongside significant pass@k gains.
We validate our reward transformations on toy experiments, which reveal the variance reducing properties of our formulations. We also include real-world examples using the open-source LLM, GEMMA-2. We find that our transformation effectively optimizes for the target k. Furthermore, higher k values enable solving more and harder problems, while annealing k boosts both the pass@1 and pass@k . Crucially, for challenging task sets where conventional pass@1 optimization stalls, our pass@k approach unblocks learning, likely due to better exploration by prioritizing joint utility over the utility of individual samples.

**arXiv ID:** 2505.15201
</details>

<details>
<summary><strong>Enhancing Hierarchical Reinforcement Learning through Change Point Detection in Time Series</strong> - Hemanath Arumugam, Falong Fan, Bo Liu - [[pdf]](https://arxiv.org/pdf/2510.24988)</summary>

**Abstract:** Hierarchical Reinforcement Learning (HRL) enhances the scalability of decision-making in long-horizon tasks by introducing temporal abstraction through options-policies that span multiple timesteps. Despite its theoretical appeal, the practical implementation of HRL suffers from the challenge of autonomously discovering semantically meaningful subgoals and learning optimal option termination boundaries. This paper introduces a novel architecture that integrates a self-supervised, Transformer-based Change Point Detection (CPD) module into the Option-Critic framework, enabling adaptive segmentation of state trajectories and the discovery of options. The CPD module is trained using heuristic pseudo-labels derived from intrinsic signals to infer latent shifts in environment dynamics without external supervision. These inferred change-points are leveraged in three critical ways: (i) to serve as supervisory signals for stabilizing termination function gradients, (ii) to pretrain intra-option policies via segment-wise behavioral cloning, and (iii) to enforce functional specialization through inter-option divergence penalties over CPD-defined state partitions. The overall optimization objective enhances the standard actor-critic loss using structure-aware auxiliary losses. In our framework, option discovery arises naturally as CPD-defined trajectory segments are mapped to distinct intra-option policies, enabling the agent to autonomously partition its behavior into reusable, semantically meaningful skills. Experiments on the Four-Rooms and Pinball tasks demonstrate that CPD-guided agents exhibit accelerated convergence, higher cumulative returns, and significantly improved option specialization. These findings confirm that integrating structural priors via change-point segmentation leads to more interpretable, sample-efficient, and robust hierarchical policies in complex environments.

**arXiv ID:** 2510.24988
</details>

<details>
<summary><strong>Reinforcement Learning Teachers of Test Time Scaling</strong> - Edoardo Cetin, Tianyu Zhao, Yujin Tang - [[pdf]](https://arxiv.org/pdf/2506.08388)</summary>

**Abstract:** Training reasoning language models (LMs) with reinforcement learning (RL) for one-hot correctness inherently relies on the LM being able to explore and solve its task with some chance at initialization. Furthermore, a key use case of reasoning LMs is to act as teachers for distilling new students and cold-starting future RL iterations rather than being deployed themselves. From these considerations, we introduce a new framework that avoids RL's exploration challenge by training a new class of Reinforcement-Learned Teachers (RLTs) focused on yielding the most effective downstream distillation. RLTs are prompted with both the question and solution to each problem, and tasked to simply "connect-the-dots" with detailed explanations tailored for their students. We train RLTs with dense rewards obtained by feeding each explanation to the student and testing its understanding of the problem's solution. In practice, the raw outputs of a 7B RLT provide higher final performance on competition and graduate-level tasks than existing distillation and cold-starting pipelines that collect and postprocess the reasoning traces of orders of magnitude larger LMs. Furthermore, RLTs maintain their effectiveness when training larger students and when applied zero-shot to out-of-distribution tasks, unlocking new levels of efficiency and re-usability for the RL reasoning framework. Code available at: this https URL

**arXiv ID:** 2506.08388
</details>

<details>
<summary><strong>SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning</strong> - Huanyu Liu, Jia Li, Hao Zhu, Kechi Zhang, Yihong Dong, Ge Li - [[pdf]](https://arxiv.org/pdf/2505.16368)</summary>

**Abstract:** How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard.
To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLMs reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions.
We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research.

**arXiv ID:** 2505.16368
</details>

<details>
<summary><strong>Sim-to-Real Gentle Manipulation of Deformable and Fragile Objects with Stress-Guided Reinforcement Learning</strong> - Kei Ikemura, Yifei Dong, David Blanco-Mulero, Alberta Longhini, Li Chen, Florian T. Pokorny - [[pdf]](https://arxiv.org/pdf/2510.25405)</summary>

**Abstract:** Robotic manipulation of deformable and fragile objects presents significant challenges, as excessive stress can lead to irreversible damage to the object. While existing solutions rely on accurate object models or specialized sensors and grippers, this adds complexity and often lacks generalization. To address this problem, we present a vision-based reinforcement learning approach that incorporates a stress-penalized reward to discourage damage to the object explicitly. In addition, to bootstrap learning, we incorporate offline demonstrations as well as a designed curriculum progressing from rigid proxies to deformables. We evaluate the proposed method in both simulated and real-world scenarios, showing that the policy learned in simulation can be transferred to the real world in a zero-shot manner, performing tasks such as picking up and pushing tofu. Our results show that the learned policies exhibit a damage-aware, gentle manipulation behavior, demonstrating their effectiveness by decreasing the stress applied to fragile objects by 36.5% while achieving the task goals, compared to vanilla RL policies.

**arXiv ID:** 2510.25405
</details>

<details>
<summary><strong>Federated Deep Reinforcement Learning for Privacy-Preserving Robotic-Assisted Surgery</strong> - Sana Hafeez, Sundas Rafat Mulkana, Muhammad Ali Imran, Michele Sevegnani - [[pdf]](https://arxiv.org/pdf/2505.12153)</summary>

**Abstract:** The integration of Reinforcement Learning (RL) into robotic-assisted surgery (RAS) holds significant promise for advancing surgical precision, adaptability, and autonomous decision-making. However, the development of robust RL models in clinical settings is hindered by key challenges, including stringent patient data privacy regulations, limited access to diverse surgical datasets, and high procedural variability. To address these limitations, this paper presents a Federated Deep Reinforcement Learning (FDRL) framework that enables decentralized training of RL models across multiple healthcare institutions without exposing sensitive patient information. A central innovation of the proposed framework is its dynamic policy adaptation mechanism, which allows surgical robots to select and tailor patient-specific policies in real-time, thereby ensuring personalized and Optimised interventions. To uphold rigorous privacy standards while facilitating collaborative learning, the FDRL framework incorporates secure aggregation, differential privacy, and homomorphic encryption techniques. Experimental results demonstrate a 60\% reduction in privacy leakage compared to conventional methods, with surgical precision maintained within a 1.5\% margin of a centralized baseline. This work establishes a foundational approach for adaptive, secure, and patient-centric AI-driven surgical robotics, offering a pathway toward clinical translation and scalable deployment across diverse healthcare environments.

**arXiv ID:** 2505.12153
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
