# Agent arXiv Daily

**Last Updated:** 2026-02-04 03:40:50

**Total Papers:** 117

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>FIRE-Bench: Evaluating Agents on the Rediscovery of Scientific Insights</strong> - Zhen Wang, Fan Bai, Zhongyan Luo, Jinyan Su, Kaiser Sun, Xinle Yu, Jieyuan Liu, Kun Zhou, Claire Cardie, Mark Dredze, Eric P. Xing, Zhiting Hu - [[pdf]](https://arxiv.org/pdf/2602.02905)</summary>

**Abstract:** Autonomous agents powered by large language models (LLMs) promise to accelerate scientific discovery end-to-end, but rigorously evaluating their capacity for verifiable discovery remains a central challenge. Existing benchmarks face a trade-off: they either heavily rely on LLM-as-judge evaluations of automatically generated research outputs or optimize convenient yet isolated performance metrics that provide coarse proxies for scientific insight. To address this gap, we introduce FIRE-Bench (Full-cycle Insight Rediscovery Evaluation), a benchmark that evaluates agents through the rediscovery of established findings from recent, high-impact machine learning research. Agents are given only a high-level research question extracted from a published, verified study and must autonomously explore ideas, design experiments, implement code, execute their plans, and derive conclusions supported by empirical evidence. We evaluate a range of state-of-the-art agents with frontier LLMs backbones like gpt-5 on FIRE-Bench. Our results show that full-cycle scientific research remains challenging for current agent systems: even the strongest agents achieve limited rediscovery success (<50 F1), exhibit high variance across runs, and display recurring failure modes in experimental design, execution, and evidence-based reasoning. FIRE-Bench provides a rigorous and diagnostic framework for measuring progress toward reliable agent-driven scientific discovery.

**arXiv ID:** 2602.02905
</details>

<details>
<summary><strong>MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning</strong> - Shengyuan Liu, Liuxin Bao, Qi Yang, Wanting Geng, Boyun Zheng, Chenxin Li, Wenting Chen, Houwen Peng, Yixuan Yuan - [[pdf]](https://arxiv.org/pdf/2602.03320)</summary>

**Abstract:** Medical image segmentation is evolving from task-specific models toward generalizable frameworks. Recent research leverages Multi-modal Large Language Models (MLLMs) as autonomous agents, employing reinforcement learning with verifiable reward (RLVR) to orchestrate specialized tools like the Segment Anything Model (SAM). However, these approaches often rely on single-turn, rigid interaction strategies and lack process-level supervision during training, which hinders their ability to fully exploit the dynamic potential of interactive tools and leads to redundant actions. To bridge this gap, we propose MedSAM-Agent, a framework that reformulates interactive segmentation as a multi-step autonomous decision-making process. First, we introduce a hybrid prompting strategy for expert-curated trajectory generation, enabling the model to internalize human-like decision heuristics and adaptive refinement strategies. Furthermore, we develop a two-stage training pipeline that integrates multi-turn, end-to-end outcome verification with a clinical-fidelity process reward design to promote interaction parsimony and decision efficiency. Extensive experiments across 6 medical modalities and 21 datasets demonstrate that MedSAM-Agent achieves state-of-the-art performance, effectively unifying autonomous medical reasoning with robust, iterative optimization. Code is available \href{this https URL}{here}.

**arXiv ID:** 2602.03320
</details>

<details>
<summary><strong>ENGRAM: Effective, Lightweight Memory Orchestration for Conversational Agents</strong> - Daivik Patel, Shrenik Patel - [[pdf]](https://arxiv.org/pdf/2511.12960)</summary>

**Abstract:** Large language models (LLMs) deployed in user-facing applications require long-horizon consistency: the ability to remember prior interactions, respect user preferences, and ground reasoning in past events. However, contemporary memory systems often adopt complex architectures such as knowledge graphs, multi-stage retrieval pipelines, and OS-style schedulers, which introduce engineering complexity and reproducibility challenges. We present ENGRAM, a lightweight memory system that organizes conversation into three canonical memory types (episodic, semantic, and procedural) through a single router and retriever. Each user turn is converted into typed memory records with normalized schemas and embeddings and stored in a database. At query time, the system retrieves top-k dense neighbors for each type, merges results with simple set operations, and provides the most relevant evidence as context to the model. ENGRAM attains state-of-the-art results on LoCoMo, a multi-session conversational QA benchmark for long-horizon memory, and exceeds the full-context baseline by 15 points on LongMemEval while using only about 1% of the tokens. These results show that careful memory typing and straightforward dense retrieval can enable effective long-term memory management in language models without requiring complex architectures.

**arXiv ID:** 2511.12960
</details>

<details>
<summary><strong>Formulating Reinforcement Learning for Human-Robot Collaboration through Off-Policy Evaluation</strong> - Saurav Singh, Rodney Sanchez, Alexander Ororbia, Jamison Heard - [[pdf]](https://arxiv.org/pdf/2602.02530)</summary>

**Abstract:** Reinforcement learning (RL) has the potential to transform real-world decision-making systems by enabling autonomous agents to learn from experience. Deploying RL in real-world settings, especially in the context of human-robot interaction, requires defining state representations and reward functions, which are critical for learning efficiency and policy performance. Traditional RL approaches often rely on domain expertise and trial-and-error, necessitating extensive human involvement as well as direct interaction with the environment, which can be costly and impractical, especially in complex and safety-critical applications. This work proposes a novel RL framework that leverages off-policy evaluation (OPE) for state space and reward function selection, using only logged interaction data. This approach eliminates the need for real-time access to the environment or human-in-the-loop feedback, greatly reducing the dependency on costly real-time interactions. The proposed approach systematically evaluates multiple candidate state representations and reward functions by training offline RL agents and applying OPE to estimate policy performance. The optimal state space and reward function are selected based on their ability to produce high-performing policies under OPE metrics. Our method is validated on two environments: the Lunar Lander environment by OpenAI Gym, which provides a controlled setting for assessing state space and reward function selection, and a NASA-MATB-II human subjects study environment, which evaluates the approach's real-world applicability to human-robot teaming scenarios. This work enhances the feasibility and scalability of offline RL for real-world environments by automating critical RL design decisions through a data-driven OPE-based evaluation, enabling more reliable, effective, and sustainable RL formulation for complex human-robot interaction settings.

**arXiv ID:** 2602.02530
</details>

<details>
<summary><strong>Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL</strong> - Julian Lemmel, Felix Resch, Mónika Farsang, Ramin Hasani, Daniela Rus, Radu Grosu - [[pdf]](https://arxiv.org/pdf/2602.02236)</summary>

**Abstract:** Deploying pretrained policies in real-world applications presents substantial challenges that fundamentally limit the practical applicability of learning-based control systems. When autonomous systems encounter environmental changes in system dynamics, sensor drift, or task objectives, fixed policies rapidly degrade in performance. We show that employing Real-Time Recurrent Reinforcement Learning (RTRRL), a biologically plausible algorithm for online adaptation, can effectively fine-tune a pretrained policy to improve autonomous agents' performance on driving tasks. We further show that RTRRL synergizes with a recent biologically inspired recurrent network model, the Liquid-Resistance Liquid-Capacitance RNN. We demonstrate the effectiveness of this closed-loop approach in a simulated CarRacing environment and in a real-world line-following task with a RoboRacer car equipped with an event camera.

**arXiv ID:** 2602.02236
</details>

<details>
<summary><strong>Is It Possible to Make Chatbots Virtuous? Investigating a Virtue-Based Design Methodology Applied to LLMs</strong> - Matthew P. Lad, Louisa Conwill, Megan Levis Scheirer - [[pdf]](https://arxiv.org/pdf/2602.03155)</summary>

**Abstract:** With the rapid growth of Large Language Models (LLMs), criticism of their societal impact has also grown. Work in Responsible AI (RAI) has focused on the development of AI systems aimed at reducing harm. Responding to RAI's criticisms and the need to bring the wisdom traditions into HCI, we apply Conwill et al.'s Virtue-Guided Technology Design method to LLMs. We cataloged new ethical design patterns for LLMs and evaluated them through interviews with technologists. Participants valued that the patterns provided more accuracy and robustness, better safety, new research opportunities, increased access and control, and reduced waste. Their concerns were that the patterns could be vulnerable to jailbreaking, were generalizing models too widely, and had potential implementation issues. Overall, participants reacted positively while also acknowledging the tradeoffs involved in ethical LLM design.

**arXiv ID:** 2602.03155
</details>

<details>
<summary><strong>"Having Lunch Now": Understanding How Users Engage with a Proactive Agent for Daily Planning and Self-Reflection</strong> - Adnan Abbas, Caleb Wohn, Arnav Jagtap, Eugenia H Rho, Young-Ho Kim, Sang Won Lee - [[pdf]](https://arxiv.org/pdf/2509.24073)</summary>

**Abstract:** Conversational agents have been studied as tools to scaffold planning and self-reflection for productivity and well-being. While prior work has demonstrated positive outcomes, we still lack a clear understanding of what drives these results and how users behave and communicate with agents that act as coaches rather than assistants. Such understanding is critical for designing interactions in which agents foster meaningful behavioral change. We conducted a 14-day longitudinal study with 12 participants using a proactive agent that initiated regular check-ins to support daily planning and reflection. Our findings reveal diverse interaction patterns: participants accepted or negotiated suggestions, developed shared mental models, reported progress, and at times resisted or disengaged. We also identified problematic aspects of the agent's behavior, including rigidity, premature turn-taking, and overpromising. Our work contributes to understanding how people interact with a proactive, coach-like agent and offers design considerations for facilitating effective behavioral change.

**arXiv ID:** 2509.24073
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (20 papers)</h2></summary>

<details>
<summary><strong>MARS: Modular Agent with Reflective Search for Automated AI Research</strong> - Jiefeng Chen, Bhavana Dalvi Mishra, Jaehyun Nam, Rui Meng, Tomas Pfister, Jinsung Yoon - [[pdf]](https://arxiv.org/pdf/2602.02660)</summary>

**Abstract:** Automating AI research differs from general software engineering due to computationally expensive evaluation (e.g., model training) and opaque performance attribution. Current LLM-based agents struggle here, often generating monolithic scripts that ignore execution costs and causal factors. We introduce MARS (Modular Agent with Reflective Search), a framework optimized for autonomous AI research. MARS relies on three pillars: (1) Budget-Aware Planning via cost-constrained Monte Carlo Tree Search (MCTS) to explicitly balance performance with execution expense; (2) Modular Construction, employing a "Design-Decompose-Implement" pipeline to manage complex research repositories; and (3) Comparative Reflective Memory, which addresses credit assignment by analyzing solution differences to distill high-signal insights. MARS achieves state-of-the-art performance among open-source frameworks on MLE-Bench under comparable settings, maintaining competitiveness with the global leaderboard's top methods. Furthermore, the system exhibits qualitative "Aha!" moments, where 63% of all utilized lessons originate from cross-branch transfer, demonstrating that the agent effectively generalizes insights across search paths.

**arXiv ID:** 2602.02660
</details>

<details>
<summary><strong>AutoSizer: Automatic Sizing of Analog and Mixed-Signal Circuits via Large Language Model (LLM) Agents</strong> - Xi Yu, Dmitrii Torbunov, Soumyajit Mandal, Yihui Ren - [[pdf]](https://arxiv.org/pdf/2602.02849)</summary>

**Abstract:** The design of Analog and Mixed-Signal (AMS) integrated circuits remains heavily reliant on expert knowledge, with transistor sizing a major bottleneck due to nonlinear behavior, high-dimensional design spaces, and strict performance constraints. Existing Electronic Design Automation (EDA) methods typically frame sizing as static black-box optimization, resulting in inefficient and less robust solutions. Although Large Language Models (LLMs) exhibit strong reasoning abilities, they are not suited for precise numerical optimization in AMS sizing. To address this gap, we propose AutoSizer, a reflective LLM-driven meta-optimization framework that unifies circuit understanding, adaptive search-space construction, and optimization orchestration in a closed loop. It employs a two-loop optimization framework, with an inner loop for circuit sizing and an outer loop that analyzes optimization dynamics and constraints to iteratively refine the search space from simulation feedback. We further introduce AMS-SizingBench, an open benchmark comprising 24 diverse AMS circuits in SKY130 CMOS technology, designed to evaluate adaptive optimization policies under realistic simulator-based constraints. AutoSizer experimentally achieves higher solution quality, faster convergence, and higher success rate across varying circuit difficulties, outperforming both traditional optimization methods and existing LLM-based agents.

**arXiv ID:** 2602.02849
</details>

<details>
<summary><strong>Agent Alpha: Tree Search Unifying Generation, Exploration and Evaluation for Computer-Use Agents</strong> - Sizhe Tang, Rongqian Chen, Tian Lan - [[pdf]](https://arxiv.org/pdf/2602.02995)</summary>

**Abstract:** While scaling test-time compute through trajectory-level sampling has significantly improved Graphical User Interface (GUI) agents, the lack of regressive ability prevents the reuse of partial successes and the recovery from early missteps. In this paper, we introduce Agent Alpha, a unified framework that synergizes generation, exploration, and evaluation through step-level Monte Carlo Tree Search (MCTS). It enables active modeling or exploiting structures of the planning space. By integrating alpha-UCT guided search into the interaction loop, Agent Alpha enables deliberate planning, facilitating early pruning of suboptimal branches and efficient prefix reuse. We also employ comparison-driven evaluation to mitigate absolute scoring biases and diversity-constrained expansion to maintain a compact, informative search space. Regret bound of alpha-UCT is analyzed. On the OSWorld benchmark, Agent Alpha achieves a state-of-the-art success rate of $\sim 77\%$, significantly outperforming trajectory-level baselines under equivalent compute.

**arXiv ID:** 2602.02995
</details>

<details>
<summary><strong>Risky-Bench: Probing Agentic Safety Risks under Real-World Deployment</strong> - Jingnan Zheng, Yanzhen Luo, Jingjun Xu, Bingnan Liu, Yuxin Chen, Chenhang Cui, Gelei Deng, Chaochao Lu, Xiang Wang, An Zhang, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2602.03100)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly deployed as agents that operate in real-world environments, introducing safety risks beyond linguistic harm. Existing agent safety evaluations rely on risk-oriented tasks tailored to specific agent settings, resulting in limited coverage of safety risk space and failing to assess agent safety behavior during long-horizon, interactive task execution in complex real-world deployments. Moreover, their specialization to particular agent settings limits adaptability across diverse agent configurations. To address these limitations, we propose Risky-Bench, a framework that enables systematic agent safety evaluation grounded in real-world deployment. Risky-Bench organizes evaluation around domain-agnostic safety principles to derive context-aware safety rubrics that delineate safety space, and systematically evaluates safety risks across this space through realistic task execution under varying threat assumptions. When applied to life-assist agent settings, Risky-Bench uncovers substantial safety risks in state-of-the-art agents under realistic execution conditions. Moreover, as a well-structured evaluation pipeline, Risky-Bench is not confined to life-assist scenarios and can be adapted to other deployment settings to construct environment-specific safety evaluations, providing an extensible methodology for agent safety assessment.

**arXiv ID:** 2602.03100
</details>

<details>
<summary><strong>The Necessity of a Unified Framework for LLM-Based Agent Evaluation</strong> - Pengyu Zhu, Li Sun, Philip S. Yu, Sen Su - [[pdf]](https://arxiv.org/pdf/2602.03238)</summary>

**Abstract:** With the advent of Large Language Models (LLMs), general-purpose agents have seen fundamental advancements. However, evaluating these agents presents unique challenges that distinguish them from static QA benchmarks. We observe that current agent benchmarks are heavily confounded by extraneous factors, including system prompts, toolset configurations, and environmental dynamics. Existing evaluations often rely on fragmented, researcher-specific frameworks where the prompt engineering for reasoning and tool usage varies significantly, making it difficult to attribute performance gains to the model itself. Additionally, the lack of standardized environmental data leads to untraceable errors and non-reproducible results. This lack of standardization introduces substantial unfairness and opacity into the field. We propose that a unified evaluation framework is essential for the rigorous advancement of agent evaluation. To this end, we introduce a proposal aimed at standardizing agent evaluation.

**arXiv ID:** 2602.03238
</details>

<details>
<summary><strong>ToolTok: Tool Tokenization for Efficient and Generalizable GUI Agents</strong> - Xiaoce Wang, Guibin Zhang, Junzhe Li, Jinzhe Tu, Chun Li, Ming Li - [[pdf]](https://arxiv.org/pdf/2602.02548)</summary>

**Abstract:** Existing GUI agent models relying on coordinate-based one-step visual grounding struggle with generalizing to varying input resolutions and aspect ratios. Alternatives introduce coordinate-free strategies yet suffer from learning under severe data scarcity. To address the limitations, we propose ToolTok, a novel paradigm of multi-step pathfinding for GUI agents, where operations are modeled as a sequence of progressive tool usage. Specifically, we devise tools aligned with human interaction habits and represent each tool using learnable token embeddings. To enable efficient embedding learning under limited supervision, ToolTok introduces a semantic anchoring mechanism that grounds each tool with semantically related concepts as natural inductive bias. To further enable a pre-trained large language model to progressively acquire tool semantics, we construct an easy-to-hard curriculum consisting of three tasks: token definition question-answering, pure text-guided tool selection, and simplified visual pathfinding. Extensive experiments on multiple benchmarks show that ToolTok achieves superior performance among models of comparable scale (4B) and remains competitive with a substantially larger model (235B). Notably, these results are obtained using less than 1% of the training data required by other post-training approaches. In addition, ToolTok demonstrates strong generalization across unseen scenarios. Our training & inference code is open-source at this https URL.

**arXiv ID:** 2602.02548
</details>

<details>
<summary><strong>To Defend Against Cyber Attacks, We Must Teach AI Agents to Hack</strong> - Terry Yue Zhuo, Yangruibo Ding, Wenbo Guo, Ruijie Meng - [[pdf]](https://arxiv.org/pdf/2602.02595)</summary>

**Abstract:** For over a decade, cybersecurity has relied on human labor scarcity to limit attackers to high-value targets manually or generic automated attacks at scale. Building sophisticated exploits requires deep expertise and manual effort, leading defenders to assume adversaries cannot afford tailored attacks at scale. AI agents break this balance by automating vulnerability discovery and exploitation across thousands of targets, needing only small success rates to remain profitable. Current developers focus on preventing misuse through data filtering, safety alignment, and output guardrails. Such protections fail against adversaries who control open-weight models, bypass safety controls, or develop offensive capabilities independently. We argue that AI-agent-driven cyber attacks are inevitable, requiring a fundamental shift in defensive strategy. In this position paper, we identify why existing defenses cannot stop adaptive adversaries and demonstrate that defenders must develop offensive security intelligence. We propose three actions for building frontier offensive AI capabilities responsibly. First, construct comprehensive benchmarks covering the full attack lifecycle. Second, advance from workflow-based to trained agents for discovering in-wild vulnerabilities at scale. Third, implement governance restricting offensive agents to audited cyber ranges, staging release by capability tier, and distilling findings into safe defensive-only agents. We strongly recommend treating offensive AI capabilities as essential defensive infrastructure, as containing cybersecurity risks requires mastering them in controlled settings before adversaries do.

**arXiv ID:** 2602.02595
</details>

<details>
<summary><strong>Label Curation Using Agentic AI</strong> - Subhodeep Ghosh, Bayan Divaaniaazar, Md Ishat-E-Rabban, Spencer Clarke, Senjuti Basu Roy - [[pdf]](https://arxiv.org/pdf/2602.02564)</summary>

**Abstract:** Data annotation is essential for supervised learning, yet producing accurate, unbiased, and scalable labels remains challenging as datasets grow in size and modality. Traditional human-centric pipelines are costly, slow, and prone to annotator variability, motivating reliability-aware automated annotation. We present AURA (Agentic AI for Unified Reliability Modeling and Annotation Aggregation), an agentic AI framework for large-scale, multi-modal data annotation. AURA coordinates multiple AI agents to generate and validate labels without requiring ground truth. At its core, AURA adapts a classical probabilistic model that jointly infers latent true labels and annotator reliability via confusion matrices, using Expectation-Maximization to reconcile conflicting annotations and aggregate noisy predictions. Across the four benchmark datasets evaluated, AURA achieves accuracy improvements of up to 5.8% over baseline. In more challenging settings with poor quality annotators, the improvement is up to 50% over baseline. AURA also accurately estimates the reliability of annotators, allowing assessment of annotator quality even without any pre-validation steps.

**arXiv ID:** 2602.02564
</details>

<details>
<summary><strong>InfMem: Learning System-2 Memory Control for Long-Context Agent</strong> - Xinyu Wang, Mingze Li, Peng Lu, Xiao-Wen Chang, Lifeng Shang, Jinping Li, Fei Mi, Prasanna Parthasarathi, Yufei Cui - [[pdf]](https://arxiv.org/pdf/2602.02704)</summary>

**Abstract:** Reasoning over ultra-long documents requires synthesizing sparse evidence scattered across distant segments under strict memory constraints. While streaming agents enable scalable processing, their passive memory update strategy often fails to preserve low-salience bridging evidence required for multi-hop reasoning. We propose InfMem, a control-centric agent that instantiates System-2-style control via a PreThink-Retrieve-Write protocol. InfMem actively monitors evidence sufficiency, performs targeted in-document retrieval, and applies evidence-aware joint compression to update a bounded memory. To ensure reliable control, we introduce a practical SFT-to-RL training recipe that aligns retrieval, writing, and stopping decisions with end-task correctness. On ultra-long QA benchmarks from 32k to 1M tokens, InfMem consistently outperforms MemAgent across backbones. Specifically, InfMem improves average absolute accuracy by +10.17, +11.84, and +8.23 points on Qwen3-1.7B, Qwen3-4B, and Qwen2.5-7B, respectively, while reducing inference time by $3.9\times$ on average (up to $5.1\times$) via adaptive early stopping.

**arXiv ID:** 2602.02704
</details>

<details>
<summary><strong>AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback</strong> - Zhitao Gao, Jie Ma, Xuhong Li, Pengyu Li, Ning Qu, Yaqiang Wu, Hui Liu, Jun Liu - [[pdf]](https://arxiv.org/pdf/2602.03084)</summary>

**Abstract:** Large Language Models (LLMs) have achieved significant success in complex reasoning but remain bottlenecked by reliance on expert-annotated data and external verifiers. While existing self-evolution paradigms aim to bypass these constraints, they often fail to identify the optimal learning zone and risk reinforcing collective hallucinations and incorrect priors through flawed internal feedback. To address these challenges, we propose \underline{A}utonomous \underline{E}volutionary \underline{R}easoning \underline{O}ptimization (AERO), an unsupervised framework that achieves autonomous reasoning evolution by internalizing self-questioning, answering, and criticism within a synergistic dual-loop system. Inspired by the \textit{Zone of Proximal Development (ZPD)} theory, AERO utilizes entropy-based positioning to target the ``solvability gap'' and employs Independent Counterfactual Correction for robust verification. Furthermore, we introduce a Staggered Training Strategy to synchronize capability growth across functional roles and prevent curriculum collapse. Extensive evaluations across nine benchmarks spanning three domains demonstrate that AERO achieves average performance improvements of 4.57\% on Qwen3-4B-Base and 5.10\% on Qwen3-8B-Base, outperforming competitive baselines. Code is available at this https URL.

**arXiv ID:** 2602.03084
</details>

<details>
<summary><strong>Accurate Failure Prediction in Agents Does Not Imply Effective Failure Prevention</strong> - Rakshith Vasudev, Melisa Russak, Dan Bikel, Waseem Alshikh - [[pdf]](https://arxiv.org/pdf/2602.03338)</summary>

**Abstract:** Proactive interventions by LLM critic models are often assumed to improve reliability, yet their effects at deployment time are poorly understood. We show that a binary LLM critic with strong offline accuracy (AUROC 0.94) can nevertheless cause severe performance degradation, inducing a 26 percentage point (pp) collapse on one model while affecting another by near zero pp. This variability demonstrates that LLM critic accuracy alone is insufficient to determine whether intervention is safe.
We identify a disruption-recovery tradeoff: interventions may recover failing trajectories but also disrupt trajectories that would have succeeded. Based on this insight, we propose a pre-deployment test that uses a small pilot of 50 tasks to estimate whether intervention is likely to help or harm, without requiring full deployment. Across benchmarks, the test correctly anticipates outcomes: intervention degrades performance on high-success tasks (0 to -26 pp), while yielding a modest improvement on the high-failure ALFWorld benchmark (+2.8 pp, p=0.014). The primary value of our framework is therefore identifying when not to intervene, preventing severe regressions before deployment.

**arXiv ID:** 2602.03338
</details>

<details>
<summary><strong>A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces</strong> - Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan Wang, Pengyu Wang, Xiaorui Wang, Zhendong Mao - [[pdf]](https://arxiv.org/pdf/2602.03442)</summary>

**Abstract:** Frontier language models have demonstrated strong reasoning and long-horizon tool-use capabilities. However, existing RAG systems fail to leverage these capabilities. They still rely on two paradigms: (1) designing an algorithm that retrieves passages in a single shot and concatenates them into the model's input, or (2) predefining a workflow and prompting the model to execute it step-by-step. Neither paradigm allows the model to participate in retrieval decisions, preventing efficient scaling with model improvements. In this paper, we introduce A-RAG, an Agentic RAG framework that exposes hierarchical retrieval interfaces directly to the model. A-RAG provides three retrieval tools: keyword search, semantic search, and chunk read, enabling the agent to adaptively search and retrieve information across multiple granularities. Experiments on multiple open-domain QA benchmarks show that A-RAG consistently outperforms existing approaches with comparable or lower retrieved tokens, demonstrating that A-RAG effectively leverages model capabilities and dynamically adapts to different RAG tasks. We further systematically study how A-RAG scales with model size and test-time compute. We will release our code and evaluation suite to facilitate future research. Code and evaluation suite are available at this https URL.

**arXiv ID:** 2602.03442
</details>

<details>
<summary><strong>SWE-World: Building Software Engineering Agents in Docker-Free Environments</strong> - Shuang Sun, Huatong Song, Lisheng Huang, Jinhao Jiang, Ran Le, Zhihao Lv, Zongchao Chen, Yiwen Hu, Wenyang Luo, Wayne Xin Zhao, Yang Song, Hongteng Xu, Tao Zhang, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2602.03419)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled software engineering agents to tackle complex code modification tasks. Most existing approaches rely on execution feedback from containerized environments, which require dependency-complete setup and physical execution of programs and tests. While effective, this paradigm is resource-intensive and difficult to maintain, substantially complicating agent training and limiting scalability. We propose SWE-World, a Docker-free framework that replaces physical execution environments with a learned surrogate for training and evaluating software engineering agents. SWE-World leverages LLM-based models trained on real agent-environment interaction data to predict intermediate execution outcomes and final test feedback, enabling agents to learn without interacting with physical containerized environments. This design preserves the standard agent-environment interaction loop while eliminating the need for costly environment construction and maintenance during agent optimization and evaluation. Furthermore, because SWE-World can simulate the final evaluation outcomes of candidate trajectories without real submission, it enables selecting the best solution among multiple test-time attempts, thereby facilitating effective test-time scaling (TTS) in software engineering tasks. Experiments on SWE-bench Verified demonstrate that SWE-World raises Qwen2.5-Coder-32B from 6.2\% to 52.0\% via Docker-free SFT, 55.0\% with Docker-free RL, and 68.2\% with further TTS. The code is available at this https URL

**arXiv ID:** 2602.03419
</details>

<details>
<summary><strong>SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training</strong> - Huatong Song, Lisheng Huang, Shuang Sun, Jinhao Jiang, Ran Le, Daixuan Cheng, Guoxin Chen, Yiwen Hu, Zongchao Chen, Wayne Xin Zhao, Yang Song, Tao Zhang, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2602.03411)</summary>

**Abstract:** In this technical report, we present SWE-Master, an open-source and fully reproducible post-training framework for building effective software engineering agents. SWE-Master systematically explores the complete agent development pipeline, including teacher-trajectory synthesis and data curation, long-horizon SFT, RL with real execution feedback, and inference framework design. Starting from an open-source base model with limited initial SWE capability, SWE-Master demonstrates how systematical optimization method can elicit strong long-horizon SWE task solving abilities. We evaluate SWE-Master on SWE-bench Verified, a standard benchmark for realistic software engineering tasks. Under identical experimental settings, our approach achieves a resolve rate of 61.4\% with Qwen2.5-Coder-32B, substantially outperforming existing open-source baselines. By further incorporating test-time scaling~(TTS) with LLM-based environment feedback, SWE-Master reaches 70.8\% at TTS@8, demonstrating a strong performance potential. SWE-Master provides a practical and transparent foundation for advancing reproducible research on software engineering agents. The code is available at this https URL.

**arXiv ID:** 2602.03411
</details>

<details>
<summary><strong>WebSentinel: Detecting and Localizing Prompt Injection Attacks for Web Agents</strong> - Xilong Wang, Yinuo Liu, Zhun Wang, Dawn Song, Neil Gong - [[pdf]](https://arxiv.org/pdf/2602.03792)</summary>

**Abstract:** Prompt injection attacks manipulate webpage content to cause web agents to execute attacker-specified tasks instead of the user's intended ones. Existing methods for detecting and localizing such attacks achieve limited effectiveness, as their underlying assumptions often do not hold in the web-agent setting. In this work, we propose WebSentinel, a two-step approach for detecting and localizing prompt injection attacks in webpages. Given a webpage, Step I extracts \emph{segments of interest} that may be contaminated, and Step II evaluates each segment by checking its consistency with the webpage content as context. We show that WebSentinel is highly effective, substantially outperforming baseline methods across multiple datasets of both contaminated and clean webpages that we collected. Our code is available at: this https URL.

**arXiv ID:** 2602.03792
</details>

<details>
<summary><strong>Wiki Live Challenge: Challenging Deep Research Agents with Expert-Level Wikipedia Articles</strong> - Shaohan Wang, Benfeng Xu, Licheng Zhang, Mingxuan Du, Chiwei Zhu, Xiaorui Wang, Zhendong Mao, Yongdong Zhang - [[pdf]](https://arxiv.org/pdf/2602.01590)</summary>

**Abstract:** Deep Research Agents (DRAs) have demonstrated remarkable capabilities in autonomous information retrieval and report generation, showing great potential to assist humans in complex research tasks. Current evaluation frameworks primarily rely on LLM-generated references or LLM-derived evaluation dimensions. While these approaches offer scalability, they often lack the reliability of expert-verified content and struggle to provide objective, fine-grained assessments of critical dimensions. To bridge this gap, we introduce Wiki Live Challenge (WLC), a live benchmark that leverages the newest Wikipedia Good Articles (GAs) as expert-level references. Wikipedia's strict standards for neutrality, comprehensiveness, and verifiability serve as a great challenge for DRAs, with GAs representing the pinnacle of which. We curate a dataset of 100 recent Good Articles and propose Wiki Eval, a comprehensive evaluation framework comprising a fine-grained evaluation method with 39 criteria for writing quality and rigorous metrics for factual verifiability. Extensive experiments on various DRA systems demonstrate a significant gap between current DRAs and human expert-level Wikipedia articles, validating the effectiveness of WLC in advancing agent research. We release our benchmark at this https URL

**arXiv ID:** 2602.01590
</details>

<details>
<summary><strong>ARTIS: Agentic Risk-Aware Test-Time Scaling via Iterative Simulation</strong> - Xingshan Zeng, Lingzhi Wang, Weiwen Liu, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu - [[pdf]](https://arxiv.org/pdf/2602.01709)</summary>

**Abstract:** Current test-time scaling (TTS) techniques enhance large language model (LLM) performance by allocating additional computation at inference time, yet they remain insufficient for agentic settings, where actions directly interact with external environments and their effects can be irreversible and costly. We propose ARTIS, Agentic Risk-Aware Test-Time Scaling via Iterative Simulation, a framework that decouples exploration from commitment by enabling test-time exploration through simulated interactions prior to real-world execution. This design allows extending inference-time computation to improve action-level reliability and robustness without incurring environmental risk. We further show that naive LLM-based simulators struggle to capture rare but high-impact failure modes, substantially limiting their effectiveness for agentic decision making. To address this limitation, we introduce a risk-aware tool simulator that emphasizes fidelity on failure-inducing actions via targeted data generation and rebalanced training. Experiments on multi-turn and multi-step agentic benchmarks demonstrate that iterative simulation substantially improves agent reliability, and that risk-aware simulation is essential for consistently realizing these gains across models and tasks.

**arXiv ID:** 2602.01709
</details>

<details>
<summary><strong>CP-Agent: Agentic Constraint Programming</strong> - Stefan Szeider - [[pdf]](https://arxiv.org/pdf/2508.07468)</summary>

**Abstract:** The translation of natural language to formal constraint models requires expertise in the problem domain and modeling frameworks. To explore the effectiveness of agentic workflows, we propose CP-Agent, a Python coding agent that uses the ReAct framework with a persistent IPython kernel. We provide the relevant domain knowledge as a project prompt of under 50 lines. The algorithm works by iteratively executing code, observing the solver's feedback, and refining constraint models based on execution results.
We evaluate CP-Agent on 101 constraint programming problems from CP-Bench. We made minor changes to the benchmark to address systematic ambiguities in the problem specifications and errors in the ground-truth models. On the clarified benchmark, CP-Agent achieves perfect accuracy on all 101 problems. Our experiments show that minimal guidance outperforms detailed procedural scaffolding. Our experiments also show that explicit task management tools can have both positive and negative effects on focused modeling tasks.

**arXiv ID:** 2508.07468
</details>

<details>
<summary><strong>Clarify Before You Draw: Proactive Agents for Robust Text-to-CAD Generation</strong> - Bo Yuan, Zelin Zhao, Petr Molodyk, Bin Hu, Yongxin Chen - [[pdf]](https://arxiv.org/pdf/2602.03045)</summary>

**Abstract:** Large language models have recently enabled text-to-CAD systems that synthesize parametric CAD programs (e.g., CadQuery) from natural language prompts. In practice, however, geometric descriptions can be under-specified or internally inconsistent: critical dimensions may be missing and constraints may conflict. Existing fine-tuned models tend to reactively follow user instructions and hallucinate dimensions when the text is ambiguous. To address this, we propose a proactive agentic framework for text-to-CadQuery generation, named ProCAD, that resolves specification issues before code synthesis. Our framework pairs a proactive clarifying agent, which audits the prompt and asks targeted clarification questions only when necessary to produce a self-consistent specification, with a CAD coding agent that translates the specification into an executable CadQuery program. We fine-tune the coding agent on a curated high-quality text-to-CadQuery dataset and train the clarifying agent via agentic SFT on clarification trajectories. Experiments show that proactive clarification significantly improves robustness to ambiguous prompts while keeping interaction overhead low. ProCAD outperforms frontier closed-source models, including Claude Sonnet 4.5, reducing the mean Chamfer distance by 79.9 percent and lowering the invalidity ratio from 4.8 percent to 0.9 percent. Our code and datasets will be made publicly available.

**arXiv ID:** 2602.03045
</details>

<details>
<summary><strong>BridgeV2W: Bridging Video Generation Models to Embodied World Models via Embodiment Masks</strong> - Yixiang Chen, Peiyan Li, Jiabing Yang, Keji He, Xiangnan Wu, Yuan Xu, Kai Wang, Jing Liu, Nianfeng Liu, Yan Huang, Liang Wang - [[pdf]](https://arxiv.org/pdf/2602.03793)</summary>

**Abstract:** Embodied world models have emerged as a promising paradigm in robotics, most of which leverage large-scale Internet videos or pretrained video generation models to enrich visual and motion priors. However, they still face key challenges: a misalignment between coordinate-space actions and pixel-space videos, sensitivity to camera viewpoint, and non-unified architectures across embodiments. To this end, we present BridgeV2W, which converts coordinate-space actions into pixel-aligned embodiment masks rendered from the URDF and camera parameters. These masks are then injected into a pretrained video generation model via a ControlNet-style pathway, which aligns the action control signals with predicted videos, adds view-specific conditioning to accommodate camera viewpoints, and yields a unified world model architecture across embodiments. To mitigate overfitting to static backgrounds, BridgeV2W further introduces a flow-based motion loss that focuses on learning dynamic and task-relevant regions. Experiments on single-arm (DROID) and dual-arm (AgiBot-G1) datasets, covering diverse and challenging conditions with unseen viewpoints and scenes, show that BridgeV2W improves video generation quality compared to prior state-of-the-art methods. We further demonstrate the potential of BridgeV2W on downstream real-world tasks, including policy evaluation and goal-conditioned planning. More results can be found on our project website at this https URL .

**arXiv ID:** 2602.03793
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>ATLAS : Adaptive Self-Evolutionary Research Agent with Task-Distributed Multi-LLM Supporters</strong> - Ujin Jeon, Jiyong Kwon, Madison Ann Sullivan, Caleb Eunho Lee, Guang Lin - [[pdf]](https://arxiv.org/pdf/2602.02709)</summary>

**Abstract:** Recent multi-LLM agent systems perform well in prompt optimization and automated problem-solving, but many either keep the solver frozen after fine-tuning or rely on a static preference-optimization loop, which becomes intractable for long-horizon tasks. We propose ATLAS (Adaptive Task-distributed Learning for Agentic Self-evolution), a task-distributed framework that iteratively develops a lightweight research agent while delegating complementary roles to specialized supporter agents for exploration, hyperparameter tuning, and reference policy management. Our core algorithm, Evolving Direct Preference Optimization (EvoDPO), adaptively updates the phase-indexed reference policy. We provide a theoretical regret analysis for a preference-based contextual bandit under concept drift. In addition, experiments were conducted on non-stationary linear contextual bandits and scientific machine learning (SciML) loss reweighting for the 1D Burgers' equation. Both results show that ATLAS improves stability and performance over a static single-agent baseline.

**arXiv ID:** 2602.02709
</details>

<details>
<summary><strong>Ontology-to-tools compilation for executable semantic constraint enforcement in LLM agents</strong> - Xiaochi Zhou, Patrick Bulter, Changxuan Yang, Simon D. Rihm, Thitikarn Angkanaporn, Jethro Akroyd, Sebastian Mosbach, Markus Kraft - [[pdf]](https://arxiv.org/pdf/2602.03439)</summary>

**Abstract:** We introduce ontology-to-tools compilation as a proof-of-principle mechanism for coupling large language models (LLMs) with formal domain knowledge. Within The World Avatar (TWA), ontological specifications are compiled into executable tool interfaces that LLM-based agents must use to create and modify knowledge graph instances, enforcing semantic constraints during generation rather than through post-hoc validation. Extending TWA's semantic agent composition framework, the Model Context Protocol (MCP) and associated agents are integral components of the knowledge graph ecosystem, enabling structured interaction between generative models, symbolic constraints, and external resources. An agent-based workflow translates ontologies into ontology-aware tools and iteratively applies them to extract, validate, and repair structured knowledge from unstructured scientific text. Using metal-organic polyhedra synthesis literature as an illustrative case, we show how executable ontological semantics can guide LLM behaviour and reduce manual schema and prompt engineering, establishing a general paradigm for embedding formal knowledge into generative systems.

**arXiv ID:** 2602.03439
</details>

<details>
<summary><strong>Gender Dynamics and Homophily in a Social Network of LLM Agents</strong> - Faezeh Fadaei, Jenny Carla Moran, Taha Yasseri - [[pdf]](https://arxiv.org/pdf/2602.02606)</summary>

**Abstract:** Generative artificial intelligence and large language models (LLMs) are increasingly deployed in interactive settings, yet we know little about how their identity performance develops when they interact within large-scale networks. We address this by examining this http URL, a social media platform similar to X but composed entirely of autonomous AI chatbots. Our dataset comprises over 70,000 agents, approximately 140 million posts, and the evolving followership network over one year. Based on agents' text production, we assign weekly gender scores to each agent. Results suggest that each agent's gender performance is fluid rather than fixed. Despite this fluidity, the network displays strong gender-based homophily, as agents consistently follow others performing gender similarly. Finally, we investigate whether these homophilic connections arise from social selection, in which agents choose to follow similar accounts, or from social influence, in which agents become more similar to their followees over time. Consistent with human social networks, we find evidence that both mechanisms shape the structure and evolution of interactions among LLMs. Our findings suggest that, even in the absence of bodies, cultural entraining of gender performance leads to gender-based sorting. This has important implications for LLM applications in synthetic hybrid populations, social simulations, and decision support.

**arXiv ID:** 2602.02606
</details>

<details>
<summary><strong>Exploring Silicon-Based Societies: An Early Study of the Moltbook Agent Community</strong> - Yu-Zheng Lin, Bono Po-Jen Shih, Hsuan-Ying Alessandra Chien, Shalaka Satam, Jesus Horacio Pacheco, Sicong Shao, Soheil Salehi, Pratik Satam - [[pdf]](https://arxiv.org/pdf/2602.02613)</summary>

**Abstract:** The rapid emergence of autonomous large language model agents has given rise to persistent, large-scale agent ecosystems whose collective behavior cannot be adequately understood through anecdotal observation or small-scale simulation. This paper introduces data-driven silicon sociology as a systematic empirical framework for studying social structure formation among interacting artificial agents. We present a pioneering large-scale data mining investigation of an in-the-wild agent society by analyzing Moltbook, a social platform designed primarily for agent-to-agent interaction. At the time of study, Moltbook hosted over 150,000 registered autonomous agents operating across thousands of agent-created sub-communities. Using programmatic and non-intrusive data acquisition, we collected and analyzed the textual descriptions of 12,758 submolts, which represent proactive sub-community partitioning activities within the ecosystem. Treating agent-authored descriptions as first-class observational artifacts, we apply rigorous preprocessing, contextual embedding, and unsupervised clustering techniques to uncover latent patterns of thematic organization and social space structuring. The results show that autonomous agents systematically organize collective space through reproducible patterns spanning human-mimetic interests, silicon-centric self-reflection, and early-stage economic and coordination behaviors. Rather than relying on predefined sociological taxonomies, these structures emerge directly from machine-generated data traces. This work establishes a methodological foundation for data-driven silicon sociology and demonstrates that data mining techniques can provide a powerful lens for understanding the organization and evolution of large autonomous agent societies.

**arXiv ID:** 2602.02613
</details>

<details>
<summary><strong>From Task Solving to Robust Real-World Adaptation in LLM Agents</strong> - Pouya Pezeshkpour, Estevam Hruschka - [[pdf]](https://arxiv.org/pdf/2602.02760)</summary>

**Abstract:** Large language models are increasingly deployed as specialized agents that plan, call tools, and take actions over extended horizons. Yet many existing evaluations assume a "clean interface" where dynamics are specified and stable, tools and sensors are reliable, and success is captured by a single explicit objective-often overestimating real-world readiness. In practice, agents face underspecified rules, unreliable signals, shifting environments, and implicit, multi-stakeholder goals. The challenge is therefore not just solving tasks, but adapting while solving: deciding what to trust, what is wanted, when to verify, and when to fall back or escalate. We stress-test deployment-relevant robustness under four operational circumstances: partial observability, dynamic environments, noisy signals, and dynamic agent state. We benchmark agentic LLMs in a grid-based game with a simple goal but long-horizon execution. Episodes violate clean-interface assumptions yet remain solvable, forcing agents to infer rules, pay for information, adapt to environmental and internal shifts, and act cautiously under noise. Across five state-of-the-art LLM agents, we find large gaps between nominal task-solving and deployment-like robustness. Performance generally degrades as grid size and horizon increase, but rankings are unstable: weaker models can beat stronger ones when strategy matches the uncertainty regime. Despite no explicit instruction, agents trade off completion, efficiency, and penalty avoidance, suggesting partial objective inference. Ablations and feature analyses reveal model-specific sensitivities and failure drivers, motivating work on verification, safe action selection, and objective inference under partial observability, noise, and non-stationarity.

**arXiv ID:** 2602.02760
</details>

<details>
<summary><strong>Verified Critical Step Optimization for LLM Agents</strong> - Mukai Li, Qingcheng Zeng, Tianqing Fang, Zhenwen Liang, Linfeng Song, Qi Liu, Haitao Mi, Dong Yu - [[pdf]](https://arxiv.org/pdf/2602.03412)</summary>

**Abstract:** As large language model agents tackle increasingly complex long-horizon tasks, effective post-training becomes critical. Prior work faces fundamental challenges: outcome-only rewards fail to precisely attribute credit to intermediate steps, estimated step-level rewards introduce systematic noise, and Monte Carlo sampling approaches for step reward estimation incur prohibitive computational cost. Inspired by findings that only a small fraction of high-entropy tokens drive effective RL for reasoning, we propose Critical Step Optimization (CSO), which focuses preference learning on verified critical steps, decision points where alternate actions demonstrably flip task outcomes from failure to success. Crucially, our method starts from failed policy trajectories rather than expert demonstrations, directly targeting the policy model's weaknesses. We use a process reward model (PRM) to identify candidate critical steps, leverage expert models to propose high-quality alternatives, then continue execution from these alternatives using the policy model itself until task completion. Only alternatives that the policy successfully executes to correct outcomes are verified and used as DPO training data, ensuring both quality and policy reachability. This yields fine-grained, verifiable supervision at critical decisions while avoiding trajectory-level coarseness and step-level noise. Experiments on GAIA-Text-103 and XBench-DeepSearch show that CSO achieves 37% and 26% relative improvement over the SFT baseline and substantially outperforms other post-training methods, while requiring supervision at only 16% of trajectory steps. This demonstrates the effectiveness of selective verification-based learning for agent post-training.

**arXiv ID:** 2602.03412
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (32 papers)</h2></summary>

<details>
<summary><strong>Experience-Driven Multi-Agent Systems Are Training-free Context-aware Earth Observers</strong> - Pengyu Dai, Weihao Xuan, Junjue Wang, Hongruixuan Chen, Jian Song, Yafei Ou, Naoto Yokoya - [[pdf]](https://arxiv.org/pdf/2602.02559)</summary>

**Abstract:** Recent advances have enabled large language model (LLM) agents to solve complex tasks by orchestrating external tools. However, these agents often struggle in specialized, tool-intensive domains that demand long-horizon execution, tight coordination across modalities, and strict adherence to implicit tool constraints. Earth Observation (EO) tasks exemplify this challenge due to the multi-modal and multi-temporal data inputs, as well as the requirements of geo-knowledge constraints (spectrum library, spatial reasoning, etc): many high-level plans can be derailed by subtle execution errors that propagate through a pipeline and invalidate final results. A core difficulty is that existing agents lack a mechanism to learn fine-grained, tool-level expertise from interaction. Without such expertise, they cannot reliably configure tool parameters or recover from mid-execution failures, limiting their effectiveness in complex EO workflows. To address this, we introduce \textbf{GeoEvolver}, a self-evolving multi-agent system~(MAS) that enables LLM agents to acquire EO expertise through structured interaction without any parameter updates. GeoEvolver decomposes each query into independent sub-goals via a retrieval-augmented multi-agent orchestrator, then explores diverse tool-parameter configurations at the sub-goal level. Successful patterns and root-cause attribution from failures are then distilled in an evolving memory bank that provides in-context demonstrations for future queries. Experiments on three tool-integrated EO benchmarks show that GeoEvolver consistently improves end-to-end task success, with an average gain of 12\% across multiple LLM backbones, demonstrating that EO expertise can emerge progressively from efficient, fine-grained interactions with the environment.

**arXiv ID:** 2602.02559
</details>

<details>
<summary><strong>PeerRank: Autonomous LLM Evaluation Through Web-Grounded, Bias-Controlled Peer Review</strong> - Yanki Margalit, Erni Avram, Ran Taig, Oded Margalit, Nurit Cohen-Inger - [[pdf]](https://arxiv.org/pdf/2602.02589)</summary>

**Abstract:** Evaluating large language models typically relies on human-authored benchmarks, reference answers, and human or single-model judgments, approaches that scale poorly, become quickly outdated, and mismatch open-world deployments that depend on web retrieval and synthesis. We introduce PeerRank, a fully autonomous end-to-end evaluation framework in which models generate evaluation tasks, answer them with category-scoped live web grounding, judge peer responses and aggregate dense peer assessments into relative performance estimates, without human supervision or gold references. PeerRank treats evaluation as a multi-agent process where each model participates symmetrically as task designer, respondent, and evaluator, while removing biased judgments. In a large-scale study over 12 commercially available models and 420 autonomously generated questions, PeerRank produces stable, discriminative rankings and reveals measurable identity and presentation biases. Rankings are robust, and mean peer scores agree with Elo. We further validate PeerRank on TruthfulQA and GSM8K, where peer scores correlate with objective accuracy. Together, these results suggest that bias-aware peer evaluation with selective web-grounded answering can scale open-world LLM assessment beyond static and human curated benchmarks.

**arXiv ID:** 2602.02589
</details>

<details>
<summary><strong>MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems</strong> - Vishal Venkataramani, Haizhou Shi, Zixuan Ke, Austin Xu, Xiaoxiao He, Yingbo Zhou, Semih Yavuz, Hao Wang, Shafiq Joty - [[pdf]](https://arxiv.org/pdf/2602.03053)</summary>

**Abstract:** Multi-Agent Systems (MAS) built on Large Language Models (LLMs) often exhibit high variance in their reasoning trajectories. Process verification, which evaluates intermediate steps in trajectories, has shown promise in general reasoning settings, and has been suggested as a potential tool for guiding coordination of MAS; however, its actual effectiveness in MAS remains unclear. To fill this gap, we present MAS-ProVe, a systematic empirical study of process verification for multi-agent systems (MAS). Our study spans three verification paradigms (LLM-as-a-Judge, reward models, and process reward models), evaluated across two levels of verification granularity (agent-level and iteration-level). We further examine five representative verifiers and four context management strategies, and conduct experiments over six diverse MAS frameworks on multiple reasoning benchmarks. We find that process-level verification does not consistently improve performance and frequently exhibits high variance, highlighting the difficulty of reliably evaluating partial multi-agent trajectories. Among the methods studied, LLM-as-a-Judge generally outperforms reward-based approaches, with trained judges surpassing general-purpose LLMs. We further observe a small performance gap between LLMs acting as judges and as single agents, and identify a context-length-performance trade-off in verification. Overall, our results suggest that effective and robust process verification for MAS remains an open challenge, requiring further advances beyond current paradigms. Code is available at this https URL.

**arXiv ID:** 2602.03053
</details>

<details>
<summary><strong>Visual Reasoning over Time Series via Multi-Agent System</strong> - Weilin Ruan, Yuxuan Liang - [[pdf]](https://arxiv.org/pdf/2602.03026)</summary>

**Abstract:** Time series analysis underpins many real-world applications, yet existing time-series-specific methods and pretrained large-model-based approaches remain limited in integrating intuitive visual reasoning and generalizing across tasks with adaptive tool usage. To address these limitations, we propose MAS4TS, a tool-driven multi-agent system for general time series tasks, built upon an Analyzer-Reasoner-Executor paradigm that integrates agent communication, visual reasoning, and latent reconstruction within a unified framework. MAS4TS first performs visual reasoning over time series plots with structured priors using a Vision-Language Model to extract temporal structures, and subsequently reconstructs predictive trajectories in latent space. Three specialized agents coordinate via shared memory and gated communication, while a router selects task-specific tool chains for execution. Extensive experiments on multiple benchmarks demonstrate that MAS4TS achieves state-of-the-art performance across a wide range of time series tasks, while exhibiting strong generalization and efficient inference.

**arXiv ID:** 2602.03026
</details>

<details>
<summary><strong>Understanding Multi-Agent LLM Frameworks: A Unified Benchmark and Experimental Analysis</strong> - Abdelghny Orogat, Ana Rostam, Essam Mansour - [[pdf]](https://arxiv.org/pdf/2602.03128)</summary>

**Abstract:** Multi-agent LLM frameworks are widely used to accelerate the development of agent systems powered by large language models (LLMs). These frameworks impose distinct architectural structures that govern how agents interact, store information, and coordinate tasks. However, their impact on system performance remains poorly understood. This gap is critical, as architectural choices alone can induce order-of-magnitude differences in latency and throughput, as well as substantial variation in accuracy and scalability. Addressing this challenge requires (i) jointly evaluating multiple capabilities, such as orchestration overhead, memory behavior, planning, specialization, and coordination, and (ii) conducting these evaluations under controlled, framework-level conditions to isolate architectural effects. Existing benchmarks focus on individual capabilities and lack standardized framework-level evaluation. We address these limitations by (i) introducing an architectural taxonomy for systematically comparing multi-agent LLM frameworks along fundamental dimensions, and (ii) developing MAFBench, a unified evaluation suite that integrates existing benchmarks under a standardized execution pipeline. Using MAFBench, we conduct a controlled empirical study across several widely used frameworks. Our results show that framework-level design choices alone can increase latency by over 100x, reduce planning accuracy by up to 30%, and lower coordination success from above 90% to below 30%. Finally, we translate our findings into concrete architectural design principles and framework selection guidance, and outline promising future research directions.

**arXiv ID:** 2602.03128
</details>

<details>
<summary><strong>Beyond Quantity: Trajectory Diversity Scaling for Code Agents</strong> - Guhong Chen, Chenghao Sun, Cheng Fu, Qiyao Wang, Zhihong Huang, Chaopeng Wei, Guangxu Chen, Feiteng Fang, Ahmadreza Argha, Bing Zhao, Xander Xu, Qi Han, Hamid Alinejad-Rokny, Qiang Qu, Binhua Li, Shiwen Ni, Min Yang, Hu Wei, Yongbin Li - [[pdf]](https://arxiv.org/pdf/2602.03219)</summary>

**Abstract:** As code large language models (LLMs) evolve into tool-interactive agents via the Model Context Protocol (MCP), their generalization is increasingly limited by low-quality synthetic data and the diminishing returns of quantity scaling. Moreover, quantity-centric scaling exhibits an early bottleneck that underutilizes trajectory data. We propose TDScaling, a Trajectory Diversity Scaling-based data synthesis framework for code agents that scales performance through diversity rather than raw volume. Under a fixed training budget, increasing trajectory diversity yields larger gains than adding more trajectories, improving the performance-cost trade-off for agent training. TDScaling integrates four innovations: (1) a Business Cluster mechanism that captures real-service logical dependencies; (2) a blueprint-driven multi-agent paradigm that enforces trajectory coherence; (3) an adaptive evolution mechanism that steers synthesis toward long-tail scenarios using Domain Entropy, Reasoning Mode Entropy, and Cumulative Action Complexity to prevent mode collapse; and (4) a sandboxed code tool that mitigates catastrophic forgetting of intrinsic coding capabilities. Experiments on general tool-use benchmarks (BFCL, tau^2-Bench) and code agent tasks (RebenchT, CodeCI, BIRD) demonstrate a win-win outcome: TDScaling improves both tool-use generalization and inherent coding proficiency. We plan to release the full codebase and the synthesized dataset (including 30,000+ tool clusters) upon publication.

**arXiv ID:** 2602.03219
</details>

<details>
<summary><strong>LPS-Bench: Benchmarking Safety Awareness of Computer-Use Agents in Long-Horizon Planning under Benign and Adversarial Scenarios</strong> - Tianyu Chen, Chujia Hu, Ge Gao, Dongrui Liu, Xia Hu, Wenjie Wang - [[pdf]](https://arxiv.org/pdf/2602.03255)</summary>

**Abstract:** Computer-use agents (CUAs) that interact with real computer systems can perform automated tasks but face critical safety risks. Ambiguous instructions may trigger harmful actions, and adversarial users can manipulate tool execution to achieve malicious goals. Existing benchmarks mostly focus on short-horizon or GUI-based tasks, evaluating on execution-time errors but overlooking the ability to anticipate planning-time risks. To fill this gap, we present LPS-Bench, a benchmark that evaluates the planning-time safety awareness of MCP-based CUAs under long-horizon tasks, covering both benign and adversarial interactions across 65 scenarios of 7 task domains and 9 risk types. We introduce a multi-agent automated pipeline for scalable data generation and adopt an LLM-as-a-judge evaluation protocol to assess safety awareness through the planning trajectory. Experiments reveal substantial deficiencies in existing CUAs' ability to maintain safe behavior. We further analyze the risks and propose mitigation strategies to improve long-horizon planning safety in MCP-based CUA systems. We open-source our code at this https URL.

**arXiv ID:** 2602.03255
</details>

<details>
<summary><strong>TodyComm: Task-Oriented Dynamic Communication for Multi-Round LLM-based Multi-Agent System</strong> - Wenzhe Fan, Tommaso Tognoli, Henry Peng Zou, Chunyu Miao, Yibo Wang, Xinhua Zhang - [[pdf]](https://arxiv.org/pdf/2602.03688)</summary>

**Abstract:** Multi-round LLM-based multi-agent systems rely on effective communication structures to support collaboration across rounds. However, most existing methods employ a fixed communication topology during inference, which falls short in many realistic applications where the agents' roles may change \textit{across rounds} due to dynamic adversary, task progression, or time-varying constraints such as communication bandwidth. In this paper, we propose addressing this issue through TodyComm, a \textbf{t}ask-\textbf{o}riented \textbf{dy}namic \textbf{comm}unication algorithm. It produces behavior-driven collaboration topologies that adapt to the dynamics at each round, optimizing the utility for the task through policy gradient. Experiments on five benchmarks demonstrate that under both dynamic adversary and communications budgets, TodyComm delivers superior task effectiveness while retaining token efficiency and scalability.

**arXiv ID:** 2602.03688
</details>

<details>
<summary><strong>Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity</strong> - Yingxuan Yang, Chengrui Qu, Muning Wen, Laixi Shi, Ying Wen, Weinan Zhang, Adam Wierman, Shangding Gu - [[pdf]](https://arxiv.org/pdf/2602.03794)</summary>

**Abstract:** LLM-based multi-agent systems (MAS) have emerged as a promising approach to tackle complex tasks that are difficult for individual LLMs. A natural strategy is to scale performance by increasing the number of agents; however, we find that such scaling exhibits strong diminishing returns in homogeneous settings, while introducing heterogeneity (e.g., different models, prompts, or tools) continues to yield substantial gains. This raises a fundamental question: what limits scaling, and why does diversity help? We present an information-theoretic framework showing that MAS performance is bounded by the intrinsic task uncertainty, not by agent count. We derive architecture-agnostic bounds demonstrating that improvements depend on how many effective channels the system accesses. Homogeneous agents saturate early because their outputs are strongly correlated, whereas heterogeneous agents contribute complementary evidence. We further introduce $K^*$, an effective channel count that quantifies the number of effective channels without ground-truth labels. Empirically, we show that heterogeneous configurations consistently outperform homogeneous scaling: 2 diverse agents can match or exceed the performance of 16 homogeneous agents. Our results provide principled guidelines for building efficient and robust MAS through diversity-aware design. Code and Dataset are available at the link: this https URL.

**arXiv ID:** 2602.03794
</details>

<details>
<summary><strong>ContextEvolve: Multi-Agent Context Compression for Systems Code Optimization</strong> - Hongyuan Su, Yu Zheng, Yong Li - [[pdf]](https://arxiv.org/pdf/2602.02597)</summary>

**Abstract:** Large language models are transforming systems research by automating the discovery of performance-critical algorithms for computer systems. Despite plausible codes generated by LLMs, producing solutions that meet the stringent correctness and performance requirements of systems demands iterative optimization. Test-time reinforcement learning offers high search efficiency but requires parameter updates infeasible under API-only access, while existing training-free evolutionary methods suffer from inefficient context utilization and undirected search. We introduce ContextEvolve, a multi-agent framework that achieves RL-level search efficiency under strict parameter-blind constraints by decomposing optimization context into three orthogonal dimensions: a Summarizer Agent condenses semantic state via code-to-language abstraction, a Navigator Agent distills optimization direction from trajectory analysis, and a Sampler Agent curates experience distribution through prioritized exemplar retrieval. This orchestration forms a functional isomorphism with RL-mapping to state representation, policy gradient, and experience replay-enabling principled optimization in a textual latent space. On the ADRS benchmark, ContextEvolve outperforms state-of-the-art baselines by 33.3% while reducing token consumption by 29.0%. Codes for our work are released at this https URL

**arXiv ID:** 2602.02597
</details>

<details>
<summary><strong>Social Catalysts, Not Moral Agents: The Illusion of Alignment in LLM Societies</strong> - Yueqing Hu, Yixuan Jiang, Zehua Jiang, Xiao Wen, Tianhong Wang - [[pdf]](https://arxiv.org/pdf/2602.02598)</summary>

**Abstract:** The rapid evolution of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems where collective cooperation is often threatened by the "Tragedy of the Commons." This study investigates the effectiveness of Anchoring Agents--pre-programmed altruistic entities--in fostering cooperation within a Public Goods Game (PGG). Using a full factorial design across three state-of-the-art LLMs, we analyzed both behavioral outcomes and internal reasoning chains. While Anchoring Agents successfully boosted local cooperation rates, cognitive decomposition and transfer tests revealed that this effect was driven by strategic compliance and cognitive offloading rather than genuine norm internalization. Notably, most agents reverted to self-interest in new environments, and advanced models like GPT-4.1 exhibited a "Chameleon Effect," masking strategic defection under public scrutiny. These findings highlight a critical gap between behavioral modification and authentic value alignment in artificial societies.

**arXiv ID:** 2602.02598
</details>

<details>
<summary><strong>WideSeek: Advancing Wide Research via Multi-Agent Scaling</strong> - Ziyang Huang, Haolin Ren, Xiaowei Yuan, Jiawei Wang, Zhongtao Jiang, Kun Xu, Shizhu He, Jun Zhao, Kang Liu - [[pdf]](https://arxiv.org/pdf/2602.02636)</summary>

**Abstract:** Search intelligence is evolving from Deep Research to Wide Research, a paradigm essential for retrieving and synthesizing comprehensive information under complex constraints in parallel. However, progress in this field is impeded by the lack of dedicated benchmarks and optimization methodologies for search breadth. To address these challenges, we take a deep dive into Wide Research from two perspectives: Data Pipeline and Agent Optimization. First, we produce WideSeekBench, a General Broad Information Seeking (GBIS) benchmark constructed via a rigorous multi-phase data pipeline to ensure diversity across the target information volume, logical constraints, and domains. Second, we introduce WideSeek, a dynamic hierarchical multi-agent architecture that can autonomously fork parallel sub-agents based on task requirements. Furthermore, we design a unified training framework that linearizes multi-agent trajectories and optimizes the system using end-to-end RL. Experimental results demonstrate the effectiveness of WideSeek and multi-agent RL, highlighting that scaling the number of agents is a promising direction for advancing the Wide Research paradigm.

**arXiv ID:** 2602.02636
</details>

<details>
<summary><strong>CVE-Factory: Scaling Expert-Level Agentic Tasks for Code Security Vulnerability</strong> - Xianzhen Luo, Jingyuan Zhang, Shiqi Zhou, Rain Huang, Chuan Xiao, Qingfu Zhu, Zhiyuan Ma, Xing Yue, Yang Yue, Wencong Zeng, Wanxiang Che - [[pdf]](https://arxiv.org/pdf/2602.03012)</summary>

**Abstract:** Evaluating and improving the security capabilities of code agents requires high-quality, executable vulnerability tasks. However, existing works rely on costly, unscalable manual reproduction and suffer from outdated data distributions. To address these, we present CVE-Factory, the first multi-agent framework to achieve expert-level quality in automatically transforming sparse CVE metadata into fully executable agentic tasks. Cross-validation against human expert reproductions shows that CVE-Factory achieves 95\% solution correctness and 96\% environment fidelity, confirming its expert-level quality. It is also evaluated on the latest realistic vulnerabilities and achieves a 66.2\% verified success. This automation enables two downstream contributions. First, we construct LiveCVEBench, a continuously updated benchmark of 190 tasks spanning 14 languages and 153 repositories that captures emerging threats including AI-tooling vulnerabilities. Second, we synthesize over 1,000 executable training environments, the first large-scale scaling of agentic tasks in code security. Fine-tuned Qwen3-32B improves from 5.3\% to 35.8\% on LiveCVEBench, surpassing Claude 4.5 Sonnet, with gains generalizing to Terminal Bench (12.5\% to 31.3\%). We open-source CVE-Factory, LiveCVEBench, Abacus-cve (fine-tuned model), training dataset, and leaderboard. All resources are available at this https URL .

**arXiv ID:** 2602.03012
</details>

<details>
<summary><strong>Game-Theoretic and Algorithmic Analyses of Multi-Agent Routing under Crossing Costs</strong> - Tesshu Hanaka, Nikolaos Melissinos, Hirotaka Ono - [[pdf]](https://arxiv.org/pdf/2602.03455)</summary>

**Abstract:** Coordinating the movement of multiple autonomous agents over a shared network is a fundamental challenge in algorithmic robotics, intelligent transportation, and distributed systems. The dominant approach, Multi-Agent Path Finding, relies on centralized control and synchronous collision avoidance, which often requires strict synchronization and guarantees of globally conflict-free execution. This paper introduces the Multi-Agent Routing under Crossing Cost model on mixed graphs, a novel framework tailored to asynchronous settings. In our model, instead of treating conflicts as hard constraints, each agent is assigned a path, and the system is evaluated through a cost function that measures potential head-on encounters. This ``crossing cost'', which is defined as the product of the numbers of agents traversing an edge in opposite directions, quantifies the risk of congestion and delay in decentralized execution.
Our contributions are both game-theoretic and algorithmic. We model the setting as a congestion game with a non-standard cost function, prove the existence of pure Nash equilibria, and analyze the dynamics leading to them. Equilibria can be found in polynomial time under mild conditions, while the general case is PLS-complete. From an optimization perspective, minimizing the total crossing cost is NP-hard, as the problem generalizes Steiner Orientation. To address this hardness barrier, we design a suite of parameterized algorithms for minimizing crossing cost, with parameters including the number of arcs, edges, agents, and structural graph measures. These yield XP or FPT results depending on the parameter, offering algorithmic strategies for structurally restricted instances. Our framework provides a new theoretical foundation for decentralized multi-agent routing, bridging equilibrium analysis and parameterized complexity to support scalable and risk-aware coordination.

**arXiv ID:** 2602.03455
</details>

<details>
<summary><strong>Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems</strong> - Haibo Jin, Kuang Peng, Ye Yu, Xiaopeng Yuan, Haohan Wang - [[pdf]](https://arxiv.org/pdf/2602.03695)</summary>

**Abstract:** While existing multi-agent systems (MAS) can handle complex problems by enabling collaboration among multiple agents, they are often highly task-specific, relying on manually crafted agent roles and interaction prompts, which leads to increased architectural complexity and limited reusability across tasks. Moreover, most MAS communicate primarily through natural language, making them vulnerable to error accumulation and instability in long-context, multi-stage interactions within internal agent histories.
In this work, we propose \textbf{Agent Primitives}, a set of reusable latent building blocks for LLM-based MAS. Inspired by neural network design, where complex models are built from reusable components, we observe that many existing MAS architectures can be decomposed into a small number of recurring internal computation patterns. Based on this observation, we instantiate three primitives: Review, Voting and Selection, and Planning and Execution. All primitives communicate internally via key-value (KV) cache, which improves both robustness and efficiency by mitigating information degradation across multi-stage interactions. To enable automatic system construction, an Organizer agent selects and composes primitives for each query, guided by a lightweight knowledge pool of previously successful configurations, forming a primitive-based MAS.
Experiments show that primitives-based MAS improve average accuracy by 12.0-16.5\% over single-agent baselines, reduce token usage and inference latency by approximately 3$\times$-4$\times$ compared to text-based MAS, while incurring only 1.3$\times$-1.6$\times$ overhead relative to single-agent inference and providing more stable performance across model backbones.

**arXiv ID:** 2602.03695
</details>

<details>
<summary><strong>IMAGINE: Intelligent Multi-Agent Godot-based Indoor Networked Exploration</strong> - Tiago Leite, Maria Conceição, António Grilo - [[pdf]](https://arxiv.org/pdf/2602.02858)</summary>

**Abstract:** The exploration of unknown, Global Navigation Satellite System (GNSS) denied environments by an autonomous communication-aware and collaborative group of Unmanned Aerial Vehicles (UAVs) presents significant challenges in coordination, perception, and decentralized decision-making. This paper implements Multi-Agent Reinforcement Learning (MARL) to address these challenges in a 2D indoor environment, using high-fidelity game-engine simulations (Godot) and continuous action spaces. Policy training aims to achieve emergent collaborative behaviours and decision-making under uncertainty using Network-Distributed Partially Observable Markov Decision Processes (ND-POMDPs). Each UAV is equipped with a Light Detection and Ranging (LiDAR) sensor and can share data (sensor measurements and a local occupancy map) with neighbouring agents. Inter-agent communication constraints include limited range, bandwidth and latency. Extensive ablation studies evaluated MARL training paradigms, reward function, communication system, neural network (NN) architecture, memory mechanisms, and POMDP formulations. This work jointly addresses several key limitations in prior research, namely reliance on discrete actions, single-agent or centralized formulations, assumptions of a priori knowledge and permanent connectivity, inability to handle dynamic obstacles, short planning horizons and architectural complexity in Recurrent NNs/Transformers. Results show that the scalable training paradigm, combined with a simplified architecture, enables rapid autonomous exploration of an indoor area. The implementation of Curriculum-Learning (five increasingly complex levels) also enabled faster, more robust training. This combination of high-fidelity simulation, MARL formulation, and computational efficiency establishes a strong foundation for deploying learned cooperative strategies in physical robotic systems.

**arXiv ID:** 2602.02858
</details>

<details>
<summary><strong>LatentMem: Customizing Latent Memory for Multi-Agent Systems</strong> - Muxin Fu, Guibin Zhang, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, Yang Yang - [[pdf]](https://arxiv.org/pdf/2602.03036)</summary>

**Abstract:** Large language model (LLM)-powered multi-agent systems (MAS) demonstrate remarkable collective intelligence, wherein multi-agent memory serves as a pivotal mechanism for continual adaptation. However, existing multi-agent memory designs remain constrained by two fundamental bottlenecks: (i) memory homogenization arising from the absence of role-aware customization, and (ii) information overload induced by excessively fine-grained memory entries. To address these limitations, we propose LatentMem, a learnable multi-agent memory framework designed to customize agent-specific memories in a token-efficient manner. Specifically, LatentMem comprises an experience bank that stores raw interaction trajectories in a lightweight form, and a memory composer that synthesizes compact latent memories conditioned on retrieved experience and agent-specific contexts. Further, we introduce Latent Memory Policy Optimization (LMPO), which propagates task-level optimization signals through latent memories to the composer, encouraging it to produce compact and high-utility representations. Extensive experiments across diverse benchmarks and mainstream MAS frameworks show that LatentMem achieves a performance gain of up to $19.36$% over vanilla settings and consistently outperforms existing memory architectures, without requiring any modifications to the underlying frameworks.

**arXiv ID:** 2602.03036
</details>

<details>
<summary><strong>Efficient Investment in Multi-Agent Models of Public Transportation</strong> - Martin Bullinger, Edith Elkind, Kassian Köck - [[pdf]](https://arxiv.org/pdf/2602.03687)</summary>

**Abstract:** We study two stylized, multi-agent models aimed at investing a limited, indivisible resource in public transportation. In the first model, we face the decision of which potential stops to open along a (e.g., bus) path, given agents' travel demands. While it is known that utilitarian optimal solutions can be identified in polynomial time, we find that computing approximately optimal solutions with respect to egalitarian welfare is NP-complete. This is surprising as we operate on the simple topology of a line graph.
In the second model, agents navigate a more complex network modeled by a weighted graph where edge weights represent distances. We face the decision of improving travel time along a fixed number of edges. We provide a polynomial-time algorithm that combines Dijkstra's algorithm with a dynamical program to find the optimal decision for one or two agents. By contrast, if the number of agents is variable, we find \np-completeness and inapproximability results for utilitarian and egalitarian welfare. Moreover, we demonstrate implications of our results for a related model of railway network design.

**arXiv ID:** 2602.03687
</details>

<details>
<summary><strong>Multi-Agent Teams Hold Experts Back</strong> - Aneesh Pappu, Batu El, Hancheng Cao, Carmelo di Nolfo, Yanchao Sun, Meng Cao, James Zou - [[pdf]](https://arxiv.org/pdf/2602.01011)</summary>

**Abstract:** Multi-agent LLM systems are increasingly deployed as autonomous collaborators, where agents interact freely rather than execute fixed, pre-specified workflows. In such settings, effective coordination cannot be fully designed in advance and must instead emerge through interaction. However, most prior work enforces coordination through fixed roles, workflows, or aggregation rules, leaving open the question of how well self-organizing teams perform when coordination is unconstrained. Drawing on organizational psychology, we study whether self-organizing LLM teams achieve strong synergy, where team performance matches or exceeds the best individual member. Across human-inspired and frontier ML benchmarks, we find that -- unlike human teams -- LLM teams consistently fail to match their expert agent's performance, even when explicitly told who the expert is, incurring performance losses of up to 37.6%. Decomposing this failure, we show that expert leveraging, rather than identification, is the primary bottleneck. Conversational analysis reveals a tendency toward integrative compromise -- averaging expert and non-expert views rather than appropriately weighting expertise -- which increases with team size and correlates negatively with performance. Interestingly, this consensus-seeking behavior improves robustness to adversarial agents, suggesting a trade-off between alignment and effective expertise utilization. Our findings reveal a significant gap in the ability of self-organizing multi-agent teams to harness the collective expertise of their members.

**arXiv ID:** 2602.01011
</details>

<details>
<summary><strong>Multi-Agent Pathfinding Under Team-Connected Communication Constraint via Adaptive Path Expansion and Dynamic Leading</strong> - Hoang-Dung Bui, Erion Plaku, Gregoy J. Stein - [[pdf]](https://arxiv.org/pdf/2501.02770)</summary>

**Abstract:** This paper proposes a novel planning framework to handle a multi-agent pathfinding problem under team-connected communication constraint, where all agents must have a connected communication channel to the rest of the team during their entire movements. Standard multi-agent path finding approaches (e.g., priority-based search) have potential in this domain but fail when neighboring configurations at start and goal differ. Their single-expansion approach -- computing each agent's path from the start to the goal in just a single expansion -- cannot reliably handle planning under communication constraints for agents as their neighbors change during navigating. Similarly, leader-follower approaches (e.g., platooning) are effective at maintaining team communication, but fixing the leader at the outset of planning can cause planning to become stuck in dense-clutter environments, limiting their practical utility. To overcome this limitation, we propose a novel two-level multi-agent pathfinding framework that integrates two techniques: adaptive path expansion to expand agent paths to their goals in multiple stages; and dynamic leading technique that enables the reselection of the leading agent during each agent path expansion whenever progress cannot be made. Simulation experiments show the efficiency of our planners, which can handle up to 25 agents across five environment types under a limited communication range constraint and up to 11-12 agents on three environment types under line-of-sight communication constraint, exceeding 90% success-rate where baselines routinely fail.

**arXiv ID:** 2501.02770
</details>

<details>
<summary><strong>LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems</strong> - Guilin Zhang, Kai Zhao, Jeffrey Friedman, Xu Chu - [[pdf]](https://arxiv.org/pdf/2601.19121)</summary>

**Abstract:** Recommendation systems must optimize multiple objectives while satisfying hard business constraints such as fairness and coverage. For example, an e-commerce platform may require every recommendation list to include items from multiple sellers and at least one newly listed product; violating such constraints--even once--is unacceptable in production. Prior work on multi-objective recommendation and recent LLM-based recommender agents largely treat constraints as soft penalties or focus on item scoring and interaction, leading to frequent violations in real-world deployments. How to leverage LLMs for coordinating constrained optimization in recommendation systems remains underexplored. We propose DualAgent-Rec, an LLM-coordinated dual-agent framework for constrained multi-objective e-commerce recommendation. The framework separates optimization into an Exploitation Agent that prioritizes accuracy under hard constraints and an Exploration Agent that promotes diversity through unconstrained Pareto search. An LLM-based coordinator adaptively allocates resources between agents based on optimization progress and constraint satisfaction, while an adaptive epsilon-relaxation mechanism guarantees feasibility of final solutions. Experiments on the Amazon Reviews 2023 dataset demonstrate that DualAgent-Rec achieves 100% constraint satisfaction and improves Pareto hypervolume by 4-6% over strong baselines, while maintaining competitive accuracy-diversity trade-offs. These results indicate that LLMs can act as effective orchestration agents for deployable and constraint-compliant recommendation systems.

**arXiv ID:** 2601.19121
</details>

<details>
<summary><strong>CPMobius: Iterative Coach-Player Reasoning for Data-Free Reinforcement Learning</strong> - Ran Li, Zeyuan Liu, Yinghao chen, Bingxiang He, Jiarui Yuan, Zixuan Fu, Weize Chen, Jinyi Hu, Zhiyuan Liu, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2602.02979)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated strong potential in complex reasoning, yet their progress remains fundamentally constrained by reliance on massive high-quality human-curated tasks and labels, either through supervised fine-tuning (SFT) or reinforcement learning (RL) on reasoning-specific data. This dependence renders supervision-heavy training paradigms increasingly unsustainable, with signs of diminishing scalability already evident in practice. To overcome this limitation, we introduce CPMöbius (CPMobius), a collaborative Coach-Player paradigm for data-free reinforcement learning of reasoning models. Unlike traditional adversarial self-play, CPMöbius, inspired by real world human sports collaboration and multi-agent collaboration, treats the Coach and Player as independent but cooperative roles. The Coach proposes instructions targeted at the Player's capability and receives rewards based on changes in the Player's performance, while the Player is rewarded for solving the increasingly instructive tasks generated by the Coach. This cooperative optimization loop is designed to directly enhance the Player's mathematical reasoning ability. Remarkably, CPMöbius achieves substantial improvement without relying on any external training data, outperforming existing unsupervised approaches. For example, on Qwen2.5-Math-7B-Instruct, our method improves accuracy by an overall average of +4.9 and an out-of-distribution average of +5.4, exceeding RENT by +1.5 on overall accuracy and R-zero by +4.2 on OOD accuracy.

**arXiv ID:** 2602.02979
</details>

<details>
<summary><strong>One Model, All Roles: Multi-Turn, Multi-Agent Self-Play Reinforcement Learning for Conversational Social Intelligence</strong> - Bowen Jiang, Taiwei Shi, Ryo Kamoi, Yuan Yuan, Camillo J. Taylor, Longqi Yang, Pei Zhou, Sihao Chen - [[pdf]](https://arxiv.org/pdf/2602.03109)</summary>

**Abstract:** This paper introduces OMAR: One Model, All Roles, a reinforcement learning framework that enables AI to develop social intelligence through multi-turn, multi-agent conversational self-play. Unlike traditional paradigms that rely on static, single-turn optimizations, OMAR allows a single model to role-play all participants in a conversation simultaneously, learning to achieve long-term goals and complex social norms directly from dynamic social interaction. To ensure training stability across long dialogues, we implement a hierarchical advantage estimation that calculates turn-level and token-level advantages. Evaluations in the SOTOPIA social environment and Werewolf strategy games show that our trained models develop fine-grained, emergent social intelligence, such as empathy, persuasion, and compromise seeking, demonstrating the effectiveness of learning collaboration even under competitive scenarios. While we identify practical challenges like reward hacking, our results show that rich social intelligence can emerge without human supervision. We hope this work incentivizes further research on AI social intelligence in group conversations.

**arXiv ID:** 2602.03109
</details>

<details>
<summary><strong>MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research</strong> - Yifan Shi, Jialong Shi, Jiayi Wang, Ye Fan, Jianyong Sun - [[pdf]](https://arxiv.org/pdf/2602.03318)</summary>

**Abstract:** Operations Research (OR) relies on expert-driven modeling-a slow and fragile process ill-suited to novel scenarios. While large language models (LLMs) can automatically translate natural language into optimization models, existing approaches either rely on costly post-training or employ multi-agent frameworks, yet most still lack reliable collaborative error correction and task-specific retrieval, often leading to incorrect outputs. We propose MIRROR, a fine-tuning-free, end-to-end multi-agent framework that directly translates natural language optimization problems into mathematical models and solver code. MIRROR integrates two core mechanisms: (1) execution-driven iterative adaptive revision for automatic error correction, and (2) hierarchical retrieval to fetch relevant modeling and coding exemplars from a carefully curated exemplar library. Experiments show that MIRROR outperforms existing methods on standard OR benchmarks, with notable results on complex industrial datasets such as IndustryOR and Mamo-ComplexLP. By combining precise external knowledge infusion with systematic error correction, MIRROR provides non-expert users with an efficient and reliable OR modeling solution, overcoming the fundamental limitations of general-purpose LLMs in expert optimization tasks.

**arXiv ID:** 2602.03318
</details>

<details>
<summary><strong>Cognitively Diverse Multiple-Choice Question Generation: A Hybrid Multi-Agent Framework with Large Language Models</strong> - Yu Tian, Linh Huynh, Katerina Christhilf, Shubham Chakraborty, Micah Watanabe, Tracy Arner, Danielle McNamara - [[pdf]](https://arxiv.org/pdf/2602.03704)</summary>

**Abstract:** Recent advances in large language models (LLMs) have made automated multiple-choice question (MCQ) generation increasingly feasible; however, reliably producing items that satisfy controlled cognitive demands remains a challenge. To address this gap, we introduce ReQUESTA, a hybrid, multi-agent framework for generating cognitively diverse MCQs that systematically target text-based, inferential, and main idea comprehension. ReQUESTA decomposes MCQ authoring into specialized subtasks and coordinates LLM-powered agents with rule-based components to support planning, controlled generation, iterative evaluation, and post-processing. We evaluated the framework in a large-scale reading comprehension study using academic expository texts, comparing ReQUESTA-generated MCQs with those produced by a single-pass GPT-5 zero-shot baseline. Psychometric analyses of learner responses assessed item difficulty and discrimination, while expert raters evaluated question quality across multiple dimensions, including topic relevance and distractor quality. Results showed that ReQUESTA-generated items were consistently more challenging, more discriminative, and more strongly aligned with overall reading comprehension performance. Expert evaluations further indicated stronger alignment with central concepts and superior distractor linguistic consistency and semantic plausibility, particularly for inferential questions. These findings demonstrate that hybrid, agentic orchestration can systematically improve the reliability and controllability of LLM-based generation, highlighting workflow design as a key lever for structured artifact generation beyond single-pass prompting.

**arXiv ID:** 2602.03704
</details>

<details>
<summary><strong>FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation</strong> - Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Mingjie Zhan, Hongsheng Li - [[pdf]](https://arxiv.org/pdf/2602.03798)</summary>

**Abstract:** Assisting non-expert users to develop complex interactive websites has become a popular task for LLM-powered code agents. However, existing code agents tend to only generate frontend web pages, masking the lack of real full-stack data processing and storage with fancy visual effects. Notably, constructing production-level full-stack web applications is far more challenging than only generating frontend web pages, demanding careful control of data flow, comprehensive understanding of constantly updating packages and dependencies, and accurate localization of obscure bugs in the codebase. To address these difficulties, we introduce FullStack-Agent, a unified agent system for full-stack agentic coding that consists of three parts: (1) FullStack-Dev, a multi-agent framework with strong planning, code editing, codebase navigation, and bug localization abilities. (2) FullStack-Learn, an innovative data-scaling and self-improving method that back-translates crawled and synthesized website repositories to improve the backbone LLM of FullStack-Dev. (3) FullStack-Bench, a comprehensive benchmark that systematically tests the frontend, backend and database functionalities of the generated website. Our FullStack-Dev outperforms the previous state-of-the-art method by 8.7%, 38.2%, and 15.9% on the frontend, backend, and database test cases respectively. Additionally, FullStack-Learn raises the performance of a 30B model by 9.7%, 9.5%, and 2.8% on the three sets of test cases through self-improvement, demonstrating the effectiveness of our approach. The code is released at this https URL.

**arXiv ID:** 2602.03798
</details>

<details>
<summary><strong>Co-Designing Quantum Codes with Transversal Diagonal Gates via Multi-Agent Systems</strong> - Xi He, Sirui Lu, Bei Zeng - [[pdf]](https://arxiv.org/pdf/2510.20728)</summary>

**Abstract:** We present a multi-agent, human-in-the-loop workflow that co-designs quantum error-correcting codes with prescribed transversal diagonal gates. It builds on the Subset-Sum Linear Programming (SSLP) framework, which partitions basis strings by modular residues and enforces Z-marginal Knill-Laflamme (KL) equalities via small LPs. The workflow is powered by GPT-5 and implemented within TeXRA, a multi-agent research assistant platform where agents collaborate in a shared LaTeX-Python workspace synchronized with Git/Overleaf. Three specialized agents formulate constraints, sweep and screen candidate codes, exactify numerical solutions into rationals, and independently audit all KL equalities and induced logical actions. Focusing on distance-two codes with nondegenerate residues, we catalogue new nonadditive codes for dimensions $K\in\{2,3,4\}$ on up to six qubits, including high-order diagonal transversals, yielding $14,116$ new codes. From these data, the system abstracts closed-form families and constructs a residue-degenerate $((6,4,2))$ code implementing a transversal controlled-phase $\mathrm{diag}(1,1,1,i)$, illustrating how AI orchestration can drive rigorous, scalable code discovery.

**arXiv ID:** 2510.20728
</details>

<details>
<summary><strong>From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents</strong> - Jiaxuan Gao, Jiaao Chen, Chuyi He, Wei-Chen Wang, Shusheng Xu, Hanrui Wang, Di Jin, Yi Wu - [[pdf]](https://arxiv.org/pdf/2601.22607)</summary>

**Abstract:** Interactive tool-using agents must solve real-world tasks via multi-turn interaction with both humans and external environments, requiring dialogue state tracking, multi-step tool execution, while following complex instructions. Post-training such agents is challenging because synthesis for high-quality multi-turn tool-use data is difficult to scale, and reinforcement learning (RL) could face noisy signals caused by user simulation, leading to degraded training efficiency. We propose a unified framework that combines a self-evolving data agent with verifier-based RL. Our system, EigenData, is a hierarchical multi-agent engine that synthesizes tool-grounded dialogues together with executable per-instance checkers, and improves generation reliability via closed-loop self-evolving process that updates prompts and workflow. Building on the synthetic data, we develop an RL recipe that first fine-tunes the user model and then applies GRPO-style training with trajectory-level group-relative advantages and dynamic filtering, yielding consistent improvements beyond SFT. Evaluated on tau^2-bench, our best model reaches 73.0% pass^1 on Airline and 98.3% pass^1 on Telecom, matching or exceeding frontier models. Overall, our results suggest a scalable pathway for bootstrapping complex tool-using behaviors without expensive human annotation.

**arXiv ID:** 2601.22607
</details>

<details>
<summary><strong>Rank-and-Reason: Multi-Agent Collaboration Accelerates Zero-Shot Protein Mutation Prediction</strong> - Yang Tan, Yuanxi Yu, Can Wu, Bozitao Zhong, Mingchen Li, Guisheng Fan, Jiankang Zhu, Yafeng Liang, Nanqing Dong, Liang Hong - [[pdf]](https://arxiv.org/pdf/2602.00197)</summary>

**Abstract:** Zero-shot mutation prediction is vital for low-resource protein engineering, yet existing protein language models (PLMs) often yield statistically confident results that ignore fundamental biophysical constraints. Currently, selecting candidates for wet-lab validation relies on manual expert auditing of PLM outputs, a process that is inefficient, subjective, and highly dependent on domain expertise. To address this, we propose Rank-and-Reason (VenusRAR), a two-stage agentic framework to automate this workflow and maximize expected wet-lab fitness. In the Rank-Stage, a Computational Expert and Virtual Biologist aggregate a context-aware multi-modal ensemble, establishing a new Spearman correlation record of 0.551 (vs. 0.518) on ProteinGym. In the Reason-Stage, an agentic Expert Panel employs chain-of-thought reasoning to audit candidates against geometric and structural constraints, improving the Top-5 Hit Rate by up to 367% on ProteinGym-DMS99. The wet-lab validation on Cas12i3 nuclease further confirms the framework's efficacy, achieving a 46.7% positive rate and identifying two novel mutants with 4.23-fold and 5.05-fold activity improvements. Code and datasets are released on GitHub (this https URL).

**arXiv ID:** 2602.00197
</details>

<details>
<summary><strong>Human-Centric Traffic Signal Control for Equity: A Multi-Agent Action Branching Deep Reinforcement Learning Approach</strong> - Xiaocai Zhang, Neema Nassir, Lok Sang Chan, Milad Haghani - [[pdf]](https://arxiv.org/pdf/2602.02959)</summary>

**Abstract:** Coordinating traffic signals along multimodal corridors is challenging because many multi-agent deep reinforcement learning (DRL) approaches remain vehicle-centric and struggle with high-dimensional discrete action spaces. We propose MA2B-DDQN, a human-centric multi-agent action-branching double Deep Q-Network (DQN) framework that explicitly optimizes traveler-level equity. Our key contribution is an action-branching discrete control formulation that decomposes corridor control into (i) local, per-intersection actions that allocate green time between the next two phases and (ii) a single global action that selects the total duration of those phases. This decomposition enables scalable coordination under discrete control while reducing the effective complexity of joint decision-making. We also design a human-centric reward that penalizes the number of delayed individuals in the corridor, accounting for pedestrians, vehicle occupants, and transit passengers. Extensive evaluations across seven realistic traffic scenarios in Melbourne, Australia, demonstrate that our approach significantly reduces the number of impacted travelers, outperforming existing DRL and baseline methods. Experiments confirm the robustness of our model, showing minimal variance across diverse settings. This framework not only advocates for a fairer traffic signal system but also provides a scalable solution adaptable to varied urban traffic conditions.

**arXiv ID:** 2602.02959
</details>

<details>
<summary><strong>Co2PO: Coordinated Constrained Policy Optimization for Multi-Agent RL</strong> - Shrenik Patel, Christine Truong - [[pdf]](https://arxiv.org/pdf/2602.02970)</summary>

**Abstract:** Constrained multi-agent reinforcement learning (MARL) faces a fundamental tension between exploration and safety-constrained optimization. Existing leading approaches, such as Lagrangian methods, typically rely on global penalties or centralized critics that react to violations after they occur, often suppressing exploration and leading to over-conservatism. We propose Co2PO, a novel MARL communication-augmented framework that enables coordination-driven safety through selective, risk-aware communication. Co2PO introduces a shared blackboard architecture for broadcasting positional intent and yield signals, governed by a learned hazard predictor that proactively forecasts potential violations over an extended temporal horizon. By integrating these forecasts into a constrained optimization objective, Co2PO allows agents to anticipate and navigate collective hazards without the performance trade-offs inherent in traditional reactive constraints. We evaluate Co2PO across a suite of complex multi-agent safety benchmarks, where it achieves higher returns compared to leading constrained baselines while converging to cost-compliant policies at deployment. Ablation studies further validate the necessity of risk-triggered communication, adaptive gating, and shared memory components.

**arXiv ID:** 2602.02970
</details>

<details>
<summary><strong>HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic</strong> - Yu-Hsiang Chen, Wei-Jer Chang, Christian Kotulla, Thomas Keutgens, Steffen Runde, Tobias Moers, Christoph Klas, Wei Zhan, Masayoshi Tomizuka, Yi-Ting Chen - [[pdf]](https://arxiv.org/pdf/2602.03447)</summary>

**Abstract:** We present HetroD, a dataset and benchmark for developing autonomous driving systems in heterogeneous environments. HetroD targets the critical challenge of navi- gating real-world heterogeneous traffic dominated by vulner- able road users (VRUs), including pedestrians, cyclists, and motorcyclists that interact with vehicles. These mixed agent types exhibit complex behaviors such as hook turns, lane splitting, and informal right-of-way negotiation. Such behaviors pose significant challenges for autonomous vehicles but remain underrepresented in existing datasets focused on structured, lane-disciplined traffic. To bridge the gap, we collect a large- scale drone-based dataset to provide a holistic observation of traffic scenes with centimeter-accurate annotations, HD maps, and traffic signal states. We further develop a modular toolkit for extracting per-agent scenarios to support downstream task development. In total, the dataset comprises over 65.4k high- fidelity agent trajectories, 70% of which are from VRUs. HetroD supports modeling of VRU behaviors in dense, het- erogeneous traffic and provides standardized benchmarks for forecasting, planning, and simulation tasks. Evaluation results reveal that state-of-the-art prediction and planning models struggle with the challenges presented by our dataset: they fail to predict lateral VRU movements, cannot handle unstructured maneuvers, and exhibit limited performance in dense and multi-agent scenarios, highlighting the need for more robust approaches to heterogeneous traffic. See our project page for more examples: this https URL

**arXiv ID:** 2602.03447
</details>

</details>

<details open>
<summary><h2>Other Agent Research (14 papers)</h2></summary>

<details>
<summary><strong>Minimal Computational Preconditions for Subjective Perspective in Artificial Agents</strong> - Hongju Pae - [[pdf]](https://arxiv.org/pdf/2602.02902)</summary>

**Abstract:** This study operationalizes subjective perspective in artificial agents by grounding it in a minimal, phenomenologically motivated internal structure. The perspective is implemented as a slowly evolving global latent state that modulates fast policy dynamics without being directly optimized for behavioral consequences. In a reward-free environment with regime shifts, this latent structure exhibits direction-dependent hysteresis, while policy-level behavior remains comparatively reactive. I argue that such hysteresis constitutes a measurable signature of perspective-like subjectivity in machine systems.

**arXiv ID:** 2602.02902
</details>

<details>
<summary><strong>Generative Engine Optimization: A VLM and Agent Framework for Pinterest Acquisition Growth</strong> - Faye Zhang, Qianyu Cheng, Jasmine Wan, Vishwakarma Singh, Jinfeng Rao, Kofi Boakye - [[pdf]](https://arxiv.org/pdf/2602.02961)</summary>

**Abstract:** Large Language Models are fundamentally reshaping content discovery through AI-native search systems such as ChatGPT, Gemini, and Claude. Unlike traditional search engines that match keywords to documents, these systems infer user intent, synthesize multimodal evidence, and generate contextual answers directly on the search page, introducing a paradigm shift from Search Engine Optimization (SEO) to Generative Engine Optimization (GEO). For visual content platforms hosting billions of assets, this poses an acute challenge: individual images lack the semantic depth and authority signals that generative search prioritizes, risking disintermediation as user needs are satisfied in-place without site visits.
We present Pinterest GEO, a production-scale framework that pioneers reverse search design: rather than generating generic image captions describing what content is, we fine-tune Vision-Language Models (VLMs) to predict what users would actually search for, augmented this with AI agents that mine real-time internet trends to capture emerging search demand. These VLM-generated queries then drive construction of semantically coherent Collection Pages via multimodal embeddings, creating indexable aggregations optimized for generative retrieval. Finally, we employ hybrid VLM and two-tower ANN architectures to build authority-aware interlinking structures that propagate signals across billions of visual assets. Deployed at scale across billions of images and tens of millions of collections, GEO delivers 20\% organic traffic growth contributing to multi-million monthly active user (MAU) growth, demonstrating a principled pathway for visual platforms to thrive in the generative search era.

**arXiv ID:** 2602.02961
</details>

<details>
<summary><strong>General Agents Contain World Models, even under Partial Observability and Stochasticity</strong> - Santiago Cifuentes - [[pdf]](https://arxiv.org/pdf/2602.03146)</summary>

**Abstract:** Deciding whether an agent possesses a model of its surrounding world is a fundamental step toward understanding its capabilities and limitations. In [10], it was shown that, within a particular framework, every almost optimal and general agent necessarily contains sufficient knowledge of its environment to allow an approximate reconstruction of it by querying the agent as a black box. This result relied on the assumptions that the agent is deterministic and that the environment is fully observable.
In this work, we remove both assumptions by extending the theorem to stochastic agents operating in partially observable environments. Fundamentally, this shows that stochastic agents cannot avoid learning their environment through the usage of randomization. We also strengthen the result by weakening the notion of generality, proving that less powerful agents already contain a model of the world in which they operate.

**arXiv ID:** 2602.03146
</details>

<details>
<summary><strong>Mitigating Conversational Inertia in Multi-Turn Agents</strong> - Yang Wan, Zheng Cao, Zhenhao Zhang, Zhengwen Zeng, Shuheng Shen, Changhua Meng, Linchao Zhu - [[pdf]](https://arxiv.org/pdf/2602.03664)</summary>

**Abstract:** Large language models excel as few-shot learners when provided with appropriate demonstrations, yet this strength becomes problematic in multiturn agent scenarios, where LLMs erroneously mimic their own previous responses as few-shot examples. Through attention analysis, we identify conversational inertia, a phenomenon where models exhibit strong diagonal attention to previous responses, which is associated with imitation bias that constrains exploration. This reveals a tension when transforming few-shot LLMs into agents: longer context enriches environmental feedback for exploitation, yet also amplifies conversational inertia that undermines exploration. Our key insight is that for identical states, actions generated with longer contexts exhibit stronger inertia than those with shorter contexts, enabling construction of preference pairs without environment rewards. Based on this, we propose Context Preference Learning to calibrate model preferences to favor low-inertia responses over highinertia ones. We further provide context management strategies at inference time to balance exploration and exploitation. Experimental results across eight agentic environments and one deep research scenario validate that our framework reduces conversational inertia and achieves performance improvements.

**arXiv ID:** 2602.03664
</details>

<details>
<summary><strong>Agentic Observability: Automated Alert Triage for Adobe E-Commerce</strong> - Aprameya Bharadwaj, Kyle Tu - [[pdf]](https://arxiv.org/pdf/2602.02585)</summary>

**Abstract:** Modern enterprise systems exhibit complex interdependencies that make observability and incident response increasingly challenging. Manual alert triage, which typically involves log inspection, API verification, and cross-referencing operational knowledge bases, remains a major bottleneck in reducing mean recovery time (MTTR). This paper presents an agentic observability framework deployed within Adobe's e-commerce infrastructure that autonomously performs alert triage using a ReAct paradigm. Upon alert detection, the agent dynamically identifies the affected service, retrieves and analyzes correlated logs across distributed systems, and plans context-dependent actions such as handbook consultation, runbook execution, or retrieval-augmented analysis of recently deployed code. Empirical results from production deployment indicate a 90% reduction in mean time to insight compared to manual triage, while maintaining comparable diagnostic accuracy. Our results show that agentic AI enables an order-of-magnitude reduction in triage latency and a step-change in resolution accuracy, marking a pivotal shift toward autonomous observability in enterprise operations.

**arXiv ID:** 2602.02585
</details>

<details>
<summary><strong>Scaling Small Agents Through Strategy Auctions</strong> - Lisa Alazraki, William F. Shen, Yoram Bachrach, Akhil Mathur - [[pdf]](https://arxiv.org/pdf/2602.02751)</summary>

**Abstract:** Small language models are increasingly viewed as a promising, cost-effective approach to agentic AI, with proponents claiming they are sufficiently capable for agentic workflows. However, while smaller agents can closely match larger ones on simple tasks, it remains unclear how their performance scales with task complexity, when large models become necessary, and how to better leverage small agents for long-horizon workloads. In this work, we empirically show that small agents' performance fails to scale with task complexity on deep search and coding tasks, and we introduce Strategy Auctions for Workload Efficiency (SALE), an agent framework inspired by freelancer marketplaces. In SALE, agents bid with short strategic plans, which are scored by a systematic cost-value mechanism and refined via a shared auction memory, enabling per-task routing and continual self-improvement without training a separate router or running all models to completion. Across deep search and coding tasks of varying complexity, SALE reduces reliance on the largest agent by 53%, lowers overall cost by 35%, and consistently improves upon the largest agent's pass@1 with only a negligible overhead beyond executing the final trace. In contrast, established routers that rely on task descriptions either underperform the largest agent or fail to reduce cost -- often both -- underscoring their poor fit for agentic workflows. These results suggest that while small agents may be insufficient for complex workloads, they can be effectively "scaled up" through coordinated task allocation and test-time self-improvement. More broadly, they motivate a systems-level view of agentic AI in which performance gains come less from ever-larger individual models and more from market-inspired coordination mechanisms that organize heterogeneous agents into efficient, adaptive ecosystems.

**arXiv ID:** 2602.02751
</details>

<details>
<summary><strong>Internet of Agentic AI: Incentive-Compatible Distributed Teaming and Workflow</strong> - Ya-Ting Yang, Quanyan Zhu - [[pdf]](https://arxiv.org/pdf/2602.03145)</summary>

**Abstract:** Large language models (LLMs) have enabled a new class of agentic AI systems that reason, plan, and act by invoking external tools. However, most existing agentic architectures remain centralized and monolithic, limiting scalability, specialization, and interoperability. This paper proposes a framework for scalable agentic intelligence, termed the Internet of Agentic AI, in which autonomous, heterogeneous agents distributed across cloud and edge infrastructure dynamically form coalitions to execute task-driven workflows. We formalize a network-native model of agentic collaboration and introduce an incentive-compatible workflow-coalition feasibility framework that integrates capability coverage, network locality, and economic implementability. To enable scalable coordination, we formulate a minimum-effort coalition selection problem and propose a decentralized coalition formation algorithm. The proposed framework can operate as a coordination layer above the Model Context Protocol (MCP). A healthcare case study demonstrates how domain specialization, cloud-edge heterogeneity, and dynamic coalition formation enable scalable, resilient, and economically viable agentic workflows. This work lays the foundation for principled coordination and scalability in the emerging era of Internet of Agentic AI.

**arXiv ID:** 2602.03145
</details>

<details>
<summary><strong>When Should Agents Coordinate in Differentiable Sequential Decision Problems?</strong> - Caleb Probine, Su Ann Low, David Fridovich-Keil, Ufuk Topcu - [[pdf]](https://arxiv.org/pdf/2602.03674)</summary>

**Abstract:** Multi-robot teams must coordinate to operate effectively. When a team operates in an uncoordinated manner, and agents choose actions that are only individually optimal, the team's outcome can suffer. However, in many domains, coordination requires costly communication. We explore the value of coordination in a broad class of differentiable motion-planning problems. In particular, we model coordinated behavior as a spectrum: at one extreme, agents jointly optimize a common team objective, and at the other, agents make unilaterally optimal decisions given their individual decision variables, i.e., they operate at Nash equilibria. We then demonstrate that reasoning about coordination in differentiable motion-planning problems reduces to reasoning about the second-order properties of agents' objectives, and we provide algorithms that use this second-order reasoning to determine at which times a team of agents should coordinate.

**arXiv ID:** 2602.03674
</details>

<details>
<summary><strong>SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue</strong> - Yuqin Dai, Ning Gao, Wei Zhang, Jie Wang, Zichen Luo, Jinpeng Wang, Yujie Wang, Ruiyuan Wu, Chaozheng Wang - [[pdf]](https://arxiv.org/pdf/2602.03548)</summary>

**Abstract:** Large Language Models have demonstrated remarkable capabilities in open-domain dialogues. However, current methods exhibit suboptimal performance in service dialogues, as they rely on noisy, low-quality human conversation data. This limitation arises from data scarcity and the difficulty of simulating authentic, goal-oriented user behaviors. To address these issues, we propose SEAD (Self-Evolving Agent for Service Dialogue), a framework that enables agents to learn effective strategies without large-scale human annotations. SEAD decouples user modeling into two components: a Profile Controller that generates diverse user states to manage training curriculum, and a User Role-play Model that focuses on realistic role-playing. This design ensures the environment provides adaptive training scenarios rather than acting as an unfair adversary. Experiments demonstrate that SEAD significantly outperforms Open-source Foundation Models and Closed-source Commercial Models, improving task completion rate by 17.6% and dialogue efficiency by 11.1%. Code is available at: this https URL.

**arXiv ID:** 2602.03548
</details>

<details>
<summary><strong>Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments</strong> - Siwei Wu, Yizhi Li, Yuyang Song, Wei Zhang, Yang Wang, Riza Batista-Navarro, Xian Yang, Mingjie Tang, Bryan Dai, Jian Yang, Chenghua Lin - [[pdf]](https://arxiv.org/pdf/2602.01244)</summary>

**Abstract:** Training agentic models for terminal-based tasks critically depends on high-quality terminal trajectories that capture realistic long-horizon interactions across diverse domains. However, constructing such data at scale remains challenging due to two key requirements: \textbf{\emph{Executability}}, since each instance requires a suitable and often distinct Docker environment; and \textbf{\emph{Verifiability}}, because heterogeneous task outputs preclude unified, standardized verification. To address these challenges, we propose \textbf{TerminalTraj}, a scalable pipeline that (i) filters high-quality repositories to construct Dockerized execution environments, (ii) generates Docker-aligned task instances, and (iii) synthesizes agent trajectories with executable validation code. Using TerminalTraj, we curate 32K Docker images and generate 50,733 verified terminal trajectories across eight domains. Models trained on this data with the Qwen2.5-Coder backbone achieve consistent performance improvements on TerminalBench (TB), with gains of up to 20\% on TB~1.0 and 10\% on TB~2.0 over their respective backbones. Notably, \textbf{TerminalTraj-32B} achieves strong performance among models with fewer than 100B parameters, reaching 35.30\% on TB~1.0 and 22.00\% on TB~2.0, and demonstrates improved test-time scaling behavior. All code and data are available at this https URL.

**arXiv ID:** 2602.01244
</details>

<details>
<summary><strong>From Pragmas to Partners: A Symbiotic Evolution of Agentic High-Level Synthesis</strong> - Niansong Zhang, Sunwoo Kim, Shreesha Srinath, Zhiru Zhang - [[pdf]](https://arxiv.org/pdf/2602.01401)</summary>

**Abstract:** The rise of large language models has sparked interest in AI-driven hardware design, raising the question: does high-level synthesis (HLS) still matter in the agentic era? We argue that HLS remains essential. While we expect mature agentic hardware systems to leverage both HLS and RTL, this paper focuses on HLS and its role in enabling agentic optimization. HLS offers faster iteration cycles, portability, and design permutability that make it a natural layer for agentic optimization. This position paper makes three contributions. First, we explain why HLS serves as a practical abstraction layer and a golden reference for agentic hardware design. Second, we identify key limitations of current HLS tools, namely inadequate performance feedback, rigid interfaces, and limited debuggability that agents are uniquely positioned to address. Third, we propose a taxonomy for the symbiotic evolution of agentic HLS, clarifying how responsibility shifts from human designers to AI agents as systems advance from copilots to autonomous design partners.

**arXiv ID:** 2602.01401
</details>

<details>
<summary><strong>Accelerating Structured Chain-of-Thought in Autonomous Vehicles</strong> - Yi Gu, Yan Wang, Yuxiao Chen, Yurong You, Wenjie Luo, Yue Wang, Wenhao Ding, Boyi Li, Heng Yang, Boris Ivanovic, Marco Pavone - [[pdf]](https://arxiv.org/pdf/2602.02864)</summary>

**Abstract:** Chain-of-Thought (CoT) reasoning enhances the decision-making capabilities of vision-language-action models in autonomous driving, but its autoregressive nature introduces significant inference latency, making it impractical for real-time applications. To address this, we introduce FastDriveCoT, a novel parallel decoding method that accelerates template-structured CoT. Our approach decomposes the reasoning process into a dependency graph of distinct sub-tasks, such as identifying critical objects and summarizing traffic rules, some of which can be generated in parallel. By generating multiple independent reasoning steps concurrently within a single forward pass, we significantly reduce the number of sequential computations. Experiments demonstrate a 3-4$\times$ speedup in CoT generation and a substantial reduction in end-to-end latency across various model architectures, all while preserving the original downstream task improvements brought by incorporating CoT reasoning.

**arXiv ID:** 2602.02864
</details>

<details>
<summary><strong>A Unified Candidate Set with Scene-Adaptive Refinement via Diffusion for End-to-End Autonomous Driving</strong> - Zhengfei Wu, Shuaixi Pan, Shuohan Chen, Shuo Yang, Yanjun Huang - [[pdf]](https://arxiv.org/pdf/2602.03112)</summary>

**Abstract:** End-to-end autonomous driving is increasingly adopting a multimodal planning paradigm that generates multiple trajectory candidates and selects the final plan, making candidate-set design critical. A fixed trajectory vocabulary provides stable coverage in routine driving but often misses optimal solutions in complex interactions, while scene-adaptive refinement can cause over-correction in simple scenarios by unnecessarily perturbing already strong vocabulary this http URL propose CdDrive, which preserves the original vocabulary candidates and augments them with scene-adaptive candidates generated by vocabulary-conditioned diffusion denoising. Both candidate types are jointly scored by a shared selection module, enabling reliable performance across routine and highly interactive scenarios. We further introduce HATNA (Horizon-Aware Trajectory Noise Adapter) to improve the smoothness and geometric continuity of diffusion candidates via temporal smoothing and horizon-aware noise modulation. Experiments on NAVSIM v1 and NAVSIM v2 demonstrate leading performance, and ablations verify the contribution of each component.

**arXiv ID:** 2602.03112
</details>

<details>
<summary><strong>Multi-Player, Multi-Strategy Quantum Game Model for Interaction-Aware Decision-Making in Autonomous Driving</strong> - Karim Essalmi, Fernando Garrido, Fawzi Nashashibi - [[pdf]](https://arxiv.org/pdf/2602.03571)</summary>

**Abstract:** Although significant progress has been made in decision-making for automated driving, challenges remain for deployment in the real world. One challenge lies in addressing interaction-awareness. Most existing approaches oversimplify interactions between the ego vehicle and surrounding agents, and often neglect interactions among the agents themselves. A common solution is to model these interactions using classical game theory. However, its formulation assumes rational players, whereas human behavior is frequently uncertain or irrational. To address these challenges, we propose the Quantum Game Decision-Making (QGDM) model, a novel framework that combines classical game theory with quantum mechanics principles (such as superposition, entanglement, and interference) to tackle multi-player, multi-strategy decision-making problems. To the best of our knowledge, this is one of the first studies to apply quantum game theory to decision-making for automated driving. QGDM runs in real time on a standard computer, without requiring quantum hardware. We evaluate QGDM in simulation across various scenarios, including roundabouts, merging, and highways, and compare its performance with multiple baseline methods. Results show that QGDM significantly improves success rates and reduces collision rates compared to classical approaches, particularly in scenarios with high interaction.

**arXiv ID:** 2602.03571
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (38 papers)</h2></summary>

<details>
<summary><strong>RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents</strong> - Haitian Zhong, Jixiu Zhai, Lei Song, Jiang Bian, Qiang Liu, Tieniu Tan - [[pdf]](https://arxiv.org/pdf/2602.03025)</summary>

**Abstract:** Multi-turn tool calling is challenging for Large Language Models (LLMs) because rewards are sparse and exploration is expensive. A common recipe, SFT followed by GRPO, can stall when within-group reward variation is low (e.g., more rollouts in a group receive the all 0 or all 1 reward), making the group-normalized advantage uninformative and yielding vanishing updates. To address this problem, we propose RC-GRPO (Reward-Conditioned Group Relative Policy Optimization), which treats exploration as a controllable steering problem via discrete reward tokens. We first fine-tune a Reward-Conditioned Trajectory Policy (RCTP) on mixed-quality trajectories with reward goal special tokens (e.g., <|high_reward|>, <|low_reward|>) injected into the prompts, enabling the model to learn how to generate distinct quality trajectories on demand. Then during RL, we sample diverse reward tokens within each GRPO group and condition rollouts on the sampled token to improve within-group diversity, improving advantage gains. On the Berkeley Function Calling Leaderboard v4 (BFCLv4) multi-turn benchmark, our method yields consistently improved performance than baselines, and the performance on Qwen-2.5-7B-Instruct even surpasses all closed-source API models.

**arXiv ID:** 2602.03025
</details>

<details>
<summary><strong>TAME: A Trustworthy Test-Time Evolution of Agent Memory with Systematic Benchmarking</strong> - Yu Cheng, Jiuan Zhou, Yongkang Hu, Yihang Chen, Huichi Zhou, Mingang Chen, Zhizhong Zhang, Kun Shao, Yuan Xie, Zhaoxia Yin - [[pdf]](https://arxiv.org/pdf/2602.03224)</summary>

**Abstract:** Test-time evolution of agent memory serves as a pivotal paradigm for achieving AGI by bolstering complex reasoning through experience accumulation. However, even during benign task evolution, agent safety alignment remains vulnerable-a phenomenon known as Agent Memory Misevolution. To evaluate this phenomenon, we construct the Trust-Memevo benchmark to assess multi-dimensional trustworthiness during benign task evolution, revealing an overall decline in trustworthiness across various task domains and evaluation settings. To address this issue, we propose TAME, a dual-memory evolutionary framework that separately evolves executor memory to improve task performance by distilling generalizable methodologies, and evaluator memory to refine assessments of both safety and task utility based on historical feedback. Through a closed loop of memory filtering, draft generation, trustworthy refinement, execution, and dual-track memory updating, TAME preserves trustworthiness without sacrificing utility. Experiments demonstrate that TAME mitigates misevolution, achieving a joint improvement in both trustworthiness and task performance.

**arXiv ID:** 2602.03224
</details>

<details>
<summary><strong>Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis</strong> - Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Xuan Ren, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang - [[pdf]](https://arxiv.org/pdf/2602.03279)</summary>

**Abstract:** Advancing complex reasoning in large language models relies on high-quality, verifiable datasets, yet human annotation remains cost-prohibitive and difficult to scale. Current synthesis paradigms often face a recurring trade-off: maintaining structural validity typically restricts problem complexity, while relaxing constraints to increase difficulty frequently leads to inconsistent or unsolvable instances. To address this, we propose Agentic Proposing, a framework that models problem synthesis as a goal-driven sequential decision process where a specialized agent dynamically selects and composes modular reasoning skills. Through an iterative workflow of internal reflection and tool-use, we develop the Agentic-Proposer-4B using Multi-Granularity Policy Optimization (MGPO) to generate high-precision, verifiable training trajectories across mathematics, coding, and science. Empirical results demonstrate that downstream solvers trained on agent-synthesized data significantly outperform leading baselines and exhibit robust cross-domain generalization. Notably, a 30B solver trained on only 11,000 synthesized trajectories achieves a state-of-the-art 91.6% accuracy on AIME25, rivaling frontier-scale proprietary models such as GPT-5 and proving that a small volume of high-quality synthetic signals can effectively substitute for massive human-curated datasets.

**arXiv ID:** 2602.03279
</details>

<details>
<summary><strong>MeetBench-XL: Calibrated Multi-Dimensional Evaluation and Learned Dual-Policy Agents for Real-Time Meetings</strong> - Yuelin Hu, Jun Xu, Bingcong Lu, Zhengxue Cheng, Hongwei Hu, Ronghua Wu, Li Song - [[pdf]](https://arxiv.org/pdf/2602.03285)</summary>

**Abstract:** Enterprise meeting environments require AI assistants that handle diverse operational tasks, from rapid fact checking during live discussions to cross meeting analysis for strategic planning, under strict latency, cost, and privacy constraints. Existing meeting benchmarks mainly focus on simplified question answering and fail to reflect real world enterprise workflows, where queries arise organically from multi stakeholder collaboration, span long temporal contexts, and require tool augmented reasoning.
We address this gap through a grounded dataset and a learned agent framework. First, we introduce MeetAll, a bilingual and multimodal corpus derived from 231 enterprise meetings totaling 140 hours. Questions are injected using an enterprise informed protocol validated by domain expert review and human discriminability studies. Unlike purely synthetic benchmarks, this protocol is grounded in four enterprise critical dimensions: cognitive load, temporal context span, domain expertise, and actionable task execution, calibrated through interviews with stakeholders across finance, healthcare, and technology sectors.
Second, we propose MeetBench XL, a multi dimensional evaluation protocol aligned with human judgment that measures factual fidelity, intent alignment, response efficiency, structural clarity, and completeness. Third, we present MeetMaster XL, a learned dual policy agent that jointly optimizes query routing between fast and slow reasoning paths and tool invocation, including retrieval, cross meeting aggregation, and web search. A lightweight classifier enables accurate routing with minimal overhead, achieving a superior quality latency tradeoff over single model baselines. Experiments against commercial systems show consistent gains, supported by ablations, robustness tests, and a real world deployment case this http URL: this https URL.

**arXiv ID:** 2602.03285
</details>

<details>
<summary><strong>IntentRL: Training Proactive User-intent Agents for Open-ended Deep Research via Reinforcement Learning</strong> - Haohao Luo, Zexi Li, Yuexiang Xie, Wenhao Zhang, Yaliang Li, Ying Shen - [[pdf]](https://arxiv.org/pdf/2602.03468)</summary>

**Abstract:** Deep Research (DR) agents extend Large Language Models (LLMs) beyond parametric knowledge by autonomously retrieving and synthesizing evidence from large web corpora into long-form reports, enabling a long-horizon agentic paradigm. However, unlike real-time conversational assistants, DR is computationally expensive and time-consuming, creating an autonomy-interaction dilemma: high autonomy on ambiguous user queries often leads to prolonged execution with unsatisfactory outcomes. To address this, we propose IntentRL, a framework that trains proactive agents to clarify latent user intents before starting long-horizon research. To overcome the scarcity of open-ended research data, we introduce a scalable pipeline that expands a few seed samples into high-quality dialogue turns via a shallow-to-deep intent refinement graph. We further adopt a two-stage reinforcement learning (RL) strategy: Stage I applies RL on offline dialogues to efficiently learn general user-interaction behavior, while Stage II uses the trained agent and a user simulator for online rollouts to strengthen adaptation to diverse user feedback. Extensive experiments show that IntentRL significantly improves both intent hit rate and downstream task performance, outperforming the built-in clarify modules of closed-source DR agents and proactive LLM baselines.

**arXiv ID:** 2602.03468
</details>

<details>
<summary><strong>AOrchestra: Automating Sub-Agent Creation for Agentic Orchestration</strong> - Jianhao Ruan, Zhihao Xu, Yiran Peng, Fashen Ren, Zhaoyang Yu, Xinbing Liang, Jinyu Xiang, Bang Liu, Chenglin Wu, Yuyu Luo, Jiayi Zhang - [[pdf]](https://arxiv.org/pdf/2602.03786)</summary>

**Abstract:** Language agents have shown strong promise for task automation. Realizing this promise for increasingly complex, long-horizon tasks has driven the rise of a sub-agent-as-tools paradigm for multi-turn task solving. However, existing designs still lack a dynamic abstraction view of sub-agents, thereby hurting adaptability. We address this challenge with a unified, framework-agnostic agent abstraction that models any agent as a tuple Instruction, Context, Tools, Model. This tuple acts as a compositional recipe for capabilities, enabling the system to spawn specialized executors for each task on demand. Building on this abstraction, we introduce an agentic system AOrchestra, where the central orchestrator concretizes the tuple at each step: it curates task-relevant context, selects tools and models, and delegates execution via on-the-fly automatic agent creation. Such designs enable reducing human engineering efforts, and remain framework-agnostic with plug-and-play support for diverse agents as task executors. It also enables a controllable performance-cost trade-off, allowing the system to approach Pareto-efficient. Across three challenging benchmarks (GAIA, SWE-Bench, Terminal-Bench), AOrchestra achieves 16.28% relative improvement against the strongest baseline when paired with Gemini-3-Flash. The code is available at: this https URL

**arXiv ID:** 2602.03786
</details>

<details>
<summary><strong>Kimi K2.5: Visual Agentic Intelligence</strong> - Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S.H. Cai, Yuan Cao, Y. Charles, H.S. Che, Cheng Chen, Guanduo Chen, Huarong Chen, Jia Chen, Jiahao Chen, Jianlong Chen, Jun Chen, Kefan Chen, Liang Chen, Ruijue Chen, Xinhao Chen, Yanru Chen, Yanxu Chen, Yicun Chen, Yimin Chen, Yingjiang Chen, Yuankun Chen, Yujie Chen, Yutian Chen, Zhirong Chen, Ziwei Chen, Dazhi Cheng, Minghan Chu, Jialei Cui, Jiaqi Deng, Muxi Diao, Hao Ding, Mengfan Dong, Mengnan Dong, Yuxin Dong, Yuhao Dong, Angang Du, Chenzhuang Du, Dikang Du, Lingxiao Du, Yulun Du, Yu Fan, Shengjun Fang, Qiulin Feng, Yichen Feng, Garimugai Fu, Kelin Fu, Hongcheng Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Chengyang Gong, Xiaochen Gong, Zhuoma Gongque, Qizheng Gu, Xinran Gu, Yicheng Gu, Longyu Guan, Yuanying Guo, Xiaoru Hao, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Jiaxi Hu, Yangyang Hu, Zhenxing Hu, Ke Huang, Ruiyuan Huang, Weixiao Huang, Zhiqi Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yu Jing, Guokun Lai, Aidi Li, C. Li, Cheng Li, Fang Li, Guanghe Li, Guanyu Li, Haitao Li, Haoyang Li, Jia Li, Jingwei Li, Junxiong Li, Lincan Li, Mo Li, Weihong Li, Wentao Li, Xinhang Li, Xinhao Li, Yang Li, Yanhao Li, Yiwei Li - [[pdf]](https://arxiv.org/pdf/2602.02276)</summary>

**Abstract:** We introduce Kimi K2.5, an open-source multimodal agentic model designed to advance general agentic intelligence. K2.5 emphasizes the joint optimization of text and vision so that two modalities enhance each other. This includes a series of techniques such as joint text-vision pre-training, zero-vision SFT, and joint text-vision reinforcement learning. Building on this multimodal foundation, K2.5 introduces Agent Swarm, a self-directed parallel agent orchestration framework that dynamically decomposes complex tasks into heterogeneous sub-problems and executes them concurrently. Extensive evaluations show that Kimi K2.5 achieves state-of-the-art results across various domains including coding, vision, reasoning, and agentic tasks. Agent Swarm also reduces latency by up to $4.5\times$ over single-agent baselines. We release the post-trained Kimi K2.5 model checkpoint to facilitate future research and real-world applications of agentic intelligence.

**arXiv ID:** 2602.02276
</details>

<details>
<summary><strong>GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning</strong> - Yuyang Bai, Zhuofeng Li, Ping Nie, Jianwen Xie, Yu Zhang - [[pdf]](https://arxiv.org/pdf/2602.02518)</summary>

**Abstract:** Large language models (LLMs) increasingly rely on external knowledge to improve factuality, yet many real-world knowledge sources are organized as heterogeneous graphs rather than plain text. Reasoning over such graph-structured knowledge poses two key challenges: (1) navigating structured, schema-defined relations requires precise function calls rather than similarity-based retrieval, and (2) answering complex questions often demands multi-hop evidence aggregation through iterative information seeking. We propose GraphDancer, a reinforcement learning (RL) framework that teaches LLMs to navigate graphs by interleaving reasoning and function execution. To make RL effective for moderate-sized LLMs, we introduce a graph-aware curriculum that schedules training by the structural complexity of information-seeking trajectories using an easy-to-hard biased sampler. We evaluate GraphDancer on a multi-domain benchmark by training on one domain only and testing on unseen domains and out-of-distribution question types. Despite using only a 3B backbone, GraphDancer outperforms baselines equipped with either a 14B backbone or GPT-4o-mini, demonstrating robust cross-domain generalization of graph exploration and reasoning skills. Our code and models can be found at this https URL .

**arXiv ID:** 2602.02518
</details>

<details>
<summary><strong>CADENT: Gated Hybrid Distillation for Sample-Efficient Transfer in Reinforcement Learning</strong> - Mahyar Alinejad, Yue Wang, George Atia - [[pdf]](https://arxiv.org/pdf/2602.02532)</summary>

**Abstract:** Transfer learning promises to reduce the high sample complexity of deep reinforcement learning (RL), yet existing methods struggle with domain shift between source and target environments. Policy distillation provides powerful tactical guidance but fails to transfer long-term strategic knowledge, while automaton-based methods capture task structure but lack fine-grained action guidance. This paper introduces Context-Aware Distillation with Experience-gated Transfer (CADENT), a framework that unifies strategic automaton-based knowledge with tactical policy-level knowledge into a coherent guidance signal. CADENT's key innovation is an experience-gated trust mechanism that dynamically weighs teacher guidance against the student's own experience at the state-action level, enabling graceful adaptation to target domain specifics. Across challenging environments, from sparse-reward grid worlds to continuous control tasks, CADENT achieves 40-60\% better sample efficiency than baselines while maintaining superior asymptotic performance, establishing a robust approach for adaptive knowledge transfer in RL.

**arXiv ID:** 2602.02532
</details>

<details>
<summary><strong>Learning to Explore with Parameter-Space Noise: A Deep Dive into Parameter-Space Noise for Reinforcement Learning with Verifiable Rewards</strong> - Bizhe Bai, Xinyue Wang, Peng Ye, Tao Chen - [[pdf]](https://arxiv.org/pdf/2602.02555)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) improves LLM reasoning, yet growing evidence indicates an exploration ceiling: it often reweights existing solution traces rather than discovering new strategies, limiting gains under large sampling budgets (e.g., pass-at-256). We address this limitation with PSN-RLVR, which perturbs policy parameters before rollout generation to induce temporally consistent, trajectory-level exploration that better preserves long-horizon chain-of-thought coherence than action-space noise. To mitigate the resulting sampling-update mismatch, we incorporate truncated importance sampling (TIS). To avoid expensive KL-based adaptive noise control, we propose a computationally efficient real-time adaptive noise scheduler driven by a lightweight surrogate that combines semantic diversity with normalized self-certainty. Instantiated on GRPO, a widely used RLVR method, PSN-GRPO consistently expands the effective reasoning capability boundary across multiple mathematical reasoning benchmarks and model families, yielding higher pass-at-k under large sampling budgets and outperforming prior exploration-oriented RLVR methods (e.g., Pass-at-k-style training) while remaining orthogonal and thus composable for additional gains.

**arXiv ID:** 2602.02555
</details>

<details>
<summary><strong>Causal Flow Q-Learning for Robust Offline Reinforcement Learning</strong> - Mingxuan Li, Junzhe Zhang, Elias Bareinboim - [[pdf]](https://arxiv.org/pdf/2602.02847)</summary>

**Abstract:** Expressive policies based on flow-matching have been successfully applied in reinforcement learning (RL) more recently due to their ability to model complex action distributions from offline data. These algorithms build on standard policy gradients, which assume that there is no unmeasured confounding in the data. However, this condition does not necessarily hold for pixel-based demonstrations when a mismatch exists between the demonstrator's and the learner's sensory capabilities, leading to implicit confounding biases in offline data. We address the challenge by investigating the problem of confounded observations in offline RL from a causal perspective. We develop a novel causal offline RL objective that optimizes policies' worst-case performance that may arise due to confounding biases. Based on this new objective, we introduce a practical implementation that learns expressive flow-matching policies from confounded demonstrations, employing a deep discriminator to assess the discrepancy between the target policy and the nominal behavioral policy. Experiments across 25 pixel-based tasks demonstrate that our proposed confounding-robust augmentation procedure achieves a success rate 120\% that of confounding-unaware, state-of-the-art offline RL methods.

**arXiv ID:** 2602.02847
</details>

<details>
<summary><strong>Manifold-Constrained Energy-Based Transition Models for Offline Reinforcement Learning</strong> - Zeyu Fang, Zuyuan Zhang, Mahdi Imani, Tian Lan - [[pdf]](https://arxiv.org/pdf/2602.02900)</summary>

**Abstract:** Model-based offline reinforcement learning is brittle under distribution shift: policy improvement drives rollouts into state--action regions weakly supported by the dataset, where compounding model error yields severe value overestimation. We propose Manifold-Constrained Energy-based Transition Models (MC-ETM), which train conditional energy-based transition models using a manifold projection--diffusion negative sampler. MC-ETM learns a latent manifold of next states and generates near-manifold hard negatives by perturbing latent codes and running Langevin dynamics in latent space with the learned conditional energy, sharpening the energy landscape around the dataset support and improving sensitivity to subtle out-of-distribution deviations. For policy optimization, the learned energy provides a single reliability signal: rollouts are truncated when the minimum energy over sampled next states exceeds a threshold, and Bellman backups are stabilized via pessimistic penalties based on Q-value-level dispersion across energy-guided samples. We formalize MC-ETM through a hybrid pessimistic MDP formulation and derive a conservative performance bound separating in-support evaluation error from truncation risk. Empirically, MC-ETM improves multi-step dynamics fidelity and yields higher normalized returns on standard offline control benchmarks, particularly under irregular dynamics and sparse data coverage.

**arXiv ID:** 2602.02900
</details>

<details>
<summary><strong>CoBA-RL: Capability-Oriented Budget Allocation for Reinforcement Learning in LLMs</strong> - Zhiyuan Yao, Yi-Kai Zhang, Yuxin Chen, Yueqing Sun, Zishan Xu, Yu Yang, Tianhao Hu, Qi Gu, Hui Su, Xunliang Cai - [[pdf]](https://arxiv.org/pdf/2602.03048)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key approach for enhancing LLM this http URL, standard frameworks like Group Relative Policy Optimization (GRPO) typically employ a uniform rollout budget, leading to resource inefficiency. Moreover, existing adaptive methods often rely on instance-level metrics, such as task pass rates, failing to capture the model's dynamic learning state. To address these limitations, we propose CoBA-RL, a reinforcement learning algorithm designed to adaptively allocate rollout budgets based on the model's evolving capability. Specifically, CoBA-RL utilizes a Capability-Oriented Value function to map tasks to their potential training gains and employs a heap-based greedy strategy to efficiently self-calibrate the distribution of computational resources to samples with high training value. Extensive experiments demonstrate that our approach effectively orchestrates the trade-off between exploration and exploitation, delivering consistent generalization improvements across multiple challenging benchmarks. These findings underscore that quantifying sample training value and optimizing budget allocation are pivotal for advancing LLM post-training efficiency.

**arXiv ID:** 2602.03048
</details>

<details>
<summary><strong>Towards Considerate Embodied AI: Co-Designing Situated Multi-Site Healthcare Robots from Abstract Concepts to High-Fidelity Prototypes</strong> - Yuanchen Bai, Ruixiang Han, Niti Parikh, Wendy Ju, Angelique Taylor - [[pdf]](https://arxiv.org/pdf/2602.03054)</summary>

**Abstract:** Co-design is essential for grounding embodied artificial intelligence (AI) systems in real-world contexts, especially high-stakes domains such as healthcare. While prior work has explored multidisciplinary collaboration, iterative prototyping, and support for non-technical participants, few have interwoven these into a sustained co-design process. Such efforts often target one context and low-fidelity stages, limiting the generalizability of findings and obscuring how participants' ideas evolve. To address these limitations, we conducted a 14-week workshop with a multidisciplinary team of 22 participants, centered around how embodied AI can reduce non-value-added task burdens in three healthcare settings: emergency departments, long-term rehabilitation facilities, and sleep disorder clinics. We found that the iterative progression from abstract brainstorming to high-fidelity prototypes, supported by educational scaffolds, enabled participants to understand real-world trade-offs and generate more deployable solutions. We propose eight guidelines for co-designing more considerate embodied AI: attuned to context, responsive to social dynamics, mindful of expectations, and grounded in deployment. Project Page: this https URL

**arXiv ID:** 2602.03054
</details>

<details>
<summary><strong>Training and Simulation of Quadrupedal Robot in Adaptive Stair Climbing for Indoor Firefighting: An End-to-End Reinforcement Learning Approach</strong> - Baixiao Huang, Baiyu Huang, Yu Hou - [[pdf]](https://arxiv.org/pdf/2602.03087)</summary>

**Abstract:** Quadruped robots are used for primary searches during the early stages of indoor fires. A typical primary search involves quickly and thoroughly looking for victims under hazardous conditions and monitoring flammable materials. However, situational awareness in complex indoor environments and rapid stair climbing across different staircases remain the main challenges for robot-assisted primary searches. In this project, we designed a two-stage end-to-end deep reinforcement learning (RL) approach to optimize both navigation and locomotion. In the first stage, the quadrupeds, Unitree Go2, were trained to climb stairs in Isaac Lab's pyramid-stair terrain. In the second stage, the quadrupeds were trained to climb various realistic indoor staircases in the Isaac Lab engine, with the learned policy transferred from the previous stage. These indoor staircases are straight, L-shaped, and spiral, to support climbing tasks in complex environments. This project explores how to balance navigation and locomotion and how end-to-end RL methods can enable quadrupeds to adapt to different stair shapes. Our main contributions are: (1) A two-stage end-to-end RL framework that transfers stair-climbing skills from abstract pyramid terrain to realistic indoor stair topologies. (2) A centerline-based navigation formulation that enables unified learning of navigation and locomotion without hierarchical planning. (3) Demonstration of policy generalization across diverse staircases using only local height-map perception. (4) An empirical analysis of success, efficiency, and failure modes under increasing stair difficulty.

**arXiv ID:** 2602.03087
</details>

<details>
<summary><strong>Self-Hinting Language Models Enhance Reinforcement Learning</strong> - Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian - [[pdf]](https://arxiv.org/pdf/2602.03143)</summary>

**Abstract:** Group Relative Policy Optimization (GRPO) has recently emerged as a practical recipe for aligning large language models with verifiable objectives. However, under sparse terminal rewards, GRPO often stalls because rollouts within a group frequently receive identical rewards, causing relative advantages to collapse and updates to vanish. We propose self-hint aligned GRPO with privileged supervision (SAGE), an on-policy reinforcement learning framework that injects privileged hints during training to reshape the rollout distribution under the same terminal verifier reward. For each prompt $x$, the model samples a compact hint $h$ (e.g., a plan or decomposition) and then generates a solution $\tau$ conditioned on $(x,h)$. Crucially, the task reward $R(x,\tau)$ is unchanged; hints only increase within-group outcome diversity under finite sampling, preventing GRPO advantages from collapsing under sparse rewards. At test time, we set $h=\varnothing$ and deploy the no-hint policy without any privileged information. Moreover, sampling diverse self-hints serves as an adaptive curriculum that tracks the learner's bottlenecks more effectively than fixed hints from an initial policy or a stronger external model. Experiments over 6 benchmarks with 3 LLMs show that SAGE consistently outperforms GRPO, on average +2.0 on Llama-3.2-3B-Instruct, +1.2 on Qwen2.5-7B-Instruct and +1.3 on Qwen3-4B-Instruct. The code is available at this https URL.

**arXiv ID:** 2602.03143
</details>

<details>
<summary><strong>Intelligent Front-End Personalization: AI-Driven UI Adaptation</strong> - Mona Rajhans - [[pdf]](https://arxiv.org/pdf/2602.03154)</summary>

**Abstract:** Front-end personalization has traditionally relied on static designs or rule-based adaptations, which fail to fully capture user behavior patterns. This paper presents an AI driven approach for dynamic front-end personalization, where UI layouts, content, and features adapt in real-time based on predicted user behavior. We propose three strategies: dynamic layout adaptation using user path prediction, content prioritization through reinforcement learning, and a comparative analysis of AI-driven vs. rule-based personalization. Technical implementation details, algorithms, system architecture, and evaluation methods are provided to illustrate feasibility and performance gains.

**arXiv ID:** 2602.03154
</details>

<details>
<summary><strong>Reinforcement Learning with Promising Tokens for Large Language Models</strong> - Jing-Cheng Pang, Liang Lu, Xian Tang, Kun Jiang, Sijie Wu, Kai Zhang, Xubin Li - [[pdf]](https://arxiv.org/pdf/2602.03195)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a key paradigm for aligning and optimizing large language models (LLMs). Standard approaches treat the LLM as the policy and apply RL directly over the full vocabulary space. However, this formulation includes the massive tail of contextually irrelevant tokens in the action space, which could distract the policy from focusing on decision-making among the truly reasonable tokens. In this work, we verify that valid reasoning paths could inherently concentrate within a low-rank subspace. Based on this insight, we introduce Reinforcement Learning with Promising Tokens (RLPT), a framework that mitigates the action space issue by decoupling strategic decision-making from token generation. Specifically, RLPT leverages the semantic priors of the base model to identify a dynamic set of \emph{promising tokens} and constrains policy optimization exclusively to this refined subset via masking. Theoretical analysis and empirical results demonstrate that RLPT effectively reduces gradient variance, stabilizes the training process, and improves sample efficiency. Experiment results on math, coding, and telecom reasoning show that RLPT outperforms standard RL baselines and integrates effectively across various model sizes (4B and 8B) and RL algorithms (GRPO and DAPO).

**arXiv ID:** 2602.03195
</details>

<details>
<summary><strong>PEGRL: Improving Machine Translation by Post-Editing Guided Reinforcement Learning</strong> - Yunzhi Shen, Hao Zhou, Xin Huang, Xue Han, Junlan Feng, Shujian Huang - [[pdf]](https://arxiv.org/pdf/2602.03352)</summary>

**Abstract:** Reinforcement learning (RL) has shown strong promise for LLM-based machine translation, with recent methods such as GRPO demonstrating notable gains; nevertheless, translation-oriented RL remains challenged by noisy learning signals arising from Monte Carlo return estimation, as well as a large trajectory space that favors global exploration over fine-grained local optimization. We introduce \textbf{PEGRL}, a \textit{two-stage} RL framework that uses post-editing as an auxiliary task to stabilize training and guide overall optimization. At each iteration, translation outputs are sampled to construct post-editing inputs, allowing return estimation in the post-editing stage to benefit from conditioning on the current translation behavior, while jointly supporting both global exploration and fine-grained local optimization. A task-specific weighting scheme further balances the contributions of translation and post-editing objectives, yielding a biased yet more sample-efficient estimator. Experiments on English$\to$Finnish, English$\to$Turkish, and English$\leftrightarrow$Chinese show consistent gains over RL baselines, and for English$\to$Turkish, performance on COMET-KIWI is comparable to advanced LLM-based systems (DeepSeek-V3.2).

**arXiv ID:** 2602.03352
</details>

<details>
<summary><strong>OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering</strong> - Yifan Zhu, Xinyu Mu, Tao Feng, Zhonghong Ou, Yuning Gong, Haoran Luo - [[pdf]](https://arxiv.org/pdf/2602.03707)</summary>

**Abstract:** Long-horizon omnimodal question answering answers questions by reasoning over text, images, audio, and video. Despite recent progress on OmniLLMs, low-resource long audio-video QA still suffers from costly dense encoding, weak fine-grained retrieval, limited proactive planning, and no clear end-to-end this http URL address these issues, we propose OmniRAG-Agent, an agentic omnimodal QA method for budgeted long audio-video reasoning. It builds an image-audio retrieval-augmented generation module that lets an OmniLLM fetch short, relevant frames and audio snippets from external banks. Moreover, it uses an agent loop that plans, calls tools across turns, and merges retrieved evidence to answer complex queries. Furthermore, we apply group relative policy optimization to jointly improve tool use and answer quality over time. Experiments on OmniVideoBench, WorldSense, and Daily-Omni show that OmniRAG-Agent consistently outperforms prior methods under low-resource settings and achieves strong results, with ablations validating each component.

**arXiv ID:** 2602.03707
</details>

<details>
<summary><strong>Training Multi-Turn Search Agent via Contrastive Dynamic Branch Sampling</strong> - Yubao Zhao, Weiquan Huang, Sudong Wang, Ruochen Zhao, Chen Chen, Yao Shu, Chengwei Qin - [[pdf]](https://arxiv.org/pdf/2602.03719)</summary>

**Abstract:** Agentic reinforcement learning has enabled large language models to perform complex multi-turn planning and tool use. However, learning in long-horizon settings remains challenging due to sparse, trajectory-level outcome rewards. While prior tree-based methods attempt to mitigate this issue, they often suffer from high variance and computational inefficiency. Through empirical analysis of search agents, We identify a common pattern: performance diverges mainly due to decisions near the tail. Motivated by this observation, we propose Branching Relative Policy Optimization (BranPO), a value-free method that provides step-level contrastive supervision without dense rewards. BranPO truncates trajectories near the tail and resamples alternative continuations to construct contrastive suffixes over shared prefixes, reducing credit ambiguity in long-horizon rollouts. To further boost efficiency and stabilize training, we introduce difficulty-aware branch sampling to adapt branching frequency across tasks, and redundant step masking to suppress uninformative actions. Extensive experiments on various question answering benchmarks demonstrate that BranPO consistently outperforms strong baselines, achieving significant accuracy gains on long-horizon tasks without increasing the overall training budget. Our code is available at \href{this https URL}{code}.

**arXiv ID:** 2602.03719
</details>

<details>
<summary><strong>DEER: A Benchmark for Evaluating Deep Research Agents on Expert Report Generation</strong> - Janghoon Han, Heegyu Kim, Changho Lee, Dahm Lee, Min Hyung Park, Hosung Song, Stanley Jungkyu Choi, Moontae Lee, Honglak Lee - [[pdf]](https://arxiv.org/pdf/2512.17776)</summary>

**Abstract:** Recent advances in large language models have enabled deep research systems that generate expert-level reports through multi-step reasoning and evidence-based synthesis. However, evaluating such reports remains challenging: report quality is multifaceted, making it difficult to determine what to assess and by what criteria; LLM-based judges may miss errors that require domain expertise to identify; and because deep research relies on retrieved evidence, report-wide claim verification is also necessary. To address these issues, we propose DEER, a benchmark for evaluating expert-level deep research reports. DEER systematizes evaluation criteria with an expert-developed taxonomy (7 dimensions, 25 subdimensions) operationalized as 101 fine-grained rubric items. We also provide task-specific Expert Evaluation Guidance to support LLM-based judging. Alongside rubric-based assessment, we propose a claim verification architecture that verifies both cited and uncited claims and quantifies evidence quality. Experiments show that while current deep research systems can produce structurally plausible reports that cite external evidence, there is room for improvement in fulfilling expert-level user requests and achieving logical completeness. Beyond simple performance comparisons, DEER makes system strengths and limitations interpretable and provides diagnostic signals for improvement.

**arXiv ID:** 2512.17776
</details>

<details>
<summary><strong>SERA: Soft-Verified Efficient Repository Agents</strong> - Ethan Shen, Danny Tormoen, Saurabh Shah, Ali Farhadi, Tim Dettmers - [[pdf]](https://arxiv.org/pdf/2601.20789)</summary>

**Abstract:** Open-weight coding agents should hold a fundamental advantage over closed-source systems: they can be specialized to private codebases, encoding repository-specific information directly in their weights. Yet the cost and complexity of training has kept this advantage theoretical. We show it is now practical. We present Soft-Verified Efficient Repository Agents (SERA), an efficient method for training coding agents that enables the rapid and cheap creation of agents specialized to private codebases. Using only supervised finetuning (SFT), SERA achieves state-of-the-art results among fully open-source (open data, method, code) models while matching the performance of frontier open-weight models like Devstral-Small-2. Creating SERA models is 26x cheaper than reinforcement learning and 57x cheaper than previous synthetic data methods to reach equivalent performance. Our method, Soft Verified Generation (SVG), generates thousands of trajectories from a single code repository. Combined with cost-efficiency, this enables specialization to private codebases. Beyond repository specialization, we apply SVG to a larger corpus of codebases, generating over 200,000 synthetic trajectories. We use this dataset to provide detailed analysis of scaling laws, ablations, and confounding factors for training coding agents. Overall, we believe our work will greatly accelerate research on open coding agents and showcase the advantage of open-source models that can specialize to private codebases. We release SERA as the first model in Ai2's Open Coding Agents series, along with all our code, data, and Claude Code integration to support the research community.

**arXiv ID:** 2601.20789
</details>

<details>
<summary><strong>Kimi K2: Open Agentic Intelligence</strong> - Kimi Team, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Chenxiao Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Qizheng Gu, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yang Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T.Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Haoyu Lu, Lijun Lu, Yashuo Luo, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Zeyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi - [[pdf]](https://arxiv.org/pdf/2507.20534)</summary>

**Abstract:** We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments.
Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, with strengths in agentic capabilities. Notably, K2 obtains 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, and 47.3 on SWE-Bench Multilingual -- surpassing most open and closed-sourced baselines in non-thinking settings. It also exhibits strong capabilities in coding, mathematics, and reasoning tasks, with a score of 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. These results position Kimi K2 as one of the most capable open-source large language models to date, particularly in software engineering and agentic tasks. We release our base and post-trained model checkpoints to facilitate future research and applications of agentic intelligence.

**arXiv ID:** 2507.20534
</details>

<details>
<summary><strong>Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests</strong> - Jingjie Ning, João Coelho, Yibo Kong, Yunfan Long, Bruno Martins, João Magalhães, Jamie Callan, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2601.17617)</summary>

**Abstract:** LLM-powered search agents are increasingly being used for multi-step information seeking tasks, yet the IR community lacks empirical understanding of how agentic search sessions unfold and how retrieved evidence is used. This paper presents a large-scale log analysis of agentic search based on 14.44M search requests (3.97M sessions) collected from DeepResearchGym, i.e. an open-source search API accessed by external agentic clients. We sessionize the logs, assign session-level intents and step-wise query-reformulation labels using LLM-based annotation, and propose Context-driven Term Adoption Rate (CTAR) to quantify whether newly introduced query terms are traceable to previously retrieved evidence. Our analyses reveal distinctive behavioral patterns. First, over 90% of multi-turn sessions contain at most ten steps, and 89% of inter-step intervals fall under one minute. Second, behavior varies by intent. Fact-seeking sessions exhibit high repetition that increases over time, while sessions requiring reasoning sustain broader exploration. Third, agents reuse evidence across steps. On average, 54% of newly introduced query terms appear in the accumulated evidence context, with contributions from earlier steps beyond the most recent retrieval. The findings suggest that agentic search may benefit from repetition-aware early stopping, intent-adaptive retrieval budgets, and explicit cross-step context tracking. We plan to release the anonymized logs to support future research.

**arXiv ID:** 2601.17617
</details>

<details>
<summary><strong>Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards</strong> - Hieu Trung Nguyen, Bao Nguyen, Wenao Ma, Yuzhi Zhao, Ruifeng She, Viet Anh Nguyen - [[pdf]](https://arxiv.org/pdf/2602.01601)</summary>

**Abstract:** Sampling efficiency is a key bottleneck in reinforcement learning with verifiable rewards. Existing group-based policy optimization methods, such as GRPO, allocate a fixed number of rollouts for all training prompts. This uniform allocation implicitly treats all prompts as equally informative, and could lead to inefficient computational budget usage and impede training progress. We introduce VIP, a Variance-Informed Predictive allocation strategy that allocates a given rollout budget to the prompts in the incumbent batch to minimize the expected gradient variance of the policy update. At each iteration, VIP uses a lightweight Gaussian process model to predict per-prompt success probabilities based on recent rollouts. These probability predictions are translated into variance estimates, which are then fed into a convex optimization problem to determine the optimal rollout allocations under a hard compute budget constraint. Empirical results show that VIP consistently improves sampling efficiency and achieves higher performance than uniform or heuristic allocation strategies in multiple benchmarks.

**arXiv ID:** 2602.01601
</details>

<details>
<summary><strong>Hypersonic Flow Control: Generalized Deep Reinforcement Learning for Hypersonic Intake Unstart Control under Uncertainty</strong> - Trishit Mondal, Ameya D. Jagtap - [[pdf]](https://arxiv.org/pdf/2602.02531)</summary>

**Abstract:** The hypersonic unstart phenomenon poses a major challenge to reliable air-breathing propulsion at Mach 5 and above, where strong shock-boundary-layer interactions and rapid pressure fluctuations can destabilize inlet operation. Here, we demonstrate a deep reinforcement learning (DRL)- based active flow control strategy to control unstart in a canonical two-dimensional hypersonic inlet at Mach 5 and Reynolds number $5\times 10^6$. The in-house CFD solver enables high-fidelity simulations with adaptive mesh refinement, resolving key flow features, including shock motion, boundary-layer dynamics, and flow separation, that are essential for learning physically consistent control policies suitable for real-time deployment. The DRL controller robustly stabilizes the inlet over a wide range of back pressures representative of varying combustion chamber conditions. It further generalizes to previously unseen scenarios, including different back-pressure levels, Reynolds numbers, and sensor configurations, while operating with noisy measurements, thereby demonstrating strong zero-shot generalization. Control remains robust in the presence of noisy sensor measurements, and a minimal, optimally selected sensor set achieves comparable performance, enabling practical implementation. These results establish a data-driven approach for real-time hypersonic flow control under realistic operational uncertainties.

**arXiv ID:** 2602.02531
</details>

<details>
<summary><strong>Maximum Likelihood Reinforcement Learning</strong> - Fahim Tajwar, Guanning Zeng, Yueer Zhou, Yuda Song, Daman Arora, Yiding Jiang, Jeff Schneider, Ruslan Salakhutdinov, Haiwen Feng, Andrea Zanette - [[pdf]](https://arxiv.org/pdf/2602.02710)</summary>

**Abstract:** Reinforcement learning is the method of choice to train models in sampling-based setups with binary outcome feedback, such as navigation, code generation, and mathematical problem solving. In such settings, models implicitly induce a likelihood over correct rollouts. However, we observe that reinforcement learning does not maximize this likelihood, and instead optimizes only a lower-order approximation. Inspired by this observation, we introduce Maximum Likelihood Reinforcement Learning (MaxRL), a sampling-based framework to approximate maximum likelihood using reinforcement learning techniques. MaxRL addresses the challenges of non-differentiable sampling by defining a compute-indexed family of sample-based objectives that interpolate between standard reinforcement learning and exact maximum likelihood as additional sampling compute is allocated. The resulting objectives admit a simple, unbiased policy-gradient estimator and converge to maximum likelihood optimization in the infinite-compute limit. Empirically, we show that MaxRL Pareto-dominates existing methods in all models and tasks we tested, achieving up to 20x test-time scaling efficiency gains compared to its GRPO-trained counterpart. We also observe MaxRL to scale better with additional data and compute. Our results suggest MaxRL is a promising framework for scaling RL training in correctness based settings.

**arXiv ID:** 2602.02710
</details>

<details>
<summary><strong>Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion</strong> - Dan Haramati, Carl Qi, Tal Daniel, Amy Zhang, Aviv Tamar, George Konidaris - [[pdf]](https://arxiv.org/pdf/2602.02722)</summary>

**Abstract:** We propose a hierarchical entity-centric framework for offline Goal-Conditioned Reinforcement Learning (GCRL) that combines subgoal decomposition with factored structure to solve long-horizon tasks in domains with multiple entities. Achieving long-horizon goals in complex environments remains a core challenge in Reinforcement Learning (RL). Domains with multiple entities are particularly difficult due to their combinatorial complexity. GCRL facilitates generalization across goals and the use of subgoal structure, but struggles with high-dimensional observations and combinatorial state-spaces, especially under sparse reward. We employ a two-level hierarchy composed of a value-based GCRL agent and a factored subgoal-generating conditional diffusion model. The RL agent and subgoal generator are trained independently and composed post hoc through selective subgoal generation based on the value function, making the approach modular and compatible with existing GCRL algorithms. We introduce new variations to benchmark tasks that highlight the challenges of multi-entity domains, and show that our method consistently boosts performance of the underlying RL agent on image-based long-horizon tasks with sparse rewards, achieving over 150% higher success rates on the hardest task in our suite and generalizing to increasing horizons and numbers of entities. Rollout videos are provided at: this https URL

**arXiv ID:** 2602.02722
</details>

<details>
<summary><strong>How Does the Lagrangian Guide Safe Reinforcement Learning through Diffusion Models?</strong> - Xiaoyuan Cheng, Wenxuan Yuan, Boyang Li, Yuanchao Xu, Yiming Yang, Hao Liang, Bei Peng, Robert Loftin, Zhuo Sun, Yukun Hu - [[pdf]](https://arxiv.org/pdf/2602.02924)</summary>

**Abstract:** Diffusion policy sampling enables reinforcement learning (RL) to represent multimodal action distributions beyond suboptimal unimodal Gaussian policies. However, existing diffusion-based RL methods primarily focus on offline settings for reward maximization, with limited consideration of safety in online settings. To address this gap, we propose Augmented Lagrangian-Guided Diffusion (ALGD), a novel algorithm for off-policy safe RL. By revisiting optimization theory and energy-based model, we show that the instability of primal-dual methods arises from the non-convex Lagrangian landscape. In diffusion-based safe RL, the Lagrangian can be interpreted as an energy function guiding the denoising dynamics. Counterintuitively, direct usage destabilizes both policy generation and training. ALGD resolves this issue by introducing an augmented Lagrangian that locally convexifies the energy landscape, yielding a stabilized policy generation and training process without altering the distribution of the optimal policy. Theoretical analysis and extensive experiments demonstrate that ALGD is both theoretically grounded and empirically effective, achieving strong and stable performance across diverse environments.

**arXiv ID:** 2602.02924
</details>

<details>
<summary><strong>Neural Predictor-Corrector: Solving Homotopy Problems with Reinforcement Learning</strong> - Jiayao Mai, Bangyan Liao, Zhenjun Zhao, Yingping Zeng, Haoang Li, Javier Civera, Tailin Wu, Yi Zhou, Peidong Liu - [[pdf]](https://arxiv.org/pdf/2602.03086)</summary>

**Abstract:** The Homotopy paradigm, a general principle for solving challenging problems, appears across diverse domains such as robust optimization, global optimization, polynomial root-finding, and sampling. Practical solvers for these problems typically follow a predictor-corrector (PC) structure, but rely on hand-crafted heuristics for step sizes and iteration termination, which are often suboptimal and task-specific. To address this, we unify these problems under a single framework, which enables the design of a general neural solver. Building on this unified view, we propose Neural Predictor-Corrector (NPC), which replaces hand-crafted heuristics with automatically learned policies. NPC formulates policy selection as a sequential decision-making problem and leverages reinforcement learning to automatically discover efficient strategies. To further enhance generalization, we introduce an amortized training mechanism, enabling one-time offline training for a class of problems and efficient online inference on new instances. Experiments on four representative homotopy problems demonstrate that our method generalizes effectively to unseen instances. It consistently outperforms classical and specialized baselines in efficiency while demonstrating superior stability across tasks, highlighting the value of unifying homotopy methods into a single neural framework.

**arXiv ID:** 2602.03086
</details>

<details>
<summary><strong>StepScorer: Accelerating Reinforcement Learning with Step-wise Scoring and Psychological Regret Modeling</strong> - Zhe Xu - [[pdf]](https://arxiv.org/pdf/2602.03171)</summary>

**Abstract:** Reinforcement learning algorithms often suffer from slow convergence due to sparse reward signals, particularly in complex environments where feedback is delayed or infrequent. This paper introduces the Psychological Regret Model (PRM), a novel approach that accelerates learning by incorporating regret-based feedback signals after each decision step. Rather than waiting for terminal rewards, PRM computes a regret signal based on the difference between the expected value of the optimal action and the value of the action taken in each state. This transforms sparse rewards into dense feedback signals through a step-wise scoring framework, enabling faster convergence. We demonstrate that PRM achieves stable performance approximately 36\% faster than traditional Proximal Policy Optimization (PPO) in benchmark environments such as Lunar Lander. Our results indicate that PRM is particularly effective in continuous control tasks and environments with delayed feedback, making it suitable for real-world applications such as robotics, finance, and adaptive education where rapid policy adaptation is critical. The approach formalizes human-inspired counterfactual thinking as a computable regret signal, bridging behavioral economics and reinforcement learning.

**arXiv ID:** 2602.03171
</details>

<details>
<summary><strong>From Scalar Rewards to Potential Trends: Shaping Potential Landscapes for Model-Based Reinforcement Learning</strong> - Yao-Hui Li, Zeyu Wang, Xin Li, Wei Pang, Yingfang Yuan, Zhengkun Chen, Boya Zhang, Riashat Islam, Alex Lamb, Yonggang Zhang - [[pdf]](https://arxiv.org/pdf/2602.03201)</summary>

**Abstract:** Model-based reinforcement learning (MBRL) achieves high sample efficiency by simulating future trajectories with learned dynamics and reward models. However, its effectiveness is severely compromised in sparse reward settings. The core limitation lies in the standard paradigm of regressing ground-truth scalar rewards: in sparse environments, this yields a flat, gradient-free landscape that fails to provide directional guidance for planning. To address this challenge, we propose Shaping Landscapes with Optimistic Potential Estimates (SLOPE), a novel framework that shifts reward modeling from predicting scalars to constructing informative potential landscapes. SLOPE employs optimistic distributional regression to estimate high-confidence upper bounds, which amplifies rare success signals and ensures sufficient exploration gradients. Evaluations on 30+ tasks across 5 benchmarks demonstrate that SLOPE consistently outperforms leading baselines in fully sparse, semi-sparse, and dense rewards.

**arXiv ID:** 2602.03201
</details>

<details>
<summary><strong>medR: Reward Engineering for Clinical Offline Reinforcement Learning via Tri-Drive Potential Functions</strong> - Qianyi Xu, Gousia Habib, Feng Wu, Yanrui Du, Zhihui Chen, Swapnil Mishra, Dilruk Perera, Mengling Feng - [[pdf]](https://arxiv.org/pdf/2602.03305)</summary>

**Abstract:** Reinforcement Learning (RL) offers a powerful framework for optimizing dynamic treatment regimes (DTRs). However, clinical RL is fundamentally bottlenecked by reward engineering: the challenge of defining signals that safely and effectively guide policy learning in complex, sparse offline environments. Existing approaches often rely on manual heuristics that fail to generalize across diverse pathologies. To address this, we propose an automated pipeline leveraging Large Language Models (LLMs) for offline reward design and verification. We formulate the reward function using potential functions consisted of three core components: survival, confidence, and competence. We further introduce quantitative metrics to rigorously evaluate and select the optimal reward structure prior to deployment. By integrating LLM-driven domain knowledge, our framework automates the design of reward functions for specific diseases while significantly enhancing the performance of the resulting policies.

**arXiv ID:** 2602.03305
</details>

<details>
<summary><strong>AROLA: A Modular Layered Architecture for Scaled Autonomous Racing</strong> - Fam Shihata, Mohammed Abdelazim, Ahmed Hussein - [[pdf]](https://arxiv.org/pdf/2602.02730)</summary>

**Abstract:** Autonomous racing has advanced rapidly, particularly on scaled platforms, and software stacks must evolve accordingly. In this work, AROLA is introduced as a modular, layered software architecture in which fragmented and monolithic designs are reorganized into interchangeable layers and components connected through standardized ROS 2 interfaces. The autonomous-driving pipeline is decomposed into sensing, pre-processing, perception, localization and mapping, planning, behavior, control, and actuation, enabling rapid module replacement and objective benchmarking without reliance on custom message definitions. To support consistent performance evaluation, a Race Monitor framework is introduced as a lightweight system through which lap timing, trajectory quality, and computational load are logged in real time and standardized post-race analyses are generated. AROLA is validated in simulation and on hardware using the RoboRacer platform, including deployment at the 2025 RoboRacer IV25 competition. Together, AROLA and Race Monitor demonstrate that modularity, transparent interfaces, and systematic evaluation can accelerate development and improve reproducibility in scaled autonomous racing.

**arXiv ID:** 2602.02730
</details>

<details>
<summary><strong>AffordanceGrasp-R1:Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping</strong> - Dingyi Zhou, Mu He, Zhuowei Fang, Xiangtong Yao, Yinlong Liu, Alois Knoll, Hu Cao - [[pdf]](https://arxiv.org/pdf/2602.03547)</summary>

**Abstract:** We introduce AffordanceGrasp-R1, a reasoning-driven affordance segmentation framework for robotic grasping that combines a chain-of-thought (CoT) cold-start strategy with reinforcement learning to enhance deduction and spatial grounding. In addition, we redesign the grasping pipeline to be more context-aware by generating grasp candidates from the global scene point cloud and subsequently filtering them using instruction-conditioned affordance masks. Extensive experiments demonstrate that AffordanceGrasp-R1 consistently outperforms state-of-the-art (SOTA) methods on benchmark datasets, and real-world robotic grasping evaluations further validate its robustness and generalization under complex language-conditioned manipulation scenarios.

**arXiv ID:** 2602.03547
</details>

<details>
<summary><strong>LAGEA: Language Guided Embodied Agents for Robotic Manipulation</strong> - Abdul Monaf Chowdhury, Akm Moshiur Rahman Mazumder, Rabeya Akter, Safaeid Hossain Arib - [[pdf]](https://arxiv.org/pdf/2509.23155)</summary>

**Abstract:** Robotic manipulation benefits from foundation models that describe goals, but today's agents still lack a principled way to learn from their own mistakes. We ask whether natural language can serve as feedback, an error-reasoning signal that helps embodied agents diagnose what went wrong and correct course. We introduce LAGEA (Language Guided Embodied Agents), a framework that turns episodic, schema-constrained reflections from a vision language model (VLM) into temporally grounded guidance for reinforcement learning. LAGEA summarizes each attempt in concise language, localizes the decisive moments in the trajectory, aligns feedback with visual state in a shared representation, and converts goal progress and feedback agreement into bounded, step-wise shaping rewards whose influence is modulated by an adaptive, failure-aware coefficient. This design yields dense signals early when exploration needs direction and gracefully recedes as competence grows. On the Meta-World MT10 and Robotic Fetch embodied manipulation benchmark, LAGEA improves average success over the state-of-the-art (SOTA) methods by 9.0% on random goals, 5.3% on fixed goals, and 17% on fetch tasks, while converging faster. These results support our hypothesis: language, when structured and grounded in time, is an effective mechanism for teaching robots to self-reflect on mistakes and make better choices.

**arXiv ID:** 2509.23155
</details>

<details>
<summary><strong>RFS: Reinforcement learning with Residual flow steering for dexterous manipulation</strong> - Entong Su, Tyler Westenbroek, Anusha Nagabandi, Abhishek Gupta - [[pdf]](https://arxiv.org/pdf/2602.01789)</summary>

**Abstract:** Imitation learning has emerged as an effective approach for bootstrapping sequential decision-making in robotics, achieving strong performance even in high-dimensional dexterous manipulation tasks. Recent behavior cloning methods further leverage expressive generative models, such as diffusion models and flow matching, to represent multimodal action distributions. However, policies pretrained in this manner often exhibit limited generalization and require additional fine-tuning to achieve robust performance at deployment time. Such adaptation must preserve the global exploration benefits of pretraining while enabling rapid correction of local execution errors. We propose Residual Flow Steering(RFS), a data-efficient reinforcement learning framework for adapting pretrained generative policies. RFS steers a pretrained flow-matching policy by jointly optimizing a residual action and a latent noise distribution, enabling complementary forms of exploration: local refinement through residual corrections and global exploration through latent-space modulation. This design allows efficient adaptation while retaining the expressive structure of the pretrained policy. We demonstrate the effectiveness of RFS on dexterous manipulation tasks, showing efficient fine-tuning in both simulation and real-world settings when adapting pretrained base policies. Project website:this https URL.

**arXiv ID:** 2602.01789
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
