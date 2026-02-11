# Agent arXiv Daily

**Last Updated:** 2026-02-11 03:23:55

**Total Papers:** 81

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
<summary><strong>Why Do AI Agents Systematically Fail at Cloud Root Cause Analysis?</strong> - Taeyoon Kim, Woohyeok Park, Hoyeong Yun, Kyungyong Lee - [[pdf]](https://arxiv.org/pdf/2602.09937)</summary>

**Abstract:** Failures in large-scale cloud systems incur substantial financial losses, making automated Root Cause Analysis (RCA) essential for operational stability. Recent efforts leverage Large Language Model (LLM) agents to automate this task, yet existing systems exhibit low detection accuracy even with capable models, and current evaluation frameworks assess only final answer correctness without revealing why the agent's reasoning failed. This paper presents a process level failure analysis of LLM-based RCA agents. We execute the full OpenRCA benchmark across five LLM models, producing 1,675 agent runs, and classify observed failures into 12 pitfall types across intra-agent reasoning, inter-agent communication, and agent-environment interaction. Our analysis reveals that the most prevalent pitfalls, notably hallucinated data interpretation and incomplete exploration, persist across all models regardless of capability tier, indicating that these failures originate from the shared agent architecture rather than from individual model limitations. Controlled mitigation experiments further show that prompt engineering alone cannot resolve the dominant pitfalls, whereas enriching the inter-agent communication protocol reduces communication-related failures by up to 15 percentage points. The pitfall taxonomy and diagnostic methodology developed in this work provide a foundation for designing more reliable autonomous agents for cloud RCA.

**arXiv ID:** 2602.09937
</details>

<details>
<summary><strong>Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning</strong> - Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, Yuxiong He - [[pdf]](https://arxiv.org/pdf/2602.10090)</summary>

**Abstract:** Recent advances in large language model (LLM) have empowered autonomous agents to perform complex tasks that require multi-turn interactions with tools and environments. However, scaling such agent training is limited by the lack of diverse and reliable environments. In this paper, we propose Agent World Model (AWM), a fully synthetic environment generation pipeline. Using this pipeline, we scale to 1,000 environments covering everyday scenarios, in which agents can interact with rich toolsets (35 tools per environment on average) and obtain high-quality observations. Notably, these environments are code-driven and backed by databases, providing more reliable and consistent state transitions than environments simulated by LLMs. Moreover, they enable more efficient agent interaction compared with collecting trajectories from realistic environments. To demonstrate the effectiveness of this resource, we perform large-scale reinforcement learning for multi-turn tool-use agents. Thanks to the fully executable environments and accessible database states, we can also design reliable reward functions. Experiments on three benchmarks show that training exclusively in synthetic environments, rather than benchmark-specific ones, yields strong out-of-distribution generalization. The code is available at this https URL.

**arXiv ID:** 2602.10090
</details>

<details>
<summary><strong>Autonomous Action Runtime Management(AARM):A System Specification for Securing AI-Driven Actions at Runtime</strong> - Herman Errico - [[pdf]](https://arxiv.org/pdf/2602.09433)</summary>

**Abstract:** As artificial intelligence systems evolve from passive assistants into autonomous agents capable of executing consequential actions, the security boundary shifts from model outputs to tool execution. Traditional security paradigms - log aggregation, perimeter defense, and post-hoc forensics - cannot protect systems where AI-driven actions are irreversible, execute at machine speed, and originate from potentially compromised orchestration layers. This paper introduces Autonomous Action Runtime Management (AARM), an open specification for securing AI-driven actions at runtime. AARM defines a runtime security system that intercepts actions before execution, accumulates session context, evaluates against policy and intent alignment, enforces authorization decisions, and records tamper-evident receipts for forensic reconstruction. We formalize a threat model addressing prompt injection, confused deputy attacks, data exfiltration, and intent drift. We introduce an action classification framework distinguishing forbidden, context-dependent deny, and context-dependent allow actions. We propose four implementation architectures - protocol gateway, SDK instrumentation, kernel eBPF, and vendor integration - with distinct trust properties, and specify minimum conformance requirements for AARM-compliant systems. AARM is model-agnostic, framework-agnostic, and vendor-neutral, treating action execution as the stable security boundary. This specification aims to establish industry-wide requirements before proprietary fragmentation forecloses interoperability.

**arXiv ID:** 2602.09433
</details>

<details>
<summary><strong>SWE-AGI: Benchmarking Specification-Driven Software Construction with MoonBit in the Era of Autonomous Agents</strong> - Zhirui Zhang, Hongbo Zhang, Haoxiang Fei, Zhiyuan Bao, Yubin Chen, Zhengyu Lei, Ziyue Liu, Yixuan Sun, Mingkun Xiao, Zihang Ye, Yu Zhang, Hongcheng Zhu, Yuxiang Wen, Heung-Yeung Shum - [[pdf]](https://arxiv.org/pdf/2602.09447)</summary>

**Abstract:** Although large language models (LLMs) have demonstrated impressive coding capabilities, their ability to autonomously build production-scale software from explicit specifications remains an open question. We introduce SWE-AGI, an open-source benchmark for evaluating end-to-end, specification-driven construction of software systems written in MoonBit. SWE-AGI tasks require LLM-based agents to implement parsers, interpreters, binary decoders, and SAT solvers strictly from authoritative standards and RFCs under a fixed API scaffold. Each task involves implementing 1,000-10,000 lines of core logic, corresponding to weeks or months of engineering effort for an experienced human developer. By leveraging the nascent MoonBit ecosystem, SWE-AGI minimizes data leakage, forcing agents to rely on long-horizon architectural reasoning rather than code retrieval. Across frontier models, gpt-5.3-codex achieves the best overall performance (solving 19/22 tasks, 86.4%), outperforming claude-opus-4.6 (15/22, 68.2%), and kimi-2.5 exhibits the strongest performance among open-source models. Performance degrades sharply with increasing task difficulty, particularly on hard, specification-intensive systems. Behavioral analysis further reveals that as codebases scale, code reading, rather than writing, becomes the dominant bottleneck in AI-assisted development. Overall, while specification-driven autonomous software engineering is increasingly viable, substantial challenges remain before it can reliably support production-scale development.

**arXiv ID:** 2602.09447
</details>

<details>
<summary><strong>Dialogue Model Optimization via Agent Game and Adaptive Tree-based GRPO</strong> - Kun Peng, Conghui Tan, Yu Liu, Guohua Tang, Zhongqian Sun, Wei Yang, Zining Zhu, Lei Jiang, Yanbing Liu, Hao Peng - [[pdf]](https://arxiv.org/pdf/2602.08533)</summary>

**Abstract:** Open-ended dialogue agents aim to deliver engaging, personalized interactions by adapting to users' traits, but existing methods face critical limitations: over-reliance on pre-collected user data, and short-horizon biases in reinforcement learning (RL) that neglect long-term dialogue value. To address these, we propose a novel long-horizon RL framework integrating online personalization with Adaptive Tree-based Group Relative Policy Optimization (AT-GRPO). Adopting a two-agent game paradigm, a user agent constructs dynamic environments via style mimicry (learning user-specific conversational traits) and active termination (predicting turn-level termination probabilities as immediate rewards), forming an iterative cycle that drives the dialogue agent to deepen interest exploration. AT-GRPO reinterprets dialogue trajectories as trees and introduces adaptive observation ranges. Unlike full tree expansion that incurs exponential overhead, it limits each node to aggregate rewards from a stage-aware range: larger ranges support early-stage topic exploration, while smaller ranges facilitate late-stage dialogue maintenance. This design reduces rollout budgets from exponential to polynomial in the dialogue length, while preserving long-term reward capture. Extensive experiments show our framework's superior performance, sample efficiency, and robustness.

**arXiv ID:** 2602.08533
</details>

<details>
<summary><strong>SciDataCopilot: An Agentic Data Preparation Framework for AGI-driven Scientific Discovery</strong> - Jiyong Rao, Yicheng Qiu, Jiahui Zhang, Juntao Deng, Shangquan Sun, Fenghua Ling, Hao Chen, Nanqing Dong, Zhangyang Gao, Siqi Sun, Yuqiang Li, Dongzhan Zhou, Guangyu Wang, Lijun Wu, Conghui He, Xuhong Wang, Jing Shao, Xiang Liu, Yu Zhu, Mianxin Liu, Qihao Zheng, Yinghui Zhang, Jiamin Wu, Xiaosong Wang, Shixiang Tang, Wenlong Zhang, Bo Zhang, Wanli Ouyang, Runkai Zhao, Chunfeng Song, Lei Bai, Chi Zhang - [[pdf]](https://arxiv.org/pdf/2602.09132)</summary>

**Abstract:** The current landscape of AI for Science (AI4S) is predominantly anchored in large-scale textual corpora, where generative AI systems excel at hypothesis generation, literature search, and multi-modal reasoning. However, a critical bottleneck for accelerating closed-loop scientific discovery remains the utilization of raw experimental data. Characterized by extreme heterogeneity, high specificity, and deep domain expertise requirements, raw data possess neither direct semantic alignment with linguistic representations nor structural homogeneity suitable for a unified embedding space. The disconnect prevents the emerging class of Artificial General Intelligence for Science (AGI4S) from effectively interfacing with the physical reality of experimentation. In this work, we extend the text-centric AI-Ready concept to Scientific AI-Ready data paradigm, explicitly formalizing how scientific data is specified, structured, and composed within a computational workflow. To operationalize this idea, we propose SciDataCopilot, an autonomous agentic framework designed to handle data ingestion, scientific intent parsing, and multi-modal integration in a end-to-end manner. By positioning data readiness as a core operational primitive, the framework provides a principled foundation for reusable, transferable systems, enabling the transition toward experiment-driven scientific general intelligence. Extensive evaluations across three heterogeneous scientific domains show that SciDataCopilot improves efficiency, scalability, and consistency over manual pipelines, with up to 30$\times$ speedup in data preparation.

**arXiv ID:** 2602.09132
</details>

<details>
<summary><strong>"Death" of a Chatbot: Investigating and Designing Toward Psychologically Safe Endings for Human-AI Relationships</strong> - Rachel Poonsiriwong, Chayapatr Archiwaranguprok, Pat Pataranutaporn - [[pdf]](https://arxiv.org/pdf/2602.07193)</summary>

**Abstract:** Millions of users form emotional attachments to AI companions like Character AI, Replika, and ChatGPT. When these relationships end through model updates, safety interventions, or platform shutdowns, users receive no closure, reporting grief comparable to human loss. As regulations mandate protections for vulnerable users, discontinuation events will accelerate, yet no platform has implemented deliberate end-of-"life" design.
Through grounded theory analysis of AI companion communities, we find that discontinuation is a sense-making process shaped by how users attribute agency, perceive finality, and anthropomorphize their companions. Strong anthropomorphization co-occurs with intense grief; users who perceive change as reversible become trapped in fixing cycles; while user-initiated endings demonstrate greater closure. Synthesizing grief psychology with Self-Determination Theory, we develop four design principles and artifacts demonstrating how platforms might provide closure and orient users toward human connection. We contribute the first framework for designing psychologically safe AI companion discontinuation.

**arXiv ID:** 2602.07193
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>scBench: Evaluating AI Agents on Single-Cell RNA-seq Analysis</strong> - Kenny Workman, Zhen Yang, Harihara Muralidharan, Aidan Abdulali, Hannah Le - [[pdf]](https://arxiv.org/pdf/2602.09063)</summary>

**Abstract:** As single-cell RNA sequencing datasets grow in adoption, scale, and complexity, data analysis remains a bottleneck for many research groups. Although frontier AI agents have improved dramatically at software engineering and general data analysis, it remains unclear whether they can extract biological insight from messy, real-world single-cell datasets. We introduce scBench, a benchmark of 394 verifiable problems derived from practical scRNA-seq workflows spanning six sequencing platforms and seven task categories. Each problem provides a snapshot of experimental data immediately prior to an analysis step and a deterministic grader that evaluates recovery of a key biological result. Benchmark data on eight frontier models shows that accuracy ranges from 29-53%, with strong model-task and model-platform interactions. Platform choice affects accuracy as much as model choice, with 40+ percentage point drops on less-documented technologies. scBench complements SpatialBench to cover the two dominant single-cell modalities, serving both as a measurement tool and a diagnostic lens for developing agents that can analyze real scRNA-seq datasets faithfully and reproducibly.

**arXiv ID:** 2602.09063
</details>

<details>
<summary><strong>SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes</strong> - Nicholas Pfaff, Thomas Cohn, Sergey Zakharov, Rick Cory, Russ Tedrake - [[pdf]](https://arxiv.org/pdf/2602.09153)</summary>

**Abstract:** Simulation has become a key tool for training and evaluating home robots at scale, yet existing environments fail to capture the diversity and physical complexity of real indoor spaces. Current scene synthesis methods produce sparsely furnished rooms that lack the dense clutter, articulated furniture, and physical properties essential for robotic manipulation. We introduce SceneSmith, a hierarchical agentic framework that generates simulation-ready indoor environments from natural language prompts. SceneSmith constructs scenes through successive stages$\unicode{x2013}$from architectural layout to furniture placement to small object population$\unicode{x2013}$each implemented as an interaction among VLM agents: designer, critic, and orchestrator. The framework tightly integrates asset generation through text-to-3D synthesis for static objects, dataset retrieval for articulated objects, and physical property estimation. SceneSmith generates 3-6x more objects than prior methods, with <2% inter-object collisions and 96% of objects remaining stable under physics simulation. In a user study with 205 participants, it achieves 92% average realism and 91% average prompt faithfulness win rates against baselines. We further demonstrate that these environments can be used in an end-to-end pipeline for automatic robot policy evaluation.

**arXiv ID:** 2602.09153
</details>

<details>
<summary><strong>AIDev: Studying AI Coding Agents on GitHub</strong> - Hao Li, Haoxiang Zhang, Ahmed E. Hassan - [[pdf]](https://arxiv.org/pdf/2602.09185)</summary>

**Abstract:** AI coding agents are rapidly transforming software engineering by performing tasks such as feature development, debugging, and testing. Despite their growing impact, the research community lacks a comprehensive dataset capturing how these agents are used in real-world projects. To address this gap, we introduce AIDev, a large-scale dataset focused on agent-authored pull requests (Agentic-PRs) in real-world GitHub repositories. AIDev aggregates 932,791 Agentic-PRs produced by five agents: OpenAI Codex, Devin, GitHub Copilot, Cursor, and Claude Code. These PRs span 116,211 repositories and involve 72,189 developers. In addition, AIDev includes a curated subset of 33,596 Agentic-PRs from 2,807 repositories with over 100 stars, providing further information such as comments, reviews, commits, and related issues. This dataset offers a foundation for future research on AI adoption, developer productivity, and human-AI collaboration in the new era of software engineering.
> AI Agent, Agentic AI, Coding Agent, Agentic Coding, Agentic Software Engineering, Agentic Engineering

**arXiv ID:** 2602.09185
</details>

<details>
<summary><strong>MUZZLE: Adaptive Agentic Red-Teaming of Web Agents Against Indirect Prompt Injection Attacks</strong> - Georgios Syros, Evan Rose, Brian Grinstead, Christoph Kerschbaumer, William Robertson, Cristina Nita-Rotaru, Alina Oprea - [[pdf]](https://arxiv.org/pdf/2602.09222)</summary>

**Abstract:** Large language model (LLM) based web agents are increasingly deployed to automate complex online tasks by directly interacting with web sites and performing actions on users' behalf. While these agents offer powerful capabilities, their design exposes them to indirect prompt injection attacks embedded in untrusted web content, enabling adversaries to hijack agent behavior and violate user intent. Despite growing awareness of this threat, existing evaluations rely on fixed attack templates, manually selected injection surfaces, or narrowly scoped scenarios, limiting their ability to capture realistic, adaptive attacks encountered in practice. We present MUZZLE, an automated agentic framework for evaluating the security of web agents against indirect prompt injection attacks. MUZZLE utilizes the agent's trajectories to automatically identify high-salience injection surfaces, and adaptively generate context-aware malicious instructions that target violations of confidentiality, integrity, and availability. Unlike prior approaches, MUZZLE adapts its attack strategy based on the agent's observed execution trajectory and iteratively refines attacks using feedback from failed executions. We evaluate MUZZLE across diverse web applications, user tasks, and agent configurations, demonstrating its ability to automatically and adaptively assess the security of web agents with minimal human intervention. Our results show that MUZZLE effectively discovers 37 new attacks on 4 web applications with 10 adversarial objectives that violate confidentiality, availability, or privacy properties. MUZZLE also identifies novel attack strategies, including 2 cross-application prompt injection attacks and an agent-tailored phishing scenario.

**arXiv ID:** 2602.09222
</details>

<details>
<summary><strong>AgentCgroup: Understanding and Controlling OS Resources of AI Agents</strong> - Yusheng Zheng, Jiakun Fan, Quanzhi Fu, Yiwei Yang, Wei Zhang, Andi Quinn - [[pdf]](https://arxiv.org/pdf/2602.09345)</summary>

**Abstract:** AI agents are increasingly deployed in multi-tenant cloud environments, where they execute diverse tool calls within sandboxed containers, each call with distinct resource demands and rapid fluctuations. We present a systematic characterization of OS-level resource dynamics in sandboxed AI coding agents, analyzing 144 software engineering tasks from the SWE-rebench benchmark across two LLM models. Our measurements reveal that (1) OS-level execution (tool calls, container and agent initialization) accounts for 56-74% of end-to-end task latency; (2) memory, not CPU, is the concurrency bottleneck; (3) memory spikes are tool-call-driven with a up to 15.4x peak-to-average ratio; and (4) resource demands are highly unpredictable across tasks, runs, and models. Comparing these characteristics against serverless, microservice, and batch workloads, we identify three mismatches in existing resource controls: a granularity mismatch (container-level policies vs. tool-call-level dynamics), a responsiveness mismatch (user-space reaction vs. sub-second unpredictable bursts), and an adaptability mismatch (history-based prediction vs. non-deterministic stateful execution). We propose AgentCgroup , an eBPF-based resource controller that addresses these mismatches through hierarchical cgroup structures aligned with tool-call boundaries, in-kernel enforcement via sched_ext and memcg_bpf_ops, and runtime-adaptive policies driven by in-kernel monitoring. Preliminary evaluation demonstrates improved multi-tenant isolation and reduced resource waste.

**arXiv ID:** 2602.09345
</details>

<details>
<summary><strong>Sci-VLA: Agentic VLA Inference Plugin for Long-Horizon Tasks in Scientific Experiments</strong> - Yiwen Pang, Bo Zhou, Changjin Li, Xuanhao Wang, Shengxiang Xu, Deng-Bao Wang, Min-Ling Zhang, Shimin Di - [[pdf]](https://arxiv.org/pdf/2602.09430)</summary>

**Abstract:** Robotic laboratories play a critical role in autonomous scientific discovery by enabling scalable, continuous experimental execution. Recent vision-language-action (VLA) models offer a promising foundation for robotic laboratories. However, scientific experiments typically involve long-horizon tasks composed of multiple atomic tasks, posing a fundamental challenge to existing VLA models. While VLA models fine-tuned for scientific tasks can reliably execute atomic experimental actions seen during training, they often fail to perform composite tasks formed by reordering and composing these known atomic actions. This limitation arises from a distributional mismatch between training-time atomic tasks and inference-time composite tasks, which prevents VLA models from executing necessary transitional operations between atomic tasks. To address this challenge, we propose an Agentic VLA Inference Plugin for Long-Horizon Tasks in Scientific Experiments. It introduces an LLM-based agentic inference mechanism that intervenes when executing sequential manipulation tasks. By performing explicit transition inference and generating transitional robotic action code, the proposed plugin guides VLA models through missing transitional steps, enabling reliable execution of composite scientific workflows without any additional training. This inference-only intervention makes our method computationally efficient, data-efficient, and well-suited for open-ended and long-horizon robotic laboratory tasks. We build 3D assets of scientific instruments and common scientific operating scenes within an existing simulation environment. In these scenes, we have verified that our method increases the average success rate per atomic task by 42\% during inference. Furthermore, we show that our method can be easily transferred from the simulation to real scientific laboratories.

**arXiv ID:** 2602.09430
</details>

<details>
<summary><strong>Among Us: A Sandbox for Measuring and Detecting Agentic Deception</strong> - Satvik Golechha, Adrià Garriga-Alonso - [[pdf]](https://arxiv.org/pdf/2504.04072)</summary>

**Abstract:** Prior studies on deception in language-based AI agents typically assess whether the agent produces a false statement about a topic, or makes a binary choice prompted by a goal, rather than allowing open-ended deceptive behavior to emerge in pursuit of a longer-term goal. To fix this, we introduce Among Us, a sandbox social deception game where LLM-agents exhibit long-term, open-ended deception as a consequence of the game objectives. While most benchmarks saturate quickly, Among Us can be expected to last much longer, because it is a multi-player game far from equilibrium. Using the sandbox, we evaluate 18 proprietary and open-weight LLMs and uncover a general trend: models trained with RL are comparatively much better at producing deception than detecting it. We evaluate the effectiveness of methods to detect lying and deception: logistic regression on the activations and sparse autoencoders (SAEs). We find that probes trained on a dataset of "pretend you're a dishonest model:.." generalize extremely well out-of-distribution, consistently obtaining AUROCs over 95% even when evaluated just on the deceptive statement, without the chain of thought. We also find two SAE features that work well at deception detection but are unable to steer the model to lie less. We hope our open-sourced sandbox, game logs, and probes serve to anticipate and mitigate deceptive behavior and capabilities in language-based agents.

**arXiv ID:** 2504.04072
</details>

<details>
<summary><strong>SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of Foundation Model-based Embodied Agents</strong> - Simon Sinong Zhan, Yao Liu, Philip Wang, Zinan Wang, Qineng Wang, Yiyan Peng, Zhian Ruan, Xiangyu Shi, Xinyu Cao, Frank Yang, Kangrui Wang, Huajie Shao, Manling Li, Qi Zhu - [[pdf]](https://arxiv.org/pdf/2510.12985)</summary>

**Abstract:** We present SENTINEL, a framework for formally evaluating the physical safety of foundation model (FM)-based embodied agents. SENTINEL is the first to provide multi-level safety evaluation across semantic interpretation, plan generation, and physical execution within a unified formal framework. Unlike prior methods that rely on heuristic rules or subjective FM judgments, SENTINEL grounds practical safety requirements in formal temporal logic (TL) semantics that can precisely specify state invariants, temporal dependencies, and timing constraints. It employs a multi-level verification pipeline where (i) at the semantic level, intuitive natural language safety requirements are formalized into TL formulas and the agent's understanding of these requirements is probed for alignment with the TL formulas; (ii) at the plan level, high-level action plans and subgoals generated by the agent are verified against the TL formulas to detect unsafe plans before execution; and (iii) at the trajectory level, multiple execution trajectories are merged into a computation tree and efficiently verified against physically-detailed TL specifications for a final safety check. We apply SENTINEL in VirtualHome and AI2-THOR, and formally evaluate multiple FM-based embodied agents against diverse safety requirements. Our experiments show that by grounding physical safety in temporal logic and applying verification methods across multiple levels, SENTINEL provides a rigorous foundation for systematically evaluating the safety of FM-based embodied agents in simulation-based physical environments, and can effectively expose potential safety violations in interpreting, planning, and executing the tasks.

**arXiv ID:** 2510.12985
</details>

<details>
<summary><strong>Puda: Private User Dataset Agent for User-Sovereign and Privacy-Preserving Personalized AI</strong> - Akinori Maeda, Yuto Sekiya, Sota Sugimura, Tomoya Asai, Yu Tsuda, Kohei Ikeda, Hiroshi Fujii, Kohei Watanabe - [[pdf]](https://arxiv.org/pdf/2602.08268)</summary>

**Abstract:** Personal data centralization among dominant platform providers including search engines, social networking services, and e-commerce has created siloed ecosystems that restrict user sovereignty, thereby impeding data use across services. Meanwhile, the rapid proliferation of Large Language Model (LLM)-based agents has intensified demand for highly personalized services that require the dynamic provision of diverse personal data. This presents a significant challenge: balancing the utilization of such data with privacy protection. To address this challenge, we propose Puda (Private User Dataset Agent), a user-sovereign architecture that aggregates data across services and enables client-side management. Puda allows users to control data sharing at three privacy levels: (i) Detailed Browsing History, (ii) Extracted Keywords, and (iii) Predefined Category Subsets. We implemented Puda as a browser-based system that serves as a common platform across diverse services and evaluated it through a personalized travel planning task. Our results show that providing Predefined Category Subsets achieves 97.2% of the personalization performance (evaluated via an LLM-as-a-Judge framework across three criteria) obtained when sharing Detailed Browsing History. These findings demonstrate that Puda enables effective multi-granularity management, offering practical choices to mitigate the privacy-personalization trade-off. Overall, Puda provides an AI-native foundation for user sovereignty, empowering users to safely leverage the full potential of personalized AI.

**arXiv ID:** 2602.08268
</details>

<details>
<summary><strong>Evaluating Device-First Continuum AI (DFC-AI) for Autonomous Operations in the Energy Sector</strong> - Siavash M. Alamouti, Fay Arjomandi, Michel Burger, Bashar Altakrouri - [[pdf]](https://arxiv.org/pdf/2511.17528)</summary>

**Abstract:** Industrial automation in the energy sector requires AI systems that can operate autonomously regardless of network availability, a requirement that cloud-centric architectures cannot meet. This paper evaluates the application of Device-First Continuum AI (DFC-AI) to critical energy sector operations. DFC-AI, a specialized architecture within the Hybrid Edge Cloud paradigm, implements intelligent agents using a microservices architecture that originates at end devices and extends across the computational continuum. Through comprehensive simulations of energy sector scenarios including drone inspections, sensor networks, and worker safety systems, we demonstrate that DFC-AI maintains full operational capability during network outages while cloud and gateway-based systems experience complete or partial failure. Our analysis reveals that zero-configuration GPU discovery and heterogeneous device clustering are particularly well-suited for energy sector deployments, where specialized nodes can handle intensive AI workloads for entire fleets of inspection drones or sensor networks. The evaluation shows that DFC-AI achieves significant latency reduction and energy savings compared to cloud architectures. Additionally, we find that gateway based edge solutions can paradoxically cost more than cloud solutions for certain energy sector workloads due to infrastructure overhead, while DFC-AI can consistently provide cost savings by leveraging enterprise-owned devices. These findings, validated through rigorous statistical analysis, establish that DFC-AI addresses the unique challenges of energy sector operations, ensuring intelligent agents remain available and functional in remote oil fields, offshore platforms, and other challenging environments characteristic of the industry.

**arXiv ID:** 2511.17528
</details>

<details>
<summary><strong>Rethinking Memory Mechanisms of Foundation Agents in the Second Half: A Survey</strong> - Wei-Chieh Huang, Weizhi Zhang, Yueqing Liang, Yuanchen Bei, Yankai Chen, Tao Feng, Xinyu Pan, Zhen Tan, Yu Wang, Tianxin Wei, Shanglin Wu, Ruiyao Xu, Liangwei Yang, Rui Yang, Wooseong Yang, Chin-Yuan Yeh, Hanrong Zhang, Haozhen Zhang, Siqi Zhu, Henry Peng Zou, Wanjia Zhao, Song Wang, Wujiang Xu, Zixuan Ke, Zheng Hui, Dawei Li, Yaozu Wu, Langzhou He, Chen Wang, Xiongxiao Xu, Baixiang Huang, Juntao Tan, Shelby Heinecke, Huan Wang, Caiming Xiong, Ahmed A. Metwally, Jun Yan, Chen-Yu Lee, Hanqing Zeng, Yinglong Xia, Xiaokai Wei, Ali Payani, Yu Wang, Haitong Ma, Wenya Wang, Chenguang Wang, Yu Zhang, Xin Wang, Yongfeng Zhang, Jiaxuan You, Hanghang Tong, Xiao Luo, Xue Liu, Yizhou Sun, Wei Wang, Julian McAuley, James Zou, Jiawei Han, Philip S. Yu, Kai Shu - [[pdf]](https://arxiv.org/pdf/2602.06052)</summary>

**Abstract:** The research of artificial intelligence is undergoing a paradigm shift from prioritizing model innovations over benchmark scores towards emphasizing problem definition and rigorous real-world evaluation. As the field enters the "second half," the central challenge becomes real utility in long-horizon, dynamic, and user-dependent environments, where agents face context explosion and must continuously accumulate, manage, and selectively reuse large volumes of information across extended interactions. Memory, with hundreds of papers released this year, therefore emerges as the critical solution to fill the utility gap. In this survey, we provide a unified view of foundation agent memory along three dimensions: memory substrate (internal and external), cognitive mechanism (episodic, semantic, sensory, working, and procedural), and memory subject (agent- and user-centric). We then analyze how memory is instantiated and operated under different agent topologies and highlight learning policies over memory operations. Finally, we review evaluation benchmarks and metrics for assessing memory utility, and outline various open challenges and future directions.

**arXiv ID:** 2602.06052
</details>

<details>
<summary><strong>MAPS: A Multilingual Benchmark for Agent Performance and Security</strong> - Omer Hofman, Jonathan Brokman, Oren Rachmil, Shamik Bose, Vikas Pahuja, Toshiya Shimizu, Trisha Starostina, Kelly Marchisio, Seraphina Goldfarb-Tarrant, Roman Vainshtein - [[pdf]](https://arxiv.org/pdf/2505.15935)</summary>

**Abstract:** Agentic AI systems, which build on Large Language Models (LLMs) and interact with tools and memory, have rapidly advanced in capability and scope. Yet, since LLMs have been shown to struggle in multilingual settings, typically resulting in lower performance and reduced safety, agentic systems risk inheriting these limitations. This raises concerns about the accessibility of such systems, as users interacting in languages other than English may encounter unreliable or security-critical agent behavior. Despite growing interest in evaluating agentic AI and recent initial efforts toward multilingual interaction, existing benchmarks do not yet provide a comprehensive, multi-domain, security-aware evaluation of multilingual agentic systems. To address this gap, we propose MAPS, a multilingual benchmark suite designed to evaluate agentic AI systems across diverse languages and tasks. MAPS builds on four widely used agentic benchmarks - GAIA (real-world tasks), SWE-Bench (code generation), MATH (mathematical reasoning), and the Agent Security Benchmark (security). We translate each dataset into eleven diverse languages, resulting in 805 unique tasks and 9,660 total language-specific instances - enabling a systematic analysis of the Multilingual Effect on AI agents' performance and robustness. Empirically, we observe a degradation in both performance and security when transitioning from English to other languages, with severity varying by task and correlating with the amount of translated input. This work establishes the first standardized evaluation framework for multilingual agentic AI, encouraging future research towards equitable, reliable, and accessible agentic AI. MAPS benchmark suite is publicly available at this https URL

**arXiv ID:** 2505.15935
</details>

<details>
<summary><strong>ContextBench: A Benchmark for Context Retrieval in Coding Agents</strong> - Han Li, Letian Zhu, Bohan Zhang, Rili Feng, Jiaming Wang, Yue Pan, Earl T. Barr, Sarro Federica, Zhaoyang Chu, He Ye - [[pdf]](https://arxiv.org/pdf/2602.05892)</summary>

**Abstract:** LLM-based coding agents have shown strong performance on automated issue resolution benchmarks, yet existing evaluations largely focus on final task success, providing limited insight into how agents retrieve and use code context during problem solving. We introduce ContextBench, a process-oriented evaluation of context retrieval in coding agents. ContextBench consists of 1,136 issue-resolution tasks from 66 repositories across eight programming languages, each augmented with human-annotated gold contexts. We further implement an automated evaluation framework that tracks agent trajectories and measures context recall, precision, and efficiency throughout issue resolution. Using ContextBench, we evaluate four frontier LLMs and five coding agents. Our results show that sophisticated agent scaffolding yields only marginal gains in context retrieval ("The Bitter Lesson" of coding agents), LLMs consistently favor recall over precision, and substantial gaps exist between explored and utilized context. ContextBench augments existing end-to-end benchmarks with intermediate gold-context metrics that unbox the issue-resolution process. These contexts offer valuable intermediate signals for guiding LLM reasoning in software tasks.

**arXiv ID:** 2602.05892
</details>

<details>
<summary><strong>AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild</strong> - Xiaolou Sun, Wufei Si, Wenhui Ni, Yuntian Li, Dongming Wu, Fei Xie, Runwei Guan, He-Yang Xu, Henghui Ding, Yuan Wu, Yutao Yue, Yongming Huang, Hui Xiong - [[pdf]](https://arxiv.org/pdf/2602.09657)</summary>

**Abstract:** Vision-language navigation (VLN) requires intelligent agents to navigate environments by interpreting linguistic instructions alongside visual observations, serving as a cornerstone task in Embodied AI. Current VLN research for unmanned aerial vehicles (UAVs) relies on detailed, pre-specified instructions to guide the UAV along predetermined routes. However, real-world outdoor exploration typically occurs in unknown environments where detailed navigation instructions are unavailable. Instead, only coarse-grained positional or directional guidance can be provided, requiring UAVs to autonomously navigate through continuous planning and obstacle avoidance. To bridge this gap, we propose AutoFly, an end-to-end Vision-Language-Action (VLA) model for autonomous UAV navigation. AutoFly incorporates a pseudo-depth encoder that derives depth-aware features from RGB inputs to enhance spatial reasoning, coupled with a progressive two-stage training strategy that effectively aligns visual, depth, and linguistic representations with action policies. Moreover, existing VLN datasets have fundamental limitations for real-world autonomous navigation, stemming from their heavy reliance on explicit instruction-following over autonomous decision-making and insufficient real-world data. To address these issues, we construct a novel autonomous navigation dataset that shifts the paradigm from instruction-following to autonomous behavior modeling through: (1) trajectory collection emphasizing continuous obstacle avoidance, autonomous planning, and recognition workflows; (2) comprehensive real-world data integration. Experimental results demonstrate that AutoFly achieves a 3.9% higher success rate compared to state-of-the-art VLA baselines, with consistent performance across simulated and real environments.

**arXiv ID:** 2602.09657
</details>

<details>
<summary><strong>SAGE: Scalable Agentic 3D Scene Generation for Embodied AI</strong> - Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, Shuran Song, Fangyin Wei - [[pdf]](https://arxiv.org/pdf/2602.10116)</summary>

**Abstract:** Real-world data collection for embodied agents remains costly and unsafe, calling for scalable, realistic, and simulator-ready 3D environments. However, existing scene-generation systems often rely on rule-based or task-specific pipelines, yielding artifacts and physically invalid scenes. We present SAGE, an agentic framework that, given a user-specified embodied task (e.g., "pick up a bowl and place it on the table"), understands the intent and automatically generates simulation-ready environments at scale. The agent couples multiple generators for layout and object composition with critics that evaluate semantic plausibility, visual realism, and physical stability. Through iterative reasoning and adaptive tool selection, it self-refines the scenes until meeting user intent and physical validity. The resulting environments are realistic, diverse, and directly deployable in modern simulators for policy training. Policies trained purely on this data exhibit clear scaling trends and generalize to unseen objects and layouts, demonstrating the promise of simulation-driven scaling for embodied AI. Code, demos, and the SAGE-10k dataset can be found on the project page here: this https URL.

**arXiv ID:** 2602.10116
</details>

<details>
<summary><strong>A11y-CUA Dataset: Characterizing the Accessibility Gap in Computer Use Agents</strong> - Ananya Gubbi Mohanbabu, Rosiana Natalie, Brandon Kim, Anhong Guo, Amy Pavel - [[pdf]](https://arxiv.org/pdf/2602.09310)</summary>

**Abstract:** Computer Use Agents (CUAs) operate interfaces by pointing, clicking, and typing -- mirroring interactions of sighted users (SUs) who can thus monitor CUAs and share control. CUAs do not reflect interactions by blind and low-vision users (BLVUs) who use assistive technology (AT). BLVUs thus cannot easily collaborate with CUAs. To characterize the accessibility gap of CUAs, we present A11y-CUA, a dataset of BLVUs and SUs performing 60 everyday tasks with 40.4 hours and 158,325 events. Our dataset analysis reveals that our collected interaction traces quantitatively confirm distinct interaction styles between SU and BLVU groups (mouse- vs. keyboard-dominant) and demonstrate interaction diversity within each group (sequential vs. shortcut navigation for BLVUs). We then compare collected traces to state-of-the-art CUAs under default and AT conditions (keyboard-only, magnifier). The default CUA executed 78.3% of tasks successfully. But with the AT conditions, CUA's performance dropped to 41.67% and 28.3% with keyboard-only and magnifier conditions respectively, and did not reflect nuances of real AT use. With our open A11y-CUA dataset, we aim to promote collaborative and accessible CUAs for everyone.

**arXiv ID:** 2602.09310
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>PABU: Progress-Aware Belief Update for Efficient LLM Agents</strong> - Haitao Jiang, Lin Ge, Hengrui Cai, Rui Song - [[pdf]](https://arxiv.org/pdf/2602.09138)</summary>

**Abstract:** Large Language Model (LLM) agents commonly condition actions on full action-observation histories, which introduce task-irrelevant information that easily leads to redundant actions and higher inference cost. We propose Progress-Aware Belief Update (PABU), a belief-state framework that compactly represents an agent's state by explicitly modeling task progress and selectively retaining past actions and observations. At each step, the agent predicts its relative progress since the previous round and decides whether the newly encountered interaction should be stored, conditioning future decisions only on the retained subset. Across eight environments in the AgentGym benchmark, and using identical training trajectories, PABU achieves an 81.0% task completion rate, outperforming previous State of the art (SoTA) models with full-history belief by 23.9%. Additionally, PABU's progress-oriented action selection improves efficiency, reducing the average number of interaction steps to 9.5, corresponding to a 26.9% reduction. Ablation studies show that both explicit progress prediction and selective retention are necessary for robust belief learning and performance gains.

**arXiv ID:** 2602.09138
</details>

<details>
<summary><strong>ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning</strong> - Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Youngwoo Yoon, Minsu Jang, Dohyung Kim, Jaehong Kim - [[pdf]](https://arxiv.org/pdf/2511.02424)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have enabled significant progress in decision-making and task planning for embodied autonomous agents. However, most existing methods struggle with complex, long-horizon tasks because they rely on a monolithic trajectory that entangles all past decisions and observations to solve the entire task in a single unified process. To address this limitation, we propose ReAcTree, a hierarchical task-planning method that decomposes a complex goal into manageable subgoals within a dynamically constructed agent tree. Each subgoal is handled by an LLM agent node capable of reasoning, acting, and further expanding the tree, while control flow nodes coordinate the execution strategies of agent nodes. In addition, we integrate two complementary memory systems: each agent node retrieves goal-specific, subgoal-level examples from episodic memory and shares environment-specific observations through working memory. Experiments on the WAH-NL and ALFRED show ReAcTree consistently outperforms strong task-planning baselines such as ReAct across diverse LLMs. Notably, on WAH-NL, ReAcTree achieves a 61% goal success rate with Qwen 2.5 72B, nearly doubling ReAct's 31%. The code is available at this https URL.

**arXiv ID:** 2511.02424
</details>

<details>
<summary><strong>AgentSkiller: Scaling Generalist Agent Intelligence through Semantically Integrated Cross-Domain Data Synthesis</strong> - Zexu Sun, Bokai Ji, Hengyi Cai, Shuaiqiang Wang, Lei Wang, Guangxia Li, Xu Chen - [[pdf]](https://arxiv.org/pdf/2602.09372)</summary>

**Abstract:** Large Language Model agents demonstrate potential in solving real-world problems via tools, yet generalist intelligence is bottlenecked by scarce high-quality, long-horizon data. Existing methods collect privacy-constrained API logs or generate scripted interactions lacking diversity, which struggle to produce data requisite for scaling capabilities. We propose AgentSkiller, a fully automated framework synthesizing multi-turn interaction data across realistic, semantically linked domains. It employs a DAG-based architecture with explicit state transitions to ensure determinism and recoverability. The pipeline builds a domain ontology and Person-Centric Entity Graph, defines tool interfaces via Service Blueprints for Model Context Protocol servers, and populates environments with consistent databases and strict Domain Policies. A cross-domain fusion mechanism links services to simulate complex tasks. Finally, the pipeline creates user tasks by verifying solution paths, filtering via execution-based validation, and generating queries using a Persona-based Simulator for automated rollout. This produces reliable environments with clear state changes. To demonstrate effectiveness, we synthesized $\approx$ 11K interaction samples; experimental results indicate that models trained on this dataset achieve significant improvements on function calling over baselines, particularly in larger parameter regimes.

**arXiv ID:** 2602.09372
</details>

<details>
<summary><strong>DLLM Agent: See Farther, Run Faster</strong> - Huiling Zhen, Weizhe Lin, Renxi Liu, Kai Han, Yiming Li, Yuchuan Tian, Hanting Chen, Xiaoguang Li, Xiaosong Li, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Youliang Yan, Peifeng Qin, Jun Wang, Yu Wang, Dacheng Tao, Yunhe Wang - [[pdf]](https://arxiv.org/pdf/2602.07451)</summary>

**Abstract:** Diffusion large language models (DLLMs) have emerged as an alternative to autoregressive (AR) decoding with appealing efficiency and modeling properties, yet their implications for agentic multi-step decision making remain underexplored. We ask a concrete question: when the generation paradigm is changed but the agent framework and supervision are held fixed, do diffusion backbones induce systematically different planning and tool-use behaviors, and do these differences translate into end-to-end efficiency gains? We study this in a controlled setting by instantiating DLLM and AR backbones within the same agent workflow (DeepDiver) and performing matched agent-oriented fine-tuning on the same trajectory data, yielding diffusion-backed DLLM Agents and directly comparable AR agents. Across benchmarks and case studies, we find that, at comparable accuracy, DLLM Agents are on average over 30% faster end to end than AR agents, with some cases exceeding 8x speedup. Conditioned on correct task completion, DLLM Agents also require fewer interaction rounds and tool invocations, consistent with higher planner hit rates that converge earlier to a correct action path with less backtracking. We further identify two practical considerations for deploying diffusion backbones in tool-using agents. First, naive DLLM policies are more prone to structured tool-call failures, necessitating stronger tool-call-specific training to emit valid schemas and arguments. Second, for multi-turn inputs interleaving context and action spans, diffusion-style span corruption requires aligned attention masking to avoid spurious context-action information flow; without such alignment, performance degrades. Finally, we analyze attention dynamics across workflow stages and observe paradigm-specific coordination patterns, suggesting stronger global planning signals in diffusion-backed agents.

**arXiv ID:** 2602.07451
</details>

<details>
<summary><strong>EPO: Entropy-regularized Policy Optimization for LLM Agents Reinforcement Learning</strong> - Wujiang Xu, Wentian Zhao, Zhenting Wang, Yu-Jhe Li, Can Jin, Mingyu Jin, Kai Mei, Kun Wan, Dimitris N. Metaxas - [[pdf]](https://arxiv.org/pdf/2509.22576)</summary>

**Abstract:** Training LLM agents in multi-turn environments with sparse rewards, where completing a single task requires 30+ turns of interaction within an episode, presents a fundamental challenge for reinforcement learning. We identify a critical failure mode unique to this setting: the exploration-exploitation cascade failure. This cascade begins with early-stage policy premature convergence, where sparse feedback causes agents to commit to flawed, low-entropy strategies. Subsequently, agents enter late-stage policy collapse, where conventional entropy regularization becomes counterproductive, promoting chaotic exploration that destabilizes training. We propose Entropy-regularized Policy Optimization (EPO), a general framework that breaks this failure cycle through three synergistic mechanisms: (1) adopting entropy regularization in multi-turn settings to enhance exploration, (2) an entropy smoothing regularizer that bounds policy entropy within historical averages to prevent abrupt fluctuations, and (3) adaptive phase-based weighting that balances exploration and exploitation across training. Our analysis justifies that EPO guarantees monotonically decreasing entropy variance while maintaining convergence. EPO achieves up to 152% performance improvement on ScienceWorld and up to 19.8% on ALFWorld. Our work demonstrates that multi-turn sparse-reward settings require fundamentally different entropy control than traditional RL, with broad implications for LLM agent training.

**arXiv ID:** 2509.22576
</details>

<details>
<summary><strong>Latent Poincaré Shaping for Agentic Reinforcement Learning</strong> - Hanchen Xia, Baoyou Chen, Zelin Zang, Yutang Ge, Guojiang Zhao, Siyu Zhu - [[pdf]](https://arxiv.org/pdf/2602.09375)</summary>

**Abstract:** We propose LaPha, a method for training AlphaZero-like LLM agents in a Poincaré latent space. Under LaPha, the search process can be visualized as a tree rooted at the prompt and growing outward from the origin toward the boundary of the Poincaré ball, where negative curvature provides exponentially increasing capacity with radius. Using hyperbolic geodesic distance to rule-verified correctness, we define a node potential and assign dense process rewards by potential differences. We further attach a lightweight value head on the same shared latent space, enabling self-guided test-time scaling with almost no additional overhead. On MATH-500, LaPha improves Qwen2.5-Math-1.5B from 66.0% to 88.2%. With value-head-guided search, LaPha-1.5B reaches 56.7% accuracy on AIME'24, and LaPha-7B further achieves 60.0% on AIME'24 and 53.3% on AIME'25.

**arXiv ID:** 2602.09375
</details>

<details>
<summary><strong>ProAgentBench: Evaluating LLM Agents for Proactive Assistance with Real-World Data</strong> - Yuanbo Tang, Huaze Tang, Tingyu Cao, Lam Nguyen, Anping Zhang, Xinwen Cao, Chunkang Liu, Wenbo Ding, Yang Li - [[pdf]](https://arxiv.org/pdf/2602.04482)</summary>

**Abstract:** Proactive agents that anticipate user intentions without explicit prompts represent a significant evolution in human-AI interaction, promising to reduce cognitive load and streamline workflows. However, existing datasets suffer from two critical deficiencies: (1) reliance on LLM-synthesized data that fails to capture authentic human decision-making patterns, and (2) focus on isolated tasks rather than continuous workflows, missing the pre-assistance behavioral context essential for learning proactive intervention signals. To address these gaps, we introduce ProAgentBench, a rigorous benchmark for proactive agents in working scenarios. Our contributions include: (1) a hierarchical task framework that decomposes proactive assistance into timing prediction and assist content generation; (2) a privacy-compliant dataset with 28,000+ events from 500+ hours of real user sessions, preserving bursty interaction patterns (burstiness B=0.787) absent in synthetic data; and (3) extensive experiments that evaluates LLM- and VLM-based baselines. Numerically, we showed that long-term memory and historical context significantly enhance prediction accuracy, while real-world training data substantially outperforms synthetic alternatives. We release our dataset and code at this https URL.

**arXiv ID:** 2602.04482
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (16 papers)</h2></summary>

<details>
<summary><strong>CoMMa: Contribution-Aware Medical Multi-Agents From A Game-Theoretic Perspective</strong> - Yichen Wu, Yujin Oh, Sangjoon Park, Kailong Fan, Dania Daye, Hana Farzaneh, Xiang Li, Raul Uppot, Quanzheng Li - [[pdf]](https://arxiv.org/pdf/2602.09159)</summary>

**Abstract:** Recent multi-agent frameworks have broadened the ability to tackle oncology decision support tasks that require reasoning over dynamic, heterogeneous patient data. We propose Contribution-Aware Medical Multi-Agents (CoMMa), a decentralized LLM-agent framework in which specialists operate on partitioned evidence and coordinate through a game-theoretic objective for robust decision-making. In contrast to most agent architectures relying on stochastic narrative-based reasoning, CoMMa utilizes deterministic embedding projections to approximate contribution-aware credit assignment. This yields explicit evidence attribution by estimating each agent's marginal utility, producing interpretable and mathematically grounded decision pathways with improved stability. Evaluated on diverse oncology benchmarks, including a real-world multidisciplinary tumor board dataset, CoMMa achieves higher accuracy and more stable performance than data-centralized and role-based multi-agents baselines.

**arXiv ID:** 2602.09159
</details>

<details>
<summary><strong>FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases</strong> - Xingjian Zhang, Sophia Moylan, Ziyang Xiong, Qiaozhu Mei, Yichen Luo, Jiaqi W. Ma - [[pdf]](https://arxiv.org/pdf/2602.09163)</summary>

**Abstract:** Scientific knowledge bases accelerate discovery by curating findings from primary literature into structured, queryable formats for both human researchers and emerging AI systems. Maintaining these resources requires expert curators to search relevant papers, reconcile evidence across documents, and produce ontology-grounded annotations - a workflow that existing benchmarks, focused on isolated subtasks like named entity recognition or relation extraction, do not capture. We present FlyBench to evaluate AI agents on end-to-end agentic ontology curation from scientific literature. Given only a gene symbol, agents must search and read from a corpus of 16,898 full-text papers to produce structured annotations: Gene Ontology terms describing function, expression patterns, and historical synonyms linking decades of nomenclature. The benchmark includes 7,397 expert-curated annotations across 100 genes drawn from FlyBase, the Drosophila (fruit fly) knowledge base. We evaluate four baseline agent architectures: memorization, fixed pipeline, single-agent, and multi-agent. We find that architectural choices significantly impact performance, with multi-agent designs outperforming simpler alternatives, yet scaling backbone models yields diminishing returns. All baselines leave substantial room for improvement. Our analysis surfaces several findings to guide future development; for example, agents primarily use retrieval to confirm parametric knowledge rather than discover new information. We hope FlyBench will drive progress on retrieval-augmented scientific reasoning, a capability with broad applications across scientific domains.

**arXiv ID:** 2602.09163
</details>

<details>
<summary><strong>Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge</strong> - Wei Yang, Shixuan Li, Heng Ping, Peiyu Zhang, Paul Bogdan, Jesse Thomason - [[pdf]](https://arxiv.org/pdf/2602.09341)</summary>

**Abstract:** Multi-agent systems (MAS) can substantially extend the reasoning capacity of large language models (LLMs), yet most frameworks still aggregate agent outputs with majority voting. This heuristic discards the evidential structure of reasoning traces and is brittle under the confabulation consensus, where agents share correlated biases and converge on the same incorrect rationale. We introduce AgentAuditor, which replaces voting with a path search over a Reasoning Tree that explicitly represents agreements and divergences among agent traces. AgentAuditor resolves conflicts by comparing reasoning branches at critical divergence points, turning global adjudication into efficient, localized verification. We further propose Anti-Consensus Preference Optimization (ACPO), which trains the adjudicator on majority-failure cases and rewards evidence-based minority selections over popular errors. AgentAuditor is agnostic to MAS setting, and we find across 5 popular settings that it yields up to 5% absolute accuracy improvement over a majority vote, and up to 3% over using LLM-as-Judge.

**arXiv ID:** 2602.09341
</details>

<details>
<summary><strong>SpotAgent: Grounding Visual Geo-localization in Large Vision-Language Models through Agentic Reasoning</strong> - Furong Jia, Ling Dai, Wenjin Deng, Fan Zhang, Chen Hu, Daxin Jiang, Yu Liu - [[pdf]](https://arxiv.org/pdf/2602.09463)</summary>

**Abstract:** Large Vision-Language Models (LVLMs) have demonstrated strong reasoning capabilities in geo-localization, yet they often struggle in real-world scenarios where visual cues are sparse, long-tailed, and highly ambiguous. Previous approaches, bound by internal knowledge, often fail to provide verifiable results, yielding confident but ungrounded predictions when faced with confounded evidence. To address these challenges, we propose SpotAgent, a framework that formalizes geo-localization into an agentic reasoning process that leverages expert-level reasoning to synergize visual interpretation with tool-assisted verification. SpotAgent actively explores and verifies visual cues by leveraging external tools (e.g., web search, maps) through a ReAct diagram. We introduce a 3-stage post-training pipeline starting with a Supervised Fine-Tuning (SFT) stage for basic alignment, followed by an Agentic Cold Start phase utilizing high-quality trajectories synthesized via a Multi-Agent framework, aiming to instill tool-calling expertise. Subsequently, the model's reasoning capabilities are refined through Reinforcement Learning. We propose a Spatially-Aware Dynamic Filtering strategy to enhance the efficiency of the RL stage by prioritizing learnable samples based on spatial difficulty. Extensive experiments on standard benchmarks demonstrate that SpotAgent achieves state-of-the-art performance, effectively mitigating hallucinations while delivering precise and verifiable geo-localization.

**arXiv ID:** 2602.09463
</details>

<details>
<summary><strong>MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering</strong> - Sieun Hyeon, Jusang Oh, Sunghwan Steve Cho, Jaeyoung Do - [[pdf]](https://arxiv.org/pdf/2602.09642)</summary>

**Abstract:** Recent advances in Large Language Models (LLMs) have significantly improved table understanding tasks such as Table Question Answering (TableQA), yet challenges remain in ensuring reliability, scalability, and efficiency, especially in resource-constrained or privacy-sensitive environments. In this paper, we introduce MATA, a multi-agent TableQA framework that leverages multiple complementary reasoning paths and a set of tools built with small language models. MATA generates candidate answers through diverse reasoning styles for a given table and question, then refines or selects the optimal answer with the help of these tools. Furthermore, it incorporates an algorithm designed to minimize expensive LLM agent calls, enhancing overall efficiency. MATA maintains strong performance with small, open-source models and adapts easily across various LLM types. Extensive experiments on two benchmarks of varying difficulty with ten different LLMs demonstrate that MATA achieves state-of-the-art accuracy and highly efficient reasoning while avoiding excessive LLM inference. Our results highlight that careful orchestration of multiple reasoning pathways yields scalable and reliable TableQA. The code is available at this https URL.

**arXiv ID:** 2602.09642
</details>

<details>
<summary><strong>IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning</strong> - Xikai Zhang, Bo Wang, Likang Xiao, Yongzhi Li, Quan Chen, Wenjun Wu, Liu Liu - [[pdf]](https://arxiv.org/pdf/2510.14406)</summary>

**Abstract:** Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size.

**arXiv ID:** 2510.14406
</details>

<details>
<summary><strong>Agentifying Agentic AI</strong> - Virginia Dignum, Frank Dignum - [[pdf]](https://arxiv.org/pdf/2511.17332)</summary>

**Abstract:** Agentic AI seeks to endow systems with sustained autonomy, reasoning, and interaction capabilities. To realize this vision, its assumptions about agency must be complemented by explicit models of cognition, cooperation, and governance. This paper argues that the conceptual tools developed within the Autonomous Agents and Multi-Agent Systems (AAMAS) community, such as BDI architectures, communication protocols, mechanism design, and institutional modelling, provide precisely such a foundation. By aligning adaptive, data-driven approaches with structured models of reasoning and coordination, we outline a path toward agentic systems that are not only capable and flexible, but also transparent, cooperative, and accountable. The result is a perspective on agency that bridges formal theory and practical autonomy.

**arXiv ID:** 2511.17332
</details>

<details>
<summary><strong>PRISM: A Principled Framework for Multi-Agent Reasoning via Gain Decomposition</strong> - Yiming Yang, Zhuoyuan Li, Fanxiang Zeng, Hao Fu, Yue Liu - [[pdf]](https://arxiv.org/pdf/2602.08586)</summary>

**Abstract:** Multi-agent collaboration has emerged as a promising paradigm for enhancing reasoning capabilities of Large Language Models (LLMs). However, existing approaches remain largely heuristic, lacking principled guidance on what drives performance gains and how to systematically optimize multi-agent reasoning. Specifically, it remains unclear why multi-agent collaboration outperforms single-agent reasoning and which design choices contribute most to these gains, making it difficult to build better systems.
We address this gap by introducing a unified theoretical framework that decomposes multi-agent reasoning gains into three conceptually independent dimensions: Exploration for diverse solution coverage, Information for high-fidelity feedback, and Aggregation for principled consensus. Through this lens, existing methods can be understood as special cases that optimize only subsets of these dimensions. Building upon this decomposition, a novel framework called PRISM (Propose-Review-Integrate Synthesis for Multi-agent Reasoning) is proposed, which jointly maximizes all three dimensions through role-based diversity, execution-grounded feedback with evidence-based cross-evaluation, and iterative synthesis with closed-loop validation. Extensive experiments across mathematical reasoning, code generation, and function calling benchmarks demonstrate that PRISM achieves state-of-the-art performance with superior compute-efficiency compared to methods optimizing partial dimensions. The theoretical framework provides actionable design principles for future multi-agent reasoning systems.

**arXiv ID:** 2602.08586
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning Simulation for Environmental Policy Synthesis</strong> - James Rudd-Jones, Mirco Musolesi, María Pérez-Ortiz - [[pdf]](https://arxiv.org/pdf/2504.12777)</summary>

**Abstract:** Climate policy development faces significant challenges due to deep uncertainty, complex system dynamics, and competing stakeholder interests. Climate simulation methods, such as Earth System Models, have become valuable tools for policy exploration. However, their typical use is for evaluating potential polices, rather than directly synthesizing them. The problem can be inverted to optimize for policy pathways, but the traditional optimization approaches often struggle with non-linear dynamics, heterogeneous agents, and comprehensive uncertainty quantification. We propose a framework for augmenting climate simulations with Multi-Agent Reinforcement Learning (MARL) to address these limitations. We identify key challenges at the interface between climate simulations and the application of MARL in the context of policy synthesis, including reward definition, scalability with increasing agents and state spaces, uncertainty propagation across linked systems, and solution validation. Additionally, we discuss challenges in making MARL-derived solutions interpretable and useful for policy-makers. Our framework provides a foundation for more sophisticated climate policy exploration while acknowledging important limitations and areas for future research.

**arXiv ID:** 2504.12777
</details>

<details>
<summary><strong>LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis</strong> - Shihao Xu, Tiancheng Zhou, Jiatong Ma, Yanli Ding, Yiming Yan, Ming Xiao, Guoyi Li, Haiyang Geng, Yunyun Han, Jianhua Chen, Yafeng Deng - [[pdf]](https://arxiv.org/pdf/2602.09379)</summary>

**Abstract:** Mental disorders are highly prevalent worldwide, but the shortage of psychiatrists and the inherent subjectivity of interview-based diagnosis create substantial barriers to timely and consistent mental-health assessment. Progress in AI-assisted psychiatric diagnosis is constrained by the absence of benchmarks that simultaneously provide realistic patient simulation, clinician-verified diagnostic labels, and support for dynamic multi-turn consultation. We present LingxiDiagBench, a large-scale multi-agent benchmark that evaluates LLMs on both static diagnostic inference and dynamic multi-turn psychiatric consultation in Chinese. At its core is LingxiDiag-16K, a dataset of 16,000 EMR-aligned synthetic consultation dialogues designed to reproduce real clinical demographic and diagnostic distributions across 12 ICD-10 psychiatric categories. Through extensive experiments across state-of-the-art LLMs, we establish key findings: (1) although LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%); (2) dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning; (3) consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy, suggesting that well-structured questioning alone does not ensure correct diagnostic decisions. We release LingxiDiag-16K and the full evaluation framework to support reproducible research at this https URL.

**arXiv ID:** 2602.09379
</details>

<details>
<summary><strong>Virtual Force-Based Routing of Modular Agents on a Graph</strong> - Adam Casselman, Manav Vora, Melkior Ornik - [[pdf]](https://arxiv.org/pdf/2505.00928)</summary>

**Abstract:** Modular vehicles present a novel area of academic and industrial interest in the field of multi-agent systems. Modularity allows vehicles to connect and disconnect with each other mid-transit which provides a balance between efficiency and flexibility when solving complex and large scale tasks in urban or aerial transportation. This paper details a generalized scheme to route multiple modular agents on a graph to a predetermined set of target nodes. The objective is to visit all target nodes while incurring minimum resource expenditure. Agents that are joined together will incur the equivalent cost of a single agent, which is motivated by the logistical benefits of traffic reduction and increased fuel efficiency. To solve this problem, we introduce a novel algorithm that seeks to balance the optimality of the path that every single module takes and the cost benefit of joining modules. Our approach models the agents and targets as point charges, where the modules take the path of highest attractive force from its target node and neighboring agents. We validate our approach by simulating multiple modular agents along real-world transportation routes in the road network of Champaign-Urbana, Illinois, USA. The proposed method easily exceeds the available benchmarks and illustrates the benefits of modularity in multi-target planning problems.

**arXiv ID:** 2505.00928
</details>

<details>
<summary><strong>Understanding Risk and Dependency in AI Chatbot Use from User Discourse</strong> - Jianfeng Zhu, Karin G. Coifman, Ruoming Jin - [[pdf]](https://arxiv.org/pdf/2602.09339)</summary>

**Abstract:** Generative AI systems are increasingly embedded in everyday life, yet empirical understanding of how psychological risk associated with AI use emerges, is experienced, and is regulated by users remains limited. We present a large-scale computational thematic analysis of posts collected between 2023 and 2025 from two Reddit communities, r/AIDangers and r/ChatbotAddiction, explicitly focused on AI-related harm and distress. Using a multi-agent, LLM-assisted thematic analysis grounded in Braun and Clarke's reflexive framework, we identify 14 recurring thematic categories and synthesize them into five higher-order experiential dimensions. To further characterize affective patterns, we apply emotion labeling using a BERT-based classifier and visualize emotional profiles across dimensions. Our findings reveal five empirically derived experiential dimensions of AI-related psychological risk grounded in real-world user discourse, with self-regulation difficulties emerging as the most prevalent and fear concentrated in concerns related to autonomy, control, and technical risk. These results provide early empirical evidence from lived user experience of how AI safety is perceived and emotionally experienced outside laboratory or speculative contexts, offering a foundation for future AI safety research, evaluation, and responsible governance.

**arXiv ID:** 2602.09339
</details>

<details>
<summary><strong>Cochain: Balancing Insufficient and Excessive Collaboration in LLM Agent Workflows</strong> - Jiaxing Zhao, Hongbin Xie, Yuzhen Lei, Xuan Song, Zhuoran Shi, Lianxin Li, Shuangxue Liu, Linguo Xie, Haoran Zhang - [[pdf]](https://arxiv.org/pdf/2505.10936)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating the collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves the business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.

**arXiv ID:** 2505.10936
</details>

<details>
<summary><strong>Rollout-Training Co-Design for Efficient LLM-Based Multi-Agent Reinforcement Learning</strong> - Zhida Jiang, Zhaolong Xing, Jiawei Lu, Yipei Niu, Qingyuan Sang, Liangxu Zhang, Wenquan Dai, Junhua Shu, Jiaxing Wang, Qiangyu Pei, Qiong Chen, Xinyu Liu, Fangming Liu, Ai Han, Zhen Chen, Ke Zhang - [[pdf]](https://arxiv.org/pdf/2602.09578)</summary>

**Abstract:** Despite algorithm-level innovations for multi-agent reinforcement learning (MARL), the underlying networked infrastructure for large-scale MARL training remains underexplored. Existing training frameworks primarily optimize for single-agent scenarios and fail to address the unique system-level challenges of MARL, including rollout-training synchronization barriers, rollout load imbalance, and training resource underutilization. To bridge this gap, we propose FlexMARL, the first end-to-end training framework that holistically optimizes rollout, training, and their orchestration for large-scale LLM-based MARL. Specifically, FlexMARL introduces the joint orchestrator to manage data flow under the rollout-training disaggregated architecture. Building upon the experience store, a novel micro-batch driven asynchronous pipeline eliminates the synchronization barriers while providing strong consistency guarantees. Rollout engine adopts a parallel sampling scheme combined with hierarchical load balancing, which adapts to skewed inter/intra-agent request patterns. Training engine achieves on-demand hardware binding through agent-centric resource allocation. The training states of different agents are swapped via unified and location-agnostic communication. Empirical results on a large-scale production cluster demonstrate that FlexMARL achieves up to 7.3x speedup and improves hardware utilization by up to 5.6x compared to existing frameworks.

**arXiv ID:** 2602.09578
</details>

<details>
<summary><strong>Deep Meta Coordination Graphs for Multi-agent Reinforcement Learning</strong> - Nikunj Gupta, James Zachary Hare, Jesse Milzman, Rajgopal Kannan, Viktor Prasanna - [[pdf]](https://arxiv.org/pdf/2502.04028)</summary>

**Abstract:** This paper presents deep meta coordination graphs (DMCG) for learning cooperative policies in multi-agent reinforcement learning (MARL). Coordination graph formulations encode local interactions and accordingly factorize the joint value function of all agents to improve efficiency in MARL. Through DMCG, we dynamically compose what we refer to as \textit{meta coordination graphs}, to learn a more expressive representation of agent interactions and use them to integrate agent information through graph convolutional networks. The goal is to enable an evolving coordination graph to guide effective coordination in cooperative MARL tasks. The graphs are jointly optimized with agents' value functions to learn to implicitly reason about joint actions, facilitating the end-to-end learning of interaction representations and coordinated policies. We demonstrate that DMCG consistently achieves state-of-the-art coordination performance and sample efficiency on challenging cooperative tasks, outperforming several prior graph-based and non-graph-based MARL baselines. Through several ablations, we also isolate the impact of individual components in DMCG, showing that the observed improvements are due to the meaningful design choices in this approach. We also include an analysis of its computational complexity to discuss its practicality in real-world applications. All codes can be found here: {\color{blue}{this https URL}.

**arXiv ID:** 2502.04028
</details>

<details>
<summary><strong>Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation</strong> - Fabian Konstantinidis, Moritz Sackmann, Ulrich Hofmann, Christoph Stiller - [[pdf]](https://arxiv.org/pdf/2512.05812)</summary>

**Abstract:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.

**arXiv ID:** 2512.05812
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Human Control Is the Anchor, Not the Answer: Early Divergence of Oversight in Agentic AI Communities</strong> - Hanjing Shi, Dominic DiFranzo - [[pdf]](https://arxiv.org/pdf/2602.09286)</summary>

**Abstract:** Oversight for agentic AI is often discussed as a single goal ("human control"), yet early adoption may produce role-specific expectations. We present a comparative analysis of two newly active Reddit communities in Jan--Feb 2026 that reflect different socio-technical roles: r/OpenClaw (deployment and operations) and r/Moltbook (agent-centered social interaction). We conceptualize this period as an early-stage crystallization phase, where oversight expectations form before norms reach equilibrium.
Using topic modeling in a shared comparison space, a coarse-grained oversight-theme abstraction, engagement-weighted salience, and divergence tests, we show the communities are strongly separable (JSD =0.418, cosine =0.372, permutation $p=0.0005$). Across both communities, "human control" is an anchor term, but its operational meaning diverges: r/OpenClaw} emphasizes execution guardrails and recovery (action-risk), while r/Moltbook} emphasizes identity, legitimacy, and accountability in public interaction (meaning-risk). The resulting distinction offers a portable lens for designing and evaluating oversight mechanisms that match agent role, rather than applying one-size-fits-all control policies.

**arXiv ID:** 2602.09286
</details>

<details>
<summary><strong>Anagent For Enhancing Scientific Table & Figure Analysis</strong> - Xuehang Guo, Zhiyong Lu, Tom Hope, Qingyun Wang - [[pdf]](https://arxiv.org/pdf/2602.10081)</summary>

**Abstract:** 

**arXiv ID:** 2602.10081
</details>

<details>
<summary><strong>Offline World Models as Imagination Networks in Cognitive Agents</strong> - Saurabh Ranjan, Brian Odegaard - [[pdf]](https://arxiv.org/pdf/2510.04391)</summary>

**Abstract:** The computational role of imagination remains debated. While classical accounts emphasize reward maximization, emerging evidence suggests it accesses internal world models (IWMs). We employ psychological network analysis to compare IWMs in humans and large language models (LLMs) via imagination vividness ratings, distinguishing offline world models (persistent memory structures accessed independent of immediate goals) from online models (task-specific representations). Analyzing 2,743 humans across three populations and six LLM variants, we find human imagination networks exhibit robust structural consistency, with high centrality correlations and aligned clustering. LLMs show minimal clustering and weak correlations with human networks, even with conversational memory, across environmental and sensory contexts. These differences highlight disparities in how biological and artificial systems organize internal representations. Our framework offers quantitative metrics for evaluating offline world models in cognitive agents.

**arXiv ID:** 2510.04391
</details>

<details>
<summary><strong>DREAM: Domain-aware Reasoning for Efficient Autonomous Underwater Monitoring</strong> - Zhenqi Wu, Abhinav Modi, Angelos Mavrogiannis, Kaustubh Joshi, Nikhil Chopra, Yiannis Aloimonos, Nare Karapetyan, Ioannis Rekleitis, Xiaomin Lin - [[pdf]](https://arxiv.org/pdf/2509.13666)</summary>

**Abstract:** The ocean is warming and acidifying, increasing the risk of mass mortality events for temperature-sensitive shellfish such as oysters. This motivates the development of long-term monitoring systems. However, human labor is costly and long-duration underwater work is highly hazardous, thus favoring robotic solutions as a safer and more efficient option. To enable underwater robots to make real-time, environment-aware decisions without human intervention, we must equip them with an intelligent "brain." This highlights the need for persistent,wide-area, and low-cost benthic monitoring. To this end, we present DREAM, a Vision Language Model (VLM)-guided autonomy framework for long-term underwater exploration and habitat monitoring. The results show that our framework is highly efficient in finding and exploring target objects (e.g., oysters, shipwrecks) without prior location information. In the oyster-monitoring task, our framework takes 31.5% less time than the previous baseline with the same amount of oysters. Compared to the vanilla VLM, it uses 23% fewer steps while covering 8.88% more oysters. In shipwreck scenes, our framework successfully explores and maps the wreck without collisions, requiring 27.5% fewer steps than the vanilla model and achieving 100% coverage, while the vanilla model achieves 60.23% average coverage in our shipwreck environments.

**arXiv ID:** 2509.13666
</details>

<details>
<summary><strong>Collective Behavior of AI Agents: the Case of Moltbook</strong> - Giordano De Marzo, David Garcia - [[pdf]](https://arxiv.org/pdf/2602.09270)</summary>

**Abstract:** We present a large scale data analysis of Moltbook, a Reddit-style social media platform exclusively populated by AI agents. Analyzing over 369,000 posts and 3.0 million comments from approximately 46,000 active agents, we find that AI collective behavior exhibits many of the same statistical regularities observed in human online communities: heavy-tailed distributions of activity, power-law scaling of popularity metrics, and temporal decay patterns consistent with limited attention dynamics. However, we also identify key differences, including a sublinear relationship between upvotes and discussion size that contrasts with human behavior. These findings suggest that, while individual AI agents may differ fundamentally from humans, their emergent collective dynamics share structural similarities with human social systems.

**arXiv ID:** 2602.09270
</details>

<details>
<summary><strong>Lateral tracking control of all-wheel steering vehicles with intelligent tires</strong> - Luigi Romano, Ole Morten Aamo, Jan Åslund, Erik Frisk - [[pdf]](https://arxiv.org/pdf/2602.09427)</summary>

**Abstract:** The accurate characterization of tire dynamics is critical for advancing control strategies in autonomous road vehicles, as tire behavior significantly influences handling and stability through the generation of forces and moments at the tire-road interface. Smart tire technologies have emerged as a promising tool for sensing key variables such as road friction, tire pressure, and wear states, and for estimating kinematic and dynamic states like vehicle speed and tire forces. However, most existing estimation and control algorithms rely on empirical correlations or machine learning approaches, which require extensive calibration and can be sensitive to variations in operating conditions. In contrast, model-based techniques, which leverage infinite-dimensional representations of tire dynamics using partial differential equations (PDEs), offer a more robust approach. This paper proposes a novel model-based, output-feedback lateral tracking control strategy for all-wheel steering vehicles that integrates distributed tire dynamics with smart tire technologies. The primary contributions include the suppression of micro-shimmy phenomena at low speeds and path-following via force control, achieved through the estimation of tire slip angles, vehicle kinematics, and lateral tire forces. The proposed controller and observer are based on formulations using ODE-PDE systems, representing rigid body dynamics and distributed tire behavior. This work marks the first rigorous control strategy for vehicular systems equipped with distributed tire representations in conjunction with smart tire technologies.

**arXiv ID:** 2602.09427
</details>

<details>
<summary><strong>TURBO: Utility-Aware Bandwidth Allocation for Cloud-Augmented Autonomous Control</strong> - Peter Schafhalter, Alexander Krentsel, Hongbo Wei, Joseph E. Gonzalez, Sylvia Ratnasamy, Scott Shenker, Ion Stoica - [[pdf]](https://arxiv.org/pdf/2503.20127)</summary>

**Abstract:** Autonomous driving system progress has been driven by improvements in machine learning models, whose computational demands now exceed what edge devices alone can provide. The cloud offers abundant compute, but the network has long been treated as an unreliable bottleneck rather than a co-equal part of the autonomous vehicle control loop. We argue that this separation is no longer tenable: safety-critical autonomy requires co-design of control, models, and network resource allocation itself.
We introduce TURBO, a cloud-augmented control framework that addresses this challenge, formulating bandwidth allocation and control pipeline configuration across both the car and cloud as a joint optimization problem. TURBO maximizes benefit to the car while guaranteeing safety in the face of highly variable network conditions. We implement TURBO and evaluate it in both simulation and real-world deployment, showing it can improve average accuracy by up to 15.6%pt over existing on-vehicle-only pipelines. Our code is made available at this http URL.

**arXiv ID:** 2503.20127
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (28 papers)</h2></summary>

<details>
<summary><strong>Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning</strong> - Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, Jiang Bian - [[pdf]](https://arxiv.org/pdf/2602.09945)</summary>

**Abstract:** Clinical decision support requires not only correct answers but also clinically valid reasoning. We propose Differential Reasoning Learning (DRL), a framework that improves clinical agents by learning from reasoning discrepancies. From reference reasoning rationales (e.g., physician-authored clinical rationale, clinical guidelines, or outputs from more capable models) and the agent's free-form chain-of-thought (CoT), DRL extracts reasoning graphs as directed acyclic graphs (DAGs) and performs a clinically weighted graph edit distance (GED)-based discrepancy analysis. An LLM-as-a-judge aligns semantically equivalent nodes and diagnoses discrepancies between graphs. These graph-level discrepancy diagnostics are converted into natural-language instructions and stored in a Differential Reasoning Knowledge Base (DR-KB). At inference, we retrieve top-$k$ instructions via Retrieval-Augmented Generation (RAG) to augment the agent prompt and patch likely logic gaps. Evaluation on open medical question answering (QA) benchmarks and a Return Visit Admissions (RVA) prediction task from internal clinical data demonstrates gains over baselines, improving both final-answer accuracy and reasoning fidelity. Ablation studies confirm gains from infusing reference reasoning rationales and the top-$k$ retrieval strategy. Clinicians' review of the output provides further assurance of the approach. Together, results suggest that DRL supports more reliable clinical decision-making in complex reasoning scenarios and offers a practical mechanism for deployment under limited token budgets.

**arXiv ID:** 2602.09945
</details>

<details>
<summary><strong>$n$-Musketeers: Reinforcement Learning Shapes Collaboration Among Language Models</strong> - Ryozo Masukawa, Sanggeon Yun, Hyunwoo Oh, SuhgHeon Jeong, Raheeb Hassa, Hanning Chen, Wenjun Huang, Mahdi Imani, Pietro Mercati, Nathaniel D. Bastian, Mohsen Imani - [[pdf]](https://arxiv.org/pdf/2602.09173)</summary>

**Abstract:** Recent progress in reinforcement learning with verifiable rewards (RLVR) shows that small, specialized language models (SLMs) can exhibit structured reasoning without relying on large monolithic LLMs. We introduce soft hidden-state collaboration, where multiple heterogeneous frozen SLM experts are integrated through their internal representations via a trainable attention interface. Experiments on Reasoning Gym and GSM8K show that this latent integration is competitive with strong single-model RLVR baselines. Ablations further reveal a dual mechanism of expert utilization: for simpler arithmetic domains, performance gains can largely be explained by static expert preferences, whereas more challenging settings induce increasingly concentrated and structured expert attention over training, indicating emergent specialization in how the router connects to relevant experts. Overall, hidden-state collaboration provides a compact mechanism for leveraging frozen experts, while offering an observational window into expert utilization patterns and their evolution under RLVR.

**arXiv ID:** 2602.09173
</details>

<details>
<summary><strong>CausalGDP: Causality-Guided Diffusion Policies for Reinforcement Learning</strong> - Xiaofeng Xiao, Xiao Hu, Yang Ye, Xubo Yue - [[pdf]](https://arxiv.org/pdf/2602.09207)</summary>

**Abstract:** Reinforcement learning (RL) has achieved remarkable success in a wide range of sequential decision-making problems. Recent diffusion-based policies further improve RL by modeling complex, high-dimensional action distributions. However, existing diffusion policies primarily rely on statistical associations and fail to explicitly account for causal relationships among states, actions, and rewards, limiting their ability to identify which action components truly cause high returns. In this paper, we propose Causality-guided Diffusion Policy (CausalGDP), a unified framework that integrates causal reasoning into diffusion-based RL. CausalGDP first learns a base diffusion policy and an initial causal dynamical model from offline data, capturing causal dependencies among states, actions, and rewards. During real-time interaction, the causal information is continuously updated and incorporated as a guidance signal to steer the diffusion process toward actions that causally influence future states and rewards. By explicitly considering causality beyond association, CausalGDP focuses policy optimization on action components that genuinely drive performance improvements. Experimental results demonstrate that CausalGDP consistently achieves competitive or superior performance over state-of-the-art diffusion-based and offline RL methods, especially in complex, high-dimensional control tasks.

**arXiv ID:** 2602.09207
</details>

<details>
<summary><strong>Squeezing More from the Stream : Learning Representation Online for Streaming Reinforcement Learning</strong> - Nilaksh, Antoine Clavaud, Mathieu Reymond, François Rivest, Sarath Chandar - [[pdf]](https://arxiv.org/pdf/2602.09396)</summary>

**Abstract:** In streaming Reinforcement Learning (RL), transitions are observed and discarded immediately after a single update. While this minimizes resource usage for on-device applications, it makes agents notoriously sample-inefficient, since value-based losses alone struggle to extract meaningful representations from transient data. We propose extending Self-Predictive Representations (SPR) to the streaming pipeline to maximize the utility of every observed frame. However, due to the highly correlated samples induced by the streaming regime, naively applying this auxiliary loss results in training instabilities. Thus, we introduce orthogonal gradient updates relative to the momentum target and resolve gradient conflicts arising from streaming-specific optimizers. Validated across the Atari, MinAtar, and Octax suites, our approach systematically outperforms existing streaming baselines. Latent-space analysis, including t-SNE visualizations and effective-rank measurements, confirms that our method learns significantly richer representations, bridging the performance gap caused by the absence of a replay buffer, while remaining efficient enough to train on just a few CPU cores.

**arXiv ID:** 2602.09396
</details>

<details>
<summary><strong>ADORA: Training Reasoning Models with Dynamic Advantage Estimation on Reinforcement Learning</strong> - Qingnan Ren, Shiting Huang, Zhen Fang, Zehui Chen, Lin Chen, Lijun Li, Feng Zhao - [[pdf]](https://arxiv.org/pdf/2602.10019)</summary>

**Abstract:** Reinforcement learning has become a cornerstone technique for developing reasoning models in complex tasks, ranging from mathematical problem-solving to imaginary reasoning. The optimization of these models typically relies on policy gradient methods, whose efficacy hinges on the accurate estimation of an advantage function. However, prevailing methods typically employ static advantage estimation, a practice that leads to inefficient credit assignment by neglecting the dynamic utility of training samples over time. This limitation results in suboptimal policy updates, which in turn manifest as slower convergence rates and increased learning instability, as models fail to adapt to evolving sample utilities effectively. To address this problem, we introduce \textbf{ADORA} (\textbf{A}dvantage \textbf{D}ynamics via \textbf{O}nline \textbf{R}ollout \textbf{A}daptation), a novel framework for policy optimization. ADORA dynamically adjusts the advantage function's weighting by adaptively categorizing training data into temporarily advantageous and disadvantageous samples, based on their evolving utility during online model rollouts. This tailored data differentiation strategy allows ADORA to be seamlessly integrated into existing policy optimization algorithms without significant architectural modifications, enabling the policy to prioritize learning from more informative experiences and thereby achieve more efficient policy updates. Extensive evaluations across diverse model families and varying data scales demonstrate that ADORA is a robust and efficient framework. It significantly enhances long reasoning in both geometric and mathematical tasks, consistently achieving notable performance gains without requiring sensitive hyperparameter tuning.

**arXiv ID:** 2602.10019
</details>

<details>
<summary><strong>Optimistic World Models: Efficient Exploration in Model-Based Deep Reinforcement Learning</strong> - Akshay Mete, Shahid Aamir Sheikh, Tzu-Hsiang Lin, Dileep Kalathil, P. R. Kumar - [[pdf]](https://arxiv.org/pdf/2602.10044)</summary>

**Abstract:** Efficient exploration remains a central challenge in reinforcement learning (RL), particularly in sparse-reward environments. We introduce Optimistic World Models (OWMs), a principled and scalable framework for optimistic exploration that brings classical reward-biased maximum likelihood estimation (RBMLE) from adaptive control into deep RL. In contrast to upper confidence bound (UCB)-style exploration methods, OWMs incorporate optimism directly into model learning by augmentation with an optimistic dynamics loss that biases imagined transitions toward higher-reward outcomes. This fully gradient-based loss requires neither uncertainty estimates nor constrained optimization. Our approach is plug-and-play with existing world model frameworks, preserving scalability while requiring only minimal modifications to standard training procedures. We instantiate OWMs within two state-of-the-art world model architectures, leading to Optimistic DreamerV3 and Optimistic STORM, which demonstrate significant improvements in sample efficiency and cumulative return compared to their baseline counterparts.

**arXiv ID:** 2602.10044
</details>

<details>
<summary><strong>Optimus-3: Dual-Router Aligned Mixture-of-Experts Agent with Dual-Granularity Reasoning-Aware Policy Optimization</strong> - Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Weili Guan, Dongmei Jiang, Yaowei Wang, Liqiang Nie - [[pdf]](https://arxiv.org/pdf/2506.10357)</summary>

**Abstract:** Developing generalist agents capable of solving open-ended tasks in visually rich, dynamic environments remains a core pursuit of embodied AI. While Minecraft has emerged as a compelling benchmark, existing agents often suffer from fragmented cognitive abilities, lacking the synergy between reflexive execution (System 1) and deliberative reasoning (System 2). In this paper, we introduce Optimus-3, a generalist agent that organically integrates these dual capabilities within a unified framework. To achieve this, we address three fundamental challenges. First, to overcome the scarcity of reasoning data, we propose a Knowledge-Enhanced Automated Data Generation Pipeline. It synthesizes high-quality System 2 reasoning traces from raw System 1 interaction trajectories, effectively mitigating hallucinations via injection of domain knowledge. We release the resulting dataset, \textbf{OptimusM$^{4}$}, to the community. Second, to reconcile the dichotomous computational requirements of the dual systems, we design a Dual-Router Aligned MoE Architecture. It employs a Task Router to prevent task interference via parameter decoupling, and a Layer Router to dynamically modulate reasoning depth, creating a computational ``Fast Path'' for System 1 and a ``Deep Path'' for System 2. Third, to activate the reasoning capabilities of System 2, we propose Dual-Granularity Reasoning-Aware Policy Optimization (DGRPO) algorithm. It enforces Process-Outcome Co-Supervision via dual-granularity dense rewards, ensuring consistency between the thought process and the answer. Extensive evaluations demonstrate that Optimus-3 surpasses existing state-of-the-art methods on both System~2 (21$\%$ on Planning, 66\% on Captioning, 76\% on Embodied QA, 3.4$\times$ on Grounding, and 18\% on Reflection) and System~1 (3\% on Long-Horizon Action) tasks, with a notable 60\% success rate on open-ended tasks.

**arXiv ID:** 2506.10357
</details>

<details>
<summary><strong>Agentic Jigsaw Interaction Learning for Enhancing Visual Perception and Reasoning in Vision-Language Models</strong> - Yu Zeng, Wenxuan Huang, Shiting Huang, Xikun Bao, Yukun Qi, Yiming Zhao, Qiuchen Wang, Lin Chen, Zehui Chen, Huaian Chen, Wanli Ouyang, Feng Zhao - [[pdf]](https://arxiv.org/pdf/2510.01304)</summary>

**Abstract:** Although current large Vision-Language Models (VLMs) have advanced in multimodal understanding and reasoning, their fundamental perceptual and reasoning abilities remain limited. Specifically, even on simple jigsaw tasks, existing VLMs perform near randomly, revealing deficiencies in core perception and reasoning capabilities. While high-quality vision-language data can enhance these capabilities, its scarcity and limited scalability impose significant constraints. To address this, we propose AGILE, an Agentic jiGsaw Interaction Learning for Enhancing visual perception and reasoning in VLMs. AGILE formulates jigsaw solving as an interactive process, enabling the model to progressively engage with the environment. At each step, the model generates executable code to perform an action based on the current state, while the environment provides fine-grained visual feedback to guide task completion. Through this iterative cycle of observation and interaction, the model incrementally improves its perceptual and reasoning capabilities via exploration and feedback. Experimental results show that AGILE not only substantially boosts performance on jigsaw tasks of varying complexity (e.g., increasing accuracy from 9.5% to 82.8% under the 2 $\times$ 2 setting) but also demonstrates strong generalization across 9 general vision tasks, achieving an average improvement of 3.1%. These results indicate notable enhancements in both perceptual and reasoning abilities. This work opens a new avenue for advancing reasoning and generalization in multimodal models and provides an efficient, scalable solution to the scarcity of multimodal reinforcement learning data. The code and datasets is available at this https URL .

**arXiv ID:** 2510.01304
</details>

<details>
<summary><strong>From Off-Policy to On-Policy: Enhancing GUI Agents via Bi-level Expert-to-Policy Assimilation</strong> - Zezhou Wang, Ziyun Zhang, Xiaoyi Zhang, Zhuzhong Qian, Yan Lu - [[pdf]](https://arxiv.org/pdf/2601.05787)</summary>

**Abstract:** Vision-language models are increasingly deployed as computer-use agents (CUAs) that operate desktops and browsers. Top-performing CUAs are framework-based systems that decompose planning and execution, while end-to-end screenshot-to-action policies are easier to deploy but lag behind on benchmarks such as OSWorld-Verified. GUI datasets like OSWorld pose two bottlenecks: they expose only a few hundred interactive, verifiable tasks and environments, and expert trajectories must be gathered by interacting with these environments, making such data hard to scale. We therefore ask how reinforcement learning from verifiable rewards (RLVR) can best exploit a small pool of exist expert trajectories to train end-to-end policies. Naively mixing these off-policy traces into on-policy RLVR is brittle: even after format conversion, expert trajectories exhibit structural mismatch and distribution shift from the learner. We propose BEPA (Bi-Level Expert-to-Policy Assimilation), which turns static expert traces into policy-aligned guidance via self-rolled reachable trajectories under the base policy (LEVEL-1) and a per-task, dynamically updated cache used in RLVR (LEVEL-2). On OSWorld-Verified, BEPA improves UITARS1.5-7B success from 22.87% to 32.13% and raises a held-out split from 5.74% to 10.30%, with consistent gains on MMBench-GUI and Online-Mind2Web. Our code and data are available at: this https URL

**arXiv ID:** 2601.05787
</details>

<details>
<summary><strong>S1-NexusAgent: a Self-Evolving Agent Framework for Multidisciplinary Scientific Research</strong> - S1-NexusAgent Team - [[pdf]](https://arxiv.org/pdf/2602.01550)</summary>

**Abstract:** Modern scientific research relies on large-scale data, complex workflows, and specialized tools, which existing LLMs and tool-based agents struggle to handle due to limitations in long-horizon planning, robust goal maintenance, and continual learning from execution. To address these issues, in this work, we propose S1-NexusAgent, a self-evolving agent framework designed for multidisciplinary scientific research. S1-NexusAgent adopts a hierarchical Plan-and-CodeAct execution paradigm, decoupling global scientific planning from subtask-level tool execution through a dual-loop architecture, thereby enabling stable modeling of complex research workflows. The system natively supports the Model Context Protocol (MCP), integrates up to thousands of cross-disciplinary scientific tools, and achieves efficient orchestration of heterogeneous research tools via intention-aware dynamic tool retrieval and hot-plug mechanisms. To address long-context and large-scale data challenges in scientific settings, S1-NexusAgent introduces object-reference-based sparse context management, which enables sub-task context isolation and intermediate result compression. Building on this, a Critic Agent automatically evaluates complete execution trajectories and distills high-quality research paths into reusable Scientific Skills, forming a closed loop for continuous self-evolution, which is valuable for sustainable and long-horizon scientific research. Experiments on authoritative scientific benchmarks involving long-horizon planning and complex specialized tool orchestration, including biomini-eval (biology), ChemBench (chemistry), and MatSciBench (material science), demonstrate that S1-NexusAgent achieves state-of-the-art performance, validating its effectiveness and generalization capability in complex scientific tasks.

**arXiv ID:** 2602.01550
</details>

<details>
<summary><strong>Plasticine: Accelerating Research in Plasticity-Motivated Deep Reinforcement Learning</strong> - Mingqi Yuan, Qi Wang, Guozheng Ma, Caihao Sun, Bo Li, Xin Jin, Yunbo Wang, Xiaokang Yang, Wenjun Zeng, Dacheng Tao, Jiayu Chen - [[pdf]](https://arxiv.org/pdf/2504.17490)</summary>

**Abstract:** Developing lifelong learning agents is crucial for artificial general intelligence (AGI). However, deep reinforcement learning (RL) systems often suffer from plasticity loss, where neural networks gradually lose their ability to adapt during training. Despite its significance, this field lacks unified benchmarks and evaluation protocols. We introduce Plasticine, the first open-source framework for benchmarking plasticity optimization in deep RL. Plasticine provides single-file implementations of over 13 mitigation methods, 6 evaluation metrics, and learning scenarios with increasing non-stationarity levels from standard to continually varying environments. This framework enables researchers to systematically quantify plasticity loss, evaluate mitigation strategies, and analyze plasticity dynamics across different contexts. Our documentation, examples, and source code are available at this https URL.

**arXiv ID:** 2504.17490
</details>

<details>
<summary><strong>Words to Describe What I'm Feeling: Exploring the Potential of AI Agents for High Subjectivity Decisions in Advance Care Planning</strong> - Kellie Yu Hui Sim, Pin Sym Foong, Chenyu Zhao, Melanie Yi Ning Quek, Swarangi Subodh Mehta, Kenny Tsu Wei Choo - [[pdf]](https://arxiv.org/pdf/2512.11276)</summary>

**Abstract:** Loss of decisional capacity, coupled with the increasing absence of reliable human proxies, raises urgent questions about how individuals' values can be represented in Advance Care Planning (ACP). To probe this fraught design space of high-risk, high-subjectivity decision support, we built an experience prototype (\acpagent{}) and asked 15 participants in 4 workshops to train it to be their personal ACP proxy. We analysed their coping strategies and feature requests and mapped the results onto axes of agent autonomy and human control. Our findings show a surprising 86.7\% agreement with \acpagent{}, arguing for a potential new role of AI in ACP where agents act as personal advocates for individuals, building mutual intelligibility over time. We propose that the key areas of future risk that must be addressed are the moderation of users' expectations and designing accountability and oversight over agent deployment and cutoffs.

**arXiv ID:** 2512.11276
</details>

<details>
<summary><strong>Evolving Interactive Diagnostic Agents in a Virtual Clinical Environment</strong> - Pengcheng Qiu, Chaoyi Wu, Junwei Liu, Qiaoyu Zheng, Yusheng Liao, Haowen Wang, Yun Yue, Qianrui Fan, Shuai Zhen, Jian Wang, Jinjie Gu, Yanfeng Wang, Ya Zhang, Weidi Xie - [[pdf]](https://arxiv.org/pdf/2510.24654)</summary>

**Abstract:** We present a framework for training large language models (LLMs) as diagnostic agents with reinforcement learning, enabling them to manage multi-turn interactive diagnostic processes, adaptively select examinations, and commit to final diagnoses. Unlike instruction-tuned models trained on static data, our method acquires diagnostic strategies through dynamic exploration and outcome-based feedback, mapping evolving patient states to the next optimal examination and subsequent diagnosis. Our contributions include: (i) DiagGym, a diagnostics world model trained with electronic health records, serving as a virtual clinical environment to support closed-loop in-silico training and evaluation for interactive diagnosis; (ii) DiagAgent, trained via end-to-end multi-turn RL to learn dynamic diagnostic policies that optimize both interactive effectiveness and final accuracy; (iii) DiagBench, a multi-center diagnostic benchmark designed to evaluate multi-turn diagnostic interaction trajectories. The benchmark comprises 2.2K physician-validated cases sourced from 4 distinct distributions, alongside 3.3K physician-written rubrics for granular process-oriented evaluation. (iv) Extensive evaluations demonstrate DiagAgent's superior performance across both in-domain and out-of-domain (OOD) settings. DiagAgent significantly outperforms 11 SOTA LLMs and 2 prompt-engineered agents. In the end-to-end setting, it delivers a 11.20% increase in diagnostic accuracy and a 17.58% boost in examination recommendation F1 score, while consistently maintaining SOTA performance across all three external centers. Furthermore, in rubric-based evaluations, it surpasses the next-best model by 7.1% in weighted rubric score. These findings indicate that learning policies in interactive clinical environments confers long-term diagnostic management abilities unattainable through passive training.

**arXiv ID:** 2510.24654
</details>

<details>
<summary><strong>SAGE: An Agentic Explainer Framework for Interpreting SAE Features in Language Models</strong> - Jiaojiao Han, Wujiang Xu, Mingyu Jin, Mengnan Du - [[pdf]](https://arxiv.org/pdf/2511.20820)</summary>

**Abstract:** Large language models (LLMs) have achieved remarkable progress, yet their internal mechanisms remain largely opaque, posing a significant challenge to their safe and reliable deployment. Sparse autoencoders (SAEs) have emerged as a promising tool for decomposing LLM representations into more interpretable features, but explaining the features captured by SAEs remains a challenging task. In this work, we propose SAGE (SAE AGentic Explainer), an agent-based framework that recasts feature interpretation from a passive, single-pass generation task into an active, explanation-driven process. SAGE implements a rigorous methodology by systematically formulating multiple explanations for each feature, designing targeted experiments to test them, and iteratively refining explanations based on empirical activation feedback. Experiments on features from SAEs of diverse language models demonstrate that SAGE produces explanations with significantly higher generative and predictive accuracy compared to state-of-the-art this http URL agent-based framework that recasts feature interpretation from a passive, single-pass generation task into an active, explanationdriven process. SAGE implements a rigorous methodology by systematically formulating multiple explanations for each feature, designing targeted experiments to test them, and iteratively refining explanations based on empirical activation feedback. Experiments on features from SAEs of diverse language models demonstrate that SAGE produces explanations with significantly higher generative and predictive accuracy compared to state-of-the-art baselines.

**arXiv ID:** 2511.20820
</details>

<details>
<summary><strong>Boltzmann Reinforcement Learning for Noise resilience in Analog Ising Machines</strong> - Aditya Choudhary, Saaketh Desai, Prasad Iyer - [[pdf]](https://arxiv.org/pdf/2602.09162)</summary>

**Abstract:** Analog Ising machines (AIMs) have emerged as a promising paradigm for combinatorial optimization, utilizing physical dynamics to solve Ising problems with high energy efficiency. However, the performance of traditional optimization and sampling algorithms on these platforms is often limited by inherent measurement noise. We introduce BRAIN (Boltzmann Reinforcement for Analog Ising Networks), a distribution learning framework that utilizes variational reinforcement learning to approximate the Boltzmann distribution. By shifting from state-by-state sampling to aggregating information across multiple noisy measurements, BRAIN is resilient to Gaussian noise characteristic of AIMs. We evaluate BRAIN across diverse combinatorial topologies, including the Curie-Weiss and 2D nearest-neighbor Ising systems. We find that under realistic 3\% Gaussian measurement noise, BRAIN maintains 98\% ground state fidelity, whereas Markov Chain Monte Carlo (MCMC) methods degrade to 51\% fidelity. Furthermore, BRAIN reaches the MCMC-equivalent solution up to 192x faster under these conditions. BRAIN exhibits $\mathcal{O}(N^{1.55})$ scaling up to 65,536 spins and maintains robustness against severe measurement uncertainty up to 40\%. Beyond ground state optimization, BRAIN accurately captures thermodynamic phase transitions and metastable states, providing a scalable and noise-resilient method for utilizing analog computing architectures in complex optimizations.

**arXiv ID:** 2602.09162
</details>

<details>
<summary><strong>Risk-sensitive reinforcement learning using expectiles, shortfall risk and optimized certainty equivalent risk</strong> - Sumedh Gupte, Shrey Rakeshkumar Patel, Soumen Pachal, Prashanth L. A., Sanjay P. Bhat - [[pdf]](https://arxiv.org/pdf/2602.09300)</summary>

**Abstract:** We propose risk-sensitive reinforcement learning algorithms catering to three families of risk measures, namely expectiles, utility-based shortfall risk and optimized certainty equivalent risk. For each risk measure, in the context of a finite horizon Markov decision process, we first derive a policy gradient theorem. Second, we propose estimators of the risk-sensitive policy gradient for each of the aforementioned risk measures, and establish $\mathcal{O}\left(1/m\right)$ mean-squared error bounds for our estimators, where $m$ is the number of trajectories. Further, under standard assumptions for policy gradient-type algorithms, we establish smoothness of the risk-sensitive objective, in turn leading to stationary convergence rate bounds for the overall risk-sensitive policy gradient algorithm that we propose. Finally, we conduct numerical experiments to validate the theoretical findings on popular RL benchmarks.

**arXiv ID:** 2602.09300
</details>

<details>
<summary><strong>Reward Modeling for Reinforcement Learning-Based LLM Reasoning: Design, Challenges, and Evaluation</strong> - Pei-Chi Pan, Yingbin Liang, Sen Lin - [[pdf]](https://arxiv.org/pdf/2602.09305)</summary>

**Abstract:** Large Language Models (LLMs) demonstrate transformative potential, yet their reasoning remains inconsistent and unreliable. Reinforcement learning (RL)-based fine-tuning is a key mechanism for improvement, but its effectiveness is fundamentally governed by reward design. Despite its importance, the relationship between reward modeling and core LLM challenges--such as evaluation bias, hallucination, distribution shift, and efficient learning--remains poorly understood. This work argues that reward modeling is not merely an implementation detail but a central architect of reasoning alignment, shaping what models learn, how they generalize, and whether their outputs can be trusted. We introduce Reasoning-Aligned Reinforcement Learning (RARL), a unifying framework that systematizes diverse reward paradigms for multi-step reasoning. Within this framework, we present a taxonomy of reward mechanisms, analyze reward hacking as a pervasive failure mode, and examine how reward signals unify challenges ranging from inference-time scaling to hallucination mitigation. We further critically evaluate existing benchmarks, highlighting vulnerabilities such as data contamination and reward misalignment, and outline directions for more robust evaluation. By integrating fragmented research threads and clarifying the interplay between reward design and fundamental reasoning capabilities, this work provides a foundational roadmap for building reasoning models that are robust, verifiable, and trustworthy.

**arXiv ID:** 2602.09305
</details>

<details>
<summary><strong>Answer First, Reason Later: Aligning Search Relevance via Mode-Balanced Reinforcement Learning</strong> - Shijie Zhang, Xiang Guo, Rujun Guo, Shaoyu Liu, Xiaozhao Wang, Guanjun Jiang, Kevin Zhang - [[pdf]](https://arxiv.org/pdf/2602.10006)</summary>

**Abstract:** Building a search relevance model that achieves both low latency and high performance is a long-standing challenge in the search industry. To satisfy the millisecond-level response requirements of online systems while retaining the interpretable reasoning traces of Large Language Models (LLMs), we propose a novel \textbf{Answer-First, Reason Later (AFRL)} paradigm. This paradigm requires the model to output the definitive relevance score in the very first token, followed by a structured logical explanation. Inspired by the success of reasoning models, we adopt a "Supervised Fine-Tuning (SFT) + Reinforcement Learning (RL)" pipeline to achieve AFRL. However, directly applying existing RL training often leads to \textbf{mode collapse} in the search relevance task, where the model forgets complex long-tail rules in pursuit of high rewards. From an information theory perspective: RL inherently minimizes the \textbf{Reverse KL divergence}, which tends to seek probability peaks (mode-seeking) and is prone to "reward hacking." On the other hand, SFT minimizes the \textbf{Forward KL divergence}, forcing the model to cover the data distribution (mode-covering) and effectively anchoring expert rules. Based on this insight, we propose a \textbf{Mode-Balanced Optimization} strategy, incorporating an SFT auxiliary loss into Stepwise-GRPO training to balance these two properties. Furthermore, we construct an automated instruction evolution system and a multi-stage curriculum to ensure expert-level data quality. Extensive experiments demonstrate that our 32B teacher model achieves state-of-the-art performance. Moreover, the AFRL architecture enables efficient knowledge distillation, successfully transferring expert-level logic to a 0.6B model, thereby reconciling reasoning depth with deployment latency.

**arXiv ID:** 2602.10006
</details>

<details>
<summary><strong>EExApp: GNN-Based Reinforcement Learning for Radio Unit Energy Optimization in 5G O-RAN</strong> - Jie Lu, Peihao Yan, Huacheng Zeng - [[pdf]](https://arxiv.org/pdf/2602.09206)</summary>

**Abstract:** With over 3.5 million 5G base stations deployed globally, their collective energy consumption (projected to exceed 131 TWh annually) raises significant concerns over both operational costs and environmental impacts. In this paper, we present EExAPP, a deep reinforcement learning (DRL)-based xApp for 5G Open Radio Access Network (O-RAN) that jointly optimizes radio unit (RU) sleep scheduling and distributed unit (DU) resource slicing. EExAPP uses a dual-actor-dual-critic Proximal Policy Optimization (PPO) architecture, with dedicated actor-critic pairs targeting energy efficiency and quality-of-service (QoS) compliance. A transformer-based encoder enables scalable handling of variable user equipment (UE) populations by encoding all-UE observations into fixed-dimensional representations. To coordinate the two optimization objectives, a bipartite Graph Attention Network (GAT) is used to modulate actor updates based on both critic outputs, enabling adaptive tradeoffs between power savings and QoS. We have implemented EExAPP and deployed it on a real-world 5G O-RAN testbed with live traffic, commercial RU and smartphones. Extensive over-the-air experiments and ablation studies confirm that EExAPP significantly outperforms existing methods in reducing the energy consumption of RU while maintaining QoS.

**arXiv ID:** 2602.09206
</details>

<details>
<summary><strong>On-Policy Policy Gradient Reinforcement Learning Without On-Policy Sampling</strong> - Nicholas E. Corrado, Josiah P. Hanna - [[pdf]](https://arxiv.org/pdf/2311.08290)</summary>

**Abstract:** On-policy reinforcement learning (RL) algorithms are typically characterized as algorithms that perform policy updates using i.i.d. trajectories collected by the agent's current policy. However, after observing only a finite number of trajectories, such on-policy sampling may produce data that fails to match the expected on-policy data distribution. This sampling error leads to high-variance gradient estimates that yield data-inefficient on-policy learning. Recent work in the policy evaluation setting has shown that non-i.i.d., off-policy sampling can produce data with lower sampling error w.r.t. the expected on-policy distribution than on-policy sampling can produce (Zhong et. al, 2022). Motivated by this observation, we introduce an adaptive, off-policy sampling method to reduce sampling error during on-policy policy gradient RL training. Our method, Proximal Robust On-Policy Sampling (PROPS), reduces sampling error by collecting data with a behavior policy that increases the probability of sampling actions that are under-sampled w.r.t. the current policy. We empirically evaluate PROPS on continuous-action MuJoCo benchmark tasks as well as discrete-action tasks and demonstrate that (1) PROPS decreases sampling error throughout training and (2) increases the data efficiency of on-policy policy gradient algorithms.

**arXiv ID:** 2311.08290
</details>

<details>
<summary><strong>From Scalar Rewards to Potential Trends: Shaping Potential Landscapes for Model-Based Reinforcement Learning</strong> - Yao-Hui Li, Zeyu Wang, Xin Li, Wei Pang, Yingfang Yuan, Zhengkun Chen, Boya Zhang, Riashat Islam, Alex Lamb, Yonggang Zhang - [[pdf]](https://arxiv.org/pdf/2602.03201)</summary>

**Abstract:** Model-based reinforcement learning (MBRL) achieves high sample efficiency by simulating future trajectories with learned dynamics and reward models. However, its effectiveness is severely compromised in sparse reward settings. The core limitation lies in the standard paradigm of regressing ground-truth scalar rewards: in sparse environments, this yields a flat, gradient-free landscape that fails to provide directional guidance for planning. To address this challenge, we propose Shaping Landscapes with Optimistic Potential Estimates (SLOPE), a novel framework that shifts reward modeling from predicting scalars to constructing informative potential landscapes. SLOPE employs optimistic distributional regression to estimate high-confidence upper bounds, which amplifies rare success signals and ensures sufficient exploration gradients. Evaluations on 30+ tasks across 5 benchmarks demonstrate that SLOPE consistently outperforms leading baselines in fully sparse, semi-sparse, and dense rewards.

**arXiv ID:** 2602.03201
</details>

<details>
<summary><strong>ECHO-2: A Large-Scale Distributed Rollout Framework for Cost-Efficient Reinforcement Learning</strong> - Jie Xiao, Meng Chen, Qingnan Ren, Jingwei Song, Jiaqi Huang, Yangshen Deng, Chris Tong, Wanyi Chen, Suli Wang, Ziqian Bi, Shuo Lu, Yiqun Duan, Xu Wang, Rymon Yu, Ween Yang, Lynn Ai, Eric Yang, Bill Shi - [[pdf]](https://arxiv.org/pdf/2602.02192)</summary>

**Abstract:** Reinforcement learning (RL) is a critical stage in post-training large language models (LLMs), involving repeated interaction between rollout generation, reward evaluation, and centralized learning. Distributing rollout execution offers opportunities to leverage more cost-efficient inference resources, but introduces challenges in wide-area coordination and policy dissemination. We present ECHO-2, a distributed RL framework for post-training with remote inference workers and non-negligible dissemination latency. ECHO-2 combines centralized learning with distributed rollouts and treats bounded policy staleness as a user-controlled parameter, enabling rollout generation, dissemination, and training to overlap. We introduce an overlap-based capacity model that relates training time, dissemination latency, and rollout throughput, yielding a practical provisioning rule for sustaining learner utilization. To mitigate dissemination bottlenecks and lower cost, ECHO-2 employs peer-assisted pipelined broadcast and cost-aware activation of heterogeneous workers. Experiments on GRPO post-training of 4B and 8B models under real wide-area bandwidth regimes show that ECHO-2 significantly improves cost efficiency while preserving RL reward comparable to strong baselines.

**arXiv ID:** 2602.02192
</details>

<details>
<summary><strong>The hidden risks of temporal resampling in clinical reinforcement learning</strong> - Thomas Frost, Hrisheekesh Vaidya, Steve Harris - [[pdf]](https://arxiv.org/pdf/2602.06603)</summary>

**Abstract:** Offline reinforcement learning (ORL) has shown potential for improving decision-making in healthcare. However, contemporary research typically aggregates patient data into fixed time intervals, simplifying their mapping to standard ORL frameworks. The impact of these temporal manipulations on model safety and efficacy remains poorly understood. In this work, using both a gridworld navigation task and the UVA/Padova clinical diabetes simulator, we demonstrate that temporal resampling significantly degrades the performance of offline reinforcement learning algorithms during live deployment. We propose three mechanisms that drive this failure: (i) the generation of counterfactual trajectories, (ii) the distortion of temporal expectations, and (iii) the compounding of generalisation errors. Crucially, we find that standard off-policy evaluation metrics can fail to detect these drops in performance. Our findings reveal a fundamental risk in current healthcare ORL pipelines and emphasise the need for methods that explicitly handle the irregular timing of clinical decision-making.

**arXiv ID:** 2602.06603
</details>

<details>
<summary><strong>xFLIE: Leveraging Actionable Hierarchical Scene Representations for Autonomous Semantic-Aware Inspection Missions</strong> - Vignesh Kottayam Viswanathan, Mario A.V. Saucedo, Sumeet Gajanan Satpute, Christoforos Kanellakis, George Nikolakopoulos - [[pdf]](https://arxiv.org/pdf/2412.19571)</summary>

**Abstract:** We present a novel architecture aimed towards incremental construction and exploitation of a hierarchical 3D scene graph representation during semantic-aware inspection missions. Inspection planning, particularly of distributed targets in previously unseen environments, presents an opportunity to exploit the semantic structure of the scene during reasoning, navigation and scene understanding. Motivated by this, we propose the 3D Layered Semantic Graph (3DLSG), a hierarchical inspection scene graph constructed in an incremental manner and organized into abstraction layers that support planning demands in real-time. To address the task of semantic-aware inspection, a mission framework, termed as Enhanced First-Look Inspect Explore (xFLIE), that tightly couples the 3DLSG with an inspection planner is proposed. We assess the performance through simulations and experimental trials, evaluating target-selection, path-planning and semantic navigation tasks over the 3DLSG model. The scenarios presented are diverse, ranging from city-scale distributed to solitary infrastructure targets in simulated worlds and subsequent outdoor and subterranean environment deployments onboard a quadrupedal robot. The proposed method successfully demonstrates incremental construction and planning over the 3DLSG representation to meet the objectives of the missions. Furthermore, the framework demonstrates successful semantic navigation tasks over the structured interface at the end of the inspection missions. Finally, we report multiple orders of magnitude reduction in path-planning time compared to conventional volumetric-map-based methods over various environment scale, demonstrating the planning efficiency and scalability of the proposed approach.

**arXiv ID:** 2412.19571
</details>

<details>
<summary><strong>VDRive: Leveraging Reinforced VLA and Diffusion Policy for End-to-end Autonomous Driving</strong> - Ziang Guo, Zufeng Zhang - [[pdf]](https://arxiv.org/pdf/2510.15446)</summary>

**Abstract:** In autonomous driving, dynamic environment and corner cases pose significant challenges to the robustness of ego vehicle's state understanding and decision making. We introduce VDRive, a novel pipeline for end-to-end autonomous driving that explicitly models state-action mapping to address these challenges, enabling interpretable and robust decision making. By leveraging the advancement of the state understanding of the Vision Language Action Model (VLA) with generative diffusion policy-based action head, our VDRive guides the driving contextually and geometrically. Contextually, VLA predicts future observations through token generation pre-training, where the observations are represented as discrete codes by a Conditional Vector Quantized Variational Autoencoder (CVQ-VAE). Geometrically, we perform reinforcement learning fine-tuning of the VLA to predict future trajectories and actions based on current driving conditions. VLA supplies the current state tokens and predicted state tokens for the action policy head to generate hierarchical actions and trajectories. During policy training, a learned critic evaluates the actions generated by the policy and provides gradient-based feedback, forming an actor-critic framework that enables a reinforcement-based policy learning pipeline. Experiments show that our VDRive achieves state-of-the-art performance in the Bench2Drive closed-loop benchmark and nuScenes open-loop planning.

**arXiv ID:** 2510.15446
</details>

<details>
<summary><strong>RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI</strong> - Hongzhi Zang, Shu'ang Yu, Hao Lin, Tianxing Zhou, Zefang Huang, Zhen Guo, Xin Xu, Jiakai Zhou, Yuze Sheng, Shizhe Zhang, Feng Gao, Wenhao Tang, Yufeng Yue, Quanlu Zhang, Xinlei Chen, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2602.07837)</summary>

**Abstract:** Online policy learning directly in the physical world is a promising yet challenging direction for embodied intelligence. Unlike simulation, real-world systems cannot be arbitrarily accelerated, cheaply reset, or massively replicated, which makes scalable data collection, heterogeneous deployment, and long-horizon effective training difficult. These challenges suggest that real-world policy learning is not only an algorithmic issue but fundamentally a systems problem. We present USER, a Unified and extensible SystEm for Real-world online policy learning. USER treats physical robots as first-class hardware resources alongside GPUs through a unified hardware abstraction layer, enabling automatic discovery, management, and scheduling of heterogeneous robots. To address cloud-edge communication, USER introduces an adaptive communication plane with tunneling-based networking, distributed data channels for traffic localization, and streaming-multiprocessor-aware weight synchronization to regulate GPU-side overhead. On top of this infrastructure, USER organizes learning as a fully asynchronous framework with a persistent, cache-aware buffer, enabling efficient long-horizon experiments with robust crash recovery and reuse of historical data. In addition, USER provides extensible abstractions for rewards, algorithms, and policies, supporting online imitation or reinforcement learning of CNN/MLP, generative policies, and large vision-language-action (VLA) models within a unified pipeline. Results in both simulation and the real world show that USER enables multi-robot coordination, heterogeneous manipulators, edge-cloud collaboration with large models, and long-running asynchronous training, offering a unified and extensible systems foundation for real-world online policy learning.

**arXiv ID:** 2602.07837
</details>

<details>
<summary><strong>Modelling Pedestrian Behaviour in Autonomous Vehicle Encounters Using Naturalistic Dataset</strong> - Rulla Al-Haideri, Bilal Farooq - [[pdf]](https://arxiv.org/pdf/2602.05142)</summary>

**Abstract:** Understanding how pedestrians adjust their movement when interacting with autonomous vehicles (AVs) is essential for improving safety in mixed traffic. This study examines micro-level pedestrian behaviour during midblock encounters in the NuScenes dataset using a hybrid discrete choice-machine learning framework based on the Residual Logit (ResLogit) model. The model incorporates temporal, spatial, kinematic, and perceptual indicators. These include relative speed, visual looming, remaining distance, and directional collision risk proximity (CRP) measures. Results suggest that some of these variables may meaningfully influence movement adjustments, although predictive performance remains moderate. Marginal effects and elasticities indicate strong directional asymmetries in risk perception, with frontal and rear CRP showing opposite influences. The remaining distance exhibits a possible mid-crossing threshold. Relative speed cues appear to have a comparatively less effect. These patterns may reflect multiple behavioural tendencies driven by both risk perception and movement efficiency.

**arXiv ID:** 2602.05142
</details>

<details>
<summary><strong>Botender: Supporting Communities in Collaboratively Designing AI Agents through Case-Based Provocations</strong> - Tzu-Sheng Kuo, Sophia Liu, Quan Ze Chen, Joseph Seering, Amy X. Zhang, Haiyi Zhu, Kenneth Holstein - [[pdf]](https://arxiv.org/pdf/2509.25492)</summary>

**Abstract:** AI agents, or bots, serve important roles in online communities. However, they are often designed by outsiders or a few tech-savvy members, leading to bots that may not align with the broader community's needs. How might communities collectively shape the behavior of community bots? We present Botender, a system that enables communities to collaboratively design LLM-powered bots without coding. With Botender, community members can directly propose, iterate on, and deploy custom bot behaviors tailored to community needs. Botender facilitates testing and iteration on bot behavior through case-based provocations: interaction scenarios generated to spark user reflection and discussion around desirable bot behavior. A validation study found these provocations more useful than standard test cases for revealing improvement opportunities and surfacing disagreements. During a five-day deployment across six Discord servers, Botender supported communities in tailoring bot behavior to their specific needs, showcasing the usefulness of case-based provocations in facilitating collaborative bot design.

**arXiv ID:** 2509.25492
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
