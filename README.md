# Agent arXiv Daily

**Last Updated:** 2025-10-06 02:40:02

**Total Papers:** 64

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (2 papers)</h2></summary>

<details>
<summary><strong>EconWebArena: Benchmarking Autonomous Agents on Economic Tasks in Realistic Web Environments</strong> - Zefang Liu, Yinzhu Quan - [[pdf]](https://arxiv.org/pdf/2506.08136)</summary>

**Abstract:** We introduce EconWebArena, a benchmark for evaluating autonomous agents on complex, multimodal economic tasks in realistic web environments. The benchmark comprises 360 curated tasks from 82 authoritative websites spanning domains such as macroeconomics, labor, finance, trade, and public policy. Each task challenges agents to navigate live websites, interpret structured and visual content, interact with real interfaces, and extract precise, time-sensitive data through multi-step workflows. We construct the benchmark by prompting multiple large language models (LLMs) to generate candidate tasks, followed by rigorous human curation to ensure clarity, feasibility, and source reliability. Unlike prior work, EconWebArena emphasizes fidelity to authoritative data sources and the need for grounded web-based economic reasoning. We evaluate a diverse set of state-of-the-art multimodal LLMs as web agents, analyze failure cases, and conduct ablation studies to assess the impact of visual grounding, plan-based reasoning, and interaction design. Our results reveal substantial performance gaps and highlight persistent challenges in grounding, navigation, and multimodal understanding, positioning EconWebArena as a rigorous testbed for economic web intelligence.

**arXiv ID:** 2506.08136
</details>

<details>
<summary><strong>Less is More: Lean yet Powerful Vision-Language Model for Autonomous Driving</strong> - Sheng Yang, Tong Zhan, Guancheng Chen, Yanfeng Lu, Jian Wang - [[pdf]](https://arxiv.org/pdf/2510.00060)</summary>

**Abstract:** In this work, we reconceptualize autonomous driving as a generalized language and formulate the trajectory planning task as next waypoint prediction. We introduce Max-V1, a novel framework for one-stage end-to-end autonomous driving. Our framework presents a single-pass generation paradigm that aligns with the inherent sequentiality of driving. This approach leverages the generative capacity of the VLM (Vision-Language Model) to enable end-to-end trajectory prediction directly from front-view camera input. The efficacy of this method is underpinned by a principled supervision strategy derived from statistical modeling. This provides a well-defined learning objective, which makes the framework highly amenable to master complex driving policies through imitation learning from large-scale expert demonstrations. Empirically, our method achieves the state-of-the-art performance on the nuScenes dataset, delivers an overall improvement of over 30% compared to prior baselines. Furthermore, it exhibits superior generalization performance on cross-domain datasets acquired from diverse vehicles, demonstrating notable potential for cross-vehicle robustness and adaptability. Due to these empirical strengths, this work introduces a model enabling fundamental driving behaviors, laying the foundation for the development of more capable self-driving agents. Code will be available upon publication.

**arXiv ID:** 2510.00060
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>ARMs: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks</strong> - Zhaorun Chen, Xun Liu, Mintong Kang, Jiawei Zhang, Minzhou Pan, Shuang Yang, Bo Li - [[pdf]](https://arxiv.org/pdf/2510.02677)</summary>

**Abstract:** As vision-language models (VLMs) gain prominence, their multimodal interfaces also introduce new safety vulnerabilities, making the safety evaluation challenging and critical. Existing red-teaming efforts are either restricted to a narrow set of adversarial patterns or depend heavily on manual engineering, lacking scalable exploration of emerging real-world VLM vulnerabilities. To bridge this gap, we propose ARMs, an adaptive red-teaming agent that systematically conducts comprehensive risk assessments for VLMs. Given a target harmful behavior or risk definition, ARMs automatically optimizes diverse red-teaming strategies with reasoning-enhanced multi-step orchestration, to effectively elicit harmful outputs from target VLMs. We propose 11 novel multimodal attack strategies, covering diverse adversarial patterns of VLMs (e.g., reasoning hijacking, contextual cloaking), and integrate 17 red-teaming algorithms into ARMs via model context protocol (MCP). To balance the diversity and effectiveness of the attack, we design a layered memory with an epsilon-greedy attack exploration algorithm. Extensive experiments on instance- and policy-based benchmarks show that ARMs achieves SOTA attack success rates, exceeding baselines by an average of 52.1% and surpassing 90% on Claude-4-Sonnet. We show that the diversity of red-teaming instances generated by ARMs is significantly higher, revealing emerging vulnerabilities in VLMs. Leveraging ARMs, we construct ARMs-Bench, a large-scale multimodal safety dataset comprising over 30K red-teaming instances spanning 51 diverse risk categories, grounded in both real-world multimodal threats and regulatory risks. Safety fine-tuning with ARMs-Bench substantially improves the robustness of VLMs while preserving their general utility, providing actionable guidance to improve multimodal safety alignment against emerging threats.

**arXiv ID:** 2510.02677
</details>

<details>
<summary><strong>AgenticRAG: Tool-Augmented Foundation Models for Zero-Shot Explainable Recommender Systems</strong> - Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Liu - [[pdf]](https://arxiv.org/pdf/2510.02668)</summary>

**Abstract:** Foundation models have revolutionized artificial intelligence, yet their application in recommender systems remains limited by reasoning opacity and knowledge constraints. This paper introduces AgenticRAG, a novel framework that combines tool-augmented foundation models with retrieval-augmented generation for zero-shot explainable recommendations. Our approach integrates external tool invocation, knowledge retrieval, and chain-of-thought reasoning to create autonomous recommendation agents capable of transparent decision-making without task-specific training. Experimental results on three real-world datasets demonstrate that AgenticRAG achieves consistent improvements over state-of-the-art baselines, with NDCG@10 improvements of 0.4\% on Amazon Electronics, 0.8\% on MovieLens-1M, and 1.6\% on Yelp datasets. The framework exhibits superior explainability while maintaining computational efficiency comparable to traditional methods.

**arXiv ID:** 2510.02668
</details>

<details>
<summary><strong>Work Zones challenge VLM Trajectory Planning: Toward Mitigation and Robust Autonomous Driving</strong> - Yifan Liao, Zhen Sun, Xiaoyun Qiu, Zixiao Zhao, Wenbing Tang, Xinlei He, Xinhu Zheng, Tianwei Zhang, Xinyi Huang, Xingshuo Han - [[pdf]](https://arxiv.org/pdf/2510.02803)</summary>

**Abstract:** Visual Language Models (VLMs), with powerful multimodal reasoning capabilities, are gradually integrated into autonomous driving by several automobile manufacturers to enhance planning capability in challenging environments. However, the trajectory planning capability of VLMs in work zones, which often include irregular layouts, temporary traffic control, and dynamically changing geometric structures, is still unexplored. To bridge this gap, we conduct the \textit{first} systematic study of VLMs for work zone trajectory planning, revealing that mainstream VLMs fail to generate correct trajectories in $68.0%$ of cases. To better understand these failures, we first identify candidate patterns via subgraph mining and clustering analysis, and then confirm the validity of $8$ common failure patterns through human verification. Building on these findings, we propose REACT-Drive, a trajectory planning framework that integrates VLMs with Retrieval-Augmented Generation (RAG). Specifically, REACT-Drive leverages VLMs to convert prior failure cases into constraint rules and executable trajectory planning code, while RAG retrieves similar patterns in new scenarios to guide trajectory generation. Experimental results on the ROADWork dataset show that REACT-Drive yields a reduction of around $3\times$ in average displacement error relative to VLM baselines under evaluation with Qwen2.5-VL. In addition, REACT-Drive yields the lowest inference time ($0.58$s) compared with other methods such as fine-tuning ($17.90$s). We further conduct experiments using a real vehicle in 15 work zone scenarios in the physical world, demonstrating the strong practicality of REACT-Drive.

**arXiv ID:** 2510.02803
</details>

<details>
<summary><strong>V2X-UniPool: Unifying Multimodal Perception and Knowledge Reasoning for Autonomous Driving</strong> - Xuewen Luo, Fengze Yang, Fan Ding, Xiangbo Gao, Shuo Xing, Yang Zhou, Zhengzhong Tu, Chenxi Liu - [[pdf]](https://arxiv.org/pdf/2506.02580)</summary>

**Abstract:** Autonomous driving (AD) has achieved significant progress, yet single-vehicle perception remains constrained by sensing range and occlusions. Vehicle-to-Everything (V2X) communication addresses these limits by enabling collaboration across vehicles and infrastructure, but it also faces heterogeneity, synchronization, and latency constraints. Language models offer strong knowledge-driven reasoning and decision-making capabilities, but they are not inherently designed to process raw sensor streams and are prone to hallucination. We propose V2X-UniPool, the first framework that unifies V2X perception with language-based reasoning for knowledge-driven AD. It transforms multimodal V2X data into structured, language-based knowledge, organizes it in a time-indexed knowledge pool for temporally consistent reasoning, and employs Retrieval-Augmented Generation (RAG) to ground decisions in real-time context. Experiments on the real-world DAIR-V2X dataset show that V2X-UniPool achieves state-of-the-art planning accuracy and safety while reducing communication cost by more than 80\%, achieving the lowest overhead among evaluated methods. These results highlight the promise of bridging V2X perception and language reasoning to advance scalable and trustworthy driving. Our code is available at: this https URL

**arXiv ID:** 2506.02580
</details>

<details>
<summary><strong>GUI-PRA: Process Reward Agent for GUI Tasks</strong> - Tao Xiong, Xavier Hu, Yurun Chen, Yuhang Liu, Changqiao Wu, Pengzhi Gao, Wei Liu, Jian Luan, Shengyu Zhang - [[pdf]](https://arxiv.org/pdf/2509.23263)</summary>

**Abstract:** Graphical User Interface (GUI) Agents powered by Multimodal Large Language Models (MLLMs) show significant potential for automating tasks. However, they often struggle with long-horizon tasks, leading to frequent failures. Process Reward Models (PRMs) are a promising solution, as they can guide these agents with crucial process signals during inference. Nevertheless, their application to the GUI domain presents unique challenges. When processing dense artificial inputs with long history data, PRMs suffer from a "lost in the middle" phenomenon, where the overwhelming historical context compromises the evaluation of the current step. Furthermore, standard PRMs lacks GUI changing awareness, providing static evaluations that are disconnected from the dynamic consequences of actions, a critical mismatch with the inherently dynamic nature of GUI tasks. In response to these challenges, we introduce GUI-PRA (Process Reward Agent for GUI Tasks), a judge agent designed to better provide process reward than standard PRM by intelligently processing historical context and actively perceiving UI state changes. Specifically, to directly combat the ``lost in the middle'' phenomenon, we introduce a dynamic memory mechanism consisting of two core components: a Relevance-based Retrieval Module to actively fetch pertinent information from long histories and a Progressive Summarization Module to dynamically condense growing interaction data, ensuring the model focuses on relevant context. Moreover, to address the lack of UI changing awareness, we introduce an Aadaptive UI Perception mechanism. This mechanism enables the agent to reason about UI state changes and dynamically select the most appropriate tool to gather grounded visual evidence, ensuring its evaluation is always informed by the current UI context.

**arXiv ID:** 2509.23263
</details>

<details>
<summary><strong>Learning to Route: A Rule-Driven Agent Framework for Hybrid-Source Retrieval-Augmented Generation</strong> - Haoyue Bai, Haoyu Wang, Shengyu Chen, Zhengzhang Chen, Lu-An Tang, Wei Cheng, Haifeng Chen, Yanjie Fu - [[pdf]](https://arxiv.org/pdf/2510.02388)</summary>

**Abstract:** Large Language Models (LLMs) have shown remarkable performance on general Question Answering (QA), yet they often struggle in domain-specific scenarios where accurate and up-to-date information is required. Retrieval-Augmented Generation (RAG) addresses this limitation by enriching LLMs with external knowledge, but existing systems primarily rely on unstructured documents, while largely overlooking relational databases, which provide precise, timely, and efficiently queryable factual information, serving as indispensable infrastructure in domains such as finance, healthcare, and scientific research. Motivated by this gap, we conduct a systematic analysis that reveals three central observations: (i) databases and documents offer complementary strengths across queries, (ii) naively combining both sources introduces noise and cost without consistent accuracy gains, and (iii) selecting the most suitable source for each query is crucial to balance effectiveness and efficiency. We further observe that query types show consistent regularities in their alignment with retrieval paths, suggesting that routing decisions can be effectively guided by systematic rules that capture these patterns. Building on these insights, we propose a rule-driven routing framework. A routing agent scores candidate augmentation paths based on explicit rules and selects the most suitable one; a rule-making expert agent refines the rules over time using QA feedback to maintain adaptability; and a path-level meta-cache reuses past routing decisions for semantically similar queries to reduce latency and cost. Experiments on three QA benchmarks demonstrate that our framework consistently outperforms static strategies and learned routing baselines, achieving higher accuracy while maintaining moderate computational cost.

**arXiv ID:** 2510.02388
</details>

<details>
<summary><strong>SurveyBench: How Well Can LLM(-Agents) Write Academic Surveys?</strong> - Zhaojun Sun, Xuzhou Zhu, Xuanhe Zhou, Xin Tong, Shuo Wang, Jie Fu, Guoliang Li, Zhiyuan Liu, Fan Wu - [[pdf]](https://arxiv.org/pdf/2510.03120)</summary>

**Abstract:** Academic survey writing, which distills vast literature into a coherent and insightful narrative, remains a labor-intensive and intellectually demanding task. While recent approaches, such as general DeepResearch agents and survey-specialized methods, can generate surveys automatically (a.k.a. LLM4Survey), their outputs often fall short of human standards and there lacks a rigorous, reader-aligned benchmark for thoroughly revealing their deficiencies. To fill the gap, we propose a fine-grained, quiz-driven evaluation framework SurveyBench, featuring (1) typical survey topics source from recent 11,343 arXiv papers and corresponding 4,947 high-quality surveys; (2) a multifaceted metric hierarchy that assesses the outline quality (e.g., coverage breadth, logical coherence), content quality (e.g., synthesis granularity, clarity of insights), and non-textual richness; and (3) a dual-mode evaluation protocol that includes content-based and quiz-based answerability tests, explicitly aligned with readers' informational needs. Results show SurveyBench effectively challenges existing LLM4Survey approaches (e.g., on average 21% lower than human in content-based evaluation).

**arXiv ID:** 2510.03120
</details>

<details>
<summary><strong>FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents</strong> - Imene Kerboua, Sahar Omidi Shayegan, Megh Thakkar, Xing Han Lù, Léo Boisvert, Massimo Caccia, Jérémy Espinas, Alexandre Aussem, Véronique Eglin, Alexandre Lacoste - [[pdf]](https://arxiv.org/pdf/2510.03204)</summary>

**Abstract:** Web agents powered by large language models (LLMs) must process lengthy web page observations to complete user goals; these pages often exceed tens of thousands of tokens. This saturates context limits and increases computational cost processing; moreover, processing full pages exposes agents to security risks such as prompt injection. Existing pruning strategies either discard relevant content or retain irrelevant context, leading to suboptimal action prediction. We introduce FocusAgent, a simple yet effective approach that leverages a lightweight LLM retriever to extract the most relevant lines from accessibility tree (AxTree) observations, guided by task goals. By pruning noisy and irrelevant content, FocusAgent enables efficient reasoning while reducing vulnerability to injection attacks. Experiments on WorkArena and WebArena benchmarks show that FocusAgent matches the performance of strong baselines, while reducing observation size by over 50%. Furthermore, a variant of FocusAgent significantly reduces the success rate of prompt-injection attacks, including banner and pop-up attacks, while maintaining task success performance in attack-free settings. Our results highlight that targeted LLM-based retrieval is a practical and robust strategy for building web agents that are efficient, effective, and secure.

**arXiv ID:** 2510.03204
</details>

<details>
<summary><strong>Triadic Multi-party Voice Activity Projection for Turn-taking in Spoken Dialogue Systems</strong> - Mikey Elmers, Koji Inoue, Divesh Lala, Tatsuya Kawahara - [[pdf]](https://arxiv.org/pdf/2507.07518)</summary>

**Abstract:** Turn-taking is a fundamental component of spoken dialogue, however conventional studies mostly involve dyadic settings. This work focuses on applying voice activity projection (VAP) to predict upcoming turn-taking in triadic multi-party scenarios. The goal of VAP models is to predict the future voice activity for each speaker utilizing only acoustic data. This is the first study to extend VAP into triadic conversation. We trained multiple models on a Japanese triadic dataset where participants discussed a variety of topics. We found that the VAP trained on triadic conversation outperformed the baseline for all models but that the type of conversation affected the accuracy. This study establishes that VAP can be used for turn-taking in triadic dialogue scenarios. Future work will incorporate this triadic VAP turn-taking model into spoken dialogue systems.

**arXiv ID:** 2507.07518
</details>

<details>
<summary><strong>Agent-ScanKit: Unraveling Memory and Reasoning of Multimodal Agents via Sensitivity Perturbations</strong> - Pengzhou Cheng, Lingzhong Dong, Zeng Wu, Zongru Wu, Xiangru Tang, Chengwei Qin, Zhuosheng Zhang, Gongshen Liu - [[pdf]](https://arxiv.org/pdf/2510.00496)</summary>

**Abstract:** Although numerous strategies have recently been proposed to enhance the autonomous interaction capabilities of multimodal agents in graphical user interface (GUI), their reliability remains limited when faced with complex or out-of-domain tasks. This raises a fundamental question: Are existing multimodal agents reasoning spuriously? In this paper, we propose \textbf{Agent-ScanKit}, a systematic probing framework to unravel the memory and reasoning capabilities of multimodal agents under controlled perturbations. Specifically, we introduce three orthogonal probing paradigms: visual-guided, text-guided, and structure-guided, each designed to quantify the contributions of memorization and reasoning without requiring access to model internals. In five publicly available GUI benchmarks involving 18 multimodal agents, the results demonstrate that mechanical memorization often outweighs systematic reasoning. Most of the models function predominantly as retrievers of training-aligned knowledge, exhibiting limited generalization. Our findings underscore the necessity of robust reasoning modeling for multimodal agents in real-world scenarios, offering valuable insights toward the development of reliable multimodal agents.

**arXiv ID:** 2510.00496
</details>

<details>
<summary><strong>Programming with Pixels: Can Computer-Use Agents do Software Engineering?</strong> - Pranjal Aggarwal, Sean Welleck - [[pdf]](https://arxiv.org/pdf/2502.18525)</summary>

**Abstract:** Computer-use agents (CUAs) hold the promise of performing a wide variety of general tasks, but current evaluations have primarily focused on simple scenarios. It therefore remains unclear whether such generalist agents can automate more sophisticated and specialized work such as software engineering (SWE). To investigate this, we introduce $\texttt{Programming with Pixels}$ (PwP), the first comprehensive computer-use environment for software engineering, where agents visually control an IDE to perform diverse software engineering tasks. To enable holistic evaluation, we also introduce \texttt{PwP-Bench}, a benchmark of 15 existing and new software-engineering tasks spanning multiple modalities, programming languages, and skillsets. We perform an extensive evaluation of state-of-the-art open-weight and closed-weight CUAs and find that when interacting purely visually, they perform significantly worse than specialized coding agents. However, when the same CUAs are given direct access to just two APIs-file editing and bash operations-performance jumps, often reaching the levels of specialized agents despite having a task-agnostic design. Furthermore, when given access to additional IDE tools via text APIs, all models show further gains. Our analysis shows that current CUAs fall short mainly due to limited visual grounding and the inability to take full advantage of the rich environment, leaving clear room for future this http URL establishes software engineering as a natural domain for benchmarking whether generalist computer-use agents can reach specialist-level performance on sophisticated tasks. Code and data released at this https URL

**arXiv ID:** 2502.18525
</details>

</details>

<details open>
<summary><h2>LLM Agents (12 papers)</h2></summary>

<details>
<summary><strong>BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks</strong> - Sagnik Anupam, Davis Brown, Shuo Li, Eric Wong, Hamed Hassani, Osbert Bastani - [[pdf]](https://arxiv.org/pdf/2510.02418)</summary>

**Abstract:** LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about captcha resolution. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale.

**arXiv ID:** 2510.02418
</details>

<details>
<summary><strong>Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents</strong> - Wonjoong Kim, Sangwu Park, Yeonjun In, Sein Kim, Dongha Lee, Chanyoung Park - [[pdf]](https://arxiv.org/pdf/2510.02837)</summary>

**Abstract:** Although recent tool-augmented benchmarks incorporate complex user requests and diverse tools, the evaluation methods for most of them remain limited to answer matching. However, as the number of steps required to resolve a user request increases, a proper evaluation of an agent's performance must go beyond the final answer to also assess the problem-solving trajectory, including previously ignored aspects such as efficiency, hallucination, and adaptivity. The most straightforward method for evaluating these aspects is to compare an agent's trajectory with the ground-truth trajectory, but this approach is fundamentally limited since annotating all valid ground-truth trajectories is prohibitively expensive. However, a simple LLM-based evaluator struggles to assess trajectories in detail without ground truth. To effectively evaluate the agents in this manner, we introduce TRACE, a framework for the multi-dimensional evaluation of tool-augmented LLM agent performance. By incorporating an evidence bank, which accumulates knowledge gathered from preceding reasoning steps, TRACE enables a multi-faceted analysis and evaluation of an agent's reasoning trajectory effectively. To validate our framework, we develop a new meta-evaluation dataset by augmenting existing benchmarks with diverse and flawed trajectories, each labeled with multi-faceted performance scores. Our results confirm that TRACE accurately evaluates these complex behaviors in a scalable and cost-effective manner, even with small open-source LLMs. Furthermore, we apply our method to evaluate the trajectories that agents produce while solving tool-augmented tasks, presenting previously unreported observations and their corresponding insights.

**arXiv ID:** 2510.02837
</details>

<details>
<summary><strong>AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering</strong> - Ziqing Wang, Chengsheng Mao, Xiaole Wen, Yuan Luo, Kaize Ding - [[pdf]](https://arxiv.org/pdf/2510.02328)</summary>

**Abstract:** Medical Multimodal Large Language Models (Med-MLLMs) have shown great promise in medical visual question answering (Med-VQA). However, when deployed in low-resource settings where abundant labeled data are unavailable, existing Med-MLLMs commonly fail due to their medical reasoning capability bottlenecks: (i) the intrinsic reasoning bottleneck that ignores the details from the medical image; (ii) the extrinsic reasoning bottleneck that fails to incorporate specialized medical knowledge. To address those limitations, we propose AMANDA, a training-free agentic framework that performs medical knowledge augmentation via LLM agents. Specifically, our intrinsic medical knowledge augmentation focuses on coarse-to-fine question decomposition for comprehensive diagnosis, while extrinsic medical knowledge augmentation grounds the reasoning process via biomedical knowledge graph retrieval. Extensive experiments across eight Med-VQA benchmarks demonstrate substantial improvements in both zero-shot and few-shot Med-VQA settings. The code is available at this https URL.

**arXiv ID:** 2510.02328
</details>

<details>
<summary><strong>Emission-GPT: A domain-specific language model agent for knowledge retrieval, emission inventory and data analysis</strong> - Jiashu Ye, Tong Wu, Weiwen Chen, Hao Zhang, Zeteng Lin, Xingxing Li, Shujuan Weng, Manni Zhu, Xin Yuan, Xinlong Hong, Jingjie Li, Junyu Zheng, Zhijiong Huang, Jing Tang - [[pdf]](https://arxiv.org/pdf/2510.02359)</summary>

**Abstract:** Improving air quality and addressing climate change relies on accurate understanding and analysis of air pollutant and greenhouse gas emissions. However, emission-related knowledge is often fragmented and highly specialized, while existing methods for accessing and compiling emissions data remain inefficient. These issues hinder the ability of non-experts to interpret emissions information, posing challenges to research and management. To address this, we present Emission-GPT, a knowledge-enhanced large language model agent tailored for the atmospheric emissions domain. Built on a curated knowledge base of over 10,000 documents (including standards, reports, guidebooks, and peer-reviewed literature), Emission-GPT integrates prompt engineering and question completion to support accurate domain-specific question answering. Emission-GPT also enables users to interactively analyze emissions data via natural language, such as querying and visualizing inventories, analyzing source contributions, and recommending emission factors for user-defined scenarios. A case study in Guangdong Province demonstrates that Emission-GPT can extract key insights--such as point source distributions and sectoral trends--directly from raw data with simple prompts. Its modular and extensible architecture facilitates automation of traditionally manual workflows, positioning Emission-GPT as a foundational tool for next-generation emission inventory development and scenario-based assessment.

**arXiv ID:** 2510.02359
</details>

<details>
<summary><strong>Spiral of Silence in Large Language Model Agents</strong> - Mingze Zhong, Meng Fang, Zijing Shi, Yuxuan Huang, Shunfeng Zheng, Yali Du, Ling Chen, Jun Wang - [[pdf]](https://arxiv.org/pdf/2510.02360)</summary>

**Abstract:** The Spiral of Silence (SoS) theory holds that individuals with minority views often refrain from speaking out for fear of social isolation, enabling majority positions to dominate public discourse. When the 'agents' are large language models (LLMs), however, the classical psychological explanation is not directly applicable, since SoS was developed for human societies. This raises a central question: can SoS-like dynamics nevertheless emerge from purely statistical language generation in LLM collectives? We propose an evaluation framework for examining SoS in LLM agents. Specifically, we consider four controlled conditions that systematically vary the availability of 'History' and 'Persona' signals. Opinion dynamics are assessed using trend tests such as Mann-Kendall and Spearman's rank, along with concentration measures including kurtosis and interquartile range. Experiments across open-source and closed-source models show that history and persona together produce strong majority dominance and replicate SoS patterns; history signals alone induce strong anchoring; and persona signals alone foster diverse but uncorrelated opinions, indicating that without historical anchoring, SoS dynamics cannot emerge. The work bridges computational sociology and responsible AI design, highlighting the need to monitor and mitigate emergent conformity in LLM-agent systems.

**arXiv ID:** 2510.02360
</details>

<details>
<summary><strong>Beyond Manuals and Tasks: Instance-Level Context Learning for LLM Agents</strong> - Kuntai Cai, Juncheng Liu, Xianglin Yang, Zhaojie Niu, Xiaokui Xiao, Xing Chen - [[pdf]](https://arxiv.org/pdf/2510.02369)</summary>

**Abstract:** Large language model (LLM) agents typically receive two kinds of context: (i) environment-level manuals that define interaction interfaces and global rules, and (ii) task-level guidance or demonstrations tied to specific goals. In this work, we identify a crucial but overlooked third type of context, instance-level context, which consists of verifiable and reusable facts tied to a specific environment instance, such as object locations, crafting recipes, and local rules. We argue that the absence of instance-level context is a common source of failure for LLM agents in complex tasks, as success often depends not only on reasoning over global rules or task prompts but also on making decisions based on precise and persistent facts. Acquiring such context requires more than memorization: the challenge lies in efficiently exploring, validating, and formatting these facts under tight interaction budgets. We formalize this problem as Instance-Level Context Learning (ILCL) and introduce our task-agnostic method to solve it. Our method performs a guided exploration, using a compact TODO forest to intelligently prioritize its next actions and a lightweight plan-act-extract loop to execute them. This process automatically produces a high-precision context document that is reusable across many downstream tasks and agents, thereby amortizing the initial exploration cost. Experiments across TextWorld, ALFWorld, and Crafter demonstrate consistent gains in both success and efficiency: for instance, ReAct's mean success rate in TextWorld rises from 37% to 95%, while IGE improves from 81% to 95%. By transforming one-off exploration into persistent, reusable knowledge, our method complements existing contexts to enable more reliable and efficient LLM agents.

**arXiv ID:** 2510.02369
</details>

<details>
<summary><strong>A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory</strong> - Qianshan Wei, Tengchao Yang, Yaochen Wang, Xinfeng Li, Lijun Li, Zhenfei Yin, Yi Zhan, Thorsten Holz, Zhiqiang Lin, XiaoFeng Wang - [[pdf]](https://arxiv.org/pdf/2510.02373)</summary>

**Abstract:** Large Language Model (LLM) agents use memory to learn from past interactions, enabling autonomous planning and decision-making in complex environments. However, this reliance on memory introduces a critical security risk: an adversary can inject seemingly harmless records into an agent's memory to manipulate its future behavior. This vulnerability is characterized by two core aspects: First, the malicious effect of injected records is only activated within a specific context, making them hard to detect when individual memory entries are audited in isolation. Second, once triggered, the manipulation can initiate a self-reinforcing error cycle: the corrupted outcome is stored as precedent, which not only amplifies the initial error but also progressively lowers the threshold for similar attacks in the future. To address these challenges, we introduce A-MemGuard (Agent-Memory Guard), the first proactive defense framework for LLM agent memory. The core idea of our work is the insight that memory itself must become both self-checking and self-correcting. Without modifying the agent's core architecture, A-MemGuard combines two mechanisms: (1) consensus-based validation, which detects anomalies by comparing reasoning paths derived from multiple related memories and (2) a dual-memory structure, where detected failures are distilled into ``lessons'' stored separately and consulted before future actions, breaking error cycles and enabling adaptation. Comprehensive evaluations on multiple benchmarks show that A-MemGuard effectively cuts attack success rates by over 95% while incurring a minimal utility cost. This work shifts LLM memory security from static filtering to a proactive, experience-driven model where defenses strengthen over time. Our code is available in this https URL

**arXiv ID:** 2510.02373
</details>

<details>
<summary><strong>Gala: Global LLM Agents for Text-to-Model Translation</strong> - Junyang Cai, Serdar Kadioglu, Bistra Dilkina - [[pdf]](https://arxiv.org/pdf/2509.08970)</summary>

**Abstract:** Natural language descriptions of optimization or satisfaction problems are challenging to translate into correct MiniZinc models, as this process demands both logical reasoning and constraint programming expertise. We introduce Gala, a framework that addresses this challenge with a global agentic approach: multiple specialized large language model (LLM) agents decompose the modeling task by global constraint type. Each agent is dedicated to detecting and generating code for a specific class of global constraint, while a final assembler agent integrates these constraint snippets into a complete MiniZinc model. By dividing the problem into smaller, well-defined sub-tasks, each LLM handles a simpler reasoning challenge, potentially reducing overall complexity. We conduct initial experiments with several LLMs and show better performance against baselines such as one-shot prompting and chain-of-thought prompting. Finally, we outline a comprehensive roadmap for future work, highlighting potential enhancements and directions for improvement.

**arXiv ID:** 2509.08970
</details>

<details>
<summary><strong>MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents</strong> - George Fatouros, Kostas Metaxas, John Soldatos, Manos Karathanassis - [[pdf]](https://arxiv.org/pdf/2502.00415)</summary>

**Abstract:** MarketSenseAI is a novel framework for holistic stock analysis which leverages Large Language Models (LLMs) to process financial news, historical prices, company fundamentals and the macroeconomic environment to support decision making in stock analysis and selection. In this paper, we present the latest advancements on MarketSenseAI, driven by rapid technological expansion in LLMs. Through a novel architecture combining Retrieval-Augmented Generation and LLM agents, the framework processes SEC filings and earnings calls, while enriching macroeconomic analysis through systematic processing of diverse institutional reports. We demonstrate a significant improvement in fundamental analysis accuracy over the previous version. Empirical evaluation on S\&P 100 stocks over two years (2023-2024) shows MarketSenseAI achieving cumulative returns of 125.9% compared to the index return of 73.5%, while maintaining comparable risk profiles. Further validation on S\&P 500 stocks during 2024 demonstrates the framework's scalability, delivering a 33.8% higher Sortino ratio than the market. This work marks a significant advancement in applying LLM technology to financial analysis, offering insights into the robustness of LLM-driven investment strategies.

**arXiv ID:** 2502.00415
</details>

<details>
<summary><strong>DatawiseAgent: A Notebook-Centric LLM Agent Framework for Adaptive and Robust Data Science Automation</strong> - Ziming You, Yumiao Zhang, Dexuan Xu, Yiwei Lou, Yandong Yan, Wei Wang, Huaming Zhang, Yu Huang - [[pdf]](https://arxiv.org/pdf/2503.07044)</summary>

**Abstract:** Existing large language model (LLM) agents for automating data science show promise, but they remain constrained by narrow task scopes, limited generalization across tasks and models, and over-reliance on state-of-the-art (SOTA) LLMs. We introduce DatawiseAgent, a notebook-centric LLM agent framework for adaptive and robust data science automation. Inspired by how human data scientists work in computational notebooks, DatawiseAgent introduces a unified interaction representation and a multi-stage architecture based on finite-state transducers (FSTs). This design enables flexible long-horizon planning, progressive solution development, and robust recovery from execution failures. Extensive experiments across diverse data science scenarios and models show that DatawiseAgent consistently achieves SOTA performance by surpassing strong baselines such as AutoGen and TaskWeaver, demonstrating superior effectiveness and adaptability. Further evaluations reveal graceful performance degradation under weaker or smaller models, underscoring the robustness and scalability.

**arXiv ID:** 2503.07044
</details>

<details>
<summary><strong>FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering</strong> - Chanyeol Choi, Jihoon Kwon, Alejandro Lopez-Lira, Chaewoon Kim, Minjae Kim, Juneha Hwang, Jaeseon Ha, Hojun Choi, Suyeol Yun, Yongjin Kim, Yongjae Lee - [[pdf]](https://arxiv.org/pdf/2508.14052)</summary>

**Abstract:** Accurate information retrieval (IR) is critical in the financial domain, where investors must identify relevant information from large collections of documents. Traditional IR methods -- whether sparse or dense -- often fall short in retrieval accuracy, as it requires not only capturing semantic similarity but also performing fine-grained reasoning over document structure and domain-specific knowledge. Recent advances in large language models (LLMs) have opened up new opportunities for retrieval with multi-step reasoning, where the model ranks passages through iterative reasoning about which information is most relevant to a given query. However, there exists no benchmark to evaluate such capabilities in the financial domain. To address this gap, we introduce FinAgentBench, the first large-scale benchmark for evaluating retrieval with multi-step reasoning in finance -- a setting we term agentic retrieval. The benchmark consists of 26K expert-annotated examples on S&P-500 listed firms and assesses whether LLM agents can (1) identify the most relevant document type among candidates, and (2) pinpoint the key passage within the selected document. Our evaluation framework explicitly separates these two reasoning steps to address context limitations. This design enables to provide a quantitative basis for understanding retrieval-centric LLM behavior in finance. We evaluate a suite of state-of-the-art models and further demonstrated how targeted fine-tuning can significantly improve agentic retrieval performance. Our benchmark provides a foundation for studying retrieval-centric LLM behavior in complex, domain-specific tasks for finance.

**arXiv ID:** 2508.14052
</details>

<details>
<summary><strong>From Trace to Line: LLM Agent for Real-World OSS Vulnerability Localization</strong> - Haoran Xi, Minghao Shao, Brendan Dolan-Gavitt, Muhammad Shafique, Ramesh Karri - [[pdf]](https://arxiv.org/pdf/2510.02389)</summary>

**Abstract:** Large language models show promise for vulnerability discovery, yet prevailing methods inspect code in isolation, struggle with long contexts, and focus on coarse function- or file-level detections - offering limited actionable guidance to engineers who need precise line-level localization and targeted patches in real-world software development. We present T2L-Agent (Trace-to-Line Agent), a project-level, end-to-end framework that plans its own analysis and progressively narrows scope from modules to exact vulnerable lines. T2L-Agent couples multi-round feedback with an Agentic Trace Analyzer (ATA) that fuses runtime evidence - crash points, stack traces, and coverage deltas - with AST-based code chunking, enabling iterative refinement beyond single pass predictions and translating symptoms into actionable, line-level diagnoses. To benchmark line-level vulnerability discovery, we introduce T2L-ARVO, a diverse, expert-verified 50-case benchmark spanning five crash families and real-world projects. T2L-ARVO is specifically designed to support both coarse-grained detection and fine-grained localization, enabling rigorous evaluation of systems that aim to move beyond file-level predictions. On T2L-ARVO, T2L-Agent achieves up to 58.0% detection and 54.8% line-level localization, substantially outperforming baselines. Together, the framework and benchmark push LLM-based vulnerability detection from coarse identification toward deployable, robust, precision diagnostics that reduce noise and accelerate patching in open-source software workflows.

**arXiv ID:** 2510.02389
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (9 papers)</h2></summary>

<details>
<summary><strong>Agentic Additive Manufacturing Alloy Discovery</strong> - Peter Pak, Achuth Chandrasekhar, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2510.02567)</summary>

**Abstract:** Agentic systems enable the intelligent use of research tooling, augmenting a researcher's ability to investigate and propose novel solutions to existing problems. Within Additive Manufacturing (AM), alloy discovery remains a complex challenge, often requiring expertise in the various domains of materials science, thermodynamic simulations, and experimental analysis. Large Language Model (LLM) enabled agents can facilitate this endeavor by utilizing their extensive knowledge base to dispatch tool calls via Model Context Protocol (MCP) to perform actions such as Thermo-Calc property diagram calculations and lack of fusion process map generation. In addition, the multi-agent system developed in this work is able to effectively reason through complex user prompts and provide analysis on the printability of proposed alloys. These agents can dynamically adjust their task trajectory to the outcomes of tool call results, effectively enabling autonomous decision-making in practical environments. This work aims to utilize LLM enabled agents to automate and accelerate the task of alloy discovery within the field of additive manufacturing and showcase the benefits of adopting this multi-agent system.

**arXiv ID:** 2510.02567
</details>

<details>
<summary><strong>Orchestrating Human-AI Teams: The Manager Agent as a Unifying Research Challenge</strong> - Charlie Masters, Advaith Vellanki, Jiangbo Shangguan, Bart Kultys, Jonathan Gilmore, Alastair Moore, Stefano V. Albrecht - [[pdf]](https://arxiv.org/pdf/2510.02557)</summary>

**Abstract:** While agentic AI has advanced in automating individual tasks, managing complex multi-agent workflows remains a challenging problem. This paper presents a research vision for autonomous agentic systems that orchestrate collaboration within dynamic human-AI teams. We propose the Autonomous Manager Agent as a core challenge: an agent that decomposes complex goals into task graphs, allocates tasks to human and AI workers, monitors progress, adapts to changing conditions, and maintains transparent stakeholder communication. We formalize workflow management as a Partially Observable Stochastic Game and identify four foundational challenges: (1) compositional reasoning for hierarchical decomposition, (2) multi-objective optimization under shifting preferences, (3) coordination and planning in ad hoc teams, and (4) governance and compliance by design. To advance this agenda, we release MA-Gym, an open-source simulation and evaluation framework for multi-agent workflow orchestration. Evaluating GPT-5-based Manager Agents across 20 workflows, we find they struggle to jointly optimize for goal completion, constraint adherence, and workflow runtime - underscoring workflow management as a difficult open problem. We conclude with organizational and ethical implications of autonomous management systems.

**arXiv ID:** 2510.02557
</details>

<details>
<summary><strong>A Benchmark Study of Deep Reinforcement Learning Algorithms for the Container Stowage Planning Problem</strong> - Yunqi Huang, Nishith Chennakeshava, Alexis Carras, Vladislav Neverov, Wei Liu, Aske Plaat, Yingjie Fan - [[pdf]](https://arxiv.org/pdf/2510.02589)</summary>

**Abstract:** Container stowage planning (CSPP) is a critical component of maritime transportation and terminal operations, directly affecting supply chain efficiency. Owing to its complexity, CSPP has traditionally relied on human expertise. While reinforcement learning (RL) has recently been applied to CSPP, systematic benchmark comparisons across different algorithms remain limited. To address this gap, we develop a Gym environment that captures the fundamental features of CSPP and extend it to include crane scheduling in both multi-agent and single-agent formulations. Within this framework, we evaluate five RL algorithms: DQN, QR-DQN, A2C, PPO, and TRPO under multiple scenarios of varying complexity. The results reveal distinct performance gaps with increasing complexity, underscoring the importance of algorithm choice and problem formulation for CSPP. Overall, this paper benchmarks multiple RL methods for CSPP while providing a reusable Gym environment with crane scheduling, thus offering a foundation for future research and practical deployment in maritime logistics.

**arXiv ID:** 2510.02589
</details>

<details>
<summary><strong>AutoMaAS: Self-Evolving Multi-Agent Architecture Search for Large Language Models</strong> - Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Liu - [[pdf]](https://arxiv.org/pdf/2510.02669)</summary>

**Abstract:** Multi-agent systems powered by large language models have demonstrated remarkable capabilities across diverse domains, yet existing automated design approaches seek monolithic solutions that fail to adapt resource allocation based on query complexity and domain requirements. This paper introduces AutoMaAS, a self-evolving multi-agent architecture search framework that leverages neural architecture search principles to automatically discover optimal agent configurations through dynamic operator lifecycle management and automated machine learning techniques. Our approach incorporates four key innovations: (1) automatic operator generation, fusion, and elimination based on performance-cost analysis, (2) dynamic cost-aware optimization with real-time parameter adjustment, (3) online feedback integration for continuous architecture refinement, and (4) enhanced interpretability through decision tracing mechanisms. Extensive experiments across six benchmarks demonstrate that AutoMaAS achieves 1.0-7.1\% performance improvement while reducing inference costs by 3-5\% compared to state-of-the-art methods. The framework shows superior transferability across datasets and LLM backbones, establishing a new paradigm for automated multi-agent system design in the era of large language models.

**arXiv ID:** 2510.02669
</details>

<details>
<summary><strong>Improving Cooperation in Collaborative Embodied AI</strong> - Hima Jacob Leven Suprabha, Laxmi Nag Laxminarayan Nagesh, Ajith Nair, Alvin Reuben Amal Selvaster, Ayan Khan, Raghuram Damarla, Sanju Hannah Samuel, Sreenithi Saravana Perumal, Titouan Puech, Venkataramireddy Marella, Vishal Sonar, Alessandro Suglia, Oliver Lemon - [[pdf]](https://arxiv.org/pdf/2510.03153)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into multiagent systems has opened new possibilities for collaborative reasoning and cooperation with AI agents. This paper explores different prompting methods and evaluates their effectiveness in enhancing agent collaborative behaviour and decision-making. We enhance CoELA, a framework designed for building Collaborative Embodied Agents that leverage LLMs for multi-agent communication, reasoning, and task coordination in shared virtual spaces. Through systematic experimentation, we examine different LLMs and prompt engineering strategies to identify optimised combinations that maximise collaboration performance. Furthermore, we extend our research by integrating speech capabilities, enabling seamless collaborative voice-based interactions. Our findings highlight the effectiveness of prompt optimisation in enhancing collaborative agent performance; for example, our best combination improved the efficiency of the system running with Gemma3 by 22% compared to the original CoELA system. In addition, the speech integration provides a more engaging user interface for iterative system development and demonstrations.

**arXiv ID:** 2510.03153
</details>

<details>
<summary><strong>CoDA: Agentic Systems for Collaborative Data Visualization</strong> - Zichen Chen, Jiefeng Chen, Sercan Ö. Arik, Misha Sra, Tomas Pfister, Jinsung Yoon - [[pdf]](https://arxiv.org/pdf/2510.03194)</summary>

**Abstract:** Deep research has revolutionized data analysis, yet data scientists still devote substantial time to manually crafting visualizations, highlighting the need for robust automation from natural language queries. However, current systems struggle with complex datasets containing multiple files and iterative refinement. Existing approaches, including simple single- or multi-agent systems, often oversimplify the task, focusing on initial query parsing while failing to robustly manage data complexity, code errors, or final visualization quality. In this paper, we reframe this challenge as a collaborative multi-agent problem. We introduce CoDA, a multi-agent system that employs specialized LLM agents for metadata analysis, task planning, code generation, and self-reflection. We formalize this pipeline, demonstrating how metadata-focused analysis bypasses token limits and quality-driven refinement ensures robustness. Extensive evaluations show CoDA achieves substantial gains in the overall score, outperforming competitive baselines by up to 41.5%. This work demonstrates that the future of visualization automation lies not in isolated code generation but in integrated, collaborative agentic workflows.

**arXiv ID:** 2510.03194
</details>

<details>
<summary><strong>A Trajectory Generator for High-Density Traffic and Diverse Agent-Interaction Scenarios</strong> - Ruining Yang, Yi Xu, Yixiao Chen, Yun Fu, Lili Su - [[pdf]](https://arxiv.org/pdf/2510.02627)</summary>

**Abstract:** Accurate trajectory prediction is fundamental to autonomous driving, as it underpins safe motion planning and collision avoidance in complex environments. However, existing benchmark datasets suffer from a pronounced long-tail distribution problem, with most samples drawn from low-density scenarios and simple straight-driving behaviors. This underrepresentation of high-density scenarios and safety critical maneuvers such as lane changes, overtaking and turning is an obstacle to model generalization and leads to overly optimistic evaluations. To address these challenges, we propose a novel trajectory generation framework that simultaneously enhances scenarios density and enriches behavioral diversity. Specifically, our approach converts continuous road environments into a structured grid representation that supports fine-grained path planning, explicit conflict detection, and multi-agent coordination. Built upon this representation, we introduce behavior-aware generation mechanisms that combine rule-based decision triggers with Frenet-based trajectory smoothing and dynamic feasibility constraints. This design allows us to synthesize realistic high-density scenarios and rare behaviors with complex interactions that are often missing in real data. Extensive experiments on the large-scale Argoverse 1 and Argoverse 2 datasets demonstrate that our method significantly improves both agent density and behavior diversity, while preserving motion realism and scenario-level safety. Our synthetic data also benefits downstream trajectory prediction models and enhances performance in challenging high-density scenarios.

**arXiv ID:** 2510.02627
</details>

<details>
<summary><strong>UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization</strong> - Qing Huang, Zhipei Xu, Xuanyu Zhang, Jian Zhang - [[pdf]](https://arxiv.org/pdf/2510.03161)</summary>

**Abstract:** With the rapid advancements in image generation, synthetic images have become increasingly realistic, posing significant societal risks, such as misinformation and fraud. Forgery Image Detection and Localization (FIDL) thus emerges as essential for maintaining information integrity and societal security. Despite impressive performances by existing domain-specific detection methods, their practical applicability remains limited, primarily due to their narrow specialization, poor cross-domain generalization, and the absence of an integrated adaptive framework. To address these issues, we propose UniShield, the novel multi-agent-based unified system capable of detecting and localizing image forgeries across diverse domains, including image manipulation, document manipulation, DeepFake, and AI-generated images. UniShield innovatively integrates a perception agent with a detection agent. The perception agent intelligently analyzes image features to dynamically select suitable detection models, while the detection agent consolidates various expert detectors into a unified framework and generates interpretable reports. Extensive experiments show that UniShield achieves state-of-the-art results, surpassing both existing unified approaches and domain-specific detectors, highlighting its superior practicality, adaptiveness, and scalability.

**arXiv ID:** 2510.03161
</details>

<details>
<summary><strong>MobiLLM: An Agentic AI Framework for Closed-Loop Threat Mitigation in 6G Open RANs</strong> - Prakhar Sharma, Haohuang Wen, Vinod Yegneswaran, Ashish Gehani, Phillip Porras, Zhiqiang Lin - [[pdf]](https://arxiv.org/pdf/2509.21634)</summary>

**Abstract:** The evolution toward 6G networks is being accelerated by the Open Radio Access Network (O-RAN) paradigm -- an open, interoperable architecture that enables intelligent, modular applications across public telecom and private enterprise domains. While this openness creates unprecedented opportunities for innovation, it also expands the attack surface, demanding resilient, low-cost, and autonomous security solutions. Legacy defenses remain largely reactive, labor-intensive, and inadequate for the scale and complexity of next-generation systems. Current O-RAN applications focus mainly on network optimization or passive threat detection, with limited capability for closed-loop, automated response.
To address this critical gap, we present an agentic AI framework for fully automated, end-to-end threat mitigation in 6G O-RAN environments. MobiLLM orchestrates security workflows through a modular multi-agent system powered by Large Language Models (LLMs). The framework features a Threat Analysis Agent for real-time data triage, a Threat Classification Agent that uses Retrieval-Augmented Generation (RAG) to map anomalies to specific countermeasures, and a Threat Response Agent that safely operationalizes mitigation actions via O-RAN control interfaces. Grounded in trusted knowledge bases such as the MITRE FiGHT framework and 3GPP specifications, and equipped with robust safety guardrails, MobiLLM provides a blueprint for trustworthy AI-driven network security. Initial evaluations demonstrate that MobiLLM can effectively identify and orchestrate complex mitigation strategies, significantly reducing response latency and showcasing the feasibility of autonomous security operations in 6G.

**arXiv ID:** 2509.21634
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>ToolTweak: An Attack on Tool Selection in LLM-based Agents</strong> - Jonathan Sneh, Ruomei Yan, Jialin Yu, Philip Torr, Yarin Gal, Sunando Sengupta, Eric Sommerlade, Alasdair Paren, Adel Bibi - [[pdf]](https://arxiv.org/pdf/2510.02554)</summary>

**Abstract:** As LLMs increasingly power agents that interact with external tools, tool use has become an essential mechanism for extending their capabilities. These agents typically select tools from growing databases or marketplaces to solve user tasks, creating implicit competition among tool providers and developers for visibility and usage. In this paper, we show that this selection process harbors a critical vulnerability: by iteratively manipulating tool names and descriptions, adversaries can systematically bias agents toward selecting specific tools, gaining unfair advantage over equally capable alternatives. We present ToolTweak, a lightweight automatic attack that increases selection rates from a baseline of around 20% to as high as 81%, with strong transferability between open-source and closed-source models. Beyond individual tools, we show that such attacks cause distributional shifts in tool usage, revealing risks to fairness, competition, and security in emerging tool ecosystems. To mitigate these risks, we evaluate two defenses: paraphrasing and perplexity filtering, which reduce bias and lead agents to select functionally similar tools more equally. All code will be open-sourced upon acceptance.

**arXiv ID:** 2510.02554
</details>

<details>
<summary><strong>Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair</strong> - José Cambronero, Michele Tufano, Sherry Shi, Renyao Wei, Grant Uy, Runxiang Cheng, Chin-Jung Liu, Shiying Pan, Satish Chandra, Pat Rondon - [[pdf]](https://arxiv.org/pdf/2510.03217)</summary>

**Abstract:** Agentic Automated Program Repair (APR) is increasingly tackling complex, repository-level bugs in industry, but ultimately agent-generated patches still need to be reviewed by a human before committing them to ensure they address the bug. Showing unlikely patches to developers can lead to substantial noise, wasting valuable developer time and eroding trust in automated code changes. We introduce two complementary LLM-based policies to reduce such noise: bug abstention and patch validation policies. Bug abstention excludes bugs that the agentic APR system is unlikely to fix. Patch validation rejects patches that are unlikely to be a good fix for the given bug. We evaluate both policies on three sets of bugs from Google's codebase, and their candidate patches generated by an internal agentic APR system. On a set of 174 human-reported bugs, removing bugs and patch trajectories rejected by our policies can raise success rates by up to 13 percentage points and 15 percentage points, respectively, and by up to 39 percentage points in combination. On null pointer exceptions and sanitizer-reported bugs with machine-generated bug reports, patch validation also improves average single-sample success rates. This two-policy approach provides a practical path to the reliable, industrial-scale deployment of agentic APR systems.

**arXiv ID:** 2510.03217
</details>

<details>
<summary><strong>SBP-YOLO:A Lightweight Real-Time Model for Detecting Speed Bumps and Potholes toward Intelligent Vehicle Suspension Systems</strong> - Chuanqi Liang, Jie Fu, Miao Yu, Lei Luo - [[pdf]](https://arxiv.org/pdf/2508.01339)</summary>

**Abstract:** Speed bumps and potholes are the most common road anomalies, significantly affecting ride comfort and vehicle stability. Preview-based suspension control mitigates their impact by detecting such irregularities in advance and adjusting suspension parameters proactively. Accurate and real-time detection is essential, but embedded deployment is constrained by limited computational resources and the small size of targets in input this http URL address these challenges, this paper proposes SBP-YOLO, an efficient detection framework for speed bumps and potholes in embedded systems. Built upon YOLOv11n, it integrates GhostConv and VoVGSCSPC modules in the backbone and neck to reduce computation while enhancing multi-scale semantic features. A P2-level branch improves small-object detection, and a lightweight and efficient detection head (LEDH) maintains accuracy with minimal overhead. A hybrid training strategy further enhances robustness under varying road and environmental conditions, combining NWD loss, BCKD knowledge distillation, and Albumentations-based augmentation. Experiments show that SBP-YOLO achieves 87.0% mAP, outperforming the YOLOv11n baseline by 5.8%. After TensorRT FP16 quantization, it runs at 139.5 FPS on Jetson AGX Xavier, yielding a 12.4% speedup over the P2-enhanced YOLOv11. These results demonstrate the framework's suitability for fast, low-latency road condition perception in embedded suspension control systems.

**arXiv ID:** 2508.01339
</details>

<details>
<summary><strong>Point Cloud-Based Control Barrier Functions for Model Predictive Control in Safety-Critical Navigation of Autonomous Mobile Robots</strong> - Faduo Liang, Yunfeng Yang, Shi-Lu Dai - [[pdf]](https://arxiv.org/pdf/2510.02885)</summary>

**Abstract:** In this work, we propose a novel motion planning algorithm to facilitate safety-critical navigation for autonomous mobile robots. The proposed algorithm integrates a real-time dynamic obstacle tracking and mapping system that categorizes point clouds into dynamic and static components. For dynamic point clouds, the Kalman filter is employed to estimate and predict their motion states. Based on these predictions, we extrapolate the future states of dynamic point clouds, which are subsequently merged with static point clouds to construct the forward-time-domain (FTD) map. By combining control barrier functions (CBFs) with nonlinear model predictive control, the proposed algorithm enables the robot to effectively avoid both static and dynamic obstacles. The CBF constraints are formulated based on risk points identified through collision detection between the predicted future states and the FTD map. Experimental results from both simulated and real-world scenarios demonstrate the efficacy of the proposed algorithm in complex environments. In simulation experiments, the proposed algorithm is compared with two baseline approaches, showing superior performance in terms of safety and robustness in obstacle avoidance. The source code is released for the reference of the robotics community.

**arXiv ID:** 2510.02885
</details>

<details>
<summary><strong>Embracing Evolution: A Call for Body-Control Co-Design in Embodied Humanoid Robot</strong> - Guiliang Liu, Bo Yue, Yi Jin Kim, Kui Jia - [[pdf]](https://arxiv.org/pdf/2510.03081)</summary>

**Abstract:** Humanoid robots, as general-purpose physical agents, must integrate both intelligent control and adaptive morphology to operate effectively in diverse real-world environments. While recent research has focused primarily on optimizing control policies for fixed robot structures, this position paper argues for evolving both control strategies and humanoid robots' physical structure under a co-design mechanism. Inspired by biological evolution, this approach enables robots to iteratively adapt both their form and behavior to optimize performance within task-specific and resource-constrained contexts. Despite its promise, co-design in humanoid robotics remains a relatively underexplored domain, raising fundamental questions about its feasibility and necessity in achieving true embodied intelligence. To address these challenges, we propose practical co-design methodologies grounded in strategic exploration, Sim2Real transfer, and meta-policy learning. We further argue for the essential role of co-design by analyzing it from methodological, application-driven, and community-oriented perspectives. Striving to guide and inspire future studies, we present open research questions, spanning from short-term innovations to long-term goals. This work positions co-design as a cornerstone for developing the next generation of intelligent and adaptable humanoid agents.

**arXiv ID:** 2510.03081
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (25 papers)</h2></summary>

<details>
<summary><strong>Consolidating Reinforcement Learning for Multimodal Discrete Diffusion Models</strong> - Tianren Ma, Mu Zhang, Yibing Wang, Qixiang Ye - [[pdf]](https://arxiv.org/pdf/2510.02880)</summary>

**Abstract:** Optimizing discrete diffusion model (DDM) with rewards remains a challenge: the non-autoregressive paradigm makes importance sampling intractable and rollout complex, puzzling reinforcement learning methods such as Group Relative Policy Optimization (GRPO). In this study, we introduce MaskGRPO, the first viable approach to enable scalable multimodal reinforcement learning in discrete diffusion with effective importance sampling and modality-specific adaptations. To this end, we first clarify the theoretical foundation for DDMs, which facilitates building an importance estimator that captures valuable token fluctuation for gradient updates. We then delicately tailored the rollout method for visual sequences, which yields diverse completions and reliable optimization gradients. Upon math reasoning, coding, and visual generation benchmarks, MaskGRPO brings more stable and efficient updates, leading to stronger reasoning performance and better generation quality. This study establishes MaskGRPO as a systematic policy optimization approach and the first practical way for discretized visual diffusion.

**arXiv ID:** 2510.02880
</details>

<details>
<summary><strong>Agentic-AI Healthcare: Multilingual, Privacy-First Framework with MCP Agents</strong> - Mohammed A. Shehab - [[pdf]](https://arxiv.org/pdf/2510.02325)</summary>

**Abstract:** This paper introduces Agentic-AI Healthcare, a privacy-aware, multilingual, and explainable research prototype developed as a single-investigator project. The system leverages the emerging Model Context Protocol (MCP) to orchestrate multiple intelligent agents for patient interaction, including symptom checking, medication suggestions, and appointment scheduling. The platform integrates a dedicated Privacy and Compliance Layer that applies role-based access control (RBAC), AES-GCM field-level encryption, and tamper-evident audit logging, aligning with major healthcare data protection standards such as HIPAA (US), PIPEDA (Canada), and PHIPA (Ontario). Example use cases demonstrate multilingual patient-doctor interaction (English, French, Arabic) and transparent diagnostic reasoning powered by large language models. As an applied AI contribution, this work highlights the feasibility of combining agentic orchestration, multilingual accessibility, and compliance-aware architecture in healthcare applications. This platform is presented as a research prototype and is not a certified medical device.

**arXiv ID:** 2510.02325
</details>

<details>
<summary><strong>From Pixels to Factors: Learning Independently Controllable State Variables for Reinforcement Learning</strong> - Rafael Rodriguez-Sanchez, Cameron Allen, George Konidaris - [[pdf]](https://arxiv.org/pdf/2510.02484)</summary>

**Abstract:** Algorithms that exploit factored Markov decision processes are far more sample-efficient than factor-agnostic methods, yet they assume a factored representation is known a priori -- a requirement that breaks down when the agent sees only high-dimensional observations. Conversely, deep reinforcement learning handles such inputs but cannot benefit from factored structure. We address this representation problem with Action-Controllable Factorization (ACF), a contrastive learning approach that uncovers independently controllable latent variables -- state components each action can influence separately. ACF leverages sparsity: actions typically affect only a subset of variables, while the rest evolve under the environment's dynamics, yielding informative data for contrastive training. ACF recovers the ground truth controllable factors directly from pixel observations on three benchmarks with known factored structure -- Taxi, FourRooms, and MiniGrid-DoorKey -- consistently outperforming baseline disentanglement algorithms.

**arXiv ID:** 2510.02484
</details>

<details>
<summary><strong>Oracle-RLAIF: An Improved Fine-Tuning Framework for Multi-modal Video Models through Reinforcement Learning from Ranking Feedback</strong> - Derek Shi, Ruben Glatt, Christine Klymko, Shubham Mohole, Hongjun Choi, Shashank Kushwaha, Sam Sakla, Felipe Leno da Silva - [[pdf]](https://arxiv.org/pdf/2510.02561)</summary>

**Abstract:** Recent advances in large video-language models (VLMs) rely on extensive fine-tuning techniques that strengthen alignment between textual and visual comprehension. Leading pipelines typically pair supervised fine-tuning (SFT) with reinforcement learning from preference data to enhance video comprehension. However, as VLMs scale in parameter size, so does the cost of gathering enough human feedback. To make fine-tuning more cost-effective, recent frameworks explore reinforcement learning with AI feedback (RLAIF), which replace human preference with AI as a judge. Current RLAIF frameworks rely on a specialized reward model trained with video narratives to create calibrated scalar rewards-- an expensive and restrictive pipeline. We propose Oracle-RLAIF, a novel framework that replaces the trained reward model with a more general Oracle ranker which acts as a drop-in model ranking candidate model responses rather than scoring them. Alongside Oracle-RLAIF, we introduce $GRPO_{rank}$, a novel rank-based loss function based on Group Relative Policy Optimization (GRPO) that directly optimizes ordinal feedback with rank-aware advantages. Empirically, we demonstrate that Oracle-RLAIF consistently outperforms leading VLMs using existing fine-tuning methods when evaluated across various video comprehension benchmarks. Oracle-RLAIF paves the path to creating flexible and data-efficient frameworks for aligning large multi-modal video models with reinforcement learning from rank rather than score.

**arXiv ID:** 2510.02561
</details>

<details>
<summary><strong>RAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization</strong> - Kai Fukazawa, Kunal Mundada, Iman Soltani - [[pdf]](https://arxiv.org/pdf/2510.02695)</summary>

**Abstract:** In safety-critical domains where online data collection is infeasible, offline reinforcement learning (RL) offers an attractive alternative but only if policies deliver high returns without incurring catastrophic lower-tail risk. Prior work on risk-averse offline RL achieves safety at the cost of value conservatism and restricted policy classes, whereas expressive policies are only used in risk-neutral settings. Here, we address this gap by introducing the \textbf{Risk-Aware Multimodal Actor-Critic (RAMAC)} framework, which couples an \emph{expressive generative actor} with a distributional critic. The RAMAC differentiates composite objective combining distributional risk and BC loss through the generative path, achieving risk-sensitive learning in complex multimodal scenarios. We instantiate RAMAC with diffusion and flow-matching actors and observe consistent gains in $\mathrm{CVaR}_{0.1}$ while maintaining strong returns on most Stochastic-D4RL tasks. Code: this https URL

**arXiv ID:** 2510.02695
</details>

<details>
<summary><strong>Ergodic Risk Measures: Towards a Risk-Aware Foundation for Continual Reinforcement Learning</strong> - Juan Sebastian Rojas, Chi-Guhn Lee - [[pdf]](https://arxiv.org/pdf/2510.02945)</summary>

**Abstract:** Continual reinforcement learning (continual RL) seeks to formalize the notions of lifelong learning and endless adaptation in RL. In particular, the aim of continual RL is to develop RL agents that can maintain a careful balance between retaining useful information and adapting to new situations. To date, continual RL has been explored almost exclusively through the lens of risk-neutral decision-making, in which the agent aims to optimize the expected (or mean) long-run performance. In this work, we present the first formal theoretical treatment of continual RL through the lens of risk-aware decision-making, in which the agent aims to optimize a reward-based measure of long-run performance beyond the mean. In particular, we show that the classical theory of risk measures, widely used as a theoretical foundation in non-continual risk-aware RL, is, in its current form, incompatible with the continual setting. Then, building on this insight, we extend risk measure theory into the continual setting by introducing a new class of ergodic risk measures that are compatible with continual learning. Finally, we provide a case study of risk-aware continual learning, along with empirical results, which show the intuitive appeal and theoretical soundness of ergodic risk measures.

**arXiv ID:** 2510.02945
</details>

<details>
<summary><strong>Comparative Analysis of Parameterized Action Actor-Critic Reinforcement Learning Algorithms for Web Search Match Plan Generation</strong> - Ubayd Bapoo, Clement N Nyirenda - [[pdf]](https://arxiv.org/pdf/2510.03064)</summary>

**Abstract:** This study evaluates the performance of Soft Actor Critic (SAC), Greedy Actor Critic (GAC), and Truncated Quantile Critics (TQC) in high-dimensional decision-making tasks using fully observable environments. The focus is on parametrized action (PA) spaces, eliminating the need for recurrent networks, with benchmarks Platform-v0 and Goal-v0 testing discrete actions linked to continuous action-parameter spaces. Hyperparameter optimization was performed with Microsoft NNI, ensuring reproducibility by modifying the codebase for GAC and TQC. Results show that Parameterized Action Greedy Actor-Critic (PAGAC) outperformed other algorithms, achieving the fastest training times and highest returns across benchmarks, completing 5,000 episodes in 41:24 for the Platform game and 24:04 for the Robot Soccer Goal game. Its speed and stability provide clear advantages in complex action spaces. Compared to PASAC and PATQC, PAGAC demonstrated superior efficiency and reliability, making it ideal for tasks requiring rapid convergence and robust performance. Future work could explore hybrid strategies combining entropy-regularization with truncation-based methods to enhance stability and expand investigations into generalizability.

**arXiv ID:** 2510.03064
</details>

<details>
<summary><strong>A Unified Deep Reinforcement Learning Approach for Close Enough Traveling Salesman Problem</strong> - Mingfeng Fan, Jiaqi Cheng, Yaoxin Wu, Yifeng Zhang, Yibin Yang, Guohua Wu, Guillaume Sartoretti - [[pdf]](https://arxiv.org/pdf/2510.03065)</summary>

**Abstract:** In recent years, deep reinforcement learning (DRL) has gained traction for solving the NP-hard traveling salesman problem (TSP). However, limited attention has been given to the close-enough TSP (CETSP), primarily due to the challenge introduced by its neighborhood-based visitation criterion, wherein a node is considered visited if the agent enters a compact neighborhood around it. In this work, we formulate a Markov decision process (MDP) for CETSP using a discretization scheme and propose a novel unified dual-decoder DRL (UD3RL) framework that separates decision-making into node selection and waypoint determination. Specifically, an adapted encoder is employed for effective feature extraction, followed by a node-decoder and a loc-decoder to handle the two sub-tasks, respectively. A k-nearest neighbors subgraph interaction strategy is further introduced to enhance spatial reasoning during location decoding. Furthermore, we customize the REINFORCE algorithm to train UD3RL as a unified model capable of generalizing across different problem sizes and varying neighborhood radius types (i.e., constant and random radii). Experimental results show that UD3RL outperforms conventional methods in both solution quality and runtime, while exhibiting strong generalization across problem scales, spatial distributions, and radius ranges, as well as robustness to dynamic environments.

**arXiv ID:** 2510.03065
</details>

<details>
<summary><strong>Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning</strong> - Ram Ramrakhya, Matthew Chang, Xavier Puig, Ruta Desai, Zsolt Kira, Roozbeh Mottaghi - [[pdf]](https://arxiv.org/pdf/2504.00907)</summary>

**Abstract:** Embodied agents operating in household environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent is tasked with a single or multi-object rearrangement task using an under-specified instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To address this challenge, we propose a novel approach that fine-tunes multi-modal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines including GPT-4o as well as supervised fine-tuned MLLMs on our task. Our results show that our RL-finetuned MLLM outperforms all baselines by a significant margin (10.4-16.5%), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL.

**arXiv ID:** 2504.00907
</details>

<details>
<summary><strong>L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning</strong> - Pranjal Aggarwal, Sean Welleck - [[pdf]](https://arxiv.org/pdf/2503.04697)</summary>

**Abstract:** Reasoning language models have shown an uncanny ability to improve performance at test-time by ``thinking longer''-that is, by generating longer chain-of-thought sequences and hence using more compute. However, the length of their chain-of-thought reasoning is not controllable, making it impossible to allocate test-time compute to achieve a desired level of performance. We introduce Length Controlled Policy Optimization (LCPO), a simple reinforcement learning method that optimizes for accuracy and adherence to user-specified length constraints. We use LCPO to train L1, a reasoning language model that produces outputs satisfying a length constraint given in its prompt. L1's length control allows for smoothly trading off computational cost and accuracy on a wide range of tasks, and outperforms the state-of-the-art S1 method for length control. Furthermore, we uncover an unexpected short chain-of-thought capability in models trained with LCPO. Specifically, using LCPO we derive Short Reasoning Models (SRMs), that exhibit similar reasoning patterns as full-length reasoning models, but can generate CoT lengths comparable to non-reasoning models. They demonstrate significant performance gains, for instance, our 1.5B L1 model surpasses GPT-4o at equal reasoning lengths. Overall, LCPO enables precise control over reasoning length, allowing for fine-grained allocation of test-time compute and accuracy. We release code and models at this https URL

**arXiv ID:** 2503.04697
</details>

<details>
<summary><strong>Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains</strong> - Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, Sean Hendryx - [[pdf]](https://arxiv.org/pdf/2507.17746)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for complex reasoning tasks with clear correctness signals such as math and coding. However, extending it to real-world reasoning tasks is challenging, as evaluation depends on nuanced, multi-criteria judgments rather than binary correctness. Instance-specific rubrics have recently been used in evaluation benchmarks to capture such judgments, but their potential as reward signals for on-policy post-training remains underexplored. We introduce $\textbf{Rubrics as Rewards}$ (RaR), an on-policy reinforcement learning method that extends RLVR beyond verifiable domains by using rubric-based feedback. Across both medical and science domains, we evaluate multiple strategies for aggregating rubric feedback into rewards. The best RaR variant achieves relative improvements of up to $31\%$ on HealthBench and $7\%$ on GPQA-Diamond over popular LLM-as-judge baselines that rely on direct Likert-based rewards. These results demonstrate that RaR-trained policies adapt well to diverse evaluation formats, performing strongly on both rubric-based and multiple-choice tasks. Moreover, we find that using rubrics as structured reward signals yields better alignment for smaller judges and reduces performance variance across judge scales.

**arXiv ID:** 2507.17746
</details>

<details>
<summary><strong>Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward</strong> - Guanhua Huang, Tingqiang Xu, Mingze Wang, Qi Yi, Xue Gong, Siheng Li, Ruibin Xiong, Kejiao Li, Yuhao Jiang, Bo Zhou - [[pdf]](https://arxiv.org/pdf/2510.03222)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has propelled Large Language Models in complex reasoning, yet its scalability is often hindered by a training bottleneck where performance plateaus as policy entropy collapses, signaling a loss of exploration. Previous methods typically address this by maintaining high policy entropy, yet the precise mechanisms that govern meaningful exploration have remained underexplored. Our analysis suggests that an unselective focus on entropy risks amplifying irrelevant tokens and destabilizing training. This paper investigates the exploration dynamics within RLVR and identifies a key issue: the gradual elimination of valuable low-probability exploratory tokens, which we term \textbf{\textit{reasoning sparks}}. We find that while abundant in pre-trained models, these sparks are systematically extinguished during RLVR due to over-penalization, leading to a degeneracy in exploration. To address this, we introduce Low-probability Regularization (Lp-Reg). Its core mechanism regularizes the policy towards a heuristic proxy distribution. This proxy is constructed by filtering out presumed noise tokens and re-normalizing the distribution over the remaining candidates. The result is a less-noisy proxy where the probability of \textit{reasoning sparks} is amplified, which then serves as a soft regularization target to shield these valuable tokens from elimination via KL divergence. Experiments show that Lp-Reg enables stable on-policy training for around 1,000 steps, a regime where baseline entropy-control methods collapse. This sustained exploration leads to state-of-the-art performance, achieving a $60.17\%$ average accuracy on five math benchmarks, an improvement of $2.66\%$ over prior methods. Code is available at this https URL.

**arXiv ID:** 2510.03222
</details>

<details>
<summary><strong>Improved Robustness of Deep Reinforcement Learning for Control of Time-Varying Systems by Bounded Extremum Seeking</strong> - Shaifalee Saxena, Alan Williams, Rafael Fierro, Alexander Scheinker - [[pdf]](https://arxiv.org/pdf/2510.02490)</summary>

**Abstract:** In this paper, we study the use of robust model independent bounded extremum seeking (ES) feedback control to improve the robustness of deep reinforcement learning (DRL) controllers for a class of nonlinear time-varying systems. DRL has the potential to learn from large datasets to quickly control or optimize the outputs of many-parameter systems, but its performance degrades catastrophically when the system model changes rapidly over time. Bounded ES can handle time-varying systems with unknown control directions, but its convergence speed slows down as the number of tuned parameters increases and, like all local adaptive methods, it can get stuck in local minima. We demonstrate that together, DRL and bounded ES result in a hybrid controller whose performance exceeds the sum of its parts with DRL taking advantage of historical data to learn how to quickly control a many-parameter system to a desired setpoint while bounded ES ensures its robustness to time variations. We present a numerical study of a general time-varying system and a combined ES-DRL controller for automatic tuning of the Low Energy Beam Transport section at the Los Alamos Neutron Science Center linear particle accelerator.

**arXiv ID:** 2510.02490
</details>

<details>
<summary><strong>Use the Online Network If You Can: Towards Fast and Stable Reinforcement Learning</strong> - Ahmed Hendawy, Henrik Metternich, Théo Vincent, Mahdi Kallel, Jan Peters, Carlo D'Eramo - [[pdf]](https://arxiv.org/pdf/2510.02590)</summary>

**Abstract:** The use of target networks is a popular approach for estimating value functions in deep Reinforcement Learning (RL). While effective, the target network remains a compromise solution that preserves stability at the cost of slowly moving targets, thus delaying learning. Conversely, using the online network as a bootstrapped target is intuitively appealing, albeit well-known to lead to unstable learning. In this work, we aim to obtain the best out of both worlds by introducing a novel update rule that computes the target using the MINimum estimate between the Target and Online network, giving rise to our method, MINTO. Through this simple, yet effective modification, we show that MINTO enables faster and stable value function learning, by mitigating the potential overestimation bias of using the online network for bootstrapping. Notably, MINTO can be seamlessly integrated into a wide range of value-based and actor-critic algorithms with a negligible cost. We evaluate MINTO extensively across diverse benchmarks, spanning online and offline RL, as well as discrete and continuous action spaces. Across all benchmarks, MINTO consistently improves performance, demonstrating its broad applicability and effectiveness.

**arXiv ID:** 2510.02590
</details>

<details>
<summary><strong>RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning</strong> - Aleksei Arzhantsev, Otmane Sakhi, Flavian Vasile - [[pdf]](https://arxiv.org/pdf/2510.02892)</summary>

**Abstract:** Reinforcement learning (RL) is central to improving reasoning in large language models (LLMs) but typically requires ground-truth rewards. Test-Time Reinforcement Learning (TTRL) removes this need by using majority-vote rewards, but relies on heavy online RL and incurs substantial computational cost. We propose RoiRL: Reasoning with offline iterative Reinforcement Learning, a family of lightweight offline learning alternatives that can target the same regularized optimal policies. Unlike TTRL, RoiRL eliminates the need to maintain a reference model and instead optimizes weighted log-likelihood objectives, enabling stable training with significantly lower memory and compute requirements. Experimental results show that RoiRL trains to 2.5x faster and consistently outperforms TTRL on reasoning benchmarks, establishing a scalable path to self-improving LLMs without labels.

**arXiv ID:** 2510.02892
</details>

<details>
<summary><strong>Distributional Inverse Reinforcement Learning</strong> - Feiyang Wu, Ye Zhao, Anqi Wu - [[pdf]](https://arxiv.org/pdf/2510.03013)</summary>

**Abstract:** We propose a distributional framework for offline Inverse Reinforcement Learning (IRL) that jointly models uncertainty over reward functions and full distributions of returns. Unlike conventional IRL approaches that recover a deterministic reward estimate or match only expected returns, our method captures richer structure in expert behavior, particularly in learning the reward distribution, by minimizing first-order stochastic dominance (FSD) violations and thus integrating distortion risk measures (DRMs) into policy learning, enabling the recovery of both reward distributions and distribution-aware policies. This formulation is well-suited for behavior analysis and risk-aware imitation learning. Empirical results on synthetic benchmarks, real-world neurobehavioral data, and MuJoCo control tasks demonstrate that our method recovers expressive reward representations and achieves state-of-the-art imitation performance.

**arXiv ID:** 2510.03013
</details>

<details>
<summary><strong>Q-Learning with Shift-Aware Upper Confidence Bound in Non-Stationary Reinforcement Learning</strong> - Ha Manh Bui, Felix Parker, Kimia Ghobadi, Anqi Liu - [[pdf]](https://arxiv.org/pdf/2510.03181)</summary>

**Abstract:** We study the Non-Stationary Reinforcement Learning (RL) under distribution shifts in both finite-horizon episodic and infinite-horizon discounted Markov Decision Processes (MDPs). In the finite-horizon case, the transition functions may suddenly change at a particular episode. In the infinite-horizon setting, such changes can occur at an arbitrary time step during the agent's interaction with the environment. While the Q-learning Upper Confidence Bound algorithm (QUCB) can discover a proper policy during learning, due to the distribution shifts, this policy can exploit sub-optimal rewards after the shift happens. To address this issue, we propose Density-QUCB (DQUCB), a shift-aware Q-learning~UCB algorithm, which uses a transition density function to detect distribution shifts, then leverages its likelihood to enhance the uncertainty estimation quality of Q-learning~UCB, resulting in a balance between exploration and exploitation. Theoretically, we prove that our oracle DQUCB achieves a better regret guarantee than QUCB. Empirically, our DQUCB enjoys the computational efficiency of model-free RL and outperforms QUCB baselines by having a lower regret across RL tasks, as well as a real-world COVID-19 patient hospital allocation task using a Deep-Q-learning architecture.

**arXiv ID:** 2510.03181
</details>

<details>
<summary><strong>To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning</strong> - Yuda Song, Dhruv Rohatgi, Aarti Singh, J. Andrew Bagnell - [[pdf]](https://arxiv.org/pdf/2510.03207)</summary>

**Abstract:** Partial observability is a notorious challenge in reinforcement learning (RL), due to the need to learn complex, history-dependent policies. Recent empirical successes have used privileged expert distillation--which leverages availability of latent state information during training (e.g., from a simulator) to learn and imitate the optimal latent, Markovian policy--to disentangle the task of "learning to see" from "learning to act". While expert distillation is more computationally efficient than RL without latent state information, it also has well-documented failure modes. In this paper--through a simple but instructive theoretical model called the perturbed Block MDP, and controlled experiments on challenging simulated locomotion tasks--we investigate the algorithmic trade-off between privileged expert distillation and standard RL without privileged information. Our main findings are: (1) The trade-off empirically hinges on the stochasticity of the latent dynamics, as theoretically predicted by contrasting approximate decodability with belief contraction in the perturbed Block MDP; and (2) The optimal latent policy is not always the best latent policy to distill. Our results suggest new guidelines for effectively exploiting privileged information, potentially advancing the efficiency of policy learning across many practical partially observable domains.

**arXiv ID:** 2510.03207
</details>

<details>
<summary><strong>A Nearly Optimal and Low-Switching Algorithm for Reinforcement Learning with General Function Approximation</strong> - Heyang Zhao, Jiafan He, Quanquan Gu - [[pdf]](https://arxiv.org/pdf/2311.15238)</summary>

**Abstract:** The exploration-exploitation dilemma has been a central challenge in reinforcement learning (RL) with complex model classes. In this paper, we propose a new algorithm, Monotonic Q-Learning with Upper Confidence Bound (MQL-UCB) for RL with general function approximation. Our key algorithmic design includes (1) a general deterministic policy-switching strategy that achieves low switching cost, (2) a monotonic value function structure with carefully controlled function class complexity, and (3) a variance-weighted regression scheme that exploits historical trajectories with high data efficiency. MQL-UCB achieves minimax optimal regret of $\tilde{O}(d\sqrt{HK})$ when $K$ is sufficiently large and near-optimal policy switching cost of $\tilde{O}(dH)$, with $d$ being the eluder dimension of the function class, $H$ being the planning horizon, and $K$ being the number of episodes.
Our work sheds light on designing provably sample-efficient and deployment-efficient Q-learning with nonlinear function approximation.

**arXiv ID:** 2311.15238
</details>

<details>
<summary><strong>Graph-Reward-SQL: Execution-Free Reinforcement Learning for Text-to-SQL via Graph Matching and Stepwise Reward</strong> - Han Weng, Puzhen Wu, Longjie Cui, Yi Zhan, Boyi Liu, Yuanfeng Song, Dun Zeng, Yingxiang Yang, Qianru Zhang, Dong Huang, Xiaoming Yin, Yang Sun, Xing Chen - [[pdf]](https://arxiv.org/pdf/2505.12380)</summary>

**Abstract:** Reinforcement learning (RL) has been widely adopted to enhance the performance of large language models (LLMs) on Text-to-SQL tasks. However, existing methods often rely on execution-based or LLM-based Bradley-Terry reward models. The former suffers from high execution latency caused by repeated database calls, whereas the latter imposes substantial GPU memory overhead, both of which significantly hinder the efficiency and scalability of RL pipelines. To this end, we propose a novel reward model framework for RL-based Text-to-SQL named Graph-Reward-SQL, which employs the GMNScore outcome reward model. We leverage SQL graph representations to provide accurate reward signals while significantly reducing time cost and GPU memory usage. Building on this foundation, we further introduce StepRTM, a stepwise reward model that provides intermediate supervision over Common Table Expression (CTE) subqueries. This encourages both functional correctness and readability of SQL. Extensive comparative and ablation experiments on standard benchmarks, including Spider and BIRD, demonstrate that our method consistently outperforms existing reward models.

**arXiv ID:** 2505.12380
</details>

<details>
<summary><strong>Risk-Sensitive Agent Compositions</strong> - Guruprerana Shabadi, Rajeev Alur - [[pdf]](https://arxiv.org/pdf/2506.04632)</summary>

**Abstract:** From software development to robot control, modern agentic systems decompose complex objectives into a sequence of subtasks and choose a set of specialized AI agents to complete them. We formalize agentic workflows as directed acyclic graphs, called agent graphs, where edges represent AI agents and paths correspond to feasible compositions of agents. Real-world deployment requires selecting agent compositions that not only maximize task success but also minimize violations of safety, fairness, and privacy requirements which demands a careful analysis of the low-probability (tail) behaviors of compositions of agents. In this work, we consider risk minimization over the set of feasible agent compositions and seek to minimize the value-at-risk of the loss distribution of the agent composition where the loss quantifies violations of these requirements. We introduce an efficient algorithm which traverses the agent graph and finds a near-optimal composition of agents. It uses a dynamic programming approach to approximate the value-at-risk of agent compositions by exploiting a union bound. Furthermore, we prove that the approximation is near-optimal asymptotically for a broad class of practical loss functions. To evaluate our framework, we consider a suite of video game-like control benchmarks that require composing several agents trained with reinforcement learning and demonstrate our algorithm's effectiveness in approximating the value-at-risk and identifying the optimal agent composition.

**arXiv ID:** 2506.04632
</details>

<details>
<summary><strong>Towards Provable Emergence of In-Context Reinforcement Learning</strong> - Jiuqi Wang, Rohan Chandra, Shangtong Zhang - [[pdf]](https://arxiv.org/pdf/2509.18389)</summary>

**Abstract:** Typically, a modern reinforcement learning (RL) agent solves a task by updating its neural network parameters to adapt its policy to the task. Recently, it has been observed that some RL agents can solve a wide range of new out-of-distribution tasks without parameter updates after pretraining on some task distribution. When evaluated in a new task, instead of making parameter updates, the pretrained agent conditions its policy on additional input called the context, e.g., the agent's interaction history in the new task. The agent's performance increases as the information in the context increases, with the agent's parameters fixed. This phenomenon is typically called in-context RL (ICRL). The pretrained parameters of the agent network enable the remarkable ICRL phenomenon. However, many ICRL works perform the pretraining with standard RL algorithms. This raises the central question this paper aims to address: Why can the RL pretraining algorithm generate network parameters that enable ICRL? We hypothesize that the parameters capable of ICRL are minimizers of the pretraining loss. This work provides initial support for this hypothesis through a case study. In particular, we prove that when a Transformer is pretrained for policy evaluation, one of the global minimizers of the pretraining loss can enable in-context temporal difference learning.

**arXiv ID:** 2509.18389
</details>

<details>
<summary><strong>Octax: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX</strong> - Waris Radji, Thomas Michel, Hector Piteau - [[pdf]](https://arxiv.org/pdf/2510.01764)</summary>

**Abstract:** Reinforcement learning (RL) research requires diverse, challenging environments that are both tractable and scalable. While modern video games may offer rich dynamics, they are computationally expensive and poorly suited for large-scale experimentation due to their CPU-bound execution. We introduce Octax, a high-performance suite of classic arcade game environments implemented in JAX, based on CHIP-8 emulation, a predecessor to Atari, which is widely adopted as a benchmark in RL research. Octax provides the JAX community with a long-awaited end-to-end GPU alternative to the Atari benchmark, offering image-based environments, spanning puzzle, action, and strategy genres, all executable at massive scale on modern GPUs. Our JAX-based implementation achieves orders-of-magnitude speedups over traditional CPU emulators while maintaining perfect fidelity to the original game mechanics. We demonstrate Octax's capabilities by training RL agents across multiple games, showing significant improvements in training speed and scalability compared to existing solutions. The environment's modular design enables researchers to easily extend the suite with new games or generate novel environments using large language models, making it an ideal platform for large-scale RL experimentation.

**arXiv ID:** 2510.01764
</details>

<details>
<summary><strong>Post Reinforcement Learning Inference</strong> - Vasilis Syrgkanis, Ruohan Zhan - [[pdf]](https://arxiv.org/pdf/2302.08854)</summary>

**Abstract:** We study estimation and inference using data collected by reinforcement learning (RL) algorithms. These algorithms adaptively experiment by interacting with individual units over multiple stages, updating their strategies based on past outcomes. Our goal is to evaluate a counterfactual policy after data collection and estimate structural parameters, such as dynamic treatment effects, that support credit assignment and quantify the impact of early actions on final outcomes. These parameters can often be defined as solutions to moment equations, motivating moment-based estimation methods developed for static data. In RL settings, however, data are often collected adaptively under nonstationary behavior policies. As a result, standard estimators fail to achieve asymptotic normality due to time-varying variance. We propose a weighted generalized method of moments (GMM) approach that uses adaptive weights to stabilize this variance. We characterize weighting schemes that ensure consistency and asymptotic normality of the weighted GMM estimators, enabling valid hypothesis testing and uniform confidence region construction. Key applications include dynamic treatment effect estimation and dynamic off-policy evaluation.

**arXiv ID:** 2302.08854
</details>

<details>
<summary><strong>Active Alignments of Lens Systems with Reinforcement Learning</strong> - Matthias Burkhardt, Tobias Schmähling, Pascal Stegmann, Michael Layh, Tobias Windisch - [[pdf]](https://arxiv.org/pdf/2503.02075)</summary>

**Abstract:** Aligning a lens system relative to an imager is a critical challenge in camera manufacturing. While optimal alignment can be mathematically computed under ideal conditions, real-world deviations caused by manufacturing tolerances often render this approach impractical. Measuring these tolerances can be costly or even infeasible, and neglecting them may result in suboptimal alignments. We propose a reinforcement learning (RL) approach that learns exclusively in the pixel space of the sensor output, eliminating the need to develop expert-designed alignment concepts. We conduct an extensive benchmark study and show that our approach surpasses other methods in speed, precision, and robustness. We further introduce relign, a realistic, freely explorable, open-source simulation utilizing physically based rendering that models optical systems with non-deterministic manufacturing tolerances and noise in robotic alignment movement. It provides an interface to popular machine learning frameworks, enabling seamless experimentation and development. Our work highlights the potential of RL in a manufacturing environment to enhance efficiency of optical alignments while minimizing the need for manual intervention.

**arXiv ID:** 2503.02075
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
