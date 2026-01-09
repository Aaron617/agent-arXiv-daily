# Agent arXiv Daily

**Last Updated:** 2026-01-09 02:27:13

**Total Papers:** 75

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
<summary><strong>Can AI Chatbots Provide Coaching in Engineering? Beyond Information Processing Toward Mastery</strong> - Junaid Qadir, Muhammad Adil Attique, Saleha Shoaib, Syed Ibrahim Ghaznavi - [[pdf]](https://arxiv.org/pdf/2601.03693)</summary>

**Abstract:** Engineering education faces a double disruption: traditional apprenticeship models that cultivated judgment and tacit skill are eroding, just as generative AI emerges as an informal coaching partner. This convergence rekindles long-standing questions in the philosophy of AI and cognition about the limits of computation, the nature of embodied rationality, and the distinction between information processing and wisdom. Building on this rich intellectual tradition, this paper examines whether AI chatbots can provide coaching that fosters mastery rather than merely delivering information. We synthesize critical perspectives from decades of scholarship on expertise, tacit knowledge, and human-machine interaction, situating them within the context of contemporary AI-driven education. Empirically, we report findings from a mixed-methods study (N = 75 students, N = 7 faculty) exploring the use of a coaching chatbot in engineering education. Results reveal a consistent boundary: participants accept AI for technical problem solving (convergent tasks; M = 3.84 on a 1-5 Likert scale) but remain skeptical of its capacity for moral, emotional, and contextual judgment (divergent tasks). Faculty express stronger concerns over risk (M = 4.71 vs. M = 4.14, p = 0.003), and privacy emerges as a key requirement, with 64-71 percent of participants demanding strict confidentiality. Our findings suggest that while generative AI can democratize access to cognitive and procedural support, it cannot replicate the embodied, value-laden dimensions of human mentorship. We propose a multiplex coaching framework that integrates human wisdom within expert-in-the-loop models, preserving the depth of apprenticeship while leveraging AI scalability to enrich the next generation of engineering education.

**arXiv ID:** 2601.03693
</details>

<details>
<summary><strong>From Laboratory to Real-World Applications: Benchmarking Agentic Code Reasoning at the Repository Level</strong> - Jia Li, Yuxin Su, Michael R. Lyu - [[pdf]](https://arxiv.org/pdf/2601.03731)</summary>

**Abstract:** As large language models (LLMs) evolve into autonomous agents, evaluating repository-level reasoning, the ability to maintain logical consistency across massive, real-world, interdependent file systems, has become critical. Current benchmarks typically fluctuate between isolated code snippets and black-box evaluations. We present RepoReason, a white-box diagnostic benchmark centered on abductive assertion verification. To eliminate memorization while preserving authentic logical depth, we implement an execution-driven mutation framework that utilizes the environment as a semantic oracle to regenerate ground-truth states. Furthermore, we establish a fine-grained diagnostic system using dynamic program slicing, quantifying reasoning via three orthogonal metrics: $ESV$ (reading load), $MCL$ (simulation depth), and $DFI$ (integration width). Comprehensive evaluations of frontier models (e.g., Claude-4.5-Sonnet, DeepSeek-v3.1-Terminus) reveal a prevalent aggregation deficit, where integration width serves as the primary cognitive bottleneck. Our findings provide granular white-box insights for optimizing the next generation of agentic software engineering.

**arXiv ID:** 2601.03731
</details>

<details>
<summary><strong>Embedding Autonomous Agents in Resource-Constrained Robotic Platforms</strong> - Negar Halakou, Juan F. Gutierrez, Ye Sun, Han Jiang, Xueming Wu, Yilun Song, Andres Gomez - [[pdf]](https://arxiv.org/pdf/2601.04191)</summary>

**Abstract:** Many embedded devices operate under resource constraints and in dynamic environments, requiring local decision-making capabilities. Enabling devices to make independent decisions in such environments can improve the responsiveness of the system and reduce the dependence on constant external control. In this work, we integrate an autonomous agent, programmed using AgentSpeak, with a small two-wheeled robot that explores a maze using its own decision-making and sensor data. Experimental results show that the agent successfully solved the maze in 59 seconds using 287 reasoning cycles, with decision phases taking less than one millisecond. These results indicate that the reasoning process is efficient enough for real-time execution on resource-constrained hardware. This integration demonstrates how high-level agent-based control can be applied to resource-constrained embedded systems for autonomous operation.

**arXiv ID:** 2601.04191
</details>

<details>
<summary><strong>Jenius Agent: Towards Experience-Driven Accuracy Optimization in Real-World Scenarios</strong> - Defei Xia, Bingfeng Pi, Shenbin Zhang, Song Hua, Yunfei Wei, Lei Zuo - [[pdf]](https://arxiv.org/pdf/2601.01857)</summary>

**Abstract:** As agent systems powered by large language models (LLMs) advance, improving the task performance of an autonomous agent, especially in context understanding, tool usage, and response generation, has become increasingly critical. Although prior studies have advanced the overall design of LLM-based agents, systematic optimization of their internal reasoning and tool-use pipelines remains underexplored. This paper introduces an agent framework grounded in real-world practical experience, with three key innovations: (1) an adaptive prompt generation strategy that aligns with the agent's state and task goals to improve reliability and robustness; (2) a context-aware tool orchestration module that performs tool categorization, semantic retrieval, and adaptive invocation based on user intent and context; and (3) a layered memory mechanism that integrates session memory, task history, and external summaries to improve relevance and efficiency through dynamic summarization and compression. An end-to-end framework named Jenius-Agent has been integrated with three key optimizations, including tools based on the Model Context Protocol (MCP), file input/output (I/O), and execution feedback. The experiments show a 20 percent improvement in task accuracy, along with a reduced token cost, response latency, and invocation failures. The framework is already deployed in Jenius (this https URL), providing a lightweight and scalable solution for robust, protocol-compatible autonomous agents.

**arXiv ID:** 2601.01857
</details>

<details>
<summary><strong>FIRE-VLM: A Vision-Language-Driven Reinforcement Learning Framework for UAV Wildfire Tracking in a Physics-Grounded Fire Digital Twin</strong> - Chris Webb, Mobin Habibpour, Mayamin Hamid Raha, Ali Reza Tavakkoli, Janice Coen, Fatemeh Afghah - [[pdf]](https://arxiv.org/pdf/2601.03449)</summary>

**Abstract:** Wildfire monitoring demands autonomous systems capable of reasoning under extreme visual degradation, rapidly evolving physical dynamics, and scarce real-world training data. Existing UAV navigation approaches rely on simplified simulators and supervised perception pipelines, and lack embodied agents interacting with physically realistic fire environments. We introduce FIRE-VLM, the first end-to-end vision-language model (VLM) guided reinforcement learning (RL) framework trained entirely within a high-fidelity, physics-grounded wildfire digital twin. Built from USGS Digital Elevation Model (DEM) terrain, LANDFIRE fuel inventories, and semi-physical fire-spread solvers, this twin captures terrain-induced runs, wind-driven acceleration, smoke plume occlusion, and dynamic fuel consumption. Within this environment, a PPO agent with dual-view UAV sensing is guided by a CLIP-style VLM. Wildfire-specific semantic alignment scores, derived from a single prompt describing active fire and smoke plumes, are integrated as potential-based reward shaping signals. Our contributions are: (1) a GIS-to-simulation pipeline for constructing wildfire digital twins; (2) a VLM-guided RL agent for UAV firefront tracking; and (3) a wildfire-aware reward design that combines physical terms with VLM semantics. Across five digital-twin evaluation tasks, our VLM-guided policy reduces time-to-detection by up to 6 times, increases time-in-FOV, and is, to our knowledge, the first RL-based UAV wildfire monitoring system demonstrated in kilometer-scale, physics-grounded digital-twin fires.

**arXiv ID:** 2601.03449
</details>

<details>
<summary><strong>Device-Native Autonomous Agents for Privacy-Preserving Negotiations</strong> - Joyjit Roy - [[pdf]](https://arxiv.org/pdf/2601.00911)</summary>

**Abstract:** Automated negotiations in insurance and business-to-business (B2B) commerce encounter substantial challenges. Current systems force a trade-off between convenience and privacy by routing sensitive financial data through centralized servers, increasing security risks, and diminishing user trust. This study introduces a device-native autonomous Artificial Intelligence (AI) agent system for privacy-preserving negotiations. The proposed system operates exclusively on user hardware, enabling real-time bargaining while maintaining sensitive constraints locally. It integrates zero-knowledge proofs to ensure privacy and employs distilled world models to support advanced on-device reasoning. The architecture incorporates six technical components within an agentic AI workflow. Agents autonomously plan negotiation strategies, conduct secure multi-party bargaining, and generate cryptographic audit trails without exposing user data to external servers. The system is evaluated in insurance and B2B procurement scenarios across diverse device configurations. Results show an average success rate of 87%, a 2.4x latency improvement over cloud baselines, and strong privacy preservation through zero-knowledge proofs. User studies show 27% higher trust scores when decision trails are available. These findings establish a foundation for trustworthy autonomous agents in privacy-sensitive financial domains.

**arXiv ID:** 2601.00911
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms</strong> - Michael C. Darling, Alan H. Hesu, Michael A. Mardikes, Brian C. McGuigan, Reed M. Milewicz - [[pdf]](https://arxiv.org/pdf/2601.03470)</summary>

**Abstract:** We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study.

**arXiv ID:** 2601.03470
</details>

<details>
<summary><strong>Deploy-Master: Automating the Deployment of 50,000+ Agent-Ready Scientific Tools in One Day</strong> - Yi Wang, Zhenting Huang, Zhaohan Ding, Ruoxue Liao, Yuan Huang, Xinzijian Liu, Jiajun Xie, Siheng Chen, Linfeng Zhang - [[pdf]](https://arxiv.org/pdf/2601.03513)</summary>

**Abstract:** Open-source scientific software is abundant, yet most tools remain difficult to compile, configure, and reuse, sustaining a small-workshop mode of scientific computing. This deployment bottleneck limits reproducibility, large-scale evaluation, and the practical integration of scientific tools into modern AI-for-Science (AI4S) and agentic workflows.
We present Deploy-Master, a one-stop agentic workflow for large-scale tool discovery, build specification inference, execution-based validation, and publication. Guided by a taxonomy spanning 90+ scientific and engineering domains, our discovery stage starts from a recall-oriented pool of over 500,000 public repositories and progressively filters it to 52,550 executable tool candidates under license- and quality-aware criteria. Deploy-Master transforms heterogeneous open-source repositories into runnable, containerized capabilities grounded in execution rather than documentation claims. In a single day, we performed 52,550 build attempts and constructed reproducible runtime environments for 50,112 scientific tools. Each successful tool is validated by a minimal executable command and registered in SciencePedia for search and reuse, enabling direct human use and optional agent-based invocation.
Beyond delivering runnable tools, we report a deployment trace at the scale of 50,000 tools, characterizing throughput, cost profiles, failure surfaces, and specification uncertainty that become visible only at scale. These results explain why scientific software remains difficult to operationalize and motivate shared, observable execution substrates as a foundation for scalable AI4S and agentic science.

**arXiv ID:** 2601.03513
</details>

<details>
<summary><strong>Wow, wo, val! A Comprehensive Embodied World Model Evaluation Turing Test</strong> - Chun-Kai Fan, Xiaowei Chi, Xiaozhu Ju, Hao Li, Yong Bao, Yu-Kai Wang, Lizhang Chen, Zhiyuan Jiang, Kuangzhi Ge, Ying Li, Weishi Mi, Qingpo Wuwu, Peidong Jia, Yulin Luo, Kevin Zhang, Zhiyuan Qin, Yong Dai, Sirui Han, Yike Guo, Shanghang Zhang, Jian Tang - [[pdf]](https://arxiv.org/pdf/2601.04137)</summary>

**Abstract:** As world models gain momentum in Embodied AI, an increasing number of works explore using video foundation models as predictive world models for downstream embodied tasks like 3D prediction or interactive generation. However, before exploring these downstream tasks, video foundation models still have two critical questions unanswered: (1) whether their generative generalization is sufficient to maintain perceptual fidelity in the eyes of human observers, and (2) whether they are robust enough to serve as a universal prior for real-world embodied agents. To provide a standardized framework for answering these questions, we introduce the Embodied Turing Test benchmark: WoW-World-Eval (Wow,wo,val). Building upon 609 robot manipulation data, Wow-wo-val examines five core abilities, including perception, planning, prediction, generalization, and execution. We propose a comprehensive evaluation protocol with 22 metrics to assess the models' generation ability, which achieves a high Pearson Correlation between the overall score and human preference (>0.93) and establishes a reliable foundation for the Human Turing Test. On Wow-wo-val, models achieve only 17.27 on long-horizon planning and at best 68.02 on physical consistency, indicating limited spatiotemporal consistency and physical reasoning. For the Inverse Dynamic Model Turing Test, we first use an IDM to evaluate the video foundation models' execution accuracy in the real world. However, most models collapse to $\approx$ 0% success, while WoW maintains a 40.74% success rate. These findings point to a noticeable gap between the generated videos and the real world, highlighting the urgency and necessity of benchmarking World Model in Embodied AI.

**arXiv ID:** 2601.04137
</details>

<details>
<summary><strong>ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering</strong> - Rachneet Kaur, Nishan Srishankar, Zhen Zeng, Sumitra Ganesh, Manuela Veloso - [[pdf]](https://arxiv.org/pdf/2510.04514)</summary>

**Abstract:** Recent multimodal LLMs have shown promise in chart-based visual question answering, but their performance declines sharply on unannotated charts-those requiring precise visual interpretation rather than relying on textual shortcuts. To address this, we introduce ChartAgent, a novel agentic framework that explicitly performs visual reasoning directly within the chart's spatial domain. Unlike textual chain-of-thought reasoning, ChartAgent iteratively decomposes queries into visual subtasks and actively manipulates and interacts with chart images through specialized actions such as drawing annotations, cropping regions (e.g., segmenting pie slices, isolating bars), and localizing axes, using a library of chart-specific vision tools to fulfill each subtask. This iterative reasoning process closely mirrors human cognitive strategies for chart comprehension. ChartAgent achieves state-of-the-art accuracy on the ChartBench and ChartX benchmarks, surpassing prior methods by up to 16.07% absolute gain overall and 17.31% on unannotated, numerically intensive queries. Furthermore, our analyses show that ChartAgent is (a) effective across diverse chart types, (b) achieves the highest scores across varying visual and reasoning complexity levels, and (c) serves as a plug-and-play framework that boosts performance across diverse underlying LLMs. Our work is among the first to demonstrate visually grounded reasoning for chart understanding using tool-augmented multimodal agents.

**arXiv ID:** 2510.04514
</details>

<details>
<summary><strong>DiVA: Fine-grained Factuality Verification with Agentic-Discriminative Verifier</strong> - Hui Huang, Muyun Yang, Yuki Arase - [[pdf]](https://arxiv.org/pdf/2601.03605)</summary>

**Abstract:** Despite the significant advancements of Large Language Models (LLMs), their factuality remains a critical challenge, fueling growing interest in factuality verification. Existing research on factuality verification primarily conducts binary judgments (e.g., correct or incorrect), which fails to distinguish varying degrees of error severity. This limits its utility for applications such as fine-grained evaluation and preference optimization. To bridge this gap, we propose the Agentic Discriminative Verifier (DiVA), a hybrid framework that synergizes the agentic search capabilities of generative models with the precise scoring aptitude of discriminative models. We also construct a new benchmark, FGVeriBench, as a robust testbed for fine-grained factuality verification. Experimental results on FGVeriBench demonstrate that our DiVA significantly outperforms existing methods on factuality verification for both general and multi-hop questions.

**arXiv ID:** 2601.03605
</details>

<details>
<summary><strong>Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning</strong> - Zheng Wu, Xingyu Lou, Xinbei Ma, Yansi Li, Weiwen Liu, Weinan Zhang, Jun Wang, Zhuosheng Zhang - [[pdf]](https://arxiv.org/pdf/2601.03641)</summary>

**Abstract:** Large Language Model (LLM)-based agents significantly extend the utility of LLMs by interacting with dynamic environments. However, enabling agents to continually learn new tasks without catastrophic forgetting remains a critical challenge, known as the stability-plasticity dilemma. In this work, we argue that this dilemma fundamentally arises from the failure to explicitly distinguish between common knowledge shared across tasks and conflicting knowledge introduced by task-specific interference. To address this, we propose Agent-Dice, a parameter fusion framework based on directional consensus evaluation. Concretely, Agent-Dice disentangles knowledge updates through a two-stage process: geometric consensus filtering to prune conflicting gradients, and curvature-based importance weighting to amplify shared semantics. We provide a rigorous theoretical analysis that establishes the validity of the proposed fusion scheme and offers insight into the origins of the stability-plasticity dilemma. Extensive experiments on GUI agents and tool-use agent domains demonstrate that Agent-Dice exhibits outstanding continual learning performance with minimal computational overhead and parameter updates. The codes are available at this https URL.

**arXiv ID:** 2601.03641
</details>

<details>
<summary><strong>FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis</strong> - Fengbin Zhu, Xiang Yao Ng, Ziyang Liu, Chang Liu, Xianwei Zeng, Chao Wang, Tianhui Tan, Xuan Yao, Pengyang Shao, Min Xu, Zixuan Wang, Jing Wang, Xin Lin, Junfeng Li, Jingxian Zhu, Yang Zhang, Wenjie Wang, Fuli Feng, Richang Hong, Huanbo Luan, Ke-Wei Huang, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2510.13936)</summary>

**Abstract:** Deep Research (DR) agents, powered by advanced Large Language Models (LLMs), have recently garnered increasing attention for their capability in conducting complex research tasks. However, existing literature lacks a rigorous and systematic evaluation of DR Agent's capabilities in critical research analysis. To address this gap, we first propose HisRubric, a novel evaluation framework with a hierarchical analytical structure and a fine-grained grading rubric for rigorously assessing DR agents' capabilities in corporate financial analysis. This framework mirrors the professional analyst's workflow, progressing from data recognition to metric calculation, and finally to strategic summarization and interpretation. Built on this framework, we construct a FinDeepResearch benchmark that comprises 64 listed companies from 8 financial markets across 4 languages, encompassing a total of 15,808 grading items. We further conduct extensive experiments on the FinDeepResearch using 16 representative methods, including 6 DR agents, 5 LLMs equipped with both deep reasoning and search capabilities, and 5 LLMs with deep reasoning capabilities only. The results reveal the strengths and limitations of these approaches across diverse capabilities, financial markets, and languages, offering valuable insights for future research and development. The benchmark and evaluation code is publicly available at this https URL.

**arXiv ID:** 2510.13936
</details>

<details>
<summary><strong>From Human Intention to Action Prediction: Intention-Driven End-to-End Autonomous Driving</strong> - Huan Zheng, Yucheng Zhou, Tianyi Yan, Jiayi Su, Hongjun Chen, Dubing Chen, Xingtai Gui, Wencheng Han, Runzhou Tao, Zhongying Qiu, Jianfei Yang, Jianbing Shen - [[pdf]](https://arxiv.org/pdf/2512.12302)</summary>

**Abstract:** While end-to-end autonomous driving has achieved remarkable progress in geometric control, current systems remain constrained by a command-following paradigm that relies on simple navigational instructions. Transitioning to genuinely intelligent agents requires the capability to interpret and fulfill high-level, abstract human intentions. However, this advancement is hindered by the lack of dedicated benchmarks and semantic-aware evaluation metrics. In this paper, we formally define the task of Intention-Driven End-to-End Autonomous Driving and present Intention-Drive, a comprehensive benchmark designed to bridge this gap. We construct a large-scale dataset featuring complex natural language intentions paired with high-fidelity sensor data. To overcome the limitations of conventional trajectory-based metrics, we introduce the Imagined Future Alignment (IFA), a novel evaluation protocol leveraging generative world models to assess the semantic fulfillment of human goals beyond mere geometric accuracy. Furthermore, we explore the solution space by proposing two distinct paradigms: an end-to-end vision-language planner and a hierarchical agent-based framework. The experiments reveal a critical dichotomy where existing models exhibit satisfactory driving stability but struggle significantly with intention fulfillment. Notably, the proposed frameworks demonstrate superior alignment with human intentions.

**arXiv ID:** 2512.12302
</details>

<details>
<summary><strong>The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective</strong> - Jiin Kim, Byeongjun Shin, Jinha Chung, Minsoo Rhu - [[pdf]](https://arxiv.org/pdf/2506.04301)</summary>

**Abstract:** Large-language-model (LLM)-based AI agents have recently showcased impressive versatility by employing dynamic reasoning, an adaptive, multi-step process that coordinates with external tools. This shift from static, single-turn inference to agentic, multi-turn workflows broadens task generalization and behavioral flexibility, but it also introduces serious concerns about system-level cost, efficiency, and sustainability. This paper presents the first comprehensive system-level analysis of AI agents, quantifying their resource usage, latency behavior, energy consumption, and datacenter-wide power consumption demands across diverse agent designs and test-time scaling strategies. We further characterize how AI agent design choices, such as few-shot prompting, reflection depth, and parallel reasoning, impact accuracy-cost tradeoffs. Our findings reveal that while agents improve accuracy with increased compute, they suffer from rapidly diminishing returns, widening latency variance, and unsustainable infrastructure costs. Through detailed evaluation of representative agents, we highlight the profound computational demands introduced by AI agent workflows, uncovering a looming sustainability crisis. These results call for a paradigm shift in agent design toward compute-efficient reasoning, balancing performance with deployability under real-world constraints.

**arXiv ID:** 2506.04301
</details>

<details>
<summary><strong>A Vision-Language-Action Model with Visual Prompt for OFF-Road Autonomous Driving</strong> - Liangdong Zhang, Yiming Nie, Haoyang Li, Fanjie Kong, Baobao Zhang, Shunxin Huang, Kai Fu, Chen Min, Liang Xiao - [[pdf]](https://arxiv.org/pdf/2601.03519)</summary>

**Abstract:** Efficient trajectory planning in off-road terrains presents a formidable challenge for autonomous vehicles, often necessitating complex multi-step pipelines. However, traditional approaches exhibit limited adaptability in dynamic environments. To address these limitations, this paper proposes OFF-EMMA, a novel end-to-end multimodal framework designed to overcome the deficiencies of insufficient spatial perception and unstable reasoning in visual-language-action (VLA) models for off-road autonomous driving scenarios. The framework explicitly annotates input images through the design of a visual prompt block and introduces a chain-of-thought with self-consistency (COT-SC) reasoning strategy to enhance the accuracy and robustness of trajectory planning. The visual prompt block utilizes semantic segmentation masks as visual prompts, enhancing the spatial understanding ability of pre-trained visual-language models for complex terrains. The COT- SC strategy effectively mitigates the error impact of outliers on planning performance through a multi-path reasoning mechanism. Experimental results on the RELLIS-3D off-road dataset demonstrate that OFF-EMMA significantly outperforms existing methods, reducing the average L2 error of the Qwen backbone model by 13.3% and decreasing the failure rate from 16.52% to 6.56%.

**arXiv ID:** 2601.03519
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts</strong> - Dhruv Trehan, Paras Chopra - [[pdf]](https://arxiv.org/pdf/2601.03315)</summary>

**Abstract:** We report a case study of four end-to-end attempts to autonomously generate ML research papers using a pipeline of six LLM agents mapped to stages of the scientific workflow. Of these four, three attempts failed during implementation or evaluation. One completed the pipeline and was accepted to Agents4Science 2025, an experimental inaugural venue that required AI systems as first authors, passing both human and multi-AI review. From these attempts, we document six recurring failure modes: bias toward training data defaults, implementation drift under execution pressure, memory and context degradation across long-horizon tasks, overexcitement that declares success despite obvious failures, insufficient domain intelligence, and weak scientific taste in experimental design. We conclude by discussing four design principles for more robust AI-scientist systems, implications for autonomous scientific discovery, and we release all prompts, artifacts, and outputs at this https URL

**arXiv ID:** 2601.03315
</details>

<details>
<summary><strong>Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents</strong> - Yuanchen Bei, Tianxin Wei, Xuying Ning, Yanjun Zhao, Zhining Liu, Xiao Lin, Yada Zhu, Hendrik Hamann, Jingrui He, Hanghang Tong - [[pdf]](https://arxiv.org/pdf/2601.03515)</summary>

**Abstract:** Long-term memory is a critical capability for multimodal large language model (MLLM) agents, particularly in conversational settings where information accumulates and evolves over time. However, existing benchmarks either evaluate multi-session memory in text-only conversations or assess multimodal understanding within localized contexts, failing to evaluate how multimodal memory is preserved, organized, and evolved across long-term conversational trajectories. Thus, we introduce Mem-Gallery, a new benchmark for evaluating multimodal long-term conversational memory in MLLM agents. Mem-Gallery features high-quality multi-session conversations grounded in both visual and textual information, with long interaction horizons and rich multimodal dependencies. Building on this dataset, we propose a systematic evaluation framework that assesses key memory capabilities along three functional dimensions: memory extraction and test-time adaptation, memory reasoning, and memory knowledge management. Extensive benchmarking across thirteen memory systems reveals several key findings, highlighting the necessity of explicit multimodal information retention and memory organization, the persistent limitations in memory reasoning and knowledge management, as well as the efficiency bottleneck of current models.

**arXiv ID:** 2601.03515
</details>

<details>
<summary><strong>Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents</strong> - Dehao Tao, Guoliang Ma, Yongfeng Huang, Minghu Jiang - [[pdf]](https://arxiv.org/pdf/2601.03785)</summary>

**Abstract:** Human-agent dialogues often exhibit topic continuity-a stable thematic frame that evolves through temporally adjacent exchanges-yet most large language model (LLM) agent memory systems fail to preserve it. Existing designs follow a fragmentation-compensation paradigm: they first break dialogue streams into isolated utterances for storage, then attempt to restore coherence via embedding-based retrieval. This process irreversibly damages narrative and causal flow, while biasing retrieval towards lexical similarity. We introduce membox, a hierarchical memory architecture centered on a Topic Loom that continuously monitors dialogue in a sliding-window fashion, grouping consecutive same-topic turns into coherent "memory boxes" at storage time. Sealed boxes are then linked by a Trace Weaver into long-range event-timeline traces, recovering macro-topic recurrences across discontinuities. Experiments on LoCoMo demonstrate that Membox achieves up to 68% F1 improvement on temporal reasoning tasks, outperforming competitive baselines (e.g., Mem0, A-MEM). Notably, Membox attains these gains while using only a fraction of the context tokens required by existing methods, highlighting a superior balance between efficiency and effectiveness. By explicitly modeling topic continuity, Membox offers a cognitively motivated mechanism for enhancing both coherence and efficiency in LLM agents.

**arXiv ID:** 2601.03785
</details>

<details>
<summary><strong>Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools</strong> - Kanghua Mo, Li Hu, Yucheng Long, Zhihao Li - [[pdf]](https://arxiv.org/pdf/2508.02110)</summary>

**Abstract:** Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface, where adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. The proposed attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even against prompt-level defenses, auditor-based detection, and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface. Notably, AMA is orthogonal to injection attacks and can be combined with them to achieve stronger attack efficacy, highlighting the need for execution-level defenses beyond prompt-level and auditor-based mechanisms. Code is available at this https URL.

**arXiv ID:** 2508.02110
</details>

<details>
<summary><strong>Improved LLM Agents for Financial Document Question Answering</strong> - Nelvin Tan, Zian Seng, Liang Zhang, Yu-Ching Shih, Dong Yang, Amol Salunkhe - [[pdf]](https://arxiv.org/pdf/2506.08726)</summary>

**Abstract:** Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance.

**arXiv ID:** 2506.08726
</details>

<details>
<summary><strong>PilotRL: Training Language Model Agents via Global Planning-Guided Progressive Reinforcement Learning</strong> - Keer Lu, Chong Chen, Xili Wang, Bin Cui, Yunhuai Liu, Wentao Zhang - [[pdf]](https://arxiv.org/pdf/2508.00344)</summary>

**Abstract:** Large Language Models (LLMs) have shown remarkable advancements in tackling agent-oriented tasks. Despite their potential, existing work faces challenges when deploying LLMs in agent-based environments. The widely adopted agent paradigm ReAct centers on integrating single-step reasoning with immediate action execution, which limits its effectiveness in complex tasks requiring long-term strategic planning. Furthermore, the coordination between the planner and executor during problem-solving is also a critical factor to consider in agent design. Additionally, current approaches predominantly rely on supervised fine-tuning, which often leads models to memorize established task completion trajectories, thereby restricting their generalization ability when confronted with novel problem contexts. To address these challenges, we introduce an adaptive global plan-based agent paradigm AdaPlan, aiming to synergize high-level explicit guidance with execution to support effective long-horizon decision-making. Based on the proposed paradigm, we further put forward PilotRL, a global planning-guided training framework for LLM agents driven by progressive reinforcement learning. We first develop the model's ability to follow explicit guidance from global plans when addressing agent tasks. Subsequently, based on this foundation, we focus on optimizing the quality of generated plans. Finally, we conduct joint optimization of the model's planning and execution coordination. Experiments indicate that PilotRL could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + PilotRL surpassing closed-sourced GPT-4o by 3.60%, while showing a more substantial gain of 55.78% comparing to GPT-4o-mini at a comparable parameter scale.

**arXiv ID:** 2508.00344
</details>

<details>
<summary><strong>Reinforcement Learning for Tool-Integrated Interleaved Thinking towards Cross-Domain Generalization</strong> - Zhengyu Chen, Jinluan Yang, Teng Xiao, Ruochen Zhou, Luan Zhang, Xiangyu Xi, Xiaowei Shi, Wei Wang, Jinggang Wang - [[pdf]](https://arxiv.org/pdf/2510.11184)</summary>

**Abstract:** Recent advances in large language models (LLMs) have demonstrated remarkable capabilities in reasoning and tool utilization. However, the generalization of tool-augmented reinforcement learning (RL) across diverse domains remains a significant challenge. Standard paradigms often treat tool usage as a linear or isolated event, which becomes brittle when transferring skills from restricted domains (e.g., mathematics) to open-ended tasks. In this work, we investigate the cross-domain generalization of an LLM agent trained exclusively on mathematical problem-solving. To facilitate robust skill transfer, we propose a {\textbf{R}einforcement Learning for \textbf{I}nterleaved \textbf{T}ool \textbf{E}xecution (RITE)}. Unlike traditional methods, RITE enforces a continuous ``Plan-Action-Reflection'' cycle, allowing the model to ground its reasoning in intermediate tool outputs and self-correct during long-horizon tasks. To effectively train this complex interleaved policy, we introduce {Dr. GRPO}, a robust optimization objective that utilizes token-level loss aggregation with importance sampling to mitigate reward sparsity and high-variance credit assignment. Furthermore, we employ a dual-component reward system and dynamic curriculum via online rollout filtering to ensure structural integrity and sample efficiency. Extensive experiments reveal that our approach, despite being trained solely on math tasks, achieves state-of-the-art performance across diverse reasoning domains, demonstrating high token efficiency and strong generalization capabilities.

**arXiv ID:** 2510.11184
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (25 papers)</h2></summary>

<details>
<summary><strong>Architecting Agentic Communities using Design Patterns</strong> - Zoran Milosevic, Fethi Rabhi - [[pdf]](https://arxiv.org/pdf/2601.03624)</summary>

**Abstract:** The rapid evolution of Large Language Models (LLM) and subsequent Agentic AI technologies requires systematic architectural guidance for building sophisticated, production-grade systems. This paper presents an approach for architecting such systems using design patterns derived from enterprise distributed systems standards, formal methods, and industry practice. We classify these patterns into three tiers: LLM Agents (task-specific automation), Agentic AI (adaptive goal-seekers), and Agentic Communities (organizational frameworks where AI agents and human participants coordinate through formal roles, protocols, and governance structures). We focus on Agentic Communities - coordination frameworks encompassing LLM Agents, Agentic AI entities, and humans - most relevant for enterprise and industrial applications. Drawing on established coordination principles from distributed systems, we ground these patterns in a formal framework that specifies collaboration agreements where AI agents and humans fill roles within governed ecosystems. This approach provides both practical guidance and formal verification capabilities, enabling expression of organizational, legal, and ethical rules through accountability mechanisms that ensure operational and verifiable governance of inter-agent communication, negotiation, and intent modeling. We validate this framework through a clinical trial matching case study. Our goal is to provide actionable guidance to practitioners while maintaining the formal rigor essential for enterprise deployment in dynamic, multi-agent ecosystems.

**arXiv ID:** 2601.03624
</details>

<details>
<summary><strong>Enhancing LLM Instruction Following: An Evaluation-Driven Multi-Agentic Workflow for Prompt Instructions Optimization</strong> - Alberto Purpura, Li Wang, Sahil Badyal, Eugenio Beaufrand, Adam Faulkner - [[pdf]](https://arxiv.org/pdf/2601.03359)</summary>

**Abstract:** Large Language Models (LLMs) often generate substantively relevant content but fail to adhere to formal constraints, leading to outputs that are conceptually correct but procedurally flawed. Traditional prompt refinement approaches focus on rephrasing the description of the primary task an LLM has to perform, neglecting the granular constraints that function as acceptance criteria for its response. We propose a novel multi-agentic workflow that decouples optimization of the primary task description from its constraints, using quantitative scores as feedback to iteratively rewrite and improve them. Our evaluation demonstrates this method produces revised prompts that yield significantly higher compliance scores from models like Llama 3.1 8B and Mixtral-8x 7B.

**arXiv ID:** 2601.03359
</details>

<details>
<summary><strong>Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions</strong> - Abhishek Rath - [[pdf]](https://arxiv.org/pdf/2601.04170)</summary>

**Abstract:** Multi-agent Large Language Model (LLM) systems have emerged as powerful architectures for complex task decomposition and collaborative problem-solving. However, their long-term behavioral stability remains largely unexamined. This study introduces the concept of agent drift, defined as the progressive degradation of agent behavior, decision quality, and inter-agent coherence over extended interaction sequences. We present a comprehensive theoretical framework for understanding drift phenomena, proposing three distinct manifestations: semantic drift (progressive deviation from original intent), coordination drift (breakdown in multi-agent consensus mechanisms), and behavioral drift (emergence of unintended strategies).
We introduce the Agent Stability Index (ASI), a novel composite metric framework for quantifying drift across twelve dimensions, including response consistency, tool usage patterns, reasoning pathway stability, and inter-agent agreement rates. Through simulation-based analysis and theoretical modeling, we demonstrate how unchecked agent drift can lead to substantial reductions in task completion accuracy and increased human intervention requirements.
We propose three mitigation strategies: episodic memory consolidation, drift-aware routing protocols, and adaptive behavioral anchoring. Theoretical analysis suggests these approaches can significantly reduce drift-related errors while maintaining system throughput. This work establishes a foundational methodology for monitoring, measuring, and mitigating agent drift in production agentic AI systems, with direct implications for enterprise deployment reliability and AI safety research.

**arXiv ID:** 2601.04170
</details>

<details>
<summary><strong>$α^3$-Bench: A Unified Benchmark of Safety, Robustness, and Efficiency for LLM-Based UAV Agents over 6G Networks</strong> - Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah - [[pdf]](https://arxiv.org/pdf/2601.03281)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly used as high level controllers for autonomous Unmanned Aerial Vehicle (UAV) missions. However, existing evaluations rarely assess whether such agents remain safe, protocol compliant, and effective under realistic next generation networking constraints. This paper introduces $\alpha^3$-Bench, a benchmark for evaluating LLM driven UAV autonomy as a multi turn conversational reasoning and control problem operating under dynamic 6G conditions. Each mission is formulated as a language mediated control loop between an LLM based UAV agent and a human operator, where decisions must satisfy strict schema validity, mission policies, speaker alternation, and safety constraints while adapting to fluctuating network slices, latency, jitter, packet loss, throughput, and edge load variations.
To reflect modern agentic workflows, $\alpha^3$-Bench integrates a dual action layer supporting both tool calls and agent to agent coordination, enabling evaluation of tool use consistency and multi agent interactions. We construct a large scale corpus of 113k conversational UAV episodes grounded in UAVBench scenarios and evaluate 17 state of the art LLMs using a fixed subset of 50 episodes per scenario under deterministic decoding. We propose a composite $\alpha^3$ metric that unifies six pillars: Task Outcome, Safety Policy, Tool Consistency, Interaction Quality, Network Robustness, and Communication Cost, with efficiency normalized scores per second and per thousand tokens. Results show that while several models achieve high mission success and safety compliance, robustness and efficiency vary significantly under degraded 6G conditions, highlighting the need for network aware and resource efficient LLM based UAV agents. The dataset is publicly available on GitHub : this https URL

**arXiv ID:** 2601.03281
</details>

<details>
<summary><strong>PC2P: Multi-Agent Path Finding via Personalized-Enhanced Communication and Crowd Perception</strong> - Guotao Li, Shaoyun Xu, Yuexing Hao, Yang Wang, Yuhui Sun - [[pdf]](https://arxiv.org/pdf/2601.03301)</summary>

**Abstract:** Distributed Multi-Agent Path Finding (MAPF) integrated with Multi-Agent Reinforcement Learning (MARL) has emerged as a prominent research focus, enabling real-time cooperative decision-making in partially observable environments through inter-agent communication. However, due to insufficient collaborative and perceptual capabilities, existing methods are inadequate for scaling across diverse environmental conditions. To address these challenges, we propose PC2P, a novel distributed MAPF method derived from a Q-learning-based MARL framework. Initially, we introduce a personalized-enhanced communication mechanism based on dynamic graph topology, which ascertains the core aspects of ``who" and ``what" in interactive process through three-stage operations: selection, generation, and aggregation. Concurrently, we incorporate local crowd perception to enrich agents' heuristic observation, thereby strengthening the model's guidance for effective actions via the integration of static spatial constraints and dynamic occupancy changes. To resolve extreme deadlock issues, we propose a region-based deadlock-breaking strategy that leverages expert guidance to implement efficient coordination within confined areas. Experimental results demonstrate that PC2P achieves superior performance compared to state-of-the-art distributed MAPF methods in varied environments. Ablation studies further confirm the effectiveness of each module for overall performance.

**arXiv ID:** 2601.03301
</details>

<details>
<summary><strong>MARVEL: A Multi Agent-based Research Validator and Enabler using Large Language Models</strong> - Nikhil Mukund, Yifang Luo, Fan Zhang, Lisa Barsotti, Erik Katsavounidis - [[pdf]](https://arxiv.org/pdf/2601.03436)</summary>

**Abstract:** We present MARVEL (this https URL), a locally deployable, open-source framework for domain-aware question answering and assisted scientific research. It is designed to address the increasing demands of a digital assistant for scientific groups that can read highly technical data, cite precisely, and operate within authenticated networks. MARVEL combines a fast path for straightforward queries with a more deliberate DeepSearch mode that integrates retrieval-augmented generation and Monte Carlo Tree Search. It explores complementary subqueries, allocates more compute to promising branches, and maintains a global evidence ledger that preserves sources during drafting. We applied this framework in the context of gravitational-wave research related to the Laser Interferometer Gravitational-wave Observatory. Answers are grounded in a curated semantic index of research literature, doctoral theses, LIGO documents, and long-running detector electronic logbooks, with targeted web searches when appropriate. Because direct benchmarking against commercial LLMs cannot be performed on private data, we evaluated MARVEL on two publicly available surrogate datasets that capture comparable semantic and technical characteristics. On these benchmarks, MARVEL matches a GPT-4o mini baseline on literature-centric queries and substantially outperforms it on detector-operations content, where domain retrieval and guided reasoning are decisive. By making the complete framework and evaluation datasets openly available, we aim to provide a reproducible foundation for developing domain-specific scientific assistants.

**arXiv ID:** 2601.03436
</details>

<details>
<summary><strong>Microeconomic Foundations of Multi-Agent Learning</strong> - Nassim Helou - [[pdf]](https://arxiv.org/pdf/2601.03451)</summary>

**Abstract:** Modern AI systems increasingly operate inside markets and institutions where data, behavior, and incentives are endogenous. This paper develops an economic foundation for multi-agent learning by studying a principal-agent interaction in a Markov decision process with strategic externalities, where both the principal and the agent learn over time. We propose a two-phase incentive mechanism that first estimates implementable transfers and then uses them to steer long-run dynamics; under mild regret-based rationality and exploration conditions, the mechanism achieves sublinear social-welfare regret and thus asymptotically optimal welfare. Simulations illustrate how even coarse incentives can correct inefficient learning under stateful externalities, highlighting the necessity of incentive-aware design for safe and welfare-aligned AI in markets and insurance.

**arXiv ID:** 2601.03451
</details>

<details>
<summary><strong>O-Researcher: An Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL</strong> - Yi Yao, He Zhu, Piaohong Wang, Jincheng Ren, Xinlong Yang, Qianben Chen, Xiaowan Li, Dingfeng Shi, Jiaxian Li, Qiexiang Wang, Sinuo Wang, Xinpeng Liu, Jiaqi Wu, Minghao Liu, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2601.03743)</summary>

**Abstract:** The performance gap between closed-source and open-source large language models (LLMs) is largely attributed to disparities in access to high-quality training data. To bridge this gap, we introduce a novel framework for the automated synthesis of sophisticated, research-grade instructional data. Our approach centers on a multi-agent workflow where collaborative AI agents simulate complex tool-integrated reasoning to generate diverse and high-fidelity data end-to-end. Leveraging this synthesized data, we develop a two-stage training strategy that integrates supervised fine-tuning with a novel reinforcement learning method, designed to maximize model alignment and capability. Extensive experiments demonstrate that our framework empowers open-source models across multiple scales, enabling them to achieve new state-of-the-art performance on the major deep research benchmark. This work provides a scalable and effective pathway for advancing open-source LLMs without relying on proprietary data or models.

**arXiv ID:** 2601.03743
</details>

<details>
<summary><strong>When Numbers Start Talking: Implicit Numerical Coordination Among LLM-Based Agents</strong> - Alessio Buscemi, Daniele Proverbio, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Liò - [[pdf]](https://arxiv.org/pdf/2601.03846)</summary>

**Abstract:** LLMs-based agents increasingly operate in multi-agent environments where strategic interaction and coordination are required. While existing work has largely focused on individual agents or on interacting agents sharing explicit communication, less is known about how interacting agents coordinate implicitly. In particular, agents may engage in covert communication, relying on indirect or non-linguistic signals embedded in their actions rather than on explicit messages. This paper presents a game-theoretic study of covert communication in LLM-driven multi-agent systems. We analyse interactions across four canonical game-theoretic settings under different communication regimes, including explicit, restricted, and absent communication. Considering heterogeneous agent personalities and both one-shot and repeated games, we characterise when covert signals emerge and how they shape coordination and strategic outcomes.

**arXiv ID:** 2601.03846
</details>

<details>
<summary><strong>HoneyTrap: Deceiving Large Language Model Attackers to Honeypot Traps with Resilient Multi-Agent Defense</strong> - Siyuan Li, Xi Lin, Jun Wu, Zehao Liu, Haoyu Li, Tianjie Ju, Xiang Chen, Jianhua Li - [[pdf]](https://arxiv.org/pdf/2601.04034)</summary>

**Abstract:** Jailbreak attacks pose significant threats to large language models (LLMs), enabling attackers to bypass safeguards. However, existing reactive defense approaches struggle to keep up with the rapidly evolving multi-turn jailbreaks, where attackers continuously deepen their attacks to exploit vulnerabilities. To address this critical challenge, we propose HoneyTrap, a novel deceptive LLM defense framework leveraging collaborative defenders to counter jailbreak attacks. It integrates four defensive agents, Threat Interceptor, Misdirection Controller, Forensic Tracker, and System Harmonizer, each performing a specialized security role and collaborating to complete a deceptive defense. To ensure a comprehensive evaluation, we introduce MTJ-Pro, a challenging multi-turn progressive jailbreak dataset that combines seven advanced jailbreak strategies designed to gradually deepen attack strategies across multi-turn attacks. Besides, we present two novel metrics: Mislead Success Rate (MSR) and Attack Resource Consumption (ARC), which provide more nuanced assessments of deceptive defense beyond conventional measures. Experimental results on GPT-4, GPT-3.5-turbo, Gemini-1.5-pro, and LLaMa-3.1 demonstrate that HoneyTrap achieves an average reduction of 68.77% in attack success rates compared to state-of-the-art baselines. Notably, even in a dedicated adaptive attacker setting with intensified conditions, HoneyTrap remains resilient, leveraging deceptive engagement to prolong interactions, significantly increasing the time and computational costs required for successful exploitation. Unlike simple rejection, HoneyTrap strategically wastes attacker resources without impacting benign queries, improving MSR and ARC by 118.11% and 149.16%, respectively.

**arXiv ID:** 2601.04034
</details>

<details>
<summary><strong>SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Science</strong> - Wonduk Seo, Juhyeon Lee, Yanjun Shao, Qingshan Zhou, Seunghyun Lee, Yi Bu - [[pdf]](https://arxiv.org/pdf/2503.23314)</summary>

**Abstract:** Large Language Models (LLMs) have enabled dynamic reasoning in automated data analytics, yet recent multi-agent systems remain limited by rigid, single-path workflows that restrict strategic exploration and often lead to suboptimal outcomes. To overcome these limitations, we propose SPIO (Sequential Plan Integration and Optimization), a framework that replaces rigid workflows with adaptive, multi-path planning across four core modules: data preprocessing, feature engineering, model selection, and hyperparameter tuning. In each module, specialized agents generate diverse candidate strategies, which are cascaded and refined by an optimization agent. SPIO offers two operating modes: SPIO-S for selecting a single optimal pipeline, and SPIO-E for ensembling top-k pipelines to maximize robustness. Extensive evaluations on Kaggle and OpenML benchmarks show that SPIO consistently outperforms state-of-the-art baselines, achieving an average performance gain of 5.6%. By explicitly exploring and integrating multiple solution paths, SPIO delivers a more flexible, accurate, and reliable foundation for automated data science.

**arXiv ID:** 2503.23314
</details>

<details>
<summary><strong>D-Artemis: A Deliberative Cognitive Framework for Mobile GUI Multi-Agents</strong> - Hongze Mi, Yibo Feng, Wenjie Lu, Yuqi Wang, Jinyuan Li, Song Cao, He Cui, Tengfei Tian, Xuelin Zhang, Haotian Luo, Di Sun, Jun Fang, Hua Chai, Naiqiang Tan, Gang Pan - [[pdf]](https://arxiv.org/pdf/2509.21799)</summary>

**Abstract:** Graphical User Interface (GUI) agents aim to automate a wide spectrum of human tasks by emulating user interaction. Despite rapid advancements, current approaches are hindered by several critical challenges: data bottleneck in end-to-end training, high cost of delayed error detection, and risk of contradictory guidance. Inspired by the human cognitive loop of Thinking, Alignment, and Reflection, we present D-Artemis -- a novel deliberative framework in this paper. D-Artemis leverages a fine-grained, app-specific tip retrieval mechanism to inform its decision-making process. It also employs a proactive Pre-execution Alignment stage, where Thought-Action Consistency (TAC) Check module and Action Correction Agent (ACA) work in concert to mitigate the risk of execution failures. A post-execution Status Reflection Agent (SRA) completes the cognitive loop, enabling strategic learning from experience. Crucially, D-Artemis enhances the capabilities of general-purpose Multimodal large language models (MLLMs) for GUI tasks without the need for training on complex trajectory datasets, demonstrating strong generalization. D-Artemis establishes new state-of-the-art (SOTA) results across both major benchmarks, achieving a 75.8% success rate on AndroidWorld and 96.8% on ScreenSpot-V2. Extensive ablation studies further demonstrate the significant contribution of each component to the framework.

**arXiv ID:** 2509.21799
</details>

<details>
<summary><strong>When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning</strong> - Hyeong Kyu Choi, Xiaojin Zhu, Sharon Li - [[pdf]](https://arxiv.org/pdf/2510.07517)</summary>

**Abstract:** Multi-agent debate (MAD) aims to improve large language model (LLM) reasoning by letting multiple agents exchange answers and then aggregate their opinions. Yet recent studies reveal that agents are not neutral: they are prone to identity-driven sycophancy and self-bias, uncritically adopting a peer's view or stubbornly adhering to their own prior output, undermining the reliability of debate. In this work, we present the first principled framework that joins sycophancy and self-bias to mitigate and quantify identity bias in MAD. First, we formalize the debate dynamics as an identity-weighted Bayesian update process. Second, we propose response anonymization: by removing identity markers from prompts, agents cannot distinguish "self" from "peer", which forces equal weights on agent identity, thereby reducing bias and improving trustworthiness. Third, we define the Identity Bias Coefficient (IBC), a principled bias metric that measures an agent's tendency to follow its peer versus itself. Empirical studies across multiple models and benchmarks confirm that identity bias is widespread, with sycophancy far more common than self-bias. Our findings highlight the need to ensure that MAD systems reason based on content rather than identity. Code is released in this https URL.

**arXiv ID:** 2510.07517
</details>

<details>
<summary><strong>Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response</strong> - Philip Drammeh - [[pdf]](https://arxiv.org/pdf/2511.15755)</summary>

**Abstract:** Large language models (LLMs) promise to accelerate incident response in production systems, yet single-agent approaches generate vague, unusable recommendations. We present this http URL, a reproducible containerized framework demonstrating that multi-agent orchestration fundamentally transforms LLM-based incident response quality. Through 348 controlled trials comparing single-agent copilot versus multi-agent systems on identical incident scenarios, we find that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches, an 80 times improvement in action specificity and 140 times improvement in solution correctness. Critically, multi-agent systems exhibit zero quality variance across all trials, enabling production SLA commitments impossible with inconsistent single-agent outputs. Both architectures achieve similar comprehension latency (approx.40s), establishing that the architectural value lies in deterministic quality, not speed. We introduce Decision Quality (DQ), a novel metric capturing validity, specificity, and correctness properties essential for operational deployment that existing LLM metrics do not address. These findings reframe multi-agent orchestration from a performance optimization to a production-readiness requirement for LLM-based incident response. All code, Docker configurations, and trial data are publicly available for reproduction.

**arXiv ID:** 2511.15755
</details>

<details>
<summary><strong>NEMO-4-PAYPAL: Leveraging NVIDIA's Nemo Framework for empowering PayPal's Commerce Agent</strong> - Sudhanshu Garg, Andrew Wang, Chaitanya Kulkarni, Ali Sahami, Farhad Farahani, Sean Yun-Shiuan Chuang, Jian Wan, Srinivasan Manoharan, Uma Kona, Nitin Sharma, Linsey Pang, Prakhar Mehrotra, Jessica Clark, Mark Moyou - [[pdf]](https://arxiv.org/pdf/2512.21578)</summary>

**Abstract:** We present the development and optimization of PayPal's Commerce Agent, powered by NEMO-4-PAYPAL, a multi-agent system designed to revolutionize agentic commerce on the PayPal platform. Through our strategic partnership with NVIDIA, we leveraged the NeMo Framework for LLM model fine-tuning to enhance agent performance. Specifically, we optimized the Search and Discovery agent by replacing our base model with a fine-tuned Nemotron small language model (SLM).
We conducted comprehensive experiments using the llama3.1-nemotron-nano-8B-v1 architecture, training LoRA-based models through systematic hyperparameter sweeps across learning rates, optimizers (Adam, AdamW), cosine annealing schedules, and LoRA ranks. Our contributions include: (1) the first application of NVIDIA's NeMo Framework to commerce-specific agent optimization, (2) LLM powered fine-tuning strategy for retrieval-focused commerce tasks, (3) demonstration of significant improvements in latency and cost while maintaining agent quality, and (4) a scalable framework for multi-agent system optimization in production e-commerce environments. Our results demonstrate that the fine-tuned Nemotron SLM effectively resolves the key performance issue in the retrieval component, which represents over 50\% of total agent response time, while maintaining or enhancing overall system performance.

**arXiv ID:** 2512.21578
</details>

<details>
<summary><strong>Computing Universal Plans for Partially Observable Multi-Agent Routing Using Answer Set Programming</strong> - Fengming Zhu, Fangzhen Lin - [[pdf]](https://arxiv.org/pdf/2305.16203)</summary>

**Abstract:** Multi-agent routing problems have gained significant attention recently due to their wide range of industrial applications, ranging from logistics warehouse automation to indoor service robots. Conventionally, they are modeled as classical planning problems. In this paper, we argue that it can be beneficial to formulate them as universal planning problems, particularly when the agents are autonomous entities and may encounter unforeseen situations. We therefore propose universal plans, also known as policies, as the solution concept, and implement a system based on Answer Set Programming (ASP) to compute them. Given an arbitrary two-dimensional map and a profile of goals for a group of partially observable agents, the system translates the problem configuration into logic programs and finds a feasible universal plan for each agent, mapping its observations to actions while ensuring that there are no collisions with other agents. We use the system to conduct experiments and obtain findings regarding the types of goal profiles and environments that lead to feasible policies, as well as how feasibility may depend on the agents' sensors. We also demonstrate how users can customize action preferences to compute more efficient policies, even (near-)optimal ones. The code is available at this https URL.

**arXiv ID:** 2305.16203
</details>

<details>
<summary><strong>Web Fraud Attacks Against LLM-Driven Multi-Agent Systems</strong> - Dezhang Kong, Hujin Peng, Yilun Zhang, Lele Zhao, Zhenhua Xu, Shi Lin, Changting Lin, Meng Han - [[pdf]](https://arxiv.org/pdf/2509.01211)</summary>

**Abstract:** With the proliferation of LLM-driven multi-agent systems (MAS), the security of Web links has become a critical concern. Once MAS is induced to trust a malicious link, attackers can use it as a springboard to expand the attack surface. In this paper, we propose Web Fraud Attacks, a novel type of attack manipulating unique structures of web links to deceive MAS. We design 12 representative attack variants that encompass various methods, such as homoglyph deception, sub-directory nesting, and parameter obfuscation. Through extensive experiments on these attack vectors, we demonstrate that Web fraud attacks not only exhibit significant destructive potential across different MAS architectures but also possess a distinct advantage in evasion: they circumvent the need for complex input design, lowering the threshold for attacks significantly. These results underscore the importance of addressing Web fraud attacks, providing new insights into MAS safety. Our code is available at this https URL.

**arXiv ID:** 2509.01211
</details>

<details>
<summary><strong>OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning</strong> - Xian Gao, Zongyun Zhang, Ting Liu, Yuzhuo Fu - [[pdf]](https://arxiv.org/pdf/2509.14803)</summary>

**Abstract:** In online learning environments, students often lack personalized peer interactions, which are crucial for cognitive development and learning engagement. Although previous studies have employed large language models (LLMs) to simulate interactive learning environments, these interactions are limited to conversational exchanges, failing to adapt to learners' individualized cognitive and psychological states. As a result, students' engagement is low and they struggle to gain inspiration. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs integrated with Theory of Mind (ToM). OnlineMate simulates peer-like roles, infers learners' psychological states such as misunderstandings and confusion during collaborative discussions, and dynamically adjusts interaction strategies to support higher-order thinking. Comprehensive evaluations, including simulation-based experiments, human assessments, and real classroom trials, demonstrate that OnlineMate significantly promotes deep learning and cognitive engagement by elevating students' average cognitive level while substantially improving emotional engagement scores.

**arXiv ID:** 2509.14803
</details>

<details>
<summary><strong>LLM-Enabled Multi-Agent Systems: Empirical Evaluation and Insights into Emerging Design Patterns & Paradigms</strong> - Harri Renney, Maxim N Nethercott, Nathan Renney, Peter Hayes - [[pdf]](https://arxiv.org/pdf/2601.03328)</summary>

**Abstract:** This paper formalises the literature on emerging design patterns and paradigms for Large Language Model (LLM)-enabled multi-agent systems (MAS), evaluating their practical utility across various domains. We define key architectural components, including agent orchestration, communication mechanisms, and control-flow strategies, and demonstrate how these enable rapid development of modular, domain-adaptive solutions. Three real-world case studies are tested in controlled, containerised pilots in telecommunications security, national heritage asset management, and utilities customer service automation. Initial empirical results show that, for these case studies, prototypes were delivered within two weeks and pilot-ready solutions within one month, suggesting reduced development overhead compared to conventional approaches and improved user accessibility. However, findings also reinforce limitations documented in the literature, including variability in LLM behaviour that leads to challenges in transitioning from prototype to production maturity. We conclude by outlining critical research directions for improving reliability, scalability, and governance in MAS architectures and the further work needed to mature MAS design patterns to mitigate the inherent challenges.

**arXiv ID:** 2601.03328
</details>

<details>
<summary><strong>Sensor to Pixels: Decentralized Swarm Gathering via Image-Based Reinforcement Learning</strong> - Yigal Koifman, Eran Iceland, Erez Koifman, Ariel Barel, Alfred M. Bruckstein - [[pdf]](https://arxiv.org/pdf/2601.03413)</summary>

**Abstract:** This study highlights the potential of image-based reinforcement learning methods for addressing swarm-related tasks. In multi-agent reinforcement learning, effective policy learning depends on how agents sense, interpret, and process inputs. Traditional approaches often rely on handcrafted feature extraction or raw vector-based representations, which limit the scalability and efficiency of learned policies concerning input order and size. In this work we propose an image-based reinforcement learning method for decentralized control of a multi-agent system, where observations are encoded as structured visual inputs that can be processed by Neural Networks, extracting its spatial features and producing novel decentralized motion control rules. We evaluate our approach on a multi-agent convergence task of agents with limited-range and bearing-only sensing that aim to keep the swarm cohesive during the aggregation. The algorithm's performance is evaluated against two benchmarks: an analytical solution proposed by Bellaiche and Bruckstein, which ensures convergence but progresses slowly, and VariAntNet, a neural network-based framework that converges much faster but shows medium success rates in hard constellations. Our method achieves high convergence, with a pace nearly matching that of VariAntNet. In some scenarios, it serves as the only practical alternative.

**arXiv ID:** 2601.03413
</details>

<details>
<summary><strong>Noncooperative Consensus via a Trading-based Auction</strong> - Jaehan Im, Filippos Fotiadis, Daniel Delahaye, Ufuk Topcu, David Fridovich-Keil - [[pdf]](https://arxiv.org/pdf/2502.03616)</summary>

**Abstract:** Noncooperative multi-agent systems often face coordination challenges due to conflicting preferences among agents. In particular, when agents act in their own self-interest, they may prefer different choices among multiple feasible outcomes, leading to suboptimal outcomes or even safety concerns. We propose an algorithm named trading auction for consensus (TACo), a decentralized approach that enables noncooperative agents to reach consensus without communicating directly or disclosing private valuations. TACo facilitates coordination through a structured trading-based auction, where agents iteratively select choices of interest and provably reach an agreement within an a priori bounded number of steps. A series of numerical experiments validate that the termination guarantees of TACo hold in practice, and show that TACo achieves a median performance that minimizes the total cost across all agents, while allocating resources significantly more fairly than baseline approaches.

**arXiv ID:** 2502.03616
</details>

<details>
<summary><strong>V-Agent: An Interactive Video Search System Using Vision-Language Models</strong> - SunYoung Park, Jong-Hyeon Lee, Youngjune Kim, Daegyu Sung, Younghyun Yu, Young-rok Cha, Jeongho Ju - [[pdf]](https://arxiv.org/pdf/2512.16925)</summary>

**Abstract:** We introduce V-Agent, a novel multi-agent platform designed for advanced video search and interactive user-system conversations. By fine-tuning a vision-language model (VLM) with a small video preference dataset and enhancing it with a retrieval vector from an image-text retrieval model, we overcome the limitations of traditional text-based retrieval systems in multimodal scenarios. The VLM-based retrieval model independently embeds video frames and audio transcriptions from an automatic speech recognition (ASR) module into a shared multimodal representation space, enabling V-Agent to interpret both visual and spoken content for context-aware video search. This system consists of three agents-a routing agent, a search agent, and a chat agent-that work collaboratively to address user intents by refining search outputs and communicating with users. The search agent utilizes the VLM-based retrieval model together with an additional re-ranking module to further enhance video retrieval quality. Our proposed framework demonstrates state-of-the-art zero-shot performance on the MultiVENT 2.0 benchmark, highlighting its potential for both academic research and real-world applications. The retrieval model and demo videos are available at this https URL.

**arXiv ID:** 2512.16925
</details>

<details>
<summary><strong>NeuronScope: A Multi-Agent Framework for Explaining Polysemantic Neurons in Language Models</strong> - Weiqi Liu, Yongliang Miao, Haiyan Zhao, Yanguang Liu, Mengnan Du - [[pdf]](https://arxiv.org/pdf/2601.03671)</summary>

**Abstract:** Neuron-level interpretation in large language models (LLMs) is fundamentally challenged by widespread polysemanticity, where individual neurons respond to multiple distinct semantic concepts. Existing single-pass interpretation methods struggle to faithfully capture such multi-concept behavior. In this work, we propose NeuronScope, a multi-agent framework that reformulates neuron interpretation as an iterative, activation-guided process. NeuronScope explicitly deconstructs neuron activations into atomic semantic components, clusters them into distinct semantic modes, and iteratively refines each explanation using neuron activation feedback. Experiments demonstrate that NeuronScope uncovers hidden polysemanticity and produces explanations with significantly higher activation correlation compared to single-pass baselines.

**arXiv ID:** 2601.03671
</details>

<details>
<summary><strong>MDAgent2: Large Language Model for Code Generation and Knowledge Q&A in Molecular Dynamics</strong> - Zhuofan Shi, Hubao A, Yufei Shao, Dongliang Huang, Hongxu An, Chunxiao Xin, Haiyang Shen, Zhenyu Wang, Yunshan Na, Gang Huang, Xiang Jing - [[pdf]](https://arxiv.org/pdf/2601.02075)</summary>

**Abstract:** Molecular dynamics (MD) simulations are essential for understanding atomic-scale behaviors in materials science, yet writing LAMMPS scripts remains highly specialized and time-consuming tasks. Although LLMs show promise in code generation and domain-specific question answering, their performance in MD scenarios is limited by scarce domain data, the high deployment cost of state-of-the-art LLMs, and low code executability. Building upon our prior MDAgent, we present MDAgent2, the first end-to-end framework capable of performing both knowledge Q&A and code generation within the MD domain. We construct a domain-specific data-construction pipeline that yields three high-quality datasets spanning MD knowledge, question answering, and code generation. Based on these datasets, we adopt a three stage post-training strategy--continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL)--to train two domain-adapted models, MD-Instruct and MD-Code. Furthermore, we introduce MD-GRPO, a closed-loop RL method that leverages simulation outcomes as reward signals and recycles low-reward trajectories for continual refinement. We further build MDAgent2-RUNTIME, a deployable multi-agent system that integrates code generation, execution, evaluation, and self-correction. Together with MD-EvalBench proposed in this work, the first benchmark for LAMMPS code generation and question answering, our models and system achieve performance surpassing several strong this http URL work systematically demonstrates the adaptability and generalization capability of large language models in industrial simulation tasks, laying a methodological foundation for automatic code generation in AI for Science and industrial-scale simulations. URL: this https URL

**arXiv ID:** 2601.02075
</details>

<details>
<summary><strong>Hierarchical GNN-Based Multi-Agent Learning for Dynamic Queue-Jump Lane and Emergency Vehicle Corridor Formation</strong> - Haoran Su - [[pdf]](https://arxiv.org/pdf/2601.04177)</summary>

**Abstract:** Emergency vehicles require rapid passage through congested traffic, yet existing strategies fail to adapt to dynamic conditions. We propose a novel hierarchical graph neural network (GNN)-based multi-agent reinforcement learning framework to coordinate connected vehicles for emergency corridor formation. Our approach uses a high-level planner for global strategy and low-level controllers for trajectory execution, utilizing graph attention networks to scale with variable agent counts. Trained via Multi-Agent Proximal Policy Optimization (MAPPO), the system reduces emergency vehicle travel time by 28.3% compared to baselines and 44.6% compared to uncoordinated traffic in simulations. The design achieves near-zero collision rates (0.3%) while maintaining 81% of background traffic efficiency. Ablation and generalization studies confirm the framework's robustness across diverse scenarios. These results demonstrate the effectiveness of combining GNNs with hierarchical learning for intelligent transportation systems.

**arXiv ID:** 2601.04177
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Current Agents Fail to Leverage World Model as Tool for Foresight</strong> - Cheng Qian, Emre Can Acikgoz, Bingxuan Li, Xiusi Chen, Yuji Zhang, Bingxiang He, Qinyu Luo, Dilek Hakkani-Tür, Gokhan Tur, Yunzhu Li, Heng Ji - [[pdf]](https://arxiv.org/pdf/2601.03905)</summary>

**Abstract:** Agents built on vision-language models increasingly face tasks that demand anticipating future states rather than relying on short-horizon reasoning. Generative world models offer a promising remedy: agents could use them as external simulators to foresee outcomes before acting. This paper empirically examines whether current agents can leverage such world models as tools to enhance their cognition. Across diverse agentic and visual question answering tasks, we observe that some agents rarely invoke simulation (fewer than 1%), frequently misuse predicted rollouts (approximately 15%), and often exhibit inconsistent or even degraded performance (up to 5%) when simulation is available or enforced. Attribution analysis further indicates that the primary bottleneck lies in the agents' capacity to decide when to simulate, how to interpret predicted outcomes, and how to integrate foresight into downstream reasoning. These findings underscore the need for mechanisms that foster calibrated, strategic interaction with world models, paving the way toward more reliable anticipatory cognition in future agent systems.

**arXiv ID:** 2601.03905
</details>

<details>
<summary><strong>ComfySearch: Autonomous Exploration and Reasoning for ComfyUI Workflows</strong> - Jinwei Su, Qizhen Lan, Zeyu Wang, Yinghui Xia, Hairu Wen, Yiqun Duan, Xi Xiao, Tianyu Shi, Yang Jingsong, Lewei He - [[pdf]](https://arxiv.org/pdf/2601.04060)</summary>

**Abstract:** AI-generated content has progressed from monolithic models to modular workflows, especially on platforms like ComfyUI, allowing users to customize complex creative pipelines. However, the large number of components in ComfyUI and the difficulty of maintaining long-horizon structural consistency under strict graph constraints frequently lead to low pass rates and workflows of limited quality. To tackle these limitations, we present ComfySearch, an agentic framework that can effectively explore the component space and generate functional ComfyUI pipelines via validation-guided workflow construction. Experiments demonstrate that ComfySearch substantially outperforms existing methods on complex and creative tasks, achieving higher executability (pass) rates, higher solution rates, and stronger generalization.

**arXiv ID:** 2601.04060
</details>

<details>
<summary><strong>AgentMark: Utility-Preserving Behavioral Watermarking for Agents</strong> - Kaibo Huang, Jin Tan, Yukun Wei, Wanling Li, Zipei Zhang, Hui Tian, Zhongliang Yang, Linna Zhou - [[pdf]](https://arxiv.org/pdf/2601.03294)</summary>

**Abstract:** LLM-based agents are increasingly deployed to autonomously solve complex tasks, raising urgent needs for IP protection and regulatory provenance. While content watermarking effectively attributes LLM-generated outputs, it fails to directly identify the high-level planning behaviors (e.g., tool and subgoal choices) that govern multi-step execution. Critically, watermarking at the planning-behavior layer faces unique challenges: minor distributional deviations in decision-making can compound during long-term agent operation, degrading utility, and many agents operate as black boxes that are difficult to intervene in directly. To bridge this gap, we propose AgentMark, a behavioral watermarking framework that embeds multi-bit identifiers into planning decisions while preserving utility. It operates by eliciting an explicit behavior distribution from the agent and applying distribution-preserving conditional sampling, enabling deployment under black-box APIs while remaining compatible with action-layer content watermarking. Experiments across embodied, tool-use, and social environments demonstrate practical multi-bit capacity, robust recovery from partial logs, and utility preservation. The code is available at this https URL.

**arXiv ID:** 2601.03294
</details>

<details>
<summary><strong>Agentic Exploration of Physics Models</strong> - Maximilian Nägele, Florian Marquardt - [[pdf]](https://arxiv.org/pdf/2509.24978)</summary>

**Abstract:** The process of scientific discovery relies on an interplay of observations, analysis, and hypothesis generation. Machine learning is increasingly being adopted to address individual aspects of this process. However, it remains an open challenge to fully automate the heuristic, iterative loop required to discover the laws of an unknown system by exploring it through experiments and analysis, without tailoring the approach to the specifics of a given task. Here, we introduce SciExplorer, an agent that leverages large language model tool-use capabilities to enable exploration of systems without any domain-specific blueprints, and apply it to physical systems that are initially unknown to the agent. We test SciExplorer on a broad set of models spanning mechanical dynamical systems, wave evolution, and quantum many-body physics. Despite using a minimal set of tools, primarily based on code execution, we observe impressive performance on tasks such as recovering equations of motion from observed dynamics and inferring Hamiltonians from expectation values. The demonstrated effectiveness of this setup opens the door towards similar scientific exploration in other domains, without the need for finetuning or task-specific instructions.

**arXiv ID:** 2509.24978
</details>

<details>
<summary><strong>FinPos: A Position-Aware Trading Agent System for Real Financial Markets</strong> - Bijia Liu, Ronghao Dang - [[pdf]](https://arxiv.org/pdf/2510.27251)</summary>

**Abstract:** The exceptional potential of large language models (LLMs) in handling text information has garnered significant attention in the field of financial trading. However, most existing trading agents operate under intraday, independent unit-based trading tasks, where decisions are made as isolated directional actions, and thus lack awareness of continuous position management. Therefore, we propose a position-aware trading task designed to simulate a more realistic market. To address this task, we propose FinPos, a position-aware trading agent system designed to explicitly model and manage continuous positions. FinPos enhances position awareness through three key mechanisms: (1) professional-level interpretation of heterogeneous market information; (2) a dual-agent decision structure that separates directional reasoning from risk-aware position adjustment; and (3) multi-timescale reward signals, allowing the agent to internalize position awareness through experiential feedback rather than static instructions alone. Extensive experiments demonstrate that FinPos surpasses state-of-the-art trading agents in the position-aware trading task, which closely mirrors real market conditions. More importantly, our findings reveal that LLM-centered agent systems exhibit a vast, largely unexplored potential in long-term market decision-making.

**arXiv ID:** 2510.27251
</details>

<details>
<summary><strong>Enabling Agents to Communicate Entirely in Latent Space</strong> - Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Yu Cheng, Bo Zheng, Wei Chen, Haochao Ying - [[pdf]](https://arxiv.org/pdf/2511.09149)</summary>

**Abstract:** While natural language is the de facto communication medium for LLM-based agents, it presents a fundamental constraint. The process of downsampling rich, internal latent states into discrete tokens inherently limits the depth and nuance of information that can be transmitted, thereby hindering collaborative problem-solving. Inspired by telepathy, which bypasses symbolic language in communication, we propose Interlat (Inter-agent Latent Space Communication), a paradigm that leverages the continuous last hidden states of an LLM as a representation of its thought for direct communication (termed latent communication). An additional learned compression process further compresses latent communication via latent space reasoning. Experiments demonstrate that Interlat outperforms both fine-tuned chain-of-thought (CoT) prompting and single-agent baselines, even across heterogeneous models, promoting more exploratory behavior and enabling genuine utilization of latent information. Further compression not only substantially accelerates inference by up to 24 times but also maintains competitive performance through an efficient information-preserving mechanism. We position this work as a feasibility study of entirely latent space inter-agent communication, and our results highlight its potential, offering valuable insights for future research.

**arXiv ID:** 2511.09149
</details>

<details>
<summary><strong>TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent</strong> - Dominik Meier, Jan Philip Wahle, Paul Röttger, Terry Ruas, Bela Gipp - [[pdf]](https://arxiv.org/pdf/2505.20118)</summary>

**Abstract:** As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.

**arXiv ID:** 2505.20118
</details>

<details>
<summary><strong>From Bits to Chips: An LLM-based Hardware-Aware Quantization Agent for Streamlined Deployment of LLMs</strong> - Kaiyuan Deng, Hangyu Zheng, Minghai Qing, Kunxiong Zhu, Gen Li, Yang Xiao, Lan Emily Zhang, Linke Guo, Bo Hui, Yanzhi Wang, Geng Yuan, Gagan Agrawal, Wei Niu, Xiaolong Ma - [[pdf]](https://arxiv.org/pdf/2601.03484)</summary>

**Abstract:** Deploying models, especially large language models (LLMs), is becoming increasingly attractive to a broader user base, including those without specialized expertise. However, due to the resource constraints of certain hardware, maintaining high accuracy with larger model while meeting the hardware requirements remains a significant challenge. Model quantization technique helps mitigate memory and compute bottlenecks, yet the added complexities of tuning and deploying quantized models further exacerbates these challenges, making the process unfriendly to most of the users. We introduce the Hardware-Aware Quantization Agent (HAQA), an automated framework that leverages LLMs to streamline the entire quantization and deployment process by enabling efficient hyperparameter tuning and hardware configuration, thereby simultaneously improving deployment quality and ease of use for a broad range of users. Our results demonstrate up to a 2.3x speedup in inference, along with increased throughput and improved accuracy compared to unoptimized models on Llama. Additionally, HAQA is designed to implement adaptive quantization strategies across diverse hardware platforms, as it automatically finds optimal settings even when they appear counterintuitive, thereby reducing extensive manual effort and demonstrating superior adaptability. Code will be released.

**arXiv ID:** 2601.03484
</details>

<details>
<summary><strong>Towards Safe Autonomous Driving: A Real-Time Motion Planning Algorithm on Embedded Hardware</strong> - Korbinian Moller, Glenn Johannes Tungka, Lucas Jürgens, Johannes Betz - [[pdf]](https://arxiv.org/pdf/2601.03904)</summary>

**Abstract:** Ensuring the functional safety of Autonomous Vehicles (AVs) requires motion planning modules that not only operate within strict real-time constraints but also maintain controllability in case of system faults. Existing safeguarding concepts, such as Online Verification (OV), provide safety layers that detect infeasible planning outputs. However, they lack an active mechanism to ensure safe operation in the event that the main planner fails. This paper presents a first step toward an active safety extension for fail-operational Autonomous Driving (AD). We deploy a lightweight sampling-based trajectory planner on an automotive-grade, embedded platform running a Real-Time Operating System (RTOS). The planner continuously computes trajectories under constrained computational resources, forming the foundation for future emergency planning architectures. Experimental results demonstrate deterministic timing behavior with bounded latency and minimal jitter, validating the feasibility of trajectory planning on safety-certifiable hardware. The study highlights both the potential and the remaining challenges of integrating active fallback mechanisms as an integral part of next-generation safeguarding frameworks. The code is available at: this https URL

**arXiv ID:** 2601.03904
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>MobileDreamer: Generative Sketch World Model for GUI Agent</strong> - Yilin Cao, Yufeng Zhong, Zhixiong Zeng, Liming Zheng, Jing Huang, Haibo Qiu, Peng Shi, Wenji Mao, Wan Guanglu - [[pdf]](https://arxiv.org/pdf/2601.04035)</summary>

**Abstract:** Mobile GUI agents have shown strong potential in real-world automation and practical applications. However, most existing agents remain reactive, making decisions mainly from current screen, which limits their performance on long-horizon tasks. Building a world model from repeated interactions enables forecasting action outcomes and supports better decision making for mobile GUI agents. This is challenging because the model must predict post-action states with spatial awareness while remaining efficient enough for practical deployment. In this paper, we propose MobileDreamer, an efficient world-model-based lookahead framework to equip the GUI agents based on the future imagination provided by the world model. It consists of textual sketch world model and rollout imagination for GUI agent. Textual sketch world model forecasts post-action states through a learning process to transform digital images into key task-related sketches, and designs a novel order-invariant learning strategy to preserve the spatial information of GUI elements. The rollout imagination strategy for GUI agent optimizes the action-selection process by leveraging the prediction capability of world model. Experiments on Android World show that MobileDreamer achieves state-of-the-art performance and improves task success by 5.25%. World model evaluations further verify that our textual sketch modeling accurately forecasts key GUI elements.

**arXiv ID:** 2601.04035
</details>

<details>
<summary><strong>Aligning Findings with Diagnosis: A Self-Consistent Reinforcement Learning Framework for Trustworthy Radiology Reporting</strong> - Kun Zhao, Siyuan Dai, Pan Wang, Jifeng Song, Hui Ji, Chenghua Lin, Liang Zhan, Haoteng Tang - [[pdf]](https://arxiv.org/pdf/2601.03321)</summary>

**Abstract:** Multimodal Large Language Models (MLLMs) have shown strong potential for radiology report generation, yet their clinical translation is hindered by architectural heterogeneity and the prevalence of factual hallucinations. Standard supervised fine-tuning often fails to strictly align linguistic outputs with visual evidence, while existing reinforcement learning approaches struggle with either prohibitive computational costs or limited exploration. To address these challenges, we propose a comprehensive framework for self-consistent radiology report generation. First, we conduct a systematic evaluation to identify optimal vision encoder and LLM backbone configurations for medical imaging. Building on this foundation, we introduce a novel "Reason-then-Summarize" architecture optimized via Group Relative Policy Optimization (GRPO). This framework restructures generation into two distinct components: a think block for detailed findings and an answer block for structured disease labels. By utilizing a multi-dimensional composite reward function, we explicitly penalize logical discrepancies between the generated narrative and the final diagnosis. Extensive experiments on the MIMIC-CXR benchmark demonstrate that our method achieves state-of-the-art performance in clinical efficacy metrics and significantly reduces hallucinations compared to strong supervised baselines.

**arXiv ID:** 2601.03321
</details>

<details>
<summary><strong>A Reinforcement Learning-Based Model for Mapping and Goal-Directed Navigation Using Multiscale Place Fields</strong> - Bekarys Dukenbaev, Andrew Gerstenslager, Alexander Johnson, Ali A. Minai - [[pdf]](https://arxiv.org/pdf/2601.03520)</summary>

**Abstract:** Autonomous navigation in complex and partially observable environments remains a central challenge in robotics. Several bio-inspired models of mapping and navigation based on place cells in the mammalian hippocampus have been proposed. This paper introduces a new robust model that employs parallel layers of place fields at multiple spatial scales, a replay-based reward mechanism, and dynamic scale fusion. Simulations show that the model improves path efficiency and accelerates learning compared to single-scale baselines, highlighting the value of multiscale spatial representations for adaptive robot navigation.

**arXiv ID:** 2601.03520
</details>

<details>
<summary><strong>ReLA: Representation Learning and Aggregation for Job Scheduling with Reinforcement Learning</strong> - Zhengyi Kwan, Wei Zhang, Aik Beng Ng, Zhengkui Wang, Simon See - [[pdf]](https://arxiv.org/pdf/2601.03646)</summary>

**Abstract:** Job scheduling is widely used in real-world manufacturing systems to assign ordered job operations to machines under various constraints. Existing solutions remain limited by long running time or insufficient schedule quality, especially when problem scale increases. In this paper, we propose ReLA, a reinforcement-learning (RL) scheduler built on structured representation learning and aggregation. ReLA first learns diverse representations from scheduling entities, including job operations and machines, using two intra-entity learning modules with self-attention and convolution and one inter-entity learning module with cross-attention. These modules are applied in a multi-scale architecture, and their outputs are aggregated to support RL decision-making. Across experiments on small, medium, and large job instances, ReLA achieves the best makespan in most tested settings over the latest solutions. On non-large instances, ReLA reduces the optimality gap of the SOTA baseline by 13.0%, while on large-scale instances it reduces the gap by 78.6%, with the average optimality gaps lowered to 7.3% and 2.1%, respectively. These results confirm that ReLA's learned representations and aggregation provide strong decision support for RL scheduling, and enable fast job completion and decision-making for real-world applications.

**arXiv ID:** 2601.03646
</details>

<details>
<summary><strong>R$^3$L: Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification</strong> - Weijie Shi, Yanxi Chen, Zexi Li, Xuchen Pan, Yuchang Sun, Jiajie Xu, Xiaofang Zhou, Yaliang Li - [[pdf]](https://arxiv.org/pdf/2601.03715)</summary>

**Abstract:** Reinforcement learning drives recent advances in LLM reasoning and agentic capabilities, yet current approaches struggle with both exploration and exploitation. Exploration suffers from low success rates on difficult tasks and high costs of repeated rollouts from scratch. Exploitation suffers from coarse credit assignment and training instability: Trajectory-level rewards penalize valid prefixes for later errors, and failure-dominated groups overwhelm the few positive signals, leaving optimization without constructive direction. To this end, we propose R$^3$L, Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification. To synthesize high-quality trajectories, R$^3$L shifts from stochastic sampling to active synthesis via reflect-then-retry, leveraging language feedback to diagnose errors, transform failed attempts into successful ones, and reduce rollout costs by restarting from identified failure points. With errors diagnosed and localized, Pivotal Credit Assignment updates only the diverging suffix where contrastive signals exist, excluding the shared prefix from gradient update. Since failures dominate on difficult tasks and reflect-then-retry produces off-policy data, risking training instability, Positive Amplification upweights successful trajectories to ensure positive signals guide the optimization process. Experiments on agentic and reasoning tasks demonstrate 5\% to 52\% relative improvements over baselines while maintaining training stability. Our code is released at this https URL.

**arXiv ID:** 2601.03715
</details>

<details>
<summary><strong>NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning</strong> - Zhongtao Miao, Kaiyan Zhao, Masaaki Nagata, Yoshimasa Tsuruoka - [[pdf]](https://arxiv.org/pdf/2601.03790)</summary>

**Abstract:** Neologism-aware machine translation aims to translate source sentences containing neologisms into target languages. This field remains underexplored compared with general machine translation (MT). In this paper, we propose an agentic framework, NeoAMT, for neologism-aware machine translation using a Wiktionary search tool. Specifically, we first create a new dataset for neologism-aware machine translation and develop a search tool based on Wiktionary. The new dataset covers 16 languages and 75 translation directions and is derived from approximately 10 million records of an English Wiktionary dump. The retrieval corpus of the search tool is also constructed from around 3 million cleaned records of the Wiktionary dump. We then use it for training the translation agent with reinforcement learning (RL) and evaluating the accuracy of neologism-aware machine translation. Based on this, we also propose an RL training framework that contains a novel reward design and an adaptive rollout generation approach by leveraging "translation difficulty" to further improve the translation quality of translation agents using our search tool.

**arXiv ID:** 2601.03790
</details>

<details>
<summary><strong>InfiniteWeb: Scalable Web Environment Synthesis for GUI Agent Training</strong> - Ziyun Zhang, Zezhou Wang, Xiaoyi Zhang, Zongyu Guo, Jiahao Li, Bin Li, Yan Lu - [[pdf]](https://arxiv.org/pdf/2601.04126)</summary>

**Abstract:** GUI agents that interact with graphical interfaces on behalf of users represent a promising direction for practical AI assistants. However, training such agents is hindered by the scarcity of suitable environments. We present InfiniteWeb, a system that automatically generates functional web environments at scale for GUI agent training. While LLMs perform well on generating a single webpage, building a realistic and functional website with many interconnected pages faces challenges. We address these challenges through unified specification, task-centric test-driven development, and a combination of website seed with reference design image to ensure diversity. Our system also generates verifiable task evaluators enabling dense reward signals for reinforcement learning. Experiments show that InfiniteWeb surpasses commercial coding agents at realistic website construction, and GUI agents trained on our generated environments achieve significant performance improvements on OSWorld and Online-Mind2Web, demonstrating the effectiveness of proposed system.

**arXiv ID:** 2601.04126
</details>

<details>
<summary><strong>DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search</strong> - Fang Wu, Weihao Xuan, Heli Qi, Ximing Lu, Aaron Tu, Li Erran Li, Yejin Choi - [[pdf]](https://arxiv.org/pdf/2509.25454)</summary>

**Abstract:** Although RLVR has become an essential component for developing advanced reasoning skills in language models, contemporary studies have documented training plateaus after thousands of optimization steps, i.e., notable decreases in performance gains despite increased computational investment. This limitation stems from the sparse exploration patterns inherent in current RLVR practices, where models rely on limited rollouts that often miss critical reasoning paths and fail to provide systematic coverage of the solution space. We present DeepSearch, a framework that integrates Monte Carlo Tree Search (MCTS) directly into RLVR training. In contrast to existing methods that rely on tree search only at inference, DeepSearch embeds structured search into the training loop, enabling systematic exploration and fine-grained credit assignment across reasoning steps. Through training-time exploration, DeepSearch addresses the fundamental bottleneck of insufficient exploration, which leads to diminishing performance improvements over prolonged training steps. Our contributions include: (1) a global frontier selection strategy that prioritizes promising nodes across the search tree, (2) selection with entropy-based guidance that identifies confident paths for supervision, and (3) adaptive replay buffer training with solution caching for efficiency. Experiments on mathematical reasoning benchmarks show that DeepSearch achieves 62.95% average accuracy and establishes a new state-of-the-art for 1.5B reasoning models, while using 5.7x fewer GPU hours than extended training approaches. These results highlight the importance of strategic exploration over brute-force scaling and demonstrate the promise of algorithmic innovation for advancing RLVR methodologies. DeepSearch establishes a new direction for scaling reasoning capabilities through systematic search rather than prolonged computation.

**arXiv ID:** 2509.25454
</details>

<details>
<summary><strong>Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots</strong> - Chenhao Li, Andreas Krause, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2504.16680)</summary>

**Abstract:** Reinforcement Learning (RL) has achieved impressive results in robotics, yet high-performing pipelines remain highly task-specific, with little reuse of prior data. Offline Model-based RL (MBRL) offers greater data efficiency by training policies entirely from existing datasets, but suffers from compounding errors and distribution shift in long-horizon rollouts. Although existing methods have shown success in controlled simulation benchmarks, robustly applying them to the noisy, biased, and partially observed datasets typical of real-world robotics remains challenging. We present a principled pipeline for making offline MBRL effective on physical robots. Our RWM-U extends autoregressive world models with epistemic uncertainty estimation, enabling temporally consistent multi-step rollouts with uncertainty effectively propagated over long horizons. We combine RWM-U with MOPO-PPO, which adapts uncertainty-penalized policy optimization to the stable, on-policy PPO framework for real-world control. We evaluate our approach on diverse manipulation and locomotion tasks in simulation and on real quadruped and humanoid, training policies entirely from offline datasets. The resulting policies consistently outperform model-free and uncertainty-unaware model-based baselines, and fusing real-world data in model learning further yields robust policies that surpass online model-free baselines trained solely in simulation.

**arXiv ID:** 2504.16680
</details>

<details>
<summary><strong>SciNetBench: A Relation-Aware Benchmark for Scientific Literature Retrieval Agents</strong> - Chenyang Shao, Yong Li, Fengli Xu - [[pdf]](https://arxiv.org/pdf/2601.03260)</summary>

**Abstract:** The rapid development of AI agent has spurred the development of advanced research tools, such as Deep Research. Achieving this require a nuanced understanding of the relations within scientific literature, surpasses the scope of keyword-based or embedding-based retrieval. Existing retrieval agents mainly focus on the content-level similarities and are unable to decode critical relational dynamics, such as identifying corroborating or conflicting studies or tracing technological lineages, all of which are essential for a comprehensive literature review. Consequently, this fundamental limitation often results in a fragmented knowledge structure, misleading sentiment interpretation, and inadequate modeling of collective scientific progress. To investigate relation-aware retrieval more deeply, we propose SciNetBench, the first Scientific Network Relation-aware Benchmark for literature retrieval agents. Constructed from a corpus of over 18 million AI papers, our benchmark systematically evaluates three levels of relations: ego-centric retrieval of papers with novel knowledge structures, pair-wise identification of scholarly relationships, and path-wise reconstruction of scientific evolutionary trajectories. Through extensive evaluation of three categories of retrieval agents, we find that their accuracy on relation-aware retrieval tasks often falls below 20%, revealing a core shortcoming of current retrieval paradigms. Notably, further experiments on the literature review tasks demonstrate that providing agents with relational ground truth leads to a substantial 23.4% performance improvement in the review quality, validating the critical importance of relation-aware retrieval. We publicly release our benchmark at this https URL to support future research on advanced retrieval systems.

**arXiv ID:** 2601.03260
</details>

<details>
<summary><strong>Interleaved Reasoning for Large Language Models via Reinforcement Learning</strong> - Roy Xie, David Qiu, Deepak Gopinath, Dong Lin, Yanchao Sun, Chong Wang, Saloni Potdar, Bhuwan Dhingra - [[pdf]](https://arxiv.org/pdf/2505.19640)</summary>

**Abstract:** Long chain-of-thought (CoT) significantly enhances the reasoning capabilities of large language models (LLMs). However, extensive reasoning traces lead to inefficiencies and increased time-to-first-token (TTFT). We propose a training paradigm that uses only reinforcement learning (RL) to guide reasoning LLMs to interleave thinking and answering for multi-hop questions. We observe that models inherently possess the ability to perform interleaved reasoning, which can be further enhanced through RL. We introduce a simple yet effective reward scheme to incentivize correct intermediate steps, guiding the policy model toward correct reasoning paths by leveraging intermediate signals generated during interleaved reasoning. Extensive experiments across five diverse datasets and three RL algorithms (PPO, GRPO, and REINFORCE++) demonstrate consistent improvements over traditional think-answer reasoning, without requiring external tools. Our method improves final task accuracy and overall efficiency by enabling more effective credit assignment during RL. Specifically, our approach achieves a 12.5% improvement in Pass@1 accuracy, while reducing overall reasoning length by 37% and TTFT by over 80% on average. Furthermore, our method, trained solely on question answering and logical reasoning datasets, exhibits strong generalization to complex reasoning datasets such as MATH, GPQA, and MMLU. Additionally, we conduct in-depth analysis to reveal several valuable insights into conditional reward modeling.

**arXiv ID:** 2505.19640
</details>

<details>
<summary><strong>WebAnchor: Anchoring Agent Planning to Stabilize Long-Horizon Web Reasoning</strong> - Xinmiao Yu, Liwen Zhang, Xiaocheng Feng, Yong Jiang, Bing Qin, Pengjun Xie, Jingren Zhou - [[pdf]](https://arxiv.org/pdf/2601.03164)</summary>

**Abstract:** Large Language Model(LLM)-based agents have shown strong capabilities in web information seeking, with reinforcement learning (RL) becoming a key optimization paradigm. However, planning remains a bottleneck, as existing methods struggle with long-horizon strategies. Our analysis reveals a critical phenomenon, plan anchor, where the first reasoning step disproportionately impacts downstream behavior in long-horizon web reasoning tasks. Current RL algorithms, fail to account for this by uniformly distributing rewards across the trajectory. To address this, we propose Anchor-GRPO, a two-stage RL framework that decouples planning and execution. In Stage 1, the agent optimizes its first-step planning using fine-grained rubrics derived from self-play experiences and human calibration. In Stage 2, execution is aligned with the initial plan through sparse rewards, ensuring stable and efficient tool usage. We evaluate Anchor-GRPO on four benchmarks: BrowseComp, BrowseComp-Zh, GAIA, and XBench-DeepSearch. Across models from 3B to 30B, Anchor-GRPO outperforms baseline GRPO and First-step GRPO, improving task success and tool efficiency. Notably, WebAnchor-30B achieves 46.0% pass@1 on BrowseComp and 76.4% on GAIA. Anchor-GRPO also demonstrates strong scalability, getting higher accuracy as model size and context length increase.

**arXiv ID:** 2601.03164
</details>

<details>
<summary><strong>Agentic Rubrics as Contextual Verifiers for SWE Agents</strong> - Mohit Raghavendra, Anisha Gunjal, Bing Liu, Yunzhong He - [[pdf]](https://arxiv.org/pdf/2601.04171)</summary>

**Abstract:** Verification is critical for improving agents: it provides the reward signal for Reinforcement Learning and enables inference-time gains through Test-Time Scaling (TTS). Despite its importance, verification in software engineering (SWE) agent settings often relies on code execution, which can be difficult to scale due to environment setup overhead. Scalable alternatives such as patch classifiers and heuristic methods exist, but they are less grounded in codebase context and harder to interpret. To this end, we explore Agentic Rubrics: an expert agent interacts with the repository to create a context-grounded rubric checklist, and candidate patches are then scored against it without requiring test execution. On SWE-Bench Verified under parallel TTS evaluation, Agentic Rubrics achieve a score of 54.2% on Qwen3-Coder-30B-A3B and 40.6% on Qwen3-32B, with at least a +3.5 percentage-point gain over the strongest baseline in our comparison set. We further analyze rubric behavior, showing that rubric scores are consistent with ground-truth tests while also flagging issues that tests do not capture. Our ablations show that agentic context gathering is essential for producing codebase-specific, unambiguous criteria. Together, these results suggest that Agentic Rubrics provide an efficient, scalable, and granular verification signal for SWE agents.

**arXiv ID:** 2601.04171
</details>

<details>
<summary><strong>Cells on Autopilot: Adaptive Cell (Re)Selection via Reinforcement Learning</strong> - Marvin Illian, Ramin Khalili, Antonio A. de A. Rocha, Lin Wang - [[pdf]](https://arxiv.org/pdf/2601.04083)</summary>

**Abstract:** The widespread deployment of 5G networks, together with the coexistence of 4G/LTE networks, provides mobile devices a diverse set of candidate cells to connect to. However, associating mobile devices to cells to maximize overall network performance, a.k.a. cell (re)selection, remains a key challenge for mobile operators. Today, cell (re)selection parameters are typically configured manually based on operator experience and rarely adapted to dynamic network conditions. In this work, we ask: Can an agent automatically learn and adapt cell (re)selection parameters to consistently improve network performance? We present a reinforcement learning (RL)-based framework called CellPilot that adaptively tunes cell (re)selection parameters by learning spatiotemporal patterns of mobile network dynamics. Our study with real-world data demonstrates that even a lightweight RL agent can outperform conventional heuristic reconfigurations by up to 167%, while generalizing effectively across different network scenarios. These results indicate that data-driven approaches can significantly improve cell (re)selection configurations and enhance mobile network performance.

**arXiv ID:** 2601.04083
</details>

<details>
<summary><strong>Graph Reinforcement Learning for Power Grids: A Comprehensive Survey</strong> - Mohamed Hassouna, Clara Holzhüter, Pawel Lytaev, Josephine Thomas, Bernhard Sick, Christoph Scholz - [[pdf]](https://arxiv.org/pdf/2407.04522)</summary>

**Abstract:** The increasing share of renewable energy and distributed electricity generation requires the development of deep learning approaches to address the lack of flexibility inherent in traditional power grid methods. In this context, Graph Neural Networks are a promising solution due to their ability to learn from graph-structured data. Combined with Reinforcement Learning, they can be used as control approaches to determine remedial actions. This review analyses how Graph Reinforcement Learning can improve representation learning and decision-making in power grid applications, particularly transmission and distribution grids. We analyze the reviewed approaches in terms of the graph structure, the Graph Neural Network architecture, and the Reinforcement Learning approach. Although Graph Reinforcement Learning has demonstrated adaptability to unpredictable events and noisy data, its current stage is primarily proof-of-concept, and it is not yet deployable to real-world applications. We highlight the open challenges and limitations for real-world applications.

**arXiv ID:** 2407.04522
</details>

<details>
<summary><strong>WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks</strong> - Hao Bai, Alexey Taymanov, Tong Zhang, Aviral Kumar, Spencer Whitehead - [[pdf]](https://arxiv.org/pdf/2601.02439)</summary>

**Abstract:** We present WebGym, the largest-to-date open-source environment for training realistic visual web agents. Real websites are non-stationary and diverse, making artificial or small-scale task sets insufficient for robust policy learning. WebGym contains nearly 300,000 tasks with rubric-based evaluations across diverse, real-world websites and difficulty levels. We train agents with a simple reinforcement learning (RL) recipe, which trains on the agent's own interaction traces (rollouts), using task rewards as feedback to guide learning. To enable scaling RL, we speed up sampling of trajectories in WebGym by developing a high-throughput asynchronous rollout system, designed specifically for web agents. Our system achieves a 4-5x rollout speedup compared to naive implementations. Second, we scale the task set breadth, depth, and size, which results in continued performance improvement. Fine-tuning a strong base vision-language model, Qwen-3-VL-8B-Instruct, on WebGym results in an improvement in success rate on an out-of-distribution test set from 26.2% to 42.9%, significantly outperforming agents based on proprietary models such as GPT-4o and GPT-5-Thinking that achieve 27.1% and 29.8%, respectively. This improvement is substantial because our test set consists only of tasks on websites never seen during training, unlike many other prior works on training visual web agents.

**arXiv ID:** 2601.02439
</details>

<details>
<summary><strong>Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail</strong> - NVIDIA, Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, Liang Feng, Greg Heinrich, Jack Huang, Peter Karkus, Boyi Li, Pinyi Li, Tsung-Yi Lin, Dongran Liu, Ming-Yu Liu, Langechuan Liu, Zhijian Liu, Jason Lu, Yunxiang Mao, Pavlo Molchanov, Lindsey Pavao, Zhenghao Peng, Mike Ranzinger, Ed Schmerling, Shida Shen, Yunfei Shi, Sarah Tariq, Ran Tian, Tilman Wekel, Xinshuo Weng, Tianjun Xiao, Eric Yang, Xiaodong Yang, Yurong You, Xiaohui Zeng, Wenyuan Zhang, Boris Ivanovic, Marco Pavone - [[pdf]](https://arxiv.org/pdf/2511.00088)</summary>

**Abstract:** End-to-end architectures trained via imitation learning have advanced autonomous driving by scaling model size and data, yet performance remains brittle in safety-critical long-tail scenarios where supervision is sparse and causal understanding is limited. We introduce Alpamayo-R1 (AR1), a vision-language-action model (VLA) that integrates Chain of Causation reasoning with trajectory planning for complex driving scenarios. Our approach features three key innovations: (1) the Chain of Causation (CoC) dataset, built through a hybrid auto-labeling and human-in-the-loop pipeline producing decision-grounded, causally linked reasoning traces aligned with driving behaviors; (2) a modular VLA architecture combining Cosmos-Reason, a vision-language model pre-trained for Physical AI, with a diffusion-based trajectory decoder that generates dynamically feasible trajectories in real time; (3) a multi-stage training strategy using supervised fine-tuning to elicit reasoning and reinforcement learning (RL) to enforce reasoning-action consistency and optimize reasoning quality. AR1 achieves up to a 12% improvement in planning accuracy on challenging cases compared to a trajectory-only baseline, with a 35% reduction in close encounter rate in closed-loop simulation. RL post-training improves reasoning quality by 45% and reasoning-action consistency by 37%. Model scaling from 0.5B to 7B parameters shows consistent improvements. On-vehicle road tests confirm real-time performance (99 ms latency) and successful urban deployment. By bridging interpretable reasoning with precise control, AR1 demonstrates a practical path towards Level 4 autonomous driving. Model weights are available at this https URL with inference code at this https URL.

**arXiv ID:** 2511.00088
</details>

<details>
<summary><strong>Dual-Attention Heterogeneous GNN for Multi-robot Collaborative Area Search via Deep Reinforcement Learning</strong> - Lina Zhu, Jiyu Cheng, Yuehu Liu, Wei Zhang - [[pdf]](https://arxiv.org/pdf/2601.03686)</summary>

**Abstract:** In multi-robot collaborative area search, a key challenge is to dynamically balance the two objectives of exploring unknown areas and covering specific targets to be rescued. Existing methods are often constrained by homogeneous graph representations, thus failing to model and balance these distinct tasks. To address this problem, we propose a Dual-Attention Heterogeneous Graph Neural Network (DA-HGNN) trained using deep reinforcement learning. Our method constructs a heterogeneous graph that incorporates three entity types: robot nodes, frontier nodes, and interesting nodes, as well as their historical states. The dual-attention mechanism comprises the relational-aware attention and type-aware attention operations. The relational-aware attention captures the complex spatio-temporal relationships among robots and candidate goals. Building on this relational-aware heterogeneous graph, the type-aware attention separately computes the relevance between robots and each goal type (frontiers vs. points of interest), thereby decoupling the exploration and coverage from the unified tasks. Extensive experiments conducted in interactive 3D scenarios within the iGibson simulator, leveraging the Gibson and MatterPort3D datasets, validate the superior scalability and generalization capability of the proposed approach.

**arXiv ID:** 2601.03686
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
