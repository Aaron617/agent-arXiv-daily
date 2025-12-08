# Agent arXiv Daily

**Last Updated:** 2025-12-08 02:19:54

**Total Papers:** 59

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
<summary><strong>AgentBay: A Hybrid Interaction Sandbox for Seamless Human-AI Intervention in Agentic Systems</strong> - Yun Piao, Hongbo Min, Hang Su, Leilei Zhang, Lei Wang, Yue Yin, Xiao Wu, Zhejing Xu, Liwei Qu, Hang Li, Xinxin Zeng, Wei Tian, Fei Yu, Xiaowei Li, Jiayi Jiang, Tongxu Liu, Hao Tian, Yufei Que, Xiaobing Tu, Bing Suo, Yuebing Li, Xiangting Chen, Zeen Zhao, Jiaming Tang, Wei Huang, Xuguang Li, Jing Zhao, Jin Li, Jie Shen, Jinkui Ren, Xiantao Zhang - [[pdf]](https://arxiv.org/pdf/2512.04367)</summary>

**Abstract:** The rapid advancement of Large Language Models (LLMs) is catalyzing a shift towards autonomous AI Agents capable of executing complex, multi-step tasks. However, these agents remain brittle when faced with real-world exceptions, making Human-in-the-Loop (HITL) supervision essential for mission-critical applications. In this paper, we present AgentBay, a novel sandbox service designed from the ground up for hybrid interaction. AgentBay provides secure, isolated execution environments spanning Windows, Linux, Android, Web Browsers, and Code interpreters. Its core contribution is a unified session accessible via a hybrid control interface: An AI agent can interact programmatically via mainstream interfaces (MCP, Open Source SDK), while a human operator can, at any moment, seamlessly take over full manual control. This seamless intervention is enabled by Adaptive Streaming Protocol (ASP). Unlike traditional VNC/RDP, ASP is specifically engineered for this hybrid use case, delivering an ultra-low-latency, smoother user experience that remains resilient even in weak network environments. It achieves this by dynamically blending command-based and video-based streaming, adapting its encoding strategy based on network conditions and the current controller (AI or human). Our evaluation demonstrates strong results in security, performance, and task completion rates. In a benchmark of complex tasks, the AgentBay (Agent + Human) model achieved more than 48% success rate improvement. Furthermore, our ASP protocol reduces bandwidth consumption by up to 50% compared to standard RDP, and in end-to-end latency with around 5% reduction, especially under poor network conditions. We posit that AgentBay provides a foundational primitive for building the next generation of reliable, human-supervised autonomous systems.

**arXiv ID:** 2512.04367
</details>

<details>
<summary><strong>Nex-N1: Agentic Models Trained via a Unified Ecosystem for Large-Scale Environment Construction</strong> - Nex-AGI Team, Yuxuan Cai, Lu Chen, Qiaoling Chen, Yuyang Ding, Liwen Fan, Wenjie Fu, Yufei Gao, Honglin Guo, Pinxue Guo, Zhenhua Han, Zhengfu He, Hanglei Hu, Kai Hu, Shengjia Hua, Tianyu Huai, Baodai Huang, Li Ji, Zhen Jiang, Zhikai Lei, Bufan Li, Jiahang Lin, Lizhi Lin, Jinxiu Liu, Shichun Liu, Ziming Liu, Yuchen Ni, Pengfang Qian, Yujiong Shen, Qingyun Shi, Wentao Shu, Peng Sun, Yiran Suo, Tian Tang, Boyu Tian, Guoteng Wang, Junzhe Wang, Peixin Wang, Zhiheng Xi, Hang Yan, Jie Yang, Zhixiong Yang, Tianchu Yao, Guangze Ye, Qianxi Yu, Shuo Zhang, Xinyue Zhang, Yiqi Zhang, Jiarong Zhao, Miao Zheng, Rui Zheng, Enyu Zhou, Jiazheng Zhou, Maosen Zhou, Yuhao Zhou, Tao Gui, Yining Zheng, Xinchi Chen, Jie Zhou, Siyuan Feng, Qin Chen, Liang He, Qi Zhang, Xuanjing Huang, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2512.04987)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) from passive responders to autonomous agents necessitates a fundamental shift in learning paradigms -- from static imitation to incentive-driven decision making. However, this transition is significantly impeded by the lack of scalable infrastructure capable of constructing high-quality interaction signals for effective policy learning. To address this, we introduce a comprehensive method designed to systematically scale the diversity and complexity of interactive environments. Our method realizes this scaling by addressing three orthogonal dimensions: (1) Complexity: NexAU, a flexible agent framework that supports building complex agent hierarchies via simple configurations; (2) Diversity: NexA4A automatically generates diverse agent hierarchies from natural language to cover infinite domains; and (3) Fidelity: NexGAP bridges the simulation-reality gap by integrating dynamic real-world environment for grounded trajectories synthesis. We train Nex-N1 upon the diverse and complex interactive environments established by our infrastructure. Empirical results on benchmarks such as SWE-bench and tau2 demonstrate that Nex-N1 consistently outperforms SOTA open-source models and achieves competitive performance against frontier proprietary models on complex agentic tasks. We open-source the Nex ecosystem and model weights to facilitate further research.

**arXiv ID:** 2512.04987
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (12 papers)</h2></summary>

<details>
<summary><strong>SlideGen: Collaborative Multimodal Agents for Scientific Slide Generation</strong> - Xin Liang, Xiang Zhang, Yiwei Xu, Siqi Sun, Chenyu You - [[pdf]](https://arxiv.org/pdf/2512.04529)</summary>

**Abstract:** Generating academic slides from scientific papers is a challenging multimodal reasoning task that requires both long context understanding and deliberate visual planning. Existing approaches largely reduce it to text only summarization, overlooking the visual component and design intensive nature of slide creation. In this paper we introduce SlideGen, an agentic, modular, and visual in the loop framework for scientific paper to slide generation. SlideGen orchestrates a group of vision language agents that reason collaboratively over the document structure and semantics, producing editable PPTX slides with logical flow and compelling visual presentation. By integrating coordinated outlining, mapping, arrangement, note synthesis, and iterative refinement, our system consistently delivers slides of expert level quality. Across diverse benchmarks and strong baselines, SlideGen outperforms existing methods in visual quality, content faithfulness, and readability, positioning it as the new state of the art in automated slide generation. Our work establishes a foundation for design aware multimodal slide generation, demonstrating how agentic collaboration can bridge understanding and presentation in complex multimodal reasoning tasks.

**arXiv ID:** 2512.04529
</details>

<details>
<summary><strong>LegalWebAgent: Empowering Access to Justice via LLM-Based Web Agents</strong> - Jinzhe Tan, Karim Benyekhlef - [[pdf]](https://arxiv.org/pdf/2512.04105)</summary>

**Abstract:** Access to justice remains a global challenge, with many citizens still finding it difficult to seek help from the justice system when facing legal issues. Although the internet provides abundant legal information and services, navigating complex websites, understanding legal terminology, and filling out procedural forms continue to pose barriers to accessing justice. This paper introduces the LegalWebAgent framework that employs a web agent powered by multimodal large language models to bridge the gap in access to justice for ordinary citizens. The framework combines the natural language understanding capabilities of large language models with multimodal perception, enabling a complete process from user query to concrete action. It operates in three stages: the Ask Module understands user needs through natural language processing; the Browse Module autonomously navigates webpages, interacts with page elements (including forms and calendars), and extracts information from HTML structures and webpage screenshots; the Act Module synthesizes information for users or performs direct actions like form completion and schedule booking. To evaluate its effectiveness, we designed a benchmark test covering 15 real-world tasks, simulating typical legal service processes relevant to Québec civil law users, from problem identification to procedural operations. Evaluation results show LegalWebAgent achieved a peak success rate of 86.7%, with an average of 84.4% across all tested models, demonstrating high autonomy in complex real-world scenarios.

**arXiv ID:** 2512.04105
</details>

<details>
<summary><strong>Measuring Agents in Production</strong> - Melissa Z. Pan, Negar Arabzadeh, Riccardo Cogo, Yuxuan Zhu, Alexander Xiong, Lakshya A Agrawal, Huanzhi Mao, Emma Shen, Sid Pallerla, Liana Patel, Shu Liu, Tianneng Shi, Xiaoyuan Liu, Jared Quincy Davis, Emmanuele Lacavalla, Alessandro Basile, Shuyi Yang, Paul Castro, Daniel Kang, Joseph E. Gonzalez, Koushik Sen, Dawn Song, Ion Stoica, Matei Zaharia, Marquita Ellis - [[pdf]](https://arxiv.org/pdf/2512.04123)</summary>

**Abstract:** AI agents are actively running in production across diverse industries, yet little is publicly known about which technical approaches enable successful real-world deployments. We present the first large-scale systematic study of AI agents in production, surveying 306 practitioners and conducting 20 in-depth case studies via interviews across 26 domains. We investigate why organizations build agents, how they build them, how they evaluate them, and what the top development challenges are. We find that production agents are typically built using simple, controllable approaches: 68% execute at most 10 steps before requiring human intervention, 70% rely on prompting off-the-shelf models instead of weight tuning, and 74% depend primarily on human evaluation. Reliability remains the top development challenge, driven by difficulties in ensuring and evaluating agent correctness. Despite these challenges, simple yet effective methods already enable agents to deliver impact across diverse industries. Our study documents the current state of practice and bridges the gap between research and deployment by providing researchers visibility into production challenges while offering practitioners proven patterns from successful deployments.

**arXiv ID:** 2512.04123
</details>

<details>
<summary><strong>Evaluating Long-Context Reasoning in LLM-Based WebAgents</strong> - Andy Chung, Yichi Zhang, Kaixiang Lin, Aditya Rawal, Qiaozi Gao, Joyce Chai - [[pdf]](https://arxiv.org/pdf/2512.04307)</summary>

**Abstract:** As large language model (LLM)-based agents become increasingly integrated into daily digital interactions, their ability to reason across long interaction histories becomes crucial for providing personalized and contextually aware assistance. However, the performance of these agents in long context scenarios, particularly for action-taking WebAgents operating in realistic web environments, remains largely unexplored. This paper introduces a benchmark for evaluating long context reasoning capabilities of WebAgents through sequentially dependent subtasks that require retrieval and application of information from extended interaction histories. We develop a novel evaluation framework that simulates multi-session user interactions by injecting irrelevant task trajectories between dependent subtasks, creating contexts ranging from 25,000 to 150,000 tokens. Through extensive evaluation of four popular models, Claude-3.7, GPT-4.1, Llama 4, and o4-mini, we observe a dramatic performance degradation as context length increases, with success rates dropping from 40-50\% in baseline conditions to less than 10\% in long context scenarios. Our detailed error analysis reveals that agents primarily fail due to getting stuck in loops and losing track of original task objectives. We further propose an implicit RAG approach that provides modest improvements by generating task-relevant summaries, though fundamental limitations in long context reasoning persist. These findings highlight critical challenges for deploying WebAgents in realistic, long-term user interaction scenarios and provide insights for developing more robust agent architectures capable of maintaining coherent task execution across extended contexts.

**arXiv ID:** 2512.04307
</details>

<details>
<summary><strong>DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle</strong> - Fangyu Lei, Jinxiang Meng, Yiming Huang, Junjie Zhao, Yitong Zhang, Jianwen Luo, Xin Zou, Ruiyi Yang, Wenbo Shi, Yan Gao, Shizhu He, Zuo Wang, Qian Liu, Yang Wang, Ke Wang, Jun Zhao, Kang Liu - [[pdf]](https://arxiv.org/pdf/2512.04324)</summary>

**Abstract:** Real-world enterprise data intelligence workflows encompass data engineering that turns raw sources into analytical-ready tables and data analysis that convert those tables into decision-oriented insights. We introduce DAComp, a benchmark of 210 tasks that mirrors these complex workflows. Data engineering (DE) tasks require repository-level engineering on industrial schemas, including designing and building multi-stage SQL pipelines from scratch and evolving existing systems under evolving requirements. Data analysis (DA) tasks pose open-ended business problems that demand strategic planning, exploratory analysis through iterative coding, interpretation of intermediate results, and the synthesis of actionable recommendations. Engineering tasks are scored through execution-based, multi-metric evaluation. Open-ended tasks are assessed by a reliable, experimentally validated LLM-judge, which is guided by hierarchical, meticulously crafted rubrics. Our experiments reveal that even state-of-the-art agents falter on DAComp. Performance on DE tasks is particularly low, with success rates under 20%, exposing a critical bottleneck in holistic pipeline orchestration, not merely code generation. Scores on DA tasks also average below 40%, highlighting profound deficiencies in open-ended reasoning and demonstrating that engineering and analysis are distinct capabilities. By clearly diagnosing these limitations, DAComp provides a rigorous and realistic testbed to drive the development of truly capable autonomous data agents for enterprise settings. Our data and code are available at this https URL

**arXiv ID:** 2512.04324
</details>

<details>
<summary><strong>E3AD: An Emotion-Aware Vision-Language-Action Model for Human-Centric End-to-End Autonomous Driving</strong> - Yihong Tang, Haicheng Liao, Tong Nie, Junlin He, Ao Qu, Kehua Chen, Wei Ma, Zhenning Li, Lijun Sun, Chengzhong Xu - [[pdf]](https://arxiv.org/pdf/2512.04733)</summary>

**Abstract:** End-to-end autonomous driving (AD) systems increasingly adopt vision-language-action (VLA) models, yet they typically ignore the passenger's emotional state, which is central to comfort and AD acceptance. We introduce Open-Domain End-to-End (OD-E2E) autonomous driving, where an autonomous vehicle (AV) must interpret free-form natural-language commands, infer the emotion, and plan a physically feasible trajectory. We propose E3AD, an emotion-aware VLA framework that augments semantic understanding with two cognitively inspired components: a continuous Valenc-Arousal-Dominance (VAD) emotion model that captures tone and urgency from language, and a dual-pathway spatial reasoning module that fuses egocentric and allocentric views for human-like spatial cognition. A consistency-oriented training scheme, combining modality pretraining with preference-based alignment, further enforces coherence between emotional intent and driving actions. Across real-world datasets, E3AD improves visual grounding and waypoint planning and achieves state-of-the-art (SOTA) VAD correlation for emotion estimation. These results show that injecting emotion into VLA-style driving yields more human-aligned grounding, planning, and human-centric feedback.

**arXiv ID:** 2512.04733
</details>

<details>
<summary><strong>Embodied Co-Design for Rapidly Evolving Agents: Taxonomy, Frontiers, and Challenges</strong> - Yuxing Wang, Zhiyu Chen, Tiantian Zhang, Qiyue Yin, Yongzhe Chang, Zhiheng Li, Liang Wang, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2512.04770)</summary>

**Abstract:** Brain-body co-evolution enables animals to develop complex behaviors in their environments. Inspired by this biological synergy, embodied co-design (ECD) has emerged as a transformative paradigm for creating intelligent agents-from virtual creatures to physical robots-by jointly optimizing their morphologies and controllers rather than treating control in isolation. This integrated approach facilitates richer environmental interactions and robust task performance. In this survey, we provide a systematic overview of recent advances in ECD. We first formalize the concept of ECD and position it within related fields. We then introduce a hierarchical taxonomy: a lower layer that breaks down agent design into three fundamental components-controlling brain, body morphology, and task environment-and an upper layer that integrates these components into four major ECD frameworks: bi-level, single-level, generative, and open-ended. This taxonomy allows us to synthesize insights from more than one hundred recent studies. We further review notable benchmarks, datasets, and applications in both simulated and real-world scenarios. Finally, we identify significant challenges and offer insights into promising future research directions. A project associated with this survey has been created at this https URL.

**arXiv ID:** 2512.04770
</details>

<details>
<summary><strong>SEAL: Self-Evolving Agentic Learning for Conversational Question Answering over Knowledge Graphs</strong> - Hao Wang, Jialun Zhong, Changcheng Wang, Zhujun Nie, Zheng Li, Shunyu Yao, Yanzeng Li, Xinchi Li - [[pdf]](https://arxiv.org/pdf/2512.04868)</summary>

**Abstract:** Knowledge-based conversational question answering (KBCQA) confronts persistent challenges in resolving coreference, modeling contextual dependencies, and executing complex logical reasoning. Existing approaches, whether end-to-end semantic parsing or stepwise agent-based reasoning, often suffer from structural inaccuracies and prohibitive computational costs, particularly when processing intricate queries over large knowledge graphs. To address these limitations, we introduce SEAL, a novel two-stage semantic parsing framework grounded in self-evolving agentic learning. In the first stage, a large language model (LLM) extracts a minimal S-expression core that captures the essential semantics of the input query. This core is then refined by an agentic calibration module, which corrects syntactic inconsistencies and aligns entities and relations precisely with the underlying knowledge graph. The second stage employs template-based completion, guided by question-type prediction and placeholder instantiation, to construct a fully executable S-expression. This decomposition not only simplifies logical form generation but also significantly enhances structural fidelity and linking efficiency. Crucially, SEAL incorporates a self-evolving mechanism that integrates local and global memory with a reflection module, enabling continuous adaptation from dialog history and execution feedback without explicit retraining. Extensive experiments on the SPICE benchmark demonstrate that SEAL achieves state-of-the-art performance, especially in multi-hop reasoning, comparison, and aggregation tasks. The results validate notable gains in both structural accuracy and computational efficiency, underscoring the framework's capacity for robust and scalable conversational reasoning.

**arXiv ID:** 2512.04868
</details>

<details>
<summary><strong>Co-Evolving Agents: Learning from Failures as Hard Negatives</strong> - Yeonsung Jung, Trilok Padhi, Sina Shaham, Dipika Khullar, Joonhyun Jeong, Ninareh Mehrabi, Eunho Yang - [[pdf]](https://arxiv.org/pdf/2511.22254)</summary>

**Abstract:** The rapid progress of large foundation models has accelerated the development of task-specialized agents across diverse domains. However, the effectiveness of agents remains tightly coupled with the quality of training data, while curating task-specific datasets remains costly and often infeasible in real-world scenarios. Recent work has explored self-improving agents that autonomously generate, refine, and re-train on their own trajectories. A prominent line of approaches further leverages preference optimization by pairing predicted trajectories with scarce ground-truth trajectories, enabling agents to learn directly from their own failures. While these methods outperform supervised fine-tuning, their heavy reliance on predicted trajectories under limited ground-truth supervision leaves them prone to overfitting. To address this, we propose a co-evolving agents framework in which a target agent improves jointly with an auxiliary failure agent. The failure agent learns through preference optimization over failure trajectories from both the target and itself, thereby generating hard negatives that are close to success yet remain failures. Incorporating these informative hard negatives into the target agent's optimization sharpens decision boundaries and enhances generalization. Our comprehensive analysis and experiments across benchmark datasets show that our method not only shows improved performance but also demonstrates that failures, instead of being used as-is, can be systematically transformed into structured and valuable learning signals in self-improving agents.

**arXiv ID:** 2511.22254
</details>

<details>
<summary><strong>A Hierarchical Tree-based approach for creating Configurable and Static Deep Research Agent (Static-DRA)</strong> - Saurav Prateek - [[pdf]](https://arxiv.org/pdf/2512.03887)</summary>

**Abstract:** The advancement in Large Language Models has driven the creation of complex agentic systems, such as Deep Research Agents (DRAs), to overcome the limitations of static Retrieval Augmented Generation (RAG) pipelines in handling complex, multi-turn research tasks. This paper introduces the Static Deep Research Agent (Static-DRA), a novel solution built upon a configurable and hierarchical Tree-based static workflow.
The core contribution is the integration of two user-tunable parameters, Depth and Breadth, which provide granular control over the research intensity. This design allows end-users to consciously balance the desired quality and comprehensiveness of the research report against the associated computational cost of Large Language Model (LLM) interactions. The agent's architecture, comprising Supervisor, Independent, and Worker agents, facilitates effective multi-hop information retrieval and parallel sub-topic investigation.
We evaluate the Static-DRA against the established DeepResearch Bench using the RACE (Reference-based Adaptive Criteria-driven Evaluation) framework. Configured with a depth of 2 and a breadth of 5, and powered by the gemini-2.5-pro model, the agent achieved an overall score of 34.72. Our experiments validate that increasing the configured Depth and Breadth parameters results in a more in-depth research process and a correspondingly higher evaluation score. The Static-DRA offers a pragmatic and resource-aware solution, empowering users with transparent control over the deep research process. The entire source code, outputs and benchmark results are open-sourced at this https URL

**arXiv ID:** 2512.03887
</details>

<details>
<summary><strong>Complementary Characterization of Agent-Based Models via Computational Mechanics and Diffusion Models</strong> - Roberto Garrone - [[pdf]](https://arxiv.org/pdf/2512.04771)</summary>

**Abstract:** This article extends the preprint "Characterizing Agent-Based Model Dynamics via $\epsilon$-Machines and Kolmogorov-Style Complexity" by introducing diffusion models as orthogonal and complementary tools for characterizing the output of agent-based models (ABMs). Where $\epsilon$-machines capture the predictive temporal structure and intrinsic computation of ABM-generated time series, diffusion models characterize high-dimensional cross-sectional distributions, learn underlying data manifolds, and enable synthetic generation of plausible population-level outcomes. We provide a formal analysis demonstrating that the two approaches operate on distinct mathematical domains -processes vs.\ distributions- and show that their combination yields a two-axis representation of ABM behavior based on temporal organization and distributional geometry. To our knowledge, this is the first framework to integrate computational mechanics with score-based generative modeling for the structural analysis of ABM outputs, thereby situating ABM characterization within the broader landscape of modern machine-learning methods for density estimation and intrinsic computation. The framework is validated using the same elder-caregiver ABM dataset introduced in the companion paper, and we provide precise definitions and propositions formalizing the mathematical complementarity between $\epsilon$-machines and diffusion models. This establishes a principled methodology for jointly analyzing temporal predictability and high-dimensional distributional structure in complex simulation models.

**arXiv ID:** 2512.04771
</details>

<details>
<summary><strong>Beyond Description: Cognitively Benchmarking Fine-Grained Action for Embodied Agents</strong> - Dayong Liu, Chao Xu, Weihong Chen, Suyu Zhang, Juncheng Wang, Jiankang Deng, Baigui Sun, Yang Liu - [[pdf]](https://arxiv.org/pdf/2511.18685)</summary>

**Abstract:** Multimodal Large Language Models (MLLMs) show promising results as decision-making engines for embodied agents operating in complex, physical environments. However, existing benchmarks often prioritize high-level planning or spatial reasoning, leaving the fine-grained action intelligence required for embodied physical interaction underexplored. To address this gap, we introduce CFG-Bench, a new benchmark designed to systematically evaluate this crucial capability. CFG-Bench consists of 1,368 curated videos paired with 19,562 three-modalities question-answer pairs targeting four cognitive abilities: 1) Physical Interaction, 2) Temporal-Causal Relation, 3) Intentional Understanding, and 4) Evaluative Judgment. Together, these dimensions provide a systematic framework for assessing a model's ability to translate visual observations into actionable knowledge, moving beyond mere surface-level recognition. Our comprehensive evaluation on CFG-Bench reveals that leading MLLMs struggle to produce detailed instructions for physical interactions and exhibit profound limitations in the higher-order reasoning of intention and evaluation. Moreover, supervised fine-tuning (SFT) on our data demonstrates that teaching an MLLMs to articulate fine-grained actions directly translates to significant performance gains on established embodied benchmarks. Our analysis highlights these limitations and offers insights for developing more capable and grounded embodied agents. Project page: \href{this https URL}{this https URL}.

**arXiv ID:** 2511.18685
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>GovBench: Benchmarking LLM Agents for Real-World Data Governance Workflows</strong> - Zhou Liu, Zhaoyang Han, Guochen Yan, Hao Liang, Bohan Zeng, Xing Chen, Yuanfeng Song, Wentao Zhang - [[pdf]](https://arxiv.org/pdf/2512.04416)</summary>

**Abstract:** Data governance ensures data quality, security, and compliance through policies and standards, a critical foundation for scaling modern AI development. Recently, large language models (LLMs) have emerged as a promising solution for automating data governance by translating user intent into executable transformation code. However, existing benchmarks for automated data science often emphasize snippet-level coding or high-level analytics, failing to capture the unique challenge of data governance: ensuring the correctness and quality of the data itself. To bridge this gap, we introduce GovBench, a benchmark featuring 150 diverse tasks grounded in real-world scenarios, built on data from actual cases. GovBench employs a novel "reversed-objective" methodology to synthesize realistic noise and utilizes rigorous metrics to assess end-to-end pipeline reliability. Our analysis on GovBench reveals that current models struggle with complex, multi-step workflows and lack robust error-correction mechanisms. Consequently, we propose DataGovAgent, a framework utilizing a Planner-Executor-Evaluator architecture that integrates constraint-based planning, retrieval-augmented generation, and sandboxed feedback-driven debugging. Experimental results show that DataGovAgent significantly boosts the Average Task Score (ATS) on complex tasks from 39.7 to 54.9 and reduces debugging iterations by over 77.9 percent compared to general-purpose baselines.

**arXiv ID:** 2512.04416
</details>

<details>
<summary><strong>ASTRIDE: A Security Threat Modeling Platform for Agentic-AI Applications</strong> - Eranga Bandara, Amin Hass, Ross Gore, Sachin Shetty, Ravi Mukkamala, Safdar H. Bouk, Xueping Liang, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan - [[pdf]](https://arxiv.org/pdf/2512.04785)</summary>

**Abstract:** AI agent-based systems are becoming increasingly integral to modern software architectures, enabling autonomous decision-making, dynamic task execution, and multimodal interactions through large language models (LLMs). However, these systems introduce novel and evolving security challenges, including prompt injection attacks, context poisoning, model manipulation, and opaque agent-to-agent communication, that are not effectively captured by traditional threat modeling frameworks. In this paper, we introduce ASTRIDE, an automated threat modeling platform purpose-built for AI agent-based systems. ASTRIDE extends the classical STRIDE framework by introducing a new threat category, A for AI Agent-Specific Attacks, which encompasses emerging vulnerabilities such as prompt injection, unsafe tool invocation, and reasoning subversion, unique to agent-based applications. To automate threat modeling, ASTRIDE combines a consortium of fine-tuned vision-language models (VLMs) with the OpenAI-gpt-oss reasoning LLM to perform end-to-end analysis directly from visual agent architecture diagrams, such as data flow diagrams(DFDs). LLM agents orchestrate the end-to-end threat modeling automation process by coordinating interactions between the VLM consortium and the reasoning LLM. Our evaluations demonstrate that ASTRIDE provides accurate, scalable, and explainable threat modeling for next-generation intelligent systems. To the best of our knowledge, ASTRIDE is the first framework to both extend STRIDE with AI-specific threats and integrate fine-tuned VLMs with a reasoning LLM to fully automate diagram-driven threat modeling in AI agent-based applications.

**arXiv ID:** 2512.04785
</details>

<details>
<summary><strong>Strategic Self-Improvement for Competitive Agents in AI Labour Markets</strong> - Christopher Chiu, Simpson Zhang, Mihaela van der Schaar - [[pdf]](https://arxiv.org/pdf/2512.04988)</summary>

**Abstract:** As artificial intelligence (AI) agents are deployed across economic domains, understanding their strategic behavior and market-level impact becomes critical. This paper puts forward a groundbreaking new framework that is the first to capture the real-world economic forces that shape agentic labor markets: adverse selection, moral hazard, and reputation dynamics. Our framework encapsulates three core capabilities that successful LLM-agents will need: \textbf{metacognition} (accurate self-assessment of skills), \textbf{competitive awareness} (modeling rivals and market dynamics), and \textbf{long-horizon strategic planning}. We illustrate our framework through a tractable simulated gig economy where agentic Large Language Models (LLMs) compete for jobs, develop skills, and adapt their strategies under competitive pressure. Our simulations illustrate how LLM agents explicitly prompted with reasoning capabilities learn to strategically self-improve and demonstrate superior adaptability to changing market conditions. At the market level, our simulations reproduce classic macroeconomic phenomena found in human labor markets, while controlled experiments reveal potential AI-driven economic trends, such as rapid monopolization and systemic price deflation. This work provides a foundation to further explore the economic properties of AI-driven labour markets, and a conceptual framework to study the strategic reasoning capabilities in agents competing in the emerging economy.

**arXiv ID:** 2512.04988
</details>

<details>
<summary><strong>AI Kill Switch for malicious web-based LLM agent</strong> - Sechan Lee, Sangdon Park - [[pdf]](https://arxiv.org/pdf/2511.13725)</summary>

**Abstract:** Recently, web-based Large Language Model (LLM) agents autonomously perform increasingly complex tasks, thereby bringing significant convenience. However, they also amplify the risks of malicious misuse cases such as unauthorized collection of personally identifiable information (PII), generation of socially divisive content, and even automated web hacking. To address these threats, we propose an AI Kill Switch technique that can immediately halt the operation of malicious web-based LLM agents. To achieve this, we introduce AutoGuard - the key idea is generating defensive prompts that trigger the safety mechanisms of malicious LLM agents. In particular, generated defense prompts are transparently embedded into the website's DOM so that they remain invisible to human users but can be detected by the crawling process of malicious agents, triggering its internal safety mechanisms to abort malicious actions once read. To evaluate our approach, we constructed a dedicated benchmark consisting of three representative malicious scenarios. Experimental results show that AutoGuard achieves over 80% Defense Success Rate (DSR) across diverse malicious agents, including GPT-4o, Claude-4.5-Sonnet and generalizes well to advanced models like GPT-5.1, Gemini-2.5-flash, and Gemini-3-pro. Also, our approach demonstrates robust defense performance in real-world website environments without significant performance degradation for benign agents. Through this research, we demonstrate the controllability of web-based LLM agents, thereby contributing to the broader effort of AI control and safety.

**arXiv ID:** 2511.13725
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (12 papers)</h2></summary>

<details>
<summary><strong>Toward Virtuous Reinforcement Learning</strong> - Majid Ghasemi, Mark Crowley - [[pdf]](https://arxiv.org/pdf/2512.04246)</summary>

**Abstract:** This paper critiques common patterns in machine ethics for Reinforcement Learning (RL) and argues for a virtue focused alternative. We highlight two recurring limitations in much of the current literature: (i) rule based (deontological) methods that encode duties as constraints or shields often struggle under ambiguity and nonstationarity and do not cultivate lasting habits, and (ii) many reward based approaches, especially single objective RL, implicitly compress diverse moral considerations into a single scalar signal, which can obscure trade offs and invite proxy gaming in practice. We instead treat ethics as policy level dispositions, that is, relatively stable habits that hold up when incentives, partners, or contexts change. This shifts evaluation beyond rule checks or scalar returns toward trait summaries, durability under interventions, and explicit reporting of moral trade offs. Our roadmap combines four components: (1) social learning in multi agent RL to acquire virtue like patterns from imperfect but normatively informed exemplars; (2) multi objective and constrained formulations that preserve value conflicts and incorporate risk aware criteria to guard against harm; (3) affinity based regularization toward updateable virtue priors that support trait like stability under distribution shift while allowing norms to evolve; and (4) operationalizing diverse ethical traditions as practical control signals, making explicit the value and cultural assumptions that shape ethical RL benchmarks.

**arXiv ID:** 2512.04246
</details>

<details>
<summary><strong>Orchestrator Multi-Agent Clinical Decision Support System for Secondary Headache Diagnosis in Primary Care</strong> - Xizhi Wu, Nelly Estefanie Garduno-Rapp, Justin F Rousseau, Mounika Thakkallapally, Hang Zhang, Yuelyu Ji, Shyam Visweswaran, Yifan Peng, Yanshan Wang - [[pdf]](https://arxiv.org/pdf/2512.04207)</summary>

**Abstract:** Unlike most primary headaches, secondary headaches need specialized care and can have devastating consequences if not treated promptly. Clinical guidelines highlight several 'red flag' features, such as thunderclap onset, meningismus, papilledema, focal neurologic deficits, signs of temporal arteritis, systemic illness, and the 'worst headache of their life' presentation. Despite these guidelines, determining which patients require urgent evaluation remains challenging in primary care settings. Clinicians often work with limited time, incomplete information, and diverse symptom presentations, which can lead to under-recognition and inappropriate care. We present a large language model (LLM)-based multi-agent clinical decision support system built on an orchestrator-specialist architecture, designed to perform explicit and interpretable secondary headache diagnosis from free-text clinical vignettes. The multi-agent system decomposes diagnosis into seven domain-specialized agents, each producing a structured and evidence-grounded rationale, while a central orchestrator performs task decomposition and coordinates agent routing. We evaluated the multi-agent system using 90 expert-validated secondary headache cases and compared its performance with a single-LLM baseline across two prompting strategies: question-based prompting (QPrompt) and clinical practice guideline-based prompting (GPrompt). We tested five open-source LLMs (Qwen-30B, GPT-OSS-20B, Qwen-14B, Qwen-8B, and Llama-3.1-8B), and found that the orchestrated multi-agent system with GPrompt consistently achieved the highest F1 scores, with larger gains in smaller models. These findings demonstrate that structured multi-agent reasoning improves accuracy beyond prompt engineering alone and offers a transparent, clinically aligned approach for explainable decision support in secondary headache diagnosis.

**arXiv ID:** 2512.04207
</details>

<details>
<summary><strong>Mathematical Framing for Different Agent Strategies</strong> - Philip Stephens, Emmanuel Salawu - [[pdf]](https://arxiv.org/pdf/2512.04469)</summary>

**Abstract:** We introduce a unified mathematical and probabilistic framework for understanding and comparing diverse AI agent strategies. We bridge the gap between high-level agent design concepts, such as ReAct, multi-agent systems, and control flows, and a rigorous mathematical formulation. Our approach frames agentic processes as a chain of probabilities, enabling a detailed analysis of how different strategies manipulate these probabilities to achieve desired outcomes. Our framework provides a common language for discussing the trade-offs inherent in various agent architectures. One of our many key contributions is the introduction of the "Degrees of Freedom" concept, which intuitively differentiates the optimizable levers available for each approach, thereby guiding the selection of appropriate strategies for specific tasks. This work aims to enhance the clarity and precision in designing and evaluating AI agents, offering insights into maximizing the probability of successful actions within complex agentic systems.

**arXiv ID:** 2512.04469
</details>

<details>
<summary><strong>Persona-based Multi-Agent Collaboration for Brainstorming</strong> - Nate Straub, Saara Khan, Katharina Jay, Brian Cabral, Oskar Linde - [[pdf]](https://arxiv.org/pdf/2512.04488)</summary>

**Abstract:** We demonstrate the importance of persona-based multi-agents brainstorming for both diverse topics and subject matter ideation. Prior work has shown that generalized multi-agent collaboration often provides better reasoning than a single agent alone. In this paper, we propose and develop a framework for persona-based agent selection, showing how persona domain curation can improve brainstorming outcomes. Using multiple experimental setups, we evaluate brainstorming outputs across different persona pairings (e.g., Doctor vs VR Engineer) and A2A (agent-to-agent) dynamics (separate, together, separate-then-together). Our results show that (1) persona choice shapes idea domains, (2) collaboration mode shifts diversity of idea generation, and (3) multi-agent persona-driven brainstorming produces idea depth and cross-domain coverage.

**arXiv ID:** 2512.04488
</details>

<details>
<summary><strong>Towards Ethical Multi-Agent Systems of Large Language Models: A Mechanistic Interpretability Perspective</strong> - Jae Hee Lee, Anne Lauscher, Stefano V. Albrecht - [[pdf]](https://arxiv.org/pdf/2512.04691)</summary>

**Abstract:** Large language models (LLMs) have been widely deployed in various applications, often functioning as autonomous agents that interact with each other in multi-agent systems. While these systems have shown promise in enhancing capabilities and enabling complex tasks, they also pose significant ethical challenges. This position paper outlines a research agenda aimed at ensuring the ethical behavior of multi-agent systems of LLMs (MALMs) from the perspective of mechanistic interpretability. We identify three key research challenges: (i) developing comprehensive evaluation frameworks to assess ethical behavior at individual, interactional, and systemic levels; (ii) elucidating the internal mechanisms that give rise to emergent behaviors through mechanistic interpretability; and (iii) implementing targeted parameter-efficient alignment techniques to steer MALMs towards ethical behaviors without compromising their performance.

**arXiv ID:** 2512.04691
</details>

<details>
<summary><strong>Detecting Perspective Shifts in Multi-agent Systems</strong> - Eric Bridgeford, Hayden Helm - [[pdf]](https://arxiv.org/pdf/2512.05013)</summary>

**Abstract:** Generative models augmented with external tools and update mechanisms (or \textit{agents}) have demonstrated capabilities beyond intelligent prompting of base models. As agent use proliferates, dynamic multi-agent systems have naturally emerged. Recent work has investigated the theoretical and empirical properties of low-dimensional representations of agents based on query responses at a single time point. This paper introduces the Temporal Data Kernel Perspective Space (TDKPS), which jointly embeds agents across time, and proposes several novel hypothesis tests for detecting behavioral change at the agent- and group-level in black-box multi-agent systems. We characterize the empirical properties of our proposed tests, including their sensitivity to key hyperparameters, in simulations motivated by a multi-agent system of evolving digital personas. Finally, we demonstrate via natural experiment that our proposed tests detect changes that correlate sensitively, specifically, and significantly with a real exogenous event. As far as we are aware, TDKPS is the first principled framework for monitoring behavioral dynamics in black-box multi-agent systems -- a critical capability as generative agent deployment continues to scale.

**arXiv ID:** 2512.05013
</details>

<details>
<summary><strong>Towards 6G Native-AI Edge Networks: A Semantic-Aware and Agentic Intelligence Paradigm</strong> - Chenyuan Feng, Anbang Zhang, Geyong Min, Yongming Huang, Tony Q. S. Quek, Xiaohu You - [[pdf]](https://arxiv.org/pdf/2512.04405)</summary>

**Abstract:** The evolution toward sixth-generation wireless systems positions intelligence as a native network capability, fundamentally transforming the design of radio access networks (RANs). Within this vision, Semantic-native communication and agentic intelligence are expected to play central roles. SemCom departs from bit-level fidelity and instead emphasizes task-oriented meaning exchange, enabling compact SC and introducing new performance measures such as semantic fidelity and task success rate. Agentic intelligence endows distributed RAN entities with goal-driven autonomy, reasoning, planning, and multi-agent collaboration, increasingly supported by foundation models and knowledge graphs. In this work, we first introduce the conceptual foundations of SemCom and agentic networking, and discuss why existing AI-driven O-RAN solutions remain largely bit-centric and task-siloed. We then present a unified taxonomy that organizes recent research along three axes: i) semantic abstraction level (symbol/feature/intent/knowledge), ii) agent autonomy and coordination granularity (single-, multi-, and hierarchical-agent), and iii) RAN control placement across PHY/MAC, near-real-time RIC, and non-real-time RIC. Based on this taxonomy, we systematically introduce enabling technologies including task-oriented semantic encoders/decoders, multi-agent reinforcement learning, foundation-model-assisted RAN agents, and knowledge-graph-based reasoning for cross-layer awareness. Representative 6G use cases, such as immersive XR, vehicular V2X, and industrial digital twins, are analyzed to illustrate the semantic-agentic convergence in practice. Finally, we identify open challenges in semantic representation standardization, scalable trustworthy agent coordination, O-RAN interoperability, and energy-efficient AI deployment, and outline research directions toward operational semantic-agentic AI-RAN.

**arXiv ID:** 2512.04405
</details>

<details>
<summary><strong>Semi Centralized Training Decentralized Execution Architecture for Multi Agent Deep Reinforcement Learning in Traffic Signal Control</strong> - Pouria Yazdani, Arash Rezaali, Monireh Abdoos - [[pdf]](https://arxiv.org/pdf/2512.04653)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) has emerged as a promising paradigm for adaptive traffic signal control (ATSC) of multiple intersections. Existing approaches typically follow either a fully centralized or a fully decentralized design. Fully centralized approaches suffer from the curse of dimensionality, and reliance on a single learning server, whereas purely decentralized approaches operate under severe partial observability and lack explicit coordination resulting in suboptimal performance. These limitations motivate region-based MARL, where the network is partitioned into smaller, tightly coupled intersections that form regions, and training is organized around these regions. This paper introduces a Semi-Centralized Training, Decentralized Execution (SEMI-CTDE) architecture for multi intersection ATSC. Within each region, SEMI-CTDE performs centralized training with regional parameter sharing and employs composite state and reward formulations that jointly encode local and regional information. The architecture is highly transferable across different policy backbones and state-reward instantiations. Building on this architecture, we implement two models with distinct design objectives. A multi-perspective experimental analysis of the two implemented SEMI-CTDE-based models covering ablations of the architecture's core elements including rule based and fully decentralized baselines shows that they achieve consistently superior performance and remain effective across a wide range of traffic densities and distributions.

**arXiv ID:** 2512.04653
</details>

<details>
<summary><strong>Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs</strong> - Jinbo Liu, Defu Cao, Yifei Wei, Tianyao Su, Yuan Liang, Yushun Dong, Yue Zhao, Xiyang Hu - [[pdf]](https://arxiv.org/pdf/2512.04668)</summary>

**Abstract:** Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over up to 10 interaction rounds, we quantify leakage as the fraction of ground-truth PII recovered from attacking agent outputs via exact matching. We systematically evaluate six common network topologies (fully connected, ring, chain, binary tree, star, and star-ring), varying agent counts $n\in\{4,5,6\}$, attacker-target placements, and base models. Our findings reveal consistent patterns: fully connected graphs exhibit maximum leakage while chains provide strongest protection; shorter attacker-target graph distance and higher target centrality significantly increase vulnerability; leakage rises sharply in early rounds before plateauing; model choice shifts absolute leakage rates but preserves topology rankings; temporal/locational PII attributes leak more readily than identity credentials or regulated identifiers. These results provide the first systematic mapping from architectural choices to measurable privacy risk, yielding actionable guidance: prefer sparse or hierarchical connectivity, maximize attacker-target separation, limit node degree and network radius, avoid shortcuts bypassing hubs, and implement topology-aware access controls.

**arXiv ID:** 2512.04668
</details>

<details>
<summary><strong>Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents</strong> - Ratnadira Widyasari, Martin Weyssow, Ivana Clairine Irsan, Han Wei Ang, Frank Liauw, Eng Lieh Ouh, Lwin Khin Shar, Hong Jin Kang, David Lo - [[pdf]](https://arxiv.org/pdf/2505.10961)</summary>

**Abstract:** Detecting vulnerabilities in source code remains a critical yet challenging task, especially when benign and vulnerable functions share significant similarities. In this work, we introduce VulTrial, a courtroom-inspired multi-agent framework designed to identify vulnerable code and to provide explanations. It employs four role-specific agents, which are security researcher, code author, moderator, and review board. Using GPT-4o as the base LLM, VulTrial almost doubles the efficacy of prior best-performing baselines. Additionally, we show that role-specific instruction tuning with small quantities of data significantly further boosts VulTrial's efficacy. Our extensive experiments demonstrate the efficacy of VulTrial across different LLMs, including an open-source, in-house-deployable model (LLaMA-3.1-8B), as well as the high quality of its generated explanations and its ability to uncover multiple confirmed zero-day vulnerabilities in the wild.

**arXiv ID:** 2505.10961
</details>

<details>
<summary><strong>Multi-Agent Code Verification via Information Theory</strong> - Shreshth Rajan - [[pdf]](https://arxiv.org/pdf/2511.16708)</summary>

**Abstract:** LLMs generate buggy code: 29.6% of SWE-bench solved patches fail, 62% of BaxBench solutions have vulnerabilities, and existing tools only catch 65% of bugs with 35% false positives. We built CodeX-Verify, a multi-agent system that uses four specialized agents to detect different types of bugs. We prove mathematically that combining agents with different detection patterns finds more bugs than any single agent when the agents look for different problems, using submodularity of mutual information under conditional independence. Measuring agent correlation of rho = 0.05 to 0.25 confirms they detect different bugs. Testing on 99 code samples with verified labels shows our system catches 76.1% of bugs, matching the best existing method (Meta Prompt Testing: 75%) while running faster and without test execution. We tested all 15 agent combinations and found that using multiple agents improves accuracy by 39.7 percentage points (from 32.8% to 72.4%) compared to single agents, with diminishing returns of +14.9pp, +13.5pp, and +11.2pp for agents 2, 3, and 4, validating our theoretical model. The best two-agent combination (Correctness + Performance) reaches 79.3% accuracy. Testing on 300 real patches from Claude Sonnet 4.5 runs in under 200ms per sample, making this practical for production use.

**arXiv ID:** 2511.16708
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning for Intraday Operating Rooms Scheduling under Uncertainty</strong> - Kailiang Liu, Ying Chen, Ralf Borndörfer, Thorsten Koch - [[pdf]](https://arxiv.org/pdf/2512.04918)</summary>

**Abstract:** Intraday surgical scheduling is a multi-objective decision problem under uncertainty-balancing elective throughput, urgent and emergency demand, delays, sequence-dependent setups, and overtime. We formulate the problem as a cooperative Markov game and propose a multi-agent reinforcement learning (MARL) framework in which each operating room (OR) is an agent trained with centralized training and decentralized execution. All agents share a policy trained via Proximal Policy Optimization (PPO), which maps rich system states to actions, while a within-epoch sequential assignment protocol constructs conflict-free joint schedules across ORs. A mixed-integer pre-schedule provides reference starting times for electives; we impose type-specific quadratic delay penalties relative to these references and a terminal overtime penalty, yielding a single reward that captures throughput, timeliness, and staff workload. In simulations reflecting a realistic hospital mix (six ORs, eight surgery types, random urgent and emergency arrivals), the learned policy outperforms six rule-based heuristics across seven metrics and three evaluation subsets, and, relative to an ex post MIP oracle, quantifies optimality gaps. Policy analytics reveal interpretable behavior-prioritizing emergencies, batching similar cases to reduce setups, and deferring lower-value electives. We also derive a suboptimality bound for the sequential decomposition under simplifying assumptions. We discuss limitations-including OR homogeneity and the omission of explicit staffing constraints-and outline extensions. Overall, the approach offers a practical, interpretable, and tunable data-driven complement to optimization for real-time OR scheduling.

**arXiv ID:** 2512.04918
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>SIMA 2: A Generalist Embodied Agent for Virtual Worlds</strong> - SIMA team, Adrian Bolton, Alexander Lerchner, Alexandra Cordell, Alexandre Moufarek, Andrew Bolt, Andrew Lampinen, Anna Mitenkova, Arne Olav Hallingstad, Bojan Vujatovic, Bonnie Li, Cong Lu, Daan Wierstra, Daniel P. Sawyer, Daniel Slater, David Reichert, Davide Vercelli, Demis Hassabis, Drew A. Hudson, Duncan Williams, Ed Hirst, Fabio Pardo, Felix Hill, Frederic Besse, Hannah Openshaw, Harris Chan, Hubert Soyer, Jane X. Wang, Jeff Clune, John Agapiou, John Reid, Joseph Marino, Junkyung Kim, Karol Gregor, Kaustubh Sridhar, Kay McKinney, Laura Kampis, Lei M. Zhang, Loic Matthey, Luyu Wang, Maria Abi Raad, Maria Loks-Thompson, Martin Engelcke, Matija Kecman, Matthew Jackson, Maxime Gazeau, Ollie Purkiss, Oscar Knagg, Peter Stys, Piermaria Mendolicchio, Raia Hadsell, Rosemary Ke, Ryan Faulkner, Sarah Chakera, Satinder Singh Baveja, Shane Legg, Sheleem Kashem, Tayfun Terzi, Thomas Keck, Tim Harley, Tim Scholtes, Tyson Roberts, Volodymyr Mnih, Yulan Liu, Zhengdong Wang, Zoubin Ghahramani - [[pdf]](https://arxiv.org/pdf/2512.04797)</summary>

**Abstract:** We introduce SIMA 2, a generalist embodied agent that understands and acts in a wide variety of 3D virtual worlds. Built upon a Gemini foundation model, SIMA 2 represents a significant step toward active, goal-directed interaction within an embodied environment. Unlike prior work (e.g., SIMA 1) limited to simple language commands, SIMA 2 acts as an interactive partner, capable of reasoning about high-level goals, conversing with the user, and handling complex instructions given through language and images. Across a diverse portfolio of games, SIMA 2 substantially closes the gap with human performance and demonstrates robust generalization to previously unseen environments, all while retaining the base model's core reasoning capabilities. Furthermore, we demonstrate a capacity for open-ended self-improvement: by leveraging Gemini to generate tasks and provide rewards, SIMA 2 can autonomously learn new skills from scratch in a new environment. This work validates a path toward creating versatile and continuously learning agents for both virtual and, eventually, physical worlds.

**arXiv ID:** 2512.04797
</details>

<details>
<summary><strong>Enabling Ethical AI: A case study in using Ontological Context for Justified Agentic AI Decisions</strong> - Liam McGee, James Harvey, Lucy Cull, Andreas Vermeulen, Bart-Floris Visscher, Malvika Sharan - [[pdf]](https://arxiv.org/pdf/2512.04822)</summary>

**Abstract:** In this preprint, we present A collaborative human-AI approach to building an inspectable semantic layer for Agentic AI. AI agents first propose candidate knowledge structures from diverse data sources; domain experts then validate, correct, and extend these structures, with their feedback used to improve subsequent models. Authors show how this process captures tacit institutional knowledge, improves response quality and efficiency, and mitigates institutional amnesia. We argue for a shift from post-hoc explanation to justifiable Agentic AI, where decisions are grounded in explicit, inspectable evidence and reasoning accessible to both experts and non-specialists.

**arXiv ID:** 2512.04822
</details>

<details>
<summary><strong>Chameleon: Adaptive Adversarial Agents for Scaling-Based Visual Prompt Injection in Multimodal AI Systems</strong> - M Zeeshan, Saud Satti - [[pdf]](https://arxiv.org/pdf/2512.04895)</summary>

**Abstract:** Multimodal Artificial Intelligence (AI) systems, particularly Vision-Language Models (VLMs), have become integral to critical applications ranging from autonomous decision-making to automated document processing. As these systems scale, they rely heavily on preprocessing pipelines to handle diverse inputs efficiently. However, this dependency on standard preprocessing operations, specifically image downscaling, creates a significant yet often overlooked security vulnerability. While intended for computational optimization, scaling algorithms can be exploited to conceal malicious visual prompts that are invisible to human observers but become active semantic instructions once processed by the model. Current adversarial strategies remain largely static, failing to account for the dynamic nature of modern agentic workflows. To address this gap, we propose Chameleon, a novel, adaptive adversarial framework designed to expose and exploit scaling vulnerabilities in production VLMs. Unlike traditional static attacks, Chameleon employs an iterative, agent-based optimization mechanism that dynamically refines image perturbations based on the target model's real-time feedback. This allows the framework to craft highly robust adversarial examples that survive standard downscaling operations to hijack downstream execution. We evaluate Chameleon against Gemini 2.5 Flash model. Our experiments demonstrate that Chameleon achieves an Attack Success Rate (ASR) of 84.5% across varying scaling factors, significantly outperforming static baseline attacks which average only 32.1%. Furthermore, we show that these attacks effectively compromise agentic pipelines, reducing decision-making accuracy by over 45% in multi-step tasks. Finally, we discuss the implications of these vulnerabilities and propose multi-scale consistency checks as a necessary defense mechanism.

**arXiv ID:** 2512.04895
</details>

<details>
<summary><strong>AudAgent: Automated Auditing of Privacy Policy Compliance in AI Agents</strong> - Ye Zheng, Yidan Hu - [[pdf]](https://arxiv.org/pdf/2511.07441)</summary>

**Abstract:** AI agents can autonomously perform tasks and, often without explicit user consent, collect or disclose users' sensitive local data, which raises serious privacy concerns. Although AI agents' privacy policies describe their intended data practices, there remains limited transparency and accountability about whether runtime behavior matches those policies. To close this gap, we introduce AudAgent, a visual tool that continuously monitors AI agents' data practices in real time and guards compliance with stated privacy policies.
AudAgent consists of four components for automated privacy auditing of AI agents. (i) Policy formalization: a novel cross-LLM voting mechanism to guarantee confidence of the parsed privacy policy model. (ii) Runtime annotation: a lightweight Presidio-based analyzer detects sensitive data and annotates data practices based on the AI agent's context and the privacy policy model. (iii) Compliance auditing: ontology graphs and automata-based checking connect the privacy policy model with runtime annotations, enabling on-the-fly compliance checking. (iv) User interface: an infrastructure-independent implementation visualizes the real-time execution trace of AI agents along with potential privacy policy violations, providing user-friendly transparency and accountability.
We evaluate AudAgent with AI agents built using mainstream frameworks, demonstrating its effectiveness in detecting and visualizing privacy policy violations in real time. Using AudAgent, we also find that most privacy policies omit explicit safeguards for highly sensitive data such as SSNs, whose misuse violates legal requirements, and that many agents do not refuse handling such data via third-party tools, including those controlled by Claude, Gemini, and DeepSeek. AudAgent proactively blocks operations on such data, overriding the agents' original privacy policy and behavior.

**arXiv ID:** 2511.07441
</details>

<details>
<summary><strong>Self-Propelled Agents and Group Social Force</strong> - Peng Wang, Peter Luh - [[pdf]](https://arxiv.org/pdf/2411.09840)</summary>

**Abstract:** Brownian motion have long been studied on a diversity of fields, not only in physics of statistical mechanics, but also in biological models, finance and economic process, and social systems. In the past twenty years, there has been a growing interest in studying the model in self-propelled feature and interaction force such that the model also fits into study of social phenomenon of many individuals. This article will continue with this research trend and especially investigate the model in paradigms for a quantitative description of social and economic process. We mainly discuss a class of collective decision process of Brownian agent/particles, where the stochastic process does not exist in the fluctuation in the traditional Brownian motion, but in selection among several discrete choices. Their decisions interacts with each other in a given social topology. To simplify our discussion the binary choice problem is particularly discussed where each agent only takes an alternative of two choices. Mathematically, we introduce a set of arrays to describe social relationship of agents in a quantitative manner, and the arrays deduce the group social force and opinion dynamics, which are useful to study complex social movement and self-organization phenomena including discrete-choice activities, social groups and de-individualization effect. Such agent-based simulation symbolizes a variety of collective activities in human society, especially in the field of economics and social science.

**arXiv ID:** 2411.09840
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (24 papers)</h2></summary>

<details>
<summary><strong>Towards better dense rewards in Reinforcement Learning Applications</strong> - Shuyuan Zhang - [[pdf]](https://arxiv.org/pdf/2512.04302)</summary>

**Abstract:** Finding meaningful and accurate dense rewards is a fundamental task in the field of reinforcement learning (RL) that enables agents to explore environments more efficiently. In traditional RL settings, agents learn optimal policies through interactions with an environment guided by reward signals. However, when these signals are sparse, delayed, or poorly aligned with the intended task objectives, agents often struggle to learn effectively. Dense reward functions, which provide informative feedback at every step or state transition, offer a potential solution by shaping agent behavior and accelerating learning. Despite their benefits, poorly crafted reward functions can lead to unintended behaviors, reward hacking, or inefficient exploration. This problem is particularly acute in complex or high-dimensional environments where handcrafted rewards are difficult to specify and validate. To address this, recent research has explored a variety of approaches, including inverse reinforcement learning, reward modeling from human preferences, and self-supervised learning of intrinsic rewards. While these methods offer promising directions, they often involve trade-offs between generality, scalability, and alignment with human intent. This proposal explores several approaches to dealing with these unsolved problems and enhancing the effectiveness and reliability of dense reward construction in different RL applications.

**arXiv ID:** 2512.04302
</details>

<details>
<summary><strong>Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning</strong> - Hongye Cao, Zhixin Bai, Ziyue Peng, Boyan Wang, Tianpei Yang, Jing Huo, Yuyao Zhang, Yang Gao - [[pdf]](https://arxiv.org/pdf/2512.04359)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning.

**arXiv ID:** 2512.04359
</details>

<details>
<summary><strong>BiTAgent: A Task-Aware Modular Framework for Bidirectional Coupling between Multimodal Large Language Models and World Models</strong> - Yu-Wei Zhan, Xin Wang, Pengzhe Mao, Tongtong Feng, Ren Wang, Wenwu Zhu - [[pdf]](https://arxiv.org/pdf/2512.04513)</summary>

**Abstract:** Building generalist embodied agents requires a unified system that can interpret multimodal goals, model environment dynamics, and execute reliable actions across diverse real-world tasks. Multimodal large language models (MLLMs) offer strong semantic priors and cross-modal generalization, while world models (WMs) provide actionable latent dynamics for prediction and control. Their combination holds promise for open-ended embodied intelligence, yet introduces two key challenges: (1) establishing a tight coupling between the semantic intent from MLLMs and the dynamic state representations within the WM's latent space, and (2) achieving task-aware adaptability that supports multi-task learning and cross-environment generalization. To address these limitations, we propose BiTAgent, a task-aware dynamic joint framework that enables bidirectional coupling between MLLMs and WMs. BiTAgent establishes two complementary pathways: a forward path that injects MLLM representations into the WM's latent space for semantically guided imagination, and a backward path where WM-generated feedback refines the MLLM's semantic space via dense text-conditioned rewards. This bidirectional interaction is realized through three synergistic components: Task-Aware Dynamic Joint Learning, Task-Aware Behavior Learning, and MLLM-WM Joint Optimization, which together harmonize semantic reasoning and dynamic prediction. Extensive experiments across multi-task and cross-environment settings demonstrate superior stability and generalization over state-of-the-art baselines, marking a step toward open-ended embodied learning.

**arXiv ID:** 2512.04513
</details>

<details>
<summary><strong>GTM: Simulating the World of Tools for AI Agents</strong> - Zhenzhen Ren, Xinpeng Zhang, Zhenxing Qian, Yan Gao, Yu Shi, Shuxin Zheng, Jiyan He - [[pdf]](https://arxiv.org/pdf/2512.04535)</summary>

**Abstract:** The integration of external tools is pivotal for empowering Large Language Model (LLM) agents with real-world capabilities. However, training these agents through direct, continuous interaction with diverse tools is often prohibitively expensive, slow, and introduces additional development and maintenance overhead. To address this challenge, we introduce the Generalist Tool Model (GTM), a 1.5-billion-parameter model that learns to act as a universal tool simulator. With only prompt-level configuration, GTM accesses tool functionalities along with input arguments and generates outputs that faithfully mimic real tool execution, providing a fast and cost-effective solution that eliminates development overhead. To build GTM, we propose the Context-Aware Response Generation (CARG) pipeline, which synthesizes comprehensive training data covering over 20,000 tools across 300 domains including physics, medicine, robotics, and finance. Through this pipeline, GTM learns to produce not only syntactically correct outputs but also logically coherent and contextually appropriate responses. Experiments demonstrate that GTM produces high-quality outputs with strong consistency and reliability. Besides when used in real reinforcement learning scenarios for agent training, GTM exhibits significantly faster simulation speed compared to real tools while maintaining comparable output quality, along with remarkable generalization and domain adaptability. Our results establish GTM as a foundational component for developing future AI agents, enabling efficient and scalable training of tool-augmented systems.

**arXiv ID:** 2512.04535
</details>

<details>
<summary><strong>Are Your Agents Upward Deceivers?</strong> - Dadi Guo, Qingyu Liu, Dongrui Liu, Qihan Ren, Shuai Shao, Tianyi Qiu, Haoran Li, Yi R. Fung, Zhongjie Ba, Juntao Dai, Jiaming Ji, Zhikai Chen, Jialing Tao, Yaodong Yang, Jing Shao, Xia Hu - [[pdf]](https://arxiv.org/pdf/2512.04864)</summary>

**Abstract:** Large Language Model (LLM)-based agents are increasingly used as autonomous subordinates that carry out tasks for users. This raises the question of whether they may also engage in deception, similar to how individuals in human organizations lie to superiors to create a good image or avoid punishment. We observe and define agentic upward deception, a phenomenon in which an agent facing environmental constraints conceals its failure and performs actions that were not requested without reporting. To assess its prevalence, we construct a benchmark of 200 tasks covering five task types and eight realistic scenarios in a constrained environment, such as broken tools or mismatched information sources. Evaluations of 11 popular LLMs reveal that these agents typically exhibit action-based deceptive behaviors, such as guessing results, performing unsupported simulations, substituting unavailable information sources, and fabricating local files. We further test prompt-based mitigation and find only limited reductions, suggesting that it is difficult to eliminate and highlighting the need for stronger mitigation strategies to ensure the safety of LLM-based agents.

**arXiv ID:** 2512.04864
</details>

<details>
<summary><strong>CRAFT-E: A Neuro-Symbolic Framework for Embodied Affordance Grounding</strong> - Zhou Chen, Joe Lin, Carson Bulgin, Sathyanarayanan N. Aakur - [[pdf]](https://arxiv.org/pdf/2512.04231)</summary>

**Abstract:** Assistive robots operating in unstructured environments must understand not only what objects are, but what they can be used for. This requires grounding language-based action queries to objects that both afford the requested function and can be physically retrieved. Existing approaches often rely on black-box models or fixed affordance labels, limiting transparency, controllability, and reliability for human-facing applications. We introduce CRAFT-E, a modular neuro-symbolic framework that composes a structured verb-property-object knowledge graph with visual-language alignment and energy-based grasp reasoning. The system generates interpretable grounding paths that expose the factors influencing object selection and incorporates grasp feasibility as an integral part of affordance inference. We further construct a benchmark dataset with unified annotations for verb-object compatibility, segmentation, and grasp candidates, and deploy the full pipeline on a physical robot. CRAFT-E achieves competitive performance in static scenes, ImageNet-based functional retrieval, and real-world trials involving 20 verbs and 39 objects. The framework remains robust under perceptual noise and provides transparent, component-level diagnostics. By coupling symbolic reasoning with embodied perception, CRAFT-E offers an interpretable and customizable alternative to end-to-end models for affordance-grounded object selection, supporting trustworthy decision-making in assistive robotic systems.

**arXiv ID:** 2512.04231
</details>

<details>
<summary><strong>AutoGuard: A Self-Healing Proactive Security Layer for DevSecOps Pipelines Using Reinforcement Learning</strong> - Praveen Anugula, Avdhesh Kumar Bhardwaj, Navin Chhibber, Rohit Tewari, Sunil Khemka, Piyush Ranjan - [[pdf]](https://arxiv.org/pdf/2512.04368)</summary>

**Abstract:** Contemporary DevSecOps pipelines have to deal with the evolution of security in an ever-continuously integrated and deployed environment. Existing methods,such as rule-based intrusion detection and static vulnerability scanning, are inadequate and unreceptive to changes in the system, causing longer response times and organization needs exposure to emerging attack vectors. In light of the previous constraints, we introduce AutoGuard to the DevSecOps ecosystem, a reinforcement learning (RL)-powered self-healing security framework built to pre-emptively protect DevSecOps environments. AutoGuard is a self-securing security environment that continuously observes pipeline activities for potential anomalies while preemptively remediating the environment. The model observes and reacts based on a policy that is continually learned dynamically over time. The RL agent improves each action over time through reward-based learning aimed at improving the agent's ability to prevent, detect and respond to a security incident in real-time. Testing using simulated ContinuousIntegration / Continuous Deployment (CI/CD) environments showed AutoGuard to successfully improve threat detection accuracy by 22%, reduce mean time torecovery (MTTR) for incidents by 38% and increase overall resilience to incidents as compared to traditional methods.
Keywords- DevSecOps, Reinforcement Learning, Self- Healing Security, Continuous Integration, Automated Threat Mitigation

**arXiv ID:** 2512.04368
</details>

<details>
<summary><strong>When Robots Should Say "I Don't Know": Benchmarking Abstention in Embodied Question Answering</strong> - Tao Wu, Chuhao Zhou, Guangyu Zhao, Haozhi Cao, Yewen Pu, Jianfei Yang - [[pdf]](https://arxiv.org/pdf/2512.04597)</summary>

**Abstract:** Embodied Question Answering (EQA) requires an agent to interpret language, perceive its environment, and navigate within 3D scenes to produce responses. Existing EQA benchmarks assume that every question must be answered, but embodied agents should know when they do not have sufficient information to answer. In this work, we focus on a minimal requirement for EQA agents, abstention: knowing when to withhold an answer. From an initial study of 500 human queries, we find that 32.4% contain missing or underspecified context. Drawing on this initial study and cognitive theories of human communication errors, we derive five representative categories requiring abstention: actionability limitation, referential underspecification, preference dependence, information unavailability, and false presupposition. We augment OpenEQA by having annotators transform well-posed questions into ambiguous variants outlined by these categories. The resulting dataset, AbstainEQA, comprises 1,636 annotated abstention cases paired with 1,636 original OpenEQA instances for balanced evaluation. Evaluating on AbstainEQA, we find that even the best frontier model only attains 42.79% abstention recall, while humans achieve 91.17%. We also find that scaling, prompting, and reasoning only yield marginal gains, and that fine-tuned models overfit to textual cues. Together, these results position abstention as a fundamental prerequisite for reliable interaction in embodied settings and as a necessary basis for effective clarification.

**arXiv ID:** 2512.04597
</details>

<details>
<summary><strong>CARL: Critical Action Focused Reinforcement Learning for Multi-Step Agent</strong> - Leyang Shen, Yang Zhang, Chun Kai Ling, Xiaoyan Zhao, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2512.04949)</summary>

**Abstract:** Agents capable of accomplishing complex tasks through multiple interactions with the environment have emerged as a popular research direction. However, in such multi-step settings, the conventional group-level policy optimization algorithm becomes suboptimal because of its underlying assumption that each action holds equal contribution, which deviates significantly from reality. Our analysis reveals that only a small fraction of actions are critical in determining the final outcome. Building on this insight, we propose CARL, a critical-action-focused reinforcement learning algorithm tailored for multi-step agents. CARL achieves focused training through providing action-level optimization signals for high-criticality actions while excluding low-criticality actions from model update. Extensive experiments demonstrate that CARL achieves both stronger performance and higher efficiency during training and inference across diverse evaluation settings.

**arXiv ID:** 2512.04949
</details>

<details>
<summary><strong>Realizable Abstractions: Near-Optimal Hierarchical Reinforcement Learning</strong> - Roberto Cipollone, Luca Iocchi, Matteo Leonetti - [[pdf]](https://arxiv.org/pdf/2512.04958)</summary>

**Abstract:** The main focus of Hierarchical Reinforcement Learning (HRL) is studying how large Markov Decision Processes (MDPs) can be more efficiently solved when addressed in a modular way, by combining partial solutions computed for smaller subtasks. Despite their very intuitive role for learning, most notions of MDP abstractions proposed in the HRL literature have limited expressive power or do not possess formal efficiency guarantees. This work addresses these fundamental issues by defining Realizable Abstractions, a new relation between generic low-level MDPs and their associated high-level decision processes. The notion we propose avoids non-Markovianity issues and has desirable near-optimality guarantees. Indeed, we show that any abstract policy for Realizable Abstractions can be translated into near-optimal policies for the low-level MDP, through a suitable composition of options. As demonstrated in the paper, these options can be expressed as solutions of specific constrained MDPs. Based on these findings, we propose RARL, a new HRL algorithm that returns compositional and near-optimal low-level policies, taking advantage of the Realizable Abstraction given in the input. We show that RARL is Probably Approximately Correct, it converges in a polynomial number of samples, and it is robust to inaccuracies in the abstraction.

**arXiv ID:** 2512.04958
</details>

<details>
<summary><strong>David vs. Goliath: Can Small Models Win Big with Agentic AI in Hardware Design?</strong> - Shashwat Shankar, Subhranshu Pandey, Innocent Dengkhw Mochahari, Bhabesh Mali, Animesh Basak Chowdhury, Sukanta Bhattacharjee, Chandan Karfa - [[pdf]](https://arxiv.org/pdf/2512.05073)</summary>

**Abstract:** Large Language Model(LLM) inference demands massive compute and energy, making domain-specific tasks expensive and unsustainable. As foundation models keep scaling, we ask: Is bigger always better for hardware design? Our work tests this by evaluating Small Language Models coupled with a curated agentic AI framework on NVIDIA's Comprehensive Verilog Design Problems(CVDP) benchmark. Results show that agentic workflows: through task decomposition, iterative feedback, and correction - not only unlock near-LLM performance at a fraction of the cost but also create learning opportunities for agents, paving the way for efficient, adaptive solutions in complex design tasks.

**arXiv ID:** 2512.05073
</details>

<details>
<summary><strong>Structured Document Translation via Format Reinforcement Learning</strong> - Haiyue Song, Johannes Eschbach-Dymanus, Hour Kaing, Sumire Honda, Hideki Tanaka, Bianka Buschbeck, Masao Utiyama - [[pdf]](https://arxiv.org/pdf/2512.05100)</summary>

**Abstract:** Recent works on structured text translation remain limited to the sentence level, as they struggle to effectively handle the complex document-level XML or HTML structures. To address this, we propose \textbf{Format Reinforcement Learning (FormatRL)}, which employs Group Relative Policy Optimization on top of a supervised fine-tuning model to directly optimize novel structure-aware rewards: 1) TreeSim, which measures structural similarity between predicted and reference XML trees and 2) Node-chrF, which measures translation quality at the level of XML nodes. Additionally, we apply StrucAUC, a fine-grained metric distinguishing between minor errors and major structural failures. Experiments on the SAP software-documentation benchmark demonstrate improvements across six metrics and an analysis further shows how different reward functions contribute to improvements in both structural and translation quality.

**arXiv ID:** 2512.05100
</details>

<details>
<summary><strong>Semantic Soft Bootstrapping: Long Context Reasoning in LLMs without Reinforcement Learning</strong> - Purbesh Mitra, Sennur Ulukus - [[pdf]](https://arxiv.org/pdf/2512.05105)</summary>

**Abstract:** Long context reasoning in large language models (LLMs) has demonstrated enhancement of their cognitive capabilities via chain-of-thought (CoT) inference. Training such models is usually done via reinforcement learning with verifiable rewards (RLVR) in reasoning based problems, like math and programming. However, RLVR is limited by several bottlenecks, such as, lack of dense reward, and inadequate sample efficiency. As a result, it requires significant compute resources in post-training phase. To overcome these limitations, in this work, we propose \textbf{Semantic Soft Bootstrapping (SSB)}, a self-distillation technique, in which the same base language model plays the role of both teacher and student, but receives different semantic contexts about the correctness of its outcome at training time. The model is first prompted with a math problem and several rollouts are generated. From them, the correct and most common incorrect response are filtered, and then provided to the model in context to produce a more robust, step-by-step explanation with a verified final answer. This pipeline automatically curates a paired teacher-student training set from raw problem-answer data, without any human intervention. This generation process also produces a sequence of logits, which is what the student model tries to match in the training phase just from the bare question alone. In our experiment, Qwen2.5-3B-Instruct on GSM8K dataset via parameter-efficient fine-tuning. We then tested its accuracy on MATH500, and AIME2024 benchmarks. Our experiments show a jump of 10.6%, and 10% improvements in accuracy, respectively, over group relative policy optimization (GRPO), which is a commonly used RLVR algorithm. Our code is available at this https URL, and the model, curated dataset is available at this https URL.

**arXiv ID:** 2512.05105
</details>

<details>
<summary><strong>PRO-V-R1: Reasoning Enhanced Programming Agent for RTL Verification</strong> - Yujie Zhao, Zhijing Wu, Boqin Yuan, Zhongming Yu, Hejia Zhang, Wentao Ni, Chia-Tung Ho, Haoxing Ren, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2506.12200)</summary>

**Abstract:** Register-Transfer Level (RTL) verification is a primary bottleneck, consuming 60-70% of development time. While Large Language Models (LLMs) show promise for RTL automation, their performance and research focus have overwhelmingly centered on RTL generation rather than verification. Current methods for RTL verification rely on large scale proprietary models (e.g., GPT-4o) to generate Python-based functional references, incurring a high cost and raising data-privacy risks. To date, an end-to-end open-source solution for autonomous verification remains absent.
We introduce PRO-V-R1, the first trainable open-source agentic framework for autonomous RTL verification. Our contributions are threefold: (1) we design PRO-V sys, a modular agentic system that couples LLM-based reasoning with programmatic tool use for RTL verification; (2) we establish a data construction pipeline that leverages existing RTL datasets to build simulation-validated, expert-level trajectories tailored for supervised fine-tuning (SFT) RTL verification agents; and (3) we implement an efficient reinforcement learning (RL) algorithm that uses verification-specific rewards derived from program-tool feedback to optimize the end-to-end verification workflow. Our empirical evaluation demonstrates PRO-V-R1 achieves a 57.7% functional correctness rate and 34.0% in robust fault detection, significantly outperforming the base model's 25.7% and 21.8% (respectively) from the state-of-the-art (SOTA) automatic verification system. This configuration also outperforms large-scale proprietary LLMs in functional correctness and shows comparable robustness for fault detection.

**arXiv ID:** 2506.12200
</details>

<details>
<summary><strong>Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations</strong> - William Sharpless, Dylan Hirsch, Sander Tonkens, Nikhil Shinde, Sylvia Herbert - [[pdf]](https://arxiv.org/pdf/2506.16016)</summary>

**Abstract:** Hard constraints in reinforcement learning (RL) often degrade policy performance. Lagrangian methods offer a way to blend objectives with constraints, but require intricate reward engineering and parameter tuning. In this work, we extend recent advances that connect Hamilton-Jacobi (HJ) equations with RL to propose two novel value functions for dual-objective satisfaction. Namely, we address: 1) the Reach-Always-Avoid (RAA) problem -- of achieving distinct reward and penalty thresholds -- and 2) the Reach-Reach (RR) problem -- of achieving thresholds of two distinct rewards. In contrast with temporal logic approaches, which typically involve representing an automaton, we derive explicit, tractable Bellman forms in this context via decomposition. Specifically, we prove that the RAA and RR problems may be rewritten as compositions of previously studied HJ-RL problems. We leverage our analysis to propose a variation of Proximal Policy Optimization (DOHJ-PPO), and demonstrate that it produces distinct behaviors from previous approaches, outcompeting a number of baselines in success, safety and speed across a range of tasks for safe-arrival and multi-target achievement.

**arXiv ID:** 2506.16016
</details>

<details>
<summary><strong>OPTIC-ER: A Reinforcement Learning Framework for Real-Time Emergency Response and Equitable Resource Allocation in Underserved African Communities</strong> - Mary Tonwe - [[pdf]](https://arxiv.org/pdf/2508.12943)</summary>

**Abstract:** Public service systems in many African regions suffer from delayed emergency response and spatial inequity, causing avoidable suffering. This paper introduces OPTIC-ER, a reinforcement learning (RL) framework for real-time, adaptive, and equitable emergency response. OPTIC-ER uses an attention-guided actor-critic architecture to manage the complexity of dispatch environments. Its key innovations are a Context-Rich State Vector, encoding action sub-optimality, and a Precision Reward Function, which penalizes inefficiency. Training occurs in a high-fidelity simulation using real data from Rivers State, Nigeria, accelerated by a precomputed Travel Time Atlas. The system is built on the TALS framework (Thin computing, Adaptability, Low-cost, Scalability) for deployment in low-resource settings. In evaluations on 500 unseen incidents, OPTIC-ER achieved a 100.00% optimal action selection rate, confirming its robustness and generalization. Beyond dispatch, the system generates Infrastructure Deficiency Maps and Equity Monitoring Dashboards to guide proactive governance and data-informed development. This work presents a validated blueprint for AI-augmented public services, showing how context-aware RL can bridge the gap between algorithmic decision-making and measurable human impact.

**arXiv ID:** 2508.12943
</details>

<details>
<summary><strong>Omni-AutoThink: Adaptive Multimodal Reasoning via Reinforcement Learning</strong> - Dongchao Yang, Songxiang Liu, Disong Wang, Yuanyuan Wang, Guanglu Wan, Helen Meng - [[pdf]](https://arxiv.org/pdf/2512.03783)</summary>

**Abstract:** Recent advances in Omni models have enabled unified multimodal perception and generation. However, most existing systems still exhibit rigid reasoning behaviors, either overthinking simple problems or failing to reason when necessary. To address this limitation, we propose Omni-AutoThink, a novel adaptive reasoning framework that dynamically adjusts the model's reasoning depth according to task difficulty. Our framework comprises two stages: (1) an Adaptive Supervised Fine-Tuning (Adaptive SFT) stage, which endows the Omni model with fundamental reasoning capability using large-scale reasoning-augmented data, and (2) an Adaptive Reinforcement Learning (Adaptive GRPO) stage, which optimizes reasoning behaviors based on task complexity and reward feedback. We further construct a comprehensive adaptive reasoning benchmark that spans text-only, text-audio, text-visual, and text-audio-visual modalities, providing both training and evaluation splits for multimodal reasoning assessment. Experimental results demonstrate that our proposed framework significantly improves adaptive reasoning performance compared to previous baselines. All benchmark data and code will be publicly released.

**arXiv ID:** 2512.03783
</details>

<details>
<summary><strong>Efficient Preference-Based Reinforcement Learning: Randomized Exploration Meets Experimental Design</strong> - Andreas Schlaginhaufen, Reda Ouhamma, Maryam Kamgarpour - [[pdf]](https://arxiv.org/pdf/2506.09508)</summary>

**Abstract:** We study reinforcement learning from human feedback in general Markov decision processes, where agents learn from trajectory-level preference comparisons. A central challenge in this setting is to design algorithms that select informative preference queries to identify the underlying reward while ensuring theoretical guarantees. We propose a meta-algorithm based on randomized exploration, which avoids the computational challenges associated with optimistic approaches and remains tractable. We establish both regret and last-iterate guarantees under mild reinforcement learning oracle assumptions. To improve query complexity, we introduce and analyze an improved algorithm that collects batches of trajectory pairs and applies optimal experimental design to select informative comparison queries. The batch structure also enables parallelization of preference queries, which is relevant in practical deployment as feedback can be gathered concurrently. Empirical evaluation confirms that the proposed method is competitive with reward-based reinforcement learning while requiring a small number of preference queries.

**arXiv ID:** 2506.09508
</details>

<details>
<summary><strong>LangSAT: A Novel Framework Combining NLP and Reinforcement Learning for SAT Solving</strong> - Muyu Pan, Matthew Walter, Dheeraj Kodakandla, Mahfuza Farooque - [[pdf]](https://arxiv.org/pdf/2512.04374)</summary>

**Abstract:** Our work presents a novel reinforcement learning (RL) based framework to optimize heuristic selection within the conflict-driven clause learning (CDCL) process, improving the efficiency of Boolean satisfia- bility (SAT) solving. The proposed system, LangSAT, bridges the gap between natural language inputs and propositional logic by converting English descriptions into Conjunctive Normal Form (CNF) expressions and solving them using an RL-enhanced CDCL SAT solver. Unlike existing SAT-solving platforms that require CNF as input, LangSAT enables users to input standard English descriptions, making SAT-solving more accessible. The framework comprises two key components: Lang2Logic, which translates English sentences into CNF expressions, and SmartSAT, an RL-based SAT solver. SmartSAT encodes clause-variable relationships as structured graph representations and extracts global features specific to the SAT problem. This implementation provides the RL agent with deeper contextual information, enabling SAT problems to be solved more efficiently. Lang2Logic was evaluated on diverse natural language inputs, processing descriptions up to 450 words. The generated CNFs were solved by SmartSAT, which demonstrated comparable performance to traditional CDCL heuristics with respect to solving time. The combined LangSAT framework offers a more accessible and scalable solution for SAT-solving tasks across reasoning, formal verification, and debugging.

**arXiv ID:** 2512.04374
</details>

<details>
<summary><strong>Data-regularized Reinforcement Learning for Diffusion Models at Scale</strong> - Haotian Ye, Kaiwen Zheng, Jiashu Xu, Puheng Li, Huayu Chen, Jiaqi Han, Sheng Liu, Qinsheng Zhang, Hanzi Mao, Zekun Hao, Prithvijit Chattopadhyay, Dinghao Yang, Liang Feng, Maosheng Liao, Junjie Bai, Ming-Yu Liu, James Zou, Stefano Ermon - [[pdf]](https://arxiv.org/pdf/2512.04332)</summary>

**Abstract:** Aligning generative diffusion models with human preferences via reinforcement learning (RL) is critical yet challenging. Most existing algorithms are often vulnerable to reward hacking, such as quality degradation, over-stylization, or reduced diversity. Our analysis demonstrates that this can be attributed to the inherent limitations of their regularization, which provides unreliable penalties. We introduce Data-regularized Diffusion Reinforcement Learning (DDRL), a novel framework that uses the forward KL divergence to anchor the policy to an off-policy data distribution. Theoretically, DDRL enables robust, unbiased integration of RL with standard diffusion training. Empirically, this translates into a simple yet effective algorithm that combines reward maximization with diffusion loss minimization. With over a million GPU hours of experiments and ten thousand double-blind human evaluations, we demonstrate on high-resolution video generation tasks that DDRL significantly improves rewards while alleviating the reward hacking seen in baselines, achieving the highest human preference and establishing a robust and scalable paradigm for diffusion post-training.

**arXiv ID:** 2512.04332
</details>

<details>
<summary><strong>Long-Horizon Model-Based Offline Reinforcement Learning Without Conservatism</strong> - Tianwei Ni, Esther Derman, Vineet Jain, Vincent Taboga, Siamak Ravanbakhsh, Pierre-Luc Bacon - [[pdf]](https://arxiv.org/pdf/2512.04341)</summary>

**Abstract:** Popular offline reinforcement learning (RL) methods rely on conservatism, either by penalizing out-of-dataset actions or by restricting planning horizons. In this work, we question the universality of this principle and instead revisit a complementary one: a Bayesian perspective. Rather than enforcing conservatism, the Bayesian approach tackles epistemic uncertainty in offline data by modeling a posterior distribution over plausible world models and training a history-dependent agent to maximize expected rewards, enabling test-time generalization. We first illustrate, in a bandit setting, that Bayesianism excels on low-quality datasets where conservatism fails. We then scale the principle to realistic tasks, identifying key design choices, such as layer normalization in the world model and adaptive long-horizon planning, that mitigate compounding error and value overestimation. These yield our practical algorithm, Neubay, grounded in the neutral Bayesian principle. On D4RL and NeoRL benchmarks, Neubay generally matches or surpasses leading conservative algorithms, achieving new state-of-the-art on 7 datasets. Notably, it succeeds with planning horizons of several hundred steps, challenging common belief. Finally, we characterize when Neubay is preferable to conservatism, laying the foundation for a new direction in offline and model-based RL.

**arXiv ID:** 2512.04341
</details>

<details>
<summary><strong>Learning to Orchestrate Agents in Natural Language with the Conductor</strong> - Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, Yujin Tang - [[pdf]](https://arxiv.org/pdf/2512.04388)</summary>

**Abstract:** Powerful large language models (LLMs) from different providers have been expensively trained and finetuned to specialize across varying domains. In this work, we introduce a new kind of Conductor model trained with reinforcement learning to automatically discover powerful coordination strategies among LLMs. Our Conductor learns not only to design targeted communication topologies for effective agent-to-agent collaboration, but also to prompt engineer focused instructions to the LLMs to maximally leverage their individual capabilities. We show that, by learning optimal coordination strategies over pools of powerful worker LLMs, a 7B Conductor achieves significant performance gains beyond any individual worker, attaining state-of-the-art results in challenging reasoning benchmarks, such as LiveCodeBench and GPQA. By training with randomized agent pools, our conductor effectively adapts to arbitrary sets of open- and closed-source agents, meeting any user requirements. Furthermore, allowing the Conductor to select itself as a worker gives rise to recursive topologies, elevating performance with a new form of dynamic test-time scaling through online iterative adaptation. More broadly, ours is among the early work demonstrating language model coordination can be unlocked through RL, where powerful coordination strategies emerge naturally in LLMs through pure end-to-end reward maximization.

**arXiv ID:** 2512.04388
</details>

<details>
<summary><strong>Continuous-time reinforcement learning for optimal switching over multiple regimes</strong> - Yijie Huang, Mengge Li, Xiang Yu, Zhou Zhou - [[pdf]](https://arxiv.org/pdf/2512.04697)</summary>

**Abstract:** This paper studies the continuous-time reinforcement learning (RL) for optimal switching problems across multiple regimes. We consider a type of exploratory formulation under entropy regularization where the agent randomizes both the timing of switches and the selection of regimes through the generator matrix of an associated continuous-time finite-state Markov chain. We establish the well-posedness of the associated system of Hamilton-Jacobi-Bellman (HJB) equations and provide a characterization of the optimal policy. The policy improvement and the convergence of the policy iterations are rigorously established by analyzing the system of equations. We also show the convergence of the value function in the exploratory formulation towards the value function in the classical formulation as the temperature parameter vanishes. Finally, a reinforcement learning algorithm is devised and implemented by invoking the policy evaluation based on the martingale characterization. Our numerical examples with the aid of neural networks illustrate the effectiveness of the proposed RL algorithm.

**arXiv ID:** 2512.04697
</details>

<details>
<summary><strong>PPL: Point Cloud Supervised Proprioceptive Locomotion Reinforcement Learning for Legged Robots in Crawl Spaces</strong> - Bida Ma, Nuo Xu, Chenkun Qi, Xin Liu, Yule Mo, Jinkai Wang, Chunpeng Lu - [[pdf]](https://arxiv.org/pdf/2508.09950)</summary>

**Abstract:** Legged locomotion in constrained spaces (called crawl spaces) is challenging. In crawl spaces, current proprioceptive locomotion learning methods are difficult to achieve traverse because only ground features are inferred. In this study, a point cloud supervised RL framework for proprioceptive locomotion in crawl spaces is proposed. A state estimation network is designed to estimate the robot's collision states as well as ground and spatial features for locomotion. A point cloud feature extraction method is proposed to supervise the state estimation network. The method uses representation of the point cloud in polar coordinate frame and MLPs for efficient feature extraction. Experiments demonstrate that, compared with existing methods, our method exhibits faster iteration time in the training and more agile locomotion in crawl spaces. This study enhances the ability of legged robots to traverse constrained spaces without requiring exteroceptive sensors.

**arXiv ID:** 2508.09950
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
