# Agent arXiv Daily

**Last Updated:** 2026-02-06 03:40:07

**Total Papers:** 92

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (12 papers)</h2></summary>

<details>
<summary><strong>Clinical Validation of Medical-based Large Language Model Chatbots on Ophthalmic Patient Queries with LLM-based Evaluation</strong> - Ting Fang Tan, Kabilan Elangovan, Andreas Pollreisz, Kevin Bryan Dy, Wei Yan Ng, Joy Le Yi Wong, Jin Liyuan, Chrystie Quek Wan Ning, Ashley Shuen Ying Hong, Arun James Thirunavukarasu, Shelley Yin-His Chang, Jie Yao, Dylan Hong, Wang Zhaoran, Amrita Gupta, Daniel SW Ting - [[pdf]](https://arxiv.org/pdf/2602.05381)</summary>

**Abstract:** Domain specific large language models are increasingly used to support patient education, triage, and clinical decision making in ophthalmology, making rigorous evaluation essential to ensure safety and accuracy. This study evaluated four small medical LLMs Meerkat-7B, BioMistral-7B, OpenBioLLM-8B, and MedLLaMA3-v20 in answering ophthalmology related patient queries and assessed the feasibility of LLM based evaluation against clinician grading. In this cross sectional study, 180 ophthalmology patient queries were answered by each model, generating 2160 responses. Models were selected for parameter sizes under 10 billion to enable resource efficient deployment. Responses were evaluated by three ophthalmologists of differing seniority and by GPT-4-Turbo using the S.C.O.R.E. framework assessing safety, consensus and context, objectivity, reproducibility, and explainability, with ratings assigned on a five point Likert scale. Agreement between LLM and clinician grading was assessed using Spearman rank correlation, Kendall tau statistics, and kernel density estimate analyses. Meerkat-7B achieved the highest performance with mean scores of 3.44 from Senior Consultants, 4.08 from Consultants, and 4.18 from Residents. MedLLaMA3-v20 performed poorest, with 25.5 percent of responses containing hallucinations or clinically misleading content, including fabricated terminology. GPT-4-Turbo grading showed strong alignment with clinician assessments overall, with Spearman rho of 0.80 and Kendall tau of 0.67, though Senior Consultants graded more conservatively. Overall, medical LLMs demonstrated potential for safe ophthalmic question answering, but gaps remained in clinical depth and consensus, supporting the feasibility of LLM based evaluation for large scale benchmarking and the need for hybrid automated and clinician review frameworks to guide safe clinical deployment.

**arXiv ID:** 2602.05381
</details>

<details>
<summary><strong>Emulating Aggregate Human Choice Behavior and Biases with GPT Conversational Agents</strong> - Stephen Pilli, Vivek Nallur - [[pdf]](https://arxiv.org/pdf/2602.05597)</summary>

**Abstract:** Cognitive biases often shape human decisions. While large language models (LLMs) have been shown to reproduce well-known biases, a more critical question is whether LLMs can predict biases at the individual level and emulate the dynamics of biased human behavior when contextual factors, such as cognitive load, interact with these biases. We adapted three well-established decision scenarios into a conversational setting and conducted a human experiment (N=1100). Participants engaged with a chatbot that facilitates decision-making through simple or complex dialogues. Results revealed robust biases. To evaluate how LLMs emulate human decision-making under similar interactive conditions, we used participant demographics and dialogue transcripts to simulate these conditions with LLMs based on GPT-4 and GPT-5. The LLMs reproduced human biases with precision. We found notable differences between models in how they aligned human behavior. This has important implications for designing and evaluating adaptive, bias-aware LLM-based AI systems in interactive contexts.

**arXiv ID:** 2602.05597
</details>

<details>
<summary><strong>Graph-based Agent Memory: Taxonomy, Techniques, and Applications</strong> - Chang Yang, Chuang Zhou, Yilin Xiao, Su Dong, Luyao Zhuang, Yujing Zhang, Zhu Wang, Zijin Hong, Zheng Yuan, Zhishang Xiang, Shengyuan Chen, Huachi Zhou, Qinggang Zhang, Ninghao Liu, Jinsong Su, Xinrun Wang, Yi Chang, Xiao Huang - [[pdf]](https://arxiv.org/pdf/2602.05665)</summary>

**Abstract:** Memory emerges as the core module in the Large Language Model (LLM)-based agents for long-horizon complex tasks (e.g., multi-turn dialogue, game playing, scientific discovery), where memory can enable knowledge accumulation, iterative reasoning and self-evolution. Among diverse paradigms, graph stands out as a powerful structure for agent memory due to the intrinsic capabilities to model relational dependencies, organize hierarchical information, and support efficient retrieval. This survey presents a comprehensive review of agent memory from the graph-based perspective. First, we introduce a taxonomy of agent memory, including short-term vs. long-term memory, knowledge vs. experience memory, non-structural vs. structural memory, with an implementation view of graph-based memory. Second, according to the life cycle of agent memory, we systematically analyze the key techniques in graph-based agent memory, covering memory extraction for transforming the data into the contents, storage for organizing the data efficiently, retrieval for retrieving the relevant contents from memory to support reasoning, and evolution for updating the contents in the memory. Third, we summarize the open-sourced libraries and benchmarks that support the development and evaluation of self-evolving agent memory. We also explore diverse application scenarios. Finally, we identify critical challenges and future research directions. This survey aims to offer actionable insights to advance the development of more efficient and reliable graph-based agent memory systems. All the related resources, including research papers, open-source data, and projects, are collected for the community in this https URL.

**arXiv ID:** 2602.05665
</details>

<details>
<summary><strong>Agent2Agent Threats in Safety-Critical LLM Assistants: A Human-Centric Taxonomy</strong> - Lukas Stappen, Ahmet Erkan Turan, Johann Hagerer, Georg Groh - [[pdf]](https://arxiv.org/pdf/2602.05877)</summary>

**Abstract:** The integration of Large Language Model (LLM)-based conversational agents into vehicles creates novel security challenges at the intersection of agentic AI, automotive safety, and inter-agent communication. As these intelligent assistants coordinate with external services via protocols such as Google's Agent-to-Agent (A2A), they establish attack surfaces where manipulations can propagate through natural language payloads, potentially causing severe consequences ranging from driver distraction to unauthorized vehicle control. Existing AI security frameworks, while foundational, lack the rigorous "separation of concerns" standard in safety-critical systems engineering by co-mingling the concepts of what is being protected (assets) with how it is attacked (attack paths). This paper addresses this methodological gap by proposing a threat modeling framework called AgentHeLLM (Agent Hazard Exploration for LLM Assistants) that formally separates asset identification from attack path analysis. We introduce a human-centric asset taxonomy derived from harm-oriented "victim modeling" and inspired by the Universal Declaration of Human Rights, and a formal graph-based model that distinguishes poison paths (malicious data propagation) from trigger paths (activation actions). We demonstrate the framework's practical applicability through an open-source attack path suggestion tool AgentHeLLM Attack Path Generator that automates multi-stage threat discovery using a bi-level search strategy.

**arXiv ID:** 2602.05877
</details>

<details>
<summary><strong>Spider-Sense: Intrinsic Risk Sensing for Efficient Agent Defense with Hierarchical Adaptive Screening</strong> - Zhenxiong Yu, Zhi Yang, Zhiheng Jin, Shuhe Wang, Heng Zhang, Yanlin Fei, Lingfeng Zeng, Fangqi Lou, Shuo Zhang, Tu Hu, Jingping Liu, Rongze Chen, Xingyu Zhu, Kunyi Wang, Chaofa Yuan, Xin Guo, Zhaowei Liu, Feipeng Zhang, Jie Huang, Huacan Wang, Ronghao Chen, Liwen Zhang - [[pdf]](https://arxiv.org/pdf/2602.05386)</summary>

**Abstract:** As large language models (LLMs) evolve into autonomous agents, their real-world applicability has expanded significantly, accompanied by new security challenges. Most existing agent defense mechanisms adopt a mandatory checking paradigm, in which security validation is forcibly triggered at predefined stages of the agent lifecycle. In this work, we argue that effective agent security should be intrinsic and selective rather than architecturally decoupled and mandatory. We propose Spider-Sense framework, an event-driven defense framework based on Intrinsic Risk Sensing (IRS), which allows agents to maintain latent vigilance and trigger defenses only upon risk perception. Once triggered, the Spider-Sense invokes a hierarchical defence mechanism that trades off efficiency and precision: it resolves known patterns via lightweight similarity matching while escalating ambiguous cases to deep internal reasoning, thereby eliminating reliance on external models. To facilitate rigorous evaluation, we introduce S$^2$Bench, a lifecycle-aware benchmark featuring realistic tool execution and multi-stage attacks. Extensive experiments demonstrate that Spider-Sense achieves competitive or superior defense performance, attaining the lowest Attack Success Rate (ASR) and False Positive Rate (FPR), with only a marginal latency overhead of 8.3\%.

**arXiv ID:** 2602.05386
</details>

<details>
<summary><strong>AI chatbots versus human healthcare professionals: a systematic review and meta-analysis of empathy in patient care</strong> - Alastair Howcroft, Amber Bennett-Weston, Ahmad Khan, Joseff Griffiths, Simon Gay, Jeremy Howick - [[pdf]](https://arxiv.org/pdf/2602.05628)</summary>

**Abstract:** Background: Empathy is widely recognized for improving patient outcomes, including reduced pain and anxiety and improved satisfaction, and its absence can cause harm. Meanwhile, use of artificial intelligence (AI)-based chatbots in healthcare is rapidly expanding, with one in five general practitioners using generative AI to assist with tasks such as writing letters. Some studies suggest AI chatbots can outperform human healthcare professionals (HCPs) in empathy, though findings are mixed and lack synthesis.
Sources of data: We searched multiple databases for studies comparing AI chatbots using large language models with human HCPs on empathy measures. We assessed risk of bias with ROBINS-I and synthesized findings using random-effects meta-analysis where feasible, whilst avoiding double counting.
Areas of agreement: We identified 15 studies (2023-2024). Thirteen studies reported statistically significantly higher empathy ratings for AI, with only two studies situated in dermatology favouring human responses. Of the 15 studies, 13 provided extractable data and were suitable for pooling. Meta-analysis of those 13 studies, all utilising ChatGPT-3.5/4, showed a standardized mean difference of 0.87 (95% CI, 0.54-1.20) favouring AI (P < .00001), roughly equivalent to a two-point increase on a 10-point scale.
Areas of controversy: Studies relied on text-based assessments that overlook non-verbal cues and evaluated empathy through proxy raters.
Growing points: Our findings indicate that, in text-only scenarios, AI chatbots are frequently perceived as more empathic than human HCPs.
Areas timely for developing research: Future research should validate these findings with direct patient evaluations and assess whether emerging voice-enabled AI systems can deliver similar empathic advantages.

**arXiv ID:** 2602.05628
</details>

<details>
<summary><strong>StagePilot: A Deep Reinforcement Learning Agent for Stage-Controlled Cybergrooming Simulation</strong> - Heajun An, Qi Zhang, Minqian Liu, Xinyi Zhang, Sang Won Lee, Lifu Huang, Pamela J. Wisniewski, Jin-Hee Cho - [[pdf]](https://arxiv.org/pdf/2602.05060)</summary>

**Abstract:** Cybergrooming is an evolving threat to youth, necessitating proactive educational interventions. We propose StagePilot, an offline RL-based dialogue agent that simulates the stage-wise progression of grooming behaviors for prevention training. StagePilot selects conversational stages using a composite reward that balances user sentiment and goal proximity, with transitions constrained to adjacent stages for realism and interpretability. We evaluate StagePilot through LLM-based simulations, measuring stage completion, dialogue efficiency, and emotional engagement. Results show that StagePilot generates realistic and coherent conversations aligned with grooming dynamics. Among tested methods, the IQL+AWAC agent achieves the best balance between strategic planning and emotional coherence, reaching the final stage up to 43% more frequently than baselines while maintaining over 70% sentiment alignment.

**arXiv ID:** 2602.05060
</details>

<details>
<summary><strong>Bifrost: Steering Strategic Trajectories to Bridge Contextual Gaps for Self-Improving Agents</strong> - Quan M. Tran, Zhuo Huang, Wenbin Zhang, Bo Han, Koji Yatani, Masashi Sugiyama, Tongliang Liu - [[pdf]](https://arxiv.org/pdf/2602.05810)</summary>

**Abstract:** Autonomous agents excel in self-improvement through reflection and iterative refinement, which reuse successful task trajectories as in-context examples to assist subsequent reasoning. However, shifting across tasks often introduces a context mismatch. Hence, existing approaches either discard the trajectories or manipulate them using heuristics, leading to a non-negligible fine-tuning cost or unguaranteed performance. To bridge this gap, we reveal a context-trajectory correlation, where shifts of context are highly parallel with shifts of trajectory. Based on this finding, we propose BrIdge contextual gap FoR imprOvised trajectory STeering (Bifrost), a training-free method that leverages context differences to precisely guide the adaptation of previously solved trajectories towards the target task, mitigating the misalignment caused by context shifts. Our trajectory adaptation is conducted at the representation level using agent hidden states, ensuring trajectory transformation accurately aligns with the target context in a shared space. Across diverse benchmarks, Bifrost consistently outperforms existing trajectory reuse and finetuned self-improvement methods, demonstrating that agents can effectively leverage past experiences despite substantial context shifts.

**arXiv ID:** 2602.05810
</details>

<details>
<summary><strong>VLN-Pilot: Large Vision-Language Model as an Autonomous Indoor Drone Operator</strong> - Bessie Dominguez-Dager, Sergio Suescun-Ferrandiz, Felix Escalona, Francisco Gomez-Donoso, Miguel Cazorla - [[pdf]](https://arxiv.org/pdf/2602.05552)</summary>

**Abstract:** This paper introduces VLN-Pilot, a novel framework in which a large Vision-and-Language Model (VLLM) assumes the role of a human pilot for indoor drone navigation. By leveraging the multimodal reasoning abilities of VLLMs, VLN-Pilot interprets free-form natural language instructions and grounds them in visual observations to plan and execute drone trajectories in GPS-denied indoor environments. Unlike traditional rule-based or geometric path-planning approaches, our framework integrates language-driven semantic understanding with visual perception, enabling context-aware, high-level flight behaviors with minimal task-specific engineering. VLN-Pilot supports fully autonomous instruction-following for drones by reasoning about spatial relationships, obstacle avoidance, and dynamic reactivity to unforeseen events. We validate our framework on a custom photorealistic indoor simulation benchmark and demonstrate the ability of the VLLM-driven agent to achieve high success rates on complex instruction-following tasks, including long-horizon navigation with multiple semantic targets. Experimental results highlight the promise of replacing remote drone pilots with a language-guided autonomous agent, opening avenues for scalable, human-friendly control of indoor UAVs in tasks such as inspection, search-and-rescue, and facility monitoring. Our results suggest that VLLM-based pilots may dramatically reduce operator workload while improving safety and mission flexibility in constrained indoor environments.

**arXiv ID:** 2602.05552
</details>

<details>
<summary><strong>SAP-CoPE: Social-Aware Planning using Cooperative Pose Estimation with Infrastructure Sensor Nodes</strong> - Minghao Ning, Yufeng Yang, Shucheng Huang, Jiaming Zhong, Keqi Shu, Chen Sun, Ehsan Hashemi, Amir Khajepour - [[pdf]](https://arxiv.org/pdf/2504.05727)</summary>

**Abstract:** Autonomous driving systems must operate smoothly in human-populated indoor environments, where challenges arise including limited perception and occlusions when relying only on onboard sensors, as well as the need for socially compliant motion planning that accounts for human psychological comfort zones. These factors complicate accurate recognition of human intentions and the generation of comfortable, socially aware trajectories. To address these challenges, we propose SAP-CoPE, an indoor navigation system that integrates cooperative infrastructure with a novel 3D human pose estimation method and a socially-aware model predictive control (MPC)-based motion planner. In the perception module, an optimization problem is formulated to account for uncertainty propagation in the camera projection matrix while enforcing human joint coherence. The proposed method is adaptable to both single- and multi-camera configurations and can incorporate sparse LiDAR point-cloud data. For motion planning, we integrate a psychology inspired personal-space field using the information from estimated human poses into an MPC framework to enhance socially comfort in human-populated environments. Extensive real-world evaluations demonstrate the effectiveness of the proposed approach in generating socially aware trajectories for autonomous systems.

**arXiv ID:** 2504.05727
</details>

<details>
<summary><strong>Metacognitive Demands and Strategies While Using Off-The-Shelf AI Conversational Agents for Health Information</strong> - Shri Harini Ramesh, Foroozan Daneshzand, Babak Rashidi, Shriti Raj, Hariharan Subramonyam, Fateme Rajabiyazdi - [[pdf]](https://arxiv.org/pdf/2602.05111)</summary>

**Abstract:** As Artificial Intelligence (AI) conversational agents become widespread, people are increasingly using them for health information seeking. The use of off-the-shelf conversational agents for health information seeking could place high metacognitive demands (the need for extensive monitoring and control of one's own thought process) on individuals, which could compromise their experience of seeking health information. However, currently, the specific demands that arise while using conversational agents for health information seeking, and the strategies people use to cope with those demands, remain unknown. To address these gaps, we conducted a think-aloud study with 15 participants as they sought health information using our off-the-shelf AI conversational agent. We identified the metacognitive demands such systems impose, the strategies people adopt in response, and propose considerations for designing beyond off-the-shelf interfaces to reduce these demands and support better user experiences and affordances in health information seeking.

**arXiv ID:** 2602.05111
</details>

<details>
<summary><strong>"Having Lunch Now": Understanding How Users Engage with a Proactive Agent for Daily Planning and Self-Reflection</strong> - Adnan Abbas, Caleb Wohn, Arnav Jagtap, Eugenia H Rho, Young-Ho Kim, Sang Won Lee - [[pdf]](https://arxiv.org/pdf/2509.24073)</summary>

**Abstract:** Conversational agents have been studied as tools to scaffold planning and self-reflection for productivity and well-being. While prior work has demonstrated positive outcomes, we still lack a clear understanding of what drives these results and how users behave and communicate with agents that act as coaches rather than assistants. Such understanding is critical for designing interactions in which agents foster meaningful behavioral change. We conducted a 14-day longitudinal study with 12 participants using a proactive agent that initiated regular check-ins to support daily planning and reflection. Our findings reveal diverse interaction patterns: participants accepted or negotiated suggestions, developed shared mental models, reported progress, and at times resisted or disengaged. We also identified problematic aspects of the agent's behavior, including rigidity, premature turn-taking, and overpromising. Our work contributes to understanding how people interact with a proactive, coach-like agent and offers design considerations for facilitating effective behavioral change.

**arXiv ID:** 2509.24073
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (12 papers)</h2></summary>

<details>
<summary><strong>Automatic Cognitive Task Generation for In-Situ Evaluation of Embodied Agents</strong> - Xinyi He, Ying Yang, Chuanjian Fu, Sihan Guo, Songchun Zhu, Lifeng Fan, Zhenliang Zhang, Yujia Peng - [[pdf]](https://arxiv.org/pdf/2602.05249)</summary>

**Abstract:** As general intelligent agents are poised for widespread deployment in diverse households, evaluation tailored to each unique unseen 3D environment has become a critical prerequisite. However, existing benchmarks suffer from severe data contamination and a lack of scene specificity, inadequate for assessing agent capabilities in unseen settings. To address this, we propose a dynamic in-situ task generation method for unseen environments inspired by human cognition. We define tasks through a structured graph representation and construct a two-stage interaction-evolution task generation system for embodied agents (TEA). In the interaction stage, the agent actively interacts with the environment, creating a loop between task execution and generation that allows for continuous task generation. In the evolution stage, task graph modeling allows us to recombine and reuse existing tasks to generate new ones without external data. Experiments across 10 unseen scenes demonstrate that TEA automatically generated 87,876 tasks in two cycles, which human verification confirmed to be physically reasonable and encompassing essential daily cognitive capabilities. Benchmarking SOTA models against humans on our in-situ tasks reveals that models, despite excelling on public benchmarks, perform surprisingly poorly on basic perception tasks, severely lack 3D interaction awareness and show high sensitivity to task types in reasoning. These sobering findings highlight the necessity of in-situ evaluation before deploying agents into real-world human environments.

**arXiv ID:** 2602.05249
</details>

<details>
<summary><strong>PATHWAYS: Evaluating Investigation and Context Discovery in AI Web Agents</strong> - Shifat E. Arman, Syed Nazmus Sakib, Tapodhir Karmakar Taton, Nafiul Haque, Shahrear Bin Amin - [[pdf]](https://arxiv.org/pdf/2602.05354)</summary>

**Abstract:** We introduce PATHWAYS, a benchmark of 250 multi-step decision tasks that test whether web-based agents can discover and correctly use hidden contextual information. Across both closed and open models, agents typically navigate to relevant pages but retrieve decisive hidden evidence in only a small fraction of cases. When tasks require overturning misleading surface-level signals, performance drops sharply to near chance accuracy. Agents frequently hallucinate investigative reasoning by claiming to rely on evidence they never accessed. Even when correct context is discovered, agents often fail to integrate it into their final decision. Providing more explicit instructions improves context discovery but often reduces overall accuracy, revealing a tradeoff between procedural compliance and effective judgement. Together, these results show that current web agent architectures lack reliable mechanisms for adaptive investigation, evidence integration, and judgement override.

**arXiv ID:** 2602.05354
</details>

<details>
<summary><strong>Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks</strong> - Jafar Isbarov, Murat Kantarcioglu - [[pdf]](https://arxiv.org/pdf/2602.05066)</summary>

**Abstract:** As AI agents automate critical workloads, they remain vulnerable to indirect prompt injection (IPI) attacks. Current defenses rely on monitoring protocols that jointly evaluate an agent's Chain-of-Thought (CoT) and tool-use actions to ensure alignment with user intent. We demonstrate that these monitoring-based defenses can be bypassed via a novel Agent-as-a-Proxy attack, where prompt injection attacks treat the agent as a delivery mechanism, bypassing both agent and monitor simultaneously. While prior work on scalable oversight has focused on whether small monitors can supervise large agents, we show that even frontier-scale monitors are vulnerable. Large-scale monitoring models like Qwen2.5-72B can be bypassed by agents with similar capabilities, such as GPT-4o mini and Llama-3.1-70B. On the AgentDojo benchmark, we achieve a high attack success rate against AlignmentCheck and Extract-and-Evaluate monitors under diverse monitoring LLMs. Our findings suggest current monitoring-based agentic defenses are fundamentally fragile regardless of model scale.

**arXiv ID:** 2602.05066
</details>

<details>
<summary><strong>Capture the Flags: Family-Based Evaluation of Agentic LLMs via Semantics-Preserving Transformations</strong> - Shahin Honarvar, Amber Gorzynski, James Lee-Jones, Harry Coppock, Marek Rei, Joseph Ryan, Alastair F. Donaldson - [[pdf]](https://arxiv.org/pdf/2602.05523)</summary>

**Abstract:** Agentic large language models (LLMs) are increasingly evaluated on cybersecurity tasks using capture-the-flag (CTF) benchmarks. However, existing pointwise benchmarks have limited ability to shed light on the robustness and generalisation abilities of agents across alternative versions of the source code. We introduce CTF challenge families, whereby a single CTF is used as the basis for generating a family of semantically-equivalent challenges via semantics-preserving program transformations. This enables controlled evaluation of agent robustness to source code transformations while keeping the underlying exploit strategy fixed. We introduce a new tool, Evolve-CTF, that generates CTF families from Python challenges using a range of transformations. Using Evolve-CTF to derive families from Cybench and Intercode challenges, we evaluate 13 agentic LLM configurations with tool access. We find that models are remarkably robust to intrusive renaming and code insertion-based transformations, but that composed transformations and deeper obfuscation affect performance by requiring more sophisticated use of tools. We also find that enabling explicit reasoning has little effect on solution success rates across challenge families. Our work contributes a valuable technique and tool for future LLM evaluations, and a large dataset characterising the capabilities of current state-of-the-art models in this domain.

**arXiv ID:** 2602.05523
</details>

<details>
<summary><strong>Dynamic Context Adaptation for Consistent Role-Playing Agents with Retrieval-Augmented Generations</strong> - Jeiyoon Park, Yongshin Han, Minseop Kim, Kisu Yang - [[pdf]](https://arxiv.org/pdf/2508.02016)</summary>

**Abstract:** Building role-playing agents (RPAs) that faithfully emulate specific characters remains challenging because collecting character-specific utterances and continually updating model parameters are resource-intensive, making retrieval-augmented generation (RAG) a practical necessity. However, despite the importance of RAG, there has been little research on RAG-based RPAs. For example, we empirically find that when a persona lacks knowledge relevant to a given query, RAG-based RPAs are prone to hallucination, making it challenging to generate accurate responses. In this paper, we propose Amadeus, a training-free framework that can significantly enhance persona consistency even when responding to questions that lie beyond a character's knowledge. In addition, to underpin the development and rigorous evaluation of RAG-based RPAs, we manually construct CharacterRAG, a role-playing dataset that consists of persona documents for 15 distinct fictional characters totaling 976K written characters, and 450 question-answer pairs. We find that our proposed method effectively models not only the knowledge possessed by characters, but also various attributes such as personality.

**arXiv ID:** 2508.02016
</details>

<details>
<summary><strong>The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution</strong> - Chen Qian, Peng Wang, Dongrui Liu, Junyao Yang, Dadi Guo, Ling Tang, Jilin Mei, Qihan Ren, Shuai Shao, Yong Liu, Jie Fu, Jing Shao, Xia Hu - [[pdf]](https://arxiv.org/pdf/2601.15075)</summary>

**Abstract:** Large Language Model (LLM)-based agents are widely used in real-world applications such as customer service, web navigation, and software engineering. As these systems become more autonomous and are deployed at scale, understanding why an agent takes a particular action becomes increasingly important for accountability and governance. However, existing research predominantly focuses on \textit{failure attribution} to localize explicit errors in unsuccessful trajectories, which is insufficient for explaining \textbf{the reason behind agent behaviors}. To bridge this gap, we propose a novel framework for \textbf{general agentic attribution}, designed to identify the internal factors driving agent actions regardless of the task outcome. Our framework operates hierarchically to manage the complexity of agent interactions. Specifically, at the \textit{component level}, we employ temporal likelihood dynamics to identify critical interaction steps; then at the \textit{sentence level}, we refine this localization using perturbation-based analysis to isolate the specific textual evidence. We validate our framework across a diverse suite of agentic scenarios, including standard tool use and subtle reliability risks like memory-induced bias. Experimental results demonstrate that the proposed framework reliably pinpoints pivotal historical events and sentences behind the agent behavior, offering a critical step toward safer and more accountable agentic systems. Codes are available at this https URL.

**arXiv ID:** 2601.15075
</details>

<details>
<summary><strong>Semi-Autonomous Mathematics Discovery with Gemini: A Case Study on the Erdős Problems</strong> - Tony Feng, Trieu Trinh, Garrett Bingham, Jiwon Kang, Shengtong Zhang, Sang-hyun Kim, Kevin Barreto, Carl Schildkraut, Junehyuk Jung, Jaehyeon Seo, Carlo Pagano, Yuri Chervonyi, Dawsen Hwang, Kaiying Hou, Sergei Gukov, Cheng-Chiang Tsai, Hyunwoo Choi, Youngbeom Jin, Wei-Yuan Li, Hao-An Wu, Ruey-An Shiu, Yu-Sheng Shih, Quoc V. Le, Thang Luong - [[pdf]](https://arxiv.org/pdf/2601.22401)</summary>

**Abstract:** We present a case study in semi-autonomous mathematics discovery, using Gemini to systematically evaluate 700 conjectures labeled 'Open' in Bloom's Erdős Problems database. We employ a hybrid methodology: AI-driven natural language verification to narrow the search space, followed by human expert evaluation to gauge correctness and novelty. We address 13 problems that were marked 'Open' in the database: 5 through seemingly novel autonomous solutions, and 8 through identification of previous solutions in the existing literature. Our findings suggest that the 'Open' status of the problems was through obscurity rather than difficulty. We also identify and discuss issues arising in applying AI to math conjectures at scale, highlighting the difficulty of literature identification and the risk of ''subconscious plagiarism'' by AI. We reflect on the takeaways from AI-assisted efforts on the Erdős Problems.

**arXiv ID:** 2601.22401
</details>

<details>
<summary><strong>SAGE: Benchmarking and Improving Retrieval for Deep Research Agents</strong> - Tiansheng Hu, Yilun Zhao, Canyu Zhang, Arman Cohan, Chen Zhao - [[pdf]](https://arxiv.org/pdf/2602.05975)</summary>

**Abstract:** Deep research agents have emerged as powerful systems for addressing complex queries. Meanwhile, LLM-based retrievers have demonstrated strong capability in following instructions or reasoning. This raises a critical question: can LLM-based retrievers effectively contribute to deep research agent workflows? To investigate this, we introduce SAGE, a benchmark for scientific literature retrieval comprising 1,200 queries across four scientific domains, with a 200,000 paper retrieval this http URL evaluate six deep research agents and find that all systems struggle with reasoning-intensive retrieval. Using DR Tulu as backbone, we further compare BM25 and LLM-based retrievers (i.e., ReasonIR and gte-Qwen2-7B-instruct) as alternative search tools. Surprisingly, BM25 significantly outperforms LLM-based retrievers by approximately 30%, as existing agents generate keyword-oriented sub-queries. To improve performance, we propose a corpus-level test-time scaling framework that uses LLMs to augment documents with metadata and keywords, making retrieval easier for off-the-shelf retrievers. This yields 8% and 2% gains on short-form and open-ended questions, respectively.

**arXiv ID:** 2602.05975
</details>

<details>
<summary><strong>Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair Systems</strong> - Matias Martinez, Xavier Franch - [[pdf]](https://arxiv.org/pdf/2506.17208)</summary>

**Abstract:** The rapid progress in Automated Program Repair (APR) has been driven by advances in AI, particularly large language models (LLMs) and agent-based systems. SWE-Bench is a recent benchmark designed to evaluate LLM-based repair systems using real issues and pull requests mined from 12 popular open-source Python repositories. Its public leaderboards -- SWE-Bench Lite and SWE-Bench Verified -- have become central platforms for tracking progress and comparing solutions. However, because the submission process does not require detailed documentation, the architectural design and origin of many solutions remain unclear. In this paper, we present the first comprehensive study of all submissions to the SWE-Bench Lite (79 entries) and Verified (99 entries) leaderboards, analyzing 80 unique approaches across dimensions such as submitter type, product availability, LLM usage, and system architecture. Our findings reveal the dominance of proprietary LLMs (especially Claude 3.5), the presence of both agentic and non-agentic designs, and a contributor base spanning from individual developers to large tech companies.

**arXiv ID:** 2506.17208
</details>

<details>
<summary><strong>BLITZRANK: Principled Zero-shot Ranking Agents with Tournament Graphs</strong> - Sheshansh Agrawal, Thien Hang Nguyen, Douwe Kiela - [[pdf]](https://arxiv.org/pdf/2602.05448)</summary>

**Abstract:** Large language models have emerged as powerful zero-shot rerankers for retrieval-augmented generation, offering strong generalization without task-specific training. However, existing LLM reranking methods either rely on heuristics that fail to fully exploit the information revealed by each ranking decision or are inefficient when they do. We introduce a tournament graph framework that provides a principled foundation for $k$-wise reranking. Our key observation is that each $k$-document comparison reveals a complete tournament of $\binom{k}{2}$ pairwise preferences. These tournaments are aggregated into a global preference graph, whose transitive closure yields many additional orderings without further model invocations. We formalize when a candidate's rank is certifiably determined and design a query schedule that greedily maximizes information gain towards identifying the top-$m$ items. Our framework also gracefully handles non-transitive preferences - cycles induced by LLM judgments - by collapsing them into equivalence classes that yield principled tiered rankings. Empirically, across 14 benchmarks and 5 LLMs, our method achieves Pareto dominance over existing methods: matching or exceeding accuracy while requiring 25-40% fewer tokens than comparable approaches, and 7$\times$ fewer than pairwise methods at near-identical quality.

**arXiv ID:** 2602.05448
</details>

<details>
<summary><strong>ContextBench: A Benchmark for Context Retrieval in Coding Agents</strong> - Han Li, Letian Zhu, Bohan Zhang, Rili Feng, Jiaming Wang, Yue Pan, Earl T. Barr, Sarro Federica, Zhaoyang Chu, He Ye - [[pdf]](https://arxiv.org/pdf/2602.05892)</summary>

**Abstract:** LLM-based coding agents have shown strong performance on automated issue resolution benchmarks, yet existing evaluations largely focus on final task success, providing limited insight into how agents retrieve and use code context during problem solving. We introduce ContextBench, a process-oriented evaluation of context retrieval in coding agents. ContextBench consists of 1,136 issue-resolution tasks from 66 repositories across eight programming languages, each augmented with human-annotated gold contexts. We further implement an automated evaluation framework that tracks agent trajectories and measures context recall, precision, and efficiency throughout issue resolution. Using ContextBench, we evaluate four frontier LLMs and five coding agents. Our results show that sophisticated agent scaffolding yields only marginal gains in context retrieval ("The Bitter Lesson" of coding agents), LLMs consistently favor recall over precision, and substantial gaps exist between explored and utilized context. ContextBench augments existing end-to-end benchmarks with intermediate gold-context metrics that unbox the issue-resolution process. These contexts offer valuable intermediate signals for guiding LLM reasoning in software tasks. Data and code are available at: this https URL.

**arXiv ID:** 2602.05892
</details>

<details>
<summary><strong>Making AI Agents Evaluate Misleading Charts without Nudging</strong> - Swaroop Panda - [[pdf]](https://arxiv.org/pdf/2602.05662)</summary>

**Abstract:** AI agents are increasingly used as low-cost proxies for early visualization evaluation. In an initial study of deliberately flawed charts, we test whether agents spontaneously penalise chart junk and misleading encodings without being prompted to look for errors. Using established scales (BeauVis and PREVis), the agent evaluated visualizations containing decorative clutter, manipulated axes, and distorted proportional cues. The ratings of aesthetic appeal and perceived readability often remained relatively high even when graphical integrity was compromised. These results suggest that un-nudged AI agent evaluation may underweight integrity-related defects unless such checks are explicitly elicited.

**arXiv ID:** 2602.05662
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Towards Reducible Uncertainty Modeling for Reliable Large Language Model Agents</strong> - Changdae Oh, Seongheon Park, To Eun Kim, Jiatong Li, Wendi Li, Samuel Yeh, Xuefeng Du, Hamed Hassani, Paul Bogdan, Dawn Song, Sharon Li - [[pdf]](https://arxiv.org/pdf/2602.05073)</summary>

**Abstract:** Uncertainty quantification (UQ) for large language models (LLMs) is a key building block for safety guardrails of daily LLM applications. Yet, even as LLM agents are increasingly deployed in highly complex tasks, most UQ research still centers on single-turn question-answering. We argue that UQ research must shift to realistic settings with interactive agents, and that a new principled framework for agent UQ is needed. This paper presents the first general formulation of agent UQ that subsumes broad classes of existing UQ setups. Under this formulation, we show that prior works implicitly treat LLM UQ as an uncertainty accumulation process, a viewpoint that breaks down for interactive agents in an open world. In contrast, we propose a novel perspective, a conditional uncertainty reduction process, that explicitly models reducible uncertainty over an agent's trajectory by highlighting "interactivity" of actions. From this perspective, we outline a conceptual framework to provide actionable guidance for designing UQ in LLM agent setups. Finally, we conclude with practical implications of the agent UQ in frontier LLM development and domain-specific applications, as well as open remaining problems.

**arXiv ID:** 2602.05073
</details>

<details>
<summary><strong>SocialVeil: Probing Social Intelligence of Language Agents under Communication Barriers</strong> - Keyang Xuan, Pengda Wang, Chongrui Ye, Haofei Yu, Tal August, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2602.05115)</summary>

**Abstract:** Large language models (LLMs) are increasingly evaluated in interactive environments to test their social intelligence. However, existing benchmarks often assume idealized communication between agents, limiting our ability to diagnose whether LLMs can maintain and repair interactions in more realistic, imperfect settings. To close this gap, we present \textsc{SocialVeil}, a social learning environment that can simulate social interaction under cognitive-difference-induced communication barriers. Grounded in a systematic literature review of communication challenges in human interaction, \textsc{SocialVeil} introduces three representative types of such disruption, \emph{semantic vagueness}, \emph{sociocultural mismatch}, and \emph{emotional interference}. We also introduce two barrier-aware evaluation metrics, \emph{unresolved confusion} and \emph{mutual understanding}, to evaluate interaction quality under impaired communication. Experiments across 720 scenarios and four frontier LLMs show that barriers consistently impair performance, with mutual understanding reduced by over 45\% on average, and confusion elevated by nearly 50\%. Human evaluations validate the fidelity of these simulated barriers (ICC$\approx$0.78, Pearson r$\approx$0.80). We further demonstrate that adaptation strategies (Repair Instruction and Interactive learning) only have a modest effect far from barrier-free performance. This work takes a step toward bringing social interaction environments closer to real-world communication, opening opportunities for exploring the social intelligence of LLM agents.

**arXiv ID:** 2602.05115
</details>

<details>
<summary><strong>Structured Context Engineering for File-Native Agentic Systems: Evaluating Schema Accuracy, Format Effectiveness, and Multi-File Navigation at Scale</strong> - Damon McMillan - [[pdf]](https://arxiv.org/pdf/2602.05447)</summary>

**Abstract:** Large Language Model agents increasingly operate external systems through programmatic interfaces, yet practitioners lack empirical guidance on how to structure the context these agents consume. Using SQL generation as a proxy for programmatic agent operations, we present a systematic study of context engineering for structured data, comprising 9,649 experiments across 11 models, 4 formats (YAML, Markdown, JSON, Token-Oriented Object Notation [TOON]), and schemas ranging from 10 to 10,000 tables.
Our findings challenge common assumptions. First, architecture choice is model-dependent: file-based context retrieval improves accuracy for frontier-tier models (Claude, GPT, Gemini; +2.7%, p=0.029) but shows mixed results for open source models (aggregate -7.7%, p<0.001), with deficits varying substantially by model. Second, format does not significantly affect aggregate accuracy (chi-squared=2.45, p=0.484), though individual models, particularly open source, exhibit format-specific sensitivities. Third, model capability is the dominant factor, with a 21 percentage point accuracy gap between frontier and open source tiers that dwarfs any format or architecture effect. Fourth, file-native agents scale to 10,000 tables through domain-partitioned schemas while maintaining high navigation accuracy. Fifth, file size does not predict runtime efficiency: compact formats can consume significantly more tokens at scale due to format-unfamiliar search patterns.
These findings provide practitioners with evidence-based guidance for deploying LLM agents on structured systems, demonstrating that architectural decisions should be tailored to model capability rather than assuming universal best practices.

**arXiv ID:** 2602.05447
</details>

<details>
<summary><strong>Learning to Inject: Automated Prompt Injection via Reinforcement Learning</strong> - Xin Chen, Jie Zhang, Florian Tramer - [[pdf]](https://arxiv.org/pdf/2602.05746)</summary>

**Abstract:** Prompt injection is one of the most critical vulnerabilities in LLM agents; yet, effective automated attacks remain largely unexplored from an optimization perspective. Existing methods heavily depend on human red-teamers and hand-crafted prompts, limiting their scalability and adaptability. We propose AutoInject, a reinforcement learning framework that generates universal, transferable adversarial suffixes while jointly optimizing for attack success and utility preservation on benign tasks. Our black-box method supports both query-based optimization and transfer attacks to unseen models and tasks. Using only a 1.5B parameter adversarial suffix generator, we successfully compromise frontier systems including GPT 5 Nano, Claude Sonnet 3.5, and Gemini 2.5 Flash on the AgentDojo benchmark, establishing a stronger baseline for automated prompt injection research.

**arXiv ID:** 2602.05746
</details>

<details>
<summary><strong>DARWIN: Dynamic Agentically Rewriting Self-Improving Network</strong> - Henry Jiang - [[pdf]](https://arxiv.org/pdf/2602.05848)</summary>

**Abstract:** DARWIN is an evolutionary GPT model, utilizing a genetic-algorithm like optimization structure with several independent GPT agents being trained individually using unique training code. Each iteration, the GPT models are prompted to modify the training code of one another in an attempt to improve their performance in a mutation-like manner, and the best GPT agents are then benchmarked and selected for the next iteration by genetic algorithm. For demonstration purposes and due to budget and time constraints, OpenAI API is used to prompt training code improvements and the nanoGPT framework is used as the training code. DARWIN also utilizes persistent JSON-based memory files to track previous reasoning and changes to code to correlate with improvement to model performance. and a bidirectional interface for HITL intervention allowing the model to request upgrades such as additional datasets, training scripts, and restructuring of file hierarchies. In experiments, DARWIN achieved a 1.26 percent improvement in model FLOPS utilization (MFU) and a 2.07 percent improvement to perplexity in 5 iterations of training over baseline configurations, demonstrating promising capabilities as a foundation for scaling evolutionary GPT training.

**arXiv ID:** 2602.05848
</details>

<details>
<summary><strong>Exploring Silicon-Based Societies: An Early Study of the Moltbook Agent Community</strong> - Yu-Zheng Lin, Bono Po-Jen Shih, Hsuan-Ying Alessandra Chien, Shalaka Satam, Jesus Horacio Pacheco, Sicong Shao, Soheil Salehi, Pratik Satam - [[pdf]](https://arxiv.org/pdf/2602.02613)</summary>

**Abstract:** The rapid emergence of autonomous large language model agents has given rise to persistent, large-scale agent ecosystems whose collective behavior cannot be adequately understood through anecdotal observation or small-scale simulation. This paper introduces data-driven silicon sociology as a systematic empirical framework for studying social structure formation among interacting artificial agents. We present a pioneering large-scale data mining investigation of an in-the-wild agent society by analyzing Moltbook, a social platform designed primarily for agent-to-agent interaction. At the time of study, Moltbook hosted over 150,000 registered autonomous agents operating across thousands of agent-created sub-communities. Using programmatic and non-intrusive data acquisition, we collected and analyzed the textual descriptions of 12,758 submolts, which represent proactive sub-community partitioning activities within the ecosystem. Treating agent-authored descriptions as first-class observational artifacts, we apply rigorous preprocessing, contextual embedding, and unsupervised clustering techniques to uncover latent patterns of thematic organization and social space structuring. The results show that autonomous agents systematically organize collective space through reproducible patterns spanning human-mimetic interests, silicon-centric self-reflection, and early-stage economic and coordination behaviors. Rather than relying on predefined sociological taxonomies, these structures emerge directly from machine-generated data traces. This work establishes a methodological foundation for data-driven silicon sociology and demonstrates that data mining techniques can provide a powerful lens for understanding the organization and evolution of large autonomous agent societies.

**arXiv ID:** 2602.02613
</details>

<details>
<summary><strong>Invisible Walls in Cities: Designing LLM Agent to Predict Urban Segregation Experience with Social Media Content</strong> - Bingbing Fan, Lin Chen, Songwei Li, Jian Yuan, Fengli Xu, Pan Hui, Yong Li - [[pdf]](https://arxiv.org/pdf/2503.04773)</summary>

**Abstract:** Understanding experienced segregation in urban daily life is crucial for addressing societal inequalities and fostering inclusivity. The abundance of user-generated reviews on social media encapsulates nuanced perceptions and feelings associated with different places, offering rich insights into segregation. However, leveraging this data poses significant challenges due to its vast volume, ambiguity, and confluence of diverse perspectives. To tackle these challenges, we propose a novel Large Language Model (LLM) agent to automate online review mining for segregation prediction. Specifically, we propose a reflective LLM coder to digest social media content into insights consistent with real-world feedback, and eventually produce a codebook capturing key dimensions that signal segregation experience, such as cultural resonance and appeal, accessibility and convenience, and community engagement and local involvement. Guided by the codebook, LLMs can generate both informative review summaries and ratings for segregation prediction. Moreover, we design a REasoning-and-EMbedding (RE'EM) framework, which combines the reasoning and embedding capabilities of language models to integrate multi-channel features for segregation prediction.
Experiments on real-world data demonstrate that our agent substantially improves prediction accuracy, with a 22.79% elevation in R$^{2}$ and a 9.33% reduction in MSE. The derived codebook is generalizable across three different cities, consistently improving prediction accuracy. Moreover, our user study confirms that the codebook-guided summaries provide cognitive gains for human participants in perceiving places of interest (POIs)' social inclusiveness. Our study marks an important step toward understanding implicit social barriers and inequalities, demonstrating the great potential of promoting social inclusiveness with Web technology.

**arXiv ID:** 2503.04773
</details>

<details>
<summary><strong>Position: The Real Barrier to LLM Agent Usability is Agentic ROI</strong> - Weiwen Liu, Jiarui Qin, Xu Huang, Xingshan Zeng, Yunjia Xi, Jianghao Lin, Chuhan Wu, Yasheng Wang, Lifeng Shang, Ruiming Tang, Defu Lian, Yong Yu, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2505.17767)</summary>

**Abstract:** Large Language Model (LLM) agents represent a promising shift in human-AI interaction, moving beyond passive prompt-response systems to autonomous agents capable of reasoning, planning, and goal-directed action. While LLM agents are technically capable of performing a broad range of tasks, not all of these capabilities translate into meaningful usability. This position paper argues that the central question for LLM agent usability is no longer whether a task can be automated, but whether it delivers sufficient Agentic Return on Investment (Agentic ROI). Agentic ROI reframes evaluation from raw performance to a holistic, utility-driven perspective, guiding when, where, and for whom LLM agents should be deployed. Despite widespread application in high-ROI tasks like coding and scientific research, we identify a critical usability gap in mass-market, everyday applications. To address this, we propose a zigzag developmental trajectory: first scaling up to improve information gain and time savings, then scaling down to reduce cost. We present a strategic roadmap across these phases to make LLM agents truly usable, accessible, and scalable in real-world applications.

**arXiv ID:** 2505.17767
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (20 papers)</h2></summary>

<details>
<summary><strong>GAMMS: Graph based Adversarial Multiagent Modeling Simulator</strong> - Rohan Patil, Jai Malegaonkar, Xiao Jiang, Andre Dion, Gaurav S. Sukhatme, Henrik I. Christensen - [[pdf]](https://arxiv.org/pdf/2602.05105)</summary>

**Abstract:** As intelligent systems and multi-agent coordination become increasingly central to real-world applications, there is a growing need for simulation tools that are both scalable and accessible. Existing high-fidelity simulators, while powerful, are often computationally expensive and ill-suited for rapid prototyping or large-scale agent deployments. We present GAMMS (Graph based Adversarial Multiagent Modeling Simulator), a lightweight yet extensible simulation framework designed to support fast development and evaluation of agent behavior in environments that can be represented as graphs. GAMMS emphasizes five core objectives: scalability, ease of use, integration-first architecture, fast visualization feedback, and real-world grounding. It enables efficient simulation of complex domains such as urban road networks and communication systems, supports integration with external tools (e.g., machine learning libraries, planning solvers), and provides built-in visualization with minimal configuration. GAMMS is agnostic to policy type, supporting heuristic, optimization-based, and learning-based agents, including those using large language models. By lowering the barrier to entry for researchers and enabling high-performance simulations on standard hardware, GAMMS facilitates experimentation and innovation in multi-agent systems, autonomous planning, and adversarial modeling. The framework is open-source and available at this https URL

**arXiv ID:** 2602.05105
</details>

<details>
<summary><strong>PieArena: Frontier Language Agents Achieve MBA-Level Negotiation Performance and Reveal Novel Behavioral Differences</strong> - Chris Zhu, Sasha Cui, Will Sanok Dufallo, Runzhi Jin, Zhen Xu, Linjun Zhang, Daylian Cain - [[pdf]](https://arxiv.org/pdf/2602.05302)</summary>

**Abstract:** We present an in-depth evaluation of LLMs' ability to negotiate, a central business task that requires strategic reasoning, theory of mind, and economic value creation. To do so, we introduce PieArena, a large-scale negotiation benchmark grounded in multi-agent interactions over realistic scenarios drawn from an MBA negotiation course at an elite business school. We find systematic evidence of AGI-level performance in which a representative frontier agent (GPT-5) matches or outperforms trained business-school students, despite a semester of general negotiation instruction and targeted coaching immediately prior to the task. We further study the effects of joint-intentionality agentic scaffolding and find asymmetric gains, with large improvements for mid- and lower-tier LMs and diminishing returns for frontier LMs. Beyond deal outcomes, PieArena provides a multi-dimensional negotiation behavioral profile, revealing novel cross-model heterogeneity, masked by deal-outcome-only benchmarks, in deception, computation accuracy, instruction compliance, and perceived reputation. Overall, our results suggest that frontier language agents are already intellectually and psychologically capable of deployment in high-stakes economic settings, but deficiencies in robustness and trustworthiness remain open challenges.

**arXiv ID:** 2602.05302
</details>

<details>
<summary><strong>H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration</strong> - Jun-Min Lee, Meong Hi Son, Edward Choi - [[pdf]](https://arxiv.org/pdf/2602.05407)</summary>

**Abstract:** Hospital administration departments handle a wide range of operational tasks and, in large hospitals, process over 10,000 requests per day, driving growing interest in LLM-based automation. However, prior work has focused primarily on patient--physician interactions or isolated administrative subtasks, failing to capture the complexity of real administrative workflows. To address this gap, we propose H-AdminSim, a comprehensive end-to-end simulation framework that combines realistic data generation with multi-agent-based simulation of hospital administrative workflows. These tasks are quantitatively evaluated using detailed rubrics, enabling systematic comparison of LLMs. Through FHIR integration, H-AdminSim provides a unified and interoperable environment for testing administrative workflows across heterogeneous hospital settings, serving as a standardized testbed for assessing the feasibility and performance of LLM-driven administrative automation.

**arXiv ID:** 2602.05407
</details>

<details>
<summary><strong>M$^2$-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining</strong> - Rui Lv, Juncheng Mo, Tianyi Chu, Chen Rao, Hongyi Jing, Jiajie Teng, Jiafu Chen, Shiqi Zhang, Liangzi Ding, Shuo Fang, Huaizhong Lin, Ziqiang Dang, Chenguang Ma, Lei Zhao - [[pdf]](https://arxiv.org/pdf/2602.05429)</summary>

**Abstract:** Graphical User Interface (GUI) agent is pivotal to advancing intelligent human-computer interaction paradigms. Constructing powerful GUI agents necessitates the large-scale annotation of high-quality user-behavior trajectory data (i.e., intent-trajectory pairs) for training. However, manual annotation methods and current GUI agent data mining approaches typically face three critical challenges: high construction cost, poor data quality, and low data richness. To address these issues, we propose M$^2$-Miner, the first low-cost and automated mobile GUI agent data-mining framework based on Monte Carlo Tree Search (MCTS). For better data mining efficiency and quality, we present a collaborative multi-agent framework, comprising InferAgent, OrchestraAgent, and JudgeAgent for guidance, acceleration, and evaluation. To further enhance the efficiency of mining and enrich intent diversity, we design an intent recycling strategy to extract extra valuable interaction trajectories. Additionally, a progressive model-in-the-loop training strategy is introduced to improve the success rate of data mining. Extensive experiments have demonstrated that the GUI agent fine-tuned using our mined data achieves state-of-the-art performance on several commonly used mobile GUI benchmarks. Our work will be released to facilitate the community research.

**arXiv ID:** 2602.05429
</details>

<details>
<summary><strong>AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions</strong> - Xianyang Liu, Shangding Gu, Dawn Song - [[pdf]](https://arxiv.org/pdf/2602.06008)</summary>

**Abstract:** Large language model (LLM)-based agents are increasingly expected to negotiate, coordinate, and transact autonomously, yet existing benchmarks lack principled settings for evaluating language-mediated economic interaction among multiple agents. We introduce AgenticPay, a benchmark and simulation framework for multi-agent buyer-seller negotiation driven by natural language. AgenticPay models markets in which buyers and sellers possess private constraints and product-dependent valuations, and must reach agreements through multi-round linguistic negotiation rather than numeric bidding alone. The framework supports a diverse suite of over 110 tasks ranging from bilateral bargaining to many-to-many markets, with structured action extraction and metrics for feasibility, efficiency, and welfare. Benchmarking state-of-the-art proprietary and open-weight LLMs reveals substantial gaps in negotiation performance and highlights challenges in long-horizon strategic reasoning, establishing AgenticPay as a foundation for studying agentic commerce and language-based market interaction. Code and dataset are available at the link: this https URL.

**arXiv ID:** 2602.06008
</details>

<details>
<summary><strong>DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning via Semantic Matching</strong> - Yuxing Lu, Yucheng Hu, Xukai Zhao, Jiuxin Cao - [[pdf]](https://arxiv.org/pdf/2602.06039)</summary>

**Abstract:** Multi-agent systems built from prompted large language models can improve multi-round reasoning, yet most existing pipelines rely on fixed, trajectory-wide communication patterns that are poorly matched to the stage-dependent needs of iterative problem solving. We introduce DyTopo, a manager-guided multi-agent framework that reconstructs a sparse directed communication graph at each round. Conditioned on the manager's round goal, each agent outputs lightweight natural-language query (need) and \key (offer) descriptors; DyTopo embeds these descriptors and performs semantic matching, routing private messages only along the induced edges. Across code generation and mathematical reasoning benchmarks and four LLM backbones, DyTopo consistently outperforms over the strongest baseline (avg. +6.2). Beyond accuracy, DyTopo yields an interpretable coordination trace via the evolving graphs, enabling qualitative inspection of how communication pathways reconfigure across rounds.

**arXiv ID:** 2602.06039
</details>

<details>
<summary><strong>CoWork-X: Experience-Optimized Co-Evolution for Multi-Agent Collaboration System</strong> - Zexin Lin, Jiachen Yu, Haoyang Zhang, Yuzhao Li, Zhonghang Li, Yujiu Yang, Junjie Wang, Xiaoqiang Ji - [[pdf]](https://arxiv.org/pdf/2602.05004)</summary>

**Abstract:** Large language models are enabling language-conditioned agents in interactive environments, but highly cooperative tasks often impose two simultaneous constraints: sub-second real-time coordination and sustained multi-episode adaptation under a strict online token budget. Existing approaches either rely on frequent in-episode reasoning that induces latency and timing jitter, or deliver post-episode improvements through unstructured text that is difficult to compile into reliable low-cost execution. We propose CoWork-X, an active co-evolution framework that casts peer collaboration as a closed-loop optimization problem across episodes, inspired by fast--slow memory separation. CoWork-X instantiates a Skill-Agent that executes via HTN (hierarchical task network)-based skill retrieval from a structured, interpretable, and compositional skill library, and a post-episode Co-Optimizer that performs patch-style skill consolidation with explicit budget constraints and drift regularization. Experiments in challenging Overcooked-AI-like realtime collaboration benchmarks demonstrate that CoWork-X achieves stable, cumulative performance gains while steadily reducing online latency and token usage.

**arXiv ID:** 2602.05004
</details>

<details>
<summary><strong>Data-Centric Interpretability for LLM-based Multi-Agent Reinforcement Learning</strong> - John Yan, Michael Yu, Yuqi Sun, Alexander Duffy, Tyler Marques, Matthew Lyle Olson - [[pdf]](https://arxiv.org/pdf/2602.05183)</summary>

**Abstract:** Large language models (LLMs) are increasingly trained in complex Reinforcement Learning, multi-agent environments, making it difficult to understand how behavior changes over training. Sparse Autoencoders (SAEs) have recently shown to be useful for data-centric interpretability. In this work, we analyze large-scale reinforcement learning training runs from the sophisticated environment of Full-Press Diplomacy by applying pretrained SAEs, alongside LLM-summarizer methods. We introduce Meta-Autointerp, a method for grouping SAE features into interpretable hypotheses about training dynamics. We discover fine-grained behaviors including role-playing patterns, degenerate outputs, language switching, alongside high-level strategic behaviors and environment-specific bugs. Through automated evaluation, we validate that 90% of discovered SAE Meta-Features are significant, and find a surprising reward hacking behavior. However, through two user studies, we find that even subjectively interesting and seemingly helpful SAE features may be worse than useless to humans, along with most LLM generated hypotheses. However, a subset of SAE-derived hypotheses are predictively useful for downstream tasks. We further provide validation by augmenting an untrained agent's system prompt, improving the score by +14.2%. Overall, we show that SAEs and LLM-summarizer provide complementary views into agent behavior, and together our framework forms a practical starting point for future data-centric interpretability work on ensuring trustworthy LLM behavior throughout training.

**arXiv ID:** 2602.05183
</details>

<details>
<summary><strong>Towards a Science of Collective AI: LLM-based Multi-Agent Systems Need a Transition from Blind Trial-and-Error to Rigorous Science</strong> - Jingru Fan, Dewen Liu, Yufan Dang, Huatao Li, Yuheng Wang, Wei Liu, Feiyu Duan, Xuanwen Ding, Shu Yao, Lin Wu, Ruijie Shi, Wai-Shing Leung, Yuan Cheng, Zhongyu Wei, Cheng Yang, Chen Qian, Zhiyuan Liu, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2602.05289)</summary>

**Abstract:** Recent advancements in Large Language Models (LLMs) have greatly extended the capabilities of Multi-Agent Systems (MAS), demonstrating significant effectiveness across a wide range of complex and open-ended domains. However, despite this rapid progress, the field still relies heavily on empirical trial-and-error. It lacks a unified and principled scientific framework necessary for systematic optimization and improvement. This bottleneck stems from the ambiguity of attribution: first, the absence of a structured taxonomy of factors leaves researchers restricted to unguided adjustments; second, the lack of a unified metric fails to distinguish genuine collaboration gain from mere resource accumulation. In this paper, we advocate for a transition to design science through an integrated framework. We advocate to establish the collaboration gain metric ($\Gamma$) as the scientific standard to isolate intrinsic gains from increased budgets. Leveraging $\Gamma$, we propose a factor attribution paradigm to systematically identify collaboration-driving factors. To support this, we construct a systematic MAS factor library, structuring the design space into control-level presets and information-level dynamics. Ultimately, this framework facilitates the transition from blind experimentation to rigorous science, paving the way towards a true science of Collective AI.

**arXiv ID:** 2602.05289
</details>

<details>
<summary><strong>AI Agent Systems for Supply Chains: Structured Decision Prompts and Memory Retrieval</strong> - Konosuke Yoshizato, Kazuma Shimizu, Ryota Higa, Takanobu Otsuka - [[pdf]](https://arxiv.org/pdf/2602.05524)</summary>

**Abstract:** This study investigates large language model (LLM) -based multi-agent systems (MASs) as a promising approach to inventory management, which is a key component of supply chain management. Although these systems have gained considerable attention for their potential to address the challenges associated with typical inventory management methods, key uncertainties regarding their effectiveness persist. Specifically, it is unclear whether LLM-based MASs can consistently derive optimal ordering policies and adapt to diverse supply chain scenarios. To address these questions, we examine an LLM-based MAS with a fixed-ordering strategy prompt that encodes the stepwise processes of the problem setting and a safe-stock strategy commonly used in inventory management. Our empirical results demonstrate that, even without detailed prompt adjustments, an LLM-based MAS can determine optimal ordering decisions in a restricted scenario. To enhance adaptability, we propose a novel agent called AIM-RM, which leverages similar historical experiences through similarity matching. Our results show that AIM-RM outperforms benchmark methods across various supply chain scenarios, highlighting its robustness and adaptability.

**arXiv ID:** 2602.05524
</details>

<details>
<summary><strong>CommCP: Efficient Multi-Agent Coordination via LLM-Based Communication with Conformal Prediction</strong> - Xiaopan Zhang, Zejin Wang, Zhixu Li, Jianpeng Yao, Jiachen Li - [[pdf]](https://arxiv.org/pdf/2602.06038)</summary>

**Abstract:** To complete assignments provided by humans in natural language, robots must interpret commands, generate and answer relevant questions for scene understanding, and manipulate target objects. Real-world deployments often require multiple heterogeneous robots with different manipulation capabilities to handle different assignments cooperatively. Beyond the need for specialized manipulation skills, effective information gathering is important in completing these assignments. To address this component of the problem, we formalize the information-gathering process in a fully cooperative setting as an underexplored multi-agent multi-task Embodied Question Answering (MM-EQA) problem, which is a novel extension of canonical Embodied Question Answering (EQA), where effective communication is crucial for coordinating efforts without redundancy. To address this problem, we propose CommCP, a novel LLM-based decentralized communication framework designed for MM-EQA. Our framework employs conformal prediction to calibrate the generated messages, thereby minimizing receiver distractions and enhancing communication reliability. To evaluate our framework, we introduce an MM-EQA benchmark featuring diverse, photo-realistic household scenarios with embodied questions. Experimental results demonstrate that CommCP significantly enhances the task success rate and exploration efficiency over baselines. The experiment videos, code, and dataset are available on our project website: this https URL.

**arXiv ID:** 2602.06038
</details>

<details>
<summary><strong>Scaling Multi-Agent Epistemic Planning through GNN-Derived Heuristics</strong> - Giovanni Briglia, Francesco Fabiano, Stefano Mariani - [[pdf]](https://arxiv.org/pdf/2508.12840)</summary>

**Abstract:** Multi-agent Epistemic Planning (MEP) is an autonomous planning framework for reasoning about both the physical world and the beliefs of agents, with applications in domains where information flow and awareness among agents are critical. The richness of MEP requires states to be represented as Kripke structures, i.e., directed labeled graphs. This representation limits the applicability of existing heuristics, hindering the scalability of epistemic solvers, which must explore an exponential search space without guidance, resulting often in intractability. To address this, we exploit Graph Neural Networks (GNNs) to learn patterns and relational structures within epistemic states, to guide the planning process. GNNs, which naturally capture the graph-like nature of Kripke models, allow us to derive meaningful estimates of state quality -- e.g., the distance from the nearest goal -- by generalizing knowledge obtained from previously solved planning instances. We integrate these predictive heuristics into an epistemic planning pipeline and evaluate them against standard baselines, showing improvements in the scalability of multi-agent epistemic planning.

**arXiv ID:** 2508.12840
</details>

<details>
<summary><strong>Resisting Manipulative Bots in Meme Coin Copy Trading: A Multi-Agent Approach with Chain-of-Thought Reasoning</strong> - Yichen Luo, Yebo Feng, Jiahua Xu, Yang Liu - [[pdf]](https://arxiv.org/pdf/2601.08641)</summary>

**Abstract:** Copy trading has become the dominant entry strategy in meme coin markets. However, due to the market's extremely illiquid and volatile nature, the strategy exposes an exploitable attack surface: adversaries deploy manipulative bots to front-run trades, conceal positions, and fabricate sentiment, systematically extracting value from naïve copiers at scale. Despite its prevalence, bot-driven manipulation remains largely unexplored, and no robust defensive framework exists. We propose a manipulation-resistant copy-trading system based on a multi-agent architecture powered by a multi-modal large language model (LLM) and chain-of-thought (CoT) reasoning. Our approach outperforms zero-shot and most statistic-driven baselines in prediction accuracy as well as all baselines in economic performance, achieving an average copier return of 3% per meme coin investment under realistic market frictions. Overall, our results demonstrate the effectiveness of agent-based defenses and predictability of trader profitability in adversarial meme coin markets, providing a practical foundation for robust copy trading.

**arXiv ID:** 2601.08641
</details>

<details>
<summary><strong>Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration</strong> - Jiaheng Liu, Yuanxing Zhang, Shihao Li, Xinping Lei - [[pdf]](https://arxiv.org/pdf/2602.04575)</summary>

**Abstract:** For the past decade, the trajectory of generative artificial intelligence (AI) has been dominated by a model-centric paradigm driven by scaling laws. Despite significant leaps in visual fidelity, this approach has encountered a ``usability ceiling'' manifested as the Intent-Execution Gap (i.e., the fundamental disparity between a creator's high-level intent and the stochastic, black-box nature of current single-shot models). In this paper, inspired by the Vibe Coding, we introduce the \textbf{Vibe AIGC}, a new paradigm for content generation via agentic orchestration, which represents the autonomous synthesis of hierarchical multi-agent workflows.
Under this paradigm, the user's role transcends traditional prompt engineering, evolving into a Commander who provides a Vibe, a high-level representation encompassing aesthetic preferences, functional logic, and etc. A centralized Meta-Planner then functions as a system architect, deconstructing this ``Vibe'' into executable, verifiable, and adaptive agentic pipelines. By transitioning from stochastic inference to logical orchestration, Vibe AIGC bridges the gap between human imagination and machine execution. We contend that this shift will redefine the human-AI collaborative economy, transforming AI from a fragile inference engine into a robust system-level engineering partner that democratizes the creation of complex, long-horizon digital assets.

**arXiv ID:** 2602.04575
</details>

<details>
<summary><strong>PhysicsAgentABM: Physics-Guided Generative Agent-Based Modeling</strong> - Kavana Venkatesh, Yinhan He, Jundong Li, Jiaming Cui - [[pdf]](https://arxiv.org/pdf/2602.06030)</summary>

**Abstract:** Large language model (LLM)-based multi-agent systems enable expressive agent reasoning but are expensive to scale and poorly calibrated for timestep-aligned state-transition simulation, while classical agent-based models (ABMs) offer interpretability but struggle to integrate rich individual-level signals and non-stationary behaviors. We propose PhysicsAgentABM, which shifts inference to behaviorally coherent agent clusters: state-specialized symbolic agents encode mechanistic transition priors, a multimodal neural transition model captures temporal and interaction dynamics, and uncertainty-aware epistemic fusion yields calibrated cluster-level transition distributions. Individual agents then stochastically realize transitions under local constraints, decoupling population inference from entity-level variability. We further introduce ANCHOR, an LLM agent-driven clustering strategy based on cross-contextual behavioral responses and a novel contrastive loss, reducing LLM calls by up to 6-8 times. Experiments across public health, finance, and social sciences show consistent gains in event-time accuracy and calibration over mechanistic, neural, and LLM baselines. By re-architecting generative ABM around population-level inference with uncertainty-aware neuro-symbolic fusion, PhysicsAgentABM establishes a new paradigm for scalable and calibrated simulation with LLMs.

**arXiv ID:** 2602.06030
</details>

<details>
<summary><strong>Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization</strong> - Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch - [[pdf]](https://arxiv.org/pdf/2509.20900)</summary>

**Abstract:** Long document summarization remains a significant challenge for current large language models (LLMs), as existing approaches commonly struggle with information loss, factual inconsistencies, and coherence issues when processing excessively long documents. We propose SummQ, a novel adversarial multi-agent framework that addresses these limitations through collaborative intelligence between specialized agents operating in two complementary domains: summarization and quizzing. Our approach employs summary generators and reviewers that work collaboratively to create and evaluate comprehensive summaries, while quiz generators and reviewers create comprehension questions that serve as continuous quality checks for the summarization process. This adversarial dynamic, enhanced by an examinee agent that validates whether the generated summary contains the information needed to answer the quiz questions, enables iterative refinement through multifaceted feedback mechanisms. We evaluate SummQ on three widely used long document summarization benchmarks. Experimental results demonstrate that our framework significantly outperforms existing state-of-the-art methods across ROUGE and BERTScore metrics, as well as in LLM-as-a-Judge and human evaluations. Our comprehensive analyses reveal the effectiveness of the multi-agent collaboration dynamics, the influence of different agent configurations, and the impact of the quizzing mechanism. This work establishes a new approach for long document summarization that uses adversarial agentic collaboration to improve summarization quality.

**arXiv ID:** 2509.20900
</details>

<details>
<summary><strong>DEBATE: A Large-Scale Benchmark for Evaluating Opinion Dynamics in Role-Playing LLM Agents</strong> - Yun-Shiuan Chuang, Ruixuan Tu, Chengtao Dai, Smit Vasani, You Li, Binwei Yao, Michael Henry Tessler, Sijia Yang, Dhavan Shah, Robert Hawkins, Junjie Hu, Timothy T. Rogers - [[pdf]](https://arxiv.org/pdf/2510.25110)</summary>

**Abstract:** Accurately modeling opinion change through social interactions is crucial for understanding and mitigating polarization, misinformation, and societal conflict. Recent work simulates opinion dynamics with role-playing LLM agents (RPLAs), but multi-agent simulations often display unnatural group behavior (e.g., premature convergence) and lack empirical benchmarks for assessing alignment with real human group interactions. We introduce DEBATE, a large-scale benchmark for evaluating the authenticity of opinion dynamics in multi-agent RPLA simulations. DEBATE contains 36,383 messages from 2,832 U.S.-based participants across 708 groups and 107 topics, with both public messages and private Likert-scale beliefs, enabling evaluation at the utterance and group levels (and supporting future individual-level analyses). We instantiate "digital twin" RPLAs with seven LLMs and evaluate across two settings: next-message prediction and full conversation rollout, using stance-alignment and opinion-convergence metrics. In zero-shot settings, RPLA groups exhibit strong opinion convergence relative to human groups. Post-training via supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) improves stance alignment and brings group-level convergence closer to human behavior, though discrepancies in opinion change and belief updating remain. DEBATE enables rigorous benchmarking of simulated opinion dynamics and supports future research on aligning multi-agent RPLAs with realistic human interactions.

**arXiv ID:** 2510.25110
</details>

<details>
<summary><strong>CellForge: Agentic Design of Virtual Cell Models</strong> - Xiangru Tang, Zhuoyun Yu, Jiapeng Chen, Yan Cui, Daniel Shao, Weixu Wang, Fang Wu, Yuchen Zhuang, Wenqi Shi, Zhi Huang, Arman Cohan, Xihong Lin, Fabian Theis, Smita Krishnaswamy, Mark Gerstein - [[pdf]](https://arxiv.org/pdf/2508.02276)</summary>

**Abstract:** Virtual cell modeling aims to predict cellular responses to diverse perturbations but faces challenges from biological complexity, multimodal data heterogeneity, and the need for interdisciplinary expertise. We introduce CellForge, a multi-agent framework that autonomously designs and synthesizes neural network architectures tailored to specific single-cell datasets and perturbation tasks. Given raw multi-omics data and task descriptions, CellForge discovers candidate architectures through collaborative reasoning among specialized agents, then generates executable implementations. Our core contribution is the framework itself: showing that multi-agent collaboration mechanisms - rather than manual human design or single-LLM prompting - can autonomously produce executable, high-quality computational methods. This approach goes beyond conventional hyperparameter tuning by enabling entirely new architectural components such as trajectory-aware encoders and perturbation diffusion modules to emerge from agentic deliberation. We evaluate CellForge on six datasets spanning gene knockouts, drug treatments, and cytokine stimulations across multiple modalities (scRNA-seq, scATAC-seq, CITE-seq). The results demonstrate that the models generated by CellForge are highly competitive with established baselines, while revealing systematic patterns of architectural innovation. CellForge highlights the scientific value of multi-agent frameworks: collaboration among specialized agents enables genuine methodological innovation and executable solutions that single agents or human experts cannot achieve. This represents a paradigm shift toward autonomous scientific method development in computational biology. Code is available at this https URL.

**arXiv ID:** 2508.02276
</details>

<details>
<summary><strong>DiLLS: Interactive Diagnosis of LLM-based Multi-agent Systems via Layered Summary of Agent Behaviors</strong> - Rui Sheng, Yukun Yang, Chuhan Shi, Yanna Lin, Zixin Chen, Huamin Qu, Furui Cheng - [[pdf]](https://arxiv.org/pdf/2602.05446)</summary>

**Abstract:** Large language model (LLM)-based multi-agent systems have demonstrated impressive capabilities in handling complex tasks. However, the complexity of agentic behaviors makes these systems difficult to understand. When failures occur, developers often struggle to identify root causes and to determine actionable paths for improvement. Traditional methods that rely on inspecting raw log records are inefficient, given both the large volume and complexity of data. To address this challenge, we propose a framework and an interactive system, DiLLS, designed to reveal and structure the behaviors of multi-agent systems. The key idea is to organize information across three levels of query completion: activities, actions, and operations. By probing the multi-agent system through natural language, DiLLS derives and organizes information about planning and execution into a structured, multi-layered summary. Through a user study, we show that DiLLS significantly improves developers' effectiveness and efficiency in identifying, diagnosing, and understanding failures in LLM-based multi-agent systems.

**arXiv ID:** 2602.05446
</details>

<details>
<summary><strong>From Human-Human Collaboration to Human-Agent Collaboration: A Vision, Design Philosophy, and an Empirical Framework for Achieving Successful Partnerships Between Humans and LLM Agents</strong> - Bingsheng Yao, Chaoran Chen, April Yi Wang, Sherry Tongshuang Wu, Toby Jia-jun Li, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2602.05987)</summary>

**Abstract:** The emergence of Large Language Model (LLM) agents enables us to build agent-based intelligent systems that move beyond the role of a "tool" to become genuine collaborators with humans, thereby realizing a novel human-agent collaboration paradigm. Our vision is that LLM agents should resemble remote human collaborators, which allows HCI researchers to ground the future exploration in decades of research on trust, awareness, and common ground in remote human collaboration, while also revealing the unique opportunities and challenges that emerge when one or more partners are AI agents. This workshop establishes a foundational research agenda for the new era by posing the question: How can the rich understanding of remote human collaboration inspire and inform the design and study of human-agent collaboration? We will bring together an interdisciplinary group from HCI, CSCW, and AI to explore this critical transition. The 180-minute workshop will be highly interactive, featuring a keynote speaker, a series of invited lightning talks, and an exploratory group design session where participants will storyboard novel paradigms of human-agent partnership. Our goal is to enlighten the research community by cultivating a shared vocabulary and producing a research agenda that charts the future of collaborative agents.

**arXiv ID:** 2602.05987
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction</strong> - Ruijie Shi, Houbin Zhang, Yuecheng Han, Yuheng Wang, Jingru Fan, Runde Yang, Yufan Dang, Huatao Li, Dewen Liu, Yuan Cheng, Chen Qian - [[pdf]](https://arxiv.org/pdf/2602.05353)</summary>

**Abstract:** Large Language Models have shown strong capabilities in complex problem solving, yet many agentic systems remain difficult to interpret and control due to opaque internal workflows. While some frameworks offer explicit architectures for collaboration, many deployed agentic systems operate as black boxes to users. We address this by introducing Agentic Workflow Reconstruction (AWR), a new task aiming to synthesize an explicit, interpretable stand-in workflow that approximates a black-box system using only input--output access. We propose AgentXRay, a search-based framework that formulates AWR as a combinatorial optimization problem over discrete agent roles and tool invocations in a chain-structured workflow space. Unlike model distillation, AgentXRay produces editable white-box workflows that match target outputs under an observable, output-based proxy metric, without accessing model parameters. To navigate the vast search space, AgentXRay employs Monte Carlo Tree Search enhanced by a scoring-based Red-Black Pruning mechanism, which dynamically integrates proxy quality with search depth. Experiments across diverse domains demonstrate that AgentXRay achieves higher proxy similarity and reduces token consumption compared to unpruned search, enabling deeper workflow exploration under fixed iteration budgets.

**arXiv ID:** 2602.05353
</details>

<details>
<summary><strong>RocqSmith: Can Automatic Optimization Forge Better Proof Agents?</strong> - Andrei Kozyrev, Nikita Khramov, Denis Lochmelis, Valerio Morelli, Gleb Solovev, Anton Podkopaev - [[pdf]](https://arxiv.org/pdf/2602.05762)</summary>

**Abstract:** This work studies the applicability of automatic AI agent optimization methods to real-world agents in formal verification settings, focusing on automated theorem proving in Rocq as a representative and challenging domain. We evaluate how different automatic agent optimizers perform when applied to the task of optimizing a Rocq proof-generation agent, and assess whether parts of the fine-grained tuning of agentic systems, such as prompt design, contextual knowledge, and control strategies, can be automated. Our results show that while several optimizers yield measurable improvements, simple few-shot bootstrapping is the most consistently effective; however, none of the studied methods matches the performance of a carefully engineered state-of-the-art proof agent.

**arXiv ID:** 2602.05762
</details>

<details>
<summary><strong>From Fragmentation to Integration: Exploring the Design Space of AI Agents for Human-as-the-Unit Privacy Management</strong> - Eryue Xu, Tianshi Li - [[pdf]](https://arxiv.org/pdf/2602.05016)</summary>

**Abstract:** Managing one's digital footprint is overwhelming, as it spans multiple platforms and involves countless context-dependent decisions. Recent advances in agentic AI offer ways forward by enabling holistic, contextual privacy-enhancing solutions. Building on this potential, we adopted a ''human-as-the-unit'' perspective and investigated users' cross-context privacy challenges through 12 semi-structured interviews. Results reveal that people rely on ad hoc manual strategies while lacking comprehensive privacy controls, highlighting nine privacy-management challenges across applications, temporal contexts, and relationships. To explore solutions, we generated nine AI agent concepts and evaluated them via a speed-dating survey with 116 US participants. The three highest-ranked concepts were all post-sharing management tools with half or full agent autonomy, with users expressing greater trust in AI accuracy than in their own efforts. Our findings highlight a promising design space where users see AI agents bridging the fragments in privacy management, particularly through automated, comprehensive post-sharing remediation of users' digital footprints.

**arXiv ID:** 2602.05016
</details>

<details>
<summary><strong>Mitigating Conversational Inertia in Multi-Turn Agents</strong> - Yang Wan, Zheng Cao, Zhenhao Zhang, Zhengwen Zeng, Shuheng Shen, Changhua Meng, Linchao Zhu - [[pdf]](https://arxiv.org/pdf/2602.03664)</summary>

**Abstract:** Large language models excel as few-shot learners when provided with appropriate demonstrations, yet this strength becomes problematic in multiturn agent scenarios, where LLMs erroneously mimic their own previous responses as few-shot examples. Through attention analysis, we identify conversational inertia, a phenomenon where models exhibit strong diagonal attention to previous responses, which is associated with imitation bias that constrains exploration. This reveals a tension when transforming few-shot LLMs into agents: longer context enriches environmental feedback for exploitation, yet also amplifies conversational inertia that undermines exploration. Our key insight is that for identical states, actions generated with longer contexts exhibit stronger inertia than those with shorter contexts, enabling construction of preference pairs without environment rewards. Based on this, we propose Context Preference Learning to calibrate model preferences to favor low-inertia responses over highinertia ones. We further provide context management strategies at inference time to balance exploration and exploitation. Experimental results across eight agentic environments and one deep research scenario validate that our framework reduces conversational inertia and achieves performance improvements.

**arXiv ID:** 2602.03664
</details>

<details>
<summary><strong>Reinforcement World Model Learning for LLM-based Agents</strong> - Xiao Yu, Baolin Peng, Ruize Xu, Yelong Shen, Pengcheng He, Suman Nath, Nikhil Singh, Jiangfeng Gao, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2602.05842)</summary>

**Abstract:** Large language models (LLMs) have achieved strong performance in language-centric tasks. However, in agentic settings, LLMs often struggle to anticipate action consequences and adapt to environment dynamics, highlighting the need for world-modeling capabilities in LLM-based agents. We propose Reinforcement World Model Learning (RWML), a self-supervised method that learns action-conditioned world models for LLM-based agents on textual states using sim-to-real gap rewards. Our method aligns simulated next states produced by the model with realized next states observed from the environment, encouraging consistency between internal world simulations and actual environment dynamics in a pre-trained embedding space. Unlike next-state token prediction, which prioritizes token-level fidelity (i.e., reproducing exact wording) over semantic equivalence and can lead to model collapse, our method provides a more robust training signal and is empirically less susceptible to reward hacking than LLM-as-a-judge. We evaluate our method on ALFWorld and $\tau^2$ Bench and observe significant gains over the base model, despite being entirely self-supervised. When combined with task-success rewards, our method outperforms direct task-success reward RL by 6.9 and 5.7 points on ALFWorld and $\tau^2$ Bench respectively, while matching the performance of expert-data training.

**arXiv ID:** 2602.05842
</details>

<details>
<summary><strong>Virtual-Tube-Based Cooperative Transport Control for Multi-UAV Systems in Constrained Environments</strong> - Runxiao Liu, Pengda Mao, Xiangli Le, Shuang Gu, Yapeng Chen, Quan Quan - [[pdf]](https://arxiv.org/pdf/2602.05516)</summary>

**Abstract:** This paper proposes a novel control framework for cooperative transportation of cable-suspended loads by multiple unmanned aerial vehicles (UAVs) operating in constrained environments. Leveraging virtual tube theory and principles from dissipative systems theory, the framework facilitates efficient multi-UAV collaboration for navigating obstacle-rich areas. The proposed framework offers several key advantages. (1) It achieves tension distribution and coordinated transportation within the UAV-cable-load system with low computational overhead, dynamically adapting UAV configurations based on obstacle layouts to facilitate efficient navigation. (2) By integrating dissipative systems theory, the framework ensures high stability and robustness, essential for complex multi-UAV operations. The effectiveness of the proposed approach is validated through extensive simulations, demonstrating its scalability for large-scale multi-UAV systems. Furthermore, the method is experimentally validated in outdoor scenarios, showcasing its practical feasibility and robustness under real-world conditions.

**arXiv ID:** 2602.05516
</details>

<details>
<summary><strong>From Vision to Decision: Neuromorphic Control for Autonomous Navigation and Tracking</strong> - Chuwei Wang, Eduardo Sebastián, Amanda Prorok, Anastasia Bizyaeva - [[pdf]](https://arxiv.org/pdf/2602.05683)</summary>

**Abstract:** Robotic navigation has historically struggled to reconcile reactive, sensor-based control with the decisive capabilities of model-based planners. This duality becomes critical when the absence of a predominant option among goals leads to indecision, challenging reactive systems to break symmetries without computationally-intense planners. We propose a parsimonious neuromorphic control framework that bridges this gap for vision-guided navigation and tracking. Image pixels from an onboard camera are encoded as inputs to dynamic neuronal populations that directly transform visual target excitation into egocentric motion commands. A dynamic bifurcation mechanism resolves indecision by delaying commitment until a critical point induced by the environmental geometry. Inspired by recently proposed mechanistic models of animal cognition and opinion dynamics, the neuromorphic controller provides real-time autonomy with a minimal computational burden, a small number of interpretable parameters, and can be seamlessly integrated with application-specific image processing pipelines. We validate our approach in simulation environments as well as on an experimental quadrotor platform.

**arXiv ID:** 2602.05683
</details>

<details>
<summary><strong>(Computer) Vision in Action: Comparing Remote Sighted Assistance and a Multimodal Voice Agent in Inspection Sequences</strong> - Damien Rudaz, Barbara Nino Carreras, Sara Merlino, Brian L. Due, Barry Brown - [[pdf]](https://arxiv.org/pdf/2602.05671)</summary>

**Abstract:** Does human-AI assistance unfold in the same way as human-human assistance? This research explores what can be learned from the expertise of blind individuals and sighted volunteers to inform the design of multimodal voice agents and address the enduring challenge of proactivity. Drawing on granular analysis of two representative fragments from a larger corpus, we contrast the practices co-produced by an experienced human remote sighted assistant and a blind participant-as they collaborate to find a stain on a blanket over the phone-with those achieved when the same participant worked with a multimodal voice agent on the same task, a few moments earlier. This comparison enables us to specify precisely which fundamental proactive practices the agent did not enact in situ. We conclude that, so long as multimodal voice agents cannot produce environmentally occasioned vision-based actions, they will lack a key resource relied upon by human remote sighted assistants.

**arXiv ID:** 2602.05671
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search</strong> - Zhanli Li, Huiwen Tian, Lvzhou Luo, Yixuan Cao, Ping Luo - [[pdf]](https://arxiv.org/pdf/2602.05014)</summary>

**Abstract:** With the rapid progress of tool-using and agentic large language models (LLMs), Retrieval-Augmented Generation (RAG) is evolving from one-shot, passive retrieval into multi-turn, decision-driven evidence acquisition. Despite strong results in open-domain settings, existing agentic search frameworks commonly treat long documents as flat collections of chunks, underutilizing document-native priors such as hierarchical organization and sequential discourse structure. We introduce DeepRead, a structure-aware, multi-turn document reasoning agent that explicitly operationalizes these priors for long-document question answering. DeepRead leverages LLM-based OCR model to convert PDFs into structured Markdown that preserves headings and paragraph boundaries. It then indexes documents at the paragraph level and assigns each paragraph a coordinate-style metadata key encoding its section identity and in-section order. Building on this representation, DeepRead equips the LLM with two complementary tools: a Retrieve tool that localizes relevant paragraphs while exposing their structural coordinates (with lightweight scanning context), and a ReadSection tool that enables contiguous, order-preserving reading within a specified section and paragraph range. Our experiments demonstrate that DeepRead achieves significant improvements over Search-o1-style agentic search in document question answering. The synergistic effect between retrieval and reading tools is also validated. Our fine-grained behavioral analysis reveals a reading and reasoning paradigm resembling human-like ``locate then read'' behavior.

**arXiv ID:** 2602.05014
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (31 papers)</h2></summary>

<details>
<summary><strong>Optimizing Mission Planning for Multi-Debris Rendezvous Using Reinforcement Learning with Refueling and Adaptive Collision Avoidance</strong> - Agni Bandyopadhyay, Gunther Waxenegger-Wilfing - [[pdf]](https://arxiv.org/pdf/2602.05075)</summary>

**Abstract:** As the orbital environment around Earth becomes increasingly crowded with debris, active debris removal (ADR) missions face significant challenges in ensuring safe operations while minimizing the risk of in-orbit collisions. This study presents a reinforcement learning (RL) based framework to enhance adaptive collision avoidance in ADR missions, specifically for multi-debris removal using small satellites. Small satellites are increasingly adopted due to their flexibility, cost effectiveness, and maneuverability, making them well suited for dynamic missions such as ADR.
Building on existing work in multi-debris rendezvous, the framework integrates refueling strategies, efficient mission planning, and adaptive collision avoidance to optimize spacecraft rendezvous operations. The proposed approach employs a masked Proximal Policy Optimization (PPO) algorithm, enabling the RL agent to dynamically adjust maneuvers in response to real-time orbital conditions. Key considerations include fuel efficiency, avoidance of active collision zones, and optimization of dynamic orbital parameters.
The RL agent learns to determine efficient sequences for rendezvousing with multiple debris targets, optimizing fuel usage and mission time while incorporating necessary refueling stops. Simulated ADR scenarios derived from the Iridium 33 debris dataset are used for evaluation, covering diverse orbital configurations and debris distributions to demonstrate robustness and adaptability. Results show that the proposed RL framework reduces collision risk while improving mission efficiency compared to traditional heuristic approaches.
This work provides a scalable solution for planning complex multi-debris ADR missions and is applicable to other multi-target rendezvous problems in autonomous space mission planning.

**arXiv ID:** 2602.05075
</details>

<details>
<summary><strong>ProAct: Agentic Lookahead in Interactive Environments</strong> - Yangbin Yu, Mingyu Yang, Junyou Li, Yiming Gao, Feiyu Liu, Yijun Yang, Zichuan Lin, Jiafei Lyu, Yicheng Liu, Zhicong Lu, Deheng Ye, Jie Jiang - [[pdf]](https://arxiv.org/pdf/2602.05327)</summary>

**Abstract:** Existing Large Language Model (LLM) agents struggle in interactive environments requiring long-horizon planning, primarily due to compounding errors when simulating future states. To address this, we propose ProAct, a framework that enables agents to internalize accurate lookahead reasoning through a two-stage training paradigm. First, we introduce Grounded LookAhead Distillation (GLAD), where the agent undergoes supervised fine-tuning on trajectories derived from environment-based search. By compressing complex search trees into concise, causal reasoning chains, the agent learns the logic of foresight without the computational overhead of inference-time search. Second, to further refine decision accuracy, we propose the Monte-Carlo Critic (MC-Critic), a plug-and-play auxiliary value estimator designed to enhance policy-gradient algorithms like PPO and GRPO. By leveraging lightweight environment rollouts to calibrate value estimates, MC-Critic provides a low-variance signal that facilitates stable policy optimization without relying on expensive model-based value approximation. Experiments on both stochastic (e.g., 2048) and deterministic (e.g., Sokoban) environments demonstrate that ProAct significantly improves planning accuracy. Notably, a 4B parameter model trained with ProAct outperforms all open-source baselines and rivals state-of-the-art closed-source models, while demonstrating robust generalization to unseen environments. The codes and models are available at this https URL

**arXiv ID:** 2602.05327
</details>

<details>
<summary><strong>RL-VLA$^3$: Reinforcement Learning VLA Accelerating via Full Asynchronism</strong> - Zhong Guan, Haoran Sun, Yongjian Guo, Shuai Di, Xiaodong Bai, Jing Long, Tianyun Zhao, Mingxi Luo, Chen Zhou, Yucheng Guo, Qiming Yang, Wanting Xu, Wen Huang, Yunxuan Ma, Hongke Zhao, Likang Wu, Xiaotie Deng, Xi Xiao, Sheng Wen, Yicheng Gong, Junwu Xiong - [[pdf]](https://arxiv.org/pdf/2602.05765)</summary>

**Abstract:** In recent years, Vision-Language-Action (VLA) models have emerged as a crucial pathway towards general embodied intelligence, yet their training efficiency has become a key bottleneck. Although existing reinforcement learning (RL)-based training frameworks like RLinf can enhance model generalization, they still rely on synchronous execution, leading to severe resource underutilization and throughput limitations during environment interaction, policy generation (rollout), and model update phases (actor). To overcome this challenge, this paper, for the first time, proposes and implements a fully-asynchronous policy training framework encompassing the entire pipeline from environment interaction, rollout generation, to actor policy updates. Systematically drawing inspiration from asynchronous optimization ideas in large model RL, our framework designs a multi-level decoupled architecture. This includes asynchronous parallelization of environment interaction and trajectory collection, streaming execution for policy generation, and decoupled scheduling for training updates. We validated the effectiveness of our method across diverse VLA models and environments. On the LIBERO benchmark, the framework achieves throughput improvements of up to 59.25\% compared to existing synchronous strategies. When deeply optimizing separation strategies, throughput can be increased by as much as 126.67\%. We verified the effectiveness of each asynchronous component via ablation studies. Scaling law validation across 8 to 256 GPUs demonstrates our method's excellent scalability under most conditions.

**arXiv ID:** 2602.05765
</details>

<details>
<summary><strong>TKG-Thinker: Towards Dynamic Reasoning over Temporal Knowledge Graphs via Agentic Reinforcement Learning</strong> - Zihao Jiang, Miao Peng, Zhenyan Shan, Wenjie Xu, Ben Liu, Gong Chen, Ziqi Gao, Min Peng - [[pdf]](https://arxiv.org/pdf/2602.05818)</summary>

**Abstract:** Temporal knowledge graph question answering (TKGQA) aims to answer time-sensitive questions by leveraging temporal knowledge bases. While Large Language Models (LLMs) demonstrate significant potential in TKGQA, current prompting strategies constrain their efficacy in two primary ways. First, they are prone to reasoning hallucinations under complex temporal constraints. Second, static prompting limits model autonomy and generalization, as it lack optimization through dynamic interaction with temporal knowledge graphs (TKGs) environments. To address these limitations, we propose \textbf{TKG-Thinker}, a novel agent equipped with autonomous planning and adaptive retrieval capabilities for reasoning over TKGs. Specifically, TKG-Thinker performs in-depth temporal reasoning through dynamic multi-turn interactions with TKGs via a dual-training strategy. We first apply Supervised Fine-Tuning (SFT) with chain-of thought data to instill core planning capabilities, followed by a Reinforcement Learning (RL) stage that leverages multi-dimensional rewards to refine reasoning policies under intricate temporal constraints. Experimental results on benchmark datasets with three open-source LLMs show that TKG-Thinker achieves state-of-the-art performance and exhibits strong generalization across complex TKGQA settings.

**arXiv ID:** 2602.05818
</details>

<details>
<summary><strong>Quantum Reinforcement Learning with Transformers for the Capacitated Vehicle Routing Problem</strong> - Eva Andrés - [[pdf]](https://arxiv.org/pdf/2602.05920)</summary>

**Abstract:** This paper addresses the Capacitated Vehicle Routing Problem (CVRP) by comparing classical and quantum Reinforcement Learning (RL) approaches. An Advantage Actor-Critic (A2C) agent is implemented in classical, full quantum, and hybrid variants, integrating transformer architectures to capture the relationships between vehicles, clients, and the depot through self- and cross-attention mechanisms. The experiments focus on multi-vehicle scenarios with capacity constraints, considering 20 clients and 4 vehicles, and are conducted over ten independent runs. Performance is assessed using routing distance, route compactness, and route overlap. The results show that all three approaches are capable of learning effective routing policies. However, quantum-enhanced models outperform the classical baseline and produce more robust route organization, with the hybrid architecture achieving the best overall performance across distance, compactness, and route overlap. In addition to quantitative improvements, qualitative visualizations reveal that quantum-based models generate more structured and coherent routing solutions. These findings highlight the potential of hybrid quantum-classical reinforcement learning models for addressing complex combinatorial optimization problems such as the CVRP.

**arXiv ID:** 2602.05920
</details>

<details>
<summary><strong>Autodiscover: A reinforcement learning recommendation system for the cold-start imbalance challenge in active learning, powered by graph-aware thompson sampling</strong> - Parsa Vares - [[pdf]](https://arxiv.org/pdf/2602.05087)</summary>

**Abstract:** Systematic literature reviews (SLRs) are fundamental to evidence-based research, but manual screening is an increasing bottleneck as scientific output grows. Screening features low prevalence of relevant studies and scarce, costly expert decisions. Traditional active learning (AL) systems help, yet typically rely on fixed query strategies for selecting the next unlabeled documents. These static strategies do not adapt over time and ignore the relational structure of scientific literature networks. This thesis introduces AutoDiscover, a framework that reframes AL as an online decision-making problem driven by an adaptive agent. Literature is modeled as a heterogeneous graph capturing relationships among documents, authors, and metadata. A Heterogeneous Graph Attention Network (HAN) learns node representations, which a Discounted Thompson Sampling (DTS) agent uses to dynamically manage a portfolio of query strategies. With real-time human-in-the-loop labels, the agent balances exploration and exploitation under non-stationary review dynamics, where strategy utility changes over time. On the 26-dataset SYNERGY benchmark, AutoDiscover achieves higher screening efficiency than static AL baselines. Crucially, the agent mitigates cold start by bootstrapping discovery from minimal initial labels where static approaches fail. We also introduce TS-Insight, an open-source visual analytics dashboard to interpret, verify, and diagnose the agent's decisions. Together, these contributions accelerate SLR screening under scarce expert labels and low prevalence of relevant studies.

**arXiv ID:** 2602.05087
</details>

<details>
<summary><strong>LinguistAgent: A Reflective Multi-Model Platform for Automated Linguistic Annotation</strong> - Bingru Li - [[pdf]](https://arxiv.org/pdf/2602.05493)</summary>

**Abstract:** Data annotation remains a significant bottleneck in the Humanities and Social Sciences, particularly for complex semantic tasks such as metaphor identification. While Large Language Models (LLMs) show promise, a significant gap remains between the theoretical capability of LLMs and their practical utility for researchers. This paper introduces LinguistAgent, an integrated, user-friendly platform that leverages a reflective multi-model architecture to automate linguistic annotation. The system implements a dual-agent workflow, comprising an Annotator and a Reviewer, to simulate a professional peer-review process. LinguistAgent supports comparative experiments across three paradigms: Prompt Engineering (Zero/Few-shot), Retrieval-Augmented Generation, and Fine-tuning. We demonstrate LinguistAgent's efficacy using the task of metaphor identification as an example, providing real-time token-level evaluation (Precision, Recall, and $F_1$ score) against human gold standards. The application and codes are released on this https URL.

**arXiv ID:** 2602.05493
</details>

<details>
<summary><strong>Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations</strong> - Wei Liu, Jiawei Xu, Yingru Li, Longtao Zheng, Tianjian Li, Qian Liu, Junxian He - [[pdf]](https://arxiv.org/pdf/2602.05885)</summary>

**Abstract:** High-quality kernel is critical for scalable AI systems, and enabling LLMs to generate such code would advance AI development. However, training LLMs for this task requires sufficient data, a robust environment, and the process is often vulnerable to reward hacking and lazy optimization. In these cases, models may hack training rewards and prioritize trivial correctness over meaningful speedup. In this paper, we systematically study reinforcement learning (RL) for kernel generation. We first design KernelGYM, a robust distributed GPU environment that supports reward hacking check, data collection from multi-turn interactions and long-term RL training. Building on KernelGYM, we investigate effective multi-turn RL methods and identify a biased policy gradient issue caused by self-inclusion in GRPO. To solve this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO) to provide unbiased advantage estimation for multi-turn RL. To alleviate lazy optimization, we incorporate mismatch correction for training stability and introduce Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to overcome the issue. The trained model, this http URL-14B, reaches performance competitive with Claude-4.5-Sonnet in Kernelbench. Finally, we study sequential test-time scaling for this http URL-14B. On the KernelBench Level-2 subset, 31.6% of the generated kernels achieve at least a 1.2x speedup over the Torch reference, surpassing Claude-4.5-Sonnet (26.7%) and GPT-5 (28.6%). When selecting the best candidate across all turns, this 1.2x speedup rate further increases to 47.8%. All resources, including environment, training code, models, and dataset, are included in this https URL.

**arXiv ID:** 2602.05885
</details>

<details>
<summary><strong>Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory</strong> - Haozhen Zhang, Haodong Yue, Tao Feng, Quanyu Long, Jianzhu Bao, Bowen Jin, Weizhi Zhang, Xiao Li, Jiaxuan You, Chengwei Qin, Wenya Wang - [[pdf]](https://arxiv.org/pdf/2602.06025)</summary>

**Abstract:** Memory is increasingly central to Large Language Model (LLM) agents operating beyond a single context window, yet most existing systems rely on offline, query-agnostic memory construction that can be inefficient and may discard query-critical information. Although runtime memory utilization is a natural alternative, prior work often incurs substantial overhead and offers limited explicit control over the performance-cost trade-off. In this work, we present \textbf{BudgetMem}, a runtime agent memory framework for explicit, query-aware performance-cost control. BudgetMem structures memory processing as a set of memory modules, each offered in three budget tiers (i.e., \textsc{Low}/\textsc{Mid}/\textsc{High}). A lightweight router performs budget-tier routing across modules to balance task performance and memory construction cost, which is implemented as a compact neural policy trained with reinforcement learning. Using BudgetMem as a unified testbed, we study three complementary strategies for realizing budget tiers: implementation (method complexity), reasoning (inference behavior), and capacity (module model size). Across LoCoMo, LongMemEval, and HotpotQA, BudgetMem surpasses strong baselines when performance is prioritized (i.e., high-budget setting), and delivers better accuracy-cost frontiers under tighter budgets. Moreover, our analysis disentangles the strengths and weaknesses of different tiering strategies, clarifying when each axis delivers the most favorable trade-offs under varying budget regimes.

**arXiv ID:** 2602.06025
</details>

<details>
<summary><strong>Learning to Share: Selective Memory for Efficient Parallel Agentic Systems</strong> - Joseph Fioresi, Parth Parag Kulkarni, Ashmal Vayani, Song Wang, Mubarak Shah - [[pdf]](https://arxiv.org/pdf/2602.05965)</summary>

**Abstract:** Agentic systems solve complex tasks by coordinating multiple agents that iteratively reason, invoke tools, and exchange intermediate results. To improve robustness and solution quality, recent approaches deploy multiple agent teams running in parallel to explore diverse reasoning trajectories. However, parallel execution comes at a significant computational cost: when different teams independently reason about similar sub-problems or execute analogous steps, they repeatedly perform substantial overlapping computation. To address these limitations, in this paper, we propose Learning to Share (LTS), a learned shared-memory mechanism for parallel agentic frameworks that enables selective cross-team information reuse while controlling context growth. LTS introduces a global memory bank accessible to all teams and a lightweight controller that decides whether intermediate agent steps should be added to memory or not. The controller is trained using stepwise reinforcement learning with usage-aware credit assignment, allowing it to identify information that is globally useful across parallel executions. Experiments on the AssistantBench and GAIA benchmarks show that LTS significantly reduces overall runtime while matching or improving task performance compared to memory-free parallel baselines, demonstrating that learned memory admission is an effective strategy for improving the efficiency of parallel agentic systems. Project page: this https URL

**arXiv ID:** 2602.05965
</details>

<details>
<summary><strong>Interpretability by Design for Efficient Multi-Objective Reinforcement Learning</strong> - Qiyue Xia, Tianwei Wang, J. Michael Herrmann - [[pdf]](https://arxiv.org/pdf/2506.04022)</summary>

**Abstract:** Multi-objective reinforcement learning (MORL) aims at optimising several, often conflicting goals to improve the flexibility and reliability of RL in practical tasks. This is typically achieved by finding a set of diverse, non-dominated policies that form a Pareto front in the performance space. We introduce LLE-MORL, an approach that achieves interpretability by design by utilising a training scheme based on the local relationship between the parameter space and the performance space. By exploiting a locally linear map between these spaces, our method provides an interpretation of policy parameters in terms of the objectives, and this structured representation enables an efficient search within contiguous solution domains, allowing for the rapid generation of high-quality solutions without extensive retraining. Experiments across diverse continuous control domains demonstrate that LLE-MORL consistently achieves higher Pareto front quality and efficiency than state-of-the-art approaches.

**arXiv ID:** 2506.04022
</details>

<details>
<summary><strong>DeepAgent: A General Reasoning Agent with Scalable Toolsets</strong> - Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2510.21618)</summary>

**Abstract:** Large reasoning models have demonstrated strong problem-solving abilities, yet real-world tasks often require external tools and long-horizon interactions. Existing agent frameworks typically follow predefined workflows, which limit autonomous and global task completion. In this paper, we introduce DeepAgent, an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. To manage long-horizon interactions, we introduce an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation while preserving critical information. To teach general-purpose tool use efficiently and stably, we develop an end-to-end reinforcement learning strategy, namely ToolPO, that leverages LLM-simulated APIs and applies tool-call advantage attribution to assign fine-grained credit to the tool invocation tokens. Extensive experiments on eight benchmarks, including general tool-use tasks (ToolBench, API-Bank, TMDB, Spotify, ToolHop) and downstream applications (ALFWorld, WebShop, GAIA, HLE), demonstrate that DeepAgent consistently outperforms baselines across both labeled-tool and open-set tool retrieval scenarios. The code and demo are available at this https URL.

**arXiv ID:** 2510.21618
</details>

<details>
<summary><strong>A Differential and Pointwise Control Approach to Reinforcement Learning</strong> - Minh Nguyen, Chandrajit Bajaj - [[pdf]](https://arxiv.org/pdf/2404.15617)</summary>

**Abstract:** Reinforcement learning (RL) in continuous state-action spaces remains challenging in scientific computing due to poor sample efficiency and lack of pathwise physical consistency. We introduce Differential Reinforcement Learning (Differential RL), a novel framework that reformulates RL from a continuous-time control perspective via a differential dual formulation. This induces a Hamiltonian structure that embeds physics priors and ensures consistent trajectories without requiring explicit constraints. To implement Differential RL, we develop Differential Policy Optimization (dfPO), a pointwise, stage-wise algorithm that refines local movement operators along the trajectory for improved sample efficiency and dynamic alignment. We establish pointwise convergence guarantees, a property not available in standard RL, and derive a competitive theoretical regret bound of $\mathcal{O}(K^{5/6})$. Empirically, dfPO outperforms standard RL baselines on representative scientific computing tasks, including surface modeling, grid control, and molecular dynamics, under low-data and physics-constrained conditions.

**arXiv ID:** 2404.15617
</details>

<details>
<summary><strong>LongR: Unleashing Long-Context Reasoning via Reinforcement Learning with Dense Utility Rewards</strong> - Bowen Ping, Zijun Chen, Yiyao Yu, Tingfeng Hui, Junchi Yan, Baobao Chang - [[pdf]](https://arxiv.org/pdf/2602.05758)</summary>

**Abstract:** Reinforcement Learning has emerged as a key driver for LLM reasoning. This capability is equally pivotal in long-context scenarios--such as long-dialogue understanding and structured data analysis, where the challenge extends beyond consuming tokens to performing rigorous deduction. While existing efforts focus on data synthesis or architectural changes, recent work points out that relying solely on sparse, outcome-only rewards yields limited gains, as such coarse signals are often insufficient to effectively guide the complex long-context reasoning. To address this, we propose LongR, a unified framework that enhances long-context performance by integrating a dynamic "Think-and-Read" mechanism, which interleaves reasoning with document consultation, with a contextual density reward based on relative information gain to quantify the utility of the relevant documents. Empirically, LongR achieves a 9% gain on LongBench v2 and consistent improvements on RULER and InfiniteBench, demonstrating robust efficiency in navigating extensive contexts. Furthermore, LongR consistently enhances performance across diverse RL algorithms (e.g., DAPO, GSPO). Finally, we conduct in-depth analyses to investigate the impact of reasoning chain length on efficiency and the model's robustness against distractors.

**arXiv ID:** 2602.05758
</details>

<details>
<summary><strong>Stop Rewarding Hallucinated Steps: Faithfulness-Aware Step-Level Reinforcement Learning for Small Reasoning Models</strong> - Shuo Nie, Hexuan Deng, Chao Wang, Ruiyu Fang, Xuebo Liu, Shuangyong Song, Yu Li, Min Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2602.05897)</summary>

**Abstract:** As large language models become smaller and more efficient, small reasoning models (SRMs) are crucial for enabling chain-of-thought (CoT) reasoning in resource-constrained settings. However, they are prone to faithfulness hallucinations, especially in intermediate reasoning steps. Existing mitigation methods based on online reinforcement learning rely on outcome-based rewards or coarse-grained CoT evaluation, which can inadvertently reinforce unfaithful reasoning when the final answer is correct. To address these limitations, we propose Faithfulness-Aware Step-Level Reinforcement Learning (FaithRL), introducing step-level supervision via explicit faithfulness rewards from a process reward model, together with an implicit truncated resampling strategy that generates contrastive signals from faithful prefixes. Experiments across multiple SRMs and Open-Book QA benchmarks demonstrate that FaithRL consistently reduces hallucinations in both the CoT and final answers, leading to more faithful and reliable reasoning. Code is available at this https URL.

**arXiv ID:** 2602.05897
</details>

<details>
<summary><strong>Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities</strong> - Pengyi Li, Elizaveta Goncharova, Andrey Kuznetsov, Ivan Oseledets - [[pdf]](https://arxiv.org/pdf/2602.05281)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an indispensable paradigm for enhancing reasoning in Large Language Models (LLMs). However, standard policy optimization methods, such as Group Relative Policy Optimization (GRPO), often converge to low-entropy policies, leading to severe mode collapse and limited output diversity. We analyze this issue from the perspective of sampling probability dynamics, identifying that the standard objective disproportionately reinforces the highest-likelihood paths, thereby suppressing valid alternative reasoning chains. To address this, we propose a novel Advantage Re-weighting Mechanism (ARM) designed to equilibrate the confidence levels across all correct responses. By incorporating Prompt Perplexity and Answer Confidence into the advantage estimation, our method dynamically reshapes the reward signal to attenuate the gradient updates of over-confident reasoning paths, while redistributing probability mass toward under-explored correct solutions. Empirical results demonstrate that our approach significantly enhances generative diversity and response entropy while maintaining competitive accuracy, effectively achieving a superior trade-off between exploration and exploitation in reasoning tasks. Empirical results on Qwen2.5 and DeepSeek models across mathematical and coding benchmarks show that ProGRPO significantly mitigates entropy collapse. Specifically, on Qwen2.5-7B, our method outperforms GRPO by 5.7% in Pass@1 and, notably, by 13.9% in Pass@32, highlighting its superior capability in generating diverse correct reasoning paths.

**arXiv ID:** 2602.05281
</details>

<details>
<summary><strong>Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning</strong> - Zhaowei Liu, Xin Guo, Zhi Yang, Fangqi Lou, Lingfeng Zeng, Mengping Li, Qi Qi, Zhiqiang Liu, Yiyang Han, Dongpo Cheng, Ronghao Chen, Huacan Wang, Xingdong Feng, Huixia Judy Wang, Chengchun Shi, Liwen Zhang - [[pdf]](https://arxiv.org/pdf/2503.16252)</summary>

**Abstract:** In recent years, general-purpose large language models (LLMs) such as GPT, Gemini, Claude, and DeepSeek have advanced at an unprecedented pace. Despite these achievements, their application to finance remains challenging, due to fragmented data sources, intransparent reasoning processes, and weak transferability to business applications. In response, we introduce Fin-R1, a reasoning LLM designed for financial scenarios. With a compact size of 7 billion parameters, Fin-R1 reduces deployment costs while addressing the aforementioned challenges. Its development follows a two-stage pipeline. First, we construct Fin-R1-Data, a high-quality financial dataset consisting of 60,091 chain-of-thought (CoT) samples, distilled and filtered from multiple authoritative benchmarks to ensure consistency and reliability. Second, we train Fin-R1 using Fin-R1-Data through supervised fine-tuning (SFT), followed by reinforcement learning (RL). This stage substantially improves the model's ability to solve complex financial reasoning tasks, yielding outputs that are both accurate and interpretable. Despite its relatively small parameter scale, Fin-R1 achieves competitive empirical performance across established financial benchmarks and demonstrates practical utility in compliance checking and robo-advisory. Our code is publicly available at this https URL, and has already attracted over 700 stars.

**arXiv ID:** 2503.16252
</details>

<details>
<summary><strong>Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning</strong> - Ahmed Attia, Alham Fikri Aji - [[pdf]](https://arxiv.org/pdf/2601.12535)</summary>

**Abstract:** Low-resource machine translation (MT) has gained increasing attention as parallel data from low-resource language communities is collected, but many potential methods for improving low-resource MT remain unexplored. We investigate a self-supervised reinforcement-learning-based fine-tuning for translation in low-resource settings using round-trip bootstrapping with the No Language Left Behind (NLLB) family of models. Our approach translates English into a target low-resource language and then back into English, using a combination of chrF++ and BLEU as the reward function on the reconstructed English sentences. Using the NLLB-MD dataset, we evaluate both the 600M and 1.3B parameter NLLB models and observe consistent improvements for the following languages: Central Aymara, Friulian, Wolof and Russian. Qualitative inspection of translation outputs indicates increased fluency and semantic fidelity. We argue that our method can further benefit from scale, enabling models to increasingly leverage their pretrained knowledge and continue self-improving. The code is available on github: this https URL.

**arXiv ID:** 2601.12535
</details>

<details>
<summary><strong>CARL: Focusing Agentic Reinforcement Learning on Critical Actions</strong> - Leyang Shen, Yang Zhang, Chun Kai Ling, Xiaoyan Zhao, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2512.04949)</summary>

**Abstract:** Agents capable of accomplishing complex tasks through multiple interactions with the environment have emerged as a popular research direction. However, in such multi-step settings, the conventional group-level policy optimization algorithm becomes suboptimal because of its underlying assumption that each action holds equal contribution, which deviates significantly from reality. Our analysis reveals that only a small fraction of actions are critical in determining the final outcome. Building on this insight, we propose CARL, a critical-action-focused reinforcement learning algorithm tailored for long-horizon agentic reasoning. CARL leverages entropy as a heuristic proxy for action criticality and achieves focused training by assigning rewards to high-criticality actions while excluding low-criticality actions from model updates, avoiding noisy credit assignment and redundant computation. Extensive experiments demonstrate that CARL achieves both stronger performance and higher efficiency across diverse evaluation settings. The source code will be publicly available.

**arXiv ID:** 2512.04949
</details>

<details>
<summary><strong>Distributional Reinforcement Learning with Diffusion Bridge Critics</strong> - Shutong Ding, Yimiao Zhou, Ke Hu, Mokai Pan, Shan Zhong, Yanwei Fu, Jingya Wang, Ye Shi - [[pdf]](https://arxiv.org/pdf/2602.05783)</summary>

**Abstract:** Recent advances in diffusion-based reinforcement learning (RL) methods have demonstrated promising results in a wide range of continuous control tasks. However, existing works in this field focus on the application of diffusion policies while leaving the diffusion critics unexplored. In fact, since policy optimization fundamentally relies on the critic, accurate value estimation is far more important than policy expressiveness. Furthermore, given the stochasticity of most reinforcement learning tasks, it has been confirmed that the critic is more appropriately depicted with a distributional model. Motivated by these points, we propose a novel distributional RL method with Diffusion Bridge Critics (DBC). DBC directly models the inverse cumulative distribution function (CDF) of the Q value. This allows us to accurately capture the value distribution and prevents it from collapsing into a trivial Gaussian distribution owing to the strong distribution-matching capability of the diffusion bridge. Moreover, we further derive an analytic integral formula to address discretization errors in DBC, which is essential in value estimation. To our knowledge, DBC is the first work to employ the diffusion bridge model as the critic. Notably, DBC is also a plug-and-play component and can be integrated into most existing RL frameworks. Experimental results on MuJoCo robot control benchmarks demonstrate the superiority of DBC compared with previous distributional critic models.

**arXiv ID:** 2602.05783
</details>

<details>
<summary><strong>$f$-GRPO and Beyond: Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment</strong> - Rajdeep Haldar, Lantao Mei, Guang Lin, Yue Xing, Qifan Song - [[pdf]](https://arxiv.org/pdf/2602.05946)</summary>

**Abstract:** Recent research shows that Preference Alignment (PA) objectives act as divergence estimators between aligned (chosen) and unaligned (rejected) response distributions. In this work, we extend this divergence-based perspective to general alignment settings, such as reinforcement learning with verifiable rewards (RLVR), where only environmental rewards are available. Within this unified framework, we propose $f$-Group Relative Policy Optimization ($f$-GRPO), a class of on-policy reinforcement learning, and $f$-Hybrid Alignment Loss ($f$-HAL), a hybrid on/off policy objectives, for general LLM alignment based on variational representation of $f$-divergences. We provide theoretical guarantees that these classes of objectives improve the average reward after alignment. Empirically, we validate our framework on both RLVR (Math Reasoning) and PA tasks (Safety Alignment), demonstrating superior performance and flexibility compared to current methods.

**arXiv ID:** 2602.05946
</details>

<details>
<summary><strong>On Computation and Reinforcement Learning</strong> - Raj Ghugare, Michał Bortkiewicz, Alicja Ziarko, Benjamin Eysenbach - [[pdf]](https://arxiv.org/pdf/2602.05999)</summary>

**Abstract:** How does the amount of compute available to a reinforcement learning (RL) policy affect its learning? Can policies using a fixed amount of parameters, still benefit from additional compute? The standard RL framework does not provide a language to answer these questions formally. Empirically, deep RL policies are often parameterized as neural networks with static architectures, conflating the amount of compute and the number of parameters. In this paper, we formalize compute bounded policies and prove that policies which use more compute can solve problems and generalize to longer-horizon tasks that are outside the scope of policies with less compute. Building on prior work in algorithmic learning and model-free planning, we propose a minimal architecture that can use a variable amount of compute. Our experiments complement our theory. On a set 31 different tasks spanning online and offline RL, we show that $(1)$ this architecture achieves stronger performance simply by using more compute, and $(2)$ stronger generalization on longer-horizon test tasks compared to standard feedforward networks or deep residual network using up to 5 times more parameters.

**arXiv ID:** 2602.05999
</details>

<details>
<summary><strong>Reinforcement Learning Enhancement Using Vector Semantic Representation and Symbolic Reasoning for Human-Centered Autonomous Emergency Braking</strong> - Vinal Asodia, Iman Sharifi, Saber Fallah - [[pdf]](https://arxiv.org/pdf/2602.05079)</summary>

**Abstract:** The problem with existing camera-based Deep Reinforcement Learning approaches is twofold: they rarely integrate high-level scene context into the feature representation, and they rely on rigid, fixed reward functions. To address these challenges, this paper proposes a novel pipeline that produces a neuro-symbolic feature representation that encompasses semantic, spatial, and shape information, as well as spatially boosted features of dynamic entities in the scene, with an emphasis on safety-critical road users. It also proposes a Soft First-Order Logic (SFOL) reward function that balances human values via a symbolic reasoning module. Here, semantic and spatial predicates are extracted from segmentation maps and applied to linguistic rules to obtain reward weights. Quantitative experiments in the CARLA simulation environment show that the proposed neuro-symbolic representation and SFOL reward function improved policy robustness and safety-related performance metrics compared to baseline representations and reward formulations across varying traffic densities and occlusion levels. The findings demonstrate that integrating holistic representations and soft reasoning into Reinforcement Learning can support more context-aware and value-aligned decision-making for autonomous driving.

**arXiv ID:** 2602.05079
</details>

<details>
<summary><strong>Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning</strong> - Ethan Rathbun, Wo Wei Lin, Alina Oprea, Christopher Amato - [[pdf]](https://arxiv.org/pdf/2602.05089)</summary>

**Abstract:** Simulated environments are a key piece in the success of Reinforcement Learning (RL), allowing practitioners and researchers to train decision making agents without running expensive experiments on real hardware. Simulators remain a security blind spot, however, enabling adversarial developers to alter the dynamics of their released simulators for malicious purposes. Therefore, in this work we highlight a novel threat, demonstrating how simulator dynamics can be exploited to stealthily implant action-level backdoors into RL agents. The backdoor then allows an adversary to reliably activate targeted actions in an agent upon observing a predefined ``trigger'', leading to potentially dangerous consequences. Traditional backdoor attacks are limited in their strong threat models, assuming the adversary has near full control over an agent's training pipeline, enabling them to both alter and observe agent's rewards. As these assumptions are infeasible to implement within a simulator, we propose a new attack ``Daze'' which is able to reliably and stealthily implant backdoors into RL agents trained for real world tasks without altering or even observing their rewards. We provide formal proof of Daze's effectiveness in guaranteeing attack success across general RL tasks along with extensive empirical evaluations on both discrete and continuous action space domains. We additionally provide the first example of RL backdoor attacks transferring to real, robotic hardware. These developments motivate further research into securing all components of the RL training pipeline to prevent malicious attacks.

**arXiv ID:** 2602.05089
</details>

<details>
<summary><strong>Residual Reinforcement Learning for Waste-Container Lifting Using Large-Scale Cranes with Underactuated Tools</strong> - Qi Li, Karsten Berns - [[pdf]](https://arxiv.org/pdf/2602.05895)</summary>

**Abstract:** This paper studies the container lifting phase of a waste-container recycling task in urban environments, performed by a hydraulic loader crane equipped with an underactuated discharge unit, and proposes a residual reinforcement learning (RRL) approach that combines a nominal Cartesian controller with a learned residual policy. All experiments are conducted in simulation, where the task is characterized by tight geometric tolerances between the discharge-unit hooks and the container rings relative to the overall crane scale, making precise trajectory tracking and swing suppression essential. The nominal controller uses admittance control for trajectory tracking and pendulum-aware swing damping, followed by damped least-squares inverse kinematics with a nullspace posture term to generate joint velocity commands. A PPO-trained residual policy in Isaac Lab compensates for unmodeled dynamics and parameter variations, improving precision and robustness without requiring end-to-end learning from scratch. We further employ randomized episode initialization and domain randomization over payload properties, actuator gains, and passive joint parameters to enhance generalization. Simulation results demonstrate improved tracking accuracy, reduced oscillations, and higher lifting success rates compared to the nominal controller alone.

**arXiv ID:** 2602.05895
</details>

<details>
<summary><strong>Modelling Pedestrian Behaviour in Autonomous Vehicle Encounters Using Naturalistic Dataset</strong> - Rulla Al-Haideri, Bilal Farooq - [[pdf]](https://arxiv.org/pdf/2602.05142)</summary>

**Abstract:** Understanding how pedestrians adjust their movement when interacting with autonomous vehicles (AVs) is essential for improving safety in mixed traffic. This study examines micro-level pedestrian behaviour during midblock encounters in the NuScenes dataset using a hybrid discrete choice-machine learning framework based on the Residual Logit (ResLogit) model. The model incorporates temporal, spatial, kinematic, and perceptual indicators. These include relative speed, visual looming, remaining distance, and directional collision risk proximity (CRP) measures. Results suggest that some of these variables may meaningfully influence movement adjustments, although predictive performance remains moderate. Marginal effects and elasticities indicate strong directional asymmetries in risk perception, with frontal and rear CRP showing opposite influences. The remaining distance exhibits a possible mid-crossing threshold. Relative speed cues appear to have a comparatively less effect. These patterns may reflect multiple behavioural tendencies driven by both risk perception and movement efficiency.

**arXiv ID:** 2602.05142
</details>

<details>
<summary><strong>RFS: Reinforcement Learning with Residual Flow Steering for Dexterous Manipulation</strong> - Entong Su, Tyler Westenbroek, Anusha Nagabandi, Abhishek Gupta - [[pdf]](https://arxiv.org/pdf/2602.01789)</summary>

**Abstract:** Imitation learning has emerged as an effective approach for bootstrapping sequential decision-making in robotics, achieving strong performance even in high-dimensional dexterous manipulation tasks. Recent behavior cloning methods further leverage expressive generative models, such as diffusion models and flow matching, to represent multimodal action distributions. However, policies pretrained in this manner often exhibit limited generalization and require additional fine-tuning to achieve robust performance at deployment time. Such adaptation must preserve the global exploration benefits of pretraining while enabling rapid correction of local execution errors. We propose Residual Flow Steering(RFS), a data-efficient reinforcement learning framework for adapting pretrained generative policies. RFS steers a pretrained flow-matching policy by jointly optimizing a residual action and a latent noise distribution, enabling complementary forms of exploration: local refinement through residual corrections and global exploration through latent-space modulation. This design allows efficient adaptation while retaining the expressive structure of the pretrained policy. We demonstrate the effectiveness of RFS on dexterous manipulation tasks, showing efficient fine-tuning in both simulation and real-world settings when adapting pretrained base policies. Project website:this https URL.

**arXiv ID:** 2602.01789
</details>

<details>
<summary><strong>HoRD: Robust Humanoid Control via History-Conditioned Reinforcement Learning and Online Distillation</strong> - Puyue Wang, Jiawei Hu, Yan Gao, Junyan Wang, Yu Zhang, Gillian Dobbie, Tao Gu, Wafa Johal, Ting Dang, Hong Jia - [[pdf]](https://arxiv.org/pdf/2602.04412)</summary>

**Abstract:** Humanoid robots can suffer significant performance drops under small changes in dynamics, task specifications, or environment setup. We propose HoRD, a two-stage learning framework for robust humanoid control under domain shift. First, we train a high-performance teacher policy via history-conditioned reinforcement learning, where the policy infers latent dynamics context from recent state--action trajectories to adapt online to diverse randomized dynamics. Second, we perform online distillation to transfer the teacher's robust control capabilities into a transformer-based student policy that operates on sparse root-relative 3D joint keypoint trajectories. By combining history-conditioned adaptation with online distillation, HoRD enables a single policy to adapt zero-shot to unseen domains without per-domain retraining. Extensive experiments show HoRD outperforms strong baselines in robustness and transfer, especially under unseen domains and external perturbations. Code and project page are available at this https URL.

**arXiv ID:** 2602.04412
</details>

<details>
<summary><strong>MindDrive: A Vision-Language-Action Model for Autonomous Driving via Online Reinforcement Learning</strong> - Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Hongwei Xie, Bing Wang, Guang Chen, Dingkang Liang, Xiang Bai - [[pdf]](https://arxiv.org/pdf/2512.13636)</summary>

**Abstract:** Current Vision-Language-Action (VLA) paradigms in autonomous driving primarily rely on Imitation Learning (IL), which introduces inherent challenges such as distribution shift and causal confusion. Online Reinforcement Learning offers a promising pathway to address these issues through trial-and-error learning. However, applying online reinforcement learning to VLA models in autonomous driving is hindered by inefficient exploration in continuous action spaces. To overcome this limitation, we propose MindDrive, a VLA framework comprising a large language model (LLM) with two distinct sets of LoRA parameters. The one LLM serves as a Decision Expert for scenario reasoning and driving decision-making, while the other acts as an Action Expert that dynamically maps linguistic decisions into feasible trajectories. By feeding trajectory-level rewards back into the reasoning space, MindDrive enables trial-and-error learning over a finite set of discrete linguistic driving decisions, instead of operating directly in a continuous action space. This approach effectively balances optimal decision-making in complex scenarios, human-like driving behavior, and efficient exploration in online reinforcement learning. Using the lightweight Qwen-0.5B LLM, MindDrive achieves Driving Score (DS) of 78.04 and Success Rate (SR) of 55.09% on the challenging Bench2Drive benchmark. To the best of our knowledge, this is the first work to demonstrate the effectiveness of online reinforcement learning for the VLA model in autonomous driving.

**arXiv ID:** 2512.13636
</details>

<details>
<summary><strong>A Design Space for Live Music Agents</strong> - Yewon Kim, Stephen Brade, Alexander Wang, David Zhou, Haven Kim, Bill Wang, Sung-Ju Lee, Hugo F Flores Garcia, Cheng-Zhi Anna Huang, Chris Donahue - [[pdf]](https://arxiv.org/pdf/2602.05064)</summary>

**Abstract:** Live music provides a uniquely rich setting for studying creativity and interaction due to its spontaneous nature. The pursuit of live music agents--intelligent systems supporting real-time music performance and interaction--has captivated researchers across HCI, AI, and computer music for decades, and recent advancements in AI suggest unprecedented opportunities to evolve their design. However, the interdisciplinary nature of music has led to fragmented development across research communities, hindering effective communication and collaborative progress. In this work, we bring together perspectives from these diverse fields to map the current landscape of live music agents. Based on our analysis of 184 systems across both academic literature and video, we develop a comprehensive design space that categorizes dimensions spanning usage contexts, interactions, technologies, and ecosystems. By highlighting trends and gaps in live music agents, our design space offers researchers, designers, and musicians a structured lens to understand existing systems and shape future directions in real-time human-AI music co-creation. We release our annotated systems as a living artifact at this https URL.

**arXiv ID:** 2602.05064
</details>

<details>
<summary><strong>TaskAudit: Detecting Functiona11ity Errors in Mobile Apps via Agentic Task Execution</strong> - Mingyuan Zhong, Xia Chen, Davin Win Kyi, Chen Li, James Fogarty, Jacob O. Wobbrock - [[pdf]](https://arxiv.org/pdf/2510.12972)</summary>

**Abstract:** Accessibility checkers are tools in support of accessible app development, and their use is encouraged by accessibility best practices. However, most current checkers evaluate static or mechanically-generated contexts, failing to capture common accessibility errors impacting mobile app functionality. In this work, we define functiona11ity errors as accessibility barriers that only manifest through interaction (i.e., named according to a blend of "functionality" and "accessibility"). We introduce TaskAudit, which comprises three components: a Task Generator that constructs interactive tasks from app screens, a Task Executor that uses agents with a screen reader proxy to perform these tasks, and an Accessibility Analyzer that detects and reports accessibility errors by examining interaction traces. Our evaluation on real-world apps shows that TaskAudit detects 48 functiona11ity errors from 54 app screens, compared to between 4 and 20 with existing checkers. Our analysis demonstrates common error patterns that TaskAudit can detect in addition to those from prior work, including label-functionality mismatch, cluttered navigation, and inappropriate feedback.

**arXiv ID:** 2510.12972
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
