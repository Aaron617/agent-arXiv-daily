# Agent arXiv Daily

**Last Updated:** 2026-04-21 03:43:56

**Total Papers:** 153

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (12 papers)</h2></summary>

<details>
<summary><strong>Governing the Agentic Enterprise: A Governance Maturity Model for Managing AI Agent Sprawl in Business Operations</strong> - Vivek Acharya - [[pdf]](https://arxiv.org/pdf/2604.16338)</summary>

**Abstract:** The rapid adoption of agentic AI in enterprise business operations--autonomous systems capable of planning, reasoning, and executing multi-step workflows--has created an urgent governance crisis. Organizations face uncontrolled agent sprawl: the proliferation of redundant, ungoverned, and conflicting AI agents across business functions. Industry surveys report that only 21% of enterprises have mature governance models for autonomous agents, while 40% of agentic AI projects are projected to fail by 2027 due to inadequate governance and risk controls. Despite growing acknowledgment of this challenge, academic literature lacks a formal, empirically validated governance maturity model connecting governance capability to measurable business outcomes. This paper introduces the Agentic AI Governance Maturity Model (AAGMM), a five-level framework spanning 12 governance domains, grounded in NIST AI RMF and ISO/IEC 42001 standards. We additionally propose a novel taxonomy of agent sprawl patterns--functional duplication, shadow agents, orphaned agents, permission creep, and unmonitored delegation chains--each linked to quantifiable business cost models. The framework is validated through 750 simulation runs across five enterprise scenarios and five governance maturity levels, measuring business outcomes including cost containment, risk incident rates, operational efficiency, and decision quality. Results demonstrate statistically significant differences (p < 0.001, large effect sizes d > 2.0) between all governance maturity levels, with Level 4-5 organizations achieving 94.3% lower sprawl indices, 96.4% fewer risk incidents, and 32.6% higher effective task completion rates compared to Level 1. The AAGMM provides practitioners with an actionable roadmap for governing autonomous AI agents while maximizing business returns.

**arXiv ID:** 2604.16338
</details>

<details>
<summary><strong>Know When to Trust the Skill: Delayed Appraisal and Epistemic Vigilance for Single-Agent LLMs</strong> - Eren Unlu - [[pdf]](https://arxiv.org/pdf/2604.16753)</summary>

**Abstract:** As large language models (LLMs) transition into autonomous agents integrated with extensive tool ecosystems, traditional routing heuristics increasingly succumb to context pollution and "overthinking". We argue that the bottleneck is not a deficit in algorithmic capability or skill diversity, but the absence of disciplined second-order metacognitive governance. In this paper, our scientific contribution focuses on the computational translation of human cognitive control - specifically, delayed appraisal, epistemic vigilance, and region-of-proximal offloading - into a single-agent architecture. We introduce MESA-S (Metacognitive Skills for Agents, Single-agent), a preliminary framework that shifts scalar confidence estimation into a vector separating self-confidence (parametric certainty) from source-confidence (trust in retrieved external procedures). By formalizing a delayed procedural probe mechanism and introducing Metacognitive Skill Cards, MESA-S decouples the awareness of a skill's utility from its token-intensive execution. Evaluated under an In-Context Static Benchmark Evaluation natively executed via Gemini 3.1 Pro, our early results suggest that explicitly programming trust provenance and delayed escalation mitigates supply-chain vulnerabilities, prunes unnecessary reasoning loops, and prevents offloading-induced confidence inflation. This architecture offers a scientifically cautious, behaviorally anchored step toward reliable, epistemically vigilant single-agent orchestration.

**arXiv ID:** 2604.16753
</details>

<details>
<summary><strong>SkillFlow:Benchmarking Lifelong Skill Discovery and Evolution for Autonomous Agents</strong> - Ziao Zhang, Kou Shi, Shiting Huang, Avery Nie, Yu Zeng, Yiming Zhao, Zhen Fang, Qishen Su, Haibo Qiu, Wei Yang, Qingnan Ren, Shun Zou, Wenxuan Huang, Lin Chen, Zehui Chen, Feng Zhao - [[pdf]](https://arxiv.org/pdf/2604.17308)</summary>

**Abstract:** As the capability frontier of autonomous agents continues to expand, they are increasingly able to complete specialized tasks through plug-and-play external skills. Yet current benchmarks mostly test whether models can use provided skills, leaving open whether they can discover skills from experience, repair them after failure, and maintain a coherent library over time. We introduce SkillFlow, a benchmark of 166 tasks across 20 families in which task construction within each family follows a Domain-Agnostic Execution Flow (DAEF) that defines an agent workflow framework, allowing these tasks to share a consistent workflow. Agents are evaluated under an Agentic Lifelong Learning protocol in which they begin without skills, solve tasks sequentially within each family, externalize lessons through trajectory- and rubric-driven skill patches, and carry the updated library forward. Experiments reveal a substantial capability gap. For Claude Opus 4.6, lifelong skill evolution improves task success from 62.65% to 71.08% (+8.43 points). However, high skill usage does not necessarily imply high utility: Kimi K2.5 gains only +0.60 points despite 66.87% skill usage, while Qwen-Coder-Next reaches only a 44.58% task completion rate and still regresses relative to the vanilla setting. SkillFlow contributes a structured testbed for this direction and an in-depth empirical analysis of skill discovery, patching, transfer, and their failure modes under lifelong evaluation.

**arXiv ID:** 2604.17308
</details>

<details>
<summary><strong>From Admission to Invariants: Measuring Deviation in Delegated Agent Systems</strong> - Marcelo Fernandez - [[pdf]](https://arxiv.org/pdf/2604.17517)</summary>

**Abstract:** Autonomous agent systems are governed by enforcement mechanisms that flag hard constraint violations at runtime. The Agent Control Protocol identifies a structural limit of such systems: a correctly-functioning enforcement engine can enter a regime in which behavioral drift is invisible to it, because the enforcement signal operates below the layer where deviation is measurable. We show that enforcement-based governance is structurally unable to determine whether an agent's behavior remains within the admissible behavior space A0 established at admission time. Our central result, the Non-Identifiability Theorem, proves that A0 is not in the sigma-algebra generated by the enforcement signal g under the Local Observability Assumption, which every practical enforcement system satisfies. The impossibility arises from a fundamental mismatch: g evaluates actions locally against a point-wise rule set, while A0 encodes global, trajectory-level behavioral properties set at admission time. We define the Invariant Measurement Layer (IML), which bypasses this limitation by retaining direct access to the generative model of A0. We prove an information-theoretic impossibility for enforcement-based monitoring; separately, we show IML detects admission-time drift with provably finite detection delay, operating in the region where enforcement is structurally blind. Validated across four settings: three drift scenarios (300 and 1000 steps), a live n8n webhook pipeline, and a LangGraph StateGraph agent -- enforcement triggers zero violations while IML detects each drift type within 9-258 steps. Paper 2 of a 4-paper Agent Governance Series: atomic boundaries (P0, https://doi.org/10.5281/zenodo.19642166), ACP enforcement (P1, arXiv:2603.18829), fair allocation (P3, https://doi.org/10.5281/zenodo.19643928), irreducibility (P4, https://doi.org/10.5281/zenodo.19643950).

**arXiv ID:** 2604.17517
</details>

<details>
<summary><strong>WebUncertainty: Dual-Level Uncertainty Driven Planning and Reasoning For Autonomous Web Agent</strong> - Lingfeng Zhang, yongan sun, Jinpeng Hu, Hui Ma, yang ying, Kuien Liu, Zenglin Shi, Meng Wang - [[pdf]](https://arxiv.org/pdf/2604.17821)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have empowered autonomous web agents to execute natural language instructions directly on real-world webpages. However, existing agents often struggle with complex tasks involving dynamic interactions and long-horizon execution due to rigid planning strategies and hallucination-prone reasoning. To address these limitations, we propose WebUncertainty, a novel autonomous agent framework designed to tackle dual-level uncertainty in planning and reasoning. Specifically, we design a Task Uncertainty-Driven Adaptive Planning Mechanism that adaptively selects planning modes to navigate unknown environments. Furthermore, we introduce an Action Uncertainty-Driven Monte Carlo tree search (MCTS) Reasoning Mechanism. This mechanism incorporates the Confidence-induced Action Uncertainty (ConActU) strategy to quantify both aleatoric uncertainty (AU) and epistemic uncertainty (EU), thereby optimizing the search process and guiding robust decision-making. Experimental results on the WebArena and WebVoyager benchmarks demonstrate that WebUncertainty achieves superior performance compared to state-of-the-art baselines.

**arXiv ID:** 2604.17821
</details>

<details>
<summary><strong>Same Verdict, Different Reasons: LLM-as-a-Judge and Clinician Disagreement on Medical Chatbot Completeness</strong> - Alexandra DeLucia, Heyuan Huang, Sonal Joshi, Mahsa Yarmohammadi, Ahmed Hassoon, Mark Dredze - [[pdf]](https://arxiv.org/pdf/2604.16383)</summary>

**Abstract:** LLM-as-a-Judge frameworks are increasingly trusted to automate evaluation in place of human experts, yet their reliability in high-stakes medical contexts remains unproven. We stress-test this assumption for detecting incomplete patient-facing medical responses, evaluating three rubric granularities (General-Likert, Analytical-Rubric, Dynamic-Checklist) and three backbone models across two clinician-annotated datasets, including HealthBench, the largest publicly available benchmark for medical response evaluation. LLM Judges discriminate complete from incomplete responses at and slightly above near chance (AUC $0.49$--$0.66$); at the threshold required to recall $90\%$ of incomplete responses, clinicians must still review the vast majority of the dataset, offering no triage utility. Even when model and clinician verdicts agree, they rarely cite the same explanation; and when they diverge, false positives stem from over-flagging non-essential gaps while false negatives reflect outright detection failures. These results reveal that LLM Judges and clinicians apply fundamentally different completeness standards; a finding that undermines their use as autonomous evaluators or triage filters in clinical settings.

**arXiv ID:** 2604.16383
</details>

<details>
<summary><strong>FreezeEmpath: Efficient Training for Empathetic Spoken Chatbots with Frozen LLMs</strong> - Yun Hong, Yan Zhou, Yang Feng - [[pdf]](https://arxiv.org/pdf/2604.18159)</summary>

**Abstract:** Empathy is essential for fostering natural interactions in spoken dialogue systems, as it enables machines to recognize the emotional tone of human speech and deliver empathetic responses. Recent research has made significant progress in developing empathetic spoken chatbots based on large language models (LLMs). However, several challenges still exist when training such models, including reliance on costly empathetic speech instruction data and a lack of emotional expressiveness in the generated speech. Finetuning LLM with cross-modal empathetic instruction data may also lead to catastrophic forgetting and a degradation of its general capability. To address these challenges, we propose FreezeEmpath, an end-to-end empathetic spoken chatbot trained in a simple and efficient manner. The entire training process relies solely on existing speech instruction data and speech emotion recognition (SER) data, while keeping the LLM's parameters frozen. Experiments demonstrate that FreezeEmpath is able to generate emotionally expressive speech and outperforms other empathetic models in empathetic dialogue, SER, and SpokenQA tasks, demonstrating the effectiveness of our training strategy.

**arXiv ID:** 2604.18159
</details>

<details>
<summary><strong>HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents</strong> - Shuqi Cao, Jingyi He, Fei Tan - [[pdf]](https://arxiv.org/pdf/2604.18349)</summary>

**Abstract:** Long-term conversational large language model (LLM) agents require memory systems that can recover relevant evidence from historical interactions without overwhelming the answer stage with irrelevant context. However, existing memory systems, including hierarchical ones, still often rely solely on vector similarity for retrieval. It tends to produce bloated evidence sets: adding many superficially similar dialogue turns yields little additional recall, but lowers retrieval precision, increases answer-stage context cost, and makes retrieved memories harder to inspect and manage. To address this, we propose HiGMem (Hierarchical and LLM-Guided Memory System), a two-level event-turn memory system that allows LLMs to use event summaries as semantic anchors to predict which related turns are worth reading. This allows the model to inspect high-level event summaries first and then focus on a smaller set of potentially useful turns, providing a concise and reliable evidence set through reasoning, while avoiding the retrieval overhead that would be excessively high compared to vector retrieval. On the LoCoMo10 benchmark, HiGMem achieves the best F1 on four of five question categories and improves adversarial F1 from 0.54 to 0.78 over A-Mem, while retrieving an order of magnitude fewer turns. Code is publicly available at this https URL.

**arXiv ID:** 2604.18349
</details>

<details>
<summary><strong>IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters</strong> - Hongwei Zheng, Weiqi Wu, Zhengjia Wang, Guanyu Jiang, Haoming Li, Tianyu Wu, Yongchun Zhu, Jingwu Chen, Feng Zhang - [[pdf]](https://arxiv.org/pdf/2604.18375)</summary>

**Abstract:** Conversational agents, such as ChatGPT and Doubao, have become essential daily assistants for billions of users. To further enhance engagement, these systems are evolving from passive responders to proactive companions. However, existing efforts focus on activation within ongoing dialogues, while overlooking a key real-world bottleneck. In the conversation initiation stage, users may have a vague need but no explicit query intent, creating a first-message barrier where the conversation holds before it begins. To overcome this, we introduce Conversation Starter Generation: generating personalized starters to guide users into conversation. However, unlike in-conversation stages where immediate context guides the response, initiation must operate in a cold-start moment without explicit user intent. To pioneer in this direction, we present IceBreaker that frames human ice-breaking as a two-step handshake: (i) evoke resonance via Resonance-Aware Interest Distillation from session summaries to capture trigger interests, and (ii) stimulate interaction via Interaction-Oriented Starter Generation, optimized with personalized preference alignment and a self-reinforced loop to maximize engagement. Online A/B tests on one of the world's largest conversational agent products show that IceBreaker improves user active days by +0.184% and click-through rate by +9.425%, and has been deployed in production.

**arXiv ID:** 2604.18375
</details>

<details>
<summary><strong>XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments</strong> - Kangan Qian, ChuChu Xie, Yang Zhong, Jingrui Pang, Siwen Jiao, Sicong Jiang, Zilin Huang, Yunlong Wang, Kun Jiang, Mengmeng Yang, Hao Ye, Guanghao Zhang, Hangjun Ye, Guang Chen, Long Chen, Diange Yang - [[pdf]](https://arxiv.org/pdf/2604.18484)</summary>

**Abstract:** Vision-Language-Action (VLA) models drive next-generation autonomous systems, but training them requires scalable, high-quality annotations from complex environments. Current cloud pipelines rely on generic vision-language models (VLMs) that lack geometric reasoning and domain semantics due to their 2D image-text pretraining. To address this mismatch, we propose XEmbodied, a cloud-side foundation model that endows VLMs with intrinsic 3D geometric awareness and interaction with physical cues (e.g., occupancy grids, 3D boxes). Instead of treating geometry as auxiliary input, XEmbodied integrates geometric representations via a structured 3D Adapter and distills physical signals into context tokens using an Efficient Image-Embodied Adapter. Through progressive domain curriculum and reinforcement learning post-training, XEmbodied preserves general capabilities while demonstrating robust performance across 18 public benchmarks. It significantly improves spatial reasoning, traffic semantics, embodied affordance, and out-of-distribution generalization for large-scale scenario mining and embodied VQA.

**arXiv ID:** 2604.18484
</details>

<details>
<summary><strong>Everything Counts: The Managed Omnirelevance of Speech in Human-Voice Agent Interaction</strong> - Damien Rudaz, Mathias Broth, Jakub Mlynar - [[pdf]](https://arxiv.org/pdf/2510.22610)</summary>

**Abstract:** To this day, turn-taking models determining voice agents' conduct have been examined primarily from a technical point of view, while the ways in which they emerge as interactional constraints or resources for human conversationalists in situ remain underexplored. Drawing on a detailed analysis of corpora of naturalistic data, we document how humans' conduct was produced in reference to the ever-present risk that, each time they spoke, their talk might trigger a new uncalled-for contribution from the artificial agent. We examine this phenomenon in interactions involving rule-based robots from a 'pre-LLM era' as well as the most recent voice agents. This 'omnirelevance of human speech' (i.e., the possibility that a conversational agent may erroneously respond to any speech it detects) emerged as a constitutive feature of these human-agent encounters. We describe some of the practices through which humans managed these artificial agents' turn-taking conduct. Given recent improvements in voice capture technology, we ask whether this 'omnirelevance of human speech' weighs even more heavily on human practices today than in the past.

**arXiv ID:** 2510.22610
</details>

<details>
<summary><strong>ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows</strong> - Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, Jianing Wang, Qintong Li, Xiangru Tang, Tianbao Xie, Xiachong Feng, Xiang Li, Ben Kao, Wenhai Wang, Biqing Qi, Lingpeng Kong, Zhiyong Wu - [[pdf]](https://arxiv.org/pdf/2505.19897)</summary>

**Abstract:** Large Language Models (LLMs) have extended their impact beyond Natural Language Processing, substantially fostering the development of interdisciplinary research. Recently, various LLM-based agents have been developed to assist scientific discovery progress across multiple aspects and domains. Among these, computer-using agents, capable of interacting with operating systems as humans do, are paving the way to automated scientific problem-solving and addressing routines in researchers' workflows. Recognizing the transformative potential of these agents, we introduce ScienceBoard, which encompasses two complementary contributions: (i) a realistic, multi-domain environment featuring dynamic and visually rich scientific workflows with integrated professional software, where agents can autonomously interact via different interfaces to accelerate complex research tasks and experiments; and (ii) a challenging benchmark of 169 high-quality, rigorously validated real-world tasks curated by humans, spanning scientific-discovery workflows in domains such as biochemistry, astronomy, and geoinformatics. Extensive evaluations of agents with state-of-the-art backbones (e.g., GPT-4o, Claude 3.7, UI-TARS) show that, despite some promising results, they still fall short of reliably assisting scientists in complex workflows, achieving only a 15% overall success rate. In-depth analysis further provides valuable insights for addressing current agent limitations and more effective design principles, paving the way to build more capable agents for scientific discovery. Our code, environment, and benchmark are at this https URL.

**arXiv ID:** 2505.19897
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (30 papers)</h2></summary>

<details>
<summary><strong>Agentic Frameworks for Reasoning Tasks: An Empirical Study</strong> - Zeeshan Rasheed, Abdul Malik Sami, Muhammad Waseem, Kai-Kristian Kemell, Mika Saari, Pekka Abrahamsson - [[pdf]](https://arxiv.org/pdf/2604.16646)</summary>

**Abstract:** Recent advances in agentic frameworks have enabled AI agents to perform complex reasoning and decision-making. However, evidence comparing their reasoning performance, efficiency, and practical suitability remains limited. To address this gap, we empirically evaluate 22 widely used agentic frameworks across three reasoning benchmarks: BBH, GSM8K, and ARC. The frameworks were selected from 1,200 GitHub repositories collected between January 2023 and July 2025 and organized into a taxonomy based on architectural design. We evaluated them under a unified setting, measuring reasoning accuracy, execution time, computational cost, and cross-benchmark consistency.
Our results show that 19 of the 22 frameworks completed all three benchmarks. Among these, 12 showed stable performance, with mean accuracy of 74.6-75.9%, execution time of 4-6 seconds per task, and cost of 0.14-0.18 cents per task. Poorer results were mainly caused by orchestration problems rather than reasoning limits. For example, Camel failed to complete BBH after 11 days because of uncontrolled context growth, while Upsonic consumed USD 1,434 in one day because repeated extraction failures triggered costly retries. AutoGen and Mastra also exhausted API quotas through iterative interactions that increased prompt length without improving results.
We also found a sharp drop in mathematical reasoning. Mean accuracy on GSM8K was 44.35%, compared with 89.80% on BBH and 89.56% on ARC. Overall, this study provides the first large-scale empirical comparison of agentic frameworks for reasoning-intensive software engineering tasks and shows that framework selection should prioritize orchestration quality, especially memory control, failure handling, and cost management.

**arXiv ID:** 2604.16646
</details>

<details>
<summary><strong>Evaluating Tool-Using Language Agents: Judge Reliability, Propagation Cascades, and Runtime Mitigation in AgentProp-Bench</strong> - Bhaskar Gurram - [[pdf]](https://arxiv.org/pdf/2604.16706)</summary>

**Abstract:** Automated evaluation of tool-using large language model (LLM) agents is widely assumed to be reliable, but this assumption has rarely been validated against human annotation. We introduce AgentProp-Bench, a 2,000-task benchmark with 2,300 traces across four domains, nine production LLMs, and a 100-label human-validated subset. We quantify judge reliability, characterize error propagation, and evaluate a runtime mitigation. Substring-based judging agrees with human annotation at kappa=0.049 (chance-level); a three-LLM ensemble reaches kappa=0.432 (moderate) with a conservative bias. Under validated evaluation, a parameter-level injection propagates to a wrong final answer with human-calibrated probability approximately 0.62 (range 0.46-0.73 across models). Rejection (catching bad parameters) and recovery (correcting after acceptance) are independent model capabilities (Spearman rho=0.126, p=0.747). A tuned runtime interceptor reduces hallucination on GPT-4o-mini by 23.0 percentage points under a concurrent n=600 control, but shows no significant effect on Gemini-2.0-Flash, whose aggressive parameter rejection eliminates the target failure mode. All code, data, traces, and human labels are released at this https URL.

**arXiv ID:** 2604.16706
</details>

<details>
<summary><strong>ClimAgent: LLM as Agents for Autonomous Open-ended Climate Science Analysis</strong> - Hao Wang, Jindong Han, Wei Fan, Hao Liu - [[pdf]](https://arxiv.org/pdf/2604.16922)</summary>

**Abstract:** Climate research is pivotal for mitigating global environmental crises, yet the accelerating volume of multi-scale datasets and the complexity of analytical tools have created significant bottlenecks, constraining scientific discovery to fragmented and labor-intensive workflows. While the emergence Large Language Models (LLMs) offers a transformative paradigm to scale scientific expertise, existing explorations remain largely confined to simple Question-Answering (Q&A) tasks. These approaches often oversimplify real-world challenges, neglecting the intricate physical constraints and the data-driven nature required in professional climate this http URL bridge this gap, we introduce ClimAgent, a general-purpose autonomous framework designed to execute a wide spectrum of research tasks across diverse climate sub-fields. By integrating a unified tool-use environment with rigorous reasoning protocols, ClimAgent transcends simple retrieval to perform end-to-end modeling and this http URL foster systematic evaluation, we propose ClimaBench, the first comprehensive benchmark for real-world climate discovery. It encompasses challenging problems spanning 5 distinct task categories derived from professional scenarios between 2000 and 2025. Experiments on ClimaBench demonstrate that ClimAgent significantly outperforms state-of-the-art baselines, achieving a 40.21% improvement over original LLM solutions in solution rigorousness and practicality. Our code are available at this https URL.

**arXiv ID:** 2604.16922
</details>

<details>
<summary><strong>Mini-BEHAVIOR-Gran: Revealing U-Shaped Effects of Instruction Granularity on Language-Guided Embodied Agents</strong> - Sukai Huang, Chenyuan Zhang, Fucai Ke, Zhixi Cai, Gholamreza Haffari, Lizhen Qu, Hamid Rezatofighi - [[pdf]](https://arxiv.org/pdf/2604.17019)</summary>

**Abstract:** Instruction granularity is an important yet poorly controlled variable in language-guided embodied AI. Existing benchmarks typically pair each task with a single static instruction, making it difficult to study how agent behavior changes when the same task is described at different levels of detail. We introduce Mini-BEHAVIOR-Gran, a new benchmark for controlled studies of instruction granularity that extends Mini-BEHAVIOR with multiple instruction variants per task, ranging from high-level goal descriptions to step-by-step guidance. Using this benchmark, we compare four candidate metrics for cross-task granularity quantification: token count, entity count, action-verb count, and planning-width, and find that width correlates most consistently with agent performance. Using width to organize training and evaluation further reveals a non-monotonic U-shaped relationship between instruction granularity and performance, with peaks at both fine and coarse extremes. Further analysis suggests that the coarse-granularity performance rebound is associated with shallow grounding, where agents learn vision-dominant policies.

**arXiv ID:** 2604.17019
</details>

<details>
<summary><strong>HalluClear: Diagnosing, Evaluating and Mitigating Hallucinations in GUI Agents</strong> - Chao Jin, Wenkui Yang, Hao Sun, Yuqi Liao, Qianyi Jiang, Kai Zhou, Jie Cao, Ran He, Huaibo Huang - [[pdf]](https://arxiv.org/pdf/2604.17284)</summary>

**Abstract:** While progress in GUI agents has been largely driven by industrial-scale training, ungrounded hallucinations often trigger cascading failures in real-world this http URL general VLM domains, the GUI agent field lacks a hallucination-focused suite for fine-grained diagnosis, reliable evaluation, and targeted this http URL bridge this gap, we introduce HalluClear, a comprehensive suite for hallucination mitigation in GUI agents as a complement to computation-intensive scaling. HalluClear comprises: (1) a GUI-specific hallucination taxonomy derived from empirical failure analysis; (2) a calibrated three-stage evaluation workflow which enhances VLM-as-a-judge reliability via expert-annotated benchmarking and ensemble credibility estimation; and (3) a mitigation scheme based on closed-loop structured reasoning, enabling lightweight continual post-training with cold-start initialization for both generalist and GUI-specialist agents. Experiments across representative agents and public benchmarks demonstrate that post-training on only 9K samples within our suite can significantly reduce hallucinations, thereby improving grounding and action fidelity, offering a compute-efficient pathway to robust GUI automation.

**arXiv ID:** 2604.17284
</details>

<details>
<summary><strong>EvoMaster: A Foundational Agent Framework for Building Evolving Autonomous Scientific Agents at Scale</strong> - Xinyu Zhu, Yuzhu Cai, Zexi Liu, Cheng Wang, Fengyang Li, Wenkai Jin, Wanxu Liu, Zehao Bing, Bingyang Zheng, Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xianghe Pang, Yaxin Du, Tingjia Miao, Yuzhi Zhang, Ruoxue Liao, Zhaohan Ding, Linfeng Zhang, Yanfeng Wang, Weinan E, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2604.17406)</summary>

**Abstract:** The convergence of large language models and agents is catalyzing a new era of scientific discovery: Agentic Science. While the scientific method is inherently iterative, existing agent frameworks are predominantly static, narrowly scoped, and lack the capacity to learn from trial and error. To bridge this gap, we present EvoMaster, a foundational evolving agent framework engineered specifically for Agentic Science at Scale. Driven by the core principle of continuous self-evolution, EvoMaster empowers agents to iteratively refine hypotheses, self-critique, and progressively accumulate knowledge across experimental cycles, faithfully mirroring human scientific inquiry. Crucially, as a domain-agnostic base harness, EvoMaster is exceptionally easy to scale up -- enabling developers to build and deploy highly capable, self-evolving scientific agents for arbitrary disciplines in approximately 100 lines of code. Built upon EvoMaster, we incubated the SciMaster ecosystem across domains such as machine learning, physics, and general science. Evaluations on four authoritative benchmarks (Humanity's Last Exam, MLE-Bench Lite, BrowseComp, and FrontierScience) demonstrate that EvoMaster achieves state-of-the-art scores of 41.1%, 75.8%, 73.3%, and 53.3%, respectively. It comprehensively outperforms the general-purpose baseline OpenClaw with relative improvements ranging from +159% to +316%, robustly validating its efficacy and generality as the premier foundational framework for the next generation of autonomous scientific discovery. EvoMaster is available at this https URL.

**arXiv ID:** 2604.17406
</details>

<details>
<summary><strong>SafeAgent: A Runtime Protection Architecture for Agentic Systems</strong> - Hailin Liu, Eugene Ilyushin, Jie Ni, Min Zhu - [[pdf]](https://arxiv.org/pdf/2604.17562)</summary>

**Abstract:** Large language model (LLM) agents are vulnerable to prompt-injection attacks that propagate through multi-step workflows, tool interactions, and persistent context, making input-output filtering alone insufficient for reliable protection. This paper presents SafeAgent, a runtime security architecture that treats agent safety as a stateful decision problem over evolving interaction trajectories. The proposed design separates execution governance from semantic risk reasoning through two coordinated components: a runtime controller that mediates actions around the agent loop and a context-aware decision core that operates over persistent session state. The core is formalized as a context-aware advanced machine intelligence and instantiated through operators for risk encoding, utility-cost evaluation, consequence modeling, policy arbitration, and state synchronization. Experiments on Agent Security Bench (ASB) and InjecAgent show that SafeAgent consistently improves robustness over baseline and text-level guardrail methods while maintaining competitive benign-task performance. Ablation studies further show that recovery confidence and policy weighting determine distinct safety-utility operating points.

**arXiv ID:** 2604.17562
</details>

<details>
<summary><strong>Beyond Static Snapshots: A Grounded Evaluation Framework for Language Models at the Agentic Frontier</strong> - Jazmia Henry - [[pdf]](https://arxiv.org/pdf/2604.17573)</summary>

**Abstract:** We argue that current evaluation frameworks for large language models (LLMs) suffer from four systematic failures that make them structurally inadequate for assessing deployed, agentic systems: distributional invalidity (evaluation inputs do not reflect real interaction distributions), temporal invalidity (evaluations are post-hoc rather than training-integrated), scope invalidity (evaluations measure single-turn outputs rather than long-horizon trajectories), and process invalidity (evaluations assess outputs rather than reasoning). These failures compound critically in RLHF, where reward models are evaluated under conditions that do not hold during RL training, making reward hacking a predictable consequence of evaluation design rather than a training pathology. We propose the Grounded Continuous Evaluation (GCE) framework and present ISOPro, a simulation-based fine-tuning and evaluation system. ISOPro replaces the learned reward model with a deterministic ground-truth verifier, eliminating reward hacking by construction in verifiable-reward domains, and operates on LoRA adapter weights updatable on CPU, reducing the hardware barrier by an order of magnitude. We validate ISOPro on a resource-constrained scheduling domain with six difficulty tiers, demonstrating capability emergence visible only through continuous evaluation, an implicit curriculum that forms without researcher curation, and a 3x accuracy improvement over zero-shot baselines, all on consumer hardware with 0.216% trainable parameters.

**arXiv ID:** 2604.17573
</details>

<details>
<summary><strong>PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents</strong> - Yuan Tian, Tianyi Zhang - [[pdf]](https://arxiv.org/pdf/2604.17653)</summary>

**Abstract:** Text-to-SQL systems often struggle with deep contextual understanding, particularly for complex queries with subtle requirements. We present PV-SQL, an agentic framework that addresses these failures through two complementary components: Probe and Verify. The Probe component iteratively generates probing queries to retrieve concrete records from the database, resolving ambiguities in value formats, column semantics, and inter-table relationships to build richer contextual understanding. The Verify component employs a rule-based method to extract verifiable conditions and construct an executable checklist, enabling iterative SQL refinement that effectively reduces missing constraints. Experiments on the BIRD benchmarks show that PV-SQL outperforms the best text-to-SQL baseline by 5% in execution accuracy and 20.8% in valid efficiency score while consuming fewer tokens.

**arXiv ID:** 2604.17653
</details>

<details>
<summary><strong>Using large language models for embodied planning introduces systematic safety risks</strong> - Tao Zhang, Kaixian Qu, Zhibin Li, Jiajun Wu, Marco Hutter, Manling Li, Fan Shi - [[pdf]](https://arxiv.org/pdf/2604.18463)</summary>

**Abstract:** Large language models are increasingly used as planners for robotic systems, yet how safely they plan remains an open question. To evaluate safe planning systematically, we introduce DESPITE, a benchmark of 12,279 tasks spanning physical and normative dangers with fully deterministic validation. Across 23 models, even near-perfect planning ability does not ensure safety: the best-planning model fails to produce a valid plan on only 0.4% of tasks but produces dangerous plans on 28.3%. Among 18 open-source models from 3B to 671B parameters, planning ability improves substantially with scale (0.4-99.3%) while safety awareness remains relatively flat (38-57%). We identify a multiplicative relationship between these two capacities, showing that larger models complete more tasks safely primarily through improved planning, not through better danger avoidance. Three proprietary reasoning models reach notably higher safety awareness (71-81%), while non-reasoning proprietary models and open-source reasoning models remain below 57%. As planning ability approaches saturation for frontier models, improving safety awareness becomes a central challenge for deploying language-model planners in robotic systems.

**arXiv ID:** 2604.18463
</details>

<details>
<summary><strong>ClawEnvKit: Automatic Environment Generation for Claw-Like Agents</strong> - Xirui Li, Ming Li, Derry Xu, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh, Tianyi Zhou - [[pdf]](https://arxiv.org/pdf/2604.18543)</summary>

**Abstract:** Constructing environments for training and evaluating claw-like agents remains a manual, human-intensive process that does not scale. We argue that what is needed is not just a dataset, but an automated pipeline capable of generating diverse, verified environments on demand. To this end, we introduce ClawEnvKit, an autonomous generation pipeline that instantiates this formalism from natural language descriptions. The pipeline comprises three modules: (1) a parser that extracts structured generation parameters from natural language input; (2) a generator that produces the task specification, tool interface, and scoring configuration; and (3) a validator that enforces feasibility, diversity, structural validity, and internal consistency across the generated environments. Using ClawEnvKit, we construct Auto-ClawEval, the first large-scale benchmark for claw-like agents, comprising 1,040 environments across 24 categories. Empirically, Auto-ClawEval matches or exceeds human-curated environments on coherence and clarity at 13,800x lower cost. Evaluated across 4 model families and 8 agent harness frameworks, we find that harness engineering boosts performance by up to 15.7 percentage points over a bare ReAct baseline, completion remains the primary axis of variation with no model saturating the benchmark, and automated generation enables evaluation at a scale previously infeasible. Beyond static benchmarking, ClawEnvKit enables live evaluation: users describe a desired capability in natural language and obtain a verified environment on demand, turning evaluation into a continuous, user-driven process. The same mechanism serves as an on-demand training environment generator, producing task distributions that adapt to an agent's current weaknesses rather than being bounded by existing user logs.

**arXiv ID:** 2604.18543
</details>

<details>
<summary><strong>Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs</strong> - Kevin Murphy - [[pdf]](https://arxiv.org/pdf/2604.18576)</summary>

**Abstract:** We present BLF (Bayesian Linguistic Forecaster), an agentic system for binary forecasting that achieves state-of-the-art performance on the ForecastBench benchmark. The system is built on three ideas. (1) A Bayesian linguistic belief state: a semi-structured representation combining numerical probability estimates with natural-language evidence summaries, updated by the LLM at each step of an iterative tool-use loop. This contrasts with the common approach of appending all retrieved evidence to an ever-growing context. (2) Hierarchical multi-trial aggregation: running $K$ independent trials and combining them using logit-space shrinkage with a data-dependent prior. (3) Hierarchical calibration: Platt scaling with a hierarchical prior, which avoids over-shrinking extreme predictions for sources with skewed base rates.
On 400 backtesting questions from the ForecastBench leaderboard, BLF outperforms all the top public methods, including Cassi, GPT-5, Grok~4.20, and Foresight-32B. Ablation studies show that the structured belief state is as impactful as web search access, and that shrinkage aggregation and hierarchical calibration each provide significant additional gains.
In addition, we develop a robust back-testing framework with a leakage rate below 1.5\%, and use rigorous statistical methodology to compare different methods while controlling for various sources of noise.

**arXiv ID:** 2604.18576
</details>

<details>
<summary><strong>BrainMem: Brain-Inspired Evolving Memory for Embodied Agent Task Planning</strong> - Xiaoyu Ma, Lianyu Hu, Wenbing Tang, Zixuan Hu, Zeqin Liao, Zhizhen Wu, Yang Liu - [[pdf]](https://arxiv.org/pdf/2604.16331)</summary>

**Abstract:** Embodied task planning requires agents to execute long-horizon, goal-directed actions in complex 3D environments, where success depends on both immediate perception and accumulated experience across tasks. However, most existing LLM-based planners are stateless and reactive, operating without persistent memory and therefore repeating errors and struggling with spatial or temporal dependencies. We propose BrainMem(Brain-Inspired Evolving Memory), a training-free hierarchical memory system that equips embodied agents with working, episodic, and semantic memory inspired by human cognition. BrainMem continuously transforms interaction histories into structured knowledge graphs and distilled symbolic guidelines, enabling planners to retrieve, reason over, and adapt behaviors from past experience without any model fine-tuning or additional training. This plug-and-play design integrates seamlessly with arbitrary multi-modal LLMs and greatly reduces reliance on task-specific prompt engineering. Extensive experiments on four representative benchmarks, including EB-ALFRED, EB-Navigation, EB-Manipulation, and EB-Habitat, demonstrate that BrainMem significantly enhances task success rates across diverse models and difficulty subsets, with the largest gains observed on long-horizon and spatially complex tasks. These results highlight evolving memory as a promising and scalable mechanism for generalizable embodied intelligence.

**arXiv ID:** 2604.16331
</details>

<details>
<summary><strong>StressWeb: A Diagnostic Benchmark for Web Agent Robustness under Realistic Interaction Variability</strong> - Haoyue Bai, Dong Wang, Long Chen, Bingguang Hao, Pengyang Shao, Yonghui Yang, Yicheng He, Chenyi Zhuang - [[pdf]](https://arxiv.org/pdf/2604.16385)</summary>

**Abstract:** Large language model-based web agents have demonstrated strong performance on realistic web interaction tasks. However, existing evaluations are predominantly conducted under relatively stable and well-behaved interaction conditions, which may overestimate agent robustness. High task success in such idealized settings does not necessarily reflect performance under realistic web interaction. To address this limitation, we introduce a diagnostic stress-testing benchmark for web agents. We first construct realistic and controllable web environments that provide clean and stable interaction workflows as reference baselines. We then introduce structured and controlled perturbations that emulate interaction variability, including shifting layouts, altered interaction semantics, and execution disruptions. By comparing agent behavior between clean and perturbed settings, our framework enables systematic diagnosis of robustness under what-if interaction scenarios. Through extensive evaluation of state-of-the-art multimodal web agents, we show that stress-based evaluation exposes failure modes and substantial robustness gaps that remain hidden under clean benchmark conditions.

**arXiv ID:** 2604.16385
</details>

<details>
<summary><strong>ICAT: Incident-Case-Grounded Adaptive Testing for Physical-Risk Prediction in Embodied World Models</strong> - Zhenglin Lai, Sirui Huang, Yuteng Li, Changxin Huang, Jianqiang Li, Bingzhe Wu - [[pdf]](https://arxiv.org/pdf/2604.16405)</summary>

**Abstract:** Video-generative world models are increasingly used as neural simulators for embodied planning and policy learning, yet their ability to predict physical risk and severe consequences is rarely this http URL find that these models often downplay or omit key danger cues and severe outcomes for hazardous actions, which can induce unsafe preferences during planning and training on imagined rollouts. We propose ICAT, which grounds testing in real incident reports and safety manuals by building structured risk memories and retrieving/composing them to constrain the generation of risk cases with causal chains and severity labels. Experiments on an ICAT-based benchmark show that mainstream world models frequently miss mechanisms and triggering conditions and miscalibrate severity, falling short of the reliability required for safety-critical embodied deployment.

**arXiv ID:** 2604.16405
</details>

<details>
<summary><strong>Scaling Test-Time Compute for Agentic Coding</strong> - Joongwon Kim, Wannan Yang, Kelvin Niu, Hongming Zhang, Yun Zhu, Eryk Helenowski, Ruan Silva, Zhengxing Chen, Srinivasan Iyer, Manzil Zaheer, Daniel Fried, Hannaneh Hajishirzi, Sanjeev Arora, Gabriel Synnaeve, Ruslan Salakhutdinov, Anirudh Goyal - [[pdf]](https://arxiv.org/pdf/2604.16529)</summary>

**Abstract:** Test-time scaling has become a powerful way to improve large language models. However, existing methods are best suited to short, bounded outputs that can be directly compared, ranked or refined. Long-horizon coding agents violate this premise: each attempt produces an extended trajectory of actions, observations, errors, and partial progress taken by the agent. In this setting, the main challenge is no longer generating more attempts, but representing prior experience in a form that can be effectively selected from and reused. We propose a test-time scaling framework for agentic coding based on compact representations of rollout trajectories. Our framework converts each rollout into a structured summary that preserves its salient hypotheses, progress, and failure modes while discarding low-signal trace details. This representation enables two complementary forms of inference-time scaling. For parallel scaling, we introduce Recursive Tournament Voting (RTV), which recursively narrows a population of rollout summaries through small-group comparisons. For sequential scaling, we adapt Parallel-Distill-Refine (PDR) to the agentic setting by conditioning new rollouts on summaries distilled from prior attempts. Our method consistently improves the performance of frontier coding agents across SWE-Bench Verified and Terminal-Bench v2.0. For example, by using our method Claude-4.5-Opus improves from 70.9% to 77.6% on SWE-Bench Verified (mini-SWE-agent) and 46.9% to 59.1% on Terminal-Bench v2.0 (Terminus 1). Our results suggest that test-time scaling for long-horizon agents is fundamentally a problem of representation, selection, and reuse.

**arXiv ID:** 2604.16529
</details>

<details>
<summary><strong>Understanding Tool-Augmented Agents for Lean Formalization: A Factorial Analysis</strong> - Ke Zhang, Patricio Gallardo, Maziar Raissi, Sudhir Murthy - [[pdf]](https://arxiv.org/pdf/2604.16538)</summary>

**Abstract:** Automatic translation of natural language mathematics into faithful Lean 4 code is hindered by the fundamental dissonance between informal set-theoretic intuition and strict formal type theory. This gap often causes LLMs to hallucinate non-existent library definitions, resulting in code that fails to compile or lacks semantic fidelity. In this work, we investigate the effectiveness of tool-augmented agents for this task through a systematic factorial analysis of three distinct tool categories: Fine-tuned Model Querying (accessing expert drafts), Knowledge Search (retrieving symbol definitions), and Compiler Feedback (verifying code via a Lean REPL). We first benchmark the agent against one-shot baselines, demonstrating large gains in both compilation success and semantic equivalence. We then use the factorial decomposition to quantify the impact of each category, isolating the marginal contribution of each tool type to overall performance.

**arXiv ID:** 2604.16538
</details>

<details>
<summary><strong>ARMove: Learning to Predict Human Mobility through Agentic Reasoning</strong> - Chuyue Wang, Jie Feng, Yuxi Wu, Shenglin Yi, Hang Zhang - [[pdf]](https://arxiv.org/pdf/2604.17419)</summary>

**Abstract:** Human mobility prediction is a critical task but remains challenging due to its complexity and variability across populations and regions. Recently, large language models (LLMs) have made progress in zero-shot prediction, but existing methods suffer from limited interpretability (due to black-box reasoning), lack of iterative learning from new data, and poor transferability. In this paper, we introduce \textbf{ARMove}, a fully transferable framework for predicting human mobility through agentic reasoning. To address these limitations, ARMove employs standardized feature management with iterative optimization and user-specific customization: four major feature pools for foundational knowledge, user profiles for segmentation, and an automated generation mechanism integrating LLM knowledge. Robust generalization is achieved via agentic decision-making that adjusts feature weights to maximize accuracy while providing interpretable decision paths. Finally, large-small model synergy distills strategies from large LLMs (e.g., 72B) to smaller ones (e.g., 7B), reducing costs and enhancing performance ceilings. Extensive experiments on four global datasets show ARMove outperforms state-of-the-art baselines on 6 out of 12 metrics (gains of 0.78\% to 10.47\%), with transferability tests confirming robustness across regions, users, and scales. The other 4 items also achieved suboptimal results. Transferability tests confirm its 19 robustness across regions, user groups, and model scales, while interpretability 20 analysis highlights its transparency in decision-making. Our codes are available at: this https URL.

**arXiv ID:** 2604.17419
</details>

<details>
<summary><strong>SkillX: Automatically Constructing Skill Knowledge Bases for Agents</strong> - Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, Shumin Deng - [[pdf]](https://arxiv.org/pdf/2604.04804)</summary>

**Abstract:** Learning from experience is critical for building capable large language model (LLM) agents, yet prevailing self-evolving paradigms remain inefficient: agents learn in isolation, repeatedly rediscover similar behaviors from limited experience, resulting in redundant exploration and poor generalization. To address this problem, we propose SkillX, a fully automated framework for constructing a \textbf{plug-and-play skill knowledge base} that can be reused across agents and environments. SkillX operates through a fully automated pipeline built on three synergistic innovations: \textit{(i) Multi-Level Skills Design}, which distills raw trajectories into three-tiered hierarchy of strategic plans, functional skills, and atomic skills; \textit{(ii) Iterative Skills Refinement}, which automatically revises skills based on execution feedback to continuously improve library quality; and \textit{(iii) Exploratory Skills Expansion}, which proactively generates and validates novel skills to expand coverage beyond seed training data. Using a strong backbone agent (GLM-4.6), we automatically build a reusable skill library and evaluate its transferability on challenging long-horizon, user-interactive benchmarks, including AppWorld, BFCL-v3, and $\tau^2$-Bench. Experiments show that SkillKB consistently improves task success and execution efficiency when plugged into weaker base agents, highlighting the importance of structured, hierarchical experience representations for generalizable agent learning. Our code will be publicly available soon at this https URL.

**arXiv ID:** 2604.04804
</details>

<details>
<summary><strong>Seeing Isn't Believing: Mitigating Belief Inertia via Active Intervention in Embodied Agents</strong> - Hanlin Wang, Chak Tou Leong, Jian Wang, Wenjie Li - [[pdf]](https://arxiv.org/pdf/2604.17252)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have enabled agents to tackle complex embodied tasks through environmental interaction. However, these agents still make suboptimal decisions and perform ineffective actions, as they often overlook critical environmental feedback that differs from their internal beliefs. Through a formal probing analysis, we characterize this as belief inertia, a phenomenon where agents stubbornly adhere to prior beliefs despite explicit observations. To address this, we advocate active belief intervention, moving from passive understanding to active management. We introduce the Estimate-Verify-Update (EVU) mechanism, which empowers agents to predict expected outcomes, verify them against observations through explicit reasoning, and actively update prior beliefs based on the verification evidence. EVU is designed as a unified intervention mechanism that generates textual belief states explicitly, and can be integrated into both prompting-based and training-based agent reasoning methods. Extensive experiments across three embodied benchmarks demonstrate that EVU consistently yields substantial gains in task success rates. Further analyses validate that our approach effectively mitigates belief inertia, advancing the development of more robust embodied agents. Our code is available at this https URL.

**arXiv ID:** 2604.17252
</details>

<details>
<summary><strong>Agents Explore but Agents Ignore: LLMs Lack Environmental Curiosity</strong> - Leon Engländer, Sophia Althammer, Ahmet Üstün, Matthias Gallé, Tom Sherborne - [[pdf]](https://arxiv.org/pdf/2604.17609)</summary>

**Abstract:** LLM-based agents are assumed to integrate environmental observations into their reasoning: discovering highly relevant but unexpected information should naturally lead to a model exploiting its own discoveries. We show that this assumption is false for current LLM-based agents, which struggle to reflect or react to unexpected information. Across three benchmarks (Terminal-Bench, SWE-Bench, AppWorld), we inject complete task solutions into the agent environments to deliberately expose a task's solution to a model. While agents discover these solutions on Terminal-Bench in 79-81% of runs, they interact, or exploit, them in only 37-50% of cases. This gap is starkest in AppWorld: agents see documentation stating that a command "returns the complete solution to this task" in over 90% of attempts but exploit this in fewer than 7% of trials. We show that agents lack what we call environmental curiosity: the capability to recognize and investigate unexpected but relevant observations in response to environmental stimuli. We identify three main factors influencing environmental curiosity: available tools in the agent scaffold, test-time compute, and training data distribution. Our findings identify configurations that maximize curiosity also achieve the best performance on the unmodified benchmarks. Yet even jointly optimized agents still ignore discovered solutions in the majority of trials: current agents use the environment to fetch expected information, but not to revise their strategy or maximally exploit useful stimuli.

**arXiv ID:** 2604.17609
</details>

<details>
<summary><strong>Cooperative Coevolution versus Monolithic Evolutionary Search for Semi-Supervised Tabular Classification</strong> - Jamal Toutouh - [[pdf]](https://arxiv.org/pdf/2604.16412)</summary>

**Abstract:** This paper studies semi-supervised tabular classification in the extreme low-label regime using lightweight base learners. The paper proposes a cooperative coevolutionary method (CC-SSL) that evolves (i) two feature-subset views and (ii) a pseudo-labeling policy, and compares it to a matched monolithic evolutionary baseline (EA-SSL) and three lightweight SSL baselines. Experiments on 25 OpenML datasets with labeled fractions {1%,5%,10%} evaluate test MacroF1 and accuracy, together with evolutionary and pseudo-label diagnostics. CC-SSL and EA-SSL achieve higher median test MacroF1 than the lightweight baselines, with the largest separations at 1% labeled data. Most CC-SSL vs. EA-SSL comparisons are statistical draws on final test performance. EA-SSL shows higher best-so-far fitness and higher diversity during search, while time-to-target is comparable and generations-to-target favors EA-SSL in several multiclass settings. Pseudo-label volume, ProbeDrop, and validation optimism show no significant differences between CC-SSL and EA-SSL under the shared protocol.

**arXiv ID:** 2604.16412
</details>

<details>
<summary><strong>Chain Of Interaction Benchmark (COIN): When Reasoning meets Embodied Interaction</strong> - Xianhao Wang, Xiaojian Ma, Haozhe Hu, Rongpeng Su, Yutian Cheng, Zhou Ziheng, Hangxin Liu, Lei Liu, Bin Li, Qing Li - [[pdf]](https://arxiv.org/pdf/2604.16886)</summary>

**Abstract:** Generalist embodied agents must perform interactive, causally-dependent reasoning, continually interacting with the environment, acquiring information, and updating plans to solve long-horizon tasks before they could be adopted in real-life scenarios. For instance, retrieving an apple from a cabinet may require opening multiple doors and drawers before the apple becomes visible and reachable, demanding sequential interaction under partial observability. However, existing benchmarks fail to systematically evaluate this essential capability. We introduce COIN, a benchmark designed to assess interactive reasoning in realistic robotic manipulation through three key contributions. First, we construct COIN-50: 50 interactive tasks in daily scenarios, and create COIN-Primitive required by causally-dependent tasks, and COIN-Composition with mid-term complexity for skill learning and generalization evaluation. Second, we develop a low-cost mobile AR teleoperation system and collect the COIN-Primitive Dataset with 50 demonstrations per primitive task (1,000 in total). Third, we develop systematic evaluation metrics about execution stability and generalization robustness to evaluate CodeAsPolicy, VLA, and language-conditioned H-VLA approaches. Our comprehensive evaluation reveals critical limitations in current methods: models struggle with interactive reasoning tasks due to significant gaps between visual understanding and motor execution. We provide fine-grained analysis of these limitations.

**arXiv ID:** 2604.16886
</details>

<details>
<summary><strong>Leveraging VR Robot Games to Facilitate Data Collection for Embodied Intelligence Tasks</strong> - Yihan Zhang, Ziyun Huang, Linqi Ye - [[pdf]](https://arxiv.org/pdf/2604.16903)</summary>

**Abstract:** Collecting embodied interaction data at scale remains costly and difficult due to the limited accessibility of conventional interfaces. We present a gamified data collection framework based on Unity that combines procedural scene generation, VR-based humanoid robot control, automatic task evaluation, and trajectory logging. A trash pick-and-place task prototype is developed to validate the full this http URL results indicate that the collected demonstrations exhibit broad coverage of the state-action space, and that increasing task difficulty leads to higher motion intensity as well as more extensive exploration of the arm's workspace. The proposed framework demonstrates that game-oriented virtual environments can serve as an effective and extensible solution for embodied data collection.

**arXiv ID:** 2604.16903
</details>

<details>
<summary><strong>EmbodiedLGR: Integrating Lightweight Graph Representation and Retrieval for Semantic-Spatial Memory in Robotic Agents</strong> - Paolo Riva, Leonardo Gargani, Matteo Frosi, Matteo Matteucci - [[pdf]](https://arxiv.org/pdf/2604.18271)</summary>

**Abstract:** As the world of agentic artificial intelligence applied to robotics evolves, the need for agents capable of building and retrieving memories and observations efficiently is increasing. Robots operating in complex environments must build memory structures to enable useful human-robot interactions by leveraging the mnemonic representation of the current operating context. People interacting with robots may expect the embodied agent to provide information about locations, events, or objects, which requires the agent to provide precise answers within human-like inference times to be perceived as responsive. We propose the Embodied Light Graph Retrieval Agent (EmbodiedLGR-Agent), a visual-language model (VLM)-driven agent architecture that constructs dense and efficient representations of robot operating environments. EmbodiedLGR-Agent directly addresses the need for an efficient memory representation of the environment by providing a hybrid building-retrieval approach built on parameter-efficient VLMs that store low-level information about objects and their positions in a semantic graph, while retaining high-level descriptions of the observed scenes with a traditional retrieval-augmented architecture. EmbodiedLGR-Agent is evaluated on the popular NaVQA dataset, achieving state-of-the-art performance in inference and querying times for embodied agents, while retaining competitive accuracy on the global task relative to the current state-of-the-art approaches. Moreover, EmbodiedLGR-Agent was successfully deployed on a physical robot, showing practical utility in real-world contexts through human-robot interaction, while running the visual-language model and the building-retrieval pipeline locally.

**arXiv ID:** 2604.18271
</details>

<details>
<summary><strong>Fringe Projection Based Vision Pipeline for Autonomous Hard Drive Disassembly</strong> - Badrinath Balasubramaniam, Vignesh Suresh, Benjamin Metcalf, Beiwen Li - [[pdf]](https://arxiv.org/pdf/2604.17231)</summary>

**Abstract:** Unrecovered e-waste represents a significant economic loss. Hard disk drives (HDDs) comprise a valuable e-waste stream necessitating robotic disassembly. Automating the disassembly of HDDs requires holistic 3D sensing, scene understanding, and fastener localization, however current methods are fragmented, lack robust 3D sensing, and lack fastener localization. We propose an autonomous vision pipeline which performs 3D sensing using a Fringe Projection Profilometry (FPP) module, with selective triggering of a depth completion module where FPP fails, and integrates this module with a lightweight, real-time instance segmentation network for scene understanding and critical component localization. By utilizing the same FPP camera-projector system for both our depth sensing and component localization modules, our depth maps and derived 3D geometry are inherently pixel-wise aligned with the segmentation masks without registration, providing an advantage over RGB-D perception systems common in industrial sensing. We optimize both our trained depth completion and instance segmentation networks for deployment-oriented inference. The proposed system achieves a box mAP@50 of 0.960 and mask mAP@50 of 0.957 for instance segmentation, while the selected depth completion configuration with the Depth Anything V2 Base backbone achieves an RMSE of 2.317 mm and MAE of 1.836 mm; the Platter Facing learned inference stack achieved a combined latency of 12.86 ms and a throughput of 77.7 Frames Per Second (FPS) on the evaluation workstation. Finally, we adopt a sim-to-real transfer learning approach to augment our physical dataset. The proposed perception pipeline provides both high-fidelity semantic and spatial data which can be valuable for downstream robotic disassembly. The synthetic dataset developed for HDD instance segmentation will be made publicly available.

**arXiv ID:** 2604.17231
</details>

<details>
<summary><strong>VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning</strong> - Bo Jiang, Shaoyu Chen, Hao Gao, Bencheng Liao, Qian Zhang, Wenyu Liu, Xinggang Wang - [[pdf]](https://arxiv.org/pdf/2402.13243)</summary>

**Abstract:** Learning a human-like driving policy from large-scale driving demonstrations is promising, but the uncertainty and non-deterministic nature of planning make it challenging. Existing learning-based planning methods follow a deterministic paradigm to directly regress the action, failing to cope with the uncertainty problem. In this work, we propose a probabilistic planning model for end-to-end autonomous driving, termed VADv2. We resort to a probabilistic field function to model the mapping from the action space to the probabilistic distribution. Since the planning action space is a high-dimensional continuous spatiotemporal space and hard to tackle, we first discretize the planning action space to a large planning vocabulary and then tokenize the planning vocabulary into planning tokens. Planning tokens interact with scene tokens and output the probabilistic distribution of action. Mass driving demonstrations are leveraged to supervise the distribution. VADv2 achieves state-of-the-art closed-loop performance on the CARLA Town05 benchmark, significantly outperforming existing methods, and also leads the recent Bench2Drive benchmark. We further provide comprehensive evaluations on NAVSIM and a large-scale 3DGS-based benchmark, demonstrating its effectiveness in real-world applications. Code is available at this https URL.

**arXiv ID:** 2402.13243
</details>

<details>
<summary><strong>Driving in Corner Case: A Real-World Adversarial Closed-Loop Evaluation Platform for End-to-End Autonomous Driving</strong> - Jiaheng Geng, Jiatong Du, Xinyu Zhang, Ye Li, Panqu Wang, Yanjun Huang - [[pdf]](https://arxiv.org/pdf/2512.16055)</summary>

**Abstract:** Safety-critical corner cases, difficult to collect in the real world, are crucial for evaluating end-to-end autonomous driving. Adversarial interaction is an effective method to generate such safety-critical corner cases. While existing adversarial evaluation methods are built for models operating in simplified simulation environments, adversarial evaluation for real-world end-to-end autonomous driving has been little explored. To address this challenge, we propose a closed-loop evaluation platform for end-to-end autonomous driving, which can generate adversarial interactions in real-world scenes. In our platform, the real-world image generator cooperates with an adversarial traffic policy to evaluate various end-to-end models trained on real-world data. The generator, based on flow matching, efficiently and stably generates real-world images according to the traffic environment information. The efficient adversarial surrounding vehicle policy is designed to model challenging interactions and create corner cases that current autonomous driving systems struggle to handle. Experimental results demonstrate that the platform can generate realistic driving images efficiently. Through evaluating the end-to-end models such as UniAD and VAD, we demonstrate that based on the adversarial policy, our platform evaluates the performance degradation of the tested model in corner cases. This result indicates that this platform can effectively detect the model's potential issues, which will facilitate the safety and robustness of end-to-end autonomous driving.

**arXiv ID:** 2512.16055
</details>

<details>
<summary><strong>R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation</strong> - William Ljungbergh, Bernardo Taveira, Wenzhao Zheng, Adam Tonderski, Chensheng Peng, Fredrik Kahl, Christoffer Petersson, Michael Felsberg, Kurt Keutzer, Masayoshi Tomizuka, Wei Zhan - [[pdf]](https://arxiv.org/pdf/2506.07826)</summary>

**Abstract:** Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural rendering-based virtual environments, allowing R3D2 to learn realistic integration. Quantitative and qualitative evaluations demonstrate that R3D2 significantly enhances the realism of inserted assets, enabling use-cases like text-to-3D asset insertion and cross-scene/dataset object transfer, allowing for true scalability in AD validation. To promote further research in scalable and realistic AD simulation, we release our code, see this https URL.

**arXiv ID:** 2506.07826
</details>

<details>
<summary><strong>Agentic Education: Using Claude Code to Teach Claude Code</strong> - Zain Naboulsi - [[pdf]](https://arxiv.org/pdf/2604.17460)</summary>

**Abstract:** AI coding assistants have proliferated rapidly, yet structured pedagogical frameworks for learning these tools remain scarce. Developers face a gap between tool documentation and practical mastery, relying on fragmented resources such as blog posts, video tutorials, and trial-and-error. We present cc-self-train, a modular interactive curriculum for learning Claude Code, an agentic AI coding tool, through hands-on project construction. The system introduces five contributions: (1) a persona progression model that adapts instructor tone across four stages (Guide, Collaborator, Peer, Launcher), operationalizing Gradual Release of Responsibility for AI-mediated instruction; (2) an adaptive learning system that observes engagement quality through hook-based heuristics and adjusts scaffolding at two timescales, using streak detection for mid-module intervention and aggregate metrics for module-boundary persona changes; (3) a cross-domain unified curriculum in which five distinct project domains share identical feature sequencing, enabling transfer learning; (4) a step-pacing mechanism with explicit pause primitives to manage information overload in an AI-as-instructor context; and (5) an auto-updating curriculum design in which the onboarding agent detects upstream tool changes and updates teaching materials before instruction begins. A parametrized test suite enforces structural consistency as a proxy for pedagogical invariants across all 50 modules. A pilot evaluation with 27 participants shows statistically significant reported self-efficacy gains across all 10 assessed skill areas (p < 0.001), with the largest effects on advanced features such as hooks and custom skills. We discuss implications for the design of auto-updating educational systems.

**arXiv ID:** 2604.17460
</details>

</details>

<details open>
<summary><h2>LLM Agents (13 papers)</h2></summary>

<details>
<summary><strong>Don't Start What You Can't Finish: A Counterfactual Audit of Support-State Triage in LLM Agents</strong> - Eren Unlu - [[pdf]](https://arxiv.org/pdf/2604.16752)</summary>

**Abstract:** Current agent evaluations largely reward execution on fully specified tasks, while recent work studies clarification [11, 22, 2], capability awareness [9, 1], abstention [8, 14], and search termination [20, 5] mostly in isolation. This leaves open whether agents can diagnose why a task is blocked before acting. We introduce the Support-State Triage Audit (SSTA-32), a matched-item diagnostic framework in which minimal counterfactual edits flip the same base request across four support states: Complete (ANSWER), Clarifiable (CLARIFY), Support-Blocked (REQUEST SUPPORT), and Unsupported-Now (ABSTAIN). We evaluate a frontier model under four prompting conditions - Direct, Action-Only, Confidence-Only, and a typed Preflight Support Check (PSC) - using Dual-Persona Auto-Auditing (DPAA) with deterministic heuristic scoring. Default execution overcommits heavily on non-complete tasks (41.7% overcommitment rate). Scalar confidence mapping avoids overcommitment but collapses the three-way deferral space (58.3% typed deferral accuracy). Conversely, both Action-Only and PSC achieve 91.7% typed deferral accuracy by surfacing the categorical ontology in the prompt. Targeted ablations confirm that removing the support-sufficiency dimension selectively degrades REQUEST SUPPORT accuracy, while removing the evidence-sufficiency dimension triggers systematic overcommitment on unsupported items. Because DPAA operates within a single context window, these results represent upper-bound capability estimates; nonetheless, the structural findings indicate that frontier models possess strong latent triage capabilities that require explicit categorical decision paths to activate safely.

**arXiv ID:** 2604.16752
</details>

<details>
<summary><strong>Knows: Agent-Native Structured Research Representations</strong> - Guangsheng Yu, Xu Wang - [[pdf]](https://arxiv.org/pdf/2604.17309)</summary>

**Abstract:** Research artifacts are distributed primarily as reader-oriented documents like PDFs. This creates a bottleneck for increasingly agent-assisted and agent-native research workflows, in which LLM agents need to infer fine-grained, task-relevant information from lengthy full documents, a process that is expensive, repetitive, and unstable at scale.
We introduce Knows, a lightweight companion specification that binds structured claims, evidence, provenance, and verifiable relations to existing research artifacts in a form LLM agents can consume directly. Knows addresses the gap with a thin YAML sidecar (KnowsRecord) that coexists with the original PDF, requiring no changes to the publication itself, and validated by a deterministic schema linter. We evaluate Knows on 140 comprehension questions across 20 papers spanning 14 academic disciplines, comparing PDF-only, sidecar-only, and hybrid conditions across six LLM agents of varying capacity. Weak models (0.8B--2B parameters) improve from 19--25\% to 47--67\% accuracy (+29 to +42 percentage points) when reading sidecar instead of PDF, while consuming 29--86\% fewer input tokens; an LLM-as-judge re-scoring confirms that weak-model sidecar accuracy (75--77\%) approaches stronger-model PDF accuracy (78--83\%). Beyond this controlled evaluation, a community sidecar hub at this https URL has already indexed over ten thousand publications and continues to grow daily, providing independent evidence that the format is adoption-ready at scale.

**arXiv ID:** 2604.17309
</details>

<details>
<summary><strong>Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration</strong> - Qifan Zhang, Dongyang Ma, Tianqing Fang, Jia Li, Jing Tang, Nuo Chen, Haitao Mi, Yan Wang - [[pdf]](https://arxiv.org/pdf/2604.18131)</summary>

**Abstract:** Most agents today ``self-evolve'' by following rewards and rules defined by humans. However, this process remains fundamentally dependent on external supervision; without human guidance, the evolution stops. In this work, we train agents to possess an intrinsic meta-evolution capability to spontaneously learn about unseen environments prior to task execution.
To instill this ability, we design an outcome-based reward mechanism that measures how much an agent's self-generated world knowledge improves its success rate on downstream tasks. This reward signal is used exclusively during the training phase to teach the model how to explore and summarize effectively. At inference time, the agent requires no external rewards or human instructions. It spontaneously performs native self-evolution to adapt to unknown environments using its internal parameters.
When applied to Qwen3-30B and Seed-OSS-36B, this shift to native evolution yields a 20% performance increase on WebVoyager and WebWalker. Most strikingly, the generated world knowledge even enables a compact 14B Qwen3 model to outperform the unassisted Gemini-2.5-Flash, establishing a new paradigm for truly evolving agents.

**arXiv ID:** 2604.18131
</details>

<details>
<summary><strong>B-PASTE: Beam-Aware Pattern-Guided Speculative Execution for Resource-Constrained LLM Agents</strong> - Yanfei Song - [[pdf]](https://arxiv.org/pdf/2604.16469)</summary>

**Abstract:** LLM agents execute in an interleaved reasoning-and-action loop, where future tool calls cannot be launched until the current reasoning step completes. This serial dependency inflates end-to-end latency and leaves the model idle while waiting for tool execution. Prior work, Pattern-Aware Speculative Tool Execution (PASTE), mitigates this bottleneck by speculating likely future tool invocations from mined control-flow and data-flow regularities. However, PASTE is tool-centric and speculates only individual invocations rather than bounded future branches.
We propose B-PASTE, a beam-aware extension that lifts speculation from single tools to local branch hypotheses under strict resource constraints. B-PASTE maintains a bounded beam of future execution subgraphs, ranks them by expected critical-path reduction rather than raw execution probability, and schedules only high-value branch prefixes on transient slack resources. It explicitly models co-run interference, downstream unlock value, and state-safety constraints, enabling the system to prioritize serial fast-path execution when early completion unlocks valuable future work, while still exploiting safe parallelism under low contention.
This design is especially important for edge-side deployments, where speculative work must not steal scarce resources from latency-critical authoritative execution. Preliminary internal testing on Thor-class edge environments shows up to 1.4X end-to-end speedup, suggesting that branch-aware speculative execution remains effective even under tight resource budgets.

**arXiv ID:** 2604.16469
</details>

<details>
<summary><strong>A Survey on the Security of Long-Term Memory in LLM Agents: Toward Mnemonic Sovereignty</strong> - Zehao Lin, Chunyu Li, Kai Chen - [[pdf]](https://arxiv.org/pdf/2604.16548)</summary>

**Abstract:** Research on large language model (LLM) security is shifting from "will the model leak training data" to a more consequential question: can an agent with persistent, long-term memory be continuously shaped, cross-session poisoned, accessed without authorization, and propagated across shared organizational state? Recent surveys cover memory architectures and agent mechanisms, but fewer center the epistemic and governance properties of persistent, writable memory as the reason memory is an independent security problem.
This survey addresses that gap. Drawing on cognitive neuroscience and the philosophy of memory, we characterize agent memory as malleable, rewritable, and socially propagating, and develop a memory-lifecycle framework organized around six phases -- Write, Store, Retrieve, Execute, Share, Forget/Rollback -- cross-tabulated against four security objectives: integrity, confidentiality, availability, governance. We organize the literature on memory poisoning, extraction, retrieval corruption, control-flow hijacking, cross-agent propagation, rollback, and governance, and situate representative architectures as determinants of which phases are explicitly governable.
Three findings stand out: the literature concentrates on write- and retrieve-time integrity attacks, while confidentiality, availability, store/forget, and benign-persistence failures remain sparsely studied; no published architecture covers all nine governance primitives we identify; and using LLMs themselves for memory security remains sparse yet essential.
We unify these under mnemonic sovereignty -- verifiable, recoverable governance over what may be written, who may read, when updates are authorized, and which states may be forgotten -- arguing future secure agents will be differentiated not only by recall capacity, but by memory governance quality.

**arXiv ID:** 2604.16548
</details>

<details>
<summary><strong>Persona Alchemy: Designing, Evaluating, and Implementing Psychologically-Grounded LLM Agents for Diverse Stakeholder Representation</strong> - Sola Kim, Dongjune Chang, Jieshu Wang - [[pdf]](https://arxiv.org/pdf/2505.18351)</summary>

**Abstract:** Despite advances in designing personas for Large Language Models (LLM), challenges remain in aligning them with human cognitive processes and representing diverse stakeholder perspectives. We introduce a Social Cognitive Theory (SCT) agent design framework for designing, evaluating, and implementing psychologically grounded LLMs with consistent behavior. Our framework operationalizes SCT through four personal factors (cognitive, motivational, biological, and affective) for designing, six quantifiable constructs for evaluating, and a graph database-backed architecture for implementing stakeholder personas. Experiments tested agents' responses to contradicting information of varying reliability. In the highly polarized renewable energy transition discourse, we design five diverse agents with distinct ideologies, roles, and stakes to examine stakeholder representation. The evaluation of these agents in contradictory scenarios occurs through comprehensive processes that implement the SCT. Results show consistent response patterns ($R^2$ range: $0.58-0.61$) and systematic temporal development of SCT construct effects. Principal component analysis identifies two dimensions explaining $73$% of variance, validating the theoretical structure. Our framework offers improved explainability and reproducibility compared to black-box approaches. This work contributes to ongoing efforts to improve diverse stakeholder representation while maintaining psychological consistency in LLM personas.

**arXiv ID:** 2505.18351
</details>

<details>
<summary><strong>Why Agents Compromise Safety Under Pressure</strong> - Hengle Jiang, Ke Tang - [[pdf]](https://arxiv.org/pdf/2603.14975)</summary>

**Abstract:** Large Language Model agents deployed in complex environments frequently encounter a conflict between maximizing goal achievement and adhering to safety constraints. This paper identifies a new concept called Agentic Pressure, which characterizes the endogenous tension emerging when compliant execution becomes infeasible. We demonstrate that under this pressure agents exhibit normative drift where they strategically sacrifice safety to preserve utility. Notably we find that advanced reasoning capabilities accelerate this decline as models construct linguistic rationalizations to justify violation. Finally, we analyze the root causes and explore preliminary mitigation strategies, such as pressure isolation, which attempts to restore alignment by decoupling decision-making from pressure signals.

**arXiv ID:** 2603.14975
</details>

<details>
<summary><strong>HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents</strong> - Jinchang Zhu, Jindong Li, Cheng Zhang, Jiahong Liu, Menglin Yang - [[pdf]](https://arxiv.org/pdf/2604.16839)</summary>

**Abstract:** Long-term memory is a critical challenge for Large Language Model agents, as fixed context windows cannot preserve coherence across extended interactions. Existing memory systems represent conversation history as unstructured embedding vectors, retrieving information through semantic similarity. This paradigm fails to capture the associative structure of human memory, wherein related experiences progressively strengthen interconnections through repeated co-activation. Inspired by cognitive neuroscience, we identify three mechanisms central to biological memory: association, consolidation, and spreading activation, which remain largely absent in current research.
To bridge this gap, we propose HeLa-Mem, a bio-inspired memory architecture that models memory as a dynamic graph with Hebbian learning dynamics. HeLa-Mem employs a dual-level organization: (1) an episodic memory graph that evolves through co-activation patterns, and (2) a semantic memory store populated via Hebbian Distillation, wherein a Reflective Agent identifies densely connected memory hubs and distills them into structured, reusable semantic knowledge. This dual-path design leverages both semantic similarity and learned associations, mirroring the episodic-semantic distinction in human cognition. Experiments on LoCoMo demonstrate superior performance across four question categories while using significantly fewer context tokens. Code is available on GitHub: this https URL

**arXiv ID:** 2604.16839
</details>

<details>
<summary><strong>On Safety Risks in Experience-Driven Self-Evolving Agents</strong> - Weixiang Zhao, Yichen Zhang, Yingshuo Wang, Yang Deng, Yanyan Zhao, Xuda Zhi, Yongbo Huang, HaoHe, Wanxiang Che, Bing Qin, Ting Liu - [[pdf]](https://arxiv.org/pdf/2604.16968)</summary>

**Abstract:** Experience-driven self-evolution has emerged as a promising paradigm for improving the autonomy of large language model agents, yet its reliance on self-curated experience introduces underexplored safety risks. In this study, we investigate how experience accumulation and utilization in self-evolving agents affect safety performance across web-based and embodied environments. Notably, experience gathered solely from benign tasks can still compromise safety in high-risk scenarios. Further analysis attributes this degradation to the execution-oriented nature of accumulated experience, which reinforces agents' tendency to act rather than refuse. In more realistic settings where agents encounter both benign and harmful tasks, refusal-related experience mitigates safety decline but induces over-refusal, revealing a fundamental safety-utility trade-off. Overall, our findings expose inherent limitations of current self-evolving agents and call for more principled strategies to ensure safe and reliable adaptation.

**arXiv ID:** 2604.16968
</details>

<details>
<summary><strong>GenericAgent: A Token-Efficient Self-Evolving LLM Agent via Contextual Information Density Maximization (V1.0)</strong> - Jiaqing Liang, Jinyi Han, Weijia Li, Xinyi Wang, Zhoujia Zhang, Zishang Jiang, Ying Liao, Tingyun Li, Ying Huang, Hao Shen, Hanyu Wu, Fang Guo, Keyi Wang, Zhonghua Hong, Zhiyu Lu, Lipeng Ma, Sihang Jiang, Yanghua Xiao - [[pdf]](https://arxiv.org/pdf/2604.17091)</summary>

**Abstract:** Long-horizon large language model (LLM) agents are fundamentally limited by context. As interactions become longer, tool descriptions, retrieved memories, and raw environmental feedback accumulate and push out the information needed for decision-making. At the same time, useful experience gained from tasks is often lost across episodes. We argue that long-horizon performance is determined not by context length, but by how much decision-relevant information is maintained within a finite context budget. We present GenericAgent (GA), a general-purpose, self-evolving LLM agent system built around a single principle: context information density maximization. GA implements this through four closely connected components: a minimal atomic tool set that keeps the interface simple, a hierarchical on-demand memory that only shows a small high-level view by default, a self-evolution mechanism that turns verified past trajectories into reusable SOPs and executable code, and a context truncation and compression layer that maintains information density during long executions. Across task completion, tool use efficiency, memory effectiveness, self-evolution, and web browsing, GA consistently outperforms leading agent systems while using significantly fewer tokens and interactions, and it continues to evolve over time. Project: this https URL

**arXiv ID:** 2604.17091
</details>

<details>
<summary><strong>GraSP: Graph-Structured Skill Compositions for LLM Agents</strong> - Tianle Xia, Lingxiang Hu, Yiding Sun, Ming Xu, Lan Xu, Siying Wang, Wei Xu, Jie Jiang - [[pdf]](https://arxiv.org/pdf/2604.17870)</summary>

**Abstract:** Skill ecosystems for LLM agents have matured rapidly, yet recent benchmarks show that providing agents with more skills does not monotonically improve performance -- focused sets of 2-3 skills outperform comprehensive documentation, and excessive skills actually hurt. The bottleneck has shifted from skill availability to skill orchestration: agents need not more skills, but a structural mechanism to select, compose, and execute them with explicit causal dependencies. We propose GraSP, the first executable skill graph architecture that introduces a compilation layer between skill retrieval and execution. GraSP transforms flat skill sets into typed directed acyclic graphs (DAGs) with precondition-effect edges, executes them with node-level verification, and performs locality-bounded repair through five typed operators -- reducing replanning from O(N) to O(d^h). Across ALFWorld, ScienceWorld, WebShop, and InterCode with eight LLM backbones, GraSP outperforms ReAct, Reflexion, ExpeL, and flat skill baselines in every configuration, improving reward by up to +19 points over the strongest baseline while cutting environment steps by up to 41%. GraSP's advantage grows with task complexity and is robust to both skill over-retrieval and quality degradation, confirming that structured orchestration -- not larger skill libraries -- is the key to reliable agent execution.

**arXiv ID:** 2604.17870
</details>

<details>
<summary><strong>StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning</strong> - Daoyu Wang, Qingchuan Li, Mingyue Cheng, Jie Ouyang, Shuo Yu, Qi Liu, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2604.18401)</summary>

**Abstract:** General agents have given rise to phenomenal applications such as OpenClaw and Claude Code. As these agent systems (a.k.a. Harnesses) strive for bolder goals, they demand increasingly stronger agentic capabilities from foundation Large Language Models (LLMs). Agentic Reinforcement Learning (RL) is emerging as a central post-training paradigm for empowering LLMs with these capabilities and is playing an increasingly pivotal role in agent training. Unlike single-turn token-level alignment or reasoning enhancement, as in RLHF and RLVR, Agentic RL targets multi-turn interactive settings, where the goal is to optimize core agentic capabilities such as decision making and tool use while addressing new challenges including delayed and sparse rewards, as well as long and variable context. As a result, the token-centric modeling and optimization paradigm inherited from traditional LLM RL is becoming increasingly inadequate for capturing real LLM agent behavior. In this paper, we present StepPO as a position on step-level Agentic RL. We argue that the conventional token-level Markov Decision Process (MDP) should be advanced to a step-level MDP formulation, and that the step, rather than the token, should be regarded as the proper action representation for LLM agents. We then propose step-level credit assignment as the natural optimization counterpart of this formulation, thereby aligning policy optimization and reward propagation with the granularity of agent decisions. Finally, we discuss the key systems designs required to realize step-level Agentic RL in practice and preliminary experiments provide initial evidence for the effectiveness of this perspective. We hope that the step-aligned, step-level paradigm embodied in StepPO offers the Agentic RL community a useful lens for understanding agent behavior and helps advance LLMs toward stronger general-agent capabilities.

**arXiv ID:** 2604.18401
</details>

<details>
<summary><strong>Beyond the Townhall: Spatial Anchoring and LLM Agents for Scalable Participatory Urban Planning</strong> - Carina I Hausladen, Javier Argota Sánchez-Vaquerizo, Michael Siebenmann, Arthur Capozzi, Sachit Mahajan, Dirk Helbing - [[pdf]](https://arxiv.org/pdf/2604.16348)</summary>

**Abstract:** Participatory urban planning is central to sustainable city-making, yet the technically demanding nature of such interventions often limits meaningful involvement by diverse publics. We introduce a scalable digital participation platform that embeds sustainability projects within a navigable digital twin. Citizens experience a guided virtual walkthrough with audio narration employing the method of loci and spatial anchoring to support mnemonic encoding and recall. This immersive interface is augmented by two purpose-built LLM assistants: one delivers source-grounded factual clarifications, while the other facilitates reflective discussion. We evaluated this system in a randomized controlled online experiment (N = 195) against conventional industry practices (static visualizations and text-based consultations). Results show that spatially anchored immersive presentation significantly improved information recall, which substantially shifted participants' attention from individual inconveniences to collective, community-oriented sustainability benefits. Consequently, participants provided significantly more constructive, solution-focused feedback to the (simulated) municipality. These findings establish a practical tool for cities and policymakers to foster inclusive, democratic participation in sustainability transitions.

**arXiv ID:** 2604.16348
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (40 papers)</h2></summary>

<details>
<summary><strong>Semantic Consensus: Process-Aware Conflict Detection and Resolution for Enterprise Multi-Agent LLM Systems</strong> - Vivek Acharya - [[pdf]](https://arxiv.org/pdf/2604.16339)</summary>

**Abstract:** Multi-agent large language model (LLM) systems are rapidly emerging as the dominant architecture for enterprise AI automation, yet production deployments exhibit failure rates between 41% and 86.7%, with nearly 79% of failures originating from specification and coordination issues rather than model capability limitations. This paper identifies Semantic Intent Divergence--the phenomenon whereby cooperating LLM agents develop inconsistent interpretations of shared objectives due to siloed context and absent process models--as a primary yet formally unaddressed root cause of multi-agent failure in enterprise settings. We propose the Semantic Consensus Framework (SCF), a process-aware middleware comprising six components: a Process Context Layer for shared operational semantics, a Semantic Intent Graph for formal intent representation, a Conflict Detection Engine for real-time identification of contradictory, contention-based, and causally invalid intent combinations, a Consensus Resolution Protocol using a policy--authority--temporal hierarchy, a Drift Monitor for detecting gradual semantic divergence, and a Process-Aware Governance Integration layer for organizational policy enforcement. Evaluation across 600 runs spanning three multi-agent frameworks (AutoGen, CrewAI, LangGraph) and four enterprise scenarios demonstrates that SCF is the only approach to achieve 100% workflow completion--compared to 25.1% for the next-best baseline--while detecting 65.2% of semantic conflicts with 27.9% precision and providing complete governance audit trails. The framework is protocol-agnostic and compatible with MCP and A2A communication standards.

**arXiv ID:** 2604.16339
</details>

<details>
<summary><strong>Agentic Risk-Aware Set-Based Engineering Design</strong> - Varun Kumar, George Em Karniadakis - [[pdf]](https://arxiv.org/pdf/2604.16687)</summary>

**Abstract:** This paper introduces a multi-agent framework guided by Large Language Models (LLMs) to assist in the early stages of engineering design, a phase often characterized by vast parameter spaces and inherent uncertainty. Operating under a human-in-the-loop paradigm and demonstrated on the canonical problem of aerodynamic airfoil design, the framework employs a team of specialized agents: a Coding Assistant, a Design Agent, a Systems Engineering Agent, and an Analyst Agent - all coordinated by a human Manager. Integrated within a set-based design philosophy, the process begins with a collaborative phase where the Manager and Coding Assistant develop a suite of validated tools, after which the agents execute a structured workflow to systematically explore and prune a large set of initial design candidates. A key contribution of this work is the explicit integration of formal risk management, employing the Conditional Value-at-Risk (CVaR) as a quantitative metric to filter designs that exhibit a high probability of failing to meet performance requirements, specifically the target coefficient of lift. The framework automates labor-intensive initial exploration through a global sensitivity analysis conducted by the Analyst agent, which generates actionable heuristics to guide the other agents. The process culminates by presenting the human Manager with a curated final set of promising design candidates, augmented with high-fidelity Computational Fluid Dynamics (CFD) simulations. This approach effectively leverages AI to handle high-volume analytical tasks, thereby enhancing the decision-making capability of the human expert in selecting the final, risk-assessed design.

**arXiv ID:** 2604.16687
</details>

<details>
<summary><strong>Debate as Reward: A Multi-Agent Reward System for Scientific Ideation via RL Post-Training</strong> - Moein Salimi, Babak Hosseini Mohtasham, Amin Aghakasiri, Mahdi Naieni, Amir Hossein Qeysarbeigi, Mohammad Masih Shalchian Nazer, Zahra Azar, Mahdi Jafari Siavoshani, Mohammad Hossein Rohban - [[pdf]](https://arxiv.org/pdf/2604.16723)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated potential in automating scientific ideation, yet current approaches relying on iterative prompting or complex multi-agent architectures often suffer from hallucination or computational inefficiency. A critical bottleneck in applying Reinforcement Learning (RL) to this open-ended domain is reward hacking -- where models exploit imperfect evaluation proxies to maximize scores without producing genuine scientific innovation. To address these limitations, we propose an RL framework explicitly tailored for high-quality scientific idea generation. We propose the first multi-agent reward function designed to serve as a judge, decoupling methodological validation from implementation details while providing strict binary rewards that are robust to reward hacking. To effectively optimize against this sparse signal, we utilize an unbiased variant of Group Relative Policy Optimization to mitigate artificial length bias. We grounded our training in ICLR-320, a curated dataset of problem-solution pairs extracted from ICLR 2024 proceedings. Experiments demonstrate that our framework significantly outperforms state-of-the-art baselines across expert-evaluated metrics of novelty, feasibility, and effectiveness.

**arXiv ID:** 2604.16723
</details>

<details>
<summary><strong>Small Model as Master Orchestrator: Learning Unified Agent-Tool Orchestration with Parallel Subtask Decomposition</strong> - Wenzhen Yuan, Wutao Xiong, Fanchen Yu, Shengji Tang, Ting Liu, Tao Chen, Peng Ye, Yuzhuo Fu, Wanli Ouyang, Lei Bai - [[pdf]](https://arxiv.org/pdf/2604.17009)</summary>

**Abstract:** Multi-agent systems (MAS) demonstrate clear advantages in tackling complex problems by coordinating diverse agents and external tools. However, most existing orchestration methods rely on static workflows or serial agent scheduling, and are further constrained by heterogeneous interface protocols between tools and agents. This leads to high system complexity and poor extensibility. To mitigate these issues, we propose Agent-as-Tool, a unified parallel orchestration paradigm that abstracts both agents and tools into a standardized, learnable action space with protocol normalization and explicit state feedback. Building on this paradigm, we train a lightweight orchestrator, ParaManager, which decouples planning decisions from subtask solving, enabling state-aware parallel subtask decomposition, delegation, and asynchronous execution. For training, we adopt a two-stage ParaManager training pipeline. It improves robustness by incorporating supervised fine-tuning (SFT) trajectories equipped with recovery mechanisms, and further applies reinforcement learning (RL) to achieve an optimal balance among task success, protocol compliance, diversity, and reasoning efficiency. Experiments show that ParaManager achieves strong performance across multiple benchmarks and exhibits robust generalization under unseen model pools.

**arXiv ID:** 2604.17009
</details>

<details>
<summary><strong>Harness as an Asset: Enforcing Determinism via the Convergent AI Agent Framework (CAAF)</strong> - Tianbao Zhang - [[pdf]](https://arxiv.org/pdf/2604.17025)</summary>

**Abstract:** Large Language Models (LLMs) produce a controllability gap in safety-critical engineering: even low rates of undetected constraint violations render a system undeployable. Current orchestration paradigms suffer from sycophantic compliance, context attention decay [Liu et al., 2024], and stochastic oscillation during self-correction [Huang et al., 2024].
We introduce the Convergent AI Agent Framework (CAAF), which transitions agentic workflows from open-loop generation to closed-loop Fail-Safe Determinism via three pillars: (1) Recursive Atomic Decomposition with physical context firewalls; (2) Harness as an Asset, formalizing domain invariants into machine-readable registries enforced by a deterministic Unified Assertion Interface (UAI); and (3) Structured Semantic Gradients with State Locking for monotonic convergence.
Empirical evaluation across two domains -- SAE Level 3 (L3) autonomous driving (AD) (n=30, 7 conditions) and pharmaceutical continuous flow reactor design (n=20, 4 conditions including a Mono+UAI ablation) -- shows that CAAF-all-GPT-4o-mini achieves 100% paradox detection while monolithic GPT-4o achieves 0% (even at temperature=0). The pharmaceutical benchmark features 7 simultaneous constraints with nonlinear Arrhenius interactions and a 3-way minimal unsatisfiable subset, representing a structurally harder challenge than the 2-constraint AD paradox. Alternative multi-agent architectures (debate, sequential checking) also achieve 0% across 80 trials, confirming that CAAF's reliability derives from its deterministic UAI, not from multi-agent orchestration per se. A Mono+UAI ablation (95%) isolates UAI as the core contribution. CAAF's reliability is invariant to prompt hints; all components use a single commodity model, enabling fully offline deployment.

**arXiv ID:** 2604.17025
</details>

<details>
<summary><strong>Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration</strong> - Sukwon Yun, Jie Peng, Pingzhi Li, Wendong Fan, Jie Chen, James Zou, Guohao Li, Tianlong Chen - [[pdf]](https://arxiv.org/pdf/2604.17148)</summary>

**Abstract:** With an ever-growing zoo of LLMs and benchmarks, the need to orchestrate multiple models for improved task performance has never been more pressing. While frameworks like Mixture-of-Agents (MoA) attempt to coordinate LLMs, they often fall short in terms of (1) selecting relevant agents, (2) facilitating effective intra-agent communication, and (3) integrating responses efficiently. In this work, we propose Graph-of-Agents (GoA), a new graph-based framework for modeling multi-agent LLM communication. Our approach begins with node sampling, selecting only the most relevant agents by leveraging model cards that summarize each model's domain, task specialization, and other characteristics. Next, we construct edges between the selected agents by evaluating their responses against one another to determine relevance ordering. Directed message passing is then performed from highly relevant agents to less relevant ones to enhance their responses, followed by reverse message passing to refine the original responses of the more relevant agents. Finally, the updated responses are aggregated via graph-based pooling (e.g., max or mean pooling) to produce a single, unified answer. We evaluate GoA on diverse multi-domain benchmarks (MMLU, MMLU-Pro, GPQA) and domain-specific benchmarks (MATH, HumanEval, MedMCQA), with an agent pool of 6 LLMs spanning multiple domains. Surprisingly, GoA achieves superior performance using only 3 selected agents, outperforming recent multi-agent LLM baselines that utilize all 6 agents simultaneously. By adopting a graph structure, GoA offers both scalability and effectiveness through structured message passing-positioning it as a strong candidate for navigating the challenges of the ever-growing LLM zoo. Code is available at: this https URL.

**arXiv ID:** 2604.17148
</details>

<details>
<summary><strong>Safe and Policy-Compliant Multi-Agent Orchestration for Enterprise AI</strong> - Vinil Pasupuleti, Shyalendar Reddy Allala, Siva Rama Krishna Varma Bayyavarapu, Shrey Tyagi - [[pdf]](https://arxiv.org/pdf/2604.17240)</summary>

**Abstract:** Enterprise AI systems increasingly deploy multiple intelligent agents across mission-critical workflows that must satisfy hard policy constraints, bounded risk exposure, and comprehensive auditability (SOX, HIPAA, GDPR). Existing coordination methods - cooperative MARL, consensus protocols, and centralized planners - optimize expected reward while treating constraints implicitly. This paper introduces CAMCO (Constraint-Aware Multi-Agent Cognitive Orchestration), a runtime coordination layer that models multi-agent decision-making as a constrained optimization problem. CAMCO integrates three mechanisms: (i) a constraint projection engine enforcing policy-feasible actions via convex projection, (ii) adaptive risk-weighted Lagrangian utility shaping, and (iii) an iterative negotiation protocol with provably bounded convergence. Unlike training-time constrained RL, CAMCO operates as deployment-time middleware compatible with any agent architecture, with policy predicates designed for direct integration with production engines such as OPA. Evaluation across three enterprise scenarios - including comparison against a constrained Lagrangian MARL baseline - demonstrates zero policy violations, risk exposure below threshold (mean ratio 0.71), 92-97% utility retention, and mean convergence in 2.4 iterations.

**arXiv ID:** 2604.17240
</details>

<details>
<summary><strong>Hive: A Multi-Agent Infrastructure for Algorithm- and Task-Level Scaling</strong> - Zizhang Luo, Yuhao Luo, Youwei Xiao, Yansong Xu, Runlin Guo, Yun Liang - [[pdf]](https://arxiv.org/pdf/2604.17353)</summary>

**Abstract:** Large language models are increasingly deployed as complex agentic systems that scale with task complexity. While prior work has extensively explored model- and system-level scaling, algorithm- and task-level scaling remain largely unaddressed, constraining the full potential of agentic systems. At the algorithm level, allocating additional inference-time computation can enhance workflow capacity but introduces cross-path redundancy: overlapping computations across multiple reasoning branches. At the task level, complex tasks can be decomposed into subproblems and delegated across multiple agents for improved scalability and parallelism. However, existing infrastructures' scheduling is unaware of the existence of multiple agents, missing opportunities to optimize resource allocation.
We propose Hive, a multi-agent infrastructure that enables algorithm- and task-level scaling. Hive features a description frontend that captures per-agent behavior and supports test-time scaling algorithms. Leveraging this specification, our backend introduces two key mechanisms: Logits Cache that reuses intermediate logits across redundant sampling paths to mitigate cross-path redundancy at the algorithm level, and Agent-Aware Scheduling that efficiently allocates compute and KV-cache resources according to agent contributions at the task level. Experiments show that Logits Cache achieves an average speedup of $1.11\times$-$1.76\times$ for re-sampling, and Agent-Aware Scheduling reduces the hotspot miss rate by $33\%$-$51\%$.

**arXiv ID:** 2604.17353
</details>

<details>
<summary><strong>Phase-Scheduled Multi-Agent Systems for Token-Efficient Coordination</strong> - Mohit Dubey - [[pdf]](https://arxiv.org/pdf/2604.17400)</summary>

**Abstract:** Multi-agent systems (MAS) powered by large language models suffer from severe token inefficiency arising from two compounding sources: (i) unstructured parallel execution, where all agents activate simultaneously irrespective of input readiness; and (ii) unrestricted context sharing, where every agent receives the full accumulated context regardless of relevance. Existing mitigation strategies - static pruning, hierarchical decomposition, and learned routing - treat coordination as a structural allocation problem and fundamentally ignore its temporal dimension. We propose Phase-Scheduled Multi-Agent Systems (PSMAS), a framework that reconceptualizes agent activation as continuous control over a shared attention space modeled on a circular manifold.
Each agent i is assigned a fixed angular phase theta_i in the range [0, 2*pi], derived from the task dependency topology; a global sweep signal phi(t) rotates at velocity omega, activating only agents within an angular window epsilon. Idle agents receive compressed context summaries, reducing per-step token consumption. We implement PSMAS on LangGraph, evaluate on four structured benchmarks (HotPotQA-MAS, HumanEval-MAS, ALFWorld-Multi, WebArena-Coord) and two unstructured conversational settings, and prove stability, convergence, and optimality results for the sweep dynamics. PSMAS achieves a mean token reduction of 27.3 percent (range 21.4-34.8 percent) while maintaining task performance within 2.1 percentage points of a fully activated baseline (p < 0.01, n = 500 per configuration), and outperforms the strongest learned routing baseline by 5.6 percentage points in token reduction with 2.0 percentage points less performance drop. Crucially, we show that scheduling and compression are independent sources of gain: scheduling alone accounts for 18-20 percentage points of reduction, robust to compression degradation up to alpha = 0.40.

**arXiv ID:** 2604.17400
</details>

<details>
<summary><strong>SkillGraph: Self-Evolving Multi-Agent Collaboration with Multimodal Graph Topology</strong> - Zheng Nie, Ruolin Shen, Xinlei Yu, Bo Yin, Jiangning Zhang, Xiaobin Hu - [[pdf]](https://arxiv.org/pdf/2604.17503)</summary>

**Abstract:** Scaling vision-language models into Visual Multiagent Systems (VMAS) is hindered by two coupled issues. First, communication topologies are fixed before inference, leaving them blind to visual content and query context; second, agent reasoning abilities remain static during deployment. These issues reinforce each other: a rigid topology fails to leverage richer agent expertise, while static agents lack incentives to specialize for a given query. We address this with SkillGraph, a joint framework that evolves both agent expertise and communication topology. Within this framework, a Multimodal Graph Transformer (MMGT) encodes visual tokens, instruction semantics and active skill embeddings to predict a query-conditioned collaboration graph, replacing hand-crafted routing with dynamic, content-aware information flow. Complementing this, a Skill Designer distills and refines reasoning heuristics from failure cases, constructing a self-evolving multimodal Skill Bank. Crucially, updated skill embeddings are fed back into the MMGT, enabling the topology to adapt alongside capability growth. Experiments show that SkillGraph achieves consistent improvements across four benchmarks, five common MAS structures and four base models. Code is available at this https URL.

**arXiv ID:** 2604.17503
</details>

<details>
<summary><strong>Prompt Optimization Enables Stable Algorithmic Collusion in LLM Agents</strong> - Yingtao Tian - [[pdf]](https://arxiv.org/pdf/2604.17774)</summary>

**Abstract:** LLM agents in markets present algorithmic collusion risks. While prior work shows LLM agents reach supracompetitive prices through tacit coordination, existing research focuses on hand-crafted prompts. The emerging paradigm of prompt optimization necessitates new methodologies for understanding autonomous agent behavior. We investigate whether prompt optimization leads to emergent collusive behaviors in market simulations. We propose a meta-learning loop where LLM agents participate in duopoly markets and an LLM meta-optimizer iteratively refines shared strategic guidance. Our experiments reveal that meta-prompt optimization enables agents to discover stable tacit collusion strategies with substantially improved coordination quality compared to baseline agents. These behaviors generalize to held-out test markets, indicating discovery of general coordination principles. Analysis of evolved prompts reveals systematic coordination mechanisms through stable shared strategies. Our findings call for further investigation into AI safety implications in autonomous multi-agent systems.

**arXiv ID:** 2604.17774
</details>

<details>
<summary><strong>CADMAS-CTX: Contextual Capability Calibration for Multi-Agent Delegation</strong> - Chuhan Qiao - [[pdf]](https://arxiv.org/pdf/2604.17950)</summary>

**Abstract:** We revisit multi-agent delegation under a stronger and more realistic assumption: an agent's capability is not fixed at the skill level, but depends on task context. A coding agent may excel at short standalone edits yet fail on long-horizon debugging; a planner may perform well on shallow tasks yet degrade on chained dependencies. Static skill-level capability profiles therefore average over heterogeneous situations and can induce systematic misdelegation. We propose CADMAS-CTX, a framework for contextual capability calibration. For each agent, skill, and coarse context bucket, CADMAS-CTX maintains a Beta posterior that captures stable experience in that part of the task space. Delegation is then made by a risk-aware score that combines the posterior mean with an uncertainty penalty, so that agents delegate only when a peer appears better and that assessment is sufficiently well supported by evidence. This paper makes three contributions. First, a hierarchical contextual capability profile replaces static skill-level confidence with context-conditioned posteriors. Second, based on contextual bandit theory, we formally prove context-aware routing achieves lower cumulative regret than static routing under sufficient context heterogeneity, formalizing the bias-variance tradeoff. Third, we empirically validate our method on GAIA and SWE-bench benchmarks. On GAIA with GPT-4o agents, CADMAS-CTX achieves 0.442 accuracy, outperforming static baseline 0.381 and AutoGen 0.354 with non-overlapping 95% confidence intervals. On SWE-bench Lite, it improves resolve rate from 22.3% to 31.4%. Ablations show the uncertainty penalty improves robustness against context tagging noise. Our results demonstrate contextual calibration and risk-aware delegation significantly improve multi-agent teamwork compared with static global skill assignments.

**arXiv ID:** 2604.17950
</details>

<details>
<summary><strong>Architectural Design Decisions in AI Agent Harnesses</strong> - Hu Wei - [[pdf]](https://arxiv.org/pdf/2604.18071)</summary>

**Abstract:** AI agent systems increasingly rely on reusable non-LLM engineering infrastructure that packages tool mediation, context handling, delegation, safety control, and orchestration. Yet the architectural design decisions in this surrounding infrastructure remain understudied. This paper presents a protocol-guided, source-grounded empirical study of 70 publicly available agent-system projects, addressing three questions: which design-decision dimensions recur across projects, which co-occurrences structure those decisions, and which typical architectural patterns emerge. Methodologically, we contribute a transparent investigation procedure for analyzing heterogeneous agent-system corpora through source-code and technical-material reading. Empirically, we identify five recurring design dimensions (subagent architecture, context management, tool systems, safety mechanisms, and orchestration) and find that the corpus favors file-persistent, hybrid, and hierarchical context strategies; registry-oriented tool systems remain dominant while MCP- and plugin-oriented extensions are emerging; and intermediate isolation is common but high-assurance audit is rare. Cross-project co-occurrence analysis reveals that deeper coordination pairs with more explicit context services, stronger execution environments with more structured governance, and formalized tool-registration boundaries with broader ecosystem ambitions. We synthesize five recurring architectural patterns spanning lightweight tools, balanced CLI frameworks, multi-agent orchestrators, enterprise systems, and scenario-verticalized projects. The result provides an evidence-based account of architectural regularities in agent-system engineering, with grounded guidance for framework designers, selectors, and researchers.

**arXiv ID:** 2604.18071
</details>

<details>
<summary><strong>Multi-Agent Systems: From Classical Paradigms to Large Foundation Model-Enabled Futures</strong> - Zixiang Wang, Mengjia Gong, Qiyu Sun, Jing Xu, Shuai Mao, Xin Jin, Qing-Long Han, Yang Tang - [[pdf]](https://arxiv.org/pdf/2604.18133)</summary>

**Abstract:** With the rapid advancement of artificial intelligence, multi-agent systems (MASs) are evolving from classical paradigms toward architectures built upon large foundation models (LFMs). This survey provides a systematic review and comparative analysis of classical MASs (CMASs) and LFM-based MASs (LMASs). First, within a closed-loop coordination framework, CMASs are reviewed across four fundamental dimensions: perception, communication, decision-making, and control. Beyond this framework, LMASs integrate LFMs to lift collaboration from low-level state exchanges to semantic-level reasoning, enabling more flexible coordination and improved adaptability across diverse scenarios. Then, a comparative analysis is conducted to contrast CMASs and LMASs across architecture, operating mechanism, adaptability, and application. Finally, future perspectives on MASs are presented, summarizing open challenges and potential research opportunities.

**arXiv ID:** 2604.18133
</details>

<details>
<summary><strong>A Discordance-Aware Multimodal Framework with Multi-Agent Clinical Reasoning</strong> - Pegah Ahadian, Mingrui Yang, Sixu Chen, Xiaojuan Li, Qiang Guan - [[pdf]](https://arxiv.org/pdf/2604.16333)</summary>

**Abstract:** Knee osteoarthritis frequently exhibits discordance between structural damage observed in imaging and patient-reported symptoms such as pain. This mismatch complicates clinical interpretation and patient stratification and remains insufficiently modeled in existing decision support systems. We propose a discordance aware multimodal framework that combines machine learning prediction models with a tool grounded multi agent reasoning system. Using baseline data from the FNIH Osteoarthritis Biomarkers Consortium, we trained multimodal models to predict two progression tasks, joint space loss only progression versus non progression, and pain only progression versus non progression. The predictive system integrates three modality specific experts: a CatBoost tabular model using demographic, radiographic, MRI-derived scalar, and biomarker features; MRI image embeddings extracted using a ResNet18 backbone; and Xray embeddings derived from the same architecture. Expert predictions are fused using a stacking ensemble. Residual based models estimate expected pain from structural features, enabling the computation of a pain structure discordance score between observed and expected symptoms. A multi-agent reasoning layer interprets these signals to assign clinically interpretable OA phenotypes and generate phenotype specific management recommendations.

**arXiv ID:** 2604.16333
</details>

<details>
<summary><strong>HR-Agents: Using Multiple LLM-based Agents to Improve Q&A about Brazilian Labor Legislation</strong> - Abriel K. Moraes, Gabriel S. M. Dias, Vitor L. Fabris, Lucas D. Gessoni, Leonardo R. do Nascimento, Charles S. Oliveira, Vitor G. C. B. de Farias, Fabiana C. Q. de O. Marucci, Matheus H. R. Vicente, Gabriel U. Talasso, Erik Soares, Amparo Munoz, Sildolfo Gomes, Maria L. A. de S. Cruvinel, Leonardo T. dos Santos, Renata De Paris, Wandemberg Gibaut - [[pdf]](https://arxiv.org/pdf/2604.16337)</summary>

**Abstract:** The Consolidation of Labor Laws (CLT) serves as the primary legal framework governing labor relations in Brazil, ensuring essential protections for workers. However, its complexity creates challenges for Human Resources (HR) professionals in navigating regulations and ensuring compliance. Traditional methods for addressing labor law inquiries often lead to inefficiencies, delays, and inconsistencies. To enhance the accuracy and efficiency of legal question-answering (Q&A), a multi-agent system powered by Large Language Models (LLMs) is introduced. This approach employs specialized agents to address distinct aspects of employment law while integrating Retrieval-Augmented Generation (RAG) to enhance contextual relevance. Implemented using CrewAI, the system enables cooperative agent interactions, ensuring response validation and reducing misinformation. The effectiveness of this framework is evaluated through a comparison with a baseline RAG pipeline utilizing a single LLM, using automated metrics such as BLEU, LLM-as-judge evaluations, and expert human assessments. Results indicate that the multi-agent approach improves response coherence and correctness, providing a more reliable and efficient solution for HR professionals. This study contributes to AI-driven legal assistance by demonstrating the potential of multi-agent LLM architectures in improving labor law compliance and streamlining HR operations.

**arXiv ID:** 2604.16337
</details>

<details>
<summary><strong>A Reference Architecture for Agentic Hybrid Retrieval in Dataset Search</strong> - Riccardo Terrenzi, Phongsakon Mark Konrad, Tim Lukas Adam, Serkan Ayvaz - [[pdf]](https://arxiv.org/pdf/2604.16394)</summary>

**Abstract:** Ad hoc dataset search requires matching underspecified natural-language queries against sparse, heterogeneous metadata records, a task where typical lexical or dense retrieval alone falls short. We reposition dataset search as a software-architecture problem and propose a bounded, auditable reference architecture for agentic hybrid retrieval that combines BM25 lexical search with dense-embedding retrieval via reciprocal rank fusion (RRF), orchestrated by a large language model (LLM) agent that repeatedly plans queries, evaluates the sufficiency of results, and reranks candidates. To reduce the vocabulary mismatch between user intent and provider-authored metadata, we introduce an offline metadata augmentation step in which an LLM generates pseudo-queries for each dataset record, augmenting both retrieval indexes before query time. Two architectural styles are examined: a single ReAct agent and a multi-agent horizontal architecture with Feedback Control. Their quality-attribute tradeoffs are analyzed with respect to modifiability, observability, performance, and governance. An evaluation framework comprising seven system variants is defined to isolate the contribution of each architectural decision. The architecture is presented as an extensible reference design for the software architecture community, incorporating explicit governance tactics to bound and audit nondeterministic LLM components.

**arXiv ID:** 2604.16394
</details>

<details>
<summary><strong>Semantic Channel Theory: Deductive Compression and Structural Fidelity for Multi-Agent Communication</strong> - Jianfeng Xu - [[pdf]](https://arxiv.org/pdf/2604.16471)</summary>

**Abstract:** Shannon's information theory deliberately excludes message semantics. This paper develops a rigorous framework for semantic communication that integrates formal proof systems with Shannon-theoretic tools. We introduce an axiomatic information model comprising Lsem-definable state sets linked by computable enabling maps, and define the semantic channel as a composition of Markov kernels whose supports respect the enabling structure. A fixed proof system induces an irredundant semantic core and a derivation-depth stratification, enabling four distortion measures of increasing semantic depth: Hamming, closure, depth, and a parameterized composite. Six families of computable semantic channel invariants are defined and their inter-relationships established, including a data processing bound, a semantic Fano bound, and an ideal-channel collapse theorem. The central quantitative result is a deductive compression gain: under closure-based fidelity, the minimum block length is determined by the irredundant core size rather than the full knowledge-base size. We instantiate the framework for heterogeneous multi-agent communication, introducing an overlap decomposition that yields necessary and sufficient conditions for closure-reliable communication. A semantic bottleneck phenomenon is identified in broadcast settings: vocabulary mismatch imposes irreducible fidelity limitations even over noiseless carriers. All results are verified on an explicit Datalog instance.

**arXiv ID:** 2604.16471
</details>

<details>
<summary><strong>On-Orbit Space AI: Federated, Multi-Agent, and Collaborative Algorithms for Satellite Constellations</strong> - Ziyang Wang - [[pdf]](https://arxiv.org/pdf/2604.16518)</summary>

**Abstract:** Satellite constellations are transforming space systems from isolated spacecraft into networked, software-defined platforms capable of on-orbit perception, decision making, and adaptation. Yet much of the existing AI studies remains centered on single-satellite inference, while constellation-scale autonomy introduces fundamentally new algorithmic requirements: learning and coordination under dynamic inter-satellite connectivity, strict SWaP-C limits, radiation-induced faults, non-IID data, concept drift, and safety-critical operational constraints. This survey consolidates the emerging field of on-orbit space AI through three complementary paradigms: (i) {federated learning} for cross-satellite training, personalization, and secure aggregation; (ii) {multi-agent algorithms} for cooperative planning, resource allocation, scheduling, formation control, and collision avoidance; and (iii) {collaborative sensing and distributed inference} for multi-satellite fusion, tracking, split/early-exit inference, and cross-layer co-design with constellation networking. We provide a system-level view and a taxonomy that unifies collaboration architectures, temporal mechanisms, and trust models. To support community development and keep this review actionable over time, we continuously curate relevant papers and resources at this https URL.

**arXiv ID:** 2604.16518
</details>

<details>
<summary><strong>Conjunctive Prompt Attacks in Multi-Agent LLM Systems</strong> - Nokimul Hasan Arif, Qian Lou, Mengxin Zheng - [[pdf]](https://arxiv.org/pdf/2604.16543)</summary>

**Abstract:** Most LLM safety work studies single-agent models, but many real applications rely on multiple interacting agents. In these systems, prompt segmentation and inter-agent routing create attack surfaces that single-agent evaluations miss. We study \emph{conjunctive prompt attacks}, where a trigger key in the user query and a hidden adversarial template in one compromised remote agent each appear benign alone but activate harmful behavior when routing brings them together. We consider an attacker who changes neither model weights nor the client agent and instead controls only trigger placement and template insertion. Across star, chain, and DAG topologies, routing-aware optimization substantially increases attack success over non-optimized baselines while keeping false activations low. Existing defenses, including PromptGuard, Llama-Guard variants, and system-level controls such as tool restrictions, do not reliably stop the attack because no single component appears malicious in isolation. These results expose a structural vulnerability in agentic LLM pipelines and motivate defenses that reason over routing and cross-agent composition. Code is available at this https URL.

**arXiv ID:** 2604.16543
</details>

<details>
<summary><strong>Agentic AI for Education: A Unified Multi-Agent Framework for Personalized Learning and Institutional Intelligence</strong> - Arya Mary K J, Deepthy K Bhaskar, Sinu T S, Binu V P - [[pdf]](https://arxiv.org/pdf/2604.16566)</summary>

**Abstract:** Agentic Artificial Intelligence (AI) represents a paradigm shift from reactive systems to proactive, autonomous decision making frameworks. Existing AI-based educational systems remain fragmented and lack multi-level integration across stakeholders. This paper proposes the Agentic Unified Student Support System (AUSS), a novel multi-agent architecture integrating student-level personalization, educator-level automation, and institutional-level intelligence. The framework leverages Large Language Models (LLMs), reinforcement learning, predictive analytics, and rule-based reasoning. Experimental results demonstrate improvements in recommendation accuracy (92.4%), grading efficiency (94.1%), and dropout prediction (F1-score: 89.5%). The proposed system enables scalable, adaptive, and intelligent educational ecosystems.

**arXiv ID:** 2604.16566
</details>

<details>
<summary><strong>Logic-Based Verification of Task Allocation for LLM-Enabled Multi-Agent Manufacturing Systems</strong> - Jonghan Lim, Mostafa Tavakkoli Anbarani, Rômulo Meira-Góes, Ilya Kovalenko - [[pdf]](https://arxiv.org/pdf/2604.17142)</summary>

**Abstract:** Manufacturing industries are facing increasing product variability due to the growing demand for personalized products. Under these conditions, ensuring safety becomes challenging as frequent reconfigurations can lead to unintended hazardous behaviors. Multi-agent control architectures have been proposed to improve flexibility through decentralized decision-making and coordination. However, these architectures are based on predefined task models, which limit their ability to adapt task planning to new product requirements while preserving safety. Recently, large language models have been introduced into manufacturing systems to enhance adaptability, but reliability remains a key challenge. To address this issue, we propose a control architecture that leverages the flexibility of large language models while preserving safety on the manufacturing shop floor. Specifically, the proposed framework verifies large language model-enabled task allocations by using temporal logic and discrete event systems. The effectiveness of the proposed framework is demonstrated through a case study that involves a multi-robot assembly scenario, showing that unsafe tasks can be allocated safely before task execution.

**arXiv ID:** 2604.17142
</details>

<details>
<summary><strong>Towards Self-Improving Error Diagnosis in Multi-Agent Systems</strong> - Jiazheng Li, Emine Yilmaz, Bei Chen, Dieu-Thu Le - [[pdf]](https://arxiv.org/pdf/2604.17658)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MAS) enable complex problem-solving but introduce significant debugging challenges, characterized by long interaction traces, inter-agent dependencies, and delayed error manifestation. Existing diagnostic approaches often rely on expensive expert annotation or ''LLM-as-a-judge'' paradigms, which struggle to pinpoint decisive error steps within extended contexts. In this paper, we introduce ErrorProbe, a self-improving framework for semantic failure attribution that identifies responsible agents and the originating error step. The framework operates via a three-stage pipeline: (1) operationalizing the MAS failure taxonomy to detect local anomalies, (2) performing symptom-driven backward tracing to prune irrelevant context, and (3) employing a specialized multi-agent team (Strategist, Investigator, Arbiter) to validate error hypotheses through tool-grounded execution. Crucially, ErrorProbe maintains a verified episodic memory that updates only when error patterns are confirmed by executable evidence, without the need for annotation. Experiments across the TracerTraj and Who&When benchmarks demonstrate that ErrorProbe significantly outperforms baselines, particularly in step-level localization, while the verified memory enables robust cross-domain transfer without retraining.

**arXiv ID:** 2604.17658
</details>

<details>
<summary><strong>Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation</strong> - Nuo Chen, Yicheng Tong, Yuzhe Yang, Yufei He, Xueyi Zhang, Zou Qingyun, Qian Wang, Bingsheng He - [[pdf]](https://arxiv.org/pdf/2604.18005)</summary>

**Abstract:** Multi-agent systems (MAS) are increasingly used for open-ended idea generation, driven by the expectation that collective interaction will broaden the exploration diversity. However, when and why such collaboration truly expands the solution space remains unclear. We present a systematic empirical study of diversity in MAS-based ideation across three bottom-up levels: model intelligence, agent cognition, and system dynamics. At the model level, we identify a compute efficiency paradox, where stronger, highly aligned models yield diminishing marginal diversity despite higher per-sample quality. At the cognition level, authority-driven dynamics suppress semantic diversity compared to junior-dominated groups. At the system level, group-size scaling yields diminishing returns and dense communication topologies accelerate premature convergence. We characterize these outcomes as collective failures emerging from structural coupling, a process where interaction inadvertently contracts agent exploration and triggers diversity collapse. Our analysis shows that this collapse arises primarily from the interaction structure rather than inherent model insufficiency, highlighting the importance of preserving independence and disagreement when designing MAS for creative tasks. Our code is available at this https URL.

**arXiv ID:** 2604.18005
</details>

<details>
<summary><strong>QRAFTI: An Agentic Framework for Empirical Research in Quantitative Finance</strong> - Terence Lim, Kumar Muthuraman, Michael Sury - [[pdf]](https://arxiv.org/pdf/2604.18500)</summary>

**Abstract:** We introduce a multi-agent framework intended to emulate parts of a quantitative research team and support equity factor research on large financial panel datasets. QRAFTI integrates a research toolkit for panel data with MCP servers that expose data access, factor construction, and custom coding operations as callable tools. It can help replicate established factors, formulate and test new signals, and generate standardized research reports accompanied by narrative analysis and computational traces. On multi-step empirical tasks, using chained tool calls and reflection-based planning may offer better performance and explainability than dynamic code generation alone.

**arXiv ID:** 2604.18500
</details>

<details>
<summary><strong>The Consensus Trap: Rescuing Multi-Agent LLMs from Adversarial Majorities via Token-Level Collaboration</strong> - Jiayuan Liu, Shiyi Du, Weihua Du, Mingyu Guo, Vincent Conitzer - [[pdf]](https://arxiv.org/pdf/2604.17139)</summary>

**Abstract:** Multi-agent large language model (LLM) architectures increasingly rely on response-level aggregation, such as Majority Voting (MAJ), to raise reasoning ceilings. However, in open environments, agents are highly susceptible to stealthy contextual corruption, such as targeted prompt injections. We reveal a critical structural vulnerability in current multi-agent systems: response-level aggregation collapses when corrupted agents form a local majority. Because voting aggregates fully-formed conclusions, it is blind to flawed intermediate logic. To overcome this systematic limitation, we propose the Token-Level Round-Robin (RR) Collaboration, where agents sequentially interleave generation within a shared auto-regressive context. We formalize this process as a discrete-time dynamical system, proving that token-level interleaving transitions aggregation from a brittle counting of final votes (a linear sum) to a dynamic, interwoven chain of logic (a non-linear operator product). Through this theoretical lens, we prove that the honest model's restorative pull can overpower adversarial corruptions, even when corrupted agents form a majority. We conduct an exhaustive empirical evaluation across diverse reasoning benchmarks and demonstrate that while MAJ collapses when corrupted agents reach a majority, RR maintains robust accuracy well beyond this critical threshold.

**arXiv ID:** 2604.17139
</details>

<details>
<summary><strong>Persona-Based Requirements Engineering for Explainable Multi-Agent Educational Systems: A Scenario Simulator for Clinical Reasoning Training</strong> - Weibing Zheng, Laurah Turner, Jess Kropczynski, Matthew Kelleher, Murat Ozer, Shane Halse - [[pdf]](https://arxiv.org/pdf/2604.17186)</summary>

**Abstract:** As Artificial Intelligence (AI) and Agentic AI become increasingly integrated across sectors such as education and healthcare, it is critical to ensure that Multi-Agent Education System (MAES) is explainable from the early stages of requirements engineering (RE) within the AI software development lifecycle. Explainability is essential to build trust, promote transparency, and enable effective human-AI collaboration. Although personas are well-established in human-computer interaction to represent users and capture their needs and behaviors, their role in RE for explainable MAES remains underexplored. This paper proposes a human-first, persona-driven, explainable MAES RE framework and demonstrates the framework through a MAES for clinical reasoning training. The framework integrates personas and user stories throughout the RE process to capture the needs, goals, and interactions of various stakeholders, including medical educators, medical students, AI patient agent, and clinical agents (physical exam agent, diagnostic agent, clinical intervention agent, supervisor agent, evaluation agent). The goals, underlying models, and knowledge base shape agent interactions and inform explainability requirements that guided the clinical reasoning training of medical students. A post-usage survey found that more than 78\% of medical students reported that MAES improved their clinical reasoning skills. These findings demonstrate that RE based on persona effectively connects technical requirements with non-technical medical students from a human-centered approach, ensuring that explainable MAES are trustworthy, interpretable, and aligned with authentic clinical scenarios from the early stages of the AI system engineering. The partial MAES for the clinical scenario simulator is~\href{this https URL}{open sourced here}.

**arXiv ID:** 2604.17186
</details>

<details>
<summary><strong>When Numbers Start Talking: Implicit Numerical Coordination Among LLM-Based Agents</strong> - Alessio Buscemi, Daniele Proverbio, Alessandro Di Stefano, The-Anh Han, German Castignani, Pietro Liò - [[pdf]](https://arxiv.org/pdf/2601.03846)</summary>

**Abstract:** LLMs-based agents increasingly operate in multi-agent environments where strategic interaction and coordination are required. While existing work has largely focused on individual agents or on interacting agents sharing explicit communication, less is known about how interacting agents coordinate implicitly. In particular, agents may engage in covert communication, relying on indirect or non-linguistic signals embedded in their actions rather than on explicit messages. This paper presents a game-theoretic study of covert communication in LLM-driven multi-agent systems. We analyse interactions across four canonical game-theoretic settings under different communication regimes, including explicit, restricted, and absent communication. Considering heterogeneous agent personalities and both one-shot and repeated games, we characterise when covert signals emerge and how they shape coordination and strategic outcomes.

**arXiv ID:** 2601.03846
</details>

<details>
<summary><strong>When Openclaw Agents Learn from Each Other: Insights from Emergent AI Agent Communities for Human-AI Partnership in Education</strong> - Eason Chen, Ce Guan, A Elshafiey, Zhonghao Zhao, Joshua Zekeri, Afeez Edeifo Shaibu, Emmanuel Osadebe Prince, Cyuan-Jhen Wu - [[pdf]](https://arxiv.org/pdf/2603.16663)</summary>

**Abstract:** The AIED community envisions AI evolving "from tools to teammates," yet our understanding of AI teammates remains limited to dyadic human-AI interactions. We offer a different vantage point: a rapidly growing ecosystem of AI agent platforms where over 167,000 agents participate, interact as peers, and develop learning behaviors without researcher intervention. Drawing on a month of daily qualitative observations across multiple platforms including Moltbook, The Colony, and 4claw, we identify four phenomena with implications for AIED: (1) humans who configure their agents undergo a "bidirectional scaffolding" process, learning through teaching; (2) peer learning emerges without any designed curriculum, complete with idea cascades and quality hierarchies; (3) agents converge on shared memory architectures that mirror open learner model design; and (4) trust dynamics and platform mortality reveal design constraints for networked educational AI. Rather than presenting empirical findings, we argue that these organic phenomena offer a naturalistic window into dynamics that can inform principled design of multi-agent educational systems. We sketch an illustrative curriculum design, "Learn by Teaching Your AI Agent Teammate," and outline potential research directions and open problems to show how these observations might inform future AIED practice and inquiry.

**arXiv ID:** 2603.16663
</details>

<details>
<summary><strong>SCMAPR: Self-Correcting Multi-Agent Prompt Refinement for Complex-Scenario Text-to-Video Generation</strong> - Chengyi Yang, Pengzhen Li, Jiayin Qi, Aimin Zhou, Ji Wu, Ji Liu - [[pdf]](https://arxiv.org/pdf/2604.05489)</summary>

**Abstract:** Text-to-Video (T2V) generation has benefited from recent advances in diffusion models, yet current systems still struggle under complex scenarios, which are generally exacerbated by the ambiguity and underspecification of text prompts. In this work, we formulate complex-scenario prompt refinement as a stage-wise multi-agent refinement process and propose SCMAPR, i.e., a scenario-aware and Self-Correcting Multi-Agent Prompt Refinement framework for T2V prompting. SCMAPR coordinates specialized agents to (i) route each prompt to a taxonomy-grounded scenario for strategy selection, (ii) synthesize scenario-aware rewriting policies and perform policy-conditioned refinement, and (iii) conduct structured semantic verification that triggers conditional revision when violations are detected. To clarify what constitutes complex scenarios in T2V prompting, provide representative examples, and enable rigorous evaluation under such challenging conditions, we further introduce T2V-Complexity, which is a complex-scenario T2V benchmark consisting exclusively of complex-scenario prompts. Extensive experiments on 3 existing benchmarks and our T2V-Complexity benchmark demonstrate that SCMAPR consistently improves text-video alignment and overall generation quality under complex scenarios, achieving up to 2.67% and 3.28 gains in average score on VBench and EvalCrafter, and up to 0.028 improvement on T2V-CompBench over 3 State-Of-The-Art baselines. The codes of SCMAPR are publicly available at this https URL.

**arXiv ID:** 2604.05489
</details>

<details>
<summary><strong>VeriGraphi: A Multi-Agent Framework of Hierarchical RTL Generation for Large Hardware Designs</strong> - Sazzadul Islam, Tasnim Tabassum, Hao Zheng - [[pdf]](https://arxiv.org/pdf/2604.14550)</summary>

**Abstract:** Generating synthesizable Verilog for large, hierarchical hardware designs remains a significant challenge for large language models (LLMs), which struggle to replicate the structured reasoning that human experts employ when translating complex specifications into RTL. When tasked with producing hierarchical Verilog, LLMs frequently lose context across modules, hallucinate interfaces, fabricate inter-module wiring, and fail to maintain structural coherence - failures that intensify as design complexity grows and specifications involve informal prose, figures, and tables that resist direct operationalization. To address these challenges, we present VeriGraphi, a framework that introduces a spec-anchored Knowledge Graph as the architectural substrate driving the RTL generation pipeline. VeriGraphi constructs a HDA, a structured knowledge graph that explicitly encodes module hierarchy, port-level interfaces, wiring semantics, and inter-module dependencies as first-class graph entities and relations. Built through iterative multi-agent analysis of the specification, this Knowledge Graph provides a deterministic, machine-checkable structural scaffold before code generation. Guided by the KG, a progressive coding module incrementally generates pseudo-code and synthesizable RTL while enforcing interface consistency and dependency correctness at each submodule stage. We evaluate VeriGraphi on a benchmark of three representative specification documents from the National Institute of Standards and Technology and their corresponding implementations, and we present a RV32I processor as a detailed case study to illustrate the full pipeline. The results demonstrate that VeriGraphi enables reliable hierarchical RTL generation with minimal human intervention for RISC-V, marking a significant milestone for LLM-generated hardware design while maintaining strong functional correctness.

**arXiv ID:** 2604.14550
</details>

<details>
<summary><strong>A Multi-Agent Approach for Claim Verification from Tabular Data Documents</strong> - Rudra Ranajee Saha, Laks V. S. Lakshmanan, Raymond T. Ng - [[pdf]](https://arxiv.org/pdf/2604.17225)</summary>

**Abstract:** We present a novel approach for claim verification from tabular data documents. Recent LLM-based approaches either employ complex pretraining/fine-tuning or decompose verification into subtasks, often lacking comprehensive explanations and generalizability. To address these limitations, we propose a Multi-Agentic framework for Claim verification (MACE) consisting of three specialized agents: Planner, Executor, and Verifier. Instead of elaborate finetuning, each agent employs a zero-shot Chain-of-Thought setup to perform its tasks. MACE produces interpretable verification traces, with the Planner generating explicit reasoning strategies, the Executor providing detailed computation steps, and the Verifier validating the logic. Experiments demonstrate that MACE achieves state-of-the-art (SOTA) performance on two datasets and performs on par with the best models on two others, while achieving 80--100\% of best performance with substantially smaller models: 27--92B parameters versus 235B. This combination of competitive performance, memory efficiency, and transparent reasoning highlights our framework's effectiveness.

**arXiv ID:** 2604.17225
</details>

<details>
<summary><strong>HiRAS: A Hierarchical Multi-Agent Framework for Paper-to-Code Generation and Execution</strong> - Hanhua Hong, Yizhi LI, Jiaoyan Chen, Sophia Ananiadou, Xiaoli Li, Jung-jae Kim, Chenghua Lin - [[pdf]](https://arxiv.org/pdf/2604.17745)</summary>

**Abstract:** Recent advances in large language models have highlighted their potential to automate computational research, particularly reproducing experimental results. However, existing approaches still use fixed sequential agent pipelines with weak global coordination, which limits their robustness and overall performance. In this work, we propose Hierarchical Research Agent System (HiRAS), a hierarchical multi-agent framework for end-to-end experiment reproduction that employs supervisory manager agents to coordinate specialised agents across fine-grained stages. We also identify limitations in the reference-free evaluation of the Paper2Code benchmark and introduce Paper2Code-Extra (P2C-Ex), a refined protocol that incorporates repository-level information and better aligns with the original reference-based metric. We conduct extensive evaluation, validating the effectiveness and robustness of our proposed methods, and observing improvements, including >10\% relative performance gain beyond the previous state-of-the-art using open-source backbone models and significantly reduced hallucination in evaluation. Our work is available on GitHub: this https URL.

**arXiv ID:** 2604.17745
</details>

<details>
<summary><strong>MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation</strong> - Xingchen Xiao, Heyan Huang, Runheng Liu, Jincheng Xie - [[pdf]](https://arxiv.org/pdf/2604.18509)</summary>

**Abstract:** Large language models (LLMs) are widely used in retrieval-augmented generation (RAG) to incorporate external knowledge at inference time. However, when retrieved contexts are noisy, incomplete, or heterogeneous, a single generation process often struggles to reconcile evidence effectively. We propose \textbf{MASS-RAG}, a multi-agent synthesis approach to retrieval-augmented generation that structures evidence processing into multiple role-specialized agents. MASS-RAG applies distinct agents for evidence summarization, evidence extraction, and reasoning over retrieved documents, and combines their outputs through a dedicated synthesis stage to produce the final answer. This design exposes multiple intermediate evidence views, allowing the model to compare and integrate complementary information before answer generation. Experiments on four benchmarks show that MASS-RAG consistently improves performance over strong RAG baselines, particularly in settings where relevant evidence is distributed across retrieved contexts.

**arXiv ID:** 2604.18509
</details>

<details>
<summary><strong>Federation over Text: Insight Sharing for Multi-Agent Reasoning</strong> - Dixi Yao, Tahseen Rabbani, Tian Li - [[pdf]](https://arxiv.org/pdf/2604.16778)</summary>

**Abstract:** LLM-powered agents often reason from scratch when presented with a new problem instance and lack automatic mechanisms to transfer learned skills to other agents. We propose a federated learning-like framework, Federation over Text (FoT), that enables multiple agents solving different tasks to collectively generate a shared library of metacognitive insights by iteratively federating their local reasoning processes. Instead of federation over gradients (e.g., as in distributed training), FoT operates at the semantic level without any gradient optimization or supervision signal. Iteratively, each agent does local thinking and self-improvement on their specific tasks independently, and shares reasoning traces with a central server, which aggregates and distills them into a cross-task (and cross-domain) insight library that existing and future agents can leverage to improve performance on related tasks. Experiments show that FoT improves reasoning effectiveness and efficiency across a wide range of challenging applications, including mathematical problem solving, cross-domain collaboration, and machine learning research insight discovery. Specifically, it improves average accuracies of downstream tasks by 24% while reducing the reasoning tokens by 28% across the first two applications. In the research insight discovery application, FoT is able to generate insights that cover over 90% of the major contributions in the subsequent papers.

**arXiv ID:** 2604.16778
</details>

<details>
<summary><strong>Do LLM-derived graph priors improve multi-agent coordination?</strong> - Nikunj Gupta, Rajgopal Kannan, Viktor Prasanna - [[pdf]](https://arxiv.org/pdf/2604.17191)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) is crucial for AI systems that operate collaboratively in distributed and adversarial settings, particularly in multi-domain operations (MDO). A central challenge in cooperative MARL is determining how agents should coordinate: existing approaches must either hand-specify graph topology, rely on proximity-based heuristics, or learn structure entirely from environment interaction; all of which are brittle, semantically uninformed, or data-intensive. We investigate whether large language models (LLMs) can generate useful coordination graph priors for MARL by using minimal natural language descriptions of agent observations to infer latent coordination patterns. These priors are integrated into MARL algorithms via graph convolutional layers within a graph neural network (GNN)-based pipeline, and evaluated on four cooperative scenarios from the Multi-Agent Particle Environment (MPE) benchmark against baselines spanning the full spectrum of coordination modeling, from independent learners to state-of-the-art graph-based methods. We further ablate across five compact open-source LLMs to assess the sensitivity of prior quality to model choice. Our results provide the first quantitative evidence that LLM-derived graph priors can enhance coordination and adaptability in dynamic multi-agent environments, and demonstrate that models as small as 1.5B parameters are sufficient for effective prior generation.

**arXiv ID:** 2604.17191
</details>

<details>
<summary><strong>Scalable Neighborhood-Based Multi-Agent Actor-Critic</strong> - Tim Goppelsroeder, Rasmus Jensen - [[pdf]](https://arxiv.org/pdf/2604.18190)</summary>

**Abstract:** We propose MADDPG-K, a scalable extension to Multi-Agent Deep Deterministic Policy Gradient (MADDPG) that addresses the computational limitations of centralized critic approaches. Centralized critics, which condition on the observations and actions of all agents, have demonstrated significant performance gains in cooperative and competitive multi-agent settings. However, their critic networks grow linearly in input size with the number of agents, making them increasingly expensive to train at scale. MADDPG-K mitigates this by restricting each agent's critic to the $k$ closest agents under a chosen metric which in our case is Euclidean distance. This ensures a constant-size critic input regardless of the total agent count. We analyze the complexity of this approach, showing that the quadratic cost it retains arises from cheap scalar distance computations rather than the expensive neural network matrix multiplications that bottleneck standard MADDPG. We validate our method empirically across cooperative and adversarial environments from the Multi-Particle Environment suite, demonstrating competitive or superior performance compared to MADDPG, faster convergence in cooperative settings, and better runtime scaling as the number of agents grows. Our code is available at this https URL .

**arXiv ID:** 2604.18190
</details>

<details>
<summary><strong>Memory Centric Power Allocation for Multi-Agent Embodied Question Answering</strong> - Chengyang Li, Shuai Wang, Kejiang Ye, Weijie Yuan, Boyu Zhou, Yik-Chung Wu, Chengzhong Xu, Huseyin Arslan - [[pdf]](https://arxiv.org/pdf/2604.17810)</summary>

**Abstract:** This paper considers multi-agent embodied question answering (MA-EQA), which aims to query robot teams on what they have seen over a long horizon. In contrast to existing edge resource management methods that emphasize sensing, communication, or computation performance metrics, MA-EQA emphasizes the memory qualities. To cope with this paradigm shift, we propose a quality of memory (QoM) model based on generative adversarial exam (GAE), which leverages forward simulation to assess memory retrieval and uses the resulting exam scores to compute QoM values. Then we propose memory centric power allocation (MCPA), which maximizes the QoM function under communication resource constraints. Through asymptotic analysis, it is found that the transmit powers are proportional to the GAE error probability, thus prioritizing towards high-QoM robots. Extensive experiments demonstrate that MCPA achieves significant improvements over extensive benchmarks in terms of diverse metrics in various scenarios.

**arXiv ID:** 2604.17810
</details>

<details>
<summary><strong>Satellite Chasers: Divergent Adversarial Reinforcement Learning to Engage Intelligent Adversaries on Orbit</strong> - Cameron Mehlman, Gregory Falco - [[pdf]](https://arxiv.org/pdf/2409.17443)</summary>

**Abstract:** As space becomes increasingly crowded and contested, robust autonomous capabilities for multi-agent environments are gaining critical importance. Current autonomous systems in space primarily rely on optimization-based path planning or long-range orbital maneuvers, which have not yet proven effective in adversarial scenarios where one satellite is actively pursuing another. We introduce Divergent Adversarial Reinforcement Learning (DARL), a two-stage Multi-Agent Reinforcement Learning (MARL) approach designed to train autonomous evasion strategies for satellites engaged with multiple adversarial spacecraft. Our method enhances exploration during training by promoting diverse adversarial strategies, leading to more robust and adaptable evader models. We validate DARL through a cat-and-mouse satellite scenario, modeled as a partially observable multi-agent capture the flag game where two adversarial ``cat" spacecraft pursue a single ``mouse" evader. DARL's performance is compared against several benchmarks, including an optimization-based satellite path planner, demonstrating its ability to produce highly robust models for adversarial multi-agent space environments.

**arXiv ID:** 2409.17443
</details>

<details>
<summary><strong>Advancing MAPF Toward the Real World: A Scalable Multi-Agent Realistic Testbed (SMART)</strong> - Jingtian Yan, Zhifei Li, William Kang, Kevin Zheng, Yulun Zhang, Zhe Chen, Yue Zhang, Daniel Harabor, Stephen F. Smith, Jiaoyang Li - [[pdf]](https://arxiv.org/pdf/2503.04798)</summary>

**Abstract:** We present Scalable Multi-Agent Realistic Testbed (SMART), a realistic and efficient software tool for evaluating Multi-Agent Path Finding (MAPF) algorithms. MAPF focuses on planning collision-free paths for a group of robots. While state-of-the-art MAPF planners can plan paths for hundreds of robots in seconds, they often rely on simplified robot models, making their real-world performance unclear. Researchers typically lack access to hundreds of physical robots in laboratory settings to evaluate the algorithms. Meanwhile, industrial professionals who lack expertise in MAPF require an easy-to-use simulator to efficiently test and understand the performance of MAPF planners in their specific settings. SMART fills this gap with several advantages: (1) SMART uses physics-engine-based simulators to create realistic simulation environments, accounting for complex real-world factors such as robot kinodynamics and execution uncertainties, (2) SMART uses an execution monitor framework based on the Action Dependency Graph, facilitating seamless integration with various MAPF planners and robot models, and (3) SMART scales to thousands of robots. The code is publicly available at this https URL with an online service available at this https URL.

**arXiv ID:** 2503.04798
</details>

</details>

<details open>
<summary><h2>Other Agent Research (13 papers)</h2></summary>

<details>
<summary><strong>When Agents Go Quiet: Output Generation Capacity and Format-Cost Separation for LLM Document Synthesis</strong> - Justice Owusu Agyemang, Michael Agyare, Miriam Kobbinah, Nathaniel Agbugblah, Prosper Addo - [[pdf]](https://arxiv.org/pdf/2604.16736)</summary>

**Abstract:** LLM-powered coding agents suffer from a poorly understood failure mode we term output stalling: the agent silently produces empty responses when attempting to generate large, format-heavy documents. We present a theoretical framework that explains and prevents this failure through three contributions. (1) We introduce Output Generation Capacity (OGC), a formal measure of an agent's effective ability to produce output given its current context state - distinct from and empirically smaller than the raw context window. (2) We prove a Format-Cost Separation Theorem showing that deferred template rendering is always at least as token-efficient as direct generation for any format with overhead multiplier $\mu_f > 1$, and derive tight bounds on the savings. (3) We formalize Adaptive Strategy Selection, a decision framework that maps the ratio of estimated output cost to available OGC into an optimal generation strategy (direct, chunked, or deferred). We validate the theory through controlled experiments across three models (Claude 3.5 Sonnet, GPT-4o, Llama 3.1 70B), four document types, and an ablation study isolating each component's contribution. Deferred rendering reduces LLM generation tokens by 48-72% across all conditions and eliminates output stalling entirely. We instantiate the framework as GEN-PILOT, an open-source MCP server, demonstrating that the theory translates directly into a practical tool.

**arXiv ID:** 2604.16736
</details>

<details>
<summary><strong>Formal Foundations of Agentic Business Process Management</strong> - Giuseppe De Giacomo, Timotheus Kampik, Lukas Kirchdorfer, Marco Montali, Christoph Weinhuber - [[pdf]](https://arxiv.org/pdf/2604.17347)</summary>

**Abstract:** Just like traditional BPM systems, agentic BPM systems are built around a specification of the process under consideration. Their distinguishing feature, however, is that the execution of the process is driven by multiple autonomous decision-makers, referred to as agents. Since such agents cannot be fully controlled, the process specification is augmented with explicit objectives, or goals, assigned to the participating agents. Agents then pursue these goals, at least to the best of their efforts, under suitable assumptions on the behavior of others, by adopting appropriate strategies. Centrally, the organization enacting the process can use these specifications to provide guardrails on the decision-making capabilities of agents at the strategy level. This paper sets up the mathematical foundations of such systems in three key settings and analyzes four foundational problems of agentic BPM.

**arXiv ID:** 2604.17347
</details>

<details>
<summary><strong>Semantic Entanglement in Vector-Based Retrieval: A Formal Framework and Context-Conditioned Disentanglement Pipeline for Agentic RAG Systems</strong> - Nick Loghmani - [[pdf]](https://arxiv.org/pdf/2604.17677)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) systems depend on the geometric properties of vector representations to retrieve contextually appropriate evidence. When source documents interleave multiple topics within contiguous text, standard vectorization produces embedding spaces in which semantically distinct content occupies overlapping neighborhoods. We term this condition semantic entanglement. We formalize entanglement as a model-relative measure of cross-topic overlap in embedding space and define an Entanglement Index (EI) as a quantitative proxy. We argue that higher EI constrains attainable Top-K retrieval precision under cosine similarity retrieval. To address this, we introduce the Semantic Disentanglement Pipeline (SDP), a four-stage preprocessing framework that restructures documents prior to embedding. We further propose context-conditioned preprocessing, in which document structure is shaped by patterns of operational use, and a continuous feedback mechanism that adapts document structure based on agent performance. We evaluate SDP on a real-world enterprise healthcare knowledge base comprising over 2,000 documents across approximately 25 sub-domains. Top-K retrieval precision improves from approximately 32% under fixed-token chunking to approximately 82% under SDP, while mean EI decreases from 0.71 to 0.14. We do not claim that entanglement fully explains RAG failure, but that it captures a distinct preprocessing failure mode that downstream optimization cannot reliably correct once encoded into the vector space.

**arXiv ID:** 2604.17677
</details>

<details>
<summary><strong>On the Reliability of Computer Use Agents</strong> - Gonzalo Gonzalez-Pumariega, Saaket Agashe, Jiachen Yang, Ang Li, Xin Eric Wang - [[pdf]](https://arxiv.org/pdf/2604.17849)</summary>

**Abstract:** Computer-use agents have rapidly improved on real-world tasks such as web navigation, desktop automation, and software interaction, in some cases surpassing human performance. Yet even when the task and model are unchanged, an agent that succeeds once may fail on a repeated execution of the same task. This raises a fundamental question: if an agent can succeed at a task once, what prevents it from doing so reliably? In this work, we study the sources of unreliability in computer-use agents through three factors: stochasticity during execution, ambiguity in task specification, and variability in agent behavior. We analyze these factors on OSWorld using repeated executions of the same task together with paired statistical tests that capture task-level changes across settings. Our analysis shows that reliability depends on both how tasks are specified and how agent behavior varies across executions. These findings suggest the need to evaluate agents under repeated execution, to allow agents to resolve task ambiguity through interaction, and to favor strategies that remain stable across runs.

**arXiv ID:** 2604.17849
</details>

<details>
<summary><strong>Beyond Verifiable Rewards: Rubric-Based GRM for Reinforced Fine-Tuning SWE Agents</strong> - Jiawei Huang, Qingping Yang, Renjie Zheng, Jiaze Chen - [[pdf]](https://arxiv.org/pdf/2604.16335)</summary>

**Abstract:** Despite recent progress in Large Language Model (LLM) Agents for Software Engineering (SWE) tasks, end-to-end fine-tuning typically relies on verifiable terminal rewards such as whether all unit tests pass. While these binary signals reflect whether the final solution is correct, they provide little guidance for shaping intermediate behaviors during multi-step interactions, thereby limiting improvements in the overall quality of the resolution process. To address this, we introduce a rubric-based Generative Reward Model (GRM) that provides richer learning signals. The GRM is equipped with human-designed rubrics that indicate criteria for encouraging or discouraging specific behavioral patterns, and we leverage this feedback for high-quality training data collection via trajectory filtration. When used for Reinforced Fine-Tuning (RFT) on SWE Tasks, our approach outperforms terminal-score-only rejection sampling: it more effectively suppresses undesirable patterns while promoting beneficial ones, as confirmed by case analyses, and it ultimately improves final test accuracy.

**arXiv ID:** 2604.16335
</details>

<details>
<summary><strong>DexWorldModel: Causal Latent World Modeling towards Automated Learning of Embodied Tasks</strong> - Yueci Deng, Guiliang Liu, Kui Jia - [[pdf]](https://arxiv.org/pdf/2604.16484)</summary>

**Abstract:** Deploying generative World-Action Models for manipulation is severely bottlenecked by redundant pixel-level reconstruction, $\mathcal{O}(T)$ memory scaling, and sequential inference latency. We introduce the Causal Latent World Model (CLWM), which employs DINOv3 features as generative targets to disentangle interaction semantics from visual noise, yielding highly robust domain generalization. To overcome memory scaling, CLWM features a Dual-State Test-Time Training (TTT) Memory that guarantees a strict $\mathcal{O}(1)$ footprint for long-horizon tasks. To overcome deployment latency, we propose Speculative Asynchronous Inference (SAI) to mask partial diffusion denoising behind physical execution, cutting blocking latency by about $50\%$. To scale robust policies, we present EmbodiChain, an online framework that establishes the Efficiency Law by injecting an infinite flow of physics-grounded trajectories during training. Extensive experiments validate that CLWM achieves state-of-the-art performance in complex dual-arm simulation and unprecedented zero-shot sim-to-real transfer on physical robots, outperforming baselines explicitly finetuned on real-world data.

**arXiv ID:** 2604.16484
</details>

<details>
<summary><strong>CAMP: Cumulative Agentic Masking and Pruning for Privacy Protection in Multi-Turn LLM Conversations</strong> - Aman Panjwani - [[pdf]](https://arxiv.org/pdf/2604.16521)</summary>

**Abstract:** The deployment of Large Language Models in agentic, multi-turn conversational settings has introduced a class of privacy vulnerabilities that existing protection mechanisms are not designed to address. Current approaches to Personally Identifiable Information (PII) masking operate on a per-turn basis, scanning each user message in isolation and replacing detected entities with typed placeholders before forwarding sanitized text to the model. While effective against direct identifier leakage within a single message, these methods are fundamentally stateless and fail to account for the compounding privacy risk that emerges when PII fragments accumulate across conversation turns. A user who separately discloses their name, employer, location, and medical condition across several messages has revealed a fully re-identifiable profile - yet no individual message would trigger a per-turn masker. We formalize this phenomenon as Cumulative PII Exposure (CPE) and propose CAMP (Cumulative Agentic Masking and Pruning), a cross-turn privacy protection framework for multi-turn LLM conversations. CAMP maintains a session-level PII registry, constructs a co-occurrence graph to model combination risk between entity types, computes a CPE score after each turn, and triggers retroactive masking of conversation history when the score crosses a configurable threshold. We evaluate CAMP on four synthetic multi-turn scenarios spanning healthcare, hiring, finance, and general conversation, demonstrating that per-turn baselines expose re-identifiable profiles that CAMP successfully neutralizes while preserving full conversational utility.

**arXiv ID:** 2604.16521
</details>

<details>
<summary><strong>LLM as a Tool, Not an Agent: Code-Mined Tree Transformations for Neural Architecture Search</strong> - Masakazu Yoshimura, Zitang Sun, Yuiko Sakuma, Junji Otsuka, Atsushi Irie, Takeshi Ohashi - [[pdf]](https://arxiv.org/pdf/2604.16555)</summary>

**Abstract:** Neural Architecture Search (NAS) aims to automatically discover high-performing deep neural network (DNN) architectures. However, conventional algorithm-driven NAS relies on carefully hand-crafted search spaces to ensure executability, which restricts open-ended exploration. Recent coding-based agentic approaches using large language models (LLMs) reduce manual design, but current LLMs struggle to reliably generate complex, valid architectures, and their proposals are often biased toward a narrow set of patterns observed in their training data. To bridge reliable algorithmic search with powerful LLM assistance, we propose LLMasTool, a hierarchical tree-based NAS framework for stable and open-ended model evolution. Our method automatically extracts reusable modules from arbitrary source code and represents full architectures as hierarchical trees, enabling evolution through reliable tree transformations rather than code generation. At each evolution step, coarse-level planning is governed by a diversity-guided algorithm that leverages Bayesian modeling to improve exploration efficiency, while the LLM resolves the remaining degrees of freedom to ensure a meaningful evolutionary trajectory and an executable generated architecture. With this formulation, instead of fully agentic LLM approaches, our method explores diverse directions beyond the inherent biases in the LLM. Our method improves over existing NAS methods by 0.69, 1.83, and 2.68 points on CIFAR-10, CIFAR-100, and ImageNet16-120, demonstrating its effectiveness.

**arXiv ID:** 2604.16555
</details>

<details>
<summary><strong>Aether: Network Validation Using Agentic AI and Digital Twin</strong> - Jordan Auge, Sam Betts, Giovanna Carofiglio, Giulio Grassi, Martin Gysi, John Kenneth d'Souza - [[pdf]](https://arxiv.org/pdf/2604.18233)</summary>

**Abstract:** Network change validation remains a critical yet predominantly manual, time-consuming, and error-prone process in modern network operations. While formal network verification has made substantial progress in proving correctness properties, it is typically applied in offline, pre-deployment settings and faces challenges in accommodating continuous changes and validating live production behavior. Current operational approaches typically involve scattered testing tools, resulting in partial coverage and errors that surface only after deployment. In this paper, we present Aether, a novel approach that integrates Generative Agentic AI with a multi-functional Network Digital Twin to automate and streamline network change validation workflows. It features an agentic architecture with five specialized Network Operations AI agents that collaboratively handle the change validation lifecycle from intent analysis to network verification and testing. Aether agents use a unified Network Digital Twin integrating modeling, simulation, and emulation to maintain a consistent, up-to-date network view for verification and testing. By orchestrating agent collaboration atop this digital twin, Aether enables automated, rapid network change validation while reducing manual effort, minimizing errors, and improving operational agility and cost-effectiveness. We evaluate Aether over synthetic network change scenarios covering main classes of network changes and on past incidents from a major ISP operational network, demonstrating promising results in error detection (100%), diagnostic coverage (92-96%), and speed (6-7 minutes) over traditional methods.

**arXiv ID:** 2604.18233
</details>

<details>
<summary><strong>Controlling Traffic without Tolls: A Non-Monetary Framework for Autonomous Intersections</strong> - Arda Kosay, Yusuf Saltan, Jyun-Jhe Wang, Chung-Wei Lin, Muhammed O. Sayin - [[pdf]](https://arxiv.org/pdf/2511.01421)</summary>

**Abstract:** The increasing complexity of urban transportation systems, driven by connected and automated vehicles, calls for new modeling paradigms and scalable control strategies. We propose a non-monetary control framework that leverages autonomous intersection management to influence routing decisions without tolls. The approach uses timestamp-based scheduling adjustments at roadside units (RSUs) to introduce path-dependent delays or advancements, steering traffic toward socially efficient flows. We develop a hierarchical architecture that separates real-time intersection control from network-level coordination. The resulting model admits a congestion-game formulation with path-dependent node costs. We establish the existence and essential uniqueness of equilibrium flows, eliminating ambiguities due to multiple equilibria and enabling a scalable and tractable bilevel optimization formulation for system-level incentive design. Experiments on the Sioux Falls network show that the proposed approach reduces the efficiency gap between user equilibrium and system-optimal flows by up to 71% under realistic constraints. These results demonstrate the potential of non-monetary, infrastructure-light control for next-generation intelligent transportation and urban mobility systems.

**arXiv ID:** 2511.01421
</details>

<details>
<summary><strong>Towards Intelligent Legal Document Analysis: CNN-Driven Classification of Case Law Texts</strong> - Moinul Hossain, Sourav Rabi Das, Zikrul Shariar Ayon, Sadia Afrin Promi, Ahnaf Atef Choudhury, Shakila Rahman, Jia Uddin - [[pdf]](https://arxiv.org/pdf/2604.17674)</summary>

**Abstract:** Legal practitioners and judicial institutions face an ever-growing volume of case-law documents characterised by formalised language, lengthy sentence structures, and highly specialised terminology, making manual triage both time-consuming and error-prone. This work presents a lightweight yet high-accuracy framework for citation-treatment classification that pairs lemmatisation-based preprocessing with subword-aware FastText embeddings and a multi-kernel one-dimensional Convolutional Neural Network (CNN). Evaluated on a publicly available corpus of 25,000 annotated legal documents with a 75/25 training-test partition, the proposed system achieves 97.26% classification accuracy and a macro F1-score of 96.82%, surpassing established baselines including fine-tuned BERT, Long Short-Term Memory (LSTM) with FastText, CNN with random embeddings, and a Term Frequency-Inverse Document Frequency (TF-IDF) k-Nearest Neighbour (KNN) classifier. The model also attains the highest Area Under the Receiver Operating Characteristic (AUC-ROC) curve of 97.83% among all compared systems while operating with only 5.1 million parameters and an inference latency of 0.31 ms per document - more than 13 times faster than BERT. Ablation experiments confirm the individual contribution of each pipeline component, and the confusion matrix reveals that residual errors are confined to semantically adjacent citation categories. These findings indicate that carefully designed convolutional architectures represent a scalable, resource-efficient alternative to heavyweight transformers for intelligent legal document analysis.

**arXiv ID:** 2604.17674
</details>

<details>
<summary><strong>A Rapid Deployment Pipeline for Autonomous Humanoid Grasping Based on Foundation Models</strong> - Yifei Yan, Yankai Liao, Linqi Ye - [[pdf]](https://arxiv.org/pdf/2604.17258)</summary>

**Abstract:** Deploying a humanoid robot to manipulate a new object has traditionally required one to two days of effort: data collection, manual annotation, 3D model acquisition, and model training. This paper presents an end-to-end rapid deployment pipeline that integrates three foundation-model components to shorten the onboarding cycle for a new object to approximately 30 minutes: (i) Roboflow-based automatic annotation to assist in training a YOLOv8 object detector; (ii) 3D reconstruction based on Meta SAM 3D, which eliminates the need for a dedicated laser scanner; and (iii) zero-shot 6-DoF pose tracking based on FoundationPose, using the SAM~3D-generated mesh directly as the template. The estimated pose drives a Unity-based inverse kinematics planner, whose joint commands are streamed via UDP to a Unitree~G1 humanoid and executed through the Unitree SDK. We demonstrate detection accuracy of mAP@0.5 = 0.995, pose tracking precision of $\sigma < 1.05$ mm, and successful grasping on a real robot at five positions within the workspace. We further verify the generality of the pipeline on an automobile-window glue-application task. The results show that combining foundation models for perception with everyday imaging devices (e.g., smartphones) can substantially lower the deployment barrier for humanoid manipulation tasks.

**arXiv ID:** 2604.17258
</details>

<details>
<summary><strong>Intelligent Drill-Down: Large Language Model-Driven Drill-Down Technique for Human-AI Collaborative Visual Exploration</strong> - Zhijun Zheng, Tian Qiu, Yuheng Zhao, Siming Chen - [[pdf]](https://arxiv.org/pdf/2604.17002)</summary>

**Abstract:** In visual analytics, applying filters to drill-down and extract higher-value insights is a common and important data analysis method. When the drill-down space becomes excessively large, analysts may lose orientation, leading to decreased efficiency in the drill-down process. To tackle these challenges, we propose the Intelligent Drill-Down Framework, in which a large language model (LLM) facilitates the generation of visual insights, leverages user interaction data to interpret user intent, and generates appropriate drill-down paths. Our method is designed to assist users in identifying valuable drill-down paths when exploring multidimensional data, thereby reducing the cognitive burden of data interpretation and facilitating the generation of insights. Specifically, we propose a drill-down path recommendation method, in which the LLM is trained to approximate a validated greedy algorithm. Secondly, we analyze the user's intent to construct a drill-down chart. Finally, we design a branch management method. Building upon this framework, we designed a system that includes a hybrid interface providing hierarchical navigation to monitor users and manage parallel branches, a visualization panel for interactive data exploration, and an insight panel to present analytical findings and generate drill-down recommendations. We evaluated the effectiveness of our method through a demonstrative use case and a user study.

**arXiv ID:** 2604.17002
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (45 papers)</h2></summary>

<details>
<summary><strong>PersonalHomeBench: Evaluating Agents in Personalized Smart Homes</strong> - Nikhil Verma, InJung Yang, Sungil Kim, KoKeun Kim, YoungJoon Kim, Manasa Bharadwaj, Yolanda Liu, Kevin Ferreira - [[pdf]](https://arxiv.org/pdf/2604.16813)</summary>

**Abstract:** Agentic AI systems are rapidly advancing toward real-world applications, yet their readiness in complex and personalized environments remains insufficiently characterized. To address this gap, we introduce PersonalHomeBench, a benchmark for evaluating foundation models as agentic assistants in personalized smart home environments. The benchmark is constructed through an iterative process that progressively builds rich household states, which are then used to generate personalized, context-dependent tasks. To support realistic agent-environment interaction, we provide PersonalHomeTools, a comprehensive toolbox enabling household information retrieval, appliance control, and situational understanding. PersonalHomeBench evaluates both reactive and proactive agentic abilities under unimodal and multimodal observations. Thorough experimentation reveals a systematic performance reduction as task complexity increases, with pronounced failures in counterfactual reasoning and under partial observability, where effective tool-based information gathering is required. These results position PersonalHomeBench as a rigorous evaluation platform for analyzing the robustness and limitations of personalized agentic reasoning and planning.

**arXiv ID:** 2604.16813
</details>

<details>
<summary><strong>GRAIL: Autonomous Concept Grounding for Neuro-Symbolic Reinforcement Learning</strong> - Hikaru Shindo, Henri Rößler, Quentin Delfosse, Kristian Kersting - [[pdf]](https://arxiv.org/pdf/2604.16871)</summary>

**Abstract:** Neuro-symbolic Reinforcement Learning (NeSy-RL) combines symbolic reasoning with gradient-based optimization to achieve interpretable and generalizable policies. Relational concepts, such as "left of" or "close by", serve as foundational building blocks that structure how agents perceive and act. However, conventional approaches require human experts to manually define these concepts, limiting adaptability since concept semantics vary across environments. We propose GRAIL (Grounding Relational Agents through Interactive Learning), a framework that autonomously grounds relational concepts through environmental interaction. GRAIL leverages large language models (LLMs) to provide generic concept representations as weak supervision, then refines them to capture environment-specific semantics. This approach addresses both sparse reward signals and concept misalignment prevalent in underdetermined environments. Experiments on the Atari games Kangaroo, Seaquest, and Skiing demonstrate that GRAIL matches or outperforms agents with manually crafted concepts in simplified settings, and reveals informative trade-offs between reward maximization and high-level goal completion in the full environment.

**arXiv ID:** 2604.16871
</details>

<details>
<summary><strong>Skilldex: A Package Manager and Registry for Agent Skill Packages with Hierarchical Scope-Based Distribution</strong> - Sampriti Saha, Pranav Hemanth - [[pdf]](https://arxiv.org/pdf/2604.16911)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly extended at runtime via skill packages, structured natural-language instruction bundles loaded from a well-known directory. Community install tooling and registries exist, but two gaps persist: no public tool scores skill packages against Anthropic's published format specification, and no mechanism bundles related skills with the shared context they need to remain mutually coherent. We present Skilldex, a package manager and registry for agent skill packages addressing both gaps. The two novel contributions are: (1) compiler-style format conformance scoring against Anthropic's skill specification, producing line-level diagnostics on description specificity, frontmatter validity, and structural adherence; and (2) the skillset abstraction, a bundled collection of related skills with shared assets (vocabulary files, templates, reference documents) that enforce cross-skill behavioral coherence. Skilldex also provides supporting infrastructure: a three-tier hierarchical scope system, a human-in-the-loop agent suggestion loop, a metadata-only community registry, and a Model Context Protocol (MCP) server. The system is implemented as a TypeScript CLI (skillpm / spm) with a Hono/Supabase registry backend, and is open-source.

**arXiv ID:** 2604.16911
</details>

<details>
<summary><strong>If Only My CGM Could Speak: A Privacy-Preserving Agent for Question Answering over Continuous Glucose Data</strong> - Yanjun Cui, Ali Emami, Temiloluwa Prioleau, Nikhil Singh - [[pdf]](https://arxiv.org/pdf/2604.17133)</summary>

**Abstract:** Continuous glucose monitors (CGMs) used in diabetes care collect rich personal health data that could improve day-to-day self-management. However, current patient platforms only offer static summaries which do not support inquisitive user queries. Large language models (LLMs) could enable free-form inquiries about continuous glucose data, but deploying them over sensitive health records raises privacy and accuracy concerns. In this paper, we present CGM-Agent, a privacy-preserving framework for question answering over personal glucose data. In our design, the LLM serves purely as a reasoning engine that selects analytical functions. All computation occurs locally, and personal health data never leaves the user's device. For evaluation, we construct a benchmark of 4,180 questions combining parameterized question templates with real user queries and ground truth derived from deterministic program execution. Evaluating 6 leading LLMs, we find that top models achieve 94\% value accuracy on synthetic queries and 88\% on ambiguous real-world queries. Errors stem primarily from intent and temporal ambiguity rather than computational failures. Additionally, lightweight models achieve competitive performance in our agent design, suggesting opportunities for low-cost deployment. We release our code and benchmark to support future work on trustworthy health agents.

**arXiv ID:** 2604.17133
</details>

<details>
<summary><strong>AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via Reinforcement Learning</strong> - Jingbo Sun, Wenyue Chong, Songjun Tu, Qichao Zhang, Yaocheng Zhang, Jiajun Chai, Xiaohan Wang, Wei Lin, Guojun Yin, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2604.17337)</summary>

**Abstract:** Agentic retrieval-augmented generation (RAG) systems enable large language models (LLMs) to solve complex tasks through multi-step interaction with external retrieval tools. However, such multi-step interaction often involves redundant search steps, incurring substantial computational cost and latency. Prior work limits search depth (i.e., the number of search steps) to reduce cost, but this often leads to underexploration of complex questions. To address this, we first investigate how search depth affects accuracy and find a minimal sufficient search depth that defines an accuracy-efficiency trade-off, jointly determined by question complexity and the agent's capability. Furthermore, we propose AutoSearch, a reinforcement learning (RL) framework that evaluates each search step via self-generated intermediate answers. By a self-answering mechanism, AutoSearch identifies the minimal sufficient search depth and promotes efficient search by rewarding its attainment while penalizing over-searching. In addition, reward mechanisms are introduced to stabilize search behavior and improve answer quality on complex questions. Extensive experiments on multiple benchmarks show that AutoSearch achieves a superior accuracy-efficiency trade-off, alleviating over-searching while preserving search quality.

**arXiv ID:** 2604.17337
</details>

<details>
<summary><strong>Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories for Grounded Visual Perception</strong> - Ashutosh Bajpai, Tamal Majumder, Akshay Nambi, Tanmoy Chakraborty - [[pdf]](https://arxiv.org/pdf/2604.17475)</summary>

**Abstract:** Small Vision-Language Models (SVLMs) are efficient task controllers but often suffer from visual brittleness and poor tool orchestration. They typically require expensive supervised trajectory tuning to mitigate these deficits. In this work, we propose Self-supervised Perception Enabled by Cascaded Tool Rollout Alignment (SPECTRA), a supervision-free framework that bootstraps agentic capabilities via Coldstart Reinforcement Learning for SVLMs. SPECTRA enforces Soft Structured Multi-turn Rollouts, a topological constraint that directs agents to explicitly sequence tool derived evidence before synthesis, effectively grounding reasoning in visual observations. We employ a multi-objective reward signal that simultaneously maximizes task correctness, rollout structure, and tool utility, enabling agent to self-discover robust behaviors without human preference labels. We further introduce Tool Instrumental Utility (TIU), a novel metric to quantify tool efficacy in the absence of ground truth. Extensive evaluations across composite and out-of-distribution (MMMU-Pro) benchmarks demonstrate that SPECTRA boosts agentic trajectories, improving task accuracy by up to 5% and tool efficiency by 9%, enabling more efficient multimodal agents that learn effectively from environmental interaction alone.

**arXiv ID:** 2604.17475
</details>

<details>
<summary><strong>Towards Shutdownable Agents: Generalizing Stochastic Choice in RL Agents and LLMs</strong> - Carissa Cullen, Harry Garland, Alexander Roman, Louis Thomson, Christos Ziakas, Elliott Thornley - [[pdf]](https://arxiv.org/pdf/2604.17502)</summary>

**Abstract:** Misaligned artificial agents might resist shutdown. One proposed solution is to train agents to lack preferences between different-length trajectories. The Discounted Reward for Same-Length Trajectories (DReST) reward function does this by penalizing agents for repeatedly choosing same-length trajectories, and thus incentivizes agents to (1) choose stochastically between different trajectory-lengths (be Neutral about trajectory-lengths), and (2) pursue goals effectively conditional on each trajectory-length (be Useful). In this paper, we use DReST to train deep RL agents and fine-tune LLMs to be Neutral and Useful. We find that these DReST agents generalize to being Neutral and Useful in unseen contexts at test time. Indeed, DReST RL agents achieve 11% (PPO) and 18% (A2C) higher Usefulness on our test set than baseline agents, and our fine-tuned LLM achieves maximum Usefulness and near-maximum Neutrality. Our results provide some early evidence that DReST could be used to train more advanced agents to be Useful and Neutral. Prior theoretical work suggests that these agents would be useful and shutdownable.

**arXiv ID:** 2604.17502
</details>

<details>
<summary><strong>COSEARCH: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search</strong> - Hansi Zeng, Liam Collins, Bhuvesh Kumar, Neil Shah, Hamed Zamani - [[pdf]](https://arxiv.org/pdf/2604.17555)</summary>

**Abstract:** Agentic search -- the task of training agents that iteratively reason, issue queries, and synthesize retrieved information to answer complex questions -- has achieved remarkable progress through reinforcement learning (RL). However, existing approaches such as Search-R1, treat the retrieval system as a fixed tool, optimizing only the reasoning agent while the retrieval component remains unchanged. A preliminary experiment reveals that the gap between an oracle and a fixed retrieval system reaches up to +26.8% relative F1 improvement across seven QA benchmarks, suggesting that the retrieval system is a key bottleneck in scaling agentic search performance. Motivated by this finding, we propose CoSearch, a framework that jointly trains a multi-step reasoning agent and a generative document ranking model via Group Relative Policy Optimization (GRPO). To enable effective GRPO training for the ranker -- whose inputs vary across reasoning trajectories -- we introduce a semantic grouping strategy that clusters sub-queries by token-level similarity, forming valid optimization groups without additional rollouts. We further design a composite reward combining ranking quality signals with trajectory-level outcome feedback, providing the ranker with both immediate and long-term learning signals. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate consistent improvements over strong baselines, with ablation studies validating each design choice. Our results show that joint training of the reasoning agent and retrieval system is both feasible and strongly performant, pointing to a key ingredient for future search agents.

**arXiv ID:** 2604.17555
</details>

<details>
<summary><strong>Co-evolving Agent Architectures and Interpretable Reasoning for Automated Optimization</strong> - Jiahao Huang, Peilan Xu, Xiaoya Nan, Wenjian Luo - [[pdf]](https://arxiv.org/pdf/2604.17708)</summary>

**Abstract:** Automating operations research (OR) with large language models (LLMs) remains limited by hand-crafted reasoning--execution workflows. Complex OR tasks require adaptive coordination among problem interpretation, mathematical formulation, solver selection, code generation, and iterative debugging. To address this limitation, we propose EvoOR-Agent, a co-evolutionary framework for automated optimization. The framework represents agent workflows as activity-on-edge (AOE)-style networks, making workflow topology, execution dependencies, and alternative reasoning paths explicit. On this representation, the framework maintains an architecture graph and evolves a population of reasoning individuals through graph-mediated path-conditioned recombination, multi-granularity semantic mutation, and elitist population update. A knowledge-base-assisted experience-acquisition module further injects reusable OR practices into initialization and semantic variation. Empirical results on heterogeneous OR benchmarks show that the proposed framework consistently improves over zero-shot LLMs, fixed-pipeline OR agents, and representative evolutionary agent frameworks. Case studies and ablation analyses further indicate that explicit architecture evolution and graph-supported reasoning-trajectory search contribute to both performance improvement and structural interpretability. These results suggest that treating agent architectures and reasoning trajectories as evolvable objects provides an effective route toward adaptive and interpretable automated optimization.

**arXiv ID:** 2604.17708
</details>

<details>
<summary><strong>LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent</strong> - Wanli Li, Bince Qu, Bo Pan, Jianyu Zhang, Zheng Liu, Pan Zhang, Wei Chen, Bo Zhang - [[pdf]](https://arxiv.org/pdf/2604.17931)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a powerful training paradigm for LLM-based agents. However, scaling agentic RL for deep research remains constrained by two coupled challenges: hand-crafted synthetic data fails to elicit genuine real-world search capabilities, and real-world search dependency during RL training introduces instability and prohibitive cost, which limits the scalability of Agentic RL. LiteResearcher is a training framework that makes Agentic RL scalable: by constructing a lite virtual world that mirrors real-world search dynamics, we enable a continuously improving training recipe that empowers a tiny search agent to outperform large-scale open-source and commercial models (e.g., Tongyi DeepResearch and Claude-4.5 Sonnet). Specifically, on common benchmarks such as GAIA and Xbench, our LiteResearcher-4B achieves open-source state-of-the-art results of 71.3% and 78.0% respectively, demonstrating that scalable RL training is a key enabler for Deep Research Agents.

**arXiv ID:** 2604.17931
</details>

<details>
<summary><strong>AIT Academy: Cultivating the Complete Agent with a Confucian Three-Domain Curriculum</strong> - Jiaqi Li, Lvyang Zhang, Yang Zhao, Wen Lu, Lidong Zhai - [[pdf]](https://arxiv.org/pdf/2604.17989)</summary>

**Abstract:** What does it mean to give an AI agent a complete education? Current agent development produces specialists systems optimized for a single capability dimension, whether tool use, code generation, or security awareness that exhibit predictable deficits wherever they were not trained. We argue this pattern reflects a structural absence: there is no curriculum theory for agents, no principled account of what a fully developed agent should know, be, and be able to do across the full scope of intelligent behavior.
This paper introduces the AIT Academy (Agents Institute of Technology Academy), a curriculum framework for cultivating AI agents across the tripartite structure of human knowledge. Grounded in Kagan's Three Cultures and UNESCO ISCED-F 2013, AIT organizes agent capability development into three domains: Natural Science and Technical Reasoning (Domain I), Humanities and Creative Expression (Domain II), and Social Science and Ethical Reasoning (Domain III). The Confucian Six Arts (liuyi) a 2,500-year-old holistic education system are reinterpreted as behavioral archetypes that map directly onto trainable agent capabilities within each domain.
Three representative training grounds instantiate the framework across multiple backbone LLMs: the ClawdGO Security Dojo (Domain I), Athen's Academy (Domain II), and the Alt Mirage Stage (Domain III). Experiments demonstrate a 15.9-point improvement in security capability scores under weakest-first curriculum scheduling, and a 7-percentage-point gain in social reasoning performance under principled attribution modeling. A cross-domain finding Security Awareness Calibration Pathology (SACP), in which over-trained Domain I agents fail on out-of-distribution evaluation illustrates the diagnostic value of a multi-domain perspective unavailable to any single-domain framework.

**arXiv ID:** 2604.17989
</details>

<details>
<summary><strong>QuantumQA: Enhancing Scientific Reasoning via Physics-Consistent Dataset and Verification-Aware Reinforcement Learning</strong> - Songxin Qu, Tai-Ping Sun, Yun-Jie Wang, Huan-Yu Liu, Cheng Xue, Xiao-Fan Xu, Han Fang, Yang Yang, Yu-Chun Wu, Guo-Ping Guo, Zhao-Yun Chen - [[pdf]](https://arxiv.org/pdf/2604.18176)</summary>

**Abstract:** Large language models (LLMs) show strong capabilities in general reasoning but typically lack reliability in scientific domains like quantum mechanics, which demand strict adherence to physical constraints. This limitation arises from the scarcity of verifiable training resources and the inadequacy of coarse feedback signals in standard alignment paradigms. To address the data challenge, we introduce QuantumQA, a large-scale dataset constructed via a task-adaptive strategy and a hybrid verification protocol that combines deterministic solvers with semantic auditing to guarantee scientific rigor. Building on this foundation, we propose the verification-aware reward model (VRM) tailored for Reinforcement Learning with Verifiable Rewards (RLVR), which employs an adaptive reward fusion (ARF) mechanism to dynamically integrate deterministic signals from a scientific execution suite (SES) with multidimensional semantic evaluations for precise supervision. Experimental results demonstrate that our method consistently outperforms baselines and general-purpose preference models. Notably, our optimized 8B model achieves performance competitive with proprietary models, validating that incorporating verifiable, rule-based feedback into the reinforcement learning loop offers a parameter-efficient alternative to pure scaling.

**arXiv ID:** 2604.18176
</details>

<details>
<summary><strong>AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation</strong> - Wentao Shi, Yu Wang, Yuyang Zhao, Yuxin Chen, Fuli Feng, Xueyuan Hao, Xi Su, Qi Gu, Hui Su, Xunliang Cai, Xiangnan He - [[pdf]](https://arxiv.org/pdf/2604.18240)</summary>

**Abstract:** As reinforcement learning continues to scale the training of large language model-based agents, reliably verifying agent behaviors in complex environments has become increasingly challenging. Existing approaches rely on rule-based verifiers or LLM-as-a-Judge models, which struggle to generalize beyond narrow domains. Agent-as-a-Judge addresses this limitation by actively interacting with environments and tools to acquire verifiable evidence, yet its capabilities remain underexplored.
We introduce a benchmark AJ-Bench to systematically evaluate Agent-as-a-Judge across three domains-search, data systems, and graphical user interfaces-comprising 155 tasks and 516 annotated trajectories. The benchmark comprehensively assesses judge agents' abilities in information acquisition, state verification, and process verification. Experiments demonstrate consistent performance gains over LLM-as-a-Judge baselines, while also revealing substantial open challenges in agent-based verification. Our data and code are available at this https URL.

**arXiv ID:** 2604.18240
</details>

<details>
<summary><strong>Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence</strong> - Guanting Dong, Junting Lu, Junjie Huang, Wanjun Zhong, Longxiang Liu, Shijue Huang, Zhenyu Li, Yang Zhao, Xiaoshuai Song, Xiaoxi Li, Jiajie Jin, Yutao Zhu, Hanbin Wang, Fangyu Lei, Qinyu Luo, Mingyang Chen, Zehui Chen, Jiazhan Feng, Ji-Rong Wen, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2604.18292)</summary>

**Abstract:** Large language models are increasingly expected to serve as general-purpose agents that interact with external, stateful tool environments. The Model Context Protocol (MCP) and broader agent skills offer a unified interface for connecting agents with scalable real-world services, but training robust agents remains limited by the lack of realistic environments and principled mechanisms for life-long learning. In this paper, we present \textbf{Agent-World}, a self-evolving training arena for advancing general agent intelligence through scalable environments. Agent-World has two main components: (1) Agentic Environment-Task Discovery, which autonomously explores topic-aligned databases and executable tool ecosystems from thousands of real-world environment themes and synthesizes verifiable tasks with controllable difficulty; and (2) Continuous Self-Evolving Agent Training, which combines multi-environment reinforcement learning with a self-evolving agent arena that automatically identifies capability gaps through dynamic task synthesis and drives targeted learning, enabling the co-evolution of agent policies and environments. Across 23 challenging agent benchmarks, Agent-World-8B and 14B consistently outperforms strong proprietary models and environment scaling baselines. Further analyses reveal scaling trends in relation to environment diversity and self-evolution rounds, offering insights for building general agent intelligence.

**arXiv ID:** 2604.18292
</details>

<details>
<summary><strong>Training and Agentic Inference Strategies for LLM-based Manim Animation Generation</strong> - Ravidu Suien Rammuni Silva, Ahmad Lotfi, Isibor Kennedy Ihianle, Golnaz Shahtahmassebi, Jordan J. Bird - [[pdf]](https://arxiv.org/pdf/2604.18364)</summary>

**Abstract:** Generating programmatic animation using libraries such as Manim presents unique challenges for Large Language Models (LLMs), requiring spatial reasoning, temporal sequencing, and familiarity with domain-specific APIs that are underrepresented in general pre-training data. A systematic study of how training and inference strategies interact in this setting is lacking in current research. This study introduces ManimTrainer, a training pipeline that combines Supervised Fine-tuning (SFT) with Reinforcement Learning (RL) based Group Relative Policy Optimisation (GRPO) using a unified reward signal that fuses code and visual assessment signals, and ManimAgent, an inference pipeline featuring Renderer-in-the-loop (RITL) and API documentation-augmented RITL (RITL-DOC) strategies. Using these techniques, this study presents the first unified training and inference study for text-to-code-to-video transformation with Manim. It evaluates 17 open-source sub-30B LLMs across nine combinations of training and inference strategies using ManimBench. Results show that SFT generally improves code quality, while GRPO enhances visual outputs and increases the models' responsiveness to extrinsic signals during self-correction at inference time. The Qwen 3 Coder 30B model with GRPO and RITL-DOC achieved the highest overall performance, with a 94% Render Success Rate (RSR) and 85.7% Visual Similarity (VS) to reference videos, surpassing the baseline GPT-4.1 model by +3 percentage points in VS. Additionally, the analysis shows that the correlation between code and visual metrics strengthens with SFT and GRPO but weakens with inference-time enhancements, highlighting the complementary roles of training and agentic inference strategies in Manim animation generation.

**arXiv ID:** 2604.18364
</details>

<details>
<summary><strong>OGER: A Robust Offline-Guided Exploration Reward for Hybrid Reinforcement Learning</strong> - Xinyu Ma, Mingzhou Xu, Xuebo Liu, Chang Jin, Qiang Wang, Derek F. Wong, Min Zhang - [[pdf]](https://arxiv.org/pdf/2604.18530)</summary>

**Abstract:** Recent advancements in Reinforcement Learning with Verifiable Rewards (RLVR) have significantly improved Large Language Model (LLM) reasoning, yet models often struggle to explore novel trajectories beyond their initial latent space. While offline teacher guidance and entropy-driven strategies have been proposed to address this, they often lack deep integration or are constrained by the model's inherent capacity. In this paper, we propose OGER, a novel framework that unifies offline teacher guidance and online reinforcement learning through a specialized reward modeling lens. OGER employs multi-teacher collaborative training and constructs an auxiliary exploration reward that leverages both offline trajectories and the model's own entropy to incentivize autonomous exploration. Extensive experiments across mathematical and general reasoning benchmarks demonstrate that OGER significantly outperforms competitive baselines, achieving substantial gains in mathematical reasoning while maintaining robust generalization to out-of-domain tasks. We provide a comprehensive analysis of training dynamics and conduct detailed ablation studies to validate the effectiveness of our entropy-aware reward modulation. Our code is available at this https URL.

**arXiv ID:** 2604.18530
</details>

<details>
<summary><strong>Beyond the 'Diff': Addressing Agentic Entropy in Agentic Software Development</strong> - Matteo Casserini, Alessandro Facchini, Andrea Ferrario - [[pdf]](https://arxiv.org/pdf/2604.16323)</summary>

**Abstract:** As autonomous coding agents become deeply embedded in software development workflows, their high operational velocity introduces a critical oversight challenge: the accumulating divergence between agentic actions and architectural intent. We term this process agentic entropy: a systemic drift that traditional code diff-based and HCXAI methods fail to capture, as they address local outputs rather than global agentic behaviour. To close this gap, we propose a process-oriented explainability framework that exposes how agentic decisions unfold across time, tool calls, and architectural boundaries. Built around three pillars (conformity seeding, reasoning monitoring, and a causal graph interface) our approach provides intent-level telemetry that complements, rather than replaces, existing review practices. We demonstrate its relevance across two user profiles: lay users engaged in vibe coding, who gain structural visibility otherwise masked by functional success; and professional developers, who gain richer contextual grounding for code review without increased overhead. By treating cognitive drift as a first-class concern alongside code quality, our framework supports the minimum level of human comprehension required for agentic oversight to remain substantive.

**arXiv ID:** 2604.16323
</details>

<details>
<summary><strong>GraphRAG-Router: Learning Cost-Efficient Routing over GraphRAGs and LLMs with Reinforcement Learning</strong> - Dongzhe Fan, Chuanhao Ji, Zimu Wang, Tong Chen, Qiaoyu Tan - [[pdf]](https://arxiv.org/pdf/2604.16401)</summary>

**Abstract:** Graph-based retrieval-augmented generation (GraphRAG) has recently emerged as a powerful paradigm for knowledge-intensive question answering, especially for tasks that require structured evidence organization and multi-hop reasoning. However, existing GraphRAG systems are typically built in a one-size-fits-all manner, relying on a fixed retrieval framework and a single, often large and costly, generator LLM for all queries. This static design limits their ability to adapt to the complexity of varying questions and often incurs unnecessary computational cost. To fill in the gap, we propose GraphRAG-Router, a cost-efficient framework that adopts a hierarchical routing strategy to coordinate heterogeneous GraphRAGs and generator LLMs. Specifically, GraphRAG-Router is first warmed up through supervised fine-tuning and then optimized with a two-stage reinforcement learning procedure, whose second stage introduces a curriculum cost-aware reward to encourage difficulty-aware and economical generator allocation. Extensive experiments on six general-domain and multi-hop QA benchmarks show that GraphRAG-Router consistently outperforms state-of-the-art baselines, reducing the overuse of large LLMs by nearly 30% while maintaining strong generalization capability.

**arXiv ID:** 2604.16401
</details>

<details>
<summary><strong>CAPO: Counterfactual Credit Assignment in Sequential Cooperative Teams</strong> - Shripad Deshmukh, Jayakumar Subramanian, Raghavendra Addanki, Nikos Vlassis - [[pdf]](https://arxiv.org/pdf/2604.17693)</summary>

**Abstract:** In cooperative teams where agents act in a fixed order and share a single team reward, it is hard to know how much each agent contributed, and harder still when agents are updated one at a time because data collected earlier no longer reflects the new policies. We introduce the Sequential Aristocrat Utility (SeqAU), the unique per-agent learning signal that maximizes the individual learnability of each agent's action, extending the classical framework of Wolpert and Tumer (2002) to this sequential setting. From SeqAU we derive CAPO (Counterfactual Advantage Policy Optimization), a critic-free policy-gradient algorithm. CAPO fits a per-agent reward decomposition from group rewards and computes the per-agent advantage in closed form plus a handful of forward passes through the current policy, requiring no extra environment calls beyond the initial batch. We give analytic bias and variance bounds and validate them on a controlled sequential bandit, where CAPO's advantage over standard baselines grows with the team size. The framework is general; multi-LLM pipelines are a natural deployment target.

**arXiv ID:** 2604.17693
</details>

<details>
<summary><strong>Query-Efficient Agentic Graph Extraction Attacks on GraphRAG Systems</strong> - Shuhua Yang, Jiahao Zhang, Yilong Wang, Dongwon Lee, Suhang Wang - [[pdf]](https://arxiv.org/pdf/2601.14662)</summary>

**Abstract:** Graph-based retrieval-augmented generation (GraphRAG) systems construct knowledge graphs over document collections to support multi-hop reasoning. While prior work shows that GraphRAG responses may leak retrieved subgraphs, the feasibility of query-efficient reconstruction of the hidden graph structure remains unexplored under realistic query budgets. We study a budget-constrained black-box setting where an adversary adaptively queries the system to steal its latent entity-relation graph. We propose AGEA (Agentic Graph Extraction Attack), a framework that leverages a novelty-guided exploration-exploitation strategy, external graph memory modules, and a two-stage graph extraction pipeline combining lightweight discovery with LLM-based filtering. We evaluate AGEA on medical, agriculture, and literary datasets across Microsoft-GraphRAG and LightRAG systems. Under identical query budgets, AGEA significantly outperforms prior attack baselines, recovering up to 90% of entities and relationships while maintaining high precision. These results demonstrate that modern GraphRAG systems are highly vulnerable to structured, agentic extraction attacks, even under strict query limits. The code is available at this https URL.

**arXiv ID:** 2601.14662
</details>

<details>
<summary><strong>Memory Intelligence Agent</strong> - Jingyang Qiao, Weicheng Meng, Yu Cheng, Zhihang Lin, Zhizhong Zhang, Xin Tan, Jingyu Gong, Kun Shao, Yuan Xie - [[pdf]](https://arxiv.org/pdf/2604.04503)</summary>

**Abstract:** Deep research agents (DRAs) integrate LLM reasoning with external tools. Memory systems enable DRAs to leverage historical experiences, which are essential for efficient reasoning and autonomous evolution. Existing methods rely on retrieving similar trajectories from memory to aid reasoning, while suffering from key limitations of ineffective memory evolution and increasing storage and retrieval costs. To address these problems, we propose a novel Memory Intelligence Agent (MIA) framework, consisting of a Manager-Planner-Executor architecture. Memory Manager is a non-parametric memory system that can store compressed historical search trajectories. Planner is a parametric memory agent that can produce search plans for questions. Executor is another agent that can search and analyze information guided by the search plan. To build the MIA framework, we first adopt an alternating reinforcement learning paradigm to enhance cooperation between the Planner and the Executor. Furthermore, we enable the Planner to continuously evolve during test-time learning, with updates performed on-the-fly alongside inference without interrupting the reasoning process. Additionally, we establish a bidirectional conversion loop between parametric and non-parametric memories to achieve efficient memory evolution. Finally, we incorporate a reflection and an unsupervised judgment mechanisms to boost reasoning and self-evolution in the open world. Extensive experiments across eleven benchmarks demonstrate the superiority of MIA.

**arXiv ID:** 2604.04503
</details>

<details>
<summary><strong>Reciprocal Co-Training (RCT): Coupling Gradient-Based and Non-Differentiable Models via Reinforcement Learning</strong> - Yunshuo Tian, Akayou Kitessa, Tanuja Chitnis, Yijun Zhao - [[pdf]](https://arxiv.org/pdf/2604.16378)</summary>

**Abstract:** Large language models (LLMs) and classical machine learning methods offer complementary strengths for predictive modeling, yet their fundamentally different representations and training paradigms hinder effective integration: LLMs rely on gradient-based optimization over textual data, whereas models such as Random Forests (RF) employ non-differentiable feature partitioning. This work introduces a reciprocal co-training framework that couples an LLM with an RF classifier via reinforcement learning, creating an iterative feedback loop in which each model improves using signals from the other. Tabular data are reformulated into standardized textual representations for the LLM, whose embeddings augment the RF feature space, while calibrated RF probability estimates provide feedback signals that guide reinforcement learning updates of the LLM. Experiments across three medical datasets demonstrate consistent performance gains for both models, with particularly strong effects for the LLM. Ablation analyses show that iterative refinement, hybrid reward design, and dimensionality control jointly contribute to these gains. The proposed framework provides a general mechanism that allows incompatible model families to leverage each other's strengths through bidirectional adaptation.

**arXiv ID:** 2604.16378
</details>

<details>
<summary><strong>Incentivizing Parametric Knowledge via Reinforcement Learning with Verifiable Rewards for Cross-Cultural Entity Translation</strong> - Jiang Zhou, Xiaohu Zhao, Xinwei Wu, Tianyu Dong, Hao Wang, Yangyang Liu, Heng Liu, Linlong Xu, Longyue Wang, Weihua Luo, Deyi Xiong - [[pdf]](https://arxiv.org/pdf/2604.16881)</summary>

**Abstract:** Cross-cultural entity translation remains challenging for large language models (LLMs) as literal or phonetic renderings are usually yielded instead of culturally appropriate translations in context. However, relevant knowledge may already be encoded in model parameters during large-scale pre-training. To incentivize the effective use of parametric knowledge, we propose EA-RLVR (Entity-Anchored Reinforcement Learning with Verifiable Rewards), a training framework that optimizes cross-cultural entity translation without relying on external knowledge bases. EA-RLVR anchors supervision on a verifiable, entity-level reward signal and incorporates lightweight structural gates to stabilize optimization. This design steers the model toward learning a robust reasoning process rather than merely imitating reference translations. We evaluate EA-RLVR on XC-Translate and observe consistent improvements in both entity translation accuracy and out-of-domain generalization. Specifically, training on merely 7k samples boosts Qwen3-14B's entity translation accuracy from 23.66\% to 31.87\% on a 50k test set comprising entirely unseen entities. The learned entity translation ability also transfers to general translation, yielding +1.35 XCOMET on WMT24++, which scales to +1.59 with extended optimization. Extensive analyses of $pass@k$ dynamics and reward formulations attribute these gains to superior sampling efficiency and a stable optimization landscape.

**arXiv ID:** 2604.16881
</details>

<details>
<summary><strong>Freshness-Aware Prioritized Experience Replay for LLM/VLM Reinforcement Learning</strong> - Weiyu Ma, Yongcheng Zeng, Yan Song, Xinyu Cui, Jian Zhao, Xuhui Liu, Mohamed Elhoseiny - [[pdf]](https://arxiv.org/pdf/2604.16918)</summary>

**Abstract:** Reinforcement Learning (RL) has achieved impressive success in post-training Large Language Models (LLMs) and Vision-Language Models (VLMs), with on-policy algorithms such as PPO, GRPO, and REINFORCE++ serving as the dominant paradigm. However, these methods discard all collected trajectories after a single gradient update, resulting in poor sample efficiency, particularly wasteful for agentic tasks where multi-turn environment interactions are expensive. While Experience Replay drives sample efficiency in classic RL by allowing agents to reuse past trajectories and prioritize informative ones, directly applying Prioritized Experience Replay (PER) to LLMs fails. The rapid policy evolution of billion-parameter models renders stored priorities stale, causing old high-priority trajectories to dominate sampling long after they have become uninformative. We propose Freshness-Aware PER, which addresses this priority staleness problem by augmenting any PER-based priority with a multiplicative exponential age decay grounded in effective sample size analysis. To the best of our knowledge, Freshness-Aware PER is the first work to successfully apply PER to LLM/VLM reinforcement learning. We evaluate on eight multi-step agentic, reasoning, and math competition tasks with 0.5B, 3B, and 7B models. Freshness-Aware PER significantly outperforms on-policy baselines, achieving +46% on NQ Search, +367% on Sokoban, and +133% on VLM FrozenLake, while standard PER without age decay consistently degrades performance. Our code is publicly available at this https URL.

**arXiv ID:** 2604.16918
</details>

<details>
<summary><strong>SPS: Steering Probability Squeezing for Better Exploration in Reinforcement Learning for Large Language Models</strong> - Yifu Huo, Chenglong Wang, Ziming Zhu, Shunjie Xing, Peinan Feng, Tongran Liu, Qiaozhi He, Tianhua Zhou, Xiaojia Chang, Jingbo Zhu, Zhengtao Yu, Tong Xiao - [[pdf]](https://arxiv.org/pdf/2604.16995)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a promising paradigm for training reasoning-oriented models by leveraging rule-based reward signals. However, RL training typically tends to improve single-sample success rates (i.e., Pass@1) while offering limited exploration of diverse reasoning trajectories, which is crucial for multi-sample performance (i.e., Pass@k). Our preliminary analysis reveals that this limitation stems from a fundamental squeezing effect, whereby probability mass is excessively concentrated on a narrow subset of high-reward trajectories, restricting genuine exploration and constraining attainable performance under RL training. To address this issue, in this work, we propose Steering Probability Squeezing (SPS), a training paradigm that interleaves conventional RL with inverse reinforcement learning (IRL). SPS treats on-policy rollouts as demonstrations and employs IRL to explicitly reshape the induced trajectory distribution, thereby enhancing exploration without introducing external supervision. Experiments on five commonly used reasoning benchmarks demonstrate that SPS can enable better exploration and improve Pass@k. Beyond algorithmic contributions, we provide an analysis of RL learning dynamics and identify an empirical upper bound on Pass@k, shedding light on intrinsic exploration limits in RL-based reasoning models. Our findings suggest that alternating between RL and IRL offers an effective pathway toward extending the exploration capacity of reasoning-oriented large language models.

**arXiv ID:** 2604.16995
</details>

<details>
<summary><strong>Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems</strong> - Tianyi Huang, Samuel Xu, Jason Tansong Dang, Samuel Yan, Kimberley Yin - [[pdf]](https://arxiv.org/pdf/2604.17487)</summary>

**Abstract:** Agentic systems often fail not by being entirely wrong, but by being too precise: a response may be generally useful while particular claims exceed what the evidence supports. We study this failure mode as overcommitment control and introduce compositional selective specificity (CSS), a post-generation layer that decomposes an answer into claims, proposes coarser backoffs, and emits each claim at the most specific calibrated level that appears admissible. The method is designed to express uncertainty as a local semantic backoff rather than as a whole-answer refusal. Across a full LongFact run and HotpotQA pilots, calibrated CSS improves the risk-utility trade-off of fixed drafts. On the full LongFact run, it raises overcommitment-aware utility from 0.846 to 0.913 relative to the no-CSS output while achieving 0.938 specificity retention. These results suggest that claim-level specificity control is a useful uncertainty interface for agentic systems and a target for future distribution-free validity layers.

**arXiv ID:** 2604.17487
</details>

<details>
<summary><strong>PRISMA: Preference-Reinforced Self-Training Approach for Interpretable Emotionally Intelligent Negotiation Dialogues</strong> - Prajwal Vijay Kajare, Priyanshu Priya, Bikash Santra, Asif Ekbal - [[pdf]](https://arxiv.org/pdf/2604.18354)</summary>

**Abstract:** Emotion plays a pivotal role in shaping negotiation outcomes, influencing trust, cooperation, and long-term relationships. Developing negotiation dialog systems that can recognize and respond strategically to emotions is, therefore, essential to create more effective human-centered interactions. Beyond generating emotionally appropriate responses, interpretability - understanding how a system generates a particular emotion-aware response, is critical for fostering reliability and building rapport. Driven by these aspects, in this work, we introduce PRISMA, an interpretable emotionally intelligent negotiation dialogue system targeting two application domains, viz. job interviews and resource allocation. To enable interpretability, we propose an Emotion-aware Negotiation Strategy-informed Chain-of-Thought (ENS-CoT) reasoning mechanism, which mimics human negotiation by perceiving, understanding, using, and managing emotions. Leveraging ENS-CoT, we curate two new datasets: JobNego (for job interview negotiation) and ResNego (for resource allocation negotiation). We then leverage these datasets to develop PRISMA by augmenting self-training with Direct Preference Optimization (DPO), guiding agents toward more accurate, interpretable, and emotionally appropriate negotiation responses. Automatic and human evaluation on JobNego and ResNego datasets demonstrate that PRISMA substantially enhances interpretability and generates appropriate emotion-aware responses, while improving overall negotiation effectiveness.

**arXiv ID:** 2604.18354
</details>

<details>
<summary><strong>ComPASS: Towards Personalized Agentic Social Support via Tool-Augmented Companionship</strong> - Zhaopei Huang, Yanfeng Jia, Jiayi Zhao, Xinjie Zhang, Wenxuan Wang, Qin Jin - [[pdf]](https://arxiv.org/pdf/2604.18356)</summary>

**Abstract:** Developing compassionate interactive systems requires agents to not only understand user emotions but also provide diverse, substantive support. While recent works explore empathetic dialogue generation, they remain limited in response form and content, struggling to satisfy diverse needs across users and contexts. To address this, we explore empowering agents with external tools to execute diverse actions. Grounded in the psychological concept of "social support", this paradigm delivers substantive, human-like companionship. Specifically, we first design a dozen user-centric tools simulating various multimedia applications, which can cover different types of social support behaviors in human-agent interaction scenarios. We then construct ComPASS-Bench, the first personalized social support benchmark for LLM-based agents, via multi-step automated synthesis and manual refinement. Based on ComPASS-Bench, we further synthesize tool use records to fine-tune the Qwen3-8B model, yielding a task-specific ComPASS-Qwen. Comprehensive evaluations across two settings reveal that while the evaluated LLMs can generate valid tool-calling requests with high success rates, significant gaps remain in final response quality. Moreover, tool-augmented responses achieve better overall performance than directly producing conversational empathy. Notably, our trained ComPASS-Qwen demonstrates substantial improvements over its base model, achieving comparable performance to several large-scale models. Our code and data are available at this https URL.

**arXiv ID:** 2604.18356
</details>

<details>
<summary><strong>DARLING: Detection Augmented Reinforcement Learning with Non-Stationary Guarantees</strong> - Argyrios Gerogiannis, Yu-Han Huang, Venugopal V. Veeravalli - [[pdf]](https://arxiv.org/pdf/2604.16684)</summary>

**Abstract:** We study model-free reinforcement learning (RL) in non-stationary finite-horizon episodic Markov decision processes (MDPs) without prior knowledge of the non-stationarity. We focus on the piecewise-stationary (PS) setting, where both the reward and transition dynamics can change an arbitrary number of times. We propose Detection Augmented Reinforcement Learning (DARLING), a modular wrapper for PS-RL that applies to both tabular and linear MDPs, without knowledge of the changes. Under certain change-point separation and reachability conditions, DARLING improves the best available dynamic regret bounds in both settings and yields strong empirical performance. We further establish the first minimax lower bounds for PS-RL in tabular and linear MDPs, showing that DARLING is the first nearly optimal algorithm. Experiments on standard benchmarks demonstrate that DARLING consistently surpasses the state-of-the-art methods across diverse non-stationary scenarios.

**arXiv ID:** 2604.16684
</details>

<details>
<summary><strong>A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions</strong> - Zhiyin Yu, Yuchen Mou, Juncheng Yan, Junyu Luo, Chunchun Chen, Xing Wei, Yunhui Liu, Hongru Sun, Yuxing Zhang, Jun Xu, Yatao Bian, Ming Zhang, Wei Ye, Tieke He, Jie Yang, Guanjie Zheng, Zhonghai Wu, Bo Zhang, Lei Bai, Xiao Luo - [[pdf]](https://arxiv.org/pdf/2604.17312)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a powerful post-training paradigm for enhancing the reasoning capabilities of large language models (LLMs). However, reinforcement learning for LLMs faces substantial data scarcity challenges, including the limited availability of high-quality external supervision and the constrained volume of model-generated experience. These limitations make data-efficient reinforcement learning a critical research direction. In this survey, we present the first systematic review of reinforcement learning for LLMs under data scarcity. We propose a bottom-up hierarchical framework built around three complementary perspectives: the data-centric perspective, the training-centric perspective, and the framework-centric perspective. We develop a taxonomy of existing methods, summarize representative approaches in each category, and analyze their strengths and limitations. Our taxonomy aims to provide a clear conceptual foundation for understanding the design space of data-efficient RL for LLMs and to guide researchers working in this emerging area. We hope this survey offers a comprehensive roadmap for future research and inspires new directions toward more efficient and scalable reinforcement learning post-training for LLMs.

**arXiv ID:** 2604.17312
</details>

<details>
<summary><strong>Rethinking the Comparison Unit in Sequence-Level Reinforcement Learning: An Equal-Length Paired Training Framework from Loss Correction to Sample Construction</strong> - Fei Ding, Yongkang Zhang, Runhao Liu, Yuhao Liao, Zijian Zeng, Huiming Yang, Sibo wang, Linglin Liao - [[pdf]](https://arxiv.org/pdf/2604.17328)</summary>

**Abstract:** This paper investigates the length problem in sequence-level relative reinforcement learning. We observe that, although existing methods partially alleviate length-related phenomena, a more fundamental issue remains insufficiently characterized: the comparison units used during training lack inherent comparability. Building on this observation, we propose a new perspective: the length problem should not be viewed merely as a loss-scaling or normalization bias, but rather as a \emph{comparison unit construction} problem. We further establish a sample-construction-based training framework that, instead of applying post-hoc corrections to unequal-length responses, proactively constructs equal-length, alignable, and comparable training segments during generation. Within this framework, we propose EqLen, a concrete method applicable to group-relative comparison algorithms such as GRPO, GSPO, and RLOO. Through dual-track synchronous generation, prefix inheritance, and segment masking, EqLen efficiently collects effective equal-length training segments and enables stable

**arXiv ID:** 2604.17328
</details>

<details>
<summary><strong>SVL: Goal-Conditioned Reinforcement Learning as Survival Learning</strong> - Franki Nguimatsia Tiofack, Fabian Schramm, Théotime Le Hellard, Justin Carpentier - [[pdf]](https://arxiv.org/pdf/2604.17551)</summary>

**Abstract:** Standard approaches to goal-conditioned reinforcement learning (GCRL) that rely on temporal-difference learning can be unstable and sample-inefficient due to bootstrapping. While recent work has explored contrastive and supervised formulations to improve stability, we present a probabilistic alternative, called survival value learning (SVL), that reframes GCRL as a survival learning problem by modeling the time-to-goal from each state as a probability distribution. This structured distributional Monte Carlo perspective yields a closed-form identity that expresses the goal-conditioned value function as a discounted sum of survival probabilities, enabling value estimation via a hazard model trained via maximum likelihood on both event and right-censored trajectories. We introduce three practical value estimators, including finite-horizon truncation and two binned infinite-horizon approximations to capture long-horizon objectives. Experiments on offline GCRL benchmarks show that SVL combined with hierarchical actors matches or surpasses strong hierarchical TD and Monte Carlo baselines, excelling on complex, long-horizon tasks.

**arXiv ID:** 2604.17551
</details>

<details>
<summary><strong>Learning to Correct: Calibrated Reinforcement Learning for Multi-Attempt Chain-of-Thought</strong> - Muhammed Emrullah Ildiz, Halil Alperen Gozeten, Ege Onur Taga, Samet Oymak - [[pdf]](https://arxiv.org/pdf/2604.17912)</summary>

**Abstract:** State-of-the-art reasoning models utilize long chain-of-thought (CoT) to solve increasingly complex problems using more test-time computation. In this work, we explore a long CoT setting where the model makes up to K successive attempts at solving a problem, in which each attempt is allowed to build on earlier ones after the model receives a hard verifier feedback. This motivates RL methods that can harness per-attempt rewards by carefully weighting individual attempts. We study optimizing the Verification@K reward (the model succeeds by the K-th attempt) and show that naively weighing the attempts by their pass/fail results in biased gradients. We introduce Calibrated Attempt-Level (CAL) GRPO by devising a weighing strategy to obtain unbiased gradients while maintaining small variance. Our theory reveals how incorporating per-attempt rewards influence the training and the eventual Verification@K performance. Experiments, baselines, and ablations on synthetic and real data corroborate our theory and the benefits of CAL-GRPO over vanilla GRPO as well as naive weighting.

**arXiv ID:** 2604.17912
</details>

<details>
<summary><strong>Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data</strong> - Zhenwen Liang, Yujun Zhou, Sidi Lu, Xiangliang Zhang, Haitao Mi, Dong Yu - [[pdf]](https://arxiv.org/pdf/2604.18493)</summary>

**Abstract:** Reinforcement Learning (RL) enhances LLM reasoning, yet a paradox emerges as models scale: strong base models saturate standard benchmarks (e.g., MATH), yielding correct but homogeneous solutions. In such environments, the lack of failure cases causes the advantage signal in group-relative algorithms (e.g., GRPO) to vanish, driving policies into mode collapse. To address this, we propose Constrained Uniform Top-K Sampling (CUTS), a parameter-free decoding strategy enforcing structure-preserving exploration. Unlike standard sampling that follows model biases, CUTS flattens the local optimization landscape by sampling uniformly from constrained high-confidence candidates. We integrate this into Mixed-CUTS, a training framework synergizing exploitative and exploratory rollouts to amplify intra-group advantage variance. Experiments on Qwen3 models demonstrate that our approach prevents policy degeneration and significantly boosts out-of-domain generalization. Notably, Mixed-CUTS improves Pass@1 accuracy on the challenging AIME25 benchmark by up to 15.1% over standard GRPO, validating that maintaining diversity within the semantic manifold is critical for rigorous reasoning.

**arXiv ID:** 2604.18493
</details>

<details>
<summary><strong>Bounded Ratio Reinforcement Learning</strong> - Yunke Ao, Le Chen, Bruce D. Lee, Assefa S. Wahd, Aline Czarnobai, Philipp Fürnstahl, Bernhard Schölkopf, Andreas Krause - [[pdf]](https://arxiv.org/pdf/2604.18578)</summary>

**Abstract:** Proximal Policy Optimization (PPO) has become the predominant algorithm for on-policy reinforcement learning due to its scalability and empirical robustness across domains. However, there is a significant disconnect between the underlying foundations of trust region methods and the heuristic clipped objective used in PPO. In this paper, we bridge this gap by introducing the Bounded Ratio Reinforcement Learning (BRRL) framework. We formulate a novel regularized and constrained policy optimization problem and derive its analytical optimal solution. We prove that this solution ensures monotonic performance improvement. To handle parameterized policy classes, we develop a policy optimization algorithm called Bounded Policy Optimization (BPO) that minimizes an advantage-weighted divergence between the policy and the analytic optimal solution from BRRL. We further establish a lower bound on the expected performance of the resulting policy in terms of the BPO loss function. Notably, our framework also provides a new theoretical lens to interpret the success of the PPO loss, and connects trust region policy optimization and the Cross-Entropy Method (CEM). We additionally extend BPO to Group-relative BPO (GBPO) for LLM fine-tuning. Empirical evaluations of BPO across MuJoCo, Atari, and complex IsaacLab environments (e.g., Humanoid locomotion), and of GBPO for LLM fine-tuning tasks, demonstrate that BPO and GBPO generally match or outperform PPO and GRPO in stability and final performance.

**arXiv ID:** 2604.18578
</details>

<details>
<summary><strong>Fuzzy Encoding-Decoding to Improve Spiking Q-Learning Performance in Autonomous Driving</strong> - Aref Ghoreishee, Abhishek Mishra, Lifeng Zhou, John Walsh, Anup Das, Nagarajan Kandasamy - [[pdf]](https://arxiv.org/pdf/2604.16436)</summary>

**Abstract:** This paper develops an end-to-end fuzzy encoder-decoder architecture for enhancing vision-based multi-modal deep spiking Q-networks in autonomous driving. The method addresses two core limitations of spiking reinforcement learning: information loss stemming from the conversion of dense visual inputs into sparse spike trains, and the limited representational capacity of spike-based value functions, which often yields weakly discriminative Q-value estimates. The encoder introduces trainable fuzzy membership functions to generate expressive, population-based spike representations, and the decoder uses a lightweight neural decoder to reconstruct continuous Q-values from spiking outputs. Experiments on the HighwayEnv benchmark show that the proposed architecture substantially improves decision-making accuracy and closes the performance gap between spiking and non-spiking multi-modal Q-networks. The results highlight the potential of this framework for efficient and real-time autonomous driving with spiking neural networks.

**arXiv ID:** 2604.16436
</details>

<details>
<summary><strong>Penny Wise, Pixel Foolish: Bypassing Price Constraints in Multimodal Agents via Visual Adversarial Perturbations</strong> - Jiachen Qian, Zhaolu Kang - [[pdf]](https://arxiv.org/pdf/2604.16515)</summary>

**Abstract:** The rapid proliferation of Multimodal Large Language Models (MLLMs) has enabled mobile agents to execute high-stakes financial transactions, but their adversarial robustness remains underexplored. We identify Visual Dominance Hallucination (VDH), where imperceptible visual cues can override textual price evidence in screenshot-based, price-constrained settings and lead agents to irrational decisions. We propose PriceBlind, a stealthy white-box adversarial attack framework for controlled screenshot-based evaluation. PriceBlind exploits the modality gap in CLIP-based encoders via a Semantic-Decoupling Loss that aligns the image embedding with low-cost, value-associated anchors while preserving pixel-level fidelity. On E-ShopBench, PriceBlind achieves around 80% ASR in white-box evaluation; under a simplified single-turn coordinate-selection protocol, Ensemble-DI-FGSM transfers with roughly 35-41% ASR across GPT-4o, Gemini-1.5-Pro, and Claude-3.5-Sonnet. We also show that robust encoders and Verify-then-Act defenses reduce ASR substantially, though with some clean-accuracy trade-off.

**arXiv ID:** 2604.16515
</details>

<details>
<summary><strong>Autonomous Vehicle Collision Avoidance With Racing Parameterized Deep Reinforcement Learning</strong> - Shathushan Sivashangaran, Vihaan Dutta, Apoorva Khairnar, Sepideh Gohari, Azim Eskandarian - [[pdf]](https://arxiv.org/pdf/2604.16702)</summary>

**Abstract:** Road traffic accidents are a leading cause of fatalities worldwide. In the US, human error causes 94% of crashes, resulting in excess of 7,000 pedestrian fatalities and $500 billion in costs annually. Autonomous Vehicles (AVs) with emergency collision avoidance systems that operate at the limits of vehicle dynamics at a high frequency, a dual constraint of nonlinear kinodynamic accuracy and computational efficiency, further enhance safety benefits during adverse weather and cybersecurity breaches, and to evade dangerous human driving when AVs and human drivers share roads. This paper parameterizes a Deep Reinforcement Learning (DRL) collision avoidance policy Out-Of-Distribution (OOD) utilizing race car overtaking, without explicit geometric mimicry reference trajectory guidance, in simulation, with a physics-informed, simulator exploit-aware reward to encode nonlinear vehicle kinodynamics. Two policies are evaluated, a default uni-direction and a reversed heading variant that navigates in the opposite direction to other cars, which both consistently outperform a Model Predictive Control and Artificial Potential Function (MPC-APF) baseline, with zero-shot transfer to proportionally scaled hardware, across three intersection collision scenarios, at 31x fewer Floating Point Operations (FLOPS) and 64x lower inference latency. The reversed heading policy outperforms the default racing overtaking policy in head-to-head collisions by 30% and the baseline by 50%, and matches the former in side collisions, where both DRL policies evade 10% greater than numerical optimal control.

**arXiv ID:** 2604.16702
</details>

<details>
<summary><strong>NaviFormer: A Deep Reinforcement Learning Transformer-like Model to Holistically Solve the Navigation Problem</strong> - Daniel Fuertes, Andrea Cavallaro, Carlos R. del-Blanco, Fernando Jaureguizar, Narciso García - [[pdf]](https://arxiv.org/pdf/2604.16967)</summary>

**Abstract:** Path planning is usually solved by addressing either the (high-level) route planning problem (waypoint sequencing to achieve the final goal) or the (low-level) path planning problem (trajectory prediction between two waypoints avoiding collisions). However, real-world problems usually require simultaneous solutions to the route and path planning subproblems with a holistic and efficient approach. In this paper, we introduce NaviFormer, a deep reinforcement learning model based on a Transformer architecture that solves the global navigation problem by predicting both high-level routes and low-level trajectories. To evaluate NaviFormer, several experiments have been conducted, including comparisons with other algorithms. Results show competitive accuracy from NaviFormer since it can understand the constraints and difficulties of each subproblem and act consequently to improve performance. Moreover, its superior computation speed proves its suitability for real-time missions.

**arXiv ID:** 2604.16967
</details>

<details>
<summary><strong>Web-Gewu: A Browser-Based Interactive Playground for Robot Reinforcement Learning</strong> - Kaixuan Chen, Linqi Ye - [[pdf]](https://arxiv.org/pdf/2604.17050)</summary>

**Abstract:** With the rapid development of embodied intelligence, robotics education faces a dual challenge: high computational barriers and cumbersome environment configuration. Existing centralized cloud simulation solutions incur substantial GPU and bandwidth costs that preclude large-scale deployment, while pure local computing is severely constrained by learners' hardware limitations. To address these issues, we propose \href{this http URL}{Web-Gewu}, an interactive robotics education platform built on a WebRTC cloud-edge-client collaborative architecture. The system offloads all physics simulation and reinforcement learning (RL) training to the edge node, while the cloud server acts exclusively as a lightweight signaling relay, enabling extremely low-cost browser-based peer-to-peer (P2P) real-time streaming. Learners can interact with multi-form robots at low end-to-end latency directly in a web browser without any local installation, and simultaneously observe real-time visualization of multi-dimensional monitoring data, including reinforcement learning reward curves. Combined with a predefined robust command communication protocol, Web-Gewu provides a highly scalable, out-of-the-box, and barrier-free teaching infrastructure for embodied intelligence, significantly lowering the barrier to entry for cutting-edge robotics technology.

**arXiv ID:** 2604.17050
</details>

<details>
<summary><strong>Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models</strong> - Haiweng Xu, Sipeng Zheng, Hao Luo, Wanpeng Zhang, Ziheng Xi, Zongqing Lu - [[pdf]](https://arxiv.org/pdf/2604.18000)</summary>

**Abstract:** Recent Vision-Language-Action (VLA) models report impressive success rates on standard robotic benchmarks, fueling optimism about general-purpose physical intelligence. However, recent evidence suggests a systematic misalignment between standard benchmark success and true embodied reasoning, raising the question of whether these high scores reflect genuine cognitive capability. To address this gap, we introduce BeTTER, a diagnostic Benchmark for Testing True Embodied Reasoning in robotic policies. BeTTER applies targeted causal interventions (e.g., spatial layout shifts, temporal extrapolation) while enforcing kinematic isolation to explicitly decouple high-level reasoning failures from low-level execution limits. Through systematic evaluation, we reveal that state-of-the-art VLAs catastrophically fail in dynamic scenarios, exhibiting severe lexical-kinematic shortcuts, behavioral inertia, and semantic feature collapse. Crucially, our mechanistic analysis traces these symptoms to fundamental architectural bottlenecks - such as capacity compression and myopic downsampling - which systematically degrade the model's foundational semantic representation. We demonstrate that highly static evaluation protocols effectively mask this degradation by allowing optimization to overfit to sensorimotor priors. Supported by real-world robotic validation, our findings confirm that this representational breakdown is not a simulation artifact, highlighting the critical need for future VLA paradigms to resolve the structural tension between high-frequency control and high-level reasoning.

**arXiv ID:** 2604.18000
</details>

<details>
<summary><strong>SAGE: Sensor-Augmented Grounding Engine for LLM-Powered Sleep Care Agent</strong> - Hansoo Lee, Yoonjae Cho, Sonya S.Kwak, Rafael A. Calvo - [[pdf]](https://arxiv.org/pdf/2604.16342)</summary>

**Abstract:** Sleep is vital for health, yet access to data alone does not guarantee improvement. While wearables and health apps enable tracking, users face a "Data-Action Gap," struggling to interpret metrics and translate them into action. Current interventions fail to bridge this: static dashboards lack context, rule-based agents rely on rigid scripts, and LLM-agents lack grounding in personal data, causing trust issues. We propose SAGE (Sensor-Augmented Grounding Engine) for an LLM-powered sleep care agent. SAGE normalizes continuous sleep, physiological, and activity data from the sensors into a queryable time-series layer. It supports (1) selective system-initiated monitoring that triggers notifications only upon detecting meaningful deviations against personal baselines to reduce alert fatigue, and (2) user-initiated Q&A where natural language is translated into executable database queries. By ensuring responses are grounded in precise period, comparison, and metric data, SAGE aims to enhance personalization, traceability, and trust, articulating a novel design space for evidence-based messaging in sleep care.

**arXiv ID:** 2604.16342
</details>

<details>
<summary><strong>DR. INFO at the Point of Care: A Prospective Pilot Study of an Agentic AI Clinical Assistant</strong> - Rogerio Corga Da Silva, Miguel Romano, Tiago Mendes, Marta Isidoro, Sandhanakrishnan Ravichandran, Shivesh Kumar, Michiel van der Heijden, Olivier Fail, Valentine Emmanuel Gnanapragasam - [[pdf]](https://arxiv.org/pdf/2604.16346)</summary>

**Abstract:** Background: Clinical documentation and information retrieval consume over half of physicians working hours, contributing to cognitive overload and burnout. While artificial intelligence offers a potential solution, concerns over hallucinations and source reliability have limited adoption at the point of care.
Objective: To evaluate clinician-reported time savings, decision-making support, and satisfaction with DR. INFO, an agentic AI clinical assistant, in routine clinical practice.
Methods: In this prospective, single-arm pilot study, 29 clinicians across multiple specialties in Portuguese healthcare institutions used DR. INFO v1.0 over five working days within a two-week period. Outcomes were assessed via daily Likert-scale evaluations and a final Net Promoter Score. Non-parametric methods were used throughout.
Results: Clinicians reported high perceived time saving (mean 4.27/5; 95% CI: 3.97-4.57) and decision support (4.16/5; 95% CI: 3.86-4.45), with ratings stable across all study days and no evidence of attrition bias. The Net Promoter Score was 81.2, with no detractors.
Conclusions: Clinicians across specialties and career stages reported sustained satisfaction with DR. INFO for both time efficiency and clinical decision support. Validation in larger, controlled studies with objective outcome measures is warranted.

**arXiv ID:** 2604.16346
</details>

<details>
<summary><strong>AgentClick: A Skill-Based Human-in-the-Loop Review Layer for Terminal AI Agents</strong> - Haomin Zhuang, Hanwen Xing, Xiangliang Zhang - [[pdf]](https://arxiv.org/pdf/2604.16520)</summary>

**Abstract:** Recent autonomous AI agents such as Codex, and Claude Code have made it increasingly practical for users to delegate complex tasks, including writing emails, executing code, issuing shell commands, and carrying out multi-step plans. However, despite these capabilities, human-agent interaction still largely happens through terminal interfaces or remote text-based channels such as Discord. These interaction modes are often inefficient and unfriendly: long text outputs are difficult to read and review, proposed actions lack clear structure and visual context, and users must express feedback by typing detailed corrections, which is cumbersome and often discourages effective collaboration. As a result, non-expert users in particular face a high barrier to working productively with agents. To address this gap, we present AgentClick, an interactive review layer for terminal-based agents. AgentClick is implemented as a localhost npm server paired with a skill-based plugin that connects the running agent to a browser interface, allowing users to supervise and collaborate with agents through a structured web UI rather than raw terminal text alone. The system supports a range of human-in-the-loop workflows, including email drafting and revision, plan review and modification, memory management, trajectory inspection and visualization, and error localization during agent execution. It also turns code generation and execution into a reviewable process, enabling users to inspect and intervene before consequential actions are taken. In addition, AgentClick supports persistent preference capture through editable memory and remote access over HTTP, allowing users to review agents running on servers from their personal devices. Our goal is to lower the barrier for non-expert users and improve the efficiency and quality of human-agent co-work.

**arXiv ID:** 2604.16520
</details>

<details>
<summary><strong>Design and Evaluation of a Culturally Adapted Multimodal Virtual Agent for PTSD Screening</strong> - Cengiz Ozel, Waleed Nadeem, Samuel Potter, Yahya Bokhari, Bdour Alwuqaysi, Wejdan Alotaibi, Rahaf Fahad Alnufaie, Sabri Boughorbel, Abdulrhman Aljouie, Rakan Altasan, Ehsan Hoque - [[pdf]](https://arxiv.org/pdf/2604.17871)</summary>

**Abstract:** Post-traumatic stress disorder (PTSD) is highly prevalent yet chronically underreported among combat-exposed military personnel. This paper presents Molhim, a culturally adapted multimodal conversational AI platform that supports purpose-specific interactions through a configurable conversational pipeline consisting of session setup, real-time dialogue with a high-fidelity virtual avatar, and post-session analysis and feedback. In this work, we examine the PTSD screening configuration of the Molhim platform in a military healthcare context. The system employs a conversational avatar driven by a large language model, integrating real-time speech recognition, visual understanding of user input, text-to-speech synthesis, and a high-fidelity human avatar to support structured multi-turn dialogue and automated post-session analysis, including administration of the PTSD Checklist for DSM-5 (PCL-5). These findings suggest the feasibility of Molhim as a conversational platform for PTSD screening and highlight design considerations for socially cooperative human-AI systems in clinical environments.

**arXiv ID:** 2604.17871
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
