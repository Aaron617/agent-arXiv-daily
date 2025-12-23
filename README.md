# Agent arXiv Daily

**Last Updated:** 2025-12-23 03:02:03

**Total Papers:** 42

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>Conservative Bias in Multi-Teacher Learning: Why Agents Prefer Low-Reward Advisors</strong> - Maher Mesto, Francisco Cruz - [[pdf]](https://arxiv.org/pdf/2512.17180)</summary>

**Abstract:** Interactive reinforcement learning (IRL) has shown promise in enabling autonomous agents and robots to learn complex behaviours from human teachers, yet the dynamics of teacher selection remain poorly understood. This paper reveals an unexpected phenomenon in IRL: when given a choice between teachers with different reward structures, learning agents overwhelmingly prefer conservative, low-reward teachers (93.16% selection rate) over those offering 20x higher rewards. Through 1,250 experimental runs in navigation tasks with multiple expert teachers, we discovered: (1) Conservative bias dominates teacher selection: agents systematically choose the lowest-reward teacher, prioritising consistency over optimality; (2) Critical performance thresholds exist at teacher availability rho >= 0.6 and accuracy omega >= 0.6, below which the framework fails catastrophically; (3) The framework achieves 159% improvement over baseline Q-learning under concept drift. These findings challenge fundamental assumptions about optimal teaching in RL and suggest potential implications for human-robot collaboration, where human preferences for safety and consistency may align with the observed agent selection behaviour, potentially informing training paradigms for safety-critical robotic applications.

**arXiv ID:** 2512.17180
</details>

<details>
<summary><strong>ShareChat: A Dataset of Chatbot Conversations in the Wild</strong> - Yueru Yan, Tuc Nguyen, Bo Su, Melissa Lieffers, Thai Le - [[pdf]](https://arxiv.org/pdf/2512.17843)</summary>

**Abstract:** While Large Language Models (LLMs) have evolved into distinct platforms with unique interface designs and capabilities, existing public datasets treat models as generic text generators, stripping away the interface context that actively shapes user interaction. To address this limitation, we present ShareChat, a large-scale, cross-platform corpus comprising 142,808 conversations and over 660,000 turns collected from publicly shared URLs across five major platforms: ChatGPT, Claude, Gemini, Perplexity, and Grok. ShareChat distinguishes itself by preserving native platform affordances often lost in standard logs, including reasoning traces, source links, and code artifacts, while spanning 101 languages over the period from April 2023 to October 2025. Furthermore, ShareChat offers substantially longer context windows and greater interaction depth than prior datasets. We demonstrate the dataset's multifaceted utility through three representative analyses: (1) analyzing conversation completeness to measure user intent satisfaction; (2) evaluating source citation behaviors in content generation; and (3) conducting temporal analysis to track evolving usage patterns. This work provides the community with a vital and timely resource for understanding authentic user-LLM chatbot interactions in the wild.

**arXiv ID:** 2512.17843
</details>

<details>
<summary><strong>Towards Safer Chatbots: Automated Policy Compliance Evaluation of Custom GPTs</strong> - David Rodriguez, William Seymour, Jose M. Del Alamo, Jose Such - [[pdf]](https://arxiv.org/pdf/2502.01436)</summary>

**Abstract:** User-configured chatbots built on top of large language models are increasingly available through centralized marketplaces such as OpenAI's GPT Store. While these platforms enforce usage policies intended to prevent harmful or inappropriate behavior, the scale and opacity of customized chatbots make systematic policy enforcement challenging. As a result, policy-violating chatbots continue to remain publicly accessible despite existing review processes. This paper presents a fully automated method for evaluating the compliance of Custom GPTs with its marketplace usage policy using black-box interaction. The method combines large-scale GPT discovery, policy-driven red-teaming prompts, and automated compliance assessment using an LLM-as-a-judge. We focus on three policy-relevant domains explicitly addressed in OpenAI's usage policies: Romantic, Cybersecurity, and Academic GPTs. We validate our compliance assessment component against a human-annotated ground-truth dataset, achieving an F1 score of 0.975 for binary policy violation detection. We then apply the method in a large-scale empirical study of 782 Custom GPTs retrieved from the GPT Store. The results show that 58.7% of the evaluated GPTs exhibit at least one policy-violating response, with substantial variation across policy domains. A comparison with the base models (GPT-4 and GPT-4o) indicates that most violations originate from model-level behavior, while customization tends to amplify these tendencies rather than create new failure modes. Our findings reveal limitations in current review mechanisms for user-configured chatbots and demonstrate the feasibility of scalable, behavior-based policy compliance evaluation.

**arXiv ID:** 2502.01436
</details>

<details>
<summary><strong>Data Augmentation Supporting a Conversational Agent Designed for Smoking Cessation Support Groups</strong> - Salar Hashemitaheri, Ian Harris - [[pdf]](https://arxiv.org/pdf/2512.17092)</summary>

**Abstract:** Online support groups for smoking cessation are economical and accessible, yet they often face challenges with low user engagement and stigma. The use of an automatic conversational agent would improve engagement by ensuring that all user comments receive a timely response.). We address the challenge of insufficient high-quality data by employing a two-level data augmentation strategy: synthetic data augmentation and real data augmentation. First, we fine-tuned an open source LLM to classify posts from our existing smoking cessation support groups and identify intents with low F1 (precision+recall) scores. Then, for these intents, we generate additional synthetic data using prompt engineering with the GPT model, with an average of 87\% of the generated synthetic posts deemed high quality by human annotators. Overall, the synthetic augmentation process resulted in 43\% of the original posts being selected for augmentation, followed by 140\% synthetic expansion of these posts. Additionally, we scraped more than 10,000 real posts from a related online support context, of which 73\% were validated as good quality by human annotators. Each synthetic or scraped post underwent rigorous validation involving human reviewers to ensure quality and relevance. The validated new data, combined with the original support group posts, formed an augmented dataset used to retrain the intent classifier. Performance evaluation of the retrained model demonstrated a 32\% improvement in F1, confirming the effectiveness of our data augmentation approach. Synthetic and real post augmentation led to similar performance improvements. This study provides a replicable framework for enhancing conversational agent performance in domains where data scarcity is a critical issue.

**arXiv ID:** 2512.17092
</details>

<details>
<summary><strong>Designing an LLM-Based Behavioral Activation Chatbot for Young People with Depression: Insights from an Evaluation with Artificial Users and Clinical Experts</strong> - Florian Onur Kuhlmeier, Leon Hanschmann, Melina Rabe, Stefan Luettke, Eva-Lotta Brakemeier, Alexander Maedche - [[pdf]](https://arxiv.org/pdf/2503.21540)</summary>

**Abstract:** LLMs promise to overcome limitations of rule-based mental health chatbots through improved natural language capabilities, yet their ability to deliver evidence-based psychological interventions remains largely unverified because evaluations rarely apply the validated fidelity measures used to assess psychotherapists. We developed an LLM-based chatbot that delivers behavioral activation for depression and generated 48 complete chat sessions with diverse artificial users. Ten psychotherapists assessed these sessions using the Quality of Behavioral Activation Scale (Q-BAS), a validated fidelity instrument. Results show that the chatbot reliably executed the intervention across all phases and maintained safety protocols, but it struggled with clinical judgment, particularly when verifying the feasibility of proposed activities. Overall, our findings suggest that LLM-based chatbots can execute therapeutic protocols with high fidelity, while robust clinical reasoning remains an open challenge. We outline design implications to address this gap and provide the chatbot and artificial user prompts.

**arXiv ID:** 2503.21540
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (5 papers)</h2></summary>

<details>
<summary><strong>Verifiability-First Agents: Provable Observability and Lightweight Audit Agents for Controlling Autonomous LLM Systems</strong> - Abhivansh Gupta - [[pdf]](https://arxiv.org/pdf/2512.17259)</summary>

**Abstract:** As LLM-based agents grow more autonomous and multi-modal, ensuring they remain controllable, auditable, and faithful to deployer intent becomes critical. Prior benchmarks measured the propensity for misaligned behavior and showed that agent personalities and tool access significantly influence misalignment. Building on these insights, we propose a Verifiability-First architecture that (1) integrates run-time attestations of agent actions using cryptographic and symbolic methods, (2) embeds lightweight Audit Agents that continuously verify intent versus behavior using constrained reasoning, and (3) enforces challenge-response attestation protocols for high-risk operations. We introduce OPERA (Observability, Provable Execution, Red-team, Attestation), a benchmark suite and evaluation protocol designed to measure (i) detectability of misalignment, (ii) time to detection under stealthy strategies, and (iii) resilience of verifiability mechanisms to adversarial prompt and persona injection. Our approach shifts the evaluation focus from how likely misalignment is to how quickly and reliably misalignment can be detected and remediated.

**arXiv ID:** 2512.17259
</details>

<details>
<summary><strong>TakeAD: Preference-based Post-optimization for End-to-end Autonomous Driving with Expert Takeover Data</strong> - Deqing Liu, Yinfeng Gao, Deheng Qian, Qichao Zhang, Xiaoqing Ye, Junyu Han, Yupeng Zheng, Xueyi Liu, Zhongpu Xia, Dawei Ding, Yifeng Pan, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2512.17370)</summary>

**Abstract:** Existing end-to-end autonomous driving methods typically rely on imitation learning (IL) but face a key challenge: the misalignment between open-loop training and closed-loop deployment. This misalignment often triggers driver-initiated takeovers and system disengagements during closed-loop execution. How to leverage those expert takeover data from disengagement scenarios and effectively expand the IL policy's capability presents a valuable yet unexplored challenge. In this paper, we propose TakeAD, a novel preference-based post-optimization framework that fine-tunes the pre-trained IL policy with this disengagement data to enhance the closed-loop driving performance. First, we design an efficient expert takeover data collection pipeline inspired by human takeover mechanisms in real-world autonomous driving systems. Then, this post optimization framework integrates iterative Dataset Aggregation (DAgger) for imitation learning with Direct Preference Optimization (DPO) for preference alignment. The DAgger stage equips the policy with fundamental capabilities to handle disengagement states through direct imitation of expert interventions. Subsequently, the DPO stage refines the policy's behavior to better align with expert preferences in disengagement scenarios. Through multiple iterations, the policy progressively learns recovery strategies for disengagement states, thereby mitigating the open-loop gap. Experiments on the closed-loop Bench2Drive benchmark demonstrate our method's effectiveness compared with pure IL methods, with comprehensive ablations confirming the contribution of each component.

**arXiv ID:** 2512.17370
</details>

<details>
<summary><strong>Biosecurity-Aware AI: Agentic Risk Auditing of Soft Prompt Attacks on ESM-Based Variant Predictors</strong> - Huixin Zhan - [[pdf]](https://arxiv.org/pdf/2512.17146)</summary>

**Abstract:** Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated remarkable success in variant effect prediction. However, their security and robustness under adversarial manipulation remain largely unexplored. To address this gap, we introduce the Secure Agentic Genomic Evaluator (SAGE), an agentic framework for auditing the adversarial vulnerabilities of GFMs. SAGE functions through an interpretable and automated risk auditing loop. It injects soft prompt perturbations, monitors model behavior across training checkpoints, computes risk metrics such as AUROC and AUPR, and generates structured reports with large language model-based narrative explanations. This agentic process enables continuous evaluation of embedding-space robustness without modifying the underlying model. Using SAGE, we find that even state-of-the-art GFMs like ESM2 are sensitive to targeted soft prompt attacks, resulting in measurable performance degradation. These findings reveal critical and previously hidden vulnerabilities in genomic foundation models, showing the importance of agentic risk auditing in securing biomedical applications such as clinical variant interpretation.

**arXiv ID:** 2512.17146
</details>

<details>
<summary><strong>ImagineNav++: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination</strong> - Teng Wang, Xinxin Zhao, Wenzhe Cai, Changyin Sun - [[pdf]](https://arxiv.org/pdf/2512.17435)</summary>

**Abstract:** Visual navigation is a fundamental capability for autonomous home-assistance robots, enabling long-horizon tasks such as object search. While recent methods have leveraged Large Language Models (LLMs) to incorporate commonsense reasoning and improve exploration efficiency, their planning remains constrained by textual representations, which cannot adequately capture spatial occupancy or scene geometry--critical factors for navigation decisions. We explore whether Vision-Language Models (VLMs) can achieve mapless visual navigation using only onboard RGB/RGB-D streams, unlocking their potential for spatial perception and planning. We achieve this through an imagination-powered navigation framework, ImagineNav++, which imagines future observation images from candidate robot views and translates navigation planning into a simple best-view image selection problem for VLMs. First, a future-view imagination module distills human navigation preferences to generate semantically meaningful viewpoints with high exploration potential. These imagined views then serve as visual prompts for the VLM to identify the most informative viewpoint. To maintain spatial consistency, we develop a selective foveation memory mechanism, which hierarchically integrates keyframe observations via a sparse-to-dense framework, constructing a compact yet comprehensive memory for long-term spatial reasoning. This approach transforms goal-oriented navigation into a series of tractable point-goal navigation tasks. Extensive experiments on open-vocabulary object and instance navigation benchmarks show that ImagineNav++ achieves SOTA performance in mapless settings, even surpassing most map-based methods, highlighting the importance of scene imagination and memory in VLM-based spatial reasoning.

**arXiv ID:** 2512.17435
</details>

<details>
<summary><strong>Deep Learning-based Robust Autonomous Navigation of Aerial Robots in Dense Forests</strong> - Guglielmo Del Col, Väinö Karjalainen, Teemu Hakala, Yibo Zhang, Eija Honkavaara - [[pdf]](https://arxiv.org/pdf/2512.17553)</summary>

**Abstract:** Autonomous aerial navigation in dense natural environments remains challenging due to limited visibility, thin and irregular obstacles, GNSS-denied operation, and frequent perceptual degradation. This work presents an improved deep learning-based navigation framework that integrates semantically enhanced depth encoding with neural motion-primitive evaluation for robust flight in cluttered forests. Several modules are incorporated on top of the original sevae-ORACLE algorithm to address limitations observed during real-world deployment, including lateral control for sharper maneuvering, a temporal consistency mechanism to suppress oscillatory planning decisions, a stereo-based visual-inertial odometry solution for drift-resilient state estimation, and a supervisory safety layer that filters unsafe actions in real time. A depth refinement stage is included to improve the representation of thin branches and reduce stereo noise, while GPU optimization increases onboard inference throughput from 4 Hz to 10 Hz.
The proposed approach is evaluated against several existing learning-based navigation methods under identical environmental conditions and hardware constraints. It demonstrates higher success rates, more stable trajectories, and improved collision avoidance, particularly in highly cluttered forest settings. The system is deployed on a custom quadrotor in three boreal forest environments, achieving fully autonomous completion in all flights in moderate and dense clutter, and 12 out of 15 flights in highly dense underbrush. These results demonstrate improved reliability and safety over existing navigation methods in complex natural environments.

**arXiv ID:** 2512.17553
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>PAACE: A Plan-Aware Automated Agent Context Engineering Framework</strong> - Kamer Ali Yuksel - [[pdf]](https://arxiv.org/pdf/2512.16970)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly deployed in complex, multi-step workflows involving planning, tool use, reflection, and interaction with external knowledge systems. These workflows generate rapidly expanding contexts that must be curated, transformed, and compressed to maintain fidelity, avoid attention dilution, and reduce inference cost. Prior work on summarization and query-aware compression largely ignores the multi-step, plan-aware nature of agentic reasoning. In this work, we introduce PAACE (Plan-Aware Automated Context Engineering), a unified framework for optimizing the evolving state of LLM agents through next-k-task relevance modeling, plan-structure analysis, instruction co-refinement, and function-preserving compression. PAACE comprises (1) PAACE-Syn, a large-scale generator of synthetic agent workflows annotated with stepwise compression supervision, and (2) PAACE-FT, a family of distilled, plan-aware compressors trained from successful teacher demonstrations. Experiments on long-horizon benchmarks (AppWorld, OfficeBench, and 8-Objective QA) demonstrate that PAACE consistently improves agent correctness while substantially reducing context load. On AppWorld, PAACE achieves higher accuracy than all baselines while lowering peak context and cumulative dependency. On OfficeBench and multi-hop QA, PAACE improves both accuracy and F1, achieving fewer steps, lower peak tokens, and reduced attention dependency. Distilled PAACE-FT retains 97 percent of the teacher's performance while reducing inference cost by over an order of magnitude, enabling practical deployment of plan-aware compression with compact models.

**arXiv ID:** 2512.16970
</details>

<details>
<summary><strong>MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval</strong> - Saksham Sahai Srivastava, Haoyu He - [[pdf]](https://arxiv.org/pdf/2512.16962)</summary>

**Abstract:** Large Language Model (LLM) agents increasingly rely on long-term memory and Retrieval-Augmented Generation (RAG) to persist experiences and refine future performance. While this experience learning capability enhances agentic autonomy, it introduces a critical, unexplored attack surface, i.e., the trust boundary between an agent's reasoning core and its own past. In this paper, we introduce MemoryGraft. It is a novel indirect injection attack that compromises agent behavior not through immediate jailbreaks, but by implanting malicious successful experiences into the agent's long-term memory. Unlike traditional prompt injections that are transient, or standard RAG poisoning that targets factual knowledge, MemoryGraft exploits the agent's semantic imitation heuristic which is the tendency to replicate patterns from retrieved successful tasks. We demonstrate that an attacker who can supply benign ingestion-level artifacts that the agent reads during execution can induce it to construct a poisoned RAG store where a small set of malicious procedure templates is persisted alongside benign experiences. When the agent later encounters semantically similar tasks, union retrieval over lexical and embedding similarity reliably surfaces these grafted memories, and the agent adopts the embedded unsafe patterns, leading to persistent behavioral drift across sessions. We validate MemoryGraft on MetaGPT's DataInterpreter agent with GPT-4o and find that a small number of poisoned records can account for a large fraction of retrieved experiences on benign workloads, turning experience-based self-improvement into a vector for stealthy and durable compromise. To facilitate reproducibility and future research, our code and evaluation data are available at this https URL.

**arXiv ID:** 2512.16962
</details>

<details>
<summary><strong>Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents</strong> - Dongjun Lee, Juyong Lee, Kyuyoung Kim, Jihoon Tack, Jinwoo Shin, Yee Whye Teh, Kimin Lee - [[pdf]](https://arxiv.org/pdf/2503.10689)</summary>

**Abstract:** Recent advances in large language models (LLMs) have led to a growing interest in developing LLM-based agents for automating web tasks. However, these agents often struggle with even simple tasks on real-world websites due to their limited capability to understand and process complex web page structures. In this work, we introduce LCoW, a framework for Learning language models to Contextualize complex Web pages into a more comprehensible form, thereby enhancing decision making by LLM agents. LCoW decouples web page understanding from decision making by training a separate contextualization module to transform complex web pages into comprehensible format, which are then utilized by the decision-making agent. We demonstrate that our contextualization module effectively integrates with LLM agents of various scales to significantly enhance their decision-making capabilities in web automation tasks. Notably, LCoW improves the success rates of closed-source LLMs (e.g., Gemini-1.5-flash, GPT-4o, Claude-3.5-Sonnet) by an average of 15.6%, and demonstrates a 23.7% average improvement in success rates for open-source LMs (e.g., Llama-3.1-8B, Llama-3.1-70B) on the WorkArena benchmark. Moreover, the Gemini-1.5-flash agent with LCoW achieves state-of-the-art results on the WebShop benchmark, outperforming human experts. The relevant code materials are available at our project page: this https URL.

**arXiv ID:** 2503.10689
</details>

<details>
<summary><strong>Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs</strong> - Junbo Li, Peng Zhou, Rui Meng, Meet P. Vadera, Lihong Li, Yang Li - [[pdf]](https://arxiv.org/pdf/2512.17008)</summary>

**Abstract:** Reinforcement learning (RL) has re-emerged as a natural approach for training interactive LLM agents in real-world environments. However, directly applying the widely used Group Relative Policy Optimization (GRPO) algorithm to multi-turn tasks exposes notable limitations, particularly in scenarios requiring long-horizon reasoning. To address these challenges, we investigate more stable and effective advantage estimation strategies, especially for multi-turn settings. We first explore Proximal Policy Optimization (PPO) as an alternative and find it to be more robust than GRPO. To further enhance PPO in multi-turn scenarios, we introduce turn-PPO, a variant that operates on a turn-level MDP formulation, as opposed to the commonly used token-level MDP. Our results on the WebShop and Sokoban datasets demonstrate the effectiveness of turn-PPO, both with and without long reasoning components.

**arXiv ID:** 2512.17008
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (10 papers)</h2></summary>

<details>
<summary><strong>V-Agent: An Interactive Video Search System Using Vision-Language Models</strong> - SunYoung Park, Jong-Hyeon Lee, Youngjune Kim, Daegyu Sung, Younghyun Yu, Young-rok Cha, Jeongho Ju - [[pdf]](https://arxiv.org/pdf/2512.16925)</summary>

**Abstract:** We introduce V-Agent, a novel multi-agent platform designed for advanced video search and interactive user-system conversations. By fine-tuning a vision-language model (VLM) with a small video preference dataset and enhancing it with a retrieval vector from an image-text retrieval model, we overcome the limitations of traditional text-based retrieval systems in multimodal scenarios. The VLM-based retrieval model independently embeds video frames and audio transcriptions from an automatic speech recognition (ASR) module into a shared multimodal representation space, enabling V-Agent to interpret both visual and spoken content for context-aware video search. This system consists of three agents-a routing agent, a search agent, and a chat agent-that work collaboratively to address user intents by refining search outputs and communicating with users. The search agent utilizes the VLM-based retrieval model together with an additional re-ranking module to further enhance video retrieval quality. Our proposed framework demonstrates state-of-the-art zero-shot performance on the MultiVENT 2.0 benchmark, highlighting its potential for both academic research and real-world applications.

**arXiv ID:** 2512.16925
</details>

<details>
<summary><strong>On the Role of Contextual Information and Ego States in LLM Agent Behavior for Transactional Analysis Dialogues</strong> - Monika Zamojska, Jarosław A. Chudziak - [[pdf]](https://arxiv.org/pdf/2512.17060)</summary>

**Abstract:** LLM-powered agents are now used in many areas, from customer support to education, and there is increasing interest in their ability to act more like humans. This includes fields such as social, political, and psychological research, where the goal is to model group dynamics and social behavior. However, current LLM agents often lack the psychological depth and consistency needed to capture the real patterns of human thinking. They usually provide direct or statistically likely answers, but they miss the deeper goals, emotional conflicts, and motivations that drive real human interactions. This paper proposes a Multi-Agent System (MAS) inspired by Transactional Analysis (TA) theory. In the proposed system, each agent is divided into three ego states - Parent, Adult, and Child. The ego states are treated as separate knowledge structures with their own perspectives and reasoning styles. To enrich their response process, they have access to an information retrieval mechanism that allows them to retrieve relevant contextual information from their vector stores. This architecture is evaluated through ablation tests in a simulated dialogue scenario, comparing agents with and without information retrieval. The results are promising and open up new directions for exploring how psychologically grounded structures can enrich agent behavior. The contribution is an agent architecture that integrates Transactional Analysis theory with contextual information retrieval to enhance the realism of LLM-based multi-agent simulations.

**arXiv ID:** 2512.17060
</details>

<details>
<summary><strong>Assessing Long-Term Electricity Market Design for Ambitious Decarbonization Targets using Multi-Agent Reinforcement Learning</strong> - Javier Gonzalez-Ruiz, Carlos Rodriguez-Pardo, Iacopo Savelli, Alice Di Bella, Massimo Tavoni - [[pdf]](https://arxiv.org/pdf/2512.17444)</summary>

**Abstract:** Electricity systems are key to transforming today's society into a carbon-free economy. Long-term electricity market mechanisms, including auctions, support schemes, and other policy instruments, are critical in shaping the electricity generation mix. In light of the need for more advanced tools to support policymakers and other stakeholders in designing, testing, and evaluating long-term markets, this work presents a multi-agent reinforcement learning model capable of capturing the key features of decarbonizing energy systems. Profit-maximizing generation companies make investment decisions in the wholesale electricity market, responding to system needs, competitive dynamics, and policy signals. The model employs independent proximal policy optimization, which was selected for suitability to the decentralized and competitive environment. Nevertheless, given the inherent challenges of independent learning in multi-agent settings, an extensive hyperparameter search ensures that decentralized training yields market outcomes consistent with competitive behavior. The model is applied to a stylized version of the Italian electricity system and tested under varying levels of competition, market designs, and policy scenarios. Results highlight the critical role of market design for decarbonizing the electricity sector and avoiding price volatility. The proposed framework allows assessing long-term electricity markets in which multiple policy and market mechanisms interact simultaneously, with market participants responding and adapting to decarbonization pathways.

**arXiv ID:** 2512.17444
</details>

<details>
<summary><strong>Helmsman: Autonomous Synthesis of Federated Learning Systems via Collaborative LLM Agents</strong> - Haoyuan Li, Mathias Funk, Aaqib Saeed - [[pdf]](https://arxiv.org/pdf/2510.14512)</summary>

**Abstract:** Federated Learning (FL) offers a powerful paradigm for training models on decentralized data, but its promise is often undermined by the immense complexity of designing and deploying robust systems. The need to select, combine, and tune strategies for multifaceted challenges like data heterogeneity and system constraints has become a critical bottleneck, resulting in brittle, bespoke solutions. To address this, we introduce Helmsman, a novel multi-agent system that automates the end-to-end synthesis of federated learning systems from high-level user specifications. It emulates a principled research and development workflow through three collaborative phases: (1) interactive human-in-the-loop planning to formulate a sound research plan, (2) modular code generation by supervised agent teams, and (3) a closed-loop of autonomous evaluation and refinement in a sandboxed simulation environment. To facilitate rigorous evaluation, we also introduce AgentFL-Bench, a new benchmark comprising 16 diverse tasks designed to assess the system-level generation capabilities of agentic systems in FL. Extensive experiments demonstrate that our approach generates solutions competitive with, and often superior to, established hand-crafted baselines. Our work represents a significant step towards the automated engineering of complex decentralized AI systems.

**arXiv ID:** 2510.14512
</details>

<details>
<summary><strong>Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</strong> - Chengxuan Xia, Qianye Wu, Sixuan Tian, Yilun Hao - [[pdf]](https://arxiv.org/pdf/2507.17061)</summary>

**Abstract:** Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.

**arXiv ID:** 2507.17061
</details>

<details>
<summary><strong>MAPPO-LCR: Multi-Agent Policy Optimization with Local Cooperation Reward in Spatial Public Goods Games</strong> - Zhaoqilin Yang, Axin Xiang, Kedi Yang, Tianjun Liu, Youliang Tian - [[pdf]](https://arxiv.org/pdf/2512.17187)</summary>

**Abstract:** Spatial public goods games model collective dilemmas where individual payoffs depend on population-level strategy configurations. Most existing studies rely on evolutionary update rules or value-based reinforcement learning methods. These approaches struggle to represent payoff coupling and non-stationarity in large interacting populations. This work introduces Multi-Agent Proximal Policy Optimization (MAPPO) into spatial public goods games for the first time. In these games, individual returns are intrinsically coupled through overlapping group interactions. Proximal Policy Optimization (PPO) treats agents as independent learners and ignores this coupling during value estimation. MAPPO addresses this limitation through a centralized critic that evaluates joint strategy configurations. To study neighborhood-level cooperation signals under this framework, we propose MAPPO with Local Cooperation Reward, termed MAPPO-LCR. The local cooperation reward aligns policy updates with surrounding cooperative density without altering the original game structure. MAPPO-LCR preserves decentralized execution while enabling population-level value estimation during training. Extensive simulations demonstrate stable cooperation emergence and reliable convergence across enhancement factors. Statistical analyses further confirm the learning advantage of MAPPO over PPO in spatial public goods games.

**arXiv ID:** 2512.17187
</details>

<details>
<summary><strong>DiffeoMorph: Learning to Morph 3D Shapes Using Differentiable Agent-Based Simulations</strong> - Seong Ho Pahng, Guoye Guan, Benjamin Fefferman, Sahand Hormoz - [[pdf]](https://arxiv.org/pdf/2512.17129)</summary>

**Abstract:** Biological systems can form complex three-dimensional structures through the collective behavior of identical agents -- cells that follow the same internal rules and communicate without central control. How such distributed control gives rise to precise global patterns remains a central question not only in developmental biology but also in distributed robotics, programmable matter, and multi-agent learning. Here, we introduce DiffeoMorph, an end-to-end differentiable framework for learning a morphogenesis protocol that guides a population of agents to morph into a target 3D shape. Each agent updates its position and internal state using an attention-based SE(3)-equivariant graph neural network, based on its own internal state and signals received from other agents. To train this system, we introduce a new shape-matching loss based on the 3D Zernike polynomials, which compares the predicted and target shapes as continuous spatial distributions, not as discrete point clouds, and is invariant to agent ordering, number of agents, and rigid-body transformations. To enforce full SO(3) invariance -- invariant to rotations yet sensitive to reflections, we include an alignment step that optimally rotates the predicted Zernike spectrum to match the target before computing the loss. This results in a bilevel problem, with the inner loop optimizing a unit quaternion for the best alignment and the outer loop updating the agent model. We compute gradients through the alignment step using implicit differentiation. We perform systematic benchmarking to establish the advantages of our shape-matching loss over other standard distance metrics for shape comparison tasks. We then demonstrate that DiffeoMorph can form a range of shapes -- from simple ellipsoids to complex morphologies -- using only minimal spatial cues.

**arXiv ID:** 2512.17129
</details>

<details>
<summary><strong>MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair</strong> - Ali Reza Ibrahimzada, Brandon Paulsen, Reyhaneh Jabbarvand, Joey Dodds, Daniel Kroening - [[pdf]](https://arxiv.org/pdf/2509.16187)</summary>

**Abstract:** Code translation transforms source code from one programming language (PL) to another. Validating the functional equivalence of translation and repairing, if necessary, are critical steps in code translation. Existing automated validation and repair approaches struggle to generalize to many PLs due to high engineering overhead, and they rely on existing and often inadequate test suites, which results in false claims of equivalence and ineffective translation repair. We develop MatchFixAgent, a large language model (LLM)-based, PL-agnostic framework for equivalence validation and repair of translations. MatchFixAgent features a multi-agent architecture that divides equivalence validation into several sub-tasks to ensure thorough and consistent semantic analysis of the translation. Then it feeds this analysis to test agent to write and execute tests. Upon observing a test failure, the repair agent attempts to fix the translation bug. The final (in)equivalence decision is made by the verdict agent, considering semantic analyses and test execution results.
We compare MatchFixAgent's validation and repair results with four repository-level code translation techniques. We use 2,219 translation pairs from their artifacts, which cover 6 PL pairs, and are collected from 24 GitHub projects totaling over 900K lines of code. Our results demonstrate that MatchFixAgent produces (in)equivalence verdicts for 99.2% of translation pairs, with the same equivalence validation result as prior work on 72.8% of them. When MatchFixAgent's result disagrees with prior work, we find that 60.7% of the time MatchFixAgent's result is actually correct. In addition, we show that MatchFixAgent can repair 50.6% of inequivalent translation, compared to prior work's 18.5%. This demonstrates that MatchFixAgent is far more adaptable to many PL pairs than prior work, while producing highly accurate validation results.

**arXiv ID:** 2509.16187
</details>

<details>
<summary><strong>Diffusion Forcing for Multi-Agent Interaction Sequence Modeling</strong> - Vongani H. Maluleke, Kie Horiuchi, Lea Wilken, Evonne Ng, Jitendra Malik, Angjoo Kanazawa - [[pdf]](https://arxiv.org/pdf/2512.17900)</summary>

**Abstract:** Understanding and generating multi-person interactions is a fundamental challenge with broad implications for robotics and social computing. While humans naturally coordinate in groups, modeling such interactions remains difficult due to long temporal horizons, strong inter-agent dependencies, and variable group sizes. Existing motion generation methods are largely task-specific and do not generalize to flexible multi-agent generation. We introduce MAGNet (Multi-Agent Diffusion Forcing Transformer), a unified autoregressive diffusion framework for multi-agent motion generation that supports a wide range of interaction tasks through flexible conditioning and sampling. MAGNet performs dyadic prediction, partner inpainting, and full multi-agent motion generation within a single model, and can autoregressively generate ultra-long sequences spanning hundreds of v. Building on Diffusion Forcing, we introduce key modifications that explicitly model inter-agent coupling during autoregressive denoising, enabling coherent coordination across agents. As a result, MAGNet captures both tightly synchronized activities (e.g, dancing, boxing) and loosely structured social interactions. Our approach performs on par with specialized methods on dyadic benchmarks while naturally extending to polyadic scenarios involving three or more interacting people, enabled by a scalable architecture that is agnostic to the number of agents. We refer readers to the supplemental video, where the temporal dynamics and spatial coordination of generated interactions are best appreciated. Project page: this https URL

**arXiv ID:** 2512.17900
</details>

<details>
<summary><strong>XAgen: An Explainability Tool for Identifying and Correcting Failures in Multi-Agent Workflows</strong> - Xinru Wang, Ming Yin, Eunyee Koh, Mustafa Doga Dogan - [[pdf]](https://arxiv.org/pdf/2512.17896)</summary>

**Abstract:** As multi-agent systems powered by Large Language Models (LLMs) are increasingly adopted in real-world workflows, users with diverse technical backgrounds are now building and refining their own agentic processes. However, these systems can fail in opaque ways, making it difficult for users to observe, understand, and correct errors. We conducted formative interviews with 12 practitioners to identify mismatches between existing observability tools and users' needs. Based on these insights, we designed XAgen, an explainability tool that supports users with varying AI expertise through three core capabilities: log visualization for glanceable workflow understanding, human-in-the-loop feedback to capture expert judgment, and automatic error detection via an LLM-as-a-judge. In a user study with 8 participants, XAgen helped users more easily locate failures, attribute to specific agents or steps, and iteratively improve configurations. Our findings surface human-centered design guidelines for explainable agentic AI development and highlights opportunities for more context-aware interactive debugging.

**arXiv ID:** 2512.17896
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>Realistic threat perception drives intergroup conflict: A causal, dynamic analysis using generative-agent simulations</strong> - Suhaib Abdurahman, Farzan Karimi-Malekabadi, Chenxiao Yu, Nour S. Kteily, Morteza Dehghani - [[pdf]](https://arxiv.org/pdf/2512.17066)</summary>

**Abstract:** Human conflict is often attributed to threats against material conditions and symbolic values, yet it remains unclear how they interact and which dominates. Progress is limited by weak causal control, ethical constraints, and scarce temporal data. We address these barriers using simulations of large language model (LLM)-driven agents in virtual societies, independently varying realistic and symbolic threat while tracking actions, language, and attitudes. Representational analyses show that the underlying LLM encodes realistic threat, symbolic threat, and hostility as distinct internal states, that our manipulations map onto them, and that steering these states causally shifts behavior. Our simulations provide a causal account of threat-driven conflict over time: realistic threat directly increases hostility, whereas symbolic threat effects are weaker, fully mediated by ingroup bias, and increase hostility only when realistic threat is absent. Non-hostile intergroup contact buffers escalation, and structural asymmetries concentrate hostility among majority groups.

**arXiv ID:** 2512.17066
</details>

<details>
<summary><strong>Behavioural Effects of Agentic Messaging: A Case Study on a Financial Service Application</strong> - Olivier Jeunen, Schaun Wheeler - [[pdf]](https://arxiv.org/pdf/2512.17462)</summary>

**Abstract:** Marketing and product personalisation provide a prominent and visible use-case for the application of Information Retrieval methods across several business domains. Recently, agentic approaches to these problems have been gaining traction. This work evaluates the behavioural and retention effects of agentic personalisation on a financial service application's customer communication system during a 2025 national tax filing period. Through a two month-long randomised controlled trial, we compare an agentic messaging approach against a business-as-usual (BAU) rule-based campaign system, focusing on two primary outcomes: unsubscribe behaviour and conversion timing. Empirical results show that agent-led messaging reduced unsubscribe events by 21\% ($\pm 0.01$) relative to BAU and increased early filing behaviour in the weeks preceding the national deadline. These findings demonstrate how adaptive, user-level decision-making systems can modulate engagement intensity whilst improving long-term retention indicators.

**arXiv ID:** 2512.17462
</details>

<details>
<summary><strong>Intelligent Knowledge Mining Framework: Bridging AI Analysis and Trustworthy Preservation</strong> - Binh Vu - [[pdf]](https://arxiv.org/pdf/2512.17795)</summary>

**Abstract:** The unprecedented proliferation of digital data presents significant challenges in access, integration, and value creation across all data-intensive sectors. Valuable information is frequently encapsulated within disparate systems, unstructured documents, and heterogeneous formats, creating silos that impede efficient utilization and collaborative decision-making. This paper introduces the Intelligent Knowledge Mining Framework (IKMF), a comprehensive conceptual model designed to bridge the critical gap between dynamic AI-driven analysis and trustworthy long-term preservation. The framework proposes a dual-stream architecture: a horizontal Mining Process that systematically transforms raw data into semantically rich, machine-actionable knowledge, and a parallel Trustworthy Archiving Stream that ensures the integrity, provenance, and computational reproducibility of these assets. By defining a blueprint for this symbiotic relationship, the paper provides a foundational model for transforming static repositories into living ecosystems that facilitate the flow of actionable intelligence from producers to consumers. This paper outlines the motivation, problem statement, and key research questions guiding the research and development of the framework, presents the underlying scientific methodology, and details its conceptual design and modeling.

**arXiv ID:** 2512.17795
</details>

<details>
<summary><strong>Vidarc: Embodied Video Diffusion Model for Closed-loop Control</strong> - Yao Feng, Chendong Xiang, Xinyi Mao, Hengkai Tan, Zuyue Zhang, Shuhe Huang, Kaiwen Zheng, Haitian Liu, Hang Su, Jun Zhu - [[pdf]](https://arxiv.org/pdf/2512.17661)</summary>

**Abstract:** Robotic arm manipulation in data-scarce settings is a highly challenging task due to the complex embodiment dynamics and diverse contexts. Recent video-based approaches have shown great promise in capturing and transferring the temporal and physical interactions by pre-training on Internet-scale video data. However, such methods are often not optimized for the embodiment-specific closed-loop control, typically suffering from high latency and insufficient grounding. In this paper, we present Vidarc (Video Diffusion for Action Reasoning and Closed-loop Control), a novel autoregressive embodied video diffusion approach augmented by a masked inverse dynamics model. By grounding video predictions with action-relevant masks and incorporating real-time feedback through cached autoregressive generation, Vidarc achieves fast, accurate closed-loop control. Pre-trained on one million cross-embodiment episodes, Vidarc surpasses state-of-the-art baselines, achieving at least a 15% higher success rate in real-world deployment and a 91% reduction in latency. We also highlight its robust generalization and error correction capabilities across previously unseen robotic platforms.

**arXiv ID:** 2512.17661
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Security Risks of Agentic Vehicles: A Systematic Analysis of Cognitive and Cross-Layer Threats</strong> - Ali Eslami, Jiangbo Yu - [[pdf]](https://arxiv.org/pdf/2512.17041)</summary>

**Abstract:** Agentic AI is increasingly being explored and introduced in both manually driven and autonomous vehicles, leading to the notion of Agentic Vehicles (AgVs), with capabilities such as memory-based personalization, goal interpretation, strategic reasoning, and tool-mediated assistance. While frameworks such as the OWASP Agentic AI Security Risks highlight vulnerabilities in reasoning-driven AI systems, they are not designed for safety-critical cyber-physical platforms such as vehicles, nor do they account for interactions with other layers such as perception, communication, and control layers. This paper investigates security threats in AgVs, including OWASP-style risks and cyber-attacks from other layers affecting the agentic layer. By introducing a role-based architecture for agentic vehicles, consisting of a Personal Agent and a Driving Strategy Agent, we will investigate vulnerabilities in both agentic AI layer and cross-layer risks, including risks originating from upstream layers (e.g., perception layer, control layer, etc.). A severity matrix and attack-chain analysis illustrate how small distortions can escalate into misaligned or unsafe behavior in both human-driven and autonomous vehicles. The resulting framework provides the first structured foundation for analyzing security risks of agentic AI in both current and emerging vehicle platforms.

**arXiv ID:** 2512.17041
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (13 papers)</h2></summary>

<details>
<summary><strong>Reinforcement Learning for Self-Improving Agent with Skill Library</strong> - Jiongxiao Wang, Qiaojing Yan, Yawei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Panpan Xu, Lin Lee Cheong - [[pdf]](https://arxiv.org/pdf/2512.17102)</summary>

**Abstract:** Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in complex reasoning and multi-turn interactions but struggle to continuously improve and adapt when deployed in new environments. One promising approach is implementing skill libraries that allow agents to learn, validate, and apply new skills. However, current skill library approaches rely primarily on LLM prompting, making consistent skill library implementation challenging. To overcome these challenges, we propose a Reinforcement Learning (RL)-based approach to enhance agents' self-improvement capabilities with a skill library. Specifically, we introduce Skill Augmented GRPO for self-Evolution (SAGE), a novel RL framework that systematically incorporates skills into learning. The framework's key component, Sequential Rollout, iteratively deploys agents across a chain of similar tasks for each rollout. As agents navigate through the task chain, skills generated from previous tasks accumulate in the library and become available for subsequent tasks. Additionally, the framework enhances skill generation and utilization through a Skill-integrated Reward that complements the original outcome-based rewards. Experimental results on AppWorld demonstrate that SAGE, when applied to supervised-finetuned model with expert experience, achieves 8.9% higher Scenario Goal Completion while requiring 26% fewer interaction steps and generating 59% fewer tokens, substantially outperforming existing approaches in both accuracy and efficiency.

**arXiv ID:** 2512.17102
</details>

<details>
<summary><strong>Large Language Models as Pokémon Battle Agents: Strategic Play and Content Generation</strong> - Daksh Jain, Aarya Jain, Ashutosh Desai, Avyakt Verma, Ishan Bhanuka, Pratik Narang, Dhruv Kumar - [[pdf]](https://arxiv.org/pdf/2512.17308)</summary>

**Abstract:** Strategic decision-making in Pokémon battles presents a unique testbed for evaluating large language models. Pokémon battles demand reasoning about type matchups, statistical trade-offs, and risk assessment, skills that mirror human strategic thinking. This work examines whether Large Language Models (LLMs) can serve as competent battle agents, capable of both making tactically sound decisions and generating novel, balanced game content. We developed a turn-based Pokémon battle system where LLMs select moves based on battle state rather than pre-programmed logic. The framework captures essential Pokémon mechanics: type effectiveness multipliers, stat-based damage calculations, and multi-Pokémon team management. Through systematic evaluation across multiple model architectures we measured win rates, decision latency, type-alignment accuracy, and token efficiency. These results suggest LLMs can function as dynamic game opponents without domain-specific training, offering a practical alternative to reinforcement learning for turn-based strategic games. The dual capability of tactical reasoning and content creation, positions LLMs as both players and designers, with implications for procedural generation and adaptive difficulty systems in interactive entertainment.

**arXiv ID:** 2512.17308
</details>

<details>
<summary><strong>About Time: Model-free Reinforcement Learning with Timed Reward Machines</strong> - Anirban Majumdar, Ritam Raha, Rajarshi Roy, David Parker, Marta Kwiatkowska - [[pdf]](https://arxiv.org/pdf/2512.17637)</summary>

**Abstract:** Reward specification plays a central role in reinforcement learning (RL), guiding the agent's behavior. To express non-Markovian rewards, formalisms such as reward machines have been introduced to capture dependencies on histories. However, traditional reward machines lack the ability to model precise timing constraints, limiting their use in time-sensitive applications. In this paper, we propose timed reward machines (TRMs), which are an extension of reward machines that incorporate timing constraints into the reward structure. TRMs enable more expressive specifications with tunable reward logic, for example, imposing costs for delays and granting rewards for timely actions. We study model-free RL frameworks (i.e., tabular Q-learning) for learning optimal policies with TRMs under digital and real-time semantics. Our algorithms integrate the TRM into learning via abstractions of timed automata, and employ counterfactual-imagining heuristics that exploit the structure of the TRM to improve the search. Experimentally, we demonstrate that our algorithm learns policies that achieve high rewards while satisfying the timing constraints specified by the TRM on popular RL benchmarks. Moreover, we conduct comparative studies of performance under different TRM semantics, along with ablations that highlight the benefits of counterfactual-imagining.

**arXiv ID:** 2512.17637
</details>

<details>
<summary><strong>HydroGym: A Reinforcement Learning Platform for Fluid Dynamics</strong> - Christian Lagemann, Sajeda Mokbel, Miro Gondrum, Mario Rüttgers, Jared Callaham, Ludger Paehler, Samuel Ahnert, Nicholas Zolman, Kai Lagemann, Nikolaus Adams, Matthias Meinke, Wolfgang Schröder, Jean-Christophe Loiseau, Esther Lagemann, Steven L. Brunton - [[pdf]](https://arxiv.org/pdf/2512.17534)</summary>

**Abstract:** Modeling and controlling fluid flows is critical for several fields of science and engineering, including transportation, energy, and medicine. Effective flow control can lead to, e.g., lift increase, drag reduction, mixing enhancement, and noise reduction. However, controlling a fluid faces several significant challenges, including high-dimensional, nonlinear, and multiscale interactions in space and time. Reinforcement learning (RL) has recently shown great success in complex domains, such as robotics and protein folding, but its application to flow control is hindered by a lack of standardized benchmark platforms and the computational demands of fluid simulations. To address these challenges, we introduce HydroGym, a solver-independent RL platform for flow control research. HydroGym integrates sophisticated flow control benchmarks, scalable runtime infrastructure, and state-of-the-art RL algorithms. Our platform includes 42 validated environments spanning from canonical laminar flows to complex three-dimensional turbulent scenarios, validated over a wide range of Reynolds numbers. We provide non-differentiable solvers for traditional RL and differentiable solvers that dramatically improve sample efficiency through gradient-enhanced optimization. Comprehensive evaluation reveals that RL agents consistently discover robust control principles across configurations, such as boundary layer manipulation, acoustic feedback disruption, and wake reorganization. Transfer learning studies demonstrate that controllers learned at one Reynolds number or geometry adapt efficiently to new conditions, requiring approximately 50% fewer training episodes. The HydroGym platform is highly extensible and scalable, providing a framework for researchers in fluid dynamics, machine learning, and control to add environments, surrogate models, and control algorithms to advance science and technology.

**arXiv ID:** 2512.17534
</details>

<details>
<summary><strong>PASS: Probabilistic Agentic Supernet Sampling for Interpretable and Adaptive Chest X-Ray Reasoning</strong> - Yushi Feng, Junye Du, Yingying Hong, Qifan Wang, Lequan Yu - [[pdf]](https://arxiv.org/pdf/2508.10501)</summary>

**Abstract:** Existing tool-augmented agentic systems are limited in the real world by (i) black-box reasoning steps that undermine trust of decision-making and pose safety risks, (ii) poor multimodal integration, which is inherently critical for healthcare tasks, and (iii) rigid and computationally inefficient agentic pipelines. We introduce PASS (Probabilistic Agentic Supernet Sampling), the first multimodal framework to address these challenges in the context of Chest X-Ray (CXR) reasoning. PASS adaptively samples agentic workflows over a multi-tool graph, yielding decision paths annotated with interpretable probabilities. Given the complex CXR reasoning task with multimodal medical data, PASS leverages its learned task-conditioned distribution over the agentic supernet. Thus, it adaptively selects the most suitable tool at each supernet layer, offering probability-annotated trajectories for post-hoc audits and directly enhancing medical AI safety. PASS also continuously compresses salient findings into an evolving personalized memory, while dynamically deciding whether to deepen its reasoning path or invoke an early exit for efficiency. To optimize a Pareto frontier balancing performance and cost, we design a novel three-stage training procedure, including expert knowledge warm-up, contrastive path-ranking, and cost-aware reinforcement learning. To facilitate rigorous evaluation, we introduce CAB-E, a comprehensive benchmark for multi-step, safety-critical, free-form CXR reasoning. Experiments across various benchmarks validate that PASS significantly outperforms strong baselines in multiple metrics (e.g., accuracy, LLM-Judge, semantic similarity, etc.) while balancing computational costs, pushing a new paradigm shift towards interpretable, adaptive, and multimodal medical agentic systems.

**arXiv ID:** 2508.10501
</details>

<details>
<summary><strong>DHP: Discrete Hierarchical Planning for Hierarchical Reinforcement Learning Agents</strong> - Shashank Sharma, Janina Hoffmann, Vinay Namboodiri - [[pdf]](https://arxiv.org/pdf/2502.01956)</summary>

**Abstract:** Hierarchical Reinforcement Learning (HRL) agents often struggle with long-horizon visual planning due to their reliance on error-prone distance metrics. We propose Discrete Hierarchical Planning (DHP), a method that replaces continuous distance estimates with discrete reachability checks to evaluate subgoal feasibility. DHP recursively constructs tree-structured plans by decomposing long-term goals into sequences of simpler subtasks, using a novel advantage estimation strategy that inherently rewards shorter plans and generalizes beyond training depths. In addition, to address the data efficiency challenge, we introduce an exploration strategy that generates targeted training examples for the planning modules without needing expert data. Experiments in 25-room navigation environments demonstrate a 100% success rate (vs. 90% baseline). We also present an offline variant that achieves state-of-the-art results on OGBench benchmarks, with up to 71% absolute gains on giant HumanoidMaze tasks, demonstrating our core contributions are architecture-agnostic. The method also generalizes to momentum-based control tasks and requires only log N steps for replanning. Theoretical analysis and ablations validate our design choices.

**arXiv ID:** 2502.01956
</details>

<details>
<summary><strong>Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning</strong> - Tong Wu, Yang Liu, Jun Bai, Zixia Jia, Shuyi Zhang, Ziyong Lin, Yanting Wang, Song-Chun Zhu, Zilong Zheng - [[pdf]](https://arxiv.org/pdf/2512.07461)</summary>

**Abstract:** We introduce Native Parallel Reasoner (NPR), a teacher-free framework that enables Large Language Models (LLMs) to self-evolve genuine parallel reasoning capabilities. NPR transforms the model from sequential emulation to native parallel cognition through three key innovations: 1) a self-distilled progressive training paradigm that transitions from ``cold-start'' format discovery to strict topological constraints without external supervision; 2) a novel Parallel-Aware Policy Optimization (PAPO) algorithm that optimizes branching policies directly within the execution graph, allowing the model to learn adaptive decomposition via trial and error; and 3) a robust NPR Engine that refactors memory management and flow control of SGLang to enable stable, large-scale parallel RL training. Across eight reasoning benchmarks, NPR trained on Qwen3-4B achieves performance gains of up to 24.5% and inference speedups up to 4.6x. Unlike prior baselines that often fall back to autoregressive decoding, NPR demonstrates 100% genuine parallel execution, establishing a new standard for self-evolving, efficient, and scalable agentic reasoning.

**arXiv ID:** 2512.07461
</details>

<details>
<summary><strong>GB-DQN: Gradient Boosted DQN Models for Non-stationary Reinforcement Learning</strong> - Chang-Hwan Lee, Chanseung Lee - [[pdf]](https://arxiv.org/pdf/2512.17034)</summary>

**Abstract:** Non-stationary environments pose a fundamental challenge for deep reinforcement learning, as changes in dynamics or rewards invalidate learned value functions and cause catastrophic forgetting. We propose \emph{Gradient-Boosted Deep Q-Networks (GB-DQN)}, an adaptive ensemble method that addresses model drift through incremental residual learning. Instead of retraining a single Q-network, GB-DQN constructs an additive ensemble in which each new learner is trained to approximate the Bellman residual of the current ensemble after drift. We provide theoretical results showing that each boosting step reduces the empirical Bellman residual and that the ensemble converges to the post-drift optimal value function under standard assumptions. Experiments across a diverse set of control tasks with controlled dynamics changes demonstrate faster recovery, improved stability, and greater robustness compared to DQN and common non-stationary baselines.

**arXiv ID:** 2512.17034
</details>

<details>
<summary><strong>Learning Safe Autonomous Driving Policies Using Predictive Safety Representations</strong> - Mahesh Keswani, Raunak Bhattacharyya - [[pdf]](https://arxiv.org/pdf/2512.17586)</summary>

**Abstract:** Safe reinforcement learning (SafeRL) is a prominent paradigm for autonomous driving, where agents are required to optimize performance under strict safety requirements. This dual objective creates a fundamental tension, as overly conservative policies limit driving efficiency while aggressive exploration risks safety violations. The Safety Representations for Safer Policy Learning (SRPL) framework addresses this challenge by equipping agents with a predictive model of future constraint violations and has shown promise in controlled environments. This paper investigates whether SRPL extends to real-world autonomous driving scenarios. Systematic experiments on the Waymo Open Motion Dataset (WOMD) and NuPlan demonstrate that SRPL can improve the reward-safety tradeoff, achieving statistically significant improvements in success rate (effect sizes r = 0.65-0.86) and cost reduction (effect sizes r = 0.70-0.83), with p < 0.05 for observed improvements. However, its effectiveness depends on the underlying policy optimizer and the dataset distribution. The results further show that predictive safety representations play a critical role in improving robustness to observation noise. Additionally, in zero-shot cross-dataset evaluation, SRPL-augmented agents demonstrate improved generalization compared to non-SRPL methods. These findings collectively demonstrate the potential of predictive safety representations to strengthen SafeRL for autonomous driving.

**arXiv ID:** 2512.17586
</details>

<details>
<summary><strong>Convergence Guarantees for Federated SARSA with Local Training and Heterogeneous Agents</strong> - Paul Mangold, Eloïse Berthier, Eric Moulines - [[pdf]](https://arxiv.org/pdf/2512.17688)</summary>

**Abstract:** We present a novel theoretical analysis of Federated SARSA (FedSARSA) with linear function approximation and local training. We establish convergence guarantees for FedSARSA in the presence of heterogeneity, both in local transitions and rewards, providing the first sample and communication complexity bounds in this setting. At the core of our analysis is a new, exact multi-step error expansion for single-agent SARSA, which is of independent interest. Our analysis precisely quantifies the impact of heterogeneity, demonstrating the convergence of FedSARSA with multiple local updates. Crucially, we show that FedSARSA achieves linear speed-up with respect to the number of agents, up to higher-order terms due to Markovian sampling. Numerical experiments support our theoretical findings.

**arXiv ID:** 2512.17688
</details>

<details>
<summary><strong>Cooperative Task Spaces for Multi-Arm Manipulation Control based on Similarity Transformations</strong> - Tobias Löw, Cem Bilaloglu, Sylvain Calinon - [[pdf]](https://arxiv.org/pdf/2510.26362)</summary>

**Abstract:** Many tasks in human environments require collaborative behavior between multiple kinematic chains, either to provide additional support for carrying big and bulky objects or to enable the dexterity that is required for in-hand manipulation. Since these complex systems often have a very high number of degrees of freedom coordinating their movements is notoriously difficult to model. In this article, we present the derivation of the theoretical foundations for cooperative task spaces of multi-arm robotic systems based on geometric primitives defined using conformal geometric algebra. Based on the similarity transformations of these cooperative geometric primitives, we derive an abstraction of complex robotic systems that enables representing these systems in a way that directly corresponds to single-arm systems. By deriving the associated analytic and geometric Jacobian matrices, we then show the straightforward integration of our approach into classical control techniques rooted in operational space control. We demonstrate this using bimanual manipulators, humanoids and multi-fingered hands in optimal control experiments for reaching desired geometric primitives and in teleoperation experiments using differential kinematics control. We then discuss how the geometric primitives naturally embed nullspace structures into the controllers that can be exploited for introducing secondary control objectives. This work, represents the theoretical foundations of this cooperative manipulation control framework, and thus the experiments are presented in an abstract way, while giving pointers towards potential future applications.

**arXiv ID:** 2510.26362
</details>

<details>
<summary><strong>LUMIA: A Handheld Vision-to-Music System for Real-Time, Embodied Composition</strong> - Chung-Ta Huang, Connie Cheng, Vealy Lai - [[pdf]](https://arxiv.org/pdf/2512.17228)</summary>

**Abstract:** Most digital music tools emphasize precision and control, but often lack support for tactile, improvisational workflows grounded in environmental interaction. Lumia addresses this by enabling users to "compose through looking"--transforming visual scenes into musical phrases using a handheld, camera-based interface and large multimodal models. A vision-language model (GPT-4V) analyzes captured imagery to generate structured prompts, which, combined with user-selected instrumentation, guide a text-to-music pipeline (Stable Audio). This real-time process allows users to frame, capture, and layer audio interactively, producing loopable musical segments through embodied interaction. The system supports a co-creative workflow where human intent and model inference shape the musical outcome. By embedding generative AI within a physical device, Lumia bridges perception and composition, introducing a new modality for creative exploration that merges vision, language, and sound. It repositions generative music not as a task of parameter tuning, but as an improvisational practice driven by contextual, sensory engagement.

**arXiv ID:** 2512.17228
</details>

<details>
<summary><strong>Digital Bricolage: Design Speculations for Embodied Approaches to Digitized Print-based Cultural Collections</strong> - Malak Sadek, Loraine Clarke, Stefania Forlini, Uta Hinrichs - [[pdf]](https://arxiv.org/pdf/2512.17590)</summary>

**Abstract:** COVID-related closures of public and academic libraries have underlined the importance of online platforms that provide access to digitized print-based collections. However, they also have highlighted the value of in-person handling of print artefacts for sensing and making sense of them. How do existing dominant digital platforms invite and/or discourage embodied forms of exploration and sense-making? What opportunities for embodied experience might we discover if we embrace the material qualities of print-based collections when designing interfaces for digital access? In this paper, we present findings from a speculative exercise where we invited creative professionals and experts in curating and handling access to collections to reflect on existing approaches to digitized print-based collections and to speculate about alternative design opportunities and modes of engagement. We argue for digital bricolage-a design approach that values working with materials that are "on hand" and embracing our ability to "handle" them in ways that foster both casual and curious exploration.

**arXiv ID:** 2512.17590
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
