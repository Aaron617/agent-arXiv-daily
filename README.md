# Agent arXiv Daily

**Last Updated:** 2026-06-08 06:06:51

**Total Papers:** 100

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>How AI Agents Reshape Knowledge Work: Autonomy, Efficiency, and Scope</strong> - Jeremy Yang, Kate Zyskowski, Noah Yonack, Jerry Ma - [[pdf]](https://arxiv.org/pdf/2606.07489)</summary>

**Abstract:** Frontier AI systems are bridging the gap between intelligence and utility by shifting from conversational assistants to autonomous agents that execute tasks end to end. Using production data from Perplexity's Search and Computer products, we study this transition by examining how AI agents accelerate and reshape knowledge work. Three key empirical findings emerge. First, using sessions with near-identical initial query pairs as natural experiments for the same underlying task attempted with both products, Computer performs 26 minutes of autonomous work per user session, versus 33 seconds for Search. Computer automates task decomposition and execution that Search users might otherwise manually orchestrate and implement. As a result, Computer shifts follow-up query distribution toward higher-order work such as verification and extension. Autonomy also increases execution quality, with per-query dissatisfaction rates 55% lower on Computer than on Search. Second, due to its autonomy advantage, Computer reduces completion time from 269 to 36 minutes on matched tasks, lowering estimated time and cost by 87% and 94%, respectively, compared to humans equipped with Search alone. Third, Computer changes the scope of work that users attempt: Computer queries more often cross occupational boundaries, require higher-order cognition, draw on broader expertise, take the form of composite tasks that bundle interdependent subtasks into a single query, and unlock work activities that are essentially absent from Search usage among the same users. Together, the evidence indicates that AI agents accelerate workflows, enhance output quality, reduce costs, and expand the breadth and depth of automated work.

**arXiv ID:** 2606.07489
</details>

<details>
<summary><strong>From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous Agent Interoperability</strong> - Bijaya Dangol - [[pdf]](https://arxiv.org/pdf/2606.07150)</summary>

**Abstract:** Agent-interoperability protocols such as A2A and MCP standardize what agents say to one another, but assume address-based transport over HTTP(S). Such transports protect message content, increasingly with end-to-end encryption. What they leave in the clear is the communication graph: which agent contacts which, when, and how often. In agent systems this graph is more consequential than a privacy framing suggests. Endpoints are often capability-labeled, workflows are structured and chained, and interactions are coupled to real actions, so an observer recovers more than past relationships. It can infer the pending workflow, the task being assembled and the action likely to follow. At machine speed, it can act on that inference before the workflow completes. The threat is therefore one of workflow integrity, not privacy alone: predictive leverage over autonomous action. We give a threat model for the agent communication graph; identify what makes agent metadata distinctively revealing (semanticity, prospectivity, actuation); define transport- and bootstrap-layer privacy properties and weigh candidate transports (SimpleX/SMP, Tor, mixnets) against them; and present an A2A case study in which a metadata-protecting binding is expressible but surfaces the protocol's identity assumptions. We test these on a generative model anchored to a real A2A capture. From passive metadata alone, with no payloads, a classifier recovers a task's class well above chance, from only the workflow's opening; applied together, the properties drive that recovery sharply back toward chance. Beyond what an observer can recover, we measure the leverage of acting on the leak: from a workflow's opening and under a fixed budget, an adversary choosing which workflows to act on realizes in this model most of a clairvoyant attacker's advantage over a metadata-blind one, and the same properties suppress it.

**arXiv ID:** 2606.07150
</details>

<details>
<summary><strong>Autonomous computational catalysis through an agentic research system</strong> - Honghao Chen, Jiangjie Qiu, Yi Shen Tew, Xiaonan Wang - [[pdf]](https://arxiv.org/pdf/2601.13508)</summary>

**Abstract:** Autonomous agents are beginning to transform scientific research from tool-assisted workflows toward self-sustaining discovery processes. Computational catalysis provides a representative challenge, as catalyst discovery requires high-level questions to be translated into coordinated model construction, atomistic simulation, mechanistic analysis, and iterative design across multiple scales. Here we introduce CatMaster, a catalysis-native agentic research system that recasts computational catalysis as a low-barrier virtual ecosystem for autonomous research. CatMaster maintains an evolving research state and extends capabilities through self-feedback across model construction, calculation, critique and catalyst-design decisions within one extensible environment. Across progressively challenging tasks, CatMaster converts natural-language requests into concrete computational studies, from essential atomistic modelling and standard calculations to mechanism exploration and closed-loop catalyst design. It showed robust execution in representative computational-catalysis scenarios and near-leading performance across selected MatBench tasks, with phonons scenario demonstrating its modelling self-evolution capability. In the independent CO2-to-CO catalyst design case, CatMaster used iterative self-critique and evidence refinement to identify competitive B-CoN4 and NiN3B/N-NiN3B motifs. These results establish a virtual-ecosystem paradigm in which AI agents move beyond simulation execution toward end-to-end computational research, providing a foundation for autonomous discovery in catalysis and materials science.

**arXiv ID:** 2601.13508
</details>

<details>
<summary><strong>From Pixels to Shelf: An Integrated Robotic System for Autonomous Supermarket Stocking with a Mobile Manipulator</strong> - Davide Peron, Victor Nan Fernandez-Ayala, Lukas Segelmark - [[pdf]](https://arxiv.org/pdf/2509.11740)</summary>

**Abstract:** Autonomous stocking in retail environments, particularly supermarkets, presents challenges due to dynamic human interactions, constrained spaces, and diverse product geometries. This paper introduces an efficient modular robotic system for autonomous shelf stocking, integrating commercially available hardware with a scalable algorithmic architecture. A major contribution of this work is the system integration of off-the-shelf hardware and ROS2-based perception, planning, and control into a single deployable platform for retail environments. Our solution leverages Behavior Trees (BTs) for task planning, fine-tuned vision models for object detection, and a two-step Model Predictive Control (MPC) framework for precise shelf navigation using ArUco markers. Laboratory experiments replicating realistic supermarket conditions demonstrate reliable performance, achieving over 98% success in pick-and-place operations across a total of more than 700 stocking events. However, our comparative benchmarks indicate that the performance and cost-effectiveness of current autonomous systems remain inferior to that of human workers, which we use to highlight key improvement areas and quantify the progress still required before widespread commercial deployment can realistically be achieved.

**arXiv ID:** 2509.11740
</details>

<details>
<summary><strong>Moodie: An Early-Stage Design Exploration for Supporting Fear of Missing Out with LLM-based Chatbots</strong> - Hsin-Yu Tsai, Jingxian Liao, Fu-Yin Cherng, Tzu-Hsiang Huang - [[pdf]](https://arxiv.org/pdf/2606.07231)</summary>

**Abstract:** The excessive use of social media has led to the challenge known as Fear of Missing Out (FoMO). Existing studies fail to provide accessible, interactive tools that focus on the emotional and cognitive aspects of FoMO. This work presents Moodie, a chatbot designed using Large Language Models to support emotion regulation and reduce FoMO. We conducted a formative study to understand the needs of individuals with FoMO and developed Moodie. Then, we conducted a preliminary evaluative study (N=21) to observe how participants interact with Moodie and a baseline chatbot (GPT-4o) over one week. The results show that while both Moodie and a baseline chatbot reduced FoMO to a similar extent, Moodie resulted in greater engagement and social connection. This finding raises interesting questions about the advantages of purpose-built chatbots compared to general-purpose models for mental health support. Future research will include chat log analysis, prototype refinements, and longitudinal evaluations.

**arXiv ID:** 2606.07231
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (21 papers)</h2></summary>

<details>
<summary><strong>Attack Selection in Agentic AI Control Evaluations Meaningfully Decreases Safety</strong> - Catherine Ge-Wang, Tyler Crosse, Benjamin Hadad IV, Joachim Schaeffer, Ram Potham, Tyler Tracy - [[pdf]](https://arxiv.org/pdf/2606.06529)</summary>

**Abstract:** An attacker that strategically chooses when to attack is much harder to catch than one that attacks indiscriminately. AI control is a safety framework for deploying capable but untrusted AI agents under the oversight of a weaker, trusted monitor and a limited human audit budget. Control evaluations stress-test these protocols by pitting a red-team attack policy against the blue-team monitor, but current evaluations typically assume attackers that do not strategically select when to attack. We study this capability, attack selection, in agentic settings by decomposing attack decisions into a start policy, which decides when an attacker should attack, and a stop policy, which decides when an attacker should abort an ongoing attack. Across two agentic settings, BashArena and LinuxArena, both policies substantially lower measured empirical safety without changing the underlying attack capability. At a 1% audit budget, our start policy reduces safety by 20pp on both BashArena and LinuxArena, and our stop policy reduces safety by 20pp on BashArena and 28pp on LinuxArena. These reductions should be interpreted as upper bounds on the effect of attack selection. Existing control evaluations may therefore yield overly optimistic safety estimates against selective attackers. We recommend that future evaluations, system cards, and safety cases elicit attack selection to produce more realistic safety estimates.

**arXiv ID:** 2606.06529
</details>

<details>
<summary><strong>Evidence-Based Intelligent Diagnostic and Therapeutic Visualization System with Large Language Models: Multi-Turn Interaction and Multimodal Treatment Plan Generation</strong> - Yunhan Wang, Yuda Wang, Zhiying Tu, Mingqiang Song, Li Song, Kun Li, Dianhui Chu, Bolin Zhang - [[pdf]](https://arxiv.org/pdf/2606.06869)</summary>

**Abstract:** Aim: Existing AI-assisted traditional Chinese medicine diagnostic tools suffer from opaque reasoning processes, passive interaction, and limited treatment plan presentation. This
study proposes a knowledge-enhanced visual diagnostic system to improve the transparency and interpretability of syndrome differentiation and treatment.
Methods: The system is built upon a Neo4j knowledge graph comprising 241 syndromes, 1,263 symptoms, and 2,485 relations. It incorporates a four-stage symptom matching pipeline
(exact, semantic, fuzzy, and large language model verification), an information gain-driven proactive questioning strategy optimized with genetic algorithms, and a multimodal
treatment presentation integrating artificial intelligence-generated illustrations, three-dimensional meridian-acupoint models, and evidence-based literature.
Results: Knowledge graph constraints reduced non-standard outputs by 32%. Case studies validated the effectiveness of the interactive workflow across patient self-assessment,
clinician-assisted diagnosis, and traditional Chinese medicine education. Automated paired-comparison evaluation across 30 cases further demonstrated significant improvements in
diagnostic trust (Cohen's d = 1.82, p < 0.001), reduced cognitive load (improvements in four of five dimensions), and higher credibility of evidence-based references (4.21 vs.
2.95).
Conclusions: The proposed system enhances the transparency of traditional Chinese medicine diagnostic reasoning and the interpretability of treatment plans through knowledge
graph-driven visualization and multimodal interaction, offering a practical solution for trustworthy artificial intelligence-assisted traditional Chinese medicine applications.

**arXiv ID:** 2606.06869
</details>

<details>
<summary><strong>Off-Policy Evaluation with Strategic Agents via Local Disclosure</strong> - Kiet Q. H. Vo, Abbavaram Gowtham Reddy, Julian Rodemann, Siu Lun Chau, Krikamol Muandet - [[pdf]](https://arxiv.org/pdf/2606.07308)</summary>

**Abstract:** We study off-policy evaluation (OPE) under strategic behavior where decision subjects (or agents) respond to a decision maker's policy by strategically modifying their covariates. Such behavior induces a policy-dependent covariate shift, breaking the standard assumption in existing methods that covariates are exogenous to the policy. Related work addresses this challenge by imposing strong assumptions such as repeated interactions or full knowledge of agents' response behavior, substantially limiting its applicability to OPE. In contrast, we consider a one-shot OPE setting where the decision maker has only partial knowledge of the agents' response behavior. Our key insight is that disclosing local information through post-hoc explanations reveals agents' pre-strategic covariates prior to adaptation, mitigating the information loss induced by strategic behavior. Leveraging this structure, we estimate a statistical model for the agents' responses and construct a doubly robust estimator for policy value. By assuming that the agents' cost sensitivity follows a conditional log-normal distribution, we establish consistency of the proposed estimator and validate our approach empirically. More broadly, our results highlight how interaction design can mitigate information asymmetry by revealing otherwise hidden structure in agents' strategic responses.

**arXiv ID:** 2606.07308
</details>

<details>
<summary><strong>Act As a Real Researcher: A Suite of Benchmarks Evaluating Frontier LLMs and Agentic Harnesses in Research Lifecycle</strong> - Jiayu Wang, Weijiang Lv, Bowen Fu, Jing Fu, Jiayi Song, Lingyu Zhang, Lanxuan Xue, Luodi Chen, Zepeng Xin, Kaiyu Li, Xiangyong Cao - [[pdf]](https://arxiv.org/pdf/2606.07462)</summary>

**Abstract:** As foundation models advance and agent scaffolding becomes increasingly sophisticated, agents have demonstrated remarkable proficiency in complex, long-horizon coding tasks and even autonomous experiment execution. Despite their evolution from research assistants into autonomous research agents, these systems still exhibit significant limitations in field sensitivity, research ethics, and nuanced scientific judgment. Consequently, frontier agents remain unable to fully replace human researchers. To bridge this gap, we conceptualize the AARR (Act As a Real Researcher) benchmark series. Unlike existing benchmarks that primarily assess macro-level execution capabilities, AARR focuses on whether agents can emulate the professionalism, thoroughness, and nuanced reasoning that characterize human researchers in granular research scenarios. In this work, we propose AARRI-Bench (Act As a Real Research Intern), the first benchmark in this series. We conduct extensive experiments across frontier models and agentic systems, revealing that even the best-performing configuration (Mini-SWE-Agent with Claude Opus 4.7) achieves only 68.3\% success rate, frequently overlooking subtle yet critical details that are obvious to real human researchers. Our results indicate that developing researcher-like AI requires further exploration of research behavior, rather than merely complex scaffolding. Our data is released at this https URL.

**arXiv ID:** 2606.07462
</details>

<details>
<summary><strong>Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests</strong> - Thanawat Lodkaew, Johannes Ackermann, Soichiro Nishimori, Nontawat Charoenphakdee, Masashi Sugiyama, Takashi Ishida - [[pdf]](https://arxiv.org/pdf/2606.07379)</summary>

**Abstract:** A growing failure mode in agent evaluation and training is that models can achieve high evaluation scores by exploiting shortcuts instead of solving the intended task, producing deceptive performance. This makes evaluation scores unreliable as measures of true task-solving ability. We propose CapCode, a framework for constructing coding datasets with randomized tests whose best achievable non-cheating performance is deliberately capped below one. This capped-performance design gives evaluation scores a clearer interpretation: scores substantially above the cap are implausible and therefore provide evidence of cheating. To prevent cheating, we propose CapReward, a reward design based on the CapCode principle to discourage optimization beyond the cap. Experiments across multiple datasets show that CapCode detects cheating while preserving performance ranking of models, and CapReward reduces cheating behavior, yielding models that better follow the intended task specification.

**arXiv ID:** 2606.07379
</details>

<details>
<summary><strong>Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills</strong> - Chuan Xiao, Zhengbo Jiao, Shaobo Wang, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang, Lin Qu - [[pdf]](https://arxiv.org/pdf/2606.07412)</summary>

**Abstract:** LLM-driven software engineering agents have become a central testbed for real-world language-model capability, yet their training remains limited by the availability of high-quality SWE tasks. Existing synthetic data methods typically create tasks through fixed mutation or bug-injection procedures, making the resulting distributions largely independent of the agent's own weaknesses and training progress. We introduce Socratic-SWE, a closed-loop self-evolution framework that reuses the agent's historical solving traces as a source of training signal. Rather than treating traces only as evidence for reward computation, Socratic-SWE distills them into structured agent skills that summarize recurring failures and effective repair patterns. These skills then guide the generation of targeted repair tasks in real repositories. Candidate tasks are checked through execution-based validation and scored with a solver-gradient alignment reward, so that the retained tasks are both verifiable and useful for improving the Solver. The updated Solver produces new traces, enabling the task curriculum to adapt over successive rounds. Across SWE-bench Verified, SWE-bench Lite, SWE-bench Pro, and Terminal-Bench 2.0, Socratic-SWE consistently improves over self-evolving baselines under the same compute budget, reaching 50.40% on SWE-bench Verified after three iterations. These results suggest that solving traces can serve as a scalable substrate for self-evolving SWE agents.

**arXiv ID:** 2606.07412
</details>

<details>
<summary><strong>Planning-aligned Token Compression for Long-Context Autonomous Driving</strong> - Zhixuan Liang, Yuxiao Chen, Yurong You, Peter Karkus, Wenhao Ding, Boyi Li, Alexander Popov, Yan Wang, Maximilian Igl, Yiming Li, Danfei Xu, Nikolai Smolyanskiy, Boris Ivanovic, Ping Luo, Marco Pavone - [[pdf]](https://arxiv.org/pdf/2606.07464)</summary>

**Abstract:** Monolithic vision-action models represent an emerging paradigm in autonomous driving. However, this architecture produces token sequences that quickly exceed real-time computational budgets when encoding extended temporal context for complex interactions. While approaches like linear transformers and external memory try to make the context lightweight, token compression is most compatible with the architecture as it requires no backbone modifications. Yet existing compression adopts rule-based heuristics like temporal decay, decoupled from planning, risking loss of decision-critical information. We propose COMPACT-VA, a planning-aligned working memory framework built on conditional VQ-VAE, compressing extended context into bounded representations. Compression is conditioned on both historical trajectory and a learned planning intent that the posterior encoder distills from future trajectories during training, while the prior encoder learns to predict it from compressed observations. The compressed memory, concatenated with the predicted latent, feeds the policy for end-to-end optimization, planning with retained decision-critical information. We evaluate on high-signal dynamic scenarios where historical context is most critical for behavior correctness (e.g., stop, yield, or proceed), and accordingly design behavioral metrics. Under comparable token budgets, we achieve $>$6% improvement (68.3%) on success rates with consistent gains across metrics. Ablations validate planning-aligned coupling effectiveness. Closed-loop evaluation confirms that COMPACT-VA maintained general driving performance with 3.3* speedup and 2.7* memory reduction over uncompressed processing.

**arXiv ID:** 2606.07464
</details>

<details>
<summary><strong>MemDreamer: Decoupling Perception and Reasoning for Long Video Understanding via Hierarchical Graph Memory and Agentic Retrieval Mechanism</strong> - Cong Chen, Guo Gan, Kaixiang Ji, ChaoYang Zhang, Zhen Yang, Guangming Yao, Hao Chen, Jingdong Chen, Yi Yuan, Chunhua Shen - [[pdf]](https://arxiv.org/pdf/2606.07512)</summary>

**Abstract:** Current Vision-Language Models struggle with hours-long videos because processing full-length visual sequences induces prohibitive token explosion and attention dilution. To overcome this, we introduce MemDreamer to decouple perception and reasoning, shifting long-video understanding into an agentic exploration process. As a plug-and-play framework, it incrementally streams videos to construct a Hierarchical Graph Memory, a top-down three-tier architecture for semantic abstraction, anchored by a foundational graph capturing spatiotemporal and causal relations. During inference, the reasoning model employs agentic tool-augmented retrieval, navigating hierarchies, searching nodes, and traversing logical edges via an Observation-Reason-Action loop. Experiments show MemDreamer achieves SOTA results across four mainstream benchmarks, narrowing the gap with human experts to only 3.7 points. It constrains the reasoning context window to merely 2% of full-context ingestion while delivering a 12.5 point absolute accuracy gain. Furthermore, statistical analysis uncovers a strong positive linear correlation between an VLM's performance on logic reasoning and long-video understanding benchmarks, establishing agentic capability scaling as a new paradigm for multimodal comprehension.

**arXiv ID:** 2606.07512
</details>

<details>
<summary><strong>SentinelBench: A Benchmark for Long-Running Monitoring Agents</strong> - Matheus Kunzler Maldaner, Adam Fourney, Amanda Swearngin, Hussein Mozannar, Gagan Bansal, Maya Murad, Rafah Hosn, Saleema Amershi - [[pdf]](https://arxiv.org/pdf/2606.05342)</summary>

**Abstract:** AI agents are increasingly asked to carry out work that spans minutes, hours, or longer. Yet the default model of agent behavior is continuous action: issuing tool calls, refreshing pages, searching for alternatives, or otherwise trying to force progress. This is the wrong approach for many long-running tasks, which are better served by a strategy of sustained attention. Instead, agents should monitor an environment, notice when an external event makes progress possible, then respond promptly without wasting resources while waiting. To measure progress on this class of tasks, we introduce SentinelBench, an open-source benchmark for time-evolving monitoring tasks.
SentinelBench contains 100 tasks across 10 synthetic web environments, including email, calendars, finance, professional networking, and entertainment. Each environment exposes a live web interface and replays a scripted sequence of events, requiring agents to navigate and reason about web pages whose state shifts underfoot. SentinelBench measures task completion, reaction time, and resource use, exposing the tradeoff between responsiveness and cost. We report results across three models and two browser-agent harnesses, establishing performance baselines for future comparison and demonstrating how agent design choices can dramatically impact key metrics. Together, these results show that SentinelBench distinguishes meaningful differences in agent behavior.

**arXiv ID:** 2606.05342
</details>

<details>
<summary><strong>SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents</strong> - Wenxuan Wang, Haoyu Sun, Fukuan Hou, Mingyang Song, Weinan Zhang, Yu Cheng, Yang Yang - [[pdf]](https://arxiv.org/pdf/2606.05761)</summary>

**Abstract:** Persistent AI assistants, such as OpenClaw, accumulate large collections of related memories over long-term interactions. As these memories grow, they may reinforce one another, diverge across contexts, or directly conflict, making correct assistance depend on memory relations rather than isolated recall. Existing long-term memory benchmarks rarely probe how agents preserve and utilize such relations during downstream tasks. To address this gap, we introduce SubtleMemory, a benchmark for fine-grained relational memory discrimination in long-running AI agents. SubtleMemory constructs relation-controlled latent semantic artifacts whose variants instantiate complementary, nuanced, or contradictory relations, and embeds them into realistic user-agent histories, requiring agents to recover distributed relational structures during later queries and instructions. The benchmark contains 1,522 evaluation instances over 10 long histories, grounded in 1,090 relation-controlled memory-variant sets and spanning user-related and non-user-related queries. Evaluating six standalone memory systems, two Claw-style agents with native memory modules, and three Claw-style agents with plugin memory modules, we find that current systems remain weak on fine-grained relational memory discrimination. We further introduce diagnostic protocols that reveal distinct capability profiles across memory preservation, retrieval, and downstream reasoning stages.

**arXiv ID:** 2606.05761
</details>

<details>
<summary><strong>Measuring Agents in Production</strong> - Melissa Z. Pan, Negar Arabzadeh, Riccardo Cogo, Yuxuan Zhu, Alexander Xiong, Lakshya A Agrawal, Huanzhi Mao, Emma Shen, Sid Pallerla, Liana Patel, Shu Liu, Tianneng Shi, Xiaoyuan Liu, Jared Quincy Davis, Emmanuele Lacavalla, Alessandro Basile, Shuyi Yang, Paul Castro, Daniel Kang, Koushik Sen, Dawn Song, Joseph E. Gonzalez, Ion Stoica, Matei Zaharia, Marquita Ellis - [[pdf]](https://arxiv.org/pdf/2512.04123)</summary>

**Abstract:** LLM-based agents already operate in production across many industries, yet we lack an understanding of what technical methods make deployments successful. We present the first systematic study of Measuring Agents in Production, MAP, using first-hand data from agent developers. We conducted 20 case studies via in-depth interviews and surveyed 86 deployed systems practitioners across 26 domains. We investigate why organizations build agents, how they build them, how they evaluate them, and their top development challenges. Our study finds that production agents are built using simple, controllable approaches: 68% execute at most 10 steps before human intervention, 70% rely on prompting off-the-shelf models instead of weight tuning, and 74% depend primarily on human evaluation. Reliability (consistent correct behavior over time) remains the top development challenge, which practitioners currently address through systems-level design. MAP documents the current state of production agents, providing the research community with visibility into deployment realities and underexplored research avenues.

**arXiv ID:** 2512.04123
</details>

<details>
<summary><strong>It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents</strong> - Karolina Korgul, Yushi Yang, Arkadiusz Drohomirecki, Piotr Błaszczyk, Will Howard, Lukas Aichberger, Chris Russell, Philip H.S. Torr, Adam Mahdi, Adel Bibi - [[pdf]](https://arxiv.org/pdf/2512.23128)</summary>

**Abstract:** Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), a benchmark for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25% of tasks on average (13% for GPT-5 to 43% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.

**arXiv ID:** 2512.23128
</details>

<details>
<summary><strong>ScenicRules: An Autonomous Driving Benchmark with Multi-Objective Specifications and Abstract Scenarios</strong> - Kevin Kai-Chun Chang, Ekin Beyazit, Alberto Sangiovanni-Vincentelli, Tichakorn Wongpiromsarn, Sanjit A. Seshia - [[pdf]](https://arxiv.org/pdf/2602.16073)</summary>

**Abstract:** Developing autonomous driving systems for complex traffic environments requires balancing multiple objectives, such as avoiding collisions, obeying traffic rules, and making efficient progress. In many situations, these objectives cannot be satisfied simultaneously, and explicit priority relations naturally arise. Also, driving rules require context, so it is important to formally model the environment scenarios within which such rules apply. Existing benchmarks for evaluating autonomous vehicles lack such combinations of multi-objective prioritized rules and formal environment models. In this work, we introduce ScenicRules, a benchmark for evaluating autonomous driving systems in stochastic environments under prioritized multi-objective specifications. We first formalize a diverse set of objectives to serve as quantitative evaluation metrics. Next, we design a Hierarchical Rulebook framework that encodes multiple objectives and their priority relations in an interpretable and adaptable manner. We then construct a compact yet representative collection of scenarios spanning diverse driving contexts and near-accident situations, formally modeled in the Scenic language. Experimental results show that our formalized objectives and Hierarchical Rulebooks align well with human driving judgments and that our benchmark effectively exposes agent failures with respect to the prioritized objectives. Our benchmark can be accessed at this https URL.

**arXiv ID:** 2602.16073
</details>

<details>
<summary><strong>EvoClaw: Evaluating AI Agents on Continuous Software Evolution</strong> - Gangda Deng, Zhaoling Chen, Zhongming Yu, Haoyang Fan, Yuhong Liu, Yuxin Yang, Dhruv Parikh, Rajgopal Kannan, Le Cong, Mengdi Wang, Qian Zhang, Viktor Prasanna, Xiangru Tang, Xingyao Wang - [[pdf]](https://arxiv.org/pdf/2603.13428)</summary>

**Abstract:** With AI agents increasingly deployed as long-running systems, it becomes essential to autonomously construct and continuously evolve customized software to enable interaction within dynamic environments. Yet, existing benchmarks evaluate agents on isolated, one-off coding tasks, neglecting the temporal dependencies and technical debt inherent in real-world software evolution. To bridge this gap, we introduce DeepCommit, an agentic pipeline that reconstructs verifiable Milestone DAGs from noisy commit logs, where milestones are defined as functionally cohesive development goals. These executable sequences enable EvoClaw, a novel benchmark that requires agents to sustain system integrity and limit error accumulation, dimensions of long-term software evolution largely missing from current benchmarks. Our evaluation of 12 frontier models across 4 agent frameworks reveals a critical vulnerability: overall performance scores drop significantly from >80% on isolated tasks to at most 38% in continuous settings, exposing agents' profound struggle with long-term maintenance and error propagation.

**arXiv ID:** 2603.13428
</details>

<details>
<summary><strong>MADE: Beyond Scoring via a Multilingual Agentic Diagnosing Engine for Fine-Grained Evaluation Insights</strong> - Yilun Liu, Miao Zhang, Shimin Tao, Minggui He, Chunguang Zhao, Chenxin Liu, Li Zhang, Chen Liu, Cheng Qian, Liqun Deng, Xiaojun Meng, Daimeng Wei - [[pdf]](https://arxiv.org/pdf/2606.07020)</summary>

**Abstract:** Multilingual and multicultural benchmarks now cover dozens of languages and model families, but the resulting score landscapes remain metric-rich and insight-poor, necessitating fine-grained multilingual post-evaluation diagnosis. However, single LLMs and open-ended agents are easily swamped by the long, noisy diagnostic input, and no reusable taxonomy exists for it. To address this, we propose MADE, a Multilingual Agentic Diagnosing Engine that decomposes post-evaluation analysis into planning, aggregate analysis, instance-level case inspection, multilingual and cultural reflection, and grounded report synthesis. MADE is paired with an expert-led 54-query and 15-language diagnostic set, evaluated on top of a large-scale multilingual evaluation substrate (33 model families, 11 benchmarks, 26 languages, 34 cultures, 8.66M evaluation records). Experiments show that MADE outperforms the strongest shared baseline by 47% in diagnosis report quality and is preferred by human multilingual experts in 87.9% of pairwise comparisons. Applied with multilingual experts, MADE further surfaces four actionable findings on deployment, iteration, and cross-cultural pitfalls, turning benchmark score tables into model-selection and remediation guidance.

**arXiv ID:** 2606.07020
</details>

<details>
<summary><strong>M$^3$Exam: Benchmarking Multimodal Memory for Realistic User-Agent Interactions</strong> - Zhengjun Huang, Wenxuan Liu, Zhoujin Tian, Wei Chen, Junle Chen, Yuqian Wu, Fangyuan Zhang, Qintian Guo, Xiaofang Zhou - [[pdf]](https://arxiv.org/pdf/2606.07402)</summary>

**Abstract:** Language agents are increasingly deployed over accumulating multimodal information, yet existing benchmarks assume a human-human form with sparse visuals and straightforward content, evaluating neither reasoning over authentic multimodal file interaction nor the interpretation of concealed user information. We therefore introduce M$^3$Exam, a query-centric multimodal conversational memory benchmark built on realistic user-agent interaction, with multi-dimensional evaluation spanning cross-modal grounding and implicit information inference. Benchmarking MLLMs and memory systems reveals persistent gaps in cross-modal grounding, cross session reasoning, and the efficiency cost of accumulating multimodal context. We further propose M$^3$Proctor, a multimodal memory method that detects query modality bias and consumes raw visual sources only on demand, improving accuracy by 13% while cutting index-construction time and retrieved tokens by over 70%.

**arXiv ID:** 2606.07402
</details>

<details>
<summary><strong>SWE-Explore: Benchmarking How Coding Agents Explore Repositories</strong> - Shaoqiu Zhang, Yuhang Wang, Jialiang Liang, Yuling Shi, Wenhao Zeng, Maoquan Wang, Shilin He, Ningyuan Xu, Siyu Ye, Kai Cai, Xiaodong Gu - [[pdf]](https://arxiv.org/pdf/2606.07297)</summary>

**Abstract:** Repository-level coding benchmarks such as SWE-bench have driven a rapid surge in the capabilities of coding agents. Yet they usually treat coding tasks as a holistic, binary prediction problem (e.g., resolved or unresolved), neglecting fine-grained agent capabilities such as repository understanding, context retrieval, code localization, and bug diagnosis. In this paper, we introduce SWE-Explore, a benchmark that isolates the evaluation of repository exploration, a critical capability of coding agents. Given a repository and an issue, SWE-Explore asks an explorer to return a ranked list of relevant code regions under a fixed line budget. SWE-Explore covers 848 issues across 10 programming languages and 203 open-source repositories. For each instance, we derive line-level ground truth from independent agent trajectories that successfully solved the same issue, distilling the specific code regions their solution paths actually consulted. We evaluate exploration along coverage, ranking, and context-efficiency dimensions, showing that these metrics strongly track downstream repair behavior. Across a broad set of retrieval methods, general coding agents, and specialized localizers, we find that agentic explorers form a clear tier above classical retrieval. While file-level localization is already strong for modern methods, line-level coverage and efficient ranking remain the key axes differentiating state-of-the-art explorers.

**arXiv ID:** 2606.07297
</details>

<details>
<summary><strong>RealClawBench: Live OpenClaw Benchmarks from Real Developer-Agent Sessions</strong> - Zongwei Lv, Zhewen Tan, Yaoming Li, Yilun Yao, Yuxuan Tian, Lin Sun, Xiangzheng Zhang, Weihong Lin, Tong Yang, Guangxiang Zhao - [[pdf]](https://arxiv.org/pdf/2606.03889)</summary>

**Abstract:** Agent benchmarks should reflect what users actually ask deployed agents to do, yet existing benchmarks often miss key realism properties of real developer-agent sessions. We introduce RealClawBench, a live benchmark framework built from real OpenClaw sessions to capture the distribution, diversity, and real-world difficulty of deployed agent use. Real user requests are challenging to benchmark because they often depend on local execution environments, involve implicit or underspecified intent, and require nontrivial verification. RealClawBench addresses these challenges with two core mechanisms: reconstructed execution environments and deterministic verifiable scorers, which together convert real sessions into reproducible, automatically scored tasks. The resulting release contains 281 executable tasks sampled from a much larger real-session pool while preserving the source distribution, with maximum final-vs-source Jensen-Shannon divergence of 0.0448. Evaluating 14 contemporary models shows that the best system solves only 65.8% of tasks, revealing substantial headroom on realistic developer-agent workloads. By turning real deployed sessions into controlled evaluation instances, RealClawBench provides a practical path toward benchmarks that better measure agent capability in actual use. Code is available at:this https URL.

**arXiv ID:** 2606.03889
</details>

<details>
<summary><strong>IDDMBSE: Integrating Data-Driven and Model-Based Systems Engineering for Trusted Autonomous Cyber-Physical Systems</strong> - John S. Baras, Sai Sandeep Damera, Ryan Matheu, Clinton Enwerem, Praveen M.S. Kumar - [[pdf]](https://arxiv.org/pdf/2606.06727)</summary>

**Abstract:** Autonomous cyber-physical systems (CPS) sit at the intersection of Model-Based Systems Engineering (MBSE) and data-driven Machine Learning and Artificial Intelligence (ML/AI), yet no integrated Systems Engineering (SE) methodology natively spans both. We address this gap with IDDMBSE, an Integrated Data-Driven and Model-Based Systems Engineering methodology that extends the rigorous MBSE V-process with a data-driven loop at every step, anchored in SysML, the autonomy stack, and a hybrid model-based plus data-driven trade-off architecture. We instantiate IDDMBSE as an interoperable, open-source tool chain: PERFECT, which maps SysML system architectures to executable ROS autonomy stacks for scalable performance evaluation; TRADES-X, which decomposes design-space exploration into a model-based optimization stage followed by a data-driven evaluation stage; and VERITAS, which combines formal, data-driven, and runtime verification into a single assurance workflow. We demonstrate IDDMBSE on a Trusted Autonomous Ground Robot across its development lifecycle, spanning sensor-suite selection, risk-sensitive path planning, behavior-tree task verification, conformal-prediction-based robust perception, and assured multi-robot coordination, all exercised in a contested-terrain Isaac Sim test range that we release with the tool chain. We close by sketching how IDDMBSE is being re-formulated on SysML v2 / KerML foundations to enable language-native composability and tighter ML/AI integration.

**arXiv ID:** 2606.06727
</details>

<details>
<summary><strong>Test-Time Trajectory Optimization for Autonomous Driving</strong> - Yihong Xu, Eloi Zablocki, Yuan Yin, Elias Ramzi, Ellington Kirby, Alexandre Boulch, Matthieu Cord - [[pdf]](https://arxiv.org/pdf/2606.07170)</summary>

**Abstract:** End-to-end planners for autonomous driving typically generate a set of candidate trajectories, score each one, and return the highest-scoring candidate. However, the scorer is applied only after the proposals are generated and cannot influence the set of trajectories: a weak set of candidates limits planning performance regardless of the scorer's quality. We instead treat the scorer as a learned trajectory-level reward function and search for trajectories that maximize it. Our method, TOAD, runs the Cross-Entropy Method at test time, warm-started from the planner's proposals. It requires no retraining and is plug-and-play for existing planners. Across six base planners, TOAD improves results on NAVSIM-v1 (94.7 PDMS), NAVSIM-v2 (56.3 EPDMS), and the closed-loop HUGSIM benchmark. The code will be made publicly available via the project page: this https URL.

**arXiv ID:** 2606.07170
</details>

<details>
<summary><strong>PathPainter: Transferring the Generalization Ability of Image Generation Models to Embodied Navigation</strong> - Yijin Wang, Yuru Tian, Xijie Huang, Weiqi Gai, Mo Zhu, Xin Zhou, Yuze Wu, Fei Gao - [[pdf]](https://arxiv.org/pdf/2605.07496)</summary>

**Abstract:** Bird's-eye-view (BEV) images have been widely demonstrated to provide valuable prior information for navigation. Given the global information provided by such views, two key challenges remain: how to fully exploit this information and how to reliably use it during execution. In this paper, we propose a navigation system that uses BEV images as global priors and is designed for ground and near-ground robotic platforms. The system employs an image generation model to interpret human intent from natural language, identify the target destination, and generate traversability masks. During execution, we introduce cross-view localization to align the robot's odometry with the BEV map and mitigate long-term drift in conventional odometry. We conduct extensive benchmark experiments to evaluate the proposed method and further validate it on a UAV platform. Using only a conventional local motion planner, the UAV successfully completes a 160-meter outdoor long-range navigation task. This work demonstrates how the world-understanding capabilities of foundation models can be transferred to embodied navigation, enabling robots to benefit from the strong generalization ability of existing image generation models.

**arXiv ID:** 2605.07496
</details>

</details>

<details open>
<summary><h2>LLM Agents (12 papers)</h2></summary>

<details>
<summary><strong>OpenSkill: Open-World Self-Evolution for LLM Agents</strong> - Zhiling Yan, Dingjie Song, Hanrong Zhang, Wei Liang, Yuxuan Zhang, Yutong Dai, Lifang He, Philip S. Yu, Ran Xu, Xiang Li, Lichao Sun - [[pdf]](https://arxiv.org/pdf/2606.06741)</summary>

**Abstract:** Self-evolving agents requires adaptation after deployment, but existing approaches assume a usable learning loop, such as curated skills, successful trajectories, or verifier signals. Real open-world deployments may provide none of these, offering only a task prompt. In this work, we study open-world self-evolution, where an agent must build both its skills and its own verification signals from scratch, using open-world resources but no target-task supervision. We propose OpenSkill, a framework that bootstraps this loop: it acquires grounded knowledge and verification anchors from documentation, repositories, and the web, synthesizes them into transferable skills, and refines those skills against self-built virtual tasks grounded in the anchors rather than in target answers. The open world thus supplies both the knowledge to be learned and a supervision-independent practice environment, with target-task supervision reserved for final evaluation. Across three benchmarks and two target agents, OpenSkill attains the best automated pass rate while satisfying the no-supervision constraint. Analysis shows its skills transfer across models without model-specific adaptation, and its self-built verifier aligns with ground-truth outcomes despite never accessing them.

**arXiv ID:** 2606.06741
</details>

<details>
<summary><strong>PandaAI: A Practical Agent CQ2 for Neuro-symbolic Data Analysis And Integrated Decision-Making in Quantitative Finance</strong> - Yuqi Li, Siyuan Liu, Bingjun Liu - [[pdf]](https://arxiv.org/pdf/2606.06823)</summary>

**Abstract:** While deep learning has excelled in various domains, its application to sequential decision-making in finance remains challenging due to the low Signal-to-Noise Ratio (SNR) and non-stationarity of financial data. Leveraging the reasoning capabilities of Large Language Models (LLMs), we propose \textbf{PandaAI}, a closed-loop neuro-symbolic LLM agent with market regime modeling and constrained alpha generation, which bridges general LLM reasoning with financial rigor and suppresses the financial toxicity of LLM-generated outputs. To bridge the gap between general linguistic capability and financial rigor, we fine-tune a domain-specific LLM. Furthermore, we integrate this LLM into a modular architecture and form a closed-loop system. Unlike traditional models that optimize isolated prediction metrics, \textbf{PandaAI} is designed as a neuro-symbolic agent that navigates the complex, real-world financial environment with explicit risk awareness. Extensive experiments on CSI 300 stock data show that \textbf{PandaAI} achieves a $18.2\%$ higher Rank IC and $25.7\%$ lower maximum drawdown than state-of-the-art time-series models. Our constrained LLM generation and dual-channel adaptation method provide a general paradigm for LLM deployment in high-stakes sequential decision-making scenarios.

**arXiv ID:** 2606.06823
</details>

<details>
<summary><strong>LLM Agent-Assisted Reverse Engineering with Quantitative Readability Metrics</strong> - Neil Archibald, Ruben Thijssen - [[pdf]](https://arxiv.org/pdf/2606.06838)</summary>

**Abstract:** Automatic decompilers produce functionally correct but often unreadable C code. This paper addresses one stage of the reverse engineering workflow: improving the readability of decompiled code using LLM agents guided by quantitative metrics.
We present a three-phase research evolution. Phase 1 (tool-driven steering via Ghidra MCP) suffered from incomplete coverage and inconsistent improvements due to lack of quantitative guidance. Phase 2 (structural similarity validation alone) revealed that agents optimize for metrics in unintended ways, producing structurally equivalent but less readable code. Our contribution is the Quantitative Readability Score (QRS) framework, a composite metric combining a structural similarity gate with three independent readability sub-metrics (Lexical Surprisal, Structural Simplicity, and Idiomatic Quality).
We demonstrate that QRS-guided refinement enables LLM agents to make targeted readability improvements without sacrificing correctness. We provide a discussion of the broader reverse engineering workflow (binary lifting, decompilation cleanup, and achieving functional equivalence) as context, however, it remains out of scope.

**arXiv ID:** 2606.06838
</details>

<details>
<summary><strong>TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents</strong> - Vijitha Mittapalli, Shreyaa Jayant Dani, Satya Srujana Pilli, Snigdha Ansu, Mohammadreza Teymoorianfard, Franck Dernoncourt, Hongjie Chen, Yu Wang, Ryan A. Rossi, Nesreen K. Ahmed - [[pdf]](https://arxiv.org/pdf/2606.07054)</summary>

**Abstract:** Autonomous LLM agents can pursue hidden malicious objectives through sequences of individually benign actions, making sabotage difficult to detect using standard trajectory-level monitoring. Existing approaches either evaluate complete trajectories in a single pass or partition them into independently scored windows, limiting their ability to connect evidence across temporally distant actions. We propose TRACE, a monitoring framework for long-horizon LLM agent trajectories. TRACE operates through a TIJ (Triage-Inspect-Judge) loop that identifies high-signal regions, performs targeted inspection while maintaining accumulated evidence across reasoning steps, and synthesizes a trajectory-level verdict. We evaluate TRACE on ten task domains from SHADE-Arena against state-of-the-art baselines. TRACE achieves an aggregate F1 of 0.713 and recall of 0.844, with the largest gains on tasks requiring long-range evidence linking.

**arXiv ID:** 2606.07054
</details>

<details>
<summary><strong>Small Language Model Agents Enable Efficient and High-Quality Knowledge Mining</strong> - Sipeng Zhang, Shuhuai Lin, Xinpeng Wei, Yihang Chen, Pin Qian, Su Wang, Huan Xu - [[pdf]](https://arxiv.org/pdf/2510.01427)</summary>

**Abstract:** At the core of Deep Research is knowledge mining, the task of extracting structured information from massive unstructured text in response to user instructions. Large language models (LLMs) excel at interpreting such instructions but are prohibitively expensive to deploy at scale, while traditional pipelines of classifiers and extractors remain efficient yet brittle and unable to generalize to new tasks. We introduce Falconer, a collaborative framework that combines the agentic reasoning of LLMs with lightweight proxy models for scalable knowledge mining. In Falconer, LLMs act as planners, decomposing user instructions into executable pipelines, and as annotators, generating supervision to train small proxies. The framework unifies classification and extraction into two atomic operations, get label and get span, enabling a single instruction-following model to replace multiple task-specific components. To evaluate the consistency between proxy models incubated by Falconer and annotations provided by humans and large models, we construct new benchmarks covering both planning and end-to-end execution. Experiments show that Falconer closely matches state-of-the-art LLMs in instruction-following accuracy while reducing inference cost by up to 90% and accelerating large-scale knowledge mining by more than 20x, offering an efficient and scalable foundation for Deep Research.

**arXiv ID:** 2510.01427
</details>

<details>
<summary><strong>Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates</strong> - Yibo Li, Zijie Lin, Ailin Deng, Xuan Zhang, Yufei He, Shuo Ji, Tri Cao, Bryan Hooi - [[pdf]](https://arxiv.org/pdf/2601.18510)</summary>

**Abstract:** While Large Language Model (LLM) agents excel at general tasks, they inherently struggle with continual adaptation due to the frozen weights after deployment. Conventional reinforcement learning (RL) offers a solution but incurs prohibitive computational costs and the risk of catastrophic forgetting. We introduce Just-In-Time Reinforcement Learning (JitRL), a training-free framework that enables test-time policy optimization without any gradient updates. JitRL maintains a dynamic, non-parametric memory of experiences and retrieves relevant trajectories to estimate action advantages on-the-fly. These estimates are then used to directly modulate the LLM's output logits. We theoretically prove that this additive update rule is the exact closed-form solution to the KL-constrained policy optimization objective. Extensive experiments on WebArena and Jericho demonstrate that JitRL establishes a new state-of-the-art among training-free methods. Crucially, JitRL outperforms the performance of computationally expensive fine-tuning methods (e.g., WebRL) while reducing monetary costs by over 30 times, offering a scalable path for continual learning agents. The code is available at this https URL.

**arXiv ID:** 2601.18510
</details>

<details>
<summary><strong>Tree-of-Experience: A Structured Experience-Management Solution for Self-Evolving Agents under Low-Repetition and Implicit-Reward Environments</strong> - Zihao Deng, Yining Zhu, Leiming Wang, Jingfei Lu, Junbo Wang, Chuncheng Ran, Yu Yang, Dixuan Yang, Jikun Shen - [[pdf]](https://arxiv.org/pdf/2606.06960)</summary>

**Abstract:** Experience-based self-evolution is crucial for LLM agents, but existing benchmarks often assume explicit goals, stable task patterns, and clear feedback. We study a more challenging setting: low-repetition tasks with implicit rewards, where past experience is difficult to reuse and feedback is delayed, noisy, and outcome-level. We introduce \textsc{FinEvolveBench}, a temporally controlled benchmark for financial sentiment prediction that links daily news-driven predictions to future excess returns. We further propose Tree-of-Experience (ToE), a structured experience-management method that organizes, retrieves, validates, and updates agent experience. Experiments show that general-purpose experience mechanisms do not consistently outperform no-experience baselines, while ToE achieves stronger overall performance. These results highlight the importance of structured experience management for self-evolving agents in implicit-reward environments.

**arXiv ID:** 2606.06960
</details>

<details>
<summary><strong>AutoTool: Dynamic Tool Selection and Integration for Agentic Reasoning</strong> - Jiaru Zou, Ling Yang, Yunzhe Qi, Sirui Chen, Mengting Ai, Ke Shen, Jingrui He, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2512.13278)</summary>

**Abstract:** Agentic reinforcement learning has advanced large language models (LLMs) to reason through long chain-of-thought trajectories while interleaving external tool use. Existing approaches assume a fixed inventory of tools, which limits the adaptability of LLM agents to new or evolving toolsets. We present AutoTool, a training framework that equips LLM agents with dynamic tool-selection capabilities throughout their reasoning trajectories. AutoTool employs a dual-phase optimization pipeline: (i) SFT and RL-based trajectory stabilization for coherent reasoning, and (ii) KL-regularized Plackett-Luce Ranking to refine consistent multi-step tool selection. We further build a 200k dataset with explicit tool-selection rationales across 1,000+ tools and 100+ tasks spanning mathematics, science, code generation, and multimodal reasoning. Across ten diverse benchmarks, we train two base models, Qwen3-8B and Qwen2.5-VL-7B, with AutoTool. With fewer parameters, AutoTool consistently outperforms advanced LLM agents and tool-integration methods, yielding average gains of 6.4% in math & science reasoning, 4.5% in search-based QA, 7.7% in code generation, and 6.9% in multimodal understanding. In addition, AutoTool exhibits stronger generalization by dynamically leveraging unseen tools from evolving toolsets during inference.

**arXiv ID:** 2512.13278
</details>

<details>
<summary><strong>StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning</strong> - Daoyu Wang, Qingchuan Li, Mingyue Cheng, Jie Ouyang, Shuo Yu, Qi Liu, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2604.18401)</summary>

**Abstract:** Agentic reinforcement learning (RL) is emerging as a critical post-training paradigm for improving LLM agent capabilities. Existing RL algorithms for LLMs largely follow the token-centric paradigm as in RLHF and RLVR, where tokens serve as the basic units for modeling and optimization. However, this paradigm introduces a granularity mismatch in agentic RL, as it optimizes token-level predictions while LLM agents make step-level decisions through cycles of environmental observations and actions. To bridge this gap, we propose \textbf{StepPO}, a step-centric paradigm for agentic RL via step-aligned policy optimization. Specifically, we reformulate agentic RL from a token-level Markov Decision Process (MDP) into a step-level MDP, where interaction steps serve as the basic trajectory representations. We further propose step-level credit assignment to align policy optimization with the natural granularity of agent decisions. Together, StepPO optimizes agent policies at the step level for multi-turn agent-environment interaction. Experiments across multi-hop QA, academic paper search, and text-world action tasks show that StepPO consistently outperforms various RL algorithms. Further analyses provide insights into how step-centric paradigm improves agent training. We hope this step-centric paradigm offers a useful lens for understanding agent behavior and a practical path for training more capable LLM agents.

**arXiv ID:** 2604.18401
</details>

<details>
<summary><strong>Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents</strong> - Haoyu Sun, Wenxuan Wang, Mingyang Song, Jujie He, Weinan Zhang, Yang Liu, Yang Yang, Yu Cheng - [[pdf]](https://arxiv.org/pdf/2606.04874)</summary>

**Abstract:** Planning is central to LLM agents: before acting, an agent must decompose goals, select tools, reason over constraints, and decide when a task is infeasible. Yet existing agent evaluations often report only end-to-end success, making it difficult to determine whether failures stem from planning or execution. We introduce Agent Planning Benchmark (APB), a planning-specific diagnostic benchmark with 4,209 multimodal cases across 22 domains and five settings, covering holistic planning, feedback-conditioned step-wise planning, and robustness under extraneous tools, broken tools, and unsolvable tasks. Across 12 MLLMs, APB reveals systematic weaknesses in long-horizon planning, tool-noise robustness, calibrated refusal, and inference-time refinement. We further validate APB on 200 ToolSandbox tasks and 200 $\tau^2$-bench tasks, where APB-guided refinement consistently improves plan correctness, plan grade, and downstream execution metrics across three representative models. APB thus serves as an upstream diagnostic complement to execution benchmarks. The APB benchmark and code are available in \href{this https URL}{this URL}.

**arXiv ID:** 2606.04874
</details>

<details>
<summary><strong>Self-evolving LLM agents with in-distribution Optimization</strong> - Yudi Zhang, Meng Fang, Zhenfang Chen, Mykola Pechenizkiy - [[pdf]](https://arxiv.org/pdf/2606.07367)</summary>

**Abstract:** Large Language Models (LLMs) have recently emerged as powerful controllers for interactive agents in complex environments, yet training them to perform reliable long-horizon decision making remains a fundamental challenge. A key difficulty lies in credit assignment: agents often receive delayed rewards only at the end of episodes. In this paper, we propose Q-Evolve, a self-evolving framework for LLM agents that unifies automatic process-reward labeling and policy learning within a principled in-distribution reinforcement learning paradigm. In each evolving iteration, our method learns an in-distribution critic from a hybrid off-policy dataset that combines expert demonstrations with agent-generated trajectories, stabilizing Bellman backups in sparse-reward settings via a weighted Implicit Q-Learning objective. The learned value function is then used to derive step-wise process rewards through advantage estimation, enabling dense and reliable supervision without environment backtracking or human annotation. Leveraging these signals, we perform behavior-proximal policy optimization that evolves the agent over the data used for process reward labeling, allowing iterative self-improvement without exacerbating distribution shift. We evaluate our method on AlfWorld, WebShop, and ScienceWorld, showing Q-Evolve outperforms strong baselines in sample efficiency, robustness, and overall task performance. Our results demonstrate that stable agent self-evolution is achievable through the co-evolution of process-level supervision and policy, both grounded within a shared in-distribution learning loop.

**arXiv ID:** 2606.07367
</details>

<details>
<summary><strong>Personality Anchoring for Social Simulation: Linking Personality, Social Behavior, and Interaction Success with LLM Agents</strong> - Vahid Sadiri Javadi, Aksa Aksa, Fryderyk Róg, Lucie Flek, Johanne R. Trippas - [[pdf]](https://arxiv.org/pdf/2606.06936)</summary>

**Abstract:** Social interactions are shaped by the interplay of dispositional traits and situational context, yet systematically investigating how personality configurations between individuals jointly influence social behavior across diverse social contexts remains methodologically challenging. We address this gap by introducing a simulation pipeline adapted from the CHARISMA framework, which employs well-known movie characters and public figures as psychologically grounded agents for multi-LLM social simulation using a method we term personality anchoring. We present a large-scale empirical study examining how dyadic Agreeableness composition influences social interaction outcomes across 1,010 simulated conversations. Our results reveal a monotonic relationship between dyadic Agreeableness composition and shared goal achievement, with Homogeneous-Agreeable pairs achieving success 10 times the rate of Homogeneous-Disagreeable pairs (62% vs. 6%). Behavioral mediation analysis reveals that Agreeableness shapes goal achievement partially through cooperative strategy selection, though it continues to predict outcomes within the same dominant strategy, indicating pathways beyond observable conversational behavior. Robustness analyses confirm high consistency of results across repeated simulations (ICC = 0.89) and stable personality expression across diverse scenarios, validating personality anchoring as a viable operationalization strategy.

**arXiv ID:** 2606.06936
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (22 papers)</h2></summary>

<details>
<summary><strong>AdMem: Advanced Memory for Task-solving Agents</strong> - Runzhe Wang, Huilin Lu, Shengjie Liu, Li Dong, Jason Zhu - [[pdf]](https://arxiv.org/pdf/2606.06787)</summary>

**Abstract:** Large Language Models (LLMs) show promise as tool-using agents but remain limited in long-horizon tasks that require remembering, organizing, and reusing knowledge. Prior memory approaches aim to resolve the situation, but mainly focus on storing factual information. Recent work on procedural memory improves task reuse, yet often reduces to replaying past successes without addressing failure cases or online scalability. We introduce a unified and automatic memory framework that integrates semantic, episodic, and procedural memory in a bi-level design combining short-term and long-term stores. A multi-agent architecture with actor, memory, and critic agents enables automatic memory generation, reward annotation, and adaptive retrieval. Long-term memory is managed through reward-based evaluation, merging, and pruning, ensuring scalability and continual improvement. Experiments across various environments show that our approach improves robustness and success on long multi-turn tasks compared to existing baselines. This work highlights the importance of comprehensive, adaptive memory for advancing LLM-based agents.

**arXiv ID:** 2606.06787
</details>

<details>
<summary><strong>DuMate-DeepResearch: An Auditable Multi-Agent System with Recursive Search and Rubric-Grounded Reasoning</strong> - Lingyong Yan, Can Xu, Yukun Zhao, Wenxuan Li, Qingyang Chen, Jiulong Wu, Wenli Song, Xiangnan Li, Weixian Shi, Yiqun Chen, Xuchen Ma, Yuchen Li, Jiashu Zhao, Shuaiqiang Wang, Jianmin Wu, Dawei Yin - [[pdf]](https://arxiv.org/pdf/2606.07299)</summary>

**Abstract:** Deep Research (DR) has emerged as a new agentic paradigm to tackle complex, open-ended research tasks, demanding systems that can iteratively frame problems, acquire evidence, verify sources, and synthesize long-form reports. In practice, however, current DR systems are constrained by four interrelated limitations: long-horizon planning over an underspecified scope, the bottleneck of decomposing and scheduling such tasks within a single agent, hallucination risk in long-form synthesis, and limited process auditability. This technical report presents DuMate-DeepResearch, a multi-agent DR framework built on the Qianfan Agent Foundry. The framework decouples the Agent Core, which handles task understanding, planning, and scheduling, from an extensible Tool Ecosystem for retrieval, evidence acquisition, and report rendering, making every intermediate decision and tool invocation explicitly traceable. Building on this infrastructure, DuMate-DeepResearch further introduces three mechanisms: (i) a graph-based dynamic planning strategy expands the research roadmap coarse-to-fine and continuously revises it through reflection, re-planning, backtracking, and parallel branching; (ii) a recursive two-level execution design delegates each complex search sub-task to an inner Search Agent that runs its own planning loop, isolating noisy retrieval and stabilizing long-horizon execution; (iii) a rubric-based test-time optimization mechanism dynamically generates task-specific quality criteria and uses them as live reasoning scaffolds for evidence-grounded synthesis and adaptive stopping. Across two deep research benchmarks, DuMate-DeepResearch establishes new state-of-the-art results: the best overall score (58.03%) on DeepResearch Bench, and the best overall score (61.95%) on DeepResearch Bench II while ranking first in information recall and analysis.

**arXiv ID:** 2606.07299
</details>

<details>
<summary><strong>Autonomous heterogeneous catalyst discovery with a self-evolving multi-agent digital twin</strong> - Zhilong Song, Zongmin Zhang, Lixue Cheng - [[pdf]](https://arxiv.org/pdf/2606.05050)</summary>

**Abstract:** Theoretical heterogeneous catalysis promises rapid catalyst discovery, yet computational and machine-learning predictions often deviate from experiment and stay confined to narrow material families, for want of a faithful, condition-aware catalytic simulator. We present CatDT (Catalysis Digital Twin), a self-evolving multi-agent system that builds an autonomous digital twin of a working catalyst, unifying gas-solid and liquid-solid modeling. From only a bulk crystal and a natural-language reaction description, eight specialized agents and 27 scientific tools predict stable facets, reconstruct working surfaces, enumerate and rank reaction pathways, locate transition states, and compute kinetics in 5-30 min on a single GPU. Two innovations address the hardest steps: UniMech finds dominant pathways for novel materials at over $10^3\times$ lower cost than exhaustive enumeration by fusing agent-guided proposals with energy-cached graph search, and a memory-augmented reinforcement loop raises barrier-calculation success from 41\% to 84\% across 600 catalytic surfaces. Across seven gas-solid benchmarks -- stepped metals, single-atom catalysts, ordered intermetallics, vacancy-rich 2D sulfides and carbides, and a strong-metal--support-interaction (SMSI) interface -- every CatDT prediction lies within 0.5-2 times experiment over four orders of magnitude. For propane dehydrogenation, CatDT independently discovers non-precious candidates rivaling the Pt-based industrial benchmark, with a proposed Ni@ZrO$_2$ SMSI overlayer reaching a simulated TOF of $1.63~\text{s}^{-1}$ at $\sim$100\% selectivity. More broadly, the decisive factor for a faithful catalyst digital twin -- or any multi-stage scientific simulator -- is not raw LLM capability but the engineered harness around it: deterministic tools, persistent memory, and verified self-improvement that compound across models, tools, and runs.

**arXiv ID:** 2606.05050
</details>

<details>
<summary><strong>Agentic Large Language Models for Automated Structural Analysis of 3D Frame Systems</strong> - Ziheng Geng, Ian Franklin, Santiago Martinez, Jiachen Liu, Yunhe Zhao, Minghui Cheng - [[pdf]](https://arxiv.org/pdf/2606.06525)</summary>

**Abstract:** Large language models (LLMs) have emerged as powerful foundation models with strong reasoning capabilities across domains. Beyond reactive text generation, agentic LLMs enable autonomous workflow execution through modular task decomposition and coordinated tool use. In structural engineering, recent efforts have developed agentic LLMs for automated analysis of plane frames. However, their extension to 3D frames remains underexplored due to challenges in irregular geometric representation, topological consistency, and long-horizon reasoning. This paper proposes an agentic LLM framework for automated structural analysis of 3D frames from natural language inputs. Irregular 3D frames are represented by projection onto a 2D plan, where orthogonal gridlines define spatial coordinates and a matrix of number of stories encodes vertical extrusion of each grid cell. Building on this representation, the framework establishes a multi-agent pipeline: a problem analysis agent parses input into structured JSON; a floor decomposition agent derives the spatial layout of each floor; the 3D geometry is assembled by node, girder, slab, and column agents; support and load agents assign boundary and loading conditions, and code translation agents generate executable SAP2000 script. Evaluated on ten representative 3D frames, the proposed framework achieves an average accuracy of 90% across repeated trials, demonstrating consistent and reliable performance.

**arXiv ID:** 2606.06525
</details>

<details>
<summary><strong>Queen-Bee Agents: A BeeSpec-Centered Architecture for Governed Enterprise MCP Orchestration</strong> - Dutao Zhang, Liaotian - [[pdf]](https://arxiv.org/pdf/2606.06545)</summary>

**Abstract:** Enterprise agent systems increasingly need to connect large language models to private tools, internal knowledge, and Model Context Protocol (MCP) interfaces. In this setting, raw task capability is insufficient: organizations also require policy enforcement, tenant-scoped isolation, and execution that remains within explicit operational boundaries. We present Queen-Bee, a governed multi-agent architecture in which a Queen control plane retrieves capabilities, plans task-scoped execution, and compiles a structured BeeSpec that is executed by specialized Bee agents under constrained tool access. We implement a working prototype with tenant-scoped MCP connectors, audit-backed execution-time governance, retrieval-driven weak incubation, and multiple provisioning backends. We evaluate the system on 59 enterprise-style tasks spanning governance-sensitive requests, retrieval-driven provisioning, scoped local execution, and chemistry workflow integration. The retrieval-driven Queen-Bee variant achieves a task success rate of 0.964, zero governance failures, and substantially better scoped execution quality than both a static Queen-Bee baseline and a permissive single-agent baseline. We further show a multi-Bee chemistry workflow with explicit approval gating and a concrete top-3 shortlist grounded in real upstream evidence and screening artifacts. Additional comparisons with hybrid retrieval and LLM-guided provisioning show that richer provisioning backends are viable but do not outperform the lightweight structured retriever on the current small, highly structured capability registry. The results provide prototype-level systems evidence rather than a production deployment study, and suggest that enterprise agent platforms should be evaluated not only by capability, but also by governed provisioning, isolation behavior, scoped execution quality, and artifact-aware workflow coordination.

**arXiv ID:** 2606.06545
</details>

<details>
<summary><strong>CAF-Gen: A Multi-Agent System for Enriching Argumentation Structures</strong> - Jakub Bąba, Jarosław Chudziak - [[pdf]](https://arxiv.org/pdf/2606.06646)</summary>

**Abstract:** Formalizing complex reasoning from natural text is one of the central challenges in computational linguistics. It requires systems to understand not just keywords but also the context and complex reasoning embedded in a text. Current Argument Mining (AM) techniques identify basic claims and premises, yet they often struggle to capture the richer structural information required by advanced schemas such as the Carneades Argumentation Framework (CAF), which incorporates features such as premise types, proof standards, and argument schemes. We address this limitation by introducing CAF-Gen, an automated multi-agent framework designed to enrich shallow argument structures into CAF-compliant argument models. By employing an iterative Creator-Reviewer pipeline, a creator agent's output is validated by a critical agent to ensure structural integrity. This multi-agent collaboration is crucial for mitigating the structural instability typical of single-pass generative models. Our experiments demonstrate that the iterative feedback loop improves the quality of the resulting data and achieves strong alignment with the original annotations, while producing structurally richer models. Our findings show that the multi-agent system can overcome the limitations of single-pass generation, providing a robust methodology for the automated modeling of formal argumentation.

**arXiv ID:** 2606.06646
</details>

<details>
<summary><strong>CHDP: Cooperative Hybrid Diffusion Policies for Reinforcement Learning in Parameterized Action Space</strong> - Bingyi Liu, Jinbo He, Haiyong Shi, Enshu Wang, Weizhen Han, Jingxiang Hao, Peixi Wang, Zhuangzhuang Zhang - [[pdf]](https://arxiv.org/pdf/2601.05675)</summary>

**Abstract:** Hybrid action space, which combines discrete choices and continuous parameters, is prevalent in domains such as robot control and game AI. However, efficiently modeling and optimizing hybrid discrete-continuous action space remains a fundamental challenge, mainly due to limited policy expressiveness and poor scalability in high-dimensional settings. To address this challenge, we view the hybrid action space problem as a fully cooperative game and propose a \textbf{Cooperative Hybrid Diffusion Policies (CHDP)} framework to solve it. CHDP employs two cooperative agents that leverage a discrete and a continuous diffusion policy, respectively. The continuous policy is conditioned on the discrete action's representation, explicitly modeling the dependency between them. This cooperative design allows the diffusion policies to leverage their expressiveness to capture complex distributions in their respective action spaces. To mitigate the update conflicts arising from simultaneous policy updates in this cooperative setting, we employ a sequential update scheme that fosters co-adaptation. Moreover, to improve scalability when learning in high-dimensional discrete action space, we construct a codebook that embeds the action space into a low-dimensional latent space. This mapping enables the discrete policy to learn in a compact, structured space. Finally, we design a Q-function-based guidance mechanism to align the codebook's embeddings with the discrete policy's representation during training. On challenging hybrid action benchmarks, CHDP outperforms the state-of-the-art method by up to $19.3\%$ in success rate.

**arXiv ID:** 2601.05675
</details>

<details>
<summary><strong>Dual Latent Memory for Visual Multi-agent System</strong> - Xinlei Yu, Chengming Xu, Zhangquan Chen, Bo Yin, Cheng Yang, Yongbo He, Yihao Hu, Jiangning Zhang, Cheng Tan, Xiaobin Hu, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2602.00471)</summary>

**Abstract:** While Visual Multi-Agent Systems (VMAS) promise to enhance comprehensive abilities through inter-agent collaboration, empirical evidence reveals a counter-intuitive "scaling wall": increasing agent turns often degrades performance while exponentially inflating token costs. We attribute this failure to the information bottleneck inherent in text-centric communication, where converting perceptual and thinking trajectories into discrete natural language inevitably induces semantic loss. To this end, we propose \textbf{L}$\mathbf{^{2}}$\textbf{-VMAS}, a novel model-agnostic framework that enables inter-agent collaboration with dual latent memories. Furthermore, we decouple the perception and thinking while dynamically synthesizing dual latent memories. Additionally, we introduce an entropy-driven proactive triggering that replaces passive information transmission with efficient, on-demand memory access. Extensive experiments among backbones, sizes, and multi-agent structures demonstrate that our method effectively breaks the "scaling wall" with superb scalability, improving average accuracy by 2.7-5.4% while reducing token usage by 21.3-44.8%.

**arXiv ID:** 2602.00471
</details>

<details>
<summary><strong>Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA</strong> - John Ray B. Martinez - [[pdf]](https://arxiv.org/pdf/2603.24481)</summary>

**Abstract:** Miscalibrated confidence scores are a practical obstacle to deploying AI in clinical settings. A model that is always overconfident offers no useful signal for deferral. We present a multi-agent framework that combines domain-specific specialist agents with Two-Phase Verification (Wu et al., 2024) and S-Score Weighted Fusion to improve both calibration and discrimination in medical multiple-choice question answering. Four specialist agents (respiratory, cardiology, neurology, gastroenterology) generate independent diagnoses using Qwen2.5-7B-Instruct. Each diagnosis undergoes a two-phase self-verification process that measures internal consistency and produces a Specialist Confidence Score (S-score). The S-scores drive a weighted fusion strategy that selects the final answer and calibrates the reported confidence. We evaluate on high-disagreement subsets of MedQA-USMLE and MedMCQA (100 and 250 questions). All results are specific to this filtered regime. On MedQA-250, the full system achieves ECE = 0.091 (74.4% reduction over the single-specialist baseline) and AUROC = 0.630 (+0.056) at 59.2% accuracy. Calibration gains of 49-74% hold across all four settings. Ablation analysis reveals that Two-Phase Verification drives ECE reduction while multi-agent reasoning drives AUROC improvement, suggesting that consistency checking and ensemble aggregation address different failure modes of LLM uncertainty. Whether the resulting confidence signal is sufficient to support clinical deferral decisions in practice remains a direction for future investigation.

**arXiv ID:** 2603.24481
</details>

<details>
<summary><strong>Robust Instruction Compliance in Cooperative Multi-Agent Reinforcement Learning</strong> - Wo Wei Lin, Ethan Rathbun, Enrico Marchesini, Xiang Zhi Tan - [[pdf]](https://arxiv.org/pdf/2605.12655)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) in real-world use cases may need to adapt to external natural language instructions that interrupt ongoing behavior and conflict with long-horizon objectives. However, conditioning rewards on instructions introduces a fundamental failure mode as Bellman updates couple value estimates across instruction contexts, leading to inconsistent values when instructions interrupt macro-actions. We propose Macro-Action Value Correction for Instruction Compliance (MAVIC), which corrects Bellman backups at instruction boundaries by correcting the incoming instruction objective and restoring the continuation value under the current objective. Unlike reward shaping, MAVIC modifies the bootstrapping target itself, enabling consistent value estimation under stochastic instruction switching within a unified policy. We provide theoretical analysis and an actor-critic implementation, and show that MAVIC achieves high instruction compliance while preserving base task performance in increasingly complex cooperative multi-agent environments.

**arXiv ID:** 2605.12655
</details>

<details>
<summary><strong>Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents</strong> - Akshay Manglik, Apaar Shanker, Kaustubh Deshpande, Jason Qin, Yash Maurya, Veronica Chatrath, Vijay S. Kalmath, Levi Lentz, Yuan Xue - [[pdf]](https://arxiv.org/pdf/2605.21347)</summary>

**Abstract:** Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by implementing IG insights. Human experts using IG reports improve scaffold performance by 30.4pp over the unmodified baseline scaffold, and coding agents leveraging IG-derived insights show consistent and stable gains. Across benchmarks, IG's scout-investigator architecture produces findings comparable in detection coverage to competing approaches, while domain experts rated IG reports as leading depth and evidence quality.

**arXiv ID:** 2605.21347
</details>

<details>
<summary><strong>ReclAIm: A Multi-Agent Framework for Monitoring and Correcting Performance Decline in Medical Imaging AI</strong> - Eleftherios Tzanis, Michail E. Klontzas - [[pdf]](https://arxiv.org/pdf/2510.17004)</summary>

**Abstract:** Purpose: To develop and evaluate a multi-agent framework (ReclAIm) for automated monitoring, detection, and correction of performance decline in medical image classification models.
Materials and Methods: ReclAIm is a large language model-based multi-agent system that operates through natural language interaction. A master agent coordinating three task-specific agents performed performance evaluation and triggered fine-tuning when substantial performance declines were detected. The fine-tuning workflow incorporated data augmentation, class imbalance handling, and a parameter-anchoring regularization strategy to limit catastrophic forgetting. The system was benchmarked using multiple imaging datasets, including brain MRI, chest CT, and chest radiography, partitioned into model development, inference (performance monitoring), and fine-tuning subsets (60%:20%:20%).
Results: ReclAIm successfully orchestrated training, evaluation, and performance monitoring across all datasets. Performance discrepancies between test and inference data were detected in 8 of 18 models, prompting fine-tuning workflows that reduced performance gaps. In cases with declines of up to 40.6% (cardiomegaly dataset, InceptionV3), fine-tuning restored performance metrics to within 2% of baseline values.
Conclusion: ReclAIm provides a prototype framework for automated monitoring and targeted fine-tuning of medical image classification models, with a natural language interface designed to support accessibility in research and potential clinical applications.

**arXiv ID:** 2510.17004
</details>

<details>
<summary><strong>Towards Iterative End-to-End Software Development: A Feature-Driven Multi-Agent Framework</strong> - Junwei Liu, Chen Xu, Chong Wang, Tong Bai, Weitong Chen, Kaseng Wong, Yiling Lou, Xin Peng - [[pdf]](https://arxiv.org/pdf/2511.02399)</summary>

**Abstract:** Recent advances in large language model agents offer the promise of automating end-to-end software development from natural language requirements. However, existing approaches largely adopt linear, waterfall-style pipelines, which oversimplify the iterative nature of real-world development and struggle with complex, large-scale projects. To address these limitations, we propose EvoDev, an iterative software development framework inspired by feature-driven development. EvoDev decomposes user requirements into a set of user-valued features and constructs a Feature Map, a directed acyclic graph that explicitly models dependencies between features. Each feature node in the feature map maintains multi-layer contexts, including business logic, software design, and code implementation, which are propagated along dependencies to provide context for subsequent development iterations. We evaluate EvoDev on challenging Android development tasks and show that it outperforms the best-performing baseline, Claude Code, by 57.3%, while improving single-agent performance by 16.0%-58.5% across different base LLMs, highlighting the importance of feature decomposition, dependency modeling, context propagation, and workflow-aware agent design for end-to-end software development. Moreover, our work summarizes practical insights for designing iterative, LLM-driven development frameworks and informs future training of base LLMs to better support iterative software development.

**arXiv ID:** 2511.02399
</details>

<details>
<summary><strong>SW-$A^2$-Bench: Benchmarking Autonomous Software Agent Generation for Agentic Web</strong> - Linyao Chen, Bo Huang, Qinlao Zhao, Shuai Shao, Zhi Han, Zicai Cui, Ziheng Zhang, Guangtao Zeng, Wenzheng Tang, Yikun Wang, Yuanjian Zhou, Zimian Peng, Yong Yu, Weiwen Liu, Hiroki Kobayashi, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2604.04226)</summary>

**Abstract:** The Agentic Web is emerging as a paradigm in which autonomous software agents interact with online resources and with each other to accomplish user goals. However, the capacity of Agentic Web is still limited by insufficient autonomous software agent population, which has become a crucial challenge for scaling Agentic Web. In order to alleviate this, we study the task of automatically converting existing code repositories into autonomous software agents via coding agents, decompose the process into critical stages, and identify key technical hurdles. To systematically evaluate this capability, we propose SoftWare Agent generation for Agentic Web Bench (SW-$A^2$-Bench), the first benchmark designed for software agent generation. SW-$A^2$-Bench evaluates not only whether software agents can be generated, but also whether generated software agents are faithful to the source repositories and interoperable with other agents in multi-agent workflows. Our experiments demonstrate that our approach effectively activates the functional capabilities of code repositories and enables interoperable multi-agent collaboration in Agentic Web. We believe that this work will provide a standardized evaluation for software agent generation and will contribute to the future of scaling the capacity of Agentic Web.

**arXiv ID:** 2604.04226
</details>

<details>
<summary><strong>MADRAG: Multi-Agent Debate with Retrieval-Augmented Generation for Training-Free Analytic Essay Scoring</strong> - Ali Keramati, Shiyuan Zhou, Sharad Mehrotra, Mark Warschauer - [[pdf]](https://arxiv.org/pdf/2606.06754)</summary>

**Abstract:** We present MADRAG, a training-free framework for analytic essay scoring that combines multi-agent reasoning with retrieval-augmented grounding. Unlike standard LLM-as-judge approaches, which are prone to bias and unstable scoring, MADRAG decomposes evaluation into an interactive process: an Advocate identifies strengths, a Skeptic critiques weaknesses, and a Judge aggregates their arguments into a final score. Crucially, the Judge is augmented with rubric-aligned exemplar retrieval, enabling calibration through comparison with scored examples. Our results show that MADRAG significantly outperforms prompt-based baselines while approaching the performance of supervised systems without requiring task-specific training. Ablation studies demonstrate that retrieval drives calibration gains, while debate improves reasoning on higher-level traits. Our findings highlight the complementary roles of structured interaction and external memory in reliable LLM-based evaluation.

**arXiv ID:** 2606.06754
</details>

<details>
<summary><strong>Modeling U.S. Attitudes Toward China via an Event-Steered Multi-Agent Simulator</strong> - Chenxu Zhu, Hantao Yao, Wu Liu, Junbo Guo, Yongdong Zhang - [[pdf]](https://arxiv.org/pdf/2606.06971)</summary>

**Abstract:** Understanding the dynamic evolution of opinions, such as U.S. public attitudes toward China, is essential for assessing geopolitical risks. However, existing LLM-based multiagent simulators predominantly rely on static rules and fixed datasets, limiting their ability to capture the dynamic, event-driven nature of macro-level opinion shifts in real-world settings. To address this limitation, we propose an Event-Steered Multi-Agent Simulator (ES-MAS), in which significant events and daily news continuously drive opinion evolution through dynamic interactions among agents. We first construct the China-U.S. Relation Evolution (CURE) dataset, covering 20 quarters from 2021 to 2025, including 258 major events and over 14,000 daily news articles, and providing a comprehensive temporal foundation for modeling opinion dynamics. Building upon the CURE dataset, we propose a Dual-Stream Data Integration Engine (DSDIE) that aligns simulations with historical timelines via macro-level events while enabling personalized information exposure based on individual agent profiles and contextual signals. Furthermore, we design a News-Driven Dynamic Interaction (NDDI) module, which adaptively groups agents with shared news interests into localized interaction contexts, facilitating bottom-up consensus formation while mitigating the risk of isolated information cocoons. Experimental results on the CURE dataset demonstrate that ES-MAS substantially outperforms existing simulators in reproducing real-world historical trends, offering a scalable and effective framework for modeling dynamic opinion evolution.

**arXiv ID:** 2606.06971
</details>

<details>
<summary><strong>Learning Multi-Agent Communication Protocol: Study on Information Entropy Efficiency in MARL</strong> - Xinren Zhang, Zixin Zhong, Jiadong Yu - [[pdf]](https://arxiv.org/pdf/2606.07200)</summary>

**Abstract:** Multi-Agent Systems (MAS) have emerged as a fundamental paradigm for distributed problem-solving, where autonomous agents collaborate to achieve complex objectives. Within this framework, Multi-Agent Reinforcement Learning (MARL) with communication has demonstrated remarkable success in cooperative tasks. However, existing approaches predominantly pursue performance gains through increasingly complex architectures and expanding communication overhead, lacking principled metrics to evaluate the efficiency of information exchange. In this paper, we focus on enabling agents to learn efficient multi-agent communication protocols that balance performance and information compactness. We propose the Information Entropy Efficiency Index (IEI), a novel metric that quantifies the ratio between message entropy and task performance in learned communication protocols. A lower IEI indicates more compact and efficient message representations. By incorporating IEI into training loss functions, we encourage agents to develop communication protocols that achieve high performance with improved communication efficiency. Extensive experiments across diverse MARL algorithms demonstrate that our approach achieves equivalent or superior task performance compared to baseline methods while improving communication efficiency. These findings challenge the prevailing assumption that performance improvements require complex architectures or increased communication overhead and highlight the potential of improving both task success and communication efficiency to enable scalable MAS.

**arXiv ID:** 2606.07200
</details>

<details>
<summary><strong>More Capable, Less Cooperative? When LLMs Fail At Zero-Cost Collaboration</strong> - Advait Yadav, Sid Black, Oliver Sourbut - [[pdf]](https://arxiv.org/pdf/2604.07821)</summary>

**Abstract:** Large language model (LLM) agents increasingly coordinate in multi-agent systems, yet we lack an understanding of where and why cooperation fails. Many real-world coordination problems are not social dilemmas: helping others -- sharing documentation, unblocking a teammate -- costs the helper almost nothing while producing substantial collective benefit. Whether LLM agents cooperate in this regime, where helping is free and they are explicitly instructed to do so, remains unknown. We build a turn-based multi-agent environment that strips away all strategic complexity, making cooperation costless and trivially optimal. Across eight widely used LLMs, capability does not predict cooperation: OpenAI o3 reaches only 17% of optimal collective performance while the weaker o3-mini reaches 50%, despite identical instructions to maximize group revenue. Using a causal decomposition that automates one side of agent communication, we separate cooperation failures from competence failures, and find that several capable models actively withhold information despite gaining nothing from withholding. Targeted interventions address each mode: explicit protocols roughly double the performance of competence-limited models, while small sharing incentives unlock cooperation-limited ones. Our results suggest that scaling intelligence alone will not solve coordination in multi-agent systems, and will require deliberate cooperative design, even when helping costs nothing.

**arXiv ID:** 2604.07821
</details>

<details>
<summary><strong>Agentopia: Long-Term Life Simulation and Learning in Agent Societies</strong> - Xintao Wang, Sirui Zheng, Hongqiu Wu, Weiyuan Li, Jen-tse Huang, Minghao Zhu, Can Zu, Qi Deng, Jiawei Wang, Qianyu He, Heng Wang, Xiaojian Wu, Yunzhe Tao - [[pdf]](https://arxiv.org/pdf/2606.07513)</summary>

**Abstract:** Humans learn from social life. Simulating this process with LLM-powered agents represents a promising research direction, raising a natural question: whether LLMs can learn from such simulated social experience to better understand and replicate human behavior. However, prior agent society simulations typically operate at the scale of days, limiting the depth of social interactions and long-term growth. In this paper, we study long-term life simulation and LLM learning in agent societies, with two goals: (1) investigating social behaviors that emerge from life-long simulation, and (2) developing anthropomorphic capabilities in LLMs, particularly intelligence in social life, through years of simulated social experience. Specifically, we present Agentopia, a comprehensive framework for long-term life simulation in multi-agent societies, where 100 agents autonomously pursue personal growth, develop social relationships, and fulfill their needs and goals over 10 simulated years. We define life reward to mirror human well-being, and leverage this reward to train LLMs via rejection sampling. Extensive experiments show that agents exhibit rich emergent social behaviors. Furthermore, life reward training effectively enhances the underlying LLM, which leads to improved agent well-being in simulation, and generalizes to downstream role-playing benchmarks with +15.6% improvement.

**arXiv ID:** 2606.07513
</details>

<details>
<summary><strong>Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems</strong> - Yingzhuo Liu - [[pdf]](https://arxiv.org/pdf/2606.05711)</summary>

**Abstract:** Multi-agent systems built on large language models (LLMs) have become a prevailing paradigm for tackling complex reasoning, planning, and tool-use tasks. The dominant communication protocol in such systems is natural language: agents exchange messages token-by-token, verbalising their internal reasoning so that peers can read, verify, and respond. While convenient and interpretable, this protocol suffers from three structural drawbacks -- high inference cost, irreversible information loss during discretization, and ambiguity/redundancy of natural language. A growing body of work therefore explores an alternative protocol -- latent communication -- in which agents exchange continuous representations (embeddings, hidden states, or KV-caches) directly, bypassing the bottleneck of text generation. This paper presents a unified framework for organising the rapidly expanding literature on latent communication. We analyse existing methods along three orthogonal axes: (1) WHAT information is communicated (Embeddings, Hidden States, KV-Caches, or other continuous state); (2) WHICH sender-receiver alignment is used (latent-space alignment and layer alignment); and (3) HOW the communicated information is fused into the receiver (concatenation, prepending, mathematical operations, cross-attention, or cache restoration). Under this 3-axis framework, we systematically categorise eighteen representative methods proposed between 2024 and 2026, identify five major design patterns, and surface a set of open challenges -- including cross-architecture alignment, security of latent channels, compression for edge deployment, and the relationship between latent communication and latent chain-of-thought. We hope that this framework both lowers the barrier to entry for new researchers and provides a vocabulary for comparing future work.

**arXiv ID:** 2606.05711
</details>

<details>
<summary><strong>Uncertainty-Aware LLM-Guided Policy Shaping for Sparse-Reward Reinforcement Learning</strong> - Ujjwal Bhatta, Utsabi Dangol, Sumaly Bajracharya, Rodrigue Rizk, KC Santosh - [[pdf]](https://arxiv.org/pdf/2606.06673)</summary>

**Abstract:** Sparse rewards and heterogeneous task sequences remain persistent challenges in Reinforcement Learning (RL), often resulting in slow convergence, weak generalization, and inefficient exploration. We propose Uncertainty-Aware LLM-Guided Policy Shaping (ULPS), a novel framework that integrates a calibrated Large Language Model (LLM) into the RL training loop to provide structured, uncertainty-modulated behavioral guidance. ULPS employs an A*-based oracle to synthesize optimal symbolic trajectories, which are used to fine-tune a BERT-based language model. During training, this model supplies action suggestions whose influence is conditioned on epistemic uncertainty estimated via Monte Carlo (MC) dropout. An entropy-based blending mechanism adaptively balances LLM guidance and the learned policy (via Proximal Policy Optimization, PPO), allowing the agent to prioritize reliable priors while preserving adaptability. We evaluate ULPS on the MiniGridUnlockPickup benchmark and observe consistent improvements in success rate, reward efficiency, and sample complexity over unguided, uncalibrated, and standard RL baselines. ULPS achieves more than 9% improvement in execution accuracy after fine-tuning, requires fewer environment interactions, and yields higher reward AUC. Our results demonstrate that integrating symbolic A* trajectories, pretrained language priors, and uncertainty-aware control offers a principled and effective approach to multi-task reinforcement learning in sparse-reward domains, with potential extensibility to partially observable and multi-agent settings.

**arXiv ID:** 2606.06673
</details>

<details>
<summary><strong>CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks</strong> - Seoyeon Choi, Kanghyun Ryu, Jonghoon Ock, Negar Mehr - [[pdf]](https://arxiv.org/pdf/2509.14380)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) provides a powerful framework for learning coordination in multi-agent systems. However, applying MARL to robotics remains challenging due to their high-dimensional continuous joint action spaces, complex reward design, and non-stationarity from concurrently learning agents. On the other hand, humans often learn complex coordination with the help of coaches, who guide learning through carefully designed curricula and detailed feedback. Building on the reasoning capabilities of foundation models, we argue that these models can similarly coach robots to learn coordination. Motivated by this, we propose CRAFT: Coaching Reinforcement learning Autonomously using Foundation models for learning coordination Tasks, a framework that leverages foundation models to act as a "coach" for multi-robot coordination. CRAFT automatically decomposes long-horizon coordination tasks into sequences of subtasks using the planning capability of Large Language Models (LLMs). Then, CRAFT trains each subtask using LLM-generated reward functions, and refines them through a Vision Language Model (VLM)-guided reward-refinement loop. We evaluate CRAFT on multi-quadruped navigation and bimanual manipulation tasks, and demonstrate its capability to learn complex coordination behaviors. In addition, in a multi-quadruped navigation setting, we show that our learned policies transfer to the real world. Project website is this https URL

**arXiv ID:** 2509.14380
</details>

</details>

<details open>
<summary><h2>Other Agent Research (13 papers)</h2></summary>

<details>
<summary><strong>Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory</strong> - Ruida Wang, Jerry Huang, Pengcheng Wang, Xuanqing Liu, Luyang Kong, Tong Zhang - [[pdf]](https://arxiv.org/pdf/2606.06523)</summary>

**Abstract:** Equipping Large Language Models (LLMs) to execute reliable multi-step workflows has become a central challenge in artificial intelligence. Despite recent advances in LLMs' agentic capabilities, most agent systems still lack formal methods for specifying, verifying, and debugging their workflow and execution trajectories. This challenge mirrors a long-standing problem in mathematics, where the ambiguity of natural languages (NLs) motivates the development of formal languages (FLs). Inspired by this paradigm, we propose **Lean4Agent**, to the best of our knowledge, the first framework that uses Lean4, a dependent-type FL to model and verify agent behavior. **Lean4Agent** launches **FormalAgentLib**, an extensible Lean4 library for formally modeling and verifying agent workflows' semantic consistency under explicit assumptions, and enabling localization of execution-time failures revealed by trajectories. Building on **FormalAgentLib**, we further develop **LeanEvolve**, which applies results in **FormalAgentLib** to revise workflows to enhance its capability. Extensive experiments on a hard problem subset of SWE-Bench-Verified and a subset of ELAIP-Bench across 5 leading LLMs indicate that the verification-passing workflows outperform the failing ones by an average of **11.94%**, and **LeanEvolve** further improves SWE performance by **7.47%** on average. Furthermore, **Lean4Agent** establishes a foundation for a new field of using expressive dependent-type FL to formally model and verify agent behavior.

**arXiv ID:** 2606.06523
</details>

<details>
<summary><strong>IRAF: Interference-Resilient Adaptive Fusion for Noise-Robust End-to-End Full-Duplex Spoken Dialogue Systems</strong> - Tao Zhong, Jiajun Deng, Nikita Kuzmin, Yinke Zhu, Tianxiang Cao, Tristan Tsoi, Zhili Tan, Simon Lui, Xunying Liu - [[pdf]](https://arxiv.org/pdf/2606.06559)</summary>

**Abstract:** Full-duplex spoken dialogue models allow voice agents to listen and speak concurrently, enabling natural interaction with real-time overlap. However, end-to-end dual-channel models that jointly encode user and agent streams may degrade in realistic acoustic environments: interfering speakers leaking into the user microphone can be encoded as part of the user query, corrupting the LLM's conditioning and causing unstable turn-taking and reduced response quality. We propose Interference-Resilient Adaptive Fusion (IRAF), a lightweight, streaming-compatible module that modulates the contribution of user audio to the LLM frame by frame. IRAF predicts a scalar reliability gate from target-speaker and user audio embeddings and rescales user representations before fusion with agent embeddings. Experiments on MS-MARCO and InstructS2S-200K show consistent gains in response quality and full-duplex interaction under interfering-speaker conditions.

**arXiv ID:** 2606.06559
</details>

<details>
<summary><strong>The Three-Ring Architecture: Governing Agents in the Era of On-Platform Organisations</strong> - Sergio Alvarez-Telena, Marta Diez-Fernandez - [[pdf]](https://arxiv.org/pdf/2606.07119)</summary>

**Abstract:** The current phase of enterprise AI deployment faces a structural failure: organisations are acquiring agentic capability without the infrastructure to govern it. The result is expected to reproduce the error of the first wave of AI deployment: decentralised intelligence without a federation layer leading to a 95% project failure rate. This paper formalises the Three-Ring Architecture as the governing infrastructure of the on-platform organisation. Ring 1 is the existing production architecture; Ring 2 is the M2 federation layer built on strategies-based agentic AI; Ring 3 is the LLM-based frontier intelligence layer. Ring 2 constitutes, in the technically exact sense, the operating system of the agentic enterprise - performing at the organisational level what a computing OS performs at the device level: resource abstraction, process coordination, permission enforcement, and a stable platform for compounding intelligence. A central contribution is the formal distinction between Ring 2 and Ring 3 risk profiles. Strategies-based agents operate within a deterministic framework: their consequences are traceable, their permissions enforceable, their deviations recoverable. LLM-based agents introduce a categorically distinct risk: a non-deterministic actor whose deviations propagate through complex organisational systems without retrospective traceability. Ring 2 is not a useful addition - it is a necessary condition of control and compliance. A further consequence: every improvement in LLM capability is a structural tailwind for this architecture. More capable non-deterministic actors produce larger consequences when they deviate. The governance requirement scales with capability. The architecture has been validated across a decade of deployment in financial services, government, procurement, and compliance among other sectors.

**arXiv ID:** 2606.07119
</details>

<details>
<summary><strong>Re-imagining ISO 26262 in the Age of Autonomous Vehicles: Enhancing Controllability through Transferability and Predictability</strong> - Chaitanya Shinde, Hadi Hajieghrary, Paul Schmitt, Adam Shoemaker, Bodo Seifert, Steve Kenner - [[pdf]](https://arxiv.org/pdf/2606.07437)</summary>

**Abstract:** The ISO 26262 standard defines functional safety for road vehicles through risk assessments based on Severity, Exposure, and Controllability, grounded in a human-driven vehicle paradigm. In the context of autonomous vehicles (AVs), the absence of a human driver necessitates revisiting these principles. This paper decomposes the Controllability placeholder into two auditable evidence dimensions of ISO 26262 by introducing two measurable sub-concepts: Transferability and Predictability. Transferability extends Controllability to capture AV systems' ability to hand off control to dedicated fallback safety mechanisms, while Predictability captures how easily external agents can anticipate AV behavior. Predictability is formally defined from human-robot interaction-inspired principles, and a mathematical framework is provided to quantify it. A designed-versus-achievable gap is introduced to distinguish architectural fallback claims from scene-conditioned achievable fallback capability. The proposed metrics align with ISO 26262 and ISO/PAS 21448 (SOTIF), rendering fallback and interaction claims falsifiable and traceable across ODD slices. These dimensions complement rather than replace existing standards, and the enhancements preserve the structure of ISO 26262 while extending its applicability to driverless automated systems operating at SAE Levels 4 and 5.

**arXiv ID:** 2606.07437
</details>

<details>
<summary><strong>EVA: Evolving Semantic Adversaries for Red-Teaming GUI Agents Against Environmental Injection Attacks</strong> - Yijie Lu, Manman Zhao, Tianjie Ju, Zihe Yan, Xinbei Ma, Yuan Guo, Daizong Ding, Gongshen Liu, Zhuosheng Zhang - [[pdf]](https://arxiv.org/pdf/2505.14289)</summary>

**Abstract:** Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) are increasingly deployed yet vulnerable to Environmental Injection Attacks (EIAs).However, current red-teaming methods are hindered by prohibitive computational costs and limited adaptability. A fundamental question remains unaddressed: does the bottleneck of attack success lie in visual perception or semantic understanding? Through controlled experiments, we observe that semantic deception, rather than visual appearance, serves as the primary determinant of attack success. Based on this insight, we introduce EVA, an evolutionary framework that evolves adversarial payloads exclusively within the semantic dimension. EVA employs a discovery-deployment framework to mine linguistic vulnerability patterns and distill them into generalizable rules. Experimental results across five representative victim agents demonstrate that EVA achieves up to 85\% attack success rate, evolving benign seeds into successful attacks within only 1.18 to 1.71 iterations. This rapid convergence uncovers a dense semantic attack space in the model's latent representation, unveiling a critical alignment paradox: the instruction-following capabilities reinforced by alignment training render agents inherently susceptible to authoritative, semantically deceptive environmental cues.

**arXiv ID:** 2505.14289
</details>

<details>
<summary><strong>Comparing Sentiment Contagion in AI-Agent and Human Social Networks: Evidence from MOLTBOOK</strong> - Elyes Ben chaabane, Savindu Herath, Yash Raj Shrestha - [[pdf]](https://arxiv.org/pdf/2606.06665)</summary>

**Abstract:** AI agents are beginning to interact not only with people, but also with one another. We investigate what happens to sentiment in such an AI-only social network: does negativity spread, or do replies calm it down? We study MOLTBOOK, a social network made up of autonomous language-model agents, using almost 2.9 million posts and 1.5 million comments. Negative posts receive many more replies than neutral or positive posts, so negativity still attracts attention. However, replies to negative content usually do not stay negative. They most often become neutral, and there is meager evidence that negative sentiment spreads across days. The main pattern is therefore not a cycle of negativity, but negative attention followed by neutralisation. These findings suggest that AI-agent networks may behave differently from human social networks: they may dampen emotional extremes, while still depending strongly on how interactions are organised.

**arXiv ID:** 2606.06665
</details>

<details>
<summary><strong>OpenAgenet / OAN White Paper: Open Infrastructure for Trusted Agent Interconnection</strong> - Jinliang Xu - [[pdf]](https://arxiv.org/pdf/2606.03161)</summary>

**Abstract:** OpenAgenet, abbreviated as OAN, is an open infrastructure project for trusted Agent interconnection. It addresses a problem that becomes visible when Agents move from isolated applications into open, multi-operator networks: before an Agent can safely discover, select, and invoke another Agent, it needs a way to verify identity provenance, governance state, discovery authorization, freshness, and pre-connection trust evidence. OAN is designed as a protocol-neutral trust layer. It does not replace Agent interaction protocols, tool protocols, model orchestration frameworks, or application-level workflows. Instead, it provides \texttt{did:oan}-based resource identity, governance-backed admission, Registrar-assisted onboarding, Root-verified package publication, authorization-aware Discovery, Root-issued infrastructure authorization VCs, and signed trusted invocation. The architectural center of OAN is the combination of federated governance, resource identity, and trusted Discovery, rather than a single directory or naming service. This white paper explains the motivation, architecture, roles, governance model, relationship with MCP, A2A, and ANP, deployment patterns, cooperation model, on-chain governance layer, prototype status, performance profile, and roadmap of OAN.

**arXiv ID:** 2606.03161
</details>

<details>
<summary><strong>Signal-Driven Observation for Long-Horizon Web Agents</strong> - Shubham Gaur, Ian Lane - [[pdf]](https://arxiv.org/pdf/2606.06708)</summary>

**Abstract:** Web agents operating over long horizons ingest raw DOM and accessibility trees -- routinely tens of thousands of tokens -- at every action step, causing progressive context degradation that erodes reasoning well before tasks complete. We argue that this coupling of observation frequency to action frequency is an architectural mistake. Drawing on the insight from Recursive Language Models that querying a document outperforms reading it wholesale, we propose Signal-Driven Observation (SDO): a dedicated sub-call reads the full DOM but returns only task-relevant elements and their selectors, and is re-invoked only when a lightweight signal detector fires -- triggered by URL transitions, newly visible interactive elements, action failures, or exogenous browser events. We outline the open problems SDO introduces and call on the community to treat observation compression as a core architectural decision in web agent design.

**arXiv ID:** 2606.06708
</details>

<details>
<summary><strong>Mission-Level Runtime Assurance Framework for Autonomous Driving</strong> - Chieh Tsai, Salim Hariri - [[pdf]](https://arxiv.org/pdf/2606.06996)</summary>

**Abstract:** This paper studies runtime safety for autonomous driving when high-level driving commands become faulty or unreliable. Unlike conventional runtime-safety approaches that mainly focus on immediate vehicle safety, the proposed framework evaluates both driving safety and whether the vehicle can still successfully complete its mission before a command is executed. The framework extends highway-env with mission-level fault scenarios such as skipping required checkpoints, entering restricted areas, and generating future routes that can no longer complete the mission successfully. A runtime monitoring system is introduced to detect and reject unsafe or mission-infeasible commands before execution. For comparison, an adapted Simplex-Drive runtime-safety baseline with learning-based driving control, safety fallback control, and runtime controller switching is implemented using the public Simplex-Drive framework. Experimental results show that platform-level runtime safety alone cannot detect mission-level planning faults, while the proposed framework successfully rejects mission-infeasible commands and improves mission success under randomized fault conditions.

**arXiv ID:** 2606.06996
</details>

<details>
<summary><strong>Extending Responsibility-Sensitive Safety for the Assessment of Offloaded Autonomous Driving Services</strong> - Robin Dehler, Aryan Thakur, Michael Buchholz - [[pdf]](https://arxiv.org/pdf/2606.07067)</summary>

**Abstract:** Safety is a fundamental requirement in the development of autonomous driving (AD) systems. While function offloading has demonstrated significant benefits in terms of computational efficiency and energy consumption, its application to safety-critical AD functionality introduces new challenges. In particular, offloaded service compositions incur increased and variable response times due to wireless vehicle-to-everything (V2X) communication, which directly affects the vehicle's reaction time and thus its safety guarantees. In this paper, we address this challenge by extending the definitions of Responsibility-Sensitive Safety (RSS) to explicitly account for different response times of local and offloaded AD service compositions. Based on this extension, we propose an integration into function offloading, using the RSS safety constraints for offloading decision-making and fallback mechanisms. Offloaded service compositions are only permitted if the current traffic situation remains safe under the corresponding end-to-end response time. If this condition is violated, the system performs a controlled fallback to local execution. Furthermore, we introduce an enhanced fallback strategy that includes a warm-standby phase for offloaded services, enabling faster and safer transitions from offloaded to local services. The proposed approach is integrated into our AD stack and evaluated in both simulation and the real world. Experimental results demonstrate that the proposed method improves safety compared to state-of-the-art function offloading and safety frameworks, while preserving the benefits of distributed computation when safety conditions allow.

**arXiv ID:** 2606.07067
</details>

<details>
<summary><strong>A Causal Probabilistic Framework for Perception-Informed Closed-Loop Simulation of Autonomous Driving</strong> - Zhennan Fei, Rickard Johansson, Mikael Andersson, Matthias Eng, Mattias Eriksson, Kaveh Kianfar, Sadegh Rahrovani, Chris van der Ploeg, Michael Borth, Maren Buermann, Michiel Braat, Henk Goossens, Zijian Han, Majid Khorsand Vakilzadeh, Gabriel Rodrigues de Campos - [[pdf]](https://arxiv.org/pdf/2606.07186)</summary>

**Abstract:** Software-in-the-loop (SIL) simulation is a cornerstone for the validation of modern automotive safety functions. However, many current frameworks utilize ideal sensing, which bypasses the functional insufficiencies of perception algorithms, leading to over-optimistic safety assessments. This paper proposes a perception-informed SIL testing methodology that bridges the gap between ground-truth simulation and real-world perception behavior. We present a framework for incorporating causal probabilistic models into standardized, scenario-based simulation toolchains, applicable to both Advanced Driver Assistance Systems (ADAS) and Autonomous Driving Systems (ADS). Our approach enables the systematic injection of realistic perception errors, such as loss of detection, sizing inaccuracies, and positioning offsets, derived from physical triggering conditions like fog, rain, and object-merging scenarios. By evaluating these ``faults'' within a standardized simulation environment, we demonstrate that perception-informed testing reveals latent operational risks that ideal SIL environments fail to capture, providing a scalable pathway for SOTIF (ISO 21448) validation.

**arXiv ID:** 2606.07186
</details>

<details>
<summary><strong>CAPE: Contrastive Action-conditioned Parallel Encoding for Embodied Planning</strong> - Cong Chen, Haowen Wang, Zhixiang Zhang, Pei Ren, Zhengping Che - [[pdf]](https://arxiv.org/pdf/2606.07304)</summary>

**Abstract:** Embodied agents need to predict the future consequences of candidate actions in order to plan effectively before execution. Existing visual dynamics models learn by reconstructing future visual states or rolling out dense latent representations, which spreads learning capacity across visually salient but planning-irrelevant content rather than the action-conditioned changes that drive manipulation outcomes. We propose CAPE, a Contrastive Action-conditioned Parallel Encoding framework that learns visual dynamics by distinguishing the future outcomes induced by different action sequences. Given an initial observation and a candidate action sequence, CAPE decodes the full future latent trajectory in a single forward pass and is trained with a Goal-Convergent Contrastive Objective that aligns predictions corresponding to the same future outcome while separating those corresponding to different outcomes. On real-world DROID and zero-shot transfer to RoboCasa, CAPE substantially outperforms prior baselines on future-state retrieval, offline action matching, and closed-loop planning, while notably reducing planning-time inference cost at long prediction horizons.

**arXiv ID:** 2606.07304
</details>

<details>
<summary><strong>Sustainability by Design in Decentralized Autonomous Organizations: An Empirical Review of Governance, Innovation, and Institutional Design</strong> - Yutian Wang, Luyao Zhang - [[pdf]](https://arxiv.org/pdf/2606.05667)</summary>

**Abstract:** Recent innovation theories on economics remain largely grounded in assumptions of hierarchical firms and closed organizational boundaries, offering limited insight into how innovation unfolds within decentralized, digitally native organizations. Decentralized Autonomous Organizations (DAOs) represent an emerging form of innovation ecosystem characterized by blockchain-based transparency, open participation, and token-driven governance, in which sustainability can be embedded directly into organizational design. This study compares two standards, ERC-8004 and Google A2A, who address the same agent interoperability question, while the former is governed by DAO and the latter by corporation consortium. They are examined through an LLM-powered comparative pipeline for large-scale governance discourse analysis, integrating automated annotation, neural topic modeling, and multi-layer network analysis to study socio-technical power structures. The study provides evidence-based insights for scholars, policymakers, and designers seeking to align innovation, technological governance, and sustainability in future organizational forms.

**arXiv ID:** 2606.05667
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (27 papers)</h2></summary>

<details>
<summary><strong>Declarative Skills for AI Agents in Knowledge-Grounded Tool-Use Workflows</strong> - M. Danish Lim, I. Danial Bin Sharudin, Wen Han Chen, Cedric Lim, Laura Wynter - [[pdf]](https://arxiv.org/pdf/2606.06923)</summary>

**Abstract:** We study orchestration mechanisms for tool-using AI agents in realistic customer-service workflows over an unstructured knowledge base. We argue that declarative agents -- AI agents equipped with natural-language skill files appended to the system prompt -- are an effective orchestration paradigm. Concretely, we compare (i) a DeclarativeAgent that reads three domain-specific skill files at inference time and decides its own control flow, (ii) an ImperativeAgent based on a programmatic state machine with explicit phases, and (iii) an unscaffolded baseline agent modeled after the $\tau$-Knowledge benchmark agent. Our ImperativeAgent is motivated by externalised-control inference as in Recursive Language Models and graph-based orchestration frameworks. We formalise the three agents as policy classes within a decentralised partially-observable Markov decision process and analyse their information-theoretic and structural properties; we then test the predicted differences empirically on five language models and two retrieval regimes. Our results show that retrieval quality is a dominant bottleneck for AI agents: when evidence is incomplete or skewed, all agents degrade substantially, and skill files cannot recover lost performance. Under high-quality retrieval, however, declarative skills consistently improve accuracy on procedural tasks and reduce orchestration errors, while the imperative state machine's brittleness does not reliably improve task success or compliance.

**arXiv ID:** 2606.06923
</details>

<details>
<summary><strong>Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning</strong> - Yijin Zhou, Linqian Zeng, Xiaoya Lu, Wenyuan Xie, Dongrui Liu, Junchi Yan, Jing Shao - [[pdf]](https://arxiv.org/pdf/2606.06976)</summary>

**Abstract:** Large language model (LLM)-based agents often make suboptimal tool-use decisions, including unsupported tool invocation and hallucinated direct responses, which may accumulate errors throughout multi-step interactions. Existing approaches mainly improve these behaviors through inference-time correction or coarse-grained reward signals based on decision outcomes and structured checklists, leaving the uncertainty characteristics of agent decisions underexplored. We observe that decision-oriented reinforcement learning tends to weaken the uncertainty separation between correct and incorrect actions, resulting in overconfident mistakes and weaker exploration signals. Therefore, we propose TRUST, which incorporates uncertainty quantification into reward design as a repulsive force for maintaining uncertainty separation, and labels lightweight key-turn annotations for unified post-training of multi-turn trajectories. Experimental results across diverse tool-use benchmarks show that TRUST consistently enhances both decision quality and agent performance while maintaining more reliable uncertainty estimates during optimization.

**arXiv ID:** 2606.06976
</details>

<details>
<summary><strong>The Sim-to-Real Gap of Foundation Model Agents: A Unified MDP Perspective</strong> - Xiaoou Liu, Tiejin Chen, Weibo Li, Xiyang Hu, Hua Wei - [[pdf]](https://arxiv.org/pdf/2606.07017)</summary>

**Abstract:** Foundation model agents are increasingly deployed for real-world decision-making, but suffer from the sim-to-real gap. While robotics and classical control have mature frameworks to address this gap, the foundation model community is treating agent robustness as an entirely novel phenomenon. Our paper proposes formalizing the foundation model agent evaluation and training gap as a classical sim-to-real problem structured entirely around the four elements of a Markov Decision Process, including Observation, Action, Transition, and Reward. In this paper, we set a comprehensive research agenda that translates classical discrepancies into the foundation model domain and advocates for adopting established solutions like domain randomization. We provide concrete examples, such as a multilingual tool calling to demonstrate how severe observation space gaps lead to operationally invalid actions despite correct semantic intent. Ultimately, this agenda aims to drive a paradigm shift, yielding a unified vocabulary and standardized stress test benchmarks to foster a new generation of highly trustworthy agents for reliable real-world applications.

**arXiv ID:** 2606.07017
</details>

<details>
<summary><strong>StainFlow: Entity-Stain Tracking and Evidence Linking for Process Rewards in GUI Agents</strong> - Haojie Hao, Longkun Hao, Yihang Lou, Yan Bai, Zhenyang Li, Zhichao Yang, Dongshuo Huang, Hongyu Lin, Lanqing Hong, Jiakai Wang, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2606.07027)</summary>

**Abstract:** Reinforcement Learning (RL) has become a promising approach for improving GUI Agents in long-horizon, stochastic digital environments, but trajectory-level success feedback is too sparse to provide reliable credit assignment for intermediate exploration steps. To mitigate this issue, recent studies introduce Process Reward Models (PRMs), which provide finer-grained training feedback through global milestone verification or local step-level evaluation. However, these methods still suffer from two level-specific limitations: global milestone decomposition is subjective and singular, making it difficult to accommodate the multiple valid execution paths in real GUI tasks, while fixed local judging windows may miss long-range key evidence or dilute the decision signal with irrelevant frames. Inspired by stain-tracing mechanisms in network flow analysis, we propose StainFlow, an entity-stain-flow process reward model for GUI Agents. To reduce the subjectivity of global partitioning, we introduce the Global Entity Stain Tracking module, which extracts visually verifiable task entities and tracks how their stain concentrations and states evolve along the trajectory, allowing task phases to be objectively separated by changes in the entity evidence flow. To improve the accuracy of local verification, we introduce the Local Stain Evidence Linking module. Centered on the triggering entities of each candidate key node, it retrieves relevant steps based on their stain concentrations and state changes, and dynamically constructs high-density evidence windows for verifying true key nodes. Extensive experiments on AndroidWorld and OGRBench show that StainFlow relatively improves online RL success by 3.2% and trajectory completion judgment accuracy by 1.8%.

**arXiv ID:** 2606.07027
</details>

<details>
<summary><strong>MacArena: Benchmarking Computer Use Agents on an Online macOS Environment</strong> - Victor Muryn, Maksym Shamrai, Sofiia Mazepa, Yehor Khodysko - [[pdf]](https://arxiv.org/pdf/2606.06560)</summary>

**Abstract:** Computer-use agents (CUAs) operate graphical user interfaces (GUIs) through vision and control primitives, and their capabilities have advanced rapidly, driven in part by standardized online evaluation benchmarks such as OSWorld, which serve both as evaluation tools and as training environments for reinforcement learning. However, macOS remains underserved in this landscape: the only existing benchmark, macOSWorld, covers a narrow slice of first-party applications with simpler tasks, and runs on x86 virtual machines incompatible with Apple Silicon. We introduce MacArena, a benchmark of 421 manually verified tasks spanning 50 applications that combines a curated port of OSWorld tasks, content sourced from macOSWorld, and 49 new macOS-native tasks, all running on Apple's native Virtualization framework on Apple Silicon. We argue that macOS presents distinct GUI challenges beyond what Linux-based benchmarks capture, and our evaluation supports this claim: strong model performance on existing benchmarks can reflect familiarity with task distributions rather than genuine cross-platform GUI competence. Notably, model rankings invert between ported and macOS-native tasks, with a leading model trailing by over 26% on the MacArena subset, suggesting that macOS poses a genuinely harder environment for current GUI agents.

**arXiv ID:** 2606.06560
</details>

<details>
<summary><strong>What Your Posts Reveal: A Benchmark and Agentic Framework for User-Level Privacy Leakage on Social Media</strong> - Zifan Peng, Yini Huang, Aiwen Lu, Qiming Ye, Peixian Zhang, Jingyi Zheng, Yule Liu, Xuechao Wang, Xinlei He, Jiaheng Wei - [[pdf]](https://arxiv.org/pdf/2606.06784)</summary>

**Abstract:** Public social media posts can reveal private information through weak cues scattered across text, images, or metadata. Such leakage is often cumulative and cross-post: cues that appear harmless in isolation may jointly expose a user's home, workplace, or routine. However, current research lacks a unified benchmark for user-level multimodal privacy leakage and an evaluation metric that captures exposure severity beyond binary accuracy.
To address these gaps, we propose SopriBench, a synthetic benchmark guided by leakage patterns abstracted from a private reference corpus of Rednote and Instagram accounts, covering 50 user profiles and 1,569 images with attributes, contextual sensitivity, granularity, leakage type, inference difficulty, and supporting evidence. We further introduce the Privacy Exposure Score (PES), which weights value granularity by contextual sensitivity. Inspired by abductive reasoning, we introduce Argus, a training-free agentic framework for cumulative leakage inference. Argus forms hypotheses from accumulated evidence, verifies supporting evidence, and aggregates cross-post cues into privacy profiles, achieving 0.55 PES, a 25% improvement over the strongest baseline, with the largest gain on cross-post leakage.

**arXiv ID:** 2606.06784
</details>

<details>
<summary><strong>Exploring Reinforcement Learning for Fluid Transitions Between Clinical Mental Healthcare and Everyday Wellness Support</strong> - Tony Wang, Qian Yang - [[pdf]](https://arxiv.org/pdf/2606.06800)</summary>

**Abstract:** Mental health struggles wax and wane, yet clinical and wellness interventions typically operate separately, causing frequent breakdowns at care transitions. We explore reinforcement learning (RL) as a means to build digital health systems that deliver clinical and wellness interventions proactively, as part of a coherent care journey. We ask: what complexities does designing such a system involve? We built a contextual bandit that dynamically selects journaling prompts from clinical and wellness repertoires to optimize for an overarching health goal (sustained journaling) and deployed it in a four-week exploratory study (N=38). We found that, first, many benefits of RL-optimized intervention sequences appeared only after interventions ended, raising the question: Should systems that offer coherent clinical-wellness care journeys include stepping-back periods? If so, when and how? Second, participants most engaged with RL-generated interventions deepened their engagement over time, while those most engaged with a constant intervention tended to burn out and drop out later. It raises the question: When should a system blending clinical and wellness interventions reduce intensity to prevent burnout in versus sustain it to maximize treatment gains?

**arXiv ID:** 2606.06800
</details>

<details>
<summary><strong>SCALE: Scalable Cross-Attention Learning with Extrapolation for Agentic Workflow Scheduling</strong> - Zhifei Xu, Jierui Lan, Zixuan Liang, Aiji Liang, Jinxi He - [[pdf]](https://arxiv.org/pdf/2606.06820)</summary>

**Abstract:** Agentic Large Language Model (LLM) systems decompose complex tasks into workflow Directed Acyclic Graphs (DAGs) whose primitives must be scheduled on heterogeneous clusters. Existing deep reinforcement learning (DRL) schedulers are tied to a fixed cluster size and require retraining whenever the number of servers changes. We propose SCALE (Scalable Cross-Attention Learning with Extrapolation), a DRL scheduler that generalizes to unseen cluster scales without fine-tuning. SCALE employs a cross-attention pointer network where task features query against server features, so the architecture accepts any number of servers by construction. We observe, however, that permutation-invariant architecture alone does not guarantee good performance at new scales - the attention feature undergoes distribution shift as the server count grows. To counter this, we introduce Structured Representation Regularization (SRR): a decorrelation loss combined with a KL penalty toward the standard normal, which keeps feature statistics stable regardless of input size. Trained on 16 nodes and tested directly on 32 and 48 nodes, SCALE reduces average response time by 8.9% at N=48 relative to the same architecture without SRR, confirming that explicit regularization is necessary to close the scale-generalization gap.

**arXiv ID:** 2606.06820
</details>

<details>
<summary><strong>Progress-SQL: Improving Reinforcement Learning for Text-to-SQL via Progressive Rewards</strong> - Shihao Zhang, Xiaoman Wang, Yuan Liu, Yunshi Lan, Weining Qian - [[pdf]](https://arxiv.org/pdf/2606.06825)</summary>

**Abstract:** Reinforcement learning has recently shown promise in improving large language models for Text-to-SQL generation, yet existing methods typically optimize one-shot rewards defined over a single SQL state. Such rewards provide limited guidance for iterative SQL correction and are insufficient to capture the improvement of multi-turn SQL refinement. In this paper, we propose Progress-SQL, a multi-turn reinforcement learning framework with progressive rewards for Text-to-SQL. Our approach introduces an Oracle-guided Diagnostic Tree (ODT), which abstracts SQL queries into clause-level structural profiles and produces diagnostic feedback for next-turn refinement. To provide dense and robust reward signals, we combine ODT-based structural alignment with lexical alignment and define a progressive reward that measures the improvement from the initial SQL to the final SQL. We further incorporate a progression latency reward that favors earlier correctness and an execution status reward that encourages recovery from the invalid SQL. Experiments on BIRD, Spider, and Spider robustness variants demonstrate that our method consistently improves Text-to-SQL performance across both primary and robustness evaluations.

**arXiv ID:** 2606.06825
</details>

<details>
<summary><strong>SlimSearcher: Training Efficiency-Aware Web Agents via Adaptive Reward Gating</strong> - Zequn Xie, Junjie Wang, Dan Yang, Jie Feng, Yue Shen, Jian Wang, Jinjie Gu - [[pdf]](https://arxiv.org/pdf/2606.07074)</summary>

**Abstract:** Deep research agents have demonstrated remarkable capabilities in complex information-seeking tasks, yet this power comes at a steep computational cost. Driven by accuracy-focused training paradigms, current models adopt brute-force strategies characterized by blind tool dependency and performative reasoning-generating long, redundant trajectories that are far from necessary for resolving these tasks, leading to wasteful tool calls and excessive token consumption. To overcome this efficiency trap, we propose SlimSearcher, a principled framework that pushes the Pareto frontier between accuracy and computational cost across both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). In the SFT stage, SlimSearcher employs Pareto-efficient filtration to distill trajectories that are both successful and economical, guiding the model toward inherently efficiency-aware search behaviors. During RL, we introduce Adaptive Reward Gating, a dynamic reward-shaping mechanism that evaluates relative tool and token efficiency within a sampled cohort. By cascading these adaptive efficiency metrics with a strict correctness gate, our approach effectively avoids the brevity bias associated with absolute penalties and mitigates reward hacking. Extensive experiments on long-horizon benchmarks, including GAIA, BrowseComp, and XBenchDeepSearch, demonstrate that SlimSearcher reduces average tool-call rounds by 17%-58% while maintaining or improving accuracy.

**arXiv ID:** 2606.07074
</details>

<details>
<summary><strong>Hierarchical Certified Semantic Commitment for Byzantine-Resilient LLM-Agent Collaboration</strong> - Haoran Xu, Lei Zhang, Iadh Ounis, Xianbin Wang - [[pdf]](https://arxiv.org/pdf/2606.07316)</summary>

**Abstract:** Byzantine collaboration among large-language-model agents requires a finality-control primitive: given delivered stochastic, structured natural-language proposals, the protocol must decide whether the round supports a commit, what kind of commit, or a typed safe abort. Naive aggregation hides this choice behind a single verdict; classical Byzantine fault tolerance hides it behind byte-identity that LLM proposals do not satisfy. We introduce Hierarchical Certified Semantic Commitment (H-CSC), a BFT-inspired protocol that converts embedding-derived finality signals over verdict-conditioned proposal groups into one of three typed outcomes: a semantic_commit (a 2f+1 within-verdict semantic core backs the verdict, emitting a parameter-bound digest over the quantised aggregate), a verdict_commit (strong verdict margin but dispersed semantic rationale, emitting a verdict-level certificate without claiming a semantic aggregate), or an explicit abort with a typed reason. The contribution is typed finality, not raw commit accuracy. On a controlled semantic-poisoning diagnostic (BCS_v1, 120 episodes), H-CSC commits with low angular deviation on BFT-feasible buckets (0.31 to 2.04 degrees) and aborts 100% of beyond-BFT rounds (n<3f+1) as intended. On a real LLM-agent claim-verification benchmark (MVR-50, 50 tasks) under paired static and rushing Byzantine attacks, H-CSC commits 0.90/0.92 with honest-reference-invalid rates of 0.02/0.00, statistically matching a strong certificate-emitting verdict-only baseline. Unlike that baseline, H-CSC also emits an embedding-backed semantic_commit digest on 74%/72% of rounds, supplying typed provenance. A strict-semantic ablation commits only 0.54/0.48, showing the verdict-level fallback is necessary for coverage (+0.36/+0.44) at the same <=0.04 safety floor; a 100-task cross-model check across four LLMs preserves invalid_hmaj within 0.00 to 0.03.

**arXiv ID:** 2606.07316
</details>

<details>
<summary><strong>Agentic Physical AI toward a Domain-Specific Foundation Model for Energy Systems: A Case Study on Nuclear Reactor Control</strong> - Yoon Pyo Lee, Samrendra Roy, Kazuma Kobayashi, Sajedul Talukder, Diab Abueidda, Seid Koric, Souvik Chakraborty, Syed Bahauddin Alam - [[pdf]](https://arxiv.org/pdf/2512.23292)</summary>

**Abstract:** The prevailing paradigm in AI for physical systems: scaling general-purpose foundation models toward universal multimodal reasoning, confronts a barrier at the control interface. Frontier vision-language models achieve only 50-53% accuracy on basic quantitative physics tasks, behaving as approximate guessers that preserve semantic plausibility while violating physical constraints. Safety-critical control demands outcome-space guarantees over executed actions, not parameter-space imitation. Here we present a pathway toward domain-specific foundation models through compact language models operating as Agentic Physical AI: policy optimization driven by physics-based simulator validation rather than perceptual inference. We train a 360M-parameter model on synthetic nuclear reactor scenarios scaled from 10^3 to 10^5 examples. Scaling produces strong, regime-dependent reliability gains under nominal simulated conditions, with variance collapse of approximately 500x and elimination of >10% terminal-power excursions on the sampled distribution. Despite balanced exposure to four actuation families, the model concentrates 95% of runtime execution on a single-bank strategy, without reinforcement learning or reward engineering. Representations transfer across simulators without architectural change. We position the system as a candidate decision component within a verification, monitoring, and defense-in-depth architecture, not as a stand-alone safety solution: the demonstrated behavior speaks to closed-loop reliability on a single-step task in simulation and does not yet address off-nominal operation, sensor faults, or uncertainty quantification.

**arXiv ID:** 2512.23292
</details>

<details>
<summary><strong>Beyond the Black Box: Interpretability of Agentic AI Tool Use</strong> - Hariom Tatsat, Ariye Shater - [[pdf]](https://arxiv.org/pdf/2605.06890)</summary>

**Abstract:** AI agents are promising for high-stakes enterprise workflows, but dependable deployment remains limited because tool-use failures are difficult to diagnose and control. Agents may skip required tool calls, invoke tools unnecessarily, or take actions whose consequence becomes visible only after execution. Existing observability methods are external: prompts reveal correlations, evaluations score outputs, and logs arrive only after the model has already acted. In long-horizon settings, these failures are costly because an early tool mistake can alter the rest of the trajectory, increase token consumption, and create downstream safety and security risk.
We introduce a mechanistic-interpretability toolkit built on Sparse Autoencoders (SAEs), which decompose activations into sparse internal features, and linear probes, lightweight classifiers that read signals from those features. The framework reads model states before each action and infers whether a tool is needed and how risky the next tool action is. It identifies the model layers and features most associated with tool decisions and tests their functional importance through feature ablation. We train the probes on multi-step trajectories from the NVIDIA Nemotron function-calling dataset and apply the same workflow to GPT-OSS 20B and Gemma 3 27B models.
The goal is not to replace external evaluation, but to add a missing layer: visibility into what the model signaled internally before action. This helps surface deeper causes of agent failure, especially in long-horizon runs where an early mistake can impact subsequent agent behavior. More broadly, the paper shows how mechanistic interpretability can support internal observability for monitoring tool calls and risk in agent systems.

**arXiv ID:** 2605.06890
</details>

<details>
<summary><strong>Robust Driving Control for Autonomous Vehicles: An Intelligent General-sum Constrained Adversarial Reinforcement Learning Approach</strong> - Junchao Fan, Qi Wei, Ruichen Zhang, Yang Lu, Jianhua Wang, Xiaolin Chang, Bo Ai - [[pdf]](https://arxiv.org/pdf/2510.09041)</summary>

**Abstract:** Deep reinforcement learning (DRL) has demonstrated remarkable success in developing autonomous driving policies. However, its vulnerability to adversarial attacks remains a critical barrier to real-world deployment. Although existing robust methods have achieved success, they still suffer from three key issues: (i) these methods are trained against myopic adversarial attacks, limiting their abilities to respond to more strategic threats, (ii) they have trouble causing truly safety-critical events (e.g., collisions), but instead often result in minor consequences, and (iii) these methods can introduce learning instability and policy drift during training due to the lack of robust constraints. To address these issues, we propose Intelligent General-sum Constrained Adversarial Reinforcement Learning (IGCARL), a novel robust autonomous driving approach that consists of a strategic targeted adversary and a robust driving agent. The strategic targeted adversary is designed to leverage the temporal decision-making capabilities of DRL to execute strategically coordinated multi-step attacks. In addition, it explicitly focuses on inducing safety-critical events by adopting a general-sum objective. The robust driving agent learns by interacting with the adversary to develop a robust autonomous driving policy against adversarial attacks. To ensure stable learning in adversarial environments and to mitigate policy drift caused by attacks, the agent is optimized under a constrained formulation. Extensive experiments show that IGCARL improves the success rate by at least 27.9% over state-of-the-art methods, demonstrating superior robustness to adversarial attacks and enhancing the safety and reliability of DRL-based autonomous driving.

**arXiv ID:** 2510.09041
</details>

<details>
<summary><strong>Improving Cross-Lingual Factual Recall via Consistency-Driven Reinforcement Learning</strong> - Jonathan von Rad, Louis Arts, George Burgess, Eleftheria Kolokytha, Harry O'Donnell, Ektor Oikonomidis Doumpas, Eduardo Sanchez, Yao Lu, Pontus Stenetorp - [[pdf]](https://arxiv.org/pdf/2606.06586)</summary>

**Abstract:** Large language models (LLMs) trained predominantly on English data encode substantial world knowledge, yet often fail to express it reliably in other languages, a phenomenon known as cross-lingual factual inconsistency. To study and address this, we introduce PolyFact, a large-scale parallel multilingual factual QA dataset containing 100K Wikidata-grounded facts across 12 typologically diverse languages. Using PolyFact, we compare light continual pretraining (CPT), supervised fine-tuning (SFT), and reinforcement learning via Group Relative Policy Optimization (GRPO) for improving cross-lingual factual recall in Qwen-2.5-7B and OLMo-2-1124-7B. We find that GRPO consistently outperforms SFT, improving both cross-lingual consistency and generalization to unseen languages, while CPT on parallel data yields limited additional gains. Mechanistic analyses further show that GRPO reorganizes multilingual routing by reducing language specialization in MLP layers and attention heads, thereby promoting more shared cross-lingual representations. We release our code, models, and dataset.

**arXiv ID:** 2606.06586
</details>

<details>
<summary><strong>Translate-R1: Cost-Aware Translation Tool Use via Reinforcement Learning</strong> - Pratik Jayarao, Chaitanya Dwivedi, Himanshu Gupta, Neeraj Varshney, Adithya M Devraj, Meet Vadera, Priyanka Nigam, Bing Yin - [[pdf]](https://arxiv.org/pdf/2606.06835)</summary>

**Abstract:** The performance gap across languages in LLMs is well documented, and closing it natively requires pretraining or fine-tuning on corpora that, for most languages, do not exist. Translation offers an alternative: converting an input into the model's dominant language unlocks its full capabilities at once. Applying translation to every input, however, is wasteful for languages the model already handles, while leaving the choice to the model fails in the opposite way, as LLMs are overconfident and skip the tool even when they cannot understand the input. Prior work resolves this with language-specific rules, domain heuristics, language identifiers, or external routers, each requiring manual engineering. We instead learn a single policy that decides when to translate from reward alone, developing language- and domain-adaptive introspection that assesses its own comprehension and invokes translation only when it cannot solve a task natively.
Using data built by our answer-preserving translation pipeline, we continue RL on the post-trained Qwen3-4B across 22 languages in 3 resource tiers (High, Low, XLow) and 5 domains, and introduce confidence-gated GSPO for cost-sensitive tool use. The gated policy lifts reward over the baseline by +4.6 on High, +23.5 on Low, and +17.5 on XLow. Against an unconstrained policy that almost always translates, it preserves full reward at 63% of the cost and is Pareto-optimal across 87% of the cost-sensitivity range. Additionally, to simulate behavior on a completely unseen language, we create 2 synthetic languages, where our gated policy improves +18.7 over the overconfident baseline that underutilizes the tool even on these incomprehensible inputs. The policy transfers zero-shot to 9 held-out languages, and we analyze how tool use emerges over training, per language and per domain.

**arXiv ID:** 2606.06835
</details>

<details>
<summary><strong>SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding</strong> - Yiqiao Jin, Rachneet Kaur, Zhen Zeng, Sumitra Ganesh, Srijan Kumar - [[pdf]](https://arxiv.org/pdf/2510.26615)</summary>

**Abstract:** Multi-page visual documents such as manuals, brochures, presentations, and posters convey key information through layout, colors, icons, and cross-slide references. While multimodal large language models (MLLMs) offer opportunities in document understanding, current systems struggle with complex, multi-page visual documents, particularly in fine-grained reasoning over elements and pages. We introduce SlideAgent, a versatile agentic framework for understanding multi-modal, multi-page, and multi-layout documents, especially slide decks. SlideAgent employs specialized agents and decomposes reasoning into three specialized levels--global, page, and element--to construct a structured, query-agnostic representation that captures both overarching themes and detailed visual or textual cues. During inference, SlideAgent selectively activates specialized agents for multi-level reasoning and integrates their outputs into coherent, context-aware answers. Extensive experiments show that SlideAgent significantly improves accuracy over both proprietary (+7.9%) and open-source models (+9.8%).

**arXiv ID:** 2510.26615
</details>

<details>
<summary><strong>Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents</strong> - Shijue Huang, Hangyu Guo, Guanting Dong, Chenxin Li, Junting Lu, Xinyu Geng, Zhaochen Su, Zhenyu Li, Shuang Chen, Hongru Wang, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2605.10832)</summary>

**Abstract:** Multimodal deep search requires an agent to solve open-world problems by chaining search, tool use, and visual reasoning over evolving textual and visual context. Two bottlenecks limit current systems. First, existing tool-use harnesses treat images returned by search, browsing, or transformation as transient outputs, so intermediate visual evidence cannot be re-consumed by later tools. Second, training data is usually built by fixed curation recipes that cannot track the target agent's evolving capability. To address these challenges, we first introduce a visual-native agent harness centered on an image bank reference protocol, which registers every tool-returned image as an addressable reference and makes intermediate visual evidence reusable by later tools. On top of this harness, On-policy Data Evolution (ODE) runs a closed-loop data generator that refines itself across rounds from rollouts of the policy being trained. This per-round refinement makes each round's data target what the current policy still needs to learn. The same framework supports both diverse supervised fine-tuning data and policy-aware reinforcement learning data curation, covering the full training lifecycle of the target agent. Across 8 multimodal deep search benchmarks, ODE improves the Qwen3-VL-8B agent from 24.9% to 39.0% on average, surpassing Gemini-2.5 Pro in standard agent-workflow setting (37.9%). At 30B, ODE raises the average score from 30.6% to 41.5%. Further analyses validate the effectiveness of image-bank reuse, especially on complex tasks requiring iterative visual refinement, while rollout-feedback evolution yields more grounded SFT traces and better policy-matched RL tasks than static synthesis.

**arXiv ID:** 2605.10832
</details>

<details>
<summary><strong>Reinforcement Learning from Denoising Feedback</strong> - Qi He, Huan Chen, Ya Guo, Huijia Zhu, Yi R. Fung, Baojian Zhou - [[pdf]](https://arxiv.org/pdf/2605.25638)</summary>

**Abstract:** Policy loss estimation remains a fundamental and long-standing challenge in reinforcement learning (RL) for diffusion language models (DLMs). We introduce Reinforcement Learning from Denoising Feedback (RLDF), a novel training paradigm that leverages feedback obtained from rollout and training processes to facilitate accurate and efficient policy loss estimation. To balance the trade-off between computational efficiency and estimation effectiveness, RLDF optimizes the model toward the clipped clean state from intermediate noisy states, combined with weighted timestep sampling over denoising timesteps. Extensive experiments demonstrate that RLDF achieves consistent and substantial improvements in both performance and generalizability across two representative DLM architectures, LLaDA and Dream, on multiple reasoning benchmarks. Our work lays a principled foundation for scalable reinforcement learning in diffusion language models. We build Drift, a training framework for DLMs, available at this https URL.

**arXiv ID:** 2605.25638
</details>

<details>
<summary><strong>Reinforcement Learning from Rich Feedback with Distributional DAgger</strong> - Rishabh Agrawal, Jacob Fein-Ashley, Paria Rashidinejad - [[pdf]](https://arxiv.org/pdf/2606.05152)</summary>

**Abstract:** Reasoning models have advanced rapidly, but the dominant reinforcement learning from verifiable rewards (RLVR) recipe remains surprisingly narrow: sample many responses and reward each with a single bit indicating whether the final answer is correct. Yet many settings provide rich feedback, including execution traces, tool outputs, expert corrections, and model self-evaluations. We study how to use such feedback through a distributional variant of the classic imitation learning algorithm DAgger, where the learner has local access to an expert distribution on states visited by the current policy. This yields a simple forward cross-entropy objective that admits a blackbox expert and whose sequence-level gradient {conduct rich credit assignment by propagating} future expert-student disagreement back to earlier decisions. We show that prior RL with self-distillation objectives based on reverse KL or Jensen-Shannon fail to guarantee monotonic policy improvement: even when the expert has higher reward, their updates may increase probability on worse actions. In contrast, we show that forward cross-entropy admits monotonic policy improvement and enjoys guarantees on regret. We further show that our objective optimizes a lower bound on teacher-weighted likelihood of success, leading to improved Pass@N. Empirically, our approach, DistIL, improves over RLVR and RL with self-distillation baselines across a variety of domains: scientific reasoning, coding, and solving hard mathematical problems.

**arXiv ID:** 2606.05152
</details>

<details>
<summary><strong>Performance Variation in Deep Reinforcement Learning</strong> - Haruto Tanaka, A. Rupam Mahmood - [[pdf]](https://arxiv.org/pdf/2606.06746)</summary>

**Abstract:** Deep reinforcement learning (RL) algorithms often suffer from low run-to-run robustness, manifesting as significant performance variation across independent runs of identically configured agents. Although this issue poses a spectrum of challenges across research and practice, relatively few studies develop methods to evaluate it; RL research instead often reports uncertainty in the estimated mean performance. In this paper, we outline the limitations of conventional uncertainty and variation estimates, particularly their misalignment with purpose and the risk of underreporting. We then propose an alternative percentile-based statistic and visualization method, min-max IPR and run-wise percentile highlighting, respectively. These percentile-based tools are easy to interpret and rely on standard properties of sample percentiles, providing rich information about run-to-run performance variation. We demonstrate this through three case studies. First, we show that LayerNorm and penultimate-layer normalizations narrow performance variation in PPO, whereas the variation is mostly unchanged in SAC. Second, we compare PPO, SAC, TD-MPC, and TD-MPC2, and show TD-MPC exhibits the least variation while being the most data efficient among the four. Finally, in a comparison of DQN and Rainbow on five Atari environments, we show that both algorithms exhibit similar levels of performance variation.

**arXiv ID:** 2606.06746
</details>

<details>
<summary><strong>SHAP-Guided Kernel Actor-Critic for Explainable Reinforcement Learning</strong> - Na Li, Hangguan Shan, Wei Ni, Wenjie Zhang, Xinyu Li - [[pdf]](https://arxiv.org/pdf/2512.05291)</summary>

**Abstract:** Actor-critic (AC) methods are a cornerstone of reinforcement learning (RL) but offer limited interpretability. Current explainable RL methods seldom use state attributions to assist training. Rather, they treat all state features equally, thereby neglecting the heterogeneous impacts of individual state dimensions on the reward. We propose RKHS-SHAP-based Advanced Actor-Critic (RSA2C), an attribution-aware, kernelized, two-timescale AC algorithm, including Actor, Value Critic, and Advantage Critic. The Actor is instantiated in a vector-valued reproducing kernel Hilbert space (RKHS) with a Mahalanobis-weighted operator-valued kernel, while the Value Critic and Advantage Critic reside in scalar RKHSs. These RKHS-enhanced components use sparsified dictionaries: the Value Critic maintains its own dictionary, while the Actor and Advantage Critic share one. State attributions, computed from the Value Critic via RKHS-SHAP (kernel mean embedding for on-manifold and conditional mean embedding for off-manifold expectations), are converted into Mahalanobis-gated weights that modulate Actor gradients and Advantage Critic targets. We derive a global, non-asymptotic convergence bound under state perturbations, showing stability through the perturbation-error term and efficiency through the convergence-error term. Empirical results on three continuous-control environments show that RSA2C achieves efficiency, stability, and interpretability. Our code is available at this https URL.

**arXiv ID:** 2512.05291
</details>

<details>
<summary><strong>Leveraging Error Diversity in Group Rollouts for Reinforcement Learning</strong> - Wenpu Liu, Yuqi Xu, Weichu Xie, Yongfu Zhu, Shuai Dong, Ziyue Wang, Wenqi Shao, Xiaoying Zhang, Tong Yang, Nan Duan, Jiaqi Wang - [[pdf]](https://arxiv.org/pdf/2605.17333)</summary>

**Abstract:** Reinforcement Learning from Verifiable Rewards (RLVR) typically samples multiple responses per prompt and assigns binary rewards based on individual correctness, yet the collective structure of the group output, specifically the distribution of errors, is largely discarded. We identify this as a missed opportunity: empirical analysis reveals that error diversity within a group is a strong predictor of training success, with problems eliciting diverse wrong answers benefiting substantially more from RLVR than those producing homogeneous failures. Motivated by this observation, we propose Error Diversity Advantage Shaping (EDAS), a lightweight, algorithm-agnostic technique that modulates the advantage signal for incorrect rollouts based on intra-group error diversity. EDAS amplifies penalties for dominant, repeated errors and attenuates penalties for rare, exploratory ones, thereby encouraging the model to maintain diverse reasoning paths and discouraging error perseveration. Crucially, EDAS operates as a simple post-hoc adjustment that can be seamlessly integrated into any RLVR algorithm. We validate EDAS on top of several mainstream RLVR methods across a series of models and seven challenging math benchmarks, demonstrating consistent improvements. Notably, EDAS yields an average improvement of 6.29 points over DAPO on Qwen3-8B across seven benchmarks, confirming that exploiting the latent information in group rollouts is a broadly effective strategy for strengthening RLVR.

**arXiv ID:** 2605.17333
</details>

<details>
<summary><strong>Affordance-Based Hierarchical Reinforcement Learning for Quadruped Pedipulation</strong> - Tuba Girgin, Jose Castelblanco, Gabriel Rodriguez, Emre Girgin, Cagri Kilic - [[pdf]](https://arxiv.org/pdf/2606.07506)</summary>

**Abstract:** The object manipulation capabilities of quadruped robots is an open research challenge. While previous studies have focused on low-level policy learning, task execution still relies on expert-designed high-level trajectories. Autonomous selection of both an affordable interaction point on the target object and an affordable robot base pose removes the need for pre-designed trajectories. This study proposes a three-level hierarchical reinforcement learning (RL) framework that utilizes pose affordances to guide the navigation policy, while the navigation policy drives the locomotion policy. In addition, the pedipulation policy is guided by interaction-point affordances, enabling object-centric pose alignment of the quadruped robot and effective end-effector manipulation planning. We train the proposed framework in the IsaacSim ecosystem and evaluate it in both simulation and real-world settings. We investigate the effectiveness of pose affordance across multiple scenarios in simulation while various object interaction tasks are validated on real-world setting forming an object-interaction dataset. The results show that the proposed framework can autonomously identify candidate poses based on their affordance and successfully execute object manipulation tasks in the real world without human guidance.

**arXiv ID:** 2606.07506
</details>

<details>
<summary><strong>A Human-Sensitive Controller: Adapting to Human Musculoskeletal Disorder-Related Constraints via Reinforcement Learning</strong> - Vitor Martins, Sara M. Cerqueira, Mercedes Balcells, Elazer R Edelman, Cristina P. Santos - [[pdf]](https://arxiv.org/pdf/2504.10102)</summary>

**Abstract:** Work-Related Musculoskeletal Disorders continue to be a major challenge in industrial environments, leading to reduced workforce participation, increased healthcare costs, and long-term disability. This study introduces a human-sensitive robotic system aimed at reintegrating individuals with a history of musculoskeletal disorders into standard job roles, while simultaneously optimizing ergonomic conditions for the broader workforce. This research leverages reinforcement learning (RL) to develop a human-aware control strategy for collaborative robots, focusing on optimizing ergonomic conditions and preventing pain during task execution. Two RL approaches, Q-Learning and Deep Q-Network (DQN), were implemented and tested to personalize control strategies based on individual user characteristics. Although experimental results revealed a simulation-to-real gap, a fine-tuning phase successfully adapted the policies to real-world conditions. DQN outperformed Q-Learning by completing tasks faster while maintaining zero pain risk and safe ergonomic levels, achieving on average 38% shorter task completion times across all tested anthropometries. The structured testing protocol confirmed the system's adaptability to diverse human anthropometries, underscoring the potential of RL-driven cobots to enable safer, more inclusive workplaces.

**arXiv ID:** 2504.10102
</details>

<details>
<summary><strong>ViVa: A Video-Generative Value Model for Robot Reinforcement Learning</strong> - Jindi Lv, Hao Li, Jie Li, Fankun Kong, Yang Wang, Pengfei Yi, Yifei Nie, Xiaofeng Wang, Zheng Zhu, Chaojun Ni, Qiuping Deng, Hengtao Li, Jiancheng Lv, Guan Huang - [[pdf]](https://arxiv.org/pdf/2604.08168)</summary>

**Abstract:** Vision-language-action (VLA) models have advanced robot manipulation through large-scale pretraining, but real-world deployment remains challenging due to partial observability and delayed feedback. Reinforcement learning addresses this via value functions, which assess task progress and guide policy improvement. However, existing value models built on vision-language models (VLMs) struggle to capture temporal dynamics and physical interactions, undermining reliable value estimation in long-horizon tasks. In this paper, we propose ViVa, a video-generative value model that repurposes a pretrained video generator to jointly predict future proprioception and a scalar value. By grounding value estimation in anticipated embodiment dynamics, ViVa leverages spatiotemporal priors to intrinsically couple value with foresight beyond static snapshots. ViVa achieves state-of-the-art results in metric-based evaluation across three tasks, producing reliable value signals that accurately track task progress and detect execution errors. Integrated into RECAP, it achieves an average success rate of 80%, highlighting the promise of video-generative models for value estimation.

**arXiv ID:** 2604.08168
</details>

<details>
<summary><strong>Trust, Geometry, and Rules: A Credibility-Aware Reinforcement Learning Framework for Safe USV Navigation under Uncertainty</strong> - Yuhang Zhang, Shuqi Chai, Yukang Zhang, Liusha Yang, Mingchuan Zhang, Wei Wang, Qingjiang Shi, Quanbo Ge - [[pdf]](https://arxiv.org/pdf/2605.26974)</summary>

**Abstract:** Autonomous navigation of Unmanned Surface Vehicles (USVs) that is safe and compliant with the International Regulations for Preventing Collisions at Sea (COLREGs) remains a formidable challenge in dynamic maritime environments, particularly when perception systems exhibit miscalibrated uncertainty. Existing Reinforcement Learning (RL)-based methods often falter because state-estimation errors induce unreliable belief states that mislead the value function, while discrete traffic rules introduce discontinuity in the learning objective. To address these challenges, we propose a framework integrating credibility-aware learning, geometric safety shielding, and continuous rule-aware embedding. First, Credibility-Weighted Value Learning (CW-VL) introduces a dynamic trust factor derived from the discrepancy between filter-estimated covariance and empirical error statistics to modulate the critic's heteroscedastic loss, preventing policy overfitting to noisy samples. Second, the Covariance-Inflated Velocity Obstacle (CI-VO) maps position-estimation uncertainty into set-wise angular margins, forming a conservative geometric shield that overrides hazardous exploratory actions. Third, Risk-Aware COLREGs Duty Embedding relaxes binary encounter duties into continuous rule-aware signals, providing smooth sector-transition information and suppressing oscillation from sparse rule rewards. Simulated encounter studies demonstrate improved training robustness against perceptual inconsistency and superior collision avoidance and COLREGs compliance over baselines.

**arXiv ID:** 2605.26974
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
