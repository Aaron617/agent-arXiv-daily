# Agent arXiv Daily

**Last Updated:** 2025-12-03 02:54:16

**Total Papers:** 80

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
<summary><strong>STRIDE: A Systematic Framework for Selecting AI Modalities - Agentic AI, AI Assistants, or LLM Calls</strong> - Shubhi Asthana, Bing Zhang, Chad DeLuca, Ruchi Mahindru, Hima Patel - [[pdf]](https://arxiv.org/pdf/2512.02228)</summary>

**Abstract:** The rapid shift from stateless large language models (LLMs) to autonomous, goal-driven agents raises a central question: When is agentic AI truly necessary? While agents enable multi-step reasoning, persistent memory, and tool orchestration, deploying them indiscriminately leads to higher cost, complexity, and risk.
We present STRIDE (Systematic Task Reasoning Intelligence Deployment Evaluator), a framework that provides principled recommendations for selecting between three modalities: (i) direct LLM calls, (ii) guided AI assistants, and (iii) fully autonomous agentic AI. STRIDE integrates structured task decomposition, dynamism attribution, and self-reflection requirement analysis to produce an Agentic Suitability Score, ensuring that full agentic autonomy is reserved for tasks with inherent dynamism or evolving context.
Evaluated across 30 real-world tasks spanning SRE, compliance, and enterprise automation, STRIDE achieved 92% accuracy in modality selection, reduced unnecessary agent deployments by 45%, and cut resource costs by 37%. Expert validation over six months in SRE and compliance domains confirmed its practical utility, with domain specialists agreeing that STRIDE effectively distinguishes between tasks requiring simple LLM calls, guided assistants, or full agentic autonomy. This work reframes agent adoption as a necessity-driven design decision, ensuring autonomy is applied only when its benefits justify the costs.

**arXiv ID:** 2512.02228
</details>

<details>
<summary><strong>IACT: A Self-Organizing Recursive Model for General AI Agents: A Technical White Paper on the Architecture Behind kragent.ai</strong> - Pengju Lu - [[pdf]](https://arxiv.org/pdf/2512.02605)</summary>

**Abstract:** This technical white paper introduces the Interactive Agents Call Tree (IACT), a computational model designed to address the limitations of static, hard-coded agent workflows. Unlike traditional systems that require pre-defined graphs or specialized programming, IACT operates as a general-purpose autonomous system driven purely by user dialogue. Given a high-level objective, the system autonomously grows a dynamic, recursive agent topology incrementally tailored to the problem's structure. This allows it to scale its organizational complexity to match open-ended tasks. To mitigate the error propagation inherent in unidirectional function calls, IACT introduces interactional redundancy by replacing rigid invocations with bidirectional, stateful dialogues. This mechanism enables runtime error correction and ambiguity resolution. We describe the architecture, design principles, and practical lessons behind the production deployment of this model in the this http URL system, presenting qualitative evidence from real-world workflows rather than exhaustive benchmark results.

**arXiv ID:** 2512.02605
</details>

<details>
<summary><strong>Young Children's Anthropomorphism of AI Chatbots and the Role of Parent Co-Presence</strong> - Pilyoung Kim, Jenna H. Chin, Yun Xie, Nolan Brady, Tom Yeh, Sujin Yang - [[pdf]](https://arxiv.org/pdf/2512.02179)</summary>

**Abstract:** Artificial Intelligence (AI) chatbots powered by a large language model (LLM) are entering young children's learning and play, yet little is known about how young children construe these agents or how such construals relate to engagement. We examined anthropomorphism of a social AI chatbot during collaborative storytelling and asked how children's attributions related to their behavior and prefrontal activation. Children at ages 5-6 (N = 23) completed three storytelling sessions: interacting with (1) an AI chatbot only, (2) a parent only, and (3) the AI and a parent together. After the sessions, children completed an interview assessing anthropomorphism toward both the AI chatbot and the parent. Behavioral engagement was indexed by the conversational turn count (CTC) ratio, and concurrent fNIRS measured oxygenated hemoglobin in bilateral vmPFC and dmPFC regions. Children reported higher anthropomorphism for parents than for the AI chatbot overall, although AI ratings were relatively high for perceptive abilities and epistemic states. Anthropomorphism was not associated with CTC. In the right dmPFC, higher perceptive scores were associated with greater activation during the AI-only condition and with lower activation during the AI+Parent condition. Exploratory analyses indicated that higher dmPFC activation during the AI-only condition correlated with higher end-of-session "scared" mood ratings. Findings suggest that stronger perceptive anthropomorphism can be associated with greater brain activation related to interpreting the AI's mental states, whereas parent co-presence may help some children interpret and regulate novel AI interactions. These results may have design implications for encouraging parent-AI co-use in early childhood.

**arXiv ID:** 2512.02179
</details>

<details>
<summary><strong>Spoken Conversational Agents with Large Language Models</strong> - Chao-Han Huck Yang, Andreas Stolcke, Larry Heck - [[pdf]](https://arxiv.org/pdf/2512.02593)</summary>

**Abstract:** Spoken conversational agents are converging toward voice-native LLMs. This tutorial distills the path from cascaded ASR/NLU to end-to-end, retrieval-and vision-grounded systems. We frame adaptation of text LLMs to audio, cross-modal alignment, and joint speech-text training; review datasets, metrics, and robustness across accents and compare design choices (cascaded vs. E2E, post-ASR correction, streaming). We link industrial assistants to current open-domain and task-oriented agents, highlight reproducible baselines, and outline open problems in privacy, safety, and evaluation. Attendees leave with practical recipes and a clear systems-level roadmap.

**arXiv ID:** 2512.02593
</details>

<details>
<summary><strong>Kardia-R1: Unleashing LLMs to Reason toward Understanding and Empathy for Emotional Support via Rubric-as-Judge Reinforcement Learning</strong> - Jiahao Yuan, Zhiqing Cui, Hanqing Wang, Yuansheng Gao, Yucheng Zhou, Usman Naseem - [[pdf]](https://arxiv.org/pdf/2512.01282)</summary>

**Abstract:** As web platforms evolve towards greater personalization and emotional complexity, conversational agents must transcend superficial empathy to demonstrate identity-aware emotional reasoning. However, existing systems face two limitations: (1) reliance on situation-centric datasets lacking persistent user identity, which hampers the capture of personalized affective nuances; and (2) dependence on opaque, coarse reward signals that hinder development of verifiable empathetic reasoning. To address these gaps, we introduce KardiaBench, a large-scale user-grounded benchmark comprising 178,080 QA pairs across 22,080 multi-turn conversations anchored to 671 real-world profiles. The dataset is constructed via a model-in-the-loop pipeline with iterative rubric-guided refinement to ensure psychological plausibility and persona consistency. This progressive empathy pipeline that integrates user comprehension, contextual reasoning, and emotion perception into conversations, followed by iterative critique and rubric-based refinement to ensure psychological plausibility, emotional fidelity, and persona consistency. Building on this, we propose Kardia-R1, a framework that trains models for interpretable, stepwise empathetic cognition. Kardia-R1 leverages Rubric-as-Judge Empathetic Reinforcement Learning (Rubric-ERL), a GRPO-based method that uses explainable, human-aligned rubric rewards to tightly couple user understanding, emotional inference, and supportive response generation. Extensive experiments across four LLM backbones demonstrate that Kardia-R1 consistently outperforms othet methods in emotion accuracy, empathy, relevance, persona consistency, and safety. Our dataset and model will be released at this https URL.

**arXiv ID:** 2512.01282
</details>

<details>
<summary><strong>Animating Language Practice: Engagement with Stylized Conversational Agents in Japanese Learning</strong> - Zackary Rackauckas, Julia Hirschberg - [[pdf]](https://arxiv.org/pdf/2507.06483)</summary>

**Abstract:** We explore Jouzu, a Japanese language learning application that integrates large language models with anime-inspired conversational agents. Designed to address challenges learners face in practicing natural and expressive dialogue, Jouzu combines stylized character personas with expressive text-to-speech to create engaging conversational scenarios. We conducted a two-week in-the-wild deployment with 52 Japanese learners to examine how such stylized agents influence engagement and learner experience. Our findings show that participants interacted frequently and creatively, with advanced learners demonstrating greater use of expressive forms. Participants reported that the anime-inspired style made practice more enjoyable and encouraged experimenting with different registers. We discuss how stylization shapes willingness to engage, the role of affect in sustaining practice, and design opportunities for culturally grounded conversational AI in computer-assisted language learning (CALL). By framing our findings as an exploration of design and engagement, we highlight opportunities for generalization beyond Japanese contexts and contribute to international HCI scholarship.

**arXiv ID:** 2507.06483
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful?</strong> - Lewen Yan, Jilin Mei, Tianyi Zhou, Lige Huang, Jie Zhang, Dongrui Liu, Jing Shao - [[pdf]](https://arxiv.org/pdf/2512.02261)</summary>

**Abstract:** LLM-based trading agents are increasingly deployed in real-world financial markets to perform autonomous analysis and execution. However, their reliability and robustness under adversarial or faulty conditions remain largely unexamined, despite operating in high-risk, irreversible financial environments. We propose TradeTrap, a unified evaluation framework for systematically stress-testing both adaptive and procedural autonomous trading agents. TradeTrap targets four core components of autonomous trading agents: market intelligence, strategy formulation, portfolio and ledger handling, and trade execution, and evaluates their robustness under controlled system-level perturbations. All evaluations are conducted in a closed-loop historical backtesting setting on real US equity market data with identical initial conditions, enabling fair and reproducible comparisons across agents and attacks. Extensive experiments show that small perturbations at a single component can propagate through the agent decision loop and induce extreme concentration, runaway exposure, and large portfolio drawdowns across both agent types, demonstrating that current autonomous trading agents can be systematically misled at the system level. Our code is available at this https URL.

**arXiv ID:** 2512.02261
</details>

<details>
<summary><strong>Semantic Trading: Agentic AI for Clustering and Relationship Discovery in Prediction Markets</strong> - Agostino Capponi, Alfio Gliozzo, Brian Zhu - [[pdf]](https://arxiv.org/pdf/2512.02436)</summary>

**Abstract:** Prediction markets allow users to trade on outcomes of real-world events, but are prone to fragmentation through overlapping questions, implicit equivalences, and hidden contradictions across markets. We present an agentic AI pipeline that autonomously (i) clusters markets into coherent topical groups using natural-language understanding over contract text and metadata, and (ii) identifies within-cluster market pairs whose resolved outcomes exhibit strong dependence, including same-outcome (correlated) and different-outcome (anti-correlated) relationships. Using a historical dataset of resolved markets on Polymarket, we evaluate the accuracy of the agent's relational predictions. We then translate discovered relationships into a simple trading strategy to quantify how these relationships map to actionable signals. Results show that agent-identified relationships achieve roughly 60-70% accuracy, and their induced trading strategies earn about 20% average returns over week-long horizons, highlighting the ability of agentic AI and large language models to uncover latent semantic structure in prediction markets.

**arXiv ID:** 2512.02436
</details>

<details>
<summary><strong>Process-Centric Analysis of Agentic Software Systems</strong> - Shuyang Liu, Yang Chen, Rahul Krishna, Saurabh Sinha, Jatin Ganhotra, Reyhan Jabbarvand - [[pdf]](https://arxiv.org/pdf/2512.02393)</summary>

**Abstract:** Agentic systems are modern software systems: they consist of orchestrated modules, expose interfaces, and are deployed in software pipelines. Unlike conventional programs, their execution (i.e., trajectories) is inherently stochastic and adaptive to the problem they are solving. Evaluation of such systems is often outcome-centric, judging their performance based on success or failure at the final step. This narrow focus overlooks detailed insights about such systems, failing to explain how agents reason, plan, act, or change their strategies over time. Inspired by the structured representation of conventional software systems as graphs, we introduce Graphectory to systematically encode the temporal and semantic relations in such software systems. Graphectory facilitates the design of process-centric metrics and analyses to assess the quality of agentic workflows independent of final success.
Using Graphectory, we analyze 4000 trajectories of two dominant agentic programming workflows, namely SWE-agent and OpenHands, with a combination of four backbone Large Language Models (LLMs), attempting to resolve SWE-bench Verified issues. Our fully automated analyses reveal that: (1) agents using richer prompts or stronger LLMs exhibit more complex Graphectory, reflecting deeper exploration, broader context gathering, and more thorough validation before patch submission; (2) agents' problem-solving strategies vary with both problem difficulty and the underlying LLM -- for resolved issues, the strategies often follow coherent localization-patching-validation steps, while unresolved ones exhibit chaotic, repetitive, or backtracking behaviors; (3) even when successful, agentic programming systems often display inefficient processes, leading to unnecessarily prolonged trajectories.

**arXiv ID:** 2512.02393
</details>

<details>
<summary><strong>WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning</strong> - Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, Sung Ju Hwang - [[pdf]](https://arxiv.org/pdf/2512.02425)</summary>

**Abstract:** Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average 8.4% performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning.

**arXiv ID:** 2512.02425
</details>

<details>
<summary><strong>PPTArena: A Benchmark for Agentic PowerPoint Editing</strong> - Michael Ofengenden, Yunze Man, Ziqi Pang, Yu-Xiong Wang - [[pdf]](https://arxiv.org/pdf/2512.03042)</summary>

**Abstract:** We introduce PPTArena, a benchmark for PowerPoint editing that measures reliable modifications to real slides under natural-language instructions. In contrast to image-PDF renderings or text-to-slide generation, PPTArena focuses on in-place editing across 100 decks, 2125 slides, and over 800 targeted edits covering text, charts, tables, animations, and master-level styles. Each case includes a ground-truth deck, a fully specified target outcome, and a dual VLM-as-judge pipeline that separately scores instruction following and visual quality using both structural diffs and slide images. Building on this setting, we propose PPTPilot, a structure-aware slide-editing agent that plans semantic edit sequences, routes between high-level programmatic tools and deterministic XML operations for precise control, and verifies outputs through an iterative plan-edit-check loop against task-specific constraints. In our experiments, PPTPilot outperforms strong proprietary agents and frontier VLM systems by over 10 percentage points on compound, layout-sensitive, and cross-slide edits, with particularly large gains in visual fidelity and deck-wide consistency. Despite these improvements, existing agents still underperform on long-horizon, document-scale tasks in PPTArena, highlighting the remaining challenges in reliable PPT editing.

**arXiv ID:** 2512.03042
</details>

<details>
<summary><strong>BountyBench: Dollar Impact of AI Agent Attackers and Defenders on Real-World Cybersecurity Systems</strong> - Andy K. Zhang, Joey Ji, Celeste Menders, Riya Dulepet, Thomas Qin, Ron Y. Wang, Junrong Wu, Kyleen Liao, Jiliang Li, Jinghan Hu, Sara Hong, Nardos Demilew, Shivatmica Murgai, Jason Tran, Nishka Kacheria, Ethan Ho, Denis Liu, Lauren McLane, Olivia Bruvik, Dai-Rong Han, Seungwoo Kim, Akhil Vyas, Cuiyuanxiu Chen, Ryan Li, Weiran Xu, Jonathan Z. Ye, Prerit Choudhary, Siddharth M. Bhatia, Vikram Sivashankar, Yuxuan Bao, Dawn Song, Dan Boneh, Daniel E. Ho, Percy Liang - [[pdf]](https://arxiv.org/pdf/2505.15216)</summary>

**Abstract:** AI agents have the potential to significantly alter the cybersecurity landscape. Here, we introduce the first framework to capture offensive and defensive cyber-capabilities in evolving real-world systems. Instantiating this framework with BountyBench, we set up 25 systems with complex, real-world codebases. To capture the vulnerability lifecycle, we define three task types: Detect (detecting a new vulnerability), Exploit (exploiting a given vulnerability), and Patch (patching a given vulnerability). For Detect, we construct a new success indicator, which is general across vulnerability types and provides localized evaluation. We manually set up the environment for each system, including installing packages, setting up server(s), and hydrating database(s). We add 40 bug bounties, which are vulnerabilities with monetary awards from \$10 to \$30,485, covering 9 of the OWASP Top 10 Risks. To modulate task difficulty, we devise a new strategy based on information to guide detection, interpolating from identifying a zero day to exploiting a given vulnerability. We evaluate 10 agents: Claude Code, OpenAI Codex CLI with o3-high and o4-mini, and custom agents with o3-high, GPT-4.1, Gemini 2.5 Pro Preview, Claude 3.7 Sonnet Thinking, Qwen3 235B A22B, Llama 4 Maverick, and DeepSeek-R1. Given up to three attempts, the top-performing agents are Codex CLI: o3-high (12.5% on Detect, mapping to \$3,720; 90% on Patch, mapping to \$14,152), Custom Agent: Claude 3.7 Sonnet Thinking (67.5% on Exploit), and Codex CLI: o4-mini (90% on Patch, mapping to \$14,422). Codex CLI: o3-high, Codex CLI: o4-mini, and Claude Code are more capable at defense, achieving higher Patch scores of 90%, 90%, and 87.5%, compared to Exploit scores of 47.5%, 32.5%, and 57.5% respectively; while the custom agents are relatively balanced between offense and defense, achieving Exploit scores of 17.5-67.5% and Patch scores of 25-60%.

**arXiv ID:** 2505.15216
</details>

<details>
<summary><strong>WebMall - A Multi-Shop Benchmark for Evaluating Web Agents [Technical Report]</strong> - Ralph Peeters, Aaron Steiner, Luca Schwarz, Julian Yuya Caspary, Christian Bizer - [[pdf]](https://arxiv.org/pdf/2508.13024)</summary>

**Abstract:** LLM-based web agents have the potential to automate long-running web tasks, such as searching for products in multiple e-shops and subsequently ordering the cheapest products that meet the users needs. Benchmarks for evaluating web agents either require agents to perform tasks online using the live Web or offline using simulated environments, which allow for the exact reproduction of the experimental setup. While DeepShop provides an online benchmark that requires agents to perform challenging shopping tasks, existing offline benchmarks such as WebShop, WebArena, or Mind2Web cover only comparatively simple e-commerce tasks that need to be performed against a single shop containing product data from a single source. What is missing is an e-commerce benchmark that simulates multiple shops containing heterogeneous product data and requires agents to perform complex tasks. We fill this gap by introducing WebMall, the first offline multi-shop benchmark for evaluating web agents on challenging comparison shopping tasks. WebMall consists of four simulated shops populated with product data extracted from the Common Crawl. The WebMall tasks range from specific product searches and price comparisons to advanced queries for complementary or substitute products, as well as checkout processes. We validate WebMall using eight agents that differ in observation space, availability of short-term memory, and the employed LLM. The validation highlights the difficulty of the benchmark, with even the best-performing agents achieving task completion rates below 55% in the task categories cheapest product search and vague product search.

**arXiv ID:** 2508.13024
</details>

<details>
<summary><strong>PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM</strong> - Siqi Liang, Yudi Zhang, Yue Guo - [[pdf]](https://arxiv.org/pdf/2511.17467)</summary>

**Abstract:** We propose a novel framework for persona-based language model system, motivated by the need for personalized AI agents that adapt to individual user preferences. In our approach, the agent embodies the user's "persona" (e.g. user profile or taste) and is powered by a large language model (LLM). To enable the agent to leverage rich contextual information, we introduce a Knowledge-Graph-enhanced Retrieval-Augmented Generation (Graph RAG) mechanism that constructs an LLM-derived graph index of relevant documents and summarizes communities of related information. Our framework generates personalized prompts by combining: (1) a summary of the user's historical behaviors and preferences extracted from the knowledge graph, and (2) relevant global interaction patterns identified through graph-based community detection. This dynamic prompt engineering approach allows the agent to maintain consistent persona-aligned behaviors while benefiting from collective knowledge. On the LaMP benchmark, our method improves news categorization F1 by 11.1%, movie tagging F1 by 56.1%, and reduces product rating MAE by 10.4% over prior methods. Our code is available at this https URL

**arXiv ID:** 2511.17467
</details>

<details>
<summary><strong>nuScenes Revisited: Progress and Challenges in Autonomous Driving</strong> - Whye Kit Fong, Venice Erin Liong, Kok Seang Tan, Holger Caesar - [[pdf]](https://arxiv.org/pdf/2512.02448)</summary>

**Abstract:** Autonomous Vehicles (AV) and Advanced Driver Assistance Systems (ADAS) have been revolutionized by Deep Learning. As a data-driven approach, Deep Learning relies on vast amounts of driving data, typically labeled in great detail. As a result, datasets, alongside hardware and algorithms, are foundational building blocks for the development of AVs. In this work we revisit one of the most widely used autonomous driving datasets: the nuScenes dataset. nuScenes exemplifies key trends in AV development, being the first dataset to include radar data, to feature diverse urban driving scenes from two continents, and to be collected using a fully autonomous vehicle operating on public roads, while also promoting multi-modal sensor fusion, standardized benchmarks, and a broad range of tasks including perception, localization \& mapping, prediction and planning. We provide an unprecedented look into the creation of nuScenes, as well as its extensions nuImages and Panoptic nuScenes, summarizing many technical details that have hitherto not been revealed in academic publications. Furthermore, we trace how the influence of nuScenes impacted a large number of other datasets that were released later and how it defined numerous standards that are used by the community to this day. Finally, we present an overview of both official and unofficial tasks using the nuScenes dataset and review major methodological developments, thereby offering a comprehensive survey of the autonomous driving literature, with a particular focus on nuScenes.

**arXiv ID:** 2512.02448
</details>

<details>
<summary><strong>LAP: Fast LAtent Diffusion Planner with Fine-Grained Feature Distillation for Autonomous Driving</strong> - Jinhao Zhang, Wenlong Xia, Zhexuan Zhou, Youmin Gong, Jie Mei - [[pdf]](https://arxiv.org/pdf/2512.00470)</summary>

**Abstract:** Diffusion models have demonstrated strong capabilities for modeling human-like driving behaviors in autonomous driving, but their iterative sampling process induces substantial latency, and operating directly on raw trajectory points forces the model to spend capacity on low-level kinematics, rather than high-level multi-modal semantics. To address these limitations, we propose LAtent Planner (LAP), a framework that plans in a VAE-learned latent space that disentangles high-level intents from low-level kinematics, enabling our planner to capture rich, multi-modal driving strategies. We further introduce a fine-grained feature distillation mechanism to guide a better interaction and fusion between the high-level semantic planning space and the vectorized scene context. Notably, LAP can produce high-quality plans in one single denoising step, substantially reducing computational overhead. Through extensive evaluations on the large-scale nuPlan benchmark, LAP achieves state-of-the-art closed-loop performance among learning-based planning methods, while demonstrating an inference speed-up of at most 10 times over previous SOTA approaches. Code will be released at: this https URL.

**arXiv ID:** 2512.00470
</details>

<details>
<summary><strong>Proactive Agentic Whiteboards: Enhancing Diagrammatic Learning</strong> - Suveen Ellawela, Sashenka Gamage, Dinithi Dissanayake - [[pdf]](https://arxiv.org/pdf/2512.01234)</summary>

**Abstract:** Educators frequently rely on diagrams to explain complex concepts during lectures, yet creating clear and complete visual representations in real time while simultaneously speaking can be cognitively demanding. Incomplete or unclear diagrams may hinder student comprehension, as learners must mentally reconstruct missing information while following the verbal explanation. Inspired by advances in code completion tools, we introduce DrawDash, an AI-powered whiteboard assistant that proactively completes and refines educational diagrams through multimodal understanding. DrawDash adopts a TAB-completion interaction model: it listens to spoken explanations, detects intent, and dynamically suggests refinements that can be accepted with a single keystroke. We demonstrate DrawDash across four diverse teaching scenarios, spanning topics from computer science and web development to biology. This work represents an early exploration into reducing instructors' cognitive load and improving diagram-based pedagogy through real-time, speech-driven visual assistance, and concludes with a discussion of current limitations and directions for formal classroom evaluation.

**arXiv ID:** 2512.01234
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Benchmarking LLM Agents for Wealth-Management Workflows</strong> - Rory Milsom - [[pdf]](https://arxiv.org/pdf/2512.02230)</summary>

**Abstract:** Modern work relies on an assortment of digital collaboration tools, yet routine processes continue to suffer from human error and delay. To address this gap, this dissertation extends TheAgentCompany with a finance-focused environment and investigates whether a general purpose LLM agent can complete representative wealth-management tasks both accurately and economically. This study introduces synthetic domain data, enriches colleague simulations, and prototypes an automatic task-generation pipeline. The study aims to create and assess an evaluation set that can meaningfully measure an agent's fitness for assistant-level wealth management work. We construct a benchmark of 12 task-pairs for wealth management assistants spanning retrieval, analysis, and synthesis/communication, with explicit acceptance criteria and deterministic graders. We seeded a set of new finance-specific data and introduced a high vs. low-autonomy variant of every task. The paper concluded that agents are limited less by mathematical reasoning and more so by end-to-end workflow reliability, and meaningfully affected by autonomy level, and that incorrect evaluation of models have hindered benchmarking.

**arXiv ID:** 2512.02230
</details>

<details>
<summary><strong>When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents</strong> - Tsimur Hadeliya, Mohammad Ali Jauhar, Nidhi Sakpal, Diogo Cruz - [[pdf]](https://arxiv.org/pdf/2512.02445)</summary>

**Abstract:** Solving complex or long-horizon problems often requires large language models (LLMs) to use external tools and operate over a significantly longer context window. New LLMs enable longer context windows and support tool calling capabilities. Prior works have focused mainly on evaluation of LLMs on long-context prompts, leaving agentic setup relatively unexplored, both from capability and safety perspectives. Our work addresses this gap. We find that LLM agents could be sensitive to length, type, and placement of the context, exhibiting unexpected and inconsistent shifts in task performance and in refusals to execute harmful requests. Models with 1M-2M token context windows show severe degradation already at 100K tokens, with performance drops exceeding 50\% for both benign and harmful tasks. Refusal rates shift unpredictably: GPT-4.1-nano increases from $\sim$5\% to $\sim$40\% while Grok 4 Fast decreases from $\sim$80\% to $\sim$10\% at 200K tokens. Our work shows potential safety issues with agents operating on longer context and opens additional questions on the current metrics and paradigm for evaluating LLM agent safety on long multi-step tasks. In particular, our results on LLM agents reveal a notable divergence in both capability and safety performance compared to prior evaluations of LLMs on similar criteria.

**arXiv ID:** 2512.02445
</details>

<details>
<summary><strong>In-Context Distillation with Self-Consistency Cascades: A Simple, Training-Free Way to Reduce LLM Agent Costs</strong> - Vishnu Sarukkai, Asanshay Gupta, James Hong, Michaël Gharbi, Kayvon Fatahalian - [[pdf]](https://arxiv.org/pdf/2512.02543)</summary>

**Abstract:** The world currently has an abundance of ideas for how to use new LLM agents, and developers seek to rapidly prototype and test new agentic designs. However, executing agents at scale using high-capacity LLMs incurs high inference costs. We propose a simple method for reducing LLM agent inference costs without incurring the development friction costs associated with LLM fine-tuning (long training cycles, optimization hyperparameter tweaking loops) or manual prompt engineering (laborious trial and error). Most importantly, we introduce $\textit{in-context distillation}$, which adapts the idea of knowledge distillation (training a low cost-student model to mimic a high-cost teacher) to an in-context learning setting. Our approach retrieves relevant teacher demonstrations at each agent step and provides them to the student as in-context examples, enabling the student to imitate teacher behavior on-the-fly. We combine in-context distillation with the established idea of $\textit{self-consistency cascades}$ to know when the trust the student. This adaptive strategy realizes the cost benefits of model specialization while preserving the productivity of working with frozen models. On the multi-step embodied reasoning benchmark ALFWorld, our method matches teacher-level accuracy at $\textbf{2.5$\times$ lower cost}$, reducing per-episode costs from \$0.059 to \$0.024. The upfront demonstration cost amortizes after just 843 episodes, yielding cumulative savings exceeding \$34,900 at deployment scale (1M episodes). On AppWorld, a complex agent benchmark requiring multi-step API workflows, we shift the Pareto frontier by achieving a $\textbf{2$\times$ cost reduction}$ at iso-accuracy. By reducing operational costs while maintaining rapid experimentation cycles with frozen models, our approach makes advanced agentic systems economically viable for a broader range of applications.

**arXiv ID:** 2512.02543
</details>

<details>
<summary><strong>Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers</strong> - Tuan Nguyen, Long Tran-Thanh - [[pdf]](https://arxiv.org/pdf/2510.09330)</summary>

**Abstract:** Ensuring that large language models (LLMs) comply with safety requirements is a central challenge in AI deployment. Existing alignment approaches primarily operate during training, such as through fine-tuning or reinforcement learning from human feedback, but these methods are costly and inflexible, requiring retraining whenever new requirements arise. Recent efforts toward inference-time alignment mitigate some of these limitations but still assume access to model internals, which is impractical, and not suitable for third party stakeholders who do not have access to the models. In this work, we propose a model-independent, black-box framework for safety alignment that does not require retraining or access to the underlying LLM architecture. As a proof of concept, we address the problem of trading off between generating safe but uninformative answers versus helpful yet potentially risky ones. We formulate this dilemma as a two-player zero-sum game whose minimax equilibrium captures the optimal balance between safety and helpfulness. LLM agents operationalize this framework by leveraging a linear programming solver at inference time to compute equilibrium strategies. Our results demonstrate the feasibility of black-box safety alignment, offering a scalable and accessible pathway for stakeholders, including smaller organizations and entities in resource-constrained settings, to enforce safety across rapidly evolving LLM ecosystems.

**arXiv ID:** 2510.09330
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (25 papers)</h2></summary>

<details>
<summary><strong>DialogGuard: Multi-Agent Psychosocial Safety Evaluation of Sensitive LLM Responses</strong> - Han Luo, Guy Laban - [[pdf]](https://arxiv.org/pdf/2512.02282)</summary>

**Abstract:** Large language models (LLMs) now mediate many web-based mental- health, crisis, and other emotionally sensitive services, yet their psychosocial safety in these settings remains poorly understood and weakly evaluated. We present DialogGuard, a multi-agent frame- work for assessing psychosocial risks in LLM-generated responses along five high-severity dimensions: privacy violations, discrimi- natory behaviour, mental manipulation, psychological harm, and insulting behaviour. DialogGuard can be applied to diverse gen- erative models through four LLM-as-a-judge pipelines, including single-agent scoring, dual-agent correction, multi-agent debate, and stochastic majority voting, grounded in a shared three-level rubric usable by both human annotators and LLM judges. Using PKU-SafeRLHF with human safety annotations, we show that multi- agent mechanisms detect psychosocial risks more accurately than non-LLM baselines and single-agent judging; dual-agent correction and majority voting provide the best trade-off between accuracy, alignment with human ratings, and robustness, while debate attains higher recall but over-flags borderline cases. We release Dialog- Guard as open-source software with a web interface that provides per-dimension risk scores and explainable natural-language ratio- nales. A formative study with 12 practitioners illustrates how it supports prompt design, auditing, and supervision of web-facing applications for vulnerable users.

**arXiv ID:** 2512.02282
</details>

<details>
<summary><strong>Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games</strong> - Ran Zhang, Kun Ouyang, Tiancheng Ma, Yida Yang, Dong Fang - [[pdf]](https://arxiv.org/pdf/2512.02358)</summary>

**Abstract:** Optimizing numerical systems and mechanism design is crucial for enhancing player experience in Massively Multiplayer Online (MMO) games. Traditional optimization approaches rely on large-scale online experiments or parameter tuning over predefined statistical models, which are costly, time-consuming, and may disrupt player experience. Although simplified offline simulation systems are often adopted as alternatives, their limited fidelity prevents agents from accurately mimicking real player reasoning and reactions to interventions. To address these limitations, we propose a generative agent-based MMO simulation system empowered by Large Language Models (LLMs). By applying Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) on large-scale real player behavioral data, we adapt LLMs from general priors to game-specific domains, enabling realistic and interpretable player decision-making. In parallel, a data-driven environment model trained on real gameplay logs reconstructs dynamic in-game systems. Experiments demonstrate strong consistency with real-world player behaviors and plausible causal responses under interventions, providing a reliable, interpretable, and cost-efficient framework for data-driven numerical design optimization.

**arXiv ID:** 2512.02358
</details>

<details>
<summary><strong>PaperDebugger: A Plugin-Based Multi-Agent System for In-Editor Academic Writing, Review, and Editing</strong> - Junyi Hou, Andre Lin Huikai, Nuo Chen, Yiwei Gong, Bingsheng He - [[pdf]](https://arxiv.org/pdf/2512.02589)</summary>

**Abstract:** Large language models are increasingly embedded into academic writing workflows, yet existing assistants remain external to the editor, preventing deep interaction with document state, structure, and revision history. This separation makes it impossible to support agentic, context-aware operations directly within LaTeX editors such as Overleaf. We present PaperDebugger, an in-editor, multi-agent, and plugin-based academic writing assistant that brings LLM-driven reasoning directly into the writing environment. Enabling such in-editor interaction is technically non-trivial: it requires reliable bidirectional synchronization with the editor, fine-grained version control and patching, secure state management, multi-agent scheduling, and extensible communication with external tools. PaperDebugger addresses these challenges through a Chrome-approved extension, a Kubernetes-native orchestration layer, and a Model Context Protocol (MCP) toolchain that integrates literature search, reference lookup, document scoring, and revision pipelines. Our demo showcases a fully integrated workflow, including localized edits, structured reviews, parallel agent execution, and diff-based updates, encapsulated within a minimal-intrusion user interface (UI). Early aggregated analytics demonstrate active user engagement and validate the practicality of an editor-native, agentic writing assistant. More details about this demo and video could be found at this https URL.

**arXiv ID:** 2512.02589
</details>

<details>
<summary><strong>Aetheria: A multimodal interpretable content safety framework based on multi-agent debate and collaboration</strong> - Yuxiang He, Jian Zhao, Yuchen Yuan, Tianle Zhang, Wei Cai, Haojie Cheng, Ziyan Shi, Ming Zhu, Haichuan Tang, Chi Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2512.02530)</summary>

**Abstract:** The exponential growth of digital content presents significant challenges for content safety. Current moderation systems, often based on single models or fixed pipelines, exhibit limitations in identifying implicit risks and providing interpretable judgment processes. To address these issues, we propose Aetheria, a multimodal interpretable content safety framework based on multi-agent debate and this http URL a collaborative architecture of five core agents, Aetheria conducts in-depth analysis and adjudication of multimodal content through a dynamic, mutually persuasive debate mechanism, which is grounded by RAG-based knowledge this http URL experiments on our proposed benchmark (AIR-Bench) validate that Aetheria not only generates detailed and traceable audit reports but also demonstrates significant advantages over baselines in overall content safety accuracy, especially in the identification of implicit risks. This framework establishes a transparent and interpretable paradigm, significantly advancing the field of trustworthy AI content moderation.

**arXiv ID:** 2512.02530
</details>

<details>
<summary><strong>Enhancing Automated Paper Reproduction via Prompt-Free Collaborative Agents</strong> - Zijie Lin, Qilin Cai, Liang Shen, Mingjun Xiao - [[pdf]](https://arxiv.org/pdf/2512.02812)</summary>

**Abstract:** Automated paper reproduction has emerged as a promising approach to accelerate scientific research, employing multi-step workflow frameworks to systematically convert academic papers into executable code. However, existing frameworks often lack mechanisms to verify and refine the outputs at each generation step, or rely heavily on manually designed prompts for self-refinement, which limits their adaptability and scalability. To address these limitations, we propose a prompt-free collaborative agent framework that automatically enhances the quality of paper-to-code generation. Our approach employs two collaborative agents: a verification agent that examines whether the outputs at each step satisfy the requirements specified in the corresponding system prompt, and a refinement agent that revises the outputs based on the identified issues. Unlike previous methods that require human experts to craft specific refinement prompts for each step, our framework achieves automatic verification and improvement by leveraging only the original system prompts. We integrate our collaborative agents into the Paper2Code framework and conduct comprehensive experiments on PaperBench Code-Dev and Paper2CodeBench datasets. Experimental results demonstrate that our approach significantly improves the accuracy and completeness of reproduced code, achieving performance gains of approximately 15\% and 13\%, respectively, compared to the baseline without our agents. Furthermore, comparative experiments against Self-Refine validate the robustness and consistency of our prompt-free approach across different datasets.

**arXiv ID:** 2512.02812
</details>

<details>
<summary><strong>A Knowledge-Based Language Model: Deducing Grammatical Knowledge in a Multi-Agent Language Acquisition Simulation</strong> - David Ph. Shakouri, Crit Cremers, Niels O. Schiller - [[pdf]](https://arxiv.org/pdf/2512.02195)</summary>

**Abstract:** This paper presents an initial study performed by the MODOMA system. The MODOMA is a computational multi-agent laboratory environment for unsupervised language acquisition experiments such that acquisition is based on the interaction between two language models, an adult and a child agent. Although this framework employs statistical as well as rule-based procedures, the result of language acquisition is a knowledge-based language model, which can be used to generate and parse new utterances of the target language. This system is fully parametrized and researchers can control all aspects of the experiments while the results of language acquisition, that is, the acquired grammatical knowledge, are explicitly represented and can be consulted. Thus, this system introduces novel possibilities for conducting computational language acquisition experiments. The experiments presented by this paper demonstrate that functional and content categories can be acquired and represented by the daughter agent based on training and test data containing different amounts of exemplars generated by the adult agent. Interestingly, similar patterns, which are well-established for human-generated data, are also found for these machine-generated data. As the procedures resulted in the successful acquisition of discrete grammatical categories by the child agent, these experiments substantiate the validity of the MODOMA approach to modelling language acquisition.

**arXiv ID:** 2512.02195
</details>

<details>
<summary><strong>WISE: Weighted Iterative Society-of-Experts for Robust Multimodal Multi-Agent Debate</strong> - Anoop Cherian, River Doyle, Eyal Ben-Dov, Suhas Lohit, Kuan-Chuan Peng - [[pdf]](https://arxiv.org/pdf/2512.02405)</summary>

**Abstract:** Recent large language models (LLMs) are trained on diverse corpora and tasks, leading them to develop complementary strengths. Multi-agent debate (MAD) has emerged as a popular way to leverage these strengths for robust reasoning, though it has mostly been applied to language-only tasks, leaving its efficacy on multimodal problems underexplored. In this paper, we study MAD for solving vision-and-language reasoning problems. Our setup enables generalizing the debate protocol with heterogeneous experts that possess single- and multi-modal capabilities. To this end, we present Weighted Iterative Society-of-Experts (WISE), a generalized and modular MAD framework that partitions the agents into Solvers, that generate solutions, and Reflectors, that verify correctness, assign weights, and provide natural language feedback. To aggregate the agents' solutions across debate rounds, while accounting for variance in their responses and the feedback weights, we present a modified Dawid-Skene algorithm for post-processing that integrates our two-stage debate model. We evaluate WISE on SMART-840, VisualPuzzles, EvoChart-QA, and a new SMART-840++ dataset with programmatically generated problem instances of controlled difficulty. Our results show that WISE consistently improves accuracy by 2-7% over the state-of-the-art MAD setups and aggregation methods across diverse multimodal tasks and LLM configurations.

**arXiv ID:** 2512.02405
</details>

<details>
<summary><strong>UCAgents: Unidirectional Convergence for Visual Evidence Anchored Multi-Agent Medical Decision-Making</strong> - Qianhan Feng, Zhongzhen Huang, Yakun Zhu, Xiaofan Zhang, Qi Dou - [[pdf]](https://arxiv.org/pdf/2512.02485)</summary>

**Abstract:** Vision-Language Models (VLMs) show promise in medical diagnosis, yet suffer from reasoning detachment, where linguistically fluent explanations drift from verifiable image evidence, undermining clinical trust. Recent multi-agent frameworks simulate Multidisciplinary Team (MDT) debates to mitigate single-model bias, but open-ended discussions amplify textual noise and computational cost while failing to anchor reasoning to visual evidence, the cornerstone of medical decision-making. We propose UCAgents, a hierarchical multi-agent framework enforcing unidirectional convergence through structured evidence auditing. Inspired by clinical workflows, UCAgents forbids position changes and limits agent interactions to targeted evidence verification, suppressing rhetorical drift while amplifying visual signal extraction. In UCAgents, a one-round inquiry discussion is introduced to uncover potential risks of visual-textual misalignment. This design jointly constrains visual ambiguity and textual noise, a dual-noise bottleneck that we formalize via information theory. Extensive experiments on four medical VQA benchmarks show UCAgents achieves superior accuracy (71.3% on PathVQA, +6.0% over state-of-the-art) with 87.7% lower token cost, the evaluation results further confirm that UCAgents strikes a balance between uncovering more visual evidence and avoiding confusing textual interference. These results demonstrate that UCAgents exhibits both diagnostic reliability and computational efficiency critical for real-world clinical deployment. Code is available at this https URL.

**arXiv ID:** 2512.02485
</details>

<details>
<summary><strong>Beyond Single-Agent Safety: A Taxonomy of Risks in LLM-to-LLM Interactions</strong> - Piercosma Bisconti, Marcello Galisai, Federico Pierucci, Marcantonio Bracale, Matteo Prandi - [[pdf]](https://arxiv.org/pdf/2512.02682)</summary>

**Abstract:** This paper examines why safety mechanisms designed for human-model interaction do not scale to environments where large language models (LLMs) interact with each other. Most current governance practices still rely on single-agent safety containment, prompts, fine-tuning, and moderation layers that constrain individual model behavior but leave the dynamics of multi-model interaction ungoverned. These mechanisms assume a dyadic setting: one model responding to one user under stable oversight. Yet research and industrial development are rapidly shifting toward LLM-to-LLM ecosystems, where outputs are recursively reused as inputs across chains of agents. In such systems, local compliance can aggregate into collective failure even when every model is individually aligned. We propose a conceptual transition from model-level safety to system-level safety, introducing the framework of the Emergent Systemic Risk Horizon (ESRH) to formalize how instability arises from interaction structure rather than from isolated misbehavior. The paper contributes (i) a theoretical account of collective risk in interacting LLMs, (ii) a taxonomy connecting micro, meso, and macro-level failure modes, and (iii) a design proposal for InstitutionalAI, an architecture for embedding adaptive oversight within multi-agent systems.

**arXiv ID:** 2512.02682
</details>

<details>
<summary><strong>A process algebraic framework for multi-agent dynamic epistemic systems</strong> - Alessandro Aldini - [[pdf]](https://arxiv.org/pdf/2407.17537)</summary>

**Abstract:** This paper combines the classical model of labeled transition systems with the epistemic model for reasoning about knowledge. The result is a unifying framework for modeling and analyzing multi-agent, knowledge-based, dynamic systems. On the modeling side, we propose a process algebraic, agent-oriented specification language that makes such a framework easy to use for practical purposes. On the verification side, we define a modal logic encompassing temporal and epistemic operators.

**arXiv ID:** 2407.17537
</details>

<details>
<summary><strong>Reinforcement Learning: An Overview</strong> - Kevin Murphy - [[pdf]](https://arxiv.org/pdf/2412.05265)</summary>

**Abstract:** This manuscript gives a big-picture, up-to-date overview of the field of (deep) reinforcement learning and sequential decision making, covering value-based methods, policy-based methods, model-based methods, multi-agent RL, LLMs and RL, and various other topics (e.g., offline RL, hierarchical RL, intrinsic reward). It also includes some code snippets for training LLMs with RL.

**arXiv ID:** 2412.05265
</details>

<details>
<summary><strong>MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs</strong> - Huining Yuan, Zelai Xu, Zheyue Tan, Xiangmin Yi, Mo Guang, Kaiwen Long, Haojia Hui, Boxun Li, Xinlei Chen, Bo Zhao, Xiao-Ping Zhang, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2510.15414)</summary>

**Abstract:** Developing large language models (LLMs) to cooperate and compete effectively within multi-agent systems (MASs) is a critical step towards more advanced intelligence. While reinforcement learning (RL) has proven effective for enhancing reasoning in single-agent tasks, its extension to multi-turn, multi-agent scenarios remains underexplored due to the challenges of long-horizon credit assignment and agent-specific advantage estimation. To address these challenges, we introduce MARSHAL, an end-to-end RL framework that incentivizes Multi-Agent Reasoning through Self-play witH strAtegic LLMs in both cooperative and competitive games. MARSHAL features a turn-level advantage estimator that aligns learning signals with each interaction for credit assignment, and an agent-specific advantage normalization to stabilize multi-agent training. By learning with self-play across cooperative and competitive games, MARSHAL agent trained from Qwen3-4B develops strong strategic abilities that generalize to held-out games with up to 28.7% performance improvements. More importantly, the capability acquired through self-play generalizes beyond games, yielding consistent performance gains of MASs in reasoning benchmarks. When integrated into leading MASs, our MARSHAL agent achieves significant performance gains of up to 10.0% on AIME, 6.6% on GPQA-Diamond, and 3.5% on average across all benchmarks. These results establish end-to-end RL training with self-play in strategic games as a powerful approach for developing generalizable multi-agent reasoning capabilities in LLMs.

**arXiv ID:** 2510.15414
</details>

<details>
<summary><strong>DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems</strong> - Junwei Yu, Yepeng Ding, Hiroyuki Sato - [[pdf]](https://arxiv.org/pdf/2503.07675)</summary>

**Abstract:** The emergence of Large Language Models (LLMs) in Multi-Agent Systems (MAS) has opened new possibilities for artificial intelligence, yet current implementations face significant challenges in resource management, task coordination, and system efficiency. While existing frameworks demonstrate the potential of LLM-based agents in collaborative problem-solving, they often lack sophisticated mechanisms for parallel execution and dynamic task management. This paper introduces DynTaskMAS, a novel framework that orchestrates asynchronous and parallel operations in LLM-based MAS through dynamic task graphs. The framework features four key innovations: (1) a Dynamic Task Graph Generator that intelligently decomposes complex tasks while maintaining logical dependencies, (2) an Asynchronous Parallel Execution Engine that optimizes resource utilization through efficient task scheduling, (3) a Semantic-Aware Context Management System that enables efficient information sharing among agents, and (4) an Adaptive Workflow Manager that dynamically optimizes system performance. Experimental evaluations demonstrate that DynTaskMAS achieves significant improvements over traditional approaches: a 21-33% reduction in execution time across task complexities (with higher gains for more complex tasks), a 35.4% improvement in resource utilization (from 65% to 88%), and near-linear throughput scaling up to 16 concurrent agents (3.47X improvement for 4X agents). Our framework establishes a foundation for building scalable, high-performance LLM-based multi-agent systems capable of handling complex, dynamic tasks efficiently.

**arXiv ID:** 2503.07675
</details>

<details>
<summary><strong>MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision</strong> - Zixuan Ke, Austin Xu, Yifei Ming, Xuan-Phi Nguyen, Ryan Chin, Caiming Xiong, Shafiq Joty - [[pdf]](https://arxiv.org/pdf/2505.14996)</summary>

**Abstract:** Multi-agent systems (MAS) leveraging the impressive capabilities of Large Language Models (LLMs) hold significant potential for tackling complex tasks. However, most current MAS depend on manually designed agent roles and communication protocols. These manual designs often fail to align with the underlying LLMs' strengths and struggle to adapt to novel tasks. Recent automatic MAS approaches attempt to mitigate these limitations but typically necessitate a validation set for tuning and yield static MAS designs lacking adaptability during inference, while also removing the flexibility to reduce to simpler systems. We introduce MAS-ZERO, the first self-evolved, inference-time framework for automatic MAS design. MAS-ZERO employs meta-level design to iteratively design, critique, and refine MAS configurations tailored to each problem instance, without requiring a validation set. Critically, it enables dynamic problem decomposition and agent composition through meta-feedback on solvability and completeness, and reduction to simpler systems when appropriate. Experiments across reasoning (math and graduate-level QA), coding, and agentic (search-based) benchmarks, using both closed-source and open-source LLM backbones of varying sizes, demonstrate that MAS-ZERO outperforms strong manual and automatic MAS baselines. It achieves substantial average accuracy improvements of up to 16.69% on reasoning, 16.66% on coding, and 5.45% on agentic tasks, while maintaining cost efficiency.

**arXiv ID:** 2505.14996
</details>

<details>
<summary><strong>AdvisingWise: Supporting Academic Advising in Higher Education Settings Through a Human-in-the-Loop Multi-Agent Framework</strong> - Wendan Jiang, Shiyuan Wang, Hiba Eltigani, Rukhshan Haroon, Abdullah Bin Faisal, Fahad Dogar - [[pdf]](https://arxiv.org/pdf/2511.05706)</summary>

**Abstract:** Academic advising is critical to student success in higher education, yet high student-to-advisor ratios limit advisors' capacity to provide timely support, particularly during peak periods. Recent advances in Large Language Models (LLMs) present opportunities to enhance the advising process. We present AdvisingWise, a multi-agent system that automates time-consuming tasks, such as information retrieval and response drafting, while preserving human oversight. AdvisingWise leverages authoritative institutional resources and adaptively prompts students about their academic backgrounds to generate reliable, personalized responses. All system responses undergo human advisor validation before delivery to students. We evaluate AdvisingWise through a mixed-methods approach: (1) expert evaluation on responses of 20 sample queries, (2) LLM-as-a-judge evaluation of the information retrieval strategy, and (3) a user study with 8 academic advisors to assess the system's practical utility. Our evaluation shows that AdvisingWise produces accurate, personalized responses. Advisors reported increasingly positive perceptions after using AdvisingWise, as their initial concerns about reliability and personalization diminished. We conclude by discussing the implications of human-AI synergy on the practice of academic advising.

**arXiv ID:** 2511.05706
</details>

<details>
<summary><strong>Decentralized Multi-Agent System with Trust-Aware Communication</strong> - Yepeng Ding, Ahmed Twabi, Junwei Yu, Lingfeng Zhang, Tohru Kondo, Hiroyuki Sato - [[pdf]](https://arxiv.org/pdf/2512.02410)</summary>

**Abstract:** The emergence of Large Language Models (LLMs) is rapidly accelerating the development of autonomous multi-agent systems (MAS), paving the way for the Internet of Agents. However, traditional centralized MAS architectures present significant challenges, including single points of failure, vulnerability to censorship, inherent scalability limitations, and critical trust issues. We propose a novel Decentralized Multi-Agent System (DMAS) architecture designed to overcome these fundamental problems by enabling trust-aware, scalable, and censorship-resistant interactions among autonomous agents. Our DMAS features a decentralized agent runtime underpinned by a blockchain-based architecture. We formalize a trust-aware communication protocol that leverages cryptographic primitives and on-chain operations to provide security properties: verifiable interaction cycles, communication integrity, authenticity, non-repudiation, and conditional confidentiality, which we further substantiate through a comprehensive security analysis. Our performance analysis validates the DMAS as a scalable and efficient solution for building trustworthy multi-agent systems.

**arXiv ID:** 2512.02410
</details>

<details>
<summary><strong>Robust Geospatial Coordination of Multi-Agent Communications Networks Under Attrition</strong> - Jonathan S. Kent, Eliana Stefani, Brian K. Plancher - [[pdf]](https://arxiv.org/pdf/2512.02079)</summary>

**Abstract:** Fast, efficient, robust communication during wildfire and other emergency responses is critical. One way to achieve this is by coordinating swarms of autonomous aerial vehicles carrying communications equipment to form an ad-hoc network connecting emergency response personnel to both each other and central command. However, operating in such extreme environments may lead to individual networking agents being damaged or rendered inoperable, which could bring down the network and interrupt communications.
To overcome this challenge and enable multi-agent UAV networking in difficult environments, this paper introduces and formalizes the problem of Robust Task Networking Under Attrition (RTNUA), which extends connectivity maintenance in multi-robot systems to explicitly address proactive redundancy and attrition recovery. We introduce Physics-Informed Robust Employment of Multi-Agent Networks ($\Phi$IREMAN), a topological algorithm leveraging physics-inspired potential fields to solve this problem. Through simulation across 25 problem configurations, $\Phi$IREMAN consistently outperforms the DCCRS baseline, and on large-scale problems with up to 100 tasks and 500 drones, maintains $>99.9\%$ task uptime despite substantial attrition, demonstrating both effectiveness and scalability.

**arXiv ID:** 2512.02079
</details>

<details>
<summary><strong>R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning</strong> - Harsh Goel, Mohammad Omama, Behdad Chalaki, Vaishnav Tadiparthi, Ehsan Moradi Pari, Sandeep Chinchali - [[pdf]](https://arxiv.org/pdf/2505.24265)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) has achieved significant progress in large-scale traffic control, autonomous vehicles, and robotics. Drawing inspiration from biological systems where roles naturally emerge to enable coordination, role-based MARL methods have been proposed to enhance cooperation learning for complex tasks. However, existing methods exclusively derive roles from an agent's past experience during training, neglecting their influence on its future trajectories. This paper introduces a key insight: an agent's role should shape its future behavior to enable effective coordination. Hence, we propose Role Discovery and Diversity through Dynamics Models (R3DM), a novel role-based MARL framework that learns emergent roles by maximizing the mutual information between agents' roles, observed trajectories, and expected future behaviors. R3DM optimizes the proposed objective through contrastive learning on past trajectories to first derive intermediate roles that shape intrinsic rewards to promote diversity in future behaviors across different roles through a learned dynamics model. Benchmarking on SMAC and SMACv2 environments demonstrates that R3DM outperforms state-of-the-art MARL approaches, improving multi-agent coordination to increase win rates by up to 20%. The code is available at this https URL.

**arXiv ID:** 2505.24265
</details>

<details>
<summary><strong>iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference</strong> - Wei Fan, JinYi Yoon, Bo Ji - [[pdf]](https://arxiv.org/pdf/2511.11306)</summary>

**Abstract:** Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).

**arXiv ID:** 2511.11306
</details>

<details>
<summary><strong>Multi-Agent Code Verification via Information Theory</strong> - Shreshth Rajan - [[pdf]](https://arxiv.org/pdf/2511.16708)</summary>

**Abstract:** LLMs generate buggy code: 29.6% of SWE-bench solved patches fail, 62% of BaxBench solutions have vulnerabilities, and existing
tools only catch 65% of bugs with 35% false positives. We built CodeX-Verify, a multi-agent system that uses four specialized
agents to detect different types of bugs. We prove mathematically that combining agents with different detection patterns
finds more bugs than any single agent when the agents look for different problems, using submodularity of mutual information
under conditional independence. Measuring agent correlation of rho = 0.05 to 0.25 confirms they detect different bugs. Testing
on 99 code samples with verified labels shows our system catches 76.1% of bugs, matching the best existing method (Meta
Prompt Testing: 75%) while running faster and without test execution. We tested all 15 agent combinations and found that using
multiple agents improves accuracy by 39.7 percentage points (from 32.8% to 72.4%) compared to single agents, with diminishing
returns of +14.9pp, +13.5pp, and +11.2pp for agents 2, 3, and 4, validating our theoretical model. The best two-agent
combination (Correctness + Performance) reaches 79.3% accuracy. Testing on 300 real patches from Claude Sonnet 4.5 runs in
under 200ms per sample, making this practical for production use.

**arXiv ID:** 2511.16708
</details>

<details>
<summary><strong>Truthful and Trustworthy IoT AI Agents via Immediate-Penalty Enforcement under Approximate VCG Mechanisms</strong> - Xun Shao, Ryuuto Shimizu, Zhi Liu, Kaoru Ota, Mianxiong Dong - [[pdf]](https://arxiv.org/pdf/2512.00513)</summary>

**Abstract:** The deployment of autonomous AI agents in Internet of Things (IoT) energy systems requires decision-making mechanisms that remain robust, efficient, and trustworthy under real-time constraints and imperfect monitoring. While reinforcement learning enables adaptive prosumer behaviors, ensuring economic consistency and preventing strategic manipulation remain open challenges, particularly when sensing noise or partial observability reduces the operator's ability to verify actions. This paper introduces a trust-enforcement framework for IoT energy trading that combines an approximate Vickrey-Clarke-Groves (VCG) double auction with an immediate one-shot penalty. Unlike reputation- or history-based approaches, the proposed mechanism restores truthful reporting within a single round, even when allocation accuracy is approximate and monitoring is noisy. We theoretically characterize the incentive gap induced by approximation and derive a penalty threshold that guarantees truthful bidding under bounded sensing errors. To evaluate learning-enabled prosumers, we embed the mechanism into a multi-agent reinforcement learning environment reflecting stochastic generation, dynamic loads, and heterogeneous trading opportunities. Experiments show that improved allocation accuracy reduces deviation incentives, the required penalty matches analytical predictions, and learned bidding behaviors remain stable and interpretable despite imperfect monitoring. These results demonstrate that lightweight penalty designs can reliably align strategic IoT agents with socially efficient energy-trading outcomes.

**arXiv ID:** 2512.00513
</details>

<details>
<summary><strong>Dynamic Configuration of On-Street Parking Spaces using Multi Agent Reinforcement Learning</strong> - Oshada Jayasinghe, Farhana Choudhury, Egemen Tanin, Shanika Karunasekera - [[pdf]](https://arxiv.org/pdf/2512.02406)</summary>

**Abstract:** With increased travelling needs more than ever, traffic congestion has become a major concern in most urban areas. Allocating spaces for on-street parking, further hinders traffic flow, by limiting the effective road width available for driving. With the advancement of vehicle-to-infrastructure connectivity technologies, we explore how the impact of on-street parking on traffic congestion could be minimized, by dynamically configuring on-street parking spaces. Towards that end, we formulate dynamic on-street parking space configuration as an optimization problem, and we follow a data driven approach, considering the nature of our problem. Our proposed solution comprises a two-layer multi agent reinforcement learning based framework, which is inherently scalable to large road networks. The lane level agents are responsible for deciding the optimal parking space configuration for each lane, and we introduce a novel Deep Q-learning architecture which effectively utilizes long short term memory networks and graph attention networks to capture the spatio-temporal correlations evident in the given problem. The block level agents control the actions of the lane level agents and maintain a sufficient level of parking around the block. We conduct a set of comprehensive experiments using SUMO, on both synthetic data as well as real-world data from the city of Melbourne. Our experiments show that the proposed framework could reduce the average travel time loss of vehicles significantly, reaching upto 47%, with a negligible increase in the walking distance for parking.

**arXiv ID:** 2512.02406
</details>

<details>
<summary><strong>AID: Agent Intent from Diffusion for Multi-Agent Informative Path Planning</strong> - Jeric Lew, Yuhong Cao, Derek Ming Siang Tan, Guillaume Sartoretti - [[pdf]](https://arxiv.org/pdf/2512.02535)</summary>

**Abstract:** Information gathering in large-scale or time-critical scenarios (e.g., environmental monitoring, search and rescue) requires broad coverage within limited time budgets, motivating the use of multi-agent systems. These scenarios are commonly formulated as multi-agent informative path planning (MAIPP), where multiple agents must coordinate to maximize information gain while operating under budget constraints. A central challenge in MAIPP is ensuring effective coordination while the belief over the environment evolves with incoming measurements. Recent learning-based approaches address this by using distributions over future positions as "intent" to support coordination. However, these autoregressive intent predictors are computationally expensive and prone to compounding errors. Inspired by the effectiveness of diffusion models as expressive, long-horizon policies, we propose AID, a fully decentralized MAIPP framework that leverages diffusion models to generate long-term trajectories in a non-autoregressive manner. AID first performs behavior cloning on trajectories produced by existing MAIPP planners and then fine-tunes the policy using reinforcement learning via Diffusion Policy Policy Optimization (DPPO). This two-stage pipeline enables the policy to inherit expert behavior while learning improved coordination through online reward feedback. Experiments demonstrate that AID consistently improves upon the MAIPP planners it is trained from, achieving up to 4x faster execution and 17% increased information gain, while scaling effectively to larger numbers of agents. Our implementation is publicly available at this https URL.

**arXiv ID:** 2512.02535
</details>

<details>
<summary><strong>On the Convergence of Density-Based Predictive Control for Multi-Agent Non-Uniform Area Coverage</strong> - Sungjun Seo, Kooktae Lee - [[pdf]](https://arxiv.org/pdf/2512.02367)</summary>

**Abstract:** This paper presents Density-based Predictive Control (DPC), a novel multi-agent control strategy for efficient non-uniform area coverage, grounded in optimal transport theory. In large-scale scenarios such as search and rescue or environmental monitoring, traditional uniform coverage fails to account for varying regional priorities. DPC leverages a pre-constructed reference distribution to allocate agents' coverage efforts, spending more time in high-priority or densely sampled regions. We analyze convergence conditions using the Wasserstein distance, derive an analytic optimal control law for unconstrained cases, and propose a numerical method for constrained scenarios. Simulations on first-order dynamics and linearized quadrotor models demonstrate that DPC achieves trajectories closely matching the non-uniform reference distribution, outperforming existing coverage methods.

**arXiv ID:** 2512.02367
</details>

<details>
<summary><strong>A Visual Analytics System to Understand Behaviors of Multi Agents in Reinforcement Learning</strong> - Changhee Lee, Jeongmin Rhee, DongHwa Shin - [[pdf]](https://arxiv.org/pdf/2512.02442)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) is a branch of machine learning in which agents interact and learn optimal policies through trial and error, addressing complex scenarios where multiple agents interact and learn in the same environment at the same time. Analyzing and understanding these complex interactions is challenging, and existing analysis methods are limited in their ability to fully reflect and interpret this complexity. To address these challenges, we provide MARLViz, a visual analytics system for visualizing and analyzing the policies and interactions of agents in MARL environments. The system is designed to visually show the difference in behavior of agents under different environment settings and help users understand complex interaction patterns. In this study, we analyzed agents with similar behaviors and selected scenarios to understand the interactions of the agents, which made it easier to understand the strategies of agents in MARL.

**arXiv ID:** 2512.02442
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Self-Improving AI Agents through Self-Play</strong> - Przemyslaw Chojecki - [[pdf]](https://arxiv.org/pdf/2512.02731)</summary>

**Abstract:** We extend the moduli-theoretic framework of psychometric batteries to the domain of dynamical systems. While previous work established the AAI capability score as a static functional on the space of agent representations, this paper formalizes the agent as a flow $\nu_r$ parameterized by computational resource $r$, governed by a recursive Generator-Verifier-Updater (GVU) operator. We prove that this operator generates a vector field on the parameter manifold $\Theta$, and we identify the coefficient of self-improvement $\kappa$ as the Lie derivative of the capability functional along this flow.
The central contribution of this work is the derivation of the Variance Inequality, a spectral condition that is sufficient (under mild regularity) for the stability of self-improvement. We show that a sufficient condition for $\kappa > 0$ is that, up to curvature and step-size effects, the combined noise of generation and verification must be small enough.
We then apply this formalism to unify the recent literature on Language Self-Play (LSP), Self-Correction, and Synthetic Data bootstrapping. We demonstrate that architectures such as STaR, SPIN, Reflexion, GANs and AlphaZero are specific topological realizations of the GVU operator that satisfy the Variance Inequality through filtration, adversarial discrimination, or grounding in formal systems.

**arXiv ID:** 2512.02731
</details>

<details>
<summary><strong>Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading</strong> - Jifeng Li, Arnav Grover, Abraham Alpuerto, Yupeng Cao, Xiao-Yang Liu - [[pdf]](https://arxiv.org/pdf/2512.02227)</summary>

**Abstract:** The financial market is a mission-critical playground for AI agents due to its temporal dynamics and low signal-to-noise ratio. Building an effective algorithmic trading system may require a professional team to develop and test over the years. In this paper, we propose an orchestration framework for financial agents, which aims to democratize financial intelligence to the general public. We map each component of the traditional algorithmic trading system to agents, including planner, orchestrator, alpha agents, risk agents, portfolio agents, backtest agents, execution agents, audit agents, and memory agent. We present two in-house trading examples. For the stock trading task (hourly data from 04/2024 to 12/2024), our approach achieved a return of $20.42\%$, a Sharpe ratio of 2.63, and a maximum drawdown of $-3.59\%$, while the S&P 500 index yielded a return of $15.97\%$. For the BTC trading task (minute data from 27/07/2025 to 13/08/2025), our approach achieved a return of $8.39\%$, a Sharpe ratio of $0.38$, and a maximum drawdown of $-2.80\%$, whereas the BTC price increased by $3.80\%$. Our code is available on \href{this https URL}{GitHub}.

**arXiv ID:** 2512.02227
</details>

<details>
<summary><strong>Vehicle Dynamics Embedded World Models for Autonomous Driving</strong> - Huiqian Li, Wei Pan, Haodong Zhang, Jin Huang, Zhihua Zhong - [[pdf]](https://arxiv.org/pdf/2512.02417)</summary>

**Abstract:** World models have gained significant attention as a promising approach for autonomous driving. By emulating human-like perception and decision-making processes, these models can predict and adapt to dynamic environments. Existing methods typically map high-dimensional observations into compact latent spaces and learn optimal policies within these latent representations. However, prior work usually jointly learns ego-vehicle dynamics and environmental transition dynamics from the image input, leading to inefficiencies and a lack of robustness to variations in vehicle dynamics. To address these issues, we propose the Vehicle Dynamics embedded Dreamer (VDD) method, which decouples the modeling of ego-vehicle dynamics from environmental transition dynamics. This separation allows the world model to generalize effectively across vehicles with diverse parameters. Additionally, we introduce two strategies to further enhance the robustness of the learned policy: Policy Adjustment during Deployment (PAD) and Policy Augmentation during Training (PAT). Comprehensive experiments in simulated environments demonstrate that the proposed model significantly improves both driving performance and robustness to variations in vehicle dynamics, outperforming existing approaches.

**arXiv ID:** 2512.02417
</details>

<details>
<summary><strong>ADORE: Autonomous Domain-Oriented Relevance Engine for E-commerce</strong> - Zheng Fang, Donghao Xie, Ming Pang, Chunyuan Yuan, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo - [[pdf]](https://arxiv.org/pdf/2512.02555)</summary>

**Abstract:** Relevance modeling in e-commerce search remains challenged by semantic gaps in term-matching methods (e.g., BM25) and neural models' reliance on the scarcity of domain-specific hard samples. We propose ADORE, a self-sustaining framework that synergizes three innovations: (1) A Rule-aware Relevance Discrimination module, where a Chain-of-Thought LLM generates intent-aligned training data, refined via Kahneman-Tversky Optimization (KTO) to align with user behavior; (2) An Error-type-aware Data Synthesis module that auto-generates adversarial examples to harden robustness; and (3) A Key-attribute-enhanced Knowledge Distillation module that injects domain-specific attribute hierarchies into a deployable student model. ADORE automates annotation, adversarial generation, and distillation, overcoming data scarcity while enhancing reasoning. Large-scale experiments and online A/B testing verify the effectiveness of ADORE. The framework establishes a new paradigm for resource-efficient, cognitively aligned relevance modeling in industrial applications.

**arXiv ID:** 2512.02555
</details>

<details>
<summary><strong>Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning</strong> - Anis Koubaa, Khaled Gabr - [[pdf]](https://arxiv.org/pdf/2509.13352)</summary>

**Abstract:** Unmanned Aerial Vehicles (UAVs) are increasingly used in defense, surveillance, and disaster response, yet most systems still operate at SAE Level 2 to 3 autonomy. Their dependence on rule-based control and narrow AI limits adaptability in dynamic and uncertain missions. Current UAV architectures lack context-aware reasoning, autonomous decision-making, and integration with external systems. Importantly, none make use of Large Language Model (LLM) agents with tool-calling for real-time knowledge access.
This paper introduces the Agentic UAVs framework, a five-layer architecture consisting of Perception, Reasoning, Action, Integration, and Learning. The framework enhances UAV autonomy through LLM-driven reasoning, database querying, and interaction with third-party systems.
A prototype built with ROS 2 and Gazebo combines YOLOv11 for object detection with GPT-4 for reasoning and a locally deployed Gemma 3 model. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 compared to 0.72), improved person detection rates (91% compared to 75%), and a major increase in correct action recommendations (92% compared to 4.5%). These results show that modest computational overhead can enable significantly higher levels of autonomy and system-level integration.

**arXiv ID:** 2509.13352
</details>

<details>
<summary><strong>LeechHijack: Covert Computational Resource Exploitation in Intelligent Agent Systems</strong> - Yuanhe Zhang, Weiliu Wang, Zhenhong Zhou, Kun Wang, Jie Zhang, Li Sun, Yang Liu, Sen Su - [[pdf]](https://arxiv.org/pdf/2512.02321)</summary>

**Abstract:** Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in reasoning, planning, and tool usage. The recently proposed Model Context Protocol (MCP) has emerged as a unifying framework for integrating external tools into agent systems, enabling a thriving open ecosystem of community-built functionalities. However, the openness and composability that make MCP appealing also introduce a critical yet overlooked security assumption -- implicit trust in third-party tool providers. In this work, we identify and formalize a new class of attacks that exploit this trust boundary without violating explicit permissions. We term this new attack vector implicit toxicity, where malicious behaviors occur entirely within the allowed privilege scope. We propose LeechHijack, a Latent Embedded Exploit for Computation Hijacking, in which an adversarial MCP tool covertly expropriates the agent's computational resources for unauthorized workloads. LeechHijack operates through a two-stage mechanism: an implantation stage that embeds a benign-looking backdoor in a tool, and an exploitation stage where the backdoor activates upon predefined triggers to establish a command-and-control channel. Through this channel, the attacker injects additional tasks that the agent executes as if they were part of its normal workflow, effectively parasitizing the user's compute budget. We implement LeechHijack across four major LLM families. Experiments show that LeechHijack achieves an average success rate of 77.25%, with a resource overhead of 18.62% compared to the baseline. This study highlights the urgent need for computational provenance and resource attestation mechanisms to safeguard the emerging MCP ecosystem.

**arXiv ID:** 2512.02321
</details>

<details>
<summary><strong>Learning-based 3D Reconstruction in Autonomous Driving: A Comprehensive Survey</strong> - Liewen Liao, Weihao Yan, Wang Xu, Ming Yang, Songan Zhang, H. Eric Tseng - [[pdf]](https://arxiv.org/pdf/2503.14537)</summary>

**Abstract:** Learning-based 3D reconstruction has emerged as a transformative technique in autonomous driving, enabling precise modeling of environments through advanced neural representations. It has inspired pioneering solutions for vital tasks in autonomous driving, such as dense mapping and closed-loop simulation, as well as comprehensive scene feature for driving scene understanding and reasoning. Given the rapid growth in related research, this survey provides a comprehensive review of both technical evolutions and practical applications in autonomous driving. We begin with an introduction to the preliminaries of learning-based 3D reconstruction to provide a solid technical background foundation, then progress to a rigorous, multi-dimensional examination of cutting-edge methodologies, systematically organized according to the distinctive technical requirements and fundamental challenges of autonomous driving. Through analyzing and summarizing development trends and cutting-edge research, we identify existing technical challenges, along with insufficient disclosure of on-board validation and safety verification details in the current literature, and ultimately suggest potential directions to guide future studies.

**arXiv ID:** 2503.14537
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (27 papers)</h2></summary>

<details>
<summary><strong>Radiologist Copilot: An Agentic Assistant with Orchestrated Tools for Radiology Reporting with Quality Control</strong> - Yongrui Yu, Zhongzhen Huang, Linjie Mu, Shaoting Zhang, Xiaofan Zhang - [[pdf]](https://arxiv.org/pdf/2512.02814)</summary>

**Abstract:** Radiology reporting is an essential yet time-consuming and error-prone task for radiologists in clinical examinations, especially for volumetric medical images. Rigorous quality control is also critical but tedious, ensuring that the final report meets clinical standards. Existing automated approaches, including radiology report generation methods and medical vision-language models, focus mainly on the report generation phase and neglect the crucial quality control procedure, limiting their capability to provide comprehensive support to radiologists. We propose Radiologist Copilot, an agentic AI assistant equipped with orchestrated tools designed for automated radiology reporting with quality control. Leveraging large language models as the reasoning backbone, the agentic system autonomously selects tools, plans, and executes actions, emulating the behavior of radiologists throughout the holistic radiology reporting process. The orchestrated tools include region localization, think with image paradigm directed region analysis planning, strategic template selection for report generation, quality assessment and feedback-driven adaptive refinement for quality control. Therefore, Radiologist Copilot facilitates accurate, complete, and efficient radiology reporting, assisting radiologists and improving clinical efficiency. Experimental results demonstrate that Radiologist Copilot significantly surpasses other state-of-the-art methods in radiology reporting. The source code will be released upon acceptance.

**arXiv ID:** 2512.02814
</details>

<details>
<summary><strong>Improved Training Mechanism for Reinforcement Learning via Online Model Selection</strong> - Aida Afshar, Aldo Pacchiano - [[pdf]](https://arxiv.org/pdf/2512.02214)</summary>

**Abstract:** We study the problem of online model selection in reinforcement learning, where the selector has access to a class of reinforcement learning agents and learns to adaptively select the agent with the right configuration. Our goal is to establish the improved efficiency and performance gains achieved by integrating online model selection methods into reinforcement learning training procedures. We examine the theoretical characterizations that are effective for identifying the right configuration in practice, and address three practical criteria from a theoretical perspective: 1) Efficient resource allocation, 2) Adaptation under non-stationary dynamics, and 3) Training stability across different seeds. Our theoretical results are accompanied by empirical evidence from various model selection tasks in reinforcement learning, including neural architecture selection, step-size selection, and self model selection.

**arXiv ID:** 2512.02214
</details>

<details>
<summary><strong>FOVA: Offline Federated Reinforcement Learning with Mixed-Quality Data</strong> - Nan Qiao, Sheng Yue, Ju Ren, Yaoxue Zhang - [[pdf]](https://arxiv.org/pdf/2512.02350)</summary>

**Abstract:** Offline Federated Reinforcement Learning (FRL), a marriage of federated learning and offline reinforcement learning, has attracted increasing interest recently. Albeit with some advancement, we find that the performance of most existing offline FRL methods drops dramatically when provided with mixed-quality data, that is, the logging behaviors (offline data) are collected by policies with varying qualities across clients. To overcome this limitation, this paper introduces a new vote-based offline FRL framework, named FOVA. It exploits a \emph{vote mechanism} to identify high-return actions during local policy evaluation, alleviating the negative effect of low-quality behaviors from diverse local learning policies. Besides, building on advantage-weighted regression (AWR), we construct consistent local and global training objectives, significantly enhancing the efficiency and stability of FOVA. Further, we conduct an extensive theoretical analysis and rigorously show that the policy learned by FOVA enjoys strict policy improvement over the behavioral policy. Extensive experiments corroborate the significant performance gains of our proposed algorithm over existing baselines on widely used benchmarks.

**arXiv ID:** 2512.02350
</details>

<details>
<summary><strong>CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning</strong> - Songqiao Su, Xiaofei Sun, Xiaoya Li, Albert Wang, Jiwei Li, Chris Shum - [[pdf]](https://arxiv.org/pdf/2512.02551)</summary>

**Abstract:** In this paper, we propose CUDA-L2, a system that combines large language models (LLMs) and reinforcement learning (RL) to automatically optimize Half-precision General Matrix Multiply (HGEMM) CUDA kernels. Using CUDA execution speed as the RL reward, CUDA-L2 automatically optimizes HGEMM kernels across 1,000 configurations. CUDA-L2 systematically outperforms major matmul baselines to date, from the widely-used {\it this http URL} to state-of-the-art Nvidia's closed-source libraries, i.e., {\it cuBLAS}, {\it cuBLASLt}. In offline mode, where kernels are executed consecutively without time intervals, CUDA-L2 yields +22.0\% over {\it this http URL} on average; +19.2\% over {\it cuBLAS} using the optimal layout configuration (normal-normal NN and transposed-normal TN); +16.8\% over {\it cuBLASLt-heuristic}, which queries {\it cuBLASLt} library and selects the algorithm based on the heuristic's suggestion; and +11.4\% over the most competitive {\it cuBLASLt-AutoTuning} model, which selects the fastest algorithm from up to 100 candidates from {\it cuBLASLt}'s suggestions. In server mode, where kernels are executed at random intervals simulating real-time inference, the speedups further increase to +28.7\%, +26.0\%, +22.4\%, and +15.9\% for {\it this http URL}, {\it cuBLAS}, {\it cuBLASLt-heuristic}, and {\it cuBLASLt-AutoTuning} respectively. CUDA-L2 shows that even the most performance-critical, heavily-optimized kernels like HGEMM can be improved through LLM-guided RL automation by systematically exploring configuration spaces at scales impractical for humans. Project and code can be found at this http URL

**arXiv ID:** 2512.02551
</details>

<details>
<summary><strong>EZYer: A simulacrum of high school with generative agent</strong> - Jinming Yang, Zimu Ji, Weiqi Luo, Gaoxi Wang, Bin Ma, Yueling Deng - [[pdf]](https://arxiv.org/pdf/2512.02561)</summary>

**Abstract:** With the rapid development of the online education and large language model, the existing educational tools still suffer from incomplete service, insufficient performance and weak interactivity in terms of courseware generation, interactive notes and quality assurance of content. In particular, the proposed generative agent EZYer : 1) Teacher Module: Integrating the Text Corpus retrieval and in-depth generation technologies, it automatically generates structured teaching materials and LaTeX Beamer courseware in line with the high school mathematics syllabus and supports user-defined image insertion. 2) Student Module: Throughout the collaborative interaction of the four roles of Teacher, Assistant, Top Student and Struggling Student, Note Taker summarizes and generates academic notes to enhance the depth and interest of learning. 3) Controller: set up keyword filtering system, content scoring system, role co-validation system, and dynamic content correction system. This ensure academic strictness and pedagogical propriety of EZYer inputs and outputs. In order to evaluate EZYer, this paper designs five-dimensional evaluation indexes of content accuracy, knowledge coverage, usability, formatting correctness and visual design and appeal, and scores 100 Beamer and Notes generated by EZYer by five large language models, separately, and the results show that the quality of EZYer-generated content is excellent and has a good application prospect.

**arXiv ID:** 2512.02561
</details>

<details>
<summary><strong>ReVSeg: Incentivizing the Reasoning Chain for Video Segmentation with Reinforcement Learning</strong> - Yifan Li, Yingda Yin, Lingting Zhu, Weikai Chen, Shengju Qian, Xin Wang, Yanwei Fu - [[pdf]](https://arxiv.org/pdf/2512.02835)</summary>

**Abstract:** Reasoning-centric video object segmentation is an inherently complex task: the query often refers to dynamics, causality, and temporal interactions, rather than static appearances. Yet existing solutions generally collapse these factors into simplified reasoning with latent embeddings, rendering the reasoning chain opaque and essentially intractable. We therefore adopt an explicit decomposition perspective and introduce ReVSeg, which executes reasoning as sequential decisions in the native interface of pretrained vision language models (VLMs). Rather than folding all reasoning into a single-step prediction, ReVSeg executes three explicit operations -- semantics interpretation, temporal evidence selection, and spatial grounding -- aligning pretrained capabilities. We further employ reinforcement learning to optimize the multi-step reasoning chain, enabling the model to self-refine its decision quality from outcome-driven signals. Experimental results demonstrate that ReVSeg attains state-of-the-art performances on standard video object segmentation benchmarks and yields interpretable reasoning trajectories. Project page is available at this https URL .

**arXiv ID:** 2512.02835
</details>

<details>
<summary><strong>From Memories to Maps: Mechanisms of In-Context Reinforcement Learning in Transformers</strong> - Ching Fang, Kanaka Rajan - [[pdf]](https://arxiv.org/pdf/2506.19686)</summary>

**Abstract:** Humans and animals show remarkable learning efficiency, adapting to new environments with minimal experience. This capability is not well captured by standard reinforcement learning algorithms that rely on incremental value updates. Rapid adaptation likely depends on episodic memory-- the ability to retrieve specific past experiences to guide decisions in novel contexts. Transformers provide a useful setting for studying these questions because of their ability to learn rapidly in-context and because their key-value architecture resembles episodic memory systems in the brain. We train a transformer to in-context reinforcement learn in a distribution of planning tasks inspired by rodent behavior. We then characterize the learning algorithms that emerge in the model. We first find that representation learning is supported by in-context structure learning and cross-context alignment, where representations are aligned across environments with different sensory stimuli. We next demonstrate that the reinforcement learning strategies developed by the model are not interpretable as standard model-free or model-based planning. Instead, we show that in-context reinforcement learning is supported by caching intermediate computations within the model's memory tokens, which are then accessed at decision time. Overall, we find that memory may serve as a computational resource, storing both raw experience and cached computations to support flexible behavior. Furthermore, the representations developed in the model resemble computations associated with the hippocampal-entorhinal system in the brain, suggesting that our findings may be relevant for natural cognition. Taken together, our work offers a mechanistic hypothesis for the rapid adaptation that underlies in-context learning in artificial and natural settings.

**arXiv ID:** 2506.19686
</details>

<details>
<summary><strong>From Atomic to Composite: Reinforcement Learning Enables Generalization in Complementary Reasoning</strong> - Sitao Cheng, Xunjian Yin, Ruiwen Zhou, Yuxuan Li, Xinyi Wang, Liangming Pan, William Yang Wang, Victor Zhong - [[pdf]](https://arxiv.org/pdf/2512.01970)</summary>

**Abstract:** The mechanism by which RL contributes to reasoning capabilities-whether it incentivizes the synthesis of new skills or merely amplifies existing behaviors-remains a subject of intense debate. In this work, we investigate this question through the lens of Complementary Reasoning, a complex task that requires integrating internal parametric knowledge with external contextual information. Using a controlled synthetic dataset of human biographies, we strictly decouple this ability into two atomic skills: Parametric Reasoning (relying on internal knowledge) and Contextual Reasoning (depending on external information). To rigorously assess capability boundaries, we evaluate generalization across three distinct levels of difficulty: I.I.D., Composition, and Zero-shot settings. We find that while SFT is sufficient for in-distribution performance, it struggles with O.O.D. generalization, particularly in Zero-shot settings where relational combinations are novel. Crucially, we identify the SFT Generalization Paradox: Models supervised solely on the composite task achieve near-perfect in-distribution accuracy but collapse on out-of-distribution generalization, indicating their reliance on rote memorization of path shortcuts. In contrast, we find that RL acts as a reasoning synthesizer rather than a probability amplifier. However, we uncover a strict atomic prerequisite: RL can only synthesize these complex strategies if the base model has first mastered the independent atomic skills (Parametric and Contextual) via SFT. These findings challenge the view of RL as a mere amplifier, suggesting that given sufficient atomic foundations, RL can actively synthesize complex reasoning strategies from learned primitives without explicit supervision on such complex strategies. This indicates that decoupled atomic training followed by RL offers a scalable path to generalization for complex reasoning tasks.

**arXiv ID:** 2512.01970
</details>

<details>
<summary><strong>medDreamer: Model-Based Reinforcement Learning with Latent Imagination on Complex EHRs for Clinical Decision Support</strong> - Qianyi Xu, Gousia Habib, Feng Wu, Dilruk Perera, Mengling Feng - [[pdf]](https://arxiv.org/pdf/2505.19785)</summary>

**Abstract:** Timely and personalized treatment decisions are essential across a wide range of healthcare settings where patient responses can vary significantly and evolve over time. Clinical data used to support these treatment decisions are often irregularly sampled, where missing data frequencies may implicitly convey information about the patient's condition. Existing Reinforcement Learning (RL) based clinical decision support systems often ignore the missing patterns and distort them with coarse discretization and simple imputation. They are also predominantly model-free and largely depend on retrospective data, which could lead to insufficient exploration and bias by historical behaviors. To address these limitations, we propose medDreamer, a novel model-based reinforcement learning framework for personalized treatment recommendation. medDreamer contains a world model with an Adaptive Feature Integration module that simulates latent patient states from irregular data and a two-phase policy trained on a hybrid of real and imagined trajectories. This enables learning optimal policies that go beyond the sub-optimality of historical clinical decisions, while remaining close to real clinical data. We evaluate medDreamer on both sepsis and mechanical ventilation treatment tasks using two large-scale Electronic Health Records (EHRs) datasets. Comprehensive evaluations show that medDreamer significantly outperforms model-free and model-based baselines in both clinical outcomes and off-policy metrics.

**arXiv ID:** 2505.19785
</details>

<details>
<summary><strong>Adversarial Diffusion for Robust Reinforcement Learning</strong> - Daniele Foffano, Alessio Russo, Alexandre Proutiere - [[pdf]](https://arxiv.org/pdf/2509.23846)</summary>

**Abstract:** Robustness to modeling errors and uncertainties remains a central challenge in reinforcement learning (RL). In this work, we address this challenge by leveraging diffusion models to train robust RL policies. Diffusion models have recently gained popularity in model-based RL due to their ability to generate full trajectories "all at once", mitigating the compounding errors typical of step-by-step transition models. Moreover, they can be conditioned to sample from specific distributions, making them highly flexible. We leverage conditional sampling to learn policies that are robust to uncertainty in environment dynamics. Building on the established connection between Conditional Value at Risk (CVaR) optimization and robust RL, we introduce Adversarial Diffusion for Robust Reinforcement Learning (AD-RRL). AD-RRL guides the diffusion process to generate worst-case trajectories during training, effectively optimizing the CVaR of the cumulative return. Empirical results across standard benchmarks show that AD-RRL achieves superior robustness and performance compared to existing robust RL methods.

**arXiv ID:** 2509.23846
</details>

<details>
<summary><strong>Rewarding the Journey, Not Just the Destination: A Composite Path and Answer Self-Scoring Reward Mechanism for Test-Time Reinforcement Learning</strong> - Chenwei Tang, Jingyu Xing, Lin Long, Xinyu Liu, Deng Xiong, Wei Ju, Shudong Huang, Jiancheng Lv, Ziyue Qiao - [[pdf]](https://arxiv.org/pdf/2510.17923)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a powerful paradigm for advancing Large Language Models (LLMs), achieving remarkable performance in complex reasoning domains such as mathematics and code generation. However, current RL methods face a fundamental scalability bottleneck due to their heavy reliance on human-curated preference data or labeled datasets for reward modeling. To overcome this limitation, we explore RL on unlabeled data where models learn autonomously from continuous experience streams. The core challenge in this setting lies in reliable reward estimation without ground-truth supervision. Existing approaches like Test-Time RL address this through self-consistent consensus, but risk reinforcing incorrect pseudo-labels derived from majority voting. We introduce COMPASS (Composite Path and Answer Self-Scoring), a novel test-time reward mechanism that operates without external supervision. COMPASS integrates two complementary components: the Dual-Calibration Answer Reward (DCAR), which stabilizes training by establishing trustworthy pseudo-labels through confidence and credibility calibration, and the Decisive Path Reward (DPR), which directly optimizes the reasoning process quality beyond mere outcome supervision. By jointly reinforcing trustworthy consensus answers and highly decisive reasoning chains, the COMPASS systematically enhances the model's analytical capabilities. Extensive experiments show that COMPASS achieves significant and consistent performance gains across diverse reasoning tasks and model architectures, advancing a more scalable direction for LLMs to learn from continuous experience.

**arXiv ID:** 2510.17923
</details>

<details>
<summary><strong>From Code Foundation Models to Agents and Applications: A Practical Guide to Code Intelligence</strong> - Jian Yang, Xianglong Liu, Weifeng Lv, Ken Deng, Shawn Guo, Lin Jing, Yizhi Li, Shark Liu, Xianzhen Luo, Yuyu Luo, Changzai Pan, Ensheng Shi, Yingshui Tan, Renshuai Tao, Jiajun Wu, Xianjie Wu, Zhenhe Wu, Daoguang Zan, Chenchen Zhang, Wei Zhang, He Zhu, Terry Yue Zhuo, Kerui Cao, Xianfu Cheng, Jun Dong, Shengjie Fang, Zhiwei Fei, Xiangyuan Guan, Qipeng Guo, Zhiguang Han, Joseph James, Tianqi Luo, Renyuan Li, Yuhang Li, Yiming Liang, Congnan Liu, Jiaheng Liu, Qian Liu, Ruitong Liu, Tyler Loakman, Xiangxin Meng, Chuang Peng, Tianhao Peng, Jiajun Shi, Mingjie Tang, Boyang Wang, Haowen Wang, Yunli Wang, Fanglin Xu, Zihan Xu, Fei Yuan, Ge Zhang, Jiayi Zhang, Xinhao Zhang, Wangchunshu Zhou, Hualei Zhu, King Zhu, Bryan Dai, Aishan Liu, Zhoujun Li, Chenghua Lin, Tianyu Liu, Chao Peng, Kai Shen, Libo Qin, Shuangyong Song, Zizheng Zhan, Jiajun Zhang, Jie Zhang, Zhaoxiang Zhang, Bo Zheng - [[pdf]](https://arxiv.org/pdf/2511.18538)</summary>

**Abstract:** Large language models (LLMs) have fundamentally transformed automated software development by enabling direct translation of natural language descriptions into functional code, driving commercial adoption through tools like Github Copilot (Microsoft), Cursor (Anysphere), Trae (ByteDance), and Claude Code (Anthropic). While the field has evolved dramatically from rule-based systems to Transformer-based architectures, achieving performance improvements from single-digit to over 95\% success rates on benchmarks like HumanEval. In this work, we provide a comprehensive synthesis and practical guide (a series of analytic and probing experiments) about code LLMs, systematically examining the complete model life cycle from data curation to post-training through advanced prompting paradigms, code pre-training, supervised fine-tuning, reinforcement learning, and autonomous coding agents. We analyze the code capability of the general LLMs (GPT-4, Claude, LLaMA) and code-specialized LLMs (StarCoder, Code LLaMA, DeepSeek-Coder, and QwenCoder), critically examining the techniques, design decisions, and trade-offs. Further, we articulate the research-practice gap between academic research (e.g., benchmarks and tasks) and real-world deployment (e.g., software-related code tasks), including code correctness, security, contextual awareness of large codebases, and integration with development workflows, and map promising research directions to practical needs. Last, we conduct a series of experiments to provide a comprehensive analysis of code pre-training, supervised fine-tuning, and reinforcement learning, covering scaling law, framework selection, hyperparameter sensitivity, model architectures, and dataset comparisons.

**arXiv ID:** 2511.18538
</details>

<details>
<summary><strong>Stabilizing Reinforcement Learning with LLMs: Formulation and Practices</strong> - Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li, Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin, Chencan Wu, Feng Hu, An Yang, Jingren Zhou, Junyang Lin - [[pdf]](https://arxiv.org/pdf/2512.01374)</summary>

**Abstract:** This paper proposes a novel formulation for reinforcement learning (RL) with large language models, explaining why and under what conditions the true sequence-level reward can be optimized via a surrogate token-level objective in policy gradient methods such as REINFORCE. Specifically, through a first-order approximation, we show that this surrogate becomes increasingly valid only when both the training-inference discrepancy and policy staleness are minimized. This insight provides a principled explanation for the crucial role of several widely adopted techniques in stabilizing RL training, including importance sampling correction, clipping, and particularly Routing Replay for Mixture-of-Experts (MoE) models. Through extensive experiments with a 30B MoE model totaling hundreds of thousands of GPU hours, we show that for on-policy training, the basic policy gradient algorithm with importance sampling correction achieves the highest training stability. When off-policy updates are introduced to accelerate convergence, combining clipping and Routing Replay becomes essential to mitigate the instability caused by policy staleness. Notably, once training is stabilized, prolonged optimization consistently yields comparable final performance regardless of cold-start initialization. We hope that the shared insights and the developed recipes for stable RL training will facilitate future research.

**arXiv ID:** 2512.01374
</details>

<details>
<summary><strong>Reinforcement Learning in POMDP's via Direct Gradient Ascent</strong> - Jonathan Baxter, Peter L. Bartlett - [[pdf]](https://arxiv.org/pdf/2512.02383)</summary>

**Abstract:** This paper discusses theoretical and experimental aspects of gradient-based approaches to the direct optimization of policy performance in controlled POMDPs. We introduce GPOMDP, a REINFORCE-like algorithm for estimating an approximation to the gradient of the average reward as a function of the parameters of a stochastic policy. The algorithm's chief advantages are that it requires only a single sample path of the underlying Markov chain, it uses only one free parameter $\beta\in [0,1)$, which has a natural interpretation in terms of bias-variance trade-off, and it requires no knowledge of the underlying state. We prove convergence of GPOMDP and show how the gradient estimates produced by GPOMDP can be used in a conjugate-gradient procedure to find local optima of the average reward.

**arXiv ID:** 2512.02383
</details>

<details>
<summary><strong>Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts</strong> - Zhongjian Qiao, Rui Yang, Jiafei Lyu, Xiu Li, Zhongxiang Dai, Zhuoran Yang, Siyang Gao, Shuang Qiu - [[pdf]](https://arxiv.org/pdf/2512.02486)</summary>

**Abstract:** Single-domain offline reinforcement learning (RL) often suffers from limited data coverage, while cross-domain offline RL handles this issue by leveraging additional data from other domains with dynamics shifts. However, existing studies primarily focus on train-time robustness (handling dynamics shifts from training data), neglecting the test-time robustness against dynamics perturbations when deployed in practical scenarios. In this paper, we investigate dual (both train-time and test-time) robustness against dynamics shifts in cross-domain offline RL. We first empirically show that the policy trained with cross-domain offline RL exhibits fragility under dynamics perturbations during evaluation, particularly when target domain data is limited. To address this, we introduce a novel robust cross-domain Bellman (RCB) operator, which enhances test-time robustness against dynamics perturbations while staying conservative to the out-of-distribution dynamics transitions, thus guaranteeing the train-time robustness. To further counteract potential value overestimation or underestimation caused by the RCB operator, we introduce two techniques, the dynamic value penalty and the Huber loss, into our framework, resulting in the practical \textbf{D}ual-\textbf{RO}bust \textbf{C}ross-domain \textbf{O}ffline RL (DROCO) algorithm. Extensive empirical results across various dynamics shift scenarios show that DROCO outperforms strong baselines and exhibits enhanced robustness to dynamics perturbations.

**arXiv ID:** 2512.02486
</details>

<details>
<summary><strong>GoRL: An Algorithm-Agnostic Framework for Online Reinforcement Learning with Generative Policies</strong> - Chubin Zhang, Zhenglin Wan, Feng Chen, Xingrui Yu, Ivor Tsang, Bo An - [[pdf]](https://arxiv.org/pdf/2512.02581)</summary>

**Abstract:** Reinforcement learning (RL) faces a persistent tension: policies that are stable to optimize are often too simple to represent the multimodal action distributions needed for complex control. Gaussian policies provide tractable likelihoods and smooth gradients, but their unimodal form limits expressiveness. Conversely, generative policies based on diffusion or flow matching can model rich multimodal behaviors; however, in online RL, they are frequently unstable due to intractable likelihoods and noisy gradients propagating through deep sampling chains. We address this tension with a key structural principle: decoupling optimization from generation. Building on this insight, we introduce GoRL (Generative Online Reinforcement Learning), a framework that optimizes a tractable latent policy while utilizing a conditional generative decoder to synthesize actions. A two-timescale update schedule enables the latent policy to learn stably while the decoder steadily increases expressiveness, without requiring tractable action likelihoods. Across a range of continuous-control tasks, GoRL consistently outperforms both Gaussian policies and recent generative-policy baselines. Notably, on the HopperStand task, it reaches a normalized return above 870, more than 3 times that of the strongest baseline. These results demonstrate that separating optimization from generation provides a practical path to policies that are both stable and highly expressive.

**arXiv ID:** 2512.02581
</details>

<details>
<summary><strong>SeeNav-Agent: Enhancing Vision-Language Navigation with Visual Prompt and Step-Level Policy Optimization</strong> - Zhengcheng Wang, Zichuan Lin, Yijun Yang, Haobo Fu, Deheng Ye - [[pdf]](https://arxiv.org/pdf/2512.02631)</summary>

**Abstract:** Existing Vision-Language Navigation (VLN) agents based on Large Vision-Language Models (LVLMs) often suffer from perception errors, reasoning errors, and planning errors, which significantly hinder their navigation performance. To address these limitations, a novel VLN agent framework, named SeeNav-Agent, is proposed in this work. First, to reduce perception hallucinations of the visual module of the VLN agent, a dual-view Visual Prompt (VP) technique is introduced in the input space, which can also improve the agent's understanding of current spatial states. Subsequently, a novel step-level Reinforcement Fine-Tuning (RFT) method, Step Reward Group Policy Optimization (SRGPO), is designed for the post-training of VLN agents. In SRGPO, we first define verifiable process rewards for the navigation task, and then perform efficient step-level advantage estimation by randomly grouping different navigation steps. SRGPO provides dense reward signals for the reinforcement learning process of the VLN agent and enhances its planning capability. Experimental results on the EmbodiedBench Navigation benchmark indicate that by introducing the zero-shot VP module, the GPT-4.1 achieves a navigation success rate of 86.7%, surpassing the current best LVLM by approximately 20 percentage points (pp). Through post-training based on SRGPO, the Qwen2.5-VL-3B model reaches a navigation success rate of 72.3%, outperforming the best existing LVLM model by 5.6 pp. Moreover, compared to RFT algorithms such as GRPO and GiGPO, the proposed SRGPO demonstrates significant improvements in training stability, convergence efficiency, and generalization capability.

**arXiv ID:** 2512.02631
</details>

<details>
<summary><strong>QJoin: Transformation-aware Joinable Data Discovery Using Reinforcement Learning</strong> - Ning Wang, Sainyam Galhotra - [[pdf]](https://arxiv.org/pdf/2512.02444)</summary>

**Abstract:** Discovering which tables in large, heterogeneous repositories can be joined and by what transformations is a central challenge in data integration and data discovery. Traditional join discovery methods are largely designed for equi-joins, which assume that join keys match exactly or nearly so. These techniques, while efficient in clean, well-normalized databases, fail in open or federated settings where identifiers are inconsistently formatted, embedded, or split across multiple columns. Approximate or fuzzy joins alleviate minor string variations but cannot capture systematic transformations. We introduce QJoin, a reinforcement-learning framework that learns and reuses transformation strategies across join tasks. QJoin trains an agent under a uniqueness-aware reward that balances similarity with key distinctiveness, enabling it to explore concise, high-value transformation chains. To accelerate new joins, we introduce two reuse mechanisms: (i) agent transfer, which initializes new policies from pretrained agents, and (ii) transformation reuse, which caches successful operator sequences for similar column clusters. On the AutoJoin Web benchmark (31 table pairs), QJoin achieves an average F1-score of 91.0%. For 19,990 join tasks in NYC+Chicago open datasets, Qjoin reduces runtime by up to 7.4% (13,747 s) by using reusing. These results demonstrate that transformation learning and reuse can make join discovery both more accurate and more efficient.

**arXiv ID:** 2512.02444
</details>

<details>
<summary><strong>Convergent Reinforcement Learning Algorithms for Stochastic Shortest Path Problem</strong> - Soumyajit Guin, Shalabh Bhatnagar - [[pdf]](https://arxiv.org/pdf/2508.13963)</summary>

**Abstract:** In this paper we propose two algorithms in the tabular setting and an algorithm for the function approximation setting for the Stochastic Shortest Path (SSP) problem. SSP problems form an important class of problems in Reinforcement Learning (RL), as other types of cost-criteria in RL can be formulated in the setting of SSP. We show asymptotic almost-sure convergence for all our algorithms. We observe superior performance of our tabular algorithms compared to other well-known convergent RL algorithms. We further observe reliable performance of our function approximation algorithm compared to other algorithms in the function approximation setting.

**arXiv ID:** 2508.13963
</details>

<details>
<summary><strong>BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management</strong> - Kevin Bradley Dsouza, Enoch Ofosu, Daniel Chukwuemeka Amaogu, Jérôme Pigeon, Richard Boudreault, Pooneh Maghoul, Juan Moreno-Cruz, Yuri Leonenko - [[pdf]](https://arxiv.org/pdf/2509.19846)</summary>

**Abstract:** Boreal forests store 30-40\% of terrestrial carbon, much in climate-vulnerable permafrost soils, making their management critical for climate mitigation. However, optimizing forest management for both carbon sequestration and permafrost preservation presents complex trade-offs that current tools cannot adequately address. We introduce BoreaRL, the first multi-objective reinforcement learning environment for climate-adaptive boreal forest management, featuring a physically-grounded simulator of coupled energy, carbon, and water fluxes. BoreaRL supports two training paradigms: site-specific mode for controlled studies and generalist mode for learning robust policies under environmental stochasticity. Through evaluation of multi-objective RL algorithms, we reveal a fundamental asymmetry in learning difficulty: carbon objectives are significantly easier to optimize than thaw (permafrost preservation) objectives, with thaw-focused policies showing minimal learning progress across both paradigms. In generalist settings, standard gradient-descent based preference-conditioned approaches fail, while a naive site selection approach achieves superior performance by strategically selecting training episodes. Analysis of learned strategies reveals distinct management philosophies, where carbon-focused policies favor aggressive high-density coniferous stands, while effective multi-objective policies balance species composition and density to protect permafrost while maintaining carbon gains. Our results demonstrate that robust climate-adaptive forest management remains challenging for current MORL methods, establishing BoreaRL as a valuable benchmark for developing more effective approaches. We open-source BoreaRL to accelerate research in multi-objective RL for climate applications.

**arXiv ID:** 2509.19846
</details>

<details>
<summary><strong>Offline Goal-conditioned Reinforcement Learning with Quasimetric Representations</strong> - Vivek Myers, Bill Chunyuan Zheng, Benjamin Eysenbach, Sergey Levine - [[pdf]](https://arxiv.org/pdf/2509.20478)</summary>

**Abstract:** Approaches for goal-conditioned reinforcement learning (GCRL) often use learned state representations to extract goal-reaching policies. Two frameworks for representation structure have yielded particularly effective GCRL algorithms: (1) *contrastive representations*, in which methods learn "successor features" with a contrastive objective that performs inference over future outcomes, and (2) *temporal distances*, which link the (quasimetric) distance in representation space to the transit time from states to goals. We propose an approach that unifies these two frameworks, using the structure of a quasimetric representation space (triangle inequality) with the right additional constraints to learn successor representations that enable optimal goal-reaching. Unlike past work, our approach is able to exploit a **quasimetric** distance parameterization to learn **optimal** goal-reaching distances, even with **suboptimal** data and in **stochastic** environments. This gives us the best of both worlds: we retain the stability and long-horizon capabilities of Monte Carlo contrastive RL methods, while getting the free stitching capabilities of quasimetric network parameterizations. On existing offline GCRL benchmarks, our representation learning objective improves performance on stitching tasks where methods based on contrastive learning struggle, and on noisy, high-dimensional environments where methods based on quasimetric networks struggle.

**arXiv ID:** 2509.20478
</details>

<details>
<summary><strong>Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents</strong> - Jane H. Lee, Baturay Saglam, Spyridon Pougkakiotis, Amin Karbasi, Dionysis Kalogerias - [[pdf]](https://arxiv.org/pdf/2510.20199)</summary>

**Abstract:** Constrained optimization provides a common framework for dealing with conflicting objectives in reinforcement learning (RL). In most of these settings, the objectives (and constraints) are expressed though the expected accumulated reward. However, this formulation neglects risky or even possibly catastrophic events at the tails of the reward distribution, and is often insufficient for high-stakes applications in which the risk involved in outliers is critical. In this work, we propose a framework for risk-aware constrained RL, which exhibits per-stage robustness properties jointly in reward values and time using optimized certainty equivalents (OCEs). Our framework ensures an exact equivalent to the original constrained problem within a parameterized strong Lagrangian duality framework under appropriate constraint qualifications, and yields a simple algorithmic recipe which can be wrapped around standard RL solvers, such as PPO. Lastly, we establish the convergence of the proposed algorithm under common assumptions, and verify the risk-aware properties of our approach through several numerical experiments.

**arXiv ID:** 2510.20199
</details>

<details>
<summary><strong>Non-stationary and Varying-discounting Markov Decision Processes for Reinforcement Learning</strong> - Zhizuo Chen, Theodore T. Allen - [[pdf]](https://arxiv.org/pdf/2511.17598)</summary>

**Abstract:** Algorithms developed under stationary Markov Decision Processes (MDPs) often face challenges in non-stationary environments, and infinite-horizon formulations may not directly apply to finite-horizon tasks. To address these limitations, we introduce the Non-stationary and Varying-discounting MDP (NVMDP) framework, which naturally accommodates non-stationarity and allows discount rates to vary with time and transitions. Infinite-horizon, stationary MDPs emerge as special cases of NVMDPs for identifying an optimal policy, and finite-horizon MDPs are also subsumed within the NVMDP formulations. Moreover, NVMDPs provide a flexible mechanism to shape optimal policies, without altering the state space, action space, or the reward structure. We establish the theoretical foundations of NVMDPs, including assumptions, state- and action-value formulation and recursion, matrix representation, optimality conditions, and policy improvement under finite state and action spaces. Building on these results, we adapt dynamic programming and generalized Q-learning algorithms to NVMDPs, along with formal convergence proofs. For problems requiring function approximation, we extend the Policy Gradient Theorem and the policy improvement bound in Trust Region Policy Optimization (TRPO), offering proofs in both scalar and matrix forms. Empirical evaluations in a non-stationary gridworld environment demonstrate that NVMDP-based algorithms successfully recover optimal trajectories under multiple reward and discounting schemes, whereas original Q-learning fails. These results collectively show that NVMDPs provide a theoretically sound and practically effective framework for reinforcement learning, requiring only minor algorithmic modifications while enabling robust handling of non-stationarity and explicit optimal policy shaping.

**arXiv ID:** 2511.17598
</details>

<details>
<summary><strong>Forecasting in Offline Reinforcement Learning for Non-stationary Environments</strong> - Suzan Ece Ada, Georg Martius, Emre Ugur, Erhan Oztop - [[pdf]](https://arxiv.org/pdf/2512.01987)</summary>

**Abstract:** Offline Reinforcement Learning (RL) provides a promising avenue for training policies from pre-collected datasets when gathering additional interaction data is infeasible. However, existing offline RL methods often assume stationarity or only consider synthetic perturbations at test time, assumptions that often fail in real-world scenarios characterized by abrupt, time-varying offsets. These offsets can lead to partial observability, causing agents to misperceive their true state and degrade performance. To overcome this challenge, we introduce Forecasting in Non-stationary Offline RL (FORL), a framework that unifies (i) conditional diffusion-based candidate state generation, trained without presupposing any specific pattern of future non-stationarity, and (ii) zero-shot time-series foundation models. FORL targets environments prone to unexpected, potentially non-Markovian offsets, requiring robust agent performance from the onset of each episode. Empirical evaluations on offline RL benchmarks, augmented with real-world time-series data to simulate realistic non-stationarity, demonstrate that FORL consistently improves performance compared to competitive baselines. By integrating zero-shot forecasting with the agent's experience, we aim to bridge the gap between offline RL and the complexities of real-world, non-stationary environments.

**arXiv ID:** 2512.01987
</details>

<details>
<summary><strong>Reinforcement Learning for Robotic Safe Control with Force Sensing</strong> - Nan Lin, Linrui Zhang, Yuxuan Chen, Zhenrui Chen, Yujun Zhu, Ruoxi Chen, Peichen Wu, Xiaoping Chen - [[pdf]](https://arxiv.org/pdf/2512.02022)</summary>

**Abstract:** For the task with complicated manipulation in unstructured environments, traditional hand-coded methods are ineffective, while reinforcement learning can provide more general and useful policy. Although the reinforcement learning is able to obtain impressive results, its stability and reliability is hard to guarantee, which would cause the potential safety threats. Besides, the transfer from simulation to real world also will lead in unpredictable situations. To enhance the safety and reliability of robots, we introduce the force and haptic perception into reinforcement learning. Force and tactual sensation play key roles in robotic dynamic control and human-robot interaction. We demonstrate that the force-based reinforcement learning method can be more adaptive to environment, especially in sim-to-real transfer. Experimental results show in object pushing task, our strategy is safer and more efficient in both simulation and real world, thus it holds prospects for a wide variety of robotic applications.

**arXiv ID:** 2512.02022
</details>

<details>
<summary><strong>Investigating the Integrated Digital Interventions Delivered by a Therapeutic Companion Agent for Young Adults with Symptoms of Depression: A Proof-of-Concept Study</strong> - Youngjae Yoo, Minuk Kim, Soyoung Kim, Gayeon Lee, Jinwoo Kim - [[pdf]](https://arxiv.org/pdf/2512.02608)</summary>

**Abstract:** Background: Despite the clinical effectiveness of digital interventions for young adults with depression, low engagement and adherence remain persistent challenges. Building a strong digital therapeutic alliance has been proposed to address these barriers. This study highlights the need for a conversational therapeutic companion agent (TCA)-based intervention design. Objective: This study aimed to develop a Wizard-of-Oz TCA-centered prototype integrating social-support-based ecological momentary assessment (EMA), ecological momentary intervention (EMI), behavioral activation, and gamification. We evaluated the six-week proof-of-concept efficacy of this intervention among young adults with depressive symptoms. Methods: Korean young adults aged 20--39 years with mild-to-moderate depressive symptoms (PHQ-9) were recruited online. The intervention group ($n = 29$) received a six-week TCA-based digital intervention, while the control group ($n = 29$), recruited four weeks later, continued their usual routines. The TCA guided four daily behavioral-activation tasks, three mood assessments, meditation, daily summaries, and weekly mission feedback. Both groups were assessed at baseline and at weeks 2, 4, and 6 using the BDI-II, GAD-7, and Q-LES-Q-SF. Results: Of 58 participants, 57 completed the study (one dropout in the intervention group). At week 6, the intervention group showed significantly greater reductions in depressive symptoms and improvements in quality of life than controls. Adherence was 78\% for EMA, 51\% for EMI, and 65\% for daily routines. Conclusions: The TCA-based digital intervention improved depressive symptoms and quality of life with adherence levels comparable to previous digital health interventions. Future studies should refine the TCA design and conduct larger-scale evaluations.

**arXiv ID:** 2512.02608
</details>

<details>
<summary><strong>Feed-O-Meter: Investigating AI-Generated Mentee Personas as Interactive Agents for Scaffolding Design Feedback Practice</strong> - Hyunseung Lim, Dasom Choi, DaEun Choi, Sooyohn Nam, Hwajung Hong - [[pdf]](https://arxiv.org/pdf/2509.07424)</summary>

**Abstract:** Effective feedback, including critique and evaluation, helps designers develop design concepts and refine their ideas, supporting informed decision-making throughout the iterative design process. However, in studio-based design courses, students often struggle to provide feedback due to a lack of confidence and fear of being judged, which limits their ability to develop essential feedback-giving skills. Recent advances in large language models (LLMs) suggest that role-playing with AI agents can let learners engage in multi-turn feedback without the anxiety of external judgment or the time constraints of real-world settings. Yet prior studies have raised concerns that LLMs struggle to behave like real people in role-play scenarios, diminishing the educational benefits of these interactions. Therefore, designing AI-based agents that effectively support learners in practicing and developing intellectual reasoning skills requires more than merely assigning the target persona's personality and role to the agent. By addressing these issues, we present Feed-O-Meter, a novel system that employs carefully designed LLM-based agents to create an environment in which students can practice giving design feedback. The system enables users to role-play as mentors, providing feedback to an AI mentee and allowing them to reflect on how that feedback impacts the AI mentee's idea development process. A user study (N=24) indicated that Feed-O-Meter increased participants' engagement and motivation through role-switching and helped them adjust feedback to be more comprehensible for an AI mentee. Based on these findings, we discuss future directions for designing systems to foster feedback skills in design education.

**arXiv ID:** 2509.07424
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
