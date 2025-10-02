# Agent arXiv Daily

**Last Updated:** 2025-10-02 01:58:41

**Total Papers:** 89

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (8 papers)</h2></summary>

<details>
<summary><strong>Less is More: Lean yet Powerful Vision-Language Model for Autonomous Driving</strong> - Sheng Yang, Tong Zhan, Guancheng Chen, Yanfeng Lu, Jian Wang - [[pdf]](https://arxiv.org/pdf/2510.00060)</summary>

**Abstract:** In this work, we reconceptualize autonomous driving as a generalized language and formulate the trajectory planning task as next waypoint prediction. We introduce Max-V1, a novel framework for one-stage end-to-end autonomous driving. Our framework presents a single-pass generation paradigm that aligns with the inherent sequentiality of driving. This approach leverages the generative capacity of the VLM (Vision-Language Model) to enable end-to-end trajectory prediction directly from front-view camera input. The efficacy of this method is underpinned by a principled supervision strategy derived from statistical modeling. This provides a well-defined learning objective, which makes the framework highly amenable to master complex driving policies through imitation learning from large-scale expert demonstrations. Empirically, our method achieves the state-of-the-art performance on the nuScenes dataset, delivers an overall improvement of over 30% compared to prior baselines. Furthermore, it exhibits superior generalization performance on cross-domain datasets acquired from diverse vehicles, demonstrating notable potential for cross-vehicle robustness and adaptability. Due to these empirical strengths, this work introduces a model enabling fundamental driving behaviors, laying the foundation for the development of more capable self-driving agents. Code will be available upon publication.

**arXiv ID:** 2510.00060
</details>

<details>
<summary><strong>Adaptive and Resource-efficient Agentic AI Systems for Mobile and Embedded Devices: A Survey</strong> - Sicong Liu, Weiye Wu, Xiangrui Xu, Teng Li, Bowen Pang, Bin Guo, Zhiwen Yu - [[pdf]](https://arxiv.org/pdf/2510.00078)</summary>

**Abstract:** Foundation models have reshaped AI by unifying fragmented architectures into scalable backbones with multimodal reasoning and contextual adaptation. In parallel, the long-standing notion of AI agents, defined by the sensing-decision-action loop, is entering a new paradigm: with FMs as their cognitive core, agents transcend rule-based behaviors to achieve autonomy, generalization, and self-reflection. This dual shift is reinforced by real-world demands such as autonomous driving, robotics, virtual assistants, and GUI agents, as well as ecosystem advances in embedded hardware, edge computing, mobile deployment platforms, and communication protocols that together enable large-scale deployment. Yet this convergence collides with reality: while applications demand long-term adaptability and real-time interaction, mobile and edge deployments remain constrained by memory, energy, bandwidth, and latency. This creates a fundamental tension between the growing complexity of FMs and the limited resources of deployment environments. This survey provides the first systematic characterization of adaptive, resource-efficient agentic AI systems. We summarize enabling techniques into elastic inference, test-time adaptation, dynamic multimodal integration, and agentic AI applications, and identify open challenges in balancing accuracy-latency-communication trade-offs and sustaining robustness under distribution shifts. We further highlight future opportunities in algorithm-system co-design, cognitive adaptation, and collaborative edge deployment. By mapping FM structures, cognition, and hardware resources, this work establishes a unified perspective toward scalable, adaptive, and resource-efficient agentic AI. We believe this survey can help readers to understand the connections between enabling technologies while promoting further discussions on the fusion of agentic intelligence and intelligent agents.

**arXiv ID:** 2510.00078
</details>

<details>
<summary><strong>Navigating the Synchrony-Stability Frontier in Adaptive Chatbots</strong> - T. James Brandt - [[pdf]](https://arxiv.org/pdf/2510.00339)</summary>

**Abstract:** Adaptive chatbots that mimic a user's linguistic style can build rapport and engagement, yet unconstrained mimicry risks an agent that feels unstable or sycophantic. We present a computational evaluation framework that makes the core design tension explicit: balancing moment-to-moment linguistic synchrony against long-term persona stability. Using an 8-dimensional style vector and a closed-loop "base+delta" prompting architecture, we simulate and compare explicit adaptation policies - Uncapped, Cap, Exponential Moving Average (EMA), Dead-Band, and Hybrids - on a human-log dataset. Our analysis maps a clear Pareto frontier: bounded policies achieve substantial gains in stability at a modest cost to synchrony. For example, a Hybrid (EMA+Cap) raises stability from 0.542 to 0.878 (+62%) while reducing synchrony by only 17%. We confirm this trade-off through large-scale replications on three public corpora (DailyDialog, Persona-Chat, EmpatheticDialogues) and LLM-in-the-loop validation across two model families. Furthermore, we quantify "prompt legibility," showing that frontier policies reduce instruction churn and cut jarring register flips (major tone changes) from 0.254 to 0.092, yielding systems that are easier to reason about and maintain. Taken together, our framework provides a general evaluation harness for style adaptation; a systematic ablation that identifies Pareto-efficient policies; robust validation across diverse datasets and models; and novel legibility metrics linking policy choices to system maintainability.

**arXiv ID:** 2510.00339
</details>

<details>
<summary><strong>Hybrid Dialogue State Tracking for Persian Chatbots: A Language Model-Based Approach</strong> - Samin Mahdipour Aghabagher, Saeedeh Momtazi - [[pdf]](https://arxiv.org/pdf/2510.01052)</summary>

**Abstract:** Dialogue State Tracking (DST) is an essential element of conversational AI with the objective of deeply understanding the conversation context and leading it toward answering user requests. Due to high demands for open-domain and multi-turn chatbots, the traditional rule-based DST is not efficient enough, since it cannot provide the required adaptability and coherence for human-like experiences in complex conversations. This study proposes a hybrid DST model that utilizes rule-based methods along with language models, including BERT for slot filling and intent detection, XGBoost for intent validation, GPT for DST, and online agents for real-time answer generation. This model is uniquely designed to be evaluated on a comprehensive Persian multi-turn dialogue dataset and demonstrated significantly improved accuracy and coherence over existing methods in Persian-based chatbots. The results demonstrate how effectively a hybrid approach may improve DST capabilities, paving the way for conversational AI systems that are more customized, adaptable, and human-like.

**arXiv ID:** 2510.01052
</details>

<details>
<summary><strong>ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs</strong> - Adi Simhi, Jonathan Herzig, Martin Tutek, Itay Itzhak, Idan Szpektor, Yonatan Belinkov - [[pdf]](https://arxiv.org/pdf/2510.00857)</summary>

**Abstract:** As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions. Benchmark & code available at this https URL.

**arXiv ID:** 2510.00857
</details>

<details>
<summary><strong>Taxonomy of Comprehensive Safety for Clinical Agents</strong> - Jean Seo, Hyunkyung Lee, Gibaeg Kim, Wooseok Han, Jaehyo Yoo, Seungseop Lim, Kihun Shin, Eunho Yang - [[pdf]](https://arxiv.org/pdf/2509.22041)</summary>

**Abstract:** Safety is a paramount concern in clinical chatbot applications, where inaccurate or harmful responses can lead to serious consequences. Existing methods--such as guardrails and tool calling--often fall short in addressing the nuanced demands of the clinical domain. In this paper, we introduce TACOS (TAxonomy of COmprehensive Safety for Clinical Agents), a fine-grained, 21-class taxonomy that integrates safety filtering and tool selection into a single user intent classification step. TACOS is a taxonomy that can cover a wide spectrum of clinical and non-clinical queries, explicitly modeling varying safety thresholds and external tool dependencies. To validate our taxonomy, we curate a TACOS-annotated dataset and perform extensive experiments. Our results demonstrate the value of a new taxonomy specialized for clinical agent settings, and reveal useful insights about train data distribution and pretrained knowledge of base models.

**arXiv ID:** 2509.22041
</details>

<details>
<summary><strong>Designing Psychometric Bias Measures for ChatBots: An Application to Racial Bias Measurement</strong> - Mouhacine Benosman - [[pdf]](https://arxiv.org/pdf/2509.13324)</summary>

**Abstract:** Artificial intelligence (AI), particularly in the form of large language models (LLMs) or chatbots, has become increasingly integrated into our daily lives. In the past five years, several LLMs have been introduced, including ChatGPT by OpenAI, Claude by Anthropic, and Llama by Meta, among others. These models have the potential to be employed across a wide range of human-machine interaction applications, such as chatbots for information retrieval, assistance in corporate hiring decisions, college admissions, financial loan approvals, parole determinations, and even in medical fields like psychotherapy delivered through chatbots. The key question is whether these chatbots will interact with humans in a bias-free manner or if they will further reinforce the existing pathological biases present in human-to-human interactions. If the latter is true, then how can we rigorously measure these biases?
We address this challenge by introducing STAMP-LLM (Standardized Test and Assessment Measurement Protocol for LLMs), a psychometric-based principled two-phase framework for designing psychometric measures to evaluate chatbot biases: (i) a Definitional phase for construct mapping, item development, and expert review; and (ii) a Data/Analysis phase for protocol control (prompts/decoding), automated sampling, pre-specified scoring, and basic reliability/validity checks. We illustrate STAMP-LLM on racial bias using one explicit and two implicit measures.

**arXiv ID:** 2509.13324
</details>

<details>
<summary><strong>"Having Lunch Now": Understanding How Users Engage with a Proactive Agent for Daily Planning and Self-Reflection</strong> - Adnan Abbas, Caleb Wohn, Arnav Jagtap, Eugenia H Rho, Sang Won Lee - [[pdf]](https://arxiv.org/pdf/2509.24073)</summary>

**Abstract:** Conversational agents have been studied as tools to scaffold planning and self-reflection for productivity and well-being. While prior work has demonstrated positive outcomes, we still lack a clear understanding of what drives these results and how users behave and communicate with agents that act as coaches rather than assistants. Such understanding is critical for designing interactions in which agents foster meaningful behavioral change. We conducted a 14-day longitudinal study with 12 participants using a proactive agent that initiated regular check-ins to support daily planning and reflection. Our findings reveal diverse interaction patterns: participants accepted or negotiated suggestions, developed shared mental models, reported progress, and at times resisted or disengaged. We also identified problematic aspects of the agent's behavior, including rigidity, premature turn-taking, and overpromising. Our work contributes to understanding how people interact with a proactive, coach-like agent and offers design considerations for facilitating effective behavioral change.

**arXiv ID:** 2509.24073
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>Drones that Think on their Feet: Sudden Landing Decisions with Embodied AI</strong> - Diego Ortiz Barbosa, Mohit Agrawal, Yash Malegaonkar, Luis Burbano, Axel Andersson, György Dán, Henrik Sandberg, Alvaro A. Cardenas - [[pdf]](https://arxiv.org/pdf/2510.00167)</summary>

**Abstract:** Autonomous drones must often respond to sudden events, such as alarms, faults, or unexpected changes in their environment, that require immediate and adaptive decision-making. Traditional approaches rely on safety engineers hand-coding large sets of recovery rules, but this strategy cannot anticipate the vast range of real-world contingencies and quickly becomes incomplete. Recent advances in embodied AI, powered by large visual language models, provide commonsense reasoning to assess context and generate appropriate actions in real time. We demonstrate this capability in a simulated urban benchmark in the Unreal Engine, where drones dynamically interpret their surroundings and decide on sudden maneuvers for safe landings. Our results show that embodied AI makes possible a new class of adaptive recovery and decision-making pipelines that were previously infeasible to design by hand, advancing resilience and safety in autonomous aerial systems.

**arXiv ID:** 2510.00167
</details>

<details>
<summary><strong>DualTune: Decoupled Fine-Tuning for On-Device Agentic Systems</strong> - Rohan Kadekodi, Zhan Jin, Keisuke Kamahori, Yile Gu, Sean Khatiri, Noah H. Bayindirli, Sergey Gorbunov, Baris Kasikci - [[pdf]](https://arxiv.org/pdf/2510.00229)</summary>

**Abstract:** The deployment of Large Language Models (LLMs) as agentic orchestrators has revolutionized task automation, but the need for privacy-preserving, cost-effective solutions demands on-device inference capabilities. However, local LLMs consistently underperform compared to frontier models in tool calling scenarios, struggling with both tool selection from large tool sets and accurate argument generation for complex parameter structures. We introduce a methodology that disaggregates a tool-calling task into two distinct subtasks: tool selection and argument generation. We propose "decoupled fine-tuning", a novel post-training approach that employs LoRA fine-tuning to create dedicated LoRA adapters for tool selection and tool-specific argument generation using separate loss masking for each of the subtasks. Furthermore, we present DualTune, an inference framework that leverages the LoRA adapters created using decoupled fine-tuning to perform efficient agent orchestration with the help of local models on end-user devices. DualTune decomposes the tool-call generation step into tool selection and argument generation, and dynamically loads the corresponding LoRA adapters to generate tool calls. Additionally, DualTune implements hierarchical orchestration to restrict the number of tools required for tool selection. Our experiments on the MCP-Bench benchmark demonstrate that the Qwen-2.5-7B model trained using decoupled fine-tuning improves the tool calling accuracy of the base model by 46%, and outperforms other local reasoning, non-reasoning and fine-tuned models of similar size in all cases, and models that are 2x larger, in most cases.

**arXiv ID:** 2510.00229
</details>

<details>
<summary><strong>Towards Self-Evolving Benchmarks: Synthesizing Agent Trajectories via Test-Time Exploration under Validate-by-Reproduce Paradigm</strong> - Dadi Guo, Tianyi Zhou, Dongrui Liu, Chen Qian, Qihan Ren, Shuai Shao, Zhiyuan Fan, Yi R. Fung, Kun Wang, Linfeng Zhang, Jing Shao - [[pdf]](https://arxiv.org/pdf/2510.00415)</summary>

**Abstract:** Recent advances in large language models (LLMs) and agent system designs have empowered agents with unprecedented levels of capability. However, existing agent benchmarks are showing a trend of rapid ceiling-hitting by newly developed agents, making it difficult to meet the demands for evaluating agent abilities. To address this problem, we propose the Trajectory-based Validated-by-Reproducing Agent-benchmark Complexity Evolution (TRACE) framework. This framework takes an original task from an existing benchmark and encourages agents to freely explore and evolve it into a new task with higher difficulty while recording validatable agent trajectories. The framework proceeds in three stages: (1) evolutionary proposal mining, which provides task evolution proposals through preliminary exploration and divergent thinking; (2) problem formation and free exploration, where proposals are conceptualized into feasible problem candidates and the agents then explore them freely while recording their execution trajectories; and (3) multi-level validation, which ensures that the evolved tasks are accompanied by validatable and reproducible trajectories. Experiments on the GAIA benchmark demonstrate that the TRACE framework consistently enhances task complexity while improving the reliability of correctness through validatable execution trajectories. This work marks a paradigm shift from static, manually curated benchmarks to dynamic, self-evolving evaluation systems, providing a sustainable and challenging runway for agent development.

**arXiv ID:** 2510.00415
</details>

<details>
<summary><strong>EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases</strong> - Kwanhyung Lee, Sungsoo Hong, Joonhyung Park, Jeonghyeop Lim, Juhwan Choi, Donghwee Yoon, Eunho Yang - [[pdf]](https://arxiv.org/pdf/2510.00549)</summary>

**Abstract:** Machine learning models for clinical prediction rely on structured data extracted from Electronic Medical Records (EMRs), yet this process remains dominated by hardcoded, database-specific pipelines for cohort definition, feature selection, and code mapping. These manual efforts limit scalability, reproducibility, and cross-institutional generalization. To address this, we introduce EMR-AGENT (Automated Generalized Extraction and Navigation Tool), an agent-based framework that replaces manual rule writing with dynamic, language model-driven interaction to extract and standardize structured clinical data. Our framework automates cohort selection, feature extraction, and code mapping through interactive querying of databases. Our modular agents iteratively observe query results and reason over schema and documentation, using SQL not just for data retrieval but also as a tool for database observation and decision making. This eliminates the need for hand-crafted, schema-specific logic. To enable rigorous evaluation, we develop a benchmarking codebase for three EMR databases (MIMIC-III, eICU, SICdb), including both seen and unseen schema settings. Our results demonstrate strong performance and generalization across these databases, highlighting the feasibility of automating a process previously thought to require expert-driven design. The code will be released publicly at this https URL. For a demonstration, please visit our anonymous demo page: this https URL

**arXiv ID:** 2510.00549
</details>

<details>
<summary><strong>TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments</strong> - Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, Rameswar Panda - [[pdf]](https://arxiv.org/pdf/2510.01179)</summary>

**Abstract:** Large Language Model (LLM) agents are rapidly emerging as powerful systems for automating tasks across domains. Yet progress in the open-source community is constrained by the lack of high quality permissively licensed tool-agentic training data. Existing datasets are often limited in diversity, realism, and complexity, particularly regarding multi-tool and multi-turn interactions. To address this gap, we introduce Toucan, the largest publicly available tool-agentic dataset to date, containing 1.5 million trajectories synthesized from nearly 500 real-world Model Context Protocols (MCPs). Unlike prior work, Toucan leverages authentic MCP environments to generate diverse, realistic, and challenging tasks with trajectories involving real tool execution. Our pipeline first produces a broad spectrum of tool-use queries using five distinct models, applies model-based quality filtering, and then generates agentic trajectories with three teacher models using two agentic frameworks. Rigorous rule-based and model-based validation ensures high-quality outputs. We also introduce three extension mechanisms to further diversify tasks and simulate multi-turn conversations. Models fine-tuned on Toucan outperform larger closed-source counterparts on the BFCL V3 benchmark and push the Pareto frontier forward on MCP-Universe Bench.

**arXiv ID:** 2510.01179
</details>

<details>
<summary><strong>Agent Fine-tuning through Distillation for Domain-specific LLMs in Microdomains</strong> - Yawen Xue, Masaya Tsunokake, Yuta Koreeda, Ekant Muljibhai Amin, Takashi Sumiyoshi, Yasuhiro Sogawa - [[pdf]](https://arxiv.org/pdf/2510.00482)</summary>

**Abstract:** Agentic large language models (LLMs) have become prominent for autonomously interacting with external environments and performing multi-step reasoning tasks. Most approaches leverage these capabilities via in-context learning with few-shot prompts, but this often results in lengthy inputs and higher computational costs. Agent fine-tuning offers an alternative by enabling LLMs to internalize procedural reasoning and domain-specific knowledge through training on relevant data and demonstration trajectories. While prior studies have focused on general domains, their effectiveness in specialized technical microdomains remains unclear. This paper explores agent fine-tuning for domain adaptation within Hitachi's JP1 middleware, a microdomain for specialized IT operations. We fine-tuned LLMs using JP1-specific datasets derived from domain manuals and distilled reasoning trajectories generated by LLMs themselves, enhancing decision making accuracy and search efficiency. During inference, we used an agentic prompt with retrieval-augmented generation and introduced a context-answer extractor to improve information relevance. On JP1 certification exam questions, our method achieved a 14% performance improvement over the base model, demonstrating the potential of agent fine-tuning for domain-specific reasoning in complex microdomains.

**arXiv ID:** 2510.00482
</details>

<details>
<summary><strong>Agent-ScanKit: Unraveling Memory and Reasoning of Multimodal Agents via Sensitivity Perturbations</strong> - Pengzhou Cheng, Lingzhong Dong, Zeng Wu, Zongru Wu, Xiangru Tang, Chengwei Qin, Zhuosheng Zhang, Gongshen Liu - [[pdf]](https://arxiv.org/pdf/2510.00496)</summary>

**Abstract:** Although numerous strategies have recently been proposed to enhance the autonomous interaction capabilities of multimodal agents in graphical user interface (GUI), their reliability remains limited when faced with complex or out-of-domain tasks. This raises a fundamental question: Are existing multimodal agents reasoning spuriously? In this paper, we propose \textbf{Agent-ScanKit}, a systematic probing framework to unravel the memory and reasoning capabilities of multimodal agents under controlled perturbations. Specifically, we introduce three orthogonal probing paradigms: visual-guided, text-guided, and structure-guided, each designed to quantify the contributions of memorization and reasoning without requiring access to model internals. In five publicly available GUI benchmarks involving 18 multimodal agents, the results demonstrate that mechanical memorization often outweighs systematic reasoning. Most of the models function predominantly as retrievers of training-aligned knowledge, exhibiting limited generalization. Our findings underscore the necessity of robust reasoning modeling for multimodal agents in real-world scenarios, offering valuable insights toward the development of reliable multimodal agents.

**arXiv ID:** 2510.00496
</details>

<details>
<summary><strong>GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness</strong> - Kung-Hsiang Huang, Haoyi Qiu, Yutong Dai, Caiming Xiong, Chien-Sheng Wu - [[pdf]](https://arxiv.org/pdf/2510.00536)</summary>

**Abstract:** Graphical user interface (GUI) agents built on vision-language models have emerged as a promising approach to automate human-computer workflows. However, they also face the inefficiency challenge as they process long sequences of high-resolution screenshots and solving long-horizon tasks, making inference slow, costly and memory-bound. While key-value (KV) caching can mitigate this, storing the full cache is prohibitive for image-heavy contexts. Existing cache-compression methods are sub-optimal as they do not account for the spatial and temporal redundancy of GUIs. In this work, we first analyze attention patterns in GUI agent workloads and find that, unlike in natural images, attention sparsity is uniformly high across all transformer layers. This insight motivates a simple uniform budget allocation strategy, which we show empirically outperforms more complex layer-varying schemes. Building on this, we introduce GUI-KV, a plug-and-play KV cache compression method for GUI agents that requires no retraining. GUI-KV combines two novel techniques: (i) spatial saliency guidance, which augments attention scores with the L2 norm of hidden states to better preserve semantically important visual tokens, and (ii) temporal redundancy scoring, which projects previous frames' keys onto the current frame's key subspace to preferentially prune redundant history. Across standard GUI agent benchmarks and models, GUI-KV outperforms competitive KV compression baselines, closely matching full-cache accuracy at modest budgets. Notably, in a 5-screenshot setting on the AgentNetBench benchmark, GUI-KV reduces decoding FLOPs by 38.9% while increasing step accuracy by 4.1% over the full-cache baseline. These results demonstrate that exploiting GUI-specific redundancies enables efficient and reliable agent performance.

**arXiv ID:** 2510.00536
</details>

<details>
<summary><strong>Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling</strong> - Pengfei Wang, Baolin Sun, Xuemei Dong, Yaxun Dai, Hongwei Yuan, Mengdie Chu, Yingqi Gao, Xiang Qi, Peng Zhang, Ying Yan - [[pdf]](https://arxiv.org/pdf/2509.24403)</summary>

**Abstract:** State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance.

**arXiv ID:** 2509.24403
</details>

<details>
<summary><strong>SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents</strong> - Jianshuo Dong, Sheng Guo, Hao Wang, Zhuotao Liu, Tianwei Zhang, Ke Xu, Minlie Huang, Han Qiu - [[pdf]](https://arxiv.org/pdf/2509.23694)</summary>

**Abstract:** Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: this https URL.

**arXiv ID:** 2509.23694
</details>

<details>
<summary><strong>ImpedanceGPT: VLM-driven Impedance Control of Swarm of Mini-drones for Intelligent Navigation in Dynamic Environment</strong> - Faryal Batool, Yasheerah Yaqoot, Malaika Zafar, Roohan Ahmed Khan, Muhammad Haris Khan, Aleksey Fedoseev, Dzmitry Tsetserukou - [[pdf]](https://arxiv.org/pdf/2503.02723)</summary>

**Abstract:** Swarm robotics plays a crucial role in enabling autonomous operations in dynamic and unpredictable environments. However, a major challenge remains ensuring safe and efficient navigation in environments filled with both dynamic alive (e.g., humans) and dynamic inanimate (e.g., non-living objects) obstacles. In this paper, we propose ImpedanceGPT, a novel system that combines a Vision-Language Model (VLM) with retrieval-augmented generation (RAG) to enable real-time reasoning for adaptive navigation of mini-drone swarms in complex environments.
The key innovation of ImpedanceGPT lies in the integration of VLM and RAG, which provides the drones with enhanced semantic understanding of their surroundings. This enables the system to dynamically adjust impedance control parameters in response to obstacle types and environmental conditions. Our approach not only ensures safe and precise navigation but also improves coordination between drones in the swarm.
Experimental evaluations demonstrate the effectiveness of the system. The VLM-RAG framework achieved an obstacle detection and retrieval accuracy of 80 % under optimal lighting. In static environments, drones navigated dynamic inanimate obstacles at 1.4 m/s but slowed to 0.7 m/s with increased separation around humans. In dynamic environments, speed adjusted to 1.0 m/s near hard obstacles, while reducing to 0.6 m/s with higher deflection to safely avoid moving humans.

**arXiv ID:** 2503.02723
</details>

<details>
<summary><strong>Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning</strong> - Bosung Kim, Prithviraj Ammanabrolu - [[pdf]](https://arxiv.org/pdf/2505.16928)</summary>

**Abstract:** We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI. $\infty$-THOR provides: (1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories; (2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and (3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences. To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.

**arXiv ID:** 2505.16928
</details>

<details>
<summary><strong>Grounded GUI Understanding for Vision-Based Spatial Intelligent Agent: Exemplified by Extended Reality Apps</strong> - Shuqing Li, Binchang Li, Yepang Liu, Cuiyun Gao, Jianping Zhang, Shing-Chi Cheung, Michael R. Lyu - [[pdf]](https://arxiv.org/pdf/2409.10811)</summary>

**Abstract:** In recent years, spatial computing a.k.a. Extended Reality (XR) has emerged as a transformative technology, offering users immersive and interactive experiences across diversified virtual environments. Users can interact with XR apps through interactable GUI elements (IGEs) on the stereoscopic three-dimensional (3D) graphical user interface (GUI). The accurate recognition of these IGEs is instrumental, serving as the foundation of many software engineering tasks, including automated testing and effective GUI search. The most recent IGE detection approaches for 2D mobile apps typically train a supervised object detection model based on a large-scale manually-labeled GUI dataset, usually with a pre-defined set of clickable GUI element categories like buttons and spinners. Such approaches can hardly be applied to IGE detection in XR apps, due to a multitude of challenges including complexities posed by open-vocabulary and heterogeneous IGE categories, intricacies of context-sensitive interactability, and the necessities of precise spatial perception and visual-semantic alignment for accurate IGE detection results. Thus, it is necessary to embark on the IGE research tailored to XR apps. In this paper, we propose the first zero-shot cOntext-sensitive inteRactable GUI ElemeNT dEtection framework for virtual Reality apps, named Orienter. By imitating human behaviors, Orienter observes and understands the semantic contexts of XR app scenes first, before performing the detection. The detection process is iterated within a feedback-directed validation and reflection loop. Specifically, Orienter contains three components, including (1) Semantic context comprehension, (2) Reflection-directed IGE candidate detection, and (3) Context-sensitive interactability classification. Extensive experiments demonstrate that Orienter is more effective than the state-of-the-art GUI element detection approaches.

**arXiv ID:** 2409.10811
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>ACON: Optimizing Context Compression for Long-horizon LLM Agents</strong> - Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A. Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, Saravan Rajmohan - [[pdf]](https://arxiv.org/pdf/2510.00615)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement.

**arXiv ID:** 2510.00615
</details>

<details>
<summary><strong>Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX</strong> - Anastasia Vepreva, Julia Razlivina, Maria Eremeeva, Nina Gubina, Anastasia Orlova, Aleksei Dmitrenko, Ksenya Kapranova, Susan Jyakhwo, Nikita Vasilev, Arsen Sarkisyan, Ivan Yu. Chernyshov, Vladimir Vinogradov, Andrei Dmitrenko - [[pdf]](https://arxiv.org/pdf/2510.00795)</summary>

**Abstract:** The emergence of agent-based systems represents a significant advancement in artificial intelligence, with growing applications in automated data extraction. However, chemical information extraction remains a formidable challenge due to the inherent heterogeneity of chemical data. Current agent-based approaches, both general-purpose and domain-specific, exhibit limited performance in this domain. To address this gap, we present ChemX, a comprehensive collection of 10 manually curated and domain-expert-validated datasets focusing on nanomaterials and small molecules. These datasets are designed to rigorously evaluate and enhance automated extraction methodologies in chemistry. To demonstrate their utility, we conduct an extensive benchmarking study comparing existing state-of-the-art agentic systems such as ChatGPT Agent and chemical-specific data extraction agents. Additionally, we introduce our own single-agent approach that enables precise control over document preprocessing prior to extraction. We further evaluate the performance of modern baselines, such as GPT-5 and GPT-5 Thinking, to compare their capabilities with agentic approaches. Our empirical findings reveal persistent challenges in chemical information extraction, particularly in processing domain-specific terminology, complex tabular and schematic representations, and context-dependent ambiguities. The ChemX benchmark serves as a critical resource for advancing automated information extraction in chemistry, challenging the generalization capabilities of existing methods, and providing valuable insights into effective evaluation strategies.

**arXiv ID:** 2510.00795
</details>

<details>
<summary><strong>A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning</strong> - Ruiyi Wang, Prithviraj Ammanabrolu - [[pdf]](https://arxiv.org/pdf/2510.01132)</summary>

**Abstract:** We study what actually works and what doesn't for training large language models as agents via multi-turn reinforcement learning. Despite rapid progress, existing frameworks and definitions are fragmented, and there is no systematic formulation or analysis of which design choices matter across tasks. We address this gap by first breaking down the design space into three inter-related pillars -- environment, reward, and policy -- and empirically derive a recipe for training LLM agents in situated textual domains. In particular, we test TextWorld and ALFWorld, popular domains for testing situated embodied reasoning, as well as SWE-Gym for more software engineering style tasks. (i) For the environment, we analyze the impacts of task complexity in terms of sizes of the state and action spaces as well as optimal solution length, finding that even simple environments within a domain can provide signal on how well an agent can generalize to more complex tasks. (ii) For the reward, we ablate relative reward sparsity, observing that while dense turn-level rewards accelerate training, performance and stability is highly dependent on the choice of RL algorithm. (iii) And for the agent's policy, we explore the interplay between reward sparsity and biased (PPO, GRPO) and unbiased (RLOO) policy gradient methods in addition to showing how to find the optimal Supervised Fine-tuning (SFT) to RL training ratio given a fixed budget. We distill these findings into a training recipe that guides co-design across the three pillars, facilitating research and practical efforts in multi-turn agentic RL. Code: this https URL

**arXiv ID:** 2510.01132
</details>

<details>
<summary><strong>Social Welfare Function Leaderboard: When LLM Agents Allocate Social Welfare</strong> - Zhengliang Shi, Ruotian Ma, Jen-tse Huang, Xinbei Ma, Xingyu Chen, Mengru Wang, Qu Yang, Yue Wang, Fanghua Ye, Ziyang Chen, Shanyi Wang, Cixing Li, Wenxuan Wang, Zhaopeng Tu, Xiaolong Li, Zhaochun Ren, Linus - [[pdf]](https://arxiv.org/pdf/2510.01164)</summary>

**Abstract:** Large language models (LLMs) are increasingly entrusted with high-stakes decisions that affect human welfare. However, the principles and values that guide these models when distributing scarce societal resources remain largely unexamined. To address this, we introduce the Social Welfare Function (SWF) Benchmark, a dynamic simulation environment where an LLM acts as a sovereign allocator, distributing tasks to a heterogeneous community of recipients. The benchmark is designed to create a persistent trade-off between maximizing collective efficiency (measured by Return on Investment) and ensuring distributive fairness (measured by the Gini coefficient). We evaluate 20 state-of-the-art LLMs and present the first leaderboard for social welfare allocation. Our findings reveal three key insights: (i) A model's general conversational ability, as measured by popular leaderboards, is a poor predictor of its allocation skill. (ii) Most LLMs exhibit a strong default utilitarian orientation, prioritizing group productivity at the expense of severe inequality. (iii) Allocation strategies are highly vulnerable, easily perturbed by output-length constraints and social-influence framing. These results highlight the risks of deploying current LLMs as societal decision-makers and underscore the need for specialized benchmarks and targeted alignment for AI governance.

**arXiv ID:** 2510.01164
</details>

<details>
<summary><strong>AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents</strong> - Akshat Naik, Patrick Quinn, Guillermo Bosch, Emma Gouné, Francisco Javier Campos Zabala, Jason Ross Brown, Edward James Young - [[pdf]](https://arxiv.org/pdf/2506.04018)</summary>

**Abstract:** As Large Language Model (LLM) agents become more widespread, associated misalignment risks increase. While prior research has studied agents' ability to produce harmful outputs or follow malicious instructions, it remains unclear how likely agents are to spontaneously pursue unintended goals in realistic deployments. In this work, we approach misalignment as a conflict between the internal goals pursued by the model and the goals intended by its deployer. We introduce a misalignment propensity benchmark, \textsc{AgentMisalignment}, a benchmark suite designed to evaluate the propensity of LLM agents to misalign in realistic scenarios. Evaluations cover behaviours such as avoiding oversight, resisting shutdown, sandbagging, and power-seeking. Testing frontier models, we find that more capable agents tend to exhibit higher misalignment on average. We also systematically vary agent personalities through different system prompts and observe that persona characteristics can strongly and unpredictably influence misalignment, sometimes more than the choice of model itself. Our results reveal the limitations of current alignment methods for autonomous LLM agents and underscore the need to rethink misalignment in realistic deployment settings.

**arXiv ID:** 2506.04018
</details>

<details>
<summary><strong>RELATE-Sim: Leveraging Turning Point Theory and LLM Agents to Predict and Understand Long-Term Relationship Dynamics through Interactive Narrative Simulations</strong> - Matthew Yue, Zhikun Xu, Vivek Gupta, Thao Ha, Liesal Sharabi, Ben Zhou - [[pdf]](https://arxiv.org/pdf/2510.00414)</summary>

**Abstract:** Most dating technologies optimize for getting together, not staying together. We present RELATE-Sim, a theory-grounded simulator that models how couples behave at consequential turning points-exclusivity talks, conflict-and-repair episodes, relocations-rather than static traits. Two persona-aligned LLM agents (one per partner) interact under a centralized Scene Master that frames each turning point as a compact set of realistic options, advances the narrative, and infers interpretable state changes and an auditable commitment estimate after each scene. On a longitudinal dataset of 71 couples with two-year follow-ups, simulation-aware predictions outperform a personas-only baseline while surfacing actionable markers (e.g., repair attempts acknowledged, clarity shifts) that explain why trajectories diverge. RELATE-Sim pushes the relationship research's focus from matchmaking to maintenance, providing a transparent, extensible platform for understanding and forecasting long-term relationship dynamics.

**arXiv ID:** 2510.00414
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>AuditAgent: Expert-Guided Multi-Agent Reasoning for Cross-Document Fraudulent Evidence Discovery</strong> - Songran Bai, Bingzhe Wu, Yiwei Zhang, Chengke Wu, Xiaolong Zheng, Yaze Yuan, Ke Wu, Jianqiang Li - [[pdf]](https://arxiv.org/pdf/2510.00156)</summary>

**Abstract:** Financial fraud detection in real-world scenarios presents significant challenges due to the subtlety and dispersion of evidence across complex, multi-year financial disclosures. In this work, we introduce a novel multi-agent reasoning framework AuditAgent, enhanced with auditing domain expertise, for fine-grained evidence chain localization in financial fraud cases. Leveraging an expert-annotated dataset constructed from enforcement documents and financial reports released by the China Securities Regulatory Commission, our approach integrates subject-level risk priors, a hybrid retrieval strategy, and specialized agent modules to efficiently identify and aggregate cross-report evidence. Extensive experiments demonstrate that our method substantially outperforms General-Purpose Agent paradigm in both recall and interpretability, establishing a new benchmark for automated, transparent financial forensics. Our results highlight the value of domain-specific reasoning and dataset construction for advancing robust financial fraud detection in practical, real-world regulatory applications.

**arXiv ID:** 2510.00156
</details>

<details>
<summary><strong>Learning to Lead Themselves: Agentic AI in MAS using MARL</strong> - Ansh Kamthan - [[pdf]](https://arxiv.org/pdf/2510.00022)</summary>

**Abstract:** As autonomous systems move from prototypes to real deployments, the ability of multiple agents to make decentralized, cooperative decisions becomes a core requirement. This paper examines how agentic artificial intelligence, agents that act independently, adaptively and proactively can improve task allocation and coordination in multi-agent systems, with primary emphasis on drone delivery and secondary relevance to warehouse automation. We formulate the problem in a cooperative multi-agent reinforcement learning setting and implement a lightweight multi-agent Proximal Policy Optimization, called IPPO, approach in PyTorch under a centralized-training, decentralized-execution paradigm. Experiments are conducted in PettingZoo environment, where multiple homogeneous drones or agents must self-organize to cover distinct targets without explicit communication.

**arXiv ID:** 2510.00022
</details>

<details>
<summary><strong>MAGIC-MASK: Multi-Agent Guided Inter-Agent Collaboration with Mask-Based Explainability for Reinforcement Learning</strong> - Maisha Maliha, Dean Hougen - [[pdf]](https://arxiv.org/pdf/2510.00274)</summary>

**Abstract:** Understanding the decision-making process of Deep Reinforcement Learning agents remains a key challenge for deploying these systems in safety-critical and multi-agent environments. While prior explainability methods like StateMask, have advanced the identification of critical states, they remain limited by computational cost, exploration coverage, and lack of adaptation to multi-agent settings. To overcome these limitations, we propose a mathematically grounded framework, MAGIC-MASK (Multi-Agent Guided Inter-agent Collaboration with Mask-Based Explainability for Reinforcement Learning), that extends perturbation-based explanation to Multi-Agent Reinforcement Learning. Our method integrates Proximal Policy Optimization, adaptive epsilon-greedy exploration, and lightweight inter-agent collaboration to share masked state information and peer experience. This collaboration enables each agent to perform saliency-guided masking and share reward-based insights with peers, reducing the time required for critical state discovery, improving explanation fidelity, and leading to faster and more robust learning. The core novelty of our approach lies in generalizing explainability from single-agent to multi-agent systems through a unified mathematical formalism built on trajectory perturbation, reward fidelity analysis, and Kullback-Leibler divergence regularization. This framework yields localized, interpretable explanations grounded in probabilistic modeling and multi-agent Markov decision processes. We validate our framework on both single-agent and multi-agent benchmarks, including a multi-agent highway driving environment and Google Research Football, demonstrating that MAGIC-MASK consistently outperforms state-of-the-art baselines in fidelity, learning efficiency, and policy robustness while offering interpretable and transferable explanations.

**arXiv ID:** 2510.00274
</details>

<details>
<summary><strong>Semantic-Driven AI Agent Communications: Challenges and Solutions</strong> - Kaiwen Yu, Mengying Sun, Zhijin Qin, Xiaodong Xu, Ping Yang, Yue Xiao, Gang Wu - [[pdf]](https://arxiv.org/pdf/2510.00381)</summary>

**Abstract:** With the rapid growth of intelligent services, communication targets are shifting from humans to artificial intelligent (AI) agents, which require new paradigms to enable real-time perception, decision-making, and collaboration. Semantic communication, which conveys task-relevant meaning rather than raw data, offers a promising solution. However, its practical deployment remains constrained by dynamic environments and limited resources. To address these issues, this article proposes a semantic-driven AI agent communication framework and develops three enabling techniques. First, semantic adaptation transmission applies fine-tuning with real or generative samples to efficiently adapt models to varying environments. Second, semantic lightweight transmission incorporates pruning, quantization, and perception-aware sampling to reduce model complexity and alleviate computational burden on edge agents. Third, semantic self-evolution control employs distributed hierarchical decision-making to optimize multi-dimensional resources, enabling robust multi-agent collaboration in dynamic environments. Simulation results show that the proposed solutions achieve faster convergence and stronger robustness, while the proposed distributed hierarchical optimization method significantly outperforms conventional decision-making schemes, highlighting its potential for AI agent communication networks.

**arXiv ID:** 2510.00381
</details>

<details>
<summary><strong>Expandable Decision-Making States for Multi-Agent Deep Reinforcement Learning in Soccer Tactical Analysis</strong> - Kenjiro Ide, Taiga Someya, Kohei Kawaguchi, Keisuke Fujii - [[pdf]](https://arxiv.org/pdf/2510.00480)</summary>

**Abstract:** Invasion team sports such as soccer produce a high-dimensional, strongly coupled state space as many players continuously interact on a shared field, challenging quantitative tactical analysis. Traditional rule-based analyses are intuitive, while modern predictive machine learning models often perform pattern-matching without explicit agent representations. The problem we address is how to build player-level agent models from data, whose learned values and policies are both tactically interpretable and robust across heterogeneous data sources. Here, we propose Expandable Decision-Making States (EDMS), a semantically enriched state representation that augments raw positions and velocities with relational variables (e.g., scoring of space, pass, and score), combined with an action-masking scheme that gives on-ball and off-ball agents distinct decision sets. Compared to prior work, EDMS maps learned value functions and action policies to human-interpretable tactical concepts (e.g., marking pressure, passing lanes, ball accessibility) instead of raw coordinate features, and aligns agent choices with the rules of play. In the experiments, EDMS with action masking consistently reduced both action-prediction loss and temporal-difference (TD) error compared to the baseline. Qualitative case studies and Q-value visualizations further indicate that EDMS highlights high-risk, high-reward tactical patterns (e.g., fast counterattacks and defensive breakthroughs). We also integrated our approach into an open-source library and demonstrated compatibility with multiple commercial and open datasets, enabling cross-provider evaluation and reproducible experiments.

**arXiv ID:** 2510.00480
</details>

<details>
<summary><strong>EpidemIQs: Prompt-to-Paper LLM Agents for Epidemic Modeling and Analysis</strong> - Mohammad Hossein Samaei, Faryad Darabi Sahneh, Lee W. Cohnstaedt, Caterina Scoglio - [[pdf]](https://arxiv.org/pdf/2510.00024)</summary>

**Abstract:** Large Language Models (LLMs) offer new opportunities to automate complex interdisciplinary research domains. Epidemic modeling, characterized by its complexity and reliance on network science, dynamical systems, epidemiology, and stochastic simulations, represents a prime candidate for leveraging LLM-driven automation. We introduce \textbf{EpidemIQs}, a novel multi-agent LLM framework that integrates user inputs and autonomously conducts literature review, analytical derivation, network modeling, mechanistic modeling, stochastic simulations, data visualization and analysis, and finally documentation of findings in a structured manuscript. We introduced two types of agents: a scientist agent for planning, coordination, reflection, and generation of final results, and a task-expert agent to focus exclusively on one specific duty serving as a tool to the scientist agent. The framework consistently generated complete reports in scientific article format. Specifically, using GPT 4.1 and GPT 4.1 mini as backbone LLMs for scientist and task-expert agents, respectively, the autonomous process completed with average total token usage 870K at a cost of about \$1.57 per study, achieving a 100\% completion success rate through our experiments. We evaluate EpidemIQs across different epidemic scenarios, measuring computational cost, completion success rate, and AI and human expert reviews of generated reports. We compare EpidemIQs to the single-agent LLM, which has the same system prompts and tools, iteratively planning, invoking tools, and revising outputs until task completion. The comparison shows consistently higher performance of the proposed framework across five different scenarios. EpidemIQs represents a step forward in accelerating scientific research by significantly reducing costs and turnaround time of discovery processes, and enhancing accessibility to advanced modeling tools.

**arXiv ID:** 2510.00024
</details>

<details>
<summary><strong>VibeCodeHPC: An Agent-Based Iterative Prompting Auto-Tuner for HPC Code Generation Using LLMs</strong> - Shun-ichiro Hayashi, Koki Morita, Daichi Mukunoki, Tetsuya Hoshino, Takahiro Katagiri - [[pdf]](https://arxiv.org/pdf/2510.00031)</summary>

**Abstract:** We propose VibeCodeHPC, an automatic tuning system for HPC programs based on multi-agent LLMs for code generation. VibeCodeHPC tunes programs through multi-agent role allocation and iterative prompt refinement. We describe the system configuration with four roles: Project Manager (PM), System Engineer (SE), Programmer (PG), and Continuous Delivery (CD). We introduce dynamic agent deployment and activity monitoring functions to facilitate effective multi-agent collaboration. In our case study, we convert and optimize CPU-based matrix-matrix multiplication code written in C to GPU code using CUDA. The multi-agent configuration of VibeCodeHPC achieved higher-quality code generation per unit time compared to a solo-agent configuration. Additionally, the dynamic agent deployment and activity monitoring capabilities facilitated more effective identification of requirement violations and other issues.

**arXiv ID:** 2510.00031
</details>

<details>
<summary><strong>A Hierarchical Agentic Framework for Autonomous Drone-Based Visual Inspection</strong> - Ethan Herron, Xian Yeow Lee, Gregory Sin, Teresa Gonzalez Diaz, Ahmed Farahat, Chetan Gupta - [[pdf]](https://arxiv.org/pdf/2510.00259)</summary>

**Abstract:** Autonomous inspection systems are essential for ensuring the performance and longevity of industrial assets. Recently, agentic frameworks have demonstrated significant potential for automating inspection workflows but have been limited to digital tasks. Their application to physical assets in real-world environments, however, remains underexplored. In this work, our contributions are two-fold: first, we propose a hierarchical agentic framework for autonomous drone control, and second, a reasoning methodology for individual function executions which we refer to as ReActEval. Our framework focuses on visual inspection tasks in indoor industrial settings, such as interpreting industrial readouts or inspecting equipment. It employs a multi-agent system comprising a head agent and multiple worker agents, each controlling a single drone. The head agent performs high-level planning and evaluates outcomes, while worker agents implement ReActEval to reason over and execute low-level actions. Operating entirely in natural language, ReActEval follows a plan, reason, act, evaluate cycle, enabling drones to handle tasks ranging from simple navigation (e.g., flying forward 10 meters and land) to complex high-level tasks (e.g., locating and reading a pressure gauge). The evaluation phase serves as a feedback and/or replanning stage, ensuring actions align with user objectives while preventing undesirable outcomes. We evaluate the framework in a simulated environment with two worker agents, assessing performance qualitatively and quantitatively based on task completion across varying complexity levels and workflow efficiency. By leveraging natural language processing for agent communication, our approach offers a novel, flexible, and user-accessible alternative to traditional drone-based solutions, enabling autonomous problem-solving for industrial inspection without extensive user intervention.

**arXiv ID:** 2510.00259
</details>

<details>
<summary><strong>MAVUL: Multi-Agent Vulnerability Detection via Contextual Reasoning and Interactive Refinement</strong> - Youpeng Li, Kartik Joshi, Xinda Wang, Eric Wong - [[pdf]](https://arxiv.org/pdf/2510.00317)</summary>

**Abstract:** The widespread adoption of open-source software (OSS) necessitates the mitigation of vulnerability risks. Most vulnerability detection (VD) methods are limited by inadequate contextual understanding, restrictive single-round interactions, and coarse-grained evaluations, resulting in undesired model performance and biased evaluation results. To address these challenges, we propose MAVUL, a novel multi-agent VD system that integrates contextual reasoning and interactive refinement. Specifically, a vulnerability analyst agent is designed to flexibly leverage tool-using capabilities and contextual reasoning to achieve cross-procedural code understanding and effectively mine vulnerability patterns. Through iterative feedback and refined decision-making within cross-role agent interactions, the system achieves reliable reasoning and vulnerability prediction. Furthermore, MAVUL introduces multi-dimensional ground truth information for fine-grained evaluation, thereby enhancing evaluation accuracy and reliability.
Extensive experiments conducted on a pairwise vulnerability dataset demonstrate MAVUL's superior performance. Our findings indicate that MAVUL significantly outperforms existing multi-agent systems with over 62% higher pairwise accuracy and single-agent systems with over 600% higher average performance. The system's effectiveness is markedly improved with increased communication rounds between the vulnerability analyst agent and the security architect agent, underscoring the importance of contextual reasoning in tracing vulnerability flows and the crucial feedback role. Additionally, the integrated evaluation agent serves as a critical, unbiased judge, ensuring a more accurate and reliable estimation of the system's real-world applicability by preventing misleading binary comparisons.

**arXiv ID:** 2510.00317
</details>

<details>
<summary><strong>Reasoning-Aware Prompt Orchestration: A Foundation Model for Multi-Agent Language Model Coordination</strong> - Hassen Dhrif - [[pdf]](https://arxiv.org/pdf/2510.00326)</summary>

**Abstract:** The emergence of large language models has enabled sophisticated multi-agent systems, yet coordinating their reasoning capabilities through prompt engineering remains challenging. We present a theoretically-grounded framework for dynamic prompt orchestration that enhances reasoning across multiple specialized agents. This framework addresses three core challenges: logical consistency preservation during agent transitions, reasoning-aware prompt adaptation, and scalable coordination of distributed inference.
Our approach formalizes agent states using prompt templates, reasoning context vectors, and capability matrices. We prove system convergence to stable coordination patterns when step sizes satisfy $\alpha < \frac{1}{2L}$ where $L$ is the Lipschitz constant of the state transition function. We implement this through a distributed architecture that dynamically routes reasoning tasks while maintaining semantic coherence.
Experimental results on 1,000 synthetic multi-agent conversations demonstrate a 42% reduction in reasoning latency, a 23% improvement in logical consistency measured by ROUGE-L score, and an 89% success rate for task completion without context loss across agent transitions. Ablation studies identify the consensus mechanism as the primary performance driver, while revealing limitations: performance degrades beyond 10 agent transitions, and the system requires 76.5GB memory for 1,000 concurrent agents. These findings establish a new paradigm for scalable reasoning in multi-agent systems, providing theoretical foundations for understanding reasoning emergence across coordinated language models.

**arXiv ID:** 2510.00326
</details>

<details>
<summary><strong>Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting</strong> - Shounak Sural, Charles Kekeh, Wenliang Liu, Federico Pecora, Mouhacine Benosman - [[pdf]](https://arxiv.org/pdf/2510.00401)</summary>

**Abstract:** Long-horizon motion forecasting for multiple autonomous robots is challenging due to non-linear agent interactions, compounding prediction errors, and continuous-time evolution of dynamics. Learned dynamics of such a system can be useful in various applications such as travel time prediction, prediction-guided planning and generative simulation. In this work, we aim to develop an efficient trajectory forecasting model conditioned on multi-agent goals. Motivated by the recent success of physics-guided deep learning for partially known dynamical systems, we develop a model based on neural Controlled Differential Equations (CDEs) for long-horizon motion forecasting. Unlike discrete-time methods such as RNNs and transformers, neural CDEs operate in continuous time, allowing us to combine physics-informed constraints and biases to jointly model multi-robot dynamics. Our approach, named PINCoDE (Physics-Informed Neural Controlled Differential Equations), learns differential equation parameters that can be used to predict the trajectories of a multi-agent system starting from an initial condition. PINCoDE is conditioned on future goals and enforces physics constraints for robot motion over extended periods of time. We adopt a strategy that scales our model from 10 robots to 100 robots without the need for additional model parameters, while producing predictions with an average ADE below 0.5 m for a 1-minute horizon. Furthermore, progressive training with curriculum learning for our PINCoDE model results in a 2.7X reduction of forecasted pose error over 4 minute horizons compared to analytical models.

**arXiv ID:** 2510.00401
</details>

<details>
<summary><strong>Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs</strong> - Yurun Chen, Xavier Hu, Yuhan Liu, Ziqi Wang, Zeyi Liao, Lin Chen, Feng Wei, Yuxi Qian, Bo Zheng, Keting Yin, Shengyu Zhang - [[pdf]](https://arxiv.org/pdf/2510.00507)</summary>

**Abstract:** As multimodal LLM-driven agents continue to advance in autonomy and generalization, evaluation based on static datasets can no longer adequately assess their true capabilities in dynamic environments and diverse tasks. Existing LLM-based synthetic data methods are largely designed for LLM training and evaluation, and thus cannot be directly applied to agent tasks that require tool use and interactive capabilities. While recent studies have explored automatic agent task generation with LLMs, most efforts remain limited to text or image analysis, without systematically modeling multi-step interactions in web environments. To address these challenges, we propose Graph2Eval, a knowledge graph-based framework that automatically generates both multimodal document comprehension tasks and web interaction tasks, enabling comprehensive evaluation of agents' reasoning, collaboration, and interactive capabilities. In our approach, knowledge graphs constructed from multi-source external data serve as the task space, where we translate semantic relations into structured multimodal tasks using subgraph sampling, task templates, and meta-paths. A multi-stage filtering pipeline based on node reachability, LLM scoring, and similarity analysis is applied to guarantee the quality and executability of the generated tasks. Furthermore, Graph2Eval supports end-to-end evaluation of multiple agent types (Single-Agent, Multi-Agent, Web Agent) and measures reasoning, collaboration, and interaction capabilities. We instantiate the framework with Graph2Eval-Bench, a curated dataset of 1,319 tasks spanning document comprehension and web interaction scenarios. Experiments show that Graph2Eval efficiently generates tasks that differentiate agent and model performance, revealing gaps in reasoning, collaboration, and web interaction across different settings and offering a new perspective for agent evaluation.

**arXiv ID:** 2510.00507
</details>

<details>
<summary><strong>Conflict-Based Search as a Protocol: A Multi-Agent Motion Planning Protocol for Heterogeneous Agents, Solvers, and Independent Tasks</strong> - Rishi Veerapaneni, Alvin Tang, Haodong He, Sophia Zhao, Viraj Shah, Yidai Cen, Ziteng Ji, Gabriel Olin, Jon Arrizabalaga, Yorai Shaoul, Jiaoyang Li, Maxim Likhachev - [[pdf]](https://arxiv.org/pdf/2510.00425)</summary>

**Abstract:** Imagine the future construction site, hospital, office, or even sophisticated household with dozens of robots bought from different manufacturers. How can we enable these different systems to effectively move in a shared environment, given that each robot may have its own independent motion planning system? This work shows how we can get efficient collision-free movements between algorithmically heterogeneous agents by using Conflict-Based Search (Sharon et al. 2015) as a protocol. At its core, the CBS Protocol requires one specific single-agent motion planning API; finding a collision-free path that satisfies certain space-time constraints. Given such an API, CBS uses a central planner to find collision-free paths - independent of how the API is implemented. We show how this protocol enables multi-agent motion planning for a heterogeneous team of agents completing independent tasks with a variety of single-agent planners including: Heuristic Search (e.g., A*), Sampling Based Search (e.g., RRT), Optimization (e.g., Direct Collocation), Diffusion, and Reinforcement Learning.

**arXiv ID:** 2510.00425
</details>

<details>
<summary><strong>Stochastic Self-Organization in Multi-Agent Systems</strong> - Nurbek Tastan, Samuel Horvath, Karthik Nandakumar - [[pdf]](https://arxiv.org/pdf/2510.00685)</summary>

**Abstract:** Multi-agent systems (MAS) based on Large Language Models (LLMs) have the potential to solve tasks that are beyond the reach of any single LLM. However, this potential can only be realized when the collaboration mechanism between agents is optimized. Specifically, optimizing the communication structure between agents is critical for fruitful collaboration. Most existing approaches rely on fixed topologies, pretrained graph generators, optimization over edges, or employ external LLM judges, thereby adding to the complexity. In this work, we introduce a response-conditioned framework that adapts communication on-the-fly. Agents independently generate responses to the user query and assess peer contributions using an approximation of the Shapley value. A directed acyclic graph (DAG) is then constructed to regulate the propagation of the responses among agents, which ensures stable and efficient message transmission from high-contributing agents to others. This graph is dynamically updated based on the agent responses from the previous collaboration round. Since the proposed framework enables the self-organization of agents without additional supervision or training, we refer to it as SelfOrg. The SelfOrg framework goes beyond task- and query-level optimization and takes into account the stochastic nature of agent responses. Experiments with both strong and weak LLM backends demonstrate robust performance, with significant gains in the weak regime where prior methods collapse. We also theoretically show that multiple agents increase the chance of correctness and that the correct responses naturally dominate the information flow.

**arXiv ID:** 2510.00685
</details>

<details>
<summary><strong>The challenge of hidden gifts in multi-agent reinforcement learning</strong> - Dane Malenfant, Blake A. Richards - [[pdf]](https://arxiv.org/pdf/2505.20579)</summary>

**Abstract:** Sometimes we benefit from actions that others have taken even when we are unaware that they took those actions. For example, if your neighbor chooses not to take a parking spot in front of your house when you are not there, you can benefit, even without being aware that they took this action. These ``hidden gifts'' represent an interesting challenge for multi-agent reinforcement learning (MARL), since assigning credit when the beneficial actions of others are hidden is non-trivial. Here, we study the impact of hidden gifts with a very simple MARL task. In this task, agents in a grid-world environment have individual doors to unlock in order to obtain individual rewards. As well, if all the agents unlock their door the group receives a larger collective reward. However, there is only one key for all of the doors, such that the collective reward can only be obtained when the agents drop the key for others after they use it. Notably, there is nothing to indicate to an agent that the other agents have dropped the key, thus this act for others is a ``hidden gift''. We show that several different state-of-the-art MARL algorithms, including MARL specific architectures, fail to learn how to obtain the collective reward in this simple task. Interestingly, we find that decentralized actor-critic policy gradient agents can succeed when we provide them with information about their own action history, but MARL agents still cannot solve the task with action history. Finally, we derive a correction term for policy gradient agents, inspired by learning aware approaches, which reduces the variance in learning and helps them to converge to collective success more reliably. These results show that credit assignment in multi-agent settings can be particularly challenging in the presence of ``hidden gifts'', and demonstrate that self learning-awareness in decentralized agents can benefit these settings.

**arXiv ID:** 2505.20579
</details>

<details>
<summary><strong>Code Like Humans: A Multi-Agent Solution for Medical Coding</strong> - Andreas Motzfeldt, Joakim Edin, Casper L. Christensen, Christian Hardmeier, Lars Maaløe, Anna Rogers - [[pdf]](https://arxiv.org/pdf/2509.05378)</summary>

**Abstract:** In medical coding, experts map unstructured clinical notes to alphanumeric codes for diagnoses and procedures. We introduce Code Like Humans: a new agentic framework for medical coding with large language models. It implements official coding guidelines for human experts, and it is the first solution that can support the full ICD-10 coding system (+70K labels). It achieves the best performance to date on rare diagnosis codes (fine-tuned discriminative classifiers retain an advantage for high-frequency codes, to which they are limited). Towards future work, we also contribute an analysis of system performance and identify its `blind spots' (codes that are systematically undercoded).

**arXiv ID:** 2509.05378
</details>

<details>
<summary><strong>CORTEX: Collaborative LLM Agents for High-Stakes Alert Triage</strong> - Bowen Wei, Yuan Shen Tay, Howard Liu, Jinhao Pan, Kun Luo, Ziwei Zhu, Chris Jordan - [[pdf]](https://arxiv.org/pdf/2510.00311)</summary>

**Abstract:** Security Operations Centers (SOCs) are overwhelmed by tens of thousands of daily alerts, with only a small fraction corresponding to genuine attacks. This overload creates alert fatigue, leading to overlooked threats and analyst burnout. Classical detection pipelines are brittle and context-poor, while recent LLM-based approaches typically rely on a single model to interpret logs, retrieve context, and adjudicate alerts end-to-end -- an approach that struggles with noisy enterprise data and offers limited transparency. We propose CORTEX, a multi-agent LLM architecture for high-stakes alert triage in which specialized agents collaborate over real evidence: a behavior-analysis agent inspects activity sequences, evidence-gathering agents query external systems, and a reasoning agent synthesizes findings into an auditable decision. To support training and evaluation, we release a dataset of fine-grained SOC investigations from production environments, capturing step-by-step analyst actions and linked tool outputs. Across diverse enterprise scenarios, CORTEX substantially reduces false positives and improves investigation quality over state-of-the-art single-agent LLMs.

**arXiv ID:** 2510.00311
</details>

<details>
<summary><strong>JoyAgent-JDGenie: Technical Report on the GAIA</strong> - Jiarun Liu, Shiyue Xu, Shangkun Liu, Yang Li, Wen Liu, Min Liu, Xiaoqing Zhou, Hanmin Wang, Shilin Jia, zhen Wang, Shaohua Tian, Hanhao Li, Junbo Zhang, Yongli Yu, Peng Cao, Haofen Wang - [[pdf]](https://arxiv.org/pdf/2510.00510)</summary>

**Abstract:** Large Language Models are increasingly deployed as autonomous agents for complex real-world tasks, yet existing systems often focus on isolated improvements without a unifying design for robustness and adaptability. We propose a generalist agent architecture that integrates three core components: a collective multi-agent framework combining planning and execution agents with critic model voting, a hierarchical memory system spanning working, semantic, and procedural layers, and a refined tool suite for search, code execution, and multimodal parsing. Evaluated on a comprehensive benchmark, our framework consistently outperforms open-source baselines and approaches the performance of proprietary systems. These results demonstrate the importance of system-level integration and highlight a path toward scalable, resilient, and adaptive AI assistants capable of operating across diverse domains and tasks.

**arXiv ID:** 2510.00510
</details>

<details>
<summary><strong>Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions</strong> - Mohammad Almansoori, Komal Kumar, Hisham Cholakkal - [[pdf]](https://arxiv.org/pdf/2503.22678)</summary>

**Abstract:** In this work, we introduce MedAgentSim, an open-source simulated clinical environment with doctor, patient, and measurement agents designed to evaluate and enhance LLM performance in dynamic diagnostic settings. Unlike prior approaches, our framework requires doctor agents to actively engage with patients through multi-turn conversations, requesting relevant medical examinations (e.g., temperature, blood pressure, ECG) and imaging results (e.g., MRI, X-ray) from a measurement agent to mimic the real-world diagnostic process. Additionally, we incorporate self improvement mechanisms that allow models to iteratively refine their diagnostic strategies. We enhance LLM performance in our simulated setting by integrating multi-agent discussions, chain-of-thought reasoning, and experience-based knowledge retrieval, facilitating progressive learning as doctor agents interact with more patients. We also introduce an evaluation benchmark for assessing the LLM's ability to engage in dynamic, context-aware diagnostic interactions. While MedAgentSim is fully automated, it also supports a user-controlled mode, enabling human interaction with either the doctor or patient agent. Comprehensive evaluations in various simulated diagnostic scenarios demonstrate the effectiveness of our approach. Our code, simulation tool, and benchmark are available at \href{this https URL}.

**arXiv ID:** 2503.22678
</details>

<details>
<summary><strong>Multi-Agent Stage-wise Conservative Linear Bandits</strong> - Amirhoseein Afsharrad, Ahmadreza Moradipari, Sanjay Lall - [[pdf]](https://arxiv.org/pdf/2510.00602)</summary>

**Abstract:** In many real-world applications such as recommendation systems, multiple learning agents must balance exploration and exploitation while maintaining safety guarantees to avoid catastrophic failures. We study the stochastic linear bandit problem in a multi-agent networked setting where agents must satisfy stage-wise conservative constraints. A network of $N$ agents collaboratively maximizes cumulative reward while ensuring that the expected reward at every round is no less than $(1-\alpha)$ times that of a baseline policy. Each agent observes local rewards with unknown parameters, but the network optimizes for the global parameter (average of local parameters). Agents communicate only with immediate neighbors, and each communication round incurs additional regret. We propose MA-SCLUCB (Multi-Agent Stage-wise Conservative Linear UCB), an episodic algorithm alternating between action selection and consensus-building phases. We prove that MA-SCLUCB achieves regret $\tilde{O}\left(\frac{d}{\sqrt{N}}\sqrt{T}\cdot\frac{\log(NT)}{\sqrt{\log(1/|\lambda_2|)}}\right)$ with high probability, where $d$ is the dimension, $T$ is the horizon, and $|\lambda_2|$ is the network's second largest eigenvalue magnitude. Our analysis shows: (i) collaboration yields $\frac{1}{\sqrt{N}}$ improvement despite local communication, (ii) communication overhead grows only logarithmically for well-connected networks, and (iii) stage-wise safety adds only lower-order regret. Thus, distributed learning with safety guarantees achieves near-optimal performance in reasonably connected networks.

**arXiv ID:** 2510.00602
</details>

<details>
<summary><strong>DBF-MA: A Differential Bayesian Filtering Planner for Multi-Agent Autonomous Racing Overtakes</strong> - Trent Weiss, Amar Kulkarni, Madhur Behl - [[pdf]](https://arxiv.org/pdf/2509.22937)</summary>

**Abstract:** A significant challenge in autonomous racing is to generate overtaking maneuvers. Racing agents must execute these maneuvers on complex racetracks with little room for error. Optimization techniques and graph-based methods have been proposed, but these methods often rely on oversimplified assumptions for collision-avoidance and dynamic constraints. In this work, we present an approach to trajectory synthesis based on an extension of the Differential Bayesian Filtering framework. Our approach for collision-free trajectory synthesis frames the problem as one of Bayesian Inference over the space of Composite Bezier Curves. Our method is derivative-free, does not require a spherical approximation of the vehicle footprint, linearization of constraints, or simplifying upper bounds on collision avoidance. We conduct a closed-loop analysis of DBF-MA and find it successfully overtakes an opponent in 87% of tested scenarios, outperforming existing methods in autonomous overtaking.

**arXiv ID:** 2509.22937
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>Exploring Network-Knowledge Graph Duality: A Case Study in Agentic Supply Chain Risk Analysis</strong> - Evan Heus, Rick Bookstaber, Dhruv Sharma - [[pdf]](https://arxiv.org/pdf/2510.01115)</summary>

**Abstract:** Large Language Models (LLMs) struggle with the complex, multi-modal, and network-native data underlying financial risk. Standard Retrieval-Augmented Generation (RAG) oversimplifies relationships, while specialist models are costly and static. We address this gap with an LLM-centric agent framework for supply chain risk analysis. Our core contribution is to exploit the inherent duality between networks and knowledge graphs (KG). We treat the supply chain network as a KG, allowing us to use structural network science principles for retrieval. A graph traverser, guided by network centrality scores, efficiently extracts the most economically salient risk paths. An agentic architecture orchestrates this graph retrieval alongside data from numerical factor tables and news streams. Crucially, it employs novel ``context shells'' -- descriptive templates that embed raw figures in natural language -- to make quantitative data fully intelligible to the LLM. This lightweight approach enables the model to generate concise, explainable, and context-rich risk narratives in real-time without costly fine-tuning or a dedicated graph database.

**arXiv ID:** 2510.01115
</details>

<details>
<summary><strong>CHAI: Command Hijacking against embodied AI</strong> - Luis Burbano, Diego Ortiz, Qi Sun, Siwei Yang, Haoqin Tu, Cihang Xie, Yinzhi Cao, Alvaro A Cardenas - [[pdf]](https://arxiv.org/pdf/2510.00181)</summary>

**Abstract:** Embodied Artificial Intelligence (AI) promises to handle edge cases in robotic vehicle systems where data is scarce by using common-sense reasoning grounded in perception and action to generalize beyond training distributions and adapt to novel real-world situations. These capabilities, however, also create new security risks. In this paper, we introduce CHAI (Command Hijacking against embodied AI), a new class of prompt-based attacks that exploit the multimodal language interpretation abilities of Large Visual-Language Models (LVLMs). CHAI embeds deceptive natural language instructions, such as misleading signs, in visual input, systematically searches the token space, builds a dictionary of prompts, and guides an attacker model to generate Visual Attack Prompts. We evaluate CHAI on four LVLM agents; drone emergency landing, autonomous driving, and aerial object tracking, and on a real robotic vehicle. Our experiments show that CHAI consistently outperforms state-of-the-art attacks. By exploiting the semantic and multimodal reasoning strengths of next-generation embodied AI systems, CHAI underscores the urgent need for defenses that extend beyond traditional adversarial robustness.

**arXiv ID:** 2510.00181
</details>

<details>
<summary><strong>Exploring and Controlling Diversity in LLM-Agent Conversation</strong> - KuanChao Chu, Yi-Pei Chen, Hideki Nakayama - [[pdf]](https://arxiv.org/pdf/2412.21102)</summary>

**Abstract:** Controlling diversity in LLM-agent simulations is essential for balancing stability in structured tasks with variability in open-ended interactions. However, we observe that dialogue diversity tends to degrade over long-term simulations. To explore the role of prompt design in this phenomenon, we modularized the utterance generation prompt and found that reducing contextual information leads to more diverse outputs. Based on this insight, we propose Adaptive Prompt Pruning (APP), a novel method that allows users to control diversity via a single parameter, lambda. APP dynamically prunes prompt segments based on attention scores and is compatible with existing diversity control methods. We demonstrate that APP effectively modulates diversity through extensive experiments and propose a method to balance the control trade-offs. Our analysis reveals that all prompt components impose constraints on diversity, with the Memory being the most influential. Additionally, high-attention contents consistently suppress output diversity.

**arXiv ID:** 2412.21102
</details>

<details>
<summary><strong>The Formation of Trust in Autonomous Vehicles after Interacting with Robotaxis on Public Roads</strong> - Xiang Chang, Zhijie Yi, Yichang Liu, Hongling Sheng, Dengbo He - [[pdf]](https://arxiv.org/pdf/2510.00120)</summary>

**Abstract:** This study investigates how pedestrian trust, receptivity, and behavior evolve during interactions with Level-4 autonomous vehicles (AVs) at uncontrolled urban intersections in a naturalistic setting. While public acceptance is critical for AV adoption, most prior studies relied on simplified simulations or field tests. We conducted a real-world experiment in a commercial Robotaxi operation zone, where 33 participants repeatedly crossed an uncontrolled intersection with frequent Level-4 Robotaxi traffic. Participants completed the Pedestrian Behavior Questionnaire (PBQ), Pedestrian Receptivity Questionnaire for Fully AVs (PRQF), pre- and post-experiment Trust in AVs Scale, and Personal Innovativeness Scale (PIS). Results showed that trust in AVs significantly increased post-experiment, with the increase positively associated with the Interaction component of PRQF. Additionally, both the Positive and Error subscales of the PBQ significantly influenced trust change. This study reveals how trust forms in real-world pedestrian-AV encounters, offering insights beyond lab-based research by accounting for population heterogeneity.

**arXiv ID:** 2510.00120
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (37 papers)</h2></summary>

<details>
<summary><strong>ToolBrain: A Flexible Reinforcement Learning Framework for Agentic Tools</strong> - Quy Minh Le, Minh Sao Khue Luu, Khanh-Tung Tran, Duc-Hai Nguyen, Hoang-Quoc-Viet Pham, Quan Le, Hoang Thanh Lam, Hoang D. Nguyen - [[pdf]](https://arxiv.org/pdf/2510.00023)</summary>

**Abstract:** Effective tool use is essential for agentic AI, yet training agents to utilize tools remains challenging due to manually designed rewards, limited training data, and poor multi-tool selection, resulting in slow adaptation, wasted computational resources, and suboptimal performance. We introduce ToolBrain, a lightweight and user-friendly framework for coaching tool use in agentic models with flexible reinforcement learning (RL), easing the barriers for researchers and practitioners to adapt LLM-based agents to specific domains. It supports a wide range of training strategies, including RL algorithms such as GRPO and DPO, as well as supervised learning. ToolBrain enables custom reward callables directly on an agent's execution traces or simply utilizes an automated LLM-as-a-judge system for reward generation. It is packed with useful capabilities, including knowledge distillation from large to small models for efficient development, automatic task generation from tool descriptions, seamless tool retrieval, efficient fine-tuning pipelines with QLoRA through Unsloth, and quantized inference via bitsandbytes. We demonstrate ToolBrain through diverse use cases, such as training a CodeAct agent to autonomously execute email search tasks, showing fast, targeted improvements (up to 30.0%) in tool-use skills while keeping the codebase simple and extensible in Agentic AI. Our framework is publicly available at this https URL.

**arXiv ID:** 2510.00023
</details>

<details>
<summary><strong>When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets</strong> - Zeshi Dai, Zimo Peng, Zerui Cheng, Ryan Yihe Li - [[pdf]](https://arxiv.org/pdf/2510.00332)</summary>

**Abstract:** We present CAIA, a benchmark exposing a critical blind spot in AI evaluation: the inability of state-of-the-art models to operate in adversarial, high-stakes environments where misinformation is weaponized and errors are irreversible. While existing benchmarks measure task completion in controlled settings, real-world deployment demands resilience against active deception. Using crypto markets as a testbed where $30 billion was lost to exploits in 2024, we evaluate 17 models on 178 time-anchored tasks requiring agents to distinguish truth from manipulation, navigate fragmented information landscapes, and make irreversible financial decisions under adversarial pressure.
Our results reveal a fundamental capability gap: without tools, even frontier models achieve only 28% accuracy on tasks junior analysts routinely handle. Tool augmentation improves performance but plateaus at 67.4% versus 80% human baseline, despite unlimited access to professional resources. Most critically, we uncover a systematic tool selection catastrophe: models preferentially choose unreliable web search over authoritative data, falling for SEO-optimized misinformation and social media manipulation. This behavior persists even when correct answers are directly accessible through specialized tools, suggesting foundational limitations rather than knowledge gaps. We also find that Pass@k metrics mask dangerous trial-and-error behavior for autonomous deployment.
The implications extend beyond crypto to any domain with active adversaries, e.g. cybersecurity, content moderation, etc. We release CAIA with contamination controls and continuous updates, establishing adversarial robustness as a necessary condition for trustworthy AI autonomy. The benchmark reveals that current models, despite impressive reasoning scores, remain fundamentally unprepared for environments where intelligence must survive active opposition.

**arXiv ID:** 2510.00332
</details>

<details>
<summary><strong>QUASAR: Quantum Assembly Code Generation Using Tool-Augmented LLMs via Agentic RL</strong> - Cong Yu, Valter Uotila, Shilong Deng, Qingyuan Wu, Tuo Shi, Songlin Jiang, Lei You, Bo Zhao - [[pdf]](https://arxiv.org/pdf/2510.00967)</summary>

**Abstract:** Designing and optimizing task-specific quantum circuits are crucial to leverage the advantage of quantum computing. Recent large language model (LLM)-based quantum circuit generation has emerged as a promising automatic solution. However, the fundamental challenges remain unaddressed: (i) parameterized quantum gates require precise numerical values for optimal performance, which also depend on multiple aspects, including the number of quantum gates, their parameters, and the layout/depth of the circuits. (ii) LLMs often generate low-quality or incorrect quantum circuits due to the lack of quantum domain-specific knowledge. We propose QUASAR, an agentic reinforcement learning (RL) framework for quantum circuits generation and optimization based on tool-augmented LLMs. To align the LLM with quantum-specific knowledge and improve the generated quantum circuits, QUASAR designs (i) a quantum circuit verification approach with external quantum simulators and (ii) a sophisticated hierarchical reward mechanism in RL training. Extensive evaluation shows improvements in both syntax and semantic performance of the generated quantum circuits. When augmenting a 4B LLM, QUASAR has achieved the validity of 99.31% in Pass@1 and 100% in Pass@10, outperforming industrial LLMs of GPT-4o, GPT-5 and DeepSeek-V3 and several supervised-fine-tuning (SFT)-only and RL-only baselines.

**arXiv ID:** 2510.00967
</details>

<details>
<summary><strong>Autonomous Multi-Robot Infrastructure for AI-Enabled Healthcare Delivery and Diagnostics</strong> - Nakhul Kalaivanan, Senthil Arumugam Muthukumaraswamy, Girish Balasubramanian - [[pdf]](https://arxiv.org/pdf/2509.26106)</summary>

**Abstract:** This research presents a multi-robot system for inpatient care, designed using swarm intelligence principles and incorporating wearable health sensors, RF-based communication, and AI-driven decision support. Within a simulated hospital environment, the system adopts a leader-follower swarm configuration to perform patient monitoring, medicine delivery, and emergency assistance. Due to ethical constraints, live patient trials were not conducted; instead, validation was carried out through controlled self-testing with wearable sensors. The Leader Robot acquires key physiological parameters, including temperature, SpO2, heart rate, and fall detection, and coordinates other robots when required. The Assistant Robot patrols corridors for medicine delivery, while a robotic arm provides direct drug administration. The swarm-inspired leader-follower strategy enhanced communication reliability and ensured continuous monitoring, including automated email alerts to healthcare staff. The system hardware was implemented using Arduino, Raspberry Pi, NRF24L01 RF modules, and a HuskyLens AI camera. Experimental evaluation showed an overall sensor accuracy above 94%, a 92% task-level success rate, and a 96% communication reliability rate, demonstrating system robustness. Furthermore, the AI-enabled decision support was able to provide early warnings of abnormal health conditions, highlighting the potential of the system as a cost-effective solution for hospital automation and patient safety.

**arXiv ID:** 2509.26106
</details>

<details>
<summary><strong>Reinforcement Learning-Based Prompt Template Stealing for Text-to-Image Models</strong> - Xiaotian Zou - [[pdf]](https://arxiv.org/pdf/2510.00046)</summary>

**Abstract:** Multimodal Large Language Models (MLLMs) have transformed text-to-image workflows, allowing designers to create novel visual concepts with unprecedented speed. This progress has given rise to a thriving prompt trading market, where curated prompts that induce trademark styles are bought and sold. Although commercially attractive, prompt trading also introduces a largely unexamined security risk: the prompts themselves can be stolen.
In this paper, we expose this vulnerability and present RLStealer, a reinforcement learning based prompt inversion framework that recovers its template from only a small set of example images. RLStealer treats template stealing as a sequential decision making problem and employs multiple similarity based feedback signals as reward functions to effectively explore the prompt space. Comprehensive experiments on publicly available benchmarks demonstrate that RLStealer gets state-of-the-art performance while reducing the total attack cost to under 13% of that required by existing baselines. Our further analysis confirms that RLStealer can effectively generalize across different image styles to efficiently steal unseen prompt templates. Our study highlights an urgent security threat inherent in prompt trading and lays the groundwork for developing protective standards in the emerging MLLMs marketplace.

**arXiv ID:** 2510.00046
</details>

<details>
<summary><strong>Intelligent 5S Audit: Application of Artificial Intelligence for Continuous Improvement in the Automotive Industry</strong> - Rafael da Silva Maciel, Lucio Veraldo Jr - [[pdf]](https://arxiv.org/pdf/2510.00067)</summary>

**Abstract:** The evolution of the 5S methodology with the support of artificial intelligence techniques represents a significant opportunity to improve industrial organization audits in the automotive chain, making them more objective, efficient and aligned with Industry 4.0 standards. This work developed an automated 5S audit system based on large-scale language models (LLM), capable of assessing the five senses (Seiri, Seiton, Seiso, Seiketsu, Shitsuke) in a standardized way through intelligent image analysis. The system's reliability was validated using Cohen's concordance coefficient (kappa = 0.75), showing strong alignment between the automated assessments and the corresponding human audits. The results indicate that the proposed solution contributes significantly to continuous improvement in automotive manufacturing environments, speeding up the audit process by 50% of the traditional time and maintaining the consistency of the assessments, with a 99.8% reduction in operating costs compared to traditional manual audits. The methodology presented establishes a new paradigm for integrating lean systems with emerging AI technologies, offering scalability for implementation in automotive plants of different sizes.

**arXiv ID:** 2510.00067
</details>

<details>
<summary><strong>Geo-R1: Unlocking VLM Geospatial Reasoning with Cross-View Reinforcement Learning</strong> - Chenhui Xu, Fuxun Yu, Michael J. Bianco, Jacob Kovarskiy, Raphael Tang, Qi Zhang, Zirui Xu, Will LeVine, Brandon Dubbs, Heming Liao, Cassandra Burgess, Suvam Bag, Jay Patravali, Rupanjali Kukal, Mikael Figueroa, Rishi Madhok, Nikolaos Karianakis, Jinjun Xiong - [[pdf]](https://arxiv.org/pdf/2510.00072)</summary>

**Abstract:** We introduce Geo-R1, a reasoning-centric post-training framework that unlocks geospatial reasoning in vision-language models by combining thinking scaffolding and elevating. In the scaffolding stage, Geo-R1 instills a ``geospatial thinking paradigm" via supervised fine-tuning on synthetic chain-of-thought exemplars, enabling models to connect visual cues with geographic priors without costly human reasoning annotations. In the elevating stage, it uses GRPO-based reinforcement learning on a weakly-supervised cross-view pairing proxy. This design supplies a verifiable and scalable reward signal: teaching models to capture and reconcile features across modalities, and harnessing reasoning for accurate prediction. Geo-R1 extends geospatial modeling from domain pretraining / supervised finetuning to reasoning-first post-training, and achieves state-of-the-art performance across various geospatial reasoning benchmarks. Our model is available at this https URL.

**arXiv ID:** 2510.00072
</details>

<details>
<summary><strong>Which Rewards Matter? Reward Selection for Reinforcement Learning under Limited Feedback</strong> - Shreyas Chaudhari, Renhao Zhang, Philip S. Thomas, Bruno Castro da Silva - [[pdf]](https://arxiv.org/pdf/2510.00144)</summary>

**Abstract:** The ability of reinforcement learning algorithms to learn effective policies is determined by the rewards available during training. However, for practical problems, obtaining large quantities of reward labels is often infeasible due to computational or financial constraints, particularly when relying on human feedback. When reinforcement learning must proceed with limited feedback -- only a fraction of samples get rewards labeled -- a fundamental question arises: which samples should be labeled to maximize policy performance? We formalize this problem of reward selection for reinforcement learning from limited feedback (RLLF), introducing a new problem formulation that facilitates the study of strategies for selecting impactful rewards. Two types of selection strategies are investigated: (i) heuristics that rely on reward-free information such as state visitation and partial value functions, and (ii) strategies pre-trained using auxiliary evaluative feedback. We find that critical subsets of rewards are those that (1) guide the agent along optimal trajectories, and (2) support recovery toward near-optimal behavior after deviations. Effective selection methods yield near-optimal policies with significantly fewer reward labels than full supervision, establishing reward selection as a powerful paradigm for scaling reinforcement learning in feedback-limited settings.

**arXiv ID:** 2510.00144
</details>

<details>
<summary><strong>Directed-MAML: Meta Reinforcement Learning Algorithm with Task-directed Approximation</strong> - Yang Zhang, Huiwen Yan, Mushuang Liu - [[pdf]](https://arxiv.org/pdf/2510.00212)</summary>

**Abstract:** Model-Agnostic Meta-Learning (MAML) is a versatile meta-learning framework applicable to both supervised learning and reinforcement learning (RL). However, applying MAML to meta-reinforcement learning (meta-RL) presents notable challenges. First, MAML relies on second-order gradient computations, leading to significant computational and memory overhead. Second, the nested structure of optimization increases the problem's complexity, making convergence to a global optimum more challenging. To overcome these limitations, we propose Directed-MAML, a novel task-directed meta-RL algorithm. Before the second-order gradient step, Directed-MAML applies an additional first-order task-directed approximation to estimate the effect of second-order gradients, thereby accelerating convergence to the optimum and reducing computational cost. Experimental results demonstrate that Directed-MAML surpasses MAML-based baselines in computational efficiency and convergence speed in the scenarios of CartPole-v1, LunarLander-v2 and two-vehicle intersection crossing. Furthermore, we show that task-directed approximation can be effectively integrated into other meta-learning algorithms, such as First-Order Model-Agnostic Meta-Learning (FOMAML) and Meta Stochastic Gradient Descent(Meta-SGD), yielding improved computational efficiency and convergence speed.

**arXiv ID:** 2510.00212
</details>

<details>
<summary><strong>Can AI agents understand spoken conversations about data visualizations in online meetings?</strong> - Rizul Sharma, Tianyu Jiang, Seokki Lee, Jillian Aurisano - [[pdf]](https://arxiv.org/pdf/2510.00245)</summary>

**Abstract:** In this short paper, we present work evaluating an AI agent's understanding of spoken conversations about data visualizations in an online meeting scenario. There is growing interest in the development of AI-assistants that support meetings, such as by providing assistance with tasks or summarizing a discussion. The quality of this support depends on a model that understands the conversational dialogue. To evaluate this understanding, we introduce a dual-axis testing framework for diagnosing the AI agent's comprehension of spoken conversations about data. Using this framework, we designed a series of tests to evaluate understanding of a novel corpus of 72 spoken conversational dialogues about data visualizations. We examine diverse pipelines and model architectures, LLM vs VLM, and diverse input formats for visualizations (the chart image, its underlying source code, or a hybrid of both) to see how this affects model performance on our tests. Using our evaluation methods, we found that text-only input modalities achieved the best performance (96%) in understanding discussions of visualizations in online meetings.

**arXiv ID:** 2510.00245
</details>

<details>
<summary><strong>DiSA-IQL: Offline Reinforcement Learning for Robust Soft Robot Control under Distribution Shifts</strong> - Linjin He, Xinda Qi, Dong Chen, Zhaojian Li, Xiaobo Tan - [[pdf]](https://arxiv.org/pdf/2510.00358)</summary>

**Abstract:** Soft snake robots offer remarkable flexibility and adaptability in complex environments, yet their control remains challenging due to highly nonlinear dynamics. Existing model-based and bio-inspired controllers rely on simplified assumptions that limit performance. Deep reinforcement learning (DRL) has recently emerged as a promising alternative, but online training is often impractical because of costly and potentially damaging real-world interactions. Offline RL provides a safer option by leveraging pre-collected datasets, but it suffers from distribution shift, which degrades generalization to unseen scenarios. To overcome this challenge, we propose DiSA-IQL (Distribution-Shift-Aware Implicit Q-Learning), an extension of IQL that incorporates robustness modulation by penalizing unreliable state-action pairs to mitigate distribution shift. We evaluate DiSA-IQL on goal-reaching tasks across two settings: in-distribution and out-of-distribution evaluation. Simulation results show that DiSA-IQL consistently outperforms baseline models, including Behavior Cloning (BC), Conservative Q-Learning (CQL), and vanilla IQL, achieving higher success rates, smoother trajectories, and improved robustness. The codes are open-sourced to support reproducibility and to facilitate further research in offline RL for soft robot control.

**arXiv ID:** 2510.00358
</details>

<details>
<summary><strong>Integrating Offline Pre-Training with Online Fine-Tuning: A Reinforcement Learning Approach for Robot Social Navigation</strong> - Run Su, Hao Fu, Shuai Zhou, Yingao Fu - [[pdf]](https://arxiv.org/pdf/2510.00466)</summary>

**Abstract:** Offline reinforcement learning (RL) has emerged as a promising framework for addressing robot social navigation challenges. However, inherent uncertainties in pedestrian behavior and limited environmental interaction during training often lead to suboptimal exploration and distributional shifts between offline training and online deployment. To overcome these limitations, this paper proposes a novel offline-to-online fine-tuning RL algorithm for robot social navigation by integrating Return-to-Go (RTG) prediction into a causal Transformer architecture. Our algorithm features a spatiotem-poral fusion model designed to precisely estimate RTG values in real-time by jointly encoding temporal pedestrian motion patterns and spatial crowd dynamics. This RTG prediction framework mitigates distribution shift by aligning offline policy training with online environmental interactions. Furthermore, a hybrid offline-online experience sampling mechanism is built to stabilize policy updates during fine-tuning, ensuring balanced integration of pre-trained knowledge and real-time adaptation. Extensive experiments in simulated social navigation environments demonstrate that our method achieves a higher success rate and lower collision rate compared to state-of-the-art baselines. These results underscore the efficacy of our algorithm in enhancing navigation policy robustness and adaptability. This work paves the way for more reliable and adaptive robotic navigation systems in real-world applications.

**arXiv ID:** 2510.00466
</details>

<details>
<summary><strong>On Predictability of Reinforcement Learning Dynamics for Large Language Models</strong> - Yuchen Cai, Ding Cao, Xin Xu, Zijun Yao, Yuqing Huang, Zhenyu Tan, Benyi Zhang, Guiquan Liu, Junfeng Fang - [[pdf]](https://arxiv.org/pdf/2510.00553)</summary>

**Abstract:** Recent advances in reasoning capabilities of large language models (LLMs) are largely driven by reinforcement learning (RL), yet the underlying parameter dynamics during RL training remain poorly understood. This work identifies two fundamental properties of RL-induced parameter updates in LLMs: (1) Rank-1 Dominance, where the top singular subspace of the parameter update matrix nearly fully determines reasoning improvements, recovering over 99\% of performance gains; and (2) Rank-1 Linear Dynamics, where this dominant subspace evolves linearly throughout training, enabling accurate prediction from early checkpoints. Extensive experiments across 8 LLMs and 7 algorithms validate the generalizability of these properties. More importantly, based on these findings, we propose AlphaRL, a plug-in acceleration framework that extrapolates the final parameter update using a short early training window, achieving up to 2.5 speedup while retaining \textgreater 96\% of reasoning performance without extra modules or hyperparameter tuning. This positions our finding as a versatile and practical tool for large-scale RL, opening a path toward principled, interpretable, and efficient training paradigm for LLMs.

**arXiv ID:** 2510.00553
</details>

<details>
<summary><strong>Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning</strong> - Luckeciano C. Melo, Alessandro Abate, Yarin Gal - [[pdf]](https://arxiv.org/pdf/2510.00819)</summary>

**Abstract:** Reinforcement Learning, particularly through policy gradient methods, has played a central role in enabling reasoning capabilities of Large Language Models. However, the optimization stability of policy gradients in this setting remains understudied. As a result, existing implementations often resort to conservative hyperparameter choices to ensure stability, which requires more training samples and increases computational costs. Hence, developing models for reliably tracking the underlying optimization dynamics and leveraging them into training enables more sample-efficient regimes and further unleashes scalable post-training. We address this gap by formalizing the stochastic optimization problem of policy gradients with explicit consideration of second-order geometry. We propose a tractable computational framework that tracks and leverages curvature information during policy updates. We further employ this framework to design interventions in the optimization process through data selection. The resultant algorithm, Curvature-Aware Policy Optimization (CAPO), identifies samples that contribute to unstable updates and masks them out. Theoretically, we establish monotonic improvement guarantees under realistic assumptions. On standard math reasoning benchmarks, we empirically show that CAPO ensures stable updates under aggressive learning regimes where baselines catastrophically fail. With minimal intervention (rejecting fewer than 8% of tokens), CAPO achieves up to 30x improvement in sample efficiency over standard GRPO for LLM reasoning.

**arXiv ID:** 2510.00819
</details>

<details>
<summary><strong>Erase to Improve: Erasable Reinforcement Learning for Search-Augmented LLMs</strong> - Ziliang Wang, Kang An, Xuhui Zheng, Faqiang Qian, Weikun Zhang, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu - [[pdf]](https://arxiv.org/pdf/2510.00861)</summary>

**Abstract:** While search-augmented large language models (LLMs) exhibit impressive capabilities, their reliability in complex multi-hop reasoning remains limited. This limitation arises from three fundamental challenges: decomposition errors, where tasks are incorrectly broken down; retrieval missing, where key evidence fails to be retrieved; and reasoning errors, where flawed logic propagates through the reasoning chain. A single failure in any of these stages can derail the final answer. We propose Erasable Reinforcement Learning (ERL), a novel framework that transforms fragile reasoning into a robust process. ERL explicitly identifies faulty steps, erases them, and regenerates reasoning in place, preventing defective logic from propagating through the reasoning chain. This targeted correction mechanism turns brittle reasoning into a more resilient process. Models trained with ERL, termed ESearch, achieve substantial improvements on HotpotQA, MuSiQue, 2Wiki, and Bamboogle, with the 3B model achieving +8.48% EM and +11.56% F1, and the 7B model achieving +5.38% EM and +7.22% F1 over previous state-of-the-art(SOTA) results. These findings suggest that erasable reinforcement learning provides a powerful paradigm shift for robust multi-step reasoning in LLMs.

**arXiv ID:** 2510.00861
</details>

<details>
<summary><strong>Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers</strong> - Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama - [[pdf]](https://arxiv.org/pdf/2510.00915)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) trains policies against automated verifiers to avoid costly human labeling. To reduce vulnerability to verifier hacking, many RLVR systems collapse rewards to binary $\{0,1\}$ during training. This choice carries a cost: it introduces \textit{false negatives} (rejecting correct answers, FNs) and \textit{false positives} (accepting incorrect ones, FPs). For instance, a rule-based checker may mark the correct fraction $\frac{12}{36}$ as wrong when compared against the canonical $\frac{1}{3}$ due to brittle parsing/equivalence rules (FN), while a large language model (LLM) judges can be gamed by superficial cues or even a single adversarial token, yielding inflated correctness for wrong solutions (FP). We formalize verifier unreliability by modeling the verifier as a stochastic reward channel with asymmetric noise rates. From this abstraction, we derive two correction algorithms for verifier errors. The first is a \textit{backward} correction that de-biases the observed binary reward to recover an \textit{unbiased} estimator of the clean policy gradient. The second is a \textit{forward} correction that reweights score-function terms so that the expected update direction aligns with the \textit{clean gradient}; notably, it requires only the FN rate. We implement both as lightweight hooks in a group relative policy optimization (GRPO)-based RLVR pipeline and evaluate them on math-reasoning models and benchmarks. Across models and datasets, both corrections improve over uncorrected training; the forward variant converges faster and remains stable under heavier noise. Finally, we show a practical appeal mechanism in which a lightweight LLM verifier estimates the FN rate online by rechecking rule-based negatives, obtaining outperformance compared with other state-of-the-art contenders.

**arXiv ID:** 2510.00915
</details>

<details>
<summary><strong>GEM: A Gym for Agentic LLMs</strong> - Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, Chuen Yang Beh, Weixun Wang, Hao Zhu, Weiyan Shi, Diyi Yang, Michael Shieh, Yee Whye Teh, Wee Sun Lee, Min Lin - [[pdf]](https://arxiv.org/pdf/2510.01051)</summary>

**Abstract:** The training paradigm for large language models (LLMs) is moving from static datasets to experience-based learning, where agents acquire skills via interacting with complex environments. To facilitate this transition we introduce GEM (General Experience Maker), an open-source environment simulator designed for the age of LLMs. Analogous to OpenAI-Gym for traditional reinforcement learning (RL), GEM provides a standardized framework for the environment-agent interface, including asynchronous vectorized execution for high throughput, and flexible wrappers for easy extensibility. GEM also features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Along with this, we also provide a set of baselines across 24 environments using REINFORCE with Return Batch Normalization (ReBN), which -- unlike GRPO -- is compatible with the full RL setting of dense per-turn rewards and offers better credit assignment. We further conduct apple-to-apple benchmarking of PPO, GRPO and REINFORCE in both single- and multi-turn settings using GEM to shed light on the algorithmic designs. Lastly, GEM also functions as a convenient evaluation toolkit besides a training environment. We hope this framework can help accelerate future agentic LLM research.

**arXiv ID:** 2510.01051
</details>

<details>
<summary><strong>TAMA: Tool-Augmented Multimodal Agent for Procedural Activity Understanding</strong> - Kimihiro Hasegawa, Wiradee Imrattanatrai, Masaki Asada, Ken Fukuda, Teruko Mitamura - [[pdf]](https://arxiv.org/pdf/2510.00161)</summary>

**Abstract:** Procedural activity assistants potentially support humans in a variety of settings, from our daily lives, e.g., cooking or assembling flat-pack furniture, to professional situations, e.g., manufacturing or biological experiments. Despite its potential use cases, the system development tailored for such an assistant is still underexplored. In this paper, we propose a novel framework, called TAMA, a Tool-Augmented Multimodal Agent, for procedural activity understanding. TAMA enables interleaved multimodal reasoning by making use of multimedia-returning tools in a training-free setting. Our experimental result on the multimodal procedural QA dataset, ProMQA-Assembly, shows that our approach can improve the performance of vision-language models, especially GPT-5 and MiMo-VL. Furthermore, our ablation studies provide empirical support for the effectiveness of two features that characterize our framework, multimedia-returning tools and agentic flexible tool selection. We believe our proposed framework and experimental results facilitate the thinking with images paradigm for video and multimodal tasks, let alone the development of procedural activity assistants.

**arXiv ID:** 2510.00161
</details>

<details>
<summary><strong>ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards</strong> - Shiyu Li, Yang Tang, Yifan Wang, Peiming Li, Xi Chen - [[pdf]](https://arxiv.org/pdf/2510.00568)</summary>

**Abstract:** Search agents powered by Large Language Models (LLMs) have demonstrated significant potential in tackling knowledge-intensive tasks. Reinforcement learning (RL) has emerged as a powerful paradigm for training these agents to perform complex, multi-step reasoning. However, prior RL-based methods often rely on sparse or rule-based rewards, which can lead agents to commit to suboptimal or erroneous reasoning paths without the ability to recover. To address these limitations, we propose ReSeek, a novel self-correcting framework for training search agents. Our framework introduces a self-correction mechanism that empowers the agent to dynamically identify and recover from erroneous search paths during an episode. By invoking a special JUDGE action, the agent can judge the information and re-plan its search strategy. To guide this process, we design a dense, instructive process reward function, which decomposes into a correctness reward for retrieving factual information and a utility reward for finding information genuinely useful for the query. Furthermore, to mitigate the risk of data contamination in existing datasets, we introduce FictionalHot, a new and challenging benchmark with recently curated questions requiring complex reasoning. Being intuitively reasonable and practically simple, extensive experiments show that agents trained with ReSeek significantly outperform SOTA baselines in task success rate and path faithfulness.

**arXiv ID:** 2510.00568
</details>

<details>
<summary><strong>Research on the Integration of Embodied Intelligence and Reinforcement Learning in Textual Domains</strong> - Haonan Wang, Junfeng Sun, Mingjia Zhao, Wei Liu - [[pdf]](https://arxiv.org/pdf/2510.01076)</summary>

**Abstract:** This article addresses embodied intelligence and reinforcement learning integration in the field of text processing, aiming to enhance text handling with more intelligence on the basis of embodied intelligence's perception and action superiority and reinforcement learning's decision optimization capability. Through detailed theoretical explanation and experimental exploration, a novel integration model is introduced. This model has been demonstrated to be very effective in a wide range oftext processing tasks, validating its applicative potential

**arXiv ID:** 2510.01076
</details>

<details>
<summary><strong>BroRL: Scaling Reinforcement Learning via Broadened Exploration</strong> - Jian Hu, Mingjie Liu, Ximing Lu, Fang Wu, Zaid Harchaoui, Shizhe Diao, Yejin Choi, Pavlo Molchanov, Jun Yang, Jan Kautz, Yi Dong - [[pdf]](https://arxiv.org/pdf/2510.01180)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key ingredient for unlocking complex reasoning capabilities in large language models. Recent work ProRL has shown promise in scaling RL by increasing the number of training steps. However, performance plateaus after thousands of steps, with clear diminishing returns from allocating more computation to additional training. In this work, we investigate a complementary paradigm for scaling RL, BroR-Lincreasing the number of rollouts per example to hundreds to exhaustively Broaden exploration, which yields continuous performance gains beyond the saturation point observed in ProRL when scaling the number of training steps. Our approach is motivated by a mass balance equation analysis allowing us to characterize the rate of change in probability mass for correct and incorrect tokens during the reinforcement process. We show that under a one-step RL assumption, sampled rollout tokens always contribute to correct-mass expansion, while unsampled tokens outside rollouts may lead to gains or losses depending on their distribution and the net reward balance. Importantly, as the number of rollouts per example N increases, the effect of unsampled terms diminishes, ensuring overall correct-mass expansion. To validate our theoretical analysis, we conduct simulations under more relaxed conditions and find that a sufficiently large rollout size N-corresponding to ample exploration-guarantees an increase in the probability mass of all correct tokens. Empirically, BroRL revives models saturated after 3K ProRL training steps and demonstrates robust, continuous improvement, achieving state-of-the-art results for the 1.5B model across diverse benchmarks.

**arXiv ID:** 2510.01180
</details>

<details>
<summary><strong>Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning</strong> - Jinyeop Song, Song Wang, Julian Shun, Yada Zhu - [[pdf]](https://arxiv.org/pdf/2509.26383)</summary>

**Abstract:** Knowledge-graph retrieval-augmented generation (KG-RAG) couples large language models (LLMs) with structured, verifiable knowledge graphs (KGs) to reduce hallucinations and expose reasoning traces. However, many KG-RAG systems compose multiple LLM modules (e.g planning, reasoning, and responding), inflating inference cost and binding behavior to a specific target KG. To address this, we introduce KG-R1, an agentic KG retrieval-augmented generation (KG-RAG) framework through reinforcement learning (RL). KG-R1 utilizes a single agent that interacts with KGs as its environment, learning to retrieve at each step and incorporating the retrieved information into its reasoning and generation. The process is optimized through end-to-end RL. In controlled experiments across Knowledge-Graph Question Answering (KGQA) benchmarks, our method demonstrates both efficiency and transferability: Using Qwen-2.5-3B, KG-R1 improves answer accuracy with fewer generation tokens than prior multi-module workflow methods that use larger foundation or fine-tuned models. Furthermore, KG-R1 enables plug and play: after training, it maintains strong accuracy on new KGs without modification. These properties make KG-R1 a promising KG-RAG framework for real-world deployment. Our code is publicly available at this https URL.

**arXiv ID:** 2509.26383
</details>

<details>
<summary><strong>GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents</strong> - Run Luo, Lu Wang, Wanwei He, Longze Chen, Jiaming Li, Xiaobo Xia - [[pdf]](https://arxiv.org/pdf/2504.10458)</summary>

**Abstract:** Existing efforts in building Graphical User Interface (GUI) agents largely rely on the training paradigm of supervised fine-tuning on Large Vision-Language Models (LVLMs). However, this approach not only demands extensive amounts of training data but also struggles to effectively understand GUI screenshots and generalize to unseen interfaces. The issue significantly limits its application in real-world scenarios, especially for high-level tasks. Inspired by Reinforcement Fine-Tuning (RFT) in large reasoning models (e.g., DeepSeek-R1), which efficiently enhances the problem-solving capabilities of large language models in real-world settings, we propose \name, the first reinforcement learning framework designed to enhance the GUI capabilities of LVLMs in high-level real-world task scenarios, through unified action space rule modeling. By leveraging a small amount of carefully curated high-quality data across multiple platforms (including Windows, Linux, MacOS, Android, and Web) and employing policy optimization algorithms such as Group Relative Policy Optimization (GRPO) to update the model, \name achieves superior performance using only 0.02\% of the data (3K vs. 13M) compared to previous state-of-the-art methods like OS-Atlas across eight benchmarks spanning three different platforms (mobile, desktop, and web). These results demonstrate the immense potential of reinforcement learning based on unified action space rule modeling in improving the execution capabilities of LVLMs for real-world GUI agent tasks.

**arXiv ID:** 2504.10458
</details>

<details>
<summary><strong>Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning</strong> - Yixuan Even Xu, Yash Savani, Fei Fang, J. Zico Kolter - [[pdf]](https://arxiv.org/pdf/2504.13818)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has emerged as the leading approach for enhancing reasoning capabilities in large language models. However, it faces a fundamental compute and memory asymmetry: rollout generation is embarrassingly parallel and memory-light, whereas policy updates are communication-heavy and memory-intensive. To address this, we introduce PODS (Policy Optimization with Down-Sampling), which decouples rollout generation from policy updates by training only on a strategically selected subset of rollouts, maintaining learning quality while dramatically reducing update costs. We propose a principled subset selection criterion, max-variance down-sampling, that maximizes reward diversity, and provide an efficient $O(n\log n)$ implementation. Empirically, Group Relative Policy Optimization (GRPO) with PODS achieves the peak test accuracy of vanilla GRPO at least $\mathbf{1.7\times}$ faster across the different reasoning benchmarks and hardware configurations we tested.

**arXiv ID:** 2504.13818
</details>

<details>
<summary><strong>Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents</strong> - Tianyi Ma, Yue Zhang, Zehao Wang, Parisa Kordjamshidi - [[pdf]](https://arxiv.org/pdf/2508.07642)</summary>

**Abstract:** Vision-and-Language Navigation (VLN) poses significant challenges for agents to interpret natural language instructions and navigate complex 3D environments. While recent progress has been driven by large-scale pre-training and data augmentation, current methods still struggle to generalize to unseen scenarios, particularly when complex spatial and temporal reasoning is required. In this work, we propose SkillNav, a modular framework that introduces structured, skill-based reasoning into Transformer-based VLN agents. Our method decomposes navigation into a set of interpretable atomic skills (e.g., Vertical Movement, Area and Region Identification, Stop and Pause), each handled by a specialized agent. To support targeted skill training without manual data annotation, we construct a synthetic dataset pipeline that generates diverse, linguistically natural, skill-specific instruction-trajectory pairs. We then introduce a novel training-free Vision-Language Model (VLM)-based router, which dynamically selects the most suitable agent at each time step by aligning sub-goals with visual observations and historical actions. SkillNav obtains competitive results on commonly used benchmarks and establishes state-of-the-art generalization to the GSA-R2R, a benchmark with novel instruction styles and unseen environments.

**arXiv ID:** 2508.07642
</details>

<details>
<summary><strong>Interactive Recommendation Agent with Active User Commands</strong> - Jiakai Tang, Yujie Luo, Xunke Xi, Fei Sun, Xueyang Feng, Sunhao Dai, Chao Yi, Dian Chen, Zhujin Gao, Yang Li, Xu Chen, Wen Chen, Jian Wu, Yuning Jiang, Bo Zheng - [[pdf]](https://arxiv.org/pdf/2509.21317)</summary>

**Abstract:** Traditional recommender systems rely on passive feedback mechanisms that limit users to simple choices such as like and dislike. However, these coarse-grained signals fail to capture users' nuanced behavior motivations and intentions. In turn, current systems cannot also distinguish which specific item attributes drive user satisfaction or dissatisfaction, resulting in inaccurate preference modeling. These fundamental limitations create a persistent gap between user intentions and system interpretations, ultimately undermining user satisfaction and harming system effectiveness.
To address these limitations, we introduce the Interactive Recommendation Feed (IRF), a pioneering paradigm that enables natural language commands within mainstream recommendation feeds. Unlike traditional systems that confine users to passive implicit behavioral influence, IRF empowers active explicit control over recommendation policies through real-time linguistic commands. To support this paradigm, we develop RecBot, a dual-agent architecture where a Parser Agent transforms linguistic expressions into structured preferences and a Planner Agent dynamically orchestrates adaptive tool chains for on-the-fly policy adjustment. To enable practical deployment, we employ simulation-augmented knowledge distillation to achieve efficient performance while maintaining strong reasoning capabilities. Through extensive offline and long-term online experiments, RecBot shows significant improvements in both user satisfaction and business outcomes.

**arXiv ID:** 2509.21317
</details>

<details>
<summary><strong>DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search</strong> - Fang Wu, Weihao Xuan, Heli Qi, Ximing Lu, Aaron Tu, Li Erran Li, Yejin Choi - [[pdf]](https://arxiv.org/pdf/2509.25454)</summary>

**Abstract:** Although RLVR has become an essential component for developing advanced reasoning skills in LLMs, contemporary studies have documented training plateaus that emerge following thousands of optimization steps, demonstrating notable decreases in performance gains despite increased computational investment. This limitation stems from the sparse exploration patterns inherent in current RLVR practices, where models rely on limited rollouts that often miss critical reasoning paths and fail to provide systematic coverage of the solution space. We present DeepSearch, a framework that integrates Monte Carlo Tree Search directly into RLVR training. In contrast to existing methods that rely on tree search only at inference, DeepSearch embeds structured search into the training loop, enabling systematic exploration and fine-grained credit assignment across reasoning steps. Through training-time exploration, DeepSearch addresses the fundamental bottleneck of insufficient exploration, which leads to diminishing performance improvements over prolonged training steps. Our contributions include: (1) a global frontier selection strategy that prioritizes promising nodes across the search tree, (2) selection with entropy-based guidance that identifies confident paths for supervision, and (3) adaptive replay buffer training with solution caching for efficiency. Experiments on mathematical reasoning benchmarks show that DeepSearch achieves 62.95% average accuracy and establishes a new state-of-the-art for 1.5B reasoning models - using 5.7x fewer GPU hours than extended training approaches. These results highlight the importance of strategic exploration over brute-force scaling and demonstrate the promise of algorithmic innovation for advancing RLVR methodologies. DeepSearch establishes a new direction for scaling reasoning capabilities through systematic search rather than prolonged computation.

**arXiv ID:** 2509.25454
</details>

<details>
<summary><strong>TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning</strong> - Marco Bagatella, Matteo Pirotta, Ahmed Touati, Alessandro Lazaric, Andrea Tirinzoni - [[pdf]](https://arxiv.org/pdf/2510.00739)</summary>

**Abstract:** Latent prediction--where agents learn by predicting their own latents--has emerged as a powerful paradigm for training general representations in machine learning. In reinforcement learning (RL), this approach has been explored to define auxiliary losses for a variety of settings, including reward-based and unsupervised RL, behavior cloning, and world modeling. While existing methods are typically limited to single-task learning, one-step prediction, or on-policy trajectory data, we show that temporal difference (TD) learning enables learning representations predictive of long-term latent dynamics across multiple policies from offline, reward-free transitions. Building on this, we introduce TD-JEPA, which leverages TD-based latent-predictive representations into unsupervised RL. TD-JEPA trains explicit state and task encoders, a policy-conditioned multi-step predictor, and a set of parameterized policies directly in latent space. This enables zero-shot optimization of any reward function at test time. Theoretically, we show that an idealized variant of TD-JEPA avoids collapse with proper initialization, and learns encoders that capture a low-rank factorization of long-term policy dynamics, while the predictor recovers their successor features in latent space. Empirically, TD-JEPA matches or outperforms state-of-the-art baselines on locomotion, navigation, and manipulation tasks across 13 datasets in ExoRL and OGBench, especially in the challenging setting of zero-shot RL from pixels.

**arXiv ID:** 2510.00739
</details>

<details>
<summary><strong>Guiding Evolutionary Molecular Design: Adding Reinforcement Learning for Mutation Selection</strong> - Gaelle Milon-Harnois, Chaimaa Touhami, Nicolas Gutowski, Benoit Da Mota, Thomas Cauchy - [[pdf]](https://arxiv.org/pdf/2510.00802)</summary>

**Abstract:** The efficient exploration of chemical space remains a central challenge, as many generative models still produce unstable or non-synthesizable compounds. To address these limitations, we present EvoMol-RL, a significant extension of the EvoMol evolutionary algorithm that integrates reinforcement learning to guide molecular mutations based on local structural context. By leveraging Extended Connectivity Fingerprints (ECFPs), EvoMol-RL learns context-aware mutation policies that prioritize chemically plausible transformations. This approach significantly improves the generation of valid and realistic molecules, reducing the frequency of structural artifacts and enhancing optimization performance. The results demonstrate that EvoMol-RL consistently outperforms its baseline in molecular pre-filtering realism. These results emphasize the effectiveness of combining reinforcement learning with molecular fingerprints to generate chemically relevant molecular structures.

**arXiv ID:** 2510.00802
</details>

<details>
<summary><strong>Rectifying Regression in Reinforcement Learning</strong> - Alex Ayoub, David Szepesvári, Alireza Baktiari, Csaba Szepesvári, Dale Schuurmans - [[pdf]](https://arxiv.org/pdf/2510.00885)</summary>

**Abstract:** This paper investigates the impact of the loss function in value-based methods for reinforcement learning through an analysis of underlying prediction objectives. We theoretically show that mean absolute error is a better prediction objective than the traditional mean squared error for controlling the learned policy's suboptimality gap. Furthermore, we present results that different loss functions are better aligned with these different regression objectives: binary and categorical cross-entropy losses with the mean absolute error and squared loss with the mean squared error. We then provide empirical evidence that algorithms minimizing these cross-entropy losses can outperform those based on the squared loss in linear reinforcement learning.

**arXiv ID:** 2510.00885
</details>

<details>
<summary><strong>Multi-Actor Multi-Critic Deep Deterministic Reinforcement Learning with a Novel Q-Ensemble Method</strong> - Andy Wu, Chun-Cheng Lin, Rung-Tzuo Liaw, Yuehua Huang, Chihjung Kuo, Chia Tong Weng - [[pdf]](https://arxiv.org/pdf/2510.01083)</summary>

**Abstract:** Reinforcement learning has gathered much attention in recent years due to its rapid development and rich applications, especially on control systems and robotics. When tackling real-world applications with reinforcement learning method, the corresponded Markov decision process may have huge discrete or even continuous state/action space. Deep reinforcement learning has been studied for handling these issues through deep learning for years, and one promising branch is the actor-critic architecture. Many past studies leveraged multiple critics to enhance the accuracy of evaluation of a policy for addressing the overestimation and underestimation issues. However, few studies have considered the architecture with multiple actors together with multiple critics. This study proposes a novel multi-actor multi-critic (MAMC) deep deterministic reinforcement learning method. The proposed method has three main features, including selection of actors based on non-dominated sorting for exploration with respect to skill and creativity factors, evaluation for actors and critics using a quantile-based ensemble strategy, and exploiting actors with best skill factor. Theoretical analysis proves the learning stability and bounded estimation bias for the MAMC. The present study examines the performance on a well-known reinforcement learning benchmark MuJoCo. Experimental results show that the proposed framework outperforms state-of-the-art deep deterministic based reinforcement learning methods. Experimental analysis also indicates the proposed components are effective. Empirical analysis further investigates the validity of the proposed method, and shows its benefit on complicated problems. The source code can be found at this https URL.

**arXiv ID:** 2510.01083
</details>

<details>
<summary><strong>Eliciting Chain-of-Thought Reasoning for Time Series Analysis using Reinforcement Learning</strong> - Felix Parker, Nimeesha Chan, Chi Zhang, Kimia Ghobadi - [[pdf]](https://arxiv.org/pdf/2510.01116)</summary>

**Abstract:** Complex numerical time series analysis often demands multi-step reasoning capabilities beyond current models' reach. Tasks like medical diagnosis and weather forecasting require sequential reasoning processes -- including counterfactual analysis, logical deduction, knowledge application, and multi-modal contextual integration -- that existing time series models cannot explicitly perform. While recent research has shown large language models (LLMs) can achieve sophisticated Chain-of-Thought (CoT) reasoning through reinforcement learning (RL), these advances have primarily focused on mathematical and coding domains, with LLMs still demonstrating poor performance on time series tasks. We introduce Chain Of thought for Understanding Numerical Time Series (COUNTS), the first framework that trains LLMs to perform CoT reasoning across diverse time series tasks using RL with verifiable rewards. Our approach employs a Residual Vector-Quantized VAE to create high-fidelity discrete tokens that seamlessly integrate into a pre-trained LLM's vocabulary. COUNTS undergoes a two-stage training process: first, supervised fine-tuning on time series analysis tasks to master our novel representations, followed by Group Relative Policy Optimization training on verifiable problems using prompting strategies that encourage explicit reasoning steps before producing final answers. Our experiments demonstrate that this RL-driven approach with intermediate CoT reasoning significantly enhances LLM performance across various time series analysis tasks, opening new possibilities for complex temporal data reasoning.

**arXiv ID:** 2510.01116
</details>

<details>
<summary><strong>Learning Human Reaching Optimality Principles from Minimal Observation Inverse Reinforcement Learning</strong> - Sarmad Mehrdad, Maxime Sabbah, Vincent Bonnet, Ludovic Righetti - [[pdf]](https://arxiv.org/pdf/2510.00329)</summary>

**Abstract:** This paper investigates the application of Minimal Observation Inverse Reinforcement Learning (MO-IRL) to model and predict human arm-reaching movements with time-varying cost weights. Using a planar two-link biomechanical model and high-resolution motion-capture data from subjects performing a pointing task, we segment each trajectory into multiple phases and learn phase-specific combinations of seven candidate cost functions. MO-IRL iteratively refines cost weights by scaling observed and generated trajectories in the maximum entropy IRL formulation, greatly reducing the number of required demonstrations and convergence time compared to classical IRL approaches. Training on ten trials per posture yields average joint-angle Root Mean Squared Errors (RMSE) of 6.4 deg and 5.6 deg for six- and eight-segment weight divisions, respectively, versus 10.4 deg using a single static weight. Cross-validation on remaining trials and, for the first time, inter-subject validation on an unseen subject's 20 trials, demonstrates comparable predictive accuracy, around 8 deg RMSE, indicating robust generalization. Learned weights emphasize joint acceleration minimization during movement onset and termination, aligning with smoothness principles observed in biological motion. These results suggest that MO-IRL can efficiently uncover dynamic, subject-independent cost structures underlying human motor control, with potential applications for humanoid robots.

**arXiv ID:** 2510.00329
</details>

<details>
<summary><strong>Strategic Fusion of Vision Language Models: Shapley-Credited Context-Aware Dawid-Skene for Multi-Label Tasks in Autonomous Driving</strong> - Yuxiang Feng, Keyang Zhang, Hassane Ouchouid, Ashwil Kaniamparambil, Ioannis Souflas, Panagiotis Angeloudis - [[pdf]](https://arxiv.org/pdf/2510.01126)</summary>

**Abstract:** Large vision-language models (VLMs) are increasingly used in autonomous-vehicle (AV) stacks, but hallucination limits their reliability in safety-critical pipelines. We present Shapley-credited Context-Aware Dawid-Skene with Agreement, a game-theoretic fusion method for multi-label understanding of ego-view dashcam video. It learns per-model, per-label, context-conditioned reliabilities from labelled history and, at inference, converts each model's report into an agreement-guardrailed log-likelihood ratio that is combined with a contextual prior and a public reputation state updated via Shapley-based team credit. The result is calibrated, thresholdable posteriors that (i) amplify agreement among reliable models, (ii) preserve uniquely correct single-model signals, and (iii) adapt to drift. To specialise general VLMs, we curate 1,000 real-world dashcam clips with structured annotations (scene description, manoeuvre recommendation, rationale) via an automatic pipeline that fuses HDD ground truth, vehicle kinematics, and YOLOv11 + BoT-SORT tracking, guided by a three-step chain-of-thought prompt; three heterogeneous VLMs are then fine-tuned with LoRA. We evaluate with Hamming distance, Micro-Macro-F1, and average per-video latency. Empirically, the proposed method achieves a 23% reduction in Hamming distance, 55% improvement in Macro-F1, and 47% improvement in Micro-F1 when comparing with the best single model, supporting VLM fusion as a calibrated, interpretable, and robust decision-support component for AV pipelines.

**arXiv ID:** 2510.01126
</details>

<details>
<summary><strong>HetSwarm: Cooperative Navigation of Heterogeneous Swarm in Dynamic and Dense Environments through Impedance-based Guidance</strong> - Malaika Zafar, Roohan Ahmed Khan, Aleksey Fedoseev, Kumar Katyayan Jaiswal, Dzmitry Tsetserukou - [[pdf]](https://arxiv.org/pdf/2502.06722)</summary>

**Abstract:** With the growing demand for efficient logistics and warehouse management, unmanned aerial vehicles (UAVs) are emerging as a valuable complement to automated guided vehicles (AGVs). UAVs enhance efficiency by navigating dense environments and operating at varying altitudes. However, their limited flight time, battery life, and payload capacity necessitate a supporting ground station. To address these challenges, we propose HetSwarm, a heterogeneous multi-robot system that combines a UAV and a mobile ground robot for collaborative navigation in cluttered and dynamic conditions. Our approach employs an artificial potential field (APF)-based path planner for the UAV, allowing it to dynamically adjust its trajectory in real time. The ground robot follows this path while maintaining connectivity through impedance links, ensuring stable coordination. Additionally, the ground robot establishes temporal impedance links with low-height ground obstacles to avoid local collisions, as these obstacles do not interfere with the UAV's flight. Experimental validation of HetSwarm in diverse environmental conditions demonstrated a 90% success rate across 30 test cases. The ground robot exhibited an average deviation of 45 cm near obstacles, confirming effective collision avoidance. Extensive simulations in the Gym PyBullet environment further validated the robustness of our system for real-world applications, demonstrating its potential for dynamic, real-time task execution in cluttered environments.

**arXiv ID:** 2502.06722
</details>

<details>
<summary><strong>Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving</strong> - Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, Liam Paull - [[pdf]](https://arxiv.org/pdf/2506.11234)</summary>

**Abstract:** We present Poutine, a 3B-parameter vision-language model (VLM) tailored for end-to-end autonomous driving in long-tail driving scenarios. Poutine is trained in two stages. To obtain strong base driving capabilities, we train Poutine-Base in a self-supervised vision-language-trajectory (VLT) next-token prediction fashion on 83 hours of CoVLA nominal driving and 11 hours of Waymo long-tail driving. Accompanying language annotations are auto-generated with a 72B-parameter VLM. Poutine is obtained by fine-tuning Poutine-Base with Group Relative Policy Optimization (GRPO) using less than 500 preference-labeled frames from the Waymo validation set. We show that both VLT pretraining and RL fine-tuning are critical to attain strong driving performance in the long-tail. Poutine-Base achieves a rater-feedback score (RFS) of 8.12 on the validation set, nearly matching Waymo's expert ground-truth RFS. The final Poutine model achieves an RFS of 7.99 on the official Waymo test set, placing 1st in the 2025 Waymo Vision-Based End-to-End Driving Challenge by a significant margin. These results highlight the promise of scalable VLT pre-training and lightweight RL fine-tuning to enable robust and generalizable autonomy.

**arXiv ID:** 2506.11234
</details>

<details>
<summary><strong>Vision-driven River Following of UAV via Safe Reinforcement Learning using Semantic Dynamics Model</strong> - Zihan Wang, Nina Mahmoudian - [[pdf]](https://arxiv.org/pdf/2508.09971)</summary>

**Abstract:** Vision-driven autonomous river following by Unmanned Aerial Vehicles is critical for applications such as rescue, surveillance, and environmental monitoring, particularly in dense riverine environments where GPS signals are unreliable. These safety-critical navigation tasks must satisfy hard safety constraints while optimizing performance. Moreover, the reward in river following is inherently history-dependent (non-Markovian) by which river segment has already been visited, making it challenging for standard safe Reinforcement Learning (SafeRL). To address these gaps, we propose three contributions. First, we introduce Marginal Gain Advantage Estimation, which refines the reward advantage function by using a sliding window baseline computed from historical episodic returns, aligning the advantage estimate with non-Markovian dynamics. Second, we develop a Semantic Dynamics Model based on patchified water semantic masks offering more interpretable and data-efficient short-term prediction of future observations compared to latent vision dynamics models. Third, we present the Constrained Actor Dynamics Estimator architecture, which integrates the actor, cost estimator, and SDM for cost advantage estimation to form a model-based SafeRL framework. Simulation results demonstrate that MGAE achieves faster convergence and superior performance over traditional critic-based methods like Generalized Advantage Estimation. SDM provides more accurate short-term state predictions that enable the cost estimator to better predict potential violations. Overall, CADE effectively integrates safety regulation into model-based RL, with the Lagrangian approach providing a "soft" balance between reward and safety during training, while the safety layer enhances inference by imposing a "hard" action overlay.

**arXiv ID:** 2508.09971
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
