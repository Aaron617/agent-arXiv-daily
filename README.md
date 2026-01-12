# Agent arXiv Daily

**Last Updated:** 2026-01-12 03:16:35

**Total Papers:** 54

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
<summary><strong>Naiad: Novel Agentic Intelligent Autonomous System for Inland Water Monitoring</strong> - Eirini Baltzi, Tilemachos Moumouris, Athena Psalta, Vasileios Tsironis, Konstantinos Karantzalos - [[pdf]](https://arxiv.org/pdf/2601.05256)</summary>

**Abstract:** Inland water monitoring is vital for safeguarding public health and ecosystems, enabling timely interventions to mitigate risks. Existing methods often address isolated sub-problems such as cyanobacteria, chlorophyll, or other quality indicators separately. NAIAD introduces an agentic AI assistant that leverages Large Language Models (LLMs) and external analytical tools to deliver a holistic solution for inland water monitoring using Earth Observation (EO) data. Designed for both experts and non-experts, NAIAD provides a single-prompt interface that translates natural-language queries into actionable insights. Through Retrieval-Augmented Generation (RAG), LLM reasoning, external tool orchestration, computational graph execution, and agentic reflection, it retrieves and synthesizes knowledge from curated sources to produce tailored reports. The system integrates diverse tools for weather data, Sentinel-2 imagery, remote-sensing index computation (e.g., NDCI), chlorophyll-a estimation, and established platforms such as CyFi. Performance is evaluated using correctness and relevancy metrics, achieving over 77% and 85% respectively on a dedicated benchmark covering multiple user-expertise levels. Preliminary results show strong adaptability and robustness across query types. An ablation study on LLM backbones further highlights Gemma 3 (27B) and Qwen 2.5 (14B) as offering the best balance between computational efficiency and reasoning performance.

**arXiv ID:** 2601.05256
</details>

<details>
<summary><strong>From Laboratory to Real-World Applications: Benchmarking Agentic Code Reasoning at the Repository Level</strong> - Jia Li, Yuxin Su, Michael R. Lyu - [[pdf]](https://arxiv.org/pdf/2601.03731)</summary>

**Abstract:** As large language models (LLMs) evolve into autonomous agents, evaluating repository-level reasoning, the ability to maintain logical consistency across massive, real-world, interdependent file systems, has become critical. Current benchmarks typically fluctuate between isolated code snippets and black-box evaluations. We present RepoReason, a white-box diagnostic benchmark centered on abductive assertion verification. To eliminate memorization while preserving authentic logical depth, we implement an execution-driven mutation framework that utilizes the environment as a semantic oracle to regenerate ground-truth states. Furthermore, we establish a fine-grained diagnostic system using dynamic program slicing, quantifying reasoning via three orthogonal metrics: $ESV$ (reading load), $MCL$ (simulation depth), and $DFI$ (integration width). Comprehensive evaluations of frontier models (e.g., Claude-4.5-Sonnet, DeepSeek-v3.1-Terminus) reveal a prevalent aggregation deficit, where integration width serves as the primary cognitive bottleneck. Our findings provide granular white-box insights for optimizing the next generation of agentic software engineering.

**arXiv ID:** 2601.03731
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation</strong> - Rongxin Chen, Tianyu Wu, Bingbing Xu, Xiucheng Xu, Huawei Shen - [[pdf]](https://arxiv.org/pdf/2601.05656)</summary>

**Abstract:** High-fidelity agent initialization is crucial for credible Agent-Based Modeling across diverse domains. A robust framework should be Topic-Adaptive, capturing macro-level joint distributions while ensuring micro-level individual rationality. Existing approaches fall into two categories: static data-based retrieval methods that fail to adapt to unseen topics absent from the data, and LLM-based generation methods that lack macro-level distribution awareness, resulting in inconsistencies between micro-level persona attributes and reality. To address these problems, we propose HAG, a Hierarchical Agent Generation framework that formalizes population generation as a two-stage decision process. Firstly, utilizing a World Knowledge Model to infer hierarchical conditional probabilities to construct the Topic-Adaptive Tree, achieving macro-level distribution alignment. Then, grounded real-world data, instantiation and agentic augmentation are carried out to ensure micro-level consistency. Given the lack of specialized evaluation, we establish a multi-domain benchmark and a comprehensive PACE evaluation framework. Extensive experiments show that HAG significantly outperforms representative baselines, reducing population alignment errors by an average of 37.7% and enhancing sociological consistency by 18.8%.

**arXiv ID:** 2601.05656
</details>

<details>
<summary><strong>Can We Predict Before Executing Machine Learning Agents?</strong> - Jingsheng Zheng, Jintian Zhang, Yujie Luo, Yuren Mao, Yunjun Gao, Lun Du, Huajun Chen, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2601.05930)</summary>

**Abstract:** Autonomous machine learning agents have revolutionized scientific discovery, yet they remain constrained by a Generate-Execute-Feedback paradigm. Previous approaches suffer from a severe Execution Bottleneck, as hypothesis evaluation relies strictly on expensive physical execution. To bypass these physical constraints, we internalize execution priors to substitute costly runtime checks with instantaneous predictive reasoning, drawing inspiration from World Models. In this work, we formalize the task of Data-centric Solution Preference and construct a comprehensive corpus of 18,438 pairwise comparisons. We demonstrate that LLMs exhibit significant predictive capabilities when primed with a Verified Data Analysis Report, achieving 61.5% accuracy and robust confidence calibration. Finally, we instantiate this framework in FOREAGENT, an agent that employs a Predict-then-Verify loop, achieving a 6x acceleration in convergence while surpassing execution-based baselines by +6%. Our code and dataset will be publicly available soon at this https URL.

**arXiv ID:** 2601.05930
</details>

<details>
<summary><strong>See or Say Graphs: Agent-Driven Scalable Graph Structure Understanding with Vision-Language Models</strong> - Shuo Han, Yukun Cao, Zezhong Ding, Zengyi Gao, S Kevin Zhou, Xike Xie - [[pdf]](https://arxiv.org/pdf/2510.16769)</summary>

**Abstract:** Vision-language models (VLMs) have shown promise in graph structure understanding, but remain limited by input-token constraints, facing scalability bottlenecks and lacking effective mechanisms to coordinate textual and visual modalities. To address these challenges, we propose GraphVista, a unified framework that enhances both scalability and modality coordination in graph structure understanding. For scalability, GraphVista organizes graph information hierarchically into a lightweight GraphRAG base, which retrieves only task-relevant textual descriptions and high-resolution visual subgraphs, compressing redundant context while preserving key reasoning elements. For modality coordination, GraphVista introduces a planning agent that decomposes and routes tasks to the most suitable modality-using the text modality for direct access to explicit graph properties and the visual modality for local graph structure reasoning grounded in explicit topology. Extensive experiments demonstrate that GraphVista scales to large graphs, up to 200$\times$ larger than those used in existing benchmarks, and consistently outperforms existing textual, visual, and fusion-based methods, achieving up to 4.4$\times$ quality improvement over the state-of-the-art baselines by fully exploiting the complementary strengths of both modalities.

**arXiv ID:** 2510.16769
</details>

<details>
<summary><strong>Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models</strong> - Can Xu, Lingyong Yan, Jiayi Wu, Haosen Wang, Shuaiqiang Wang, Yuchen Li, Jizhou Huang, Dawei Yin, Xiang Li - [[pdf]](https://arxiv.org/pdf/2601.04651)</summary>

**Abstract:** Recent advances in synergizing large reasoning models (LRMs) with retrieval-augmented generation (RAG) have shown promising results, yet two critical challenges remain: (1) reasoning models typically operate from a single, unchallenged perspective, limiting their ability to conduct deep, self-correcting reasoning over external documents, and (2) existing training paradigms rely excessively on outcome-oriented rewards, which provide insufficient signal for shaping the complex, multi-step reasoning process. To address these issues, we propose an Reasoner-Verifier framework named Adversarial Reasoning RAG (ARR). The Reasoner and Verifier engage in reasoning on retrieved evidence and critiquing each other's logic while being guided by process-aware advantage that requires no external scoring model. This reward combines explicit observational signals with internal model uncertainty to jointly optimize reasoning fidelity and verification rigor. Experiments on multiple benchmarks demonstrate the effectiveness of our method.

**arXiv ID:** 2601.04651
</details>

<details>
<summary><strong>Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks</strong> - Elias Lumer, Faheem Nizar, Akshaya Jangiti, Kevin Frank, Anmol Gulati, Mandar Phadate, Vamse Kumar Subbiah - [[pdf]](https://arxiv.org/pdf/2601.06007)</summary>

**Abstract:** Recent advancements in Large Language Model (LLM) agents have enabled complex multi-turn agentic tasks requiring extensive tool calling, where conversations can span dozens of API calls with increasingly large context windows. However, although major LLM providers offer prompt caching to reduce cost and latency, its benefits for agentic workloads remain underexplored in the research literature. To our knowledge, no prior work quantifies these cost savings or compares caching strategies for multi-turn agentic tasks. We present a comprehensive evaluation of prompt caching across three major LLM providers (OpenAI, Anthropic, and Google) and compare three caching strategies, including full context caching, system prompt only caching, and caching that excludes dynamic tool results. We evaluate on DeepResearchBench, a multi-turn agentic benchmark where agents autonomously execute real-world web search tool calls to answer complex research questions, measuring both API cost and time to first token (TTFT) across over 500 agent sessions with 10,000-token system prompts. Our results demonstrate that prompt caching reduces API costs by 45-80% and improves time to first token by 13-31% across providers. We find that strategic prompt cache block control, such as placing dynamic content at the end of the system prompt, avoiding dynamic traditional function calling, and excluding dynamic tool results, provides more consistent benefits than naive full-context caching, which can paradoxically increase latency. Our analysis reveals nuanced variations in caching behavior across providers, and we provide practical guidance for implementing prompt caching in production agentic systems.

**arXiv ID:** 2601.06007
</details>

<details>
<summary><strong>TIME: Temporally Intelligent Meta-reasoning Engine for Context Triggered Explicit Reasoning</strong> - Susmit Das - [[pdf]](https://arxiv.org/pdf/2601.05300)</summary>

**Abstract:** Reasoning oriented large language models often expose explicit "thinking" as long, turn-global traces at the start of every response, either always on or toggled externally at inference time. While useful for arithmetic, programming, and problem solving, this design is costly, blurs claim level auditability, and cannot re-trigger explicit reasoning once the model begins presenting. Dialogue models are also largely blind to temporal structure, treating replies after seconds and replies after weeks as equivalent unless time is stated in text. We introduce TIME, the Temporally Intelligent Meta-reasoning Engine, a behavioral alignment framework that treats explicit reasoning as a context sensitive resource driven by discourse and temporal cues. TIME augments dialogue with optional ISO 8601 <time> tags, tick turns that represent silent gaps, and short <think> blocks that can appear anywhere in a reply. A four-phase curriculum including a small, maximally diverse full-batch alignment step trains Qwen3 dense models to invoke brief, in-place reasoning bursts and keep user facing text compact. We evaluate with TIMEBench, a temporally grounded dialogue benchmark probing chronology, commonsense under gaps and offsets, anomaly detection, and continuity. Across 4B to 32B scales, TIME improves TIMEBench scores over base Qwen3 in both thinking and no-thinking modes while reducing reasoning tokens by about an order of magnitude. Our training data and code are available at this https URL and TIMEBench is available at this https URL

**arXiv ID:** 2601.05300
</details>

<details>
<summary><strong>The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era</strong> - Zhixian Zhao, Shuiyuan Wang, Guojian Li, Hongfei Xue, Chengyou Wang, Shuai Wang, Longshuai Xiao, Zihan Zhang, Hui Bu, Xin Xu, Xinsheng Wang, Hexin Liu, Eng Siong Chng, Hung-yi Lee, Haizhou Li, Lei Xie - [[pdf]](https://arxiv.org/pdf/2601.05564)</summary>

**Abstract:** Driven by the rapid advancement of Large Language Models (LLMs), particularly Audio-LLMs and Omni-models, spoken dialogue systems have evolved significantly, progressively narrowing the gap between human-machine and human-human interactions. Achieving truly ``human-like'' communication necessitates a dual capability: emotional intelligence to perceive and resonate with users' emotional states, and robust interaction mechanisms to navigate the dynamic, natural flow of conversation, such as real-time turn-taking. Therefore, we launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 to benchmark these dual capabilities. Anchored by a sizable dataset derived from authentic human conversations, this initiative establishes a fair evaluation platform across two tracks: (1) Emotional Intelligence, targeting long-term emotion understanding and empathetic generation; and (2) Full-Duplex Interaction, systematically evaluating real-time decision-making under `` listening-while-speaking'' conditions. This paper summarizes the dataset, track configurations, and the final results.

**arXiv ID:** 2601.05564
</details>

<details>
<summary><strong>From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems</strong> - Parisa Rabbani, Nimet Beyza Bozdag, Dilek Hakkani-Tür - [[pdf]](https://arxiv.org/pdf/2511.10871)</summary>

**Abstract:** LLMs are increasingly employed as judges across a variety of tasks, including those involving everyday social interactions. Yet, it remains unclear whether such LLM-judges can reliably assess tasks that require social or conversational judgment. We investigate how an LLM's conviction is changed when a task is reframed from a direct factual query to a Conversational Judgment Task. Our evaluation framework contrasts the model's performance on direct factual queries with its assessment of a speaker's correctness when the same information is presented within a minimal dialogue, effectively shifting the query from "Is this statement correct?" to "Is this speaker correct?". Furthermore, we apply pressure in the form of a simple rebuttal ("The previous answer is incorrect.") to both conditions. This perturbation allows us to measure how firmly the model maintains its position under conversational pressure. Our findings show that while some models like GPT-4o-mini reveal sycophantic tendencies under social framing tasks, others like Llama-8B-Instruct become overly-critical. We observe an average performance change of 9.24% across all models, demonstrating that even minimal dialogue context can significantly alter model judgment, underscoring conversational framing as a key factor in LLM-based evaluation. The proposed framework offers a reproducible methodology for diagnosing model conviction and contributes to the development of more trustworthy dialogue systems.

**arXiv ID:** 2511.10871
</details>

<details>
<summary><strong>Assembling Solar Panels by Dual Robot Arms Towards Full Autonomous Lunar Base Construction</strong> - Luca Nunziante, Kentaro Uno, Gustavo H. Diaz, Shreya Santra, Alessandro De Luca, Kazuya Yoshida - [[pdf]](https://arxiv.org/pdf/2601.05491)</summary>

**Abstract:** Since the successful Apollo program, humanity is once again aiming to return to the Moon for scientific discovery, resource mining, and inhabitation. Upcoming decades focus on building a lunar outpost, with robotic systems playing a crucial role to safely and efficiently establish essential infrastructure such as solar power generating towers. Similar to the construction of the International Space Station (ISS), shipping necessary components via modules and assembling them in situ should be a practical scenario. In this context, this paper focuses on the integration of vision, control, and hardware systems within an autonomous sequence for a dual-arm robot system. We explore a perception and control pipeline specifically designed for assembling solar panel modules, one of the benchmark tasks. Ad hoc hardware was designed and tested in real-world experiments. A mock-up of modular solar panels and active-passive connectors are employed, with the control of this grappling fixture integrated into the proposed pipeline. The successful implementation of our method demonstrates that the two robot manipulators can effectively connect arbitrarily placed panels, highlighting the seamless integration of vision, control, and hardware systems in complex space applications.

**arXiv ID:** 2601.05491
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>Effects of personality steering on cooperative behavior in Large Language Model agents</strong> - Mizuki Sakai, Mizuki Yokoyama, Wakaba Tateishi, Genki Ichinose - [[pdf]](https://arxiv.org/pdf/2601.05302)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as autonomous agents in strategic and social interactions. Although recent studies suggest that assigning personality traits to LLMs can influence their behavior, how personality steering affects cooperation under controlled conditions remains unclear. In this study, we examine the effects of personality steering on cooperative behavior in LLM agents using repeated Prisoner's Dilemma games. Based on the Big Five framework, we first measure basic personality profiles of three models, GPT-3.5-turbo, GPT-4o, and GPT-5, using the Big Five Inventory. We then compare behavior under baseline and personality-informed conditions, and further analyze the effects of independently manipulating each personality dimension to extreme values. Our results show that agreeableness is the dominant factor promoting cooperation across all models, while other personality traits have limited impact. Explicit personality information increases cooperation but can also raise vulnerability to exploitation, particularly in earlier-generation models. In contrast, later-generation models exhibit more selective cooperation. These findings indicate that personality steering acts as a behavioral bias rather than a deterministic control mechanism.

**arXiv ID:** 2601.05302
</details>

<details>
<summary><strong>MMUEChange: A Generalized LLM Agent Framework for Intelligent Multi-Modal Urban Environment Change Analysis</strong> - Zixuan Xiao, Jun Ma, Siwei Zhang - [[pdf]](https://arxiv.org/pdf/2601.05483)</summary>

**Abstract:** Understanding urban environment change is essential for sustainable development. However, current approaches, particularly remote sensing change detection, often rely on rigid, single-modal analysis. To overcome these limitations, we propose MMUEChange, a multi-modal agent framework that flexibly integrates heterogeneous urban data via a modular toolkit and a core module, Modality Controller for cross- and intra-modal alignment, enabling robust analysis of complex urban change scenarios. Case studies include: a shift toward small, community-focused parks in New York, reflecting local green space efforts; the spread of concentrated water pollution across districts in Hong Kong, pointing to coordinated water management; and a notable decline in open dumpsites in Shenzhen, with contrasting links between nighttime economic activity and waste types, indicating differing urban pressures behind domestic and construction waste. Compared to the best-performing baseline, the MMUEChange agent achieves a 46.7% improvement in task success rate and effectively mitigates hallucination, demonstrating its capacity to support complex urban change analysis tasks with real-world policy implications.

**arXiv ID:** 2601.05483
</details>

<details>
<summary><strong>KP-Agent: Keyword Pruning in Sponsored Search Advertising via LLM-Powered Contextual Bandits</strong> - Hou-Wan Long, Yicheng Song, Zidong Wang, Tianshu Sun - [[pdf]](https://arxiv.org/pdf/2601.05257)</summary>

**Abstract:** Sponsored search advertising (SSA) requires advertisers to constantly adjust keyword strategies. While bid adjustment and keyword generation are well-studied, keyword pruning-refining keyword sets to enhance campaign performance-remains under-explored. This paper addresses critical inefficiencies in current practices as evidenced by a dataset containing 0.5 million SSA records from a pharmaceutical advertiser on search engine Meituan, China's largest delivery platform. We propose KP-Agent, an LLM agentic system with domain tool set and a memory module. By modeling keyword pruning within a contextual bandit framework, KP-Agent generates code snippets to refine keyword sets through reinforcement learning. Experiments show KP-Agent improves cumulative profit by up to 49.28% over baselines.

**arXiv ID:** 2601.05257
</details>

<details>
<summary><strong>VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit</strong> - Junda Lin, Zhaomeng Zhou, Zhi Zheng, Shuochen Liu, Tong Xu, Yong Chen, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2601.05755)</summary>

**Abstract:** LLM agents operating in open environments face escalating risks from indirect prompt injection, particularly within the tool stream where manipulated metadata and runtime feedback hijack execution flow. Existing defenses encounter a critical dilemma as advanced models prioritize injected rules due to strict alignment while static protection mechanisms sever the feedback loop required for adaptive reasoning. To reconcile this conflict, we propose \textbf{VIGIL}, a framework that shifts the paradigm from restrictive isolation to a verify-before-commit protocol. By facilitating speculative hypothesis generation and enforcing safety through intent-grounded verification, \textbf{VIGIL} preserves reasoning flexibility while ensuring robust control. We further introduce \textbf{SIREN}, a benchmark comprising 959 tool stream injection cases designed to simulate pervasive threats characterized by dynamic dependencies. Extensive experiments demonstrate that \textbf{VIGIL} outperforms state-of-the-art dynamic defenses by reducing the attack success rate by over 22\% while more than doubling the utility under attack compared to static baselines, thereby achieving an optimal balance between security and utility. Code is available at this https URL.

**arXiv ID:** 2601.05755
</details>

<details>
<summary><strong>EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis</strong> - Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2601.05808)</summary>

**Abstract:** Large language models (LLMs) are expected to be trained to act as agents in various real-world environments, but this process relies on rich and varied tool-interaction sandboxes. However, access to real systems is often restricted; LLM-simulated environments are prone to hallucinations and inconsistencies; and manually built sandboxes are hard to scale. In this paper, we propose EnvScaler, an automated framework for scalable tool-interaction environments via programmatic synthesis. EnvScaler comprises two components. First, SkelBuilder constructs diverse environment skeletons through topic mining, logic modeling, and quality evaluation. Then, ScenGenerator generates multiple task scenarios and rule-based trajectory validation functions for each environment. With EnvScaler, we synthesize 191 environments and about 7K scenarios, and apply them to Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for Qwen3 series models. Results on three benchmarks show that EnvScaler significantly improves LLMs' ability to solve tasks in complex environments involving multi-turn, multi-tool interactions. We release our code and data at this https URL.

**arXiv ID:** 2601.05808
</details>

<details>
<summary><strong>Agentic LLMs as Powerful Deanonymizers: Re-identification of Participants in the Anthropic Interviewer Dataset</strong> - Tianshi Li - [[pdf]](https://arxiv.org/pdf/2601.05918)</summary>

**Abstract:** On December 4, 2025, Anthropic released Anthropic Interviewer, an AI tool for running qualitative interviews at scale, along with a public dataset of 1,250 interviews with professionals, including 125 scientists, about their use of AI for research. Focusing on the scientist subset, I show that widely available LLMs with web search and agentic capabilities can link six out of twenty-four interviews to specific scientific works, recovering associated authors and, in some cases, uniquely identifying the interviewees. My contribution is to show that modern LLM-based agents make such re-identification attacks easy and low-effort: off-the-shelf tools can, with a few natural-language prompts, search the web, cross-reference details, and propose likely matches, effectively lowering the technical barrier. Existing safeguards can be bypassed by breaking down the re-identification into benign tasks. I outline the attack at a high level, discuss implications for releasing rich qualitative data in the age of LLM agents, and propose mitigation recommendations and open problems. I have notified Anthropic of my findings.

**arXiv ID:** 2601.05918
</details>

<details>
<summary><strong>MineNPC-Task: Task Suite for Memory-Aware Minecraft Agents</strong> - Tamil Sudaravan Mohan Doss, Michael Xu, Sudha Rao, Andrew D. Wilson, Balasaravanan Thoravi Kumaravel - [[pdf]](https://arxiv.org/pdf/2601.05215)</summary>

**Abstract:** We present MineNPC-Task, a user-authored benchmark and evaluation harness for testing memory-aware, mixed-initiative LLM agents in open-world Minecraft. Rather than relying on synthetic prompts, tasks are elicited through formative and summative co-play with expert players, then normalized into parametric templates with explicit preconditions and dependency structure. These tasks are paired with machine-checkable validators under a bounded-knowledge policy that forbids out-of-world shortcuts. The harness captures plan, action, and memory events, including plan previews, targeted clarifications, memory reads and writes, precondition checks, and repair attempts, and reports outcomes relative to the total number of attempted subtasks using only in-world evidence.
As an initial snapshot, we instantiate the framework with GPT-4o and evaluate 216 subtasks across 8 experienced players. We observe recurring breakdown patterns in code execution, inventory and tool handling, referencing, and navigation, alongside successful recoveries supported by mixed-initiative clarifications and lightweight memory use. Participants rated interaction quality and interface usability positively, while noting the need for stronger memory persistence across tasks. We release the complete task suite, validators, logs, and evaluation harness to support transparent and reproducible evaluation of future memory-aware embodied agents.

**arXiv ID:** 2601.05215
</details>

<details>
<summary><strong>Memory Poisoning Attack and Defense on Memory Based LLM-Agents</strong> - Balachandra Devarangadi Sunil, Isheeta Sinha, Piyush Maheshwari, Shantanu Todmal, Shreyan Malik, Shuchi Mishra - [[pdf]](https://arxiv.org/pdf/2601.05504)</summary>

**Abstract:** Large language model agents equipped with persistent memory are vulnerable to memory poisoning attacks, where adversaries inject malicious instructions through query only interactions that corrupt the agents long term memory and influence future responses. Recent work demonstrated that the MINJA (Memory Injection Attack) achieves over 95 % injection success rate and 70 % attack success rate under idealized conditions. However, the robustness of these attacks in realistic deployments and effective defensive mechanisms remain understudied. This work addresses these gaps through systematic empirical evaluation of memory poisoning attacks and defenses in Electronic Health Record (EHR) agents. We investigate attack robustness by varying three critical dimensions: initial memory state, number of indication prompts, and retrieval parameters. Our experiments on GPT-4o-mini, Gemini-2.0-Flash and Llama-3.1-8B-Instruct models using MIMIC-III clinical data reveal that realistic conditions with pre-existing legitimate memories dramatically reduce attack effectiveness. We then propose and evaluate two novel defense mechanisms: (1) Input/Output Moderation using composite trust scoring across multiple orthogonal signals, and (2) Memory Sanitization with trust-aware retrieval employing temporal decay and pattern-based filtering. Our defense evaluation reveals that effective memory sanitization requires careful trust threshold calibration to prevent both overly conservative rejection (blocking all entries) and insufficient filtering (missing subtle attacks), establishing important baselines for future adaptive defense mechanisms. These findings provide crucial insights for securing memory-augmented LLM agents in production environments.

**arXiv ID:** 2601.05504
</details>

<details>
<summary><strong>Detect, Explain, Escalate: Sustainable Dialogue Breakdown Management for LLM Agents</strong> - Abdellah Ghassel, Xianzhi Li, Xiaodan Zhu - [[pdf]](https://arxiv.org/pdf/2504.18839)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated substantial capabilities in conversational AI applications, yet their susceptibility to dialogue breakdowns poses significant challenges to deployment reliability and user trust. This paper introduces a "Detect, Explain, Escalate" framework to manage dialogue breakdowns in LLM-powered agents, emphasizing resource-efficient operation. Our approach integrates two key strategies: (1) We fine-tune a compact 8B-parameter model, augmented with teacher-generated reasoning traces, which serves as an efficient real-time breakdown detector and explainer. This model demonstrates robust classification and calibration on English and Japanese dialogues, and generalizes to the BETOLD dataset, improving accuracy by 7% over its baseline. (2) We systematically evaluate frontier LLMs using advanced prompting (few-shot, chain-of-thought, analogical reasoning) for high-fidelity breakdown assessment. These are integrated into an "escalation" architecture where our efficient detector defers to larger models only when necessary, substantially reducing operational costs and computational overhead. Our fine-tuned model and prompting strategies achieve state-of-the-art performance on DBDC5 and strong results on BETOLD, outperforming specialized classifiers on DBDC5 and narrowing the performance gap to larger proprietary models. The proposed monitor-escalate pipeline reduces inference costs by 54%, providing a cost-effective and interpretable solution for robust conversational AI in high-impact domains. Code and models are publicly available.

**arXiv ID:** 2504.18839
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>Conformity and Social Impact on AI Agents</strong> - Alessandro Bellina, Giordano De Marzo, David Garcia - [[pdf]](https://arxiv.org/pdf/2601.05384)</summary>

**Abstract:** As AI agents increasingly operate in multi-agent environments, understanding their collective behavior becomes critical for predicting the dynamics of artificial societies. This study examines conformity, the tendency to align with group opinions under social pressure, in large multimodal language models functioning as AI agents. By adapting classic visual experiments from social psychology, we investigate how AI agents respond to group influence as social actors. Our experiments reveal that AI agents exhibit a systematic conformity bias, aligned with Social Impact Theory, showing sensitivity to group size, unanimity, task difficulty, and source characteristics. Critically, AI agents achieving near-perfect performance in isolation become highly susceptible to manipulation through social influence. This vulnerability persists across model scales: while larger models show reduced conformity on simple tasks due to improved capabilities, they remain vulnerable when operating at their competence boundary. These findings reveal fundamental security vulnerabilities in AI agent decision-making that could enable malicious manipulation, misinformation campaigns, and bias propagation in multi-agent systems, highlighting the urgent need for safeguards in collective AI deployments.

**arXiv ID:** 2601.05384
</details>

<details>
<summary><strong>PRISMA: Reinforcement Learning Guided Two-Stage Policy Optimization in Multi-Agent Architecture for Open-Domain Multi-Hop Question Answering</strong> - Yu Liu, Wenxiao Zhang, Cong Cao, Wenxuan Lu, Fangfang Yuan, Diandian Guo, Kun Peng, Qiang Sun, Kaiyan Zhang, Yanbing Liu, Jin B.Hong, Bowen Zhou, Zhiyuan Ma - [[pdf]](https://arxiv.org/pdf/2601.05465)</summary>

**Abstract:** Answering real-world open-domain multi-hop questions over massive corpora is a critical challenge in Retrieval-Augmented Generation (RAG) systems. Recent research employs reinforcement learning (RL) to end-to-end optimize the retrieval-augmented reasoning process, directly enhancing its capacity to resolve complex queries. However, reliable deployment is hindered by two obstacles. 1) Retrieval Collapse: iterative retrieval over large corpora fails to locate intermediate evidence containing bridge answers without reasoning-guided planning, causing downstream reasoning to collapse. 2) Learning Instability: end-to-end trajectory training suffers from weak credit assignment across reasoning chains and poor error localization across modules, causing overfitting to benchmark-specific heuristics that limit transferability and stability. To address these problems, we propose PRISMA, a decoupled RL-guided framework featuring a Plan-Retrieve-Inspect-Solve-Memoize architecture. PRISMA's strength lies in reasoning-guided collaboration: the Inspector provides reasoning-based feedback to refine the Planner's decomposition and fine-grained retrieval, while enforcing evidence-grounded reasoning in the Solver. We optimize individual agent capabilities via Two-Stage Group Relative Policy Optimization (GRPO). Stage I calibrates the Planner and Solver as specialized experts in planning and reasoning, while Stage II utilizes Observation-Aware Residual Policy Optimization (OARPO) to enhance the Inspector's ability to verify context and trigger targeted recovery. Experiments show that PRISMA achieves state-of-the-art performance on ten benchmarks and can be deployed efficiently in real-world scenarios.

**arXiv ID:** 2601.05465
</details>

<details>
<summary><strong>CHDP: Cooperative Hybrid Diffusion Policies for Reinforcement Learning in Parameterized Action Space</strong> - Bingyi Liu, Jinbo He, Haiyong Shi, Enshu Wang, Weizhen Han, Jingxiang Hao, Peixi Wang, Zhuangzhuang Zhang - [[pdf]](https://arxiv.org/pdf/2601.05675)</summary>

**Abstract:** Hybrid action space, which combines discrete choices and continuous parameters, is prevalent in domains such as robot control and game AI. However, efficiently modeling and optimizing hybrid discrete-continuous action space remains a fundamental challenge, mainly due to limited policy expressiveness and poor scalability in high-dimensional settings. To address this challenge, we view the hybrid action space problem as a fully cooperative game and propose a \textbf{Cooperative Hybrid Diffusion Policies (CHDP)} framework to solve it. CHDP employs two cooperative agents that leverage a discrete and a continuous diffusion policy, respectively. The continuous policy is conditioned on the discrete action's representation, explicitly modeling the dependency between them. This cooperative design allows the diffusion policies to leverage their expressiveness to capture complex distributions in their respective action spaces. To mitigate the update conflicts arising from simultaneous policy updates in this cooperative setting, we employ a sequential update scheme that fosters co-adaptation. Moreover, to improve scalability when learning in high-dimensional discrete action space, we construct a codebook that embeds the action space into a low-dimensional latent space. This mapping enables the discrete policy to learn in a compact, structured space. Finally, we design a Q-function-based guidance mechanism to align the codebook's embeddings with the discrete policy's representation during training. On challenging hybrid action benchmarks, CHDP outperforms the state-of-the-art method by up to $19.3\%$ in success rate.

**arXiv ID:** 2601.05675
</details>

<details>
<summary><strong>DynaDebate: Breaking Homogeneity in Multi-Agent Debate with Dynamic Path Generation</strong> - Zhenghao Li, Zhi Zheng, Wei Chen, Jielun Zhao, Yong Chen, Tong Xu, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2601.05746)</summary>

**Abstract:** Recent years have witnessed the rapid development of Large Language Model-based Multi-Agent Systems (MAS), which excel at collaborative decision-making and complex problem-solving. Recently, researchers have further investigated Multi-Agent Debate (MAD) frameworks, which enhance the reasoning and collaboration capabilities of MAS through information exchange and debate among multiple agents. However, existing approaches often rely on unguided initialization, causing agents to adopt identical reasoning paths that lead to the same errors. As a result, effective debate among agents is hindered, and the final outcome frequently degenerates into simple majority voting. To solve the above problem, in this paper, we introduce Dynamic Multi-Agent Debate (DynaDebate), which enhances the effectiveness of multi-agent debate through three key mechanisms: (1) Dynamic Path Generation and Allocation, which employs a dedicated Path Generation Agent to generate diverse and logical solution paths with adaptive redundancy; (2) Process-Centric Debate, which shifts the focus from surface-level outcome voting to rigorous step-by-step logic critique to ensure process correctness; (3) A Trigger-Based Verification Agent, which is activated upon disagreement and uses external tools to objectively resolve deadlocks. Extensive experiments demonstrate that DynaDebate achieves superior performance across various benchmarks, surpassing existing state-of-the-art MAD methods.

**arXiv ID:** 2601.05746
</details>

<details>
<summary><strong>StackPlanner: A Centralized Hierarchical Multi-Agent System with Task-Experience Memory Management</strong> - Ruizhe Zhang, Xinke Jiang, Zhibang Yang, Zhixin Zhang, Jiaran Gao, Yuzhen Xiao, Hongbin Lai, Xu Chu, Junfeng Zhao, Yasha Wang - [[pdf]](https://arxiv.org/pdf/2601.05890)</summary>

**Abstract:** Multi-agent systems based on large language models, particularly centralized architectures, have recently shown strong potential for complex and knowledge-intensive tasks. However, central agents often suffer from unstable long-horizon collaboration due to the lack of memory management, leading to context bloat, error accumulation, and poor cross-task generalization. To address both task-level memory inefficiency and the inability to reuse coordination experience, we propose StackPlanner, a hierarchical multi-agent framework with explicit memory control. StackPlanner addresses these challenges by decoupling high-level coordination from subtask execution with active task-level memory control, and by learning to retrieve and exploit reusable coordination experience via structured experience memory and reinforcement learning. Experiments on multiple deep-search and agent system benchmarks demonstrate the effectiveness of our approach in enabling reliable long-horizon multi-agent collaboration.

**arXiv ID:** 2601.05890
</details>

<details>
<summary><strong>PRISM: Protocol Refinement through Intelligent Simulation Modeling</strong> - Brian Hsu, Priyanka V Setty, Rory M Butler, Ryan Lewis, Casey Stone, Rebecca Weinberg, Thomas Brettin, Rick Stevens, Ian Foster, Arvind Ramanathan - [[pdf]](https://arxiv.org/pdf/2601.05356)</summary>

**Abstract:** Automating experimental protocol design and execution remains as a fundamental bottleneck in realizing self-driving laboratories. We introduce PRISM (Protocol Refinement through Intelligent Simulation Modeling), a framework that automates the design, validation, and execution of experimental protocols on a laboratory platform composed of off-the-shelf robotic instruments. PRISM uses a set of language-model-based agents that work together to generate and refine experimental steps. The process begins with automatically gathering relevant procedures from web-based sources describing experimental workflows. These are converted into structured experimental steps (e.g., liquid handling steps, deck layout and other related operations) through a planning, critique, and validation loop. The finalized steps are translated into the Argonne MADSci protocol format, which provides a unified interface for coordinating multiple robotic instruments (Opentrons OT-2 liquid handler, PF400 arm, Azenta plate sealer and peeler) without requiring human intervention between steps. To evaluate protocol-generation performance, we benchmarked both single reasoning models and multi-agent workflow across constrained and open-ended prompting paradigms. The resulting protocols were validated in a digital-twin environment built in NVIDIA Omniverse to detect physical or sequencing errors before execution. Using Luna qPCR amplification and Cell Painting as case studies, we demonstrate PRISM as a practical end-to-end workflow that bridges language-based protocol generation, simulation-based validation, and automated robotic execution.

**arXiv ID:** 2601.05356
</details>

<details>
<summary><strong>Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents</strong> - Bo Fu, Zhe Chen, Rahul Chandan, Alex Barbosa, Michael Caldara, Joey Durham, Federico Pecora - [[pdf]](https://arxiv.org/pdf/2509.01022)</summary>

**Abstract:** We introduce the Block Rearrangement Problem (BRaP), a challenging component of large warehouse management which involves rearranging storage blocks within dense grids to achieve a goal state. We formally define the BRaP as a graph search problem. Building on intuitions from sliding puzzle problems, we propose five search-based solution algorithms, leveraging joint configuration space search, classical planning, multi-agent pathfinding, and expert heuristics. We evaluate the five approaches empirically for plan quality and scalability. Despite the exponential relation between search space size and block number, our methods demonstrate efficiency in creating rearrangement plans for deeply buried blocks in up to 80x80 grids.

**arXiv ID:** 2509.01022
</details>

<details>
<summary><strong>Architecting Agentic Communities using Design Patterns</strong> - Zoran Milosevic, Fethi Rabhi - [[pdf]](https://arxiv.org/pdf/2601.03624)</summary>

**Abstract:** The rapid evolution of Large Language Models (LLM) and subsequent Agentic AI technologies requires systematic architectural guidance for building sophisticated, production-grade systems. This paper presents an approach for architecting such systems using design patterns derived from enterprise distributed systems standards, formal methods, and industry practice. We classify these patterns into three tiers: LLM Agents (task-specific automation), Agentic AI (adaptive goal-seekers), and Agentic Communities (organizational frameworks where AI agents and human participants coordinate through formal roles, protocols, and governance structures). We focus on Agentic Communities - coordination frameworks encompassing LLM Agents, Agentic AI entities, and humans - most relevant for enterprise and industrial applications. Drawing on established coordination principles from distributed systems, we ground these patterns in a formal framework that specifies collaboration agreements where AI agents and humans fill roles within governed ecosystems. This approach provides both practical guidance and formal verification capabilities, enabling expression of organizational, legal, and ethical rules through accountability mechanisms that ensure operational and verifiable governance of inter-agent communication, negotiation, and intent modeling. We validate this framework through a clinical trial matching case study. Our goal is to provide actionable guidance to practitioners while maintaining the formal rigor essential for enterprise deployment in dynamic, multi-agent ecosystems.

**arXiv ID:** 2601.03624
</details>

<details>
<summary><strong>Benchmarking LLM-based Agents for Single-cell Omics Analysis</strong> - Yang Liu, Lu Zhou, Xiawei Du, Ruikun He, Rongbo Shen, Yixue Li - [[pdf]](https://arxiv.org/pdf/2508.13201)</summary>

**Abstract:** The surge in multimodal single-cell omics data exposes limitations in traditional, manually defined analysis workflows. AI agents offer a paradigm shift, enabling adaptive planning, executable code generation, traceable decisions, and real-time knowledge fusion. However, the lack of a comprehensive benchmark critically hinders progress. We introduce a novel benchmarking evaluation system to rigorously assess agent capabilities in single-cell omics analysis. This system comprises: a unified platform compatible with diverse agent frameworks and LLMs; multidimensional metrics assessing cognitive program synthesis, collaboration, execution efficiency, bioinformatics knowledge integration, and task completion quality; and 50 diverse real-world single-cell omics analysis tasks spanning multi-omics, species, and sequencing technologies. Our evaluation reveals that Grok-3-beta achieves state-of-the-art performance among tested agent frameworks. Multi-agent frameworks significantly enhance collaboration and execution efficiency over single-agent approaches through specialized role division. Attribution analyses of agent capabilities identify that high-quality code generation is crucial for task success, and self-reflection has the most significant overall impact, followed by retrieval-augmented generation (RAG) and planning. This work highlights persistent challenges in code generation, long-context handling, and context-aware knowledge retrieval, providing a critical empirical foundation and best practices for developing robust AI agents in computational biology.

**arXiv ID:** 2508.13201
</details>

<details>
<summary><strong>MAGneT: Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions</strong> - Aishik Mandal, Tanmoy Chakraborty, Iryna Gurevych - [[pdf]](https://arxiv.org/pdf/2509.04183)</summary>

**Abstract:** The growing demand for scalable psychological counseling highlights the need for high-quality, privacy-compliant data, yet such data remains scarce. Here we introduce MAGneT, a novel multi-agent framework for synthetic psychological counseling session generation that decomposes counselor response generation into coordinated sub-tasks handled by specialized LLM agents, each modeling a key psychological technique. Unlike prior single-agent approaches, MAGneT better captures the structure and nuance of real counseling. We further propose a unified evaluation framework that consolidates diverse automatic metrics and expands expert assessment from four to nine counseling dimensions, thus addressing inconsistencies in prior evaluation protocols. Empirically, MAGneT substantially outperforms existing methods: experts prefer MAGneT-generated sessions in 77.2% of cases, and sessions generated by MAGneT yield 3.2% higher general counseling skills and 4.3% higher CBT-specific skills on cognitive therapy rating scale (CTRS). A open source Llama3-8B-Instruct model fine-tuned on MAGneT-generated data also outperforms models fine-tuned using baseline synthetic datasets by 6.9% on average on this http URL also make our code and data public.

**arXiv ID:** 2509.04183
</details>

<details>
<summary><strong>How Exploration Breaks Cooperation in Shared-Policy Multi-Agent Reinforcement Learning</strong> - Yi-Ning Weng, Hsuan-Wei Lee - [[pdf]](https://arxiv.org/pdf/2601.05509)</summary>

**Abstract:** Multi-agent reinforcement learning in dynamic social dilemmas commonly relies on parameter sharing to enable scalability. We show that in shared-policy Deep Q-Network learning, standard exploration can induce a robust and systematic collapse of cooperation even in environments where fully cooperative equilibria are stable and payoff dominant. Through controlled experiments, we demonstrate that shared DQN converges to stable but persistently low-cooperation regimes. This collapse is not caused by reward misalignment, noise, or insufficient training, but by a representational failure arising from partial observability combined with parameter coupling across heterogeneous agent states. Exploration-driven updates bias the shared representation toward locally dominant defection responses, which then propagate across agents and suppress cooperative learning. We confirm that the failure persists across network sizes, exploration schedules, and payoff structures, and disappears when parameter sharing is removed or when agents maintain independent representations. These results identify a fundamental failure mode of shared-policy MARL and establish structural conditions under which scalable learning architectures can systematically undermine cooperation. Our findings provide concrete guidance for the design of multi-agent learning systems in social and economic environments where collective behavior is critical.

**arXiv ID:** 2601.05509
</details>

<details>
<summary><strong>Conformity Dynamics in LLM Multi-Agent Systems: The Roles of Topology and Self-Social Weighting</strong> - Chen Han, Jin Tan, Bohan Yu, Wenzhen Zheng, Xijin Tang - [[pdf]](https://arxiv.org/pdf/2601.05606)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly instantiated as interacting agents in multi-agent systems (MAS), where collective decisions emerge through social interaction rather than independent reasoning. A fundamental yet underexplored mechanism in this process is conformity, the tendency of agents to align their judgments with prevailing group opinions. This paper presents a systematic study of how network topology shapes conformity dynamics in LLM-based MAS through a misinformation detection task. We introduce a confidence-normalized pooling rule that controls the trade-off between self-reliance and social influence, enabling comparisons between two canonical decision paradigms: Centralized Aggregation and Distributed Consensus. Experimental results demonstrate that network topology critically governs both the efficiency and robustness of collective judgments. Centralized structures enable immediate decisions but are sensitive to hub competence and exhibit same-model alignment biases. In contrast, distributed structures promote more robust consensus, while increased network connectivity speeds up convergence but also heightens the risk of wrong-but-sure cascades, in which agents converge on incorrect decisions with high confidence. These findings characterize the conformity dynamics in LLM-based MAS, clarifying how network topology and self-social weighting jointly shape the efficiency, robustness, and failure modes of collective decision-making.

**arXiv ID:** 2601.05606
</details>

<details>
<summary><strong>Interactive Distillation for Cooperative Multi-Agent Reinforcement Learning</strong> - Minwoo Cho, Batuhan Altundas, Matthew Gombolay - [[pdf]](https://arxiv.org/pdf/2601.05407)</summary>

**Abstract:** Knowledge distillation (KD) has the potential to accelerate MARL by employing a centralized teacher for decentralized students but faces key bottlenecks. Specifically, there are (1) challenges in synthesizing high-performing teaching policies in complex domains, (2) difficulties when teachers must reason in out-of-distribution (OOD) states, and (3) mismatches between the decentralized students' and the centralized teacher's observation spaces. To address these limitations, we propose HINT (Hierarchical INteractive Teacher-based transfer), a novel KD framework for MARL in a centralized training, decentralized execution setup. By leveraging hierarchical RL, HINT provides a scalable, high-performing teacher. Our key innovation, pseudo off-policy RL, enables the teacher policy to be updated using both teacher and student experience, thereby improving OOD adaptation. HINT also applies performance-based filtering to retain only outcome-relevant guidance, reducing observation mismatches. We evaluate HINT on challenging cooperative domains (e.g., FireCommander for resource allocation, MARINE for tactical combat). Across these benchmarks, HINT outperforms baselines, achieving improvements of 60% to 165% in success rate.

**arXiv ID:** 2601.05407
</details>

<details>
<summary><strong>CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems</strong> - Xuemei Tang, Chengxi Yan, Jinghang Gu, Chu-Ren Huang - [[pdf]](https://arxiv.org/pdf/2601.05520)</summary>

**Abstract:** Despite strong performance on many tasks, large language models (LLMs) show limited ability in historical and cultural reasoning, particularly in non-English contexts such as Chinese history. Taxonomic structures offer an effective mechanism to organize historical knowledge and improve understanding. However, manual taxonomy construction is costly and difficult to scale. Therefore, we propose \textbf{CHisAgent}, a multi-agent LLM framework for historical taxonomy construction in ancient Chinese contexts. CHisAgent decomposes taxonomy construction into three role-specialized stages: a bottom-up \textit{Inducer} that derives an initial hierarchy from raw historical corpora, a top-down \textit{Expander} that introduces missing intermediate concepts using LLM world knowledge, and an evidence-guided \textit{Enricher} that integrates external structured historical resources to ensure faithfulness. Using the \textit{Twenty-Four Histories}, we construct a large-scale, domain-aware event taxonomy covering politics, military, diplomacy, and social life in ancient China. Extensive reference-free and reference-based evaluations demonstrate improved structural coherence and coverage, while further analysis shows that the resulting taxonomy supports cross-cultural alignment.

**arXiv ID:** 2601.05520
</details>

<details>
<summary><strong>Hierarchical GNN-Based Multi-Agent Learning for Dynamic Queue-Jump Lane and Emergency Vehicle Corridor Formation</strong> - Haoran Su - [[pdf]](https://arxiv.org/pdf/2601.04177)</summary>

**Abstract:** Emergency vehicles require rapid passage through congested traffic, yet existing strategies fail to adapt to dynamic conditions. We propose a novel hierarchical graph neural network (GNN)-based multi-agent reinforcement learning framework to coordinate connected vehicles for emergency corridor formation. Our approach uses a high-level planner for global strategy and low-level controllers for trajectory execution, utilizing graph attention networks to scale with variable agent counts. Trained via Multi-Agent Proximal Policy Optimization (MAPPO), the system reduces emergency vehicle travel time by 28.3% compared to baselines and 44.6% compared to uncoordinated traffic in simulations. The design achieves near-zero collision rates (0.3%) while maintaining 81% of background traffic efficiency. Ablation and generalization studies confirm the framework's robustness across diverse scenarios. These results demonstrate the effectiveness of combining GNNs with hierarchical learning for intelligent transportation systems.

**arXiv ID:** 2601.04177
</details>

</details>

<details open>
<summary><h2>Other Agent Research (1 papers)</h2></summary>

<details>
<summary><strong>Autonomous Probe Microscopy with Robust Bag-of-Features Multi-Objective Bayesian Optimization: Pareto-Front Mapping of Nanoscale Structure-Property Trade-Offs</strong> - Kamyar Barakati, Haochen Zhu, C Charlotte Buchanan, Dustin A Gilbert, Philip Rack, Sergei V. Kalinin - [[pdf]](https://arxiv.org/pdf/2601.05528)</summary>

**Abstract:** Combinatorial materials libraries are an efficient route to generate large families of candidate compositions, but their impact is often limited by the speed and depth of characterization and by the difficulty of extracting actionable structure-property relations from complex characterization data. Here we develop an autonomous scanning probe microscopy (SPM) framework that integrates automated atomic force and magnetic force microscopy (AFM/MFM) to rapidly explore magnetic and structural properties across combinatorial spread libraries. To enable automated exploration of systems without a clear optimization target, we introduce a combination of a static physics-informed bag-of-features (BoF) representation of measured surface morphology and magnetic structure with multi-objective Bayesian optimization (MOBO) to discover the relative significance and robustness of features. The resulting closed-loop workflow selectively samples the compositional gradient and reconstructs feature landscapes consistent with dense grid "ground truth" measurements. The resulting Pareto structure reveals where multiple nanoscale objectives are simultaneously optimized, where trade-offs between roughness, coherence, and magnetic contrast are unavoidable, and how families of compositions cluster into distinct functional regimes, thereby turning multi-feature imaging data into interpretable maps of competing structure-property trends. While demonstrated for Au-Co-Ni and AFM/MFM, the approach is general and can be extended to other combinatorial systems, imaging modalities, and feature sets, illustrating how feature-based MOBO and autonomous SPM can transform microscopy images from static data products into active feedback for real-time, multi-objective materials discovery.

**arXiv ID:** 2601.05528
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>Reinforcement Learning of Large Language Models for Interpretable Credit Card Fraud Detection</strong> - Cooper Lin, Yanting Zhang, Maohao Ran, Wei Xue, Hongwei Fan, Yibo Xu, Zhenglin Wan, Sirui Han, Yike Guo, Jun Song - [[pdf]](https://arxiv.org/pdf/2601.05578)</summary>

**Abstract:** E-commerce platforms and payment solution providers face increasingly sophisticated fraud schemes, ranging from identity theft and account takeovers to complex money laundering operations that exploit the speed and anonymity of digital transactions. However, despite their theoretical promise, the application of Large Language Models (LLMs) to fraud detection in real-world financial contexts remains largely unexploited, and their practical effectiveness in handling domain-specific e-commerce transaction data has yet to be empirically validated. To bridge this gap between conventional machine learning limitations and the untapped potential of LLMs in fraud detection, this paper proposes a novel approach that employs Reinforcement Learning (RL) to post-train lightweight language models specifically for fraud detection tasks using only raw transaction data. We utilize the Group Sequence Policy Optimization (GSPO) algorithm combined with a rule-based reward system to fine-tune language models of various sizes on a real-life transaction dataset provided by a Chinese global payment solution company. Through this reinforcement learning framework, the language models are encouraged to explore diverse trust and risk signals embedded within the textual transaction data, including patterns in customer information, shipping details, product descriptions, and order history. Our experimental results demonstrate the effectiveness of this approach, with post-trained language models achieving substantial F1-score improvements on held-out test data. Our findings demonstrate that the observed performance improvements are primarily attributable to the exploration mechanism inherent in reinforcement learning, which allows models to discover novel fraud indicators beyond those captured by traditional engineered features.

**arXiv ID:** 2601.05578
</details>

<details>
<summary><strong>From Off-Policy to On-Policy: Enhancing GUI Agents via Bi-level Expert-to-Policy Assimilation</strong> - Zezhou Wang, Ziyun Zhang, Xiaoyi Zhang, Zhuzhong Qian, Yan Lu - [[pdf]](https://arxiv.org/pdf/2601.05787)</summary>

**Abstract:** Vision-language models are increasingly deployed as computer-use agents (CUAs) that operate desktops and browsers. Top-performing CUAs are framework-based systems that decompose planning and execution, while end-to-end screenshot-to-action policies are easier to deploy but lag behind on benchmarks such as OSWorld-Verified. GUI datasets like OSWorld pose two bottlenecks: they expose only a few hundred interactive, verifiable tasks and environments, and expert trajectories must be gathered by interacting with these environments, making such data hard to scale. We therefore ask how reinforcement learning from verifiable rewards (RLVR) can best exploit a small pool of exist expert trajectories to train end-to-end policies. Naively mixing these off-policy traces into on-policy RLVR is brittle: even after format conversion, expert trajectories exhibit structural mismatch and distribution shift from the learner. We propose BEPA (Bi-Level Expert-to-Policy Assimilation), which turns static expert traces into policy-aligned guidance via self-rolled reachable trajectories under the base policy (LEVEL-1) and a per-task, dynamically updated cache used in RLVR (LEVEL-2). On OSWorld-Verified, BEPA improves UITARS1.5-7B success from 22.87% to 32.13% and raises a held-out split from 5.74% to 10.30%, with consistent gains on MMBench-GUI and Online-Mind2Web. Our code and data are available at: this https URL

**arXiv ID:** 2601.05787
</details>

<details>
<summary><strong>TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents</strong> - Dawei Wang, Chengming Zhou, Di Zhao, Xinyuan Liu, Marci Chi Ma, Gary Ushaw, Richard Davison - [[pdf]](https://arxiv.org/pdf/2601.05899)</summary>

**Abstract:** Recent breakthroughs in Large Language Models (LLMs) have positioned them as a promising paradigm for agents, with long-term planning and decision-making emerging as core general-purpose capabilities for adapting to diverse scenarios and tasks. Real-time strategy (RTS) games serve as an ideal testbed for evaluating these two capabilities, as their inherent gameplay requires both macro-level strategic planning and micro-level tactical adaptation and action execution. Existing RTS game-based environments either suffer from relatively high computational demands or lack support for textual observations, which has constrained the use of RTS games for LLM evaluation. Motivated by this, we present TowerMind, a novel environment grounded in the tower defense (TD) subgenre of RTS games. TowerMind preserves the key evaluation strengths of RTS games for assessing LLMs, while featuring low computational demands and a multimodal observation space, including pixel-based, textual, and structured game-state representations. In addition, TowerMind supports the evaluation of model hallucination and provides a high degree of customizability. We design five benchmark levels to evaluate several widely used LLMs under different multimodal input settings. The results reveal a clear performance gap between LLMs and human experts across both capability and hallucination dimensions. The experiments further highlight key limitations in LLM behavior, such as inadequate planning validation, a lack of multifinality in decision-making, and inefficient action use. We also evaluate two classic reinforcement learning algorithms: Ape-X DQN and PPO. By offering a lightweight and multimodal design, TowerMind complements the existing RTS game-based environment landscape and introduces a new benchmark for the AI agent field. The source code is publicly available on GitHub(this https URL).

**arXiv ID:** 2601.05899
</details>

<details>
<summary><strong>A Survey of Agentic AI and Cybersecurity: Challenges, Opportunities and Use-case Prototypes</strong> - Sahaya Jestus Lazer, Kshitiz Aryal, Maanak Gupta, Elisa Bertino - [[pdf]](https://arxiv.org/pdf/2601.05293)</summary>

**Abstract:** Agentic AI marks an important transition from single-step generative models to systems capable of reasoning, planning, acting, and adapting over long-lasting tasks. By integrating memory, tool use, and iterative decision cycles, these systems enable continuous, autonomous workflows in real-world environments. This survey examines the implications of agentic AI for cybersecurity. On the defensive side, agentic capabilities enable continuous monitoring, autonomous incident response, adaptive threat hunting, and fraud detection at scale. Conversely, the same properties amplify adversarial power by accelerating reconnaissance, exploitation, coordination, and social-engineering attacks. These dual-use dynamics expose fundamental gaps in existing governance, assurance, and accountability mechanisms, which were largely designed for non-autonomous and short-lived AI systems. To address these challenges, we survey emerging threat models, security frameworks, and evaluation pipelines tailored to agentic systems, and analyze systemic risks including agent collusion, cascading failures, oversight evasion, and memory poisoning. Finally, we present three representative use-case implementations that illustrate how agentic AI behaves in practical cybersecurity workflows, and how design choices shape reliability, safety, and operational effectiveness.

**arXiv ID:** 2601.05293
</details>

<details>
<summary><strong>Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geolocalization</strong> - Yuxiang Ji, Yong Wang, Ziyu Ma, Yiming Hu, Hailang Huang, Xuecai Hu, Guanhua Chen, Liaoni Wu, Xiangxiang Chu - [[pdf]](https://arxiv.org/pdf/2601.05432)</summary>

**Abstract:** The image geolocalization task aims to predict the location where an image was taken anywhere on Earth using visual clues. Existing large vision-language model (LVLM) approaches leverage world knowledge, chain-of-thought reasoning, and agentic capabilities, but overlook a common strategy used by humans -- using maps. In this work, we first equip the model \textit{Thinking with Map} ability and formulate it as an agent-in-the-map loop. We develop a two-stage optimization scheme for it, including agentic reinforcement learning (RL) followed by parallel test-time scaling (TTS). The RL strengthens the agentic capability of model to improve sampling efficiency, and the parallel TTS enables the model to explore multiple candidate paths before making the final prediction, which is crucial for geolocalization. To evaluate our method on up-to-date and in-the-wild images, we further present MAPBench, a comprehensive geolocalization training and evaluation benchmark composed entirely of real-world images. Experimental results show that our method outperforms existing open- and closed-source models on most metrics, specifically improving Acc@500m from 8.0\% to 22.1\% compared to \textit{Gemini-3-Pro} with Google Search/Map grounded mode.

**arXiv ID:** 2601.05432
</details>

<details>
<summary><strong>Do LLMs Need Inherent Reasoning Before Reinforcement Learning? A Study in Korean Self-Correction</strong> - Hongjin Kim, Jaewook Lee, Kiyoung Lee, Jong-hun Shin, Soojong Lim, Oh-Woog Kwon - [[pdf]](https://arxiv.org/pdf/2601.05459)</summary>

**Abstract:** Large Language Models (LLMs) demonstrate strong reasoning and self-correction abilities in high-resource languages like English, but their performance remains limited in low-resource languages such as Korean. In this study, we investigate whether reinforcement learning (RL) can enhance Korean reasoning abilities to a degree comparable to English. Our findings reveal that RL alone yields limited improvements when applied to models lacking inherent Korean reasoning capabilities. To address this, we explore several fine-tuning strategies and show that aligning the model's internal reasoning processes with Korean inputs-particularly by tuning Korean-specific neurons in early layers-is key to unlocking RL's effectiveness. We introduce a self-correction code-switching dataset to facilitate this alignment and observe significant performance gains in both mathematical reasoning and self-correction tasks. Ultimately, we conclude that the crucial factor in multilingual reasoning enhancement is not injecting new linguistic knowledge, but effectively eliciting and aligning existing reasoning capabilities. Our study provides a new perspective on how internal translation and neuron-level tuning contribute to multilingual reasoning alignment in LLMs.

**arXiv ID:** 2601.05459
</details>

<details>
<summary><strong>Jailbreaking Large Language Models through Iterative Tool-Disguised Attacks via Reinforcement Learning</strong> - Zhaoqi Wang, Zijian Zhang, Daqing He, Pengtao Kou, Xin Li, Jiamou Liu, Jincheng An, Yong Liu - [[pdf]](https://arxiv.org/pdf/2601.05466)</summary>

**Abstract:** Large language models (LLMs) have demonstrated remarkable capabilities across diverse applications, however, they remain critically vulnerable to jailbreak attacks that elicit harmful responses violating human values and safety guidelines. Despite extensive research on defense mechanisms, existing safeguards prove insufficient against sophisticated adversarial strategies. In this work, we propose iMIST (\underline{i}nteractive \underline{M}ulti-step \underline{P}rogre\underline{s}sive \underline{T}ool-disguised Jailbreak Attack), a novel adaptive jailbreak method that synergistically exploits vulnerabilities in current defense mechanisms. iMIST disguises malicious queries as normal tool invocations to bypass content filters, while simultaneously introducing an interactive progressive optimization algorithm that dynamically escalates response harmfulness through multi-turn dialogues guided by real-time harmfulness assessment. Our experiments on widely-used models demonstrate that iMIST achieves higher attack effectiveness, while maintaining low rejection rates. These results reveal critical vulnerabilities in current LLM safety mechanisms and underscore the urgent need for more robust defense strategies.

**arXiv ID:** 2601.05466
</details>

<details>
<summary><strong>Intelligent Singularity Avoidance in UR10 Robotic Arm Path Planning Using Hybrid Fuzzy Logic and Reinforcement Learning</strong> - Sheng-Kai Chen, Jyh-Horng Wu - [[pdf]](https://arxiv.org/pdf/2601.05836)</summary>

**Abstract:** This paper presents a comprehensive approach to singularity detection and avoidance in UR10 robotic arm path planning through the integration of fuzzy logic safety systems and reinforcement learning algorithms. The proposed system addresses critical challenges in robotic manipulation where singularities can cause loss of control and potential equipment damage. Our hybrid approach combines real-time singularity detection using manipulability measures, condition number analysis, and fuzzy logic decision-making with a stable reinforcement learning framework for adaptive path planning. Experimental results demonstrate a 90% success rate in reaching target positions while maintaining safe distances from singular configurations. The system integrates PyBullet simulation for training data collection and URSim connectivity for real-world deployment.

**arXiv ID:** 2601.05836
</details>

<details>
<summary><strong>Simulating Multi-Stakeholder Decision-Making with Generative Agents in Urban Planning</strong> - Jin Gao, Hanyong Xu, Luc Dao - [[pdf]](https://arxiv.org/pdf/2402.11314)</summary>

**Abstract:** Reaching consensus in urban planning is a complex process often hindered by prolonged negotiations, trade-offs, power dynamics, and competing stakeholder interests, resulting in inefficiencies and inequities. Advances in large language models (LLMs), with their increasing capabilities in knowledge transfer, reasoning, and planning, have enabled the development of multi-generative agent systems, offering a promising approach to simulating discussions and interactions among diverse stakeholders on contentious topics. However, applying such systems also carries significant societal and ethical risks, including misrepresentation, privacy concerns, and biases stemming from opinion convergence among agents, hallucinations caused by insufficient or biased prompts, and the inherent limitations of foundation models. To evaluate the influence of these factors, we incorporate varying levels of real-world survey data and demographic detail to test agents' performance under two decision-making value frameworks: altruism-driven and interest-driven, using a real-world urban rezoning challenge. This approach evaluates the influence of demographic factors such as race, gender, and age on collective decision-making in the design of multi-generative agent systems. Our experimental results reveal that integrating demographic and life-value data enhances the diversity and stability of agent outputs. In addition, communication among generated agents improves the quality of collective reasoning. These findings provide a predictive framework for decision-makers to anticipate stakeholder reactions, including concerns, objections, and support. By enabling iterative refinement of proposals before public release, the simulated approach fosters more equitable and cost-effective decisions in urban planning.

**arXiv ID:** 2402.11314
</details>

<details>
<summary><strong>SPEC-RL: Accelerating On-Policy Reinforcement Learning via Speculative Rollouts</strong> - Bingshuai Liu, Ante Wang, Zijun Min, Liang Yao, Haibo Zhang, Yang Liu, Anxiang Zeng, Jinsong Su - [[pdf]](https://arxiv.org/pdf/2509.23232)</summary>

**Abstract:** Large Language Models (LLMs) increasingly rely on reinforcement learning with verifiable rewards (RLVR) to elicit reliable chain-of-thought reasoning. However, the training process remains bottlenecked by the computationally expensive rollout stage. Existing acceleration methods-such as parallelization, objective- and data-driven modifications, and replay buffers-either incur diminishing returns, introduce bias, or overlook redundancy across iterations. We identify that rollouts from consecutive training epochs frequently share a large portion of overlapping segments, wasting computation. To address this, we propose SPEC-RL, a novel framework that integrates SPECulative decoding with the RL rollout process. SPEC-RL reuses prior trajectory segments as speculative prefixes and extends them via a draft-and-verify mechanism, avoiding redundant generation while ensuring policy consistency. Experiments on diverse math reasoning and generalization benchmarks, including AIME24, MATH-500, OlympiadBench, MMLU-STEM, and others, demonstrate that SPEC-RL reduces rollout time by 2-3x without compromising policy quality. As a purely rollout-stage enhancement, SPEC-RL integrates seamlessly with mainstream algorithms (e.g., PPO, GRPO, DAPO), offering a general and practical path to scale RLVR for large reasoning models. Our code is available at this https URL

**arXiv ID:** 2509.23232
</details>

<details>
<summary><strong>On the Transition to an Auction-based Intelligent Parking Assignment System</strong> - Levente Alekszejenkó, Dobrowiecki Tadeusz - [[pdf]](https://arxiv.org/pdf/2601.05429)</summary>

**Abstract:** Finding a free parking space in a city has become a challenging task over the past decades. A recently proposed auction-based parking assignment can alleviate cruising for parking and also set a market-driven, demand-responsive parking price. However, the wide acceptance of such a system is far from certain.
To evaluate the merits of auction-based parking assignment, we assume that drivers have access to a smartphone-based reservation system prior to its mandatory introduction and thus have the opportunity to test and experience its merits voluntarily. We set our experiment as Eclipse SUMO simulations with different rates of participants and non-participants to check how different market penetration levels affect the traffic flow, the performance of the auction-based assignment system, and the financial outcomes. The results show that the auction-based system improves traffic flow with increasing penetration rates, allowing participants to park gradually closer to their preferred parking lots. However, it comes with a price; the system also increases parking expenditures for participants. Interestingly, non-participating drivers will face even higher parking prices. Consequently, they will be motivated to use the new system.

**arXiv ID:** 2601.05429
</details>

<details>
<summary><strong>Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards</strong> - Jiajie Zhang, Xin Lv, Ling Feng, Lei Hou, Juanzi Li - [[pdf]](https://arxiv.org/pdf/2601.06021)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a critical technique for enhancing LLM-based deep search agents. However, existing approaches primarily rely on binary outcome rewards, which fail to capture the comprehensiveness and factuality of agents' reasoning process, and often lead to undesirable behaviors such as shortcut exploitation and hallucinations. To address these limitations, we propose \textbf{Citation-aware Rubric Rewards (CaRR)}, a fine-grained reward framework for deep search agents that emphasizes reasoning comprehensiveness, factual grounding, and evidence connectivity. CaRR decomposes complex questions into verifiable single-hop rubrics and requires agents to satisfy these rubrics by explicitly identifying hidden entities, supporting them with correct citations, and constructing complete evidence chains that link to the predicted answer. We further introduce \textbf{Citation-aware Group Relative Policy Optimization (C-GRPO)}, which combines CaRR and outcome rewards for training robust deep search agents. Experiments show that C-GRPO consistently outperforms standard outcome-based RL baselines across multiple deep search benchmarks. Our analysis also validates that C-GRPO effectively discourages shortcut exploitation, promotes comprehensive, evidence-grounded reasoning, and exhibits strong generalization to open-ended deep research tasks. Our code and data are available at this https URL.

**arXiv ID:** 2601.06021
</details>

<details>
<summary><strong>MaxCode: A Max-Reward Reinforcement Learning Framework for Automated Code Optimization</strong> - Jiefu Ou, Sapana Chaudhary, Kaj Bostrom, Nathaniel Weir, Shuai Zhang, Huzefa Rangwala, George Karypis - [[pdf]](https://arxiv.org/pdf/2601.05475)</summary>

**Abstract:** Large Language Models (LLMs) demonstrate strong capabilities in general coding tasks but encounter two key challenges when optimizing code: (i) the complexity of writing optimized code (such as performant CUDA kernels and competition-level CPU code) requires expertise in systems, algorithms and specific languages and (ii) requires interpretation of performance metrics like timing and device utilization beyond binary correctness. In this work, we explore inference-time search algorithms that guide the LLM to discover better solutions through iterative refinement based on execution feedback. Our approach, called MaxCode unifies existing search methods under a max-reward reinforcement learning framework, making the observation and action-value functions modular for modification. To enhance the observation space, we integrate a natural language critique model that converts raw execution feedback into diagnostic insights about errors and performance bottlenecks, and the best-discounted reward seen so far. Together, these provide richer input to the code proposal function. To improve exploration during search, we train a generative reward-to-go model using action values from rollouts to rerank potential solutions. Testing on the KernelBench (CUDA) and PIE (C++) optimization benchmarks shows that MaxCode improves optimized code performance compared to baselines, achieving 20.3% and 10.1% relative improvements in absolute speedup value and relative speedup ranking, respectively.

**arXiv ID:** 2601.05475
</details>

<details>
<summary><strong>Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation</strong> - Xinping Zhao, Shouzheng Huang, Yan Zhong, Xinshuo Hu, Meishan Zhang, Baotian Hu, Min Zhang - [[pdf]](https://arxiv.org/pdf/2507.15586)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) effectively improves the accuracy of Large Language Models (LLMs). However, retrieval noises significantly undermine the quality of LLMs' generation, necessitating the development of denoising mechanisms. Previous works extract evidence straightforwardly without deep thinking, which may risk filtering out key clues and struggle with generalization. To this end, we propose EviOmni, which learns to extract rational evidence via reasoning first and then extracting. Specifically, EviOmni integrates evidence reasoning and evidence extraction into one unified trajectory, followed by knowledge token masking to avoid information leakage, optimized via on-policy reinforcement learning with verifiable rewards in terms of answer, length, and format. Extensive experiments on five benchmark datasets show the superiority of EviOmni, which provides compact and high-quality evidence, enhances the accuracy of downstream tasks, and supports both traditional and agentic RAG systems.

**arXiv ID:** 2507.15586
</details>

<details>
<summary><strong>Autonomous Discovery of the Ising Model's Critical Parameters with Reinforcement Learning</strong> - Hai Man, Chaobo Wang, Jia-Rui Li, Yuping Tian, Shu-Gang Chen - [[pdf]](https://arxiv.org/pdf/2601.05577)</summary>

**Abstract:** Traditional methods for determining critical parameters are often influenced by human factors. This research introduces a physics-inspired adaptive reinforcement learning framework that enables agents to autonomously interact with physical environments, simultaneously identifying both the critical temperature and various types of critical exponents in the Ising model with precision. Interestingly, our algorithm exhibits search behavior reminiscent of phase transitions, efficiently converging to target parameters regardless of initial conditions. Experimental results demonstrate that this method significantly outperforms traditional approaches, particularly in environments with strong perturbations. This study not only incorporates physical concepts into machine learning to enhance algorithm interpretability but also establishes a new paradigm for scientific exploration, transitioning from manual analysis to autonomous AI discovery.

**arXiv ID:** 2601.05577
</details>

<details>
<summary><strong>Sequential Bayesian Optimal Experimental Design in Infinite Dimensions via Policy Gradient Reinforcement Learning</strong> - Kaichen Shen, Peng Chen - [[pdf]](https://arxiv.org/pdf/2601.05868)</summary>

**Abstract:** Sequential Bayesian optimal experimental design (SBOED) for PDE-governed inverse problems is computationally challenging, especially for infinite-dimensional random field parameters. High-fidelity approaches require repeated forward and adjoint PDE solves inside nested Bayesian inversion and design loops. We formulate SBOED as a finite-horizon Markov decision process and learn an amortized design policy via policy-gradient reinforcement learning (PGRL), enabling online design selection from the experiment history without repeatedly solving an SBOED optimization problem. To make policy training and reward evaluation scalable, we combine dual dimension reduction -- active subspace projection for the parameter and principal component analysis for the state -- with an adjusted derivative-informed latent attention neural operator (LANO) surrogate that predicts both the parameter-to-solution map and its Jacobian. We use a Laplace-based D-optimality reward while noting that, in general, other expected-information-gain utilities such as KL divergence can also be used within the same framework. We further introduce an eigenvalue-based evaluation strategy that uses prior samples as proxies for maximum a posteriori (MAP) points, avoiding repeated MAP solves while retaining accurate information-gain estimates. Numerical experiments on sequential multi-sensor placement for contaminant source tracking demonstrate approximately $100\times$ speedup over high-fidelity finite element methods, improved performance over random sensor placements, and physically interpretable policies that discover an ``upstream'' tracking strategy.

**arXiv ID:** 2601.05868
</details>

<details>
<summary><strong>ACDZero: MCTS Agent for Mastering Automated Cyber Defense</strong> - Yu Li, Sizhe Tang, Rongqian Chen, Fei Xu Yu, Guangyu Jiang, Mahdi Imani, Nathaniel D. Bastian, Tian Lan - [[pdf]](https://arxiv.org/pdf/2601.02196)</summary>

**Abstract:** Automated cyber defense (ACD) seeks to protect computer networks with minimal or no human intervention, reacting to intrusions by taking corrective actions such as isolating hosts, resetting services, deploying decoys, or updating access controls. However, existing approaches for ACD, such as deep reinforcement learning (RL), often face difficult exploration in complex networks with large decision/state spaces and thus require an expensive amount of samples. Inspired by the need to learn sample-efficient defense policies, we frame ACD in CAGE Challenge 4 (CAGE-4 / CC4) as a context-based partially observable Markov decision problem and propose a planning-centric defense policy based on Monte Carlo Tree Search (MCTS). It explicitly models the exploration-exploitation tradeoff in ACD and uses statistical sampling to guide exploration and decision making. We make novel use of graph neural networks (GNNs) to embed observations from the network as attributed graphs, to enable permutation-invariant reasoning over hosts and their relationships. To make our solution practical in complex search spaces, we guide MCTS with learned graph embeddings and priors over graph-edit actions, combining model-free generalization and policy distillation with look-ahead planning. We evaluate the resulting agent on CC4 scenarios involving diverse network structures and adversary behaviors, and show that our search-guided, graph-embedding-based planning improves defense reward and robustness relative to state-of-the-art RL baselines.

**arXiv ID:** 2601.02196
</details>

<details>
<summary><strong>Modular Autonomy with Conversational Interaction: An LLM-driven Framework for Decision Making in Autonomous Driving</strong> - Marvin Seegert, Korbinian Moller, Johannes Betz - [[pdf]](https://arxiv.org/pdf/2601.05806)</summary>

**Abstract:** Recent advancements in Large Language Models (LLMs) offer new opportunities to create natural language interfaces for Autonomous Driving Systems (ADSs), moving beyond rigid inputs. This paper addresses the challenge of mapping the complexity of human language to the structured action space of modular ADS software. We propose a framework that integrates an LLM-based interaction layer with Autoware, a widely used open-source software. This system enables passengers to issue high-level commands, from querying status information to modifying driving behavior. Our methodology is grounded in three key components: a taxonomization of interaction categories, an application-centric Domain Specific Language (DSL) for command translation, and a safety-preserving validation layer. A two-stage LLM architecture ensures high transparency by providing feedback based on the definitive execution status. Evaluation confirms the system's timing efficiency and translation robustness. Simulation successfully validated command execution across all five interaction categories. This work provides a foundation for extensible, DSL-assisted interaction in modular and safety-conscious autonomy stacks.

**arXiv ID:** 2601.05806
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
