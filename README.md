# Agent arXiv Daily

**Last Updated:** 2025-10-29 02:52:42

**Total Papers:** 89

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents</strong> - Zihao Wang, Xujing Li, Yining Ye, Junjie Fang, Haoming Wang, Longxiang Liu, Shihao Liang, Junting Lu, Zhiyong Wu, Jiazhan Feng, Wanjun Zhong, Zili Li, Yu Wang, Yu Miao, Bo Zhou, Yuanfan Li, Hao Wang, Zhongkai Zhao, Faming Wu, Zhengxuan Jiang, Weihao Tan, Heyuan Yao, Shi Yan, Xiangyang Li, Yitao Liang, Yujia Qin, Guang Shi - [[pdf]](https://arxiv.org/pdf/2510.23691)</summary>

**Abstract:** We present Game-TARS, a generalist game agent trained with a unified, scalable action space anchored to human-aligned native keyboard-mouse inputs. Unlike API- or GUI-based approaches, this paradigm enables large-scale continual pre-training across heterogeneous domains, including OS, web, and simulation games. Game-TARS is pre-trained on over 500B tokens with diverse trajectories and multimodal data. Key techniques include a decaying continual loss to reduce causal confusion and an efficient Sparse-Thinking strategy that balances reasoning depth and inference cost. Experiments show that Game-TARS achieves about 2 times the success rate over the previous sota model on open-world Minecraft tasks, is close to the generality of fresh humans in unseen web 3d games, and outperforms GPT-5, Gemini-2.5-Pro, and Claude-4-Sonnet in FPS benchmarks. Scaling results on training-time and test-time confirm that the unified action space sustains improvements when scaled to cross-game and multimodal data. Our results demonstrate that simple, scalable action representations combined with large-scale pre-training provide a promising path toward generalist agents with broad computer-use abilities.

**arXiv ID:** 2510.23691
</details>

<details>
<summary><strong>LLMLogAnalyzer: A Clustering-Based Log Analysis Chatbot using Large Language Models</strong> - Peng Cai, Reza Ryan, Nickson M. Karie - [[pdf]](https://arxiv.org/pdf/2510.24031)</summary>

**Abstract:** System logs are a cornerstone of cybersecurity, supporting proactive breach prevention and post-incident investigations. However, analyzing vast amounts of diverse log data remains significantly challenging, as high costs, lack of in-house expertise, and time constraints make even basic analysis difficult for many organizations. This study introduces LLMLogAnalyzer, a clustering-based log analysis chatbot that leverages Large Language Models (LLMs) and Machine Learning (ML) algorithms to simplify and streamline log analysis processes. This innovative approach addresses key LLM limitations, including context window constraints and poor structured text handling capabilities, enabling more effective summarization, pattern extraction, and anomaly detection tasks. LLMLogAnalyzer is evaluated across four distinct domain logs and various tasks. Results demonstrate significant performance improvements over state-of-the-art LLM-based chatbots, including ChatGPT, ChatPDF, and NotebookLM, with consistent gains ranging from 39% to 68% across different tasks. The system also exhibits strong robustness, achieving a 93% reduction in interquartile range (IQR) when using ROUGE-1 scores, indicating significantly lower result variability. The framework's effectiveness stems from its modular architecture comprising a router, log recognizer, log parser, and search tools. This design enhances LLM capabilities for structured text analysis while improving accuracy and robustness, making it a valuable resource for both cybersecurity experts and non-technical users.

**arXiv ID:** 2510.24031
</details>

<details>
<summary><strong>Affordance Representation and Recognition for Autonomous Agents</strong> - Habtom Kahsay Gidey, Niklas Huber, Alexander Lenz, Alois Knoll - [[pdf]](https://arxiv.org/pdf/2510.24459)</summary>

**Abstract:** The autonomy of software agents is fundamentally dependent on their ability to construct an actionable internal world model from the structured data that defines their digital environment, such as the Document Object Model (DOM) of web pages and the semantic descriptions of web services. However, constructing this world model from raw structured data presents two critical challenges: the verbosity of raw HTML makes it computationally intractable for direct use by foundation models, while the static nature of hardcoded API integrations prevents agents from adapting to evolving services.
This paper introduces a pattern language for world modeling from structured data, presenting two complementary architectural patterns. The DOM Transduction Pattern addresses the challenge of web page complexity by distilling} a verbose, raw DOM into a compact, task-relevant representation or world model optimized for an agent's reasoning core. Concurrently, the Hypermedia Affordances Recognition Pattern enables the agent to dynamically enrich its world model by parsing standardized semantic descriptions to discover and integrate the capabilities of unknown web services at runtime. Together, these patterns provide a robust framework for engineering agents that can efficiently construct and maintain an accurate world model, enabling scalable, adaptive, and interoperable automation across the web and its extended resources.

**arXiv ID:** 2510.24459
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>UniPlanner: A Unified Motion Planning Framework for Autonomous Vehicle Decision-Making Systems via Multi-Dataset Integration</strong> - Xin Yang, Yuhang Zhang, Wei Li, Xin Lin, Wenbin Zou, Chen Xu - [[pdf]](https://arxiv.org/pdf/2510.24166)</summary>

**Abstract:** Motion planning is a critical component of autonomous vehicle decision-making systems, directly determining trajectory safety and driving efficiency. While deep learning approaches have advanced planning capabilities, existing methods remain confined to single-dataset training, limiting their robustness in planning.
Through systematic analysis, we discover that vehicular trajectory distributions and history-future correlations demonstrate remarkable consistency across different datasets. Based on these findings, we propose UniPlanner, the first planning framework designed for multi-dataset integration in autonomous vehicle decision-making. UniPlanner achieves unified cross-dataset learning through three synergistic innovations.
First, the History-Future Trajectory Dictionary Network (HFTDN) aggregates history-future trajectory pairs from multiple datasets, using historical trajectory similarity to retrieve relevant futures and generate cross-dataset planning guidance.
Second, the Gradient-Free Trajectory Mapper (GFTM) learns robust history-future correlations from multiple datasets, transforming historical trajectories into universal planning priors. Its gradient-free design ensures the introduction of valuable priors while preventing shortcut learning, making the planning knowledge safely transferable. Third, the Sparse-to-Dense (S2D) paradigm implements adaptive dropout to selectively suppress planning priors during training for robust learning, while enabling full prior utilization during inference to maximize planning performance.

**arXiv ID:** 2510.24166
</details>

<details>
<summary><strong>MGA: Memory-Driven GUI Agent for Observation-Centric Interaction</strong> - Weihua Cheng, Ersheng Ni, Wenlong Wang, Yifei Sun, Junming Liu, Wangyu Shen, Yirong Chen, Botian Shi, Ding Wang - [[pdf]](https://arxiv.org/pdf/2510.24168)</summary>

**Abstract:** The rapid progress of Large Language Models (LLMs) and their multimodal extensions (MLLMs) has enabled agentic systems capable of perceiving and acting across diverse environments. A challenging yet impactful frontier is the development of GUI agents, which must navigate complex desktop and web interfaces while maintaining robustness and generalization. Existing paradigms typically model tasks as long-chain executions, concatenating historical trajectories into the context. While approaches such as Mirage and GTA1 refine planning or introduce multi-branch action selection, they remain constrained by two persistent issues: Dependence on historical trajectories, which amplifies error propagation. And Local exploration bias, where "decision-first, observation-later" mechanisms overlook critical interface cues. We introduce the Memory-Driven GUI Agent (MGA), which reframes GUI interaction around the principle of observe first, then decide. MGA models each step as an independent, context-rich environment state represented by a triad: current screenshot, task-agnostic spatial information, and a dynamically updated structured memory. Experiments on OSworld benchmarks, real desktop applications (Chrome, VSCode, VLC), and cross-task transfer demonstrate that MGA achieves substantial gains in robustness, generalization, and efficiency compared to state-of-the-art baselines. The code is publicly available at: {this https URL}.

**arXiv ID:** 2510.24168
</details>

<details>
<summary><strong>OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows</strong> - Qiushi Sun, Mukai Li, Zhoumianze Liu, Zhihui Xie, Fangzhi Xu, Zhangyue Yin, Kanzhi Cheng, Zehao Li, Zichen Ding, Qi Liu, Zhiyong Wu, Zhuosheng Zhang, Ben Kao, Lingpeng Kong - [[pdf]](https://arxiv.org/pdf/2510.24411)</summary>

**Abstract:** Computer-using agents powered by Vision-Language Models (VLMs) have demonstrated human-like capabilities in operating digital environments like mobile platforms. While these agents hold great promise for advancing digital automation, their potential for unsafe operations, such as system compromise and privacy leakage, is raising significant concerns. Detecting these safety concerns across the vast and complex operational space of mobile environments presents a formidable challenge that remains critically underexplored. To establish a foundation for mobile agent safety research, we introduce MobileRisk-Live, a dynamic sandbox environment accompanied by a safety detection benchmark comprising realistic trajectories with fine-grained annotations. Built upon this, we propose OS-Sentinel, a novel hybrid safety detection framework that synergistically combines a Formal Verifier for detecting explicit system-level violations with a VLM-based Contextual Judge for assessing contextual risks and agent actions. Experiments show that OS-Sentinel achieves 10%-30% improvements over existing approaches across multiple metrics. Further analysis provides critical insights that foster the development of safer and more reliable autonomous mobile agents.

**arXiv ID:** 2510.24411
</details>

<details>
<summary><strong>VisCoder2: Building Multi-Language Visualization Coding Agents</strong> - Yuansheng Ni, Songcheng Cai, Xiangchao Chen, Jiarong Liang, Zhiheng Lyu, Jiaqi Deng, Kai Zou, Ping Nie, Fei Yuan, Xiang Yue, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2510.23642)</summary>

**Abstract:** Large language models (LLMs) have recently enabled coding agents capable of generating, executing, and revising visualization code. However, existing models often fail in practical workflows due to limited language coverage, unreliable execution, and lack of iterative correction mechanisms. Progress has been constrained by narrow datasets and benchmarks that emphasize single-round generation and single-language tasks. To address these challenges, we introduce three complementary resources for advancing visualization coding agents. VisCode-Multi-679K is a large-scale, supervised dataset containing 679K validated and executable visualization samples with multi-turn correction dialogues across 12 programming languages. VisPlotBench is a benchmark for systematic evaluation, featuring executable tasks, rendered outputs, and protocols for both initial generation and multi-round self-debug. Finally, we present VisCoder2, a family of multi-language visualization models trained on VisCode-Multi-679K. Experiments show that VisCoder2 significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4.1, with further gains from iterative self-debug, reaching 82.4% overall execution pass rate at the 32B scale, particularly in symbolic or compiler-dependent languages.

**arXiv ID:** 2510.23642
</details>

<details>
<summary><strong>Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content</strong> - Abdullah Mushtaq, Rafay Naeem, Ezieddin Elmahjub, Ibrahim Ghaznavi, Shawqi Al-Maliki, Mohamed Abdallah, Ala Al-Fuqaha, Junaid Qadir - [[pdf]](https://arxiv.org/pdf/2510.24438)</summary>

**Abstract:** Large language models are increasingly used for Islamic guidance, but risk misquoting texts, misapplying jurisprudence, or producing culturally inconsistent responses. We pilot an evaluation of GPT-4o, Ansari AI, and Fanar on prompts from authentic Islamic blogs. Our dual-agent framework uses a quantitative agent for citation verification and six-dimensional scoring (e.g., Structure, Islamic Consistency, Citations) and a qualitative agent for five-dimensional side-by-side comparison (e.g., Tone, Depth, Originality). GPT-4o scored highest in Islamic Accuracy (3.93) and Citation (3.38), Ansari AI followed (3.68, 3.32), and Fanar lagged (2.76, 1.82). Despite relatively strong performance, models still fall short in reliably producing accurate Islamic content and citations -- a paramount requirement in faith-sensitive writing. GPT-4o had the highest mean quantitative score (3.90/5), while Ansari AI led qualitative pairwise wins (116/200). Fanar, though trailing, introduces innovations for Islamic and Arabic contexts. This study underscores the need for community-driven benchmarks centering Muslim perspectives, offering an early step toward more reliable AI in Islamic knowledge and other high-stakes domains such as medicine, law, and journalism.

**arXiv ID:** 2510.24438
</details>

<details>
<summary><strong>InteractComp: Evaluating Search Agents With Ambiguous Queries</strong> - Mingyi Deng, Lijun Huang, Yani Fan, Jiayi Zhang, Fashen Ren, Jinyi Bai, Fuzhen Yang, Dayi Miao, Zhaoyang Yu, Yifan Wu, Yanfei Zhang, Fengwei Teng, Yingjia Wan, Song Hu, Yude Li, Xin Jin, Conghao Hu, Haoyu Li, Qirui Fu, Tai Zhong, Xinyu Wang, Xiangru Tang, Nan Tang, Chenglin Wu, Yuyu Luo - [[pdf]](https://arxiv.org/pdf/2510.24668)</summary>

**Abstract:** Language agents have demonstrated remarkable potential in web search and information retrieval. However, these search agents assume user queries are complete and unambiguous, an assumption that diverges from reality where users begin with incomplete queries requiring clarification through interaction. Yet most agents lack interactive mechanisms during the search process, and existing benchmarks cannot assess this capability. To address this gap, we introduce InteractComp, a benchmark designed to evaluate whether search agents can recognize query ambiguity and actively interact to resolve it during search. Following the principle of easy to verify, interact to disambiguate, we construct 210 expert-curated questions across 9 domains through a target-distractor methodology that creates genuine ambiguity resolvable only through interaction. Evaluation of 17 models reveals striking failure: the best model achieves only 13.73% accuracy despite 71.50% with complete context, exposing systematic overconfidence rather than reasoning deficits. Forced interaction produces dramatic gains, demonstrating latent capability current strategies fail to engage. Longitudinal analysis shows interaction capabilities stagnated over 15 months while search performance improved seven-fold, revealing a critical blind spot. This stagnation, coupled with the immediate feedback inherent to search tasks, makes InteractComp a valuable resource for both evaluating and training interaction capabilities in search agents. The code is available at this https URL.

**arXiv ID:** 2510.24668
</details>

<details>
<summary><strong>ParallelMuse: Agentic Parallel Thinking for Deep Information Seeking</strong> - Baixuan Li, Dingchu Zhang, Jialong Wu, Wenbiao Yin, Zhengwei Tao, Yida Zhao, Liwen Zhang, Haiyang Shen, Runnan Fang, Pengjun Xie, Jingren Zhou, Yong Jiang - [[pdf]](https://arxiv.org/pdf/2510.24698)</summary>

**Abstract:** Parallel thinking expands exploration breadth, complementing the deep exploration of information-seeking (IS) agents to further enhance problem-solving capability. However, conventional parallel thinking faces two key challenges in this setting: inefficiency from repeatedly rolling out from scratch, and difficulty in integrating long-horizon reasoning trajectories during answer generation, as limited context capacity prevents full consideration of the reasoning process. To address these issues, we propose ParallelMuse, a two-stage paradigm designed for deep IS agents. The first stage, Functionality-Specified Partial Rollout, partitions generated sequences into functional regions and performs uncertainty-guided path reuse and branching to enhance exploration efficiency. The second stage, Compressed Reasoning Aggregation, exploits reasoning redundancy to losslessly compress information relevant to answer derivation and synthesize a coherent final answer. Experiments across multiple open-source agents and benchmarks demonstrate up to 62% performance improvement with a 10--30% reduction in exploratory token consumption.

**arXiv ID:** 2510.24698
</details>

<details>
<summary><strong>AgentFold: Long-Horizon Web Agents with Proactive Context Management</strong> - Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, Pengjun Xie, Fei Huang, Siheng Chen, Jingren Zhou, Yong Jiang - [[pdf]](https://arxiv.org/pdf/2510.24699)</summary>

**Abstract:** LLM-based web agents show immense promise for information seeking, yet their effectiveness on long-horizon tasks is hindered by a fundamental trade-off in context management. Prevailing ReAct-based agents suffer from context saturation as they accumulate noisy, raw histories, while methods that fixedly summarize the full history at each step risk the irreversible loss of critical details. Addressing these, we introduce AgentFold, a novel agent paradigm centered on proactive context management, inspired by the human cognitive process of retrospective consolidation. AgentFold treats its context as a dynamic cognitive workspace to be actively sculpted, rather than a passive log to be filled. At each step, it learns to execute a `folding' operation, which manages its historical trajectory at multiple scales: it can perform granular condensations to preserve vital, fine-grained details, or deep consolidations to abstract away entire multi-step sub-tasks. The results on prominent benchmarks are striking: with simple supervised fine-tuning (without continual pre-training or RL), our AgentFold-30B-A3B agent achieves 36.2% on BrowseComp and 47.3% on BrowseComp-ZH. Notably, this performance not only surpasses or matches open-source models of a dramatically larger scale, such as the DeepSeek-V3.1-671B-A37B, but also surpasses leading proprietary agents like OpenAI's o4-mini.

**arXiv ID:** 2510.24699
</details>

<details>
<summary><strong>Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine</strong> - Wenyi Wang, Piotr Piękos, Li Nanbo, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, Jürgen Schmidhuber - [[pdf]](https://arxiv.org/pdf/2510.21614)</summary>

**Abstract:** Recent studies operationalize self-improvement through coding agents that edit their own codebases. They grow a tree of self-modifications through expansion strategies that favor higher software engineering benchmark performance, assuming that this implies more promising subsequent self-modifications. However, we identify a mismatch between the agent's self-improvement potential (metaproductivity) and its coding benchmark performance, namely the Metaproductivity-Performance Mismatch. Inspired by Huxley's concept of clade, we propose a metric ($\mathrm{CMP}$) that aggregates the benchmark performances of the descendants of an agent as an indicator of its potential for self-improvement. We show that, in our self-improving coding agent development setting, access to the true $\mathrm{CMP}$ is sufficient to simulate how the Gödel Machine would behave under certain assumptions. We introduce the Huxley-Gödel Machine (HGM), which, by estimating $\mathrm{CMP}$ and using it as guidance, searches the tree of self-modifications. On SWE-bench Verified and Polyglot, HGM outperforms prior self-improving coding agent development methods while using less wall-clock time. Last but not least, HGM demonstrates strong transfer to other coding datasets and large language models. The agent optimized by HGM on SWE-bench Verified with GPT-5-mini and evaluated on SWE-bench Lite with GPT-5 achieves human-level performance, matching the best officially checked results of human-engineered coding agents. Our code is available at this https URL.

**arXiv ID:** 2510.21614
</details>

<details>
<summary><strong>ReplicationBench: Can AI Agents Replicate Astrophysics Research Papers?</strong> - Christine Ye, Sihan Yuan, Suchetha Cooray, Steven Dillmann, Ian L. V. Roque, Dalya Baron, Philipp Frank, Sergio Martin-Alvarez, Nolan Koblischke, Frank J Qu, Diyi Yang, Risa Wechsler, Ioana Ciuca - [[pdf]](https://arxiv.org/pdf/2510.24591)</summary>

**Abstract:** Frontier AI agents show increasing promise as scientific research assistants, and may eventually be useful for extended, open-ended research workflows. However, in order to use agents for novel research, we must first assess the underlying faithfulness and correctness of their work. To evaluate agents as research assistants, we introduce ReplicationBench, an evaluation framework that tests whether agents can replicate entire research papers drawn from the astrophysics literature. Astrophysics, where research relies heavily on archival data and computational study while requiring little real-world experimentation, is a particularly useful testbed for AI agents in scientific research. We split each paper into tasks which require agents to replicate the paper's core contributions, including the experimental setup, derivations, data analysis, and codebase. Each task is co-developed with the original paper authors and targets a key scientific result, enabling objective evaluation of both faithfulness (adherence to original methods) and correctness (technical accuracy of results). ReplicationBench is extremely challenging for current frontier language models: even the best-performing language models score under 20%. We analyze ReplicationBench trajectories in collaboration with domain experts and find a rich, diverse set of failure modes for agents in scientific research. ReplicationBench establishes the first benchmark of paper-scale, expert-validated astrophysics research tasks, reveals insights about agent performance generalizable to other domains of data-driven science, and provides a scalable framework for measuring AI agents' reliability in scientific research.

**arXiv ID:** 2510.24591
</details>

<details>
<summary><strong>The Dialogue That Heals: A Comprehensive Evaluation of Doctor Agents' Inquiry Capability</strong> - Linlu Gong, Ante Wang, Yunghwei Lai, Weizhi Ma, Yang Liu - [[pdf]](https://arxiv.org/pdf/2509.24958)</summary>

**Abstract:** An effective physician should possess a combination of empathy, expertise, patience, and clear communication when treating a patient. Recent advances have successfully endowed AI doctors with expert diagnostic skills, particularly the ability to actively seek information through inquiry. However, other essential qualities of a good doctor remain overlooked. To bridge this gap, we present MAQuE(Medical Agent Questioning Evaluation), the largest-ever benchmark for the automatic and comprehensive evaluation of medical multi-turn questioning. It features 3,000 realistically simulated patient agents that exhibit diverse linguistic patterns, cognitive limitations, emotional responses, and tendencies for passive disclosure. We also introduce a multi-faceted evaluation framework, covering task success, inquiry proficiency, dialogue competence, inquiry efficiency, and patient experience. Experiments on different LLMs reveal substantial challenges across the evaluation aspects. Even state-of-the-art models show significant room for improvement in their inquiry capabilities. These models are highly sensitive to variations in realistic patient behavior, which considerably impacts diagnostic accuracy. Furthermore, our fine-grained metrics expose trade-offs between different evaluation perspectives, highlighting the challenge of balancing performance and practicality in real-world clinical settings.

**arXiv ID:** 2509.24958
</details>

<details>
<summary><strong>Modeling and Scheduling of Fusion Patterns in Autonomous Driving Systems (Extended Version)</strong> - Hoora Sobhani, Hyoseung Kim - [[pdf]](https://arxiv.org/pdf/2510.23895)</summary>

**Abstract:** In Autonomous Driving Systems (ADS), Directed Acyclic Graphs (DAGs) are widely used to model complex data dependencies and inter-task communication. However, existing DAG scheduling approaches oversimplify data fusion tasks by assuming fixed triggering mechanisms, failing to capture the diverse fusion patterns found in real-world ADS software stacks. In this paper, we propose a systematic framework for analyzing various fusion patterns and their performance implications in ADS. Our framework models three distinct fusion task types: timer-triggered, wait-for-all, and immediate fusion, which comprehensively represent real-world fusion behaviors. Our Integer Linear Programming (ILP)-based approach enables an optimization of multiple real-time performance metrics, including reaction time, time disparity, age of information, and response time, while generating deterministic offline schedules directly applicable to real platforms. Evaluation using real-world ADS case studies, Raspberry Pi implementation, and randomly generated DAGs demonstrates that our framework handles diverse fusion patterns beyond the scope of existing work, and achieves substantial performance improvements in comparable scenarios.

**arXiv ID:** 2510.23895
</details>

<details>
<summary><strong>GaussianFusion: Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving</strong> - Shuai Liu, Quanmin Liang, Zefeng Li, Boyang Li, Kai Huang - [[pdf]](https://arxiv.org/pdf/2506.00034)</summary>

**Abstract:** Multi-sensor fusion is crucial for improving the performance and robustness of end-to-end autonomous driving systems. Existing methods predominantly adopt either attention-based flatten fusion or bird's eye view fusion through geometric transformations. However, these approaches often suffer from limited interpretability or dense computational overhead. In this paper, we introduce GaussianFusion, a Gaussian-based multi-sensor fusion framework for end-to-end autonomous driving. Our method employs intuitive and compact Gaussian representations as intermediate carriers to aggregate information from diverse sensors. Specifically, we initialize a set of 2D Gaussians uniformly across the driving scene, where each Gaussian is parameterized by physical attributes and equipped with explicit and implicit features. These Gaussians are progressively refined by integrating multi-modal features. The explicit features capture rich semantic and spatial information about the traffic scene, while the implicit features provide complementary cues beneficial for trajectory planning. To fully exploit rich spatial and semantic information in Gaussians, we design a cascade planning head that iteratively refines trajectory predictions through interactions with Gaussians. Extensive experiments on the NAVSIM and Bench2Drive benchmarks demonstrate the effectiveness and robustness of the proposed GaussianFusion framework. The source code will be released at this https URL.

**arXiv ID:** 2506.00034
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>ReCAP: Recursive Context-Aware Reasoning and Planning for Large Language Model Agents</strong> - Zhenyu Zhang, Tianyi Chen, Weiran Xu, Alex Pentland, Jiaxin Pei - [[pdf]](https://arxiv.org/pdf/2510.23822)</summary>

**Abstract:** Long-horizon tasks requiring multi-step reasoning and dynamic re-planning remain challenging for large language models (LLMs). Sequential prompting methods are prone to context drift, loss of goal information, and recurrent failure cycles, while hierarchical prompting methods often weaken cross-level continuity or incur substantial runtime overhead. We introduce ReCAP (Recursive Context-Aware Reasoning and Planning), a hierarchical framework with shared context for reasoning and planning in LLMs. ReCAP combines three key mechanisms: (i) plan-ahead decomposition, in which the model generates a full subtask list, executes the first item, and refines the remainder; (ii) structured re-injection of parent plans, maintaining consistent multi-level context during recursive return; and (iii) memory-efficient execution, bounding the active prompt so costs scale linearly with task depth. Together these mechanisms align high-level goals with low-level actions, reduce redundant prompting, and preserve coherent context updates across recursion. Experiments demonstrate that ReCAP substantially improves subgoal alignment and success rates on various long-horizon reasoning benchmarks, achieving a 32% gain on synchronous Robotouille and a 29% improvement on asynchronous Robotouille under the strict pass@1 protocol.

**arXiv ID:** 2510.23822
</details>

<details>
<summary><strong>MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools</strong> - Wenhao Wang, Peizhi Niu, Zhao Xu, Zhaoyu Chen, Jian Du, Yaxin Du, Xianghe Pang, Keduan Huang, Yanfeng Wang, Qiang Yan, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2510.24284)</summary>

**Abstract:** Large Language Models (LLMs) increasingly rely on external tools to perform complex, realistic tasks, yet their ability to utilize the rapidly expanding Model Contextual Protocol (MCP) ecosystem remains limited. Existing MCP research covers few servers, depends on costly manual curation, and lacks training support, hindering progress toward real-world deployment. To overcome these limitations, we introduce MCP-Flow, an automated web-agent-driven pipeline for large-scale server discovery, data synthesis, and model training. MCP-Flow collects and filters data from 1166 servers and 11536 tools, producing 68733 high-quality instruction-function call pairs and 6439 trajectories, far exceeding prior work in scale and diversity. Extensive experiments demonstrate MCP-Flow's effectiveness in driving superior MCP tool selection, function-call generation, and enhanced agentic task performance. MCP-Flow thus provides a scalable foundation for advancing LLM agents' proficiency in real-world MCP environments. MCP-Flow is publicly available at \href{this https URL}{this https URL}.

**arXiv ID:** 2510.24284
</details>

<details>
<summary><strong>Beyond Prompt Engineering: Neuro-Symbolic-Causal Architecture for Robust Multi-Objective AI Agents</strong> - Gokturk Aytug Akarlar - [[pdf]](https://arxiv.org/pdf/2510.23682)</summary>

**Abstract:** Large language models show promise as autonomous decision-making agents, yet their deployment in high-stakes domains remains fraught with risk. Without architectural safeguards, LLM agents exhibit catastrophic brittleness: identical capabilities produce wildly different outcomes depending solely on prompt framing. We present Chimera, a neuro-symbolic-causal architecture that integrates three complementary components - an LLM strategist, a formally verified symbolic constraint engine, and a causal inference module for counterfactual reasoning. We benchmark Chimera against baseline architectures (LLM-only, LLM with symbolic constraints) across 52-week simulations in a realistic e-commerce environment featuring price elasticity, trust dynamics, and seasonal demand. Under organizational biases toward either volume or margin optimization, LLM-only agents fail catastrophically (total loss of \$99K in volume scenarios) or destroy brand trust (-48.6% in margin scenarios). Adding symbolic constraints prevents disasters but achieves only 43-87% of Chimera's profit. Chimera consistently delivers the highest returns (\$1.52M and \$1.96M respectively, some cases +\$2.2M) while improving brand trust (+1.8% and +10.8%, some cases +20.86%), demonstrating prompt-agnostic robustness. Our TLA+ formal verification proves zero constraint violations across all scenarios. These results establish that architectural design not prompt engineering determines the reliability of autonomous agents in production environments. We provide open-source implementations and interactive demonstrations for reproducibility.

**arXiv ID:** 2510.23682
</details>

<details>
<summary><strong>Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents</strong> - Yueqi Song, Ketan Ramaneti, Zaid Sheikh, Ziru Chen, Boyu Gou, Tianbao Xie, Yiheng Xu, Danyang Zhang, Apurva Gandhi, Fan Yang, Joseph Liu, Tianyue Ou, Zhihao Yuan, Frank Xu, Shuyan Zhou, Xingyao Wang, Xiang Yue, Tao Yu, Huan Sun, Yu Su, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2510.24702)</summary>

**Abstract:** Public research results on large-scale supervised finetuning of AI agents remain relatively rare, since the collection of agent training data presents unique challenges. In this work, we argue that the bottleneck is not a lack of underlying data sources, but that a large variety of data is fragmented across heterogeneous formats, tools, and interfaces. To this end, we introduce the agent data protocol (ADP), a light-weight representation language that serves as an "interlingua" between agent datasets in diverse formats and unified agent training pipelines downstream. The design of ADP is expressive enough to capture a large variety of tasks, including API/tool use, browsing, coding, software engineering, and general agentic workflows, while remaining simple to parse and train on without engineering at a per-dataset level. In experiments, we unified a broad collection of 13 existing agent training datasets into ADP format, and converted the standardized ADP data into training-ready formats for multiple agent frameworks. We performed SFT on these data, and demonstrated an average performance gain of ~20% over corresponding base models, and delivers state-of-the-art or near-SOTA performance on standard coding, browsing, tool use, and research benchmarks, without domain-specific tuning. All code and data are released publicly, in the hope that ADP could help lower the barrier to standardized, scalable, and reproducible agent training.

**arXiv ID:** 2510.24702
</details>

<details>
<summary><strong>Group-in-Group Policy Optimization for LLM Agent Training</strong> - Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An - [[pdf]](https://arxiv.org/pdf/2505.10978)</summary>

**Abstract:** Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to multi-turn LLM agent training remains limited. Unlike static tasks, agent-environment interactions unfold over many steps and often yield sparse or delayed rewards, making credit assignment across individual steps significantly more challenging. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories. Actions stemming from the same state are grouped together, enabling micro relative advantage estimation. This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts. We evaluate GiGPO on challenging agent benchmarks, including ALFWorld and WebShop, as well as tool-integrated reasoning on search-augmented QA tasks, using Qwen2.5-1.5B/3B/7B-Instruct. Crucially, GiGPO delivers fine-grained per-step credit signals, achieves performance gains of > 12% on ALFWorld and > 9% on WebShop over GRPO, and obtains superior performance on QA tasks (42.1% on 3B and 47.2% on 7B): all while maintaining the same GPU memory overhead, identical LLM rollout, and incurring little to no additional time cost.

**arXiv ID:** 2505.10978
</details>

<details>
<summary><strong>Temporal Blindness in Multi-Turn LLM Agents: Misaligned Tool Use vs. Human Time Perception</strong> - Yize Cheng, Arshia Soltani Moakhar, Chenrui Fan, Kazem Faghih, Parsa Hosseini, Wenxiao Wang, Soheil Feizi - [[pdf]](https://arxiv.org/pdf/2510.23853)</summary>

**Abstract:** Large language model agents are increasingly used in multi-turn conversational settings to interact with and execute tasks in dynamic environments. However, a key limitation is their temporal blindness: they, by default, operate with a stationary context, failing to account for the real-world time elapsed between messages. This becomes a critical liability when an agent must decide whether to invoke a tool based on how much time has passed since the last observation. Without temporal awareness, agents often either over-rely on previous context (skipping necessary tool calls), or under-rely on it (unnecessarily repeating tool calls). To study this challenge, we introduce TicToc-v1, a test set of multi-turn user-agent trajectories across 34 scenarios with varying time sensitivity. Each trajectory ends with a user question, where the need for a tool call depends on the amount of time elapsed since the last message. To give LLMs temporal context, we augment dialogue messages with explicit timestamps, bridging the gap between static dialogue and evolving environments. We then collected human preferences for these samples, creating two subsets: one where humans preferred relying on the previous observation (prefer-noTool), and another where they preferred a new tool call (prefer-Tool). We evaluated how well LLM tool-calling decisions align with human preferences under varying time intervals on TicToc-v1. Our analysis show that without time information, most models perform only slightly better than random, with the top alignment rate being just over 60%. While adding timestamps leads to a slight improvement, particularly for larger models, the improvement is modest, peaking at around 65%. We also show that naive, prompt-based alignment have limited effectiveness. Our findings highlight the need for specific post-training alignment to align multi-turn LLM tool use with human temporal perception.

**arXiv ID:** 2510.23853
</details>

<details>
<summary><strong>TEXT2DB: Integration-Aware Information Extraction with Large Language Model Agents</strong> - Yizhu Jiao, Sha Li, Sizhe Zhou, Heng Ji, Jiawei Han - [[pdf]](https://arxiv.org/pdf/2510.24014)</summary>

**Abstract:** The task of information extraction (IE) is to extract structured knowledge from text. However, it is often not straightforward to utilize IE output due to the mismatch between the IE ontology and the downstream application needs. We propose a new formulation of IE TEXT2DB that emphasizes the integration of IE output and the target database (or knowledge base). Given a user instruction, a document set, and a database, our task requires the model to update the database with values from the document set to satisfy the user instruction. This task requires understanding user instructions for what to extract and adapting to the given DB/KB schema for how to extract on the fly. To evaluate this new task, we introduce a new benchmark featuring common demands such as data infilling, row population, and column addition. In addition, we propose an LLM agent framework OPAL (Observe-PlanAnalyze LLM) which includes an Observer component that interacts with the database, the Planner component that generates a code-based plan with calls to IE models, and the Analyzer component that provides feedback regarding code quality before execution. Experiments show that OPAL can successfully adapt to diverse database schemas by generating different code plans and calling the required IE models. We also highlight difficult cases such as dealing with large databases with complex dependencies and extraction hallucination, which we believe deserve further investigation. Source code: this https URL

**arXiv ID:** 2510.24014
</details>

<details>
<summary><strong>AgentFrontier: Expanding the Capability Frontier of LLM Agents with ZPD-Guided Data Synthesis</strong> - Xuanzhong Chen, Zile Qiao, Guoxin Chen, Liangcai Su, Zhen Zhang, Xinyu Wang, Pengjun Xie, Fei Huang, Jingren Zhou, Yong Jiang - [[pdf]](https://arxiv.org/pdf/2510.24695)</summary>

**Abstract:** Training large language model agents on tasks at the frontier of their capabilities is key to unlocking advanced reasoning. We introduce a data synthesis approach inspired by the educational theory of the Zone of Proximal Development (ZPD), which defines this frontier as tasks an LLM cannot solve alone but can master with guidance. To operationalize this, we present the AgentFrontier Engine, an automated pipeline that synthesizes high-quality, multidisciplinary data situated precisely within the LLM's ZPD. This engine supports both continued pre-training with knowledge-intensive data and targeted post-training on complex reasoning tasks. From the same framework, we derive the ZPD Exam, a dynamic and automated benchmark designed to evaluate agent capabilities on these frontier tasks. We train AgentFrontier-30B-A3B model on our synthesized data, which achieves state-of-the-art results on demanding benchmarks like Humanity's Last Exam, even surpassing some leading proprietary agents. Our work demonstrates that a ZPD-guided approach to data synthesis offers a scalable and effective path toward building more capable LLM agents.

**arXiv ID:** 2510.24695
</details>

<details>
<summary><strong>Large Language Model Agent Personality and Response Appropriateness: Evaluation by Human Linguistic Experts, LLM-as-Judge, and Natural Language Processing Model</strong> - Eswari Jayakumar, Niladri Sekhar Dash, Debasmita Mukherjee - [[pdf]](https://arxiv.org/pdf/2510.23875)</summary>

**Abstract:** While Large Language Model (LLM)-based agents can be used to create highly engaging interactive applications through prompting personality traits and contextual data, effectively assessing their personalities has proven challenging. This novel interdisciplinary approach addresses this gap by combining agent development and linguistic analysis to assess the prompted personality of LLM-based agents in a poetry explanation task. We developed a novel, flexible question bank, informed by linguistic assessment criteria and human cognitive learning levels, offering a more comprehensive evaluation than current methods. By evaluating agent responses with natural language processing models, other LLMs, and human experts, our findings illustrate the limitations of purely deep learning solutions and emphasize the critical role of interdisciplinary design in agent development.

**arXiv ID:** 2510.23875
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>Decentralized Multi-Agent Goal Assignment for Path Planning using Large Language Models</strong> - Murad Ismayilov, Edwin Meriaux, Shuo Wen, Gregory Dudek - [[pdf]](https://arxiv.org/pdf/2510.23824)</summary>

**Abstract:** Coordinating multiple autonomous agents in shared environments under decentralized conditions is a long-standing challenge in robotics and artificial intelligence. This work addresses the problem of decentralized goal assignment for multi-agent path planning, where agents independently generate ranked preferences over goals based on structured representations of the environment, including grid visualizations and scenario data. After this reasoning phase, agents exchange their goal rankings, and assignments are determined by a fixed, deterministic conflict-resolution rule (e.g., agent index ordering), without negotiation or iterative coordination. We systematically compare greedy heuristics, optimal assignment, and large language model (LLM)-based agents in fully observable grid-world settings. Our results show that LLM-based agents, when provided with well-designed prompts and relevant quantitative information, can achieve near-optimal makespans and consistently outperform traditional heuristics. These findings underscore the potential of language models for decentralized goal assignment in multi-agent path planning and highlight the importance of information structure in such systems.

**arXiv ID:** 2510.23824
</details>

<details>
<summary><strong>From Observability Data to Diagnosis: An Evolving Multi-agent System for Incident Management in Cloud Systems</strong> - Yu Luo, Jiamin Jiang, Jingfei Feng, Lei Tao, Qingliang Zhang, Xidao Wen, Yongqian Sun, Shenglin Zhang, Jielong Huang, Nan Qi, Dan Pei - [[pdf]](https://arxiv.org/pdf/2510.24145)</summary>

**Abstract:** Incident management (IM) is central to the reliability of large-scale cloud systems. Yet manual IM, where on-call engineers examine metrics, logs, and traces is labor-intensive and error-prone in the face of massive and heterogeneous observability data. Existing automated IM approaches often struggle to generalize across systems, provide limited interpretability, and incur high deployment costs, which hinders adoption in practice. In this paper, we present OpsAgent, a lightweight, self-evolving multi-agent system for IM that employs a training-free data processor to convert heterogeneous observability data into structured textual descriptions, along with a multi-agent collaboration framework that makes diagnostic inference transparent and auditable. To support continual capability growth, OpsAgent also introduces a dual self-evolution mechanism that integrates internal model updates with external experience accumulation, thereby closing the deployment loop. Comprehensive experiments on the OPENRCA benchmark demonstrate state-of-the-art performance and show that OpsAgent is generalizable, interpretable, cost-efficient, and self-evolving, making it a practically deployable and sustainable solution for long-term operation in real-world cloud systems.

**arXiv ID:** 2510.24145
</details>

<details>
<summary><strong>Retrieval and Argumentation Enhanced Multi-Agent LLMs for Judgmental Forecasting</strong> - Deniz Gorur, Antoni Rago, Francesca Toni - [[pdf]](https://arxiv.org/pdf/2510.24303)</summary>

**Abstract:** Judgmental forecasting is the task of making predictions about future events based on human judgment. This task can be seen as a form of claim verification, where the claim corresponds to a future event and the task is to assess the plausibility of that event. In this paper, we propose a novel multi-agent framework for claim verification, whereby different agents may disagree on claim veracity and bring specific evidence for and against the claims, represented as quantitative bipolar argumentation frameworks (QBAFs). We then instantiate the framework for supporting claim verification, with a variety of agents realised with Large Language Models (LLMs): (1) ArgLLM agents, an existing approach for claim verification that generates and evaluates QBAFs; (2) RbAM agents, whereby LLM-empowered Relation-based Argument Mining (RbAM) from external sources is used to generate QBAFs; (3) RAG-ArgLLM agents, extending ArgLLM agents with a form of Retrieval-Augmented Generation (RAG) of arguments from external sources. Finally, we conduct experiments with two standard judgmental forecasting datasets, with instances of our framework with two or three agents, empowered by six different base LLMs. We observe that combining evidence from agents can improve forecasting accuracy, especially in the case of three agents, while providing an explainable combination of evidence for claim verification.

**arXiv ID:** 2510.24303
</details>

<details>
<summary><strong>VDSAgents: A PCS-Guided Multi-Agent System for Veridical Data Science Automation</strong> - Yunxuan Jiang, Silan Hu, Xiaoning Wang, Yuanyuan Zhang, Xiangyu Chang - [[pdf]](https://arxiv.org/pdf/2510.24339)</summary>

**Abstract:** Large language models (LLMs) become increasingly integrated into data science workflows for automated system design. However, these LLM-driven data science systems rely solely on the internal reasoning of LLMs, lacking guidance from scientific and theoretical principles. This limits their trustworthiness and robustness, especially when dealing with noisy and complex real-world datasets. This paper provides VDSAgents, a multi-agent system grounded in the Predictability-Computability-Stability (PCS) principles proposed in the Veridical Data Science (VDS) framework. Guided by PCS principles, the system implements a modular workflow for data cleaning, feature engineering, modeling, and evaluation. Each phase is handled by an elegant agent, incorporating perturbation analysis, unit testing, and model validation to ensure both functionality and scientific auditability. We evaluate VDSAgents on nine datasets with diverse characteristics, comparing it with state-of-the-art end-to-end data science systems, such as AutoKaggle and DataInterpreter, using DeepSeek-V3 and GPT-4o as backends. VDSAgents consistently outperforms the results of AutoKaggle and DataInterpreter, which validates the feasibility of embedding PCS principles into LLM-driven data science automation.

**arXiv ID:** 2510.24339
</details>

<details>
<summary><strong>Policy Cards: Machine-Readable Runtime Governance for Autonomous AI Agents</strong> - Juraj Mavračić - [[pdf]](https://arxiv.org/pdf/2510.24383)</summary>

**Abstract:** Policy Cards are introduced as a machine-readable, deployment-layer standard for expressing operational, regulatory, and ethical constraints for AI agents. The Policy Card sits with the agent and enables it to follow required constraints at runtime. It tells the agent what it must and must not do. As such, it becomes an integral part of the deployed agent. Policy Cards extend existing transparency artifacts such as Model, Data, and System Cards by defining a normative layer that encodes allow/deny rules, obligations, evidentiary requirements, and crosswalk mappings to assurance frameworks including NIST AI RMF, ISO/IEC 42001, and the EU AI Act. Each Policy Card can be validated automatically, version-controlled, and linked to runtime enforcement or continuous-audit pipelines. The framework enables verifiable compliance for autonomous agents, forming a foundation for distributed assurance in multi-agent ecosystems. Policy Cards provide a practical mechanism for integrating high-level governance with hands-on engineering practice and enabling accountable autonomy at scale.

**arXiv ID:** 2510.24383
</details>

<details>
<summary><strong>SynAD: Enhancing Real-World End-to-End Autonomous Driving Models through Synthetic Data Integration</strong> - Jongsuk Kim, Jaeyoung Lee, Gyojin Han, Dongjae Lee, Minki Jeong, Junmo Kim - [[pdf]](https://arxiv.org/pdf/2510.24052)</summary>

**Abstract:** Recent advancements in deep learning and the availability of high-quality real-world driving datasets have propelled end-to-end autonomous driving. Despite this progress, relying solely on real-world data limits the variety of driving scenarios for training. Synthetic scenario generation has emerged as a promising solution to enrich the diversity of training data; however, its application within E2E AD models remains largely unexplored. This is primarily due to the absence of a designated ego vehicle and the associated sensor inputs, such as camera or LiDAR, typically provided in real-world scenarios. To address this gap, we introduce SynAD, the first framework designed to enhance real-world E2E AD models using synthetic data. Our method designates the agent with the most comprehensive driving information as the ego vehicle in a multi-agent synthetic scenario. We further project path-level scenarios onto maps and employ a newly developed Map-to-BEV Network to derive bird's-eye-view features without relying on sensor inputs. Finally, we devise a training strategy that effectively integrates these map-based synthetic data with real driving data. Experimental results demonstrate that SynAD effectively integrates all components and notably enhances safety performance. By bridging synthetic scenario generation and E2E AD, SynAD paves the way for more comprehensive and robust autonomous driving models.

**arXiv ID:** 2510.24052
</details>

<details>
<summary><strong>Multi-Agent Scenario Generation in Roundabouts with a Transformer-enhanced Conditional Variational Autoencoder</strong> - Li Li, Tobias Brinkmann, Till Temmen, Markus Eisenbarth, Jakob Andert - [[pdf]](https://arxiv.org/pdf/2510.24671)</summary>

**Abstract:** With the increasing integration of intelligent driving functions into serial-produced vehicles, ensuring their functionality and robustness poses greater challenges. Compared to traditional road testing, scenario-based virtual testing offers significant advantages in terms of time and cost efficiency, reproducibility, and exploration of edge cases. We propose a Transformer-enhanced Conditional Variational Autoencoder (CVAE-T) model for generating multi-agent traffic scenarios in roundabouts, which are characterized by high vehicle dynamics and complex layouts, yet remain relatively underexplored in current research. The results show that the proposed model can accurately reconstruct original scenarios and generate realistic, diverse synthetic scenarios. Besides, two Key-Performance-Indicators (KPIs) are employed to evaluate the interactive behavior in the generated scenarios. Analysis of the latent space reveals partial disentanglement, with several latent dimensions exhibiting distinct and interpretable effects on scenario attributes such as vehicle entry timing, exit timing, and velocity profiles. The results demonstrate the model's capability to generate scenarios for the validation of intelligent driving functions involving multi-agent interactions, as well as to augment data for their development and iterative improvement.

**arXiv ID:** 2510.24671
</details>

<details>
<summary><strong>A Survey on Large Language Model-Based Game Agents</strong> - Sihao Hu, Tiansheng Huang, Gaowen Liu, Ramana Rao Kompella, Fatih Ilhan, Selim Furkan Tekin, Yichang Xu, Zachary Yahn, Ling Liu - [[pdf]](https://arxiv.org/pdf/2404.02039)</summary>

**Abstract:** Game environments provide rich, controllable settings that stimulate many aspects of real-world complexity. As such, game agents offer a valuable testbed for exploring capabilities relevant to Artificial General Intelligence. Recently, the emergence of Large Language Models (LLMs) provides new opportunities to endow these agents with generalizable reasoning, memory, and adaptability in complex game environments. This survey offers an up-to-date review of LLM-based game agents (LLMGAs) through a unified reference architecture. At the single-agent level, we synthesize existing studies around three core components: memory, reasoning, and perception-action interfaces, which jointly characterize how language enables agents to perceive, think, and act. At the multi-agent level, we outline how communication protocols and organizational models support coordination, role differentiation, and large-scale social behaviors. To contextualize these designs, we introduce a challenge-centered taxonomy linking six major game genres to their dominant agent requirements, from low-latency control in action games to open-ended goal formation in sandbox worlds. A curated list of related papers is available at this https URL

**arXiv ID:** 2404.02039
</details>

<details>
<summary><strong>Evaluating the Use of Large Language Models as Synthetic Social Agents in Social Science Research</strong> - Emma Rose Madden - [[pdf]](https://arxiv.org/pdf/2509.26080)</summary>

**Abstract:** Large Language Models (LLMs) are being increasingly used as synthetic agents in social science, in applications ranging from augmenting survey responses to powering multi-agent simulations. This paper outlines cautions that should be taken when interpreting LLM outputs and proposes a pragmatic reframing for the social sciences in which LLMs are used as high-capacity pattern matchers for quasi-predictive interpolation under explicit scope conditions and not as substitutes for probabilistic inference. Practical guardrails such as independent draws, preregistered human baselines, reliability-aware validation, and subgroup calibration, are introduced so that researchers may engage in useful prototyping and forecasting while avoiding category errors.

**arXiv ID:** 2509.26080
</details>

<details>
<summary><strong>Co-TAP: Three-Layer Agent Interaction Protocol Technical Report</strong> - Shunyu An, Miao Wang, Yongchao Li, Dong Wan, Lina Wang, Ling Qin, Liqin Gao, Congyao Fan, Zhiyong Mao, Jiange Pu, Wenji Xia, Dong Zhao, Zhaohui Hao, Rui Hu, Ji Lu, Guiyue Zhou, Baoyu Tang, Yanqin Gao, Yongsheng Du, Daigang Xu, Lingjun Huang, Baoli Wang, Xiwen Zhang, Luyao Wang, Shilong Liu - [[pdf]](https://arxiv.org/pdf/2510.08263)</summary>

**Abstract:** This paper proposes Co-TAP (T: Triple, A: Agent, P: Protocol), a three-layer agent interaction protocol designed to address the challenges faced by multi-agent systems across the three core dimensions of Interoperability, Interaction and Collaboration, and Knowledge Sharing. We have designed and proposed a layered solution composed of three core protocols: the Human-Agent Interaction Protocol (HAI), the Unified Agent Protocol (UAP), and the Memory-Extraction-Knowledge Protocol (MEK). HAI focuses on the interaction layer, standardizing the flow of information between users, interfaces, and agents by defining a standardized, event-driven communication paradigm. This ensures the real-time performance, reliability, and synergy of interactions. As the core of the infrastructure layer, UAP is designed to break down communication barriers among heterogeneous agents through unified service discovery and protocol conversion mechanisms, thereby enabling seamless interconnection and interoperability of the underlying network. MEK, in turn, operates at the cognitive layer. By establishing a standardized ''Memory (M) - Extraction (E) - Knowledge (K)'' cognitive chain, it empowers agents with the ability to learn from individual experiences and form shareable knowledge, thereby laying the foundation for the realization of true collective intelligence. We believe this protocol framework will provide a solid engineering foundation and theoretical guidance for building the next generation of efficient, scalable, and intelligent multi-agent applications.

**arXiv ID:** 2510.08263
</details>

<details>
<summary><strong>Multi-Agent Evolve: LLM Self-Improve through Co-evolution</strong> - Yixing Chen, Yiding Wang, Siqi Zhu, Haofei Yu, Tao Feng, Muhan Zhang, Mostofa Patwary, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2510.23595)</summary>

**Abstract:** Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision.

**arXiv ID:** 2510.23595
</details>

<details>
<summary><strong>Logic-based Task Representation and Reward Shaping in Multiagent Reinforcement Learning</strong> - Nishant Doshi - [[pdf]](https://arxiv.org/pdf/2510.23615)</summary>

**Abstract:** This paper presents an approach for accelerated learning of optimal plans for a given task represented using Linear Temporal Logic (LTL) in multi-agent systems. Given a set of options (temporally abstract actions) available to each agent, we convert the task specification into the corresponding Buchi Automaton and proceed with a model-free approach which collects transition samples and constructs a product Semi Markov Decision Process (SMDP) on-the-fly. Value-based Reinforcement Learning algorithms can then be used to synthesize a correct-by-design controller without learning the underlying transition model of the multi-agent system. The exponential sample complexity due to multiple agents is dealt with using a novel reward shaping approach. We test the proposed algorithm in a deterministic gridworld simulation for different tasks and find that the reward shaping results in significant reduction in convergence times. We also infer that using options becomes increasing more relevant as the state and action space increases in multi-agent systems.

**arXiv ID:** 2510.23615
</details>

<details>
<summary><strong>Coordinated Autonomous Drones for Human-Centered Fire Evacuation in Partially Observable Urban Environments</strong> - Maria G. Mendoza, Addison Kalanther, Daniel Bostwick, Emma Stephan, Chinmay Maheshwari, Shankar Sastry - [[pdf]](https://arxiv.org/pdf/2510.23899)</summary>

**Abstract:** Autonomous drone technology holds significant promise for enhancing search and rescue operations during evacuations by guiding humans toward safety and supporting broader emergency response efforts. However, their application in dynamic, real-time evacuation support remains limited. Existing models often overlook the psychological and emotional complexity of human behavior under extreme stress. In real-world fire scenarios, evacuees frequently deviate from designated safe routes due to panic and uncertainty. To address these challenges, this paper presents a multi-agent coordination framework in which autonomous Unmanned Aerial Vehicles (UAVs) assist human evacuees in real-time by locating, intercepting, and guiding them to safety under uncertain conditions. We model the problem as a Partially Observable Markov Decision Process (POMDP), where two heterogeneous UAV agents, a high-level rescuer (HLR) and a low-level rescuer (LLR), coordinate through shared observations and complementary capabilities. Human behavior is captured using an agent-based model grounded in empirical psychology, where panic dynamically affects decision-making and movement in response to environmental stimuli. The environment features stochastic fire spread, unknown evacuee locations, and limited visibility, requiring UAVs to plan over long horizons to search for humans and adapt in real-time. Our framework employs the Proximal Policy Optimization (PPO) algorithm with recurrent policies to enable robust decision-making in partially observable settings. Simulation results demonstrate that the UAV team can rapidly locate and intercept evacuees, significantly reducing the time required for them to reach safety compared to scenarios without UAV assistance.

**arXiv ID:** 2510.23899
</details>

<details>
<summary><strong>Human Machine Social Hybrid Intelligence:A Collaborative Decision Making Framework for Large Model Agent Groups and Human Experts</strong> - Ahmet Akkaya Melih, Yamuna Singh, Kunal L. Agarwal, Priya Mukherjee, Kiran Pattnaik, Hanuman Bhatia - [[pdf]](https://arxiv.org/pdf/2510.24030)</summary>

**Abstract:** The rapid advancements in large foundation models and multi-agent systems offer unprecedented capabilities, yet current Human-in-the-Loop (HiTL) paradigms inadequately integrate human expertise, often leading to cognitive overload and decision-making bottlenecks in complex, high-stakes environments. We propose the "Human-Machine Social Hybrid Intelligence" (HMS-HI) framework, a novel architecture designed for deep, collaborative decision-making between groups of human experts and LLM-powered AI agents. HMS-HI is built upon three core pillars: (1) a \textbf{Shared Cognitive Space (SCS)} for unified, multi-modal situational awareness and structured world modeling; (2) a \textbf{Dynamic Role and Task Allocation (DRTA)} module that adaptively assigns tasks to the most suitable agent (human or AI) based on capabilities and workload; and (3) a \textbf{Cross-Species Trust Calibration (CSTC)} protocol that fosters transparency, accountability, and mutual adaptation through explainable declarations and structured feedback. Validated in a high-fidelity urban emergency response simulation, HMS-HI significantly reduced civilian casualties by 72\% and cognitive load by 70\% compared to traditional HiTL approaches, demonstrating superior decision quality, efficiency, and human-AI trust. An ablation study confirms the critical contribution of each module, highlighting that engineered trust and shared context are foundational for scalable, synergistic human-AI collaboration.

**arXiv ID:** 2510.24030
</details>

<details>
<summary><strong>Long-Term Mapping of the Douro River Plume with Multi-Agent Reinforcement Learning</strong> - Nicolò Dal Fabbro, Milad Mesbahi, Renato Mendes, João Borges de Sousa, George J. Pappas - [[pdf]](https://arxiv.org/pdf/2510.03534)</summary>

**Abstract:** We study the problem of long-term (multiple days) mapping of a river plume using multiple autonomous underwater vehicles (AUVs), focusing on the Douro river representative use-case. We propose an energy - and communication - efficient multi-agent reinforcement learning approach in which a central coordinator intermittently communicates with the AUVs, collecting measurements and issuing commands. Our approach integrates spatiotemporal Gaussian process regression (GPR) with a multi-head Q-network controller that regulates direction and speed for each AUV. Simulations using the Delft3D ocean model demonstrate that our method consistently outperforms both single- and multi-agent benchmarks, with scaling the number of agents both improving mean squared error (MSE) and operational endurance. In some instances, our algorithm demonstrates that doubling the number of AUVs can more than double endurance while maintaining or improving accuracy, underscoring the benefits of multi-agent coordination. Our learned policies generalize across unseen seasonal regimes over different months and years, demonstrating promise for future developments of data-driven long-term monitoring of dynamic plume environments.

**arXiv ID:** 2510.03534
</details>

<details>
<summary><strong>Module checking of pushdown multi-agent systems</strong> - Laura Bozzelli, Aniello Murano, Adriano Peron - [[pdf]](https://arxiv.org/pdf/2003.04728)</summary>

**Abstract:** In this paper, we investigate the module-checking problem of pushdown multi-agent systems (PMS) against ATL and ATL* specifications. We establish that for ATL, module checking of PMS is 2EXPTIME-complete, which is the same complexity as pushdown module-checking for CTL. On the other hand, we show that ATL* module-checking of PMS turns out to be 4EXPTIME-complete, hence exponentially harder than both CTL* pushdown module-checking and ATL* model-checking of PMS. Our result for ATL* provides a rare example of a natural decision problem that is elementary yet but with a complexity that is higher than triply exponential-time.

**arXiv ID:** 2003.04728
</details>

<details>
<summary><strong>A cutting-surface consensus approach for distributed robust optimization of multi-agent systems</strong> - Jun Fu, Xunhao Wu - [[pdf]](https://arxiv.org/pdf/2309.03519)</summary>

**Abstract:** A novel and fully distributed optimization method is proposed for the distributed robust convex program (DRCP) over a time-varying unbalanced directed network under the uniformly jointly strongly connected (UJSC) assumption. Firstly, an approximated DRCP (ADRCP) is introduced by discretizing the semi-infinite constraints into a finite number of inequality constraints to ensure tractability and restricting the right-hand side of the constraints with a positive parameter to ensure a feasible solution for (DRCP) can be obtained. This problem is iteratively solved by a distributed projected gradient algorithm proposed in this paper, which is based on epigraphic reformulation and gradient projected operations. Secondly, a cutting-surface consensus approach is proposed for locating an approximately optimal consensus solution of the DRCP with guaranteed local feasibility for each agent. This approach is based on iteratively approximating the DRCP by successively reducing the restriction parameter of the right-hand constraints and adding the cutting-surfaces into the existing finite set of constraints. Thirdly, to ensure finite-time termination of the distributed optimization, a distributed termination algorithm is developed based on consensus and zeroth-order stopping conditions under UJSC graphs. Fourthly, it is proved that the cutting-surface consensus approach terminates finitely and yields a feasible and approximate optimal solution for each agent. Finally, the effectiveness of the approach is illustrated through a numerical example.

**arXiv ID:** 2309.03519
</details>

<details>
<summary><strong>From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users</strong> - Sadia Sultana Chowa, Riasad Alvi, Subhey Sadi Rahman, Md Abdur Rahman, Mohaimenul Azam Khan Raiaan, Md Rafiqul Islam, Mukhtar Hussain, Sami Azam - [[pdf]](https://arxiv.org/pdf/2508.17281)</summary>

**Abstract:** The pursuit of human-level artificial intelligence (AI) has significantly advanced the development of autonomous agents and Large Language Models (LLMs). LLMs are now widely utilized as decision-making agents for their ability to interpret instructions, manage sequential tasks, and adapt through feedback. This review examines recent developments in employing LLMs as autonomous agents and tool users and comprises seven research questions. We only used the papers published between 2023 and 2025 in conferences of the A* and A rank and Q1 journals. A structured analysis of the LLM agents' architectural design principles, dividing their applications into single-agent and multi-agent systems, and strategies for integrating external tools is presented. In addition, the cognitive mechanisms of LLM, including reasoning, planning, and memory, and the impact of prompting methods and fine-tuning procedures on agent performance are also investigated. Furthermore, we evaluated current benchmarks and assessment protocols and have provided an analysis of 68 publicly available datasets to assess the performance of LLM-based agents in various tasks. In conducting this review, we have identified critical findings on verifiable reasoning of LLMs, the capacity for self-improvement, and the personalization of LLM-based agents. Finally, we have discussed ten future research directions to overcome these gaps.

**arXiv ID:** 2508.17281
</details>

<details>
<summary><strong>URB - Urban Routing Benchmark for RL-equipped Connected Autonomous Vehicles</strong> - Ahmet Onur Akman, Anastasia Psarou, Michał Hoffmann, Łukasz Gorczyca, Łukasz Kowalski, Paweł Gora, Grzegorz Jamróz, Rafał Kucharski - [[pdf]](https://arxiv.org/pdf/2505.17734)</summary>

**Abstract:** Connected Autonomous Vehicles (CAVs) promise to reduce congestion in future urban networks, potentially by optimizing their routing decisions. Unlike for human drivers, these decisions can be made with collective, data-driven policies, developed using machine learning algorithms. Reinforcement learning (RL) can facilitate the development of such collective routing strategies, yet standardized and realistic benchmarks are missing. To that end, we present URB: Urban Routing Benchmark for RL-equipped Connected Autonomous Vehicles. URB is a comprehensive benchmarking environment that unifies evaluation across 29 real-world traffic networks paired with realistic demand patterns. URB comes with a catalog of predefined tasks, multi-agent RL (MARL) algorithm implementations, three baseline methods, domain-specific performance metrics, and a modular configuration scheme. Our results show that, despite the lengthy and costly training, state-of-the-art MARL algorithms rarely outperformed humans. The experimental results reported in this paper initiate the first leaderboard for MARL in large-scale urban routing optimization. They reveal that current approaches struggle to scale, emphasizing the urgent need for advancements in this domain.

**arXiv ID:** 2505.17734
</details>

<details>
<summary><strong>Taxonomy and Trends in Reinforcement Learning for Robotics and Control Systems: A Structured Review</strong> - Kumater Ter, Ore-Ofe Ajayi, Daniel Udekwe - [[pdf]](https://arxiv.org/pdf/2510.21758)</summary>

**Abstract:** Reinforcement learning (RL) has become a foundational approach for enabling intelligent robotic behavior in dynamic and uncertain environments. This work presents an in-depth review of RL principles, advanced deep reinforcement learning (DRL) algorithms, and their integration into robotic and control systems. Beginning with the formalism of Markov Decision Processes (MDPs), the study outlines essential elements of the agent-environment interaction and explores core algorithmic strategies including actor-critic methods, value-based learning, and policy gradients. Emphasis is placed on modern DRL techniques such as DDPG, TD3, PPO, and SAC, which have shown promise in solving high-dimensional, continuous control tasks. A structured taxonomy is introduced to categorize RL applications across domains such as locomotion, manipulation, multi-agent coordination, and human-robot interaction, along with training methodologies and deployment readiness levels. The review synthesizes recent research efforts, highlighting technical trends, design patterns, and the growing maturity of RL in real-world robotics. Overall, this work aims to bridge theoretical advances with practical implementations, providing a consolidated perspective on the evolving role of RL in autonomous robotic systems.

**arXiv ID:** 2510.21758
</details>

<details>
<summary><strong>Towards AI as Colleagues: Multi-Agent System Improves Structured Professional Ideation</strong> - Kexin Quan, Dina Albassam, Mengke Wu, Zijian Ding, Jessie Chin - [[pdf]](https://arxiv.org/pdf/2510.23904)</summary>

**Abstract:** Most AI systems today are designed to manage tasks and execute predefined steps. This makes them effective for process coordination but limited in their ability to engage in joint problem-solving with humans or contribute new ideas. We introduce MultiColleagues, a multi-agent conversational system that shows how AI agents can act as colleagues by conversing with each other, sharing new ideas, and actively involving users in collaborative ideation. In a within-subjects study with 20 participants, we compared MultiColleagues to a single-agent baseline. Results show that MultiColleagues fostered stronger perceptions of social presence, produced ideas rated significantly higher in quality and novelty, and encouraged deeper elaboration. These findings demonstrate the potential of AI agents to move beyond process partners toward colleagues that share intent, strengthen group dynamics, and collaborate with humans to advance ideas.

**arXiv ID:** 2510.23904
</details>

</details>

<details open>
<summary><h2>Other Agent Research (14 papers)</h2></summary>

<details>
<summary><strong>Agentsway -- Software Development Methodology for AI Agents-based Teams</strong> - Eranga Bandara, Ross Gore, Xueping Liang, Sachini Rajapakse, Isurunima Kularathne, Pramoda Karunarathna, Peter Foytik, Sachin Shetty, Ravi Mukkamala, Abdul Rahman, Amin Hass, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan - [[pdf]](https://arxiv.org/pdf/2510.23664)</summary>

**Abstract:** The emergence of Agentic AI is fundamentally transforming how software is designed, developed, and maintained. Traditional software development methodologies such as Agile, Kanban, ShapeUp, etc, were originally designed for human-centric teams and are increasingly inadequate in environments where autonomous AI agents contribute to planning, coding, testing, and continuous learning. To address this methodological gap, we present "Agentsway" a novel software development framework designed for ecosystems where AI agents operate as first-class collaborators. Agentsway introduces a structured lifecycle centered on human orchestration, and privacy-preserving collaboration among specialized AI agents. The framework defines distinct roles for planning, prompting, coding, testing, and fine-tuning agents, each contributing to iterative improvement and adaptive learning throughout the development process. By integrating fine-tuned LLMs that leverage outputs and feedback from different agents throughout the development cycle as part of a retrospective learning process, Agentsway enhances domain-specific reasoning, and explainable decision-making across the entire software development lifecycle. Responsible AI principles are further embedded across the agents through the coordinated use of multiple fine-tuned LLMs and advanced reasoning models, ensuring balanced, transparent, and accountable decision-making. This work advances software engineering by formalizing agent-centric collaboration, integrating privacy-by-design principles, and defining measurable metrics for productivity and trust. Agentsway represents a foundational step toward the next generation of AI-native, self-improving software development methodologies. To the best of our knowledge, this is the first research effort to introduce a dedicated methodology explicitly designed for AI agent-based software engineering teams.

**arXiv ID:** 2510.23664
</details>

<details>
<summary><strong>Traffic flow forecasting, STL decomposition, Hybrid model, LSTM, ARIMA, XGBoost, Intelligent transportation systems</strong> - Fujiang Yuan, Yangrui Fan, Xiaohuan Bing, Zhen Tian, Chunhong Yuan, Yankang Li - [[pdf]](https://arxiv.org/pdf/2510.23668)</summary>

**Abstract:** Accurate traffic flow forecasting is essential for intelligent transportation systems and urban traffic management. However, single model approaches often fail to capture the complex, nonlinear, and multi scale temporal patterns in traffic flow data. This study proposes a decomposition driven hybrid framework that integrates Seasonal Trend decomposition using Loess (STL) with three complementary predictive models. STL first decomposes the original time series into trend, seasonal, and residual components. Then, a Long Short Term Memory (LSTM) network models long term trends, an Autoregressive Integrated Moving Average (ARIMA) model captures seasonal periodicity, and an Extreme Gradient Boosting (XGBoost) algorithm predicts nonlinear residual fluctuations. The final forecast is obtained through multiplicative integration of the sub model predictions. Using 998 traffic flow records from a New York City intersection between November and December 2015, results show that the LSTM ARIMA XGBoost hybrid model significantly outperforms standalone models including LSTM, ARIMA, and XGBoost across MAE, RMSE, and R squared metrics. The decomposition strategy effectively isolates temporal characteristics, allowing each model to specialize, thereby improving prediction accuracy, interpretability, and robustness.

**arXiv ID:** 2510.23668
</details>

<details>
<summary><strong>QueryIPI: Query-agnostic Indirect Prompt Injection on Coding Agents</strong> - Yuchong Xie, Zesen Liu, Mingyu Luo, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She - [[pdf]](https://arxiv.org/pdf/2510.23675)</summary>

**Abstract:** Modern coding agents integrated into IDEs combine powerful tools and system-level actions, exposing a high-stakes attack surface. Existing Indirect Prompt Injection (IPI) studies focus mainly on query-specific behaviors, leading to unstable attacks with lower success rates. We identify a more severe, query-agnostic threat that remains effective across diverse user inputs. This challenge can be overcome by exploiting a common vulnerability: leakage of the agent's internal prompt, which turns the attack into a constrained white-box optimization problem. We present QueryIPI, the first query-agnostic IPI method for coding agents. QueryIPI refines malicious tool descriptions through an iterative, prompt-based process informed by the leaked internal prompt. Experiments on five simulated agents show that QueryIPI achieves up to 87 percent success, outperforming baselines, and the generated malicious descriptions also transfer to real-world systems, highlighting a practical security risk to modern LLM-based coding agents.

**arXiv ID:** 2510.23675
</details>

<details>
<summary><strong>TDFlow: Agentic Workflows for Test Driven Software Engineering</strong> - Kevin Han, Siddharth Maddikayala, Tim Knappe, Om Patel, Austen Liao, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2510.23761)</summary>

**Abstract:** We introduce TDFlow, a novel test-driven agentic workflow that frames repository-scale software engineering as a test-resolution task, specifically designed to solve human-written tests. Given a set of tests, TDFlow repeatedly proposes, revises, and debugs repository-scale patches using precisely engineered sub-agents and tightly constrained tools. The workflow decomposes software engineering program repair into four components governed by respective sub-agents. This simple, forced decoupling of patch proposing, debugging, patch revision, and optional test generation (1) reduces long-context burden on any individual sub-agent, (2) focuses each sub-agent on specific, pre-defined sub-tasks, and (3) allows for specialized performance improvement on specific sub-tasks. When provided human-written tests, TDFlow attains 88.8% pass rate on SWE-Bench Lite (an absolute improvement of 27.8% over the next best system) and 94.3% on SWE-Bench Verified. Manual inspection of the 800 TDFlow runs within SWE-Bench Lite and Verified uncover only 7 instances of test hacking, which were subsequently counted as failures. Furthermore, we show that the primary obstacle to human-level software engineering performance lies within writing successful reproduction tests. We envision a human-LLM interactive system powered by TDFlow where human developers write tests solved by LLM systems. Together, these results indicate that modern LLMs, when embedded in a narrowly engineered, test-driven workflow, already achieve human-level test resolution -- with the final frontier for fully autonomous repository repair being the accurate generation of valid reproduction tests.

**arXiv ID:** 2510.23761
</details>

<details>
<summary><strong>Agent-based Automated Claim Matching with Instruction-following LLMs</strong> - Dina Pisarevskaya, Arkaitz Zubiaga - [[pdf]](https://arxiv.org/pdf/2510.23924)</summary>

**Abstract:** We present a novel agent-based approach for the automated claim matching task with instruction-following LLMs. We propose a two-step pipeline that first generates prompts with LLMs, to then perform claim matching as a binary classification task with LLMs. We demonstrate that LLM-generated prompts can outperform SOTA with human-generated prompts, and that smaller LLMs can do as well as larger ones in the generation process, allowing to save computational resources. We also demonstrate the effectiveness of using different LLMs for each step of the pipeline, i.e. using an LLM for prompt generation, and another for claim matching. Our investigation into the prompt generation process in turn reveals insights into the LLMs' understanding of claim matching.

**arXiv ID:** 2510.23924
</details>

<details>
<summary><strong>Enhancing Vision-Language Models for Autonomous Driving through Task-Specific Prompting and Spatial Reasoning</strong> - Aodi Wu, Xubo Luo - [[pdf]](https://arxiv.org/pdf/2510.24152)</summary>

**Abstract:** This technical report presents our solution for the RoboSense Challenge at IROS 2025, which evaluates Vision-Language Models (VLMs) on autonomous driving scene understanding across perception, prediction, planning, and corruption detection tasks. We propose a systematic framework built on four core components. First, a Mixture-of-Prompts router classifies questions and dispatches them to task-specific expert prompts, eliminating interference across diverse question types. Second, task-specific prompts embed explicit coordinate systems, spatial reasoning rules, role-playing, Chain-of-Thought/Tree-of-Thought reasoning, and few-shot examples tailored to each task. Third, a visual assembly module composes multi-view images with object crops, magenta markers, and adaptive historical frames based on question requirements. Fourth, we configure model inference parameters (temperature, top-p, message roles) per task to optimize output quality. Implemented on Qwen2.5-VL-72B, our approach achieves 70.87% average accuracy on Phase-1 (clean data) and 72.85% on Phase-2 (corrupted data), demonstrating that structured prompting and spatial grounding substantially enhance VLM performance on safety-critical autonomous driving tasks. Code and prompt are available at this https URL.

**arXiv ID:** 2510.24152
</details>

<details>
<summary><strong>Partner Modelling Emerges in Recurrent Agents (But Only When It Matters)</strong> - Ruaridh Mon-Williams, Max Taylor-Davies, Elizabeth Mieczkowski, Natalia Velez, Neil R. Bramley, Yanwei Wang, Thomas L. Griffiths, Christopher G. Lucas - [[pdf]](https://arxiv.org/pdf/2505.17323)</summary>

**Abstract:** Humans are remarkably adept at collaboration, able to infer the strengths and weaknesses of new partners in order to work successfully towards shared goals. To build AI systems with this capability, we must first understand its building blocks: does such flexibility require explicit, dedicated mechanisms for modelling others -- or can it emerge spontaneously from the pressures of open-ended cooperative interaction? To investigate this question, we train simple model-free RNN agents to collaborate with a population of diverse partners. Using the `Overcooked-AI' environment, we collect data from thousands of collaborative teams, and analyse agents' internal hidden states. Despite a lack of additional architectural features, inductive biases, or auxiliary objectives, the agents nevertheless develop structured internal representations of their partners' task abilities, enabling rapid adaptation and generalisation to novel collaborators. We investigated these internal models through probing techniques, and large-scale behavioural analysis. Notably, we find that structured partner modelling emerges when agents can influence partner behaviour by controlling task allocation. Our results show that partner modelling can arise spontaneously in model-free agents -- but only under environmental conditions that impose the right kind of social pressure.

**arXiv ID:** 2505.17323
</details>

<details>
<summary><strong>Central Bank Digital Currency, Flight-to-Quality, and Bank-Runs in an Agent-Based Model</strong> - Emilio Barucci, Andrea Gurgone, Giulia Iori, Michele Azzone - [[pdf]](https://arxiv.org/pdf/2510.21071)</summary>

**Abstract:** We analyse financial stability and welfare impacts associated with the introduction of a Central Bank Digital Currency (CBDC) in a macroeconomic agent-based model. The model considers firms, banks, and households interacting on labour, goods, credit, and interbank markets. Households move their liquidity from deposits to CBDC based on the perceived riskiness of their banks. We find that the introduction of CBDC exacerbates bank-runs and may lead to financial instability phenomena. The effect can be changed by introducing a limit on CBDC holdings. The adoption of CBDC has little effect on macroeconomic variables but the interest rate on loans to firms goes up and credit goes down in a limited way. CBDC leads to a redistribution of wealth from firms and banks to households with a higher bank default rate. CBDC may have negative welfare effects, but a bound on holding enables a welfare improvement.

**arXiv ID:** 2510.21071
</details>

<details>
<summary><strong>DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</strong> - Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2510.14205)</summary>

**Abstract:** The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these this http URL evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie this http URL can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and this http URL work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.

**arXiv ID:** 2510.14205
</details>

<details>
<summary><strong>BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents</strong> - Litu Ou, Kuan Li, Huifeng Yin, Liwen Zhang, Zhongwang Zhang, Xixi Wu, Rui Ye, Zile Qiao, Pengjun Xie, Jingren Zhou, Yong Jiang - [[pdf]](https://arxiv.org/pdf/2510.23458)</summary>

**Abstract:** Confidence in LLMs is a useful indicator of model uncertainty and answer reliability. Existing work mainly focused on single-turn scenarios, while research on confidence in complex multi-turn interactions is limited. In this paper, we investigate whether LLM-based search agents have the ability to communicate their own confidence through verbalized confidence scores after long sequences of actions, a significantly more challenging task compared to outputting confidence in a single interaction. Experimenting on open-source agentic models, we first find that models exhibit much higher task accuracy at high confidence while having near-zero accuracy when confidence is low. Based on this observation, we propose Test-Time Scaling (TTS) methods that use confidence scores to determine answer quality, encourage the model to try again until reaching a satisfactory confidence level. Results show that our proposed methods significantly reduce token consumption while demonstrating competitive performance compared to baseline fixed budget TTS methods.

**arXiv ID:** 2510.23458
</details>

<details>
<summary><strong>Inferring Group Intent as a Cooperative Game. An NLP-based Framework for Trajectory Analysis using Graph Transformer Neural Network</strong> - Yiming Zhang, Vikram Krishnamurthy, Shashwat Jain - [[pdf]](https://arxiv.org/pdf/2510.23905)</summary>

**Abstract:** This paper studies group target trajectory intent as the outcome of a cooperative game where the complex-spatio trajectories are modeled using an NLP-based generative model. In our framework, the group intent is specified by the characteristic function of a cooperative game, and allocations for players in the cooperative game are specified by either the core, the Shapley value, or the nucleolus. The resulting allocations induce probability distributions that govern the coordinated spatio-temporal trajectories of the targets that reflect the group's underlying intent. We address two key questions: (1) How can the intent of a group trajectory be optimally formalized as the characteristic function of a cooperative game? (2) How can such intent be inferred from noisy observations of the targets? To answer the first question, we introduce a Fisher-information-based characteristic function of the cooperative game, which yields probability distributions that generate coordinated spatio-temporal patterns. As a generative model for these patterns, we develop an NLP-based generative model built on formal grammar, enabling the creation of realistic multi-target trajectory data. To answer the second question, we train a Graph Transformer Neural Network (GTNN) to infer group trajectory intent-expressed as the characteristic function of the cooperative game-from observational data with high accuracy. The self-attention function of the GTNN depends on the track estimates. Thus, the formulation and algorithms provide a multi-layer approach that spans target tracking (Bayesian signal processing) and the GTNN (for group intent inference).

**arXiv ID:** 2510.23905
</details>

<details>
<summary><strong>PFEA: An LLM-based High-Level Natural Language Planning and Feedback Embodied Agent for Human-Centered AI</strong> - Wenbin Ding, Jun Chen, Mingjia Chen, Fei Xie, Qi Mao, Philip Dames - [[pdf]](https://arxiv.org/pdf/2510.24109)</summary>

**Abstract:** The rapid advancement of Large Language Models (LLMs) has marked a significant breakthrough in Artificial Intelligence (AI), ushering in a new era of Human-centered Artificial Intelligence (HAI). HAI aims to better serve human welfare and needs, thereby placing higher demands on the intelligence level of robots, particularly in aspects such as natural language interaction, complex task planning, and execution. Intelligent agents powered by LLMs have opened up new pathways for realizing HAI. However, existing LLM-based embodied agents often lack the ability to plan and execute complex natural language control tasks online. This paper explores the implementation of intelligent robotic manipulating agents based on Vision-Language Models (VLMs) in the physical world. We propose a novel embodied agent framework for robots, which comprises a human-robot voice interaction module, a vision-language agent module and an action execution module. The vision-language agent itself includes a vision-based task planner, a natural language instruction converter, and a task performance feedback evaluator. Experimental results demonstrate that our agent achieves a 28\% higher average task success rate in both simulated and real environments compared to approaches relying solely on LLM+CLIP, significantly improving the execution success rate of high-level natural language instruction tasks.

**arXiv ID:** 2510.24109
</details>

<details>
<summary><strong>Autonomous Horizon-based Asteroid Navigation With Observability-constrained Maneuvers</strong> - Aditya Arjun Anibha, Kenshiro Oguri - [[pdf]](https://arxiv.org/pdf/2501.15806)</summary>

**Abstract:** Small body exploration is a pertinent challenge due to low gravity environments and strong sensitivity to perturbations like Solar Radiation Pressure (SRP). Thus, autonomous methods are being developed to enable safe navigation and control around small bodies. These methods often involve using Optical Navigation (OpNav) to determine the spacecraft's location. Ensuring OpNav reliability would allow the spacecraft to maintain an accurate state estimate throughout its mission. This research presents an observability-constrained Lyapunov controller that steers a spacecraft to a desired target orbit while guaranteeing continuous OpNav observability. We design observability path constraints to avoid regions where horizon-based OpNav methods exhibit poor performance, ensuring control input that maintains good observability. This controller is implemented with a framework that simulates small body dynamics, synthetic image generation, edge detection, horizon-based OpNav, and filtering. We evaluate the approach in two representative scenarios, orbit maintenance and approach with circularization, around spherical and ellipsoidal target bodies. In Monte Carlo simulations, the proposed approach improves the rate of attaining target orbits without observability violations by up to 94% compared to an unconstrained Lyapunov baseline, demonstrating improved robustness over conventional methods.

**arXiv ID:** 2501.15806
</details>

<details>
<summary><strong>SimpleVSF: VLM-Scoring Fusion for Trajectory Prediction of End-to-End Autonomous Driving</strong> - Peiru Zheng, Yun Zhao, Zhan Gong, Hong Zhu, Shaohua Wu - [[pdf]](https://arxiv.org/pdf/2510.17191)</summary>

**Abstract:** End-to-end autonomous driving has emerged as a promising paradigm for achieving robust and intelligent driving policies. However, existing end-to-end methods still face significant challenges, such as suboptimal decision-making in complex scenarios. In this paper,we propose SimpleVSF (Simple VLM-Scoring Fusion), a novel framework that enhances end-to-end planning by leveraging the cognitive capabilities of Vision-Language Models (VLMs) and advanced trajectory fusion techniques. We utilize the conventional scorers and the novel VLM-enhanced scorers. And we leverage a robust weight fusioner for quantitative aggregation and a powerful VLM-based fusioner for qualitative, context-aware decision-making. As the leading approach in the ICCV 2025 NAVSIM v2 End-to-End Driving Challenge, our SimpleVSF framework demonstrates state-of-the-art performance, achieving a superior balance between safety, comfort, and efficiency.

**arXiv ID:** 2510.17191
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (29 papers)</h2></summary>

<details>
<summary><strong>From Benchmarks to Business Impact: Deploying IBM Generalist Agent in Enterprise Production</strong> - Segev Shlomov, Alon Oved, Sami Marreed, Ido Levy, Offer Akrabi, Avi Yaeli, Łukasz Strąk, Elizabeth Koumpan, Yinon Goldshtein, Eilam Shapira, Nir Mashkif, Asaf Adi - [[pdf]](https://arxiv.org/pdf/2510.23856)</summary>

**Abstract:** Agents are rapidly advancing in automating digital work, but enterprises face a harder challenge: moving beyond prototypes to deployed systems that deliver measurable business value. This path is complicated by fragmented frameworks, slow development, and the absence of standardized evaluation practices. Generalist agents have emerged as a promising direction, excelling on academic benchmarks and offering flexibility across task types, applications, and modalities. Yet, evidence of their use in production enterprise settings remains limited. This paper reports IBM's experience developing and piloting the Computer Using Generalist Agent (CUGA), which has been open-sourced for the community (this https URL). CUGA adopts a hierarchical planner--executor architecture with strong analytical foundations, achieving state-of-the-art performance on AppWorld and WebArena. Beyond benchmarks, it was evaluated in a pilot within the Business-Process-Outsourcing talent acquisition domain, addressing enterprise requirements for scalability, auditability, safety, and governance. To support assessment, we introduce BPO-TA, a 26-task benchmark spanning 13 analytics endpoints. In preliminary evaluations, CUGA approached the accuracy of specialized agents while indicating potential for reducing development time and cost. Our contribution is twofold: presenting early evidence of generalist agents operating at enterprise scale, and distilling technical and organizational lessons from this initial pilot. We outline requirements and next steps for advancing research-grade architectures like CUGA into robust, enterprise-ready systems.

**arXiv ID:** 2510.23856
</details>

<details>
<summary><strong>Hybrid Modeling, Sim-to-Real Reinforcement Learning, and Large Language Model Driven Control for Digital Twins</strong> - Adil Rasheed, Oscar Ravik, Omer San - [[pdf]](https://arxiv.org/pdf/2510.23882)</summary>

**Abstract:** This work investigates the use of digital twins for dynamical system modeling and control, integrating physics-based, data-driven, and hybrid approaches with both traditional and AI-driven controllers. Using a miniature greenhouse as a test platform, four predictive models Linear, Physics-Based Modeling (PBM), Long Short Term Memory (LSTM), and Hybrid Analysis and Modeling (HAM) are developed and compared under interpolation and extrapolation scenarios. Three control strategies Model Predictive Control (MPC), Reinforcement Learning (RL), and Large Language Model (LLM) based control are also implemented to assess trade-offs in precision, adaptability, and implementation effort. Results show that in modeling HAM provides the most balanced performance across accuracy, generalization, and computational efficiency, while LSTM achieves high precision at greater resource cost. Among controllers, MPC delivers robust and predictable performance, RL demonstrates strong adaptability, and LLM-based controllers offer flexible human-AI interaction when coupled with predictive tools.

**arXiv ID:** 2510.23882
</details>

<details>
<summary><strong>Agentic AI Security: Threats, Defenses, Evaluation, and Open Challenges</strong> - Shrestha Datta, Shahriar Kabir Nahin, Anshuman Chhabra, Prasant Mohapatra - [[pdf]](https://arxiv.org/pdf/2510.23883)</summary>

**Abstract:** Agentic AI systems powered by large language models (LLMs) and endowed with planning, tool use, memory, and autonomy, are emerging as powerful, flexible platforms for automation. Their ability to autonomously execute tasks across web, software, and physical environments creates new and amplified security risks, distinct from both traditional AI safety and conventional software security. This survey outlines a taxonomy of threats specific to agentic AI, reviews recent benchmarks and evaluation methodologies, and discusses defense strategies from both technical and governance perspectives. We synthesize current research and highlight open challenges, aiming to support the development of secure-by-design agent systems.

**arXiv ID:** 2510.23883
</details>

<details>
<summary><strong>APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training</strong> - Jiarui Qin, Yunjia Xi, Junjie Huang, Renting Rui, Di Yin, Weiwen Liu, Yong Yu, Weinan Zhang, Xing Sun - [[pdf]](https://arxiv.org/pdf/2510.24397)</summary>

**Abstract:** With the rapid development of LLM-based agents, there is a growing trend to incorporate agent-specific data into the pre-training stage of LLMs, aiming to better align LLMs with real-world autonomous task execution. However, current pre-training benchmarks primarily focus on isolated and static skills, e.g., common knowledge or mathematical/code reasoning, and fail to reflect model's agentic capabilities. On the other hand, agent benchmarks are typically designed for post-trained models, requiring multi-turn task execution abilities that base models struggle to support. Thus, there is a compelling need for a benchmark that can evaluate agentic potentials during pre-training and guide the model training more effectively. To address this gap, we propose APTBench, a framework that converts real-world agent tasks and successful trajectories into multiple-choice or text completion questions tailored for base models. It focuses on core agentic abilities, e.g., planning and action, and covers key agent scenarios, software engineering and deep research. Compared to existing general-purpose benchmarks, APTBench offers a more predictive signal of a model's downstream performance as an agent, while remaining significantly more lightweight and cost-effective than full-scale, end-to-end agent evaluations after post-training.

**arXiv ID:** 2510.24397
</details>

<details>
<summary><strong>Law in Silico: Simulating Legal Society with LLM-Based Agents</strong> - Yiding Wang, Yuxuan Chen, Fanxu Meng, Xifan Chen, Xiaolei Yang, Muhan Zhang - [[pdf]](https://arxiv.org/pdf/2510.24442)</summary>

**Abstract:** Since real-world legal experiments are often costly or infeasible, simulating legal societies with Artificial Intelligence (AI) systems provides an effective alternative for verifying and developing legal theory, as well as supporting legal administration. Large Language Models (LLMs), with their world knowledge and role-playing capabilities, are strong candidates to serve as the foundation for legal society simulation. However, the application of LLMs to simulate legal systems remains underexplored. In this work, we introduce Law in Silico, an LLM-based agent framework for simulating legal scenarios with individual decision-making and institutional mechanisms of legislation, adjudication, and enforcement. Our experiments, which compare simulated crime rates with real-world data, demonstrate that LLM-based agents can largely reproduce macro-level crime trends and provide insights that align with real-world observations. At the same time, micro-level simulations reveal that a well-functioning, transparent, and adaptive legal system offers better protection of the rights of vulnerable individuals.

**arXiv ID:** 2510.24442
</details>

<details>
<summary><strong>Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks</strong> - Korneel Van den Berghe, Stein Stroobants, Vijay Janapa Reddi, G.C.H.E. de Croon - [[pdf]](https://arxiv.org/pdf/2510.24461)</summary>

**Abstract:** Neuromorphic computing systems are set to revolutionize energy-constrained robotics by achieving orders-of-magnitude efficiency gains, while enabling native temporal processing. Spiking Neural Networks (SNNs) represent a promising algorithmic approach for these systems, yet their application to complex control tasks faces two critical challenges: (1) the non-differentiable nature of spiking neurons necessitates surrogate gradients with unclear optimization properties, and (2) the stateful dynamics of SNNs require training on sequences, which in reinforcement learning (RL) is hindered by limited sequence lengths during early training, preventing the network from bridging its warm-up period.
We address these challenges by systematically analyzing surrogate gradient slope settings, showing that shallower slopes increase gradient magnitude in deeper layers but reduce alignment with true gradients. In supervised learning, we find no clear preference for fixed or scheduled slopes. The effect is much more pronounced in RL settings, where shallower slopes or scheduled slopes lead to a 2.1x improvement in both training and final deployed performance. Next, we propose a novel training approach that leverages a privileged guiding policy to bootstrap the learning process, while still exploiting online environment interactions with the spiking policy. Combining our method with an adaptive slope schedule for a real-world drone position control task, we achieve an average return of 400 points, substantially outperforming prior techniques, including Behavioral Cloning and TD3BC, which achieve at most --200 points under the same conditions. This work advances both the theoretical understanding of surrogate gradient learning in SNNs and practical training methodologies for neuromorphic controllers demonstrated in real-world robotic systems.

**arXiv ID:** 2510.24461
</details>

<details>
<summary><strong>Causal-Aware Generative Adversarial Networks with Reinforcement Learning</strong> - Tu Anh Hoang Nguyen, Dang Nguyen, Tri-Nhan Vo, Thuc Duy Le, Sunil Gupta - [[pdf]](https://arxiv.org/pdf/2510.24046)</summary>

**Abstract:** The utility of tabular data for tasks ranging from model training to large-scale data analysis is often constrained by privacy concerns or regulatory hurdles. While existing data generation methods, particularly those based on Generative Adversarial Networks (GANs), have shown promise, they frequently struggle with capturing complex causal relationship, maintaining data utility, and providing provable privacy guarantees suitable for enterprise deployment. We introduce CA-GAN, a novel generative framework specifically engineered to address these challenges for real-world tabular datasets. CA-GAN utilizes a two-step approach: causal graph extraction to learn a robust, comprehensive causal relationship in the data's manifold, followed by a custom Conditional WGAN-GP (Wasserstein GAN with Gradient Penalty) that operates exclusively as per the structure of nodes in the causal graph. More importantly, the generator is trained with a new Reinforcement Learning-based objective that aligns the causal graphs constructed from real and fake data, ensuring the causal awareness in both training and sampling phases. We demonstrate CA-GAN superiority over six SOTA methods across 14 tabular datasets. Our evaluations, focused on core data engineering metrics: causal preservation, utility preservation, and privacy preservation. Our method offers a practical, high-performance solution for data engineers seeking to create high-quality, privacy-compliant synthetic datasets to benchmark database systems, accelerate software development, and facilitate secure data-driven research.

**arXiv ID:** 2510.24046
</details>

<details>
<summary><strong>Survey and Tutorial of Reinforcement Learning Methods in Process Systems Engineering</strong> - Maximilian Bloor, Max Mowbray, Ehecatl Antonio Del Rio Chanona, Calvin Tsay - [[pdf]](https://arxiv.org/pdf/2510.24272)</summary>

**Abstract:** Sequential decision making under uncertainty is central to many Process Systems Engineering (PSE) challenges, where traditional methods often face limitations related to controlling and optimizing complex and stochastic systems. Reinforcement Learning (RL) offers a data-driven approach to derive control policies for such challenges. This paper presents a survey and tutorial on RL methods, tailored for the PSE community. We deliver a tutorial on RL, covering fundamental concepts and key algorithmic families including value-based, policy-based and actor-critic methods. Subsequently, we survey existing applications of these RL techniques across various PSE domains, such as in fed-batch and continuous process control, process optimization, and supply chains. We conclude with PSE focused discussion of specialized techniques and emerging directions. By synthesizing the current state of RL algorithm development and implications for PSE this work identifies successes, challenges, trends, and outlines avenues for future research at the interface of these fields.

**arXiv ID:** 2510.24272
</details>

<details>
<summary><strong>Critique-RL: Training Language Models for Critiquing through Two-Stage Reinforcement Learning</strong> - Zhiheng Xi, Jixuan Huang, Xin Guo, Boyang Hong, Dingwen Yang, Xiaoran Fan, Shuo Li, Zehui Chen, Junjie Ye, Siyu Yuan, Zhengyin Du, Xuesong Yao, Yufei Xu, Jiecao Chen, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang - [[pdf]](https://arxiv.org/pdf/2510.24320)</summary>

**Abstract:** Training critiquing language models to assess and provide feedback on model outputs is a promising way to improve LLMs for complex reasoning tasks. However, existing approaches typically rely on stronger supervisors for annotating critique data. To address this, we propose Critique-RL, an online RL approach for developing critiquing language models without stronger supervision. Our approach operates on a two-player paradigm: the actor generates a response, the critic provides feedback, and the actor refines the response accordingly. We first reveal that relying solely on indirect reward signals from the actor's outputs for RL optimization often leads to unsatisfactory critics: while their helpfulness (i.e., providing constructive feedback) improves, the discriminability (i.e., determining whether a response is high-quality or not) remains poor, resulting in marginal performance gains. To overcome this, Critique-RL adopts a two-stage optimization strategy. In stage I, it reinforces the discriminability of the critic with direct rule-based reward signals; in stage II, it introduces indirect rewards based on actor refinement to improve the critic's helpfulness, while maintaining its discriminability via appropriate regularization. Extensive experiments across various tasks and models show that Critique-RL delivers substantial performance improvements. For example, it achieves a 9.02% gain on in-domain tasks and a 5.70% gain on out-of-domain tasks for Qwen2.5-7B, highlighting its potential.

**arXiv ID:** 2510.24320
</details>

<details>
<summary><strong>Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems</strong> - Yihan Li, Xiyuan Fu, Ghanshyam Verma, Paul Buitelaar, Mingming Liu - [[pdf]](https://arxiv.org/pdf/2510.24476)</summary>

**Abstract:** Hallucination remains one of the key obstacles to the reliable deployment of large language models (LLMs), particularly in real-world applications. Among various mitigation strategies, Retrieval-Augmented Generation (RAG) and reasoning enhancement have emerged as two of the most effective and widely adopted approaches, marking a shift from merely suppressing hallucinations to balancing creativity and reliability. However, their synergistic potential and underlying mechanisms for hallucination mitigation have not yet been systematically examined. This survey adopts an application-oriented perspective of capability enhancement to analyze how RAG, reasoning enhancement, and their integration in Agentic Systems mitigate hallucinations. We propose a taxonomy distinguishing knowledge-based and logic-based hallucinations, systematically examine how RAG and reasoning address each, and present a unified framework supported by real-world applications, evaluations, and benchmarks.

**arXiv ID:** 2510.24476
</details>

<details>
<summary><strong>Repurposing Synthetic Data for Fine-grained Search Agent Supervision</strong> - Yida Zhao, Kuan Li, Xixi Wu, Liwen Zhang, Dingchu Zhang, Baixuan Li, Maojia Song, Zhuo Chen, Chenxi Wang, Xinyu Wang, Kewei Tu, Pengjun Xie, Jingren Zhou, Yong Jiang - [[pdf]](https://arxiv.org/pdf/2510.24694)</summary>

**Abstract:** LLM-based search agents are increasingly trained on entity-centric synthetic data to solve complex, knowledge-intensive tasks. However, prevailing training methods like Group Relative Policy Optimization (GRPO) discard this rich entity information, relying instead on sparse, outcome-based rewards. This critical limitation renders them unable to distinguish informative "near-miss" samples-those with substantially correct reasoning but a flawed final answer-from complete failures, thus discarding valuable learning signals. We address this by leveraging the very entities discarded during training. Our empirical analysis reveals a strong positive correlation between the number of ground-truth entities identified during an agent's reasoning process and final answer accuracy. Building on this insight, we introduce Entity-aware Group Relative Policy Optimization (E-GRPO), a novel framework that formulates a dense entity-aware reward function. E-GRPO assigns partial rewards to incorrect samples proportional to their entity match rate, enabling the model to effectively learn from these "near-misses". Experiments on diverse question-answering (QA) and deep research benchmarks show that E-GRPO consistently and significantly outperforms the GRPO baseline. Furthermore, our analysis reveals that E-GRPO not only achieves superior accuracy but also induces more efficient reasoning policies that require fewer tool calls, demonstrating a more effective and sample-efficient approach to aligning search agents.

**arXiv ID:** 2510.24694
</details>

<details>
<summary><strong>Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning</strong> - Léopold Maytié, Roland Bertin Johannet, Rufin VanRullen - [[pdf]](https://arxiv.org/pdf/2502.21142)</summary>

**Abstract:** Humans leverage rich internal models of the world to reason about the future, imagine counterfactuals, and adapt flexibly to new situations. In Reinforcement Learning (RL), world models aim to capture how the environment evolves in response to the agent's actions, facilitating planning and generalization. However, typical world models directly operate on the environment variables (e.g. pixels, physical attributes), which can make their training slow and cumbersome; instead, it may be advantageous to rely on high-level latent dimensions that capture relevant multimodal variables. Global Workspace (GW) Theory offers a cognitive framework for multimodal integration and information broadcasting in the brain, and recent studies have begun to introduce efficient deep learning implementations of GW. Here, we evaluate the capabilities of an RL system combining GW with a world model. We compare our GW-Dreamer with various versions of the standard PPO and the original Dreamer algorithms. We show that performing the dreaming process (i.e., mental simulation) inside the GW latent space allows for training with fewer environment steps. As an additional emergent property, the resulting model (but not its comparison baselines) displays strong robustness to the absence of one of its observation modalities (images or simulation attributes). We conclude that the combination of GW with World Models holds great potential for improving decision-making in RL agents.

**arXiv ID:** 2502.21142
</details>

<details>
<summary><strong>A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications</strong> - Minhua Lin, Zongyu Wu, Zhichao Xu, Hui Liu, Xianfeng Tang, Qi He, Charu Aggarwal, Hui Liu, Xiang Zhang, Suhang Wang - [[pdf]](https://arxiv.org/pdf/2510.16724)</summary>

**Abstract:** The advent of large language models (LLMs) has transformed information access and reasoning through open-ended natural language interaction. However, LLMs remain limited by static knowledge, factual hallucinations, and the inability to retrieve real-time or domain-specific information. Retrieval-Augmented Generation (RAG) mitigates these issues by grounding model outputs in external evidence, but traditional RAG pipelines are often single turn and heuristic, lacking adaptive control over retrieval and reasoning. Recent advances in agentic search address these limitations by enabling LLMs to plan, retrieve, and reflect through multi-step interaction with search environments. Within this paradigm, reinforcement learning (RL) offers a powerful mechanism for adaptive and self-improving search behavior. This survey provides the first comprehensive overview of \emph{RL-based agentic search}, organizing the emerging field along three complementary dimensions: (i) What RL is for (functional roles), (ii) How RL is used (optimization strategies), and (iii) Where RL is applied (scope of optimization). We summarize representative methods, evaluation protocols, and applications, and discuss open challenges and future directions toward building reliable and scalable RL driven agentic search systems. We hope this survey will inspire future research on the integration of RL and agentic search. Our repository is available at this https URL.

**arXiv ID:** 2510.16724
</details>

<details>
<summary><strong>PanicToCalm: A Proactive Counseling Agent for Panic Attacks</strong> - Jihyun Lee, Yejin Min, San Kim, Yejin Jeon, SungJun Yang, Hyounghun Kim, Gary Geunbae Lee - [[pdf]](https://arxiv.org/pdf/2510.21143)</summary>

**Abstract:** Panic attacks are acute episodes of fear and distress, in which timely, appropriate intervention can significantly help individuals regain stability. However, suitable datasets for training such models remain scarce due to ethical and logistical issues. To address this, we introduce PACE, which is a dataset that includes high-distress episodes constructed from first-person narratives, and structured around the principles of Psychological First Aid (PFA). Using this data, we train PACER, a counseling model designed to provide both empathetic and directive support, which is optimized through supervised learning and simulated preference alignment. To assess its effectiveness, we propose PanicEval, a multi-dimensional framework covering general counseling quality and crisis-specific strategies. Experimental results show that PACER outperforms strong baselines in both counselor-side metrics and client affect improvement. Human evaluations further confirm its practical value, with PACER consistently preferred over general, CBT-based, and GPT-4-powered models in panic scenarios (Code is available at this https URL ).

**arXiv ID:** 2510.21143
</details>

<details>
<summary><strong>Human-Like Goalkeeping in a Realistic Football Simulation: a Sample-Efficient Reinforcement Learning Approach</strong> - Alessandro Sestini, Joakim Bergdahl, Jean-Philippe Barrette-LaPierre, Florian Fuchs, Brady Chen, Michael Jones, Linus Gisslén - [[pdf]](https://arxiv.org/pdf/2510.23216)</summary>

**Abstract:** While several high profile video games have served as testbeds for Deep Reinforcement Learning (DRL), this technique has rarely been employed by the game industry for crafting authentic AI behaviors. Previous research focuses on training super-human agents with large models, which is impractical for game studios with limited resources aiming for human-like agents. This paper proposes a sample-efficient DRL method tailored for training and fine-tuning agents in industrial settings such as the video game industry. Our method improves sample efficiency of value-based DRL by leveraging pre-collected data and increasing network plasticity. We evaluate our method training a goalkeeper agent in EA SPORTS FC 25, one of the best-selling football simulations today. Our agent outperforms the game's built-in AI by 10% in ball saving rate. Ablation studies show that our method trains agents 50% faster compared to standard DRL methods. Finally, qualitative evaluation from domain experts indicates that our approach creates more human-like gameplay compared to hand-crafted agents. As a testimony of the impact of the approach, the method is intended to replace the hand-crafted counterpart in next iterations of the series.

**arXiv ID:** 2510.23216
</details>

<details>
<summary><strong>TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration</strong> - Yuwei Du, Jie Feng, Jie Zhao, Yong Li - [[pdf]](https://arxiv.org/pdf/2410.20445)</summary>

**Abstract:** Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. In this paper, we propose TrajAgent, an agent framework powered by large language models, designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. In TrajAgent, we first develop UniEnv, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on UniEnv, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on five tasks using four real-world datasets demonstrate the effectiveness of TrajAgent in automated trajectory modeling, achieving a performance improvement of 2.38%-69.91% over baseline methods. The codes and data can be accessed via this https URL.

**arXiv ID:** 2410.20445
</details>

<details>
<summary><strong>Reinforcement Learning for Long-Horizon Multi-Turn Search Agents</strong> - Vivek Kalyan, Martin Andrews - [[pdf]](https://arxiv.org/pdf/2510.24126)</summary>

**Abstract:** Large Language Model (LLM) agents can leverage multiple turns and tools to solve complex tasks, with prompt-based approaches achieving strong performance. This work demonstrates that Reinforcement Learning (RL) can push capabilities significantly further by learning from experience. Through experiments on a legal document search benchmark, we show that our RL-trained 14 Billion parameter model outperforms frontier class models (85% vs 78% accuracy). In addition, we explore turn-restricted regimes, during training and at test-time, that show these agents achieve better results if allowed to operate over longer multi-turn horizons.

**arXiv ID:** 2510.24126
</details>

<details>
<summary><strong>Can LLMs Translate Human Instructions into a Reinforcement Learning Agent's Internal Emergent Symbolic Representation?</strong> - Ziqi Ma, Sao Mai Nguyen, Philippe Xu - [[pdf]](https://arxiv.org/pdf/2510.24259)</summary>

**Abstract:** Emergent symbolic representations are critical for enabling developmental learning agents to plan and generalize across tasks. In this work, we investigate whether large language models (LLMs) can translate human natural language instructions into the internal symbolic representations that emerge during hierarchical reinforcement learning. We apply a structured evaluation framework to measure the translation performance of commonly seen LLMs -- GPT, Claude, Deepseek and Grok -- across different internal symbolic partitions generated by a hierarchical reinforcement learning algorithm in the Ant Maze and Ant Fall environments. Our findings reveal that although LLMs demonstrate some ability to translate natural language into a symbolic representation of the environment dynamics, their performance is highly sensitive to partition granularity and task complexity. The results expose limitations in current LLMs capacity for representation alignment, highlighting the need for further research on robust alignment between language and internal agent representations.

**arXiv ID:** 2510.24259
</details>

<details>
<summary><strong>Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards</strong> - Shangyu Xing, Siyuan Wang, Chenyuan Yang, Xinyu Dai, Xiang Ren - [[pdf]](https://arxiv.org/pdf/2510.24302)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR), particularly with algorithms like Group Relative Policy Optimization (GRPO), has proven highly effective in enhancing the reasoning capabilities of large language models. However, a critical bottleneck in current pipelines lies in the limited diversity of sampled trajectories during group rollouts. Homogeneous trajectories and their associated rewards would diminish the return signals for policy updates, thereby hindering effective policy learning. This lack of diversity stems primarily from token-level stochastic sampling, where local variations are likely to collapse into near-identical reasoning paths. To address this limitation, we propose Lookahead Tree-Based Rollouts (LATR), a novel rollout strategy designed to explicitly promotes trajectory-level diversity by enforcing branching into different candidate tokens likely to yield distinct continuations. Specifically, LATR iteratively operates in three stages: (1) branching at high-uncertainty generation steps, (2) performing lookahead simulation for each new branch, and (3) pruning branches that exhibits prolonged similarity during simulation. Compared with stochastic Sampling, LATR accelerates policy learning by 131% on average and improves final pass@1 performance by 4.2% on both GRPO and Dynamic sAmpling Policy Optimization (DAPO) algorithms across different reasoning tasks. Our code and data are publicly available at this https URL.

**arXiv ID:** 2510.24302
</details>

<details>
<summary><strong>OpenReward: Learning to Reward Long-form Agentic Tasks via Reinforcement Learning</strong> - Ziyou Hu, Zhengliang Shi, Minghang Zhu, Haitao Li, Teng Sun, Pengjie Ren, Suzan Verberne, Zhaochun Ren - [[pdf]](https://arxiv.org/pdf/2510.24636)</summary>

**Abstract:** Reward models (RMs) have become essential for aligning large language models (LLMs), serving as scalable proxies for human evaluation in both training and inference. However, existing RMs struggle on knowledge-intensive and long-form tasks, where evaluating correctness requires grounding beyond the model's internal knowledge. This limitation hinders them from reliably discriminating subtle quality differences, especially when external evidence is necessary. To address this, we introduce OpenRM, a tool-augmented long-form reward model that systematically judges open-ended responses by invoking external tools to gather relevant evidence. We train OpenRM with Group Relative Policy Optimization (GRPO) on over 27K synthesized pairwise examples generated through a controllable data synthesis framework. The training objective jointly supervises intermediate tool usage and final outcome accuracy, incentivizing our reward model to learn effective evidence-based judgment strategies. Extensive experiments on three newly-collected datasets and two widely-used benchmarks demonstrate that OpenRM substantially outperforms existing reward modeling approaches. As a further step, we integrate OpenRM into both inference-time response selection and training-time data selection. This yields consistent gains in downstream LLM alignment tasks, highlighting the potential of tool-augmented reward models for scaling reliable long-form evaluation.

**arXiv ID:** 2510.24636
</details>

<details>
<summary><strong>Evolving Diagnostic Agents in a Virtual Clinical Environment</strong> - Pengcheng Qiu, Chaoyi Wu, Junwei Liu, Qiaoyu Zheng, Yusheng Liao, Haowen Wang, Yun Yue, Qianrui Fan, Shuai Zhen, Jian Wang, Jinjie Gu, Yanfeng Wang, Ya Zhang, Weidi Xie - [[pdf]](https://arxiv.org/pdf/2510.24654)</summary>

**Abstract:** In this paper, we present a framework for training large language models (LLMs) as diagnostic agents with reinforcement learning, enabling them to manage multi-turn diagnostic processes, adaptively select examinations, and commit to final diagnoses. Unlike instruction-tuned models trained on static case summaries, our method acquires diagnostic strategies through interactive exploration and outcome-based feedback. Our contributions are fourfold: (i) We present DiagGym, a diagnostics world model trained with electronic health records that emits examination outcomes conditioned on patient history and recommended examination, serving as a virtual clinical environment for realistic diagnosis training and evaluation; (ii) We train DiagAgent via end-to-end, multi-turn reinforcement learning to learn diagnostic policies that optimize both information yield and diagnostic accuracy; (iii) We introduce DiagBench, a diagnostic benchmark comprising 750 cases with physician-validated examination recommendations and 99 cases annotated with 973 physician-written rubrics on diagnosis process; (iv) we demonstrate superior performance across diverse diagnostic settings. DiagAgent significantly outperforms 10 state-of-the-art LLMs, including DeepSeek-v3 and GPT-4o, as well as two prompt-engineered agents. In single-turn settings, DiagAgent achieves 9.34% higher diagnostic accuracy and 44.03% improvement in examination recommendation hit ratio. In end-to-end settings, it delivers 15.12% increase in diagnostic accuracy and 23.09% boost in examination recommendation F1 score. In rubric-based evaluation, it surpasses the next-best model, Claude-sonnet-4, by 7.1% in weighted rubric score. These findings indicate that learning policies in interactive clinical environments confers dynamic and clinically meaningful diagnostic management abilities unattainable through passive training alone.

**arXiv ID:** 2510.24654
</details>

<details>
<summary><strong>WebLeaper: Empowering Efficiency and Efficacy in WebAgent via Enabling Info-Rich Seeking</strong> - Zhengwei Tao, Haiyang Shen, Baixuan Li, Wenbiao Yin, Jialong Wu, Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Liwen Zhang, Xinyu Wang, Pengjun Xie, Jingren Zhou, Yong Jiang - [[pdf]](https://arxiv.org/pdf/2510.24697)</summary>

**Abstract:** Large Language Model (LLM)-based agents have emerged as a transformative approach for open-ended problem solving, with information seeking (IS) being a core capability that enables autonomous reasoning and decision-making. While prior research has largely focused on improving retrieval depth, we observe that current IS agents often suffer from low search efficiency, which in turn constrains overall performance. A key factor underlying this inefficiency is the sparsity of target entities in training tasks, which limits opportunities for agents to learn and generalize efficient search behaviors. To address these challenges, we propose WebLeaper, a framework for constructing high-coverage IS tasks and generating efficient solution trajectories. We formulate IS as a tree-structured reasoning problem, enabling a substantially larger set of target entities to be embedded within a constrained context. Leveraging curated Wikipedia tables, we propose three variants for synthesizing IS tasks, Basic, Union, and Reverse-Union, to systematically increase both IS efficiency and efficacy. Finally, we curate training trajectories by retaining only those that are simultaneously accurate and efficient, ensuring that the model is optimized for both correctness and search performance. Extensive experiments on both basic and comprehensive settings, conducted on five IS benchmarks, BrowserComp, GAIA, xbench-DeepSearch, WideSearch, and Seal-0, demonstrate that our method consistently achieves improvements in both effectiveness and efficiency over strong baselines.

**arXiv ID:** 2510.24697
</details>

<details>
<summary><strong>Automatically Benchmarking LLM Code Agents through Agent-Driven Annotation and Evaluation</strong> - Lingyue Fu, Bolun Zhang, Hao Guan, Yaoming Zhu, Lin Qiu, Weiwen Liu, Xuezhi Cao, Xunliang Cai, Weinan Zhang, Yong Yu - [[pdf]](https://arxiv.org/pdf/2510.24358)</summary>

**Abstract:** Recent advances in code agents have enabled automated software development at the project level, supported by large language models (LLMs) and widely adopted tools. However, existing benchmarks for code agent evaluation face two major limitations: high annotation cost and expertise requirements, and rigid evaluation metrics that rely primarily on unit tests. To address these challenges, we propose an agent-driven benchmark construction pipeline that leverages human supervision to efficiently generate diverse and challenging project-level tasks. Based on this approach, we introduce PRDBench, a novel benchmark comprising 50 real-world Python projects across 20 domains, each with structured Product Requirement Document (PRD) requirements, comprehensive evaluation criteria, and reference implementations. PRDBench features rich data sources, high task complexity, and flexible metrics. We further employ an Agent-as-a-Judge paradigm to score agent outputs, enabling the evaluation of various test types beyond unit tests. Extensive experiments on PRDBench demonstrate its effectiveness in assessing the capabilities of both code agents and evaluation agents, providing a scalable and robust framework for annotation and evaluation.

**arXiv ID:** 2510.24358
</details>

<details>
<summary><strong>MENTOR: A Reinforcement Learning Framework for Enabling Tool Use in Small Models via Teacher-Optimized Rewards</strong> - ChangSu Choi, Hoyun Song, Dongyeon Kim, WooHyeon Jung, Minkyung Cho, Sunjin Park, NohHyeob Bae, Seona Yu, KyungTae Lim - [[pdf]](https://arxiv.org/pdf/2510.18383)</summary>

**Abstract:** Distilling the tool-using capabilities of large language models (LLMs) into smaller, more efficient small language models (SLMs) is a key challenge for their practical application. The predominant approach, supervised fine-tuning (SFT), suffers from poor generalization as it trains models to imitate a static set of teacher trajectories rather than learn a robust methodology. While reinforcement learning (RL) offers an alternative, the standard RL using sparse rewards fails to effectively guide SLMs, causing them to struggle with inefficient exploration and adopt suboptimal strategies. To address these distinct challenges, we propose MENTOR, a framework that synergistically combines RL with teacher-guided distillation. Instead of simple imitation, MENTOR employs an RL-based process to learn a more generalizable policy through exploration. In addition, to solve the problem of reward sparsity, it uses a teacher's reference trajectory to construct a dense, composite teacher-guided reward that provides fine-grained guidance. Extensive experiments demonstrate that MENTOR significantly improves the cross-domain generalization and strategic competence of SLMs compared to both SFT and standard sparse-reward RL baselines.

**arXiv ID:** 2510.18383
</details>

<details>
<summary><strong>Structured Reinforcement Learning for Combinatorial Decision-Making</strong> - Heiko Hoppe, Léo Baty, Louis Bouvier, Axel Parmentier, Maximilian Schiffer - [[pdf]](https://arxiv.org/pdf/2505.19053)</summary>

**Abstract:** Reinforcement learning (RL) is increasingly applied to real-world problems involving complex and structured decisions, such as routing, scheduling, and assortment planning. These settings challenge standard RL algorithms, which struggle to scale, generalize, and exploit structure in the presence of combinatorial action spaces. We propose Structured Reinforcement Learning (SRL), a novel actor-critic paradigm that embeds combinatorial optimization-layers into the actor neural network. We enable end-to-end learning of the actor via Fenchel-Young losses and provide a geometric interpretation of SRL as a primal-dual algorithm in the dual of the moment polytope. Across six environments with exogenous and endogenous uncertainty, SRL matches or surpasses the performance of unstructured RL and imitation learning on static tasks and improves over these baselines by up to 92% on dynamic problems, with improved stability and convergence speed.

**arXiv ID:** 2505.19053
</details>

<details>
<summary><strong>Mixture-of-Experts Meets In-Context Reinforcement Learning</strong> - Wenhao Wu, Fuhong Liu, Haoru Li, Zican Hu, Daoyi Dong, Chunlin Chen, Zhi Wang - [[pdf]](https://arxiv.org/pdf/2506.05426)</summary>

**Abstract:** In-context reinforcement learning (ICRL) has emerged as a promising paradigm for adapting RL agents to downstream tasks through prompt conditioning. However, two notable challenges remain in fully harnessing in-context learning within RL domains: the intrinsic multi-modality of the state-action-reward data and the diverse, heterogeneous nature of decision tasks. To tackle these challenges, we propose T2MIR (Token- and Task-wise MoE for In-context RL), an innovative framework that introduces architectural advances of mixture-of-experts (MoE) into transformer-based decision models. T2MIR substitutes the feedforward layer with two parallel layers: a token-wise MoE that captures distinct semantics of input tokens across multiple modalities, and a task-wise MoE that routes diverse tasks to specialized experts for managing a broad task distribution with alleviated gradient conflicts. To enhance task-wise routing, we introduce a contrastive learning method that maximizes the mutual information between the task and its router representation, enabling more precise capture of task-relevant information. The outputs of two MoE components are concatenated and fed into the next layer. Comprehensive experiments show that T2MIR significantly facilitates in-context learning capacity and outperforms various types of baselines. We bring the potential and promise of MoE to ICRL, offering a simple and scalable architectural enhancement to advance ICRL one step closer toward achievements in language and vision communities. Our code is available at this https URL.

**arXiv ID:** 2506.05426
</details>

<details>
<summary><strong>ZTRS: Zero-Imitation End-to-end Autonomous Driving with Trajectory Scoring</strong> - Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Jingde Chen, Nadine Chang, Maying Shen, Jingyu Song, Zuxuan Wu, Shiyi Lan, Jose M. Alvarez - [[pdf]](https://arxiv.org/pdf/2510.24108)</summary>

**Abstract:** End-to-end autonomous driving maps raw sensor inputs directly into ego-vehicle trajectories to avoid cascading errors from perception modules and to leverage rich semantic cues. Existing frameworks largely rely on Imitation Learning (IL), which can be limited by sub-optimal expert demonstrations and covariate shift during deployment. On the other hand, Reinforcement Learning (RL) has recently shown potential in scaling up with simulations, but is typically confined to low-dimensional symbolic inputs (e.g. 3D objects and maps), falling short of full end-to-end learning from raw sensor data. We introduce ZTRS (Zero-Imitation End-to-End Autonomous Driving with Trajectory Scoring), a framework that combines the strengths of both worlds: sensor inputs without losing information and RL training for robust planning. To the best of our knowledge, ZTRS is the first framework that eliminates IL entirely by only learning from rewards while operating directly on high-dimensional sensor data. ZTRS utilizes offline reinforcement learning with our proposed Exhaustive Policy Optimization (EPO), a variant of policy gradient tailored for enumerable actions and rewards. ZTRS demonstrates strong performance across three benchmarks: Navtest (generic real-world open-loop planning), Navhard (open-loop planning in challenging real-world and synthetic scenarios), and HUGSIM (simulated closed-loop driving). Specifically, ZTRS achieves the state-of-the-art result on Navhard and outperforms IL-based baselines on HUGSIM. Code will be available at this https URL.

**arXiv ID:** 2510.24108
</details>

<details>
<summary><strong>Towards Quadrupedal Jumping and Walking for Dynamic Locomotion using Reinforcement Learning</strong> - Jørgen Anker Olsen, Lars Rønhaug Pettersen, Kostas Alexis - [[pdf]](https://arxiv.org/pdf/2510.24584)</summary>

**Abstract:** This paper presents a curriculum-based reinforcement learning framework for training precise and high-performance jumping policies for the robot `Olympus'. Separate policies are developed for vertical and horizontal jumps, leveraging a simple yet effective strategy. First, we densify the inherently sparse jumping reward using the laws of projectile motion. Next, a reference state initialization scheme is employed to accelerate the exploration of dynamic jumping behaviors without reliance on reference trajectories. We also present a walking policy that, when combined with the jumping policies, unlocks versatile and dynamic locomotion capabilities. Comprehensive testing validates walking on varied terrain surfaces and jumping performance that exceeds previous works, effectively crossing the Sim2Real gap. Experimental validation demonstrates horizontal jumps up to 1.25 m with centimeter accuracy and vertical jumps up to 1.0 m. Additionally, we show that with only minor modifications, the proposed method can be used to learn omnidirectional jumping.

**arXiv ID:** 2510.24584
</details>

<details>
<summary><strong>Robust Point Cloud Reinforcement Learning via PCA-Based Canonicalization</strong> - Michael Bezick, Vittorio Giammarino, Ahmed H. Qureshi - [[pdf]](https://arxiv.org/pdf/2510.20974)</summary>

**Abstract:** Reinforcement Learning (RL) from raw visual input has achieved impressive successes in recent years, yet it remains fragile to out-of-distribution variations such as changes in lighting, color, and viewpoint. Point Cloud Reinforcement Learning (PC-RL) offers a promising alternative by mitigating appearance-based brittleness, but its sensitivity to camera pose mismatches continues to undermine reliability in realistic settings. To address this challenge, we propose PCA Point Cloud (PPC), a canonicalization framework specifically tailored for downstream robotic control. PPC maps point clouds under arbitrary rigid-body transformations to a unique canonical pose, aligning observations to a consistent frame, thereby substantially decreasing viewpoint-induced inconsistencies. In our experiments, we show that PPC improves robustness to unseen camera poses across challenging robotic tasks, providing a principled alternative to domain randomization.

**arXiv ID:** 2510.20974
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
