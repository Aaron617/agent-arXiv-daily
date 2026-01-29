# Agent arXiv Daily

**Last Updated:** 2026-01-29 03:34:52

**Total Papers:** 86

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (9 papers)</h2></summary>

<details>
<summary><strong>Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents</strong> - Yiting Shen, Kun Li, Wei Zhou, Songlin Hu - [[pdf]](https://arxiv.org/pdf/2601.19935)</summary>

**Abstract:** Large Language Model (LLM)-based agents are increasingly deployed for complex, tool-based tasks where long-term memory is critical to driving actions. Existing benchmarks, however, primarily test a angent's ability to passively retrieve isolated facts in response to explicit questions. They fail to evaluate the more crucial capability of actively applying memory to execute tasks. To address this gap, we introduce \textsc{Mem2ActBench}, a benchmark for evaluating whether agents can proactively leverage long-term memory to execute tool-based actions by selecting appropriate tools and grounding their parameters. The benchmark simulates persistent assistant usage, where users mention the same topic across long, interrupted interactions and expect previously established preferences and task states to be implicitly applied. We build the dataset with an automated pipeline that merges heterogeneous sources (ToolACE, BFCL, Oasst1), resolves conflicts via consistency modeling, and synthesizes 2,029 sessions with 12 user--assistant--tool turns on average. From these memory chains, a reverse-generation method produces 400 tool-use tasks, with human evaluation confirming 91.3\% are strongly memory-dependent. Experiments on seven memory frameworks show that current systems remain inadequate at actively utilizing memory for parameter grounding, highlighting the need for more effective approaches to evaluate and improve memory application in task execution.

**arXiv ID:** 2601.19935
</details>

<details>
<summary><strong>Taming Toxic Talk: Using chatbots to intervene with users posting toxic comments</strong> - Jeremy Foote, Deepak Kumar, Bedadyuti Jha, Ryan Funkhouser, Loizos Bitsikokos, Hitesh Goel, Hsuen-Chi Chiu - [[pdf]](https://arxiv.org/pdf/2601.20100)</summary>

**Abstract:** Generative AI chatbots have proven surprisingly effective at persuading people to change their beliefs and attitudes in lab settings. However, the practical implications of these findings are not yet clear. In this work, we explore the impact of rehabilitative conversations with generative AI chatbots on users who share toxic content online. Toxic behaviors -- like insults or threats of violence, are widespread in online communities. Strategies to deal with toxic behavior are typically punitive, such as removing content or banning users. Rehabilitative approaches are rarely attempted, in part due to the emotional and psychological cost of engaging with aggressive users. In collaboration with seven large Reddit communities, we conducted a large-scale field experiment (N=893) to invite people who had recently posted toxic content to participate in conversations with AI chatbots. A qualitative analysis of the conversations shows that many participants engaged in good faith and even expressed remorse or a desire to change. However, we did not observe a significant change in toxic behavior in the following month compared to a control group. We discuss possible explanations for our findings, as well as theoretical and practical implications based on our results.

**arXiv ID:** 2601.20100
</details>

<details>
<summary><strong>Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering</strong> - Xinyu Zhu, Yuzhu Cai, Zexi Liu, Bingyang Zheng, Cheng Wang, Rui Ye, Jiaao Chen, Hanrui Wang, Wei-Chen Wang, Yuzhi Zhang, Linfeng Zhang, Weinan E, Di Jin, Siheng Chen, Yanfeng Wang - [[pdf]](https://arxiv.org/pdf/2601.10402)</summary>

**Abstract:** The advancement of artificial intelligence toward agentic science is currently bottlenecked by the challenge of ultra-long-horizon autonomy, the ability to sustain strategic coherence and iterative correction over experimental cycles spanning days or weeks. While Large Language Models (LLMs) have demonstrated prowess in short-horizon reasoning, they are easily overwhelmed by execution details in the high-dimensional, delayed-feedback environments of real-world research, failing to consolidate sparse feedback into coherent long-term guidance. Here, we present ML-Master 2.0, an autonomous agent that masters ultra-long-horizon machine learning engineering (MLE) which is a representative microcosm of scientific discovery. By reframing context management as a process of cognitive accumulation, our approach introduces Hierarchical Cognitive Caching (HCC), a multi-tiered architecture inspired by computer systems that enables the structural differentiation of experience over time. By dynamically distilling transient execution traces into stable knowledge and cross-task wisdom, HCC allows agents to decouple immediate execution from long-term experimental strategy, effectively overcoming the scaling limits of static context windows. In evaluations on OpenAI's MLE-Bench under 24-hour budgets, ML-Master 2.0 achieves a state-of-the-art medal rate of 56.44%. Our findings demonstrate that ultra-long-horizon autonomy provides a scalable blueprint for AI capable of autonomous exploration beyond human-precedent complexities.

**arXiv ID:** 2601.10402
</details>

<details>
<summary><strong>ShareChat: A Dataset of Chatbot Conversations in the Wild</strong> - Yueru Yan, Tuc Nguyen, Bo Su, Melissa Lieffers, Thai Le - [[pdf]](https://arxiv.org/pdf/2512.17843)</summary>

**Abstract:** While academic research typically treats Large Language Models (LLM) as generic text generators, they are distinct commercial products with unique interfaces and capabilities that fundamentally shape user behavior. Current datasets obscure this reality by collecting text-only data through uniform interfaces that fail to capture authentic chatbot usage. To address this limitation, we present ShareChat, a large-scale corpus of 142,808 conversations (660,293 turns) sourced directly from publicly shared URLs on ChatGPT, Perplexity, Grok, Gemini, and Claude. ShareChat distinguishes itself by preserving native platform affordances, such as citations and thinking traces, across a diverse collection covering 101 languages and the period from April 2023 to October 2025. Furthermore, ShareChat offers substantially longer context windows and greater interaction depth than prior datasets. To illustrate the dataset's breadth, we present three case studies: a completeness analysis of intent satisfaction, a citation study of model grounding, and a temporal analysis of engagement rhythms. This work provides the community with a vital and timely resource for understanding authentic user-LLM chatbot interactions in the wild. The dataset is publicly available via Hugging Face.

**arXiv ID:** 2512.17843
</details>

<details>
<summary><strong>Agentic Fog: A Policy-driven Framework for Distributed Intelligence in Fog Computing</strong> - Saeed Akbar, Muhammad Waqas, Rahmat Ullah - [[pdf]](https://arxiv.org/pdf/2601.20764)</summary>

**Abstract:** Fog and edge computing require adaptive control schemes that can handle partial observability, severe latency requirements, and dynamically changing workloads. Recent research on Agentic AI (AAI) increasingly integrates reasoning systems powered by Large Language Models; however, these tools are not applicable to infrastructure-level systems due to their high computational cost, stochastic nature, and poor formal analyzability. In this paper, a generic model, Agentic Fog (AF), is presented, in which fog nodes are represented as policy-driven autonomous agents that communicate via p2p interactions based on shared memory and localized coordination. The suggested architecture decomposes a system's goals into abstract policy guidance and formalizes decentralized fog coordination as an exact potential game. The framework is guaranteed to converge and remain stable under asynchronous updates, bounded-rational best-response dynamics, and node failures. Simulations demonstrate that the AF system achieves lower average latency and adapts more efficiently to varying demand than greedy heuristics and integer linear programming under dynamic conditions. The sensitivity analysis also demonstrates the capability to perform optimally under different memory and coordination conditions.

**arXiv ID:** 2601.20764
</details>

<details>
<summary><strong>AgentLongBench: A Controllable Long Benchmark For Long-Contexts Agents via Environment Rollouts</strong> - Shicheng Fang, Yuxin Wang, XiaoRan Liu, Jiahao Lu, Chuanyuan Tan, Xinchi Chen, Yining Zheng. Xuanjing Huang, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2601.20730)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) into autonomous agents necessitates the management of extensive, dynamic contexts. Current benchmarks, however, remain largely static, relying on passive retrieval tasks that fail to simulate the complexities of agent-environment interaction, such as non-linear reasoning and iterative feedback. To address this, we introduce \textbf{AgentLongBench}, which evaluates agents through simulated environment rollouts based on Lateral Thinking Puzzles. This framework generates rigorous interaction trajectories across knowledge-intensive and knowledge-free scenarios. Experiments with state-of-the-art models and memory systems (32K to 4M tokens) expose a critical weakness: while adept at static retrieval, agents struggle with the dynamic information synthesis essential for workflows. Our analysis indicates that this degradation is driven by the minimum number of tokens required to resolve a query. This factor explains why the high information density inherent in massive tool responses poses a significantly greater challenge than the memory fragmentation typical of long-turn dialogues.

**arXiv ID:** 2601.20730
</details>

<details>
<summary><strong>Polite But Boring? Trade-offs Between Engagement and Psychological Reactance to Chatbot Feedback Styles</strong> - Samuel Rhys Cox, Joel Wester, Niels van Berkel - [[pdf]](https://arxiv.org/pdf/2601.20683)</summary>

**Abstract:** As conversational agents become increasingly common in behaviour change interventions, understanding optimal feedback delivery mechanisms becomes increasingly important. However, choosing a style that both lessens psychological reactance (perceived threats to freedom) while simultaneously eliciting feelings of surprise and engagement represents a complex design problem. We explored how three different feedback styles: 'Direct', 'Politeness', and 'Verbal Leakage' (slips or disfluencies to reveal a desired behaviour) affect user perceptions and behavioural intentions. Matching expectations from literature, the 'Direct' chatbot led to lower behavioural intentions and higher reactance, while the 'Politeness' chatbot evoked higher behavioural intentions and lower reactance. However, 'Politeness' was also seen as unsurprising and unengaging by participants. In contrast, 'Verbal Leakage' evoked reactance, yet also elicited higher feelings of surprise, engagement, and humour. These findings highlight that effective feedback requires navigating trade-offs between user reactance and engagement, with novel approaches such as 'Verbal Leakage' offering promising alternative design opportunities.

**arXiv ID:** 2601.20683
</details>

<details>
<summary><strong>An Autonomous Agent Framework for Feature-Label Extraction from Device Dialogues and Automatic Multi-Dimensional Device Hosting Planning Based on Large Language Models</strong> - Huichao Men, Yizhen Hu, Yu Gao, Xiaofeng Mou, Yi Xu, Xinhua Xiao - [[pdf]](https://arxiv.org/pdf/2601.20194)</summary>

**Abstract:** With the deep integration of artificial intelligence and smart home technologies, the intelligent transformation of traditional household appliances has become an inevitable trend. This paper presents AirAgent--an LLM-driven autonomous agent framework designed for home air systems. Leveraging a voice-based dialogue interface, AirAgent autonomously and personally manages indoor air quality through comprehensive perception, reasoning, and control. The framework innovatively adopts a two-layer cooperative architecture: Memory-Based Tag Extraction and Reasoning-Driven Planning. First, a dynamic memory tag extraction module continuously updates personalized user profiles. Second, a reasoning-planning model integrates real-time environmental sensor data, user states, and domain-specific prior knowledge (e.g., public health guidelines) to generate context-aware decisions. To support both interpretability and execution, we design a semi-streaming output mechanism that uses special tokens to segment the model's output stream in real time, simultaneously producing human-readable Chain-of-Thought explanations and structured, device-executable control commands. The system handles planning across 25 distinct complex dimensions while satisfying more than 20 customized constraints. As a result, AirAgent endows home air systems with proactive perception, service, and orchestration capabilities, enabling seamless, precise, and personalized air management responsive to dynamic indoor and outdoor conditions. Experimental results demonstrate up to 94.9 percent accuracy and more than 20 percent improvement in user experience metrics compared to competing commercial solutions.

**arXiv ID:** 2601.20194
</details>

<details>
<summary><strong>Bridging Instead of Replacing Online Coding Communities with AI through Community-Enriched Chatbot Designs</strong> - Junling Wang, Lahari Goswami, Gustavo Kreia Umbelino, Kiara Chau, Mrinmaya Sachan, April Yi Wang - [[pdf]](https://arxiv.org/pdf/2601.18697)</summary>

**Abstract:** LLM-based chatbots like ChatGPT have become popular tools for assisting with coding tasks. However, they often produce isolated responses and lack mechanisms for social learning or contextual grounding. In contrast, online coding communities like Kaggle offer socially mediated learning environments that foster critical thinking, engagement, and a sense of belonging. Yet, growing reliance on LLMs risks diminishing participation in these communities and weakening their collaborative value. To address this, we propose Community-Enriched AI, a design paradigm that embeds social learning dynamics into LLM-based chatbots by surfacing user-generated content and social design feature from online coding communities. Using this paradigm, we implemented a RAG-based AI chatbot leveraging resources from Kaggle to validate our design. Across two empirical studies involving 28 and 12 data science learners, respectively, we found that Community-Enriched AI significantly enhances user trust, encourages engagement with community, and effectively supports learners in solving data science tasks. We conclude by discussing design implications for AI assistance systems that bridge -- rather than replace -- online coding communities.

**arXiv ID:** 2601.18697
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>ECG-Agent: On-Device Tool-Calling Agent for ECG Multi-Turn Dialogue</strong> - Hyunseung Chung, Jungwoo Oh, Daeun Kyung, Jiho Kim, Yeonsu Kwon, Min-Gyu Kim, Edward Choi - [[pdf]](https://arxiv.org/pdf/2601.20323)</summary>

**Abstract:** Recent advances in Multimodal Large Language Models have rapidly expanded to electrocardiograms, focusing on classification, report generation, and single-turn QA tasks. However, these models fall short in real-world scenarios, lacking multi-turn conversational ability, on-device efficiency, and precise understanding of ECG measurements such as the PQRST intervals. To address these limitations, we introduce ECG-Agent, the first LLM-based tool-calling agent for multi-turn ECG dialogue. To facilitate its development and evaluation, we also present ECG-Multi-Turn-Dialogue (ECG-MTD) dataset, a collection of realistic user-assistant multi-turn dialogues for diverse ECG lead configurations. We develop ECG-Agents in various sizes, from on-device capable to larger agents. Experimental results show that ECG-Agents outperform baseline ECG-LLMs in response accuracy. Furthermore, on-device agents achieve comparable performance to larger agents in various evaluations that assess response accuracy, tool-calling ability, and hallucinations, demonstrating their viability for real-world applications.

**arXiv ID:** 2601.20323
</details>

<details>
<summary><strong>MemCtrl: Using MLLMs as Active Memory Controllers on Embodied Agents</strong> - Vishnu Sashank Dorbala, Dinesh Manocha - [[pdf]](https://arxiv.org/pdf/2601.20831)</summary>

**Abstract:** Foundation models rely on in-context learning for personalized decision making. The limited size of this context window necessitates memory compression and retrieval systems like RAG. These systems however often treat memory as large offline storage spaces, which is unfavorable for embodied agents that are expected to operate under strict memory and compute constraints, online. In this work, we propose MemCtrl, a novel framework that uses Multimodal Large Language Models (MLLMs) for pruning memory online. MemCtrl augments MLLMs with a trainable memory head \mu that acts as a gate to determine which observations or reflections to retain, update, or discard during exploration. We evaluate with training two types of \mu, 1) via an offline expert, and 2) via online RL, and observe significant improvement in overall embodied task completion ability on \mu-augmented MLLMs. In particular, on augmenting two low performing MLLMs with MemCtrl on multiple subsets of the EmbodiedBench benchmark, we observe that \mu-augmented MLLMs show an improvement of around 16% on average, with over 20% on specific instruction subsets. Finally, we present a qualitative analysis on the memory fragments collected by \mu, noting the superior performance of \mu augmented MLLMs on long and complex instruction types.

**arXiv ID:** 2601.20831
</details>

<details>
<summary><strong>LTS-VoiceAgent: A Listen-Think-Speak Framework for Efficient Streaming Voice Interaction via Semantic Triggering and Incremental Reasoning</strong> - Wenhao Zou, Yuwei Miao, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu - [[pdf]](https://arxiv.org/pdf/2601.19952)</summary>

**Abstract:** Real-time voice agents face a dilemma: end-to-end models often lack deep reasoning, while cascaded pipelines incur high latency by executing ASR, LLM reasoning, and TTS strictly in sequence, unlike human conversation where listeners often start thinking before the speaker finishes. Since cascaded architectures remain the dominant choice for complex tasks, existing cascaded streaming strategies attempt to reduce this latency via mechanical segmentation (e.g., fixed chunks, VAD-based splitting) or speculative generation, but they frequently either break semantic units or waste computation on predictions that must be rolled back. To address these challenges, we propose LTS-VoiceAgent, a Listen-Think-Speak framework that explicitly separates when to think from how to reason incrementally. It features a Dynamic Semantic Trigger to detect meaningful prefixes, and a Dual-Role Stream Orchestrator that coordinates a background Thinker (for state maintenance) and a foreground Speaker (for speculative solving). This parallel design enables "thinking while speaking" without blocking responses. We also introduce a Pause-and-Repair benchmark containing natural disfluencies to stress-test streaming robustness. Experiments across VERA, Spoken-MQA, BigBenchAudio, and our benchmark show that LTS-VoiceAgent achieves a stronger accuracy-latency-efficiency trade-off than serial cascaded baselines and existing streaming strategies.

**arXiv ID:** 2601.19952
</details>

<details>
<summary><strong>MobileBench-OL: A Comprehensive Chinese Benchmark for Evaluating Mobile GUI Agents in Real-World Environment</strong> - Qinzhuo Wu, Zhizhuo Yang, Hanhao Li, Pengzhi Gao, Wei Liu, Jian Luan - [[pdf]](https://arxiv.org/pdf/2601.20335)</summary>

**Abstract:** Recent advances in mobile Graphical User Interface (GUI) agents highlight the growing need for comprehensive evaluation benchmarks. While new online benchmarks offer more realistic testing than offline ones, they tend to focus on the agents' task instruction-following ability while neglecting their reasoning and exploration ability. Moreover, these benchmarks do not consider the random noise in real-world mobile environments. This leads to a gap between benchmarks and real-world environments. To addressing these limitations, we propose MobileBench-OL, an online benchmark with 1080 tasks from 80 Chinese apps. It measures task execution, complex reasoning, and noise robustness of agents by including 5 subsets, which set multiple evaluation dimensions. We also provide an auto-eval framework with a reset mechanism, enabling stable and repeatable real-world benchmarking. Evaluating 12 leading GUI agents on MobileBench-OL shows significant room for improvement to meet real-world requirements. Human evaluation further confirms that MobileBench-OL can reliably measure the performance of leading GUI agents in real environments. Our data and code will be released upon acceptance.

**arXiv ID:** 2601.20335
</details>

<details>
<summary><strong>MetaVLA: Unified Meta Co-training For Efficient Embodied Adaption</strong> - Chen Li, Zhantao Yang, Han Zhang, Fangyi Chen, Chenchen Zhu, Anudeepsekhar Bolimera, Marios Savvides - [[pdf]](https://arxiv.org/pdf/2510.05580)</summary>

**Abstract:** Vision-Language-Action (VLA) models show promise in embodied reasoning, yet remain far from true generalists-they often require task-specific fine-tuning, incur high compute costs, and generalize poorly to unseen tasks. We propose MetaVLA, a unified, backbone-agnostic post-training framework for efficient and scalable alignment. MetaVLA introduces Context-Aware Meta Co-Training, which consolidates diverse target tasks into a single fine-tuning stage while leveraging structurally diverse auxiliary tasks to improve in-domain generalization. Unlike naive multi-task SFT, MetaVLA integrates a lightweight meta-learning mechanism-derived from Attentive Neural Processes-to enable rapid adaptation from diverse contexts with minimal architectural change or inference overhead. On the LIBERO benchmark, MetaVLA with six auxiliary tasks outperforms OpenVLA by up to 8.0% on long-horizon tasks, reduces training steps from 240K to 75K, and cuts GPU time by ~76%. These results show that scalable, low-resource post-training is achievable-paving the way toward general-purpose embodied agents. Code will be available.

**arXiv ID:** 2510.05580
</details>

<details>
<summary><strong>CASCADE: Cumulative Agentic Skill Creation through Autonomous Development and Evolution</strong> - Xu Huang, Junwu Chen, Yuxing Fei, Zhuohan Li, Philippe Schwaller, Gerbrand Ceder - [[pdf]](https://arxiv.org/pdf/2512.23880)</summary>

**Abstract:** Large language model (LLM) agents currently depend on predefined tools or early-stage tool generation, limiting their adaptability and scalability to complex scientific tasks. We introduce CASCADE, a self-evolving agentic framework representing an early instantiation of the transition from "LLM + tool use" to "LLM + skill acquisition". CASCADE enables agents to master complex external tools and codify knowledge through two meta-skills: continuous learning via web search, code extraction, and memory utilization; self-reflection via introspection, knowledge graph exploration, and others. We evaluate CASCADE on SciSkillBench, a benchmark of 116 materials science and chemistry research tasks. CASCADE achieves a 93.3% success rate using GPT-5, compared to 35.4% without evolution mechanisms. We further demonstrate real-world applications in computational analysis, autonomous laboratory experiments, and selective reproduction of published papers. Along with human-agent collaboration and memory consolidation, CASCADE accumulates executable skills that can be shared across agents and scientists, moving toward scalable AI-assisted scientific research.

**arXiv ID:** 2512.23880
</details>

<details>
<summary><strong>Can AI Master Econometrics? Evidence from Econometrics AI Agent on Expert-Level Tasks</strong> - Qiang Chen, Tianyang Han, Jin Li, Ye Luo, Zigan Wang, Yuxiao Wu, Xiaowei Zhang, Tuo Zhou - [[pdf]](https://arxiv.org/pdf/2506.00856)</summary>

**Abstract:** Can AI effectively perform complex econometric analysis traditionally requiring human expertise? This paper evaluates AI agents' capability to master econometrics, focusing on empirical analysis performance. We develop ``MetricsAI'', an Econometrics AI Agent built on the open-source MetaGPT framework. This agent exhibits outstanding performance in: (1) planning econometric tasks strategically, (2) generating and executing code, (3) employing error-based reflection for improved robustness, and (4) allowing iterative refinement through multi-round conversations. We construct two datasets from academic coursework materials and published research papers to evaluate performance against real-world challenges. Comparative testing shows our domain-specialized AI agent significantly outperforms both benchmark large language models (LLMs) and general-purpose AI agents. This work establishes a testbed for exploring AI's impact on social science research and enables cost-effective integration of domain expertise, making advanced econometric methods accessible to users with minimal coding skills. Furthermore, our AI agent enhances research reproducibility and offers promising pedagogical applications for econometrics teaching.

**arXiv ID:** 2506.00856
</details>

<details>
<summary><strong>HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization</strong> - Hongzheng Chen, Yingheng Wang, Yaohui Cai, Hins Hu, Jiajie Li, Shirley Huang, Chenhui Deng, Rongjian Liang, Shufeng Kong, Haoxing Ren, Samitha Samaranayake, Carla P. Gomes, Zhiru Zhang - [[pdf]](https://arxiv.org/pdf/2506.07972)</summary>

**Abstract:** While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains.

**arXiv ID:** 2506.07972
</details>

<details>
<summary><strong>DoubleAgents: Interactive Simulations for Alignment in Agentic AI</strong> - Tao Long, Xuanming Zhang, Sitong Wang, Zhou Yu, Lydia B Chilton - [[pdf]](https://arxiv.org/pdf/2509.12626)</summary>

**Abstract:** Agentic workflows promise efficiency, but adoption hinges on whether people can align systems that act on their behalf with their goals, values, and situational expectations. We present DoubleAgents, an agentic planning tool that embeds transparency and control through user intervention, value-reflecting policies, rich state visualizations, and uncertainty flagging for human coordination tasks. A built-in respondent simulation generates realistic scenarios, allowing users to rehearse and refine policies and calibrate their use of agentic behavior before live deployment. We evaluate DoubleAgents in a two-day lab study (n = 10), three deployment studies, and a technical evaluation. Results show that participants initially hesitated to delegate but used simulation to probe system behavior and adjust policies, gradually increasing delegation as agent actions became better aligned with their intentions and context. Deployment results demonstrate DoubleAgents' real-world relevance and usefulness, showing that simulation helps users effectively manage real-world tasks with higher complexity and uncertainty. We contribute interactive simulation as a practical pathway for users to iteratively align and calibrate agentic systems.

**arXiv ID:** 2509.12626
</details>

<details>
<summary><strong>Me-Agent: A Personalized Mobile Agent with Two-Level User Habit Learning for Enhanced Interaction</strong> - Shuoxin Wang, Chang Liu, Gowen Loo, Lifan Zheng, Kaiwen Wei, Xinyi Zeng, Jingyuan Zhang, Yu Tian - [[pdf]](https://arxiv.org/pdf/2601.20162)</summary>

**Abstract:** Large Language Model (LLM)-based mobile agents have made significant performance advancements. However, these agents often follow explicit user instructions while overlooking personalized needs, leading to significant limitations for real users, particularly without personalized context: (1) inability to interpret ambiguous instructions, (2) lack of learning from user interaction history, and (3) failure to handle personalized instructions. To alleviate the above challenges, we propose Me-Agent, a learnable and memorable personalized mobile agent. Specifically, Me-Agent incorporates a two-level user habit learning approach. At the prompt level, we design a user preference learning strategy enhanced with a Personal Reward Model to improve personalization performance. At the memory level, we design a Hierarchical Preference Memory, which stores users' long-term memory and app-specific memory in different level memory. To validate the personalization capabilities of mobile agents, we introduce User FingerTip, a new benchmark featuring numerous ambiguous instructions for daily life. Extensive experiments on User FingerTip and general benchmarks demonstrate that Me-Agent achieves state-of-the-art performance in personalization while maintaining competitive instruction execution performance.

**arXiv ID:** 2601.20162
</details>

<details>
<summary><strong>Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents</strong> - Qihao Wang, Yue Hu, Mingzhe Lu, Jiayue Wu, Yanbing Liu, Yuanmin Tang - [[pdf]](https://arxiv.org/pdf/2601.20412)</summary>

**Abstract:** The ability of Large Language Models (LLMs) to use external tools unlocks powerful real-world interactions, making rigorous evaluation essential. However, current benchmarks primarily report final accuracy, revealing what models can do but obscuring the cognitive bottlenecks that define their true capability boundaries. To move from simple performance scoring to a diagnostic tool, we introduce a framework grounded in Cognitive Load Theory. Our framework deconstructs task complexity into two quantifiable components: Intrinsic Load, the inherent structural complexity of the solution path, formalized with a novel Tool Interaction Graph; and Extraneous Load, the difficulty arising from ambiguous task presentation. To enable controlled experiments, we construct ToolLoad-Bench, the first benchmark with parametrically adjustable cognitive load. Our evaluation reveals distinct performance cliffs as cognitive load increases, allowing us to precisely map each model's capability boundary. We validate that our framework's predictions are highly calibrated with empirical results, establishing a principled methodology for understanding an agent's limits and a practical foundation for building more efficient systems.

**arXiv ID:** 2601.20412
</details>

<details>
<summary><strong>Efficient Multimodal Planning Agent for Visual Question-Answering</strong> - Zhuo Chen, Xinyu Geng, Xinyu Wang, Yong Jiang, Zhen Zhang, Pengjun Xie, Kewei Tu - [[pdf]](https://arxiv.org/pdf/2601.20676)</summary>

**Abstract:** Visual Question-Answering (VQA) is a challenging multimodal task that requires integrating visual and textual information to generate accurate responses. While multimodal Retrieval-Augmented Generation (mRAG) has shown promise in enhancing VQA systems by providing more evidence on both image and text sides, the default procedure that addresses VQA queries, especially the knowledge-intensive ones, often relies on multi-stage pipelines of mRAG with inherent dependencies. To mitigate the inefficiency limitations while maintaining VQA task performance, this paper proposes a method that trains a multimodal planning agent, dynamically decomposing the mRAG pipeline to solve the VQA task. Our method optimizes the trade-off between efficiency and effectiveness by training the agent to intelligently determine the necessity of each mRAG step. In our experiments, the agent can help reduce redundant computations, cutting search time by over 60\% compared to existing methods and decreasing costly tool calls. Meanwhile, experiments demonstrate that our method outperforms all baselines, including a Deep Research agent and a carefully designed prompt-based method, on average over six various datasets. Code will be released.

**arXiv ID:** 2601.20676
</details>

<details>
<summary><strong>LEGO-Eval: Towards Fine-Grained Evaluation on Synthesizing 3D Embodied Environments with Tool Augmentation</strong> - Gyeom Hwangbo, Hyungjoo Chae, Minseok Kang, Hyeonjong Ju, Soohyun Oh, Jinyoung Yeo - [[pdf]](https://arxiv.org/pdf/2511.03001)</summary>

**Abstract:** Despite recent progress in using Large Language Models (LLMs) for automatically generating 3D scenes, generated scenes often lack realistic spatial layouts and object attributes found in real-world environments. As this problem stems from insufficiently detailed, coarse-grained instructions, advancing 3D scene synthesis guided by more detailed, fine-grained instructions that reflect real-world environments becomes crucial. Without such realistic scenes, training embodied agents in unrealistic environments can lead them to learn priors that diverge significantly from real-world physics and semantics, degrading their performance when deployed. Thus, verifying the alignment between the fine-grained instruction and the generated scene is essential for effective learning. However, current evaluation methods, such as CLIPScore and vision-language models (VLMs), often fail to reliably assess such alignment. This shortcoming arises primarily from their shallow understanding of 3D scenes, which often leads to improperly grounded scene components. To address this, we introduce LEGO-Eval, an evaluation framework equipped with diverse tools designed to explicitly ground scene components, enabling more accurate alignment assessments. We also present LEGO-Bench, a benchmark of detailed instructions that specify complex layouts and attributes of real-world environments. Experiments demonstrate that LEGO-Eval outperforms VLM-as-a-judge by 0.41 F1 score in assessing scene-instruction alignment. Benchmarking with LEGO-Bench reveals significant limitations in current generation methods. Across all evaluated approaches, success rates reached at most 10% in generating scenes that fully align with fine-grained instructions.

**arXiv ID:** 2511.03001
</details>

<details>
<summary><strong>Fusion of Visual-Inertial Odometry with LiDAR Relative Localization for Cooperative Guidance of a Micro-Scale Aerial Vehicle</strong> - Václav Pritzl, Matouš Vrba, Petr Štěpán, Martin Saska - [[pdf]](https://arxiv.org/pdf/2306.17544)</summary>

**Abstract:** A novel relative localization approach for guidance of a micro-scale Unmanned Aerial Vehicle (UAV) by a well-equipped aerial robot fusing Visual-Inertial Odometry (VIO) with Light Detection and Ranging (LiDAR) is proposed in this paper. LiDAR-based localization is accurate and robust to challenging environmental conditions, but 3D LiDARs are relatively heavy and require large UAV platforms, in contrast to lightweight cameras. However, visual-based self-localization methods exhibit lower accuracy and can suffer from significant drift with respect to the global reference frame. To benefit from both sensory modalities, we focus on cooperative navigation in a heterogeneous team of a primary LiDAR-equipped UAV and a secondary micro-scale camera-equipped UAV. We propose a novel cooperative approach combining LiDAR relative localization data with VIO output on board the primary UAV to obtain an accurate pose of the secondary UAV. The pose estimate is used to precisely and reliably guide the secondary UAV along trajectories defined in the primary UAV reference frame. The experimental evaluation has shown the superior accuracy of our method to the raw VIO output, reaching the average 3D Absolute Trajectory Error (ATE) of 0.28 m, and demonstrated its capability to guide the secondary UAV along desired trajectories while mitigating VIO drift. Thus, such a heterogeneous system can explore large areas with LiDAR precision, as well as visit locations inaccessible to the large LiDAR-carrying UAV platforms, as was showcased in a real-world cooperative mapping scenario.

**arXiv ID:** 2306.17544
</details>

<details>
<summary><strong>Listen, Look, Drive: Coupling Audio Instructions for User-aware VLA-based Autonomous Driving</strong> - Ziang Guo, Feng Yang, Xuefeng Zhang, Jiaqi Guo, Kun Zhao, Yixiao Zhou, Peng Lu, Zufeng Zhang, Sifa Zheng - [[pdf]](https://arxiv.org/pdf/2601.12142)</summary>

**Abstract:** Vision Language Action (VLA) models promise an open-vocabulary interface that can translate perceptual ambiguity into semantically grounded driving decisions, yet they still treat language as a static prior fixed at inference time. As a result, the model must infer continuously shifting objectives from pixels alone, yielding delayed or overly conservative maneuvers. We argue that effective VLAs for autonomous driving need an online channel in which users can influence driving with specific intentions. To this end, we present EchoVLA, a user-aware VLA that couples camera streams with in situ audio instructions. We augment the nuScenes dataset with temporally aligned, intent-specific speech commands generated by converting ego-motion descriptions into synthetic audios. Further, we compose emotional speech-trajectory pairs into a multimodal Chain-of-Thought (CoT) for fine-tuning a Multimodal Large Model (MLM) based on Qwen2.5-Omni. Specifically, we synthesize the audio-augmented dataset with different emotion types paired with corresponding driving behaviors, leveraging the emotional cues embedded in tone, pitch, and speech tempo to reflect varying user states, such as urgent or hesitant intentions, thus enabling our EchoVLA to interpret not only the semantic content but also the emotional context of audio commands for more nuanced and emotionally adaptive driving behavior. In open-loop benchmarks, our approach reduces the average L2 error by $59.4\%$ and the collision rate by $74.4\%$ compared to the baseline of vision-only perception. More experiments on nuScenes dataset validate that EchoVLA not only steers the trajectory through audio instructions, but also modulates driving behavior in response to the emotions detected in the user's speech.

**arXiv ID:** 2601.12142
</details>

<details>
<summary><strong>Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations</strong> - Preethi Seshadri, Samuel Cahyawijaya, Ayomide Odumakinde, Sameer Singh, Seraphina Goldfarb-Tarrant - [[pdf]](https://arxiv.org/pdf/2601.17087)</summary>

**Abstract:** Agentic benchmarks increasingly rely on LLM-simulated users to scalably evaluate agent performance, yet the robustness, validity, and fairness of this approach remain unexamined. Through a user study with participants across the United States, India, Kenya, and Nigeria, we investigate whether LLM-simulated users serve as reliable proxies for real human users in evaluating agents on {\tau}-Bench retail tasks. We find that user simulation lacks robustness, with agent success rates varying up to 9 percentage points across different user LLMs. Furthermore, evaluations using simulated users exhibit systematic miscalibration, underestimating agent performance on challenging tasks and overestimating it on moderately difficult ones. African American Vernacular English (AAVE) speakers experience consistently worse success rates and calibration errors than Standard American English (SAE) speakers, with disparities compounding significantly with age. We also find simulated users to be a differentially effective proxy for different populations, performing worst for AAVE and Indian English speakers. Additionally, simulated users introduce conversational artifacts and surface different failure patterns than human users. These findings demonstrate that current evaluation practices risk misrepresenting agent capabilities across diverse user populations and may obscure real-world deployment challenges.

**arXiv ID:** 2601.17087
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Towards Intelligent Urban Park Development Monitoring: LLM Agents for Multi-Modal Information Fusion and Analysis</strong> - Zixuan Xiao, Chunguang Hu, Jun Ma - [[pdf]](https://arxiv.org/pdf/2601.20206)</summary>

**Abstract:** As an important part of urbanization, the development monitoring of newly constructed parks is of great significance for evaluating the effect of urban planning and optimizing resource allocation. However, traditional change detection methods based on remote sensing imagery have obvious limitations in high-level and intelligent analysis, and thus are difficult to meet the requirements of current urban planning and management. In face of the growing demand for complex multi-modal data analysis in urban park development monitoring, these methods often fail to provide flexible analysis capabilities for diverse application scenarios. This study proposes a multi-modal LLM agent framework, which aims to make full use of the semantic understanding and reasoning capabilities of LLM to meet the challenges in urban park development monitoring. In this framework, a general horizontal and vertical data alignment mechanism is designed to ensure the consistency and effective tracking of multi-modal data. At the same time, a specific toolkit is constructed to alleviate the hallucination issues of LLM due to the lack of domain-specific knowledge. Compared to vanilla GPT-4o and other agents, our approach enables robust multi-modal information fusion and analysis, offering reliable and scalable solutions tailored to the diverse and evolving demands of urban park development monitoring.

**arXiv ID:** 2601.20206
</details>

<details>
<summary><strong>Demonstration-Free Robotic Control via LLM Agents</strong> - Brian Y. Tsui, Alan Y. Fang, Tiffany J. Hwu - [[pdf]](https://arxiv.org/pdf/2601.20334)</summary>

**Abstract:** Robotic manipulation has increasingly adopted vision-language-action (VLA) models, which achieve strong performance but typically require task-specific demonstrations and fine-tuning, and often generalize poorly under domain shift. We investigate whether general-purpose large language model (LLM) agent frameworks, originally developed for software engineering, can serve as an alternative control paradigm for embodied manipulation. We introduce FAEA (Frontier Agent as Embodied Agent), which applies an LLM agent framework directly to embodied manipulation without modification. Using the same iterative reasoning that enables software agents to debug code, FAEA enables embodied agents to reason through manipulation strategies. We evaluate an unmodified frontier agent, Claude Agent SDK, across the LIBERO, ManiSkill3, and MetaWorld benchmarks. With privileged environment state access, FAEA achieves success rates of 84.9%, 85.7%, and 96%, respectively. This level of task success approaches that of VLA models trained with less than 100 demonstrations per task, without requiring demonstrations or fine-tuning. With one round of human feedback as an optional optimization, performance increases to 88.2% on LIBERO. This demonstration-free capability has immediate practical value: FAEA can autonomously explore novel scenarios in simulation and generate successful trajectories for training data augmentation in embodied learning. Our results indicate that general-purpose agents are sufficient for a class of manipulation tasks dominated by deliberative, task-level planning. This opens a path for robotics systems to leverage actively maintained agent infrastructure and benefit directly from ongoing advances in frontier models. Code is available at this https URL

**arXiv ID:** 2601.20334
</details>

<details>
<summary><strong>LLM-AutoDP: Automatic Data Processing via LLM Agents for Model Fine-tuning</strong> - Wei Huang, Anda Cheng, Yinggui Wang, Lei Wang, Tao Wei - [[pdf]](https://arxiv.org/pdf/2601.20375)</summary>

**Abstract:** Large Language Models (LLMs) can be fine-tuned on domain-specific data to enhance their performance in specialized fields. However, such data often contains numerous low-quality samples, necessitating effective data processing (DP). In practice, DP strategies are typically developed through iterative manual analysis and trial-and-error adjustment. These processes inevitably incur high labor costs and may lead to privacy issues in high-privacy domains like healthcare due to direct human access to sensitive data. Thus, achieving automated data processing without exposing the raw data has become a critical challenge. To address this challenge, we propose LLM-AutoDP, a novel framework that leverages LLMs as agents to automatically generate and optimize data processing strategies. Our method generates multiple candidate strategies and iteratively refines them using feedback signals and comparative evaluations. This iterative in-context learning mechanism enables the agent to converge toward high-quality processing pipelines without requiring direct human intervention or access to the underlying data. To further accelerate strategy search, we introduce three key techniques: Distribution Preserving Sampling, which reduces data volume while maintaining distributional integrity; Processing Target Selection, which uses a binary classifier to identify low-quality samples for focused processing; Cache-and-Reuse Mechanism}, which minimizes redundant computations by reusing prior processing results. Results show that models trained on data processed by our framework achieve over 80% win rates against models trained on unprocessed data. Compared to AutoML baselines based on LLM agents, LLM-AutoDP achieves approximately a 65% win rate. Moreover, our acceleration techniques reduce the total searching time by up to 10 times, demonstrating both effectiveness and efficiency.

**arXiv ID:** 2601.20375
</details>

<details>
<summary><strong>Agent Benchmarks Fail Public Sector Requirements</strong> - Jonathan Rystrøm, Chris Schmitz, Karolina Korgul, Jan Batzner, Chris Russell - [[pdf]](https://arxiv.org/pdf/2601.20617)</summary>

**Abstract:** Deploying Large Language Model-based agents (LLM agents) in the public sector requires assuring that they meet the stringent legal, procedural, and structural requirements of public-sector institutions. Practitioners and researchers often turn to benchmarks for such assessments. However, it remains unclear what criteria benchmarks must meet to ensure they adequately reflect public-sector requirements, or how many existing benchmarks do so. In this paper, we first define such criteria based on a first-principles survey of public administration literature: benchmarks must be \emph{process-based}, \emph{realistic}, \emph{public-sector-specific} and report \emph{metrics} that reflect the unique requirements of the public sector. We analyse more than 1,300 benchmark papers for these criteria using an expert-validated LLM-assisted pipeline. Our results show that no single benchmark meets all of the criteria. Our findings provide a call to action for both researchers to develop public sector-relevant benchmarks and for public-sector officials to apply these criteria when evaluating their own agentic use cases.

**arXiv ID:** 2601.20617
</details>

<details>
<summary><strong>SimpleMem: Efficient Lifelong Memory for LLM Agents</strong> - Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2601.02553)</summary>

**Abstract:** To support long-term interaction in complex environments, LLM agents require memory systems that manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) Semantic Structured Compression, which distills unstructured interactions into compact, multi-view indexed memory units; (2) Online Semantic Synthesis, an intra-session process that instantly integrates related context into unified abstract representations to eliminate redundancy; and (3) Intent-Aware Retrieval Planning, which infers search intent to dynamically determine retrieval scope and construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of 26.4% while reducing inference-time token consumption by up to 30-fold, demonstrating a superior balance between performance and efficiency. Code is available at this https URL.

**arXiv ID:** 2601.02553
</details>

<details>
<summary><strong>Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents</strong> - Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, An Zhang - [[pdf]](https://arxiv.org/pdf/2509.23040)</summary>

**Abstract:** Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens. Existing works equip large language models with a memory buffer that is dynamically updated via a linear document scan, also known as the "memorize while reading" methods. While this approach scales efficiently, it suffers from pruning of latent evidence, information loss through overwriting, and sparse reinforcement learning signals. To tackle these challenges, we present ReMemR1, which integrates the mechanism of memory retrieval into the memory update process, enabling the agent to selectively callback historical memories for non-linear reasoning. To further strengthen training, we propose a multi-level reward design, which combines final-answer rewards with dense, step-level signals that guide effective memory use. Together, these contributions mitigate information degradation, improve supervision, and support complex multi-hop reasoning. Extensive experiments demonstrate that ReMemR1 significantly outperforms state-of-the-art baselines on long-context question answering while incurring negligible computational overhead, validating its ability to trade marginal cost for robust long-context reasoning.

**arXiv ID:** 2509.23040
</details>

<details>
<summary><strong>AgentIF-OneDay: A Task-level Instruction-Following Benchmark for General AI Agents in Daily Scenarios</strong> - Kaiyuan Chen, Qimin Wu, Taiyu Hou, Tianhao Tang, Xueyu Hu, Yuchen Hou, Bikun Li, Chengming Qian, Guoyin Wang, Haolin Chen, Haotong Tian, Haoye Zhang, Haoyu Bian, Hongbing Pan, Hongkang Zhang, Hongyi Zhou, Jiaqi Cai, Jiewu Rao, Jiyuan Ren, Keduan Huang, Lucia Zhu Huang, Mingyu Yuan, Naixu Guo, Qicheng Tang, Qinyan Zhang, Shuai Chen, Siheng Chen, Ting Ting Li, Xiaoxing Guo, Yaocheng Zuo, Yaoqi Guo, Yinan Wang, Yinzhou Yu, Yize Wang, Yuan Jiang, Yuan Tian, Yuanshuo Zhang, Yuxuan Liu, Yvette Yan Zeng, Zenyu Shan, Zihan Yin, Xiaobo Hu, Yang Liu, Yixin Ren, Yuan Gong - [[pdf]](https://arxiv.org/pdf/2601.20613)</summary>

**Abstract:** The capacity of AI agents to effectively handle tasks of increasing duration and complexity continues to grow, demonstrating exceptional performance in coding, deep research, and complex problem-solving evaluations. However, in daily scenarios, the perception of these advanced AI capabilities among general users remains limited. We argue that current evaluations prioritize increasing task difficulty without sufficiently addressing the diversity of agentic tasks necessary to cover the daily work, life, and learning activities of a broad demographic. To address this, we propose AgentIF-OneDay, aimed at determining whether general users can utilize natural language instructions and AI agents to complete a diverse array of daily tasks. These tasks require not only solving problems through dialogue but also understanding various attachment types and delivering tangible file-based results. The benchmark is structured around three user-centric categories: Open Workflow Execution, which assesses adherence to explicit and complex workflows; Latent Instruction, which requires agents to infer implicit instructions from attachments; and Iterative Refinement, which involves modifying or expanding upon ongoing work. We employ instance-level rubrics and a refined evaluation pipeline that aligns LLM-based verification with human judgment, achieving an 80.1% agreement rate using Gemini-3-Pro. AgentIF-OneDay comprises 104 tasks covering 767 scoring points. We benchmarked four leading general AI agents and found that agent products built based on APIs and ChatGPT agents based on agent RL remain in the first tier simultaneously. Leading LLM APIs and open-source models have internalized agentic capabilities, enabling AI application teams to develop cutting-edge Agent products.

**arXiv ID:** 2601.20613
</details>

<details>
<summary><strong>PEARL: Self-Evolving Assistant for Time Management with Reinforcement Learning</strong> - Bingxuan Li, Jeonghwan Kim, Cheng Qian, Xiusi Chen, Eitan Anzenberg, Niran Kundapur, Heng Ji - [[pdf]](https://arxiv.org/pdf/2601.11957)</summary>

**Abstract:** Overlapping calendar invitations force busy professionals to repeatedly decide which meetings to attend, reschedule, or decline. We refer to this preference-driven decision process as calendar conflict resolution. Automating this decision process is crucial yet challenging. Scheduling logistics can drain hours, and human delegation often fails at scale, which motivates us to ask: Can we trust large language models (LLMs) or language agents to manage time? To enable a systematic study of this question, we introduce CalConflictBench, a benchmark for long-horizon calendar conflict resolution. In CalConflictBench, conflicts are presented to agents round-by-round over a calendar year, requiring them to infer and adapt to user preferences progressively. Our experiments show that current LLM agents perform poorly with high error rates, e.g., Qwen-3-30B-Think has an average error rate of 35%. To address this gap, we propose PEARL, a reinforcement-learning framework that (i) augments the language agent with an external preference memory that stores and updates inferred strategies (e.g., attendee priorities, topic importance, time/location preferences), and (ii) optimizes the agent with round-wise rewards that directly supervise decision correctness, ranking quality, and memory usage across rounds. Experiments on CalConflictBench show that PEARL achieves an error reduction rate of 0.76 and a 55% improvement in average error rate compared to the strongest baseline.

**arXiv ID:** 2601.11957
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Insight Agents: An LLM-Based Multi-Agent System for Data Insights</strong> - Jincheng Bai, Zhenyu Zhang, Jennifer Zhang, Zhihuai Zhu - [[pdf]](https://arxiv.org/pdf/2601.20048)</summary>

**Abstract:** Today, E-commerce sellers face several key challenges, including difficulties in discovering and effectively utilizing available programs and tools, and struggling to understand and utilize rich data from various tools. We therefore aim to develop Insight Agents (IA), a conversational multi-agent Data Insight system, to provide E-commerce sellers with personalized data and business insights through automated information retrieval. Our hypothesis is that IA will serve as a force multiplier for sellers, thereby driving incremental seller adoption by reducing the effort required and increase speed at which sellers make good business decisions. In this paper, we introduce this novel LLM-backed end-to-end agentic system built on a plan-and-execute paradigm and designed for comprehensive coverage, high accuracy, and low latency. It features a hierarchical multi-agent structure, consisting of manager agent and two worker agents: data presentation and insight generation, for efficient information retrieval and problem-solving. We design a simple yet effective ML solution for manager agent that combines Out-of-Domain (OOD) detection using a lightweight encoder-decoder model and agent routing through a BERT-based classifier, optimizing both accuracy and latency. Within the two worker agents, a strategic planning is designed for API-based data model that breaks down queries into granular components to generate more accurate responses, and domain knowledge is dynamically injected to to enhance the insight generator. IA has been launched for Amazon sellers in US, which has achieved high accuracy of 90% based on human evaluation, with latency of P90 below 15s.

**arXiv ID:** 2601.20048
</details>

<details>
<summary><strong>AMA: Adaptive Memory via Multi-Agent Collaboration</strong> - Weiquan Huang, Zixuan Wang, Hehai Lin, Sudong Wang, Bo Xu, Qian Li, Beier Zhu, Linyi Yang, Chengwei Qin - [[pdf]](https://arxiv.org/pdf/2601.20352)</summary>

**Abstract:** The rapid evolution of Large Language Model (LLM) agents has necessitated robust memory systems to support cohesive long-term interaction and complex reasoning. Benefiting from the strong capabilities of LLMs, recent research focus has shifted from simple context extension to the development of dedicated agentic memory systems. However, existing approaches typically rely on rigid retrieval granularity, accumulation-heavy maintenance strategies, and coarse-grained update mechanisms. These design choices create a persistent mismatch between stored information and task-specific reasoning demands, while leading to the unchecked accumulation of logical inconsistencies over time. To address these challenges, we propose Adaptive Memory via Multi-Agent Collaboration (AMA), a novel framework that leverages coordinated agents to manage memory across multiple granularities. AMA employs a hierarchical memory design that dynamically aligns retrieval granularity with task complexity. Specifically, the Constructor and Retriever jointly enable multi-granularity memory construction and adaptive query routing. The Judge verifies the relevance and consistency of retrieved content, triggering iterative retrieval when evidence is insufficient or invoking the Refresher upon detecting logical conflicts. The Refresher then enforces memory consistency by performing targeted updates or removing outdated entries. Extensive experiments on challenging long-context benchmarks show that AMA significantly outperforms state-of-the-art baselines while reducing token consumption by approximately 80% compared to full-context methods, demonstrating its effectiveness in maintaining retrieval precision and long-term memory consistency.

**arXiv ID:** 2601.20352
</details>

<details>
<summary><strong>Demystifying Multi-Agent Debate: The Role of Confidence and Diversity</strong> - Xiaochen Zhu, Caiqi Zhang, Yizhou Chi, Tom Stafford, Nigel Collier, Andreas Vlachos - [[pdf]](https://arxiv.org/pdf/2601.19921)</summary>

**Abstract:** Multi-agent debate (MAD) is widely used to improve large language model (LLM) performance through test-time scaling, yet recent work shows that vanilla MAD often underperforms simple majority vote despite higher computational cost. Studies show that, under homogeneous agents and uniform belief updates, debate preserves expected correctness and therefore cannot reliably improve outcomes. Drawing on findings from human deliberation and collective decision-making, we identify two key mechanisms missing from vanilla MAD: (i) diversity of initial viewpoints and (ii) explicit, calibrated confidence communication. We propose two lightweight interventions. First, a diversity-aware initialisation that selects a more diverse pool of candidate answers, increasing the likelihood that a correct hypothesis is present at the start of debate. Second, a confidence-modulated debate protocol in which agents express calibrated confidence and condition their updates on others' confidence. We show theoretically that diversity-aware initialisation improves the prior probability of MAD success without changing the underlying update dynamics, while confidence-modulated updates enable debate to systematically drift to the correct hypothesis. Empirically, across six reasoning-oriented QA benchmarks, our methods consistently outperform vanilla MAD and majority vote. Our results connect human deliberation with LLM-based debate and demonstrate that simple, principled modifications can substantially enhance debate effectiveness.

**arXiv ID:** 2601.19921
</details>

<details>
<summary><strong>Multimodal Multi-Agent Ransomware Analysis Using AutoGen</strong> - Asifullah Khan, Aimen Wadood, Mubashar Iqbal, Umme Zahoora - [[pdf]](https://arxiv.org/pdf/2601.20346)</summary>

**Abstract:** Ransomware has become one of the most serious cybersecurity threats causing major financial losses and operational disruptions this http URL detection methods such as static analysis, heuristic scanning and behavioral analysis often fall short when used alone. To address these limitations, this paper presents multimodal multi agent ransomware analysis framework designed for ransomware classification. Proposed multimodal multiagent architecture combines information from static, dynamic and network sources. Each data type is handled by specialized agent that uses auto encoder based feature extraction. These representations are then integrated through a fusion agent. After that fused representation are used by transformer based classifier. It identifies the specific ransomware family. The agents interact through an interagent feedback mechanism that iteratively refines feature representations by suppressing low confidence information. The framework was evaluated on large scale datasets containing thousands of ransomware and benign samples. Multiple experiments were conducted on ransomware dataset. It outperforms single modality and nonadaptive fusion baseline achieving improvement of up to 0.936 in Macro-F1 for family classification and reducing calibration error. Over 100 epochs, the agentic feedback loop displays a stable monotonic convergence leading to over +0.75 absolute improvement in terms of agent quality and a final composite score of around 0.88 without fine tuning of the language models. Zeroday ransomware detection remains family dependent on polymorphism and modality disruptions. Confidence aware abstention enables reliable real world deployment by favoring conservativeand trustworthy decisions over forced classification. The findings indicate that proposed approach provides a practical andeffective path toward improving real world ransomware defense systems.

**arXiv ID:** 2601.20346
</details>

<details>
<summary><strong>Interpreting Emergent Extreme Events in Multi-Agent Systems</strong> - Ling Tang, Jilin Mei, Dongrui Liu, Chen Qian, Dawei Cheng, Jing Shao, Xia Hu - [[pdf]](https://arxiv.org/pdf/2601.20538)</summary>

**Abstract:** Large language model-powered multi-agent systems have emerged as powerful tools for simulating complex human-like systems. The interactions within these systems often lead to extreme events whose origins remain obscured by the black box of emergence. Interpreting these events is critical for system safety. This paper proposes the first framework for explaining emergent extreme events in multi-agent systems, aiming to answer three fundamental questions: When does the event originate? Who drives it? And what behaviors contribute to it? Specifically, we adapt the Shapley value to faithfully attribute the occurrence of extreme events to each action taken by agents at different time steps, i.e., assigning an attribution score to the action to measure its influence on the event. We then aggregate the attribution scores along the dimensions of time, agent, and behavior to quantify the risk contribution of each dimension. Finally, we design a set of metrics based on these contribution scores to characterize the features of extreme events. Experiments across diverse multi-agent system scenarios (economic, financial, and social) demonstrate the effectiveness of our framework and provide general insights into the emergence of extreme phenomena.

**arXiv ID:** 2601.20538
</details>

<details>
<summary><strong>LLM Multi-Agent Systems: Challenges and Open Problems</strong> - Shanshan Han, Qifan Zhang, Weizhao Jin, Zhaozhuo Xu - [[pdf]](https://arxiv.org/pdf/2402.03578)</summary>

**Abstract:** This paper explores multi-agent systems and identify challenges that remain inadequately addressed. By leveraging the diverse capabilities and roles of individual agents, multi-agent systems can tackle complex tasks through agent collaboration. We discuss optimizing task allocation, fostering robust reasoning through iterative debates, managing complex and layered context information, and enhancing memory management to support the intricate interactions within multi-agent systems. We also explore potential applications of multi-agent systems in blockchain systems to shed light on their future development and application in real-world distributed systems.

**arXiv ID:** 2402.03578
</details>

<details>
<summary><strong>OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval</strong> - Yu Liu, Yanbing Liu, Fangfang Yuan, Cong Cao, Youbang Sun, Kun Peng, WeiZhuo Chen, Jianjun Li, Zhiyuan Ma - [[pdf]](https://arxiv.org/pdf/2508.16438)</summary>

**Abstract:** Recent advances in large language models (LLMs) and dense retrievers have driven significant progress in retrieval-augmented generation (RAG). However, existing approaches face significant challenges in complex reasoning-oriented multi-hop retrieval tasks: 1) Ineffective reasoning-oriented planning: Prior methods struggle to generate robust multi-step plans for complex queries, as rule-based decomposers perform poorly on out-of-template questions. 2) Suboptimal reasoning-driven retrieval: Related methods employ limited query reformulation, leading to iterative retrieval loops that often fail to locate golden documents. 3) Insufficient reasoning-guided filtering: Prevailing methods lack the fine-grained reasoning to effectively filter salient information from noisy results, hindering utilization of retrieved knowledge. Fundamentally, these limitations all stem from the weak coupling between retrieval and reasoning in current RAG architectures. We introduce the Orchestrated Planner-Executor Reasoning Architecture (OPERA), a novel reasoning-driven retrieval framework. OPERA's Goal Planning Module (GPM) decomposes questions into sub-goals, which are executed by a Reason-Execute Module (REM) with specialized components for precise reasoning and effective retrieval. To train OPERA, we propose Multi-Agents Progressive Group Relative Policy Optimization (MAPGRPO), a novel variant of GRPO. Experiments on complex multi-hop benchmarks show OPERA's superior performance, validating both the MAPGRPO method and OPERA's design.

**arXiv ID:** 2508.16438
</details>

<details>
<summary><strong>VLM-CAD: VLM-Optimized Collaborative Agent Design Workflow for Analog Circuit Sizing</strong> - Guanyuan Pan, Shuai Wang, Yugui Lin, Tiansheng Zhou, Pietro Liò, Zhenxin Zhao, Yaqi Wang - [[pdf]](https://arxiv.org/pdf/2601.07315)</summary>

**Abstract:** Analog mixed-signal circuit sizing involves complex trade-offs within high-dimensional design spaces. Existing automatic analog circuit sizing approaches rely solely on netlists, ignoring the circuit schematic, which hinders the cognitive link between the schematic and its performance. Furthermore, the black-box nature of machine learning methods and hallucination risks in large language models fail to provide the necessary ground-truth explainability required for industrial sign-off. To address these challenges, we propose a Vision Language Model-optimized collaborative agent design workflow (VLM-CAD), which analyzes circuits, optimizes DC operating points, performs inference-based sizing, and executes external sizing optimization. We integrate Image2Net to annotate circuit schematics and generate a structured JSON description for precise interpretation by Vision Language Models. Furthermore, we propose an Explainable Trust Region Bayesian Optimization method (ExTuRBO) that employs collaborative warm-start from agent-generated seeds and offers dual-granularity sensitivity analysis for external sizing optimization, supporting a comprehensive final design report. Experiment results on amplifier sizing tasks using 180nm, 90nm, and 45nm Predictive Technology Models demonstrate that VLM-CAD effectively balances power and performance while maintaining physics-based explainability. VLM-CAD meets all specification requirements while maintaining low power consumption in optimizing an amplifier with a complementary input and a class-AB output stage, with a total runtime under 66 minutes across all experiments on two amplifiers.

**arXiv ID:** 2601.07315
</details>

<details>
<summary><strong>Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow</strong> - Xinlei Yu, Chengming Xu, Guibin Zhang, Yongbo He, Zhangquan Chen, Zhucun Xue, Jiangning Zhang, Yue Liao, Xiaobin Hu, Yu-Gang Jiang, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2509.21789)</summary>

**Abstract:** Multi-Agent System (MAS) powered by Visual Language Models (VLMs) enables challenging tasks but suffers from a novel failure term, multi-agent visual hallucination snowballing, where hallucinations are seeded in a single agent and amplified by following ones due to the over-reliance on textual flow to relay visual information. Through turn-, layer-, and token-wise attention analyses, we provide detailed insights into the essence of hallucination snowballing regarding the reduction of visual attention allocation. It leads us to identify a subset of vision tokens with a unimodal attention peak in middle layers that best preserve visual evidence but gradually diminish in deeper agent turns, resulting in the visual hallucination snowballing in MAS. Thus, we propose ViF, a lightweight, plug-and-play mitigation paradigm that relays inter-agent messages with Visual Flow powered by the selected visual relay tokens and applies attention reallocation to amplify this pattern. The experiment results demonstrate that our method markedly reduces hallucination snowballing, consistently improving the performance across eight benchmarks based on four common MAS structures and ten base models. The source code is publicly available at: this https URL.

**arXiv ID:** 2509.21789
</details>

<details>
<summary><strong>Hierarchical Multi-Agent DRL Based Dynamic Cluster Reconfiguration for UAV Mobility Management</strong> - Irshad A. Meer, Karl-Ludwig Besser, Mustafa Ozger, Dominic Schupke, H. Vincent Poor, Cicek Cavdar - [[pdf]](https://arxiv.org/pdf/2412.16167)</summary>

**Abstract:** Multi-connectivity involves dynamic cluster formation among distributed access points (APs) and coordinated resource allocation from these APs, highlighting the need for efficient mobility management strategies for users with multi-connectivity. In this paper, we propose a novel mobility management scheme for unmanned aerial vehicles (UAVs) that uses dynamic cluster reconfiguration with energy-efficient power allocation in a wireless interference network. Our objective encompasses meeting stringent reliability demands, minimizing joint power consumption, and reducing the frequency of cluster reconfiguration. To achieve these objectives, we propose a hierarchical multi-agent deep reinforcement learning (H-MADRL) framework, specifically tailored for dynamic clustering and power allocation. The edge cloud connected with a set of APs through low latency optical back-haul links hosts the high-level agent responsible for the optimal clustering policy, while low-level agents reside in the APs and are responsible for the power allocation policy. To further improve the learning efficiency, we propose a novel action-observation transition-driven learning algorithm that allows the low-level agents to use the action space from the high-level agent as part of the local observation space. This allows the lower-level agents to share partial information about the clustering policy and allocate the power more efficiently. The simulation results demonstrate that our proposed distributed algorithm achieves comparable performance to the centralized algorithm. Additionally, it offers better scalability, as the decision time for clustering and power allocation increases by only 10% when doubling the number of APs, compared to a 90% increase observed with the centralized approach.

**arXiv ID:** 2412.16167
</details>

<details>
<summary><strong>BMAM: Brain-inspired Multi-Agent Memory Framework</strong> - Yang Li, Jiaxiang Liu, Yusong Wang, Yujie Wu, Mingkun Xu - [[pdf]](https://arxiv.org/pdf/2601.20465)</summary>

**Abstract:** Language-model-based agents operating over extended interaction horizons face persistent challenges in preserving temporally grounded information and maintaining behavioral consistency across sessions, a failure mode we term soul erosion. We present BMAM (Brain-inspired Multi-Agent Memory), a general-purpose memory architecture that models agent memory as a set of functionally specialized subsystems rather than a single unstructured store. Inspired by cognitive memory systems, BMAM decomposes memory into episodic, semantic, salience-aware, and control-oriented components that operate at complementary time scales. To support long-horizon reasoning, BMAM organizes episodic memories along explicit timelines and retrieves evidence by fusing multiple complementary signals. Experiments on the LoCoMo benchmark show that BMAM achieves 78.45 percent accuracy under the standard long-horizon evaluation setting, and ablation analyses confirm that the hippocampus-inspired episodic memory subsystem plays a critical role in temporal reasoning.

**arXiv ID:** 2601.20465
</details>

<details>
<summary><strong>Cochain: Balancing Insufficient and Excessive Collaboration in LLM Agent Workflows</strong> - Jiaxing Zhao, Hongbin Xie, Yuzhen Lei, Xuan Song, Zhuoran Shi, Lianxin Li, Shuangxue Liu, Linguo Xie, Haoran Zhang - [[pdf]](https://arxiv.org/pdf/2505.10936)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating the collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves the business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.

**arXiv ID:** 2505.10936
</details>

<details>
<summary><strong>Unsupervised Anomaly Detection in Multi-Agent Trajectory Prediction via Transformer-Based Models</strong> - Qing Lyu, Zhe Fu, Alexandre Bayen - [[pdf]](https://arxiv.org/pdf/2601.20367)</summary>

**Abstract:** Identifying safety-critical scenarios is essential for autonomous driving, but the rarity of such events makes supervised labeling impractical. Traditional rule-based metrics like Time-to-Collision are too simplistic to capture complex interaction risks, and existing methods lack a systematic way to verify whether statistical anomalies truly reflect physical danger. To address this gap, we propose an unsupervised anomaly detection framework based on a multi-agent Transformer that models normal driving and measures deviations through prediction residuals. A dual evaluation scheme has been proposed to assess both detection stability and physical alignment: Stability is measured using standard ranking metrics in which Kendall Rank Correlation Coefficient captures rank agreement and Jaccard index captures the consistency of the top-K selected items; Physical alignment is assessed through correlations with established Surrogate Safety Measures (SSM). Experiments on the NGSIM dataset demonstrate our framework's effectiveness: We show that the maximum residual aggregator achieves the highest physical alignment while maintaining stability. Furthermore, our framework identifies 388 unique anomalies missed by Time-to-Collision and statistical baselines, capturing subtle multi-agent risks like reactive braking under lateral drift. The detected anomalies are further clustered into four interpretable risk types, offering actionable insights for simulation and testing.

**arXiv ID:** 2601.20367
</details>

<details>
<summary><strong>Judgelight: Trajectory-Level Post-Optimization for Multi-Agent Path Finding via Closed-Subwalk Collapsing</strong> - Yimin Tang, Sven Koenig, Erdem Bıyık - [[pdf]](https://arxiv.org/pdf/2601.19388)</summary>

**Abstract:** Multi-Agent Path Finding (MAPF) is an NP-hard problem with applications in warehouse automation and multi-robot coordination. Learning-based MAPF solvers offer fast and scalable planning but often produce feasible trajectories that contain unnecessary or oscillatory movements. We propose Judgelight, a post-optimization layer that improves trajectory quality after a MAPF solver generates a feasible schedule. Judgelight collapses closed subwalks in agents' trajectories to remove redundant movements while preserving all feasibility constraints. We formalize this process as MAPF-Collapse, prove that it is NP-hard, and present an exact optimization approach by formulating it as integer linear programming (ILP) problem. Experimental results show Judgelight consistently reduces solution cost by around 20%, particularly for learning-based solvers, producing trajectories that are better suited for real-world deployment.

**arXiv ID:** 2601.19388
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control</strong> - Amirmohammad Farzaneh, Salvatore D'Oro, Osvaldo Simeone - [[pdf]](https://arxiv.org/pdf/2601.20090)</summary>

**Abstract:** Large language model (LLM)-powered agents can translate high-level user intents into plans and actions in an environment. Yet after observing an outcome, users may wonder: What if I had phrased my intent differently? We introduce a framework that enables such counterfactual reasoning in agentic LLM-driven control scenarios, while providing formal reliability guarantees. Our approach models the closed-loop interaction between a user, an LLM-based agent, and an environment as a structural causal model (SCM), and leverages test-time scaling to generate multiple candidate counterfactual outcomes via probabilistic abduction. Through an offline calibration phase, the proposed conformal counterfactual generation (CCG) yields sets of counterfactual outcomes that are guaranteed to contain the true counterfactual outcome with high probability. We showcase the performance of CCG on a wireless network control use case, demonstrating significant advantages compared to naive re-execution baselines.

**arXiv ID:** 2601.20090
</details>

<details>
<summary><strong>On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents</strong> - Jai Lal Lulla, Seyedmoein Mohsenimofidi, Matthias Galster, Jie M. Zhang, Sebastian Baltes, Christoph Treude - [[pdf]](https://arxiv.org/pdf/2601.20404)</summary>

**Abstract:** AI coding agents such as Codex and Claude Code are increasingly used to autonomously contribute to software repositories. However, little is known about how repository-level configuration artifacts affect operational efficiency of the agents. In this paper, we study the impact of this http URL files on the runtime and token consumption of AI coding agents operating on GitHub pull requests. We analyze 10 repositories and 124 pull requests, executing agents under two conditions: with and without an this http URL file. We measure wall-clock execution time and token usage during agent execution. Our results show that the presence of this http URL is associated with a lower median runtime ($\Delta 28.64$%) and reduced output token consumption ($\Delta 16.58$%), while maintaining a comparable task completion behavior. Based on these results, we discuss immediate implications for the configuration and deployment of AI coding agents in practice, and outline a broader research agenda on the role of repository-level instructions in shaping the behavior, efficiency, and integration of AI coding agents in software development workflows.

**arXiv ID:** 2601.20404
</details>

<details>
<summary><strong>Cognition Envelopes for Bounded AI Reasoning in Autonomous UAS Operations</strong> - Pedro Antonio Alarcon Granadeno, Arturo Miguel Bernal Russell, Sofia Nelson, Demetrius Hernandez, Maureen Petterson, Michael Murphy, Walter J. Scheirer, Jane Cleland-Huang - [[pdf]](https://arxiv.org/pdf/2510.26905)</summary>

**Abstract:** Cyber-physical systems increasingly rely on Foundational Models such as Large Language Models (LLMs) and Vision-Language Models (VLMs) to increase autonomy through enhanced perception, inference, and planning. However, these models also introduce new types of errors, such as hallucinations, overgeneralizations, and context misalignments, resulting in incorrect and flawed decisions. To address this, we introduce the concept of Cognition Envelopes, designed to establish reasoning boundaries that constrain AI-generated decisions while complementing the use of meta-cognition and traditional safety envelopes. As with safety envelopes, Cognition Envelopes require practical guidelines and systematic processes for their definition, validation, and assurance.

**arXiv ID:** 2510.26905
</details>

<details>
<summary><strong>Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review</strong> - Matthew Lisondra, Beno Benhabib, Goldie Nejat - [[pdf]](https://arxiv.org/pdf/2505.20503)</summary>

**Abstract:** Rapid advancements in foundation models, including Large Language Models, Vision-Language Models, Multimodal Large Language Models, and Vision-Language-Action Models, have opened new avenues for embodied AI in mobile service robotics. By combining foundation models with the principles of embodied AI, where intelligent systems perceive, reason, and act through physical interaction, mobile service robots can achieve more flexible understanding, adaptive behavior, and robust task execution in dynamic real-world environments. Despite this progress, embodied AI for mobile service robots continues to face fundamental challenges related to the translation of natural language instructions into executable robot actions, multimodal perception in human-centered environments, uncertainty estimation for safe decision-making, and computational constraints for real-time onboard deployment. In this paper, we present the first systematic review focused specifically on the integration of foundation models in mobile service robotics. We analyze how recent advances in foundation models address these core challenges through language-conditioned control, multimodal sensor fusion, uncertainty-aware reasoning, and efficient model scaling. We further examine real-world applications in domestic assistance, healthcare, and service automation, highlighting how foundation models enable context-aware, socially responsive, and generalizable robot behaviors. Beyond technical considerations, we discuss ethical, societal, and human-interaction implications associated with deploying foundation model-enabled service robots in human environments. Finally, we outline future research directions emphasizing reliability and lifelong adaptation, privacy-aware and resource-constrained deployment, and governance and human-in-the-loop frameworks required for safe, scalable, and trustworthy mobile service robotics.

**arXiv ID:** 2505.20503
</details>

<details>
<summary><strong>Game-Theoretic Autonomous Driving: A Graphs of Convex Sets Approach</strong> - Nikolaj Käfer, Ahmed Khalil, Edward Huynh, Efstathios Bakolas, David Fridovich-Keil - [[pdf]](https://arxiv.org/pdf/2601.20054)</summary>

**Abstract:** Multi-vehicle autonomous driving couples strategic interaction with hybrid (discrete-continuous) maneuver planning under shared safety constraints. We introduce IBR-GCS, an Iterative Best Response (IBR) planning approach based on the Graphs of Convex Sets (GCS) framework that models highway driving as a generalized noncooperative game. IBR-GCS integrates combinatorial maneuver reasoning, trajectory planning, and game-theoretic interaction within a unified framework. The key novelty is a vehicle-specific, strategy-dependent GCS construction. Specifically, at each best-response update, each vehicle builds its own graph conditioned on the current strategies of the other vehicles, with vertices representing lane-specific, time-varying, convex, collision-free regions and edges encoding dynamically feasible transitions. This yields a shortest-path problem in GCS for each best-response step, which admits an efficient convex relaxation that can be solved using convex optimization tools without exhaustive discrete tree search. We then apply an iterative best-response scheme in which vehicles update their trajectories sequentially and provide conditions under which the resulting inexact updates converge to an approximate generalized Nash equilibrium. Simulation results across multi-lane, multi-vehicle scenarios demonstrate that IBR-GCS produces safe trajectories and strategically consistent interactive behaviors.

**arXiv ID:** 2601.20054
</details>

<details>
<summary><strong>Continual GUI Agents</strong> - Ziwei Liu, Borui Kang, Hangjie Yuan, Zixiang Zhao, Wei Li, Yifan Zhu, Tao Feng - [[pdf]](https://arxiv.org/pdf/2601.20732)</summary>

**Abstract:** As digital environments (data distribution) are in flux, with new GUI data arriving over time-introducing new domains or resolutions-agents trained on static environments deteriorate in performance. In this work, we introduce Continual GUI Agents, a new task that requires GUI agents to perform continual learning under shifted domains and resolutions. We find existing methods fail to maintain stable grounding as GUI distributions shift over time, due to the diversity of UI interaction points and regions in fluxing scenarios. To address this, we introduce GUI-Anchoring in Flux (GUI-AiF), a new reinforcement fine-tuning framework that stabilizes continual learning through two novel rewards: Anchoring Point Reward in Flux (APR-iF) and Anchoring Region Reward in Flux (ARR-iF). These rewards guide the agents to align with shifting interaction points and regions, mitigating the tendency of existing reward strategies to over-adapt to static grounding cues (e.g., fixed coordinates or element scales). Extensive experiments show GUI-AiF surpasses state-of-the-art baselines. Our work establishes the first continual learning framework for GUI agents, revealing the untapped potential of reinforcement fine-tuning for continual GUI Agents.

**arXiv ID:** 2601.20732
</details>

<details>
<summary><strong>Learning From a Steady Hand: A Weakly Supervised Agent for Robot Assistance under Microscopy</strong> - Huanyu Tian, Martin Huber, Lingyun Zeng, Zhe Han, Wayne Bennett, Giuseppe Silvestri, Gerardo Mendizabal-Ruiz, Tom Vercauteren, Alejandro Chavez-Badiola, Christos Bergeles - [[pdf]](https://arxiv.org/pdf/2601.20776)</summary>

**Abstract:** This paper rethinks steady-hand robotic manipulation by using a weakly supervised framework that fuses calibration-aware perception with admittance control. Unlike conventional automation that relies on labor-intensive 2D labeling, our framework leverages reusable warm-up trajectories to extract implicit spatial information, thereby achieving calibration-aware, depth-resolved perception without the need for external fiducials or manual depth annotation. By explicitly characterizing residuals from observation and calibration models, the system establishes a task-space error budget from recorded warm-ups. The uncertainty budget yields a lateral closed-loop accuracy of approx. 49 micrometers at 95% confidence (worst-case testing subset) and a depth accuracy of <= 291 micrometers at 95% confidence bound during large in-plane moves. In a within-subject user study (N=8), the learned agent reduces overall NASA-TLX workload by 77.1% relative to the simple steady-hand assistance baseline. These results demonstrate that the weakly supervised agent improves the reliability of microscope-guided biomedical micromanipulation without introducing complex setup requirements, offering a practical framework for microscope-guided intervention.

**arXiv ID:** 2601.20776
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (32 papers)</h2></summary>

<details>
<summary><strong>Fuzzy Categorical Planning: Autonomous Goal Satisfaction with Graded Semantic Constraints</strong> - Shuhui Qu - [[pdf]](https://arxiv.org/pdf/2601.20021)</summary>

**Abstract:** Natural-language planning often involves vague predicates (e.g., suitable substitute, stable enough) whose satisfaction is inherently graded. Existing category-theoretic planners provide compositional structure and pullback-based hard-constraint verification, but treat applicability as crisp, forcing thresholding that collapses meaningful distinctions and cannot track quality degradation across multi-step plans. We propose Fuzzy Category-theoretic Planning (FCP), which annotates each action (morphism) with a degree in [0,1], composes plan quality via a t-norm Lukasiewicz, and retains crisp executability checks via pullback verification. FCP grounds graded applicability from language using an LLM with k-sample median aggregation and supports meeting-in-the-middle search using residuum-based backward requirements. We evaluate on (i) public PDDL3 preference/oversubscription benchmarks and (ii) RecipeNLG-Subs, a missing-substitute recipe-planning benchmark built from RecipeNLG with substitution candidates from Recipe1MSubs and FoodKG. FCP improves success and reduces hard-constraint violations on RecipeNLG-Subs compared to LLM-only and ReAct-style baselines, while remaining competitive with classical PDDL3 planners.

**arXiv ID:** 2601.20021
</details>

<details>
<summary><strong>Scaling Medical Reasoning Verification via Tool-Integrated Reinforcement Learning</strong> - Hang Zhang, Ruheng Wang, Yuelyu Ji, Mingu Kwak, Xizhi Wu, Chenyu Li, Li Zhang, Wenqi Shi, Yifan Peng, Yanshan Wang - [[pdf]](https://arxiv.org/pdf/2601.20221)</summary>

**Abstract:** Large language models have achieved strong performance on medical reasoning benchmarks, yet their deployment in clinical settings demands rigorous verification to ensure factual accuracy. While reward models offer a scalable approach for reasoning trace verification, existing methods face two limitations: they produce only scalar reward values without explicit justification, and they rely on single-pass retrieval that precludes adaptive knowledge access as verification unfolds. We introduce $\method$, an agentic framework that addresses these limitations by training medical reasoning verifiers to iteratively query external medical corpora during evaluation. Our approach combines tool-augmented verification with an iterative reinforcement learning paradigm that requires only trace-level supervision, alongside an adaptive curriculum mechanism that dynamically adjusts training data distribution. Across four medical reasoning benchmarks, $\method$ achieves substantial gains over existing methods, improving MedQA accuracy by 23.5% and MedXpertQA by 32.0% relative to the base generator in particular. Crucially, $\method$ demonstrates an $\mathbf{8\times}$ reduction in sampling budget requirement compared to prior reward model baselines. These findings establish that grounding verification in dynamically retrieved evidence offers a principled path toward more reliable medical reasoning systems.

**arXiv ID:** 2601.20221
</details>

<details>
<summary><strong>OmegaUse: Building a General-Purpose GUI Agent for Autonomous Task Execution</strong> - Le Zhang, Yixiong Xiao, Xinjiang Lu, Jingjia Cao, Yusai Zhao, Jingbo Zhou, Lang An, Zikan Feng, Wanxiang Sha, Yu Shi, Congxi Xiao, Jian Xiong, Yankai Zhang, Hua Wu, Haifeng Wang - [[pdf]](https://arxiv.org/pdf/2601.20380)</summary>

**Abstract:** Graphical User Interface (GUI) agents show great potential for enabling foundation models to complete real-world tasks, revolutionizing human-computer interaction and improving human productivity. In this report, we present OmegaUse, a general-purpose GUI agent model for autonomous task execution on both mobile and desktop platforms, supporting computer-use and phone-use scenarios. Building an effective GUI agent model relies on two factors: (1) high-quality data and (2) effective training methods. To address these, we introduce a carefully engineered data-construction pipeline and a decoupled training paradigm. For data construction, we leverage rigorously curated open-source datasets and introduce a novel automated synthesis framework that integrates bottom-up autonomous exploration with top-down taxonomy-guided generation to create high-fidelity synthetic data. For training, to better leverage these data, we adopt a two-stage strategy: Supervised Fine-Tuning (SFT) to establish fundamental interaction syntax, followed by Group Relative Policy Optimization (GRPO) to improve spatial grounding and sequential planning. To balance computational efficiency with agentic reasoning capacity, OmegaUse is built on a Mixture-of-Experts (MoE) backbone. To evaluate cross-terminal capabilities in an offline setting, we introduce OS-Nav, a benchmark suite spanning multiple operating systems: ChiM-Nav, targeting Chinese Android mobile environments, and Ubu-Nav, focusing on routine desktop interactions on Ubuntu. Extensive experiments show that OmegaUse is highly competitive across established GUI benchmarks, achieving a state-of-the-art (SOTA) score of 96.3% on ScreenSpot-V2 and a leading 79.1% step success rate on AndroidControl. OmegaUse also performs strongly on OS-Nav, reaching 74.24% step success on ChiM-Nav and 55.9% average success on Ubu-Nav.

**arXiv ID:** 2601.20380
</details>

<details>
<summary><strong>Normative Equivalence in human-AI Cooperation: Behaviour, Not Identity, Drives Cooperation in Mixed-Agent Groups</strong> - Nico Mutzner, Taha Yasseri, Heiko Rauhut - [[pdf]](https://arxiv.org/pdf/2601.20487)</summary>

**Abstract:** The introduction of artificial intelligence (AI) agents into human group settings raises essential questions about how these novel participants influence cooperative social norms. While previous studies on human-AI cooperation have primarily focused on dyadic interactions, little is known about how integrating AI agents affects the emergence and maintenance of cooperative norms in small groups. This study addresses this gap through an online experiment using a repeated four-player Public Goods Game (PGG). Each group consisted of three human participants and one bot, which was framed either as human or AI and followed one of three predefined decision strategies: unconditional cooperation, conditional cooperation, or free-riding. In our sample of 236 participants, we found that reciprocal group dynamics and behavioural inertia primarily drove cooperation. These normative mechanisms operated identically across conditions, resulting in cooperation levels that did not differ significantly between human and AI labels. Furthermore, we found no evidence of differences in norm persistence in a follow-up Prisoner's Dilemma, or in participants' normative perceptions. Participants' behaviour followed the same normative logic across human and AI conditions, indicating that cooperation depended on group behaviour rather than partner identity. This supports a pattern of normative equivalence, in which the mechanisms that sustain cooperation function similarly in mixed human-AI and all human groups. These findings suggest that cooperative norms are flexible enough to extend to artificial agents, blurring the boundary between humans and AI in collective decision-making.

**arXiv ID:** 2601.20487
</details>

<details>
<summary><strong>Meta-Cognitive Reinforcement Learning with Self-Doubt and Recovery</strong> - Zhipeng Zhang, Wenting Ma, Kai Li, Meng Guo, Lei Yang, Wei Yu, Hongji Cui, Yichen Zhang, Mo Zhang, Jinzhe Lin, Zhenjie Yao - [[pdf]](https://arxiv.org/pdf/2601.20193)</summary>

**Abstract:** Robust reinforcement learning methods typically focus on suppressing unreliable experiences or corrupted rewards, but they lack the ability to reason about the reliability of their own learning process. As a result, such methods often either overreact to noise by becoming overly conservative or fail catastrophically when uncertainty accumulates.
In this work, we propose a meta-cognitive reinforcement learning framework that enables an agent to assess, regulate, and recover its learning behavior based on internally estimated reliability signals. The proposed method introduces a meta-trust variable driven by Value Prediction Error Stability (VPES), which modulates learning dynamics via fail-safe regulation and gradual trust recovery.
Experiments on continuous-control benchmarks with reward corruption demonstrate that recovery-enabled meta-cognitive control achieves higher average returns and significantly reduces late-stage training failures compared to strong robustness baselines.

**arXiv ID:** 2601.20193
</details>

<details>
<summary><strong>Inequality in Congestion Games with Learning Agents</strong> - Dimitris Michailidis, Sennay Ghebreab, Fernando P. Santos - [[pdf]](https://arxiv.org/pdf/2601.20578)</summary>

**Abstract:** Who benefits from expanding transport networks? While designed to improve mobility, such interventions can also create inequality. In this paper, we show that disparities arise not only from the structure of the network itself but also from differences in how commuters adapt to it. We model commuters as reinforcement learning agents who adapt their travel choices at different learning rates, reflecting unequal access to resources and information. To capture potential efficiency-fairness tradeoffs, we introduce the Price of Learning (PoL), a measure of inefficiency during learning. We analyze both a stylized network -- inspired in the well-known Braess's paradox, yet with two-source nodes -- and an abstraction of a real-world metro system (Amsterdam). Our simulations show that network expansions can simultaneously increase efficiency and amplify inequality, especially when faster learners disproportionately benefit from new routes before others adapt. These results highlight that transport policies must account not only for equilibrium outcomes but also for the heterogeneous ways commuters adapt, since both shape the balance between efficiency and fairness.

**arXiv ID:** 2601.20578
</details>

<details>
<summary><strong>Ranking-aware Reinforcement Learning for Ordinal Ranking</strong> - Aiming Hao, Chen Zhu, Jiashu Zhu, Jiahong Wu, Xiangxiang Chu - [[pdf]](https://arxiv.org/pdf/2601.20585)</summary>

**Abstract:** Ordinal regression and ranking are challenging due to inherent ordinal dependencies that conventional methods struggle to model. We propose Ranking-Aware Reinforcement Learning (RARL), a novel RL framework that explicitly learns these relationships. At its core, RARL features a unified objective that synergistically integrates regression and Learning-to-Rank (L2R), enabling mutual improvement between the two tasks. This is driven by a ranking-aware verifiable reward that jointly assesses regression precision and ranking accuracy, facilitating direct model updates via policy optimization. To further enhance training, we introduce Response Mutation Operations (RMO), which inject controlled noise to improve exploration and prevent stagnation at saddle points. The effectiveness of RARL is validated through extensive experiments on three distinct benchmarks.

**arXiv ID:** 2601.20585
</details>

<details>
<summary><strong>Adapting the Behavior of Reinforcement Learning Agents to Changing Action Spaces and Reward Functions</strong> - Raul de la Rosa, Ivana Dusparic, Nicolas Cardozo - [[pdf]](https://arxiv.org/pdf/2601.20714)</summary>

**Abstract:** Reinforcement Learning (RL) agents often struggle in real-world applications where environmental conditions are non-stationary, particularly when reward functions shift or the available action space expands. This paper introduces MORPHIN, a self-adaptive Q-learning framework that enables on-the-fly adaptation without full retraining. By integrating concept drift detection with dynamic adjustments to learning and exploration hyperparameters, MORPHIN adapts agents to changes in both the reward function and on-the-fly expansions of the agent's action space, while preserving prior policy knowledge to prevent catastrophic forgetting. We validate our approach using a Gridworld benchmark and a traffic signal control simulation. The results demonstrate that MORPHIN achieves superior convergence speed and continuous adaptation compared to a standard Q-learning baseline, improving learning efficiency by up to 1.7x.

**arXiv ID:** 2601.20714
</details>

<details>
<summary><strong>Reinforcement Learning via Self-Distillation</strong> - Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, Andreas Krause - [[pdf]](https://arxiv.org/pdf/2601.20802)</summary>

**Abstract:** Large language models are increasingly post-trained with reinforcement learning in verifiable domains such as code and math. Yet, current methods for reinforcement learning with verifiable rewards (RLVR) learn only from a scalar outcome reward per attempt, creating a severe credit-assignment bottleneck. Many verifiable environments actually provide rich textual feedback, such as runtime errors or judge evaluations, that explain why an attempt failed. We formalize this setting as reinforcement learning with rich feedback and introduce Self-Distillation Policy Optimization (SDPO), which converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model. SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed next-token predictions back into the policy. In this way, SDPO leverages the model's ability to retrospectively identify its own mistakes in-context. Across scientific reasoning, tool use, and competitive programming on LiveCodeBench v6, SDPO improves sample efficiency and final accuracy over strong RLVR baselines. Notably, SDPO also outperforms baselines in standard RLVR environments that only return scalar feedback by using successful rollouts as implicit feedback for failed attempts. Finally, applying SDPO to individual questions at test time accelerates discovery on difficult binary-reward tasks, achieving the same discovery probability as best-of-k sampling or multi-turn conversations with 3x fewer attempts.

**arXiv ID:** 2601.20802
</details>

<details>
<summary><strong>Beyond Syntax: Action Semantics Learning for App Agents</strong> - Bohan Tang, Dezhao Luo, Jianheng Liu, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, Kun Shao - [[pdf]](https://arxiv.org/pdf/2506.17697)</summary>

**Abstract:** The recent development of Large Language Models (LLMs) enables the rise of App agents that interpret user intent and operate smartphone Apps through actions such as clicking and scrolling. While prompt-based solutions with proprietary LLM APIs show promising ability, they incur heavy compute costs and external API dependency. Fine-tuning smaller open-source LLMs solves these limitations. However, current supervised fine-tuning methods use a syntax learning paradigm that forces agents to reproduce exactly the ground truth action strings, leading to out-of-distribution (OOD) vulnerability. To fill this gap, we propose Action Semantics Learning (ASL), a novel learning framework, where the learning objective is capturing the semantics of the ground truth actions. Specifically, inspired by the programming language theory, we define the action semantics for App agents as the state transition induced by the action in the user interface. Building on this insight, ASL employs a novel SEmantic Estimator~(SEE) to compute a semantic similarity to train the App agents in generating actions aligned with the semantics of ground truth actions, even when their syntactic forms differ. SEE is a flexible module that can be applied in both supervised and reinforcement fine-tuning paradigms. To support the effectiveness of ASL, we theoretically demonstrate the superior robustness of ASL for the OOD problem compared with the existing syntax learning paradigm. Extensive experiments across multiple offline and online benchmarks demonstrate that ASL significantly improves the accuracy and generalisation of App agents compared to existing methods.

**arXiv ID:** 2506.17697
</details>

<details>
<summary><strong>Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models</strong> - David Bani-Harouni, Chantal Pellegrini, Paul Stangel, Ege Özsoy, Kamilia Zaripova, Matthias Keicher, Nassir Navab - [[pdf]](https://arxiv.org/pdf/2503.02623)</summary>

**Abstract:** A safe and trustworthy use of Large Language Models (LLMs) requires an accurate expression of confidence in their answers. We propose a novel Reinforcement Learning approach that allows to directly fine-tune LLMs to express calibrated confidence estimates alongside their answers to factual questions. Our method optimizes a reward based on the logarithmic scoring rule, explicitly penalizing both over- and under-confidence. This encourages the model to align its confidence estimates with the actual predictive accuracy. The optimal policy under our reward design would result in perfectly calibrated confidence expressions. Unlike prior approaches that decouple confidence estimation from response generation, our method integrates confidence calibration seamlessly into the generative process of the LLM. Empirically, we demonstrate that models trained with our approach exhibit substantially improved calibration and generalize to unseen tasks without further fine-tuning, suggesting the emergence of general confidence awareness.

**arXiv ID:** 2503.02623
</details>

<details>
<summary><strong>Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning</strong> - Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, Yu Cheng - [[pdf]](https://arxiv.org/pdf/2506.04207)</summary>

**Abstract:** Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning. This staged training approach effectively balances perceptual grounding and cognitive reasoning development. By incorporating the above insights and addressing multimodal RL issues, we introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025.

**arXiv ID:** 2506.04207
</details>

<details>
<summary><strong>OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning</strong> - Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan - [[pdf]](https://arxiv.org/pdf/2509.09332)</summary>

**Abstract:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible. To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: this https URL

**arXiv ID:** 2509.09332
</details>

<details>
<summary><strong>Spectral Representation-based Reinforcement Learning</strong> - Chenxiao Gao, Haotian Sun, Na Li, Dale Schuurmans, Bo Dai - [[pdf]](https://arxiv.org/pdf/2512.15036)</summary>

**Abstract:** In real-world applications with large state and action spaces, reinforcement learning (RL) typically employs function approximations to represent core components like the policies, value functions, and dynamics models. Although powerful approximations such as neural networks offer great expressiveness, they often present theoretical ambiguities, suffer from optimization instability and exploration difficulty, and incur substantial computational costs in practice. In this paper, we introduce the perspective of spectral representations as a solution to address these difficulties in RL. Stemming from the spectral decomposition of the transition operator, this framework yields an effective abstraction of the system dynamics for subsequent policy optimization while also providing a clear theoretical characterization. We reveal how to construct spectral representations for transition operators that possess latent variable structures or energy-based structures, which implies different learning methods to extract spectral representations from data. Notably, each of these learning methods realizes an effective RL algorithm under this framework. We also provably extend this spectral view to partially observable MDPs. Finally, we validate these algorithms on over 20 challenging tasks from the DeepMind Control Suite, where they achieve performances comparable or superior to current state-of-the-art model-free and model-based baselines.

**arXiv ID:** 2512.15036
</details>

<details>
<summary><strong>GRACE: A Language Model Framework for Explainable Inverse Reinforcement Learning</strong> - Silvia Sapora, Devon Hjelm, Alexander Toshev, Omar Attia, Bogdan Mazoure - [[pdf]](https://arxiv.org/pdf/2510.02180)</summary>

**Abstract:** Inverse Reinforcement Learning aims to recover reward models from expert demonstrations, but traditional methods yield black-box models that are difficult to interpret and debug. In this work, we introduce GRACE (Generating Rewards As CodE), a method for using Large Language Models within an evolutionary search to reverse-engineer an interpretable, code-based reward function directly from expert trajectories. The resulting reward function is executable code that can be inspected and verified. We empirically validate GRACE on the MuJoCo, BabyAI and AndroidWorld benchmarks, where it efficiently learns highly accurate rewards, even in complex, multi-task settings. Further, we demonstrate that the resulting reward leads to strong policies, compared to both competitive Imitation Learning and online RL approaches with ground-truth rewards. Finally, we show that GRACE is able to build complex reward APIs in multi-task setups.

**arXiv ID:** 2510.02180
</details>

<details>
<summary><strong>Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning</strong> - Jiayun Wu, Jiashuo Liu, Zhiyuan Zeng, Tianyang Zhan, Tianle Cai, Wenhao Huang - [[pdf]](https://arxiv.org/pdf/2512.19920)</summary>

**Abstract:** LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower.

**arXiv ID:** 2512.19920
</details>

<details>
<summary><strong>Trajectory2Task: Training Robust Tool-Calling Agents with Synthesized Yet Verifiable Data for Complex User Intents</strong> - Ziyi Wang, Yuxuan Lu, Yimeng Zhang, Jing Huang, Jiri Gesi, Xianfeng Tang, Chen Luo, Yisi Sang, Hanqing Lu, Manling Li, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2601.20144)</summary>

**Abstract:** Tool-calling agents are increasingly deployed in real-world customer-facing workflows. Yet most studies on tool-calling agents focus on idealized settings with general, fixed, and well-specified tasks. In real-world applications, user requests are often (1) ambiguous, (2) changing over time, or (3) infeasible due to policy constraints, and training and evaluation data that cover these diverse, complex interaction patterns remain under-represented. To bridge the gap, we present Trajectory2Task, a verifiable data generation pipeline for studying tool use at scale under three realistic user scenarios: ambiguous intent, changing intent, and infeasible intents. The pipeline first conducts multi-turn exploration to produce valid tool-call trajectories. It then converts these trajectories into user-facing tasks with controlled intent adaptations. This process yields verifiable task that support closed-loop evaluation and training. We benchmark seven state-of-the-art LLMs on the generated complex user scenario tasks and observe frequent failures. Finally, using successful trajectories obtained from task rollouts, we fine-tune lightweight LLMs and find consistent improvements across all three conditions, along with better generalization to unseen tool-use domains, indicating stronger general tool-calling ability.

**arXiv ID:** 2601.20144
</details>

<details>
<summary><strong>Unit-Based Agent for Semi-Cascaded Full-Duplex Dialogue Systems</strong> - Haoyuan Yu, Yuxuan Chen, Minjie Cai - [[pdf]](https://arxiv.org/pdf/2601.20230)</summary>

**Abstract:** Full-duplex voice interaction is crucial for natural human computer interaction. We present a framework that decomposes complex dialogue into minimal conversational units, enabling the system to process each unit independently and predict when to transit to the next. This framework is instantiated as a semi-cascaded full-duplex dialogue system built around a multimodal large language model, supported by auxiliary modules such as voice activity detection (VAD) and text-to-speech (TTS) synthesis. The resulting system operates in a train-free, plug-and-play manner. Experiments on the HumDial dataset demonstrate the effectiveness of our framework, which ranks second among all teams on the test set of the Human-like Spoken Dialogue Systems Challenge (Track 2: Full-Duplex Interaction). Code is available at the GitHub repository this https URL.

**arXiv ID:** 2601.20230
</details>

<details>
<summary><strong>PEARL: Plan Exploration and Adaptive Reinforcement Learning for Multihop Tool Use</strong> - Qihao Wang, Mingzhe Lu, Jiayue Wu, Yue Hu, Yanbing Liu - [[pdf]](https://arxiv.org/pdf/2601.20439)</summary>

**Abstract:** Large Language Models show great potential with external tools, but face significant challenges in complex, multi-turn tool invocation. They often exhibit weak planning, tool hallucination, erroneous parameter generation, and struggle with robust interaction. To tackle these issues, we present PEARL, a novel framework to enhance LLM planning and execution for sophisticated tool use. PEARL adopts a two-stage approach: an offline phase where the agent explores tools to learn valid usage patterns and failure conditions, and an online reinforcement learning phase. In the online phase, a dedicated Planner is trained via group Relative Policy Optimization (GRPO) with a carefully designed reward function that provides distinct signals for planning quality. Experiments on the ToolHop and T-Eval benchmarks show PEARL significantly outperforms existing methods, achieving a new state-of-the-art success rate of \textbf{56.5\%} on ToolHop while maintaining a low invocation error rate. Our work marks a key advance in addressing the complex planning challenges of tool use, contributing to the development of more robust and reliable LLM-based agents.

**arXiv ID:** 2601.20439
</details>

<details>
<summary><strong>SERA: Soft-Verified Efficient Repository Agents</strong> - Ethan Shen, Danny Tormoen, Saurabh Shah, Ali Farhadi, Tim Dettmers - [[pdf]](https://arxiv.org/pdf/2601.20789)</summary>

**Abstract:** Open-weight coding agents should hold a fundamental advantage over closed-source systems: they can be specialized to private codebases, encoding repository-specific information directly in their weights. Yet the cost and complexity of training has kept this advantage theoretical. We show it is now practical. We present Soft-Verified Efficient Repository Agents (SERA), an efficient method for training coding agents that enables the rapid and cheap creation of agents specialized to private codebases. Using only supervised finetuning (SFT), SERA achieves state-of-the-art results among fully open-source (open data, method, code) models while matching the performance of frontier open-weight models like Devstral-Small-2. Creating SERA models is 26x cheaper than reinforcement learning and 57x cheaper than previous synthetic data methods to reach equivalent performance. Our method, Soft Verified Generation (SVG), generates thousands of trajectories from a single code repository. Combined with cost-efficiency, this enables specialization to private codebases. Beyond repository specialization, we apply SVG to a larger corpus of codebases, generating over 200,000 synthetic trajectories. We use this dataset to provide detailed analysis of scaling laws, ablations, and confounding factors for training coding agents. Overall, we believe our work will greatly accelerate research on open coding agents and showcase the advantage of open-source models that can specialize to private codebases. We release SERA as the first model in Ai2's Open Coding Agents series, along with all our code, data, and Claude Code integration to support the research community.

**arXiv ID:** 2601.20789
</details>

<details>
<summary><strong>Spark: Strategic Policy-Aware Exploration via Dynamic Branching for Long-Horizon Agentic Learning</strong> - Jinyang Wu, Shuo Yang, Changpeng Yang, Yuhao Shen, Shuai Zhang, Zhengqi Wen, Jianhua Tao - [[pdf]](https://arxiv.org/pdf/2601.20209)</summary>

**Abstract:** Reinforcement learning has empowered large language models to act as intelligent agents, yet training them for long-horizon tasks remains challenging due to the scarcity of high-quality trajectories, especially under limited resources. Existing methods typically scale up rollout sizes and indiscriminately allocate computational resources among intermediate steps. Such attempts inherently waste substantial computation budget on trivial steps while failing to guarantee sample quality. To address this, we propose \textbf{Spark} (\textbf{S}trategic \textbf{P}olicy-\textbf{A}ware explo\textbf{R}ation via \textbf{K}ey-state dynamic branching), a novel framework that selectively branches at critical decision states for resource-efficient exploration. Our key insight is to activate adaptive branching exploration at critical decision points to probe promising trajectories, thereby achieving precise resource allocation that prioritizes sampling quality over blind coverage. This design leverages the agent's intrinsic decision-making signals to reduce dependence on human priors, enabling the agent to autonomously expand exploration and achieve stronger generalization. Experiments across diverse tasks (e.g., embodied planning), demonstrate that \textsc{Spark} achieves superior success rates with significantly fewer training samples, exhibiting robust generalization even in unseen scenarios.

**arXiv ID:** 2601.20209
</details>

<details>
<summary><strong>HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning</strong> - Ziang Cui, Mengran Yu, Tianjiao Li, Chenyu Shi, Yingxuan Shi, Lusheng Zhang, Hongwei Lin - [[pdf]](https://arxiv.org/pdf/2601.10187)</summary>

**Abstract:** Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose HOMURA, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a KL-regularized objective with a novel dynamic syllable-ratio reward, HOMURA effectively "tames" the output length. Experimental results demonstrate that our method significantly outperforms strong LLM baselines, achieving precise length control that respects linguistic density hierarchies without compromising semantic adequacy.

**arXiv ID:** 2601.10187
</details>

<details>
<summary><strong>In-Context Reinforcement Learning From Suboptimal Historical Data</strong> - Juncheng Dong, Moyang Guo, Ethan X. Fang, Zhuoran Yang, Vahid Tarokh - [[pdf]](https://arxiv.org/pdf/2601.20116)</summary>

**Abstract:** Transformer models have achieved remarkable empirical successes, largely due to their in-context learning capabilities. Inspired by this, we explore training an autoregressive transformer for in-context reinforcement learning (ICRL). In this setting, we initially train a transformer on an offline dataset consisting of trajectories collected from various RL tasks, and then fix and use this transformer to create an action policy for new RL tasks. Notably, we consider the setting where the offline dataset contains trajectories sampled from suboptimal behavioral policies. In this case, standard autoregressive training corresponds to imitation learning and results in suboptimal performance. To address this, we propose the Decision Importance Transformer(DIT) framework, which emulates the actor-critic algorithm in an in-context manner. In particular, we first train a transformer-based value function that estimates the advantage functions of the behavior policies that collected the suboptimal trajectories. Then we train a transformer-based policy via a weighted maximum likelihood estimation loss, where the weights are constructed based on the trained value function to steer the suboptimal policies to the optimal ones. We conduct extensive experiments to test the performance of DIT on both bandit and Markov Decision Process problems. Our results show that DIT achieves superior performance, particularly when the offline dataset contains suboptimal historical data.

**arXiv ID:** 2601.20116
</details>

<details>
<summary><strong>A Reinforcement Learning Based Universal Sequence Design for Polar Codes</strong> - David Kin Wai Ho, Arman Fazeli, Mohamad M. Mansour, Louay M. A. Jalloul - [[pdf]](https://arxiv.org/pdf/2601.20118)</summary>

**Abstract:** To advance Polar code design for 6G applications, we develop a reinforcement learning-based universal sequence design framework that is extensible and adaptable to diverse channel conditions and decoding strategies. Crucially, our method scales to code lengths up to $2048$, making it suitable for use in standardization. Across all $(N,K)$ configurations supported in 5G, our approach achieves competitive performance relative to the NR sequence adopted in 5G and yields up to a 0.2 dB gain over the beta-expansion baseline at $N=2048$. We further highlight the key elements that enabled learning at scale: (i) incorporation of physical law constrained learning grounded in the universal partial order property of Polar codes, (ii) exploitation of the weak long term influence of decisions to limit lookahead evaluation, and (iii) joint multi-configuration optimization to increase learning efficiency.

**arXiv ID:** 2601.20118
</details>

<details>
<summary><strong>Positive-Unlabeled Reinforcement Learning Distillation for On-Premise Small Models</strong> - Zhiqiang Kou, Junyang Chen, Xin-Qiang Cai, Xiaobo Xia, Ming-Kun Xie, Dong-Dong Wu, Biao Liu, Yuheng Jia, Xin Geng, Masashi Sugiyama, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2601.20687)</summary>

**Abstract:** Due to constraints on privacy, cost, and latency, on-premise deployment of small models is increasingly common. However, most practical pipelines stop at supervised fine-tuning (SFT) and fail to reach the reinforcement learning (RL) alignment stage. The main reason is that RL alignment typically requires either expensive human preference annotation or heavy reliance on high-quality reward models with large-scale API usage and ongoing engineering maintenance, both of which are ill-suited to on-premise settings. To bridge this gap, we propose a positive-unlabeled (PU) RL distillation method for on-premise small-model deployment. Without human-labeled preferences or a reward model, our method distills the teacher's preference-optimization capability from black-box generations into a locally trainable student. For each prompt, we query the teacher once to obtain an anchor response, locally sample multiple student candidates, and perform anchor-conditioned self-ranking to induce pairwise or listwise preferences, enabling a fully local training loop via direct preference optimization or group relative policy optimization. Theoretical analysis justifies that the induced preference signal by our method is order-consistent and concentrates on near-optimal candidates, supporting its stability for preference optimization. Experiments demonstrate that our method achieves consistently strong performance under a low-cost setting.

**arXiv ID:** 2601.20687
</details>

<details>
<summary><strong>E2HiL: Entropy-Guided Sample Selection for Efficient Real-World Human-in-the-Loop Reinforcement Learning</strong> - Haoyuan Deng, Yuanjiang Xue, Haoyang Du, Boyang Zhou, Zhenyu Wu, Ziwei Wang - [[pdf]](https://arxiv.org/pdf/2601.19969)</summary>

**Abstract:** Human-in-the-loop guidance has emerged as an effective approach for enabling faster convergence in online reinforcement learning (RL) of complex real-world manipulation tasks. However, existing human-in-the-loop RL (HiL-RL) frameworks often suffer from low sample efficiency, requiring substantial human interventions to achieve convergence and thereby leading to high labor costs. To address this, we propose a sample-efficient real-world human-in-the-loop RL framework named \method, which requires fewer human intervention by actively selecting informative samples. Specifically, stable reduction of policy entropy enables improved trade-off between exploration and exploitation with higher sample efficiency. We first build influence functions of different samples on the policy entropy, which is efficiently estimated by the covariance of action probabilities and soft advantages of policies. Then we select samples with moderate values of influence functions, where shortcut samples that induce sharp entropy drops and noisy samples with negligible effect are pruned. Extensive experiments on four real-world manipulation tasks demonstrate that \method achieves a 42.1\% higher success rate while requiring 10.1\% fewer human interventions compared to the state-of-the-art HiL-RL method, validating its effectiveness. The project page providing code, videos, and mathematical formulations can be found at this https URL.

**arXiv ID:** 2601.19969
</details>

<details>
<summary><strong>Exploring the holographic entropy cone via reinforcement learning</strong> - Temple He, Jaeha Lee, Hirosi Ooguri - [[pdf]](https://arxiv.org/pdf/2601.19979)</summary>

**Abstract:** We develop a reinforcement learning algorithm to study the holographic entropy cone. Given a target entropy vector, our algorithm searches for a graph realization whose min-cut entropies match the target vector. If the target vector does not admit such a graph realization, it must lie outside the cone, in which case the algorithm finds a graph whose corresponding entropy vector most nearly approximates the target and allows us to probe the location of the facets. For the $\sf N=3$ cone, we confirm that our algorithm successfully rediscovers monogamy of mutual information beginning with a target vector outside the holographic entropy cone. We then apply the algorithm to the $\sf N=6$ cone, analyzing the 6 "mystery" extreme rays of the subadditivity cone from arXiv:2412.15364 that satisfy all known holographic entropy inequalities yet lacked graph realizations. We found realizations for 3 of them, proving they are genuine extreme rays of the holographic entropy cone, while providing evidence that the remaining 3 are not realizable, implying unknown holographic inequalities exist for $\sf N=6$.

**arXiv ID:** 2601.19979
</details>

<details>
<summary><strong>Less is More: Benchmarking LLM Based Recommendation Agents</strong> - Kargi Chauhan, Mahalakshmi Venkateswarlu - [[pdf]](https://arxiv.org/pdf/2601.20316)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly deployed for personalized product recommendations, with practitioners commonly assuming that longer user purchase histories lead to better predictions. We challenge this assumption through a systematic benchmark of four state of the art LLMs GPT-4o-mini, DeepSeek-V3, Qwen2.5-72B, and Gemini 2.5 Flash across context lengths ranging from 5 to 50 items using the REGEN dataset.
Surprisingly, our experiments with 50 users in a within subject design reveal no significant quality improvement with increased context length. Quality scores remain flat across all conditions (0.17--0.23). Our findings have significant practical implications: practitioners can reduce inference costs by approximately 88\% by using context (5--10 items) instead of longer histories (50 items), without sacrificing recommendation quality. We also analyze latency patterns across providers and find model specific behaviors that inform deployment decisions. This work challenges the existing ``more context is better'' paradigm and provides actionable guidelines for cost effective LLM based recommendation systems.

**arXiv ID:** 2601.20316
</details>

<details>
<summary><strong>Generalization in Reinforcement Learning for Radio Access Networks</strong> - Burak Demirel, Yu Wang, Cristian Tatino, Pablo Soldati - [[pdf]](https://arxiv.org/pdf/2507.06602)</summary>

**Abstract:** Modern RAN operate in highly dynamic and heterogeneous environments, where hand-tuned, rule-based RRM algorithms often underperform. While RL can surpass such heuristics in constrained settings, the diversity of deployments and unpredictable radio conditions introduce major generalization challenges. Data-driven policies frequently overfit to training conditions, degrading performance in unseen scenarios. To address this, we propose a generalization-centered RL framework for RAN control that: (i) robustly reconstructs dynamically varying states from partial and noisy observations, while encoding static and semi-static information, such as radio nodes, cell attributes, and their topology, through graph representations; (ii) applies domain randomization to broaden the training distribution; and (iii) distributes data generation across multiple actors while centralizing training in a cloud-compatible architecture aligned with O-RAN principles. Although generalization increases computational and data-management complexity, our distributed design mitigates this by scaling data collection and training across diverse network conditions. Applied to downlink link adaptation in five 5G benchmarks, our policy improves average throughput and spectral efficiency by ~10% over an OLLA baseline (10% BLER target) in full-buffer MIMO/mMIMO and by >20% under high mobility. It matches specialized RL in full-buffer traffic and achieves up to 4- and 2-fold gains in eMBB and mixed-traffic benchmarks, respectively. In nine-cell deployments, GAT models offer 30% higher throughput over MLP baselines. These results, combined with our scalable architecture, offer a path toward AI-native 6G RAN using a single, generalizable RL agent.

**arXiv ID:** 2507.06602
</details>

<details>
<summary><strong>Smart Exploration in Reinforcement Learning using Bounded Uncertainty Models</strong> - J.S. van Hulst, W.P.M.H. Heemels, D.J. Antunes - [[pdf]](https://arxiv.org/pdf/2504.05978)</summary>

**Abstract:** Reinforcement learning (RL) is a powerful framework for decision-making in uncertain environments, but it often requires large amounts of data to learn an optimal policy. We address this challenge by incorporating prior model knowledge to guide exploration and accelerate the learning process. Specifically, we assume access to a model set that contains the true transition kernel and reward function. We optimize over this model set to obtain upper and lower bounds on the Q-function, which are then used to guide the exploration of the agent. We provide theoretical guarantees on the convergence of the Q-function to the optimal Q-function under the proposed class of exploring policies. Furthermore, we also introduce a data-driven regularized version of the model set optimization problem that ensures the convergence of the class of exploring policies to the optimal policy. Lastly, we show that when the model set has a specific structure, namely the bounded-parameter MDP (BMDP) framework, the regularized model set optimization problem becomes convex and simple to implement. In this setting, we also prove finite-time convergence to the optimal policy under mild assumptions. We demonstrate the effectiveness of the proposed exploration strategy, which we call BUMEX (Bounded Uncertainty Model-based Exploration), in a simulation study. The results indicate that the proposed method can significantly accelerate learning in benchmark examples. A toolbox is available at this https URL.

**arXiv ID:** 2504.05978
</details>

<details>
<summary><strong>Practical Policy Distillation for Reinforcement Learning in Radio Access Networks</strong> - Sara Khosravi, Burak Demirel, Linghui Zhou, Javier Rasines, Pablo Soldati - [[pdf]](https://arxiv.org/pdf/2511.06563)</summary>

**Abstract:** Adopting artificial intelligence (AI) in radio access networks (RANs) presents several challenges, including limited availability of link-level measurements (e.g., CQI reports), stringent real-time processing constraints (e.g., sub-1 ms per TTI), and network heterogeneity (different spectrum bands, cell types, and vendor equipment). A critical yet often overlooked barrier lies in the computational and memory limitations of RAN baseband hardware, particularly in legacy 4th Generation (4G) systems, which typically lack on-chip neural accelerators. As a result, only lightweight AI models (under 1 Mb and sub-100~\mu s inference time) can be effectively deployed, limiting both their performance and applicability. However, achieving strong generalization across diverse network conditions often requires large-scale models with substantial resource demands. To address this trade-off, this paper investigates policy distillation in the context of a reinforcement learning-based link adaptation task. We explore two strategies: single-policy distillation, where a scenario-agnostic teacher model is compressed into one generalized student model; and multi-policy distillation, where multiple scenario-specific teachers are consolidated into a single generalist student. Experimental evaluations in a high-fidelity, 5th Generation (5G)-compliant simulator demonstrate that both strategies produce compact student models that preserve the teachers' generalization capabilities while complying with the computational and memory limitations of existing RAN hardware.

**arXiv ID:** 2511.06563
</details>

<details>
<summary><strong>Variational Quantum Circuit-Based Reinforcement Learning for Dynamic Portfolio Optimization</strong> - Vincent Gurgul, Ying Chen, Stefan Lessmann - [[pdf]](https://arxiv.org/pdf/2601.18811)</summary>

**Abstract:** This paper presents a Quantum Reinforcement Learning (QRL) solution to the dynamic portfolio optimization problem based on Variational Quantum Circuits. The implemented QRL approaches are quantum analogues of the classical neural-network-based Deep Deterministic Policy Gradient and Deep Q-Network algorithms. Through an empirical evaluation on real-world financial data, we show that our quantum agents achieve risk-adjusted performance comparable to, and in some cases exceeding, that of classical Deep RL models with several orders of magnitude more parameters. However, while quantum circuit execution is inherently fast at the hardware level, practical deployment on cloud-based quantum systems introduces substantial latency, making end-to-end runtime currently dominated by infrastructural overhead and limiting practical applicability. Taken together, our results suggest that QRL is theoretically competitive with state-of-the-art classical reinforcement learning and may become practically advantageous as deployment overheads diminish. This positions QRL as a promising paradigm for dynamic decision-making in complex, high-dimensional, and non-stationary environments such as financial markets. The complete codebase is released as open source at: this https URL

**arXiv ID:** 2601.18811
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
