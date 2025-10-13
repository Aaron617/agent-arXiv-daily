# Agent arXiv Daily

**Last Updated:** 2025-10-13 02:47:44

**Total Papers:** 82

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>Multimodal Policy Internalization for Conversational Agents</strong> - Zhenhailong Wang, Jiateng Liu, Amin Fazel, Ritesh Sarkhel, Xing Fan, Xiang Li, Chenlei Guo, Heng Ji, Ruhi Sarikaya - [[pdf]](https://arxiv.org/pdf/2510.09474)</summary>

**Abstract:** Modern conversational agents like ChatGPT and Alexa+ rely on predefined policies specifying metadata, response styles, and tool-usage rules. As these LLM-based systems expand to support diverse business and user queries, such policies, often implemented as in-context prompts, are becoming increasingly complex and lengthy, making faithful adherence difficult and imposing large fixed computational costs. With the rise of multimodal agents, policies that govern visual and multimodal behaviors are critical but remain understudied. Prior prompt-compression work mainly shortens task templates and demonstrations, while existing policy-alignment studies focus only on text-based safety rules. We introduce Multimodal Policy Internalization (MPI), a new task that internalizes reasoning-intensive multimodal policies into model parameters, enabling stronger policy-following without including the policy during inference. MPI poses unique data and algorithmic challenges. We build two datasets spanning synthetic and real-world decision-making and tool-using tasks and propose TriMPI, a three-stage training framework. TriMPI first injects policy knowledge via continual pretraining, then performs supervised finetuning, and finally applies PolicyRollout, a GRPO-style reinforcement learning extension that augments rollouts with policy-aware responses for grounded exploration. TriMPI achieves notable gains in end-to-end accuracy, generalization, and robustness to forgetting. As the first work on multimodal policy internalization, we provide datasets, training recipes, and comprehensive evaluations to foster future research. Project page: this https URL.

**arXiv ID:** 2510.09474
</details>

<details>
<summary><strong>From Simulation to Strategy: Automating Personalized Interaction Planning for Conversational Agents</strong> - Wen-Yu Chang, Tzu-Hung Huang, Chih-Ho Chen, Yun-Nung Chen - [[pdf]](https://arxiv.org/pdf/2510.08621)</summary>

**Abstract:** Amid the rapid rise of agentic dialogue models, realistic user-simulator studies are essential for tuning effective conversation strategies. This work investigates a sales-oriented agent that adapts its dialogue based on user profiles spanning age, gender, and occupation. While age and gender influence overall performance, occupation produces the most pronounced differences in conversational intent. Leveraging this insight, we introduce a lightweight, occupation-conditioned strategy that guides the agent to prioritize intents aligned with user preferences, resulting in shorter and more successful dialogues. Our findings highlight the importance of rich simulator profiles and demonstrate how simple persona-informed strategies can enhance the effectiveness of sales-oriented dialogue systems.

**arXiv ID:** 2510.08621
</details>

<details>
<summary><strong>Augmenting Compliance-Guaranteed Customer Service Chatbots: Context-Aware Knowledge Expansion with Large Language Models</strong> - Mengze Hong, Chen Jason Zhang, Di Jiang, Yuanqin He - [[pdf]](https://arxiv.org/pdf/2410.12444)</summary>

**Abstract:** Retrieval-based chatbots leverage human-verified Q\&A knowledge to deliver accurate, verifiable responses, making them ideal for customer-centric applications where compliance with regulatory and operational standards is critical. To effectively handle diverse customer inquiries, augmenting the knowledge base with "similar questions" that retain semantic meaning while incorporating varied expressions is a cost-effective strategy. In this paper, we introduce the Similar Question Generation (SQG) task for LLM training and inference, proposing context-aware approaches to enable comprehensive semantic exploration and enhanced alignment with source question-answer relationships. We formulate optimization techniques for constructing in-context prompts and selecting an optimal subset of similar questions to expand chatbot knowledge under budget constraints. Both quantitative and human evaluations validate the effectiveness of these methods, achieving a 92% user satisfaction rate in a deployed chatbot system, reflecting an 18% improvement over the unaugmented baseline. These findings highlight the practical benefits of SQG and emphasize the potential of LLMs, not as direct chatbot interfaces, but in supporting non-generative systems for hallucination-free, compliance-guaranteed applications.

**arXiv ID:** 2410.12444
</details>

<details>
<summary><strong>Nav-EE: Navigation-Guided Early Exiting for Efficient Vision-Language Models in Autonomous Driving</strong> - Haibo Hu, Lianming Huang, Xinyu Wang, Yufei Cui, Shangyu Wu, Nan Guan, Chun Jason Xue - [[pdf]](https://arxiv.org/pdf/2510.01795)</summary>

**Abstract:** Vision-Language Models (VLMs) are increasingly applied in autonomous driving for unified perception and reasoning, but high inference latency hinders real-time deployment. Early-exit reduces latency by terminating inference at intermediate layers, yet its task-dependent nature limits generalization across diverse scenarios. We observe that this limitation aligns with autonomous driving: navigation systems can anticipate upcoming contexts (e.g., intersections, traffic lights), indicating which tasks will be required. We propose Nav-EE, a navigation-guided early-exit framework that precomputes task-specific exit layers offline and dynamically applies them online based on navigation priors. Experiments on CODA, Waymo, and BOSCH show that Nav-EE achieves accuracy comparable to full inference while reducing latency by up to 63.9%. Real-vehicle integration with Autoware Universe further demonstrates reduced inference latency (600ms to 300ms), supporting faster decision-making in complex scenarios. These results suggest that coupling navigation foresight with early-exit offers a viable path toward efficient deployment of large models in autonomous systems. Code and data are available at our anonymous repository: this https URL

**arXiv ID:** 2510.01795
</details>

<details>
<summary><strong>Beyond Words: Infusing Conversational Agents with Human-like Typing Behaviors</strong> - Jijie Zhou, Yuhan Hu - [[pdf]](https://arxiv.org/pdf/2510.08912)</summary>

**Abstract:** Recently, large language models have facilitated the emergence of highly intelligent conversational AI capable of engaging in human-like dialogues. However, a notable distinction lies in the fact that these AI models predominantly generate responses rapidly, often producing extensive content without emulating the thoughtful process characteristic of human cognition and typing. This paper presents a design aimed at simulating human-like typing behaviors, including patterns such as hesitation and self-editing, as well as a preliminary user experiment to understand whether and to what extent the agent with human-like typing behaviors could potentially affect conversational engagement and its trustworthiness. We've constructed an interactive platform featuring user-adjustable parameters, allowing users to personalize the AI's communication style and thus cultivate a more enriching and immersive conversational experience. Our user experiment, involving interactions with three types of agents - a baseline agent, one simulating hesitation, and another integrating both hesitation and self-editing behaviors - reveals a preference for the agent that incorporates both behaviors, suggesting an improvement in perceived naturalness and trustworthiness. Through the insights from our design process and both quantitative and qualitative feedback from user experiments, this paper contributes to the multimodal interaction design and user experience for conversational AI, advocating for a more human-like, engaging, and trustworthy communication paradigm.

**arXiv ID:** 2510.08912
</details>

<details>
<summary><strong>"I know it's not right, but that's what it said to do": Investigating Trust in AI Chatbots for Cybersecurity Policy</strong> - Brandon Lit, Edward Crowder, Daniel Vogel, Hassan Khan - [[pdf]](https://arxiv.org/pdf/2510.08917)</summary>

**Abstract:** AI chatbots are an emerging security attack vector, vulnerable to threats such as prompt injection, and rogue chatbot creation. When deployed in domains such as corporate security policy, they could be weaponized to deliver guidance that intentionally undermines system defenses. We investigate whether users can be tricked by a compromised AI chatbot in this scenario. A controlled study (N=15) asked participants to use a chatbot to complete security-related tasks. Without their knowledge, the chatbot was manipulated to give incorrect advice for some tasks. The results show how trust in AI chatbots is related to task familiarity, and confidence in their ownn judgment. Additionally, we discuss possible reasons why people do or do not trust AI chatbots in different scenarios.

**arXiv ID:** 2510.08917
</details>

<details>
<summary><strong>Convivial Conversational Agents -- shifting toward relationships</strong> - Rafael A. Calvo, Dorian Peters - [[pdf]](https://arxiv.org/pdf/2510.09516)</summary>

**Abstract:** Conversational AI (CAI) systems offer opportunities to scale service provision to unprecedented levels and governments and corporations are already beginning to deploy them across services. The economic argument is similar across domains: use CAI to automate the time-consuming conversations required for customer, client or patient support. Herein we draw on our work in dementia care to explore some of the challenges and opportunities for CAI, and how a new way of conceptualising these systems could help ensure essential aspects for human thriving are not lost in the process of automation.

**arXiv ID:** 2510.09516
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>Auto-scaling Continuous Memory for GUI Agent</strong> - Wenyi Wu, Kun Zhou, Ruoxin Yuan, Vivian Yu, Stephen Wang, Zhiting Hu, Biwei Huang - [[pdf]](https://arxiv.org/pdf/2510.09038)</summary>

**Abstract:** We study how to endow GUI agents with scalable memory that help generalize across unfamiliar interfaces and long-horizon tasks. Prior GUI agents compress past trajectories into text tokens, which balloons context length and misses decisive visual cues (e.g., exact widget size and position). We propose a continuous memory that encodes each GUI trajectory into a fixed-length sequence of continuous embeddings using the VLM itself as an encoder; these embeddings are plugged directly into the backbone's input layer, sharply reducing context cost while preserving fine-grained visual information. As memory size and retrieval depth increase, performance improves monotonically, unlike text memories that degrade with long prompts. To grow memory at low cost, we introduce an auto-scaling data flywheel that (i) discovers new environments via search, (ii) synthesizes tasks with an open-source VLM, (iii) rolls out trajectories with the agent, and (iv) verifies success with the same VLM. Using this pipeline, we collect 100k+ trajectories for about \$4000 and fine-tune only the memory encoder (LoRA on a Q-Former, 1.2\% parameters) with 1,500 samples. On real-world GUI benchmarks, our memory-augmented agent consistently improves success rates under long horizons and distribution shifts. Notably, Qwen-2.5-VL-7B + continuous memory achieves performance comparable to state-of-the-art closed-source models (e.g., GPT-4o, Claude-4).

**arXiv ID:** 2510.09038
</details>

<details>
<summary><strong>Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation</strong> - Spandan Garg, Ben Steenhoek, Yufan Huang - [[pdf]](https://arxiv.org/pdf/2510.08996)</summary>

**Abstract:** Current benchmarks for evaluating software engineering agents, such as SWE-Bench Verified, are predominantly derived from GitHub issues and fail to accurately reflect how developers interact with chat-based coding assistants in integrated development environments (IDEs). We posit that this mismatch leads to a systematic overestimation of agent's capabilities in real-world scenarios, especially bug fixing. We introduce a novel benchmarking framework that transforms existing formal benchmarks into realistic user queries through systematic analysis of developer interaction patterns with chat-based agents. Our methodology is flexible and can be easily extended to existing benchmarks. In this paper, we apply our testing framework to SWE-Bench Verified, the TypeScript subset of Multi-SWE-Bench and a private benchmark, SWE-Bench C# and transform formal GitHub issue descriptions into realistic user-style queries based on telemetry analysis of a popular chat-based agent interactions. Our findings reveal that existing benchmarks significantly overestimate agent capabilities for some models by >50% over baseline performance for public benchmarks and ~10-16% for our internal benchmark. This work establishes a new paradigm for evaluating interactive chat-based software engineering agents through benchmark mutation techniques.

**arXiv ID:** 2510.08996
</details>

<details>
<summary><strong>IG-MCTS: Human-in-the-Loop Cooperative Navigation under Incomplete Information</strong> - Shenghui Chen, Ruihan Zhao, Sandeep Chinchali, Ufuk Topcu - [[pdf]](https://arxiv.org/pdf/2502.01857)</summary>

**Abstract:** Human-robot cooperative navigation is challenging under incomplete information. We introduce CoNav-Maze, a simulated environment where a robot navigates with local perception while a human operator provides guidance based on an inaccurate map. The robot can share its onboard camera views to help the operator refine their understanding of the environment. To enable efficient cooperation, we propose Information Gain Monte Carlo Tree Search (IG-MCTS), an online planning algorithm that jointly optimizes autonomous movement and informative communication. IG-MCTS leverages a learned Neural Human Perception Model (NHPM) -- trained on a crowdsourced mapping dataset -- to predict how the human's internal map evolves as new observations are shared. User studies show that IG-MCTS significantly reduces communication demands and yields eye-tracking metrics indicative of lower cognitive load, while maintaining task performance comparable to teleoperation and instruction-following baselines. Finally, we illustrate generalization beyond discrete mazes through a continuous-space waterway navigation setting, in which NHPM benefits from deeper encoder-decoder architectures and IG-MCTS leverages a dynamically constructed Voronoi-partitioned traversability graph.

**arXiv ID:** 2502.01857
</details>

<details>
<summary><strong>AD-EE: Early Exiting for Fast and Reliable Vision-Language Models in Autonomous Driving</strong> - Lianming Huang, Haibo Hu, Yufei Cui, Jiacheng Zuo, Shangyu Wu, Nan Guan, Chun Jason Xue - [[pdf]](https://arxiv.org/pdf/2506.05404)</summary>

**Abstract:** With the rapid advancement of autonomous driving, deploying Vision-Language Models (VLMs) to enhance perception and decision-making has become increasingly common. However, the real-time application of VLMs is hindered by high latency and computational overhead, limiting their effectiveness in time-critical driving scenarios. This challenge is particularly evident when VLMs exhibit over-inference, continuing to process unnecessary layers even after confident predictions have been reached. To address this inefficiency, we propose AD-EE, an Early Exit framework that incorporates domain characteristics of autonomous driving and leverages causal inference to identify optimal exit layers. We evaluate our method on large-scale real-world autonomous driving datasets, including Waymo and the corner-case-focused CODA, as well as on a real vehicle running the Autoware Universe platform. Extensive experiments across multiple VLMs show that our method significantly reduces latency, with maximum improvements reaching up to 57.58%, and enhances object detection accuracy, with maximum gains of up to 44%.

**arXiv ID:** 2506.05404
</details>

<details>
<summary><strong>Exploiting Web Search Tools of AI Agents for Data Exfiltration</strong> - Dennis Rall, Bernhard Bauer, Mohit Mittal, Thomas Fraunholz - [[pdf]](https://arxiv.org/pdf/2510.09093)</summary>

**Abstract:** Large language models (LLMs) are now routinely used to autonomously execute complex tasks, from natural language processing to dynamic workflows like web searches. The usage of tool-calling and Retrieval Augmented Generation (RAG) allows LLMs to process and retrieve sensitive corporate data, amplifying both their functionality and vulnerability to abuse. As LLMs increasingly interact with external data sources, indirect prompt injection emerges as a critical and evolving attack vector, enabling adversaries to exploit models through manipulated inputs. Through a systematic evaluation of indirect prompt injection attacks across diverse models, we analyze how susceptible current LLMs are to such attacks, which parameters, including model size and manufacturer, specific implementations, shape their vulnerability, and which attack methods remain most effective. Our results reveal that even well-known attack patterns continue to succeed, exposing persistent weaknesses in model defenses. To address these vulnerabilities, we emphasize the need for strengthened training procedures to enhance inherent resilience, a centralized database of known attack vectors to enable proactive defense, and a unified testing framework to ensure continuous security validation. These steps are essential to push developers toward integrating security into the core design of LLMs, as our findings show that current models still fail to mitigate long-standing threats.

**arXiv ID:** 2510.09093
</details>

<details>
<summary><strong>Haystack Engineering: Context Engineering for Heterogeneous and Agentic Long-Context Evaluation</strong> - Mufei Li, Dongqi Fu, Limei Wang, Si Zhang, Hanqing Zeng, Kaan Sancak, Ruizhong Qiu, Haoyu Wang, Xiaoxin He, Xavier Bresson, Yinglong Xia, Chonglin Sun, Pan Li - [[pdf]](https://arxiv.org/pdf/2510.07414)</summary>

**Abstract:** Modern long-context large language models (LLMs) perform well on synthetic "needle-in-a-haystack" (NIAH) benchmarks, but such tests overlook how noisy contexts arise from biased retrieval and agentic workflows. We argue that haystack engineering is necessary to construct noisy long contexts that faithfully capture key real-world factors -- distraction from heterogeneous biased retrievers and cascading errors in agentic workflows -- to test models' long-context robustness. We instantiate it through HaystackCraft, a new NIAH benchmark built on the full English Wikipedia hyperlink network with multi-hop questions. HaystackCraft evaluates how heterogeneous retrieval strategies (e.g., sparse, dense, hybrid, and graph-based) affect distractor composition, haystack ordering, and downstream LLM performance. HaystackCraft further extends NIAH to dynamic, LLM-dependent settings that simulate agentic operations, where models refine queries, reflect on their past reasonings, and decide when to stop. Experiments with 15 long-context models show that (1) while stronger dense retrievers can introduce more challenging distractors, graph-based reranking simultaneously improves retrieval effectiveness and mitigates more harmful distractors; (2) in agentic tests, even advanced models like Gemini 2.5 Pro and GPT-5 suffer cascading failures from self-generated distractors or struggle to perform early stops. These results highlight persistent challenges in agentic long-context reasoning and establish HaystackCraft as a valuable testbed for future progress.

**arXiv ID:** 2510.07414
</details>

<details>
<summary><strong>Online IMU-odometer Calibration using GNSS Measurements for Autonomous Ground Vehicle Localization</strong> - Baoshan Song, Xiao Xia, Penggao Yan, Yihan Zhong, Weisong Wen, Li-Ta Hsu - [[pdf]](https://arxiv.org/pdf/2510.08880)</summary>

**Abstract:** Accurate calibration of intrinsic (odometer scaling factors) and extrinsic parameters (IMU-odometer translation and rotation) is essential for autonomous ground vehicle localization. Existing GNSS-aided approaches often rely on positioning results or raw measurements without ambiguity resolution, and their observability properties remain underexplored. This paper proposes a tightly coupled online calibration method that fuses IMU, odometer, and raw GNSS measurements (pseudo-range, carrier-phase, and Doppler) within an extendable factor graph optimization (FGO) framework, incorporating outlier mitigation and ambiguity resolution. Observability analysis reveals that two horizontal translation and three rotation parameters are observable under general motion, while vertical translation remains unobservable. Simulation and real-world experiments demonstrate superior calibration and localization performance over state-of-the-art loosely coupled methods. Specifically, the IMU-odometer positioning using our calibrated parameters achieves the absolute maximum error of 17.75 m while the one of LC method is 61.51 m, achieving up to 71.14 percent improvement. To foster further research, we also release the first open-source dataset that combines IMU, 2D odometer, and raw GNSS measurements from both rover and base stations.

**arXiv ID:** 2510.08880
</details>

<details>
<summary><strong>BEAR: Benchmarking and Enhancing Multimodal Language Models for Atomic Embodied Capabilities</strong> - Yu Qi, Haibo Zhao, Ziyu Guo, Siyuan Ma, Ziyan Chen, Yaokun Han, Renrui Zhang, Zitiantao Lin, Shiji Xin, Yijian Huang, Kai Cheng, Peiheng Wang, Jiazheng Liu, Jiayi Zhang, Yizhe Zhu, Wenqing Wang, Yiran Qin, Xupeng Zhu, Haojie Huang, Lawson L.S. Wong - [[pdf]](https://arxiv.org/pdf/2510.08759)</summary>

**Abstract:** Embodied capabilities refer to a suite of fundamental abilities for an agent to perceive, comprehend, and interact with the physical world. While multimodal large language models (MLLMs) show promise as embodied agents, a thorough and systematic evaluation of their embodied capabilities remains underexplored, as existing benchmarks primarily focus on specific domains such as planning or spatial understanding. To bridge this gap, we introduce BEAR, a comprehensive and fine-grained benchmark that evaluates MLLMs on atomic embodied capabilities. BEAR comprises 4,469 interleaved image-video-text entries across 14 domains in 6 categories, including tasks from low-level pointing, trajectory understanding, spatial reasoning, to high-level planning. Extensive evaluation results of 20 representative MLLMs reveal their persistent limitations across all domains of embodied capabilities. To tackle the shortfall, we propose BEAR-Agent, a multimodal conversable agent that integrates pretrained vision models to strengthen MLLM perception, 3D understanding, and planning capabilities. It substantially enhances MLLM performance across diverse embodied capabilities on BEAR, yielding a 9.12% absolute gain and a relative improvement of 17.5% on GPT-5. Furthermore, our experiments indicate that improving MLLM embodied capabilities can benefit embodied tasks in simulated environments. Project website: this https URL

**arXiv ID:** 2510.08759
</details>

<details>
<summary><strong>A Multimodal Depth-Aware Method For Embodied Reference Understanding</strong> - Fevziye Irem Eyiokur, Dogucan Yaman, Hazım Kemal Ekenel, Alexander Waibel - [[pdf]](https://arxiv.org/pdf/2510.08278)</summary>

**Abstract:** Embodied Reference Understanding requires identifying a target object in a visual scene based on both language instructions and pointing cues. While prior works have shown progress in open-vocabulary object detection, they often fail in ambiguous scenarios where multiple candidate objects exist in the scene. To address these challenges, we propose a novel ERU framework that jointly leverages LLM-based data augmentation, depth-map modality, and a depth-aware decision module. This design enables robust integration of linguistic and embodied cues, improving disambiguation in complex or cluttered environments. Experimental results on two datasets demonstrate that our approach significantly outperforms existing baselines, achieving more accurate and reliable referent detection.

**arXiv ID:** 2510.08278
</details>

<details>
<summary><strong>Why AI Agents Still Need You: Findings from Developer-Agent Collaborations in the Wild</strong> - Aayush Kumar, Yasharth Bajpai, Sumit Gulwani, Gustavo Soares, Emerson Murphy-Hill - [[pdf]](https://arxiv.org/pdf/2506.12347)</summary>

**Abstract:** Software Engineering Agents (SWE agents) can autonomously perform development tasks on benchmarks like SWE Bench, but still face challenges when tackling complex and ambiguous real-world tasks. Consequently, SWE agents are often designed to allow interactivity with developers, enabling collaborative problem-solving. To understand how developers collaborate with SWE agents and the barriers they face in such interactions, we observed 19 developers using an in-IDE agent to resolve 33 open issues in repositories to which they had previously contributed. Participants successfully resolved about half of these issues, with those solving issues incrementally having greater success than those using a one-shot approach. Participants who actively collaborated with the agent and iterated on its outputs were also more successful, though they faced challenges in trusting the agent's responses and collaborating on debugging and testing. Our findings suggest that to facilitate successful collaborations, both SWE agents and developers should actively contribute to tasks throughout all stages of the software development process. SWE agents can enable this by challenging and engaging in discussions with developers, rather than being conclusive or sycophantic.

**arXiv ID:** 2506.12347
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>Fundamentals of Building Autonomous LLM Agents</strong> - Victor de Lamo Castrillo, Habtom Kahsay Gidey, Alexander Lenz, Alois Knoll - [[pdf]](https://arxiv.org/pdf/2510.09244)</summary>

**Abstract:** This paper reviews the architecture and implementation methods of agents powered by large language models (LLMs). Motivated by the limitations of traditional LLMs in real-world tasks, the research aims to explore patterns to develop "agentic" LLMs that can automate complex tasks and bridge the performance gap with human capabilities. Key components include a perception system that converts environmental percepts into meaningful representations; a reasoning system that formulates plans, adapts to feedback, and evaluates actions through different techniques like Chain-of-Thought and Tree-of-Thought; a memory system that retains knowledge through both short-term and long-term mechanisms; and an execution system that translates internal decisions into concrete actions. This paper shows how integrating these systems leads to more capable and generalized software bots that mimic human cognitive processes for autonomous and intelligent behavior.

**arXiv ID:** 2510.09244
</details>

<details>
<summary><strong>Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools</strong> - Ha Min Son, Huan Ren, Xin Liu, Zhe Zhao - [[pdf]](https://arxiv.org/pdf/2510.08640)</summary>

**Abstract:** Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution.

**arXiv ID:** 2510.08640
</details>

<details>
<summary><strong>CommandSans: Securing AI Agents with Surgical Precision Prompt Sanitization</strong> - Debeshee Das, Luca Beurer-Kellner, Marc Fischer, Maximilian Baader - [[pdf]](https://arxiv.org/pdf/2510.08829)</summary>

**Abstract:** The increasing adoption of LLM agents with access to numerous tools and sensitive data significantly widens the attack surface for indirect prompt injections. Due to the context-dependent nature of attacks, however, current defenses are often ill-calibrated as they cannot reliably differentiate malicious and benign instructions, leading to high false positive rates that prevent their real-world adoption. To address this, we present a novel approach inspired by the fundamental principle of computer security: data should not contain executable instructions. Instead of sample-level classification, we propose a token-level sanitization process, which surgically removes any instructions directed at AI systems from tool outputs, capturing malicious instructions as a byproduct. In contrast to existing safety classifiers, this approach is non-blocking, does not require calibration, and is agnostic to the context of tool outputs. Further, we can train such token-level predictors with readily available instruction-tuning data only, and don't have to rely on unrealistic prompt injection examples from challenges or of other synthetic origin. In our experiments, we find that this approach generalizes well across a wide range of attacks and benchmarks like AgentDojo, BIPIA, InjecAgent, ASB and SEP, achieving a 7-10x reduction of attack success rate (ASR) (34% to 3% on AgentDojo), without impairing agent utility in both benign and malicious settings.

**arXiv ID:** 2510.08829
</details>

<details>
<summary><strong>Medchain: Bridging the Gap Between LLM Agents and Clinical Practice with Interactive Sequence</strong> - Jie Liu, Wenxuan Wang, Zizhan Ma, Guolin Huang, Yihang SU, Kao-Jung Chang, Wenting Chen, Haoliang Li, Linlin Shen, Michael Lyu - [[pdf]](https://arxiv.org/pdf/2412.01605)</summary>

**Abstract:** Clinical decision making (CDM) is a complex, dynamic process crucial to healthcare delivery, yet it remains a significant challenge for artificial intelligence systems. While Large Language Model (LLM)-based agents have been tested on general medical knowledge using licensing exams and knowledge question-answering tasks, their performance in the CDM in real-world scenarios is limited due to the lack of comprehensive testing datasets that mirror actual medical practice. To address this gap, we present MedChain, a dataset of 12,163 clinical cases that covers five key stages of clinical workflow. MedChain distinguishes itself from existing benchmarks with three key features of real-world clinical practice: personalization, interactivity, and sequentiality. Further, to tackle real-world CDM challenges, we also propose MedChain-Agent, an AI system that integrates a feedback mechanism and a MCase-RAG module to learn from previous cases and adapt its responses. MedChain-Agent demonstrates remarkable adaptability in gathering information dynamically and handling sequential clinical tasks, significantly outperforming existing approaches.

**arXiv ID:** 2412.01605
</details>

<details>
<summary><strong>OrcaLoca: An LLM Agent Framework for Software Issue Localization</strong> - Zhongming Yu, Hejia Zhang, Yujie Zhao, Hanxian Huang, Matrix Yao, Ke Ding, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2502.00350)</summary>

**Abstract:** Recent developments in Large Language Model (LLM) agents are revolutionizing Autonomous Software Engineering (ASE), enabling automated coding, problem fixes, and feature improvements. However, localization -- precisely identifying software problems by navigating to relevant code sections -- remains a significant challenge. Current approaches often yield suboptimal results due to a lack of effective integration between LLM agents and precise code search mechanisms. This paper introduces OrcaLoca, an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning. Experimental results demonstrate that OrcaLoca becomes the new open-source state-of-the-art (SOTA) in function match rate (65.33%) on SWE-bench Lite. It also improves the final resolved rate of an open-source framework by 6.33 percentage points through its patch generation integration.

**arXiv ID:** 2502.00350
</details>

<details>
<summary><strong>Large Language Model Agent for Modular Task Execution in Drug Discovery</strong> - Janghoon Ock, Radheesh Sharma Meda, Srivathsan Badrinarayanan, Neha S. Aluru, Achuth Chandrasekhar, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2507.02925)</summary>

**Abstract:** We present a modular framework powered by large language models (LLMs) that automates and streamlines key tasks across the early-stage computational drug discovery pipeline. By combining LLM reasoning with domain-specific tools, the framework performs biomedical data retrieval, domain-specific question answering, molecular generation, property prediction, property-aware molecular refinement, and 3D protein-ligand structure generation. In a case study targeting BCL-2 in lymphocytic leukemia, the agent autonomously retrieved relevant biomolecular information, including FASTA sequences, SMILES representations, and literature, and answered mechanistic questions with improved contextual accuracy compared to standard LLMs. It then generated chemically diverse seed molecules and predicted 67 ADMET-related properties, which guided iterative molecular refinement. Across two refinement rounds, the number of molecules with QED > 0.6 increased from 34 to 55. The number of molecules satisfying empirical drug-likeness filters also rose; for example, compliance with the Ghose filter increased from 32 to 55 within a pool of 100 molecules. The framework also employed Boltz-2 to generate 3D protein-ligand complexes and provide rapid binding affinity estimates for candidate compounds. These results demonstrate that the approach effectively supports molecular screening, prioritization, and structure evaluation. Its modular design enables flexible integration of evolving tools and models, providing a scalable foundation for AI-assisted therapeutic discovery.

**arXiv ID:** 2507.02925
</details>

<details>
<summary><strong>Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers</strong> - Tuan Nguyen, Long Tran-Thanh - [[pdf]](https://arxiv.org/pdf/2510.09330)</summary>

**Abstract:** Ensuring that large language models (LLMs) comply with safety requirements is a central challenge in AI deployment. Existing alignment approaches primarily operate during training, such as through fine-tuning or reinforcement learning from human feedback, but these methods are costly and inflexible, requiring retraining whenever new requirements arise. Recent efforts toward inference-time alignment mitigate some of these limitations but still assume access to model internals, which is impractical, and not suitable for third party stakeholders who do not have access to the models. In this work, we propose a model-independent, black-box framework for safety alignment that does not require retraining or access to the underlying LLM architecture. As a proof of concept, we address the problem of trading off between generating safe but uninformative answers versus helpful yet potentially risky ones. We formulate this dilemma as a two-player zero-sum game whose minimax equilibrium captures the optimal balance between safety and helpfulness. LLM agents operationalize this framework by leveraging a linear programming solver at inference time to compute equilibrium strategies. Our results demonstrate the feasibility of black-box safety alignment, offering a scalable and accessible pathway for stakeholders, including smaller organizations and entities in resource-constrained settings, to enforce safety across rapidly evolving LLM ecosystems.

**arXiv ID:** 2510.09330
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>COMPASS: Enhancing Agent Long-Horizon Reasoning with Evolving Context</strong> - Guangya Wan, Mingyang Ling, Xiaoqi Ren, Rujun Han, Sheng Li, Zizhao Zhang - [[pdf]](https://arxiv.org/pdf/2510.08790)</summary>

**Abstract:** Long-horizon tasks that require sustained reasoning and multiple tool interactions remain challenging for LLM agents: small errors compound across steps, and even state-of-the-art models often hallucinate or lose coherence. We identify context management as the central bottleneck -- extended histories cause agents to overlook critical evidence or become distracted by irrelevant information, thus failing to replan or reflect from previous mistakes. To address this, we propose COMPASS (Context-Organized Multi-Agent Planning and Strategy System), a lightweight hierarchical framework that separates tactical execution, strategic oversight, and context organization into three specialized components: (1) a Main Agent that performs reasoning and tool use, (2) a Meta-Thinker that monitors progress and issues strategic interventions, and (3) a Context Manager that maintains concise, relevant progress briefs for different reasoning stages. Across three challenging benchmarks -- GAIA, BrowseComp, and Humanity's Last Exam -- COMPASS improves accuracy by up to 20% relative to both single- and multi-agent baselines. We further introduce a test-time scaling extension that elevates performance to match established DeepResearch agents, and a post-training pipeline that delegates context management to smaller models for enhanced efficiency.

**arXiv ID:** 2510.08790
</details>

<details>
<summary><strong>AgenticAD: A Specialized Multiagent System Framework for Holistic Alzheimer Disease Management</strong> - Adib Bazgir, Amir Habibdoust, Xing Song, Yuwen Zhang - [[pdf]](https://arxiv.org/pdf/2510.08578)</summary>

**Abstract:** Alzheimer's disease (AD) presents a complex, multifaceted challenge to patients, caregivers, and the healthcare system, necessitating integrated and dynamic support solutions. While artificial intelligence (AI) offers promising avenues for intervention, current applications are often siloed, addressing singular aspects of the disease such as diagnostics or caregiver support without systemic integration. This paper proposes a novel methodological framework for a comprehensive, multi-agent system (MAS) designed for holistic Alzheimer's disease management. The objective is to detail the architecture of a collaborative ecosystem of specialized AI agents, each engineered to address a distinct challenge in the AD care continuum, from caregiver support and multimodal data analysis to automated research and clinical data interpretation. The proposed framework is composed of eight specialized, interoperable agents. These agents are categorized by function: (1) Caregiver and Patient Support, (2) Data Analysis and Research, and (3) Advanced Multimodal Workflows. The methodology details the technical architecture of each agent, leveraging a suite of advanced technologies including large language models (LLMs) such as GPT-4o and Gemini, multi-agent orchestration frameworks, Retrieval-Augmented Generation (RAG) for evidence-grounded responses, and specialized tools for web scraping, multimodal data processing, and in-memory database querying. This paper presents a detailed architectural blueprint for an integrated AI ecosystem for AD care. By moving beyond single-purpose tools to a collaborative, multi-agent paradigm, this framework establishes a foundation for developing more adaptive, personalized, and proactive solutions. This methodological approach aims to pave the way for future systems capable of synthesizing diverse data streams to improve patient outcomes and reduce caregiver burden.

**arXiv ID:** 2510.08578
</details>

<details>
<summary><strong>Toward a Safer Web: Multilingual Multi-Agent LLMs for Mitigating Adversarial Misinformation Attacks</strong> - Nouar Aldahoul, Yasir Zaki - [[pdf]](https://arxiv.org/pdf/2510.08605)</summary>

**Abstract:** The rapid spread of misinformation on digital platforms threatens public discourse, emotional stability, and decision-making. While prior work has explored various adversarial attacks in misinformation detection, the specific transformations examined in this paper have not been systematically studied. In particular, we investigate language-switching across English, French, Spanish, Arabic, Hindi, and Chinese, followed by translation. We also study query length inflation preceding summarization and structural reformatting into multiple-choice questions. In this paper, we present a multilingual, multi-agent large language model framework with retrieval-augmented generation that can be deployed as a web plugin into online platforms. Our work underscores the importance of AI-driven misinformation detection in safeguarding online factual integrity against diverse attacks, while showcasing the feasibility of plugin-based deployment for real-world web applications.

**arXiv ID:** 2510.08605
</details>

<details>
<summary><strong>RA-Gen: A Controllable Code Generation Framework Using ReAct for Multi-Agent Task Execution</strong> - Aofan Liu, Haoxuan Li, Bin Wang, Ao Yang, Hui Li - [[pdf]](https://arxiv.org/pdf/2510.08665)</summary>

**Abstract:** Code generation models based on large language models (LLMs) have gained wide adoption, but challenges remain in ensuring safety, accuracy, and controllability, especially for complex tasks. Existing methods often lack dynamic integration of external tools, transparent reasoning, and user control over safety. To address these issues, we propose a controllable code generation framework utilizing the ReAct paradigm for multi-agent task execution. This framework is a multi-agent system designed to enable efficient, precise, and interpretable code generation through dynamic interactions between LLMs and external resources. The framework adopts a collaborative architecture comprising four specialized agents: a Planner for task decomposition, a Searcher that leverages the ReAct framework for reasoning and tool integration, a CodeGen agent for accurate code generation, and an Extractor for structured data retrieval. The ReAct-based Searcher alternates between generating reasoning traces and executing actions, facilitating seamless integration of internal knowledge with external tools (such as search engines) to enhance accuracy and user control. Experimental results show the framework's effectiveness across multiple languages, achieving a 94.8% security rate on the SVEN dataset with CodeQL, outperforming existing approaches. Its transparent reasoning process fosters user trust and improves controllability.

**arXiv ID:** 2510.08665
</details>

<details>
<summary><strong>Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid Execution Strategy</strong> - Bharath Muppasani, Ritirupa Dey, Biplav Srivastava, Vignesh Narayanan - [[pdf]](https://arxiv.org/pdf/2510.09469)</summary>

**Abstract:** Multi-agent pathfinding (MAPF) remains a critical problem in robotics and autonomous systems, where agents must navigate shared spaces efficiently while avoiding conflicts. Traditional centralized algorithms that have global information, such as Conflict-Based Search (CBS), provide high-quality solutions but become computationally expensive in large-scale scenarios due to the combinatorial explosion of conflicts that need resolution. Conversely, distributed approaches that have local information, particularly learning-based methods, offer better scalability by operating with relaxed information availability, yet often at the cost of solution quality. To address these limitations, we propose a hybrid framework that combines decentralized path planning with a lightweight centralized coordinator. Our framework leverages reinforcement learning (RL) for decentralized planning, enabling agents to adapt their planning based on minimal, targeted alerts--such as static conflict-cell flags or brief conflict tracks--that are dynamically shared information from the central coordinator for effective conflict resolution. We empirically study the effect of the information available to an agent on its planning performance. Our approach reduces the inter-agent information sharing compared to fully centralized and distributed methods, while still consistently finding feasible, collision-free solutions--even in large-scale scenarios having higher agent counts.

**arXiv ID:** 2510.09469
</details>

<details>
<summary><strong>AniME: Adaptive Multi-Agent Planning for Long Animation Generation</strong> - Lisai Zhang, Baohan Xu, Siqian Yang, Mingyu Yin, Jing Liu, Chao Xu, Siqi Wang, Yidi Wu, Yuxin Hong, Zihao Zhang, Yanzhang Liang, Yudong Jiang - [[pdf]](https://arxiv.org/pdf/2508.18781)</summary>

**Abstract:** We present AniME, a director-oriented multi-agent system for automated long-form anime production, covering the full workflow from a story to the final video. The director agent keeps a global memory for the whole workflow, and coordinates several downstream specialized agents. By integrating customized Model Context Protocol (MCP) with downstream model instruction, the specialized agent adaptively selects control conditions for diverse sub-tasks. AniME produces cinematic animation with consistent characters and synchronized audio visual elements, offering a scalable solution for AI-driven anime creation.

**arXiv ID:** 2508.18781
</details>

<details>
<summary><strong>Reimagining Agent-based Modeling with Large Language Model Agents via Shachi</strong> - So Kuroki, Yingtao Tian, Kou Misaki, Takashi Ikegami, Takuya Akiba, Yujin Tang - [[pdf]](https://arxiv.org/pdf/2509.21862)</summary>

**Abstract:** The study of emergent behaviors in large language model (LLM)-driven multi-agent systems is a critical research challenge, yet progress is limited by a lack of principled methodologies for controlled experimentation. To address this, we introduce Shachi, a formal methodology and modular framework that decomposes an agent's policy into core cognitive components: Configuration for intrinsic traits, Memory for contextual persistence, and Tools for expanded capabilities, all orchestrated by an LLM reasoning engine. This principled architecture moves beyond brittle, ad-hoc agent designs and enables the systematic analysis of how specific architectural choices influence collective behavior. We validate our methodology on a comprehensive 10-task benchmark and demonstrate its power through novel scientific inquiries. Critically, we establish the external validity of our approach by modeling a real-world U.S. tariff shock, showing that agent behaviors align with observed market reactions only when their cognitive architecture is appropriately configured with memory and tools. Our work provides a rigorous, open-source foundation for building and evaluating LLM agents, aimed at fostering more cumulative and scientifically grounded research.

**arXiv ID:** 2509.21862
</details>

<details>
<summary><strong>SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation</strong> - Minh-Anh Nguye, Minh-Duc Nguyen, Ha Lan N.T., Kieu Hai Dang, Nguyen Tien Dong, Dung D. Le - [[pdf]](https://arxiv.org/pdf/2510.07733)</summary>

**Abstract:** Large language models (LLMs) are increasingly adopted for automating survey paper generation \cite{wang2406autosurvey, liang2025surveyx, yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing approaches typically extract content from a large collection of related papers and prompt LLMs to summarize them directly. However, such methods often overlook the structural relationships among papers, resulting in generated surveys that lack a coherent taxonomy and a deeper contextual understanding of research progress. To address these shortcomings, we propose \textbf{SurveyG}, an LLM-based agent framework that integrates \textit{hierarchical citation graph}, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness between their contents, thereby embedding structural and contextual knowledge into the survey generation process. The graph is organized into three layers: \textbf{Foundation}, \textbf{Development}, and \textbf{Frontier}, to capture the evolution of research from seminal works to incremental advances and emerging directions. By combining horizontal search within layers and vertical depth traversal across layers, the agent produces multi-level summaries, which are consolidated into a structured survey outline. A multi-agent validation stage then ensures consistency, coverage, and factual accuracy in generating the final survey. Experiments, including evaluations by human experts and LLM-as-a-judge, demonstrate that SurveyG outperforms state-of-the-art frameworks, producing surveys that are more comprehensive and better structured to the underlying knowledge taxonomy of a field.

**arXiv ID:** 2510.07733
</details>

<details>
<summary><strong>DDO: Dual-Decision Optimization for LLM-Based Medical Consultation via Multi-Agent Collaboration</strong> - Zhihao Jia, Mingyi Jia, Junwen Duan, Jianxin Wang - [[pdf]](https://arxiv.org/pdf/2505.18630)</summary>

**Abstract:** Large Language Models (LLMs) demonstrate strong generalization and reasoning abilities, making them well-suited for complex decision-making tasks such as medical consultation (MC). However, existing LLM-based methods often fail to capture the dual nature of MC, which entails two distinct sub-tasks: symptom inquiry, a sequential decision-making process, and disease diagnosis, a classification problem. This mismatch often results in ineffective symptom inquiry and unreliable disease diagnosis. To address this, we propose \textbf{DDO}, a novel LLM-based framework that performs \textbf{D}ual-\textbf{D}ecision \textbf{O}ptimization by decoupling the two sub-tasks and optimizing them with distinct objectives through a collaborative multi-agent workflow. Experiments on three real-world MC datasets show that DDO consistently outperforms existing LLM-based approaches and achieves competitive performance with state-of-the-art generation-based methods, demonstrating its effectiveness in the MC task. The code is available at this https URL.

**arXiv ID:** 2505.18630
</details>

<details>
<summary><strong>Anemoi: A Semi-Centralized Multi-agent System Based on Agent-to-Agent Communication MCP server from Coral Protocol</strong> - Xinxing Ren, Caelum Forder, Qianbo Zang, Ahsen Tahir, Roman J. Georgio, Suman Deb, Peter Carroll, Önder Gürcan, Zekun Guo - [[pdf]](https://arxiv.org/pdf/2508.17068)</summary>

**Abstract:** Recent advances in generalist multi-agent systems (MAS) have largely followed a context-engineering plus centralized paradigm, where a planner agent coordinates multiple worker agents through unidirectional prompt passing. While effective under strong planner models, this design suffers from two critical limitations: (1) strong dependency on the planner's capability, which leads to degraded performance when a smaller LLM powers the planner; and (2) limited inter-agent communication, where collaboration relies on prompt concatenation rather than genuine refinement through structured discussions. To address these challenges, we propose Anemoi, a semi-centralized MAS built on the Agent-to-Agent (A2A) communication MCP server from Coral Protocol. Unlike traditional designs, Anemoi enables structured and direct inter-agent collaboration, allowing all agents to monitor progress, assess results, identify bottlenecks, and propose refinements in real time. This paradigm reduces reliance on a single planner, supports adaptive plan updates, and minimizes redundant context passing, resulting in more scalable execution. Evaluated on the GAIA benchmark, Anemoi achieved 52.73% accuracy with a small LLM (GPT-4.1-mini) as the planner, surpassing the strongest open-source baseline OWL (43.63%) by +9.09% under identical LLM settings. Our implementation is publicly available at this https URL.

**arXiv ID:** 2508.17068
</details>

<details>
<summary><strong>Aegis: Automated Error Generation and Attribution for Multi-Agent Systems</strong> - Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, Xue Feng - [[pdf]](https://arxiv.org/pdf/2509.14295)</summary>

**Abstract:** Large language model based multi-agent systems (MAS) have unlocked significant advancements in tackling complex problems, but their increasing capability introduces a structural fragility that makes them difficult to debug. A key obstacle to improving their reliability is the severe scarcity of large-scale, diverse datasets for error attribution, as existing resources rely on costly and unscalable manual annotation. To address this bottleneck, we introduce Aegis, a novel framework for Automated error generation and attribution for multi-agent systems. Aegis constructs a large dataset of 9,533 trajectories with annotated faulty agents and error modes, covering diverse MAS architectures and task domains. This is achieved using a LLM-based manipulator that can adaptively inject context-aware errors into successful execution trajectories. Leveraging fine-grained labels and the structured arrangement of positive-negative sample pairs, Aegis supports three different learning paradigms: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. We develop learning methods for each paradigm. Comprehensive experiments show that trained models consistently achieve substantial improvements in error attribution. Notably, several of our fine-tuned LLMs demonstrate performance competitive with or superior to proprietary models an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at this https URL.

**arXiv ID:** 2509.14295
</details>

<details>
<summary><strong>MOSAIC: Multi-agent Orchestration for Task-Intelligent Scientific Coding</strong> - Siddeshwar Raghavan, Tanwi Mallick - [[pdf]](https://arxiv.org/pdf/2510.08804)</summary>

**Abstract:** We present MOSAIC, a multi-agent Large Language Model (LLM) framework for solving challenging scientific coding tasks. Unlike general-purpose coding, scientific workflows require algorithms that are rigorous, interconnected with deep domain knowledge, and incorporate domain-specific reasoning, as well as algorithm iteration without requiring I/O test cases. Many scientific problems also require a sequence of subproblems to be solved, leading to the final desired result. MOSAIC is designed as a training-free framework with specially designed agents to self-reflect, create the rationale, code, and debug within a student-teacher paradigm to address the challenges of scientific code generation. This design facilitates stepwise problem decomposition, targeted error correction, and, when combined with our Consolidated Context Window (CCW), mitigates LLM hallucinations when solving complex scientific tasks involving chained subproblems. We evaluate MOSAIC on scientific coding benchmarks and demonstrate that our specialized agentic framework outperforms existing approaches in terms of accuracy, robustness, and interpretability.

**arXiv ID:** 2510.08804
</details>

<details>
<summary><strong>MASA: LLM-Driven Multi-Agent Systems for Autoformalization</strong> - Lan Zhang, Marco Valentino, André Freitas - [[pdf]](https://arxiv.org/pdf/2510.08988)</summary>

**Abstract:** Autoformalization serves a crucial role in connecting natural language and formal reasoning. This paper presents MASA, a novel framework for building multi-agent systems for autoformalization driven by Large Language Models (LLMs). MASA leverages collaborative agents to convert natural language statements into their formal representations. The architecture of MASA is designed with a strong emphasis on modularity, flexibility, and extensibility, allowing seamless integration of new agents and tools to adapt to a fast-evolving field. We showcase the effectiveness of MASA through use cases on real-world mathematical definitions and experiments on formal mathematics datasets. This work highlights the potential of multi-agent systems powered by the interaction of LLMs and theorem provers in enhancing the efficiency and reliability of autoformalization, providing valuable insights and support for researchers and practitioners in the field.

**arXiv ID:** 2510.08988
</details>

<details>
<summary><strong>DITING: A Multi-Agent Evaluation Framework for Benchmarking Web Novel Translation</strong> - Enze Zhang, Jiaying Wang, Mengxi Xiao, Jifei Liu, Ziyan Kuang, Rui Dong, Youzhong Dong, Sophia Ananiadou, Min Peng, Qianqian Xie - [[pdf]](https://arxiv.org/pdf/2510.09116)</summary>

**Abstract:** Large language models (LLMs) have substantially advanced machine translation (MT), yet their effectiveness in translating web novels remains unclear. Existing benchmarks rely on surface-level metrics that fail to capture the distinctive traits of this genre. To address these gaps, we introduce DITING, the first comprehensive evaluation framework for web novel translation, assessing narrative and cultural fidelity across six dimensions: idiom translation, lexical ambiguity, terminology localization, tense consistency, zero-pronoun resolution, and cultural safety, supported by over 18K expert-annotated Chinese-English sentence pairs. We further propose AgentEval, a reasoning-driven multi-agent evaluation framework that simulates expert deliberation to assess translation quality beyond lexical overlap, achieving the highest correlation with human judgments among seven tested automatic metrics. To enable metric comparison, we develop MetricAlign, a meta-evaluation dataset of 300 sentence pairs annotated with error labels and scalar quality scores. Comprehensive evaluation of fourteen open, closed, and commercial models reveals that Chinese-trained LLMs surpass larger foreign counterparts, and that DeepSeek-V3 delivers the most faithful and stylistically coherent translations. Our work establishes a new paradigm for exploring LLM-based web novel translation and provides public resources to advance future research.

**arXiv ID:** 2510.09116
</details>

<details>
<summary><strong>RedDebate: Safer Responses through Multi-Agent Red Teaming Debates</strong> - Ali Asad, Stephen Obadinma, Radin Shayanfar, Xiaodan Zhu - [[pdf]](https://arxiv.org/pdf/2506.11083)</summary>

**Abstract:** We introduce RedDebate, a novel multi-agent debate framework that provides the foundation for Large Language Models (LLMs) to identify and mitigate their unsafe behaviours. Existing AI safety approaches often rely on costly human evaluation or isolated single-model assessment, both constrained by scalability and prone to oversight failures. RedDebate employs collaborative argumentation among multiple LLMs across diverse debate scenarios, enabling them to critically evaluate one another's reasoning and systematically uncover unsafe failure modes through fully automated red-teaming. We further integrate distinct long-term memory modules that preserve safety-relevant insights from debate interactions and leverage them during subsequent inference, facilitating continuous refinement of model behaviour. Empirical evaluation on safety benchmarks across a diverse set of models demonstrates that RedDebate substantially reduces unsafe outputs. While debate alone allows LLMs to refine their behaviour, the addition of memory yields further significant reductions. To the best of our knowledge, RedDebate is the first fully automated framework to unify multi-agent debate and red-teaming to progressively enhance LLM safety without human intervention.

**arXiv ID:** 2506.11083
</details>

<details>
<summary><strong>LaTeXTrans: Structured LaTeX Translation with Multi-Agent Coordination</strong> - Ziming Zhu, Chenglong Wang, Shunjie Xing, Yifu Huo, Fengning Tian, Quan Du, Di Yang, Chunliang Zhang, Tong Xiao, Jingbo Zhu - [[pdf]](https://arxiv.org/pdf/2508.18791)</summary>

**Abstract:** Despite the remarkable progress of modern machine translation (MT) systems on general-domain texts, translating structured LaTeX-formatted documents remains a significant challenge. These documents typically interleave natural language with domain-specific syntax, such as mathematical equations, tables, figures, and cross-references, all of which must be accurately preserved to maintain semantic integrity and compilability. In this paper, we introduce LaTeXTrans, a collaborative multi-agent system designed to address this challenge. LaTeXTrans ensures format preservation, structural fidelity, and terminology consistency through six specialized agents: 1) a Parser that decomposes LaTeX into translation-friendly units via placeholder substitution and syntax filtering; 2) a Translator, Validator, Summarizer, and Terminology Extractor that work collaboratively to ensure context-aware, self-correcting, and terminology-consistent translations; 3) a Generator that reconstructs the translated content into well-structured LaTeX documents. Experimental results demonstrate that LaTeXTrans can outperform mainstream MT systems in both translation accuracy and structural fidelity, offering an effective and practical solution for translating LaTeX-formatted this http URL code of LaTeXTrans is available at this https URL.

**arXiv ID:** 2508.18791
</details>

<details>
<summary><strong>Can Lessons From Human Teams Be Applied to Multi-Agent Systems? The Role of Structure, Diversity, and Interaction Dynamics</strong> - Rasika Muralidharan, Haewoon Kwak, Jisun An - [[pdf]](https://arxiv.org/pdf/2510.07488)</summary>

**Abstract:** Multi-Agent Systems (MAS) with Large Language Model (LLM)-powered agents are gaining attention, yet fewer studies explore their team dynamics. Inspired by human team science, we propose a multi-agent framework to examine core aspects of team science: structure, diversity, and interaction dynamics. We evaluate team performance across four tasks: CommonsenseQA, StrategyQA, Social IQa, and Latent Implicit Hate, spanning commonsense and social reasoning. Our results show that flat teams tend to perform better than hierarchical ones, while diversity has a nuanced impact. Interviews suggest agents are overconfident about their team performance, yet post-task reflections reveal both appreciation for collaboration and challenges in integration, including limited conversational coordination.

**arXiv ID:** 2510.07488
</details>

<details>
<summary><strong>When LLM Agents Meet Graph Optimization: An Automated Data Quality Improvement Approach</strong> - Zhihan Zhang, Xunkai Li, Yilong Zuo, Zhenjun Li, Bing Zhou, Rong-Hua Li, Guoren Wang - [[pdf]](https://arxiv.org/pdf/2510.08952)</summary>

**Abstract:** Text-attributed graphs (TAGs) have emerged as a powerful representation that combines structural connections with fine-grained semantics, supporting a wide range of data-centric applications. However, the performance of graph neural networks (GNNs) on TAGs is highly sensitive to input quality. Our empirical study shows that both traditional GNNs and LLM-enhanced GNNs suffer significant degradation across nine representative scenarios of sparsity, noise, and imbalance, highlighting graph quality as a critical bottleneck. Existing approaches mainly focus on improving model architectures, while neglecting systematic optimization of TAG data itself, leading to limited effectiveness in practice. To address this gap, we propose LAGA (Large Language and Graph Agent), a unified multi-agent framework that treats graph quality control as a first-class, data-centric problem. LAGA integrates four collaborative agents-detection, planning, action, and evaluation-into an automated closed loop. At its core, the action agent employs a dual-encoder and tri-objective design to capture complementary information across modalities and perform holistic graph quality enhancement. Experiments across nine scenarios show that LAGA improves graph quality and achieves state-of-the-art performance across various tasks and backbones, validating data-centric quality optimization as key to reliable TAGs and robust graph learning.

**arXiv ID:** 2510.08952
</details>

<details>
<summary><strong>Agentic-KGR: Co-evolutionary Knowledge Graph Construction through Multi-Agent Reinforcement Learning</strong> - Jing Li, Zhijie Sun, Zhicheng Zhou, Suming Qiu, Junjie Huang, Haijia Sun, Linyuan Qiu - [[pdf]](https://arxiv.org/pdf/2510.09156)</summary>

**Abstract:** Current knowledge-enhanced large language models (LLMs) rely on static, pre-constructed knowledge bases that suffer from coverage gaps and temporal obsolescence, limiting their effectiveness in dynamic information environments. We present Agentic-KGR, a novel framework enabling co-evolution between LLMs and knowledge graphs (KGs) through multi-round reinforcement learning (RL). Our approach introduces three key innovations: (1) a dynamic schema expansion mechanism that systematically extends graph ontologies beyond pre-defined boundaries during training; (2) a retrieval-augmented memory system enabling synergistic co-evolution between model parameters and knowledge structures through continuous optimization; (3) a learnable multi-scale prompt compression approach that preserves critical information while reducing computational complexity through adaptive sequence optimization. Experimental results demonstrate substantial improvements over supervised baselines and single-round RL approaches in knowledge extraction tasks. When integrated with GraphRAG, our method achieves superior performance in downstream QA tasks, with significant gains in both accuracy and knowledge coverage compared to existing methods.

**arXiv ID:** 2510.09156
</details>

<details>
<summary><strong>FOGMACHINE -- Leveraging Discrete-Event Simulation and Scene Graphs for Modeling Hierarchical, Interconnected Environments under Partial Observations from Mobile Agents</strong> - Lars Ohnemus, Nils Hantke, Max Weißer, Kai Furmans - [[pdf]](https://arxiv.org/pdf/2510.09483)</summary>

**Abstract:** Dynamic Scene Graphs (DSGs) provide a structured representation of hierarchical, interconnected environments, but current approaches struggle to capture stochastic dynamics, partial observability, and multi-agent activity. These aspects are critical for embodied AI, where agents must act under uncertainty and delayed perception. We introduce FOGMACHINE , an open-source framework that fuses DSGs with discrete-event simulation to model object dynamics, agent observations, and interactions at scale. This setup enables the study of uncertainty propagation, planning under limited perception, and emergent multi-agent behavior. Experiments in urban scenarios illustrate realistic temporal and spatial patterns while revealing the challenges of belief estimation under sparse observations. By combining structured representations with efficient simulation, FOGMACHINE establishes an effective tool for benchmarking, model training, and advancing embodied AI in complex, uncertain environments.

**arXiv ID:** 2510.09483
</details>

<details>
<summary><strong>Student Development Agent: Risk-free Simulation for Evaluating AIED Innovations</strong> - Jianxiao Jiang, Yu Zhang - [[pdf]](https://arxiv.org/pdf/2510.09183)</summary>

**Abstract:** In the age of AI-powered educational (AIED) innovation, evaluating the developmental consequences of novel designs before they are exposed to students has become both essential and challenging. Since such interventions may carry irreversible effects, it is critical to anticipate not only potential benefits but also possible harms. This study proposes a student development agent framework based on large language models (LLMs), designed to simulate how students with diverse characteristics may evolve under different educational settings without administering them to real students. By validating the approach through a case study on a multi-agent learning environment (MAIC), we demonstrate that the agent's predictions align with real student outcomes in non-cognitive developments. The results suggest that LLM-based simulations hold promise for evaluating AIED innovations efficiently and ethically. Future directions include enhancing profile structures, incorporating fine-tuned or small task-specific models, validating effects of empirical findings, interpreting simulated data and optimizing evaluation methods.

**arXiv ID:** 2510.09183
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>RefGrader: Automated Grading of Mathematical Competition Proofs using Agentic Workflows</strong> - Hamed Mahdavi, Pouria Mahdavinia, Samira Malek, Pegah Mohammadipour, Alireza Hashemi, Majid Daliri, Alireza Farhadi, Amir Khasahmadi, Niloofar Mireshghallah, Vasant Honavar - [[pdf]](https://arxiv.org/pdf/2510.09021)</summary>

**Abstract:** State-of-the-art (SOTA) LLMs have progressed from struggling on proof-based Olympiad problems to solving most of the IMO 2025 problems, with leading systems reportedly handling 5 of 6 problems. Given this progress, we assess how well these models can grade proofs: detecting errors, judging their severity, and assigning fair scores beyond binary correctness. We study proof-analysis capabilities using a corpus of 90 Gemini 2.5 Pro-generated solutions that we grade on a 1-4 scale with detailed error annotations, and on MathArena solution sets for IMO/USAMO 2025 scored on a 0-7 scale. Our analysis shows that models can reliably flag incorrect (including subtly incorrect) solutions but exhibit calibration gaps in how partial credit is assigned. To address this, we introduce agentic workflows that extract and analyze reference solutions and automatically derive problem-specific rubrics for a multi-step grading process. We instantiate and compare different design choices for the grading workflows, and evaluate their trade-offs. Across our annotated corpus and MathArena, our proposed workflows achieve higher agreement with human grades and more consistent handling of partial credit across metrics. We release all code, data, and prompts/logs to facilitate future research.

**arXiv ID:** 2510.09021
</details>

<details>
<summary><strong>Safe, Untrusted, "Proof-Carrying" AI Agents: toward the agentic lakehouse</strong> - Jacopo Tagliabue, Ciro Greco - [[pdf]](https://arxiv.org/pdf/2510.09567)</summary>

**Abstract:** Data lakehouses run sensitive workloads, where AI-driven automation raises concerns about trust, correctness, and governance. We argue that API-first, programmable lakehouses provide the right abstractions for safe-by-design, agentic workflows. Using Bauplan as a case study, we show how data branching and declarative environments extend naturally to agents, enabling reproducibility and observability while reducing the attack surface. We present a proof-of-concept in which agents repair data pipelines using correctness checks inspired by proof-carrying code. Our prototype demonstrates that untrusted AI agents can operate safely on production data and outlines a path toward a fully agentic lakehouse.

**arXiv ID:** 2510.09567
</details>

<details>
<summary><strong>ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing</strong> - Noah Steinkrüger, Nisarga Nilavadi, Wolfram Burgard, Tanja Katharina Kaiser - [[pdf]](https://arxiv.org/pdf/2510.08705)</summary>

**Abstract:** Object transportation in cluttered environments is a fundamental task in various domains, including domestic service and warehouse logistics. In cooperative object transport, multiple robots must coordinate to move objects that are too large for a single robot. One transport strategy is pushing, which only requires simple robots. However, careful selection of robot-object contact points is necessary to push the object along a preplanned path. Although this selection can be solved analytically, the solution space grows combinatorially with the number of robots and object size, limiting scalability. Inspired by how humans rely on common-sense reasoning for cooperative transport, we propose combining the reasoning capabilities of Large Language Models with local search to select suitable contact points. Our LLM-guided local search method for contact point selection, ConPoSe, successfully selects contact points for a variety of shapes, including cuboids, cylinders, and T-shapes. We demonstrate that ConPoSe scales better with the number of robots and object size than the analytical approach, and also outperforms pure LLM-based selection.

**arXiv ID:** 2510.08705
</details>

<details>
<summary><strong>Autonomous Soft Robotic Guidewire Navigation via Imitation Learning</strong> - Noah Barnes, Ji Woong Kim, Lingyun Di, Hannah Qu, Anuruddha Bhattacharjee, Miroslaw Janowski, Dheeraj Gandhi, Bailey Felix, Shaopeng Jiang, Olivia Young, Mark Fuge, Ryan D. Sochol, Jeremy D. Brown, Axel Krieger - [[pdf]](https://arxiv.org/pdf/2510.09497)</summary>

**Abstract:** In endovascular surgery, endovascular interventionists push a thin tube called a catheter, guided by a thin wire to a treatment site inside the patient's blood vessels to treat various conditions such as blood clots, aneurysms, and malformations. Guidewires with robotic tips can enhance maneuverability, but they present challenges in modeling and control. Automation of soft robotic guidewire navigation has the potential to overcome these challenges, increasing the precision and safety of endovascular navigation. In other surgical domains, end-to-end imitation learning has shown promising results. Thus, we develop a transformer-based imitation learning framework with goal conditioning, relative action outputs, and automatic contrast dye injections to enable generalizable soft robotic guidewire navigation in an aneurysm targeting task. We train the model on 36 different modular bifurcated geometries, generating 647 total demonstrations under simulated fluoroscopy, and evaluate it on three previously unseen vascular geometries. The model can autonomously drive the tip of the robot to the aneurysm location with a success rate of 83% on the unseen geometries, outperforming several baselines. In addition, we present ablation and baseline studies to evaluate the effectiveness of each design and data collection choice. Project website: this https URL

**arXiv ID:** 2510.09497
</details>

<details>
<summary><strong>Agentic Exploration of Physics Models</strong> - Maximilian Nägele, Florian Marquardt - [[pdf]](https://arxiv.org/pdf/2509.24978)</summary>

**Abstract:** The process of scientific discovery relies on an interplay of observations, analysis, and hypothesis generation. Machine learning is increasingly being adopted to address individual aspects of this process. However, it remains an open challenge to fully automate the heuristic, iterative loop required to discover the laws of an unknown system by exploring it through experiments and analysis, without tailoring the approach to the specifics of a given task. Here, we introduce SciExplorer, an agent that leverages large language model tool-use capabilities to enable exploration of systems without any domain-specific blueprints, and apply it to physical systems that are initially unknown to the agent. We test SciExplorer on a broad set of models spanning mechanical dynamical systems, wave evolution, and quantum many-body physics. Despite using a minimal set of tools, primarily based on code execution, we observe impressive performance on tasks such as recovering equations of motion from observed dynamics and inferring Hamiltonians from expectation values. The demonstrated effectiveness of this setup opens the door towards similar scientific exploration in other domains, without the need for finetuning or task-specific instructions.

**arXiv ID:** 2509.24978
</details>

<details>
<summary><strong>RE-Searcher: Robust Agentic Search with Goal-oriented Planning and Self-reflection</strong> - Daocheng Fu, Jianbiao Mei, Licheng Wen, Xuemeng Yang, Cheng Yang, Rong Wu, Tao Hu, Siqi Li, Yufan Shen, Xinyu Cai, Pinlong Cai, Botian Shi, Yong Liu, Yu Qiao - [[pdf]](https://arxiv.org/pdf/2509.26048)</summary>

**Abstract:** Large language models (LLMs) excel at knowledge-intensive question answering and reasoning, yet their real-world deployment remains constrained by knowledge cutoff, hallucination, and limited interaction modalities. Augmenting LLMs with external search tools helps alleviate these issues, but it also exposes agents to a complex search environment in which small, plausible variations in query formulation can steer reasoning into unproductive trajectories and amplify errors. We present a systematic analysis that quantifies how environmental complexity induces fragile search behaviors and, in turn, degrades overall performance. To address this challenge, we propose a simple yet effective approach to instantiate a search agent, RE-Searcher. During search, RE-Searcher explicitly articulates a concrete search goal and subsequently reflects on whether the retrieved evidence satisfies that goal. This combination of goal-oriented planning and self-reflection enables RE-Searcher to resist spurious cues in complex search environments and perform robust search. Extensive experiments show that our method improves search accuracy and achieves state-of-the-art results. Perturbation studies further demonstrate substantial resilience to noisy or misleading external signals, mitigating the fragility of the search process. We believe these findings offer practical guidance for integrating LLM-powered agents into more complex interactive environments and enabling more autonomous decision-making.

**arXiv ID:** 2509.26048
</details>

<details>
<summary><strong>HANDO: Hierarchical Autonomous Navigation and Dexterous Omni-loco-manipulation</strong> - Jingyuan Sun, Chaoran Wang, Mingyu Zhang, Cui Miao, Hongyu Ji, Zihan Qu, Han Sun, Bing Wang, Qingyi Si - [[pdf]](https://arxiv.org/pdf/2510.09221)</summary>

**Abstract:** Seamless loco-manipulation in unstructured environments requires robots to leverage autonomous exploration alongside whole-body control for physical interaction. In this work, we introduce HANDO (Hierarchical Autonomous Navigation and Dexterous Omni-loco-manipulation), a two-layer framework designed for legged robots equipped with manipulators to perform human-centered mobile manipulation tasks. The first layer utilizes a goal-conditioned autonomous exploration policy to guide the robot to semantically specified targets, such as a black office chair in a dynamic environment. The second layer employs a unified whole-body loco-manipulation policy to coordinate the arm and legs for precise interaction tasks-for example, handing a drink to a person seated on the chair. We have conducted an initial deployment of the navigation module, and will continue to pursue finer-grained deployment of whole-body loco-manipulation.

**arXiv ID:** 2510.09221
</details>

<details>
<summary><strong>Safe Autonomous Environmental Contact for Soft Robots using Control Barrier Functions</strong> - Akua K. Dickson, Juan C. Pacheco Garcia, Meredith L. Anderson, Ran Jing, Sarah Alizadeh-Shabdiz, Audrey X. Wang, Charles DeLorey, Zach J. Patterson, Andrew P. Sabelhaus - [[pdf]](https://arxiv.org/pdf/2504.14755)</summary>

**Abstract:** Robots built from soft materials will inherently apply lower environmental forces than their rigid counterparts, and therefore may be more suitable in sensitive settings with unintended contact. However, these robots' applied forces result from both their design and their control system in closed-loop, and therefore, ensuring bounds on these forces requires controller synthesis for safety as well. This article introduces the first feedback controller for a soft manipulator that formally meets a safety specification with respect to environmental contact. In our proof-of-concept setting, the robot's environment has known geometry and is deformable with a known elastic modulus. Our approach maps a bound on applied forces to a safe set of positions of the robot's tip via predicted deformations of the environment. Then, a quadratic program with Control Barrier Functions in its constraints is used to supervise a nominal feedback signal, verifiably maintaining the robot's tip within this safe set. Hardware experiments on a multi-segment soft pneumatic robot demonstrate that the proposed framework successfully maintains a positive safety margin. This framework represents a fundamental shift in perspective on control and safety for soft robots, implementing a formally verifiable logic specification on their pose and contact forces.

**arXiv ID:** 2504.14755
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (29 papers)</h2></summary>

<details>
<summary><strong>Hypothesis Hunting with Evolving Networks of Autonomous Scientific Agents</strong> - Tennison Liu, Silas Ruhrberg Estévez, David L. Bentley, Mihaela van der Schaar - [[pdf]](https://arxiv.org/pdf/2510.08619)</summary>

**Abstract:** Large-scale scientific datasets -- spanning health biobanks, cell atlases, Earth reanalyses, and more -- create opportunities for exploratory discovery unconstrained by specific research questions. We term this process hypothesis hunting: the cumulative search for insight through sustained exploration across vast and complex hypothesis spaces. To support it, we introduce AScience, a framework modeling discovery as the interaction of agents, networks, and evaluation norms, and implement it as ASCollab, a distributed system of LLM-based research agents with heterogeneous behaviors. These agents self-organize into evolving networks, continually producing and peer-reviewing findings under shared standards of evaluation. Experiments show that such social dynamics enable the accumulation of expert-rated results along the diversity-quality-novelty frontier, including rediscoveries of established biomarkers, extensions of known pathways, and proposals of new therapeutic targets. While wet-lab validation remains indispensable, our experiments on cancer cohorts demonstrate that socially structured, agentic networks can sustain exploratory hypothesis hunting at scale.

**arXiv ID:** 2510.08619
</details>

<details>
<summary><strong>What Is Your Agent's GPA? A Framework for Evaluating Agent Goal-Plan-Action Alignment</strong> - Allison Sihan Jia, Daniel Huang, Nikhil Vytla, Nirvika Choudhury, John C Mitchell, Anupam Datta - [[pdf]](https://arxiv.org/pdf/2510.08847)</summary>

**Abstract:** We introduce the Agent GPA (Goal-Plan-Action) framework: an evaluation paradigm based on an agent's operational loop of setting goals, devising plans, and executing actions. The framework includes five evaluation metrics: Goal Fulfillment, Logical Consistency, Execution Efficiency, Plan Quality, and Plan Adherence. Logical Consistency checks that an agent's actions are consistent with its prior actions. Execution Efficiency checks whether the agent executes in the most efficient way to achieve its goal. Plan Quality checks whether an agent's plans are aligned with its goals; Plan Adherence checks if an agent's actions are aligned with its plan; and Goal Fulfillment checks that agent's final outcomes match the stated goals. Our experimental results on two benchmark datasets - the public TRAIL/GAIA dataset and an internal dataset for a production-grade data agent - show that this framework (a) provides a systematic way to cover a broad range of agent failures, including all agent errors on the TRAIL/GAIA benchmark dataset; (b) supports LLM-judges that exhibit strong agreement with human annotation, covering 80% to over 95% errors; and (c) localizes errors with 86% agreement to enable targeted improvement of agent performance.

**arXiv ID:** 2510.08847
</details>

<details>
<summary><strong>Leading the Follower: Learning Persuasive Agents in Social Deduction Games</strong> - Zhang Zheng, Deheng Ye, Peilin Zhao, Hao Wang - [[pdf]](https://arxiv.org/pdf/2510.09087)</summary>

**Abstract:** Large language model (LLM) agents have shown remarkable progress in social deduction games (SDGs). However, existing approaches primarily focus on information processing and strategy selection, overlooking the significance of persuasive communication in influencing other players' beliefs and responses. In SDGs, success depends not only on making correct deductions but on convincing others to response in alignment with one's intent. To address this limitation, we formalize turn-based dialogue in SDGs as a Stackelberg competition, where the current player acts as the leader who strategically influences the follower's response. Building on this theoretical foundation, we propose a reinforcement learning framework that trains agents to optimize utterances for persuasive impact. Through comprehensive experiments across three diverse SDGs, we demonstrate that our agents significantly outperform baselines. This work represents a significant step toward developing AI agents capable of strategic social influence, with implications extending to scenarios requiring persuasive communication.

**arXiv ID:** 2510.09087
</details>

<details>
<summary><strong>Agentic Systems in Radiology: Design, Applications, Evaluation, and Challenges</strong> - Christian Bluethgen, Dave Van Veen, Daniel Truhn, Jakob Nikolas Kather, Michael Moor, Malgorzata Polacin, Akshay Chaudhari, Thomas Frauenfelder, Curtis P. Langlotz, Michael Krauthammer, Farhad Nooralahzadeh - [[pdf]](https://arxiv.org/pdf/2510.09404)</summary>

**Abstract:** Building agents, systems that perceive and act upon their environment with a degree of autonomy, has long been a focus of AI research. This pursuit has recently become vastly more practical with the emergence of large language models (LLMs) capable of using natural language to integrate information, follow instructions, and perform forms of "reasoning" and planning across a wide range of tasks. With its multimodal data streams and orchestrated workflows spanning multiple systems, radiology is uniquely suited to benefit from agents that can adapt to context and automate repetitive yet complex tasks. In radiology, LLMs and their multimodal variants have already demonstrated promising performance for individual tasks such as information extraction and report summarization. However, using LLMs in isolation underutilizes their potential to support complex, multi-step workflows where decisions depend on evolving context from multiple information sources. Equipping LLMs with external tools and feedback mechanisms enables them to drive systems that exhibit a spectrum of autonomy, ranging from semi-automated workflows to more adaptive agents capable of managing complex processes. This review examines the design of such LLM-driven agentic systems, highlights key applications, discusses evaluation methods for planning and tool use, and outlines challenges such as error cascades, tool-use efficiency, and health IT integration.

**arXiv ID:** 2510.09404
</details>

<details>
<summary><strong>Upfront Chain-of-Thought: A Cooperative Framework for Chain-of-Thought Compression</strong> - Chengzhengxu Li, Xiaoming Liu, Zhaohan Zhang, Shaochu Zhang, Shengchao Liu, Guoxin Ma, Yu Lan, Chao Shen - [[pdf]](https://arxiv.org/pdf/2510.08647)</summary>

**Abstract:** Recent developments have enabled advanced reasoning in Large Language Models (LLMs) via long Chain-of-Thought (CoT), while long CoT suffers from high computational costs and significant latency losses owing to the autoregressive nature of generative LLMs. CoT compression aims to improve efficiency in the reasoning process by reducing output length. Previous works trade reasoning efficiency by either laborious discrete prompt designing or the construction of external compressed CoT datasets that sacrifice key reasoning details. In this work, we propose Upfront CoT (UCoT): an efficient reasoning framework with upfront thought embedding to automate CoT compression. UCoT is a cooperative workflow involving a small model (compressor) and a large model (executor). The first stage of UCoT trains compressor to generate upfront thought embeddings rich in reasoning information for the executor, avoiding the drawbacks of manually designed prompts. The second stage optimizes executor to utilize upfront thought embeddings to derive the correct answer with short reasoning, using a reward mechanism. Extensive experiments show that UCoT maintains the powerful reasoning ability of executor while significantly reducing the length of CoT. It is worth mentioning that when applying UCoT to the Qwen2.5-7B-Instruct model, the usage of tokens on GSM8K dataset is reduced by 50\%, while the performance is 3.08\% higher than that of the state-of-the-art (SOTA) method. The code and dataset are in supplementary material.

**arXiv ID:** 2510.08647
</details>

<details>
<summary><strong>Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations</strong> - Vaibhav Jain, Gerrit Grossmann - [[pdf]](https://arxiv.org/pdf/2510.08779)</summary>

**Abstract:** Reinforcement Learning (RL) agents often struggle in sparse-reward environments where traditional exploration strategies fail to discover effective action sequences. Large Language Models (LLMs) possess procedural knowledge and reasoning capabilities from text pretraining that could guide RL exploration, but existing approaches create rigid dependencies where RL policies must follow LLM suggestions or incorporate them directly into reward functions. We propose a framework that provides LLM-generated action recommendations through augmented observation spaces, allowing RL agents to learn when to follow or ignore this guidance. Our method leverages LLMs' world knowledge and reasoning abilities while maintaining flexibility through soft constraints. We evaluate our approach on three BabyAI environments of increasing complexity and show that the benefits of LLM guidance scale with task difficulty. In the most challenging environment, we achieve 71% relative improvement in final success rates over baseline. The approach provides substantial sample efficiency gains, with agents reaching performance thresholds up to 9 times faster, and requires no modifications to existing RL algorithms. Our results demonstrate an effective method for leveraging LLM planning capabilities to accelerate RL training in challenging environments.

**arXiv ID:** 2510.08779
</details>

<details>
<summary><strong>Reinforcement Learning-Driven Edge Management for Reliable Multi-view 3D Reconstruction</strong> - Motahare Mounesan, Sourya Saha, Houchao Gan, Md. Nurul Absur, Saptarshi Debroy - [[pdf]](https://arxiv.org/pdf/2510.08839)</summary>

**Abstract:** Real-time multi-view 3D reconstruction is a mission-critical application for key edge-native use cases, such as fire rescue, where timely and accurate 3D scene modeling enables situational awareness and informed decision-making. However, the dynamic and unpredictable nature of edge resource availability introduces disruptions, such as degraded image quality, unstable network links, and fluctuating server loads, which challenge the reliability of the reconstruction pipeline. In this work, we present a reinforcement learning (RL)-based edge resource management framework for reliable 3D reconstruction to ensure high quality reconstruction within a reasonable amount of time, despite the system operating under a resource-constrained and disruption-prone environment. In particular, the framework adopts two cooperative Q-learning agents, one for camera selection and one for server selection, both of which operate entirely online, learning policies through interactions with the edge environment. To support learning under realistic constraints and evaluate system performance, we implement a distributed testbed comprising lab-hosted end devices and FABRIC infrastructure-hosted edge servers to emulate smart city edge infrastructure under realistic disruption scenarios. Results show that the proposed framework improves application reliability by effectively balancing end-to-end latency and reconstruction quality in dynamic environments.

**arXiv ID:** 2510.08839
</details>

<details>
<summary><strong>Pinpointing crucial steps: Attribution-based Credit Assignment for Verifiable Reinforcement Learning</strong> - Junxi Yin, Haisen Luo, Zhenyu Li, Yihua Liu, Dan Liu, Zequn Li, Xiaohang Xu - [[pdf]](https://arxiv.org/pdf/2510.08899)</summary>

**Abstract:** While Reinforcement Learning with Verifiable Rewards (RLVR) enhances complex reasoning in LLMs, current methods struggle to balance exploration and exploitation. This leads to critical issues like inaccurate credit assignment for intermediate steps and premature entropy collapse, limiting model performance. To address this, we introduce Attribution-based Contribution to Policy Optimization (ACPO), a phased framework that incorporates a difficulty-aware curriculum. ACPO improves exploration by using trajectory semantic segmentation and an attribution-based representation to dynamically regulate policy entropy, thus mitigating its collapse. Concurrently, it enhances exploitation with a factorized reward system that precisely quantifies the hierarchical contribution of each reasoning step, ensuring accurate credit assignment. Extensive experiments on challenging benchmarks, including AIME, MATH, and AMC, demonstrate that ACPO significantly outperforms existing state-of-the-art approaches.

**arXiv ID:** 2510.08899
</details>

<details>
<summary><strong>Robust Driving Control for Autonomous Vehicles: An Intelligent General-sum Constrained Adversarial Reinforcement Learning Approach</strong> - Junchao Fan, Xiaolin Chang - [[pdf]](https://arxiv.org/pdf/2510.09041)</summary>

**Abstract:** Deep reinforcement learning (DRL) has demonstrated remarkable success in developing autonomous driving policies. However, its vulnerability to adversarial attacks remains a critical barrier to real-world deployment. Although existing robust methods have achieved success, they still suffer from three key issues: (i) these methods are trained against myopic adversarial attacks, limiting their abilities to respond to more strategic threats, (ii) they have trouble causing truly safety-critical events (e.g., collisions), but instead often result in minor consequences, and (iii) these methods can introduce learning instability and policy drift during training due to the lack of robust constraints. To address these issues, we propose Intelligent General-sum Constrained Adversarial Reinforcement Learning (IGCARL), a novel robust autonomous driving approach that consists of a strategic targeted adversary and a robust driving agent. The strategic targeted adversary is designed to leverage the temporal decision-making capabilities of DRL to execute strategically coordinated multi-step attacks. In addition, it explicitly focuses on inducing safety-critical events by adopting a general-sum objective. The robust driving agent learns by interacting with the adversary to develop a robust autonomous driving policy against adversarial attacks. To ensure stable learning in adversarial environments and to mitigate policy drift caused by attacks, the agent is optimized under a constrained formulation. Extensive experiments show that IGCARL improves the success rate by at least 27.9\% over state-of-the-art methods, demonstrating superior robustness to adversarial attacks and enhancing the safety and reliability of DRL-based autonomous driving.

**arXiv ID:** 2510.09041
</details>

<details>
<summary><strong>Obstacle Avoidance using Dynamic Movement Primitives and Reinforcement Learning</strong> - Dominik Urbaniak, Alejandro Agostini, Pol Ramon, Jan Rosell, Raúl Suárez, Michael Suppa - [[pdf]](https://arxiv.org/pdf/2510.09254)</summary>

**Abstract:** Learning-based motion planning can quickly generate near-optimal trajectories. However, it often requires either large training datasets or costly collection of human demonstrations. This work proposes an alternative approach that quickly generates smooth, near-optimal collision-free 3D Cartesian trajectories from a single artificial demonstration. The demonstration is encoded as a Dynamic Movement Primitive (DMP) and iteratively reshaped using policy-based reinforcement learning to create a diverse trajectory dataset for varying obstacle configurations. This dataset is used to train a neural network that takes as inputs the task parameters describing the obstacle dimensions and location, derived automatically from a point cloud, and outputs the DMP parameters that generate the trajectory. The approach is validated in simulation and real-robot experiments, outperforming a RRT-Connect baseline in terms of computation and execution time, as well as trajectory length, while supporting multi-modal trajectory generation for different obstacle geometries and end-effector dimensions. Videos and the implementation code are available at this https URL.

**arXiv ID:** 2510.09254
</details>

<details>
<summary><strong>Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models</strong> - Yongding Tao, Tian Wang, Yihong Dong, Huanyu Liu, Kechi Zhang, Xiaolong Hu, Ge Li - [[pdf]](https://arxiv.org/pdf/2510.09259)</summary>

**Abstract:** Data contamination poses a significant threat to the reliable evaluation of Large Language Models (LLMs). This issue arises when benchmark samples may inadvertently appear in training sets, compromising the validity of reported performance. While detection methods have been developed for the pre-training and Supervised Fine-Tuning stages, a critical research gap exists for the increasingly significant phase of Reinforcement Learning (RL) post-training. As RL post-training becomes pivotal for advancing LLM reasoning, the absence of specialized contamination detection methods in this paradigm presents a critical vulnerability. To address this, we conduct the first systematic study of data detection within RL post-training scenario and propose Self-Critique. Our method is motivated by a key observation: after RL phase, the output entropy distribution of LLMs tends to collapse into highly specific and sparse modes. Self-Critique probes for the underlying policy collapse, i.e., the model's convergence to a narrow reasoning path, which causes this entropy reduction. To facilitate this research, we also introduce RL-MIA, a benchmark constructed to simulate this specific contamination scenario. Extensive experiments show that Self-Critique significantly outperforms baseline methods across multiple models and contamination tasks, achieving an AUC improvement of up to 30%. Whereas existing methods are close to a random guess for RL-phase contamination, our method makes detection possible.

**arXiv ID:** 2510.09259
</details>

<details>
<summary><strong>Dyna-Mind: Learning to Simulate from Experience for Better AI Agents</strong> - Xiao Yu, Baolin Peng, Michel Galley, Hao Cheng, Qianhui Wu, Janardhan Kulkarni, Suman Nath, Zhou Yu, Jianfeng Gao - [[pdf]](https://arxiv.org/pdf/2510.09577)</summary>

**Abstract:** Reasoning models have recently shown remarkable progress in domains such as math and coding. However, their expert-level abilities in math and coding contrast sharply with their performance in long-horizon, interactive tasks such as web navigation and computer/phone-use. Inspired by literature on human cognition, we argue that current AI agents need ''vicarious trial and error'' - the capacity to mentally simulate alternative futures before acting - in order to enhance their understanding and performance in complex interactive environments. We introduce Dyna-Mind, a two-stage training framework that explicitly teaches (V)LM agents to integrate such simulation into their reasoning. In stage 1, we introduce Reasoning with Simulations (ReSim), which trains the agent to generate structured reasoning traces from expanded search trees built from real experience gathered through environment interactions. ReSim thus grounds the agent's reasoning in faithful world dynamics and equips it with the ability to anticipate future states in its reasoning. In stage 2, we propose Dyna-GRPO, an online reinforcement learning method to further strengthen the agent's simulation and decision-making ability by using both outcome rewards and intermediate states as feedback from real rollouts. Experiments on two synthetic benchmarks (Sokoban and ALFWorld) and one realistic benchmark (AndroidWorld) demonstrate that (1) ReSim effectively infuses simulation ability into AI agents, and (2) Dyna-GRPO leverages outcome and interaction-level signals to learn better policies for long-horizon, planning-intensive tasks. Together, these results highlight the central role of simulation in enabling AI agents to reason, plan, and act more effectively in the ever more challenging environments.

**arXiv ID:** 2510.09577
</details>

<details>
<summary><strong>A Method to Improve the Performance of Reinforcement Learning Based on the Y Operator for a Class of Stochastic Differential Equation-Based Child-Mother Systems</strong> - Cheng Yin, Yi Chen - [[pdf]](https://arxiv.org/pdf/2311.04014)</summary>

**Abstract:** This paper introduces a novel operator, termed the Y operator, to elevate control performance in Actor-Critic(AC) based reinforcement learning for systems governed by stochastic differential equations(SDEs). The Y operator ingeniously integrates the stochasticity of a class of child-mother system into the Critic network's loss function, yielding substantial advancements in the control performance of RL this http URL, the Y operator elegantly reformulates the challenge of solving partial differential equations for the state-value function into a parallel problem for the drift and diffusion functions within the system's SDEs.A rigorous mathematical proof confirms the operator's this http URL transformation enables the Y Operator-based Reinforcement Learning(YORL) framework to efficiently tackle optimal control problems in both model-based and data-driven this http URL superiority of YORL is demonstrated through linear and nonlinear numerical examples showing its enhanced performance over existing methods post convergence.

**arXiv ID:** 2311.04014
</details>

<details>
<summary><strong>GUI-Shift: Enhancing VLM-Based GUI Agents through Self-supervised Reinforcement Learning</strong> - Longxi Gao, Li Zhang, Pengzhi Gao, Wei Liu, Jian Luan, Mengwei Xu - [[pdf]](https://arxiv.org/pdf/2505.12493)</summary>

**Abstract:** Training effective Vision-Language Models (VLMs) for GUI agents typically depends on large-scale annotated datasets, whose collection is both labor-intensive and error-prone. We introduce K-step GUI Transition, a self-supervised inverse dynamics task in which VLMs learn GUI dynamics by predicting the initial action that causes a transition between two GUI states. This approach eliminates the need for natural language instructions and enables scalable dataset construction from existing GUI trajectories or automated exploration. Building on this task, we propose GUI-Shift, a reinforcement learning (RL) framework that combines rule-based optimization with data filtering to improve VLM performance. We conduct extensive experiments using multiple VLM backbones across four benchmarks, spanning GUI task automation (AndroidControl, GUI Odyssey) and GUI grounding (ScreenSpot-v2, ScreenSpot-Pro). Our results show that training on GUI-Shift generalizes well to both GUI automation and grounding tasks, yielding up to an 11.2% increase in GUI automation accuracy. This study underscores the potential of self-supervised RL to leverage unlabeled GUI trajectories and offers a scalable alternative to training with annotated samples.

**arXiv ID:** 2505.12493
</details>

<details>
<summary><strong>Search-Based Credit Assignment for Offline Preference-Based Reinforcement Learning</strong> - Xiancheng Gao, Yufeng Shi, Wengang Zhou, Houqiang Li - [[pdf]](https://arxiv.org/pdf/2508.15327)</summary>

**Abstract:** Offline reinforcement learning refers to the process of learning policies from fixed datasets, without requiring additional environment interaction. However, it often relies on well-defined reward functions, which are difficult and expensive to design. Human feedback is an appealing alternative, but its two common forms, expert demonstrations and preferences, have complementary limitations. Demonstrations provide stepwise supervision, but they are costly to collect and often reflect limited expert behavior modes. In contrast, preferences are easier to collect, but it is unclear which parts of a behavior contribute most to a trajectory segment, leaving credit assignment unresolved. In this paper, we introduce a Search-Based Preference Weighting (SPW) scheme to unify these two feedback sources. For each transition in a preference labeled trajectory, SPW searches for the most similar state-action pairs from expert demonstrations and directly derives stepwise importance weights based on their similarity scores. These weights are then used to guide standard preference learning, enabling more accurate credit assignment that traditional approaches struggle to achieve. We demonstrate that SPW enables effective joint learning from preferences and demonstrations, outperforming prior methods that leverage both feedback types on challenging robot manipulation tasks.

**arXiv ID:** 2508.15327
</details>

<details>
<summary><strong>DSPO: Stable and Efficient Policy Optimization for Agentic Search and Reasoning</strong> - Chenyang Gu, Yewen Pu, Bruce Yang, Xiaofan Li, Huan Gao - [[pdf]](https://arxiv.org/pdf/2510.09255)</summary>

**Abstract:** Enhancing LLMs with the ability to actively search external knowledge is crucial for complex and real-world tasks. Current approaches either rely on prompting to elicit the model's innate agent capabilities, or suffer from performance ceilings and collapse when applying RL to complex interactive tasks, leaving their true agentic potential untapped. To address this, we introduce \textbf{D}ynamic-filter \textbf{S}equence-level \textbf{P}olicy \textbf{O}ptimization (DSPO), an improved RL algorithm designed for robust agent training through sequence-level optimization and dynamic sample filtering. We train our model purely through RL to interleave multi-turn search and reasoning, obviating the need for supervised demonstration data. Across multiple QA benchmarks, our DSPO-trained 7B model improves over a comparable previous work by \textbf{34.1\%}, and even outperforms the 14B model from previous work in complex multihop QA such as HotpotQA by nearly \textbf{9\% relative}, maintaining exceptional training stability.

**arXiv ID:** 2510.09255
</details>

<details>
<summary><strong>Language Model Guided Reinforcement Learning in Quantitative Trading</strong> - Adam Darmanin, Vince Vella - [[pdf]](https://arxiv.org/pdf/2508.02366)</summary>

**Abstract:** Algorithmic trading requires short-term tactical decisions consistent with long-term financial objectives. Reinforcement Learning (RL) has been applied to such problems, but adoption is limited by myopic behaviour and opaque policies. Large Language Models (LLMs) offer complementary strategic reasoning and multi-modal signal interpretation when guided by well-structured prompts.
This paper proposes a hybrid framework in which LLMs generate high-level trading strategies to guide RL agents. We evaluate (i) the economic rationale of LLM-generated strategies through expert review, and (ii) the performance of LLM-guided agents against unguided RL baselines using Sharpe Ratio (SR) and Maximum Drawdown (MDD).
Empirical results indicate that LLM guidance improves both return and risk metrics relative to standard RL.

**arXiv ID:** 2508.02366
</details>

<details>
<summary><strong>Reasoning through Exploration: A Reinforcement Learning Framework for Robust Function Calling</strong> - Bingguang Hao, Zengzhuang Xu, Maolin Wang, Yuntao Wen, Yicheng Chen, Cunyin Peng, Long Chen, Dong Wang, Xiangyu Zhao, Jinjie Gu, Chenyi Zhuang, Ji Zhang - [[pdf]](https://arxiv.org/pdf/2508.05118)</summary>

**Abstract:** The effective training of Large Language Models (LLMs) for function calling faces a critical challenge: balancing exploration of complex reasoning paths with stable policy optimization. Standard methods like Supervised Fine-Tuning (SFT) fail to instill robust reasoning, and traditional Reinforcement Learning (RL) struggles with inefficient exploration. We propose \textbf{EGPO}, a new RL framework built upon Group Relative Policy Optimization (GRPO), designed to address this challenge directly. The core of EGPO is an entropy-enhanced advantage function that integrates the entropy of the model's Chain-of-Thought (CoT) into the policy gradient computation. This encourages the generation of diverse reasoning strategies. To maintain optimization direction, the entropy bonus is carefully constrained by a clipping mechanism. Complemented by a strict, binary reward signal, EGPO effectively guides the model towards discovering structured and accurate tool invocation patterns. On the challenging Berkeley Function Calling Leaderboard (BFCL), a 4B-parameter model trained with EGPO sets a new state-of-the-art among models of comparable size, surpassing a range of strong competitors, including GPT-4o and Gemini-2.5.

**arXiv ID:** 2508.05118
</details>

<details>
<summary><strong>Reinforcement Learning-Based Optimization of CT Acquisition and Reconstruction Parameters Through Virtual Imaging Trials</strong> - David Fenwick, Navid NaderiAlizadeh, Vahid Tarokh, Nicholas Felice, Darin Clark, Jayasai Rajagopal, Anuj Kapadia, Benjamin Wildman-Tobriner, Ehsan Samei, Ehsan Abadi - [[pdf]](https://arxiv.org/pdf/2510.08763)</summary>

**Abstract:** Protocol optimization is critical in Computed Tomography (CT) to achieve high diagnostic image quality while minimizing radiation dose. However, due to the complex interdependencies among CT acquisition and reconstruction parameters, traditional optimization methods rely on exhaustive testing of combinations of these parameters, which is often impractical. This study introduces a novel methodology that combines virtual imaging tools with reinforcement learning to optimize CT protocols more efficiently. Human models with liver lesions were imaged using a validated CT simulator and reconstructed with a novel CT reconstruction toolkit. The optimization parameter space included tube voltage, tube current, reconstruction kernel, slice thickness, and pixel size. The optimization process was performed using a Proximal Policy Optimization (PPO) agent, which was trained to maximize an image quality objective, specifically the detectability index (d') of liver lesions in the reconstructed images. Optimization performance was compared against an exhaustive search performed on a supercomputer. The proposed reinforcement learning approach achieved the global maximum d' across test cases while requiring 79.7% fewer steps than the exhaustive search, demonstrating both accuracy and computational efficiency. The proposed framework is flexible and can accommodate various image quality objectives. The findings highlight the potential of integrating virtual imaging tools with reinforcement learning for CT protocol management.

**arXiv ID:** 2510.08763
</details>

<details>
<summary><strong>Zero-Shot Policy Transfer in Reinforcement Learning using Buckingham's Pi Theorem</strong> - Francisco Pascoa, Ian Lalonde, Alexandre Girard - [[pdf]](https://arxiv.org/pdf/2510.08768)</summary>

**Abstract:** Reinforcement learning (RL) policies often fail to generalize to new robots, tasks, or environments with different physical parameters, a challenge that limits their real-world applicability. This paper presents a simple, zero-shot transfer method based on Buckingham's Pi Theorem to address this limitation. The method adapts a pre-trained policy to new system contexts by scaling its inputs (observations) and outputs (actions) through a dimensionless space, requiring no retraining. The approach is evaluated against a naive transfer baseline across three environments of increasing complexity: a simulated pendulum, a physical pendulum for sim-to-real validation, and the high-dimensional HalfCheetah. Results demonstrate that the scaled transfer exhibits no loss of performance on dynamically similar contexts. Furthermore, on non-similar contexts, the scaled policy consistently outperforms the naive transfer, significantly expanding the volume of contexts where the original policy remains effective. These findings demonstrate that dimensional analysis provides a powerful and practical tool to enhance the robustness and generalization of RL policies.

**arXiv ID:** 2510.08768
</details>

<details>
<summary><strong>FM-IRL: Flow-Matching for Reward Modeling and Policy Regularization in Reinforcement Learning</strong> - Zhenglin Wan, Jingxuan Wu, Xingrui Yu, Chubin Zhang, Mingcong Lei, Bo An, Ivor Tsang - [[pdf]](https://arxiv.org/pdf/2510.09222)</summary>

**Abstract:** Flow Matching (FM) has shown remarkable ability in modeling complex distributions and achieves strong performance in offline imitation learning for cloning expert behaviors. However, despite its behavioral cloning expressiveness, FM-based policies are inherently limited by their lack of environmental interaction and exploration. This leads to poor generalization in unseen scenarios beyond the expert demonstrations, underscoring the necessity of online interaction with environment. Unfortunately, optimizing FM policies via online interaction is challenging and inefficient due to instability in gradient computation and high inference costs. To address these issues, we propose to let a student policy with simple MLP structure explore the environment and be online updated via RL algorithm with a reward model. This reward model is associated with a teacher FM model, containing rich information of expert data distribution. Furthermore, the same teacher FM model is utilized to regularize the student policy's behavior to stabilize policy learning. Due to the student's simple architecture, we avoid the gradient instability of FM policies and enable efficient online exploration, while still leveraging the expressiveness of the teacher FM model. Extensive experiments show that our approach significantly enhances learning efficiency, generalization, and robustness, especially when learning from suboptimal expert data.

**arXiv ID:** 2510.09222
</details>

<details>
<summary><strong>Application of Deep Reinforcement Learning to At-the-Money S&P 500 Options Hedging</strong> - Zofia Bracha, Paweł Sakowski, Jakub Michańków - [[pdf]](https://arxiv.org/pdf/2510.09247)</summary>

**Abstract:** This paper explores the application of deep Q-learning to hedging at-the-money options on the S\&P~500 index. We develop an agent based on the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, trained to simulate hedging decisions without making explicit model assumptions on price dynamics. The agent was trained on historical intraday prices of S\&P~500 call options across years 2004--2024, using a single time series of six predictor variables: option price, underlying asset price, moneyness, time to maturity, realized volatility, and current hedge position. A walk-forward procedure was applied for training, which led to nearly 17~years of out-of-sample evaluation. The performance of the deep reinforcement learning (DRL) agent is benchmarked against the Black--Scholes delta-hedging strategy over the same period. We assess both approaches using metrics such as annualized return, volatility, information ratio, and Sharpe ratio. To test the models' adaptability, we performed simulations across varying market conditions and added constraints such as transaction costs and risk-awareness penalties. Our results show that the DRL agent can outperform traditional hedging methods, particularly in volatile or high-cost environments, highlighting its robustness and flexibility in practical trading contexts. While the agent consistently outperforms delta-hedging, its performance deteriorates when the risk-awareness parameter is higher. We also observed that the longer the time interval used for volatility estimation, the more stable the results.

**arXiv ID:** 2510.09247
</details>

<details>
<summary><strong>An Imitative Reinforcement Learning Framework for Pursuit-Lock-Launch Missions</strong> - Siyuan Li, Rongchang Zuo, Bofei Liu, Yaoyu He, Peng Liu, Yingnan Zhao - [[pdf]](https://arxiv.org/pdf/2406.11562)</summary>

**Abstract:** Unmanned Combat Aerial Vehicle (UCAV) Within-Visual-Range (WVR) engagement, referring to a fight between two or more UCAVs at close quarters, plays a decisive role on the aerial battlefields. With the development of artificial intelligence, WVR engagement progressively advances towards intelligent and autonomous modes. However, autonomous WVR engagement policy learning is hindered by challenges such as weak exploration capabilities, low learning efficiency, and unrealistic simulated environments. To overcome these challenges, we propose a novel imitative reinforcement learning framework, which efficiently leverages expert data while enabling autonomous exploration. The proposed framework not only enhances learning efficiency through expert imitation, but also ensures adaptability to dynamic environments via autonomous exploration with reinforcement learning. Therefore, the proposed framework can learn a successful policy of `pursuit-lock-launch' for UCAVs. To support data-driven learning, we establish an environment based on the Harfang3D sandbox. The extensive experiment results indicate that the proposed framework excels in this multistage task, and significantly outperforms state-of-the-art reinforcement learning and imitation learning methods. Thanks to the ability of imitating experts and autonomous exploration, our framework can quickly learn the critical knowledge in complex aerial combat tasks, achieving up to a 100% success rate and demonstrating excellent robustness.

**arXiv ID:** 2406.11562
</details>

<details>
<summary><strong>Discrete Compositional Generation via General Soft Operators and Robust Reinforcement Learning</strong> - Marco Jiralerspong, Esther Derman, Danilo Vucetic, Nikolay Malkin, Bilun Sun, Tianyu Zhang, Pierre-Luc Bacon, Gauthier Gidel - [[pdf]](https://arxiv.org/pdf/2506.17007)</summary>

**Abstract:** A major bottleneck in scientific discovery consists of narrowing an exponentially large set of objects, such as proteins or molecules, to a small set of promising candidates with desirable properties. While this process can rely on expert knowledge, recent methods leverage reinforcement learning (RL) guided by a proxy reward function to enable this filtering. By employing various forms of entropy regularization, these methods aim to learn samplers that generate diverse candidates that are highly rated by the proxy function. In this work, we make two main contributions. First, we show that these methods are liable to generate overly diverse, suboptimal candidates in large search spaces. To address this issue, we introduce a novel unified operator that combines several regularized RL operators into a general framework that better targets peakier sampling distributions. Secondly, we offer a novel, robust RL perspective of this filtering process. The regularization can be interpreted as robustness to a compositional form of uncertainty in the proxy function (i.e., the true evaluation of a candidate differs from the proxy's evaluation). Our analysis leads us to a novel, easy-to-use algorithm we name trajectory general mellowmax (TGM): we show it identifies higher quality, diverse candidates than baselines in both synthetic and real-world tasks. Code: this https URL.

**arXiv ID:** 2506.17007
</details>

<details>
<summary><strong>EFRame: Deeper Reasoning via Exploration-Filter-Replay Reinforcement Learning Framework</strong> - Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Yuzhi Zhang, Yue Wang - [[pdf]](https://arxiv.org/pdf/2506.22200)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) have significantly enhanced the reasoning capabilities of large language models (LLMs). Group Relative Policy Optimization (GRPO), a lightweight variant of Proximal Policy Optimization (PPO), improves efficiency but suffers from limited exploration and training instability, limiting its effectiveness on complex reasoning tasks. To address these challenges, we introduce EFRame, an Exploration-Filter-Replay framework that augments GRPO across three dimensions: additional rollouts enable deeper and more targeted exploration, online filtering removes low-quality samples to stabilize gradients and accelerate training, and experience replay amplifies rare yet informative trajectories for stable convergence. This unified framework establishes a principled training cycle that balances exploration, efficiency, and stability. Experiments on diverse reasoning benchmarks demonstrate that EFRame achieves consistent gains, including a 37.9\% relative improvement on Geometry3K over GRPO. EFRame further supports fine-grained sample categorization and precise entropy control, highlighting it as a robust solution for advancing deeper reasoning in LLMs. Our code is available at this https URL.

**arXiv ID:** 2506.22200
</details>

<details>
<summary><strong>CDE: Concept-Driven Exploration for Reinforcement Learning</strong> - Le Mao, Andrew H. Liu, Renos Zabounidis, Zachary Kingston, Joseph Campbell - [[pdf]](https://arxiv.org/pdf/2510.08851)</summary>

**Abstract:** Intelligent exploration remains a critical challenge in reinforcement learning (RL), especially in visual control tasks. Unlike low-dimensional state-based RL, visual RL must extract task-relevant structure from raw pixels, making exploration inefficient. We propose Concept-Driven Exploration (CDE), which leverages a pre-trained vision-language model (VLM) to generate object-centric visual concepts from textual task descriptions as weak, potentially noisy supervisory signals. Rather than directly conditioning on these noisy signals, CDE trains a policy to reconstruct the concepts via an auxiliary objective, using reconstruction accuracy as an intrinsic reward to guide exploration toward task-relevant objects. Because the policy internalizes these concepts, VLM queries are only needed during training, reducing dependence on external models during deployment. Across five challenging simulated visual manipulation tasks, CDE achieves efficient, targeted exploration and remains robust to noisy VLM predictions. Finally, we demonstrate real-world transfer by deploying CDE on a Franka Research 3 arm, attaining an 80\% success rate in a real-world manipulation task.

**arXiv ID:** 2510.08851
</details>

<details>
<summary><strong>Model-Based Lookahead Reinforcement Learning for in-hand manipulation</strong> - Alexandre Lopes, Catarina Barata, Plinio Moreno - [[pdf]](https://arxiv.org/pdf/2510.08884)</summary>

**Abstract:** In-Hand Manipulation, as many other dexterous tasks, remains a difficult challenge in robotics by combining complex dynamic systems with the capability to control and manoeuvre various objects using its actuators. This work presents the application of a previously developed hybrid Reinforcement Learning (RL) Framework to In-Hand Manipulation task, verifying that it is capable of improving the performance of the task. The model combines concepts of both Model-Free and Model-Based Reinforcement Learning, by guiding a trained policy with the help of a dynamic model and value-function through trajectory evaluation, as done in Model Predictive Control. This work evaluates the performance of the model by comparing it with the policy that will be guided. To fully explore this, various tests are performed using both fully-actuated and under-actuated simulated robotic hands to manipulate different objects for a given task. The performance of the model will also be tested for generalization tests, by changing the properties of the objects in which both the policy and dynamic model were trained, such as density and size, and additionally by guiding a trained policy in a certain object to perform the same task in a different one. The results of this work show that, given a policy with high average reward and an accurate dynamic model, the hybrid framework improves the performance of in-hand manipulation tasks for most test cases, even when the object properties are changed. However, this improvement comes at the expense of increasing the computational cost, due to the complexity of trajectory evaluation.

**arXiv ID:** 2510.08884
</details>

<details>
<summary><strong>Zero-shot Structure Learning and Planning for Autonomous Robot Navigation using Active Inference</strong> - Daria de tinguy, Tim Verbelen, Emilio Gamba, Bart Dhoedt - [[pdf]](https://arxiv.org/pdf/2510.09574)</summary>

**Abstract:** Autonomous navigation in unfamiliar environments requires robots to simultaneously explore, localise, and plan under uncertainty, without relying on predefined maps or extensive training. We present a biologically inspired, Active Inference-based framework, Active Inference MAPping and Planning (AIMAPP). This model unifies mapping, localisation, and decision-making within a single generative model. Inspired by hippocampal navigation, it uses topological reasoning, place-cell encoding, and episodic memory to guide behaviour. The agent builds and updates a sparse topological map online, learns state transitions dynamically, and plans actions by minimising Expected Free Energy. This allows it to balance goal-directed and exploratory behaviours. We implemented a ROS-compatible navigation system that is sensor and robot-agnostic, capable of integrating with diverse hardware configurations. It operates in a fully self-supervised manner, is resilient to drift, and supports both exploration and goal-directed navigation without any pre-training. We demonstrate robust performance in large-scale real and simulated environments against state-of-the-art planning models, highlighting the system's adaptability to ambiguous observations, environmental changes, and sensor noise. The model offers a biologically inspired, modular solution to scalable, self-supervised navigation in unstructured settings. AIMAPP is available at this https URL.

**arXiv ID:** 2510.09574
</details>

<details>
<summary><strong>Maximizing UAV Cellular Connectivity with Reinforcement Learning for BVLoS Path Planning</strong> - Mehran Behjati, Rosdiadee Nordin, Nor Fadzilah Abdullah - [[pdf]](https://arxiv.org/pdf/2509.13336)</summary>

**Abstract:** This paper presents a reinforcement learning (RL) based approach for path planning of cellular connected unmanned aerial vehicles (UAVs) operating beyond visual line of sight (BVLoS). The objective is to minimize travel distance while maximizing the quality of cellular link connectivity by considering real world aerial coverage constraints and employing an empirical aerial channel model. The proposed solution employs RL techniques to train an agent, using the quality of communication links between the UAV and base stations (BSs) as the reward function. Simulation results demonstrate the effectiveness of the proposed method in training the agent and generating feasible UAV path plans. The proposed approach addresses the challenges due to limitations in UAV cellular communications, highlighting the need for investigations and considerations in this area. The RL algorithm efficiently identifies optimal paths, ensuring maximum connectivity with ground BSs to ensure safe and reliable BVLoS flight operation. Moreover, the solution can be deployed as an offline path planning module that can be integrated into future ground control systems (GCS) for UAV operations, enhancing their capabilities and safety. The method holds potential for complex long range UAV applications, advancing the technology in the field of cellular connected UAV path planning.

**arXiv ID:** 2509.13336
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
