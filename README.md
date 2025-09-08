# Agent arXiv Daily

**Last Updated:** 2025-09-08 02:44:40

**Total Papers:** 53

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
<summary><strong>Internet 3.0: Architecture for a Web-of-Agents with it's Algorithm for Ranking Agents</strong> - Rajesh Tembarai Krishnamachari, Srividya Rajesh - [[pdf]](https://arxiv.org/pdf/2509.04979)</summary>

**Abstract:** AI agents -- powered by reasoning-capable large language models (LLMs) and integrated with tools, data, and web search -- are poised to transform the internet into a \emph{Web of Agents}: a machine-native ecosystem where autonomous agents interact, collaborate, and execute tasks at scale. Realizing this vision requires \emph{Agent Ranking} -- selecting agents not only by declared capabilities but by proven, recent performance. Unlike Web~1.0's PageRank, a global, transparent network of agent interactions does not exist; usage signals are fragmented and private, making ranking infeasible without coordination.
We propose \textbf{DOVIS}, a five-layer operational protocol (\emph{Discovery, Orchestration, Verification, Incentives, Semantics}) that enables the collection of minimal, privacy-preserving aggregates of usage and performance across the ecosystem. On this substrate, we implement \textbf{AgentRank-UC}, a dynamic, trust-aware algorithm that combines \emph{usage} (selection frequency) and \emph{competence} (outcome quality, cost, safety, latency) into a unified ranking. We present simulation results and theoretical guarantees on convergence, robustness, and Sybil resistance, demonstrating the viability of coordinated protocols and performance-aware ranking in enabling a scalable, trustworthy Agentic Web.

**arXiv ID:** 2509.04979
</details>

<details>
<summary><strong>UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning</strong> - Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, Aoyan Li, Bo Li, Chen Dun, Chong Liu, Daoguang Zan, Fuxing Leng, Hanbin Wang, Hao Yu, Haobin Chen, Hongyi Guo, Jing Su, Jingjia Huang, Kai Shen, Kaiyu Shi, Lin Yan, Peiyao Zhao, Pengfei Liu, Qinghao Ye, Renjie Zheng, Shulin Xin, Wayne Xin Zhao, Wen Heng, Wenhao Huang, Wenqian Wang, Xiaobo Qin, Yi Lin, Youbin Wu, Zehui Chen, Zihao Wang, Baoquan Zhong, Xinchun Zhang, Xujing Li, Yuanfan Li, Zhongkai Zhao, Chengquan Jiang, Faming Wu, Haotian Zhou, Jinlin Pang, Li Han, Qi Liu, Qianli Ma, Siyao Liu, Songhua Cai, Wenqi Fu, Xin Liu, Yaohui Wang, Zhi Zhang, Bo Zhou, Guoliang Li, Jiajun Shi, Jiale Yang, Jie Tang, Li Li, Qihua Han, Taoran Lu, Woyu Lin, Xiaokang Tong, Xinyao Li, Yichi Zhang, Yu Miao, Zhengxuan Jiang, Zili Li, Ziyuan Zhao, Chenxin Li, Dehua Ma, Feng Lin, Ge Zhang, Haihua Yang, Hangyu Guo, Hongda Zhu, Jiaheng Liu, Junda Du, Kai Cai, Kuanye Li, Lichen Yuan, Meilan Han, Minchao Wang, Shuyue Guo, Tianhao Cheng, Xiaobo Ma, Xiaojun Xiao, Xiaolong Huang, Xinjie Chen, Yidi Du - [[pdf]](https://arxiv.org/pdf/2509.02544)</summary>

**Abstract:** The development of autonomous agents for graphical user interfaces (GUIs) presents major challenges in artificial intelligence. While recent advances in native agent models have shown promise by unifying perception, reasoning, action, and memory through end-to-end learning, open problems remain in data scalability, multi-turn reinforcement learning (RL), the limitations of GUI-only operation, and environment stability. In this technical report, we present UI-TARS-2, a native GUI-centered agent model that addresses these challenges through a systematic training methodology: a data flywheel for scalable data generation, a stabilized multi-turn RL framework, a hybrid GUI environment that integrates file systems and terminals, and a unified sandbox platform for large-scale rollouts. Empirical evaluation demonstrates that UI-TARS-2 achieves significant improvements over its predecessor UI-TARS-1.5. On GUI benchmarks, it reaches 88.2 on Online-Mind2Web, 47.5 on OSWorld, 50.6 on WindowsAgentArena, and 73.3 on AndroidWorld, outperforming strong baselines such as Claude and OpenAI agents. In game environments, it attains a mean normalized score of 59.8 across a 15-game suite-roughly 60% of human-level performance-and remains competitive with frontier proprietary models (e.g., OpenAI o3) on LMGame-Bench. Additionally, the model can generalize to long-horizon information-seeking tasks and software engineering benchmarks, highlighting its robustness across diverse agent tasks. Detailed analyses of training dynamics further provide insights into achieving stability and efficiency in large-scale agent RL. These results underscore UI-TARS-2's potential to advance the state of GUI agents and exhibit strong generalization to real-world interactive scenarios.

**arXiv ID:** 2509.02544
</details>

<details>
<summary><strong>Advancing SLM Tool-Use Capability using Reinforcement Learning</strong> - Dhruvi Paprunia, Vansh Kharidia, Pankti Doshi - [[pdf]](https://arxiv.org/pdf/2509.04518)</summary>

**Abstract:** Large Language Models (LLMs) have progressed beyond simple text creation, and tool use has become increasingly important for complex, real-world tasks. Tool use in LLMs refers to their ability to utilize external resources such as APIs, databases, or software functions to extend their functionality beyond generating this http URL are used for tasks such as performing calculations, making API calls to retrieve the current time and date, and more. This capability enables models to fetch real-time data, execute commands, or solve problems requiring dynamic interaction, making it indispensable for applications like AI agents in virtual assistants, robotic control, or automated workflows.
However, while LLMs are usually adept tool use, their vast resource requirements and computation complexity restrict their use in every use this http URL a result, there is an increasing need for more compact and efficient Small Language Models (SLMs). Small language models (SLMs) struggle in tool use compared to large language models (LLMs). As soon in Table 1. SLMs are typically trained on smaller, more specific datasets, resulting in a narrower knowledge base and limited contextual understanding compared to LLMs.
This research addresses these challenges by using Reinforcement Learning (RL), specifically Group Relative Policy Optimization (GRPO), to enhance tool-use proficiency in SLMs. Unlike conventional fine-tuning approaches that require heavy computation and often lack adaptability, our method provides an efficient, effective solution that significantly boosts SLM tool-use accuracy, increasing their practical utility.

**arXiv ID:** 2509.04518
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>RECAP: REwriting Conversations for Intent Understanding in Agentic Planning</strong> - Kushan Mitra, Dan Zhang, Hannah Kim, Estevam Hruschka - [[pdf]](https://arxiv.org/pdf/2509.04472)</summary>

**Abstract:** Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agent planning in open-domain dialogue systems.

**arXiv ID:** 2509.04472
</details>

<details>
<summary><strong>GUI Agents: A Survey</strong> - Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, Xintong Li, Jing Shi, Hongjie Chen, Viet Dac Lai, Zhouhang Xie, Sungchul Kim, Ruiyi Zhang, Tong Yu, Mehrab Tanjim, Nesreen K. Ahmed, Puneet Mathur, Seunghyun Yoon, Lina Yao, Branislav Kveton, Thien Huu Nguyen, Trung Bui, Tianyi Zhou, Ryan A. Rossi, Franck Dernoncourt - [[pdf]](https://arxiv.org/pdf/2412.13501)</summary>

**Abstract:** Graphical User Interface (GUI) agents, powered by Large Foundation Models, have emerged as a transformative approach to automating human-computer interaction. These agents autonomously interact with digital systems or software applications via GUIs, emulating human actions such as clicking, typing, and navigating visual elements across diverse platforms. Motivated by the growing interest and fundamental importance of GUI agents, we provide a comprehensive survey that categorizes their benchmarks, evaluation metrics, architectures, and training methods. We propose a unified framework that delineates their perception, reasoning, planning, and acting capabilities. Furthermore, we identify important open challenges and discuss key future directions. Finally, this work serves as a basis for practitioners and researchers to gain an intuitive understanding of current progress, techniques, benchmarks, and critical open problems that remain to be addressed.

**arXiv ID:** 2412.13501
</details>

<details>
<summary><strong>Advancing Mobile GUI Agents: A Verifier-Driven Approach to Practical Deployment</strong> - Gaole Dai, Shiqi Jiang, Ting Cao, Yuanchun Li, Yuqing Yang, Rui Tan, Mo Li, Lili Qiu - [[pdf]](https://arxiv.org/pdf/2503.15937)</summary>

**Abstract:** We propose V-Droid, a mobile GUI task automation agent. Unlike previous mobile agents that utilize Large Language Models (LLMs) as generators to directly generate actions at each step, V-Droid employs LLMs as verifiers to evaluate candidate actions before making final decisions. To realize this novel paradigm, we introduce a comprehensive framework for constructing verifier-driven mobile agents: the discretized action space construction coupled with the prefilling-only workflow to accelerate the verification process, the pair-wise progress preference training to significantly enhance the verifier's decision-making capabilities, and the scalable human-agent joint annotation scheme to efficiently collect the necessary data at scale.
V-Droid obtains a substantial task success rate across several public mobile task automation benchmarks: 59.5% on AndroidWorld, 38.3% on AndroidLab, and 49% on MobileAgentBench, surpassing existing agents by 5.2%, 2.1%, and 9%, respectively. Furthermore, V-Droid achieves a remarkably low latency of 4.3s per step, which is 6.1X faster compared with existing mobile agents. The source code is available at this https URL.

**arXiv ID:** 2503.15937
</details>

<details>
<summary><strong>Graph RAG as Human Choice Model: Building a Data-Driven Mobility Agent with Preference Chain</strong> - Kai Hu, Parfait Atchade-Adelomou, Carlo Adornetto, Adrian Mora-Carrero, Luis Alonso-Pastor, Ariel Noyman, Yubo Liu, Kent Larson - [[pdf]](https://arxiv.org/pdf/2508.16172)</summary>

**Abstract:** Understanding human behavior in urban environments is a crucial field within city sciences. However, collecting accurate behavioral data, particularly in newly developed areas, poses significant challenges. Recent advances in generative agents, powered by Large Language Models (LLMs), have shown promise in simulating human behaviors without relying on extensive datasets. Nevertheless, these methods often struggle with generating consistent, context-sensitive, and realistic behavioral outputs. To address these limitations, this paper introduces the Preference Chain, a novel method that integrates Graph Retrieval-Augmented Generation (RAG) with LLMs to enhance context-aware simulation of human behavior in transportation systems. Experiments conducted on the Replica dataset demonstrate that the Preference Chain outperforms standard LLM in aligning with real-world transportation mode choices. The development of the Mobility Agent highlights potential applications of proposed method in urban mobility modeling for emerging cities, personalized travel behavior analysis, and dynamic traffic forecasting. Despite limitations such as slow inference and the risk of hallucination, the method offers a promising framework for simulating complex human behavior in data-scarce environments, where traditional data-driven models struggle due to limited data availability.

**arXiv ID:** 2508.16172
</details>

<details>
<summary><strong>AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection</strong> - Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu - [[pdf]](https://arxiv.org/pdf/2508.01249)</summary>

**Abstract:** Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.

**arXiv ID:** 2508.01249
</details>

<details>
<summary><strong>Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs</strong> - Ilham Wicaksono, Zekun Wu, Theo King, Adriano Koshiyama, Philip Treleaven - [[pdf]](https://arxiv.org/pdf/2509.04802)</summary>

**Abstract:** As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.

**arXiv ID:** 2509.04802
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>Maestro: Joint Graph & Config Optimization for Reliable AI Agents</strong> - Wenxiao Wang, Priyatham Kattakinda, Soheil Feizi - [[pdf]](https://arxiv.org/pdf/2509.04642)</summary>

**Abstract:** Building reliable LLM agents requires decisions at two levels: the graph (which modules exist and how information flows) and the configuration of each node (models, prompts, tools, control knobs). Most existing optimizers tune configurations while holding the graph fixed, leaving structural failure modes unaddressed. We introduce Maestro, a framework-agnostic holistic optimizer for LLM agents that jointly searches over graphs and configurations to maximize agent quality, subject to explicit rollout/token budgets. Beyond numeric metrics, Maestro leverages reflective textual feedback from traces to prioritize edits, improving sample efficiency and targeting specific failure modes. On the IFBench and HotpotQA benchmarks, Maestro consistently surpasses leading prompt optimizers--MIPROv2, GEPA, and GEPA+Merge--by an average of 12%, 4.9%, and 4.86%, respectively; even when restricted to prompt-only optimization, it still leads by 9.65%, 2.37%, and 2.41%. Maestro achieves these results with far fewer rollouts than GEPA. We further show large gains on two applications (interviewer & RAG agents), highlighting that joint graph & configuration search addresses structural failure modes that prompt tuning alone cannot fix.

**arXiv ID:** 2509.04642
</details>

<details>
<summary><strong>COCORELI: Cooperative, Compositional Reconstitution \& Execution of Language Instructions</strong> - Swarnadeep Bhar, Omar Naim, Eleni Metheniti, Bastien Navarri, Loïc Cabannes, Morteza Ezzabady, Nicholas Asher - [[pdf]](https://arxiv.org/pdf/2509.04470)</summary>

**Abstract:** We present COCORELI, a hybrid agent framework designed to tackle the limitations of large language models (LLMs) in tasks requiring: following complex instructions, minimizing hallucination, and spatial reasoning. COCORELI integrates medium-sized LLM agents with novel abstraction mechanisms and a discourse module to parse instructions to in-context learn dynamic, high-level representations of the environment. Experiments on natural collaborative construction tasks show that COCORELI outperforms single-LLM CoT and agentic LLM systems, all using larger LLMs. It manages to largely avoid hallucinations, identify missing information, ask for clarifications, and update its learned objects. COCORELI's abstraction abilities extend beyond ENVIRONMENT, as shown in the ToolBench API completion task.

**arXiv ID:** 2509.04470
</details>

<details>
<summary><strong>Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</strong> - Ryosuke Takata, Atsushi Masumori, Takashi Ikegammi - [[pdf]](https://arxiv.org/pdf/2509.04537)</summary>

**Abstract:** We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60\% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.

**arXiv ID:** 2509.04537
</details>

<details>
<summary><strong>FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction</strong> - Zhiyuan Zeng, Jiashuo Liu, Siyuan Chen, Tianci He, Yali Liao, Yixiao Tian, Jinpeng Wang, Zaiyuan Wang, Yang Yang, Lingyue Yin, Mingren Yin, Zhenwei Zhu, Tianle Cai, Zehui Chen, Jiecao Chen, Yantao Du, Xiang Gao, Jiacheng Guo, Liang Hu, Jianpeng Jiao, Xiangsheng Li, Jingkai Liu, Shuang Ni, Zhoufutu Wen, Ge Zhang, Kaiyuan Zhang, Xin Zhou, Jose Blanchet, Xipeng Qiu, Mengdi Wang, Wenhao Huang - [[pdf]](https://arxiv.org/pdf/2508.11987)</summary>

**Abstract:** Future prediction is a complex task for LLM agents, requiring a high level of analytical thinking, information gathering, contextual understanding, and decision-making under uncertainty. Agents must not only gather and interpret vast amounts of dynamic information but also integrate diverse data sources, weigh uncertainties, and adapt predictions based on emerging trends, just as human experts do in fields like politics, economics, and finance. Despite its importance, no large-scale benchmark exists for evaluating agents on future prediction, largely due to challenges in handling real-time updates and retrieving timely, accurate answers. To address this, we introduce $\textbf{FutureX}$, a dynamic and live evaluation benchmark specifically designed for LLM agents performing future prediction tasks. FutureX is the largest and most diverse live benchmark for future prediction, supporting real-time daily updates and eliminating data contamination through an automated pipeline for question gathering and answer collection. We evaluate 25 LLM/agent models, including those with reasoning, search capabilities, and integration of external tools such as the open-source Deep Research Agent and closed-source Deep Research models. This comprehensive evaluation assesses agents' adaptive reasoning and performance in dynamic environments. Additionally, we provide in-depth analyses of agents' failure modes and performance pitfalls in future-oriented tasks, including the vulnerability to fake web pages and the temporal validity. Our goal is to establish a dynamic, contamination-free evaluation standard that drives the development of LLM agents capable of performing at the level of professional human analysts in complex reasoning and predictive thinking.

**arXiv ID:** 2508.11987
</details>

<details>
<summary><strong>PersonaGym: Evaluating Persona Agents and LLMs</strong> - Vinay Samuel, Henry Peng Zou, Yue Zhou, Shreyas Chaudhari, Ashwin Kalyan, Tanmay Rajpurohit, Ameet Deshpande, Karthik Narasimhan, Vishvak Murahari - [[pdf]](https://arxiv.org/pdf/2407.18416)</summary>

**Abstract:** Persona agents, which are LLM agents conditioned to act according to an assigned persona, enable contextually rich and user aligned interactions across domains like education and healthcare. However, evaluating how faithfully these agents adhere to their personas remains a significant challenge, particularly in free-form settings that demand consistency across diverse, persona-relevant environments. We introduce PersonaGym, the first dynamic evaluation framework for persona agents, and PersonaScore, a human-aligned automatic metric grounded in decision theory that enables comprehensive large-scale evaluation. Our evaluation of 10 leading LLMs across 200 personas and 10,000 questions reveals significant advancement opportunities. For example, GPT-4.1 had the exact same PersonaScore as LLaMA-3-8b despite being a more recent and advanced closed source model. Importantly, increased model size and complexity do not necessarily enhance persona agent capabilities, underscoring the need for algorithmic and architectural innovation toward faithful, performant persona agents.

**arXiv ID:** 2407.18416
</details>

<details>
<summary><strong>AutoPDL: Automatic Prompt Optimization for LLM Agents</strong> - Claudio Spiess, Mandana Vaziri, Louis Mandel, Martin Hirzel - [[pdf]](https://arxiv.org/pdf/2504.04365)</summary>

**Abstract:** The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.21\pm15.46$ percentage points), up to 67.5pp, and reveal that selected prompting strategies vary across models and tasks.

**arXiv ID:** 2504.04365
</details>

<details>
<summary><strong>First Steps Towards Overhearing LLM Agents: A Case Study With Dungeons & Dragons Gameplay</strong> - Andrew Zhu, Evan Osgood, Chris Callison-Burch - [[pdf]](https://arxiv.org/pdf/2505.22809)</summary>

**Abstract:** Much work has been done on conversational LLM agents which directly assist human users with tasks. We present an alternative paradigm for interacting with LLM agents, which we call "overhearing agents". These overhearing agents do not actively participate in conversation -- instead, they "listen in" on human-to-human conversations and perform background tasks or provide suggestions to assist the user. In this work, we explore the overhearing agents paradigm through the lens of Dungeons & Dragons gameplay. We present an in-depth study using large multimodal audio-language models as overhearing agents to assist a Dungeon Master. We perform a human evaluation to examine the helpfulness of such agents and find that some large audio-language models have the emergent ability to perform overhearing agent tasks using implicit audio cues. Finally, we release Python libraries and our project code to support further research into the overhearing agents paradigm at this https URL.

**arXiv ID:** 2505.22809
</details>

<details>
<summary><strong>Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions</strong> - Xinbei Ma, Yiting Wang, Yao Yao, Tongxin Yuan, Aston Zhang, Zhuosheng Zhang, Hai Zhao - [[pdf]](https://arxiv.org/pdf/2408.02544)</summary>

**Abstract:** This paper investigates the faithfulness of multimodal large language model (MLLM) agents in a graphical user interface (GUI) environment, aiming to address the research question of whether multimodal GUI agents can be distracted by environmental context. A general scenario is proposed where both the user and the agent are benign, and the environment, while not malicious, contains unrelated content. A wide range of MLLMs are evaluated as GUI agents using a simulated dataset, following three working patterns with different levels of perception. Experimental results reveal that even the most powerful models, whether generalist agents or specialist GUI agents, are susceptible to distractions. While recent studies predominantly focus on the helpfulness of agents, our findings first indicate that these agents are prone to environmental distractions. Furthermore, we implement an adversarial environment injection and analyze the approach to improve faithfulness, calling for a collective focus on this important topic.

**arXiv ID:** 2408.02544
</details>

<details>
<summary><strong>Reinforcement Learning for Aligning Large Language Models Agents with Interactive Environments: Quantifying and Mitigating Prompt Overfitting</strong> - Mohamed Salim Aissi, Clement Romac, Thomas Carta, Sylvain Lamprier, Pierre-Yves Oudeyer, Olivier Sigaud, Laure Soulier, Nicolas Thome - [[pdf]](https://arxiv.org/pdf/2410.19920)</summary>

**Abstract:** Reinforcement learning (RL) is a promising approach for aligning large language models (LLMs) knowledge with sequential decision-making tasks. However, few studies have thoroughly investigated the impact on LLM agents capabilities of fine-tuning them with RL in a specific environment. In this paper, we propose a novel framework to analyze the sensitivity of LLMs to prompt formulations following RL training in a textual environment. Our findings reveal that the performance of LLMs degrades when faced with prompt formulations different from those used during the RL training phase. Besides, we analyze the source of this sensitivity by examining the model's internal representations and salient tokens. Finally, we propose to use a contrastive loss to mitigate this sensitivity and improve the robustness and generalization capabilities of LLMs.

**arXiv ID:** 2410.19920
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (13 papers)</h2></summary>

<details>
<summary><strong>TalkToAgent: A Human-centric Explanation of Reinforcement Learning Agents with Large Language Models</strong> - Haechang Kim, Hao Chen, Can Li, Jong Min Lee - [[pdf]](https://arxiv.org/pdf/2509.04809)</summary>

**Abstract:** Explainable Reinforcement Learning (XRL) has emerged as a promising approach in improving the transparency of Reinforcement Learning (RL) agents. However, there remains a gap between complex RL policies and domain experts, due to the limited comprehensibility of XRL results and isolated coverage of current XRL approaches that leave users uncertain about which tools to employ. To address these challenges, we introduce TalkToAgent, a multi-agent Large Language Models (LLM) framework that delivers interactive, natural language explanations for RL policies. The architecture with five specialized LLM agents (Coordinator, Explainer, Coder, Evaluator, and Debugger) enables TalkToAgent to automatically map user queries to relevant XRL tools and clarify an agent's actions in terms of either key state variables, expected outcomes, or counterfactual explanations. Moreover, our approach extends previous counterfactual explanations by deriving alternative scenarios from qualitative behavioral descriptions, or even new rule-based policies. We validated TalkToAgent on quadruple-tank process control problem, a well-known nonlinear control benchmark. Results demonstrated that TalkToAgent successfully mapped user queries into XRL tasks with high accuracy, and coder-debugger interactions minimized failures in counterfactual generation. Furthermore, qualitative evaluation confirmed that TalkToAgent effectively interpreted agent's actions and contextualized their meaning within the problem domain.

**arXiv ID:** 2509.04809
</details>

<details>
<summary><strong>Language-Driven Hierarchical Task Structures as Explicit World Models for Multi-Agent Learning</strong> - Brennen Hill - [[pdf]](https://arxiv.org/pdf/2509.04731)</summary>

**Abstract:** The convergence of Language models, Agent models, and World models represents a critical frontier for artificial intelligence. While recent progress has focused on scaling Language and Agent models, the development of sophisticated, explicit World Models remains a key bottleneck, particularly for complex, long-horizon multi-agent tasks. In domains such as robotic soccer, agents trained via standard reinforcement learning in high-fidelity but structurally-flat simulators often fail due to intractable exploration spaces and sparse rewards. This position paper argues that the next frontier in developing capable agents lies in creating environments that possess an explicit, hierarchical World Model. We contend that this is best achieved through hierarchical scaffolding, where complex goals are decomposed into structured, manageable subgoals. Drawing evidence from a systematic review of 2024 research in multi-agent soccer, we identify a clear and decisive trend towards integrating symbolic and hierarchical methods with multi-agent reinforcement learning (MARL). These approaches implicitly or explicitly construct a task-based world model to guide agent learning. We then propose a paradigm shift: leveraging Large Language Models to dynamically generate this hierarchical scaffold, effectively using language to structure the World Model on the fly. This language-driven world model provides an intrinsic curriculum, dense and meaningful learning signals, and a framework for compositional learning, enabling Agent Models to acquire sophisticated, strategic behaviors with far greater sample efficiency. By building environments with explicit, language-configurable task layers, we can bridge the gap between low-level reactive behaviors and high-level strategic team play, creating a powerful and generalizable framework for training the next generation of intelligent agents.

**arXiv ID:** 2509.04731
</details>

<details>
<summary><strong>OSC: Cognitive Orchestration through Dynamic Knowledge Alignment in Multi-Agent LLM Collaboration</strong> - Jusheng Zhang, Yijia Fan, Kaitong Cai, Xiaofei Sun, Keze Wang - [[pdf]](https://arxiv.org/pdf/2509.04876)</summary>

**Abstract:** This paper introduces OSC (Orchestrating Cognitive Synergy), a knowledge-aware adaptive collaboration framework designed to enhance cognitive synergy in multi-agent systems with large language models. While prior work has advanced agent selection and result aggregation, efficient linguistic interactions for deep collaboration among expert agents remain a critical bottleneck. OSC addresses this gap as a pivotal intermediate layer between selection and aggregation, introducing Collaborator Knowledge Models (CKM) to enable each agent to dynamically perceive its collaborators' cognitive states. Through real-time cognitive gap analysis, agents adaptively adjust communication behaviors, including content focus, detail level, and expression style, using learned strategies. Experiments on complex reasoning and problem-solving benchmarks demonstrate that OSC significantly improves task performance and communication efficiency, transforming "parallel-working individuals'' into a "deeply collaborative cognitive team.'' This framework not only optimizes multi-agent collaboration but also offers new insights into LLM agent interaction behaviors.

**arXiv ID:** 2509.04876
</details>

<details>
<summary><strong>LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration</strong> - Zheyan Qu, Wenbo Wang, Zitong Yu, Boquan Sun, Yang Li, Xing Zhang - [[pdf]](https://arxiv.org/pdf/2509.04993)</summary>

**Abstract:** The ubiquitous computing resources in 6G networks provide ideal environments for the fusion of large language models (LLMs) and intelligent services through the agent framework. With auxiliary modules and planning cores, LLM-enabled agents can autonomously plan and take actions to deal with diverse environment semantics and user intentions. However, the limited resources of individual network devices significantly hinder the efficient operation of LLM-enabled agents with complex tool calls, highlighting the urgent need for efficient multi-level device collaborations. To this end, the framework and method of the LLM-enabled multi-agent system with dual-loop terminal-edge collaborations are proposed in 6G networks. Firstly, the outer loop consists of the iterative collaborations between the global agent and multiple sub-agents deployed on edge servers and terminals, where the planning capability is enhanced through task decomposition and parallel sub-task distribution. Secondly, the inner loop utilizes sub-agents with dedicated roles to circularly reason, execute, and replan the sub-task, and the parallel tool calling generation with offloading strategies is incorporated to improve efficiency. The improved task planning capability and task execution efficiency are validated through the conducted case study in 6G-supported urban safety governance. Finally, the open challenges and future directions are thoroughly analyzed in 6G networks, accelerating the advent of the 6G era.

**arXiv ID:** 2509.04993
</details>

<details>
<summary><strong>Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment</strong> - Jiahuan Pei, Fanghua Ye, Xin Sun, Wentao Deng, Koen Hindriks, Junxiao Wang - [[pdf]](https://arxiv.org/pdf/2507.05528)</summary>

**Abstract:** Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced.

**arXiv ID:** 2507.05528
</details>

<details>
<summary><strong>HyperAgent: Generalist Software Engineering Agents to Solve Coding Tasks at Scale</strong> - Huy Nhat Phan, Tien N. Nguyen, Phong X. Nguyen, Nghi D. Q. Bui - [[pdf]](https://arxiv.org/pdf/2409.16299)</summary>

**Abstract:** Large Language Models (LLMs) have revolutionized software engineering (SE), showcasing remarkable proficiency in various coding tasks. Despite recent advancements that have enabled the creation of autonomous software agents utilizing LLMs for end-to-end development tasks, these systems are typically designed for specific SE functions. We introduce HyperAgent, an innovative generalist multi-agent system designed to tackle a wide range of SE tasks across different programming languages by mimicking the workflows of human developers. HyperAgent features four specialized agents-Planner, Navigator, Code Editor, and Executor-capable of handling the entire lifecycle of SE tasks, from initial planning to final verification. HyperAgent sets new benchmarks in diverse SE tasks, including GitHub issue resolution on the renowned SWE-Bench benchmark, outperforming robust baselines. Furthermore, HyperAgent demonstrates exceptional performance in repository-level code generation (RepoExec) and fault localization and program repair (Defects4J), often surpassing state-of-the-art baselines.

**arXiv ID:** 2409.16299
</details>

<details>
<summary><strong>Skill-Aligned Fairness in Multi-Agent Learning for Collaboration in Healthcare</strong> - Promise Osaine Ekpo, Brian La, Thomas Wiener, Saesha Agarwal, Arshia Agrawal, Gonzalo Gonzalez-Pumariega, Lekan P. Molu, Angelique Taylor - [[pdf]](https://arxiv.org/pdf/2508.18708)</summary>

**Abstract:** Fairness in multi-agent reinforcement learning (MARL) is often framed as a workload balance problem, overlooking agent expertise and the structured coordination required in real-world domains. In healthcare, equitable task allocation requires workload balance or expertise alignment to prevent burnout and overuse of highly skilled agents. Workload balance refers to distributing an approximately equal number of subtasks or equalised effort across healthcare workers, regardless of their expertise. We make two contributions to address this problem. First, we propose FairSkillMARL, a framework that defines fairness as the dual objective of workload balance and skill-task alignment. Second, we introduce MARLHospital, a customizable healthcare-inspired environment for modeling team compositions and energy-constrained scheduling impacts on fairness, as no existing simulators are well-suited for this problem. We conducted experiments to compare FairSkillMARL in conjunction with four standard MARL methods, and against two state-of-the-art fairness metrics. Our results suggest that fairness based solely on equal workload might lead to task-skill mismatches and highlight the need for more robust metrics that capture skill-task misalignment. Our work provides tools and a foundation for studying fairness in heterogeneous multi-agent systems where aligning effort with expertise is critical.

**arXiv ID:** 2508.18708
</details>

<details>
<summary><strong>Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense</strong> - Aditya Vikram Singh, Ethan Rathbun, Emma Graham, Lisa Oakley, Simona Boboila, Alina Oprea, Peter Chin - [[pdf]](https://arxiv.org/pdf/2410.17351)</summary>

**Abstract:** Recent advances in multi-agent reinforcement learning (MARL) have created opportunities to solve complex real-world tasks. Cybersecurity is a notable application area, where defending networks against sophisticated adversaries remains a challenging task typically performed by teams of security operators. In this work, we explore novel MARL strategies for building autonomous cyber network defenses that address challenges such as large policy spaces, partial observability, and stealthy, deceptive adversarial strategies. To facilitate efficient and generalized learning, we propose a hierarchical Proximal Policy Optimization (PPO) architecture that decomposes the cyber defense task into specific sub-tasks like network investigation and host recovery. Our approach involves training sub-policies for each sub-task using PPO enhanced with cybersecurity domain expertise. These sub-policies are then leveraged by a master defense policy that coordinates their selection to solve complex network defense tasks. Furthermore, the sub-policies can be fine-tuned and transferred with minimal cost to defend against shifts in adversarial behavior or changes in network settings. We conduct extensive experiments using CybORG Cage 4, the state-of-the-art MARL environment for cyber defense. Comparisons with multiple baselines across different adversaries show that our hierarchical learning approach achieves top performance in terms of convergence speed, episodic return, and several interpretable metrics relevant to cybersecurity, including the fraction of clean machines on the network, precision, and false positives.

**arXiv ID:** 2410.17351
</details>

<details>
<summary><strong>Adaptation of Parameters in Heterogeneous Multi-agent Systems</strong> - Hyungbo Shim, Jin Gyu Lee, B. D. O. Anderson - [[pdf]](https://arxiv.org/pdf/2509.00801)</summary>

**Abstract:** This paper proposes an adaptation mechanism for heterogeneous multi-agent systems to align the agents' internal parameters, based on enforced consensus through strong couplings. Unlike homogeneous systems, where exact consensus is attainable, the heterogeneity in node dynamics precludes perfect synchronization. Nonetheless, previous work has demonstrated that strong coupling can induce approximate consensus, whereby the agents exhibit emergent collective behavior governed by the so-called blended dynamics. Building on this observation, we introduce an adaptation law that gradually aligns the internal parameters of agents without requiring direct parameter communication. The proposed method reuses the same coupling signal employed for state synchronization, which may result in a biologically or sociologically plausible adaptation process. Under a persistent excitation condition, we prove that the linearly parametrized vector fields of the agents converge to each other, thereby making the dynamics asymptotically homogeneous, and leading to exact consensus of the state variables.

**arXiv ID:** 2509.00801
</details>

<details>
<summary><strong>ProST: Progressive Sub-task Training for Pareto-Optimal Multi-agent Systems Using Small Language Models</strong> - Biddut Sarker Bijoy, Mohammad Saqib Hasan, Pegah Alipoormolabashi, Avirup Sil, Aruna Balasubramanian, Niranjan Balasubramanian - [[pdf]](https://arxiv.org/pdf/2509.04508)</summary>

**Abstract:** Multi-agent systems with smaller language models (SLMs) present a viable alternative to single agent systems powered by large language models (LLMs) for addressing complex problems. In this work, we study how these alternatives compare in terms of both effectiveness and efficiency. To study this trade-off, we instantiate single and multi-agent systems for the complex problems in the AppWorld environment using different sized language models.
We find that difficulties with long-trajectory learning in smaller language models (SLMs) limit their performance. Even when trained for specialized roles, SLMs fail to learn all subtasks effectively. To address this issue, we introduce a simple progressive sub-task training strategy, which introduces new sub-tasks progressively in each training epoch. We find that this novel strategy, analogous to instance level curriculum learning, consistently improves the effectiveness of multi-agents at all configurations. Our Pareto analysis shows that fine-tuned multi-agent systems yield better effectiveness-efficiency trade-offs. Additional ablations and analyses shows the importance of our progressive training strategy and its ability to reduce subtask error rates.

**arXiv ID:** 2509.04508
</details>

<details>
<summary><strong>MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading</strong> - Siyi Wu, Junqiao Wang, Zhaoyang Guan, Leyi Zhao, Xinyuan Song, Xinyu Ying, Dexu Yu, Jinhao Wang, Hanlin Zhang, Michele Pak, Yangfan He, Yi Xin, Jianhui Wang, Tianyu Shi - [[pdf]](https://arxiv.org/pdf/2507.20474)</summary>

**Abstract:** Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.

**arXiv ID:** 2507.20474
</details>

<details>
<summary><strong>QCA-MolGAN: Quantum Circuit Associative Molecular GAN with Multi-Agent Reinforcement Learning</strong> - Aaron Mark Thomas, Yu-Cheng Chen, Hubert Okadome Valencia, Sharu Theresa Jose, Ronin Wu - [[pdf]](https://arxiv.org/pdf/2509.05051)</summary>

**Abstract:** Navigating the vast chemical space of molecular structures to design novel drug molecules with desired target properties remains a central challenge in drug discovery. Recent advances in generative models offer promising solutions. This work presents a novel quantum circuit Born machine (QCBM)-enabled Generative Adversarial Network (GAN), called QCA-MolGAN, for generating drug-like molecules. The QCBM serves as a learnable prior distribution, which is associatively trained to define a latent space aligning with high-level features captured by the GANs discriminator. Additionally, we integrate a novel multi-agent reinforcement learning network to guide molecular generation with desired targeted properties, optimising key metrics such as quantitative estimate of drug-likeness (QED), octanol-water partition coefficient (LogP) and synthetic accessibility (SA) scores in conjunction with one another. Experimental results demonstrate that our approach enhances the property alignment of generated molecules with the multi-agent reinforcement learning agents effectively balancing chemical properties.

**arXiv ID:** 2509.05051
</details>

<details>
<summary><strong>Shared Autonomy through LLMs and Reinforcement Learning for Applications to Ship Hull Inspections</strong> - Cristiano Caissutti, Estelle Gerbier, Ehsan Khorrambakht, Paolo Marinelli, Andrea Munafo', Andrea Caiti - [[pdf]](https://arxiv.org/pdf/2509.05042)</summary>

**Abstract:** Shared autonomy is a promising paradigm in robotic systems, particularly within the maritime domain, where complex, high-risk, and uncertain environments necessitate effective human-robot collaboration. This paper investigates the interaction of three complementary approaches to advance shared autonomy in heterogeneous marine robotic fleets: (i) the integration of Large Language Models (LLMs) to facilitate intuitive high-level task specification and support hull inspection missions, (ii) the implementation of human-in-the-loop interaction frameworks in multi-agent settings to enable adaptive and intent-aware coordination, and (iii) the development of a modular Mission Manager based on Behavior Trees to provide interpretable and flexible mission control. Preliminary results from simulation and real-world lake-like environments demonstrate the potential of this multi-layered architecture to reduce operator cognitive load, enhance transparency, and improve adaptive behaviour alignment with human intent. Ongoing work focuses on fully integrating these components, refining coordination mechanisms, and validating the system in operational port scenarios. This study contributes to establishing a modular and scalable foundation for trustworthy, human-collaborative autonomy in safety-critical maritime robotics applications.

**arXiv ID:** 2509.05042
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>i-Mask: An Intelligent Mask for Breath-Driven Activity Recognition</strong> - Ashutosh Kumar Sinha, Ayush Patel, Mitul Dudhat, Pritam Anand, Rahul Mishra - [[pdf]](https://arxiv.org/pdf/2509.04544)</summary>

**Abstract:** The patterns of inhalation and exhalation contain important physiological signals that can be used to anticipate human behavior, health trends, and vital parameters. Human activity recognition (HAR) is fundamentally connected to these vital signs, providing deeper insights into well-being and enabling real-time health monitoring. This work presents i-Mask, a novel HAR approach that leverages exhaled breath patterns captured using a custom-developed mask equipped with integrated sensors. Data collected from volunteers wearing the mask undergoes noise filtering, time-series decomposition, and labeling to train predictive models. Our experimental results validate the effectiveness of the approach, achieving over 95\% accuracy and highlighting its potential in healthcare and fitness applications.

**arXiv ID:** 2509.04544
</details>

<details>
<summary><strong>SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching</strong> - Melik Ozolcer, Sang Won Bae - [[pdf]](https://arxiv.org/pdf/2509.04752)</summary>

**Abstract:** This paper introduces SePA (Search-enhanced Predictive AI Agent), a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance. SePA combines: (1) Individualized models predicting daily stress, soreness, and injury risk from wearable sensor data (28 users, 1260 data points); and (2) A retrieval module that grounds LLM-generated feedback in expert-vetted web content to ensure contextual relevance and reliability. Our predictive models, evaluated with rolling-origin cross-validation and group k-fold cross-validation show that personalized models outperform generalized baselines. In a pilot expert study (n=4), SePA's retrieval-based advice was preferred over a non-retrieval baseline, yielding meaningful practical effect (Cliff's $\delta$=0.3, p=0.05). We also quantify latency performance trade-offs between response quality and speed, offering a transparent blueprint for next-generation, trustworthy personal health informatics systems.

**arXiv ID:** 2509.04752
</details>

<details>
<summary><strong>A Knowledge-Driven Diffusion Policy for End-to-End Autonomous Driving Based on Expert Routing</strong> - Chengkai Xu, Jiaqi Liu, Yicheng Guo, Peng Hang, Jian Sun - [[pdf]](https://arxiv.org/pdf/2509.04853)</summary>

**Abstract:** End-to-end autonomous driving remains constrained by the need to generate multi-modal actions, maintain temporal stability, and generalize across diverse scenarios. Existing methods often collapse multi-modality, struggle with long-horizon consistency, or lack modular adaptability. This paper presents KDP, a knowledge-driven diffusion policy that integrates generative diffusion modeling with a sparse mixture-of-experts routing mechanism. The diffusion component generates temporally coherent and multi-modal action sequences, while the expert routing mechanism activates specialized and reusable experts according to context, enabling modular knowledge composition. Extensive experiments across representative driving scenarios demonstrate that KDP achieves consistently higher success rates, reduced collision risk, and smoother control compared to prevailing paradigms. Ablation studies highlight the effectiveness of sparse expert activation and the Transformer backbone, and activation analyses reveal structured specialization and cross-scenario reuse of experts. These results establish diffusion with expert routing as a scalable and interpretable paradigm for knowledge-driven end-to-end autonomous driving.

**arXiv ID:** 2509.04853
</details>

<details>
<summary><strong>AI Agents for Web Testing: A Case Study in the Wild</strong> - Naimeng Ye, Xiao Yu, Ruize Xu, Tianyi Peng, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2509.05197)</summary>

**Abstract:** Automated web testing plays a critical role in ensuring high-quality user experiences and delivering business value. Traditional approaches primarily focus on code coverage and load testing, but often fall short of capturing complex user behaviors, leaving many usability issues undetected. The emergence of large language models (LLM) and AI agents opens new possibilities for web testing by enabling human-like interaction with websites and a general awareness of common usability problems. In this work, we present WebProber, a prototype AI agent-based web testing framework. Given a URL, WebProber autonomously explores the website, simulating real user interactions, identifying bugs and usability issues, and producing a human-readable report. We evaluate WebProber through a case study of 120 academic personal websites, where it uncovered 29 usability issues--many of which were missed by traditional tools. Our findings highlight agent-based testing as a promising direction while outlining directions for developing next-generation, user-centered testing frameworks.

**arXiv ID:** 2509.05197
</details>

<details>
<summary><strong>The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management</strong> - Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, Yaroslav Zharov - [[pdf]](https://arxiv.org/pdf/2508.21433)</summary>

**Abstract:** Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering ( SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these strategies within SWE-agent on SWE-bench Verified across five diverse model configurations. We find that a simple observation-masking strategy halves cost relative to a raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. For example, with Qwen3-Coder 480B, masking improves solve rate from 53.8% (raw agent) to 54.8%, while remaining competitive with summarization at a lower cost. These results suggest that, at least within SWE-agent on SWE-bench Verified, the most effective and efficient context management can be the simplest. We release code and data for reproducibility

**arXiv ID:** 2508.21433
</details>

<details>
<summary><strong>Robust Model Predictive Control Design for Autonomous Vehicles with Perception-based Observers</strong> - Nariman Niknejad, Gokul S. Sankar, Bahare Kiumarsi, Hamidreza Modares - [[pdf]](https://arxiv.org/pdf/2509.05201)</summary>

**Abstract:** This paper presents a robust model predictive control (MPC) framework that explicitly addresses the non-Gaussian noise inherent in deep learning-based perception modules used for state estimation. Recognizing that accurate uncertainty quantification of the perception module is essential for safe feedback control, our approach departs from the conventional assumption of zero-mean noise quantification of the perception error. Instead, it employs set-based state estimation with constrained zonotopes to capture biased, heavy-tailed uncertainties while maintaining bounded estimation errors. To improve computational efficiency, the robust MPC is reformulated as a linear program (LP), using a Minkowski-Lyapunov-based cost function with an added slack variable to prevent degenerate solutions. Closed-loop stability is ensured through Minkowski-Lyapunov inequalities and contractive zonotopic invariant sets. The largest stabilizing terminal set and its corresponding feedback gain are then derived via an ellipsoidal approximation of the zonotopes. The proposed framework is validated through both simulations and hardware experiments on an omnidirectional mobile robot along with a camera and a convolutional neural network-based perception module implemented within a ROS2 framework. The results demonstrate that the perception-aware MPC provides stable and accurate control performance under heavy-tailed noise conditions, significantly outperforming traditional Gaussian-noise-based designs in terms of both state estimation error bounding and overall control performance.

**arXiv ID:** 2509.05201
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (16 papers)</h2></summary>

<details>
<summary><strong>Cloning a Conversational Voice AI Agent from Call\,Recording Datasets for Telesales</strong> - Krittanon Kaewtawee, Wachiravit Modecrua, Krittin Pachtrachai, Touchapon Kraisingkorn - [[pdf]](https://arxiv.org/pdf/2509.04871)</summary>

**Abstract:** Recent advances in language and speech modelling have made it possible to build autonomous voice assistants that understand and generate human dialogue in real time. These systems are increasingly being deployed in domains such as customer service and healthcare care, where they can automate repetitive tasks, reduce operational costs, and provide constant support around the clock. In this paper, we present a general methodology for cloning a conversational voice AI agent from a corpus of call recordings. Although the case study described in this paper uses telesales data to illustrate the approach, the underlying process generalizes to any domain where call transcripts are available. Our system listens to customers over the telephone, responds with a synthetic voice, and follows a structured playbook learned from top performing human agents. We describe the domain selection, knowledge extraction, and prompt engineering used to construct the agent, integrating automatic speech recognition, a large language model based dialogue manager, and text to speech synthesis into a streaming inference pipeline. The cloned agent is evaluated against human agents on a rubric of 22 criteria covering introduction, product communication, sales drive, objection handling, and closing. Blind tests show that the AI agent approaches human performance in routine aspects of the call while underperforming in persuasion and objection handling. We analyze these shortcomings and refine the prompt accordingly. The paper concludes with design lessons and avenues for future research, including large scale simulation and automated evaluation.

**arXiv ID:** 2509.04871
</details>

<details>
<summary><strong>Emotionally-Aware Agents for Dispute Resolution</strong> - Sushrita Rakshit, James Hale, Kushal Chawla, Jeanne M. Brett, Jonathan Gratch - [[pdf]](https://arxiv.org/pdf/2509.04465)</summary>

**Abstract:** In conflict, people use emotional expressions to shape their counterparts' thoughts, feelings, and actions. This paper explores whether automatic text emotion recognition offers insight into this influence in the context of dispute resolution. Prior work has shown the promise of such methods in negotiations; however, disputes evoke stronger emotions and different social processes. We use a large corpus of buyer-seller dispute dialogues to investigate how emotional expressions shape subjective and objective outcomes. We further demonstrate that large-language models yield considerably greater explanatory power than previous methods for emotion intensity annotation and better match the decisions of human annotators. Findings support existing theoretical models for how emotional expressions contribute to conflict escalation and resolution and suggest that agent-based systems could be useful in managing disputes by recognizing and potentially mitigating emotional escalation.

**arXiv ID:** 2509.04465
</details>

<details>
<summary><strong>Understanding Reinforcement Learning for Model Training, and future directions with GRAPE</strong> - Rohit Patel - [[pdf]](https://arxiv.org/pdf/2509.04501)</summary>

**Abstract:** This paper provides a self-contained, from-scratch, exposition of key algorithms for instruction tuning of models: SFT, Rejection Sampling, REINFORCE, Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO). Explanations of these algorithms often assume prior knowledge, lack critical details, and/or are overly generalized and complex. Here, each method is discussed and developed step by step using simplified and explicit notation focused on LLMs, aiming to eliminate ambiguity and provide a clear and intuitive understanding of the concepts. By minimizing detours into the broader RL literature and connecting concepts to LLMs, we eliminate superfluous abstractions and reduce cognitive overhead. Following this exposition, we provide a literature review of new techniques and approaches beyond those detailed. Finally, new ideas for research and exploration in the form of GRAPE (Generalized Relative Advantage Policy Evolution) are presented.

**arXiv ID:** 2509.04501
</details>

<details>
<summary><strong>Bootstrapping Reinforcement Learning with Sub-optimal Policies for Autonomous Driving</strong> - Zhihao Zhang, Chengyang Peng, Ekim Yurtsever, Keith A. Redmill - [[pdf]](https://arxiv.org/pdf/2509.04712)</summary>

**Abstract:** Automated vehicle control using reinforcement learning (RL) has attracted significant attention due to its potential to learn driving policies through environment interaction. However, RL agents often face training challenges in sample efficiency and effective exploration, making it difficult to discover an optimal driving strategy. To address these issues, we propose guiding the RL driving agent with a demonstration policy that need not be a highly optimized or expert-level controller. Specifically, we integrate a rule-based lane change controller with the Soft Actor Critic (SAC) algorithm to enhance exploration and learning efficiency. Our approach demonstrates improved driving performance and can be extended to other driving scenarios that can similarly benefit from demonstration-based guidance.

**arXiv ID:** 2509.04712
</details>

<details>
<summary><strong>DeGuV: Depth-Guided Visual Reinforcement Learning for Generalization and Interpretability in Manipulation</strong> - Tien Pham, Xinyun Chi, Khang Nguyen, Manfred Huber, Angelo Cangelosi - [[pdf]](https://arxiv.org/pdf/2509.04970)</summary>

**Abstract:** Reinforcement learning (RL) agents can learn to solve complex tasks from visual inputs, but generalizing these learned skills to new environments remains a major challenge in RL application, especially robotics. While data augmentation can improve generalization, it often compromises sample efficiency and training stability. This paper introduces DeGuV, an RL framework that enhances both generalization and sample efficiency. In specific, we leverage a learnable masker network that produces a mask from the depth input, preserving only critical visual information while discarding irrelevant pixels. Through this, we ensure that our RL agents focus on essential features, improving robustness under data augmentation. In addition, we incorporate contrastive learning and stabilize Q-value estimation under augmentation to further enhance sample efficiency and training stability. We evaluate our proposed method on the RL-ViGen benchmark using the Franka Emika robot and demonstrate its effectiveness in zero-shot sim-to-real transfer. Our results show that DeGuV outperforms state-of-the-art methods in both generalization and sample efficiency while also improving interpretability by highlighting the most relevant regions in the visual input

**arXiv ID:** 2509.04970
</details>

<details>
<summary><strong>AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning</strong> - Lang Mei, Zhihan Yang, Chong Chen - [[pdf]](https://arxiv.org/pdf/2508.20368)</summary>

**Abstract:** Recent studies have explored integrating Large Language Models (LLMs) with search engines to leverage both the LLMs' internal pre-trained knowledge and external information. Specially, reinforcement learning (RL) has emerged as a promising paradigm for enhancing LLM reasoning through multi-turn interactions with search engines. However, existing RL-based search agents rely on a single LLM to handle both search planning and question-answering (QA) tasks in an end-to-end manner, which limits their ability to optimize both capabilities simultaneously. In practice, sophisticated AI search systems often employ a large, frozen LLM (e.g., GPT-4, DeepSeek-R1) to ensure high-quality QA. Thus, a more effective and efficient approach is to utilize a small, trainable LLM dedicated to search planning. In this paper, we propose \textbf{AI-SearchPlanner}, a novel reinforcement learning framework designed to enhance the performance of frozen QA models by focusing on search planning. Specifically, our approach introduces three key innovations: 1) Decoupling the Architecture of the Search Planner and Generator, 2) Dual-Reward Alignment for Search Planning, and 3) Pareto Optimization of Planning Utility and Cost, to achieve the objectives. Extensive experiments on real-world datasets demonstrate that AI SearchPlanner outperforms existing RL-based search agents in both effectiveness and efficiency, while exhibiting strong generalization capabilities across diverse frozen QA models and data domains.

**arXiv ID:** 2508.20368
</details>

<details>
<summary><strong>Dynamic Speculative Agent Planning</strong> - Yilin Guan, Wenyue Hua, Qingfeng Lan, Sun Fei, Dujian Ding, Devang Acharya, Chi Wang, William Yang Wang - [[pdf]](https://arxiv.org/pdf/2509.01920)</summary>

**Abstract:** Despite their remarkable success in complex tasks propelling widespread adoption, large language-model-based agents still face critical deployment challenges due to prohibitive latency and inference costs. While recent work has explored various methods to accelerate inference, existing approaches suffer from significant limitations: they either fail to preserve performance fidelity, require extensive offline training of router modules, or incur excessive operational costs. Moreover, they provide minimal user control over the tradeoff between acceleration and other performance metrics. To address these gaps, we introduce Dynamic Speculative Planning (DSP), an asynchronous online reinforcement learning framework that provides lossless acceleration with substantially reduced costs without requiring additional pre-deployment preparation. DSP explicitly optimizes a joint objective balancing end-to-end latency against dollar cost, allowing practitioners to adjust a single parameter that steers the system toward faster responses, cheaper operation, or any point along this continuum. Experiments on two standard agent benchmarks demonstrate that DSP achieves comparable efficiency to the fastest lossless acceleration method while reducing total cost by 30% and unnecessary cost up to 60%. Our code and data are available through this https URL.

**arXiv ID:** 2509.01920
</details>

<details>
<summary><strong>An Arbitration Control for an Ensemble of Diversified DQN variants in Continual Reinforcement Learning</strong> - Wonseo Jang, Dongjae Kim - [[pdf]](https://arxiv.org/pdf/2509.04815)</summary>

**Abstract:** Deep reinforcement learning (RL) models, despite their efficiency in learning an optimal policy in static environments, easily loses previously learned knowledge (i.e., catastrophic forgetting). It leads RL models to poor performance in continual reinforcement learning (CRL) scenarios. To address this, we present an arbitration control mechanism over an ensemble of RL agents. It is motivated by and closely aligned with how humans make decisions in a CRL context using an arbitration control of multiple RL agents in parallel as observed in the prefrontal cortex. We integrated two key ideas into our model: (1) an ensemble of RLs (i.e., DQN variants) explicitly trained to have diverse value functions and (2) an arbitration control that prioritizes agents with higher reliability (i.e., less error) in recent trials. We propose a framework for CRL, an Arbitration Control for an Ensemble of Diversified DQN variants (ACED-DQN). We demonstrate significant performance improvements in both static and continual environments, supported by empirical evidence showing the effectiveness of arbitration control over diversified DQNs during training. In this work, we introduced a framework that enables RL agents to continuously learn, with inspiration from the human brain.

**arXiv ID:** 2509.04815
</details>

<details>
<summary><strong>Quantitative Resilience Modeling for Autonomous Cyber Defense</strong> - Xavier Cadet, Simona Boboila, Edward Koh, Peter Chin, Alina Oprea - [[pdf]](https://arxiv.org/pdf/2503.02780)</summary>

**Abstract:** Cyber resilience is the ability of a system to recover from an attack with minimal impact on system operations. However, characterizing a network's resilience under a cyber attack is challenging, as there are no formal definitions of resilience applicable to diverse network topologies and attack patterns. In this work, we propose a quantifiable formulation of resilience that considers multiple defender operational goals, the criticality of various network resources for daily operations, and provides interpretability to security operators about their system's resilience under attack. We evaluate our approach within the CybORG environment, a reinforcement learning (RL) framework for autonomous cyber defense, analyzing trade-offs between resilience, costs, and prioritization of operational goals. Furthermore, we introduce methods to aggregate resilience metrics across time-variable attack patterns and multiple network topologies, comprehensively characterizing system resilience. Using insights gained from our resilience metrics, we design RL autonomous defensive agents and compare them against several heuristic baselines, showing that proactive network hardening techniques and prompt recovery of compromised machines are critical for effective cyber defenses.

**arXiv ID:** 2503.02780
</details>

<details>
<summary><strong>ACE-RL: Adaptive Constraint-Enhanced Reward for Long-form Generation Reinforcement Learning</strong> - Jianghao Chen, Wei Sun, Qixiang Yin, Lingxing Kong, Zhixing Tan, Jiajun Zhang - [[pdf]](https://arxiv.org/pdf/2509.04903)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable progress in long-context understanding, yet they face significant challenges in high-quality long-form generation. Existing studies primarily suffer from two limitations: (1) A heavy reliance on scarce, high-quality long-form response data for supervised fine-tuning (SFT) or for pairwise preference reward in reinforcement learning (RL). (2) Focus on coarse-grained quality optimization dimensions, such as relevance, coherence, and helpfulness, overlooking the fine-grained specifics inherent to diverse long-form generation scenarios. To address this issue, we propose a framework using Adaptive Constraint-Enhanced reward for long-form generation Reinforcement Learning (ACE-RL). ACE-RL first automatically deconstructs each instruction into a set of fine-grained, adaptive constraint criteria by identifying its underlying intents and demands. Subsequently, we design a reward mechanism that quantifies the quality of long-form responses based on their satisfaction over corresponding constraints, converting subjective quality evaluation into constraint verification. Finally, we utilize reinforcement learning to guide models toward superior long-form generation capabilities. Experimental results demonstrate that our ACE-RL framework significantly outperforms existing SFT and RL baselines by 20.70% and 7.32% on WritingBench, and our top-performing model even surpasses proprietary systems like GPT-4o by 7.10%, providing a more effective training paradigm for LLMs to generate high-quality content across diverse long-form generation scenarios.

**arXiv ID:** 2509.04903
</details>

<details>
<summary><strong>Topology-Aware Graph Reinforcement Learning for Dynamic Routing in Cloud Networks</strong> - Yuxi Wang, Heyao Liu, Guanzi Yao, Nyutian Long, Yue Kang - [[pdf]](https://arxiv.org/pdf/2509.04973)</summary>

**Abstract:** This paper proposes a topology-aware graph reinforcement learning approach to address the routing policy optimization problem in cloud server environments. The method builds a unified framework for state representation and structural evolution by integrating a Structure-Aware State Encoding (SASE) module and a Policy-Adaptive Graph Update (PAGU) mechanism. It aims to tackle the challenges of decision instability and insufficient structural awareness under dynamic topologies. The SASE module models node states through multi-layer graph convolution and structural positional embeddings, capturing high-order dependencies in the communication topology and enhancing the expressiveness of state representations. The PAGU module adjusts the graph structure based on policy behavior shifts and reward feedback, enabling adaptive structural updates in dynamic environments. Experiments are conducted on the real-world GEANT topology dataset, where the model is systematically evaluated against several representative baselines in terms of throughput, latency control, and link balance. Additional experiments, including hyperparameter sensitivity, graph sparsity perturbation, and node feature dimensionality variation, further explore the impact of structure modeling and graph updates on model stability and decision quality. Results show that the proposed method outperforms existing graph reinforcement learning models across multiple performance metrics, achieving efficient and robust routing in dynamic and complex cloud networks.

**arXiv ID:** 2509.04973
</details>

<details>
<summary><strong>Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning</strong> - Bastien Dubail, Stefan Stojanovic, Alexandre Proutière - [[pdf]](https://arxiv.org/pdf/2509.05193)</summary>

**Abstract:** Low-rank structure is a common implicit assumption in many modern reinforcement learning (RL) algorithms. For instance, reward-free and goal-conditioned RL methods often presume that the successor measure admits a low-rank representation. In this work, we challenge this assumption by first remarking that the successor measure itself is not low-rank. Instead, we demonstrate that a low-rank structure naturally emerges in the shifted successor measure, which captures the system dynamics after bypassing a few initial transitions. We provide finite-sample performance guarantees for the entry-wise estimation of a low-rank approximation of the shifted successor measure from sampled entries. Our analysis reveals that both the approximation and estimation errors are primarily governed by the so-called spectral recoverability of the corresponding matrix. To bound this parameter, we derive a new class of functional inequalities for Markov chains that we call Type II Poincaré inequalities and from which we can quantify the amount of shift needed for effective low-rank approximation and estimation. This analysis shows in particular that the required shift depends on decay of the high-order singular values of the shifted successor measure and is hence typically small in practice. Additionally, we establish a connection between the necessary shift and the local mixing properties of the underlying dynamical system, which provides a natural way of selecting the shift. Finally, we validate our theoretical findings with experiments, and demonstrate that shifting the successor measure indeed leads to improved performance in goal-conditioned RL.

**arXiv ID:** 2509.05193
</details>

<details>
<summary><strong>Greener Deep Reinforcement Learning: Analysis of Energy and Carbon Efficiency Across Atari Benchmarks</strong> - Jason Gardner, Ayan Dutta, Swapnoneel Roy, O. Patrick Kreidl, Ladislau Boloni - [[pdf]](https://arxiv.org/pdf/2509.05273)</summary>

**Abstract:** The growing computational demands of deep reinforcement learning (DRL) have raised concerns about the environmental and economic costs of training large-scale models. While algorithmic efficiency in terms of learning performance has been extensively studied, the energy requirements, greenhouse gas emissions, and monetary costs of DRL algorithms remain largely unexplored. In this work, we present a systematic benchmarking study of the energy consumption of seven state-of-the-art DRL algorithms, namely DQN, TRPO, A2C, ARS, PPO, RecurrentPPO, and QR-DQN, implemented using Stable Baselines. Each algorithm was trained for one million steps each on ten Atari 2600 games, and power consumption was measured in real-time to estimate total energy usage, CO2-Equivalent emissions, and electricity cost based on the U.S. national average electricity price. Our results reveal substantial variation in energy efficiency and training cost across algorithms, with some achieving comparable performance while consuming up to 24% less energy (ARS vs. DQN), emitting nearly 68% less CO2, and incurring almost 68% lower monetary cost (QR-DQN vs. RecurrentPPO) than less efficient counterparts. We further analyze the trade-offs between learning performance, training time, energy use, and financial cost, highlighting cases where algorithmic choices can mitigate environmental and economic impact without sacrificing learning performance. This study provides actionable insights for developing energy-aware and cost-efficient DRL practices and establishes a foundation for incorporating sustainability considerations into future algorithmic design and evaluation.

**arXiv ID:** 2509.05273
</details>

<details>
<summary><strong>Deep Reinforcement Learning for Ranking Utility Tuning in the Ad Recommender System at Pinterest</strong> - Xiao Yang, Mehdi Ben Ayed, Longyu Zhao, Fan Zhou, Yuchen Shen, Abe Engle, Jinfeng Zhuang, Ling Leng, Jiajing Xu, Charles Rosenberg, Prathibha Deshikachar - [[pdf]](https://arxiv.org/pdf/2509.05292)</summary>

**Abstract:** The ranking utility function in an ad recommender system, which linearly combines predictions of various business goals, plays a central role in balancing values across the platform, advertisers, and users. Traditional manual tuning, while offering simplicity and interpretability, often yields suboptimal results due to its unprincipled tuning objectives, the vast amount of parameter combinations, and its lack of personalization and adaptability to seasonality. In this work, we propose a general Deep Reinforcement Learning framework for Personalized Utility Tuning (DRL-PUT) to address the challenges of multi-objective optimization within ad recommender systems. Our key contributions include: 1) Formulating the problem as a reinforcement learning task: given the state of an ad request, we predict the optimal hyperparameters to maximize a pre-defined reward. 2) Developing an approach to directly learn an optimal policy model using online serving logs, avoiding the need to estimate a value function, which is inherently challenging due to the high variance and unbalanced distribution of immediate rewards. We evaluated DRL-PUT through an online A/B experiment in Pinterest's ad recommender system. Compared to the baseline manual utility tuning approach, DRL-PUT improved the click-through rate by 9.7% and the long click-through rate by 7.7% on the treated segment. We conducted a detailed ablation study on the impact of different reward definitions and analyzed the personalization aspect of the learned policy model.

**arXiv ID:** 2509.05292
</details>

<details>
<summary><strong>Solving Robotics Tasks with Prior Demonstration via Exploration-Efficient Deep Reinforcement Learning</strong> - Chengyandan Shen, Christoffer Sloth - [[pdf]](https://arxiv.org/pdf/2509.04069)</summary>

**Abstract:** This paper proposes an exploration-efficient Deep Reinforcement Learning with Reference policy (DRLR) framework for learning robotics tasks that incorporates demonstrations. The DRLR framework is developed based on an algorithm called Imitation Bootstrapped Reinforcement Learning (IBRL). We propose to improve IBRL by modifying the action selection module. The proposed action selection module provides a calibrated Q-value, which mitigates the bootstrapping error that otherwise leads to inefficient exploration. Furthermore, to prevent the RL policy from converging to a sub-optimal policy, SAC is used as the RL policy instead of TD3. The effectiveness of our method in mitigating bootstrapping error and preventing overfitting is empirically validated by learning two robotics tasks: bucket loading and open drawer, which require extensive interactions with the environment. Simulation results also demonstrate the robustness of the DRLR framework across tasks with both low and high state-action dimensions, and varying demonstration qualities. To evaluate the developed framework on a real-world industrial robotics task, the bucket loading task is deployed on a real wheel loader. The sim2real results validate the successful deployment of the DRLR framework.

**arXiv ID:** 2509.04069
</details>

<details>
<summary><strong>UAV-Based Intelligent Traffic Surveillance System: Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis</strong> - Ali Khanpour, Tianyi Wang, Afra Vahidi-Shams, Wim Ectors, Farzam Nakhaie, Amirhossein Taheri, Christian Claudel - [[pdf]](https://arxiv.org/pdf/2509.04624)</summary>

**Abstract:** Traffic congestion and violations pose significant challenges for urban mobility and road safety. Traditional traffic monitoring systems, such as fixed cameras and sensor-based methods, are often constrained by limited coverage, low adaptability, and poor scalability. To address these challenges, this paper introduces an advanced unmanned aerial vehicle (UAV)-based traffic surveillance system capable of accurate vehicle detection, classification, tracking, and behavioral analysis in real-world, unconstrained urban environments. The system leverages multi-scale and multi-angle template matching, Kalman filtering, and homography-based calibration to process aerial video data collected from altitudes of approximately 200 meters. A case study in urban area demonstrates robust performance, achieving a detection precision of 91.8%, an F1-score of 90.5%, and tracking metrics (MOTA/MOTP) of 92.1% and 93.7%, respectively. Beyond precise detection, the system classifies five vehicle types and automatically detects critical traffic violations, including unsafe lane changes, illegal double parking, and crosswalk obstructions, through the fusion of geofencing, motion filtering, and trajectory deviation analysis. The integrated analytics module supports origin-destination tracking, vehicle count visualization, inter-class correlation analysis, and heatmap-based congestion modeling. Additionally, the system enables entry-exit trajectory profiling, vehicle density estimation across road segments, and movement direction logging, supporting comprehensive multi-scale urban mobility analytics. Experimental results confirms the system's scalability, accuracy, and practical relevance, highlighting its potential as an enforcement-aware, infrastructure-independent traffic monitoring solution for next-generation smart cities.

**arXiv ID:** 2509.04624
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
