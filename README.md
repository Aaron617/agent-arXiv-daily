# Agent arXiv Daily

**Last Updated:** 2025-10-10 02:01:11

**Total Papers:** 105

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
<summary><strong>AI Knowledge Assist: An Automated Approach for the Creation of Knowledge Bases for Conversational AI Agents</strong> - Md Tahmid Rahman Laskar, Julien Bouvier Tremblay, Xue-Yong Fu, Cheng Chen, Shashi Bhushan TN - [[pdf]](https://arxiv.org/pdf/2510.08149)</summary>

**Abstract:** The utilization of conversational AI systems by leveraging Retrieval Augmented Generation (RAG) techniques to solve customer problems has been on the rise with the rapid progress of Large Language Models (LLMs). However, the absence of a company-specific dedicated knowledge base is a major barrier to the integration of conversational AI systems in contact centers. To this end, we introduce AI Knowledge Assist, a system that extracts knowledge in the form of question-answer (QA) pairs from historical customer-agent conversations to automatically build a knowledge base. Fine-tuning a lightweight LLM on internal data demonstrates state-of-the-art performance, outperforming larger closed-source LLMs. More specifically, empirical evaluation on 20 companies demonstrates that the proposed AI Knowledge Assist system that leverages the LLaMA-3.1-8B model eliminates the cold-start gap in contact centers by achieving above 90% accuracy in answering information-seeking questions. This enables immediate deployment of RAG-powered chatbots.

**arXiv ID:** 2510.08149
</details>

<details>
<summary><strong>Understanding Teen Overreliance on AI Companion Chatbots Through Self-Reported Reddit Narratives</strong> - Mohammad Namvarpour, Brandon Brofsky, Jessica Medina, Mamtaj Akter, Afsaneh Razi - [[pdf]](https://arxiv.org/pdf/2507.15783)</summary>

**Abstract:** AI companion chatbots are increasingly popular with teens, while these interactions are entertaining, they also risk overuse that can potentially disrupt offline daily life. We examined how adolescents describe reliance on AI companions, mapping their experiences onto behavioral addiction frameworks and exploring pathways to disengagement, by analyzing 318 Reddit posts made by users who self-disclosed as 13-17 years old on the this http URL subreddit. We found teens often begin using chatbots for support or creative play, but these activities can deepen into strong attachments marked by conflict, withdrawal, tolerance, relapse, and mood regulation. Reported consequences include sleep loss, academic decline, and strained real-world connections. Disengagement commonly arises when teens recognize harm, re-engage with offline life, or encounter restrictive platform changes. We highlight specific risks of character-based companion chatbots based on teens' perspectives and introduce a design framework (CARE) for guidance for safer systems and setting directions for future teen-centered research.

**arXiv ID:** 2507.15783
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (19 papers)</h2></summary>

<details>
<summary><strong>TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering</strong> - Penghang Liu, Elizabeth Fons, Svitlana Vyetrenko, Daniel Borrajo, Vamsi Potluru, Manuela Veloso - [[pdf]](https://arxiv.org/pdf/2510.07432)</summary>

**Abstract:** Large language models (LLMs) have shown strong abilities in reasoning and problem solving, but recent studies reveal that they still struggle with time series reasoning tasks, where outputs are often affected by hallucination or knowledge leakage. In this work we propose TS-Agent, a time series reasoning agent that leverages LLMs strictly for what they excel at, i.e., gathering evidence and synthesizing it into conclusions through step-by-step reasoning, while delegating the extraction of statistical and structural information to time series analytical tools. Instead of mapping time series into text tokens, images, or embeddings, our agent interacts with raw numeric sequences through atomic operators, records outputs in an explicit evidence log, and iteratively refines its reasoning under the guidance of a self-critic and a final quality gate. This design avoids multi-modal alignment training, preserves the native form of time series, ensures interpretability and verifiability, and mitigates knowledge leakage or hallucination. Empirically, we evaluate the agent on established benchmarks. Our experiments show that TS-Agent achieves performance comparable to state-of-the-art LLMs on understanding benchmarks, and delivers significant improvements on reasoning tasks, where existing models often rely on memorization and fail in zero-shot settings.

**arXiv ID:** 2510.07432
</details>

<details>
<summary><strong>Multimodal Safety Evaluation in Generative Agent Social Simulations</strong> - Alhim Vera, Karen Sanchez, Carlos Hinojosa, Haidar Bin Hamid, Donghoon Kim, Bernard Ghanem - [[pdf]](https://arxiv.org/pdf/2510.07709)</summary>

**Abstract:** Can generative agents be trusted in multimodal environments? Despite advances in large language and vision-language models that enable agents to act autonomously and pursue goals in rich settings, their ability to reason about safety, coherence, and trust across modalities remains limited. We introduce a reproducible simulation framework for evaluating agents along three dimensions: (1) safety improvement over time, including iterative plan revisions in text-visual scenarios; (2) detection of unsafe activities across multiple categories of social situations; and (3) social dynamics, measured as interaction counts and acceptance ratios of social exchanges. Agents are equipped with layered memory, dynamic planning, multimodal perception, and are instrumented with SocialMetrics, a suite of behavioral and structural metrics that quantifies plan revisions, unsafe-to-safe conversions, and information diffusion across networks. Experiments show that while agents can detect direct multimodal contradictions, they often fail to align local revisions with global safety, reaching only a 55 percent success rate in correcting unsafe plans. Across eight simulation runs with three models - Claude, GPT-4o mini, and Qwen-VL - five agents achieved average unsafe-to-safe conversion rates of 75, 55, and 58 percent, respectively. Overall performance ranged from 20 percent in multi-risk scenarios with GPT-4o mini to 98 percent in localized contexts such as fire/heat with Claude. Notably, 45 percent of unsafe actions were accepted when paired with misleading visuals, showing a strong tendency to overtrust images. These findings expose critical limitations in current architectures and provide a reproducible platform for studying multimodal safety, coherence, and social dynamics.

**arXiv ID:** 2510.07709
</details>

<details>
<summary><strong>Haibu Mathematical-Medical Intelligent Agent:Enhancing Large Language Model Reliability in Medical Tasks via Verifiable Reasoning Chains</strong> - Yilun Zhang, Dexing Kong - [[pdf]](https://arxiv.org/pdf/2510.07748)</summary>

**Abstract:** Large Language Models (LLMs) show promise in medicine but are prone to factual and logical errors, which is unacceptable in this high-stakes field. To address this, we introduce the "Haibu Mathematical-Medical Intelligent Agent" (MMIA), an LLM-driven architecture that ensures reliability through a formally verifiable reasoning process. MMIA recursively breaks down complex medical tasks into atomic, evidence-based steps. This entire reasoning chain is then automatically audited for logical coherence and evidence traceability, similar to theorem proving. A key innovation is MMIA's "bootstrapping" mode, which stores validated reasoning chains as "theorems." Subsequent tasks can then be efficiently solved using Retrieval-Augmented Generation (RAG), shifting from costly first-principles reasoning to a low-cost verification model. We validated MMIA across four healthcare administration domains, including DRG/DIP audits and medical insurance adjudication, using expert-validated benchmarks. Results showed MMIA achieved an error detection rate exceeding 98% with a false positive rate below 1%, significantly outperforming baseline LLMs. Furthermore, the RAG matching mode is projected to reduce average processing costs by approximately 85% as the knowledge base matures. In conclusion, MMIA's verifiable reasoning framework is a significant step toward creating trustworthy, transparent, and cost-effective AI systems, making LLM technology viable for critical applications in medicine.

**arXiv ID:** 2510.07748
</details>

<details>
<summary><strong>Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents</strong> - Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, Xiangmin Xu - [[pdf]](https://arxiv.org/pdf/2510.07920)</summary>

**Abstract:** LLM-based financial agents have attracted widespread excitement for their ability to trade like human experts. However, most systems exhibit a "profit mirage": dazzling back-tested returns evaporate once the model's knowledge window ends, because of the inherent information leakage in LLMs. In this paper, we systematically quantify this leakage issue across four dimensions and release FinLake-Bench, a leakage-robust evaluation benchmark. Furthermore, to mitigate this issue, we introduce FactFin, a framework that applies counterfactual perturbations to compel LLM-based agents to learn causal drivers instead of memorized outcomes. FactFin integrates four core components: Strategy Code Generator, Retrieval-Augmented Generation, Monte Carlo Tree Search, and Counterfactual Simulator. Extensive experiments show that our method surpasses all baselines in out-of-sample generalization, delivering superior risk-adjusted performance.

**arXiv ID:** 2510.07920
</details>

<details>
<summary><strong>Haystack Engineering: Context Engineering for Heterogeneous and Agentic Long-Context Evaluation</strong> - Mufei Li, Dongqi Fu, Limei Wang, Si Zhang, Hanqing Zeng, Kaan Sancak, Ruizhong Qiu, Haoyu Wang, Xiaoxin He, Xavier Bresson, Yinglong Xia, Chonglin Sun, Pan Li - [[pdf]](https://arxiv.org/pdf/2510.07414)</summary>

**Abstract:** Modern long-context large language models (LLMs) perform well on synthetic "needle-in-a-haystack" (NIAH) benchmarks, but such tests overlook how noisy contexts arise from biased retrieval and agentic workflows. We argue that haystack engineering is necessary to construct noisy long contexts that faithfully capture key real-world factors -- distraction from heterogeneous biased retrievers and cascading errors in agentic workflows -- to test models' long-context robustness. We instantiate it through HaystackCraft, a new NIAH benchmark built on the full English Wikipedia hyperlink network with multi-hop questions. HaystackCraft evaluates how heterogeneous retrieval strategies (e.g., sparse, dense, hybrid, and graph-based) affect distractor composition, haystack ordering, and downstream LLM performance. HaystackCraft further extends NIAH to dynamic, LLM-dependent settings that simulate agentic operations, where models refine queries, reflect on their past reasonings, and decide when to stop. Experiments with 15 long-context models show that (1) while stronger dense retrievers can introduce more challenging distractors, graph-based reranking simultaneously improves retrieval effectiveness and mitigates more harmful distractors; (2) in agentic tests, even advanced models like Gemini 2.5 Pro and GPT-5 suffer cascading failures from self-generated distractors or struggle to perform early stops. These results highlight persistent challenges in agentic long-context reasoning and establish HaystackCraft as a valuable testbed for future progress.

**arXiv ID:** 2510.07414
</details>

<details>
<summary><strong>LLM4Cell: A Survey of Large Language and Agentic Models for Single-Cell Biology</strong> - Sajib Acharjee Dip, Adrika Zafor, Bikash Kumar Paul, Uddip Acharjee Shuvo, Muhit Islam Emon, Xuan Wang, Liqing Zhang - [[pdf]](https://arxiv.org/pdf/2510.07793)</summary>

**Abstract:** Large language models (LLMs) and emerging agentic frameworks are beginning to transform single-cell biology by enabling natural-language reasoning, generative annotation, and multimodal data integration. However, progress remains fragmented across data modalities, architectures, and evaluation standards. LLM4Cell presents the first unified survey of 58 foundation and agentic models developed for single-cell research, spanning RNA, ATAC, multi-omic, and spatial modalities. We categorize these methods into five families-foundation, text-bridge, spatial, multimodal, epigenomic, and agentic-and map them to eight key analytical tasks including annotation, trajectory and perturbation modeling, and drug-response prediction. Drawing on over 40 public datasets, we analyze benchmark suitability, data diversity, and ethical or scalability constraints, and evaluate models across 10 domain dimensions covering biological grounding, multi-omics alignment, fairness, privacy, and explainability. By linking datasets, models, and evaluation domains, LLM4Cell provides the first integrated view of language-driven single-cell intelligence and outlines open challenges in interpretability, standardization, and trustworthy model development.

**arXiv ID:** 2510.07793
</details>

<details>
<summary><strong>HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation</strong> - Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Kaiyu He, Xinya Du, Zhiyu Chen - [[pdf]](https://arxiv.org/pdf/2510.07794)</summary>

**Abstract:** Agentic RAG is a powerful technique for incorporating external information that LLMs lack, enabling better problem solving and question answering. However, suboptimal search behaviors exist widely, such as over-search (retrieving information already known) and under-search (failing to search when necessary), which leads to unnecessary overhead and unreliable outputs. Current training methods, which typically rely on outcome-based rewards in a RL framework, lack the fine-grained control needed to address these inefficiencies. To overcome this, we introduce Hierarchical Process Rewards for Efficient agentic RAG (HiPRAG), a training methodology that incorporates a fine-grained, knowledge-grounded process reward into the RL training. Our approach evaluates the necessity of each search decision on-the-fly by decomposing the agent's reasoning trajectory into discrete, parsable steps. We then apply a hierarchical reward function that provides an additional bonus based on the proportion of optimal search and non-search steps, on top of commonly used outcome and format rewards. Experiments on the Qwen2.5 and Llama-3.2 models across seven diverse QA benchmarks show that our method achieves average accuracies of 65.4% (3B) and 67.2% (7B). This is accomplished while improving search efficiency, reducing the over-search rate to just 2.3% and concurrently lowering the under-search rate. These results demonstrate the efficacy of optimizing the reasoning process itself, not just the final outcome. Further experiments and analysis demonstrate that HiPRAG shows good generalizability across a wide range of RL algorithms, model families, sizes, and types. This work demonstrates the importance and potential of fine-grained control through RL, for improving the efficiency and optimality of reasoning for search agents.

**arXiv ID:** 2510.07794
</details>

<details>
<summary><strong>NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions</strong> - Haolin Yang, Yuxing Long, Zhuoyuan Yu, Zihan Yang, Minghan Wang, Jiapeng Xu, Yihan Wang, Ziyan Yu, Wenzhe Cai, Lei Kang, Hao Dong - [[pdf]](https://arxiv.org/pdf/2510.08173)</summary>

**Abstract:** Instruction-following navigation is a key step toward embodied intelligence. Prior benchmarks mainly focus on semantic understanding but overlook systematically evaluating navigation agents' spatial perception and reasoning capabilities. In this work, we introduce the NavSpace benchmark, which contains six task categories and 1,228 trajectory-instruction pairs designed to probe the spatial intelligence of navigation agents. On this benchmark, we comprehensively evaluate 22 navigation agents, including state-of-the-art navigation models and multimodal large language models. The evaluation results lift the veil on spatial intelligence in embodied navigation. Furthermore, we propose SNav, a new spatially intelligent navigation model. SNav outperforms existing navigation agents on NavSpace and real robot tests, establishing a strong baseline for future work.

**arXiv ID:** 2510.08173
</details>

<details>
<summary><strong>MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning</strong> - Tajamul Ashraf, Umair Nawaz, Abdelrahman M. Shaker, Rao Anwer, Philip Torr, Fahad Shahbaz Khan, Salman Khan - [[pdf]](https://arxiv.org/pdf/2510.08567)</summary>

**Abstract:** Vision language models (VLMs) are increasingly deployed as controllers with access to external tools for complex reasoning and decision-making, yet their effectiveness remains limited by the scarcity of high-quality multimodal trajectories and the cost of manual annotation. We address this challenge with a vision-centric agent tuning framework that automatically synthesizes multimodal trajectories, generates step-wise preference pairs, and trains a VLM controller for robust tool-use reasoning. Our pipeline first constructs M-TRACE, a large-scale dataset of 28.5K multimodal tasks with 177K verified trajectories, enabling imitation-based trajectory tuning. Building on this, we develop MATRIX Agent, a controller finetuned on M-TRACE for step-wise tool reasoning. To achieve finer alignment, we further introduce Pref-X, a set of 11K automatically generated preference pairs, and optimize MATRIX on it via step-wise preference learning. Across three benchmarks, Agent-X, GTA, and GAIA, MATRIX consistently surpasses both open- and closed-source VLMs, demonstrating scalable and effective multimodal tool use. Our data and code is avaliable at this https URL.

**arXiv ID:** 2510.08567
</details>

<details>
<summary><strong>BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation</strong> - Rocktim Jyoti Das, Harsh Singh, Diana Turmakhan, Muhammad Abdullah Sohail, Mingfei Han, Preslav Nakov, Fabio Pizzati, Ivan Laptev - [[pdf]](https://arxiv.org/pdf/2510.08572)</summary>

**Abstract:** Scaling data and models has played a pivotal role in the remarkable progress of computer vision and language. Inspired by these domains, recent efforts in robotics have similarly focused on scaling both data and model size to develop more generalizable and robust policies. However, unlike vision and language, robotics lacks access to internet-scale demonstrations across diverse robotic tasks and environments. As a result, the scale of existing datasets typically suffers from the need for manual data collection and curation. To address this problem, here we propose BLAZER, a framework that learns manipulation policies from automatically generated training data. We build on the zero-shot capabilities of LLM planners and automatically generate demonstrations for diverse manipulation tasks in simulation. Successful examples are then used to finetune an LLM and to improve its planning capabilities without human supervision. Notably, while BLAZER training requires access to the simulator's state, we demonstrate direct transfer of acquired skills to sensor-based manipulation. Through extensive experiments, we show BLAZER to significantly improve zero-shot manipulation in both simulated and real environments. Moreover, BLAZER improves on tasks outside of its training pool and enables downscaling of LLM models. Our code and data will be made publicly available on the project page.

**arXiv ID:** 2510.08572
</details>

<details>
<summary><strong>Multiple Memory Systems for Enhancing the Long-term Memory of Agent</strong> - Gaoke Zhang, Bo Wang, Yunlong Ma, Dongming Zhao, Zifei Yu - [[pdf]](https://arxiv.org/pdf/2508.15294)</summary>

**Abstract:** An agent powered by large language models have achieved impressive results, but effectively handling the vast amounts of historical data generated during interactions remains a challenge. The current approach is to design a memory module for the agent to process these data. However, existing methods, such as MemoryBank and A-MEM, have poor quality of stored memory content, which affects recall performance and response quality. In order to better construct high-quality long-term memory content, we have designed a multiple memory system (MMS) inspired by cognitive psychology theory. The system processes short-term memory to multiple long-term memory fragments, and constructs retrieval memory units and contextual memory units based on these fragments, with a one-to-one correspondence between the two. During the retrieval phase, MMS will match the most relevant retrieval memory units based on the user's query. Then, the corresponding contextual memory units is obtained as the context for the response stage to enhance knowledge, thereby effectively utilizing historical data. Experiments on LoCoMo dataset compared our method with three others, proving its effectiveness. Ablation studies confirmed the rationality of our memory units. We also analyzed the robustness regarding the number of selected memory segments and the storage overhead, demonstrating its practical value.

**arXiv ID:** 2508.15294
</details>

<details>
<summary><strong>Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents</strong> - Zonghan Yang, Shengjie Wang, Kelin Fu, Wenyang He, Weimin Xiong, Yibo Liu, Yibo Miao, Bofei Gao, Yejie Wang, Yingwei Ma, Yanhao Li, Yue Liu, Zhenxing Hu, Kaitai Zhang, Shuyi Wang, Huarong Chen, Flood Sung, Yang Liu, Yang Gao, Zhilin Yang, Tianyu Liu - [[pdf]](https://arxiv.org/pdf/2509.23045)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly applied to software engineering (SWE), with SWE-bench as a key benchmark. Solutions are split into SWE-Agent frameworks with multi-turn interactions and workflow-based Agentless methods with single-turn verifiable steps. We argue these paradigms are not mutually exclusive: reasoning-intensive Agentless training induces skill priors, including localization, code edit, and self-reflection that enable efficient and effective SWE-Agent adaptation. In this work, we first curate the Agentless training recipe and present Kimi-Dev, an open-source SWE LLM achieving 60.4\% on SWE-bench Verified, the best among workflow approaches. With additional SFT adaptation on 5k publicly-available trajectories, Kimi-Dev powers SWE-Agents to 48.6\% pass@1, on par with that of Claude 3.5 Sonnet (241022 version). These results show that structured skill priors from Agentless training can bridge workflow and agentic frameworks for transferable coding agents.

**arXiv ID:** 2509.23045
</details>

<details>
<summary><strong>Building Resource-Constrained Language Agents: A Korean Case Study on Chemical Toxicity Information</strong> - Hojun Cho, Donghu Kim, Soyoung Yang, Chan Lee, Hunjoo Lee, Jaegul Choo - [[pdf]](https://arxiv.org/pdf/2503.17753)</summary>

**Abstract:** Language agents powered by large language models (LLMs) face significant deployment challenges in resource-constrained environments, particularly for specialized domains and less-common languages. This paper presents Tox-chat, a Korean chemical toxicity information agent devised within these limitations. We propose two key innovations: a context-efficient architecture that reduces token consumption through hierarchical section search, and a scenario-based dialogue generation methodology that effectively distills tool-using capabilities from larger models. Experimental evaluations demonstrate that our fine-tuned 8B parameter model substantially outperforms both untuned models and baseline approaches, in terms of DB faithfulness and preference. Our work offers valuable insights for researchers developing domain-specific language agents under practical constraints.

**arXiv ID:** 2503.17753
</details>

<details>
<summary><strong>Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain</strong> - Yue Li, Ran Tao, Derek Hommel, Yusuf Denizay Dönder, Sungyong Chang, David Mimno, Unso Eun Seo Jo - [[pdf]](https://arxiv.org/pdf/2510.07309)</summary>

**Abstract:** In the business domain, where data-driven decision making is crucial, text-to-SQL is fundamental for easy natural language access to structured data. While recent LLMs have achieved strong performance in code generation, existing text-to-SQL benchmarks remain focused on factual retrieval of past records. We introduce CORGI, a new benchmark specifically designed for real-world business contexts. CORGI is composed of synthetic databases inspired by enterprises such as Doordash, Airbnb, and Lululemon. It provides questions across four increasingly complex categories of business queries: descriptive, explanatory, predictive, and recommendational. This challenge calls for causal reasoning, temporal forecasting, and strategic recommendation, reflecting multi-level and multi-step agentic intelligence. We find that LLM performance drops on high-level questions, struggling to make accurate predictions and offer actionable plans. Based on execution success rate, the CORGI benchmark is about 21% more difficult than the BIRD benchmark. This highlights the gap between popular LLMs and the need for real-world business intelligence. We release a public dataset and evaluation framework, and a website for public submissions.

**arXiv ID:** 2510.07309
</details>

<details>
<summary><strong>A Modality-Aware Cooperative Co-Evolutionary Framework for Multimodal Graph Neural Architecture Search</strong> - Sixuan Wang, Jiao Yin, Jinli Cao, Mingjian Tang, Yong-Feng Ge - [[pdf]](https://arxiv.org/pdf/2510.07325)</summary>

**Abstract:** Co-exploitation attacks on software vulnerabilities pose severe risks to enterprises, a threat that can be mitigated by analyzing heterogeneous and multimodal vulnerability data. Multimodal graph neural networks (MGNNs) are well-suited to integrate complementary signals across modalities, thereby improving attack-prediction accuracy. However, designing an effective MGNN architecture is challenging because it requires coordinating modality-specific components at each layer, which is infeasible through manual tuning. Genetic algorithm (GA)-based graph neural architecture search (GNAS) provides a natural solution, yet existing methods are confined to single modalities and overlook modality heterogeneity. To address this limitation, we propose a modality-aware cooperative co-evolutionary algorithm for multimodal graph neural architecture search, termed MACC-MGNAS. First, we develop a modality-aware cooperative co-evolution (MACC) framework under a divide-and-conquer paradigm: a coordinator partitions a global chromosome population into modality-specific gene groups, local workers evolve them independently, and the coordinator reassembles chromosomes for joint evaluation. This framework effectively captures modality heterogeneity ignored by single-modality GNAS. Second, we introduce a modality-aware dual-track surrogate (MADTS) method to reduce evaluation cost and accelerate local gene evolution. Third, we design a similarity-based population diversity indicator (SPDI) strategy to adaptively balance exploration and exploitation, thereby accelerating convergence and avoiding local optima. On a standard vulnerabilities co-exploitation (VulCE) dataset, MACC-MGNAS achieves an F1-score of 81.67% within only 3 GPU-hours, outperforming the state-of-the-art competitor by 8.7% F1 while reducing computation cost by 27%.

**arXiv ID:** 2510.07325
</details>

<details>
<summary><strong>Injecting Hallucinations in Autonomous Vehicles: A Component-Agnostic Safety Evaluation Framework</strong> - Alexandre Moreira Nascimento, Gabriel Kenji Godoy Shimanuki, Lúcio Flavio Vismari, João Batista Camargo Jr, Jorge Rady de Almeida Jr, Paulo Sergio Cugnasca, Anna Carolina Muller Queiroz, Jeremy Noah Bailenson - [[pdf]](https://arxiv.org/pdf/2510.07749)</summary>

**Abstract:** Perception failures in autonomous vehicles (AV) remain a major safety concern because they are the basis for many accidents. To study how these failures affect safety, researchers typically inject artificial faults into hardware or software components and observe the outcomes. However, existing fault injection studies often target a single sensor or machine perception (MP) module, resulting in siloed frameworks that are difficult to generalize or integrate into unified simulation environments. This work addresses that limitation by reframing perception failures as hallucinations, false perceptions that distort an AV situational awareness and may trigger unsafe control actions. Since hallucinations describe only observable effects, this abstraction enables analysis independent of specific sensors or algorithms, focusing instead on how their faults manifest along the MP pipeline. Building on this concept, we propose a configurable, component-agnostic hallucination injection framework that induces six plausible hallucination types in an iterative open-source simulator. More than 18,350 simulations were executed in which hallucinations were injected while AVs crossed an unsignalized transverse street with traffic. The results statistically validate the framework and quantify the impact of each hallucination type on collisions and near misses. Certain hallucinations, such as perceptual latency and drift, significantly increase the risk of collision in the scenario tested, validating the proposed paradigm can stress the AV system safety. The framework offers a scalable, statistically validated, component agnostic, and fully interoperable toolset that simplifies and accelerates AV safety validations, even those with novel MP architectures and components. It can potentially reduce the time-to-market of AV and lay the foundation for future research on fault tolerance, and resilient AV design.

**arXiv ID:** 2510.07749
</details>

<details>
<summary><strong>Scalable Offline Metrics for Autonomous Driving</strong> - Animikh Aich, Adwait Kulkarni, Eshed Ohn-Bar - [[pdf]](https://arxiv.org/pdf/2510.08571)</summary>

**Abstract:** Real-World evaluation of perception-based planning models for robotic systems, such as autonomous vehicles, can be safely and inexpensively conducted offline, i.e., by computing model prediction error over a pre-collected validation dataset with ground-truth annotations. However, extrapolating from offline model performance to online settings remains a challenge. In these settings, seemingly minor errors can compound and result in test-time infractions or collisions. This relationship is understudied, particularly across diverse closed-loop metrics and complex urban maneuvers. In this work, we revisit this undervalued question in policy evaluation through an extensive set of experiments across diverse conditions and metrics. Based on analysis in simulation, we find an even worse correlation between offline and online settings than reported by prior studies, casting doubts on the validity of current evaluation practices and metrics for driving policies. Next, we bridge the gap between offline and online evaluation. We investigate an offline metric based on epistemic uncertainty, which aims to capture events that are likely to cause errors in closed-loop settings. The resulting metric achieves over 13% improvement in correlation compared to previous offline metrics. We further validate the generalization of our findings beyond the simulation environment in real-world settings, where even greater gains are observed.

**arXiv ID:** 2510.08571
</details>

<details>
<summary><strong>A Multimodal Depth-Aware Method For Embodied Reference Understanding</strong> - Fevziye Irem Eyiokur, Dogucan Yaman, Hazım Kemal Ekenel, Alexander Waibel - [[pdf]](https://arxiv.org/pdf/2510.08278)</summary>

**Abstract:** Embodied Reference Understanding requires identifying a target object in a visual scene based on both language instructions and pointing cues. While prior works have shown progress in open-vocabulary object detection, they often fail in ambiguous scenarios where multiple candidate objects exist in the scene. To address these challenges, we propose a novel ERU framework that jointly leverages LLM-based data augmentation, depth-map modality, and a depth-aware decision module. This design enables robust integration of linguistic and embodied cues, improving disambiguation in complex or cluttered environments. Experimental results on two datasets demonstrate that our approach significantly outperforms existing baselines, achieving more accurate and reliable referent detection.

**arXiv ID:** 2510.08278
</details>

<details>
<summary><strong>ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving</strong> - Zhiyu Zheng, Shaoyu Chen, Haoran Yin, Xinbang Zhang, Jialv Zou, Xinggang Wang, Qian Zhang, Lefei Zhang - [[pdf]](https://arxiv.org/pdf/2510.08562)</summary>

**Abstract:** End-to-end autonomous driving (E2EAD) systems, which learn to predict future trajectories directly from sensor data, are fundamentally challenged by the inherent spatio-temporal imbalance of trajectory data. This imbalance creates a significant optimization burden, causing models to learn spurious correlations instead of causal inference, while also prioritizing uncertain, distant predictions, thereby compromising immediate safety. To address these issues, we propose ResAD, a novel Normalized Residual Trajectory Modeling framework. Instead of predicting the future trajectory directly, our approach reframes the learning task to predict the residual deviation from a deterministic inertial reference. The inertial reference serves as a counterfactual, forcing the model to move beyond simple pattern recognition and instead identify the underlying causal factors (e.g., traffic rules, obstacles) that necessitate deviations from a default, inertially-guided path. To deal with the optimization imbalance caused by uncertain, long-term horizons, ResAD further incorporates Point-wise Normalization of the predicted residual. It re-weights the optimization objective, preventing large-magnitude errors associated with distant, uncertain waypoints from dominating the learning signal. Extensive experiments validate the effectiveness of our framework. On the NAVSIM benchmark, ResAD achieves a state-of-the-art PDMS of 88.6 using a vanilla diffusion policy with only two denoising steps, demonstrating that our approach significantly simplifies the learning task and improves model performance. The code will be released to facilitate further research.

**arXiv ID:** 2510.08562
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>AutoQual: An LLM Agent for Automated Discovery of Interpretable Features for Review Quality Assessment</strong> - Xiaochong Lan, Jie Feng, Yinxing Liu, Xinlei Shi, Yong Li - [[pdf]](https://arxiv.org/pdf/2510.08081)</summary>

**Abstract:** Ranking online reviews by their intrinsic quality is a critical task for e-commerce platforms and information services, impacting user experience and business outcomes. However, quality is a domain-dependent and dynamic concept, making its assessment a formidable challenge. Traditional methods relying on hand-crafted features are unscalable across domains and fail to adapt to evolving content patterns, while modern deep learning approaches often produce black-box models that lack interpretability and may prioritize semantics over quality. To address these challenges, we propose AutoQual, an LLM-based agent framework that automates the discovery of interpretable features. While demonstrated on review quality assessment, AutoQual is designed as a general framework for transforming tacit knowledge embedded in data into explicit, computable features. It mimics a human research process, iteratively generating feature hypotheses through reflection, operationalizing them via autonomous tool implementation, and accumulating experience in a persistent memory. We deploy our method on a large-scale online platform with a billion-level user base. Large-scale A/B testing confirms its effectiveness, increasing average reviews viewed per user by 0.79% and the conversion rate of review readers by 0.27%.

**arXiv ID:** 2510.08081
</details>

<details>
<summary><strong>CaRT: Teaching LLM Agents to Know When They Know Enough</strong> - Grace Liu, Yuxiao Qu, Jeff Schneider, Aarti Singh, Aviral Kumar - [[pdf]](https://arxiv.org/pdf/2510.08517)</summary>

**Abstract:** Many tasks require learned models to strategically gather relevant information over multiple rounds of interaction before actually acting on a task. Strategic information gathering requires models to know not only how to effectively acquire information, but also when to stop gathering information and make a decision, in order to avoid overthinking or getting derailed when acting. In this paper, we formalize this problem and introduce Counterfactuals and Reasoning for Termination (CaRT), an approach for teaching LLMs when to stop seeking information. To appropriately learn when to terminate, CaRT fine-tunes LLMs using counterfactual pairs of trajectories, one where termination is appropriate and a minimally modified version of the same trajectory where it is not. It trains the LLM to explain the rationale for the termination decision in either case via verbal reasoning, and imbues this capability into the base LLM via fine-tuning. We instantiate CaRT in two domains: interactive medical diagnosis and math problem solving. In both domains, we find that CaRT improves the efficiency of information gathering and task success rate compared to other fine-tuning methods.

**arXiv ID:** 2510.08517
</details>

<details>
<summary><strong>Self-Improving LLM Agents at Test-Time</strong> - Emre Can Acikgoz, Cheng Qian, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur - [[pdf]](https://arxiv.org/pdf/2510.07841)</summary>

**Abstract:** One paradigm of language model (LM) fine-tuning relies on creating large training datasets, under the assumption that high quantity and diversity will enable models to generalize to novel tasks after post-training. In practice, gathering large sets of data is inefficient, and training on them is prohibitively expensive; worse, there is no guarantee that the resulting model will handle complex scenarios or generalize better. Moreover, existing techniques rarely assess whether a training sample provides novel information or is redundant with the knowledge already acquired by the model, resulting in unnecessary costs. In this work, we explore a new test-time self-improvement method to create more effective and generalizable agentic LMs on-the-fly. The proposed algorithm can be summarized in three steps: (i) first it identifies the samples that model struggles with (self-awareness), (ii) then generates similar examples from detected uncertain samples (self-data augmentation), and (iii) uses these newly generated samples at test-time fine-tuning (self-improvement). We study two variants of this approach: Test-Time Self-Improvement (TT-SI), where the same model generates additional training examples from its own uncertain cases and then learns from them, and contrast this approach with Test-Time Distillation (TT-D), where a stronger model generates similar examples for uncertain cases, enabling student to adapt using distilled supervision. Empirical evaluations across different agent benchmarks demonstrate that TT-SI improves the performance with +5.48% absolute accuracy gain on average across all benchmarks and surpasses other standard learning methods, yet using 68x less training samples. Our findings highlight the promise of TT-SI, demonstrating the potential of self-improvement algorithms at test-time as a new paradigm for building more capable agents toward self-evolution.

**arXiv ID:** 2510.07841
</details>

<details>
<summary><strong>Learning on the Job: An Experience-Driven Self-Evolving Agent for Long-Horizon Tasks</strong> - Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Botian Shi, Yu Qiao, Haifeng Li - [[pdf]](https://arxiv.org/pdf/2510.08002)</summary>

**Abstract:** Large Language Models have demonstrated remarkable capabilities across diverse domains, yet significant challenges persist when deploying them as AI agents for real-world long-horizon tasks. Existing LLM agents suffer from a critical limitation: they are test-time static and cannot learn from experience, lacking the ability to accumulate knowledge and continuously improve on the job. To address this challenge, we propose MUSE, a novel agent framework that introduces an experience-driven, self-evolving system centered around a hierarchical Memory Module. MUSE organizes diverse levels of experience and leverages them to plan and execute long-horizon tasks across multiple applications. After each sub-task execution, the agent autonomously reflects on its trajectory, converting the raw trajectory into structured experience and integrating it back into the Memory Module. This mechanism enables the agent to evolve beyond its static pretrained parameters, fostering continuous learning and self-evolution. We evaluate MUSE on the long-horizon productivity benchmark TAC. It achieves new SOTA performance by a significant margin using only a lightweight Gemini-2.5 Flash model. Sufficient Experiments demonstrate that as the agent autonomously accumulates experience, it exhibits increasingly superior task completion capabilities, as well as robust continuous learning and self-evolution capabilities. Moreover, the accumulated experience from MUSE exhibits strong generalization properties, enabling zero-shot improvement on new tasks. MUSE establishes a new paradigm for AI agents capable of real-world productivity task automation.

**arXiv ID:** 2510.08002
</details>

<details>
<summary><strong>From Correction to Mastery: Reinforced Distillation of Large Language Model Agents</strong> - Yuanjie Lyu, Chengyu Wang, Jun Huang, Tong Xu - [[pdf]](https://arxiv.org/pdf/2509.14257)</summary>

**Abstract:** Large Language Model agents excel at solving complex tasks through iterative reasoning and tool use, but typically depend on ultra-large, costly backbones. Existing distillation approaches train smaller students to imitate full teacher trajectories, yet reasoning and knowledge gaps between the teacher and student can cause compounding errors. We propose SCoRe, a student-centered framework in which the student generates training trajectories and the teacher corrects only the earliest error, producing training data matched to the student's ability and exposing specific weaknesses. The student is first fine-tuned on corrected trajectories. Subsequently, short-horizon reinforcement learning starts from the verified prefix preceding the earliest error, with target rewards assigned at that step. This design encourages autonomous problem-solving beyond imitation and enhances training stability. On 12 challenging benchmarks, a 7B-parameter student distilled with SCoRe matches the agentic performance of a 72B-parameter teacher.

**arXiv ID:** 2509.14257
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (36 papers)</h2></summary>

<details>
<summary><strong>ProSEA: Problem Solving via Exploration Agents</strong> - William Nguyen, Vinh Luong, Christopher Nguyen - [[pdf]](https://arxiv.org/pdf/2510.07423)</summary>

**Abstract:** Large language models (LLMs) have empowered AI agents to tackle increasingly complex tasks. However, most existing agents remain limited to static planning and brittle interactions, falling short of true collaboration or adaptive reasoning. We introduce ProSEA, a modular, general-purpose multi-agent framework designed for iterative problem solving through exploration and plan evolution. ProSEA features a hierarchical architecture in which a Manager Agent orchestrates domain-specialized Expert Agents, decomposes tasks, and adaptively replans based on structured feedback from failed attempts. Unlike prior systems, ProSEA agents report not only success or failure but also detailed reasons for failure and newly discovered constraints, enabling dynamic plan refinement informed by exploratory traces. The framework operates autonomously but supports seamless integration with human collaborators when needed. Experiments on the challenging FinanceBench benchmark demonstrate that ProSEA, even without human feedback, outperforms state-of-the-art baselines and achieves robust performance across reasoning-heavy tasks. These results underscore ProSEA's potential as a foundation for more transparent, adaptive, and human-aligned AI agents.

**arXiv ID:** 2510.07423
</details>

<details>
<summary><strong>L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)</strong> - Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu - [[pdf]](https://arxiv.org/pdf/2510.07363)</summary>

**Abstract:** The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.

**arXiv ID:** 2510.07363
</details>

<details>
<summary><strong>CompassLLM: A Multi-Agent Approach toward Geo-Spatial Reasoning for Popular Path Query</strong> - Md. Nazmul Islam Ananto, Shamit Fatin, Mohammed Eunus Ali, Md Rizwan Parvez - [[pdf]](https://arxiv.org/pdf/2510.07516)</summary>

**Abstract:** The popular path query - identifying the most frequented routes between locations from historical trajectory data - has important applications in urban planning, navigation optimization, and travel recommendations. While traditional algorithms and machine learning approaches have achieved success in this domain, they typically require model training, parameter tuning, and retraining when accommodating data updates. As Large Language Models (LLMs) demonstrate increasing capabilities in spatial and graph-based reasoning, there is growing interest in exploring how these models can be applied to geo-spatial problems.
We introduce CompassLLM, a novel multi-agent framework that intelligently leverages the reasoning capabilities of LLMs into the geo-spatial domain to solve the popular path query. CompassLLM employs its agents in a two-stage pipeline: the SEARCH stage that identifies popular paths, and a GENERATE stage that synthesizes novel paths in the absence of an existing one in the historical trajectory data. Experiments on real and synthetic datasets show that CompassLLM demonstrates superior accuracy in SEARCH and competitive performance in GENERATE while being cost-effective.

**arXiv ID:** 2510.07516
</details>

<details>
<summary><strong>Measuring and Mitigating Identity Bias in Multi-Agent Debate via Anonymization</strong> - Hyeong Kyu Choi, Xiaojin Zhu, Yixuan Li - [[pdf]](https://arxiv.org/pdf/2510.07517)</summary>

**Abstract:** Multi-agent debate (MAD) aims to improve large language model (LLM) reasoning by letting multiple agents exchange answers and then aggregate their opinions. Yet recent studies reveal that agents are not neutral: they are prone to identity-driven sycophancy and self-bias, uncritically adopting a peer's view or stubbornly adhering to their own prior output, undermining the reliability of debate. In this work, we present the first principled framework that joins sycophancy and self-bias to mitigate and quantify identity bias in MAD. First, we formalize the debate dynamics as an identity-weighted Bayesian update process. Second, we propose response anonymization: by removing identity markers from prompts, agents cannot distinguish "self" from "peer", which forces equal weights on agent identity, thereby reducing bias. Third, we define the Identity Bias Coefficient (IBC), a principled metric that measures how often an agent follows a peer versus itself. Empirical studies across multiple models, datasets and debate rounds confirm that identity bias is widespread, with sycophancy far more common than self-bias. Our findings highlight the need to "mask" identity to ensure that MAD systems reason based on content rather than source identity. Code is released in this https URL.

**arXiv ID:** 2510.07517
</details>

<details>
<summary><strong>AgentAsk: Multi-Agent Systems Need to Ask</strong> - Bohan Lin, Kuo Yang, Yingchuan Lai, Yudong Zhang, Chen Zhang, Guibin Zhang, Xinlei Yu, Miao Yu, Xu Wang, Yang Wang - [[pdf]](https://arxiv.org/pdf/2510.07593)</summary>

**Abstract:** Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving capabilities through collaborative division of labor. However, they frequently underperform single-agent baselines due to edge-level error cascades: minor inaccuracies at one message handoff propagate across the entire chain. We propose AgentAsk, a lightweight and plug-and-play clarification module that treats every inter-agent message as a potential failure point and inserts minimally necessary questions to arrest error propagation. AgentAsk follows a three-stage pipeline: (i) distilling edge-level judgments from curated failure traces into a compact policy, (ii) supervising the policy to determine when/what/whom/how to ask, and (iii) optimizing online with E-GRPO, a reinforcement learning objective that balances accuracy, latency, and cost. The module is architecture-agnostic and easy to integrate into existing orchestration. Across math, reasoning, and coding benchmarks, AgentAsk consistently improves accuracy and robustness over public multi-agent implementations while keeping overhead minimal, with latency and extra cost all less than 5%, approaching the performance of a strong evaluator. Beyond empirical improvements, we contribute a principled taxonomy of edge-level errors and a practical recipe for link-local intervention, offering a scalable pathway toward more reliable LLM-based multi-agent systems.

**arXiv ID:** 2510.07593
</details>

<details>
<summary><strong>Traceability and Accountability in Role-Specialized Multi-Agent LLM Pipelines</strong> - Amine Barrak - [[pdf]](https://arxiv.org/pdf/2510.07614)</summary>

**Abstract:** Sequential multi-agent systems built with large language models (LLMs) can automate complex software tasks, but they are hard to trust because errors quietly pass from one stage to the next. We study a traceable and accountable pipeline, meaning a system with clear roles, structured handoffs, and saved records that let us trace who did what at each step and assign blame when things go wrong. Our setting is a Planner -> Executor -> Critic pipeline. We evaluate eight configurations of three state-of-the-art LLMs on three benchmarks and analyze where errors start, how they spread, and how they can be fixed. Our results show: (1) adding a structured, accountable handoff between agents markedly improves accuracy and prevents the failures common in simple pipelines; (2) models have clear role-specific strengths and risks (e.g., steady planning vs. high-variance critiquing), which we quantify with repair and harm rates; and (3) accuracy-cost-latency trade-offs are task-dependent, with heterogeneous pipelines often the most efficient. Overall, we provide a practical, data-driven method for designing, tracing, and debugging reliable, predictable, and accountable multi-agent systems.

**arXiv ID:** 2510.07614
</details>

<details>
<summary><strong>SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation</strong> - Minh-Anh Nguye, Minh-Duc Nguyen, Nguyen Thi Ha Lan, Kieu Hai Dang, Nguyen Tien Dong, Le Duy Dung - [[pdf]](https://arxiv.org/pdf/2510.07733)</summary>

**Abstract:** Large language models (LLMs) are increasingly adopted for automating survey paper generation \cite{wang2406autosurvey, liang2025surveyx, yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing approaches typically extract content from a large collection of related papers and prompt LLMs to summarize them directly. However, such methods often overlook the structural relationships among papers, resulting in generated surveys that lack a coherent taxonomy and a deeper contextual understanding of research progress. To address these shortcomings, we propose \textbf{SurveyG}, an LLM-based agent framework that integrates \textit{hierarchical citation graph}, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness between their contents, thereby embedding structural and contextual knowledge into the survey generation process. The graph is organized into three layers: \textbf{Foundation}, \textbf{Development}, and \textbf{Frontier}, to capture the evolution of research from seminal works to incremental advances and emerging directions. By combining horizontal search within layers and vertical depth traversal across layers, the agent produces multi-level summaries, which are consolidated into a structured survey outline. A multi-agent validation stage then ensures consistency, coverage, and factual accuracy in generating the final survey. Experiments, including evaluations by human experts and LLM-as-a-judge, demonstrate that SurveyG outperforms state-of-the-art frameworks, producing surveys that are more comprehensive and better structured to the underlying knowledge taxonomy of a field.

**arXiv ID:** 2510.07733
</details>

<details>
<summary><strong>Enabling Personalized Long-term Interactions in LLM-based Agents through Persistent Memory and User Profiles</strong> - Rebecca Westhäußer, Wolfgang Minker, Sebatian Zepf - [[pdf]](https://arxiv.org/pdf/2510.07925)</summary>

**Abstract:** Large language models (LLMs) increasingly serve as the central control unit of AI agents, yet current approaches remain limited in their ability to deliver personalized interactions. While Retrieval Augmented Generation enhances LLM capabilities by improving context-awareness, it lacks mechanisms to combine contextual information with user-specific data. Although personalization has been studied in fields such as human-computer interaction or cognitive science, existing perspectives largely remain conceptual, with limited focus on technical implementation. To address these gaps, we build on a unified definition of personalization as a conceptual foundation to derive technical requirements for adaptive, user-centered LLM-based agents. Combined with established agentic AI patterns such as multi-agent collaboration or multi-source retrieval, we present a framework that integrates persistent memory, dynamic coordination, self-validation, and evolving user profiles to enable personalized long-term interactions. We evaluate our approach on three public datasets using metrics such as retrieval accuracy, response correctness, or BertScore. We complement these results with a five-day pilot user study providing initial insights into user feedback on perceived personalization. The study provides early indications that guide future work and highlights the potential of integrating persistent memory and user profiles to improve the adaptivity and perceived personalization of LLM-based agents.

**arXiv ID:** 2510.07925
</details>

<details>
<summary><strong>Agent-Based Genetic Algorithm for Crypto Trading Strategy Optimization</strong> - Qiushi Tian, Churong Liang, Kairan Hong, Runnan Li - [[pdf]](https://arxiv.org/pdf/2510.07943)</summary>

**Abstract:** Cryptocurrency markets present formidable challenges for trading strategy optimization due to extreme volatility, non-stationary dynamics, and complex microstructure patterns that render conventional parameter optimization methods fundamentally inadequate. We introduce Cypto Genetic Algorithm Agent (CGA-Agent), a pioneering hybrid framework that synergistically integrates genetic algorithms with intelligent multi-agent coordination mechanisms for adaptive trading strategy parameter optimization in dynamic financial environments. The framework uniquely incorporates real-time market microstructure intelligence and adaptive strategy performance feedback through intelligent mechanisms that dynamically guide evolutionary processes, transcending the limitations of static optimization approaches. Comprehensive empirical evaluation across three cryptocurrencies demonstrates systematic and statistically significant performance improvements on both total returns and risk-adjusted metrics.

**arXiv ID:** 2510.07943
</details>

<details>
<summary><strong>ReInAgent: A Context-Aware GUI Agent Enabling Human-in-the-Loop Mobile Task Navigation</strong> - Haitao Jia, Ming He, Zimo Yin, Likang Wu, Jianping Fan, Jitao Sang - [[pdf]](https://arxiv.org/pdf/2510.07988)</summary>

**Abstract:** Mobile GUI agents exhibit substantial potential to facilitate and automate the execution of user tasks on mobile phones. However, exist mobile GUI agents predominantly privilege autonomous operation and neglect the necessity of active user engagement during task execution. This omission undermines their adaptability to information dilemmas including ambiguous, dynamically evolving, and conflicting task scenarios, leading to execution outcomes that deviate from genuine user requirements and preferences. To address these shortcomings, we propose ReInAgent, a context-aware multi-agent framework that leverages dynamic information management to enable human-in-the-loop mobile task navigation. ReInAgent integrates three specialized agents around a shared memory module: an information-managing agent for slot-based information management and proactive interaction with the user, a decision-making agent for conflict-aware planning, and a reflecting agent for task reflection and information consistency validation. Through continuous contextual information analysis and sustained user-agent collaboration, ReInAgent overcomes the limitation of existing approaches that rely on clear and static task assumptions. Consequently, it enables more adaptive and reliable mobile task navigation in complex, real-world scenarios. Experimental results demonstrate that ReInAgent effectively resolves information dilemmas and produces outcomes that are more closely aligned with genuine user preferences. Notably, on complex tasks involving information dilemmas, ReInAgent achieves a 25% higher success rate than Mobile-Agent-v2.

**arXiv ID:** 2510.07988
</details>

<details>
<summary><strong>Co-TAP: Three-Layer Agent Interaction Protocol Technical Report</strong> - Shunyu An, Miao Wang, Yongchao Li, Dong Wan, Lina Wang, Ling Qin, Liqin Gao, Congyao Fan, Zhiyong Mao, Jiange Pu, Wenji Xia, Dong Zhao, Rui Hu, Ji Lu, Guiyue Zhou, Baoyu Tang, Yanqin Gao, Yongsheng Du, Daigang Xu, Lingjun Huang, Baoli Wang, Xiwen Zhang, Luyao Wang, Shilong Liu - [[pdf]](https://arxiv.org/pdf/2510.08263)</summary>

**Abstract:** This paper proposes Co-TAP (T: Triple, A: Agent, P: Protocol), a three-layer agent interaction protocol designed to address the challenges faced by multi-agent systems across the three core dimensions of Interoperability, Interaction and Collaboration, and Knowledge Sharing. We have designed and proposed a layered solution composed of three core protocols: the Human-Agent Interaction Protocol (HAI), the Unified Agent Protocol (UAP), and the Memory-Extraction-Knowledge Protocol (MEK). HAI focuses on the interaction layer, standardizing the flow of information between users, interfaces, and agents by defining a standardized, event-driven communication paradigm. This ensures the real-time performance, reliability, and synergy of interactions. As the core of the infrastructure layer, UAP is designed to break down communication barriers among heterogeneous agents through unified service discovery and protocol conversion mechanisms, thereby enabling seamless interconnection and interoperability of the underlying network. MEK, in turn, operates at the cognitive layer. By establishing a standardized ''Memory (M) - Extraction (E) - Knowledge (K)'' cognitive chain, it empowers agents with the ability to learn from individual experiences and form shareable knowledge, thereby laying the foundation for the realization of true collective intelligence. We believe this protocol framework will provide a solid engineering foundation and theoretical guidance for building the next generation of efficient, scalable, and intelligent multi-agent applications.

**arXiv ID:** 2510.08263
</details>

<details>
<summary><strong>Can Lessons From Human Teams Be Applied to Multi-Agent Systems? The Role of Structure, Diversity, and Interaction Dynamics</strong> - Rasika Muralidharan, Jaewoon Kwak, Jisun An - [[pdf]](https://arxiv.org/pdf/2510.07488)</summary>

**Abstract:** Multi-Agent Systems (MAS) with Large Language Model (LLM)-powered agents are gaining attention, yet fewer studies explore their team dynamics. Inspired by human team science, we propose a multi-agent framework to examine core aspects of team science: structure, diversity, and interaction dynamics. We evaluate team performance across four tasks: CommonsenseQA, StrategyQA, Social IQa, and Latent Implicit Hate, spanning commonsense and social reasoning. Our results show that flat teams tend to perform better than hierarchical ones, while diversity has a nuanced impact. Interviews suggest agents are overconfident about their team performance, yet post-task reflections reveal both appreciation for collaboration and challenges in integration, including limited conversational coordination.

**arXiv ID:** 2510.07488
</details>

<details>
<summary><strong>Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models</strong> - Eric Hanchen Jiang, Guancheng Wan, Sophia Yin, Mengting Li, Yuchen Wu, Xiao Liang, Xinfeng Li, Yizhou Sun, Wei Wang, Kai-Wei Chang, Ying Nian Wu - [[pdf]](https://arxiv.org/pdf/2510.07799)</summary>

**Abstract:** The efficiency of multi-agent systems driven by large language models (LLMs) largely hinges on their communication topology. However, designing an optimal topology is a non-trivial challenge, as it requires balancing competing objectives such as task performance, communication cost, and robustness. Existing frameworks often rely on static or hand-crafted topologies, which inherently fail to adapt to diverse task requirements, leading to either excessive token consumption for simple problems or performance bottlenecks for complex ones. To address this challenge, we introduce a novel generative framework called \textit{Guided Topology Diffusion (GTD)}. Inspired by conditional discrete graph diffusion models, GTD formulates topology synthesis as an iterative construction process. At each step, the generation is steered by a lightweight proxy model that predicts multi-objective rewards (e.g., accuracy, utility, cost), enabling real-time, gradient-free optimization towards task-adaptive topologies. This iterative, guided synthesis process distinguishes GTD from single-step generative frameworks, enabling it to better navigate complex design trade-offs. We validated GTD across multiple benchmarks, and experiments show that this framework can generate highly task-adaptive, sparse, and efficient communication topologies, significantly outperforming existing methods in LLM agent collaboration.

**arXiv ID:** 2510.07799
</details>

<details>
<summary><strong>An Adaptive Multi Agent Bitcoin Trading System</strong> - Aadi Singhi - [[pdf]](https://arxiv.org/pdf/2510.08068)</summary>

**Abstract:** This paper presents a Multi Agent Bitcoin Trading system that utilizes Large Lan- guage Models (LLMs) for alpha generation and portfolio management in the cryptocur- rencies market. Unlike equities, cryptocurrencies exhibit extreme volatility and are heavily influenced by rapidly shifting market sentiments and regulatory announcements, making them difficult to model using static regression models or neural networks trained solely on historical data [53]. The proposed framework overcomes this by structuring LLMs into specialised agents for technical analysis, sentiment evaluation, decision-making, and performance reflection. The system improves over time through a novel verbal feedback mechanism where a Reflect agent provides daily and weekly natural-language critiques of trading decisions. These textual evaluations are then injected into future prompts, al- lowing the system to adjust indicator priorities, sentiment weights, and allocation logic without parameter updates or finetuning. Back-testing on Bitcoin price data from July 2024 to April 2025 shows consistent outperformance across market regimes: the Quantita- tive agent delivered over 30% higher returns in bullish phases and 15% overall gains versus buy-and-hold, while the sentiment-driven agent turned sideways markets from a small loss into a gain of over 100%. Adding weekly feedback further improved total performance by 31% and reduced bearish losses by 10%. The results demonstrate that verbal feedback represents a new, scalable, and low-cost method of tuning LLMs for financial goals.

**arXiv ID:** 2510.08068
</details>

<details>
<summary><strong>Opponent Shaping in LLM Agents</strong> - Marta Emili Garcia Segura, Stephen Hailes, Mirco Musolesi - [[pdf]](https://arxiv.org/pdf/2510.08255)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly being deployed as autonomous agents in real-world environments. As these deployments scale, multi-agent interactions become inevitable, making it essential to understand strategic behavior in such systems. A central open question is whether LLM agents, like reinforcement learning agents, can shape the learning dynamics and influence the behavior of others through interaction alone. In this paper, we present the first investigation of opponent shaping (OS) with LLM-based agents. Existing OS algorithms cannot be directly applied to LLMs, as they require higher-order derivatives, face scalability constraints, or depend on architectural components that are absent in transformers. To address this gap, we introduce ShapeLLM, an adaptation of model-free OS methods tailored for transformer-based agents. Using ShapeLLM, we examine whether LLM agents can influence co-players' learning dynamics across diverse game-theoretic environments. We demonstrate that LLM agents can successfully guide opponents toward exploitable equilibria in competitive games (Iterated Prisoner's Dilemma, Matching Pennies, and Chicken) and promote coordination and improve collective welfare in cooperative games (Iterated Stag Hunt and a cooperative version of the Prisoner's Dilemma). Our findings show that LLM agents can both shape and be shaped through interaction, establishing opponent shaping as a key dimension of multi-agent LLM research.

**arXiv ID:** 2510.08255
</details>

<details>
<summary><strong>ClauseLens: Clause-Grounded, CVaR-Constrained Reinforcement Learning for Trustworthy Reinsurance Pricing</strong> - Stella C. Dong, James R. Finlay - [[pdf]](https://arxiv.org/pdf/2510.08429)</summary>

**Abstract:** Reinsurance treaty pricing must satisfy stringent regulatory standards, yet current quoting practices remain opaque and difficult to audit. We introduce ClauseLens, a clause-grounded reinforcement learning framework that produces transparent, regulation-compliant, and risk-aware treaty quotes.
ClauseLens models the quoting task as a Risk-Aware Constrained Markov Decision Process (RA-CMDP). Statutory and policy clauses are retrieved from legal and underwriting corpora, embedded into the agent's observations, and used both to constrain feasible actions and to generate clause-grounded natural language justifications.
Evaluated in a multi-agent treaty simulator calibrated to industry data, ClauseLens reduces solvency violations by 51%, improves tail-risk performance by 27.9% (CVaR_0.10), and achieves 88.2% accuracy in clause-grounded explanations with retrieval precision of 87.4% and recall of 91.1%.
These findings demonstrate that embedding legal context into both decision and explanation pathways yields interpretable, auditable, and regulation-aligned quoting behavior consistent with Solvency II, NAIC RBC, and the EU AI Act.

**arXiv ID:** 2510.08429
</details>

<details>
<summary><strong>CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards</strong> - Xiangyuan Xue, Yifan Zhou, Guibin Zhang, Zaibin Zhang, Yijiang Li, Chen Zhang, Zhenfei Yin, Philip Torr, Wanli Ouyang, Lei Bai - [[pdf]](https://arxiv.org/pdf/2510.08529)</summary>

**Abstract:** Self-evolution is a central research topic in enabling large language model (LLM)-based agents to continually improve their capabilities after pretraining. Recent research has witnessed a transition from reinforcement learning (RL)-free to RL-based methods. Current RL-based methods either rely on dense external reward signals or extract intrinsic reward signals from LLMs themselves. However, these approaches diverge from the self-evolution mechanisms observed in human intelligence, where individuals learn and improve through mutual discussion and collaboration. In this work, we introduce Co-Evolving Multi-Agent Systems (CoMAS), a novel framework that enables agents to improve autonomously by learning from inter-agent interactions without external supervision. CoMAS generates intrinsic rewards from rich discussion dynamics, employs an LLM-as-a-judge mechanism to formulate these rewards, and optimizes each agent's policy through RL, thereby enabling decentralized and scalable co-evolution. Experimental results demonstrate that CoMAS consistently outperforms untrained agents and achieves state-of-the-art performance across most evaluation settings. Ablation studies confirm the necessity of interaction-based reward signals and reveal promising scalability as the number and diversity of agents increase. These findings establish CoMAS as a novel and effective paradigm for self-evolution in LLM-based agents.

**arXiv ID:** 2510.08529
</details>

<details>
<summary><strong>AutoAgent: A Fully-Automated and Zero-Code Framework for LLM Agents</strong> - Jiabin Tang, Tianyu Fan, Chao Huang - [[pdf]](https://arxiv.org/pdf/2502.05957)</summary>

**Abstract:** Large Language Model (LLM) Agents have demonstrated remarkable capabilities in task automation and intelligent decision-making, driving the widespread adoption of agent development frameworks such as LangChain and AutoGen. However, these frameworks predominantly serve developers with extensive technical expertise - a significant limitation considering that only 0.03 % of the global population possesses the necessary programming skills. This stark accessibility gap raises a fundamental question: Can we enable everyone, regardless of technical background, to build their own LLM agents using natural language alone? To address this challenge, we introduce AutoAgent-a Fully-Automated and highly Self-Developing framework that enables users to create and deploy LLM agents through Natural Language Alone. Operating as an autonomous Agent Operating System, AutoAgent comprises four key components: i) Agentic System Utilities, ii) LLM-powered Actionable Engine, iii) Self-Managing File System, and iv) Self-Play Agent Customization module. This lightweight yet powerful system enables efficient and dynamic creation and modification of tools, agents, and workflows without coding requirements or manual intervention. Beyond its code-free agent development capabilities, AutoAgent also serves as a versatile multi-agent system for General AI Assistants. Comprehensive evaluations on the GAIA benchmark demonstrate AutoAgent's effectiveness in generalist multi-agent tasks, surpassing existing state-of-the-art methods. Furthermore, AutoAgent's Retrieval-Augmented Generation (RAG)-related capabilities have shown consistently superior performance compared to many alternative LLM-based solutions.

**arXiv ID:** 2502.05957
</details>

<details>
<summary><strong>Position Paper: Towards Open Complex Human-AI Agents Collaboration Systems for Problem Solving and Knowledge Management</strong> - Ju Wu, Calvin K.L. Or - [[pdf]](https://arxiv.org/pdf/2505.00018)</summary>

**Abstract:** We propose a technology-agnostic, collaboration-ready stance for Human-AI Agents Collaboration Systems (HAACS) that closes long-standing gaps in prior stages (automation; flexible autonomy; agentic multi-agent collectives). Reading empirical patterns through a seven-dimension collaboration spine and human-agent contrasts, we identify missing pieces: principled budgeting of initiative, instantaneous and auditable reconfiguration, a system-wide knowledge backbone with an epistemic promotion gate, capacity-aware human interfaces; and, as a prerequisite to all of the above, unified definitions of agent and formal collaborative dynamics. We respond with (i) a boundary-centric ontology of agenthood synthesized with cybernetics; (ii) a Petri net family (colored and interpreted) that models ownership, cross-boundary interaction, concurrency, guards, and rates with collaboration transitions; and (iii) a three-level orchestration (meta, agent, execution) that governs behavior families via guard flips. On the knowledge side, we ground collaborative learning in Conversation Theory and SECI with teach-back gates and an evolving backbone; on the problem-solving side, we coordinate routine MEA-style control with practice-guided open-ended discovery. The result is the Hierarchical Exploration-Exploitation Net (HE2-Net): a policy-controlled stance that splits provisional from validated assets, promotes only after tests and peer checks, and budgets concurrent probing while keeping reuse fast and safe. We show interoperability with emerging agent protocols without ad hoc glue and sketch bio-cybernetic extensions (autopoiesis, autogenesis, evolving boundaries, synergetics, etc). Altogether, the framework keeps humans central to setting aims, justifying knowledge, and steering theory-practice dynamics, while scaling agents as reliable collaborators within audited governance.

**arXiv ID:** 2505.00018
</details>

<details>
<summary><strong>Reducing Cognitive Overhead in Tool Use via Multi-Small-Agent Reinforcement Learning</strong> - Dayu Wang, Jiaye Yang, Weikang Li, Jiahui Liang, Yang Li - [[pdf]](https://arxiv.org/pdf/2508.08882)</summary>

**Abstract:** Recent advances in multi-agent systems highlight the potential of specialized small agents that collaborate via division of labor. Existing tool-integrated reasoning systems, however, often follow a single-agent paradigm in which one large model interleaves long-horizon reasoning with precise tool operations, leading to cognitive-load interference and unstable coordination. We present MSARL, a Multi-Small-Agent Reinforcement Learning framework that explicitly decouples reasoning from tool use. In MSARL, a Reasoning Agent decomposes problems and plans tool invocations, while multiple Tool Agents specialize in specific external tools, each trained via a combination of imitation learning and reinforcement learning with role-specific rewards. On mathematical problem solving with code execution, MSARL significantly improves reasoning stability and final-answer accuracy over single-agent baselines. Moreover, the architecture generalizes to diverse tool-use tasks, demonstrating that cognitive-role decoupling with small agents is a scalable blueprint for multi-agent AI design.

**arXiv ID:** 2508.08882
</details>

<details>
<summary><strong>Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers</strong> - Ran Xin, Zeyu Zheng, Yanchen Nie, Kun Yuan, Xia Xiao - [[pdf]](https://arxiv.org/pdf/2509.06493)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search.

**arXiv ID:** 2509.06493
</details>

<details>
<summary><strong>Neuro-Symbolic Agents with Modal Logic for Autonomous Diagnostics</strong> - Antonin Sulc, Thorsten Hellert - [[pdf]](https://arxiv.org/pdf/2509.11943)</summary>

**Abstract:** The development of intelligent agents, particularly those powered by language models (LMs), has shown the critical role in various environments that require intelligent and autonomous decision. Environments are not passive testing grounds and they represent the data required for agents to learn and exhibit very challenging conditions that require adaptive, complex and autonomous capacity to make decisions. While the paradigm of scaling models and datasets has led to remarkable emergent capabilities, we argue that scaling the structure, fidelity, and logical consistency of agent reasoning within these environments is a crucial, yet underexplored, dimension of AI research. This paper introduces a neuro-symbolic multi-agent architecture where the belief states of individual agents are formally represented as Kripke models. This foundational choice enables them to reason about known concepts of \emph{possibility} and \emph{necessity} using the formal language of modal logic. In this work, we use of immutable, domain-specific knowledge to make infere information, which is encoded as logical constraints essential for proper diagnosis. In the proposed model, we show constraints that actively guide the hypothesis generation of LMs, effectively preventing them from reaching physically or logically untenable conclusions. In a high-fidelity simulated particle accelerator environment, our system successfully diagnoses complex, cascading failures by combining the powerful semantic intuition of LMs with the rigorous, verifiable validation of modal logic and a factual world model and showcasing a viable path toward more robust, reliable, and verifiable autonomous agents.

**arXiv ID:** 2509.11943
</details>

<details>
<summary><strong>Network Topology and Information Efficiency of Multi-Agent Systems: Study based on MARL</strong> - Xinren Zhang, Sixi Cheng, Zixin Zhong, Jiadong Yu - [[pdf]](https://arxiv.org/pdf/2510.07888)</summary>

**Abstract:** Multi-agent systems (MAS) solve complex problems through coordinated autonomous entities with individual decision-making capabilities. While Multi-Agent Reinforcement Learning (MARL) enables these agents to learn intelligent strategies, it faces challenges of non-stationarity and partial observability. Communications among agents offer a solution, but questions remain about its optimal structure and evaluation. This paper explores two underexamined aspects: communication topology and information efficiency. We demonstrate that directed and sequential topologies improve performance while reducing communication overhead across both homogeneous and heterogeneous tasks. Additionally, we introduce two metrics -- Information Entropy Efficiency Index (IEI) and Specialization Efficiency Index (SEI) -- to evaluate message compactness and role differentiation. Incorporating these metrics into training objectives improves success rates and convergence speed. Our findings highlight that designing adaptive communication topologies with information-efficient messaging is essential for effective coordination in complex MAS.

**arXiv ID:** 2510.07888
</details>

<details>
<summary><strong>Climate Surrogates for Scalable Multi-Agent Reinforcement Learning: A Case Study with CICERO-SCM</strong> - Oskar Bohn Lassen, Serio Angelo Maria Agriesti, Filipe Rodrigues, Francisco Camara Pereira - [[pdf]](https://arxiv.org/pdf/2510.07971)</summary>

**Abstract:** Climate policy studies require models that capture the combined effects of multiple greenhouse gases on global temperature, but these models are computationally expensive and difficult to embed in reinforcement learning. We present a multi-agent reinforcement learning (MARL) framework that integrates a high-fidelity, highly efficient climate surrogate directly in the environment loop, enabling regional agents to learn climate policies under multi-gas dynamics. As a proof of concept, we introduce a recurrent neural network architecture pretrained on ($20{,}000$) multi-gas emission pathways to surrogate the climate model CICERO-SCM. The surrogate model attains near-simulator accuracy with global-mean temperature RMSE $\approx 0.0004 \mathrm{K}$ and approximately $1000\times$ faster one-step inference. When substituted for the original simulator in a climate-policy MARL setting, it accelerates end-to-end training by $>\!100\times$. We show that the surrogate and simulator converge to the same optimal policies and propose a methodology to assess this property in cases where using the simulator is intractable. Our work allows to bypass the core computational bottleneck without sacrificing policy fidelity, enabling large-scale multi-agent experiments across alternative climate-policy regimes with multi-gas dynamics and high-fidelity climate response.

**arXiv ID:** 2510.07971
</details>

<details>
<summary><strong>$\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks</strong> - Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Fleming, Tianlong Chen - [[pdf]](https://arxiv.org/pdf/2504.00218)</summary>

**Abstract:** Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.

**arXiv ID:** 2504.00218
</details>

<details>
<summary><strong>MARLIN: Multi-Agent Reinforcement Learning with Murmuration Intelligence and LLM Guidance for Reservoir Management</strong> - Heming Fu, Guojun Xiong, Shan Lin - [[pdf]](https://arxiv.org/pdf/2509.25034)</summary>

**Abstract:** As climate change intensifies extreme weather events, water disasters pose growing threats to global communities, making adaptive reservoir management critical for protecting vulnerable populations and ensuring water security. Modern water resource management faces unprecedented challenges from cascading uncertainties propagating through interconnected reservoir networks. These uncertainties, rooted in physical water transfer losses and environmental variability, make precise control difficult. For example, sending 10 tons downstream may yield only 8-12 tons due to evaporation and seepage. Traditional centralized optimization approaches suffer from exponential computational complexity and cannot effectively handle such real-world uncertainties, while existing multi-agent reinforcement learning (MARL) methods fail to achieve effective coordination under uncertainty. To address these challenges, we present MARLIN, a decentralized reservoir management framework inspired by starling murmurations intelligence. Integrating bio-inspired alignment, separation, and cohesion rules with MARL, MARLIN enables individual reservoirs to make local decisions while achieving emergent global coordination. In addition, a LLM provides real-time reward shaping signals, guiding agents to adapt to environmental changes and human-defined preferences. Experiments on real-world USGS data show that MARLIN improves uncertainty handling by 23\%, cuts computation by 35\%, and accelerates flood response by 68\%, exhibiting super-linear coordination, with complexity scaling 5.4x from 400 to 10,000 nodes. These results demonstrate MARLIN's potential for disaster prevention and protecting communities through intelligent, scalable water resource management.

**arXiv ID:** 2509.25034
</details>

<details>
<summary><strong>ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning</strong> - Yu Sun, Xingyu Qian, Weiwen Xu, Hao Zhang, Chenghao Xiao, Long Li, Deli Zhao, Wenbing Huang, Tingyang Xu, Qifeng Bai, Yu Rong - [[pdf]](https://arxiv.org/pdf/2506.09513)</summary>

**Abstract:** Reasoning-based large language models have excelled in mathematics and programming, yet their potential in knowledge-intensive medical question answering remains underexplored and insufficiently validated in clinical contexts. To bridge this gap, we introduce ReasonMed, the largest medical reasoning dataset to date, comprising 370k high-quality examples distilled from 1.75 million initial reasoning paths generated by complementary LLMs and curated through a cost-efficient easy-medium-difficult (EMD) pipeline. ReasonMed is built through a multi-agent generation, verification, and refinement process, in which an Error Refiner improves reasoning paths by correcting error-prone steps identified by a verifier. Using ReasonMed, we investigate effective strategies for training medical reasoning models and find that integrating detailed CoT reasoning with concise answer summaries yields the most robust fine-tuning results. Models trained on ReasonMed set a new benchmark: ReasonMed-7B surpasses the prior best sub-10B models by 4.17% and even exceeds LLaMA3.1-70B on PubMedQA by 4.60%. When scaled to ReasonMed-14B, it remains highly competitive, underscoring consistent scaling potential. The codes and datasets are available at this https URL.

**arXiv ID:** 2506.09513
</details>

<details>
<summary><strong>MAViS: A Multi-Agent Framework for Long-Sequence Video Storytelling</strong> - Qian Wang, Ziqi Huang, Ruoxi Jia, Paul Debevec, Ning Yu - [[pdf]](https://arxiv.org/pdf/2508.08487)</summary>

**Abstract:** Despite recent advances, long-sequence video generation frameworks still suffer from significant limitations: poor assistive capability, suboptimal visual quality, and limited expressiveness. To mitigate these limitations, we propose MAViS, a multi-agent collaborative framework designed to assist in long-sequence video storytelling by efficiently translating ideas into visual narratives. MAViS orchestrates specialized agents across multiple stages, including script writing, shot designing, character modeling, keyframe generation, video animation, and audio generation. In each stage, agents operate under the 3E Principle--Explore, Examine, and Enhanc--to ensure the completeness of intermediate outputs. Considering the capability limitations of current generative models, we propose the Script Writing Guidelines to optimize compatibility between scripts and generative tools. Experimental results demonstrate that MAViS achieves state-of-the-art performance in assistive capability, visual quality, and video expressiveness. Its modular framework further enables scalability with diverse generative models and tools. With just a brief idea description, MAViS enables users to rapidly explore diverse visual storytelling and creative directions for sequential video generation by efficiently producing high-quality, complete long-sequence videos. To the best of our knowledge, MAViS is the only framework that provides multimodal design output -- videos with narratives and background music.

**arXiv ID:** 2508.08487
</details>

<details>
<summary><strong>MAHL: Multi-Agent LLM-Guided Hierarchical Chiplet Design with Adaptive Debugging</strong> - Jinwei Tang, Jiayin Qin, Nuo Xu, Pragnya Sudershan Nalla, Yu Cao, Yang, Zhao, Caiwen Ding - [[pdf]](https://arxiv.org/pdf/2508.14053)</summary>

**Abstract:** As program workloads (e.g., AI) increase in size and algorithmic complexity, the primary challenge lies in their high dimensionality, encompassing computing cores, array sizes, and memory hierarchies. To overcome these obstacles, innovative approaches are required. Agile chip design has already benefited from machine learning integration at various stages, including logic synthesis, placement, and routing. With Large Language Models (LLMs) recently demonstrating impressive proficiency in Hardware Description Language (HDL) generation, it is promising to extend their abilities to 2.5D integration, an advanced technique that saves area overhead and development costs. However, LLM-driven chiplet design faces challenges such as flatten design, high validation cost and imprecise parameter optimization, which limit its chiplet design capability. To address this, we propose MAHL, a hierarchical LLM-based chiplet design generation framework that features six agents which collaboratively enable AI algorithm-hardware mapping, including hierarchical description generation, retrieval-augmented code generation, diverseflow-based validation, and multi-granularity design space exploration. These components together enhance the efficient generation of chiplet design with optimized Power, Performance and Area (PPA). Experiments show that MAHL not only significantly improves the generation accuracy of simple RTL design, but also increases the generation accuracy of real-world chiplet design, evaluated by Pass@5, from 0 to 0.72 compared to conventional LLMs under the best-case scenario. Compared to state-of-the-art CLARIE (expert-based), MAHL achieves comparable or even superior PPA results under certain optimization objectives.

**arXiv ID:** 2508.14053
</details>

<details>
<summary><strong>MAPRO: Recasting Multi-Agent Prompt Optimization as Maximum a Posteriori Inference</strong> - Zheyuan Zhang, Lin Ge, Hongjiang Li, Weicheng Zhu, Chuxu Zhang, Yanfang Ye - [[pdf]](https://arxiv.org/pdf/2510.07475)</summary>

**Abstract:** Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, and LLM-based agents further extend these abilities to various practical workflows. While recent progress shows that multi-agent systems (MAS) can outperform single agents by coordinating specialized roles, designing effective MAS remains difficult due to prompt sensitivity and the compounded instability MAS creates. To cope with the challenge, recent efforts in automated prompt design have reduced manual effort. However, multi-agent prompt optimization remains largely unexplored. Challenges like exponentially expanding search space and ambiguous credit assignment together make systematic design intractable without principled methods. Therefore, we introduce M}ulti-Agent PRompt Optimization (MAPRO), a four-stage framework that first formulates MAS prompt optimization as a Maximum a Posteriori (MAP) inference problem and solves it using a language-guided variant of max-product belief propagation algorithm. To address credit assignment and updates the system iteratively, MAPRO employs a topology-aware refinement mechanism that integrates execution feedback and downstream blames to selectively update agent prompts. Through this process, MAPRO progressively converges to a coordinated set of agent-specific prompt policies. Across benchmarks in various tasks, MAPRO achieves state-of-the-art performance, consistently surpassing manually engineered baselines and recent automated alternatives. Beyond performance, our MAP-based formulation also delivers general guidelines for building more reliable and principled multi-agent systems in the future

**arXiv ID:** 2510.07475
</details>

<details>
<summary><strong>The Alignment Waltz: Jointly Training Agents to Collaborate for Safety</strong> - Jingyu Zhang, Haozhu Wang, Eric Michael Smith, Sid Wang, Amr Sharaf, Mahesh Pasupuleti, Benjamin Van Durme, Daniel Khashabi, Jason Weston, Hongyuan Zhan - [[pdf]](https://arxiv.org/pdf/2510.08240)</summary>

**Abstract:** Harnessing the power of LLMs requires a delicate dance between being helpful and harmless. This creates a fundamental tension between two competing challenges: vulnerability to adversarial attacks that elicit unsafe content, and a tendency for overrefusal on benign but sensitive prompts. Current approaches often navigate this dance with safeguard models that completely reject any content that contains unsafe portions. This approach cuts the music entirely-it may exacerbate overrefusals and fails to provide nuanced guidance for queries it refuses. To teach models a more coordinated choreography, we propose WaltzRL, a novel multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. WaltzRL jointly trains a conversation agent and a feedback agent, where the latter is incentivized to provide useful suggestions that improve the safety and helpfulness of the conversation agent's responses. At the core of WaltzRL is a Dynamic Improvement Reward (DIR) that evolves over time based on how well the conversation agent incorporates the feedback. At inference time, unsafe or overrefusing responses from the conversation agent are improved rather than discarded. The feedback agent is deployed together with the conversation agent and only engages adaptively when needed, preserving helpfulness and low latency on safe queries. Our experiments, conducted across five diverse datasets, demonstrate that WaltzRL significantly reduces both unsafe responses (e.g., from 39.0% to 4.6% on WildJailbreak) and overrefusals (from 45.3% to 9.9% on OR-Bench) compared to various baselines. By enabling the conversation and feedback agents to co-evolve and adaptively apply feedback, WaltzRL enhances LLM safety without degrading general capabilities, thereby advancing the Pareto front between helpfulness and harmlessness.

**arXiv ID:** 2510.08240
</details>

<details>
<summary><strong>CoCoA: Collaborative Chain-of-Agents for Parametric-Retrieved Knowledge Synergy</strong> - Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Lizhe Zhang, Yan Liu, Bing Qin - [[pdf]](https://arxiv.org/pdf/2508.01696)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs), especially for knowledge-intensive tasks. Despite its advantages, current RAG methods often struggle to fully exploit knowledge during generation. In particular, the synergy between the model's internal parametric knowledge and external retrieved knowledge remains limited. Retrieved contents may sometimes mislead generation, while certain generated content can guide the model toward more accurate outputs. In this work, we propose Collaborative Chain-of-Agents, a framework designed to enhance explicitly synergy over both parametric and retrieved knowledge. Specifically, we first introduce CoCoA-zero, a multi-agent RAG framework that first performs conditional knowledge induction and then reasons answers. Building on this, we develop CoCoA, a long-chain training strategy that synthesizes extended multi-agent reasoning trajectories from CoCoA-zero to fine-tune the LLM. This strategy enhances the model's capability to explicitly integrate and jointly leverage parametric and retrieved knowledge. Experimental results demonstrate the superiority of CoCoA in open-domain QA and multi-hop QA.

**arXiv ID:** 2508.01696
</details>

<details>
<summary><strong>Adaptive Tool Generation with Models as Tools and Reinforcement Learning</strong> - Chenpeng Wang, Xiaojie Cheng, Chunye Wang, Linfeng Yang, Lei Zhang - [[pdf]](https://arxiv.org/pdf/2510.06825)</summary>

**Abstract:** Tool-augmented language models have demonstrated strong capabilities, but their reliance on live API access creates scalability and reliability challenges during training and deployment. We propose MTR, a simulation-first training framework for tool-augmented reasoning. Instead of relying on live APIs, MTR learns from complete ReAct traces with schema-validated, simulated observations. Our approach operates through a multi-agent architecture where a ToolMaker generates task-specific, OpenAI-compatible tool interfaces, an AutoAgent produces structured think-act-observe sequences, and a ToolActor simulates realistic responses. Training proceeds in two stages: Stage-1 Supervised Fine-Tuning (SFT) teaches 'trace grammar' from complete reasoning sequences; Stage-2 Group Relative Policy Optimization (GRPO) optimizes strategy with a composite trace reward that balances answer correctness and internal consistency. Across four multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA, Bamboogle), MTR attains competitive Exact Match (EM) scores to live-API systems and excels on reasoning-intensive tasks, suggesting that effective tool reasoning can be learned from structured traces without live interactions.

**arXiv ID:** 2510.06825
</details>

<details>
<summary><strong>PEAR: Planner-Executor Agent Robustness Benchmark</strong> - Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing - [[pdf]](https://arxiv.org/pdf/2510.07505)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

**arXiv ID:** 2510.07505
</details>

<details>
<summary><strong>Hierarchical Reinforcement Learning with Low-Level MPC for Multi-Agent Control</strong> - Max Studt, Georg Schildbach - [[pdf]](https://arxiv.org/pdf/2509.15799)</summary>

**Abstract:** Achieving safe and coordinated behavior in dynamic, constraint-rich environments remains a major challenge for learning-based control. Pure end-to-end learning often suffers from poor sample efficiency and limited reliability, while model-based methods depend on predefined references and struggle to generalize. We propose a hierarchical framework that combines tactical decision-making via reinforcement learning (RL) with low-level execution through Model Predictive Control (MPC). For the case of multi-agent systems this means that high-level policies select abstract targets from structured regions of interest (ROIs), while MPC ensures dynamically feasible and safe motion. Tested on a predator-prey benchmark, our approach outperforms end-to-end and shielding-based RL baselines in terms of reward, safety, and consistency, underscoring the benefits of combining structured learning with model-based control.

**arXiv ID:** 2509.15799
</details>

<details>
<summary><strong>Simulating Teams with LLM Agents: Interactive 2D Environments for Studying Human-AI Dynamics</strong> - Mohammed Almutairi, Charles Chiang, Haoze Guo, Matthew Belcher, Nandini Banerjee, Maria Milkowski, Svitlana Volkova, Daniel Nguyen, Tim Weninger, Michael Yankoski, Trenton W. Ford, Diego Gomez-Zara - [[pdf]](https://arxiv.org/pdf/2510.08242)</summary>

**Abstract:** Enabling users to create their own simulations offers a powerful way to study team dynamics and performance. We introduce VirTLab, a system that allows researchers and practitioners to design interactive, customizable simulations of team dynamics with LLM-based agents situated in 2D spatial environments. Unlike prior frameworks that restrict scenarios to predefined or static tasks, our approach enables users to build scenarios, assign roles, and observe how agents coordinate, move, and adapt over time. By bridging team cognition behaviors with scalable agent-based modeling, our system provides a testbed for investigating how environments influence coordination, collaboration, and emergent team behaviors. We demonstrate its utility by aligning simulated outcomes with empirical evaluations and a user study, underscoring the importance of customizable environments for advancing research on multi-agent simulations. This work contributes to making simulations accessible to both technical and non-technical users, supporting the design, execution, and analysis of complex multi-agent experiments.

**arXiv ID:** 2510.08242
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>ExpertAgent: Enhancing Personalized Education through Dynamic Planning and Retrieval-Augmented Long-Chain Reasoning</strong> - Binrong Zhu, Guiran Liu, Nina Jiang - [[pdf]](https://arxiv.org/pdf/2510.07456)</summary>

**Abstract:** The application of advanced generative artificial intelligence in education is often constrained by the lack of real-time adaptability, personalization, and reliability of the content. To address these challenges, we propose ExpertAgent - an intelligent agent framework designed for personalized education that provides reliable knowledge and enables highly adaptive learning experiences. Therefore, we developed ExpertAgent, an innovative learning agent that provides users with a proactive and personalized learning experience. ExpertAgent dynamic planning of the learning content and strategy based on a continuously updated student model. Therefore, overcoming the limitations of traditional static learning content to provide optimized teaching strategies and learning experience in real time. All instructional content is grounded in a validated curriculum repository, effectively reducing hallucination risks in large language models and improving reliability and trustworthiness.

**arXiv ID:** 2510.07456
</details>

<details>
<summary><strong>Optimizing Ethical Risk Reduction for Medical Intelligent Systems with Constraint Programming</strong> - Clotilde Brayé, Aurélien Bricout, Arnaud Gotlieb, Nadjib Lazaar, Quentin Vallet - [[pdf]](https://arxiv.org/pdf/2510.07491)</summary>

**Abstract:** Medical Intelligent Systems (MIS) are increasingly integrated into healthcare workflows, offering significant benefits but also raising critical safety and ethical concerns. According to the European Union AI Act, most MIS will be classified as high-risk systems, requiring a formal risk management process to ensure compliance with the ethical requirements of trust- worthy AI. In this context, we focus on risk reduction optimization problems, which aim to reduce risks with ethical considerations by finding the best balanced assignment of risk assessment values according to their coverage of trustworthy AI ethical requirements. We formalize this problem as a constrained optimization task and investigate three resolution paradigms: Mixed Integer Programming (MIP), Satisfiability (SAT), and Constraint Pro- gramming(CP).Our contributions include the mathematical formulation of this optimization problem, its modeling with the Minizinc constraint modeling language, and a comparative experimental study that analyzes the performance, expressiveness, and scalability of each ap- proach to solving. From the identified limits of the methodology, we draw some perspectives of this work regarding the integration of the Minizinc model into a complete trustworthy AI ethical risk management process for MIS.

**arXiv ID:** 2510.07491
</details>

<details>
<summary><strong>Chain-of-Trigger: An Agentic Backdoor that Paradoxically Enhances Agentic Robustness</strong> - Jiyang Qiu, Xinbei Ma, Yunqing Xu, Zhuosheng Zhang, Hai Zhao - [[pdf]](https://arxiv.org/pdf/2510.08238)</summary>

**Abstract:** The rapid deployment of large language model (LLM)-based agents in real-world applications has raised serious concerns about their trustworthiness. In this work, we reveal the security and robustness vulnerabilities of these agents through backdoor attacks. Distinct from traditional backdoors limited to single-step control, we propose the Chain-of-Trigger Backdoor (CoTri), a multi-step backdoor attack designed for long-horizon agentic control. CoTri relies on an ordered sequence. It starts with an initial trigger, and subsequent ones are drawn from the environment, allowing multi-step manipulation that diverts the agent from its intended task. Experimental results show that CoTri achieves a near-perfect attack success rate (ASR) while maintaining a near-zero false trigger rate (FTR). Due to training data modeling the stochastic nature of the environment, the implantation of CoTri paradoxically enhances the agent's performance on benign tasks and even improves its robustness against environmental distractions. We further validate CoTri on vision-language models (VLMs), confirming its scalability to multimodal agents. Our work highlights that CoTri achieves stable, multi-step control within agents, improving their inherent robustness and task capabilities, which ultimately makes the attack more stealthy and raises potential safty risks.

**arXiv ID:** 2510.08238
</details>

<details>
<summary><strong>IntentionVLA: Generalizable and Efficient Embodied Intention Reasoning for Human-Robot Interaction</strong> - Yandu Chen, Kefan Gu, Yuqing Wen, Yucheng Zhao, Tiancai Wang, Liqiang Nie - [[pdf]](https://arxiv.org/pdf/2510.07778)</summary>

**Abstract:** Vision-Language-Action (VLA) models leverage pretrained vision-language models (VLMs) to couple perception with robotic control, offering a promising path toward general-purpose embodied intelligence. However, current SOTA VLAs are primarily pretrained on multimodal tasks with limited relevance to embodied scenarios, and then finetuned to map explicit instructions to actions. Consequently, due to the lack of reasoning-intensive pretraining and reasoning-guided manipulation, these models are unable to perform implicit human intention reasoning required for complex, real-world interactions. To overcome these limitations, we propose \textbf{IntentionVLA}, a VLA framework with a curriculum training paradigm and an efficient inference mechanism. Our proposed method first leverages carefully designed reasoning data that combine intention inference, spatial grounding, and compact embodied reasoning, endowing the model with both reasoning and perception capabilities. In the following finetuning stage, IntentionVLA employs the compact reasoning outputs as contextual guidance for action generation, enabling fast inference under indirect instructions. Experimental results show that IntentionVLA substantially outperforms $\pi_0$, achieving 18\% higher success rates with direct instructions and 28\% higher than ECoT under intention instructions. On out-of-distribution intention tasks, IntentionVLA achieves over twice the success rate of all baselines, and further enables zero-shot human-robot interaction with 40\% success rate. These results highlight IntentionVLA as a promising paradigm for next-generation human-robot interaction (HRI) systems.

**arXiv ID:** 2510.07778
</details>

<details>
<summary><strong>Effective and Stealthy One-Shot Jailbreaks on Deployed Mobile Vision-Language Agents</strong> - Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu - [[pdf]](https://arxiv.org/pdf/2510.07809)</summary>

**Abstract:** Large vision-language models (LVLMs) enable autonomous mobile agents to operate smartphone user interfaces, yet vulnerabilities to UI-level attacks remain critically understudied. Existing research often depends on conspicuous UI overlays, elevated permissions, or impractical threat models, limiting stealth and real-world applicability. In this paper, we present a practical and stealthy one-shot jailbreak attack that leverages in-app prompt injections: malicious applications embed short prompts in UI text that remain inert during human interaction but are revealed when an agent drives the UI via ADB (Android Debug Bridge). Our framework comprises three crucial components: (1) low-privilege perception-chain targeting, which injects payloads into malicious apps as the agent's visual inputs; (2) stealthy user-invisible activation, a touch-based trigger that discriminates agent from human touches using physical touch attributes and exposes the payload only during agent operation; and (3) one-shot prompt efficacy, a heuristic-guided, character-level iterative-deepening search algorithm (HG-IDA*) that performs one-shot, keyword-level detoxification to evade on-device safety filters. We evaluate across multiple LVLM backends, including closed-source services and representative open-source models within three Android applications, and we observe high planning and execution hijack rates in single-shot scenarios (e.g., GPT-4o: 82.5% planning / 75.0% execution). These findings expose a fundamental security vulnerability in current mobile agents with immediate implications for autonomous smartphone operation.

**arXiv ID:** 2510.07809
</details>

<details>
<summary><strong>Iterated Agent for Symbolic Regression</strong> - Zhuo-Yang Song, Zeyu Cai, Shutao Zhang, Jiashen Wei, Jichen Pan, Shi Qiu, Qing-Hong Cao, Tie-Jiun Hou, Xiaohui Liu, Ming-xing Luo, Hua Xing Zhu - [[pdf]](https://arxiv.org/pdf/2510.08317)</summary>

**Abstract:** Symbolic regression (SR), the automated discovery of mathematical expressions from data, is a cornerstone of scientific inquiry. However, it is often hindered by the combinatorial explosion of the search space and a tendency to overfit. Popular methods, rooted in genetic programming, explore this space syntactically, often yielding overly complex, uninterpretable models. This paper introduces IdeaSearchFitter, a framework that employs Large Language Models (LLMs) as semantic operators within an evolutionary search. By generating candidate expressions guided by natural-language rationales, our method biases discovery towards models that are not only accurate but also conceptually coherent and interpretable. We demonstrate IdeaSearchFitter's efficacy across diverse challenges: it achieves competitive, noise-robust performance on the Feynman Symbolic Regression Database (FSReD), outperforming several strong baselines; discovers mechanistically aligned models with good accuracy-complexity trade-offs on real-world data; and derives compact, physically-motivated parametrizations for Parton Distribution Functions in a frontier high-energy physics application. IdeaSearchFitter is a specialized module within our broader iterated agent framework, IdeaSearch, which is publicly available at this https URL.

**arXiv ID:** 2510.08317
</details>

<details>
<summary><strong>What Do Agents Think One Another Want? Level-2 Inverse Games for Inferring Agents' Estimates of Others' Objectives</strong> - Hamzah I. Khan, Jingqi Li, David Fridovich-Keil - [[pdf]](https://arxiv.org/pdf/2508.03824)</summary>

**Abstract:** Effectively interpreting strategic interactions among multiple agents requires us to infer each agent's objective from limited information. Existing inverse game-theoretic approaches frame this challenge in terms of a "level-1" inference problem, in which we take the perspective of a third-party observer and assume that individual agents share complete knowledge of one another's objectives. However, this assumption breaks down in decentralized, real-world scenarios like urban driving and bargaining, in which agents may act based on conflicting views of one another's objectives. We demonstrate the necessity of inferring agents' different estimates of each other's objectives through empirical examples, and by theoretically characterizing the prediction error of level-1 inference on fictitious gameplay data from linear-quadratic games. To address this fundamental issue, we propose a framework for level-2 inference to address the question: "What does each agent believe about other agents' objectives?" We prove that the level-2 inference problem is non-convex even in benign settings like linear-quadratic games, and we develop an efficient gradient-based approach for identifying local solutions. Experiments on a synthetic urban driving example show that our approach uncovers nuanced misalignments that level-1 methods miss.

**arXiv ID:** 2508.03824
</details>

<details>
<summary><strong>IASC: Interactive Agentic System for ConLangs</strong> - Chihiro Taguchi, Richard Sproat - [[pdf]](https://arxiv.org/pdf/2510.07591)</summary>

**Abstract:** We present a system that uses LLMs as a tool in the development of Constructed Languages. The system is modular in that one first creates a target phonology for the language using an agentic approach that refines its output at each step with commentary feedback on its previous attempt. Next, a set of sentences is 'translated' from their English original into a morphosyntactic markup that reflects the word order and morphosyntactic feature specifications of the desired target language, with affixes represented as morphosyntactic feature bundles. From this translated corpus, a lexicon is constructed using the phonological model and the set of morphemes (stems and affixes) extracted from the 'translated' sentences. The system is then instructed to provide an orthography for the language, using an existing script such as Latin or Cyrillic. Finally, the system writes a brief grammatical handbook of the language. The system can also translate further sentences into the target language.
Our goal is twofold. First, we hope that these tools will be fun to use for creating artificially constructed languages. Second, we are interested in exploring what LLMs 'know' about language-not what they know about any particular language or linguistic phenomenon, but how much they know about and understand language and linguistic concepts. As we shall see, there is a fairly wide gulf in capabilities both among different LLMs and among different linguistic specifications, with it being notably easier for systems to deal with more common patterns than rarer ones. An additional avenue that we explore is the application of our approach to translating from high-resource into low-resource languages. While the results so far are mostly negative, we provide some evidence that an improved version of the present system could afford some real gains in such tasks.
this https URL

**arXiv ID:** 2510.07591
</details>

<details>
<summary><strong>Beyond hospital reach: Autonomous lightweight ultrasound robot for liver sonography</strong> - Zihan Li, Yixiao Xu, Lei Zhang, Taiyu Han, Xinshan Yang, Yingni Wang, Mingxuan Liu, Shenghai Xin, Linxun Liu, Hongen Liao, Guochen Ning - [[pdf]](https://arxiv.org/pdf/2510.08106)</summary>

**Abstract:** Liver disease is a major global health burden. While ultrasound is the first-line diagnostic tool, liver sonography requires locating multiple non-continuous planes from positions where target structures are often not visible, for biometric assessment and lesion detection, requiring significant expertise. However, expert sonographers are severely scarce in resource-limited regions. Here, we develop an autonomous lightweight ultrasound robot comprising an AI agent that integrates multi-modal perception with memory attention for localization of unseen target structures, and a 588-gram 6-degrees-of-freedom cable-driven robot. By mounting on the abdomen, the system enhances robustness against motion. Our robot can autonomously acquire expert-level standard liver ultrasound planes and detect pathology in patients, including two from Xining, a 2261-meter-altitude city with limited medical resources. Our system performs effectively on rapid-motion individuals and in wilderness environments. This work represents the first demonstration of autonomous sonography across multiple challenging scenarios, potentially transforming access to expert-level diagnostics in underserved regions.

**arXiv ID:** 2510.08106
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (34 papers)</h2></summary>

<details>
<summary><strong>Control Synthesis of Cyber-Physical Systems for Real-Time Specifications through Causation-Guided Reinforcement Learning</strong> - Xiaochen Tang, Zhenya Zhang, Miaomiao Zhang, Jie An - [[pdf]](https://arxiv.org/pdf/2510.07715)</summary>

**Abstract:** In real-time and safety-critical cyber-physical systems (CPSs), control synthesis must guarantee that generated policies meet stringent timing and correctness requirements under uncertain and dynamic conditions. Signal temporal logic (STL) has emerged as a powerful formalism of expressing real-time constraints, with its semantics enabling quantitative assessment of system behavior. Meanwhile, reinforcement learning (RL) has become an important method for solving control synthesis problems in unknown environments. Recent studies incorporate STL-based reward functions into RL to automatically synthesize control policies. However, the automatically inferred rewards obtained by these methods represent the global assessment of a whole or partial path but do not accumulate the rewards of local changes accurately, so the sparse global rewards may lead to non-convergence and unstable training performances. In this paper, we propose an online reward generation method guided by the online causation monitoring of STL. Our approach continuously monitors system behavior against an STL specification at each control step, computing the quantitative distance toward satisfaction or violation and thereby producing rewards that reflect instantaneous state dynamics. Additionally, we provide a smooth approximation of the causation semantics to overcome the discontinuity of the causation semantics and make it differentiable for using deep-RL methods. We have implemented a prototype tool and evaluated it in the Gym environment on a variety of continuously controlled benchmarks. Experimental results show that our proposed STL-guided RL method with online causation semantics outperforms existing relevant STL-guided RL methods, providing a more robust and efficient reward generation framework for deep-RL.

**arXiv ID:** 2510.07715
</details>

<details>
<summary><strong>An LLM-Powered Cooperative Framework for Large-Scale Multi-Vehicle Navigation</strong> - Yuping Zhou, Siqi Lai, Jindong Han, Hao Liu - [[pdf]](https://arxiv.org/pdf/2510.07825)</summary>

**Abstract:** The rise of Internet of Vehicles (IoV) technologies is transforming traffic management from isolated control to a collective, multi-vehicle process. At the heart of this shift is multi-vehicle dynamic navigation, which requires simultaneously routing large fleets under evolving traffic conditions. Existing path search algorithms and reinforcement learning methods struggle to scale to city-wide networks, often failing to capture the nonlinear, stochastic, and coupled dynamics of urban traffic. To address these challenges, we propose CityNav, a hierarchical, LLM-powered framework for large-scale multi-vehicle navigation. CityNav integrates a global traffic allocation agent, which coordinates strategic traffic flow distribution across regions, with local navigation agents that generate locally adaptive routes aligned with global directives. To enable effective cooperation, we introduce a cooperative reasoning optimization mechanism, in which agents are jointly trained with a dual-reward structure: individual rewards promote per-vehicle efficiency, while shared rewards encourage network-wide coordination and congestion reduction. Extensive experiments on four real-world road networks of varying scales (up to 1.6 million roads and 430,000 intersections) and traffic datasets demonstrate that CityNav consistently outperforms nine classical path search and RL-based baselines in city-scale travel efficiency and congestion mitigation. Our results highlight the potential of LLMs to enable scalable, adaptive, and cooperative city-wide traffic navigation, providing a foundation for intelligent, large-scale vehicle routing in complex urban environments. Our project is available at this https URL.

**arXiv ID:** 2510.07825
</details>

<details>
<summary><strong>TaoSR-SHE: Stepwise Hybrid Examination Reinforcement Learning Framework for E-commerce Search Relevance</strong> - Pengkun Jiao, Yiming Jin, Jianhui Yang, Chenhe Dong, Zerui Huang, Shaowei Yao, Xiaojiang Zhou, Dan Ou, Haihong Tang - [[pdf]](https://arxiv.org/pdf/2510.07972)</summary>

**Abstract:** Query-product relevance analysis is a foundational technology in e-commerce search engines and has become increasingly important in AI-driven e-commerce. The recent emergence of large language models (LLMs), particularly their chain-of-thought (CoT) reasoning capabilities, offers promising opportunities for developing relevance systems that are both more interpretable and more robust. However, existing training paradigms have notable limitations: SFT and DPO suffer from poor generalization on long-tail queries and from a lack of fine-grained, stepwise supervision to enforce rule-aligned reasoning. In contrast, reinforcement learning with verification rewards (RLVR) suffers from sparse feedback, which provides insufficient signal to correct erroneous intermediate steps, thereby undermining logical consistency and limiting performance in complex inference scenarios.
To address these challenges, we introduce the Stepwise Hybrid Examination Reinforcement Learning framework for Taobao Search Relevance (TaoSR-SHE). At its core is Stepwise Reward Policy Optimization (SRPO), a reinforcement learning algorithm that leverages step-level rewards generated by a hybrid of a high-quality generative stepwise reward model and a human-annotated offline verifier, prioritizing learning from critical correct and incorrect reasoning steps. TaoSR-SHE further incorporates two key techniques: diversified data filtering to encourage exploration across varied reasoning paths and mitigate policy entropy collapse, and multi-stage curriculum learning to foster progressive capability growth. Extensive experiments on real-world search benchmarks show that TaoSR-SHE improves both reasoning quality and relevance-prediction accuracy in large-scale e-commerce settings, outperforming SFT, DPO, GRPO, and other baselines, while also enhancing interpretability and robustness.

**arXiv ID:** 2510.07972
</details>

<details>
<summary><strong>VoiceAgentBench: Are Voice Assistants ready for agentic tasks?</strong> - Dhruv Jain, Harshit Shukla, Gautam Rajeev, Ashish Kulkarni, Chandra Khatri, Shubham Agarwal - [[pdf]](https://arxiv.org/pdf/2510.07978)</summary>

**Abstract:** Large-scale Speech Language Models (SpeechLMs) have enabled voice assistants capable of understanding natural spoken queries and performing complex tasks. However, existing speech benchmarks primarily focus on isolated capabilities such as transcription, or question-answering, and do not systematically evaluate agentic scenarios encompassing multilingual and cultural understanding, as well as adversarial robustness. To address this, we introduce VoiceAgentBench, a comprehensive benchmark designed to evaluate SpeechLMs in realistic spoken agentic settings. It comprises over 5,500 synthetic spoken queries, including dialogues grounded in Indian context, covering single-tool invocations, multi-tool workflows, multi-turn interactions, and safety evaluations. The benchmark supports English, Hindi, and 5 other Indian languages, reflecting real-world linguistic and cultural diversity. We simulate speaker variability using a novel sampling algorithm that selects audios for TTS voice conversion based on its speaker embeddings, maximizing acoustic and speaker diversity. Our evaluation measures tool selection accuracy, structural consistency, and the correctness of tool invocations, including adversarial robustness. Our experiments reveal significant gaps in contextual tool orchestration tasks, Indic generalization, and adversarial robustness, exposing critical limitations of current SpeechLMs.

**arXiv ID:** 2510.07978
</details>

<details>
<summary><strong>QAgent: A modular Search Agent with Interactive Query Understanding</strong> - Yi Jiang, Lei Shen, Lujie Niu, Sendong Zhao, Wenbo Su, Bo Zheng - [[pdf]](https://arxiv.org/pdf/2510.08383)</summary>

**Abstract:** Large language models (LLMs) excel at natural language tasks but are limited by their static parametric knowledge, especially in knowledge-intensive task. Retrieval-augmented generation (RAG) mitigates this by integrating external information. However, (1) traditional RAG struggles with complex query understanding, and (2) even search agents trained with reinforcement learning (RL), despite their promise, still face generalization and deployment challenges. To address these limitations, we propose QAgent, a unified agentic RAG framework that employs a search agent for adaptive retrieval. This agent optimizes its understanding of the query through interactive reasoning and retrieval. To facilitate real-world application, we focus on modular search agent for query understanding that are plug-and-play in complex systems. Secifically, the agent follows a multi-step decision process trained with RL to maximize retrieval quality and support accurate downstream answers. We further analyze the strengths and weaknesses of end-to-end RL and propose a strategy that focuses on effective retrieval, thereby enhancing generalization in LLM applications. Experiments show QAgent excels at QA and serves as a plug-and-play module for real-world deployment.

**arXiv ID:** 2510.08383
</details>

<details>
<summary><strong>AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents</strong> - Shangheng Du, Xiangchao Yan, Dengyang Jiang, Jiakang Yuan, Yusong Hu, Xin Li, Liang He, Bo Zhang, Lei Bai - [[pdf]](https://arxiv.org/pdf/2510.08511)</summary>

**Abstract:** Large language models (LLMs) have shown impressive performance in general programming tasks. However, in Machine Learning Engineering (MLE) scenarios such as AutoML and Kaggle competitions, achieving high performance depends heavily on expert intervention and repeated adjustments rather than simply generating correct code. When applied directly to these tasks, LLMs often lack fine-grained domain priors, and existing MLE approaches that use linear or tree-structured searches limit knowledge transfer to adjacent hierarchical links. As a result, they cannot leverage past full trajectories or share information across branches, limiting self-evolving ability and search space diversity. To address these limitations, we introduce AutoMLGen, an LLM-based coding agent that integrates a domain knowledge base for high-quality prior guidance and Monte Carlo Graph Search (MCGS) for efficient exploration. MCGS retains the tree-guided exploration of MCTS while embedding a graph structure into the expansion stage to enable dynamic path reorganization, historical trajectory reuse, and multi-solution fusion to support both self-evolution and collaborative learning. Combined with fine-grained operator sets, this design improves stability and accelerates convergence. Evaluation on the MLE-Bench shows that AutoMLGen achieves state-of-the-art performance in numerous dimensions, such as the average medal rate and the valid submission rate, under a 12-hour budget (half the standard runtime). The code is available at this https URL.

**arXiv ID:** 2510.08511
</details>

<details>
<summary><strong>Agent Learning via Early Experience</strong> - Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Xiaohan Fu, Jian Xie, Yuxuan Sun, Boyu Gou, Qi Qi, Zihang Meng, Jianwei Yang, Ning Zhang, Xian Li, Ashish Shah, Dat Huynh, Hengduo Li, Zi Yang, Sara Cao, Lawrence Jang, Shuyan Zhou, Jiacheng Zhu, Huan Sun, Jason Weston, Yu Su, Yifan Wu - [[pdf]](https://arxiv.org/pdf/2510.08558)</summary>

**Abstract:** A long-term goal of language agents is to learn and improve through their own experience, ultimately outperforming humans in complex, real-world tasks. However, training agents from experience data with reinforcement learning remains difficult in many environments, which either lack verifiable rewards (e.g., websites) or require inefficient long-horizon rollouts (e.g., multi-turn tool use). As a result, most current agents rely on supervised fine-tuning on expert data, which is challenging to scale and generalizes poorly. This limitation stems from the nature of expert demonstrations: they capture only a narrow range of scenarios and expose the agent to limited environment diversity. We address this limitation with a middle-ground paradigm we call early experience: interaction data generated by the agent's own actions, where the resulting future states serve as supervision without reward signals. Within this paradigm we study two strategies of using such data: (1) Implicit world modeling, which uses collected states to ground the policy in environment dynamics; and (2) Self-reflection, where the agent learns from its suboptimal actions to improve reasoning and decision-making. We evaluate across eight diverse environments and multiple model families. Our approaches consistently improve effectiveness and out-of-domain generalization, highlighting the value of early experience. Moreover, in environments with verifiable rewards, our results provide promising signals that early experience offers a strong foundation for subsequent reinforcement learning, positioning it as a practical bridge between imitation learning and fully experience-driven agents.

**arXiv ID:** 2510.08558
</details>

<details>
<summary><strong>A$^2$Search: Ambiguity-Aware Question Answering with Reinforcement Learning</strong> - Fengji Zhang, Xinyao Niu, Chengyang Ying, Guancheng Lin, Zhongkai Hao, Zhou Fan, Chengen Huang, Jacky Keung, Bei Chen, Junyang Lin - [[pdf]](https://arxiv.org/pdf/2510.07958)</summary>

**Abstract:** Recent advances in Large Language Models (LLMs) and Reinforcement Learning (RL) have led to strong performance in open-domain question answering (QA). However, existing models still struggle with questions that admit multiple valid answers. Standard QA benchmarks, which typically assume a single gold answer, overlook this reality and thus produce inappropriate training signals. Existing attempts to handle ambiguity often rely on costly manual annotation, which is difficult to scale to multi-hop datasets such as HotpotQA and MuSiQue. In this paper, we present A$^2$Search, an annotation-free, end-to-end training framework to recognize and handle ambiguity. At its core is an automated pipeline that detects ambiguous questions and gathers alternative answers via trajectory sampling and evidence verification. The model is then optimized with RL using a carefully designed $\mathrm{AnsF1}$ reward, which naturally accommodates multiple answers. Experiments on eight open-domain QA benchmarks demonstrate that A$^2$Search achieves new state-of-the-art performance. With only a single rollout, A$^2$Search-7B yields an average $\mathrm{AnsF1}@1$ score of $48.4\%$ across four multi-hop benchmarks, outperforming all strong baselines, including the substantially larger ReSearch-32B ($46.2\%$). Extensive analyses further show that A$^2$Search resolves ambiguity and generalizes across benchmarks, highlighting that embracing ambiguity is essential for building more reliable QA systems. Our code, data, and model weights can be found at this https URL

**arXiv ID:** 2510.07958
</details>

<details>
<summary><strong>TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevance</strong> - Jianhui Yang, Yiming Jin, Pengkun Jiao, Chenhe Dong, Zerui Huang, Shaowei Yao, Xiaojiang Zhou, Dan Ou, Haihong Tang - [[pdf]](https://arxiv.org/pdf/2510.08048)</summary>

**Abstract:** Query-product relevance prediction is fundamental to e-commerce search and has become even more critical in the era of AI-powered shopping, where semantic understanding and complex reasoning directly shape the user experience and business conversion. Large Language Models (LLMs) enable generative, reasoning-based approaches, typically aligned via supervised fine-tuning (SFT) or preference optimization methods like Direct Preference Optimization (DPO). However, the increasing complexity of business rules and user queries exposes the inability of existing methods to endow models with robust reasoning capacity for long-tail and challenging cases. Efforts to address this via reinforcement learning strategies like Group Relative Policy Optimization (GRPO) often suffer from sparse terminal rewards, offering insufficient guidance for multi-step reasoning and slowing convergence. To address these challenges, we propose TaoSR-AGRL, an Adaptive Guided Reinforcement Learning framework for LLM-based relevance prediction in Taobao Search Relevance. TaoSR-AGRL introduces two key innovations: (1) Rule-aware Reward Shaping, which decomposes the final relevance judgment into dense, structured rewards aligned with domain-specific relevance criteria; and (2) Adaptive Guided Replay, which identifies low-accuracy rollouts during training and injects targeted ground-truth guidance to steer the policy away from stagnant, rule-violating reasoning patterns toward compliant trajectories. TaoSR-AGRL was evaluated on large-scale real-world datasets and through online side-by-side human evaluations on Taobao Search. It consistently outperforms DPO and standard GRPO baselines in offline experiments, improving relevance accuracy, rule adherence, and training stability. The model trained with TaoSR-AGRL has been successfully deployed in the main search scenario on Taobao, serving hundreds of millions of users.

**arXiv ID:** 2510.08048
</details>

<details>
<summary><strong>Quantum Agents for Algorithmic Discovery</strong> - Iordanis Kerenidis, El-Amine Cherrat - [[pdf]](https://arxiv.org/pdf/2510.08159)</summary>

**Abstract:** We introduce quantum agents trained by episodic, reward-based reinforcement learning to autonomously rediscover several seminal quantum algorithms and protocols. In particular, our agents learn: efficient logarithmic-depth quantum circuits for the Quantum Fourier Transform; Grover's search algorithm; optimal cheating strategies for strong coin flipping; and optimal winning strategies for the CHSH and other nonlocal games. The agents achieve these results directly through interaction, without prior access to known optimal solutions. This demonstrates the potential of quantum intelligence as a tool for algorithmic discovery, opening the way for the automated design of novel quantum algorithms and protocols.

**arXiv ID:** 2510.08159
</details>

<details>
<summary><strong>Expressive Value Learning for Scalable Offline Reinforcement Learning</strong> - Nicolas Espinosa-Dice, Kiante Brantley, Wen Sun - [[pdf]](https://arxiv.org/pdf/2510.08218)</summary>

**Abstract:** Reinforcement learning (RL) is a powerful paradigm for learning to make sequences of decisions. However, RL has yet to be fully leveraged in robotics, principally due to its lack of scalability. Offline RL offers a promising avenue by training agents on large, diverse datasets, avoiding the costly real-world interactions of online RL. Scaling offline RL to increasingly complex datasets requires expressive generative models such as diffusion and flow matching. However, existing methods typically depend on either backpropagation through time (BPTT), which is computationally prohibitive, or policy distillation, which introduces compounding errors and limits scalability to larger base policies. In this paper, we consider the question of how to develop a scalable offline RL approach without relying on distillation or backpropagation through time. We introduce Expressive Value Learning for Offline Reinforcement Learning (EVOR): a scalable offline RL approach that integrates both expressive policies and expressive value functions. EVOR learns an optimal, regularized Q-function via flow matching during training. At inference-time, EVOR performs inference-time policy extraction via rejection sampling against the expressive value function, enabling efficient optimization, regularization, and compute-scalable search without retraining. Empirically, we show that EVOR outperforms baselines on a diverse set of offline RL tasks, demonstrating the benefit of integrating expressive value learning into offline RL.

**arXiv ID:** 2510.08218
</details>

<details>
<summary><strong>DeepEN: Personalized Enteral Nutrition for Critically Ill Patients using Deep Reinforcement Learning</strong> - Daniel Jason Tan, Jiayang Chen, Dilruk Perera, Kay Choong See, Mengling Feng - [[pdf]](https://arxiv.org/pdf/2510.08350)</summary>

**Abstract:** We introduce DeepEN, a deep reinforcement learning (RL) framework for personalized enteral nutrition (EN) in critically ill patients. Trained offline on over 11,000 ICU patients from the MIMIC-IV database, DeepEN generates 4-hourly recommendations for caloric, protein, and fluid intake tailored to each patient's evolving physiology. The model integrates a curated, clinically informed state space with a custom reward function that balances short-term physiological and nutrition-related goals with long-term survival outcomes. Using a dueling double deep Q-network with conservative Q-learning regularization, DeepEN learns clinically realistic policies that align with high-value clinician actions while discouraging unsafe deviations. Across various qualitative and quantitative metrics, DeepEN outperforms clinician-derived and guideline-based policies, achieving a 3.7 $\pm$ 0.17 percentage-point reduction in estimated mortality (18.8% vs 22.5%) and improvements in key nutritional biomarkers. These findings highlight the potential of safe, data-driven personalization of EN therapy to improve outcomes beyond traditional guideline- or heuristic-based approaches.

**arXiv ID:** 2510.08350
</details>

<details>
<summary><strong>xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning</strong> - Cheng Qian, Zuxin Liu, Shirley Kokane, Akshara Prabhakar, Jielin Qiu, Haolin Chen, Zhiwei Liu, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang - [[pdf]](https://arxiv.org/pdf/2510.08439)</summary>

**Abstract:** Modern LLM deployments confront a widening cost-performance spectrum: premium models deliver strong reasoning but are expensive, while lightweight models are economical yet brittle on complex tasks. Static escalation rules and keyword heuristics under-utilize this spectrum and fail to adapt across task types. We present xRouter, a tool-calling-based routing system in which a learned router can either answer directly or invoke one or more external models. The router is trained end-to-end with reinforcement learning using an explicit, cost-aware reward that encodes cost-performance trade-offs, eliminating the need for hand-engineered routing rules. Our implementation encompasses the full reinforcement learning framework, including reward and cost accounting, as well as the deployment and evaluation pipelines. Across diverse benchmarks, xRouter achieves strong cost-performance trade-offs (e.g., substantial cost reductions at comparable task completion rates), and provides empirical insights into what reliably helps learned routing and what does not, ranging from model trainability to the difficulty of eliciting sophisticated orchestration behaviors in small open models. We hope these findings and our open implementation will serve as a practical substrate for advancing learned, cost-aware LLM orchestration.

**arXiv ID:** 2510.08439
</details>

<details>
<summary><strong>Towards Urban Planing AI Agent in the Age of Agentic AI</strong> - Rui Liu, Tao Zhe, Zhong-Ren Peng, Necati Catbas, Xinyue Ye, Dongjie Wang, Yanjie Fu - [[pdf]](https://arxiv.org/pdf/2507.14730)</summary>

**Abstract:** Generative AI, large language models, and agentic AI have emerged separately of urban planning. However, the convergence between AI and urban planning presents an interesting opportunity towards AI urban planners. Existing studies conceptualizes urban planning as a generative AI task, where AI synthesizes land-use configurations under geospatial, social, and human-centric constraints and reshape automated urban design. We further identify critical gaps of existing generative urban planning studies: 1) the generative structure has to be predefined with strong assumption: all of adversarial generator-discriminator, forward and inverse diffusion structures, hierarchical zone-POI generative structure are predefined by humans; 2) ignore the power of domain expert developed tools: domain urban planners have developed various tools in the urban planning process guided by urban theory, while existing pure neural networks based generation ignore the power of the tools developed by urban planner practitioners. To address these limitations, we outline a future research direction agentic urban AI planner, calling for a new synthesis of agentic AI and participatory urbanism.

**arXiv ID:** 2507.14730
</details>

<details>
<summary><strong>WebRenderBench: Enhancing Web Interface Generation through Layout-Style Consistency and Reinforcement Learning</strong> - Peichao Lai, Jinhui Zhuang, Kexuan Zhang, Ningchang Xiong, Shengjie Wang, Yanwei Xu, Chong Chen, Yilei Wang, Bin Cui - [[pdf]](https://arxiv.org/pdf/2510.04097)</summary>

**Abstract:** Automating the conversion of UI images into web code is a critical task for front-end development and rapid prototyping. Advances in multimodal large language models (MLLMs) have made WebUI-to-Code increasingly feasible, yet existing benchmarks remain limited in data diversity and evaluation reliability. To address these issues, we present WebRenderBench, a large-scale benchmark of 45.1k webpages collected from real-world portal sites, offering greater diversity, complexity, and realism than prior benchmarks. We further propose a novel evaluation metric that measures layout and style consistency from the final rendered pages. Unlike vision-based methods that rely on costly LLM reasoning or structure-based comparisons vulnerable to noise and asymmetry, our approach enables more efficient, objective, and reliable UI quality assessment. Finally, we introduce the Automated Layout and Style Inspection Agent (ALISA), which integrates this metric into reinforcement learning as a reward signal to enhance training on crawled asymmetric webpages. Experiments show that ALISA significantly boosts generation performance, achieving state-of-the-art results across multiple metrics.

**arXiv ID:** 2510.04097
</details>

<details>
<summary><strong>Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support</strong> - Cen Mia Zhao, Tiantian Zhang, Hanchen Su, Yufeng Wayne Zhang, Shaowei Su, Mingzhi Xu, Yu Elaine Liu, Wei Han, Jeremy Werner, Claire Na Cheng, Yashar Mehdad - [[pdf]](https://arxiv.org/pdf/2510.06674)</summary>

**Abstract:** We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system.

**arXiv ID:** 2510.06674
</details>

<details>
<summary><strong>Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning</strong> - Yulei Qin, Xiaoyu Tan, Zhengbao He, Gang Li, Haojia Lin, Zongyi Li, Zihan Xu, Yuchen Shi, Siqi Cai, Renting Rui, Shaofei Cai, Yuzheng Cai, Xuan Zhang, Sheng Ye, Ke Li, Xing Sun - [[pdf]](https://arxiv.org/pdf/2509.22601)</summary>

**Abstract:** Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL training instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a curriculum-based self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL framework, where a replay buffer stores self-generated promising trajectories for off-policy update, by gradually steering the policy evolution within a well-balanced range of entropy across stages. Specifically, our approach incorporates a curriculum to manage the exploration process, utilizing intrinsic rewards to foster skill-level exploration and facilitating action-level exploration through SIL. At first, the auxiliary tool call reward plays a critical role in the accumulation of tool-use skills, enabling broad exposure to the unfamiliar distributions of the environment feedback with an upward entropy trend. As training progresses, self-imitation gets strengthened to exploit existing successful patterns from replayed experiences for comparative action-level exploration, accelerating solution iteration without unbounded entropy growth. To further stabilize training, we recalibrate the advantages of experiences in the replay buffer to address the potential policy drift. Reugularizations such as the clipping of tokens with high covariance between probability and advantage are introduced to the trajectory-level entropy control to curb over-confidence.

**arXiv ID:** 2509.22601
</details>

<details>
<summary><strong>ToolExpander: Extending the Frontiers of Tool-Using Reinforcement Learning to Weak LLMs</strong> - Fu Chen, Peng Wang, Xiyin Li, Wen Li, Shichi Lei, Dongdong Xiang - [[pdf]](https://arxiv.org/pdf/2510.07737)</summary>

**Abstract:** Training Large Language Models (LLMs) with Group Relative Policy Optimization (GRPO) encounters a significant challenge: models often fail to produce accurate responses, particularly in small-scale architectures. This limitation not only diminishes performance improvements and undermines the potential of GRPO but also frequently leads to mid-training collapse, adversely affecting stability and final efficacy. To address these issues, we propose ToolExpander, a novel framework that advances tool-oriented reinforcement learning for resource-constrained LLMs through two key innovations:(1) Dynamic Multi-Round Hard Sampling, which dynamically substitutes challenging samples(those without correct outputs over 10 rollouts) with high-quality few-shot demonstrations during training, coupled with an exponential learning rate decay strategy to mitigate oscillations;(2) Self-Exemplifying Thinking, an enhanced GRPO framework that eliminates KL divergence and incorporates adjusted clipping coefficients, encouraging models to autonomously generate and analyze few-shot examples via a minimal additional reward (0.01).Experimental results demonstrate that ToolExpander significantly enhances tool-using capabilities in LLMs, especially in weaker small-scale models, improving both training stability and overall performance.

**arXiv ID:** 2510.07737
</details>

<details>
<summary><strong>Beyond Turn Limits: Training Deep Search Agents with Dynamic Context Window</strong> - Qiaoyu Tang, Hao Xiang, Le Yu, Bowen Yu, Yaojie Lu, Xianpei Han, Le Sun, WenJuan Zhang, Pengbo Wang, Shixuan Liu, Zhenru Zhang, Jianhong Tu, Hongyu Lin, Junyang Lin - [[pdf]](https://arxiv.org/pdf/2510.08276)</summary>

**Abstract:** While recent advances in reasoning models have demonstrated cognitive behaviors through reinforcement learning, existing approaches struggle to invoke deep reasoning capabilities in multi-turn agents with long-horizon interactions. We propose DeepMiner, a novel framework that elicits such abilities by introducing high-difficulty training tasks and dynamic context window. DeepMiner presents a reverse construction method to generate complex but verifiable question-answer pairs from authentic web sources, which ensures the challenge and reliability of training data while injecting cognitive capabilities into multi-turn reasoning scenarios. We further design an elegant yet effective dynamic context management strategy for both training and inference, utilizing sliding window mechanisms while eliminating the dependency on external summarization models, thereby efficiently empowering the model to handle continuously expanding long-horizon contexts. Through reinforcement learning on Qwen3-32B, we develop DeepMiner-32B, which achieves substantial performance improvements across multiple search agent benchmarks. DeepMiner attains 33.5% accuracy on BrowseComp-en, surpassing the previous best open-source agent by almost 20 percentage points, and demonstrates consistent improvements on BrowseComp-zh, XBench-DeepSearch, and GAIA. Notably, our dynamic context management enables sustained interactions of nearly 100 turns within standard 32k context length, effectively addressing the context limitations that constrain existing multi-turn interaction systems.

**arXiv ID:** 2510.08276
</details>

<details>
<summary><strong>LiveThinking: Enabling Real-Time Efficient Reasoning for AI-Powered Livestreaming via Reinforcement Learning</strong> - Yuhan Sun, Zhiwei Huang, Wanqing Cui, Shaopan Xiong, Yazhi Guo, Meiguang Jin, Junfeng Ma - [[pdf]](https://arxiv.org/pdf/2510.07685)</summary>

**Abstract:** In AI-powered e-commerce livestreaming, digital avatars require real-time responses to drive engagement, a task for which high-latency Large Reasoning Models (LRMs) are ill-suited. We introduce LiveThinking, a practical two-stage optimization framework to bridge this gap. First, we address computational cost by distilling a 670B teacher LRM into a lightweight 30B Mixture-of-Experts (MoE) model (3B active) using Rejection Sampling Fine-Tuning (RFT). This reduces deployment overhead but preserves the teacher's verbose reasoning, causing latency. To solve this, our second stage employs reinforcement learning with Group Relative Policy Optimization (GRPO) to compress the model's reasoning path, guided by a multi-objective reward function balancing correctness, helpfulness, and brevity. LiveThinking achieves a 30-fold reduction in computational cost, enabling sub-second latency. In real-world application on Taobao Live, it improved response correctness by 3.3% and helpfulness by 21.8%. Tested by hundreds of thousands of viewers, our system led to a statistically significant increase in Gross Merchandise Volume (GMV), demonstrating its effectiveness in enhancing user experience and commercial performance in live, interactive settings.

**arXiv ID:** 2510.07685
</details>

<details>
<summary><strong>WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning</strong> - Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, Lihong Li - [[pdf]](https://arxiv.org/pdf/2505.16421)</summary>

**Abstract:** While reinforcement learning (RL) has demonstrated remarkable success in enhancing large language models (LLMs), it has primarily focused on single-turn tasks such as solving math problems. Training effective web agents for multi-turn interactions remains challenging due to the complexity of long-horizon decision-making across dynamic web interfaces. In this work, we present WebAgent-R1, a simple yet effective end-to-end multi-turn RL framework for training web agents. It learns directly from online interactions with web environments by asynchronously generating diverse trajectories, entirely guided by binary rewards depending on task success. Experiments on the WebArena-Lite benchmark demonstrate the effectiveness of WebAgent-R1, boosting the task success rate of Qwen-2.5-3B from 6.1% to 33.9% and Llama-3.1-8B from 8.5% to 44.8%, significantly outperforming existing state-of-the-art methods and strong proprietary models such as OpenAI o3. In-depth analyses reveal the effectiveness of the thinking-based prompting strategy and test-time scaling through increased interactions for web tasks. We further investigate different RL initialization policies by introducing two variants, namely WebAgent-R1-Zero and WebAgent-R1-CoT, which highlight the importance of the warm-up training stage (i.e., behavior cloning) and provide insights on incorporating long chain-of-thought (CoT) reasoning in web agents.

**arXiv ID:** 2505.16421
</details>

<details>
<summary><strong>Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty</strong> - Peilin Wu, Mian Zhang, Xinlu Zhang, Xinya Du, Zhiyu Zoey Chen - [[pdf]](https://arxiv.org/pdf/2505.17281)</summary>

**Abstract:** Agentic Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by enabling dynamic, multi-step reasoning and information retrieval. However, these systems often exhibit sub-optimal search behaviors like over-search (retrieving redundant information) and under-search (failing to retrieve necessary information), which hinder efficiency and reliability. This work formally defines and quantifies these behaviors, revealing their prevalence across multiple QA datasets and agentic RAG systems (e.g., one model could have avoided searching in 27.7% of its search steps). Furthermore, we demonstrate a crucial link between these inefficiencies and the models' uncertainty regarding their own knowledge boundaries, where response accuracy correlates with model's uncertainty in its search decisions. To address this, we propose $\beta$-GRPO, a reinforcement learning-based training method that incorporates confidence threshold to reward high-certainty search decisions. Experiments on seven QA benchmarks show that $\beta$-GRPO enable a 3B model with better agentic RAG ability, outperforming other strong baselines with a 4% higher average exact match score.

**arXiv ID:** 2505.17281
</details>

<details>
<summary><strong>Med-R$^3$: Enhancing Medical Retrieval-Augmented Reasoning of LLMs via Progressive Reinforcement Learning</strong> - Keer Lu, Zheng Liang, Youquan Li, Jiejun Tan, Da Pan, Shusen Zhang, Guosheng Dong, Zhonghai Wu, Huang Leng, Bin Cui, Wentao Zhang - [[pdf]](https://arxiv.org/pdf/2507.23541)</summary>

**Abstract:** In medical scenarios, effectively retrieving external knowledge and leveraging it for rigorous logical reasoning is of significant importance. Despite their potential, existing work has predominantly focused on enhancing either retrieval or reasoning capabilities of the models in isolation, with little attention given to their joint optimization, which leads to limited coordination between the two processes. Additionally, current methods rely heavily on supervised fine-tuning (SFT), which can cause models to memorize existing problem-solving pathways, thereby restricting their generalization ability when confronted with novel problem contexts. Furthermore, while some studies have explored to improve retrieval-augmented reasoning in general domains via reinforcement learning, their reward function designs do not adequately capture the specific demands of the medical domain. To address these challenges, we introduce **Med-R$^3$**, a **Med**ical **R**etrieval-augmented **R**easoning framework driven by progressive **R**einforcement learning. In this framework, we first develop the model's ability to perform logical reasoning over medical problems. Subsequently, on the basis of this foundation, we adaptively optimize the retrieval capability to better align with the characteristics of knowledge corpus and external information utilization throughout the reasoning process. Finally, we conduct joint optimization of the model's retrieval and reasoning coordination. Extensive experiments indicate that **Med-R$^3$** could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + Med-R$^3$ surpassing closed-sourced GPT-4o-mini by 3.93\% at a comparable parameter scale, while Qwen2.5-14B augmented with Med-R$^3$ shows a more substantial gain of 13.53\%.

**arXiv ID:** 2507.23541
</details>

<details>
<summary><strong>A Survey of Reinforcement Learning for Large Reasoning Models</strong> - Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, Yu Fu, Xingtai Lv, Yuchen Zhang, Sihang Zeng, Shang Qu, Haozhan Li, Shijie Wang, Yuru Wang, Xinwei Long, Fangfu Liu, Xiang Xu, Jiaze Ma, Xuekai Zhu, Ermo Hua, Yihao Liu, Zonglin Li, Huayu Chen, Xiaoye Qu, Yafu Li, Weize Chen, Zhenzhao Yuan, Junqi Gao, Dong Li, Zhiyuan Ma, Ganqu Cui, Zhiyuan Liu, Biqing Qi, Ning Ding, Bowen Zhou - [[pdf]](https://arxiv.org/pdf/2509.08827)</summary>

**Abstract:** In this paper, we survey recent advances in Reinforcement Learning (RL) for reasoning with Large Language Models (LLMs). RL has achieved remarkable success in advancing the frontier of LLM capabilities, particularly in addressing complex logical tasks such as mathematics and coding. As a result, RL has emerged as a foundational methodology for transforming LLMs into LRMs. With the rapid progress of the field, further scaling of RL for LRMs now faces foundational challenges not only in computational resources but also in algorithm design, training data, and infrastructure. To this end, it is timely to revisit the development of this domain, reassess its trajectory, and explore strategies to enhance the scalability of RL toward Artificial SuperIntelligence (ASI). In particular, we examine research applying RL to LLMs and LRMs for reasoning abilities, especially since the release of DeepSeek-R1, including foundational components, core problems, training resources, and downstream applications, to identify future opportunities and directions for this rapidly evolving area. We hope this review will promote future research on RL for broader reasoning models. Github: this https URL

**arXiv ID:** 2509.08827
</details>

<details>
<summary><strong>Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning</strong> - Jinyeop Song, Song Wang, Julian Shun, Yada Zhu - [[pdf]](https://arxiv.org/pdf/2509.26383)</summary>

**Abstract:** Knowledge-graph retrieval-augmented generation (KG-RAG) couples large language models (LLMs) with structured, verifiable knowledge graphs (KGs) to reduce hallucinations and expose reasoning traces. However, many KG-RAG systems compose multiple LLM modules (e.g planning, reasoning, and responding), inflating inference cost and binding behavior to a specific target KG. To address this, we introduce KG-R1, an agentic KG retrieval-augmented generation (KG-RAG) framework through reinforcement learning (RL). KG-R1 utilizes a single agent that interacts with KGs as its environment, learning to retrieve at each step and incorporating the retrieved information into its reasoning and generation. The process is optimized through end-to-end RL. In controlled experiments across Knowledge-Graph Question Answering (KGQA) benchmarks, our method demonstrates both efficiency and transferability: Using Qwen-2.5-3B, KG-R1 improves answer accuracy with fewer generation tokens than prior multi-module workflow methods that use larger foundation or fine-tuned models. Furthermore, KG-R1 enables plug and play: after training, it maintains strong accuracy on new KGs without modification. These properties make KG-R1 a promising KG-RAG framework for real-world deployment. Our code is publicly available at this https URL.

**arXiv ID:** 2509.26383
</details>

<details>
<summary><strong>Incremental Summarization for Customer Support via Progressive Note-Taking and Agent Feedback</strong> - Yisha Wu, Cen Mia Zhao, Yuanpei Cao, Xiaoqing Su, Yashar Mehdad, Mindy Ji, Claire Na Cheng - [[pdf]](https://arxiv.org/pdf/2510.06677)</summary>

**Abstract:** We introduce an incremental summarization system for customer support agents that intelligently determines when to generate concise bullet notes during conversations, reducing agents' context-switching effort and redundant review. Our approach combines a fine-tuned Mixtral-8x7B model for continuous note generation with a DeBERTa-based classifier to filter trivial content. Agent edits refine the online notes generation and regularly inform offline model retraining, closing the agent edits feedback loop. Deployed in production, our system achieved a 3% reduction in case handling time compared to bulk summarization (with reductions of up to 9% in highly complex cases), alongside high agent satisfaction ratings from surveys. These results demonstrate that incremental summarization with continuous feedback effectively enhances summary quality and agent productivity at scale.

**arXiv ID:** 2510.06677
</details>

<details>
<summary><strong>Reinforcement Learning-based Task Offloading in the Internet of Wearable Things</strong> - Waleed Bin Qaim, Aleksandr Ometov, Claudia Campolo, Antonella Molinaro, Elena Simona Lohan, Jari Nurmi - [[pdf]](https://arxiv.org/pdf/2510.07487)</summary>

**Abstract:** Over the years, significant contributions have been made by the research and industrial sectors to improve wearable devices towards the Internet of Wearable Things (IoWT) paradigm. However, wearables are still facing several challenges. Many stem from the limited battery power and insufficient computation resources available on wearable devices. On the other hand, with the popularity of smart wearables, there is a consistent increase in the development of new computationally intensive and latency-critical applications. In such a context, task offloading allows wearables to leverage the resources available on nearby edge devices to enhance the overall user experience. This paper proposes a framework for Reinforcement Learning (RL)-based task offloading in the IoWT. We formulate the task offloading process considering the tradeoff between energy consumption and task accomplishment time. Moreover, we model the task offloading problem as a Markov Decision Process (MDP) and utilize the Q-learning technique to enable the wearable device to make optimal task offloading decisions without prior knowledge. We evaluate the performance of the proposed framework through extensive simulations for various applications and system configurations conducted in the ns-3 network simulator. We also show how varying the main system parameters of the Q-learning algorithm affects the overall performance in terms of average task accomplishment time, average energy consumption, and percentage of tasks offloaded.

**arXiv ID:** 2510.07487
</details>

<details>
<summary><strong>GRADE: Personalized Multi-Task Fusion via Group-relative Reinforcement Learning with Adaptive Dirichlet Exploratio</strong> - Tingfeng Hong, Pingye Ren, Xinlong Xiao, Chao Wang, Chenyi Lei, Wenwu Ou, Han Li - [[pdf]](https://arxiv.org/pdf/2510.07919)</summary>

**Abstract:** Overall architecture of the personalized multi-objective ranking system. It comprises: (1) a Feature Center and Prerank Model for initial feature processing and candidate generation; (2) a Multi-Task Learning (MTL) model predicting various user feedback signals; (3) a Multi-Task Fusion (MTF) module (our proposed GRADE framework) that learns personalized weights ($w_1, \dots, w_n$); these weights are then applied to calculate final scores and sorted to generate a blended ranking by the Blended Ranking Model, which ultimately delivers results to users.

**arXiv ID:** 2510.07919
</details>

<details>
<summary><strong>Reinforcement Learning from Probabilistic Forecasts for Safe Decision-Making via Conditional Value-at-Risk Planning</strong> - Michal Koren, Or Peretz, Tai Dinh, Philip S. Yu - [[pdf]](https://arxiv.org/pdf/2510.08226)</summary>

**Abstract:** Sequential decisions in volatile, high-stakes settings require more than maximizing expected return; they require principled uncertainty management. This paper presents the Uncertainty-Aware Markov Decision Process (UAMDP), a unified framework that couples Bayesian forecasting, posterior-sampling reinforcement learning, and planning under a conditional value-at-risk (CVaR) constraint. In a closed loop, the agent updates its beliefs over latent dynamics, samples plausible futures via Thompson sampling, and optimizes policies subject to preset risk tolerances. We establish regret bounds that converge to the Bayes-optimal benchmark under standard regularity conditions. We evaluate UAMDP in two domains-high-frequency equity trading and retail inventory control-both marked by structural uncertainty and economic volatility. Relative to strong deep learning baselines, UAMDP improves long-horizon forecasting accuracy (RMSE decreases by up to 25\% and sMAPE by 32\%), and these gains translate into economic performance: the trading Sharpe ratio rises from 1.54 to 1.74 while maximum drawdown is roughly halved. These results show that integrating calibrated probabilistic modeling, exploration aligned with posterior uncertainty, and risk-aware control yields a robust, generalizable approach to safer and more profitable sequential decision-making.

**arXiv ID:** 2510.08226
</details>

<details>
<summary><strong>Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning</strong> - Yash Jhaveri, Harley Wiltzer, Patrick Shafto, Marc G. Bellemare, David Meger - [[pdf]](https://arxiv.org/pdf/2510.08526)</summary>

**Abstract:** In the pursuit of finding an optimal policy, reinforcement learning (RL) methods generally ignore the properties of learned policies apart from their expected return. Thus, even when successful, it is difficult to characterize which policies will be learned and what they will do. In this work, we present a theoretical framework for policy optimization that guarantees convergence to a particular optimal policy, via vanishing entropy regularization and a temperature decoupling gambit. Our approach realizes an interpretable, diversity-preserving optimal policy as the regularization temperature vanishes and ensures the convergence of policy derived objects--value functions and return distributions. In a particular instance of our method, for example, the realized policy samples all optimal actions uniformly. Leveraging our temperature decoupling gambit, we present an algorithm that estimates, to arbitrary accuracy, the return distribution associated to its interpretable, diversity-preserving optimal policy.

**arXiv ID:** 2510.08526
</details>

<details>
<summary><strong>Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning</strong> - Sanghyeon Lee, Sangjun Bae, Yisak Park, Seungyul Han - [[pdf]](https://arxiv.org/pdf/2502.03752)</summary>

**Abstract:** Meta-reinforcement learning (Meta-RL) facilitates rapid adaptation to unseen tasks but faces challenges in long-horizon environments. Skill-based approaches tackle this by decomposing state-action sequences into reusable skills and employing hierarchical decision-making. However, these methods are highly susceptible to noisy offline demonstrations, leading to unstable skill learning and degraded performance. To address this, we propose Self-Improving Skill Learning (SISL), which performs self-guided skill refinement using decoupled high-level and skill improvement policies, while applying skill prioritization via maximum return relabeling to focus updates on task-relevant trajectories, resulting in robust and stable adaptation even under noisy and suboptimal data. By mitigating the effect of noise, SISL achieves reliable skill learning and consistently outperforms other skill-based meta-RL methods on diverse long-horizon tasks.

**arXiv ID:** 2502.03752
</details>

<details>
<summary><strong>On The Sample Complexity Bounds In Bilevel Reinforcement Learning</strong> - Mudit Gaur, Utsav Singh, Amrit Singh Bedi, Raghu Pasupathu, Vaneet Aggarwal - [[pdf]](https://arxiv.org/pdf/2503.17644)</summary>

**Abstract:** Bilevel reinforcement learning (BRL) has emerged as a powerful framework for aligning generative models, yet its theoretical foundations, especially sample complexity bounds, remain underexplored. In this work, we present the first sample complexity bound for BRL, establishing a rate of $\mathcal{O}(\epsilon^{-3})$ in continuous state-action spaces. Traditional MDP analysis techniques do not extend to BRL due to its nested structure and non-convex lower-level problems. We overcome these challenges by leveraging the Polyak-Łojasiewicz (PL) condition and the MDP structure to obtain closed-form gradients, enabling tight sample complexity analysis. Our analysis also extends to general bi-level optimization settings with non-convex lower levels, where we achieve state-of-the-art sample complexity results of $\mathcal{O}(\epsilon^{-3})$ improving upon existing bounds of $\mathcal{O}(\epsilon^{-6})$. Additionally, we address the computational bottleneck of hypergradient estimation by proposing a fully first-order, Hessian-free algorithm suitable for large-scale problems.

**arXiv ID:** 2503.17644
</details>

<details>
<summary><strong>Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots</strong> - Boyu Li, Siyuan He, Hang Xu, Haoqi Yuan, Yu Zang, Liwei Hu, Junpeng Yue, Zhenxiong Jiang, Pengbo Hu, Börje F. Karlsson, Yehui Tang, Zongqing Lu - [[pdf]](https://arxiv.org/pdf/2510.07882)</summary>

**Abstract:** In recent years, Multimodal Large Language Models (MLLMs) have demonstrated the ability to serve as high-level planners, enabling robots to follow complex human instructions. However, their effectiveness, especially in long-horizon tasks involving dual-arm humanoid robots, remains limited. This limitation arises from two main challenges: (i) the absence of simulation platforms that systematically support task evaluation and data collection for humanoid robots, and (ii) the insufficient embodiment awareness of current MLLMs, which hinders reasoning about dual-arm selection logic and body positions during planning. To address these issues, we present DualTHOR, a new dual-arm humanoid simulator, with continuous transition and a contingency mechanism. Building on this platform, we propose Proprio-MLLM, a model that enhances embodiment awareness by incorporating proprioceptive information with motion-based position embedding and a cross-spatial encoder. Experiments show that, while existing MLLMs struggle in this environment, Proprio-MLLM achieves an average improvement of 19.75% in planning performance. Our work provides both an essential simulation platform and an effective model to advance embodied intelligence in humanoid robotics. The code is available at this https URL.

**arXiv ID:** 2510.07882
</details>

<details>
<summary><strong>Practicing a Second Language Without Fear: Mixed Reality Agents for Interactive Group Conversation</strong> - Mariana Fernandez-Espinosa, Kai Zhang, Jad Bendarkawi, Ashley Ponce, Sean Chidozie Mata, Aminah Aliu, Lei Zhang, Francisco Fernandez Medina, Elena Mangione-Lora, Andres Monroy-Hernandez, Diego Gomez-Zara - [[pdf]](https://arxiv.org/pdf/2510.08227)</summary>

**Abstract:** Developing speaking proficiency in a second language can be cognitively demanding and emotionally taxing, often triggering fear of making mistakes or being excluded from larger groups. While current learning tools show promise for speaking practice, most focus on dyadic, scripted scenarios, limiting opportunities for dynamic group interactions. To address this gap, we present ConversAR, a Mixed Reality system that leverages Generative AI and XR to support situated and personalized group conversations. It integrates embodied AI agents, scene recognition, and generative 3D props anchored to real-world surroundings. Based on a formative study with experts in language acquisition, we developed and tested this system with a user study with 21 second-language learners. Results indicate that the system enhanced learner engagement, increased willingness to communicate, and offered a safe space for speaking. We discuss the implications for integrating Generative AI and XR into the design of future language learning applications.

**arXiv ID:** 2510.08227
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
