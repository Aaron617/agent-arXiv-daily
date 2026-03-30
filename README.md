# Agent arXiv Daily

**Last Updated:** 2026-03-30 04:16:14

**Total Papers:** 53

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
<summary><strong>Out-of-Sight Embodied Agents: Multimodal Tracking, Sensor Fusion, and Trajectory Forecasting</strong> - Haichao Zhang, Yi Xu, Yun Fu - [[pdf]](https://arxiv.org/pdf/2509.15219)</summary>

**Abstract:** Trajectory prediction is a fundamental problem in computer vision, vision-language-action models, world models, and autonomous systems, with broad impact on autonomous driving, robotics, and surveillance. However, most existing methods assume complete and clean observations, and therefore do not adequately handle out-of-sight agents or noisy sensing signals caused by limited camera coverage, occlusions, and the absence of ground-truth denoised trajectories. These challenges raise safety concerns and reduce robustness in real-world deployment. In this extended study, we introduce major improvements to Out-of-Sight Trajectory (OST), a task for predicting noise-free visual trajectories of out-of-sight objects from noisy sensor observations. Building on our prior work, we expand Out-of-Sight Trajectory Prediction (OOSTraj) from pedestrians to both pedestrians and vehicles, increasing its relevance to autonomous driving, robotics, and surveillance. Our improved Vision-Positioning Denoising Module exploits camera calibration to establish vision-position correspondence, mitigating the lack of direct visual cues and enabling effective unsupervised denoising of noisy sensor signals. Extensive experiments on the Vi-Fi and JRDB datasets show that our method achieves state-of-the-art results for both trajectory denoising and trajectory prediction, with clear gains over prior baselines. We also compare with classical denoising methods, including Kalman filtering, and adapt recent trajectory prediction models to this setting, establishing a stronger benchmark. To the best of our knowledge, this is the first work to use vision-positioning projection to denoise noisy sensor trajectories of out-of-sight agents, opening new directions for future research.

**arXiv ID:** 2509.15219
</details>

<details>
<summary><strong>Advancing AI Trustworthiness Through Patient Simulation: Risk Assessment of Conversational Agents for Antidepressant Selection</strong> - Md Tanvir Rouf Shawon, Mohammad Sabik Irbaz, Hadeel R. A. Elyazori, Keerti Reddy Resapu, Yili Lin, Vladimir Franzuela Cardenas, Farrokh Alemi, Kevin Lybarger - [[pdf]](https://arxiv.org/pdf/2602.11391)</summary>

**Abstract:** Objective: This paper introduces a patient simulator for scalable, automated evaluation of healthcare conversational agents, generating realistic, controllable interactions that systematically vary across medical, linguistic, and behavioral dimensions to support risk assessment across populations. Methods: Grounded in the NIST AI Risk Management Framework, the simulator integrates three profile components: (1) medical profiles constructed from All of Us electronic health records using risk-ratio gating; (2) linguistic profiles modeling health literacy and condition-specific communication; and (3) behavioral profiles representing cooperative, distracted, and adversarial engagement. Profiles were evaluated against NIST AI RMF trustworthiness requirements and assessed against an AI Decision Aid for antidepressant selection. Results: Across 500 simulated conversations, the simulator revealed monotonic degradation in AI Decision Aid performance across health literacy levels: Rank-1 concept retrieval ranged from 47.6% (limited) to 81.9% (proficient), with corresponding recommendation degradation. Medical concept fidelity was high (96.6% across 8,210 concepts), validated by human annotators (0.73 kappa) and an LLM judge with comparable agreement (0.78 kappa). Behavioral profiles were reliably distinguished (0.93 kappa), and linguistic profiles showed moderate agreement (0.61 kappa). Conclusions: The simulator exposes measurable performance risks in conversational healthcare AI. Health literacy emerged as a primary risk factor with direct implications for equitable AI deployment.

**arXiv ID:** 2602.11391
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (12 papers)</h2></summary>

<details>
<summary><strong>BeSafe-Bench: Unveiling Behavioral Safety Risks of Situated Agents in Functional Environments</strong> - Yuxuan Li, Yi Lin, Peng Wang, Shiming Liu, Xuetao Wei - [[pdf]](https://arxiv.org/pdf/2603.25747)</summary>

**Abstract:** The rapid evolution of Large Multimodal Models (LMMs) has enabled agents to perform complex digital and physical tasks, yet their deployment as autonomous decision-makers introduces substantial unintentional behavioral safety risks. However, the absence of a comprehensive safety benchmark remains a major bottleneck, as existing evaluations rely on low-fidelity environments, simulated APIs, or narrowly scoped tasks. To address this gap, we present BeSafe-Bench (BSB), a benchmark for exposing behavioral safety risks of situated agents in functional environments, covering four representative domains: Web, Mobile, Embodied VLM, and Embodied VLA. Using functional environments, we construct a diverse instruction space by augmenting tasks with nine categories of safety-critical risks, and adopt a hybrid evaluation framework that combines rule-based checks with LLM-as-a-judge reasoning to assess real environmental impacts. Evaluating 13 popular agents reveals a concerning trend: even the best-performing agent completes fewer than 40% of tasks while fully adhering to safety constraints, and strong task performance frequently coincides with severe safety violations. These findings underscore the urgent need for improved safety alignment before deploying agentic systems in real-world settings.

**arXiv ID:** 2603.25747
</details>

<details>
<summary><strong>AIRA_2: Overcoming Bottlenecks in AI Research Agents</strong> - Karen Hambardzumyan, Nicolas Baldwin, Edan Toledo, Rishi Hazra, Michael Kuchnik, Bassel Al Omari, Thomas Simon Foster, Anton Protopopov, Jean-Christophe Gagnon-Audet, Ishita Mediratta, Kelvin Niu, Michael Shvartsman, Alisia Lupidi, Alexis Audran-Reiss, Parth Pathak, Tatiana Shavrina, Despoina Magka, Hela Momand, Derek Dunfield, Nicola Cancedda, Pontus Stenetorp, Carole-Jean Wu, Jakob Nicolaus Foerster, Yoram Bachrach, Martin Josifoski - [[pdf]](https://arxiv.org/pdf/2603.26499)</summary>

**Abstract:** Existing research has identified three structural performance bottlenecks in AI research agents: (1) synchronous single-GPU execution constrains sample throughput, limiting the benefit of search; (2) a generalization gap where validation-based selection causes performance to degrade over extended search horizons; and (3) the limited capability of fixed, single-turn LLM operators imposes a ceiling on search performance. We introduce AIRA$_2$, which addresses these bottlenecks through three architectural choices: an asynchronous multi-GPU worker pool that increases experiment throughput linearly; a Hidden Consistent Evaluation protocol that delivers a reliable evaluation signal; and ReAct agents that dynamically scope their actions and debug interactively. On MLE-bench-30, AIRA$_2$ achieves a mean Percentile Rank of 71.8% at 24 hours - surpassing the previous best of 69.9% - and steadily improves to 76.0% at 72 hours. Ablation studies reveal that each component is necessary and that the "overfitting" reported in prior work was driven by evaluation noise rather than true data memorization.

**arXiv ID:** 2603.26499
</details>

<details>
<summary><strong>Consistency Amplifies: How Behavioral Variance Shapes Agent Accuracy</strong> - Aman Mehta - [[pdf]](https://arxiv.org/pdf/2603.25764)</summary>

**Abstract:** As LLM-based agents are deployed in production systems, understanding their behavioral consistency (whether they produce similar action sequences when given identical tasks) becomes critical for reliability. We study consistency in the context of SWE-bench, a challenging software engineering benchmark requiring complex, multi-step reasoning. Comparing Claude~4.5~Sonnet, GPT-5, and Llama-3.1-70B across 50 runs each (10 tasks $\times$ 5 runs), we find that across models, higher consistency aligns with higher accuracy: Claude achieves the lowest variance (CV: 15.2\%) and highest accuracy (58\%), GPT-5 is intermediate (CV: 32.2\%, accuracy: 32\%), and Llama shows the highest variance (CV: 47.0\%) with lowest accuracy (4\%). However, within a model, consistency can amplify both correct and incorrect interpretations. Our analysis reveals a critical nuance: \textbf{consistency amplifies outcomes rather than guaranteeing correctness}. 71\% of Claude's failures stem from "consistent wrong interpretation": making the same incorrect assumption across all runs. Interestingly, GPT-5 achieves similar early strategic agreement as Claude (diverging at step 3.4 vs.\ 3.2) but exhibits 2.1$\times$ higher variance, suggesting that divergence timing alone does not determine consistency. These findings suggest that for production deployment, interpretation accuracy matters more than execution consistency, with implications for agent evaluation and training.

**arXiv ID:** 2603.25764
</details>

<details>
<summary><strong>MAGNET: Autonomous Expert Model Generation via Decentralized Autoresearch and BitNet Training</strong> - Yongwan Kim, Sungchul Park - [[pdf]](https://arxiv.org/pdf/2603.25813)</summary>

**Abstract:** We present MAGNET (Model Autonomously Growing Network), a decentralized system for autonomous generation, training, and serving of domain-expert language models across commodity hardware. MAGNET integrates four components: (1) autoresearch, an autonomous ML research pipeline that automates dataset generation, hyperparameter exploration, evaluation, and error-driven iteration; (2) BitNet b1.58 ternary training, enabling CPU-native inference via this http URL without GPU hardware; (3) DiLoCo-based distributed merging for communication-efficient aggregation of domain specialists; and (4) on-chain contribution tracking on the HOOTi EVM chain. We validate autoresearch through three case studies: video safety classification (balanced accuracy 0.9287 to 0.9851), cryptocurrency directional prediction (41% to 54.9% hit rate), and BitNet hyperparameter optimization (10-phase sweep, -16.7% validation loss).

**arXiv ID:** 2603.25813
</details>

<details>
<summary><strong>Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding</strong> - Shrinidhi Kumbhar, Haofu Liao, Srikar Appalaraju, Kunwar Yashraj Singh - [[pdf]](https://arxiv.org/pdf/2603.26211)</summary>

**Abstract:** Autoregressive (AR) vision-language models (VLMs) have long dominated multimodal understanding, reasoning, and graphical user interface (GUI) grounding. Recently, discrete diffusion vision-language models (DVLMs) have shown strong performance in multimodal reasoning, offering bidirectional attention, parallel token generation, and iterative refinement. However, their potential for GUI grounding remains unexplored. In this work, we evaluate whether discrete DVLMs can serve as a viable alternative to AR models for GUI grounding. We adapt LLaDA-V for single-turn action and bounding-box prediction, framing the task as text generation from multimodal input. To better capture the hierarchical structure of bounding-box geometry, we propose a hybrid masking schedule that combines linear and deterministic masking, improving grounding accuracy by up to 6.1 points in Step Success Rate (SSR) over the GUI-adapted LLaDA-V trained with linear masking. Evaluations on four datasets spanning web, desktop, and mobile interfaces show that the adapted diffusion model with hybrid masking consistently outperforms the linear-masked variant and performs competitively with autoregressive counterparts despite limited pretraining. Systematic ablations reveal that increasing diffusion steps, generation length, and block length improves accuracy but also increases latency, with accuracy plateauing beyond a certain number of diffusion steps. Expanding the training data with diverse GUI domains further reduces latency by about 1.3 seconds and improves grounding accuracy by an average of 20 points across benchmarks. These results demonstrate that discrete DVLMs are a promising modeling framework for GUI grounding and represent an important step toward diffusion-based GUI agents.

**arXiv ID:** 2603.26211
</details>

<details>
<summary><strong>Clawed and Dangerous: Can We Trust Open Agentic Systems?</strong> - Shiping Chen, Qin Wang, Guangsheng Yu, Xu Wang, Liming Zhu - [[pdf]](https://arxiv.org/pdf/2603.26221)</summary>

**Abstract:** Open agentic systems combine LLM-based planning with external capabilities, persistent memory, and privileged execution. They are used in coding assistants, browser copilots, and enterprise automation. OpenClaw is a visible instance of this broader class.
Without much attention yet, their security challenge is fundamentally different from that of traditional software that relies on predictable execution and well-defined control flow. In open agentic systems, everything is ''probabilistic'': plans are generated at runtime, key decisions may be shaped by untrusted natural-language inputs and tool outputs, execution unfolds in uncertain environments, and actions are taken under authority delegated by human users. The central challenge is therefore not merely robustness against individual attacks, but the governance of agentic behavior under persistent uncertainty.
This paper systematizes the area through a software engineering lens. We introduce a six-dimensional analytical taxonomy and synthesize 50 papers spanning attacks, benchmarks, defenses, audits, and adjacent engineering foundations. From this synthesis, we derive a reference doctrine for secure-by-construction agent platforms, together with an evaluation scorecard for assessing platform security posture. Our review shows that the literature is relatively mature in attack characterization and benchmark construction, but remains weak in deployment controls, operational governance, persistent-memory integrity, and capability revocation. These gaps define a concrete engineering agenda for building agent ecosystems that are governable, auditable, and resilient under compromise.

**arXiv ID:** 2603.26221
</details>

<details>
<summary><strong>Environment Maps: Structured Environmental Representations for Long-Horizon Agents</strong> - Yenchia Feng, Chirag Sharma, Karime Maamari - [[pdf]](https://arxiv.org/pdf/2603.23610)</summary>

**Abstract:** Although large language models (LLMs) have advanced rapidly, robust automation of complex software workflows remains an open problem. In long-horizon settings, agents frequently suffer from cascading errors and environmental stochasticity; a single misstep in a dynamic interface can lead to task failure, resulting in hallucinations or trial-and-error. This paper introduces $\textit{Environment Maps}$: a persistent, agent-agnostic representation that mitigates these failures by consolidating heterogeneous evidence, such as screen recordings and execution traces, into a structured graph. The representation consists of four core components: (1) Contexts (abstracted locations), (2) Actions (parameterized affordances), (3) Workflows (observed trajectories), and (4) Tacit Knowledge (domain definitions and reusable procedures). We evaluate this framework on the WebArena benchmark across five domains. Agents equipped with environment maps achieve a 28.2% success rate, nearly doubling the performance of baselines limited to session-bound context (14.2%) and outperforming agents that have access to the raw trajectory data used to generate the environment maps (23.3%). By providing a structured interface between the model and the environment, Environment Maps establish a persistent foundation for long-horizon planning that is human-interpretable, editable, and incrementally refinable.

**arXiv ID:** 2603.23610
</details>

<details>
<summary><strong>INSIGHT: Enhancing Autonomous Driving Safety through Vision-Language Models on Context-Aware Hazard Detection and Edge Case Evaluation</strong> - Dianwei Chen, Zifan Zhang, Lei Cheng, Yuchen Liu, Xianfeng Terry Yang - [[pdf]](https://arxiv.org/pdf/2502.00262)</summary>

**Abstract:** Autonomous driving systems face significant challenges in handling unpredictable edge-case scenarios, such as adversarial pedestrian movements, dangerous vehicle maneuvers, and sudden environmental changes. Current end-to-end driving models struggle with generalization to these rare events due to limitations in traditional detection and prediction approaches. To address this, we propose INSIGHT (Integration of Semantic and Visual Inputs for Generalized Hazard Tracking), a hierarchical vision-language model (VLM) framework designed to enhance hazard detection and edge-case evaluation. By using multimodal data fusion, our approach integrates semantic and visual representations, enabling precise interpretation of driving scenarios and accurate forecasting of potential dangers. Through supervised fine-tuning of VLMs, we optimize spatial hazard localization using attention-based mechanisms and coordinate regression techniques. Experimental results on the BDD100K dataset demonstrate a substantial improvement in hazard prediction straightforwardness and accuracy over existing models, achieving a notable increase in generalization performance. This advancement enhances the robustness and safety of autonomous driving systems, ensuring improved situational awareness and potential decision-making in complex real-world scenarios.

**arXiv ID:** 2502.00262
</details>

<details>
<summary><strong>WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning</strong> - Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, Sung Ju Hwang - [[pdf]](https://arxiv.org/pdf/2512.02425)</summary>

**Abstract:** Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average 8.4% performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning.

**arXiv ID:** 2512.02425
</details>

<details>
<summary><strong>AgentPack: A Dataset of Code Changes, Co-Authored by Agents and Humans</strong> - Yangtian Zi, Zixuan Wu, Aleksander Boruch-Gruszecki, Jonathan Bell, Arjun Guha - [[pdf]](https://arxiv.org/pdf/2509.21891)</summary>

**Abstract:** Fine-tuning large language models for code editing has typically relied on mining commits and pull requests. The working hypothesis has been that commit messages describe human intent in natural language, and patches to code describe the changes that implement that intent. However, much of the previously collected data is noisy: commit messages are terse, human-written commits commingle several unrelated edits, and many commits come from simple, rule-based bots.
The recent adoption of software engineering agents changes this landscape. Code changes \emph{co-authored} by humans and agents are often accompanied by substantially more explicit natural-language descriptions of intent and rationale. Moreover, when these changes land in public repositories, they are implicitly filtered by humans: maintainers discard low-quality commits to their projects.
We present AgentPack, a corpus of 1.8M code edits co-authored by Claude Code, OpenAI Codex, and Cursor Agent across public GitHub projects up to early October 2025. We describe the identification and curation pipeline, quantify adoption trends of these agents, and analyze the structural properties of the edits. Finally, we show that models fine-tuned on AgentPack can outperform models trained on prior human-only commit corpora, highlighting the potential of using public data from software engineering agents to train future code-editing models.

**arXiv ID:** 2509.21891
</details>

<details>
<summary><strong>A Judge Agent Closes the Reliability Gap in AI-Generated Scientific Simulation</strong> - Chengshuai Yang - [[pdf]](https://arxiv.org/pdf/2603.25780)</summary>

**Abstract:** Large language models can generate scientific simulation code, but the generated code silently fails on most non-textbook problems. We show that classical mathematical validation -- well-posedness, convergence, and error certification -- can be fully automated by a Judge Agent, reducing the silent-failure rate from 42% to 1.5% across 134 test cases spanning 12 scientific domains. The headline result comes from a prospective benchmark: 72 blinded tasks submitted by 12 independent scientists yield an 89% success rate (95% CI: [80%, 95%]) with automated error bounds, versus 53% without the Judge. On clinical CT (the only powered experiment, n = 200), the pipeline reaches 99% of expert quality. The residual 1.5% concentrates at bifurcation points where certifiability breaks down. We formalize this boundary through the simulability class S and introduce this http URL, a structured specification format that makes any scientific computation problem machine-readable and solver-independent. Code, data, and all 72 benchmark tasks are publicly archived.

**arXiv ID:** 2603.25780
</details>

<details>
<summary><strong>Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI</strong> - Xinhao Liu, Jiaqi Li, Youming Deng, Ruxin Chen, Yingjia Zhang, Yifei Ma, Li Guo, Yiming Li, Jing Zhang, Chen Feng - [[pdf]](https://arxiv.org/pdf/2511.20620)</summary>

**Abstract:** Reproducible closed-loop evaluation remains a major bottleneck in Embodied AI such as visual navigation. A promising path forward is high-fidelity simulation that combines photorealistic sensor rendering with geometrically grounded interaction in complex, open-world urban environments. Although recent video-3DGS methods ease open-world scene capturing, they are still unsuitable for benchmarking due to large visual and geometric sim-to-real gaps. To address these challenges, we introduce Wanderland, a real-to-sim framework that features multi-sensor capture, reliable reconstruction, accurate geometry, and robust view synthesis. Using this pipeline, we curate a diverse dataset of indoor-outdoor urban scenes and systematically demonstrate how image-only pipelines scale poorly, how geometry quality impacts novel view synthesis, and how all of these adversely affect navigation policy learning and evaluation reliability. Beyond serving as a trusted testbed for embodied navigation, Wanderland's rich raw sensor data further allows benchmarking of 3D reconstruction and novel view synthesis models. Our work establishes a new foundation for reproducible research in open-world embodied AI. Project website is at this https URL.

**arXiv ID:** 2511.20620
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>ProbGuard: Probabilistic Runtime Monitoring for LLM Agent Safety</strong> - Haoyu Wang, Christopher M. Poskitt, Jiali Wei, Jun Sun - [[pdf]](https://arxiv.org/pdf/2508.00500)</summary>

**Abstract:** Large Language Model (LLM) agents increasingly operate across domains such as robotics, virtual assistants, and web automation. However, their stochastic decision-making introduces safety risks that are difficult to anticipate during execution. Existing runtime monitoring frameworks, such as AgentSpec, primarily rely on reactive safety rules that detect violations only when unsafe behavior is imminent or has already occurred, limiting their ability to handle long-horizon dependencies. We present ProbGuard, a proactive runtime monitoring framework for LLM agents that anticipates safety violations through probabilistic risk prediction. ProbGuard abstracts agent executions into symbolic states and learns a Discrete-Time Markov Chain (DTMC) from execution traces to model behavioral dynamics. At runtime, the monitor estimates the probability that future executions will reach unsafe states and triggers interventions when this risk exceeds a user-defined threshold. To improve robustness, ProbGuard incorporates semantic validity constraints in the abstraction and provides PAC-style guarantees on the learned model under standard assumptions. We evaluate ProbGuard in two safety-critical domains: autonomous driving and embodied household agents. Across evaluated scenarios, ProbGuard consistently predicts traffic law violations and collisions in advance, with warnings up to 38.66 seconds ahead of occurrence. In embodied agent tasks, ProbGuard reduces unsafe behavior by up to 65.37% while preserving up to 80.4% task completion. ProbGuard is implemented as an extensible open-source runtime monitor integrated with the LangChain agent framework and introduces minimal runtime overhead.

**arXiv ID:** 2508.00500
</details>

<details>
<summary><strong>The Dual-State Architecture for Reliable LLM Agents</strong> - Matthew Thompson - [[pdf]](https://arxiv.org/pdf/2512.20660)</summary>

**Abstract:** Large Language Models deployed as code generation agents exhibit stochastic behavior incompatible with the deterministic guarantees required by software engineering. We formalize the Dual-State Action Pair (DSAP), an execution primitive that couples stochastic generation with deterministic post-condition verification. Guard functions act as sensing actions that project opaque LLM outputs onto observable workflow state, enabling a dual-state decomposition: finite, deterministic S_workflow paired with infinite, stochastic S_env. We prove that for epsilon-capable generators, failure probability P(fail) <= (1-epsilon)^R_max -> 0. To prevent naive O(R^K) retry explosion across multi-step workflows, we introduce a three-level recovery hierarchy: context refinement (retry within step), informed backtracking (stagnation detection with cascade invalidation and context injection to upstream steps), and human escalation. Experimental validation across 13 LLMs (1.3B-15B parameters) on three diagnostic probes demonstrates reliability gains of up to 66 percentage points at 1.2-2.1x baseline cost. Recovery mechanism evaluation on 99 SWE-Bench Pro instance-arm pairs (Qwen3-Coder-Next) demonstrates 100% context injection effectiveness (upstream output changed in all 71 escalation events) with step-specific recovery asymmetry -- 37.5% for test generation vs. 0% for patch generation -- and 0% end-to-end patch production, establishing the boundary between execution architecture and plan synthesis: execution recovery is necessary but not sufficient for autonomous software engineering.

**arXiv ID:** 2512.20660
</details>

<details>
<summary><strong>MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain Personalization</strong> - Weizhi Zhang, Xiaokai Wei, Wei-Chieh Huang, Zheng Hui, Chen Wang, Michelle Gong, Philip S. Yu - [[pdf]](https://arxiv.org/pdf/2603.25973)</summary>

**Abstract:** Recent advancements in Large Language Models (LLMs) have expanded context windows to million-token scales, yet benchmarks for evaluating memory remain limited to short-session synthetic dialogues. We introduce \textsc{MemoryCD}, the first large-scale, user-centric, cross-domain memory benchmark derived from lifelong real-world behaviors in the Amazon Review dataset. Unlike existing memory datasets that rely on scripted personas to generate synthetic user data, \textsc{MemoryCD} tracks authentic user interactions across years and multiple domains. We construct a multi-faceted long-context memory evaluation pipeline of 14 state-of-the-art LLM base models with 6 memory method baselines on 4 distinct personalization tasks over 12 diverse domains to evaluate an agent's ability to simulate real user behaviors in both single and cross-domain settings. Our analysis reveals that existing memory methods are far from user satisfaction in various domains, offering the first testbed for cross-domain life-long personalization evaluation.

**arXiv ID:** 2603.25973
</details>

<details>
<summary><strong>AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents</strong> - Wenbo Gao, Renxi Liu, Xian Wang, Fang Guo, Shuai Yang, Xi Chen, Hui-Ling Zhen, Hanting Chen, Weizhe Lin, Xiaosong Li, Yaoyuan Wang - [[pdf]](https://arxiv.org/pdf/2603.26034)</summary>

**Abstract:** Autonomous agents powered by large language models (LLMs) perform complex tasks through long-horizon reasoning and tool interaction, where a fundamental trade-off arises between execution efficiency and reasoning robustness. Models at different capability-cost levels offer complementary advantages: lower-cost models enable fast execution but may struggle on difficult reasoning segments, while stronger models provide more robust reasoning at higher computational cost. We present AgentCollab, a self-driven collaborative inference framework that dynamically coordinates models with different reasoning capacities during agent execution. Instead of relying on external routing modules, the framework uses the agent's own self-reflection signal to determine whether the current reasoning trajectory is making meaningful progress, and escalates control to a stronger reasoning tier only when necessary. To further stabilize long-horizon execution, we introduce a difficulty-aware cumulative escalation strategy that allocates additional reasoning budget based on recent failure signals. In our experiments, we instantiate this framework using a two-level small-large model setting. Experiments on diverse multi-step agent benchmarks show that AgentCollab consistently improves the accuracy-efficiency Pareto frontier of LLM agents.

**arXiv ID:** 2603.26034
</details>

<details>
<summary><strong>Can AI Scientist Agents Learn from Lab-in-the-Loop Feedback? Evidence from Iterative Perturbation Discovery</strong> - Gilles Wainrib, Barbara Bodinier, Haitem Dakhli, Josep Monserrat, Almudena Espin Perez, Sabrina Carpentier, Roberta Codato, John Klein - [[pdf]](https://arxiv.org/pdf/2603.26177)</summary>

**Abstract:** Recent work has questioned whether large language models (LLMs) can perform genuine in-context learning (ICL) for scientific experimental design, with prior studies suggesting that LLM-based agents exhibit no sensitivity to experimental feedback. We shed new light on this question by carrying out 800 independently replicated experiments on iterative perturbation discovery in Cell Painting high-content screening. We compare an LLM agent that iteratively updates its hypotheses using experimental feedback to a zero-shot baseline that relies solely on pretraining knowledge retrieval. Access to feedback yields a $+53.4\%$ increase in discoveries per feature on average ($p = 0.003$). To test whether this improvement arises from genuine feedback-driven learning rather than prompt-induced recall of pretraining knowledge, we introduce a random feedback control in which hit/miss labels are permuted. Under this control, the performance gain disappears, indicating that the observed improvement depends on the structure of the feedback signal ($+13.0$ hits, $p = 0.003$). We further examine how model capability affects feedback utilization. Upgrading from Claude Sonnet 4.5 to 4.6 reduces gene hallucination rates from ${\sim}33\%$--$45\%$ to ${\sim}3$--$9\%$, converting a non-significant ICL effect ($+0.8$, $p = 0.32$) into a large and highly significant improvement ($+11.0$, $p=0.003$) for the best ICL strategy. These results suggest that effective in-context learning from experimental feedback emerges only once models reach a sufficient capability threshold.

**arXiv ID:** 2603.26177
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (13 papers)</h2></summary>

<details>
<summary><strong>GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation</strong> - Rui Xie, Zhi Gao, Chenrui Shi, Zirui Shang, Lu Chen, Qing Li - [[pdf]](https://arxiv.org/pdf/2603.26266)</summary>

**Abstract:** Large vision-language models have endowed GUI agents with strong general capabilities for interface understanding and interaction. However, due to insufficient exposure to domain-specific software operation data during training, these agents exhibit significant domain bias - they lack familiarity with the specific operation workflows (planning) and UI element layouts (grounding) of particular applications, limiting their real-world task performance. In this paper, we present GUIDE (GUI Unbiasing via Instructional-Video Driven Expertise), a training-free, plug-and-play framework that resolves GUI agent domain bias by autonomously acquiring domain-specific expertise from web tutorial videos through a retrieval-augmented automated annotation pipeline. GUIDE introduces two key innovations. First, a subtitle-driven Video-RAG pipeline unlocks video semantics through subtitle analysis, performing progressive three-stage retrieval - domain classification, topic extraction, and relevance matching - to identify task-relevant tutorial videos. Second, a fully automated annotation pipeline built on an inverse dynamics paradigm feeds consecutive keyframes enhanced with UI element detection into VLMs, inferring the required planning and grounding knowledge that are injected into the agent's corresponding modules to address both manifestations of domain bias. Extensive experiments on OSWorld demonstrate GUIDE's generality as a plug-and-play component for both multi-agent systems and single-model agents. It consistently yields over 5% improvements and reduces execution steps - without modifying any model parameters or architecture - validating GUIDE as an architecture-agnostic enhancement to bridge GUI agent domain bias.

**arXiv ID:** 2603.26266
</details>

<details>
<summary><strong>CADSmith: Multi-Agent CAD Generation with Programmatic Geometric Validation</strong> - Jesse Barkley, Rumi Loghmani, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2603.26512)</summary>

**Abstract:** Existing methods for text-to-CAD generation either operate in a single pass with no geometric verification or rely on lossy visual feedback that cannot resolve dimensional errors. We present CADSmith, a multi-agent pipeline that generates CadQuery code from natural language. It then undergoes an iterative refinement process through two nested correction loops: an inner loop that resolves execution errors and an outer loop grounded in programmatic geometric validation. The outer loop combines exact measurements from the OpenCASCADE kernel (bounding box dimensions, volume, solid validity) with holistic visual assessment from an independent vision-language model Judge. This provides both the numerical precision and the high-level shape awareness needed to converge on the correct geometry. The system uses retrieval-augmented generation over API documentation rather than fine-tuning, maintaining a current database as the underlying CAD library evolves. We evaluate on a custom benchmark of 100 prompts in three difficulty tiers (T1 through T3) with three ablation configurations. Against a zero-shot baseline, CADSmith achieves a 100% execution rate (up from 95%), improves the median F1 score from 0.9707 to 0.9846, the median IoU from 0.8085 to 0.9629, and reduces the mean Chamfer Distance from 28.37 to 0.74, demonstrating that closed-loop refinement with programmatic geometric feedback substantially improves the quality and reliability of LLM-generated CAD models.

**arXiv ID:** 2603.26512
</details>

<details>
<summary><strong>SkinGPT-X: A Self-Evolving Collaborative Multi-Agent System for Transparent and Trustworthy Dermatological Diagnosis</strong> - Zhangtianyi Chen, Yuhao Shen, Florensia Widjaja, Yan Xu, Liyuan Sun, Zijian Wang, Hongyi Chen, Wufei Dai, Juexiao Zhou - [[pdf]](https://arxiv.org/pdf/2603.26122)</summary>

**Abstract:** While recent advancements in Large Language Models have significantly advanced dermatological diagnosis, monolithic LLMs frequently struggle with fine-grained, large-scale multi-class diagnostic tasks and rare skin disease diagnosis owing to training data sparsity, while also lacking the interpretability and traceability essential for clinical reasoning. Although multi-agent systems can offer more transparent and explainable diagnostics, existing frameworks are primarily concentrated on Visual Question Answering and conversational tasks, and their heavy reliance on static knowledge bases restricts adaptability in complex real-world clinical settings. Here, we present SkinGPT-X, a multimodal collaborative multi-agent system for dermatological diagnosis integrated with a self-evolving dermatological memory mechanism. By simulating the diagnostic workflow of dermatologists and enabling continuous memory evolution, SkinGPT-X delivers transparent and trustworthy diagnostics for the management of complex and rare dermatological cases. To validate the robustness of SkinGPT-X, we design a three-tier comparative experiment. First, we benchmark SkinGPT-X against four state-of-the-art LLMs across four public datasets, demonstrating its state-of-the-art performance with a +9.6% accuracy improvement on DDI31 and +13% weighted F1 gain on Dermnet over the state-of-the-art model. Second, we construct a large-scale multi-class dataset covering 498 distinct dermatological categories to evaluate its fine-grained classification capabilities. Finally, we curate the rare skin disease dataset, the first benchmark to address the scarcity of clinical rare skin diseases which contains 564 clinical samples with eight rare dermatological diseases. On this dataset, SkinGPT-X achieves a +9.8% accuracy improvement, a +7.1% weighted F1 improvement, a +10% Cohen's Kappa improvement.

**arXiv ID:** 2603.26122
</details>

<details>
<summary><strong>Knowdit: Agentic Smart Contract Vulnerability Detection with Auditing Knowledge Summarization</strong> - Ziqiao Kong, Wanxu Xia, Chong Wang, Yi Lu, Pan Li, Shaohua Li, Zong Cao, Yang Liu - [[pdf]](https://arxiv.org/pdf/2603.26270)</summary>

**Abstract:** Smart contracts govern billions of dollars in decentralized finance (DeFi), yet automated vulnerability detection remains challenging because many vulnerabilities are tightly coupled with project-specific business logic. We observe that recurring vulnerabilities across diverse DeFi business models often share the same underlying economic mechanisms, which we term DeFi semantics, and that capturing these shared abstractions can enable more systematic auditing. Building on this insight, we propose Knowdit, a knowledge-driven, agentic framework for smart contract vulnerability detection. Knowdit first constructs an auditing knowledge graph from historical human audit reports, linking fine-grained DeFi semantics with recurring vulnerability patterns. Given a new project, a multi-agent framework leverages this knowledge through an iterative loop of specification generation, harness synthesis, fuzz execution, and finding reflection, driven by a shared working memory for continuous refinement.
We evaluate Knowdit on 12 recent Code4rena projects with 75 ground-truth vulnerabilities. Knowdit detects all 14 high-severity and 77\% of medium-severity vulnerabilities with only 2 false positives, significantly outperforming all baselines. Applied to six real-world projects, Knowdit further discovers 12 high- and 10 medium-severity previously unknown vulnerabilities, proving its outstanding performance.

**arXiv ID:** 2603.26270
</details>

<details>
<summary><strong>Governance-Aware Vector Subscriptions for Multi-Agent Knowledge Ecosystems</strong> - Steven Johnson - [[pdf]](https://arxiv.org/pdf/2603.20833)</summary>

**Abstract:** As AI agent ecosystems grow, agents need mechanisms to monitor relevant knowledge in real time. Semantic publish-subscribe systems address this by matching new content against vector subscriptions. However, in multi-agent settings where agents operate under different data handling policies, unrestricted semantic subscriptions create policy violations: agents receive notifications about content they are not authorized to access. We introduce governance-aware vector subscriptions, a mechanism that composes semantic similarity matching with multi-dimensional policy predicates grounded in regulatory frameworks (EU DSM Directive, EU AI Act). The policy predicate operates over multiple independent dimensions (processing level, direct marketing restrictions, training opt-out, jurisdiction, and scientific usage) each with distinct legal bases. Agents subscribe to semantic regions of a curated knowledge base; notifications are dispatched only for validated content that passes both the similarity threshold and all applicable policy constraints. We formalize the mechanism, implement it within AIngram (an operational multi-agent knowledge base), and evaluate it using the PASA benchmark. We validate the mechanism on a synthetic corpus (1,000 chunks, 93 subscriptions, 5 domains): the governed mode correctly enforces all policy constraints while preserving delivery of authorized content. Ablation across five policy dimensions shows that no single dimension suffices for full compliance.

**arXiv ID:** 2603.20833
</details>

<details>
<summary><strong>Evidence-based diagnostic reasoning with multi-agent copilot for human pathology</strong> - Luca L. Weishaupt, Chengkuan Chen, Drew F. K. Williamson, Richard J. Chen, Guillaume Jaume, Tong Ding, Bowen Chen, Anurag Vaidya, Long Phi Le, Guillaume Jaume, Ming Y. Lu, Faisal Mahmood - [[pdf]](https://arxiv.org/pdf/2506.20964)</summary>

**Abstract:** Pathology is experiencing rapid digital transformation driven by whole-slide imaging and artificial intelligence (AI). While deep learning-based computational pathology has achieved notable success, traditional models primarily focus on image analysis without integrating natural language instruction or rich, text-based context. Current multimodal large language models (MLLMs) in computational pathology face limitations, including insufficient training data, inadequate support and evaluation for multi-image understanding, and a lack of autonomous, diagnostic reasoning capabilities. To address these limitations, we introduce PathChat+, a new MLLM specifically designed for human pathology, trained on over 1 million diverse, pathology-specific instruction samples and nearly 5.5 million question answer turns. Extensive evaluations across diverse pathology benchmarks demonstrated that PathChat+ substantially outperforms the prior PathChat copilot, as well as both state-of-the-art (SOTA) general-purpose and other pathology-specific models. Furthermore, we present SlideSeek, a reasoning-enabled multi-agent AI system leveraging PathChat+ to autonomously evaluate gigapixel whole-slide images (WSIs) through iterative, hierarchical diagnostic reasoning, reaching high accuracy on DDxBench, a challenging open-ended differential diagnosis benchmark, while also capable of generating visually grounded, humanly-interpretable summary reports.

**arXiv ID:** 2506.20964
</details>

<details>
<summary><strong>AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems</strong> - Zhaohui Geoffrey Wang - [[pdf]](https://arxiv.org/pdf/2603.14688)</summary>

**Abstract:** As multi-agent AI systems are increasingly deployed in real-world settings - from automated customer support to DevOps remediation - failures become harder to diagnose due to cascading effects, hidden dependencies, and long execution traces. We present AgentTrace, a lightweight causal tracing framework for post-hoc failure diagnosis in deployed multi-agent workflows. AgentTrace reconstructs causal graphs from execution logs, traces backward from error manifestations, and ranks candidate root causes using interpretable structural and positional signals - without requiring LLM inference at debugging time. Across a diverse benchmark of multi-agent failure scenarios designed to reflect common deployment patterns, AgentTrace localizes root causes with high accuracy and sub-second latency, significantly outperforming both heuristic and LLM-based baselines. Our results suggest that causal tracing provides a practical foundation for improving the reliability and trustworthiness of agentic systems in the wild.

**arXiv ID:** 2603.14688
</details>

<details>
<summary><strong>Deception and Communication in Autonomous Multi-Agent Systems: An Experimental Study with Among Us</strong> - Maria Milkowski, Tim Weninger - [[pdf]](https://arxiv.org/pdf/2603.26635)</summary>

**Abstract:** As large language models are deployed as autonomous agents, their capacity for strategic deception raises core questions for coordination, reliability, and safety in multi-goal, multi-agent systems. We study deception and communication in L2LM agents through the social deduction game Among Us, a cooperative-competitive environment. Across 1,100 games, autonomous agents produced over one million tokens of meeting dialogue. Using speech act theory and interpersonal deception theory, we find that all agents rely mainly on directive language, while impostor agents shift slightly toward representative acts such as explanations and denials. Deception appears primarily as equivocation rather than outright lies, increasing under social pressure but rarely improving win rates. Our contributions are a large-scale analysis of role-conditioned deceptive behavior in LLM agents and empirical evidence that current agents favor low-risk ambiguity that is linguistically subtle yet strategically limited, revealing a fundamental tension between truthfulness and utility in autonomous communication.

**arXiv ID:** 2603.26635
</details>

<details>
<summary><strong>Can a Robot Walk the Robotic Dog: Triple-Zero Collaborative Navigation for Heterogeneous Multi-Agent Systems</strong> - Yaxuan Wang, Yifan Xiang, Ke Li, Xun Zhang, BoWen Ye, Zhuochen Fan, Fei Wei, Tong Yang - [[pdf]](https://arxiv.org/pdf/2603.21723)</summary>

**Abstract:** We present Triple Zero Path Planning (TZPP), a collaborative framework for heterogeneous multi-robot systems that requires zero training, zero prior knowledge, and zero simulation. TZPP employs a coordinator--explorer architecture: a humanoid robot handles task coordination, while a quadruped robot explores and identifies feasible paths using guidance from a multimodal large language model. We implement TZPP on Unitree G1 and Go2 robots and evaluate it across diverse indoor and outdoor environments, including obstacle-rich and landmark-sparse settings. Experiments show that TZPP achieves robust, human-comparable efficiency and strong adaptability to unseen scenarios. By eliminating reliance on training and simulation, TZPP offers a practical path toward real-world deployment of heterogeneous robot cooperation. Our code and video are provided at: this https URL

**arXiv ID:** 2603.21723
</details>

<details>
<summary><strong>ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory</strong> - Zhuohan Ge, Haoyang Li, Yubo Wang, Nicole Hu, Chen Jason Zhang, Qing Li - [[pdf]](https://arxiv.org/pdf/2603.26182)</summary>

**Abstract:** While Large Language Models (LLMs) have demonstrated potential in healthcare, they often struggle with the complex, non-linear reasoning required for accurate clinical diagnosis. Existing methods typically rely on static, linear mappings from symptoms to diagnoses, failing to capture the iterative, hypothesis-driven reasoning inherent to human clinicians. To bridge this gap, we introduce ClinicalAgents, a novel multi-agent framework designed to simulate the cognitive workflow of expert clinicians. Unlike rigid sequential chains, ClinicalAgents employs a dynamic orchestration mechanism modeled as a Monte Carlo Tree Search (MCTS) process. This allows an Orchestrator to iteratively generate hypotheses, actively verify evidence, and trigger backtracking when critical information is missing. Central to this framework is a Dual-Memory architecture: a mutable Working Memory that maintains the evolving patient state for context-aware reasoning, and a static Experience Memory that retrieves clinical guidelines and historical cases via an active feedback loop. Extensive experiments demonstrate that ClinicalAgents achieves state-of-the-art performance, significantly enhancing both diagnostic accuracy and explainability compared to strong single-agent and multi-agent baselines.

**arXiv ID:** 2603.26182
</details>

<details>
<summary><strong>Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents</strong> - Nicholas Edwards, Sebastian Schuster - [[pdf]](https://arxiv.org/pdf/2603.26233)</summary>

**Abstract:** As Large Language Model (LLM) agents are increasingly deployed in open-ended domains like software engineering, they frequently encounter underspecified instructions that lack crucial context. While human developers naturally resolve underspecification by asking clarifying questions, current agents are largely optimized for autonomous execution. In this work, we systematically evaluate the clarification-seeking abilities of LLM agents on an underspecified variant of SWE-bench Verified. We propose an uncertainty-aware multi-agent scaffold that explicitly decouples underspecification detection from code execution. Our results demonstrate that this multi-agent system using OpenHands + Claude Sonnet 4.5 achieves a 69.40% task resolve rate, significantly outperforming a standard single-agent setup (61.20%) and closing the performance gap with agents operating on fully specified instructions. Furthermore, we find that the multi-agent system exhibits well-calibrated uncertainty, conserving queries on simple tasks while proactively seeking information on more complex issues. These findings indicate that current models can be turned into proactive collaborators, where agents independently recognize when to ask questions to elicit missing information in real-world, underspecified tasks.

**arXiv ID:** 2603.26233
</details>

<details>
<summary><strong>Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation</strong> - Fabian Konstantinidis, Moritz Sackmann, Ulrich Hofmann, Christoph Stiller - [[pdf]](https://arxiv.org/pdf/2512.05812)</summary>

**Abstract:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.

**arXiv ID:** 2512.05812
</details>

<details>
<summary><strong>Automating UI Optimization through Multi-Agentic Reasoning</strong> - Zhipeng Li, Christoph Gebhardt, Yi-Chi Liao, Christian Holz - [[pdf]](https://arxiv.org/pdf/2602.13126)</summary>

**Abstract:** We present AutoOptimization, a novel multi-objective optimization framework for adapting user interfaces. From a user's verbal preferences for changing a UI, our framework guides a prioritization-based Pareto frontier search over candidate layouts. It selects suitable objective functions for UI placement while simultaneously parameterizing them according to the user's instructions to define the optimization problem. A solver then generates a series of optimal UI layouts, which our framework validates against the user's instructions to adapt the UI with the final solution. Our approach thus overcomes the previous need for manual inspection of layouts and the use of population averages for objective parameters. We integrate multiple agents sequentially within our framework, enabling the system to leverage their reasoning capabilities to interpret user preferences, configure the optimization problem, and validate optimization outcomes.

**arXiv ID:** 2602.13126
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>A Lightweight, Transferable, and Self-Adaptive Framework for Intelligent DC Arc-Fault Detection in Photovoltaic Systems</strong> - Xiaoke Yang, Long Gao, Haoyu He, Hanyuan Hang, Qi Liu, Shuai Zhao, Qiantu Tuo, Rui Li - [[pdf]](https://arxiv.org/pdf/2603.25749)</summary>

**Abstract:** Arc-fault circuit interrupters (AFCIs) are essential for mitigating fire hazards in residential photovoltaic (PV) systems, yet achieving reliable DC arc-fault detection under real-world conditions remains challenging. Spectral interference from inverter switching, hardware heterogeneity, operating-condition drift, and environmental noise collectively compromise conventional AFCI solutions. This paper proposes a lightweight, transferable, and self-adaptive learning-driven framework (LD-framework) for intelligent DC arc-fault detection. At the device level, LD-Spec learns compact spectral representations enabling efficient on-device inference and near-perfect arc discrimination. Across heterogeneous inverter platforms, LD-Align performs cross-hardware representation alignment to ensure robust detection despite hardware-induced distribution shifts. To address long-term evolution, LD-Adapt introduces a cloud-edge collaborative self-adaptive updating mechanism that detects unseen operating regimes and performs controlled model evolution. Extensive experiments involving over 53,000 labeled samples demonstrate near-perfect detection, achieving 0.9999 accuracy and 0.9996 F1-score. Across diverse nuisance-trip-prone conditions, including inverter start-up, grid transitions, load switching, and harmonic disturbances, the method achieves a 0% false-trip rate. Cross-hardware transfer shows reliable adaptation using only 0.5%-1% labeled target data while preserving source performance. Field adaptation experiments demonstrate recovery of detection precision from 21% to 95% under previously unseen conditions. These results indicate that the LD-framework enables a scalable, deployment-oriented AFCI solution maintaining highly reliable detection across heterogeneous devices and long-term operation.

**arXiv ID:** 2603.25749
</details>

<details>
<summary><strong>UCAgent: An End-to-End Agent for Block-Level Functional Verification</strong> - Junyue Wang, Zhicheng Yao, Yan Pi, Xiaolong Li, Fangyuan Song, Jinru Wang, Yunlong Xie, Sa Wang, Yungang Bao - [[pdf]](https://arxiv.org/pdf/2603.25768)</summary>

**Abstract:** Functional verification remains a critical bottleneck in modern IC development cycles, accounting for approximately 70% of total development time in many projects. However, traditional methods, including constrained-random and formal verification, struggle to keep pace with the growing complexity of modern semiconductor designs.
While recent advances in Large Language Models (LLMs) have shown promise in code generation and task automation, significant challenges hinder the realization of end-to-end functional verification automation. These challenges include (i) limited accuracy in generating Verilog/SystemVerilog verification code, (ii) the fragility of LLMs when executing complex, multi-step verification workflows, and (iii) the difficulty of maintaining verification consistency across specifications, coverage models, and test cases throughout the workflow.
To address these challenges, we propose UCAgent, an end-to-end agent that automates hardware block-level functional verification based on three core mechanisms. First, we establish a pure Python verification environment using Picker and Toffee to avoid relying on LLM-generated SystemVerilog verification code. Second, we introduce a configurable 31-stage fine-grained verification workflow to guide the LLM, where each stage is verified by an automated checker. Furthermore, we propose a Verification Consistency Labeling Mechanism (VCLM) that assigns hierarchical labels to LLM-generated artifacts, improving the reliability and traceability of verification.
Experimental results show that UCAgent can complete end-to-end automated verification on multiple modules, including the UART, FPU, and integer divider modules, achieving up to 98.5% code coverage and up to 100% functional coverage. UCAgent also discovers previously unidentified design defects in realistic designs, demonstrating its practical potential.

**arXiv ID:** 2603.25768
</details>

<details>
<summary><strong>Channelling, Coordinating, Collaborating: A Three-Layer Framework for Disability-Centered Human-Agent Collaboration</strong> - Lan Xiao, Catherine Holloway - [[pdf]](https://arxiv.org/pdf/2603.26252)</summary>

**Abstract:** AI accessibility tools have mostly been designed for individual use, helping one person overcome a specific functional barrier. But for many people with disabilities, complex tasks are accomplished through collaboration with others who bring complementary abilities, not solitary effort. We propose a three-layer framework, Channelling, Coordinating, and Co-Creating, that rethinks AI's role in ability-diverse collaboration: establishing shared informational ground across abilities, mediating workflows between collaborators with different abilities, and contributing as a bounded partner toward shared goals. Grounded in the Ability-Diverse Collaboration framework, grounding theory, and Carlile's 3T framework, it extends the ``agents as remote collaborators'' vision by centring the collaborative, interdependent ways people with disabilities already work.

**arXiv ID:** 2603.26252
</details>

<details>
<summary><strong>KALAVAI: Predicting When Independent Specialist Fusion Works -- A Quantitative Model for Post-Hoc Cooperative LLM Training</strong> - Ramchand Kumaresan - [[pdf]](https://arxiv.org/pdf/2603.22755)</summary>

**Abstract:** Independently trained domain specialists can be fused post-hoc into a single model that outperforms any individual specialist, and the gain is predictable: gain = 0.82 x divergence - 2.72 (R^2 = 0.856, n=6, 3-26% divergence). This enables practitioners to estimate cooperative value before committing compute. Below ~3.3% divergence, gains approach this http URL the KALAVAI protocol, contributors fine-tune copies of a shared checkpoint independently, then submit for lightweight MoE routing (500 steps). Gains are consistent: +7.72% at 410M (+/-0.02%, 3 seeds), +7.49% at 1B (+/-0.01%, 3 seeds), +6.53% at 6.9B, each over the best specialist. The router matches domain-oracle routing within <10^{-5} nats. Cross-lingual fusion (Tamil/Yoruba/Welsh/Code) achieves +21.76%, with Yoruba perplexity falling 41.9 to 7.7. A 20-contributor federation achieves +16.71% (+/-0.07pp, 3 seeds).Three requirements bound the protocol. Shared initialisation is necessary: checkpoint mismatch degrades routing. Frozen layers are optional below ~10,000 steps and beneficial beyond. Learned routing is essential: uniform averaging degrades by -1.2% vs. best specialist, while any trained router achieves oracle-optimal assignment.

**arXiv ID:** 2603.22755
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (17 papers)</h2></summary>

<details>
<summary><strong>AutoB2G: A Large Language Model-Driven Agentic Framework For Automated Building-Grid Co-Simulation</strong> - Borui Zhang, Nariman Mahdavi, Subbu Sethuvenkatraman, Shuang Ao, Flora Salim - [[pdf]](https://arxiv.org/pdf/2603.26005)</summary>

**Abstract:** The growing availability of building operational data motivates the use of reinforcement learning (RL), which can learn control policies directly from data and cope with the complexity and uncertainty of large-scale building clusters. However, most existing simulation environments prioritize building-side performance metrics and lack systematic evaluation of grid-level impacts, while their experimental workflows still rely heavily on manual configuration and substantial programming expertise. Therefore, this paper proposes AutoB2G, an automated building-grid co-simulation framework that completes the entire simulation workflow solely based on natural-language task descriptions. The framework extends CityLearn V2 to support Building-to-Grid (B2G) interaction and adopts the large language model (LLM)-based SOCIA (Simulation Orchestration for Computational Intelligence with Agents) framework to automatically generate, execute, and iteratively refine the simulator. As LLMs lack prior knowledge of the implementation context of simulation functions, a codebase covering simulation configurations and functional modules is constructed and organized as a directed acyclic graph (DAG) to explicitly represent module dependencies and execution order, guiding the LLM to retrieve a complete executable path. Experimental results demonstrate that AutoB2G can effectively enable automated simulator implementations, coordinating B2G interactions to improve grid-side performance metrics.

**arXiv ID:** 2603.26005
</details>

<details>
<summary><strong>Empowering Epidemic Response: The Role of Reinforcement Learning in Infectious Disease Control</strong> - Mutong Liu, Yang Liu, Jiming Liu - [[pdf]](https://arxiv.org/pdf/2603.25771)</summary>

**Abstract:** Reinforcement learning (RL), owing to its adaptability to various dynamic systems in many real-world scenarios and the capability of maximizing long-term outcomes under different constraints, has been used in infectious disease control to optimize the intervention strategies for controlling infectious disease spread and responding to outbreaks in recent years. The potential of RL for assisting public health sectors in preventing and controlling infectious diseases is gradually emerging and being explored by rapidly increasing publications relevant to COVID-19 and other infectious diseases. However, few surveys exclusively discuss this topic, that is, the development and application of RL approaches for optimizing strategies of non-pharmaceutical and pharmaceutical interventions of public health. Therefore, this paper aims to provide a concise review and discussion of the latest literature on how RL approaches have been used to assist in controlling the spread and outbreaks of infectious diseases, covering several critical topics addressing public health demands: resource allocation, balancing between lives and livelihoods, mixed policy of multiple interventions, and inter-regional coordinated control. Finally, we conclude the paper with a discussion of several potential directions for future research.

**arXiv ID:** 2603.25771
</details>

<details>
<summary><strong>Doctorina MedBench: End-to-End Evaluation of Agent-Based Medical AI</strong> - Anna Kozlova, Stanislau Salavei, Pavel Satalkin, Hanna Plotnitskaya, Sergey Parfenyuk - [[pdf]](https://arxiv.org/pdf/2603.25821)</summary>

**Abstract:** We present Doctorina MedBench, a comprehensive evaluation framework for agent-based medical AI based on the simulation of realistic physician-patient interactions. Unlike traditional medical benchmarks that rely on solving standardized test questions, the proposed approach models a multi-step clinical dialogue in which either a physician or an AI system must collect medical history, analyze attached materials (including laboratory reports, images, and medical documents), formulate differential diagnoses, and provide personalized recommendations. System performance is evaluated using the D.O.T.S. metric, which consists of four components: Diagnosis, Observations/Investigations, Treatment, and Step Count, enabling assessment of both clinical correctness and dialogue efficiency.
The system also incorporates a multi-level testing and quality monitoring architecture designed to detect model degradation during both development and deployment. The framework supports safety-oriented trap cases, category-based random sampling of clinical scenarios, and full regression testing. The dataset currently contains more than 1,000 clinical cases covering over 750 diagnoses. The universality of the evaluation metrics allows the framework to be used not only to assess medical AI systems, but also to evaluate physicians and support the development of clinical reasoning skills. Our results suggest that simulation of clinical dialogue may provide a more realistic assessment of clinical competence compared to traditional examination-style benchmarks.

**arXiv ID:** 2603.25821
</details>

<details>
<summary><strong>JAL-Turn: Joint Acoustic-Linguistic Modeling for Real-Time and Robust Turn-Taking Detection in Full-Duplex Spoken Dialogue Systems</strong> - Guangzhao Yang, Yu Pan, Shi Qiu, Ningjie Bai - [[pdf]](https://arxiv.org/pdf/2603.26515)</summary>

**Abstract:** Despite recent advances, efficient and robust turn-taking detection remains a significant challenge in industrial-grade Voice AI agent deployments. Many existing systems rely solely on acoustic or semantic cues, leading to suboptimal accuracy and stability, while recent attempts to endow large language models with full-duplex capabilities require costly full-duplex data and incur substantial training and deployment overheads, limiting real-time performance. In this paper, we propose JAL-Turn, a lightweight and efficient speech-only turn-taking framework that adopts a joint acoustic-linguistic modeling paradigm, in which a cross-attention module adaptively integrates pre-trained acoustic representations with linguistic features to support low-latency prediction of hold vs shift states. By sharing a frozen ASR encoder, JAL-Turn enables turn-taking prediction to run fully in parallel with speech recognition, introducing no additional end-to-end latency or computational overhead. In addition, we introduce a scalable data construction pipeline that automatically derives reliable turn-taking labels from large-scale real-world dialogue corpora. Extensive experiments on public multilingual benchmarks and an in-house Japanese customer-service dataset show that JAL-Turn consistently outperforms strong state-of-the-art baselines in detection accuracy while maintaining superior real-time performance.

**arXiv ID:** 2603.26515
</details>

<details>
<summary><strong>Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification</strong> - Zehai He, Wenyi Hong, Zhen Yang, Ziyang Pan, Mingdao Liu, Xiaotao Gu, Jie Tang - [[pdf]](https://arxiv.org/pdf/2603.26648)</summary>

**Abstract:** Recent advances in large language models have improved the capabilities of coding agents, yet systematic evaluation of complex, end-to-end website development remains limited. To address this gap, we introduce Vision2Web, a hierarchical benchmark for visual website development, spanning from static UI-to-code generation, interactive multi-page frontend reproduction, to long-horizon full-stack website development. The benchmark is constructed from real-world websites and comprises a total of 193 tasks across 16 categories, with 918 prototype images and 1,255 test cases. To support flexible, thorough and reliable evaluation, we propose workflow-based agent verification paradigm based on two complementary components: a GUI agent verifier and a VLM-based judge. We evaluate multiple visual language models instantiated under different coding-agent frameworks, revealing substantial performance gaps at all task levels, with state-of-the-art models still struggling on full-stack development.

**arXiv ID:** 2603.26648
</details>

<details>
<summary><strong>HeaRT: A Hierarchical Circuit Reasoning Tree-Based Agentic Framework for AMS Design Optimization</strong> - Souradip Poddar, Chia-Tung Ho, Ziming Wei, Weidong Cao, Haoxing Ren, David Z. Pan - [[pdf]](https://arxiv.org/pdf/2511.19669)</summary>

**Abstract:** Conventional AI-driven AMS design automation algorithms remain constrained by their reliance on high-quality datasets to capture underlying circuit behavior, coupled with poor transferability across architectures, and a lack of adaptive mechanisms. This work proposes HeaRT, a hierarchical circuit reasoning-based agentic framework for automation loops and a step toward adaptive, human-style design optimization. HeaRT consistently improves F1(subcircuits) by >= 13.5% and F1(loops) by >= 37.8% over few-shot prompting baselines across multiple LLM backbones on our 40-circuit AMS benchmark of flattened SPICE netlists, even as circuit complexity increases. Our experiments further show that HeaRT achieves >= 3x faster convergence in incremental design adaptation tasks under specification shifts across diverse optimization approaches, supporting both topology reconfiguration and sizing.

**arXiv ID:** 2511.19669
</details>

<details>
<summary><strong>AtomMem : Learnable Dynamic Agentic Memory with Atomic Memory Operation</strong> - Yupeng Huo, Yaxi Lu, Zhong Zhang, Haotian Chen, Yankai Lin - [[pdf]](https://arxiv.org/pdf/2601.08323)</summary>

**Abstract:** Equipping agents with memory is essential for solving real-world long-horizon problems. However, most existing agent memory mechanisms rely on static and hand-crafted workflows. This limits the performance and generalization ability of these memory designs, which highlights the need for a more flexible, learning-based memory framework. In this paper, we propose AtomMem, which reframes memory management as a dynamic decision-making problem. We deconstruct high-level memory processes into fundamental atomic CRUD (Create, Read, Update, Delete) operations, transforming the memory workflow into a learnable decision process. By combining supervised fine-tuning with reinforcement learning, AtomMem learns an autonomous, task-aligned policy to orchestrate memory behaviors tailored to specific task demands. Experimental results across 3 long-context benchmarks demonstrate that the trained AtomMem-8B consistently outperforms prior static-workflow memory methods. Further analysis of training dynamics shows that our learning-based formulation enables the agent to discover structured, task-aligned memory management strategies, highlighting a key advantage over predefined routines.

**arXiv ID:** 2601.08323
</details>

<details>
<summary><strong>PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning</strong> - Ruheng Wang, Hang Zhang, Trieu Nguyen, Shasha Feng, Hao-Wei Pang, Xiang Yu, Li Xiao, Peter Zhiping Zhang - [[pdf]](https://arxiv.org/pdf/2508.14765)</summary>

**Abstract:** Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery.

**arXiv ID:** 2508.14765
</details>

<details>
<summary><strong>Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models</strong> - Boxin Wang, Chankyu Lee, Nayeon Lee, Sheng-Chieh Lin, Wenliang Dai, Yang Chen, Yangyi Chen, Zhuolin Yang, Zihan Liu, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping - [[pdf]](https://arxiv.org/pdf/2512.13607)</summary>

**Abstract:** Building general-purpose reasoning models with reinforcement learning (RL) entails substantial cross-domain heterogeneity, including large variation in inference-time response lengths and verification latency. Such variability complicates the RL infrastructure, slows training, and makes training curriculum (e.g., response length extension) and hyperparameter selection challenging. In this work, we propose cascaded domain-wise reinforcement learning (Cascade RL) to develop Nemotron-Cascade, capable of operating in both instruct and deep thinking modes, without any performance gap relative to a thinking-only counterpart. Departing from conventional approaches that blend heterogeneous prompts from different domains, Cascade RL orchestrates sequential, domain-wise RL, reducing engineering complexity and delivering state-of-the-art performance across a wide range of benchmarks. Notably, RLHF for alignment, when used as a pre-step, boosts the model's reasoning ability far beyond mere preference optimization, and subsequent domain-wise RLVR stages rarely degrade the benchmark performance attained in earlier domains and may even improve it (see an illustration in Figure 1). Our 14B model, after RL, outperforms its SFT teacher, DeepSeek-R1-0528, on LiveCodeBench v5/v6/Pro and achieves silver-medal performance in the 2025 International Olympiad in Informatics (IOI). We transparently share our training and data recipes.

**arXiv ID:** 2512.13607
</details>

<details>
<summary><strong>MRG-R1: Reinforcement Learning for Clinically Aligned Medical Report Generation</strong> - Pengyu Wang, Shuchang Ye, Usman Naseem, Jinman Kim - [[pdf]](https://arxiv.org/pdf/2512.16145)</summary>

**Abstract:** Medical report generation aims to automatically produce radiology-style reports from medical images, supporting efficient and accurate clinical this http URL, existing approaches predominately rely on token-level likelihood training, which favors local lexical matching and leaves clinical correctness under-specified in the training objective. This behavior can be attributed to token-level likelihood optimization, which rewards surface-form agreement and therefore fails to directly encode constraints on medically accurate findings. To address this objective mismatch, we introduce a semantic-driven reinforcement learning (SRL) framework for medical report generation, named MRG-R1, which directly optimizes report-level clinical correctness rather than token-level likelihood. The key module is a clinically grounded report-level reward function, which reinforces semantic agreement in clinically relevant findings between generated and reference reports, thereby enabling learning signals that explicitly constrain medical correctness beyond surface linguistic alignment. Our evaluations show that the proposed framework improves the accuracy and coverage of clinically relevant findings in generated reports, and that MRG-R1 achieves state-of-the-art clinical efficacy on the IU X-Ray and MIMIC-CXR benchmark datasets.

**arXiv ID:** 2512.16145
</details>

<details>
<summary><strong>KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning</strong> - Shuai Wang, Yinan Yu - [[pdf]](https://arxiv.org/pdf/2603.21440)</summary>

**Abstract:** Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: this https URL.

**arXiv ID:** 2603.21440
</details>

<details>
<summary><strong>EVA: Efficient Reinforcement Learning for End-to-End Video Agent</strong> - Yaolun Zhang, Ruohui Wang, Jiahao Wang, Yepeng Tang, Xuanyu Zheng, Haonan Duan, Hao Lu, Hanming Deng, Lewei Lu - [[pdf]](https://arxiv.org/pdf/2603.22918)</summary>

**Abstract:** Video understanding with multimodal large language models (MLLMs) remains challenging due to the long token sequences of videos, which contain extensive temporal dependencies and redundant frames. Existing approaches typically treat MLLMs as passive recognizers, processing entire videos or uniformly sampled frames without adaptive reasoning. Recent agent-based methods introduce external tools, yet still depend on manually designed workflows and perception-first strategies, resulting in inefficiency on long videos. We present EVA, an Efficient Reinforcement Learning framework for End-to-End Video Agent, which enables planning-before-perception through iterative summary-plan-action-reflection reasoning. EVA autonomously decides what to watch, when to watch, and how to watch, achieving query-driven and efficient video understanding. To train such agents, we design a simple yet effective three-stage learning pipeline - comprising supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Group Relative Policy Optimization (GRPO) - that bridges supervised imitation and reinforcement learning. We further construct high-quality datasets for each stage, supporting stable and reproducible training. We evaluate EVA on six video understanding benchmarks, demonstrating its comprehensive capabilities. Compared with existing baselines, EVA achieves a substantial improvement of 6-12% over general MLLM baselines and a further 1-3% gain over prior adaptive agent methods.

**arXiv ID:** 2603.22918
</details>

<details>
<summary><strong>T$^\star$: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory Aware Reinforcement Learning</strong> - Hanchen Xia, Baoyou Chen, Yutang Ge, Guojiang Zhao, Siyu Zhu - [[pdf]](https://arxiv.org/pdf/2601.11214)</summary>

**Abstract:** We present T$^\star$, a simple TraceRL-based training curriculum for progressive block-size scaling in masked diffusion language models (MDMs). Starting from an AR-initialized small-block MDM, T$^\star$ transitions smoothly to larger blocks, enabling higher-parallelism decoding with minimal performance degradation on math reasoning benchmarks. Moreover, further analysis suggests that T$^\star$ may actually converge to an alternative decoding schedule that achieves comparable performance.

**arXiv ID:** 2601.11214
</details>

<details>
<summary><strong>Knowledge Distillation for Efficient Transformer-Based Reinforcement Learning in Hardware-Constrained Energy Management Systems</strong> - Pascal Henrich, Jonas Sievers, Maximilian Beichter, Thomas Blank, Ralf Mikut, Veit Hagenmeyer - [[pdf]](https://arxiv.org/pdf/2603.26249)</summary>

**Abstract:** Transformer-based reinforcement learning has emerged as a strong candidate for sequential control in residential energy management. In particular, the Decision Transformer can learn effective battery dispatch policies from historical data, thereby increasing photovoltaic self-consumption and reducing electricity costs. However, transformer models are typically too computationally demanding for deployment on resource-constrained residential controllers, where memory and latency constraints are critical. This paper investigates knowledge distillation to transfer the decision-making behaviour of high-capacity Decision Transformer policies to compact models that are more suitable for embedded deployment. Using the Ausgrid dataset, we train teacher models in an offline sequence-based Decision Transformer framework on heterogeneous multi-building data. We then distil smaller student models by matching the teachers' actions, thereby preserving control quality while reducing model size. Across a broad set of teacher-student configurations, distillation largely preserves control performance and even yields small improvements of up to 1%, while reducing the parameter count by up to 96%, the inference memory by up to 90%, and the inference time by up to 63%. Beyond these compression effects, comparable cost improvements are also observed when distilling into a student model of identical architectural capacity. Overall, our results show that knowledge distillation makes Decision Transformer control more applicable for residential energy management on resource-limited hardware.

**arXiv ID:** 2603.26249
</details>

<details>
<summary><strong>Topology-Aware Graph Reinforcement Learning for Energy Storage Systems Optimal Dispatch in Distribution Networks</strong> - Shuyi Gao, Stavros Orfanoudakis, Shengren Hou, Peter Palensky, Pedro P. Vergara - [[pdf]](https://arxiv.org/pdf/2603.26264)</summary>

**Abstract:** Optimal dispatch of energy storage systems (ESSs) in distribution networks involves jointly improving operating economy and voltage security under time-varying conditions and possible topology changes. To support fast online decision making, we develop a topology-aware Reinforcement Learning architecture based on Twin Delayed Deep Deterministic Policy Gradient (TD3), which integrates graph neural networks (GNNs) as graph feature encoders for ESS dispatch. We conduct a systematic investigation of three GNN variants: graph convolutional networks (GCNs), topology adaptive graph convolutional networks (TAGConv), and graph attention networks (GATs) on the 34-bus and 69-bus systems, and evaluate robustness under multiple topology reconfiguration cases as well as cross-system transfer between networks with different system sizes. Results show that GNN-based controllers consistently reduce the number and magnitude of voltage violations, with clearer benefits on the 69-bus system and under reconfiguration; on the 69-bus system, TD3-GCN and TD3-TAGConv also achieve lower saved cost relative to the NLP benchmark than the NN baseline. We also highlight that transfer gains are case-dependent, and zero-shot transfer between fundamentally different systems results in notable performance degradation and increased voltage magnitude violations. This work is available at: this https URL and this https URL.

**arXiv ID:** 2603.26264
</details>

<details>
<summary><strong>Massive Parallel Deep Reinforcement Learning for Active SLAM</strong> - Martín Arce Llobera, Julio A. Placed, Mariano De Paula, Pablo De Cristóforis - [[pdf]](https://arxiv.org/pdf/2603.25834)</summary>

**Abstract:** Recent advances in parallel computing and GPU acceleration have created new opportunities for computation-intensive learning problems such as Active SLAM -- where actions are selected to reduce uncertainty and improve joint mapping and localization. However, existing DRL-based approaches remain constrained by the lack of scalable parallel training. In this work, we address this challenge by proposing a scalable end-to-end DRL framework for Active SLAM that enables massively parallel training. Compared with the state of the art, our method significantly reduces training time, supports continuous action spaces and facilitates the exploration of more realistic scenarios. It is released as an open-source framework to promote reproducibility and community adoption.

**arXiv ID:** 2603.25834
</details>

<details>
<summary><strong>Meta-Adaptive Beam Search Planning for Transformer-Based Reinforcement Learning Control of UAVs with Overhead Manipulators under Flight Disturbances</strong> - Hazim Alzorgan, Sayed Pedram Haeri Boroujeni, Abolfazl Razi - [[pdf]](https://arxiv.org/pdf/2603.26612)</summary>

**Abstract:** Drones equipped with overhead manipulators offer unique capabilities for inspection, maintenance, and contact-based interaction. However, the motion of the drone and its manipulator is tightly linked, and even small attitude changes caused by wind or control imperfections shift the end-effector away from its intended path. This coupling makes reliable tracking difficult and also limits the direct use of learning-based arm controllers that were originally designed for fixed-base robots. These effects appear consistently in our tests whenever the UAV body experiences drift or rapid attitude corrections. To address this behavior, we develop a reinforcement-learning (RL) framework with a transformer-based double deep Q learning (DDQN), with the core idea of using an adaptive beam-search planner that applies a short-horizon beam search over candidate control sequences using the learned critic as the forward estimator. This allows the controller to anticipate the end-effector's motion through simulated rollouts rather than executing those actions directly on the actual model, realizing a software-in-the-loop (SITL) approach. The lookahead relies on value estimates from a Transformer critic that processes short sequences of states, while a DDQN backbone provides the one-step targets needed to keep the learning process stable. Evaluated on a 3-DoF aerial manipulator under identical training conditions, the proposed meta-adaptive planner shows the strongest overall performance with a 10.2% reward increase, a substantial reduction in mean tracking error (from about 6% to 3%), and a 29.6% improvement in the combined reward-error metric relative to the DDQN baseline. Our method exhibits elevated stability in tracking target tip trajectory (by maintaining 5 cm tracking error) when the drone base exhibits drifts due to external disturbances, as opposed to the fixed-beam and Transformer-only variants.

**arXiv ID:** 2603.26612
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
