<div  align="center">
    <h1>CANON: Conditional Advantage Estimation for Reinforcement Learning in Large Reasoning Models</h1>

  <span style="color:red">📢 <strong><i>If you also engaged in the research of LRM or RL, we welcome your suggestions. And feel free to create an issue, when you have any questions about the code.
  If you are interested in our work, please star ⭐ our repository, Thx 💕.</i></strong></span>

  <h4>
    <img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=biuboomc.CANON" />
    <img src="https://img.shields.io/github/stars/biuboomc/CANON?style=flat-square&logo=github" alt="Stars">
    <img src="https://img.shields.io/github/issues/biuboomc/CANON?color=red" alt="Issues">
  </h4>
  
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2509.23962) [![Github](https://img.shields.io/badge/CANON-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/biuboomc/CANON)

</div>




---

# 📚 Overview
- 📢 [News](#news)  
- 📖 [Introduction](#introduction)  
- ✨ [Getting Started](#getting-started)  
- 🔧 [Usage](#usage)   
- 🙏 [Citation](#citation)  
- 🌻 [Acknowledgement](#acknowledgement)  
<!-- - 📈 [Star History](#star-history) -->


<div align="center">
  <hr width="100%">
</div>

# 📢News

- **[2025/04/20]** CANON paper available on [arXiv](https://arxiv.org/pdf/2509.23962). 

<!-- - **[2025/04/20]** The models and datasets are released on [HuggingFace](https://huggingface.co/collections/Elliott/luffy-rl-6804e1f5d1ebe66ba8ac92f4).
- **[2025/04/20]** LUFFY codebase is released along with evaluation scripts. Try it out! -->

---
# 📖Introduction

![intro](media/intro.svg)

<p>We introduce Conditional advANtage estimatiON (CANON), which amplifies the impact of specific metric changes by regrouping the sampled responses into two groups based on the values of a given metric. Rather than comparing against the mean value of all responses like DR.GRPO, CANON selects the direction of metric change that offers greater contributions to performance through inter-group comparison and favors responses that exhibit better performance within groups following the same trend in its intra-group comparison. DR.GRPO can be expressed as the average of CANON’s two advantage estimates and is therefore a special case of CANON.</p>

<div align="center">
  <hr width="100%">
</div>


# ✨ Getting Started

To setup the environment, run;
```
git clone https://github.com/biuboomc/CANON.git
pip install -e .
```
---
# 🔧 Usage

For the Llama series model, we construct a dataset with 35k queries. For the Qwen series model, we utilize the data with 47k queries released in [Huggingface](https://huggingface.co/datasets/Elliott/Openr1-Math-46k-8192).

Our code is based on [VeRL](https://github.com/volcengine/verl), and you can utilize CANON with different adv_estimators:


| **adv_estimator**                        | **Method**                                                   |
| ---------------------------------------- | ------------------------------------------------------------ |
| dr_entropy_token_budget                  | ***CANON*** based on *per-token* *genration* *entropy*       |
| dr_length_on_mean                        | ***CANON*** based on response length                         |
| dr_entropy_token_budget_annel            | ***CANON*** based on *Entropy* with First-Inter-Later-Intra and First-Intra-Later-Inter |
| dr_entropy_token_budget_cosine_restart   | ***CANON*** based on Cosin-First-Intra-Later-Inter           |
| dr_entropy_token_budget_cosine_restart_r | ***CANON*** based on Cosin-First-Inter-Later-Intra           |
| dr_random                                | ***CANON*** *based on* random regrouping                     |

And we introduce two hypeparameters for ***CANON***:

| **hyperparameter** | **Description** |
| ------------------ | --------------- |
| **alpha**     | alpha in Eq. 8 |
| **_lambda**      | mu in Eq. 5 |

---

## 🙏 Citation

If you find this work useful, please consider citing:

```bibtex
TBD
```

## 🌻 Acknowledgements

The codes are based on [VeRL](https://github.com/volcengine/verl). Sincere thanks to their wonderful works.
