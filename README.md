<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** https://github.com/othneildrew/Best-README-Template/blob/main/BLANK_README.md
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://kyotoai.org">
    <img src="images/SEIMEI_overlook.png" alt="Logo" width="640" height="360">
  </a>

<h3 align="center">RMSearch</h3>

  <p align="center">
    RMSearch
  </p>
  <p align="center">
    RMSearch is a high intellectual search tool using reward model instead of semantic embedding model.
    <br />
    <a href="https://github.com/kyotoai/SEIMEI/tree/main/demo"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kyotoai/SEIMEI/tree/main/demo">View Demo</a>
    ·
    <a href="https://github.com/kyotoai/SEIMEI/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/kyotoai/SEIMEI/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

### Search The Best Agent To Make Deep Reasoning
<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
<br />
<div align="center">
  <img src="images/SEIMEI_example.png" alt="seimei" width="640" height="360">
</div>

<br />
Here's the example of how SEIMEI works. Each agent interacts with LLM and document and makes inference. These inferences are automatically integrated by search engine and gives an answer of question.

<br />
<div align="center">
  <img src="images/SEIMEI_train_example.png" alt="seimei" width="640" height="360">
</div>

<br />
By training search engine, we can optimize the thinking steps like o1 or deepseek-r1!!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### The Most Intelligent Search Engine
<div align="center">
  <img src="images/Comparison.png" alt="seimei" width="400" height="360">
</div>

Our proprietary search model performs better than semantic embedding model(so called vector search). The graph above shows the result of training our model (3b) and e5-mistral-7b model to search best agents. While the vector search model cannot really retrieve best agents(because problems and agents do not have similar sentences), our proprietary search model can learn what agents are needed to solve a question and retrive the best ones!!

<a href="https://github.com/kyotoai/SEIMEI/tree/main/demo"><strong>See more details »</strong></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Achives State Of The Art Result

<div align="center">
  <img src="images/Improvement.png" alt="seimei" width="500" height="360">
</div>

We acheved an improvement of bigcodebench/deepseek-r1 by our search engine!!

<a href="https://github.com/kyotoai/SEIMEI/tree/main/demo"><strong>See more details »</strong></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Applications of SEIMEI

<div align="center">
  <img src="images/Application1.png" alt="seimei" width="640" height="360">
</div>

<div align="center">
  <img src="images/Application2.png" alt="seimei" width="640" height="360">
</div>

<div align="center">
  <img src="images/Application3.png" alt="seimei" width="640" height="360">
</div>

SEIMEI can be applied to make these useful functions!!

<a href="https://github.com/kyotoai/SEIMEI/tree/main/demo"><strong>See more details »</strong></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![vLLM][vllm.ai]][vllm-url]
* [![Hugging Face][huggingface.co]][huggingface-url]
  
<!--
* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]
-->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you build SEIMEI on local gpu or rental server gpu.
You can use it by installing seimei using `pip install` or downloading this directory into your local folder.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* transformers
  ```sh
  pip install transformers
  ```
* sentence_transformers
  ```sh
  pip install sentence_transformers
  ```
* vLLM
  ```sh
  pip install vllm
  pip install ray
  pip install packaging
  pip install typing
  ```

### Installation

* by `pip install`
  
1. Install seimei (not prepared yet)
   ```sh
   pip install seimei
   ```


* by downloading SEIMEI repository
  
1. Download the repo
   ```sh
   git clone https://github.com/kyotoai/SEIMEI.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

We are still developing this library. Please wait for it completing soon!

### Quick Start

1. Prepare chunks of text
    ```py
    from Prepare import Prepare

    # Locate your database folder in data_path below
    data_path = "(your database path)"
    save_path = "./processed"
    
    # designate all the files with 'extensions' inside 'folder_path'
    file_info = [
        {"folder_path":"(relative path for database you want to investigate)", "extensions":[".py"]},
    ]
    
    
    # about where the key starts to split the text
    # index : words to be where text should be split
    # first element(0 to 1): process_text_size * element is the start point of the key splitting. the samller the element is, the more likely it is for the key to split the text.
    # second element(0 or 1): the first element should become   if 0: <text1><key> | <text2>,  if 1: <text1> | <key><text2>
    rules = [
        {
            "class " : 1,
        },
    
        {
            "def " : 1,
        },
    
        {
            "if " : 1,
        },
    
        {
            "else " : 1,
            "elif " : 1,
        },
        
    
        {
            "\n\n" : 0,
            "<0x0A><0x0A>" : 0,
            "\x0A\x0A" : 0,
        },
    
        {
            "\n" : 0,
            "<0x0A>" : 0,
            "\x0A" : 0,
        },
    ]
    
    prepare = Prepare(
        database_path = data_path,
        save_path = save_path,
        rules = rules, 
        file_info=file_info, 
        model_name = "gpt2",
        max_tokens = 10000,
        min_tokens = 3000,
    )

    prepare.make_chunks()
    ```
    
2. Define seimei
    ```py
    from SEIMEI import SEIMEI
    import asyncio
    
    processed_path = "./processed"  # input path same as save_path you used in Preparation
    expert_class_names = ["Answer", "CheckInf", "MetaSurvey2"]
    se_restrictions = ["MetaSurvey2"]  # search engine only hits classes in this list normally (except when adding expert_restriction in kwargs)
    expert_module_names = ["Experts.Code.Modify"]
    
    seimei = SEIMEI(
        processed_path = processed_path,
        expert_class_names = expert_class_names,
        expert_module_names = expert_module_names,
        se_restrictions = se_restrictions,
        max_inference_time = 1000,
        tensor_parallel_size = 1,
    )
    ```
    
3. Get answer by seimei
    ```py
    original_question = "Give me the whole structure of this code file?"
    final_answer = await seimei.get_answer(query = original_question) # return final answer
    
    print()
    print()
    print(final_answer)
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Search Integration Mechanism for Experts
  - [ ] Permanent Expert
- [ ] Auto Log System in Jupyter Notebook
- [ ] High Precision Code & Textbook RAG

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<!--<a href="https://github.com/github_username/repo_name/graphs/contributors">-->
<a href="https://github.com/kyotoai/SEIMEI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyotoai/SEIMEI" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the Apache-2.0 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT --><!-- [@twitter_handle](https://twitter.com/twitter_handle) -->
## Contact

* Kentaro Seki - seki.kentaro@kyotoai.org

KyotoAI homepage: [https://kyotoai.org](https://kyotoai.org)

Project Link: [https://github.com/kyotoai/SEIMEI](https://github.com/kyotoai/SEIMEI)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [vLLM](https://docs.vllm.ai/en/latest/)
* [Huggingface](https://huggingface.co)
* [Kyoto University Library](https://www.kulib.kyoto-u.ac.jp/mainlib/en/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[vllm.ai]: https://img.shields.io/badge/vLLM-blue
[vllm-url]: https://docs.vllm.ai/en/latest/
[huggingface.co]: https://img.shields.io/badge/huggingface-yellow
[huggingface-url]: https://huggingface.co
[contributors-shield]: https://img.shields.io/github/contributors/kyotoai/SEIMEI.svg?style=for-the-badge
[contributors-url]: https://github.com/kyotoai/SEIMEI/graphs/contributors
[license-shield]: https://img.shields.io/github/license/kyotoai/SEIMEI.svg?style=for-the-badge
[license-url]: https://github.com/kyotoai/SEIMEI/LICENSE.txt

[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
