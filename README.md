# gender-trial-paper
Repository with custom code used in the manuscript: Scientists' gender drives different citation outcomes in randomized web experiments

A static version of the data used for the pilot analysis and experiment setup can be downloaded at:  

[doi/10.5281/zenodo.10728225](https://zenodo.org/doi/10.5281/zenodo.10728225)


**Before running each file, please adjust the `<path>` variable according to local directory structure.**

## Package requirements. Anaconda python installation is highly recommended

```
openai textstat jsonlines json tqdm tenacity metaphone jellyfish pandas numpy statsmodels scipy seaborn matplotlib researchpy plotly
```

## List of scripts in this repository:
```
├── 1_abstract_writing
│   ├── openai_key
│   ├── utils.py
│   └── write_abstract_openalex-kw_gpt4_multistep.py
├── 3_openalex_analysis
│   ├── build_db_contacts_multistep.py
│   ├── harvest_openalex_api.py
│   ├── power_analysis.py
│   ├── process_openalex_files.py
│   └── process_pickle_files.py
├── 5_qualtrics_analysis
│   ├── info-theory_hypotheses_testing.py
│   └── plot_qualtrics_results.py
├── LICENSE
├── README.md
└── envs
    ├── conda-nlpmodels.txt
    ├── conda-openai.txt
    └── conda-scifairness.txt
```

- The specific development environment used along the research can be replicated using the conda spec list files in the directory `envs`.

To create the environment from a spec list run:

`conda create --name myenv --file spec-file.txt`

## Help

For any issues or help, please check [previous issues](https://github.com/BiasExplained/gender-trial-paper/issues) or open a new one
