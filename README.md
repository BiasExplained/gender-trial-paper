# gender-trial-paper
Repository with custom code used in the manuscript: Large-scale randomized trials show no evidence of gender bias in evaluating scientific abstracts

A static version of the data used for the pilot and main results analysis with the experiment setup can be downloaded at:  

[doi/10.5281/zenodo.10728225](https://zenodo.org/doi/10.5281/zenodo.10728225)


**Before running each file, please adjust the `<path>` variable according to local directory structure.**  
This can be easily done by creating a symbolic link to your data folder using the shell command `ln -s /path/to/source data`. This will create a folder called `data` in the current directory point to `/path/to/source`.

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
│   ├── __pycache__
│   │   └── preprocessing_results.cpython-311.pyc
│   ├── info-theory_hypotheses_testing_main.py
│   ├── odds-ratio.py
│   ├── odds_ci_plots.txt
│   ├── odds_ttest.csv
│   ├── pilot_analysis
│   │   ├── info-theory_hypotheses_testing.py
│   │   └── plot_qualtrics_results.py
│   ├── plot_likert_scales.py
│   ├── plot_qualtrics_results_main.py
│   └── preprocessing_results.py
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

All analysis were done using the `Pulsar` editor with the `hydrogen` extension, allowing interactive coding without the versioning issues caused by `jupyter notebooks`.  

## Help

For any issues or help, please check [previous issues](https://github.com/BiasExplained/gender-trial-paper/issues) or open a new one
