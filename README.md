# Example of distributed training with Neuro platform

## Description

This project is created from [Neuro Platform Project Template](https://github.com/neuromation/cookiecutter-neuro-project).
The main idea is to show how it's easy to train your models in a distributed way on Neuro platform.
  
## Development Environment

This project is designed to run on [Neuro Platform](https://neu.ro), so you can jump into problem-solving right away.

### Directory structure

| Local directory                      | Description       | Storage URI                                                                  | Environment mounting point |
|:------------------------------------ |:----------------- |:---------------------------------------------------------------------------- |:-------------------------- | 
| `data/`                              | Data              | `storage:ml-recipe-distributed-pytorch/data/`                              | `/ml-recipe-distributed-pytorch/data/` | 
| `modules/` | Python modules    | `storage:ml-recipe-distributed-pytorch/modules/` | `/ml-recipe-distributed-pytorch/modules/` |
| `configs/`                            | Configuration files | `storage:ml-recipe-distributed-pytorch/config/`                          | `/ml-recipe-distributed-pytorch/modules/` |
| `notebooks/`                         | Jupyter notebooks | `storage:ml-recipe-distributed-pytorch/notebooks/`                         | `/ml-recipe-distributed-pytorch/notebooks/` |
| `scripts/`                         | `Bash` scripts  | `storage:ml-recipe-distributed-pytorch/scripts/`                         | `/ml-recipe-distributed-pytorch/scripts/` |
| `results/`                         | Logs and results  | `storage:ml-recipe-distributed-pytorch/results/`                           | `/ml-recipe-distributed-pytorch/results/` |

## Development

Follow the instructions below to set up the environment and start a distributed training.

### Setup development environment

```bash
neuro-flow build myimage
neuro-flow mkvolumes
neuro-flow upload ALL
```

* Several files from the local project are uploaded to the platform storage (namely, `requirements.txt`, 
  `apt.txt`, `setup.cfg`).
* A new job is started in our [base environment](https://hub.docker.com/r/neuromation/base). 
* Pip requirements from `requirements.txt` and apt applications from `apt.txt` are installed in this environment.
* The updated environment is saved under a new project-dependent name and is used further on.
* Project folders are created on storage and local files are uploaded there.

### Run distributed training on Neuro platform

To try to fine-tune your own Bert model on the platform you just need to run script 
`modules/scripts/run_distributed_on_platform.sh`. You don't need to download any data 
to do it, because we provide a "dummy" dataset which allows running the project without
 any preparations except login on platform.  
 
In case, you want to try to train your own model with real data, just download 
[TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) 
contest dataset.  

### Run Jupyter with GPU

`neuro-flow run jupyter`

* The content of `modules` and `notebooks` directories is uploaded to the platform storage.
* A job with Jupyter is started, and its web interface is opened in the local web browser window.

### Kill Jupyter

`neuro-flow kill jupyter`

* The job with Jupyter Notebooks is terminated. The notebooks are saved on the platform storage. You may run 
  `neuro-flow download notebooks` to download them to the local `notebooks/` directory.

## Data

### Uploading to the Storage via Web UI

On local machine, run `neuro-flow run filebrowser` and open job's URL on your mobile device or desktop.
Through a simple file explorer interface, you can upload test images and perform file operations.

### Uploading to the Storage via CLI

On local machine, run `neuro-flow upload data`. This command pushes local files stored in `./data`
into `storage:ml-recipe-distributed-pytorch/data` mounted to your development environment's `/project/data`.


## Run development job

If you want to debug your code on GPU, you can run a job via `neuro-flow run remote_debug`, it will open a terminal (type `exit` or `^D` to close it), see its logs via `neuro-flow logs remote_debug`, or use it for remote debugging, since port 2022 is open for SSH.

Please don't forget to kill your job via `neuro-flow kill remote_debug` not to waste your quota!