## Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning

## Requirements
pytorch==1.6.0, mujoco-py==2.0.2.13.
All the requirements are specified in requirements.txt.

## Code Usage
We demonstrate with Half-Cheetah-Vel environment. For other environments, change the argument `--env-type` according to the table:

Environment  | Argument
------------- | -------------
Point-Robot  | point_robot_v1
Half-Cheetah-Vel  | cheetah_vel
Ant-Dir | ant_dir
Hopper-Param | hopper_param
Walker-Param | walker_param

#### Data Collection
Copy the following code into a shell script, and run the script.
```
for seed in {1..40}
do
	python train_data_collection.py --env-type cheetah_vel --save-models 1 --log-tensorboard 1 --seed $seed
done
```

#### Train the Task Encoder
If use generative modeling, run `python train_generative_model.py --env-type cheetah_vel` to pre-train the CVAE.
Run `python train_contrastive.py --env-type cheetah_vel  --relabel-type generative --generative-model-path logs/***  --output-file-prefix contrastive_generative` to train the encoder. Specify `--generative-model-path` with the path of the last saved CVAE model.
If use reward randomization, specify `--relabel-type ` with `reward_randomize`.

#### Offline Meta-RL
Specify `--encoder-model-path` with the last saved encoder, then run:
`python train_offpolicy_with_trained_encoder.py --env-type cheetah_vel  --encoder-model-path logs/*** --output-file-prefix offpolicy_contrastive_generative`.
Check for the training result using Tensorboard.

#### OOD Test
Replace the content in the file `ood_test_config/cheetah_vel.txt` with paths of sampled behavior policies. Modify line 343~346 of `test_ood_context.py` to set the correct test model path. Then run `python test_ood_context.py --env-type cheetah_vel`.


## Citation
If you are using the codes, please cite our paper.

	@inproceedings{yuan2022robust,
        	title={Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning},
  			author={Yuan, Haoqi and Lu, Zongqing},
  			booktitle={International Conference on Machine Learning},
  			pages={25747--25759},
  			year={2022},
  			organization={PMLR}
	}
