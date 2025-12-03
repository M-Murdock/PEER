## Getting Started
- Use `black_box.py` to train and run any of the MuJoCo environments.

- `dot_simulator_SA.py` creates a simple simulation environment in which you control a dot with arrow keys. This is a shared autonomy system with multiple trained policies.

    - `trained_policies/` contains all the trained policies used in the SA system
    - `training/` contains all the code for training the policies 

- `util.py` contains the utilities needed for `dot_simulator_SA.py`
    - `predictors.py`: Techniques for the inference step
    - `SA_types.py`: Enums of the options for SA inference, assistance, and arbitration steps
    - `selector.py`: Creates a gui which lets users select an item from a list
    - `shared_auto.py`: Assistance methods
 



## Resources 
Available robot models: https://github.com/google-deepmind/mujoco_menagerie/ 

Training agent: https://gymnasium.farama.org/introduction/train_agent/ 

mujoco tutorialhttps://github.com/tayalmanan28/MuJoCo-Tutorial/blob/main/tutorial/tutorial_2.ipynb 

https://www.gymlibrary.dev/environments/mujoco/ 

