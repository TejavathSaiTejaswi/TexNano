Follow the instructions https://pixi.sh/latest/ here

After installing pixi


Ensure you've got pixi set up. If running pixi doesn't show the help, see the getting started if it doesn't.

`pixi`


Initialize a new project and navigate to the project directory.


`pixi init pixi-texnano`
`cd pixi-texnano`

Add the dependencies you would like to use.


`pixi add python`
`pixi add scikit-learn`
`pixi add matplotlib`
`pixi add optuna`
`pixi add wandb`
`pixi add plotly`

Create a file named hello_world.py in the directory and paste the following code into the file.

Get the code from https://gist.github.com/abhijitramesh/ce35b80197f460754606d30340e3ce0e   
Place the code in the folder and run the code inside the environment.

To set the description of the project
`pixi project description set "MLOPS using wandb and optuna"`

To add platforms to the project
`pixi project platform add osx-64 linux-64 win-64 osx-arm64`

To run the file
`pixi run python mlops_task.py`

You can also put this run command in a task.
`pixi task add mlops_task python mlops_task.py`

After adding the task, you can run the task using its name.
`pixi run mlops_task`


