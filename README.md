Follow the instructions https://pixi.sh/latest/ here

After installing pixi


Ensure you've got pixi set up. If running pixi doesn't show the help, see the getting started if it doesn't.

`pixi`


Initialize a new project and navigate to the project directory.


pixi init pixi-texnano
cd pixi-texnano

Add the dependencies you would like to use.


pixi add python
pixi add scikit-learn
pixi add matplotlib
pixi add optuna
pixi add wandb
pixi add plotly



Create a file named hello_world.py in the directory and paste the following code into the file.

hello_world.py

def hello():
    print("Hello World, to the new revolution in package management.")

if __name__ == "__main__":
    hello()
Run the code inside the environment.


pixi run python hello_world.py
You can also put this run command in a task.


pixi task add hello python hello_world.py
After adding the task, you can run the task using its name.


pixi run hello
Use the shell command to activate the environment and start a new shell in there.


pixi shell
python
exit
You've just learned the basic features of pixi:

initializing a project
adding a dependency.
adding a task, and executing it.
running a program.
Feel free to play around with what you just learned like adding more tasks, dependencies or code.

Happy coding!

Use pixi as a global installation tool#
Use pixi to install tools on your machine.

Some notable examples:


# Awesome cross shell prompt, huge tip when using pixi!
pixi global install starship

# Want to try a different shell?
pixi global install fish

# Install other prefix.dev tools
pixi global install rattler-build

# Install a linter you want to use in multiple projects.
pixi global install ruff