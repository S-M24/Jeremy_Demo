Dependencies
Python 3.x
Libraries: numpy, scipy, matplotlib, pyserial
Serial port access (specific hardware setup, if required)
Visual Studio Code
Installing
Clone the repository to your local machine or download the source code.
Ensure Python 3.x is installed on your system.
Install Visual Studio Code from Visual Studio Code Website.
Setting up the Python Environment in Visual Studio Code
Install Python Extension for Visual Studio Code:

Open Visual Studio Code.
Go to Extensions (Ctrl+Shift+X) and search for 'Python'.
Install the Python extension by Microsoft.
Creating a Virtual Environment:

Open a terminal in Visual Studio Code.
Navigate to the project directory.
Run python -m venv venv to create a virtual environment named 'venv'.
Activating the Virtual Environment:

Windows: Run .\venv\Scripts\activate.
macOS/Linux: Run source venv/bin/activate.
Installing Required Libraries:

With the virtual environment activated, install required libraries using pip:
pip install numpy scipy matplotlib pyserial

Executing the Program
Turn on mirrors and laser
Check device manager to ensure laser is at com13 still, otherwise change in AutoRun script by searching (Ctrl + f) 'COM13' and replacing the instance with the new com port.
With all three files in the same directory, first run SetUpMirrors.py
Now open the script AutoRun.py in Visual Studio Code.
Run the script using the play button in the top-right corner of the editor or by pressing F5.
