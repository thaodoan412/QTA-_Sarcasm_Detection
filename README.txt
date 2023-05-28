1. Install Build Tools for Visual Studio 2022: https://visualstudio.microsoft.com/downloads/?q=build+tools
   -> In the installer select the workload Desktop development with C++
2. Install Python 3.11 from python.org
3. Clone the git repo
4. Open command line go to the code directory and install the requirements with: pip install -r requirements.txt
5. Enjoy the time while waiting for the dependencies to be installed and built
5. Run the code: python sarcasm_new.py
6. Enjoy the time while waiting for the computation to be finished
Attention: when you need to re-run the code, an OSError: Can't load tokenizer.... 
You need to delete the Cardiffnlp folder, which has been automatically created in your directory to be able to re-run the code.
Using Linux or having an NVIDIA GPU will shorten the running time significantly. 