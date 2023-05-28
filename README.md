Taking into account the earlier suggestion, the team also tried to detect sarcasm in the current data set. To the best of our knowledge, Python ecosystems have more developed sarcasm detection tools than R. Thus, in this phase, Python will be primarily used to calibrate our method. After pre-processing the data in R, we export it and use that data set just for sarcasm detection in Python to assure the consistency of the input. The paper from Abaskohi(Abaskohi, Rasouli, Zeraati, & Bahrak, 2022) is our main source of inspiration. Given the time limitation for the preliminary research as well as implementation, the team decided to choose RoBERTa with twitter-roberta-base, which has been trained on close to 58 million tweets and calibrated for sentiment analysis (Barbieri, Camacho-Collados, Neves, & Anke, 2020). RoBERTa stands for Robustly Optimized BERT Pre-training Approach which essentially have a similar structure like BERT with a few minor changes in design and training methodology to enhance the performance (Barbieri et al., 2020). We use iSarcasm (Oprea & Magdy, 2020) as the train data set in our model.The iSarcasm data set includes 4484 tweets: 3707 non-sarcastic and 777 sarcastic. However, after pre-processing, we are left with 3468 tweets: 2401 non-sarcastic and 867 sarcastic in the train data set. Due to the imbalanced data set, we only achieved the F1-score of 0.389.

# Replication instruction:
1. Install Build Tools for Visual Studio 2022: https://visualstudio.microsoft.com/downloads/?q=build+tools
   -> In the installer select the workload Desktop development with C++
2. Install Python 3.11 from python.org
3. Clone the git repo
4. Open command line go to the code directory and install the requirements with: pip install -r requirements.txt
5. Enjoy the time while waiting for the dependencies to be installed and built
5. Run the code: python sarcasm_new.py
6. Enjoy the time while waiting for the computation to be finished
### Attention: when you need to re-run the code, an OSError: Can't load tokenizer.... 
### You need to delete the Cardiffnlp folder, which has been automatically created in your directory to be able to re-run the code.
### Using Linux or having an NVIDIA GPU will shorten the running time significantly. 
