pip install -U transformers
pip install --upgrade nltk
pip install datasets -q
pip install -q transformers==4.57.1
pip install -q evaluate
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q wandb
pip install -q nltk
pip install -q PyPDF2
pip install -q tqdm numpy pandas
pip install rouge_score
pip install -q transformers datasets rouge_score
pip install -q "transformers==4.46.2" "datasets==3.0.2" "accelerate==1.2.1" \
               "rouge-score==0.1.2" "bert-score==0.3.13" "textstat==0.7.4" \
               "pandas==2.2.2" "evaluate==0.4.3" "tqdm>=4.67"
pip install -q peft==0.13.2
pip install pymupdf



git clone https://github.com/ereverter/bertsum-hf ./Extraction/bertsum-hf
